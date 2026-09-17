# geval · 大模型通用能力评测框架

对应你项目里的 **阶段 1：对盘古7B / Qwen8B / Qwen3-32B 做通用能力评测**。

目标不是"跑几个榜"，而是回答一个技术选型问题：

> 用「小模型 + 微调」替代 Qwen3-32B，在**通用能力可接受的退化范围内**，能换来多少准确率与时延收益？

所以这套框架除了准确率，还强制输出**置信区间、配对显著性检验、时延分布**和**解析失败率**——
没有这些，68.2% 与 65.7% 的差距你无法判断是真差距还是抽样噪声。

> **⚠️ 关于「是不是主流方法」与「要不要用开源框架」，请先读 [`docs/EVAL_METHODOLOGY.md`](docs/EVAL_METHODOLOGY.md)。**
> 一句话结论：geval 使用的**基准与指标是主流的**（与 OpenCompass / EvalScope / lm-eval 同批题目、
> 同套口径，含 lm-eval 的 `acc` / `acc_norm`），但**框架是自建的，没有第三方验证，
> 绝对值不可与公开榜单直接对齐**。
>
> 既然依赖可以从外网导入内网，**推荐混合**：阶段 1 的正式数字用 **lm-eval** 出（离线部署步骤见
> [`docs/OFFLINE_DEPLOY.md`](docs/OFFLINE_DEPLOY.md)，`lm-eval` 基础依赖里不含 torch，完全可行），
> geval 负责 prompt 协议消融、微调前后配对检验、以及后续的专业能力测评。
> 各管一段的理由见方法论文档 §6。

---

## 1. 快速开始

> **要在内网跑？直接看 [`docs/INTRANET_RUNBOOK.md`](docs/INTRANET_RUNBOOK.md)**（行内运行手册，
> 含部署、五步执行、故障排查速查表）。下面是在开发机上快速上手的版本。

```bash
# 离线自检：不需要模型、不联网，验证整条链路（约 1 秒）
python -m geval selftest

# 查看内置任务
python -m geval list-tasks

# 先看实际发给模型的 prompt，确认格式再跑全量
python -m geval preview --task ceval
python -m geval preview --task json_schema     # 与意图识别最贴近的那个

# 干跑：mock 模型 + 合成数据，验证配置与报告
python -m geval run --config config.smoke.json --smoke

# 真实评测（改好 config.example.json 里的模型地址后）
cp config.example.json config.json
python -m geval probe  --config config.json            # 端点探活
# 阶段 1：只跑微调前已有的 3 个模型
python -m geval run    --config config.json --smoke --limit 8 \
       --models pangu7b-pre,qwen8b-pre,qwen3-32b       # 先验证接线
python -m geval run    --config config.json --limit 40 \
       --models pangu7b-pre,qwen8b-pre,qwen3-32b       # 先看排序
python -m geval run    --config config.json \
       --models pangu7b-pre,qwen8b-pre,qwen3-32b --resume   # 阶段 1 正式跑

# 阶段 1.5：微调完成后跑全量 5 个对象，报告自动产出 pre→post 退化对比
python -m geval run    --config config.json --resume
python -m geval report --run runs/geval-general-5objects

# 报告（也可在评价微调后模型时单独重出）
python -m geval report --run runs/geval-general-5objects
```

或者直接跑完整脚本：

```bash
bash scripts/run_stage1.sh
```

**运行环境要求：Python 3.9+，零第三方依赖。** 内网 ARM64 离线可用，拷目录即可：

```bash
python scripts/verify_deps.py     # 输出「第三方依赖：无」
```

---

## 2. 评测维度与任务映射

| 能力维度 | 为什么与意图识别相关 | 对应任务 |
| :--- | :--- | :--- |
| 中文语言理解 | 中文 query 的语义边界判断 | `ceval` `cmmlu` |
| 通用知识 | 领域词、实体、专有名词识别 | `mmlu` `mmlu_pro` `arc` |
| 逻辑与数学推理 | 多意图、条件意图、否定意图 | `bbh` `gsm8k` `math` |
| **指令遵循** | 格式约束守不守得住、会不会答非所问 | `ifeval` |
| **结构化输出** | 直接产出 intent + slots JSON | `json_schema` |
| 长上下文与鲁棒性 | 多轮对话下的指代消解 | 见 §8 扩展建议 |

加粗的两项是**性价比最高的两项**：生产环境里意图识别的失败绝大多数不是"不懂知识"，
而是**输出不可解析**和**边界判断漂移**。所以框架内建了 `json_schema` 和 `ifeval` 两个专项，
并把结构化输出拆成四层可诊断的子指标。

### `json_schema` 专项的四层指标

| 指标 | 含义 | 出问题时该改什么 |
| :--- | :--- | :--- |
| `json_valid_rate` | 输出能否被解析成 JSON | prompt 约束、是否开了 json mode |
| `schema_ok_rate` | 是否符合 schema（必填字段、类型、枚举） | schema 描述、枚举是否列举清楚 |
| `intent_accuracy` | 意图分类是否正确 | 模型能力 / 训练数据 |
| `slots_exact_rate` | 槽位是否完全一致 | 标注一致性 / 槽位定义 |
| `extra_prose_rate` | 是否夹带解释性文字（严格模式下判失败） | prompt 里是否明确"只输出 JSON" |

日常经验：小模型掉分主要掉在前两层，不在第三层。如果看到 `json_valid_rate` 只有 60%，
先改 prompt 和 schema，不要急着上微调。

---

## 3. 五个关键设计决策

### 3.1 同一套代码跑所有模型，prompt 协议作为实验变量

`protocol` 可切换，且同一任务可用 `tag` 起别名并行对比：

- `direct`：只回答选项字母 → 小模型最稳、时延最低
- `answer_line`：允许推理，最后一行以「答案：X」收口 → 大模型通常最优
- `fewshot`：按基准原始格式给 k 个示例 → 与公开榜单数字最可比

实践中常见结论是 **7B/8B 用 `direct` 反而比 `answer_line` 高几个点**（推理过程把结论带偏），
而 32B 用 `answer_line` 更优。这本身就是阶段 1 该产出的结论之一。对比写法：

```json
{ "name": "ceval", "tag": "ceval-direct",  "protocol": "direct" }
{ "name": "ceval", "tag": "ceval-fewshot", "protocol": "fewshot", "num_fewshot": 5 }
{ "name": "ceval", "tag": "ceval-logprob", "protocol": "fewshot", "mode": "logprob" }
```

### 3.2 单选题支持三种打分路径

| `mode` | 机制 | 请求数/样本 | 用途 |
| :--- | :--- | :--- | :--- |
| `gen`（默认） | 生成文本后抽取选项字母 | 1 | 日常全量评测 |
| `logprob` | 读首生成位置的 `top_logprobs` 取 argmax | 1 | 降低小模型「输出跑偏字母」的噪声 |
| `loglikelihood` | `/completions` + `echo` 取续写 token 级 logprob，算 **acc / acc_norm** | N 选项 | **与 lm-eval / 公开榜单口径对齐** |

`loglikelihood` 是 lm-eval 在选择题上的主指标实现：

- `acc` = 续写 logprob 之和最大者
- `acc_norm` = **长度归一化**后的最大值（默认主指标，可用 `"primary": "acc"` 切换）
- 长度归一化不是细节：否则「续写越长 logprob 越低」会让短选项系统性占优
- 逐样本同时记录 `acc_choice` / `acc_norm_choice` / `acc_norm_margin`，
  两者不一致的样本是「格式偏好」而非「知识差异」的候选，值得单独看

建议：**选择题抽样跑 `loglikelihood` 出对标数字，全量仍用 `gen`**（4 选项 = 4 倍请求量）。
任一模式在服务端能力不足（不支持 `logprobs` / `echo`）时都会**自动降级**为生成式，不会让整任务失败。

配置写法：

```json
{ "name": "ceval", "tag": "ceval-likelihood", "mode": "loglikelihood", "primary": "acc_norm", "num_fewshot": 5 }
```

### 3.3 解析失败率被单独统计并上报

每次抽取都记录 `strategy`（命中了哪条规则）与 `ambiguous` 标记，报告里有专门的"解析质量"一节。

> 解析失败率 > 2% 时，先改 prompt 协议或补抽取规则，再谈模型能力。

这样把"模型答错"和"抽取器没读懂"彻底分开——混在一起会直接导致误判。

### 3.4 统计量是硬性输出，不是可选装饰

- **Wilson 区间** + **bootstrap 95% CI**：准确率的误差范围
- **McNemar 精确检验** + **配对 bootstrap**：同一样本上两个模型的差距是否显著
- **时延 p50/p90/p95/p99**、tokens/s
- **帕累托图**：准确率 vs 时延，定位性价比拐点

报告里只有「显著」级别的差距才建议作为选型依据。

### 3.5 复现指纹

每次运行都写 `manifest.json`：框架版本、Python、平台、并发数、seed、生效配置、
数据集文件 sha256、few-shot 数量与协议。换数据或改 prompt 后数字不可比时，能立刻发现。

---

## 4. 报告解读

`runs/<run_name>/report.html`（自包含单文件，内联 SVG，无外部引用，可直接发群）
与 `report.md`。

| 章节 | 看什么 |
| :--- | :--- |
| 1 准确率矩阵 | 全貌，带 ±CI。**不要只看小数点** |
| 2 综合与时延 | 帕累托图：越靠左上越优 |
| 3 配对显著性 | **决定能不能下结论**。CI 跨 0 或 p ≥ 0.05 的差距不算数 |
| 4 解析质量 | 解析失败率、主要抽取策略。分低是抽取器的锅还是模型的锅 |
| 5 结构化输出专项 | 意图识别的四层指标 |
| 6 指令遵循专项 | 样本级 / 指令级通过率，以及未覆盖的指令类型 |
| 8 分组明细 | 按 subject 定位短板 |
| 9 错例抽样 | 人工快速复核 |
| 10 复现信息 | 数据集指纹与运行期错误 |

---

## 5. 配置说明

详见 `config.example.json`。要点：

| 字段 | 说明 |
| :--- | :--- |
| `proxy` | **默认 `bypass`**。服务器上常配 `HTTP_PROXY` 指向外网代理，urllib 会默认读取，导致内网模型请求被发给外网代理，表现为**超时或 502**。`bypass` 显式清空代理；`env` 跟随环境变量；也可填 `http://ip:port` |
| `concurrency` | 并发请求数。vLLM/SGLang 部署下建议 8~32，先用 `probe` 确认单请求时延再定 |
| `extra_body` | 原样透传的额外请求体。Qwen3 系列关闭思考模式：`{"chat_template_kwargs": {"enable_thinking": false}}` |
| `supports_logprobs` | 服务端是否支持 `logprobs`。不确定时设为 `false`，避免无效请求 |
| `cache_dir` | 磁盘缓存，仅缓存 `temperature == 0` 的确定性请求，避免缓存污染 |
| `tasks[].limit` | 分层抽样上限。`0` = 全量 |
| `tasks[].tag` | 同一任务起别名，用于并行对比不同 prompt 协议 |
| `report.baseline` | 配对检验的基线模型 id，一般填现网的 Qwen3-32B |
| `report.pairs` | 指定配对，用于微调前 / 后对比 |

---

## 6. 数据集

见 `datasets/README.md`。核心：

- 公开基准在**联网机器**上执行 `python scripts/prepare_datasets.py` 转成 JSONL，再把目录拷进内网；
- 自己的业务数据直接导出 JSONL，`json_schema` 任务配套放一个 `schema.json` 声明意图与槽位枚举；
- **阶段 0 必须先冻结 test 集**并记录 sha256，任何进入训练的数据都不能出现在 test 里。

---

## 7. 通用能力评测的 5 个对象：阶段 1 + 阶段 1.5

**通用能力评测的被测对象是 5 个**：盘古7B（微调前/后）、Qwen8B（微调前/后）、Qwen3-32B。
分两个时点执行，最后合并出「微调影响评估」：

| 时点 | 执行 | 说明 |
| :--- | :--- | :--- |
| 阶段 1（微调前） | `--models pangu7b-pre,qwen8b-pre,qwen3-32b` | 建立基线，同时回答「小模型该用哪种 prompt 协议」 |
| 阶段 1.5（微调后） | 直接跑全量（配置里 5 个模型都已定义） | 补上两个 post 模型，报告自动产出 pre→post 对比 |

配置见 `config.example.json`（5 个模型已一次性定义好）。阶段 1 时后两个模型还不存在，
用 `--models` 过滤即可；不指定就是 5 个全跑。

### 为什么这一步必须做

小模型 SFT 之后最常见的副作用是**灾难性遗忘**。如果通用能力只测微调前一次，
你无法判断「意图识别涨的 3 个点」是不是以通用能力崩了 5 个点为代价换来的。

报告第 4 节「微调影响评估」会逐任务给出 pre→post 差值与判定：

```
| 任务        | 微调前 | 微调后 | 差值      | 差值 95% CI      | McNemar p | 判定     |
| ceval-likelihood | 86.7% | 40.0% | -46.67 pt | [-73.33, -20.00] | 0.0156    | 显著退化 |
```

判定规则：只有 **p < 0.05** 才敢说「显著退化 / 显著提升」，否则一律记为「不显著（视为持平）」。
差值符号约定为 **post - pre**，正数 = 微调后更好。

### 配对对比的两个硬前提

配对检验比单独跑严格得多。不满足这两条，差值就不可信：

1. **同一批样本、同一套 prompt。** 报告会自动检查并在表内出告警：
   样本覆盖率不足 90%、或两模型的 `protocol` / `mode` / `num_fewshot` 不一致时，
   该行结论标记为不可用。
2. **微调没有改动 chat template。** SFT 常见的连带改动就是模板（尤其从 base 微调、
   或为塞入 system prompt 调了模板）。一旦改动，同一段 prompt 在两个模型上被拼成
   **不同的实际输入**，差值里混入格式差异，方向还不确定。

   处理方式按优先级：① 微调时不改模板（推荐）→ ② 必须改就把微调前的模型也用新模板重测 →
   ③ 至少在结论中显式标注该混淆因素。
   验证方法：对两个端点各发一条相同请求，比对服务端日志里的最终拼串。

### 建议的验收口径

提前定死，避免事后找理由。例如：

> 专业任务准确率提升 ≥ 显著，同时 `ceval`/`ifeval` 退化 ≤ 2 个点且**不显著**、
> `json_valid_rate` 不下降 → 可替代。

```bash
# 阶段 1.5
python -m geval run   --config config.json --resume
python -m geval report --run runs/geval-general-5objects
```

---

## 8. 已知边界与扩展建议

诚实说明本框架**没有**覆盖的部分：

| 项 | 状态 | 建议 |
| :--- | :--- | :--- |
| IFEval 指令类型 | 覆盖 24 类常用校验，`language:response_language` 需语言识别库，未实现 | 未实现类型会**从统计中排除并单列**，不会伪装成"不通过"。扩展点：`geval/ifeval.py::CHECKERS` |
| IFEval 宽松模式 | 仅实现严格模式 | 若要与官方 loose 口径对齐，需在 `check_response` 前做首尾行处理 |
| 长上下文 / 多轮 | 未实现 | 若多轮 history 较长，建议补 RULER 或 Needle-in-Haystack，并加一个"多轮指代"自定义任务 |
| 人工/裁判模型评测 | 未实现 | 开放式生成可加 LLM-as-judge，用 Qwen3-32B 当裁判，但需注意位置偏差与自我偏好 |
| 安全与拒答 | 未实现 | 若有合规要求需单列 |

新增一个基准只需要在 `geval/tasks.py` 里加一个子类并注册进 `TASK_REGISTRY`。

---

## 9. 目录结构

```
llm-general-eval/
  config.example.json      真实模型配置模板
  config.smoke.json        干跑配置（mock 模型，不连真实服务）
  requirements.txt         无第三方依赖（说明性文件）
  geval/
    __main__.py            CLI：doctor / run / report / selftest / count / preview / probe
    client.py              OpenAI 兼容客户端（并发安全、重试、磁盘缓存、logprobs）
    http.py                urllib 传输层（代理策略、退避重试）
    datasets.py            多格式归一化、去重、分层抽样、合成数据
    prompts.py             三种 prompt 协议 + 各任务模板
    answers.py             答案抽取（选项 / 数值 / JSON）+ 思维链剥离
    metrics.py             准确率、Wilson / bootstrap 区间、McNemar、时延分位
    tasks.py               任务注册表（MCQ / 数值 / 短答案 / JSON / IFEval）
    ifeval.py              24 类规则化指令校验器
    runner.py              编排：并发执行、断点续跑、落盘、指纹
    report.py              Markdown + 自包含 HTML（内联 SVG 图表）
    mock.py                确定性 mock 模型（离线自检用）
  datasets/
    README.md              数据格式与准备方式
    json_schema/           意图 + 槽位示例数据集（含 schema.json）
  docs/
    INTRANET_RUNBOOK.md    ★ 行内运行手册（部署、执行、故障排查，实操优先）
    COVERAGE.md            五项需求的覆盖度对照（支持什么、不支持什么、缺口在哪）
    EVAL_METHODOLOGY.md    与开源框架的关系、可比性边界、专业能力测评设计
    OFFLINE_DEPLOY.md      把 lm-eval / EvalScope 离线装进内网 ARM64 的实操指南
  scripts/
    prepare_datasets.py    公开基准 -> JSONL（联网机器执行）
    download_wheels.sh     为 aarch64 / py3.11 下载开源框架的离线 wheel（联网机器）
    install_offline.sh     从离线 wheel 目录安装（内网机器）
    pack_bundle.py         打成单个交付包 + 生成 sha256 清单（--code-only 出增量包）
    verify_install.py      ★ 核对内网代码与最新版本是否一致（内网机器，见手册 §2.4）
    verify_deps.py         依赖核查（确认 geval 运行时零第三方依赖）
    run_stage1.sh          阶段 1 推荐执行顺序
  dist/                    交付包与校验值（拷进内网用，不入库）
    llm-general-eval-*.tar.gz   全量包（含预下载 wheel，约 133 MB）
    geval-src-*.tar.gz          源码增量包（约 150 KB，更新内网用，排除 datasets/ 与 wheels/）
    CODE_MANIFEST.txt           源码 sha256 清单，随增量包分发
  wheels/                  预下载的离线 wheel（bootstrap + lm-eval，不入库）
  runs/<run_name>/         运行产物：manifest / 明细 JSONL / summary / report
```

---

## 10. 常见问题

**Q：`probe` 报了 `HTTP 0 URLError` 或超时，但 `curl` 是通的？**
检查 `proxy` 是否为 `bypass`。这是内网部署最高频的坑：`HTTP_PROXY` 把内网请求发给了外网代理。

**Q：`mode: "logprob"` 下分数和 `gen` 不一样？**
正常现象。似然打分读的是首 token 分布，不受"模型后半段话把结论带偏"的影响。两者都值得看，
差异大的任务说明抽取噪声高。

**Q：`json_schema` 的 `extra_prose_rate` 很高？**
确认 prompt 里是否有"只输出一个 JSON 对象，不要输出解释、不要使用 markdown 代码块"，
以及服务端是否支持严格的 json 输出模式。`strict_json: false` 可以放宽为只判语义正确。

**Q：跑全量太贵/太慢？**
先用 `--limit 40` 分层抽样看排序，排序稳定后再全量。抽样运行的报告会自动标注不可与榜单直接对齐。

**Q：中断了怎么办？**
所有结果逐条落盘。加 `--resume` 续跑，会自动跳过已完成的样本。
