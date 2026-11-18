# 交接文档 · geval 大模型通用能力评测框架

> 版本：geval 0.1.0 ｜ 交接基准日期：2026-11-10 ｜ 代码位置：本仓库根目录
> 适用对象：接手本项目的工程师，以及该工程师使用的 AI Agent。
>
> **一句话定位**：geval 是一个**零第三方依赖**的 Python 评测框架，用来回答
> 「小模型（盘古7B / Qwen8B）+ 微调，能否在通用能力可接受退化内替代 Qwen3-32B」。
> 它不只出准确率，还强制输出**置信区间、配对显著性检验、时延分布、解析失败率**。

---

## 0. 三分钟速览（如果只读一段）

| 项目 | 现状 |
| :--- | :--- |
| 代码 | 完整可运行。`python -m geval selftest` 全 PASS（7 任务 + 并行 + 合并，约 1 秒） |
| 依赖 | 运行时**零第三方依赖**（Python 3.9+，本机实测 3.13.12） |
| 数据 | 6 个数据集已就绪（ceval / cmmlu / gsm8k / bbh / ifeval / json_schema），去重后 19,889 条 |
| 配置 | `config.json` 已指向真实通道 `llm-chat-core`（单模型、6 任务），**含密钥，不入库** |
| 真实评测 | **尚未执行**。`runs/` 里只有 `smoke` / `smoke_verify` / `smoke_verify2`，全部是 mock 模型产物 |
| 报告 | 自包含 `report.html` + `report.md`，最多 11 个章节（按运行内容增减，编号动态） |
| 主要缺口 | 专业能力指标（混淆矩阵 / 槽位 F1 / 幻觉槽位 / 拒识率）未实现；lm-eval 锚点未部署 |
| 最大风险 | 单模型跑出来的数字**不能用于选型结论**；真实端点未探活 |

**接手后的第一件事**：在项目根目录执行 `python -m geval doctor --config config.json`，
确认无阻塞项后再谈跑数。

---

## 1. 项目背景：要回答什么问题

### 1.1 业务背景

上游是银行客服的**意图识别**项目：用户一句话进来，要判定意图（查询余额 / 转账 / 挂失……）
并抽取槽位（账户、金额、卡号、日期范围、期限）。现网用 Qwen3-32B，成本与时延都高。

技术路线是「**小模型 + 全参数微调**」替代 32B，但必须证明两件事：

1. 专业任务（意图识别）能力追得上或接近 32B；
2. 通用能力没有因为微调而崩掉（灾难性遗忘）。

geval 负责第 2 件事的**通用能力侧**，并为第 1 件事预留了扩展点。

### 1.2 为什么不是「跑几个榜就完事」

68.2% 与 65.7% 的差距，可能是真实差距，也可能只是抽样噪声。没有置信区间和配对检验，
就无法判断微调后掉的 2 个点是不是灾难性遗忘。所以框架把统计量做成**硬性输出**。

### 1.3 两种被测对象口径（极易混淆，务必分清）

| 口径 | 配置文件 | 被测对象 | 用途 |
| :--- | :--- | :--- | :--- |
| **当前实际口径** | `config.json` | 单模型 `llm-chat-core`，6 个任务 | 验证工具链 / 现网主通道摸底 |
| **5 对象模板（可选）** | `config.example.json` → 另存 `config.5objects.json` | 盘古7B pre/post、Qwen8B pre/post、Qwen3-32B | 阶段 1 基线 + 阶段 1.5 微调前后对比 |

- 单模型口径下，报告会自动跳过「配对显著性」和「微调影响评估」，并写明原因。
- 5 对象口径下，Qwen3-32B 走 `reqbody_gateway`（鉴权放请求体内），其余走标准 OpenAI 兼容。
- **不要把两套口径的结果混在同一个 run 目录里**。样本集与模型集合都不同，`--resume` 会按 id 跳过，
  混跑会造成样本不一致。

---

## 2. 现状快照（2026-11-10 实测核验）

### 2.1 代码与版本

| 项 | 值 |
| :--- | :--- |
| 版本 | `geval 0.1.0`（`geval/__init__.py`） |
| 规模 | `geval/` 16 个模块约 7,450 行；`scripts/` 6 个 Python 脚本约 1,260 行（另有 3 个 shell 脚本） |
| 运行时依赖 | 无。只用标准库（urllib / concurrent.futures / json / math …） |
| 离线自检 | 通过。覆盖 7 任务、抽取层、acc/acc_norm、退化判定、续跑、报告、任务并行、merge-runs |
| Git | 4 个提交，HEAD `9053ef3`（2026-10-12）。工作区有未提交改动（文档对齐 + 报告权重 + merge 补写 config + 配置快照打码） |

### 2.2 数据集（已备齐 6/6）

| 任务 | 原始行数 | 去重后（`geval count`） | 说明 |
| :--- | ---: | ---: | :--- |
| ceval | 1,346 | 1,337 | 配置写 `split: val`，实际落到 `test.jsonl`（无 val.jsonl） |
| cmmlu | 11,582 | 11,408 | 最大的一项，占全量 57% |
| gsm8k | 1,319 | 1,319 | 未发现重复题面，去重前后一致 |
| bbh | 6,511 | 5,268 | 重复较多，去重差 1,243 |
| ifeval | 541 | 541 | |
| json_schema | 16 | 16 | 仅够验证链路，**不构成评测** |
| **合计** | **21,315** | **19,889** | 单模型全量 ≈ 19,889 次请求 |

> `doctor` 报原始条数，`count` 报去重后条数，两者不一致是**正常**的。
> 评测与进度条都以 `count` 为准。

### 2.3 运行产物

`runs/` 下只有三个目录：`smoke`、`smoke_verify`、`smoke_verify2`，均由 mock 模型 + 合成数据生成。
**没有任何真实模型的评测结果。** 也就是说：报告链路验证过，但真实数字一个都没有。

### 2.4 环境提醒（本机）

- 当前 shell 存在 `HTTP_PROXY` / `HTTPS_PROXY`，`doctor` 会告警。geval 的 `proxy: "bypass"`
  会显式清空代理，但仍建议 `unset` 后再跑，尤其是端点在内网时。
- 项目内未建 venv；若要跑 lm-eval，需单独建（geval 不需要）。
- `config.json` 含真实网关地址与 `api_key`，已被 `.gitignore` 排除；**打包、外发、贴群前先检查**。

---

## 3. 已有特性清单

### 3.1 CLI：9 个子命令

| 子命令 | 作用 | 是否发网络请求 |
| :--- | :--- | :--- |
| `selftest` | 离线端到端自检（mock 模型 + 合成数据） | 否 |
| `list-tasks` | 列出 12 个内置任务、默认 few-shot、切分 | 否 |
| `preview --task X` | 打印实际下发给模型的 prompt | 否 |
| `doctor` | 就绪检查：环境 / 依赖 / 配置 / 数据 / 端点 / lm-eval | 仅加 `--probe` 时 |
| `probe` | 端点探活，打印时延与服务端模型名 | 是 |
| `count` | 推算样本量与请求量、估算耗时 | 否 |
| `run` | 执行评测（支持冒烟、抽样、续跑、补跑、任务并行） | 是 |
| `report` | 用已有结果重出报告，可改基线 / 配对 / 权重 | 否 |
| `merge-runs` | 合并多个分批 run 目录，重建总表并出报告 | 否 |

### 3.2 评测能力

- **12 个内置任务**：ceval、cmmlu、mmlu、mmlu_pro、arc、agieval、gsm8k、math、bbh、ifeval、
  json_schema、intent_json。新增任务只需在 `geval/tasks.py::TASK_REGISTRY` 注册子类。
- **3 种 prompt 协议**（`protocol`）：`direct` / `answer_line` / `fewshot`。
  同一任务可用 `tag` 起别名并行对比，例如 `ceval-fewshot` 与 `ceval-direct`。
- **3 种打分模式**（`mode`）：
  | mode | 机制 | 请求/样本 | 说明 |
  | :--- | :--- | ---: | :--- |
  | `gen`（默认） | 生成文本后抽取字母 / 数值 / JSON | 1 | 全量用这个 |
  | `logprob` | 读首 token `top_logprobs` 取 argmax | 1 | 降抽取噪声 |
  | `loglikelihood` | `/completions` + `echo`，算 `acc` / `acc_norm` | = 选项数 | 与 lm-eval 口径对齐，建议 `--limit 200` 抽样 |
  服务端不支持 `logprobs` / `echo` 时**自动降级**为生成式，不会让整任务失败。
- **结构化输出四层指标**：`json_valid_rate`（可解析）、`schema_ok_rate`（符合 schema）、
  `intent_accuracy`（意图正确）、`slots_exact_rate`（槽位完全一致），另有 `extra_prose_rate`（夹带散文）。
- **IFEval**：24 类规则化校验，未实现的指令类型会从统计中排除并单列，不伪装成失败。
- **数据层**：多格式归一化（题面/选项/答案的多种写法）、题面去重、按 subject 分层抽样（固定 seed）、
  缺 dev 时用测试集前 k 条做 few-shot 并自动从评测集剔除（防泄漏）。

### 3.3 统计与报告

- Wilson 区间、bootstrap 95% CI、配对 bootstrap 差值 CI、McNemar 精确检验。
- 时延 p50 / p90 / p95 / p99、tokens/s、token 用量。
- 自包含 HTML（内联 SVG，无外部引用，可直接发群）+ Markdown。
- **最多 11 个章节**（编号动态，按标题定位）：准确率矩阵、综合与时延、配对显著性、微调影响评估、
  解析质量、续写似然对照、结构化输出、指令遵循、分组明细、错例抽样、复现信息。
- **综合分权重**，优先级：命令行 `--weights` > 任务 `weight` > `dimension_weights` 维度平分 > 等权。
  `dimension` 写 `null` = 只做专项诊断、不计入综合分。
- **可比性守卫**：配对检验会检查样本覆盖率、`protocol` / `mode` / `num_fewshot` 是否一致，
  不满足时在表内告警并标记结论不可用。

### 3.4 工程能力

| 能力 | 入口 | 要点 |
| :--- | :--- | :--- |
| 断点续跑 | `run --resume` | 按样本 id 跳过已落盘行 |
| 失败补跑 | `run --resume --retry-failed` | 先剔除 `error` 行再重跑，避免限流失败被永久固化 |
| 任务并行 | `run --parallel-tasks N` | 总在途 = N × `concurrency`；模型之间仍串行 |
| 分批合并 | `merge-runs DIR…` | 同「模型 × 任务」取 n 更大的；会补写 `config.json` 以保留维度权重 |
| 磁盘缓存 | `cache_dir` | **只缓存 `temperature == 0`**，避免污染 |
| 请求限流 | `rate_limit_qps` | 网关 / 现网通道建议 1~2 |
| 逐样本归因 | `scripts/diagnose.py` | 六桶：error / empty / truncated / thinking / unparsed / scored |
| 离线重判 | `scripts/rescore.py` | 不重发请求，按新规则重算；受 4000 字符截断限制 |
| 版本核对 | `scripts/verify_install.py` | 按 `CODE_MANIFEST.txt` 的 sha256 核对内网代码 |
| 打包 | `scripts/pack_bundle.py --code-only` | 生成源码增量包 + 清单，排除 datasets / 配置 |

### 3.5 模型接入

`provider` 支持三类：

- `openai_compat`（别名 `openai` / `vllm` / `sglang` / `tgi` / `lmdeploy`）：标准 OpenAI 兼容协议；
- `reqbody_gateway`：整个请求体包进 `reqBody`、鉴权放 body 内（现网 Qwen3-32B 用这个）；
- `mock`：确定性假模型，用于自检 / 干跑。

常用模型级字段：`base_url`、`api_key`、`model`、`params`（temperature / max_tokens）、
`extra_body`（如 `{"chat_template_kwargs": {"enable_thinking": false}}` 关闭 Qwen3 思考模式）、
`supports_logprobs`、`user_suffix`（如 `/no_think`）、`rate_limit_qps`。

---

## 4. 使用说明（按顺序执行）

> 前置：所有命令**必须在项目根目录**执行（`geval` 是项目内的包）。
> 换目录会报 `No module named geval`；必须换目录时设 `PYTHONPATH`。

### 4.1 五分钟验证框架本身

```bash
python -m geval selftest          # 期望：全部自检项通过，约 1 秒
python -m geval list-tasks        # 期望：12 个任务
python -m geval preview --task json_schema   # 看与业务最贴近的 prompt
```

### 4.2 就绪检查

```bash
python -m geval doctor --config config.json
# 退出码 0 = 无阻塞；1 = 有阻塞项，按提示修复
# 加 --probe 会顺带探活（发真实请求）
```

### 4.3 真实评测的五步（当前单模型口径）

```bash
python -m geval probe   --config config.json            # 1. 端点探活，确认 model 名
python -m geval run     --config config.json --smoke --limit 8   # 2. 冒烟，验证接线
python -m geval count   --config config.json            # 3. 用量预估（19,889 请求）
python -m geval run     --config config.json --limit 40 --resume # 4. 抽样看排序
python -m geval run     --config config.json --resume            # 5. 全量
```

报告自动生成在 `runs/<run_name>/report.html` 与 `report.md`。
`run` 结束时已顺带出报告；只改口径时用 `report`：

```bash
python -m geval report --run runs/geval-llm-chat-core \
       --weights chinese=0.35,reasoning=0.3,instruction=0.3,math=0.05,json_schema=0
```

### 4.4 5 对象模板（阶段 1 / 1.5）

```bash
cp config.example.json config.5objects.json     # 填 5 个通道的 base_url / model

# 阶段 1：微调前 3 个
python -m geval run --config config.5objects.json \
       --models pangu7b-pre,qwen8b-pre,qwen3-32b --resume

# 阶段 1.5：补上 2 个 post，报告自动出 pre→post 退化对比
python -m geval run --config config.5objects.json --resume
```

也可用 `CFG=config.5objects.json bash scripts/run_stage1.sh`（脚本默认已改为读这份配置）。

### 4.5 大数据量：并行与分批

```bash
# 任务并行（首选）
python -m geval run --config config.json --parallel-tasks 3 --resume

# 分批 + 合并（必须各用各的 run 目录）
python -m geval run --config config.json --run-name p-cmmlu --tasks cmmlu --resume &
python -m geval run --config config.json --run-name p-rest  --tasks ceval,gsm8k,bbh,ifeval,json_schema --resume &
wait
python -m geval merge-runs runs/p-cmmlu runs/p-rest --run-name merged
```

细节见 `docs/BATCH_RUNBOOK.md`。

### 4.6 结果异常时的排查顺序

1. `python scripts/diagnose.py runs/<run> --models <id> --tasks <task>` 看失败归因；
2. `summary.json → errors` 与 `manifests.client_stats`（`http_retries` / `errors`）；
3. 失败行先 `--resume --retry-failed` 补跑，**不要直接出报告**；
4. 判分规则修过而数据没变 → `python scripts/rescore.py runs/<run>` 离线重判。

---

## 5. 关键概念（读代码前必须建立的心智模型）

### 5.1 `name` vs `tag`

`name` 是任务类型（`ceval`），`tag` 是本次配置里的别名（`ceval-direct`）。
`--tasks` **只匹配 `name`**，写 tag 会匹配不到并报错；同一 `name` 下多个 tag 会被一起跑。
报告和落盘文件名用 `tag`。

### 5.2 `protocol` 与 `mode` 是两件正交的事

`protocol` 决定**怎么问**（直接答 / 最后一行收口 / 给 few-shot），
`mode` 决定**怎么判**（生成后抽取 / 首 token 概率 / 续写似然）。
常见组合是 `fewshot + gen`，对齐榜单用 `fewshot + loglikelihood`。

### 5.3 去重与抽样

所有任务默认按题面去重（`"dedupe": false` 可关）。`--limit N` 是**按 subject 分层等比抽样**，
每个分组至少 1 条，`seed` 固定才可复现。抽样结果不能与公开榜单直接对齐。

### 5.4 `--resume` 与 `error` 行

失败样本会以带 `error` 的行写进 jsonl。`--resume` 默认把它们当成「已完成」，
所以被限流打崩之后必须用 `--resume --retry-failed`，否则失败率被永久固化。

### 5.5 `summary.json` 是整体覆写

两个进程共用一个 run 目录，后结束的会覆盖先结束的 `summary.json`。
**分批必须分 `--run-name`**，跑完再 `merge-runs`。

### 5.6 综合分权重优先级

`--weights`（仅报告） > 任务 `weight`（绝对） > `dimension_weights` 维度份额（同维度平分） > 等权。
权重只影响综合分，不改任何逐样本结果，不需要重跑模型。

### 5.7 报告章节号是动态的

可选章节（配对 / 微调影响 / 续写似然 / 结构化 / 指令遵循）缺失时，后续编号前移。
引用报告时按标题，不要写死「第 4 节」。

### 5.8 chat template 是配对对比的隐藏变量

SFT 若改了 chat template，同一段 prompt 在两个模型上会被拼成不同输入，差值方向不可预期，
现象与灾难性遗忘完全一样。**微调时不要改模板**；必须改就把微调前模型也用新模板重测。

---

## 6. 交接后：文档阅读清单（给接手的 Agent）

### 6.1 分层阅读顺序

| 层级 | 文档 | 读什么 | 何时读 |
| :--- | :--- | :--- | :--- |
| **L0 必读** | `HANDOVER.md`（本文） | §1 背景、§2 现状、§3 特性、§5 概念 | 接手第一分钟 |
| **L0 必读** | `README.md` | §1 快速开始、§3 五个设计决策、§4 报告解读、§5 配置、§10 FAQ | 动手前 |
| **L0 必读** | `docs/ARCHITECTURE_SUMMARY.md` | §2 仓库结构、§3 分层与数据流、§4 五步启动 | 建立代码地图 |
| **L1 执行前** | `docs/INTRANET_RUNBOOK.md` | §1 前置核查、§2 部署、§3 配置、§4 数据、§5 执行、§7 故障排查 | 第一次跑真实评测前 |
| **L1 执行前** | `datasets/README.md` | 目录约定、字段兼容写法、冻结纪律 | 换数据 / 加数据时 |
| **L1 执行前** | `config.example.json` | 5 对象模板的字段与注释（`_说明_*`） | 配通道时 |
| **L2 解释结果前** | `docs/EVAL_METHODOLOGY.md` | §3 能力对照、§4 acc/acc_norm、§5 可比性边界、§6 混合方案 | 要对外汇报前，**必读** |
| **L2 解释结果前** | `docs/COVERAGE.md` | 五项需求覆盖度、专业能力缺口、前置确认项 | 排期 / 对齐预期时 |
| **L3 规模化时** | `docs/BATCH_RUNBOOK.md` | §1 执行结构、§3 任务并行、§4 分批合并、§7 进度排查 | 跑全量 / 多机时 |
| **L3 可选** | `docs/OFFLINE_DEPLOY.md` | lm-eval 离线部署、数据集预热、两个已知坑 | 需要对齐公开榜单时 |

### 6.2 代码入口地图（按修改需求定位）

| 想改什么 | 看哪里 |
| :--- | :--- |
| CLI 参数 / 子命令 | `geval/__main__.py` |
| 调度、续跑、并行、落盘、指纹 | `geval/runner.py` |
| 新增任务 / 改判分 | `geval/tasks.py`（`TASK_REGISTRY`） |
| prompt 模板 | `geval/prompts.py` |
| 答案抽取（选项 / 数值 / JSON） | `geval/answers.py` |
| 统计（CI / McNemar / 分位） | `geval/metrics.py` |
| 报告章节 / 图表 / 权重 | `geval/report.py` |
| HTTP / 重试 / 代理 | `geval/http.py`、`geval/client.py` |
| 现网网关适配 | `geval/reqbody_gateway.py` |
| 合并 run | `geval/merge.py` |
| IFEval 校验器 | `geval/ifeval.py`（`CHECKERS`） |

### 6.3 给接手 Agent 的启动提示（可直接复制）

```text
你在 /path/to/llm-general-eval 工作。这是一个零第三方依赖的 Python 评测框架 geval 0.1.0。
先按顺序读：HANDOVER.md → README.md（§1/§3/§4/§5）→ docs/ARCHITECTURE_SUMMARY.md
→ docs/INTRANET_RUNBOOK.md（§1-§5、§7）→ docs/EVAL_METHODOLOGY.md（§3-§6）
→ docs/COVERAGE.md → datasets/README.md。
硬约束：
1) 所有命令必须在项目根目录执行；
2) config.json 含真实密钥，不要打印、不要提交、不要写进任何文档；
3) 不要直接跑全量；先 selftest → doctor → probe → smoke → count → limit 40；
4) --tasks 只匹配 name 不含 tag；分批必须用不同 --run-name；
5) 失败行必须先 --resume --retry-failed 补跑，再出报告；
6) 单模型结果不能用于选型结论，报告章节号是动态的。
```

---

## 7. 需要了解的背景知识

### 7.1 评测方法论

- **基准与指标是主流的，框架是自建的**。geval 用的题与 lm-eval / OpenCompass 同批，
  `acc` / `acc_norm` 口径也对齐，但 prompt 模板细节不同，**绝对值不可与公开榜单直接比**。
- **`acc_norm` 的意义**：对续写 logprob 做长度归一化，否则短选项系统性占优。
- **few-shot 泄漏**：geval 在缺 dev 切分时会拿测试集前 k 条当示例，并自动从评测集剔除。
- **可比性**：模型间相对差距、微调前后差值可信；对外报绝对值必须用 lm-eval / 官方模板。

### 7.2 统计

- **Wilson 区间**：小样本比例指标的置信区间，比正态近似稳。
- **bootstrap**：重采样估计准确率与差值的分布，报告里的 ± 即 95% 半宽。
- **McNemar 精确检验**：同一批样本上两个模型的配对差异检验，看的是「一个对一个错」的 discordant 对。
- **判定纪律**：CI 跨 0 或 p ≥ 0.05 → 视为不显著，不能作为选型依据。
- **配对的两个前提**：同一批样本、同一套 prompt；外加 chat template 未被改动。

### 7.3 模型服务与 API

- **OpenAI 兼容**：geval 走 `/chat/completions`；`loglikelihood` 额外需要 `/completions` + `echo` + `logprobs`。
- **`local-completions` vs `local-chat-completions`**（lm-eval）：只有前者能出 `acc_norm`。
- **thinking 模型**：Qwen3 系列需 `extra_body.chat_template_kwargs.enable_thinking=false`，
  否则 token 预算被思考过程吃光，表现为输出为空或整片 0 分（落盘的 `finish_reason` 可归因）。
- **网关限流**：`rate_limit_qps` + 低并发；多进程叠加会打爆配额，表现为大面积 429 / 超时。

### 7.4 业务与训练

- **意图识别指标**：意图准确率只是第一层，真正影响线上的是**幻觉槽位**（用户没说的填了）、
  **意图串位**、**该拒识却硬答**。这些 geval 目前只实现了前两层，其余待补。
- **灾难性遗忘**：SFT 后通用能力下降是常态，必须用配对检验判断「降的这点是否显著」。
- **级联架构**：若做「小模型初筛 + 32B 精识别」，评测对象变成路由系统，
  要额外测初筛命中率、升级率、**路由错误率**（最致命）、端到端时延与成本。
- **数据冻结**：test 集必须在阶段 0 冻结并记 sha256；任何进入训练的数据不能出现在 test 里。

---

## 8. 已知边界与坑（按踩坑频率排序）

| # | 现象 | 原因 | 处置 |
| :- | :--- | :--- | :--- |
| 1 | `HTTP 0 URLError` / 超时 / 502，但 `curl` 通 | `HTTP_PROXY` 把内网请求发给外网代理 | `unset` 代理；确认 `proxy: "bypass"` |
| 2 | `404 model not found` | `model` 字段与服务端加载名不一致 | `probe` 看服务端实际模型名 |
| 3 | 报告少了一半任务 | 多进程共用 run 目录，`summary.json` 互相覆盖 | 分批必须不同 `--run-name`，再 `merge-runs` |
| 4 | 加了 `--parallel-tasks` 没变快 | 服务端已排队 | 看 `client_stats.http_retries`，涨了就调小 |
| 5 | `--tasks ceval-likelihood` 报错 / 没跑 | `--tasks` 只认 `name`，不认 tag | 用 `--tasks ceval`，或配置里只留目标变体 |
| 6 | 报告里找不到「第 4 节」 | 章节号动态，可选章节缺失会前移 | 按标题定位 |
| 7 | `doctor` 与 `count` 条数不一致 | 前者原始行数，后者题面去重后 | 以 `count` 为准 |
| 8 | 限流打崩后分数异常低 | `error` 行被 `--resume` 当成已完成 | `--resume --retry-failed` 补跑 |
| 9 | 选择题分数异常、`json_valid_rate` 骤降 | thinking 未关 / `max_tokens` 截断 / 抽取失败 | `diagnose.py` 归因；看 `finish_reason` |
| 10 | 微调后全面掉点，像灾难性遗忘 | chat template 被改 | 比对两端 `tokenizer_config.json` 的 `chat_template` |
| 11 | `rescore.py` 报「响应可能被截断」 | jsonl 的 response 落盘截到 4000 字符 | 该类样本重跑，不要重判 |
| 12 | 内网无法 `pip install` / 无外网 | 设计如此 | geval 零依赖；只有 lm-eval 与数据准备需要额外准备 |
| 13 | `huggingface.co` 直连不通 / 缓存不可写（本机） | 网络与权限限制 | `HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME=<项目>/.cache/hf` |
| 14 | ARM64 上装不上训练框架 | ARM64 + GPU 生态窄 | 训练大概率换 x86 + GPU 机器；评测留在 ARM64 |
| 15 | 归档/发群 `runs/` 时担心泄露密钥 | 旧版把 `config.json` 原样落盘（含 `api_key`） | 新版已对 `api_key` 打码为 `***`；**升级前产出的 run 目录需人工检查** |

### 明确未覆盖的能力

| 项 | 状态 | 建议 |
| :--- | :--- | :--- |
| IFEval `language:response_language` | 未实现 | 未实现类型会排除并单列；扩展点 `geval/ifeval.py::CHECKERS` |
| IFEval 宽松模式 | 仅严格 | 对齐官方 loose 需在 `check_response` 前处理首尾行 |
| 长上下文 / 多轮 | 未实现 | 补 RULER / Needle-in-Haystack，或自定义多轮任务 |
| LLM-as-judge | 未实现 | 可用 Qwen3-32B 当裁判，注意位置偏差与自我偏好 |
| 安全 / 拒答 | 未实现 | 有合规要求时单列 |
| 专业指标 | 未实现 | 混淆矩阵、槽位 F1、幻觉槽位率、拒识率 / 误触发率 |
| 性能压测（TTFT / P99 / 吞吐） | 仅单请求时延 | 用 EvalScope `perf` 或 vLLM bench |

---

## 9. 交接待办（按优先级）

### P0 · 接手后 1~2 天内必须确认

| # | 事项 | 验收标准 |
| :- | :--- | :--- |
| 1 | 确认 `config.json` 的端点、`model` 名、鉴权有效 | `probe` 返回 1/1 可用，时延合理 |
| 2 | 跑通冒烟 | `--smoke --limit 8` 无 `[error]`，解析失败率接近 0 |
| 3 | 冻结评测集并记录 sha256 | `manifest.json` 有各数据集指纹；与交付数据一致 |
| 4 | 明确报告口径负责人 | 谁签「显著 / 不显著」的结论 |

### P1 · 阶段 1 正式评测

| # | 事项 | 验收标准 |
| :- | :--- | :--- |
| 5 | 单模型全量（19,889 请求） | 必备章节齐全、`errors` 为空、解析失败率 < 2% |
| 6 | lm-eval 锚点（可选但推荐） | 同一批 200 条，ceval 差值 1~3 个点内 |
| 7 | 5 对象通道就绪 | 5 个 `base_url` 全部 `probe` 通过；chat template 未变 |
| 8 | 阶段 1 基线（3 模型） | 出准确率矩阵 + 时延；确定 prompt 协议 |

### P2 · 阶段 1.5 / 专业能力

| # | 事项 | 验收标准 |
| :- | :--- | :--- |
| 9 | 补齐 post 模型，出 pre→post 对比 | 每任务有差值、CI、p 值、判定 |
| 10 | 接入真实业务测试集（含难例 / 模糊 / 拒识） | 替换 16 条示例；冻结并记 sha256 |
| 11 | 实现专业指标 | 混淆矩阵 + 槽位 F1 + 幻觉槽位率 + 拒识率 |
| 12 | 定死验收口径 | 例如「专业提升显著，且 ceval/ifeval 退化 ≤ 2 点且不显著」 |

### P3 · 长期

| # | 事项 |
| :- | :--- |
| 13 | 长上下文 / 多轮任务 |
| 14 | LLM-as-judge 开放式评测 |
| 15 | 安全与拒答专项 |
| 16 | 训练-评测闭环（与 LLaMA-Factory / ms-swift 对接） |

---

## 10. 变更记录（相对上一版文档）

本次交接做了以下对齐与修正，均已落到文件：

| 文件 | 变更 |
| :--- | :--- |
| `README.md` | 报告章节表改为实际的 11 节并说明编号动态；目录树按现状更新（datasets / dist / runs）；顶部加交接入口 |
| `docs/INTRANET_RUNBOOK.md` | `count` 期望输出改为去重后实测值（19,889）；澄清 doctor 与 count 的差异；修正 `--tasks` 误用 tag 的两处命令；报告章节表补全；补充 config.json 密钥提醒 |
| `docs/BATCH_RUNBOOK.md` | 规模表改为去重后条数与占比（19,889），删除过时的「14.8 倍」表述；说明原始/去重差异 |
| `docs/EVAL_METHODOLOGY.md` | 标准库模块数改为实测口径；修正 `--tasks ceval-likelihood` 的无效命令；锚点建议统一为 lm-eval |
| `docs/COVERAGE.md` | 「报告第 4 节」改为按标题定位 |
| `docs/ARCHITECTURE_SUMMARY.md` | 文档索引加入本文 |
| `datasets/README.md` | 任务清单补 `intent_json`；补充去重口径说明 |
| `geval/__main__.py` | `doctor` 的下一步提示不再写死「三个端点」，按实际模型数生成 |
| `geval/runner.py` | 落盘的配置快照对 `models[].api_key` 打码，避免 `runs/` 归档或发群时泄露密钥 |
| `scripts/run_stage1.sh` | 默认配置改为 `config.5objects.json`，避免单模型配置下误跑 5 对象模型 id |
| `scripts/diagnose.py` | 桶数文案「五个」→「六个」 |

---

## 11. 交接确认

| 项 | 交接人 | 接收人 | 日期 |
| :--- | :--- | :--- | :--- |
| 代码与文档 | | | |
| 数据集与 sha256 | | | |
| 配置与端点权限 | | | |
| 历史报告与结论 | | | |

> 最后提醒：**当前没有任何真实评测结果**。任何引用「geval 跑出 XX%」的说法，
> 都必须先确认对应的 `runs/<run_name>/manifest.json` 与明细 jsonl 存在，且不是 mock 产物。
