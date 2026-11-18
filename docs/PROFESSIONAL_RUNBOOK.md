# 专业能力评测系统 · 命令手册（阶段 4：语音导航意图树路由）

> 面向操作人员。所有命令都经过实测，输出示例来自本机 `config.professional.json`（mock 配置）。
> 配套代码说明见 `geval/professional.py`、`geval/telemetry.py` 的模块注释；
> 方法论与验收口径见 `EVAL_METHODOLOGY.md` §7、`HANDOVER.md` §9。

---

## 0. 先认清三个目录（最容易踩的第一步）

| 名称 | 绝对路径 | 里有什么 |
| :--- | :--- | :--- |
| 工作区根 | `D:/工作/文件/模型评测` | 业务标注 `export_11.xlsx`、两个意图树模板、`语音导航意图识别评测数据集.jsonl`、`build_dataset.py` |
| **项目根** | `D:/工作/文件/模型评测/llm-general-eval/llm-general-eval` | `geval/`、`scripts/`、`config*.json`、`datasets/`、`runs/`、`docs/` |
| 评测数据集 | `<项目根>/datasets/intent_routing/` | `test.jsonl`、`labels.json`、`dataset_info.json` |

**所有命令必须在项目根执行**，因为 `python -m geval` 要能 import 到 `geval` 包。你贴的第一行
`cd llm-general-eval/llm-general-eval` 是相对**工作区根**的，完整写法：

```bash
cd "D:/工作/文件/模型评测/llm-general-eval/llm-general-eval"
```

Windows Git Bash / PowerShell 都用正斜杠；路径含中文，务必加引号。
后文所有命令都默认已经在这个目录下。

---

## 1. 你贴的四条命令，逐条是什么意思

| # | 命令 | 干什么 | 调模型吗 | 产出 |
| :- | :--- | :--- | :--- | :--- |
| 1 | `cd llm-general-eval/llm-general-eval` | 进入项目根 | 否 | — |
| 2 | `python scripts/prepare_professional_dataset.py` | 把业务 JSONL 转成框架能读的格式 | 否 | `datasets/intent_routing/` 三个文件 |
| 3 | `python -m geval preview --task intent_routing --config config.professional.json` | 打印**真正发给模型**的那段 prompt | 否 | 只打印到屏幕 |
| 4 | `python -m geval run --config config.professional.json --resume` | **唯一会调用模型的一条**：逐条推理、判分、落盘 | **是** | `runs/<run_name>/` 整个目录 |
| 5 | `python scripts/professional_report.py runs/geval-professional-intent --cost-in 0.5 --cost-out 2.0` | 读已落盘明细，重算专业指标与成本 | 否 | 专项报告 md + json |

> 注意是 **5 条**（你贴的是 1 条 `cd` + 4 条命令）。前三步是准备与检查，第 4 步花钱花时间，第 5 步可以随时反复跑。

### 1.1 `prepare_professional_dataset.py` — 数据准备

读工作区根的 `语音导航意图识别评测数据集.jsonl`（432 条），字段映射：

```
question <- prompt          （自包含提示词：498 节点候选树 + 历史 + 当前句 + 小模型意图）
answer   <- ground_truth    （人工标注意图名称，真实答案）
subject  <- sample_type     （业务/FAQ意图 | 通用交互意图，用于分层）
meta     <- 会话ID / 轮次 / 小模型意图 / 源行号 等追溯字段
```

产出 `datasets/intent_routing/` 下三个文件：`test.jsonl`（432 条）、`labels.json`（498 条层级路径 / 495 个末级标签）、`dataset_info.json`（**源文件 sha256，用于冻结测试集**）。

实测输出：

```
已生成 D:\...\datasets\intent_routing
  样本 432 条 | 候选路径 498 个 | 末级标签 495 个
  源文件 sha256 7e902bf7dc9f83adf561a287348b3646801bde8dca22764f37336a1911057db0
  test.jsonl 7778.4 KB
```

常用变体：

```bash
python scripts/prepare_professional_dataset.py --limit 40 --out datasets/intent_routing_smoke  # 只转前 40 条
python scripts/prepare_professional_dataset.py --input "D:/其他路径/xxx.jsonl"                 # 换源
```

**什么时候必须重跑**：业务数据集重新生成、意图树改了、标注修订了。它是幂等的，重复执行只会覆盖同名文件。
**不要跳过的一步**：`labels.json` 缺失时任务会退化为 12 个兜底标签，幻觉意图率和准确率全部失真。

### 1.2 `preview` — 发请求前看一眼 prompt

只读、不写文件、不联网。打印第 1 条样本的完整 prompt 和标准答案，用来确认：候选树是否完整塞进去了、历史对话格式对不对、当前句和小模型意图在不在、末尾的输出约束有没有被改坏。

```bash
python -m geval preview --task intent_routing --config config.professional.json
python -m geval preview --task intent_routing --config config.professional.json --index 5   # 看第 6 条
```

实测开头：

```
  [info] intent_routing: 已加载 495 个候选意图名称 / 498 条层级路径
# 任务 intent_routing ｜ 协议 direct ｜ 模式 gen ｜ few-shot 0 条
# 标准答案：个人账户交易明细查询
[user]
你是银行语音导航系统的意图识别专家。请...
【候选意图树】（共 498 个节点，每行一个层级路径，"/" 分隔层级）
applepay介绍
...
```

### 1.3 `run --resume` — 真正跑评测

**这是整套系统里唯一会调用模型的命令。** 它按配置里的 `models × tasks` 并发执行，每条样本：
发 prompt → 抽取模型输出的意图名称 → 与人工标注比对 → 连同 token / 时延 / 缓存 / 截断信息**逐条 flush 落盘**。

`--resume` = 断点续跑：读取已有的 `<model>/<tag>.jsonl`，跳过已完成的样本 ID。第一次跑加不加都行，加上没坏处；跑到一半被限流/断网中断后，**必须加**才能接着跑。

实测输出（mock 配置、432 条）：

```
[意图评测-离线自检 × intent-routing] 样本 432 条，协议 direct，模式 gen
  intent-mock/intent-routing 432/432 (100.0%) 427.1/s ETA 0s
完成，用时 1.8s
model               intent-routing
intent-mock                  76.2%
```

> ⚠️ **当前 `config.professional.json` 里是 `provider: "mock"` 的假模型**（`accuracy: 0.78`），
> 上面 76.2% 只是验证链路，不是任何模型的真实能力。正式评测前必须按 §3 改配置。

常用变体：

```bash
# 先小样接线（强烈建议）：分层抽 8 条，单独一个 run 目录，不污染正式结果
python -m geval run --config config.professional.json --limit 8 --run-name pro-smoke8 --tasks intent_routing

# 正式全量（可中断，重跑同一命令即续跑）
python -m geval run --config config.professional.json --resume

# 被网关限流打崩了一批：剔除 error 行后只补跑失败样本
python -m geval run --config config.professional.json --resume --retry-failed

# 换一个运行目录做 A/B（例如换 prompt / 换模型），不要覆盖旧结果
python -m geval run --config config.professional.json --run-name pro-qwen32b --resume

# 并发与安静模式
python -m geval run --config config.professional.json --resume --quiet
```

### 1.4 `professional_report.py` — 出专项报告

不重跑模型，只读 `runs/<run>/<model>/*.jsonl` 明细重算，因此**改单价、改分组口径都可以反复执行，不花一分钱**。

```bash
python scripts/professional_report.py runs/geval-professional-intent
python scripts/professional_report.py runs/geval-professional-intent --cost-in 0.5 --cost-out 2.0
python scripts/professional_report.py --model intent-mock          # 省略 run 时自动取 runs/ 下最近修改的目录
```

**`--cost-in 0.5 --cost-out 2.0` 是「每 1000 个 token」的输入/输出单价，那两个数字只是示例，不是任何真实报价。**
计算方式：`prompt_tokens/1000 × cost-in + completion_tokens/1000 × cost-out`。
不传单价、或配置里 `cost` 全为 0，报告显示「未配置单价」，不会臆造成本。

实测输出：

```
专业能力专项报告已生成
  ...\runs\professional-full-mock\professional_report.md
  ...\runs\professional-full-mock\professional_metrics.json
  样本 432（有效 432，失败 0）｜意图准确率 76.16%｜小模型基线 85.88%｜增益 -9.72%｜p=0.0004
  total tokens 1376984｜平均 3187.5/样本｜p95 608 ms｜墙钟 1.00 s｜成本 未配置单价
```

带 `--cost-in 0.5 --cost-out 2.0` 时末列变成 `成本 690.5830`（示例单价算出的数，别拿去汇报）。

---

## 2. 完整命令速查表

### 准备与自检（都不调用被测模型）

| 命令 | 用途 |
| :--- | :--- |
| `python -m geval selftest` | 离线端到端自检，验证框架本身没坏；改动代码后第一件事 |
| `python -m geval list-tasks` | 列出 13 个任务，确认 `intent_routing` 已注册 |
| `python scripts/verify_deps.py` | 确认零第三方依赖 |
| `python scripts/verify_install.py` | 与 `CODE_MANIFEST.txt` 比对源码 sha256（内网部署必跑） |
| `python -m geval doctor --config config.professional.json` | 就绪检查：Python / 配置 / 数据集 / 代理 / 端点 |
| `python -m geval doctor --config config.professional.json --no-selftest` | 同上但跳过自检，更快 |
| `python -m geval count --config config.professional.json` | **跑之前先看清样本量与请求量**，不发任何请求 |
| `python -m geval preview --task intent_routing --config config.professional.json` | 看 prompt |
| `python -m geval probe --config config.professional.json` | 端点探活（每个模型发 1 条最小请求） |
| `python scripts/prepare_professional_dataset.py` | 重新生成专业数据集 |

`count` 实测（用来估算预算）：

```
intent-routing      gen      432    1    1    432
合计                         432         1    432
时长估算：总请求 432 × 时延(秒) ÷ 并发 4    例：单请求 0.5s → 约 0.9 分钟
```

### 执行与报告

| 命令 | 用途 |
| :--- | :--- |
| `python scripts/run_professional_eval.py --dry-run` | ★ 一键脚本：打印计划、0 请求；去掉 `--dry-run` 即执行全流程（见 §3） |
| `python -m geval run --config config.professional.json --limit 8 --run-name <name>` | 小样接线 |
| `python -m geval run --config config.professional.json --resume` | 正式全量 / 断点续跑 |
| `python -m geval run --config config.professional.json --resume --retry-failed` | 补跑失败样本 |
| `python -m geval report --run <run目录>` | 重出**通用**报告（report.md / report.html，不重跑模型） |
| `python scripts/professional_report.py <run目录>` | 出**专业**专项报告 |
| `python scripts/diagnose.py <run目录>` | 逐样本归因：请求失败 / 输出为空 / 思维链污染 / 抽取失败 |
| `python -m geval merge-runs <run1> <run2> --run-name merged` | 多个分批运行合成一张总表 |

> `scripts/rescore.py` 是给通用任务的离线重判脚本，未适配 `intent_routing`，不要用在专业评测上。

---

## 3. 正式评测前：把 mock 换成真实模型

用编辑器打开 `config.professional.json`，把 `models` 数组里的第一个对象**整块**替换为文件内 `_真实端点模板` 的内容，然后：

1. `api_key` 从同目录 `config.json` 复制（**不要写进版本库 / 不要发群**；`runs/*/config.json` 快照会自动打码成 `***`）。
2. `base_url` 填到 `/v1` 为止，代码自动拼 `/chat/completions`。
3. `params.temperature` 保持 `0` —— 只有确定性请求才走磁盘缓存，重跑不会重复烧钱。
4. `max_tokens: 64` 足够输出一个意图名；开太大只会让模型多说废话，拉高 `extra_prose_rate`。
5. `rate_limit_qps` 按网关配额设（现网通道建议 1~2）；`proxy: "bypass"` 不要改。
6. `cost.prompt_per_1k` / `completion_per_1k` 填**真实合同单价**，之后出报告就不用再带 `--cost-in/--cost-out`。

### 更省事：一键脚本（推荐）

`scripts/run_professional_eval.py` 把下面 5 步封装成一条命令，并自动从 `config.json` 读取真实端点
（**api_key 只在内存里注入，不写出含密钥的配置文件**；runner 的快照会自动打码）。
默认先探活，失败立即停止，不会浪费 432 次请求。

```bash
python scripts/run_professional_eval.py --dry-run                              # 只看计划，0 请求
python scripts/run_professional_eval.py --mock --full --run-name pro-selftest  # 离线自检链路
python scripts/run_professional_eval.py --limit 8 --run-name pro-wirecheck    # 真实端点小样接线
python scripts/run_professional_eval.py --full --resume                       # 全量正式评测
python scripts/run_professional_eval.py --full --resume --retry-failed         # 补跑失败样本
python scripts/run_professional_eval.py --full --resume --cost-in <单价> --cost-out <单价>
```

常用参数：`--model <id>`（选 config.json 里的模型）、`--qps 2`、`--concurrency 8`、
`--no-probe`（跳过探活）。完整参数 `python scripts/run_professional_eval.py --help`。

**推荐执行顺序**（不用一键脚本时，每一步都在项目根）：

```bash
python -m geval selftest
python -m geval doctor --config config.professional.json          # 看代理警告
python -m geval probe  --config config.professional.json          # 1/1 可用才继续
python -m geval count  --config config.professional.json          # 确认 432 条 / 432 请求
python -m geval run    --config config.professional.json --limit 8 --run-name pro-wirecheck
python -m geval run    --config config.professional.json --resume
python scripts/professional_report.py runs/geval-professional-intent
```

`doctor` 在本机会提示检测到 `HTTP_PROXY` 等环境变量；配置里 `proxy=bypass` 已经规避，但如果用真实端点出现超时/502，先在 shell 里 `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy`。

---

## 4. 输出目录里都是什么

```
runs/geval-professional-intent/
├── manifest.json                    运行环境指纹：版本/Python/并发/seed/模型(不含 key)/任务
├── config.json                      生效配置快照（api_key 已打码）
├── summary.json                     模型×任务总表 + client_stats + 错误列表
├── report.md / report.html          通用报告（框架自动生成）
├── professional_report.md           ★ 专业专项报告（本手册 §5）
├── professional_metrics.json        ★ 同批指标的机器可读版（可接看板）
└── <model_id>/
    ├── intent-routing.jsonl          ★ 逐样本明细，每条含 pred/gold/correct/
    │                                  prompt_tokens/completion_tokens/total_tokens/
    │                                  latency_ms/request_wall_ms/wall_start/wall_end/
    │                                  cached/finish_reason/strategy/extras
    └── intent-routing.summary.json   任务级聚合（准确率+CI+时延+task_metrics）
```

明细是逐条 flush 的，所以中断后 `wc -l` 就能看进度；`--resume` 靠 `id` 判断哪些已完成。
换模型/换 prompt 时**一定换 `--run-name`**，否则会复用同一份缓存结果，看起来"没变化"。

---

## 5. 报告怎么读

| 指标 | 含义 | 该看什么 |
| :--- | :--- | :--- |
| 意图准确率（严格全等） | 模型输出的意图名称 == 人工标注意图名称 | 主指标。注意候选树有 495 个标签，不是选择题 |
| 小模型基线准确率 | 同一批样本上 export 表「机器识别意图」的命中率 | 本次数据是 **85.88%**，任何大模型必须跟它比 |
| 大模型相对增益 / 95% CI | 两者逐样本配对之差 | CI 跨 0 就是没拉开差距 |
| McNemar p / 判定 | 配对显著性检验 | **只有 p < 0.05 才能说"显著优于/劣于"**，否则一律记为不显著 |
| 输出落在候选树内 / 幻觉意图率 | 模型有没有编造树外标签 | 幻觉意图会直接让下游路由崩，比答错更严重 |
| 解析失败率 / 夹带解释率 / 截断率 | 是抽取器或 prompt 的问题，不是模型能力 | >2% 先改 prompt 和 `max_tokens`，别急着下结论 |
| 分层表现（业务 / 通用交互） | 按 `sample_type` 拆开 | 整体分会被「人工服务」这类高频简单意图主导，必须分层看 |
| 意图混淆 Top 12 | 真实意图 → 模型输出 | 定位是哪两个业务节点在打架 |

**验收口径（沿用 `EVAL_METHODOLOGY.md` §7.5，建议提前定死）**：
专业准确率提升显著，且 `ceval`/`ifeval` 通用能力退化 ≤ 2 个点且不显著，才可考虑替换 32B。

### Token 与时间消耗章节

| 字段 | 口径 |
| :--- | :--- |
| prompt / completion / total tokens | 服务端 `usage` 原值逐样本累加；`total_tokens` 也单独落盘 |
| 平均 tokens/样本 | 判断能否承担线上成本的核心数；本数据集 prompt 很长（约 6.3k 字符），输入占绝对大头 |
| 单请求时延 mean / p50 / p90 / p95 / p99 | 客户端 `perf_counter` 计时，**不含**限流排队时间 |
| 累计请求时延 | 各请求时延之和 |
| 运行墙钟时间 | 首条请求开始 → 末条请求结束；与累计时延之比反映并发度 |
| completion tokens/s、样本吞吐 | 按累计时延算，并发下的参考值 |
| 缓存命中 / 输出被截断 | `cached`、`finish_reason=="length"` |
| 估算成本 | 按每 1K token 单价；未配单价显示「未配置单价」 |

报告末尾还有一张 `client_stats` 汇总表，与逐样本口径互为校验（本机两者均为 1,376,984 tokens，一致）。

---

## 6. 故障速查

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| `ModuleNotFoundError: geval` | 不在项目根 | `cd` 到含 `geval/` 的那层目录 |
| 数据集只有 1 条 | 通用去重按 prompt 前 200 字去重 | 配置里 `dedupe` 必须为 `false`（任务类里已强制，别删） |
| `找不到数据集 intent_routing/test.jsonl` | 没跑 prepare 脚本 | 执行 §1.1 |
| 标签只有十几个、幻觉率异常 | `labels.json` 缺失，用了兜底标签 | 重跑 prepare 脚本 |
| 真实端点超时 / 502 | 环境变量代理 / 地址或鉴权错 | `doctor` → `probe`；`unset *_PROXY`；核对 `proxy: bypass` |
| 大面积 429 / 超时 | 并发超网关配额 | 调低 `concurrency`、设 `rate_limit_qps`，再 `--resume --retry-failed` |
| 重跑后错误率不降 | `--resume` 把 error 行当成已完成 | 加 `--retry-failed` |
| 输出为空 / 整片 0 分 | thinking 没关或 `max_tokens` 被思考过程吃光 | `extra_body.chat_template_kwargs.enable_thinking=false`；看明细 `finish_reason` |
| 换模型后数字没变 | 复用了同一 run 目录 / 命中缓存 | 换 `--run-name`，或清 `.cache/geval/<model>/` |
| `verify_install.py` 报「需更新」 | 改过源码但没重生成清单 | `python scripts/pack_bundle.py --code-only --list-only` 后把 `dist/CODE_MANIFEST.txt` 覆盖到项目根 |
| 内网机器上没有 `config.professional.json` | 增量包漏带根目录配置 | 该文件已加入 `pack_bundle.py` 的 `INCLUDE_FILES` 白名单；用 `--code-only` 重新打包并覆盖 |

---

## 7. 三条禁忌

1. **不要拿 `config.json` 跑专业评测** —— 那是阶段 1 的 6 个通用基准（ceval/cmmlu/gsm8k/bbh/ifeval/json_schema），与意图树无关。专业评测只用 `config.professional.json`。
2. **不要用 mock 数字下任何结论** —— mock 的 `accuracy` 是写死的 0.78，只用于验证链路、统计量和断点续跑。
3. **测试集不能进训练 / few-shot** —— `dataset_info.json` 里的 sha256 就是冻结凭据；本任务 `num_fewshot: 0`，不要改成非 0。

---

## 8. 与原有通用能力评测的关系

两套配置、两套报告互不干扰：

- `config.json` → 通用能力（阶段 1），`python -m geval run` 自动生成 `report.md/html`。
- `config.professional.json` → 专业能力（阶段 4），同样走 `run`，但额外多一份 `professional_report.md`。
- 阶段 4 的最终结论要**两张报告一起看**：专业准确率涨了多少（显著的），代价是通用能力退化多少、P99 时延与单样本 token 成本涨了多少。
