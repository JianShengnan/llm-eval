# Qwen3-8B 语音导航意图识别 · 专业能力评测说明

> 面向操作人员。本文回答三件事：**数据集能不能用来评 Qwen3-8B**、**评测代码是什么**、**具体怎么跑**。
> 配套配置：同目录 `config.qwen3-8b.json`；分布分析见 `../01_分布分析/`，微调数据集见 `../02_微调数据集/`。
> 本文所有命令按 Windows Git Bash 书写；PowerShell 把 `$CFG` 换成 `$env:CFG` 或直接写全路径。

---

## 0. 一句话结论

**可以。** 这份 432 条评测集是「模型无关」的封闭标签任务（prompt → 唯一末级意图名），
`llm-general-eval` 通过 OpenAI 兼容协议调模型，把 Qwen3-8B 以 vLLM / SGLang 等 OpenAI 兼容服务暴露后，
**换一份配置即可直接评**，数据集和评测代码都不用改。

但要接受三个前提：

1. 框架不直接加载模型权重，必须先起一个 `/v1/chat/completions` 服务；
2. Qwen3 必须关闭 thinking、`temperature=0`、`max_tokens=64`，否则输出会被思维链吃光；
3. 这份测试集只覆盖 **53/498 个节点（10.64%）**，结论只能表述为「在 53 个高频意图上的准确率」，
   不能表述为「全意图树能力」（详见 `../01_分布分析/意图节点分布分析报告.md`）。

> ⚠️ 如果你用 `../02_微调数据集/sft_train_with_eval.jsonl` 训练过模型，
> **这 432 条已经是训练集的一部分，本评测立即失效**，必须另建独立测试集（见该目录说明 §4.1）。

---

## 1. 数据集说明

### 1.1 来源与位置

| 层级 | 路径 | 内容 |
| :--- | :--- | :--- |
| 业务原始集 | `D:/工作/文件/模型评测/语音导航意图识别评测数据集.jsonl` | 432 条，含自包含 `prompt`、`ground_truth`、`history_text`、`small_model_intent`、`sample_type` |
| 框架测试集 | `<项目根>/datasets/intent_routing/test.jsonl` | 432 条，由 `scripts/prepare_professional_dataset.py` 转换 |
| 标签集 | `<项目根>/datasets/intent_routing/labels.json` | 498 条层级路径 / 495 个末级名称，用于答案抽取与幻觉判定 |
| 冻结凭据 | `<项目根>/datasets/intent_routing/dataset_info.json` | 源 JSONL 的 sha256 |

**项目根** = `D:/工作/文件/模型评测/llm-general-eval/llm-general-eval`（`python -m geval` 必须在此执行）。

字段映射（转换脚本）：

```
question <- prompt          自包含提示词：498 节点候选树 + 历史对话 + 当前句 + 小模型意图 + 输出约束
answer   <- ground_truth    人工标注意图名称（真实答案）
subject  <- sample_type     业务/FAQ意图 | 通用交互意图（分层用）
meta     <- 会话ID / 轮次 / 小模型意图 / 人工标注路径 / 源行号
```

`IntentRoutingTask.build_messages()` 直接返回 `[{"role":"user","content": question}]`，
**只有一条 user message，没有 system** —— 这点也决定了微调数据的格式（见 `../02_微调数据集/` §3）。

### 1.2 关键数字（来自 `../01_分布分析/`）

| 指标 | 值 |
| :--- | :--- |
| 样本 / 会话 | 432 条 / 140 个会话 |
| 命中末级标签 | 53 / 495 = **10.71%** |
| 命中层级节点 | 53 / 498 = **10.64%**，445 个零命中 |
| 按来源 | FAQ 节点 16/326（4.91%），业务菜单节点 37/172（21.51%） |
| 按层级 | 一级 40/399，二级 8/50，三级 5/49 |
| 分布偏斜 | `人工服务` 164 条（37.96%），Top5 60.19%，20 个标签仅出现 1 次 |
| 样本类型 | 通用交互意图 293 条，业务/FAQ 意图 139 条 |
| **小模型基线** | **371/432 = 85.88%**（任何 Qwen3-8B 结果必须跟它做配对比较） |
| 源 sha256 | `7e902bf7dc9f83adf561a287348b3646801bde8dca22764f37336a1911057db0` |
| 平均 prompt | 6369 字符（约 3.2k tokens），单样本 1 次请求 |

### 1.3 为什么它能评 Qwen3-8B

- **输出空间封闭**：只能输出 495 个末级名称之一或 `集外`，判分是严格字符串比对，不需要主观打分或裁判模型。
- **有配对基线**：`机器识别意图` 提供了逐样本配对的小模型结果，可做 McNemar 精确检验与配对 bootstrap，
  8B 是否真的比小模型强有统计口径，而不是只看一个百分数。
- **框架与模型解耦**：`geval/client.py` 只要求 OpenAI 兼容的 `/chat/completions`，
  `provider: "openai_compat"` 对 vLLM / SGLang / Ollama / FastChat 都适用。
- **token / 时延已内建**：`Usage`、`latency_ms`、`client_stats()` 逐样本落盘，
  `geval/telemetry.py` 聚合 p50/p90/p95/p99、墙钟、tokens/s、成本估算。

---

## 2. 评测代码说明

### 2.1 代码清单

| 文件（相对项目根） | 职责 |
| :--- | :--- |
| `geval/tasks.py` | 任务注册表；文件末尾**延迟注册** `intent_routing`（避免循环导入） |
| `geval/professional.py` | ★ 核心：`IntentRoutingTask`。`prepare()` 加载数据集与标签、`build_messages()` 单 user message、`extract_intent()` 六级答案抽取、`run_one()` 判分 + 小模型配对、`summarize()` 专业指标 |
| `geval/telemetry.py` | ★ `usage_summary()` / `usage_by_group()` / `run_wall_clock_s()` / `cost_estimate()` |
| `scripts/prepare_professional_dataset.py` | 业务 JSONL → 框架格式（幂等，只在数据集/意图树变更后重跑） |
| `config.professional.json` | 基础配置：任务 `intent_routing`、`dedupe:false`、默认 mock |
| `config.qwen3-8b.json` | ★ 本次新增（`result/03_Qwen3-8B评测/`）：把 models[0] 换成 Qwen3-8B |
| `scripts/run_professional_eval.py` | ★ 一键脚本：前置校验 → 计划 → 探活 → Runner → 通用报告 → 专业报告 |
| `scripts/professional_report.py` | 不重跑模型，只读明细重算专业指标、成本、分层、混淆 |
| `geval/runner.py` / `geval/report.py` | 并发、断点续跑、逐样本 flush、manifest；通用 `report.md/html` |
| `geval/client.py` | OpenAI 兼容客户端，URL = `base_url.rstrip("/") + "/chat/completions"`，`extra_body` 原样透传 |

### 2.2 一次评测的数据流

```
test.jsonl (432) ──> IntentRoutingTask.prepare()  [强制 dedupe=False，加载 labels.json]
   │
   ├─ build_messages(): [{role:user, content: 自包含 prompt}]      ← 无 system
   │
   ├─ Runner 并发 ──> client.chat(max_tokens=64) ──> Qwen3-8B /chat/completions
   │                    └─ 逐样本落盘 usage / latency_ms / cached / finish_reason
   │
   ├─ extract_intent(reply.text): exact_last_line → normalized_last_line →
   │     exact_whole → normalized_whole → label_in_text → unmapped
   │     （同时记录 in_tree / hallucinated_intent / ambiguous / strategy）
   │
   ├─ 判分：pred == ground_truth（严格全等）；并算 small_model_correct
   │
   └─ 报告：rp.build_report() → report.md/html
            professional_report.py → professional_report.md / professional_metrics.json
```

### 2.3 指标口径

| 指标 | 含义 |
| :--- | :--- |
| 意图准确率（严格全等） | **主指标**；候选树 495 标签，不是选择题 |
| 归一化准确率 | 去空白/标点后比对，用来区分"能力差"与"格式差" |
| 小模型基线 / 配对增益 / 95% CI | 逐样本配对；CI 跨 0 = 没拉开差距 |
| McNemar 精确检验 p | **只有 p<0.05 才能说显著优于/劣于** |
| 幻觉意图率 / 输出落在树内 | 模型是否编造树外标签，比答错更危险 |
| 解析失败率 / 夹带解释率 / 截断率 | 抽取器与 prompt 的问题，>2% 先改 prompt，别急着下模型结论 |
| Top 混淆对 / 业务·通用分层准确率 | 定位哪两个节点打架；整体分会被「人工服务」主导 |
| prompt/completion/total tokens、平均/样本 | 服务端 `usage` 原值逐样本累加 |
| 单请求时延 mean/p50/p90/p95/p99、墙钟、tokens/s | `perf_counter` 计时，不含限流排队 |
| 估算成本 | 每 1K token 单价；自建端点填 0，报告显示「未配置单价」 |

---

## 3. 前置：把 Qwen3-8B 跑成 OpenAI 兼容端点

框架不加载权重，必须起服务。以 vLLM 为例（**示例命令，按显卡/版本调整**）：

```bash
# 3.1 原始 Qwen3-8B（微调前基线）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 \
  --dtype auto --gpu-memory-utilization 0.9
```

- `--served-model-name Qwen3-8B` 必须与配置里的 `model` 字段**完全一致**（`client.py` 会把它放进请求体）。
- `--max-model-len` 至少 8192：单条 prompt 约 3.2k tokens，加上输出与余量。
- `api_key` 用 `EMPTY` 即可（vLLM 默认不校验）；若你的网关要鉴权，把配置里的 `api_key` 换成真实值，
  **不要提交到版本库**（`runs/*/config.json` 快照会自动打码成 `***`）。

```bash
# 3.2 微调后：建议先把 LoRA 合并导出成完整模型，再用同样的命令起一个新服务
#     （避免 vLLM 多 LoRA 名称与配置里 model 字段对不上）
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen3-8B \
  --adapter_name_or_path <你的 lora_adapter 目录> \
  --template qwen3 --finetuning_type lora \
  --export_dir ./Qwen3-8B-voice-nav
# 然后把 --model 指向 ./Qwen3-8B-voice-nav，--served-model-name 改成 Qwen3-8B-SFT，
# 并同步修改 config.qwen3-8b.json 的 model 字段（或另存一份 config.qwen3-8b-sft.json）。
```

连通性自检（不经过评测框架，先确认服务本身活着）：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"回复一个字：好"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

**Qwen3 的 thinking 开关**：配置通过 `extra_body.chat_template_kwargs.enable_thinking=false` 透传
（`client.py` 会把 `extra_body` 直接 `payload.update()`）。若你的服务端版本不接受该字段并返回 400，
就在服务端指定非 thinking 的 chat template，同时把配置里的 `extra_body` 整块删掉；
**不要**靠调大 `max_tokens` 绕过——那只会让思维链被判成「夹带解释」，且成本翻倍。

---

## 4. 配置文件 `config.qwen3-8b.json`

放在 `result/03_Qwen3-8B评测/`，用绝对路径 `--config` / `--source-config` 引用。
关键字段：

| 字段 | 值 | 为什么 |
| :--- | :--- | :--- |
| `models[0].id` | `qwen3-8b` | 运行目录 `<run>/qwen3-8b/`，也是 `--model` 选择依据 |
| `models[0].provider` | `openai_compat` | 走标准 OpenAI 兼容协议 |
| `models[0].base_url` | `http://127.0.0.1:8000/v1` | 只写到 `/v1`，代码自动拼 `/chat/completions` |
| `models[0].model` | `Qwen3-8B` | 必须等于 `--served-model-name` |
| `params.temperature` | `0` | 确定性请求才会命中磁盘缓存，重跑不重复烧算力 |
| `params.max_tokens` / `tasks[0].max_tokens` | `64` | 只输出一个意图名；开大会拉高夹带解释率 |
| `extra_body.chat_template_kwargs.enable_thinking` | `false` | 关闭 Qwen3 思维链 |
| `proxy` | `bypass` | 绕过环境里的 `HTTP_PROXY`，否则本地 127.0.0.1 可能被代理劫持 |
| `tasks[0].dedupe` | `false` | **必须**：432 条 prompt 前 200 字相同，通用去重会压成 1 条 |
| `tasks[0].num_fewshot` | `0` | 测试集不能进 few-shot |
| `rate_limit_qps` / `concurrency` | `2` / `4` | 按显存与吞吐调整；429 就调低 |
| `cost` | `0 / 0` | 自建端点无计费；报告显示「未配置单价」 |

> 基础任务参数仍来自 `config.professional.json`（一键脚本以它为模板、只在内存里替换 `models`，
> **不会写出含 api_key 的新配置文件**）。直接用 `python -m geval run --config` 时，
> 本配置已自带完整的 `tasks` / `report` / `cost`，可独立工作。

---

## 5. 完整命令

### 5.0 每次开 shell 先设好

```bash
cd "D:/工作/文件/模型评测/llm-general-eval/llm-general-eval"
CFG="D:/工作/文件/模型评测/result/03_Qwen3-8B评测/config.qwen3-8b.json"
```

所有命令都在此项目根执行；`data_root: datasets` 是相对 cwd 的。

### 5.1 准备与自检（都不调用被测模型）

```bash
python -m geval selftest                                  # 框架自检，改动代码后第一件事
python -m geval list-tasks                                # 确认 intent_routing 已注册
python -m geval count --config "$CFG"                     # 432 条 / 432 请求，不发请求
python -m geval preview --task intent_routing --config "$CFG"   # 看真正发给模型的 prompt
python scripts/run_professional_eval.py --source-config "$CFG" --model qwen3-8b --dry-run
python -m geval probe --config "$CFG"                     # 服务起来后再跑：1 条最小请求探活
```

`count --config "$CFG"` 实测：

```
intent-routing      gen      432    1    1    432
合计                         432         1    432
时长估算：总请求 432 × 时延(秒) ÷ 并发 4    例：单请求 0.5s → 约 0.9 分钟
```

一键脚本 `--dry-run` 实测：

```
模型 qwen3-8b (真实端点) ｜ 端点 http://127.0.0.1:8000/v1
数据集 432 条 / 498 候选路径 / 495 末级标签 ｜ 本次 8 条 ｜ 预计 8 次请求
并发/QPS 4/2.0 ｜ 测试集 sha256 7e902bf7... ｜ dry-run：未发送任何请求
```

### 5.2 推荐主路径：一键脚本（含探活 + 两份报告）

```bash
# ① 离线验证链路（不需要 Qwen3-8B 在线，mock 假模型）
python scripts/run_professional_eval.py --mock --full --run-name pro-selftest

# ② 真实端点小样接线：8 条，先确认 prompt / 输出格式 / thinking 都对了
python scripts/run_professional_eval.py --source-config "$CFG" --model qwen3-8b \
  --limit 8 --run-name qwen3-8b-wirecheck

# ③ 全量正式评测（可中断，重跑同一命令即续跑）
python scripts/run_professional_eval.py --source-config "$CFG" --model qwen3-8b \
  --full --resume --run-name qwen3-8b-full

# ④ 有失败样本时补跑（--resume 会把 error 行当已完成，必须加 --retry-failed）
python scripts/run_professional_eval.py --source-config "$CFG" --model qwen3-8b \
  --full --resume --retry-failed --run-name qwen3-8b-full

# ⑤ 只重出报告 / 换单价（不重跑模型，不花一分钱）
python scripts/professional_report.py runs/qwen3-8b-full
python scripts/professional_report.py runs/qwen3-8b-full --cost-in <输入单价> --cost-out <输出单价>
```

一键脚本会自动 `probe_models()`（发 1 条最小请求），探活失败立即返回 2，不会浪费 432 次请求。
其他参数：`--qps 2`、`--concurrency 8`、`--no-probe`、`--cost-in/--cost-out`；
完整列表 `python scripts/run_professional_eval.py --help`。

> `--source-config` 只用来取真实端点；基础配置仍是 `config.professional.json`。
> `--model qwen3-8b` 是按 `models[].id` 选模型，可省略（默认第一个非 mock）。
> **`--run-name` 必须给不同值**：换模型 / 换 prompt / 微调前后对比若复用同一目录，会命中缓存，数字看起来"没变化"。

### 5.3 手动等价路径（不用一键脚本）

```bash
python -m geval doctor --config "$CFG" --no-selftest
python -m geval probe  --config "$CFG"
python -m geval run    --config "$CFG" --limit 8 --run-name qwen3-8b-wirecheck
python -m geval run    --config "$CFG" --resume                    # run_name 取配置里的 geval-qwen3-8b-intent
python -m geval run    --config "$CFG" --resume --retry-failed
python -m geval report --run runs/geval-qwen3-8b-intent           # 通用报告
python scripts/professional_report.py runs/geval-qwen3-8b-intent   # ★ 专业专项报告
python scripts/diagnose.py runs/geval-qwen3-8b-intent              # 逐样本归因
```

`python -m geval run` 是**唯一会调用模型**的命令，其余都是准备 / 检查 / 报告。

### 5.4 微调前 vs 微调后 A/B

```bash
# 基线：原始 Qwen3-8B 服务（model=Qwen3-8B）
python scripts/run_professional_eval.py --source-config "$CFG" --model qwen3-8b \
  --full --resume --run-name qwen3-8b-base

# 微调后：另存 config.qwen3-8b-sft.json，把 model 改成 Qwen3-8B-SFT、id 改成 qwen3-8b-sft
SFT_CFG="D:/工作/文件/模型评测/result/03_Qwen3-8B评测/config.qwen3-8b-sft.json"
python scripts/run_professional_eval.py --source-config "$SFT_CFG" --model qwen3-8b-sft \
  --full --resume --run-name qwen3-8b-sft

# 两份报告并排看：runs/qwen3-8b-base/professional_report.md 与 runs/qwen3-8b-sft/professional_report.md
```

A/B 的有效前提：两边用**同一套冻结测试集**（sha256 一致）、同样的 `temperature/max_tokens/thinking` 设置。
若微调用了 `sft_train_with_eval.jsonl`，这组 A/B 只能看训练拟合度，不能作为效果结论。

---

## 6. 输出与报告解读

```
runs/qwen3-8b-full/
├── manifest.json                  版本 / Python / seed / 模型（不含 key）/ 任务
├── config.json                    生效配置快照（api_key 打码）
├── summary.json                   模型×任务总表 + client_stats + 错误列表
├── report.md / report.html        通用报告
├── professional_report.md         ★ 专业专项报告
├── professional_metrics.json      ★ 机器可读指标（可接看板）
└── qwen3-8b/
    ├── intent-routing.jsonl       ★ 逐样本：pred/gold/correct/strategy/
    │                               prompt_tokens/completion_tokens/total_tokens/
    │                               latency_ms/request_wall_ms/cached/finish_reason/extras
    └── intent-routing.summary.json
```

逐样本逐条 flush，中断后 `wc -l` 即可看进度，`--resume` 靠 `id` 跳过已完成样本。

读报告的顺序：

1. **意图准确率 vs 小模型基线 85.88%** —— 看配对增益和 95% CI，别只看单点差值；
2. **McNemar p** —— p≥0.05 一律写"不显著"，不能说"提升/下降"；
3. **幻觉意图率** —— 8B 在 495 个标签上容易编造树外名称，这项比答错更该关注；
4. **解析失败率 / 夹带解释率 / 截断率** —— 高就先查 thinking 是否真的关掉、`max_tokens` 是否够、prompt 有没有被改；
5. **业务/通用分层准确率 + Top 混淆对** —— 整体分会被「人工服务」主导；
6. **平均 tokens/样本、p95/p99、tokens/s** —— 判断 8B 相对小模型是否划算。

---

## 7. 验收口径

沿用 `docs/EVAL_METHODOLOGY.md` §7.5，建议提前定死再跑：

- 专业能力：意图准确率相对小模型基线的提升**统计显著**（McNemar p<0.05 且配对 bootstrap CI 不跨 0）；
- 通用能力：`ceval` / `ifeval` 等退化 ≤ 2 个点且不显著（用 `config.json` 另跑阶段 1）；
- 工程可用性：幻觉意图率、解析失败率、P99 时延、单样本 token 成本在可接受区间；
- 只有以上同时满足，才支持「用微调后的 Qwen3-8B 替换现有方案」的结论。

**表述纪律**：因为覆盖率只有 10.64%，对外一律写「在 432 条 / 53 个高频意图上的专业准确率」，
不要写「全意图树准确率」或「语音导航整体能力」。

---

## 8. 故障速查

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| 探活失败 / `Connection refused` | vLLM 没起、端口不对、`base_url` 少了 `/v1` | 先跑 §3 的 curl；核对 `--host/--port` 与配置 |
| 404 / 模型名不匹配 | `model` 字段 ≠ `--served-model-name` | 两边改成完全一致 |
| 401 / 鉴权失败 | `api_key` 与网关不一致 | vLLM 用 `EMPTY`；计费网关填真实 key（不要提交） |
| 输出为空、整片 0 分 | Qwen3 thinking 没关，64 token 被思维链吃光 | `extra_body.chat_template_kwargs.enable_thinking=false`；看明细 `finish_reason` |
| 大量夹带解释 / 解析失败 | 服务端忽略 `chat_template_kwargs`，或 `max_tokens` 过大 | 服务端指定非 thinking 模板；`max_tokens` 回到 64 |
| 请求体报 400 不认识 `chat_template_kwargs` | vLLM 版本较旧 | 删掉配置里的 `extra_body`，在服务端用非 thinking 模板 |
| 数据集只剩 1 条 | 通用去重按 prompt 前 200 字去重 | `dedupe: false`（任务类与配置双保险，别删） |
| 标签只有十几个、幻觉率异常 | `labels.json` 缺失，用了兜底标签 | `python scripts/prepare_professional_dataset.py` |
| 超时 / 502 | 环境变量代理劫持了 127.0.0.1 | `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy`；`proxy: bypass` |
| 大面积 429 / OOM | 并发超服务端能力 | 调低 `concurrency` / `rate_limit_qps`，再 `--resume --retry-failed` |
| 重跑后错误率不降 | `--resume` 把 error 行当成已完成 | 加 `--retry-failed` |
| 微调后数字没变 | 复用同一 run 目录 / 命中缓存 | 换 `--run-name`，或清 `.cache/geval/<model>/` |
| `ModuleNotFoundError: geval` | 不在项目根 | `cd` 到含 `geval/` 的那一层 |

---

## 9. 已知限制

- **未实现 TTFT / 排队耗时拆分**：框架是非流式单请求，只记端到端时延；TTFT 需要流式改造。
- **槽位类指标未实现**：当前任务是从意图树选名称，没有槽位标注；`EVAL_METHODOLOGY.md` 里的
  槽位 F1 / 幻觉槽位 / 拒识率等需另建数据与任务。
- **测试集覆盖偏低**：445 个节点零命中，二三级业务菜单覆盖尤其少；补标后建议做第二套评测集，
  用 `scripts/prepare_professional_dataset.py --input <新集> --out datasets/intent_routing_v2` 转换。
- **成本**：自建端点没有单价，报告显示「未配置单价」；若走计费网关再填 `cost`。
- 本文只给出**配置与命令**，没有连接真实 Qwen3-8B 端点执行正式评测；所有真实数字需运行 §5.2 ③ 后产生。
