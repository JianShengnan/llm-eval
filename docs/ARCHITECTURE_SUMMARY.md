# geval 架构概览与快速启动指南

> **场景技能**: `system-modeler` + `graphviz`
> **范围**: 当前状态 (current-state)，L2-L3 视图
> **证据来源**: 源码目录结构、`README.md`、`config.example.json`、`CODE_MANIFEST.txt`
> **置信度**: high — 所有节点和关系均来自源码和文档直接证据

---

## 1 项目定位

**geval** 是一个零第三方依赖的 Python 大模型通用能力评测框架，服务于银行客服意图识别项目的**技术选型决策**：

> 用「小模型 (7B/8B) + 全参数微调」替代现网 Qwen3-32B，在通用能力可接受的退化范围内，换取准确率与时延收益。

框架不只输出准确率，还强制输出**置信区间、配对显著性检验、时延分布和解析失败率**，让模型间差异的判定有据可依。

---

## 2 仓库结构总览

```
llm-general-eval/
├── geval/                    # 核心框架（14 个模块，纯标准库）
│   ├── __main__.py           # CLI 入口：6 个子命令
│   ├── runner.py             # 编排引擎：并发执行、断点续跑、指纹
│   ├── client.py             # OpenAI 兼容客户端（并发安全、重试、缓存）
│   ├── http.py               # urllib 传输层（代理策略、退避重试）
│   ├── datasets.py           # 数据加载：多格式归一化、去重、分层抽样
│   ├── prompts.py            # 3 种 prompt 协议 + 各任务模板
│   ├── answers.py            # 答案抽取（选项/数值/JSON）+ 思维链剥离
│   ├── metrics.py            # 统计：Wilson CI、bootstrap、McNemar、时延
│   ├── tasks.py              # 任务注册表（11 个内置基准）
│   ├── ifeval.py             # 24 类指令遵循校验器
│   ├── report.py             # 报告生成：Markdown + 自包含 HTML（内联 SVG）
│   └── mock.py               # 确定性 mock 模型（离线自检用）
├── datasets/                 # 评测数据集（JSONL 格式）
│   └── json_schema/          # 意图+槽位示例（含 schema.json）
├── scripts/                  # 运维与数据准备脚本
├── docs/                     # 文档（运行手册、方法论、覆盖度）
├── config.example.json       # 配置模板（5 个模型已预定义）
├── config.smoke.json         # 干跑配置（mock 模型，不连真实服务）
└── requirements.txt          # 声明零第三方依赖
```

---

## 3 架构图

### 3.1 系统架构（模块关系）

见 [`docs/architecture-overview.dot`](architecture-overview.dot)

核心分层：

| 层 | 模块 | 职责 |
|---|---|---|
| **CLI 层** | `__main__.py` | 解析命令、加载配置、分发到内部模块 |
| **编排层** | `runner.py`, `tasks.py` | 并发调度、断点续跑、任务注册与解析 |
| **处理管线** | `datasets.py` → `prompts.py` → `client.py` → `answers.py` | 数据加载 → prompt 组装 → 模型调用 → 答案抽取 |
| **分析输出** | `metrics.py`, `ifeval.py`, `report.py` | 统计计算、指令校验、报告生成 |
| **传输层** | `http.py` | urllib 请求、代理策略、退避重试 |

### 3.2 数据流（从配置到报告）

见 [`docs/dataflow.dot`](dataflow.dot)

```
config.json + datasets/*.jsonl
    → 加载归一化 → 分层抽样 → prompt 组装
    → 并发请求 5 个模型端点
    → 答案抽取 → 统计打分
    → manifest + details + summary
    → report.html + report.md
```

---

## 4 快速启动（5 步）

### 第 1 步：离线自检（验证框架本身）

```bash
python -m geval selftest
```

不需要模型、不联网，用 mock 模型跑通整条链路，约 1 秒。

### 第 2 步：了解内置任务

```bash
python -m geval list-tasks          # 查看 11 个内置基准
python -m geval preview --task ceval       # 看实际发给模型的 prompt
python -m geval preview --task json_schema  # 与意图识别最贴近的任务
```

### 第 3 步：干跑验证（mock 模型 + 合成数据）

```bash
python -m geval run --config config.smoke.json --smoke
```

验证配置解析、数据加载、报告生成的完整链路，不连真实模型。

### 第 4 步：连接真实模型

```bash
cp config.example.json config.json
# 编辑 config.json，填入实际的模型 base_url 和 model 名称

python -m geval probe --config config.json           # 端点探活
python -m geval run --config config.json --smoke --limit 8 \
    --models pangu7b-pre,qwen8b-pre,qwen3-32b         # 先验证接线
```

### 第 5 步：正式评测

```bash
# 阶段 1：微调前基线（3 个模型）
python -m geval run --config config.json \
    --models pangu7b-pre,qwen8b-pre,qwen3-32b --resume

# 阶段 1.5：微调后全量（5 个模型，自动产出 pre→post 对比）
python -m geval run --config config.json --resume

# 生成报告
python -m geval report --run runs/geval-general-5objects
```

报告产物在 `runs/<run_name>/` 下：`report.html`（自包含，可直接发群）和 `report.md`。

---

## 5 关键设计决策

| 决策 | 理由 |
|---|---|
| **零第三方依赖** | 内网 ARM64 离线机器「拷目录即可运行」，不涉及 pip |
| **3 种 prompt 协议可切换** | 回答「小模型该用哪种问法」——7B/8B 常用 `direct` 更优，32B 用 `answer_line` |
| **单选题 3 种打分模式** | `gen`(生成式) / `logprob`(首token) / `loglikelihood`(与 lm-eval 对齐) |
| **统计量是硬性输出** | Wilson CI + bootstrap + McNemar，只有「显著」差距才作为选型依据 |
| **复现指纹** | 每次运行写 `manifest.json`，数据或 prompt 变更后数字不可比时能立刻发现 |
| **解析失败率单独统计** | 把「模型答错」和「抽取器没读懂」分开，避免误判 |

---

## 6 评测维度与任务映射

| 能力维度 | 与意图识别的关系 | 对应任务 |
|---|---|---|
| 中文语言理解 | 中文 query 语义边界判断 | `ceval` `cmmlu` |
| 通用知识 | 领域词、实体识别 | `mmlu` `mmlu_pro` `arc` |
| 逻辑与数学推理 | 多意图、条件意图 | `bbh` `gsm8k` `math` |
| **指令遵循** | 格式约束守不守得住 | `ifeval` |
| **结构化输出** | 直接产出 intent + slots JSON | `json_schema` |

加粗的两项是**性价比最高的两项**——生产环境中意图识别的失败绝大多数不是"不懂知识"，而是输出不可解析和边界判断漂移。

---

## 7 被测对象（5 个模型）

| 模型 ID | 说明 | 阶段 |
|---|---|---|
| `pangu7b-pre` | 盘古7B 微调前 | 阶段 1 |
| `pangu7b-post` | 盘古7B 微调后 | 阶段 1.5 |
| `qwen8b-pre` | Qwen8B 微调前 | 阶段 1 |
| `qwen8b-post` | Qwen8B 微调后 | 阶段 1.5 |
| `qwen3-32b` | Qwen3-32B 现网基线 | 阶段 1 |

---

## 8 文档索引

| 文档 | 何时阅读 |
|---|---|
| [`README.md`](../README.md) | 完整使用说明、配置字段、报告解读 |
| [`docs/INTRANET_RUNBOOK.md`](INTRANET_RUNBOOK.md) | 内网部署、五步执行、故障排查 |
| [`docs/EVAL_METHODOLOGY.md`](EVAL_METHODOLOGY.md) | 与开源框架的关系、可比性边界 |
| [`docs/COVERAGE.md`](COVERAGE.md) | 五项需求覆盖度对照 |
| [`docs/OFFLINE_DEPLOY.md`](OFFLINE_DEPLOY.md) | lm-eval/EvalScope 离线部署 |
| [`datasets/README.md`](../datasets/README.md) | 数据格式与准备方式 |

---

## 9 环境要求

- **Python 3.9+**（框架本身零第三方依赖）
- 仅 `scripts/prepare_datasets.py` 需要 `pip install datasets`（在联网机器上运行一次）
- 内网 ARM64 可直接拷贝整个目录运行

---

*生成时间: 2026-09-17 | 技能: system-modeler + graphviz | 置信度: high*
