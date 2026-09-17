# 数据集目录说明

geval 只认 **JSONL**，一条样本一行。目录结构约定：

```
datasets/
  <任务名>/
    test.jsonl      # 评测集（必需）
    dev.jsonl       # few-shot 示例（可选，缺失时会在评测集里取并自动剔除，避免示例泄漏）
    schema.json     # 仅 json_schema 任务使用：意图枚举 / 槽位枚举 / JSON Schema
```

任务名从 `geval/tasks.py::TASK_REGISTRY` 取：`ceval` `cmmlu` `mmlu` `mmlu_pro` `arc` `agieval`
`gsm8k` `math` `bbh` `ifeval` `json_schema`。

## 字段格式

geval 的归一化层能识别多种上游写法，**不要求手工整理**。最通用的写法：

```json
{"id": "q-0001", "question": "水的化学式是什么？", "choices": ["H2O", "CO2", "O2", "NaCl"], "answer": "A", "subject": "化学"}
```

兼容的等价写法：

| 字段 | 可用写法 |
| :--- | :--- |
| 题面 | `question` / `Question` / `query` / `input` / `prompt` / `problem` |
| 选项 | `choices` / `options`（列表或 dict），或 `A` `B` `C` `D` 平铺字段，或 `"A. xx\nB. yy"` 字符串 |
| 答案 | `answer` / `Answer` / `target` / `label`；允许 **索引**（`0`~`3`）、**字母**（`"A"`）、**选项原文**、**数值** |
| 分组 | `subject` / `category` / `topic`（用于报告里的短板定位） |
| 数值题 | 无 `choices`，`answer` 为数值或 `"... #### 42"` |

数值答案会被归一化后做**带容差比较**，`1,234` / `1234.0` / `$1234` 视为相等。

## 准备方式

**方式一：联网机器转公开基准（推荐）**

```bash
pip install datasets
python scripts/prepare_datasets.py --list
python scripts/prepare_datasets.py --task ceval --out datasets/ceval --split val
python scripts/prepare_datasets.py --task cmmlu --out datasets/cmmlu --split test
python scripts/prepare_datasets.py --task gsm8k --out datasets/gsm8k --split test
python scripts/prepare_datasets.py --task bbh   --out datasets/bbh   --split test
python scripts/prepare_datasets.py --task ifeval --out datasets/ifeval --split train
```

生成的只是 JSONL，把整个 `datasets/` 目录拷进内网即可，内网机器不需要 `datasets` 库。

**方式二：用自己的业务数据**

直接把意图识别的标注数据导出成 JSONL。`json_schema` 任务的答案是一个 JSON 对象：

```json
{"id": "intent-0001", "question": "用户说：「帮我查一下这张卡的余额」。请输出结构化结果。",
 "answer": {"intent": "查询余额", "slots": {}}, "subject": "查询余额"}
```

同时在该目录放一个 `schema.json` 声明意图与槽位枚举（geval 会自动加载，见 `datasets/json_schema/schema.json`）。

> **重要：冻结纪律。**
> 阶段 0 就要把数据切分成 train / val / test，并给 `test.jsonl` 记下 sha256。
> 报告的"复现信息"一节会自动记录数据集指纹，方便你确认两次运行用的是同一份数据。
> 任何进入训练集的数据，都不能再出现在 test.jsonl 里——否则微调前后的对比没有意义。

## 成本控制：分层抽样

全量跑 `C-Eval(1346) + CMMLU(11582) + MMLU(14042) + GSM8K(1319)` 请求量很大。
用 `--limit` 按 `subject` 分层等比抽样，几百条就能得到与全量高度相关的**排序**结论：

```bash
python -m geval run --config config.json --limit 40        # 冒烟：先看排序
python -m geval run --config config.json                   # 全量：出正式数字
```

抽样运行的报告中会自动带上「不可与公开榜单数字直接对齐」的提示，避免误引用。
