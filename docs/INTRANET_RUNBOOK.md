# 通用能力评测 · 行内运行手册

> 适用：内网 Linux（ARM64 / x86_64 均可）离线环境
> 框架：`geval`（本目录），**运行时零第三方依赖，无需 pip、无需外网**
> 被测对象：当前实际配置为单模型 `llm-chat-core`（6 个任务）。
> 5 对象模板（盘古7B / Qwen8B 的 pre、post + Qwen3-32B）见 `config.example.json`，属于可选流程。

---

## 0. 一分钟总览

| 项目 | 说明 |
| :--- | :--- |
| **做什么** | 对配置中的模型通道跑统一题集，产出可比的能力矩阵与时延数据 |
| **产出什么** | `report.html`（自包含单文件，可直接发群）、`report.md`、逐样本明细 JSONL、`summary.json` |
| **要不要外网** | **不要。** 框架本身零依赖；数据集需事先转成 JSONL 拷进来 |
| **要不要 pip** | **不要**（除非额外装 lm-eval，见附录 A） |
| **多久** | 自检约 1 秒；抽样评测几分钟；全量按 `geval count` 预估 |
| **失败重试** | 支持断点续跑，中断后加 `--resume` 继续，不重复消耗 |

---

## 1. 前置条件核查

在目标机上逐条确认：

| # | 检查项 | 要求 | 检查命令 |
| :- | :--- | :--- | :--- |
| 1 | Python 版本 | ≥ 3.9（实测 3.13；3.11 亦可） | `python3 -V` |
| 2 | 工作目录 | 已拷贝 `llm-general-eval/` 整个目录 | `ls` |
| 3 | 写权限 | 工作目录可写（要生成 `runs/`、`.cache/`） | `touch .wtest && rm .wtest` |
| 4 | 模型端点可达 | 配置中的 OpenAI 兼容端点（当前 1 个） | `curl -s <base_url>/models` |
| 5 | **环境变量无代理** | `HTTP_PROXY` 必须为空 | `env | grep -i proxy` |
| 6 | 磁盘空间 | 建议 ≥ 2 GB（明细 JSONL + 缓存） | `df -h .` |

第 5 条是内网部署最高频的坑：服务器上常配了指向外网代理的 `HTTP_PROXY`，
发往内网模型的请求会被送到代理，表现为**超时或 502**。
geval 的 `proxy` 默认已是 `bypass`（显式清空代理），但仍然建议确认环境干净：

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
```

---

## 2. 首次部署

### 2.0 一条命令验证全部前置条件（推荐先做这个）

```bash
python3 -m geval doctor --config config.json
```

它会按「阻塞程度」逐项检查，并对每个问题给出**可直接执行**的修复动作：

```
── 1. 运行环境 ──────────────────────────────────────────────
[OK] Python 3.11.9（需 ≥ 3.9）
[OK] 在项目根目录执行（geval 可正常导入）
── 2. 框架自检 ──────────────────────────────────────────────
[OK] 运行时依赖：无（零第三方依赖），标准库模块 25 个   ← 数量随 Python 版本略有差异
[OK] 离线自检通过（7 个任务、抽取层、acc/acc_norm、退化判定、续跑、报告生成）
── 3. 配置与模型端点 ────────────────────────────────────────
[OK] 配置已加载 config.json：1 个模型、6 个任务条目
[OK] 1 个真实模型都填了 base_url 与 model
[警告] 检测到代理环境变量 ['HTTP_PROXY']
       → unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
── 4. 数据集 ────────────────────────────────────────────────
[OK] json_schema：16 条  ←  datasets/json_schema/test.jsonl
[OK] 全部 6 个任务的数据集就绪
── 5. lm-eval 离线锚点（可选） ──────────────────────────────
[OK] ./venv：Python 3.11.9，lm-eval 0.4.13，datasets 已装   ← 仅当项目内已建 venv
```

**退出码 0 = 无阻塞，可以开跑；1 = 有阻塞项，先按提示修复。**
加 `--probe` 会顺带做端点探活（会发真实请求）。

常用参数：

| 参数 | 说明 |
| :--- | :--- |
| `--config config.json` | 指定配置（默认读同路径） |
| `--probe` | 顺带探活模型端点 |
| `--venv /path/to/venv` | 手动指定虚拟环境（默认自动发现） |
| `--no-selftest` | 跳过离线自检，更快 |

### 2.0.1 关于虚拟环境（如果你已经在项目里建了 venv）

**两件事要分清：**

1. **`geval` 本身零第三方依赖，不需要任何 venv。** 用系统 `python3` 或 venv 里的 python 都能跑。
   所以「geval 跑不起来」和「venv 建得对不对」没有关系。
2. **`lm-eval` 才需要 venv**，因为它是带一堆依赖的第三方框架。

因此推荐的用法是：**用 venv 里的 python 跑所有命令**，这样两个工具共用一个解释器，
不会出现「lm-eval 装在这个 python 里、geval 跑在另一个 python 里」的混淆：

```bash
# 假设你在项目根目录建了 venv（名字随意，doctor 靠 pyvenv.cfg 自动识别）
./venv/bin/python -m geval doctor --config config.json
./venv/bin/python -m geval selftest
./venv/bin/python -m geval run --config config.json --smoke --limit 8
```

Linux 是 `venv/bin/python`，Windows 是 `venv\Scripts\python.exe`。

> **注意**：`python3 -m geval` 要求当前目录是**项目根目录**（因为 `geval` 是项目里的包）。
> 换目录执行会报 `No module named geval`。若必须换目录，设 `PYTHONPATH`：
> ```bash
> PYTHONPATH=/opt/llm-general-eval python3 -m geval doctor
> ```

### 2.1 拷贝并进入目录

```bash
cd /opt/llm-general-eval        # 换成你实际放置的路径
```

### 2.2 环境自检（一条命令，约 1 秒）

```bash
python3 scripts/verify_deps.py
```

**预期输出**（关键行）：

```
[运行时] 第三方依赖：无 -> 内网离线可直接运行
[OK] 运行时依赖核查通过
```

如果这里报出第三方依赖，说明拷贝不完整，重新拷一遍整个目录。

### 2.3 框架自检（离线，不发任何请求）

```bash
python3 -m geval selftest
```

**预期输出**：

```
[PASS] 全部自检项通过：
  · 7 个任务全部产出结果，强/弱 mock 的分数序关系正确
  · 解析失败率统计、抽取策略分布均可用
  · 结构化输出的 4 项子指标与 IFEval 覆盖统计正常
  · acc / acc_norm 续写似然解析口径正确
  · 微调影响评估的退化判定方向正确
  · 断点续跑生效，HTML / Markdown 报告已生成
```

**只要这一步不是全 PASS，就不要往下走。** 这 1 秒钟能排掉绝大多数装配问题。
可以在浏览器里打开报告确认渲染正常：

```
.cache/geval-selftest/selftest/report.html
```

### 2.4 后续更新（拿到新版本代码时）

**不用重传 133 MB 的全量包。** 框架源码只有约 150 KB，交付时会给一个增量包
`geval-src-<日期>.tar.gz`，里面刻意**排除了 `datasets/` 和 `wheels/`**，
覆盖上去不会碰到你准备的数据集、`config.json`、`venv/` 和 `runs/`。

**第一步：确认到底需不需要更新。** 增量包里带了一份 `CODE_MANIFEST.txt`
（每个源码文件的 sha256），在内网项目根目录执行：

```bash
python3 scripts/verify_install.py
```

三种结果：

| 输出 | 含义 | 动作 |
| :--- | :--- | :--- |
| `[OK] 框架代码全部一致` | 内网已是最新 | 无需更新 |
| `[需更新] N 个框架文件…` | 内网是旧版本 | 按下述步骤覆盖 |
| `[缺失] 某个文件` | 旧版本里还没有这个文件 | 同上，覆盖即可补上 |

退出码 `0` = 可继续跑评测，`1` = 建议先更新。`--quiet` 只打印不一致项。

> 首次升级时内网还没有 `verify_install.py`。可以先把增量包里的
> `scripts/verify_install.py` 与 `CODE_MANIFEST.txt` 两个文件拷进去再跑；
> 或者直接跳到第二步覆盖，覆盖后再跑一次核对（等价，且更省事）。

**第二步：覆盖。** 先删后拷 —— `geval/`、`scripts/`、`docs/` 是纯代码目录，
不含你的数据，删掉重建可以顺带清掉旧版本遗留的模块文件（合并覆盖做不到这点）：

```bash
cd /path/to/llm-general-eval
tar xzf /tmp/geval-src-20260916.tar.gz -C /tmp

rm -rf geval scripts docs
cp -r /tmp/geval-src-20260916/geval /tmp/geval-src-20260916/scripts /tmp/geval-src-20260916/docs .
cp /tmp/geval-src-20260916/{README.md,requirements.txt,config.example.json,config.smoke.json,.gitignore,CODE_MANIFEST.txt} .
```

**第三步：复验。**

```bash
python3 scripts/verify_install.py    # 期望 [OK] 框架代码全部一致
python3 -m geval selftest            # 期望全 PASS
```

**不要覆盖的东西**（增量包已排除，但仍请留意）：

| 路径 | 原因 |
| :--- | :--- |
| `datasets/` | 你准备的数据集，可能在联网机器上辛苦转出来的 |
| `config.json` | 你填的真实模型地址与鉴权信息（模板是 `config.example.json`） |
| `venv/` `.venv/` | 装好 lm-eval 的虚拟环境 |
| `runs/` | 历史评测结果，`--resume` 依赖它 |

---

## 3. 配置

### 3.1 确认当前配置（不要直接覆盖）

当前仓库已带 `config.json`（单模型 `llm-chat-core`、6 个任务），直接改它的模型段即可。
**不要**用 `config.example.json` 覆盖它 —— 那是一份 5 对象模板；只有启用模板流程时才复制：

```bash
cp config.example.json config.5objects.json
```

> ⚠️ `config.json` 内含真实网关地址与 `api_key`，已在 `.gitignore` 中排除，**不要提交**。
> 外发或打包前先确认增量包里没有它（`scripts/pack_bundle.py` 已排除）。

### 3.2 必须修改的几处

打开 `config.json`，只需改模型段（其余保持默认）：

```json
{
  "id": "llm-chat-core",
  "label": "llm-chat-core",
  "base_url": "http://<实际地址>/v1",   ← 改成实际地址与端口
  "api_key": "EMPTY",                   ← 服务端未开鉴权就保持 EMPTY
  "model": "llm-chat-core"              ← 必须与服务端加载的模型名完全一致
}
```

> `model` 字段写错是最常见的失败原因（报 404 model not found）。
> 用 `python3 -m geval probe --config config.json` 会打印服务端**实际返回的模型名**，照着改即可。

Qwen3 系列需要关掉思考模式，模板里已配好：

```json
"extra_body": { "chat_template_kwargs": { "enable_thinking": false } }
```

### 3.3 常用参数

| 参数 | 默认 | 说明 |
| :--- | :--- | :--- |
| `concurrency` | 8 | 并发请求数。先跑 `probe` 看单请求时延再定；服务端报 503 就调小 |
| `request_timeout_s` | 180 | 单请求超时 |
| `retries` | 3 | 传输层重试次数（429/5xx/超时自动退避重试） |
| `proxy` | `bypass` | **保持 bypass**；内网不要跟随环境变量 |
| `seed` | 42 | 抽样随机种子，影响分层抽样的选样。**多次运行必须一致** |
| `rate_limit_qps` | 不设 | 单进程自我限流；网关/现网通道建议 1~2，否则多进程叠加会打爆配额 |
| `dimension_weights` | 不设 | 综合分维度权重；需配合任务条目的 `dimension`。当前 `config.json` 已配置 |
| `report.baseline` | 5 对象模板为 `qwen3-32b` | 配对显著性检验的对照模型；单模型时该章节自动跳过 |
| `report.pairs` | 模板为两对 pre/post | 微调影响评估的配对，无 post 模型时会自动跳过 |

---

## 4. 数据集准备

**这是唯一需要外部输入的部分。** 三条路径，按你的情况选一条：

### 路径 A：先验证链路（最快，5 分钟）

仓库里已带一套意图+槽位示例数据集，可以立刻跑通：

```bash
python3 -m geval run --config config.json --smoke --limit 8
```

`--smoke` 会允许缺失的数据集回退到内置合成数据，**目的只是验证端到端接线**
（地址、鉴权、字段、解析、报告），不消耗真实题目。

### 路径 B：准备公开基准（推荐，用于正式评测）

在**能上外网的机器**上执行：

```bash
pip install datasets
python3 scripts/prepare_datasets.py --list
python3 scripts/prepare_datasets.py --task ceval  --out datasets/ceval  --split val
python3 scripts/prepare_datasets.py --task cmmlu  --out datasets/cmmlu  --split test
python3 scripts/prepare_datasets.py --task mmlu   --out datasets/mmlu   --split test
python3 scripts/prepare_datasets.py --task gsm8k  --out datasets/gsm8k  --split test
python3 scripts/prepare_datasets.py --task bbh    --out datasets/bbh    --split test
python3 scripts/prepare_datasets.py --task ifeval --out datasets/ifeval --split train
```

然后把整个 `datasets/` 目录拷进内网。内网机器**不需要** `datasets` 库，只要能读 JSONL。

### 路径 C：已有自己的评测数据

整理成 JSONL，放到 `datasets/<任务名>/test.jsonl`：

```json
{"id": "q-0001", "question": "水的化学式是什么？", "choices": ["H2O", "CO2", "O2", "NaCl"], "answer": "A", "subject": "化学"}
```

字段兼容多种写法（索引/字母/选项原文均可作为答案），详见 `datasets/README.md`。

> **冻结纪律**：正式评测前先把 test 集定下来并记录 sha256。
> 报告最后一节会自动记录数据集指纹，用于确认两次运行用的是同一份数据。
> **任何进入训练的数据都不能出现在 test 里。**

---

## 5. 执行（分步进行）

每一步都产出可看的报告，**不要跳过前面的步骤直接跑全量**。

### 5.0 只有 1 个模型通道时：先验证流程联通性

**完全可以。** 框架与模型数量解耦，单模型跑全部任务没有任何问题。
报告会自动跳过「配对显著性检验」（需要 ≥2 个模型）和「微调影响评估」（需要 pre/post 成对），
并把跳过原因写明，不会出现空表或报错。

建议按三个梯度依次走：

**梯度 1 · 合成数据 + 全任务（不消耗真实题目）**

```bash
python3 -m geval doctor --config config.json --no-selftest    # 先看有无阻塞
python3 -m geval probe --config config.json
python3 -m geval run --config config.json --smoke --limit 8 \
    --run-name single-check
```

这一步验证的是**工具链**：端点可达、鉴权正确、`model` 名对得上、
所有任务的 prompt 能正常下发、答案能被抽取、报告能生成。

要看的两个信号：

1. 没有 `[error]` 行
2. 报告「解析质量」一节里，**解析失败率接近 0**

**梯度 2 · 真实数据 + 单个任务（真实请求，成本极低）**

仓库里 `datasets/json_schema/` 带了 16 条真实风格的意图+槽位样本，开箱即用：

```bash
python3 -m geval run --config config.json \
    --tasks json_schema --run-name single-check
```

16 次请求就能看到结构化输出的四层指标（JSON 可解析率 / Schema 通过率 /
意图准确率 / 槽位完全一致率）是否正常——这也是与业务最贴近的一项。

**梯度 3 · 真实公开基准（需先按 §4 路径 B 准备 JSONL）**

```bash
python3 -m geval count --config config.json --p50-ms <实测单请求毫秒>
python3 -m geval run --config config.json --limit 40 --resume
```

> **这一步的价值是验证链路，不是评价模型。**
> 单通道跑出来的数字**不要用来做选型结论** —— 它只证明工具链没问题。
> 等多个通道（或启用 5 对象模板）就绪后，用同一套题目重跑，那时的对比才有意义。
>
> 另外务必加 `--run-name single-check` 单独存放，
> 避免以后正式评测时与这次的结果混在一起（`--resume` 是按样本 id 跳过的，
> 混在一起会造成样本集不一致）。

### 5.1 第 0 步 · 离线自检（不发请求）

```bash
python3 -m geval selftest
```

### 5.2 第 1 步 · 端点探活

```bash
python3 -m geval probe --config config.json
```

**预期输出**：

```
[OK ] llm-chat-core  http://<实际地址>/v1
        时延 312.4 ms ｜ 服务端模型名 llm-chat-core ｜ 样例输出 '好'

1/1 个模型可用
```

失败时的处理见第 7 节。

> 若改用 5 对象模板（`config.5objects.json`），阶段 1 期间两个「微调后」端点尚不存在，
> 探活会失败 —— 这是**正常的**，只要阶段 1 要用的三个是 OK 即可。

### 5.3 第 2 步 · 冒烟（合成数据，验证接线）

```bash
python3 -m geval run --config config.json --smoke --limit 8
```

看两件事：
1. 有没有 `[error]` 行
2. 报告「解析质量」一节里，**解析失败率是否接近 0**

解析失败率高说明 prompt 协议或抽取规则需要调整，此时**不要**继续往下跑。

### 5.4 第 3 步 · 用量预估（不发请求）

```bash
python3 -m geval count --config config.json --p50-ms <实测单请求毫秒>
```

**预期输出**（当前单模型 6 任务，实测值）：

```
任务                 打分模式              样本数    请求/样本   模型数      总请求
ceval-fewshot       gen                     1337          1       1        1337
cmmlu               gen                    11408          1       1       11408
gsm8k               gen                     1319          1       1        1319
bbh                 gen                     5268          1       1        5268
ifeval              gen                      541          1       1         541
json_schema         gen                       16          1       1          16

时长估算：用实测单请求时延代入 —— 总请求 19889 × 时延(秒) ÷ 并发 8
          例：单请求 0.5 s、并发 8 → 约 20.7 分钟
```

> **样本数与 `doctor` 为什么不一样？** `doctor` 报的是 JSONL 原始条数
> （ceval 1346 / cmmlu 11582 / bbh 6511），`count` 报的是**题面去重后**实际要发的条数
> （ceval 1337 / cmmlu 11408 / bbh 5268）。评测以 `count` 为准。

先看这个数字再决定跑不跑全量。

### 5.5 第 4 步 · 抽样评测（分层抽样，看排序）

```bash
python3 -m geval run --config config.json --limit 40 --resume
```

`--limit 40` 按 `subject` 分层等比抽样，每任务约 40 条，成本约为全量的 3%，
**足以看出能力排序和 prompt 协议的相对优劣**。报告会自动标注「不可与公开榜单直接对齐」。

### 5.6 第 5 步 · 全量正式评测

```bash
python3 -m geval run --config config.json --resume
```

`--resume` 会跳过已完成的样本，所以第 4 步跑过的部分不会重复消耗。

**当前单模型评测到此结束。** 报告位置：

```
runs/geval-llm-chat-core/report.html
```

### 5.7 可选模板：阶段 1.5（微调完成后补测）

> 仅适用于 5 对象模板 `config.5objects.json`；当前单模型 `config.json` 没有 post 通道，跳过本节。

1. 启动微调后的两个模型服务，端口与 `config.5objects.json` 里的 `pangu7b-post` / `qwen8b-post` 一致
2. **确认微调没有改动 `chat_template`**（见第 7 节最后一条，这是最容易踩的坑）
3. 跑全量（新模型执行，已完成的三个自动跳过）：

```bash
python3 -m geval run --config config.5objects.json --resume
```

报告里的「微调影响评估」一节会自动产出：逐任务的 pre→post 差值、
95% 置信区间、McNemar p 值，以及**显著退化 / 显著提升 / 不显著**的判定。

### 5.8 可选：确认数字可信（与 lm-eval 交叉验证）

geval 的自建实现有一项已知代价：**prompt 模板细节与公开榜单不同，绝对值不可直接对外引用**。
如果这份数字要用于对外汇报，建议做一次交叉验证——两套工具跑**同一批样本**，看差值是否稳定：

```bash
# 1) 先设置离线变量（内网跑 lm-eval 必须）
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# 2) lm-eval 跑一个小子集（注意用 local-completions，否则拿不到 acc_norm）
#    下面以当前单模型 llm-chat-core 为例；5 对象模板换成 pangu-7b-instruct 与对应 base_url
lm_eval --model local-completions \
  --model_args model=llm-chat-core,base_url=http://<实际地址>/v1/completions,num_concurrent=8 \
  --tasks ceval-valid --num_fewshot 5 --limit 200 \
  --output_path ./lmeval_out/llm-chat-core-200

# 3) geval 跑同一批量级，口径对齐
python3 -m geval run --config config.json --tasks ceval --limit 200
```

> `--tasks` 只匹配任务的 **`name`**（不含 `tag`）。`ceval-likelihood` 是
> `name=ceval` 的 tag 变体，写 `--tasks ceval` 会把该 name 下的所有变体一起跑；
> 只想跑某一个变体，请在配置里删掉其它 ceval 条目或另存一份 config。

**怎么判断**：两边的 ceval 准确率差值通常在 **1~3 个百分点**内（模板细节差异所致）。
差值稳定 → geval 的实现可信，后续配对对比与专业能力测评可以放心只看相对变化，不必每次跑 lm-eval。

如果差值超过 5 个点，优先怀疑三件事，按顺序排查：

1. few-shot 示例的选取与顺序（两边是否一致）
2. 选项渲染格式（`A. 内容` vs `A) 内容` vs `A、内容`）
3. 续写候选串的写法（`" A"` 带空格 vs `"A"`）—— 这会同时影响 acc 与 acc_norm

---

## 6. 用量与耗时参考

全量评测的请求量 ≈ `样本数 × 每样本请求数 × 模型数`。

其中`每样本请求数`取决于打分模式：

| 打分模式 | 请求数/样本 | 用途 |
| :--- | :--- | :--- |
| `gen`（默认） | 1 | 生成文本后抽取答案，日常全量用这个 |
| `logprob` | 1 | 读首 token 概率分布取 argmax |
| `loglikelihood` | **= 选项数**（4 选项即 4） | lm-eval 口径的 acc / acc_norm，**只建议抽样跑** |

各基准的样本量级（供估算，实际值请用 `geval count` 确认）：

| 任务 | 量级 | 备注 |
| :--- | :--- | :--- |
| ceval | ~1.3 千 | 52 学科验证集 |
| cmmlu | ~11.6 千 | **全量最大的一项** |
| mmlu / mmlu_pro | 万级 | 如需跨语言对照再开 |
| gsm8k | ~1.3 千 | |
| bbh | ~6.5 千 | 27 个子任务 |
| ifeval | ~541 | |
| json_schema | 自建 | 用你自己的业务数据 |

**降本建议**：需要与 lm-eval 对齐的 acc/acc_norm 时，把对应选择题任务切成
`loglikelihood` 模式并单独用 `--limit 200` 抽样跑（如模板里的 `ceval-likelihood`）；
全量仍用 `gen` 模式。

---

## 7. 故障排查速查表

| 现象 | 原因 | 处置 |
| :--- | :--- | :--- |
| `HTTP 0 URLError` / 请求超时 | 请求被发给了外网代理 | `unset HTTP_PROXY HTTPS_PROXY`；确认配置 `proxy: "bypass"` |
| `HTTP 502 Bad Gateway` | 同上（代理返回的 502） | 同上 |
| `HTTP 401` / `403` | 鉴权失败 | 检查 `api_key`；部分平台还需要自定义请求头 |
| `HTTP 404 ... model not found` | `model` 字段与服务端加载名不一致 | 跑 `geval probe` 看服务端实际返回的模型名 |
| `Connection refused` | 端点没起 / 端口写错 | 检查服务进程与端口；`curl` 确认 |
| `HTTP 503` / 大批超时 | 并发过高把服务打满 | 调小 `concurrency`（如 8 → 4）后 `--resume` |
| `找不到数据集` | 未准备 JSONL | 见第 4 节；或先用 `--smoke` 验证链路 |
| `样本缺少题面字段` | JSONL 字段名不被识别 | 检查是否含 `question`/`input`/`prompt` 之一，见 `datasets/README.md` |
| 报告「解析失败率」偏高（>2%） | prompt 协议或抽取规则不匹配 | 先 `geval preview --task <任务>` 看实际 payload，再改协议 |
| 结果与预期差很多 | 抽样量太小（噪声） | 看报告第 3 节的置信区间与 p 值；`--limit` 加大 |

### 特别提醒：chat template

阶段 1.5 之前**必须人工确认**微调没有改动 `chat_template`。一旦改动或丢失：

- 微调后的模型会因输入格式不匹配而**全面掉点**
- 现象与「灾难性遗忘」**完全一样**，会让你误判成「微调把模型搞坏了」

确认方法（在能读模型目录的机器上）：

```bash
python3 -c "import json;print(json.load(open('模型目录/tokenizer_config.json'))['chat_template'])"
```

对微调前后的两个模型各执行一次，输出应完全一致。

---

## 8. 运维

| 场景 | 操作 |
| :--- | :--- |
| 中途被中断（Ctrl+C / 掉线） | 重新执行原命令并加 `--resume`，自动跳过已完成样本 |
| 只想重跑某个任务 | `--tasks ceval,gsm8k`（匹配 `name`；同名 tag 变体会一起跑） |
| 只想重跑某个模型 | `--models llm-chat-core`（5 对象模板用 `--models pangu7b-pre` 等） |
| 从头重跑 | 先 `rm -rf runs/<run_name>` 去掉 `--resume` |
| 结果给他人查看 | 直接发 `report.html`（单文件、无外部引用） |
| 结果需要归档 | `tar czf result.tar.gz runs/<run_name>` |
| 磁盘紧张 | 可安全删除 `.cache/`（只影响缓存，下次重算）；**不要删 `runs/`** |

### 结果可复现性

`runs/<run_name>/manifest.json` 记录了完整指纹：
框架版本、Python、平台、并发数、随机种子、生效配置、各数据集文件 sha256、few-shot 条数与协议。

两次运行的 `manifest.json` 对比一致，才说明数字可比。

---

## 9. 产出物清单

```
runs/<run_name>/
├── report.html            ← 主交付物，自包含单文件，可直接发给他人
├── report.md              ← 同内容 Markdown，便于贴进文档系统
├── summary.json           ← 全部聚合结果（给下游程序消费）
├── manifest.json          ← 运行指纹（复现依据）
├── config.json            ← 本次生效的配置快照（`api_key` 已打码为 `***`）
└── <模型 id>/
    ├── <任务>.jsonl              ← 逐样本明细（含原始输出，用于人工复核错例）
    └── <任务>.summary.json       ← 该任务该模型的聚合结果
```

`report.html` 的章节：

| 章节（按标题定位） | 内容 |
| :--- | :--- |
| 1 准确率矩阵 | 模型 × 任务，带 bootstrap 95% 置信区间 |
| 2 综合表现与时延 | 综合分（有 `dimension_weights` 时为加权，否则等权宏平均）、时延 p50/p95、准确率-时延帕累托图 |
| 3 与基线的配对显著性检验 | 与基线逐任务的差值、CI、McNemar p 值（单模型时输出跳过说明） |
| 4 微调影响评估 | pre→post 差值与退化判定（仅当配置了 `report.pairs` 且两端都有结果） |
| 5 解析质量与抽取策略分布 | 解析失败率、抽取策略分布 |
| 6 续写似然打分 | `acc` / `acc_norm` 对照（仅当有 `loglikelihood` 结果） |
| 7 结构化输出专项 | JSON 可解析率 / Schema 通过率 / 意图准确率 / 槽位一致率 |
| 8 指令遵循专项 | 样本级 / 指令级通过率与未覆盖类型（仅当有 `ifeval`） |
| 9 分组明细 | 按 subject 定位短板（前 20 组） |
| 10 错例抽样 | 人工复核用 |
| 11 复现信息 | 数据集指纹、运行期错误、client_stats |

> **章节号是动态的**：3/4/6/7/8 为可选章节，缺失时后续编号前移。
> 引用报告内容请按标题定位，不要写死「第 4 节」。

**结果解读的一条纪律**：不要只看准确率小数点。
只有置信区间不跨 0 且 `McNemar p < 0.05` 的差距才可作为结论。

---

## 10. 命令速查卡

```bash
# 环境与就绪检查
python3 scripts/verify_deps.py                          # 零依赖核查
python3 -m geval doctor --config config.json            # ★ 一条命令查全部前置条件
python3 -m geval selftest                               # 离线全链路自检
python3 -m geval list-tasks                             # 内置任务清单

# 准备与检查
python3 -m geval preview --task json_schema             # 看实际下发的 prompt
python3 -m geval probe   --config config.json           # 端点探活
python3 -m geval probe   --config config.json --models llm-chat-core  # 只探活指定通道
python3 -m geval count   --config config.json           # 用量预估（不发请求）

# 只有 1 个通道时的联通性验证（详见 §5.0）
python3 -m geval run --config config.json --smoke --limit 8 \
    --run-name single-check
python3 -m geval run --config config.json \
    --tasks json_schema --run-name single-check

# 执行（当前单模型 config.json）
python3 -m geval run --config config.json --smoke --limit 8       # 冒烟
python3 -m geval run --config config.json --limit 40 --resume    # 抽样
python3 -m geval run --config config.json --resume               # 全量
python3 -m geval run --config config.json --resume --retry-failed # 限流后补跑失败样本

# 执行（可选 5 对象模板 config.5objects.json）
python3 -m geval run --config config.5objects.json --models pangu7b-pre,qwen8b-pre,qwen3-32b --resume
python3 -m geval run --config config.5objects.json --resume

# 报告
python3 -m geval report --run runs/<run_name>                                # 重新生成
python3 -m geval report --run runs/<run_name> --baseline qwen3-32b           # 指定基线（模板口径）
python3 -m geval report --run runs/<run_name> --pairs pangu7b-pre:pangu7b-post
python3 -m geval report --run runs/<run_name> --weights chinese=0.35,reasoning=0.3,instruction=0.3,math=0.05,json_schema=0

# 版本核对（拿到新源码包后，详见 §2.4）
python3 scripts/verify_install.py                # 0=一致，1=需更新
python3 scripts/verify_install.py --quiet        # 只看不一致项
```

---

## 附录 A：可选 —— 用 lm-eval 出可对标公开榜单的锚点数字

`geval` 的绝对值**不可与公开榜单直接比较**（prompt 模板细节不同）。
若需要一份可对外引用的数字，可离线部署 `lm-eval`：

```bash
# 内网机器（已预先拷入 wheels/）
bash scripts/install_offline.sh wheels/bootstrap wheels/lm-eval --venv /opt/evalenv --bootstrap
/opt/evalenv/bin/pip check
```

运行前必须设置（否则会尝试联网拉数据集而超时）：

```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
```

**注意 backend 的选择**——它决定你能否拿到 `acc_norm`：

| backend | 打哪个接口 | 能出 acc_norm |
| :--- | :--- | :--- |
| `local-completions` | `/v1/completions` | ✅ |
| `local-chat-completions` | `/v1/chat/completions` | ❌ |

```bash
/opt/evalenv/bin/lm_eval --model local-completions \
  --model_args model=pangu-7b-instruct,base_url=http://127.0.0.1:8001/v1/completions,num_concurrent=8 \
  --tasks ceval-valid,cmmlu,gsm8k --num_fewshot 5 \
  --output_path ./lmeval_out/pangu7b
```

先跑 `--tasks list` 确认任务名（各版本命名有差异，不要照抄上面的名字）。

**离线部署的两个已知坑**（下载脚本已自动处理）：

- `sqlitedict`、`word2number`、`langdetect` 在 PyPI 上**只有 sdist、没有 wheel**，
  `--only-binary=:all:` 会失败；而**不 pin 版本时 pip 会静默回退到旧版本（0.3.0），
  把 90MB 的 torch 当核心依赖一起拉下来**。`download_wheels.sh` 已 pin 到
  `0.4.13` 并自动为这些包本地构建纯 Python wheel。
- 数据集必须另外预热并拷进内网，见 `docs/OFFLINE_DEPLOY.md` 坑 1。

---

## 附录 B：内网完全无 pip 源时的注意事项

`geval` 运行时不需要任何第三方包，因此内网无 pip 源**完全不影响**主流程。
只有以下两种情况需要额外准备：

| 情况 | 需要什么 | 怎么做 |
| :--- | :--- | :--- |
| 要在内网跑 `prepare_datasets.py` | `datasets` 库 | 不推荐：这个脚本设计上就是在联网机器跑的 |
| 要离线装 `lm-eval` | 预先下载的 wheels | 见附录 A，`wheels/` 目录已就绪 |

**不要**在目标机的系统 Python 里直接 `pip install --user` 装东西 ——
统一用 `install_offline.sh` 建独立的 venv，避免污染系统环境。
