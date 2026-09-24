# 批量评测操作手册（任务并行 / 分批跑 / 合并）

> 适用版本：geval 0.1.0（含 `--parallel-tasks` 与 `merge-runs`）
> 面向场景：内网、单模型或多模型、一次性跑多个大数据集（CMMLU / BBH 这类上万条的）
>
> 一句话：**能批量跑。** 两种姿势——进程内「任务并行」和进程间「分批 + 合并」，
> 二者可以叠加使用。本文给出选型依据、可直接复制的命令、合并规则与排障方法。

---

## 1. 先看清：默认是一次串行的双层循环

`Runner.run()` 的执行结构是**模型在外、任务在内**的两层串行循环，
唯一的并行点是「单个任务内部」的 `ThreadPoolExecutor(concurrency)`：

```
for 每个 model          ← 串行
    for 每个 task       ← 串行
        ThreadPoolExecutor(concurrency) 打完这个任务的全部样本   ← 唯一并行
        写 summary
最后 写 summary.json / report
```

由此得出三条硬约束：

| 约束 | 含义 |
| :--- | :--- |
| **跨任务不并行** | 前一个数据集没跑完，后一个不会开始。BBH 排队时 CMMLU 只能干等 |
| **一条请求=一条样本** | 没有 batch API，请求总数 = 有效样本总数（`loglikelihood` 模式则是 ×选项数） |
| **summary.json 是整体覆写** | 两个进程共用一个 run 目录时，后结束的会覆盖先结束的（所以分批必须分目录） |

耗时公式：

```
总耗时 ≈ 总请求数 × 单请求时延 ÷ 并发数
```

你当前这份 `config.json`（单模型 `llm-chat-core`，6 个任务，全为 `gen` 模式）的规模：

| 任务 | 样本数 | 占比 | 备注 |
| :--- | ---: | ---: | :--- |
| cmmlu | 11582 | 54.3% | 最大的一块，独占一半以上时间 |
| bbh | 6511 | 30.5% | prompt 长、生成长，单请求更贵 |
| ceval | 1346 | 6.3% | **你已跑完的只是这 6%** |
| gsm8k | 1319 | 6.2% | |
| ifeval | 541 | 2.5% | |
| json_schema | 16 | 0.1% | |
| **合计** | **21315** | | 剩余量约为已跑完部分的 **14.8 倍** |

> `ceval` 配置里写的 `"split": "val"`，但 `datasets/ceval/` 下只有 `dev.jsonl` / `test.jsonl`，
> 按 `resolve_dataset_path` 的候选顺序会落到 `test.jsonl`（1346 条）。数据没问题，只是提醒别以为少算了。

先量一次真实时延，再决定开多大并行（这一步不发大量请求）：

```bash
python -m geval probe --config config.json
python -m geval count --config config.json --models llm-chat-core --p50-ms <probe 出来的时延>
```

---

## 2. 三种跑法怎么选

| 方式 | 命令 | 适用 | 出报告 |
| :--- | :--- | :--- | :--- |
| **A. 任务并行** | `run --parallel-tasks N` | 一台机器、一个模型、多个数据集 | 一次运行直接出完整报告，**最省事** |
| **B. 分批 + 合并** | 多进程 `run` + `merge-runs` | 想跨机/跨卡拆、想按数据集控制先后顺序、担心一次跑崩 | 合并后出一张总表 |
| **C. A+B** | 每批内部再开并行 | 数据量大且机器吃得下 | 同 B |

经验：**先 A，撑不住再 B。**
A 的好处是不需要处理任何中间状态，报告天然完整；B 的价值在于可以把最贵的那个数据集单独拎出来、断点返工，以及跨机器摊薄。

---

## 3. 方案 A：`--parallel-tasks` 任务级并行

### 3.1 用法

```bash
python -m geval run --config config.json --parallel-tasks 3
#                                         └─ 同一模型内最多 3 个数据集同时跑
```

启动时会打印实际在途量：

```
并发数     8（任务并行 3 → 实际在途最多 24）
```

即：**总并发 = `parallel_tasks` × `concurrency`**。两者语义不同，别混淆：

- `concurrency`：单个任务内部的线程池大小（不变的行为）
- `parallel_tasks`：同时推进几个数据集（新增）

### 3.2 怎么定这两个数

1. `probe` 拿到单请求时延 `T`（毫秒）与服务端是否稳定；
2. **先固定 `concurrency`，用 `parallel_tasks` 放大**。这样每个数据集的语义不变，排障时仍然是一行任务对应一串请求；
3. 逐步加：`--parallel-tasks 2` → `3` → `4`，每档跑 2 分钟看 `summary.json` 里 `client_stats` 的
   `http_retries` / `errors`。**重试明显变多说明服务端已经排队或限流了，再加只是把延迟从计算搬到队列里**；
4. 目标：既没有大量重试，又让所有数据集都在动。

经验起点（8~32 视服务端而定）：

```bash
python -m geval run --config config.json --parallel-tasks 3 --resume
```

### 3.3 行为细节

- **进度输出**：并行时每个数据集的进度改为**逐行打印**（带 `模型/任务` 前缀），不再用 `\r` 互相覆盖。刷新节奏是「条数 + 时间」双节流：条数步长上限 100 条，且串行在终端里至少每 2 秒刷一次、逐行输出场景至少每 10 秒刷一次（可用 `--progress-interval` 改）。
- **模型仍串行**：多模型配置下，模型是一个个跑的，只是每个模型内部的数据集并行。目的是避免多个端点互相抢，也让报告逐模型成型。
- **失败隔离**：某个数据集炸了（数据缺失、配置写错）只影响它自己，其余任务照常跑完并出报告。
- **与 `--resume` 正交**：`--parallel-tasks 3 --resume` 完全支持，已落盘的样本照旧跳过。
- **指纹留痕**：`manifest.json` 里新增 `parallel_tasks` 字段，写进每一次运行的复现信息。

---

## 4. 方案 B：多进程分批 + `merge-runs`

### 4.1 为什么要分目录

`Runner` 收尾时把本次结果整体写进 `summary.json`。两个进程若共用 run 目录，
明细文件（`<model>/<task>.jsonl`）各写各的不会冲突，但**最后写入 summary.json 的那进程会覆盖前者的**。
结果一定是「丢了一批」，而且丢谁取决于谁先结束——不可预测。所以：**分批必须分 run 目录。**

### 4.2 完整流程（可直接复制）

```bash
CFG=config.json

# 1) 分批并行：每个批次一个 run 目录、一组数据集
python -m geval run --config $CFG --run-name p-ceval  --tasks ceval       --resume &
python -m geval run --config $CFG --run-name p-cmmlu  --tasks cmmlu       --resume &
python -m geval run --config $CFG --run-name p-reason --tasks bbh,gsm8k   --resume &
python -m geval run --config $CFG --run-name p-format \
       --tasks ifeval,json_schema --resume &
wait                                   # 等所有批次结束

# 2) 合并成一个新的 run 目录，并直接出报告
python -m geval merge-runs runs/p-ceval runs/p-cmmlu runs/p-reason runs/p-format \
       --run-name merged
```

产物：

```
runs/merged/
  summary.json            重建的总表（含 merged.sources 追溯信息）
  manifest.json           合并来源清单
  merged_sources/         各批次生效配置快照（<源目录名>.config.json）
  <model>/<task>.jsonl    逐样本明细（从各批次拷贝而来）
  <model>/<task>.summary.json
  report.md / report.html  ★ 一张完整总表
```

合并目录就是一个**标准 run 目录**，因此还能继续当 experiments 接着跑：

```bash
# 某个数据集没跑完，直接在合并目录里续跑
python -m geval run --config config.json --run-name merged --tasks cmmlu --resume
python -m geval report --run runs/merged
```

### 4.3 合并规则（确定性，避免"谁后写谁赢"）

| 情形 | 处理 |
| :--- | :--- |
| 同一「模型 × 任务」出现在多个批次 | 取**样本数 n 更大**的一份；n 相同时取靠前的源；有 `error` 的让位给正常的。冲突清单会打印并写入 `summary.json → merged.conflicts` |
| 目标文件已存在 | **不覆盖**（按字节数判断：更大或相等就跳过），因此重复执行 `merge-runs` 是安全的 |
| 源目录缺 `summary.json` | 自动降级：用 `<model>/<task>.summary.json` 逐个任务文件重建（`summary_from_task_files`），并给出 `! …已由任务级 summary 文件重建` 提示 |
| 源目录里一个结果都没有 | 跳过，列入 `merged.sources` 之外的 warnings，不会失败整个合并 |
| 输出目录等于某个源目录 | 直接报错中止（防止原地覆写） |

`client_stats` 合并时对可累加项（`calls` / `errors` / `cache_hits` / `http_requests` / `http_retries`、
`usage`）求和，时延分位数取首个来源的值——**分位数不能相加**，跨批次看时延只能看趋势，这也是 merge 输出保留各源配置快照的原因：把延迟关掉的那批跟开着的那批放一起比时延，本身就不可比。

### 4.4 跨机器合并

明细与任务聚合都是纯文本文件，合并只依赖这些文件，因此**可以在不同机器上跑不同批次，再把 `runs/<batch>` 整个目录拷到一台机器上合并**：

```bash
# 在汇总机上
python -m geval merge-runs /data/from-a/p-cmmlu /data/from-b/p-bbh --run-name merged
```

前提：各批次用的是**同一份数据源**（sha256 一致）、同一套 prompt 协议。
不同挡位的 `limit` / `protocol` / `num_fewshot` 混在一起，报告里的配对检验会给出告警——这不是合并的 bug，是数据本身不可比。

---

## 5. 少测某个数据集 / 抽样

四种粒度，按推荐顺序：

**(1) 配置里逐任务设 `limit`（最灵活，推荐）**

```jsonc
{ "name": "cmmlu", "split": "test", "num_fewshot": 5, "protocol": "fewshot", "limit": 1500 },
{ "name": "bbh",   "split": "test", "num_fewshot": 0, "protocol": "answer_line", "limit": 1000 },
{ "name": "json_schema", "limit": 0 }        // 0 = 全量，那里只有 16 条
```

按 `subject` 分层等比抽样（`seed=42`），比随机截断更能代表全量。

**(2) CLI 只跑指定任务**

```bash
python -m geval run --config config.json --tasks ifeval,json_schema   # 557 条，十分钟以内出结论
```

注意 `--tasks` 匹配的是任务的 **`name`**，不包含 `tag`。同一 `name` 起了多个 tag（如 three ceval 变体）时，
`--tasks ceval` 会把所有 ceval 变体都跑一遍；只想跑某一个协议变体，请在配置里删掉其它条目或另存一份 config。

**(3) CLI 全局抽样** `--limit N`

```bash
python -m geval run --config config.json --limit 300 --run-name quick
```

`(4) 直接删配置里的 tasks 条目**：最彻底，也最容易事后忘掉，改了记得在报告里注明。

抽样跑的顺序建议：`json_schema` → `ifeval` → `ceval` → `gsm8k` → `bbh` → `cmmlu`。
前两个是与意图识别最相关的专项，也最便宜；排在最后的是最贵的。

---

## 6. 提速清单（不动数据集的前提下）

| 手段 | 预期收益 | 代价 / 注意 |
| :--- | :--- | :--- |
| `--parallel-tasks 2~4` | 近线性，直到服务端排队为止 | 重试变多就停手 |
| `temperature: 0` | **启用磁盘缓存**：重复/重跑不再花钱，且结果可复现 | 与 `0.3` 的结果口径不同，改了建议重跑；缓存只在 temp=0 生效 |
| 选择题改 `"protocol": "direct", "num_fewshot": 0` | prompt 短 + 不写推理过程，生成 token 大幅下降 | 与 few-shot 口径不可直接比，用 `tag` 起别名对比更稳 |
| 逐任务 `"max_tokens": 16`（配 `direct`） | 防止模型在选择题上长篇大论 | **必须配 `direct`**；`answer_line` 协议下截断会丢答案 |
| 提高 `concurrency` | 与 parallel-tasks 相乘 | 服务端限流的首要触发点 |
| `retries: 1` + 更短 `request_timeout_s` | 卡住的请求更快失败、不再反复退避 | 网络抖动时样本更容易失败；看 `client_stats` 决定 |

一条实用的组合（选择题任务）：

```jsonc
{ "name": "cmmlu", "split": "test", "protocol": "direct", "num_fewshot": 0, "max_tokens": 16, "limit": 2000 }
```

---

## 7. 故障排查

| 现象 | 原因与处理 |
| :--- | :--- |
| `merge-runs` 报「没有任何可合并的结果」 | 源目录下既没有 `summary.json` 也没有 `<model>/*.summary.json`。确认路径写对、批次是否真的跑完 |
| 合并后任务数少于预期 | 某批次的 `<model>/<task>.summary.json` 被误删；用 `--no-report` 合并后看 `summary.json → merged.sources`，逐个批次核对 |
| 合并后某数据集分数与分批时不一致 | 只有一种可能：该「模型 × 任务」在多个批次里都存在且 n 不同，取了 n 更大的那份。冲突清单在 `merged.conflicts` |
| `--parallel-tasks` 加了但没变快 | 服务端已排队。看 `summary.json → manifests.client_stats.http_retries`，上涨就是被打满了 |
| 某个大数据集一直不出进度 | 见下面的「看不到进度条怎么办」 |
| 并行后进度看不清 | 并行模式故意改成逐行输出；加 `--quiet` 可以完全关掉，看 jsonl 行数判断进度：`wc -l runs/<name>/*/*.jsonl` |
| 某任务报「执行失败」但其它正常 | 失败被隔离了。看 `summary.json → errors`，常见是数据集缺失或 `tag` 重名 |
| `probe` 超时但 curl 通 | 服务器上的 `HTTP_PROXY` 把内网请求发到外网了。确认 `proxy: "bypass"` |
| 中断了怎么办 | 每条样本实时落盘，直接 `--resume` 续跑；分批就用原来的 `--run-name` 重跑同一条命令 |

### 进度条上的数字是什么意思

```
  pangu7b-pre/cmmlu 1563/10426 ( 15.0%) 12.3/s ETA 720s
  └── 模型id/任务名   └─ 已完成/本次需处理  └速率 └预计剩余
```

重点是**分母 10426 不是数据集的原始条数**，它经过三层折算：

| 阶段 | 条数 | 说明 |
| :--- | ---: | :--- |
| `datasets/cmmlu/test.jsonl` 原始 | 11582 | 数据集文件本身 |
| 归一化 + 题面去重后 | **11408** | 去掉 174 条重复题面；**这就是任务开跑时打印的「样本 N 条」** |
| 减去 `--resume` 已完成的 | **10426** | 本次真正要发的请求数，**这才是进度条的分母** |

所以 `--resume` 续跑时，分母会比「样本 N 条」小，两者不相等是正常的
（上例说明 jsonl 里已有 11408 − 10426 = 982 条）。同理分子 1563 也只统计
**本次新跑的**，不含续跑跳过的那批。

要判断「这个数据集整体跑完了百分之多少」，看落盘行数而不是进度条：

```bash
echo "$(cat runs/<run>/<model>/cmmlu.jsonl | wc -l) / 11408"
```

另一种让分母变小的可能是配置里的 `limit`（抽样跑），
`config.json` 里 cmmlu 若设了 `limit: 1500`，分母就是 1500。

### 看不到进度条怎么办

CMMLU / BBH 这类上万条的任务，最容易遇到的观感问题是「跑了半天一行都没有」。
三个成因，逐条对应处理：

**(1) 刷新步长按条数算，大任务被稀释**（已修复）
旧逻辑是 `every = 总样本数 // 20`，CMMLU 算出来是 **579 条才刷一次**。
并发 8、单请求 3s 时，前 579 条要跑约 3.6 分钟 —— 这段时间终端完全静默，
看起来就像这个数据集没有进度条。现在改成**条数步长上限 100 + 时间兜底**
（终端 2 秒、逐行输出 10 秒必刷一次），并且任务一启动就先打印一行 `0/N`。

**(2) `\r` 原地刷新在日志里看不见**（已修复）
进度写的是 **stderr**，用 `\r` 原地刷新。一旦把输出重定向到文件（`2>log.txt`）、
跑在 IDE 输出面板或 CI 里，`\r` 不会换行，几百次刷新叠成一行，日志里等于没进度。
现在检测到 stderr 不是终端时**自动切成逐行输出**。

**(3) 多进程各刷各的，互相覆盖**
按 §4 分批并行时，多个进程往同一个终端写，`\r` 会互相冲掉。
**每个进程单独重定向到自己的日志**，进度就干净了：

```bash
python -m geval run --config config.json --run-name p-cmmlu --tasks cmmlu --resume \
  2> logs/cmmlu.log &
python -m geval run --config config.json --run-name p-bbh   --tasks bbh   --resume \
  2> logs/bbh.log &
```

**最可靠的进度观测：直接数落盘行数。** 样本是逐条 flush 的，
jsonl 的行数就是真实进度，跟进度条怎么刷无关：

```bash
watch -n 5 'wc -l runs/<run_name>/<model>/*.jsonl'
# 或者看整体：已处理 / 总量
watch -n 5 'cat runs/<run_name>/<model>/*.jsonl | wc -l'
```

另外提示：`task.prepare()` 阶段（读数据、拼 prompt、算哈希）本身没有输出，
CMMLU 一万多条要安静几十秒才进入请求阶段，这是正常的，不是卡死。

---

## 8. 内网怎么同步这批改动

框架本体零第三方依赖（新代码只用标准库 `shutil` / `threading`），拷目录即可：

```bash
# 联网/开发机：出增量源码包（排除 datasets/ 与 wheels/，不碰内网的数据与 venv）
python scripts/pack_bundle.py --code-only
# → dist/geval-src-<日期>.tar.gz（约 150 KB）+ dist/CODE_MANIFEST.txt

# 内网机：覆盖后核对
python3 scripts/verify_install.py
# 确认报「一致」且 geval/merge.py 存在：
python3 -m geval selftest          # 应出现「任务并行…」「merge-runs …」两行通过项
```

`selftest` 已把新能力纳入离线校验，不需要模型也能确认这批代码在内网跑通：

```
· 任务并行（parallel-tasks>1）结果与串行完全一致，无竞态
· merge-runs 能把两个分批目录合并回完整的总表并出报告
```

---

## 9. 参数速查

| 参数 | 作用 |
| :--- | :--- |
| `--tasks a,b` | 只跑指定数据集（按 `name`，不含 tag） |
| `--models x,y` | 只跑指定模型 id |
| `--limit N` | 全局每任务抽样上限（分层抽样，含 subject 保底 1 条） |
| `--parallel-tasks N` | 同模型内并行跑几个数据集；总并发 = N × concurrency |
| `--progress-interval S` | 进度最小刷新间隔（秒）；0=自动（终端 2s / 逐行 10s）。数据集大却看不到进度时调成 1 |
| `--resume` | 断点续跑，跳过已落盘样本 |
| `--run-name` | run 目录名（分批与合并的核心控制点） |
| `merge-runs DIR [DIR…] [--out] [--run-name] [--baseline] [--pairs] [--no-report]` | 合并多个 run 目录并出总报告 |
| 任务级 `"limit"` / `"max_tokens"` / `"protocol"` / `"mode"` | 逐数据集精细控制，优先级高于全局 CLI 参数 |
