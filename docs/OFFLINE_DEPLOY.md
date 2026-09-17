# 离线部署指南：把开源评测框架装进内网 ARM64

> 前提：内网 Linux ARM64（aarch64）+ Python 3.11，**无外网、无 pip 源**，
> 但可以把文件从外网机器拷进去。
>
> 结论：**完全可行，推荐 lm-eval-harness。** 但有两个坑，第二个比第一个更隐蔽。

---

## 1. 先看依赖重量：为什么 lm-eval 是首选

实测两套框架在「只评测 OpenAI 兼容端点」场景下要装什么：

### lm-eval-harness

基础依赖（`pip install lm-eval`）：

```
datasets  numpy  evaluate  jinja2  pytablewriter  rouge-score  sacrebleu
scikit-learn  sqlitedict  dill  word2number  more_itertools  typing_extensions  tqdm
```

评测 API 端点需再加 `[api]` extra：`requests  aiohttp  tenacity  tqdm  tiktoken`
IFEval 任务需再加 `[ifeval]` extra：`langdetect  immutabledict  nltk`

**关键：`torch` 不在其中。** torch 只被 `[hf]` extra 引入，而 `[hf]` 是「直接用 HuggingFace 权重评本地模型」才需要的。
你的模型本来就是 vLLM 服务化的，走 API 就完全绕开了整个 GPU 依赖栈。
上列包全部有 aarch64 wheel 或是纯 Python 包。

### EvalScope

基础依赖含 `transformers>=4.33`、`modelscope[datasets]`、`pandas`、`plotly`、`pillow`、`litellm` 等。
`openai_api` 是内置 eval-type，不需要额外 extra。

同样**不含 torch**，但比 lm-eval 重不少，aarch64 轮子体量与出错面都更大。
优点是自带 Web Dashboard 和性能压测，如果你后面要做压测，值得单独装。

**建议：阶段 1 用 lm-eval 出对标数字；需要压测或可视化看板时再装 EvalScope。**

---

## 2. 三步流程

> 已预下载：`wheels/bootstrap`（4 个）与 `wheels/lm-eval`（67 个，135.9 MB，lm-eval 0.4.13，无 torch）。
> 校验值见 `dist/WHEELS_MANIFEST.txt`。已打包为 `dist/llm-general-eval-<日期>.tar.gz`，可直接拷进内网。
> 若目标机不是 Python 3.11 / glibc 2.17+，用下面的命令重新下载即可。

### 第 1 步：联网机器下载轮子

```bash
# 先建一个与目标环境同版本的干净目录（避免被本机已装的包影响）
bash scripts/download_wheels.sh bootstrap   # pip / setuptools / wheel
bash scripts/download_wheels.sh lm-eval     # 框架本体
bash scripts/download_wheels.sh evalscope   # 可选

# 或者一次全下
bash scripts/download_wheels.sh all

# 目标不是 Python 3.11 或不是 glibc 2.17+ 时，用环境变量覆盖
PY_VER=312 PLATFORM=manylinux_2_28_aarch64 bash scripts/download_wheels.sh lm-eval
```

脚本内置了三道保护：

1. `--only-binary=:all:`：遇到**没有 aarch64 wheel 的包会立刻报错**，
   让问题在联网机器上暴露，而不是拷进内网才失败；
2. 下载后逐个检查 wheel 文件名，平台标签不是 `aarch64` 或 `none-any` 就直接报错；
3. 明确不装 `[hf]` extra，避免误把 torch 拉进来（会让包体从几十 MB 涨到几个 GB）。

**更可靠的做法（推荐，若有 Docker）**：在 ARM64 容器内下载，天然匹配平台与 ABI：

```bash
USE_DOCKER=1 bash scripts/download_wheels.sh lm-eval
```

### 第 2 步：打包并拷进内网

```bash
tar czf wheels.tar.gz wheels/
# 通过你现有的文件导入通道把 wheels.tar.gz 送进内网
```

### 第 3 步：内网机器安装

```bash
tar xzf wheels.tar.gz
bash scripts/install_offline.sh wheels/bootstrap wheels/lm-eval --venv /opt/evalenv --bootstrap

# 校验
/opt/evalenv/bin/pip check
/opt/evalenv/bin/lm_eval --help
```

`--no-index` 保证**完全不访问网络**：万一漏了依赖，pip 会直接告诉你缺哪个包，
而不是默默去外网拉然后卡住。

---

## 3. 六个坑（按踩到的概率排序）

### 坑 1（最容易忽略）：数据集也要离线

**装好轮子不等于能跑。** lm-eval 启动时会从 HuggingFace Hub 下载数据集，
内网环境下会卡在连接超时上——而且报错信息通常很含糊。

三个办法，推荐第一个：

```bash
# 办法 A（推荐）：在联网机器上先触发一次数据集缓存，再把缓存整个拷进去
HF_HOME=/tmp/hfcache lm_eval --model local-chat-completions \
  --model_args model=x,base_url=http://127.0.0.1:9/v1/chat/completions \
  --tasks ceval-valid,mmlu,gsm8k --limit 1 --output_path /tmp/warmup
# 这一步会因为连不上模型而失败，但数据集已经下好了 —— 目的就是拿数据集
tar czf hf_cache.tar.gz -C /tmp/hfcache .
# 拷进内网，解压到目标机 ~/.cache/huggingface，然后：
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

```bash
# 办法 B：直接指定本地数据集目录（需要自己改写 task yaml 里的 dataset path）
# 办法 C：用 geval 把数据集转成 JSONL，自己写 lm-eval 的本地 dataset 适配器
#        （工作是有的，除非你有特殊格式需求，否则不建议）
```

> 别忘了 `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`。不设的话，
> 即使缓存已存在，datasets 库仍会先尝试联网检查版本，然后超时。

### 坑 2：pip 版本太旧，不认识新的 manylinux 标签

症状：内网 `pip install` 报 `... is not a supported wheel on this platform`，
但 wheel 文件名看起来完全正确。

原因：较新的包会打出 `manylinux_2_28_aarch64` 这类标签，
旧 pip（< 20.3）只认识 `manylinux2014_aarch64`。

解决：把 `pip/setuptools/wheel` 一起下下来，先离线升级，再装框架：

```bash
bash scripts/download_wheels.sh bootstrap
bash scripts/install_offline.sh wheels/bootstrap wheels/lm-eval --bootstrap
```

### 坑 3：环境标记按「当前机器」解析，跨平台下载可能解析错

`pip download --platform ...` 只影响 **wheel 选择**，
但依赖图里的**环境标记**（`sys_platform`、`platform_machine`、`python_version` 等）
是按执行下载的那台机器求值的。

后果：某些包在 Windows/macOS 上会解析出错误的可选依赖组合。

解决：这是 `download_wheels.sh` 推荐 `USE_DOCKER=1` 的原因。
如果只能用跨平台方式，**装完必须在内网跑 `pip check`**，
它会报出缺失或冲突；报错了就回到联网机器补下缺失的包。

### 坑 4：上游只有 sdist 的包（已实测确认，脚本会自动处理）

**这不是理论风险，是实测踩到的。** 以下三个包在 PyPI 上**只发布 sdist（.tar.gz），没有 wheel**：

| 包 | 谁需要它 |
| :--- | :--- |
| `sqlitedict` | `lm-eval` 核心依赖 |
| `word2number` | `lm-eval` 核心依赖 |
| `langdetect` | `lm-eval[ifeval]` extra |

`--only-binary=:all:` 遇到它们会直接失败，报错信息相当绕：

```
Additionally, some packages in these conflicts have no matching distributions
available for your environment:
    black
    sqlitedict
```

**更严重的是它的连带后果 —— pip 会静默回退到旧版本。**
实测中，因为 `word2number` 找不到，pip 自动把 `lm-eval` 从 0.4.13 **回退到了 0.3.0**，
而 0.3.0 把 `torch` 当核心依赖 —— 于是白下了 **90 MB 的 torch**，
依赖树也和预期完全不同（多了 transformers / tokenizers / safetensors / scipy）。

**两个必须做的动作**：

1. **显式 pin 版本**，否则 pip 的回退是静默的、你只会觉得「怎么装出来不一样」：
   ```bash
   pip download "lm-eval[api,ifeval]==0.4.13" ...
   ```
2. **为 sdist-only 的包本地预构建 wheel**，再把它作为查找源：
   ```bash
   pip wheel sqlitedict word2number langdetect --wheel-dir wheels/lm-eval --no-deps
   pip download "lm-eval[api,ifeval]==0.4.13" --dest wheels/lm-eval \
       --only-binary=:all: --platform manylinux2014_aarch64 \
       --python-version 3.11 --implementation cp --abi cp311 \
       --find-links wheels/lm-eval
   ```

**关键前提**：这些包必须是**纯 Python** 的，本地构建才能产出跨平台的
`py3-none-any.whl`。如果某个包含 C 扩展，在 x86 上下载机上构建出来的
wheel 是**错的平台**，拷进 ARM64 会装不上 —— 所以构建后必须检查产物标签。

`scripts/download_wheels.sh` 已经把这套逻辑做进去了：

- 内置 `KNOWN_SDIST_ONLY`（上述三个包）预先构建；
- 自带**自愈循环**：解析 pip 报错里「没有可用 wheel」的包名 → 本地构建 → 重试，
  因此**未知的** sdist-only 包也能被发现（实测有效）；
- 构建后**强制校验产物是 `-none-any.whl`**，不是就报错，避免跨平台的错误产物；
- 下载完成后校验：版本必须等于 pin 的那个，且**结果里不允许出现 torch**。

真遇到了含 C 扩展的 sdist-only 包，优先级依次是：
找纯 Python 替代 → 在 ARM64 机器或 ARM64 容器里构建（`USE_DOCKER=1`）→ 放弃该功能。

### 坑 5：别顺手装 `[hf]` extra

`lm-eval[hf]` 会引入 `torch` + `transformers` + `accelerate` + `peft`。
在 ARM64 上 torch 的轮子体积是 GB 级，而且不同版本对 aarch64 的支持差异很大。

你的评测对象是 vLLM 服务化的模型，走 API 就够了：`lm-eval[api]`。

### 坑 6：`/completions` 与 `/chat/completions` 不是一回事

这直接决定你**能不能拿到 acc_norm**：

| lm-eval backend | 打的接口 | 支持 loglikelihood | 能出 acc_norm |
| :--- | :--- | :--- | :--- |
| `local-completions` | `/v1/completions` | ✅ | ✅ |
| `local-chat-completions` | `/v1/chat/completions` | ❌ | ❌（只能出生成式指标） |

所以要在 C-Eval / MMLU 上得到可对标榜单的 `acc_norm`，
必须用 **`--model local-completions`**，并且确认你的推理服务暴露了 `/v1/completions`
（vLLM 默认支持）。

```bash
# 先确认端点是否存在
curl -s http://127.0.0.1:8001/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"你的模型名","prompt":"1+1=","max_tokens":4,"echo":true,"logprobs":1}' | head -c 400
```

如果这个 `echo: true` 的请求返回了 `logprobs.tokens` 和 `logprobs.token_logprobs`，
说明 acc_norm 路径是通的。

---

## 4. 内网运行 lm-eval 的实际命令

```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# 内网禁走外网代理（与 geval 的 proxy=bypass 是同一个坑）
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

LM=/opt/evalenv/bin/lm_eval

# 先看实际可用的任务名 —— 不要照抄本文档里的名字，各版本命名有差异
$LM --tasks list | grep -iE 'ceval|cmmlu|mmlu|gsm8k|bbh|ifeval'

# 单模型、双任务冒烟（10 条）
$LM --model local-completions \
    --model_args model=qwen3-32b,base_url=http://127.0.0.1:8003/v1/completions,num_concurrent=8,max_retries=3 \
    --tasks ceval-valid,mmlu --num_fewshot 5 --limit 10 \
    --output_path ./lmeval_out/qwen3-32b-smoke

# 正式跑（注意 num_concurrent 要与推理服务的并发能力匹配）
$LM --model local-completions \
    --model_args model=qwen3-32b,base_url=http://127.0.0.1:8003/v1/completions,num_concurrent=16,max_retries=3 \
    --tasks ceval-valid,cmmlu,mmlu,gsm8k --num_fewshot 5 \
    --output_path ./lmeval_out/qwen3-32b
```

对三个模型各跑一遍即可。生成式任务（GSM8K）也可以用 `local-chat-completions`
搭配 `--apply_chat_template`，但为了口径统一，建议全程用 `local-completions`。

---

## 5. 两套系统怎么配合（这是重点）

装好开源框架后，**不是要你放弃 geval**，而是各管一段：

| 需求 | 用什么 | 理由 |
| :--- | :--- | :--- |
| 阶段 1：三个模型的通用能力**正式数字** | **lm-eval** | 官方模板、社区验证、`acc` / `acc_norm` 与榜单同口径 |
| 阶段 1：prompt 协议消融（direct / CoT / fewshot 哪个好） | geval | 改 JSON 就行，不用动框架配置；带配对显著性 |
| 阶段 1.5：微调前后**配对对比** | geval | 没有框架会做「你的模型 A vs 你的模型 B 的 McNemar 检验」 |
| 阶段 4：专业能力测评（意图 + 槽位） | geval | 开源框架没有你的业务数据与指标定义 |
| 推理服务性能压测（TTFT / P99 / 吞吐） | EvalScope `perf` 或 vLLM bench | lm-eval 不做压测 |

### 对齐一次，之后可以放心用 geval

```bash
# 1. lm-eval 跑一个小子集（例如每任务 200 条），拿到锚点数字
# 2. geval 跑同一批样本，配置对齐：
#    "num_fewshot": 5, "mode": "loglikelihood", "primary": "acc_norm", "limit": 200
python -m geval run --config config.json --tasks ceval-likelihood --limit 200

# 3. 比对两者差值
```

差值通常在 1~3 个百分点内（模板细节差异）。**差值稳定**说明 geval 的实现可信，
之后阶段 1.5 / 阶段 4 就可以放心用 geval 看相对变化，不必每次都跑 lm-eval。

如果差值很大（>5 个点），优先怀疑三件事：
few-shot 示例的选取顺序、选项渲染格式（`A. 内容` vs `A) 内容`）、
以及续写候选串的写法（`" A"` 带空格 vs `"A"`）。

---

## 6. 上线前校验清单

在**联网机器**上：

- [ ] `download_wheels.sh` 无报错（`--only-binary=:all:` 没失败）
- [ ] 脚本的 wheel 标签检查全绿
- [ ] `torch` 不在下载列表里
- [ ] HF 数据集缓存已生成并打包

在**内网机器**上：

- [ ] `python -c "import sys; print(sys.version)"` 与 `PY_VER` 一致
- [ ] `pip check` 无缺失、无冲突
- [ ] `lm_eval --tasks list` 能跑起来（说明包与数据集元数据都在）
- [ ] `curl` 测 `/v1/completions` 的 `echo`+`logprobs` 返回正常
- [ ] 每个任务 `--limit 10` 冒烟跑通
- [ ] `HF_DATASETS_OFFLINE=1` 已设置（否则会联网超时）
- [ ] `HTTP_PROXY` 已 unset（否则内网请求被发给外网代理）

---

## 7. 如果最终决定只用 geval

也完全可以，只要清楚代价（见 `docs/EVAL_METHODOLOGY.md` §3）：

- 与公开榜单的**绝对值不可比**，只能看相对差值
- 没有第三方验证，实现细节自带偏差
- 基准数量少（12 个任务）

代价是可量化的，不是致命的。但既然依赖可以导入，**让 lm-eval 负责「对外可引用」的那部分数字，
成本很低、收益很明确** —— 这是我推荐混合而不是二选一的原因。
