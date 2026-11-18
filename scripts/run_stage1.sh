#!/usr/bin/env bash
# 通用能力评测：推荐执行顺序。
#
# 被测对象共 5 个（盘古7B 微调前/后、Qwen8B 微调前/后、Qwen3-32B）。
# 阶段 1 先跑微调前已有的 3 个；阶段 1.5 微调完成后跑全量，报告自动产出退化对比。
#
# 每一步都产出可看的报告，不必等全量跑完。

set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
CFG="${CFG:-config.5objects.json}"
PRE_MODELS="${PRE_MODELS:-pangu7b-pre,qwen8b-pre,qwen3-32b}"
# 同一模型内并行跑几个数据集。总并发 = PARALLEL_TASKS × concurrency。
# 数据集量大时（CMMLU/BBH 上万条）调到 2~4 可显著缩短墙钟时间，
# 服务端开始重试/报错就调回去。详见 docs/BATCH_RUNBOOK.md
PARALLEL_TASKS="${PARALLEL_TASKS:-1}"

if [[ ! -f "$CFG" ]]; then
  echo "找不到 $CFG；先执行： cp config.example.json config.5objects.json  然后填好 5 个通道的地址" >&2
  echo "（单模型流程请不要用本脚本，直接按 README §1 的顺序执行）" >&2
  exit 1
fi

echo "=========================================================="
echo " 第 0 步 · 离线自检（不需要模型、不联网）"
echo "=========================================================="
"$PY" scripts/verify_deps.py
"$PY" -m geval selftest

echo
echo "=========================================================="
echo " 第 1 步 · 端点探活（确认地址 / 鉴权 / 模型名）"
echo "         阶段 1 期间，微调后的两个端点还不可用属正常"
echo "=========================================================="
"$PY" -m geval probe --config "$CFG" || true

echo
echo "=========================================================="
echo " 第 2 步 · 冒烟：合成数据 + 微调前 3 个模型，验证接线"
echo "         不消耗真实题目，只确认字段与解析链路正确"
echo "=========================================================="
"$PY" -m geval run --config "$CFG" --smoke --limit 8 --models "$PRE_MODELS"

echo
echo "=========================================================="
echo " 第 3 步 · 小样本真实评测：先看能力排序与 prompt 协议优劣"
echo "         分层抽样，成本约全量的 3%，排序结论已可用"
echo "=========================================================="
"$PY" -m geval run --config "$CFG" --limit 40 --models "$PRE_MODELS" --resume \
      --parallel-tasks "$PARALLEL_TASKS"

echo
echo "=========================================================="
echo " 第 4 步 · 阶段 1 正式评测（微调前 3 个模型，全量）"
echo "         任务并行度 PARALLEL_TASKS=$PARALLEL_TASKS"
echo "=========================================================="
"$PY" -m geval run --config "$CFG" --models "$PRE_MODELS" --resume \
      --parallel-tasks "$PARALLEL_TASKS"

echo
echo "阶段 1 完成。报告位于 runs/<run_name>/report.html"
echo
cat <<'EOF'
──────────────────────────────────────────────────────────
下一步：阶段 1.5（微调完成后）

  1. 确认微调【没有改动 chat template】。若改过，需把微调前的模型也用
     新模板重测，否则 pre→post 的差值里混入了 prompt 格式差异。
  2. 启动微调后的两个模型服务，端口与 config.json 里的 pangu7b-post /
     qwen8b-post 保持一致。
  3. 跑全量 5 个对象（新模型会执行，已完成的 3 个自动跳过）：

       python -m geval run --config config.json --resume

  4. 报告第 4 节会给出逐任务的 pre→post 差值与显著性判定：
     只有 p < 0.05 才判定为「显著退化 / 显著提升」，否则记为持平。
     差值符号约定为 post - pre，正数表示微调后更好。

  5. 验收口径建议提前定死，例如：
     专业准确率提升 ≥ 显著，且 ceval/ifeval 退化 ≤ 2 个点且不显著。
──────────────────────────────────────────────────────────
EOF
