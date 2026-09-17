#!/usr/bin/env bash
# 在**内网机器**上执行：从离线 wheel 目录安装评测框架。
#
# 用法：
#   bash scripts/install_offline.sh wheels/lm-eval
#   bash scripts/install_offline.sh wheels/evalscope --venv /opt/evalenv
#   bash scripts/install_offline.sh wheels/bootstrap --venv /opt/evalenv --bootstrap
#
# 参数：
#   $1                 wheel 目录（可传多个，见下）
#   --venv PATH        虚拟环境路径，默认 ./evalenv
#   --bootstrap        先离线升级 pip / setuptools / wheel（首次建环境时用）
#   --system           直接装进当前环境（不用 venv）
#
# 多个 wheel 目录会被合并成一个 --find-links 列表，例如：
#   bash scripts/install_offline.sh wheels/bootstrap wheels/lm-eval
#
# 关于 --no-index：
#   加上它才能确保「完全离线、绝不访问网络」。如果某次遗漏了依赖，
#   pip 会直接报错告诉你缺哪个包，而不是默默去外网拉——这正是内网想要的行为。

set -euo pipefail

WHEEL_DIRS=()
VENV="evalenv"
BOOTSTRAP=0
USE_SYSTEM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)      VENV="$2"; shift 2 ;;
    --bootstrap) BOOTSTRAP=1; shift ;;
    --system)    USE_SYSTEM=1; shift ;;
    -h|--help)   sed -n '2,26p' "$0"; exit 0 ;;
    *)           WHEEL_DIRS+=("$1"); shift ;;
  esac
done

info() { printf '\033[36m[info]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

[[ ${#WHEEL_DIRS[@]} -gt 0 ]] || die "请指定至少一个 wheel 目录，例如：bash $0 wheels/lm-eval"

# 构造 --find-links 参数
FIND_LINKS=()
for d in "${WHEEL_DIRS[@]}"; do
  [[ -d "$d" ]] || die "目录不存在：$d"
  n=$(find "$d" -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')
  [[ "$n" -gt 0 ]] || die "目录里没有 .whl 文件：$d"
  info "wheel 目录 ${d}（${n} 个轮子）"
  FIND_LINKS+=(--find-links "$d")
done

# --------------------------------------------------------------------------- #
# Python 版本检查：wheel 的 ABI 标签必须与解释器匹配
# --------------------------------------------------------------------------- #
PY_BIN="python3"
command -v "$PY_BIN" >/dev/null || die "未找到 python3"

PY_FULL="$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_TAG="$("$PY_BIN" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
info "系统 Python ${PY_FULL}（ABI ${PY_TAG}）"

# 从轮子文件名反推它期望的 ABI，提前报错，避免装到一半才失败。
# 用 basename 循环而非 find -printf —— 后者是 GNU 扩展，BSD/macOS 上不可用。
EXPECTED_ABI=""
for d in "${WHEEL_DIRS[@]}"; do
  while IFS= read -r whl; do
    base="$(basename "$whl")"
    if [[ "$base" =~ -(cp3[0-9]+)- ]]; then
      EXPECTED_ABI="${BASH_REMATCH[1]}"
      break 2
    fi
  done < <(find "$d" -maxdepth 1 -name '*.whl' | sort)
done
if [[ -n "$EXPECTED_ABI" && "$EXPECTED_ABI" != "$PY_TAG" ]]; then
  die "轮子针对 ${EXPECTED_ABI} 编译，但当前解释器是 ${PY_TAG}。
     请改用匹配的 Python，或回到联网机器用 PY_VER=${PY_FULL} 重新下载。"
fi
info "轮子 ABI 标签：${EXPECTED_ABI:-未发现二进制轮子（全部为纯 Python）}"

# 平台检查
if [[ "$(uname -s)" == "Linux" ]]; then
  MACH="$(uname -m)"
  info "平台 $(uname -s) ${MACH}"
  if [[ "$MACH" != "aarch64" && "$MACH" != "arm64" ]]; then
    if find "${WHEEL_DIRS[@]}" -maxdepth 1 -name '*aarch64*.whl' | grep -q .; then
      warn "当前机器是 ${MACH}，但轮子是 aarch64 的 —— 本机装不上，属预期（请在内网 ARM64 机器执行本脚本）"
    fi
  fi
fi

# --------------------------------------------------------------------------- #
# 建 venv
# --------------------------------------------------------------------------- #
if [[ "$USE_SYSTEM" == "1" ]]; then
  PIP="$PY_BIN -m pip"
  info "直接安装到当前环境"
else
  if [[ ! -d "$VENV" ]]; then
    info "创建虚拟环境 ${VENV}"
    "$PY_BIN" -m venv "$VENV" || die "创建 venv 失败（Debian/Ubuntu 可能需先 apt install python3-venv）"
  fi
  PIP="${VENV}/bin/pip"
  info "使用虚拟环境 ${VENV}"
fi

# --------------------------------------------------------------------------- #
# 可选：先离线升级 pip。旧 pip 可能不认识较新的 manylinux 标签。
# --------------------------------------------------------------------------- #
if [[ "$BOOTSTRAP" == "1" ]]; then
  info "离线升级 pip / setuptools / wheel"
  $PIP install --no-index "${FIND_LINKS[@]}" --upgrade pip setuptools wheel
fi

info "当前 pip 版本：$($PIP --version)"

# --------------------------------------------------------------------------- #
# 安装。先试「带依赖解析」，失败则回退为「只装框架本体」并提示缺什么。
# --------------------------------------------------------------------------- #
install_target() {
  $PIP install --no-index "${FIND_LINKS[@]}" "$@" --no-cache-dir
}

if install_target lm-eval 2>/tmp/install_lmeval.err; then
  INSTALLED="lm-eval"
elif install_target evalscope 2>/tmp/install_evalscope.err; then
  INSTALLED="evalscope"
else
  warn "带依赖解析失败，错误摘要："
  sed -n '1,6p' /tmp/install_lmeval.err >&2 || true
  warn "尝试只安装框架本体（--no-deps），缺什么再单独补："
  if install_target lm-eval --no-deps; then
    INSTALLED="lm-eval（--no-deps，依赖需手工补）"
  elif install_target evalscope --no-deps; then
    INSTALLED="evalscope（--no-deps，依赖需手工补）"
  else
    die "安装失败。请检查 wheel 目录是否完整，或用 --no-deps 手工逐个安装。"
  fi
fi

info "已安装：${INSTALLED}"

# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
echo
info "依赖完整性检查（pip check）"
if $PIP check; then
  info "依赖关系自洽"
else
  warn "pip check 报告了缺失或冲突 —— 这通常意味着下载轮子时漏了包。
       在联网机器上重新执行 download_wheels.sh，或按提示的包名单独下载后放入 wheel 目录。"
fi

echo
info "已安装的关键包"
if ! $PIP list 2>/dev/null | grep -iE '^(lm[-_]eval|evalscope|torch|transformers) '; then
  info "  （未发现 lm-eval / evalscope / torch / transformers）"
fi

if $PIP list 2>/dev/null | grep -qiE '^torch '; then
  warn "检测到 torch 被安装。评测 API 端点并不需要它，包体却是 GB 级 ——
       通常意味着下载时误装了 [hf] extra。请回到联网机器核对包清单。"
fi

echo
info "框架导入自检"
PYTHON_BIN="$PY_BIN"
[[ "$USE_SYSTEM" == "1" ]] || PYTHON_BIN="${VENV}/bin/python"
if "$PYTHON_BIN" -c "import lm_eval; print('lm_eval', lm_eval.__version__)" 2>/dev/null; then
  info "lm_eval 可导入"
elif "$PYTHON_BIN" -c "import evalscope; print('evalscope', evalscope.__version__)" 2>/dev/null; then
  info "evalscope 可导入"
else
  warn "框架模块导入失败。先看上面的 pip check 输出，按缺失的包名补下轮子。"
fi

cat <<EOF

──────────────────────────────────────────────────────────────
安装完成。接下来在 config 里把模型指向内网的 OpenAI 兼容端点，然后：

  # lm-eval（acc / acc_norm 口径，与公开榜单对齐）
  ${VENV}/bin/lm_eval --help

  # 注意两个 backend 的区别：
  #   local-completions       走 /v1/completions，支持 loglikelihood -> 能出 acc_norm
  #   local-chat-completions  走 /v1/chat/completions，只支持生成式任务
  ${VENV}/bin/lm_eval --model local-completions \\
      --model_args model=<模型名>,base_url=http://127.0.0.1:8001/v1/completions,num_concurrent=8 \\
      --tasks ceval-valid,mmlu,gsm8k --num_fewshot 5 --output_path ./lmeval_out

  详细说明与对齐流程见 docs/OFFLINE_DEPLOY.md 与 docs/EVAL_METHODOLOGY.md
──────────────────────────────────────────────────────────────
EOF
