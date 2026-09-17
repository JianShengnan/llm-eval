#!/usr/bin/env bash
# 在**联网机器**上执行：为目标内网环境（Linux ARM64 / aarch64 + Python 3.11）下载全部 wheel。
#
# 用法：
#   bash scripts/download_wheels.sh lm-eval        # 推荐：lm-eval-harness（最轻，无 torch）
#   bash scripts/download_wheels.sh evalscope      # 备选：EvalScope（含 transformers，较重）
#   bash scripts/download_wheels.sh all
#
# 环境变量：
#   PY_VER=311            目标 Python 版本（写成 3.11 或 311 都行）
#   PLATFORM=manylinux2014_aarch64
#                        目标平台标签。glibc ≥ 2.17 用 manylinux2014_aarch64；
#                        较新的发行版也可用 manylinux_2_28_aarch64
#   OUT=wheels            输出目录
#   USE_DOCKER=1          在 ARM64 容器内下载（最可靠，见 docs/OFFLINE_DEPLOY.md）
#
# 本脚本会处理一个已知的真实坑：
#   lm-eval 依赖 sqlitedict，而 sqlitedict 在 PyPI 上**只发布 sdist、没有 wheel**。
#   `pip download --only-binary=:all:` 会因此直接失败，报错信息还很绕。
#   脚本会自动识别这类包，本地构建 wheel，校验产物必须是纯 Python，然后重试。

set -euo pipefail

PY_VER="${PY_VER:-311}"
PLATFORM="${PLATFORM:-manylinux2014_aarch64}"
OUT="${OUT:-wheels}"
PROFILE="${1:-lm-eval}"
USE_DOCKER="${USE_DOCKER:-0}"
MAX_HEAL_ROUNDS="${MAX_HEAL_ROUNDS:-3}"

# lm-eval 必须**显式 pin 版本**。踩过的坑：不 pin 时 pip 会因为某个依赖找不到
# wheel 而静默回退到旧版本，而旧版本（0.3.0）把 torch 当核心依赖 ——
# 白下 90MB 的 torch，而且依赖树与预期完全不同。
LMEVAL_VERSION="${LMEVAL_VERSION:-0.4.13}"

# 上游只发布 sdist、没有 wheel 的包（实测确认，lm-eval 依赖树里至少这三个）。
# pip 的 --only-binary=:all: 遇到它们会失败，而且是「静默回退到旧版本」的诱因，
# 所以提前本地构建 wheel 补齐。注意：这个清单只是省一轮重试，
# 真正兜底的是 download_one 里的自愈循环 —— 未知的 sdist-only 包也能被发现。
KNOWN_SDIST_ONLY="${KNOWN_SDIST_ONLY:-sqlitedict word2number langdetect}"

# 接受 "3.11" 与 "311" 两种写法
_raw="${PY_VER//./}"
if [[ ! "$_raw" =~ ^3[0-9][0-9]$ ]]; then
  echo "PY_VER 请写成 3.11 或 311 的形式，当前为：${PY_VER}" >&2
  exit 1
fi
PY_VER="$_raw"
PY_DOT="${PY_VER:0:1}.${PY_VER:1}"
ABI="cp${PY_VER}"

info() { printf '\033[36m[info]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

list_wheels() { find "$1" -maxdepth 1 -name '*.whl' 2>/dev/null | sort; }

# --------------------------------------------------------------------------- #
# 每个 profile 的包清单
#
# lm-eval 的 [api] extra 是评测 OpenAI 兼容端点所必需的；
# [ifeval] 提供 langdetect / immutabledict / nltk，IFEval 任务需要；
# 刻意**不装** [hf] extra —— 它才会引入 torch（GB 级），而本项目只走 API 评测。
# --------------------------------------------------------------------------- #
packages_for() {
  case "$1" in
    lm-eval)   echo "lm-eval[api,ifeval]==${LMEVAL_VERSION}" ;;
    evalscope) echo 'evalscope' ;;
    # 目标机 venv 的引导工具。较新的 manylinux 标签需要新 pip 才能识别，
    # 内网又装不了 pip，所以一并带上。
    bootstrap) echo 'pip setuptools wheel' ;;
    *) die "未知 profile：$1（可选 lm-eval / evalscope / bootstrap / all）" ;;
  esac
}

# 从 wheel 文件名里取版本号，例如 lm_eval-0.4.13-py3-none-any.whl -> 0.4.13
wheel_version() { basename "$1" | sed -E 's/\.whl$//' | awk -F- '{print $2}'; }

# 下载完成后校验：版本必须是 pin 的那个，且绝不接受 torch
verify_download() {
  local profile="$1" dest="$2"
  local whl ver

  if [[ "$profile" == "lm-eval" ]]; then
    whl="$(list_wheels "$dest" | grep -E '/lm_eval-' | head -1)"
    if [[ -z "$whl" ]]; then
      die "没有下载到 lm-eval 本体，详见 ${OUT}/_${profile}_download.log"
    fi
    ver="$(wheel_version "$whl")"
    if [[ "$ver" != "$LMEVAL_VERSION" ]]; then
      die "版本不符：期望 lm-eval ${LMEVAL_VERSION}，实际拿到 ${ver}。
     这通常意味着 pip 因为某个依赖缺 wheel 而静默回退到了旧版本。请检查上面的下载日志。"
    fi

    # torch 只可能来自 lm-eval 的 [hf] extra 或旧版本的核心依赖，两者都不该出现
    if list_wheels "$dest" | grep -qE '/torch-[0-9]'; then
      die "下载结果里出现了 torch —— 说明解析出了错误的依赖树（很可能是版本回退）。
     评测 API 端点不需要 torch，请检查下载日志。"
    fi
    info "  版本校验通过：lm-eval ${ver}（无 torch）"
  fi
}

# --------------------------------------------------------------------------- #
# 从 pip 报错里解析「没有可用 wheel」的包名
# 匹配段落：some packages ... have no matching distributions available for your environment:
#              black
#              sqlitedict
# --------------------------------------------------------------------------- #
parse_sdist_only() {
  {
    # 格式一：冲突汇总段落
    awk '
      /no matching distributions available/ { flag = 1; next }
      flag && /^[[:space:]]+[A-Za-z0-9_.+-]+[[:space:]]*$/ {
        gsub(/^[[:space:]]+/, ""); gsub(/[[:space:]]+$/, ""); print $0; next
      }
      flag && NF && $0 !~ /^[[:space:]]/ { flag = 0 }
    ' "$1"
    # 格式二：pin 版本时的直接报错（更精确，只在这一个包上出现）
    #   ERROR: Could not find a version that satisfies the requirement word2number
    #          (from lm-eval) (from versions: none)
    grep -oE 'Could not find a version that satisfies the requirement [A-Za-z0-9_.+-]+' "$1" \
      | awk '{print $NF}'
    # 格式三：No matching distribution found for word2number
    grep -oE 'No matching distribution found for [A-Za-z0-9_.+-]+' "$1" \
      | awk '{print $NF}'
  } | grep -vE '^(lm[-_]eval|lm-eval)$' | sort -u
}

# 为「只有 sdist」的包本地构建 wheel，并强制校验产物是纯 Python
build_pure_wheels() {
  local dest="$1"; shift
  [[ $# -gt 0 ]] || return 0

  local before after f pkg bad=0
  before="$(list_wheels "$dest")"

  for pkg in "$@"; do
    info "  本地构建 wheel：${pkg}（上游只提供 sdist，没有现成 wheel）"
    if ! python3 -m pip wheel "$pkg" --wheel-dir "$dest" --no-deps --no-cache-dir >/dev/null 2>&1; then
      die "  构建 ${pkg} 失败。该包可能不是纯 Python，需要目标平台的编译环境。"
    fi
  done

  after="$(list_wheels "$dest")"
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    case "$before" in *"$f"*) continue ;; esac
    if [[ "$f" != *"-none-any.whl" ]]; then
      warn "  ✗ 构建产物 $(basename "$f") 不是纯 Python 轮子"
      bad=1
    else
      info "  ✓ $(basename "$f")（纯 Python，跨平台可用）"
    fi
  done <<< "$after"

  if [[ "$bad" != "0" ]]; then
    die "存在平台相关的构建产物 —— 在 $(uname -s)/$(uname -m) 上编译出来的轮子无法用于 aarch64。
     请改用 USE_DOCKER=1 在 ARM64 容器里下载，或直接在 ARM64 机器上执行本脚本。"
  fi
}

# --------------------------------------------------------------------------- #
download_one() {
  local profile="$1"
  local pkgs="$2"
  [[ -n "$pkgs" ]] || die "profile ${profile} 没有对应的包清单"
  local dest="${OUT}/${profile}"
  mkdir -p "$dest"

  info "profile=${profile}  python=${PY_DOT}  platform=${PLATFORM}  -> ${dest}"

  if [[ "$USE_DOCKER" == "1" ]]; then
    command -v docker >/dev/null || die "未找到 docker"
    info "使用 ARM64 容器下载（最可靠，但依赖 qemu 模拟，速度较慢）"
    docker run --rm --platform linux/arm64 \
      -v "$(cd "$dest" && pwd)":/wheels \
      "python:${PY_DOT}-slim" \
      pip download $pkgs --dest /wheels --only-binary=:all: --no-cache-dir
  else
    local log="${OUT}/_${profile}_download.log"
    local round=0 healed_any=0

    # 先把已知的 sdist-only 包本地构建好，省掉一轮失败重试
    if [[ -n "$KNOWN_SDIST_ONLY" ]]; then
      # shellcheck disable=SC2086
      build_pure_wheels "$dest" $KNOWN_SDIST_ONLY
    fi

    while :; do
      round=$((round + 1))
      set +e
      python3 -m pip download $pkgs \
        --dest "$dest" \
        --only-binary=:all: \
        --platform "$PLATFORM" \
        --python-version "$PY_DOT" \
        --implementation cp \
        --abi "$ABI" \
        --find-links "$dest" \
        --no-cache-dir > "$log" 2>&1
      local rc=$?
      set -e

      [[ "$rc" == "0" ]] && break

      # 自愈：识别「只有 sdist」的包，本地构建 wheel 后重试
      local sdist_only
      sdist_only="$(parse_sdist_only "$log" || true)"
      if [[ -z "$sdist_only" ]]; then
        warn "pip 下载失败，且未识别出「只有 sdist」的包。错误摘要："
        grep -E 'ERROR|error:|WARNING' "$log" | head -8 >&2 || true
        die "详见 $log"
      fi
      if [[ "$round" -gt "$MAX_HEAL_ROUNDS" ]]; then
        die "已尝试 ${MAX_HEAL_ROUNDS} 轮仍未成功，详见 $log"
      fi

      warn "第 ${round} 轮：发现上游只有 sdist 的包 -> $(echo "$sdist_only" | tr '\n' ' ')"
      # shellcheck disable=SC2086
      build_pure_wheels "$dest" $sdist_only
      healed_any=1
      info "  已补齐，重试下载…"
    done

    if [[ "$healed_any" == "1" ]]; then
      info "  本轮有包通过本地构建补齐（日志 $log）"
    fi
    verify_download "$profile" "$dest"
  fi

  local n
  n=$(list_wheels "$dest" | wc -l | tr -d ' ')
  info "  下载完成：${n} 个 wheel，合计 $(du -sh "$dest" | cut -f1)"
}

summarize() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  info "检查 ${dir} 中的平台标签："
  local bad=0 f base
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    base="$(basename "$f")"
    if [[ "$base" == *"-none-any.whl" ]]; then
      printf '        纯 Python（跨平台通用）  %s\n' "$base"
    elif [[ "$base" == *"aarch64"* ]]; then
      printf '        ARM64 二进制                %s\n' "$base"
    else
      printf '\033[31m        平台不匹配！             %s\033[0m\n' "$base"
      bad=$((bad + 1))
    fi
  done <<< "$(list_wheels "$dir")"
  if [[ "$bad" -gt 0 ]]; then
    die "有 ${bad} 个 wheel 的平台标签不是 aarch64 / none-any，拷进内网会安装失败"
  fi
}

# --------------------------------------------------------------------------- #
main() {
  command -v python3 >/dev/null || die "未找到 python3"

  # 先在父进程里校验 profile：若放进命令替换里判断，
  # die 只会终止子 shell，父进程会带着空包清单继续跑 pip download。
  case "$PROFILE" in
    all | lm-eval | evalscope | bootstrap) ;;
    *) die "未知 profile：$PROFILE（可选 lm-eval / evalscope / bootstrap / all）" ;;
  esac

  case "$PROFILE" in
    all)
      for p in bootstrap lm-eval evalscope; do download_one "$p" "$(packages_for "$p")"; done
      ;;
    *)
      download_one "$PROFILE" "$(packages_for "$PROFILE")"
      ;;
  esac

  for d in "${OUT}"/*/; do summarize "${d%/}"; done

  cat <<EOF

──────────────────────────────────────────────────────────────
下载完成。下一步：

1) 打包并拷进内网：
     tar czf wheels.tar.gz ${OUT}
     # 拷贝 wheels.tar.gz 到内网机器

2) 内网机器上安装：
     bash scripts/install_offline.sh ${OUT}/bootstrap ${OUT}/lm-eval --venv /opt/evalenv --bootstrap

3) 校验：
     /opt/evalenv/bin/pip check
     /opt/evalenv/bin/lm_eval --help

注意事项：
  · 目标机的 Python 必须是 ${PY_DOT}（cp${PY_VER} ABI）。版本不一致会导致装不上。
  · 数据集也要离线，只装框架是不够的 —— 见 docs/OFFLINE_DEPLOY.md 坑 1。
  · 环境标记（sys_platform / platform_machine 等）是按**当前机器**解析的，
    跨平台下载在某些包上可能解析出错误的可选依赖。装完后务必在内网跑 pip check。
──────────────────────────────────────────────────────────────
EOF
}

main
