#!/usr/bin/env python
"""把整个项目（含离线 wheel）打成**一个文件**，便于拷进内网，并生成 sha256 清单。

用法（在联网机器上、项目根目录执行）：
    python scripts/pack_bundle.py                    # 生成 dist/llm-general-eval-<日期>.tar.gz
    python scripts/pack_bundle.py --name myeval      # 自定义文件名
    python scripts/pack_bundle.py --no-wheels        # 不含 wheel（只拷框架代码与数据）
    python scripts/pack_bundle.py --code-only        # ★ 仅框架代码+文档，用于增量更新内网
    python scripts/pack_bundle.py --list-only         # 只生成清单，不打包

产物：
    dist/<name>.tar.gz
    dist/<name>.sha256              归档本身的校验值
    dist/WHEELS_MANIFEST.txt        每个 wheel 的大小与 sha256（交接与核对用）
    dist/CODE_MANIFEST.txt          源码/文档的 sha256 清单，内网用 verify_install.py 比对

关于 --code-only：源码改动后不必重传 133 MB 的全量包。这个模式只打包
geval/ scripts/ docs/ 与顶层说明文件，**刻意排除 datasets/ 与 wheels/**，
所以覆盖上去不会碰到你在内网准备的数据集、config.json、venv 和 runs/。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import time
from typing import Iterable, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 打进包里：代码、数据、文档、脚本、离线 wheel
INCLUDE_DIRS = ["geval", "docs", "scripts", "datasets", "wheels"]
# --code-only 模式：只含框架与文档，排除 datasets / wheels
CODE_ONLY_DIRS = ["geval", "docs", "scripts"]
INCLUDE_FILES = [
    "README.md",
    "HANDOVER.md",
    "requirements.txt",
    "config.example.json",
    "config.smoke.json",
    "config.professional.json",
    ".gitignore",
]
# 排除：运行产物、缓存、下载日志、Python 缓存
EXCLUDE_PARTS = {"__pycache__", ".cache", "runs", "dist", ".git"}
EXCLUDE_SUFFIX = (".pyc", ".pyo")
EXCLUDE_NAMES = {"_lmeval_download.log", "_lm-eval_download.log"}


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def iter_files(include_wheels: bool, code_only: bool = False) -> Iterable[str]:
    for name in INCLUDE_FILES:
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            yield p
    for d in (CODE_ONLY_DIRS if code_only else INCLUDE_DIRS):
        if d == "wheels" and not include_wheels:
            continue
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in EXCLUDE_PARTS]
            for fn in filenames:
                if fn.endswith(EXCLUDE_SUFFIX) or fn in EXCLUDE_NAMES:
                    continue
                if not include_wheels and dirpath.startswith(os.path.join(ROOT, "wheels")):
                    continue
                yield os.path.join(dirpath, fn)


def classify(rel: str) -> str:
    """给清单里的文件分类，内网比对时用于区分「必须一致」与「可自行保留」。"""
    if rel.startswith("geval/"):
        return "code"
    if rel.startswith("scripts/"):
        return "script"
    if rel.startswith("datasets/"):
        return "data"
    return "doc"


def write_code_manifest(files: List[str], out_dir: str, mode: str) -> str:
    """生成源码/文档 sha256 清单，供内网 verify_install.py 逐文件比对。"""
    entries = [f for f in files if not f.endswith(".whl")]
    path = os.path.join(out_dir, "CODE_MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# 框架源码与文档清单（不含 wheel）\n")
        fh.write(f"# 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"# 打包模式：{mode}\n")
        fh.write("# 列：sha256  大小(字节)  类别  相对路径\n")
        fh.write("#   类别 code/script 为框架本体，内网必须与本清单一致；\n")
        fh.write("#        doc 为文档与配置模板，建议同步但不影响运行；\n")
        fh.write("#        data 为数据集，可能是你自己准备的，不做判定。\n")
        fh.write("# 用法（内网机、项目根目录）：python3 scripts/verify_install.py\n\n")
        for f in entries:
            rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
            fh.write(f"{sha256_of(f)}  {os.path.getsize(f):>9}  {classify(rel):<7}  {rel}\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="打包交付件 + 生成 sha256 清单")
    ap.add_argument("--name", default=time.strftime("llm-general-eval-%Y%m%d"))
    ap.add_argument("--no-wheels", action="store_true", help="不打包 wheels/ 目录")
    ap.add_argument("--code-only", action="store_true", help="仅打包框架代码与文档（增量更新用，排除 datasets/ 与 wheels/）")
    ap.add_argument("--list-only", action="store_true", help="只生成清单，不打包")
    ap.add_argument("--out", default="dist", help="输出目录，默认 dist")
    args = ap.parse_args()

    include_wheels = not args.no_wheels and not args.code_only
    files = sorted(set(iter_files(include_wheels, code_only=args.code_only)))
    if not files:
        print("没有找到任何文件，确认在项目根目录执行本脚本", file=sys.stderr)
        return 1

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    man_path = write_code_manifest(
        files,
        out_dir,
        "仅源码（不含 datasets/ 与 wheels/）" if args.code_only else "全量（含预下载 wheel）",
    )
    n_code = len([f for f in files if not f.endswith(".whl")])
    print(f"源码清单已生成：{os.path.relpath(man_path, ROOT)}（{n_code} 个文件）")

    # 清单也放进包里，解压后即可直接运行 verify_install.py（清单自身不列入条目，避免自指）
    if not args.list_only:
        files = sorted(set(files + [man_path]))

    # ---- wheel 清单（交接与核对用）----
    wheels = [f for f in files if f.endswith(".whl")]
    if wheels:
        # 注意：变量名不要复用 man_path —— 上面那个是源码清单，打包循环还要用它比对
        wheel_man = os.path.join(out_dir, "WHEELS_MANIFEST.txt")
        total = 0
        with open(wheel_man, "w", encoding="utf-8") as fh:
            fh.write("# 离线 wheel 清单\n")
            fh.write(f"# 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write("# 列：sha256  大小(字节)  平台类别  文件名\n")
            fh.write("#   平台类别 pure = 纯 Python（跨平台通用）；aarch64 = ARM64 二进制\n\n")
            for w in wheels:
                size = os.path.getsize(w)
                total += size
                kind = "pure" if w.endswith("-none-any.whl") else ("aarch64" if "aarch64" in os.path.basename(w) else "UNKNOWN")
                fh.write(f"{sha256_of(w)}  {size:>12}  {kind:<8}  {os.path.relpath(w, ROOT)}\n")
            fh.write(f"\n# 合计 {len(wheels)} 个 wheel，{total} 字节（{human(total)}）\n")
        print(f"wheel 清单已生成：{os.path.relpath(wheel_man, ROOT)}")

    if args.list_only:
        print(f"共 {len(files)} 个文件，{human(sum(os.path.getsize(f) for f in files))}")
        return 0

    # ---- 打包 ----
    archive = os.path.join(out_dir, args.name + ".tar.gz")
    print(f"正在打包 {len(files)} 个文件…")
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for path in files:
            if path == man_path:
                # 清单在 dist/ 下生成，放进包根而不是 dist/ 子目录
                arcname = os.path.join(args.name, "CODE_MANIFEST.txt")
            else:
                arcname = os.path.join(args.name, os.path.relpath(path, ROOT))
            tar.add(path, arcname=arcname)

    size = os.path.getsize(archive)
    digest = sha256_of(archive)
    sha_path = os.path.join(out_dir, args.name + ".sha256")
    with open(sha_path, "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  {os.path.basename(archive)}\n")

    print()
    print("=" * 62)
    print(f"  归档    {os.path.relpath(archive, ROOT)}")
    print(f"  大小    {human(size)}")
    print(f"  sha256  {digest}")
    print(f"  校验值  {os.path.relpath(sha_path, ROOT)}")
    print("=" * 62)
    print()
    print("拷进内网后在目标机上执行：")
    print(f"  sha256sum -c {os.path.basename(archive)}.sha256   # 先核对完整性")
    if args.code_only:
        print(f"  tar xzf {os.path.basename(archive)}")
        print(f"  # 用包内源码覆盖到现有项目（不碰 datasets/ config.json venv/ runs/）")
        print(f"  cp -r {args.name}/geval {args.name}/scripts {args.name}/docs /path/to/项目/")
        print("  python3 scripts/verify_install.py   # 逐文件核对，应显示全部一致")
        print("  python3 -m geval selftest")
    else:
        print(f"  tar xzf {os.path.basename(archive)}")
        print(f"  cd {args.name}")
        print("  python3 scripts/verify_deps.py")
        print("  python3 -m geval selftest")
    print("  详见 docs/INTRANET_RUNBOOK.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
