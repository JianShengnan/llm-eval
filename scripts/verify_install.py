#!/usr/bin/env python
"""核对内网项目里的框架代码是否与最新交付包一致。

用途：拿到新版本的源码包后，先跑这个脚本，它会告诉你**哪些文件需要更新**，
而不是靠猜「我上次拷的是哪一版」。

在**内网机器、项目根目录**执行：
    python3 scripts/verify_install.py
    python3 scripts/verify_install.py --manifest /path/to/CODE_MANIFEST.txt
    python3 scripts/verify_install.py --quiet      # 只打印不一致项

退出码：0 = 框架代码全部一致；1 = 有文件需更新或缺失。

分类说明：
    code / script  框架本体，必须与清单一致（不一致说明内网是旧版本）
    doc            文档与配置模板，建议同步，不影响运行
    data           数据集，可能是你自己准备或修改过的，只提示不判定
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 清单里这些类别视为「框架本体」，不一致就要更新
CORE_KINDS = {"code", "script"}
# 这些类别只提示，不参与一致性判定
SOFT_KINDS = {"data"}

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def _color(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{RESET}" if enabled else text


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_manifest(explicit: Optional[str]) -> Optional[str]:
    """按优先级定位清单：显式参数 -> 脚本同级 -> 项目根 -> 常见解压目录。"""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    cands = [
        os.path.join(HERE, "CODE_MANIFEST.txt"),
        os.path.join(ROOT, "CODE_MANIFEST.txt"),
        os.path.join(ROOT, "dist", "CODE_MANIFEST.txt"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def parse_manifest(path: str) -> List[Tuple[str, int, str, str]]:
    """返回 [(sha256, size, kind, relpath)]，跳过注释与空行。"""
    out: List[Tuple[str, int, str, str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            digest, size, kind, rel = parts[0], parts[1], parts[2], " ".join(parts[3:])
            try:
                n = int(size)
            except ValueError:
                n = -1
            out.append((digest, n, kind, rel))
    return out


def check(entries: List[Tuple[str, int, str, str]]) -> Dict[str, List]:
    result: Dict[str, List] = {"same": [], "diff": [], "missing": [], "soft": []}
    for digest, size, kind, rel in entries:
        local = os.path.join(ROOT, rel.replace("/", os.sep))
        if kind in SOFT_KINDS:
            result["soft"].append((rel, os.path.isfile(local)))
            continue
        if not os.path.isfile(local):
            result["missing"].append((rel, kind, digest))
            continue
        try:
            got = sha256_of(local)
        except OSError as exc:
            result["missing"].append((rel, kind, f"读取失败：{exc}"))
            continue
        if got == digest:
            result["same"].append((rel, kind))
        else:
            result["diff"].append((rel, kind, got, digest))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="核对框架代码与最新交付包的一致性")
    ap.add_argument("--manifest", help="CODE_MANIFEST.txt 路径（默认自动查找）")
    ap.add_argument("--quiet", action="store_true", help="只打印不一致项与结论")
    ap.add_argument("--no-color", action="store_true", help="不使用颜色（管道输出时用）")
    args = ap.parse_args()

    use_color = sys.stdout.isatty() and not args.no_color

    man = find_manifest(args.manifest)
    if not man:
        print("找不到 CODE_MANIFEST.txt。", file=sys.stderr)
        print("它随源码包一起分发，通常位于脚本同级或项目根目录；", file=sys.stderr)
        print("也可用 --manifest <路径> 指定，或从交付包的 dist/ 目录拷贝过来。", file=sys.stderr)
        return 2

    entries = parse_manifest(man)
    if not entries:
        print(f"清单为空或格式异常：{man}", file=sys.stderr)
        return 2

    res = check(entries)
    core_diff = [x for x in res["diff"] if x[1] in CORE_KINDS]
    core_missing = [x for x in res["missing"] if x[1] in CORE_KINDS]

    print(f"清单：{os.path.relpath(man, ROOT) if man.startswith(ROOT) else man}")
    print(f"项目根：{ROOT}")
    print()

    unchanged = [r for r, k in res["same"] if k in CORE_KINDS]
    if core_diff or core_missing or not args.quiet:
        print("框架代码（不一致即说明内网是旧版本，需要更新）")
        print("-" * 60)
    if core_diff or core_missing:
        for rel, _kind, got, want in core_diff:
            print(_color("  [需更新] ", YELLOW, use_color) + rel)
            print(f"           本地 {got[:16]}…  ≠  期望 {want[:16]}…")
        for rel, _kind, extra in core_missing:
            print(_color("  [缺失]   ", RED, use_color) + rel + f"  （{extra}）")
        if unchanged and not args.quiet:
            shown = ", ".join(unchanged[:6]) + ("…" if len(unchanged) > 6 else "")
            print(_color(f"  [一致] 其余 {len(unchanged)} 个：{shown}", DIM, use_color))
    elif not args.quiet:
        print(_color(f"  [一致] {len(unchanged)} 个框架文件全部与清单相同", GREEN, use_color))

    # 文档与配置
    docs_diff = [r for r, k, *_ in res["diff"] if k == "doc"]
    docs_same = [r for r, k in res["same"] if k == "doc"]
    docs_missing = [r for r, k, *_ in res["missing"] if k == "doc"]
    if docs_diff or docs_missing or (docs_same and not args.quiet):
        print()
        print("文档与配置模板（建议同步，不影响运行）")
        print("-" * 60)
        for r in docs_diff:
            print(_color("  [需更新] ", YELLOW, use_color) + r)
        for r in docs_missing:
            print(_color("  [缺失]   ", RED, use_color) + r)
        if docs_same and not args.quiet:
            shown = ", ".join(docs_same[:6]) + ("…" if len(docs_same) > 6 else "")
            print(_color(f"  [一致] {len(docs_same)} 个：{shown}", DIM, use_color))

    # 数据集：清单里若有 data 条目就逐个报状态；否则看本地目录是否存在，打消「会不会覆盖我的数据」的疑虑
    local_ds = os.path.join(ROOT, "datasets")
    if res["soft"] or os.path.isdir(local_ds):
        print()
        print("数据集（你自己的内容，不做判定）")
        print("-" * 60)
        for rel, exists in res["soft"]:
            print(f"  {rel:<52} {'存在' if exists else '未准备'}")
        if not res["soft"]:
            n = sum(len(fs) for _d, _s, fs in os.walk(local_ds))
            print(f"  本地 datasets/ 下 {n} 个文件（不在本清单范围内）")
        print(_color("  源码包刻意排除了 datasets/，覆盖源码不会动到你的数据", DIM, use_color))

    # 结论
    print()
    print("=" * 60)
    core_bad = [r for r, k, *_ in res["diff"] if k in CORE_KINDS] + [
        r for r, k, *_ in res["missing"] if k in CORE_KINDS
    ]
    if not core_bad:
        print(_color(f"[OK] 框架代码全部一致（{len(unchanged)} 个文件）", GREEN, use_color))
        if docs_diff or docs_missing:
            print("     仅文档有差异，可选同步。")
        return 0

    print(_color(f"[需更新] {len(core_bad)} 个框架文件与最新版本不一致", YELLOW, use_color))
    for r in core_bad[:10]:
        print(f"    - {r}")
    if len(core_bad) > 10:
        print(f"    … 另外 {len(core_bad) - 10} 个")
    print()
    print("覆盖方式（在项目根目录，路径按实际调整）：")
    print("  tar xzf <源码包>.tar.gz")
    print("  cp -r <包名>/geval <包名>/scripts <包名>/docs /path/to/项目/")
    print("  python3 scripts/verify_install.py    # 再验一次，应全部一致")
    print("  python3 -m geval selftest")
    return 1


if __name__ == "__main__":
    sys.exit(main())
