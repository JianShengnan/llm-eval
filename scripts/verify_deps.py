#!/usr/bin/env python
"""依赖核查：确认 geval 在离线环境下真的零第三方依赖。

做法：用 ast 解析 geval/ 与 scripts/ 下所有 .py 的顶层 import，
      与 sys.stdlib_module_names 比对，列出所有非标准库模块。

用法：
    python scripts/verify_deps.py            # 检查 geval 运行时（应为空）
    python scripts/verify_deps.py --strict    # 有任何第三方依赖则以非零码退出
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIRS = ["geval"]
TOOL_DIRS = ["scripts", "tests"]


def top_level_imports(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    mods: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # 相对导入，属于本包
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods


def scan(dirs) -> dict:
    found: dict = {}
    for d in dirs:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                for mod in top_level_imports(path):
                    found.setdefault(mod, set()).add(os.path.relpath(path, ROOT))
    return found


def classify(mods: dict) -> tuple:
    stdlib = set(sys.stdlib_module_names)
    local = {"geval"}
    third = {m: f for m, f in mods.items() if m not in stdlib and m not in local}
    return third, sorted(m for m in mods if m in stdlib)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="存在第三方依赖时返回非零码")
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]} ｜ 标准库清单条目 {len(sys.stdlib_module_names)}\n")

    runtime = scan(RUNTIME_DIRS)
    third, std = classify(runtime)
    print(f"[运行时] geval/ 共引用 {len(std)} 个标准库模块")
    print("  " + ", ".join(std))
    if third:
        print("\n[运行时] 发现第三方依赖：")
        for mod, files in sorted(third.items()):
            print(f"  - {mod}  <- {', '.join(sorted(files))}")
    else:
        print("\n[运行时] 第三方依赖：无 -> 内网离线可直接运行")

    tools = scan(TOOL_DIRS)
    third_tools, _ = classify(tools)
    if third_tools:
        print("\n[工具脚本] 需要额外安装（不影响内网运行）：")
        for mod, files in sorted(third_tools.items()):
            print(f"  - {mod}  <- {', '.join(sorted(files))}")
        print("  说明：这些脚本只在联网机器上用于准备数据集，内网机器不需要执行。")

    if args.strict and third:
        print("\n[FAIL] 运行时存在第三方依赖")
        return 1
    print("\n[OK] 运行时依赖核查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
