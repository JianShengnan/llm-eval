#!/usr/bin/env python
"""把主流公开基准转成 geval 需要的 JSONL（在**联网**机器上执行）。

为什么需要这一步：
  内网 ARM64 离线环境装不了 datasets 库、也拉不到外网数据集。
  所以在联网机器上跑一次本脚本，生成 JSONL，把 datasets/ 目录拷进内网即可。

用法：
    pip install datasets
    python scripts/prepare_datasets.py --list
    python scripts/prepare_datasets.py --task ceval  --out datasets/ceval  --split val
    python scripts/prepare_datasets.py --task cmmlu  --out datasets/cmmlu  --split test
    python scripts/prepare_datasets.py --task mmlu   --out datasets/mmlu   --split test
    python scripts/prepare_datasets.py --task gsm8k  --out datasets/gsm8k  --split test
    python scripts/prepare_datasets.py --task bbh    --out datasets/bbh    --split test
    python scripts/prepare_datasets.py --task ifeval --out datasets/ifeval --split train
    python scripts/prepare_datasets.py --all --root datasets

输出：
    <out>/test.jsonl   评测集
    <out>/dev.jsonl    few-shot 示例（若上游提供 dev / validation 切分）

字段说明（geval 会自动识别多种写法，无需手工整理）：
    题面：question / Question / query / input / prompt / problem
    选项：choices / options（列表或 dict），或 A/B/C/D 平铺字段
    答案：answer / Answer / target / label，允许 索引(0-3)、字母(A-D)、选项原文、数值
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

# task -> (HF 仓库, 是否按 subject 拆 config, 默认切分, 可用切分)
REGISTRY: Dict[str, Dict[str, Any]] = {
    "ceval": {
        "repo": "ceval/ceval-exam",
        "per_subject": True,
        "subject_splits": ["val", "dev"],
        "note": "52 个学科；val 为官方验证集（无公开答案的 test 不适用）",
    },
    "cmmlu": {
        "repo": "haonan-li/cmmlu",
        "per_subject": True,
        "subject_splits": ["test", "dev"],
        "note": "67 个学科，中文通识与专业",
    },
    "mmlu": {
        "repo": "cais/mmlu",
        "per_subject": False,
        "config": "all",
        "splits": {"test": "test", "dev": "dev", "val": "validation"},
        "note": "57 个学科，英文",
    },
    "mmlu_pro": {
        "repo": "TIGER-Lab/MMLU-Pro",
        "per_subject": False,
        "splits": {"test": "test", "val": "validation"},
        "note": "10 选项版本，区分度高于 MMLU",
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "per_subject": False,
        "config": "main",
        "splits": {"test": "test", "dev": "train"},
        "note": "小学数学应用题，答案形如 '... #### 42'",
    },
    "math": {
        "repo": "HuggingFaceH4/MATH-500",
        "per_subject": False,
        "splits": {"test": "test"},
        "note": "MATH-500 竞赛数学",
    },
    "bbh": {
        "repo": "lukaemon/bbh",
        "per_subject": True,
        "subject_splits": ["test"],
        "note": "BIG-Bench Hard，27 个子任务；字段为 input/target",
    },
    "arc": {
        "repo": "allenai/ai2_arc",
        "per_subject": False,
        "config": "ARC-Challenge",
        "splits": {"test": "test", "dev": "validation"},
        "note": "常识推理，英文",
    },
    "ifeval": {
        "repo": "google/IFEval",
        "per_subject": False,
        "splits": {"test": "train"},
        "note": "指令遵循；字段为 prompt / instruction_id_list / kwargs",
    },
}


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError:
        sys.exit(
            "缺少 datasets 库。请在联网机器执行：pip install datasets\n"
            "（内网机器不需要本脚本，只需要脚本产出的 JSONL 文件）"
        )


def _iter_rows(spec: Dict[str, Any], split: str) -> Iterable[Dict[str, Any]]:
    """逐个 config 迭代，把上游字段映射成 geval 的通用 JSONL 结构。"""
    from datasets import get_dataset_config_names, load_dataset

    repo = spec["repo"]
    if spec.get("per_subject"):
        subjects = list(spec.get("subjects") or get_dataset_config_names(repo))
        if not subjects:
            sys.exit(f"{repo} 没有发现任何 config；可能仓库名已变更，请核对")
        for subject in subjects:
            try:
                ds = load_dataset(repo, subject)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {subject}: {exc}", file=sys.stderr)
                continue
            if split not in ds:
                continue
            for i, row in enumerate(ds[split]):
                yield _convert(row, subject, i)
    else:
        config = spec.get("config")
        ds = load_dataset(repo, config) if config else load_dataset(repo)
        if split not in ds:
            print(f"  [warn] {repo} 没有 {split} 切分，可用：{list(ds.keys())}", file=sys.stderr)
            return
        for i, row in enumerate(ds[split]):
            yield _convert(row, row.get("subject") or row.get("category") or "", i)


def _convert(row: Dict[str, Any], subject: str, index: int) -> Dict[str, Any]:
    """尽量保留原始字段，交给 geval 的归一化层处理。"""
    out: Dict[str, Any] = {"id": row.get("id") or f"{subject or 'q'}-{index:05d}"}
    out["subject"] = str(subject or row.get("subject") or row.get("category") or "").strip()

    question = _first(row, ("question", "Question", "input", "prompt", "problem", "query"))
    if question is None:
        raise ValueError(f"样本缺少题面字段：{list(row.keys())}")
    out["question"] = str(question)

    choices = _first(row, ("choices", "options", "Choices", "Options"))
    if choices is None:
        letters = [c for c in "ABCDEFGHIJ" if c in row and isinstance(row.get(c), str)]
        if len(letters) >= 2:
            choices = [row[c] for c in letters]
    if choices is not None:
        out["choices"] = list(choices.values()) if isinstance(choices, dict) else list(choices)

    answer = _first(row, ("answer", "Answer", "target", "label", "answer_index"))
    if answer is not None:
        out["answer"] = answer

    # IFEval 专用
    if "instruction_id_list" in row:
        out["instruction_id_list"] = list(row["instruction_id_list"])
        out["kwargs"] = [dict(k) if isinstance(k, dict) else {} for k in (row.get("kwargs") or [])]

    if row.get("explanation"):
        out["explanation"] = str(row["explanation"])[:500]
    return out


def _first(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写出 {len(rows):>7} 条 -> {path}")


def prepare(task: str, out_dir: str, split: Optional[str]) -> None:
    spec = REGISTRY[task]
    if spec.get("per_subject"):
        eval_split = split or spec["subject_splits"][0]
        dev_split = spec["subject_splits"][1] if len(spec["subject_splits"]) > 1 else None
    else:
        splits = spec.get("splits") or {"test": "test"}
        eval_key = split or next(iter(splits))
        eval_split = splits.get(eval_key, eval_key)
        dev_split = None
        for k, v in splits.items():
            if k in ("dev", "val") and v != eval_split:
                dev_split = v
                break

    print(f"\n[{task}] {spec['repo']}  ({spec.get('note', '')})")
    rows = list(_iter_rows(spec, eval_split))
    if not rows:
        sys.exit(f"[{task}] 没有取到任何样本，请检查仓库名与切分名")
    write_jsonl(os.path.join(out_dir, f"{'test' if eval_split == 'test' else 'test'}.jsonl"), rows)

    if dev_split and dev_split != eval_split:
        dev_rows = list(_iter_rows(spec, dev_split))
        if dev_rows:
            write_jsonl(os.path.join(out_dir, "dev.jsonl"), dev_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="公开基准 -> geval JSONL")
    ap.add_argument("--list", action="store_true", help="列出支持的基准")
    ap.add_argument("--task", help="基准名")
    ap.add_argument("--all", action="store_true", help="转换全部（耗时且占空间大）")
    ap.add_argument("--out", help="输出目录")
    ap.add_argument("--root", default="datasets", help="--all 时的根目录")
    ap.add_argument("--split", help="指定切分（默认取各基准的标准评测切分）")
    args = ap.parse_args()

    if args.list or (not args.task and not args.all):
        print("支持的基准：\n")
        for name, spec in sorted(REGISTRY.items()):
            print(f"  {name:<10} {spec['repo']:<26} {spec.get('note', '')}")
        print("\n示例： python scripts/prepare_datasets.py --task ceval --out datasets/ceval")
        return 0

    _require_datasets()
    if args.all:
        for name in REGISTRY:
            prepare(name, os.path.join(args.root, name), None)
    else:
        if args.task not in REGISTRY:
            sys.exit(f"未知基准 {args.task}；用 --list 查看支持列表")
        prepare(args.task, args.out or os.path.join(args.root, args.task), args.split)

    print("\n完成。把整个 datasets/ 目录拷到内网机器，然后执行：")
    print("  python -m geval probe --config config.json")
    print("  python -m geval run   --config config.json --smoke --limit 20")
    print("  python -m geval run   --config config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
