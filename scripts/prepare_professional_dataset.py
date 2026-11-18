# -*- coding: utf-8 -*-
"""把业务标注平台导出的「语音导航意图识别评测数据集.jsonl」转成 geval 数据集格式。

用法（在项目根目录执行）：
    python scripts/prepare_professional_dataset.py
    python scripts/prepare_professional_dataset.py --limit 40 --out datasets/intent_routing_smoke

产出：
    datasets/intent_routing/test.jsonl   geval 归一化后可直接加载的样本
    datasets/intent_routing/labels.json  候选意图树路径 + 末级名称（幻觉意图判定依据）
    datasets/intent_routing/dataset_info.json  来源 sha256 / 样本量 / 字段口径

字段映射：
    question <- prompt            （自包含提示词：候选树 + 历史 + 当前句 + 小模型意图）
    answer   <- ground_truth      （人工标注意图名称，真实答案）
    subject  <- sample_type       （业务/FAQ意图 | 通用交互意图，用于分层）
    meta     <- 追溯与基线所需的最小字段（不重复存放 prompt，避免文件翻倍）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
DEFAULT_INPUT = os.path.join(WORKSPACE, "语音导航意图识别评测数据集.jsonl")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "datasets", "intent_routing")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(path: str, limit: int) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise SystemExit(
            "找不到业务数据集：%s\n"
            "  请先用工作区根目录的 build_dataset.py 生成，或用 --input 指定路径。" % path
        )
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print("  [warn] 第 %d 行 JSON 解析失败，已跳过：%s" % (i, exc), file=sys.stderr)
                continue
            for key in ("prompt", "ground_truth", "sample_id"):
                if not obj.get(key):
                    raise SystemExit("第 %d 行缺少必要字段 %s" % (i, key))
            rows.append(obj)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit("数据集为空：%s" % path)
    return rows


def convert(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        history = r.get("history") or []
        out.append({
            "id": str(r["sample_id"]),
            "question": str(r["prompt"]),
            "choices": None,
            "answer": str(r["ground_truth"]).strip(),
            "subject": str(r.get("sample_type") or "unknown"),
            "meta": {
                "sample_id": r.get("sample_id"),
                "session_id": r.get("session_id"),
                "turn_index": r.get("turn_index"),
                "source_row": r.get("source_row"),
                "current_utterance": r.get("current_utterance"),
                "small_model_intent": r.get("small_model_intent"),
                "human_intent_path": r.get("human_intent_path"),
                "is_invalid": r.get("is_invalid"),
                "sample_type": r.get("sample_type"),
                "ground_truth_in_tree": r.get("ground_truth_in_tree"),
                "history_len": len(history),
                "intent_tree_size": len(r.get("intent_tree") or []),
            },
        })
    return out


def build_labels(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    paths = [str(x).strip() for x in (rows[0].get("intent_tree") or []) if str(x).strip()]
    if not paths:
        raise SystemExit("首行缺少 intent_tree，无法构建候选意图标签集")
    labels = sorted({p.split("/")[-1] for p in paths})
    return {
        "source": "语音导航-FAQ列表 + 储蓄卡业务菜单层级（合并去重）",
        "n_paths": len(paths),
        "n_labels": len(labels),
        "paths": paths,
        "labels": labels,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="业务意图数据集 -> geval 格式")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="业务评测 JSONL 路径")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出数据集目录")
    ap.add_argument("--limit", type=int, default=0, help="只转换前 N 条（0=全量）")
    args = ap.parse_args(argv)

    src = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    rows = load_source(src, args.limit)
    records = convert(rows)
    labels = build_labels(rows)

    test_path = os.path.join(out_dir, "test.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    info = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "source_path": src,
        "source_sha256": sha256_file(src),
        "n_records": len(records),
        "n_labels": labels["n_labels"],
        "n_paths": labels["n_paths"],
        "task": "intent_routing",
        "split": "test",
        "field_mapping": {
            "question": "prompt", "answer": "ground_truth", "subject": "sample_type",
        },
        "note": "test 集已冻结；同一份数据不得进入训练集或 few-shot 示例。",
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("已生成 %s" % out_dir)
    print("  样本 %d 条 | 候选路径 %d 个 | 末级标签 %d 个" %
          (len(records), labels["n_paths"], labels["n_labels"]))
    print("  源文件 sha256 %s" % info["source_sha256"])
    print("  test.jsonl %.1f KB" % (os.path.getsize(test_path) / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
