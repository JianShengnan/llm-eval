#!/usr/bin/env python3
"""离线重判：判定规则修好后，不必重发请求，直接把已有 run 目录的分数重算一遍。

背景
----
GSM8K / MATH 数据集的 answer 字段是**整段 CoT**（"...\\n#### 18"）。
旧版直接拿它做数值比对，`float(gold)` 抛 ValueError 后退化成字符串全等比较，
于是「模型答对了」也判 0 分。修复判定规则后，用本脚本把历史结果重判即可，
省掉一轮上万条请求（判定是纯离线计算，不需要访问模型服务）。

用法
----
    # 先看差异，不落盘（建议先跑这个）
    python scripts/rescore.py runs/geval-llm-chat-core --dry-run

    # 确认后落盘重判
    python scripts/rescore.py runs/geval-llm-chat-core

    # 只重判指定任务
    python scripts/rescore.py runs/geval-llm-chat-core --tasks gsm8k,math

重判会同步刷新 `<model>/<task>.jsonl`、`<model>/<task>.summary.json` 与 `summary.json`，
之后直接 `python -m geval report --run ...` 出新报告即可。

注意：jsonl 里的 response 落盘时被截断到 4000 字符（MAX_RESPONSE_CHARS）。
若某条的最终答案在截断之后，重判会报「响应可能被截断」，这类样本建议重跑而不是重判。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geval.answers import canonical_gold_numeric, extract_number, numbers_equal  # noqa: E402
from geval.metrics import aggregate  # noqa: E402
from geval.runner import _breakdown, _examples, _parse_note  # noqa: E402

# 只对这些数值类任务重判；其它任务（选择题 / IFEval / JSON）判定规则未变，不动
DEFAULT_TASKS = ("gsm8k", "math")
MAX_RESPONSE_CHARS = 4000


# --------------------------------------------------------------------------- #
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
def rescore(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """就地重算判定字段，返回统计。"""
    stat = {"n": 0, "flipped": 0, "parse_fail": 0, "truncated": 0, "gold_unparsed": 0}
    for r in rows:
        stat["n"] += 1
        if r.get("error"):
            continue
        gold = canonical_gold_numeric(r.get("gold"))
        if gold is None:
            stat["gold_unparsed"] += 1
            continue
        resp = r.get("response") or ""
        if len(resp) >= MAX_RESPONSE_CHARS:
            stat["truncated"] += 1
        pred, strategy = extract_number(resp)
        before = bool(r.get("correct"))
        r["gold"] = gold
        r["pred"] = pred
        r["strategy"] = strategy
        r["parse_ok"] = pred is not None
        r["correct"] = numbers_equal(pred, gold, 1e-4)
        if pred is None:
            stat["parse_fail"] += 1
        if bool(r["correct"]) != before:
            stat["flipped"] += 1
    return stat


def rebuild_agg(old: Dict[str, Any], rows: List[Dict[str, Any]], task_key: str, seed: int) -> Dict[str, Any]:
    agg = dict(old)
    agg.update(aggregate(rows, bootstrap=2000, alpha=0.05, seed=seed))
    # aggregate 不产出标识字段，从旧 summary 里补回
    for k in (
        "task",
        "task_name",
        "model",
        "model_label",
        "kind",
        "protocol",
        "mode",
        "num_fewshot",
        "dataset_source",
        "task_metrics",
    ):
        if k in old:
            agg[k] = old[k]
    agg["breakdown"] = _breakdown(rows, task_key)
    agg["examples_failed"] = _examples(rows, want_correct=False, n=5)
    agg["examples_passed"] = _examples(rows, want_correct=True, n=3)
    agg["parse_note"] = _parse_note(rows)
    return agg


def patch_payload(node: Any, task_key: str, model_id: str, agg: Dict[str, Any]) -> None:
    """把重算后的结果块同步进 summary.json（results / task_summaries 两处结构都要改）。"""
    if isinstance(node, dict):
        if node.get("task") == task_key and str(node.get("model")) == str(model_id):
            node.update(agg)
        for v in node.values():
            patch_payload(v, task_key, model_id, agg)
    elif isinstance(node, list):
        for v in node:
            patch_payload(v, task_key, model_id, agg)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="离线重判已有 run 目录的数值类任务")
    ap.add_argument("run", help="run 目录，如 runs/geval-llm-chat-core")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="要重判的任务，逗号分隔")
    ap.add_argument("--dry-run", action="store_true", help="只打印差异，不写文件")
    args = ap.parse_args()

    run_dir = args.run
    if not os.path.isdir(run_dir) and os.path.isdir(os.path.join("runs", run_dir)):
        run_dir = os.path.join("runs", run_dir)
    if not os.path.isdir(run_dir):
        print(f"找不到 run 目录：{args.run}", file=sys.stderr)
        return 1

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    seed = 42
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            seed = int(read_json(cfg_path).get("seed") or 42)
        except (ValueError, TypeError, OSError):
            seed = 42

    summary_path = os.path.join(run_dir, "summary.json")
    payload = read_json(summary_path) if os.path.exists(summary_path) else None
    touched = 0

    for model_id in sorted(os.listdir(run_dir)):
        mdir = os.path.join(run_dir, model_id)
        if not os.path.isdir(mdir):
            continue
        for fn in sorted(os.listdir(mdir)):
            if not fn.endswith(".jsonl"):
                continue
            task_key = fn[: -len(".jsonl")]
            if task_key not in tasks:
                continue
            jsonl_path = os.path.join(mdir, fn)
            rows = read_jsonl(jsonl_path)
            if not rows:
                continue

            old_score = sum(1 for r in rows if r.get("correct")) / len(rows)
            stat = rescore(rows)
            new_score = sum(1 for r in rows if r.get("correct")) / len(rows)

            print(
                f"[{model_id} × {task_key}] {len(rows)} 条："
                f" {old_score:.1%} → {new_score:.1%}"
                f"（判定翻转 {stat['flipped']} 条，解析失败 {stat['parse_fail']} 条）"
            )
            if stat["truncated"]:
                print(f"    ⚠ {stat['truncated']} 条 response 被截断到 {MAX_RESPONSE_CHARS} 字符，"
                      f"重判可能不准，建议这几条重跑")
            if stat["gold_unparsed"]:
                print(f"    ⚠ {stat['gold_unparsed']} 条 gold 无法解析成数值，已跳过")
            touched += 1

            if args.dry_run:
                continue

            write_jsonl(jsonl_path, rows)

            agg_path = os.path.join(mdir, f"{task_key}.summary.json")
            if os.path.exists(agg_path):
                old_agg = read_json(agg_path)
                agg = rebuild_agg(old_agg, rows, task_key, seed)
                write_json(agg_path, agg)
                if payload is not None:
                    patch_payload(payload, task_key, model_id, agg)

    if not touched:
        print(f"没有需要重判的任务（当前任务过滤：{sorted(tasks)}）")
        return 0

    if not args.dry_run:
        if payload is not None:
            write_json(summary_path, payload)
        print("\n已落盘。重新出报告：")
        print(f"  python -m geval report --run {run_dir}")
    else:
        print("\n这是预演（--dry-run），没有写任何文件。确认无误后去掉 --dry-run 再跑一次。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
