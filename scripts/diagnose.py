#!/usr/bin/env python3
"""诊断「大模型分数反而低于小模型」的根因。

思路：分数只告诉你结果，逐样本 jsonl 才告诉你原因。本脚本把每个
「模型 × 任务」的明细归到六个互斥桶里，一眼看出是**模型不会**还是**链路出错**：

  error       请求失败（超时 / 限流 / 鉴权）——先修服务与并发
  empty       content 为空（thinking 模型把正文放 reasoning_content，或被 max_tokens 截断在思考阶段）
  truncated   response 被截到 4000 字符上限，说明输出极长（多半是思维链没关）
  thinking    响应里带 <think>/<reasoning> 块，污染了 IFEval / JSON 的格式判定
  unparsed    有输出但抽取器拿不到答案——prompt 协议与抽取规则不匹配
  scored      正常参与判分

用法：
  # 对比同一次运行里两个模型的若干任务
  python scripts/diagnose.py runs/geval-xxx --models qwen3.6-27b,openpangu-7b \
      --tasks gsm8k,ifeval,json_schema,ceval-fewshot

  # 或直指具体明细文件
  python scripts/diagnose.py runs/geval-xxx/qwen3.6-27b/gsm8k.jsonl

判定优先级：error > empty > thinking > truncated > unparsed > scored。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

THINK_RE = re.compile(r"<think(?:ing)?\b|<reasoning\b", re.I)
# geval/tasks.py 落盘时把 response 截到 MAX_RESPONSE_CHARS=4000
TRUNC_AT = 4000
SAMPLE_CHARS = 240


# --------------------------------------------------------------------------- #
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def _err_kind(msg: str) -> str:
    """把 error 文本归到可执行的三类。

    「请求失败」这个说法没有行动价值——限流要降并发、超时要加预算、连接要查网络，
    三者的处理方式完全不同，所以这里必须拆开。
    """
    m = str(msg).lower()
    if "429" in m or "503" in m or "rate limit" in m or "too many requests" in m or "限流" in m:
        return "限流"
    if "timed out" in m or "timeout" in m or "超时" in m:
        return "超时"
    if "connection" in m or "refused" in m or "reset" in m or "连接" in m:
        return "连接"
    return "其他"


def bucket_of(row: Dict[str, Any]) -> str:
    if row.get("error"):
        return "error"
    resp = row.get("response") or ""
    if not resp.strip():
        return "empty"
    if THINK_RE.search(resp):
        return "thinking"
    if len(resp) >= TRUNC_AT:
        return "truncated"
    if not row.get("parse_ok"):
        return "unparsed"
    return "scored"


def pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] if lo == hi else s[lo] * (hi - k) + s[hi] * (k - lo)


def analyze(path: str) -> Dict[str, Any]:
    rows = read_jsonl(path)
    n = len(rows)
    out: Dict[str, Any] = {
        "file": path,
        "n": n,
        "score": None,
        "len_p50": 0.0,
        "len_p95": 0.0,
        "top_strategy": "-",
        "errors": [],
        "sample": "",
    }
    if not n:
        return out

    buckets: Dict[str, int] = {}
    for r in rows:
        b = bucket_of(r)
        buckets[b] = buckets.get(b, 0) + 1
    out.update({f"n_{k}": v for k, v in buckets.items()})
    out.update({f"r_{k}": v / n for k, v in buckets.items()})

    out["score"] = sum(1 for r in rows if r.get("correct")) / n
    lens = [len(r.get("response") or "") for r in rows]
    out["len_p50"], out["len_p95"] = pct(lens, 50), pct(lens, 95)

    hist: Dict[str, int] = {}
    for r in rows:
        hist[str(r.get("strategy") or "none")] = hist.get(str(r.get("strategy") or "none"), 0) + 1
    out["top_strategy"] = max(hist.items(), key=lambda kv: kv[1])[0] if hist else "-"

    errs: Dict[str, int] = {}
    for r in rows:
        if r.get("error"):
            key = str(r["error"])[:80]
            errs[key] = errs.get(key, 0) + 1
    out["errors"] = sorted(errs.items(), key=lambda kv: -kv[1])[:3]

    kinds: Dict[str, int] = {}
    for r in rows:
        if r.get("error"):
            k = _err_kind(r["error"])
            kinds[k] = kinds.get(k, 0) + 1
    out["err_kinds"] = sorted(kinds.items(), key=lambda kv: -kv[1])

    bad = next(
        (r for r in rows if r.get("error") or not (r.get("response") or "").strip()),
        None,
    )
    if bad is None:
        bad = next((r for r in rows if not r.get("correct")), rows[0])
    out["sample"] = (
        f"[id={bad.get('id')}] strategy={bad.get('strategy')} "
        f"parse_ok={bad.get('parse_ok')} error={bad.get('error')}\n"
        f"    resp: {(bad.get('response') or '')[:SAMPLE_CHARS]!r}"
    )
    return out


# --------------------------------------------------------------------------- #
def collect(paths: List[str]) -> List[str]:
    found: List[str] = []
    for p in paths:
        if os.path.isfile(p) and p.endswith(".jsonl"):
            found.append(p)
        elif os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in sorted(files):
                    if fn.endswith(".jsonl"):
                        found.append(os.path.join(root, fn))
    return found


def print_table(results: List[Dict[str, Any]], show_samples: bool) -> None:
    cols = ["error", "empty", "thinking", "truncated", "unparsed"]
    head = f"{'model/task':<34}{'n':>5}{'score':>8}" + "".join(f"{c[:7]:>9}" for c in cols) + f"{'p95_len':>9}{'strategy':>16}"
    print(head)
    print("-" * len(head))
    for r in results:
        name = os.path.basename(r["file"])[:-6]
        parent = os.path.basename(os.path.dirname(r["file"]))
        label = f"{parent}/{name}"
        score = "n/a" if r["score"] is None else f"{r['score'] * 100:.1f}"
        line = f"{label:<34}{r['n']:>5}{score:>8}"
        for c in cols:
            line += f"{r.get('r_' + c, 0.0) * 100:>8.1f}%"
        line += f"{r['len_p95']:>9.0f}{r['top_strategy']:>16}"
        print(line)
        for msg, cnt in r["errors"]:
            print(f"{'':<34}  └─ error ×{cnt}: {msg}")
        if show_samples:
            print(f"{'':<34}  └─ {r['sample']}")


def verdict(results: List[Dict[str, Any]]) -> None:
    """把统计结果翻译成一句可执行的结论。"""
    print("\n判读：")
    for r in results:
        if not r["n"]:
            continue
        name = os.path.join(os.path.basename(os.path.dirname(r["file"])), os.path.basename(r["file"])[:-6])
        tips = []
        if r.get("r_error", 0) > 0.05:
            kinds = "、".join(f"{k}×{c}" for k, c in r.get("err_kinds", [])[:2]) or "未知"
            if r.get("err_kinds") and r["err_kinds"][0][0] == "限流":
                tips.append(f"限流（{kinds}）→ 把 concurrency 降到 1~2、给该模型加 rate_limit_qps、"
                            "多个任务不要同时起进程；失败样本用 --resume --retry-failed 补跑")
            elif r.get("err_kinds") and r["err_kinds"][0][0] == "超时":
                tips.append(f"超时（{kinds}）→ 调大 request_timeout_s（27B 长 CoT 建议 300+）、"
                            "同时降低并发，否则超时会雪崩")
            else:
                tips.append(f"请求失败（{kinds}）→ 查端点健康与网络（proxy=bypass / 鉴权 / 模型名）")
        if r.get("r_empty", 0) > 0.05:
            tips.append("响应体为空 → 多半是 thinking 模式没关或 max_tokens 在思考阶段就耗尽；"
                        "检查 extra_body.enable_thinking / user_suffix，并把任务 max_tokens 放大")
        if r.get("r_thinking", 0) > 0.05:
            tips.append("响应混入 <think> 块 → IFEval/JSON 的格式判定会被思维链污染，先剥离再判分")
        if r.get("r_truncated", 0) > 0.05:
            tips.append("输出超长 → 提高 max_tokens，并确认思考开关")
        if r.get("r_unparsed", 0) > 0.05 and not tips:
            tips.append("有输出但抽不到答案 → 校对 prompt 协议与抽取规则（不是模型能力问题）")
        if not tips:
            tips.append("链路正常，低分确实反映模型能力或数据分布差异")
        print(f"  {name}：{'；'.join(tips)}")


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="定位大模型评测低分是链路问题还是能力问题")
    ap.add_argument("paths", nargs="+", help="明细 jsonl 文件、模型目录或 run 目录")
    ap.add_argument("--models", default="", help="只保留这些模型 id（逗号分隔）")
    ap.add_argument("--tasks", default="", help="只保留这些任务名（逗号分隔，如 gsm8k,ifeval）")
    ap.add_argument("--samples", action="store_true", help="打印一条失败样例的原文")
    args = ap.parse_args(argv)

    files = collect(args.paths)
    if args.models:
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        files = [f for f in files if os.path.basename(os.path.dirname(f)) in want]
    if args.tasks:
        want = {t.strip() for t in args.tasks.split(",") if t.strip()}
        files = [f for f in files if os.path.basename(f)[:-6] in want]
    if not files:
        print("没有匹配的 .jsonl 明细文件", file=sys.stderr)
        return 1

    results = [analyze(f) for f in files]
    print_table(results, args.samples)
    verdict(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
