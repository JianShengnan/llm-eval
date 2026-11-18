# -*- coding: utf-8 -*-
"""调用成本与耗时统计：把逐样本 token / 时延明细聚合成可上报的口径。

框架原本已在 BaseTask._row 落盘 latency_ms / prompt_tokens / completion_tokens，
并在 metrics.aggregate 给出 p50~p99 与 tokens/s。本模块补齐三类缺口：

  1. 逐样本 total_tokens、缓存命中、截断、请求墙钟时间（professional.py 写入）
  2. 按意图类型 / 意图标签分组的 token 与时延（专业能力评测的成本归因）
  3. 运行级 wall-clock、累计请求耗时、按单价的成本估算

全部纯标准库，不改动原框架的 client / runner / metrics。
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from .metrics import mean, percentile

_TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _num(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _parse_ts(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.strptime(str(value), _TS_FORMAT)
    except ValueError:
        return None


def run_wall_clock_s(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """从第 1 条请求开始到最后 1 条结束的真实墙钟时间（含并发重叠，不重复累加）。"""
    starts = [_parse_ts(r.get("wall_start")) for r in rows]
    ends = [_parse_ts(r.get("wall_end")) for r in rows]
    starts = [x for x in starts if x is not None]
    ends = [x for x in ends if x is not None]
    if not starts or not ends:
        return None
    return max(0.0, (max(ends) - min(starts)).total_seconds())


def cost_estimate(rows: Sequence[Dict[str, Any]], *,
                  price_in_per_1k: float = 0.0,
                  price_out_per_1k: float = 0.0) -> Optional[float]:
    """按配置单价估算调用成本；任一单价为 0 时返回 None（不臆造单价）。"""
    if price_in_per_1k <= 0 and price_out_per_1k <= 0:
        return None
    pin = sum(_num(r, "prompt_tokens") for r in rows) / 1000.0
    pout = sum(_num(r, "completion_tokens") for r in rows) / 1000.0
    return pin * price_in_per_1k + pout * price_out_per_1k


def usage_summary(rows: Sequence[Dict[str, Any]], *,
                  price_in_per_1k: float = 0.0,
                  price_out_per_1k: float = 0.0) -> Dict[str, Any]:
    """一组样本的 token / 时延 / 质量汇总。rows 为已落盘的逐样本明细。"""
    n = len(rows)
    lat = [_num(r, "latency_ms") for r in rows if r.get("latency_ms") is not None]
    wall = [_num(r, "request_wall_ms") for r in rows if r.get("request_wall_ms") is not None]
    pt = sum(_num(r, "prompt_tokens") for r in rows)
    ct = sum(_num(r, "completion_tokens") for r in rows)
    tt = sum(_num(r, "total_tokens") for r in rows) or (pt + ct)
    total_lat_s = sum(lat) / 1000.0

    out: Dict[str, Any] = {
        "n": n,
        "prompt_tokens": int(pt),
        "completion_tokens": int(ct),
        "total_tokens": int(tt),
        "avg_prompt_tokens": round(pt / n, 1) if n else None,
        "avg_completion_tokens": round(ct / n, 1) if n else None,
        "avg_total_tokens": round(tt / n, 1) if n else None,
        "cached_hits": sum(1 for r in rows if r.get("cached")),
        "length_truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
        "latency_ms": {
            "mean": mean(lat),
            "p50": percentile(lat, 50),
            "p90": percentile(lat, 90),
            "p95": percentile(lat, 95),
            "p99": percentile(lat, 99),
            "max": max(lat) if lat else None,
        },
        # 单请求时延之和 ≠ 墙钟时间：并发下两者差距反映并行度
        "cumulative_request_ms": round(sum(lat), 1),
        "cumulative_wall_ms": round(sum(wall), 1),
        "run_wall_clock_s": run_wall_clock_s(rows),
        "completion_tokens_per_s": round(ct / total_lat_s, 1) if total_lat_s > 0 else None,
        "samples_per_s": round(n / (total_lat_s), 3) if total_lat_s > 0 else None,
        "estimated_cost": cost_estimate(rows, price_in_per_1k=price_in_per_1k,
                                        price_out_per_1k=price_out_per_1k),
    }
    return out


def usage_by_group(rows: Sequence[Dict[str, Any]], *,
                   key: str = "subject",
                   price_in_per_1k: float = 0.0,
                   price_out_per_1k: float = 0.0) -> List[Dict[str, Any]]:
    """按 subject / sample_type / gold 等维度拆分 token 与耗时，定位成本大头。"""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(key) or "unknown")].append(r)
    out = []
    for name, group in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        s = usage_summary(group, price_in_per_1k=price_in_per_1k,
                          price_out_per_1k=price_out_per_1k)
        out.append({
            key: name, "n": s["n"], "total_tokens": s["total_tokens"],
            "avg_total_tokens": s["avg_total_tokens"],
            "p50_latency_ms": s["latency_ms"]["p50"],
            "p95_latency_ms": s["latency_ms"]["p95"],
            "intent_accuracy": (sum(bool(r.get("correct")) for r in group) / len(group)),
        })
    return out
