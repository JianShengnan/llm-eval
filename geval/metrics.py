"""指标与统计：准确率、置信区间、配对显著性检验、时延分布。

为什么必须带统计量：
  3000 题的评测集上，68.2% 与 65.7% 的差距完全可能是抽样噪声。
  不做区间估计与配对检验，就无法判断「小模型是否真的更差」，
  也无法判断微调带来的提升是否显著 —— 这会直接导致错误的技术选型。
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .util import md_table

# --------------------------------------------------------------------------- #
# 基础量
# --------------------------------------------------------------------------- #
def mean(xs: Sequence[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    k = (len(vals) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(vals[int(k)])
    return float(vals[lo] * (hi - k) + vals[hi] * (k - lo))


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson 得分区间：小样本 / 接近 0 或 1 时比正态近似可靠得多。"""
    if n <= 0:
        return 0.0, 0.0
    z = _z_for(alpha)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _z_for(alpha: float) -> float:
    table = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004, 0.001: 3.2905267314919255}
    if alpha in table:
        return table[alpha]
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _norm_cdf(mid) < 1 - alpha / 2.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    flags: Sequence[bool],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260916,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """对任意「逐样本 0/1」指标做百分位 bootstrap，返回 (均值, lo, hi)。"""
    vals = [1.0 if f else 0.0 for f in flags]
    n = len(vals)
    if n == 0:
        return None, None, None
    obs = sum(vals) / n
    if n_boot <= 0 or n == 1:
        return obs, obs, obs
    rng = random.Random(seed)
    stats: List[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    lo = stats[max(0, int(math.floor((alpha / 2.0) * n_boot)))]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2.0) * n_boot)) - 1)]
    return obs, lo, hi


def paired_bootstrap_diff(
    a_flags: Sequence[bool],
    b_flags: Sequence[bool],
    *,
    n_boot: int = 4000,
    seed: int = 424242,
) -> Dict[str, Optional[float]]:
    """配对 bootstrap：检验 A 与 B 在同一批样本上的准确率差异是否显著。

    a_flags / b_flags 必须逐样本对齐（同一条样本、同一个模型各自对错）。
    返回 {diff, lo, hi, p_two_sided}，diff = acc(A) - acc(B)。
    """
    n = min(len(a_flags), len(b_flags))
    if n == 0:
        return {"diff": None, "lo": None, "hi": None, "p_two_sided": None}
    av = [1.0 if a_flags[i] else 0.0 for i in range(n)]
    bv = [1.0 if b_flags[i] else 0.0 for i in range(n)]
    obs = (sum(av) - sum(bv)) / n

    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        sa = sb = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            sa += av[idx]
            sb += bv[idx]
        diffs.append((sa - sb) / n)
    diffs.sort()
    lo = diffs[max(0, int(math.floor(0.025 * n_boot)))]
    hi = diffs[min(n_boot - 1, int(math.ceil(0.975 * n_boot)) - 1)]
    below = sum(1 for d in diffs if d <= 0) / n_boot
    above = sum(1 for d in diffs if d >= 0) / n_boot
    p = min(1.0, 2.0 * min(below, above))
    return {"diff": obs, "lo": lo, "hi": hi, "p_two_sided": p}


def mcnemar_exact_p(a_flags: Sequence[bool], b_flags: Sequence[bool]) -> Dict[str, Any]:
    """McNemar 精确检验（二项版）。比近似卡方更适用于中小评测集。

    b = A 对 B 错；c = A 错 B 对。若 b+c 很小则无统计学意义。
    """
    n = min(len(a_flags), len(b_flags))
    b = c = 0
    for i in range(n):
        a_ok, b_ok = bool(a_flags[i]), bool(b_flags[i])
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
    total = b + c
    if total == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_two_sided": 1.0}
    k = min(b, c)
    tail = sum(math.comb(total, i) for i in range(0, k + 1)) * (0.5 ** total)
    p = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "n_discordant": total, "p_two_sided": p}


def significance_label(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "极显著 ***"
    if p < 0.01:
        return "显著 **"
    if p < 0.05:
        return "显著 *"
    if p < 0.10:
        return "边缘"
    return "不显著"


# --------------------------------------------------------------------------- #
# 评测结果聚合
# --------------------------------------------------------------------------- #
def aggregate(
    scored: Sequence[Dict[str, Any]],
    *,
    metric: str = "correct",
    bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260916,
) -> Dict[str, Any]:
    """把一个任务的全部答卷聚合成一份可上报的结果块。"""
    n = len(scored)
    if n == 0:
        return {"n": 0, "score": None}

    flags = [bool(r.get(metric)) for r in scored]
    k = sum(1 for f in flags if f)
    point, lo, hi = bootstrap_ci(flags, n_boot=bootstrap, alpha=alpha, seed=seed)
    wlo, whi = wilson_interval(k, n, alpha)

    latencies = [float(r["latency_ms"]) for r in scored if r.get("latency_ms") is not None]
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in scored)
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in scored)

    parse_fail = sum(1 for r in scored if r.get("parse_ok") is False)
    ambiguous = sum(1 for r in scored if r.get("ambiguous"))
    errors = sum(1 for r in scored if r.get("error"))

    total_latency_s = sum(latencies) / 1000.0 if latencies else 0.0
    return {
        "n": n,
        "n_correct": k,
        "score": point,
        "ci_bootstrap": [lo, hi],
        "ci_wilson": [wlo, whi],
        "stderr": math.sqrt(point * (1 - point) / n) if point is not None else None,
        "parse_fail": parse_fail,
        "parse_fail_rate": (parse_fail / n) if n else None,
        "ambiguous": ambiguous,
        "ambiguous_rate": (ambiguous / n) if n else None,
        "errors": errors,
        "latency_ms": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else None,
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
            "completion_per_s": (completion_tokens / total_latency_s) if total_latency_s > 0 else None,
        },
        "extra": _merge_extras(scored),
    }


def _merge_extras(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """合并各任务特有的子指标（如 IFEval 的指令级通过率）。"""
    buckets: Dict[str, List[float]] = {}
    for row in scored:
        for key, val in (row.get("extras") or {}).items():
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            if isinstance(val, (int, float)):
                buckets.setdefault(key, []).append(float(val))
    return {k: mean(v) for k, v in sorted(buckets.items())}


def group_breakdown(
    scored: Sequence[Dict[str, Any]],
    *,
    key: str = "subject",
    min_n: int = 1,
) -> List[Dict[str, Any]]:
    """按 subject / 意图类别 等维度分组统计，用于定位能力短板。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in scored:
        g = str(row.get(key) or "unknown")
        buckets.setdefault(g, []).append(row)
    out: List[Dict[str, Any]] = []
    for g, rows in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(rows) < min_n:
            continue
        flags = [bool(r.get("correct")) for r in rows]
        out.append(
            {
                key: g,
                "n": len(rows),
                "score": (sum(flags) / len(flags)) if flags else None,
                "parse_fail_rate": (sum(1 for r in rows if r.get("parse_ok") is False) / len(rows)),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 矩阵渲染
# --------------------------------------------------------------------------- #
def render_leaderboard(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    model_labels: Dict[str, str],
    task_names: Sequence[str],
) -> str:
    """results[model_id][task] -> aggregate dict；输出模型 × 任务的准确率矩阵。"""
    rows: List[List[str]] = []
    for mid, per_task in results.items():
        row = [model_labels.get(mid, mid)]
        for task in task_names:
            agg = per_task.get(task)
            if not agg or agg.get("score") is None:
                row.append("—")
            else:
                lo, hi = (agg.get("ci_bootstrap") or [None, None])[:2]
                half = (hi - lo) / 2 * 100 if lo is not None and hi is not None else None
                row.append(f"{agg['score'] * 100:.1f}%" + (f" ±{half:.1f}" if half is not None else ""))
        rows.append(row)
    return md_table(["模型"] + list(task_names), rows)
