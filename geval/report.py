"""报告生成：Markdown + 自包含 HTML（内联 SVG，无任何外部引用）。

为什么不用 ECharts / Chart.js：
  内网离线环境加载不到 CDN；这里用纯 Python 生成内联 SVG，HTML 单文件即可分享。

报告回答四个问题：
  1. 谁更准？（准确率矩阵 + 置信区间）
  2. 差距是真的吗？（配对 bootstrap + McNemar 精确检验）
  3. 快多少、值不值？（时延分布 + 准确率/时延帕累托图）
  4. 分低是模型的错还是抽取器的错？（解析失败率 + 策略分布 + 错例抽样）
"""

from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import (
    mcnemar_exact_p,
    paired_bootstrap_diff,
    significance_label,
)
from .util import fmt_num, fmt_pct, jdump, md_table, now_iso, read_json, read_jsonl, write_json

PALETTE = ["#185FA5", "#0F6E56", "#BA7517", "#A32D2D", "#534AB7", "#888780", "#D4537E", "#639922"]


# --------------------------------------------------------------------------- #
# 数据装配
# --------------------------------------------------------------------------- #
def load_run(run_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"找不到 {summary_path}；请先执行 run")
    data = read_json(summary_path)
    data["run_dir"] = run_dir
    manifest_path = os.path.join(run_dir, "manifest.json")
    data["manifest"] = read_json(manifest_path, default={})
    return data


def load_flags(run_dir: str, model: str, task: str) -> Dict[str, bool]:
    """读取逐样本对错，供配对检验使用（按 id 对齐）。"""
    path = os.path.join(run_dir, model, f"{task}.jsonl")
    if not os.path.exists(path):
        return {}
    out: Dict[str, bool] = {}
    for row in read_jsonl(path):
        out[str(row.get("id"))] = bool(row.get("correct"))
    return out


def compare(run_dir: str, model_a: str, model_b: str, task: str, *, n_boot: int = 4000) -> Optional[Dict[str, Any]]:
    """同一样本上对齐后做配对检验。

    配对检验的前提是「两个模型跑的是同一批样本」。这里同时返回样本覆盖率，
    覆盖率不足时会给出警告 —— 否则可能拿两个只有部分交集的集合算出假的差值。
    """
    fa = load_flags(run_dir, model_a, task)
    fb = load_flags(run_dir, model_b, task)
    common = sorted(set(fa) & set(fb))
    if len(common) < 10:
        return None
    a = [fa[i] for i in common]
    b = [fb[i] for i in common]
    boot = paired_bootstrap_diff(a, b, n_boot=n_boot)
    mc = mcnemar_exact_p(a, b)
    acc_a = sum(a) / len(a)
    acc_b = sum(b) / len(b)

    n_a, n_b = len(fa), len(fb)
    coverage = len(common) / max(1, max(n_a, n_b))
    warnings: List[str] = []
    if coverage < 0.9:
        warnings.append(
            f"样本覆盖率仅 {coverage:.1%}（{model_a}={n_a}, {model_b}={n_b}, 交集={len(common)}），"
            f"两模型跑的很可能不是同一批样本，差值不可信"
        )

    return {
        "task": task,
        "model_a": model_a,
        "model_b": model_b,
        "n": len(common),
        "n_a": n_a,
        "n_b": n_b,
        "coverage": coverage,
        "warnings": warnings,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "diff": acc_a - acc_b,
        "ci": [boot["lo"], boot["hi"]],
        "p_bootstrap": boot["p_two_sided"],
        "p_mcnemar": mc["p_two_sided"],
        "n_discordant": mc["n_discordant"],
        "label": significance_label(mc["p_two_sided"]),
    }


def compatible_config(model_a: str, model_b: str, agg_a: Dict[str, Any], agg_b: Dict[str, Any]) -> List[str]:
    """检查两个模型在同一任务上的运行配置是否一致。

    配置不一致时，差值里混入了「配置差异」而非「能力差异」，配对检验失去意义。
    尤其是 SFT 时若改过 chat template，即使配置相同也可能引入格式混淆，
    需要人工确认（报告中会提示）。
    """
    issues: List[str] = []
    for field, name in (("protocol", "prompt 协议"), ("mode", "打分模式"), ("num_fewshot", "few-shot 条数")):
        va, vb = agg_a.get(field), agg_b.get(field)
        if va is not None and vb is not None and va != vb:
            issues.append(f"{name}不一致：{model_a}={va} vs {model_b}={vb}")
    ta, tb = agg_a.get("n"), agg_b.get("n")
    if ta and tb and abs(ta - tb) / max(ta, tb) > 0.1:
        issues.append(f"样本数差异过大：{model_a}={ta} vs {model_b}={tb}")
    return issues


def degradation_label(diff: Optional[float], p: Optional[float], alpha: float = 0.05) -> str:
    """微调影响判定。diff 的约定是 **post - pre**：正数=微调后更好，负数=退化。

    只有达到统计显著才敢说「退化 / 提升」——否则小幅波动很容易被误读成能力变化。
    """
    if diff is None or p is None:
        return "无法判定"
    if p >= alpha:
        return "不显著（视为持平）"
    return "显著提升" if diff > 0 else "显著退化"


def impact_delta(cmp: Dict[str, Any]) -> Dict[str, Any]:
    """把 compare() 的 (pre - post) 翻转成 (post - pre)，并同步翻转置信区间。

    compare() 的语义是 acc_a - acc_b，在微调评估里 a=微调前、b=微调后，
    因此必须翻转符号，否则「退化」会被报成「提升」。
    """
    delta = cmp["acc_b"] - cmp["acc_a"]
    lo, hi = cmp["ci"][0], cmp["ci"][1]
    return {"delta": delta, "ci": [-hi, -lo] if lo is not None and hi is not None else [None, None]}


def build_impact_section(
    summary: Dict[str, Any],
    pairs: Sequence[Sequence[str]],
    tasks: Sequence[str],
    *,
    alpha: float = 0.05,
) -> List[str]:
    """生成「微调影响评估」章节：逐任务的 pre→post 差值与判定 + 总体代价汇总。

    只有至少一对模型都跑过时才输出内容 —— 否则（例如手上只有一个模型通道做链路验证）
    会输出一堆「缺少结果，跳过」的空章节，反而让人以为流程出了问题。
    """
    run_dir = summary["run_dir"]
    labels = summary.get("model_labels") or {}
    results = summary.get("results") or {}
    body: List[str] = []

    for a, b in pairs:
        if a not in results or b not in results:
            continue  # 该配对的两端还没跑过，整段略过

        rows: List[List[str]] = []
        deltas: List[float] = []
        degraded: List[str] = []
        improved: List[str] = []
        invalid = False

        for t in tasks:
            aa = (results.get(a) or {}).get(t) or {}
            ab = (results.get(b) or {}).get(t) or {}
            cmp = compare(run_dir, a, b, t)
            if not cmp:
                continue
            for issue in compatible_config(a, b, aa, ab):
                rows.append([t, "—", "—", "—", "—", "—", f"⚠️ {issue}"])
                invalid = True
            for w in cmp["warnings"]:
                rows.append([t, "—", "—", "—", "—", "—", f"⚠️ {w}"])
                invalid = True

            # a=微调前、b=微调后，compare() 给的是 (前-后)，这里翻转成 (后-前)
            imp = impact_delta(cmp)
            judge = degradation_label(imp["delta"], cmp["p_mcnemar"], alpha)
            deltas.append(imp["delta"])
            if "退化" in judge:
                degraded.append(t)
            elif "提升" in judge:
                improved.append(t)
            rows.append(
                [
                    t,
                    fmt_pct(cmp["acc_a"], 1),
                    fmt_pct(cmp["acc_b"], 1),
                    f"{imp['delta'] * 100:+.2f} pt",
                    f"[{imp['ci'][0] * 100:+.2f}, {imp['ci'][1] * 100:+.2f}]",
                    f"{cmp['p_mcnemar']:.4f}",
                    judge,
                ]
            )

        if not rows:
            continue

        body.append(f"### {labels.get(a, a)} → {labels.get(b, b)}")
        body.append("")
        body.append(
            md_table(
                ["任务", "微调前", "微调后", "差值", "差值 95% CI", "McNemar p", "判定"],
                rows,
                ["l", "r", "r", "r", "r", "r", "l"],
            )
        )
        body.append("")
        if deltas:
            avg = sum(deltas) / len(deltas)
            body.append(f"- 宏平均变化：**{avg * 100:+.2f} pt**（跨 {len(deltas)} 个任务等权平均）")
            body.append(f"- 显著退化任务：{('、'.join(degraded)) if degraded else '无'}")
            body.append(f"- 显著提升任务：{('、'.join(improved)) if improved else '无'}")
            body.append("")
        if invalid:
            body.append("> ⚠️ 上表存在告警行，相关结论不可用。请先修正运行配置或样本集后重跑。")
            body.append("")

    if not body:
        return []

    L: List[str] = [
        "> **读这张表的前提**：两次运行必须用**同一批样本、同一套 prompt**。"
        "表中出现「样本覆盖率」或「配置不一致」告警时，该行结论不可用。",
        "> 另需人工确认一件事：**微调是否改动过 chat template**。若改过，"
        "差值里就混入了 prompt 格式差异，不是纯粹的能力差异（详见下方说明）。",
        "",
    ]
    L.extend(body)
    L.append("### chat template 的混淆风险（务必人工确认）")
    L.append("")
    L.append(
        "SFT 常见的连带改动是 chat template（尤其是从 base 模型微调、"
        "或者为了塞入 system prompt 而调整了模板）。一旦改动："
    )
    L.append("")
    L.append(
        "1. 同一段 prompt 在两个模型上被拼接成**不同的实际输入**，"
        "差值里混入格式差异，不再是纯能力差异；\n"
        "2. 表现可能既像「能力提升」也可能像「能力退化」，方向不可预期。"
    )
    L.append("")
    L.append("**处理方式（按优先级）**：")
    L.append("")
    L.append(
        "1. 微调时**不要修改 chat template**，保持与基座一致（最省事，推荐）；\n"
        "2. 若必须改，则**把微调前的模型也用新模板重新测一遍**，保证两边输入格式一致；\n"
        "3. 退而求其次：在报告里显式标注该混淆因素，并配合人工抽检若干样本的实际输入确认。"
    )
    L.append("")
    L.append("验证方法：对两个模型端点各发一条相同请求，比对服务端日志里的最终拼串。")
    L.append("")
    return L


def available_tasks(summary: Dict[str, Any]) -> List[str]:
    seen: List[str] = []
    for per_task in (summary.get("results") or {}).values():
        for t in per_task:
            if t not in seen:
                seen.append(t)
    order = ["ceval", "cmmlu", "mmlu", "mmlu_pro", "arc", "agieval", "gsm8k", "math", "bbh", "ifeval", "json_schema"]
    return sorted(seen, key=lambda t: (order.index(t) if t in order else 99, t))


def model_order(summary: Dict[str, Any]) -> List[str]:
    return list((summary.get("results") or {}).keys())


def macro_scores(summary: Dict[str, Any], tasks: Sequence[str]) -> List[Dict[str, Any]]:
    """宏平均：只对实际跑过的任务取等权平均，并在报告中显式披露覆盖数。"""
    out = []
    for mid, per_task in (summary.get("results") or {}).items():
        scores = [per_task[t]["score"] for t in tasks if per_task.get(t, {}).get("score") is not None]
        lat = [per_task[t]["latency_ms"]["p50"] for t in tasks if per_task.get(t, {}).get("latency_ms", {}).get("p50")]
        out.append(
            {
                "model": mid,
                "label": (summary.get("model_labels") or {}).get(mid, mid),
                "macro_score": (sum(scores) / len(scores)) if scores else None,
                "n_tasks": len(scores),
                "p50_latency": sorted(lat)[len(lat) // 2] if lat else None,
                "p95_latency": max(
                    (per_task[t]["latency_ms"]["p95"] or 0) for t in tasks if per_task.get(t, {}).get("latency_ms")
                )
                if lat
                else None,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 内联 SVG 图表
# --------------------------------------------------------------------------- #
def svg_grouped_bars(
    labels: Sequence[str],
    series: Sequence[Dict[str, Any]],
    *,
    y_max: float = 100.0,
    height: int = 300,
    width: int = 900,
) -> str:
    n_groups = max(1, len(labels))
    n_series = max(1, len(series))
    left, right, top, bottom = 56, 16, 16, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / n_groups
    bar_w = max(4.0, (group_w * 0.72) / n_series)

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">',
        f"<title>准确率对比</title>",
    ]
    # 网格线
    for i in range(5):
        y = top + plot_h * i / 4
        val = y_max * (1 - i / 4)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#D3D1C7" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y:.1f}" text-anchor="end" dominant-baseline="central" '
            f'font-size="11" fill="#888780" font-family="system-ui, sans-serif">{val:.0f}</text>'
        )
    # 柱子
    for gi, label in enumerate(labels):
        gx = left + group_w * gi
        for si, s in enumerate(series):
            val = s["values"][gi] if gi < len(s["values"]) else None
            if val is None:
                continue
            h = max(1.0, plot_h * (val / y_max))
            x = gx + group_w * 0.14 + bar_w * si
            y = top + plot_h - h
            color = s.get("color") or PALETTE[si % len(PALETTE)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="1.5"/>')
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#5F5E5A" font-family="system-ui, sans-serif">{val:.0f}</text>'
            )
        parts.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{top + plot_h + 16:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#444441" font-family="system-ui, sans-serif">{html.escape(str(label))}</text>'
        )
    # 图例
    lx = left
    for si, s in enumerate(series):
        color = s.get("color") or PALETTE[si % len(PALETTE)]
        parts.append(f'<rect x="{lx}" y="{top + plot_h + 32}" width="10" height="10" rx="2" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 15}" y="{top + plot_h + 37}" dominant-baseline="central" font-size="11" '
            f'fill="#5F5E5A" font-family="system-ui, sans-serif">{html.escape(str(s["label"]))}</text>'
        )
        lx += 12 + 8 * len(str(s["label"])) + 16
    parts.append("</svg>")
    return "".join(parts)


def svg_pareto(points: Sequence[Dict[str, Any]], *, width: int = 900, height: int = 340) -> str:
    """准确率（y）vs 时延（x）：越靠左上越优，用来定位性价比拐点。"""
    pts = [p for p in points if p.get("macro_score") is not None and p.get("p50_latency")]
    if not pts:
        return '<p style="font-size:13px;color:#5F5E5A">缺少时延数据，无法绘制帕累托图。</p>'

    left, right, top, bottom = 60, 24, 24, 52
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(p["p50_latency"]) for p in pts]
    ys = [float(p["macro_score"]) * 100 for p in pts]
    x_min, x_max = min(xs), max(xs)
    if x_max - x_min < 1e-6:
        x_max = x_min + 1.0
    y_min = max(0.0, min(ys) - 8)
    y_max = min(100.0, max(ys) + 6)

    def px(x: float) -> float:
        return left + plot_w * (x - x_min) / (x_max - x_min)

    def py(y: float) -> float:
        return top + plot_h * (1 - (y - y_min) / max(1e-6, y_max - y_min))

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">',
        "<title>准确率与时延帕累托图</title>",
    ]
    for i in range(5):
        y = top + plot_h * i / 4
        val = y_max - (y_max - y_min) * i / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#D3D1C7" stroke-width="0.5"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y:.1f}" text-anchor="end" dominant-baseline="central" font-size="11" '
            f'fill="#888780" font-family="system-ui, sans-serif">{val:.0f}%</text>'
        )
    for i in range(5):
        x = left + plot_w * i / 4
        val = x_min + (x_max - x_min) * i / 4
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" font-size="11" '
            f'fill="#888780" font-family="system-ui, sans-serif">{val:.0f}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 40:.1f}" text-anchor="middle" font-size="12" '
        f'fill="#444441" font-family="system-ui, sans-serif">单请求时延 p50（毫秒）</text>'
    )
    for i, p in enumerate(sorted(pts, key=lambda q: float(q["p50_latency"]))):
        cx, cy = px(float(p["p50_latency"])), py(float(p["macro_score"]) * 100)
        color = PALETTE[i % len(PALETTE)]
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}"/>')
        parts.append(
            f'<text x="{cx + 10:.1f}" y="{cy - 8:.1f}" font-size="12" fill="#2C2C2A" '
            f'font-family="system-ui, sans-serif">{html.escape(str(p["label"]))}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{top - 8}" font-size="12" fill="#5F5E5A" font-family="system-ui, sans-serif">'
        f"宏平均准确率（越高越好）</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_latency_bars(rows: Sequence[Dict[str, Any]], *, width: int = 900) -> str:
    """每个模型的时延分位横向条形图。"""
    if not rows:
        return ""
    row_h, gap, left, right, top = 34, 12, 130, 60, 12
    height = top + (row_h + gap) * len(rows)
    plot_w = width - left - right
    max_lat = max((float(r.get("latency_ms", {}).get("p95") or 0) for r in rows), default=1.0) or 1.0

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">',
        "<title>时延分位</title>",
    ]
    for i, r in enumerate(rows):
        y = top + i * (row_h + gap)
        p50 = float((r.get("latency_ms") or {}).get("p50") or 0)
        p95 = float((r.get("latency_ms") or {}).get("p95") or 0)
        parts.append(
            f'<text x="{left - 10}" y="{y + row_h / 2:.1f}" text-anchor="end" dominant-baseline="central" '
            f'font-size="12" fill="#444441" font-family="system-ui, sans-serif">{html.escape(str(r.get("model_label")))}</text>'
        )
        parts.append(f'<rect x="{left}" y="{y}" width="{plot_w * p95 / max_lat:.1f}" height="{row_h}" rx="3" fill="#E6F1FB"/>')
        parts.append(f'<rect x="{left}" y="{y + 6}" width="{plot_w * p50 / max_lat:.1f}" height="{row_h - 12}" rx="3" fill="#185FA5"/>')
        parts.append(
            f'<text x="{left + plot_w * p95 / max_lat + 8:.1f}" y="{y + row_h / 2:.1f}" dominant-baseline="central" '
            f'font-size="11" fill="#5F5E5A" font-family="system-ui, sans-serif">p50 {p50:.0f} / p95 {p95:.0f} ms</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Markdown 报告
# --------------------------------------------------------------------------- #
def build_markdown(summary: Dict[str, Any], *, baseline: Optional[str] = None, pairs: Optional[List[List[str]]] = None) -> str:
    run_dir = summary["run_dir"]
    labels = summary.get("model_labels") or {}
    tasks = available_tasks(summary)
    models = model_order(summary)
    results = summary.get("results") or {}
    manifest = summary.get("manifest") or {}

    L: List[str] = []
    _sec = [0]

    def nxt() -> int:
        """章节号自增：避免可选章节缺失时编号出现跳号。"""
        _sec[0] += 1
        return _sec[0]

    L.append(f"# 通用能力评测报告 · {summary.get('run_name')}")
    L.append("")
    L.append(f"- 生成时间：{now_iso()}")
    L.append(f"- 运行目录：`{run_dir}`")
    L.append(f"- 评测框架版本：{manifest.get('geval_version')} ｜ Python {manifest.get('python')} ｜ {manifest.get('platform')}")
    L.append(f"- 被测模型：{', '.join(labels.get(m, m) for m in models)}")
    L.append(f"- 任务：{', '.join(tasks)}")
    if manifest.get("limit_override"):
        L.append(f"- ⚠️ 抽样上限：{manifest['limit_override']} 条/任务（结果仅供横向比较，不能与公开榜单直接对比）")
    L.append("")

    # 1. 准确率矩阵
    L.append(f"## {nxt()}. 准确率矩阵（± 为 bootstrap 95% 半宽）")
    L.append("")
    rows = []
    for mid in models:
        row = [labels.get(mid, mid)]
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            if agg.get("score") is None:
                row.append("—")
            else:
                lo, hi = (agg.get("ci_bootstrap") or [None, None])[:2]
                half = (hi - lo) / 2 * 100 if lo is not None and hi is not None else None
                row.append(f"{agg['score'] * 100:.1f}%" + (f" ±{half:.1f}" if half else ""))
        rows.append(row)
    L.append(md_table(["模型"] + tasks, rows, ["l"] + ["r"] * len(tasks)))
    L.append("")

    # 2. 宏平均 + 时延
    macro = macro_scores(summary, tasks)
    L.append(f"## {nxt()}. 综合表现与时延")
    L.append("")
    rows = []
    for m in macro:
        rows.append(
            [
                m["label"],
                fmt_pct(m["macro_score"], 1),
                str(m["n_tasks"]),
                fmt_num(m["p50_latency"], 0),
                fmt_num(m["p95_latency"], 0),
            ]
        )
    L.append(md_table(["模型", "宏平均准确率", "覆盖任务数", "时延 p50/ms", "时延 p95/ms"], rows, ["l", "r", "r", "r", "r"]))
    L.append("")
    L.append("> 宏平均为等权平均，只统计该模型实际运行过的任务；未跑全的任务不要与其他模型直接比较宏平均。")
    L.append("")

    # 3. 显著性
    base = baseline or (models[-1] if models else None)
    if base and len(models) > 1:
        L.append(f"## {nxt()}. 与基线「{labels.get(base, base)}」的配对显著性检验")
        L.append("")
        rows = []
        for mid in models:
            if mid == base:
                continue
            for t in tasks:
                cmp = compare(run_dir, mid, base, t)
                if not cmp:
                    continue
                rows.append(
                    [
                        labels.get(mid, mid),
                        t,
                        fmt_pct(cmp["acc_a"], 1),
                        fmt_pct(cmp["acc_b"], 1),
                        f"{cmp['diff'] * 100:+.2f} pt",
                        f"[{cmp['ci'][0] * 100:+.2f}, {cmp['ci'][1] * 100:+.2f}]",
                        f"{cmp['p_mcnemar']:.4f}",
                        cmp["label"],
                    ]
                )
        if rows:
            L.append(
                md_table(
                    ["模型", "任务", "本模型", "基线", "差值", "差值 95% CI", "McNemar p", "结论"],
                    rows,
                    ["l", "l", "r", "r", "r", "r", "r", "l"],
                )
            )
            L.append("")
            L.append(
                "> 差值 CI 跨过 0，或 McNemar p ≥ 0.05，说明该任务上的差距**可能只是抽样噪声**，"
                "不应据此下结论。只有「显著」级别的差距才值得作为选型依据。"
            )
            L.append("")
    elif len(models) < 2:
        # 单模型场景（例如只有一条模型通道，先验证流程联通性）
        L.append(f"## {nxt()}. 配对显著性检验")
        L.append("")
        L.append(
            "> 本次只有 1 个模型参与评测。配对显著性检验需要**至少两个模型跑同一批样本**才有意义，"
            "已跳过。补齐其他模型通道后重新运行，该章节会自动出现。"
        )
        L.append("")

    # 4. 微调影响评估（阶段 1.5 的核心产出）
    #    注意 nxt() 只在真正输出章节时调用，否则单模型运行时章节号会跳号
    if pairs:
        impact = build_impact_section(summary, pairs, tasks)
        if impact:
            L.append(f"## {nxt()}. 微调影响评估：通用能力代价（微调前 / 微调后）")
            L.append("")
            L.extend(impact)

    # 5. 解析质量（诊断「是模型错还是抽取器错」）
    L.append(f"## {nxt()}. 解析质量与抽取策略分布")
    L.append("")
    rows = []
    for mid in models:
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            note = agg.get("parse_note") or {}
            if not note:
                continue
            top = sorted((note.get("strategies") or {}).items(), key=lambda kv: -kv[1])[:3]
            rows.append(
                [
                    labels.get(mid, mid),
                    t,
                    fmt_pct(note.get("parse_fail_rate"), 2),
                    fmt_pct(agg.get("ambiguous_rate"), 2),
                    ", ".join(f"{k}:{v}" for k, v in top),
                ]
            )
    if rows:
        L.append(md_table(["模型", "任务", "解析失败率", "歧义率", "主要抽取策略"], rows, ["l", "l", "r", "r", "l"]))
        L.append("")
        L.append("> 解析失败率偏高（>2%）时，先改 prompt 协议或补充抽取规则，再谈模型能力。")
        L.append("")

    # 续写似然打分（mode: loglikelihood）的 acc / acc_norm 对照
    lk_rows = []
    for mid in models:
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            ex = agg.get("extra") or {}
            if ex.get("acc_correct") is None or ex.get("acc_norm_correct") is None:
                continue
            lk_rows.append(
                [
                    labels.get(mid, mid),
                    t,
                    fmt_pct(ex.get("acc_correct"), 1),
                    fmt_pct(ex.get("acc_norm_correct"), 1),
                    fmt_num(ex.get("acc_norm_margin"), 2),
                    fmt_num(ex.get("n_requests"), 1),
                ]
            )
    if lk_rows:
        L.append(f"## {nxt()}. 续写似然打分：acc 与 acc_norm 对照（lm-eval 口径）")
        L.append("")
        L.append(
            md_table(
                ["模型", "任务", "acc", "acc_norm", "平均 acc_norm 间隔", "平均请求数/样本"],
                lk_rows,
                ["l", "l", "r", "r", "r", "r"],
            )
        )
        L.append("")
        L.append(
            "> `acc` 用 logprob 之和，`acc_norm` 做长度归一化（lm-eval 在选择题上的主指标）。"
            "两者差异大的任务，说明模型的偏好受「选项文本长度」影响明显，"
            "此时 acc_norm 更能反映真实的知识差异。"
        )
        L.append("")

    # 6. 结构化输出专项（与意图识别最相关）
    json_rows = []
    for mid in models:
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            tm = agg.get("task_metrics") or {}
            if tm.get("json_valid_rate") is None:
                continue
            json_rows.append(
                [
                    labels.get(mid, mid),
                    fmt_pct(tm.get("json_valid_rate"), 1),
                    fmt_pct(tm.get("schema_ok_rate"), 1),
                    fmt_pct(tm.get("intent_accuracy"), 1),
                    fmt_pct(tm.get("slots_exact_rate"), 1),
                    fmt_pct(tm.get("extra_prose_rate"), 1),
                ]
            )
    if json_rows:
        L.append(f"## {nxt()}. 结构化输出专项（意图 + 槽位）")
        L.append("")
        L.append(md_table(["模型", "JSON 可解析率", "Schema 通过率", "意图准确率", "槽位完全一致率", "夹带解释率"], json_rows, ["l", "r", "r", "r", "r", "r"]))
        L.append("")

    # 7. IFEval 专项
    if_rows = []
    for mid in models:
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            tm = agg.get("task_metrics") or {}
            if tm.get("prompt_level_accuracy") is None:
                continue
            if_rows.append(
                [
                    labels.get(mid, mid),
                    fmt_pct(tm.get("prompt_level_accuracy"), 1),
                    fmt_pct(tm.get("instruction_level_accuracy"), 1),
                    f"{tm.get('n_covered_types', 0)}/{tm.get('n_instruction_types', 0)}",
                ]
            )
    if if_rows:
        L.append(f"## {nxt()}. 指令遵循专项（IFEval）")
        L.append("")
        L.append(md_table(["模型", "样本级通过率", "指令级通过率", "覆盖指令类型"], if_rows, ["l", "r", "r", "r"]))
        L.append("")
        L.append("> 未实现的指令类型已从统计中排除，不会伪装成「不通过」。扩展方式见 geval/ifeval.py 的 CHECKERS。")
        L.append("")

    # 8. 分组短板
    L.append(f"## {nxt()}. 分组明细（前 20 组，用于定位能力短板）")
    L.append("")
    for t in tasks:
        rows_by_model = []
        for mid in models:
            agg = (results.get(mid) or {}).get(t) or {}
            for item in agg.get("breakdown") or []:
                if item["n"] < 3:
                    continue
                rows_by_model.append(
                    [labels.get(mid, mid), item["subject"], item["n"], fmt_pct(item["score"], 1), item["parse_fail"]]
                )
        if rows_by_model:
            L.append(f"### {t}")
            L.append("")
            L.append(md_table(["模型", "分组", "n", "准确率", "解析失败"], rows_by_model[:60], ["l", "l", "r", "r", "r"]))
            L.append("")

    # 9. 错例抽样
    L.append(f"## {nxt()}. 错例抽样（人工复核用）")
    L.append("")
    shown = 0
    for mid in models:
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            for ex in (agg.get("examples_failed") or [])[:2]:
                if shown >= 15:
                    break
                L.append(f"**{labels.get(mid, mid)} · {t} · {ex.get('id')}**（解析策略 {ex.get('strategy')}，解析成功 {ex.get('parse_ok')}）")
                L.append("")
                L.append(f"- 标准答案：`{ex.get('gold')}`")
                L.append(f"- 模型判定：`{ex.get('pred')}`")
                L.append(f"- 原始输出：{str(ex.get('response') or '')[:300]}")
                L.append("")
                shown += 1
    if shown == 0:
        L.append("本次运行没有产生错例。")
        L.append("")

    # 10. 复现信息
    L.append(f"## {nxt()}. 复现信息")
    L.append("")
    L.append("```json")
    L.append(jdump((summary.get("manifests") or {}).get("datasets") or {}, indent=2))
    L.append("```")
    L.append("")
    if summary.get("errors"):
        L.append("### 运行期错误")
        L.append("")
        for err in summary["errors"][:30]:
            L.append(f"- {err}")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# HTML 报告（自包含、内联 SVG、无外部引用）
# --------------------------------------------------------------------------- #
def build_html(
    summary: Dict[str, Any],
    md_text: str,
    *,
    baseline: Optional[str] = None,
    pairs: Optional[List[List[str]]] = None,
) -> str:
    run_dir = summary["run_dir"]
    labels = summary.get("model_labels") or {}
    tasks = available_tasks(summary)
    models = model_order(summary)
    results = summary.get("results") or {}

    chart_series = []
    for i, mid in enumerate(models):
        vals = []
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            vals.append(round(agg["score"] * 100, 1) if agg.get("score") is not None else None)
        chart_series.append({"label": labels.get(mid, mid), "values": vals, "color": PALETTE[i % len(PALETTE)]})

    bar_svg = svg_grouped_bars(tasks, chart_series)
    macro = macro_scores(summary, tasks)
    pareto_svg = svg_pareto(macro)

    latency_rows = []
    for t in tasks:
        for mid in models:
            agg = (results.get(mid) or {}).get(t) or {}
            if agg.get("latency_ms"):
                latency_rows.append(
                    {
                        "model_label": f"{labels.get(mid, mid)} · {t}",
                        "latency_ms": agg["latency_ms"],
                    }
                )
    latency_svg = svg_latency_bars(latency_rows[:12])

    # 显著性表
    base = baseline or (models[-1] if models else None)
    sig_rows = []
    if base:
        for mid in models:
            if mid == base:
                continue
            for t in tasks:
                cmp = compare(run_dir, mid, base, t)
                if not cmp:
                    continue
                cls = "better" if cmp["diff"] > 0 and cmp["p_mcnemar"] < 0.05 else ("worse" if cmp["diff"] < 0 and cmp["p_mcnemar"] < 0.05 else "flat")
                sig_rows.append(
                    f"<tr><td>{html.escape(labels.get(mid, mid))}</td><td>{html.escape(t)}</td>"
                    f"<td class='num'>{cmp['acc_a'] * 100:.1f}%</td><td class='num'>{cmp['acc_b'] * 100:.1f}%</td>"
                    f"<td class='num {cls}'>{cmp['diff'] * 100:+.2f}</td>"
                    f"<td class='num'>[{cmp['ci'][0] * 100:+.2f}, {cmp['ci'][1] * 100:+.2f}]</td>"
                    f"<td class='num'>{cmp['p_mcnemar']:.4f}</td><td>{html.escape(cmp['label'])}</td></tr>"
                )

    acc_rows = []
    for mid in models:
        cells = []
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            if agg.get("score") is None:
                cells.append("<td class='num'>—</td>")
            else:
                lo, hi = (agg.get("ci_bootstrap") or [None, None])[:2]
                half = (hi - lo) / 2 * 100 if lo is not None and hi is not None else 0.0
                cells.append(f"<td class='num'>{agg['score'] * 100:.1f}<span class='sub'>±{half:.1f}</span></td>")
        acc_rows.append(f"<tr><th class='rowh'>{html.escape(labels.get(mid, mid))}</th>{''.join(cells)}</tr>")

    parse_rows = []
    for mid in models:
        for t in tasks:
            agg = (results.get(mid) or {}).get(t) or {}
            note = agg.get("parse_note") or {}
            if not note:
                continue
            top = sorted((note.get("strategies") or {}).items(), key=lambda kv: -kv[1])[:3]
            warn = " warn" if (note.get("parse_fail_rate") or 0) > 0.02 else ""
            parse_rows.append(
                f"<tr><td>{html.escape(labels.get(mid, mid))}</td><td>{html.escape(t)}</td>"
                f"<td class='num{warn}'>{(note.get('parse_fail_rate') or 0) * 100:.2f}%</td>"
                f"<td class='num'>{(agg.get('ambiguous_rate') or 0) * 100:.2f}%</td>"
                f"<td class='mono'>{html.escape(', '.join(f'{k}:{v}' for k, v in top))}</td></tr>"
            )

    json_rows = []
    for mid in models:
        for t in tasks:
            tm = ((results.get(mid) or {}).get(t) or {}).get("task_metrics") or {}
            if tm.get("json_valid_rate") is None:
                continue
            json_rows.append(
                f"<tr><td>{html.escape(labels.get(mid, mid))}</td>"
                f"<td class='num'>{tm['json_valid_rate'] * 100:.1f}%</td>"
                f"<td class='num'>{(tm.get('schema_ok_rate') or 0) * 100:.1f}%</td>"
                f"<td class='num'>{(tm.get('intent_accuracy') or 0) * 100:.1f}%</td>"
                f"<td class='num'>{(tm.get('slots_exact_rate') or 0) * 100:.1f}%</td>"
                f"<td class='num'>{(tm.get('extra_prose_rate') or 0) * 100:.1f}%</td></tr>"
            )

    task_headers = "".join(f"<th class='num'>{html.escape(t)}</th>" for t in tasks)
    manifest = summary.get("manifest") or {}

    # 显著性章节：只有 2 个以上模型才有意义。单模型时输出一个带编号的说明段，
    # 而不是「自己和自己比」的空表 —— 既让人知道为什么没有，也保证章节号连续。
    _sig_p = (
        "<p>仅当置信区间不跨 0 且 McNemar p &lt; 0.05 时，差距才可作为选型依据。"
        "差值正数表示本模型更准。</p>"
    )
    if len(models) < 2:
        sig_html = (
            "<h2>3 · 配对显著性检验</h2>"
            '<div class="note">本次只有 1 个模型参与评测，配对显著性检验需要至少两个模型才有意义，'
            "已跳过。补齐其他模型通道后重新运行，该章节会自动出现。</div>"
        )
    elif sig_rows:
        sig_html = (
            f'<h2>3 · 与基线 {html.escape(labels.get(base, str(base)))} 的配对显著性检验</h2>'
            + _sig_p
            + '<div class="card"><table><thead><tr><th>模型</th><th>任务</th><th class="num">本模型</th>'
            '<th class="num">基线</th><th class="num">差值(pt)</th><th class="num">差值 95% CI</th>'
            '<th class="num">McNemar p</th><th>结论</th></tr></thead>'
            f"<tbody>{''.join(sig_rows)}</tbody></table></div>"
        )
    else:
        sig_html = (
            "<h2>3 · 配对显著性检验</h2>"
            '<div class="note">参与的模型之间没有可比任务（或共同样本不足），'
            "该章节暂不可用。确认各模型跑的是同一批样本后重跑。</div>"
        )

    # 微调影响评估（阶段 1.5）：pre → post 的逐任务差值与判定
    impact_blocks: List[str] = []
    for a, b in pairs or []:
        if a not in (summary.get("results") or {}) or b not in (summary.get("results") or {}):
            continue
        rows_html: List[str] = []
        warn_html: List[str] = []
        deltas: List[float] = []
        n_deg = n_imp = 0
        for t in tasks:
            aa = ((summary.get("results") or {}).get(a) or {}).get(t) or {}
            ab = ((summary.get("results") or {}).get(b) or {}).get(t) or {}
            cmp = compare(run_dir, a, b, t)
            if not cmp:
                continue
            for issue in compatible_config(a, b, aa, ab):
                warn_html.append(f"⚠️ {html.escape(t)}：{html.escape(issue)}")
            for w in cmp["warnings"]:
                warn_html.append(f"⚠️ {html.escape(t)}：{html.escape(w)}")
            imp = impact_delta(cmp)
            judge = degradation_label(imp["delta"], cmp["p_mcnemar"])
            deltas.append(imp["delta"])
            if "退化" in judge:
                n_deg += 1
            elif "提升" in judge:
                n_imp += 1
            cls = "worse" if "退化" in judge else ("better" if "提升" in judge else "flat")
            rows_html.append(
                f"<tr><td>{html.escape(t)}</td>"
                f"<td class='num'>{cmp['acc_a'] * 100:.1f}%</td>"
                f"<td class='num'>{cmp['acc_b'] * 100:.1f}%</td>"
                f"<td class='num {cls}'>{imp['delta'] * 100:+.2f}</td>"
                f"<td class='num'>[{imp['ci'][0] * 100:+.2f}, {imp['ci'][1] * 100:+.2f}]</td>"
                f"<td class='num'>{cmp['p_mcnemar']:.4f}</td>"
                f"<td class='{cls}'>{html.escape(judge)}</td></tr>"
            )
        if not rows_html:
            continue
        avg = (sum(deltas) / len(deltas) * 100) if deltas else 0.0
        impact_blocks.append(
            f"<h3>{html.escape(labels.get(a, a))} → {html.escape(labels.get(b, b))}</h3>"
            + (''.join(f'<div class="note">{w}</div>' for w in warn_html) if warn_html else '')
            + f"""<div class="grid">
<div class="kpi"><div class="k">宏平均变化</div><div class="v {'worse' if avg < 0 else ('better' if avg > 0 else '')}">{avg:+.2f} pt</div><div class="k">跨 {len(deltas)} 个任务等权平均</div></div>
<div class="kpi"><div class="k">显著退化任务</div><div class="v worse">{n_deg}</div><div class="k">p &lt; 0.05 且为负</div></div>
<div class="kpi"><div class="k">显著提升任务</div><div class="v better">{n_imp}</div><div class="k">p &lt; 0.05 且为正</div></div>
</div>
<table><thead><tr><th>任务</th><th class="num">微调前</th><th class="num">微调后</th><th class="num">差值(pt)</th><th class="num">差值 95% CI</th><th class="num">McNemar p</th><th>判定</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>"""
        )
    impact_html = (
        """<h2>3b · 微调影响评估（通用能力代价）</h2>
<p>阶段 1.5 的核心产出：同一个模型微调前后在通用能力上的变化。只有达到统计显著的才判定为「退化 / 提升」。</p>
<div class="card">"""
        + "".join(impact_blocks)
        + """</div>
<div class="note"><strong>两个前提必须成立，否则上表不可用：</strong>
① 两次运行使用<strong>同一批样本、同一套 prompt</strong>（配置不一致时表内会出告警）；
② 微调<strong>没有改动 chat template</strong>。若改过，差值里混入了 prompt 格式差异，
需把微调前的模型也用新模板重测，或至少在结论中显式标注该混淆因素。</div>"""
        if impact_blocks
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>通用能力评测报告 · {html.escape(str(summary.get('run_name')))}</title>
<style>
:root{{--bg:#FFFFFF;--panel:#F8F7F4;--line:#E4E2DB;--text:#2C2C2A;--muted:#5F5E5A;--hint:#888780;--blue:#185FA5;--green:#0F6E56;--red:#A32D2D;--amber:#854F0B}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.65}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 64px}}
h1{{font-size:22px;font-weight:500;margin:0 0 6px}}
h2{{font-size:17px;font-weight:500;margin:38px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}}
h3{{font-size:15px;font-weight:500;margin:22px 0 8px}}
p{{margin:8px 0;color:var(--muted)}}
.meta{{color:var(--hint);font-size:13px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
thead th{{color:var(--muted);font-weight:500;background:#F1EFE8}}
th.rowh{{color:var(--text);font-weight:500}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.mono{{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--muted)}}
.sub{{color:var(--hint);font-size:11px;margin-left:2px}}
/* 准确率语义：提升=绿，退化=红。列内均有文字判定兜底，避免颜色歧义。 */
.better{{color:var(--green);font-weight:500}}
.worse{{color:var(--red);font-weight:500}}
.flat{{color:var(--hint)}}
.warn{{color:var(--amber);font-weight:500}}
.note{{background:#FAEEDA;border:1px solid #FAC775;border-radius:10px;padding:10px 14px;font-size:13px;color:#633806;margin:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}}
.kpi .k{{font-size:12px;color:var(--hint)}}
.kpi .v{{font-size:20px;font-weight:500;margin-top:2px}}
details{{margin:10px 0}}
summary{{cursor:pointer;color:var(--blue);font-size:13px}}
pre{{background:#F1EFE8;border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:12px}}
</style>
</head>
<body>
<div class="wrap">
<h1>通用能力评测报告 · {html.escape(str(summary.get('run_name')))}</h1>
<p class="meta">生成时间 {now_iso()} ｜ 框架 geval {html.escape(str(manifest.get('geval_version')))} ｜ Python {html.escape(str(manifest.get('python')))} ｜ {html.escape(str(manifest.get('platform')))}</p>
{('<div class="note">本次为抽样运行，limit = ' + str(manifest.get('limit_override')) + ' 条/任务。结果适合横向比较，不可与公开榜单数字直接对齐。</div>') if manifest.get('limit_override') else ''}

<h2>1 · 准确率矩阵</h2>
<p>数值为准确率（%），± 为 bootstrap 95% 置信区间半宽。判断两个模型是否有差距，请以第 3 节的配对检验为准，不要只看这里的小数点。</p>
<div class="card">
<table><thead><tr><th class="rowh">模型</th>{task_headers}</tr></thead>
<tbody>{''.join(acc_rows)}</tbody></table>
</div>
<div class="card">{bar_svg}</div>

<h2>2 · 综合表现与时延</h2>
<div class="grid">
{''.join(f'<div class="kpi"><div class="k">{html.escape(m["label"])}</div><div class="v">{fmt_pct(m["macro_score"],1)}</div><div class="k">宏平均 · 覆盖 {m["n_tasks"]} 个任务</div><div class="k">p50 {fmt_num(m["p50_latency"],0)} ms · p95 {fmt_num(m["p95_latency"],0)} ms</div></div>' for m in macro)}
</div>
<div class="card">{pareto_svg}</div>
<p>帕累托图：越靠左上越优（更准且更快）。如果一个小模型明显位于 32B 的左上侧，它就是这个场景下的性价比拐点。</p>
<div class="card">{latency_svg}</div>

{sig_html}

{impact_html}

<h2>4 · 解析质量与抽取策略</h2>
<p>解析失败率偏高意味着「模型其实答对了，但抽取器没读懂」。这类问题应该改 prompt 协议或抽取规则，而不是归因到模型能力。</p>
<div class="card">
<table><thead><tr><th>模型</th><th>任务</th><th class="num">解析失败率</th><th class="num">歧义率</th><th>主要抽取策略</th></tr></thead>
<tbody>{''.join(parse_rows) or '<tr><td colspan="5">—</td></tr>'}</tbody></table>
</div>

<h2>5 · 结构化输出专项（意图 + 槽位）</h2>
<p>生产环境中意图识别的失败集中在第一、二层：输出不可解析、schema 不符，而不是分类本身错。</p>
<div class="card">
<table><thead><tr><th>模型</th><th class="num">JSON 可解析率</th><th class="num">Schema 通过率</th><th class="num">意图准确率</th><th class="num">槽位完全一致率</th><th class="num">夹带解释率</th></tr></thead>
<tbody>{''.join(json_rows) or '<tr><td colspan="6">未运行 json_schema 任务</td></tr>'}</tbody></table>
</div>

<h2>6 · 复现信息</h2>
<details><summary>展开数据集与运行指纹</summary>
<pre>{html.escape(jdump((summary.get('manifests') or {}).get('datasets') or {}, indent=2))}</pre>
</details>
<details><summary>展开完整 Markdown 报告</summary>
<pre>{html.escape(md_text[:60000])}</pre>
</details>
</div>
</body>
</html>"""


# --------------------------------------------------------------------------- #
def build_report(
    run_dir: str,
    *,
    baseline: Optional[str] = None,
    pairs: Optional[List[List[str]]] = None,
) -> Dict[str, str]:
    summary = load_run(run_dir)
    md_text = build_markdown(summary, baseline=baseline, pairs=pairs)
    html_text = build_html(summary, md_text, baseline=baseline, pairs=pairs)
    md_path = os.path.join(run_dir, "report.md")
    html_path = os.path.join(run_dir, "report.html")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return {"markdown": md_path, "html": html_path}
