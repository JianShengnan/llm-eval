# -*- coding: utf-8 -*-
"""专业能力评测专项报告：从 run 目录的逐样本明细重算指标并落盘。

用法（项目根目录执行）：
    python scripts/professional_report.py runs/geval-professional-intent
    python scripts/professional_report.py runs/<run> --cost-in 0.5 --cost-out 2.0

产出：
    runs/<run>/professional_report.md      人读：结论 / 分层 / 混淆 / 成本与时延
    runs/<run>/professional_metrics.json   机读：同一批指标，便于接入看板

为什么不改 report.py：原报告 1300 行、章节号动态、HTML 内联 SVG，
插入专业章节风险高；这里独立读明细重算，原报告与自检完全不受影响。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from geval.metrics import mcnemar_exact_p, paired_bootstrap_diff, significance_label  # noqa: E402
from geval.telemetry import usage_by_group, usage_summary  # noqa: E402
from geval.util import md_table, read_json, read_jsonl  # noqa: E402


def find_detail_files(run_dir: str, task_key: str) -> List[str]:
    out: List[str] = []
    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            if name.endswith(".jsonl") and task_key in name:
                out.append(os.path.join(root, name))
    return sorted(out)


def pct(x: Optional[float], digits: int = 2) -> str:
    return "—" if x is None else ("%.*f%%" % (digits, x * 100))


def num(x: Optional[float], digits: int = 1) -> str:
    return "—" if x is None else ("%.*f" % (digits, x))


def load_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        for r in read_jsonl(p):
            if r.get("task") == "intent_routing" or "sample_type" in r:
                rows.append(r)
    return rows


def fmt_usage(u: Dict[str, Any]) -> List[List[str]]:
    lat = u.get("latency_ms") or {}
    cost = u.get("estimated_cost")
    return [
        ["样本数 / 有效请求", str(u["n"])],
        ["prompt tokens 合计", str(u["prompt_tokens"])],
        ["completion tokens 合计", str(u["completion_tokens"])],
        ["total tokens 合计", str(u["total_tokens"])],
        ["平均 prompt / completion / total", "%s / %s / %s" % (
            num(u["avg_prompt_tokens"]), num(u["avg_completion_tokens"]), num(u["avg_total_tokens"]))],
        ["单请求时延 mean / p50 / p90 / p95 / p99 (ms)",
         " / ".join(num(lat.get(k), 0) for k in ("mean", "p50", "p90", "p95", "p99"))],
        ["单请求时延最大 (ms)", num(lat.get("max"), 0)],
        ["累计请求时延 (s)", num(u["cumulative_request_ms"] / 1000.0, 2)],
        ["运行墙钟时间 (s)", num(u["run_wall_clock_s"], 2)],
        ["completion tokens / s", num(u["completion_tokens_per_s"])],
        ["样本吞吐 (样本/s，按累计时延)", num(u["samples_per_s"], 2)],
        ["缓存命中 / 输出被截断", "%d / %d" % (u["cached_hits"], u["length_truncated"])],
        ["估算成本（元或计价单位）", num(cost, 4) if cost is not None else "未配置单价"],
    ]


def build_report(run_dir: str, *, model_filter: Optional[str] = None,
                 task_key: str = "intent-routing",
                 price_in: float = 0.0, price_out: float = 0.0) -> Dict[str, Any]:
    summary = read_json(os.path.join(run_dir, "summary.json"), {}) or {}
    manifest = read_json(os.path.join(run_dir, "manifest.json"), {}) or {}
    cfg = read_json(os.path.join(run_dir, "config.json"), {}) or {}
    cost_cfg = cfg.get("cost") or {}
    dinfo = read_json(os.path.join(PROJECT_ROOT, "datasets", "intent_routing",
                                  "dataset_info.json"), {}) or {}
    if price_in == 0.0:
        price_in = float(cost_cfg.get("prompt_per_1k") or 0)
    if price_out == 0.0:
        price_out = float(cost_cfg.get("completion_per_1k") or 0)

    files = find_detail_files(run_dir, task_key)
    if model_filter:
        files = [p for p in files if model_filter in os.path.relpath(p, run_dir)]
    if not files:
        raise SystemExit("在 %s 下找不到 %s 的明细 JSONL" % (run_dir, task_key))
    rows = load_rows(files)
    if not rows:
        raise SystemExit("明细为空")
    errors = [r for r in rows if r.get("error")]
    valid = [r for r in rows if not r.get("error")]
    n = len(valid)
    n_paths = int(valid[0].get("intent_tree_size") or 0) if valid else 0

    def ex(r: Dict[str, Any], key: str) -> bool:
        return bool((r.get("extras") or {}).get(key))

    a = [bool(r.get("correct")) for r in valid]
    b = [ex(r, "small_model_correct") for r in valid]
    llm_acc = sum(a) / n if n else 0.0
    small_acc = sum(b) / n if n else 0.0
    mc = mcnemar_exact_p(a, b)
    pb = paired_bootstrap_diff(a, b, n_boot=2000, seed=20261110)
    usage = usage_summary(valid, price_in_per_1k=price_in, price_out_per_1k=price_out)

    conf = Counter((str(r.get("gold")), str(r.get("pred"))) for r in valid if not r.get("correct"))
    conf_rows = [[g, p, str(c)] for (g, p), c in conf.most_common(12)]

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in valid:
        by_type[str(r.get("sample_type") or "unknown")].append(r)
    group_rows = []
    for g, rs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        u = usage_summary(rs, price_in_per_1k=price_in, price_out_per_1k=price_out)
        group_rows.append([
            g, str(len(rs)), pct(sum(bool(r.get("correct")) for r in rs) / len(rs)),
            pct(sum(ex(r, "small_model_correct") for r in rs) / len(rs)),
            str(u["total_tokens"]), num(u["avg_total_tokens"]),
            num((u["latency_ms"] or {}).get("p95"), 0),
        ])

    intent_rows = usage_by_group(valid, key="gold", price_in_per_1k=price_in,
                                 price_out_per_1k=price_out)[:12]
    intent_table = [[r["gold"], str(r["n"]), pct(r["intent_accuracy"]),
                     str(r["total_tokens"]), num(r["avg_total_tokens"]),
                     num(r["p95_latency_ms"], 0)] for r in intent_rows]

    strategies = Counter(str(r.get("strategy") or "none") for r in valid)
    fail_rows = [
        ["意图准确率（严格全等）", pct(llm_acc)],
        ["归一化后准确率", pct(sum(ex(r, "normalized_correct") for r in valid) / n if n else 0)],
        ["小模型基线准确率", pct(small_acc)],
        ["大模型相对增益", pct(llm_acc - small_acc)],
        ["增益 95% CI", "[%s, %s]" % (pct(pb.get("lo")), pct(pb.get("hi")))],
        ["McNemar 配对检验 p / 判定", "%s / %s" % (
            num(mc.get("p_two_sided"), 4), significance_label(mc.get("p_two_sided")))],
        ["输出落在候选树内比率", pct(sum(ex(r, "in_tree") for r in valid) / n if n else 0)],
        ["幻觉意图率（树外标签）", pct(sum(ex(r, "hallucinated_intent") for r in valid) / n if n else 0)],
        ["解析失败率", pct(sum(1 for r in valid if not r.get("parse_ok")) / n if n else 0)],
        ["夹带解释文字率", pct(sum(ex(r, "extra_prose") for r in valid) / n if n else 0)],
        ["输出被 max_tokens 截断率", pct(sum(ex(r, "length_truncated") for r in valid) / n if n else 0)],
    ]

    err_examples = [
        [str(r.get("id")), str(r.get("gold")), str(r.get("pred")),
         str(r.get("strategy")), (str(r.get("response") or ""))[:80].replace("\n", " ")]
        for r in valid if not r.get("correct")
    ][:10]

    L: List[str] = []
    L.append("# 专业能力评测报告 · 语音导航意图树路由")
    L.append("")
    L.append("生成时间：%s　|　运行目录：`%s`" % (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), os.path.abspath(run_dir)))
    L.append("")
    L.append("## 1. 结论摘要")
    L.append("")
    L.append("> 主问题：**大模型在业务意图树上是否显著优于现有小模型，代价是多少 token 与时延？**")
    L.append("")
    L.append(md_table(["指标", "数值"], fail_rows, ["l", "r"]))
    L.append("")
    if n:
        verdict = ("大模型显著优于小模型" if (mc.get("p_two_sided") or 1) < 0.05 and llm_acc > small_acc
                   else "增益不显著，不足以支撑替换" if (mc.get("p_two_sided") or 1) >= 0.05
                   else "大模型显著劣于小模型")
        L.append("**判定：%s。** 只有 p < 0.05 的差距才可作为选型依据；"
                 "同时需结合下方 token / 时延与「通用能力退化」一起看。" % verdict)
    L.append("")
    L.append("## 2. 评测口径与复现信息")
    L.append("")
    L.append(md_table(["项目", "值"], [
        ["样本数 / 请求失败数", "%d / %d" % (len(rows), len(errors))],
        ["候选意图树", "%d 条层级路径 / %s 个末级标签" % (
            n_paths, dinfo.get("n_labels") or "未记录")],
        ["测试集 sha256", str(dinfo.get("source_sha256") or "未记录")],
        ["测试集样本数（冻结时）", str(dinfo.get("n_records") or "未记录")],
        ["geval 版本 / Python / 平台", "%s / %s / %s" % (
            manifest.get("geval_version"), manifest.get("python"), manifest.get("platform"))],
        ["并发数 / seed", "%s / %s" % (manifest.get("concurrency"), manifest.get("seed"))],
        ["模型", ", ".join(str(m.get("id")) for m in (manifest.get("models") or [])) or "—"],
    ], ["l", "l"]))
    L.append("")
    L.append("> 逐样本明细含 `wall_start` / `wall_end` / `request_wall_ms` / `total_tokens` / "
             "`cached` / `finish_reason`，可据此复核任一请求。")
    L.append("")
    L.append("## 3. 分层表现（整体分会被高频简单意图主导，必须分层看）")
    L.append("")
    L.append(md_table(["样本类型", "n", "大模型准确率", "小模型准确率",
                       "总 tokens", "平均 tokens/样本", "p95 时延(ms)"],
                      group_rows, ["l", "r", "r", "r", "r", "r", "r"]))
    L.append("")
    L.append("### 按真实意图 Top 12（成本归因）")
    L.append("")
    L.append(md_table(["真实意图", "n", "准确率", "总 tokens", "平均 tokens/样本", "p95 时延(ms)"],
                      intent_table, ["l", "r", "r", "r", "r", "r"]))
    L.append("")
    L.append("## 4. 意图混淆矩阵（Top 12 错误流向）")
    L.append("")
    L.append(md_table(["真实意图", "模型输出", "次数"], conf_rows, ["l", "l", "r"]) if conf_rows
             else "本次运行无错例。")
    L.append("")
    L.append("## 5. 答案抽取策略分布")
    L.append("")
    L.append(md_table(["策略", "次数"], [[k, str(v)] for k, v in strategies.most_common()], ["l", "r"]))
    L.append("")
    L.append("## 6. Token 与时间消耗（核心成本口径）")
    L.append("")
    L.append(md_table(["项目", "值"], fmt_usage(usage), ["l", "r"]))
    L.append("")
    L.append("> `单请求时延` 由客户端 perf_counter 计时；`累计请求时延` 是各请求时延之和，"
             "`运行墙钟时间` 由首条请求开始到末条请求结束计算，两者之比反映并行度。"
             "缓存命中请求不产生真实网络时延，已单列。成本按每 1K token 单价估算，"
             "单价未配置时显示「未配置单价」，不做臆造。")
    L.append("")
    cs = (summary.get("manifests") or {}).get("client_stats") or {}
    if cs:
        L.append("### 客户端侧汇总（与逐样本口径互为校验）")
        L.append("")
        cs_rows = []
        for mid, st in cs.items():
            u = st.get("usage") or {}
            cs_rows.append([mid, str(st.get("calls")), str(st.get("errors")),
                            str(st.get("cache_hits")), str(u.get("total_tokens")),
                            num(st.get("latency_ms_p50"), 0), num(st.get("latency_ms_p95"), 0)])
        L.append(md_table(["模型", "调用数", "错误数", "缓存命中", "服务端 usage 总 tokens",
                           "p50(ms)", "p95(ms)"], cs_rows, ["l", "r", "r", "r", "r", "r", "r"]))
        L.append("")
    L.append("## 7. 错例抽样（人工复核前 10 条）")
    L.append("")
    L.append(md_table(["样本ID", "真实意图", "模型输出", "抽取策略", "原始响应(截断)"],
                      err_examples, ["l", "l", "l", "l", "l"]) if err_examples
             else "本次运行无错例。")
    L.append("")

    md_path = os.path.join(run_dir, "professional_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    metrics = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "run_dir": os.path.abspath(run_dir),
        "n_samples": len(rows), "n_valid": n, "n_errors": len(errors),
        "intent_accuracy": llm_acc, "small_model_accuracy": small_acc,
        "paired_uplift": llm_acc - small_acc,
        "uplift_ci95": [pb.get("lo"), pb.get("hi")],
        "mcnemar_p": mc.get("p_two_sided"),
        "top_confusions": [{"gold": g, "pred": p, "n": c} for (g, p), c in conf.most_common(12)],
        "usage": usage,
        "by_group": usage_by_group(valid, key="sample_type",
                                   price_in_per_1k=price_in, price_out_per_1k=price_out),
    }
    json_path = os.path.join(run_dir, "professional_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return {"md": md_path, "json": json_path, "metrics": metrics}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="专业能力评测专项报告")
    ap.add_argument("run", nargs="?", help="run 目录；缺省取 runs/ 下最近修改的目录")
    ap.add_argument("--model", help="只看指定模型 id")
    ap.add_argument("--task-key", default="intent-routing", help="明细文件名中的任务 tag")
    ap.add_argument("--cost-in", type=float, default=0.0, help="每 1K prompt tokens 单价")
    ap.add_argument("--cost-out", type=float, default=0.0, help="每 1K completion tokens 单价")
    args = ap.parse_args(argv)

    run_dir = args.run
    if not run_dir:
        base = os.path.join(PROJECT_ROOT, "runs")
        candidates = [os.path.join(base, d) for d in os.listdir(base)
                      if os.path.isdir(os.path.join(base, d))] if os.path.isdir(base) else []
        if not candidates:
            raise SystemExit("runs/ 下没有运行目录")
        run_dir = max(candidates, key=os.path.getmtime)
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit("run 目录不存在：%s" % run_dir)

    out = build_report(run_dir, model_filter=args.model, task_key=args.task_key,
                       price_in=args.cost_in, price_out=args.cost_out)
    m = out["metrics"]
    print("专业能力专项报告已生成")
    print("  %s" % out["md"])
    print("  %s" % out["json"])
    print("  样本 %d（有效 %d，失败 %d）｜意图准确率 %s｜小模型基线 %s｜增益 %s｜p=%s" % (
        m["n_samples"], m["n_valid"], m["n_errors"], pct(m["intent_accuracy"]),
        pct(m["small_model_accuracy"]), pct(m["paired_uplift"]), num(m["mcnemar_p"], 4)))
    u = m["usage"]
    print("  total tokens %d｜平均 %s/样本｜p95 %s ms｜墙钟 %s s｜成本 %s" % (
        u["total_tokens"], num(u["avg_total_tokens"]),
        num((u["latency_ms"] or {}).get("p95"), 0), num(u["run_wall_clock_s"], 2),
        "未配置单价" if u["estimated_cost"] is None else num(u["estimated_cost"], 4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
