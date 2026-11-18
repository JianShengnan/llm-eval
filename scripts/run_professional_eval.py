# -*- coding: utf-8 -*-
"""一键跑「语音导航意图识别」专业能力评测。

把手册里的 5 步流水线封装成一条命令：前置校验 → 端点探活 → 小样/全量推理 →
断点续跑 → 通用报告 + 专业专项报告。真实模型端点从 config.json 读取，
**只在内存中注入，不写出含 api_key 的配置文件**（runner 的快照会自动打码）。

    # 1) 只看将要花多少请求，不发任何请求
    python scripts/run_professional_eval.py --dry-run

    # 2) 离线验证整条链路（mock 模型，不联网）
    python scripts/run_professional_eval.py --mock --full --run-name pro-selftest

    # 3) 真实模型：先 8 条接线，再全量
    python scripts/run_professional_eval.py --limit 8 --run-name pro-wirecheck
    python scripts/run_professional_eval.py --full --resume
    python scripts/run_professional_eval.py --full --resume --retry-failed   # 补跑失败样本

    # 4) 带真实单价出成本（可反复跑，不重跑模型）
    python scripts/run_professional_eval.py --full --resume --cost-in 0.5 --cost-out 2.0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
for p in (PROJECT_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from geval import report as rp  # noqa: E402
from geval.runner import Runner, make_run_spec, probe_models  # noqa: E402
from geval.util import read_json  # noqa: E402
from professional_report import build_report as build_professional_report  # noqa: E402

BASE_CONFIG = os.path.join(PROJECT_ROOT, "config.professional.json")
SOURCE_CONFIG = os.path.join(PROJECT_ROOT, "config.json")
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "intent_routing")


def precheck() -> Dict[str, Any]:
    """数据侧前置条件：文件齐、标签集可用、能报出冻结指纹。"""
    test_path = os.path.join(DATASET_DIR, "test.jsonl")
    labels_path = os.path.join(DATASET_DIR, "labels.json")
    if not os.path.isfile(test_path) or not os.path.isfile(labels_path):
        raise SystemExit(
            "专业数据集未就绪，缺少 test.jsonl / labels.json。\n"
            "  先执行：python scripts/prepare_professional_dataset.py"
        )
    labels = read_json(labels_path, {}) or {}
    if not labels.get("labels"):
        raise SystemExit("labels.json 为空，幻觉意图率会失真；请重跑 prepare_professional_dataset.py")
    n = 0
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    if n == 0:
        raise SystemExit("test.jsonl 为空：%s" % test_path)
    info = read_json(os.path.join(DATASET_DIR, "dataset_info.json"), {}) or {}
    return {"n_records": n, "n_labels": len(labels["labels"]),
            "n_paths": len(labels.get("paths") or []),
            "sha256": info.get("source_sha256") or "未记录",
            "dataset_dir": DATASET_DIR}


def pick_model(source_cfg: Dict[str, Any], wanted: Optional[str]) -> Dict[str, Any]:
    models = [m for m in (source_cfg.get("models") or [])
              if str(m.get("provider") or "").lower() != "mock"]
    if not models:
        raise SystemExit("config.json 里没有非 mock 的真实模型端点；可用 --mock 做离线验证")
    if wanted:
        models = [m for m in models if str(m.get("id")) == wanted]
        if not models:
            raise SystemExit("config.json 中找不到模型 id：%s" % wanted)
    return copy.deepcopy(models[0])


def build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    base = read_json(BASE_CONFIG, {}) or {}
    if not base:
        raise SystemExit("缺少基础配置：%s" % BASE_CONFIG)
    cfg = copy.deepcopy(base)
    for key in list(cfg):
        if key.startswith("_"):  # 去掉说明性占位键，manifest 快照更干净
            cfg.pop(key, None)

    if args.mock:
        cfg["models"] = copy.deepcopy(base.get("models") or [])
    else:
        src = read_json(args.source_config, {}) or {}
        model = pick_model(src, args.model)
        model.setdefault("rate_limit_qps", args.qps)
        cfg["models"] = [model]

    cfg["run_name"] = args.run_name
    if args.concurrency:
        cfg["concurrency"] = args.concurrency
    cfg["cost"] = {"prompt_per_1k": args.cost_in, "completion_per_1k": args.cost_out}
    return cfg


def print_plan(cfg: Dict[str, Any], info: Dict[str, Any], args: argparse.Namespace) -> None:
    m = cfg["models"][0]
    limit = 0 if args.full else args.limit
    n_eval = min(info["n_records"], limit) if limit > 0 else info["n_records"]
    print("── 评测计划 " + "─" * 56)
    print("  模型          %s  (%s)" % (m.get("id"), "mock 离线" if args.mock else "真实端点"))
    print("  端点          %s" % ("mock://local" if args.mock else m.get("base_url")))
    print("  数据集        %d 条 / %d 候选路径 / %d 末级标签" %
          (info["n_records"], info["n_paths"], info["n_labels"]))
    print("  本次样本      %d 条%s" % (n_eval, "（分层抽样接线）" if limit > 0 else "（全量）"))
    print("  预计请求      %d 次（1 请求/样本）" % n_eval)
    print("  并发 / QPS    %s / %s" % (cfg.get("concurrency"), args.qps))
    print("  运行目录      runs/%s" % args.run_name)
    print("  测试集 sha256 %s" % info["sha256"])
    print("  单价          in %s / out %s（每 1K tokens）" % (args.cost_in, args.cost_out))
    print("─" * 66)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="一键跑语音导航意图识别专业能力评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", help="从 config.json 选指定模型 id（默认第一个非 mock）")
    ap.add_argument("--source-config", default=SOURCE_CONFIG, help="真实端点配置，默认 config.json")
    ap.add_argument("--mock", action="store_true", help="用 mock 模型离线验证链路，不联网")
    ap.add_argument("--limit", type=int, default=8, help="抽样条数，默认 8 条接线")
    ap.add_argument("--full", action="store_true", help="全量（覆盖 --limit）")
    ap.add_argument("--run-name", default="geval-professional-intent", help="运行目录名")
    ap.add_argument("--resume", action="store_true", help="断点续跑，跳过已完成样本")
    ap.add_argument("--retry-failed", action="store_true", help="配合 --resume：补跑失败样本")
    ap.add_argument("--no-probe", action="store_true", help="跳过端点探活")
    ap.add_argument("--concurrency", type=int, default=0, help="覆盖并发数")
    ap.add_argument("--qps", type=float, default=2.0, help="单进程限流，默认 2")
    ap.add_argument("--cost-in", type=float, default=0.0, help="每 1K prompt tokens 单价")
    ap.add_argument("--cost-out", type=float, default=0.0, help="每 1K completion tokens 单价")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不发任何请求")
    args = ap.parse_args(argv)

    # mock 默认另起目录，避免覆盖正式评测结果
    if args.mock and args.run_name == "geval-professional-intent":
        args.run_name = "geval-professional-selftest"

    info = precheck()
    cfg = build_cfg(args)
    print_plan(cfg, info, args)
    if args.dry_run:
        print("dry-run：未发送任何请求。去掉 --dry-run 即开始评测。")
        return 0

    if not args.no_probe and not args.mock:
        results = probe_models(cfg, timeout=20.0)
        for r in results:
            if not r.get("ok"):
                print("[error] 端点探活失败：%s — %s" % (r.get("id"), r.get("error")), file=sys.stderr)
                print("        检查 base_url / api_key / 代理；或加 --no-probe 强制继续。", file=sys.stderr)
                return 2
            print("  探活 OK  %s  %s ms" % (r.get("id"), r.get("latency_ms")))

    limit_override = None if args.full else (args.limit if args.limit > 0 else None)
    spec = make_run_spec(
        cfg,
        run_name=args.run_name,
        resume=bool(args.resume),
        retry_failed=bool(args.retry_failed),
        limit_override=limit_override,
        allow_synthetic=False,
        quiet=False,
        parallel_tasks=1,
    )
    result = Runner(spec).run()
    rp.build_report(spec.run_dir)
    out = build_professional_report(spec.run_dir,
                                    price_in=args.cost_in, price_out=args.cost_out)

    m = out["metrics"]
    u = m["usage"]
    print("\n── 评测完成 " + "─" * 56)
    print("  样本 %d（有效 %d，失败 %d）" % (m["n_samples"], m["n_valid"], m["n_errors"]))
    print("  意图准确率 %.2f%% ｜ 小模型基线 %.2f%% ｜ 增益 %+.2f%% ｜ McNemar p=%s" % (
        100 * m["intent_accuracy"], 100 * m["small_model_accuracy"],
        100 * m["paired_uplift"],
        "n/a" if m["mcnemar_p"] is None else "%.4f" % m["mcnemar_p"]))
    print("  total tokens %d ｜ 平均 %.1f/样本 ｜ p95 %s ms" % (
        u["total_tokens"], u["avg_total_tokens"] or 0,
        "n/a" if (u["latency_ms"] or {}).get("p95") is None else "%.0f" % u["latency_ms"]["p95"]))
    print("  专业报告  %s" % out["md"])
    print("  通用报告  %s" % os.path.join(spec.run_dir, "report.html"))
    print("  逐样本    %s" % os.path.join(spec.run_dir, cfg["models"][0]["id"],
                                          "intent-routing.jsonl"))
    if result.get("errors"):
        print("  [warn] %d 条错误；用 --resume --retry-failed 补跑" % len(result["errors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
