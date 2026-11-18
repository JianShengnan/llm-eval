"""把多次（通常是并行分批的）运行合并成一个 run 目录，重建 summary.json 并出报告。

为什么需要它
------------
`Runner` 在执行结束时会把本次的全部结果整体写入 `summary.json`。因此**不能让两个进程
共用同一个 run 目录**：它们的明细文件（`<model>/<task>.jsonl`）互不冲突，
但退出时后一个进程会用自己那份结果整体覆盖 summary.json，先跑完的那批就"消失"了。

正确的分批方式是：每个进程写各自的 run 目录，最后用本模块合并：

    python -m geval run --config config.json --run-name p-ceval --tasks ceval   &
    python -m geval run --config config.json --run-name p-cmmlu --tasks cmmlu   &
    wait
    python -m geval merge-runs runs/p-ceval runs/p-cmmlu --run-name merged

合并规则（确定性优先，避免"谁后写谁赢"这种不可复现的行为）：

1. 基本单元是「模型 × 任务」，来自各源 `summary.json` 的 `results` 字段；
2. 同一「模型 × 任务」出现多次时，取**样本数 n 更大**的一份，n 相同则取靠前的源，
   并在返回值里列出冲突清单，供人工确认；
3. 合并结果写到**新的** run 目录，源目录只读不删，可以随时重来；
4. 已存在同名目标文件时不覆盖（幂等：重复执行 merge-runs 到同一目录不会破坏结果）。
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .report import build_report, load_run
from .util import now_iso, read_json, write_json

Key = Tuple[str, str]

# client_stats 里可以跨批次累加的计数器（其它如 p95 分位数只能取其一，不能相加）
_ADDITIVE_STATS = ("calls", "errors", "cache_hits", "http_requests", "http_retries")


# --------------------------------------------------------------------------- #
def scan_task_summaries(run_dir: str) -> Dict[Key, Dict[str, Any]]:
    """扫描 `run_dir/<model>/<task>.summary.json`，不依赖 summary.json 也能还原总表。

    用处有二：merge 的降级路径（源目录的 summary.json 缺失/损坏），
    以及自检里对合并结果做独立校验。
    """
    out: Dict[Key, Dict[str, Any]] = {}
    if not os.path.isdir(run_dir):
        return out
    for model in sorted(os.listdir(run_dir)):
        mdir = os.path.join(run_dir, model)
        if not os.path.isdir(mdir):
            continue
        for fname in sorted(os.listdir(mdir)):
            if not fname.endswith(".summary.json"):
                continue
            agg = read_json(os.path.join(mdir, fname), default=None)
            if not isinstance(agg, dict):
                continue
            mid = str(agg.get("model") or model)
            task_key = str(agg.get("task") or fname[: -len(".summary.json")])
            out[(mid, task_key)] = agg
    return out


def summary_from_task_files(run_dir: str) -> Dict[str, Any]:
    """不依赖 summary.json，仅用 `<model>/<task>.summary.json` 重建一份总表 payload。

    summary.json 在某个批次里丢了不会导致那批结果白跑，这里给出恢复路径。
    """
    results: Dict[str, Dict[str, Any]] = {}
    task_summaries: Dict[str, Dict[str, Any]] = {}
    model_labels: Dict[str, str] = {}
    datasets: Dict[str, Any] = {}
    for (mid, task_key), agg in scan_task_summaries(run_dir).items():
        results.setdefault(mid, {})[task_key] = agg
        task_summaries.setdefault(task_key, {})[mid] = agg
        model_labels.setdefault(mid, str(agg.get("model_label") or mid))
        if agg.get("dataset_source"):
            datasets[task_key] = {"source": agg.get("dataset_source")}
    return {
        "run_name": os.path.basename(os.path.abspath(run_dir)),
        "generated_at": now_iso(),
        "model_labels": model_labels,
        "results": results,
        "task_summaries": task_summaries,
        "manifests": {"datasets": datasets},
        "errors": [],
    }


def load_prepared(run_dir: str) -> Optional[Dict[str, Any]]:
    """读取一个 run 目录的 summary.json；缺失或为空时用任务级 summary 文件重建。"""
    data: Optional[Dict[str, Any]] = None
    try:
        data = load_run(run_dir)
    except (FileNotFoundError, ValueError, OSError):
        data = None
    if data is not None and (data.get("results") or {}):
        data["_rebuilt"] = False
        return data

    rebuilt = summary_from_task_files(run_dir)
    if not rebuilt["results"]:
        return None
    rebuilt["_rebuilt"] = True
    rebuilt["_source_dir"] = os.path.abspath(run_dir)
    return rebuilt


# --------------------------------------------------------------------------- #
def _n_of(agg: Dict[str, Any]) -> int:
    try:
        return int(agg.get("n") or 0)
    except (TypeError, ValueError):
        return 0


def _pick(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """同一「模型 × 任务」的取舍：样本数多者优先；出错的让位给没出错的。"""
    err_a, err_b = bool(a["agg"].get("error")), bool(b["agg"].get("error"))
    if err_a != err_b:
        return b if err_a else a
    return b if _n_of(b["agg"]) > _n_of(a["agg"]) else a


def _safe_join(root: str, *parts: str) -> Optional[str]:
    """拼接路径并确认没有越出 root（任务名来自配置，防手滑写成 ../../）。"""
    path = os.path.normpath(os.path.join(root, *parts))
    root_abs = os.path.abspath(root)
    if os.path.abspath(path) != root_abs and not os.path.abspath(path).startswith(root_abs + os.sep):
        return None
    return path


def _merge_client_stats(dst: Dict[str, Dict[str, Any]], src: Dict[str, Dict[str, Any]]) -> None:
    for mid, s in (src or {}).items():
        if mid not in dst:
            dst[mid] = dict(s)
            continue
        d = dst[mid]
        for k, v in s.items():
            if k in _ADDITIVE_STATS and isinstance(v, (int, float)) and isinstance(d.get(k), (int, float)):
                d[k] = d[k] + v
            elif d.get(k) is None:
                d[k] = v
        usage_s, usage_d = s.get("usage") or {}, d.get("usage") or {}
        if isinstance(usage_s, dict) and isinstance(usage_d, dict):
            merged = dict(usage_d)
            for k, v in usage_s.items():
                if isinstance(v, (int, float)) and isinstance(merged.get(k), (int, float)):
                    merged[k] = merged[k] + v
            d["usage"] = merged


# --------------------------------------------------------------------------- #
def merge_runs(
    sources: Sequence[str],
    dst: str,
    *,
    baseline: Optional[str] = None,
    pairs: Optional[List[List[str]]] = None,
    make_report: bool = True,
    quiet: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """合并多个 run 目录到 dst，并重建 summary.json / manifest.json。

    weights 为综合分权重覆盖项，只在出报告时生效，不影响任何逐样本结果。
    合并目录会补写一份 config.json（继承各源的 tasks 与 dimension_weights），
    这样后续单独跑 `report` 时权重仍然可读。

    返回 {"run_dir", "sources", "tasks", "models", "conflicts", "warnings", "report"}。
    """
    if not sources:
        raise ValueError("至少需要一个源目录")
    dst_abs = os.path.abspath(dst)
    os.makedirs(dst_abs, exist_ok=True)

    entries: Dict[Key, Dict[str, Any]] = {}
    conflicts: List[str] = []
    warnings: List[str] = []
    skipped: List[str] = []
    model_labels: Dict[str, str] = {}
    client_stats: Dict[str, Dict[str, Any]] = {}
    datasets: Dict[str, Any] = {}
    errors: List[str] = []
    seen_errors = set()
    base_manifest: Dict[str, Any] = {}
    source_notes: List[Dict[str, Any]] = []

    for src in sources:
        src_abs = os.path.abspath(src)
        payload = load_prepared(src_abs)
        if payload is None:
            skipped.append(src_abs)
            warnings.append(f"跳过 {src_abs}：既没有可用的 summary.json，也没有任何任务级 summary 文件")
            continue
        if payload.get("_rebuilt"):
            warnings.append(f"{os.path.basename(src_abs)}：summary.json 缺失，已由任务级 summary 文件重建")
        if not base_manifest:
            base_manifest = dict(payload.get("manifest") or {})
        source_notes.append(
            {
                "dir": src_abs,
                "run_name": payload.get("run_name"),
                "rebuilt": bool(payload.get("_rebuilt")),
                "models": sorted((payload.get("results") or {}).keys()),
            }
        )

        for mid, label in (payload.get("model_labels") or {}).items():
            prev = model_labels.setdefault(str(mid), str(label or mid))
            if prev != str(label or mid):
                warnings.append(f"模型 {mid} 的 label 在不同批次中不一致（{prev} vs {label}），已取前者")

        for mid, per_task in (payload.get("results") or {}).items():
            for task_key, agg in (per_task or {}).items():
                key = (str(mid), str(task_key))
                cand = {"src": src_abs, "agg": agg}
                prev = entries.get(key)
                if prev is None:
                    entries[key] = cand
                    continue
                winner = _pick(prev, cand)
                if winner is not prev:
                    conflicts.append(
                        f"{mid}/{task_key}：{os.path.basename(prev['src'])}(n={_n_of(prev['agg'])}) → "
                        f"{os.path.basename(winner['src'])}(n={_n_of(winner['agg'])})"
                    )
                entries[key] = winner

        manifests = payload.get("manifests") or {}
        _merge_client_stats(client_stats, manifests.get("client_stats") or {})
        for t, meta in (manifests.get("datasets") or {}).items():
            datasets.setdefault(str(t), meta)
        for err in payload.get("errors") or []:
            if err not in seen_errors:
                seen_errors.add(err)
                errors.append(err)

    if not entries:
        raise ValueError("没有任何可合并的结果")

    # ---- 拷贝明细与任务级聚合文件到合并目录（已存在则不覆盖，保证幂等） ----
    copied = 0
    for (mid, task_key), entry in sorted(entries.items()):
        src = entry["src"]
        target_dir = _safe_join(dst_abs, mid)
        if target_dir is None:
            warnings.append(f"跳过 {mid}/{task_key}：模型目录名不合法")
            continue
        os.makedirs(target_dir, exist_ok=True)
        for name in (f"{task_key}.jsonl", f"{task_key}.summary.json"):
            src_path = _safe_join(src, mid, name)
            dst_path = _safe_join(dst_abs, mid, name)
            if src_path is None or dst_path is None or not os.path.exists(src_path):
                continue
            if os.path.exists(dst_path) and os.path.getsize(dst_path) >= os.path.getsize(src_path):
                continue
            shutil.copy2(src_path, dst_path)
            copied += 1

    # 各批次生效配置的快照留作凭证（分批跑时各批的 concurrency / limit 可能不同，必须留痕）
    provenance: Dict[str, str] = {}
    for src in sources:
        src_cfg = os.path.join(os.path.abspath(src), "config.json")
        if not os.path.exists(src_cfg):
            continue
        base = os.path.basename(os.path.abspath(src))
        dst_cfg = os.path.join(dst_abs, "merged_sources", f"{base}.config.json")
        os.makedirs(os.path.dirname(dst_cfg), exist_ok=True)
        shutil.copy2(src_cfg, dst_cfg)
        provenance[os.path.abspath(src)] = os.path.relpath(dst_cfg, dst_abs)

    # ---- 重建 summary.json ----
    results: Dict[str, Dict[str, Any]] = {}
    task_summaries: Dict[str, Dict[str, Any]] = {}
    for (mid, task_key), entry in sorted(entries.items()):
        agg = dict(entry["agg"])
        agg.setdefault("task", task_key)
        agg.setdefault("model", mid)
        agg.setdefault("model_label", model_labels.get(mid, mid))
        results.setdefault(mid, {})[task_key] = agg
        task_summaries.setdefault(task_key, {})[mid] = agg

    payload: Dict[str, Any] = {
        "run_name": os.path.basename(dst_abs),
        "generated_at": now_iso(),
        "model_labels": model_labels,
        "results": results,
        "task_summaries": task_summaries,
        "manifests": {"client_stats": client_stats, "datasets": datasets},
        "errors": errors,
        "merged": {
            "sources": source_notes,
            "conflicts": conflicts,
            "copied_files": copied,
            "configs": provenance,
            "merged_at": now_iso(),
        },
    }
    write_json(os.path.join(dst_abs, "summary.json"), payload)

    merged_manifest = dict(base_manifest)
    merged_manifest.update(
        {
            "run_name": os.path.basename(dst_abs),
            "merged_from": [s["dir"] for s in source_notes],
            "merged_at": payload["merged"]["merged_at"],
            "note": "本目录由 geval merge-runs 从多个分批运行合并而来",
        }
    )
    write_json(os.path.join(dst_abs, "manifest.json"), merged_manifest)

    # ---- 补写 config.json ----
    # 合并目录原本没有 config.json（各源的只存到 merged_sources/），而报告要从
    # run_dir/config.json 读任务条目上的 dimension / weight。这里把各源配置并一份，
    # 任务按 tag/name 去重，保证合并后仍能按维度算综合分。
    merged_cfg: Dict[str, Any] = {}
    merged_tasks: List[Dict[str, Any]] = []
    seen_keys = set()
    for s in source_notes:
        src_cfg = read_json(os.path.join(s["dir"], "config.json"), default={}) or {}
        if src_cfg and not merged_cfg:
            merged_cfg = dict(src_cfg)
        if "dimension_weights" in src_cfg and "dimension_weights" not in merged_cfg:
            merged_cfg["dimension_weights"] = src_cfg["dimension_weights"]
        for ts in src_cfg.get("tasks") or []:
            if not isinstance(ts, dict):
                continue
            key = ts.get("tag") or ts.get("name")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            merged_tasks.append(ts)
    if merged_cfg or merged_tasks:
        merged_cfg["tasks"] = merged_tasks
        merged_cfg["_merged_note"] = "由 geval merge-runs 合并各源 config.json 生成，供报告读取综合分权重"
        write_json(os.path.join(dst_abs, "config.json"), merged_cfg)

    out: Dict[str, Any] = {
        "run_dir": dst_abs,
        "sources": [s["dir"] for s in source_notes],
        "models": sorted(results.keys()),
        "tasks": sorted(task_summaries.keys()),
        "n_entries": len(entries),
        "conflicts": conflicts,
        "warnings": warnings,
        "skipped": skipped,
        "copied_files": copied,
        "report": None,
    }

    if make_report:
        if not baseline:
            base_cfg = read_json(os.path.join(source_notes[0]["dir"], "config.json"), default={}) or {}
            baseline = ((base_cfg.get("report") or {}) or {}).get("baseline")
        paths = build_report(dst_abs, baseline=baseline, pairs=pairs, weights=weights)
        out["report"] = paths

    if not quiet:
        print(f"合并完成：{dst_abs}")
        print(f"  源目录   {len(source_notes)} 个，跳过 {len(skipped)} 个")
        print(f"  模型     {', '.join(out['models'])}")
        print(f"  任务     {', '.join(out['tasks'])}")
        print(f"  明细文件 复制 {copied} 个")
        if conflicts:
            print(f"  ⚠ 同一「模型 × 任务」重复出现 {len(conflicts)} 处，已取样本数更大的一份：")
            for c in conflicts[:10]:
                print(f"    · {c}")
        for w in warnings[:10]:
            print(f"  ! {w}")
    return out
