"""评测编排：模型 × 任务的并发执行、断点续跑、结果落盘与复现指纹。

落盘结构：
  runs/<run_name>/
    manifest.json              本次运行的完整环境与配置指纹
    config.json                生效配置快照
    <model>/<task>.jsonl       逐样本明细（可中断续跑、可逐条复核）
    <model>/<task>.summary.json 任务级聚合
    summary.json               模型 × 任务总表
    report.md / report.html    报告
"""

from __future__ import annotations

import concurrent.futures as futures
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from . import tasks as tk
from .client import build_client
from .metrics import aggregate
from .util import JsonlWriter, Progress, ensure_dir, jdump, now_iso, read_jsonl, write_json


# --------------------------------------------------------------------------- #
@dataclass
class RunSpec:
    cfg: Dict[str, Any]
    run_dir: str
    run_name: str
    concurrency: int = 8
    resume: bool = False
    # 续跑时是否重跑上一轮失败（error）的样本。限流打崩一批之后，
    # 只补跑失败样本比整任务重来省一个数量级的请求量。
    retry_failed: bool = False
    limit_override: Optional[int] = None
    task_filter: Optional[List[str]] = None
    model_filter: Optional[List[str]] = None
    allow_synthetic: bool = False
    quiet: bool = False
    cache_dir: Optional[str] = None
    # 同一模型内的任务并行度。>1 时多个数据集同时跑，实际总并发 = parallel_tasks × concurrency。
    parallel_tasks: int = 1
    # 进度最小刷新间隔（秒）。0 = 自动（终端 2s / 逐行输出 10s）。
    progress_interval: float = 0.0


def make_run_spec(cfg: Dict[str, Any], **kw: Any) -> RunSpec:
    out_dir = str(cfg.get("output_dir") or "runs")
    run_name = str(kw.pop("run_name", None) or cfg.get("run_name") or time.strftime("geval-%Y%m%d-%H%M%S"))
    run_dir = ensure_dir(os.path.join(out_dir, run_name))
    return RunSpec(
        cfg=cfg,
        run_dir=run_dir,
        run_name=run_name,
        concurrency=int(cfg.get("concurrency") or 8),
        cache_dir=cfg.get("cache_dir"),
        **kw,
    )


# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.cfg = spec.cfg
        self.summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.model_labels: Dict[str, str] = {}
        self.task_summaries: Dict[str, Dict[str, Any]] = {}
        self.manifests: Dict[str, Dict[str, Any]] = {}
        self.errors: List[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _models(self) -> List[Dict[str, Any]]:
        models = list(self.cfg.get("models") or [])
        if not models:
            raise ValueError("配置中 models 为空；请参考 config.example.json")
        if self.spec.model_filter:
            wanted = set(self.spec.model_filter)
            models = [m for m in models if str(m.get("id")) in wanted]
            missing = wanted - {str(m.get("id")) for m in models}
            if missing:
                raise ValueError(f"配置中不存在这些模型 id：{sorted(missing)}")
        return models

    def _tasks(self) -> List[Dict[str, Any]]:
        specs = list(self.cfg.get("tasks") or [])
        if not specs:
            raise ValueError("配置中 tasks 为空；至少指定一个任务")
        if self.spec.task_filter:
            wanted = set(self.spec.task_filter)
            specs = [t for t in specs if str(t.get("name")) in wanted]
            if not specs:
                return [{"name": n} for n in self.spec.task_filter]
        for t in specs:
            if self.spec.limit_override:
                t["limit"] = int(self.spec.limit_override)
        return specs

    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, Any]:
        data_root = str(self.cfg.get("data_root") or "datasets")
        models = self._models()
        task_specs = list(self._tasks())

        self._write_manifest(models, task_specs, data_root)

        parallel = max(1, int(self.spec.parallel_tasks or 1))
        for mcfg in models:
            mid = str(mcfg["id"])
            self.model_labels[mid] = str(mcfg.get("label") or mid)
            client = build_client(
                mcfg,
                cache_dir=_sub(self.spec.cache_dir, mid) if self.spec.cache_dir else None,
                timeout=float(self.cfg.get("request_timeout_s") or 180),
                retries=int(self.cfg.get("retries") or 3),
                proxy=str(self.cfg.get("proxy") or "bypass"),
            )
            self.summary.setdefault(mid, {})
            if parallel > 1 and len(task_specs) > 1:
                aggs = self._run_tasks_parallel(client, mcfg, task_specs, data_root, parallel)
            else:
                aggs = [self._execute_task(client, mcfg, ts, data_root) for ts in task_specs]
            # 保持与配置一致的顺序写回汇总，报告列顺序才稳定
            for tspec, agg in zip(task_specs, aggs):
                self.summary[mid][str(tspec.get("tag") or tspec.get("name"))] = agg

            stats = client.stats()
            with self._lock:
                self.manifests.setdefault("client_stats", {})[mid] = stats

        self._write_summary()
        return {
            "run_dir": self.spec.run_dir,
            "run_name": self.spec.run_name,
            "summary": self.summary,
            "errors": self.errors,
        }

    # ------------------------------------------------------------------ #
    def _execute_task(
        self,
        client: Any,
        mcfg: Dict[str, Any],
        tspec: Dict[str, Any],
        data_root: str,
    ) -> Dict[str, Any]:
        """跑单个「模型 × 任务」，把异常收敛成一行错误记录，避免拖垮整次运行。"""
        task_name = str(tspec.get("name"))
        task_key = str(tspec.get("tag") or task_name)
        try:
            return self._run_task(client, mcfg, task_name, task_key, tspec, data_root)
        except Exception as exc:  # noqa: BLE001
            msg = f"{mcfg['id']}/{task_key} 执行失败：{type(exc).__name__}: {exc}"
            with self._lock:
                self.errors.append(msg)
            print(f"[error] {msg}", file=sys.stderr)
            return {
                "n": 0,
                "score": None,
                "error": str(exc),
                "task": task_key,
                "task_name": task_name,
                "model": str(mcfg["id"]),
                "model_label": self.model_labels.get(str(mcfg["id"]), str(mcfg["id"])),
            }

    def _run_tasks_parallel(
        self,
        client: Any,
        mcfg: Dict[str, Any],
        task_specs: List[Dict[str, Any]],
        data_root: str,
        parallel: int,
    ) -> List[Dict[str, Any]]:
        """同一模型内的多个数据集并行执行。

        意义不在"省请求"，而在两点：
        1. 尾延迟隔离：慢的数据集不再阻塞快的；
        2. 可以在不修改 per-task concurrency 的前提下放大总并发。
        返回结果按输入顺序排列，保证 report 的列顺序稳定。
        """
        workers = max(1, min(int(parallel), len(task_specs)))
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(self._execute_task, client, mcfg, ts, data_root) for ts in task_specs]
            return [f.result() for f in futs]

    # ------------------------------------------------------------------ #
    def _run_task(
        self,
        client: Any,
        mcfg: Dict[str, Any],
        task_name: str,
        task_key: str,
        tspec: Dict[str, Any],
        data_root: str,
    ) -> Dict[str, Any]:
        tspec = dict(tspec)
        tspec["_quiet"] = self.spec.quiet
        # 生成长度预算的优先级：任务级显式配置 > 模型级 params.max_tokens > 1024。
        # 任务级没有单独配 max_tokens 时必须落到模型级，否则 tasks.BaseTask 的默认值 1024
        # 会静默覆盖模型配置（如 2048）——thinking 模型或长 CoT 任务会因此整片截断。
        if not tspec.get("max_tokens"):
            model_max = int(((mcfg.get("params") or {}).get("max_tokens")) or 0)
            if model_max > 0:
                tspec["max_tokens"] = model_max
        task = tk.get_task(task_name, tspec)
        records, examples = task.prepare(
            data_root,
            allow_synthetic=self.spec.allow_synthetic,
            synthetic_n=int(self.cfg.get("synthetic_n") or 60),
        )
        if not records:
            raise ValueError("评测集为空")

        items = [task.build(r, examples) for r in records]
        out_path = os.path.join(self.spec.run_dir, str(mcfg["id"]), f"{task_key}.jsonl")
        done_rows = _done_rows(out_path, retry_failed=self.spec.retry_failed) if self.spec.resume else []
        done_ids = {str(r.get("id")) for r in done_rows if r.get("id") is not None}
        todo = [it for it in items if str(it.record.get("id")) not in done_ids]

        if not self.spec.quiet:
            note = f"（续跑：跳过已完成 {len(items) - len(todo)} 条）" if done_ids else ""
            print(
                f"[{self.model_labels[str(mcfg['id'])]} × {task_key}] "
                f"样本 {len(records)} 条，协议 {task.protocol}，模式 {task.mode}{note}",
                file=sys.stderr,
            )

        # mock 客户端需要旁路注入标准答案
        if hasattr(client, "register"):
            for it in items:
                client.register(it.prompt_key, it.mock_entry)

        scored: List[Dict[str, Any]] = []
        if done_rows:
            scored.extend(done_rows)

        if todo:
            writer = JsonlWriter(out_path, append=bool(done_ids))
            # 多任务并行时每个任务各自刷 \r 会互相覆盖，改为逐行输出，方便按数据集看进度
            parallel = max(1, int(self.spec.parallel_tasks or 1))
            every = max(1, len(todo) // (10 if parallel > 1 and len(todo) > 200 else 20))
            every = min(every, 100)  # 大数据集别让步长大到「前几百条一行不出」
            progress = (
                None
                if self.spec.quiet
                else Progress(
                    len(todo),
                    label=f"{mcfg['id']}/{task_key}",
                    every=every,
                    line_mode=parallel > 1,
                    min_interval=_progress_interval(self.spec.progress_interval, parallel),
                )
            )
            if progress:
                progress.kickoff()
            try:
                with futures.ThreadPoolExecutor(max_workers=max(1, self.spec.concurrency)) as pool:
                    future_map = {pool.submit(task.run_one, it, client): it for it in todo}
                    for fut in futures.as_completed(future_map):
                        item = future_map[fut]
                        try:
                            row = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            row = task._row(item, client)
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            with self._lock:
                                self.errors.append(f"{mcfg['id']}/{task_name}/{item.record.get('id')}: {row['error']}")
                        with self._lock:
                            scored.append(row)
                        writer.write(row)
                        if progress:
                            progress.tick()
            finally:
                writer.close()
                if progress:
                    progress.finish()

        agg = aggregate(
            scored,
            bootstrap=int((self.cfg.get("report") or {}).get("bootstrap") or 2000),
            alpha=float((self.cfg.get("report") or {}).get("alpha") or 0.05),
            seed=int(self.cfg.get("seed") or 42),
        )
        agg["task"] = task_key
        agg["task_name"] = task_name
        agg["model"] = str(mcfg["id"])
        agg["model_label"] = self.model_labels[str(mcfg["id"])]
        agg["kind"] = task.kind
        agg["protocol"] = task.protocol
        agg["mode"] = task.mode
        # 记下来供报告做「可比性守卫」：两个模型的这几项不一致时，配对差值不可信
        agg["num_fewshot"] = len(examples)
        agg["dataset_source"] = task.manifest.get("source")
        extra = task.summarize(scored)
        if extra:
            agg["task_metrics"] = extra
        agg["breakdown"] = _breakdown(scored, task_key)
        agg["examples_failed"] = _examples(scored, want_correct=False, n=5)
        agg["examples_passed"] = _examples(scored, want_correct=True, n=3)
        agg["parse_note"] = _parse_note(scored)

        write_json(os.path.join(self.spec.run_dir, str(mcfg["id"]), f"{task_key}.summary.json"), agg)

        with self._lock:
            self.task_summaries.setdefault(task_key, {})[str(mcfg["id"])] = agg
            self.manifests.setdefault("datasets", {})[task_key] = task.manifest
        return agg

    # ------------------------------------------------------------------ #
    def _write_manifest(self, models, task_specs, data_root) -> None:
        manifest = {
            "geval_version": __version__,
            "run_name": self.spec.run_name,
            "created_at": now_iso(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "data_root": os.path.abspath(data_root),
            "concurrency": self.spec.concurrency,
            "parallel_tasks": max(1, int(self.spec.parallel_tasks or 1)),
            "seed": self.cfg.get("seed"),
            "proxy": self.cfg.get("proxy") or "bypass",
            "limit_override": self.spec.limit_override,
            "models": [{k: v for k, v in m.items() if k != "api_key"} for m in models],
            "tasks": task_specs,
        }
        write_json(os.path.join(self.spec.run_dir, "manifest.json"), manifest)
        write_json(os.path.join(self.spec.run_dir, "config.json"), self.cfg)

    def _write_summary(self) -> None:
        payload = {
            "run_name": self.spec.run_name,
            "generated_at": now_iso(),
            "model_labels": self.model_labels,
            "results": self.summary,
            "task_summaries": self.task_summaries,
            "manifests": self.manifests,
            "errors": self.errors,
        }
        write_json(os.path.join(self.spec.run_dir, "summary.json"), payload)


# --------------------------------------------------------------------------- #
def _done_rows(path: str, *, retry_failed: bool = False) -> List[Dict[str, Any]]:
    """续跑时读取已完成的行。

    retry_failed=True 会把上一轮带 error 的样本从磁盘剔除，让它们重新进入 todo。
    这一步很关键：限流打崩的那一批如果留在文件里，--resume 会认为它们「已完成」，
    于是失败率被永久固化进结果；剔除后再跑才是真正的补跑。
    """
    if not os.path.exists(path):
        return []
    try:
        rows = list(read_jsonl(path))
    except (OSError, ValueError):
        return []
    if not retry_failed:
        return rows

    kept = [r for r in rows if not r.get("error")]
    dropped = len(rows) - len(kept)
    if dropped:
        tmp = path + f".prune{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for r in kept:
                    f.write(jdump(r) + "\n")
            os.replace(tmp, path)
        except OSError:
            pass
        print(f"  [info] 剔除 {dropped} 条失败样本，本轮重跑：{path}", file=sys.stderr)
    return kept


def _sub(root: Optional[str], name: str) -> str:
    return os.path.join(str(root), name) if root else ""


def _progress_interval(explicit: float, parallel: int) -> float:
    """进度的最小刷新间隔（秒）。

    串行且输出是终端时用 `\r` 原地刷新，2 秒一次最跟手也不会刷屏；
    逐行输出的场景（任务并行 / 输出被重定向到日志）会真的占行数，放宽到 10 秒。
    """
    try:
        explicit = float(explicit or 0.0)
    except (TypeError, ValueError):
        explicit = 0.0
    if explicit > 0:
        return explicit
    is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    return 2.0 if (is_tty and parallel <= 1) else 10.0


def _breakdown(scored: Sequence[Dict[str, Any]], task_name: str) -> List[Dict[str, Any]]:
    """按 subject 分组统计；分组过多时只保留样本量前 20 组，避免报告爆表。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in scored:
        g = str(row.get("subject") or "unknown")
        buckets.setdefault(g, []).append(row)
    out = []
    for g, rows in sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:20]:
        ok = sum(1 for r in rows if r.get("correct"))
        pf = sum(1 for r in rows if r.get("parse_ok") is False)
        out.append({"subject": g, "n": len(rows), "score": ok / len(rows), "parse_fail": pf})
    return out


def _examples(scored: Sequence[Dict[str, Any]], *, want_correct: bool, n: int) -> List[Dict[str, Any]]:
    """抽取错例/正例，供人工快速判断是模型问题还是抽取问题。"""
    picks = []
    for row in scored:
        if bool(row.get("correct")) != want_correct:
            continue
        if row.get("error"):
            continue
        picks.append(
            {
                "id": row.get("id"),
                "subject": row.get("subject"),
                "gold": row.get("gold"),
                "pred": row.get("pred"),
                "strategy": row.get("strategy"),
                "parse_ok": row.get("parse_ok"),
                "response": (row.get("response") or "")[:600],
            }
        )
        if len(picks) >= n:
            break
    return picks


def _parse_note(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(scored)
    if not n:
        return {}
    pf = [r for r in scored if r.get("parse_ok") is False and not r.get("error")]
    return {
        "n": n,
        "parse_fail": len(pf),
        "parse_fail_rate": len(pf) / n,
        "strategies": _strategy_histogram(scored),
        "hint": (
            "解析失败率偏高时，优先检查 prompt 协议与抽取规则，而不是模型能力"
            if len(pf) / n > 0.02
            else "解析失败率正常"
        ),
    }


def _strategy_histogram(scored: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for row in scored:
        s = str(row.get("strategy") or "none")
        hist[s] = hist.get(s, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
def probe_models(cfg: Dict[str, Any], *, timeout: float = 20.0) -> List[Dict[str, Any]]:
    """快速探活：对每个模型端点发一条最小请求，确认服务、模型名与鉴权是否可用。"""
    results: List[Dict[str, Any]] = []
    for mcfg in cfg.get("models") or []:
        mid = str(mcfg.get("id"))
        rec: Dict[str, Any] = {"id": mid, "label": mcfg.get("label") or mid, "base_url": mcfg.get("base_url")}
        if str(mcfg.get("provider") or "").lower() == "mock":
            rec.update({"ok": True, "note": "mock 模型无需网络"})
            results.append(rec)
            continue
        try:
            client = build_client(
                mcfg,
                timeout=timeout,
                retries=0,
                proxy=str(cfg.get("proxy") or "bypass"),
            )
            reply = client.chat([{"role": "user", "content": "回复一个字：好"}], max_tokens=8)
            rec.update(
                {
                    "ok": reply.ok,
                    "latency_ms": round(reply.latency_ms, 1),
                    "sample": (reply.text or "").strip()[:40],
                    "service_model": (reply.raw or {}).get("model"),
                    "error": reply.error,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        results.append(rec)
    return results
