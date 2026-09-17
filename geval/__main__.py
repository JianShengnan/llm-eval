"""geval 命令行入口。

    python -m geval selftest                      离线端到端自检（推荐第一步）
    python -m geval list-tasks                    查看内置任务与默认设置
    python -m geval preview --task ceval          打印真实 prompt 样例，先看再跑
    python -m geval probe   --config config.json  模型端点探活
    python -m geval run     --config config.json [--smoke] [--resume]
    python -m geval report  --run runs/<name> [--baseline qwen3-32b]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

from . import __version__
from . import datasets as ds
from . import report as rp
from . import tasks as tk
from .runner import Runner, make_run_spec, probe_models
from .util import ensure_dir, jdump, read_json, write_json


# --------------------------------------------------------------------------- #
def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        raise SystemExit("缺少 --config；可先复制 config.example.json")
    if not os.path.exists(path):
        raise SystemExit(f"配置文件不存在：{path}")
    cfg = read_json(path)
    if not isinstance(cfg, dict):
        raise SystemExit("配置文件顶层必须是 JSON 对象")
    return cfg


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "concurrency", None):
        cfg["concurrency"] = int(args.concurrency)
    if getattr(args, "data_root", None):
        cfg["data_root"] = args.data_root
    if getattr(args, "out", None):
        cfg["output_dir"] = args.out
    if getattr(args, "run_name", None):
        cfg["run_name"] = args.run_name
    if getattr(args, "proxy", None):
        cfg["proxy"] = args.proxy
    return cfg


def _split(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


# --------------------------------------------------------------------------- #
def cmd_list_tasks(args: argparse.Namespace) -> int:
    kinds = tk.task_kinds()
    print(f"geval {__version__} — 内置任务\n")
    print(f"{'任务':<14}{'类型':<10}{'默认 few-shot':<14}{'默认切分':<10}覆盖能力")
    print("-" * 78)
    coverage = {
        "ceval": "中文语言理解 / 学科知识",
        "cmmlu": "中文语言理解 / 通识",
        "mmlu": "英文通识知识（跨语言稳健性）",
        "mmlu_pro": "高难通识知识（区分度高，适合拉开差距）",
        "arc": "常识推理（英文）",
        "agieval": "综合学术与逻辑（中文）",
        "gsm8k": "小学数学应用题推理",
        "math": "竞赛数学推理",
        "bbh": "复杂逻辑推理（BIG-Bench Hard）",
        "ifeval": "指令遵循（规则化判定，无需裁判模型）",
        "json_schema": "结构化输出：意图 + 槽位（与业务最贴近）",
        "intent_json": "同 json_schema，适合放你自己的业务意图数据集",
    }
    for name in sorted(tk.TASK_REGISTRY):
        cls = tk.TASK_REGISTRY[name]
        print(
            f"{name:<14}{kinds[name]:<10}{tk.DEFAULT_FEWSHOT.get(name, 0):<14}"
            f"{getattr(cls, 'default_split', 'test'):<10}{coverage.get(name, '')}"
        )
    print("\n提示：同一任务可用 tag 起别名，以对比不同 prompt 协议，例如")
    print('  {"name": "ceval", "tag": "ceval-cot", "protocol": "answer_line"}')
    print('  {"name": "ceval", "tag": "ceval-direct", "protocol": "direct"}')
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config) if args.config else {}
    data_root = args.data_root or str(cfg.get("data_root") or "datasets")
    task_name = args.task
    cfg_task = {}
    for spec in cfg.get("tasks") or []:
        if str(spec.get("name")) == task_name:
            cfg_task = spec
            break
    cfg_task.setdefault("limit", 0)
    task = tk.get_task(task_name, cfg_task)
    records, examples = task.prepare(data_root, allow_synthetic=True, synthetic_n=8)
    if not records:
        raise SystemExit("没有可用样本")
    idx = max(0, min(int(args.index), len(records) - 1))
    item = task.build(records[idx], examples)
    print(f"# 任务 {task_name} ｜ 协议 {task.protocol} ｜ 模式 {task.mode} ｜ few-shot {len(examples)} 条")
    print(f"# 标准答案：{records[idx].get('answer')}")
    print("-" * 78)
    for msg in item.messages:
        print(f"[{msg['role']}]")
        print(msg["content"])
        print("-" * 78)
    print("提示：上面就是实际发给模型的 payload。发现格式问题要先改这里，再跑全量。")
    return 0


def _filter_models(cfg: Dict[str, Any], wanted: Optional[List[str]]) -> Dict[str, Any]:
    """按 --models 过滤配置里的模型；用于「手上只有部分通道」的场景。

    不筛选的话，探活会把还没部署的端点也报成失败，淹没真正要看的信号。
    """
    if not wanted:
        return cfg
    all_models = list(cfg.get("models") or [])
    ids = {str(m.get("id")) for m in all_models}
    unknown = sorted(set(wanted) - ids)
    if unknown:
        raise SystemExit(f"配置里没有这些模型 id：{unknown}；可用：{sorted(ids)}")
    kept = [m for m in all_models if str(m.get("id")) in set(wanted)]
    print(f"仅针对指定的 {len(kept)} 个模型（配置里共 {len(all_models)} 个）："
          f"{', '.join(str(m.get('id')) for m in kept)}\n")
    out = dict(cfg)
    out["models"] = kept
    return out


def cmd_probe(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(_load_config(args.config), args)
    cfg = _filter_models(cfg, _split(args.models))
    print("模型端点探活（每个模型发一条最小请求）\n")
    rows = probe_models(cfg)
    ok = 0
    for r in rows:
        flag = "OK " if r.get("ok") else "FAIL"
        if r.get("ok"):
            ok += 1
        print(f"[{flag}] {r.get('label')}  {r.get('base_url')}")
        if r.get("ok"):
            print(f"        时延 {r.get('latency_ms')} ms ｜ 服务端模型名 {r.get('service_model')} ｜ 样例输出 {r.get('sample')!r}")
        else:
            print(f"        错误：{r.get('error')}")
    print(f"\n{ok}/{len(rows)} 个模型可用")
    if ok == len(rows):
        print("端点全部就绪，可以继续：python3 -m geval run --config <cfg> --smoke --limit 8")
    else:
        print("有端点不可用。若这些是尚未部署的模型（如微调后的通道），")
        print("用 --models 只探活手上已有的通道即可，例如：")
        print("  python3 -m geval probe --config <cfg> --models qwen3-32b")
    return 0 if ok == len(rows) else 1


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(_load_config(args.config), args)
    if args.smoke:
        cfg["synthetic_n"] = cfg.get("synthetic_n") or 24
    spec = make_run_spec(
        cfg,
        run_name=args.run_name,
        resume=bool(args.resume),
        limit_override=args.limit if args.limit else None,
        task_filter=_split(args.tasks),
        model_filter=_split(args.models),
        # --smoke 同时允许退回合成数据：可先用真实模型跑通接线（地址/鉴权/字段），
        # 不消耗真实题目，确认无误后再切全量。
        allow_synthetic=bool(args.synthetic or args.smoke),
        quiet=bool(args.quiet),
    )
    print(f"geval {__version__} — 运行开始：{spec.run_name}")
    print(f"  输出目录   {spec.run_dir}")
    print(f"  并发数     {spec.concurrency}")
    print(f"  代理策略   {cfg.get('proxy') or 'bypass'}")
    if spec.limit_override:
        print(f"  抽样上限   {spec.limit_override} 条/任务")
    print()

    t0 = time.time()
    runner = Runner(spec)
    result = runner.run()
    elapsed = time.time() - t0

    paths = rp.build_report(
        spec.run_dir,
        baseline=args.baseline or (cfg.get("report") or {}).get("baseline"),
        pairs=(cfg.get("report") or {}).get("pairs"),
    )

    print(f"\n完成，用时 {elapsed:.1f}s")
    _print_leaderboard(result["summary"])
    if result["errors"]:
        print(f"\n有 {len(result['errors'])} 条错误，详见 summary.json 的 errors 字段：")
        for err in result["errors"][:5]:
            print(f"  - {err}")
    print(f"\n报告：{paths['html']}")
    print(f"      {paths['markdown']}")
    return 0


def cmd_count(args: argparse.Namespace) -> int:
    """推算本次评测的样本量与请求量，不发任何网络请求。

    行内跑全量前必须先看这个数字：CMMLU 单独就有上万条，三个模型叠加很容易
    打出几万次请求。这里让你在花钱之前就看清规模。
    """
    from . import datasets as _ds

    cfg = _apply_overrides(_load_config(args.config), args)
    data_root = str(cfg.get("data_root") or "datasets")
    models = list(cfg.get("models") or [])
    wanted = _split(args.models)
    if wanted:
        models = [m for m in models if str(m.get("id")) in set(wanted)]
    n_models = max(1, len(models))

    rows: List[List[Any]] = []
    total_requests = 0
    total_samples = 0
    missing: List[str] = []

    for tspec in cfg.get("tasks") or []:
        name = str(tspec.get("name"))
        key = str(tspec.get("tag") or name)
        try:
            task = tk.get_task(name, tspec)
            records, _examples = task.prepare(
                data_root,
                allow_synthetic=bool(args.synthetic),
                synthetic_n=int(cfg.get("synthetic_n") or 60),
            )
        except _ds.DatasetError:
            rows.append([key, "—", "缺失", "—", "—", "—"])
            missing.append(f"{name}（期望 datasets/{name}/test.jsonl）")
            continue
        except KeyError as exc:
            rows.append([key, "—", "配置错误", "—", "—", str(exc)])
            continue

        per = 4 if task.mode == "loglikelihood" else 1
        n = len(records)
        req = n * per * n_models
        total_samples += n
        total_requests += req
        rows.append([key, task.mode, n, per, n_models, req])

    print(f"geval {__version__} — 用量推算（未发送任何请求）\n")
    print(f"  数据根目录  {os.path.abspath(data_root)}")
    print(f"  参与模型    {n_models} 个：" + ", ".join(str(m.get("id")) for m in models))
    print()
    from .util import display_width, pad

    head = (
        pad("任务", 20) + pad("打分模式", 16) + pad("样本数", 9, align="right")
        + pad("请求/样本", 11, align="right") + pad("模型数", 8, align="right") + pad("总请求", 12, align="right")
    )
    print(head)
    print("-" * display_width(head))
    for r in rows:
        print(
            pad(r[0], 20) + pad(r[1], 16) + pad(r[2], 9, align="right")
            + pad(r[3], 11, align="right") + pad(r[4], 8, align="right") + pad(r[5], 12, align="right")
        )
    print("-" * display_width(head))
    print(
        pad("合计", 20) + pad("", 16) + pad(total_samples, 9, align="right")
        + pad("", 11) + pad(n_models, 8, align="right") + pad(total_requests, 12, align="right")
    )
    print()

    if missing:
        print("以下任务的数据集缺失，已按 0 计入：")
        for m in missing:
            print(f"  · {m}")
        print("  准备方式见 datasets/README.md，或先加 --synthetic 验证链路。\n")

    if total_requests == 0:
        print("没有可执行的样本。请先准备数据集，或用 python -m geval run --smoke 验证链路。")
        return 1

    # 时长估算：请求数 × 单请求时延 ÷ 并发
    p50 = args.p50_ms
    conc = int(cfg.get("concurrency") or 8)
    if p50:
        seconds = total_requests * (float(p50) / 1000.0) / max(1, conc)
        print(f"按单请求 {float(p50):.0f} ms、并发 {conc} 估算：约 {seconds / 60:.1f} 分钟")
    else:
        print(f"时长估算：用实测单请求时延代入 —— 总请求 {total_requests} × 时延(秒) ÷ 并发 {conc}")
        example_min = total_requests * 0.5 / max(1, conc) / 60
        print(f"          例：单请求 0.5 s、并发 {conc} → 约 {example_min:.1f} 分钟")

    if total_requests > 20000:
        print()
        print("⚠️ 请求量较大。建议先抽样确认排序稳定，再决定是否跑全量：")
        print("   python -m geval run --config config.json --limit 40")
    return 0


def _find_venvs() -> List[str]:
    """找虚拟环境。

    不靠目录名猜，而是靠特征识别：含 pyvenv.cfg 且带解释器。
    这样用户给 venv 起任何名字（venv / .venv / myenv / lmenv …）都能被发现。
    """
    skip = {"wheels", "dist", "datasets", "docs", "geval", "scripts", "runs", ".cache", ".git", "__pycache__"}
    roots = [os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    cands: List[str] = []

    def add(p: str) -> None:
        if p not in cands and os.path.isdir(p) and _venv_python(p):
            cands.append(p)

    for root in roots:
        if not os.path.isdir(root):
            continue
        # 常见名字优先
        for n in ("venv", ".venv", "env", ".env", "evalenv", ".evalenv", "lmenv", ".lmenv"):
            add(os.path.join(root, n))
        # 再按特征扫一遍（应对自定义名字）
        try:
            for entry in sorted(os.listdir(root)):
                if entry in skip or entry.startswith("."):
                    continue
                p = os.path.join(root, entry)
                if not os.path.isdir(p):
                    continue
                if os.path.exists(os.path.join(p, "pyvenv.cfg")):
                    add(p)
        except OSError:
            pass
    return cands


def _venv_python(venv: str) -> Optional[str]:
    # 用 path parts 而非 "bin/python" 字符串，避免在 Windows 上拼出
    # "venv\Scripts/python.exe" 这种混合分隔符（提示信息会被用户复制粘贴）
    for parts in (("bin", "python"), ("bin", "python3"), ("Scripts", "python.exe")):
        p = os.path.join(venv, *parts)
        if os.path.exists(p):
            return p
    return None


def _probe_venv(venv: str) -> Dict[str, Any]:
    """用一个独立子进程检查 venv 里的 Python 与 lm-eval。"""
    import json as _json
    import subprocess

    py = _venv_python(venv)
    if not py:
        return {"ok": False, "error": "找不到解释器（bin/python 或 Scripts/python.exe）"}

    code = (
        "import json,sys,importlib.util as u\n"
        "d={'python':sys.version.split()[0]}\n"
        "try:\n"
        "    import lm_eval; d['lm_eval']=getattr(lm_eval,'__version__','?')\n"
        "except Exception as e:\n"
        "    d['lm_eval']=None; d['lm_eval_error']=type(e).__name__+': '+str(e)[:120]\n"
        "d['torch']= u.find_spec('torch') is not None\n"
        "d['transformers']= u.find_spec('transformers') is not None\n"
        "d['datasets']= u.find_spec('datasets') is not None\n"
        "print(json.dumps(d))\n"
    )
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out = (r.stdout or "").strip().splitlines()
    if not out:
        return {"ok": False, "error": (r.stderr or "无输出").strip()[:200]}
    try:
        d = _json.loads(out[-1])
    except _json.JSONDecodeError:
        return {"ok": False, "error": out[-1][:200]}
    d["ok"] = True
    d["python_bin"] = py

    # 顺带确认 lm-eval 能列出任务（纯本地操作，不联网），并检查任务名是否齐全
    if d.get("lm_eval"):
        try:
            r2 = subprocess.run([py, "-m", "lm_eval", "--tasks", "list"], capture_output=True, text=True, timeout=300)
            listing = (r2.stdout or "") + (r2.stderr or "")
            d["tasks_list_ok"] = r2.returncode == 0 and len(listing) > 200
            d["found_tasks"] = sorted(
                {t for t in ("ceval", "cmmlu", "mmlu", "gsm8k", "bbh", "ifeval") if t in listing}
            )
            d["n_task_lines"] = listing.count("\n")
        except (OSError, subprocess.SubprocessError) as exc:
            d["tasks_list_ok"] = False
            d["tasks_list_error"] = f"{type(exc).__name__}"
    return d


def _hf_cache_info() -> Dict[str, Any]:
    """检查 HuggingFace 数据集缓存。lm-eval 离线跑之前必须有它，否则会联网超时。"""
    home = os.path.expanduser("~")
    cands = [
        os.path.join(os.environ["HF_HOME"], "datasets") if os.environ.get("HF_HOME") else "",
        os.path.join(home, ".cache", "huggingface", "datasets"),
        os.environ.get("HF_DATASETS_CACHE", ""),
    ]
    for p in cands:
        if p and os.path.isdir(p):
            entries = [x for x in os.listdir(p) if not x.startswith(".")]
            return {"path": p, "entries": len(entries), "exists": True}
    return {"path": cands[1], "entries": 0, "exists": False}


def cmd_doctor(args: argparse.Namespace) -> int:
    """一条命令检查「是否已经可以跑评测」，并给出缺失项的修复动作。

    检查顺序刻意按「阻塞程度」排：框架能否跑 → 数据集有没有 → 端点通不通。
    前面的项没过，后面跑了也是白跑。
    """
    from . import datasets as _ds
    from .util import display_width, pad

    blockers: List[str] = []
    warns: List[str] = []
    next_steps: List[str] = []

    def head(title: str) -> None:
        print()
        print(f"── {title} " + "─" * max(0, 66 - display_width(title)))

    def item(level: str, text: str, fix: str = "") -> None:
        tag = {"ok": "\033[32m[OK]\033[0m", "warn": "\033[33m[警告]\033[0m", "fail": "\033[31m[阻塞]\033[0m"}[level]
        print(f"{tag} {text}")
        if fix:
            print(f"       → {fix}")
        if level == "fail":
            blockers.append(f"{text}" + (f" ｜ {fix}" if fix else ""))
        elif level == "warn":
            warns.append(text)

    print(f"geval {__version__} — 就绪状态检查")
    print(f"  工作目录  {os.getcwd()}")

    # ---------------- 1. 运行环境 ----------------
    head("1. 运行环境")
    py_ok = sys.version_info >= (3, 9)
    item("ok" if py_ok else "fail", f"Python {sys.version.split()[0]}（需 ≥ 3.9）",
         "" if py_ok else "升级 Python")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    item("ok", f"解释器 {sys.executable}" + ("（虚拟环境）" if in_venv else "（系统环境）"))

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.normcase(pkg_root) != os.path.normcase(os.getcwd()):
        item("warn", f"geval 包位于 {pkg_root}，与当前工作目录不同",
             "在项目根目录执行，或设置 PYTHONPATH 指向项目根")
    else:
        item("ok", "在项目根目录执行（geval 可正常导入）")

    # ---------------- 2. 框架自检 ----------------
    head("2. 框架自检")
    try:
        import importlib.util as _ilu

        vd_path = os.path.join(pkg_root, "scripts", "verify_deps.py")
        spec = _ilu.spec_from_file_location("_geval_verify_deps", vd_path)
        vd = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(vd)  # type: ignore[union-attr]
        third, std = vd.classify(vd.scan(vd.RUNTIME_DIRS))
        item("ok" if not third else "fail",
             f"运行时依赖：{'无（零第三方依赖）' if not third else ', '.join(sorted(third))}，标准库模块 {len(std)} 个",
             "" if not third else "拷贝不完整，重新拷贝整个项目目录")
    except Exception as exc:  # noqa: BLE001
        item("warn", f"依赖核查跳过：{type(exc).__name__}: {exc}")

    if args.no_selftest:
        item("warn", "已跳过离线自检（--no-selftest）")
    else:
        import subprocess

        try:
            r = subprocess.run(
                [sys.executable, "-m", "geval", "selftest", "--out", os.path.join(".cache", "_doctor")],
                capture_output=True, text=True, timeout=600,
            )
            passed = r.returncode == 0 and "[PASS]" in (r.stdout or "")
            item("ok" if passed else "fail",
                 "离线自检通过（7 个任务、抽取层、acc/acc_norm、退化判定、续跑、报告生成）" if passed
                 else "离线自检未通过",
                 "" if passed else "把 `python3 -m geval selftest` 的完整输出发给维护方")
        except (OSError, subprocess.SubprocessError) as exc:
            item("warn", f"自检未能执行：{type(exc).__name__}: {exc}")

    # ---------------- 3. 模型端点配置 ----------------
    head("3. 配置与模型端点")
    cfg: Dict[str, Any] = {}
    cfg_path = args.config
    if cfg_path and os.path.exists(cfg_path):
        try:
            cfg = read_json(cfg_path)
            item("ok", f"配置已加载 {cfg_path}：{len(cfg.get('models') or [])} 个模型、"
                       f"{len(cfg.get('tasks') or [])} 个任务条目")
        except Exception as exc:  # noqa: BLE001
            item("fail", f"配置解析失败 {cfg_path}：{exc}", "检查 JSON 语法（多余逗号、中文引号）")
    else:
        item("warn", f"未找到配置文件 {cfg_path or 'config.json'}",
             "cp config.example.json config.json 并填入模型地址")
        next_steps.append(f"cp config.example.json config.json   # 然后填 base_url / model")

    models = list(cfg.get("models") or [])
    if models:
        # mock 模型不发网络请求，本来就不需要 base_url / model
        real = [m for m in models if str(m.get("provider") or "openai_compat").lower() != "mock"]
        bad = [m.get("id") for m in real if not (m.get("base_url") and m.get("model"))]
        if bad:
            item("fail", f"这些模型缺少 base_url 或 model 字段：{bad}", "补齐后再探活")
        elif real:
            item("ok", f"{len(real)} 个真实模型都填了 base_url 与 model"
                       + (f"（另有 {len(models) - len(real)} 个 mock 模型）" if len(models) != len(real) else ""))
        else:
            item("ok", f"{len(models)} 个模型全部为 mock（干跑配置，不需要网络）")
        pairs = ((cfg.get("report") or {}).get("pairs")) or []
        ids = {str(m.get("id")) for m in models}
        missing = [x for p in pairs for x in p if x not in ids]
        if pairs and missing:
            item("warn", f"report.pairs 引用了不存在的模型 id：{missing}", "修正或删除该配对")

    proxy_env = {k: os.environ[k] for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy") if os.environ.get(k)}
    if proxy_env:
        item("warn", f"检测到代理环境变量 {list(proxy_env)}",
             "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy   （否则内网请求会被发给外网代理，表现为超时或 502）")
    else:
        item("ok", "未检测到代理环境变量")
    item("ok" if (cfg.get("proxy") or "bypass") == "bypass" else "warn",
         f"配置 proxy = {cfg.get('proxy') or 'bypass'}" + ("" if (cfg.get("proxy") or "bypass") == "bypass" else "（内网建议 bypass）"))

    if args.probe and models:
        head("3b. 端点探活")
        try:
            from .runner import probe_models

            probe_cfg = _filter_models(cfg, _split(args.models)) if getattr(args, "models", None) else cfg
            for rec in probe_models(probe_cfg, timeout=float(cfg.get("request_timeout_s") or 30)):
                if str(rec.get("provider")) == "mock" or rec.get("note"):
                    item("ok", f"{rec.get('label')}：mock 模型，无需网络")
                elif rec.get("ok"):
                    item("ok", f"{rec.get('label')}：时延 {rec.get('latency_ms')} ms，"
                               f"服务端模型名 {rec.get('service_model')}")
                else:
                    item("fail", f"{rec.get('label')} 不可用：{rec.get('error')}",
                         "检查服务是否已启动、端口与 model 名是否正确")
        except Exception as exc:  # noqa: BLE001
            item("warn", f"探活失败：{type(exc).__name__}: {exc}")
    elif models:
        next_steps.append("python3 -m geval probe --config config.json   # 确认三个端点可用")

    # ---------------- 4. 数据集 ----------------
    head("4. 数据集")
    data_root = str(cfg.get("data_root") or "datasets")
    tasks_cfg = cfg.get("tasks") or []
    ready = 0
    missing_tasks: List[str] = []
    for tspec in tasks_cfg:
        name = str(tspec.get("name"))
        key = str(tspec.get("tag") or name)
        split = str(tspec.get("split") or getattr(tk.TASK_REGISTRY.get(name), "default_split", "test"))
        path = _ds.resolve_dataset_path(name, split, data_root, tspec.get("path"))
        if not path:
            # 用 tag（若有）标识，否则多个同名任务会重复列出同一个名字
            missing_tasks.append(f"{key}（期望 {data_root}/{name}/{split}.jsonl）")
            continue
        try:
            n = 0
            first_ok = True
            with open(path, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    if i == 0:
                        import json as _json

                        try:
                            _json.loads(line)
                        except _json.JSONDecodeError:
                            first_ok = False
                    n += 1
            lim = int(tspec.get("limit") or 0)
            shown = f"{n} 条" + (f"，本次限 {lim} 条" if lim else "")
            item("ok" if first_ok else "fail", f"{key}：{shown}  ←  {path}",
                 "" if first_ok else "首行不是合法 JSON，检查文件格式（须为 JSONL，一行一条）")
            ready += 1
        except OSError as exc:
            item("fail", f"{key}：无法读取 {path}（{exc}）")

    if missing_tasks:
        item("warn", f"{len(missing_tasks)}/{len(tasks_cfg) or 0} 个任务的数据集缺失",
             "见下方「数据集准备」；或先加 --smoke 用合成数据验证链路")
        for m in missing_tasks[:8]:
            print(f"       · {m}")
        if len(missing_tasks) > 8:
            print(f"       · …另有 {len(missing_tasks) - 8} 项")
        next_steps.append("准备数据集：见 docs/INTRANET_RUNBOOK.md §4（三条路径）")
    elif tasks_cfg:
        item("ok", f"全部 {ready} 个任务的数据集就绪")

    # ---------------- 5. lm-eval（可选） ----------------
    head("5. lm-eval 离线锚点（可选，不影响 geval 主流程）")
    venvs = ([args.venv] if args.venv else []) + _find_venvs()
    tested = False
    for venv in venvs:
        if not venv or not os.path.isdir(venv):
            continue
        tested = True
        info = _probe_venv(venv)
        if not info.get("ok"):
            item("warn", f"{venv}：{info.get('error')}", "确认这是虚拟环境目录，或指定 --venv 路径")
            continue
        if not info.get("lm_eval"):
            item("warn", f"{venv}：Python {info['python']}，但 lm-eval 未安装",
                 f"{_venv_python(venv)} -m pip install --no-index --find-links wheels/bootstrap --find-links wheels/lm-eval 'lm-eval[api,ifeval]==0.4.13'")
            continue
        item("ok", f"{venv}：Python {info['python']}，lm-eval {info['lm_eval']}"
                   + ("，datasets 已装" if info.get("datasets") else "，**datasets 未装**"))
        if info.get("torch"):
            item("warn", "该环境里装了 torch（评测 API 端点并不需要）",
                 "若是误装 lm-eval[hf] extra 导致，可忽略；但要留意包体与依赖树")
        if info.get("tasks_list_ok"):
            found = info.get("found_tasks") or []
            item("ok" if len(found) >= 4 else "warn",
                 f"lm-eval 可列出任务；本次关心的任务命中 {len(found)}/6：{', '.join(found) or '无'}",
                 "" if len(found) >= 4 else "任务命名可能随版本变化，用 `lm_eval --tasks list | grep -i ceval` 确认实际名字")
        else:
            item("warn", "lm-eval 无法列出任务（可能是依赖不全）", "在该 venv 里执行 pip check")
        break

    if not tested:
        item("warn", "项目目录下未发现虚拟环境",
             "若已装 lm-eval，用 --venv 指定路径；或忽略本项（geval 不需要它）")

    hf = _hf_cache_info()
    if not hf["exists"] or hf["entries"] == 0:
        item("warn", f"HF 数据集缓存为空（{hf['path']}）",
             "lm-eval 离线跑之前必须先预热并拷入缓存，否则会联网超时 —— 见 docs/OFFLINE_DEPLOY.md 坑 1")
    else:
        item("ok", f"HF 数据集缓存：{hf['entries']} 个条目（{hf['path']}）")
    if not os.environ.get("HF_DATASETS_OFFLINE"):
        item("warn", "未设置 HF_DATASETS_OFFLINE=1",
             "内网跑 lm-eval 前必须 export，否则即使缓存已存在也会先尝试联网检查版本")

    # ---------------- 6. 结论 ----------------
    print()
    print("═" * 68)
    if blockers:
        print(f"\033[31m有 {len(blockers)} 项阻塞，先解决再跑评测：\033[0m")
        for b in blockers:
            print(f"  · {b}")
    elif warns:
        print(f"\033[33m无阻塞项，但有 {len(warns)} 条提醒（多数不影响主流程）：\033[0m")
        for w in warns:
            print(f"  · {w}")
    else:
        print("\033[32m全部就绪。\033[0m")
    print("═" * 68)

    if next_steps:
        print("\n建议的下一步：")
        for s in next_steps:
            print(f"  {s}")

    print()
    print("验证通过后，按顺序执行（详见 docs/INTRANET_RUNBOOK.md §5）：")
    print("  python3 -m geval probe --config config.json                 # 端点探活")
    print("  python3 -m geval run --config config.json --smoke --limit 8  # 冒烟，验证接线")
    print("  python3 -m geval count --config config.json                 # 用量预估")
    print("  python3 -m geval run --config config.json --limit 40 --resume  # 抽样看排序")
    print("  python3 -m geval run --config config.json --resume           # 全量")
    return 1 if blockers else 0


def _print_leaderboard(summary: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    from .util import display_width, pad

    tasks = rp.available_tasks({"results": summary})
    models = list(summary.keys())
    if not tasks or not models:
        return
    width = max(20, max((display_width(m) for m in models), default=20) + 2)
    head = pad("model", width) + "".join(pad(t, 14, align="right") for t in tasks)
    print(head)
    print("-" * display_width(head))
    for mid in models:
        cells = []
        for t in tasks:
            agg = (summary.get(mid) or {}).get(t) or {}
            cells.append(pad("—" if agg.get("score") is None else f"{agg['score'] * 100:.1f}%", 14, align="right"))
        print(pad(mid, width) + "".join(cells))


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = args.run
    if not run_dir:
        out_dir = str(_load_config(args.config).get("output_dir") or "runs") if args.config else "runs"
        if not os.path.isdir(out_dir):
            raise SystemExit(f"找不到 runs 目录：{out_dir}")
        candidates = sorted(
            (d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))),
            key=lambda d: os.path.getmtime(os.path.join(out_dir, d)),
        )
        if not candidates:
            raise SystemExit("runs 目录下没有可用的运行结果")
        run_dir = os.path.join(out_dir, candidates[-1])
        print(f"未指定 --run，使用最近一次运行：{run_dir}")
    paths = rp.build_report(run_dir, baseline=args.baseline, pairs=args.pairs)
    print(f"报告已生成：\n  {paths['html']}\n  {paths['markdown']}")
    return 0


# --------------------------------------------------------------------------- #
def _selftest_config(base_dir: str, n: int) -> Dict[str, Any]:
    return {
        "run_name": "selftest",
        "output_dir": base_dir,
        "data_root": os.path.join(base_dir, "datasets"),
        "concurrency": 8,
        "seed": 42,
        "proxy": "bypass",
        "synthetic_n": n,
        "report": {
            "bootstrap": 300,
            "alpha": 0.05,
            "baseline": "mock-weak",
            # 用 mid -> weak 作为「微调后变差」的模拟对，用于校验退化判定的方向性
            "pairs": [["mock-mid", "mock-weak"]],
        },
        "models": [
            {"id": "mock-strong", "label": "模拟-强", "provider": "mock", "accuracy": 0.92, "noise": 0.15, "latency_ms": 900},
            {"id": "mock-mid", "label": "模拟-中", "provider": "mock", "accuracy": 0.74, "noise": 0.12, "latency_ms": 460},
            {"id": "mock-weak", "label": "模拟-弱", "provider": "mock", "accuracy": 0.48, "noise": 0.10, "latency_ms": 300},
        ],
        "tasks": [
            {"name": "ceval", "num_fewshot": 3},
            {"name": "ceval", "tag": "ceval-logprob", "mode": "logprob", "num_fewshot": 3},
            {"name": "ceval", "tag": "ceval-loglikelihood", "mode": "loglikelihood", "num_fewshot": 3},
            {"name": "gsm8k"},
            {"name": "bbh"},
            {"name": "ifeval"},
            {"name": "json_schema"},
        ],
    }


def _check_loglikelihood_math() -> List[str]:
    """离线验证 acc / acc_norm 的续写似然解析是否与 lm-eval 口径一致。

    不依赖任何服务：直接喂一个形状与 vLLM /completions echo 响应相同的 JSON。
    """
    from .client import _sum_continuation, _token_logprobs

    problems: List[str] = []

    # 情形一：续写是单个 token（" A"），acc 与 acc_norm 相同
    echo_single = {
        "choices": [
            {
                "text": "问题：1+1=?\nA. 1\nB. 2\n答案： A",
                "logprobs": {
                    "tokens": ["问题", "：", "1", "+", "1", "=?", "\n", "A", ".", " ", "1", "\n", "B", ".", " ", "2", "\n", "答案", "：", " A"],
                    "token_logprobs": [None, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.9],
                },
            }
        ]
    }
    pairs = _token_logprobs(echo_single)
    lp, n = _sum_continuation(pairs, "A")
    if n != 1 or abs(lp - (-0.9)) > 1e-6:
        problems.append(f"单 token 续写解析错误：logprob={lp} n_tokens={n}（期望 -0.9 / 1）")

    # 情形二：续写被切成两个 token（" " + "A"），应把两者的 logprob 都算进去
    echo_split = {
        "choices": [
            {
                "text": "...答案： A",
                "logprobs": {
                    "tokens": ["答案", "：", " ", "A"],
                    "token_logprobs": [None, -0.1, -0.4, -0.6],
                },
            }
        ]
    }
    lp2, n2 = _sum_continuation(_token_logprobs(echo_split), "A")
    if n2 != 2 or abs(lp2 - (-1.0)) > 1e-6:
        problems.append(f"多 token 续写解析错误：logprob={lp2} n_tokens={n2}（期望 -1.0 / 2）")

    # 情形三：acc 与 acc_norm 必须能给出不同排序（长度归一化的意义所在）
    #   选项 A：" A"        1 token,  logprob -1.2  -> acc_norm -1.20
    #   选项 B：" 选项B"    5 tokens, logprob -3.0  -> acc_norm -0.60
    tab = {"A": {"logprob": -1.2, "n_tokens": 1.0}, "B": {"logprob": -3.0, "n_tokens": 5.0}}
    by_acc = max(tab.items(), key=lambda kv: kv[1]["logprob"])[0]
    by_norm = max(tab.items(), key=lambda kv: kv[1]["logprob"] / kv[1]["n_tokens"])[0]
    if by_acc != "A" or by_norm != "B":
        problems.append(f"acc / acc_norm 排序逻辑异常：acc 选 {by_acc}，acc_norm 选 {by_norm}")

    # 情形四：找不到匹配时不得抛异常，且应退化为取最后一个有效 token
    lp3, n3 = _sum_continuation(_token_logprobs(echo_split), "ZZZZ")
    if n3 != 1 or abs(lp3 - (-0.6)) > 1e-6:
        problems.append(f"兜底路径异常：logprob={lp3} n_tokens={n3}（期望 -0.6 / 1）")

    return problems


def cmd_selftest(args: argparse.Namespace) -> int:
    base_dir = os.path.abspath(args.out or os.path.join(".cache", "geval-selftest"))
    ensure_dir(base_dir)
    cfg = _selftest_config(base_dir, int(args.n))
    spec = make_run_spec(cfg, run_name="selftest", allow_synthetic=True, quiet=True)

    print(f"geval {__version__} — 离线自检")
    print(f"  使用确定性 mock 模型与内置合成数据，不产生任何网络请求")
    print(f"  输出目录 {spec.run_dir}\n")

    t0 = time.time()
    result = Runner(spec).run()
    elapsed = time.time() - t0

    paths = rp.build_report(spec.run_dir, baseline="mock-weak")

    print(f"\n自检完成，用时 {elapsed:.1f}s\n")
    _print_leaderboard(result["summary"])

    # ---- 断言式校验：验证的是「流水线本身」，而不是模型能力 ----
    problems: List[str] = []
    # 纯数学校验：不依赖任何服务，直接验证 acc / acc_norm 的续写似然解析口径
    problems.extend(_check_loglikelihood_math())
    summary = result["summary"]
    tasks = rp.available_tasks({"results": summary})

    for t in tasks:
        per = {m: (summary.get(m) or {}).get(t) or {} for m in summary}
        if not any(agg.get("n") for agg in per.values()):
            problems.append(f"任务 {t} 没有产出任何样本")
            continue
        strong = (per.get("mock-strong") or {}).get("score")
        weak = (per.get("mock-weak") or {}).get("score")
        if strong is None or weak is None:
            problems.append(f"任务 {t} 缺少强/弱模型的分数")
        elif strong <= weak:
            problems.append(f"任务 {t} 分数与 mock 设定矛盾：强模型 {strong:.3f} ≤ 弱模型 {weak:.3f}")
        note = (per.get("mock-strong") or {}).get("parse_note") or {}
        pf = note.get("parse_fail_rate")
        if pf is not None and pf > 0.25:
            problems.append(f"任务 {t} 解析失败率异常偏高（{pf:.1%}），抽取层可能有缺陷")

    # 解析失败率必须被统计出来（说明诊断链路可用）
    ceval_note = ((summary.get("mock-mid") or {}).get("ceval") or {}).get("parse_note") or {}
    if not ceval_note.get("strategies"):
        problems.append("抽取策略分布为空，诊断链路异常")

    # 结构化输出子指标
    json_metrics = ((summary.get("mock-strong") or {}).get("json_schema") or {}).get("task_metrics") or {}
    for key in ("json_valid_rate", "schema_ok_rate", "intent_accuracy", "slots_exact_rate"):
        if key not in json_metrics:
            problems.append(f"结构化输出子指标缺失：{key}")
    if json_metrics.get("json_valid_rate", 0) < json_metrics.get("schema_ok_rate", 0):
        problems.append("schema_ok_rate 高于 json_valid_rate，子指标口径异常")

    # 方向性校验：mid -> weak 是「微调后全面变差」的模拟，
    # 必须被判为退化，且绝不能出现「提升」—— 这条专门守符号约定（diff 应为 post - pre）。
    # 注意只匹配表格里的判定单元格（"| 显著退化 |"），汇总行的「显著退化任务：」不算。
    loaded = rp.load_run(spec.run_dir)
    impact = "\n".join(rp.build_impact_section(loaded, [["mock-mid", "mock-weak"]], tasks))
    if "| 显著退化 |" not in impact:
        problems.append("微调影响评估未识别出退化（mock-weak 明显弱于 mock-mid），退化判定可能失效")
    if "| 显著提升 |" in impact:
        problems.append("微调影响评估出现方向性错误：mock-weak 全面弱于 mock-mid，不应判定为「显著提升」")

    # IFEval 专项
    if_metrics = ((summary.get("mock-mid") or {}).get("ifeval") or {}).get("task_metrics") or {}
    if not if_metrics.get("n_instruction_types"):
        problems.append("IFEval 未统计到指令类型覆盖情况")

    # 报告文件
    for key in ("html", "markdown"):
        if not os.path.exists(paths[key]):
            problems.append(f"报告未生成：{paths[key]}")

    # 断点续跑：必须复现完全一致的结果，且不得重写已完成的样本文件
    first_scores = {
        (m, t): ((summary.get(m) or {}).get(t) or {}).get("score") for m in summary for t in tasks
    }
    probe_path = os.path.join(spec.run_dir, "mock-mid", "ceval.jsonl")
    before_stat = (os.path.getmtime(probe_path), os.path.getsize(probe_path))
    spec2 = make_run_spec(cfg, run_name="selftest", allow_synthetic=True, resume=True, quiet=True)
    t1 = time.time()
    second = Runner(spec2).run()
    resume_elapsed = time.time() - t1
    second_scores = {
        (m, t): ((second["summary"].get(m) or {}).get(t) or {}).get("score")
        for m in second["summary"]
        for t in rp.available_tasks({"results": second["summary"]})
    }
    if first_scores != second_scores:
        problems.append("续跑结果与首次运行不一致，续跑逻辑可能丢失或重复样本")
    if before_stat != (os.path.getmtime(probe_path), os.path.getsize(probe_path)):
        problems.append("续跑时重写了已完成的样本文件（应直接复用）")

    if problems:
        print("\n自检发现问题：")
        for p in problems:
            print(f"  [FAIL] {p}")
        print(f"\n报告：{paths['html']}")
        return 1

    print("\n[PASS] 全部自检项通过：")
    print("  · 7 个任务全部产出结果，强/弱 mock 的分数序关系正确")
    print("  · 抽取层、解析失败率统计、策略分布均可用")
    print("  · 结构化输出的 4 项子指标与 IFEval 覆盖统计正常")
    print("  · acc / acc_norm 续写似然解析口径正确（单 token / 多 token / 排序 / 兜底）")
    print("  · 微调影响评估的退化判定方向正确（无符号反转）")
    print("  · 断点续跑生效，HTML / Markdown 报告已生成")
    print(f"\n报告：{paths['html']}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geval", description="轻量级大模型通用能力评测框架")
    p.add_argument("--version", action="version", version=f"geval {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="配置文件路径（config.example.json）")
    common.add_argument("--out", help="输出根目录（覆盖 output_dir）")
    common.add_argument("--run-name", help="本次运行名称（覆盖 run_name）")
    common.add_argument("--data-root", help="数据集根目录（覆盖 data_root）")
    common.add_argument("--proxy", help="代理策略：bypass / env / http://ip:port")
    common.add_argument("--concurrency", type=int, help="并发请求数")

    r = sub.add_parser("run", parents=[common], help="执行评测")
    r.add_argument("--smoke", action="store_true", help="冒烟模式：合成数据快速跑通")
    r.add_argument("--synthetic", action="store_true", help="缺失数据集时允许退回内置合成数据")
    r.add_argument("--limit", type=int, default=0, help="每个任务抽样上限（分层抽样）")
    r.add_argument("--tasks", help="仅运行指定任务（逗号分隔）")
    r.add_argument("--models", help="仅运行指定模型 id（逗号分隔）")
    r.add_argument("--resume", action="store_true", help="断点续跑，跳过已完成样本")
    r.add_argument("--baseline", help="报告中作为对比基线的模型 id")
    r.add_argument("--quiet", action="store_true", help="不打印进度")
    r.set_defaults(func=cmd_run)

    rp_ = sub.add_parser("report", parents=[common], help="根据已有结果生成报告")
    rp_.add_argument("--run", help="run 目录（默认取 runs 下最近一次）")
    rp_.add_argument("--baseline", help="基线模型 id")
    rp_.add_argument("--pairs", help="配对对比，格式 a:b,c:d（用于微调前后对比）")
    rp_.set_defaults(func=cmd_report)

    st = sub.add_parser("selftest", help="离线端到端自检（mock 模型 + 合成数据）")
    st.add_argument("--out", help="自检输出目录")
    st.add_argument("--n", type=int, default=60, help="每个任务的合成样本数")
    st.set_defaults(func=cmd_selftest)

    lt = sub.add_parser("list-tasks", help="列出内置任务")
    lt.set_defaults(func=cmd_list_tasks)

    pv = sub.add_parser("preview", parents=[common], help="打印实际发送给模型的 prompt")
    pv.add_argument("--task", default="ceval", help="任务名")
    pv.add_argument("--index", type=int, default=0, help="第几条样本")
    pv.set_defaults(func=cmd_preview)

    ct = sub.add_parser("count", parents=[common], help="推算样本量与请求量（不发请求）")
    ct.add_argument("--models", help="仅统计指定模型 id（逗号分隔）")
    ct.add_argument("--p50-ms", type=float, help="已知单请求时延，用于估算总耗时")
    ct.add_argument("--synthetic", action="store_true", help="缺失数据集时按内置合成数据估算")
    ct.set_defaults(func=cmd_count)

    dr = sub.add_parser("doctor", parents=[common], help="★ 就绪状态检查：一条命令查全部前置条件")
    dr.add_argument("--probe", action="store_true", help="顺带做端点探活（会发请求）")
    dr.add_argument("--venv", help="指定虚拟环境目录（默认自动发现项目内的 venv）")
    dr.add_argument("--no-selftest", action="store_true", help="跳过离线自检（更快）")
    dr.add_argument("--models", help="探活只针对指定模型 id（逗号分隔，配合 --probe）")
    dr.set_defaults(func=cmd_doctor)

    pb = sub.add_parser("probe", parents=[common], help="模型端点探活")
    pb.add_argument("--models", help="只探活指定模型 id（逗号分隔）")
    pb.set_defaults(func=cmd_probe)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "report" and args.pairs:
        args.pairs = [[a, b] for a, b in (pair.split(":") for pair in args.pairs.split(",") if ":" in pair)]
    else:
        args.pairs = None
    cmd = args.func
    for key in ("pairs",):
        if not hasattr(args, key):
            setattr(args, key, None)
    try:
        return int(cmd(args))
    except KeyboardInterrupt:
        print("\n已中断（已完成的样本已落盘，可加 --resume 续跑）", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("GEVAL_DEBUG"):
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
