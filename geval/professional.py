# -*- coding: utf-8 -*-
"""专业能力评测（阶段 4）：银行语音导航 · 意图树路由。

与通用基准的差别
----------------
通用基准判「答对 / 答错」；这里要回答的是「模型能不能在 498 个业务节点里
稳定选中正确意图，且成本与时延是否可接受」，因此除准确率外还必须产出：

  * 小模型基线对照（同一批样本、同一套标签，逐样本配对）
  * 幻觉意图率（输出了候选树之外的标签 —— 下游会直接崩）
  * 意图混淆对（定位是哪两个节点在打架）
  * 业务 / 通用交互分层准确率（整体分会被「人工服务」这类高频简单意图主导）
  * 逐样本与聚合的 token、时延、缓存命中、截断情况

数据集由 scripts/prepare_professional_dataset.py 从业务标注 JSONL 转换而来：
  {"id", "question"(完整 prompt), "answer"(人工意图名称), "subject"(样本类型), "meta"}
同目录的 labels.json 提供候选意图树的全部末级名称与层级路径。

本模块只依赖 geval 已有的抽取层 / 统计层 / 落盘层，不改动原有任务逻辑。
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import datasets as ds
from .answers import strip_reasoning
from .tasks import MAX_RESPONSE_CHARS, BaseTask, WorkItem
from .util import now_iso, truncate

# 兜底标签：数据集 labels.json 缺失时（离线自检）仍可跑通链路
FALLBACK_LABELS = [
    "人工服务", "集外", "友好问候", "肯定", "否定", "简短词",
    "个人账户余额查询", "个人账户交易明细查询", "账户开户行查询",
    "借记卡人工挂失", "信用卡业务", "投诉建议",
]

_PREFIX = re.compile(r"^(?:意图|答案|输出|结果|intent|answer)\s*(?:名称)?\s*[是为:：\-=>]*\s*", re.I)
_EDGE = "。.、,，;；:：\"'()（）[]【】{}`*<>/\\"
_FENCE = re.compile(r"```(?:[A-Za-z]*)?\s*(.*?)```", re.S)


def normalize_label(text: Optional[str]) -> str:
    """比对用的归一化：去标点/空白、全角转半角、小写。"""
    if not text:
        return ""
    s = str(text).translate(str.maketrans("ＡＢＣＤＥＦＧＨＩＪａｂｃｄｅｆｇｈｉｊ０１２３４５６７８９（）［］",
                                          "ABCDEFGHIJabcdefghij0123456789()[]"))
    s = re.sub(r"[\s\-_/\\]+", "", s.lower())
    return re.sub(r"[。，、；：！？,.;:!?\"'“”‘’()\[\]{}<>《》【】`*]", "", s)


def extract_intent(text: str, labels: Sequence[str]) -> Tuple[Optional[str], str, bool, bool]:
    """从自由文本里抽出意图名称。

    返回 (pred | None, strategy, in_tree, ambiguous)。
    in_tree=False 表示模型输出了候选树之外的标签（幻觉意图），或根本没给出可识别标签。
    """
    label_set: Set[str] = {str(x).strip() for x in labels if str(x).strip()}
    norm_map: Dict[str, str] = {normalize_label(x): x for x in label_set}
    cleaned = strip_reasoning(text or "")
    fence = _FENCE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned.strip():
        return None, "empty", False, False

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    tail = _PREFIX.sub("", lines[-1] if lines else "").strip().strip(_EDGE).strip()
    if tail in label_set:
        return tail, "exact_last_line", True, False
    ntail = normalize_label(tail)
    if ntail in norm_map:
        return norm_map[ntail], "normalized_last_line", True, False

    # 整段短输出（模型把标签和引号/句号写在同一行）
    whole = _PREFIX.sub("", cleaned.strip()).strip().strip(_EDGE).strip()
    if whole in label_set:
        return whole, "exact_whole", True, False
    if normalize_label(whole) in norm_map:
        return norm_map[normalize_label(whole)], "normalized_whole", True, False

    # 兜底：文本里出现过的已知标签，取「最后出现位置」最靠后的（结论通常在末尾）
    hits: List[Tuple[int, str]] = []
    for lab in label_set:
        pos = cleaned.rfind(lab)
        if pos >= 0:
            hits.append((pos, lab))
    if hits:
        hits.sort(key=lambda x: (-x[0], -len(x[1])))
        return hits[0][1], "label_in_text", True, len(hits) > 1

    return (tail or None), "unmapped", False, False


def _synthetic_records(n: int = 12) -> List[Dict[str, Any]]:
    """离线自检用的专业样本：覆盖业务意图、通用交互、集外与多轮上下文。"""
    cases = [
        ("查明细。", "个人账户交易明细查询", "业务/FAQ意图", "个人账户交易明细查询"),
        ("我要转人工。", "人工服务", "通用交互意图", "人工服务"),
        ("卡丢了怎么办？", "借记卡人工挂失", "业务/FAQ意图", "未识别"),
        ("我想问一下余额", "个人账户余额查询", "业务/FAQ意图", "个人账户余额查询"),
        ("你好", "友好问候", "通用交互意图", "友好问候"),
        ("随便看看", "集外", "通用交互意图", "未识别"),
        ("帮我查开户行", "账户开户行查询", "业务/FAQ意图", "账户开户行查询"),
        ("不对", "否定", "通用交互意图", "否定"),
        ("投诉", "投诉建议", "通用交互意图", "投诉建议"),
        ("信用卡怎么还款", "信用卡业务", "业务/FAQ意图", "信用卡业务"),
        ("嗯", "肯定", "通用交互意图", "肯定"),
        ("挂失", "借记卡人工挂失", "业务/FAQ意图", "简短词"),
    ]
    labels = ", ".join(FALLBACK_LABELS)
    out: List[Dict[str, Any]] = []
    for i in range(min(n, len(cases))):
        utt, gold, stype, small = cases[i]
        prompt = (
            "你是银行语音导航系统的意图识别专家。请从候选意图树中选出唯一一个最匹配的意图，"
            "只输出意图名称，不要解释。\n\n"
            "【候选意图树】\n%s\n\n"
            "【历史对话】\n客服：您好，请问有什么可以帮您？\n\n"
            "【客户当前对话】\n%s\n\n"
            "【小模型识别意图】\n%s\n\n【输出】\n" % (labels, utt, small)
        )
        out.append({
            "id": "syn-intent-%04d" % i,
            "question": prompt,
            "choices": None,
            "answer": gold,
            "subject": stype,
            "meta": {
                "sample_id": "syn-intent-%04d" % i,
                "current_utterance": utt,
                "small_model_intent": small,
                "sample_type": stype,
                "human_intent_path": gold,
                "session_id": "syn-%d" % i,
                "turn_index": 1,
                "history_len": 1,
                "intent_tree_size": len(FALLBACK_LABELS),
                "prompt": prompt,
            },
        })
    return out


class IntentRoutingTask(BaseTask):
    """语音导航意图识别：从候选意图树中选唯一节点，人工标注为真实答案。"""

    name = "intent_routing"
    kind = "intent"
    default_protocol = "direct"
    default_split = "test"
    lang = "zh"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(cfg)
        # 每条样本共享同一份 498 节点候选树，prompt 前 200 字完全相同。
        # 通用 dedupe 按题面前缀去重会把 432 条压成 1 条，专业任务必须关闭。
        self.cfg["dedupe"] = False
        self.label_paths: List[str] = []
        self.labels: List[str] = list(FALLBACK_LABELS)

    # ------------------------------------------------------------------ #
    def prepare(self, data_root: str, *, allow_synthetic: bool = False,
                synthetic_n: int = 60) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        meta_path = os.path.join(data_root, self.name, "labels.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                self.label_paths = list(obj.get("paths") or [])
                self.labels = list(obj.get("labels") or []) or list(FALLBACK_LABELS)
                self._log("  [info] %s: 已加载 %d 个候选意图名称 / %d 条层级路径"
                          % (self.name, len(self.labels), len(self.label_paths)))
            except (OSError, json.JSONDecodeError) as exc:
                self._log("  [warn] %s: labels.json 解析失败，使用兜底标签：%s" % (self.name, exc))

        resolved = ds.resolve_dataset_path(self.name, self.split, data_root, self.cfg.get("path"))
        if resolved is None and allow_synthetic:
            records = _synthetic_records(synthetic_n)
            self.manifest = {
                "task": self.name, "split": self.split, "source": "<内置合成专业样本>",
                "n_eval": len(records), "n_fewshot": 0, "protocol": self.protocol,
                "mode": self.mode, "limit": int(self.cfg.get("limit") or 0),
                "n_labels": len(self.labels),
            }
            return records, []
        return super().prepare(data_root, allow_synthetic=allow_synthetic, synthetic_n=synthetic_n)

    # ------------------------------------------------------------------ #
    def build_messages(self, record: Dict[str, Any],
                       examples: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        # 数据集的 question 已是自包含 prompt（候选树 + 历史 + 当前句 + 小模型意图），
        # 不能再套通用基准的模板，否则会把业务 prompt 二次包装。
        return [{"role": "user", "content": str(record.get("question") or "")}]

    def mock_entry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "gold": record.get("answer"),
            "kind": "intent",
            "mock_answer": record.get("answer"),
            "labels": self.labels,
        }

    # ------------------------------------------------------------------ #
    def run_one(self, item: WorkItem, client: Any) -> Dict[str, Any]:
        row = self._row(item, client)
        rec = item.record
        meta = rec.get("meta") or {}
        gold = str(rec.get("answer") or "").strip()
        small = str(meta.get("small_model_intent") or "").strip()
        stype = str(rec.get("subject") or meta.get("sample_type") or "unknown")

        row.update({
            "gold": gold,
            "subject": stype,
            "sample_type": stype,
            "small_model_intent": small,
            "session_id": meta.get("session_id"),
            "turn_index": meta.get("turn_index"),
            "human_intent_path": meta.get("human_intent_path"),
            "history_len": meta.get("history_len"),
            "intent_tree_size": len(self.label_paths) or len(self.labels),
            "prompt_chars": len(item.messages[-1]["content"]) if item.messages else 0,
        })

        wall_start = now_iso()
        t0 = time.perf_counter()
        reply = client.chat(item.messages, max_tokens=self.max_tokens)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        wall_end = now_iso()

        pt = int(reply.usage.prompt_tokens or 0)
        ct = int(reply.usage.completion_tokens or 0)
        row.update({
            "latency_ms": reply.latency_ms,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": int(reply.usage.total_tokens or 0) or (pt + ct),
            "finish_reason": reply.finish_reason,
            "cached": bool(reply.cached),
            "request_wall_ms": round(wall_ms, 2),
            "wall_start": wall_start,
            "wall_end": wall_end,
            "response": truncate(reply.text, MAX_RESPONSE_CHARS),
        })
        if reply.error:
            row["error"] = reply.error
            return row

        pred, strategy, in_tree, ambiguous = extract_intent(reply.text, self.labels)
        strict = bool(pred and pred == gold)
        normalized = bool(pred and normalize_label(pred) == normalize_label(gold))
        small_ok = bool(small and small == gold)
        extra_prose = bool(reply.text and len(strip_reasoning(reply.text).strip()) > len(pred or "") + 12)

        row.update({
            "pred": pred,
            "strategy": strategy,
            "ambiguous": ambiguous,
            "parse_ok": pred is not None,
            "correct": strict,
            "extras": {
                "strict_correct": strict,
                "normalized_correct": normalized,
                "in_tree": in_tree,
                "hallucinated_intent": (pred is not None and not in_tree),
                "small_model_correct": small_ok,
                "paired_uplift": (1.0 if strict else 0.0) - (1.0 if small_ok else 0.0),
                "extra_prose": extra_prose,
                "length_truncated": reply.finish_reason == "length",
                "confusion": ("%s->%s" % (gold, pred or "<none>")) if not strict else None,
                "sample_type": stype,
                "small_model_intent": small,
            },
        })
        return row

    # ------------------------------------------------------------------ #
    def summarize(self, scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        from .metrics import mcnemar_exact_p, paired_bootstrap_diff
        from .telemetry import usage_by_group, usage_summary

        valid = [r for r in scored if not r.get("error")]
        n = len(valid)
        if not n:
            return {"n_valid": 0, "n_error": len(scored) - n}

        def ex(r: Dict[str, Any], key: str) -> bool:
            return bool((r.get("extras") or {}).get(key))

        a_flags = [bool(r.get("correct")) for r in valid]
        b_flags = [ex(r, "small_model_correct") for r in valid]
        llm_acc = sum(a_flags) / n
        small_acc = sum(b_flags) / n
        mc = mcnemar_exact_p(a_flags, b_flags)
        pb = paired_bootstrap_diff(a_flags, b_flags, n_boot=2000, seed=20261110)

        conf = Counter((r.get("gold"), r.get("pred")) for r in valid if not r.get("correct"))
        top_confusions = [
            {"gold": g, "pred": p, "n": c}
            for (g, p), c in conf.most_common(10)
        ]

        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in valid:
            groups[str(r.get("sample_type") or "unknown")].append(r)
        by_group = {}
        for g, rows in groups.items():
            by_group[g] = {
                "n": len(rows),
                "intent_accuracy": sum(bool(r.get("correct")) for r in rows) / len(rows),
                "small_model_accuracy": sum(ex(r, "small_model_correct") for r in rows) / len(rows),
                "usage": usage_summary(rows),
            }

        return {
            "n_valid": n,
            "n_error": len(scored) - n,
            "intent_accuracy": llm_acc,
            "strict_accuracy": llm_acc,
            "normalized_accuracy": sum(ex(r, "normalized_correct") for r in valid) / n,
            "small_model_accuracy": small_acc,
            "paired_uplift": llm_acc - small_acc,
            "uplift_ci95": [pb.get("lo"), pb.get("hi")],
            "mcnemar_p": mc.get("p_two_sided"),
            "mcnemar_discordant": mc.get("n_discordant"),
            "in_tree_rate": sum(ex(r, "in_tree") for r in valid) / n,
            "hallucinated_intent_rate": sum(ex(r, "hallucinated_intent") for r in valid) / n,
            "parse_fail_rate": sum(1 for r in valid if not r.get("parse_ok")) / n,
            "extra_prose_rate": sum(ex(r, "extra_prose") for r in valid) / n,
            "length_truncated_rate": sum(ex(r, "length_truncated") for r in valid) / n,
            "top_confusions": top_confusions,
            "by_group": by_group,
            "usage": usage_summary(valid),
            "n_candidate_labels": len(self.labels),
        }


def register(registry: Dict[str, Any]) -> None:
    """由 geval/tasks.py 在模块末尾调用，避免顶部循环导入。"""
    registry["intent_routing"] = IntentRoutingTask
