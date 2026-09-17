"""评测任务：从数据集到逐样本判定结果的完整定义。

每个任务负责四件事：
  1. prepare        加载数据（含 few-shot 示例，且严格避免示例泄漏进测试集）
  2. build          构造 prompt 与 mock 旁路答案
  3. run_one        调用模型 → 抽取答案 → 判定对错 → 产出子指标
  4. summarize      任务级特殊指标（如 IFEval 的指令级通过率、JSON 的各项通过率）

新增一个基准只需要：定义子类 → 注册进 TASK_REGISTRY。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import datasets as ds
from . import prompts as pm
from .answers import (
    CHOICE_LETTERS,
    equal_loose,
    extract_choice,
    extract_json,
    extract_last_line,
    extract_number,
    numbers_equal,
    strip_reasoning,
)
from .ifeval import check_response, coverage_report
from .util import sha256_text, truncate

MAX_RESPONSE_CHARS = 4000


# --------------------------------------------------------------------------- #
@dataclass
class WorkItem:
    task: str
    record: Dict[str, Any]
    messages: List[Dict[str, str]]
    prompt_key: str
    mock_entry: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
class BaseTask:
    name = "base"
    kind = "text"
    default_split = "test"
    default_protocol = "fewshot"
    lang = "zh"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg: Dict[str, Any] = dict(cfg or {})
        self.protocol = str(self.cfg.get("protocol") or self.default_protocol)
        self.mode = str(self.cfg.get("mode") or "gen").lower()
        self.split = str(self.cfg.get("split") or self.default_split)
        self.num_fewshot = int(self.cfg.get("num_fewshot") or 0)
        self.max_tokens = int(self.cfg.get("max_tokens") or 1024)
        self.manifest: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if not self.cfg.get("_quiet"):
            print(msg, file=sys.stderr)

    def prepare(
        self, data_root: str, *, allow_synthetic: bool = False, synthetic_n: int = 60
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        explicit = self.cfg.get("path")
        records = ds.load_task_dataset(
            self.name,
            self.split,
            data_root,
            explicit_path=explicit,
            allow_synthetic=allow_synthetic,
            synthetic_n=synthetic_n,
        )
        if self.cfg.get("dedupe", True):
            before = len(records)
            records = ds.dedupe(records)
            if len(records) != before:
                self._log(f"  [info] {self.name}: 去重移除 {before - len(records)} 条重复题面")

        limit = int(self.cfg.get("limit") or 0)
        if limit > 0:
            records = ds.stratified_sample(
                records,
                limit,
                key=str(self.cfg.get("stratify_key") or "subject"),
                seed=int(self.cfg.get("seed") or 42),
            )

        examples: List[Dict[str, Any]] = []
        if self.num_fewshot > 0:
            examples = self._load_fewshot(data_root, records)
            if examples and examples[0].get("_leaked_from_eval"):
                leaked = {r["id"] for r in examples}
                records = [r for r in records if r["id"] not in leaked]
                self._log(
                    f"  [warn] {self.name}: 未找到 dev 切分，已改用测试集前 {len(leaked)} 条做 few-shot，"
                    f"并从评测集合中剔除以避免示例泄漏"
                )

        # 记录本任务的可复现指纹（数据来源、协议、抽样规模）
        self.manifest = {
            "task": self.name,
            "split": self.split,
            "source": str(explicit or resolve_hint(self.name, self.split, data_root)),
            "n_eval": len(records),
            "n_fewshot": len(examples),
            "protocol": self.protocol,
            "mode": self.mode,
            "limit": int(self.cfg.get("limit") or 0),
        }
        return records, examples

    def _load_fewshot(self, data_root: str, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        split = str(self.cfg.get("fewshot_split") or "dev")
        path = ds.resolve_dataset_path(self.name, split, data_root)
        if path:
            return ds.load_jsonl(path, source=str(split))[: self.num_fewshot]
        out = []
        for r in records[: self.num_fewshot]:
            item = dict(r)
            item["_leaked_from_eval"] = True
            out.append(item)
        return out

    # ------------------------------------------------------------------ #
    def build_messages(self, record: Dict[str, Any], examples: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        return pm.build_prompt(
            self.name,
            record,
            protocol=self.protocol,
            lang=self.lang,
            num_fewshot=self.num_fewshot,
            examples=examples,
            cfg=self.cfg,
        )

    def build(self, record: Dict[str, Any], examples: Sequence[Dict[str, Any]]) -> WorkItem:
        messages = self.build_messages(record, examples)
        return WorkItem(
            task=self.name,
            record=record,
            messages=messages,
            prompt_key=sha256_text(messages[-1]["content"] if messages else ""),
            mock_entry=self.mock_entry(record),
        )

    def mock_entry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {"gold": record.get("answer"), "kind": self.kind}

    # ------------------------------------------------------------------ #
    def run_one(self, item: WorkItem, client: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def summarize(self, scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _row(item: WorkItem, client: Any) -> Dict[str, Any]:
        return {
            "id": item.record.get("id"),
            "task": item.task,
            "subject": item.record.get("subject") or "",
            "gold": item.record.get("answer"),
            "pred": None,
            "correct": False,
            "parse_ok": False,
            "ambiguous": False,
            "strategy": "",
            "latency_ms": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "response": "",
            "error": None,
            "extras": {},
        }


# --------------------------------------------------------------------------- #
class MCQTask(BaseTask):
    kind = "mcq"
    default_protocol = "fewshot"
    lang = "zh"

    def mock_entry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        choices = record.get("choices") or []
        return {
            "gold": record.get("answer"),
            "kind": "mcq",
            "letters": CHOICE_LETTERS[: max(2, len(choices))],
        }

    def build_messages(self, record: Dict[str, Any], examples: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        # 续写似然打分必须让候选字母成为「答案位」上紧接的续写，
        # 因此固定用纯文本 + 「答案：」收口的形式。
        if self.mode == "loglikelihood":
            text = pm.build_mcq_completion_prompt(record, examples=examples, lang=self.lang)
            return [{"role": "user", "content": text}]
        return super().build_messages(record, examples)

    def run_one(self, item: WorkItem, client: Any) -> Dict[str, Any]:
        row = self._row(item, client)
        record = item.record
        choices = list(record.get("choices") or [])
        n_choices = max(2, len(choices))
        gold = str(record.get("answer") or "").strip().upper()
        row["gold"] = gold
        letters = CHOICE_LETTERS[:n_choices]

        if self.mode == "loglikelihood" and hasattr(client, "score_continuations"):
            done = self._score_loglikelihood(item, client, row, letters, gold)
            if done is not None:
                return done
            # 服务端不支持 echo/logprobs → 降级，避免整任务失败

        if self.mode == "logprob" and hasattr(client, "choice_logprobs_reply"):
            scores, reply = client.choice_logprobs_reply(item.messages, letters)
            row["latency_ms"] = reply.latency_ms
            row["prompt_tokens"] = reply.usage.prompt_tokens
            row["completion_tokens"] = reply.usage.completion_tokens
            if reply.error:
                row["error"] = reply.error
                return row
            if scores:
                pred = max(scores.items(), key=lambda kv: kv[1])[0]
                row.update(
                    {
                        "pred": pred,
                        "parse_ok": True,
                        "strategy": "logprob_argmax",
                        "correct": pred == gold,
                        "extras": {"logprob_margin": _margin(scores), "top_choice_logprobs": _round_scores(scores)},
                    }
                )
                return row
            # 服务端不支持 logprobs → 自动降级为生成式，避免整任务失败

        reply = client.chat(item.messages, max_tokens=self.max_tokens)
        return self._score_generative(row, record, choices, n_choices, gold, reply)

    def _score_loglikelihood(
        self,
        item: WorkItem,
        client: Any,
        row: Dict[str, Any],
        letters: str,
        gold: str,
    ) -> Optional[Dict[str, Any]]:
        """lm-eval 口径的 acc / acc_norm 续写似然打分。"""
        prompt_text = item.messages[-1]["content"]
        res = client.score_continuations(prompt_text, list(letters))
        row["latency_ms"] = res.get("latency_ms")
        row["prompt_tokens"] = res.get("prompt_tokens", 0)
        row["completion_tokens"] = res.get("completion_tokens", 0)

        scores = res.get("scores") or {}
        if not scores:
            return None

        # acc：logprob 之和最大者
        by_acc = max(scores.items(), key=lambda kv: kv[1]["logprob"])[0]
        # acc_norm：按续写 token 数归一化后最大者（lm-eval 的主指标）
        by_norm = max(scores.items(), key=lambda kv: kv[1]["logprob"] / max(1.0, kv[1]["n_tokens"]))[0]

        primary = str(self.cfg.get("primary") or "acc_norm").lower()
        pred = by_acc if primary == "acc" else by_norm
        ranked = sorted(scores.items(), key=lambda kv: kv[1]["logprob"] / max(1.0, kv[1]["n_tokens"]), reverse=True)
        margin = None
        if len(ranked) >= 2:
            margin = (ranked[0][1]["logprob"] / max(1.0, ranked[0][1]["n_tokens"])) - (
                ranked[1][1]["logprob"] / max(1.0, ranked[1][1]["n_tokens"])
            )

        row.update(
            {
                "pred": pred,
                "parse_ok": True,
                "strategy": f"loglikelihood_{primary}",
                "correct": pred == gold,
                "extras": {
                    "acc_choice": by_acc,
                    "acc_norm_choice": by_norm,
                    "acc_norm_margin": None if margin is None else round(float(margin), 4),
                    "acc_correct": by_acc == gold,
                    "acc_norm_correct": by_norm == gold,
                    "n_requests": res.get("n_requests", 0),
                    "choice_logprobs": {k: round(float(v["logprob"]), 3) for k, v in scores.items()},
                },
            }
        )
        return row

    def _score_generative(self, row, record, choices, n_choices, gold, reply) -> Dict[str, Any]:
        row["latency_ms"] = reply.latency_ms
        row["prompt_tokens"] = reply.usage.prompt_tokens
        row["completion_tokens"] = reply.usage.completion_tokens
        row["response"] = truncate(reply.text, MAX_RESPONSE_CHARS)
        if reply.error:
            row["error"] = reply.error
            return row

        pred, strategy, ambiguous = extract_choice(reply.text, n_choices, choices=choices)
        row.update(
            {
                "pred": pred,
                "strategy": strategy,
                "ambiguous": ambiguous,
                "parse_ok": pred is not None,
                "correct": (pred is not None and pred == gold),
            }
        )
        if pred is not None and pred != gold:
            row["extras"]["confusion"] = f"{gold}->{pred}"
        return row


def _margin(scores: Dict[str, float]) -> Optional[float]:
    if len(scores) < 2:
        return None
    vals = sorted(scores.values(), reverse=True)
    return vals[0] - vals[1]


def _round_scores(scores: Dict[str, float]) -> Dict[str, float]:
    return {k: round(float(v), 3) for k, v in scores.items()}


# --------------------------------------------------------------------------- #
class NumericTask(BaseTask):
    kind = "numeric"
    default_protocol = "direct"
    lang = "zh"

    def mock_entry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {"gold": record.get("answer"), "kind": "numeric"}

    def run_one(self, item: WorkItem, client: Any) -> Dict[str, Any]:
        row = self._row(item, client)
        gold = item.record.get("answer")
        row["gold"] = gold
        reply = client.chat(item.messages, max_tokens=self.max_tokens)
        row["latency_ms"] = reply.latency_ms
        row["prompt_tokens"] = reply.usage.prompt_tokens
        row["completion_tokens"] = reply.usage.completion_tokens
        row["response"] = truncate(reply.text, MAX_RESPONSE_CHARS)
        if reply.error:
            row["error"] = reply.error
            return row

        pred, strategy = extract_number(reply.text)
        tol = float(self.cfg.get("tol") or 1e-4)
        row.update(
            {
                "pred": pred,
                "strategy": strategy,
                "parse_ok": pred is not None,
                "correct": numbers_equal(pred, gold, tol),
            }
        )
        return row


# --------------------------------------------------------------------------- #
class BBHTask(BaseTask):
    name = "bbh"
    kind = "text"
    default_protocol = "answer_line"
    lang = "zh"

    _PATTERNS = (
        r"所以答案是\s*(.+)",
        r"答\s*案\s*[是为:：]\s*(.+)",
        r"final\s+answer\s*[:\s]+\s*(.+)",
    )

    def mock_entry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {"gold": record.get("answer"), "kind": "text"}

    def run_one(self, item: WorkItem, client: Any) -> Dict[str, Any]:
        import re

        row = self._row(item, client)
        gold = item.record.get("answer")
        row["gold"] = gold
        reply = client.chat(item.messages, max_tokens=self.max_tokens)
        row["latency_ms"] = reply.latency_ms
        row["prompt_tokens"] = reply.usage.prompt_tokens
        row["completion_tokens"] = reply.usage.completion_tokens
        row["response"] = truncate(reply.text, MAX_RESPONSE_CHARS)
        if reply.error:
            row["error"] = reply.error
            return row

        pred, strategy = self._extract(reply.text)
        row.update(
            {
                "pred": pred,
                "strategy": strategy,
                "parse_ok": pred is not None,
                "correct": equal_loose(pred, gold),
            }
        )
        return row

    def _extract(self, text: str) -> Tuple[Optional[str], str]:
        import re

        cleaned = strip_reasoning(text or "")
        for i, pat in enumerate(self._PATTERNS):
            hits = re.findall(pat, cleaned, re.I)
            if hits:
                cand = hits[-1].split("\n")[0].strip().strip("。.、,，;；:：\"'()（）[]")
                if cand:
                    return cand, f"pattern_{i}"
        last = extract_last_line(cleaned)
        last = last.strip().strip("。.、,，;；:：\"'()（）[]")
        return (last or None), "last_line"


# --------------------------------------------------------------------------- #
class JSONSchemaTask(BaseTask):
    """意图识别 + 槽位抽取的结构化输出评测。

    这是与真实业务最贴近的一项，因此把一个「对/错」拆成四个可诊断的层级：
      json_valid      —— 输出能不能被解析成 JSON
      schema_ok       —— 是否符合给定 schema（必填字段、类型）
      intent_correct  —— 意图分类是否正确
      slots_exact     —— 槽位是否完全一致
    生产中的失败绝大多数卡在第一、二层，而不是第三层。
    """

    kind = "json"
    default_protocol = "direct"
    lang = "zh"
    name = "json_schema"

    def mock_entry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {"gold": record.get("answer"), "kind": "json", "mock_answer": record.get("answer")}

    def prepare(self, data_root: str, **kw: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """加载同目录下的 schema.json：意图枚举 / 槽位枚举 / JSON Schema 随数据走。

        这样换一套业务意图时只改数据目录，不需要动配置文件。
        """
        import json
        import os

        meta_path = os.path.join(data_root, self.name, "schema.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    extra = json.load(f)
                for key in ("schema", "intent_labels", "slot_names"):
                    if key in extra and key not in self.cfg:
                        self.cfg[key] = extra[key]
                self._log(f"  [info] {self.name}: 已从 {meta_path} 加载 schema 与枚举定义")
            except (OSError, json.JSONDecodeError) as exc:
                self._log(f"  [warn] {self.name}: {meta_path} 解析失败，已忽略：{exc}")
        return super().prepare(data_root, **kw)

    def run_one(self, item: WorkItem, client: Any) -> Dict[str, Any]:
        row = self._row(item, client)
        gold_raw = item.record.get("answer")
        try:
            gold_obj = json.loads(gold_raw) if isinstance(gold_raw, str) else gold_raw
        except (TypeError, json.JSONDecodeError):
            gold_obj = None
        row["gold"] = truncate(gold_raw, 600)

        reply = client.chat(item.messages, max_tokens=self.max_tokens)
        row["latency_ms"] = reply.latency_ms
        row["prompt_tokens"] = reply.usage.prompt_tokens
        row["completion_tokens"] = reply.usage.completion_tokens
        row["response"] = truncate(reply.text, MAX_RESPONSE_CHARS)
        if reply.error:
            row["error"] = reply.error
            return row

        pred_obj, _, strategy = extract_json(reply.text)
        schema = self.cfg.get("schema") or pm.DEFAULT_INTENT_SCHEMA
        json_valid = pred_obj is not None
        schema_ok = bool(json_valid and validate_schema(pred_obj, schema))
        extra_text = bool(json_valid and _has_extra_prose(reply.text))

        intent_ok = bool(
            json_valid
            and isinstance(pred_obj, dict)
            and gold_obj
            and str(pred_obj.get("intent")) == str(gold_obj.get("intent"))
        )
        slots_ok = bool(
            json_valid
            and isinstance(pred_obj, dict)
            and gold_obj
            and _slots_equal(pred_obj.get("slots"), gold_obj.get("slots"))
        )

        # strict_json：夹带解释性文字也判为失败。
        # 生产上游解析器通常只接受纯 JSON，这里默认按严格口径统计。
        strict = bool(self.cfg.get("strict_json", True))
        correct = bool(json_valid and schema_ok and intent_ok and slots_ok and (not extra_text or not strict))

        row.update(
            {
                "pred": truncate(json.dumps(pred_obj, ensure_ascii=False) if json_valid else reply.text, 600),
                "strategy": strategy,
                "parse_ok": json_valid,
                "correct": correct,
                "extras": {
                    "json_valid": json_valid,
                    "schema_ok": schema_ok,
                    "intent_correct": intent_ok,
                    "slots_exact": slots_ok,
                    "extra_text": extra_text,
                },
            }
        )
        return row

    def summarize(self, scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(scored)
        if not n:
            return {}
        return {
            "json_valid_rate": sum(1 for r in scored if r["extras"].get("json_valid")) / n,
            "schema_ok_rate": sum(1 for r in scored if r["extras"].get("schema_ok")) / n,
            "intent_accuracy": sum(1 for r in scored if r["extras"].get("intent_correct")) / n,
            "slots_exact_rate": sum(1 for r in scored if r["extras"].get("slots_exact")) / n,
            "extra_prose_rate": sum(1 for r in scored if r["extras"].get("extra_text")) / n,
        }


def _slots_equal(pred: Any, gold: Any) -> bool:
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return False
    norm = lambda d: {str(k): str(v).strip() for k, v in d.items()}  # noqa: E731
    return norm(pred) == norm(gold)


def _has_extra_prose(text: str) -> bool:
    stripped = (text or "").strip()
    return not (stripped.startswith("{") and stripped.endswith("}"))


def validate_schema(obj: Any, schema: Dict[str, Any], path: str = "$") -> bool:
    """极简 JSON Schema 校验：支持 type / required / properties / enum / items。

    刻意不引入 jsonschema 依赖；意图识别场景下的 schema 足够简单。
    """
    if not isinstance(schema, dict):
        return True

    stype = schema.get("type")
    if stype == "object":
        if not isinstance(obj, dict):
            return False
        for key in schema.get("required") or []:
            if key not in obj:
                return False
        props = schema.get("properties") or {}
        for key, subschema in props.items():
            if key in obj and not validate_schema(obj[key], subschema, f"{path}.{key}"):
                return False
        if schema.get("additionalProperties") is False:
            extra = set(obj) - set(props)
            if extra:
                return False
        return True
    if stype == "array":
        if not isinstance(obj, list):
            return False
        item_schema = schema.get("items") or {}
        return all(validate_schema(x, item_schema, f"{path}[]") for x in obj)
    if stype == "string":
        if not isinstance(obj, str):
            return False
    elif stype == "integer":
        if not isinstance(obj, int) or isinstance(obj, bool):
            return False
    elif stype == "number":
        if not isinstance(obj, (int, float)) or isinstance(obj, bool):
            return False
    elif stype == "boolean":
        if not isinstance(obj, bool):
            return False
    elif stype == "null":
        if obj is not None:
            return False

    if "enum" in schema and obj not in (schema["enum"] or []):
        return False
    return True


# --------------------------------------------------------------------------- #
class IFEvalTask(BaseTask):
    name = "ifeval"
    kind = "ifeval"
    default_protocol = "direct"
    lang = "zh"

    def mock_entry(self, record: Dict[str, Any]) -> Dict[str, Any]:
        meta = record.get("meta") or {}
        ref = meta.get("reference_response") or ""
        return {"gold": ref, "kind": "ifeval", "mock_answer": ref}

    def run_one(self, item: WorkItem, client: Any) -> Dict[str, Any]:
        row = self._row(item, client)
        meta = item.record.get("meta") or {}
        ids = list(meta.get("instruction_id_list") or [])
        kwargs = list(meta.get("kwargs") or [])
        row["gold"] = ",".join(ids)
        row["subject"] = item.record.get("subject") or (ids[0].split(":")[0] if ids else "")

        reply = client.chat(item.messages, max_tokens=self.max_tokens)
        row["latency_ms"] = reply.latency_ms
        row["prompt_tokens"] = reply.usage.prompt_tokens
        row["completion_tokens"] = reply.usage.completion_tokens
        row["response"] = truncate(reply.text, MAX_RESPONSE_CHARS)
        if reply.error:
            row["error"] = reply.error
            return row

        res = check_response(reply.text, ids, kwargs)
        row.update(
            {
                "pred": "pass" if res["prompt_pass"] else "fail",
                "strategy": "rule_check",
                "parse_ok": res["n_supported"] > 0,
                "correct": bool(res["prompt_pass"]),
                "extras": {
                    "instruction_pass": res["instruction_pass"] if res["instruction_pass"] is not None else None,
                    "n_supported": res["n_supported"],
                    "n_unsupported": res["n_unsupported"],
                    "empty_response": not (reply.text or "").strip(),
                },
            }
        )
        row["instruction_details"] = res["details"]
        return row

    def summarize(self, scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(scored)
        if not n:
            return {}
        all_ids: List[str] = []
        for r in scored:
            for d in r.get("instruction_details") or []:
                all_ids.append(str(d.get("instruction_id")))
        ins = [r["extras"].get("instruction_pass") for r in scored]
        ins = [x for x in ins if x is not None]
        out: Dict[str, Any] = {
            "prompt_level_accuracy": sum(1 for r in scored if r.get("correct")) / n,
            "instruction_level_accuracy": (sum(ins) / len(ins)) if ins else None,
        }
        out.update(coverage_report(all_ids))
        return out


# --------------------------------------------------------------------------- #
TASK_REGISTRY: Dict[str, Any] = {
    "ceval": type("CEvalTask", (MCQTask,), {"name": "ceval", "lang": "zh"}),
    "cmmlu": type("CMMLUTask", (MCQTask,), {"name": "cmmlu", "lang": "zh"}),
    "mmlu": type("MMLUTask", (MCQTask,), {"name": "mmlu", "lang": "en", "default_split": "test"}),
    "mmlu_pro": type("MMLUProTask", (MCQTask,), {"name": "mmlu_pro", "lang": "en", "default_split": "test"}),
    "arc": type("ARCTask", (MCQTask,), {"name": "arc", "lang": "en", "default_split": "test"}),
    "agieval": type("AGIEvalTask", (MCQTask,), {"name": "agieval", "lang": "zh", "default_split": "test"}),
    "gsm8k": type("GSM8KTask", (NumericTask,), {"name": "gsm8k", "lang": "zh", "default_split": "test"}),
    "math": type("MathTask", (NumericTask,), {"name": "math", "lang": "zh", "default_split": "test"}),
    "bbh": BBHTask,
    "json_schema": JSONSchemaTask,
    "ifeval": IFEvalTask,
    "intent_json": type("IntentJsonTask", (JSONSchemaTask,), {"name": "intent_json"}),
}

# 中文基准与英文基准的默认 few-shot 数量，贴近公开榜单设置
DEFAULT_FEWSHOT = {
    "ceval": 5,
    "cmmlu": 5,
    "mmlu": 5,
    "mmlu_pro": 5,
    "arc": 0,
    "agieval": 0,
    "gsm8k": 4,
    "math": 4,
    "bbh": 0,
    "json_schema": 0,
    "ifeval": 0,
}


def resolve_hint(name: str, split: str, data_root: str) -> str:
    """给出一句人可读的数据来源提示，便于回看结果时确认用的是哪份数据。"""
    import os

    path = ds.resolve_dataset_path(name, split, data_root)
    if path:
        return os.path.relpath(path, data_root) if data_root else path
    return f"<{name}/{split}.jsonl 缺失，使用内置合成数据>"


def get_task(name: str, cfg: Optional[Dict[str, Any]] = None) -> BaseTask:
    cls = TASK_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"未知任务 {name!r}；可用任务：{', '.join(sorted(TASK_REGISTRY))}")
    merged = dict(cfg or {})
    if "num_fewshot" not in merged:
        merged["num_fewshot"] = DEFAULT_FEWSHOT.get(name, 0)
    return cls(merged)


def task_kinds() -> Dict[str, str]:
    return {name: getattr(cls, "kind", "?") for name, cls in sorted(TASK_REGISTRY.items())}
