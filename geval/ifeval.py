"""IFEval 指令遵循校验器（规则化，无需裁判模型）。

为什么选 IFEval 作为「指令遵循」维度的主基准：
  1. 全部由确定性规则判定，不依赖 LLM-as-judge；
  2. 内网离线环境无需额外模型，也没有裁判模型带来的偏好偏差；
  3. 与意图识别高度相关：真实系统的失败大多是「格式没按约定输出」「约束没守住」。

实现说明：
  - 覆盖官方指令类型的一个实用子集；未实现的类型返回 None，从统计中排除，
    并在报告里单列「未覆盖指令类型」，避免把「不支持」伪装成「不通过」。
  - 中文适配：number_words 的计数方式为「CJK 字符逐个计数 + 非 CJK 按空白分词」，
    这是对官方英文实现的必要扩展，否则中文文本的词数恒为 1。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answers import extract_json

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


# --------------------------------------------------------------------------- #
# 文本工具
# --------------------------------------------------------------------------- #
def count_words(text: str) -> int:
    """中文适配的词数计数。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    latin = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text))
    return cjk + latin


def count_sentences(text: str) -> int:
    """近似句数：按中英文句末标点切分，过滤空段。"""
    if not text:
        return 0
    parts = re.split(r"(?<=[.!?。！？；;])\s*", text.strip())
    return len([p for p in parts if p.strip()])


def count_paragraphs(text: str) -> int:
    """段落 = 至少含一个非空白字符、且以换行分隔的块。"""
    if not text:
        return 0
    blocks = re.split(r"\n\s*\n", text.strip())
    return len([b for b in blocks if b.strip()])


def _relation_ok(actual: int, relation: str, target: int) -> bool:
    rel = (relation or "at least").strip().lower()
    if rel in ("at least", "more than or equal to", ">="):
        return actual >= target
    if rel in ("at most", "less than or equal to", "<="):
        return actual <= target
    if rel in ("less than", "<"):
        return actual < target
    if rel in ("more than", ">"):
        return actual > target
    if rel in ("equal to", "="):
        return actual == target
    return actual >= target


# --------------------------------------------------------------------------- #
# 各指令类型校验
# --------------------------------------------------------------------------- #
def _c_punctuation_no_comma(resp: str, kw: Dict[str, Any]) -> bool:
    return ("," not in resp) and ("，" not in resp)


def _c_number_words(resp: str, kw: Dict[str, Any]) -> bool:
    return _relation_ok(count_words(resp), kw.get("relation", "at least"), int(kw.get("num_words", 0)))


def _c_number_sentences(resp: str, kw: Dict[str, Any]) -> bool:
    return _relation_ok(count_sentences(resp), kw.get("relation", "at least"), int(kw.get("num_sentences", 0)))


def _c_number_paragraphs(resp: str, kw: Dict[str, Any]) -> bool:
    return count_paragraphs(resp) == int(kw.get("num_paragraphs", 0))


def _c_nth_paragraph_first_word(resp: str, kw: Dict[str, Any]) -> bool:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", resp.strip()) if b.strip()]
    idx = int(kw.get("nth_paragraph", 1)) - 1
    if idx < 0 or idx >= len(blocks):
        return False
    words = blocks[idx].split()
    return bool(words) and words[0].strip().lower() == str(kw.get("first_word", "")).strip().lower()


def _c_number_placeholders(resp: str, kw: Dict[str, Any]) -> bool:
    return resp.count("[") >= int(kw.get("num_placeholders", 0))


def _c_postscript(resp: str, kw: Dict[str, Any]) -> bool:
    marker = str(kw.get("postscript_marker", "P.S."))
    tail = resp.strip()[-40:]
    return marker in tail or marker.replace(".", "") in tail


def _c_number_bullet_lists(resp: str, kw: Dict[str, Any]) -> bool:
    bullets = re.findall(r"^\s*[\*\-]\s+", resp, re.M)
    return len(bullets) == int(kw.get("num_bullets", 0))


def _c_number_highlighted_sections(resp: str, kw: Dict[str, Any]) -> bool:
    hits = re.findall(r"\*[^\*\n]+\*", resp)
    return len(hits) >= int(kw.get("num_highlights", 0))


def _c_multiple_sections(resp: str, kw: Dict[str, Any]) -> bool:
    spliter = str(kw.get("section_spliter", "==="))
    sections = [s for s in resp.split(spliter)]
    return len(sections) == int(kw.get("num_sections", 0))


def _c_json_format(resp: str, kw: Dict[str, Any]) -> bool:
    obj, _, _ = extract_json(resp)
    return obj is not None


def _c_title(resp: str, kw: Dict[str, Any]) -> bool:
    return bool(re.search(r"<<[^<>]+>>", resp))


def _c_constrained_response(resp: str, kw: Dict[str, Any]) -> bool:
    allowed = ("my answer is yes", "my answer is no", "my answer is maybe")
    return resp.strip().lower() in allowed


def _c_two_responses(resp: str, kw: Dict[str, Any]) -> bool:
    if "******" not in resp:
        return False
    parts = [p.strip() for p in resp.split("******")]
    return len([p for p in parts if p]) == 2


def _c_repeat_prompt(resp: str, kw: Dict[str, Any]) -> bool:
    target = str(kw.get("prompt_to_repeat", "")).strip()
    return bool(target) and resp.strip().startswith(target)


def _c_keywords_existence(resp: str, kw: Dict[str, Any]) -> bool:
    return all(str(k) in resp for k in (kw.get("keywords") or []))


def _c_keywords_frequency(resp: str, kw: Dict[str, Any]) -> bool:
    kwd = str(kw.get("keyword", ""))
    return _relation_ok(resp.count(kwd), kw.get("relation", "at least"), int(kw.get("frequency", 0)))


def _c_keywords_forbidden(resp: str, kw: Dict[str, Any]) -> bool:
    low = resp.lower()
    return all(str(w).lower() not in low for w in (kw.get("forbidden_words") or []))


def _c_letter_frequency(resp: str, kw: Dict[str, Any]) -> bool:
    letter = str(kw.get("letter", "")).lower()
    return _relation_ok(
        resp.lower().count(letter), kw.get("let_relation", "at least"), int(kw.get("let_frequency", 0))
    )


def _c_end_checker(resp: str, kw: Dict[str, Any]) -> bool:
    return resp.strip().endswith(str(kw.get("end_phrase", "")))


def _c_start_checker(resp: str, kw: Dict[str, Any]) -> bool:
    return resp.strip().startswith(str(kw.get("start_phrase", "")))


def _c_quotation(resp: str, kw: Dict[str, Any]) -> bool:
    body = resp.strip()
    return len(body) >= 2 and body[0] == '"' and body[-1] == '"'


def _c_english_capital(resp: str, kw: Dict[str, Any]) -> bool:
    letters = re.findall(r"[A-Za-z]", resp)
    return bool(letters) and all(c.isupper() for c in letters)


def _c_english_lowercase(resp: str, kw: Dict[str, Any]) -> bool:
    letters = re.findall(r"[A-Za-z]", resp)
    return bool(letters) and all(c.islower() for c in letters)


def _c_capital_word_frequency(resp: str, kw: Dict[str, Any]) -> bool:
    caps = [w for w in re.findall(r"\b[A-Z]{2,}\b", resp)]
    return _relation_ok(len(caps), kw.get("capital_relation", "at least"), int(kw.get("capital_frequency", 0)))


CHECKERS: Dict[str, Any] = {
    "punctuation:no_comma": _c_punctuation_no_comma,
    "length_constraints:number_words": _c_number_words,
    "length_constraints:number_sentences": _c_number_sentences,
    "length_constraints:number_paragraphs": _c_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _c_nth_paragraph_first_word,
    "detectable_content:number_placeholders": _c_number_placeholders,
    "detectable_content:postscript": _c_postscript,
    "detectable_format:number_bullet_lists": _c_number_bullet_lists,
    "detectable_format:number_highlighted_sections": _c_number_highlighted_sections,
    "detectable_format:multiple_sections": _c_multiple_sections,
    "detectable_format:json_format": _c_json_format,
    "detectable_format:title": _c_title,
    "detectable_format:constrained_response": _c_constrained_response,
    "combination:two_responses": _c_two_responses,
    "combination:repeat_prompt": _c_repeat_prompt,
    "keywords:existence": _c_keywords_existence,
    "keywords:frequency": _c_keywords_frequency,
    "keywords:forbidden_words": _c_keywords_forbidden,
    "keywords:letter_frequency": _c_letter_frequency,
    "startend:end_checker": _c_end_checker,
    "startend:start_checker": _c_start_checker,
    "startend:quotation": _c_quotation,
    "change_case:english_capital": _c_english_capital,
    "change_case:english_lowercase": _c_english_lowercase,
    "change_case:capital_word_frequency": _c_capital_word_frequency,
}

# 明确不支持的（需要语言识别等外部依赖），统计时单列
UNSUPPORTED = {"language:response_language"}


def check_response(
    response: str,
    instruction_id_list: Sequence[str],
    kwargs_list: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """逐条校验，返回指令级与样本级结果。

    instruction_pass: 已覆盖指令的通过率（None 表示该样本全部未覆盖）
    prompt_pass:      该样本的已覆盖指令是否全部通过（IFEval 官方 prompt-level 指标）
    """
    resp = response or ""
    details: List[Dict[str, Any]] = []
    supported: List[bool] = []
    unsupported: List[str] = []

    for i, iid in enumerate(instruction_id_list or []):
        kw = dict(kwargs_list[i]) if i < len(kwargs_list or []) and isinstance(kwargs_list[i], dict) else {}
        checker = CHECKERS.get(iid)
        if checker is None:
            unsupported.append(iid)
            details.append({"instruction_id": iid, "passed": None, "reason": "未实现的指令类型"})
            continue
        try:
            ok = bool(checker(resp, kw))
            err = ""
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        supported.append(ok)
        details.append({"instruction_id": iid, "passed": ok, "kwargs": kw, "error": err})

    ins_pass = (sum(1 for x in supported if x) / len(supported)) if supported else None
    return {
        "instruction_pass": ins_pass,
        "prompt_pass": (all(supported) if supported else None),
        "n_supported": len(supported),
        "n_unsupported": len(unsupported),
        "unsupported_ids": unsupported,
        "details": details,
    }


def coverage_report(all_ids: Sequence[str]) -> Dict[str, int]:
    """统计本次评测涉及的指令类型覆盖率（用于报告中的诚实披露）。"""
    from collections import Counter

    cnt = Counter(all_ids)
    covered = {k: v for k, v in cnt.items() if k in CHECKERS}
    missing = {k: v for k, v in cnt.items() if k not in CHECKERS}
    return {
        "n_instruction_types": len(cnt),
        "n_covered_types": len(covered),
        "n_missing_types": len(missing),
        "missing": missing,
    }
