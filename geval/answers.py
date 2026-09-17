"""答案抽取：把模型自由文本解析成可判定的规范答案。

核心原则：抽取失败率必须被单独统计并上报。
否则「模型错」和「抽取器错」会混在一起，你会误判模型能力。
每个抽取结果都带 strategy 与 ambiguous 标记，便于回溯。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

CHOICE_LETTERS = "ABCDEFGHIJ"

_FULLWIDTH = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪａｂｃｄｅｆｇｈｉｊ０１２３４５６７８９（）［］｛｝",
    "ABCDEFGHIJabcdefghij0123456789()[]{}",
)

# 思维链残留：Qwen / DeepSeek 等 thinking 模型可能把推理过程混在正文里
_REASONING_BLOCKS = [
    re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I),
    re.compile(r"<reasoning>.*?</reasoning>", re.S | re.I),
    re.compile(r"^\s*思考[:：].*?(?=\n\s*\n)", re.S),
]

# 按优先级排列；同一策略内取「最后一次」匹配（CoT 的结论在末尾）
CHOICE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("answer_label", re.compile(r"答\s*案\s*[是为:：\-—={0,2}]*\s*[\(（\[]?\s*([A-Ja-j])\s*[\)）\]]?")),
    ("boxed", re.compile(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?([A-Ja-j])\s*\}?\s*\}")),
    ("bold_or_code", re.compile(r"(?:\*\*|__|\*|`)\s*([A-Ja-j])\s*(?:\*\*|__|\*|`)")),
    ("choose_verb", re.compile(r"(?:正确选项|正确答案|应选|选择|选|选项|option)\s*[是为:：\-]*\s*[\(（\[]?\s*([A-Ja-j])(?![A-Za-z])")),
    ("paren_letter", re.compile(r"[\(（]\s*([A-Ja-j])\s*[\)）]")),
    ("single_token", re.compile(r"^\s*[\(（\[]?\s*([A-Ja-j])\s*[\.。、,，\)）\]]?\s*$")),
    ("standalone_upper", re.compile(r"(?<![A-Za-z0-9])([A-J])(?![A-Za-z0-9])")),
]


# --------------------------------------------------------------------------- #
# 文本规范化
# --------------------------------------------------------------------------- #
def normalize_ws(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[ \t\u3000]+", " ", str(text).replace("\r", "")).strip()


def strip_reasoning(text: str) -> str:
    """剥离思维链块。若剥离后为空，则保留原文（避免把答案一起丢掉）。"""
    if not text:
        return ""
    cleaned = text
    for pat in _REASONING_BLOCKS:
        cleaned = pat.sub("", cleaned)
    return cleaned.strip() or text.strip()


def _norm_key(text: str) -> str:
    """用于宽松比对的归一化：去掉标点与空白、统一全角、小写。"""
    s = normalize_ws(text).translate(_FULLWIDTH).lower()
    s = re.sub(r"[\s\-_/\\]+", "", s)
    s = re.sub(r"[。，、；：！？,.;:!?\"'“”‘’()\[\]{}<>《》]", "", s)
    return s


# --------------------------------------------------------------------------- #
# 单选题
# --------------------------------------------------------------------------- #
def extract_choice(
    text: str,
    n_choices: int = 4,
    *,
    choices: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], str, bool]:
    """抽取选项字母。

    返回 (letter | None, strategy, ambiguous)。
    ambiguous=True 表示命中了低置信度策略且文本中存在多个候选字母。
    """
    letters = CHOICE_LETTERS[: max(1, int(n_choices))]
    allowed = set(letters)
    cleaned = strip_reasoning(text)
    if not cleaned:
        return None, "empty", False

    for name, pat in CHOICE_PATTERNS:
        matches = [m.group(1).upper() for m in pat.finditer(cleaned)]
        matches = [m for m in matches if m in allowed]
        if not matches:
            continue
        if name == "standalone_upper":
            # 低置信度策略：出现多个不同字母时取末尾（结论通常在后），并标记歧义
            uniq = set(matches)
            ambiguous = len(uniq) > 1
            return matches[-1], name, ambiguous
        return matches[-1], name, False

    # 兜底：模型直接复述了选项内容
    if choices:
        idx = match_choice_content(cleaned, choices)
        if idx is not None:
            return letters[idx], "content_match", True

    return None, "not_found", False


def match_choice_content(text: str, choices: Sequence[str]) -> Optional[int]:
    """在响应中匹配选项原文，取最长命中的那一个（避免短选项误命中）。"""
    hay = _norm_key(text)
    if not hay:
        return None
    best_idx: Optional[int] = None
    best_len = 0
    for i, choice in enumerate(choices):
        needle = _norm_key(choice)
        if len(needle) < 2:
            continue
        if needle in hay and len(needle) > best_len:
            best_idx, best_len = i, len(needle)
    return best_idx


def choice_candidates_in(text: str, n_choices: int = 4) -> List[str]:
    """返回文本中出现的所有候选字母（去重、保序），用于诊断提示混淆。"""
    letters = set(CHOICE_LETTERS[:n_choices])
    cleaned = strip_reasoning(text)
    seen: List[str] = []
    for m in CHOICE_PATTERNS[-1][1].finditer(cleaned):
        ch = m.group(1).upper()
        if ch in letters and ch not in seen:
            seen.append(ch)
    return seen


# --------------------------------------------------------------------------- #
# 数值（GSM8K 类）
# --------------------------------------------------------------------------- #
_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def canonical_number(raw: str) -> Optional[str]:
    """把各种写法归一成可比较的数值字符串；无法解析返回 None。"""
    if raw is None:
        return None
    s = normalize_ws(str(raw)).translate(_FULLWIDTH)
    if not s:
        return None
    s = s.strip("。.、,，;；:：)]}）]》\"'")
    s = s.replace("$", "").replace("¥", "").replace("￥", "").replace(" ", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    frac = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
    if frac:
        try:
            num = float(frac.group(1)) / float(frac.group(2))
        except ZeroDivisionError:
            return None
    else:
        s2 = s.replace(",", "")
        m = re.fullmatch(r"-?\d+(?:\.\d+)?", s2)
        if not m:
            return None
        num = float(s2)
    if pct:
        return f"{num}"
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.10f}".rstrip("0").rstrip(".")


def extract_number(text: str) -> Tuple[Optional[str], str]:
    """按优先级抽取最终数值，返回 (canonical | None, strategy)。"""
    cleaned = strip_reasoning(text)
    if not cleaned:
        return None, "empty"

    candidates: List[Tuple[str, str]] = []

    for m in re.finditer(r"####\s*([^\n]+)", cleaned):
        candidates.append((m.group(1), "hash_marker"))
    for m in re.finditer(r"\\boxed\s*\{\s*([^}]+)\}", cleaned):
        candidates.append((m.group(1), "boxed"))
    for m in re.finditer(
        rf"(?:答案|结果|值为|等于|总共|一共|因此|所以|总计)[^\d\-+{{}}]{{0,12}}({_NUM})", cleaned
    ):
        candidates.append((m.group(1), "answer_phrase"))

    nums = re.findall(_NUM, cleaned)
    if nums:
        candidates.append((nums[-1], "last_number"))

    for raw, strategy in candidates:
        canon = canonical_number(raw)
        if canon is not None:
            return canon, strategy
    return None, "not_found"


def numbers_equal(pred: Optional[str], gold: Optional[str], tol: float = 1e-4) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) <= tol * max(1.0, abs(float(gold)))
    except (TypeError, ValueError):
        return pred == gold


# --------------------------------------------------------------------------- #
# 短答案（BBH / 开放式）
# --------------------------------------------------------------------------- #
def extract_last_line(text: str) -> str:
    cleaned = strip_reasoning(text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_after_answer_marker(text: str) -> Optional[str]:
    m = re.findall(r"答\s*案\s*[是为:：\-—]*\s*(.+)", strip_reasoning(text))
    if not m:
        return None
    return m[-1].strip().split("\n")[0].strip()


def equal_loose(pred: Optional[str], gold: Optional[str]) -> bool:
    """宽松相等：归一化后互相包含即算命中（适配 BBH 的 yes/no、名称、短句）。"""
    if pred is None or gold is None:
        return False
    p, g = _norm_key(pred), _norm_key(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if len(g) <= 24 and g in p:
        return True
    if len(p) <= 24 and p in g:
        return True
    # 形如 (A) 与 A 的等价
    pl = re.findall(r"\(([a-j])\)", pred.lower())
    gl = re.findall(r"\(([a-j])\)", gold.lower())
    if pl and gl and pl[-1] == gl[-1]:
        return True
    return False


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Tuple[Optional[Any], Optional[str], str]:
    """从响应中提取第一个合法 JSON 对象。

    返回 (obj | None, raw_json_text | None, strategy)。
    先尝试整体解析（含剥离 markdown 围栏），再退化为花括号配平扫描。
    """
    if not text:
        return None, None, "empty"

    cleaned = strip_reasoning(text)
    attempts: List[Tuple[str, str]] = [("whole", cleaned.strip())]
    for i, fence in enumerate(_FENCE.findall(cleaned)):
        attempts.append((f"fence_{i}", fence.strip()))

    for name, candidate in attempts:
        if not candidate:
            continue
        try:
            return json.loads(candidate), candidate, name
        except json.JSONDecodeError:
            pass

    snippet = _first_balanced_object(cleaned)
    if snippet:
        # 常见的尾逗号问题单独修一次
        fixed = re.sub(r",\s*([}\]])", r"\1", snippet)
        for name, candidate in (("balanced", snippet), ("balanced_fixed", fixed)):
            try:
                return json.loads(candidate), candidate, name
            except json.JSONDecodeError:
                continue

    return None, None, "not_found"


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None
