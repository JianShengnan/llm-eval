"""Prompt 构建：三种协议，可在评测时按任务切换。

协议对比（这是第一阶段评测要回答的问题之一）：
  direct       只回答选项字母，输出最短 → 小模型最稳、时延最低，但无法体现推理能力
  answer_line  允许推理，最后一行以「答案：X」收口 → 32B 类模型通常最优
  fewshot      按基准原始格式给 k 个示例 → 与公开榜单数字最可比

实践中常见结论：7B/8B 用 direct 反而比 answer_line 准确率高几个点
（推理过程容易把结论带偏），而 32B 用 answer_line 更优。
所以协议必须做成可切换的实验变量，而不是写死。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .answers import CHOICE_LETTERS

MCQ_ZH_OPTIONS = "选项：\n{options}"
MCQ_EN_OPTIONS = "Options:\n{options}"

DIRECT_ZH = "请只输出正确选项的字母（A/B/C/D…），不要输出任何解释或其他文字。"
DIRECT_EN = "Answer with only the letter of the correct option. Do not add any explanation."

ANSWER_LINE_ZH = (
    "请逐步分析，并在最后一行严格以「答案：X」的格式给出选项字母（X 为选项字母）。"
)
ANSWER_LINE_EN = (
    "Think step by step, then put your final answer on the last line in the exact "
    "format 'Answer: X' where X is the option letter."
)


# --------------------------------------------------------------------------- #
# 选项渲染
# --------------------------------------------------------------------------- #
def render_options(choices: Sequence[str], *, lang: str = "zh") -> str:
    lines = [f"{CHOICE_LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(choices)]
    return "\n".join(lines)


def _subject_phrase(subject: str, lang: str) -> str:
    if not subject:
        return ""
    return f"（{subject}）" if lang == "zh" else f" ({subject})"


# --------------------------------------------------------------------------- #
# 单选题
# --------------------------------------------------------------------------- #
def build_mcq(
    record: Dict[str, Any],
    *,
    examples: Optional[Sequence[Dict[str, Any]]] = None,
    protocol: str = "fewshot",
    lang: str = "zh",
) -> List[Dict[str, str]]:
    choices = list(record.get("choices") or [])
    subject = str(record.get("subject") or "")
    header_zh = f"以下是一道{subject.replace('_', ' ')}单项选择题。" if subject else "以下是一道单项选择题。"
    header_en = f"The following is a multiple-choice question on {subject.replace('_', ' ')}." if subject else "The following is a multiple-choice question."

    protocol = (protocol or "fewshot").lower()

    if protocol == "direct":
        instr = DIRECT_ZH if lang == "zh" else DIRECT_EN
        user = (
            f"{header_zh}\n\n{record['question']}\n\n{render_options(choices)}\n\n{instr}"
            if lang == "zh"
            else f"{header_en}\n\n{record['question']}\n\n{render_options(choices)}\n\n{instr}"
        )
        return [{"role": "user", "content": user}]

    if protocol == "answer_line":
        instr = ANSWER_LINE_ZH if lang == "zh" else ANSWER_LINE_EN
        user = (
            f"{header_zh}\n\n{record['question']}\n\n{render_options(choices)}\n\n{instr}"
            if lang == "zh"
            else f"{header_en}\n\n{record['question']}\n\n{render_options(choices)}\n\n{instr}"
        )
        return [{"role": "user", "content": user}]

    # fewshot：严格按照基准原始格式给示例，模型续写「答案：」
    blocks: List[str] = []
    for ex in examples or []:
        ex_choices = list(ex.get("choices") or [])
        gold = str(ex.get("answer") or "").strip().upper()
        if not ex_choices or gold not in CHOICE_LETTERS[: len(ex_choices)]:
            continue
        if lang == "zh":
            blocks.append(
                f"问题：{ex['question']}\n{render_options(ex_choices)}\n答案：{gold}"
            )
        else:
            blocks.append(
                f"Question: {ex['question']}\n{render_options(ex_choices, lang='en')}\nAnswer: {gold}"
            )

    lead = (
        f"以下是一道{subject.replace('_', ' ')}单项选择题，请作答。"
        if lang == "zh" and subject
        else ("以下是一道单项选择题，请作答。" if lang == "zh" else "Answer the following multiple-choice question.")
    )
    body = f"问题：{record['question']}\n{render_options(choices)}\n答案：" if lang == "zh" else (
        f"Question: {record['question']}\n{render_options(choices, lang='en')}\nAnswer:"
    )
    content = lead + "\n\n" + ("\n\n".join(blocks) + "\n\n" if blocks else "") + body
    return [{"role": "user", "content": content}]


def build_mcq_completion_prompt(
    record: Dict[str, Any],
    *,
    examples: Optional[Sequence[Dict[str, Any]]] = None,
    lang: str = "zh",
) -> str:
    """供 /completions 续写似然打分使用的纯文本 prompt。

    选择题的似然打分要求 prompt 收在一个「答案位」上，让候选选项字母成为
    紧接着的续写。因此这里固定使用 few-shot + 「答案：」的收口形式，
    不随 protocol 配置变化 —— 否则 prompt 与续写的边界会不确定。
    """
    choices = list(record.get("choices") or [])
    blocks: List[str] = []
    for ex in examples or []:
        ex_choices = list(ex.get("choices") or [])
        gold = str(ex.get("answer") or "").strip().upper()
        if not ex_choices or gold not in CHOICE_LETTERS[: len(ex_choices)]:
            continue
        if lang == "zh":
            blocks.append(f"问题：{ex['question']}\n{render_options(ex_choices)}\n答案：{gold}")
        else:
            blocks.append(f"Question: {ex['question']}\n{render_options(ex_choices, lang='en')}\nAnswer: {gold}")

    if lang == "zh":
        body = f"问题：{record['question']}\n{render_options(choices)}\n答案："
        lead = "以下是一道单项选择题，请作答。\n\n"
    else:
        body = f"Question: {record['question']}\n{render_options(choices, lang='en')}\nAnswer: "
        lead = "Answer the following multiple-choice question.\n\n"

    prefix = lead + ("\n\n".join(blocks) + "\n\n" if blocks else "")
    return prefix + body


def build_mcq_logprob(
    record: Dict[str, Any],
    *,
    examples: Optional[Sequence[Dict[str, Any]]] = None,
    protocol: str = "fewshot",
    lang: str = "zh",
) -> List[Dict[str, str]]:
    """似然打分模式：prompt 收在「答案：」处以让首 token 恰好是选项字母。"""
    if protocol == "direct":
        return build_mcq(record, examples=None, protocol="direct", lang=lang)
    msgs = build_mcq(record, examples=examples, protocol="fewshot", lang=lang)
    return msgs


# --------------------------------------------------------------------------- #
# 数值推理（GSM8K / MATH）
# --------------------------------------------------------------------------- #
_GSM_FEWSHOT = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "In April, Natalia sold 48 clips. In May, she sold half: 48 / 2 = 24. Total: 48 + 24 = 72.\n#### 72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Per minute: 12 / 60 = 0.2 dollars. For 50 minutes: 0.2 * 50 = 10 dollars.\n#### 10",
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. She has half of the money she needs. Her parents give her $15, and her grandparents give her twice that amount. How much more money does she need?",
        "answer": "Betty has 100 / 2 = 50 dollars. Parents give 15, grandparents give 2 * 15 = 30. She has 50 + 15 + 30 = 95. She needs 100 - 95 = 5.\n#### 5",
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "Each week he writes 3 * 2 = 6 pages per friend-set, so 6 * 2 = 12 pages. Per year: 12 * 52 = 624.\n#### 624",
    },
]


def build_numeric(
    record: Dict[str, Any],
    *,
    num_fewshot: int = 4,
    lang: str = "zh",
) -> List[Dict[str, str]]:
    blocks: List[str] = []
    for ex in _GSM_FEWSHOT[: max(0, int(num_fewshot))]:
        blocks.append(f"问题：{ex['question']}\n解答：{ex['answer']}")
    body = (
        f"问题：{record['question']}\n"
        f"解答：请逐步推理，并在最后一行以「#### 数值」的形式给出最终数值答案。"
    )
    content = ("\n\n".join(blocks) + "\n\n" if blocks else "") + body
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# 短答案推理（BBH / 逻辑类）
# --------------------------------------------------------------------------- #
BBH_INSTRUCTION_ZH = (
    "请逐步推理，并在最后一行严格以「所以答案是 X」的格式给出最终答案，"
    "X 应当尽量简短（例如一个词、一个数字或一个选项字母）。"
)


def build_bbh(
    record: Dict[str, Any],
    *,
    examples: Optional[Sequence[Dict[str, Any]]] = None,
    num_fewshot: int = 0,
) -> List[Dict[str, str]]:
    blocks: List[str] = []
    for ex in (examples or [])[: max(0, int(num_fewshot))]:
        blocks.append(f"问题：{ex['question']}\n解答：{ex.get('answer')}\n所以答案是 {ex.get('answer')}")
    body = f"问题：{record['question']}\n解答：{BBH_INSTRUCTION_ZH}"
    content = ("\n\n".join(blocks) + "\n\n" if blocks else "") + body
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# 结构化输出（意图 + 槽位）—— 与真实业务最贴近
# --------------------------------------------------------------------------- #
DEFAULT_INTENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["intent", "slots"],
    "properties": {
        "intent": {"type": "string", "description": "意图标签，必须来自给定意图枚举"},
        "slots": {
            "type": "object",
            "description": "槽位键值对，键必须来自给定槽位枚举；无对应槽位时用空对象",
        },
    },
}


def build_json_schema(
    record: Dict[str, Any],
    *,
    schema: Optional[Dict[str, Any]] = None,
    intent_labels: Optional[Sequence[str]] = None,
    slot_names: Optional[Sequence[str]] = None,
    examples: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """构造「意图识别 + 槽位填充」的结构化输出 prompt。

    评测这个任务的意义：意图识别系统在生产中最常见的失败不是分类错，
    而是输出不可解析（多个 JSON、夹带解释、字段名漂移、枚举越界）。
    """
    import json

    meta = record.get("meta") or {}
    labels = list(intent_labels or meta.get("intent_labels") or [])
    slots = list(slot_names or meta.get("slot_names") or [])
    sch = schema or DEFAULT_INTENT_SCHEMA

    lines = [
        "你是一个意图识别与槽位抽取模块。请分析用户输入，输出结构化结果。",
        "",
        f"意图枚举（intent 必须严格取其中一个值）：{', '.join(labels)}" if labels else "",
        f"槽位枚举（slots 的键必须取自其中）：{', '.join(slots)}" if slots else "",
        "",
        "输出必须满足以下 JSON Schema：",
        json.dumps(sch, ensure_ascii=False, indent=2),
        "",
        "要求：只输出一个 JSON 对象，不要输出解释、不要使用 markdown 代码块、不要输出多余字段。",
    ]
    if examples:
        lines.append("")
        lines.append("示例：")
        for ex in examples[:2]:
            gold = ex.get("answer")
            if not isinstance(gold, str):
                gold = json.dumps(gold, ensure_ascii=False)
            lines.append(f"输入：{ex['question']}")
            lines.append(f"输出：{gold}")
    lines += ["", f"输入：{record['question']}", "输出："]

    content = "\n".join(ln for ln in lines if ln is not None)
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# 指令遵循（IFEval）
# --------------------------------------------------------------------------- #
def build_ifeval(record: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": str(record["question"])}]


# --------------------------------------------------------------------------- #
def build_prompt(
    task: str,
    record: Dict[str, Any],
    *,
    protocol: str = "fewshot",
    lang: str = "zh",
    num_fewshot: int = 0,
    examples: Optional[Sequence[Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """任务 → prompt 的统一入口。"""
    cfg = cfg or {}
    if task in ("ceval", "cmmlu", "mmlu", "mmlu_pro", "arc", "agieval", "bbh_mcq", "custom_mcq"):
        return build_mcq(record, examples=examples, protocol=protocol, lang=lang)
    if task in ("gsm8k", "math"):
        return build_numeric(record, num_fewshot=num_fewshot, lang=lang)
    if task == "bbh":
        return build_bbh(record, examples=examples, num_fewshot=num_fewshot)
    if task == "json_schema":
        return build_json_schema(
            record,
            schema=cfg.get("schema"),
            intent_labels=cfg.get("intent_labels"),
            slot_names=cfg.get("slot_names"),
            examples=examples,
        )
    if task == "ifeval":
        return build_ifeval(record)
    # 未知任务：退化为直接问答
    return [{"role": "user", "content": str(record["question"])}]
