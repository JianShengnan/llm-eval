"""数据集加载与归一化。

不同基准的字段名和答案格式各不相同，这一层负责全部吸收掉：
  C-Eval      question / A~D / answer = "0"~"3"（索引）
  CMMLU       Question / A~D / Answer = "A"~"D"（字母）
  MMLU        question / choices[4] / answer = 0~3（整数索引）
  MMLU-Pro    question / options[10] / answer_index
  GSM8K       question / answer = "... #### 42"
  BBH         input / target（无选项）
  IFEval      prompt / instruction_id_list / kwargs

统一后的样本结构：
  {"id", "question", "choices"|None, "answer", "subject", "meta"}

离线部署建议：在联网机器上用 scripts/prepare_datasets.py 转成 JSONL，
内网机器只放 JSONL，无需安装 datasets 库、无需外网。
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .answers import CHOICE_LETTERS, canonical_number, strip_reasoning
from .util import read_jsonl, sha256_file, truncate

QUESTION_KEYS = ("question", "Question", "query", "input", "prompt", "problem", "text", "instruction")
SUBJECT_KEYS = ("subject", "Subject", "category", "Category", "field", "topic", "subtask", "task", "dataset")
ANSWER_KEYS = ("answer", "Answer", "target", "label", "gold", "output")
CHOICE_KEYS = ("choices", "options", "Choices", "Options")


class DatasetError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# 归一化
# --------------------------------------------------------------------------- #
def _first(raw: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in raw and raw[k] not in (None, ""):
            return raw[k]
    return None


def _parse_option_string(text: str) -> Optional[List[str]]:
    """解析 "A. 北京\\nB. 上海" 这类把选项塞进一个字符串的写法。"""
    matches = re.findall(r"(?:^|\n)\s*[\(\（]?([A-Ja-j])[\)\）\.、:：]\s*(.+)", str(text))
    if len(matches) < 2:
        return None
    seen: Dict[str, str] = {}
    for letter, body in matches:
        seen.setdefault(letter.upper(), body.strip())
    letters = [c for c in CHOICE_LETTERS if c in seen]
    if len(letters) < 2:
        return None
    return [seen[c] for c in letters]


def _extract_choices(raw: Dict[str, Any]) -> Optional[List[str]]:
    for key in CHOICE_KEYS:
        val = raw.get(key)
        if isinstance(val, list) and len(val) >= 2:
            return [str(x).strip() for x in val]
        if isinstance(val, dict) and len(val) >= 2:
            keys = sorted(val.keys(), key=lambda k: str(k))
            return [str(val[k]).strip() for k in keys]
        if isinstance(val, str) and len(val) > 8:
            parsed = _parse_option_string(val)
            if parsed:
                return parsed

    # C-Eval / CMMLU 风格：A/B/C/D 平铺成独立字段
    letters = [c for c in CHOICE_LETTERS if c in raw and isinstance(raw[c], str) and raw[c].strip()]
    if len(letters) >= 2:
        return [str(raw[c]).strip() for c in letters]
    return None


def _normalize_answer(raw: Dict[str, Any], choices: Optional[List[str]]) -> Optional[str]:
    ans = _first(raw, ANSWER_KEYS)
    if ans is None:
        return None

    # 整数索引 / 数字字符串索引（仅当存在选项时）
    if choices:
        idx: Optional[int] = None
        if isinstance(ans, bool):
            idx = None
        elif isinstance(ans, int):
            idx = ans
        elif isinstance(ans, str) and re.fullmatch(r"\d{1,2}", ans.strip()):
            n = int(ans.strip())
            idx = n if 0 <= n < len(choices) else None
        if idx is not None and 0 <= idx < len(choices):
            return CHOICE_LETTERS[idx]

        text = str(ans).strip()
        if re.fullmatch(r"[A-Ja-j]", text):
            return text.upper()
        # 答案写成了选项原文
        for i, choice in enumerate(choices):
            if choice.strip() == text:
                return CHOICE_LETTERS[i]

    if isinstance(ans, bool):
        return "yes" if ans else "no"
    if isinstance(ans, (dict, list)):
        import json

        return json.dumps(ans, ensure_ascii=False, sort_keys=True)
    if isinstance(ans, (int, float)):
        return canonical_number(str(ans))
    return strip_reasoning(str(ans)).strip() or None


def normalize_record(raw: Dict[str, Any], *, index: int = 0, source: str = "") -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise DatasetError(f"样本必须是 JSON 对象，实际为 {type(raw).__name__}（{source}）")

    question = _first(raw, QUESTION_KEYS)
    if question is None:
        raise DatasetError(f"样本缺少题面字段（尝试过 {QUESTION_KEYS}）：{truncate(str(raw), 200)}")
    question = str(question).strip()

    choices = _extract_choices(raw)
    answer = _normalize_answer(raw, choices)

    rid = raw.get("id") or raw.get("index") or f"{source or 'item'}-{index}"
    subject = _first(raw, SUBJECT_KEYS)
    meta: Dict[str, Any] = {}
    # 数据集自带的 meta 原样保留：意图枚举、槽位枚举等定义可以随数据一起走，
    # 避免评测时还要在 config 里重复声明一遍。
    if isinstance(raw.get("meta"), dict):
        meta.update(raw["meta"])
    if "instruction_id_list" in raw:
        meta["instruction_id_list"] = list(raw.get("instruction_id_list") or [])
        meta["kwargs"] = list(raw.get("kwargs") or [])
    if "explanation" in raw and raw["explanation"]:
        meta["explanation"] = truncate(str(raw["explanation"]), 400)

    return {
        "id": str(rid),
        "question": question,
        "choices": choices,
        "answer": answer,
        "subject": str(subject).strip() if subject else "",
        "meta": meta,
    }


def load_jsonl(path: str, *, source: Optional[str] = None, strict: bool = False) -> List[Dict[str, Any]]:
    src = source or os.path.basename(path)
    out: List[Dict[str, Any]] = []
    skipped = 0
    for i, raw in enumerate(read_jsonl(path)):
        try:
            out.append(normalize_record(raw, index=i, source=src))
        except DatasetError:
            skipped += 1
            if strict:
                raise
    if skipped:
        import sys

        print(f"  [warn] {src}: 跳过 {skipped} 条无法归一化的样本", file=sys.stderr)
    return out


def resolve_dataset_path(task_name: str, split: str, data_root: str, explicit: Optional[str] = None) -> Optional[str]:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        os.path.join(data_root, task_name, f"{split}.jsonl"),
        os.path.join(data_root, f"{task_name}.{split}.jsonl"),
        os.path.join(data_root, f"{task_name}.jsonl"),
        os.path.join(data_root, task_name, "test.jsonl"),
        os.path.join(data_root, task_name, "val.jsonl"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_task_dataset(
    task_name: str,
    split: str,
    data_root: str,
    *,
    explicit_path: Optional[str] = None,
    allow_synthetic: bool = False,
    synthetic_n: int = 60,
) -> List[Dict[str, Any]]:
    path = resolve_dataset_path(task_name, split, data_root, explicit_path)
    if path is None:
        if allow_synthetic:
            return synthetic_dataset(task_name, synthetic_n)
        raise DatasetError(
            f"找不到数据集：{task_name} / split={split}\n"
            f"  查找目录：{os.path.abspath(data_root)}/{task_name}/\n"
            f"  期望文件：{split}.jsonl\n"
            f"  解决办法：在联网机器执行 scripts/prepare_datasets.py，把生成的 JSONL 拷进内网；\n"
            f"           或先把 config 里的 tasks[].path 指到你已有的 JSONL。\n"
            f"           仅想验证流程可用：python -m geval selftest（使用内置合成数据）"
        )
    return load_jsonl(path, source=os.path.basename(path))


def dataset_manifest(path: str, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """记录数据集的版本指纹，保证评测结果可复现、可追溯。"""
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "n_records": len(records),
        "subjects": len({r.get("subject") for r in records if r.get("subject")}),
    }


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #
def stratified_sample(
    records: Sequence[Dict[str, Any]],
    limit: int,
    *,
    key: str = "subject",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """按 key 分层等比抽样，保证每个分组至少 1 条。

    全量跑 C-Eval(1346) + CMMLU(11582) + MMLU(14042) 成本很高；
    smoke 阶段用分层抽样能在几百条内得到与全量高度相关的排序结论。
    """
    if limit <= 0 or limit >= len(records):
        return list(records)

    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        buckets.setdefault(str(r.get(key) or "unknown"), []).append(r)

    groups = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    total = len(records)
    picked: List[Dict[str, Any]] = []
    for name, rows in groups:
        quota = max(1, int(round(limit * len(rows) / total)))
        quota = min(quota, len(rows))
        picked.extend(rng.sample(rows, quota))

    if len(picked) > limit:
        picked = rng.sample(picked, limit)
    picked.sort(key=lambda r: str(r["id"]))
    return picked


def dedupe(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按题面归一化去重，避免同一题重复计入导致分数虚高。"""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        key = re.sub(r"\s+", "", str(r.get("question") or ""))[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# 内置合成数据（selftest / 流程验证专用）
# --------------------------------------------------------------------------- #
_CHOICE_POOL = [
    ("以下哪一项是中国的首都？", ["北京", "上海", "广州", "深圳"]),
    ("水的化学式是什么？", ["H2O", "CO2", "O2", "NaCl"]),
    ("一天中有多少个小时？", ["24", "12", "48", "60"]),
    ("下列哪个是编程语言？", ["Python", "JPEG", "HTTP", "MP4"]),
    ("一年有几个季度？", ["4", "2", "6", "12"]),
    ("光速大约是多少米每秒？", ["3 亿", "3 万", "30 万", "3000"]),
    ("下列哪个不是水果？", ["土豆", "苹果", "香蕉", "橙子"]),
    ("计算机中 1 字节等于多少位？", ["8", "4", "16", "32"]),
    ("中国的陆地面积约为多少万平方公里？", ["960", "560", "1200", "300"]),
    ("下列哪个是操作系统？", ["Linux", "Oracle", "Photoshop", "Excel"]),
    ("太阳系中体积最大的行星是？", ["木星", "地球", "火星", "金星"]),
    ("下列哪个单位用于衡量电流？", ["安培", "焦耳", "帕斯卡", "摩尔"]),
    ("人体最大的器官是？", ["皮肤", "肝脏", "心脏", "肺"]),
    ("下列哪个是非关系型数据库？", ["Redis", "MySQL", "Oracle", "SQL Server"]),
    ("HTTP 状态码 404 表示什么？", ["资源未找到", "服务器错误", "请求成功", "重定向"]),
    ("下列哪位是相对论的提出者？", ["爱因斯坦", "牛顿", "达尔文", "居里夫人"]),
    ("一公斤等于多少克？", ["1000", "100", "10", "10000"]),
    ("下列哪种气体在空气中含量最高？", ["氮气", "氧气", "二氧化碳", "氢气"]),
    ("《红楼梦》的作者是？", ["曹雪芹", "吴承恩", "施耐庵", "罗贯中"]),
    ("下列哪个是版本控制工具？", ["Git", "Docker", "Nginx", "Kafka"]),
]

_SHORT_POOL = [
    ("判断：所有的鸟都会飞。回答 yes 或 no。", "no"),
    ("判断：水在标准大气压下 100 摄氏度沸腾。回答 yes 或 no。", "yes"),
    ("判断：1 是质数。回答 yes 或 no。", "no"),
    ("判断：光速比声速快。回答 yes 或 no。", "yes"),
    ("5 加 7 等于多少？只回答数字。", "12"),
    ("12 除以 4 等于多少？只回答数字。", "3"),
    ("下面哪个不属于哺乳动物？A 鲸 B 蝙蝠 C 企鹅 D 大象。只回答字母。", "(C)"),
    ("下面哪个不是编程范式？A 面向对象 B 函数式 C 面向过程 D 面向汇编。只回答字母。", "(D)"),
    ("判断：地球是太阳系中最大的行星。回答 yes 或 no。", "no"),
    ("判断：二进制只有 0 和 1 两个数字。回答 yes 或 no。", "yes"),
    ("9 乘以 9 等于多少？只回答数字。", "81"),
    ("下面哪个是关系型数据库？A Nginx B MySQL C Kafka D Docker。只回答字母。", "(B)"),
]

_MATH_POOL = [
    ("小明有 {a} 个苹果，又买了 {b} 个，一共多少个？", lambda a, b: a + b),
    ("一个班有 {a} 名学生，平均分成 {b} 组，每组多少人？", lambda a, b: a // b),
    ("每件商品 {a} 元，买 {b} 件需要多少钱？", lambda a, b: a * b),
    ("仓库有 {a} 箱货物，运走 {b} 箱，还剩多少箱？", lambda a, b: a - b),
]

_INTENTS = [
    ("查询余额", ["account"], "查询账户余额"),
    ("转账", ["account", "amount"], "向他人账户转账"),
    ("修改密码", ["account"], "修改登录密码"),
    ("挂失银行卡", ["account", "card"], "挂失银行卡"),
    ("查询交易明细", ["account", "date_range"], "查询交易流水"),
    ("申请贷款", ["amount", "term"], "申请个人贷款"),
]

_SLOT_VALUES = {
    "account": ["6222 0000 1111 2222", "6225 8888 9999 0000"],
    "amount": ["1000 元", "5000 元", "20000 元"],
    "card": ["尾号 1234", "尾号 5678"],
    "date_range": ["近一个月", "近三个月"],
    "term": ["12 期", "24 期"],
}


def synthetic_mcq(n: int = 60, seed: int = 11) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        q, opts = _CHOICE_POOL[i % len(_CHOICE_POOL)]
        shuffled = list(opts)
        rng.shuffle(shuffled)
        gold_idx = shuffled.index(opts[0])
        out.append(
            {
                "id": f"syn-mcq-{i:04d}",
                "question": f"{q}",
                "choices": shuffled,
                "answer": CHOICE_LETTERS[gold_idx],
                "subject": f"syn-subject-{i % 5}",
                "meta": {},
            }
        )
    return out


def synthetic_numeric(n: int = 40, seed: int = 12) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        tpl, fn = _MATH_POOL[i % len(_MATH_POOL)]
        a = rng.randint(2, 90)
        b = rng.randint(2, 12)
        if "平均分成" in tpl:
            a = b * rng.randint(2, 20)
        out.append(
            {
                "id": f"syn-math-{i:04d}",
                "question": tpl.format(a=a, b=b),
                "choices": None,
                "answer": canonical_number(str(fn(a, b))),
                "subject": "syn-math",
                "meta": {},
            }
        )
    return out


def synthetic_json_schema(n: int = 40, seed: int = 13) -> List[Dict[str, Any]]:
    """构造「意图 + 槽位」结构化输出样本 —— 与真实意图识别最贴近。

    槽位取值按轮次确定性轮转，保证每条题面唯一（避免被去重压掉）。
    """
    import json

    out: List[Dict[str, Any]] = []
    for i in range(n):
        intent, slots, desc = _INTENTS[i % len(_INTENTS)]
        rnd = i // len(_INTENTS)
        slot_obj = {k: _SLOT_VALUES[k][(rnd + j) % len(_SLOT_VALUES[k])] for j, k in enumerate(slots)}
        gold = {"intent": intent, "slots": slot_obj}
        detail = "，".join(f"{k}是{v}" for k, v in slot_obj.items())
        out.append(
            {
                "id": f"syn-json-{i:04d}",
                "question": f"用户说：「帮我{desc}，{detail}」。请输出结构化结果。",
                "choices": None,
                "answer": json.dumps(gold, ensure_ascii=False, sort_keys=True),
                "subject": "syn-intent",
                "meta": {
                    "intent": intent,
                    "slots": slot_obj,
                    "intent_labels": [x[0] for x in _INTENTS],
                    "slot_names": sorted(_SLOT_VALUES.keys()),
                },
            }
        )
    return out


_IFE_TOPICS = ["北京", "海洋", "登山", "手冲咖啡", "垃圾分类", "城市骑行", "古典音乐", "咖啡因"]


def synthetic_ifeval(n: int = 30, seed: int = 14) -> List[Dict[str, Any]]:
    """构造规则可判定的指令遵循样本；reference_response 是能满足约束的参考输出。

    题面按「指令类型 × 主题」组合，保证唯一，从而让去重不会压缩样本量。
    """
    out: List[Dict[str, Any]] = []
    for i in range(n):
        topic = _IFE_TOPICS[i % len(_IFE_TOPICS)]
        kind = (i // len(_IFE_TOPICS)) % 5
        if kind == 0:
            ids = ["punctuation:no_comma"]
            kwargs = [{}]
            prompt = f"请用一段话介绍{topic}，不要使用任何逗号。"
            ref = f"{topic}是一个非常有意思的话题它涉及很多方面的内容值得我们去了解和探索"
        elif kind == 1:
            ids = ["length_constraints:number_words", "keywords:existence"]
            kwargs = [{"relation": "at least", "num_words": 30}, {"keywords": [topic]}]
            prompt = f"写一段至少 30 个字的短文介绍{topic}，并且必须提到「{topic}」这个词。"
            ref = (
                f"{topic}是一个非常值得深入了解的话题它包含着丰富的知识值得我们去学习"
                f"并且能够在日常生活中发挥实际作用让人受益匪浅"
            )
        elif kind == 2:
            ids = ["detectable_format:json_format"]
            kwargs = [{}]
            prompt = f"用 JSON 格式回答：{topic}属于哪个大类？只输出 JSON，不要任何解释。"
            ref = '{"类别": "通识话题"}'
        elif kind == 3:
            ids = ["startend:quotation"]
            kwargs = [{}]
            prompt = f"请用一句话介绍{topic}，并且用双引号把整段话包起来。"
            ref = f'"{topic}值得花时间去了解。"'
        else:
            ids = ["startend:end_checker"]
            kwargs = [{"end_phrase": "值得一试。"}]
            prompt = f"写一段话介绍{topic}，最后必须以「值得一试。」结尾。"
            ref = f"{topic}能带来很多乐趣也让人收获新的知识。值得一试。"
        out.append(
            {
                "id": f"syn-ifeval-{i:04d}",
                "question": prompt,
                "choices": None,
                "answer": None,
                "subject": ids[0].split(":")[0],
                "meta": {"instruction_id_list": ids, "kwargs": kwargs, "reference_response": ref},
            }
        )
    return out


def synthetic_short(n: int = 40, seed: int = 15) -> List[Dict[str, Any]]:
    """短答案推理样本，对应 BBH 的「所以答案是 X」判定路径。"""
    out: List[Dict[str, Any]] = []
    for i in range(n):
        q, a = _SHORT_POOL[i % len(_SHORT_POOL)]
        out.append(
            {
                "id": f"syn-short-{i:04d}",
                "question": q,
                "choices": None,
                "answer": a,
                "subject": "syn-short",
                "meta": {},
            }
        )
    return out


_SYNTHETIC_BUILDERS = {
    "mcq": synthetic_mcq,
    "numeric": synthetic_numeric,
    "short": synthetic_short,
    "json": synthetic_json_schema,
    "ifeval": synthetic_ifeval,
}


def synthetic_dataset(task_name: str, n: int = 60) -> List[Dict[str, Any]]:
    """按任务名返回合成数据，用于全链路离线自检。"""
    mapping = {
        "ceval": "mcq",
        "cmmlu": "mcq",
        "mmlu": "mcq",
        "mmlu_pro": "mcq",
        "arc": "mcq",
        "agieval": "mcq",
        "gsm8k": "numeric",
        "math": "numeric",
        "bbh": "short",
        "json_schema": "json",
        "ifeval": "ifeval",
        "custom": "mcq",
    }
    kind = mapping.get(task_name, "mcq")
    return _SYNTHETIC_BUILDERS[kind](n)
