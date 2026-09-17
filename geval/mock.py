"""确定性 mock 模型：让整条评测链路可以完全离线自检。

它不是一个「永远答对」的假模型，而是刻意模拟真实退化：
  - 按 accuracy 返回正确 / 错误答案；
  - 按 noise 概率制造格式噪声（多余解释、缺 JSON 括号、字母写成中文等），
    用来验证答案抽取层在脏输出下是否仍然工作。

答案来源通过 register() 旁路注入（真实模型不可能知道标准答案）。
"""

from __future__ import annotations

import json
import random
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

from .client import Message, ModelClient, Reply, Usage
from .util import seed_from, sha256_text

_LETTERS = "ABCDEFGHIJ"


class MockClient(ModelClient):
    provider = "mock"

    def __init__(self, cfg: Dict[str, Any], **kw: Any) -> None:
        # 刻意不调用父类 __init__：mock 不发任何网络请求。
        self.cfg = dict(cfg)
        self.id = str(cfg["id"])
        self.label = str(cfg.get("label") or self.id)
        self.model = f"mock::{self.id}"
        self.base_url = "mock://local"
        self.supports_logprobs = True
        self.system_prompt = cfg.get("system_prompt")
        self.params = {"temperature": 0, "max_tokens": 1024}
        self.extra_body = {}
        self.cache_dir = None
        self.cache_hits = 0
        self.calls = 0
        self.errors = 0
        self.latencies_ms: List[float] = []
        self.usage = Usage()
        self.http = None  # type: ignore[assignment]

        self.accuracy = float(cfg.get("accuracy", 0.75))
        self.noise = float(cfg.get("noise", 0.12))
        self.err_rate = float(cfg.get("error_rate", 0.0))
        self.latency_ms_base = float(cfg.get("latency_ms", 420.0))
        self.latency_ms_jitter = float(cfg.get("latency_jitter_ms", 120.0))
        self.seed = int(cfg.get("seed", 7))

        self._registry: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def register(self, prompt_key: str, entry: Dict[str, Any]) -> None:
        """由 runner 在构造样本时注入标准答案 / 期望输出。"""
        with self._lock:
            self._registry[prompt_key] = entry

    def stats(self) -> Dict[str, Any]:
        from .client import _pct  # 复用同一套分位数实现

        with self._lock:
            lat = sorted(self.latencies_ms)
            return {
                "model": self.id,
                "label": self.label,
                "calls": self.calls,
                "errors": self.errors,
                "cache_hits": self.cache_hits,
                "http_requests": 0,
                "http_retries": 0,
                "usage": self.usage.to_dict(),
                "latency_ms_p50": _pct(lat, 50),
                "latency_ms_p95": _pct(lat, 95),
                "latency_ms_mean": (sum(lat) / len(lat)) if lat else None,
                "mock": True,
            }

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Union[str, Sequence[Message]],
        *,
        stop: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Reply:
        msgs = self._build(messages)
        prompt_key = sha256_text(msgs[-1]["content"] if msgs else "")
        with self._lock:
            entry = self._registry.get(prompt_key)

        rng = random.Random(self.seed ^ seed_from(prompt_key))
        latency = max(1.0, rng.gauss(self.latency_ms_base, self.latency_ms_jitter))

        with self._lock:
            self.calls += 1
            self.latencies_ms.append(latency)

        if self.err_rate and rng.random() < self.err_rate:
            with self._lock:
                self.errors += 1
            return Reply(text="", latency_ms=latency, error="mock: simulated transport error")

        if entry is None:
            text = "（mock 未注册该样本，返回空）"
        else:
            text = self._render(entry, rng)

        usage = Usage(
            prompt_tokens=int(len(msgs[-1]["content"]) / 2) if msgs else 0,
            completion_tokens=int(len(text) / 2),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        with self._lock:
            self.usage.merge(usage)

        return Reply(
            text=text,
            usage=usage,
            latency_ms=latency,
            finish_reason="stop",
            top_logprobs=None,
            raw={"mock": True},
        )

    # ------------------------------------------------------------------ #
    def first_token_logprobs(
        self, messages: Union[str, Sequence[Message]], *, top_n: int = 24, **kw: Any
    ) -> Optional[Dict[str, float]]:
        msgs = self._build(messages)
        prompt_key = sha256_text(msgs[-1]["content"] if msgs else "")
        with self._lock:
            entry = self._registry.get(prompt_key)
        if not entry or not entry.get("gold"):
            return None

        rng = random.Random(self.seed ^ seed_from(prompt_key) ^ 0x5EED)
        gold = str(entry["gold"]).strip().upper()
        letters = [c for c in _LETTERS if c in set(list(entry.get("letters") or _LETTERS[:4]))]
        if gold not in letters:
            return None

        hit = rng.random() < self.accuracy
        out: Dict[str, float] = {}
        for ch in letters:
            out[ch] = 0.4 * rng.random() - 3.0
        out[gold] = (0.0 if hit else -1.2) + 0.4 * rng.random()
        if not hit:
            others = [c for c in letters if c != gold]
            if others:
                pick = others[rng.randrange(len(others))]
                out[pick] = 0.3 + 0.4 * rng.random()
        return out

    def choice_logprobs(
        self, messages: Union[str, Sequence[Message]], letters: Sequence[str], **kw: Any
    ) -> Optional[Dict[str, float]]:
        return self.first_token_logprobs(messages, **kw)

    def choice_logprobs_reply(
        self, messages: Union[str, Sequence[Message]], letters: Sequence[str], **kw: Any
    ) -> Any:
        """似然打分模式下也要计入调用次数与时延，以便自检覆盖统计链路。"""
        reply = self.chat(messages)
        scores = self.first_token_logprobs(messages, **kw)
        return scores, reply

    # ------------------------------------------------------------------ #
    def raw_completion(
        self, prompt: str, *, echo: Optional[bool] = None, logprobs: Optional[int] = None, **kw: Any
    ) -> Reply:
        """返回一个形状合法的 /completions echo 响应，用于离线验证解析链路。"""
        toks = [t for t in str(prompt).split(" ") if t]
        with self._lock:
            self.calls += 1
            self.latencies_ms.append(self.latency_ms_base)
        raw = {
            "choices": [
                {
                    "text": prompt,
                    "logprobs": {
                        "tokens": toks,
                        "token_logprobs": [None] + [-0.1] * max(0, len(toks) - 1),
                    },
                }
            ],
            "usage": {"prompt_tokens": len(toks), "completion_tokens": 0, "total_tokens": len(toks)},
        }
        return Reply(text=str(prompt), latency_ms=self.latency_ms_base, raw=raw)

    def score_continuations(
        self, prompt_text: str, candidates: Sequence[str], *, echo: bool = True
    ) -> Dict[str, Any]:
        """续写似然打分：命中时让正确答案拿到最高分，未命中时让某个错误选项胜出。"""
        key = sha256_text(str(prompt_text))
        with self._lock:
            entry = self._registry.get(key)
            self.calls += 1
            latency = max(1.0, self.latency_ms_base)
            self.latencies_ms.append(latency)
        if entry is None:
            return {"scores": {}, "latency_ms": latency, "prompt_tokens": 0, "completion_tokens": 0, "n_requests": 0}

        rng = random.Random(self.seed ^ seed_from(key) ^ 0x11CE)
        gold = str(entry.get("gold") or "").strip().upper()
        out: Dict[str, Dict[str, float]] = {}
        for cand in candidates:
            # n_tokens 故意设成 2，用来验证 acc 与 acc_norm 会产生不同排序
            out[cand] = {"logprob": -3.0 - rng.random(), "n_tokens": 2.0}

        gold_cand = next((c for c in candidates if str(c).strip().upper() == gold), None)
        if gold_cand is not None:
            if rng.random() < self.accuracy:
                out[gold_cand] = {"logprob": -0.1 - 0.1 * rng.random(), "n_tokens": 1.0}
            else:
                others = [c for c in candidates if c != gold_cand]
                if others:
                    pick = others[rng.randrange(len(others))]
                    out[pick] = {"logprob": -0.1 - 0.1 * rng.random(), "n_tokens": 1.0}
                out[gold_cand] = {"logprob": -5.0 - 0.1 * rng.random(), "n_tokens": 1.0}

        return {
            "scores": out,
            "latency_ms": latency * len(candidates),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "n_requests": len(candidates),
        }

    # ------------------------------------------------------------------ #
    def _build(self, messages: Union[str, Sequence[Message]]) -> List[Message]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return [dict(m) for m in messages]

    def _render(self, entry: Dict[str, Any], rng: random.Random) -> str:
        gold = str(entry.get("gold") or "")
        scripted = entry.get("mock_answer")
        target = str(scripted) if scripted is not None else gold
        kind = entry.get("kind") or "text"
        hit = rng.random() < self.accuracy

        # 注意：各类型必须独立处理「命中 / 未命中」，不能因为有 scripted
        # 就统一提前返回，否则未命中分支会被吞掉，mock 的 accuracy 参数失效。

        if kind == "ifeval":
            # 约束多为「开头/结尾/标点/字数」，任何啰嗦噪声都会破坏约束，
            # 因此该类型不加噪声；未命中时给出一个看似合理但不满足约束的回答。
            return target if hit else _NONCOMPLIANT_RESPONSE

        if kind == "json":
            if hit:
                body = target
                if rng.random() < self.noise:
                    body = _add_noise(body, entry, rng, corrupt=False)  # 夹带解释 -> 严格模式判失败
                return body
            # 未命中：既改错意图（语义错误），又随机破坏语法（格式错误）
            return _wrong_json(target, rng)

        if kind == "mcq":
            letters = [c for c in _LETTERS if c in set(list(entry.get("letters") or _LETTERS[:4]))]
            ans = gold.strip().upper()
            if not hit:
                pool = [c for c in letters if c != ans] or letters
                ans = pool[rng.randrange(len(pool))] if pool else ans
            body = f"答案是 {ans}。"
        elif kind == "numeric":
            ans = gold
            if not hit:
                try:
                    ans = str(float(ans) + rng.choice([1, -1, 2, -2, 10]))
                except ValueError:
                    ans = ans + "0"
            body = f"逐步计算后得到结果。\n#### {ans}"
        else:
            # 短答案类（BBH）：命中时按约定格式收口，未命中时给出明确不同的结论
            ans = gold if hit else _mutate_short(gold, rng)
            body = f"根据题目条件逐步推理，得到结论。\n所以答案是 {ans}"

        if rng.random() < self.noise:
            body = _add_noise(body, entry, rng, corrupt=False)
        return body


# --------------------------------------------------------------------------- #
# 一个「看似合理但不满足任何常见约束」的回答：
# 含中文逗号（破坏 no_comma）、偏短（破坏 ≥N 字）、不是 JSON（破坏 json_format）、
# 不以指定短语结尾（破坏 end_checker）、未加引号（破坏 quotation）。
_NONCOMPLIANT_RESPONSE = "很抱歉，我暂时无法完成这个请求。"


def _wrong_json(text: str, rng: random.Random) -> str:
    """JSON 未命中：先改错意图（语义错误），再随机破坏语法（格式错误）。"""
    try:
        obj = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return _corrupt_json(text, rng)

    if isinstance(obj, dict):
        if "intent" in obj:
            obj["intent"] = "未知意图"
        if "slots" in obj and isinstance(obj["slots"], dict) and obj["slots"]:
            first_key = sorted(obj["slots"].keys())[0]
            obj["slots"][first_key] = "未提供"
    mutated = json.dumps(obj, ensure_ascii=False)
    return _corrupt_json(mutated, rng) if rng.random() < 0.5 else mutated


def _mutate_short(ans: str, rng: random.Random) -> str:
    """把短答案改成一个明确不同的结论，避免宽松相等误判为命中。"""
    low = ans.strip().lower()
    if low in ("yes", "true"):
        return "no"
    if low in ("no", "false"):
        return "yes"
    if low.isdigit():
        return str(int(low) + rng.choice([1, 2, -1]))
    if re.fullmatch(r"[a-j]", low):
        return "z"
    return "无法确定"


def _add_noise(body: str, entry: Dict[str, Any], rng: random.Random, *, corrupt: bool) -> str:
    """在不改变语义的前提下加噪声，模拟真实模型的啰嗦输出。"""
    prefixes = [
        "好的，我来分析一下这个问题。",
        "根据题目条件逐步推理：\n",
        "让我们仔细看这道题。\n",
        "**分析过程**\n",
    ]
    suffixes = ["\n\n希望这个解答对你有帮助。", "\n\n（以上为推理过程）", ""]
    text = rng.choice(prefixes) + body + rng.choice(suffixes)
    if corrupt and entry.get("kind") == "json":
        text = _corrupt_json(text, rng)
    return text


def _corrupt_json(text: str, rng: random.Random) -> str:
    """制造典型的 JSON 失败：尾逗号、多余说明、缺引号、截断、裸 markdown。"""
    modes = ["trailing_comma", "prose_wrap", "truncate", "missing_quote", "comment"]
    mode = modes[rng.randrange(len(modes))]
    if mode == "trailing_comma":
        return text.replace("}", ",}", 1)
    if mode == "prose_wrap":
        return "好的，结果如下：\n" + text + "\n以上。"
    if mode == "truncate":
        return text[: max(1, int(len(text) * 0.7))]
    if mode == "missing_quote":
        return text.replace('"', "'", 1)
    return text.replace("{", "```json\n{", 1) + "\n```"
