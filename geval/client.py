"""模型客户端：OpenAI 兼容 /chat/completions，线程安全，带磁盘缓存与统计。

两种调用形态：
  chat()                    —— 生成式，返回文本，适用于 GSM8K / IFEval / JSON 等任务；
  first_token_logprobs()    —— 读取首生成位置的 top_logprobs，用于对单项选择题做
                               「选项似然打分」。小模型生成式抽取容易输出跑偏的
                               字母，似然打分能显著降低这种噪声。

缓存约束：仅当采样完全确定（temperature==0）时才落盘，避免缓存污染。
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .http import HttpClient, HttpError
from .util import ensure_dir, jdump, sha256_text

Message = Dict[str, str]

DEFAULT_PARAMS: Dict[str, Any] = {
    "temperature": 0,
    "top_p": 1.0,
    "max_tokens": 1024,
}


# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def merge(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Reply:
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str = ""
    top_logprobs: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    cached: bool = False
    raw: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------- #
class ModelClient:
    """单个模型端点的客户端。实例可被多线程共享。"""

    provider = "openai_compat"

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        cache_dir: Optional[str] = None,
        timeout: float = 180.0,
        retries: int = 3,
        proxy: str = "bypass",
    ) -> None:
        self.cfg = dict(cfg)
        self.id: str = str(cfg["id"])
        self.label: str = str(cfg.get("label") or self.id)
        self.model: str = str(cfg.get("model") or self.id)
        self.base_url: str = str(cfg["base_url"]).rstrip("/")

        self.params: Dict[str, Any] = dict(DEFAULT_PARAMS)
        self.params.update(cfg.get("params") or {})
        self.extra_body: Dict[str, Any] = dict(cfg.get("extra_body") or {})
        self.fewshot_sep: str = str(cfg.get("fewshot_sep") or "\n\n")
        self.supports_logprobs: bool = bool(cfg.get("supports_logprobs", True))
        self.system_prompt: Optional[str] = cfg.get("system_prompt")

        self.http = HttpClient(
            self.base_url,
            api_key=cfg.get("api_key"),
            timeout=timeout,
            retries=retries,
            proxy=proxy,
            headers=cfg.get("headers"),
            ssl_unverified=bool(cfg.get("ssl_unverified", False)),
        )
        self.cache_dir = ensure_dir(cache_dir) if cache_dir else None
        self.cache_hits = 0
        self._lock = threading.Lock()
        self.usage = Usage()
        self.calls = 0
        self.errors = 0
        self.latencies_ms: List[float] = []

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            lat = sorted(self.latencies_ms)
            return {
                "model": self.id,
                "label": self.label,
                "calls": self.calls,
                "errors": self.errors,
                "cache_hits": self.cache_hits,
                "http_requests": self.http.stats["requests"],
                "http_retries": self.http.stats["retries"],
                "usage": self.usage.to_dict(),
                "latency_ms_p50": _pct(lat, 50),
                "latency_ms_p95": _pct(lat, 95),
                "latency_ms_mean": (sum(lat) / len(lat)) if lat else None,
            }

    # ------------------------------------------------------------------ #
    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, key[:2], key + ".json")

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.cache_hits += 1
            return obj
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_put(self, key: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        if not path:
            return
        ensure_dir(os.path.dirname(path))
        tmp = path + f".tmp{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(jdump(payload))
            os.replace(tmp, path)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    def _build_messages(self, messages: Union[str, Sequence[Message]]) -> List[Message]:
        if isinstance(messages, str):
            msgs: List[Message] = [{"role": "user", "content": messages}]
        else:
            msgs = [dict(m) for m in messages]
        if self.system_prompt and not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": self.system_prompt})
        return msgs

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
        payload: Dict[str, Any] = {"model": self.model, "messages": self._build_messages(messages)}
        payload.update(self.params)
        payload.update(self.extra_body)
        _apply_overrides(
            payload,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra=extra,
        )
        return self._request("/chat/completions", payload)

    # ------------------------------------------------------------------ #
    def raw_completion(
        self,
        prompt: str,
        *,
        stop: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = 1,
        temperature: Optional[float] = None,
        logprobs: Optional[int] = None,
        echo: Optional[bool] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Reply:
        """直接调用 /completions。

        续写似然打分（lm-eval 的 acc / acc_norm）必须用它：
        chat 接口无法表达 echo，也就拿不到续写部分的 token 级 logprob。
        """
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt}
        payload.update(self.params)
        payload.update(self.extra_body)
        _apply_overrides(
            payload,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            extra=extra,
        )
        if echo is not None:
            payload["echo"] = bool(echo)
        return self._request("/completions", payload)

    def score_continuations(
        self,
        prompt_text: str,
        candidates: Sequence[str],
        *,
        echo: bool = True,
    ) -> Dict[str, Any]:
        """对每个候选续写做似然打分。

        返回 {"scores": {候选: {"logprob", "n_tokens"}}, "latency_ms", "usage", "n_requests"}。

        实现思路与 lm-eval 一致：把候选拼到 prompt 末尾，用 echo 取回整串的 token 级
        logprob，再从末尾向前累加直到拼出的文本等于候选。
        - acc      用 logprob 之和
        - acc_norm 用长度归一化后的值，消除「续写越长 logprob 越低」的长度偏置，
                   这是 lm-eval 在选择题上的主指标

        代价：每个候选一次请求（4 选项 = 4 次/样本），所以默认仍用 gen 模式，
        只在需要与公开榜单口径对齐时切到本模式。
        """
        import time

        scores: Dict[str, Dict[str, float]] = {}
        t0 = time.perf_counter()
        prompt_tokens = completion_tokens = 0
        n_requests = 0

        for cand in candidates:
            reply = self.raw_completion(prompt_text + cand, echo=echo, max_tokens=1, logprobs=1)
            n_requests += 1
            if not reply.ok:
                continue
            prompt_tokens += reply.usage.prompt_tokens
            completion_tokens += reply.usage.completion_tokens
            token_lps = _token_logprobs(reply.raw)
            if not token_lps:
                continue
            lp_sum, n = _sum_continuation(token_lps, cand)
            if n <= 0:
                continue
            scores[cand] = {"logprob": lp_sum, "n_tokens": float(n)}

        with self._lock:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        return {
            "scores": scores,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "n_requests": n_requests,
        }

    # ------------------------------------------------------------------ #
    def _request(self, path: str, payload: Dict[str, Any]) -> Reply:
        deterministic = float(payload.get("temperature", 0) or 0) == 0.0
        cache_key = sha256_text(
            jdump({"path": path, "base_url": self.base_url, "payload": payload}, sort_keys=True)
        )
        if deterministic:
            hit = self._cache_get(cache_key)
            if hit is not None:
                return Reply(
                    text=hit.get("text", ""),
                    usage=Usage(**hit.get("usage", {})),
                    latency_ms=float(hit.get("latency_ms", 0.0)),
                    finish_reason=hit.get("finish_reason", ""),
                    top_logprobs=hit.get("top_logprobs"),
                    cached=True,
                    raw=hit.get("raw"),
                )

        import time

        t0 = time.perf_counter()
        try:
            data = self.http.post_json(path, payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            reply = self._parse(data, latency_ms)
        except HttpError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            reply = Reply(text="", latency_ms=latency_ms, error=str(exc))

        with self._lock:
            self.calls += 1
            self.latencies_ms.append(reply.latency_ms)
            if reply.error:
                self.errors += 1
            else:
                self.usage.merge(reply.usage)

        if deterministic and reply.ok:
            self._cache_put(
                cache_key,
                {
                    "text": reply.text,
                    "usage": reply.usage.to_dict(),
                    "latency_ms": reply.latency_ms,
                    "finish_reason": reply.finish_reason,
                    "top_logprobs": reply.top_logprobs,
                    # loglikelihood 模式需要 token 级 logprob，必须一起缓存
                    "raw": reply.raw,
                },
            )
        return reply

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(data: Dict[str, Any], latency_ms: float) -> Reply:
        choices = data.get("choices") or []
        if not choices:
            return Reply(text="", latency_ms=latency_ms, error=f"响应缺少 choices：{jdump(data)[:300]}")

        choice = choices[0]
        msg = choice.get("message") or {}
        text = msg.get("content")
        if text is None:
            text = choice.get("text") or ""
        # thinking 模型的思维链可能单独放在 reasoning_content，不参与答案抽取
        finish_reason = str(choice.get("finish_reason") or "")

        usage_raw = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            total_tokens=int(usage_raw.get("total_tokens") or 0),
        )

        return Reply(
            text=str(text),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            top_logprobs=_extract_top_logprobs(choice),
            raw=data,
        )

    # ------------------------------------------------------------------ #
    def first_token_logprobs(
        self,
        messages: Union[str, Sequence[Message]],
        *,
        top_n: int = 24,
        **kw: Any,
    ) -> Optional[Dict[str, float]]:
        """返回首个生成位置上 top-N token 的 logprob（token 已 strip）。

        服务端不支持 logprobs 时返回 None，调用方应回退到生成式抽取。
        """
        if not self.supports_logprobs:
            return None
        reply = self.chat(
            messages,
            logprobs=True,
            top_logprobs=int(top_n),
            max_tokens=1,
            stop=None,
            **kw,
        )
        if not reply.ok:
            return None
        return reply.top_logprobs

    def choice_logprobs(
        self,
        messages: Union[str, Sequence[Message]],
        letters: Sequence[str],
        *,
        top_n: int = 24,
        **kw: Any,
    ) -> Optional[Dict[str, float]]:
        """对候选选项字母做似然打分，返回 {字母: logprob}。"""
        return _score_letters(self.first_token_logprobs(messages, top_n=top_n, **kw), letters)

    def choice_logprobs_reply(
        self,
        messages: Union[str, Sequence[Message]],
        letters: Sequence[str],
        *,
        top_n: int = 24,
        **kw: Any,
    ) -> Any:
        """似然打分 + 原始 Reply（用于同时采集时延与 token 统计）。

        返回 (scores | None, Reply)。scores 为 None 时调用方应回退生成式抽取。
        """
        reply = self.chat(
            messages,
            logprobs=True,
            top_logprobs=max(int(top_n), len(letters) + 8),
            max_tokens=1,
            stop=None,
            **kw,
        )
        if not reply.ok:
            return None, reply
        return _score_letters(reply.top_logprobs, letters), reply


# --------------------------------------------------------------------------- #
def _apply_overrides(payload: Dict[str, Any], **kw: Any) -> None:
    """把可选参数合并进请求体；None 表示不覆盖，沿用 params/extra_body 的默认值。"""
    stop = kw.get("stop")
    if stop:
        payload["stop"] = list(stop)
    for field in ("max_tokens", "temperature", "logprobs", "top_logprobs"):
        value = kw.get(field)
        if value is not None:
            payload[field] = int(value) if field in ("max_tokens", "logprobs", "top_logprobs") else float(value)
    extra = kw.get("extra")
    if extra:
        payload.update(extra)


def _token_logprobs(data: Optional[Dict[str, Any]]) -> List[tuple]:
    """从 /completions 的 echo 响应中取出 (token, logprob) 序列。

    首 token 的 logprob 为 null，直接跳过。
    """
    if not data:
        return []
    choices = data.get("choices") or []
    if not choices:
        return []
    lp = choices[0].get("logprobs") or {}
    tokens = lp.get("tokens") or []
    lps = lp.get("token_logprobs") or []
    out: List[tuple] = []
    for i, tok in enumerate(tokens):
        val = lps[i] if i < len(lps) else None
        out.append((str(tok), None if val is None else float(val)))
    return out


def _sum_continuation(pairs: Sequence[tuple], candidate: str) -> tuple:
    """从 token 序列末尾向前累加，直到拼出的文本等于候选续写。

    返回 (logprob 之和, token 数)。token 数用于 acc_norm 的长度归一化。

    关键细节：命中之后要**继续吸收紧邻的纯空白 token**。当续写是 " A" 而
    分词器把它切成 [" ", "A"] 时，这个空格落在 lm-eval 的「上下文 / 续写」
    切分边界上，属于续写的一部分；若不吸收，acc_norm 的归一化长度会不一致。

    找不到匹配时退化为「只取最后一个有效 token」，保证不会抛异常。
    """
    target = str(candidate).strip()
    if not target or not pairs:
        return 0.0, 0

    acc = ""
    total = 0.0
    count = 0
    limit = len(target) + 4
    for token, logprob in reversed(pairs):
        if logprob is None:
            break
        already_hit = count > 0 and acc.strip() == target
        if already_hit and token.strip():
            break  # 命中后遇到实际内容，说明已经跨过了续写边界
        acc = token + acc
        total += logprob
        count += 1
        if not already_hit and len(acc.strip()) > limit:
            break

    if count > 0 and acc.strip() == target:
        return total, count

    for _token, logprob in reversed(pairs):
        if logprob is not None:
            return logprob, 1
    return 0.0, 0


def _score_letters(
    tops: Optional[Dict[str, float]], letters: Sequence[str]
) -> Optional[Dict[str, float]]:
    """从首 token 的 top_logprobs 中挑出候选字母的分数。

    首 token 可能是 "A"、"A"（带前导空格）、"（A）"、"答案" 等，这里统一 strip
    后再比对；同一字母多次出现取最大 logprob。
    """
    if not tops:
        return None
    want = {str(x).strip().upper() for x in letters}
    out: Dict[str, float] = {}
    for token, logprob in tops.items():
        key = str(token).strip().strip("()（）[]。.:：").upper()
        if key in want:
            prev = out.get(key)
            if prev is None or logprob > prev:
                out[key] = float(logprob)
    return out or None


# --------------------------------------------------------------------------- #
def _extract_top_logprobs(choice: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """兼容两种服务端返回结构：
    chat 风格：choices[0].logprobs.content[0].top_logprobs = [{token, logprob}, ...]
    completion 风格：choices[0].logprobs.top_logprobs[0] = {token: logprob}
    """
    lp = choice.get("logprobs")
    if not lp:
        return None

    content = lp.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        tops = first.get("top_logprobs") if isinstance(first, dict) else None
        if isinstance(tops, list):
            out: Dict[str, float] = {}
            for item in tops:
                if isinstance(item, dict) and "token" in item:
                    out[str(item["token"])] = float(item.get("logprob") or 0.0)
            return out or None

    tops = lp.get("top_logprobs")
    if isinstance(tops, list) and tops and isinstance(tops[0], dict):
        return {str(k): float(v) for k, v in tops[0].items()}
    return None


def _pct(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


# --------------------------------------------------------------------------- #
def build_client(cfg: Dict[str, Any], **kw: Any) -> ModelClient:
    """工厂：provider == "mock" 时返回可离线运行的假模型客户端。"""
    provider = str(cfg.get("provider") or "openai_compat").lower()
    if provider == "mock":
        from .mock import MockClient

        return MockClient(cfg, **kw)  # type: ignore[return-value]
    if provider not in ("openai_compat", "openai", "vllm", "sglang", "tgi", "lmdeploy"):
        raise ValueError(f"未知 provider：{provider}（支持 openai_compat / mock）")
    return ModelClient(cfg, **kw)
