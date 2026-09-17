"""reqBody 包装网关的模型客户端（provider = "reqbody_gateway"）。

现网 Qwen3-32B 网关（…/model/v1/chat/completions）与标准 OpenAI 协议的差异：
  1. 整个请求体要包进 {"reqBody": {...}} 外壳；
  2. 鉴权不走 Authorization 请求头，而是 reqBody 内的 authorization 字段。

适配策略：只替换传输层——HttpClient.post_json 负责出站包装与入站解壳，
ModelClient 的缓存、统计、重试、解析逻辑全部复用原实现。缓存 key 在包装
之前计算，外壳不影响缓存命中。

可选配置键（仅本 provider 识别）：
  user_suffix   追加到每条请求最后一条 user 消息末尾的字符串，
                如 Qwen3 软开关 "/no_think"（关不关思考由配置决定）。

响应解壳：部分网关会把标准响应再包一层（respBody / data / body / result）。
若顶层已有 choices 则原样返回，不额外处理。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from .client import Message, ModelClient
from .http import HttpClient

WRAP_KEY = "reqBody"
AUTH_KEY = "authorization"
RESP_UNWRAP_KEYS = ("respBody", "data", "body", "result")


def _auth_value(api_key: Any) -> Optional[str]:
    if not api_key:
        return None
    s = str(api_key)
    return s if s.lower().startswith("bearer ") else f"Bearer {s}"


def _unwrap(data: Any) -> Any:
    """剥响应外壳：顶层没有 choices 时，尝试常见外壳键下钻一层。"""
    if not isinstance(data, dict) or "choices" in data:
        return data
    for key in RESP_UNWRAP_KEYS:
        inner = data.get(key)
        if isinstance(inner, dict) and "choices" in inner:
            return inner
    if len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, dict) and "choices" in only:
            return only
    return data


class ReqBodyHttpClient(HttpClient):
    """出站：包 reqBody 外壳、鉴权放 body、不发 Authorization 头；入站：解壳。"""

    def __init__(self, base_url: str, *, api_key: Optional[str] = None, **kw: Any) -> None:
        # 鉴权必须放在 body 内，这里显式不传 api_key，避免父类生成 Authorization 头
        super().__init__(base_url, api_key=None, **kw)
        self.auth_value = _auth_value(api_key)

    def post_json(self, path: str, payload: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
        body = dict(payload)
        body.setdefault("stream", False)
        if self.auth_value:
            body.setdefault(AUTH_KEY, self.auth_value)
        data = super().post_json(path, {WRAP_KEY: body}, **kw)
        return _unwrap(data)


class ReqBodyClient(ModelClient):
    """provider = "reqbody_gateway" 的模型客户端：仅替换传输层。"""

    def __init__(self, cfg: Dict[str, Any], **kw: Any) -> None:
        super().__init__(cfg, **kw)
        suffix = cfg.get("user_suffix")
        self.user_suffix: Optional[str] = str(suffix) if suffix else None
        self.http = ReqBodyHttpClient(
            self.base_url,
            api_key=cfg.get("api_key"),
            timeout=float(kw.get("timeout", 180.0)),
            retries=int(kw.get("retries", 3)),
            proxy=str(kw.get("proxy", "bypass")),
            headers=cfg.get("headers"),
            ssl_unverified=bool(cfg.get("ssl_unverified", False)),
        )

    def _build_messages(self, messages: Union[str, Sequence[Message]]) -> List[Message]:
        msgs = super()._build_messages(messages)
        if self.user_suffix:
            for m in reversed(msgs):
                if m.get("role") == "user":
                    m["content"] = f"{m['content']}{self.user_suffix}"
                    break
        return msgs
