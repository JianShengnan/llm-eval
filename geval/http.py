"""极简 HTTP 传输层：纯标准库 urllib，内建重试、退避、代理策略。

为什么不用 requests：
  内网 ARM64 离线环境希望零第三方依赖，`python -m geval selftest` 开箱即用。

代理策略（proxy 字段）是本模块最容易被忽略但最关键的部分：
  服务器上常配置 HTTP_PROXY / HTTPS_PROXY 指向外网代理。urllib 默认读取这些
  环境变量，于是发往内网模型服务的请求会被送到外网代理，表现为「超时」或
  「502 Bad Gateway」。因此默认值 "bypass" 会显式清空代理。
"""

from __future__ import annotations

import json
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


class HttpError(RuntimeError):
    """统一包装的 HTTP / 网络错误。status=0 表示连接层失败。"""

    def __init__(self, status: int, body: str, url: str = ""):
        self.status = status
        self.body = body or ""
        self.url = url
        super().__init__(f"HTTP {status} {url} :: {self.body[:400]}")

    @property
    def retryable(self) -> bool:
        return self.status == 0 or self.status in RETRY_STATUS


def build_opener(proxy: str = "bypass", *, ssl_unverified: bool = False) -> urllib.request.OpenerDirector:
    handlers: list = []
    mode = (proxy or "bypass").strip()
    if mode == "env":
        pass  # 交给 urllib 默认行为（读取环境变量）
    elif mode == "bypass":
        handlers.append(urllib.request.ProxyHandler({}))
    else:
        handlers.append(urllib.request.ProxyHandler({"http": mode, "https": mode}))

    if ssl_unverified:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=ctx))

    return urllib.request.build_opener(*handlers)


class HttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 180.0,
        retries: int = 3,
        proxy: str = "bypass",
        headers: Optional[Dict[str, str]] = None,
        ssl_unverified: bool = False,
        backoff_base: float = 1.5,
        backoff_cap: float = 20.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.headers: Dict[str, str] = {"User-Agent": "geval/0.1"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self.headers.update({k: str(v) for k, v in headers.items()})
        self.opener = build_opener(proxy, ssl_unverified=ssl_unverified)
        self.stats = {"requests": 0, "retries": 0, "errors": 0}

    # ------------------------------------------------------------------ #
    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _sleep(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None and retry_after > 0:
            delay = min(self.backoff_cap, retry_after)
        else:
            delay = min(self.backoff_cap, self.backoff_base ** attempt)
        time.sleep(delay + random.uniform(0, 0.35))

    def request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = self.url(path)
        max_retries = self.retries if retries is None else max(0, int(retries))
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_err: Optional[HttpError] = None
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=body, method=method)
            for k, v in self.headers.items():
                req.add_header(k, v)
            if body is not None:
                req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")

            self.stats["requests"] += 1
            try:
                with self.opener.open(req, timeout=timeout or self.timeout) as resp:
                    raw = resp.read()
                text = raw.decode("utf-8", "replace")
                if not text.strip():
                    return {}
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise HttpError(resp.status, f"响应不是合法 JSON：{text[:400]}", url) from exc

            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    detail = ""
                err = HttpError(exc.code, detail or exc.reason or "", url)
                status = err.status
                if status in RETRY_STATUS and attempt < max_retries:
                    self.stats["retries"] += 1
                    last_err = err
                    self._sleep(attempt, _retry_after(exc))
                    continue
                self.stats["errors"] += 1
                raise err

            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionResetError, OSError) as exc:
                err = HttpError(0, f"{type(exc).__name__}: {exc}", url)
                if attempt < max_retries:
                    self.stats["retries"] += 1
                    last_err = err
                    self._sleep(attempt, None)
                    continue
                self.stats["errors"] += 1
                raise err

        raise last_err or HttpError(0, "unknown transport failure", url)

    def get_json(self, path: str, **kw: Any) -> Dict[str, Any]:
        return self.request_json(path, method="GET", **kw)

    def post_json(self, path: str, payload: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
        return self.request_json(path, method="POST", payload=payload, **kw)


def _retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
    raw = exc.headers.get("Retry-After") if exc.headers else None
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def is_local(url: str) -> bool:
    return any(h in (url or "") for h in LOCAL_HOSTS)
