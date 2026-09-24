"""通用工具：JSON / JSONL 读写、哈希、线程安全写入器、格式化。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import unicodedata
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# 时间 / 哈希
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    """本地时间 ISO 字符串（秒精度），用于 manifest。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def seed_from(text: str) -> int:
    """把任意字符串映射成稳定整数种子，用于按样本 id 做确定性随机。"""
    return int(sha256_text(text)[:12], 16)


# --------------------------------------------------------------------------- #
# 目录 / 文件
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def jdump(obj: Any, *, indent: Optional[int] = None, sort_keys: bool = False) -> str:
    """统一的 JSON 序列化：不转义中文、失败兜底为 str。"""
    return json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys, default=str)


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any, *, indent: int = 2) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(jdump(obj, indent=indent))
        f.write("\n")
    return path


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """逐行读取 JSONL；空行与 # 注释行跳过；解析失败时报出准确行号。"""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} 期望 JSON 对象，实际为 {type(obj).__name__}")
            yield obj


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]], *, append: bool = False) -> int:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    count = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(jdump(row) + "\n")
            count += 1
    return count


class JsonlWriter:
    """线程安全的增量 JSONL 写入器（逐条 flush，保证中断可续跑）。"""

    def __init__(self, path: str, *, append: bool = False) -> None:
        self.path = path
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        self._lock = threading.Lock()
        self._fh = open(path, "a" if append else "w", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        line = jdump(row) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# 计数 / 格式化
# --------------------------------------------------------------------------- #
class Progress:
    """极简进度打印器：无 tqdm 依赖，**条数 + 时间双节流**。

    为什么必须加时间节流：早年只按 `done % every == 0` 判断，而 every 由
    `总样本数 // 20` 推出 —— CMMLU 11582 条算出来是 579，意味着前 579 条跑完之前
    一行都不打印。并发 8、单请求 3s 时要等三分钟才看到第一行，观感就是
    「这个数据集没有进度条 / 卡死了」。现在保底每 `min_interval` 秒刷新一次。

    另一半问题是 `\r` 原地刷新在**非终端**（重定向进日志、IDE 输出面板、CI）里
    不会换行，几百次刷新全部叠在同一行，日志里等于没有进度。因此检测到
    输出不是 TTY 时自动切成逐行输出。
    """

    def __init__(
        self,
        total: int,
        label: str = "",
        every: int = 20,
        stream=None,
        line_mode: bool = False,
        min_interval: float = 0.0,
    ) -> None:
        import sys

        self.total = max(0, int(total))
        self.label = label
        self.every = max(1, int(every))
        self.stream = stream or sys.stderr
        self.done = 0
        self._lock = threading.Lock()
        self._t0 = time.time()
        self.min_interval = max(0.0, float(min_interval))
        self._last_render = self._t0
        # line_mode：多任务并行、或输出不是终端时改为换行输出
        if not line_mode:
            try:
                line_mode = not self.stream.isatty()
            except Exception:  # noqa: BLE001  # 无 isatty（如被包装的流）时保守地逐行输出
                line_mode = True
        self.line_mode = bool(line_mode)

    def kickoff(self) -> None:
        """任务刚开跑就渲染一次 0/N，立刻让人知道这批已经启动。"""
        with self._lock:
            self._render()

    def tick(self, n: int = 1) -> None:
        with self._lock:
            self.done += n
            due = bool(self.total) and (self.done % self.every == 0 or self.done >= self.total)
            stale = self.min_interval > 0 and (time.time() - self._last_render) >= self.min_interval
            if due or stale:
                self._render()

    def _render(self) -> None:
        elapsed = max(1e-6, time.time() - self._t0)
        rate = self.done / elapsed
        pct = (100.0 * self.done / self.total) if self.total else 0.0
        eta = ((self.total - self.done) / rate) if rate > 0 and self.total else 0.0
        self.stream.write(
            ("" if self.line_mode else "\r")
            + f"  {self.label} {self.done}/{self.total} ({pct:5.1f}%) "
            f"{rate:5.1f}/s ETA {eta:6.0f}s   "
            + ("\n" if self.line_mode else "")
        )
        self._last_render = time.time()
        self.stream.flush()

    def finish(self) -> None:
        with self._lock:
            self._render()
            if not self.line_mode:
                self.stream.write("\n")
                self.stream.flush()


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f}%"


def display_width(text: Any) -> int:
    """按终端显示列宽计算字符串宽度：CJK 全角字符占 2 列。

    直接用 len() 会把「样本数」算成 3 列而不是 6 列，导致中文表头错位。
    """
    total = 0
    for ch in str(text):
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def pad(text: Any, width: int, *, align: str = "left") -> str:
    """按显示列宽补齐，用于终端表格对齐。"""
    text = str(text)
    space = max(0, int(width) - display_width(text))
    if align == "right":
        return " " * space + text
    if align == "center":
        left = space // 2
        return " " * left + text + " " * (space - left)
    return text + " " * space


def fmt_num(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def truncate(text: str, limit: int = 240) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], aligns: Optional[Sequence[str]] = None) -> str:
    """生成 Markdown 表格。aligns 取 'l' / 'r' / 'c'。"""
    headers = [str(h) for h in headers]
    aligns = list(aligns or ["l"] * len(headers))
    while len(aligns) < len(headers):
        aligns.append("l")
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out: List[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(sep.get(a, ":---") for a in aligns[: len(headers)]) + " |")
    for row in rows:
        cells = [("" if c is None else str(c)).replace("\n", "<br>").replace("|", "\\|") for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
