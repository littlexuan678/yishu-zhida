# -*- coding: utf-8 -*-
"""
耗时埋点工具
============
用于度量 PPT 中的关键性能指标：
  * 语义解析响应 ≤ 500ms
  * 各阶段耗时写入 QAResponse.latency_ms，前端可解释面板可直接展示
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator

from app.utils.logger import get_logger

logger = get_logger(__name__)


class Stopwatch:
    """可分段计时的秒表"""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._marks: Dict[str, float] = {}
        self._last = self._t0

    def mark(self, name: str) -> float:
        """记录自上一个标记（或起点）到现在的耗时（毫秒），并累计到该名字"""
        now = time.perf_counter()
        elapsed = (now - self._last) * 1000.0
        self._marks[name] = round(self._marks.get(name, 0.0) + elapsed, 2)
        self._last = now
        return round(elapsed, 2)

    def lap(self, name: str) -> float:
        """记录自起点到现在的累计耗时（毫秒），不重置 _last"""
        elapsed = (time.perf_counter() - self._t0) * 1000.0
        self._marks[name] = round(elapsed, 2)
        return round(elapsed, 2)

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 2)

    @property
    def marks(self) -> Dict[str, float]:
        return dict(self._marks)


@contextmanager
def timed(label: str, store: Dict[str, float] | None = None) -> Iterator[None]:
    """上下文管理器：with timed('llm', marks): ..."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        cost = round((time.perf_counter() - t0) * 1000.0, 2)
        if store is not None:
            store[label] = round(store.get(label, 0.0) + cost, 2)
        logger.debug("[timed] %s = %.2f ms", label, cost)
