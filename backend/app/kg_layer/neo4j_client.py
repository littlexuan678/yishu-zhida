# -*- coding: utf-8 -*-
"""
Neo4j 客户端封装（Neo4j 5.8 社区版）
====================================
设计要点
--------
1. **驱动懒加载**：应用启动时尝试连接 Neo4j；若失败且 `NEO4J_FALLBACK_TO_MEMORY=true`，
   自动降级到 `memory_store.MemoryGraphStore`，保证系统在无 Neo4j 环境下仍可完整演示。
2. **异步执行**：所有查询在 `asyncio.to_thread` 中执行，避免阻塞 FastAPI 事件循环
   （neo4j 同步驱动 + 线程池 = 轻量异步，无需额外引入 async 驱动复杂度）。
3. **只读白名单**：`execute_readonly()` 会对 Cypher 做写操作关键字校验，
   防止 `/graph/cypher` 接口被用于篡改图谱。
4. **统一返回**：查询结果统一为 `(columns, rows: List[dict])`，与内存实现保持一致。
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
#  只读校验
# ---------------------------------------------------------------------------
_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|CALL\s+\{)\b",
    re.IGNORECASE,
)
#: 允许的只读过程（APOC 白名单）
_ALLOWED_PROC = re.compile(r"\bapoc\.(meta|path|text|coll|number)\.", re.IGNORECASE)


class CypherSecurityError(ValueError):
    """Cypher 语句违反只读约束"""


def assert_readonly(cypher: str) -> None:
    """校验 Cypher 为只读语句，否则抛 CypherSecurityError"""
    stmt = (cypher or "").strip()
    if not stmt:
        raise CypherSecurityError("Cypher 语句为空")
    # 允许多语句分隔，但每一段都不能含写关键字
    for seg in stmt.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        if not re.match(r"^(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN|CALL|USE|EXPLAIN|PROFILE|SHOW)", seg, re.IGNORECASE):
            raise CypherSecurityError("仅允许以 MATCH / WITH / UNWIND / RETURN / CALL / SHOW 开头的只读查询")
        m = _WRITE_KEYWORDS.search(seg)
        if m:
            raise CypherSecurityError(f"检测到写操作关键字：{m.group(0)}，本接口仅支持只读查询")
        if "call" in seg.lower() and not _ALLOWED_PROC.search(seg):
            raise CypherSecurityError("CALL 仅允许白名单内的 APOC 只读过程")


# ---------------------------------------------------------------------------
#  连接管理器
# ---------------------------------------------------------------------------
class Neo4jClient:
    """Neo4j 驱动封装 + 内存降级"""

    def __init__(self) -> None:
        self._driver = None
        self._available: bool = False
        self._fallback = None          # MemoryGraphStore（降级时使用）
        self._error: str = ""
        self._connect()

    # ----------------------------- 连接 -----------------------------
    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError:
            self._error = "未安装 neo4j 驱动（pip install neo4j==5.19.0）"
            logger.warning("Neo4j 驱动缺失：%s", self._error)
            self._activate_fallback()
            return

        try:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
                connection_timeout=settings.NEO4J_CONNECTION_TIMEOUT,
            )
            self._driver.verify_connectivity()
            self._available = True
            logger.info("Neo4j 连接成功：%s (database=%s)", settings.NEO4J_URI, settings.NEO4J_DATABASE)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            logger.warning("Neo4j 连接失败：%s", exc)
            self._driver = None
            self._activate_fallback()

    def _activate_fallback(self) -> None:
        if not settings.NEO4J_FALLBACK_TO_MEMORY:
            logger.error("Neo4j 不可用且未开启内存降级，图谱接口将返回空数据")
            return
        from app.kg_layer.memory_store import MemoryGraphStore

        self._fallback = MemoryGraphStore.instance()
        logger.warning("已降级到内存图存储（离线演示模式），节点数=%d", self._fallback.node_count)

    # ----------------------------- 状态 -----------------------------
    @property
    def available(self) -> bool:
        return self._available

    @property
    def status(self) -> str:
        if self._available:
            return "connected"
        if self._fallback is not None:
            return "fallback_memory"
        return "error"

    @property
    def error(self) -> str:
        return self._error

    @property
    def using_fallback(self) -> bool:
        return (not self._available) and self._fallback is not None

    # ----------------------------- 同步查询 -----------------------------
    def _run_sync(self, cypher: str, params: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        assert self._driver is not None
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(cypher, params or {})
            columns = list(result.keys())
            rows = [dict(record) for record in result]
        return columns, rows

    # ----------------------------- 异步入口 -----------------------------
    async def query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        readonly: bool = True,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """执行 Cypher；Neo4j 不可用时自动路由到内存实现"""
        params = params or {}
        if readonly and settings.CYPHER_READONLY_WHITELIST:
            assert_readonly(cypher)

        if self._available and self._driver is not None:
            try:
                return await asyncio.to_thread(self._run_sync, cypher, params)
            except Exception as exc:  # noqa: BLE001
                logger.error("Cypher 执行失败：%s | %s", exc, cypher[:200])
                if not settings.NEO4J_FALLBACK_TO_MEMORY:
                    raise
                self._available = False
                self._error = str(exc)
                self._activate_fallback()

        if self._fallback is not None:
            return self._fallback.query(cypher, params)

        return [], []

    # ----------------------------- 便捷方法 -----------------------------
    async def single(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        _, rows = await self.query(cypher, params)
        return rows[0] if rows else None

    async def scalar(self, cypher: str, params: Optional[Dict[str, Any]] = None, default: Any = 0) -> Any:
        row = await self.single(cypher, params)
        if not row:
            return default
        return next(iter(row.values()), default)

    async def write(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """写操作（仅供 scripts/ 与 pipelines 使用，API 层不暴露）"""
        return await self.query(cypher, params, readonly=False)

    # ----------------------------- 生命周期 -----------------------------
    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:  # noqa: BLE001
                pass
        self._driver = None
        self._available = False

    def ping(self) -> bool:
        """同步探活（供 /health 使用）"""
        if not self._available or self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client


def reset_neo4j_client() -> None:
    """测试用：重置单例"""
    global _client
    if _client is not None:
        _client.close()
    _client = None
