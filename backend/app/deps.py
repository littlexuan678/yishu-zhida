# -*- coding: utf-8 -*-
"""
依赖注入（Dependency Injection）
================================
FastAPI 的 `Depends` 提供者统一在此定义，优点：
  * 服务实例全局单例，避免重复加载模型/建立连接
  * 测试时可通过 `app.dependency_overrides` 轻松替换
  * 服务初始化失败不会阻塞应用启动（接口内部再做降级）
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from app.config import settings
from app.kg_layer.graph_service import GraphService, get_graph_service
from app.kg_layer.graph_reasoner import GraphReasoner, get_graph_reasoner
from app.kg_layer.neo4j_client import Neo4jClient, get_neo4j_client
from app.llm_layer.entity_linker import EntityLinker, get_entity_linker
from app.llm_layer.intent_classifier import IntentClassifier, get_intent_classifier
from app.llm_layer.llm_client import LLMClient, get_llm_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
#  知识图谱层
# ---------------------------------------------------------------------------
def get_graph() -> GraphService:
    """图谱业务服务（单例）"""
    return get_graph_service()


def get_reasoner() -> GraphReasoner:
    """图谱推理与补全（单例）"""
    return get_graph_reasoner()


def get_client() -> Neo4jClient:
    """Neo4j 客户端（单例）"""
    return get_neo4j_client()


# ---------------------------------------------------------------------------
#  LLM 增强层
# ---------------------------------------------------------------------------
def get_intent() -> IntentClassifier:
    """TextCNN 意图分类器（单例）"""
    return get_intent_classifier()


def get_linker() -> EntityLinker:
    """实体链接器（单例）"""
    return get_entity_linker()


def get_llm() -> LLMClient:
    """大模型客户端（单例）"""
    return get_llm_client()


def get_rag():
    """
    RAG 引擎（单例，延迟导入避免循环依赖）

    注意：`rag_engine` 会导入 retriever/context_builder/hallucination_guard，
    因此这里采用函数内导入。
    """
    from app.llm_layer.rag.rag_engine import get_rag_engine

    return get_rag_engine()


# ---------------------------------------------------------------------------
#  系统状态快照（供 /health 使用）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _models_status_cache() -> Dict[str, bool]:
    return {}


def collect_health() -> Dict[str, Any]:
    """收集各组件健康状态（任何子组件异常都不影响本函数返回）"""
    status: Dict[str, Any] = {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "ok",
        "neo4j": "unknown",
        "llm": "unknown",
        "rag_templates": 0,
        "models_loaded": {},
    }

    # Neo4j
    try:
        client = get_neo4j_client()
        status["neo4j"] = client.status
        status["neo4j_uri"] = settings.NEO4J_URI
    except Exception as exc:  # noqa: BLE001
        logger.warning("健康检查 - Neo4j 异常：%s", exc)
        status["neo4j"] = "error"

    # LLM
    try:
        status["llm"] = get_llm_client().provider
    except Exception as exc:  # noqa: BLE001
        logger.warning("健康检查 - LLM 异常：%s", exc)
        status["llm"] = "error"

    # RAG 模板库
    try:
        from app.llm_layer.rag.prompt_templates import get_prompt_library

        lib = get_prompt_library()
        status["rag_templates"] = len(lib.templates)
        status["rag_validate"] = lib.validate()
    except Exception as exc:  # noqa: BLE001
        logger.warning("健康检查 - Prompt 模板库异常：%s", exc)
        status["rag_templates"] = 0

    # 模型加载状态
    try:
        from app.kg_layer.entity_extractor import get_entity_extractor
        from app.kg_layer.relation_extractor import get_relation_extractor

        status["models_loaded"] = {
            "ner_bert_bilstm_crf": get_entity_extractor().backend == "bert_bilstm_crf",
            "pcnn_medical_attention": get_relation_extractor().backend == "improved_pcnn",
            "intent_textcnn": get_intent_classifier().backend == "textcnn",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("健康检查 - 模型状态异常：%s", exc)

    return status
