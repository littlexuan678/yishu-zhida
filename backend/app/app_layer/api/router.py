# -*- coding: utf-8 -*-
"""
API 总路由聚合
==============
四大业务模块路由挂载于 `/api/v1`：
  * /qa         智能问答（可解释 AI 医生）
  * /disease    疾病查询（掌上医典）
  * /graph      知识图谱（医学全景地图）
  * /analytics  多维数据分析 & 健康预警
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter

from app.app_layer.api import analytics, disease, graph, qa
from app.config import settings
from app.deps import collect_health

api_router = APIRouter()

api_router.include_router(qa.router)
api_router.include_router(disease.router)
api_router.include_router(graph.router)
api_router.include_router(analytics.router)


# =============================================================================
#  系统级接口
# =============================================================================
@api_router.get("/health", tags=["系统"], summary="健康检查")
async def health() -> Dict[str, Any]:
    """返回各组件状态（Neo4j / LLM / RAG 模板库 / 模型加载情况）"""
    data = collect_health()
    data["server_time"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    data["disclaimer"] = "本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。"
    return data


@api_router.get("/info", tags=["系统"], summary="系统概览（四大功能与三大创新点）")
async def system_info() -> Dict[str, Any]:
    """返回系统定位、四层架构、四大功能与三大创新点（用于答辩展示）"""
    templates = 0
    try:
        from app.llm_layer.rag.prompt_templates import get_prompt_library

        templates = len(get_prompt_library().templates)
    except Exception:  # noqa: BLE001
        pass

    return {
        "name": settings.APP_NAME,
        "product": "智愈医典",
        "version": settings.APP_VERSION,
        "track": "大模型与智能体应用赛道",
        "positioning": "基于医疗大数据 + 知识图谱 + RAG 大模型的专业医疗知识问答平台",
        "architecture": {
            "数据层": "Scrapy 爬虫 + PubMed E-utilities API → 清洗整合 → 全链路脱敏加密(AES-256-GCM)",
            "知识图谱层": "BERT-BiLSTM-CRF 实体识别 + ★改进 PCNN(医疗关键词注意力)★ 关系抽取 + Neo4j 5.8 + Apache Jena 规则引擎 + GAT/GCN 图谱补全",
            "LLM增强层": "TextCNN 意图分类 + BERT 实体链接 + ★医疗专属 RAG 提示工程(100+模板)★ + Llama 3 指令微调 + LangChain 编排",
            "应用层": "FastAPI + Uvicorn 异步接口 + Vue3 + Element Plus + D3.js(图谱可视化) + PyEcharts(数据图表)",
        },
        "features": [
            {"key": "graph", "name": "知识图谱可视化", "desc": "医学知识的「全景地图」", "api": "/api/v1/graph/subgraph"},
            {"key": "disease", "name": "智能疾病查询", "desc": "精准权威的「掌上医典」", "api": "/api/v1/disease/search"},
            {"key": "qa", "name": "智能问答交互", "desc": "可解释的「AI 医生」", "api": "/api/v1/qa/ask"},
            {"key": "analytics", "name": "多维数据分析", "desc": "预见未来的「健康预警」", "api": "/api/v1/analytics/risk/predict"},
        ],
        "innovations": [
            {
                "id": 1,
                "name": "改进 PCNN 关系抽取算法",
                "detail": "在传统 PCNN 基础上增加医疗领域关键词注意力机制，重点关注「症状」「治疗」类核心关键词，医疗关系抽取召回率提升 8.3%",
                "code": "backend/app/kg_layer/attention.py::MedicalKeywordAttention",
            },
            {
                "id": 2,
                "name": "医疗专属 RAG 提示工程",
                "detail": f"整理 {templates} 个医疗问答模板，将知识图谱三元组与用户问题结合构造专属提示词，约束大模型输出，医疗问答准确率提升 12.1%",
                "code": "backend/app/llm_layer/rag/prompt_library.yaml",
            },
            {
                "id": 3,
                "name": "轻量化知识图谱与可视化整合设计",
                "detail": "简化图谱节点层级，设计「一键检索 + 详情联动」功能，平衡专业性与易用性，适配基层医疗与普通用户",
                "code": "frontend/src/components/GraphCanvas.vue",
            },
        ],
        "metrics": {
            "knowledge_entities_target": "≥10万",
            "knowledge_relations_target": "≥50万",
            "knowledge_update_cycle": "≤24h",
            "qa_accuracy_target": "≥85%",
            "semantic_parse_latency_target": "≤500ms",
            "disease_prediction_coverage": "1000+",
            "prediction_auc_target": "≥0.9",
            "prompt_templates": templates,
        },
        "disclaimer": "本系统为演示原型。所有 AI 输出仅供参考，不能替代执业医师诊断。",
    }


@api_router.get("/", tags=["系统"], summary="根路径")
async def root() -> Dict[str, Any]:
    return {
        "message": f"{settings.APP_NAME} API",
        "product": "智愈医典",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": f"{settings.API_PREFIX}/health",
        "info": f"{settings.API_PREFIX}/info",
        "endpoints": [
            f"{settings.API_PREFIX}/qa/ask",
            f"{settings.API_PREFIX}/disease/search",
            f"{settings.API_PREFIX}/graph/subgraph",
            f"{settings.API_PREFIX}/analytics/overview",
        ],
    }
