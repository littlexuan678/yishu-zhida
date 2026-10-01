# -*- coding: utf-8 -*-
"""
疾病查询接口（掌上医典）
========================
对应 PPT 功能二：智能疾病查询
  * 关键词 / 自然语言搜索 → `GET /disease/search`
  * 卡片式返回（定义/病因/症状/诊断/治疗/预后）→ `GET /disease/{id}`
  * 支持增量更新 → `POST /disease/{id}/refresh-source`
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.deps import get_graph
from app.kg_layer.graph_service import GraphService
from app.schemas import (
    DiseaseDetail,
    DiseaseSearchResponse,
    RelatedDiseaseResponse,
    SourceItem,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/disease", tags=["疾病查询 · 掌上医典"])


# =============================================================================
#  GET /disease/hot-keywords —— 热门搜索词
#  注意：必须放在 /{disease_id} 之前，否则会被路径参数吞掉
# =============================================================================
@router.get(
    "/hot-keywords",
    response_model=List[str],
    summary="热门搜索关键词",
    description="返回 PPT 演示中的热门搜索词（感冒/高血压/糖尿病/冠心病/肺炎/胃炎…）。",
)
async def hot_keywords(
    topn: int = Query(6, ge=1, le=20, description="返回条数"),
    graph: GraphService = Depends(get_graph),
) -> List[str]:
    return graph.hot_keywords(topn)


# =============================================================================
#  GET /disease/search —— 疾病检索
# =============================================================================
@router.get(
    "/search",
    response_model=DiseaseSearchResponse,
    summary="疾病检索（关键词 / 自然语言）",
    description=(
        "支持多维命中并按相关度排序：疾病名 > 别名 > 症状 > 分类/科室 > 定义/病因 > 模糊匹配。\n\n"
        "返回卡片式字段：一级分类、二级分类、症状、治疗、易感人群（与 PPT 界面一致）。"
    ),
)
async def search_disease(
    q: str = Query("", description="搜索关键词，如 '高血压' 或 '头晕 血压高'", max_length=120),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(6, ge=1, le=60, description="每页条数"),
    category1: Optional[str] = Query(None, description="按一级分类过滤，如 '内科'"),
    graph: GraphService = Depends(get_graph),
) -> DiseaseSearchResponse:
    try:
        total, items = graph.search_diseases(q, page=page, page_size=page_size, category1=category1)
        message = f"找到 {total} 条相关疾病信息" if q else f"共收录 {total} 条疾病信息"
        return DiseaseSearchResponse(
            keyword=q, total=total, page=page, page_size=page_size,
            items=items, hot_keywords=graph.hot_keywords(),
            message=message,
            searched_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("疾病检索异常：%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"疾病检索失败：{exc}") from exc


# =============================================================================
#  GET /disease/{disease_id} —— 疾病详情
# =============================================================================
@router.get(
    "/{disease_id}",
    response_model=DiseaseDetail,
    summary="疾病详情（定义/病因/症状/诊断/治疗/预后）",
    description="`disease_id` 可传疾病 ID（D0001）或疾病名称（原发性高血压）。返回 6 大结构化板块 + 权威来源溯源。",
)
async def disease_detail(
    disease_id: str = Path(..., description="疾病 ID 或疾病名称", max_length=80),
    graph: GraphService = Depends(get_graph),
) -> DiseaseDetail:
    detail = graph.get_disease_detail(disease_id)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"知识库中未找到疾病「{disease_id}」。请尝试使用标准疾病名称，如「原发性高血压」。",
        )
    return detail


# =============================================================================
#  GET /disease/{disease_id}/related —— 关联疾病
# =============================================================================
@router.get(
    "/{disease_id}/related",
    response_model=RelatedDiseaseResponse,
    summary="关联疾病（并发症 / 鉴别诊断）",
)
async def related_disease(
    disease_id: str = Path(..., description="疾病 ID 或疾病名称", max_length=80),
    graph: GraphService = Depends(get_graph),
) -> RelatedDiseaseResponse:
    resp = graph.get_related_diseases(disease_id)
    if not resp.disease_id and not resp.complications and not resp.differential:
        raise HTTPException(status_code=404, detail=f"知识库中未找到疾病「{disease_id}」")
    return resp


# =============================================================================
#  GET /disease/{disease_id}/sources —— 知识溯源
# =============================================================================
@router.get(
    "/{disease_id}/sources",
    response_model=List[SourceItem],
    summary="疾病知识来源（100% 可溯源）",
    description="返回该疾病条目挂载的全部权威来源（临床指南 / PubMed 文献 / 教材），含 PMID 与 DOI。",
)
async def disease_sources(
    disease_id: str = Path(..., description="疾病 ID 或疾病名称", max_length=80),
    graph: GraphService = Depends(get_graph),
) -> List[SourceItem]:
    sources = graph.sources_of(disease_id)
    if not sources:
        # 用 detail 再尝试一次（支持别名/模糊）
        detail = graph.get_disease_detail(disease_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"知识库中未找到疾病「{disease_id}」")
        return detail.sources
    return sources


# =============================================================================
#  GET /disease/{disease_id}/facts —— 疾病三元组事实（图谱视图）
# =============================================================================
@router.get(
    "/{disease_id}/facts",
    summary="疾病三元组事实（按关系类型聚合）",
    description="返回该疾病在图谱中的全部直接关系（症状/治疗/用药/科室/检查/并发症/鉴别/易感人群），供疾病详情页的图谱联动使用。",
)
async def disease_facts(
    disease_id: str = Path(..., description="疾病 ID 或疾病名称", max_length=80),
    graph: GraphService = Depends(get_graph),
) -> Dict[str, Any]:
    detail = graph.get_disease_detail(disease_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"知识库中未找到疾病「{disease_id}」")
    facts = graph.facts_for_disease(detail.name)
    return {
        "disease_id": detail.disease_id,
        "name": detail.name,
        "facts": facts,
        "triple_count": sum(len(v) for v in facts.values()),
    }


# =============================================================================
#  POST /disease/{disease_id}/refresh-source —— 增量更新（≤24h 知识更新周期）
# =============================================================================
@router.post(
    "/{disease_id}/refresh-source",
    summary="触发知识增量更新（爬虫 → 抽取 → 写图）",
    description=(
        "对应 PPT「知识更新周期 ≤24h」指标。\n\n"
        "生产环境会：① 用 PubMed API 增量拉取该疾病最新文献 → "
        "② 清洗脱敏 → ③ BERT-BiLSTM-CRF + 改进 PCNN 抽取三元组 → "
        "④ MERGE 幂等写入 Neo4j（不覆盖既有高置信度事实）。\n\n"
        "本演示原型返回更新任务编排说明，不实际发起外网请求。"
    ),
)
async def refresh_source(
    disease_id: str = Path(..., description="疾病 ID 或疾病名称", max_length=80),
    graph: GraphService = Depends(get_graph),
) -> Dict[str, Any]:
    detail = graph.get_disease_detail(disease_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"知识库中未找到疾病「{disease_id}」")
    return {
        "code": 0,
        "message": "知识增量更新任务已编排",
        "disease": detail.name,
        "pipeline": [
            "1. Scrapy/PubMed E-utilities 增量采集（关键词：%s，时间范围：近 24h 新入库）" % detail.name,
            "2. 数据清洗与去重（preprocess.py）",
            "3. 全链路脱敏加密（privacy.py，AES-256-GCM）",
            "4. 实体识别（BERT-BiLSTM-CRF）",
            "5. 关系抽取（★改进 PCNN + 医疗关键词注意力★）",
            "6. MERGE 幂等写图（Neo4j，保留高置信度既有事实）",
            "7. 图谱补全（Apache Jena 规则 + GAT/GCN 链接预测）",
        ],
        "target_cycle": "≤24h",
        "cli": "python scripts/crawler_run.py --since 1d && python scripts/load_kg.py --incremental",
        "note": "演示原型未实际发起外网采集请求。",
    }
