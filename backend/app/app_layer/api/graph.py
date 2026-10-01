# -*- coding: utf-8 -*-
"""
知识图谱接口（医学全景地图）
============================
对应 PPT 功能一：知识图谱可视化模块
  * 展示疾病/症状/药物/科室/治疗方法等实体网络 → `GET /graph/subgraph`
  * 支持输入实体名称检索图谱 → `GET /graph/search`
  * 节点图例区分实体类型 → `GET /graph/entity-types`
  * 统计节点数量、关系数量 → `GET /graph/stats`
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.deps import get_reasoner, get_graph
from app.kg_layer.graph_reasoner import GraphReasoner
from app.kg_layer.graph_service import GraphService
from app.kg_layer.neo4j_client import CypherSecurityError
from app.schemas import (
    CypherRequest,
    CypherResponse,
    EntitySearchResponse,
    EntityTypeMeta,
    EvidenceItem,
    GraphStatsResponse,
    SubGraphResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/graph", tags=["知识图谱 · 医学全景地图"])


# =============================================================================
#  GET /graph/stats —— 图谱统计
# =============================================================================
@router.get(
    "/stats",
    response_model=GraphStatsResponse,
    summary="图谱统计（节点数 / 关系数 / 实体类型数）",
    description="对应 PPT 知识图谱页面顶部的三个统计卡片：节点数量、关系数量、实体类型。",
)
async def graph_stats(graph: GraphService = Depends(get_graph)) -> GraphStatsResponse:
    return await graph.get_stats()


# =============================================================================
#  GET /graph/entity-types —— 实体类型图例
# =============================================================================
@router.get(
    "/entity-types",
    response_model=List[EntityTypeMeta],
    summary="实体类型与配色图例",
    description="返回 8 类实体（疾病/症状/科室/药物/治疗方法/检查项目/易感人群/文献来源）及其配色，供前端图例渲染。",
)
async def entity_types(graph: GraphService = Depends(get_graph)) -> List[EntityTypeMeta]:
    return graph.get_entity_types()


# =============================================================================
#  GET /graph/search —— 实体检索（输入联想）
# =============================================================================
@router.get(
    "/search",
    response_model=EntitySearchResponse,
    summary="实体检索（图谱搜索框输入联想）",
)
async def search_entity(
    q: str = Query("", description="实体关键词，如 '血压'", max_length=80),
    limit: int = Query(20, ge=1, le=100),
    types: Optional[str] = Query(None, description="按类型过滤，多个用逗号分隔，如 'Disease,Symptom'"),
    graph: GraphService = Depends(get_graph),
) -> EntitySearchResponse:
    type_list = [t.strip() for t in types.split(",") if t.strip()] if types else None
    keyword, items = graph.search_entities(q, limit=limit, types=type_list)
    return EntitySearchResponse(keyword=keyword, items=items, total=len(items))


# =============================================================================
#  GET /graph/subgraph —— 子图查询（D3 渲染数据）
# =============================================================================
@router.get(
    "/subgraph",
    response_model=SubGraphResponse,
    summary="子图查询（知识图谱可视化数据源）",
    description=(
        "以指定实体为中心做 N 跳 BFS 展开，返回 D3.js 力导向图所需的 `nodes` 与 `links`。\n\n"
        "`depth=1` 展开直接邻居；`depth=2` 支持多跳探索。数据量由 `limit` 控制，"
        "响应中 `truncated` 标识是否被截断。"
    ),
)
async def subgraph(
    entity: str = Query(..., description="中心实体名，如 '高血压'", min_length=1, max_length=80),
    depth: int = Query(1, ge=1, le=3, description="展开跳数"),
    limit: int = Query(200, ge=10, le=800, description="最大节点数"),
    rels: Optional[str] = Query(None, description="关系类型过滤，逗号分隔，如 'HAS_SYMPTOM,TREATED_BY'"),
    types: Optional[str] = Query(None, description="实体类型过滤，逗号分隔"),
    graph: GraphService = Depends(get_graph),
) -> SubGraphResponse:
    rel_filter = [r.strip() for r in rels.split(",") if r.strip()] if rels else None
    type_filter = [t.strip() for t in types.split(",") if t.strip()] if types else None
    try:
        return await graph.get_subgraph(
            entity=entity, depth=depth, limit=limit,
            rel_filter=rel_filter, entity_types=type_filter,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("子图查询异常：%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"子图查询失败：{exc}") from exc


# =============================================================================
#  GET /graph/evidence —— 关系溯源证据
# =============================================================================
@router.get(
    "/evidence",
    response_model=Optional[EvidenceItem],
    summary="关系溯源证据（知识来源可追溯）",
    description="给定三元组 (head, relation, tail)，返回其来源文献（PubMed PMID / DOI / 指南名称）与置信度。",
)
async def graph_evidence(
    head: str = Query(..., description="头实体名", max_length=80),
    rel: Optional[str] = Query(None, description="关系英文类型，如 HAS_SYMPTOM"),
    tail: Optional[str] = Query(None, description="尾实体名"),
    graph: GraphService = Depends(get_graph),
) -> Optional[EvidenceItem]:
    ev = graph.get_evidence(head, rel, tail, "KG-1")
    if ev is None:
        raise HTTPException(status_code=404, detail=f"未找到「{head}」的匹配关系")
    return ev


# =============================================================================
#  GET /graph/triples —— 实体三元组列表
# =============================================================================
@router.get(
    "/triples",
    summary="实体三元组列表",
    description="返回指定实体的关联三元组（用于 RAG 上下文预览与图谱联动）。",
)
async def graph_triples(
    entity: str = Query(..., description="实体名", max_length=80),
    rels: Optional[str] = Query(None, description="关系类型过滤，逗号分隔"),
    limit: int = Query(30, ge=1, le=200),
    direction: str = Query("both", description="both | out | in"),
    graph: GraphService = Depends(get_graph),
) -> Dict[str, Any]:
    rel_filter = [r.strip() for r in rels.split(",") if r.strip()] if rels else None
    triples = graph.triples_of(entity, rels=rel_filter, limit=limit, direction=direction)
    return {"entity": entity, "total": len(triples), "triples": triples}


# =============================================================================
#  GET /graph/reasoning —— 图谱推理与补全
# =============================================================================
@router.get(
    "/reasoning",
    summary="图谱推理补全（Apache Jena 规则 + GAT/GCN 链接预测）",
    description=(
        "执行医疗规则推理（共病关联、并发症症状传递、传染性标记等）与图神经网络链接预测，"
        "挖掘图谱中隐藏的语义关联。返回新增的潜在三元组及其置信度与推理依据。"
    ),
)
async def graph_reasoning(
    mode: str = Query("hybrid", description="rule | gnn | hybrid"),
    limit: int = Query(50, ge=1, le=300),
    reasoner: GraphReasoner = Depends(get_reasoner),
) -> Dict[str, Any]:
    try:
        if mode == "rule":
            triples = reasoner.infer_by_rules(limit=limit)
            return {"mode": "rule", "total": len(triples), "backend": reasoner.info(),
                    "triples": triples}
        if mode == "gnn":
            triples = reasoner.complete_by_gnn(top_k=limit)
            return {"mode": "gnn", "total": len(triples), "backend": reasoner.info(),
                    "triples": triples}
        result = reasoner.hybrid(rule_limit=limit, gnn_limit=limit // 2)
        return {"mode": "hybrid", **result, "backend": reasoner.info()}
    except Exception as exc:  # noqa: BLE001
        logger.error("图谱推理异常：%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"图谱推理失败：{exc}") from exc


# =============================================================================
#  GET /graph/path —— 多跳路径解释（可解释推理）
# =============================================================================
@router.get(
    "/path",
    summary="多跳路径解释（可解释推理链路）",
    description="返回两个实体之间的多跳关联路径，用于向用户解释「结论是怎么推导出来的」。",
)
async def graph_path(
    start: str = Query(..., description="起点实体", max_length=80),
    end: str = Query(..., description="终点实体", max_length=80),
    max_hops: int = Query(3, ge=1, le=4),
    reasoner: GraphReasoner = Depends(get_reasoner),
) -> Dict[str, Any]:
    paths = reasoner.explain_path(start, end, max_hops=max_hops)
    return {"start": start, "end": end, "max_hops": max_hops,
            "total": len(paths), "paths": paths}


# =============================================================================
#  POST /graph/cypher —— 受控 Cypher 查询（只读白名单）
# =============================================================================
@router.post(
    "/cypher",
    response_model=CypherResponse,
    summary="受控 Cypher 查询（仅只读，安全白名单校验）",
    description=(
        "⚠️ 安全设计：仅允许 `MATCH / OPTIONAL MATCH / WITH / UNWIND / RETURN / CALL / SHOW` 开头的**只读**语句，"
        "禁止 CREATE / MERGE / DELETE / SET / REMOVE / DROP / LOAD CSV，且 CALL 仅允许 APOC 只读过程白名单。\n\n"
        "示例：`MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) RETURN d.name AS 疾病, collect(s.name)[0..5] AS 症状 LIMIT 10`"
    ),
)
async def run_cypher(
    payload: CypherRequest,
    graph: GraphService = Depends(get_graph),
) -> CypherResponse:
    try:
        cols, rows, elapsed = await graph.cypher_readonly(
            payload.cypher, payload.params, limit=payload.limit
        )
        return CypherResponse(columns=cols, rows=rows, elapsed_ms=elapsed)
    except CypherSecurityError as exc:
        raise HTTPException(status_code=403, detail=f"Cypher 安全校验未通过：{exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Cypher 执行异常：%s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Cypher 执行失败：{exc}") from exc


# =============================================================================
#  GET /graph/schema —— 图谱 Schema
# =============================================================================
@router.get(
    "/schema",
    summary="图谱 Schema（节点标签与关系类型）",
    description="返回知识图谱的节点标签、关系类型及其中文名，供前端展示与 Cypher 编写参考。",
)
async def graph_schema() -> Dict[str, Any]:
    from app.kg_layer.memory_store import REL_LABELS

    return {
        "nodes": [
            {"label": "Disease", "cn": "疾病",
             "props": ["name", "alias", "category1", "category2", "definition", "cause",
                       "diagnosis", "treatment", "prognosis", "population", "is_infectious"]},
            {"label": "Symptom", "cn": "症状", "props": ["name", "alias"]},
            {"label": "Drug", "cn": "药物", "props": ["name", "alias", "category"]},
            {"label": "Department", "cn": "科室", "props": ["name", "alias", "category"]},
            {"label": "Treatment", "cn": "治疗方法", "props": ["name", "alias", "category"]},
            {"label": "Check", "cn": "检查项目", "props": ["name", "alias"]},
            {"label": "Population", "cn": "易感人群", "props": ["name"]},
            {"label": "Source", "cn": "文献来源",
             "props": ["pmid", "doi", "title", "journal", "year", "url", "authority"]},
        ],
        "relations": [
            {"type": k, "cn": v} for k, v in REL_LABELS.items()
            if k not in ("PROVES", "IS_INFECTIOUS")
        ] + [
            {"type": "PROVES", "cn": "知识来源"},
            {"type": "IS_INFECTIOUS", "cn": "传染性"},
        ],
        "constraints": [
            "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT dept_name IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE",
        ],
        "indexes": [
            "CREATE INDEX disease_category IF NOT EXISTS FOR (d:Disease) ON (d.category1)",
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Disease|Symptom|Drug|Department) ON EACH [n.name, n.alias]",
        ],
    }
