# -*- coding: utf-8 -*-
"""
图谱业务服务（GraphService）
============================
**所有图谱相关 API 的唯一数据访问入口**，屏蔽底层差异：
  * Neo4j 可用   → 通过 Cypher 查询真实图数据库
  * Neo4j 不可用 → 自动路由到 `MemoryGraphStore`（离线演示模式）

对外提供：
  * 子图查询（知识图谱可视化）
  * 图谱统计（节点数 / 关系数 / 实体类型）
  * 实体检索（输入联想）
  * 疾病卡片检索 + 疾病详情（掌上医典）
  * 三元组溯源证据构建（可解释性核心）
  * 推荐追问生成
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.config import settings
from app.kg_layer.memory_store import MemoryGraphStore, REL_LABELS
from app.kg_layer.neo4j_client import Neo4jClient, get_neo4j_client
from app.schemas import (
    ENTITY_TYPE_META,
    DiseaseCard,
    DiseaseDetail,
    EntitySearchItem,
    EntityTypeMeta,
    EvidenceItem,
    GraphLink,
    GraphNode,
    GraphStatsResponse,
    RelatedDiseaseResponse,
    SourceItem,
    SubGraphResponse,
)
from app.utils.logger import get_logger
from app.utils.text import similarity, truncate

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class GraphService:
    """图谱业务服务"""

    def __init__(self, client: Optional[Neo4jClient] = None) -> None:
        self.client = client or get_neo4j_client()
        self.memory = MemoryGraphStore.instance()
        self._template_counter = 0

    # ==================================================================
    #  基础
    # ==================================================================
    @property
    def data_source(self) -> str:
        return "neo4j" if self.client.available else "memory"

    def _node_to_model(self, n: Dict[str, Any]) -> GraphNode:
        return GraphNode(
            id=str(n["id"]),
            name=n["name"],
            type=n["type"],
            label=n.get("label") or ENTITY_TYPE_META.get(n["type"], {}).get("label", n["type"]),
            color=n.get("color") or ENTITY_TYPE_META.get(n["type"], {}).get("color", "#409EFF"),
            degree=int(n.get("degree", 0)),
            category1=(n.get("properties") or {}).get("category1"),
            category2=(n.get("properties") or {}).get("category2"),
            properties=self._public_props(n.get("properties") or {}),
        )

    @staticmethod
    def _public_props(props: Dict[str, Any]) -> Dict[str, Any]:
        """过滤掉大字段，避免前端图数据膨胀"""
        out = {}
        for k, v in props.items():
            if k in ("definition", "cause", "diagnosis", "treatment", "prognosis"):
                out[k] = truncate(str(v), 120)
            elif k == "sources":
                out["source_count"] = len(v or [])
            elif isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, list) and len(v) <= 12:
                out[k] = v
        return out

    def _link_to_model(self, e: Dict[str, Any]) -> GraphLink:
        return GraphLink(
            source=str(e["source"]),
            target=str(e["target"]),
            rel=e["rel"],
            label=e.get("label") or REL_LABELS.get(e["rel"], e["rel"]),
            properties=e.get("properties") or {},
        )

    # ==================================================================
    #  1) 子图查询（知识图谱可视化模块）
    # ==================================================================
    async def get_subgraph(
        self,
        entity: str,
        depth: int = 1,
        limit: int = 200,
        rel_filter: Optional[Iterable[str]] = None,
        entity_types: Optional[Iterable[str]] = None,
    ) -> SubGraphResponse:
        """
        Neo4j 路径：可变长模式匹配 `MATCH p=(n)-[*1..depth]-(m)`
        内存路径：BFS（与 Cypher 语义一致）
        """
        depth = max(1, min(int(depth), 3))
        limit = max(10, min(int(limit), 800))

        if self.client.available:
            try:
                raw = await self._subgraph_cypher(entity, depth, limit)
                if raw and raw.get("nodes"):
                    return self._build_subgraph_response(raw, entity, depth, limit, entity_types)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Neo4j 子图查询失败，回落内存实现：%s", exc)

        raw = self.memory.subgraph(entity, depth=depth, limit=limit, rel_filter=rel_filter)
        return self._build_subgraph_response(raw, entity, depth, limit, entity_types)

    async def _subgraph_cypher(self, entity: str, depth: int, limit: int) -> Dict[str, Any]:
        """Neo4j 子图查询（多跳变长模式）"""
        cypher = f"""
        MATCH (center)
        WHERE (center.name = $entity OR $entity IN coalesce(center.alias, []))
          AND NOT center:Source
        WITH center LIMIT 1
        MATCH path = (center)-[*1..{depth}]-(neighbor)
        WHERE NOT neighbor:Source
        WITH center, collect(DISTINCT neighbor)[0..{limit}] AS neighbors
        WITH center, neighbors, neighbors + [center] AS all_nodes
        UNWIND all_nodes AS a
        OPTIONAL MATCH (a)-[r]-(b)
        WHERE b IN all_nodes
        RETURN
          collect(DISTINCT {{
            id: elementId(a), name: a.name, type: labels(a)[0],
            props: properties(a)
          }}) AS nodes,
          collect(DISTINCT {{
            source: elementId(startNode(r)), target: elementId(endNode(r)),
            rel: type(r), props: properties(r)
          }}) AS links
        """
        _, rows = await self.client.query(cypher, {"entity": entity}, readonly=True)
        if not rows:
            return {}
        row = rows[0]
        nodes = []
        for n in row.get("nodes") or []:
            props = dict(n.get("props") or {})
            ntype = n.get("type") or "Other"
            meta = ENTITY_TYPE_META.get(ntype, {})
            nodes.append({
                "id": n.get("id"), "name": n.get("name"), "type": ntype,
                "label": meta.get("label", ntype), "color": meta.get("color", "#909399"),
                "degree": 0, "properties": props,
            })
        links = []
        for l in row.get("links") or []:
            if not l.get("source") or not l.get("target"):
                continue
            links.append({
                "source": l["source"], "target": l["target"], "rel": l.get("rel", "RELATED_TO"),
                "label": REL_LABELS.get(l.get("rel", ""), l.get("rel", "相关")),
                "properties": l.get("props") or {},
            })
        # 计算 degree
        deg: Dict[str, int] = {}
        for l in links:
            deg[l["source"]] = deg.get(l["source"], 0) + 1
            deg[l["target"]] = deg.get(l["target"], 0) + 1
        for n in nodes:
            n["degree"] = deg.get(n["id"], 0)
        return {"nodes": nodes, "links": links,
                "node_count": len(nodes), "link_count": len(links)}

    def _build_subgraph_response(
        self,
        raw: Dict[str, Any],
        entity: str,
        depth: int,
        limit: int,
        entity_types: Optional[Iterable[str]] = None,
    ) -> SubGraphResponse:
        nodes_raw = raw.get("nodes") or []
        links_raw = raw.get("links") or []

        type_set = set(entity_types) if entity_types else None
        nodes = [self._node_to_model(n) for n in nodes_raw
                 if not type_set or n.get("type") in type_set]
        keep = {n.id for n in nodes}
        links = [self._link_to_model(e) for e in links_raw
                 if str(e["source"]) in keep and str(e["target"]) in keep]

        return SubGraphResponse(
            nodes=nodes,
            links=links,
            node_count=len(nodes),
            link_count=len(links),
            type_count=len({n.type for n in nodes}),
            center=raw.get("center") or entity,
            depth=depth,
            truncated=bool(raw.get("truncated")) or len(nodes_raw) >= limit or len(nodes) >= limit,
            message=raw.get("message", ""),
        )

    # ==================================================================
    #  2) 统计
    # ==================================================================
    async def get_stats(self) -> GraphStatsResponse:
        if self.client.available:
            try:
                stats = await self._stats_cypher()
                if stats and stats.get("total_nodes"):
                    return GraphStatsResponse(**stats)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Neo4j 统计失败，回落内存实现：%s", exc)

        s = self.memory.stats()
        return GraphStatsResponse(
            total_nodes=s["total_nodes"],
            total_links=s["total_links"],
            entity_type_count=s["entity_type_count"],
            entity_types=[EntityTypeMeta(**t) for t in s["entity_types"]],
            disease_count=s["disease_count"],
            symptom_count=s["symptom_count"],
            drug_count=s["drug_count"],
            department_count=s["department_count"],
            treatment_count=s["treatment_count"],
            source_count=s["source_count"],
            data_source="memory",
        )

    async def _stats_cypher(self) -> Dict[str, Any]:
        _, rows = await self.client.query(
            """
            MATCH (n) WHERE NOT n:Source
            WITH labels(n)[0] AS t, count(n) AS c
            RETURN collect({type: t, count: c}) AS types,
                   sum(c) AS total
            """,
            readonly=True,
        )
        if not rows:
            return {}
        row = rows[0]
        types = []
        by_type: Dict[str, int] = {}
        for t in row.get("types") or []:
            tname = t.get("type") or "Other"
            cnt = int(t.get("count") or 0)
            by_type[tname] = cnt
            meta = ENTITY_TYPE_META.get(tname, {})
            types.append({
                "type": tname, "label": meta.get("label", tname),
                "color": meta.get("color", "#909399"), "count": cnt,
            })
        types.sort(key=lambda x: -x["count"])

        _, rel_rows = await self.client.query(
            "MATCH ()-[r]->() WHERE NOT startNode(r):Source AND NOT endNode(r):Source RETURN count(r) AS c",
            readonly=True,
        )
        total_links = int(rel_rows[0]["c"]) if rel_rows else 0
        return {
            "total_nodes": int(row.get("total") or 0),
            "total_links": total_links,
            "entity_type_count": len(types),
            "entity_types": types,
            "disease_count": by_type.get("Disease", 0),
            "symptom_count": by_type.get("Symptom", 0),
            "drug_count": by_type.get("Drug", 0),
            "department_count": by_type.get("Department", 0),
            "treatment_count": by_type.get("Treatment", 0),
            "source_count": by_type.get("Source", 0),
            "data_source": "neo4j",
        }

    def get_entity_types(self) -> List[EntityTypeMeta]:
        """实体类型图例（含各类计数）"""
        counts: Dict[str, int] = {}
        for n in self.memory.nodes.values():
            counts[n["type"]] = counts.get(n["type"], 0) + 1
        order = list(ENTITY_TYPE_META.keys())
        out: List[EntityTypeMeta] = []
        for t in order:
            meta = ENTITY_TYPE_META[t]
            out.append(EntityTypeMeta(type=t, label=meta["label"], color=meta["color"],
                                      count=counts.get(t, 0)))
        return out

    # ==================================================================
    #  3) 实体检索（输入联想）
    # ==================================================================
    def search_entities(self, keyword: str, limit: int = 20,
                        types: Optional[Iterable[str]] = None) -> Tuple[str, List[EntitySearchItem]]:
        hits = self.memory.search(keyword, limit=limit, types=types)
        items = [
            EntitySearchItem(
                id=h["id"], name=h["name"], type=h["type"],
                label=h.get("label", ""), color=h.get("color", "#409EFF"),
                score=h.get("score", 1.0),
                category1=(h.get("properties") or {}).get("category1"),
            )
            for h in hits
        ]
        return keyword, items

    # ==================================================================
    #  4) 疾病检索与详情（掌上医典）
    # ==================================================================
    def search_diseases(self, keyword: str, page: int = 1, page_size: int = 6,
                        category1: Optional[str] = None) -> Tuple[int, List[DiseaseCard]]:
        """疾病检索：支持疾病名 / 别名 / 症状 / 科室 / 分类 多维命中，按相关度排序"""
        kw = (keyword or "").strip()
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 60))

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for did, d in self.memory.diseases.items():
            if category1 and d.get("category1") != category1:
                continue
            score = 0.0
            if not kw:
                score = 0.5
            else:
                low = kw.lower()
                name = d.get("name", "")
                aliases = [str(a) for a in (d.get("alias") or [])]
                # 1) 名称精确 / 包含
                if name == kw:
                    score = 1.0
                elif low in name.lower():
                    score = 0.93
                elif any(low in a.lower() for a in aliases):
                    score = 0.88
                # 2) 症状命中
                elif any(low in s.lower() for s in (d.get("symptoms") or [])):
                    score = 0.72
                # 3) 分类 / 科室命中
                elif low in str(d.get("category1", "")).lower() or low in str(d.get("category2", "")).lower():
                    score = 0.66
                elif low in str(d.get("department", "")).lower():
                    score = 0.64
                # 4) 定义 / 病因 命中
                elif low in str(d.get("definition", "")).lower() or low in str(d.get("cause", "")).lower():
                    score = 0.58
                # 5) 模糊
                else:
                    s1 = similarity(kw, name)
                    s2 = max((similarity(kw, a) for a in aliases), default=0.0)
                    score = max(s1, s2) * 0.85
            if score >= 0.5:
                scored.append((score, d))

        scored.sort(key=lambda kv: (-kv[0], kv[1].get("name", "")))
        total = len(scored)
        start = (page - 1) * page_size
        page_items = scored[start : start + page_size]
        return total, [self._to_card(d) for _, d in page_items]

    def _to_card(self, d: Dict[str, Any]) -> DiseaseCard:
        return DiseaseCard(
            disease_id=d.get("disease_id", ""),
            name=d.get("name", ""),
            category1=d.get("category1"),
            category2=d.get("category2"),
            symptoms=(d.get("symptoms") or [])[:12],
            treatments=(d.get("treatments") or [])[:8],
            population=d.get("population"),
            is_infectious=bool(d.get("is_infectious")),
            has_detail=True,
        )

    def get_disease_detail(self, disease_id_or_name: str) -> Optional[DiseaseDetail]:
        """疾病详情：支持 disease_id 或疾病名"""
        d = self._find_disease(disease_id_or_name)
        if not d:
            return None
        dept = d.get("department") or d.get("category2") or d.get("category1")
        return DiseaseDetail(
            disease_id=d.get("disease_id", ""),
            name=d.get("name", ""),
            alias=[str(a) for a in (d.get("alias") or [])],
            category1=d.get("category1"),
            category2=d.get("category2"),
            definition=d.get("definition", ""),
            cause=d.get("cause", ""),
            symptoms=d.get("symptoms") or [],
            diagnosis=d.get("diagnosis", ""),
            checks=d.get("checks") or [],
            treatment=d.get("treatment", ""),
            treatments=d.get("treatments") or [],
            drugs=d.get("drugs") or [],
            department=dept,
            prognosis=d.get("prognosis", ""),
            population=d.get("population"),
            complications=d.get("complications") or [],
            differential=d.get("differential") or [],
            is_infectious=bool(d.get("is_infectious")),
            sources=[SourceItem(**{k: v for k, v in s.items() if k in SourceItem.model_fields})
                     for s in (d.get("sources") or [])],
            updated_at=d.get("updated_at"),
        )

    def _find_disease(self, key: str) -> Optional[Dict[str, Any]]:
        key = (key or "").strip()
        if not key:
            return None
        if key in self.memory.diseases:
            return self.memory.diseases[key]
        did = self.memory.disease_by_name.get(key)
        if did:
            return self.memory.diseases[did]
        # 别名 / 模糊
        nid = self.memory.resolve(key, "Disease") or self.memory.resolve(key)
        if nid:
            node = self.memory.nodes.get(nid)
            if node:
                did = self.memory.disease_by_name.get(node["name"])
                if did:
                    return self.memory.diseases[did]
        best, best_score = None, 0.0
        for _, d in self.memory.diseases.items():
            s = similarity(key, d.get("name", ""))
            if s > best_score:
                best, best_score = d, s
        return best if best_score >= 0.6 else None

    def get_related_diseases(self, disease_id_or_name: str) -> RelatedDiseaseResponse:
        d = self._find_disease(disease_id_or_name)
        if not d:
            return RelatedDiseaseResponse(disease_id="", complications=[], differential=[])

        def cards(names: List[str]) -> List[DiseaseCard]:
            out = []
            for n in names:
                f = self._find_disease(n)
                if f:
                    out.append(self._to_card(f))
            return out

        return RelatedDiseaseResponse(
            disease_id=d.get("disease_id", ""),
            complications=cards(d.get("complications") or []),
            differential=cards(d.get("differential") or []),
        )

    def hot_keywords(self, topn: int = 6) -> List[str]:
        """热门搜索词：按疾病重要性（关系度数）排序"""
        defaults = ["感冒", "高血压", "糖尿病", "冠心病", "肺炎", "胃炎"]
        scored: List[Tuple[int, str]] = []
        for did, d in self.memory.diseases.items():
            nid = self.memory.index.get(("Disease", d["name"]))
            deg = self.memory.nodes[nid]["degree"] if nid else 0
            scored.append((deg, d["name"]))
        scored.sort(reverse=True)
        picked = [n for _, n in scored[:topn]]
        # 保证默认热词在前（与 PPT 截图一致）
        merged: List[str] = []
        for k in defaults:
            if k in self.memory.disease_by_name and k not in merged:
                merged.append(k)
        for n in picked:
            if n not in merged:
                merged.append(n)
        return merged[:topn]

    # ==================================================================
    #  5) 三元组 / 溯源证据（可解释性核心）
    # ==================================================================
    def get_evidence(
        self, head: str, rel: Optional[str] = None, tail: Optional[str] = None,
        triple_id: str = "KG-1",
    ) -> Optional[EvidenceItem]:
        """构造单条三元组的溯源证据（含来源文献）"""
        hid = self.memory.resolve(head)
        if not hid:
            return None
        for i in self.memory.adjacency.get(hid, []):
            e = self.memory.edges[i]
            other_id = e["target"] if e["source"] == hid else e["source"]
            other = self.memory.nodes.get(other_id)
            if not other:
                continue
            if rel and e["rel"] != rel:
                continue
            if tail and other["name"] != tail:
                continue
            return self._edge_to_evidence(e, self.memory.nodes[hid], other, triple_id)
        return None

    def _edge_to_evidence(self, e: Dict[str, Any], head_node: Dict[str, Any],
                          tail_node: Dict[str, Any], triple_id: str) -> EvidenceItem:
        sources: List[SourceItem] = []
        for node in (head_node, tail_node):
            for s in (node.get("properties") or {}).get("sources") or []:
                try:
                    sources.append(SourceItem(**{k: v for k, v in s.items()
                                                 if k in SourceItem.model_fields}))
                except Exception:  # noqa: BLE001
                    continue
        # 去重
        uniq, seen = [], set()
        for s in sources:
            key = (s.pmid, s.doi, s.title)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
        return EvidenceItem(
            triple_id=triple_id,
            head=head_node["name"], head_type=head_node["type"],
            relation=e["rel"], relation_label=e.get("label", REL_LABELS.get(e["rel"], e["rel"])),
            tail=tail_node["name"], tail_type=tail_node["type"],
            confidence=float((e.get("properties") or {}).get("confidence", 0.95)),
            sources=uniq[:3],
        )

    def triples_of(self, entity: str, rels: Optional[Iterable[str]] = None,
                   limit: int = 20, direction: str = "both") -> List[Dict[str, Any]]:
        """获取某实体的三元组（用于 RAG 上下文构建）"""
        nid = self.memory.resolve(entity)
        if not nid:
            return []
        rel_set = set(rels) if rels else None
        out: List[Dict[str, Any]] = []
        for i in self.memory.adjacency.get(nid, []):
            e = self.memory.edges[i]
            if rel_set and e["rel"] not in rel_set:
                continue
            if direction == "out" and e["source"] != nid:
                continue
            if direction == "in" and e["target"] != nid:
                continue
            other_id = e["target"] if e["source"] == nid else e["source"]
            other = self.memory.nodes.get(other_id)
            if not other or other["type"] == "Source":
                continue
            out.append({
                "head": self.memory.nodes[nid]["name"], "head_type": self.memory.nodes[nid]["type"],
                "relation": e["rel"], "relation_label": e.get("label", e["rel"]),
                "tail": other["name"], "tail_type": other["type"],
                "confidence": float((e.get("properties") or {}).get("confidence", 0.95)),
            })
        out.sort(key=lambda x: -x["confidence"])
        return out[:limit]

    def facts_for_disease(self, name: str) -> Dict[str, List[str]]:
        """按关系类型聚合疾病事实（RAG 上下文的分槽位组织）"""
        triples = self.triples_of(name, rels=[
            "HAS_SYMPTOM", "TREATED_BY", "USES_DRUG", "BELONGS_TO",
            "NEEDS_CHECK", "AFFECTS", "HAS_COMPLICATION", "DIFFERENTIAL_WITH",
        ], limit=200, direction="out")
        facts: Dict[str, List[str]] = {r: [] for r in REL_LABELS}
        for t in triples:
            facts.setdefault(t["relation"], []).append(t["tail"])
        return {k: v for k, v in facts.items() if v}

    def sources_of(self, entity: str) -> List[SourceItem]:
        nid = self.memory.resolve(entity)
        if not nid:
            return []
        node = self.memory.nodes[nid]
        out = []
        for s in (node.get("properties") or {}).get("sources") or []:
            try:
                out.append(SourceItem(**{k: v for k, v in s.items() if k in SourceItem.model_fields}))
            except Exception:  # noqa: BLE001
                continue
        return out

    # ==================================================================
    #  6) 推荐追问
    # ==================================================================
    _QUESTION_TEMPLATES: Dict[str, List[str]] = {
        "HAS_SYMPTOM": ["{e}的症状有哪些？", "{e}有哪些典型表现？"],
        "TREATED_BY": ["{e}的治疗方法是什么？", "{e}怎么治疗？"],
        "USES_DRUG": ["{e}常用什么药物？", "治疗{e}的一线用药有哪些？"],
        "BELONGS_TO": ["{e}应该挂什么科？", "{e}看哪个科室？"],
        "NEEDS_CHECK": ["{e}需要做哪些检查？", "确诊{e}要做哪些检查？"],
        "HAS_COMPLICATION": ["{e}会引起哪些并发症？", "{e}的并发症有哪些？"],
        "DIFFERENTIAL_WITH": ["{e}需要和哪些疾病鉴别？"],
        "AFFECTS": ["{e}多见于哪些人群？"],
    }

    def related_questions(self, entity: Optional[str], n: int = 5) -> List[str]:
        """基于实体在图谱中的关系类型生成推荐追问"""
        out: List[str] = []
        if entity:
            nid = self.memory.resolve(entity)
            if nid:
                rels: List[str] = []
                for i in self.memory.adjacency.get(nid, []):
                    rels.append(self.memory.edges[i]["rel"])
                for rel in dict.fromkeys(rels):
                    for tpl in self._QUESTION_TEMPLATES.get(rel, []):
                        q = tpl.format(e=entity)
                        if q not in out:
                            out.append(q)
                    if len(out) >= n:
                        return out[:n]
        # 兜底：全局高频问题
        defaults = [
            "成人呼吸窘迫综合征的症状有哪些？",
            "肺栓塞应该挂什么科？",
            "肺心病的治疗方法是什么？",
            "继发性肺动脉高压是否传染？",
            "睡眠呼吸暂停综合征的治愈率如何？",
            "2型糖尿病的常用药物有哪些？",
        ]
        for q in defaults:
            if q not in out:
                out.append(q)
            if len(out) >= n:
                break
        return out[:n]

    # ==================================================================
    #  7) 图谱推理与补全（Apache Jena 规则 + GAT/GCN）
    # ==================================================================
    async def cypher_readonly(self, cypher: str, params: Dict[str, Any],
                              limit: int = 200) -> Tuple[List[str], List[Dict[str, Any]], float]:
        """受控 Cypher 查询（Neo4j 不可用时路由到内存子集解释器）"""
        import time

        t0 = time.perf_counter()
        cols, rows = await self.client.query(cypher, params, readonly=True)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return cols, rows[:limit], elapsed


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_service: Optional[GraphService] = None


def get_graph_service() -> GraphService:
    global _service
    if _service is None:
        _service = GraphService()
    return _service
