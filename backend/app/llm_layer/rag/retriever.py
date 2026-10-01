# -*- coding: utf-8 -*-
"""
混合检索器（HybridRetriever）—— 以「知识图谱三元组」为检索单元
================================================================

★ 核心设计决策：检索单位是 **KG 三元组**，而不是通用 RAG 的 **文本 chunk** ★

理由（也是本项目相对通用 RAG 的创新点）：
  1. **可溯源**：医疗问答必须能回答"这条结论从哪来"。三元组自带
     「头实体 —关系→ 尾实体」+ 文献来源，可以直接生成 `[KG-n]` 角标；
     文本 chunk 只能给出一段话，无法定位到原子事实。
  2. **可校验**：幻觉守卫需要逐条比对"答案里出现的事实是否真的在上下文中"。
     三元组是原子事实（h, r, t），可做精确比对；chunk 的语义边界模糊，无法做事实级校验。
  3. **省 token**：一条三元组平均 20 字，12 条约 240 字；等效信息量的原文 chunk 需 2000+ 字，
     在 500ms 语义解析与 llama3:8b 本地推理的约束下，三元组上下文显著更快更稳。
  4. **结构化对齐**：三元组的 relation 类型天然对应「症状 / 治疗 / 用药 / 科室 / 检查 / 并发症 /
     鉴别诊断」槽位，与医疗 prompt 的「分点作答」要求同构，模型几乎不会跑偏。

检索策略（五路召回 + 统一排序）
-------------------------------
  A. 槽位驱动直接事实         `facts_for_disease` + `triples_of(..., direction="out")`，按槽位配额截断
  B. 症状 → 疾病反向检索      `memory.neighbors(symptom, ["HAS_SYMPTOM"])`，只保留"疾病 → 症状"方向
  C. 多跳路径事实             `memory.multi_hop_paths` / `reasoner.explain_path`，产出可读推理链
  D. 规则推理补全事实         `reasoner.infer_by_rules`（只算一次并缓存，代价高），仅保留命中查询实体者
  E. 相似疾病扩展             `memory.search(name, types=["Disease"])` 的邻居直接事实（事实不足 top_k 时）

排序公式（已在 `_rank` 中实现，可解释、可调参）
------------------------------------------------------------------
    score = source_weight × (0.45 × confidence
                           + 0.30 × intent_slot_prior
                           + 0.15 × entity_proximity
                           + 0.10 × degree_normalized)

  * `confidence`        三元组自身置信度（0~1）
  * `intent_slot_prior` 意图槽位先验：科室导诊提升 BELONGS_TO/AFFECTS，治疗咨询提升
                        TREATED_BY/USES_DRUG，症状咨询提升 HAS_SYMPTOM/HAS_COMPLICATION，
                        疾病查询对所有关系等权（1.0）
  * `entity_proximity`  事实与问句实体的贴近程度：主实体 1.0 / 问句实体 0.8 / 一跳邻居 0.55 / 其他 0.3
  * `degree_normalized` 头尾节点度数除以全图最大度数（降低孤立实体的噪声权重）
  * `source_weight`     召回路径权重：直接事实 1.00、多跳 0.92、相似扩展 0.85、推理补全 0.80
                        （越"远"的推断越要排在直接事实之后）

排序之后还有一步 **多样性名额分配**（`_select_diverse`）：主实体的直接事实往往足够填满 top_k，
若不保留名额，"相关疾病 / 多跳路径 / 推理补全"会被全部挤出上下文，RAG 就退化成单疾病罗列。
因此按 `DIVERSITY_RESERVE` 先给每类召回路径留位置，再按分数回填剩余名额。

降级保障：整条链路只依赖 `MemoryGraphStore` + `GraphService`（可选的可视化/推理模块失败会静默降级），
没有 Neo4j、没有 torch、没有网络也能产出完整的三元组上下文。
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.config import settings
from app.kg_layer.graph_service import get_graph_service
from app.kg_layer.memory_store import MemoryGraphStore, REL_LABELS
from app.schemas import SourceItem
from app.utils.logger import get_logger
from app.utils.text import truncate

logger = get_logger(__name__)

#: 槽位配额：主实体每个关系槽位最多贡献多少条三元组
SLOT_QUOTAS: Dict[str, int] = {
    "HAS_SYMPTOM": 4,        # 症状
    "TREATED_BY": 3,         # 治疗
    "USES_DRUG": 3,          # 用药
    "BELONGS_TO": 2,         # 科室
    "NEEDS_CHECK": 3,        # 检查
    "HAS_COMPLICATION": 2,   # 并发症
    "DIFFERENTIAL_WITH": 2,  # 鉴别诊断
    "AFFECTS": 2,            # 易感人群
}

#: 不上台面的关系（来源标记、布尔标记），不进 RAG 上下文
SKIP_RELATIONS: Set[str] = {"PROVES", "IS_INFECTIOUS"}

#: 非配额槽位的兜底配额
DEFAULT_QUOTA: int = 1

#: 每类意图关心的关系槽位（intent_slot_prior）
INTENT_SLOT_PRIOR: Dict[str, Dict[str, float]] = {
    "department_query": {
        "BELONGS_TO": 1.00, "AFFECTS": 0.90, "DIFFERENTIAL_WITH": 0.55,
        "HAS_SYMPTOM": 0.50, "NEEDS_CHECK": 0.50, "HAS_COMPLICATION": 0.45,
        "RELATED_TO": 0.45, "TREATED_BY": 0.40, "USES_DRUG": 0.35,
    },
    "treatment_query": {
        "TREATED_BY": 1.00, "USES_DRUG": 1.00, "NEEDS_CHECK": 0.60,
        "BELONGS_TO": 0.50, "HAS_COMPLICATION": 0.50, "DIFFERENTIAL_WITH": 0.45,
        "RELATED_TO": 0.45, "HAS_SYMPTOM": 0.40, "AFFECTS": 0.40,
    },
    "symptom_consult": {
        "HAS_SYMPTOM": 1.00, "HAS_COMPLICATION": 0.95, "DIFFERENTIAL_WITH": 0.80,
        "BELONGS_TO": 0.70, "RELATED_TO": 0.60, "NEEDS_CHECK": 0.60,
        "AFFECTS": 0.50, "TREATED_BY": 0.40, "USES_DRUG": 0.35,
    },
    "_default": {
        "HAS_SYMPTOM": 0.70, "TREATED_BY": 0.70, "USES_DRUG": 0.70,
        "BELONGS_TO": 0.80, "NEEDS_CHECK": 0.70, "HAS_COMPLICATION": 0.70,
        "DIFFERENTIAL_WITH": 0.65, "AFFECTS": 0.65, "RELATED_TO": 0.70,
    },
}
#: 疾病查询：所有关系等权
INTENT_SLOT_PRIOR["disease_query"] = {k: 1.0 for k in REL_LABELS}
INTENT_SLOT_PRIOR["disease_query"]["_default"] = 1.0

#: 召回路径权重
SOURCE_WEIGHT: Dict[str, float] = {
    "kg_1hop": 1.00,      # 直接事实 / 反向检索
    "kg_2hop": 0.92,      # 多跳路径
    "kg_similar": 0.85,   # 相似疾病扩展
    "kg_reasoned": 0.80,  # 规则推理补全
}

#: 相似疾病扩展时只保留这几类"高价值槽位"
SIMILAR_RELS: Tuple[str, ...] = (
    "HAS_SYMPTOM", "TREATED_BY", "USES_DRUG", "BELONGS_TO", "NEEDS_CHECK",
)

#: 多样性保留名额：主实体直接事实往往足够多，若不保留名额，
#: "相关疾病 / 多跳路径 / 推理补全" 三类高价值事实会被全部挤出 top_k，
#: 上下文就退化成单疾病罗列，丢失跨疾病鉴别与推理能力。
DIVERSITY_RESERVE: Dict[str, int] = {
    "related": 2,       # 反向检索到的相关疾病 + 相似疾病扩展
    "kg_2hop": 2,       # 多跳推理路径
    "kg_reasoned": 1,   # 规则/GNN 推理补全
}

#: 实体字典中的键（顺序即展示顺序）
ENTITY_KEYS: Tuple[str, ...] = (
    "diseases", "symptoms", "departments", "drugs", "checks", "treatments", "populations",
)

#: 主排序公式的权重（改这里即可调参）
W_CONFIDENCE: float = 0.45
W_SLOT_PRIOR: float = 0.30
W_PROXIMITY: float = 0.15
W_DEGREE: float = 0.10


def _names(entities: Optional[Dict[str, Any]], *keys: str) -> List[str]:
    """从实体链接结果里安全地取出实体名列表（容忍 None / 字符串 / 对象值）"""
    out: List[str] = []
    seen: Set[str] = set()
    if not isinstance(entities, dict):
        return out
    for key in keys:
        vals = entities.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, (list, tuple, set)):
            continue
        for v in vals:
            if v is None:
                continue
            if not isinstance(v, str):
                v = getattr(v, "kg_name", None) or getattr(v, "text", None) or str(v)
            v = (v or "").strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return out


@dataclass
class RetrievedFact:
    """一条被召回的三元组事实（RAG 上下文的最小单元）"""

    triple_id: str = ""
    head: str = ""
    head_type: str = ""
    relation: str = ""
    relation_label: str = ""
    tail: str = ""
    tail_type: str = ""
    confidence: float = 0.9
    score: float = 0.0
    source: str = "kg_1hop"
    sources: List[SourceItem] = field(default_factory=list)
    explain: str = ""

    def key(self) -> Tuple[str, str, str]:
        return (self.head, self.relation, self.tail)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triple_id": self.triple_id, "head": self.head, "head_type": self.head_type,
            "relation": self.relation, "relation_label": self.relation_label,
            "tail": self.tail, "tail_type": self.tail_type,
            "confidence": round(float(self.confidence), 4),
            "score": round(float(self.score), 4), "source": self.source,
            "explain": self.explain,
        }


class HybridRetriever:
    """五路召回 + 统一排序的混合检索器"""

    def __init__(self, graph_service: Any = None) -> None:
        self.graph_service = graph_service or get_graph_service()
        self.memory: MemoryGraphStore = getattr(self.graph_service, "memory", None) \
            or MemoryGraphStore.instance()
        self._reasoner: Any = None
        self._reasoner_tried: bool = False
        self._reasoned_cache: Optional[List[Dict[str, Any]]] = None
        self._degree_cache: Optional[int] = None
        self.last_stats: Dict[str, int] = {}

    # ==================================================================
    #  0) 工具
    # ==================================================================
    def _mk(self, **kwargs: Any) -> RetrievedFact:
        """构造事实并补齐中文关系标签"""
        rel = kwargs.get("relation", "") or ""
        kwargs.setdefault("relation_label", REL_LABELS.get(rel, rel))
        kwargs.setdefault("head_type", "")
        kwargs.setdefault("tail_type", "")
        kwargs.setdefault("score", 0.0)
        kwargs.setdefault("triple_id", "")
        return RetrievedFact(**kwargs)

    def _get_reasoner(self) -> Any:
        """惰性获取图谱推理器（torch 缺失时返回 None，不抛异常）"""
        if self._reasoner_tried:
            return self._reasoner
        self._reasoner_tried = True
        try:
            from app.kg_layer.graph_reasoner import get_graph_reasoner

            self._reasoner = get_graph_reasoner()
        except Exception as exc:  # noqa: BLE001
            logger.warning("图谱推理器不可用，跳过规则推理补全：%s", exc)
            self._reasoner = None
        return self._reasoner

    def _max_degree(self) -> int:
        if self._degree_cache is None:
            degs = [int(n.get("degree", 0) or 0) for n in self.memory.nodes.values()]
            self._degree_cache = max(degs) if degs else 1
        return max(1, self._degree_cache)

    def _node_name_type(self, nid: str) -> Tuple[str, str]:
        node = self.memory.nodes.get(nid) or {}
        return str(node.get("name", "")), str(node.get("type", ""))

    def _neighbour_names(self, names: Sequence[str]) -> Set[str]:
        """查询实体的一跳邻居名集合（用于 entity_proximity 打分的中间档）"""
        out: Set[str] = set()
        visited = 0
        for name in names:
            nid = self.memory.resolve(name)
            if not nid:
                continue
            for nb in self.memory.neighbors(nid):
                node = nb.get("node") or {}
                if node.get("name"):
                    out.add(str(node["name"]))
                visited += 1
                if visited > 400:  # 防止超大图上的组合爆炸
                    return out
        return out

    def _evidence_sources(self, fact: RetrievedFact, triple_id: str) -> List[SourceItem]:
        """按 三元组证据 → 头实体来源 的顺序补齐文献溯源"""
        try:
            ev = self.graph_service.get_evidence(
                fact.head, fact.relation or None, fact.tail or None, triple_id or "KG-1",
            )
            if ev is not None and getattr(ev, "sources", None):
                return list(ev.sources)
        except Exception as exc:  # noqa: BLE001
            logger.debug("三元组证据构造失败（%s），回落头实体来源", exc)
        try:
            return list(self.graph_service.sources_of(fact.head) or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("头实体来源构造失败：%s", exc)
            return []

    # ==================================================================
    #  1) 槽位驱动直接事实
    # ==================================================================
    def _slot_direct_facts(self, diseases: Sequence[str]) -> List[RetrievedFact]:
        out: List[RetrievedFact] = []
        for disease in diseases:
            facts_map: Dict[str, List[str]] = {}
            try:
                facts_map = self.graph_service.facts_for_disease(disease) or {}
            except Exception as exc:  # noqa: BLE001
                logger.debug("facts_for_disease(%s) 失败：%s", disease, exc)
            try:
                triples = self.graph_service.triples_of(disease, direction="out", limit=120) or []
            except Exception as exc:  # noqa: BLE001
                logger.debug("triples_of(%s) 失败：%s", disease, exc)
                triples = []

            by_rel: Dict[str, List[Dict[str, Any]]] = {}
            for t in triples:
                by_rel.setdefault(str(t.get("relation", "")), []).append(t)

            # 槽位顺序：配额槽位优先，其次图谱里实际存在的关系
            order: List[str] = [r for r in SLOT_QUOTAS if r in by_rel or r in facts_map]
            for rel in list(by_rel.keys()) + list(facts_map.keys()):
                if rel not in order and rel not in SKIP_RELATIONS:
                    order.append(rel)

            for rel in order:
                if rel in SKIP_RELATIONS:
                    continue
                quota = SLOT_QUOTAS.get(rel, DEFAULT_QUOTA)
                picked = 0
                for t in sorted(by_rel.get(rel, []), key=lambda x: -float(x.get("confidence", 0.9))):
                    if picked >= quota:
                        break
                    out.append(self._mk(
                        head=str(t.get("head", disease)),
                        head_type=str(t.get("head_type", "Disease")),
                        relation=rel,
                        relation_label=str(t.get("relation_label") or REL_LABELS.get(rel, rel)),
                        tail=str(t.get("tail", "")),
                        tail_type=str(t.get("tail_type", "")),
                        confidence=float(t.get("confidence", 0.9) or 0.9),
                        source="kg_1hop",
                        explain=f"「{disease}」的直接事实（槽位：{REL_LABELS.get(rel, rel)}）",
                    ))
                    picked += 1
        return out

    # ==================================================================
    #  2) 症状 → 疾病反向检索
    # ==================================================================
    def _reverse_symptom_facts(self, symptoms: Sequence[str]) -> Tuple[List[RetrievedFact], List[str]]:
        """返回 (事实, 反向检索发现的疾病名列表（按置信度降序）)"""
        facts: List[RetrievedFact] = []
        discovered: List[Tuple[float, str]] = []
        for sym in symptoms:
            nid = self.memory.resolve(sym, "Symptom") or self.memory.resolve(sym)
            if not nid:
                continue
            sym_name, sym_type = self._node_name_type(nid)
            for nb in self.memory.neighbors(nid, ["HAS_SYMPTOM"]):
                edge = nb.get("edge") or {}
                node = nb.get("node") or {}
                # neighbors() 对两个方向都返回，这里只保留「疾病 —HAS_SYMPTOM→ 症状」
                if edge.get("target") != nid:
                    continue
                if str(node.get("type", "")) != "Disease":
                    continue
                conf = float((edge.get("properties") or {}).get("confidence", 0.9) or 0.9)
                facts.append(self._mk(
                    head=str(node.get("name", "")), head_type="Disease",
                    relation="HAS_SYMPTOM", relation_label=REL_LABELS.get("HAS_SYMPTOM", "症状"),
                    tail=sym_name, tail_type=sym_type or "Symptom",
                    confidence=conf, source="kg_1hop",
                    explain=f"反向检索：症状「{sym_name}」→ 疾病「{node.get('name', '')}」",
                ))
                discovered.append((conf, str(node.get("name", ""))))
        discovered.sort(key=lambda kv: -kv[0])
        uniq: List[str] = []
        for _, name in discovered:
            if name and name not in uniq:
                uniq.append(name)
        return facts, uniq

    # ==================================================================
    #  3) 多跳路径事实
    # ==================================================================
    @staticmethod
    def _chain_text(steps: Sequence[Dict[str, Any]]) -> str:
        """把路径步骤渲染成可读推理链：头晕 —HAS_SYMPTOM→ 原发性高血压 —BELONGS_TO→ 心血管内科"""
        if not steps:
            return ""
        parts = [str(steps[0].get("from", ""))]
        for st in steps:
            parts.append(f"—{st.get('relation', '')}→")
            parts.append(str(st.get("to", "")))
        return " ".join(parts)

    def _multi_hop_facts(
        self, symptoms: Sequence[str], targets: Sequence[str], max_hops: int, cap: int = 6,
    ) -> List[RetrievedFact]:
        if not symptoms or not targets:
            return []
        out: List[RetrievedFact] = []
        seen_chains: Set[str] = set()
        for sym in symptoms:
            for dis in targets:
                if sym == dis:
                    continue
                paths: List[List[Dict[str, Any]]] = []
                try:
                    paths = self.memory.multi_hop_paths(sym, dis, max_hops=max_hops, max_paths=2) or []
                except Exception as exc:  # noqa: BLE001
                    logger.debug("multi_hop_paths(%s,%s) 失败：%s", sym, dis, exc)
                if not paths:
                    reasoner = self._get_reasoner()
                    if reasoner is not None:
                        try:
                            paths = reasoner.explain_path(sym, dis, max_hops=max_hops) or []
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("explain_path(%s,%s) 失败：%s", sym, dis, exc)
                for steps in paths:
                    if len(steps) < 2:
                        continue  # 单跳就是直接事实，不重复计入多跳
                    chain = self._chain_text(steps)
                    if not chain or chain in seen_chains:
                        continue
                    seen_chains.add(chain)
                    confs = [float(s.get("confidence", 0.9) or 0.9) for s in steps]
                    out.append(self._mk(
                        head=str(steps[0].get("from", "")),
                        head_type=str(steps[0].get("from_type", "")),
                        relation="RELATED_TO",
                        relation_label="相关（多跳推理）",
                        tail=str(steps[-1].get("to", "")),
                        tail_type=str(steps[-1].get("to_type", "")),
                        confidence=min(confs) if confs else 0.85,
                        source="kg_2hop",
                        explain=chain,
                    ))
                    if len(out) >= cap:
                        return out
        return out

    # ==================================================================
    #  4) 规则推理补全（结果缓存，代价高）
    # ==================================================================
    def _reasoned(self) -> List[Dict[str, Any]]:
        if self._reasoned_cache is not None:
            return self._reasoned_cache
        data: List[Dict[str, Any]] = []
        reasoner = self._get_reasoner()
        if reasoner is not None:
            try:
                data = reasoner.infer_by_rules(limit=60) or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("规则推理补全失败，跳过：%s", exc)
                data = []
        self._reasoned_cache = data
        logger.info("推理补全事实缓存完成：%d 条（仅本次进程计算一次）", len(data))
        return data

    def _reasoned_facts(self, entity_names: Sequence[str], cap: int = 12) -> List[RetrievedFact]:
        if not entity_names:
            return []
        name_set = set(entity_names)
        out: List[RetrievedFact] = []
        for t in self._reasoned():
            head = str(t.get("head", ""))
            tail = str(t.get("tail", ""))
            if head not in name_set and tail not in name_set:
                continue
            method = str(t.get("method", "rule"))
            method_cn = {"rule": "Jena 规则", "gat": "GAT", "gcn": "GCN",
                         "heuristic": "启发式补全"}.get(method, method)
            out.append(self._mk(
                head=head, head_type=str(t.get("head_type", "")),
                relation=str(t.get("relation", "RELATED_TO")),
                relation_label=str(t.get("relation_label") or REL_LABELS.get(
                    str(t.get("relation", "")), str(t.get("relation", "")))),
                tail=tail, tail_type=str(t.get("tail_type", "")),
                confidence=float(t.get("confidence", 0.7) or 0.7),
                source="kg_reasoned",
                explain=f"{method_cn}推理（{t.get('rule_id', '')}）：{truncate(str(t.get('explain', '')), 70)}",
            ))
            if len(out) >= cap:
                break
        return out

    # ==================================================================
    #  5) 相似疾病扩展
    # ==================================================================
    def _similar_facts(self, diseases: Sequence[str], cap: int = 12) -> List[RetrievedFact]:
        out: List[RetrievedFact] = []
        for disease in diseases[:2]:
            try:
                hits = self.memory.search(disease, limit=3, types=["Disease"]) or []
            except Exception as exc:  # noqa: BLE001
                logger.debug("相似疾病检索失败：%s", exc)
                continue
            for h in hits:
                name = str(h.get("name", ""))
                if not name or name == disease:
                    continue
                score = float(h.get("score", 0.0) or 0.0)
                try:
                    triples = self.graph_service.triples_of(
                        name, rels=SIMILAR_RELS, direction="out", limit=20) or []
                except Exception as exc:  # noqa: BLE001
                    logger.debug("相似疾病三元组失败：%s", exc)
                    continue
                for t in triples:
                    out.append(self._mk(
                        head=str(t.get("head", name)), head_type=str(t.get("head_type", "Disease")),
                        relation=str(t.get("relation", "")),
                        relation_label=str(t.get("relation_label")
                                           or REL_LABELS.get(str(t.get("relation", "")), "")),
                        tail=str(t.get("tail", "")), tail_type=str(t.get("tail_type", "")),
                        confidence=float(t.get("confidence", 0.9) or 0.9) * 0.95,
                        source="kg_similar",
                        explain=f"相似疾病「{name}」扩展（相似度 {score:.2f}，相似于「{disease}」）",
                    ))
                    if len(out) >= cap:
                        return out
        return out

    # ==================================================================
    #  排序
    # ==================================================================
    def _proximity(
        self, fact: RetrievedFact, primary: str,
        entity_names: Set[str], neighbour_names: Set[str],
    ) -> float:
        heads = {fact.head, fact.tail}
        if primary and primary in heads:
            return 1.0
        if heads & entity_names:
            return 0.8
        if heads & neighbour_names:
            return 0.55
        return 0.3

    def _degree_normalized(self, fact: RetrievedFact) -> float:
        best = 0
        for name in (fact.head, fact.tail):
            nid = self.memory.resolve(name)
            if not nid:
                continue
            best = max(best, int(self.memory.nodes.get(nid, {}).get("degree", 0) or 0))
        return min(1.0, best / float(self._max_degree()))

    def _rank(
        self, facts: List[RetrievedFact], intent: str,
        primary: str, entity_names: Sequence[str], neighbour_names: Set[str], limit: int,
    ) -> List[RetrievedFact]:
        prior_map = INTENT_SLOT_PRIOR.get(intent) or INTENT_SLOT_PRIOR["_default"]
        name_set = set(entity_names)

        # 1) 按 (head, relation, tail) 去重；同分时优先"更直接"的召回路径
        source_rank = {"kg_1hop": 0, "kg_2hop": 1, "kg_similar": 2, "kg_reasoned": 3}
        merged: Dict[Tuple[str, str, str], RetrievedFact] = {}
        for f in facts:
            if not f.head or not f.tail or not f.relation:
                continue
            old = merged.get(f.key())
            if old is None:
                merged[f.key()] = f
                continue
            better = (f.confidence, -source_rank.get(f.source, 9)) > \
                     (old.confidence, -source_rank.get(old.source, 9))
            if better:
                merged[f.key()] = f

        # 2) 打分
        for f in merged.values():
            conf = max(0.0, min(1.0, float(f.confidence)))
            prior = float(prior_map.get(f.relation, prior_map.get("_default", 0.6)))
            prox = self._proximity(f, primary, name_set, neighbour_names)
            deg = self._degree_normalized(f)
            base = (W_CONFIDENCE * conf + W_SLOT_PRIOR * prior
                    + W_PROXIMITY * prox + W_DEGREE * deg)
            f.score = round(base * SOURCE_WEIGHT.get(f.source, 0.9), 4)

        ranked = sorted(merged.values(), key=lambda x: (-x.score, -x.confidence, x.head))

        # 3) 低置信度过滤（仅当过滤后仍 ≥3 条，避免把上下文清空）
        min_conf = float(getattr(settings, "RAG_MIN_CONFIDENCE", 0.55) or 0.55)
        kept = [f for f in ranked if f.confidence >= min_conf]
        if len(kept) >= 3:
            ranked = kept

        return ranked[:max(1, int(limit))]

    def _category(self, fact: RetrievedFact, primary: str) -> str:
        """召回路径归类（用于多样性名额分配）"""
        if fact.source == "kg_reasoned":
            return "kg_reasoned"
        if fact.source == "kg_2hop":
            return "kg_2hop"
        if fact.source == "kg_similar":
            return "related"
        if primary and fact.head == primary:
            return "primary"
        return "related"

    def _select_diverse(
        self, ranked: List[RetrievedFact], primary: str, top_k: int,
    ) -> List[RetrievedFact]:
        """按召回路径保留名额后再按分数截断：既保证主实体事实占多数，又保证跨疾病/推理事实不被挤空"""
        if len(ranked) <= top_k:
            return ranked
        buckets: Dict[str, List[RetrievedFact]] = {}
        for f in ranked:
            buckets.setdefault(self._category(f, primary), []).append(f)

        picked: List[RetrievedFact] = []
        picked_keys: Set[Tuple[str, str, str]] = set()
        for cat, quota in DIVERSITY_RESERVE.items():
            for f in buckets.get(cat, [])[:quota]:
                if f.key() not in picked_keys:
                    picked.append(f)
                    picked_keys.add(f.key())
        for f in ranked:  # 剩余名额按分数回填
            if len(picked) >= top_k:
                break
            if f.key() not in picked_keys:
                picked.append(f)
                picked_keys.add(f.key())

        picked = picked[:top_k]
        picked.sort(key=lambda x: (-x.score, -x.confidence, x.head))
        return picked

    # ==================================================================
    #  主入口
    # ==================================================================
    async def retrieve(
        self,
        question: str,
        entities: Optional[Dict[str, List[str]]] = None,
        intent: str = "disease_query",
        top_k: int = 12,
        max_hops: int = 2,
    ) -> List[RetrievedFact]:
        """五路召回 → 去重 → 排序 → 编号（KG-1 … KG-n）→ 补溯源来源"""
        t0 = time.perf_counter()
        entities = entities if isinstance(entities, dict) else {}
        top_k = int(top_k or getattr(settings, "RAG_TOP_K", 12) or 12)
        max_hops = int(max_hops or getattr(settings, "RAG_MAX_HOPS", 2) or 2)

        diseases = _names(entities, "diseases")
        symptoms = _names(entities, "symptoms")
        all_names = _names(entities, *ENTITY_KEYS)

        candidates: List[RetrievedFact] = []
        # A) 槽位驱动直接事实
        candidates += self._slot_direct_facts(diseases)
        # B) 症状 → 疾病反向检索
        reverse_facts, discovered = self._reverse_symptom_facts(symptoms)
        candidates += reverse_facts
        # C) 多跳路径
        hop_targets: List[str] = list(diseases)
        for name in discovered:
            if name not in hop_targets:
                hop_targets.append(name)
        candidates += self._multi_hop_facts(symptoms, hop_targets[:3], max_hops)
        # D) 规则推理补全（命中查询实体者）
        candidates += self._reasoned_facts(all_names or diseases)
        # E) 相似疾病扩展（事实不足 top_k 时才做，避免稀释直接事实）
        if len(candidates) < top_k:
            seed = list(diseases) or discovered[:1]
            candidates += self._similar_facts(seed)

        primary = diseases[0] if diseases else (discovered[0] if discovered else "")
        neighbour_names = self._neighbour_names(all_names)
        ranked = self._rank(candidates, intent, primary, all_names, neighbour_names,
                            max(top_k * 4, 24))
        ranked = self._select_diverse(ranked, primary, top_k)

        # 编号 + 溯源（编号即排序名次，与回答中的 [KG-n] 一一对应）
        for i, f in enumerate(ranked, start=1):
            f.triple_id = f"KG-{i}"
            f.sources = self._evidence_sources(f, f.triple_id)

        self.last_stats = self._count_sources(ranked)
        cost = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "混合检索完成：直接事实 %d / 反向检索 %d / 多跳 %d / 推理补全 %d / 相似扩展 %d（合计 %d 条，%.1fms）",
            self.last_stats.get("kg_1hop", 0) - self.last_stats.get("reverse", 0),
            self.last_stats.get("reverse", 0), self.last_stats.get("kg_2hop", 0),
            self.last_stats.get("kg_reasoned", 0), self.last_stats.get("kg_similar", 0),
            len(ranked), cost,
        )
        await asyncio.sleep(0)  # 让出事件循环，保持异步语义
        return ranked

    @staticmethod
    def _count_sources(facts: Sequence[RetrievedFact]) -> Dict[str, int]:
        stats: Dict[str, int] = {"kg_1hop": 0, "kg_2hop": 0, "kg_reasoned": 0, "kg_similar": 0, "reverse": 0}
        for f in facts:
            stats[f.source] = stats.get(f.source, 0) + 1
            if f.explain.startswith("反向检索"):
                stats["reverse"] = stats.get("reverse", 0) + 1
        return stats

    def retrieve_sync(
        self,
        question: str,
        entities: Optional[Dict[str, List[str]]] = None,
        intent: str = "disease_query",
        top_k: int = 12,
        max_hops: int = 2,
    ) -> List[RetrievedFact]:
        """同步封装：已有事件循环时放进子线程执行，避免 RuntimeError"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.retrieve(question, entities, intent, top_k, max_hops))

        box: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                box["result"] = asyncio.run(
                    self.retrieve(question, entities, intent, top_k, max_hops))
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        thread = threading.Thread(target=_worker, name="rag-retrieve-sync", daemon=True)
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]
        return list(box.get("result") or [])


__all__ = ["HybridRetriever", "RetrievedFact", "SLOT_QUOTAS", "INTENT_SLOT_PRIOR",
           "SOURCE_WEIGHT", "DIVERSITY_RESERVE"]
