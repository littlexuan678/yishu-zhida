# -*- coding: utf-8 -*-
"""
内存图存储（MemoryGraphStore）
==============================
作用
----
1. **离线降级后端**：当 Neo4j 不可用时，`Neo4jClient` 自动降级到本实现，
   保证"智愈医典"在任何环境下都能完整演示（图谱可视化 / 疾病查询 / 问答溯源 / 数据分析）。
2. **算法侧图对象**：为 GAT/GCN 图谱补全、Jena 规则推理提供统一的 `nodes/edges` 视图。

数据结构
--------
  nodes: {node_id: {"id","name","type","label","color","degree","properties"}}
  edges: [{"source","target","rel","label","properties"}]
  index: {(type, name): node_id}  —— 支持 O(1) 实体链接
  adjacency: {node_id: [edge_index, ...]}

同时实现了一小部分 **Cypher 子集**（`query()`），用于兼容直接调用 Cypher 的场景。
"""
from __future__ import annotations

import json
import re
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.config import settings
from app.schemas import ENTITY_TYPE_META
from app.utils.logger import get_logger
from app.utils.text import similarity

logger = get_logger(__name__)

#: 关系英文 → 中文标签
REL_LABELS: Dict[str, str] = {
    "HAS_SYMPTOM": "症状",
    "TREATED_BY": "治疗",
    "USES_DRUG": "用药",
    "BELONGS_TO": "科室",
    "NEEDS_CHECK": "检查",
    "AFFECTS": "易感人群",
    "HAS_COMPLICATION": "并发症",
    "DIFFERENTIAL_WITH": "鉴别诊断",
    "PROVES": "知识来源",
    "IS_INFECTIOUS": "传染性",
    "RELATED_TO": "相关",
}


class MemoryGraphStore:
    """内存知识图谱（线程安全单例）"""

    _instance: Optional["MemoryGraphStore"] = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.index: Dict[Tuple[str, str], str] = {}
        self.adjacency: Dict[str, List[int]] = defaultdict(list)
        self.diseases: Dict[str, Dict[str, Any]] = {}      # disease_id -> 完整疾病对象
        self.disease_by_name: Dict[str, str] = {}          # name -> disease_id
        self._loaded = False
        self.load()

    @classmethod
    def instance(cls) -> "MemoryGraphStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ==================================================================
    #  加载种子数据
    # ==================================================================
    def load(self, seed_dir: Optional[Path] = None) -> None:
        seed_dir = seed_dir or settings.seed_dir
        diseases_f = seed_dir / "diseases.json"
        relations_f = seed_dir / "relations.json"
        if not diseases_f.exists():
            logger.error("疾病种子文件不存在：%s", diseases_f)
            self._loaded = True
            return

        with diseases_f.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        disease_list = payload.get("diseases", payload if isinstance(payload, list) else [])

        # ---- 1) 疾病节点 + 疾病属性节点 ----
        for d in disease_list:
            self._add_disease(d)

        # ---- 2) 三元组关系 ----
        if relations_f.exists():
            with relations_f.open("r", encoding="utf-8") as f:
                rel_payload = json.load(f)
            for r in rel_payload.get("relations", []):
                self.add_edge(
                    head=r["head"],
                    head_type=r.get("head_type", "Disease"),
                    rel=r["relation"],
                    tail=r["tail"],
                    tail_type=r.get("tail_type", "Symptom"),
                    confidence=float(r.get("confidence", 0.95)),
                    weight=float(r.get("weight", 1.0)),
                    source_ref=r.get("source_ref", ""),
                )

        self._recompute_degrees()
        self._loaded = True
        logger.info(
            "内存图谱加载完成：节点 %d，关系 %d，疾病 %d",
            len(self.nodes), len(self.edges), len(self.diseases),
        )

    def reload(self) -> None:
        with self._lock:
            self.nodes.clear()
            self.edges.clear()
            self.index.clear()
            self.adjacency.clear()
            self.diseases.clear()
            self.disease_by_name.clear()
            self._loaded = False
            self.load()

    # ------------------------------------------------------------------
    def _add_disease(self, d: Dict[str, Any]) -> None:
        did = d["disease_id"]
        self.diseases[did] = d
        self.disease_by_name[d["name"]] = did

        props = {k: v for k, v in d.items() if k not in ("disease_id", "name", "sources")}
        props["sources"] = d.get("sources", [])
        self.add_node(d["name"], "Disease", properties=props, node_id=f"Disease:{d['name']}")

        cat1 = d.get("category1")
        if cat1:
            self.add_edge(d["name"], "Disease", "BELONGS_TO", cat1, "Department")
        cat2 = d.get("category2")
        if cat2 and cat2 != cat1:
            self.add_edge(d["name"], "Disease", "BELONGS_TO", cat2, "Department")
        dept = d.get("department")
        if dept and dept not in (cat1, cat2):
            self.add_edge(d["name"], "Disease", "BELONGS_TO", dept, "Department")

        for s in d.get("symptoms", []):
            self.add_edge(d["name"], "Disease", "HAS_SYMPTOM", s, "Symptom")
        for t in d.get("treatments", []) or ([d["treatment"]] if d.get("treatment") else []):
            self.add_edge(d["name"], "Disease", "TREATED_BY", t, "Treatment")
        for dr in d.get("drugs", []):
            self.add_edge(d["name"], "Disease", "USES_DRUG", dr, "Drug")
        for c in d.get("checks", []):
            self.add_edge(d["name"], "Disease", "NEEDS_CHECK", c, "Check")
        for cp in d.get("complications", []):
            self.add_edge(d["name"], "Disease", "HAS_COMPLICATION", cp, "Disease")
        for df in d.get("differential", []):
            self.add_edge(d["name"], "Disease", "DIFFERENTIAL_WITH", df, "Disease")

    # ------------------------------------------------------------------
    def add_node(
        self,
        name: str,
        ntype: str,
        properties: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        name = (name or "").strip()
        if not name:
            return ""
        nid = node_id or f"{ntype}:{name}"
        if nid in self.nodes:
            if properties:
                self.nodes[nid]["properties"].update({k: v for k, v in properties.items() if v})
            return nid
        meta = ENTITY_TYPE_META.get(ntype, {"label": ntype, "color": "#909399"})
        self.nodes[nid] = {
            "id": nid,
            "name": name,
            "type": ntype,
            "label": meta["label"],
            "color": meta["color"],
            "degree": 0,
            "properties": properties or {},
        }
        self.index[(ntype, name)] = nid
        return nid

    def add_edge(
        self,
        head: str,
        head_type: str,
        rel: str,
        tail: str,
        tail_type: str,
        confidence: float = 0.95,
        weight: float = 1.0,
        source_ref: str = "",
    ) -> None:
        sid = self.add_node(head, head_type)
        tid = self.add_node(tail, tail_type)
        if not sid or not tid or sid == tid:
            return
        for i in self.adjacency[sid]:
            e = self.edges[i]
            if e["source"] == sid and e["target"] == tid and e["rel"] == rel:
                return  # 去重
        idx = len(self.edges)
        self.edges.append({
            "source": sid,
            "target": tid,
            "rel": rel,
            "label": REL_LABELS.get(rel, rel),
            "properties": {"confidence": confidence, "weight": weight, "source_ref": source_ref},
        })
        self.adjacency[sid].append(idx)
        self.adjacency[tid].append(idx)

    def _recompute_degrees(self) -> None:
        deg: Dict[str, int] = defaultdict(int)
        for e in self.edges:
            deg[e["source"]] += 1
            deg[e["target"]] += 1
        for nid, n in self.nodes.items():
            n["degree"] = deg.get(nid, 0)

    # ==================================================================
    #  查询能力
    # ==================================================================
    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def resolve(self, text: str, ntype: Optional[str] = None) -> Optional[str]:
        """把实体名解析为 node_id（精确 → 不区分大小写 → 别名 → 模糊）"""
        text = (text or "").strip()
        if not text:
            return None
        if ntype:
            nid = self.index.get((ntype, text))
            if nid:
                return nid
        # 全类型精确
        for (t, n), nid in self.index.items():
            if n == text and (ntype is None or t == ntype):
                return nid
        # 别名
        low = text.lower()
        for nid, n in self.nodes.items():
            if ntype and n["type"] != ntype:
                continue
            alias = n["properties"].get("alias") or []
            if isinstance(alias, str):
                alias = [alias]
            for a in alias:
                if str(a).lower() == low:
                    return nid
        # 模糊
        cands = [n["name"] for n in self.nodes.values() if not ntype or n["type"] == ntype]
        best, score = None, 0.0
        for c in cands:
            s = similarity(text, c)
            if s > score:
                best, score = c, s
        if best and score >= 0.62:
            t = ntype or self.nodes[self.index_of(best, ntype)]["type"]
            return self.index_of(best, t)
        return None

    def index_of(self, name: str, ntype: Optional[str] = None) -> str:
        if ntype:
            return self.index.get((ntype, name), "")
        for (t, n), nid in self.index.items():
            if n == name:
                return nid
        return ""

    def search(self, keyword: str, limit: int = 20, types: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """实体检索（前缀/包含/模糊），按相关度排序"""
        kw = (keyword or "").strip()
        if not kw:
            return []
        type_set = set(types) if types else None
        scored: List[Tuple[float, Dict[str, Any]]] = []
        low = kw.lower()
        for nid, n in self.nodes.items():
            if type_set and n["type"] not in type_set:
                continue
            name = n["name"]
            if name == kw:
                score = 1.0
            elif name.lower().startswith(low):
                score = 0.94
            elif low in name.lower():
                score = 0.88
            else:
                alias = n["properties"].get("alias") or []
                if isinstance(alias, str):
                    alias = [alias]
                hit = any(low in str(a).lower() for a in alias)
                score = 0.8 if hit else similarity(kw, name)
            if score >= 0.45:
                scored.append((score, n))
        scored.sort(key=lambda kv: (-kv[0], -kv[1]["degree"]))
        return [dict(n, score=round(s, 3)) for s, n in scored[:limit]]

    def neighbors(self, node_id: str, rels: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """一跳邻居，返回 [(边, 对端节点)]"""
        rel_set = set(rels) if rels else None
        out = []
        for i in self.adjacency.get(node_id, []):
            e = self.edges[i]
            if rel_set and e["rel"] not in rel_set:
                continue
            other = e["target"] if e["source"] == node_id else e["source"]
            out.append({"edge": e, "node": self.nodes.get(other)})
        return out

    def subgraph(
        self,
        entity: str,
        depth: int = 1,
        limit: int = 200,
        rel_filter: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """
        BFS 抽取以 entity 为中心的子图（PPT 知识图谱可视化模块数据源）

        两阶段实现，保证 `link_count` 准确与 `truncated` 语义清晰：
          ① **选点**：BFS 收集节点集合（受 `limit` 约束）
          ② **建边**：扫描节点集合内**全部**边，取两端都在集合内的边
             说明：若在 BFS 过程中顺带建边，会漏掉"两端都已访问"的边
             （例如 A 的两个邻居 B、C 之间也存在关系），导致前端图不完整。
          `truncated` 表示"因 limit 约束而**放弃**了部分可达节点"，
          正常情况下（邻居数未超限）为 False。
        """
        center_id = self.resolve(entity)
        if not center_id:
            return {
                "nodes": [], "links": [], "node_count": 0, "link_count": 0,
                "type_count": 0, "center": entity, "depth": depth, "truncated": False,
                "message": f"知识库中未找到实体「{entity}」，请尝试其他名称",
            }

        limit = max(1, int(limit))
        rel_set = set(rel_filter) if rel_filter else None

        # ---------- 阶段 ①：BFS 选点 ----------
        visited: Set[str] = {center_id}
        frontier = deque([(center_id, 0)])
        truncated = False

        while frontier:
            nid, d = frontier.popleft()
            if d >= depth:
                continue
            for i in self.adjacency.get(nid, []):
                e = self.edges[i]
                if rel_set and e["rel"] not in rel_set:
                    continue
                other = e["target"] if e["source"] == nid else e["source"]
                if other in visited:
                    continue
                if len(visited) >= limit:
                    truncated = True          # 因 limit 放弃了可达节点
                    continue
                visited.add(other)
                frontier.append((other, d + 1))

        # ---------- 阶段 ②：建边（节点集合内的全部边）----------
        link_idx: Set[int] = set()
        for nid in visited:
            for i in self.adjacency.get(nid, []):
                e = self.edges[i]
                if rel_set and e["rel"] not in rel_set:
                    continue
                if e["source"] in visited and e["target"] in visited:
                    link_idx.add(i)

        nodes = [self.nodes[n] for n in visited if n in self.nodes]
        links = [self.edges[i] for i in sorted(link_idx)]
        return {
            "nodes": nodes,
            "links": links,
            "node_count": len(nodes),
            "link_count": len(links),
            "type_count": len({n["type"] for n in nodes}),
            "center": self.nodes[center_id]["name"],
            "depth": depth,
            "truncated": truncated,
            "message": "",
        }

    def multi_hop_paths(
        self, start: str, end: str, max_hops: int = 3, max_paths: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """两实体之间的多跳路径（可解释推理链路的核心）"""
        s = self.resolve(start)
        t = self.resolve(end)
        if not s or not t:
            return []
        paths: List[List[Dict[str, Any]]] = []
        queue: deque = deque([(s, [s])])
        while queue and len(paths) < max_paths:
            cur, path = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue
            for i in self.adjacency.get(cur, []):
                e = self.edges[i]
                nxt = e["target"] if e["source"] == cur else e["source"]
                if nxt in path:
                    continue
                new_path = path + [nxt]
                if nxt == t:
                    paths.append(self._path_to_steps(new_path))
                    if len(paths) >= max_paths:
                        break
                else:
                    queue.append((nxt, new_path))
        return paths

    def _path_to_steps(self, path: List[str]) -> List[Dict[str, Any]]:
        steps = []
        for a, b in zip(path, path[1:]):
            e = next((x for x in self.edges
                      if (x["source"] == a and x["target"] == b) or (x["source"] == b and x["target"] == a)), None)
            if not e:
                continue
            steps.append({
                "from": self.nodes[a]["name"], "from_type": self.nodes[a]["type"],
                "relation": e["rel"], "relation_label": e["label"],
                "to": self.nodes[b]["name"], "to_type": self.nodes[b]["type"],
                "confidence": e["properties"].get("confidence", 0.95),
            })
        return steps

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """图谱统计（PPT：节点数 / 关系数 / 实体类型）"""
        by_type: Dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            by_type[n["type"]] += 1
        entity_types = [
            {
                "type": t,
                "label": ENTITY_TYPE_META.get(t, {}).get("label", t),
                "color": ENTITY_TYPE_META.get(t, {}).get("color", "#909399"),
                "count": c,
            }
            for t, c in sorted(by_type.items(), key=lambda kv: -kv[1])
        ]
        disease_objs = list(self.diseases.values())
        return {
            "total_nodes": len(self.nodes),
            "total_links": len(self.edges),
            "entity_type_count": len(by_type),
            "entity_types": entity_types,
            "disease_count": by_type.get("Disease", 0),
            "symptom_count": by_type.get("Symptom", 0),
            "drug_count": by_type.get("Drug", 0),
            "department_count": by_type.get("Department", 0),
            "treatment_count": by_type.get("Treatment", 0),
            "check_count": by_type.get("Check", 0),
            "population_count": by_type.get("Population", 0),
            "source_count": by_type.get("Source", 0),
            "infectious_count": sum(1 for d in disease_objs if d.get("is_infectious")),
            "category1_count": len({d.get("category1") for d in disease_objs if d.get("category1")}),
            "category1_distribution": self._count_by(disease_objs, "category1"),
            "data_source": "memory",
        }

    @staticmethod
    def _count_by(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        dist: Dict[str, int] = defaultdict(int)
        for it in items:
            v = it.get(key)
            if v:
                dist[v] += 1
        return dict(sorted(dist.items(), key=lambda kv: -kv[1]))

    # ==================================================================
    #  Cypher 子集（兼容直接调用 Cypher 的场景）
    # ==================================================================
    _RE_COUNT_NODES = re.compile(r"MATCH\s*\(\s*\w*\s*(?::(\w+))?\s*\)\s*RETURN\s+count", re.I)
    _RE_COUNT_RELS = re.compile(r"MATCH\s*\(\s*\)\s*\[.*?\]\s*->\s*\(\s*\)\s*RETURN\s+count", re.I)

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        支持 Cypher 子集：
          * MATCH (n[:Label]) RETURN count(n) AS c
          * MATCH ()-[r]->() RETURN count(r) AS c
          * MATCH (n[:Label]) RETURN n.name AS name, n.type AS type LIMIT k
          * MATCH (a)-[r]->(b) WHERE ... RETURN ...（退化为全边遍历 + 关键词过滤）
        其余语句返回空结果并在日志中提示，避免静默错误。
        """
        params = params or {}
        stmt = " ".join((cypher or "").split())
        limit = self._extract_limit(stmt)

        m = self._RE_COUNT_RELS.search(stmt)
        if m:
            return ["c"], [{"c": len(self.edges)}]

        m = self._RE_COUNT_NODES.search(stmt)
        if m:
            label = m.group(1)
            return ["c"], [{"c": len(self._nodes_of(label))}]

        if re.search(r"\bcount\s*\(", stmt, re.I) and "MATCH" in stmt.upper():
            # 通用 count：按 RETURN 别名返回
            alias = self._extract_alias(stmt) or "c"
            label = self._extract_label(stmt)
            if label:
                return [alias], [{alias: len(self._nodes_of(label))}]
            return [alias], [{alias: len(self.nodes)}]

        if stmt.upper().startswith("MATCH") and "RETURN" in stmt.upper():
            label = self._extract_label(stmt)
            nodes = self._nodes_of(label)
            kw = params.get("kw") or self._extract_literal(stmt, "name")
            if kw:
                nodes = [n for n in nodes if kw in n["name"]]
            nodes = sorted(nodes, key=lambda n: -n["degree"])[:limit]
            return ["name", "type", "degree"], [
                {"name": n["name"], "type": n["type"], "degree": n["degree"]} for n in nodes
            ]

        if stmt.upper().startswith("SHOW"):
            return ["result"], [{"result": "memory backend: SHOW 语句已忽略"}]

        logger.warning("内存后端不支持的 Cypher（返回空结果）：%s", stmt[:160])
        return [], []

    def _nodes_of(self, label: Optional[str]) -> List[Dict[str, Any]]:
        if not label:
            return list(self.nodes.values())
        return [n for n in self.nodes.values() if n["type"].lower() == label.lower()]

    @staticmethod
    def _extract_limit(stmt: str) -> int:
        m = re.search(r"\bLIMIT\s+(\d+)", stmt, re.I)
        return int(m.group(1)) if m else 200

    @staticmethod
    def _extract_alias(stmt: str) -> Optional[str]:
        m = re.search(r"\bAS\s+(\w+)", stmt, re.I)
        return m.group(1) if m else None

    @staticmethod
    def _extract_label(stmt: str) -> Optional[str]:
        m = re.search(r"\(\s*\w*\s*:\s*(\w+)", stmt)
        return m.group(1) if m else None

    @staticmethod
    def _extract_literal(stmt: str, key: str) -> Optional[str]:
        m = re.search(rf"{key}\s*=\s*['\"]([^'\"]+)['\"]", stmt)
        return m.group(1) if m else None
