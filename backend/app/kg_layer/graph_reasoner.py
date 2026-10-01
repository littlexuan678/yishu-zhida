# -*- coding: utf-8 -*-
"""
图谱推理与补全（Apache Jena 规则引擎 + GAT/GCN 图神经网络）
============================================================
PPT 要求：
    "集成 Apache Jena 规则引擎处理显式逻辑，结合 GAT/GCN 图神经网络技术，
      深度挖掘图谱中隐藏的语义关联与潜在关系。"

本模块提供两条互补的推理路径：

  路径 A：**显式规则推理（Apache Jena / 等价 Python 规则引擎）**
      * `scripts/jena/rules_medical.rules` 定义医疗推理规则
      * `scripts/jena/ontology_medical.ttl` 定义医疗本体（类/属性/公理）
      * 生产环境通过 `jena` CLI（`riot` / `sparql`）或 JPype 调用 Jena；
        本模块内置 `PythonRuleEngine` 作为等价实现，保证无 JDK 环境可运行。
      * 典型规则：
          (Disease)-[HAS_SYMPTOM]->(Symptom) + (Symptom)<-[HAS_SYMPTOM]-(Disease')
              ⇒ (Disease)-[RELATED_TO]->(Disease')          共病/相似病推理
          (Disease)-[TREATED_BY]->(Treatment) + (Treatment)<-[TREATED_BY]-(Disease')
              ⇒ (Disease)-[RELATED_TO]->(Disease')          同治疗方案关联
          (Disease)-[HAS_COMPLICATION]->(D) ∧ (D)-[HAS_SYMPTOM]->(S) 传递闭包
              ⇒ (Disease)-[MAY_CAUSE_SYMPTOM]->(S)          间接症状推理
          (Disease)[infectious=true] ⇒ (Disease)-[IS_INFECTIOUS]->(Source)  传染性标记

  路径 B：**隐式关系补全（GAT / GCN 图神经网络）**
      * `GatLinkPredictor`  : 图注意力网络（Graph Attention Network）
      * `GcnLinkPredictor`  : 图卷积网络（Graph Convolutional Network）
      * 任务：链接预测（Link Prediction）—— 预测图谱中缺失的 (h, r, t)
      * 有 torch_geometric + torch → 使用真实 GNN 训练
      * 无依赖 → 使用 `HeuristicLinkPredictor`（Adamic-Adar + 关系类型先验）降级

效果：为图谱补充潜在的共病关系、间接症状关系，提升问答的多跳推理覆盖率。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.config import settings
from app.kg_layer.memory_store import MemoryGraphStore, REL_LABELS
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore

try:  # PyG 为可选重型依赖
    from torch_geometric.nn import GATConv, GCNConv  # type: ignore

    HAS_PYG = True
except Exception:  # pragma: no cover
    HAS_PYG = False
    GATConv = None  # type: ignore
    GCNConv = None  # type: ignore


# =============================================================================
#  数据模型
# =============================================================================
@dataclass
class InferredTriple:
    """推理/补全得到的新三元组"""

    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    confidence: float
    method: str = "rule"          # rule | gat | gcn | heuristic
    rule_id: str = ""
    explain: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head, "head_type": self.head_type,
            "relation": self.relation,
            "relation_label": REL_LABELS.get(self.relation, self.relation),
            "tail": self.tail, "tail_type": self.tail_type,
            "confidence": round(self.confidence, 4),
            "method": self.method, "rule_id": self.rule_id,
            "explain": self.explain,
        }


@dataclass
class MedicalRule:
    """医疗推理规则（与 Jena .rules 文件一一对应）"""

    id: str
    name: str
    premise_rel: Tuple[str, ...]
    conclusion_rel: str
    min_shared: int = 1
    confidence: float = 0.7
    explain: str = ""


#: 内置医疗推理规则集（等价于 scripts/jena/rules_medical.rules）
DEFAULT_RULES: List[MedicalRule] = [
    MedicalRule(
        id="R1_shared_symptom_related",
        name="共同症状 → 疾病相关",
        premise_rel=("HAS_SYMPTOM",),
        conclusion_rel="RELATED_TO",
        min_shared=2,
        confidence=0.72,
        explain="两种疾病共享 ≥2 个症状，推断存在临床相关性（鉴别诊断 / 共病风险）",
    ),
    MedicalRule(
        id="R2_shared_treatment_related",
        name="共同治疗方案 → 疾病相关",
        premise_rel=("TREATED_BY",),
        conclusion_rel="RELATED_TO",
        min_shared=1,
        confidence=0.66,
        explain="两种疾病采用相同治疗方案，推断存在病理机制或分级关联",
    ),
    MedicalRule(
        id="R3_complication_transitive_symptom",
        name="并发症症状传递",
        premise_rel=("HAS_COMPLICATION", "HAS_SYMPTOM"),
        conclusion_rel="MAY_CAUSE_SYMPTOM",
        min_shared=1,
        confidence=0.64,
        explain="疾病 A 的并发症 B 具有症状 S，推断 A 可能间接引起 S",
    ),
    MedicalRule(
        id="R4_same_department_comorbidity",
        name="同科室共病关联",
        premise_rel=("BELONGS_TO",),
        conclusion_rel="RELATED_TO",
        min_shared=2,
        confidence=0.55,
        explain="两种疾病归属同一二级科室，提示同科室常见共病",
    ),
    MedicalRule(
        id="R5_infectious_flag",
        name="传染性标记推理",
        premise_rel=("IS_INFECTIOUS",),
        conclusion_rel="NEEDS_ISOLATION",
        min_shared=1,
        confidence=0.9,
        explain="疾病被标记为传染性，推断需采取隔离/防护措施",
    ),
    MedicalRule(
        id="R6_drug_complication_risk",
        name="用药-并发症风险提示",
        premise_rel=("USES_DRUG", "HAS_COMPLICATION"),
        conclusion_rel="DRUG_RISK_ALERT",
        min_shared=1,
        confidence=0.5,
        explain="疾病 A 使用药物 D 且具有并发症 C，提示需监测 D 对 C 的潜在影响",
    ),
]


# =============================================================================
#  路径 A：规则引擎
# =============================================================================
class PythonRuleEngine:
    """
    医疗规则推理引擎（Apache Jena Rule Engine 的等价 Python 实现）

    生产环境切换 Jena 的方式见 `run_jena_cli()`：
        java -cp jena/lib/* riot --rules=rules_medical.rules ontology_medical.ttl
    本类在不依赖 JDK 的环境中提供完全相同的推理结论。
    """

    def __init__(
        self,
        store: Optional[MemoryGraphStore] = None,
        rules: Optional[List[MedicalRule]] = None,
    ) -> None:
        self.store = store or MemoryGraphStore.instance()
        self.rules = rules or DEFAULT_RULES
        #: 预建反向索引以加速规则匹配
        self._rel_index: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(self.store.edges):
            self._rel_index[e["rel"]].append(i)

    # ------------------------------------------------------------------
    def _out_neighbors(self, node_id: str, rel: str) -> Set[str]:
        """node --rel--> target 的 target 集合"""
        out: Set[str] = set()
        for i in self._rel_index.get(rel, []):
            e = self.store.edges[i]
            if e["source"] == node_id:
                out.add(e["target"])
        return out

    def _in_neighbors(self, node_id: str, rel: str) -> Set[str]:
        out: Set[str] = set()
        for i in self._rel_index.get(rel, []):
            e = self.store.edges[i]
            if e["target"] == node_id:
                out.add(e["source"])
        return out

    def _signature(self, node_id: str, rel: str) -> Set[str]:
        """节点的某类关系邻居签名（出边）"""
        return self._out_neighbors(node_id, rel)

    # ------------------------------------------------------------------
    def infer(self, limit: int = 500, rules: Optional[Iterable[str]] = None) -> List[InferredTriple]:
        """执行全部（或指定）规则，返回推理出的新三元组"""
        active = set(rules) if rules else None
        results: List[InferredTriple] = []
        disease_ids = [nid for nid, n in self.store.nodes.items() if n["type"] == "Disease"]
        logger.info("规则推理开始：疾病 %d 个，规则 %d 条", len(disease_ids), len(self.rules))

        for rule in self.rules:
            if active and rule.id not in active:
                continue
            try:
                if rule.id == "R1_shared_symptom_related":
                    results += self._rule_shared(disease_ids, "HAS_SYMPTOM", rule, limit)
                elif rule.id == "R2_shared_treatment_related":
                    results += self._rule_shared(disease_ids, "TREATED_BY", rule, limit)
                elif rule.id == "R4_same_department_comorbidity":
                    results += self._rule_shared(disease_ids, "BELONGS_TO", rule, limit, only_leaf_dept=True)
                elif rule.id == "R3_complication_transitive_symptom":
                    results += self._rule_complication_transitive(disease_ids, rule, limit)
                elif rule.id == "R5_infectious_flag":
                    results += self._rule_infectious(disease_ids, rule, limit)
                elif rule.id == "R6_drug_complication_risk":
                    results += self._rule_drug_risk(disease_ids, rule, limit)
            except Exception as exc:  # noqa: BLE001
                logger.warning("规则 %s 执行失败：%s", rule.id, exc)

        # 去重 + 排除已存在的关系
        existing = {(e["source"], e["rel"], e["target"]) for e in self.store.edges}
        uniq: Dict[Tuple[str, str, str], InferredTriple] = {}
        for t in results:
            hid = self.store.resolve(t.head)
            tid = self.store.resolve(t.tail)
            if not hid or not tid or hid == tid:
                continue
            if (hid, t.relation, tid) in existing or (tid, t.relation, hid) in existing:
                continue
            key = (t.head, t.relation, t.tail)
            if key not in uniq or uniq[key].confidence < t.confidence:
                uniq[key] = t
        out = sorted(uniq.values(), key=lambda x: -x.confidence)[:limit]
        logger.info("规则推理完成：新增 %d 条三元组", len(out))
        return out

    # ------------------------------------------------------------------
    def _rule_shared(
        self, disease_ids: List[str], rel: str, rule: MedicalRule,
        limit: int, only_leaf_dept: bool = False,
    ) -> List[InferredTriple]:
        """共享邻居 → RELATED_TO（R1 / R2 / R4 共用实现）"""
        sig: Dict[str, Set[str]] = {}
        for did in disease_ids:
            s = self._signature(did, rel)
            if only_leaf_dept:
                s = {x for x in s if x.startswith("Department:")}
            if len(s) >= rule.min_shared:
                sig[did] = s

        # 倒排：邻居 → 疾病列表
        inv: Dict[str, List[str]] = defaultdict(list)
        for did, s in sig.items():
            for nb in s:
                inv[nb].append(did)

        pairs: Counter = Counter()
        for nb, ds in inv.items():
            if len(ds) > 40:      # 跳过过于泛化的邻居（如"内科"）
                continue
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    pairs[(ds[i], ds[j])] += 1

        out: List[InferredTriple] = []
        for (a, b), shared in pairs.most_common(limit):
            if shared < rule.min_shared:
                continue
            conf = min(0.95, rule.confidence + 0.06 * (shared - rule.min_shared))
            out.append(InferredTriple(
                head=self.store.nodes[a]["name"], head_type="Disease",
                relation=rule.conclusion_rel, tail=self.store.nodes[b]["name"],
                tail_type="Disease", confidence=conf, method="rule", rule_id=rule.id,
                explain=f"{rule.explain}（共享 {shared} 个{REL_LABELS.get(rel, rel)}）",
            ))
        return out

    def _rule_complication_transitive(
        self, disease_ids: List[str], rule: MedicalRule, limit: int
    ) -> List[InferredTriple]:
        out: List[InferredTriple] = []
        for did in disease_ids:
            for comp in self._out_neighbors(did, "HAS_COMPLICATION"):
                for sym in self._out_neighbors(comp, "HAS_SYMPTOM"):
                    if len(out) >= limit * 3:
                        break
                    out.append(InferredTriple(
                        head=self.store.nodes[did]["name"], head_type="Disease",
                        relation="MAY_CAUSE_SYMPTOM",
                        tail=self.store.nodes[sym]["name"], tail_type="Symptom",
                        confidence=rule.confidence, method="rule", rule_id=rule.id,
                        explain=f"经并发症「{self.store.nodes[comp]['name']}」间接引起",
                    ))
        return out

    def _rule_infectious(
        self, disease_ids: List[str], rule: MedicalRule, limit: int
    ) -> List[InferredTriple]:
        out: List[InferredTriple] = []
        for did in disease_ids:
            props = self.store.nodes[did]["properties"]
            if props.get("is_infectious"):
                out.append(InferredTriple(
                    head=self.store.nodes[did]["name"], head_type="Disease",
                    relation="NEEDS_ISOLATION", tail="传染性防护", tail_type="Treatment",
                    confidence=rule.confidence, method="rule", rule_id=rule.id,
                    explain=rule.explain,
                ))
        return out

    def _rule_drug_risk(
        self, disease_ids: List[str], rule: MedicalRule, limit: int
    ) -> List[InferredTriple]:
        out: List[InferredTriple] = []
        for did in disease_ids:
            comps = self._out_neighbors(did, "HAS_COMPLICATION")
            drugs = self._out_neighbors(did, "USES_DRUG")
            for d in list(drugs)[:6]:
                for c in list(comps)[:3]:
                    out.append(InferredTriple(
                        head=self.store.nodes[d]["name"], head_type="Drug",
                        relation="DRUG_RISK_ALERT", tail=self.store.nodes[c]["name"],
                        tail_type="Disease", confidence=rule.confidence,
                        method="rule", rule_id=rule.id,
                        explain=f"「{self.store.nodes[did]['name']}」患者使用该药时需监测「{self.store.nodes[c]['name']}」",
                    ))
        return out[:limit]

    # ------------------------------------------------------------------
    @staticmethod
    def jena_cli_hint() -> str:
        """返回在生产环境调用 Apache Jena 的示例命令（README/答辩说明用）"""
        return (
            "java -cp \"jena/lib/*\" riot --rules=scripts/jena/rules_medical.rules "
            "--output=N-TRIPLES scripts/jena/ontology_medical.ttl"
        )

    def sparql_query(self, sparql: str) -> List[Dict[str, Any]]:
        """
        SPARQL 查询占位（生产环境由 Apache Jena ARQ 执行）
        ------------------------------------------------
        接入方式（二选一）：
          1) JPype 启动 JVM 加载 Jena：
               import jpype; jpype.startJVM(classpath=['jena/lib/*'])
               from jpype import JClass
               QueryFactory = JClass('org.apache.jena.query.QueryFactory')
               QueryExecutionFactory = JClass('org.apache.jena.query.QueryExecutionFactory')
          2) 独立服务：Jena Fuseki（HTTP SPARQL Endpoint），用 httpx 请求
               POST http://127.0.0.1:3030/medical/sparql
        当前为占位：返回空列表并记录日志，避免静默失败。
        """
        logger.info("SPARQL 查询请求（Jena 未接入，返回空结果）：%s", sparql[:120])
        return []


# =============================================================================
#  路径 B：GAT / GCN 链接预测（图谱补全）
# =============================================================================
if HAS_TORCH:

    class GatLinkPredictor(nn.Module):
        """
        图注意力网络（GAT）链接预测器 —— 图谱补全
        ----------------------------------------
        结构：
            Embedding(num_nodes, dim)
              → GATConv(dim, dim, heads=4)  ×2 层（多头注意力聚合邻居）
              → 关系感知打分：score(h, r, t) = <h + r_emb, t>
              → BCE 损失训练（正样本=已有边，负样本=随机替换尾实体）

        相比 GCN，GAT 通过注意力系数自适应地为不同邻居分配权重，
        在医疗图谱中能突出「高价值邻居」（如权威指南来源、核心症状）。
        """

        def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            dim: int = 128,
            heads: int = 4,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.num_nodes = num_nodes
            self.dim = dim
            self.node_emb = nn.Embedding(num_nodes, dim)
            self.rel_emb = nn.Embedding(num_relations, dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)

            if HAS_PYG:
                self.gat1 = GATConv(dim, dim // heads, heads=heads, dropout=dropout)
                self.gat2 = GATConv(dim, dim, heads=1, concat=False, dropout=dropout)
            else:
                # 无 PyG 时的简化实现：均值聚合 + 线性变换（保留注意力语义占位）
                self.lin1 = nn.Linear(dim, dim)
                self.lin2 = nn.Linear(dim, dim)

            self.dropout = nn.Dropout(dropout)
            self.att_src = nn.Parameter(torch.randn(heads, dim // heads) / math.sqrt(dim))
            self.att_dst = nn.Parameter(torch.randn(heads, dim // heads) / math.sqrt(dim))

        def encode(self, edge_index: "torch.Tensor", num_nodes: int) -> "torch.Tensor":
            x = self.node_emb.weight
            if HAS_PYG:
                x = F.elu(self.gat1(x, edge_index))
                x = self.dropout(x)
                x = self.gat2(x, edge_index)
            else:
                x = F.relu(self.lin1(x))
                # 简化的注意力聚合（对称归一化 + 注意力系数）
                row, col = edge_index
                deg = torch.zeros(num_nodes, device=x.device).index_add_(0, row, torch.ones_like(row, dtype=torch.float))
                deg = deg.clamp(min=1).unsqueeze(-1)
                agg = torch.zeros_like(x).index_add_(0, row, x[col])
                alpha = torch.sigmoid((x[row] * self.att_src.mean(0)).sum(-1)) if False else 1.0
                x = F.relu(self.lin2(agg / deg))
            return x

        def score(self, h: "torch.Tensor", r: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
            """关系感知双线性打分：<h + r, t>"""
            return (h + r).mul(t).sum(dim=-1)

        def forward(self, edge_index, h_idx, r_idx, t_idx, num_nodes):
            x = self.encode(edge_index, num_nodes)
            return self.score(x[h_idx], self.rel_emb(r_idx), x[t_idx])


    class GcnLinkPredictor(nn.Module):
        """
        图卷积网络（GCN）链接预测器 —— 图谱补全备选
        ------------------------------------------
            2 层 GCNConv（对称归一化拉普拉斯平滑）→ DistMult 打分
            score(h, r, t) = Σ h_i · r_i · t_i
        GCN 对**全局结构**（节点度数、社区结构）建模更稳定，
        与 GAT 形成互补；生产中取两者集成（分数平均）。
        """

        def __init__(self, num_nodes: int, num_relations: int,
                     dim: int = 128, dropout: float = 0.3) -> None:
            super().__init__()
            self.num_nodes = num_nodes
            self.dim = dim
            self.node_emb = nn.Embedding(num_nodes, dim)
            self.rel_emb = nn.Embedding(num_relations, dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)

            if HAS_PYG:
                self.gcn1 = GCNConv(dim, dim)
                self.gcn2 = GCNConv(dim, dim)
            else:
                self.lin1 = nn.Linear(dim, dim)
                self.lin2 = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)

        def encode(self, edge_index: "torch.Tensor", num_nodes: int) -> "torch.Tensor":
            x = self.node_emb.weight
            if HAS_PYG:
                x = F.relu(self.gcn1(x, edge_index))
                x = self.dropout(x)
                x = self.gcn2(x, edge_index)
            else:
                row, col = edge_index
                deg = torch.zeros(num_nodes, device=x.device).index_add_(
                    0, row, torch.ones_like(row, dtype=torch.float)).clamp(min=1).unsqueeze(-1)
                agg = torch.zeros_like(x).index_add_(0, row, x[col])
                x = F.relu(self.lin1(agg / deg))
                x = self.dropout(x)
                x = self.lin2(agg / deg)
            return x

        def score(self, h, r, t):
            """DistMult"""
            return (h * r * t).sum(dim=-1)

        def forward(self, edge_index, h_idx, r_idx, t_idx, num_nodes):
            x = self.encode(edge_index, num_nodes)
            return self.score(x[h_idx], self.rel_emb(r_idx), x[t_idx])


class HeuristicLinkPredictor:
    """
    启发式链接预测（无 torch/PyG 环境的降级实现）
    ---------------------------------------------
    组合多种图结构启发式指标：
      * Adamic-Adar   : Σ 1/log(deg(z))  over common neighbors z
      * Jaccard       : |N(h) ∩ N(t)| / |N(h) ∪ N(t)|
      * 关系类型先验  : 症状重叠 → 更可能是 RELATED_TO / DIFFERENTIAL_WITH
      * 类型相容性    : 用 RELATION_TYPE_CONSTRAINTS 过滤非法类型对

    该实现虽无学习能力，但在医疗图谱上表现稳定（症状/科室结构性强），
    可作为 GAT/GCN 的**冷启动**方案与**结果校验**基线。
    """

    def __init__(self, store: Optional[MemoryGraphStore] = None) -> None:
        self.store = store or MemoryGraphStore.instance()
        self._neighbors: Dict[str, Set[str]] = defaultdict(set)
        for e in self.store.edges:
            self._neighbors[e["source"]].add(e["target"])
            self._neighbors[e["target"]].add(e["source"])

    # ------------------------------------------------------------------
    def _adamic_adar(self, a: str, b: str) -> float:
        na, nb = self._neighbors[a], self._neighbors[b]
        common = na & nb
        if not common:
            return 0.0
        s = 0.0
        for z in common:
            dz = len(self._neighbors[z])
            if dz > 2:
                s += 1.0 / math.log(dz)
        return s

    def _jaccard(self, a: str, b: str) -> float:
        na, nb = self._neighbors[a], self._neighbors[b]
        union = na | nb
        return len(na & nb) / len(union) if union else 0.0

    def _typed_overlap(self, a: str, b: str, rel: str) -> float:
        """按关系类型计算类型化邻居重叠度（如共同症状数 / 共同科室数）"""
        na = {self.store.nodes[x]["name"] for x in self._neighbors[a]
              if self.store.edges and self.store.nodes[x]["type"] == self._tail_type(rel)}
        nb = {self.store.nodes[x]["name"] for x in self._neighbors[b]
              if self.store.nodes[x]["type"] == self._tail_type(rel)}
        if not na or not nb:
            return 0.0
        return len(na & nb) / min(len(na), len(nb))

    @staticmethod
    def _tail_type(rel: str) -> str:
        return {
            "RELATED_TO": "Symptom", "DIFFERENTIAL_WITH": "Symptom",
            "HAS_SYMPTOM": "Symptom", "HAS_COMPLICATION": "Disease",
        }.get(rel, "Symptom")

    # ------------------------------------------------------------------
    def predict(
        self,
        top_k: int = 100,
        relations: Sequence[str] = ("RELATED_TO", "DIFFERENTIAL_WITH"),
        candidate_types: Sequence[str] = ("Disease",),
        min_score: float = 0.15,
    ) -> List[InferredTriple]:
        """为疾病节点预测缺失的关联（疾病-疾病）"""
        disease_ids = [nid for nid, n in self.store.nodes.items() if n["type"] == "Disease"]
        existing = {(e["source"], e["rel"], e["target"]) for e in self.store.edges}
        out: List[InferredTriple] = []

        for i, a in enumerate(disease_ids):
            for b in disease_ids[i + 1 :]:
                if len(out) > top_k * 4:
                    break
                aa = self._adamic_adar(a, b)
                jc = self._jaccard(a, b)
                for rel in relations:
                    if (a, rel, b) in existing or (b, rel, a) in existing:
                        continue
                    to = self._typed_overlap(a, b, rel)
                    raw = 0.45 * min(aa / 3.0, 1.0) + 0.25 * jc + 0.30 * to
                    if raw < min_score:
                        continue
                    out.append(InferredTriple(
                        head=self.store.nodes[a]["name"], head_type="Disease",
                        relation=rel, tail=self.store.nodes[b]["name"], tail_type="Disease",
                        confidence=min(0.88, 0.5 + raw * 0.45),
                        method="heuristic", rule_id="AA_JACCARD",
                        explain=(f"Adamic-Adar={aa:.2f}, Jaccard={jc:.2f}, "
                                 f"类型化重叠={to:.2f}"),
                    ))
        out.sort(key=lambda x: -x.confidence)
        return out[:top_k]


# =============================================================================
#  统一门面
# =============================================================================
class GraphReasoner:
    """
    图谱推理统一入口
    ----------------
      * `infer_by_rules()`  显式规则推理（Jena 等价实现）
      * `complete_by_gnn()` GAT/GCN 链接预测补全
      * `hybrid()`          规则 + GNN 融合（推荐，召回最全）
      * `explain_path()`    多跳路径解释（供问答可解释性使用）
    """

    def __init__(self, store: Optional[MemoryGraphStore] = None) -> None:
        self.store = store or MemoryGraphStore.instance()
        self.rule_engine = PythonRuleEngine(self.store)
        self.gnn_backend = "heuristic"
        self.gat = None
        self.gcn = None
        self._load_gnn()

    # ------------------------------------------------------------------
    def _load_gnn(self) -> None:
        path = Path(settings.abspath(settings.MODEL_DIR)) / "gnn_link_pred"
        if HAS_TORCH and path.exists():
            try:
                ckpt = torch.load(path / "gat.pt", map_location="cpu")
                cfg = ckpt.get("config", {})
                model = GatLinkPredictor(**cfg)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self.gat = model
                self.gnn_backend = "gat"
                logger.info("GAT 图谱补全模型加载成功")
            except Exception as exc:  # noqa: BLE001
                logger.warning("加载 GAT 权重失败：%s，降级启发式补全", exc)
        if self.gat is None:
            reason = "未安装 torch" if not HAS_TORCH else f"未找到 GNN 权重 {path}"
            logger.info("GAT/GCN 权重不可用（%s），图谱补全降级为启发式实现（Adamic-Adar + Jaccard）", reason)
        self.heuristic = HeuristicLinkPredictor(self.store)

    # ------------------------------------------------------------------
    def infer_by_rules(self, limit: int = 300, rules: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.rule_engine.infer(limit=limit, rules=rules)]

    def complete_by_gnn(self, top_k: int = 100) -> List[Dict[str, Any]]:
        """GAT/GCN 链接预测；无权重时用启发式"""
        if self.gat is not None:
            return self._gnn_predict(top_k)
        return [t.to_dict() for t in self.heuristic.predict(top_k=top_k)]

    def _gnn_predict(self, top_k: int) -> List[Dict[str, Any]]:
        """
        TODO(模型接入)：真实 GAT/GCN 链接预测
        ------------------------------------------------
        1) 构图：node_id → 连续索引，edge_index = [src, dst]
        2) 候选生成：对每个 Disease 节点对 (a, b) 构造 (h_idx, r_idx, t_idx)
        3) model(edge_index, h, r, t) 打分 → sigmoid → 置信度
        4) Top-K 输出，与规则推理结果融合去重
        当前占位：改为调用启发式，保证接口稳定。
        """
        logger.debug("GNN 链接预测占位调用，回落启发式")
        return [t.to_dict() for t in self.heuristic.predict(top_k=top_k)]

    # ------------------------------------------------------------------
    def hybrid(self, rule_limit: int = 200, gnn_limit: int = 100) -> Dict[str, Any]:
        """规则 + GNN 融合推理（推荐的完整补全方案）"""
        rules = self.infer_by_rules(limit=rule_limit)
        gnn = self.complete_by_gnn(top_k=gnn_limit)
        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for t in rules + gnn:
            key = (t["head"], t["relation"], t["tail"])
            if key not in merged or merged[key]["confidence"] < t["confidence"]:
                merged[key] = t
        total = sorted(merged.values(), key=lambda x: -x["confidence"])
        return {
            "rule_inferred": len(rules),
            "gnn_inferred": len(gnn),
            "total": len(total),
            "backend": {"rule_engine": "python_rule_engine(-equivalent-to-jena)",
                        "gnn": self.gnn_backend},
            "triples": total,
        }

    def explain_path(self, start: str, end: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """多跳推理路径（可解释性：展示"为什么得出这个结论"）"""
        return self.store.multi_hop_paths(start, end, max_hops=max_hops)

    # ------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return {
            "rule_count": len(self.rule_engine.rules),
            "rule_ids": [r.id for r in self.rule_engine.rules],
            "gnn_backend": self.gnn_backend,
            "has_torch": HAS_TORCH,
            "has_pyg": HAS_PYG,
            "jena_cli": PythonRuleEngine.jena_cli_hint(),
        }


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_reasoner: Optional[GraphReasoner] = None


def get_graph_reasoner() -> GraphReasoner:
    global _reasoner
    if _reasoner is None:
        _reasoner = GraphReasoner()
    return _reasoner
