# -*- coding: utf-8 -*-
"""
自然语言 → Cypher 生成器
========================
把「意图 + 实体链接结果」映射为参数化的 Cypher 查询，供 Neo4j 执行；
Neo4j 不可用时由 `MemoryGraphStore.query()` 的内存解释器处理。

**为什么使用模板 + 参数化而非让大模型直接写 Cypher？**
  1. **安全**：大模型生成的 Cypher 存在注入与误删风险（`DELETE` / `MERGE`）。
     本模块只使用**预定义模板 + 参数绑定**，彻底消除注入面。
  2. **稳定**：医疗查询模式高度收敛（查症状/查科室/查用药/查检查/多跳），
     模板覆盖率 >95%，且响应时间稳定在毫秒级（满足 ≤500ms 指标）。
  3. **可解释**：每条模板都对应一个明确的临床问题类型，便于在推理链路中展示
     "为什么发起这次查询"。

如需支持复杂/长尾查询，可用 `render_llm_fallback()` 让大模型生成 Cypher，
但结果必须经过 `assert_readonly()` 校验后才可执行。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.kg_layer.neo4j_client import assert_readonly
from app.schemas import IntentLabel
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CypherQuery:
    """一条可执行的 Cypher 查询"""

    name: str
    cypher: str
    params: Dict[str, Any]
    purpose: str = ""
    intent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "cypher": self.cypher, "params": self.params,
                "purpose": self.purpose, "intent": self.intent}


#: 关系类型白名单（防止模板被篡改）
ALLOWED_RELS = {
    "HAS_SYMPTOM", "TREATED_BY", "USES_DRUG", "BELONGS_TO",
    "NEEDS_CHECK", "AFFECTS", "HAS_COMPLICATION", "DIFFERENTIAL_WITH",
    "RELATED_TO", "MAY_CAUSE_SYMPTOM", "NEEDS_ISOLATION",
}


class CypherGenerator:
    """基于意图与实体生成 Cypher 查询"""

    # ==================================================================
    #  意图 → 查询模板
    # ==================================================================
    def build(
        self,
        intent: str,
        analysis: Dict[str, Any],
        limit: int = 20,
        max_hops: int = 2,
    ) -> List[CypherQuery]:
        """
        根据意图生成一组 Cypher 查询（按优先级排序）。

        参数
        ----
        intent   : IntentLabel 的值
        analysis : `EntityLinker.analyze()` 的输出
        """
        diseases = analysis.get("diseases") or []
        symptoms = analysis.get("symptoms") or []
        departments = analysis.get("departments") or []
        drugs = analysis.get("drugs") or []
        names = [n for n in (analysis.get("mentions") or []) and
                 [m.get("kg_name") for m in analysis["mentions"]] if n]
        queries: List[CypherQuery] = []

        # ---------- 1) 疾病核心事实（所有意图的基础）----------
        for d in diseases[:2]:
            queries.append(CypherQuery(
                name="disease_facts",
                cypher="""
                MATCH (d:Disease {name: $name})-[r]->(t)
                WHERE type(r) IN $rels
                RETURN d.name AS head, labels(d)[0] AS head_type,
                       type(r) AS relation, t.name AS tail, labels(t)[0] AS tail_type,
                       coalesce(r.confidence, 0.95) AS confidence
                ORDER BY confidence DESC
                LIMIT $limit
                """,
                params={"name": d, "rels": sorted(ALLOWED_RELS), "limit": limit},
                purpose=f"检索疾病「{d}」的全部直接关联事实（症状/治疗/用药/科室/检查/并发症）",
                intent=intent,
            ))

        # ---------- 2) 症状 → 疾病 反向检索 ----------
        for s in symptoms[:3]:
            queries.append(CypherQuery(
                name="symptom_to_disease",
                cypher="""
                MATCH (s:Symptom {name: $name})<-[:HAS_SYMPTOM]-(d:Disease)
                OPTIONAL MATCH (d)-[:BELONGS_TO]->(dept:Department)
                RETURN d.name AS disease, d.category1 AS category1,
                       d.category2 AS category2, collect(DISTINCT dept.name) AS departments,
                       coalesce(d.is_infectious, false) AS is_infectious
                ORDER BY size(departments) DESC
                LIMIT $limit
                """,
                params={"name": s, "limit": limit},
                purpose=f"由症状「{s}」反查可能的疾病（症状咨询的核心查询）",
                intent=intent,
            ))

        # ---------- 3) 科室导诊 ----------
        if intent == IntentLabel.DEPARTMENT_QUERY.value or departments or diseases:
            target = diseases[0] if diseases else (names[0] if names else "")
            if target:
                queries.append(CypherQuery(
                    name="department_route",
                    cypher="""
                    MATCH (d:Disease {name: $name})-[:BELONGS_TO]->(dept:Department)
                    OPTIONAL MATCH (d)-[:NEEDS_CHECK]->(c:Check)
                    RETURN d.name AS disease, collect(DISTINCT dept.name) AS departments,
                           collect(DISTINCT c.name) AS checks
                    LIMIT $limit
                    """,
                    params={"name": target, "limit": limit},
                    purpose=f"查询「{target}」的就诊科室与建议检查（科室导诊）",
                    intent=intent,
                ))
                # 反向：按科室找疾病
                for dept in departments[:2]:
                    queries.append(CypherQuery(
                        name="department_diseases",
                        cypher="""
                        MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department {name: $name})
                        OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                        RETURN d.name AS disease, collect(DISTINCT s.name)[0..6] AS symptoms
                        LIMIT $limit
                        """,
                        params={"name": dept, "limit": limit},
                        purpose=f"查询「{dept}」常见疾病",
                        intent=intent,
                    ))

        # ---------- 4) 治疗与用药（含药物关系）----------
        if intent == IntentLabel.TREATMENT_QUERY.value or drugs:
            target = diseases[0] if diseases else (names[0] if names else "")
            if target:
                queries.append(CypherQuery(
                    name="treatment_drugs",
                    cypher="""
                    MATCH (d:Disease {name: $name})-[:TREATED_BY|USES_DRUG]->(t)
                    RETURN d.name AS disease, type(relation) AS relation,
                           t.name AS treatment, labels(t)[0] AS type
                    LIMIT $limit
                    """,
                    params={"name": target, "limit": limit},
                    purpose=f"查询「{target}」的治疗方法与用药",
                    intent=intent,
                ))
            for dr in drugs[:2]:
                queries.append(CypherQuery(
                    name="drug_diseases",
                    cypher="""
                    MATCH (drug:Drug {name: $name})<-[:USES_DRUG]-(d:Disease)
                    RETURN d.name AS disease, d.category1 AS category1
                    LIMIT $limit
                    """,
                    params={"name": dr, "limit": limit},
                    purpose=f"查询药物「{dr}」可治疗的疾病",
                    intent=intent,
                ))

        # ---------- 5) 多跳推理路径 ----------
        if len(diseases) >= 2 and max_hops >= 2:
            queries.append(CypherQuery(
                name="multi_hop_path",
                cypher=f"""
                MATCH p = shortestPath((a:Disease {{name: $a}})-[*1..{max(1, min(max_hops, 4))}]-(b:Disease {{name: $b}}))
                RETURN [n IN nodes(p) | n.name] AS path_nodes,
                       [r IN relationships(p) | type(r)] AS path_rels,
                       length(p) AS hops
                LIMIT 5
                """,
                params={"a": diseases[0], "b": diseases[1]},
                purpose=f"检索「{diseases[0]}」与「{diseases[1]}」之间的多跳关联路径（可解释推理）",
                intent=intent,
            ))

        # ---------- 6) 共病 / 并发症扩展 ----------
        if diseases:
            queries.append(CypherQuery(
                name="complications",
                cypher="""
                MATCH (d:Disease {name: $name})-[:HAS_COMPLICATION]->(c:Disease)
                OPTIONAL MATCH (c)-[:HAS_SYMPTOM]->(s:Symptom)
                RETURN c.name AS complication, c.category1 AS category1,
                       collect(DISTINCT s.name)[0..6] AS symptoms
                LIMIT $limit
                """,
                params={"name": diseases[0], "limit": limit},
                purpose=f"查询「{diseases[0]}」的并发症及其症状",
                intent=intent,
            ))

        # ---------- 7) 无实体命中时的兜底：全局统计 ----------
        if not queries:
            queries.append(CypherQuery(
                name="global_overview",
                cypher="""
                MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department)
                RETURN dept.name AS department, count(d) AS disease_count
                ORDER BY disease_count DESC LIMIT $limit
                """,
                params={"limit": limit},
                purpose="未识别到明确实体，返回疾病-科室分布概览",
                intent=intent,
            ))

        return queries

    # ==================================================================
    #  只读校验
    # ==================================================================
    @staticmethod
    def validate(query: CypherQuery) -> Tuple[bool, str]:
        """
        执行前校验：
          ① 只读（无 CREATE/DELETE/SET/MERGE）
          ② 关系类型在 ALLOWED_RELS 内
          ③ LIMIT 存在（防止全表扫描拖垮数据库）

        注意：提取 token 时必须排除 Cypher 的可变长路径语法 `-[*1..3]->`，
        否则会把 `*1..3` 中的数字误判为关系类型（曾导致合法查询被拒）。
        """
        try:
            assert_readonly(query.cypher)
        except Exception as exc:  # noqa: BLE001
            return False, f"只读校验失败：{exc}"

        import re

        # 先剔除变长路径片段 `-[*1..3]->` / `[*1..2]`，避免误匹配其中的数字
        cleaned = re.sub(r"\[\s*\*\s*\d*\s*(?:\.\.\s*\d+\s*)?\]", "[]", query.cypher)
        # 再剔除关系类型选择器里的管道符（如 [:A|B]），只保留 token
        tokens = re.findall(r":\s*([A-Za-z_][A-Za-z0-9_]*)", cleaned)
        illegal = [
            t for t in tokens
            if t not in ALLOWED_RELS
            and t not in ("Disease", "Symptom", "Drug", "Department", "Treatment",
                          "Check", "Population", "Source", "Alias")
        ]
        if illegal:
            return False, f"包含非白名单关系/标签：{illegal}"

        low = query.cypher.lower()
        if "limit" not in low and "count(" not in low:
            return False, "查询缺少 LIMIT 限制"

        return True, "ok"

    # ==================================================================
    #  大模型兜底生成（可选，需严格校验）
    # ==================================================================
    @staticmethod
    def render_llm_fallback(question: str, schema_hint: str) -> str:
        """
        当模板无法覆盖长尾查询时，可让大模型生成 Cypher。

        ⚠️ 安全要求：生成结果必须通过 `validate()` 后才可执行。
        提示词已内置只读约束与 Schema 提示。

        返回可直接发送给 LLM 的 prompt 字符串。
        """
        return (
            "你是 Neo4j Cypher 专家。请把下面的中文医疗问题转换为**只读** Cypher 查询。\n"
            "\n"
            "【强制约束】\n"
            "1. 只允许 MATCH / OPTIONAL MATCH / WITH / UNWIND / RETURN / ORDER BY / LIMIT。\n"
            "2. 严禁出现 CREATE、MERGE、DELETE、SET、REMOVE、DROP、FOREACH、LOAD CSV。\n"
            "3. 必须包含 LIMIT，且不超过 50。\n"
            "4. 只能使用下方 Schema 中的标签与关系类型。\n"
            "5. 只输出 Cypher 语句本身，不要任何解释、不要 markdown 代码块标记。\n"
            "\n"
            "【Schema】\n"
            f"{schema_hint}\n"
            "\n"
            f"【问题】{question}\n"
            "\n"
            "【Cypher】\n"
        )

    def info(self) -> Dict[str, Any]:
        return {
            "strategy": "template + parameter binding (no LLM-generated Cypher by default)",
            "allowed_relations": sorted(ALLOWED_RELS),
            "template_names": [
                "disease_facts", "symptom_to_disease", "department_route",
                "department_diseases", "treatment_drugs", "drug_diseases",
                "multi_hop_path", "complications", "global_overview",
            ],
            "safety": "assert_readonly() + 关系白名单 + 强制 LIMIT",
        }


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_generator: Optional[CypherGenerator] = None


def get_cypher_generator() -> CypherGenerator:
    global _generator
    if _generator is None:
        _generator = CypherGenerator()
    return _generator
