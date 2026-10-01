# -*- coding: utf-8 -*-
"""
智能问答服务（可解释 AI 医生）
==============================
职责：
  * 编排 RAG 主流程（意图 → 实体链接 → KG 检索 → Prompt → LLM → 幻觉守卫）
  * 维护多轮会话历史
  * 提供推理链路与溯源证据（可解释性核心）
  * 当 `rag_engine` 不可用（例如依赖缺失）时，退化为**纯图谱结构化回答**，
    保证问答接口永不 500。
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings
from app.kg_layer.graph_service import GraphService
from app.llm_layer.entity_linker import EntityLinker
from app.llm_layer.intent_classifier import IntentClassifier
from app.schemas import (
    ChatMessage,
    EvidenceItem,
    IntentResult,
    LatencyBreakdown,
    LinkedEntity,
    QAResponse,
    ReasoningStep,
    SourceItem,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _now_ts() -> str:
    return datetime.now().strftime("%H:%M")


class QAService:
    """问答业务服务"""

    MAX_HISTORY_PER_SESSION = 40

    def __init__(
        self,
        graph: GraphService,
        intent: IntentClassifier,
        linker: EntityLinker,
        rag=None,
    ) -> None:
        self.graph = graph
        self.intent = intent
        self.linker = linker
        self._rag = rag
        self._history: Dict[str, List[ChatMessage]] = {}

    # ------------------------------------------------------------------
    @property
    def rag(self):
        """延迟获取 RAG 引擎（避免启动期循环依赖与不必要的模型加载）"""
        if self._rag is None:
            try:
                from app.llm_layer.rag.rag_engine import get_rag_engine

                self._rag = get_rag_engine()
            except Exception as exc:  # noqa: BLE001
                logger.error("RAG 引擎不可用：%s（将使用纯图谱结构化回答）", exc)
        return self._rag

    # ==================================================================
    #  主入口
    # ==================================================================
    async def ask(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 12,
        max_hops: int = 2,
        explain: bool = True,
        use_llm: bool = True,
    ) -> QAResponse:
        history = self._history.get(session_id, [])
        rag = self.rag

        if rag is not None:
            try:
                resp = await rag.answer(
                    question=question,
                    session_id=session_id,
                    top_k=top_k,
                    max_hops=max_hops,
                    explain=explain,
                    use_llm=use_llm,
                    history=[m.model_dump() for m in history[-6:]],
                    slots=self._slots_of(session_id),
                )
                self._remember(session_id, question, resp)
                return resp
            except Exception as exc:  # noqa: BLE001
                logger.error("RAG 主流程异常，降级为纯图谱回答：%s", exc, exc_info=True)

        resp = await self._graph_only_answer(question, session_id, top_k, max_hops)
        self._remember(session_id, question, resp)
        return resp

    # ==================================================================
    #  降级：纯知识图谱结构化回答（无 RAG 引擎时）
    # ==================================================================
    async def _graph_only_answer(
        self, question: str, session_id: str, top_k: int, max_hops: int
    ) -> QAResponse:
        t0 = time.perf_counter()
        trace: List[ReasoningStep] = []
        latency = LatencyBreakdown()

        # 1) 意图
        s = time.perf_counter()
        it = self.intent.classify(question)
        latency.intent = round((time.perf_counter() - s) * 1000, 2)
        trace.append(ReasoningStep(
            step=1, action="intent_classify", title="意图识别",
            detail=f"{it.method} → {it.label}（{it.label_cn}），置信度 {it.confidence:.2f}",
            elapsed_ms=latency.intent, payload={"scores": it.scores},
        ))

        # 2) 实体链接
        s = time.perf_counter()
        analysis = self.linker.analyze(question)
        latency.entity_link = round((time.perf_counter() - s) * 1000, 2)
        mentions = analysis.get("mentions") or []
        trace.append(ReasoningStep(
            step=2, action="entity_link", title="实体识别与实体链接",
            detail="；".join(
                f"「{m['text']}」→ {m['type']}/{m['kg_name']}（{m['method']}, {m['score']:.2f}）"
                for m in mentions[:6]
            ) or "未识别到知识图谱实体",
            elapsed_ms=latency.entity_link,
            payload={"entities": mentions[:10]},
        ))

        # 3) 图谱检索
        s = time.perf_counter()
        main = analysis.get("main_entity") or ""
        evidences: List[EvidenceItem] = []
        triples: List[Dict[str, Any]] = []
        if main:
            triples = self.graph.triples_of(main, limit=top_k, direction="out")
        for i, t in enumerate(triples, 1):
            ev = self.graph.get_evidence(t["head"], t["relation"], t["tail"], f"KG-{i}")
            if ev:
                evidences.append(ev)
        latency.kg_query = round((time.perf_counter() - s) * 1000, 2)
        latency.semantic_parse_total = round(
            latency.intent + latency.entity_link + latency.kg_query, 2
        )
        if latency.semantic_parse_total > 500:
            logger.warning("语义解析耗时 %.1fms 超过 500ms 指标", latency.semantic_parse_total)
        trace.append(ReasoningStep(
            step=3, action="kg_retrieve", title="知识图谱检索",
            detail=f"实体「{main or '未命中'}」召回 {len(triples)} 条三元组",
            elapsed_ms=latency.kg_query,
            payload={"triple_count": len(triples)},
        ))

        # 4) 结构化答案拼装
        s = time.perf_counter()
        answer = self._render_graph_answer(main, triples)
        latency.retrieve = round((time.perf_counter() - s) * 1000, 2)
        trace.append(ReasoningStep(
            step=4, action="template_generate", title="知识图谱结构化回答生成",
            detail="RAG 引擎不可用，直接以图谱三元组拼装答案（无幻觉风险）",
            elapsed_ms=latency.retrieve,
        ))

        latency.total = round((time.perf_counter() - t0) * 1000, 2)
        confidence = 0.75 if triples else 0.25

        return QAResponse(
            question=question,
            answer=answer,
            answer_html=self._to_html(answer),
            intent=IntentResult(
                label=it.label, label_cn=it.label_cn,
                confidence=it.confidence, method=it.method, scores=it.scores,
            ),
            entities=[LinkedEntity(**{k: v for k, v in m.items()
                                      if k in LinkedEntity.model_fields}) for m in mentions],
            evidences=evidences,
            kg_context=self._render_kg_context(triples),
            reasoning_trace=trace,
            confidence=confidence,
            prompt_template_id="graph_only_fallback",
            prompt_used="",
            guard_result={"passed": True, "score": 1.0,
                          "summary": "纯图谱输出，未经过大模型，无幻觉风险",
                          "suggested_action": "accept", "violations": [],
                          "stats": {}},
            latency_ms=latency,
            llm_model="graph-only",
            session_id=session_id,
            disclaimer=settings.disclaimer,
            related_questions=self.graph.related_questions(main, 5),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _render_graph_answer(entity: str, triples: List[Dict[str, Any]]) -> str:
        if not triples:
            return (
                "## 知识库未收录\n\n"
                "很抱歉，当前知识库暂未收录与您问题直接相关的内容。为避免误导，我不做推测性回答。\n\n"
                "建议您换用更常见的疾病或症状名称重新提问，或前往正规医疗机构就诊咨询执业医师。\n\n"
                f"{settings.disclaimer}"
            )
        # 按关系分组
        groups: Dict[str, List[tuple]] = {}
        for i, t in enumerate(triples, 1):
            groups.setdefault(t["relation_label"], []).append((i, t["tail"]))
        lines = [f"## {entity} 相关知识（来自知识图谱）", ""]
        for label, items in groups.items():
            lines.append(f"### {label}")
            for idx, tail in items:
                lines.append(f"- {tail} [KG-{idx}]")
            lines.append("")
        lines.append("以上内容直接来自医疗知识图谱的结构化事实，每条均可溯源。")
        lines.append("")
        lines.append(settings.disclaimer)
        return "\n".join(lines)

    @staticmethod
    def _render_kg_context(triples: List[Dict[str, Any]]) -> str:
        if not triples:
            return "（无）"
        lines = [f"【知识图谱事实 —— 共 {len(triples)} 条】", ""]
        for i, t in enumerate(triples, 1):
            lines.append(
                f"[KG-{i}] {t['head']} —{t['relation_label']}→ {t['tail']}"
                f"（置信度 {t.get('confidence', 0.95)}）"
            )
        return "\n".join(lines)

    @staticmethod
    def _to_html(md: str) -> str:
        """极简 markdown → HTML（用于前端 v-html 渲染 + 引用角标）"""
        import html as _html
        import re

        out = _html.escape(md or "")
        out = re.sub(r"^### (.+)$", r"<h5>\1</h5>", out, flags=re.M)
        out = re.sub(r"^## (.+)$", r"<h4>\1</h4>", out, flags=re.M)
        out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
        out = out.replace("\n", "<br/>")
        out = re.sub(
            r"\[KG-(\d+)\]",
            r'<sup class="kg-cite" data-cite="KG-\1">[KG-\1]</sup>',
            out,
        )
        return out

    # ==================================================================
    #  会话历史
    # ==================================================================
    def _remember(self, session_id: str, question: str, resp: QAResponse) -> None:
        msgs = self._history.setdefault(session_id, [])
        msgs.append(ChatMessage(role="user", content=question, timestamp=_now_ts()))
        msgs.append(ChatMessage(
            role="assistant", content=resp.answer, timestamp=_now_ts(),
            answer_html=resp.answer_html, evidences=resp.evidences,
            reasoning_trace=resp.reasoning_trace if len(resp.reasoning_trace) < 20 else [],
            confidence=resp.confidence,
        ))
        if len(msgs) > self.MAX_HISTORY_PER_SESSION:
            del msgs[: len(msgs) - self.MAX_HISTORY_PER_SESSION]

    def history(self, session_id: str) -> List[ChatMessage]:
        return list(self._history.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    def _slots_of(self, session_id: str) -> Dict[str, Any]:
        """
        从会话历史中抽取槽位（多轮上下文继承）。
        简化策略：收集历史中识别到的疾病/症状名，作为已确认槽位。
        """
        msgs = self._history.get(session_id, [])
        if not msgs:
            return {}
        diseases, symptoms = [], []
        for m in msgs[-6:]:
            if m.role != "user":
                continue
            try:
                a = self.linker.analyze(m.content)
            except Exception:  # noqa: BLE001
                continue
            for d in a.get("diseases") or []:
                if d not in diseases:
                    diseases.append(d)
            for s in a.get("symptoms") or []:
                if s not in symptoms:
                    symptoms.append(s)
        slots: Dict[str, Any] = {}
        if diseases:
            slots["已讨论疾病"] = "、".join(diseases[:3])
        if symptoms:
            slots["已提及症状"] = "、".join(symptoms[:5])
        return slots

    def session_count(self) -> int:
        return len(self._history)

    def info(self) -> Dict[str, Any]:
        return {
            "rag_available": self.rag is not None,
            "sessions": len(self._history),
            "max_history_per_session": self.MAX_HISTORY_PER_SESSION,
        }
