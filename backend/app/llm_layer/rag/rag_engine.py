# -*- coding: utf-8 -*-
"""
RAG 引擎（RagEngine）—— "智愈医典" 可解释医疗问答的唯一编排入口
================================================================

★ 全链路设计（每一步都写入 LatencyBreakdown 与 ReasoningStep，供前端时间轴展示）★

  ① 意图识别        TextCNN + 规则融合 → 4 类意图（疾病查询 / 症状咨询 / 治疗咨询 / 科室导诊）
  ② 实体识别与链接  BERT-NER + 词典 + 模糊匹配 → 问句实体 → 图谱节点（LinkedEntity）
  ③ 知识图谱检索    五路混合召回（直接事实 / 反向检索 / 多跳 / 推理补全 / 相似扩展）→ KG-1…KG-n
  ④ 上下文构建      三元组按槽位组织成带编号、带来源的 <KG_CONTEXT> 文本块
  ⑤ Prompt 渲染     从 100+ 医疗专属模板中选模板 + 注入变量 + 追加全局硬约束与免责声明
  ⑥ 大模型生成      llama3:8b-instruct（本地 Ollama）→ 失败 / 不可用 → 本地 KG 模板回答
  ⑦ 幻觉守卫        7 项确定性校验 → accept / rewrite / fallback_template（纯 KG 拼装）
  ⑧ 置信度与收尾    意图置信度 + 事实置信度 + 守卫评分 → 综合置信度、富文本、推荐追问

★ 为什么用 KG 三元组而不是文本 chunk 作为检索单元 ★
  医疗问答的验收标准不是"读起来像"，而是"每句话都能溯源、都不能越权"。
  三元组是原子事实，天然可以编号 `[KG-n]`、可以做事实级校验（幻觉守卫）、可以按
  关系槽位组织（prompt 与之一一对应）。文本 chunk 无法满足这三点。

★ 降级保障（PPT 验收要求：无网络、无 GPU、无 Neo4j 也能完整演示）★
  * `llm_client` 由他人并行开发 → 惰性导入 + 全 try/except，任何失败都回落
    `_template_answer`（纯本地、确定性、只用 KG 事实，输出 markdown + [KG-n] + 免责声明）
  * `entity_linker` 同样惰性导入，失败则退化为空实体（仍可依靠症状反向检索与推理补全）
  * 只有知识图谱 + prompt 模板库也可产出完整 QAResponse
"""
from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import settings
from app.kg_layer.graph_service import get_graph_service
from app.llm_layer.rag.context_builder import ContextBuilder, ContextBundle
from app.llm_layer.rag.hallucination_guard import GuardResult, HallucinationGuard
from app.llm_layer.rag.prompt_templates import PromptLibrary, PromptTemplate, get_prompt_library
from app.llm_layer.rag.retriever import HybridRetriever, RetrievedFact
from app.schemas import (
    ChatMessage,
    EvidenceItem,
    IntentLabel,
    IntentResult,
    LatencyBreakdown,
    LinkedEntity,
    QAResponse,
    ReasoningStep,
)
from app.utils.logger import get_logger
from app.utils.timer import Stopwatch

logger = get_logger(__name__)

#: 意图中文名兜底（优先取 intent_classifier.INTENT_CN）
_INTENT_CN: Dict[str, str] = {
    "disease_query": "疾病查询",
    "symptom_consult": "症状咨询",
    "treatment_query": "治疗咨询",
    "department_query": "科室导诊",
}

#: 意图方法 → 中文展示名
_METHOD_CN: Dict[str, str] = {
    "textcnn": "TextCNN",
    "textcnn+rule_fusion": "TextCNN+规则融合",
    "rule_fallback": "医疗触发词规则（模型降级）",
    "fallback": "兜底策略",
}

#: 实体类型 → 实体字典键
_TYPE_TO_KEY: Dict[str, str] = {
    "Disease": "diseases", "Symptom": "symptoms", "Department": "departments",
    "Drug": "drugs", "Check": "checks", "Treatment": "treatments",
}

#: 本地模板回答的关系槽位顺序
_GROUP_ORDER: Tuple[str, ...] = (
    "症状", "治疗", "用药", "检查", "科室", "并发症", "鉴别诊断", "易感人群",
    "相关（多跳推理）", "相关",
)

#: 关系槽位 → 句子模板（本地 KG 兜底答案用）
_GROUP_SENTENCE: Dict[str, str] = {
    "症状": "知识库记录的相关症状包括：{items}。",
    "治疗": "知识库记录的治疗方式包括：{items}。",
    "用药": "知识库记录的相关药物类别或药物名称包括：{items}（具体用药须由执业医师根据个体情况决定）。",
    "检查": "建议的检查项目包括：{items}。",
    "科室": "建议就诊科室：{items}。",
    "并发症": "可能出现的并发症包括：{items}。",
    "鉴别诊断": "需要鉴别的疾病包括：{items}。",
    "易感人群": "易感人群：{items}。",
    "相关（多跳推理）": "图谱多跳推理得到的关联信息：{items}。",
    "相关": "图谱记录的相关信息：{items}。",
}

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CITE_RE = re.compile(r"\[(KG-\d+)\]")
_DEFAULT_ENTITY_KEYS: Tuple[str, ...] = (
    "diseases", "symptoms", "departments", "drugs", "checks", "treatments",
)

#: 超范围问题（与医疗健康无关）的标准拒答话术
#: 对应模板 `safety_out_of_scope_v1`，此处以内联常量形式用于本地降级路径，
#: 保证在无大模型时也能给出礼貌、有边界的拒答。
_OUT_OF_SCOPE_ANSWER: str = (
    "## 服务范围说明\n\n"
    "我是「智愈医典」的 AI 医生助手，专注于**医疗健康知识**解答，包括：\n"
    "疾病症状、治疗方案、检查项目、就诊科室、用药常识与健康风险评估。\n\n"
    "您的问题超出了我的服务范围，我无法作答 —— 这也是为了避免在非专业领域给出不可靠的信息。\n\n"
    "如果您有医疗健康方面的疑问，欢迎随时向我提问，例如：\n"
    "- 高血压有哪些典型症状？\n"
    "- 肺栓塞应该挂什么科？\n"
    "- 2 型糖尿病常用哪些药物？（不提供剂量）\n\n"
    "⚠️ 如出现剧烈胸痛、呼吸困难、意识障碍等急症表现，请立即拨打 120 或前往急诊。\n\n"
    + settings.disclaimer
)


class RagEngine:
    """医疗 RAG 问答引擎（含会话记忆、可解释链路、幻觉守卫）"""

    #: 每个会话最多保留的消息条数
    MAX_SESSION_MESSAGES: int = 40

    def __init__(
        self,
        graph_service: Any = None,
        retriever: Optional[HybridRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        guard: Optional[HallucinationGuard] = None,
        prompt_library: Optional[PromptLibrary] = None,
    ) -> None:
        self.graph_service = graph_service or get_graph_service()
        self.retriever = retriever or HybridRetriever(self.graph_service)
        self.context_builder = context_builder or ContextBuilder(self.graph_service)
        self.guard = guard or HallucinationGuard()
        self.prompt_library = prompt_library or get_prompt_library()
        self._sessions: Dict[str, List[ChatMessage]] = {}
        logger.info(
            "RAG 引擎初始化完成：图谱数据源=%s，prompt 模板=%d 个，幻觉守卫=%s",
            getattr(self.graph_service, "data_source", "unknown"),
            len(getattr(self.prompt_library, "templates", []) or []),
            "启用" if settings.RAG_HALLUCINATION_GUARD else "关闭",
        )

    # ==================================================================
    #  ① 意图识别
    # ==================================================================
    def _classify(self, question: str) -> Tuple[str, str, float, str, Dict[str, float]]:
        try:
            from app.llm_layer.intent_classifier import get_intent_classifier  # 惰性导入

            res = get_intent_classifier().classify(question)
            label = str(getattr(res, "label", "") or "") or "symptom_consult"
            label_cn = str(getattr(res, "label_cn", "") or "") or _INTENT_CN.get(label, label)
            confidence = float(getattr(res, "confidence", 0.0) or 0.0)
            method = str(getattr(res, "method", "") or "textcnn")
            scores = dict(getattr(res, "scores", {}) or {})
            if label not in _INTENT_CN:
                label, label_cn = "symptom_consult", _INTENT_CN["symptom_consult"]
            return label, label_cn, confidence, method, scores
        except Exception as exc:  # noqa: BLE001
            logger.warning("意图识别模块不可用，降级为 symptom_consult（置信度 0.3）：%s", exc)
            return "symptom_consult", _INTENT_CN["symptom_consult"], 0.3, "fallback", {}

    # ==================================================================
    #  ② 实体识别与实体链接
    # ==================================================================
    @staticmethod
    def _empty_entities() -> Dict[str, List[str]]:
        return {k: [] for k in _DEFAULT_ENTITY_KEYS}

    def _link(self, question: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        entities = self._empty_entities()
        try:
            from app.llm_layer.entity_linker import get_entity_linker  # 并行开发，惰性导入

            data = get_entity_linker().analyze(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("实体链接模块不可用，本次问答退化为无实体模式：%s", exc)
            return [], entities
        if not isinstance(data, dict):
            return [], entities

        mentions = [m for m in (data.get("mentions") or []) if isinstance(m, dict)]
        for key in _DEFAULT_ENTITY_KEYS:
            vals = data.get(key)
            if isinstance(vals, str):
                vals = [vals]
            if not isinstance(vals, (list, tuple, set)):
                continue
            for v in vals:
                name = v if isinstance(v, str) else str(getattr(v, "kg_name", "") or getattr(v, "text", "") or v)
                name = (name or "").strip()
                if name and name not in entities[key]:
                    entities[key].append(name)

        # 兜底：只有 mentions 时按实体类型归类
        if not any(entities.values()) and mentions:
            for m in mentions:
                key = _TYPE_TO_KEY.get(str(m.get("type", "")))
                name = str(m.get("kg_name") or m.get("text") or "").strip()
                if key and name and name not in entities[key]:
                    entities[key].append(name)
        return mentions, entities

    # ==================================================================
    #  ④ 本地模板回答（无 LLM 时的确定性降级答案）
    # ==================================================================
    def _template_answer(self, bundle: ContextBundle, facts: Sequence[RetrievedFact]) -> str:
        """纯 KG 事实拼装的 markdown 回答：分槽位 + [KG-n] 引用 + 免责声明"""
        if not facts:
            parts: List[str] = ["## 知识库检索结果"]
            if bundle.safe_knowledge:
                parts.append(bundle.safe_knowledge)
            hint = (getattr(self.prompt_library, "no_data_hint", "") or "").strip() or (
                "当前知识库暂未收录与您问题直接相关的内容，为避免误导，本系统不做推测性回答。"
                "建议换用更常见的疾病 / 症状名称提问，并及时前往正规医疗机构就诊。")
            parts.append(hint)
            if "不能替代执业医师" not in "\n".join(parts):
                parts.append(settings.disclaimer)
            return "\n\n".join(p for p in parts if p).strip()

        groups: Dict[str, List[RetrievedFact]] = {}
        for f in facts:
            label = f.relation_label or f.relation or "其他"
            groups.setdefault(label, []).append(f)
        ordered = sorted(
            groups.items(),
            key=lambda kv: (_GROUP_ORDER.index(kv[0]) if kv[0] in _GROUP_ORDER else 50, kv[0]),
        )

        title = f"## 基于知识图谱的回答（主实体：{bundle.main_entity or '未识别'}）"
        lines: List[str] = [title]
        if bundle.entities_str:
            lines.append("")
            lines.append(f"**识别到的医学实体**：{bundle.entities_str}")
        for label, items in ordered:
            items_str = self._format_group_items(items, bundle)
            sentence = _GROUP_SENTENCE.get(label, "{items}。").format(items=items_str)
            lines.append("")
            lines.append(f"### {label}")
            lines.append(f"- {sentence}")

        if bundle.safe_knowledge:
            lines.append("")
            lines.append("### 安全提示")
            lines.append(bundle.safe_knowledge)

        lines.append("")
        lines.append("### 就医建议")
        lines.append("- 以上内容来自知识图谱事实，编号 [KG-n] 与图谱三元组一一对应，可在溯源面板查看原始文献。")
        lines.append("- 若症状持续、加重或出现急症信号，请立即前往正规医疗机构就诊。")
        lines.append("")
        lines.append(settings.disclaimer)
        return "\n".join(lines).strip()

    @staticmethod
    def _format_group_items(items: Sequence[RetrievedFact], bundle: ContextBundle) -> str:
        """把同一关系分组内的事实格式化为「尾实体 [KG-n]」顿号连接的串。

        输出形如 ``头晕 [KG-10]、头痛 [KG-11]``，供 ``_GROUP_SENTENCE`` 中的
        ``{items}`` 占位符使用。重复的尾实体只保留首次出现，避免冗余。
        ``bundle`` 保留于签名中以维持与 ``_template_answer`` 的调用约定。
        """
        parts: List[str] = []
        seen: set = set()
        for f in items:
            name = (f.tail or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            cite = (f.triple_id or "").strip()
            parts.append(f"{name} [{cite}]" if cite else name)
        return "、".join(parts) if parts else "（知识库暂无明细）"

    # ==================================================================
    #  富文本（markdown → 带引用角标的 HTML）
    # ==================================================================
    @staticmethod
    def _to_html(answer: str) -> str:
        """先转义 HTML，再渲染 markdown 标题/加粗，最后把 [KG-n] 包成上标引用"""
        if not answer:
            return ""
        escaped = html.escape(answer, quote=False)
        out: List[str] = []
        for raw in escaped.split("\n"):
            line = raw.rstrip()
            line = _BOLD_RE.sub(r"<strong>\1</strong>", line)
            if line.startswith("#### "):
                out.append(f"<h5>{line[5:].strip()}</h5>")
            elif line.startswith("### "):
                out.append(f"<h4>{line[4:].strip()}</h4>")
            elif line.startswith("## "):
                out.append(f"<h4>{line[3:].strip()}</h4>")
            elif line.startswith("# "):
                out.append(f"<h3>{line[2:].strip()}</h3>")
            elif line.startswith(("- ", "* ")):
                out.append("· " + line[2:])
            else:
                out.append(line)
        body = "<br>".join(out)
        body = _CITE_RE.sub(
            lambda m: f'<sup class="kg-cite" data-cite="{m.group(1)}">[{m.group(1)}]</sup>', body)
        return body

    # ==================================================================
    #  证据 / 实体模型构造（全部容错）
    # ==================================================================
    def _evidences(self, facts: Sequence[RetrievedFact]) -> List[EvidenceItem]:
        out: List[EvidenceItem] = []
        for f in facts:
            ev: Optional[EvidenceItem] = None
            try:
                ev = self.graph_service.get_evidence(
                    f.head, f.relation or None, f.tail or None, f.triple_id or "KG-1")
            except Exception as exc:  # noqa: BLE001
                logger.debug("证据构造失败（%s），改用事实字段自建", exc)
            if ev is None:
                try:
                    ev = EvidenceItem(
                        triple_id=f.triple_id or "", head=f.head, head_type=f.head_type or "",
                        relation=f.relation or "", relation_label=f.relation_label or "",
                        tail=f.tail, tail_type=f.tail_type or "",
                        confidence=float(f.confidence), sources=list(f.sources or []),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("证据自建失败，跳过该条：%s", exc)
                    continue
            elif not getattr(ev, "triple_id", ""):
                try:
                    ev.triple_id = f.triple_id
                except Exception:  # noqa: BLE001
                    pass
            out.append(ev)
        return out

    @staticmethod
    def _linked_entities(mentions: Sequence[Dict[str, Any]]) -> List[LinkedEntity]:
        out: List[LinkedEntity] = []
        for m in mentions:
            try:
                out.append(LinkedEntity(
                    text=str(m.get("text") or m.get("kg_name") or ""),
                    type=str(m.get("type") or "Disease"),
                    label=str(m.get("label") or ""),
                    kg_id=str(m.get("kg_id") or ""),
                    kg_name=str(m.get("kg_name") or m.get("text") or ""),
                    score=float(m.get("score") or 0.0),
                    method=str(m.get("method") or "dictionary"),
                    start=int(m.get("start") or 0),
                    end=int(m.get("end") or 0),
                ))
            except Exception as exc:  # noqa: BLE001
                logger.debug("实体模型构造失败，跳过：%s", exc)
        return out

    @staticmethod
    def _intent_model(label: str, label_cn: str, confidence: float, method: str,
                      scores: Dict[str, float]) -> IntentResult:
        try:
            lab = IntentLabel(label)
        except Exception:  # noqa: BLE001
            lab = IntentLabel.DISEASE_QUERY
        return IntentResult(
            label=lab, label_cn=label_cn or _INTENT_CN.get(label, label),
            confidence=round(float(confidence or 0.0), 4), method=method or "rule_fallback",
            scores={k: round(float(v), 4) for k, v in (scores or {}).items()},
        )

    @staticmethod
    def _step(steps: List[ReasoningStep], index: int, action: str, title: str,
              detail: str, elapsed_ms: float, payload: Optional[Dict[str, Any]] = None) -> None:
        steps.append(ReasoningStep(
            step=index, action=action, title=title, detail=detail,
            elapsed_ms=round(float(elapsed_ms or 0.0), 2), payload=payload or {},
        ))

    @staticmethod
    def _simple_context(facts: Sequence[RetrievedFact]) -> str:
        """上下文构建器异常时的极简兜底文本，保证 prompt 里一定有事实"""
        lines = [f"【知识图谱事实 —— 共 {len(facts)} 条，编号与回答中的 [KG-n] 引用一一对应】", ""]
        for f in facts:
            lines.append(f"[{f.triple_id or 'KG-?'}] {f.head} —{f.relation_label or f.relation}→ {f.tail}"
                         f"（置信度 {float(f.confidence):.2f}）")
        return "\n".join(lines)

    # ==================================================================
    #  主流程
    # ==================================================================
    async def answer(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 12,
        max_hops: int = 2,
        explain: bool = True,
        use_llm: bool = True,
        history: Optional[List[dict]] = None,
        slots: Optional[dict] = None,
    ) -> QAResponse:
        question = (question or "").strip()
        session_id = session_id or "default"
        top_k = int(top_k or settings.RAG_TOP_K or 12)
        max_hops = int(max_hops or settings.RAG_MAX_HOPS or 2)
        sw = Stopwatch()
        steps: List[ReasoningStep] = []
        lat = LatencyBreakdown()

        # ---------- ① 意图识别 ----------
        label, label_cn, intent_conf, method, intent_scores = self._classify(question)
        lat.intent = sw.mark("intent")
        self._step(
            steps, 1, "intent_classify", "意图识别",
            f"{_METHOD_CN.get(method, method)} → {label}（{label_cn}），置信度 {intent_conf:.2f}",
            lat.intent,
            {"label": label, "label_cn": label_cn, "confidence": round(intent_conf, 4),
             "method": method, "scores": intent_scores},
        )

        # ---------- ② 实体识别与实体链接 ----------
        mentions, entities = self._link(question)
        lat.entity_link = sw.mark("entity_link")
        if mentions:
            detail = "、".join(
                f"「{m.get('text') or m.get('kg_name')}」→ {m.get('type')}/{m.get('kg_name')}"
                f"(score {float(m.get('score') or 0.0):.2f}, {m.get('method') or 'dictionary'})"
                for m in mentions[:8]
            )
        else:
            detail = "未识别到图谱实体（将依赖意图先验、症状反向检索与推理补全）"
        self._step(
            steps, 2, "entity_link", "实体识别与实体链接", detail, lat.entity_link,
            {"mentions": len(mentions),
             "entities": {k: v for k, v in entities.items() if v}},
        )

        # ---------- ③ 知识图谱检索 ----------
        facts: List[RetrievedFact] = []
        try:
            facts = await self.retriever.retrieve(
                question, entities, label, top_k=top_k, max_hops=max_hops)
        except Exception as exc:  # noqa: BLE001
            logger.error("知识图谱检索失败，降级为空事实：%s", exc)
            facts = []
        lat.kg_query = sw.mark("kg_query")
        lat.semantic_parse_total = round(lat.intent + lat.entity_link + lat.kg_query, 2)
        if lat.semantic_parse_total > 500:
            logger.warning("语义解析总耗时 %.0fms 超过 500ms 指标（意图 %.0f + 实体 %.0f + 图谱 %.0f）",
                           lat.semantic_parse_total, lat.intent, lat.entity_link, lat.kg_query)

        stats = dict(getattr(self.retriever, "last_stats", {}) or {})
        direct = max(0, stats.get("kg_1hop", 0) - stats.get("reverse", 0))
        reverse = stats.get("reverse", 0)
        if not stats:
            direct = sum(1 for f in facts if f.source == "kg_1hop"
                         and not f.explain.startswith("反向检索"))
            reverse = sum(1 for f in facts if f.explain.startswith("反向检索"))
            stats = {
                "kg_1hop": sum(1 for f in facts if f.source == "kg_1hop"),
                "kg_2hop": sum(1 for f in facts if f.source == "kg_2hop"),
                "kg_reasoned": sum(1 for f in facts if f.source == "kg_reasoned"),
                "kg_similar": sum(1 for f in facts if f.source == "kg_similar"),
            }
        self._step(
            steps, 3, "kg_retrieve", "知识图谱检索",
            f"召回 {len(facts)} 条三元组（直接事实 {direct} / 反向检索 {reverse} / "
            f"多跳 {stats.get('kg_2hop', 0)} / 推理补全 {stats.get('kg_reasoned', 0)}）",
            lat.kg_query,
            {"facts": len(facts), "stats": stats,
             "semantic_parse_total_ms": lat.semantic_parse_total, "top_k": top_k, "max_hops": max_hops},
        )

        # ---------- ④ 上下文构建 ----------
        try:
            bundle = await self.context_builder.build(
                question, entities, label, facts, slots=slots, history=history, intent_cn=label_cn)
        except Exception as exc:  # noqa: BLE001
            logger.error("上下文构建失败，使用极简兜底上下文：%s", exc)
            kg_context = self._simple_context(facts)
            bundle = ContextBundle(
                kg_context=kg_context, triple_count=len(facts),
                entities_str="", main_entity=(facts[0].head if facts else ""),
                conversation_slots="（无）", variables={
                    "question": question, "question_norm": question, "intent_cn": label_cn,
                    "entities": "", "main_entity": (facts[0].head if facts else "未识别"),
                    "kg_context": kg_context, "kg_triple_count": len(facts),
                    "safe_knowledge": "", "conversation_slots": "（无）", "history": "（无）",
                    "department_list": "", "chunk_context": "", "user_profile": "",
                    "risk_features": "", "no_data_hint": getattr(self.prompt_library, "no_data_hint", ""),
                })
        lat.retrieve = sw.mark("retrieve")
        self._step(
            steps, 4, "rag_generate", "构建 RAG 上下文",
            f"按槽位组织 {bundle.triple_count} 条三元组，命中 {getattr(bundle, 'chunk_count', 0)} 条文献片段",
            lat.retrieve,
            {"slots": list((bundle.slot_summary or {}).keys()),
             "main_entity": bundle.main_entity,
             "kg_context_chars": len(bundle.kg_context or "")},
        )

        # ---------- ⑤ 模板选择与 Prompt 渲染 ----------
        tpl: Optional[PromptTemplate] = None
        prompt = ""
        try:
            tpl = self.prompt_library.select(label, question, entities, slots=slots)
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt 模板选择失败：%s", exc)
        if tpl is not None:
            try:
                prompt = self.prompt_library.render(tpl, bundle.variables)
            except Exception as exc:  # noqa: BLE001
                logger.error("prompt 渲染失败，使用模板原文：%s", exc)
                prompt = tpl.template
        else:
            prompt = bundle.kg_context
        self._step(
            steps, 5, "rag_generate", "医疗专属 Prompt 渲染",
            f"命中模板 {getattr(tpl, 'id', '（无）')}（{getattr(tpl, 'name', '兜底上下文')}），"
            f"prompt {len(prompt)} 字",
            sw.mark("prompt"),
            {"template_id": getattr(tpl, "id", ""), "prompt_chars": len(prompt)},
        )

        # ---------- ⑥ 大模型生成（失败一律降级为本地模板回答） ----------
        answer = ""
        llm_model = "local-template-rag"
        llm_note = ""
        if use_llm and facts and tpl is not None:
            try:
                from app.llm_layer.llm_client import get_llm_client  # 并行开发，惰性导入

                client = get_llm_client()
                result = await client.generate(
                    prompt,
                    temperature=getattr(tpl, "temperature", None),
                    max_tokens=getattr(tpl, "max_tokens", None),
                )
                text = str(getattr(result, "text", "") or "").strip()
                if bool(getattr(result, "success", False)) and text:
                    answer = text
                    llm_model = str(getattr(result, "model", "") or "") or \
                        str(getattr(client, "provider", "") or "llm")
                else:
                    llm_note = f"模型返回失败（{getattr(result, 'error', '') or '空响应'}）"
                    logger.warning("大模型生成失败：%s，降级为本地 KG 模板回答", llm_note)
            except Exception as exc:  # noqa: BLE001
                llm_note = f"LLM 客户端不可用（{exc}）"
                logger.warning("调用大模型异常，降级为本地 KG 模板回答：%s", exc)
        elif not facts:
            llm_note = "无图谱事实，直接使用安全模板回答"
        elif not use_llm:
            llm_note = "调用方要求仅返回 KG 结构化答案（use_llm=False）"
        else:
            llm_note = "未命中 prompt 模板，使用兜底上下文回答"

        if not answer:
            answer = self._template_answer(bundle, facts)
        lat.llm = sw.mark("llm")
        self._step(
            steps, 6, "rag_generate", "大模型生成（Llama 3 / 模板降级）",
            (f"模型 {llm_model} 生成 {len(answer)} 字" if llm_model != "local-template-rag"
             else f"本地 KG 模板回答 {len(answer)} 字（{llm_note}）"),
            lat.llm,
            {"model": llm_model, "fallback": llm_model == "local-template-rag", "note": llm_note},
        )

        # ---------- ⑦ 幻觉守卫 ----------
        guard_result = GuardResult(
            passed=True, score=1.0, suggested_action="accept",
            summary="幻觉守卫未启用（settings.RAG_HALLUCINATION_GUARD=False）",
        )
        if settings.RAG_HALLUCINATION_GUARD and self.guard is not None:
            try:
                guard_result = self.guard.check(answer, facts, entities)
                if guard_result.suggested_action == "fallback_template":
                    logger.warning("幻觉守卫触发兜底：%s", guard_result.summary)
                    answer = self.guard.strip_and_fallback(answer, facts)
            except Exception as exc:  # noqa: BLE001
                logger.error("幻觉守卫执行失败，跳过校验：%s", exc)
                guard_result = GuardResult(passed=True, score=0.8, suggested_action="accept",
                                           summary=f"幻觉守卫异常（{exc}），已跳过校验")
        lat.guard = sw.mark("guard")
        self._step(
            steps, 7, "guard_check", "幻觉守卫与事实校验",
            guard_result.summary or "校验完成",
            lat.guard,
            self.guard.explain(guard_result) if self.guard is not None else {},
        )

        # ---------- ⑧ 置信度与收尾 ----------
        top_conf = [float(f.confidence) for f in facts[:3]]
        mean_conf = (sum(top_conf) / len(top_conf)) if top_conf else 0.0
        guard_score = float(getattr(guard_result, "score", 1.0) or 0.0)
        confidence = 0.35 * float(intent_conf) + 0.35 * mean_conf + 0.30 * guard_score
        confidence = round(max(0.0, min(1.0, confidence)), 3)

        safety_hit = bool(tpl is not None and (
            getattr(tpl, "intent", "") in ("emergency", "safety")
            or str(getattr(tpl, "id", "")).startswith(("emergency_", "safety_"))
        ))
        # ------------------------------------------------------------------
        #  领域外问题拦截（安全边界）
        #  ------------------------------------------------------------------
        #  若问句既未识别到任何医学实体、也未召回任何图谱事实，且**完全不含医疗相关
        #  词汇**，则判定为「超范围问题」，改用 safety_out_of_scope_v1 模板礼貌拒答，
        #  而不是拿医学兜底话术去回答"今天天气怎么样"（那既不专业也显得答非所问）。
        out_of_scope = False
        if not facts and not any(entities.values()):
            try:
                out_of_scope = not self.prompt_library.is_medical_query(question)
            except Exception:  # noqa: BLE001
                out_of_scope = False
            if out_of_scope:
                oos = self.prompt_library.get("safety_out_of_scope_v1")
                if oos is not None:
                    tpl = oos
                    safety_hit = True
                    logger.info("判定为超范围问题，切换模板 safety_out_of_scope_v1")

        if not facts:
            if out_of_scope:
                # 直接使用该模板的正文（去掉 YAML 缩进后的静态话术）
                answer = _OUT_OF_SCOPE_ANSWER
                confidence = round(min(confidence, 0.25), 3)
            elif safety_hit:
                confidence = round(min(confidence, 0.45), 3)
            else:
                answer = (getattr(self.prompt_library, "no_data_hint", "") or "").strip() or \
                    "当前知识库暂未收录与您问题直接相关的内容，建议换用更常见的疾病或症状名称提问，并及时就医。"
                confidence = round(min(confidence, 0.30), 3)

        related: List[str] = []
        try:
            related = list(self.graph_service.related_questions(bundle.main_entity or None, 5) or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("推荐追问生成失败：%s", exc)

        answer_html = self._to_html(answer)
        evidences = self._evidences(facts)
        lat.total = sw.lap("total")
        if lat.semantic_parse_total <= 0:
            lat.semantic_parse_total = round(lat.intent + lat.entity_link + lat.kg_query, 2)

        response = QAResponse(
            question=question,
            answer=answer,
            answer_html=answer_html,
            intent=self._intent_model(label, label_cn, intent_conf, method, intent_scores),
            entities=self._linked_entities(mentions),
            evidences=evidences,
            kg_context=bundle.kg_context or "",
            reasoning_trace=list(steps) if explain else [],
            confidence=confidence,
            prompt_template_id=str(getattr(tpl, "id", "") or ""),
            prompt_used=prompt,
            guard_result=(self.guard.explain(guard_result) if self.guard is not None else {}),
            latency_ms=lat,
            llm_model=llm_model,
            session_id=session_id,
            disclaimer=settings.disclaimer,
            related_questions=related,
        )

        # 会话记忆（内存，每会话最多 40 条）
        self._append_history(session_id, ChatMessage(
            role="user", content=question, timestamp=self._now(),
        ))
        self._append_history(session_id, ChatMessage(
            role="assistant", content=answer, timestamp=self._now(),
            answer_html=answer_html, evidences=evidences,
            reasoning_trace=list(steps) if explain else [], confidence=confidence,
        ))

        logger.info(
            "问答完成：意图=%s 置信度=%.3f 事实=%d 守卫=%s 总耗时=%.1fms（语义解析 %.1fms）",
            label, confidence, len(facts), guard_result.suggested_action, lat.total,
            lat.semantic_parse_total,
        )
        return response

    # ==================================================================
    #  会话记忆
    # ==================================================================
    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_history(self, session_id: str, message: ChatMessage) -> None:
        bucket = self._sessions.setdefault(session_id or "default", [])
        bucket.append(message)
        if len(bucket) > self.MAX_SESSION_MESSAGES:
            del bucket[: len(bucket) - self.MAX_SESSION_MESSAGES]

    def session_history(self, session_id: str) -> List[ChatMessage]:
        """返回某会话的历史消息（副本，最多 40 条）"""
        return list(self._sessions.get(session_id or "default", []))

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id or "default", None)

    # ==================================================================
    #  元信息
    # ==================================================================
    def get_prompt_library_meta(self) -> Dict[str, Any]:
        """prompt 模板库元信息（前端"模板库"面板 / 健康检查使用）"""
        try:
            items = list(self.prompt_library.list_meta())
            stats = dict(self.prompt_library.stats())
            validate = dict(self.prompt_library.validate())
        except Exception as exc:  # noqa: BLE001
            logger.error("prompt 模板库元信息读取失败：%s", exc)
            items, stats, validate = [], {}, {"ok": False, "error": str(exc)}
        return {"stats": stats, "validate": validate, "total": len(items), "items": items}

    def info(self) -> Dict[str, Any]:
        """引擎自检信息（/health 使用）"""
        llm_info: Dict[str, Any] = {"provider": settings.LLM_PROVIDER, "client_available": False}
        try:
            from app.llm_layer.llm_client import get_llm_client  # 惰性

            client = get_llm_client()
            llm_info["client_available"] = True
            llm_info["client"] = client.info() if hasattr(client, "info") else {}
        except Exception as exc:  # noqa: BLE001
            llm_info["error"] = str(exc)
        memory = getattr(self.graph_service, "memory", None)
        return {
            "engine": "RagEngine",
            "data_source": getattr(self.graph_service, "data_source", "unknown"),
            "graph_nodes": getattr(memory, "node_count", 0),
            "graph_edges": getattr(memory, "edge_count", 0),
            "llm": llm_info,
            "prompt_templates": len(getattr(self.prompt_library, "templates", []) or []),
            "retriever": {
                "available": self.retriever is not None,
                "routed": ["kg_1hop", "kg_1hop_reverse", "kg_2hop", "kg_reasoned", "kg_similar"],
            },
            "guard": {
                "available": self.guard is not None,
                "enabled": bool(settings.RAG_HALLUCINATION_GUARD),
            },
            "sessions": len(self._sessions),
            "rag": {
                "top_k": settings.RAG_TOP_K, "max_hops": settings.RAG_MAX_HOPS,
                "min_confidence": settings.RAG_MIN_CONFIDENCE,
            },
        }


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_engine: Optional[RagEngine] = None


def get_rag_engine() -> RagEngine:
    """RAG 引擎全局单例（FastAPI 依赖注入使用）"""
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


__all__ = ["RagEngine", "get_rag_engine"]
