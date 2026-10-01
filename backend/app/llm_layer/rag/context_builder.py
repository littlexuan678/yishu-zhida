# -*- coding: utf-8 -*-
"""
RAG 上下文构建器（ContextBuilder）
=================================

★ 为什么把 KG 三元组组织成「带编号的槽位化文本块」★

通用 RAG 把检索到的文本 chunk 拼接后塞进 prompt，模型面对的是"一大段散文"：
既不知道哪句话对应哪条事实，也无法按要求输出引用角标。本模块做了三件事：

1. **槽位化（slot-organized）**：把三元组按「症状 / 治疗 / 用药 / 科室 / 检查 / 并发症 /
   鉴别诊断 / 多跳推理 / 推理补全」分组，与医疗 prompt 的分点作答结构同构，
   llama3:8b 这类小模型也能稳定按槽位复述，显著降低幻觉。
2. **编号化（numbered）**：每条事实分配唯一编号 `[KG-n]`，`KG-n` 与 `EvidenceItem.triple_id`
   严格一一对应。模型在答案里写 `[KG-3]`，前端就能把角标点开、跳到溯源面板，
   实现"每一句结论都可追溯"。
3. **来源内联（inline provenance）**：在事实行尾附 `（置信度 0.98，来源：PubMed 32130469）`，
   让模型在组织语言时"看得见"证据等级。

其它降级设计：
  * `chunk_context`：若 `CORPUS_DIR/corpus.jsonl` 存在则做关键词重叠检索，返回 2~3 条 PubMed
    片段；文件缺失 / 解析失败 / 无命中一律返回空串，**永不抛异常**。
  * `safe_knowledge`：与图谱无关的通用安全常识池（急症信号、孕期、儿童、剂量红线…），
    保证即使图谱检索为空，回答也不会忽略"该就医"的信号。
  * 整个构建过程不依赖 Neo4j、torch、网络；只有 JSON/JSONL 读取是可选增强。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.config import settings
from app.kg_layer.graph_service import get_graph_service
from app.schemas import ENTITY_TYPE_META
from app.utils.logger import get_logger
from app.utils.text import keywords, normalize, truncate

logger = get_logger(__name__)

#: 兜底意图中文名（优先使用 intent_classifier.INTENT_CN，避免导入 torch 依赖时回退到这里）
_FALLBACK_INTENT_CN: Dict[str, str] = {
    "disease_query": "疾病查询",
    "symptom_consult": "症状咨询",
    "treatment_query": "治疗咨询",
    "department_query": "科室导诊",
}

#: 实体类型 → 中文名（与 schemas.ENTITY_TYPE_META 保持一致，仅做容错）
_TYPE_CN: Dict[str, str] = {k: v.get("label", k) for k, v in ENTITY_TYPE_META.items()}

#: 分组标题（与 PPT / 前端溯源面板文案一致）
GROUP_PRIMARY_FALLBACK = "目标疾病直接事实"
GROUP_RELATED = "相关疾病（由症状反向检索得到）"
GROUP_REASONED = "图谱推理补全（Jang 规则 / GAT-GCN）"
GROUP_MULTIHOP = "多跳推理路径"

#: 安全常识池：风险类型 → (触发关键词, 提示文案)
#: 这部分**不依赖知识图谱**，是医学常识底线，保证任何情况下都给出就医指引。
SAFE_KNOWLEDGE: List[Tuple[str, Tuple[str, ...], str]] = [
    ("急症信号",
     ("胸痛", "胸闷", "心前区", "濒死", "呼吸困难", "喘不上气", "憋气", "意识", "晕厥", "昏倒"),
     "如出现剧烈胸痛、呼吸困难、意识障碍，请立即拨打120。"),
    ("卒中信号",
     ("口角歪斜", "言语不清", "说不出话", "半身不遂", "偏瘫", "肢体无力", "麻木", "抽搐"),
     "突发口角歪斜、言语不清、一侧肢体无力，请立即拨打120，争取溶栓/取栓时间窗。"),
    ("出血信号",
     ("出血", "呕血", "咯血", "便血", "血便", "黑便", "止不住血"),
     "持续或大量出血需立即急诊就医，不要自行服用止血药或止痛药掩盖病情。"),
    ("高热与惊厥",
     ("高热", "发烧", "发热", "惊厥", "抽风", "寒战"),
     "体温持续超过39℃或伴抽搐、意识改变时应立即就医；儿童抽搐时侧卧防误吸，勿强行按压肢体。"),
    ("孕期哺乳期",
     ("孕妇", "怀孕", "妊娠", "哺乳", "备孕", "孕期", "月经"),
     "孕期、哺乳期用药必须由产科或专科医师评估，任何药物都不要自行服用。"),
    ("儿童用药",
     ("儿童", "孩子", "小孩", "宝宝", "婴儿", "小儿", "新生儿"),
     "儿童用药剂量按体重计算，禁止直接套用成人剂量，请咨询儿科医师。"),
    ("老年与多重用药",
     ("老人", "老年", "高龄", "基础病", "慢性病", "长期服药"),
     "老年人常合并多种慢病与多重用药，就诊时请携带完整用药清单与既往检查报告。"),
    ("用药剂量红线",
     ("剂量", "几片", "几粒", "多少毫克", "一天几次", "怎么吃", "用量", "加量"),
     "本系统不提供处方剂量；具体剂量、频次与疗程必须由执业医师根据个体情况决定。"),
    ("药物相互作用",
     ("一起吃", "同时吃", "同服", "相互作用", "配伍", "叠加"),
     "多种药物联用可能产生相互作用，请由医师或药师审核后再服用，不要自行加减。"),
    ("过敏反应",
     ("过敏", "皮疹", "瘙痒", "红肿", "面唇肿"),
     "出现皮疹、瘙痒、面唇肿胀或呼吸困难等过敏表现，应立即停药并就医。"),
    ("心理危机",
     ("自杀", "自残", "自伤", "不想活", "轻生", "活不下去"),
     "如出现伤害自己的念头，请立即联系家人，并拨打心理援助热线（12356）或前往急诊。"),
    ("一般就医指引",
     (),
     "本系统提供医学知识科普与就医指引，任何不适请及时前往正规医疗机构由执业医师面诊。"),
]

#: 单条事实行内三元组的对齐宽度（中文按字符计，取一个视觉上整齐的近似值）
_LINE_PAD = 38


def _intent_cn(intent: str, override: str = "") -> str:
    if override:
        return override
    try:  # 惰性导入，避免拉起 intent_classifier 里的可选重依赖
        from app.llm_layer.intent_classifier import INTENT_CN  # type: ignore

        if intent in INTENT_CN:
            return str(INTENT_CN[intent])
    except Exception:  # noqa: BLE001
        pass
    return _FALLBACK_INTENT_CN.get(intent, intent)


@dataclass
class ContextBundle:
    """喂给大模型 prompt 的完整变量包（字段名与 prompt_library.yaml 占位符对应）"""

    kg_context: str = ""
    triple_count: int = 0
    slot_summary: Dict[str, List[str]] = field(default_factory=dict)
    entities_str: str = ""
    main_entity: str = ""
    conversation_slots: str = "（无）"
    department_list: str = ""
    chunk_context: str = ""
    safe_knowledge: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triple_count": self.triple_count, "main_entity": self.main_entity,
            "entities_str": self.entities_str, "slot_summary": self.slot_summary,
            "department_list": self.department_list, "chunk_count": self.chunk_count,
            "kg_context": self.kg_context,
        }


class ContextBuilder:
    """把检索到的三元组事实组织成「槽位化 + 编号化 + 带来源」的 prompt 上下文"""

    #: corpus.jsonl 进程内缓存：{路径: [记录, ...]}
    _CORPUS_CACHE: Dict[str, List[Dict[str, str]]] = {}

    def __init__(self, graph_service: Any = None) -> None:
        self.graph_service = graph_service or get_graph_service()

    # ==================================================================
    #  对外主入口
    # ==================================================================
    async def build(
        self,
        question: str,
        entities: Optional[Dict[str, List[str]]] = None,
        intent: str = "disease_query",
        facts: Optional[Sequence[Any]] = None,
        slots: Optional[Dict[str, Any]] = None,
        history: Optional[List[dict]] = None,
        intent_cn: str = "",
    ) -> ContextBundle:
        facts = list(facts or [])
        entities = entities if isinstance(entities, dict) else {}
        question = question or ""

        primary = self._main_entity(entities, facts)
        kg_context = self._render_kg_context(facts, primary)
        slot_summary = self._slot_summary(facts, primary)
        entities_str = self._entities_str(entities)
        departments = self._department_list(facts) if intent == "department_query" else ""
        safe_knowledge = self._safe_knowledge(question)
        chunk_context, chunk_count = self._chunk_context(question)
        conversation_slots = self._slots_text(slots)
        history_text = self._history_text(history)

        variables: Dict[str, Any] = {
            "question": question,
            "question_norm": normalize(question),
            "intent_cn": _intent_cn(intent, intent_cn),
            "entities": entities_str,
            "main_entity": primary or "未识别",
            "kg_context": kg_context,
            "kg_triple_count": len(facts),
            "safe_knowledge": safe_knowledge,
            "conversation_slots": conversation_slots,
            "history": history_text,
            "department_list": departments,
            "chunk_context": chunk_context,
            "user_profile": "",
            "risk_features": "",
            "entities_echo": primary or "该情况",
        }
        try:
            from app.llm_layer.rag.prompt_templates import get_prompt_library  # type: ignore

            variables["no_data_hint"] = get_prompt_library().no_data_hint
        except Exception:  # noqa: BLE001
            variables["no_data_hint"] = "当前知识库暂未收录与您问题直接相关的内容，建议换用更常见的疾病或症状名称提问，并及时就医。"

        return ContextBundle(
            kg_context=kg_context,
            triple_count=len(facts),
            slot_summary=slot_summary,
            entities_str=entities_str,
            main_entity=primary,
            conversation_slots=conversation_slots,
            department_list=departments,
            chunk_context=chunk_context,
            safe_knowledge=safe_knowledge,
            variables=variables,
            chunk_count=chunk_count,
        )

    # ==================================================================
    #  主实体 / 实体串 / 槽位
    # ==================================================================
    @staticmethod
    def _ordered_entity_names(entities: Dict[str, Any]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        seen: set = set()
        order = ("diseases", "symptoms", "departments", "drugs", "checks", "treatments", "populations")
        for key in order:
            for v in entities.get(key) or []:
                name = v if isinstance(v, str) else str(getattr(v, "kg_name", "") or getattr(v, "text", "") or v)
                name = (name or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append((name, key))
        return out

    def _main_entity(self, entities: Dict[str, Any], facts: Sequence[Any]) -> str:
        """主实体：第一个疾病 → 第一个其他类型实体 → 图谱事实里出现最多的疾病"""
        ordered = self._ordered_entity_names(entities)
        for name, key in ordered:
            if key == "diseases":
                return name
        if ordered:
            return ordered[0][0]
        counter: Dict[str, int] = {}
        for f in facts:
            if str(getattr(f, "head_type", "")) == "Disease" and getattr(f, "head", ""):
                counter[str(f.head)] = counter.get(str(f.head), 0) + 1
        if counter:
            return max(counter.items(), key=lambda kv: kv[1])[0]
        return ""

    def _entities_str(self, entities: Dict[str, Any], limit: int = 8) -> str:
        """'原发性高血压(Disease/疾病)、头晕(Symptom/症状)'"""
        parts: List[str] = []
        type_map = {"diseases": "Disease", "symptoms": "Symptom", "departments": "Department",
                    "drugs": "Drug", "checks": "Check", "treatments": "Treatment",
                    "populations": "Population"}
        for name, key in self._ordered_entity_names(entities)[:limit]:
            etype = type_map.get(key, key)
            parts.append(f"{name}({etype}/{_TYPE_CN.get(etype, etype)})")
        return "、".join(parts)

    def _slot_summary(self, facts: Sequence[Any], primary: str) -> Dict[str, List[str]]:
        """{关系中文名: [尾实体, ...]}，优先统计主实体直接事实"""
        target = [f for f in facts if primary and str(getattr(f, "head", "")) == primary]
        if not target:
            target = list(facts)
        summary: Dict[str, List[str]] = {}
        for f in target:
            label = str(getattr(f, "relation_label", "") or getattr(f, "relation", "") or "其他")
            tail = str(getattr(f, "tail", ""))
            if not tail:
                continue
            bucket = summary.setdefault(label, [])
            if tail not in bucket:
                bucket.append(tail)
        return summary

    def _department_list(self, facts: Sequence[Any]) -> str:
        out: List[str] = []
        for f in facts:
            if str(getattr(f, "relation", "")) != "BELONGS_TO":
                continue
            tail = str(getattr(f, "tail", "")).strip()
            if tail and tail not in out:
                out.append(tail)
        return "、".join(out)

    @staticmethod
    def _slots_text(slots: Optional[Dict[str, Any]]) -> str:
        """多轮槽位 → 可读中文行"""
        if not slots:
            return "（无）"
        lines: List[str] = []
        for k, v in slots.items():
            if v in (None, "", [], {}):
                continue
            if isinstance(v, (list, tuple, set)):
                val = "、".join(str(x) for x in v)
            elif isinstance(v, dict):
                val = "；".join(f"{ik}：{iv}" for ik, iv in v.items())
            else:
                val = str(v)
            lines.append(f"- {k}：{val}")
        return "\n".join(lines) if lines else "（无）"

    @staticmethod
    def _history_text(history: Optional[List[dict]], limit: int = 6) -> str:
        if not history:
            return "（无）"
        lines: List[str] = []
        for msg in history[-limit:]:
            if not isinstance(msg, dict):
                continue
            role = {"user": "用户", "assistant": "AI 医生"}.get(str(msg.get("role", "")), str(msg.get("role", "")))
            content = truncate(str(msg.get("content", "")).replace("\n", " "), 80)
            if content:
                lines.append(f"- {role}：{content}")
        return "\n".join(lines) if lines else "（无）"

    # ==================================================================
    #  KG 上下文渲染（核心）
    # ==================================================================
    def _group_facts(self, facts: Sequence[Any], primary: str,
                     number: Optional[Dict[int, int]] = None) -> List[Tuple[str, List[Any]]]:
        primary_facts: List[Any] = []
        related: List[Any] = []
        reasoned: List[Any] = []
        multihop: List[Any] = []
        for f in facts:
            source = str(getattr(f, "source", ""))
            if source == "kg_reasoned":
                reasoned.append(f)
            elif source == "kg_2hop":
                multihop.append(f)
            elif primary and str(getattr(f, "head", "")) == primary:
                primary_facts.append(f)
            else:
                related.append(f)

        groups: List[Tuple[str, List[Any]]] = []
        if primary_facts:
            heading = self._primary_heading(primary) if primary else GROUP_PRIMARY_FALLBACK
            groups.append((heading, primary_facts))
        if related:
            groups.append((GROUP_RELATED, related))
        if reasoned:
            groups.append((GROUP_REASONED, reasoned))
        if multihop:
            groups.append((GROUP_MULTIHOP, multihop))
        if not groups and facts:  # 极端兜底：没有任何分组标题时也把事实吐出来
            groups.append((GROUP_PRIMARY_FALLBACK, list(facts)))

        # 分组顺序：主实体分组固定第一，其余按组内最小 [KG-n] 编号排序，
        # 这样渲染出来的编号是单调递增的（模型引用与前端角标都不会"跳号"）
        if number and len(groups) > 1:
            first = groups[0]
            rest = sorted(
                groups[1:],
                key=lambda kv: min((number.get(id(f), 10 ** 6) for f in kv[1]), default=10 ** 6),
            )
            groups = [first] + rest
        return groups

    def _primary_heading(self, primary: str) -> str:
        category = ""
        try:
            detail = self.graph_service.get_disease_detail(primary)
            if detail is not None:
                c1, c2 = getattr(detail, "category1", None), getattr(detail, "category2", None)
                if c1 or c2:
                    category = f"（一级分类：{c1 or '未分类'} / 二级分类：{c2 or '未分类'}）"
        except Exception as exc:  # noqa: BLE001
            logger.debug("疾病分类信息获取失败（%s），降级为纯疾病名标题", exc)
        return f"疾病：{primary}{category}" if primary else GROUP_PRIMARY_FALLBACK

    @staticmethod
    def _source_suffix(fact: Any) -> str:
        """行内来源：PubMed <pmid> / <journal> <year> / <title(40)>，最多一个来源"""
        sources = list(getattr(fact, "sources", None) or [])
        if not sources:
            return ""
        s = sources[0]
        pmid = getattr(s, "pmid", None)
        journal = getattr(s, "journal", None)
        year = getattr(s, "year", None)
        title = getattr(s, "title", "") or ""
        if pmid:
            tag = f"PubMed {pmid}"
        elif journal and year:
            tag = f"{journal} {year}"
        elif journal:
            tag = str(journal)
        elif title:
            tag = truncate(str(title), 40)
        else:
            tag = str(getattr(s, "source_type", "") or "")
        return f"，来源：{tag}" if tag else ""

    def _render_kg_context(self, facts: Sequence[Any], primary: str) -> str:
        if not facts:
            return ("【知识图谱事实 —— 共 0 条】\n"
                    "当前知识库中没有检索到与该问题直接相关的三元组事实，"
                    "请勿依据模型自身记忆作答，只能给出就医指引与安全提示。")

        number: Dict[int, int] = {id(f): i for i, f in enumerate(facts, start=1)}
        header = (f"【知识图谱事实 —— 共 {len(facts)} 条，编号与回答中的 [KG-n] 引用一一对应】")
        lines: List[str] = [header]
        for heading, group in self._group_facts(facts, primary, number):
            lines.append("")
            lines.append(f"■ {heading}")
            lines.append("")
            for f in group:
                idx = number.get(id(f), 1)
                cid = str(getattr(f, "triple_id", "") or f"KG-{idx}")
                label = str(getattr(f, "relation_label", "") or getattr(f, "relation", ""))
                text = f"[{cid}] {getattr(f, 'head', '')} —{label}→ {getattr(f, 'tail', '')}"
                padded = text.ljust(max(_LINE_PAD, len(text) + 2))
                conf = float(getattr(f, "confidence", 0.0) or 0.0)
                suffix = self._source_suffix(f)
                if str(getattr(f, "source", "")) == "kg_2hop" and getattr(f, "explain", ""):
                    lines.append(f"{padded}（置信度 {conf:.2f}，路径：{getattr(f, 'explain', '')}）")
                else:
                    lines.append(f"{padded}（置信度 {conf:.2f}{suffix}）")
        return "\n".join(lines)

    # ==================================================================
    #  安全常识池
    # ==================================================================
    @staticmethod
    def _safe_knowledge(question: str, max_items: int = 3) -> str:
        q = question or ""
        hits: List[str] = []
        for name, kws, tip in SAFE_KNOWLEDGE:
            if not kws:
                continue
            if any(k in q for k in kws):
                hits.append(f"- 【{name}】{tip}")
                if len(hits) >= max_items:
                    break
        if not hits:
            hits = [f"- 【一般就医指引】{SAFE_KNOWLEDGE[-1][2]}"]
        return "\n".join(hits)

    # ==================================================================
    #  PubMed 文献片段（可选增强，永不抛异常）
    # ==================================================================
    @classmethod
    def _load_corpus(cls, path: Path, max_lines: int = 2000) -> List[Dict[str, str]]:
        key = str(path)
        if key in cls._CORPUS_CACHE:
            return cls._CORPUS_CACHE[key]
        records: List[Dict[str, str]] = []
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    for i, raw in enumerate(fh):
                        if i >= max_lines:
                            break
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except Exception:  # noqa: BLE001
                            continue
                        if not isinstance(obj, dict):
                            continue
                        records.append({
                            "pmid": str(obj.get("pmid") or obj.get("PMID") or ""),
                            "title": str(obj.get("title") or obj.get("Title") or ""),
                            "abstract": str(obj.get("abstract") or obj.get("Abstract") or ""),
                        })
                logger.info("文献语料加载完成：%d 条（%s）", len(records), path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("文献语料加载失败（已忽略）：%s", exc)
        cls._CORPUS_CACHE[key] = records
        return records

    @staticmethod
    def _snippet(text: str, kws: Sequence[str], max_len: int = 200) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return ""
        pos = -1
        for k in kws:
            p = text.find(k)
            if p >= 0 and (pos < 0 or p < pos):
                pos = p
        if pos < 0:
            return truncate(text, max_len)
        start = max(0, pos - 40)
        piece = text[start:start + max_len]
        if start > 0:
            piece = "…" + piece
        if start + max_len < len(text):
            piece = piece + "…"
        return piece

    def _chunk_context(self, question: str) -> Tuple[str, int]:
        """关键词重叠检索 corpus.jsonl，返回 (文本, 命中片段数)；任何异常都返回空"""
        try:
            cands: List[Path] = []
            try:
                cands.append(settings.abspath(settings.CORPUS_DIR) / "corpus.jsonl")
            except Exception:  # noqa: BLE001
                pass
            cands.append(Path(settings.CORPUS_DIR) / "corpus.jsonl")

            records: List[Dict[str, str]] = []
            for path in cands:
                records = self._load_corpus(path)
                if records:
                    break
            if not records:
                return "", 0

            kws = [k for k in keywords(question, topk=8) if len(k) >= 2]
            if not kws:
                return "", 0

            scored: List[Tuple[int, Dict[str, str]]] = []
            for rec in records:
                blob = f"{rec.get('title', '')} {rec.get('abstract', '')}"
                hit = sum(1 for k in kws if k in blob)
                if hit > 0:
                    scored.append((hit, rec))
            if not scored:
                return "", 0
            scored.sort(key=lambda kv: -kv[0])

            out: List[str] = []
            for _, rec in scored[:3]:
                snip = self._snippet(rec.get("abstract") or rec.get("title", ""), kws, 200)
                if not snip:
                    continue
                out.append(
                    f"[文献{len(out) + 1}] PMID:{rec.get('pmid', '')} "
                    f"《{truncate(rec.get('title', ''), 60)}》\n{snip}"
                )
            if not out:
                return "", 0
            return "\n".join(out), len(out)
        except Exception as exc:  # noqa: BLE001
            logger.debug("文献片段检索跳过：%s", exc)
            return "", 0


__all__ = ["ContextBuilder", "ContextBundle", "SAFE_KNOWLEDGE"]
