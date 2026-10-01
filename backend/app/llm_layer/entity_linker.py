# -*- coding: utf-8 -*-
"""
实体链接（Entity Linking）
==========================
把用户自然语言问句中的**表面提及（mention）** 对齐到知识图谱的**唯一节点**。

流程
----
    问句
     ├─(1) 规则/词典候选生成   —— MedicalKeywordMatcher + 图谱实体词典最长匹配
     ├─(2) BERT-NER 补充召回    —— BERT-BiLSTM-CRF 识别词典未覆盖的实体
     ├─(3) 证据打分             —— 名称精确 / 别名 / 类型相容 / 上下文共现
     ├─(4) 消歧                 —— 同类型多候选时按上下文与度数选择
     └─(5) 输出 LinkedEntity 列表（含 kg_id / kg_name / score / method）

为何需要实体链接：医疗问句大量使用口语化简称（"血压高"、"血糖"、"心脏病"），
必须先归一化到图谱标准名（"原发性高血压"、"2型糖尿病"、"冠心病"），
后续 Cypher 查询与 RAG 检索才能命中。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.kg_layer.entity_extractor import EntityExtractor, EntityMention, get_entity_extractor
from app.kg_layer.memory_store import MemoryGraphStore
from app.schemas import ENTITY_TYPE_META
from app.utils.logger import get_logger
from app.utils.text import normalize, similarity

logger = get_logger(__name__)


# =============================================================================
#  口语化别名词表（医疗简称 → 图谱标准名候选）
# =============================================================================
#: 用户常用口语 → 标准疾病名（实体链接的强先验）
COLLOQUIAL_MAP: Dict[str, str] = {
    # 注意：「高血压」「糖尿病」等**泛指词**必须显式映射到最常见的具体病种，
    # 否则词典会随机链接到「假性高血压」「妊娠高血压疾病」等少见条目（严重误导）。
    # 更具体的名称（如「妊娠高血压疾病」）由词典最长匹配优先命中，不受此影响。
    "血压高": "原发性高血压",
    "血压偏高": "原发性高血压",
    "高血压病": "原发性高血压",
    "高血压": "原发性高血压",
    "血糖高": "2型糖尿病",
    "糖尿病": "2型糖尿病",
    "心脏病": "冠心病",
    "冠心病": "冠心病",
    "心梗": "冠心病",
    "中风": "脑卒中",
    "脑梗": "脑卒中",
    "脑出血": "脑卒中",
    "肺栓塞": "肺栓塞",
    "肺梗": "肺栓塞",
    "慢阻肺": "慢性阻塞性肺疾病",
    "copd": "慢性阻塞性肺疾病",
    "肺心病": "肺源性心脏病",
    "呼衰": "急性呼吸窘迫综合征",
    "ards": "急性呼吸窘迫综合征",
    "老年痴呆": "阿尔茨海默病",
    "帕金森": "帕金森病",
    "甲亢": "甲状腺功能亢进症",
    "贫血": "缺铁性贫血",
    "胃溃疡": "消化性溃疡",
    "胃炎": "急性单纯性胃炎",
    "胃病": "急性单纯性胃炎",
    "脂肪肝": "脂肪肝",
    "痛风": "痛风",
    "类风湿": "类风湿关节炎",
    "红斑狼疮": "系统性红斑狼疮",
    "荨麻疹": "荨麻疹",
    "阑尾炎": "急性阑尾炎",
    "胆结石": "胆囊结石",
    "腰间盘突出": "腰椎间盘突出症",
    "腰椎间盘突出": "腰椎间盘突出症",
    "新冠": "新型冠状病毒感染",
    "新冠肺炎": "新型冠状病毒感染",
    "流感": "流行性感冒",
    "感冒": "感冒",
    "肺结核": "肺结核",
    "乙肝": "病毒性肝炎",
    "丙肝": "病毒性肝炎",
    "睡眠呼吸暂停": "睡眠呼吸暂停综合征",
    "打呼噜": "睡眠呼吸暂停综合征",
    "抑郁": "抑郁症",
    "焦虑": "焦虑症",
    "哮喘": "支气管哮喘",
    "尿路感染": "尿路感染",
    "尿感": "尿路感染",
    "肾病": "慢性肾脏病",
    "慢性肾病": "慢性肾脏病",
    "呼吸衰竭": "急性呼吸窘迫综合征",
    "肺动脉高压": "继发性肺动脉高压",
}

#: 科室口语 → 标准科室名
DEPARTMENT_ALIAS: Dict[str, str] = {
    "心内科": "心血管内科",
    "心血管科": "心血管内科",
    "心脏科": "心血管内科",
    "呼吸科": "呼吸内科",
    "消化科": "消化内科",
    "内分泌": "内分泌科",
    "神经科": "神经内科",
    "肾科": "肾内科",
    "血液科": "血液内科",
    "风湿科": "风湿免疫科",
    "传染科": "感染性疾病科",
    "感染科": "感染性疾病科",
    "普外科": "普通外科",
    "泌尿科": "泌尿外科",
    "脑外科": "神经外科",
    "妇科": "妇科",
    "产科": "产科",
    "儿科": "小儿内科",
    "小儿科": "小儿内科",
    "皮肤科": "皮肤科",
    "五官科": "耳鼻咽喉科",
    "耳鼻喉科": "耳鼻咽喉科",
    "精神科": "精神心理科",
    "心理科": "精神心理科",
    "肿瘤科": "肿瘤科",
    "急诊": "急诊科",
    "重症科": "重症医学科",
}

#: 症状口语 → 标准症状名
SYMPTOM_ALIAS: Dict[str, str] = {
    "发烧": "发热",
    "发高烧": "高热",
    "低烧": "低热",
    "喘不上气": "呼吸困难",
    "气短": "呼吸困难",
    "憋气": "呼吸困难",
    "心慌": "心悸",
    "胸口疼": "胸痛",
    "胸口痛": "胸痛",
    "头疼": "头痛",
    "晕": "头晕",
    "眩晕": "眩晕",
    "恶心想吐": "恶心",
    "拉肚子": "腹泻",
    "肚子疼": "腹痛",
    "肚子痛": "腹痛",
    "呕血": "呕血",
    "咳血": "咯血",
    "没力气": "乏力",
    "浑身没劲": "乏力",
    "消瘦": "体重下降",
    "瘦了": "体重下降",
    "出汗": "盗汗",
    "睡不着": "失眠",
    "水肿": "水肿",
    "浮肿": "水肿",
    "皮肤痒": "瘙痒",
    "起疹子": "皮疹",
    "手脚麻": "麻木",
    "抽风": "抽搐",
    "晕倒": "晕厥",
    "黄疸": "黄疸",
    "便秘": "便秘",
}


@dataclass
class LinkCandidate:
    """候选链接"""

    text: str
    type: str
    kg_id: str
    kg_name: str
    score: float
    method: str
    start: int = 0
    end: int = 0
    evidences: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.evidences is None:
            self.evidences = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "type": self.type,
            "label": ENTITY_TYPE_META.get(self.type, {}).get("label", self.type),
            "kg_id": self.kg_id, "kg_name": self.kg_name,
            "score": round(self.score, 4), "method": self.method,
            "start": self.start, "end": self.end,
        }


class EntityLinker:
    """
    实体链接器（统一门面）
    """

    def __init__(self, extractor: Optional[EntityExtractor] = None) -> None:
        self.extractor = extractor or get_entity_extractor()
        self.store = MemoryGraphStore.instance()
        self._build_alias_index()

    # ------------------------------------------------------------------
    def _build_alias_index(self) -> None:
        """构建别名索引：口语 → 标准名（仅保留图谱中真实存在的目标）"""
        self.alias_index: Dict[str, Tuple[str, str]] = {}   # 口语 -> (类型, 标准名)

        def put(alias: str, ntype: str, canon: str) -> None:
            if not alias or alias == canon:
                return
            if (ntype, canon) not in self.store.index:
                return
            self.alias_index.setdefault(alias, (ntype, canon))

        for alias, canon in COLLOQUIAL_MAP.items():
            put(alias, "Disease", canon)
        for alias, canon in DEPARTMENT_ALIAS.items():
            put(alias, "Department", canon)
        for alias, canon in SYMPTOM_ALIAS.items():
            put(alias, "Symptom", canon)

        logger.info("实体链接别名词典构建完成：%d 条（口语 → 标准名）", len(self.alias_index))

    # ==================================================================
    #  主流程
    # ==================================================================
    def analyze(self, text: str, max_entities: int = 12) -> Dict[str, Any]:
        """
        识别 + 链接一体化。

        返回：
            {
              "backend": "dictionary" | "bert_bilstm_crf",
              "mentions": [LinkedEntity dict, ...],
              "diseases": [标准名, ...],
              "symptoms": [...], "departments": [...],
              "drugs": [...], "checks": [...], "treatments": [...],
              "main_entity": 主实体标准名,
              "colloquial_hits": {口语: 标准名, ...},
            }
        """
        text = normalize(text or "")
        if not text:
            return self._empty()

        candidates: List[LinkCandidate] = []

        # ---- (1) 模型/词典识别 ----
        mentions: List[EntityMention] = self.extractor.recognize(text)
        linked = self.extractor.link_to_kg(mentions)
        for m in linked:
            candidates.append(LinkCandidate(
                text=m.text, type=m.type,
                kg_id=m.kg_id or self.store.index.get((m.type, m.kg_name), ""),
                kg_name=m.kg_name or m.text,
                score=m.score, method=m.method,
                start=m.start, end=m.end,
                evidences=["图谱实体词典/模型识别"],
            ))

        # ---- (2) 口语别名匹配（强先验，补充词典漏召）----
        for alias, (ntype, canon) in self.alias_index.items():
            start = text.find(alias)
            if start < 0:
                continue
            # 与已识别的区间重叠则跳过（避免重复）
            if any(not (start + len(alias) <= c.start or start >= c.end) for c in candidates):
                continue
            candidates.append(LinkCandidate(
                text=alias, type=ntype,
                kg_id=self.store.index.get((ntype, canon), ""),
                kg_name=canon,
                score=0.9, method="colloquial_alias",
                start=start, end=start + len(alias),
                evidences=[f"口语别名「{alias}」→ 标准名「{canon}」"],
            ))

        # ---- (3) 模糊补充：对未命中的长度≥3 的片段做相似度匹配 ----
        if len(candidates) < 3:
            candidates += self._fuzzy_candidates(text, candidates)

        # ---- (4) 消歧 + 类型感知重排 ----
        final = self._disambiguate(candidates, text, max_entities)

        # ---- (5) 聚合输出 ----
        out: Dict[str, Any] = {
            "backend": self.extractor.backend,
            "mentions": [c.to_dict() for c in final],
            "diseases": self._names(final, "Disease"),
            "symptoms": self._names(final, "Symptom"),
            "departments": self._names(final, "Department"),
            "drugs": self._names(final, "Drug"),
            "checks": self._names(final, "Check"),
            "treatments": self._names(final, "Treatment"),
            "colloquial_hits": {c.text: c.kg_name for c in final if c.method == "colloquial_alias"},
        }
        # 主实体：优先疾病 → 症状 → 科室
        for t in ("Disease", "Symptom", "Department", "Drug"):
            names = out[{"Disease": "diseases", "Symptom": "symptoms",
                         "Department": "departments", "Drug": "drugs"}[t]]
            if names:
                out["main_entity"] = names[0]
                break
        else:
            out["main_entity"] = final[0].kg_name if final else ""
        return out

    # ------------------------------------------------------------------
    def _fuzzy_candidates(self, text: str, existing: List[LinkCandidate]) -> List[LinkCandidate]:
        """
        对问句中的滑动窗口片段做模糊匹配（窗口长度 2-8）。
        医疗问句常出现图谱名称的子串（如"肥胖性心肌病"中的"心肌病"），
        滑窗模糊匹配可显著提升召回。
        """
        out: List[LinkCandidate] = []
        n = len(text)
        taken = [(c.start, c.end) for c in existing]
        for L in (8, 7, 6, 5, 4, 3, 2):
            for i in range(0, n - L + 1):
                if any(not (i + L <= s or i >= e) for s, e in taken):
                    continue
                frag = text[i : i + L]
                if not re.search(r"[\u4e00-\u9fff]", frag):
                    continue
                best_name, best_type, best_score = "", "", 0.0
                for (ntype, name) in self.store.index.keys():
                    s = similarity(frag, name)
                    if s > best_score:
                        best_name, best_type, best_score = name, ntype, s
                if best_score >= 0.72:
                    out.append(LinkCandidate(
                        text=frag, type=best_type,
                        kg_id=self.store.index.get((best_type, best_name), ""),
                        kg_name=best_name, score=round(best_score * 0.85, 4),
                        method="fuzzy", start=i, end=i + L,
                        evidences=[f"滑窗模糊匹配「{frag}」≈「{best_name}」(相似度 {best_score:.2f})"],
                    ))
                    taken.append((i, i + L))
                    break
        return out

    # ------------------------------------------------------------------
    def _disambiguate(
        self, candidates: List[LinkCandidate], text: str, max_entities: int
    ) -> List[LinkCandidate]:
        """
        消歧与重排：
          ① 同一区间多个候选 → 取分数最高
          ② 同一标准名多类型 → 保留图谱中真实存在的类型
          ③ 上下文加权：实体在问句中的位置（越靠前越可能是主题）+ 图谱度数
        """
        # ① 区间去重
        by_span: Dict[Tuple[int, int], LinkCandidate] = {}
        for c in candidates:
            key = (c.start, c.end)
            if key not in by_span or by_span[key].score < c.score:
                by_span[key] = c
        uniq = list(by_span.values())

        # ② 同标准名去重（保留分数高的）
        by_name: Dict[Tuple[str, str], LinkCandidate] = {}
        for c in uniq:
            key = (c.type, c.kg_name)
            if key not in by_name or by_name[key].score < c.score:
                by_name[key] = c
        uniq = list(by_name.values())

        # ③ 上下文加权
        for c in uniq:
            node = self.store.nodes.get(c.kg_id) if c.kg_id else None
            degree_bonus = min(0.06, (node["degree"] if node else 0) / 400.0)
            position_bonus = 0.05 if c.start <= len(text) * 0.5 else 0.0
            type_bonus = {"Disease": 0.04, "Symptom": 0.02,
                          "Department": 0.01, "Drug": 0.01}.get(c.type, 0.0)
            c.score = min(0.999, c.score + degree_bonus + position_bonus + type_bonus)
            if node:
                c.evidences.append(
                    f"图谱节点度数 {node['degree']}，类型 {c.type}"
                )

        uniq.sort(key=lambda x: (-x.score, x.start))
        return uniq[:max_entities]

    # ------------------------------------------------------------------
    @staticmethod
    def _names(cands: Sequence[LinkCandidate], ntype: str) -> List[str]:
        seen, out = set(), []
        for c in cands:
            if c.type == ntype and c.kg_name not in seen:
                seen.add(c.kg_name)
                out.append(c.kg_name)
        return out

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {
            "backend": "none", "mentions": [],
            "diseases": [], "symptoms": [], "departments": [],
            "drugs": [], "checks": [], "treatments": [],
            "main_entity": "", "colloquial_hits": {},
        }

    # ==================================================================
    #  辅助：Cypher 参数构建
    # ==================================================================
    def to_cypher_params(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """把链接结果转成 Cypher 查询参数（供 cypher_generator 使用）"""
        return {
            "diseases": analysis.get("diseases") or [],
            "symptoms": analysis.get("symptoms") or [],
            "departments": analysis.get("departments") or [],
            "drugs": analysis.get("drugs") or [],
            "checks": analysis.get("checks") or [],
            "main": analysis.get("main_entity") or "",
            "names": [m.get("kg_name") for m in analysis.get("mentions") or [] if m.get("kg_name")],
        }

    def info(self) -> Dict[str, Any]:
        return {
            "extractor_backend": self.extractor.backend,
            "alias_count": len(self.alias_index),
            "colloquial_map_size": len(COLLOQUIAL_MAP),
            "department_alias_size": len(DEPARTMENT_ALIAS),
            "symptom_alias_size": len(SYMPTOM_ALIAS),
            "lexicon_size": len(self.store.index),
        }


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_linker: Optional[EntityLinker] = None


def get_entity_linker() -> EntityLinker:
    global _linker
    if _linker is None:
        _linker = EntityLinker()
    return _linker
