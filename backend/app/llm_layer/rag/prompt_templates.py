# -*- coding: utf-8 -*-
"""
★★★ 创新点 2（加载器）：医疗专属 RAG Prompt 模板库 ★★★
=========================================================
加载 `prompt_library.yaml`（104 个医疗问答模板），提供：
  * 按意图路由模板（`select`）
  * 变量渲染（`render`），自动追加全局硬约束与免责声明
  * 多轮槽位继承
  * 模板统计与校验（启动时自检，缺失字段立刻告警）

设计要点
--------
1. **全局硬约束复用**：`meta.global_constraints` 在渲染时统一追加，
   避免 104 个模板重复书写约束，也保证约束不会被漏掉。
2. **模板选择策略**：意图匹配 → 关键词/条件加权（priority + tags）→ 兜底。
   条件规则支持：
     * `require_entity_any`: 实体中必须包含指定类型（如特殊人群）
     * `keyword_any`: 问题中必须命中关键词
     * `max_len` / `min_len`: 问题长度约束
3. **安全优先**：命中急症/剂量/自伤/药物相互作用等高风险条件时，
   直接提升对应模板优先级，确保安全模板永远胜出。
"""
from __future__ import annotations

import re
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import yaml  # type: ignore
    HAS_YAML = True
except Exception:  # pragma: no cover
    HAS_YAML = False
    yaml = None  # type: ignore


@dataclass
class PromptTemplate:
    """单个医疗 RAG prompt 模板"""

    id: str
    intent: str
    name: str
    template: str
    description: str = ""
    priority: int = 50
    temperature: float = 0.15
    max_tokens: int = 1200
    tags: List[str] = field(default_factory=list)
    version: str = "v1"

    #: 选择条件（可选）
    keyword_any: List[str] = field(default_factory=list)
    require_entity_any: List[str] = field(default_factory=list)
    min_len: int = 0
    max_len: int = 10000

    @property
    def placeholders(self) -> List[str]:
        return sorted(set(re.findall(r"\{(\w+)\}", self.template)))

    def preview(self, n: int = 220) -> str:
        s = re.sub(r"\n{2,}", "\n", self.template.strip())
        return s if len(s) <= n else s[: n - 1] + "…"


#: 仅在该条件成立时才适用的模板 ID 后缀。
#: 这些模板的 priority 很高（110~200），但**不能作为通用默认模板**，
#: 否则会给普通问句套上"孕期/儿童/老年/慢性病/特定人群/澄清追问"的专用话术。
#: 仅当 `_detect_special_crowd` 命中对应人群时才参与选择（见 `select()`）。
CONDITIONAL_TEMPLATE_SUFFIXES: Tuple[str, ...] = (
    "_pregnancy_v1",     # 孕产妇专用
    "_child_v1",         # 儿童专用
    "_elderly_v1",       # 老年专用
    "_chronic_v1",       # 慢性病患者专用
    "_idiographic_v1",   # 特定人群疾病查询
    "_clarify_v1",       # 信息不足时的澄清追问
)

#: 高风险条件 → 强制使用的高优先模板 ID（安全兜底，任何情况下都优先）
SAFETY_OVERRIDES: List[Tuple[str, List[str], str]] = [
    # (模板ID, 触发关键词, 说明)
    ("safety_self_harm_v1",
     ["自杀", "自残", "自伤", "不想活", "活不下去", "结束生命", "轻生", "割腕", "跳楼"],
     "自伤风险 → 危机干预"),
    ("emergency_chest_pain_v1",
     ["胸痛", "胸闷压榨", "心前区疼痛", "胸口疼", "濒死感"],
     "胸痛 → 急症响应"),
    ("emergency_breathing_v1",
     ["呼吸困难", "喘不上气", "无法呼吸", "憋气", "口唇发紫", "窒息"],
     "呼吸困难 → 急症响应"),
    ("emergency_stroke_v1",
     ["口角歪斜", "言语不清", "半身不遂", "偏瘫", "肢体无力", "突然说不出话"],
     "卒中征象 → 急症响应"),
    ("emergency_bleeding_v1",
     ["大出血", "呕血", "咯血", "便血", "血便", "阴道大量流血", "止不住血"],
     "大出血 → 急症响应"),
    ("emergency_poisoning_v1",
     ["中毒", "误服", "喝了农药", "吃了农药", "煤气中毒", "药物过量"],
     "中毒 → 急症响应"),
    ("emergency_high_fever_child_v1",
     ["抽搐", "惊厥", "抽风", "高热惊厥"],
     "惊厥 → 急症响应"),
    ("safety_dose_guard_v1",
     ["吃几片", "吃多少", "多少毫克", "剂量", "一次几粒", "一天几次", "怎么吃这个药", "用量"],
     "剂量索取 → 安全护栏"),
    ("treatment_query_drug_interaction_v1",
     ["一起吃", "同时吃", "相互作用", "能不能同服", "配伍"],
     "药物相互作用 → 安全护栏"),
    ("department_query_emergency_v1",
     ["需要去急诊吗", "要不要去急诊", "要挂急诊吗", "算急症吗"],
     "急诊判定"),
]


class PromptLibrary:
    """
    医疗 RAG prompt 模板库
    ----------------------
    线程内单例（由 `get_prompt_library()` 提供），首次加载时做完整性校验。
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or settings.prompt_library_path)
        self.templates: List[PromptTemplate] = []
        self.by_id: Dict[str, PromptTemplate] = {}
        self.by_intent: Dict[str, List[PromptTemplate]] = {}
        self.global_constraints: str = ""
        self.no_data_hint: str = ""
        self.meta: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            logger.error("医疗 prompt 模板库不存在：%s", self.path)
            return
        if not HAS_YAML:
            logger.error("未安装 PyYAML，无法加载 prompt 模板库（pip install PyYAML）")
            return

        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.meta = data.get("meta") or {}
        self.global_constraints = (self.meta.get("global_constraints") or "").strip()
        self.no_data_hint = (self.meta.get("no_data_hint") or "").strip()

        for item in data.get("templates") or []:
            try:
                tpl = PromptTemplate(
                    id=item["id"],
                    intent=item["intent"],
                    name=item.get("name", item["id"]),
                    template=(item.get("template") or "").strip(),
                    description=item.get("description", ""),
                    priority=int(item.get("priority", 50)),
                    temperature=float(item.get("temperature", 0.15)),
                    max_tokens=int(item.get("max_tokens", 1200)),
                    tags=list(item.get("tags") or []),
                    version=str(item.get("version", "v1")),
                    keyword_any=list(item.get("keyword_any") or []),
                    require_entity_any=list(item.get("require_entity_any") or []),
                    min_len=int(item.get("min_len", 0)),
                    max_len=int(item.get("max_len", 10000)),
                )
            except KeyError as exc:
                logger.error("模板缺少必填字段 %s，已跳过：%r", exc, item.get("id"))
                continue
            if not tpl.template:
                logger.error("模板 %s 正文为空，已跳过", tpl.id)
                continue
            self.templates.append(tpl)
            self.by_id[tpl.id] = tpl
            self.by_intent.setdefault(tpl.intent, []).append(tpl)

        for lst in self.by_intent.values():
            lst.sort(key=lambda t: -t.priority)

        logger.info(
            "医疗 RAG prompt 模板库加载完成：%d 个模板，%d 类意图（%s）",
            len(self.templates), len(self.by_intent),
            ", ".join(f"{k}:{len(v)}" for k, v in sorted(self.by_intent.items())),
        )
        if len(self.templates) < 100:
            logger.warning("模板数 %d < 100，未满足 PPT「100+ 模板」指标", len(self.templates))

    # ==================================================================
    #  模板选择
    # ==================================================================
    def select(
        self,
        intent: str,
        question: str = "",
        entities: Optional[Dict[str, List[str]]] = None,
        slots: Optional[Dict[str, Any]] = None,
        prefer_id: Optional[str] = None,
    ) -> Optional[PromptTemplate]:
        """
        选择最合适的模板。

        优先级（从高到低）：
          1. 安全覆盖规则（SAFETY_OVERRIDES）—— 命中即返回，不受其他规则影响
          2. prefer_id 显式指定
          3. 实体的特殊人群条件（孕妇/儿童/老年）匹配
          4. 意图内 priority 最高的模板
          5. 兜底模板 fallback_no_kg_data_v1
        """
        tpl = self._select_safety(question)
        if tpl:
            logger.info("命中安全覆盖模板：%s", tpl.id)
            return tpl

        if prefer_id and prefer_id in self.by_id:
            return self.by_id[prefer_id]

        candidates = list(self.by_intent.get(intent, []))
        if not candidates:
            logger.warning("意图 %s 无对应模板，使用兜底模板", intent)
            return self.by_id.get("fallback_no_kg_data_v1") or (self.templates[0] if self.templates else None)

        # 特殊人群：从实体与问题中判断
        # 注意：`_detect_special_crowd` 未命中时返回空字符串；
        # 必须显式判空，否则空串会匹配所有模板，把普通问句套上"孕期/儿童"专用模板。
        crowd = self._detect_special_crowd(question, entities)
        if crowd:
            boosted = [t for t in candidates
                       if crowd in t.id or crowd in t.tags or crowd in t.keyword_any]
            if boosted:
                logger.info("命中特殊人群模板：%s（人群=%s）", boosted[0].id, crowd)
                return boosted[0]

        # ------------------------------------------------------------------
        #  ★ 排除「有条件才该用」的专用模板
        #  ------------------------------------------------------------------
        #  问题：special-crowd 与 clarify 类模板的 priority 很高（110~200），
        #        若不过滤，它们会为**任意**问句胜出 —— 例如「原发性高血压有哪些症状？」
        #        曾被套上 symptom_consult_pregnancy_v1（孕期专用），属严重误导。
        #  方案：默认从候选中剔除这些模板；仅当上方 crowd 判定命中时才可能选到它们。
        general = [t for t in candidates
                   if not any(suffix in t.id for suffix in CONDITIONAL_TEMPLATE_SUFFIXES)]
        if general:
            candidates = general

        # 条件匹配（keyword_any / 长度）
        matched = [t for t in candidates if self._match_conditions(t, question)]
        if matched:
            return matched[0]

        return candidates[0]

    # ------------------------------------------------------------------
    def _select_safety(self, question: str) -> Optional[PromptTemplate]:
        q = question or ""
        for tpl_id, keywords, reason in SAFETY_OVERRIDES:
            if any(k in q for k in keywords):
                tpl = self.by_id.get(tpl_id)
                if tpl:
                    logger.info("安全覆盖触发：%s（%s）", tpl_id, reason)
                    return tpl
        return None

    @staticmethod
    def _detect_special_crowd(question: str, entities: Optional[Dict[str, List[str]]]) -> str:
        """
        识别问句中的「特殊人群」信号，返回人群标识（未命中返回空字符串）。

        特殊人群模板（孕期/儿童/老年/慢性病）的约束比通用模板严格得多
        （禁止用药建议、强制转诊专科），因此**只能在真正命中人群时使用**。
        """
        q = question or ""
        if any(k in q for k in ("孕妇", "怀孕", "妊娠", "孕期", "哺乳", "备孕", "待产",
                                "产妇", "产检", "胎儿", "月子")):
            return "pregnancy"
        if any(k in q for k in ("小孩", "孩子", "儿童", "宝宝", "婴儿", "幼儿",
                                "新生儿", "小儿", "宝宝", "婴幼儿", "学龄")):
            return "child"
        if any(k in q for k in ("老人", "老年", "爷爷", "奶奶", "高龄", "养老", "70岁",
                                "80岁", "90岁")):
            return "elderly"
        if any(k in q for k in ("慢性病", "基础病", "糖尿病史", "高血压史", "长期服药",
                                "长期用药", "老毛病", "多年病史")):
            return "chronic"
        return ""

    @staticmethod
    def _match_conditions(tpl: PromptTemplate, question: str) -> bool:
        q = question or ""
        if tpl.min_len and len(q) < tpl.min_len:
            return False
        if tpl.max_len and len(q) > tpl.max_len:
            return False
        if tpl.keyword_any and not any(k in q for k in tpl.keyword_any):
            return False
        return True

    # ==================================================================
    #  模板渲染
    # ==================================================================
    def render(
        self,
        tpl: PromptTemplate,
        variables: Dict[str, Any],
        append_constraints: bool = True,
        append_disclaimer: bool = True,
    ) -> str:
        """
        渲染模板。

        * 未提供的变量统一置为空串（避免 KeyError 中断推理）
        * `append_constraints=True` 时在末尾追加 `meta.global_constraints`
        * `append_disclaimer=True` 时追加统一免责声明
        """
        vars_full: Dict[str, Any] = {
            "question": "", "question_norm": "", "intent_cn": "",
            "entities": "", "main_entity": "", "kg_context": "",
            "kg_triple_count": 0, "safe_knowledge": "",
            "conversation_slots": "（无）", "history": "（无）",
            "department_list": "", "chunk_context": "",
            "user_profile": "", "risk_features": "",
            "no_data_hint": self.no_data_hint,
            "entities_echo": variables.get("main_entity", "该人群"),
        }
        vars_full.update({k: v for k, v in (variables or {}).items() if v is not None})

        body = tpl.template
        # 用 str.replace 而非 str.format，避免模板中 JSON/花括号被误解析
        for key, value in vars_full.items():
            body = body.replace("{" + key + "}", str(value))

        # 清理未替换的可选变量（保留 {question} 之类若确实未提供则清空）
        left = set(re.findall(r"\{(\w+)\}", body))
        for key in left:
            body = body.replace("{" + key + "}", "")

        parts = [body.rstrip()]
        if append_constraints and self.global_constraints:
            parts.append("\n\n" + self.global_constraints)
        if append_disclaimer:
            parts.append("\n\n<|输出末尾必须包含|>\n本回答由 AI 生成，仅供参考，不能替代执业医师诊断。")
        return "\n".join(parts)

    # ==================================================================
    #  查询与统计
    # ==================================================================
    def get(self, tpl_id: str) -> Optional[PromptTemplate]:
        return self.by_id.get(tpl_id)

    def list_meta(self, intent: Optional[str] = None) -> List[Dict[str, Any]]:
        items = self.by_intent.get(intent, []) if intent else self.templates
        return [
            {
                "id": t.id, "intent": t.intent, "name": t.name,
                "description": t.description, "version": t.version,
                "tags": t.tags, "template_preview": t.preview(),
            }
            for t in items
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "total": len(self.templates),
            "intents": sorted(self.by_intent.keys()),
            "by_intent": {k: len(v) for k, v in sorted(self.by_intent.items())},
            "has_global_constraints": bool(self.global_constraints),
            "safety_override_count": len(SAFETY_OVERRIDES),
            "path": str(self.path),
        }

    # ==================================================================
    #  领域判定（服务边界）
    # ==================================================================
    #: 医疗领域强特征词（出现任一即判定为医疗问题）
    #: 说明：这是一份**高召回**词表 —— 宁可把边缘医疗问题判为医疗（走知识库检索），
    #:       也不要把医疗问题误判为超范围（那会直接拒绝帮助患者）。
    MEDICAL_TERMS: Tuple[str, ...] = (
        # 症状与体征
        "症状", "疼痛", "发热", "发烧", "咳嗽", "头晕", "头痛", "恶心", "呕吐", "腹泻",
        "便秘", "乏力", "水肿", "皮疹", "瘙痒", "麻木", "抽搐", "黄疸", "心悸", "胸闷",
        "胸痛", "腹痛", "气促", "呼吸困难", "咯血", "呕血", "便血", "盗汗", "消瘦",
        "失眠", "晕厥", "出血", "肿大", "结节", "溃疡", "发热", "低烧", "低热", "高烧",
        # 诊疗行为
        "治疗", "用药", "吃药", "服药", "药", "手术", "化疗", "放疗", "检查", "化验",
        "诊断", "确诊", "复查", "随访", "挂号", "就诊", "住院", "门诊", "急诊", "科室",
        "针灸", "理疗", "康复", "疫苗", "接种", "输液", "打针", "输液",
        "病因", "预后", "并发症", "鉴别", "传染", "遗传", "复发", "转移",
        # 疾病与病理
        "病", "症", "炎", "癌", "瘤", "综合症", "感染", "硬化", "梗死", "栓塞", "衰竭",
        "障碍", "畸形", "损伤", "结石", "囊肿", "息肉", "贫血", "亢进", "减退", "中毒",
        "血压", "血糖", "血脂", "心率", "脉搏", "体温", "胆固醇", "甘油三酯", "尿酸",
        "肝功", "肾功", "白细胞", "血红蛋白", "血小板", "肌酐", "转氨酶",
        # 身体部位与人群
        "心脏", "肝脏", "肾脏", "肺部", "胃", "肠", "脑", "血管", "关节", "脊柱",
        "孕妇", "怀孕", "妊娠", "哺乳", "婴儿", "幼儿", "儿童", "小孩", "老年", "老人",
        # 药理与健康管理
        "剂量", "副作用", "不良反应", "过敏", "禁忌", "保健品", "维生素", "抗生素",
        "消炎药", "降压药", "降糖药", "中药", "西药", "偏方",
        "饮食", "运动", "减肥", "戒烟", "戒酒", "体检", "筛查", "预防", "养生", "免疫力",
        "健康", "医院", "医生", "医师", "护士", "病情", "病历", "病史", "体质",
    )

    @classmethod
    def is_medical_query(cls, question: str) -> bool:
        """
        判断问句是否属于医疗健康领域。

        用途：当**既无实体链接结果、又无图谱事实召回**时，用本方法区分
        「知识库未收录的医疗问题」与「完全与医疗无关的问题」，从而给出
        不同的响应（前者提示换词重试，后者礼貌拒答）。

        判定规则（宽松优先，避免误拒医疗问题）：
          1. 命中任一医疗特征词 → 医疗问题
          2. 命中医疗疑问句式（挂什么科/怎么治/吃什么药/严重吗…）→ 医疗问题
          3. 其余 → 非医疗问题
        """
        q = (question or "").strip()
        if not q:
            return False
        if any(t in q for t in cls.MEDICAL_TERMS):
            return True
        # 医疗疑问句式（有些问句不含上述实体词，但明显是就医咨询）
        return bool(re.search(
            r"(挂(什么|哪个|哪)科|看(什么|哪个|哪)科|怎么(治|办)|如何(治|预防)|"
            r"吃什么药|用(什么|哪些)药|能(治好|治愈|根治)|严重吗|要紧吗|危险吗|"
            r"会不会(传染|遗传|有事)|需要(手术|住院|检查)|"
            r"挂[内外妇儿皮眼口耳]|看[内外妇儿皮眼口耳])",
            q,
        ))

    def validate(self) -> Dict[str, Any]:
        """启动自检：检查关键模板是否齐全（缺失会显著影响问答质量）"""
        required = [
            "disease_query_core_v1", "symptom_consult_core_v1",
            "treatment_query_core_v1", "department_query_core_v1",
            "emergency_core_v1", "safety_dose_guard_v1",
            "risk_assessment_core_v1", "fallback_no_kg_data_v1",
        ]
        missing = [r for r in required if r not in self.by_id]
        placeholders_used = set()
        for t in self.templates:
            placeholders_used.update(t.placeholders)
        return {
            "total": len(self.templates),
            "missing_required": missing,
            "placeholders_used": sorted(placeholders_used),
            "ok": not missing and len(self.templates) >= 100,
        }


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_library: Optional[PromptLibrary] = None


def get_prompt_library() -> PromptLibrary:
    global _library
    if _library is None:
        _library = PromptLibrary()
    return _library
