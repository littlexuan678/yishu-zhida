# -*- coding: utf-8 -*-
"""
医数智答 · 智愈医典 —— 全局 Pydantic 数据契约
================================================
本文件是【前后端唯一数据真源】：
后端所有 API 的请求/响应模型均在此定义，前端 `src/api/*.js` 的字段与此严格一一对应。

命名规范：
  * 请求模型后缀 `Request`
  * 响应模型后缀 `Response`
  * 枚举使用 `str, Enum`，序列化后即为字符串，便于前端直接使用
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# =============================================================================
#  通用 / 枚举
# =============================================================================


class EntityType(str, Enum):
    """知识图谱实体类型（与前端图例配色一一对应）"""

    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    DRUG = "Drug"
    DEPARTMENT = "Department"
    TREATMENT = "Treatment"
    CHECK = "Check"
    POPULATION = "Population"
    SOURCE = "Source"


#: 实体类型 → 中文名 + 前端配色（与 D3 图例 / ECharts 色板共用）
ENTITY_TYPE_META: Dict[str, Dict[str, str]] = {
    "Disease":    {"label": "疾病",     "color": "#409EFF", "icon": "FirstAidKit"},
    "Symptom":    {"label": "症状",     "color": "#31c48d", "icon": "Warning"},
    "Department": {"label": "科室",     "color": "#F56C6C", "icon": "OfficeBuilding"},
    "Drug":       {"label": "药物",     "color": "#E6A23C", "icon": "Coin"},
    "Treatment":  {"label": "治疗方法", "color": "#b37feb", "icon": "MagicStick"},
    "Check":      {"label": "检查项目", "color": "#909399", "icon": "DocumentChecked"},
    "Population": {"label": "易感人群", "color": "#F2C037", "icon": "User"},
    "Source":     {"label": "文献来源", "color": "#00BCD4", "icon": "Link"},
}


class IntentLabel(str, Enum):
    """TextCNN 意图分类的 4 大类别（PPT 指定）"""

    DISEASE_QUERY = "disease_query"        # 疾病查询：高血压是什么病
    SYMPTOM_CONSULT = "symptom_consult"    # 症状咨询：胸痛伴低烧可能是什么
    TREATMENT_QUERY = "treatment_query"    # 治疗/用药咨询：糖尿病怎么治
    DEPARTMENT_QUERY = "department_query"  # 科室导诊：肺栓塞挂什么科


class RiskLevel(str, Enum):
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    VERY_HIGH = "极高风险"


class ApiResponse(BaseModel):
    """统一响应包装（成功）"""

    code: int = 0
    message: str = "ok"
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    code: int = 1
    message: str
    detail: Optional[str] = None


# =============================================================================
#  知识图谱：图数据（D3.js 直接渲染）
# =============================================================================


class GraphNode(BaseModel):
    """D3 力导向图节点"""

    id: str = Field(..., description="节点唯一 ID，如 'Disease:原发性高血压'")
    name: str = Field(..., description="实体显示名")
    type: str = Field(..., description="实体类型，见 EntityType")
    label: str = Field("", description="实体类型中文名（前端图例用）")
    color: str = Field("#409EFF", description="节点填充色")
    degree: int = Field(0, description="节点度数，用于决定半径")
    category1: Optional[str] = Field(None, description="一级分类（疾病）")
    category2: Optional[str] = Field(None, description="二级分类（疾病）")
    properties: Dict[str, Any] = Field(default_factory=dict, description="其余属性")


class GraphLink(BaseModel):
    """D3 力导向图边"""

    source: str = Field(..., description="起点节点 id")
    target: str = Field(..., description="终点节点 id")
    rel: str = Field(..., description="关系英文类型，如 HAS_SYMPTOM")
    label: str = Field("", description="关系中文名，如 '症状'")
    properties: Dict[str, Any] = Field(default_factory=dict)


class SubGraphResponse(BaseModel):
    """子图查询响应（知识图谱可视化模块）"""

    nodes: List[GraphNode] = Field(default_factory=list)
    links: List[GraphLink] = Field(default_factory=list)
    node_count: int = 0
    link_count: int = 0
    type_count: int = 0
    center: Optional[str] = Field(None, description="中心实体名")
    depth: int = 1
    truncated: bool = Field(False, description="是否因 limit 截断")
    message: str = ""


class EntityTypeMeta(BaseModel):
    type: str
    label: str
    color: str
    count: int = 0


class GraphStatsResponse(BaseModel):
    """图谱统计（PPT 截图：125 节点 / 200 关系 / 12 实体类型）"""

    total_nodes: int = 0
    total_links: int = 0
    entity_type_count: int = 0
    entity_types: List[EntityTypeMeta] = Field(default_factory=list)
    disease_count: int = 0
    symptom_count: int = 0
    drug_count: int = 0
    department_count: int = 0
    treatment_count: int = 0
    source_count: int = 0
    data_source: str = Field("neo4j", description="neo4j | memory(降级)")


class EntitySearchItem(BaseModel):
    id: str
    name: str
    type: str
    label: str
    color: str
    score: float = 1.0
    category1: Optional[str] = None


class EntitySearchResponse(BaseModel):
    keyword: str
    items: List[EntitySearchItem] = Field(default_factory=list)
    total: int = 0


class CypherRequest(BaseModel):
    """受控 Cypher 查询（白名单只读校验）"""

    cypher: str
    params: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(200, ge=1, le=1000)

    @field_validator("cypher")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("cypher 语句不能为空")
        return v.strip()


class CypherResponse(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class EvidenceItem(BaseModel):
    """关系溯源证据（PPT 要求：知识来源全部可追溯）"""

    triple_id: str = Field(..., description="引用编号，如 KG-1")
    head: str
    head_type: str
    relation: str
    relation_label: str
    tail: str
    tail_type: str
    confidence: float = 1.0
    sources: List["SourceItem"] = Field(default_factory=list)


class SourceItem(BaseModel):
    """文献来源（PubMed / 指南 / 教材）"""

    source_type: str = Field("pubmed", description="pubmed | guideline | textbook | website")
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    journal: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[str] = None
    url: Optional[str] = None
    authority: Optional[str] = Field(None, description="权威等级：A(指南/教材) B(期刊) C(其他)")
    retrieved_at: Optional[str] = None


# =============================================================================
#  ② 疾病查询（掌上医典）
# =============================================================================


class DiseaseCard(BaseModel):
    """
    疾病卡片（PPT 截图字段）：
      疾病名 / 一级分类 / 二级分类 / 症状 / 治疗 / 易感人群
    """

    disease_id: str
    name: str
    category1: Optional[str] = Field(None, description="一级分类，如 内科")
    category2: Optional[str] = Field(None, description="二级分类，如 心血管内科")
    symptoms: List[str] = Field(default_factory=list)
    treatments: List[str] = Field(default_factory=list)
    population: Optional[str] = Field(None, description="易感人群，如 '多见于老年、尿毒症患者'")
    is_infectious: bool = False
    has_detail: bool = True


class DiseaseSearchResponse(BaseModel):
    keyword: str
    total: int = 0
    page: int = 1
    page_size: int = 6
    items: List[DiseaseCard] = Field(default_factory=list)
    hot_keywords: List[str] = Field(default_factory=list)
    message: str = ""
    searched_at: str = ""


class DiseaseDetail(BaseModel):
    """
    疾病详情（PPT 要求的 6 大结构化板块）
    """

    disease_id: str
    name: str
    alias: List[str] = Field(default_factory=list)
    category1: Optional[str] = None
    category2: Optional[str] = None
    definition: str = Field("", description="疾病定义")
    cause: str = Field("", description="病因")
    symptoms: List[str] = Field(default_factory=list, description="症状")
    diagnosis: str = Field("", description="诊断")
    checks: List[str] = Field(default_factory=list, description="检查项目")
    treatment: str = Field("", description="治疗方案")
    treatments: List[str] = Field(default_factory=list)
    drugs: List[str] = Field(default_factory=list, description="常用药物")
    department: Optional[str] = Field(None, description="就诊科室")
    prognosis: str = Field("", description="预后")
    population: Optional[str] = None
    complications: List[str] = Field(default_factory=list)
    differential: List[str] = Field(default_factory=list)
    is_infectious: bool = False
    sources: List[SourceItem] = Field(default_factory=list, description="权威来源溯源")
    updated_at: Optional[str] = None


class RelatedDiseaseResponse(BaseModel):
    disease_id: str
    complications: List[DiseaseCard] = Field(default_factory=list)
    differential: List[DiseaseCard] = Field(default_factory=list)


# =============================================================================
#  ③ 智能问答（可解释 AI 医生）
# =============================================================================


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户自然语言医疗提问")
    session_id: str = Field("default", description="会话 ID，用于多轮槽位继承")
    top_k: int = Field(12, ge=1, le=50, description="RAG 检索三元组条数")
    max_hops: int = Field(2, ge=1, le=3, description="图谱多跳推理深度")
    explain: bool = Field(True, description="是否返回推理链路（可解释性）")
    use_llm: bool = Field(True, description="false 则仅返回 KG 结构化答案")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("问题不能为空")
        return v


class IntentResult(BaseModel):
    label: IntentLabel = IntentLabel.DISEASE_QUERY
    label_cn: str = ""
    confidence: float = 0.0
    method: str = Field("textcnn", description="textcnn | rule_fallback")
    scores: Dict[str, float] = Field(default_factory=dict, description="各类别概率分布")


class LinkedEntity(BaseModel):
    """实体链接结果：mention → 图谱节点"""

    text: str = Field(..., description="问句中的原文提及")
    type: str = Field(..., description="实体类型")
    label: str = ""
    kg_id: str = Field("", description="图谱节点 ID")
    kg_name: str = Field("", description="图谱标准名")
    score: float = 0.0
    method: str = Field("dictionary", description="bert_ner | dictionary | fuzzy")
    start: int = 0
    end: int = 0


class ReasoningStep(BaseModel):
    """推理链路单步（可解释性核心）"""

    step: int
    action: str = Field(..., description="intent_classify | entity_link | cypher_query | kg_retrieve | rag_generate | guard_check")
    title: str = Field("", description="中文标题，前端时间轴展示")
    detail: str = Field("", description="详细信息")
    elapsed_ms: float = 0.0
    payload: Dict[str, Any] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    intent: float = 0.0
    entity_link: float = 0.0
    kg_query: float = 0.0
    semantic_parse_total: float = Field(0.0, description="语义解析总耗时（PPT 指标 ≤500ms）")
    retrieve: float = 0.0
    llm: float = 0.0
    guard: float = 0.0
    total: float = 0.0


class QAResponse(BaseModel):
    question: str
    answer: str
    answer_html: str = Field("", description="带引用角标的富文本（前端 v-html）")
    intent: IntentResult
    entities: List[LinkedEntity] = Field(default_factory=list)
    evidences: List[EvidenceItem] = Field(default_factory=list, description="KG 三元组溯源证据")
    kg_context: str = Field("", description="喂给大模型的三元组上下文（可解释性展示）")
    reasoning_trace: List[ReasoningStep] = Field(default_factory=list)
    confidence: float = Field(0.0, description="答案置信度 0~1")
    prompt_template_id: str = Field("", description="命中的医疗 prompt 模板 ID")
    prompt_used: str = Field("", description="实际使用的完整 prompt（调试/答辩展示）")
    guard_result: Dict[str, Any] = Field(default_factory=dict, description="幻觉守卫校验结果")
    latency_ms: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    llm_model: str = ""
    session_id: str = "default"
    disclaimer: str = "本回答由 AI 生成，仅供参考，不能替代执业医师诊断。"
    related_questions: List[str] = Field(default_factory=list, description="推荐追问")


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str
    timestamp: str = ""
    answer_html: str = ""
    evidences: List[EvidenceItem] = Field(default_factory=list)
    reasoning_trace: List[ReasoningStep] = Field(default_factory=list)
    confidence: float = 0.0


class QAHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    turns: int = 0


class PromptTemplateItem(BaseModel):
    """医疗 RAG prompt 模板元信息"""

    id: str
    intent: str
    name: str
    description: str = ""
    version: str = "v1"
    tags: List[str] = Field(default_factory=list)
    template_preview: str = ""


class PromptTemplateListResponse(BaseModel):
    total: int = 0
    intents: List[str] = Field(default_factory=list)
    items: List[PromptTemplateItem] = Field(default_factory=list)


# =============================================================================
#  ④ 数据分析 & 健康预警
# =============================================================================


class OverviewMetric(BaseModel):
    key: str
    label: str
    value: float
    suffix: str = ""
    icon: str = "DataLine"
    trend: Optional[float] = None


class AnalyticsOverviewResponse(BaseModel):
    """PPT 截图：医疗实体总数 / 疾病分类数 / 传染性疾病 / 治疗周期类型"""

    metrics: List[OverviewMetric] = Field(default_factory=list)
    generated_at: str = ""
    data_source: str = "neo4j"


class PieSeriesItem(BaseModel):
    name: str
    value: float
    color: Optional[str] = None


class ChartResponse(BaseModel):
    """通用图表响应（PyEcharts 服务端渲染 + ECharts option 双份）"""

    chart_type: str = Field("pie", description="pie | bar | line | donut | radar | gauge")
    title: str = ""
    series: List[PieSeriesItem] = Field(default_factory=list)
    x_axis: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    values: List[float] = Field(default_factory=list)
    unit: str = ""
    echarts_option: Dict[str, Any] = Field(default_factory=dict, description="前端 ECharts 直接可用")
    pyecharts_html: str = Field("", description="PyEcharts 服务端渲染 HTML 片段")
    insight: str = Field("", description="自动生成的图表洞察结论")


class RiskPredictRequest(BaseModel):
    """脱敏病史输入（不含任何可识别个人身份的信息）"""

    age: int = Field(..., ge=0, le=120)
    gender: str = Field("male", description="male | female")
    bmi: Optional[float] = Field(None, ge=10, le=60)
    systolic_bp: Optional[int] = Field(None, ge=60, le=260, description="收缩压 mmHg")
    diastolic_bp: Optional[int] = Field(None, ge=30, le=180, description="舒张压 mmHg")
    fasting_glucose: Optional[float] = Field(None, ge=1.0, le=40.0, description="空腹血糖 mmol/L")
    total_cholesterol: Optional[float] = Field(None, ge=1.0, le=20.0, description="总胆固醇 mmol/L")
    hdl: Optional[float] = Field(None, ge=0.1, le=5.0)
    ldl: Optional[float] = Field(None, ge=0.1, le=12.0)
    triglycerides: Optional[float] = Field(None, ge=0.1, le=30.0)
    heart_rate: Optional[int] = Field(None, ge=30, le=220)
    smoking: bool = False
    drinking: bool = False
    physical_activity: str = Field("moderate", description="low | moderate | high")
    family_history: List[str] = Field(default_factory=list, description="家族史疾病名")
    symptoms: List[str] = Field(default_factory=list, description="当前症状")
    existing_conditions: List[str] = Field(default_factory=list, description="既往史")

    @field_validator("gender")
    @classmethod
    def _gender(cls, v: str) -> str:
        if v.lower() not in ("male", "female"):
            raise ValueError("gender 必须为 male 或 female")
        return v.lower()


class RiskFactor(BaseModel):
    factor: str
    contribution: float = Field(..., description="该因素对风险的贡献度 0~1")
    direction: str = Field("increase", description="increase | decrease")


class DiseaseRisk(BaseModel):
    disease: str
    disease_id: str = ""
    risk: float = Field(..., ge=0.0, le=1.0, description="发病概率")
    risk_percent: str = Field("", description="格式化百分比，如 '82.0%'")
    level: RiskLevel = RiskLevel.LOW
    top_factors: List[RiskFactor] = Field(default_factory=list)
    kg_evidence: List[EvidenceItem] = Field(default_factory=list, description="风险关联的图谱依据")


class RiskModelInfo(BaseModel):
    name: str = "MedicalRiskEnsemble"
    version: str = "1.0.0"
    auc: float = 0.923
    disease_coverage: int = 1000
    method: str = Field("xgboost_ensemble", description="xgboost_ensemble | rule_fallback")
    features_used: List[str] = Field(default_factory=list)


class RiskPredictResponse(BaseModel):
    """疾病风险预测响应"""

    predictions: List[DiseaseRisk] = Field(default_factory=list)
    overall_level: RiskLevel = RiskLevel.LOW
    health_score: float = Field(0.0, description="综合健康评分 0~100")
    model: RiskModelInfo = Field(default_factory=RiskModelInfo)
    summary: str = ""
    disclaimer: str = "风险预测结果仅供参考，不能替代执业医师诊断。"
    latency_ms: float = 0.0


class InterventionPlanRequest(RiskPredictRequest):
    """基于预测结果生成干预建议"""

    risk_diseases: List[str] = Field(default_factory=list, description="需重点干预的疾病列表")


class InterventionItem(BaseModel):
    category: str = Field(..., description="diet | exercise | examination | medication | lifestyle")
    category_cn: str = ""
    icon: str = "Apple"
    title: str = ""
    items: List[str] = Field(default_factory=list)
    priority: str = Field("medium", description="high | medium | low")


class InterventionPlanResponse(BaseModel):
    plan: List[InterventionItem] = Field(default_factory=list)
    follow_up: str = Field("", description="随访建议")
    disclaimer: str = "干预建议由 AI 生成，仅供参考，请遵医嘱。"


# =============================================================================
#  系统 / 健康检查
# =============================================================================


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = "医数智答·智愈医典"
    version: str = "1.0.0"
    neo4j: str = Field("unknown", description="connected | fallback_memory | error")
    llm: str = Field("unknown", description="provider 名 或 template")
    rag_templates: int = 0
    models_loaded: Dict[str, bool] = Field(default_factory=dict)
    uptime_seconds: float = 0.0
    server_time: str = ""


EvidenceItem.model_rebuild()
