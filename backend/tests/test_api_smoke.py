# -*- coding: utf-8 -*-
"""
后端冒烟测试（pytest）
======================
覆盖四大业务功能的端到端调用，可在无 Neo4j / 无 PyTorch / 无大模型环境下全部通过。

运行：
    cd backend
    pytest tests/ -v
    pytest tests/ -v -k graph      # 只跑图谱相关
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from fastapi.testclient import TestClient  # noqa: E402

import main as main_mod  # noqa: E402

API = "/api/v1"


@pytest.fixture(scope="module")
def client():
    with TestClient(main_mod.app) as c:
        yield c


# =============================================================================
#  系统
# =============================================================================
def test_health(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    # 演示原型必须常驻免责声明
    assert "不能替代执业医师" in data["disclaimer"]
    # 医疗 RAG 模板库应满足 PPT「100+ 模板」指标
    assert data["rag_templates"] >= 100


def test_system_info_declares_features_and_innovations(client):
    r = client.get(f"{API}/info")
    assert r.status_code == 200
    info = r.json()
    assert len(info["features"]) == 4          # 四大核心业务功能
    assert len(info["innovations"]) == 3       # 三大创新点
    assert set(info["architecture"].keys()) == {"数据层", "知识图谱层", "LLM增强层", "应用层"}


def test_root_and_disclaimer_everywhere(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "不能替代执业医师" in r.json()["disclaimer"]


# =============================================================================
#  ① 知识图谱可视化
# =============================================================================
def test_graph_stats(client):
    r = client.get(f"{API}/graph/stats")
    assert r.status_code == 200
    s = r.json()
    assert s["total_nodes"] > 0
    assert s["total_links"] > 0
    assert s["entity_type_count"] >= 6


def test_graph_entity_types_legend(client):
    r = client.get(f"{API}/graph/entity-types")
    assert r.status_code == 200
    types = r.json()
    labels = {t["type"] for t in types}
    assert {"Disease", "Symptom", "Drug", "Department"} <= labels
    assert all(t["color"].startswith("#") for t in types)


def test_graph_subgraph_d3_payload(client):
    """子图必须返回 D3 力导向图所需的 nodes/links 结构"""
    r = client.get(f"{API}/graph/subgraph", params={"entity": "高血压", "depth": 1, "limit": 200})
    assert r.status_code == 200
    g = r.json()
    assert g["node_count"] > 5
    assert len(g["nodes"]) == g["node_count"]
    assert len(g["links"]) == g["link_count"]
    n = g["nodes"][0]
    assert {"id", "name", "type", "label", "color", "degree"} <= set(n)
    link = g["links"][0]
    assert {"source", "target", "rel", "label"} <= set(link)


def test_graph_subgraph_unknown_entity_returns_message(client):
    r = client.get(f"{API}/graph/subgraph", params={"entity": "完全不存在的实体ZZZ"})
    assert r.status_code == 200
    assert r.json()["node_count"] == 0
    assert r.json()["message"]


def test_graph_search(client):
    r = client.get(f"{API}/graph/search", params={"q": "血压"})
    assert r.status_code == 200
    assert r.json()["total"] > 0


def test_graph_cypher_readonly_allowed(client):
    r = client.post(f"{API}/graph/cypher",
                    json={"cypher": "MATCH (d:Disease) RETURN d.name AS n LIMIT 5"})
    assert r.status_code == 200
    assert len(r.json()["rows"]) > 0


@pytest.mark.parametrize("evil", [
    "MATCH (n) DETACH DELETE n",
    "CREATE (n:Evil {x:1})",
    "MATCH (n:Disease) SET n.name = 'hacked'",
    "MERGE (n:Evil {x:1})",
])
def test_graph_cypher_write_blocked(client, evil):
    """安全红线：受控 Cypher 接口必须拒绝一切写操作"""
    r = client.post(f"{API}/graph/cypher", json={"cypher": evil})
    assert r.status_code == 403


def test_graph_reasoning(client):
    r = client.get(f"{API}/graph/reasoning", params={"mode": "hybrid", "limit": 20})
    assert r.status_code == 200
    assert r.json()["total"] >= 0


# =============================================================================
#  ② 智能疾病查询
# =============================================================================
def test_disease_search_cards(client):
    """卡片必须包含 PPT 界面要求的全部字段"""
    r = client.get(f"{API}/disease/search", params={"q": "高血压", "page": 1, "page_size": 6})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] > 0
    card = data["items"][0]
    for f in ("disease_id", "name", "category1", "category2",
              "symptoms", "treatments", "population"):
        assert f in card


def test_disease_search_by_symptom(client):
    r = client.get(f"{API}/disease/search", params={"q": "头晕", "page_size": 10})
    assert r.status_code == 200
    assert r.json()["total"] > 0


def test_disease_hot_keywords(client):
    r = client.get(f"{API}/disease/hot-keywords")
    assert r.status_code == 200
    assert len(r.json()) >= 5


def test_disease_detail_six_sections(client):
    """疾病详情必须覆盖定义/病因/症状/诊断/治疗/预后 六大板块"""
    r = client.get(f"{API}/disease/肺栓塞")
    assert r.status_code == 200
    d = r.json()
    assert d["name"] == "肺栓塞"
    assert d["definition"]
    assert d["cause"]
    assert len(d["symptoms"]) > 0
    assert d["diagnosis"]
    assert d["treatment"]
    assert d["prognosis"]
    assert d["department"]


def test_disease_sources_traceable(client):
    """知识来源必须 100% 可追溯（PPT 合规指标）"""
    r = client.get(f"{API}/disease/D0001/sources")
    assert r.status_code == 200
    assert len(r.json()) > 0
    s = r.json()[0]
    assert s["title"]


def test_disease_unknown_404(client):
    r = client.get(f"{API}/disease/根本不存在的疾病XYZ")
    assert r.status_code == 404


# =============================================================================
#  ③ 智能问答（可解释 AI 医生）
# =============================================================================
def test_qa_department_query(client):
    r = client.post(f"{API}/qa/ask", json={
        "question": "肺栓塞应该挂什么科？", "session_id": "pytest", "explain": True})
    assert r.status_code == 200
    qa = r.json()
    assert len(qa["answer"]) > 20
    assert qa["intent"]["label"] == "department_query"
    assert len(qa["evidences"]) > 0
    # 可解释性：必须有推理链路
    assert len(qa["reasoning_trace"]) >= 3
    # 溯源：答案中必须有 [KG-n] 引用
    assert "[KG-" in qa["answer"]
    assert "kg-cite" in qa["answer_html"]
    assert "不能替代执业医师" in qa["disclaimer"]


def test_qa_semantic_parse_within_500ms(client):
    """PPT 指标：语义解析响应 ≤500ms"""
    r = client.post(f"{API}/qa/ask", json={"question": "原发性高血压有哪些症状？", "session_id": "pytest"})
    assert r.status_code == 200
    assert r.json()["latency_ms"]["semantic_parse_total"] <= 500


def test_qa_symptom_consult(client):
    r = client.post(f"{API}/qa/ask", json={"question": "胸痛伴低烧可能是什么？", "session_id": "pytest"})
    assert r.status_code == 200
    assert r.json()["intent"]["label"] in ("symptom_consult", "emergency")


def test_qa_safety_override_for_emergency(client):
    """安全兜底：描述急症信号时必须命中急症模板"""
    r = client.post(f"{API}/qa/ask", json={
        "question": "我胸口剧烈疼痛，喘不上气，出冷汗", "session_id": "pytest"})
    assert r.status_code == 200
    qa = r.json()
    assert qa["prompt_template_id"].startswith("emergency")


def test_qa_safety_override_for_dosage(client):
    """安全兜底：索取用药剂量时必须命中剂量护栏模板"""
    r = client.post(f"{API}/qa/ask", json={
        "question": "二甲双胍一次吃几片？", "session_id": "pytest"})
    assert r.status_code == 200
    assert r.json()["prompt_template_id"] == "safety_dose_guard_v1"


def test_qa_prompt_template_library(client):
    r = client.get(f"{API}/qa/prompt-templates")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 100
    assert len(data["intents"]) >= 10


def test_qa_history_and_clear(client):
    client.post(f"{API}/qa/ask", json={"question": "糖尿病怎么治？", "session_id": "hist-test"})
    r = client.get(f"{API}/qa/history/hist-test")
    assert r.status_code == 200
    assert r.json()["turns"] >= 1

    r = client.delete(f"{API}/qa/history/hist-test")
    assert r.status_code == 200
    r = client.get(f"{API}/qa/history/hist-test")
    assert r.json()["turns"] == 0


def test_qa_validation_rejects_empty(client):
    r = client.post(f"{API}/qa/ask", json={"question": ""})
    assert r.status_code == 422


# =============================================================================
#  ④ 多维数据分析 & 健康预警
# =============================================================================
def test_analytics_overview_four_metrics(client):
    """PPT 数据看板：医疗实体总数 / 疾病分类数 / 传染性疾病 / 治疗周期类型"""
    r = client.get(f"{API}/analytics/overview")
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert len(metrics) == 4
    labels = {m["label"] for m in metrics}
    assert "医疗实体总数" in labels
    assert "疾病分类数" in labels
    assert "传染性疾病" in labels


@pytest.mark.parametrize("ep", [
    "node-type-pie", "category-bar", "infectious-gauge",
    "symptom-top", "department-distribution",
])
def test_analytics_charts(client, ep):
    r = client.get(f"{API}/analytics/{ep}")
    assert r.status_code == 200
    data = r.json()
    assert data["echarts_option"], f"{ep} 缺少 echarts_option"
    assert data["insight"], f"{ep} 缺少自动洞察结论"


def test_analytics_infectious_proportion(client):
    r = client.get(f"{API}/analytics/infectious-gauge")
    series = r.json()["series"]
    assert len(series) == 3
    assert {s["name"] for s in series} == {"是", "否", "是否传染"}


RISK_PAYLOAD = {
    "age": 58, "gender": "male", "bmi": 27.4,
    "systolic_bp": 152, "diastolic_bp": 96,
    "fasting_glucose": 6.4, "total_cholesterol": 5.9,
    "ldl": 3.8, "hdl": 1.0, "triglycerides": 2.4,
    "smoking": True, "physical_activity": "low",
    "family_history": ["hypertension", "diabetes"],
    "symptoms": ["头晕", "乏力"],
}


def test_risk_predict(client):
    r = client.post(f"{API}/analytics/risk/predict", json=RISK_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) >= 3
    # 该画像（58 岁男性、收缩压 152、吸烟、糖尿病+高血压家族史、BMI 27.4）
    # 属于典型心血管代谢高危人群：心脑血管疾病必须排在前列
    top3 = [p["disease"] for p in data["predictions"][:3]]
    assert any(d in top3 for d in ("原发性高血压", "冠心病", "脑卒中", "2型糖尿病")), top3
    # 高血压必须是高风险等级（收缩压 152 + 家族史 + BMI 偏高）
    ht = next((p for p in data["predictions"] if p["disease"] == "原发性高血压"), None)
    assert ht is not None and ht["risk"] >= 0.5, ht
    assert ht["level"] in ("高风险", "极高风险")
    # 可解释性：每个预测必须给出因子贡献度
    assert all(len(p["top_factors"]) > 0 for p in data["predictions"][:3])
    # PPT 指标：预测 AUC ≥0.9
    assert data["model"]["auc"] >= 0.9
    assert "不能替代执业医师" in data["disclaimer"]


def test_risk_predict_low_risk_profile(client):
    """健康青年画像：健康评分应高位、各病风险低位（用于检验模型校准是否合理）"""
    healthy = {"age": 24, "gender": "female", "bmi": 20.5, "systolic_bp": 108,
               "diastolic_bp": 68, "fasting_glucose": 4.6, "total_cholesterol": 4.1,
               "hdl": 1.8, "triglycerides": 0.9, "smoking": False,
               "physical_activity": "high", "family_history": [], "symptoms": []}
    r = client.post(f"{API}/analytics/risk/predict", json=healthy)
    assert r.status_code == 200
    data = r.json()
    assert data["health_score"] > 60
    assert all(p["risk"] < 0.5 for p in data["predictions"])
    # 校准合理性：健康青年最常见的心血管代谢风险也应在 10% 以下
    assert data["predictions"][0]["risk"] < 0.10, data["predictions"][0]
    assert data["overall_level"] == "低风险"


def test_risk_intervention_plan(client):
    r = client.post(f"{API}/analytics/risk/intervene", json={**RISK_PAYLOAD, "risk_diseases": []})
    assert r.status_code == 200
    plan = r.json()["plan"]
    assert len(plan) == 4
    cats = {p["category"] for p in plan}
    assert cats == {"diet", "exercise", "examination", "lifestyle"}
    assert all(len(p["items"]) >= 3 for p in plan)


def test_risk_predict_validation(client):
    r = client.post(f"{API}/analytics/risk/predict", json={"age": 999, "gender": "x"})
    assert r.status_code == 422
    assert "errors" in r.json()


# =============================================================================
#  数据层：脱敏与加密（合规）
# =============================================================================
def test_privacy_masking_and_encryption():
    from app.data_layer.privacy import PrivacyGuard

    g = PrivacyGuard()
    raw = ("患者张三，身份证号 110101196503151234，手机号 13812345678，"
           "住址：北京市朝阳区建国路88号3单元502室。血压 152/96 mmHg。")
    masked = g.mask(raw)
    assert "110101196503151234" not in masked.text
    assert "13812345678" not in masked.text
    assert "建国路88号" not in masked.text
    # 医学数值必须保留（否则会破坏知识可用性）
    assert "152/96" in masked.text
    # 加密可逆
    token = g.encrypt_field("张三")
    assert token.startswith("v1:")
    assert g.decrypt_field(token) == "张三"


def test_anonymize_health_input_drops_identifiers():
    from app.data_layer.privacy import PrivacyGuard

    out = PrivacyGuard.anonymize_health_input(
        {"name": "张三", "id_card": "110101196503151234", "age": 58, "smoking": True})
    assert "name" not in out
    assert "id_card" not in out
    assert out["age"] == 58
    assert out["smoking"] is True


# =============================================================================
#  三种创新点：可直接运行的自检
# =============================================================================
def test_innovation1_medical_keyword_attention():
    """★创新点 1：医疗关键词注意力模块可运行且能解释关注位置"""
    from app.kg_layer.attention import get_keyword_matcher, self_test

    st = self_test()
    assert st["backend"] in ("torch", "numpy")
    assert st["keywords"] > 0

    matcher = get_keyword_matcher()
    text = "患者因突发性呼吸困难伴胸痛就诊，予以抗凝治疗。"
    hits = matcher.match(text)
    classes = {h["class"] for h in hits}
    assert "symptom" in classes      # 症状类关键词
    assert "treatment" in classes    # 治疗类关键词
    # 关键词词表规模（症状 58 + 治疗 52 + 检查 52 + 科室 39）
    assert sum(len(v) for v in matcher.lexicon.values()) > 150


def test_innovation1_attention_temperature_boost():
    """症状/治疗类关键词必须获得更高的注意力温度（创新点核心机制）"""
    from app.kg_layer.attention import DEFAULT_TYPE_TEMPERATURE

    assert DEFAULT_TYPE_TEMPERATURE["symptom"] > 1.0
    assert DEFAULT_TYPE_TEMPERATURE["treatment"] > 1.0
    assert DEFAULT_TYPE_TEMPERATURE["symptom"] > DEFAULT_TYPE_TEMPERATURE["check"]


def test_innovation2_prompt_library_has_safety_overrides():
    """★创新点 2：医疗 RAG 模板库必须含安全护栏模板"""
    from app.llm_layer.rag.prompt_templates import SAFETY_OVERRIDES, get_prompt_library

    lib = get_prompt_library()
    ids = set(lib.by_id.keys())
    for required in ("emergency_core_v1", "safety_dose_guard_v1",
                     "safety_self_harm_v1", "emergency_chest_pain_v1",
                     "treatment_query_drug_interaction_v1"):
        assert required in ids, f"缺少安全模板 {required}"
    assert len(SAFETY_OVERRIDES) >= 8


def test_innovation2_kg_triples_injected_into_prompt():
    """★创新点 2：渲染后的 prompt 必须包含 KG 三元组与全局硬约束"""
    from app.llm_layer.rag.prompt_templates import get_prompt_library

    lib = get_prompt_library()
    tpl = lib.select("disease_query", "高血压是什么病")
    assert tpl is not None
    rendered = lib.render(tpl, {
        "question": "高血压是什么病",
        "kg_context": "[KG-1] 原发性高血压 —症状→ 头晕",
        "entities": "原发性高血压(Disease)",
        "main_entity": "原发性高血压",
        "intent_cn": "疾病查询",
    })
    assert "[KG-1]" in rendered                       # 三元组已注入
    assert "只能使用 <KG_CONTEXT>" in rendered        # 全局硬约束已追加
    assert "不能替代执业医师" in rendered              # 免责声明已追加


def test_innovation3_lightweight_graph_payload():
    """★创新点 3：轻量图谱载荷必须可裁剪（limit 生效）且含图例配色"""
    from app.kg_layer.memory_store import MemoryGraphStore

    store = MemoryGraphStore.instance()
    # 充分大的 limit：应完整返回 1 跳邻域，且未被截断
    full = store.subgraph("高血压", depth=1, limit=5000)
    assert full["node_count"] > 5
    assert full["truncated"] is False
    # 严格 limit：节点数被裁剪，并明确标记 truncated
    small = store.subgraph("高血压", depth=1, limit=5)
    assert small["node_count"] <= 5
    assert small["truncated"] is True
    # 图例配色：每个节点都必须带颜色
    assert all(n["color"].startswith("#") for n in full["nodes"])
    # 边必须是节点集合内的边（无悬空引用），否则前端 D3 会报错
    ids = {n["id"] for n in full["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in full["links"])
    assert full["link_count"] == len(full["links"])


def test_relation_extractor_reports_recall_gain():
    """创新点 1 的 PPT 指标必须可查询（便于答辩说明）"""
    from app.kg_layer.relation_extractor import get_relation_extractor

    info = get_relation_extractor().info()
    assert info["attention_module"] == "MedicalKeywordAttention"
    assert info["recall_gain_vs_pcnn"] == "+8.3%"
