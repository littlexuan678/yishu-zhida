# -*- coding: utf-8 -*-
"""End-to-end smoke test for 医数智答 backend."""
import asyncio
import json
import os
import sys
import traceback

sys.path.insert(0, r"C:\Users\ac010\Documents\deepseek-harness\default-workspace\医数智答\backend")
os.chdir(r"C:\Users\ac010\Documents\deepseek-harness\default-workspace\医数智答\backend")

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (("  :: " + str(extra)[:300]) if extra else ""))


async def main():
    print("=" * 78)
    print(" 1. 内存图谱加载")
    print("=" * 78)
    from app.kg_layer.memory_store import MemoryGraphStore
    store = MemoryGraphStore.instance()
    print(f"  nodes={store.node_count} edges={store.edge_count} diseases={len(store.diseases)}")
    check("memory graph loaded", store.node_count > 400 and store.edge_count > 1500)
    check("59 diseases present", len(store.diseases) >= 50, len(store.diseases))

    print()
    print("=" * 78)
    print(" 2. 知识图谱层：实体识别 / 关系抽取(改进PCNN) / 图谱服务")
    print("=" * 78)
    from app.kg_layer.entity_extractor import get_entity_extractor
    from app.kg_layer.relation_extractor import get_relation_extractor
    ex = get_entity_extractor()
    print("  entity backend:", ex.backend, "| lexicon:", ex.info()["lexicon_size"])
    m = ex.recognize("我最近头晕还恶心，血压有点高，之前有高血压病史")
    check("entity recognition works", len(m) > 0, [(x.text, x.type) for x in m])
    linked = ex.link_to_kg(m)
    check("entity linking works", any(x.kg_name for x in linked),
          [(x.text, x.kg_name) for x in linked][:5])

    rx = get_relation_extractor()
    print("  relation backend:", rx.backend)
    print("  attention self-test:", json.dumps(rx.info()["attention_self_test"], ensure_ascii=False))
    check("improved PCNN attention module available",
          rx.info()["attention_module"] == "MedicalKeywordAttention")
    exp = rx.explain_attention("患者因突发性呼吸困难伴胸痛就诊，经CT肺动脉造影确诊为肺栓塞，予以抗凝治疗。",
                               entity_types=("Disease", "Treatment"))
    check("medical keyword attention explain works",
          len(exp["keywords"]) > 0 and len(exp["highlighted"]) > 0,
          f"kw={len(exp['keywords'])} class={exp['keyword_class']} tau={exp['temperature']}")

    from app.kg_layer.graph_service import get_graph_service
    gs = get_graph_service()
    st = await gs.get_stats()
    print(f"  graph stats: nodes={st.total_nodes} links={st.total_links} types={st.entity_type_count} src={st.data_source}")
    check("graph stats ok", st.total_nodes > 400 and st.total_links > 1500)

    sub = await gs.get_subgraph("高血压", depth=1, limit=200)
    print(f"  subgraph(高血压): nodes={sub.node_count} links={sub.link_count} center={sub.center} truncated={sub.truncated}")
    check("subgraph query works", sub.node_count > 5 and sub.link_count > 5)

    _, hits = gs.search_entities("血压", limit=5)
    check("entity search works", len(hits) > 0, [h.name for h in hits])

    total, cards = gs.search_diseases("高血压", page=1, page_size=6)
    print(f"  disease search 高血压: total={total}")
    for c in cards[:3]:
        print(f"    - {c.name} | {c.category1}/{c.category2} | 症状={len(c.symptoms)} | 治疗={c.treatments} | 人群={c.population}")
    check("disease search works", total > 3, total)

    total2, cards2 = gs.search_diseases("胃炎", page=1, page_size=6)
    print(f"  disease search 胃炎: total={total2} -> {[c.name for c in cards2]}")
    check("gastritis search works", total2 >= 3, total2)

    d = gs.get_disease_detail("肺栓塞")
    check("disease detail works", d is not None and d.name == "肺栓塞" and len(d.symptoms) > 0,
          f"{d.name if d else None} dept={d.department if d else None} symptoms={d.symptoms[:4] if d else None}")
    check("disease detail has sources", d is not None and len(d.sources) > 0,
          [s.pmid or s.title for s in (d.sources if d else [])][:3])

    ev = gs.get_evidence("原发性高血压", "HAS_SYMPTOM", None, "KG-1")
    check("evidence/source traceability works", ev is not None and ev.head == "原发性高血压",
          f"{ev.head}->{ev.tail} conf={ev.confidence} srcs={len(ev.sources)}" if ev else None)

    print("  hot keywords:", gs.hot_keywords())
    check("hot keywords", len(gs.hot_keywords()) >= 5)
    print("  related questions:", gs.related_questions("肺栓塞", 4)[:4])
    check("related questions", len(gs.related_questions("肺栓塞", 4)) > 0)

    print()
    print("=" * 78)
    print(" 3. 图谱推理补全（Jena 规则 + GAT/GCN 降级）")
    print("=" * 78)
    from app.kg_layer.graph_reasoner import get_graph_reasoner
    rs = get_graph_reasoner()
    print("  reasoner info:", json.dumps({k: v for k, v in rs.info().items() if k != "jena_cli"}, ensure_ascii=False))
    inf = rs.infer_by_rules(limit=20)
    check("rule-based inference works", len(inf) > 0, f"{len(inf)} triples; e.g. {inf[0] if inf else None}")
    comp = rs.complete_by_gnn(top_k=10)
    check("gnn/heuristic completion works", len(comp) > 0, f"{len(comp)} triples")
    paths = rs.explain_path("头晕", "原发性高血压", max_hops=3)
    check("multi-hop path explain works", len(paths) > 0,
          paths[0] if paths else None)

    print()
    print("=" * 78)
    print(" 4. LLM 增强层：TextCNN 意图 / 实体链接 / Cypher 生成 / Prompt 库")
    print("=" * 78)
    from app.llm_layer.intent_classifier import get_intent_classifier
    ic = get_intent_classifier()
    print("  intent backend:", ic.backend)
    for q in ["高血压是什么病", "胸痛伴低烧可能是什么", "糖尿病怎么治", "肺栓塞应该挂什么科"]:
        r = ic.classify(q)
        print(f"    {q:<22} -> {r.label:<18} ({r.label_cn}) conf={r.confidence:.3f} via {r.method}")
    check("intent classification 4 classes",
          ic.classify("肺栓塞应该挂什么科").label == "department_query"
          and ic.classify("糖尿病怎么治").label == "treatment_query")

    from app.llm_layer.entity_linker import get_entity_linker
    el = get_entity_linker()
    a = el.analyze("我血压高，最近老是头晕")
    print("  entity linker:", a["diseases"], a["symptoms"], "| colloquial:", a["colloquial_hits"])
    check("colloquial entity linking (血压高->原发性高血压)",
          "原发性高血压" in (a["diseases"] or []), a["diseases"])

    from app.llm_layer.cypher_generator import get_cypher_generator
    cg = get_cypher_generator()
    qs = cg.build("department_query", a, limit=20)
    ok_all = all(cg.validate(q)[0] for q in qs)
    check("cypher generation + readonly validation", len(qs) > 0 and ok_all,
          f"{len(qs)} queries, all readonly={ok_all}")

    from app.llm_layer.rag.prompt_templates import get_prompt_library
    lib = get_prompt_library()
    v = lib.validate()
    print("  prompt library stats:", json.dumps(lib.stats(), ensure_ascii=False))
    print("  validate:", v["ok"], "missing:", v["missing_required"])
    check("100+ medical prompt templates loaded", len(lib.templates) >= 100, len(lib.templates))
    tpl = lib.select("department_query", "肺栓塞应该挂什么科", a)
    rendered = lib.render(tpl, {"question": "肺栓塞应该挂什么科", "kg_context": "[KG-1] 肺栓塞 —科室→ 呼吸内科",
                                "entities": "肺栓塞(Disease)", "main_entity": "肺栓塞",
                                "intent_cn": "科室导诊"})
    check("template selection + render", tpl is not None and "肺栓塞" in rendered,
          f"tpl={tpl.id if tpl else None} len={len(rendered)}")
    safe = lib.select("symptom_consult", "我胸口疼得厉害，喘不上气", a)
    check("safety override triggers emergency template", safe is not None and "emergency" in safe.id,
          safe.id if safe else None)

    print()
    print("=" * 78)
    print(" 5. RAG 引擎端到端（可解释问答）")
    print("=" * 78)
    from app.llm_layer.rag.rag_engine import get_rag_engine
    rag = get_rag_engine()
    for q in ["肺栓塞应该挂什么科？", "原发性高血压有哪些症状？", "2型糖尿病怎么治疗？"]:
        r = await rag.answer(q, session_id="smoke", use_llm=False)
        print(f"\n  Q: {q}")
        print(f"    intent={r.intent.label}({r.intent.label_cn}) conf={r.intent.confidence:.2f}"
              f" entities={[e.kg_name or e.text for e in r.entities]}")
        print(f"    evidences={len(r.evidences)} trace={len(r.reasoning_trace)} steps"
              f" prompt={r.prompt_template_id} conf={r.confidence:.2f}")
        print(f"    semantic_parse={r.latency_ms.semantic_parse_total}ms total={r.latency_ms.total}ms")
        print(f"    guard={r.guard_result.get('suggested_action')} score={r.guard_result.get('score')}")
        print(f"    answer[:150]={r.answer[:150]!r}")
        check(f"RAG answer non-empty: {q}", len(r.answer) > 20)
        check(f"RAG evidences>0: {q}", len(r.evidences) > 0, len(r.evidences))
        check(f"RAG trace has steps: {q}", len(r.reasoning_trace) >= 3, len(r.reasoning_trace))
        check(f"semantic parse <=500ms: {q}", r.latency_ms.semantic_parse_total <= 500,
              r.latency_ms.semantic_parse_total)

    print()
    print("=" * 78)
    print(" 6. 应用服务层：问答服务 / 分析 / 风险预测 / 干预")
    print("=" * 78)
    from app.app_layer.services.qa_service import QAService
    qs_svc = QAService(graph=gs, intent=ic, linker=el)
    r = await qs_svc.ask("高血压要注意什么？", session_id="svc", use_llm=False)
    check("QAService.ask works", len(r.answer) > 20)
    check("QAService session history", len(qs_svc.history("svc")) == 2, len(qs_svc.history("svc")))

    from app.app_layer.services.analytics_service import get_analytics_service
    an = get_analytics_service()
    ov = an.overview()
    print("  overview metrics:", [(m.label, m.value) for m in ov.metrics])
    check("analytics overview 4 metrics", len(ov.metrics) == 4)
    pie = an.node_type_pie()
    check("node type pie chart", len(pie.series) > 0 and bool(pie.echarts_option),
          f"{len(pie.series)} slices, pyecharts={'yes' if pie.pyecharts_html else 'no'}")
    bar = an.category_bar()
    check("category bar chart", len(bar.categories) > 0 and len(bar.values) > 0,
          f"{len(bar.categories)} categories {bar.categories}")
    g = an.infectious_gauge()
    check("infectious gauge", len(g.series) == 3, [(s.name, s.value) for s in g.series])
    print("  insights:", pie.insight[:80], "|", bar.insight[:80])

    from app.app_layer.services.risk_predictor import get_risk_predictor
    from app.schemas import RiskPredictRequest, InterventionPlanRequest
    rp = get_risk_predictor()
    print("  risk model info:", json.dumps(rp.info(), ensure_ascii=False)[:220])
    req = RiskPredictRequest(age=58, gender="male", bmi=27.4, systolic_bp=152, diastolic_bp=96,
                             fasting_glucose=6.4, total_cholesterol=5.9, ldl=3.8, hdl=1.0,
                             triglycerides=2.4, smoking=True, physical_activity="low",
                             family_history=["hypertension", "diabetes"], symptoms=["头晕", "乏力"])
    pred = rp.predict(req)
    print(f"  health_score={pred.health_score} overall={pred.overall_level} model={pred.model.name} auc={pred.model.auc} method={pred.model.method}")
    for p in pred.predictions[:4]:
        print(f"    {p.disease:<16} risk={p.risk_percent:<8} level={p.level:<6} factors={[(f.factor, f.contribution) for f in p.top_factors[:2]]}")
    check("risk prediction returns results", len(pred.predictions) >= 3, len(pred.predictions))
    top3 = [p.disease for p in pred.predictions[:3]]
    check("cardiometabolic diseases rank top for this high-risk profile",
          any(d in top3 for d in ("原发性高血压", "冠心病", "脑卒中", "2型糖尿病")), top3)
    ht = next((p for p in pred.predictions if p.disease == "原发性高血压"), None)
    check("hypertension flagged high-risk for this profile",
          ht is not None and ht.risk >= 0.5 and ht.level.value in ("高风险", "极高风险"),
          f"{ht.risk_percent} {ht.level.value}" if ht else None)
    check("risk factors explained", all(len(p.top_factors) > 0 for p in pred.predictions[:3]))
    check("model AUC reported >=0.9", pred.model.auc >= 0.9, pred.model.auc)

    plan = rp.intervention(InterventionPlanRequest(**req.model_dump(), risk_diseases=[]))
    print("  intervention plan:", [(i.category_cn, len(i.items)) for i in plan.plan])
    check("intervention plan 4 categories", len(plan.plan) == 4, [i.category_cn for i in plan.plan])
    check("intervention has concrete items", all(len(i.items) >= 3 for i in plan.plan))

    print()
    print("=" * 78)
    print(" 7. 数据层：脱敏 / 加密 / 清洗")
    print("=" * 78)
    from app.data_layer.privacy import PrivacyGuard
    pg = PrivacyGuard()
    demo = ("患者张三，男，58岁，身份证号 110101196503151234，手机号 13812345678，"
            "住院号：ZY20240115，住址：北京市朝阳区建国路88号3单元502室。"
            "血压 152/96 mmHg，空腹血糖 6.4 mmol/L，参考 PMID: 32130469。")
    res = pg.mask(demo)
    print("  original :", demo[:60], "...")
    print("  masked   :", res.text[:130], "...")
    print("  hits     :", res.hits)
    check("PII masking removes id card", "110101196503151234" not in res.text)
    check("PII masking removes phone", "13812345678" not in res.text)
    check("PII masking removes address", "建国路88号" not in res.text)
    check("medical values preserved (152/96, 6.4)", "152/96" in res.text and "6.4" in res.text)
    check("PMID preserved", "32130469" in res.text)
    tok = pg.encrypt_field("张三")
    check("AES-256-GCM encrypt/decrypt roundtrip",
          pg.decrypt_field(tok) == "张三" and tok.startswith("v1:"), tok[:40])
    check("anon health input strips free text",
          "name" not in pg.anonymize_health_input({"name": "张三", "age": 58}), pg.anonymize_health_input({"name": "张三", "age": 58}))

    from app.data_layer.preprocess import Preprocessor
    pp = Preprocessor()
    docs = pp.process_batch([
        {"title": "高血压的治疗进展", "abstract": "原发性高血压患者常表现为头晕、头痛，治疗以药物治疗与饮食治疗为主，血压 152/96 mmHg。", "pmid": "123", "year": 2024},
        {"title": "广告", "abstract": "高血压最新治疗方法！加微信 xxx123 免费领取专家方案，限时优惠！"},
        {"title": "高血压的治疗进展", "abstract": "原发性高血压患者常表现为头晕、头痛，治疗以药物治疗与饮食治疗为主，血压 152/96 mmHg。", "pmid": "123", "year": 2024},
    ])
    print("  preprocess stats:", pp.info()["stats"])
    check("preprocess drops ads and duplicates", len(docs) == 1, len(docs))

    print()
    print("=" * 78)
    print(" 8. FastAPI 应用（全部路由 / OpenAPI）")
    print("=" * 78)
    from fastapi.testclient import TestClient
    import main as main_mod
    with TestClient(main_mod.app) as client:
        r = client.get("/api/v1/health")
        check("GET /health 200", r.status_code == 200, r.status_code)
        h = r.json()
        print("  health:", json.dumps({k: h[k] for k in ("status", "neo4j", "llm", "rag_templates") if k in h}, ensure_ascii=False))
        check("health reports 100+ rag templates", h.get("rag_templates", 0) >= 100, h.get("rag_templates"))

        r = client.get("/api/v1/info")
        check("GET /info 200", r.status_code == 200)
        info = r.json()
        check("info lists 4 features", len(info.get("features", [])) == 4)
        check("info lists 3 innovations", len(info.get("innovations", [])) == 3)

        r = client.get("/api/v1/graph/stats")
        check("GET /graph/stats 200", r.status_code == 200, r.status_code)
        r = client.get("/api/v1/graph/entity-types")
        check("GET /graph/entity-types 200", r.status_code == 200 and len(r.json()) >= 6, r.status_code)
        r = client.get("/api/v1/graph/subgraph", params={"entity": "高血压", "depth": 1, "limit": 200})
        check("GET /graph/subgraph 200", r.status_code == 200 and r.json()["node_count"] > 5, r.status_code)
        r = client.get("/api/v1/graph/search", params={"q": "血压"})
        check("GET /graph/search 200", r.status_code == 200 and r.json()["total"] > 0, r.status_code)
        r = client.get("/api/v1/graph/reasoning", params={"mode": "hybrid", "limit": 20})
        check("GET /graph/reasoning 200", r.status_code == 200, r.status_code)
        r = client.post("/api/v1/graph/cypher", json={"cypher": "MATCH (d:Disease) RETURN d.name AS n LIMIT 5"})
        check("POST /graph/cypher readonly allowed", r.status_code == 200 and len(r.json()["rows"]) > 0, r.status_code)
        r = client.post("/api/v1/graph/cypher", json={"cypher": "MATCH (n) DETACH DELETE n"})
        check("POST /graph/cypher WRITE blocked (403)", r.status_code == 403, f"{r.status_code} {r.text[:120]}")
        r = client.post("/api/v1/graph/cypher", json={"cypher": "CREATE (n:Evil {x:1})"})
        check("POST /graph/cypher CREATE blocked (403)", r.status_code == 403, r.status_code)

        r = client.get("/api/v1/disease/search", params={"q": "高血压", "page": 1, "page_size": 6})
        check("GET /disease/search 200", r.status_code == 200 and r.json()["total"] > 3, r.status_code)
        d0 = r.json()["items"][0]
        check("disease card has required fields",
              all(k in d0 for k in ("disease_id", "name", "category1", "category2", "symptoms", "treatments", "population")),
              list(d0.keys()))
        r = client.get("/api/v1/disease/hot-keywords")
        check("GET /disease/hot-keywords 200", r.status_code == 200 and len(r.json()) >= 5, r.json())
        r = client.get("/api/v1/disease/D0013")
        check("GET /disease/{id} 200", r.status_code == 200 and r.json()["name"] == "肺栓塞",
              f"{r.status_code} {r.json().get('name')}")
        r = client.get("/api/v1/disease/肺栓塞/related")
        check("GET /disease/{id}/related 200", r.status_code == 200, r.status_code)
        r = client.get("/api/v1/disease/不存在病XYZ")
        check("GET /disease unknown -> 404", r.status_code == 404, r.status_code)

        r = client.post("/api/v1/qa/ask", json={"question": "肺栓塞应该挂什么科？", "session_id": "api", "top_k": 12, "explain": True})
        check("POST /qa/ask 200", r.status_code == 200, r.status_code)
        qa = r.json()
        check("qa response contract complete",
              all(k in qa for k in ("answer", "answer_html", "intent", "entities", "evidences",
                                    "reasoning_trace", "confidence", "latency_ms", "disclaimer", "related_questions")),
              list(qa.keys()))
        check("qa answer mentions KG citation", "[KG-" in qa["answer"], qa["answer"][:120])
        check("qa answer_html has clickable citations", "kg-cite" in qa["answer_html"])
        check("qa disclaimer present", "不能替代执业医师" in qa["disclaimer"])
        print("  answer:", qa["answer"][:200].replace("\n", " "))
        r = client.get("/api/v1/qa/prompt-templates")
        check("GET /qa/prompt-templates 200", r.status_code == 200 and r.json()["total"] >= 100,
              r.json().get("total"))
        r = client.get("/api/v1/qa/history/api")
        check("GET /qa/history 200", r.status_code == 200 and r.json()["turns"] >= 1, r.json().get("turns"))
        r = client.get("/api/v1/qa/info")
        check("GET /qa/info 200", r.status_code == 200, r.status_code)

        r = client.get("/api/v1/analytics/overview")
        check("GET /analytics/overview 200", r.status_code == 200 and len(r.json()["metrics"]) == 4, r.status_code)
        for ep in ("node-type-pie", "category-bar", "infectious-gauge", "symptom-top", "department-distribution"):
            r = client.get(f"/api/v1/analytics/{ep}")
            check(f"GET /analytics/{ep} 200", r.status_code == 200 and bool(r.json().get("echarts_option")), r.status_code)
        r = client.get("/api/v1/analytics/charts")
        check("GET /analytics/charts 200", r.status_code == 200 and len(r.json()) == 6, r.status_code)
        r = client.post("/api/v1/analytics/risk/predict", json=req.model_dump())
        check("POST /analytics/risk/predict 200", r.status_code == 200 and len(r.json()["predictions"]) >= 3, r.status_code)
        r = client.post("/api/v1/analytics/risk/intervene", json={**req.model_dump(), "risk_diseases": []})
        check("POST /analytics/risk/intervene 200", r.status_code == 200 and len(r.json()["plan"]) == 4, r.status_code)
        r = client.post("/api/v1/analytics/risk/predict", json={"age": 999, "gender": "x"})
        check("risk predict validation -> 422", r.status_code == 422, r.status_code)

        openapi = client.get("/openapi.json").json()
        paths = openapi["paths"]
        check("OpenAPI has all 4 modules", all(any(p.startswith(f"/api/v1/{m}") for p in paths)
                                              for m in ("qa", "disease", "graph", "analytics")))
        print(f"  OpenAPI: {len(paths)} paths, {len(openapi.get('components', {}).get('schemas', {}))} schemas")

    print()
    print("=" * 78)
    print(f" RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 78)
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print("  -", f)
        return 1
    print("ALL BACKEND SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
