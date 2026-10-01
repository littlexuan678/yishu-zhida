# -*- coding: utf-8 -*-
"""实际调用问答接口，展示聊天效果。"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "http://127.0.0.1:8000/api/v1/qa/ask"


def ask(question, session="demo"):
    payload = json.dumps({
        "question": question, "session_id": session,
        "top_k": 12, "max_hops": 2, "explain": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        API, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


QUESTIONS = [
    "肺栓塞应该挂什么科？",
    "原发性高血压有哪些症状？",
    "我最近头晕还恶心，是怎么回事？",
    "二甲双胍一次吃几片？",
    "今天天气怎么样？",
]

for q in QUESTIONS:
    print("=" * 78)
    print(f"👤 我：{q}")
    print("-" * 78)
    try:
        r = ask(q)
    except Exception as exc:  # noqa: BLE001
        print(f"   [请求失败] {exc}")
        continue

    it = r["intent"]
    ents = "、".join(f"{e['kg_name'] or e['text']}" for e in r["entities"]) or "无"
    print(f"🤖 智愈医典（意图：{it['label_cn']} · 置信度 {it['confidence']:.2f} · 识别实体：{ents}）")
    print()
    # 答案正文（截断展示）
    ans = r["answer"].strip()
    for line in ans.split("\n")[:18]:
        print("   " + line)
    if len(ans.split("\n")) > 18:
        print("   ...（已截断）")
    print()
    print(f"   ├ 命中模板     : {r['prompt_template_id']}")
    print(f"   ├ 知识溯源     : {len(r['evidences'])} 条")
    if r["evidences"]:
        e = r["evidences"][0]
        src = e["sources"][0] if e.get("sources") else {}
        print(f"   │   {e['triple_id']}: {e['head']} —{e['relation_label']}→ {e['tail']}"
              f"（置信度 {e['confidence']}）")
        if src:
            print(f"   │     来源：{src.get('title', '')[:50]} {src.get('pmid') or ''}")
    print(f"   ├ 推理链路     : {len(r['reasoning_trace'])} 步")
    for s in r["reasoning_trace"][:3]:
        print(f"   │   {s['step']}. {s['title']}：{s['detail'][:70]}")
    print(f"   ├ 答案置信度   : {r['confidence']:.2f}")
    print(f"   ├ 幻觉守卫     : {r['guard_result'].get('suggested_action')}"
          f"（score {r['guard_result'].get('score')}）")
    print(f"   └ 耗时         : 语义解析 {r['latency_ms']['semantic_parse_total']}ms / "
          f"总计 {r['latency_ms']['total']}ms")
    print()
