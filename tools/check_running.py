# -*- coding: utf-8 -*-
"""确认前后端是否正在运行。"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def probe(url, timeout=6):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


print("=" * 70)
st, body = probe("http://127.0.0.1:8000/api/v1/health")
if st == 200:
    d = json.loads(body)
    print("✅ 后端 FastAPI 正在运行  http://127.0.0.1:8000")
    print(f"   应用名   : {d.get('app')}")
    print(f"   Neo4j    : {d.get('neo4j')}")
    print(f"   大模型   : {d.get('llm')}")
    print(f"   RAG 模板 : {d.get('rag_templates')} 个")
    print(f"   接口文档 : http://127.0.0.1:8000/docs")
else:
    print("❌ 后端未运行  ->", body[:120])

print("-" * 70)
st, body = probe("http://127.0.0.1:5173/")
if st == 200 and 'id="app"' in body.replace("'", '"'):
    print("✅ 前端 Vite 正在运行  http://127.0.0.1:5173")
else:
    print("❌ 前端未运行  ->", str(body)[:120])

print("-" * 70)
st, body = probe("http://127.0.0.1:5173/api/v1/graph/stats")
if st == 200:
    d = json.loads(body)
    print(f"✅ 前后端已打通（Vite 代理 /api → FastAPI）")
    print(f"   图谱：{d.get('total_nodes')} 节点 / {d.get('total_links')} 关系 / {d.get('entity_type_count')} 类实体")
else:
    print("⚠️  代理未打通 ->", str(body)[:120])
print("=" * 70)
