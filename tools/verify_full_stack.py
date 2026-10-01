# -*- coding: utf-8 -*-
"""Verify the Vite dev server serves the app and proxies /api to the backend."""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:5173"   # Vite dev server
results = []


def get(url, timeout=15):
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", errors="replace")


# 1) Vite serves the SPA entry
try:
    st, body = get(BASE + "/")
    ok = st == 200 and "<div id=\"app\">" in body.replace("'", '"')
    results.append(("Vite serves SPA index.html", ok, f"status={st} len={len(body)}"))
except Exception as e:
    results.append(("Vite serves SPA index.html", False, str(e)[:200]))

# 2) Vite serves main.js (module graph alive)
try:
    st, body = get(BASE + "/src/main.js")
    results.append(("Vite serves /src/main.js", st == 200 and "createApp" in body,
                    f"status={st} len={len(body)}"))
except Exception as e:
    results.append(("Vite serves /src/main.js", False, str(e)[:200]))

# 3) PROXY: /api/v1/health must reach FastAPI through Vite
try:
    st, body = get(BASE + "/api/v1/health")
    data = json.loads(body)
    results.append(("Proxy /api -> FastAPI /health", st == 200 and data.get("status") == "ok",
                    f"status={st} app={data.get('app')} neo4j={data.get('neo4j')} "
                    f"templates={data.get('rag_templates')}"))
except Exception as e:
    results.append(("Proxy /api -> FastAPI /health", False, str(e)[:200]))

# 4) PROXY: graph stats through the proxy
try:
    st, body = get(BASE + "/api/v1/graph/stats")
    data = json.loads(body)
    results.append(("Proxy -> /graph/stats", st == 200 and data.get("total_nodes", 0) > 0,
                    f"nodes={data.get('total_nodes')} links={data.get('total_links')} "
                    f"types={data.get('entity_type_count')} src={data.get('data_source')}"))
except Exception as e:
    results.append(("Proxy -> /graph/stats", False, str(e)[:200]))

# 5) PROXY: disease search through the proxy
try:
    st, body = get(BASE + "/api/v1/disease/search?q=%E9%AB%98%E8%A1%80%E5%8E%8B&page_size=3")
    data = json.loads(body)
    results.append(("Proxy -> /disease/search", st == 200 and data.get("total", 0) > 0,
                    f"total={data.get('total')} first={data['items'][0]['name'] if data.get('items') else None}"))
except Exception as e:
    results.append(("Proxy -> /disease/search", False, str(e)[:200]))

# 6) backend docs reachable directly
try:
    st, body = get("http://127.0.0.1:8000/docs")
    results.append(("Backend /docs (Swagger)", st == 200 and "swagger" in body.lower(),
                    f"status={st} len={len(body)}"))
except Exception as e:
    results.append(("Backend /docs (Swagger)", False, str(e)[:200]))

# 7) OpenAPI schema completeness
try:
    st, body = get("http://127.0.0.1:8000/openapi.json")
    spec = json.loads(body)
    paths = [p for p in spec["paths"] if p.startswith("/api/v1")]
    results.append(("OpenAPI /api/v1 paths", len(paths) >= 25,
                    f"{len(paths)} endpoints, {len(spec.get('components', {}).get('schemas', {}))} schemas"))
except Exception as e:
    results.append(("OpenAPI /api/v1 paths", False, str(e)[:200]))

print("=" * 78)
for name, ok, extra in results:
    print(("  [OK]   " if ok else "  [FAIL] ") + name + ("  :: " + str(extra) if extra else ""))
print("=" * 78)
fails = [r for r in results if not r[1]]
print(f"{len(results) - len(fails)}/{len(results)} passed")
sys.exit(1 if fails else 0)
