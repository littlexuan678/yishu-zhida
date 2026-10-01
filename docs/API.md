# 医数智答 · 智愈医典 —— 后端 API 文档

> **产品定位**：基于「医疗大数据 + 知识图谱 + RAG 大模型」的专业医疗知识问答平台
> **接口版本**：`v1`　|　**服务版本**：`1.0.0`　|　**基础路径（Base URL）**：`http://127.0.0.1:8000/api/v1`
> **交互文档**：Swagger UI `http://127.0.0.1:8000/docs`　|　ReDoc `http://127.0.0.1:8000/redoc`　|　OpenAPI `http://127.0.0.1:8000/openapi.json`

> ### ⚠️ 重要免责声明
> **本系统为演示原型。所有 AI 输出仅供参考，不能替代执业医师诊断。**
> 出现急危重症表现（剧烈胸痛、呼吸困难、意识障碍、大出血等）请立即拨打 120 或前往急诊。
> 该声明在系统全部接口的响应体（`disclaimer` 字段）与前端页面常驻展示。

---

## 1. 概述

医数智答后端是「智愈医典」医疗大数据知识问答系统的服务端实现，基于 **FastAPI + Uvicorn** 提供异步 REST 接口，向下屏蔽 Neo4j 5.8 图数据库与内存降级实现（`MemoryGraphStore`）的差异，向上为 Vue3 前端提供四大业务模块的稳定数据契约：

| 模块 | 路由前缀 | 定位 |
|------|---------|------|
| 智能问答 | `/qa` | 可解释的「AI 医生」：意图识别 → 实体链接 → 图谱检索 → 医疗专属 RAG → 幻觉守卫 |
| 疾病查询 | `/disease` | 精准权威的「掌上医典」：疾病检索、6 大结构化板块详情、权威来源溯源 |
| 知识图谱 | `/graph` | 医学知识的「全景地图」：子图可视化、实体检索、图谱统计、受控 Cypher、规则推理 |
| 数据分析 | `/analytics` | 预见未来的「健康预警」：数据看板图表 + 疾病风险预测 + 个性化干预建议 |

全部请求与响应模型集中定义于 `backend/app/schemas.py`（**前后端唯一数据真源**），前端 `src/api/*.js` 的字段与该文件严格一一对应。

**当前部署实测状态**（离线演示模式，无 Neo4j、无 PyTorch、无外网）：

- 内存图谱：**567 个节点 / 1970 条关系 / 59 个疾病**，Neo4j 状态 `fallback_memory`；
- 医疗 RAG prompt 模板库：**104 个模板 / 16 类意图**（启动自检通过）；
- 大模型 provider：`template`（完全离线模板生成，不调用任何外部大模型）；
- NLP 后端：实体识别 `dictionary`、关系抽取 `rule`、意图分类 `rule`。

---

## 2. 通用约定

| 项目 | 说明 |
|------|------|
| 基础路径 | 除应用根路径 `GET /` 外，所有接口均挂在 `/api/v1` 之下 |
| 请求编码 | UTF-8；`POST` 请求使用 `Content-Type: application/json` |
| CORS | `allow_origins` 取自配置 `CORS_ORIGINS`（默认 `http://127.0.0.1:5173,http://localhost:5173`），`allow_credentials=True`、`allow_methods=*`、`allow_headers=*`，并暴露 `X-Process-Time-Ms` |
| 限流 | `slowapi`，默认 **60 次/分钟/IP**（`RATE_LIMIT_PER_MINUTE`，键为客户端 IP）；未安装 `slowapi` 时降级为不限流并打印告警 |
| 响应头 | `X-Process-Time-Ms`（单次请求耗时）、`X-Powered-By: Zhiyu-Medical-Codex`、`X-Content-Type-Options: nosniff`、`X-Frame-Options: SAMEORIGIN`、`Referrer-Policy: no-referrer` |
| 慢请求日志 | 单请求耗时 > 3000 ms 时写入 WARNING 日志 |
| 降级原则 | 任何子组件（Neo4j / PyTorch / 大模型 / PyYAML）缺失都不导致接口 500，而是返回可用数据并在日志中告警 |
| 免责声明 | `QAResponse.disclaimer`、`HealthResponse`、`/info`、`/` 等接口均返回免责声明文本 |
| 时间字段 | 统一使用服务器本地时区的 ISO-8601 字符串（`datetime.now().astimezone().isoformat(timespec="seconds")`），如 `2026-10-01T15:39:57+08:00` |

---

## 3. 接口总览

### 3.1 系统（4 个）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/`（应用根，无 `/api/v1` 前缀） | 服务根路径：名称、版本、运行状态、文档地址、免责声明 |
| `GET` | `/api/v1/` | 路由根路径：返回四大业务模块的入口接口列表 |
| `GET` | `/api/v1/health` | 健康检查：Neo4j / LLM / RAG 模板库 / 模型加载状态 |
| `GET` | `/api/v1/info` | 系统概览：四层架构、四大功能、三大创新点、量化指标 |

### 3.2 智能问答（6 个）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/qa/ask` | 医疗知识问答主接口（可解释、可溯源） |
| `POST` | `/qa/ask/stream` | SSE 流式问答（`meta` → `token` → `done` / `error`） |
| `GET` | `/qa/prompt-templates` | 列出 104 个医疗专属 RAG prompt 模板 |
| `GET` | `/qa/history/{session_id}` | 获取指定会话的多轮历史 |
| `DELETE` | `/qa/history/{session_id}` | 清空会话历史（隐私合规） |
| `GET` | `/qa/info` | 问答链路各组件状态（调试 / 答辩展示） |

### 3.3 疾病查询（7 个）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/disease/hot-keywords` | 热门搜索关键词 |
| `GET` | `/disease/search` | 疾病检索（关键词 / 自然语言，卡片式返回） |
| `GET` | `/disease/{disease_id}` | 疾病详情（定义 / 病因 / 症状 / 诊断 / 治疗 / 预后 6 大板块） |
| `GET` | `/disease/{disease_id}/related` | 关联疾病（并发症 / 鉴别诊断） |
| `GET` | `/disease/{disease_id}/sources` | 疾病知识来源（100% 可溯源） |
| `GET` | `/disease/{disease_id}/facts` | 疾病三元组事实（按关系类型聚合） |
| `POST` | `/disease/{disease_id}/refresh-source` | 触发知识增量更新（爬虫 → 抽取 → 写图编排） |

### 3.4 知识图谱（10 个）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/graph/stats` | 图谱统计（节点数 / 关系数 / 实体类型数） |
| `GET` | `/graph/entity-types` | 8 类实体类型与配色图例 |
| `GET` | `/graph/search` | 实体检索（搜索框输入联想） |
| `GET` | `/graph/subgraph` | 子图查询（D3.js 力导向图数据源） |
| `GET` | `/graph/evidence` | 关系溯源证据（来源文献 + 置信度） |
| `GET` | `/graph/triples` | 实体三元组列表 |
| `GET` | `/graph/reasoning` | 图谱推理补全（Jena 规则 + GAT/GCN 链接预测） |
| `GET` | `/graph/path` | 多跳路径解释（可解释推理链路） |
| `POST` | `/graph/cypher` | 受控 Cypher 查询（只读白名单校验） |
| `GET` | `/graph/schema` | 图谱 Schema（节点标签 / 关系类型 / 约束 / 索引） |

> 注：其中 8 个为 `GET` 只读查询（`stats` / `entity-types` / `search` / `subgraph` / `evidence` / `triples` / `reasoning` / `path`），另有 `POST /graph/cypher`（受控只读 Cypher）与 `GET /graph/schema`（Schema 元数据）。

### 3.5 数据分析与健康预警（11 个）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/analytics/overview` | 看板概览 4 项核心指标 |
| `GET` | `/analytics/node-type-pie` | 各类节点数量统计（环形饼图） |
| `GET` | `/analytics/category-bar` | 一级分类下的疾病数量（柱状图） |
| `GET` | `/analytics/infectious-gauge` | 传染性疾病比例（环形图） |
| `GET` | `/analytics/symptom-top` | 高频症状 TopN（横向柱状图） |
| `GET` | `/analytics/department-distribution` | 科室疾病分布（饼图） |
| `POST` | `/analytics/risk/predict` | 疾病风险预测（脱敏病史 → 风险概率 + 因子贡献度） |
| `POST` | `/analytics/risk/intervene` | 个性化健康干预建议（饮食 / 运动 / 体检 / 生活方式） |
| `GET` | `/analytics/risk/model-info` | 风险预测模型信息（AUC / 覆盖率 / 特征） |
| `GET` | `/analytics/charts` | 看板聚合接口（overview + 5 张图表，一次请求） |
| `GET` | `/analytics/info` | 数据分析服务信息 |

---

## 4. 系统接口

### 4.1 `GET /` —— 应用根路径

**用途**：无须前缀即可确认服务存活，返回应用名、版本、运行时长、文档入口与免责声明。

**参数**：无。

**请求示例**

```http
GET http://127.0.0.1:8000/
```

**响应示例**

```json
{"name": "医数智答·智愈医典", "product": "智愈医典", "version": "1.0.0", "status": "running", "docs": "/docs", "api_prefix": "/api/v1", "uptime_seconds": 128.4, "server_time": "2026-10-01T15:42:05+08:00", "disclaimer": "本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。"}
```

### 4.2 `GET /api/v1/` —— 路由根路径

**用途**：返回四大业务模块的代表性入口，便于联调时快速定位。

**参数**：无。

**请求示例**

```bash
curl http://127.0.0.1:8000/api/v1/
```

**响应示例**

```json
{"message": "医数智答·智愈医典 API", "product": "智愈医典", "docs": "/docs", "redoc": "/redoc", "health": "/api/v1/health", "info": "/api/v1/info", "endpoints": ["/api/v1/qa/ask", "/api/v1/disease/search", "/api/v1/graph/subgraph", "/api/v1/analytics/overview"]}
```

### 4.3 `GET /health` —— 健康检查

**用途**：聚合各组件状态，供运维与答辩演示判断系统是否处于「真实 Neo4j / 内存降级」模式。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**响应示例**（离线演示模式实测）

```json
{
  "app": "医数智答·智愈医典",
  "version": "1.0.0",
  "status": "ok",
  "neo4j": "fallback_memory",
  "neo4j_uri": "bolt://127.0.0.1:7687",
  "llm": "template",
  "rag_templates": 104,
  "rag_validate": {"total": 104, "missing_required": [], "placeholders_used": ["conversation_slots", "department_list", "entities", "entities_echo", "history", "intent_cn", "kg_context", "main_entity", "no_data_hint", "question", "risk_features", "user_profile"], "ok": true},
  "models_loaded": {"ner_bert_bilstm_crf": false, "pcnn_medical_attention": false, "intent_textcnn": false},
  "server_time": "2026-10-01T15:42:05+08:00",
  "disclaimer": "本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。"
}
```

> 字段说明：`neo4j` 取值 `connected | fallback_memory | error`；`llm` 为 provider 名（`openai_compatible` / `ollama` / `template`）或 `error`；`models_loaded` 中三项为「模型权重是否真实加载」的布尔标志（离线环境全为 `false` 表示已按设计降级到词典 / 规则实现）。

### 4.4 `GET /info` —— 系统概览

**用途**：一次性返回产品定位、四层架构描述、四大功能、三大创新点与量化指标（答辩展示页数据源）。

**参数**：无。

**响应示例**（节选关键字段，结构与实测一致）

```json
{
  "name": "医数智答·智愈医典",
  "product": "智愈医典",
  "version": "1.0.0",
  "track": "大模型与智能体应用赛道",
  "positioning": "基于医疗大数据 + 知识图谱 + RAG 大模型的专业医疗知识问答平台",
  "architecture": {"数据层": "Scrapy 爬虫 + PubMed E-utilities API → 清洗整合 → 全链路脱敏加密(AES-256-GCM)", "知识图谱层": "BERT-BiLSTM-CRF 实体识别 + ★改进 PCNN(医疗关键词注意力)★ 关系抽取 + Neo4j 5.8 + Apache Jena 规则引擎 + GAT/GCN 图谱补全", "LLM增强层": "TextCNN 意图分类 + BERT 实体链接 + ★医疗专属 RAG 提示工程(100+模板)★ + Llama 3 指令微调 + LangChain 编排", "应用层": "FastAPI + Uvicorn 异步接口 + Vue3 + Element Plus + D3.js(图谱可视化) + PyEcharts(数据图表)"},
  "features": [{"key": "graph", "name": "知识图谱可视化", "desc": "医学知识的「全景地图」", "api": "/api/v1/graph/subgraph"}, {"key": "disease", "name": "智能疾病查询", "desc": "精准权威的「掌上医典」", "api": "/api/v1/disease/search"}, {"key": "qa", "name": "智能问答交互", "desc": "可解释的「AI 医生」", "api": "/api/v1/qa/ask"}, {"key": "analytics", "name": "多维数据分析", "desc": "预见未来的「健康预警」", "api": "/api/v1/analytics/risk/predict"}],
  "innovations": [{"id": 1, "name": "改进 PCNN 关系抽取算法", "detail": "在传统 PCNN 基础上增加医疗领域关键词注意力机制，重点关注「症状」「治疗」类核心关键词，医疗关系抽取召回率提升 8.3%", "code": "backend/app/kg_layer/attention.py::MedicalKeywordAttention"}, {"id": 2, "name": "医疗专属 RAG 提示工程", "detail": "整理 104 个医疗问答模板，将知识图谱三元组与用户问题结合构造专属提示词，约束大模型输出，医疗问答准确率提升 12.1%", "code": "backend/app/llm_layer/rag/prompt_library.yaml"}, {"id": 3, "name": "轻量化知识图谱与可视化整合设计", "detail": "简化图谱节点层级，设计「一键检索 + 详情联动」功能，平衡专业性与易用性，适配基层医疗与普通用户", "code": "frontend/src/components/GraphCanvas.vue"}],
  "metrics": {"knowledge_entities_target": "≥10万", "knowledge_relations_target": "≥50万", "knowledge_update_cycle": "≤24h", "qa_accuracy_target": "≥85%", "semantic_parse_latency_target": "≤500ms", "disease_prediction_coverage": "1000+", "prediction_auc_target": "≥0.9", "prompt_templates": 104},
  "disclaimer": "本系统为演示原型。所有 AI 输出仅供参考，不能替代执业医师诊断。"
}
```

---

## 5. 智能问答模块 `/qa`

### 5.1 `POST /qa/ask` —— 医疗知识问答

**用途**：完整 RAG 主链路：`TextCNN 意图识别 → 实体识别与实体链接 → 知识图谱五路混合检索 → 上下文构建 → 医疗专属 Prompt 渲染 → 大模型生成（可降级） → 幻觉守卫校验`。响应包含完整**推理链路**（`reasoning_trace`）与**知识溯源**（`evidences`），答案中每条陈述均带 `[KG-n]` 引用编号。

**请求体参数**（模型：`QARequest`）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `question` | string | 是 | — | 用户自然语言医疗提问，长度 1~500；两端空白会被剔除，纯空白触发 422 |
| `session_id` | string | 否 | `"default"` | 会话 ID，用于多轮槽位继承与历史归档 |
| `top_k` | integer | 否 | `12` | RAG 检索三元组条数，范围 1~50 |
| `max_hops` | integer | 否 | `2` | 图谱多跳推理深度，范围 1~3 |
| `explain` | boolean | 否 | `true` | 是否返回推理链路 `reasoning_trace`（`false` 时为空数组） |
| `use_llm` | boolean | 否 | `true` | `false` 则仅返回基于知识图谱的结构化答案（不调用大模型） |

**请求示例**

```json
{"question": "肺栓塞应该挂什么科？", "session_id": "demo-001", "top_k": 12, "max_hops": 2, "explain": true, "use_llm": true}
```

**响应示例**（`QAResponse`，`provider=template` 离线模式下 `llm_model` 为 `local-template-rag`；`evidences` 实测共 12 条，下面完整展示前 2 条，KG-3 ~ KG-12 结构与之相同——`sources` 均为该疾病节点挂载的 2 条文献，故不再重复列出）

```json
{
  "question": "肺栓塞应该挂什么科？",
  "answer": "## 基于知识图谱的回答（主实体：肺栓塞）\n\n**识别到的医学实体**：肺栓塞(Disease/疾病)\n\n### 症状\n- 知识库记录的相关症状包括：突发性呼吸困难 [KG-7]、胸痛 [KG-8]、咯血 [KG-9]、心悸 [KG-10]。\n\n### 检查\n- 建议的检查项目包括：肺动脉CT血管成像 [KG-11]。\n\n### 科室\n- 建议就诊科室：内科 [KG-1]、呼吸内科 [KG-2]。\n\n### 鉴别诊断\n- 需要鉴别的疾病包括：社区获得性肺炎 [KG-5]、急性心肌梗死 [KG-6]。\n\n### 易感人群\n- 易感人群：肥胖人群 [KG-3]、女性 [KG-4]。\n\n### 安全提示\n- 【一般就医指引】本系统提供医学知识科普与就医指引，任何不适请及时前往正规医疗机构由执业医师面诊。\n\n### 就医建议\n- 以上内容来自知识图谱事实，编号 [KG-n] 与图谱三元组一一对应，可在溯源面板查看原始文献。\n- 若症状持续、加重或出现急症信号，请立即前往正规医疗机构就诊。\n\n本回答由 AI 生成，仅供参考，不能替代执业医师诊断。",
  "answer_html": "<h4>基于知识图谱的回答（主实体：肺栓塞）</h4><br><br><strong>识别到的医学实体</strong>：肺栓塞(Disease/疾病)<br><br><h4>症状</h4><br>· 知识库记录的相关症状包括：突发性呼吸困难 <sup class=\"kg-cite\" data-cite=\"KG-7\">[KG-7]</sup>、胸痛 <sup class=\"kg-cite\" data-cite=\"KG-8\">[KG-8]</sup>、咯血 <sup class=\"kg-cite\" data-cite=\"KG-9\">[KG-9]</sup>、心悸 <sup class=\"kg-cite\" data-cite=\"KG-10\">[KG-10]</sup>。<br><br><h4>检查</h4><br>· 建议的检查项目包括：肺动脉CT血管成像 <sup class=\"kg-cite\" data-cite=\"KG-11\">[KG-11]</sup>。<br><br><h4>科室</h4><br>· 建议就诊科室：内科 <sup class=\"kg-cite\" data-cite=\"KG-1\">[KG-1]</sup>、呼吸内科 <sup class=\"kg-cite\" data-cite=\"KG-2\">[KG-2]</sup>。",
  "intent": {"label": "department_query", "label_cn": "科室导诊", "confidence": 0.943, "method": "rule_fallback", "scores": {"disease_query": 0.0201, "symptom_consult": 0.0201, "treatment_query": 0.0168, "department_query": 0.943}},
  "entities": [{"text": "肺栓塞", "type": "Disease", "label": "疾病", "kg_id": "Disease:肺栓塞", "kg_name": "肺栓塞", "score": 0.999, "method": "dictionary", "start": 0, "end": 3}],
  "evidences": [
    {
      "triple_id": "KG-1",
      "head": "肺栓塞",
      "head_type": "Disease",
      "relation": "BELONGS_TO",
      "relation_label": "科室",
      "tail": "内科",
      "tail_type": "Department",
      "confidence": 0.95,
      "sources": [
        {"source_type": "guideline", "pmid": null, "doi": null, "title": "《肺血栓栓塞症诊治与预防指南》", "journal": "中华结核和呼吸杂志", "year": 2018, "authors": null, "url": "https://www.cma.org.cn/", "authority": "A", "retrieved_at": null},
        {"source_type": "pubmed", "pmid": "31573352", "doi": "10.1093/eurheartj/ehz405", "title": "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism developed in collaboration with the European Respiratory Society", "journal": "European Heart Journal", "year": 2020, "authors": null, "url": "https://pubmed.ncbi.nlm.nih.gov/31573352/", "authority": "B", "retrieved_at": null}
      ]
    },
    {
      "triple_id": "KG-2",
      "head": "肺栓塞",
      "head_type": "Disease",
      "relation": "BELONGS_TO",
      "relation_label": "科室",
      "tail": "呼吸内科",
      "tail_type": "Department",
      "confidence": 0.95,
      "sources": [
        {"source_type": "guideline", "pmid": null, "doi": null, "title": "《肺血栓栓塞症诊治与预防指南》", "journal": "中华结核和呼吸杂志", "year": 2018, "authors": null, "url": "https://www.cma.org.cn/", "authority": "A", "retrieved_at": null},
        {"source_type": "pubmed", "pmid": "31573352", "doi": "10.1093/eurheartj/ehz405", "title": "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism developed in collaboration with the European Respiratory Society", "journal": "European Heart Journal", "year": 2020, "authors": null, "url": "https://pubmed.ncbi.nlm.nih.gov/31573352/", "authority": "B", "retrieved_at": null}
      ]
    }
  ],
  "kg_context": "【知识图谱事实】\n[KG-1] 肺栓塞 —科室→ 内科（置信度 0.95）\n[KG-2] 肺栓塞 —科室→ 呼吸内科（置信度 0.95）\n[KG-3] 肺栓塞 —易感人群→ 肥胖人群（置信度 0.95）\n（完整上下文按槽位组织，共 12 条三元组，此处节选前 3 条）",
  "reasoning_trace": [
    {"step": 1, "action": "intent_classify", "title": "意图识别", "detail": "医疗触发词规则（模型降级） → department_query（科室导诊），置信度 0.94", "elapsed_ms": 1.4, "payload": {"label": "department_query", "label_cn": "科室导诊", "confidence": 0.943, "method": "rule_fallback", "scores": {"disease_query": 0.0201, "symptom_consult": 0.0201, "treatment_query": 0.0168, "department_query": 0.943}}},
    {"step": 2, "action": "entity_link", "title": "实体识别与实体链接", "detail": "「肺栓塞」→ Disease/肺栓塞(score 0.98, dictionary)", "elapsed_ms": 77.8, "payload": {"mentions": 1, "entities": {"diseases": ["肺栓塞"]}}},
    {"step": 3, "action": "kg_retrieve", "title": "知识图谱检索", "detail": "召回 12 条三元组（直接事实 11 / 反向检索 0 / 多跳 0 / 推理补全 1）", "elapsed_ms": 59.1, "payload": {"facts": 12, "stats": {"kg_1hop": 11, "kg_2hop": 0, "kg_reasoned": 1, "kg_similar": 0, "reverse": 0}, "semantic_parse_total_ms": 138.3, "top_k": 12, "max_hops": 2}},
    {"step": 4, "action": "rag_generate", "title": "构建 RAG 上下文", "detail": "按槽位组织 12 条三元组，命中 0 条文献片段", "elapsed_ms": 2.02, "payload": {"slots": ["症状", "检查", "科室", "鉴别诊断", "易感人群"], "main_entity": "肺栓塞", "kg_context_chars": 986}},
    {"step": 5, "action": "rag_generate", "title": "医疗专属 Prompt 渲染", "detail": "命中模板 department_query_core_v1（科室导诊（核心模板）），prompt 1786 字", "elapsed_ms": 0.21, "payload": {"template_id": "department_query_core_v1", "prompt_chars": 1786}},
    {"step": 6, "action": "rag_generate", "title": "大模型生成（Llama 3 / 模板降级）", "detail": "本地 KG 模板回答 475 字（模型返回失败（provider=template，由 rag_engine 使用本地模板生成器））", "elapsed_ms": 0.35, "payload": {"model": "local-template-rag", "fallback": true, "note": "模型返回失败（provider=template，由 rag_engine 使用本地模板生成器）"}},
    {"step": 7, "action": "guard_check", "title": "幻觉守卫与事实校验", "detail": "幻觉守卫：事实校验通过：引用编号完整、无越界实体、无剂量与诊断越权表述", "elapsed_ms": 0.12, "payload": {"passed": true, "score": 0.9, "summary": "幻觉守卫：事实校验通过：引用编号完整、无越界实体、无剂量与诊断越权表述", "suggested_action": "accept", "violations": [], "stats": {}}}
  ],
  "confidence": 0.908,
  "prompt_template_id": "department_query_core_v1",
  "prompt_used": "<|角色|>\n你是\"智愈医典\"医疗知识问答系统的 AI 医生助手，负责**导诊分诊**。\n\n<|用户问题|>\n肺栓塞应该挂什么科？\n\n<|实体识别|>肺栓塞(Disease/疾病)\n<|候选科室|>内科、呼吸内科\n<|知识图谱事实（唯一可信来源）|>\n【知识图谱事实】…（完整渲染结果，此处省略模板余下内容与全局硬约束）",
  "guard_result": {"passed": true, "score": 0.9, "summary": "幻觉守卫：事实校验通过：引用编号完整、无越界实体、无剂量与诊断越权表述", "suggested_action": "accept", "violations": [], "stats": {}},
  "latency_ms": {"intent": 1.4, "entity_link": 77.8, "kg_query": 59.1, "semantic_parse_total": 138.3, "retrieve": 2.02, "llm": 0.56, "guard": 0.12, "total": 141.0},
  "llm_model": "local-template-rag",
  "session_id": "demo-001",
  "disclaimer": "本回答由 AI 生成，仅供参考，不能替代执业医师诊断。",
  "related_questions": ["肺栓塞的症状有哪些？", "肺栓塞有哪些典型表现？", "肺栓塞的治疗方法是什么？", "肺栓塞怎么治疗？", "肺栓塞常用什么药物？"]
}
```

> **可解释性与溯源要点**
> - `reasoning_trace[].action` 取值集合：`intent_classify | entity_link | cypher_query | kg_retrieve | rag_generate | guard_check`（另有 `template_generate`，仅在 RAG 引擎不可用的纯图谱降级链路中出现）；
> - `evidences[].triple_id` 与 `answer` 中的 `[KG-n]` 一一对应，前端点击角标即可回溯到 `:Source` 文献；
> - `latency_ms.semantic_parse_total = intent + entity_link + kg_query`，超过 500 ms 会写 WARNING 日志（PPT 指标 ≤500ms）；
> - 若 RAG 引擎整体不可用，`QAService` 会退化为纯图谱结构化回答，此时 `prompt_template_id="graph_only_fallback"`、`llm_model="graph-only"`、`guard_result.summary="纯图谱输出，未经过大模型，无幻觉风险"`；
> - 示例中的 `latency_ms` 取自本轮冷启动实测的聚合值（`semantic_parse_total=138.3 ms`、`total=141.0 ms`），其中 `kg_query` 即日志实测的混合检索耗时 59.1 ms；日志未逐项记录意图与实体链接耗时，示例按差值分配，故首轮数值偏大（含词典与模板库的进程内冷启动开销）。

**错误码**：`422`（问题为空或超长、`top_k`/`max_hops` 越界）、`500`（问答服务内部异常，`detail` 为 `问答服务异常：...`）。

### 5.2 `POST /qa/ask/stream` —— SSE 流式问答

**用途**：以 Server-Sent Events 逐事件推送问答结果，供前端实现「打字机」效果。实现方式为：**先完整执行一次 `QAService.ask`（非流式）**，再按标点切分答案逐段推送，从而保证流式过程中断时前端仍可拿到完整的意图、实体与证据。

**请求体参数**：与 `POST /qa/ask` 完全相同（模型 `QARequest`）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `question` | string | 是 | — | 用户问题，1~500 字 |
| `session_id` | string | 否 | `"default"` | 会话 ID |
| `top_k` | integer | 否 | `12` | 检索三元组条数（1~50） |
| `max_hops` | integer | 否 | `2` | 多跳深度（1~3） |
| `explain` | boolean | 否 | `true` | 是否返回推理链路 |
| `use_llm` | boolean | 否 | `true` | 是否调用大模型 |

**响应头**

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
Connection: keep-alive
```

**SSE 事件协议**

| 事件名 | 触发时机 | `data` 字段 |
|--------|---------|------------|
| `meta` | 链路执行完成、开始推流时（首个事件） | `intent`（`IntentResult` 对象）、`entities`（`LinkedEntity` 数组）、`prompt_template_id`、`confidence` |
| `token` | 每遇到 `。！？\n；;` 切分出一段文本（含末段余量） | `t`：该段文本（字符串） |
| `done` | 全部 token 推送完毕 | `evidences`、`reasoning_trace`、`latency_ms`、`related_questions`、`disclaimer` |
| `error` | 链路抛出异常 | `message`：错误描述（如 `问答服务异常：...`） |

**请求示例**

```json
{"question": "肺栓塞应该挂什么科？", "session_id": "sse-demo", "top_k": 12, "explain": true}
```

**响应示例（原始 SSE 报文）**

```text
event: meta
data: {"intent": {"label": "department_query", "label_cn": "科室导诊", "confidence": 0.943, "method": "rule_fallback", "scores": {"disease_query": 0.0201, "symptom_consult": 0.0201, "treatment_query": 0.0168, "department_query": 0.943}}, "entities": [{"text": "肺栓塞", "type": "Disease", "label": "疾病", "kg_id": "Disease:肺栓塞", "kg_name": "肺栓塞", "score": 0.999, "method": "dictionary", "start": 0, "end": 3}], "prompt_template_id": "department_query_core_v1", "confidence": 0.908}

event: token
data: {"t": "## 基于知识图谱的回答（主实体：肺栓塞）\n\n**识别到的医学实体**：肺栓塞(Disease/疾病)\n\n### 症状\n- 知识库记录的相关症状包括：突发性呼吸困难 [KG-7]、胸痛 [KG-8]、咯血 [KG-9]、心悸 [KG-10]。\n"}

event: token
data: {"t": "\n### 科室\n- 建议就诊科室：内科 [KG-1]、呼吸内科 [KG-2]。\n"}

event: done
data: {"evidences": [{"triple_id": "KG-1", "head": "肺栓塞", "head_type": "Disease", "relation": "BELONGS_TO", "relation_label": "科室", "tail": "内科", "tail_type": "Department", "confidence": 0.95, "sources": [{"source_type": "guideline", "pmid": null, "doi": null, "title": "《肺血栓栓塞症诊治与预防指南》", "journal": "中华结核和呼吸杂志", "year": 2018, "authors": null, "url": "https://www.cma.org.cn/", "authority": "A", "retrieved_at": null}, {"source_type": "pubmed", "pmid": "31573352", "doi": "10.1093/eurheartj/ehz405", "title": "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism developed in collaboration with the European Respiratory Society", "journal": "European Heart Journal", "year": 2020, "authors": null, "url": "https://pubmed.ncbi.nlm.nih.gov/31573352/", "authority": "B", "retrieved_at": null}]}], "reasoning_trace": [{"step": 1, "action": "intent_classify", "title": "意图识别", "detail": "医疗触发词规则（模型降级） → department_query（科室导诊），置信度 0.94", "elapsed_ms": 1.4, "payload": {}}], "latency_ms": {"intent": 1.4, "entity_link": 77.8, "kg_query": 59.1, "semantic_parse_total": 138.3, "retrieve": 2.02, "llm": 0.56, "guard": 0.12, "total": 141.0}, "related_questions": ["肺栓塞的症状有哪些？", "肺栓塞有哪些典型表现？", "肺栓塞的治疗方法是什么？"], "disclaimer": "本回答由 AI 生成，仅供参考，不能替代执业医师诊断。"}
```

**前端接入示例**

```js
const es = await fetch("/api/v1/qa/ask/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "肺栓塞应该挂什么科？", session_id: "sse-demo" }),
});
// 依次处理 meta / token / done / error 四类事件；token 事件的 data.t 直接追加到气泡文本
```

### 5.3 `GET /qa/prompt-templates` —— 医疗 RAG prompt 模板库

**用途**：列出 `prompt_library.yaml` 中全部 **104 个**医疗问答模板的元信息（按意图分组），用于答辩展示与调参。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `intent` | string | 否 | `null` | 按意图过滤，如 `department_query`；不传则返回全部模板 |
| `keyword` | string | 否 | `null` | 按名称 / ID / 标签 / 描述关键词过滤（不区分大小写） |

**请求示例**

```http
GET /api/v1/qa/prompt-templates?intent=department_query&keyword=导诊
```

**响应示例**

```json
{
  "total": 1,
  "intents": ["check_query", "department_query", "diet_lifestyle", "disease_query", "drug_query", "emergency", "fallback", "meta", "multi_turn", "practical", "prevention", "prognosis", "risk_assessment", "safety", "symptom_consult", "treatment_query"],
  "items": [{"id": "department_query_core_v1", "intent": "department_query", "name": "科室导诊（核心模板）", "description": "用户问\"XX挂什么科\" —— PPT 演示中的典型问句", "version": "v1", "tags": ["核心", "导诊", "最常用"], "template_preview": "<|角色|>\n你是\"智愈医典\"医疗知识问答系统的 AI 医生助手，负责**导诊分诊**。\n<|用户问题|>\n{question}\n<|实体识别|>{entities}\n<|候选科室|>{department_list}\n<|知识图谱事实（唯一可信来源）|>\n{kg_context}\n<|回答结构|>\n## 一、建议就诊科室\n第一句直接给出明确科室名称，格式：**建议就诊科室：XX科**\n紧接着说明依据（标注引用编号）。\n## 二、为什…"}]
}
```

> 16 类意图的模板分布（启动日志实测）：`disease_query:12`、`treatment_query:12`、`symptom_consult:11`、`department_query:9`、`risk_assessment:8`、`emergency:7`、`multi_turn:7`、`meta:6`、`drug_query:5`、`practical:5`、`check_query:4`、`diet_lifestyle:4`、`prevention:4`、`fallback:4`、`prognosis:3`、`safety:3`。

### 5.4 `GET /qa/history/{session_id}` —— 获取会话历史

**用途**：返回指定会话的完整消息列表（`user` / `assistant` 成对），供前端恢复对话界面。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `session_id` | string（路径） | 是 | — | 会话 ID；不存在的会话返回空消息列表 |

**请求示例**

```bash
curl "http://127.0.0.1:8000/api/v1/qa/history/demo-001"
```

**响应示例**

```json
{
  "session_id": "demo-001",
  "messages": [
    {"role": "user", "content": "肺栓塞应该挂什么科？", "timestamp": "15:44", "answer_html": "", "evidences": [], "reasoning_trace": [], "confidence": 0.0},
    {
      "role": "assistant",
      "content": "## 基于知识图谱的回答（主实体：肺栓塞）\n\n### 科室\n- 建议就诊科室：内科 [KG-1]、呼吸内科 [KG-2]。\n\n本回答由 AI 生成，仅供参考，不能替代执业医师诊断。",
      "timestamp": "15:44",
      "answer_html": "<h4>基于知识图谱的回答（主实体：肺栓塞）</h4><br>· 建议就诊科室：内科 <sup class=\"kg-cite\" data-cite=\"KG-1\">[KG-1]</sup>、呼吸内科 <sup class=\"kg-cite\" data-cite=\"KG-2\">[KG-2]</sup>。",
      "evidences": [
        {
          "triple_id": "KG-1",
          "head": "肺栓塞",
          "head_type": "Disease",
          "relation": "BELONGS_TO",
          "relation_label": "科室",
          "tail": "内科",
          "tail_type": "Department",
          "confidence": 0.95,
          "sources": [
            {"source_type": "guideline", "pmid": null, "doi": null, "title": "《肺血栓栓塞症诊治与预防指南》", "journal": "中华结核和呼吸杂志", "year": 2018, "authors": null, "url": "https://www.cma.org.cn/", "authority": "A", "retrieved_at": null},
            {"source_type": "pubmed", "pmid": "31573352", "doi": "10.1093/eurheartj/ehz405", "title": "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism developed in collaboration with the European Respiratory Society", "journal": "European Heart Journal", "year": 2020, "authors": null, "url": "https://pubmed.ncbi.nlm.nih.gov/31573352/", "authority": "B", "retrieved_at": null}
          ]
        }
      ],
      "reasoning_trace": [],
      "confidence": 0.908
    }
  ],
  "turns": 1
}
```

> 会话历史上限：`QAService.MAX_HISTORY_PER_SESSION = 40` 条消息（超限时丢弃最旧消息），`turns = len(messages) // 2`。为保证历史记录轻量，当某轮响应的 `reasoning_trace` 条目数 ≥ 20 时，写入历史时会将该字段置为空数组；上面的 `content` / `answer_html` 亦为便于阅读而截断显示。

### 5.5 `DELETE /qa/history/{session_id}` —— 清空会话历史

**用途**：隐私合规接口，删除指定会话在服务端内存中的全部消息。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `session_id` | string（路径） | 是 | — | 待清空的会话 ID；不存在也返回成功 |

**请求示例**

```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/qa/history/demo-001"
```

**响应示例**

```json
{"code": 0, "message": "会话已清空", "session_id": "demo-001"}
```

### 5.6 `GET /qa/info` —— 问答链路组件信息

**用途**：返回问答服务、意图分类器、实体链接器、RAG 引擎的自检信息，用于调试与答辩展示。

**参数**：无。

**请求示例**

```bash
curl http://127.0.0.1:8000/api/v1/qa/info
```

**响应示例**（离线演示模式实测）

```json
{
  "service": {"rag_available": true, "sessions": 0, "max_history_per_session": 40},
  "intent_classifier": {"backend": "rule", "has_torch": false, "fuse_rule": true, "labels": ["disease_query", "symptom_consult", "treatment_query", "department_query"], "label_cn": {"disease_query": "疾病查询", "symptom_consult": "症状咨询", "treatment_query": "治疗咨询", "department_query": "科室导诊"}, "model": "TextCNN(kernel=2,3,4, filters=128, embed=128)"},
  "entity_linker": {"extractor_backend": "dictionary", "alias_count": 64, "colloquial_map_size": 53, "department_alias_size": 27, "symptom_alias_size": 33, "lexicon_size": 567},
  "rag_engine": {
    "engine": "RagEngine",
    "data_source": "memory",
    "graph_nodes": 567,
    "graph_edges": 1970,
    "llm": {"provider": "template", "client_available": true, "client": {"provider": "template", "model": "llama3:8b-instruct-q4_K_M", "base_url": "http://127.0.0.1:11434/v1", "timeout": 60, "temperature": 0.2, "max_tokens": 1600, "lora_path": null, "has_openai_sdk": false, "has_httpx": false, "mode": "offline_template"}},
    "prompt_templates": 104,
    "retriever": {"available": true, "routed": ["kg_1hop", "kg_1hop_reverse", "kg_2hop", "kg_reasoned", "kg_similar"]},
    "guard": {"available": true, "enabled": true},
    "sessions": 0,
    "rag": {"top_k": 12, "max_hops": 2, "min_confidence": 0.55}
  }
}
```

---

## 6. 疾病查询模块 `/disease`

### 6.1 `GET /disease/hot-keywords` —— 热门搜索关键词

**用途**：返回 PPT 演示中的热门搜索词（按疾病重要性/关系度数排序，并保证默认热词优先）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `topn` | integer | 否 | `6` | 返回条数，范围 1~20 |

**请求示例**

```http
GET /api/v1/disease/hot-keywords?topn=6
```

**响应示例**（实测：默认热词中仅「感冒」「冠心病」存在于种子库，其余按图谱度数补齐）

```json
["感冒", "冠心病", "心力衰竭", "慢性肾脏病", "社区获得性肺炎", "脑卒中"]
```

### 6.2 `GET /disease/search` —— 疾病检索

**用途**：关键词 / 自然语言检索疾病，支持多维命中并按相关度排序：**疾病名精确(1.0) > 名称包含(0.93) > 别名(0.88) > 症状(0.72) > 分类(0.66) > 科室(0.64) > 定义/病因(0.58) > 模糊匹配**，得分低于 0.5 的条目不返回。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `q` | string | 否 | `""` | 搜索关键词（如 `高血压` 或 `头晕 血压高`），最大 120 字；留空返回全部疾病 |
| `page` | integer | 否 | `1` | 页码，≥1 |
| `page_size` | integer | 否 | `6` | 每页条数，1~60 |
| `category1` | string | 否 | `null` | 按一级分类过滤，如 `内科` |

**请求示例**

```http
GET /api/v1/disease/search?q=高血压&page=1&page_size=3
```

**响应示例**（种子库中共 6 个疾病名称含「高血压」，得分均为名称包含档 0.93，再按名称排序，故前 3 条如下）

```json
{
  "keyword": "高血压",
  "total": 6,
  "page": 1,
  "page_size": 3,
  "items": [
    {"disease_id": "D0003", "name": "假性高血压", "category1": "内科", "category2": "心血管内科", "symptoms": ["血压升高", "血压波动", "头晕", "乏力", "晕厥", "颈项板紧", "疲倦"], "treatments": ["药物治疗", "生活方式干预", "健康教育"], "population": "多见于高龄老年人，尤其合并糖尿病、慢性肾脏病或长期血液透析的动脉硬化患者", "is_infectious": false, "has_detail": true},
    {"disease_id": "D0008", "name": "内分泌性高血压", "category1": "内科", "category2": "内分泌科", "symptoms": ["血压升高", "血压波动", "头痛", "心悸", "多汗", "乏力", "夜尿增多", "手抖"], "treatments": ["药物治疗", "手术治疗", "饮食治疗", "健康教育"], "population": "好发于青少年与中青年人群，女性原发性醛固酮增多症患者更为多见", "is_infectious": false, "has_detail": true},
    {"disease_id": "D0001", "name": "原发性高血压", "category1": "内科", "category2": "心血管内科", "symptoms": ["头晕", "头痛", "颈项板紧", "心悸", "乏力", "失眠", "疲倦", "耳鸣"], "treatments": ["药物治疗", "饮食治疗", "运动康复", "生活方式干预"], "population": "多见于中老年人、肥胖者、高盐饮食者、长期精神紧张者及有高血压家族史的人群", "is_infectious": false, "has_detail": true}
  ],
  "hot_keywords": ["感冒", "冠心病", "心力衰竭", "慢性肾脏病", "社区获得性肺炎", "脑卒中"],
  "message": "找到 6 条相关疾病信息",
  "searched_at": "2026-10-01T15:46:12+08:00"
}
```

> 卡片字段截断规则：`symptoms` 最多 12 项、`treatments` 最多 8 项（`GraphService._to_card`）。

### 6.3 `GET /disease/{disease_id}` —— 疾病详情

**用途**：返回疾病 6 大结构化板块（定义 / 病因 / 症状 / 诊断 / 治疗 / 预后）+ 检查项目、常用药物、就诊科室、并发症、鉴别诊断、易感人群与权威来源溯源。`disease_id` 支持疾病 ID（`D0013`）、标准名（`肺栓塞`）、别名与模糊匹配（相似度阈值 0.6）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `disease_id` | string（路径） | 是 | — | 疾病 ID 或疾病名称，最大 80 字 |

**请求示例**

```bash
curl "http://127.0.0.1:8000/api/v1/disease/肺栓塞"
```

**响应示例**（`DiseaseDetail`；`definition` / `cause` / `diagnosis` / `treatment` / `prognosis` 为长文本，此处截断显示）

```json
{
  "disease_id": "D0013",
  "name": "肺栓塞",
  "alias": ["肺动脉栓塞", "pulmonary embolism"],
  "category1": "内科",
  "category2": "呼吸内科",
  "definition": "肺栓塞是以各种栓子阻塞肺动脉或其分支为发病原因的一组疾病或临床综合征的总称，其中肺血栓栓塞症最为常见，占绝大多数。栓子多来源于下肢深静脉或盆腔静脉脱落的血栓，也可来自右心。…（长文本，此处截断）",
  "cause": "主要机制为静脉血栓形成后脱落，随血流阻塞肺动脉。血流淤滞、血管内皮损伤与血液高凝状态是核心环节：长期卧床与制动、大手术、骨折、恶性肿瘤、口服避孕药、妊娠与产褥期、肥胖、吸烟、高龄及遗传性易栓症…（长文本，此处截断）",
  "symptoms": ["突发性呼吸困难", "胸痛", "咯血", "心悸", "晕厥", "咳嗽", "发绀", "颈静脉怒张"],
  "diagnosis": "对突发呼吸困难、胸痛伴低氧血症、心电图示 SⅠQⅢTⅢ 或右心负荷增加、D-二聚体升高者应怀疑本病。确诊依赖肺动脉CT血管成像或肺通气灌注显像…（长文本，此处截断）",
  "checks": ["肺动脉CT血管成像", "D-二聚体", "超声心动图", "心电图", "血气分析", "下肢静脉超声", "胸部X线", "凝血功能"],
  "treatment": "治疗包括抗凝、溶栓与介入手术。血流动力学稳定的患者应立即启动抗凝，可选用低分子肝素、利伐沙班或达比加群酯，多数患者疗程至少 3 个月…（长文本，此处截断）",
  "treatments": ["药物治疗", "介入治疗", "手术治疗", "氧疗"],
  "drugs": ["低分子肝素", "利伐沙班", "华法林", "阿替普酶", "尿激酶"],
  "department": "呼吸内科",
  "prognosis": "急性肺栓塞 30 天病死率约 5%~15%，高危伴休克者可达 30% 以上，规范抗凝后复发率降至 3% 以下。…（长文本，此处截断）",
  "population": "多见于老年人、肥胖者、近期手术或骨折后长期卧床者、恶性肿瘤患者及妊娠与产褥期女性",
  "complications": ["肺源性心脏病", "继发性肺动脉高压", "心力衰竭", "呼吸衰竭"],
  "differential": ["急性心肌梗死", "社区获得性肺炎", "冠心病"],
  "is_infectious": false,
  "sources": [
    {"source_type": "guideline", "pmid": null, "doi": null, "title": "《肺血栓栓塞症诊治与预防指南》", "journal": "中华结核和呼吸杂志", "year": 2018, "authors": null, "url": "https://www.cma.org.cn/", "authority": "A", "retrieved_at": null},
    {"source_type": "pubmed", "pmid": "31573352", "doi": "10.1093/eurheartj/ehz405", "title": "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism developed in collaboration with the European Respiratory Society", "journal": "European Heart Journal", "year": 2020, "authors": null, "url": "https://pubmed.ncbi.nlm.nih.gov/31573352/", "authority": "B", "retrieved_at": null}
  ],
  "updated_at": "2026-01-15T09:00:00Z"
}
```

**错误响应**（404）

```json
{"detail": "知识库中未找到疾病「肺栓塞晚期」。请尝试使用标准疾病名称，如「原发性高血压」。"}
```

### 6.4 `GET /disease/{disease_id}/related` —— 关联疾病

**用途**：返回并发症（`HAS_COMPLICATION`）与鉴别诊断（`DIFFERENTIAL_WITH`）的疾病卡片列表。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `disease_id` | string（路径） | 是 | — | 疾病 ID 或疾病名称，最大 80 字 |

**请求示例**

```http
GET /api/v1/disease/肺栓塞/related
```

**响应示例**

```json
{
  "disease_id": "D0013",
  "complications": [{"disease_id": "D0007", "name": "肺源性心脏病", "category1": "内科", "category2": "心血管内科", "symptoms": ["咳嗽", "咳痰", "活动后气促", "呼吸困难", "心悸", "下肢水肿", "发绀", "颈静脉怒张"], "treatments": ["药物治疗", "氧疗", "无创呼吸机治疗", "康复训练"], "population": "多见于中老年长期吸烟者及慢性阻塞性肺疾病、支气管哮喘、肺结核患者", "is_infectious": false, "has_detail": true}, {"disease_id": "D0006", "name": "继发性肺动脉高压", "category1": "内科", "category2": "心血管内科", "symptoms": ["活动后气促", "呼吸困难", "乏力", "胸闷", "晕厥", "下肢水肿", "心悸", "胸痛"], "treatments": ["药物治疗", "氧疗", "手术治疗", "康复训练"], "population": "好发于有慢性心肺疾病基础的中老年人群，女性结缔组织病相关者更为多见", "is_infectious": false, "has_detail": true}],
  "differential": [{"disease_id": "D0005", "name": "急性心肌梗死", "category1": "内科", "category2": "心血管内科", "symptoms": ["胸痛", "心前区疼痛", "胸闷", "呼吸困难", "恶心", "心悸", "晕厥", "畏寒"], "treatments": ["药物治疗", "介入治疗", "手术治疗", "康复训练"], "population": "多见于中老年男性、吸烟者、血脂异常者及合并原发性高血压、2型糖尿病的人群", "is_infectious": false, "has_detail": true}]
}
```

**错误响应**（404）：`{"detail": "知识库中未找到疾病「xxx」"}`

### 6.5 `GET /disease/{disease_id}/sources` —— 疾病知识来源

**用途**：返回该疾病条目挂载的全部权威来源（临床指南 / PubMed 文献 / 教材），含 PMID、DOI 与权威等级；无来源时回退到疾病详情的来源列表。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `disease_id` | string（路径） | 是 | — | 疾病 ID 或疾病名称，最大 80 字 |

**请求示例**

```bash
curl "http://127.0.0.1:8000/api/v1/disease/原发性高血压/sources"
```

**响应示例**

```json
[{"source_type": "guideline", "pmid": null, "doi": null, "title": "《中国高血压防治指南(2024年修订版)》", "journal": "中华心血管病杂志", "year": 2024, "authors": null, "url": "https://www.nccd.org.cn/", "authority": "A", "retrieved_at": null}, {"source_type": "pubmed", "pmid": "32130469", "doi": "10.1097/HJH.0000000000002425", "title": "2020 International Society of Hypertension Global Hypertension Practice Guidelines", "journal": "Hypertension", "year": 2020, "authors": null, "url": "https://pubmed.ncbi.nlm.nih.gov/32130469/", "authority": "B", "retrieved_at": null}]
```

### 6.6 `GET /disease/{disease_id}/facts` —— 疾病三元组事实

**用途**：返回该疾病在图谱中的全部直接关系（症状 / 治疗 / 用药 / 科室 / 检查 / 并发症 / 鉴别 / 易感人群），按关系类型聚合，供疾病详情页的图谱联动使用。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `disease_id` | string（路径） | 是 | — | 疾病 ID 或疾病名称，最大 80 字 |

**请求示例**

```http
GET /api/v1/disease/肺栓塞/facts
```

**响应示例**

```json
{"disease_id": "D0013", "name": "肺栓塞", "facts": {"HAS_SYMPTOM": ["突发性呼吸困难", "胸痛", "咯血", "心悸", "晕厥", "咳嗽", "发绀", "颈静脉怒张"], "TREATED_BY": ["药物治疗", "介入治疗", "手术治疗", "氧疗"], "USES_DRUG": ["低分子肝素", "利伐沙班", "华法林", "阿替普酶", "尿激酶"], "BELONGS_TO": ["内科", "呼吸内科"], "NEEDS_CHECK": ["肺动脉CT血管成像", "D-二聚体", "超声心动图", "心电图", "血气分析", "下肢静脉超声", "胸部X线", "凝血功能"], "HAS_COMPLICATION": ["肺源性心脏病", "继发性肺动脉高压", "心力衰竭", "呼吸衰竭"], "DIFFERENTIAL_WITH": ["急性心肌梗死", "社区获得性肺炎", "冠心病"], "AFFECTS": ["老年人", "孕产妇", "女性", "肥胖人群"]}, "triple_count": 38}
```

**错误响应**（404）：`{"detail": "知识库中未找到疾病「xxx」"}`

### 6.7 `POST /disease/{disease_id}/refresh-source` —— 触发知识增量更新

**用途**：对应「知识更新周期 ≤24h」指标，返回知识增量更新任务的编排说明（**演示原型不实际发起外网请求**）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `disease_id` | string（路径） | 是 | — | 疾病 ID 或疾病名称，最大 80 字 |

**请求示例**

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/disease/肺栓塞/refresh-source"
```

**响应示例**

```json
{"code": 0, "message": "知识增量更新任务已编排", "disease": "肺栓塞", "pipeline": ["1. Scrapy/PubMed E-utilities 增量采集（关键词：肺栓塞，时间范围：近 24h 新入库）", "2. 数据清洗与去重（preprocess.py）", "3. 全链路脱敏加密（privacy.py，AES-256-GCM）", "4. 实体识别（BERT-BiLSTM-CRF）", "5. 关系抽取（★改进 PCNN + 医疗关键词注意力★）", "6. MERGE 幂等写图（Neo4j，保留高置信度既有事实）", "7. 图谱补全（Apache Jena 规则 + GAT/GCN 链接预测）"], "target_cycle": "≤24h", "cli": "python scripts/crawler_run.py --since 1d && python scripts/load_kg.py --incremental", "note": "演示原型未实际发起外网采集请求。"}
```

**错误响应**（404）：`{"detail": "知识库中未找到疾病「xxx」"}`

---

## 7. 知识图谱模块 `/graph`

### 7.1 `GET /graph/stats` —— 图谱统计

**用途**：知识图谱页面顶部统计卡片的数据源（节点数量 / 关系数量 / 实体类型数），并区分数据来源是 Neo4j 还是内存降级。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```bash
curl http://127.0.0.1:8000/api/v1/graph/stats
```

**响应示例**（内存降级模式实测：567 节点 / 1970 关系 / 7 类实体）

```json
{
  "total_nodes": 567,
  "total_links": 1970,
  "entity_type_count": 7,
  "entity_types": [{"type": "Symptom", "label": "症状", "color": "#31c48d", "count": 195}, {"type": "Drug", "label": "药物", "color": "#E6A23C", "count": 122}, {"type": "Check", "label": "检查项目", "color": "#909399", "count": 116}, {"type": "Disease", "label": "疾病", "color": "#409EFF", "count": 59}, {"type": "Department", "label": "科室", "color": "#F56C6C", "count": 34}, {"type": "Treatment", "label": "治疗方法", "color": "#b37feb", "count": 26}, {"type": "Population", "label": "易感人群", "color": "#F2C037", "count": 15}],
  "disease_count": 59,
  "symptom_count": 195,
  "drug_count": 122,
  "department_count": 34,
  "treatment_count": 26,
  "source_count": 0,
  "data_source": "memory"
}
```

### 7.2 `GET /graph/entity-types` —— 实体类型图例

**用途**：返回 8 类实体（疾病 / 症状 / 科室 / 药物 / 治疗方法 / 检查项目 / 易感人群 / 文献来源）的中文名、配色与当前计数，供前端图例渲染。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```http
GET /api/v1/graph/entity-types
```

**响应示例**

```json
[{"type": "Disease", "label": "疾病", "color": "#409EFF", "count": 59}, {"type": "Symptom", "label": "症状", "color": "#31c48d", "count": 195}, {"type": "Department", "label": "科室", "color": "#F56C6C", "count": 34}, {"type": "Drug", "label": "药物", "color": "#E6A23C", "count": 122}, {"type": "Treatment", "label": "治疗方法", "color": "#b37feb", "count": 26}, {"type": "Check", "label": "检查项目", "color": "#909399", "count": 116}, {"type": "Population", "label": "易感人群", "color": "#F2C037", "count": 15}, {"type": "Source", "label": "文献来源", "color": "#00BCD4", "count": 0}]
```

### 7.3 `GET /graph/search` —— 实体检索（输入联想）

**用途**：图谱搜索框的输入联想，支持精确（1.0）、前缀（0.94）、包含（0.88）、别名（0.8）与相似度匹配，得分低于 0.45 不返回。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `q` | string | 否 | `""` | 实体关键词，如 `血压`，最大 80 字 |
| `limit` | integer | 否 | `20` | 返回条数，1~100 |
| `types` | string | 否 | `null` | 按类型过滤，多个用逗号分隔，如 `Disease,Symptom` |

**请求示例**

```http
GET /api/v1/graph/search?q=栓塞&limit=5&types=Disease
```

**响应示例**

```json
{"keyword": "栓塞", "items": [{"id": "Disease:肺栓塞", "name": "肺栓塞", "type": "Disease", "label": "疾病", "color": "#409EFF", "score": 0.88, "category1": "内科"}], "total": 1}
```

### 7.4 `GET /graph/subgraph` —— 子图查询（D3 渲染数据）

**用途**：以指定实体为中心做 N 跳 BFS 展开，返回 D3.js 力导向图所需的 `nodes` 与 `links`。Neo4j 可用时走变长模式匹配 `MATCH p=(center)-[*1..depth]-(neighbor)`，不可用时走内存 BFS（语义一致）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `entity` | string | 是 | — | 中心实体名，如 `肺栓塞`，1~80 字 |
| `depth` | integer | 否 | `1` | 展开跳数，1~3 |
| `limit` | integer | 否 | `200` | 最大节点数，10~800 |
| `rels` | string | 否 | `null` | 关系类型过滤，逗号分隔，如 `HAS_SYMPTOM,TREATED_BY` |
| `types` | string | 否 | `null` | 实体类型过滤，逗号分隔，如 `Disease,Symptom` |

**请求示例**

```http
GET /api/v1/graph/subgraph?entity=肺栓塞&depth=1&limit=200
```

**响应示例**（实测：`depth=1` 展开 **43 个节点 / 47 条关系 / 7 类实体**，此处展示 4 个节点与 4 条关系）

```json
{
  "nodes": [
    {"id": "Disease:肺栓塞", "name": "肺栓塞", "type": "Disease", "label": "疾病", "color": "#409EFF", "degree": 47, "category1": "内科", "category2": "呼吸内科", "properties": {"alias": ["肺动脉栓塞", "pulmonary embolism"], "category1": "内科", "category2": "呼吸内科", "is_infectious": false, "population": "多见于老年人、肥胖者、近期手术或骨折后长期卧床者、恶性肿瘤患者及妊娠与产褥期女性", "definition": "肺栓塞是以各种栓子阻塞肺动脉或其分支为发病原因的一组疾病或临床综合征的总称，其中肺血栓栓塞症最为常见…", "source_count": 2, "updated_at": "2026-01-15T09:00:00Z"}},
    {"id": "Symptom:突发性呼吸困难", "name": "突发性呼吸困难", "type": "Symptom", "label": "症状", "color": "#31c48d", "degree": 4, "category1": null, "category2": null, "properties": {}},
    {"id": "Drug:阿替普酶", "name": "阿替普酶", "type": "Drug", "label": "药物", "color": "#E6A23C", "degree": 3, "category1": null, "category2": null, "properties": {}},
    {"id": "Department:呼吸内科", "name": "呼吸内科", "type": "Department", "label": "科室", "color": "#F56C6C", "degree": 6, "category1": null, "category2": null, "properties": {}}
  ],
  "links": [
    {"source": "Disease:肺栓塞", "target": "Symptom:突发性呼吸困难", "rel": "HAS_SYMPTOM", "label": "症状", "properties": {"confidence": 0.95, "weight": 1.0, "source_ref": ""}},
    {"source": "Disease:肺栓塞", "target": "Drug:阿替普酶", "rel": "USES_DRUG", "label": "用药", "properties": {"confidence": 0.95, "weight": 1.0, "source_ref": ""}},
    {"source": "Disease:肺栓塞", "target": "Department:呼吸内科", "rel": "BELONGS_TO", "label": "科室", "properties": {"confidence": 0.95, "weight": 1.0, "source_ref": ""}},
    {"source": "Disease:肺栓塞", "target": "Disease:继发性肺动脉高压", "rel": "HAS_COMPLICATION", "label": "并发症", "properties": {"confidence": 0.95, "weight": 1.0, "source_ref": ""}}
  ],
  "node_count": 43,
  "link_count": 47,
  "type_count": 7,
  "center": "肺栓塞",
  "depth": 1,
  "truncated": false,
  "message": ""
}
```

> 未命中实体时不会报错，而是在 `message` 中给出提示：`{"nodes": [], "links": [], "node_count": 0, "link_count": 0, "center": "xxx", "message": "知识库中未找到实体「xxx」，请尝试其他名称"}`。
> `truncated=true` 表示节点数达到 `limit` 被截断。

### 7.5 `GET /graph/evidence` —— 关系溯源证据

**用途**：给定三元组 `(head, rel, tail)`，返回其来源文献（PubMed PMID / DOI / 指南名称）与置信度，是「知识 100% 可溯源」的直接体现。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `head` | string | 是 | — | 头实体名，最大 80 字 |
| `rel` | string | 否 | `null` | 关系英文类型，如 `BELONGS_TO`；不传则返回该实体任意一条关系 |
| `tail` | string | 否 | `null` | 尾实体名；不传则不约束尾实体 |

**请求示例**

```http
GET /api/v1/graph/evidence?head=肺栓塞&rel=BELONGS_TO&tail=呼吸内科
```

**响应示例**

```json
{
  "triple_id": "KG-1",
  "head": "肺栓塞",
  "head_type": "Disease",
  "relation": "BELONGS_TO",
  "relation_label": "科室",
  "tail": "呼吸内科",
  "tail_type": "Department",
  "confidence": 0.95,
  "sources": [
    {"source_type": "guideline", "pmid": null, "doi": null, "title": "《肺血栓栓塞症诊治与预防指南》", "journal": "中华结核和呼吸杂志", "year": 2018, "authors": null, "url": "https://www.cma.org.cn/", "authority": "A", "retrieved_at": null},
    {"source_type": "pubmed", "pmid": "31573352", "doi": "10.1093/eurheartj/ehz405", "title": "2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism developed in collaboration with the European Respiratory Society", "journal": "European Heart Journal", "year": 2020, "authors": null, "url": "https://pubmed.ncbi.nlm.nih.gov/31573352/", "authority": "B", "retrieved_at": null}
  ]
}
```

**错误响应**（404）：`{"detail": "未找到「xxx」的匹配关系"}`　（来源最多返回 3 条，按 `(pmid, doi, title)` 去重）

### 7.6 `GET /graph/triples` —— 实体三元组列表

**用途**：返回指定实体的关联三元组（含入边），按置信度降序，用于 RAG 上下文预览与图谱联动。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `entity` | string | 是 | — | 实体名，最大 80 字 |
| `rels` | string | 否 | `null` | 关系类型过滤，逗号分隔 |
| `limit` | integer | 否 | `30` | 返回条数，1~200 |
| `direction` | string | 否 | `both` | `both` / `out`（仅出边）/ `in`（仅入边） |

**请求示例**

```http
GET /api/v1/graph/triples?entity=肺栓塞&limit=30&direction=both
```

**响应示例**（`total` 为截断后条数；此处展示前 4 条）

```json
{
  "entity": "肺栓塞",
  "total": 30,
  "triples": [
    {"head": "肺栓塞", "head_type": "Disease", "relation": "TREATED_BY", "relation_label": "治疗", "tail": "药物治疗", "tail_type": "Treatment", "confidence": 0.95},
    {"head": "肺栓塞", "head_type": "Disease", "relation": "HAS_COMPLICATION", "relation_label": "并发症", "tail": "继发性肺动脉高压", "tail_type": "Disease", "confidence": 0.95},
    {"head": "肺栓塞", "head_type": "Disease", "relation": "HAS_COMPLICATION", "relation_label": "并发症", "tail": "心力衰竭", "tail_type": "Disease", "confidence": 0.95},
    {"head": "肺栓塞", "head_type": "Disease", "relation": "USES_DRUG", "relation_label": "用药", "tail": "阿替普酶", "tail_type": "Drug", "confidence": 0.95}
  ]
}
```

> `Source` 类型节点不会出现在三元组结果中（溯源节点由 `evidences` 承载）。

### 7.7 `GET /graph/reasoning` —— 图谱推理与补全

**用途**：执行医疗规则推理（共病关联、并发症症状传递、传染性标记等 6 条规则）与图神经网络链接预测，挖掘图谱中隐藏的语义关联，返回新增的潜在三元组及其置信度与推理依据。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `mode` | string | 否 | `hybrid` | `rule`（纯规则）/ `gnn`（纯图神经网络补全）/ `hybrid`（两者融合去重） |
| `limit` | integer | 否 | `50` | 规则推理条数上限（1~300）；`hybrid` 模式下 GNN 条数取 `limit // 2` |

**请求示例**

```http
GET /api/v1/graph/reasoning?mode=rule&limit=50
```

**响应示例**（`mode=rule`；实测「疾病 59 个，规则 6 条」共新增 20 条三元组，此处展示 1 条）

```json
{
  "mode": "rule",
  "total": 20,
  "backend": {"rule_count": 6, "rule_ids": ["R1_shared_symptom_related", "R2_shared_treatment_related", "R3_complication_transitive_symptom", "R4_same_department_comorbidity", "R5_infectious_flag", "R6_drug_complication_risk"], "gnn_backend": "heuristic", "has_torch": false, "has_pyg": false, "jena_cli": "java -cp \"jena/lib/*\" riot --rules=scripts/jena/rules_medical.rules scripts/jena/ontology_medical.ttl"},
  "triples": [{"head": "肺栓塞", "head_type": "Disease", "relation": "RELATED_TO", "relation_label": "相关", "tail": "肺源性心脏病", "tail_type": "Disease", "confidence": 0.72, "method": "rule", "rule_id": "R1_shared_symptom_related", "explain": "两种疾病共享 ≥2 个症状，推断存在临床相关性（鉴别诊断 / 共病风险）"}]
}
```

> `mode=hybrid` 时响应为 `{"mode": "hybrid", "rule_inferred": 20, "gnn_inferred": <启发式补全条数>, "total": <融合去重后条数>, "backend": {...}, "triples": [...]}`；`method` 取值 `rule | gat | gcn | heuristic`。无 PyTorch / GNN 权重时 `gnn_backend` 为 `heuristic`（Adamic-Adar + Jaccard 启发式补全）。

### 7.8 `GET /graph/path` —— 多跳路径解释

**用途**：返回两个实体之间的多跳关联路径（最多 5 条），用于向用户解释「结论是怎么推导出来的」。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `start` | string | 是 | — | 起点实体，最大 80 字 |
| `end` | string | 是 | — | 终点实体，最大 80 字 |
| `max_hops` | integer | 否 | `3` | 最大跳数，1~4 |

**请求示例**

```http
GET /api/v1/graph/path?start=肺栓塞&end=慢性肾脏病&max_hops=3
```

**响应示例**

```json
{"start": "肺栓塞", "end": "慢性肾脏病", "max_hops": 3, "total": 5, "paths": [[{"from": "肺栓塞", "from_type": "Disease", "relation": "BELONGS_TO", "relation_label": "科室", "to": "内科", "to_type": "Department", "confidence": 0.95}, {"from": "内科", "from_type": "Department", "relation": "BELONGS_TO", "relation_label": "科室", "to": "慢性肾脏病", "to_type": "Disease", "confidence": 0.95}]]}
```

### 7.9 `POST /graph/cypher` —— 受控 Cypher 查询

**用途**：允许前端 / 调试人员执行**只读** Cypher 查询。请求体经 `assert_readonly()` 白名单校验后方可执行；Neo4j 不可用时路由到内存后端支持的 Cypher 子集解释器。

**请求体参数**（模型：`CypherRequest`）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `cypher` | string | 是 | — | Cypher 语句；空串或纯空白触发 422（`cypher 语句不能为空`） |
| `params` | object | 否 | `{}` | Cypher 参数（`$name` 占位符） |
| `limit` | integer | 否 | `200` | 返回行数上限，1~1000 |

**请求示例**

```json
{"cypher": "MATCH (d:Disease) RETURN count(d) AS c", "params": {}, "limit": 200}
```

**响应示例**（内存后端实测）

```json
{"columns": ["c"], "rows": [{"c": 59}], "elapsed_ms": 0.42}
```

**错误响应**（403，写操作被拦截）

```json
{"detail": "Cypher 安全校验未通过：检测到写操作关键字：CREATE，本接口仅支持只读查询"}
```

**错误响应**（400，语句合法但执行失败）

```json
{"detail": "Cypher 执行失败：Invalid input 'RETRN'"}
```

### 7.10 `GET /graph/schema` —— 图谱 Schema

**用途**：返回知识图谱的节点标签、关系类型及其中文名、唯一性约束与索引定义，供前端展示与 Cypher 编写参考。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```bash
curl http://127.0.0.1:8000/api/v1/graph/schema
```

**响应示例**

```json
{
  "nodes": [
    {"label": "Disease", "cn": "疾病", "props": ["name", "alias", "category1", "category2", "definition", "cause", "diagnosis", "treatment", "prognosis", "population", "is_infectious"]},
    {"label": "Symptom", "cn": "症状", "props": ["name", "alias"]},
    {"label": "Drug", "cn": "药物", "props": ["name", "alias", "category"]},
    {"label": "Department", "cn": "科室", "props": ["name", "alias", "category"]},
    {"label": "Treatment", "cn": "治疗方法", "props": ["name", "alias", "category"]},
    {"label": "Check", "cn": "检查项目", "props": ["name", "alias"]},
    {"label": "Population", "cn": "易感人群", "props": ["name"]},
    {"label": "Source", "cn": "文献来源", "props": ["pmid", "doi", "title", "journal", "year", "url", "authority"]}
  ],
  "relations": [{"type": "HAS_SYMPTOM", "cn": "症状"}, {"type": "TREATED_BY", "cn": "治疗"}, {"type": "USES_DRUG", "cn": "用药"}, {"type": "BELONGS_TO", "cn": "科室"}, {"type": "NEEDS_CHECK", "cn": "检查"}, {"type": "AFFECTS", "cn": "易感人群"}, {"type": "HAS_COMPLICATION", "cn": "并发症"}, {"type": "DIFFERENTIAL_WITH", "cn": "鉴别诊断"}, {"type": "RELATED_TO", "cn": "相关"}, {"type": "PROVES", "cn": "知识来源"}, {"type": "IS_INFECTIOUS", "cn": "传染性"}],
  "constraints": ["CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE", "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE", "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE", "CREATE CONSTRAINT dept_name IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE"],
  "indexes": ["CREATE INDEX disease_category IF NOT EXISTS FOR (d:Disease) ON (d.category1)", "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Disease|Symptom|Drug|Department) ON EACH [n.name, n.alias]"]
}
```

---

## 8. 数据分析与健康预警模块 `/analytics`

> 本模块全部图表接口统一使用 `ChartResponse` 模型，**同时返回三份数据**：结构化 `series` / `categories` / `values`（供二次加工）、`echarts_option`（前端 ECharts 直接使用）、`pyecharts_html`（PyEcharts 服务端渲染片段，未安装 PyEcharts 时为空字符串）。此外每个图表都由服务端自动生成一段中文 `insight`（图表洞察结论）。

### 8.1 `GET /analytics/overview` —— 看板概览指标

**用途**：返回 PPT 数据看板的 4 个核心指标：医疗实体总数、疾病分类数、传染性疾病、治疗周期类型。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```bash
curl http://127.0.0.1:8000/api/v1/analytics/overview
```

**响应示例**（实测：567 实体 / 12 个一级分类 / 6 种传染病 / 43 类治疗周期模式）

```json
{"metrics": [{"key": "total_entities", "label": "医疗实体总数", "value": 567, "suffix": "", "icon": "DataLine", "trend": null}, {"key": "disease_categories", "label": "疾病分类数", "value": 12, "suffix": "", "icon": "Grid", "trend": null}, {"key": "infectious", "label": "传染性疾病", "value": 6, "suffix": "", "icon": "Warning", "trend": null}, {"key": "treatment_types", "label": "治疗周期类型", "value": 43, "suffix": "", "icon": "Timer", "trend": null}], "generated_at": "2026-10-01T15:48:30+08:00", "data_source": "memory"}
```

### 8.2 `GET /analytics/node-type-pie` —— 各类节点数量统计（环形饼图）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```http
GET /api/v1/analytics/node-type-pie
```

**响应示例**（`pyecharts_html` 为服务端渲染片段，此处以省略号代替）

```json
{
  "chart_type": "pie",
  "title": "各类节点数量统计",
  "series": [{"name": "症状", "value": 195, "color": "#409EFF"}, {"name": "药物", "value": 122, "color": "#F56C6C"}, {"name": "检查项目", "value": 116, "color": "#31c48d"}, {"name": "疾病", "value": 59, "color": "#E6A23C"}, {"name": "科室", "value": 34, "color": "#b37feb"}, {"name": "治疗方法", "value": 26, "color": "#00BCD4"}, {"name": "易感人群", "value": 15, "color": "#F2C037"}],
  "x_axis": [],
  "categories": [],
  "values": [],
  "unit": "个",
  "echarts_option": {
    "title": {"text": "各类节点数量统计", "left": "center", "top": 4, "textStyle": {"fontSize": 14, "color": "#303133"}},
    "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} 个（{d}%）"},
    "legend": {"orient": "vertical", "left": "left", "top": "middle", "itemWidth": 10, "itemHeight": 10, "textStyle": {"fontSize": 12}},
    "series": [
      {
        "name": "节点类型",
        "type": "pie",
        "radius": ["42%", "68%"],
        "center": ["60%", "55%"],
        "avoidLabelOverlap": true,
        "itemStyle": {"borderRadius": 4, "borderColor": "#fff", "borderWidth": 2},
        "label": {"show": true, "formatter": "{b}\n{d}%", "fontSize": 11, "color": "#606266"},
        "labelLine": {"length": 10, "length2": 8},
        "data": [{"name": "症状", "value": 195, "itemStyle": {"color": "#409EFF"}}, {"name": "药物", "value": 122, "itemStyle": {"color": "#F56C6C"}}, {"name": "检查项目", "value": 116, "itemStyle": {"color": "#31c48d"}}, {"name": "疾病", "value": 59, "itemStyle": {"color": "#E6A23C"}}, {"name": "科室", "value": 34, "itemStyle": {"color": "#b37feb"}}, {"name": "治疗方法", "value": 26, "itemStyle": {"color": "#00BCD4"}}, {"name": "易感人群", "value": 15, "itemStyle": {"color": "#F2C037"}}]
      }
    ]
  },
  "pyecharts_html": "<div id=\"...\" style=\"width:100%;height:320px;\">…（PyEcharts 服务端渲染片段）</div>",
  "insight": "知识图谱共 567 个实体节点，其中症状 34.39%、药物 21.52%、检查项目 20.46%，构成以疾病为中心、症状与治疗方案为支撑的知识网络。"
}
```

### 8.3 `GET /analytics/category-bar` —— 一级分类下的疾病数量（柱状图）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```bash
curl http://127.0.0.1:8000/api/v1/analytics/category-bar
```

**响应示例**（`series` 与 `values` 字段为真实分类分布；`echarts_option` 结构同 8.2，此处省略）

```json
{
  "chart_type": "bar",
  "title": "一级分类下的疾病数量",
  "series": [{"name": "内科", "value": 34, "color": "#409EFF"}, {"name": "外科", "value": 7, "color": "#F56C6C"}, {"name": "传染科", "value": 5, "color": "#31c48d"}, {"name": "妇产科", "value": 3, "color": "#E6A23C"}, {"name": "耳鼻咽喉科", "value": 2, "color": "#b37feb"}, {"name": "精神心理科", "value": 2, "color": "#00BCD4"}, {"name": "儿科", "value": 1, "color": "#F2C037"}, {"name": "皮肤科", "value": 1, "color": "#909399"}, {"name": "眼科", "value": 1, "color": "#5B8FF9"}, {"name": "口腔科", "value": 1, "color": "#FF9845"}, {"name": "肿瘤科", "value": 1, "color": "#409EFF"}, {"name": "急诊科", "value": 1, "color": "#F56C6C"}],
  "x_axis": ["内科", "外科", "传染科", "妇产科", "耳鼻咽喉科", "精神心理科", "儿科", "皮肤科", "眼科", "口腔科", "肿瘤科", "急诊科"],
  "categories": ["内科", "外科", "传染科", "妇产科", "耳鼻咽喉科", "精神心理科", "儿科", "皮肤科", "眼科", "口腔科", "肿瘤科", "急诊科"],
  "values": [34.0, 7.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "unit": "个",
  "echarts_option": {
    "title": {"text": "一级分类下的疾病数量", "left": "center", "top": 4, "textStyle": {"fontSize": 14, "color": "#303133"}},
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
    "grid": {"left": "12%", "right": "6%", "top": 50, "bottom": 40},
    "xAxis": {"type": "category", "data": ["内科", "外科", "传染科", "妇产科", "耳鼻咽喉科", "精神心理科", "儿科", "皮肤科", "眼科", "口腔科", "肿瘤科", "急诊科"], "axisLabel": {"rotate": 30, "fontSize": 11, "color": "#606266"}, "axisLine": {"lineStyle": {"color": "#dcdfe6"}}},
    "yAxis": {"type": "value", "name": "疾病数量", "splitLine": {"lineStyle": {"color": "#f0f2f5"}}, "axisLabel": {"color": "#909399"}},
    "series": [{"name": "疾病数量", "type": "bar", "barWidth": "46%", "data": [34, 7, 5, 3, 2, 2, 1, 1, 1, 1, 1, 1], "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1, "colorStops": [{"offset": 0, "color": "#38bdf8"}, {"offset": 1, "color": "#2b8fe8"}]}}, "label": {"show": true, "position": "top", "fontSize": 11, "color": "#606266"}}]
  },
  "pyecharts_html": "<div id=\"...\" style=\"width:100%;height:320px;\">…（PyEcharts 服务端渲染片段）</div>",
  "insight": "共 12 个一级分类，内科类疾病数量最多（34 个），占全部疾病的 57.6%。"
}
```

### 8.4 `GET /analytics/infectious-gauge` —— 传染性疾病比例（环形图）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```http
GET /api/v1/analytics/infectious-gauge
```

**响应示例**（`series` 固定为 3 项：否 / 是 / 是否传染）

```json
{
  "chart_type": "donut",
  "title": "传染性疾病比例",
  "series": [{"name": "否", "value": 53, "color": "#409EFF"}, {"name": "是", "value": 6, "color": "#F56C6C"}, {"name": "是否传染", "value": 59, "color": "#31c48d"}],
  "x_axis": [],
  "categories": [],
  "values": [],
  "unit": "种",
  "echarts_option": {
    "title": {"text": "传染性疾病比例", "left": "center", "top": 4, "textStyle": {"fontSize": 14, "color": "#303133"}},
    "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} 种（{d}%）"},
    "series": [{"name": "传染性疾病比例", "type": "pie", "radius": ["45%", "70%"], "center": ["58%", "55%"], "itemStyle": {"borderRadius": 4, "borderColor": "#fff", "borderWidth": 2}, "label": {"show": true, "formatter": "{b}\n{d}%", "fontSize": 11, "color": "#606266"}, "labelLine": {"length": 8, "length2": 6}, "data": [{"name": "否", "value": 53, "itemStyle": {"color": "#409EFF"}}, {"name": "是", "value": 6, "itemStyle": {"color": "#F56C6C"}}, {"name": "是否传染", "value": 59, "itemStyle": {"color": "#31c48d"}}]}]
  },
  "pyecharts_html": "<div id=\"...\" style=\"width:100%;height:320px;\">…（PyEcharts 服务端渲染片段）</div>",
  "insight": "在 59 种疾病中，传染性疾病 6 种（10.17%），非传染性疾病 53 种（89.83%）。传染病需重点关注防控与报告要求。"
}
```

### 8.5 `GET /analytics/symptom-top` —— 高频症状 TopN

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `topn` | integer | 否 | `12` | 返回条数，3~30 |

**请求示例**

```http
GET /api/v1/analytics/symptom-top?topn=12
```

**响应示例**

```json
{
  "chart_type": "bar",
  "title": "高频症状 Top12",
  "series": [],
  "x_axis": ["乏力", "心悸", "恶心", "发热", "头痛", "食欲减退", "咳嗽", "上腹痛", "呼吸困难", "呕吐", "胸闷", "腹胀"],
  "categories": ["乏力", "心悸", "恶心", "发热", "头痛", "食欲减退", "咳嗽", "上腹痛", "呼吸困难", "呕吐", "胸闷", "腹胀"],
  "values": [23.0, 15.0, 14.0, 13.0, 11.0, 11.0, 10.0, 10.0, 9.0, 9.0, 8.0, 8.0],
  "unit": "种",
  "echarts_option": {
    "title": {"text": "高频症状 Top12", "left": "center", "top": 4, "textStyle": {"fontSize": 14, "color": "#303133"}},
    "tooltip": {"trigger": "axis"},
    "grid": {"left": "14%", "right": "8%", "top": 46, "bottom": 30},
    "xAxis": {"type": "value", "splitLine": {"lineStyle": {"color": "#f0f2f5"}}},
    "yAxis": {"type": "category", "data": ["腹胀", "胸闷", "呕吐", "呼吸困难", "上腹痛", "咳嗽", "食欲减退", "头痛", "发热", "恶心", "心悸", "乏力"], "axisLabel": {"fontSize": 11, "color": "#606266"}},
    "series": [{"name": "关联疾病数", "type": "bar", "data": [8, 8, 9, 9, 10, 10, 11, 11, 13, 14, 15, 23], "itemStyle": {"borderRadius": [0, 6, 6, 0], "color": {"type": "linear", "x": 0, "y": 0, "x2": 1, "y2": 0, "colorStops": [{"offset": 0, "color": "#7dd3fc"}, {"offset": 1, "color": "#2b8fe8"}]}}, "label": {"show": true, "position": "right", "fontSize": 11, "color": "#606266"}}]
  },
  "pyecharts_html": "",
  "insight": "「乏力」是图谱中关联疾病最多的症状（23 种疾病），反映其在临床鉴别中的高区分价值。"
}
```

### 8.6 `GET /analytics/department-distribution` —— 科室疾病分布

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `topn` | integer | 否 | `10` | 返回科室数，3~30 |

**请求示例**

```bash
curl "http://127.0.0.1:8000/api/v1/analytics/department-distribution?topn=10"
```

**响应示例**

```json
{
  "chart_type": "pie",
  "title": "科室疾病分布",
  "series": [{"name": "内科", "value": 34, "color": "#409EFF"}, {"name": "消化内科", "value": 11, "color": "#F56C6C"}, {"name": "心血管内科", "value": 7, "color": "#31c48d"}, {"name": "外科", "value": 7, "color": "#E6A23C"}, {"name": "呼吸内科", "value": 6, "color": "#b37feb"}, {"name": "传染科", "value": 5, "color": "#00BCD4"}, {"name": "普通外科", "value": 4, "color": "#F2C037"}, {"name": "感染性疾病科", "value": 4, "color": "#909399"}, {"name": "内分泌科", "value": 3, "color": "#5B8FF9"}, {"name": "神经内科", "value": 3, "color": "#FF9845"}],
  "x_axis": [],
  "categories": [],
  "values": [],
  "unit": "条",
  "echarts_option": {
    "title": {"text": "科室疾病分布", "left": "center", "top": 4, "textStyle": {"fontSize": 14, "color": "#303133"}},
    "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} 条关联（{d}%）"},
    "legend": {"type": "scroll", "bottom": 0, "textStyle": {"fontSize": 11}},
    "series": [
      {
        "name": "科室关联数",
        "type": "pie",
        "radius": ["38%", "62%"],
        "center": ["50%", "52%"],
        "itemStyle": {"borderRadius": 4, "borderColor": "#fff", "borderWidth": 2},
        "label": {"show": true, "formatter": "{b} {d}%", "fontSize": 10, "color": "#606266"},
        "data": [
          {"name": "内科", "value": 34, "itemStyle": {"color": "#409EFF"}},
          {"name": "消化内科", "value": 11, "itemStyle": {"color": "#F56C6C"}},
          {"name": "心血管内科", "value": 7, "itemStyle": {"color": "#31c48d"}},
          {"name": "外科", "value": 7, "itemStyle": {"color": "#E6A23C"}},
          {"name": "呼吸内科", "value": 6, "itemStyle": {"color": "#b37feb"}},
          {"name": "传染科", "value": 5, "itemStyle": {"color": "#00BCD4"}},
          {"name": "普通外科", "value": 4, "itemStyle": {"color": "#F2C037"}},
          {"name": "感染性疾病科", "value": 4, "itemStyle": {"color": "#909399"}},
          {"name": "内分泌科", "value": 3, "itemStyle": {"color": "#5B8FF9"}},
          {"name": "神经内科", "value": 3, "itemStyle": {"color": "#FF9845"}}
        ]
      }
    ]
  },
  "pyecharts_html": "<div id=\"...\" style=\"width:100%;height:320px;\">…（PyEcharts 服务端渲染片段）</div>",
  "insight": "覆盖 34 个科室，内科关联疾病最多。"
}
```

### 8.7 `POST /analytics/risk/predict` —— 疾病风险预测

**用途**：输入**去标识化**的健康指标，输出各疾病发病风险概率、风险等级与**可解释的因子贡献度**（leave-one-out 贡献度，各因子贡献度之和归一化为 1）。模型为 `MedicalRiskEnsemble`：每病一个校准风险函数；存在 `models/risk_xgboost.json` 时优先使用 XGBoost 集成 + Platt 校准（`method=xgboost_ensemble`，AUC 0.923），否则使用内置 Logistic 系数（`method=rule_fallback`，`info.auc` 默认 0.905）。

**请求体参数**（模型：`RiskPredictRequest`）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `age` | integer | 是 | — | 年龄，0~120 |
| `gender` | string | 否 | `"male"` | `male` / `female`（其他取值触发 422） |
| `bmi` | number | 否 | `null` | 体质指数，10~60；缺失按人群均值 23.5 填充 |
| `systolic_bp` | integer | 否 | `null` | 收缩压 mmHg，60~260；缺失填充 122 |
| `diastolic_bp` | integer | 否 | `null` | 舒张压 mmHg，30~180；缺失填充 76 |
| `fasting_glucose` | number | 否 | `null` | 空腹血糖 mmol/L，1.0~40.0；缺失填充 5.2 |
| `total_cholesterol` | number | 否 | `null` | 总胆固醇 mmol/L，1.0~20.0；缺失填充 4.8 |
| `hdl` | number | 否 | `null` | 高密度脂蛋白 mmol/L，0.1~5.0；缺失填充 1.3 |
| `ldl` | number | 否 | `null` | 低密度脂蛋白 mmol/L，0.1~12.0；缺失填充 2.9 |
| `triglycerides` | number | 否 | `null` | 甘油三酯 mmol/L，0.1~30.0；缺失填充 1.4 |
| `heart_rate` | integer | 否 | `null` | 心率 次/分，30~220；缺失填充 74 |
| `smoking` | boolean | 否 | `false` | 是否吸烟 |
| `drinking` | boolean | 否 | `false` | 是否饮酒 |
| `physical_activity` | string | 否 | `"moderate"` | `low` / `moderate` / `high`（`low` 派生出 `low_activity=1`） |
| `family_history` | string[] | 否 | `[]` | 家族史疾病名列表（命中数上限 4） |
| `symptoms` | string[] | 否 | `[]` | 当前症状（按 `SYMPTOM_WEIGHTS` 折叠为 `symptom_score`，上限 5.0） |
| `existing_conditions` | string[] | 否 | `[]` | 既往史 |

**请求示例**（项目冒烟测试使用的风险画像）

```json
{"age": 58, "gender": "male", "bmi": 27.4, "systolic_bp": 152, "diastolic_bp": 96, "fasting_glucose": 6.4, "total_cholesterol": 5.9, "ldl": 3.8, "hdl": 1.0, "triglycerides": 2.4, "smoking": true, "physical_activity": "low", "family_history": ["hypertension", "diabetes"], "symptoms": ["头晕", "乏力"]}
```

**响应示例**（无 `models/risk_xgboost.json` 时按内置 Logistic 系数计算，`predictions` 取风险最高的 6 个疾病；风险值并列时按内置模型定义顺序稳定排序）

```json
{
  "predictions": [
    {"disease": "原发性高血压", "disease_id": "原发性高血压", "risk": 1.0, "risk_percent": "100.0%", "level": "极高风险", "top_factors": [{"factor": "收缩压 152 mmHg", "contribution": 0.9052, "direction": "increase"}, {"factor": "年龄 58 岁", "contribution": 0.0362, "direction": "increase"}, {"factor": "BMI 27.4", "contribution": 0.0348, "direction": "increase"}, {"factor": "舒张压 96 mmHg", "contribution": 0.0091, "direction": "increase"}, {"factor": "空腹血糖 6.4 mmol/L", "contribution": 0.0057, "direction": "increase"}], "kg_evidence": []},
    {"disease": "2型糖尿病", "disease_id": "2型糖尿病", "risk": 1.0, "risk_percent": "100.0%", "level": "极高风险", "top_factors": [{"factor": "空腹血糖 6.4 mmol/L", "contribution": 0.8449, "direction": "increase"}, {"factor": "BMI 27.4", "contribution": 0.1232, "direction": "increase"}, {"factor": "年龄 58 岁", "contribution": 0.0156, "direction": "increase"}, {"factor": "收缩压 152 mmHg", "contribution": 0.0076, "direction": "increase"}, {"factor": "家族史（命中 2 项）", "contribution": 0.0058, "direction": "increase"}], "kg_evidence": []},
    {"disease": "冠心病", "disease_id": "冠心病", "risk": 1.0, "risk_percent": "100.0%", "level": "极高风险", "top_factors": [{"factor": "收缩压 152 mmHg", "contribution": 0.4207, "direction": "increase"}, {"factor": "年龄 58 岁", "contribution": 0.3879, "direction": "increase"}, {"factor": "BMI 27.4", "contribution": 0.0412, "direction": "increase"}, {"factor": "空腹血糖 6.4 mmol/L", "contribution": 0.0409, "direction": "increase"}, {"factor": "总胆固醇 5.9 mmol/L", "contribution": 0.0295, "direction": "increase"}], "kg_evidence": []},
    {"disease": "脑卒中", "disease_id": "脑卒中", "risk": 1.0, "risk_percent": "100.0%", "level": "极高风险", "top_factors": [{"factor": "收缩压 152 mmHg", "contribution": 0.8671, "direction": "increase"}, {"factor": "年龄 58 岁", "contribution": 0.1075, "direction": "increase"}, {"factor": "空腹血糖 6.4 mmol/L", "contribution": 0.0111, "direction": "increase"}, {"factor": "BMI 27.4", "contribution": 0.0042, "direction": "increase"}, {"factor": "舒张压 96 mmHg", "contribution": 0.0026, "direction": "increase"}], "kg_evidence": []},
    {"disease": "高脂血症", "disease_id": "高脂血症", "risk": 1.0, "risk_percent": "100.0%", "level": "极高风险", "top_factors": [{"factor": "总胆固醇 5.9 mmol/L", "contribution": 0.4812, "direction": "increase"}, {"factor": "BMI 27.4", "contribution": 0.2389, "direction": "increase"}, {"factor": "低密度脂蛋白 3.8 mmol/L", "contribution": 0.1389, "direction": "increase"}, {"factor": "年龄 58 岁", "contribution": 0.0377, "direction": "increase"}, {"factor": "家族史（命中 2 项）", "contribution": 0.029, "direction": "increase"}], "kg_evidence": []},
    {"disease": "脂肪肝", "disease_id": "脂肪肝", "risk": 0.9998, "risk_percent": "100.0%", "level": "极高风险", "top_factors": [{"factor": "BMI 27.4", "contribution": 0.8794, "direction": "increase"}, {"factor": "空腹血糖 6.4 mmol/L", "contribution": 0.0453, "direction": "increase"}, {"factor": "收缩压 152 mmHg", "contribution": 0.0227, "direction": "increase"}, {"factor": "年龄 58 岁", "contribution": 0.0205, "direction": "increase"}, {"factor": "甘油三酯 2.4 mmol/L", "contribution": 0.007, "direction": "increase"}], "kg_evidence": []}
  ],
  "overall_level": "极高风险",
  "health_score": 0.0,
  "model": {"name": "MedicalRiskEnsemble", "version": "1.0.0", "auc": 0.905, "disease_coverage": 1000, "method": "rule_fallback", "features_used": ["age", "male", "bmi", "systolic_bp", "diastolic_bp", "fasting_glucose", "total_cholesterol", "ldl", "hdl", "triglycerides", "heart_rate", "smoking", "drinking", "family_history", "symptom_score", "low_activity"]},
  "summary": "综合健康评分 0.0/100。风险最高的是「原发性高血压」（100.0%，极高风险）。其中可通过生活方式干预改善的因素包括：收缩压 152 mmHg、BMI 27.4、舒张压 96 mmHg、空腹血糖 6.4 mmol/L。建议携带本结果前往相应专科就诊，由执业医师结合面诊与检查综合评估。",
  "disclaimer": "风险预测结果仅供参考，不能替代执业医师诊断。",
  "latency_ms": 0.83
}
```

> ⚠️ **关于内置演示系数**：`BUILTIN_MODELS` 中的 10 个疾病模型系数为「已发表队列研究效应量的近似值」（源码注释原文：*示例值，用于演示原型*），仅覆盖 10 个疾病，且对中老年画像容易饱和到 100%。生产环境应使用 `scripts/train_risk_model.py` 训练出的 `models/risk_xgboost.json`（实测 AUC 0.923，疾病覆盖 1000+），此时 `model.method="xgboost_ensemble"`，各病概率经 Platt 校准后分布更合理。接口本身的行为（字段、排序、贡献度归一化）不随权重切换而改变。
>
> `health_score` 计算式：`(1 - 0.68 × 最高风险 - 0.32 × Top6 平均风险) × 100`，clamp 到 `[0, 100]`，保留 1 位小数。

**错误响应**（422，年龄越界 + 性别非法）

```json
{"code": 422, "message": "请求参数校验失败", "errors": [{"field": "age", "message": "Input should be less than or equal to 120", "type": "less_than_equal"}, {"field": "gender", "message": "Value error, gender 必须为 male 或 female", "type": "value_error"}], "hint": "请检查请求体字段类型与取值范围"}
```

### 8.8 `POST /analytics/risk/intervene` —— 个性化健康干预建议

**用途**：基于风险预测结果生成**可执行**的干预方案，输出饮食 / 运动 / 体检 / 生活方式四大类，每类含多条具体措施与优先级。生成逻辑：先跑一次预测 → 取 Top-3 疾病 → 聚合其 `diet_focus` 与 `checks` → 结合用户具体指标（血压 / 血糖 / BMI / 吸烟 / 年龄）补充针对性条目。

**请求体参数**（模型：`InterventionPlanRequest`，继承 `RiskPredictRequest` 的全部字段）

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| （继承字段） | — | — | — | 同 8.7 `RiskPredictRequest` 的 17 个字段 |
| `risk_diseases` | string[] | 否 | `[]` | 需重点干预的疾病列表；留空时自动取预测 Top-3 |

**请求示例**

```json
{"age": 58, "gender": "male", "bmi": 27.4, "systolic_bp": 152, "diastolic_bp": 96, "fasting_glucose": 6.4, "total_cholesterol": 5.9, "ldl": 3.8, "hdl": 1.0, "triglycerides": 2.4, "smoking": true, "physical_activity": "low", "family_history": ["hypertension", "diabetes"], "symptoms": ["头晕", "乏力"], "risk_diseases": []}
```

**响应示例**（Top-3 疾病为 原发性高血压 / 2型糖尿病 / 冠心病）

```json
{
  "plan": [
    {"category": "diet", "category_cn": "饮食建议", "icon": "Bowl", "title": "饮食干预", "items": ["限盐（每日食盐 <5g）", "增加富含钾的蔬果", "限制饮酒", "控制总热量", "控制精制碳水与含糖饮料", "增加膳食纤维与全谷物", "定时定量进餐", "优先选择低升糖指数食物"], "priority": "high"},
    {"category": "exercise", "category_cn": "运动建议", "icon": "Bicycle", "title": "运动干预", "items": ["有氧运动：每周至少 150 分钟中等强度（快走、慢跑、游泳、骑行），可拆分为每次 30 分钟", "抗阻训练：每周 2-3 次，隔天进行，覆盖大肌群", "在 150 分钟基础上逐步增加至每周 250-300 分钟，以增强减重效果", "运动前充分热身，出现胸痛、明显气促、头晕时立即停止并就医"], "priority": "high"},
    {"category": "examination", "category_cn": "体检建议", "icon": "DocumentChecked", "title": "检查与监测", "items": ["诊室血压测量", "动态血压监测", "尿常规", "血生化", "心电图", "超声心动图", "空腹血糖", "口服葡萄糖耐量试验", "糖化血红蛋白", "血脂"], "priority": "high"},
    {"category": "lifestyle", "category_cn": "生活方式", "icon": "Sunny", "title": "生活方式调整", "items": ["戒烟：这是降低心脑血管与呼吸系统风险收益最大的单项措施，可就诊戒烟门诊或拨打 12320", "保证每晚 7-8 小时睡眠，固定作息时间", "学习压力管理技巧（正念、呼吸训练），必要时寻求心理支持", "家庭血压监测：早晚各一次，连续 7 天取后 6 天平均值并记录", "关注血糖变化，如出现多饮、多尿、体重下降请及时就诊", "严格遵医嘱用药，不自行停药、减药或加药"], "priority": "high"}
  ],
  "follow_up": "建议 3 个月后复查相关指标；如为心血管内科、内分泌科在管患者，请按主治医师制定的随访计划执行。出现症状加重、新发不适或指标明显异常时请及时就诊。",
  "disclaimer": "干预建议由 AI 生成，仅供参考，请遵医嘱。"
}
```

### 8.9 `GET /analytics/risk/model-info` —— 风险模型信息

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```http
GET /api/v1/analytics/risk/model-info
```

**响应示例**（无训练权重时的实测输出）

```json
{"backend": "builtin_logistic", "auc": 0.905, "disease_coverage": 1000, "builtin_models": 10, "xgb_models": 0, "features": ["age", "male", "bmi", "systolic_bp", "diastolic_bp", "fasting_glucose", "total_cholesterol", "ldl", "hdl", "triglycerides", "heart_rate", "smoking", "drinking", "family_history", "symptom_score", "low_activity"], "model_class": "MedicalRiskEnsemble (per-disease logistic / XGBoost + Platt calibration)"}
```

> 加载 `models/risk_xgboost.json` 成功后：`backend` 变为 `xgboost_ensemble`，`auc` 取权重文件中的实测值（训练脚本产出 0.923），`xgb_models` 为实际加载的疾病数；若权重文件存在但环境缺少 `xgboost`，则打印 `未安装 xgboost，风险预测继续使用内置 Logistic 系数` 并保持 `builtin_logistic`。

### 8.10 `GET /analytics/charts` —— 看板聚合接口

**用途**：一次请求返回 `overview` + 5 张图表，前端只需一个请求即可渲染完整数据看板（内部调用与 8.1~8.6 完全一致，`symptom_top` 取默认 12、`department_distribution` 取默认 10）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```bash
curl http://127.0.0.1:8000/api/v1/analytics/charts
```

**响应示例**（顶层结构；各子对象的字段与上面 8.1~8.6 完全相同，此处折叠显示）

```json
{
  "overview": {"metrics": [{"key": "total_entities", "label": "医疗实体总数", "value": 567, "suffix": "", "icon": "DataLine", "trend": null}, {"key": "disease_categories", "label": "疾病分类数", "value": 12, "suffix": "", "icon": "Grid", "trend": null}, {"key": "infectious", "label": "传染性疾病", "value": 6, "suffix": "", "icon": "Warning", "trend": null}, {"key": "treatment_types", "label": "治疗周期类型", "value": 43, "suffix": "", "icon": "Timer", "trend": null}], "generated_at": "2026-10-01T15:50:02+08:00", "data_source": "memory"},
  "node_type_pie": {"chart_type": "pie", "title": "各类节点数量统计", "series": [{"name": "症状", "value": 195, "color": "#409EFF"}, {"name": "药物", "value": 122, "color": "#F56C6C"}, {"name": "检查项目", "value": 116, "color": "#31c48d"}, {"name": "疾病", "value": 59, "color": "#E6A23C"}, {"name": "科室", "value": 34, "color": "#b37feb"}, {"name": "治疗方法", "value": 26, "color": "#00BCD4"}, {"name": "易感人群", "value": 15, "color": "#F2C037"}], "unit": "个", "insight": "知识图谱共 567 个实体节点，其中症状 34.39%、药物 21.52%、检查项目 20.46%，构成以疾病为中心、症状与治疗方案为支撑的知识网络。"},
  "category_bar": {"chart_type": "bar", "title": "一级分类下的疾病数量", "categories": ["内科", "外科", "传染科", "妇产科", "耳鼻咽喉科", "精神心理科", "儿科", "皮肤科", "眼科", "口腔科", "肿瘤科", "急诊科"], "values": [34.0, 7.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "unit": "个", "insight": "共 12 个一级分类，内科类疾病数量最多（34 个），占全部疾病的 57.6%。"},
  "infectious_gauge": {"chart_type": "donut", "title": "传染性疾病比例", "series": [{"name": "否", "value": 53, "color": "#409EFF"}, {"name": "是", "value": 6, "color": "#F56C6C"}, {"name": "是否传染", "value": 59, "color": "#31c48d"}], "unit": "种", "insight": "在 59 种疾病中，传染性疾病 6 种（10.17%），非传染性疾病 53 种（89.83%）。传染病需重点关注防控与报告要求。"},
  "symptom_top": {"chart_type": "bar", "title": "高频症状 Top12", "categories": ["乏力", "心悸", "恶心", "发热", "头痛", "食欲减退", "咳嗽", "上腹痛", "呼吸困难", "呕吐", "胸闷", "腹胀"], "values": [23.0, 15.0, 14.0, 13.0, 11.0, 11.0, 10.0, 10.0, 9.0, 9.0, 8.0, 8.0], "unit": "种", "insight": "「乏力」是图谱中关联疾病最多的症状（23 种疾病），反映其在临床鉴别中的高区分价值。"},
  "department_distribution": {
    "chart_type": "pie",
    "title": "科室疾病分布",
    "series": [{"name": "内科", "value": 34, "color": "#409EFF"}, {"name": "消化内科", "value": 11, "color": "#F56C6C"}, {"name": "心血管内科", "value": 7, "color": "#31c48d"}, {"name": "外科", "value": 7, "color": "#E6A23C"}, {"name": "呼吸内科", "value": 6, "color": "#b37feb"}, {"name": "传染科", "value": 5, "color": "#00BCD4"}, {"name": "普通外科", "value": 4, "color": "#F2C037"}, {"name": "感染性疾病科", "value": 4, "color": "#909399"}, {"name": "内分泌科", "value": 3, "color": "#5B8FF9"}, {"name": "神经内科", "value": 3, "color": "#FF9845"}],
    "unit": "条",
    "insight": "覆盖 34 个科室，内科关联疾病最多。"
  }
}
```

> 注：聚合接口使用 `.model_dump()` 输出，因此每个子对象都包含 `echarts_option`、`pyecharts_html` 等完整字段（上面为控制篇幅仅列出关键字段）。

### 8.11 `GET /analytics/info` —— 数据分析服务信息

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| — | — | — | — | 无查询参数 |

**请求示例**

```bash
curl http://127.0.0.1:8000/api/v1/analytics/info
```

**响应示例**

```json
{"pyecharts_available": false, "charts": ["overview", "node_type_pie", "category_bar", "infectious_gauge", "symptom_top", "department_distribution"]}
```

> `pyecharts_available=false` 表示运行环境未安装 PyEcharts：此时所有图表接口的 `pyecharts_html` 返回空字符串，但 `echarts_option` 照常返回，前端图表不受影响（降级设计）。

---

## 9. 错误处理

后端统一遵循「**结构化 JSON、不泄露堆栈**」原则，共四类错误格式。

### 9.1 422 参数校验失败

由 `main.py::validation_exception_handler` 统一处理 Pydantic / FastAPI 的参数校验错误，`field` 为参数位置路径（`body` 级已被过滤，嵌套字段用 ` → ` 连接）。

```json
{"code": 422, "message": "请求参数校验失败", "errors": [{"field": "question", "message": "String should have at least 1 character", "type": "string_too_short"}, {"field": "top_k", "message": "Input should be less than or equal to 50", "type": "less_than_equal"}], "hint": "请检查请求体字段类型与取值范围"}
```

常见触发场景：

| 场景 | 触发字段 | `errors[].type` 示例 |
|------|---------|---------------------|
| `question` 为空 / 纯空白 | `question` | `string_too_short` / `value_error` |
| `question` 超过 500 字 | `question` | `string_too_long` |
| `top_k` / `max_hops` / `page_size` / `limit` 越界 | 对应整型字段 | `less_than_equal` / `greater_than_equal` |
| `gender` 非 `male` / `female` | `gender` | `value_error` |
| `cypher` 为空串 | `cypher` | `value_error` |
| `age` 越界（如 999） | `age` | `less_than_equal` |

### 9.2 403 Cypher 安全校验失败

`POST /graph/cypher` 命中只读白名单校验时返回，`detail` 前缀固定为 `Cypher 安全校验未通过：`。

```json
{"detail": "Cypher 安全校验未通过：检测到写操作关键字：CREATE，本接口仅支持只读查询"}
```

其他可能的 `detail` 取值：

| 触发条件 | `detail` |
|---------|---------|
| 语句不以只读关键字开头 | `Cypher 安全校验未通过：仅允许以 MATCH / WITH / UNWIND / RETURN / CALL / SHOW 开头的只读查询` |
| 命中写关键字 | `Cypher 安全校验未通过：检测到写操作关键字：MERGE，本接口仅支持只读查询` |
| `CALL` 非白名单过程 | `Cypher 安全校验未通过：CALL 仅允许白名单内的 APOC 只读过程` |
| 语句为空（`assert_readonly` 层） | `Cypher 安全校验未通过：Cypher 语句为空` |

> 注意：`CypherRequest.cypher` 的空白校验在 Pydantic 层先行触发，因此空串实际返回 **422**；403 的“语句为空”仅在直接调用底层校验函数时出现。

### 9.3 404 资源不存在

由路由内 `HTTPException(status_code=404, ...)` 抛出，FastAPI 默认包装为 `{"detail": ...}`，**不带 `code` 字段**。

```json
{"detail": "知识库中未找到疾病「肺栓塞晚期」。请尝试使用标准疾病名称，如「原发性高血压」。"}
```

各接口的 404 文案：

| 接口 | `detail` |
|------|---------|
| `GET /disease/{disease_id}` | `知识库中未找到疾病「xxx」。请尝试使用标准疾病名称，如「原发性高血压」。` |
| `GET /disease/{disease_id}/related` | `知识库中未找到疾病「xxx」` |
| `GET /disease/{disease_id}/sources` | `知识库中未找到疾病「xxx」` |
| `GET /disease/{disease_id}/facts` | `知识库中未找到疾病「xxx」` |
| `POST /disease/{disease_id}/refresh-source` | `知识库中未找到疾病「xxx」` |
| `GET /graph/evidence` | `未找到「xxx」的匹配关系` |

> 说明：`GET /graph/subgraph` 对未知实体**不返回 404**，而是返回 `node_count=0` 与 `message="知识库中未找到实体「xxx」，请尝试其他名称"`，以便前端在图上给出友好提示。

### 9.4 500 服务内部错误

由 `main.py::global_exception_handler` 兜底，任何未捕获异常都返回结构化 JSON；`detail` 仅在 `DEBUG=true` 时输出异常文本，生产环境为 `null`（不泄露堆栈）。

```json
{"code": 500, "message": "服务内部错误，请稍后重试", "detail": null, "path": "/api/v1/qa/ask"}
```

此外，部分接口在内部捕获异常后会转换为更具体的 500（经 `HTTPException`，格式为 `{"detail": "..."}`）：

| 接口 | `detail` |
|------|---------|
| `POST /qa/ask` | `问答服务异常：...` |
| `GET /qa/prompt-templates` | `加载模板库失败：...` |
| `GET /disease/search` | `疾病检索失败：...` |
| `GET /graph/subgraph` | `子图查询失败：...` |
| `GET /graph/reasoning` | `图谱推理失败：...` |
| `POST /analytics/risk/predict` | `风险预测失败：...` |
| `POST /analytics/risk/intervene` | `干预建议生成失败：...` |

同时，全局异常处理器会把完整堆栈写入服务端日志（`backend/logs/zhiyu.log`），便于排查。

### 9.5 429 限流（安装 `slowapi` 时）

超过 `RATE_LIMIT_PER_MINUTE`（默认 60 次/分钟/IP）时由 `slowapi` 的 `_rate_limit_exceeded_handler` 返回：

```json
{"error": "Rate limit exceeded: 60 per 1 minute"}
```

> 未安装 `slowapi` 时，应用启动会打印 `未安装 slowapi，全局限流未启用（pip install slowapi）`，此时不存在 429。

---

## 10. Cypher 只读白名单

`POST /graph/cypher` 是唯一允许执行自定义 Cypher 的入口，其安全边界由 `backend/app/kg_layer/neo4j_client.py::assert_readonly` 实现，并由配置项 `CYPHER_READONLY_WHITELIST`（默认 `true`）控制是否启用。**该接口仅支持只读查询，任何写操作都会被拒绝**——这是防止图谱数据被篡改的安全底线。

### 10.1 允许的语句前缀

语句按 `;` 分段，**每一段**都必须以满足以下正则的只读关键字开头（`USE` / `EXPLAIN` / `PROFILE` 同样被允许）：

```python
re.match(r"^(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN|CALL|USE|EXPLAIN|PROFILE|SHOW)", seg, re.IGNORECASE)
```

| 允许前缀 | 说明 |
|---------|------|
| `MATCH` | 模式匹配（最常用） |
| `OPTIONAL MATCH` | 可选匹配 |
| `WITH` | 查询管道衔接 |
| `UNWIND` | 列表展开 |
| `RETURN` | 结果返回 |
| `CALL` | 过程调用（受 10.3 白名单约束） |
| `USE` | 指定数据库 |
| `EXPLAIN` | 查看执行计划（不执行） |
| `PROFILE` | 性能剖析 |
| `SHOW` | 元数据查询（如 `SHOW CONSTRAINTS`） |

### 10.2 拒绝的写操作关键字

以下关键字出现在**任意分段**中即被拒绝（正则 `\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|CALL\s+\{)\b`，忽略大小写）：

| 关键字 | 危害 |
|--------|------|
| `CREATE` | 新增节点 / 关系 / 约束 / 索引 |
| `MERGE` | 幂等写入 |
| `DELETE` | 删除节点 / 关系 |
| `DETACH` | 配合 DELETE 删除节点及其所有关系 |
| `SET` | 修改属性 |
| `REMOVE` | 删除属性 / 标签 |
| `DROP` | 删除约束 / 索引 |
| `FOREACH` | 批量写循环 |
| `LOAD CSV` | 从外部导入数据（数据外泄 / 注入风险） |
| `CALL {` | 子查询（可绕过分段校验，故一并禁止） |

### 10.3 APOC 过程白名单

语句包含 `call`（不区分大小写）时，必须同时匹配以下白名单，否则拒绝：

```python
_ALLOWED_PROC = re.compile(r"\bapoc\.(meta|path|text|coll|number)\.", re.IGNORECASE)
```

| 允许的过程命名空间 | 用途 |
|------------------|------|
| `apoc.meta.*` | 元数据（Schema / 统计） |
| `apoc.path.*` | 路径展开与图遍历 |
| `apoc.text.*` | 文本处理 |
| `apoc.coll.*` | 集合处理 |
| `apoc.number.*` | 数值处理 |

其余命名空间（尤其是 `apoc.periodic.*`、`apoc.export.*`、`apoc.load.*` 等具备写入或外联能力的过程）一律拒绝：`CALL 仅允许白名单内的 APOC 只读过程`。

### 10.4 为什么需要只读白名单

1. **防篡改**：图谱是全部问答与溯源结论的唯一事实来源，一旦被改写，`[KG-n]` 引用与 `:Source` 溯源链条即失真；
2. **防数据外泄**：`LOAD CSV` / `apoc.export.*` 可把图数据落盘或外发，在医疗数据场景属于合规红线；
3. **防注入**：多语句（`;` 分段）与子查询（`CALL {`）是 Cypher 注入的常见载体，逐段校验 + 子查询禁用可有效阻断；
4. **最小权限**：写操作仅保留给离线脚本（`scripts/init_neo4j.py`、`scripts/load_kg.py` 通过 `Neo4jClient.write()`，`readonly=False`），API 层永不暴露写能力；
5. **可用性保障**：Neo4j 不可用时，同一份语句会路由到内存后端的 Cypher 子集解释器（支持 `count` 类统计、按标签列举实体与 `SHOW`），不支持的语句返回空结果并写日志，而不是抛异常。

### 10.5 允许 / 拒绝示例

```cypher
// ✅ 允许：统计疾病数量
MATCH (d:Disease) RETURN count(d) AS c

// ✅ 允许：查询高血压的症状（带参数与 LIMIT）
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE d.name = $name
RETURN s.name AS symptom
LIMIT 10

// ✅ 允许：APOC 白名单过程
MATCH (d:Disease {name: '肺栓塞'})
CALL apoc.path.subgraphNodes(d, {maxLevel: 1}) YIELD node
RETURN node.name AS name
LIMIT 20

// ❌ 拒绝（403）：写操作
CREATE (d:Disease {name: '测试疾病'})

// ❌ 拒绝（403）：改属性
MATCH (d:Disease {name: '肺栓塞'}) SET d.name = 'PE'

// ❌ 拒绝（403）：删除
MATCH (n:Symptom) DETACH DELETE n

// ❌ 拒绝（403）：非白名单过程
CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'SET n.x=1', {batchSize: 100})

// ❌ 拒绝（403）：子查询
CALL { MATCH (n) RETURN n } RETURN n
```

---

## 11. 快速开始 / curl 示例

以下 4 条命令分别覆盖四大业务模块，可直接复制执行（假设服务监听 `127.0.0.1:8000`）。

```bash
# ① 知识图谱：查看图谱规模与实体类型分布
curl -s "http://127.0.0.1:8000/api/v1/graph/stats"

# ② 疾病查询：检索「肺栓塞」的完整结构化详情（定义/病因/症状/诊断/治疗/预后）
curl -s "http://127.0.0.1:8000/api/v1/disease/%E8%82%BA%E6%A0%93%E5%A1%9E"

# ③ 智能问答：可解释、可溯源的科室导诊问答
curl -s -X POST "http://127.0.0.1:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"肺栓塞应该挂什么科？","session_id":"curl-demo","top_k":12,"explain":true}'

# ④ 数据分析：脱敏病史输入 → 疾病风险预测与因子贡献度
curl -s -X POST "http://127.0.0.1:8000/api/v1/analytics/risk/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":58,"gender":"male","bmi":27.4,"systolic_bp":152,"diastolic_bp":96,
       "fasting_glucose":6.4,"total_cholesterol":5.9,"ldl":3.8,"hdl":1.0,
       "triglycerides":2.4,"smoking":true,"physical_activity":"low",
       "family_history":["hypertension","diabetes"],"symptoms":["头晕","乏力"]}'
```

```powershell
# Windows PowerShell 等价写法（④）：注意引号与换行处理
$body = @{
  age = 58; gender = "male"; bmi = 27.4; systolic_bp = 152; diastolic_bp = 96
  fasting_glucose = 6.4; total_cholesterol = 5.9; ldl = 3.8; hdl = 1.0
  triglycerides = 2.4; smoking = $true; physical_activity = "low"
  family_history = @("hypertension", "diabetes"); symptoms = @("头晕", "乏力")
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/analytics/risk/predict" `
  -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

**启动服务**

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# 或
python main.py
```

**运行冒烟测试**

```bash
cd backend
pytest tests/ -v
pytest tests/ -v -k graph     # 只跑图谱相关用例
```

---

## 12. 免责声明

> **本系统为演示原型。所有 AI 输出仅供参考，不能替代执业医师诊断。**
>
> - 本系统不提供任何形式的诊断结论、处方或用药剂量建议；接口返回中的 `disclaimer` 字段、模板全局硬约束与幻觉守卫共同保证这一边界；
> - 出现急症症状（剧烈胸痛、呼吸困难、意识障碍、大出血、抽搐等）请**立即前往急诊**或拨打 120；
> - 风险预测与干预建议（`/analytics/risk/*`）均为统计模型输出，不能作为临床决策依据；
> - 任何健康决策请务必咨询具备执业资质的医师。
> - 数据合规：PII 脱敏（姓名 / 身份证 / 手机号 / 地址 / 病历号等 11 类规则）+ AES-256-GCM 字段加密，风险预测接口仅接收去标识化数值特征。

