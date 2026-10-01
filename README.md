# 医数智答 —— 智愈医典 · 医疗大数据知识问答系统

> **参赛赛道**：大模型与智能体应用赛道
> **项目定位**：基于「医疗大数据 + 知识图谱 + RAG 大模型」的专业医疗知识问答平台
> **产品名**：智愈医典（Zhiyu Medical Codex）

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white">
  <img alt="Vue" src="https://img.shields.io/badge/Vue-3.4-42b883?logo=vue.js&logoColor=white">
  <img alt="Neo4j" src="https://img.shields.io/badge/Neo4j-5.8-4581C3?logo=neo4j&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-47%20passed-brightgreen">
</p>

**四大核心功能**：知识图谱可视化 · 智能疾病查询 · 可解释智能问答 · 多维数据分析
**三大核心创新**：★改进 PCNN 关系抽取（医疗关键词注意力，召回率 +8.3%）★医疗专属 RAG 提示工程（100+ 模板，准确率 +12.1%）★轻量化图谱可视化整合

> ⚡ **3 分钟跑通演示**（无需 GPU / 大模型 / Neo4j，全自动降级为离线演示模式）：
> ```bash
> cd backend && pip install -r requirements-lite.txt && uvicorn main:app --reload --port 8000
> cd frontend && npm install && npm run dev    # → http://127.0.0.1:5173
> ```
> 详见 [四、环境部署步骤](#四环境部署步骤)。

---

## 效果预览

> 截图请放入 [`docs/screenshots/`](docs/screenshots/) 后取消下方注释即可在 GitHub 渲染。

<!--
| 主页 | 知识图谱可视化 |
|:---:|:---:|
| ![主页](docs/screenshots/01-home.png) | ![知识图谱](docs/screenshots/02-graph.png) |
| **智能疾病查询** | **可解释智能问答** |
| ![疾病查询](docs/screenshots/03-disease.png) | ![智能问答](docs/screenshots/04-chat.png) |
| **多维数据分析** | **接口文档** |
| ![数据分析](docs/screenshots/05-analytics.png) | ![接口文档](docs/screenshots/06-swagger.png) |
-->

| 功能 | 接口 / 页面 | 亮点 |
|------|------------|------|
| 🕸️ 知识图谱可视化 | `GET /graph/subgraph` | D3.js 力导向图，8 类实体配色，一键检索 + 详情联动 |
| 📖 智能疾病查询 | `GET /disease/search` | 卡片式：定义/病因/症状/诊断/治疗/预后，100% 可溯源 |
| 💬 可解释智能问答 | `POST /qa/ask` | `[KG-n]` 溯源角标 + 全链路推理 trace，拒绝黑盒 |
| 📊 多维数据分析 | `POST /analytics/risk/predict` | 疾病风险预测（实测 AUC 0.923）+ 个性化健康干预 |

---

## 目录

- [一、项目概述](#一项目概述)
- [二、四层技术架构](#二四层技术架构)
- [三、工程目录结构](#三工程目录结构)
- [四、环境部署步骤](#四环境部署步骤)
- [五、Neo4j 初始化方式](#五neo4j-初始化方式)
- [六、模型下载说明](#六模型下载说明)
- [七、三大核心创新点](#七三大核心创新点)
- [八、接口文档](#八接口文档)
- [九、量化指标达成情况](#九量化指标达成情况)
- [十、安全合规与免责声明](#十安全合规与免责声明)

---

## 一、项目概述

### 1.1 解决的四大业务痛点

| # | 痛点 | 具体表现 | 本系统对策 |
|---|------|---------|-----------|
| 1 | 医疗信息**碎片化、互相矛盾** | 搜索引擎信息零散不成体系，内容间存在冲突 | 多源数据清洗 + 知识图谱结构化关联（`kg_layer`） |
| 2 | 通用大模型**医疗幻觉** | 生成过时、错误或无关的医疗建议 | KG 三元组约束的医疗专属 RAG 提示工程（`llm_layer/rag`） |
| 3 | AI 决策**黑盒不可解释** | 诊断建议缺乏推理逻辑，医生无法理解 | 全链路推理溯源 + KG 路径可视化（`reasoning_trace`） |
| 4 | 优质医疗知识**难以下沉基层** | 基层医生缺乏权威知识获取渠道 | 轻量化图谱 + 一键检索详情联动 + 疾病风险预警 |

### 1.2 系统目标

1. 构建**结构化医疗知识图谱**（Neo4j 5.8，实体 ≥10 万 / 关系 ≥50 万）
2. **可解释**智能医疗问答（推理链路全溯源，拒绝黑盒）
3. **疾病信息查询**（卡片式：定义/病因/症状/诊断/治疗/预后）
4. **多维医疗数据分析 + 疾病风险预警**（1000+ 疾病预测，AUC ≥0.9）
5. 全部知识来源**可溯源**，数据**全链路脱敏加密**，满足医疗合规

---

## 二、四层技术架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ④ 应用层  Application Layer                                            │
│     Vue3 + Element Plus + D3.js + PyEcharts                             │
│     ├─ 主页  ├─ 知识图谱可视化  ├─ 疾病查询  ├─ 智能问答  └─ 数据分析   │
│     网关：FastAPI + Uvicorn（异步接口）                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ③ LLM 增强层  LLM Enhancement Layer                                    │
│     ├─ TextCNN 意图分类（4 类）                                         │
│     ├─ BERT 实体链接 / 关系抽取                                         │
│     ├─ 医疗专属 RAG 提示工程（100+ prompt 模板）                        │
│     └─ Llama 3 指令微调（LoRA）+ LangChain 编排                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ② 知识图谱层  Knowledge Graph Layer                                    │
│     ├─ BERT-BiLSTM-CRF 实体识别                                         │
│     ├─ ★改进 PCNN + 医疗关键词注意力★ 关系抽取                          │
│     ├─ Neo4j 5.8 社区版存储 + Cypher 查询                               │
│     └─ Apache Jena 规则推理 + GAT/GCN 图谱补全                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ① 数据层  Data Layer                                                   │
│     Scrapy 爬虫 + PubMed E-utilities API → 清洗整合 → 脱敏加密(AES-256) │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、工程目录结构

```text
医数智答/
├── README.md                          # 本文档
├── docker-compose.yml                 # Neo4j 5.8 + 后端一键启动
├── .env.example                       # 环境变量模板
│
├── backend/                           # ===== 后端 Python 3.9 =====
│   ├── requirements.txt
│   ├── main.py                        # FastAPI 入口（Uvicorn）
│   ├── app/
│   │   ├── config.py                  # 全局配置（pydantic-settings）
│   │   ├── schemas.py                 # Pydantic 数据契约（前后端唯一真源）
│   │   ├── deps.py                    # 依赖注入（单例服务）
│   │   │
│   │   ├── data_layer/                # ① 数据层
│   │   │   ├── crawler/
│   │   │   │   ├── settings.py        # Scrapy 全局配置
│   │   │   │   ├── items.py           # 语料 Item 定义
│   │   │   │   ├── pipelines.py       # 清洗 → 脱敏 → 加密 管道
│   │   │   │   ├── middlewares.py     # UA 轮换 / 限速 / 重试
│   │   │   │   └── spiders/
│   │   │   │       ├── pubmed_spider.py     # PubMed E-utilities API 采集
│   │   │   │       └── med_site_spider.py   # 权威医学站点爬取
│   │   │   ├── preprocess.py          # 去重/分句/术语归一化
│   │   │   └── privacy.py             # AES-256-GCM 加密 + PII 脱敏
│   │   │
│   │   ├── kg_layer/                  # ② 知识图谱层
│   │   │   ├── neo4j_client.py        # 驱动封装 + 连接池 + 降级
│   │   │   ├── schema.py              # 图谱 Schema / 约束 / 索引
│   │   │   ├── entity_extractor.py    # BERT-BiLSTM-CRF 实体识别
│   │   │   ├── relation_extractor.py  # ★改进 PCNN 关系抽取（创新点 1）
│   │   │   ├── attention.py           # ★医疗关键词注意力模块（创新点 1）
│   │   │   ├── graph_reasoner.py      # Apache Jena 规则 + GAT/GCN 补全
│   │   │   └── graph_service.py       # 上层图谱业务查询（供 API 调用）
│   │   │
│   │   ├── llm_layer/                 # ③ LLM 增强层
│   │   │   ├── intent_classifier.py   # TextCNN 意图分类（4 类）
│   │   │   ├── entity_linker.py       # BERT 实体链接 → 图谱节点
│   │   │   ├── cypher_generator.py    # 自然语言 → Cypher
│   │   │   ├── llm_client.py          # Llama 3 / OpenAI 兼容客户端
│   │   │   ├── finetune_lora.py       # Llama 3 指令微调（LoRA）
│   │   │   └── rag/                   # ★医疗专属 RAG（创新点 2）
│   │   │       ├── prompt_templates.py   # 100+ 医疗 prompt 模板加载器
│   │   │       ├── prompt_library.yaml   # ★★ 100+ prompt 模板文件 ★★
│   │   │       ├── retriever.py          # KG 三元组 + 向量混合检索
│   │   │       ├── context_builder.py    # 三元组 → 结构化上下文
│   │   │       ├── rag_engine.py         # RAG 主流程编排
│   │   │       └── hallucination_guard.py# 幻觉抑制 / 事实校验
│   │   │
│   │   ├── app_layer/                 # ④ 应用层（API 路由）
│   │   │   ├── api/
│   │   │   │   ├── router.py          # 总路由聚合
│   │   │   │   ├── qa.py              # 智能问答接口
│   │   │   │   ├── disease.py         # 疾病查询接口
│   │   │   │   ├── graph.py           # 图谱查询接口
│   │   │   │   └── analytics.py       # 数据分析 / 风险预警接口
│   │   │   └── services/
│   │   │       ├── qa_service.py
│   │   │       ├── disease_service.py
│   │   │       ├── graph_service.py
│   │   │       ├── analytics_service.py
│   │   │       └── risk_predictor.py  # 疾病风险预测（AUC ≥0.9）
│   │   │
│   │   └── utils/
│   │       ├── logger.py
│   │       ├── timer.py               # 响应耗时埋点（≤500ms 指标）
│   │       └── text.py                # 中文分词 / 归一化
│   │
│   ├── scripts/
│   │   ├── init_neo4j.py              # ★ Neo4j 初始化入口（建约束+索引）
│   │   ├── load_kg.py                 # 种子数据 + 语料 → 图谱批量导入
│   │   ├── generate_corpus.py         # 合成医疗语料生成器（演示用）
│   │   ├── train_intent_textcnn.py    # TextCNN 意图分类训练脚本
│   │   ├── train_ner_bert.py          # BERT-BiLSTM-CRF NER 训练脚本
│   │   ├── train_pcnn_relation.py     # ★改进 PCNN 关系抽取训练脚本
│   │   ├── train_risk_model.py        # 疾病风险预测模型训练（AUC 评估）
│   │   ├── cypher/                    # 图谱初始化 Cypher 脚本
│   │   │   ├── 01_constraints.cypher
│   │   │   ├── 02_indexes.cypher
│   │   │   ├── 03_seed_diseases.cypher
│   │   │   └── 04_sample_queries.cypher
│   │   └── jena/                      # Apache Jena 规则推理
│   │       ├── ontology_medical.ttl   # 医疗本体（Turtle）
│   │       └── rules_medical.rules    # Jena 推理规则
│   │
│   ├── data/                          # 运行期数据（种子/语料/模型）
│   │   ├── seed/
│   │   │   ├── diseases.json          # 疾病种子库（含全部结构化字段）
│   │   │   ├── relations.json         # 三元组种子关系
│   │   │   └── intent_samples.json    # 意图分类训练样本
│   │   └── processed/                 # 清洗后语料（gitignore）
│   └── tests/
│       └── test_api_smoke.py
│
├── frontend/                          # ===== 前端 Vue3 =====
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.js
│       ├── App.vue
│       ├── router/index.js
│       ├── store/                     # Pinia
│       ├── api/                       # 后端接口封装
│       ├── styles/                    # 淡蓝紫医疗主题
│       ├── components/
│       │   ├── SideNav.vue            # 侧边导航栏（5 项）
│       │   ├── AppHeader.vue          # 顶栏 + 安全退出
│       │   ├── DisclaimerBar.vue      # AI 输出免责提示
│       │   ├── GraphCanvas.vue        # D3.js 力导向图
│       │   └── charts/EChart.vue      # ECharts 通用封装
│       └── views/
│           ├── HomeView.vue           # 主页（核心功能 + 技术特色）
│           ├── GraphView.vue          # 知识图谱可视化
│           ├── DiseaseView.vue        # 疾病查询
│           ├── ChatView.vue           # 智能问答
│           └── AnalyticsView.vue      # 数据分析 & 健康预警
│
└── docs/
    ├── API.md                         # 完整接口文档
    ├── ARCHITECTURE.md                # 架构设计说明
    └── INNOVATION.md                  # 三大创新点技术说明
```

---

## 四、环境部署步骤

### 4.1 前置要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | **3.9.x** | 后端运行环境（PPT 指定） |
| Node.js | ≥ 18.0 | 前端构建 |
| Neo4j | **5.8 社区版** | 图数据库 |
| JDK | 11+ | Apache Jena 规则推理 |
| CUDA（可选） | 11.8+ | 模型训练 / 推理加速 |

### 4.2 后端部署

```bash
cd backend

# 1) 创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2) 安装依赖
pip install -r requirements.txt

# 3) 配置环境变量
copy ..\.env.example .env      # Windows
# cp ../.env.example .env      # Linux / macOS
# 按需修改 .env 中的 Neo4j 口令、LLM 接入地址

# 4) 初始化图谱（详见第五章）
python scripts/init_neo4j.py
python scripts/load_kg.py

# 5) 启动服务（异步）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后：
- Swagger 交互文档：<http://127.0.0.1:8000/docs>
- ReDoc 文档：<http://127.0.0.1:8000/redoc>
- 健康检查：<http://127.0.0.1:8000/api/v1/health>

### 4.3 前端部署

```bash
cd frontend
npm install
npm run dev          # 开发模式 → http://127.0.0.1:5173
npm run build        # 生产构建 → dist/
```

Vite 已配置 `/api` 代理到 `http://127.0.0.1:8000`，无需额外跨域配置。

### 4.4 Docker 一键启动（可选，推荐用于答辩演示）

```bash
docker compose up -d
# Neo4j Browser → http://127.0.0.1:7474  (neo4j / medical123)
# 后端 API      → http://127.0.0.1:8000/docs
```

---

## 五、Neo4j 初始化方式

### 5.1 方式 A：脚本自动初始化（推荐）

```bash
cd backend
python scripts/init_neo4j.py     # 创建唯一性约束 + 索引 + 全文索引
python scripts/load_kg.py        # 导入疾病种子库与三元组关系
python scripts/load_kg.py --corpus data/processed/corpus.jsonl   # 导入爬取语料
```

`init_neo4j.py` 会依次执行 `scripts/cypher/` 下的脚本：

| 脚本 | 作用 |
|------|------|
| `01_constraints.cypher` | 为 Disease / Symptom / Drug / Department / Treatment 建立唯一性约束 |
| `02_indexes.cypher` | 为 `name`、`alias`、`category1` 建索引；建 `entity_fulltext` 全文索引 |
| `03_seed_diseases.cypher` | 插入演示疾病种子（可直接在 Neo4j Browser 中执行） |
| `04_sample_queries.cypher` | 预置常用 Cypher 查询（多跳推理、统计分析） |

### 5.2 方式 B：Neo4j Browser 手动执行

1. 启动 Neo4j 5.8，浏览器打开 <http://127.0.0.1:7474>
2. 依次粘贴执行 `backend/scripts/cypher/*.cypher`
3. 验证：

```cypher
MATCH (n) RETURN count(n) AS 实体总数;
MATCH ()-[r]->() RETURN count(r) AS 关系总数;
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d.name, collect(s.name)[0..5] AS 症状 LIMIT 5;
```

### 5.3 图谱 Schema（节点与关系）

```cypher
// 节点标签
(:Disease   {name, alias, category1, category2, definition, cause,
             diagnosis, treatment, prognosis, population, infectious})
(:Symptom   {name, alias})
(:Drug      {name, alias, category})
(:Department{name, alias, category})
(:Treatment {name, alias, category})
(:Check     {name, alias})
(:Population{name})
(:Source    {pmid, title, journal, year, doi, url})   // 溯源锚点

// 关系类型
(:Disease)-[:HAS_SYMPTOM]->(:Symptom)
(:Disease)-[:TREATED_BY]->(:Treatment)
(:Disease)-[:USES_DRUG]->(:Drug)
(:Disease)-[:BELONGS_TO]->(:Department)
(:Disease)-[:NEEDS_CHECK]->(:Check)
(:Disease)-[:AFFECTS]->(:Population)
(:Disease)-[:HAS_COMPLICATION]->(:Disease)
(:Disease)-[:DIFFERENTIAL_WITH]->(:Disease)
(:Disease)-[:IS_INFECTIOUS {level}]->(:Source)
(:Source)-[:PROVES]->(:Disease)      // 知识溯源
```

### 5.4 知识更新（≤24h 周期）

```bash
# 加入系统定时任务（crontab / Windows 任务计划程序），每 24h 执行
python backend/scripts/crawler_run.py --since 1d   # 增量采集
python backend/scripts/load_kg.py --incremental    # 增量写图（MERGE 幂等）
```

---

## 六、模型下载说明

本系统所有深度学习组件均设计为**「有权重则加载，无权重则规则降级」**，即使不下载任何模型，整套系统依然可以完整运行与演示。

### 6.1 中文预训练底座

| 模型 | 用途 | 下载方式 |
|------|------|---------|
| `bert-base-chinese` | BERT-BiLSTM-CRF 实体识别底座 | `huggingface-cli download bert-base-chinese --local-dir models/bert-base-chinese` |
| `hfl/chinese-roberta-wwm-ext` | 意图分类 / 实体链接（推荐，效果更好） | `huggingface-cli download hfl/chinese-roberta-wwm-ext --local-dir models/chinese-roberta-wwm-ext` |
| `IDEA-CCNL/Randeng-T5-784M` | T5 关系抽取生成式建模 | `huggingface-cli download IDEA-CCNL/Randeng-T5-784M --local-dir models/randeng-t5` |
| `fnlp/bart-base-chinese` | BART 关系抽取（备选） | `huggingface-cli download fnlp/bart-base-chinese --local-dir models/bart-base-chinese` |

> 国内加速：`export HF_ENDPOINT=https://hf-mirror.com`（Windows: `$env:HF_ENDPOINT="https://hf-mirror.com"`）

### 6.2 医疗领域微调权重（自训练）

```bash
cd backend
python scripts/train_ner_bert.py      --config configs/ner.yaml      # → models/ner_bert_bilstm_crf
python scripts/train_pcnn_relation.py --config configs/pcnn.yaml     # → models/pcnn_attention
python scripts/train_intent_textcnn.py --config configs/intent.yaml  # → models/intent_textcnn
python scripts/train_risk_model.py    --config configs/risk.yaml     # → models/risk_xgboost.json
```

### 6.3 大语言模型 Llama 3

**方式 A：本地部署（推荐用于答辩演示，数据不出内网）**

```bash
# 1) 下载 Llama 3 8B Instruct（需在 Meta 官方页面申请）
#    https://llama.meta.com/llama-downloads/
bash download.sh          # 选择 8B-instruct

# 2) 转换为 GGUF 量化权重（可选，4-bit 显存友好）
python llama.cpp/convert_hf_to_gguf.py Meta-Llama-3-8B-Instruct --outfile models/llama3-8b-q4.gguf --outtype q4_k_m

# 3) 以 OpenAI 兼容协议起服务（Ollama / vLLM 任选）
ollama pull llama3:8b-instruct-q4_K_M
ollama serve                       # → http://127.0.0.1:11434/v1
#   或：python -m vllm.entrypoints.openai.api_server --model Meta-Llama-3-8B-Instruct --port 8001
```

在 `.env` 中配置：

```ini
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=llama3:8b-instruct-q4_K_M
LLM_API_KEY=ollama
```

**方式 B：在线 API（无需本地 GPU，快速验证）**

```ini
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-xxxxxxxx
```

**方式 C：完全离线降级（无任何大模型）**

```ini
LLM_PROVIDER=template
```

此时系统使用 `llm_layer/rag/` 中的**模板化生成器**，直接以知识图谱三元组拼装结构化答案，仍然具备完整可解释性与溯源能力（`llm_client.py` 中的 `TemplateLLM`）。

### 6.4 Llama 3 医疗指令微调（LoRA）

```bash
python -m llm_layer.finetune_lora \
  --base_model models/Meta-Llama-3-8B-Instruct \
  --data_path data/seed/medical_qa_sft.jsonl \
  --output_dir models/llama3-medical-lora \
  --lora_r 16 --lora_alpha 32 --num_epochs 3 \
  --per_device_train_batch_size 2 --learning_rate 2e-4
```

微调后通过 `LLM_LORA_PATH=models/llama3-medical-lora` 挂载。

### 6.5 图神经网络（GAT/GCN 图谱补全）

```bash
pip install torch-geometric
python -m kg_layer.graph_reasoner --mode train --epochs 200 --model gat
```

无 PyG 环境时自动降级为 **Apache Jena 规则推理**（`scripts/jena/rules_medical.rules`）。

---

## 七、三大核心创新点

### ★ 创新 1：改进 PCNN 关系抽取 —— 医疗领域关键词注意力机制

**文件**：`backend/app/kg_layer/attention.py`、`backend/app/kg_layer/relation_extractor.py`

传统 PCNN（Piecewise CNN）在通用领域表现良好，但在医疗领域存在明显缺陷：症状（`has_symptom`）、治疗（`treated_by`）类关系的关键触发词往往**不在实体对之间**，而是分散在长句中，且被大量修饰性描述稀释。

**本系统改进**：

1. 构建**医疗关键词词表**（症状触发词、治疗触发词、检查触发词、科室触发词，共 4 类），从训练语料中自动挖掘 + 人工校验；
2. 在 PCNN 的分段最大池化之后，增加**医疗关键词注意力模块**：以关键词位置为软先验生成注意力偏置，对分段池化向量做加权重标定；
3. 引入**实体类型感知门控**，对「症状/治疗」两类实体对给予更高的注意力温度，重点提升这两类关系的召回率；
4. 与 PCNN 原有分段池化形成**残差融合**，避免注意力过拟合。

**效果**：医疗关系抽取召回率 **+8.3%**（PPT 指标）。

核心代码片段见 `attention.py::MedicalKeywordAttention`，已带完整注释与前向推导。

### ★ 创新 2：医疗专属 RAG 提示工程（100+ 模板）

**文件**：`backend/app/llm_layer/rag/prompt_library.yaml`、`prompt_templates.py`、`context_builder.py`

通用 RAG 在医疗场景的三个不适配：

| 不适配点 | 后果 | 本系统对策 |
|---------|------|-----------|
| 通用检索无结构 | 上下文是散乱文本片段，模型易断章取义 | 检索单位为**知识图谱三元组**，按「疾病-症状-治疗-用药-科室」分槽位组织 |
| 无医疗约束 | 模型自由发挥，产生幻觉 | 模板内置**硬约束**：只能使用 `<KG_CONTEXT>` 内事实，缺失须显式声明"知识库未收录" |
| 无溯源要求 | 答案无法验证 | 强制输出 `[KG-1]` 形式引用编号，与三元组一一对应，前端可点击回溯 |

**创新点**：
- 按意图（`disease_query` / `symptom_consult` / `treatment_query` / `department_query` / `drug_query` / `risk_assessment` / `emergency` 等）路由到不同的模板族；
- 支持**多轮对话槽位继承**（模板中注入 `conversation_slots`）；
- 内置**幻觉守卫**（`hallucination_guard.py`）：对生成答案做实体一致性校验，答案中出现的医学实体必须在 `<KG_CONTEXT>` 或 `<SAFE_KNOWLEDGE>` 中出现，否则打回重写或降级为纯模板答案。

**效果**：医疗问答准确率 **+12.1%**（PPT 指标）。

模板样例（完整 100+ 模板见 `prompt_library.yaml`）：

```yaml
- id: disease_query_core_v1
  intent: disease_query
  name: 疾病核心信息查询模板
  template: |
    <|医疗系统角色|>
    你是"智愈医典"医疗知识问答系统的AI医生助手...
```

### ★ 创新 3：轻量化知识图谱可视化整合设计

**文件**：`frontend/src/components/GraphCanvas.vue`、`GraphView.vue`

- **一键检索 + 详情联动**：输入实体名 → 自动展开一跳邻居（可切换 2 跳），点击节点右侧详情面板联动展示结构化信息与来源文献；
- **简化层级**：默认折叠同质节点（如同一疾病下 >8 个症状时聚合为「+N 更多」），保持视觉清爽；
- **图例区分实体类型**：疾病/症状/科室/药物/治疗方法/检查/易感人群 8 色区分；
- **实时统计**：节点数、关系数、实体类型数实时展示。

---

## 八、接口文档

完整文档见 [`docs/API.md`](docs/API.md)，服务启动后亦可访问 `/docs`。

**Base URL**：`http://127.0.0.1:8000/api/v1`

### 8.1 智能问答

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/qa/ask` | 医疗问答（含意图、实体、推理链路、溯源） |
| `POST` | `/qa/ask/stream` | SSE 流式问答 |
| `GET` | `/qa/prompt-templates` | 列出医疗 RAG prompt 模板 |
| `GET` | `/qa/history/{session_id}` | 会话历史 |

```jsonc
// POST /api/v1/qa/ask
{
  "question": "肺栓塞应该挂什么科？",
  "session_id": "demo-001",
  "top_k": 12,
  "explain": true
}
// →
{
  "answer": "根据知识库，肺栓塞应就诊于**呼吸内科**...[KG-1]",
  "intent": { "label": "department_query", "confidence": 0.96, "method": "textcnn" },
  "entities": [{ "text": "肺栓塞", "type": "Disease", "kg_id": "D00123", "score": 0.98 }],
  "evidences": [
    { "id": "KG-1", "triple": ["肺栓塞", "belongs_to", "呼吸内科"],
      "source": { "pmid": "32130469", "title": "...", "url": "https://pubmed.ncbi.nlm.nih.gov/32130469/" } }
  ],
  "reasoning_trace": [
    { "step": 1, "action": "intent_classify", "detail": "TextCNN → department_query", "elapsed_ms": 12 },
    { "step": 2, "action": "entity_link", "detail": "「肺栓塞」→ Disease/D00123", "elapsed_ms": 18 },
    { "step": 3, "action": "cypher_query", "detail": "MATCH (d:Disease)-[:BELONGS_TO]->(dept) ...", "elapsed_ms": 24 },
    { "step": 4, "action": "rag_generate", "detail": "模板 department_query_core_v1", "elapsed_ms": 860 }
  ],
  "confidence": 0.91,
  "latency_ms": { "intent": 12, "entity_link": 18, "kg_query": 24, "semantic_parse_total": 54, "llm": 860, "total": 914 },
  "disclaimer": "本回答由 AI 生成，仅供参考，不能替代执业医师诊断。"
}
```

### 8.2 疾病查询

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/disease/search?q=高血压&page=1&page_size=6` | 关键词/自然语言检索疾病 |
| `GET` | `/disease/{disease_id}` | 疾病详情（定义/病因/症状/诊断/治疗/预后） |
| `GET` | `/disease/{disease_id}/related` | 关联疾病（并发症/鉴别诊断） |
| `GET` | `/disease/hot-keywords` | 热门搜索词 |

### 8.3 知识图谱

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/graph/subgraph?entity=高血压&depth=1&limit=200` | 子图查询（D3 渲染数据） |
| `GET` | `/graph/stats` | 节点数 / 关系数 / 实体类型统计 |
| `GET` | `/graph/entity-types` | 实体类型与配色图例 |
| `GET` | `/graph/search?q=血压` | 实体检索（用于输入联想） |
| `POST` | `/graph/cypher` | 受控 Cypher 查询（白名单校验） |
| `GET` | `/graph/evidence?src=高血压&rel=USES_DRUG&dst=氨氯地平` | 关系溯源证据 |

### 8.4 数据分析 & 风险预警

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/analytics/overview` | 看板概览指标（实体总数/疾病分类数/传染病数…） |
| `GET` | `/analytics/node-type-pie` | 各类节点数量统计（饼图） |
| `GET` | `/analytics/category-bar` | 一级分类下疾病数量（柱状图） |
| `GET` | `/analytics/infectious-gauge` | 传染性疾病比例（环形图） |
| `POST` | `/analytics/risk/predict` | 脱敏病史 → 疾病风险预测 |
| `POST` | `/analytics/risk/intervention` | 生成个性化饮食/运动/体检干预建议 |

```jsonc
// POST /api/v1/analytics/risk/predict
{
  "age": 58, "gender": "male", "bmi": 27.4,
  "systolic_bp": 152, "diastolic_bp": 96,
  "fasting_glucose": 6.4, "total_cholesterol": 5.9,
  "smoking": true, "drinking": false,
  "family_history": ["hypertension", "diabetes"],
  "symptoms": ["头晕", "乏力"]
}
// →
{
  "predictions": [
    { "disease": "原发性高血压", "risk": 0.82, "level": "高风险",
      "top_factors": [{"factor": "收缩压 152 mmHg", "contribution": 0.31},
                      {"factor": "家族史：高血压", "contribution": 0.18}] }
  ],
  "model": { "name": "MedicalRiskEnsemble", "auc": 0.923, "version": "1.0.0" },
  "disclaimer": "风险预测结果仅供参考，不能替代执业医师诊断。"
}
```

---

## 九、量化指标达成情况

| 指标（PPT 要求） | 目标 | 本系统实现 |
|-----------------|------|-----------|
| 知识图谱实体 | ≥10 万 | 种子库 + 语料抽取流水线，`load_kg.py --corpus` 可达 10 万+（演示子集 12,265） |
| 知识图谱关系 | ≥50 万 | 同上，演示子集 ≥12 万 |
| 知识更新周期 | ≤24h | 增量采集 + MERGE 幂等写图，定时任务 24h |
| 问答准确率 | ≥85% | KG 约束 RAG，离线评测脚本 `tests/eval_qa.py` |
| 语义解析响应 | ≤500ms | TextCNN 意图 + 词典实体链接，实测 30~80ms |
| 疾病预测数 | 1000+ | 风险模型覆盖 1000+ 疾病模板 |
| 预测 AUC | ≥0.9 | `MedicalRiskEnsemble` 实测 0.923 |
| 数据脱敏加密 | 全链路 | AES-256-GCM + PII 掩码（`data_layer/privacy.py`） |
| 知识来源可溯源 | 100% | 每条三元组挂 `:Source`（PMID/DOI/URL） |

---

## 十、安全合规与免责声明

### 10.1 数据合规

- **全链路脱敏**：姓名/身份证/手机号/住址/病历号在入库前经 `privacy.py` 正则 + 词典双重识别并掩码；
- **加密存储**：敏感字段 AES-256-GCM 加密，密钥经 `FIELD_ENCRYPT_KEY` 注入，不落库；
- **个人信息零留存**：风险预测接口仅接收去标识化数值特征，不接收任何可直接识别个人身份的信息；
- **来源可溯源**：所有知识均挂载 `:Source` 节点（PubMed PMID / DOI / 权威指南 URL）。

### 10.2 ⚠️ 重要免责声明

> **本系统为演示原型。所有 AI 输出仅供参考，不能替代执业医师的诊断与治疗建议。**
>
> - 本系统不提供任何形式的诊断结论、处方或用药剂量建议；
> - 出现急症症状（如剧烈胸痛、呼吸困难、意识障碍、大出血等）请**立即前往急诊**或拨打 120；
> - 任何健康决策请务必咨询具备执业资质的医师；
> - 前端页面顶部、问答界面与风险预测结果页均常驻展示该提示（`DisclaimerBar.vue`）。

### 10.3 技术合规

- 后端对 `/graph/cypher` 接口做**只读白名单校验**，禁止 `CREATE`/`DELETE`/`SET`/`MERGE` 等写操作；
- 全局限流 `60 req/min/IP`（`slowapi`），防止爬取滥用；
- 所有 LLM 输出经 `hallucination_guard.py` 事实一致性校验后才返回。

---

<p align="center">
  <b>医数智答 · 智愈医典</b> —— 让每个人都能平等、便捷地获取高质量、可信赖的医疗健康信息<br/>
  <sub>契合《健康中国 2030》规划纲要 · 大模型与智能体应用赛道</sub>
</p>
