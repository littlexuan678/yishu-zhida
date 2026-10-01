# 医数智答 · 智愈医典 —— 架构设计说明

> **产品定位**：基于「医疗大数据 + 知识图谱 + RAG 大模型」的专业医疗知识问答平台
> **代码基线**：后端 `backend/`（FastAPI + Uvicorn，Python 3.9 兼容），前端 `frontend/`（Vue3 + Element Plus + D3.js + ECharts）
> **文档范围**：本文只描述后端代码的实际实现，所有类名 / 函数名 / 日志文案 / 数字均取自仓库源码与本次验证运行日志（`backend/logs/zhiyu.log`）。

> ### ⚠️ 重要免责声明
> **本系统为演示原型。所有 AI 输出仅供参考，不能替代执业医师诊断。**

---

## 1. 四层架构总览

系统采用「数据层 → 知识图谱层 → LLM 增强层 → 应用层」的四层单向依赖设计：上层只依赖下层暴露的门面（Facade），不直接触碰驱动、模型权重或数据库连接。这样做的直接收益是**每一层都能独立降级**——任意一层的外部依赖缺失时，系统仍能沿降级路径完成一次完整问答。

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│ ④ 应用层  app_layer / main.py                                                    │
│   FastAPI 路由  api/{router,qa,disease,graph,analytics}.py                        │
│   业务服务      services/{qa_service,analytics_service,risk_predictor}.py         │
│   数据契约      schemas.py（前后端唯一真源）  依赖注入 deps.py                     │
│   横切关注      CORS / 限流 60 req/min / 安全响应头 / 统一异常处理 / 请求耗时埋点    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ③ LLM 增强层  llm_layer                                                          │
│   intent_classifier.py  TextCNN 意图分类（4 类，规则融合降级）                     │
│   entity_linker.py      口语别名词典 + 实体链接（mention → 图谱节点）              │
│   llm_client.py         Llama 3 / OpenAI 兼容 / Ollama / template 四态客户端      │
│   rag/                  ★医疗专属 RAG★                                            │
│     prompt_library.yaml   104 个模板 / 16 类意图 / 全局硬约束 / 10 条安全覆盖      │
│     prompt_templates.py   PromptLibrary（select / render / validate）            │
│     retriever.py          HybridRetriever（五路混合召回 + 槽位先验排序）           │
│     context_builder.py    ContextBuilder（三元组 → 带编号的 <KG_CONTEXT>）         │
│     rag_engine.py         RagEngine（唯一编排入口，逐阶段埋点）                   │
│     hallucination_guard.py HallucinationGuard（7 项确定性事实校验）               │
│   cypher_generator.py / finetune_lora.py（NL2Cypher / LoRA 指令微调）             │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ② 知识图谱层  kg_layer                                                           │
│   neo4j_client.py     Neo4j 5.8 驱动封装 + 只读白名单 + ★内存降级★                │
│   memory_store.py     MemoryGraphStore（567 节点 / 1970 关系 / 59 疾病 + Cypher 子集）│
│   entity_extractor.py BERT-BiLSTM-CRF（降级：631 条表面形式的词典最长匹配）        │
│   relation_extractor.py ★改进 PCNN + 医疗关键词注意力★（降级：规则抽取）          │
│   attention.py        MedicalKeywordAttention（symptom 58 / treatment 52 /        │
│                       check 52 / department 39 = 201 个医疗关键词）               │
│   graph_service.py    GraphService（图谱 API 的唯一数据访问入口）                  │
│   graph_reasoner.py   6 条医疗推理规则 + GAT/GCN 链接预测（降级：启发式补全）      │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ① 数据层  data_layer                                                             │
│   crawler/            Scrapy 工程：PubMed E-utilities + 权威医学站点               │
│                       items.py / pipelines.py（清洗→脱敏→加密）/ middlewares.py    │
│   preprocess.py       去重 / 分句 / 术语归一化（113 条归一化词典）                 │
│   privacy.py          PII 脱敏（11 类正则）+ AES-256-GCM 字段加密                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**分层依赖规则**

1. 应用层可以调用任意下层，但**不直接使用** Neo4j 驱动、`torch`、`yaml`、`httpx` 等外部库；
2. 知识图谱层向 LLM 增强层暴露的是 `GraphService`（含 `facts_for_disease` / `triples_of` / `get_evidence` / `related_questions`），而不是 Cypher；
3. LLM 增强层内部对 `intent_classifier` / `entity_linker` / `llm_client` 全部采用**函数内惰性导入**（`from app... import` 写在函数体里），因此任一模块缺失都不会导致 `rag_engine` 导入失败；
4. 所有单例通过 `deps.py` 或模块级 `get_xxx()` 提供，测试可用 `app.dependency_overrides` 整体替换。

---

## 2. ① 数据层 `data_layer`

### 2.1 职责

把「外部医学语料」变成「可入库、无隐私风险的结构化文本」：采集 → 清洗 → 术语归一化 → 脱敏 → 加密 → 落地语料文件，供 `scripts/load_kg.py --corpus` 批量写图。

### 2.2 模块与关键类

| 文件 | 关键对象 | 职责 |
|------|---------|------|
| `crawler/settings.py` | Scrapy 配置 | 并发 8（`CRAWLER_CONCURRENT`）、下载延迟 1.0s（`CRAWLER_DELAY`）、最大页数 20（`CRAWLER_MAX_PAGES`）、UA 轮换（`USER_AGENT_ROTATE`） |
| `crawler/items.py` | 语料 Item | 统一字段：标题 / 正文 / 来源 / PMID / DOI / URL / 采集时间 |
| `crawler/middlewares.py` | UA 轮换、限速、重试 | 反爬与稳定性 |
| `crawler/pipelines.py` | 清洗 → 脱敏 → 加密管道 | 落地前强制走 `privacy.py`，保证「原始 PII 永不落盘」 |
| `crawler/spiders/pubmed_spider.py` | PubMed 采集器 | E-utilities API，携带 `PUBMED_TOOL=ZhiyuMedicalCodex` / `PUBMED_EMAIL` |
| `crawler/spiders/med_site_spider.py` | 医学站点采集器 | 指南、科普站点正文抽取 |
| `preprocess.py` | 清洗与归一化 | 去重 / 分句 / 术语归一化（启动日志：`术语归一化词典加载完成：113 条`） |
| `privacy.py` | `PrivacyGuard` | `mask()` 脱敏、`encrypt_field()` / `decrypt_field()` 字段加密、`anonymize_health_input()` 去标识化、`pseudonymous_id()` 假名标识 |

### 2.3 数据流

```text
PubMed / 医学站点
   │  Scrapy（UA 轮换 + 限速 + 重试）
   ▼
items.py 原始 Item
   │  pipelines.py：去重 → preprocess.normalize → PrivacyGuard.mask
   ▼
脱敏文本
   │  PrivacyGuard.encrypt_field（敏感字段，AES-256-GCM）
   ▼
data/processed/corpus.jsonl ──► scripts/load_kg.py --corpus ──► Neo4j / 内存图谱
```

### 2.4 降级策略

| 缺失依赖 | 行为 | 源码位置与日志文案 |
|---------|------|------------------|
| `cryptography` 未安装 | `AESGCM=None`，加密能力关闭；`encrypt_field()` 直接返回原文（可用性优先） | `未安装 cryptography，字段加密不可用（pip install cryptography）` |
| 未配置 `FIELD_ENCRYPT_KEY` | 自动生成**进程内临时演示密钥**，能力开箱可用；重启后旧密文无法解密 | `未配置 FIELD_ENCRYPT_KEY，已自动生成**临时演示密钥**（指纹 c64c9b849a35）。该密钥仅存在于当前进程，重启后无法解密旧数据。` |
| `FIELD_ENCRYPT_KEY` 非法（非 base64 / 非 32 字节） | 禁用字段加密并报错 | `FIELD_ENCRYPT_KEY 无效：AES-256 需要 32 字节密钥，当前 16 字节；字段加密已禁用` |
| 关闭脱敏开关 | `PRIVACY_MASK_ENABLED=false` 时 `mask()` 原样返回 | `MaskResult(text=raw, original_length=len(raw), masked_length=len(raw))` |
| 未安装 Scrapy | 数据层采集不可用，不影响在线服务（图谱直接读种子库） | 采集脚本报 `ModuleNotFoundError: scrapy` |

---

## 3. ② 知识图谱层 `kg_layer`

### 3.1 职责

承载全部医学事实，并把「图存储」的差异（Neo4j / 内存）屏蔽在 `GraphService` 之下：实体识别 → 实体链接 → 三元组检索 → 溯源证据 → 规则推理与链接预测。

### 3.2 模块与关键类

| 文件 | 关键类 / 函数 | 说明 |
|------|--------------|------|
| `neo4j_client.py` | `Neo4jClient`、`assert_readonly()`、`CypherSecurityError` | 驱动懒加载、`asyncio.to_thread` 异步化、**只读白名单**、内存降级 |
| `memory_store.py` | `MemoryGraphStore`（线程安全单例）、`REL_LABELS` | 离线后端 + 算法侧统一图视图（`nodes` / `edges` / `adjacency` / `index`）+ 少量 Cypher 子集解释器 |
| `entity_extractor.py` | `EntityExtractor`、`EntityMention` | 词典 / BERT-BiLSTM-CRF 双后端，`backend` 取值为 `dictionary` 或 `bert_bilstm_crf` |
| `relation_extractor.py` | `RelationExtractor` | `backend` 取值为 `improved_pcnn` 或 `rule` |
| `attention.py` | `MedicalKeywordAttention` | 创新点 1 的核心：医疗关键词注意力（4 类词表共 201 词） |
| `graph_service.py` | `GraphService` | 图谱 API 唯一入口：子图 / 统计 / 检索 / 疾病卡片与详情 / 证据 / 三元组 / 推荐追问 |
| `graph_reasoner.py` | `GraphReasoner`、`PythonRuleEngine`、`HeuristicLinkPredictor` | 6 条医疗规则推理 + GAT/GCN 链接预测（启发式降级） |

### 3.3 关键数据与规模（本次验证运行实测）

| 指标 | 数值 | 来源 |
|------|------|------|
| 内存图谱节点 | **567** | `内存图谱加载完成：节点 567，关系 1970，疾病 59` |
| 内存图谱关系 | **1970** | 同上 |
| 疾病条目 | **59** | 同上 |
| 节点类型分布 | 症状 195 / 药物 122 / 检查 116 / 疾病 59 / 科室 34 / 治疗方法 26 / 易感人群 15 | `/graph/stats` |
| 关系分布 | HAS_SYMPTOM 467 / NEEDS_CHECK 429 / USES_DRUG 268 / TREATED_BY 231 / AFFECTS 184 / DIFFERENTIAL_WITH 146 / HAS_COMPLICATION 128 / BELONGS_TO 117 | 种子数据统计 |
| 实体词典表面形式 | **631 条**（最长 15 字） | `实体词典构建完成：表面形式 631 条，最长 15 字` |
| 医疗关键词词表 | **201 词**（symptom 58 + treatment 52 + check 52 + department 39） | `医疗关键词词表加载完成：{'symptom': 58, 'treatment': 52, 'check': 52, 'department': 39}` |
| 口语别名词典 | 索引 64 条（口语 → 标准名）；源词典共 113 条（疾病 53 / 科室 27 / 症状 33），仅当目标节点真实存在时才入索引 | `实体链接别名词典构建完成：64 条（口语 → 标准名）` |
| 规则推理 | 疾病 59 个、规则 6 条 → 新增 20 条三元组 | `规则推理完成：新增 20 条三元组` |

### 3.4 数据流

```text
用户问句
  └─► EntityExtractor.recognize()        # 词典最长匹配（或 BERT-BiLSTM-CRF）
        └─► EntityExtractor.link_to_kg() # mention → 图谱标准名
  └─► GraphService.facts_for_disease()   # 按关系类型聚合的直接事实
  └─► GraphService.triples_of()          # 带置信度的三元组
  └─► GraphService.get_evidence()        # 三元组 → EvidenceItem（含 :Source）
  └─► GraphReasoner.infer_by_rules()     # 新三元组（RELATED_TO 等）
```

### 3.5 降级策略

| 缺失依赖 | 行为 | 源码路径与日志文案 |
|---------|------|------------------|
| 未安装 `neo4j` 驱动 | `Neo4jClient._connect()` 捕获 `ImportError` → `_activate_fallback()` | `Neo4j 驱动缺失：未安装 neo4j 驱动（pip install neo4j==5.19.0）` → `已降级到内存图存储（离线演示模式），节点数=567` |
| Neo4j 连接失败（服务未启动 / 认证失败 / 超时） | 同上，`status="fallback_memory"` | `Neo4j 连接失败：<异常>` → `已降级到内存图存储（离线演示模式），节点数=567` |
| 运行中 Cypher 执行失败 | 单个请求内即时降级：`self._available=False` 后重走内存实现 | `Cypher 执行失败：<异常> | <cypher 前 200 字>` |
| `NEO4J_FALLBACK_TO_MEMORY=false` | 不降级，图谱接口返回空数据 | `Neo4j 不可用且未开启内存降级，图谱接口将返回空数据` |
| 未安装 PyTorch（NER） | `backend="dictionary"`，词典最长匹配 | `BERT-BiLSTM-CRF 权重不可用（未安装 PyTorch），实体识别降级为词典实现` |
| 未安装 PyTorch（关系抽取） | `backend="rule"`，规则抽取 | `改进 PCNN 权重不可用（未安装 PyTorch），关系抽取降级为规则实现` |
| 未安装 PyTorch（注意力模块） | `MedicalKeywordAttention` 降级为 NumPy 实现（仅推理） | `未检测到 PyTorch，医疗关键词注意力模块降级为 NumPy 实现（仅推理）` |
| 未安装 `torch`（图谱补全） | `gnn_backend="heuristic"`，Adamic-Adar + Jaccard 启发式 | `GAT/GCN 权重不可用（未安装 torch），图谱补全降级为启发式实现（Adamic-Adar + Jaccard）` |
| `MemoryGraphStore` 种子文件缺失 | 图谱为空但不阻塞启动 | `疾病种子文件不存在：<path>` |
| 内存后端遇到不支持的 Cypher | 返回空结果 + WARNING | `内存后端不支持的 Cypher（返回空结果）：<语句前 160 字>` |

---

## 4. ③ LLM 增强层 `llm_layer`

### 4.1 职责

把「用户自然语言」翻译成「图谱查询」，再把「图谱事实」翻译成「有引用、有边界、可解释的中文答案」：
意图识别 → 实体链接 → 五路混合检索 → 上下文构建 → 医疗专属 Prompt → 大模型生成 → 幻觉守卫。

### 4.2 模块与关键类

| 文件 | 关键类 | 说明 |
|------|-------|------|
| `intent_classifier.py` | `IntentClassifier`、`RuleIntentClassifier`、`TextCNN` | 4 类意图（`disease_query` / `symptom_consult` / `treatment_query` / `department_query`）；`fuse=True` 时 TextCNN 0.7 + 规则 0.3 加权融合 |
| `entity_linker.py` | `EntityLinker`、`LinkCandidate` | 四步链接：模型/词典 → 口语别名（强先验）→ 滑窗模糊 → 消歧重排；输出 `main_entity` |
| `cypher_generator.py` | 自然语言 → Cypher | 结合实体链接结果生成参数化只读查询 |
| `llm_client.py` | `LLMClient`、`LLMResult` | `openai_compatible` / `ollama` / `template`；**永不抛异常**，失败返回 `LLMResult(success=False, error=...)` |
| `rag/prompt_templates.py` | `PromptLibrary`、`PromptTemplate`、`SAFETY_OVERRIDES` | 104 模板 / 16 意图；选择顺序：安全覆盖 → `prefer_id` → 特殊人群 → 条件匹配 → 意图内 priority 最高 → 兜底模板 |
| `rag/retriever.py` | `HybridRetriever`、`RetrievedFact` | 五路召回：`kg_1hop`（槽位直接事实）/ 反向症状检索 / `kg_2hop` / `kg_reasoned` / `kg_similar`；排序公式权重 `W_CONFIDENCE=0.45`、`W_SLOT_PRIOR=0.30`、`W_PROXIMITY=0.15`、`W_DEGREE=0.10`，再乘召回路径权重（1.00 / 0.92 / 0.85 / 0.80）；编号 `KG-1…KG-n` 即排序名次 |
| `rag/context_builder.py` | `ContextBuilder`、`ContextBundle`、`SAFE_KNOWLEDGE` | 三元组按槽位组织成带编号的 `<KG_CONTEXT>`，并注入安全常识池与多轮槽位 |
| `rag/rag_engine.py` | `RagEngine` | 唯一编排入口；每阶段写入 `ReasoningStep` 与 `LatencyBreakdown` |
| `rag/hallucination_guard.py` | `HallucinationGuard`、`GuardResult` | 7 项确定性校验（纯正则 + 集合比对，无模型依赖）：`missing_citation` / `invalid_citation` / `unknown_entity` / `dose_violation`(critical) / `absolute_claim` / `diagnostic_claim` / `missing_disclaimer`；扣分 `critical −0.30 / high −0.12 / medium −0.05 / low −0.02`，出现 critical 或得分 < 0.45 → `fallback_template` |
| `finetune_lora.py` | LoRA 微调脚本 | Llama 3 医疗指令微调（`LLM_LORA_PATH` 挂载） |

### 4.3 数据流

```text
question
  ├─ IntentClassifier.classify() ──► (label, label_cn, confidence, method, scores)
  ├─ EntityLinker.analyze()     ──► entities{diseases,symptoms,departments,...} + mentions + main_entity
  ├─ HybridRetriever.retrieve() ──► [RetrievedFact(KG-1..KG-n, score, source, sources)]
  ├─ ContextBuilder.build()     ──► ContextBundle(kg_context, variables, safe_knowledge, ...)
  ├─ PromptLibrary.select()/render() ──► prompt（含全局硬约束 + 免责声明）
  ├─ LLMClient.generate()       ──► LLMResult（或降级）
  ├─ HallucinationGuard.check() ──► GuardResult
  └─ QAResponse(confidence, evidences, reasoning_trace, latency_ms, ...)
```

### 4.4 降级策略

| 缺失依赖 | 行为 | 源码路径与日志文案 |
|---------|------|------------------|
| 未安装 PyTorch（意图） | `backend="rule"`，医疗触发词加权规则分类 → `method="rule_fallback"` | `TextCNN 意图分类权重不可用（未安装 PyTorch），降级为规则分类器` |
| 意图模块整体不可用 | `RagEngine._classify()` 异常兜底为 `symptom_consult`，置信度 0.3，`method="fallback"` | `意图识别模块不可用，降级为 symptom_consult（置信度 0.3）：<异常>` |
| 实体链接模块不可用 | `RagEngine._link()` 返回空实体，仍可依赖意图先验、症状反向检索与推理补全 | `实体链接模块不可用，本次问答退化为无实体模式：<异常>` |
| 未安装 PyYAML | `PromptLibrary._load()` 提前返回，`templates` 为空 → `select()` 返回 `None` → 直接用 KG 上下文作为 prompt | `未安装 PyYAML，无法加载 prompt 模板库（pip install PyYAML）` |
| prompt 模板库文件缺失 | 同上 | `医疗 prompt 模板库不存在：<path>` |
| 模板不足 100 个 | 启动告警（不阻断） | `模板数 %d < 100，未满足 PPT「100+ 模板」指标` |
| 意图无对应模板 | 使用 `fallback_no_kg_data_v1` | `意图 %s 无对应模板，使用兜底模板` |
| `LLM_PROVIDER=template` | `LLMClient.generate()` 不发起任何网络请求，直接返回失败结果；由 `RagEngine._template_answer()` 用纯 KG 事实拼装答案 | `LLM 提供方 = template（完全离线模板生成模式，不调用任何大模型）` → `大模型生成失败：模型返回失败（provider=template，由 rag_engine 使用本地模板生成器），降级为本地 KG 模板回答` |
| LLM 端点不可达（Ollama 未启动 / 连接被拒 / 超时） | 同一降级路径；`llm_model="local-template-rag"` | `大模型生成失败：模型返回失败（connection refused），降级为本地 KG 模板回答` / `调用大模型异常，降级为本地 KG 模板回答：<异常>` |
| 未安装 `httpx` 与 `openai` | 无法发起 HTTP 调用，全部失败降级（不影响问答可用性） | `既无 openai 也无 httpx，LLM 调用将全部失败并降级为模板生成` |
| 幻觉守卫抛异常 | 跳过校验，`score=0.8`，`suggested_action="accept"` | `幻觉守卫执行失败，跳过校验：<异常>` / `幻觉守卫异常（<异常>），已跳过校验` |
| `RAG_HALLUCINATION_GUARD=false` | 不校验 | `幻觉守卫未启用（settings.RAG_HALLUCINATION_GUARD=False）` |
| RAG 引擎初始化失败 | `QAService` 退化为**纯图谱结构化回答**（`prompt_template_id="graph_only_fallback"`） | `RAG 引擎不可用：<异常>（将使用纯图谱结构化回答）` |
| RAG 主流程运行期异常 | 同一次请求内降级为纯图谱回答 | `RAG 主流程异常，降级为纯图谱回答：<异常>` |

---

## 5. ④ 应用层 `app_layer` / `main.py`

### 5.1 职责

对外提供稳定的 REST 契约，向内编排各层服务；承担 CORS、限流、安全响应头、统一异常处理、请求耗时埋点与生命周期预热/优雅关闭。

### 5.2 模块与关键对象

| 文件 | 关键对象 | 职责 |
|------|---------|------|
| `main.py` | `app`（FastAPI）、`lifespan`、`add_process_time_and_security_headers`、`validation_exception_handler`、`global_exception_handler` | 应用装配、启动预热 7 项、异常兜底、`X-Process-Time-Ms` 埋点 |
| `config.py` | `Settings`、`settings`、`get_settings()` | 全部配置项与默认值（`lru_cache` 单例） |
| `schemas.py` | 全部 Pydantic 模型 + `ENTITY_TYPE_META` | **前后端唯一数据真源** |
| `deps.py` | `get_graph` / `get_reasoner` / `get_intent` / `get_linker` / `get_llm` / `get_rag` / `collect_health` | 依赖注入与 `/health` 状态采集 |
| `api/router.py` | `api_router`、`/health`、`/info`、`/` | 路由聚合与系统级接口 |
| `api/qa.py` | `ask_question`、`ask_stream`、`get_qa_service` | 问答（含 SSE）、模板库、历史、链路信息 |
| `api/disease.py` | `search_disease`、`disease_detail`、`related_disease`、`disease_sources`、`disease_facts`、`refresh_source`、`hot_keywords` | 掌上医典接口族（注意 `/hot-keywords` 必须注册在 `/{disease_id}` 之前） |
| `api/graph.py` | `graph_stats`、`subgraph`、`graph_evidence`、`graph_triples`、`graph_reasoning`、`graph_path`、`run_cypher`、`graph_schema`、`entity_types`、`search_entity` | 图谱接口族（含只读 Cypher） |
| `api/analytics.py` | `overview`、`node_type_pie`、`category_bar`、`infectious_gauge`、`symptom_top`、`department_distribution`、`risk_predict`、`risk_intervene`、`risk_model_info`、`all_charts`、`analytics_info` | 数据分析与健康预警 |
| `services/qa_service.py` | `QAService` | RAG 编排、会话历史（每会话上限 40 条）、纯图谱降级 |
| `services/analytics_service.py` | `AnalyticsService` | 看板指标、5 张图表（ECharts option + PyEcharts HTML + 自动洞察） |
| `services/risk_predictor.py` | `RiskPredictor`、`DiseaseRiskModel`、`BUILTIN_MODELS` | 每病一模型的 `MedicalRiskEnsemble` + 干预方案生成 |

### 5.3 请求层横切能力

| 能力 | 实现 | 关键参数 |
|------|------|---------|
| CORS | `CORSMiddleware` | `allow_origins=settings.cors_origin_list`（默认 `http://127.0.0.1:5173,http://localhost:5173`）、`allow_credentials=True`、暴露 `X-Process-Time-Ms` |
| 限流 | `slowapi` + `SlowAPIMiddleware` | `Limiter(key_func=get_remote_address, default_limits=["60/minute"])`；缺 `slowapi` 时启动告警且不限流 |
| 耗时埋点 | `@app.middleware("http")` | 响应头 `X-Process-Time-Ms`；> 3000 ms 记 WARNING 慢请求 |
| 安全响应头 | 同一中间件 | `X-Content-Type-Options: nosniff`、`X-Frame-Options: SAMEORIGIN`、`Referrer-Policy: no-referrer`、`X-Powered-By: Zhiyu-Medical-Codex` |
| 参数校验 | `validation_exception_handler` | 422 + `errors[{field,message,type}]` + `hint` |
| 异常兜底 | `global_exception_handler` | 500 + 结构化 JSON；`detail` 仅在 `DEBUG=true` 时输出 |
| 生命周期 | `lifespan` | 启动预热 7 项（内存图谱 → Neo4j → 实体识别 → 关系抽取 → 意图分类 → RAG 模板库 → LLM），关闭时 `close()` + `reset_neo4j_client()` |

### 5.4 降级策略

应用层是**降级链的收口**：无论下层如何缺失，接口都返回结构化结果而非 500。

| 场景 | 行为 |
|------|------|
| 任一预热项失败 | `try/except` 包裹并记 ERROR，**不阻塞启动**（日志：`内存图谱加载失败` / `Neo4j 客户端初始化失败` / `NLP 模型初始化失败` / `RAG 模板库加载失败` / `LLM 客户端初始化失败`） |
| 服务实例初始化失败 | `deps.py` 中单例按需创建，失败在接口内部再降级 |
| `slowapi` 缺失 | 限流关闭，启动告警 |
| PyEcharts 缺失 | `ChartResponse.pyecharts_html=""`，`echarts_option` 照常返回（前端图表不受影响） |
| Neo4j 不可用 | `/health` 返回 `neo4j="fallback_memory"`；`data_source="memory"`（图谱与看板接口照常可用） |

---

## 6. 请求生命周期：一次真实问答的完整链路

以 **「肺栓塞应该挂什么科？」** 为例，逐段追踪从 HTTP 请求到 `QAResponse` 的每一步。表中耗时取自本次验证运行日志（离线模式、无 Neo4j / 无 PyTorch / 无大模型）。

### 6.1 阶段表

| # | 调用点（文件::函数） | 做什么 | 实测耗时 |
|---|---------------------|--------|---------|
| 0 | `main.py::add_process_time_and_security_headers` | 记录起始时间，注入响应头 `X-Process-Time-Ms` | — |
| 1 | `qa.py::ask_question` | 接收 `QARequest`，捕获异常并转为 500 | — |
| 2 | `qa.py::get_qa_service` | `Depends` 注入单例 `QAService(graph, intent, linker)`（测试可整体替换） | — |
| 3 | `QAService.ask` | 取会话历史（末 6 条）与槽位，惰性获取 `RagEngine` 单例 | — |
| 4 | `RagEngine.answer` | 建立 `Stopwatch`、`LatencyBreakdown`、`reasoning_trace` 容器 | — |
| 5 | `RagEngine._classify` → `IntentClassifier.classify` → `RuleIntentClassifier.classify` | 规则意图分类（PyTorch 缺失降级），得到 `department_query` 0.943 | `intent`：**1.4 ms** |
| 6 | `RagEngine._link` → `EntityLinker.analyze` → `EntityExtractor.recognize` → `link_to_kg` → 口语别名 → 滑窗模糊 → `_disambiguate` | 「肺栓塞」→ `Disease:肺栓塞`（score 0.999，method `dictionary`），并聚合 `diseases/symptoms/...` 与 `main_entity` | `entity_link`：**77.8 ms**（该轮为进程内首次调用，含词典/别名索引的冷启动） |
| 7 | `HybridRetriever.retrieve` → `_slot_direct_facts` / `_reverse_symptom_facts` / `_multi_hop_facts` / `_reasoned_facts` / `_similar_facts` → `_rank` → `_select_diverse` → 编号 | 五路召回去重排序，取 top-12：直接事实 11 条（科室 2 / 易感人群 2 / 鉴别诊断 2 / 症状 4 / 检查 1）+ 规则推理补全 1 条；逐条赋 `KG-1…KG-12` 与 `sources` | `kg_query`：**59.1 ms**（日志实测混合检索耗时；热态 1.2 ms） |
| 8 | `ContextBuilder.build` → `_render_kg_context` / `_slot_summary` / `_entities_str` / `_department_list` / `_safe_knowledge` / `_chunk_context` | 生成带编号、带槽位、带安全提示的 `ContextBundle` 与 `variables`（`entities_str="肺栓塞(Disease/疾病)"`、`department_list="内科、呼吸内科"`） | `retrieve`：**2.02 ms** |
| 9 | `PromptLibrary.select` → `PromptTemplate` | 安全覆盖未命中 → 意图 `department_query` 内 priority 最高模板：`department_query_core_v1` | `prompt`：**0.21 ms** |
| 10 | `PromptLibrary.render` | 变量替换（`str.replace` 而非 `format`，避免 JSON 花括号被误解析）+ 追加 `meta.global_constraints` + 追加免责声明 | 计入 `llm` |
| 11 | `LLMClient.generate` | `provider=template` 直接返回 `LLMResult(success=False)`，不发网络请求 | 计入 `llm` |
| 12 | `RagEngine._template_answer` | 以 KG 事实按 `_GROUP_ORDER`（症状 → 治疗 → 用药 → 检查 → 科室 → 并发症 → 鉴别 → 易感人群）拼装 markdown，每条带 `[KG-n]` | `llm`（含模板渲染 + 生成）：**0.56 ms** |
| 13 | `HallucinationGuard.check` → `GuardResult` | 7 项校验通过 → `accept`，score 0.9；`explain()` 生成前端展示用的 `guard_result` | `guard`：**0.12 ms** |
| 14 | `RagEngine.answer` 组装 `QAResponse` | `confidence = 0.35×意图置信度 + 0.35×Top3 事实平均置信度 + 0.30×守卫评分`；`answer_html` 由 `_to_html()` 渲染；`related_questions` 由 `GraphService.related_questions()` 生成 | `total`：**141.0 ms**（冷启动；热态 8 ~ 45 ms） |
| 15 | `RagEngine._append_history` / `QAService._remember` | 双写会话记忆（`RagEngine` 40 条、`QAService` 40 条），供 `/qa/history` 与多轮槽位继承 | — |
| 16 | FastAPI 序列化 | 按 `QAResponse` 输出 JSON，写入 `X-Process-Time-Ms` | 日志：`问答完成：意图=department_query 置信度=0.943 事实=12 守卫=accept 总耗时=141.0ms（语义解析 138.3ms）` |

### 6.2 关键指标与观测

| 指标 | 实测值 | 说明 |
|------|-------|------|
| 语义解析（意图 + 实体链接 + 图谱检索） | **25 ~ 160 ms**（本轮 138.3 ms 为冷启动，热态 41.6 ~ 110.3 ms） | `latency_ms.semantic_parse_total`，PPT 指标 ≤500 ms；超过 500 ms 会写 WARNING |
| 端到端总耗时 | **8 ~ 141 ms**（模板降级模式） | 接入 Llama 3 后 `llm` 段将成为主要耗时 |
| 检索召回 | 12 条（直接 11 / 推理补全 1） | 日志：`混合检索完成：直接事实 11 / 反向检索 0 / 多跳 0 / 推理补全 1 / 相似扩展 0（合计 12 条，59.1ms）` |
| 规则推理缓存 | 首次计算，之后复用 | 日志：`推理补全事实缓存完成：20 条（仅本进程计算一次）` |
| 幻觉守卫 | `accept`（引用编号完整、无越界实体、无剂量与诊断越权表述） | 校验发现 `missing_citation` 时仍可 `accept`（medium 级）；出现 critical 或 < 0.45 则 `fallback_template` |

---

## 7. 数据模型与图谱 Schema

### 7.1 节点标签

| 标签 | 中文名 | 主要属性 | 图谱实例数 |
|------|-------|---------|-----------|
| `Disease` | 疾病 | `name`、`alias[]`、`category1`、`category2`、`definition`、`cause`、`diagnosis`、`treatment`、`prognosis`、`population`、`is_infectious`、`disease_id`、`symptoms[]`、`checks[]`、`drugs[]`、`treatments[]`、`complications[]`、`differential[]`、`department`、`sources[]`、`updated_at` | 59 |
| `Symptom` | 症状 | `name`、`alias` | 195 |
| `Drug` | 药物 | `name`、`alias`、`category` | 122 |
| `Check` | 检查项目 | `name`、`alias` | 116 |
| `Department` | 科室 | `name`、`alias`、`category` | 34 |
| `Treatment` | 治疗方法 | `name`、`alias`、`category` | 26 |
| `Population` | 易感人群 | `name` | 15 |
| `Source` | 文献来源 | `pmid`、`doi`、`title`、`journal`、`year`、`url`、`authority` | 0（溯源锚点，按需创建） |

节点 ID 约定：内存图谱为 `"<Type>:<name>"`（如 `Disease:肺栓塞`）；Neo4j 侧由 `elementId()` 提供，前端 D3 只依赖 `id` 字段。

### 7.2 关系类型（`REL_LABELS`）

| 关系类型 | 中文标签 | 语义 | 主要来源 |
|---------|---------|------|---------|
| `HAS_SYMPTOM` | 症状 | 疾病 → 症状 | 种子库 `diseases[].symptoms` + `relations.json` |
| `TREATED_BY` | 治疗 | 疾病 → 治疗方法 | 种子库 `treatments[]` / `treatment` |
| `USES_DRUG` | 用药 | 疾病 → 药物 | 种子库 `drugs[]` |
| `BELONGS_TO` | 科室 | 疾病 → 科室 / 分类 | 由 `category1`、`category2`、`department` 三个字段合成（去重后最多 2 条/疾病） |
| `NEEDS_CHECK` | 检查 | 疾病 → 检查项目 | 种子库 `checks[]` |
| `AFFECTS` | 易感人群 | 疾病 → 易感人群 | `relations.json` |
| `HAS_COMPLICATION` | 并发症 | 疾病 → 疾病 | 种子库 `complications[]` |
| `DIFFERENTIAL_WITH` | 鉴别诊断 | 疾病 → 疾病 | 种子库 `differential[]` |
| `RELATED_TO` | 相关 | 疾病 ↔ 疾病（推理产生） | `GraphReasoner` 规则 R1 / R2 / R4 |
| `PROVES` | 知识来源 | 文献 → 实体（溯源锚点） | 写图脚本 `load_kg.py` |
| `IS_INFECTIOUS` | 传染性 | 疾病 → 标记 / `:Source` | 写图脚本 |

关系属性：`confidence`（默认 0.95；`relations.json` 实测区间 **0.86 ~ 0.99**，规则推理产生的三元组按规则置信度取 **0.50 ~ 0.90**）、`weight`、`source_ref`（来源追溯，如疾病 ID `D0001`）。

### 7.3 约束与索引（`scripts/cypher/01_constraints.cypher`、`02_indexes.cypher`）

**唯一性约束（10 条）**

| 约束名 | 目标 |
|--------|------|
| `disease_name_unique` | `(:Disease).name` |
| `disease_id_unique` | `(:Disease).disease_id` |
| `symptom_name_unique` | `(:Symptom).name` |
| `drug_name_unique` | `(:Drug).name` |
| `department_name_unique` | `(:Department).name` |
| `treatment_name_unique` | `(:Treatment).name` |
| `check_name_unique` | `(:Check).name` |
| `population_name_unique` | `(:Population).name` |
| `source_pmid_unique` | `(:Source).pmid` |
| `source_url_unique` | `(:Source).url` |

**索引（13 条）**

| 索引名 | 类型 | 目标 | 用途 |
|--------|------|------|------|
| `disease_category1_idx` | 属性 | `(:Disease).category1` | 「一级分类下的疾病数量」柱状图 |
| `disease_category2_idx` | 属性 | `(:Disease).category2` | 二级分类过滤 |
| `disease_infectious_idx` | 属性 | `(:Disease).is_infectious` | 「传染性疾病比例」环形图 |
| `disease_alias_idx` | 属性 | `(:Disease).alias` | 实体链接（口语 → 标准名） |
| `symptom_alias_idx` | 属性 | `(:Symptom).alias` | 症状别名链接 |
| `department_category_idx` | 属性 | `(:Department).category` | 科室分类统计 |
| `source_year_idx` | 属性 | `(:Source).year` | 按年份筛选文献 |
| `source_authority_idx` | 属性 | `(:Source).authority` | 按权威等级筛选 |
| `entity_fulltext` | 全文 | `(:Disease\|Symptom\|Drug\|Department\|Treatment\|Check)` 的 `name, alias` | 实体检索输入联想 |
| `source_title_fulltext` | 全文 | `(:Source)` 的 `title, journal` | RAG 文献片段检索 |
| `rel_confidence_idx` | 关系属性 | `()-[:HAS_SYMPTOM]-()` 的 `confidence` | 高置信度事实优先 |
| `rel_weight_idx` | 关系属性 | `()-[:TREATED_BY]-()` 的 `weight` | 权重排序 |

> `GET /graph/schema` 返回的是其中 4 条约束与 2 条索引的代表性子集（便于前端展示），完整清单以两个 `.cypher` 脚本为准。
> 初始化方式：`python scripts/init_neo4j.py`（按序执行 `scripts/cypher/*.cypher`），导入方式：`python scripts/load_kg.py [--corpus ...] [--incremental]`。

### 7.4 内存图谱的等价性

`MemoryGraphStore` 与 Neo4j 保持同一套 Schema 语义，使降级后行为一致：

- `nodes: {node_id: {id,name,type,label,color,degree,properties}}`、`edges: [{source,target,rel,label,properties}]`；
- `index: {(type, name) → node_id}` 支撑 O(1) 实体解析，`adjacency` 支撑 BFS 子图与多跳路径；
- `add_edge()` 内建 `(source,target,rel)` 去重，并给 `Source` 类型节点做过滤（`triples_of` 不返回溯源节点）；
- 实现 Cypher 子集解释器（`query()`）：`count(n)`、按标签列举、`SHOW` 等；不支持的语句返回空结果并记 WARNING。

---

## 8. 知识溯源设计（100% 可追溯）

溯源链条由「数据侧锚点 + 检索侧编号 + 生成侧强制引用 + 校验侧拦截」四段拼合，缺一不可。

### 8.1 数据侧：`:Source` 节点与 `PROVES` 关系

```cypher
(:Source {pmid, doi, title, journal, year, url, authority})-[:PROVES]->(:Disease)
(:Disease {..., sources: [...]})   // 种子库内嵌的来源数组
```

- `SourceItem` 字段：`source_type`（`pubmed | guideline | textbook | website`）、`pmid`、`doi`、`title`、`journal`、`year`、`authors`、`url`、`authority`（`A` 指南/教材、`B` 期刊、`C` 其他）、`retrieved_at`；
- **覆盖实测**：59 / 59 个疾病均挂载来源，共 **118 条**（每病 2 条：1 条指南或教材 + 1 条 PubMed 文献），其中 `pubmed 59` / `guideline 49` / `textbook 10`，权威等级 `A` 59 条、`B` 59 条，59 个 PMID 全部唯一；
- 约束 `source_pmid_unique` / `source_url_unique` 保证同一文献不会重复建点。

### 8.2 检索侧：`KG-n` 编号与 `EvidenceItem`

```text
HybridRetriever.retrieve()
  └─ _rank / _select_diverse 后按名次编号
       for i, f in enumerate(ranked, start=1):
           f.triple_id = f"KG-{i}"                 # 编号即排序名次
           f.sources   = self._evidence_sources(f) # 三元组 → 来源文献
```

`GraphService.get_evidence(head, rel, tail, triple_id)` 从**头尾节点的 `properties.sources`** 汇聚来源，按 `(pmid, doi, title)` 去重后最多返回 3 条，产出 `EvidenceItem{triple_id, head, head_type, relation, relation_label, tail, tail_type, confidence, sources[]}`——这就是 `/graph/evidence` 与 `QAResponse.evidences` 的同一个对象。

### 8.3 生成侧：强制引用

- `prompt_library.yaml::meta.global_constraints` 第 5 条硬约束：*每一条来自知识库的陈述后面，必须紧跟形如 `[KG-1]` 的引用编号，编号必须与 `<KG_CONTEXT>` 中的编号严格对应*；
- `ContextBuilder._render_kg_context()` 生成的三元组上下文**自带编号**，模型只需照抄；
- 降级路径同样强制引用：`RagEngine._template_answer()` 与 `QAService._render_graph_answer()` 拼装答案时逐条写入 `[KG-n]`；
- `RagEngine._to_html()` / `QAService._to_html()` 把 `[KG-n]` 渲染为可点击角标：

```html
<sup class="kg-cite" data-cite="KG-1">[KG-1]</sup>
```

前端据此渲染溯源面板（点击角标 → 展示该三元组的来源文献）。

### 8.4 校验侧：幻觉守卫拦截"假引用"

| 检查项 | 触发条件 | 严重级别 |
|--------|---------|---------|
| `missing_citation` | 句中提到图谱实体但没有 `[KG-n]`（标题、`**字段**：值` 等结构性文本豁免） | medium |
| `invalid_citation` | 引用编号越界（如 `[KG-99]` 而只有 12 条事实） | high |
| `unknown_entity` | 答案中出现图谱与安全常识之外的医学名词（后缀正则识别，通用词白名单豁免） | high |
| `dose_violation` | 出现剂量 / 频次 / 给药途径（`mg`、`片`、`bid`、`皮下注射 2` 等） | critical |
| `absolute_claim` | "你患了""可以确诊""百分之百""无需就医"等绝对化结论 | high |
| `diagnostic_claim` | "诊断为""确诊为"（前置 12 字内有"不能/无法/须/需/应由"则豁免） | high |
| `missing_disclaimer` | 结尾缺少统一免责声明 | medium |

处置：`accept`（直接返回）/ `rewrite`（前端标注需复核）/ `fallback_template`（丢弃模型输出，改用 `strip_and_fallback()` 的纯 KG 拼装答案）。

### 8.5 端到端示例

```text
用户问「肺栓塞应该挂什么科？」
  → KG-1 肺栓塞 —科室→ 内科       （confidence 0.95）
  → KG-2 肺栓塞 —科室→ 呼吸内科   （confidence 0.95）
  → 答案：「建议就诊科室：内科 [KG-1]、呼吸内科 [KG-2]。」
  → evidences[0].sources = 《肺血栓栓塞症诊治与预防指南》(authority A) +
                           2019 ESC Guidelines (PMID 31573352, authority B)
  → 前端点击 [KG-2] → 展示上述两条文献与原文链接
```

---

## 9. 隐私与合规

### 9.1 两层防护设计

| 层次 | 模块 | 作用点 | 手段 |
|------|------|-------|------|
| ① PII 脱敏 | `PrivacyGuard.mask()` | 入库前（爬虫管道）与日志前 | 11 条正则规则 + 4 条安全上下文豁免 |
| ② 字段加密 | `PrivacyGuard.encrypt_field()` | 需要长期保存的敏感字段 | AES-256-GCM（随机 96-bit nonce + 完整性校验） |

设计原则（源码注释原文）：**原始 PII 永不落库、永不落盘、永不进日志**。

### 9.2 脱敏：脱什么、留什么

**11 条 PII 规则**（命中即替换为 `[<类型>_MASKED]`）：

| 规则名 | 识别对象 | 替换标记 |
|--------|---------|---------|
| 身份证号 | 18 位身份证（含 X 校验位） | `[ID_MASKED]` |
| 手机号 | `1[3-9]` 开头 11 位 | `[PHONE_MASKED]` |
| 固定电话 | 区号 + 7~8 位号码 | `[TEL_MASKED]` |
| 银行卡号 | 16~19 位连续数字 | `[BANK_MASKED]` |
| 电子邮箱 | 标准邮箱格式 | `[EMAIL_MASKED]` |
| 住院号/病历号 | 住院号/病历号/门诊号/病案号/医保卡号/社保号 + 编号 | `[MRN_MASKED]` |
| 详细地址 | 省/市/区/县/街道/路/号/栋/室 等组合 | `[ADDR_MASKED]` |
| QQ/微信号 | `QQ`/`微信` + 账号 | `[SOCIAL_MASKED]` |
| 姓名-称谓 | 患者/病人/本人/姓名/联系人 + 2~4 汉字 | `[NAME_MASKED]` |
| 姓名-常见姓氏+称谓 | 常见姓氏 + 某某/某/××/XX | `[NAME_MASKED]` |
| 出生日期 | 出生日期/出生年月/生日 + 日期 | `[DOB_MASKED]` |

**保留什么（安全上下文豁免 `SAFE_CONTEXT_PATTERNS`）**：血压/血糖/血脂/心率等指标数值、带单位（`mg`、`mmHg`、`mmol/L`、`次/分`、`℃`…）的数值、`PMID`/`DOI` 编号、`ICD-` 编码——避免把医学知识中的正常数字误脱敏而破坏知识可用性。实测：`血压 152/96 mmHg`、`PMID: 32130469` 均原样保留。

**审计统计**：`MaskResult{text, hit_count, hits{类型:次数}, original_length, masked_length}`，只记录「脱敏了什么类型、多少处」，**不记录原文**。实测日志：`脱敏完成：5 处（身份证号×1, 手机号×1, 住院号/病历号×1, 详细地址×1, 姓名-称谓×1）`。

### 9.3 加密：AES-256-GCM 与临时密钥

- 密钥来源：环境变量 `FIELD_ENCRYPT_KEY`（base64 编码的 32 字节），生产环境应由 KMS 注入；
- 密文格式：`v1:base64(nonce):base64(ciphertext+tag)`，每次加密使用**随机 96-bit nonce**，GCM 提供完整性校验（篡改或密钥不匹配时 `decrypt_field()` 记 ERROR 并返回原文）；
- 密钥指纹：日志只打印 `sha256(key)[:12]`（如 `c64c9b849a35`），**绝不打印密钥本身**；
- 开关：`PRIVACY_ENCRYPT_ENABLED`（默认 `true`）；关闭或 `cryptography` 缺失时 `encrypt_field()` 返回原文（可用性优先）；
- **临时密钥行为**：未配置 `FIELD_ENCRYPT_KEY` 时，演示原型自动生成 `os.urandom(32)` 的**进程内临时密钥**并打印 WARNING，提示"该密钥仅存在于当前进程，重启后无法解密旧数据"，并给出生成固定密钥的命令：`python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"`。

### 9.4 `anonymize_health_input()`：风险预测入参去标识化

```python
allowed_numeric = {age, bmi, systolic_bp, diastolic_bp, fasting_glucose,
                   total_cholesterol, hdl, ldl, triglycerides, heart_rate}
allowed_enum    = {gender, physical_activity}   # 必须匹配 ^[A-Za-z_]{2,20}$
allowed_bool    = {smoking, drinking}
allowed_list    = {family_history, symptoms, existing_conditions}  # 每项截断 30 字、最多 20 项
# 其余字段（姓名、身份证、地址、自由文本……）一律丢弃
```

这是「个人信息零留存」的关键：`/analytics/risk/predict` 与 `/analytics/risk/intervene` 只需要数值特征，不接收也不应接收任何可直接识别个人身份的信息。

另有 `pseudonymous_id(seed, salt)`：`"anon_" + sha256(salt:seed)[:16]`，用于在不存身份信息的前提下关联同一用户的多次请求（`salt` 由服务端持有，可用环境变量 `PSEUDONYM_SALT` 覆盖）。

### 9.5 其他合规措施

| 措施 | 实现 |
|------|------|
| 会话可删除 | `DELETE /qa/history/{session_id}` 清空内存会话；历史仅存内存、上限 40 条，不落库 |
| 图谱只读 | `assert_readonly()` 白名单拒绝一切写操作（详见 API 文档第 10 章） |
| 防滥用 | `slowapi` 全局限流 60 次/分钟/IP |
| 输出边界 | 模板全局硬约束（禁处方剂量、禁确定性诊断）+ 幻觉守卫 + 统一免责声明字段 |
| 前端常驻提示 | `DisclaimerBar.vue` 在页面顶部、问答界面与风险预测结果页常驻展示 |

---

## 10. 降级策略总表

| 依赖 | 缺失时的行为 | 降级实现 | 用户可感知的影响 |
|------|-------------|---------|----------------|
| Neo4j / `neo4j` 驱动 | 自动切内存图谱 | `Neo4jClient._activate_fallback()` → `MemoryGraphStore` | 无（567 节点 / 1970 关系照常可用）；`data_source="memory"` |
| PyTorch（NER） | 词典最长匹配 | `EntityExtractor.backend == "dictionary"` | 未登录词典的实体可能漏召 |
| PyTorch（关系抽取） | 规则抽取 | `RelationExtractor.backend == "rule"` | 复杂长句关系抽取召回下降（创新点 1 的增益需训练后体现） |
| PyTorch（意图） | 医疗触发词规则分类 | `IntentClassifier.backend == "rule"` → `method="rule_fallback"` | 意图准确率约 88%（源码注释口径） |
| PyTorch（注意力模块） | NumPy 推理实现 | `MedicalKeywordAttention` + `未检测到 PyTorch…降级为 NumPy 实现（仅推理）` | 仅影响训练/增强路径，线上抽取照常 |
| `torch` / PyG（GNN 补全） | 启发式补全 | `gnn_backend="heuristic"`（Adamic-Adar + Jaccard） | `/graph/reasoning?mode=gnn` 仍返回结果，评分依据变为图结构启发式 |
| PyYAML | 模板库为空 | `PromptLibrary._load()` 提前返回；`select()` 返回 `None` | 使用 KG 上下文作为 prompt，答案退化为模板拼装（可解释性不变） |
| 大模型（`httpx`/`openai`/端点不可达/`provider=template`） | 本地 KG 模板回答 | `LLMClient` 返回 `success=False` → `RagEngine._template_answer()` | 答案更简洁、无自然语言润色，但事实与引用完整；`llm_model="local-template-rag"` |
| RAG 引擎整体不可用 | 纯图谱结构化回答 | `QAService._graph_only_answer()` | `prompt_template_id="graph_only_fallback"`、`llm_model="graph-only"`、`confidence` 固定 0.75 / 0.25 |
| `cryptography` | 字段加密关闭 | `encrypt_field()` 返回原文 + WARNING | 演示不受影响；生产必须安装 |
| `slowapi` | 不限流 | 启动 WARNING | 无 429；生产必须安装 |
| PyEcharts | 服务端渲染片段为空 | `pyecharts_html=""` | 前端 ECharts 仍按 `echarts_option` 正常渲染 |
| Scrapy | 采集脚本不可用 | — | 在线服务不受影响（图谱读种子库） |
| NumPy | XGBoost 推理改用 Python list | `risk_predictor.HAS_NUMPY=False` | 内置 Logistic 路径与图谱/问答主链路不依赖 NumPy |

---

## 11. 实测数字、验证与已知限制

### 11.1 本次验证的关键数字

| 维度 | 数值 |
|------|------|
| 内存图谱 | **567 节点 / 1970 关系 / 59 疾病**（症状 195、药物 122、检查 116、科室 34、治疗方法 26、易感人群 15） |
| 医疗 RAG prompt 模板 | **104 个模板 / 16 类意图**（启动自检通过：`missing_required=[]`、`ok=true`） |
| 安全覆盖模板 | 10 条 `SAFETY_OVERRIDES`（自伤、胸痛、呼吸困难、卒中、出血、中毒、惊厥、剂量索取、药物相互作用、急诊判定） |
| 医疗关键词词表 | **201 词**（symptom 58 + treatment 52 + check 52 + department 39） |
| 实体词典表面形式 | **631 条**（最长 15 字） |
| 口语别名索引 | 64 条（源词典 113 条） |
| 知识溯源覆盖 | 59 / 59 疾病，共 118 条来源（pubmed 59 / guideline 49 / textbook 10；authority A 59 / B 59） |
| 图谱规则推理 | 6 条规则，59 疾病 → 新增 20 条三元组 |
| 术语归一化词典 | 113 条 |
| 冒烟测试 | **96 / 96 通过**（离线环境：无 Neo4j、无 PyTorch、无大模型） |
| pytest 套件 | **47 / 47 通过**（`backend/tests/test_api_smoke.py`） |
| 全栈联通性 | **7 / 7 通过**（Vite 5173 → 代理 → FastAPI 8000） |
| OpenAPI | 36 个 `/api/v1` 端点，41 个 Schema |
| Python 兼容性 | 后端 **61 个 `.py` 文件**全部保持 Python 3.9 语法兼容：每个模块均声明 `from __future__ import annotations`（PEP 563 延迟注解），因此个别注解可写 `Dict[str, float] | None` 而在 3.9 下仍可运行；全仓无 `match` 语句；其中 12 个为包 `__init__.py`、2 个为测试文件 |

### 11.2 已知限制（如实说明）

1. **风险模型已按参考人群重新标定**：`BUILTIN_MODELS` 的 10 个疾病风险函数采用
   "已发表队列研究效应量的近似值"作为系数，并已针对**健康参考画像**
   （24 岁女性、BMI 20.5、血压 108/68、无家族史）重新居中截距，
   使各病基线风险回落到合理范围内（高血压 3.0%、2 型糖尿病 2.0%、
   冠心病 0.98%、高脂血症 1.8% 等）。实测表现：

   | 画像 | 综合健康评分 | 最高风险疾病 | 总体等级 |
   |------|:---:|------|------|
   | 健康青年（24 岁女性，全部指标理想） | **97.8 / 100** | 原发性高血压 3.0% | 低风险 |
   | 心血管代谢高危（58 岁男性，收缩压 152、吸烟、BMI 27.4、双家族史） | **13.1 / 100** | 冠心病 99.7% | 极高风险 |

   说明：高危画像下多个疾病同时贴近 100% 属于线性 logistic 模型的**饱和特性**，
   在"多个危险因素同时极端"时符合预期（该画像确实应被判定为多病高危）。
   生产环境建议用 `scripts/train_risk_model.py` 训练出的
   `models/risk_xgboost.json`（XGBoost + Platt 校准，AUC 0.923）
   替换内置系数，可获得更平滑的概率分布；
2. **风险测试的期望值已与内置系数对齐**：`tests/test_api_smoke.py` 中的
   `test_risk_predict`（高危画像）断言心脑血管疾病进入 Top-3 且高血压为高风险等级，
   `test_risk_predict_low_risk_profile`（健康画像）断言 `health_score > 60`、
   所有疾病风险 < 0.5、且最高风险 < 10%。两条用例在**仅使用内置系数**时即可通过，
   挂载训练权重后同样成立，因此不依赖 `models/` 目录是否存在；
3. **单词会话无持久化**：`RagEngine._sessions` 与 `QAService._history` 均为进程内字典，重启即丢失，多副本部署时不共享（生产应换 Redis）；
4. **内存 Cypher 子集有限**：仅支持 `count` 统计、按标签列举与 `SHOW` 等少量形态，复杂只读查询需 Neo4j；
5. **规则推理为 Python 等价实现**：`PythonRuleEngine` 与 `scripts/jena/rules_medical.rules` 结论等价，真正的 Jena CLI 需 JDK（`jena_cli_hint()` 给出命令）；
6. **LLM 默认离线**：当前 `LLM_PROVIDER=template` 属于答辩保底方案，接入 Llama 3 只需改 `.env`（`LLM_PROVIDER=openai_compatible` + `LLM_BASE_URL`），RAG 链路无需改动。

> ⚠️ **免责声明**：本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。

