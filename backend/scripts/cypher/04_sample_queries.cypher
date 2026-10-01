// =============================================================================
//  Neo4j 常用查询集（04_sample_queries.cypher）
//  用途：Neo4j Browser 中直接演示；或作为 API 层 Cypher 模板的参考
// =============================================================================

// =============================================================================
//  一、知识图谱可视化（PPT 功能一）
// =============================================================================

// 1.1 以「高血压」为中心展开 1 跳子图（前端 D3 力导向图数据源）
MATCH (center:Disease)
WHERE center.name CONTAINS '高血压'
WITH center LIMIT 1
MATCH path = (center)-[*1..1]-(n)
WHERE NOT n:Source
RETURN path LIMIT 200;

// 1.2 展开 2 跳（深度探索）
MATCH (center:Disease {name: '原发性高血压'})
MATCH path = (center)-[*1..2]-(n)
WHERE NOT n:Source
RETURN path LIMIT 300;

// 1.3 图谱统计（页面顶部三个统计卡片）
MATCH (n) WHERE NOT n:Source
RETURN labels(n)[0] AS 实体类型, count(n) AS 节点数量
ORDER BY 节点数量 DESC;

MATCH ()-[r]->()
RETURN count(r) AS 关系总数;

MATCH (n) WHERE NOT n:Source
RETURN count(DISTINCT labels(n)[0]) AS 实体类型数, count(n) AS 节点总数;

// 1.4 环形图：各类节点占比（数据分析页面）
MATCH (n) WHERE NOT n:Source
WITH labels(n)[0] AS 类型, count(n) AS 数量
WITH collect({name: 类型, value: 数量}) AS data, sum(数量) AS total
UNWIND data AS d
RETURN d.name AS 类别, d.value AS 数量,
       round(toFloat(d.value) / total * 10000) / 100 AS 占比百分比
ORDER BY 数量 DESC;

// =============================================================================
//  二、疾病查询（PPT 功能二 · 掌上医典）
// =============================================================================

// 2.1 疾病检索（卡片式返回：分类 / 症状 / 治疗 / 易感人群）
MATCH (d:Disease)
WHERE d.name CONTAINS '高血压'
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
OPTIONAL MATCH (d)-[:BELONGS_TO]->(dept:Department)
OPTIONAL MATCH (d)-[:AFFECTS]->(p:Population)
RETURN d.name AS 疾病名称,
       d.category1 AS 一级分类,
       d.category2 AS 二级分类,
       collect(DISTINCT s.name)[0..6] AS 症状,
       collect(DISTINCT t.name) AS 治疗,
       collect(DISTINCT dept.name) AS 科室,
       collect(DISTINCT p.name) AS 易感人群
ORDER BY d.name
LIMIT 20;

// 2.2 疾病详情（定义 / 病因 / 症状 / 诊断 / 治疗 / 预后）
MATCH (d:Disease {name: '原发性高血压'})
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)-[:USES_DRUG]->(dr:Drug)
OPTIONAL MATCH (d)-[:NEEDS_CHECK]->(c:Check)
OPTIONAL MATCH (d)-[:HAS_COMPLICATION]->(cp:Disease)
OPTIONAL MATCH (d)-[:DIFFERENTIAL_WITH]->(df:Disease)
OPTIONAL MATCH (d)-[:PROVES]-(src:Source)
RETURN d.name AS 疾病, d.definition AS 定义, d.cause AS 病因,
       d.diagnosis AS 诊断, d.treatment AS 治疗方案, d.prognosis AS 预后,
       collect(DISTINCT s.name) AS 症状,
       collect(DISTINCT dr.name) AS 常用药物,
       collect(DISTINCT c.name) AS 检查项目,
       collect(DISTINCT cp.name) AS 并发症,
       collect(DISTINCT df.name) AS 鉴别诊断,
       collect(DISTINCT {title: src.title, pmid: src.pmid, url: src.url}) AS 来源;

// =============================================================================
//  三、智能问答检索（PPT 功能三 · 可解释 AI 医生）
// =============================================================================

// 3.1 科室导诊（"肺栓塞应该挂什么科？"）
MATCH (d:Disease {name: '肺栓塞'})-[:BELONGS_TO]->(dept:Department)
OPTIONAL MATCH (d)-[:NEEDS_CHECK]->(c:Check)
RETURN d.name AS 疾病,
       collect(DISTINCT dept.name) AS 建议科室,
       collect(DISTINCT c.name) AS 建议检查;

// 3.2 症状反查疾病（"胸痛伴低烧可能是什么？"）
MATCH (s:Symptom)<-[:HAS_SYMPTOM]-(d:Disease)
WHERE s.name IN ['胸痛', '发热']
WITH d, collect(DISTINCT s.name) AS 命中症状, count(DISTINCT s) AS 命中数
OPTIONAL MATCH (d)-[:BELONGS_TO]->(dept:Department)
RETURN d.name AS 可能疾病,
       命中症状, 命中数,
       collect(DISTINCT dept.name) AS 科室,
       d.category1 AS 一级分类
ORDER BY 命中数 DESC, d.name
LIMIT 15;

// 3.3 多跳推理路径（可解释性：为什么得出这个结论）
MATCH p = shortestPath(
    (a:Symptom {name: '头晕'})-[*1..4]-(b:Disease {name: '原发性高血压'})
)
RETURN [n IN nodes(p) | n.name] AS 路径节点,
       [r IN relationships(p) | type(r)] AS 路径关系,
       length(p) AS 跳数;

// 3.4 RAG 上下文构建：按关系类型分槽位取事实
MATCH (d:Disease {name: '原发性高血压'})-[r]->(t)
WHERE NOT t:Source
WITH type(r) AS 关系类型, collect(t.name) AS 实体
RETURN 关系类型, 实体;

// 3.5 药物反向查询（"氨氯地平是治什么的？"）
MATCH (dr:Drug {name: '氨氯地平'})<-[:USES_DRUG]-(d:Disease)
OPTIONAL MATCH (d)-[:BELONGS_TO]->(dept:Department)
RETURN d.name AS 疾病, d.category1 AS 一级分类,
       collect(DISTINCT dept.name) AS 科室;

// =============================================================================
//  四、数据分析（PPT 功能四）
// =============================================================================

// 4.1 一级分类下的疾病数量（柱状图）
MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department {category: '一级科室'})
RETURN dept.name AS 一级分类, count(d) AS 疾病数量
ORDER BY 疾病数量 DESC;

// 4.2 传染性疾病比例（环形图）
MATCH (d:Disease)
WITH count(d) AS 总数, sum(CASE WHEN d.is_infectious THEN 1 ELSE 0 END) AS 传染病数
RETURN 总数, 传染病数, 总数 - 传染病数 AS 非传染病数,
       round(toFloat(传染病数) / 总数 * 10000) / 100 AS 传染病占比;

// 4.3 科室疾病分布
MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department {category: '二级科室'})
RETURN dept.name AS 科室, count(d) AS 疾病数量
ORDER BY 疾病数量 DESC LIMIT 10;

// 4.4 高频症状 TopN
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name AS 症状, count(d) AS 关联疾病数
ORDER BY 关联疾病数 DESC LIMIT 12;

// 4.5 风险共病分析（高血压患者还需关注哪些疾病）
MATCH (d:Disease {name: '原发性高血压'})-[:HAS_COMPLICATION]->(cp:Disease)
OPTIONAL MATCH (cp)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN cp.name AS 并发症, collect(DISTINCT s.name)[0..5] AS 相关症状
ORDER BY cp.name;

// =============================================================================
//  五、知识溯源（合规要求：来源 100% 可追溯）
// =============================================================================

// 5.1 某疾病全部知识来源
MATCH (d:Disease {name: '原发性高血压'})-[:PROVES]-(src:Source)
RETURN d.name AS 疾病, src.title AS 文献标题, src.journal AS 期刊,
       src.year AS 年份, src.pmid AS PMID, src.doi AS DOI,
       src.authority AS 权威等级, src.url AS 链接
ORDER BY src.year DESC;

// 5.2 权威等级分布（A=指南/教材，B=期刊，C=其他）
MATCH (src:Source)
RETURN src.authority AS 权威等级, count(src) AS 数量
ORDER BY 权威等级;

// 5.3 找出缺乏来源溯源的知识（数据治理用）
MATCH (d:Disease)
WHERE NOT (d)-[:PROVES]-(:Source)
RETURN d.name AS 缺少来源溯源的疾病
LIMIT 50;

// =============================================================================
//  六、图谱推理与补全（Apache Jena 规则 + GAT/GCN 对应查询）
// =============================================================================

// 6.1 规则 R1：共享 ≥2 个症状 → 疾病相关（共病/鉴别提示）
MATCH (a:Disease)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(b:Disease)
WHERE a.name < b.name
WITH a, b, collect(DISTINCT s.name) AS 共同症状
WHERE size(共同症状) >= 2
RETURN a.name AS 疾病A, b.name AS 疾病B,
       共同症状, size(共同症状) AS 共同症状数
ORDER BY 共同症状数 DESC LIMIT 30;

// 6.2 规则 R3：并发症症状传递（A 的并发症 B 有症状 S → A 可能间接引起 S）
MATCH (a:Disease)-[:HAS_COMPLICATION]->(b:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE NOT (a)-[:HAS_SYMPTOM]->(s)
RETURN a.name AS 疾病, b.name AS 经并发症, s.name AS 可能间接症状
LIMIT 30;

// 6.3 图结构：度数最高的核心实体（图谱枢纽分析）
MATCH (n) WHERE NOT n:Source
WITH n, size([(n)--() | 1]) AS 度数
RETURN labels(n)[0] AS 类型, n.name AS 实体, 度数
ORDER BY 度数 DESC LIMIT 20;

// 6.4 孤立节点检测（图谱补全的候选）
MATCH (n) WHERE NOT n:Source
WITH n, size([(n)--() | 1]) AS 度数
WHERE 度数 <= 1
RETURN labels(n)[0] AS 类型, n.name AS 实体, 度数
LIMIT 50;

// =============================================================================
//  七、安全只读校验示例（/graph/cypher 接口允许的语句形态）
// =============================================================================
// ✅ 允许：以 MATCH / WITH / UNWIND / RETURN / CALL / SHOW 开头 + 含 LIMIT
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d.name AS 疾病, collect(s.name)[0..5] AS 症状
LIMIT 10;

// ❌ 以下语句会被 /graph/cypher 的只读白名单拒绝（403）：
// CREATE (n:Test {name: 'x'})
// MATCH (n) DETACH DELETE n
// MATCH (n:Disease) SET n.name = 'hacked'
// LOAD CSV FROM 'http://evil.com/x.csv' AS line
