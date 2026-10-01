# =============================================================================
#  Neo4j 5.8 图谱初始化 —— 02 索引
#  目的：让实体链接、分类统计、全文检索达到毫秒级响应
# =============================================================================

// ------------------------- 属性索引 -------------------------
// 疾病分类（数据分析页面的"一级分类下的疾病数量"柱状图直接依赖）
CREATE INDEX disease_category1_idx IF NOT EXISTS
FOR (d:Disease) ON (d.category1);

CREATE INDEX disease_category2_idx IF NOT EXISTS
FOR (d:Disease) ON (d.category2);

// 传染性标记（"传染性疾病比例"环形图依赖）
CREATE INDEX disease_infectious_idx IF NOT EXISTS
FOR (d:Disease) ON (d.is_infectious);

// 别名（实体链接的关键索引：口语简称 → 标准名）
CREATE INDEX disease_alias_idx IF NOT EXISTS
FOR (d:Disease) ON (d.alias);

CREATE INDEX symptom_alias_idx IF NOT EXISTS
FOR (s:Symptom) ON (s.alias);

CREATE INDEX department_category_idx IF NOT EXISTS
FOR (d:Department) ON (d.category);

// 溯源：按年份筛选文献
CREATE INDEX source_year_idx IF NOT EXISTS
FOR (s:Source) ON (s.year);

CREATE INDEX source_authority_idx IF NOT EXISTS
FOR (s:Source) ON (s.authority);

// ------------------------- 全文索引（实体检索输入联想）-------------------------
// Neo4j 5.x 全文索引语法：FOR (n:Label1|Label2) ON EACH [props]
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Disease|Symptom|Drug|Department|Treatment|Check)
ON EACH [n.name, n.alias];

// 文献标题全文索引（RAG 文献片段检索）
CREATE FULLTEXT INDEX source_title_fulltext IF NOT EXISTS
FOR (s:Source) ON EACH [s.title, s.journal];

// ------------------------- 关系属性索引 -------------------------
CREATE INDEX rel_confidence_idx IF NOT EXISTS
FOR ()-[r:HAS_SYMPTOM]-() ON (r.confidence);

CREATE INDEX rel_weight_idx IF NOT EXISTS
FOR ()-[r:TREATED_BY]-() ON (r.weight);

// 查看已创建的索引
// SHOW INDEXES;
