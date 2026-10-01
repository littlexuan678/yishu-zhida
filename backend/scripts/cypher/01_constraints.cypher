# -*- coding: utf-8 -*-
# =============================================================================
#  Neo4j 5.8 图谱初始化 —— 01 唯一性约束
#  执行方式：
#     python scripts/init_neo4j.py                （推荐，自动按序执行）
#     或在 Neo4j Browser 中粘贴本文件全部语句执行
# =============================================================================

// ------------------------- 疾病 -------------------------
CREATE CONSTRAINT disease_name_unique IF NOT EXISTS
FOR (d:Disease) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT disease_id_unique IF NOT EXISTS
FOR (d:Disease) REQUIRE d.disease_id IS UNIQUE;

// ------------------------- 症状 -------------------------
CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS
FOR (s:Symptom) REQUIRE s.name IS UNIQUE;

// ------------------------- 药物 -------------------------
CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
FOR (d:Drug) REQUIRE d.name IS UNIQUE;

// ------------------------- 科室 -------------------------
CREATE CONSTRAINT department_name_unique IF NOT EXISTS
FOR (d:Department) REQUIRE d.name IS UNIQUE;

// ------------------------- 治疗方法 -------------------------
CREATE CONSTRAINT treatment_name_unique IF NOT EXISTS
FOR (t:Treatment) REQUIRE t.name IS UNIQUE;

// ------------------------- 检查项目 -------------------------
CREATE CONSTRAINT check_name_unique IF NOT EXISTS
FOR (c:Check) REQUIRE c.name IS UNIQUE;

// ------------------------- 易感人群 -------------------------
CREATE CONSTRAINT population_name_unique IF NOT EXISTS
FOR (p:Population) REQUIRE p.name IS UNIQUE;

// ------------------------- 文献来源（溯源锚点）-------------------------
CREATE CONSTRAINT source_pmid_unique IF NOT EXISTS
FOR (s:Source) REQUIRE s.pmid IS UNIQUE;

CREATE CONSTRAINT source_url_unique IF NOT EXISTS
FOR (s:Source) REQUIRE s.url IS UNIQUE;

// 查看已创建的约束
// SHOW CONSTRAINTS;
