// =============================================================================
//  Neo4j 5.8 演示种子数据 —— 03 疾病种子（示例 6 条，可完整演示四大功能）
//  用法：在 Neo4j Browser 中直接执行；或 python scripts/load_kg.py 自动导入全部 46+ 条
// =============================================================================

// ------------------------- 1. 疾病节点 -------------------------
MERGE (d1:Disease {name: '原发性高血压'})
SET d1.disease_id = 'D0001',
    d1.alias = ['高血压病', 'essential hypertension'],
    d1.category1 = '内科',
    d1.category2 = '心血管内科',
    d1.definition = '原发性高血压是以体循环动脉压升高为主要临床表现的心血管综合征，简称高血压。它是我国最常见的慢性病之一，也是心脑血管病最主要的危险因素。',
    d1.cause = '病因尚不完全明确，目前认为是遗传因素与环境因素长期相互作用的结果，遗传因素约占40%，高盐饮食、超重肥胖、过量饮酒、长期精神紧张为重要环境因素。',
    d1.diagnosis = '在未使用降压药物的情况下，非同日3次测量诊室血压，收缩压≥140mmHg和/或舒张压≥90mmHg即可诊断为高血压。',
    d1.treatment = '治疗目标是降低心脑血管事件风险，包括生活方式干预与药物治疗两部分，需长期坚持并定期随访。',
    d1.prognosis = '长期规范降压治疗可使脑卒中风险显著降低，血压达标者预后良好。',
    d1.population = '多见于中老年人、肥胖者、高盐饮食者及有高血压家族史的人群',
    d1.is_infectious = false,
    d1.updated_at = '2026-01-15T08:00:00Z';

MERGE (d2:Disease {name: '2型糖尿病'})
SET d2.disease_id = 'D0014',
    d2.alias = ['成人发病型糖尿病', 'T2DM'],
    d2.category1 = '内科',
    d2.category2 = '内分泌科',
    d2.definition = '2型糖尿病是以胰岛素抵抗伴相对胰岛素分泌不足为特征的慢性代谢性疾病，以高血糖为主要标志。',
    d2.cause = '由遗传因素与环境因素共同作用引起，超重肥胖、体力活动不足、高热量饮食是主要危险因素。',
    d2.diagnosis = '空腹血糖≥7.0mmol/L，或OGTT 2小时血糖≥11.1mmol/L，或糖化血红蛋白≥6.5%，或有典型症状伴随机血糖≥11.1mmol/L。',
    d2.treatment = '综合治疗：医学营养治疗、运动治疗、血糖监测、健康教育及药物治疗（口服降糖药与胰岛素）。',
    d2.prognosis = '长期良好控糖可显著延缓并发症发生，预后取决于血糖控制水平与并发症管理。',
    d2.population = '多见于中老年人、肥胖人群及有糖尿病家族史人群',
    d2.is_infectious = false,
    d2.updated_at = '2026-01-15T08:00:00Z';

MERGE (d3:Disease {name: '肺栓塞'})
SET d3.disease_id = 'D0021',
    d3.alias = ['肺动脉栓塞', 'PE'],
    d3.category1 = '内科',
    d3.category2 = '呼吸内科',
    d3.definition = '肺栓塞是由内源性或外源性栓子阻塞肺动脉或其分支引起肺循环障碍的临床和病理生理综合征。',
    d3.cause = '最常见为深静脉血栓脱落，危险因素包括长期卧床、手术、创伤、恶性肿瘤、口服避孕药及遗传性易栓症。',
    d3.diagnosis = '结合D-二聚体、CT肺动脉造影（CTPA）等检查明确，需进行临床可能性评分。',
    d3.treatment = '以抗凝治疗为基础，高危患者需溶栓或介入/手术治疗，同时处理原发血栓来源。',
    d3.prognosis = '早期诊断并规范抗凝者预后良好，未治疗的高危肺栓塞病死率高。',
    d3.population = '多见于长期卧床、术后、恶性肿瘤及有血栓病史的人群',
    d3.is_infectious = false,
    d3.updated_at = '2026-01-15T08:00:00Z';

MERGE (d4:Disease {name: '流行性感冒'})
SET d4.disease_id = 'D0013',
    d4.alias = ['流感'],
    d4.category1 = '传染科',
    d4.category2 = '感染性疾病科',
    d4.definition = '流行性感冒是由流感病毒引起的急性呼吸道传染病，传染性强，常呈季节性流行。',
    d4.cause = '由甲型或乙型流感病毒感染引起，经呼吸道飞沫与接触传播。',
    d4.diagnosis = '结合流行病学史、临床表现与流感病毒核酸检测或抗原检测。',
    d4.treatment = '以对症支持治疗为主，重症或高危人群应尽早使用抗病毒药物。',
    d4.prognosis = '多数患者1-2周内自愈，老年人、孕妇及有基础疾病者可能发展为重症。',
    d4.population = '所有人群均易感，老年人、孕产妇、儿童及慢性病患者风险更高',
    d4.is_infectious = true,
    d4.updated_at = '2026-01-15T08:00:00Z';

MERGE (d5:Disease {name: '急性单纯性胃炎'})
SET d5.disease_id = 'D0027',
    d5.alias = ['急性胃炎'],
    d5.category1 = '内科',
    d5.category2 = '消化内科',
    d5.definition = '急性单纯性胃炎是由多种原因引起的胃黏膜急性炎症，是最常见的急性胃黏膜病变。',
    d5.cause = '常由不洁饮食、刺激性食物、药物（如非甾体抗炎药）、酒精及应激因素引起。',
    d5.diagnosis = '主要依据病史与临床表现，必要时行胃镜检查明确。',
    d5.treatment = '去除病因，对症治疗，必要时使用胃黏膜保护剂与抑酸药物。',
    d5.prognosis = '去除病因后多可自愈，预后良好。',
    d5.population = '所有人群',
    d5.is_infectious = false,
    d5.updated_at = '2026-01-15T08:00:00Z';

// ------------------------- 2. 症状节点 + 关系 -------------------------
MERGE (s1:Symptom {name: '头晕'})
MERGE (s2:Symptom {name: '头痛'})
MERGE (s3:Symptom {name: '心悸'})
MERGE (s4:Symptom {name: '突发性呼吸困难'})
MERGE (s5:Symptom {name: '胸痛'})
MERGE (s6:Symptom {name: '咯血'})
MERGE (s7:Symptom {name: '发热'})
MERGE (s8:Symptom {name: '咳嗽'})
MERGE (s9:Symptom {name: '恶心'})
MERGE (s10:Symptom {name: '呕吐'})
MERGE (s11:Symptom {name: '腹痛'})
MERGE (s12:Symptom {name: '上腹部疼痛'})
MERGE (s13:Symptom {name: '多饮'})
MERGE (s14:Symptom {name: '多尿'})
MERGE (s15:Symptom {name: '体重下降'})

MERGE (d1)-[:HAS_SYMPTOM {confidence: 0.98, weight: 1.0}]->(s1)
MERGE (d1)-[:HAS_SYMPTOM {confidence: 0.97, weight: 1.0}]->(s2)
MERGE (d1)-[:HAS_SYMPTOM {confidence: 0.92, weight: 0.8}]->(s3)
MERGE (d3)-[:HAS_SYMPTOM {confidence: 0.96, weight: 1.0}]->(s4)
MERGE (d3)-[:HAS_SYMPTOM {confidence: 0.95, weight: 1.0}]->(s5)
MERGE (d3)-[:HAS_SYMPTOM {confidence: 0.88, weight: 0.8}]->(s6)
MERGE (d4)-[:HAS_SYMPTOM {confidence: 0.97, weight: 1.0}]->(s7)
MERGE (d4)-[:HAS_SYMPTOM {confidence: 0.93, weight: 0.9}]->(s8)
MERGE (d4)-[:HAS_SYMPTOM {confidence: 0.85, weight: 0.8}]->(s2)
MERGE (d5)-[:HAS_SYMPTOM {confidence: 0.94, weight: 1.0}]->(s9)
MERGE (d5)-[:HAS_SYMPTOM {confidence: 0.93, weight: 1.0}]->(s10)
MERGE (d5)-[:HAS_SYMPTOM {confidence: 0.95, weight: 1.0}]->(s12)
MERGE (d5)-[:HAS_SYMPTOM {confidence: 0.90, weight: 1.0}]->(s11)
MERGE (d2)-[:HAS_SYMPTOM {confidence: 0.93, weight: 1.0}]->(s13)
MERGE (d2)-[:HAS_SYMPTOM {confidence: 0.93, weight: 1.0}]->(s14)
MERGE (d2)-[:HAS_SYMPTOM {confidence: 0.88, weight: 0.9}]->(s15)
MERGE (d2)-[:HAS_SYMPTOM {confidence: 0.80, weight: 0.7}]->(s1)

// ------------------------- 3. 科室节点 + 关系 -------------------------
MERGE (dept1:Department {name: '内科', category: '一级科室'})
MERGE (dept2:Department {name: '心血管内科', category: '二级科室'})
MERGE (dept3:Department {name: '内分泌科', category: '二级科室'})
MERGE (dept4:Department {name: '呼吸内科', category: '二级科室'})
MERGE (dept5:Department {name: '消化内科', category: '二级科室'})
MERGE (dept6:Department {name: '感染性疾病科', category: '二级科室'})
MERGE (dept7:Department {name: '传染科', category: '一级科室'})

MERGE (d1)-[:BELONGS_TO {confidence: 0.99}]->(dept1)
MERGE (d1)-[:BELONGS_TO {confidence: 0.99}]->(dept2)
MERGE (d2)-[:BELONGS_TO {confidence: 0.99}]->(dept1)
MERGE (d2)-[:BELONGS_TO {confidence: 0.99}]->(dept3)
MERGE (d3)-[:BELONGS_TO {confidence: 0.99}]->(dept1)
MERGE (d3)-[:BELONGS_TO {confidence: 0.99}]->(dept4)
MERGE (d4)-[:BELONGS_TO {confidence: 0.99}]->(dept7)
MERGE (d4)-[:BELONGS_TO {confidence: 0.95}]->(dept6)
MERGE (d5)-[:BELONGS_TO {confidence: 0.99}]->(dept1)
MERGE (d5)-[:BELONGS_TO {confidence: 0.99}]->(dept5)

// ------------------------- 4. 治疗 / 药物 / 检查 -------------------------
MERGE (t1:Treatment {name: '药物治疗', category: '内科治疗'})
MERGE (t2:Treatment {name: '饮食治疗', category: '生活方式干预'})
MERGE (t3:Treatment {name: '运动康复', category: '康复治疗'})
MERGE (t4:Treatment {name: '抗凝治疗', category: '药物治疗'})
MERGE (t5:Treatment {name: '溶栓治疗', category: '急诊治疗'})
MERGE (t6:Treatment {name: '对症支持治疗', category: '内科治疗'})

MERGE (d1)-[:TREATED_BY {confidence: 0.97}]->(t1)
MERGE (d1)-[:TREATED_BY {confidence: 0.93}]->(t2)
MERGE (d1)-[:TREATED_BY {confidence: 0.88}]->(t3)
MERGE (d2)-[:TREATED_BY {confidence: 0.97}]->(t1)
MERGE (d2)-[:TREATED_BY {confidence: 0.95}]->(t2)
MERGE (d3)-[:TREATED_BY {confidence: 0.96}]->(t4)
MERGE (d3)-[:TREATED_BY {confidence: 0.88}]->(t5)
MERGE (d4)-[:TREATED_BY {confidence: 0.94}]->(t6)
MERGE (d5)-[:TREATED_BY {confidence: 0.92}]->(t1)

MERGE (dr1:Drug {name: '氨氯地平', category: '钙通道阻滞剂'})
MERGE (dr2:Drug {name: '缬沙坦', category: 'ARB 类降压药'})
MERGE (dr3:Drug {name: '美托洛尔', category: 'β 受体阻滞剂'})
MERGE (dr4:Drug {name: '二甲双胍', category: '双胍类降糖药'})
MERGE (dr5:Drug {name: '阿卡波糖', category: 'α-糖苷酶抑制剂'})
MERGE (dr6:Drug {name: '利伐沙班', category: '直接口服抗凝药'})
MERGE (dr7:Drug {name: '华法林', category: '维生素K拮抗剂'})
MERGE (dr8:Drug {name: '奥美拉唑', category: '质子泵抑制剂'})
MERGE (dr9:Drug {name: '奥司他韦', category: '神经氨酸酶抑制剂'})

MERGE (d1)-[:USES_DRUG {confidence: 0.96}]->(dr1)
MERGE (d1)-[:USES_DRUG {confidence: 0.95}]->(dr2)
MERGE (d1)-[:USES_DRUG {confidence: 0.92}]->(dr3)
MERGE (d2)-[:USES_DRUG {confidence: 0.97}]->(dr4)
MERGE (d2)-[:USES_DRUG {confidence: 0.90}]->(dr5)
MERGE (d3)-[:USES_DRUG {confidence: 0.95}]->(dr6)
MERGE (d3)-[:USES_DRUG {confidence: 0.90}]->(dr7)
MERGE (d5)-[:USES_DRUG {confidence: 0.93}]->(dr8)
MERGE (d4)-[:USES_DRUG {confidence: 0.92}]->(dr9)

MERGE (c1:Check {name: '诊室血压测量'})
MERGE (c2:Check {name: '动态血压监测'})
MERGE (c3:Check {name: '血生化'})
MERGE (c4:Check {name: '尿常规'})
MERGE (c5:Check {name: '心电图'})
MERGE (c6:Check {name: '超声心动图'})
MERGE (c7:Check {name: '空腹血糖'})
MERGE (c8:Check {name: '糖化血红蛋白'})
MERGE (c9:Check {name: '口服葡萄糖耐量试验'})
MERGE (c10:Check {name: 'D-二聚体'})
MERGE (c11:Check {name: 'CT肺动脉造影'})
MERGE (c12:Check {name: '胃镜'})

MERGE (d1)-[:NEEDS_CHECK {confidence: 0.98}]->(c1)
MERGE (d1)-[:NEEDS_CHECK {confidence: 0.95}]->(c2)
MERGE (d1)-[:NEEDS_CHECK {confidence: 0.93}]->(c3)
MERGE (d1)-[:NEEDS_CHECK {confidence: 0.92}]->(c4)
MERGE (d1)-[:NEEDS_CHECK {confidence: 0.94}]->(c5)
MERGE (d1)-[:NEEDS_CHECK {confidence: 0.90}]->(c6)
MERGE (d2)-[:NEEDS_CHECK {confidence: 0.98}]->(c7)
MERGE (d2)-[:NEEDS_CHECK {confidence: 0.96}]->(c8)
MERGE (d2)-[:NEEDS_CHECK {confidence: 0.90}]->(c9)
MERGE (d3)-[:NEEDS_CHECK {confidence: 0.94}]->(c10)
MERGE (d3)-[:NEEDS_CHECK {confidence: 0.97}]->(c11)
MERGE (d5)-[:NEEDS_CHECK {confidence: 0.90}]->(c12)
MERGE (d4)-[:NEEDS_CHECK {confidence: 0.88}]->(c3)

// ------------------------- 5. 易感人群 -------------------------
MERGE (p1:Population {name: '中老年人'})
MERGE (p2:Population {name: '肥胖人群'})
MERGE (p3:Population {name: '有家族史人群'})
MERGE (p4:Population {name: '所有人群'})
MERGE (p5:Population {name: '免疫力低下人群'})

MERGE (d1)-[:AFFECTS {confidence: 0.92}]->(p1)
MERGE (d1)-[:AFFECTS {confidence: 0.88}]->(p2)
MERGE (d1)-[:AFFECTS {confidence: 0.85}]->(p3)
MERGE (d2)-[:AFFECTS {confidence: 0.90}]->(p2)
MERGE (d2)-[:AFFECTS {confidence: 0.86}]->(p3)
MERGE (d2)-[:AFFECTS {confidence: 0.80}]->(p1)
MERGE (d4)-[:AFFECTS {confidence: 0.90}]->(p4)
MERGE (d4)-[:AFFECTS {confidence: 0.82}]->(p5)
MERGE (d3)-[:AFFECTS {confidence: 0.84}]->(p1)
MERGE (d5)-[:AFFECTS {confidence: 0.90}]->(p4)

// ------------------------- 6. 并发症与鉴别诊断 -------------------------
MERGE (d6:Disease {name: '冠心病'})
MERGE (d7:Disease {name: '慢性肾脏病'})
MERGE (d8:Disease {name: '脑卒中'})
MERGE (d9:Disease {name: '继发性高血压'})

MERGE (d1)-[:HAS_COMPLICATION {confidence: 0.90}]->(d6)
MERGE (d1)-[:HAS_COMPLICATION {confidence: 0.85}]->(d7)
MERGE (d1)-[:HAS_COMPLICATION {confidence: 0.92}]->(d8)
MERGE (d1)-[:DIFFERENTIAL_WITH {confidence: 0.88}]->(d9)
MERGE (d2)-[:HAS_COMPLICATION {confidence: 0.86}]->(d7)
MERGE (d2)-[:HAS_COMPLICATION {confidence: 0.88}]->(d6)
MERGE (d4)-[:DIFFERENTIAL_WITH {confidence: 0.85}]->(d5)

// ------------------------- 7. 文献来源（知识溯源）-------------------------
MERGE (src1:Source {pmid: '32130469'})
SET src1.source_type = 'pubmed',
    src1.title = '2020 International Society of Hypertension Global Hypertension Practice Guidelines',
    src1.journal = 'Hypertension',
    src1.year = 2020,
    src1.doi = '10.1097/HJH.0000000000002425',
    src1.url = 'https://pubmed.ncbi.nlm.nih.gov/32130469/',
    src1.authority = 'A',
    src1.retrieved_at = '2026-01-15T08:00:00Z';

MERGE (src2:Source {pmid: '33525379'})
SET src2.source_type = 'guideline',
    src2.title = '中国高血压防治指南(2024年修订版)',
    src2.journal = '中华心血管病杂志',
    src2.year = 2024,
    src2.url = 'https://www.nccd.org.cn/',
    src2.authority = 'A',
    src2.retrieved_at = '2026-01-15T08:00:00Z';

MERGE (d1)-[:PROVES]->(src1)
MERGE (d1)-[:PROVES]->(src2)

// ------------------------- 8. 验证查询 -------------------------
// 查看导入结果
MATCH (n) RETURN labels(n)[0] AS 节点类型, count(n) AS 数量 ORDER BY 数量 DESC;
MATCH ()-[r]->() RETURN type(r) AS 关系类型, count(r) AS 数量 ORDER BY 数量 DESC;
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d.name AS 疾病, collect(s.name) AS 症状 LIMIT 5;
MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department {name: '呼吸内科'})
RETURN d.name AS 呼吸内科疾病;
