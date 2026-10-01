/**
 * 离线演示数据（后端不可用时自动兜底）
 * 所有结构严格对齐后端 API 契约（snake_case）。
 */

/* ============================== 知识图谱 ============================== */

export const ENTITY_COLORS = {
  疾病: '#409EFF',
  症状: '#31c48d',
  科室: '#F56C6C',
  药物: '#E6A23C',
  治疗方法: '#b37feb',
  检查项目: '#909399',
  易感人群: '#F2C037',
  文献来源: '#00BCD4'
}

export const entityTypes = [
  { type: 'disease', label: '疾病', color: '#409EFF', count: 128 },
  { type: 'symptom', label: '症状', color: '#31c48d', count: 356 },
  { type: 'department', label: '科室', color: '#F56C6C', count: 42 },
  { type: 'drug', label: '药物', color: '#E6A23C', count: 214 },
  { type: 'treatment', label: '治疗方法', color: '#b37feb', count: 96 },
  { type: 'examination', label: '检查项目', color: '#909399', count: 138 },
  { type: 'population', label: '易感人群', color: '#F2C037', count: 64 },
  { type: 'literature', label: '文献来源', color: '#00BCD4', count: 82 }
]

export const graphStats = {
  total_nodes: 1120,
  total_links: 3684,
  entity_type_count: 8,
  entity_types: entityTypes,
  disease_count: 128,
  symptom_count: 356,
  drug_count: 214,
  department_count: 42,
  treatment_count: 96,
  source_count: 82,
  data_source: '离线演示数据'
}

const REL_LABELS = {
  has_symptom: '症状',
  belongs_to: '所属科室',
  treated_by: '治疗方式',
  used_drug: '常用药物',
  needs_exam: '检查项目',
  susceptible_to: '易感人群',
  complicated_with: '并发症',
  cited_from: '来源文献',
  differential_from: '鉴别诊断',
  caused_by: '病因'
}

/**
 * 生成以某个实体为中心的子图（离线兜底）
 */
export function buildSubgraph (entity = '高血压', depth = 1, limit = 200) {
  const centerName = entity && entity.trim() ? entity.trim() : '高血压'

  const KNOWN = {
    高血压: {
      type: '疾病',
      symptoms: ['头晕', '头痛', '颈项板紧', '心悸', '耳鸣', '视物模糊', '失眠', '乏力'],
      drugs: ['氨氯地平', '缬沙坦', '氢氯噻嗪', '美托洛尔', '硝苯地平'],
      treatments: ['生活方式干预', '降压药物治疗', '限盐饮食', '规律有氧运动'],
      department: '心血管内科',
      exams: ['动态血压监测', '血生化检查', '心电图', '心脏彩超'],
      population: '中老年人群',
      literature: '《中国高血压防治指南(2023年修订版)》'
    },
    糖尿病: {
      type: '疾病',
      symptoms: ['多饮', '多尿', '多食', '体重下降', '乏力', '视物模糊'],
      drugs: ['二甲双胍', '格列美脲', '阿卡波糖', '胰岛素'],
      treatments: ['饮食控制', '运动疗法', '口服降糖药', '胰岛素治疗'],
      department: '内分泌科',
      exams: ['空腹血糖', '糖化血红蛋白', '口服葡萄糖耐量试验'],
      population: '肥胖人群',
      literature: '《中国2型糖尿病防治指南(2020年版)》'
    },
    冠心病: {
      type: '疾病',
      symptoms: ['胸痛', '胸闷', '气促', '心悸', '出汗'],
      drugs: ['阿司匹林', '阿托伐他汀', '硝酸甘油', '氯吡格雷'],
      treatments: ['经皮冠状动脉介入治疗', '冠状动脉旁路移植术', '抗血小板治疗'],
      department: '心血管内科',
      exams: ['冠状动脉造影', '运动负荷试验', '心脏彩超'],
      population: '中老年男性',
      literature: '《稳定性冠心病诊断与治疗指南》'
    },
    肺炎: {
      type: '疾病',
      symptoms: ['发热', '咳嗽', '咳痰', '胸痛', '呼吸困难'],
      drugs: ['阿莫西林', '左氧氟沙星', '阿奇霉素', '氨溴索'],
      treatments: ['抗感染治疗', '氧疗', '对症支持治疗'],
      department: '呼吸内科',
      exams: ['胸部X线', '胸部CT', '痰培养', '血常规'],
      population: '老年人及儿童',
      literature: '《中国成人社区获得性肺炎诊断和治疗指南》'
    },
    胃炎: {
      type: '疾病',
      symptoms: ['上腹痛', '腹胀', '反酸', '嗳气', '恶心'],
      drugs: ['奥美拉唑', '铝碳酸镁', '莫沙必利', '阿莫西林'],
      treatments: ['抑酸治疗', '根除幽门螺杆菌', '饮食调理'],
      department: '消化内科',
      exams: ['胃镜', '幽门螺杆菌检测', '胃功能三项'],
      population: '饮食不规律人群',
      literature: '《中国慢性胃炎共识意见》'
    },
    感冒: {
      type: '疾病',
      symptoms: ['鼻塞', '流涕', '咽痛', '咳嗽', '发热', '乏力'],
      drugs: ['连花清瘟胶囊', '对乙酰氨基酚', '氯雷他定'],
      treatments: ['对症治疗', '多饮水休息'],
      department: '呼吸内科',
      exams: ['血常规', 'C反应蛋白'],
      population: '全人群',
      literature: '《普通感冒规范诊治的专家共识》'
    }
  }

  const info = KNOWN[centerName] || KNOWN['高血压']
  const isFallback = !KNOWN[centerName]

  const nodes = []
  const links = []
  let seq = 0
  const pushNode = (name, type, degree, extra = {}) => {
    const id = `n${++seq}`
    nodes.push({
      id,
      name,
      type,
      label: type,
      color: ENTITY_COLORS[type] || '#409EFF',
      degree,
      category1: extra.category1 || '',
      category2: extra.category2 || '',
      properties: extra.properties || {}
    })
    return id
  }

  const centerId = pushNode(centerName, '疾病', info.symptoms.length + info.drugs.length + 4, {
    category1: '内科疾病',
    category2: '心血管系统疾病',
    properties: {
      别名: isFallback ? '' : '原发性高血压',
      是否传染: '否',
      英文名: 'Hypertension',
      数据来源: '离线演示数据'
    }
  })

  const link = (source, target, rel) => {
    links.push({ source, target, rel, label: REL_LABELS[rel] || rel, properties: {} })
  }

  info.symptoms.forEach((s, i) => {
    const id = pushNode(s, '症状', Math.max(2, 8 - i))
    link(centerId, id, 'has_symptom')
  })
  info.drugs.forEach((d) => {
    const id = pushNode(d, '药物', 3)
    link(centerId, id, 'used_drug')
  })
  info.treatments.forEach((t) => {
    const id = pushNode(t, '治疗方法', 3)
    link(centerId, id, 'treated_by')
  })
  info.exams.forEach((e) => {
    const id = pushNode(e, '检查项目', 2)
    link(centerId, id, 'needs_exam')
  })
  const depId = pushNode(info.department, '科室', 6, { properties: { 类型: '临床科室' } })
  link(centerId, depId, 'belongs_to')
  const popId = pushNode(info.population, '易感人群', 3)
  link(centerId, popId, 'susceptible_to')
  const litId = pushNode(info.literature, '文献来源', 2, {
    properties: { 年份: '2023', 来源类型: '临床指南' }
  })
  link(centerId, litId, 'cited_from')

  // 二级关联：让图谱更像真实知识网络
  const secondary = [
    ['头晕', '贫血', 'differential_from', '疾病'],
    ['心悸', '心律失常', 'complicated_with', '疾病'],
    ['氨氯地平', '外周水肿', 'caused_by', '症状'],
    ['缬沙坦', '高钾血症', 'caused_by', '症状']
  ]
  secondary.forEach(([a, b, rel, bType]) => {
    const aNode = nodes.find((n) => n.name === a)
    if (!aNode) return
    const bNode = nodes.find((n) => n.name === b)
    const bid = bNode ? bNode.id : pushNode(b, bType, 2)
    link(aNode.id, bid, rel)
  })

  const limited = nodes.slice(0, Math.max(4, Number(limit) || 200))
  const idSet = new Set(limited.map((n) => n.id))
  const limitedLinks = links.filter((l) => {
    const s = typeof l.source === 'object' ? l.source.id : l.source
    const t = typeof l.target === 'object' ? l.target.id : l.target
    return idSet.has(s) && idSet.has(t)
  })

  return {
    nodes: limited,
    links: limitedLinks,
    node_count: limited.length,
    link_count: limitedLinks.length,
    type_count: new Set(limited.map((n) => n.type)).size,
    center: centerName,
    depth: Number(depth) || 1,
    truncated: nodes.length > limited.length,
    message: isFallback
      ? `未收录「${centerName}」，已展示「高血压」示例子图（离线演示数据）`
      : `已构建以「${centerName}」为中心的知识子图（离线演示数据）`
  }
}

export function graphSearch (keyword = '') {
  const kw = (keyword || '').trim() || '血压'
  const pool = [
    { name: '高血压', type: '疾病' },
    { name: '高血压性心脏病', type: '疾病' },
    { name: '原发性高血压', type: '疾病' },
    { name: '头晕', type: '症状' },
    { name: '血压升高', type: '症状' },
    { name: '心血管内科', type: '科室' },
    { name: '氨氯地平', type: '药物' },
    { name: '动态血压监测', type: '检查项目' }
  ]
  const items = pool
    .filter((p) => p.name.includes(kw) || kw.includes(p.name) || p.type === '疾病')
    .slice(0, 8)
    .map((p, i) => ({
      id: `s${i + 1}`,
      name: p.name,
      type: p.type,
      label: p.type,
      color: ENTITY_COLORS[p.type] || '#409EFF',
      score: Number((0.98 - i * 0.07).toFixed(2)),
      category1: '内科疾病'
    }))
  return { keyword: kw, total: items.length, items }
}

/* ============================== 疾病查询 ============================== */

const DISEASES = [
  {
    disease_id: 'D001',
    name: '高血压',
    alias: ['原发性高血压', '血压偏高'],
    category1: '内科疾病',
    category2: '心血管系统疾病',
    definition: '高血压是以体循环动脉血压持续升高为主要表现的临床综合征，通常指未使用降压药物的情况下，非同日三次测量收缩压≥140mmHg和/或舒张压≥90mmHg。',
    cause: '病因尚未完全明确，与遗传因素、高钠低钾饮食、超重和肥胖、过量饮酒、长期精神紧张、缺乏体力活动、年龄增长等密切相关。',
    symptoms: ['头晕', '头痛', '颈项板紧', '心悸', '耳鸣', '视物模糊', '失眠', '乏力'],
    diagnosis: '依据诊室血压、家庭自测血压及24小时动态血压监测结果综合判断，同时评估靶器官损害与心血管总体风险分层。',
    checks: ['动态血压监测', '血生化检查', '尿常规', '心电图', '心脏彩超', '颈动脉超声'],
    treatment: '以生活方式干预为基础，联合个体化降压药物治疗，长期平稳达标并保护靶器官。',
    treatments: ['生活方式干预', '降压药物治疗', '限盐饮食', '规律有氧运动', '减重与戒烟限酒'],
    drugs: ['氨氯地平', '缬沙坦', '氢氯噻嗪', '美托洛尔', '硝苯地平'],
    department: '心血管内科',
    prognosis: '长期规范治疗者血压可长期达标，预后良好；未控制者可出现心、脑、肾等靶器官损害。',
    population: '中老年人群、肥胖人群、有高血压家族史者、高盐饮食人群',
    complications: ['脑卒中', '心力衰竭', '冠心病', '慢性肾脏病', '视网膜病变'],
    differential: ['白大衣高血压', '继发性高血压', '肾性高血压', '原发性醛固酮增多症'],
    is_infectious: false,
    sources: [
      {
        source_type: 'guideline',
        pmid: '37130140',
        doi: '10.3760/cma.j.cn112148-20230110-00021',
        title: '中国高血压防治指南(2023年修订版)',
        journal: '中华心血管病杂志',
        year: 2023,
        authors: '中国高血压防治指南修订委员会',
        url: 'https://doi.org/10.3760/cma.j.cn112148-20230110-00021',
        authority: 'A级',
        retrieved_at: '2024-03-11'
      },
      {
        source_type: 'review',
        pmid: '36220816',
        doi: '10.1016/S0140-6736(22)01872-0',
        title: 'Hypertension: a global perspective',
        journal: 'The Lancet',
        year: 2022,
        authors: 'Zhou B, Perel P, Mensah GA, et al.',
        url: 'https://pubmed.ncbi.nlm.nih.gov/36220816/',
        authority: 'A级',
        retrieved_at: '2024-03-11'
      }
    ],
    updated_at: '2024-03-11'
  },
  {
    disease_id: 'D002',
    name: '2型糖尿病',
    alias: ['成人发病型糖尿病', '非胰岛素依赖型糖尿病'],
    category1: '内科疾病',
    category2: '内分泌与代谢疾病',
    definition: '2型糖尿病是以胰岛素抵抗伴相对胰岛素分泌不足为特征的慢性代谢性疾病，长期高血糖可导致多系统慢性并发症。',
    cause: '遗传易感性叠加超重肥胖、久坐少动、高热量饮食、增龄等因素，导致胰岛素抵抗与β细胞功能减退。',
    symptoms: ['多饮', '多尿', '多食', '体重下降', '乏力', '视物模糊', '肢体麻木'],
    diagnosis: '空腹血糖≥7.0mmol/L，或OGTT 2小时血糖≥11.1mmol/L，或糖化血红蛋白≥6.5%，或有典型症状伴随机血糖≥11.1mmol/L。',
    checks: ['空腹血糖', '糖化血红蛋白', '口服葡萄糖耐量试验', '胰岛素释放试验', '尿微量白蛋白'],
    treatment: '以医学营养治疗和运动治疗为基础，联合口服降糖药与胰岛素治疗，综合管理血糖、血压、血脂和体重。',
    treatments: ['饮食控制', '运动疗法', '口服降糖药', '胰岛素治疗', '血糖监测'],
    drugs: ['二甲双胍', '格列美脲', '阿卡波糖', '达格列净', '胰岛素'],
    department: '内分泌科',
    prognosis: '早期强化管理可使部分患者获得缓解；长期血糖控制不佳者并发症风险明显升高。',
    population: '肥胖人群、有糖尿病家族史者、40岁以上人群、妊娠期糖尿病史女性',
    complications: ['糖尿病肾病', '糖尿病视网膜病变', '糖尿病周围神经病变', '糖尿病足', '心脑血管疾病'],
    differential: ['1型糖尿病', '特殊类型糖尿病', '应激性高血糖', '尿崩症'],
    is_infectious: false,
    sources: [
      {
        source_type: 'guideline',
        pmid: '34879974',
        doi: '10.3760/cma.j.cn115791-20210330-00195',
        title: '中国2型糖尿病防治指南(2020年版)',
        journal: '中华糖尿病杂志',
        year: 2021,
        authors: '中华医学会糖尿病学分会',
        url: 'https://doi.org/10.3760/cma.j.cn115791-20210330-00195',
        authority: 'A级',
        retrieved_at: '2024-02-18'
      }
    ],
    updated_at: '2024-02-18'
  },
  {
    disease_id: 'D003',
    name: '冠心病',
    alias: ['冠状动脉粥样硬化性心脏病', '缺血性心脏病'],
    category1: '内科疾病',
    category2: '心血管系统疾病',
    definition: '冠心病是冠状动脉粥样硬化导致管腔狭窄或闭塞，引起心肌缺血、缺氧或坏死的心脏病，临床分为稳定性心绞痛与急性冠脉综合征。',
    cause: '主要危险因素包括血脂异常、高血压、糖尿病、吸烟、肥胖、缺乏运动、年龄与家族史。',
    symptoms: ['胸痛', '胸闷', '气促', '心悸', '出汗', '乏力'],
    diagnosis: '结合典型症状、心电图、心肌损伤标志物、运动负荷试验及冠状动脉造影或CTA结果综合诊断。',
    checks: ['冠状动脉造影', '冠状动脉CTA', '运动负荷试验', '心脏彩超', '心肌酶谱'],
    treatment: '抗血小板、调脂稳定斑块、抗心肌缺血治疗，必要时行经皮冠状动脉介入治疗或冠状动脉旁路移植术。',
    treatments: ['经皮冠状动脉介入治疗', '冠状动脉旁路移植术', '抗血小板治疗', '他汀调脂治疗', '心脏康复'],
    drugs: ['阿司匹林', '氯吡格雷', '阿托伐他汀', '硝酸甘油', '美托洛尔'],
    department: '心血管内科',
    prognosis: '规范二级预防可显著降低心血管事件风险，左主干或三支病变者预后相对较差。',
    population: '中老年男性、吸烟人群、血脂异常者、糖尿病患者',
    complications: ['急性心肌梗死', '心力衰竭', '心律失常', '心源性猝死'],
    differential: ['主动脉夹层', '肺栓塞', '心包炎', '反流性食管炎'],
    is_infectious: false,
    sources: [
      {
        source_type: 'guideline',
        pmid: '30180455',
        doi: '10.3760/cma.j.issn.0253-3758.2018.09.003',
        title: '稳定性冠心病诊断与治疗指南',
        journal: '中华心血管病杂志',
        year: 2018,
        authors: '中华医学会心血管病学分会',
        url: 'https://doi.org/10.3760/cma.j.issn.0253-3758.2018.09.003',
        authority: 'A级',
        retrieved_at: '2024-01-22'
      }
    ],
    updated_at: '2024-01-22'
  },
  {
    disease_id: 'D004',
    name: '社区获得性肺炎',
    alias: ['CAP', '院外肺炎'],
    category1: '呼吸系统疾病',
    category2: '感染性疾病',
    definition: '社区获得性肺炎是患者在医院外罹患的肺实质感染性炎症，是常见的感染性疾病之一。',
    cause: '常见病原体包括肺炎链球菌、支原体、流感嗜血杆菌及呼吸道病毒等。',
    symptoms: ['发热', '咳嗽', '咳痰', '胸痛', '呼吸困难', '乏力'],
    diagnosis: '依据新近出现的咳嗽咳痰或原有呼吸道症状加重、发热、肺实变体征及胸部影像学新发浸润影，并排除其他疾病。',
    checks: ['胸部X线', '胸部CT', '血常规', 'C反应蛋白', '痰培养', '降钙素原'],
    treatment: '尽早经验性抗感染治疗，结合病原学结果调整，辅以氧疗、祛痰及对症支持治疗。',
    treatments: ['抗感染治疗', '氧疗', '祛痰治疗', '对症支持治疗'],
    drugs: ['阿莫西林', '左氧氟沙星', '阿奇霉素', '氨溴索', '莫西沙星'],
    department: '呼吸内科',
    prognosis: '轻中症患者规范治疗后多数可痊愈，老年及有基础疾病者病死率升高。',
    population: '老年人、儿童、免疫功能低下者、慢性病患者',
    complications: ['脓毒症', '呼吸衰竭', '胸腔积液', '肺脓肿'],
    differential: ['肺结核', '肺癌', '肺栓塞', '急性支气管炎'],
    is_infectious: true,
    sources: [
      {
        source_type: 'guideline',
        pmid: '29338782',
        doi: '10.3760/cma.j.issn.1001-0939.2016.04.005',
        title: '中国成人社区获得性肺炎诊断和治疗指南(2016年版)',
        journal: '中华结核和呼吸杂志',
        year: 2016,
        authors: '中华医学会呼吸病学分会',
        url: 'https://doi.org/10.3760/cma.j.issn.1001-0939.2016.04.005',
        authority: 'A级',
        retrieved_at: '2024-02-02'
      }
    ],
    updated_at: '2024-02-02'
  },
  {
    disease_id: 'D005',
    name: '慢性胃炎',
    alias: ['慢性浅表性胃炎', '慢性萎缩性胃炎'],
    category1: '消化系统疾病',
    category2: '胃肠疾病',
    definition: '慢性胃炎是由多种病因引起的胃黏膜慢性炎症性病变，以幽门螺杆菌感染最常见。',
    cause: '幽门螺杆菌感染、长期服用非甾体抗炎药、胆汁反流、不良饮食习惯及自身免疫因素。',
    symptoms: ['上腹痛', '腹胀', '反酸', '嗳气', '恶心', '食欲不振'],
    diagnosis: '主要依据胃镜检查及胃黏膜活检病理组织学检查，同时检测幽门螺杆菌。',
    checks: ['胃镜', '幽门螺杆菌检测', '胃功能三项', '胃黏膜活检'],
    treatment: '根除幽门螺杆菌、抑酸保护胃黏膜、促动力治疗并调整饮食生活方式。',
    treatments: ['抑酸治疗', '根除幽门螺杆菌', '胃黏膜保护', '饮食调理'],
    drugs: ['奥美拉唑', '铝碳酸镁', '莫沙必利', '阿莫西林', '克拉霉素'],
    department: '消化内科',
    prognosis: '多数患者经规范治疗症状可缓解，萎缩性胃炎伴肠化者需定期随访。',
    population: '饮食不规律人群、幽门螺杆菌感染者、长期服药人群',
    complications: ['消化性溃疡', '胃出血', '胃黏膜肠化生', '胃癌'],
    differential: ['功能性消化不良', '消化性溃疡', '胃癌', '胆囊炎'],
    is_infectious: false,
    sources: [
      {
        source_type: 'consensus',
        pmid: '32341162',
        doi: '10.3760/cma.j.issn.0254-1432.2017.11.002',
        title: '中国慢性胃炎共识意见(2017年，上海)',
        journal: '中华消化杂志',
        year: 2017,
        authors: '中华医学会消化病学分会',
        url: 'https://doi.org/10.3760/cma.j.issn.0254-1432.2017.11.002',
        authority: 'A级',
        retrieved_at: '2024-01-30'
      }
    ],
    updated_at: '2024-01-30'
  },
  {
    disease_id: 'D006',
    name: '流行性感冒',
    alias: ['流感', '季节性流感'],
    category1: '呼吸系统疾病',
    category2: '感染性疾病',
    definition: '流行性感冒由流感病毒引起的急性呼吸道传染病，具有明显的季节性流行特征。',
    cause: '甲型、乙型流感病毒感染，经呼吸道飞沫和密切接触传播。',
    symptoms: ['高热', '头痛', '全身酸痛', '乏力', '咳嗽', '咽痛'],
    diagnosis: '流行季节出现发热伴呼吸道症状，结合流感病毒抗原或核酸检测阳性可确诊。',
    checks: ['流感病毒抗原检测', '流感病毒核酸检测', '血常规', '胸部X线'],
    treatment: '发病48小时内尽早使用神经氨酸酶抑制剂抗病毒治疗，辅以退热、补液等对症处理。',
    treatments: ['抗病毒治疗', '对症治疗', '隔离休息', '补液支持'],
    drugs: ['奥司他韦', '扎那米韦', '对乙酰氨基酚', '连花清瘟胶囊'],
    department: '感染科',
    prognosis: '多数患者1-2周内自愈，老年人及慢病患者可发展为重症肺炎。',
    population: '儿童、老年人、孕妇、慢性基础疾病患者',
    complications: ['病毒性肺炎', '继发细菌感染', '心肌炎', '呼吸衰竭'],
    differential: ['普通感冒', '新型冠状病毒感染', '支原体肺炎', '细菌性肺炎'],
    is_infectious: true,
    sources: [
      {
        source_type: 'guideline',
        pmid: '33720324',
        doi: '10.3760/cma.j.issn.1674-2397.2020.06.001',
        title: '流行性感冒诊疗方案(2020年版)',
        journal: '中华临床感染病杂志',
        year: 2020,
        authors: '国家卫生健康委员会',
        url: 'https://doi.org/10.3760/cma.j.issn.1674-2397.2020.06.001',
        authority: 'A级',
        retrieved_at: '2024-02-06'
      }
    ],
    updated_at: '2024-02-06'
  },
  {
    disease_id: 'D007',
    name: '脑卒中',
    alias: ['中风', '脑血管意外'],
    category1: '神经系统疾病',
    category2: '脑血管疾病',
    definition: '脑卒中是由于脑血管突然破裂或阻塞导致脑组织损伤的急性脑血管疾病，分为缺血性和出血性两类。',
    cause: '高血压、动脉粥样硬化、心房颤动、糖尿病、血脂异常、吸烟等为主要危险因素。',
    symptoms: ['突发偏瘫', '口角歪斜', '言语不清', '意识障碍', '剧烈头痛', '视物障碍'],
    diagnosis: '急诊头颅CT/MRI明确病变性质与部位，结合血管超声、CTA或DSA评估血管情况。',
    checks: ['头颅CT', '头颅MRI', '颈动脉超声', '脑血管CTA', '心电图'],
    treatment: '缺血性卒中在时间窗内行静脉溶栓或血管内取栓，出血性卒中控制血压并评估手术指征。',
    treatments: ['静脉溶栓', '血管内取栓', '抗血小板治疗', '康复训练'],
    drugs: ['阿替普酶', '阿司匹林', '氯吡格雷', '阿托伐他汀'],
    department: '神经内科',
    prognosis: '疗效与就诊时间高度相关，越早治疗预后越好，部分患者遗留功能障碍。',
    population: '高血压患者、老年人、心房颤动患者、吸烟人群',
    complications: ['脑水肿', '颅内压增高', '肺部感染', '下肢深静脉血栓'],
    differential: ['短暂性脑缺血发作', '低血糖昏迷', '癫痫发作', '颅内肿瘤'],
    is_infectious: false,
    sources: [
      {
        source_type: 'guideline',
        pmid: '33128434',
        doi: '10.3760/cma.j.issn.1006-7876.2018.09.002',
        title: '中国急性缺血性脑卒中诊治指南2018',
        journal: '中华神经科杂志',
        year: 2018,
        authors: '中华医学会神经病学分会',
        url: 'https://doi.org/10.3760/cma.j.issn.1006-7876.2018.09.002',
        authority: 'A级',
        retrieved_at: '2024-02-11'
      }
    ],
    updated_at: '2024-02-11'
  },
  {
    disease_id: 'D008',
    name: '慢性阻塞性肺疾病',
    alias: ['慢阻肺', 'COPD'],
    category1: '呼吸系统疾病',
    category2: '气道疾病',
    definition: '慢性阻塞性肺疾病是以持续气流受限为特征的常见慢性呼吸系统疾病，气流受限呈进行性发展。',
    cause: '长期吸烟是最重要的危险因素，此外还包括生物燃料烟雾暴露、职业粉尘接触和遗传因素。',
    symptoms: ['慢性咳嗽', '咳痰', '气短', '喘息', '胸闷', '活动耐力下降'],
    diagnosis: '吸入支气管舒张剂后FEV1/FVC<70%可确诊持续气流受限，并结合症状与急性加重史进行综合评估。',
    checks: ['肺功能检查', '胸部CT', '血气分析', '六分钟步行试验'],
    treatment: '戒烟、支气管舒张剂吸入治疗为基础，急性加重期加用糖皮质激素与抗感染治疗，必要时长期氧疗。',
    treatments: ['戒烟干预', '吸入支气管舒张剂', '肺康复训练', '长期家庭氧疗'],
    drugs: ['噻托溴铵', '沙美特罗替卡松', '布地奈德', '氨溴索'],
    department: '呼吸内科',
    prognosis: '戒烟与规范吸入治疗可延缓肺功能下降，反复急性加重者预后较差。',
    population: '长期吸烟者、粉尘作业人群、40岁以上人群',
    complications: ['慢性呼吸衰竭', '肺源性心脏病', '自发性气胸', '骨质疏松'],
    differential: ['支气管哮喘', '支气管扩张', '肺结核', '心力衰竭'],
    is_infectious: false,
    sources: [
      {
        source_type: 'guideline',
        pmid: '34488741',
        doi: '10.3760/cma.j.cn112147-20210125-00063',
        title: '慢性阻塞性肺疾病诊治指南(2021年修订版)',
        journal: '中华结核和呼吸杂志',
        year: 2021,
        authors: '中华医学会呼吸病学分会慢阻肺学组',
        url: 'https://doi.org/10.3760/cma.j.cn112147-20210125-00063',
        authority: 'A级',
        retrieved_at: '2024-02-14'
      }
    ],
    updated_at: '2024-02-14'
  }
]

export const hotKeywords = ['感冒', '高血压', '糖尿病', '冠心病', '肺炎', '胃炎']

export function searchDisease (q = '', page = 1, pageSize = 6) {
  const kw = (q || '').trim()
  let items = DISEASES
  if (kw) {
    const lower = kw.toLowerCase()
    items = DISEASES.filter((d) =>
      d.name.toLowerCase().includes(lower) ||
      d.alias.some((a) => String(a).toLowerCase().includes(lower)) ||
      d.category1.includes(kw) ||
      d.category2.includes(kw) ||
      d.symptoms.some((s) => s.includes(kw))
    )
    if (!items.length) {
      // 演示兜底：关键词无命中时返回最相关的两条
      items = DISEASES.slice(0, 2)
    }
  }
  const total = items.length
  const size = Math.max(1, Number(pageSize) || 6)
  const p = Math.max(1, Number(page) || 1)
  const start = (p - 1) * size
  const pageItems = items.slice(start, start + size).map((d) => ({
    disease_id: d.disease_id,
    name: d.name,
    category1: d.category1,
    category2: d.category2,
    symptoms: d.symptoms,
    treatments: d.treatments,
    population: d.population,
    is_infectious: d.is_infectious,
    has_detail: true
  }))
  return {
    keyword: kw,
    total,
    page: p,
    page_size: size,
    items: pageItems,
    hot_keywords: hotKeywords,
    message: kw ? `共找到 ${total} 条与「${kw}」相关的疾病信息（离线演示数据）` : `共 ${total} 条疾病信息（离线演示数据）`,
    searched_at: new Date().toISOString()
  }
}

export function diseaseDetail (diseaseId) {
  const d = DISEASES.find((x) => x.disease_id === diseaseId) || DISEASES[0]
  return JSON.parse(JSON.stringify(d))
}

/* ============================== 智能问答 ============================== */

export function askQuestion (payload = {}) {
  const question = (payload.question || '').trim() || '高血压的治疗方法是什么？'
  const sid = payload.session_id || 'demo-session'
  const isDept = /科|挂号|就诊/.test(question)
  const isDrug = /药|用药|服用/.test(question)
  const isSymptom = /症状|表现|征象/.test(question)

  let answer = ''
  let evidences = []
  let trace = []

  if (isDept) {
    answer =
      '根据医疗知识图谱中的实体关联，该疾病通常建议就诊于「心血管内科」，部分医院设有高血压专科门诊。\n' +
      '若出现急性胸痛、呼吸困难、意识障碍等急症表现，请立即前往急诊科就诊。\n' +
      '就诊前建议携带既往血压记录、正在服用的降压药物清单以及近期的血液生化与心电图检查结果，以便医生快速评估病情。'
    evidences = [
      {
        triple_id: 'KG-1',
        head: '高血压',
        head_type: '疾病',
        relation: 'belongs_to',
        relation_label: '所属科室',
        tail: '心血管内科',
        tail_type: '科室',
        confidence: 0.96,
        sources: [{ title: '中国高血压防治指南(2023年修订版)', journal: '中华心血管病杂志', year: 2023, pmid: '37130140', url: 'https://pubmed.ncbi.nlm.nih.gov/37130140/' }]
      },
      {
        triple_id: 'KG-2',
        head: '高血压',
        head_type: '疾病',
        relation: 'complicated_with',
        relation_label: '并发症',
        tail: '脑卒中',
        tail_type: '疾病',
        confidence: 0.88,
        sources: [{ title: 'Hypertension: a global perspective', journal: 'The Lancet', year: 2022, pmid: '36220816', url: 'https://pubmed.ncbi.nlm.nih.gov/36220816/' }]
      }
    ]
    trace = [
      { step: 1, action: 'intent', title: '意图识别', detail: '识别为「就诊科室咨询」，置信度 0.94', elapsed_ms: 46, payload: {} },
      { step: 2, action: 'entity_link', title: '实体链接', detail: '从问句中链接到疾病实体「高血压」(D001)', elapsed_ms: 38, payload: {} },
      { step: 3, action: 'kg_query', title: '图谱检索', detail: '沿 belongs_to 关系检索 2 跳邻域，命中 6 条三元组', elapsed_ms: 63, payload: {} },
      { step: 4, action: 'retrieve', title: '证据筛选', detail: '按置信度与来源权威性排序，保留 2 条核心证据', elapsed_ms: 27, payload: {} },
      { step: 5, action: 'llm', title: '答案生成', detail: '基于提示模板 qa_department_v2 生成回答', elapsed_ms: 612, payload: {} },
      { step: 6, action: 'guard', title: '安全校验', detail: '未检出违规诊疗建议，已附加免责声明', elapsed_ms: 21, payload: {} }
    ]
  } else if (isDrug) {
    answer =
      '常用降压药物包括钙通道阻滞剂（氨氯地平、硝苯地平）、血管紧张素受体拮抗剂（缬沙坦）、利尿剂（氢氯噻嗪）以及β受体阻滞剂（美托洛尔）。\n' +
      '用药应遵循小剂量起始、优先选用长效制剂、联合用药与个体化治疗的原则，并坚持长期规律服用。\n' +
      '请勿自行调整剂量或停药，具体方案需由执业医师根据血压水平、合并疾病与耐受情况制定。'
    evidences = [
      {
        triple_id: 'KG-1',
        head: '高血压',
        head_type: '疾病',
        relation: 'used_drug',
        relation_label: '常用药物',
        tail: '氨氯地平',
        tail_type: '药物',
        confidence: 0.93,
        sources: [{ title: '中国高血压防治指南(2023年修订版)', journal: '中华心血管病杂志', year: 2023, pmid: '37130140', url: 'https://pubmed.ncbi.nlm.nih.gov/37130140/' }]
      },
      {
        triple_id: 'KG-2',
        head: '高血压',
        head_type: '疾病',
        relation: 'treated_by',
        relation_label: '治疗方式',
        tail: '降压药物治疗',
        tail_type: '治疗方法',
        confidence: 0.9,
        sources: [{ title: '中国高血压防治指南(2023年修订版)', journal: '中华心血管病杂志', year: 2023, pmid: '37130140', url: 'https://pubmed.ncbi.nlm.nih.gov/37130140/' }]
      }
    ]
    trace = [
      { step: 1, action: 'intent', title: '意图识别', detail: '识别为「用药指导」，置信度 0.91', elapsed_ms: 41, payload: {} },
      { step: 2, action: 'entity_link', title: '实体链接', detail: '链接疾病实体「高血压」与药物类实体约束', elapsed_ms: 35, payload: {} },
      { step: 3, action: 'kg_query', title: '图谱检索', detail: '检索 used_drug / treated_by 关系，命中 11 条三元组', elapsed_ms: 58, payload: {} },
      { step: 4, action: 'retrieve', title: '证据筛选', detail: '按药物类别聚类去重，保留 2 条核心证据', elapsed_ms: 25, payload: {} },
      { step: 5, action: 'llm', title: '答案生成', detail: '基于提示模板 qa_drug_v3 生成回答', elapsed_ms: 688, payload: {} },
      { step: 6, action: 'guard', title: '安全校验', detail: '检出用药相关表述，已追加医嘱提示', elapsed_ms: 19, payload: {} }
    ]
  } else if (isSymptom) {
    answer =
      '高血压常见症状包括头晕、头痛、颈项板紧、心悸、耳鸣、视物模糊、失眠与乏力等。\n' +
      '需要特别注意的是，多数早期高血压并无明显症状，因此被称为「沉默的杀手」，仅凭症状判断并不可靠。\n' +
      '建议成年人定期测量血压：非同日三次诊室血压达到收缩压≥140mmHg和/或舒张压≥90mmHg即可考虑高血压诊断。'
    evidences = [
      {
        triple_id: 'KG-1',
        head: '高血压',
        head_type: '疾病',
        relation: 'has_symptom',
        relation_label: '症状',
        tail: '头晕',
        tail_type: '症状',
        confidence: 0.95,
        sources: [{ title: '中国高血压防治指南(2023年修订版)', journal: '中华心血管病杂志', year: 2023, pmid: '37130140', url: 'https://pubmed.ncbi.nlm.nih.gov/37130140/' }]
      },
      {
        triple_id: 'KG-2',
        head: '高血压',
        head_type: '疾病',
        relation: 'has_symptom',
        relation_label: '症状',
        tail: '颈项板紧',
        tail_type: '症状',
        confidence: 0.87,
        sources: [{ title: 'Hypertension: a global perspective', journal: 'The Lancet', year: 2022, pmid: '36220816', url: 'https://pubmed.ncbi.nlm.nih.gov/36220816/' }]
      }
    ]
    trace = [
      { step: 1, action: 'intent', title: '意图识别', detail: '识别为「症状查询」，置信度 0.95', elapsed_ms: 39, payload: {} },
      { step: 2, action: 'entity_link', title: '实体链接', detail: '链接疾病实体「高血压」(D001)', elapsed_ms: 33, payload: {} },
      { step: 3, action: 'kg_query', title: '图谱检索', detail: '检索 has_symptom 关系，命中 8 条三元组', elapsed_ms: 52, payload: {} },
      { step: 4, action: 'retrieve', title: '证据筛选', detail: '保留 2 条高置信度症状证据', elapsed_ms: 22, payload: {} },
      { step: 5, action: 'llm', title: '答案生成', detail: '基于提示模板 qa_symptom_v2 生成回答', elapsed_ms: 574, payload: {} },
      { step: 6, action: 'guard', title: '安全校验', detail: '通过，已附加免责声明', elapsed_ms: 18, payload: {} }
    ]
  } else {
    answer =
      '高血压的治疗以生活方式干预为基础，结合个体化药物治疗：\n' +
      '1）生活方式干预：限制钠盐摄入（每日＜5g）、控制体重、戒烟限酒、规律有氧运动、保持心理平衡；\n' +
      '2）药物治疗：常用钙通道阻滞剂、血管紧张素受体拮抗剂、利尿剂与β受体阻滞剂，多需联合用药；\n' +
      '3）长期管理：家庭自测血压并记录，定期复查血生化与心电图，评估靶器官损害。\n' +
      '血压控制目标一般为＜140/90mmHg，能耐受者可进一步降至＜130/80mmHg。'
    evidences = [
      {
        triple_id: 'KG-1',
        head: '高血压',
        head_type: '疾病',
        relation: 'treated_by',
        relation_label: '治疗方式',
        tail: '生活方式干预',
        tail_type: '治疗方法',
        confidence: 0.94,
        sources: [{ title: '中国高血压防治指南(2023年修订版)', journal: '中华心血管病杂志', year: 2023, pmid: '37130140', url: 'https://pubmed.ncbi.nlm.nih.gov/37130140/' }]
      },
      {
        triple_id: 'KG-2',
        head: '高血压',
        head_type: '疾病',
        relation: 'used_drug',
        relation_label: '常用药物',
        tail: '氨氯地平',
        tail_type: '药物',
        confidence: 0.91,
        sources: [{ title: '中国高血压防治指南(2023年修订版)', journal: '中华心血管病杂志', year: 2023, pmid: '37130140', url: 'https://pubmed.ncbi.nlm.nih.gov/37130140/' }]
      },
      {
        triple_id: 'KG-3',
        head: '高血压',
        head_type: '疾病',
        relation: 'needs_exam',
        relation_label: '检查项目',
        tail: '动态血压监测',
        tail_type: '检查项目',
        confidence: 0.86,
        sources: [{ title: '中国高血压防治指南(2023年修订版)', journal: '中华心血管病杂志', year: 2023, pmid: '37130140', url: 'https://pubmed.ncbi.nlm.nih.gov/37130140/' }]
      }
    ]
    trace = [
      { step: 1, action: 'intent', title: '意图识别', detail: '识别为「治疗方案咨询」，置信度 0.92', elapsed_ms: 43, payload: {} },
      { step: 2, action: 'entity_link', title: '实体链接', detail: '链接疾病实体「高血压」(D001)', elapsed_ms: 36, payload: {} },
      { step: 3, action: 'kg_query', title: '图谱检索', detail: '2 跳邻域检索，命中 24 条三元组', elapsed_ms: 71, payload: {} },
      { step: 4, action: 'retrieve', title: '证据筛选', detail: '语义重排后保留 3 条核心证据', elapsed_ms: 31, payload: {} },
      { step: 5, action: 'llm', title: '答案生成', detail: '基于提示模板 qa_treatment_v3 生成回答', elapsed_ms: 703, payload: {} },
      { step: 6, action: 'guard', title: '安全校验', detail: '通过，未检出违规内容', elapsed_ms: 20, payload: {} }
    ]
  }

  const answerHtml = answer
    .split('\n')
    .map((line, i) => {
      const marker = i === 0 ? '<sup class="kg-ref" data-ref="KG-1">[KG-1]</sup>' : ''
      return `<p>${line}${marker}</p>`
    })
    .join('')

  return {
    question,
    answer,
    answer_html: answerHtml,
    intent: { label: isDept ? 'department' : isDrug ? 'drug' : isSymptom ? 'symptom' : 'treatment', label_cn: isDept ? '就诊科室' : isDrug ? '用药指导' : isSymptom ? '症状查询' : '治疗方案', confidence: 0.93, method: 'rule+bert', scores: {} },
    entities: [{ text: '高血压', type: 'disease', label: '疾病', kg_id: 'D001', kg_name: '高血压', score: 0.97, method: 'dictionary', start: 0, end: 3 }],
    evidences,
    kg_context: evidences.map((e) => `${e.head} -[${e.relation}]-> ${e.tail}`),
    reasoning_trace: trace,
    confidence: 0.93,
    prompt_template_id: isDept ? 'qa_department_v2' : isDrug ? 'qa_drug_v3' : isSymptom ? 'qa_symptom_v2' : 'qa_treatment_v3',
    prompt_used: '你是严谨的医疗知识助手，请仅依据知识图谱证据作答，并附加免责声明。',
    guard_result: { passed: true, level: 'low', hits: [] },
    latency_ms: { intent: 43, entity_link: 36, kg_query: 71, semantic_parse_total: 79, retrieve: 31, llm: 703, guard: 20, total: 983 },
    llm_model: 'deepseek-chat (演示模式)',
    session_id: sid,
    disclaimer: '本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。',
    related_questions: [
      '成人呼吸窘迫综合征的症状有哪些？',
      '肺栓塞应该挂什么科？',
      '肺心病的治疗方法是什么？',
      '急性呼吸窘迫综合征的治疗费用大概多少？',
      '继发性肺动脉高压是否传染？'
    ]
  }
}

/* ============================== 数据分析 ============================== */

export const analyticsOverview = {
  metrics: [
    { key: 'total_entities', label: '医疗实体总数', value: 1120, suffix: '个', icon: 'Share', trend: 8.6 },
    { key: 'disease_categories', label: '疾病分类数', value: 26, suffix: '类', icon: 'Grid', trend: 3.2 },
    { key: 'infectious_diseases', label: '传染性疾病', value: 37, suffix: '种', icon: 'Warning', trend: -1.4 },
    { key: 'treatment_cycles', label: '治疗周期类型', value: 12, suffix: '种', icon: 'Calendar', trend: 5.1 }
  ],
  generated_at: new Date().toISOString(),
  data_source: '离线演示数据'
}

export const nodeTypePie = {
  chart_type: 'pie',
  title: '各类节点数量统计',
  categories: [],
  values: [],
  series: [
    { name: '症状', value: 356, color: '#31c48d' },
    { name: '药物', value: 214, color: '#E6A23C' },
    { name: '检查项目', value: 138, color: '#909399' },
    { name: '疾病', value: 128, color: '#409EFF' },
    { name: '治疗方法', value: 96, color: '#b37feb' },
    { name: '文献来源', value: 82, color: '#00BCD4' },
    { name: '易感人群', value: 64, color: '#F2C037' },
    { name: '科室', value: 42, color: '#F56C6C' }
  ],
  echarts_option: null,
  pyecharts_html: '',
  insight: '症状类节点数量最多，占比约 31.8%，说明知识图谱以症状-疾病关联为主要组织方式。'
}

export const categoryBar = {
  chart_type: 'bar',
  title: '一级分类下的疾病数量',
  categories: ['内科疾病', '呼吸系统疾病', '消化系统疾病', '神经系统疾病', '内分泌疾病', '感染性疾病', '骨科疾病', '皮肤科疾病'],
  x_axis: ['内科疾病', '呼吸系统疾病', '消化系统疾病', '神经系统疾病', '内分泌疾病', '感染性疾病', '骨科疾病', '皮肤科疾病'],
  values: [34, 22, 18, 15, 12, 11, 9, 7],
  series: [],
  echarts_option: null,
  pyecharts_html: '',
  insight: '内科疾病数量最多（34 种），呼吸系统疾病次之，与常见病谱分布基本一致。'
}

export const infectiousGauge = {
  chart_type: 'pie',
  title: '传染性疾病比例',
  categories: ['传染性疾病', '非传染性疾病'],
  values: [37, 91],
  series: [
    { name: '传染性疾病', value: 37, color: '#F56C6C' },
    { name: '非传染性疾病', value: 91, color: '#409EFF' }
  ],
  echarts_option: null,
  pyecharts_html: '',
  insight: '传染性疾病占全部疾病实体的约 28.9%，需重点关注呼吸道传染病的防控知识普及。'
}

export function predictRisk (form = {}) {
  const age = Number(form.age) || 45
  const bmi = Number(form.bmi) || 23
  const sbp = Number(form.systolic) || 125
  const dbp = Number(form.diastolic) || 80
  const glucose = Number(form.glucose) || 5.2
  const chol = Number(form.cholesterol) || 4.8
  const smoking = form.smoking === true || form.smoking === 'yes'
  const drinking = form.drinking === true || form.drinking === 'yes'
  const family = Array.isArray(form.family_history) ? form.family_history : []
  const symptoms = Array.isArray(form.symptoms) ? form.symptoms : []

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v))
  const bpRisk = clamp(((sbp - 110) / 70) * 100, 3, 96)
  const bmiRisk = clamp(((bmi - 21) / 14) * 100, 3, 95)
  const glucoseRisk = clamp(((glucose - 4.8) / 6) * 100, 3, 95)
  const cholRisk = clamp(((chol - 4.0) / 4) * 100, 3, 92)
  const ageRisk = clamp(((age - 30) / 45) * 100, 3, 95)
  const smokeRisk = smoking ? 78 : 12
  const drinkRisk = drinking ? 58 : 14
  const famRisk = family.length ? 68 : 18

  const w = (base, factors) => {
    const nums = factors.map((f) => f[0] * f[1])
    const sum = nums.reduce((a, b) => a + b, 0)
    return clamp(Math.round(base + sum), 2, 97)
  }

  const hypertension = w(10, [[0.42, bpRisk], [0.16, bmiRisk], [0.12, ageRisk], [0.12, smokeRisk], [0.1, famRisk], [0.08, glucoseRisk]])
  const diabetes = w(8, [[0.34, bmiRisk], [0.3, glucoseRisk], [0.14, ageRisk], [0.1, famRisk], [0.12, drinkRisk]])
  const coronary = w(7, [[0.28, cholRisk], [0.26, bpRisk], [0.18, smokeRisk], [0.14, ageRisk], [0.14, famRisk]])
  const fattyLiver = w(9, [[0.32, bmiRisk], [0.24, drinkRisk], [0.2, cholRisk], [0.14, glucoseRisk], [0.1, ageRisk]])
  const stroke = w(5, [[0.4, bpRisk], [0.18, ageRisk], [0.16, smokeRisk], [0.14, cholRisk], [0.12, glucoseRisk]])

  const levelOf = (p) => (p >= 75 ? '极高风险' : p >= 55 ? '高风险' : p >= 35 ? '中风险' : '低风险')

  const mk = (disease, diseaseId, percent, factors) => ({
    disease,
    disease_id: diseaseId,
    risk: Number((percent / 100).toFixed(3)),
    risk_percent: percent,
    level: levelOf(percent),
    top_factors: factors.map(([factor, contribution, direction]) => ({
      factor,
      contribution: Number(contribution.toFixed(1)),
      direction
    })),
    kg_evidence: [
      `${disease} -[has_symptom]-> ${symptoms[0] || '头晕'}`,
      `${disease} -[susceptible_to]-> ${family[0] ? '家族史人群' : '中老年人群'}`
    ]
  })

  const predictions = [
    mk('高血压', 'D001', hypertension, [
      ['收缩压', Math.min(42, bpRisk * 0.42), 'up'],
      ['BMI', Math.min(24, bmiRisk * 0.24), 'up'],
      ['吸烟史', smokeRisk * 0.14, 'up'],
      ['家族史', famRisk * 0.12, 'up'],
      ['年龄', Math.min(20, ageRisk * 0.2), 'up']
    ]),
    mk('2型糖尿病', 'D002', diabetes, [
      ['空腹血糖', Math.min(40, glucoseRisk * 0.4), 'up'],
      ['BMI', Math.min(30, bmiRisk * 0.3), 'up'],
      ['家族史', famRisk * 0.16, 'up'],
      ['饮酒史', drinkRisk * 0.14, 'up']
    ]),
    mk('冠心病', 'D003', coronary, [
      ['总胆固醇', Math.min(34, cholRisk * 0.34), 'up'],
      ['收缩压', Math.min(30, bpRisk * 0.3), 'up'],
      ['吸烟史', smokeRisk * 0.2, 'up'],
      ['年龄', Math.min(16, ageRisk * 0.16), 'up']
    ]),
    mk('非酒精性脂肪性肝病', 'D009', fattyLiver, [
      ['BMI', Math.min(36, bmiRisk * 0.36), 'up'],
      ['饮酒史', drinkRisk * 0.24, 'up'],
      ['总胆固醇', Math.min(22, cholRisk * 0.22), 'up'],
      ['空腹血糖', Math.min(18, glucoseRisk * 0.18), 'up']
    ]),
    mk('脑卒中', 'D007', stroke, [
      ['收缩压', Math.min(44, bpRisk * 0.44), 'up'],
      ['年龄', Math.min(20, ageRisk * 0.2), 'up'],
      ['吸烟史', smokeRisk * 0.18, 'up'],
      ['总胆固醇', Math.min(18, cholRisk * 0.18), 'up']
    ])
  ].sort((a, b) => b.risk_percent - a.risk_percent)

  const maxRisk = predictions[0].risk_percent
  const healthScore = clamp(Math.round(100 - maxRisk * 0.72 - (smoking ? 4 : 0) - (drinking ? 3 : 0)), 32, 98)
  const overallLevel = levelOf(maxRisk)

  return {
    predictions,
    overall_level: overallLevel,
    health_score: healthScore,
    model: {
      name: 'MKW-RiskNet',
      version: 'v1.2.0-demo',
      auc: 0.874,
      disease_coverage: 5,
      method: '逻辑回归 + 知识图谱规则增强',
      features_used: 12
    },
    summary: `综合评估结果显示整体风险等级为「${overallLevel}」，健康评分 ${healthScore} 分。` +
      `其中「${predictions[0].disease}」风险最高（${predictions[0].risk_percent}%），` +
      (smoking ? '吸烟是当前可干预的主要危险因素，建议尽早戒烟；' : '') +
      (bmi > 24 ? '体重指数偏高，建议控制体重并增加有氧运动；' : '体重指数处于合理区间，请继续保持；') +
      '建议定期监测血压、血糖与血脂，并于每年进行一次全面体检。',
    disclaimer: '风险预测结果仅供参考，不能替代执业医师诊断。',
    latency_ms: 168
  }
}

export function interventionPlan (payload = {}) {
  const bmi = Number(payload.bmi) || 23
  const smoking = payload.smoking === true || payload.smoking === 'yes'
  const plan = [
    {
      category: 'diet',
      category_name: '饮食建议',
      category_cn: '饮食建议',
      icon: 'Bowl',
      title: '低盐低脂均衡膳食',
      priority: 'high',
      items: [
        '每日食盐摄入量控制在 5 克以内，减少酱油、腌制品与加工肉类',
        '增加新鲜蔬菜（每日 300-500g）与全谷物摄入，保证膳食纤维充足',
        '限制饱和脂肪与反式脂肪，烹调以蒸、煮、炖为主',
        bmi > 24 ? '控制总热量摄入，每日减少约 300-500 千卡' : '保持三餐规律，避免暴饮暴食'
      ]
    },
    {
      category: 'exercise',
      category_name: '运动建议',
      category_cn: '运动建议',
      icon: 'Basketball',
      title: '规律中等强度有氧运动',
      priority: 'high',
      items: [
        '每周进行 5 次、每次 30 分钟的中等强度有氧运动（快走、慢跑、游泳）',
        '每周增加 2 次抗阻训练，每次 20 分钟，改善胰岛素敏感性',
        '运动前后做好热身与拉伸，避免清晨血压高峰期剧烈运动',
        '久坐人群每小时起身活动 3-5 分钟'
      ]
    },
    {
      category: 'examination',
      category_name: '体检建议',
      category_cn: '体检建议',
      icon: 'FirstAidKit',
      title: '定期专项筛查',
      priority: 'medium',
      items: [
        '每 3 个月监测一次血压与空腹血糖，建立个人健康档案',
        '每年检测一次血脂四项、糖化血红蛋白与肝肾功能',
        '每年进行一次心电图与颈动脉超声检查',
        '40 岁以上人群建议每 1-2 年进行一次胸部低剂量 CT 筛查'
      ]
    },
    {
      category: 'lifestyle',
      category_name: '生活方式',
      category_cn: '生活方式',
      icon: 'MoonNight',
      title: '戒烟限酒与规律作息',
      priority: 'high',
      items: [
        smoking ? '尽快戒烟，必要时寻求戒烟门诊与药物辅助支持' : '坚持不吸烟并远离二手烟环境',
        '限制酒精摄入，男性每日酒精量不超过 25 克，女性不超过 15 克',
        '保证每日 7-8 小时睡眠，避免长期熬夜',
        '学习压力管理技巧，保持情绪稳定与心理平衡'
      ]
    }
  ]
  return {
    plan,
    follow_up: '建议 3 个月后复查血压、血糖与血脂，根据指标变化动态调整干预方案；如出现胸闷、胸痛、肢体麻木等症状请及时就诊。',
    disclaimer: '本建议由演示模型生成，仅供参考，不能替代执业医师诊断与处方。'
  }
}
