import { ElMessage } from 'element-plus'
import request, { markOffline, markOnline } from './request'
import * as mock from './mock'

const OFFLINE_TIP = '[医数智答] 后端不可用，已切换到离线演示数据'
let offlineTipShown = false

function notifyOffline (err) {
  markOffline(err)
  // 控制台始终打印；页面只提示一次，避免刷屏
  console.warn(OFFLINE_TIP, err && (err.friendlyMessage || err.message || err))
  if (!offlineTipShown) {
    offlineTipShown = true
    try {
      ElMessage({
        message: '后端服务不可用，已切换到离线演示数据',
        type: 'warning',
        duration: 4000,
        showClose: true
      })
    } catch (e) {
      /* ElMessage 在极早期调用时可能不可用，忽略 */
    }
  }
}

function withTimeout (promise, ms) {
  let timer = null
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`请求超时（>${ms}ms）`)), ms)
  })
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) clearTimeout(timer)
  })
}

/**
 * 统一的"真实接口 + 离线兜底"包装器
 * @param {Function} realCall 返回 Promise 的真实请求
 * @param {Function} mockCall 返回离线演示数据的函数
 * @param {number} timeoutMs  真实请求超时时间，超时后立即使用离线数据
 */
export async function withFallback (realCall, mockCall, timeoutMs = 6000) {
  try {
    const data = await withTimeout(Promise.resolve().then(realCall), timeoutMs)
    if (data === undefined || data === null) {
      throw new Error('后端返回空数据')
    }
    markOnline()
    offlineTipShown = false
    return { data, offline: false }
  } catch (err) {
    notifyOffline(err)
    return { data: typeof mockCall === 'function' ? mockCall() : mockCall, offline: true }
  }
}

/** 需要弹错误提示的场景（如问答失败）手工调用 */
export function toastError (err, fallbackMsg = '操作失败，请稍后重试') {
  const msg = (err && (err.friendlyMessage || err.message)) || fallbackMsg
  try {
    ElMessage.error(msg)
  } catch (e) {
    console.error(msg)
  }
}

/* ============================== 知识图谱 ============================== */

export async function getGraphStats () {
  const r = await withFallback(() => request.get('/graph/stats'), () => mock.graphStats)
  const d = r.data || {}
  // 若后端未返回 entity_types，用本地图例补齐
  if (!Array.isArray(d.entity_types) || !d.entity_types.length) {
    d.entity_types = mock.entityTypes.map((t) => ({ ...t }))
  }
  return { ...d, _offline: r.offline }
}

export async function getEntityTypes () {
  const r = await withFallback(() => request.get('/graph/entity-types'), () => mock.entityTypes)
  const list = Array.isArray(r.data) ? r.data : (r.data && r.data.items) || []
  return (list.length ? list : mock.entityTypes).map((t) => ({ ...t }))
}

export async function getSubgraph (entity = '高血压', depth = 1, limit = 200) {
  const r = await withFallback(
    () => request.get('/graph/subgraph', { params: { entity, depth, limit } }),
    () => mock.buildSubgraph(entity, depth, limit)
  )
  const d = r.data || {}
  if (!Array.isArray(d.nodes)) d.nodes = []
  if (!Array.isArray(d.links)) d.links = []
  // 兼容后端线上字段：edges -> links
  if (!d.links.length && Array.isArray(d.edges)) d.links = d.edges
  return { ...d, _offline: r.offline }
}

export async function searchEntity (q) {
  const r = await withFallback(
    () => request.get('/graph/search', { params: { q } }),
    () => mock.graphSearch(q)
  )
  return r.data || { keyword: q, total: 0, items: [] }
}

/* ============================== 疾病查询 ============================== */

export async function searchDisease (q = '', page = 1, pageSize = 6) {
  const r = await withFallback(
    () => request.get('/disease/search', { params: { q, page, page_size: pageSize } }),
    () => mock.searchDisease(q, page, pageSize)
  )
  const d = r.data || {}
  if (!Array.isArray(d.items)) d.items = []
  if (!Array.isArray(d.hot_keywords) || !d.hot_keywords.length) d.hot_keywords = mock.hotKeywords
  return { ...d, total: Number(d.total) || d.items.length, page: Number(d.page) || page, page_size: Number(d.page_size) || pageSize, _offline: r.offline }
}

export async function getDiseaseDetail (diseaseId) {
  const r = await withFallback(
    () => request.get(`/disease/${encodeURIComponent(diseaseId)}`),
    () => mock.diseaseDetail(diseaseId)
  )
  return r.data || {}
}

export async function getHotKeywords () {
  const r = await withFallback(() => request.get('/disease/hot-keywords'), () => mock.hotKeywords)
  const list = Array.isArray(r.data) ? r.data : (r.data && r.data.hot_keywords) || []
  return list.length ? list : mock.hotKeywords
}

/* ============================== 智能问答 ============================== */

export async function askQuestion (payload) {
  const body = {
    question: (payload && payload.question) || '',
    session_id: (payload && payload.session_id) || 'demo-session',
    top_k: (payload && payload.top_k) != null ? payload.top_k : 12,
    max_hops: (payload && payload.max_hops) != null ? payload.max_hops : 2,
    explain: payload && payload.explain !== false,
    use_llm: payload && payload.use_llm !== false
  }
  const r = await withFallback(() => request.post('/qa/ask', body), () => mock.askQuestion(body), 60000)
  const d = r.data || {}
  if (!Array.isArray(d.evidences)) d.evidences = []
  if (!Array.isArray(d.reasoning_trace)) d.reasoning_trace = []
  if (!Array.isArray(d.related_questions)) d.related_questions = []
  return { ...d, _offline: r.offline }
}

/* ============================== 数据分析 ============================== */

export async function getOverview () {
  const r = await withFallback(() => request.get('/analytics/overview'), () => mock.analyticsOverview)
  const d = r.data || {}
  if (!Array.isArray(d.metrics) || !d.metrics.length) d.metrics = mock.analyticsOverview.metrics
  return { ...d, _offline: r.offline }
}

export async function getNodeTypePie () {
  const r = await withFallback(() => request.get('/analytics/node-type-pie'), () => mock.nodeTypePie)
  return r.data || {}
}

export async function getCategoryBar () {
  const r = await withFallback(() => request.get('/analytics/category-bar'), () => mock.categoryBar)
  return r.data || {}
}

export async function getInfectiousGauge () {
  const r = await withFallback(() => request.get('/analytics/infectious-gauge'), () => mock.infectiousGauge)
  return r.data || {}
}

export async function predictRisk (form) {
  const r = await withFallback(
    () => request.post('/analytics/risk/predict', form),
    () => mock.predictRisk(form)
  )
  const d = r.data || {}
  if (!Array.isArray(d.predictions)) d.predictions = []
  return { ...d, _offline: r.offline }
}

export async function getInterventionPlan (form) {
  const r = await withFallback(
    () => request.post('/analytics/risk/intervene', form),
    () => mock.interventionPlan(form)
  )
  const d = r.data || {}
  if (!Array.isArray(d.plan)) d.plan = []
  return { ...d, _offline: r.offline }
}

export default {
  withFallback,
  toastError,
  getGraphStats,
  getEntityTypes,
  getSubgraph,
  searchEntity,
  searchDisease,
  getDiseaseDetail,
  getHotKeywords,
  askQuestion,
  getOverview,
  getNodeTypePie,
  getCategoryBar,
  getInfectiousGauge,
  predictRisk,
  getInterventionPlan
}
