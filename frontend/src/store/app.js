import { defineStore } from 'pinia'
import { getGraphStats, getEntityTypes, getOverview } from '@/api'

const SESSION_KEY = 'mkw_session_id'

function randomSessionId () {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
  let s = ''
  for (let i = 0; i < 16; i += 1) {
    s += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return `mkw-${Date.now().toString(36)}-${s}`
}

/** 8 类实体默认图例（后端未返回时使用） */
const DEFAULT_LEGEND = [
  { type: 'disease', label: '疾病', color: '#409EFF', count: 0 },
  { type: 'symptom', label: '症状', color: '#31c48d', count: 0 },
  { type: 'department', label: '科室', color: '#F56C6C', count: 0 },
  { type: 'drug', label: '药物', color: '#E6A23C', count: 0 },
  { type: 'treatment', label: '治疗方法', color: '#b37feb', count: 0 },
  { type: 'examination', label: '检查项目', color: '#909399', count: 0 },
  { type: 'population', label: '易感人群', color: '#F2C037', count: 0 },
  { type: 'literature', label: '文献来源', color: '#00BCD4', count: 0 }
]

export const useAppStore = defineStore('app', {
  state: () => ({
    /** 会话 id：持久化在 localStorage，保证刷新后上下文连续 */
    sessionId: '',
    /** 侧边栏当前高亮路由 */
    activeMenu: '/',
    /** 知识图谱统计 */
    graphStats: {
      total_nodes: 0,
      total_links: 0,
      entity_type_count: 0,
      entity_types: []
    },
    /** 实体类型图例 */
    legend: DEFAULT_LEGEND.map((t) => ({ ...t })),
    /** 数据分析概览指标 */
    overviewMetrics: [],
    /** 后端是否处于离线演示模式 */
    offline: false,
    loading: false,
    /** 移动端抽屉导航开关（≤900px 时侧边栏改为抽屉式） */
    mobileNavOpen: false
  }),

  getters: {
    /** 图例：类型 -> 颜色 */
    colorMap (state) {
      const map = {}
      state.legend.forEach((t) => {
        map[t.type] = t.color
        map[t.label] = t.color
      })
      return map
    },
    colorOf (state) {
      return (typeOrLabel) => {
        const hit = state.legend.find(
          (t) => t.type === typeOrLabel || t.label === typeOrLabel
        )
        return (hit && hit.color) || '#409EFF'
      }
    },
    labelOf (state) {
      return (typeOrLabel) => {
        const hit = state.legend.find(
          (t) => t.type === typeOrLabel || t.label === typeOrLabel
        )
        return (hit && hit.label) || typeOrLabel || '未知'
      }
    },
    entityTypeCount (state) {
      return Number(state.graphStats.entity_type_count) || state.legend.length
    }
  },

  actions: {
    /** 初始化会话 id（localStorage 随机字符串） */
    initSession () {
      let sid = ''
      try {
        sid = localStorage.getItem(SESSION_KEY) || ''
      } catch (e) {
        sid = ''
      }
      if (!sid) {
        sid = randomSessionId()
        try {
          localStorage.setItem(SESSION_KEY, sid)
        } catch (e) {
          /* 隐私模式下写入可能失败，忽略 */
        }
      }
      this.sessionId = sid
      return sid
    },

    resetSession () {
      const sid = randomSessionId()
      this.sessionId = sid
      try {
        localStorage.setItem(SESSION_KEY, sid)
      } catch (e) {
        /* ignore */
      }
      return sid
    },

    setActiveMenu (path) {
      this.activeMenu = path
    },

    toggleMobileNav () {
      this.mobileNavOpen = !this.mobileNavOpen
    },

    closeMobileNav () {
      this.mobileNavOpen = false
    },

    async fetchGraphStats () {
      const data = await getGraphStats()
      this.graphStats = {
        total_nodes: Number(data.total_nodes) || 0,
        total_links: Number(data.total_links) || 0,
        entity_type_count: Number(data.entity_type_count) || 0,
        entity_types: Array.isArray(data.entity_types) ? data.entity_types : [],
        disease_count: Number(data.disease_count) || 0,
        symptom_count: Number(data.symptom_count) || 0,
        drug_count: Number(data.drug_count) || 0,
        department_count: Number(data.department_count) || 0,
        treatment_count: Number(data.treatment_count) || 0,
        source_count: Number(data.source_count) || 0,
        data_source: data.data_source || ''
      }
      if (data._offline) this.offline = true
      if (Array.isArray(data.entity_types) && data.entity_types.length) {
        this.legend = data.entity_types.map((t) => ({ ...t }))
      }
      return this.graphStats
    },

    async fetchLegend () {
      const list = await getEntityTypes()
      if (Array.isArray(list) && list.length) {
        this.legend = list.map((t) => ({ ...t }))
      }
      return this.legend
    },

    async fetchOverview () {
      const data = await getOverview()
      this.overviewMetrics = Array.isArray(data.metrics) ? data.metrics : []
      if (data._offline) this.offline = true
      return this.overviewMetrics
    }
  }
})

export default useAppStore
