<template>
  <div class="mkw-page graph-view">
    <section class="mkw-hero">
      <h1 class="mkw-hero__title">医疗知识图谱</h1>
      <p class="mkw-hero__subtitle">可视化医疗知识网络，直观展示医疗实体间的复杂关联关系。</p>
      <p class="mkw-hero__hint">提示：为保证浏览器性能和流畅体验，默认仅展示部分关键节点和关系。</p>
    </section>

    <!-- 搜索行 -->
    <section class="mkw-card graph-view__search">
      <el-input
        v-model="keyword"
        class="graph-view__input"
        size="large"
        clearable
        placeholder="请输入疾病名称、症状或实体名称进行关系图谱查询..."
        @keyup.enter="buildGraph"
      >
        <template #prefix>
          <el-icon><Search /></el-icon>
        </template>
      </el-input>
      <el-button class="mkw-btn-primary" size="large" :icon="Search" :loading="loading" @click="buildGraph">
        构建图谱
      </el-button>

      <div v-if="suggestions.length" class="graph-view__suggest">
        <span class="graph-view__suggest-label">实体推荐：</span>
        <span
          v-for="s in suggestions"
          :key="`${s.type}-${s.name}`"
          class="mkw-pill mkw-pill--blue graph-view__suggest-item"
          @click="buildGraph(s.name)"
        >
          {{ s.name }}
          <em>{{ s.label || s.type }}</em>
        </span>
      </div>
    </section>

    <!-- 统计瓦片 -->
    <section class="graph-view__stats">
      <div class="mkw-stat">
        <div class="mkw-stat__icon"><el-icon><Share /></el-icon></div>
        <div>
          <div class="mkw-stat__value">{{ displayStats.node_count }}</div>
          <div class="mkw-stat__label">节点数量</div>
        </div>
      </div>
      <div class="mkw-stat">
        <div class="mkw-stat__icon graph-view__stat-icon--green"><el-icon><Connection /></el-icon></div>
        <div>
          <div class="mkw-stat__value">{{ displayStats.link_count }}</div>
          <div class="mkw-stat__label">关系数量</div>
        </div>
      </div>
      <div class="mkw-stat">
        <div class="mkw-stat__icon graph-view__stat-icon--purple"><el-icon><Grid /></el-icon></div>
        <div>
          <div class="mkw-stat__value">{{ displayStats.type_count }}</div>
          <div class="mkw-stat__label">实体类型</div>
        </div>
      </div>
    </section>

    <!-- 图谱主体 -->
    <section class="graph-view__main">
      <div class="mkw-card graph-view__canvas-card">
        <div class="mkw-card__header">
          <el-icon class="mkw-card__title-icon"><Share /></el-icon>
          <span class="mkw-card__title">知识图谱可视化</span>
          <span v-if="graphMessage" class="graph-view__message">{{ graphMessage }}</span>
          <div class="mkw-card__actions">
            <el-tooltip content="全屏查看" placement="top">
              <el-button circle :icon="FullScreen" @click="toggleFullscreen" />
            </el-tooltip>
            <el-tooltip content="重置缩放" placement="top">
              <el-button circle :icon="Refresh" @click="resetZoom" />
            </el-tooltip>
            <el-tooltip content="下载图片" placement="top">
              <el-button circle :icon="Download" @click="downloadImage" />
            </el-tooltip>
          </div>
        </div>

        <div ref="canvasWrap" class="graph-view__canvas" :class="{ 'is-fullscreen': fullscreen }">
          <div v-if="loading" class="graph-view__loading">
            <el-icon class="is-loading" :size="26"><Loading /></el-icon>
            <p>正在构建知识图谱…</p>
          </div>
          <GraphCanvas
            ref="canvasRef"
            :nodes="graph.nodes"
            :links="graph.links"
            :color-map="store.colorMap"
            :rel-labels="REL_LABELS"
            :height="560"
            @node-click="openNode"
          />
        </div>
      </div>

      <!-- 右侧图例 -->
      <aside class="graph-view__legend">
        <div class="mkw-card graph-view__legend-card">
          <div class="mkw-card__header">
            <el-icon class="mkw-card__title-icon"><Grid /></el-icon>
            <span class="mkw-card__title">实体类型</span>
          </div>
          <ul class="graph-view__legend-list">
            <li v-for="t in store.legend" :key="t.type" class="graph-view__legend-item">
              <span class="graph-view__legend-dot" :style="{ background: t.color }"></span>
              <span class="graph-view__legend-label">{{ t.label }}</span>
              <span class="graph-view__legend-count">{{ t.count != null ? t.count : '-' }}</span>
            </li>
          </ul>

          <div class="graph-view__legend-detail">
            <div class="graph-view__legend-detail-title">数据概览</div>
            <div class="graph-view__legend-detail-row">
              <span>节点总量</span><b>{{ stats.total_nodes || displayStats.node_count }}</b>
            </div>
            <div class="graph-view__legend-detail-row">
              <span>关系总量</span><b>{{ stats.total_links || displayStats.link_count }}</b>
            </div>
            <div class="graph-view__legend-detail-row">
              <span>疾病实体</span><b>{{ stats.disease_count || '-' }}</b>
            </div>
            <div class="graph-view__legend-detail-row">
              <span>症状实体</span><b>{{ stats.symptom_count || '-' }}</b>
            </div>
            <div v-if="stats.data_source" class="graph-view__legend-source">数据来源：{{ stats.data_source }}</div>
          </div>
        </div>
      </aside>
    </section>

    <!-- 节点详情抽屉 -->
    <el-drawer v-model="drawerVisible" title="实体详情" size="420px" :with-header="true">
      <div v-if="current" class="node-detail">
        <div class="node-detail__head">
          <span class="node-detail__dot" :style="{ background: current.color || store.colorOf(current.type) }"></span>
          <h3 class="node-detail__name">{{ current.name }}</h3>
        </div>
        <div class="node-detail__tags">
          <span class="mkw-pill mkw-pill--blue">{{ current.label || store.labelOf(current.type) }}</span>
          <span v-if="current.category1" class="mkw-pill mkw-pill--green">{{ current.category1 }}</span>
          <span v-if="current.category2" class="mkw-pill mkw-pill--amber">{{ current.category2 }}</span>
          <span class="mkw-pill mkw-pill--gray">连接度 {{ current.degree || 0 }}</span>
        </div>

        <div class="node-detail__section">
          <div class="node-detail__section-title">属性信息</div>
          <ul v-if="propertyList.length" class="node-detail__props">
            <li v-for="p in propertyList" :key="p.key">
              <span class="node-detail__prop-key">{{ p.key }}</span>
              <span class="node-detail__prop-val">{{ p.value }}</span>
            </li>
          </ul>
          <div v-else class="node-detail__empty">暂无扩展属性</div>
        </div>

        <div class="node-detail__section">
          <div class="node-detail__section-title">关联关系（{{ relatedLinks.length }}）</div>
          <ul v-if="relatedLinks.length" class="node-detail__relations">
            <li v-for="(r, i) in relatedLinks" :key="i" class="node-detail__relation">
              <span class="node-detail__rel-rel">{{ r.rel }}</span>
              <span class="node-detail__rel-text">
                {{ r.head }} <em>—{{ r.relLabel }}→</em> {{ r.tail }}
              </span>
            </li>
          </ul>
          <div v-else class="node-detail__empty">暂无关联关系</div>
        </div>

        <div v-if="sources.length" class="node-detail__section">
          <div class="node-detail__section-title">来源文献（{{ sources.length }}）</div>
          <ul class="node-detail__sources">
            <li v-for="(s, i) in sources" :key="i">
              <el-icon><Document /></el-icon>
              <div class="node-detail__source-body">
                <div class="node-detail__source-title">{{ s.title }}</div>
                <div class="node-detail__source-meta">
                  <span v-if="s.journal">{{ s.journal }}</span>
                  <span v-if="s.year">· {{ s.year }}</span>
                  <a
                    v-if="s.url"
                    class="mkw-link"
                    :href="s.url"
                    target="_blank"
                    rel="noopener noreferrer"
                  >{{ s.pmid ? `PMID ${s.pmid}` : (s.doi ? `DOI ${s.doi}` : '查看原文') }}</a>
                </div>
              </div>
            </li>
          </ul>
        </div>

        <div class="node-detail__actions">
          <el-button class="mkw-btn-primary" :icon="Aim" @click="centerOn(current.name)">
            以此节点为中心
          </el-button>
          <el-button class="mkw-btn-ghost" :icon="CopyDocument" @click="copyName(current.name)">复制名称</el-button>
        </div>
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Search,
  Share,
  Connection,
  Grid,
  FullScreen,
  Refresh,
  Download,
  Loading,
  Document,
  Aim,
  CopyDocument
} from '@element-plus/icons-vue'
import GraphCanvas from '@/components/GraphCanvas.vue'
import { getGraphStats, getSubgraph, searchEntity } from '@/api'
import { useAppStore } from '@/store/app'

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

const store = useAppStore()
const canvasRef = ref(null)
const canvasWrap = ref(null)

const keyword = ref('高血压')
const loading = ref(false)
const fullscreen = ref(false)
const graphMessage = ref('')
const suggestions = ref([])
const drawerVisible = ref(false)
const current = ref(null)

const graph = reactive({ nodes: [], links: [] })
const stats = reactive({
  total_nodes: 0,
  total_links: 0,
  entity_type_count: 0,
  disease_count: 0,
  symptom_count: 0,
  data_source: ''
})
const subStats = reactive({ node_count: 0, link_count: 0, type_count: 0 })

const displayStats = computed(() => ({
  node_count: subStats.node_count || graph.nodes.length,
  link_count: subStats.link_count || graph.links.length,
  type_count: subStats.type_count || new Set(graph.nodes.map((n) => n.type)).size
}))

const propertyList = computed(() => {
  const c = current.value
  if (!c) return []
  const list = []
  const props = c.properties && typeof c.properties === 'object' ? c.properties : {}
  Object.keys(props).forEach((k) => {
    const v = props[k]
    if (v === null || v === undefined || v === '') return
    list.push({ key: k, value: Array.isArray(v) ? v.join('、') : String(v) })
  })
  if (c.category1) list.push({ key: '一级分类', value: c.category1 })
  if (c.category2) list.push({ key: '二级分类', value: c.category2 })
  if (c.type) list.push({ key: '实体类型', value: c.label || store.labelOf(c.type) })
  return list
})

const sources = computed(() => {
  const c = current.value
  if (!c) return []
  const raw = (c.properties && (c.properties.sources || c.properties.来源文献)) || c.sources
  return Array.isArray(raw) ? raw : []
})

const relatedLinks = computed(() => {
  const c = current.value
  if (!c) return []
  const id = c.id
  return graph.links
    .filter((l) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source
      const t = typeof l.target === 'object' ? l.target.id : l.target
      return s === id || t === id
    })
    .slice(0, 40)
    .map((l) => {
      const s = typeof l.source === 'object' ? l.source : graph.nodes.find((n) => n.id === l.source) || {}
      const t = typeof l.target === 'object' ? l.target : graph.nodes.find((n) => n.id === l.target) || {}
      const rel = l.rel || l.relation || 'related_to'
      return {
        rel,
        relLabel: REL_LABELS[rel] || l.label || rel,
        head: s.name || s.id,
        tail: t.name || t.id
      }
    })
})

async function loadStats () {
  const data = await getGraphStats()
  Object.assign(stats, {
    total_nodes: Number(data.total_nodes) || 0,
    total_links: Number(data.total_links) || 0,
    entity_type_count: Number(data.entity_type_count) || 0,
    disease_count: Number(data.disease_count) || 0,
    symptom_count: Number(data.symptom_count) || 0,
    data_source: data.data_source || ''
  })
  if (Array.isArray(data.entity_types) && data.entity_types.length) {
    store.legend = data.entity_types.map((t) => ({ ...t }))
  }
}

async function buildGraph (name) {
  const entity = typeof name === 'string' && name.trim() ? name.trim() : keyword.value.trim()
  if (!entity) {
    ElMessage.warning('请输入要查询的实体名称')
    return
  }
  keyword.value = entity
  loading.value = true
  try {
    const data = await getSubgraph(entity, 1, 200)
    graph.nodes = Array.isArray(data.nodes) ? data.nodes : []
    graph.links = Array.isArray(data.links) ? data.links : []
    subStats.node_count = Number(data.node_count) || graph.nodes.length
    subStats.link_count = Number(data.link_count) || graph.links.length
    subStats.type_count = Number(data.type_count) || new Set(graph.nodes.map((n) => n.type)).size
    graphMessage.value = data.message || ''
    if (!graph.nodes.length) {
      ElMessage.warning(`未找到与「${entity}」相关的图谱数据`)
    } else {
      ElMessage.success(`已构建「${entity}」知识子图：${subStats.node_count} 个节点 / ${subStats.link_count} 条关系`)
    }
  } catch (e) {
    ElMessage.error(e.friendlyMessage || '图谱构建失败')
  } finally {
    loading.value = false
  }
}

async function loadSuggestions (q) {
  if (!q || !q.trim()) {
    suggestions.value = []
    return
  }
  try {
    const data = await searchEntity(q.trim())
    suggestions.value = (data.items || []).slice(0, 6)
  } catch (e) {
    suggestions.value = []
  }
}

function openNode (node) {
  current.value = node
  drawerVisible.value = true
}

function centerOn (name) {
  drawerVisible.value = false
  buildGraph(name).then(() => loadSuggestions(name))
}

function resetZoom () {
  if (canvasRef.value) canvasRef.value.resetZoom()
}

function toggleFullscreen () {
  fullscreen.value = !fullscreen.value
  setTimeout(() => {
    if (canvasRef.value) canvasRef.value.resetZoom()
  }, 260)
}

function downloadImage () {
  if (!canvasRef.value) return
  const str = canvasRef.value.toSvgString('#ffffff')
  if (!str) {
    ElMessage.warning('暂无可导出的图谱内容')
    return
  }
  const blob = new Blob([str], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `医疗知识图谱-${keyword.value || 'graph'}.svg`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 1500)
  ElMessage.success('图谱已导出为 SVG 图片')
}

async function copyName (name) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(name)
      ElMessage.success('已复制实体名称')
      return
    }
    throw new Error('clipboard unavailable')
  } catch (e) {
    ElMessage.warning(`复制失败，实体名称为：${name}`)
  }
}

let suggestTimer = null
function onKeywordInput (val) {
  clearTimeout(suggestTimer)
  suggestTimer = setTimeout(() => loadSuggestions(val), 320)
}

onMounted(async () => {
  await Promise.all([
    loadStats().catch(() => {}),
    buildGraph('高血压')
  ])
  loadSuggestions('高血压')
})

// 输入时给出实体推荐
watch(keyword, (val) => onKeywordInput(val))

onBeforeUnmount(() => {
  clearTimeout(suggestTimer)
})
</script>

<style scoped lang="scss">
.graph-view__search {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 18px;
}

.graph-view__input {
  flex: 1 1 420px;
  min-width: 260px;
}

.graph-view__suggest {
  flex: 1 1 100%;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding-top: 4px;
}

.graph-view__suggest-label {
  font-size: 12px;
  color: var(--mkw-text-secondary);
}

.graph-view__suggest-item {
  cursor: pointer;
  transition: background 0.2s ease;
  display: inline-flex;
  align-items: center;
  gap: 4px;

  em {
    font-style: normal;
    font-size: 11px;
    color: #7ea6d8;
  }

  &:hover {
    background: #d8ebff;
  }
}

.graph-view__stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  margin-bottom: 18px;
}

.graph-view__stat-icon--green {
  background: linear-gradient(135deg, #6fe0ac 0%, #31c48d 100%);
  box-shadow: 0 4px 12px rgba(49, 196, 141, 0.26);
}

.graph-view__stat-icon--purple {
  background: linear-gradient(135deg, #c2a4f5 0%, #b37feb 100%);
  box-shadow: 0 4px 12px rgba(179, 127, 235, 0.26);
}

.graph-view__main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 18px;
  align-items: start;
}

.graph-view__canvas-card {
  padding: 18px 20px 20px;
}

.graph-view__message {
  margin-left: 10px;
  font-size: 12px;
  color: var(--mkw-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 320px;
}

.graph-view__canvas {
  position: relative;
  border-radius: 12px;
  overflow: hidden;
}

.graph-view__canvas.is-fullscreen {
  position: fixed;
  inset: 0;
  z-index: 2500;
  background: #eef4fd;
  padding: 16px;
  border-radius: 0;
}

.graph-view__loading {
  position: absolute;
  inset: 0;
  z-index: 5;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  background: rgba(255, 255, 255, 0.72);
  color: var(--mkw-primary);
  font-size: 13px;

  p {
    margin: 0;
  }
}

/* ---- 图例 ---- */
.graph-view__legend-card {
  position: sticky;
  top: 80px;
  padding: 18px 20px;
}

.graph-view__legend-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.graph-view__legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: var(--mkw-text-regular);
}

.graph-view__legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex: 0 0 auto;
  box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.9);
}

.graph-view__legend-label {
  flex: 1 1 auto;
}

.graph-view__legend-count {
  font-size: 12px;
  color: var(--mkw-text-secondary);
}

.graph-view__legend-detail {
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px dashed #e6ecf8;
}

.graph-view__legend-detail-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--mkw-text);
  margin-bottom: 10px;
}

.graph-view__legend-detail-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12.5px;
  color: var(--mkw-text-secondary);
  padding: 4px 0;

  b {
    color: var(--mkw-primary);
    font-size: 13px;
  }
}

.graph-view__legend-source {
  margin-top: 10px;
  font-size: 11px;
  color: var(--mkw-text-placeholder);
}

/* ---- 节点详情 ---- */
.node-detail__head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.node-detail__dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  flex: 0 0 auto;
}

.node-detail__name {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  color: var(--mkw-text);
}

.node-detail__tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 18px;
}

.node-detail__section {
  margin-bottom: 20px;
}

.node-detail__section-title {
  font-size: 14px;
  font-weight: 700;
  color: var(--mkw-text);
  margin-bottom: 10px;
  padding-left: 9px;
  border-left: 3px solid var(--mkw-primary);
  line-height: 1.2;
}

.node-detail__props {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;

  li {
    display: flex;
    gap: 10px;
    font-size: 13px;
    line-height: 1.6;
  }
}

.node-detail__prop-key {
  flex: 0 0 78px;
  color: var(--mkw-text-secondary);
}

.node-detail__prop-val {
  flex: 1 1 auto;
  color: var(--mkw-text-regular);
  word-break: break-all;
}

.node-detail__relations {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.node-detail__relation {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  border-radius: 8px;
  background: #f7faff;
}

.node-detail__rel-rel {
  font-size: 11px;
  color: #7ea6d8;
  font-family: Consolas, Monaco, monospace;
}

.node-detail__rel-text {
  font-size: 13px;
  color: var(--mkw-text-regular);

  em {
    font-style: normal;
    color: var(--mkw-primary);
  }
}

.node-detail__sources {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;

  li {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    font-size: 12.5px;
    color: var(--mkw-text-regular);
  }
}

.node-detail__source-body {
  min-width: 0;
}

.node-detail__source-title {
  line-height: 1.6;
  color: var(--mkw-text);
}

.node-detail__source-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 2px;
  font-size: 11.5px;
  color: var(--mkw-text-secondary);
}

.node-detail__empty {
  font-size: 12.5px;
  color: var(--mkw-text-placeholder);
}

.node-detail__actions {
  display: flex;
  gap: 10px;
  padding-top: 6px;
}

@media (max-width: 1280px) {
  .graph-view__main {
    grid-template-columns: minmax(0, 1fr);
  }

  .graph-view__legend-card {
    position: static;
  }
}

@media (max-width: 900px) {
  .graph-view__stats {
    grid-template-columns: 1fr;
  }
}
</style>
