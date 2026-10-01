<template>
  <div ref="wrapRef" class="graph-canvas">
    <svg ref="svgRef" class="graph-canvas__svg"></svg>

    <div v-if="!nodes.length" class="graph-canvas__empty">
      <el-icon :size="28"><Share /></el-icon>
      <p>暂无图谱数据，请输入实体名称后点击「构建图谱」</p>
    </div>

    <div v-else class="graph-canvas__tip">
      拖拽节点可调整布局 · 滚轮缩放 · 悬停高亮邻居 · 点击查看实体详情
    </div>
  </div>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as d3 from 'd3'
import { Share } from '@element-plus/icons-vue'

const props = defineProps({
  nodes: {
    type: Array,
    default: () => []
  },
  links: {
    type: Array,
    default: () => []
  },
  /** 类型 -> 颜色 */
  colorMap: {
    type: Object,
    default: () => ({})
  },
  /** 关系英文名 -> 中文名 */
  relLabels: {
    type: Object,
    default: () => ({})
  },
  height: {
    type: Number,
    default: 560
  }
})

const emit = defineEmits(['node-click', 'node-hover'])

const wrapRef = ref(null)
const svgRef = ref(null)

let svg = null
let gZoom = null
let gLinks = null
let gLinkLabels = null
let gNodes = null
let gLabels = null
let zoomBehavior = null
let simulation = null
let resizeObserver = null
let rafId = 0
let currentTransform = d3.zoomIdentity

const FALLBACK_COLOR = '#409EFF'

function pickColor (d) {
  const key = d && (d.type || d.label)
  return (d && d.color) || props.colorMap[key] || FALLBACK_COLOR
}

function nodeRadius (d) {
  const deg = Number(d.degree) || 1
  return Math.max(16, Math.min(24, 15 + Math.sqrt(deg) * 2.4))
}

function edgeId (v) {
  return typeof v === 'object' && v !== null ? v.id : v
}

/** 把后端 links 中可能是字符串的 source/target 解析为节点对象 */
function normalize () {
  const nodeMap = new Map()
  const nodes = (props.nodes || []).map((n, i) => {
    const id = n.id != null ? String(n.id) : `node-${i}`
    return {
      ...n,
      id,
      name: n.name || n.label || id,
      degree: Number(n.degree) || 1
    }
  })
  nodes.forEach((n) => nodeMap.set(n.id, n))

  const links = []
  ;(props.links || []).forEach((l) => {
    const s = nodeMap.get(String(edgeId(l.source)))
    const t = nodeMap.get(String(edgeId(l.target)))
    if (!s || !t) return
    links.push({
      ...l,
      source: s,
      target: t,
      rel: l.rel || l.relation || 'related_to',
      label: l.label || props.relLabels[l.rel || l.relation] || l.rel || l.relation || 'related_to'
    })
  })
  return { nodes, links }
}

function size () {
  const el = wrapRef.value
  const w = el && el.clientWidth ? el.clientWidth : 900
  const h = props.height || (el && el.clientHeight ? el.clientHeight : 560)
  return { w: Math.max(360, w), h: Math.max(360, h) }
}

function clearSvg () {
  if (svg) {
    svg.selectAll('*').remove()
  }
}

function render () {
  if (!svgRef.value || !wrapRef.value) return
  const { w, h } = size()
  const data = normalize()
  if (!data.nodes.length) {
    clearSvg()
    return
  }

  clearSvg()

  svg = d3
    .select(svgRef.value)
    .attr('viewBox', `0 0 ${w} ${h}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('width', '100%')
    .attr('height', '100%')

  const defs = svg.append('defs')

  // 箭头 marker（正向 + 悬停高亮）
  const marker = (id, color, opacity) =>
    defs
      .append('marker')
      .attr('id', id)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 26)
      .attr('refY', 0)
      .attr('markerWidth', 7)
      .attr('markerHeight', 7)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', color)
      .attr('opacity', opacity)

  marker('mkw-arrow', '#7fb2e8', 0.75)
  marker('mkw-arrow-active', '#2b8fe8', 1)

  // 节点投影
  const filter = defs
    .append('filter')
    .attr('id', 'mkw-node-shadow')
    .attr('x', '-50%')
    .attr('y', '-50%')
    .attr('width', '200%')
    .attr('height', '200%')
  filter
    .append('feDropShadow')
    .attr('dx', 0)
    .attr('dy', 2)
    .attr('stdDeviation', 3)
    .attr('flood-color', '#4a6bb0')
    .attr('flood-opacity', 0.35)

  gZoom = svg.append('g').attr('class', 'graph-root')

  zoomBehavior = d3
    .zoom()
    .scaleExtent([0.25, 4])
    .on('zoom', (event) => {
      currentTransform = event.transform
      gZoom.attr('transform', event.transform)
    })

  svg.call(zoomBehavior)
  svg.call(zoomBehavior.transform, d3.zoomIdentity)
  currentTransform = d3.zoomIdentity

  gLinks = gZoom.append('g').attr('class', 'graph-links')
  gLinkLabels = gZoom.append('g').attr('class', 'graph-link-labels')
  gNodes = gZoom.append('g').attr('class', 'graph-nodes')
  gLabels = gZoom.append('g').attr('class', 'graph-node-labels')

  // ---------- links ----------
  const linkSel = gLinks
    .selectAll('path')
    .data(data.links)
    .join('path')
    .attr('class', 'graph-link')
    .attr('fill', 'none')
    .attr('stroke', '#7fb2e8')
    .attr('stroke-opacity', 0.45)
    .attr('stroke-width', 1.6)
    .attr('marker-end', 'url(#mkw-arrow)')

  // ---------- link labels ----------
  gLinkLabels
    .selectAll('text')
    .data(data.links)
    .join('text')
    .attr('class', 'graph-link-label')
    .attr('font-size', 8.5)
    .attr('fill', '#8b95a8')
    .attr('text-anchor', 'middle')
    .attr('dy', -2)
    .style('paint-order', 'stroke')
    .style('stroke', '#ffffff')
    .style('stroke-width', '2.6px')
    .style('pointer-events', 'none')
    .text((d) => d.rel || '')

  // ---------- nodes ----------
  const nodeSel = gNodes
    .selectAll('circle')
    .data(data.nodes)
    .join('circle')
    .attr('class', 'graph-node')
    .attr('r', (d) => nodeRadius(d))
    .attr('fill', (d) => pickColor(d))
    .attr('stroke', '#ffffff')
    .attr('stroke-width', 3)
    .attr('filter', 'url(#mkw-node-shadow)')
    .style('cursor', 'pointer')

  // ---------- node labels ----------
  gLabels
    .selectAll('text')
    .data(data.nodes)
    .join('text')
    .attr('class', 'graph-node-label')
    .attr('text-anchor', 'middle')
    .attr('dy', (d) => nodeRadius(d) + 13)
    .attr('font-size', 11)
    .attr('font-weight', 700)
    .attr('fill', '#ffffff')
    .style('paint-order', 'stroke')
    .style('stroke', 'rgba(43,143,232,.55)')
    .style('stroke-width', '3px')
    .style('pointer-events', 'none')
    .text((d) => d.name)

  // ---------- 邻居表 ----------
  const neighbors = new Map()
  data.nodes.forEach((n) => neighbors.set(n.id, new Set([n.id])))
  data.links.forEach((l) => {
    neighbors.get(l.source.id).add(l.target.id)
    neighbors.get(l.target.id).add(l.source.id)
  })

  let pinnedId = null

  function applyHighlight (focusId) {
    const focus = focusId || pinnedId
    if (!focus) {
      nodeSel.attr('opacity', 1)
      linkSel.attr('stroke-opacity', 0.45).attr('marker-end', 'url(#mkw-arrow)')
      gLinkLabels.attr('opacity', 1)
      gLabels.attr('opacity', 1)
      return
    }
    const keep = neighbors.get(focus) || new Set([focus])
    nodeSel
      .attr('opacity', (d) => (keep.has(d.id) ? 1 : 0.16))
      .style('cursor', 'pointer')
    linkSel
      .attr('stroke-opacity', (l) =>
        l.source.id === focus || l.target.id === focus ? 0.95 : 0.08
      )
      .attr('marker-end', (l) =>
        l.source.id === focus || l.target.id === focus
          ? 'url(#mkw-arrow-active)'
          : 'url(#mkw-arrow)'
      )
    gLinkLabels.attr('opacity', (l) =>
      l.source.id === focus || l.target.id === focus ? 1 : 0.1
    )
    gLabels.attr('opacity', (d) => (keep.has(d.id) ? 1 : 0.2))
  }

  nodeSel
    .on('mouseenter', function (event, d) {
      applyHighlight(d.id)
      emit('node-hover', d)
    })
    .on('mouseleave', function () {
      applyHighlight(null)
      emit('node-hover', null)
    })
    .on('click', function (event, d) {
      event.stopPropagation()
      pinnedId = d.id
      applyHighlight(d.id)
      emit('node-click', d)
    })

  svg.on('click', () => {
    pinnedId = null
    applyHighlight(null)
  })

  // ---------- 拖拽 ----------
  const drag = d3
    .drag()
    .on('start', function (event, d) {
      if (!event.active) simulation.alphaTarget(0.28).restart()
      d.fx = d.x
      d.fy = d.y
    })
    .on('drag', function (event, d) {
      d.fx = event.x
      d.fy = event.y
    })
    .on('end', function (event, d) {
      if (!event.active) simulation.alphaTarget(0)
      d.fx = null
      d.fy = null
    })

  nodeSel.call(drag)

  // ---------- 力导向仿真 ----------
  simulation = d3
    .forceSimulation(data.nodes)
    .force(
      'link',
      d3
        .forceLink(data.links)
        .id((d) => d.id)
        .distance((l) => 110 + Math.min(70, (Number(l.source.degree) || 1) * 6))
        .strength(0.45)
    )
    .force('charge', d3.forceManyBody().strength(-460).distanceMax(520))
    .force('center', d3.forceCenter(w / 2, h / 2))
    .force('collide', d3.forceCollide().radius((d) => nodeRadius(d) + 18).iterations(2))
    .force('x', d3.forceX(w / 2).strength(0.045))
    .force('y', d3.forceY(h / 2).strength(0.045))
    .alpha(1)
    .alphaDecay(0.028)

  simulation.on('tick', () => {
    if (rafId) return
    rafId = requestAnimationFrame(() => {
      rafId = 0
      linkSel.attr('d', (l) => {
        const dx = l.target.x - l.source.x
        const dy = l.target.y - l.source.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const curve = Math.min(46, dist * 0.22)
        const mx = (l.source.x + l.target.x) / 2 - (dy / dist) * curve
        const my = (l.source.y + l.target.y) / 2 + (dx / dist) * curve
        return `M${l.source.x},${l.source.y} Q${mx},${my} ${l.target.x},${l.target.y}`
      })

      gLinkLabels
        .selectAll('text')
        .attr('x', (l) => (l.source.x + l.target.x) / 2)
        .attr('y', (l) => (l.source.y + l.target.y) / 2 - 4)

      nodeSel.attr('cx', (d) => d.x).attr('cy', (d) => d.y)
      gLabels
        .selectAll('text')
        .attr('x', (d) => d.x)
        .attr('y', (d) => d.y + nodeRadius(d) + 13)
    })
  })
}

/** 重置缩放 */
function resetZoom () {
  if (!svg || !zoomBehavior) return
  svg.transition().duration(360).call(zoomBehavior.transform, d3.zoomIdentity)
}

function zoomBy (factor) {
  if (!svg || !zoomBehavior) return
  svg.transition().duration(220).call(zoomBehavior.scaleBy, factor)
}

/** 导出 SVG 字符串（在组件内序列化，不依赖外部库） */
function toSvgString (bgColor = '#ffffff') {
  if (!svgRef.value) return ''
  const clone = svgRef.value.cloneNode(true)
  const { w, h } = size()
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
  clone.setAttribute('width', String(w))
  clone.setAttribute('height', String(h))
  clone.setAttribute('viewBox', `0 0 ${w} ${h}`)
  // 让导出内容自动居中适配
  const root = clone.querySelector('.graph-root')
  if (root) root.setAttribute('transform', '')
  if (bgColor) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    rect.setAttribute('x', '0')
    rect.setAttribute('y', '0')
    rect.setAttribute('width', String(w))
    rect.setAttribute('height', String(h))
    rect.setAttribute('fill', bgColor)
    clone.insertBefore(rect, clone.firstChild)
  }
  return new XMLSerializer().serializeToString(clone)
}

defineExpose({
  resetZoom,
  zoomBy,
  toSvgString,
  rerender: () => nextTick().then(render)
})

onMounted(() => {
  render()
  if (typeof ResizeObserver !== 'undefined' && wrapRef.value) {
    let t = 0
    resizeObserver = new ResizeObserver(() => {
      clearTimeout(t)
      t = setTimeout(() => {
        if (!simulation) return
        const { w, h } = size()
        if (svg) svg.attr('viewBox', `0 0 ${w} ${h}`)
        simulation.force('center', d3.forceCenter(w / 2, h / 2))
        simulation.force('x', d3.forceX(w / 2).strength(0.045))
        simulation.force('y', d3.forceY(h / 2).strength(0.045))
        simulation.alpha(0.35).restart()
      }, 180)
    })
    resizeObserver.observe(wrapRef.value)
  }
})

watch(
  () => [props.nodes, props.links],
  () => {
    nextTick().then(render)
  },
  { deep: false }
)

onBeforeUnmount(() => {
  if (rafId) cancelAnimationFrame(rafId)
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
  if (simulation) {
    simulation.stop()
    simulation = null
  }
  if (zoomBehavior && svg) {
    svg.on('.zoom', null)
  }
  clearSvg()
  svg = null
})
</script>

<style scoped lang="scss">
.graph-canvas {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 560px;
  border-radius: 12px;
  background: radial-gradient(circle at 50% 42%, #f7fbff 0%, #eef4fd 100%);
  overflow: hidden;
}

.graph-canvas__svg {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 560px;
  cursor: grab;
}

.graph-canvas__svg:active {
  cursor: grabbing;
}

.graph-canvas__empty {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--mkw-text-secondary);
  font-size: 13px;

  p {
    margin: 0;
  }
}

.graph-canvas__tip {
  position: absolute;
  left: 12px;
  bottom: 10px;
  font-size: 11px;
  color: #8b95a8;
  background: rgba(255, 255, 255, 0.82);
  border-radius: 8px;
  padding: 4px 10px;
  pointer-events: none;
}

:deep(.graph-node) {
  transition: opacity 0.18s ease;
}

:deep(.graph-link) {
  transition: stroke-opacity 0.18s ease;
}

:deep(.graph-node-label),
:deep(.graph-link-label) {
  user-select: none;
}
</style>
