<template>
  <div ref="chartRef" class="echart" :style="{ height: `${height}px` }"></div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  /** ECharts option */
  option: {
    type: Object,
    default: () => ({})
  },
  height: {
    type: Number,
    default: 300
  }
})

const emit = defineEmits(['chart-ready', 'chart-click'])

const chartRef = ref(null)
let chart = null
let resizeObserver = null

function render () {
  if (!chartRef.value) return
  if (!chart) {
    chart = echarts.init(chartRef.value)
    chart.on('click', (params) => emit('chart-click', params))
    emit('chart-ready', chart)
  }
  const opt = JSON.parse(JSON.stringify(props.option || {}))
  chart.setOption(opt, true)
}

function resize () {
  if (chart) chart.resize()
}

onMounted(async () => {
  await nextTick()
  render()
  window.addEventListener('resize', resize)
  if (typeof ResizeObserver !== 'undefined' && chartRef.value) {
    resizeObserver = new ResizeObserver(() => resize())
    resizeObserver.observe(chartRef.value)
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
  if (chart) {
    chart.dispose()
    chart = null
  }
})

watch(
  () => props.option,
  async () => {
    await nextTick()
    render()
  },
  { deep: true }
)

defineExpose({
  getChart: () => chart,
  resize,
  /** 获取 PNG dataURL，用于下载图片 */
  getDataURL: () => (chart ? chart.getDataURL({ pixelRatio: 2, backgroundColor: '#ffffff' }) : '')
})
</script>

<style scoped lang="scss">
.echart {
  width: 100%;
  min-height: 180px;
}
</style>
