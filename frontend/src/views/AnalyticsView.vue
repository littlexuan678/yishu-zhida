<template>
  <div class="mkw-page analytics-view">
    <section class="mkw-hero">
      <h1 class="mkw-hero__title">数据分析中心</h1>
      <p class="mkw-hero__subtitle">挖掘医疗数据价值，提供多维度统计分析和可视化展示</p>
    </section>

    <!-- 指标瓦片 -->
    <section class="analytics-view__metrics">
      <div v-for="(m, i) in metrics" :key="m.key || i" class="mkw-stat">
        <div class="mkw-stat__icon">
          <el-icon><component :is="metricIcon(m)" /></el-icon>
        </div>
        <div class="analytics-view__metric-body">
          <div class="mkw-stat__value">
            {{ formatValue(m.value) }}<small>{{ m.suffix || '' }}</small>
          </div>
          <div class="mkw-stat__label">{{ m.label }}</div>
        </div>
      </div>
    </section>

    <!-- 图表行 -->
    <section class="analytics-view__charts">
      <div v-for="card in chartCards" :key="card.key" class="mkw-card analytics-chart">
        <div class="mkw-card__header">
          <el-icon class="mkw-card__title-icon"><component :is="card.icon" /></el-icon>
          <span class="mkw-card__title">{{ card.title }}</span>
          <div class="mkw-card__actions">
            <el-tooltip content="下载图片" placement="top">
              <el-button circle size="small" :icon="Download" @click="downloadChart(card.key)" />
            </el-tooltip>
          </div>
        </div>
        <div v-loading="card.loading" class="analytics-chart__body">
          <EChart
            :ref="(el) => setChartRef(card.key, el)"
            :option="card.option"
            :height="300"
          />
        </div>
        <p v-if="card.insight" class="analytics-chart__insight">
          <el-icon><InfoFilled /></el-icon>
          {{ card.insight }}
        </p>
      </div>
    </section>

    <!-- 风险预警 -->
    <section id="risk" class="mkw-card analytics-view__risk">
      <div class="mkw-card__header">
        <el-icon class="mkw-card__title-icon"><WarningFilled /></el-icon>
        <span class="mkw-card__title">疾病风险预警</span>
        <div class="mkw-card__actions">
          <span v-if="riskModel" class="analytics-view__model">
            模型 {{ riskModel.name }} {{ riskModel.version }}（AUC {{ riskModel.auc }}）
          </span>
        </div>
      </div>

      <div class="analytics-view__risk-grid">
        <!-- 表单 -->
        <div class="analytics-view__form-wrap">
          <el-form
            :model="form"
            label-width="92px"
            label-position="left"
            class="analytics-view__form"
          >
            <el-form-item label="年龄">
              <el-input-number v-model="form.age" :min="1" :max="120" controls-position="right" />
            </el-form-item>
            <el-form-item label="性别">
              <el-select v-model="form.gender" placeholder="请选择">
                <el-option label="男" value="male" />
                <el-option label="女" value="female" />
              </el-select>
            </el-form-item>
            <el-form-item label="BMI">
              <el-input-number v-model="form.bmi" :min="10" :max="45" :step="0.1" :precision="1" controls-position="right" />
            </el-form-item>

            <el-form-item label="收缩压">
              <el-input-number v-model="form.systolic" :min="70" :max="230" controls-position="right" />
            </el-form-item>
            <el-form-item label="舒张压">
              <el-input-number v-model="form.diastolic" :min="40" :max="150" controls-position="right" />
            </el-form-item>
            <el-form-item label="空腹血糖">
              <el-input-number v-model="form.glucose" :min="2" :max="25" :step="0.1" :precision="1" controls-position="right" />
            </el-form-item>

            <el-form-item label="总胆固醇">
              <el-input-number v-model="form.cholesterol" :min="1" :max="15" :step="0.1" :precision="1" controls-position="right" />
            </el-form-item>
            <el-form-item label="是否吸烟">
              <el-radio-group v-model="form.smoking">
                <el-radio-button :value="true">是</el-radio-button>
                <el-radio-button :value="false">否</el-radio-button>
              </el-radio-group>
            </el-form-item>
            <el-form-item label="是否饮酒">
              <el-radio-group v-model="form.drinking">
                <el-radio-button :value="true">是</el-radio-button>
                <el-radio-button :value="false">否</el-radio-button>
              </el-radio-group>
            </el-form-item>

            <el-form-item label="运动习惯">
              <el-select v-model="form.exercise" placeholder="请选择">
                <el-option label="几乎不运动" value="none" />
                <el-option label="每周1-2次" value="low" />
                <el-option label="每周3-4次" value="medium" />
                <el-option label="每周5次以上" value="high" />
              </el-select>
            </el-form-item>
            <el-form-item label="家族史">
              <el-select
                v-model="form.family_history"
                multiple
                collapse-tags
                collapse-tags-tooltip
                placeholder="可多选"
              >
                <el-option v-for="f in FAMILY_OPTIONS" :key="f" :label="f" :value="f" />
              </el-select>
            </el-form-item>
            <el-form-item label="当前症状">
              <el-select
                v-model="form.symptoms"
                multiple
                collapse-tags
                collapse-tags-tooltip
                placeholder="可多选"
              >
                <el-option v-for="s in SYMPTOM_OPTIONS" :key="s" :label="s" :value="s" />
              </el-select>
            </el-form-item>
          </el-form>

          <div class="analytics-view__form-actions">
            <el-button class="mkw-btn-primary" size="large" :icon="Aim" :loading="riskLoading" @click="runRisk">
              开始风险评估
            </el-button>
            <el-button class="mkw-btn-ghost" size="large" :icon="RefreshLeft" @click="resetForm">
              重置
            </el-button>
          </div>
        </div>

        <!-- 结果 -->
        <div class="analytics-view__result">
          <div v-if="!riskResult && !riskLoading" class="analytics-view__result-empty">
            <el-icon :size="30"><DataAnalysis /></el-icon>
            <p>填写左侧健康指标后点击「开始风险评估」，即可获得疾病风险预测结果</p>
          </div>

          <div v-else-if="riskLoading" class="analytics-view__result-empty">
            <el-icon class="is-loading" :size="26"><Loading /></el-icon>
            <p>正在计算风险评分…</p>
          </div>

          <template v-else>
            <div class="risk-summary">
              <div class="risk-summary__score">
                <div class="risk-summary__score-value">{{ riskResult.health_score }}</div>
                <div class="risk-summary__score-label">综合健康评分</div>
                <el-progress
                  class="risk-summary__score-bar"
                  :percentage="Number(riskResult.health_score) || 0"
                  :stroke-width="10"
                  :show-text="false"
                  :color="scoreColor"
                />
              </div>
              <div class="risk-summary__level">
                <span class="risk-summary__level-label">整体风险等级</span>
                <span class="mkw-pill risk-level-pill" :class="levelClass(riskResult.overall_level)">
                  {{ riskResult.overall_level || '未知' }}
                </span>
              </div>
            </div>

            <p v-if="riskResult.summary" class="risk-summary__text">{{ riskResult.summary }}</p>

            <div class="risk-list">
              <div v-for="(p, i) in riskResult.predictions || []" :key="p.disease_id || i" class="risk-item">
                <div class="risk-item__head">
                  <span class="risk-item__name">{{ p.disease }}</span>
                  <span class="risk-item__percent">{{ p.risk_percent }}%</span>
                  <span class="mkw-pill risk-level-pill" :class="levelClass(p.level)">{{ p.level }}</span>
                </div>
                <el-progress
                  :percentage="Number(p.risk_percent) || 0"
                  :stroke-width="9"
                  :show-text="false"
                  :color="levelColor(p.level)"
                />
                <el-collapse v-if="(p.top_factors || []).length" class="risk-item__factors">
                  <el-collapse-item :name="`f-${i}`">
                    <template #title>
                      <span class="risk-item__factor-title">
                        <el-icon><TrendCharts /></el-icon>
                        主要影响因素
                        <em>{{ p.top_factors.length }} 项</em>
                      </span>
                    </template>
                    <div v-for="(f, fi) in p.top_factors" :key="fi" class="factor-row">
                      <span class="factor-row__name">
                        {{ f.factor }}
                        <em v-if="f.direction === 'up'">↑</em>
                        <em v-else-if="f.direction === 'down'">↓</em>
                      </span>
                      <el-progress
                        class="factor-row__bar"
                        :percentage="Number(f.contribution) || 0"
                        :stroke-width="8"
                        :show-text="false"
                        color="#38bdf8"
                      />
                      <span class="factor-row__value">{{ Number(f.contribution).toFixed(1) }}%</span>
                    </div>
                    <div v-if="(p.kg_evidence || []).length" class="risk-item__kg">
                      <span class="risk-item__kg-label">图谱证据：</span>
                      <span v-for="(k, ki) in p.kg_evidence" :key="ki" class="mkw-pill mkw-pill--gray">{{ k }}</span>
                    </div>
                  </el-collapse-item>
                </el-collapse>
              </div>
            </div>

            <!-- 干预建议 -->
            <div class="intervene">
              <div class="intervene__title">
                <el-icon><FirstAidKit /></el-icon>
                个性化干预建议
              </div>
              <div v-if="interveneLoading" class="intervene__loading">
                <el-icon class="is-loading"><Loading /></el-icon>
                正在生成干预方案…
              </div>
              <template v-else>
                <div v-if="interveneBlocks.length" class="intervene__grid">
                  <div v-for="(b, i) in interveneBlocks" :key="i" class="intervene__block">
                    <div class="intervene__block-head">
                      <span class="intervene__block-icon">
                        <el-icon><component :is="interveneIcon(b)" /></el-icon>
                      </span>
                      <span class="intervene__block-title">{{ blockTitle(b) }}</span>
                    </div>
                    <ul class="intervene__list">
                      <li v-for="(it, ii) in b.items || []" :key="ii">{{ it }}</li>
                    </ul>
                  </div>
                </div>
                <p v-else class="intervene__fallback">{{ interveneSummary }}</p>
                <div v-if="interveneFollowUp" class="intervene__follow">
                  <el-icon><Bell /></el-icon>
                  {{ interveneFollowUp }}
                </div>
              </template>
            </div>

            <DisclaimerBar
              inline
              text="⚠️ 风险预测结果仅供参考，不能替代执业医师诊断。"
            />
          </template>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Download,
  DataAnalysis,
  WarningFilled,
  InfoFilled,
  Aim,
  RefreshLeft,
  Loading,
  TrendCharts,
  FirstAidKit,
  Bell,
  Grid,
  Calendar,
  PieChart,
  Histogram,
  Bowl,
  Basketball,
  MoonNight,
  Share
} from '@element-plus/icons-vue'
import EChart from '@/components/EChart.vue'
import DisclaimerBar from '@/components/DisclaimerBar.vue'
import {
  getCategoryBar,
  getInfectiousGauge,
  getNodeTypePie,
  getOverview,
  getInterventionPlan,
  predictRisk
} from '@/api'

const FAMILY_OPTIONS = ['高血压', '糖尿病', '冠心病', '脑卒中', '肿瘤', '无家族史']
const SYMPTOM_OPTIONS = ['头晕', '头痛', '胸闷', '胸痛', '气促', '心悸', '多饮多尿', '乏力', '视物模糊', '肢体麻木', '咳嗽', '腹胀']

const FALLBACK_METRICS = [
  { key: 'total_entities', label: '医疗实体总数', value: 1120, suffix: '个', icon: 'Share' },
  { key: 'disease_categories', label: '疾病分类数', value: 26, suffix: '类', icon: 'Grid' },
  { key: 'infectious_diseases', label: '传染性疾病', value: 37, suffix: '种', icon: 'Warning' },
  { key: 'treatment_cycles', label: '治疗周期类型', value: 12, suffix: '种', icon: 'Calendar' }
]

const ICON_MAP = {
  Share,
  Grid,
  Warning: WarningFilled,
  Calendar,
  PieChart,
  Histogram,
  DataAnalysis
}

const metrics = ref([...FALLBACK_METRICS])
const chartRefs = reactive({})
const riskLoading = ref(false)
const riskResult = ref(null)
const riskModel = ref(null)
const interveneLoading = ref(false)
const interveneBlocks = ref([])
const interveneFollowUp = ref('')
const interveneSummary = ref('')

const form = reactive(defaultForm())

const chartCards = reactive([
  {
    key: 'pie',
    title: '各类节点数量统计',
    icon: PieChart,
    loading: false,
    insight: '',
    option: {}
  },
  {
    key: 'bar',
    title: '一级分类下的疾病数量',
    icon: Histogram,
    loading: false,
    insight: '',
    option: {}
  },
  {
    key: 'gauge',
    title: '传染性疾病比例',
    icon: PieChart,
    loading: false,
    insight: '',
    option: {}
  }
])

function defaultForm () {
  return {
    age: 45,
    gender: 'male',
    bmi: 24.5,
    systolic: 138,
    diastolic: 88,
    glucose: 5.6,
    cholesterol: 5.2,
    smoking: false,
    drinking: false,
    exercise: 'low',
    family_history: [],
    symptoms: ['头晕']
  }
}

const scoreColor = computed(() => {
  const s = Number(riskResult.value && riskResult.value.health_score) || 0
  if (s >= 80) return '#31a05f'
  if (s >= 60) return '#2b8fe8'
  if (s >= 40) return '#e6a23c'
  return '#f56c6c'
})

function metricIcon (m) {
  return ICON_MAP[m.icon] || DataAnalysis
}

function formatValue (v) {
  const n = Number(v)
  if (Number.isNaN(n)) return v
  return n.toLocaleString('zh-CN')
}

function setChartRef (key, el) {
  chartRefs[key] = el
}

function levelClass (level) {
  const l = String(level || '')
  if (l.includes('极高')) return 'risk-pill--extreme'
  if (l.includes('高')) return 'risk-pill--high'
  if (l.includes('中')) return 'risk-pill--mid'
  return 'risk-pill--low'
}

function levelColor (level) {
  const l = String(level || '')
  if (l.includes('极高')) return '#8b1a1a'
  if (l.includes('高')) return '#f56c6c'
  if (l.includes('中')) return '#e6a23c'
  return '#31a05f'
}

function blockTitle (b) {
  return b.category_name || b.category_cn || b.title || b.category
}

function interveneIcon (b) {
  const key = String(b.icon || b.category || '')
  if (key.includes('Bowl') || key === 'diet') return Bowl
  if (key.includes('Basketball') || key === 'exercise') return Basketball
  if (key.includes('FirstAid') || key === 'examination') return FirstAidKit
  if (key.includes('Moon') || key === 'lifestyle') return MoonNight
  return FirstAidKit
}

/* ---------------- 图表 option 构建 ---------------- */

function buildPieOption (data) {
  if (data && data.echarts_option && Object.keys(data.echarts_option).length) {
    return data.echarts_option
  }
  const series = Array.isArray(data.series) && data.series.length
    ? data.series
    : (data.categories || []).map((name, i) => ({
      name,
      value: Number((data.values || [])[i]) || 0
    }))
  const total = series.reduce((a, b) => a + (Number(b.value) || 0), 0) || 1
  return {
    tooltip: {
      trigger: 'item',
      formatter: (p) => `${p.name}<br/>${p.value} 个 (${((p.value / total) * 100).toFixed(2)}%)`
    },
    legend: {
      orient: 'vertical',
      left: 6,
      top: 'middle',
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { fontSize: 12, color: '#606266' }
    },
    series: [
      {
        name: data.title || '节点数量',
        type: 'pie',
        radius: ['42%', '66%'],
        center: ['64%', '52%'],
        avoidLabelOverlap: true,
        itemStyle: { borderColor: '#fff', borderWidth: 2, borderRadius: 4 },
        label: {
          show: true,
          formatter: '{b} {d}%',
          fontSize: 11,
          color: '#606266'
        },
        labelLine: { length: 8, length2: 8 },
        data: series.map((s) => ({
          name: s.name,
          value: s.value,
          itemStyle: { color: s.color }
        }))
      }
    ]
  }
}

function buildBarOption (data) {
  if (data && data.echarts_option && Object.keys(data.echarts_option).length) {
    return data.echarts_option
  }
  const cats = (Array.isArray(data.categories) && data.categories.length
    ? data.categories
    : data.x_axis) || []
  const values = Array.isArray(data.values) ? data.values : []
  return {
    grid: { left: 8, right: 16, top: 24, bottom: 6, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: {
      type: 'category',
      data: cats,
      axisLabel: { fontSize: 11, color: '#606266', interval: 0, rotate: 24 },
      axisLine: { lineStyle: { color: '#e4ebf7' } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      name: '疾病数量',
      nameTextStyle: { fontSize: 11, color: '#909399' },
      axisLabel: { fontSize: 11, color: '#909399' },
      splitLine: { lineStyle: { color: '#f0f4fc' } }
    },
    series: [
      {
        type: 'bar',
        barWidth: '46%',
        data: values,
        itemStyle: {
          borderRadius: [8, 8, 0, 0],
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: '#38bdf8' },
              { offset: 1, color: '#2b8fe8' }
            ]
          }
        },
        label: { show: true, position: 'top', fontSize: 11, color: '#2b8fe8', fontWeight: 600 }
      }
    ]
  }
}

function buildGaugeOption (data) {
  if (data && data.echarts_option && Object.keys(data.echarts_option).length) {
    return data.echarts_option
  }
  const series = Array.isArray(data.series) && data.series.length
    ? data.series
    : (data.categories || []).map((name, i) => ({
      name,
      value: Number((data.values || [])[i]) || 0
    }))
  const total = series.reduce((a, b) => a + (Number(b.value) || 0), 0) || 1
  return {
    tooltip: {
      trigger: 'item',
      formatter: (p) => `${p.name}<br/>${p.value} 种 (${((p.value / total) * 100).toFixed(2)}%)`
    },
    legend: {
      orient: 'vertical',
      left: 6,
      top: 'middle',
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { fontSize: 12, color: '#606266' }
    },
    series: [
      {
        name: data.title || '传染性疾病比例',
        type: 'pie',
        radius: ['48%', '70%'],
        center: ['64%', '52%'],
        itemStyle: { borderColor: '#fff', borderWidth: 2 },
        label: {
          show: true,
          formatter: '{b}\n{d}%',
          fontSize: 11,
          color: '#606266'
        },
        labelLine: { length: 8, length2: 8 },
        data: series.map((s) => ({
          name: s.name,
          value: s.value,
          itemStyle: { color: s.color }
        }))
      }
    ]
  }
}

/* ---------------- 数据加载 ---------------- */

async function loadMetrics () {
  try {
    const data = await getOverview()
    if (Array.isArray(data.metrics) && data.metrics.length) {
      metrics.value = data.metrics.slice(0, 4)
    }
  } catch (e) {
    metrics.value = [...FALLBACK_METRICS]
  }
}

async function loadCharts () {
  const tasks = [
    { key: 'pie', call: getNodeTypePie, builder: buildPieOption },
    { key: 'bar', call: getCategoryBar, builder: buildBarOption },
    { key: 'gauge', call: getInfectiousGauge, builder: buildGaugeOption }
  ]
  await Promise.all(
    tasks.map(async (t) => {
      const card = chartCards.find((c) => c.key === t.key)
      if (!card) return
      card.loading = true
      try {
        const data = await t.call()
        card.option = t.builder(data || {})
        card.insight = (data && data.insight) || ''
      } catch (e) {
        card.insight = '数据加载失败'
      } finally {
        card.loading = false
      }
    })
  )
}

function downloadChart (key) {
  const inst = chartRefs[key]
  const url = inst && typeof inst.getDataURL === 'function' ? inst.getDataURL() : ''
  if (!url) {
    ElMessage.warning('图表尚未渲染完成')
    return
  }
  const card = chartCards.find((c) => c.key === key)
  const a = document.createElement('a')
  a.href = url
  a.download = `${(card && card.title) || '图表'}.png`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  ElMessage.success('图表已下载')
}

/* ---------------- 风险评估 ---------------- */

function payloadOfForm () {
  return {
    age: Number(form.age),
    gender: form.gender,
    bmi: Number(form.bmi),
    systolic: Number(form.systolic),
    diastolic: Number(form.diastolic),
    glucose: Number(form.glucose),
    cholesterol: Number(form.cholesterol),
    smoking: !!form.smoking,
    drinking: !!form.drinking,
    exercise: form.exercise,
    family_history: [...form.family_history],
    symptoms: [...form.symptoms]
  }
}

async function runRisk () {
  const payload = payloadOfForm()
  if (payload.systolic <= payload.diastolic) {
    ElMessage.warning('收缩压应大于舒张压，请检查输入')
    return
  }
  riskLoading.value = true
  riskResult.value = null
  interveneBlocks.value = []
  interveneFollowUp.value = ''
  interveneSummary.value = ''
  try {
    const data = await predictRisk(payload)
    riskResult.value = data
    riskModel.value = data.model || null
    await loadIntervention(payload, data)
    ElMessage.success('风险评估完成')
  } catch (e) {
    ElMessage.error(e.friendlyMessage || '风险评估失败，请稍后重试')
  } finally {
    riskLoading.value = false
  }
}

async function loadIntervention (payload, riskData) {
  interveneLoading.value = true
  interveneSummary.value =
    (riskData && riskData.summary) ||
    '建议保持低盐低脂饮食、规律有氧运动、戒烟限酒并定期体检，3 个月后复查血压、血糖与血脂。'
  try {
    const data = await getInterventionPlan({ ...payload, predictions: (riskData && riskData.predictions) || [] })
    const plan = Array.isArray(data.plan) ? data.plan : []
    interveneBlocks.value = plan
    interveneFollowUp.value = data.follow_up || ''
    if (!plan.length) interveneBlocks.value = []
  } catch (e) {
    // 干预接口 404 / 不可用：退化为仅展示 summary
    interveneBlocks.value = []
    interveneFollowUp.value = ''
  } finally {
    interveneLoading.value = false
  }
}

function resetForm () {
  Object.assign(form, defaultForm())
  riskResult.value = null
  riskModel.value = null
  interveneBlocks.value = []
  interveneFollowUp.value = ''
  interveneSummary.value = ''
  ElMessage.info('已重置风险评估表单')
}

onMounted(() => {
  loadMetrics()
  loadCharts()
})
</script>

<style scoped lang="scss">
.analytics-view__metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 18px;
  margin-bottom: 18px;
}

.analytics-view__metric-body {
  min-width: 0;
}

.mkw-stat__value small {
  font-size: 13px;
  font-weight: 600;
  margin-left: 2px;
}

.analytics-view__charts {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  margin-bottom: 18px;
}

.analytics-chart {
  padding: 18px 20px;
}

.analytics-chart__body {
  min-height: 300px;
}

.analytics-chart__insight {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  margin: 10px 0 0;
  padding-top: 10px;
  border-top: 1px dashed #eef2fb;
  font-size: 12px;
  line-height: 1.7;
  color: var(--mkw-text-secondary);
}

/* ---- 风险预警 ---- */
.analytics-view__model {
  font-size: 12px;
  color: var(--mkw-text-secondary);
}

.analytics-view__risk-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr);
  gap: 22px;
  align-items: start;
}

.analytics-view__form {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0 14px;

  :deep(.el-form-item) {
    margin-bottom: 14px;
  }

  :deep(.el-input-number),
  :deep(.el-select) {
    width: 100%;
  }
}

.analytics-view__form-actions {
  display: flex;
  gap: 12px;
  padding-top: 4px;
}

.analytics-view__result {
  border-radius: 12px;
  background: #f8fbff;
  border: 1px solid #e8effb;
  padding: 16px 18px;
  min-height: 300px;
}

.analytics-view__result-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  height: 100%;
  min-height: 260px;
  color: var(--mkw-text-secondary);
  font-size: 13px;
  text-align: center;

  p {
    margin: 0;
    max-width: 320px;
    line-height: 1.8;
  }
}

.risk-summary {
  display: flex;
  align-items: center;
  gap: 22px;
  padding-bottom: 14px;
  border-bottom: 1px dashed #e2ecfa;
}

.risk-summary__score {
  flex: 0 0 160px;
}

.risk-summary__score-value {
  font-size: 40px;
  font-weight: 800;
  color: var(--mkw-primary);
  line-height: 1.05;
}

.risk-summary__score-label {
  font-size: 12.5px;
  color: var(--mkw-text-secondary);
  margin: 2px 0 8px;
}

.risk-summary__score-bar {
  width: 100%;
}

.risk-summary__level {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-start;
}

.risk-summary__level-label {
  font-size: 12.5px;
  color: var(--mkw-text-secondary);
}

.risk-level-pill {
  font-weight: 700;
  padding: 3px 14px;
}

.risk-pill--low {
  background: #e9f9ef;
  color: #31a05f;
}

.risk-pill--mid {
  background: #fdf5e6;
  color: #d78d16;
}

.risk-pill--high {
  background: #fdeeee;
  color: #e05a5a;
}

.risk-pill--extreme {
  background: #8b1a1a;
  color: #ffffff;
}

.risk-summary__text {
  margin: 12px 0 0;
  font-size: 12.5px;
  line-height: 1.8;
  color: var(--mkw-text-regular);
}

.risk-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 14px;
}

.risk-item {
  background: #ffffff;
  border: 1px solid #e8effb;
  border-radius: 10px;
  padding: 12px 14px;
}

.risk-item__head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.risk-item__name {
  font-size: 14px;
  font-weight: 700;
  color: var(--mkw-text);
  flex: 1 1 auto;
  min-width: 0;
}

.risk-item__percent {
  font-size: 15px;
  font-weight: 800;
  color: var(--mkw-primary);
}

.risk-item__factors {
  margin-top: 8px;
}

.risk-item__factor-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12.5px;

  em {
    font-style: normal;
    color: var(--mkw-text-secondary);
    font-size: 11.5px;
  }
}

.factor-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 3px 0;
}

.factor-row__name {
  flex: 0 0 88px;
  font-size: 12.5px;
  color: var(--mkw-text-regular);

  em {
    font-style: normal;
    color: #f56c6c;
    margin-left: 2px;
  }
}

.factor-row__bar {
  flex: 1 1 auto;
  min-width: 0;
}

.factor-row__value {
  flex: 0 0 48px;
  text-align: right;
  font-size: 12px;
  color: var(--mkw-text-secondary);
}

.risk-item__kg {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed #eef2fb;
}

.risk-item__kg-label {
  font-size: 12px;
  color: var(--mkw-text-secondary);
}

/* ---- 干预建议 ---- */
.intervene {
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px dashed #e2ecfa;
}

.intervene__title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 700;
  color: var(--mkw-text);
  margin-bottom: 12px;
}

.intervene__loading {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12.5px;
  color: var(--mkw-text-secondary);
}

.intervene__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.intervene__block {
  background: #ffffff;
  border: 1px solid #e8effb;
  border-radius: 10px;
  padding: 12px 14px;
}

.intervene__block-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.intervene__block-icon {
  width: 26px;
  height: 26px;
  border-radius: 8px;
  background: #e8f3ff;
  color: var(--mkw-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
}

.intervene__block-title {
  font-size: 13.5px;
  font-weight: 700;
  color: var(--mkw-text);
}

.intervene__list {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 5px;

  li {
    font-size: 12.5px;
    line-height: 1.7;
    color: var(--mkw-text-regular);
  }
}

.intervene__fallback {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.8;
  color: var(--mkw-text-regular);
}

.intervene__follow {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 10px;
  background: #f2f7ff;
  font-size: 12.5px;
  line-height: 1.7;
  color: var(--mkw-text-regular);
}

@media (max-width: 1400px) {
  .analytics-view__charts {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .analytics-view__risk-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 1100px) {
  .analytics-view__metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .analytics-view__form {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .analytics-view__metrics,
  .analytics-view__charts,
  .analytics-view__form,
  .intervene__grid {
    grid-template-columns: 1fr;
  }
}
</style>
