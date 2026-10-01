<template>
  <div class="mkw-page home-view">
    <section class="home-welcome">
      <h1 class="home-welcome__title">欢迎朱朱</h1>
      <p class="home-welcome__subtitle">智愈医典 · 医疗大数据知识问答系统</p>
    </section>

    <section class="mkw-hero">
      <h1 class="mkw-hero__title">核心功能</h1>
      <p class="mkw-hero__subtitle">
        基于医疗知识图谱与大语言模型，提供知识可视化、疾病检索、智能问答与数据分析的一站式服务
      </p>
    </section>

    <!-- 三大核心功能 -->
    <section class="home-view__grid home-view__grid--3">
      <article
        v-for="item in coreFeatures"
        :key="item.title"
        class="mkw-card mkw-card--hover home-feature"
        @click="go(item.path)"
      >
        <div class="mkw-circle-icon">
          <el-icon><component :is="item.icon" /></el-icon>
        </div>
        <h3 class="home-feature__title">{{ item.title }}</h3>
        <p class="home-feature__desc">{{ item.desc }}</p>
        <span class="home-feature__go">
          立即体验
          <el-icon><ArrowRight /></el-icon>
        </span>
      </article>
    </section>

    <!-- 技术特色 -->
    <h2 class="mkw-section-title">技术特色</h2>
    <section class="home-view__grid home-view__grid--3">
      <article v-for="tech in techFeatures" :key="tech.title" class="mkw-card home-tech">
        <div class="home-tech__emoji">{{ tech.emoji }}</div>
        <div class="home-tech__body">
          <h4 class="home-tech__title">{{ tech.title }}</h4>
          <p class="home-tech__desc">{{ tech.desc }}</p>
        </div>
      </article>
    </section>

    <!-- 数据分析 & 健康预警 -->
    <section class="home-view__grid home-view__grid--2">
      <article class="mkw-card mkw-card--hover home-banner" @click="go('/analytics')">
        <div class="mkw-circle-icon mkw-circle-icon--sm">
          <el-icon><DataAnalysis /></el-icon>
        </div>
        <div class="home-banner__body">
          <h4 class="home-banner__title">数据分析</h4>
          <p class="home-banner__desc">
            多维度统计医疗实体分布，挖掘疾病分类结构与传染性疾病比例等数据价值
          </p>
        </div>
        <el-icon class="home-banner__arrow"><ArrowRight /></el-icon>
      </article>

      <article class="mkw-card mkw-card--hover home-banner" @click="go('/analytics#risk')">
        <div class="mkw-circle-icon mkw-circle-icon--sm home-banner__icon--warn">
          <el-icon><WarningFilled /></el-icon>
        </div>
        <div class="home-banner__body">
          <h4 class="home-banner__title">健康预警</h4>
          <p class="home-banner__desc">
            输入体检指标即可获得疾病风险预测、风险等级与个性化干预建议
          </p>
        </div>
        <el-icon class="home-banner__arrow"><ArrowRight /></el-icon>
      </article>
    </section>

    <!-- 数据统计条 -->
    <section class="home-view__stats">
      <div v-for="(m, i) in statMetrics" :key="m.key || i" class="mkw-stat">
        <div class="mkw-stat__icon">
          <el-icon><component :is="iconOf(m)" /></el-icon>
        </div>
        <div class="mkw-stat__body">
          <div class="mkw-stat__value">
            {{ formatValue(m.value) }}<small>{{ m.suffix || '' }}</small>
          </div>
          <div class="mkw-stat__label">{{ m.label }}</div>
        </div>
        <span v-if="m.trend != null" class="home-stat__trend" :class="Number(m.trend) >= 0 ? 'is-up' : 'is-down'">
          <el-icon><component :is="Number(m.trend) >= 0 ? CaretTop : CaretBottom" /></el-icon>
          {{ Math.abs(Number(m.trend)).toFixed(1) }}%
        </span>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  Share,
  Search,
  ChatDotRound,
  DataAnalysis,
  WarningFilled,
  ArrowRight,
  Grid,
  Calendar,
  TrendCharts,
  CaretTop,
  CaretBottom
} from '@element-plus/icons-vue'
import { useAppStore } from '@/store/app'

const router = useRouter()
const store = useAppStore()

const coreFeatures = [
  {
    title: '知识图谱',
    path: '/graph',
    icon: Share,
    desc: '可视化医疗知识网络，直观展示疾病、症状、治疗方法等医疗实体间的关联关系'
  },
  {
    title: '疾病查询',
    path: '/disease',
    icon: Search,
    desc: '快速检索疾病信息，获取详细的症状、治疗方案、预后等全面医疗知识'
  },
  {
    title: '智能问答',
    path: '/chat',
    icon: ChatDotRound,
    desc: '专业的医疗知识问答服务，为您提供准确的医疗咨询和建议'
  }
]

const techFeatures = [
  { emoji: '🤖', title: '智能问答', desc: '采用先进的BERT模型进行自然语言理解' },
  { emoji: '🕸', title: '知识图谱', desc: '构建完整的医疗知识网络结构' },
  { emoji: '🎯', title: '精准匹配', desc: '多层次语义理解和智能匹配' }
]

const FALLBACK_METRICS = [
  { key: 'total_entities', label: '医疗实体总数', value: 1120, suffix: '个', icon: 'Share', trend: 8.6 },
  { key: 'disease_categories', label: '疾病分类数', value: 26, suffix: '类', icon: 'Grid', trend: 3.2 },
  { key: 'infectious_diseases', label: '传染性疾病', value: 37, suffix: '种', icon: 'Warning', trend: -1.4 },
  { key: 'treatment_cycles', label: '治疗周期类型', value: 12, suffix: '种', icon: 'Calendar', trend: 5.1 }
]

const statMetrics = computed(() => {
  const list = store.overviewMetrics
  return Array.isArray(list) && list.length ? list.slice(0, 4) : FALLBACK_METRICS
})

const ICON_MAP = {
  Share,
  Grid,
  Warning: WarningFilled,
  Calendar,
  TrendCharts,
  DataAnalysis
}

function iconOf (m) {
  return ICON_MAP[m.icon] || DataAnalysis
}

function formatValue (v) {
  const n = Number(v)
  if (Number.isNaN(n)) return v
  return n.toLocaleString('zh-CN')
}

function go (path) {
  if (path.includes('#')) {
    const [p, hash] = path.split('#')
    router.push(p).then(() => {
      setTimeout(() => {
        const el = document.getElementById(hash)
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 120)
    })
    return
  }
  router.push(path)
}

onMounted(() => {
  if (!store.overviewMetrics.length) {
    store.fetchOverview().catch(() => {})
  }
})
</script>

<style scoped lang="scss">
/* ---- 欢迎横幅 ---- */
.home-welcome {
  position: relative;
  text-align: center;
  padding: 52px 20px 44px;
  margin-bottom: 26px;
  border-radius: 18px;
  background: linear-gradient(135deg, #1d6fd1 0%, #2f9e8f 100%);
  box-shadow: 0 10px 30px rgba(29, 111, 209, 0.25);
  overflow: hidden;
}

.home-welcome::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.18), transparent 45%),
    radial-gradient(circle at 85% 80%, rgba(255, 255, 255, 0.12), transparent 40%);
  pointer-events: none;
}

.home-welcome__title {
  position: relative;
  margin: 0;
  font-size: 56px;
  font-weight: 800;
  letter-spacing: 10px;
  color: #ffffff;
  text-shadow: 0 3px 12px rgba(0, 0, 0, 0.22);
  animation: home-welcome-pop 0.7s cubic-bezier(0.22, 1.2, 0.36, 1) both;
}

.home-welcome__subtitle {
  position: relative;
  margin: 14px 0 0;
  font-size: 17px;
  letter-spacing: 3px;
  color: rgba(255, 255, 255, 0.88);
}

@keyframes home-welcome-pop {
  from {
    opacity: 0;
    transform: translateY(18px) scale(0.94);
  }

  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

@media (max-width: 900px) {
  .home-welcome {
    padding: 38px 14px 32px;
  }

  .home-welcome__title {
    font-size: 38px;
    letter-spacing: 6px;
  }

  .home-welcome__subtitle {
    font-size: 14px;
    letter-spacing: 2px;
  }
}

.home-view__grid {
  display: grid;
  gap: 20px;
  margin-bottom: 22px;
}

.home-view__grid--3 {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.home-view__grid--2 {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

/* ---- 核心功能卡 ---- */
.home-feature {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: 14px;
  padding: 30px 24px 26px;
  cursor: pointer;
}

.home-feature__title {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  color: var(--mkw-text);
}

.home-feature__desc {
  margin: 0;
  font-size: 13.5px;
  line-height: 1.75;
  color: var(--mkw-text-regular);
  min-height: 48px;
}

.home-feature__go {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  color: var(--mkw-primary);
  font-weight: 600;
}

/* ---- 技术特色卡 ---- */
.home-tech {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 22px 24px;
}

.home-tech__emoji {
  width: 54px;
  height: 54px;
  border-radius: 14px;
  background: #eef5ff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 26px;
  flex: 0 0 auto;
}

.home-tech__title {
  margin: 0 0 6px;
  font-size: 16px;
  font-weight: 700;
  color: var(--mkw-text);
}

.home-tech__desc {
  margin: 0;
  font-size: 13px;
  line-height: 1.7;
  color: var(--mkw-text-regular);
}

/* ---- 横幅卡 ---- */
.home-banner {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 22px 24px;
  cursor: pointer;
}

.home-banner__icon--warn {
  background: linear-gradient(135deg, #ffb36b 0%, #f56c6c 100%);
  box-shadow: 0 4px 12px rgba(245, 108, 108, 0.26);
}

.home-banner__body {
  flex: 1 1 auto;
  min-width: 0;
}

.home-banner__title {
  margin: 0 0 6px;
  font-size: 16px;
  font-weight: 700;
  color: var(--mkw-text);
}

.home-banner__desc {
  margin: 0;
  font-size: 13px;
  line-height: 1.7;
  color: var(--mkw-text-regular);
}

.home-banner__arrow {
  color: #c3d3ec;
  font-size: 18px;
  flex: 0 0 auto;
}

/* ---- 统计条 ---- */
.home-view__stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 18px;
}

.home-view__stats .mkw-stat {
  position: relative;
}

.mkw-stat__body {
  min-width: 0;
}

.mkw-stat__value small {
  font-size: 13px;
  font-weight: 600;
  margin-left: 2px;
}

.home-stat__trend {
  position: absolute;
  top: 12px;
  right: 14px;
  display: inline-flex;
  align-items: center;
  gap: 2px;
  font-size: 11px;
  padding: 1px 7px;
  border-radius: 20px;
}

.home-stat__trend.is-up {
  color: #31a05f;
  background: #e9f9ef;
}

.home-stat__trend.is-down {
  color: #e05a5a;
  background: #fdeeee;
}

@media (max-width: 1280px) {
  .home-view__stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .home-view__grid--3,
  .home-view__grid--2 {
    grid-template-columns: 1fr;
  }

  .home-view__stats {
    grid-template-columns: 1fr;
  }
}
</style>
