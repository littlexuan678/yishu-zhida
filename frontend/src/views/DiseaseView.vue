<template>
  <div class="mkw-page disease-view">
    <section class="mkw-hero">
      <h1 class="mkw-hero__title">疾病信息查询</h1>
      <p class="mkw-hero__subtitle">快速检索疾病信息，获取详细的症状、治疗方案等全面医疗知识</p>
    </section>

    <!-- 搜索区 -->
    <section class="mkw-card disease-view__search">
      <div class="disease-view__search-row">
        <el-input
          v-model="keyword"
          class="disease-view__input"
          size="large"
          clearable
          placeholder="请输入疾病名称，如：高血压"
          @keyup.enter="doSearch(1)"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
        <el-button class="mkw-btn-primary" size="large" :icon="Search" :loading="loading" @click="doSearch(1)">
          查询
        </el-button>
      </div>

      <div class="disease-view__hot">
        <span class="disease-view__hot-label">热门搜索：</span>
        <span
          v-for="k in hotKeywords"
          :key="k"
          class="mkw-pill mkw-pill--blue disease-view__hot-item"
          @click="quickSearch(k)"
        >
          {{ k }}
        </span>
      </div>
    </section>

    <el-alert
      v-if="searched"
      class="disease-view__alert"
      type="success"
      :closable="false"
      show-icon
      :title="`找到 ${total} 条相关疾病信息`"
      :description="resultMessage"
    />

    <!-- 结果 -->
    <section class="mkw-card disease-view__results">
      <div class="mkw-card__header">
        <el-icon class="mkw-card__title-icon"><Document /></el-icon>
        <span class="mkw-card__title">搜索结果 ({{ total }}条)</span>
        <div class="mkw-card__actions">
          <span class="disease-view__meta">
            第 {{ page }} / {{ totalPages }} 页
          </span>
        </div>
      </div>

      <div v-loading="loading" class="disease-view__grid-wrap">
        <div v-if="items.length" class="disease-view__grid">
          <DiseaseCard
            v-for="item in items"
            :key="item.disease_id"
            :disease="item"
            @detail="openDetail"
          />
        </div>
        <div v-else class="mkw-empty">
          <el-icon :size="26"><Search /></el-icon>
          <p>未找到相关疾病信息，请尝试其他关键词</p>
        </div>
      </div>

      <el-pagination
        v-if="total > pageSize"
        background
        layout="prev, pager, next, jumper, total"
        :current-page="page"
        :page-size="pageSize"
        :total="total"
        @current-change="onPageChange"
      />
    </section>

    <!-- 详情抽屉 -->
    <el-drawer v-model="drawerVisible" size="720px" :title="detail.name || '疾病详情'">
      <div v-loading="detailLoading" class="disease-detail">
        <template v-if="detail.name">
          <header class="disease-detail__head">
            <h2 class="disease-detail__name">{{ detail.name }}</h2>
            <div class="disease-detail__alias">
              <span
                v-for="a in detail.alias || []"
                :key="a"
                class="mkw-pill mkw-pill--gray"
              >{{ a }}</span>
              <span v-if="detail.is_infectious" class="mkw-pill mkw-pill--red">传染性疾病</span>
            </div>
            <div class="disease-detail__cats">
              <span class="disease-detail__cat-label">一级分类：</span>
              <span class="mkw-pill mkw-pill--blue">{{ detail.category1 || '未分类' }}</span>
              <span class="disease-detail__cat-label">二级分类：</span>
              <span class="mkw-pill mkw-pill--green">{{ detail.category2 || '未分类' }}</span>
            </div>
          </header>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">疾病定义</h4>
            <p class="disease-detail__text">{{ detail.definition || '暂无' }}</p>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">病因</h4>
            <p class="disease-detail__text">{{ detail.cause || '暂无' }}</p>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">症状</h4>
            <div class="disease-detail__pills">
              <span v-for="s in detail.symptoms || []" :key="s" class="mkw-pill mkw-pill--green">{{ s }}</span>
              <span v-if="!(detail.symptoms || []).length" class="disease-detail__text">暂无</span>
            </div>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">检查项目</h4>
            <div class="disease-detail__pills">
              <span v-for="c in detail.checks || []" :key="c" class="mkw-pill mkw-pill--blue">{{ c }}</span>
              <span v-if="!(detail.checks || []).length" class="disease-detail__text">暂无</span>
            </div>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">诊断</h4>
            <p class="disease-detail__text">{{ detail.diagnosis || '暂无' }}</p>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">治疗方案</h4>
            <p class="disease-detail__text">{{ detail.treatment || '暂无' }}</p>
            <div v-if="(detail.treatments || []).length" class="disease-detail__pills disease-detail__pills--top">
              <span v-for="t in detail.treatments" :key="t" class="mkw-pill mkw-pill--amber">{{ t }}</span>
            </div>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">常用药物</h4>
            <div class="disease-detail__pills">
              <span v-for="d in detail.drugs || []" :key="d" class="mkw-pill mkw-pill--blue">{{ d }}</span>
              <span v-if="!(detail.drugs || []).length" class="disease-detail__text">暂无</span>
            </div>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">就诊科室</h4>
            <div class="disease-detail__pills">
              <span v-if="detail.department" class="mkw-pill mkw-pill--green">{{ detail.department }}</span>
              <span v-else class="disease-detail__text">暂无</span>
            </div>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">预后</h4>
            <p class="disease-detail__text">{{ detail.prognosis || '暂无' }}</p>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">易感人群</h4>
            <p class="disease-detail__text">
              <el-icon class="disease-detail__inline-icon"><User /></el-icon>
              {{ detail.population || '暂无' }}
            </p>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">并发症</h4>
            <div class="disease-detail__pills">
              <span v-for="c in detail.complications || []" :key="c" class="mkw-pill mkw-pill--red">{{ c }}</span>
              <span v-if="!(detail.complications || []).length" class="disease-detail__text">暂无</span>
            </div>
          </section>

          <section class="disease-detail__section">
            <h4 class="disease-detail__section-title">鉴别诊断</h4>
            <div class="disease-detail__pills">
              <span v-for="d in detail.differential || []" :key="d" class="mkw-pill mkw-pill--gray">{{ d }}</span>
              <span v-if="!(detail.differential || []).length" class="disease-detail__text">暂无</span>
            </div>
          </section>

          <section v-if="(detail.sources || []).length" class="disease-detail__section">
            <h4 class="disease-detail__section-title">知识来源</h4>
            <ul class="disease-detail__sources">
              <li v-for="(s, i) in detail.sources" :key="i" class="disease-detail__source">
                <el-icon class="disease-detail__source-icon"><Document /></el-icon>
                <div class="disease-detail__source-body">
                  <div class="disease-detail__source-title">{{ s.title }}</div>
                  <div class="disease-detail__source-meta">
                    <span v-if="s.journal">{{ s.journal }}</span>
                    <span v-if="s.year">· {{ s.year }}</span>
                    <span v-if="s.authority">· 权威度 {{ s.authority }}</span>
                  </div>
                  <div v-if="s.authors" class="disease-detail__source-authors">{{ s.authors }}</div>
                  <div class="disease-detail__source-links">
                    <a
                      v-if="s.pmid"
                      class="mkw-link"
                      :href="s.url || `https://pubmed.ncbi.nlm.nih.gov/${s.pmid}/`"
                      target="_blank"
                      rel="noopener noreferrer"
                    >PMID: {{ s.pmid }}</a>
                    <a
                      v-if="s.doi"
                      class="mkw-link"
                      :href="s.url || `https://doi.org/${s.doi}`"
                      target="_blank"
                      rel="noopener noreferrer"
                    >DOI: {{ s.doi }}</a>
                    <a
                      v-if="!s.pmid && !s.doi && s.url"
                      class="mkw-link"
                      :href="s.url"
                      target="_blank"
                      rel="noopener noreferrer"
                    >查看原文</a>
                  </div>
                </div>
              </li>
            </ul>
          </section>

          <div class="disease-detail__updated">
            最后更新：{{ detail.updated_at || '—' }}
          </div>

          <DisclaimerBar inline />
        </template>

        <div v-else-if="!detailLoading" class="mkw-empty">
          <p>暂无疾病详情</p>
        </div>
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Search, Document, User } from '@element-plus/icons-vue'
import DiseaseCard from '@/components/DiseaseCard.vue'
import DisclaimerBar from '@/components/DisclaimerBar.vue'
import { getDiseaseDetail, getHotKeywords, searchDisease } from '@/api'

const FALLBACK_HOT = ['感冒', '高血压', '糖尿病', '冠心病', '肺炎', '胃炎']

const keyword = ref('')
const loading = ref(false)
const searched = ref(false)
const items = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(6)
const resultMessage = ref('')
const hotKeywords = ref([...FALLBACK_HOT])

const drawerVisible = ref(false)
const detailLoading = ref(false)
const detail = ref({})

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize.value)))

async function doSearch (p = 1) {
  loading.value = true
  try {
    const data = await searchDisease(keyword.value.trim(), p, pageSize.value)
    items.value = Array.isArray(data.items) ? data.items : []
    total.value = Number(data.total) || items.value.length
    page.value = Number(data.page) || p
    pageSize.value = Number(data.page_size) || pageSize.value
    resultMessage.value = data.message || ''
    searched.value = true
    if (Array.isArray(data.hot_keywords) && data.hot_keywords.length) {
      hotKeywords.value = data.hot_keywords
    }
    if (!items.value.length) {
      ElMessage.info('未找到相关疾病信息')
    }
  } catch (e) {
    ElMessage.error(e.friendlyMessage || '查询失败，请稍后重试')
    items.value = []
    total.value = 0
    searched.value = true
  } finally {
    loading.value = false
  }
}

function quickSearch (k) {
  keyword.value = k
  doSearch(1)
}

function onPageChange (p) {
  page.value = p
  doSearch(p)
}

async function openDetail (disease) {
  if (!disease || !disease.disease_id) return
  drawerVisible.value = true
  detailLoading.value = true
  detail.value = {}
  try {
    const data = await getDiseaseDetail(disease.disease_id)
    detail.value = data && data.name ? data : { ...disease, definition: '暂无详情数据' }
  } catch (e) {
    ElMessage.error(e.friendlyMessage || '详情加载失败')
    detail.value = { ...disease }
  } finally {
    detailLoading.value = false
  }
}

onMounted(async () => {
  try {
    const list = await getHotKeywords()
    if (Array.isArray(list) && list.length) hotKeywords.value = list
  } catch (e) {
    hotKeywords.value = [...FALLBACK_HOT]
  }
  doSearch(1)
})
</script>

<style scoped lang="scss">
.disease-view__search {
  margin-bottom: 16px;
}

.disease-view__search-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.disease-view__input {
  flex: 1 1 auto;
}

.disease-view__hot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
}

.disease-view__hot-label {
  font-size: 13px;
  color: var(--mkw-text-secondary);
}

.disease-view__hot-item {
  cursor: pointer;
  transition: background 0.2s ease;

  &:hover {
    background: #d8ebff;
  }
}

.disease-view__alert {
  margin-bottom: 16px;
}

.disease-view__results {
  min-height: 260px;
}

.disease-view__meta {
  font-size: 12px;
  color: var(--mkw-text-secondary);
}

.disease-view__grid-wrap {
  min-height: 180px;
}

.disease-view__grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.mkw-empty p {
  margin: 8px 0 0;
}

/* ---- 详情抽屉 ---- */
.disease-detail__head {
  padding-bottom: 14px;
  border-bottom: 1px dashed #e6ecf8;
  margin-bottom: 16px;
}

.disease-detail__name {
  margin: 0 0 10px;
  font-size: 22px;
  font-weight: 800;
  color: var(--mkw-text);
}

.disease-detail__alias,
.disease-detail__cats {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.disease-detail__alias {
  margin-bottom: 10px;
}

.disease-detail__cat-label {
  font-size: 13px;
  color: var(--mkw-text-secondary);
}

.disease-detail__section {
  margin-bottom: 18px;
}

.disease-detail__section-title {
  margin: 0 0 8px;
  font-size: 14.5px;
  font-weight: 700;
  color: var(--mkw-text);
  padding-left: 9px;
  border-left: 3px solid var(--mkw-primary);
  line-height: 1.2;
}

.disease-detail__text {
  margin: 0;
  font-size: 13.5px;
  line-height: 1.85;
  color: var(--mkw-text-regular);
  white-space: pre-wrap;
}

.disease-detail__inline-icon {
  color: var(--mkw-primary);
  margin-right: 4px;
  vertical-align: -2px;
}

.disease-detail__pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.disease-detail__pills--top {
  margin-top: 10px;
}

.disease-detail__sources {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.disease-detail__source {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  padding: 10px 12px;
  border-radius: 10px;
  background: #f7faff;
}

.disease-detail__source-icon {
  color: var(--mkw-primary);
  margin-top: 2px;
  flex: 0 0 auto;
}

.disease-detail__source-body {
  min-width: 0;
}

.disease-detail__source-title {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--mkw-text);
  line-height: 1.6;
}

.disease-detail__source-meta {
  font-size: 12px;
  color: var(--mkw-text-secondary);
  margin-top: 2px;
}

.disease-detail__source-authors {
  font-size: 11.5px;
  color: var(--mkw-text-placeholder);
  margin-top: 2px;
}

.disease-detail__source-links {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 6px;
  font-size: 12px;
}

.disease-detail__updated {
  margin-top: 8px;
  font-size: 11.5px;
  color: var(--mkw-text-placeholder);
}

@media (max-width: 1400px) {
  .disease-view__grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .disease-view__grid {
    grid-template-columns: 1fr;
  }
}
</style>
