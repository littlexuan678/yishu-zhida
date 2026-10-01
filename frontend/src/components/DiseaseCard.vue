<template>
  <article class="disease-card mkw-card mkw-card--hover" @click="$emit('detail', disease)">
    <header class="disease-card__head">
      <h3 class="disease-card__title" :title="disease.name">{{ disease.name }}</h3>
      <span class="disease-card__link" @click.stop="$emit('detail', disease)">
        <el-icon><View /></el-icon>
        查看详情
      </span>
    </header>

    <div class="disease-card__row">
      <span class="disease-card__label">一级分类：</span>
      <span class="mkw-pill mkw-pill--blue">{{ disease.category1 || '未分类' }}</span>
      <span v-if="disease.is_infectious" class="mkw-pill mkw-pill--red disease-card__infectious">传染性</span>
    </div>

    <div class="disease-card__row">
      <span class="disease-card__label">二级分类：</span>
      <span class="mkw-pill mkw-pill--green">{{ disease.category2 || '未分类' }}</span>
    </div>

    <div class="disease-card__row disease-card__row--text">
      <el-icon class="disease-card__row-icon disease-card__row-icon--warn"><Warning /></el-icon>
      <span class="disease-card__label">症状：</span>
      <span class="disease-card__value mkw-ellipsis" :title="symptomText">{{ symptomText }}</span>
    </div>

    <div class="disease-card__row disease-card__row--text">
      <el-icon class="disease-card__row-icon disease-card__row-icon--plus"><CirclePlus /></el-icon>
      <span class="disease-card__label">治疗：</span>
      <span class="disease-card__value mkw-ellipsis" :title="treatmentText">{{ treatmentText }}</span>
    </div>

    <footer class="disease-card__foot">
      <el-icon class="disease-card__foot-icon"><User /></el-icon>
      <span class="mkw-ellipsis" :title="disease.population">{{ disease.population || '易感人群未知' }}</span>
    </footer>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import { View, Warning, CirclePlus, User } from '@element-plus/icons-vue'

const props = defineProps({
  disease: {
    type: Object,
    required: true
  }
})

defineEmits(['detail'])

const symptomText = computed(() => {
  const list = Array.isArray(props.disease.symptoms) ? props.disease.symptoms : []
  return list.length ? list.join('、') : '暂无症状信息'
})

const treatmentText = computed(() => {
  const list = Array.isArray(props.disease.treatments) ? props.disease.treatments : []
  return list.length ? list.join('、') : '暂无治疗方案信息'
})
</script>

<style scoped lang="scss">
.disease-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  cursor: pointer;
  padding: 18px 20px;
  height: 100%;
}

.disease-card__head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.disease-card__title {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: var(--mkw-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1 1 auto;
}

.disease-card__link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  color: var(--mkw-primary);
  cursor: pointer;
  flex: 0 0 auto;

  &:hover {
    text-decoration: underline;
  }
}

.disease-card__row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--mkw-text-regular);
  min-width: 0;
}

.disease-card__row--text {
  align-items: center;
}

.disease-card__label {
  flex: 0 0 auto;
  color: var(--mkw-text-secondary);
}

.disease-card__value {
  flex: 1 1 auto;
  min-width: 0;
  color: var(--mkw-text-regular);
}

.disease-card__row-icon {
  font-size: 15px;
  flex: 0 0 auto;
}

.disease-card__row-icon--warn {
  color: #e6a23c;
}

.disease-card__row-icon--plus {
  color: #31a05f;
}

.disease-card__infectious {
  margin-left: auto;
}

.disease-card__foot {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 2px;
  padding-top: 10px;
  border-top: 1px dashed #eef2fb;
  font-size: 12px;
  color: var(--mkw-text-secondary);
  min-width: 0;
}

.disease-card__foot-icon {
  font-size: 13px;
  flex: 0 0 auto;
}
</style>
