<template>
  <div class="mkw-disclaimer" :class="{ 'mkw-disclaimer--inline': inline }">
    <span class="mkw-disclaimer__icon">⚠</span>
    <span class="mkw-disclaimer__text">{{ text }}</span>
  </div>
</template>

<script setup>
defineProps({
  /** 内联模式：用于聊天气泡 / 风险预测结果面板 */
  inline: {
    type: Boolean,
    default: false
  },
  text: {
    type: String,
    default: '本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。'
  }
})
</script>

<style scoped lang="scss">
.mkw-disclaimer {
  position: relative;
}

.mkw-disclaimer__icon {
  flex: 0 0 auto;
  font-size: 13px;
}

.mkw-disclaimer__text {
  flex: 0 1 auto;
}

.mkw-disclaimer--inline {
  background: #fffaf0;
  border: 1px dashed #f0d9a8;
  padding: 8px 12px;
}
</style>
