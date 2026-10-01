<template>
  <div class="app-shell">
    <SideNav />

    <!-- 移动端抽屉遮罩：点击关闭侧边栏 -->
    <div
      v-if="store.mobileNavOpen"
      class="app-shell__mask"
      @click="store.closeMobileNav()"
    ></div>

    <div class="app-shell__main">
      <AppHeader />

      <main class="app-shell__content">
        <div class="app-shell__view">
          <router-view v-slot="{ Component }">
            <transition name="fade-slide" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </div>

        <DisclaimerBar class="app-shell__disclaimer" />
      </main>
    </div>
  </div>
</template>

<script setup>
import { onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import SideNav from '@/components/SideNav.vue'
import AppHeader from '@/components/AppHeader.vue'
import DisclaimerBar from '@/components/DisclaimerBar.vue'
import { useAppStore } from '@/store/app'

const store = useAppStore()
const route = useRoute()

// 路由切换后自动收起移动端抽屉导航
watch(() => route.path, () => {
  store.closeMobileNav()
})

onMounted(() => {
  store.initSession()
  // 全局预取：图例 + 概览指标（失败自动走离线演示数据）
  store.fetchLegend().catch(() => {})
  store.fetchOverview().catch(() => {})
  store.setActiveMenu(route.path)
})
</script>

<style scoped lang="scss">
.app-shell {
  min-height: 100vh;
  background: var(--mkw-bg);
}

.app-shell__main {
  margin-left: var(--mkw-sidebar-width);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--mkw-bg);
}

.app-shell__content {
  flex: 1 1 auto;
  padding: 20px 24px 6px;
  overflow-y: auto;
  background: var(--mkw-bg-alt);
  display: flex;
  flex-direction: column;
}

.app-shell__disclaimer {
  margin-top: auto;
}

.app-shell__view {
  flex: 1 1 auto;
  min-height: 0;
}

.app-shell__mask {
  display: none;
}

@media (max-width: 900px) {
  .app-shell__main {
    margin-left: 0;
  }

  .app-shell__content {
    padding: 14px 12px 6px;
  }

  .app-shell__mask {
    display: block;
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.45);
    z-index: 99;
  }
}
</style>
