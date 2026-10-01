<template>
  <aside class="side-nav" :class="{ 'is-open': store.mobileNavOpen }">
    <div class="side-nav__logo">
      <div class="side-nav__logo-mark">
        <el-icon :size="22"><FirstAidKit /></el-icon>
      </div>
      <span class="side-nav__logo-text">智能医学百科</span>
      <el-icon class="side-nav__close" :size="20" @click="store.closeMobileNav()">
        <Close />
      </el-icon>
    </div>

    <nav class="side-nav__menu">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        class="side-nav__item"
        :class="{ 'is-active': isActive(item.path) }"
        :to="item.path"
      >
        <el-icon class="side-nav__item-icon" :size="17">
          <component :is="item.icon" />
        </el-icon>
        <span class="side-nav__item-text">{{ item.title }}</span>
      </router-link>
    </nav>

    <div class="side-nav__footer">
      <span class="side-nav__footer-dot"></span>
      <span>Knowledge Graph QA</span>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import {
  HomeFilled,
  Share,
  Search,
  ChatDotRound,
  DataAnalysis,
  FirstAidKit,
  Close
} from '@element-plus/icons-vue'
import { useAppStore } from '@/store/app'

const route = useRoute()
const store = useAppStore()

const navItems = [
  { path: '/', title: '主页', icon: HomeFilled },
  { path: '/graph', title: '知识图谱', icon: Share },
  { path: '/disease', title: '疾病查询', icon: Search },
  { path: '/chat', title: '智能问答', icon: ChatDotRound },
  { path: '/analytics', title: '数据分析', icon: DataAnalysis }
]

const isActive = computed(() => (path) => {
  if (path === '/') return route.path === '/'
  return route.path === path || route.path.startsWith(`${path}/`)
})
</script>

<style scoped lang="scss">
.side-nav {
  position: fixed;
  top: 0;
  left: 0;
  width: var(--mkw-sidebar-width);
  height: 100vh;
  background: var(--mkw-gradient-sidebar);
  display: flex;
  flex-direction: column;
  padding: 18px 12px 14px;
  z-index: 100;
  overflow: hidden;

  &::after {
    content: '';
    position: absolute;
    right: -60px;
    bottom: -60px;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    pointer-events: none;
  }
}

.side-nav__logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 8px 20px;
}

.side-nav__logo-mark {
  width: 38px;
  height: 38px;
  border-radius: 11px;
  background: #ffffff;
  color: #29c5e8;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(40, 60, 140, 0.18);
  flex: 0 0 auto;
}

.side-nav__logo-text {
  color: #ffffff;
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.side-nav__menu {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1 1 auto;
  overflow-y: auto;
}

.side-nav__item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 20px;
  border-radius: 10px;
  color: rgba(255, 255, 255, 0.88);
  font-size: 15px;
  font-weight: 500;
  text-decoration: none;
  cursor: pointer;
  transition: background 0.2s ease, color 0.2s ease;
  user-select: none;
}

.side-nav__item:hover {
  background: rgba(255, 255, 255, 0.12);
  color: #ffffff;
}

.side-nav__item.is-active {
  background: rgba(255, 255, 255, 0.22);
  color: #ffffff;
  font-weight: 700;
}

.side-nav__item-icon {
  flex: 0 0 auto;
}

.side-nav__item-text {
  white-space: nowrap;
}

.side-nav__footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding-top: 12px;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.66);
  letter-spacing: 0.4px;
  white-space: nowrap;
}

.side-nav__footer-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #7ef0b0;
  box-shadow: 0 0 6px #7ef0b0;
}

.side-nav__close {
  display: none;
  margin-left: auto;
  color: rgba(255, 255, 255, 0.85);
  cursor: pointer;
  flex: 0 0 auto;
}

@media (max-width: 900px) {
  .side-nav {
    width: 236px;
    transform: translateX(-110%);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
  }

  .side-nav.is-open {
    transform: translateX(0);
    box-shadow: 0 0 48px rgba(15, 23, 42, 0.35);
  }

  .side-nav__close {
    display: inline-flex;
  }

  .side-nav__item {
    padding: 13px 16px;
    font-size: 14px;
  }

  .side-nav__logo-text {
    font-size: 14px;
  }
}
</style>
