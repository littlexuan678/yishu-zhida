import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: '主页', icon: 'HomeFilled' }
  },
  {
    path: '/graph',
    name: 'graph',
    component: () => import('@/views/GraphView.vue'),
    meta: { title: '知识图谱', icon: 'Share' }
  },
  {
    path: '/disease',
    name: 'disease',
    component: () => import('@/views/DiseaseView.vue'),
    meta: { title: '疾病查询', icon: 'Search' }
  },
  {
    path: '/chat',
    name: 'chat',
    component: () => import('@/views/ChatView.vue'),
    meta: { title: '智能问答', icon: 'ChatDotRound' }
  },
  {
    path: '/analytics',
    name: 'analytics',
    component: () => import('@/views/AnalyticsView.vue'),
    meta: { title: '数据分析', icon: 'DataAnalysis' }
  },
  { path: '/:pathMatch(.*)*', redirect: '/' }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 })
})

router.afterEach((to) => {
  const base = '智愈医典·医疗知识信息问答系统'
  const name = to.meta && to.meta.title ? `${to.meta.title} · ` : ''
  document.title = `${name}${base}`
})

export default router
