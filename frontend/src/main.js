import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

// 样式：Element Plus 全量样式 + 项目主题
import 'element-plus/dist/index.css'
import '@/styles/theme.scss'

import App from './App.vue'
import router from './router'

const app = createApp(App)

// 显式全量注册 Element Plus 图标（<el-icon><HomeFilled /></el-icon> 可直接使用）
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  if (!/^[A-Z]/.test(key)) continue
  app.component(key, component)
}

app.use(createPinia())
app.use(router)
app.use(ElementPlus, { locale: zhCn, size: 'default', zIndex: 3000 })

app.config.errorHandler = (err, instance, info) => {
  console.error('[医数智答] 运行异常：', err, info)
}

app.mount('#app')
