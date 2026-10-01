import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

/**
 * 医数智答 · 智愈医典 —— 前端构建配置
 * ---------------------------------------------------------------------------
 * 说明：Element Plus 与全部图标已在 `src/main.js` 中**全量注册**
 *      （`app.use(ElementPlus)` + 遍历注册 `@element-plus/icons-vue`），
 *      因此这里不再引入 unplugin-auto-import / unplugin-vue-components，
 *      避免额外的构建期依赖与按需解析带来的解析失败风险。
 */
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  css: {
    preprocessorOptions: {
      scss: {
        api: 'modern-compiler'
      }
    }
  },
  server: {
    port: 5173,
    host: '127.0.0.1',
    open: false,
    // 开发环境把 /api 反向代理到 FastAPI 后端，规避跨域
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        // 把体积较大的库拆包，提升首屏加载速度
        manualChunks: {
          vue: ['vue', 'vue-router', 'pinia'],
          element: ['element-plus', '@element-plus/icons-vue'],
          charts: ['echarts'],
          graph: ['d3']
        }
      }
    }
  }
})
