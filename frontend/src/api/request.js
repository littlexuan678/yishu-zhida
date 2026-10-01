import axios from 'axios'

/**
 * 统一 axios 实例
 * - baseURL 来自 VITE_API_BASE（默认 /api/v1，开发环境由 vite proxy 转发至 127.0.0.1:8000）
 * - 请求拦截器：自动附加 session_id / 时间戳
 * - 响应拦截器：兼容两种后端返回 —— 裸 JSON 对象，或 {code,message,data} 包装体
 */
const request = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '/api/v1',
  timeout: 20000,
  headers: { 'Content-Type': 'application/json' }
})

/** 标记：后端当前是否疑似不可用（供页面显示"离线演示数据"提示） */
export const backendState = {
  offline: false,
  lastError: ''
}

export function markOffline (err) {
  backendState.offline = true
  backendState.lastError = (err && err.message) || String(err || 'unknown')
}

export function markOnline () {
  backendState.offline = false
  backendState.lastError = ''
}

request.interceptors.request.use(
  (config) => {
    let sessionId = ''
    try {
      sessionId = localStorage.getItem('mkw_session_id') || ''
    } catch (e) {
      sessionId = ''
    }
    if (sessionId) {
      config.headers['X-Session-Id'] = sessionId
    }
    config.headers['X-Requested-With'] = 'XMLHttpRequest'
    return config
  },
  (error) => Promise.reject(error)
)

request.interceptors.response.use(
  (response) => {
    const body = response.data
    // 裸 JSON（本后端契约）：直接返回
    if (body === null || body === undefined) return body
    if (typeof body !== 'object') return body
    // 包装体：{code, message, data}
    if (Object.prototype.hasOwnProperty.call(body, 'code') &&
        (Object.prototype.hasOwnProperty.call(body, 'data') ||
         Object.prototype.hasOwnProperty.call(body, 'message'))) {
      const code = Number(body.code)
      if (!Number.isNaN(code) && code !== 0 && code !== 200) {
        const err = new Error(body.message || `接口返回异常 code=${body.code}`)
        err.code = code
        err.payload = body
        return Promise.reject(err)
      }
      return body.data !== undefined ? body.data : body
    }
    return body
  },
  (error) => {
    let msg = '网络请求失败'
    if (error.response) {
      const st = error.response.status
      const d = error.response.data
      msg = (d && (d.message || d.detail || d.msg)) || `请求失败（HTTP ${st}）`
      error.status = st
    } else if (error.code === 'ECONNABORTED' || /timeout/i.test(error.message || '')) {
      msg = '请求超时，请稍后重试'
    } else if (error.message) {
      msg = error.message
    }
    error.friendlyMessage = msg
    return Promise.reject(error)
  }
)

export default request
