<template>
  <div class="mkw-page chat-view">
    <section class="mkw-hero">
      <h1 class="mkw-hero__title">智能问答</h1>
      <p class="mkw-hero__subtitle">基于医疗知识图谱与大语言模型，为您提供有据可循的医疗知识问答服务</p>
    </section>

    <!-- 助手头部 -->
    <section class="mkw-card chat-view__header">
      <div class="chat-view__avatar">
        <el-icon :size="30"><Cpu /></el-icon>
        <span class="chat-view__avatar-cross">✚</span>
      </div>
      <div class="chat-view__header-body">
        <h3 class="chat-view__header-title">AI医疗助手</h3>
        <div class="chat-view__header-status">
          <span class="chat-view__dot"></span>
          在线 · 基于LLM模型
        </div>
      </div>
      <div class="chat-view__header-stat">
        <div class="chat-view__header-stat-value">{{ messages.length }}</div>
        <div class="chat-view__header-stat-label">对话数</div>
      </div>
    </section>

    <!-- 消息区 -->
    <section class="mkw-card chat-view__messages-wrap">
      <div ref="scrollRef" class="chat-view__messages">
        <div v-for="(msg, mi) in messages" :key="msg.id" class="chat-msg" :class="`chat-msg--${msg.role}`">
          <div class="chat-msg__avatar" :class="`chat-msg__avatar--${msg.role}`">
            <el-icon :size="18">
              <component :is="msg.role === 'assistant' ? Cpu : UserFilled" />
            </el-icon>
          </div>

          <div class="chat-msg__body">
            <!-- 助手 -->
            <template v-if="msg.role === 'assistant'">
              <div class="chat-msg__bubble chat-msg__bubble--assistant">
                <div class="chat-msg__time">{{ msg.time }}</div>

                <div v-if="msg.pending" class="chat-typing">
                  <span></span><span></span><span></span>
                  <em>正在检索知识图谱并生成回答…</em>
                </div>

                <template v-else>
                  <div
                    v-if="msg.answerHtml"
                    class="chat-msg__answer chat-msg__answer--html"
                    v-html="msg.answerHtml"
                    @click="onAnswerClick($event, mi)"
                  ></div>
                  <div v-else class="chat-msg__answer">{{ msg.answer }}</div>

                  <div class="chat-msg__meta-row">
                    <span
                      v-if="msg.confidence != null"
                      class="mkw-pill"
                      :class="confidenceClass(msg.confidence)"
                    >
                      置信度 {{ (Number(msg.confidence) * 100).toFixed(0) }}%
                    </span>
                    <span v-if="msg.intentLabel" class="mkw-pill mkw-pill--gray">
                      意图：{{ msg.intentLabel }}
                    </span>
                    <span v-if="msg.promptTemplateId" class="mkw-pill mkw-pill--amber">
                      模板 {{ msg.promptTemplateId }}
                    </span>
                    <span v-if="msg.llmModel" class="mkw-pill mkw-pill--gray">{{ msg.llmModel }}</span>
                  </div>

                  <!-- 推理链路 -->
                  <el-collapse
                    v-if="(msg.reasoningTrace || []).length"
                    v-model="msg.openReasoning"
                    class="chat-msg__collapse"
                  >
                    <el-collapse-item name="reasoning">
                      <template #title>
                        <span class="chat-msg__collapse-title">
                          <el-icon><Guide /></el-icon>
                          推理链路
                          <em>{{ msg.reasoningTrace.length }} 步</em>
                        </span>
                      </template>
                      <ol class="reason-trace">
                        <li v-for="(t, ti) in msg.reasoningTrace" :key="ti" class="reason-trace__item">
                          <span class="reason-trace__step">{{ t.step != null ? t.step : ti + 1 }}</span>
                          <div class="reason-trace__body">
                            <div class="reason-trace__head">
                              <span class="reason-trace__title">{{ t.title || t.action || '步骤' }}</span>
                              <span v-if="t.action" class="reason-trace__action">{{ t.action }}</span>
                              <span v-if="t.elapsed_ms != null" class="reason-trace__ms">{{ t.elapsed_ms }} ms</span>
                            </div>
                            <div class="reason-trace__detail">{{ t.detail }}</div>
                          </div>
                        </li>
                      </ol>
                    </el-collapse-item>
                  </el-collapse>

                  <!-- 知识溯源 -->
                  <el-collapse
                    v-if="(msg.evidences || []).length"
                    v-model="msg.openEvidence"
                    class="chat-msg__collapse"
                  >
                    <el-collapse-item name="evidence">
                      <template #title>
                        <span class="chat-msg__collapse-title">
                          <el-icon><Link /></el-icon>
                          知识溯源
                          <em>{{ msg.evidences.length }} 条</em>
                        </span>
                      </template>
                      <ul class="evidences">
                        <li
                          v-for="ev in msg.evidences"
                          :key="ev.triple_id"
                          :id="`evidence-${msg.id}-${ev.triple_id}`"
                          class="evidence"
                        >
                          <div class="evidence__head">
                            <span class="evidence__ref">{{ formatRef(ev.triple_id) }}</span>
                            <span class="evidence__triple">
                              {{ ev.head }}
                              <em>—{{ ev.relation_label || ev.relation }}→</em>
                              {{ ev.tail }}
                            </span>
                            <span
                              v-if="ev.confidence != null"
                              class="mkw-pill"
                              :class="confidenceClass(ev.confidence)"
                            >{{ (Number(ev.confidence) * 100).toFixed(0) }}%</span>
                          </div>
                          <div class="evidence__types">
                            <span class="mkw-pill mkw-pill--blue">{{ ev.head_type || '实体' }}</span>
                            <span class="evidence__arrow">→</span>
                            <span class="mkw-pill mkw-pill--green">{{ ev.tail_type || '实体' }}</span>
                            <span class="evidence__rel">{{ ev.relation }}</span>
                          </div>
                          <div v-if="(ev.sources || []).length" class="evidence__sources">
                            <a
                              v-for="(s, si) in ev.sources"
                              :key="si"
                              class="evidence__source"
                              :href="s.url || (s.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${s.pmid}/` : '#')"
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              <el-icon><Document /></el-icon>
                              {{ s.title || s.journal || '来源文献' }}
                              <em v-if="s.year">({{ s.year }})</em>
                            </a>
                          </div>
                        </li>
                      </ul>
                    </el-collapse-item>
                  </el-collapse>

                  <DisclaimerBar inline :text="msg.disclaimer || undefined" />

                  <div class="chat-msg__actions">
                    <span
                      class="chat-msg__action"
                      :class="{ 'is-active': msg.voted === 'up' }"
                      @click="vote(msg, 'up')"
                    >
                      <el-icon><Select /></el-icon>
                      有用
                    </span>
                    <span class="chat-msg__action" @click="copyAnswer(msg)">
                      <el-icon><CopyDocument /></el-icon>
                      复制
                    </span>
                    <span v-if="msg.latency" class="chat-msg__latency">耗时 {{ msg.latency }} ms</span>
                  </div>
                </template>
              </div>
            </template>

            <!-- 用户 -->
            <template v-else>
              <div class="chat-msg__bubble chat-msg__bubble--user">
                <div class="chat-msg__time chat-msg__time--user">{{ msg.time }}</div>
                <div class="chat-msg__answer">{{ msg.content }}</div>
              </div>
            </template>
          </div>
        </div>
      </div>
    </section>

    <!-- 推荐问题 -->
    <div class="chat-view__suggest">
      <span
        v-for="q in suggestedQuestions"
        :key="q"
        class="chat-view__suggest-chip"
        @click="send(q)"
      >{{ q }}</span>
    </div>

    <!-- 输入区 -->
    <section class="mkw-card chat-view__input-row">
      <el-input
        v-model="input"
        class="chat-view__input"
        size="large"
        type="textarea"
        :rows="1"
        :autosize="{ minRows: 1, maxRows: 4 }"
        resize="none"
        placeholder="请输入您的医疗问题..."
        @keydown.enter.exact.prevent="onEnter"
      />
      <el-button
        class="mkw-btn-primary chat-view__send"
        size="large"
        :icon="Promotion"
        :loading="loading"
        :disabled="loading"
        @click="onEnter"
      >
        发送
      </el-button>
    </section>

    <p class="chat-view__tip">提示：您可以询问疾病症状、治疗方法、用药指导等医疗相关问题</p>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, reactive, ref } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Cpu, UserFilled, Promotion, Guide, Link, Document, Select, CopyDocument } from '@element-plus/icons-vue'
import DisclaimerBar from '@/components/DisclaimerBar.vue'
import { askQuestion } from '@/api'
import { useAppStore } from '@/store/app'

const route = useRoute()

const DEFAULT_QUESTIONS = [
  '成人呼吸窘迫综合征的症状有哪些？',
  '肺栓塞应该挂什么科？',
  '肺心病的治疗方法是什么？',
  '急性呼吸窘迫综合征的治疗费用大概多少？',
  '继发性肺动脉高压是否传染？'
]

const store = useAppStore()
const scrollRef = ref(null)
const input = ref('')
const loading = ref(false)
const messages = reactive([])
const relatedQuestions = ref([])

let msgSeq = 0

const suggestedQuestions = computed(() => {
  const list = relatedQuestions.value && relatedQuestions.value.length
    ? relatedQuestions.value
    : DEFAULT_QUESTIONS
  return list.slice(0, 6)
})

function nowTime () {
  const d = new Date()
  const p = (n) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

function confidenceClass (v) {
  const n = Number(v)
  if (Number.isNaN(n)) return 'mkw-pill--gray'
  if (n >= 0.85) return 'mkw-pill--green'
  if (n >= 0.6) return 'mkw-pill--blue'
  return 'mkw-pill--amber'
}

function formatRef (id) {
  if (!id) return '[KG-?]'
  const s = String(id)
  return s.startsWith('[') ? s : `[${s.startsWith('KG-') ? s : `KG-${s}`}]`
}

/** 简易 HTML 清洗：仅保留安全的展示型标签 */
function sanitizeHtml (html) {
  if (!html || typeof html !== 'string') return ''
  let out = html
  out = out.replace(/<script[\s\S]*?<\/script>/gi, '')
  out = out.replace(/<style[\s\S]*?<\/style>/gi, '')
  out = out.replace(/<iframe[\s\S]*?<\/iframe>/gi, '')
  out = out.replace(/\son\w+\s*=\s*"[^"]*"/gi, '')
  out = out.replace(/\son\w+\s*=\s*'[^']*'/gi, '')
  out = out.replace(/\son\w+\s*=\s*[^\s>]+/gi, '')
  out = out.replace(/javascript:/gi, '')
  // 把 <sup ...>[KG-n]</sup> 规范化为可点击标记；再处理裸文本 [KG-n]，避免重复包裹
  out = out.replace(/<sup([^>]*)>(\s*\[?)(KG-\d+)(\]?\s*)<\/sup>/gi, (_m, attrs, pre, ref, post) => {
    const clean = String(attrs)
      .replace(/\s*data-ref\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, '')
      .replace(/\s*class\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, '')
      .trim()
    return `<sup class="kg-ref" data-ref="${ref}"${clean ? ` ${clean}` : ''}>[${ref}]</sup>`
  })
  out = out.replace(/\[(KG-\d+)\](?!<\/sup>)/g, (m, ref) => {
    return `<sup class="kg-ref" data-ref="${ref}">[${ref}]</sup>`
  })
  return out
}

function pushAssistant (payload) {
  const msg = {
    id: `a${++msgSeq}`,
    role: 'assistant',
    time: nowTime(),
    answer: payload.answer || '',
    answerHtml: sanitizeHtml(payload.answer_html || ''),
    evidences: Array.isArray(payload.evidences) ? payload.evidences : [],
    reasoningTrace: Array.isArray(payload.reasoning_trace) ? payload.reasoning_trace : [],
    confidence: payload.confidence,
    promptTemplateId: payload.prompt_template_id || '',
    llmModel: payload.llm_model || '',
    intentLabel: (payload.intent && (payload.intent.label_cn || payload.intent.label)) || '',
    disclaimer: payload.disclaimer || '',
    latency: payload.latency_ms ? payload.latency_ms.total : null,
    openReasoning: [],
    openEvidence: [],
    voted: '',
    pending: false
  }
  messages.push(msg)
  return msg
}

function pushUser (text) {
  messages.push({
    id: `u${++msgSeq}`,
    role: 'user',
    time: nowTime(),
    content: text
  })
}

async function scrollToBottom () {
  await nextTick()
  const el = scrollRef.value
  if (el) el.scrollTop = el.scrollHeight
}

async function send (text) {
  const question = (typeof text === 'string' ? text : input.value).trim()
  if (!question) {
    ElMessage.warning('请输入您想咨询的医疗问题')
    return
  }
  if (loading.value) return

  pushUser(question)
  input.value = ''
  loading.value = true

  const pendingMsg = {
    id: `a${++msgSeq}`,
    role: 'assistant',
    time: nowTime(),
    pending: true,
    openReasoning: [],
    openEvidence: [],
    evidences: [],
    reasoningTrace: []
  }
  messages.push(pendingMsg)
  await scrollToBottom()

  try {
    const data = await askQuestion({
      question,
      session_id: store.sessionId,
      top_k: 12,
      max_hops: 2,
      explain: true
    })
    const idx = messages.findIndex((m) => m.id === pendingMsg.id)
    if (idx !== -1) messages.splice(idx, 1)
    pushAssistant(data)
    if (Array.isArray(data.related_questions) && data.related_questions.length) {
      relatedQuestions.value = data.related_questions
    }
    if (data.session_id && data.session_id !== store.sessionId) {
      store.sessionId = data.session_id
      try {
        localStorage.setItem('mkw_session_id', data.session_id)
      } catch (e) {
        /* ignore */
      }
    }
  } catch (e) {
    const idx = messages.findIndex((m) => m.id === pendingMsg.id)
    if (idx !== -1) messages.splice(idx, 1)
    pushAssistant({
      answer: `抱歉，本次问答请求失败：${e.friendlyMessage || e.message || '未知错误'}。请稍后重试。`,
      confidence: null,
      disclaimer: '本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。'
    })
    ElMessage.error(e.friendlyMessage || '问答请求失败，请稍后重试')
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

function onEnter () {
  send()
}

function vote (msg, type) {
  msg.voted = msg.voted === type ? '' : type
  if (msg.voted) ElMessage.success('感谢您的反馈，我们会持续优化回答质量')
}

async function copyAnswer (msg) {
  const text = (msg.answer || '').replace(/<[^>]+>/g, '')
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text)
    } else {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }
    ElMessage.success('回答已复制到剪贴板')
  } catch (e) {
    ElMessage.warning('复制失败，请手动选择文本复制')
  }
}

/** 点击答案中的 [KG-n] 上标 → 展开知识溯源并滚动定位 */
function onAnswerClick (event, mi) {
  const target = event.target
  if (!target || !target.classList) return
  if (!target.classList.contains('kg-ref')) return
  const msg = messages[mi]
  if (!msg) return
  const ref = target.getAttribute('data-ref') || target.textContent
  const norm = String(ref).replace(/[\[\]]/g, '')
  msg.openEvidence = ['evidence']
  nextTick(() => {
    const el = document.getElementById(`evidence-${msg.id}-${norm}`)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      el.classList.add('is-flash')
      setTimeout(() => el.classList.remove('is-flash'), 1600)
    } else {
      ElMessage.info(`未找到 ${norm} 对应的证据条目`)
    }
  })
}

onMounted(() => {
  store.initSession()
  pushAssistant({
    answer:
      '您好，我是AI医疗助手。我可以基于医疗知识图谱为您解答疾病症状、就诊科室、治疗方法与用药指导等问题。\n' +
      '请尽量描述清楚具体的疾病名称或症状，例如「高血压的治疗方法是什么？」「肺栓塞应该挂什么科？」。',
    confidence: 0.99,
    prompt_template_id: 'qa_greeting_v1',
    llm_model: '医疗知识图谱问答引擎',
    intent: { label_cn: '问候', label: 'greeting' },
    disclaimer: '本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。'
  })
  // 支持 ?q=xxx 进入（如知识图谱节点「去问答了解它」）：自动提问
  const preset = typeof route.query.q === 'string' ? route.query.q.trim() : ''
  if (preset) send(preset)
})
</script>

<style scoped lang="scss">
/* ---- 头部 ---- */
.chat-view__header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
  padding: 18px 22px;
}

.chat-view__avatar {
  position: relative;
  width: 58px;
  height: 58px;
  border-radius: 50%;
  background: var(--mkw-gradient-circle);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 6px 16px rgba(43, 143, 232, 0.3);
  flex: 0 0 auto;
}

.chat-view__avatar-cross {
  position: absolute;
  right: -2px;
  bottom: -2px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #fff;
  color: #31c48d;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
}

.chat-view__header-body {
  flex: 1 1 auto;
  min-width: 0;
}

.chat-view__header-title {
  margin: 0 0 4px;
  font-size: 18px;
  font-weight: 700;
  color: var(--mkw-text);
}

.chat-view__header-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12.5px;
  color: var(--mkw-text-secondary);
}

.chat-view__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #31c48d;
  box-shadow: 0 0 0 3px rgba(49, 196, 141, 0.16);
}

.chat-view__header-stat {
  text-align: center;
  padding: 8px 18px;
  border-radius: 12px;
  background: #f2f7ff;
  flex: 0 0 auto;
}

.chat-view__header-stat-value {
  font-size: 20px;
  font-weight: 800;
  color: var(--mkw-primary);
  line-height: 1.1;
}

.chat-view__header-stat-label {
  font-size: 12px;
  color: var(--mkw-text-secondary);
}

/* ---- 消息区 ---- */
.chat-view__messages-wrap {
  padding: 16px 20px;
  margin-bottom: 14px;
}

.chat-view__messages {
  max-height: 62vh;
  overflow-y: auto;
  padding-right: 6px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.chat-msg {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}

.chat-msg--user {
  flex-direction: row-reverse;
}

.chat-msg__avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  flex: 0 0 auto;
  box-shadow: 0 3px 10px rgba(80, 120, 220, 0.2);
}

.chat-msg__avatar--assistant {
  background: var(--mkw-gradient-circle);
}

.chat-msg__avatar--user {
  background: linear-gradient(135deg, #ffb36b 0%, #f56c6c 100%);
}

.chat-msg__body {
  max-width: 82%;
  min-width: 0;
}

.chat-msg__bubble {
  position: relative;
  border-radius: 14px;
  padding: 14px 16px 12px;
  font-size: 13.5px;
  line-height: 1.85;
}

.chat-msg__bubble--assistant {
  background: #f7faff;
  border: 1px solid #e8effb;
  color: var(--mkw-text);
}

.chat-msg__bubble--user {
  background: var(--mkw-gradient-primary);
  color: #ffffff;
  box-shadow: 0 6px 16px rgba(43, 143, 232, 0.24);
}

.chat-msg__time {
  font-size: 11px;
  color: var(--mkw-text-placeholder);
  text-align: right;
  margin-bottom: 6px;
}

.chat-msg__time--user {
  color: rgba(255, 255, 255, 0.78);
}

.chat-msg__answer {
  white-space: pre-wrap;
  word-break: break-word;
}

.chat-msg__answer--html :deep(p) {
  margin: 0 0 8px;
}

.chat-msg__answer--html :deep(p:last-child) {
  margin-bottom: 0;
}

.chat-msg__answer--html :deep(ul),
.chat-msg__answer--html :deep(ol) {
  margin: 6px 0;
  padding-left: 20px;
}

.chat-msg__answer--html :deep(.kg-ref) {
  display: inline-block;
  margin-left: 3px;
  padding: 0 5px;
  border-radius: 8px;
  background: #e8f3ff;
  color: #2b8fe8;
  font-size: 10.5px;
  line-height: 16px;
  font-weight: 700;
  cursor: pointer;
  vertical-align: super;
  transition: background 0.2s ease;
}

.chat-msg__answer--html :deep(.kg-ref:hover) {
  background: #cfe6ff;
}

.chat-msg__meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.chat-msg__collapse {
  margin-top: 10px;
}

.chat-msg__collapse-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;

  em {
    font-style: normal;
    font-size: 11.5px;
    color: var(--mkw-text-secondary);
  }
}

/* ---- 打字中 ---- */
.chat-typing {
  display: flex;
  align-items: center;
  gap: 5px;

  em {
    font-style: normal;
    font-size: 12.5px;
    color: var(--mkw-text-secondary);
    margin-left: 6px;
  }

  span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #a9c8ee;
    animation: chat-blink 1.2s infinite ease-in-out;

    &:nth-child(2) {
      animation-delay: 0.15s;
    }

    &:nth-child(3) {
      animation-delay: 0.3s;
    }
  }
}

@keyframes chat-blink {
  0%,
  80%,
  100% {
    opacity: 0.35;
    transform: translateY(0);
  }

  40% {
    opacity: 1;
    transform: translateY(-3px);
  }
}

/* ---- 推理链路 ---- */
.reason-trace {
  list-style: none;
  margin: 0;
  padding: 0 0 0 6px;
}

.reason-trace__item {
  position: relative;
  display: flex;
  gap: 12px;
  padding: 0 0 14px 6px;

  &::before {
    content: '';
    position: absolute;
    left: 17px;
    top: 24px;
    bottom: 0;
    width: 1px;
    background: #dbe7f8;
  }

  &:last-child::before {
    display: none;
  }
}

.reason-trace__step {
  position: relative;
  z-index: 1;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--mkw-gradient-primary);
  color: #fff;
  font-size: 12px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
}

.reason-trace__body {
  min-width: 0;
  flex: 1 1 auto;
}

.reason-trace__head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.reason-trace__title {
  font-size: 13px;
  font-weight: 700;
  color: var(--mkw-text);
}

.reason-trace__action {
  font-size: 11px;
  color: #7ea6d8;
  font-family: Consolas, Monaco, monospace;
  background: #eef5ff;
  padding: 1px 7px;
  border-radius: 20px;
}

.reason-trace__ms {
  font-size: 11px;
  color: var(--mkw-text-secondary);
  margin-left: auto;
}

.reason-trace__detail {
  font-size: 12.5px;
  color: var(--mkw-text-regular);
  line-height: 1.7;
  margin-top: 3px;
}

/* ---- 知识溯源 ---- */
.evidences {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.evidence {
  padding: 10px 12px;
  border-radius: 10px;
  background: #ffffff;
  border: 1px solid #e8effb;
  transition: background 0.3s ease;
}

.evidence.is-flash {
  background: #eaf4ff;
  border-color: #a9d3f7;
}

.evidence__head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.evidence__ref {
  font-size: 11px;
  font-weight: 700;
  color: #2b8fe8;
  background: #e8f3ff;
  border-radius: 8px;
  padding: 1px 7px;
}

.evidence__triple {
  font-size: 13px;
  color: var(--mkw-text);
  flex: 1 1 auto;
  min-width: 0;

  em {
    font-style: normal;
    color: var(--mkw-primary);
    margin: 0 2px;
  }
}

.evidence__types {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 7px;
  flex-wrap: wrap;
}

.evidence__arrow {
  color: #b9c9e0;
  font-size: 12px;
}

.evidence__rel {
  font-size: 11px;
  color: #8b95a8;
  font-family: Consolas, Monaco, monospace;
}

.evidence__sources {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed #e8effb;
}

.evidence__source {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: var(--mkw-primary);
  text-decoration: none;

  em {
    font-style: normal;
    color: var(--mkw-text-secondary);
  }

  &:hover {
    text-decoration: underline;
  }
}

/* ---- 底部操作 ---- */
.chat-msg__actions {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px dashed #e8effb;
}

.chat-msg__action {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12.5px;
  color: var(--mkw-text-secondary);
  cursor: pointer;
  transition: color 0.2s ease;
}

.chat-msg__action:hover {
  color: var(--mkw-primary);
}

.chat-msg__action.is-active {
  color: #31a05f;
  font-weight: 600;
}

.chat-msg__latency {
  margin-left: auto;
  font-size: 11.5px;
  color: var(--mkw-text-placeholder);
}

/* ---- 推荐问题 ---- */
.chat-view__suggest {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}

.chat-view__suggest-chip {
  font-size: 12.5px;
  color: var(--mkw-primary);
  background: #e8f3ff;
  border-radius: 20px;
  padding: 6px 14px;
  cursor: pointer;
  transition: background 0.2s ease, transform 0.2s ease;
}

.chat-view__suggest-chip:hover {
  background: #d3e8ff;
  transform: translateY(-1px);
}

/* ---- 输入区 ---- */
.chat-view__input-row {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  padding: 14px 18px;
}

.chat-view__input {
  flex: 1 1 auto;

  :deep(.el-textarea__inner) {
    border-radius: 22px !important;
    padding: 10px 16px;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 0 0 1px #e2ecfa inset;
  }
}

.chat-view__send {
  flex: 0 0 auto;
  height: 42px;
  padding: 0 24px;
}

.chat-view__tip {
  margin: 10px 0 0;
  text-align: center;
  font-size: 12px;
  color: var(--mkw-text-secondary);
}

@media (max-width: 900px) {
  .chat-msg__body {
    max-width: 92%;
  }
}
</style>
