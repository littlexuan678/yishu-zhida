# =============================================================================
#  医数智答 · 智愈医典 —— 前端 Docker 镜像（Vue3 → Nginx 静态托管）
#  构建：docker build -t zhiyu-frontend:1.0.0 .
#  说明：多阶段构建；Nginx 同时承担 /api 反向代理，规避跨域
# =============================================================================

# ------------------------- 阶段一：构建 -------------------------
FROM node:18-alpine AS builder

WORKDIR /app

# 依赖层（利用 Docker 缓存）
COPY package.json pnpm-lock.yaml* .npmrc* ./
RUN corepack enable && corepack prepare pnpm@latest --activate \
    && (pnpm install --frozen-lockfile || pnpm install)

# 源码层
COPY . .

# 构建期注入 API 基地址（Vite 变量在 build 阶段生效，运行时不可改）
ARG VITE_API_BASE=/api/v1
ENV VITE_API_BASE=$VITE_API_BASE

RUN pnpm run build

# ------------------------- 阶段二：运行 -------------------------
FROM nginx:1.25-alpine

LABEL description="智愈医典 · 医疗知识信息问答系统 —— 前端（Vue3 + Element Plus + D3.js + ECharts）" \
      version="1.0.0"

# 自定义 Nginx 配置：SPA history 路由回退 + /api 反向代理到后端
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 静态产物
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
