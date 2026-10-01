# =============================================================================
#  医数智答 · 智愈医典 —— 一体化单容器镜像（前端 + 后端）
# =============================================================================
#  构建：docker build -t zhiyu-all-in-one:1.0.0 .
#  运行：docker run -p 7860:7860 zhiyu-all-in-one:1.0.0
#  访问：http://127.0.0.1:7860/  （前端界面 + /api/v1 后端接口同域）
#
#  与 docker-compose.yml 的区别：
#    * compose 面向「完整开发环境」：Neo4j + 后端 + Nginx 前端，三容器
#    * 本文件面向「演示/云平台部署」：单容器，内置内存图谱（无需 Neo4j），
#      后端直接托管前端静态产物（STATIC_DIR 模式），监听 7860 端口
#      （7860 为 Hugging Face Spaces Docker SDK 约定端口，本地可任意映射）
# =============================================================================

# ------------------------- 阶段一：前端构建 -------------------------
FROM node:18-alpine AS frontend-builder

WORKDIR /build

# 依赖层（利用 Docker 缓存；pnpm-lock.yaml 锁定版本）
COPY frontend/package.json frontend/pnpm-lock.yaml* frontend/.npmrc* ./
RUN corepack enable && corepack prepare pnpm@latest --activate \
    && (pnpm install --frozen-lockfile || pnpm install)

# 源码层
COPY frontend/ .

# 构建期注入 API 基地址（同源部署，相对路径即可）
ARG VITE_API_BASE=/api/v1
ENV VITE_API_BASE=$VITE_API_BASE

RUN pnpm run build

# ------------------------- 阶段二：后端运行时 -------------------------
FROM python:3.9-slim

LABEL maintainer="医数智答项目组" \
      description="智愈医典 · 医疗大数据知识问答系统 —— 一体化演示镜像（FastAPI + Vue3 + 内存图谱 + 医疗RAG）" \
      version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8

# lxml/Scrapy 编译工具链 + curl（健康检查）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖（lite 版本：无 torch 等重型依赖，云平台可快速构建）
COPY backend/requirements.txt backend/requirements-lite.txt* ./
RUN pip install --upgrade pip && pip install -r requirements-lite.txt

# 后端代码（含 data/seed 种子数据 —— 内存图谱的数据源）
COPY backend/ .

# 前端构建产物 → 由 FastAPI 以 STATIC_DIR 模式托管
COPY --from=frontend-builder /build/dist /app/static

# 运行期环境：离线演示（内存图谱 + 模板 LLM + 规则模型降级）
ENV STATIC_DIR=/app/static \
    HOST=0.0.0.0 \
    PORT=7860 \
    DEBUG=false \
    NEO4J_FALLBACK_TO_MEMORY=true \
    LLM_PROVIDER=template \
    CORS_ORIGINS="*" \
    RATE_LIMIT_PER_MINUTE=60 \
    LOG_LEVEL=INFO \
    LOG_DIR=/app/logs

RUN mkdir -p /app/logs /app/data/processed /app/models \
    && useradd -m -u 1000 zhiyu && chown -R zhiyu:zhiyu /app
USER zhiyu

EXPOSE 7860

# 单 worker：演示流量足够，且避免多进程重复预热（内存图谱只读约 20MB）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]

HEALTHCHECK --interval=20s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -fsS http://127.0.0.1:7860/api/v1/health || exit 1
