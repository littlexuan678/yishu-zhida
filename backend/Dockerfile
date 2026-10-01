# =============================================================================
#  医数智答 · 智愈医典 —— 后端 Docker 镜像（Python 3.9）
#  构建：docker build -t zhiyu-backend:1.0.0 .
#  说明：基础镜像严格使用 Python 3.9（与 PPT 指定环境一致）
# =============================================================================
FROM python:3.9-slim

LABEL maintainer="医数智答项目组" \
      description="智愈医典 · 医疗大数据知识问答系统 —— 后端服务（FastAPI + Neo4j + RAG）" \
      version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8

# ---------------------------------------------------------------------------
#  系统依赖
#  lxml/Scrapy 需要编译工具链；curl 用于健康检查
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
#  依赖安装（分层缓存：先装依赖，再拷代码）
#  说明：
#    * 演示部署可用 requirements-lite.txt（不含 torch，安装快、体积小）
#    * 需要训练/推理深度学习模型时使用完整 requirements.txt
# ---------------------------------------------------------------------------
COPY requirements.txt requirements-lite.txt* ./
ARG INSTALL_PROFILE=lite
RUN if [ "$INSTALL_PROFILE" = "full" ]; then \
        pip install --upgrade pip && pip install -r requirements.txt ; \
    elif [ -f requirements-lite.txt ]; then \
        pip install --upgrade pip && pip install -r requirements-lite.txt ; \
    else \
        pip install --upgrade pip && \
        grep -v -E '^(torch|transformers|tokenizers|sentencepiece|datasets|accelerate|peft|sentence-transformers|langchain|openai|#)' requirements.txt > /tmp/req.txt && \
        pip install -r /tmp/req.txt ; \
    fi

# ---------------------------------------------------------------------------
#  应用代码
# ---------------------------------------------------------------------------
COPY . .

# 运行期目录
RUN mkdir -p /app/logs /app/data/processed /app/models

# 非 root 用户运行（安全加固）
RUN useradd -m -u 10001 zhiyu && chown -R zhiyu:zhiyu /app
USER zhiyu

EXPOSE 8000

# ---------------------------------------------------------------------------
#  启动
#  生产模式关闭 --reload，开启多 worker（异步接口 + 4 worker 可支撑并发）
#  注意：多 worker 下每个进程各持一份内存图谱（只读，约 20MB/进程），可接受
# ---------------------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]

# 健康检查（与 docker-compose 中的 healthcheck 互补）
HEALTHCHECK --interval=20s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1
