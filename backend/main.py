# -*- coding: utf-8 -*-
"""
医数智答 · 智愈医典 —— FastAPI 应用入口
========================================
启动方式：
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
或：
    python main.py

主要职责：
  1. 创建 FastAPI 应用并挂载 `/api/v1` 路由
  2. 配置 CORS（前端 Vue3 跨域）
  3. 配置全局限流（slowapi，60 req/min/IP）与安全响应头
  4. 启动/关闭钩子：预热单例（Neo4j / 模型 / RAG 模板库），优雅关闭连接
  5. 统一异常处理（任何未捕获异常都返回结构化 JSON，不泄露堆栈）
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.app_layer.api.router import api_router
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

START_TIME = time.time()

# ---------------------------------------------------------------------------
#  限流（slowapi，可选依赖）
# ---------------------------------------------------------------------------
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
    )
    HAS_SLOWAPI = True
except Exception:  # pragma: no cover
    limiter = None  # type: ignore
    HAS_SLOWAPI = False


# ---------------------------------------------------------------------------
#  生命周期
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用启动/关闭钩子"""
    logger.info("=" * 78)
    logger.info("  %s v%s 启动中…", settings.APP_NAME, settings.APP_VERSION)
    logger.info("  参赛赛道：大模型与智能体应用赛道")
    logger.info("  产品定位：医疗大数据 + 知识图谱 + RAG 大模型的专业医疗知识问答平台")
    logger.info("=" * 78)

    # ---- 预热各层单例（失败不阻塞启动）----
    try:
        from app.kg_layer.memory_store import MemoryGraphStore

        store = MemoryGraphStore.instance()
        logger.info("① 数据/图谱层就绪：内存图谱节点 %d，关系 %d，疾病 %d",
                    store.node_count, store.edge_count, len(store.diseases))
    except Exception as exc:  # noqa: BLE001
        logger.error("内存图谱加载失败：%s", exc)

    try:
        from app.kg_layer.neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        logger.info("② Neo4j 状态：%s（%s）", client.status, settings.NEO4J_URI)
        if client.using_fallback:
            logger.warning("   Neo4j 不可用，已降级到内存图存储（离线演示模式）")
    except Exception as exc:  # noqa: BLE001
        logger.error("Neo4j 客户端初始化失败：%s", exc)

    try:
        from app.llm_layer.intent_classifier import get_intent_classifier
        from app.kg_layer.entity_extractor import get_entity_extractor
        from app.kg_layer.relation_extractor import get_relation_extractor

        logger.info("③ 实体识别后端：%s", get_entity_extractor().backend)
        logger.info("④ 关系抽取后端：%s（改进 PCNN + 医疗关键词注意力）",
                    get_relation_extractor().backend)
        logger.info("⑤ 意图分类后端：%s", get_intent_classifier().backend)
    except Exception as exc:  # noqa: BLE001
        logger.error("NLP 模型初始化失败：%s", exc)

    try:
        from app.llm_layer.rag.prompt_templates import get_prompt_library

        lib = get_prompt_library()
        v = lib.validate()
        logger.info("⑥ 医疗 RAG 模板库：%d 个模板，%d 类意图（校验：%s）",
                    len(lib.templates), len(lib.by_intent),
                    "通过" if v["ok"] else f"缺少 {v['missing_required']}")
    except Exception as exc:  # noqa: BLE001
        logger.error("RAG 模板库加载失败：%s", exc)

    try:
        from app.llm_layer.llm_client import get_llm_client

        info = get_llm_client().info()
        logger.info("⑦ 大模型：provider=%s, model=%s（%s）",
                    info["provider"], info["model"], info["mode"])
        if info["mode"] == "offline_template":
            logger.warning("   当前为完全离线模板模式；如需接入 Llama 3，"
                           "请修改 .env 的 LLM_PROVIDER 与 LLM_BASE_URL")
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM 客户端初始化失败：%s", exc)

    logger.info("-" * 78)
    logger.info("  接口文档：http://127.0.0.1:%d/docs", settings.PORT)
    logger.info("  ⚠️  本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。")
    logger.info("-" * 78)

    yield

    # ---- 关闭 ----
    logger.info("正在关闭 %s …", settings.APP_NAME)
    try:
        from app.kg_layer.neo4j_client import get_neo4j_client, reset_neo4j_client

        get_neo4j_client().close()
        reset_neo4j_client()
        logger.info("Neo4j 连接已关闭")
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 Neo4j 连接时出现异常（忽略）：%s", exc)
    logger.info("已停止。")


# ---------------------------------------------------------------------------
#  应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title=f"{settings.APP_NAME} API",
    description=(
        "## 医数智答 · 智愈医典\n"
        "**医疗大数据知识问答系统** —— 大模型与智能体应用赛道\n\n"
        "基于「医疗大数据 + 知识图谱 + RAG 大模型」的专业医疗知识问答平台。\n\n"
        "### 四层技术架构\n"
        "1. **数据层**：Scrapy 爬虫 + PubMed API → 清洗整合 → 全链路脱敏加密\n"
        "2. **知识图谱层**：BERT-BiLSTM-CRF + ★改进 PCNN（医疗关键词注意力）★ + Neo4j 5.8 + Jena + GAT/GCN\n"
        "3. **LLM 增强层**：TextCNN 意图分类 + 实体链接 + ★医疗专属 RAG（100+ 模板）★ + Llama 3\n"
        "4. **应用层**：FastAPI 异步接口 + Vue3 + Element Plus + D3.js + PyEcharts\n\n"
        "### 四大核心功能\n"
        "| 功能 | 接口 | 定位 |\n"
        "|------|------|------|\n"
        "| 知识图谱可视化 | `GET /api/v1/graph/subgraph` | 医学知识的「全景地图」 |\n"
        "| 智能疾病查询 | `GET /api/v1/disease/search` | 精准权威的「掌上医典」 |\n"
        "| 智能问答交互 | `POST /api/v1/qa/ask` | 可解释的「AI 医生」 |\n"
        "| 多维数据分析 | `POST /api/v1/analytics/risk/predict` | 预见未来的「健康预警」 |\n\n"
        "### 三大核心创新点\n"
        "1. **改进 PCNN 关系抽取**：新增医疗领域关键词注意力机制，召回率 +8.3%\n"
        "2. **医疗专属 RAG 提示工程**：100+ 医疗 prompt 模板 + KG 三元组约束，准确率 +12.1%\n"
        "3. **轻量化知识图谱可视化整合**：一键检索 + 详情联动，兼顾专业性与易用性\n\n"
        "---\n"
        "### ⚠️ 重要免责声明\n"
        "**本系统为演示原型。所有 AI 输出仅供参考，不能替代执业医师诊断。**\n"
        "出现急危重症表现请立即拨打 120 或前往急诊。"
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={"name": "医数智答项目组", "url": "https://github.com/"},
    license_info={"name": "仅供参赛演示使用"},
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms"],
)

# ---- 限流 ----
if HAS_SLOWAPI and limiter is not None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore
    app.add_middleware(SlowAPIMiddleware)
    logger.info("全局限流已启用：%d 次/分钟/IP", settings.RATE_LIMIT_PER_MINUTE)
else:
    logger.warning("未安装 slowapi，全局限流未启用（pip install slowapi）")


# ---------------------------------------------------------------------------
#  中间件：请求耗时 + 安全响应头
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_and_security_headers(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    cost_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{cost_ms:.2f}"
    # 注意：HTTP 响应头必须是 latin-1 可编码的 ASCII，中文应用名会触发
    # UnicodeEncodeError，因此这里使用 ASCII 化的标识，中文名放在响应体里。
    response.headers["X-Powered-By"] = "Zhiyu-Medical-Codex"
    # 安全响应头
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "no-referrer"
    if cost_ms > 3000:
        logger.warning("慢请求：%s %s 耗时 %.0fms", request.method, request.url.path, cost_ms)
    return response


# ---------------------------------------------------------------------------
#  异常处理
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """参数校验失败 → 结构化错误（含中文字段说明）"""
    errors = []
    for e in exc.errors():
        loc = " → ".join(str(x) for x in e.get("loc", []) if x != "body")
        errors.append({"field": loc, "message": e.get("msg", ""), "type": e.get("type", "")})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": 422,
            "message": "请求参数校验失败",
            "errors": errors,
            "hint": "请检查请求体字段类型与取值范围",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """未捕获异常 → 结构化 JSON（不向前端泄露堆栈）"""
    logger.error("未捕获异常 %s %s：%s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": 500,
            "message": "服务内部错误，请稍后重试",
            "detail": str(exc) if settings.DEBUG else None,
            "path": request.url.path,
        },
    )


# ---------------------------------------------------------------------------
#  挂载路由
# ---------------------------------------------------------------------------
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["系统"], summary="服务根路径", include_in_schema=False)
async def root() -> Dict[str, Any]:
    return {
        "name": settings.APP_NAME,
        "product": "智愈医典",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.API_PREFIX,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "server_time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "disclaimer": "本系统为演示原型，所有 AI 输出仅供参考，不能替代执业医师诊断。",
    }


# ---------------------------------------------------------------------------
#  直接运行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
