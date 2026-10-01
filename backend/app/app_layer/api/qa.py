# -*- coding: utf-8 -*-
"""
智能问答接口（可解释 AI 医生）
==============================
对应 PPT 功能三：智能问答交互
  * 聊天对话界面 → `POST /qa/ask`
  * BERT 意图识别 + 实体链接 → 响应中的 `intent` / `entities`
  * 医疗专属 RAG → 响应中的 `prompt_template_id` / `kg_context` / `evidences`
  * 推理链路溯源 → 响应中的 `reasoning_trace`
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.app_layer.services.qa_service import QAService
from app.deps import get_graph, get_intent, get_linker
from app.kg_layer.graph_service import GraphService
from app.llm_layer.intent_classifier import IntentClassifier
from app.llm_layer.entity_linker import EntityLinker
from app.schemas import (
    QAHistoryResponse,
    QARequest,
    QAResponse,
    PromptTemplateItem,
    PromptTemplateListResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/qa", tags=["智能问答 · 可解释 AI 医生"])

#: 问答服务单例（依赖注入的服务在此组装）
_qa_service: Optional[QAService] = None


def get_qa_service(
    graph: GraphService = Depends(get_graph),
    intent: IntentClassifier = Depends(get_intent),
    linker: EntityLinker = Depends(get_linker),
) -> QAService:
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService(graph=graph, intent=intent, linker=linker)
    else:
        # 依赖可能被测试替换，保持引用最新
        _qa_service.graph = graph
        _qa_service.intent = intent
        _qa_service.linker = linker
    return _qa_service


# =============================================================================
#  POST /qa/ask —— 医疗问答主接口
# =============================================================================
@router.post(
    "/ask",
    response_model=QAResponse,
    summary="医疗知识问答（可解释、可溯源）",
    description=(
        "完整 RAG 流程：TextCNN 意图识别 → BERT 实体链接 → Neo4j 图谱检索 → "
        "医疗专属 Prompt 渲染 → Llama 3 生成 → 幻觉守卫校验。\n\n"
        "响应包含完整**推理链路**（`reasoning_trace`）与**知识溯源**（`evidences`），"
        "每条陈述均带 `[KG-n]` 引用编号，杜绝黑盒输出。\n\n"
        "⚠️ 所有 AI 输出仅供参考，不能替代执业医师诊断。"
    ),
)
async def ask_question(
    payload: QARequest,
    service: QAService = Depends(get_qa_service),
) -> QAResponse:
    try:
        resp = await service.ask(
            question=payload.question,
            session_id=payload.session_id,
            top_k=payload.top_k,
            max_hops=payload.max_hops,
            explain=payload.explain,
            use_llm=payload.use_llm,
        )
        return resp
    except Exception as exc:  # noqa: BLE001
        logger.error("问答接口异常：%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答服务异常：{exc}") from exc


# =============================================================================
#  POST /qa/ask/stream —— SSE 流式问答
# =============================================================================
@router.post(
    "/ask/stream",
    summary="医疗问答（SSE 流式输出）",
    description="以 Server-Sent Events 逐事件推送：meta（意图/实体）→ token（逐字）→ done（证据/推理链路）。",
)
async def ask_stream(
    payload: QARequest,
    service: QAService = Depends(get_qa_service),
) -> StreamingResponse:
    async def event_gen():
        try:
            # 先做一次非流式请求获取意图/实体/证据（保证流式失败时前端仍有完整信息）
            full = await service.ask(
                question=payload.question,
                session_id=payload.session_id,
                top_k=payload.top_k,
                max_hops=payload.max_hops,
                explain=payload.explain,
                use_llm=payload.use_llm,
            )
            meta = {
                "intent": full.intent.model_dump(),
                "entities": [e.model_dump() for e in full.entities],
                "prompt_template_id": full.prompt_template_id,
                "confidence": full.confidence,
            }
            yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

            # 逐段推送答案（按标点切分，模拟打字机效果；真实 LLM 流式见 llm_client.stream_generate）
            answer = full.answer or ""
            buf = ""
            for ch in answer:
                buf += ch
                if ch in "。！？\n；;":
                    yield f"event: token\ndata: {json.dumps({'t': buf}, ensure_ascii=False)}\n\n"
                    buf = ""
                    await asyncio.sleep(0.01)
            if buf:
                yield f"event: token\ndata: {json.dumps({'t': buf}, ensure_ascii=False)}\n\n"

            done = {
                "evidences": [e.model_dump() for e in full.evidences],
                "reasoning_trace": [r.model_dump() for r in full.reasoning_trace],
                "latency_ms": full.latency_ms.model_dump(),
                "related_questions": full.related_questions,
                "disclaimer": full.disclaimer,
            }
            yield f"event: done\ndata: {json.dumps(done, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("流式问答异常：%s", exc, exc_info=True)
            err = {"message": f"问答服务异常：{exc}"}
            yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# =============================================================================
#  GET /qa/prompt-templates —— 医疗 RAG prompt 模板库
# =============================================================================
@router.get(
    "/prompt-templates",
    response_model=PromptTemplateListResponse,
    summary="列出医疗专属 RAG prompt 模板",
    description="返回 100+ 医疗问答 prompt 模板的元信息（按意图分组），用于答辩展示与调参。",
)
async def list_prompt_templates(
    intent: Optional[str] = Query(None, description="按意图过滤，如 department_query"),
    keyword: Optional[str] = Query(None, description="按名称/标签关键词过滤"),
) -> PromptTemplateListResponse:
    try:
        from app.llm_layer.rag.prompt_templates import get_prompt_library

        lib = get_prompt_library()
        items = lib.list_meta(intent)
        if keyword:
            kw = keyword.lower()
            items = [i for i in items
                     if kw in i["name"].lower() or kw in i["id"].lower()
                     or any(kw in t.lower() for t in i.get("tags") or [])
                     or kw in (i.get("description") or "").lower()]
        return PromptTemplateListResponse(
            total=len(items),
            intents=sorted(lib.by_intent.keys()),
            items=[PromptTemplateItem(**i) for i in items],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("加载 prompt 模板库失败：%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"加载模板库失败：{exc}") from exc


# =============================================================================
#  GET /qa/history/{session_id} —— 会话历史
# =============================================================================
@router.get(
    "/history/{session_id}",
    response_model=QAHistoryResponse,
    summary="获取会话历史",
)
async def get_history(
    session_id: str,
    service: QAService = Depends(get_qa_service),
) -> QAHistoryResponse:
    msgs = service.history(session_id)
    return QAHistoryResponse(session_id=session_id, messages=msgs, turns=len(msgs) // 2)


# =============================================================================
#  DELETE /qa/history/{session_id} —— 清空会话（隐私合规）
# =============================================================================
@router.delete(
    "/history/{session_id}",
    summary="清空会话历史（隐私合规）",
)
async def clear_history(
    session_id: str,
    service: QAService = Depends(get_qa_service),
) -> Dict[str, Any]:
    service.clear(session_id)
    return {"code": 0, "message": "会话已清空", "session_id": session_id}


# =============================================================================
#  GET /qa/info —— 问答链路组件信息（调试/答辩）
# =============================================================================
@router.get("/info", summary="问答链路组件信息")
async def qa_info(service: QAService = Depends(get_qa_service)) -> Dict[str, Any]:
    info: Dict[str, Any] = {"service": service.info()}
    try:
        info["intent_classifier"] = service.intent.info()
    except Exception as exc:  # noqa: BLE001
        info["intent_classifier"] = {"error": str(exc)}
    try:
        info["entity_linker"] = service.linker.info()
    except Exception as exc:  # noqa: BLE001
        info["entity_linker"] = {"error": str(exc)}
    if service.rag is not None:
        try:
            info["rag_engine"] = service.rag.info()
        except Exception as exc:  # noqa: BLE001
            info["rag_engine"] = {"error": str(exc)}
    return info
