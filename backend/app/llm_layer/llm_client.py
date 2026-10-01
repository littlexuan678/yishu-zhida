# -*- coding: utf-8 -*-
"""
大语言模型客户端（Llama 3 接入层）
==================================
统一封装三种接入方式（由 `.env` 的 `LLM_PROVIDER` 决定）：

  ① `openai_compatible`  —— 任意兼容 OpenAI `/v1/chat/completions` 协议的端点
       * vLLM   : python -m vllm.entrypoints.openai.api_server --model Meta-Llama-3-8B-Instruct
       * DeepSeek: https://api.deepseek.com/v1
       * 通义/智谱/Kimi 等国内厂商
  ② `ollama`             —— 本地 Ollama（llama3:8b-instruct-q4_K_M）
  ③ `template`           —— **完全离线模板生成**（无大模型）
       直接以知识图谱三元组拼装结构化答案，零依赖、零幻觉，
       仍具备完整可解释性与溯源能力（答辩演示保底方案）。

设计要点
--------
* **永不抛异常**：任何失败都返回 `LLMResult(success=False, error=...)`，
  由 `rag_engine` 决定降级策略，保证问答服务不中断。
* **异步**：使用 `httpx.AsyncClient`，复用连接池，避免每次请求新建 TCP。
* **LangChain 兼容**：`as_langchain_llm()` 返回 LangChain `BaseLLM`
  适配器，便于用 LangChain 的 Chain/Agent 编排（PPT 指定框架）。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import httpx

    HAS_HTTPX = True
except Exception:  # pragma: no cover
    HAS_HTTPX = False
    httpx = None  # type: ignore

try:
    from openai import AsyncOpenAI  # type: ignore

    HAS_OPENAI = True
except Exception:  # pragma: no cover
    HAS_OPENAI = False
    AsyncOpenAI = None  # type: ignore


#: 系统提示词（所有请求共用，与 prompt_library 的 global_constraints 互补）
SYSTEM_PROMPT = (
    "你是\"智愈医典\"医疗知识问答系统的 AI 医生助手，服务于普通患者与基层医生。\n"
    "你的回答必须严格基于用户提供的【知识图谱事实】，做到准确、可溯源、可解释。\n"
    "你不得使用自身记忆补充任何医学事实，不得给出具体的处方与用药剂量，"
    "不得给出确定性诊断结论。\n"
    "所有 AI 输出仅供参考，不能替代执业医师诊断。"
)


@dataclass
class LLMResult:
    """统一的大模型调用结果"""

    text: str = ""
    model: str = ""
    provider: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = False
    error: str = ""
    finish_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_len": len(self.text), "model": self.model, "provider": self.provider,
            "usage": self.usage, "latency_ms": self.latency_ms,
            "success": self.success, "error": self.error,
        }


class LLMClient:
    """
    大语言模型统一客户端
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.provider = (provider or settings.LLM_PROVIDER or "template").lower()
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.api_key = api_key or settings.LLM_API_KEY
        self.timeout = timeout or settings.LLM_TIMEOUT
        self._client = None
        self._init_client()

    # ------------------------------------------------------------------
    def _init_client(self) -> None:
        if self.provider == "template":
            logger.info("LLM 提供方 = template（完全离线模板生成模式，不调用任何大模型）")
            return

        if self.provider == "ollama" and not self.base_url.endswith("/v1"):
            # Ollama 原生 /api/chat 与 OpenAI 兼容 /v1/chat/completions 都支持，
            # 这里统一走 OpenAI 兼容协议
            self.base_url = self.base_url + "/v1" if not self.base_url.endswith("/v1") else self.base_url

        if HAS_OPENAI:
            try:
                self._client = AsyncOpenAI(
                    base_url=self.base_url, api_key=self.api_key or "sk-none",
                    timeout=float(self.timeout),
                )
                logger.info("LLM 客户端初始化完成（openai sdk）：provider=%s, model=%s, base=%s",
                            self.provider, self.model, self.base_url)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("初始化 openai sdk 失败：%s，回落到 httpx 直连", exc)

        if HAS_HTTPX:
            logger.info("LLM 客户端初始化完成（httpx）：provider=%s, model=%s, base=%s",
                        self.provider, self.model, self.base_url)
        else:
            logger.error("既无 openai 也无 httpx，LLM 调用将全部失败并降级为模板生成")

    # ==================================================================
    #  主调用
    # ==================================================================
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: str = "",
        stop: Optional[Sequence[str]] = None,
    ) -> LLMResult:
        """生成回答（永不抛异常）"""
        t0 = time.perf_counter()
        temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        sys_prompt = system or SYSTEM_PROMPT

        if self.provider == "template":
            return LLMResult(
                text="", model="template", provider="template",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                success=False, error="provider=template，由 rag_engine 使用本地模板生成器",
            )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # ---- 优先走 openai sdk ----
        if self._client is not None:
            try:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self.model, messages=messages,
                        temperature=float(temperature), max_tokens=int(max_tokens),
                        stop=list(stop) if stop else None,
                    ),
                    timeout=float(self.timeout),
                )
                text = (resp.choices[0].message.content or "").strip()
                usage = {}
                if getattr(resp, "usage", None):
                    usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
                return LLMResult(
                    text=text, model=self.model, provider=self.provider, usage=usage,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                    success=bool(text), finish_reason=getattr(resp.choices[0], "finish_reason", "") or "",
                )
            except asyncio.TimeoutError:
                err = f"LLM 调用超时（{self.timeout}s）"
                logger.error(err)
                return LLMResult(model=self.model, provider=self.provider, success=False,
                                 error=err, latency_ms=round((time.perf_counter() - t0) * 1000, 2))
            except Exception as exc:  # noqa: BLE001
                logger.error("LLM 调用失败（openai sdk）：%s", exc)
                # 继续尝试 httpx 兜底
                if not HAS_HTTPX:
                    return LLMResult(model=self.model, provider=self.provider, success=False,
                                     error=str(exc), latency_ms=round((time.perf_counter() - t0) * 1000, 2))

        # ---- httpx 直连兜底 ----
        if not HAS_HTTPX:
            return LLMResult(model=self.model, provider=self.provider, success=False,
                             error="未安装 httpx/openai，无法调用大模型",
                             latency_ms=round((time.perf_counter() - t0) * 1000, 2))

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'sk-none'}",
        }
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": float(temperature), "max_tokens": int(max_tokens),
        }
        if stop:
            payload["stop"] = list(stop)

        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            text = ""
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                text = (msg.get("content") or "").strip()
            return LLMResult(
                text=text, model=data.get("model", self.model), provider=self.provider,
                usage=data.get("usage") or {},
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                success=bool(text),
                finish_reason=(choices[0].get("finish_reason") if choices else "") or "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM 调用失败（httpx）：%s", exc)
            return LLMResult(model=self.model, provider=self.provider, success=False, error=str(exc),
                             latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    # ==================================================================
    #  流式调用（供 SSE 接口使用）
    # ==================================================================
    async def stream_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: str = "",
    ):
        """
        异步生成器，逐块 yield 文本片段。
        任何异常都会结束生成（调用方负责拼接已生成内容）。
        """
        if self.provider == "template" or self._client is None:
            # 模板模式：一次性 yield（上层会对空流做降级）
            return
        temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            stream = await self._client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=float(temperature), max_tokens=int(max_tokens), stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM 流式调用失败：%s", exc)
            return

    # ==================================================================
    #  健康检查
    # ==================================================================
    async def ping(self) -> Dict[str, Any]:
        """探活：发一条极短的请求验证端点可用"""
        if self.provider == "template":
            return {"ok": True, "provider": "template", "note": "离线模板模式，无需探活"}
        res = await self.generate("你好", temperature=0.0, max_tokens=8)
        return {
            "ok": res.success, "provider": self.provider, "model": self.model,
            "base_url": self.base_url, "latency_ms": res.latency_ms,
            "error": res.error,
        }

    # ==================================================================
    #  LangChain 适配器（PPT 指定 LangChain 框架）
    # ==================================================================
    def as_langchain_llm(self):
        """
        返回一个 LangChain 兼容的 LLM 适配器。

        用法：
            from app.llm_layer.llm_client import get_llm_client
            from langchain.chains import LLMChain
            llm = get_llm_client().as_langchain_llm()
            chain = LLMChain(llm=llm, prompt=my_prompt)

        说明：LangChain 的 LLM 接口是同步的，这里用 `asyncio.run` 包装异步调用；
        在已有事件循环的环境中（FastAPI），请直接使用本类的 `generate()`。
        """
        try:
            from langchain_core.language_models.llms import LLM  # type: ignore
            from langchain_core.callbacks import CallbackManagerForLLMRun  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "未安装 langchain-core，无法创建 LangChain 适配器。"
                "请执行：pip install langchain langchain-community"
            ) from exc

        client = self

        class _ZhiyuLangChainLLM(LLM):  # type: ignore[misc]
            """智愈医典 LLM 的 LangChain 适配器"""

            @property
            def _llm_type(self) -> str:
                return f"zhiyu-medical-{client.provider}"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional["CallbackManagerForLLMRun"] = None,
                **kwargs: Any,
            ) -> str:
                res = asyncio.run(client.generate(prompt, stop=stop))
                if run_manager is not None:
                    run_manager.on_llm_end(res.text)  # type: ignore[arg-type]
                return res.text

        return _ZhiyuLangChainLLM()

    # ==================================================================
    def info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "lora_path": settings.LLM_LORA_PATH or None,
            "has_openai_sdk": HAS_OPENAI,
            "has_httpx": HAS_HTTPX,
            "mode": "offline_template" if self.provider == "template" else "remote_llm",
        }


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm_client() -> None:
    global _client
    _client = None
