# -*- coding: utf-8 -*-
"""
全局配置（pydantic-settings）
============================
所有配置项均可通过 `.env` 文件或环境变量覆盖，优先级：环境变量 > .env > 默认值。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 后端根目录（backend/）
BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", str(BASE_DIR / ".env")),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------- 应用 -------------------------
    APP_NAME: str = "医数智答·智愈医典"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "http://127.0.0.1:5173,http://localhost:5173"

    # ------------------------- Neo4j -------------------------
    NEO4J_URI: str = "bolt://127.0.0.1:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "medical123"
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = 50
    NEO4J_CONNECTION_TIMEOUT: int = 15
    NEO4J_FALLBACK_TO_MEMORY: bool = True

    # ------------------------- LLM -------------------------
    LLM_PROVIDER: str = "template"  # openai_compatible | ollama | template
    LLM_BASE_URL: str = "http://127.0.0.1:11434/v1"
    LLM_MODEL: str = "llama3:8b-instruct-q4_K_M"
    LLM_API_KEY: str = "ollama"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1600
    LLM_TIMEOUT: int = 60
    LLM_LORA_PATH: str = ""

    # ------------------------- 模型路径 -------------------------
    MODEL_DIR: str = "./models"
    NER_MODEL_PATH: str = "./models/ner_bert_bilstm_crf"
    PCNN_MODEL_PATH: str = "./models/pcnn_attention"
    INTENT_MODEL_PATH: str = "./models/intent_textcnn"
    RISK_MODEL_PATH: str = "./models/risk_xgboost.json"
    EMBEDDING_MODEL_PATH: str = "./models/text2vec-base-chinese"
    BERT_BASE_PATH: str = "./models/bert-base-chinese"

    # ------------------------- RAG -------------------------
    RAG_TOP_K: int = 12
    RAG_MAX_HOPS: int = 2
    RAG_MIN_CONFIDENCE: float = 0.55
    RAG_HALLUCINATION_GUARD: bool = True
    RAG_PROMPT_LIBRARY: str = "./app/llm_layer/rag/prompt_library.yaml"

    # ------------------------- 数据层 -------------------------
    PUBMED_API_KEY: str = ""
    PUBMED_EMAIL: str = "medical-kg@example.com"
    PUBMED_TOOL: str = "ZhiyuMedicalCodex"
    CRAWLER_MAX_PAGES: int = 20
    CRAWLER_DELAY: float = 1.0
    CRAWLER_CONCURRENT: int = 8
    USER_AGENT_ROTATE: bool = True
    CORPUS_DIR: str = "./data/processed"

    # ------------------------- 隐私合规 -------------------------
    FIELD_ENCRYPT_KEY: str = ""
    PRIVACY_MASK_ENABLED: bool = True
    PRIVACY_ENCRYPT_ENABLED: bool = True

    # ------------------------- 安全 -------------------------
    RATE_LIMIT_PER_MINUTE: int = 60
    CYPHER_READONLY_WHITELIST: bool = True

    # ------------------------- 日志 -------------------------
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"

    # ------------------------------------------------------------------
    @field_validator("CORS_ORIGINS")
    @classmethod
    def _strip_origins(cls, v: str) -> str:
        return v.strip()

    # ------------------------- 便捷属性 -------------------------
    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    def abspath(self, relative: str) -> Path:
        """把配置里的相对路径解析为相对 backend/ 的绝对路径"""
        p = Path(relative)
        return p if p.is_absolute() else (BASE_DIR / p).resolve()

    @property
    def seed_dir(self) -> Path:
        return BASE_DIR / "data" / "seed"

    @property
    def corpus_dir(self) -> Path:
        return self.abspath(self.CORPUS_DIR)

    @property
    def log_dir(self) -> Path:
        return self.abspath(self.LOG_DIR)

    @property
    def prompt_library_path(self) -> Path:
        return self.abspath(self.RAG_PROMPT_LIBRARY)

    @property
    def disclaimer(self) -> str:
        return "本回答由 AI 生成，仅供参考，不能替代执业医师诊断。"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """全局单例配置"""
    s = Settings()
    s.log_dir.mkdir(parents=True, exist_ok=True)
    s.corpus_dir.mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
