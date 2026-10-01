# -*- coding: utf-8 -*-
"""
RAG 子包（app.llm_layer.rag）
=============================

"智愈医典" 医疗 RAG 流水线的四个模块 + 统一出口：

  * `retriever.py`           混合检索器 —— 以**知识图谱三元组**为检索单元（五路召回 + 统一排序）
  * `context_builder.py`     上下文构建器 —— 三元组槽位化 / 编号化 / 内联来源，产出 prompt 变量
  * `hallucination_guard.py` 幻觉守卫 —— 7 项确定性事实校验 + 纯 KG 兜底回答
  * `rag_engine.py`          引擎编排 —— 意图 → 实体 → 检索 → 上下文 → prompt → LLM → 守卫 → 置信度
  * `prompt_templates.py`    医疗专属 prompt 模板库（100+ 模板，YAML 驱动）

对外只暴露 `get_rag_engine` 与 `RagEngine`，其余类型按需从子模块导入。
"""
from __future__ import annotations

from app.llm_layer.rag.rag_engine import RagEngine, get_rag_engine

__all__ = ["get_rag_engine", "RagEngine"]
