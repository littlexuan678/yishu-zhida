# -*- coding: utf-8 -*-
"""中文文本处理工具：分词、归一化、相似度"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Set

try:  # jieba 为可选依赖
    import jieba  # type: ignore

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False

#: 中文停用词（医疗检索场景）
STOPWORDS: Set[str] = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "会", "着", "没有", "看", "好", "自己",
    "这", "那", "什么", "怎么", "如何", "哪些", "请问", "一下", "可以", "需要", "应该",
    "吗", "呢", "吧", "啊", "呀", "请问", "帮我", "想知道", "了解", "关于", "对于",
}

_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+")


def normalize(text: str) -> str:
    """全角转半角 + 去除多余空白 + 统一大小写"""
    if not text:
        return ""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    s = "".join(out)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_punct(text: str) -> str:
    return _PUNCT_RE.sub(" ", text or "").strip()


def tokenize(text: str) -> List[str]:
    """中文分词，返回去停用词后的词表"""
    text = normalize(text)
    if not text:
        return []
    if _HAS_JIEBA:
        words = [w.strip() for w in jieba.lcut(text)]
    else:
        # 无 jieba 时的降级：单字 + 2-gram
        chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        words = list(chars)
        words += ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
    return [w for w in words if w and w not in STOPWORDS and len(w) > 0]


def keywords(text: str, topk: int = 8) -> List[str]:
    """粗粒度关键词抽取：按词长加权排序"""
    toks = tokenize(text)
    seen: dict = {}
    for t in toks:
        if len(t) < 2:
            continue
        seen[t] = seen.get(t, 0) + len(t)
    return [w for w, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:topk]]


def similarity(a: str, b: str) -> float:
    """字符串相似度 0~1（含子串加成，用于实体模糊匹配）"""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.82 + 0.16 * (min(len(a), len(b)) / max(len(a), len(b)))
    return SequenceMatcher(None, a, b).ratio()


def best_match(query: str, candidates: Iterable[str], threshold: float = 0.55):
    """在候选集中找最相似的字符串，返回 (候选, 分数) 或 (None, 0.0)"""
    best, best_score = None, 0.0
    for c in candidates:
        s = similarity(query, c)
        if s > best_score:
            best, best_score = c, s
    if best_score >= threshold:
        return best, best_score
    return None, 0.0


def truncate(text: str, max_len: int = 60, suffix: str = "…") -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + suffix


def split_sentences(text: str) -> List[str]:
    """中文分句"""
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；!?;\n])", text)
    return [p.strip() for p in parts if p and p.strip()]
