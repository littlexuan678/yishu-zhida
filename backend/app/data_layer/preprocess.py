# -*- coding: utf-8 -*-
"""
数据清洗与整合（preprocess）
============================
把爬取到的异构原始医学文本，清洗为可用于实体/关系抽取的标准化语料。

处理步骤
--------
  1. **编码与噪声清洗**：全角半角统一、去除 HTML 标签/参考文献标记/控制字符
  2. **分句与去重**：中文分句，SimHash 近似去重（阈值可调）
  3. **医学术语归一化**：把口语/别名映射到图谱标准名（复用实体链接词典）
  4. **质量过滤**：过短、纯数字、无医学实体、广告性内容的样本直接丢弃
  5. **输出标准化语料**：JSONL，每行一条 `{"id","title","text","source","pmid",...}`

用法
----
    from app.data_layer.preprocess import Preprocessor
    pp = Preprocessor()
    clean = pp.clean_text(raw)
    sentences = pp.split_and_dedup(clean)
    records = pp.process_document({"title": ..., "abstract": ..., "pmid": ...})
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.config import settings
from app.utils.logger import get_logger
from app.utils.text import normalize, split_sentences, truncate

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
#  正则
# ---------------------------------------------------------------------------
_RE_HTML = re.compile(r"<[^>]+>")
_RE_SCRIPT = re.compile(r"(?:function|var|let|const|window\.|document\.)\s*\w*", re.I)
_RE_REF_MARK = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")
_RE_SUP = re.compile(r"[\u00b9\u00b2\u00b3\u2070-\u209f]")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_MULTI_SPACE = re.compile(r"[ \t\u00a0\u3000]{2,}")
_RE_MULTI_NL = re.compile(r"\n{3,}")
_RE_URL = re.compile(r"https?://\S+")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_COPYRIGHT = re.compile(r"(?:版权所有|Copyright|©|\(c\)\s*20\d\d|京ICP备)\S*", re.I)
_RE_AD = re.compile(r"(?:加微信|扫码关注|点击购买|在线咨询|免费领取|限时优惠|"
                    r"专家在线|预约挂号电话|广告|推广|微商|代理加盟)")
_RE_SECTION = re.compile(r"^(?:摘要|引言|方法|结果|讨论|结论|背景|目的|对象与方法|"
                         r"Abstract|Introduction|Methods|Results|Discussion|Conclusion)\s*[:：]?\s*$", re.I)

#: 医学实体后缀（用于质量过滤：文本是否含医学信息）
_RE_MED_TERM = re.compile(
    r"[\u4e00-\u9fff]{2,12}(?:病|症|炎|癌|瘤|综合征|综合症|感染|硬化|梗死|栓塞|"
    r"衰竭|障碍|畸形|损伤|溃疡|结石|囊肿|息肉|贫血|亢进|减退|中毒|"
    r"治疗|用药|药物|检查|诊断|手术|症状|体征|科室|指南|疗效|预后)"
)


# =============================================================================
#  SimHash 近似去重
# =============================================================================
class SimHash:
    """
    SimHash 64 位指纹 + 海明距离去重
    --------------------------------
    对中文按 2-gram 取 sha1 前 64 位作为特征位串，加权求和后二值化。
    海明距离 ≤ 3 视为重复（医学文献大量近似转载，必须去重）。
    """

    def __init__(self, bits: int = 64) -> None:
        self.bits = bits

    def _grams(self, text: str) -> List[str]:
        t = re.sub(r"\s+", "", text or "")
        if len(t) < 2:
            return [t] if t else []
        return [t[i : i + 2] for i in range(len(t) - 1)]

    def fingerprint(self, text: str) -> int:
        v = [0] * self.bits
        grams = self._grams(text)
        if not grams:
            return 0
        for g in grams:
            h = int(hashlib.sha1(g.encode("utf-8")).hexdigest()[:16], 16)
            for i in range(self.bits):
                v[i] += 1 if (h >> i) & 1 else -1
        fp = 0
        for i in range(self.bits):
            if v[i] > 0:
                fp |= 1 << i
        return fp

    @staticmethod
    def hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")


# =============================================================================
#  预处理器
# =============================================================================
@dataclass
class ProcessedDoc:
    """清洗后的文档"""

    doc_id: str
    title: str
    text: str
    sentences: List[str] = field(default_factory=list)
    source: str = "pubmed"
    pmid: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    language: str = "zh"
    term_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.doc_id, "title": self.title, "text": self.text,
            "sentences": self.sentences, "source": self.source,
            "pmid": self.pmid, "doi": self.doi, "journal": self.journal,
            "year": self.year, "url": self.url, "language": self.language,
            "term_hits": self.term_hits,
        }


class Preprocessor:
    """医学文本清洗与整合"""

    def __init__(self, simhash_bits: int = 64, dup_threshold: int = 3) -> None:
        self.simhash = SimHash(simhash_bits)
        self.dup_threshold = dup_threshold
        self._fingerprints: List[int] = []
        self._seen_hashes: Set[str] = set()
        self.stats: Dict[str, int] = {
            "documents_in": 0, "documents_out": 0, "duplicates": 0,
            "too_short": 0, "no_medical_term": 0, "ads": 0, "sentences": 0,
        }
        self._load_normalization_map()

    # ------------------------------------------------------------------
    def _load_normalization_map(self) -> None:
        """加载医学术语归一化词典（复用实体链接的口语别名映射）"""
        self.norm_map: Dict[str, str] = {}
        try:
            from app.llm_layer.entity_linker import (
                COLLOQUIAL_MAP, DEPARTMENT_ALIAS, SYMPTOM_ALIAS,
            )

            self.norm_map.update(COLLOQUIAL_MAP)
            self.norm_map.update(DEPARTMENT_ALIAS)
            self.norm_map.update(SYMPTOM_ALIAS)
            logger.info("术语归一化词典加载完成：%d 条", len(self.norm_map))
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载术语归一化词典失败（不影响清洗流程）：%s", exc)

    # ==================================================================
    #  1) 文本清洗
    # ==================================================================
    def clean_text(self, text: str) -> str:
        """去除 HTML/脚本/参考文献标记/广告，并做全角半角与空白归一化"""
        if not text:
            return ""
        s = str(text)
        s = _RE_HTML.sub(" ", s)
        s = _RE_SCRIPT.sub(" ", s)
        s = _RE_URL.sub(" ", s)
        s = _RE_EMAIL.sub(" ", s)
        s = _RE_COPYRIGHT.sub(" ", s)
        s = _RE_REF_MARK.sub(" ", s)
        s = _RE_SUP.sub("", s)
        s = _RE_CTRL.sub(" ", s)
        s = normalize(s)
        s = _RE_MULTI_SPACE.sub(" ", s)
        s = _RE_MULTI_NL.sub("\n\n", s)
        # 去掉重复标点
        s = re.sub(r"([。！？；，、])\1{1,}", r"\1", s)
        return s.strip()

    # ==================================================================
    #  2) 术语归一化
    # ==================================================================
    def normalize_terms(self, text: str) -> str:
        """
        把口语/别名替换为图谱标准名（长词优先，避免子串误替换）。
        例："血压高" → "原发性高血压"；"发烧" → "发热"；"心内科" → "心血管内科"
        """
        if not text or not self.norm_map:
            return text
        out = text
        for alias in sorted(self.norm_map, key=len, reverse=True):
            if alias in out:
                out = out.replace(alias, self.norm_map[alias])
        return out

    # ==================================================================
    #  3) 分句 + 去重
    # ==================================================================
    def split_and_dedup(self, text: str, min_len: int = 10, max_len: int = 300) -> List[str]:
        """分句、过滤过短/过长句、句内去重"""
        out: List[str] = []
        seen: Set[str] = set()
        for s in split_sentences(text):
            s = s.strip(" 　")
            if _RE_SECTION.match(s):
                continue
            if len(s) < min_len or len(s) > max_len:
                continue
            if not _RE_MED_TERM.search(s) and not re.search(r"\d+\s*(?:mg|mmHg|mmol)", s):
                continue
            key = hashlib.md5(s.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def is_duplicate(self, text: str) -> bool:
        """SimHash 近似去重 + MD5 精确去重"""
        if not text:
            return True
        md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        if md5 in self._seen_hashes:
            return True
        fp = self.simhash.fingerprint(text)
        for prev in self._fingerprints:
            if self.simhash.hamming(fp, prev) <= self.dup_threshold:
                return True
        self._seen_hashes.add(md5)
        self._fingerprints.append(fp)
        # 控制内存：保留最近 20 万条指纹
        if len(self._fingerprints) > 200000:
            self._fingerprints = self._fingerprints[-100000:]
        return False

    # ==================================================================
    #  4) 质量过滤
    # ==================================================================
    def quality_ok(self, text: str, min_len: int = 30) -> Tuple[bool, str]:
        """质量检查，返回 (是否通过, 拒绝原因)"""
        if not text or len(text) < min_len:
            return False, "too_short"
        if _RE_AD.search(text):
            return False, "ads"
        terms = _RE_MED_TERM.findall(text)
        if len(terms) < 2 and not re.search(r"\d+\s*(?:mg|mmHg|mmol)", text):
            return False, "no_medical_term"
        # 中文字符占比过低（可能是英文残留或乱码）
        cn = len(re.findall(r"[\u4e00-\u9fff]", text))
        if cn < len(text) * 0.2:
            return False, "low_chinese_ratio"
        return True, "ok"

    # ==================================================================
    #  5) 文档级处理
    # ==================================================================
    def process_document(self, raw: Dict[str, Any], normalize_terms: bool = True) -> Optional[ProcessedDoc]:
        """
        处理一篇原始文档（PubMed 记录 / 网页抓取结果），返回清洗后的文档或 None。
        """
        self.stats["documents_in"] += 1
        title = self.clean_text(raw.get("title") or "")
        body_parts = []
        for key in ("abstract", "text", "content", "body", "summary", "definition",
                    "cause", "diagnosis", "treatment", "prognosis"):
            v = raw.get(key)
            if v:
                body_parts.append(str(v))
        body = self.clean_text("\n".join(body_parts))
        full = f"{title}\n{body}".strip()

        if not full:
            self.stats["too_short"] += 1
            return None

        if normalize_terms:
            full = self.normalize_terms(full)
            title = self.normalize_terms(title)

        if self.is_duplicate(full):
            self.stats["duplicates"] += 1
            return None

        ok, reason = self.quality_ok(full)
        if not ok:
            self.stats[reason] = self.stats.get(reason, 0) + 1
            return None

        sentences = self.split_and_dedup(full)
        self.stats["sentences"] += len(sentences)

        doc_id = raw.get("id") or hashlib.md5(full.encode("utf-8")).hexdigest()[:16]
        doc = ProcessedDoc(
            doc_id=str(doc_id), title=title, text=full, sentences=sentences,
            source=raw.get("source", "pubmed"),
            pmid=str(raw["pmid"]) if raw.get("pmid") else None,
            doi=raw.get("doi"), journal=raw.get("journal"),
            year=int(raw["year"]) if str(raw.get("year") or "").isdigit() else None,
            url=raw.get("url"), language=raw.get("language", "zh"),
            term_hits=len(_RE_MED_TERM.findall(full)),
        )
        self.stats["documents_out"] += 1
        return doc

    # ------------------------------------------------------------------
    def process_batch(
        self, raws: Iterable[Dict[str, Any]], apply_privacy: bool = True
    ) -> List[ProcessedDoc]:
        """
        批量处理，并在最后统一执行 PII 脱敏（全链路脱敏要求）。
        """
        docs: List[ProcessedDoc] = []
        for raw in raws:
            try:
                doc = self.process_document(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("处理文档失败（跳过）：%s", exc)
                continue
            if doc is None:
                continue
            if apply_privacy:
                from app.data_layer.privacy import get_privacy_guard

                guard = get_privacy_guard()
                doc.title = guard.mask_text(doc.title)
                doc.text = guard.mask_text(doc.text)
                doc.sentences = [guard.mask_text(s) for s in doc.sentences]
            docs.append(doc)
        logger.info(
            "批量清洗完成：输入 %d，输出 %d，去重 %d，过短 %d，无医学术语 %d",
            self.stats["documents_in"], self.stats["documents_out"],
            self.stats["duplicates"], self.stats["too_short"],
            self.stats.get("no_medical_term", 0),
        )
        return docs

    # ------------------------------------------------------------------
    def save_jsonl(self, docs: Sequence[ProcessedDoc], path: Optional[Path] = None) -> Path:
        """保存为 JSONL（语料标准落盘格式）"""
        out = Path(path or (settings.corpus_dir / "corpus.jsonl"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
        logger.info("已保存 %d 条语料 → %s", len(docs), out)
        return out

    @staticmethod
    def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """读取 JSONL 语料"""
        rows: List[Dict[str, Any]] = []
        if not Path(path).exists():
            return rows
        with Path(path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def info(self) -> Dict[str, Any]:
        return {
            "simhash_bits": self.simhash.bits,
            "dup_threshold": self.dup_threshold,
            "normalization_terms": len(self.norm_map),
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
_pp: Optional[Preprocessor] = None


def get_preprocessor() -> Preprocessor:
    global _pp
    if _pp is None:
        _pp = Preprocessor()
    return _pp
