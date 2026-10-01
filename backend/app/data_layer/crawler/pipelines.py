# -*- coding: utf-8 -*-
"""
Scrapy 管道（Pipelines）
========================
数据流水线（顺序与 `settings.py` 中的 ITEM_PIPELINES 权重一致）：

    100 CleanTextPipeline          去 HTML/噪声、全角半角统一
    200 MedicalTermNormalizePipeline 口语别名 → 图谱标准名
    300 QualityFilterPipeline      质量过滤（过短/广告/无医学术语）
    400 DeduplicatePipeline        SimHash 近似去重 + MD5 精确去重
    500 PrivacyMaskPipeline        ★ PII 脱敏（全链路脱敏要求）
    600 FieldEncryptPipeline       ★ 敏感字段 AES-256-GCM 加密
    700 SourceTracePipeline        溯源信息补全（权威等级、抓取时间）
    900 JsonlWriterPipeline        标准化 JSONL 落盘
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

from app.config import settings
from app.data_layer.preprocess import Preprocessor, SimHash
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# =============================================================================
#  100 文本清洗
# =============================================================================
class CleanTextPipeline:
    """清洗 HTML、参考文献标记、控制字符，统一空白与全半角"""

    def __init__(self) -> None:
        self.pp = Preprocessor()

    def process_item(self, item, spider):
        a = ItemAdapter(item)
        for field in ("title", "abstract", "text", "keywords"):
            v = a.get(field)
            if not v:
                continue
            if isinstance(v, (list, tuple)):
                a[field] = [self.pp.clean_text(str(x)) for x in v if str(x).strip()]
            else:
                a[field] = self.pp.clean_text(str(v))
        return item


# =============================================================================
#  200 术语归一化
# =============================================================================
class MedicalTermNormalizePipeline:
    """把口语/别名统一为知识图谱标准术语，保证后续实体链接可用"""

    def __init__(self) -> None:
        self.pp = Preprocessor()

    def process_item(self, item, spider):
        a = ItemAdapter(item)
        for field in ("title", "abstract", "text"):
            v = a.get(field)
            if isinstance(v, str) and v:
                a[field] = self.pp.normalize_terms(v)
        return item


# =============================================================================
#  300 质量过滤
# =============================================================================
class QualityFilterPipeline:
    """丢弃低质量样本：过短、纯广告、无医学实体、中文占比过低"""

    def __init__(self) -> None:
        self.pp = Preprocessor()
        self.dropped = 0

    def process_item(self, item, spider):
        a = ItemAdapter(item)
        text = " ".join(str(a.get(f) or "") for f in ("title", "abstract", "text"))
        ok, reason = self.pp.quality_ok(text)
        if not ok:
            self.dropped += 1
            raise DropItem(f"[QualityFilter] {reason}: {text[:60]}")
        a["quality_score"] = round(min(1.0, len(text) / 1500.0), 3)
        return item

    def close_spider(self, spider):
        logger.info("[QualityFilter] 共丢弃 %d 条低质量样本", self.dropped)


# =============================================================================
#  400 去重
# =============================================================================
class DeduplicatePipeline:
    """MD5 精确去重 + SimHash 近似去重（医学文献转载严重，近似去重很有必要）"""

    def __init__(self) -> None:
        self.simhash = SimHash(64)
        self.hashes: set = set()
        self.fps: list = []
        self.dups = 0

    def process_item(self, item, spider):
        a = ItemAdapter(item)
        text = " ".join(str(a.get(f) or "") for f in ("title", "abstract", "text"))
        md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        if md5 in self.hashes:
            self.dups += 1
            raise DropItem(f"[Dedup] MD5 重复: {text[:50]}")
        fp = self.simhash.fingerprint(text)
        for prev in self.fps:
            if self.simhash.hamming(fp, prev) <= 3:
                self.dups += 1
                raise DropItem(f"[Dedup] SimHash 近似重复: {text[:50]}")
        self.hashes.add(md5)
        self.fps.append(fp)
        if len(self.fps) > 200000:
            self.fps = self.fps[-100000:]
        return item

    def close_spider(self, spider):
        logger.info("[Dedup] 共去重 %d 条", self.dups)


# =============================================================================
#  500 隐私脱敏
# =============================================================================
class PrivacyMaskPipeline:
    """
    ★ 全链路脱敏：PII 在入库前完成掩码，原始 PII 不落盘、不进日志。
    对 title / abstract / text 全部执行，并记录脱敏审计（仅统计，不记录原文）。
    """

    def __init__(self) -> None:
        from app.data_layer.privacy import get_privacy_guard

        self.guard = get_privacy_guard()
        self.total_hits = 0

    def process_item(self, item, spider):
        a = ItemAdapter(item)
        hits = 0
        for field in ("title", "abstract", "text"):
            v = a.get(field)
            if isinstance(v, str) and v:
                res = self.guard.mask(v)
                a[field] = res.text
                hits += res.hit_count
        a["pii_masked"] = True
        if hits:
            self.total_hits += hits
            logger.debug("[Privacy] 脱敏 %d 处（%s）", hits, a.get("url"))
        return item

    def close_spider(self, spider):
        logger.info("[Privacy] 本轮共脱敏 PII %d 处", self.total_hits)


# =============================================================================
#  600 字段加密
# =============================================================================
class FieldEncryptPipeline:
    """
    ★ 敏感字段 AES-256-GCM 加密。

    加密对象：一切可能含个体信息的字段（作者姓名、机构、原始 URL 中的 token 等）。
    知识性字段（title/abstract/text）不加密，因为它们是脱敏后的公共医学知识，
    需要参与全文检索（加密会导致无法检索）。

    说明：本管道对 `authors` 与 `institution` 执行加密，密文以 `enc:` 前缀标记，
    读取时按需解密（见 `privacy.decrypt_field`）。
    """

    SENSITIVE_FIELDS = ("authors", "institution", "corresponding_author")

    def __init__(self) -> None:
        from app.data_layer.privacy import get_privacy_guard

        self.guard = get_privacy_guard()
        self.encrypted = 0

    def process_item(self, item, spider):
        a = ItemAdapter(item)
        done = []
        for field in self.SENSITIVE_FIELDS:
            v = a.get(field)
            if isinstance(v, str) and v and not v.startswith("enc:"):
                a[field] = "enc:" + self.guard.encrypt_field(v)
                done.append(field)
        # 作者列表：逐个加密后以分号连接，保持可解密
        authors = a.get("authors")
        if isinstance(authors, (list, tuple)) and authors:
            a["authors"] = "enc:" + ";".join(
                self.guard.encrypt_field(str(x)) for x in authors
            )
            done.append("authors")
        if done:
            self.encrypted += len(done)
        a["encrypted_fields"] = done
        return item

    def close_spider(self, spider):
        logger.info("[Encrypt] 共加密敏感字段 %d 个", self.encrypted)


# =============================================================================
#  700 溯源补全
# =============================================================================
class SourceTracePipeline:
    """
    补全溯源信息（PPT 指标：知识来源 100% 可追溯）。

    权威等级判定规则：
        A —— 临床指南、行业标准、规划教材、国家卫健委文件
        B —— SCI/核心期刊论文（有 PMID / DOI）
        C —— 其他网络来源
    """

    GUIDELINE_HINTS = ("指南", "共识", "规范", "标准", "纲要", "专家建议", "诊疗方案")
    TEXTBOOK_HINTS = ("教材", "教科书", "医学百科", "词典")

    def process_item(self, item, spider):
        a = ItemAdapter(item)
        source_type = (a.get("source") or "website").lower()
        title = str(a.get("title") or "")
        journal = str(a.get("journal") or "")

        if a.get("pmid") or a.get("doi"):
            authority = "B"
            source_type = "pubmed"
        elif any(h in title for h in self.GUIDELINE_HINTS) or any(h in journal for h in self.GUIDELINE_HINTS):
            authority = "A"
            source_type = "guideline"
        elif any(h in title for h in self.TEXTBOOK_HINTS):
            authority = "A"
            source_type = "textbook"
        else:
            authority = "C"

        a["authority"] = a.get("authority") or authority
        a["source"] = source_type
        a["retrieved_at"] = a.get("retrieved_at") or _now_iso()
        a["crawl_batch"] = a.get("crawl_batch") or datetime.now().strftime("%Y%m%d")
        if not a.get("id"):
            a["id"] = str(a.get("pmid") or hashlib.md5(
                (title + str(a.get("url") or "")).encode("utf-8")).hexdigest()[:16])
        return item


# =============================================================================
#  900 JSONL 落盘
# =============================================================================
class JsonlWriterPipeline:
    """标准化 JSONL 输出（每行一条记录，UTF-8，ensure_ascii=False）"""

    def __init__(self, out_dir: str = "") -> None:
        self.out_dir = out_dir
        self.fh = None
        self.count = 0

    @classmethod
    def from_crawler(cls, crawler):
        return cls(out_dir=str(settings.corpus_dir))

    def open_spider(self, spider):
        from pathlib import Path

        d = Path(self.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{spider.name}.jsonl"
        self.fh = path.open("a", encoding="utf-8")
        logger.info("[Writer] 输出文件：%s", path)

    def process_item(self, item, spider):
        if self.fh is None:
            raise DropItem("[Writer] 文件句柄未打开")
        row = {k: v for k, v in ItemAdapter(item).asdict().items() if v not in (None, "", [])}
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.count += 1
        return item

    def close_spider(self, spider):
        if self.fh:
            self.fh.close()
            self.fh = None
        logger.info("[Writer] 共写入 %d 条记录", self.count)
