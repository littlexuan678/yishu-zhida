# -*- coding: utf-8 -*-
"""
PubMed 文献采集爬虫（PubMed E-utilities API）
=============================================
PubMed 官方提供 E-utilities REST API，**推荐用它而非直接爬 HTML 页面**：
  * 稳定：接口有明确契约，不会因页面改版而失效
  * 高效：一次 `efetch` 可批量取回 200 条完整记录
  * 合规：官方许可，只需提供 tool 与 email 标识并控制速率（≤3 req/s 无 key）

采集流程（两阶段）
------------------
  ① **esearch**  按检索式（疾病名 + 医学主题词）取回 PMID 列表
       GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
           ?db=pubmed&term=<query>&retmax=200&retmode=json&sort=date
  ② **efetch**   按 PMID 批量取回完整记录（XML），解析标题/摘要/MeSH/作者/期刊
       GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
           ?db=pubmed&id=<pmid1,pmid2,...>&retmode=xml

使用
----
    # 采集高血压相关近 1 年文献
    scrapy runspider app/data_layer/crawler/spiders/pubmed_spider.py -a query="hypertension" -a retmax=200

    # 批量采集种子疾病库
    python scripts/crawler_run.py --since 1d
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import scrapy
from scrapy.http import Response

from app.config import settings
from app.data_layer.crawler.items import MedicalDocumentItem
from app.utils.logger import get_logger

logger = get_logger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubmedSpider(scrapy.Spider):
    """
    PubMed 文献采集爬虫

    参数
    ----
    query   : 检索式（支持 PubMed 语法，如 "hypertension AND 2024[dp]"）
    retmax  : 单次最多取回条数（默认 200，PubMed 上限 10000）
    days    : 仅取近 N 天入库文献（配合增量更新，实现 ≤24h 更新周期）
    batch   : efetch 单批 PMID 数（默认 100，官方建议 ≤200）
    """

    name = "pubmed"
    allowed_domains = ["ncbi.nlm.nih.gov", "eutils.ncbi.nlm.nih.gov"]
    custom_settings = {
        "DOWNLOAD_DELAY": 0.4,          # 无 API key 限 3 req/s，留出余量
        "CONCURRENT_REQUESTS": 2,
        "ROBOTSTXT_OBEY": True,
    }

    #: 医学主题词检索式模板（用 MeSH 提升召回质量）
    MESH_QUERIES: Dict[str, str] = {
        "高血压": '(hypertension[MeSH Terms])',
        "糖尿病": '(diabetes mellitus, type 2[MeSH Terms])',
        "冠心病": '(coronary disease[MeSH Terms])',
        "脑卒中": '(stroke[MeSH Terms])',
        "肺炎": '(pneumonia[MeSH Terms])',
        "慢性阻塞性肺疾病": '(pulmonary disease, chronic obstructive[MeSH Terms])',
        "肺栓塞": '(pulmonary embolism[MeSH Terms])',
        "哮喘": '(asthma[MeSH Terms])',
        "胃炎": '(gastritis[MeSH Terms])',
        "消化性溃疡": '(peptic ulcer[MeSH Terms])',
        "慢性肾脏病": '(renal insufficiency, chronic[MeSH Terms])',
        "甲状腺功能亢进症": '(hyperthyroidism[MeSH Terms])',
        "抑郁症": '(depressive disorder[MeSH Terms])',
        "阿尔茨海默病": '(alzheimer disease[MeSH Terms])',
        "类风湿关节炎": '(arthritis, rheumatoid[MeSH Terms])',
        "肺结核": '(tuberculosis[MeSH Terms])',
    }

    def __init__(
        self,
        query: str = "",
        retmax: int = 200,
        days: int = 0,
        batch: int = 100,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.user_query = query
        self.retmax = int(retmax)
        self.days = int(days)
        self.batch_size = min(int(batch), 200)
        self.pmid_buffer: List[str] = []
        self.fetched = 0
        self.term: Optional[str] = None

    # ==================================================================
    #  入口
    # ==================================================================
    def start_requests(self):
        """
        第一个请求：esearch 取回 PMID 列表。
        若未指定 query，则遍历 `MESH_QUERIES` 逐个采集。
        """
        if self.user_query:
            self.term = self._build_term(self.user_query)
            yield self._esearch_request(self.term)
        else:
            # 默认采集种子疾病主题词（每类取 retmax/len 条）
            per = max(20, self.retmax // max(len(self.MESH_QUERIES), 1))
            for disease, mesh in self.MESH_QUERIES.items():
                term = self._build_term(mesh)
                logger.info("采集主题：%s → %s", disease, term)
                yield self._esearch_request(term, retmax=per)

    # ------------------------------------------------------------------
    def _build_term(self, query: str) -> str:
        """拼接检索式（含日期范围过滤，支持增量采集）"""
        term = query
        if self.days > 0:
            since = (datetime.now(timezone.utc) - timedelta(days=self.days)).strftime("%Y/%m/%d")
            term = f"({term}) AND (\"{since}\"[Date - Entry])"
        return term

    def _esearch_request(self, term: str, retmax: Optional[int] = None) -> scrapy.Request:
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": retmax or self.retmax,
            "retmode": "json",
            "sort": "date",
            "tool": settings.PUBMED_TOOL,
            "email": settings.PUBMED_EMAIL,
        }
        if settings.PUBMED_API_KEY:
            params["api_key"] = settings.PUBMED_API_KEY
        url = f"{EUTILS}/esearch.fcgi?" + "&".join(f"{k}={v}" for k, v in params.items())
        return scrapy.Request(url, callback=self.parse_esearch, meta={"term": term}, dont_filter=True)

    # ==================================================================
    #  解析 esearch → 触发 efetch
    # ==================================================================
    def parse_esearch(self, response: Response):
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            logger.error("esearch 响应不是合法 JSON：%s", response.text[:200])
            return
        result = data.get("esearchresult") or {}
        pmids = result.get("idlist") or []
        logger.info(
            "esearch 完成：term=%s，命中总数 %s，本次取回 %d 条",
            response.meta.get("term", ""), result.get("count", "?"), len(pmids),
        )
        if not pmids:
            return
        # 分批 efetch（官方建议每次 ≤200 个 ID）
        for i in range(0, len(pmids), self.batch_size):
            chunk = pmids[i : i + self.batch_size]
            yield self._efetch_request(chunk)

    def _efetch_request(self, pmids: List[str]) -> scrapy.Request:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": settings.PUBMED_TOOL,
            "email": settings.PUBMED_EMAIL,
        }
        if settings.PUBMED_API_KEY:
            params["api_key"] = settings.PUBMED_API_KEY
        url = f"{EUTILS}/efetch.fcgi?" + "&".join(f"{k}={v}" for k, v in params.items())
        return scrapy.Request(url, callback=self.parse_efetch, dont_filter=True)

    # ==================================================================
    #  解析 efetch XML
    # ==================================================================
    def parse_efetch(self, response: Response):
        try:
            root = ET.fromstring(response.body)
        except ET.ParseError as exc:
            logger.error("efetch XML 解析失败：%s", exc)
            return

        for article in root.findall(".//PubmedArticle"):
            item = self._parse_article(article)
            if item is not None:
                self.fetched += 1
                yield item

    def _parse_article(self, article: ET.Element) -> Optional[MedicalDocumentItem]:
        """把一个 PubmedArticle XML 节点转为 Item"""
        try:
            medline = article.find("MedlineCitation")
            if medline is None:
                return None
            pmid_el = medline.find("PMID")
            pmid = pmid_el.text if pmid_el is not None else None

            art = medline.find("Article")
            if art is None:
                return None

            item = MedicalDocumentItem()
            item["pmid"] = pmid
            item["id"] = f"PMID{pmid}" if pmid else None

            # ---- 标题 ----
            title_el = art.find("ArticleTitle")
            item["title"] = self._text(title_el)

            # ---- 摘要（可能分多个 AbstractText 段，带 Label）----
            abstract_parts: List[str] = []
            for ab in art.findall(".//Abstract/AbstractText"):
                label = ab.get("Label")
                txt = self._text(ab)
                if txt:
                    abstract_parts.append(f"{label}：{txt}" if label else txt)
            item["abstract"] = "\n".join(abstract_parts)

            # ---- 关键词 ----
            keywords = [self._text(k) for k in art.findall(".//KeywordList/Keyword")]
            item["keywords"] = [k for k in keywords if k]

            # ---- MeSH 主题词 ----
            mesh = []
            for mh in medline.findall(".//MeshHeading/DescriptorName"):
                t = self._text(mh)
                if t:
                    mesh.append(t)
            item["mesh_terms"] = mesh

            # ---- 期刊 / 年份 ----
            journal_el = art.find(".//Journal/Title")
            item["journal"] = self._text(journal_el)
            year = (
                self._text(art.find(".//JournalIssue/PubDate/Year"))
                or self._text(art.find(".//JournalIssue/PubDate/MedlineDate"))
                or self._text(art.find(".//ArticleDate/Year"))
            )
            item["year"] = year[:4] if year else None

            item["volume"] = self._text(art.find(".//JournalIssue/Volume"))
            item["issue"] = self._text(art.find(".//JournalIssue/Issue"))
            item["pages"] = self._text(art.find(".//Pagination/MedlinePgn"))

            # ---- 作者（Pipeline 中会加密）----
            authors = []
            for a in art.findall(".//AuthorList/Author"):
                last = self._text(a.find("LastName")) or ""
                fore = self._text(a.find("ForeName")) or ""
                name = f"{last} {fore}".strip()
                if name:
                    authors.append(name)
            item["authors"] = authors

            # ---- DOI / PMCID ----
            for aid in article.findall(".//ArticleIdList/ArticleId"):
                id_type = (aid.get("IdType") or "").lower()
                if id_type == "doi":
                    item["doi"] = self._text(aid)
                elif id_type == "pmc":
                    item["pmcid"] = self._text(aid)

            # ---- 语言 ----
            item["language"] = self._text(art.find(".//Language")) or "eng"

            # ---- 溯源 ----
            item["source"] = "pubmed"
            item["source_name"] = "PubMed"
            item["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
            item["retrieved_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            item["authority"] = "B"
            item["crawl_batch"] = datetime.now().strftime("%Y%m%d")
            return item
        except Exception as exc:  # noqa: BLE001
            logger.warning("解析 PubmedArticle 失败：%s", exc)
            return None

    @staticmethod
    def _text(el: Optional[ET.Element]) -> Optional[str]:
        """提取 XML 元素全部文本（含子节点，如 <i>、<sup>）"""
        if el is None:
            return None
        return "".join(el.itertext()).strip() or None

    # ==================================================================
    def closed(self, reason: str):
        logger.info("PubMed 采集结束：共产出 %d 条文献（原因：%s）", self.fetched, reason)
