# -*- coding: utf-8 -*-
"""
权威医学站点爬虫（结构化疾病条目采集）
======================================
目标：从公开的权威医学知识页面抽取**结构化疾病条目**（定义/病因/症状/诊断/
治疗/预后），直接对应知识图谱的 `Disease` 节点属性。

设计要点
--------
1. **站点配置化**：`SITES` 字典声明式定义每个站点的列表页与详情页选择器，
   新增数据源只需增加配置，无需改动爬虫逻辑。
2. **合规**：遵守 robots.txt、限速 ≥1s、仅抓取公开科普/指南类页面，
   不抓取任何患者信息（PHI）。抓取内容入 Pipeline 后统一脱敏。
3. **容错**：任一选择器失效不影响整体运行（记为 warning，继续下一个字段）。

使用
----
    scrapy runspider app/data_layer/crawler/spiders/med_site_spider.py -a site=msd
    scrapy runspider app/data_layer/crawler/spiders/med_site_spider.py          # 全部站点
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import scrapy
from scrapy.http import Response

from app.data_layer.crawler.items import DiseaseKnowledgeItem
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MedSiteSpider(scrapy.Spider):
    """
    权威医学站点爬虫

    参数
    ----
    site   : 站点 key（见 `SITES`），留空则遍历全部
    limit  : 每个站点最多抓取的详情页数（默认 50，礼貌抓取）
    """

    name = "med_site"
    custom_settings = {
        "DOWNLOAD_DELAY": 1.5,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "ROBOTSTXT_OBEY": True,
    }

    # ==================================================================
    #  站点配置（声明式）
    # ==================================================================
    SITES: Dict[str, Dict[str, Any]] = {
        # 示例站点名与结构（真实部署时替换为目标权威站点的域名与选择器）
        "medical_wiki": {
            "name": "医学百科（示例）",
            "base": "https://www.example-medical-wiki.org",
            "list_urls": [
                "/disease/内科",
                "/disease/外科",
                "/disease/妇产科",
                "/disease/儿科",
                "/disease/传染科",
            ],
            "list_item": "div.disease-list a.disease-link::attr(href)",
            "detail": {
                "disease_name": "h1.disease-title::text",
                "category1": "span.category1::text",
                "category2": "span.category2::text",
                "definition": "div.section-definition::text",
                "cause": "div.section-cause::text",
                "symptoms": "div.section-symptoms li::text",
                "diagnosis": "div.section-diagnosis::text",
                "checks": "div.section-checks li::text",
                "treatments": "div.section-treatment li::text",
                "drugs": "div.section-drugs li::text",
                "department": "span.department::text",
                "prognosis": "div.section-prognosis::text",
                "population": "div.section-population::text",
                "complications": "div.section-complications li::text",
            },
            "authority": "A",
        },
        "guideline_center": {
            "name": "临床指南中心（示例）",
            "base": "https://www.example-guideline.org",
            "list_urls": ["/guidelines", "/guidelines/internal-medicine"],
            "list_item": "ul.guideline-list a::attr(href)",
            "detail": {
                "disease_name": "h1.guideline-title::text",
                "definition": "div.guideline-summary::text",
                "treatments": "div.recommendation li::text",
                "checks": "div.diagnosis-criteria li::text",
            },
            "authority": "A",
        },
        "nhc_official": {
            "name": "国家卫生健康委（示例）",
            "base": "https://www.example-nhc.gov.cn",
            "list_urls": ["/health-topics"],
            "list_item": "div.topic-list a::attr(href)",
            "detail": {
                "disease_name": "h2.topic-title::text",
                "definition": "div.topic-content p::text",
                "population": "div.target-population::text",
            },
            "authority": "A",
        },
    }

    def __init__(self, site: str = "", limit: int = 50, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.site_keys = [site] if site else list(self.SITES.keys())
        self.limit = int(limit)
        self.page_count = 0
        self.item_count = 0
        # 仅允许配置内的域名（防止跟到站外）
        self.allowed_domains = []
        for k in self.site_keys:
            cfg = self.SITES.get(k)
            if cfg:
                self.allowed_domains.append(
                    cfg["base"].replace("https://", "").replace("http://", "").strip("/")
                )

    # ==================================================================
    def start_requests(self):
        for key in self.site_keys:
            cfg = self.SITES.get(key)
            if not cfg:
                logger.warning("未知站点配置：%s（可选：%s）", key, list(self.SITES))
                continue
            logger.info("开始采集站点：%s（%s）", key, cfg["name"])
            for path in cfg["list_urls"]:
                url = urljoin(cfg["base"], path)
                yield scrapy.Request(
                    url, callback=self.parse_list, dont_filter=True,
                    meta={"site_key": key},
                    errback=self.on_error,
                )

    # ------------------------------------------------------------------
    def parse_list(self, response: Response):
        key = response.meta["site_key"]
        cfg = self.SITES[key]
        links = response.css(cfg["list_item"]).getall()
        if not links:
            logger.warning("列表页未匹配到条目（选择器可能需更新）：%s", response.url)
            return
        logger.info("列表页 %s 发现 %d 个条目链接", response.url, len(links))
        for href in links[: self.limit]:
            url = urljoin(cfg["base"], href)
            yield scrapy.Request(
                url, callback=self.parse_detail, dont_filter=True,
                meta={"site_key": key}, errback=self.on_error,
            )

    # ------------------------------------------------------------------
    def parse_detail(self, response: Response):
        key = response.meta["site_key"]
        cfg = self.SITES[key]
        sel = cfg["detail"]

        item = DiseaseKnowledgeItem()
        missing: List[str] = []

        for field, css in sel.items():
            try:
                if css.endswith("::text"):
                    values = response.css(css).getall()
                    value = self._join(values) if field in self.LIST_FIELDS else (
                        values[0].strip() if values else None)
                elif "::attr(" in css:
                    value = response.css(css).get()
                else:
                    values = [v.strip() for v in response.css(f"{css} ::text").getall()]
                    value = self._join(values)
                if value:
                    item[field] = self._clean(value)
                else:
                    missing.append(field)
            except Exception as exc:  # noqa: BLE001
                logger.debug("字段 %s 抽取失败：%s", field, exc)
                missing.append(field)

        if not item.get("disease_name"):
            logger.debug("详情页无疾病名，跳过：%s", response.url)
            return

        # 结构化后处理
        item["disease_name"] = self._clean_name(item.get("disease_name", ""))
        item["is_infectious"] = self._detect_infectious(response.text, item)
        item["source"] = "website"
        item["url"] = response.url
        item["authority"] = cfg["authority"]
        item["retrieved_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

        self.item_count += 1
        if missing:
            logger.debug("%s 缺失字段：%s", item.get("disease_name"), missing)
        yield item

    #: 需要合并为列表的字段
    LIST_FIELDS = {"symptoms", "checks", "treatments", "drugs", "complications", "alias"}

    #: 传染病提示词
    INFECTIOUS_HINTS = ("传染病", "具有传染性", "可通过", "传播途径", "需隔离",
                        "法定传染病", "呼吸道传播", "消化道传播", "血液传播")

    def _detect_infectious(self, html: str, item: Any) -> bool:
        text = html or ""
        hits = sum(1 for h in self.INFECTIOUS_HINTS if h in text)
        # 明确标注"非传染"时优先
        if re.search(r"(?:不传染|无传染性|非传染性|不具有传染性)", text):
            return False
        return hits >= 2

    # ------------------------------------------------------------------
    @staticmethod
    def _join(values: List[str]) -> str:
        parts = [v.strip() for v in values if v and v.strip()]
        # 去重保序
        seen, out = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return "、".join(out)

    @staticmethod
    def _clean(text: str) -> str:
        s = re.sub(r"<[^>]+>", " ", str(text or ""))
        s = re.sub(r"[\s\u3000]+", " ", s)
        s = re.sub(r"\[\d+\]", "", s)
        return s.strip()

    @staticmethod
    def _clean_name(name: str) -> str:
        s = re.sub(r"[（(].*?[)）]", "", str(name or ""))
        s = re.sub(r"(?:的症状|是什么|简介|概述|-医学百科| - 医学百科)$", "", s)
        return s.strip()[:40]

    # ------------------------------------------------------------------
    def on_error(self, failure):
        logger.warning("请求失败：%s → %s", failure.request.url, failure.value)

    def closed(self, reason: str):
        logger.info("医学站点采集结束：共产出 %d 条结构化疾病条目（原因：%s）",
                    self.item_count, reason)
