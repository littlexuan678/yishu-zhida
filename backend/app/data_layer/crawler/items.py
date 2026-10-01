# -*- coding: utf-8 -*-
"""
Scrapy Item 定义
================
统一多源医学数据的字段结构，屏蔽 PubMed / 网页 / 指南 PDF 的差异。
"""
from __future__ import annotations

import scrapy


class MedicalDocumentItem(scrapy.Item):
    """一篇医学文献 / 网页文档"""

    # ---- 标识 ----
    id = scrapy.Field()               # 内部唯一 ID（md5 或 pmid）
    pmid = scrapy.Field()             # PubMed ID
    doi = scrapy.Field()
    pmcid = scrapy.Field()

    # ---- 内容 ----
    title = scrapy.Field()
    abstract = scrapy.Field()
    text = scrapy.Field()
    keywords = scrapy.Field()
    mesh_terms = scrapy.Field()       # MeSH 主题词（医学受控词表）
    section = scrapy.Field()

    # ---- 出版信息 ----
    journal = scrapy.Field()
    year = scrapy.Field()
    authors = scrapy.Field()
    volume = scrapy.Field()
    issue = scrapy.Field()
    pages = scrapy.Field()
    language = scrapy.Field()

    # ---- 溯源 ----
    source = scrapy.Field()           # pubmed | guideline | website | textbook
    source_name = scrapy.Field()      # 站点/数据库名
    url = scrapy.Field()
    retrieved_at = scrapy.Field()
    authority = scrapy.Field()        # A(指南/教材) | B(期刊) | C(其他)

    # ---- 质量与合规 ----
    quality_score = scrapy.Field()
    pii_masked = scrapy.Field()       # 是否已脱敏
    encrypted_fields = scrapy.Field() # 已加密字段列表
    crawl_batch = scrapy.Field()


class DiseaseKnowledgeItem(scrapy.Item):
    """从权威医学百科页面抽取的结构化疾病条目"""

    disease_name = scrapy.Field()
    alias = scrapy.Field()
    category1 = scrapy.Field()
    category2 = scrapy.Field()
    definition = scrapy.Field()
    cause = scrapy.Field()
    symptoms = scrapy.Field()
    diagnosis = scrapy.Field()
    checks = scrapy.Field()
    treatments = scrapy.Field()
    drugs = scrapy.Field()
    department = scrapy.Field()
    prognosis = scrapy.Field()
    population = scrapy.Field()
    complications = scrapy.Field()
    differential = scrapy.Field()
    is_infectious = scrapy.Field()

    source = scrapy.Field()
    url = scrapy.Field()
    retrieved_at = scrapy.Field()
    authority = scrapy.Field()
