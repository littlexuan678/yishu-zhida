# -*- coding: utf-8 -*-
"""
Scrapy 全局配置
===============
运行方式（在 backend/ 目录下）：
    scrapy runspider app/data_layer/crawler/spiders/pubmed_spider.py -O data/processed/pubmed.jsonl
或通过 scrapy.cfg 定义的项目：
    scrapy crawl pubmed -O data/processed/pubmed.jsonl

合规要求：
  * 遵守 robots.txt（ROBOTSTXT_OBEY=True）
  * 单站点并发不超过 8，下载延迟 ≥1s（AUTOTHROTTLE 自适应）
  * PubMed 官方要求提供 tool 与 email 标识（见 pubmed_spider）
  * 所有抓取内容在 Pipeline 中完成脱敏与加密后才落盘
"""
from __future__ import annotations

import os

from app.config import settings

BOT_NAME = "zhiyu_medical_codex"

SPIDER_MODULES = ["app.data_layer.crawler.spiders"]
NEWSPIDER_MODULE = "app.data_layer.crawler.spiders"

# ------------------------- 礼貌抓取 -------------------------
ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = int(os.getenv("CRAWLER_CONCURRENT", settings.CRAWLER_CONCURRENT))
CONCURRENT_REQUESTS_PER_DOMAIN = 4
DOWNLOAD_DELAY = float(os.getenv("CRAWLER_DELAY", settings.CRAWLER_DELAY))
RANDOMIZE_DOWNLOAD_DELAY = True
DOWNLOAD_TIMEOUT = 30
RETRY_TIMES = 3
RETRY_HTTP_CODES = [429, 500, 502, 503, 504, 522, 524, 408]

# ------------------------- 自适应限速 -------------------------
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 15.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 4.0
AUTOTHROTTLE_DEBUG = False

# ------------------------- 请求头 -------------------------
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}

# ------------------------- 中间件 -------------------------
DOWNLOADER_MIDDLEWARES = {
    "app.data_layer.crawler.middlewares.UserAgentRotateMiddleware": 400,
    "app.data_layer.crawler.middlewares.RateLimitGuardMiddleware": 410,
    "app.data_layer.crawler.middlewares.RetryWithBackoffMiddleware": 550,
    "app.data_layer.crawler.middlewares.StatsCollectorMiddleware": 900,
}

# ------------------------- 管道（核心：清洗 → 脱敏 → 加密 → 落盘）-------------------------
ITEM_PIPELINES = {
    "app.data_layer.crawler.pipelines.CleanTextPipeline": 100,
    "app.data_layer.crawler.pipelines.MedicalTermNormalizePipeline": 200,
    "app.data_layer.crawler.pipelines.QualityFilterPipeline": 300,
    "app.data_layer.crawler.pipelines.DeduplicatePipeline": 400,
    "app.data_layer.crawler.pipelines.PrivacyMaskPipeline": 500,
    "app.data_layer.crawler.pipelines.FieldEncryptPipeline": 600,
    "app.data_layer.crawler.pipelines.SourceTracePipeline": 700,
    "app.data_layer.crawler.pipelines.JsonlWriterPipeline": 900,
}

# ------------------------- 编码与缓存 -------------------------
FEED_EXPORT_ENCODING = "utf-8"
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400          # 24 小时内复用缓存，避免重复请求
HTTPCACHE_DIR = "data/httpcache"
HTTPCACHE_IGNORE_HTTP_CODES = [429, 500, 502, 503, 504]
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# ------------------------- 日志 -------------------------
LOG_LEVEL = settings.LOG_LEVEL
LOG_FILE = str(settings.log_dir / "crawler.log") if os.getenv("CRAWLER_LOG_FILE") else None

# ------------------------- 其他 -------------------------
TELNETCONSOLE_ENABLED = False
COOKIES_ENABLED = False
AJAXCRAWL_ENABLED = False
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
