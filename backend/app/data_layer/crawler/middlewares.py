# -*- coding: utf-8 -*-
"""
Scrapy 下载中间件
=================
  * `UserAgentRotateMiddleware`   UA 轮换（可选 fake-useragent，缺失时用内置池）
  * `RateLimitGuardMiddleware`    429 响应自动降速（指数退避），保护目标站点
  * `RetryWithBackoffMiddleware`  5xx/超时自动重试 + 抖动退避
  * `StatsCollectorMiddleware`    抓取统计（成功率、平均耗时、状态码分布）

**合规声明**：本爬虫仅抓取公开的学术文献元数据与摘要，遵守 robots.txt，
严格限速，不抓取任何个人健康信息（PHI）。所有内容在入库前完成脱敏。
"""
from __future__ import annotations

import random
import time
from collections import Counter

from scrapy import signals
from scrapy.downloadermiddlewares.retry import get_retry_request
from scrapy.exceptions import IgnoreRequest

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: 内置 UA 池（无需第三方依赖）
_DEFAULT_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]


class UserAgentRotateMiddleware:
    """为每个请求随机分配 User-Agent"""

    def __init__(self) -> None:
        self.pool = list(_DEFAULT_UA_POOL)
        if settings.USER_AGENT_ROTATE:
            try:
                from fake_useragent import UserAgent  # type: ignore

                self.ua = UserAgent(browsers=["chrome", "edge", "firefox"], os=["windows", "macos"])
                logger.info("fake-useragent 已启用（实时 UA 池）")
            except Exception:  # noqa: BLE001
                self.ua = None
                logger.info("未安装 fake-useragent，使用内置 UA 池（%d 个）", len(self.pool))
        else:
            self.ua = None

    def process_request(self, request, spider):
        try:
            ua = self.ua.random if self.ua is not None else random.choice(self.pool)
        except Exception:  # noqa: BLE001
            ua = random.choice(self.pool)
        request.headers.setdefault(b"User-Agent", ua.encode())
        # 合规标识：告知对方这是学术研究爬虫
        request.headers.setdefault(
            b"From", (settings.PUBMED_EMAIL or "medical-kg@example.com").encode()
        )
        return None


class RateLimitGuardMiddleware:
    """
    429 / 403 自动降速
    ----------------
    医学文献站点对高频访问敏感，收到 429 时主动暂停并发请求，
    避免触发封禁（也符合《网络安全法》与站点服务条款的合规要求）。
    """

    def __init__(self) -> None:
        self.penalty_until = 0.0
        self.penalty_seconds = 5.0
        self.hits_429 = 0

    def process_request(self, request, spider):
        now = time.time()
        if now < self.penalty_until:
            time.sleep(min(self.penalty_until - now, 10.0))
        return None

    def process_response(self, request, response, spider):
        if response.status in (429, 403):
            self.hits_429 += 1
            self.penalty_seconds = min(self.penalty_seconds * 1.8, 120.0)
            self.penalty_until = time.time() + self.penalty_seconds
            logger.warning(
                "收到 %d，全局降速 %.1fs（累计 %d 次）: %s",
                response.status, self.penalty_seconds, self.hits_429, request.url,
            )
        elif response.status == 200:
            # 成功后逐步恢复
            self.penalty_seconds = max(5.0, self.penalty_seconds * 0.9)
        return response

    def close_spider(self, spider):
        if self.hits_429:
            logger.info("[RateLimit] 本轮共触发限速 %d 次", self.hits_429)


class RetryWithBackoffMiddleware:
    """5xx / 超时重试，带指数退避与抖动"""

    def __init__(self) -> None:
        self.retries = 0

    def process_response(self, request, response, spider):
        if response.status in (500, 502, 503, 504, 522, 524, 408):
            reason = f"HTTP {response.status}"
            self.retries += 1
            new_req = get_retry_request(
                request, spider=spider, reason=reason,
                max_retry_times=3, priority_adjust=-1,
            )
            if new_req is not None:
                delay = min(2 ** int(request.meta.get("retry_times", 0)) + random.random(), 30)
                new_req.meta["download_slot_delay"] = delay
                logger.info("重试 %s（%s），退避 %.1fs", request.url, reason, delay)
                return new_req
        return response

    def process_exception(self, request, exception, spider):
        self.retries += 1
        new_req = get_retry_request(
            request, spider=spider, reason=f"exception {exception}",
            max_retry_times=3, priority_adjust=-1,
        )
        if new_req is not None:
            logger.info("异常重试 %s：%s", request.url, exception)
            return new_req
        return None

    def close_spider(self, spider):
        if self.retries:
            logger.info("[Retry] 共重试 %d 次", self.retries)


class StatsCollectorMiddleware:
    """抓取统计：状态码分布、平均响应耗时、成功率"""

    def __init__(self) -> None:
        self.status_counter: Counter = Counter()
        self.total_ms = 0.0
        self.count = 0

    @classmethod
    def from_crawler(cls, crawler):
        obj = cls()
        crawler.signals.connect(obj.spider_closed, signal=signals.spider_closed)
        return obj

    def process_response(self, request, response, spider):
        self.status_counter[response.status] += 1
        self.count += 1
        # Scrapy 会在 response.flags / request.meta 中记录耗时，这里用下载时间近似
        self.total_ms += float(request.meta.get("download_latency", 0.0)) * 1000
        return response

    def spider_closed(self, spider, reason):
        ok = self.status_counter.get(200, 0)
        rate = ok / self.count * 100 if self.count else 0.0
        avg = self.total_ms / self.count if self.count else 0.0
        logger.info(
            "[Stats] 抓取 %d 条，成功 %d（%.1f%%），平均耗时 %.0fms，状态码分布 %s，原因 %s",
            self.count, ok, rate, avg, dict(self.status_counter), reason,
        )
