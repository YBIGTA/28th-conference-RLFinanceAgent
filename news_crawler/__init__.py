"""Incremental Naver news crawler for the demo's fixed KOSPI universe."""

from news_crawler.service import COLUMNS, DEFAULT_MARKET_KEYWORDS, crawl_latest_news

__all__ = ["COLUMNS", "DEFAULT_MARKET_KEYWORDS", "crawl_latest_news"]
