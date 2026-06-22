from __future__ import annotations

import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Iterable

from news_crawler.naver import NaverNewsClient


COLUMNS = ["date", "company", "ticker", "keyword", "title", "press", "source"]
DEFAULT_MARKET_KEYWORDS = [
    "코스피",
    "국내 증시",
    "외국인 순매수",
    "기관 순매수",
    "금리",
    "환율",
    "유가",
    "물가",
    "반도체",
    "미국 증시",
    "FOMC",
    "중국 경기",
]


class _RequestThrottle:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.0, interval_seconds)
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            request_at = max(now, self._next_request_at)
            self._next_request_at = request_at + self.interval_seconds
        delay = request_at - now
        if delay > 0:
            time.sleep(delay)


def _write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _deduplicate(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for record in records:
        key = (
            record.get("date", ""),
            record.get("ticker", ""),
            " ".join(record.get("title", "").casefold().split()),
        )
        if key[2]:
            unique.setdefault(key, record)
    return sorted(
        unique.values(),
        key=lambda item: (item["date"], item["company"], item["title"]),
    )


def _ensure_nonempty(
    market_records: list[dict[str, str]],
    stock_records: list[dict[str, str]],
) -> None:
    if not market_records and not stock_records:
        raise RuntimeError(
            "Naver returned no news for the requested date; "
            "check the date and search-page selectors before retrying."
        )


def crawl_latest_news(
    *,
    target_date: date,
    tickers: list[tuple[str, str]],
    cache_dir: Path,
    market_keywords: list[str] | None = None,
    max_market_per_keyword: int = 2,
    max_stock_per_ticker: int = 1,
    workers: int = 4,
    request_interval_seconds: float = 0.8,
    force: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    market_path = cache_dir / "market_news.csv"
    stock_path = cache_dir / "stock_news.csv"
    if not force and market_path.exists() and stock_path.exists():
        market_records = _read_csv(market_path)
        stock_records = _read_csv(stock_path)
        _ensure_nonempty(market_records, stock_records)
        return market_records, stock_records

    keywords = market_keywords or DEFAULT_MARKET_KEYWORDS
    thread_local = threading.local()
    throttle = _RequestThrottle(request_interval_seconds)

    def client() -> NaverNewsClient:
        if not hasattr(thread_local, "client"):
            thread_local.client = NaverNewsClient()
        return thread_local.client

    def crawl_market(keyword: str) -> list[dict[str, str]]:
        throttle.wait()
        results = client().search(
            keyword,
            target_date=target_date,
            max_results=max_market_per_keyword,
        )
        return [
            {
                **result,
                "company": "시장전체",
                "ticker": "",
                "keyword": keyword,
            }
            for result in results
        ]

    def crawl_stock(ticker: str, company: str) -> list[dict[str, str]]:
        throttle.wait()
        results = client().search(
            company,
            target_date=target_date,
            max_results=max_stock_per_ticker,
        )
        return [
            {
                **result,
                "company": company,
                "ticker": ticker.removesuffix(".KS"),
                "keyword": "",
            }
            for result in results
        ]

    market_records: list[dict[str, str]] = []
    stock_records: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        market_futures = [executor.submit(crawl_market, keyword) for keyword in keywords]
        stock_futures = [
            executor.submit(crawl_stock, ticker, company) for ticker, company in tickers
        ]
        for future in as_completed(market_futures):
            market_records.extend(future.result())
        for future in as_completed(stock_futures):
            stock_records.extend(future.result())

    market_records = _deduplicate(market_records)
    stock_records = _deduplicate(stock_records)
    _ensure_nonempty(market_records, stock_records)
    _write_csv(market_path, market_records)
    _write_csv(stock_path, stock_records)
    return market_records, stock_records
