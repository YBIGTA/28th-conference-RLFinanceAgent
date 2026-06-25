from __future__ import annotations

import csv
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Iterable

from news_crawler.naver import NaverNewsClient


COLUMNS = ["date", "company", "ticker", "keyword", "title", "link", "press", "source"]
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

MARKET_SEARCH_QUERIES = {
    "코스피": "코스피 증시",
    "국내 증시": "국내 증시",
    "외국인 순매수": "외국인 순매수 코스피",
    "기관 순매수": "기관 순매수 코스피",
    "금리": "금리 증시",
    "환율": "환율 증시",
    "유가": "유가 증시",
    "물가": "물가 증시",
    "반도체": "반도체 증시",
    "미국 증시": "미국 증시",
    "FOMC": "FOMC 증시",
    "중국 경기": "중국 경기 증시",
}

MARKET_RELEVANCE_TERMS = {
    "코스피": ("코스피", "증시", "주가", "시총"),
    "국내 증시": ("코스피", "코스닥", "국내증시", "증시"),
    "외국인 순매수": ("외국인", "순매수", "수급"),
    "기관 순매수": ("기관", "순매수", "수급"),
    "금리": ("금리", "채권", "국채"),
    "환율": ("환율", "원달러", "달러", "외환"),
    "유가": ("유가", "원유", "석유", "두바이유", "WTI", "브렌트"),
    "물가": ("물가", "인플레이션", "소비자물가", "생산자물가"),
    "반도체": ("반도체", "삼성전자", "SK하이닉스", "마이크론"),
    "미국 증시": ("미국증시", "뉴욕증시", "나스닥", "다우", "S&P"),
    "FOMC": ("FOMC", "연준", "파월", "기준금리"),
    "중국 경기": ("중국", "중화", "경기", "위안", "中"),
}

COMPANY_ALIASES = {
    "HD건설기계": ("HD현대건설기계",),
    "HD한국조선해양": ("한국조선해양",),
    "HD현대중공업": ("HD현대중공업", "HD현대重"),
    "LIG디펜스앤에어로스페이스": ("LIG D&A", "LIG디앤에이"),
    "NAVER": ("네이버",),
    "메리츠금융지주": ("메리츠금융",),
    "삼성에스디에스": ("삼성SDS",),
    "하나금융지주": ("하나금융",),
    "한국전력": ("한전",),
    "한국타이어앤테크놀로지": ("한국타이어",),
}

PREFERRED_SHARE_SUFFIXES = ("2우B", "우")
AMBIGUOUS_PARENT_NAMES = {"HD현대", "LS", "한화"}
SPORTS_TERMS = (
    "KBO",
    "강우콜드",
    "야구",
    "연승",
    "연패",
    "이글스",
    "이닝",
    "타율",
    "타자",
    "투수",
    "홈런",
)


def _normalise_match_text(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣中&]", "", value.casefold())


def _contains_standalone_name(title: str, company: str) -> bool:
    pattern = "\\s*".join(re.escape(character) for character in company)
    particle = r"(?:은|는|이|가|을|를|의|와|과|도|만)?"
    return bool(
        re.search(
            rf"(?<![0-9A-Za-z가-힣]){pattern}{particle}(?![0-9A-Za-z가-힣])",
            title,
            flags=re.IGNORECASE,
        )
    )


def _is_stock_news_relevant(company: str, title: str) -> bool:
    if not title or any(term.casefold() in title.casefold() for term in SPORTS_TERMS):
        return False

    aliases = {company, *COMPANY_ALIASES.get(company, ())}
    for suffix in PREFERRED_SHARE_SUFFIXES:
        if company.endswith(suffix):
            base = company.removesuffix(suffix)
            aliases.update((f"{base}우선주", f"{base} 우선주"))

    if company in AMBIGUOUS_PARENT_NAMES:
        return any(_contains_standalone_name(title, alias) for alias in aliases)

    normalised_title = _normalise_match_text(title)
    return any(
        normalised_alias and normalised_alias in normalised_title
        for alias in aliases
        if (normalised_alias := _normalise_match_text(alias))
    )


def _is_market_news_relevant(keyword: str, title: str) -> bool:
    normalised_title = _normalise_match_text(title)
    terms = MARKET_RELEVANCE_TERMS.get(keyword, (keyword,))
    return any(
        normalised_term and normalised_term in normalised_title
        for term in terms
        if (normalised_term := _normalise_match_text(term))
    )


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
        cached_market_records = _read_csv(market_path)
        cached_stock_records = _read_csv(stock_path)
        market_records = [
            record
            for record in cached_market_records
            if _is_market_news_relevant(
                record.get("keyword", ""),
                record.get("title", ""),
            )
        ]
        stock_records = [
            record
            for record in cached_stock_records
            if _is_stock_news_relevant(
                record.get("company", ""),
                record.get("title", ""),
            )
        ]
        _ensure_nonempty(market_records, stock_records)
        if len(market_records) != len(cached_market_records):
            _write_csv(market_path, market_records)
        if len(stock_records) != len(cached_stock_records):
            _write_csv(stock_path, stock_records)
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
        candidate_limit = max(6, max_market_per_keyword * 4)
        results = client().search(
            MARKET_SEARCH_QUERIES.get(keyword, keyword),
            target_date=target_date,
            max_results=candidate_limit,
        )
        return [
            {
                **result,
                "company": "시장전체",
                "ticker": "",
                "keyword": keyword,
            }
            for result in results
            if _is_market_news_relevant(keyword, result.get("title", ""))
        ][:max_market_per_keyword]

    def crawl_stock(ticker: str, company: str) -> list[dict[str, str]]:
        throttle.wait()
        search_company = "네이버" if company == "NAVER" else company
        candidate_limit = max(8, max_stock_per_ticker * 8)
        results = client().search(
            f"{search_company} 주가",
            target_date=target_date,
            max_results=candidate_limit,
        )
        return [
            {
                **result,
                "company": company,
                "ticker": ticker.removesuffix(".KS"),
                "keyword": "",
            }
            for result in results
            if _is_stock_news_relevant(company, result.get("title", ""))
        ][:max_stock_per_ticker]

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
