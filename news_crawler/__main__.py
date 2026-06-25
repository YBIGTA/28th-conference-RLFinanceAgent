from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from news_crawler.service import DEFAULT_MARKET_KEYWORDS, crawl_latest_news


REPO_ROOT = Path(__file__).resolve().parents[1]
TICKER_PATH = REPO_ROOT / "demo_app/backend/artifacts/best_model_action_tickers.txt"
STOCK_EMBEDDINGS = REPO_ROOT / "Embeddings/stock_embeddings_16d.jsonl"


def load_tickers(limit: int | None = None) -> list[tuple[str, str]]:
    tickers = [
        line.split("\t", maxsplit=1)[1]
        for line in TICKER_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    wanted = set(tickers)
    names: dict[str, str] = {}
    with STOCK_EMBEDDINGS.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_ticker = str(record.get("ticker", "")).strip()
            ticker = f"{raw_ticker}.KS" if raw_ticker.isdigit() else raw_ticker
            company = str(record.get("company", "")).strip()
            if ticker in wanted and company:
                names[ticker] = company
                if len(names) == len(wanted):
                    break
    pairs = [(ticker, names.get(ticker, ticker.removesuffix(".KS"))) for ticker in tickers]
    return pairs[:limit] if limit else pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl latest Naver market and stock news.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--stock-limit", type=int)
    parser.add_argument("--keyword-limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-interval", type=float, default=0.8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date)
    keywords = DEFAULT_MARKET_KEYWORDS[: args.keyword_limit] if args.keyword_limit else None
    market, stock = crawl_latest_news(
        target_date=target_date,
        tickers=load_tickers(args.stock_limit),
        cache_dir=args.output_dir,
        market_keywords=keywords,
        workers=args.workers,
        request_interval_seconds=args.request_interval,
        force=args.force,
    )
    print(f"market_news.csv: {len(market)} rows")
    print(f"stock_news.csv: {len(stock)} rows")
    print(f"output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
