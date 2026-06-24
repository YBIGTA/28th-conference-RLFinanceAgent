from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from news_crawler.naver import parse_display_date, parse_search_results
from news_crawler.service import (
    _is_market_news_relevant,
    _is_stock_news_relevant,
    crawl_latest_news,
)


HTML = """
<ul class="list_news">
  <li>
    <a data-heatmap-target=".tit" href="https://news.example.com/semiconductor">삼성전자, 반도체 투자 확대</a>
    <div data-sds-comp="Profile"><span>테스트경제</span><span>5분 전</span></div>
  </li>
  <li>
    <a data-heatmap-target=".tit">시장 금리 안정 기대</a>
    <div data-sds-comp="Profile"><span>샘플일보</span><span>2026.06.22.</span></div>
  </li>
</ul>
"""


class NaverParserTest(unittest.TestCase):
    def test_relative_and_absolute_dates(self):
        reference = date(2026, 6, 22)
        self.assertEqual(parse_display_date("5분 전", reference_date=reference), reference)
        self.assertEqual(parse_display_date("2026.06.22.", reference_date=reference), reference)

    def test_parse_search_results(self):
        records = parse_search_results(
            HTML,
            target_date=date(2026, 6, 22),
            max_results=2,
            reference_date=date(2026, 6, 22),
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["press"], "테스트경제")
        self.assertEqual(records[0]["link"], "https://news.example.com/semiconductor")

    def test_relative_date_uses_crawl_date(self):
        html = """
        <a data-heatmap-target=".tit">지난 거래일 뉴스</a>
        <div data-sds-comp="Profile"><span>테스트경제</span><span>4일 전</span></div>
        """
        records = parse_search_results(
            html,
            target_date=date(2026, 6, 19),
            max_results=1,
            reference_date=date(2026, 6, 23),
        )
        self.assertEqual(len(records), 1)

    def test_stock_relevance_rejects_unrelated_and_sports_news(self):
        self.assertTrue(
            _is_stock_news_relevant(
                "삼성전자",
                "삼성전자, 차세대 반도체 투자 확대",
            )
        )
        self.assertTrue(
            _is_stock_news_relevant(
                "NAVER",
                "네이버, AI 검색 서비스 개편",
            )
        )
        self.assertFalse(
            _is_stock_news_relevant(
                "KB금융",
                "안송이, 노련한 핀 공략",
            )
        )
        self.assertFalse(
            _is_stock_news_relevant(
                "한화",
                "한화 7연패 탈출, 투수 호투",
            )
        )

    def test_preferred_share_and_parent_company_matching(self):
        self.assertTrue(
            _is_stock_news_relevant(
                "삼성전자우",
                "삼성전자 우선주 거래량 증가",
            )
        )
        self.assertFalse(
            _is_stock_news_relevant(
                "삼성전자우",
                "삼성전자 신제품 공개",
            )
        )
        self.assertFalse(
            _is_stock_news_relevant(
                "HD현대",
                "HD현대중공업, 신규 수주",
            )
        )

    def test_market_relevance_uses_keyword_synonyms(self):
        self.assertTrue(
            _is_market_news_relevant(
                "금리",
                "국채 금리 상승에 코스피 약세",
            )
        )
        self.assertFalse(
            _is_market_news_relevant(
                "중국 경기",
                "김성현, 골프 대회 2라운드 1위",
            )
        )

    @patch("news_crawler.service.NaverNewsClient.search")
    def test_crawl_writes_and_reuses_cache(self, search):
        def search_result(query, **_kwargs):
            title = (
                "코스피 상승 마감"
                if query == "코스피 증시"
                else "삼성전자, 반도체 투자 확대"
            )
            return [
                {
                    "date": "2026.06.22",
                    "title": title,
                    "link": "https://news.example.com/article",
                    "press": "테스트경제",
                    "source": "naver",
                }
            ]

        search.side_effect = search_result
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            first = crawl_latest_news(
                target_date=date(2026, 6, 22),
                tickers=[("005930.KS", "삼성전자")],
                cache_dir=cache_dir,
                market_keywords=["코스피"],
                workers=1,
                request_interval_seconds=0,
            )
            second = crawl_latest_news(
                target_date=date(2026, 6, 22),
                tickers=[("005930.KS", "삼성전자")],
                cache_dir=cache_dir,
                market_keywords=["코스피"],
                workers=1,
                request_interval_seconds=0,
            )
            self.assertEqual(first, second)
            self.assertEqual(search.call_count, 2)
            self.assertTrue((cache_dir / "market_news.csv").exists())
            self.assertTrue((cache_dir / "stock_news.csv").exists())

    @patch("news_crawler.service.NaverNewsClient.search", return_value=[])
    def test_empty_crawl_is_rejected(self, _search):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "returned no news"):
                crawl_latest_news(
                    target_date=date(2026, 6, 22),
                    tickers=[("005930.KS", "삼성전자")],
                    cache_dir=Path(temporary),
                    market_keywords=["코스피"],
                    workers=1,
                    request_interval_seconds=0,
                )


if __name__ == "__main__":
    unittest.main()
