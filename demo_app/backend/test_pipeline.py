import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from demo_app.backend import pipeline
from demo_app.backend.pipeline import get_pipeline_service


class PipelineServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = get_pipeline_service()

    def test_saved_model_contract(self):
        health = self.service.health()
        self.assertEqual(health["observation_dim"], 1467)
        self.assertEqual(health["action_dim"], 50)
        self.assertEqual(health["ticker_mapping"], 50)

    def test_latest_date_limit_excludes_open_market_day(self):
        before_close = datetime(2026, 6, 23, 9, 0, tzinfo=pipeline.KST)
        after_close = datetime(2026, 6, 23, 16, 0, tzinfo=pipeline.KST)
        self.assertEqual(
            pipeline._latest_completed_date_limit(before_close),
            "2026-06-22",
        )
        self.assertEqual(
            pipeline._latest_completed_date_limit(after_close),
            "2026-06-23",
        )

    def test_sample_inference_returns_all_tickers(self):
        result = self.service.infer(
            mode="sample",
            cash=10_000_000,
            holdings={},
            news_text="",
        )
        self.assertEqual(len(result["rows"]), 50)
        self.assertEqual(
            result["summary"]["buy_count"]
            + result["summary"]["sell_count"]
            + result["summary"]["hold_count"],
            50,
        )

    def test_unknown_holding_ticker_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "모델 대상이 아닌 종목"):
            self.service.infer(
                mode="sample",
                cash=10_000_000,
                holdings={"999999.KS": 1},
                news_text="",
            )

    def test_buy_cash_is_distributed_across_equal_scores(self):
        actions = np.zeros(50, dtype=np.float32)
        actions[:3] = 1.0
        prices = np.full(50, 100_000.0, dtype=np.float32)
        holdings = np.zeros(50, dtype=int)

        quantities, remaining_cash = pipeline._allocate_proportional_buys(
            actions,
            prices,
            holdings,
            1_000_000,
        )

        np.testing.assert_array_equal(quantities[:3], [3, 3, 3])
        self.assertGreaterEqual(remaining_cash, 0)

    def test_buy_cash_follows_relative_action_scores(self):
        actions = np.zeros(50, dtype=np.float32)
        actions[0] = 1.0
        actions[1] = 0.5
        prices = np.full(50, 100_000.0, dtype=np.float32)
        holdings = np.zeros(50, dtype=int)

        quantities, _ = pipeline._allocate_proportional_buys(
            actions,
            prices,
            holdings,
            1_000_000,
        )

        np.testing.assert_array_equal(quantities[:2], [6, 3])

    def test_buy_allocation_respects_position_cap(self):
        actions = np.zeros(50, dtype=np.float32)
        actions[0] = 1.0
        prices = np.full(50, 10_000.0, dtype=np.float32)
        holdings = np.zeros(50, dtype=int)
        holdings[0] = pipeline.MAX_SHARES - 1

        quantities, _ = pipeline._allocate_proportional_buys(
            actions,
            prices,
            holdings,
            1_000_000,
        )

        self.assertEqual(quantities[0], 1)

    def test_latest_cached_news_uses_newest_available_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            news_root = Path(temporary) / "news"
            older = news_root / "2026-06-22"
            latest = news_root / "2026-06-23"
            older.mkdir(parents=True)
            latest.mkdir(parents=True)
            (older / "market_news.csv").write_text(
                "date,title\n2026-06-22,이전 뉴스\n",
                encoding="utf-8",
            )
            (latest / "market_news.csv").write_text(
                "date,title\n2026-06-23,최신 시장 뉴스\n",
                encoding="utf-8",
            )
            (latest / "stock_news.csv").write_text(
                "date,company,ticker,title\n2026-06-23,삼성전자,005930,최신 종목 뉴스\n",
                encoding="utf-8",
            )

            with patch.object(pipeline, "RUNTIME_DIR", Path(temporary)):
                result = pipeline.load_latest_cached_news()

        self.assertEqual(result["date"], "2026-06-23")
        self.assertEqual(result["market_count"], 1)
        self.assertEqual(result["stock_count"], 1)
        self.assertEqual(result["market_news"][0]["title"], "최신 시장 뉴스")

    def test_crawled_news_embeddings_are_cached(self):
        market_records = [{"title": "코스피 상승 마감"}]
        stock_records = [{"ticker": "005930", "title": "삼성전자 실적 개선"}]
        embedding_response = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.0] * 4096) for _ in range(2)]
        )

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(pipeline, "RUNTIME_DIR", Path(temporary)),
                patch.object(pipeline, "OpenAI") as openai,
                patch.dict(pipeline.os.environ, {"UPSTAGE_API_KEY": "test-key"}),
            ):
                openai.return_value.embeddings.create.return_value = embedding_response
                first, first_mode = self.service._build_live_embeddings(
                    "",
                    "2026-06-22",
                    market_records=market_records,
                    stock_records=stock_records,
                )
                second, second_mode = self.service._build_live_embeddings(
                    "",
                    "2026-06-22",
                    market_records=market_records,
                    stock_records=stock_records,
                )

        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertIn("신규 임베딩", first_mode)
        self.assertIn("임베딩 캐시", second_mode)
        self.assertEqual(openai.return_value.embeddings.create.call_count, 1)

    @patch(
        "news_crawler.service.crawl_latest_news",
        side_effect=RuntimeError("Naver returned no news"),
    )
    def test_latest_news_crawl_failure_falls_back(self, crawl_latest_news):
        market, stock, failed = self.service._collect_latest_news("2026-06-22")

        self.assertEqual(market, [])
        self.assertEqual(stock, [])
        self.assertTrue(failed)
        crawl_latest_news.assert_called_once()


if __name__ == "__main__":
    unittest.main()
