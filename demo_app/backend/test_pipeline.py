import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
