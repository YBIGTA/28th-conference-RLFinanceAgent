import unittest

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


if __name__ == "__main__":
    unittest.main()
