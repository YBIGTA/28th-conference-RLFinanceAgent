import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from demo_app.backend import app as backend_app
from demo_app.backend import pipeline


class DemoAuthTest(unittest.TestCase):
    def test_access_token_requires_matching_cookie(self):
        with patch.dict(os.environ, {"DEMO_PASSWORD": "conference-pass"}):
            missing = SimpleNamespace(cookies={})
            matching = SimpleNamespace(
                cookies={
                    backend_app.AUTH_COOKIE: backend_app._access_token(
                        "conference-pass"
                    )
                }
            )

            self.assertFalse(backend_app._has_demo_access(missing))
            self.assertTrue(backend_app._has_demo_access(matching))

    def test_access_token_changes_when_password_changes(self):
        self.assertNotEqual(
            backend_app._access_token("first"),
            backend_app._access_token("second"),
        )

    def test_latest_news_requires_authentication(self):
        with patch.dict(os.environ, {"DEMO_PASSWORD": "conference-pass"}):
            with self.assertRaises(HTTPException) as raised:
                backend_app.latest_news(SimpleNamespace(cookies={}))

        self.assertEqual(raised.exception.status_code, 401)

    def test_latest_news_returns_authenticated_cache(self):
        request = SimpleNamespace(
            cookies={
                backend_app.AUTH_COOKIE: backend_app._access_token(
                    "conference-pass"
                )
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            news_dir = Path(temporary) / "news" / "2026-06-23"
            news_dir.mkdir(parents=True)
            (news_dir / "market_news.csv").write_text(
                "date,title\n2026-06-23,시장 뉴스\n",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {"DEMO_PASSWORD": "conference-pass"}),
                patch.object(pipeline, "RUNTIME_DIR", Path(temporary)),
            ):
                result = backend_app.latest_news(request)

        self.assertEqual(result["date"], "2026-06-23")
        self.assertEqual(result["market_count"], 1)


if __name__ == "__main__":
    unittest.main()
