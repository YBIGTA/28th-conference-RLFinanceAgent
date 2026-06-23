import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from demo_app.backend import app as backend_app


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


if __name__ == "__main__":
    unittest.main()
