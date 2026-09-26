"""Fail-closed checks for the private paired MT diagnostic."""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import translator_mt_mdc_pair as pair
from translator_sidecar.provider_contract import Language


class MtMdcPairTests(unittest.TestCase):
    def test_http_proxy_is_disabled(self) -> None:
        self.assertFalse(
            any(
                handler.__class__.__name__ == "ProxyHandler"
                for handler in pair.OPENER.handlers
            )
        )

    def test_redirect_is_rejected(self) -> None:
        handler = pair.RejectRedirect()
        with self.assertRaisesRegex(ValueError, "redirect refused"):
            handler.redirect_request(
                None, None, 302, "Found", {}, "https://example.com"
            )

    def test_incomplete_translation_is_not_accepted(self) -> None:
        for reason in ("length", None):
            with self.subTest(reason=reason):
                response = io.BytesIO(
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {"content": "partial"},
                                    "finish_reason": reason,
                                }
                            ]
                        }
                    ).encode()
                )
                with (
                    patch.object(pair, "validate_server"),
                    patch.object(pair.OPENER, "open", return_value=response),
                    self.assertRaisesRegex(RuntimeError, "incomplete translation"),
                ):
                    pair.hy_translate("private input", Language.EN, 1, 1.0)

    def test_wrong_listener_is_rejected(self) -> None:
        process = SimpleNamespace(
            create_time=lambda: 1.0,
            is_running=lambda: True,
            net_connections=lambda kind: [
                SimpleNamespace(
                    status="LISTEN", laddr=SimpleNamespace(ip="0.0.0.0", port=11578)
                )
            ],
        )
        with (
            patch.object(pair.psutil, "Process", return_value=process),
            self.assertRaisesRegex(RuntimeError, "listener is unavailable"),
        ):
            pair.validate_server(123, 1.0)


if __name__ == "__main__":
    unittest.main()
