"""Fail-closed checks for the private paired MT diagnostic."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import translator_mt_mdc_pair as pair
from translator_sidecar.provider_contract import Language


class MtMdcPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "hy.gguf"
        self.model.write_bytes(b"pinned-test-model")
        self.turbo = self.root / "turbo.json"
        self.turbo.write_text(
            json.dumps(
                {
                    "source_head": "test-head",
                    "results": [
                        {
                            "origin_id": f"case-{index}",
                            "condition": "clean",
                            "language": "ru_ru" if index < 6 else "en_us",
                            "status": "completed",
                            "transcript": "Тест." if index < 6 else "Test.",
                        }
                        for index in range(12)
                    ],
                }
            )
        )
        self.screen = self.root / "screen.json"
        self.screen.write_text(
            json.dumps(
                {
                    "turbo_report_sha256": pair.sha256(self.turbo),
                    "cases": [
                        {"origin_id": f"case-{index}", "condition": "clean"}
                        for index in range(12)
                    ],
                }
            )
        )

    def command(self, layers: str, *, extras: tuple[str, ...] = ()) -> list[str]:
        return [
            str(pair.HY_SERVER),
            "--model",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            "11578",
            "--threads",
            "2",
            "--threads-batch",
            "2",
            "--parallel",
            "1",
            "--ctx-size",
            "2048",
            "--gpu-layers",
            layers,
            "--jinja",
            "--log-disable",
            *extras,
        ]

    def run_hy(
        self,
        command: list[str],
        requested: str | None,
        output: Path,
        *,
        exe: str | None = None,
        listen_host: str = "127.0.0.1",
        gguf_digest: str | None = None,
    ) -> None:
        process = SimpleNamespace(
            uids=lambda: SimpleNamespace(real=os.getuid()),
            exe=lambda: exe or str(pair.HY_SERVER),
            cmdline=lambda: command,
            cpu_affinity=lambda: [0, 1],
            create_time=lambda: 1.0,
            is_running=lambda: True,
            net_connections=lambda kind: [
                SimpleNamespace(
                    status="LISTEN", laddr=SimpleNamespace(ip=listen_host, port=11578)
                )
            ],
        )
        argv = [
            "runner",
            "--screen",
            str(self.screen),
            "--screen-sha256",
            pair.sha256(self.screen),
            "--case-count",
            "12",
            "--turbo",
            str(self.turbo),
            "--backend",
            "hy_mt2",
            "--output",
            str(output),
            "--hy-model",
            str(self.model),
            "--hy-server-pid",
            "1234",
        ]
        if requested is not None:
            argv.extend(("--hy-gpu-layers", requested))
        original_sha256 = pair.sha256
        with (
            patch.object(sys, "argv", argv),
            patch.object(pair.psutil, "Process", return_value=process),
            patch.object(
                pair,
                "sha256",
                side_effect=lambda path: (
                    "test-server-binary"
                    if path == pair.HY_SERVER
                    else original_sha256(path)
                ),
            ),
            patch.object(
                pair, "HY_GGUF_SHA256", gguf_digest or pair.sha256(self.model)
            ),
        ):
            pair.main()

    def test_exact_cpu_and_gpu_commands_record_requested_layers(self) -> None:
        for requested, layers in ((None, 0), ("99", 99)):
            with self.subTest(layers=layers):
                output = self.root / f"valid-{layers}.json"
                with patch.object(pair, "hy_translate", return_value="translated"):
                    self.run_hy(self.command(str(layers)), requested, output)
                report = json.loads(output.read_text())
                self.assertEqual(report["requested_hy_gpu_layers"], layers)
                self.assertEqual(len(report["cases"]), 12)
                self.assertEqual(report["hy_server_command"], self.command(str(layers)))
                self.assertEqual(report["hy_server_cpu_affinity"], [0, 1])

    def test_mismatched_ambiguous_or_missing_command_rejects_before_http(self) -> None:
        bad = [
            (self.command("0"), "99"),
            (self.command("99"), "0"),
            (self.command("0", extras=("--gpu-layers", "99")), "0"),
            (self.command("0", extras=("--model", str(self.model))), "0"),
            (self.command("0", extras=("--lora", "private.gguf")), "0"),
            (self.command("0", extras=("--control-vector", "private.gguf")), "0"),
            (self.command("0", extras=("--chat-template-file", "private.jinja")), "0"),
            (self.command("0", extras=("-ngl", "99")), "0"),
            (self.command("0", extras=("--n-gpu-layers=99",)), "0"),
            (self.command("0", extras=("-m", str(self.model))), "0"),
            (self.command("0", extras=("--model=other",)), "0"),
            ([str(pair.HY_SERVER), "-ngl", "99", *self.command("0")[1:]], "0"),
            ([str(pair.HY_SERVER), "--model=other", *self.command("0")[1:]], "0"),
            ([str(pair.HY_SERVER), "--gpu-layers=99", *self.command("0")[1:]], "0"),
            ([str(pair.HY_SERVER), "--gpu-layers", "0", "--jinja"], "0"),
            ([str(pair.HY_SERVER), "--model", "--gpu-layers", "0", "--jinja"], "0"),
            ([str(pair.HY_SERVER), "--model", str(self.model), "--jinja"], "0"),
            (self.command("0")[:-2] + ["--gpu-layers"], "0"),
            (self.command("0"), "-1"),
        ]
        for index, (command, requested) in enumerate(bad):
            with self.subTest(index=index), patch.object(pair.OPENER, "open") as http:
                output = self.root / f"bad-{index}.json"
                with self.assertRaisesRegex(ValueError, "GPU layers|server identity"):
                    self.run_hy(command, requested, output)
                http.assert_not_called()
                self.assertFalse(output.exists())

    def test_nllb_rejects_gpu_option_before_model_load(self) -> None:
        argv = [
            "runner",
            "--screen",
            str(self.screen),
            "--screen-sha256",
            pair.sha256(self.screen),
            "--turbo",
            str(self.turbo),
            "--backend",
            "nllb",
            "--output",
            str(self.root / "nllb.json"),
            "--hy-gpu-layers",
            "99",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(pair.NllbTranslator, "load") as load,
            patch.object(pair.OPENER, "open") as http,
            self.assertRaisesRegex(ValueError, "Hy GPU layers"),
        ):
            pair.main()
        load.assert_not_called()
        http.assert_not_called()

    def test_wrong_executable_model_or_listener_rejects_before_http(self) -> None:
        for index, overrides in enumerate(
            (
                {"exe": "/usr/bin/other-server"},
                {"gguf_digest": "0" * 64},
                {"listen_host": "0.0.0.0"},
            )
        ):
            with self.subTest(index=index), patch.object(pair.OPENER, "open") as http:
                output = self.root / f"wrong-server-{index}.json"
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_hy(self.command("0"), None, output, **overrides)
                http.assert_not_called()
                self.assertFalse(output.exists())

    def test_non_numeric_gpu_option_fails_argparse_before_http(self) -> None:
        argv = [
            "runner",
            "--screen",
            str(self.screen),
            "--turbo",
            str(self.turbo),
            "--backend",
            "hy_mt2",
            "--output",
            str(self.root / "bad-numeric.json"),
            "--hy-gpu-layers",
            "unknown",
        ]
        stderr = io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch.object(pair.OPENER, "open") as http,
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as error,
        ):
            pair.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid int value", stderr.getvalue())
        http.assert_not_called()

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
