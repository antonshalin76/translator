"""Fail-closed checks for the private, audio-first adjudication probe."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import translator_audio_adjudication as adjudication


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AudioAdjudicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(mode=0o700)
        self.model_blob = self.blobs / ("sha256-" + "a" * 64)
        self.model_blob.write_bytes(b"local-model")
        self.target = self.root / "target.wav"
        self.contrast = self.root / "contrast.wav"
        self.target.write_bytes(b"target-audio")
        self.contrast.write_bytes(b"contrast-audio")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "purpose": "asr_independent_holdout",
                    "samples": [
                        {
                            "origin_id": "target",
                            "condition": "clean",
                            "language": "en_us",
                            "audio_file": "target.wav",
                            "sha256": digest(self.target),
                        },
                        {
                            "origin_id": "contrast",
                            "condition": "clean",
                            "language": "en_us",
                            "audio_file": "contrast.wav",
                            "sha256": digest(self.contrast),
                        },
                    ],
                }
            )
        )
        self.screen = self.root / "screen.json"
        self.screen.write_text(
            json.dumps({"cases": [{"origin_id": "target", "condition": "clean"}]})
        )
        self.spec = self.root / "spec.json"
        self.spec.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "oracle_source": "written_reference_hypothesis",
                    "manifest_sha256": digest(self.manifest),
                    "screen_sha256": digest(self.screen),
                    "cases": [
                        {
                            "origin_id": "target",
                            "condition": "clean",
                            "question": "What animal is named? Answer one word.",
                            "expected_aliases": ["dog"],
                            "contrast_audio_file": "contrast.wav",
                        }
                    ],
                }
            )
        )
        os.chmod(self.spec, 0o600)
        self.output = self.root / "attempts.jsonl"
        self.calls: list[tuple[str, dict | None]] = []

    def post(self, path: str, body: dict | None) -> dict:
        self.calls.append((path, body))
        if path == "/api/tags":
            return {
                "models": [
                    {
                        "name": "test-local:latest",
                        "digest": "b" * 64,
                        "size": 2_000_000_000,
                    }
                ]
            }
        if path == "/api/show":
            return {
                "details": {"format": "gguf"},
                "capabilities": ["audio"],
                "modelfile": f"FROM {self.model_blob}\n",
                "tensors": [{}],
            }
        assert path == "/v1/chat/completions"
        content = body["messages"][0]["content"]
        if len(content) == 1:
            answer, tokens = "NO_AUDIO", 20
        else:
            audio = base64.b64decode(content[1]["input_audio"]["data"])
            answer, tokens = (
                ("dog", 100)
                if audio == self.target.read_bytes()
                else ("NOT_STATED", 90)
            )
        return {
            "usage": {"prompt_tokens": tokens},
            "choices": [{"message": {"content": answer}}],
        }

    def run_probe(self, post=None) -> list[dict]:
        return adjudication.run(
            self.manifest,
            self.screen,
            self.spec,
            digest(self.spec),
            "test-local:latest",
            "b" * 64,
            self.output,
            self.blobs,
            post or self.post,
        )

    def test_valid_audio_and_two_controls_only_agree_with_written_reference(
        self,
    ) -> None:
        rows = self.run_probe()
        self.assertEqual(rows[0]["decision"], "AGREES_WITH_WRITTEN_REFERENCE")
        with contextlib.redirect_stdout(io.StringIO()) as summary:
            self.assertEqual(adjudication.finish(rows), 0)
        self.assertIn("errors=0", summary.getvalue())
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(
            [path for path, _ in self.calls],
            ["/api/tags", "/api/show"] + ["/v1/chat/completions"] * 3,
        )
        self.assertEqual(
            [
                row["decision"]
                for row in map(json.loads, self.output.read_text().splitlines())
            ],
            [rows[0]["decision"]],
        )
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)
        requests = [body for _, body in self.calls[2:]]
        self.assertEqual(
            [len(body["messages"][0]["content"]) for body in requests], [1, 2, 2]
        )
        self.assertEqual(
            [
                base64.b64decode(
                    body["messages"][0]["content"][1]["input_audio"]["data"]
                )
                for body in requests[1:]
            ],
            [self.contrast.read_bytes(), self.target.read_bytes()],
        )
        self.assertTrue(all(body["max_tokens"] <= 64 for body in requests))
        self.assertTrue(
            all(
                body["messages"][0]["content"][1]["input_audio"]["format"] == "wav"
                for body in requests[1:]
            )
        )
        prompts = [body["messages"][0]["content"][0]["text"] for body in requests]
        self.assertEqual(len(set(prompts)), 1)
        for _, body in self.calls[2:]:
            prompt = body["messages"][0]["content"][0]["text"]
            self.assertNotIn("dog", prompt.casefold())
            self.assertNotIn("reference", json.dumps(body).casefold())

    def test_tampered_target_or_contrast_stops_before_http_and_output(self) -> None:
        for audio in (self.target, self.contrast):
            original = audio.read_bytes()
            audio.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                self.run_probe()
            self.assertEqual(self.calls, [])
            self.assertFalse(self.output.exists())
            audio.write_bytes(original)

    def test_missing_audio_usage_or_wrong_answer_is_unresolved(self) -> None:
        for defect in (
            "no_audio",
            "contrast",
            "contrast_usage",
            "target_usage",
            "missing_usage",
            "target_answer",
        ):
            with self.subTest(defect=defect):
                self.calls.clear()
                self.output = self.root / f"{defect}.jsonl"

                def broken_post(
                    path: str, body: dict | None, defect: str = defect
                ) -> dict:
                    response = self.post(path, body)
                    if path != "/v1/chat/completions":
                        return response
                    content = body["messages"][0]["content"]
                    if defect == "no_audio" and len(content) == 1:
                        response["choices"][0]["message"]["content"] = "dog"
                    elif len(content) == 2:
                        audio = base64.b64decode(content[1]["input_audio"]["data"])
                        is_target = audio == self.target.read_bytes()
                        if defect == "contrast" and not is_target:
                            response["choices"][0]["message"]["content"] = "dog"
                        if (
                            defect == "contrast_usage"
                            and not is_target
                            or defect == "target_usage"
                            and is_target
                        ):
                            response["usage"]["prompt_tokens"] = 20
                        if defect == "missing_usage" and is_target:
                            response.pop("usage")
                        if defect == "target_answer" and is_target:
                            response["choices"][0]["message"]["content"] = "cat"
                    return response

                self.assertEqual(
                    self.run_probe(broken_post)[0]["decision"], "UNRESOLVED"
                )

    def test_no_audio_not_stated_abstention_still_runs_audio_controls(self) -> None:
        def abstaining_post(path: str, body: dict | None) -> dict:
            response = self.post(path, body)
            if (
                path == "/v1/chat/completions"
                and len(body["messages"][0]["content"]) == 1
            ):
                response["choices"][0]["message"]["content"] = "NOT_STATED"
            return response

        self.assertEqual(
            self.run_probe(abstaining_post)[0]["decision"],
            "AGREES_WITH_WRITTEN_REFERENCE",
        )
        self.assertEqual(len(self.calls), 5)

    def test_invalid_spec_or_hash_rejected_before_model_access(self) -> None:
        self.screen.write_text(self.screen.read_text() + " ")
        with self.assertRaises(ValueError):
            self.run_probe()
        self.assertEqual(self.calls, [])
        self.screen.write_text(self.screen.read_text().rstrip())
        original_spec = json.loads(self.spec.read_text())
        for cases in ([], original_spec["cases"] * 2):
            with self.subTest(cases=len(cases)):
                spec = dict(original_spec)
                spec["cases"] = cases
                self.spec.write_text(json.dumps(spec))
                with self.assertRaises(ValueError):
                    self.run_probe()
                self.assertEqual(self.calls, [])
        self.assertFalse(self.output.exists())
        self.spec.write_text(json.dumps(original_spec))
        self.manifest.write_text(self.manifest.read_text() + " ")
        with self.assertRaises(ValueError):
            self.run_probe()
        self.assertEqual(self.calls, [])

    def test_cloud_or_wrong_local_model_is_rejected_before_audio(self) -> None:
        for defect in ("cloud", "digest", "capability"):
            with self.subTest(defect=defect):
                self.calls.clear()

                def invalid_model(
                    path: str, body: dict | None, defect: str = defect
                ) -> dict:
                    result = self.post(path, body)
                    if defect == "cloud" and path == "/api/show":
                        result["modelfile"] = "FROM https://ollama.com/cloud-model\n"
                    if defect == "digest" and path == "/api/tags":
                        result["models"][0]["digest"] = "c" * 64
                    if defect == "capability" and path == "/api/show":
                        result["capabilities"] = ["completion"]
                    return result

                with self.assertRaises(ValueError):
                    self.run_probe(invalid_model)
                self.assertFalse(self.output.exists())
                self.assertNotIn(
                    "/v1/chat/completions", [path for path, _ in self.calls]
                )

    def test_private_output_rejects_symlink_and_repo_location(self) -> None:
        link = self.root / "link"
        link.symlink_to(self.root, target_is_directory=True)
        for output in (
            link / "attempts.jsonl",
            Path(__file__).parent / "attempts.jsonl",
        ):
            with self.subTest(output=output):
                self.output = output
                with self.assertRaises(ValueError):
                    self.run_probe()
                self.assertEqual(self.calls, [])
                self.assertFalse(output.exists())

    def test_model_error_is_retained_without_retry(self) -> None:
        original_post = self.post
        attempts = []

        def failing_post(path: str, body: dict | None) -> dict:
            if path == "/v1/chat/completions":
                attempts.append(len(body["messages"][0]["content"]))
            if (
                path == "/v1/chat/completions"
                and len(body["messages"][0]["content"]) == 2
            ):
                raise TimeoutError("private payload must not enter evidence")
            return original_post(path, body)

        rows = self.run_probe(failing_post)
        self.assertEqual(rows[0]["decision"], "UNRESOLVED")
        self.assertEqual(rows[0]["attempt_status"], "ERROR")
        self.assertEqual(rows[0]["error_type"], "TimeoutError")
        with contextlib.redirect_stdout(io.StringIO()) as summary:
            self.assertEqual(adjudication.finish(rows), 1)
        self.assertIn("errors=1", summary.getvalue())
        self.assertNotIn("private payload", self.output.read_text())
        self.assertEqual(attempts, [1, 2])
        with self.assertRaises(FileExistsError):
            self.run_probe()

    def test_audio_changed_after_preflight_is_never_sent(self) -> None:
        for audio in (self.contrast, self.target):
            with self.subTest(audio=audio.name):
                self.calls.clear()
                self.output = self.root / f"changed-{audio.name}.jsonl"
                original = audio.read_bytes()

                def mutating_post(
                    path: str, body: dict | None, audio: Path = audio
                ) -> dict:
                    response = self.post(path, body)
                    if path == "/api/show":
                        audio.write_bytes(b"changed-after-preflight")
                    return response

                try:
                    rows = self.run_probe(mutating_post)
                    self.assertEqual(rows[0]["attempt_status"], "ERROR")
                    self.assertEqual(rows[0]["decision"], "UNRESOLVED")
                    submitted = [
                        base64.b64decode(
                            body["messages"][0]["content"][1]["input_audio"]["data"]
                        )
                        for path, body in self.calls
                        if path == "/v1/chat/completions"
                        and len(body["messages"][0]["content"]) == 2
                    ]
                    self.assertNotIn(b"changed-after-preflight", submitted)
                finally:
                    audio.write_bytes(original)

    def test_transport_uses_loopback_without_environment_proxy(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b"{}"

        class Opener:
            def open(self, request, timeout):
                self.request, self.timeout = request, timeout
                return Response()

        opener = Opener()
        with patch(
            "translator_audio_adjudication.urllib.request.build_opener",
            return_value=opener,
        ) as build:
            self.assertEqual(adjudication.post_json("/api/tags", None), {})
        self.assertEqual(opener.request.full_url, "http://127.0.0.1:11434/api/tags")
        self.assertLessEqual(opener.timeout, 120)
        self.assertEqual(build.call_args.args[0].proxies, {})

    def test_transport_never_follows_local_redirect_with_audio(self) -> None:
        paths = []
        leaks = []

        class Leak(BaseHTTPRequestHandler):
            def do_GET(self):
                leaks.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *_):
                pass

        class Redirect(BaseHTTPRequestHandler):
            def do_GET(self):
                self.reply()

            def do_POST(self):
                self.reply()

            def reply(self):
                paths.append(self.path)
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{leak_server.server_port}/leak"
                )
                self.end_headers()

            def log_message(self, *_):
                pass

        leak_server = HTTPServer(("127.0.0.1", 0), Leak)
        leak_thread = threading.Thread(target=leak_server.serve_forever, daemon=True)
        leak_thread.start()
        server = HTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(
                adjudication, "OLLAMA", f"http://127.0.0.1:{server.server_port}"
            ):
                for path, body in (
                    ("/api/tags", None),
                    ("/v1/chat/completions", {"audio": "private"}),
                ):
                    with self.assertRaises(ValueError):
                        adjudication.post_json(path, body)
            self.assertEqual(paths, ["/api/tags", "/v1/chat/completions"])
            self.assertEqual(leaks, [])
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
            leak_server.shutdown()
            leak_thread.join(timeout=2)
            leak_server.server_close()


if __name__ == "__main__":
    unittest.main()
