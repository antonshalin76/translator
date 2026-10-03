"""Contract checks for the bounded, private cloud diagnostic."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import translator_cloud_diagnostic as diagnostic


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CloudDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = []
        self.screen_cases = []
        original = []
        candidate = []
        for index in range(24):
            language = "ru_ru" if index < 12 else "en_us"
            origin_id = f"case-{index:02d}"
            audio = self.root / f"{origin_id}.wav"
            audio.write_bytes(b"RIFF" + bytes([index]) * 16)
            sample = {
                "origin_id": origin_id,
                "condition": "clean",
                "language": language,
                "speaker_id": f"speaker-{index}",
                "reference": f"reference {index}",
                "critical_labels": ["numbers"] if index % 3 == 0 else [],
                "audio_file": audio.name,
                "sha256": digest(audio),
            }
            self.samples.append(sample)
            self.screen_cases.append({"origin_id": origin_id, "condition": "clean"})
            common = {
                "type": "attempt",
                "origin_id": origin_id,
                "condition": "clean",
                "language": language,
                "speaker_id": sample["speaker_id"],
                "reference": sample["reference"],
                "critical_labels": sample["critical_labels"],
                "wav_sha256": sample["sha256"],
                "mode": "quality_first",
                "voice_gender": "female",
                "status": "completed",
                "asr_text": f"heard {index}",
                "mt_text": f"translated {index}",
            }
            original_models = {
                "asr": {"id": "small", "state": "ready", "device": "cuda"},
                "mt": {
                    "id": "nllb-200-distilled-600m-ct2-int8",
                    "state": "ready",
                    "device": "cuda",
                },
                "tts": {"id": "piper-medium", "state": "ready", "device": "cpu"},
            }
            original.append(
                common
                | {
                    "backend": "original_main_small_nllb_piper",
                    "effective_models_open": {
                        "provider_state": "ready",
                        "models": original_models,
                    },
                    "effective_models_after": {
                        "provider_state": "ready",
                        "models": original_models,
                    },
                }
            )
            for backend in ("small_nllb_same_code", "nllb", "hy"):
                models = {
                    "asr": {
                        "id": "faster-whisper-small"
                        if backend == "small_nllb_same_code"
                        else "faster-whisper-large-v3-turbo",
                        "state": "ready",
                        "device": "cuda",
                    },
                    "mt": {
                        "id": "hy-mt2-1.8b-gguf-q4-k-m"
                        if backend == "hy"
                        else "nllb-200-distilled-600m-ct2-int8",
                        "state": "ready",
                        "device": "cuda",
                    },
                    "tts": {"id": "piper-medium", "state": "ready", "device": "cpu"},
                }
                candidate.append(
                    common
                    | {
                        "backend": backend,
                        "effective_models_open": models,
                        "effective_models_after": models,
                    }
                )
        self.manifest = self.root / "manifest.json"
        self.screen = self.root / "screen.json"
        self.original = self.root / "original.jsonl"
        self.candidate = self.root / "candidate.jsonl"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "purpose": "asr_independent_holdout",
                    "samples": self.samples,
                }
            )
        )
        self.screen.write_text(json.dumps({"cases": self.screen_cases}))
        self.original.write_text("\n".join(map(json.dumps, original)) + "\n")
        self.candidate.write_text("\n".join(map(json.dumps, candidate)) + "\n")

    def load(self):
        return diagnostic.load_cases(
            self.manifest,
            self.screen,
            self.original,
            self.candidate,
            tuple(
                digest(path)
                for path in (self.manifest, self.screen, self.original, self.candidate)
            ),
        )

    def test_preflight_binds_all_four_arms_before_requests(self) -> None:
        cases = self.load()
        self.assertEqual(len(cases), 24)
        self.assertEqual(set(cases[0]["arms"]), set(diagnostic.ARMS))
        rows = [json.loads(line) for line in self.candidate.read_text().splitlines()]
        rows[-1]["reference"] = "wrong source"
        self.candidate.write_text("\n".join(map(json.dumps, rows)) + "\n")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            self.load()

    def test_blinding_is_balanced_across_24_cases(self) -> None:
        label_counts = {arm: {label: 0 for label in "ABCD"} for arm in diagnostic.ARMS}
        for index in range(24):
            mapping = diagnostic.blind_mapping(index)
            for label, arm in mapping.items():
                label_counts[arm][label] += 1
        self.assertTrue(
            all(
                count == 6
                for labels in label_counts.values()
                for count in labels.values()
            )
        )

    def test_fallback_model_rejected_before_requests(self) -> None:
        rows = [json.loads(line) for line in self.candidate.read_text().splitlines()]
        rows[-1]["effective_models_after"]["asr"]["id"] = "fallback-small"
        self.candidate.write_text("\n".join(map(json.dumps, rows)) + "\n")
        with self.assertRaisesRegex(ValueError, "model identity"):
            self.load()

    def test_failed_provider_remains_in_private_receipt_and_denominator(self) -> None:
        cases = self.load()
        output = self.root / "result.jsonl"
        calls = []

        def openai(audio, language):
            calls.append("openai")
            if len(calls) == 1:
                raise TimeoutError("secret raw error")
            return {"text": "heard", "model": "openai-test"}

        def google(audio, language):
            calls.append("google")
            return {"text": "heard", "model": "google-test"}

        def judge(source, variants):
            calls.append("claude")
            return {
                label: {
                    basis: {"verdict": "PASS", "critical_errors": []}
                    for basis in ("reference", "own_asr")
                }
                for label in variants
            }

        rows = diagnostic.run_cases(cases, output, openai, google, judge, limit=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["openai"]["status"], "ERROR")
        self.assertEqual(rows[0]["status"], "INCOMPLETE")
        self.assertNotIn("secret", output.read_text())
        self.assertEqual(rows[0]["google"]["status"], "COMPLETED")
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(len(output.read_text().splitlines()), 2)
        self.assertEqual(diagnostic.summarize(rows)["requested_cases"], 2)
        self.assertFalse(diagnostic.summarize(rows)["complete_24"])

    def test_send_time_hash_mismatch_skips_all_external_calls(self) -> None:
        cases = self.load()
        (self.root / self.samples[0]["audio_file"]).write_bytes(b"changed")
        calls = []

        def external(*args):
            calls.append(args)
            raise AssertionError("must not call provider")

        rows = diagnostic.run_cases(
            cases,
            self.root / "changed.jsonl",
            external,
            external,
            external,
            limit=1,
        )
        self.assertEqual(calls, [])
        self.assertEqual(rows[0]["status"], "AUDIO_CHANGED")

    def test_remote_cleanup_uncertainty_stops_all_later_requests(self) -> None:
        calls = []

        def openai(audio, language):
            calls.append("openai")
            return {"text": "heard", "model": "openai-test"}

        def google(audio, language):
            calls.append("google")
            raise diagnostic.RemoteCleanupUncertain("files/owned-test")

        def judge(source, variants):
            calls.append("claude")
            raise AssertionError("must not judge after uncertain cleanup")

        rows = diagnostic.run_cases(
            self.load(),
            self.root / "cleanup.jsonl",
            openai,
            google,
            judge,
            limit=2,
        )
        self.assertEqual(calls, ["openai", "google"])
        self.assertEqual(rows[0]["status"], "INCOMPLETE")
        self.assertEqual(rows[0]["google"]["remote_identity"], "files/owned-test")
        self.assertEqual(rows[1]["status"], "BLOCKED_REMOTE_CLEANUP")
        self.assertEqual(rows[1]["openai"]["status"], "NOT_RUN")
        self.assertEqual(
            diagnostic.summarize(rows)["providers"]["google"],
            {
                "COMPLETED": 0,
                "ERROR": 1,
                "NOT_RUN": 1,
            },
        )

    def test_claude_structured_text_after_thinking_block(self) -> None:
        provider = diagnostic.Providers({"ANTHROPIC_API_KEY": "test"})
        self.addCleanup(provider.close)
        verdicts = {
            label: {
                basis: {"verdict": "PASS", "critical_errors": []}
                for basis in ("reference", "own_asr")
            }
            for label in "ABCD"
        }
        provider.post = lambda *args, **kwargs: {
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "internal"},
                {"type": "text", "text": json.dumps(verdicts)},
            ],
        }
        self.assertEqual(provider.claude("source", {}), verdicts)

    def test_control_cleanup_uncertainty_retains_private_identity(self) -> None:
        class Providers:
            def openai(self, audio, language):
                return {"text": "", "model": "test"}

            def google(self, audio, language):
                raise diagnostic.RemoteCleanupUncertain("files/owned-test")

        control_results = diagnostic.controls(Providers())
        output = self.root / "control-failure.jsonl"
        diagnostic.write_control_failure(output, control_results, ("a", "b", "c", "d"))
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        record = json.loads(output.read_text())
        self.assertEqual(
            record["controls"]["google"]["remote_identity"], "files/owned-test"
        )
        self.assertEqual(record["controls"]["openai"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
