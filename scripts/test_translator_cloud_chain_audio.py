"""Exact whole-chain output custody and automatic diagnostic acceptance."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import translator_cloud_chain_audio as chain
from translator_cloud_diagnostic import ARMS, EXPECTED_MODELS, RemoteCleanupUncertain
from translator_product_pcm import PcmArtifactStore


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verdict(status="PASS", errors=None):
    return {"verdict": status, "critical_errors": errors or []}


def judgments(status="PASS"):
    return {
        label: {basis: verdict(status) for basis in chain.BASES} for label in "ABCD"
    }


GOOD_CONTROLS = {
    name: {"status": "PASS"} for name in ("openai", "google", "claude", "openai_judge")
}


class FakeProviders:
    def __init__(self):
        self.calls = []
        self.fail_google = False
        self.uploads = []

    def openai(self, audio, language):
        self.calls.append("openai")
        self.uploads.append(("openai", audio, language))
        return {
            "model": "test-openai",
            "text": "Do not mute the microphone."
            if language == "en"
            else "Не отключайте микрофон.",
        }

    def google(self, audio, language):
        self.calls.append("google")
        self.uploads.append(("google", audio, language))
        if self.fail_google:
            raise RemoteCleanupUncertain("files/safe-identity")
        return {
            "model": "test-google",
            "text": "Do not mute the microphone."
            if language == "en"
            else "Не отключайте микрофон.",
        }

    def judge(self, provider, source, variants):
        self.calls.append(provider)
        return judgments()


class ChainAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = []
        self.keys = [("ru-a", "clean"), ("en-b", "clean")]
        for kind, arms in (("original", ARMS[:1]), ("candidate", ARMS[1:])):
            folder = self.root / kind
            folder.mkdir(mode=0o700)
            rows = [
                {
                    "type": "header",
                    "schema": "translator.original-main-baseline.v1"
                    if kind == "original"
                    else "translator.product-audio-pair.v2",
                    "input_count": 2,
                    "manifest_sha256": "a" * 64,
                    "screen_sha256": "b" * 64,
                    "turbo_sha256": "c" * 64,
                    "mode": "quality_first",
                    "voice_gender": "female",
                }
            ]
            with PcmArtifactStore(folder) as store:
                for arm in arms:
                    models = {
                        key: {"id": model, "state": "ready", "device": device}
                        for key, model, device in zip(
                            ("asr", "mt", "tts"),
                            EXPECTED_MODELS[arm],
                            ("cuda", "cuda", "cpu"),
                        )
                    }
                    health = (
                        {"provider_state": "ready", "models": models}
                        if kind == "original"
                        else models
                    )
                    for index, (origin, condition) in enumerate(self.keys):
                        target = "en" if index == 0 else "ru"
                        reference = (
                            "Не отключайте микрофон."
                            if index == 0
                            else "Do not mute the microphone."
                        )
                        translated = (
                            "Do not mute the microphone."
                            if index == 0
                            else "Не отключайте микрофон."
                        )
                        pcm = bytes([1 + 2 * ARMS.index(arm) + index, 2]) * 480
                        artifact = store.write_pcm(
                            pcm, hashlib.sha256(pcm).hexdigest(), f"{uuid4().hex}.wav"
                        )
                        artifact.update(
                            session_id=str(uuid4()),
                            direction_id="microphone" if index == 0 else "speaker",
                            stream_id=str(uuid4()),
                            utterance_id=str(uuid4()),
                            input_wav_sha256="d" * 64,
                            target_language=target,
                            requested_voice={
                                "language": target,
                                "gender": "female",
                                "engine": "piper",
                            },
                            effective_models_open=health,
                            effective_models_after=health,
                        )
                        rows.append(
                            {
                                "type": "attempt",
                                "status": "completed",
                                "backend": arm,
                                "origin_id": origin,
                                "condition": condition,
                                "language": "ru_ru" if index == 0 else "en_us",
                                "speaker_id": "publisher-speaker",
                                "reference": reference,
                                "critical_labels": ["negation"],
                                "asr_text": reference,
                                "mt_text": translated,
                                "mode": "quality_first",
                                "voice_gender": "female",
                                "wav_sha256": "d" * 64,
                                "pcm_sha256": artifact["pcm_sha256"],
                                "pcm_artifact": artifact,
                                "effective_models_open": health,
                                "effective_models_after": health,
                            }
                        )
                    if kind == "candidate":
                        rows.append(
                            {"type": "arm_end", "backend": arm, "status": "complete"}
                        )
            rows.append(
                {"type": "terminal", "status": "complete", "attempts": len(arms) * 2}
            )
            path = folder / "receipt.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            path.chmod(0o600)
            self.inputs.append((path, digest(path), folder))

    def load(self):
        return chain.load_cases(
            self.inputs,
            self.keys,
            {
                "manifest_sha256": "a" * 64,
                "screen_sha256": "b" * 64,
                "turbo_sha256": "c" * 64,
            },
        )

    def test_same_case_keys_cannot_relabel_another_screen(self):
        with self.assertRaises(ValueError):
            chain.load_cases(
                self.inputs,
                self.keys,
                {
                    "manifest_sha256": "a" * 64,
                    "screen_sha256": "e" * 64,
                    "turbo_sha256": "c" * 64,
                },
            )

    def test_mutation_after_preflight_stops_subsequent_uploads(self):
        cases = self.load()
        providers = FakeProviders()
        original = providers.openai
        artifact = cases[0]["arms"][ARMS[0]]["pcm_artifact"]

        def mutate(audio, language):
            result = original(audio, language)
            (self.inputs[0][2] / artifact["filename"]).write_bytes(
                b"changed after preflight"
            )
            return result

        providers.openai = mutate
        rows = chain.run_cases(
            cases,
            self.root / "out.jsonl",
            providers,
            lambda text, lang: text,
            GOOD_CONTROLS,
        )
        self.assertEqual(providers.calls, ["openai"])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "AUDIO_CHANGED" for row in rows))

    def rewrite(self, index, mutate):
        path, _, folder = self.inputs[index]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.inputs[index] = path, digest(path), folder

    def test_bound_complete_output_matrix(self):
        cases = self.load()
        self.assertEqual(len(cases), 2)
        self.assertEqual(set(cases[0]["arms"]), set(ARMS))

    def test_complete_success_reaches_both_asr_and_judges_but_not_release(self):
        cases = self.load()
        providers = FakeProviders()
        rows = chain.run_cases(
            cases,
            self.root / "out.jsonl",
            providers,
            lambda text, lang: text,
            GOOD_CONTROLS,
        )
        summary = chain.summarize(rows)
        self.assertTrue(summary["diagnostic_accepted"])
        self.assertFalse(summary["release"])
        self.assertTrue(all(row["release"] is False for row in rows))
        expected = []
        for case in cases:
            for arm in ARMS:
                item = case["arms"][arm]
                with PcmArtifactStore(item["artifact_directory"]) as store:
                    audio = store.read_verified(item["pcm_artifact"])
                expected.extend(
                    (name, audio, item["pcm_artifact"]["target_language"])
                    for name in ("openai", "google")
                )
        self.assertEqual(providers.uploads, expected)
        self.assertEqual(providers.calls.count("claude"), 2)
        self.assertEqual(providers.calls.count("openai_judge"), 2)

    def test_transcription_and_judge_failures_are_retained_without_retry(self):
        for failed in ("openai", "judge"):
            with self.subTest(failed=failed):
                providers = FakeProviders()

                def broken(*args, providers=providers, failed=failed):
                    providers.calls.append(failed)
                    raise ValueError("provider_http_503")

                setattr(providers, failed, broken)
                rows = chain.run_cases(
                    self.load(),
                    self.root / f"{failed}.jsonl",
                    providers,
                    lambda text, lang: text,
                    GOOD_CONTROLS,
                )
                self.assertEqual(len(rows), 2)
                self.assertFalse(chain.summarize(rows)["diagnostic_accepted"])
                self.assertEqual(
                    providers.calls.count(failed), 8 if failed == "openai" else 4
                )

    def test_model_hash_and_duplicate_stream_binding_rejected(self):
        saved = self.inputs[1][0].read_bytes()
        mutations = [
            lambda rows: rows[1]["effective_models_open"]["asr"].__setitem__(
                "id", "foreign"
            ),
            lambda rows: rows[1]["pcm_artifact"].__setitem__(
                "session_id", "not-a-uuid"
            ),
            lambda rows: rows[2]["pcm_artifact"].__setitem__(
                "stream_id", rows[1]["pcm_artifact"]["stream_id"]
            ),
            lambda rows: rows.__setitem__(
                slice(None), [row for row in rows if row is not rows[1]]
            ),
        ]
        for mutate in mutations:
            self.rewrite(1, mutate)
            with self.assertRaises(ValueError):
                self.load()
            self.inputs[1][0].write_bytes(saved)
            path, _, folder = self.inputs[1]
            self.inputs[1] = path, digest(path), folder
        path, _, folder = self.inputs[1]
        self.inputs[1] = path, "0" * 64, folder
        with self.assertRaises(ValueError):
            self.load()

    def test_failed_or_missing_cleanup_terminal_cannot_score(self):
        for fault in ("arm_end", "terminal", "attempt", "duplicate"):
            with self.subTest(fault=fault):
                saved = self.inputs[1][0].read_bytes()
                if fault == "duplicate":
                    self.rewrite(1, lambda rows: rows.insert(1, rows[1]))
                else:
                    self.rewrite(
                        1,
                        lambda rows, fault=fault: (
                            rows.__setitem__(
                                slice(None),
                                [row for row in rows if row["type"] != fault],
                            )
                            if fault == "arm_end"
                            else next(
                                row for row in rows if row["type"] == fault
                            ).__setitem__("status", "failed")
                        ),
                    )
                with self.assertRaises(ValueError):
                    self.load()
                self.inputs[1][0].write_bytes(saved)
                path, _, folder = self.inputs[1]
                self.inputs[1] = path, digest(path), folder

    def test_foreign_identity_and_model_and_receipt_hash_rejected(self):
        self.rewrite(
            1,
            lambda rows: rows[1]["pcm_artifact"].__setitem__(
                "input_wav_sha256", "e" * 64
            ),
        )
        with self.assertRaises(ValueError):
            self.load()

    def test_send_mutation_retains_denominator_without_cloud_calls(self):
        cases = self.load()
        artifact = cases[0]["arms"][ARMS[0]]["pcm_artifact"]
        (self.inputs[0][2] / artifact["filename"]).write_bytes(b"changed")
        providers = FakeProviders()
        rows = chain.run_cases(
            cases,
            self.root / "out.jsonl",
            providers,
            lambda text, lang: text,
            GOOD_CONTROLS,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(providers.calls, [])
        self.assertTrue(all(row["status"] == "AUDIO_CHANGED" for row in rows))

    def test_remote_cleanup_failure_stops_and_retains_every_case(self):
        providers = FakeProviders()
        providers.fail_google = True
        rows = chain.run_cases(
            self.load(),
            self.root / "out.jsonl",
            providers,
            lambda text, lang: text,
            GOOD_CONTROLS,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(providers.calls, ["openai", "google"])
        self.assertEqual(rows[1]["status"], "BLOCKED_REMOTE_CLEANUP")
        self.assertFalse(chain.summarize(rows)["diagnostic_accepted"])
        self.assertTrue(all(row["release"] is False for row in rows))

    def test_control_failure_has_zero_calls_and_full_denominator(self):
        providers = FakeProviders()
        rows = chain.run_cases(
            self.load(),
            self.root / "out.jsonl",
            providers,
            lambda text, lang: text,
            {"openai": {"status": "FAIL"}, "google": {"status": "PASS"}},
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(providers.calls, [])
        self.assertTrue(all(row["status"] == "CONTROL_FAILED" for row in rows))

    def test_audible_only_corruption_is_not_accepted_by_zero_wer(self):
        providers = FakeProviders()

        def judge(provider, source, variants):
            result = judgments()
            result["A"]["openai_tts"] = verdict("FAIL", ["negation"])
            return result

        providers.judge = judge
        rows = chain.run_cases(
            self.load(),
            self.root / "out.jsonl",
            providers,
            lambda text, lang: text,
            GOOD_CONTROLS,
        )
        self.assertEqual(
            rows[0]["arms"][ARMS[0]]["openai"]["metrics"]["normalized"]["wer"], 0
        )
        self.assertFalse(chain.summarize(rows)["diagnostic_accepted"])

    def test_uncertain_or_malformed_judgment_cannot_pass(self):
        for value in (judgments("UNCERTAIN"), {"A": {}}, judgments()):
            if len(value) == 4 and value["A"]["mt"]["verdict"] == "PASS":
                value["A"]["mt"]["critical_errors"] = ["negation"]
            if len(value) != 4 or value["A"]["mt"].get("critical_errors"):
                with self.assertRaises(ValueError):
                    chain.validate_judgment(value)
            else:
                self.assertFalse(chain.accepted(chain.validate_judgment(value)))


class ChainProviderTests(unittest.TestCase):
    def test_output_ru_language_reaches_google_ru_control(self):
        provider = chain.ChainProviders.__new__(chain.ChainProviders)
        with patch.object(
            chain.Providers, "google", return_value={"text": "Речь"}
        ) as base:
            provider.google(b"wav", "ru")
            base.assert_called_once_with(b"wav", "ru_ru")

    def test_openai_judge_uses_pinned_model_stateless_schema_and_rejects_refusal(self):
        provider = chain.ChainProviders.__new__(chain.ChainProviders)
        provider.keys = {"OPENAI_API_KEY": "test-key"}
        observed = []

        def post(url, headers, **kwargs):
            observed.append((url, kwargs["json"]))
            return {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(judgments())}
                        ],
                    }
                ],
            }

        provider.post = post
        self.assertTrue(chain.accepted(provider.judge("openai_judge", "source", {})))
        url, payload = observed[0]
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(payload["model"], chain.OPENAI_JUDGE_MODEL)
        self.assertFalse(payload["store"])
        self.assertTrue(payload["text"]["format"]["strict"])
        provider.post = lambda *args, **kwargs: {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}
            ],
        }
        with self.assertRaises(ValueError):
            provider.judge("openai_judge", "source", {})

    def test_claude_truncation_cannot_be_accepted(self):
        provider = chain.ChainProviders.__new__(chain.ChainProviders)
        provider.keys = {"ANTHROPIC_API_KEY": "test-key"}
        provider.post = lambda *args, **kwargs: {
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": json.dumps(judgments())}],
        }
        with self.assertRaises(ValueError):
            provider.judge("claude", "source", {})


if __name__ == "__main__":
    unittest.main()
