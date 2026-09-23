"""Focused offline checks for the MDC ASR evidence harness."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import translator_mdc_asr_run as runner
import translator_mdc_asr_score as scorer
import translator_mdc_corpus as corpus


class CorpusTests(unittest.TestCase):
    def test_speaker_balanced_selection_keeps_critical_cases(self) -> None:
        forced = set().union(*corpus.CRITICAL_TEST["en"].values())
        ids = sorted(forced) + [str(100000 + index) for index in range(45)]
        rows = [
            {
                "audio_id": audio_id,
                "client_id": f"speaker-{index}",
                "split": "test",
                "transcription": "This is not a quiet recording",
                "duration_ms": "8000",
                "quality_tags": "",
            }
            for index, audio_id in enumerate(ids)
        ]
        selected = corpus.select(rows, "en", "test")
        self.assertEqual(len(selected), 40)
        self.assertEqual(len({row["client_id"] for row in selected}), 40)
        self.assertTrue(forced <= {row["audio_id"] for row in selected})
        self.assertEqual(len(corpus.select_noise(selected, "en")), 8)
        self.assertEqual(selected, corpus.select(rows, "en", "test"))

    def test_speech_shaped_noise_is_deterministic_at_target_snr(self) -> None:
        signal = (
            np.convolve(
                np.random.default_rng(7).normal(size=16000),
                np.ones(17) / 17,
                mode="same",
            ).astype(np.float32)
            * 0.2
        )
        first, snr = corpus.noise_at_10db(signal, "fixed-case")
        second, _ = corpus.noise_at_10db(signal, "fixed-case")
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(snr, 10.0, places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise.wav"
            sf.write(path, first, 16000, subtype="PCM_16")
            stored, rate = sf.read(path, dtype="float32")
            self.assertEqual(rate, 16000)
            self.assertAlmostEqual(
                corpus.realized_snr_db(signal, stored), 10.0, delta=0.5
            )


class ReportTests(unittest.TestCase):
    def test_audio_tamper_fails_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "clip.wav"
            audio.write_bytes(b"initial-audio")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "purpose": "asr_model_selection",
                        "samples": [
                            {
                                "origin_id": "fixture",
                                "audio_file": "clip.wav",
                                "sha256": hashlib.sha256(
                                    audio.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                )
            )
            expected = runner.sha256(manifest)
            self.assertEqual(len(runner.validate_manifest(manifest, expected)), 1)
            audio.write_bytes(b"different-audio")
            with self.assertRaises(ValueError):
                runner.validate_manifest(manifest, expected)

    def test_paired_delta_and_speaker_cluster(self) -> None:
        baseline = [
            {
                "audio_file": "a",
                "speaker_id": "one",
                "reference": "no five",
                "transcript": "no five",
            },
            {
                "audio_file": "b",
                "speaker_id": "two",
                "reference": "one two",
                "transcript": "one two",
            },
        ]
        candidate = [
            {**baseline[0], "transcript": "no"},
            {**baseline[1], "transcript": "one"},
        ]
        result = scorer.paired_speaker_interval(baseline, candidate, repetitions=100)
        self.assertEqual(result["qwen_minus_turbo_wer"], 0.5)
        self.assertEqual(result["speaker_cluster_bootstrap_95pct"], [0.5, 0.5])
        candidate[0]["reference"] = "not five"
        with self.assertRaises(ValueError):
            scorer.paired_speaker_interval(baseline, candidate, repetitions=100)


if __name__ == "__main__":
    unittest.main()
