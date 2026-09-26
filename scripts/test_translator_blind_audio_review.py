"""Focused privacy and identity tests for the offline listening packet."""

from __future__ import annotations

import base64
import json
import os
import stat
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import translator_blind_audio_review as review
from translator_blind_audio_review import prepare
from translator_mdc_asr_run import sha256


class BlindAudioReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        samples = []
        for origin, word in (
            ("one", "FIRST PRIVATE REFERENCE"),
            ("two", "SECOND PRIVATE REFERENCE"),
        ):
            audio = self.root / f"{origin}.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\0\0" * 160)
            samples.append(
                {
                    "origin_id": origin,
                    "condition": "clean",
                    "language": "en_us",
                    "audio_file": audio.name,
                    "sha256": sha256(audio),
                    "reference": word,
                }
            )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {"schema": 1, "purpose": "asr_independent_holdout", "samples": samples}
            )
        )
        self.screen = self.root / "screen.json"
        self.screen.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "manifest_sha256": sha256(self.manifest),
                    "cases": [
                        {"origin_id": item["origin_id"], "condition": "clean"}
                        for item in samples
                    ],
                }
            )
        )
        self.output = self.root / "review"
        self.key_dir = self.root / "operator-key"

    def run_prepare(self) -> dict:
        return prepare(
            self.manifest,
            sha256(self.manifest),
            self.screen,
            sha256(self.screen),
            self.output,
            self.key_dir,
        )

    def test_packet_is_blind_private_and_audio_bound(self) -> None:
        receipt = self.run_prepare()
        page = (self.output / "review.html").read_text()
        key = json.loads((self.key_dir / "answer-key.json").read_text())
        self.assertEqual(receipt["cases"], 2)
        self.assertEqual(key["pack_id"], receipt["pack_id"])
        self.assertEqual({item["origin_id"] for item in key["cases"]}, {"one", "two"})
        self.assertNotIn("PRIVATE REFERENCE", page)
        self.assertNotIn("origin_id", page)
        self.assertEqual({p.name for p in self.output.iterdir()}, {"review.html"})
        self.assertIn("if(responses.some(x=>!x.heard))", page)
        for origin in ("one", "two"):
            self.assertIn(
                base64.b64encode((self.root / f"{origin}.wav").read_bytes()).decode(),
                page,
            )
        for path in (
            self.output,
            self.key_dir,
            *self.output.iterdir(),
            *self.key_dir.iterdir(),
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0)

    def test_changed_audio_is_rejected_before_packet_creation(self) -> None:
        (self.root / "one.wav").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "frozen audio changed"):
            self.run_prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.key_dir.exists())

    def test_existing_output_is_not_overwritten(self) -> None:
        self.run_prepare()
        initial_hash = sha256(self.output / "review.html")
        with self.assertRaisesRegex(ValueError, "new private directory"):
            self.run_prepare()
        self.assertEqual(sha256(self.output / "review.html"), initial_hash)

    def test_wrong_screen_hash_is_rejected_before_packet_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen screen changed"):
            prepare(
                self.manifest,
                sha256(self.manifest),
                self.screen,
                "0" * 64,
                self.output,
                self.key_dir,
            )
        self.assertFalse(self.output.exists())

    def test_oversized_packet_is_rejected_before_creation(self) -> None:
        with (
            patch.object(review, "MAX_TOTAL_AUDIO_BYTES", 1),
            self.assertRaisesRegex(ValueError, "audio byte limit"),
        ):
            self.run_prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.key_dir.exists())

    def test_world_readable_parent_is_rejected(self) -> None:
        os.chmod(self.root, 0o755)
        with self.assertRaisesRegex(ValueError, "private directory"):
            self.run_prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
