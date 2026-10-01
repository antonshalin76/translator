"""Fail-closed contracts for the saved direct-synthesis TTS pilot."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

import translator_cloud_tts_pilot as pilot


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CloudTtsPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reports = {}
        for backend in ("piper", "supertonic"):
            rows = []
            for index in range(8):
                wav = self.root / f"{backend}-{index}.wav"
                with wave.open(str(wav), "wb") as source:
                    source.setnchannels(1)
                    source.setsampwidth(2)
                    source.setframerate(16000)
                    source.writeframes(bytes([index]) * 320)
                rows.append(
                    {
                        "case": "negation_amount"
                        if index % 4 < 2
                        else "time_correction",
                        "lang": "ru" if index < 4 else "en",
                        "gender": "female" if index % 2 == 0 else "male",
                        "text": f"Target number {index}",
                        "wav": wav.name,
                        "wav_sha256": digest(wav),
                    }
                )
            path = self.root / f"{backend}.json"
            path.write_text(json.dumps({"backend": backend, "results": rows}))
            self.reports[backend] = path

    def load(self):
        return pilot.load_cases(
            self.reports["piper"],
            self.reports["supertonic"],
            digest(self.reports["piper"]),
            digest(self.reports["supertonic"]),
        )

    def test_pairs_and_hashes_required(self) -> None:
        self.assertEqual(len(self.load()), 16)
        rows = json.loads(self.reports["supertonic"].read_text())
        rows["results"][0]["text"] = "different target"
        self.reports["supertonic"].write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "unpaired"):
            self.load()

    def test_failed_transcription_is_retained_and_not_passed(self) -> None:
        attempts = []

        def transcribe(audio, language):
            attempts.append(language)
            if len(attempts) == 1:
                raise TimeoutError("private error")
            return {"model": "test-model", "text": "Target number 1"}

        output = self.root / "private.jsonl"
        rows = pilot.run_cases(self.load(), output, transcribe)
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0]["status"], "ERROR")
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("private error", output.read_text())
        self.assertFalse(pilot.summarize(rows)["complete_16"])

    def test_duplicate_supertonic_case_is_rejected(self) -> None:
        rows = json.loads(self.reports["supertonic"].read_text())
        rows["results"][1]["case"] = rows["results"][0]["case"]
        rows["results"][1]["gender"] = rows["results"][0]["gender"]
        rows["results"][1]["text"] = rows["results"][0]["text"]
        self.reports["supertonic"].write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()

    def test_wav_path_escape_and_symlink_are_rejected(self) -> None:
        rows = json.loads(self.reports["piper"].read_text())
        rows["results"][0]["wav"] = "../outside.wav"
        self.reports["piper"].write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "basename"):
            self.load()
        rows["results"][0]["wav"] = "link.wav"
        (self.root / "link.wav").symlink_to(self.root / "piper-0.wav")
        self.reports["piper"].write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.load()


if __name__ == "__main__":
    unittest.main()
