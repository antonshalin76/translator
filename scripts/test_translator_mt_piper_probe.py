"""Fail-closed checks for the saved-MT-text product Piper probe."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from translator_mt_piper_probe import (
    CASE_IDS,
    SCREEN_SHA256,
    measure_pcm,
    paired_cases,
    read_report,
    run_order,
)


class MtPiperProbeTests(unittest.TestCase):
    def test_hashed_report_bytes_are_the_parsed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            original = json.dumps({"pinned": True}).encode()
            path.write_bytes(original)
            read_bytes = Path.read_bytes
            calls = 0

            def replace_after_read(candidate: Path) -> bytes:
                nonlocal calls
                calls += 1
                content = read_bytes(candidate)
                candidate.write_bytes(b'{"pinned": false}')
                return content

            with patch.object(Path, "read_bytes", replace_after_read):
                parsed = read_report(path, hashlib.sha256(original).hexdigest())
            self.assertEqual(calls, 1)
            self.assertEqual(parsed, {"pinned": True})
            self.assertNotEqual(path.read_bytes(), original)

    def setUp(self) -> None:
        self.nllb = {
            "screen_sha256": SCREEN_SHA256,
            "manifest_sha256": "manifest",
            "cases": [
                {
                    "origin_id": case_id,
                    "condition": "clean",
                    "source_language": case_id.split("-")[0],
                    "source": f"source-{case_id}",
                    "output": f"nllb-{case_id}",
                }
                for case_id in CASE_IDS
            ],
        }
        self.hy = {
            **self.nllb,
            "cases": [
                {**row, "output": f"hy-{row['origin_id']}"}
                for row in self.nllb["cases"]
            ],
        }

    def test_pairing_requires_same_clean_unique_sources(self) -> None:
        self.assertEqual(len(paired_cases(self.nllb, self.hy)), 8)
        for altered in (
            {**self.hy, "cases": self.hy["cases"][:-1]},
            {**self.hy, "cases": [self.hy["cases"][0], *self.hy["cases"]]},
            {
                **self.hy,
                "cases": [
                    {**self.hy["cases"][0], "source": "changed"},
                    *self.hy["cases"][1:],
                ],
            },
            {
                **self.hy,
                "cases": [{**self.hy["cases"][0], "output": ""}, *self.hy["cases"][1:]],
            },
            {
                **self.hy,
                "cases": [
                    {**self.hy["cases"][0], "condition": "noisy"},
                    *self.hy["cases"][1:],
                ],
            },
            {**self.hy, "manifest_sha256": "different"},
        ):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                paired_cases(self.nllb, altered)

    def test_two_pass_order_is_counterbalanced(self) -> None:
        self.assertEqual(run_order(0), ("nllb", "hy_gpu"))
        self.assertEqual(run_order(1), ("hy_gpu", "nllb"))
        with self.assertRaises(ValueError):
            run_order(2)

    def test_product_frames_must_be_exact_and_non_silent(self) -> None:
        class FakeTts:
            def __init__(self, frames: list[bytes]) -> None:
                self.frames = frames
                self.kwargs = None

            def synthesize_frames(self, _text: str, **kwargs):
                self.kwargs = kwargs
                yield from self.frames

        good = FakeTts([b"\x01\x00" * 480, b"\x00" * 960])
        result = measure_pcm(good, "hello", "en")
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["duration_ms"], 40)
        self.assertEqual(
            result["pcm_sha256"],
            hashlib.sha256(b"\x01\x00" * 480 + b"\x00" * 960).hexdigest(),
        )
        self.assertFalse(good.kwargs["continuation"])
        for frames in ([], [b"\x00" * 960], [b"\x01" * 959]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                measure_pcm(FakeTts(frames), "hello", "en")


if __name__ == "__main__":
    unittest.main()
