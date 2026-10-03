"""Offline contracts for the AppTek publisher-reference ASR diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import translator_apptek_asr_run as runner
import translator_apptek_asr_score as scorer
import translator_apptek_corpus as corpus


def turn(start: float, end: float, speaker: str, text: str) -> dict:
    return {
        "start": start,
        "end": end,
        "speaker_id": speaker,
        "role": "agent" if speaker == "a" else "customer",
        "gender": "female" if speaker == "a" else "male",
        "text": text,
    }


def call(name: str, turns: list[dict]) -> dict:
    return {
        "file_name": f"audio/{name}.wav",
        "duration": 120.0,
        "segments": turns,
    }


class SelectionTests(unittest.TestCase):
    def test_selects_separate_general_and_critical_calls_independent_of_input_order(
        self,
    ) -> None:
        rows = [
            call(
                "first",
                [
                    turn(1, 5, "a", "Please tell me your request today"),
                    turn(10, 15, "b", "I would like to ask a question"),
                ],
            ),
            call(
                "second",
                [
                    turn(1, 5, "a", "No that is not correct"),
                    turn(10, 15, "b", "It will cost twenty five pounds"),
                    turn(20, 25, "a", "My name is Alice Smith"),
                    turn(30, 35, "b", "Please send the correct details"),
                ],
            ),
            call(
                "third",
                [
                    turn(1, 5, "a", "No that is not correct"),
                    turn(10, 15, "b", "It will cost twenty five pounds"),
                    turn(20, 25, "a", "My name is Alice Smith"),
                ],
            ),
            call(
                "fourth",
                [
                    turn(1, 5, "a", "Please tell me your request today"),
                    turn(10, 15, "b", "I would like to ask a question"),
                ],
            ),
        ]
        selected = corpus.select_accent(rows, "en-AU", excluded=set())
        self.assertEqual(
            selected,
            corpus.select_accent(list(reversed(rows)), "en-AU", excluded=set()),
        )
        self.assertEqual(
            [item["cohort"] for item in selected],
            ["general", "general", "critical", "critical", "critical"],
        )
        self.assertEqual(len({item["origin_id"] for item in selected}), 5)
        self.assertEqual(len({item["speaker_id"] for item in selected[:2]}), 2)
        self.assertEqual(
            selected[0]["source_file"], "diarization/en-AU/audio/first.wav"
        )
        self.assertEqual(
            selected[2]["source_file"], "diarization/en-AU/audio/second.wav"
        )

    def test_full_selection_has_all_accents_and_distinct_call_cohorts(self) -> None:
        rows_by_accent = {}
        for accent in corpus.ACCENTS:
            rows_by_accent[accent] = [
                call(
                    f"{accent}-general",
                    [
                        turn(1, 5, "a", "Please answer my question"),
                        turn(10, 15, "b", "I can answer your question"),
                    ],
                ),
                call(
                    f"{accent}-critical-a",
                    [
                        turn(1, 5, "a", "No that is not correct"),
                        turn(10, 15, "b", "It costs twenty pounds"),
                        turn(20, 25, "a", "My name is Alice Smith"),
                    ],
                ),
                call(
                    f"{accent}-critical-b",
                    [
                        turn(1, 5, "a", "No that is not correct"),
                        turn(10, 15, "b", "It costs twenty pounds"),
                        turn(20, 25, "a", "My name is Alice Smith"),
                    ],
                ),
            ]
        selected = corpus.select_corpus(rows_by_accent, excluded=set())
        self.assertEqual(len(selected), 70)
        self.assertEqual(len({item["origin_id"] for item in selected}), 70)
        self.assertEqual({item["accent"] for item in selected}, set(corpus.ACCENTS))
        self.assertEqual(len({item["source_file"] for item in selected}), 28)
        self.assertEqual(sum(item["cohort"] == "general" for item in selected), 28)
        self.assertEqual(sum(item["cohort"] == "critical" for item in selected), 42)

    def test_rejects_overlap_and_missing_critical_category(self) -> None:
        rows = [
            call(
                "general",
                [
                    turn(1, 5, "a", "Please answer my question"),
                    turn(10, 15, "b", "I can answer your question"),
                ],
            ),
            call(
                "bad-critical",
                [
                    turn(1, 5, "a", "No that is not correct"),
                    turn(5.1, 9.1, "b", "It costs twenty pounds"),
                    turn(20, 25, "a", "My name is Alice Smith"),
                ],
            ),
        ]
        with self.assertRaises(ValueError):
            corpus.select_accent(rows, "en-AU", excluded=set())
        rows[1]["segments"][1]["start"] = 10
        rows[1]["segments"][1]["end"] = 15
        self.assertEqual(len(corpus.select_accent(rows, "en-AU", excluded=set())), 5)
        rows[1]["segments"][2]["text"] = "Please check the address again"
        with self.assertRaises(ValueError):
            corpus.select_accent(rows, "en-AU", excluded=set())

    def test_sample_exact_extraction_and_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x01\x00" * 16000 * 4)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            corpus.verify_source(path, expected, path.stat().st_size)
            data = corpus.extract_clip(path, 8000, 24000)
            with wave.open(str(path), "rb") as wav:
                wav.setpos(8000)
                self.assertEqual(data, wav.readframes(16000))
            with self.assertRaises(ValueError):
                corpus.extract_clip(path, 24000, 8000)
            with self.assertRaises(ValueError):
                corpus.extract_clip(path, 0, 16000 * 5)
            path.write_bytes(b"\x02\x00" * 16000 * 4)
            with self.assertRaises(ValueError):
                corpus.verify_source(path, expected, path.stat().st_size)

    def test_inventory_is_bound_to_fixed_revision_and_hash(self) -> None:
        self.assertEqual(
            corpus.DATASET_REVISION, "b98967d9946f7f59f58d08624a2a00fe98fe0219"
        )
        self.assertEqual(
            corpus.INVENTORY_SHA256,
            "869b00a1654823360aaf51c81e3d0c876712a6d9afd9c1fdc7e912b0cc30cedd",
        )
        with tempfile.TemporaryDirectory() as temporary:
            inventory = Path(temporary) / "inventory.json"
            inventory.write_text(
                json.dumps({"schema": 1, "revision": "wrong", "sources": {}})
            )
            with self.assertRaises(ValueError):
                corpus.load_inventory(inventory)
            packet = {
                "schema": 1,
                "dataset": "apptek-com/apptek_callcenter_dialogues",
                "revision": corpus.DATASET_REVISION,
                "sources": {
                    "diarization/en-AU/audio/source.wav": {
                        "sha256": "a" * 64,
                        "size": 100,
                    }
                },
            }
            inventory.write_text(json.dumps(packet, sort_keys=True))
            expected = hashlib.sha256(inventory.read_bytes()).hexdigest()
            with patch.object(corpus, "INVENTORY_SHA256", expected):
                self.assertEqual(
                    corpus.load_inventory(inventory)["sources"], packet["sources"]
                )
                packet["sources"]["diarization/en-AU/audio/source.wav"]["size"] = 101
                inventory.write_text(json.dumps(packet, sort_keys=True))
                with self.assertRaises(ValueError):
                    corpus.load_inventory(inventory)

    def test_padding_excludes_adjacent_turns_and_pins_metadata_bytes(self) -> None:
        row = call(
            "edge",
            [
                turn(1, 5, "a", "Please answer my question"),
                turn(5.1, 9.1, "b", "It costs twenty pounds"),
            ],
        )
        self.assertEqual(corpus.eligible_turns(row), [])
        row["segments"][1]["start"] = 10
        row["segments"][1]["end"] = 15
        eligible = corpus.eligible_turns(row)
        self.assertEqual(
            (eligible[0]["start_sample"], eligible[0]["end_sample"]), (12000, 84000)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "en-AU-metadata.jsonl"
            path.write_text(json.dumps({**row, "accent": "en-AU"}) + "\n")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(corpus.read_metadata(path, expected, "en-AU")), 1)
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(ValueError):
                corpus.read_metadata(path, expected, "en-AU")


class ReportTests(unittest.TestCase):
    def test_runner_rejects_unpinned_source_or_collapsed_call_cohorts(self) -> None:
        sources = {}
        samples = []
        for accent in corpus.ACCENTS:
            for cohort, labels in (
                ("general", [None, None]),
                ("critical", list(corpus.CRITICAL)),
            ):
                source = f"diarization/{accent}/audio/{cohort}.wav"
                sources[source] = {"sha256": "a" * 64, "size": 100}
                for index, label in enumerate(labels):
                    samples.append(
                        {
                            "origin_id": f"{accent}-{cohort}-{index}",
                            "accent": accent,
                            "cohort": cohort,
                            "critical_label": label,
                            "source_file": source,
                            "source_sha256": "a" * 64,
                            "speaker_id": str(index),
                        }
                    )
        inventory = {"sources": sources}
        runner.verify_manifest_sources(samples, inventory)
        first = samples[0]
        for key, value in (
            ("source_sha256", "b" * 64),
            ("source_file", samples[2]["source_file"]),
            ("accent", "en-ZA"),
        ):
            altered = [{**sample} for sample in samples]
            altered[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.verify_manifest_sources(altered, inventory)
        self.assertEqual(
            first["source_sha256"], sources[first["source_file"]]["sha256"]
        )

    def test_score_rejects_relabelled_model_and_invalid_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = [
                {
                    "origin_id": f"clip-{index}",
                    "reference": "No twenty pounds",
                    "cohort": "general" if index % 5 < 2 else "critical",
                    "accent": corpus.ACCENTS[index // 5],
                    "critical_label": None if index % 5 < 2 else "number",
                }
                for index in range(70)
            ]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "purpose": "apptek_publisher_reference_diagnostic",
                        "revision": corpus.DATASET_REVISION,
                        "samples": samples,
                    }
                )
            )
            base = {
                "schema": 1,
                "purpose": "apptek_publisher_reference_diagnostic",
                "manifest_sha256": corpus.sha256(manifest_path),
                "source_head": "test-head",
                "harness_sha256": {
                    "builder": "c6f2e6280e0760e12edd129866a61a1908aa99b106e3b154c9f378cec8f7e346",
                    "runner": "ea9a814ccd5b3860d2f8e93e18c85c34fd350b6d5d9e42e5336c372f68def221",
                    "model_loader": "8dea8dca8aafcd3558f941aff8b893abdcab1020ff074eb77ec5b470b43cfe6d",
                },
                "results": [
                    {
                        **sample,
                        "status": "completed",
                        "transcript": sample["reference"],
                        "elapsed_ms": 1.0,
                    }
                    for sample in samples
                ],
            }
            turbo = {
                **base,
                "model_id": "turbo",
                "model_identity": {"directory_sha256": "wrong"},
            }
            qwen = {
                **base,
                "model_id": "qwen17",
                "model_identity": {"directory_sha256": "wrong"},
            }
            turbo_path = root / "turbo.json"
            qwen_path = root / "qwen.json"
            turbo_path.write_text(json.dumps(turbo))
            qwen_path.write_text(json.dumps(qwen))
            with self.assertRaises(ValueError):
                scorer.score(manifest_path, turbo_path, qwen_path, root / "score.json")
            turbo["model_identity"] = scorer.EXPECTED_MODEL_IDENTITY["turbo"]
            qwen["model_identity"] = scorer.EXPECTED_MODEL_IDENTITY["qwen17"]
            turbo_path.write_text(json.dumps(turbo))
            qwen_path.write_text(json.dumps(qwen))
            self.assertEqual(
                scorer.score(manifest_path, turbo_path, qwen_path, root / "score.json")[
                    "metrics"
                ]["general"]["n"],
                28,
            )
            qwen_path.write_text(json.dumps({**turbo, "model_id": "qwen17"}))
            with self.assertRaises(ValueError):
                scorer.score(
                    manifest_path, turbo_path, qwen_path, root / "relabelled.json"
                )

    def test_audio_tamper_and_duplicate_ids_fail_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "clip.wav"
            audio.write_bytes(b"initial")
            sample = {
                "origin_id": "one",
                "audio_file": "clip.wav",
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "reference": "No, twenty pounds",
                "cohort": "critical",
                "accent": "en-AU",
                "source_file": "audio/source.wav",
                "source_sha256": "source-hash",
                "speaker_id": "speaker-a",
                "start_sample": 0,
                "end_sample": 16000,
                "critical_label": "negation",
            }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "purpose": "apptek_publisher_reference_diagnostic",
                        "samples": [sample],
                    }
                )
            )
            expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(len(runner.validate_corpus(manifest, expected)), 1)
            audio.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                runner.validate_corpus(manifest, expected)
            audio.write_bytes(b"initial")
            manifest.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "purpose": "apptek_publisher_reference_diagnostic",
                        "samples": [sample, sample],
                    }
                )
            )
            with self.assertRaises(ValueError):
                runner.validate_corpus(
                    manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()
                )

    def test_scorer_rejects_incomplete_or_consistently_corrupted_reports(self) -> None:
        sample = {
            "origin_id": "one",
            "audio_file": "clip.wav",
            "sha256": "audio-hash",
            "reference": "No, twenty pounds",
            "cohort": "critical",
            "accent": "en-AU",
            "speaker_id": "speaker-a",
            "start_sample": 10,
            "end_sample": 16010,
            "critical_label": "negation",
        }
        report_row = {
            **sample,
            "status": "completed",
            "transcript": "No twenty pounds",
            "elapsed_ms": 10.0,
        }
        result = scorer.compare([sample], [report_row], [report_row])
        self.assertEqual(result["critical"]["turbo_wer"], 0.0)
        with self.assertRaises(ValueError):
            scorer.compare(
                [sample],
                [{**report_row, "reference": "wrong"}],
                [{**report_row, "reference": "wrong"}],
            )
        with self.assertRaises(ValueError):
            scorer.compare([sample], [report_row], [{**report_row, "status": "failed"}])
        with self.assertRaises(ValueError):
            scorer.compare([sample], [report_row], [])
        for changed in (
            {"sha256": "different"},
            {"cohort": "general"},
            {"start_sample": 11},
        ):
            with self.assertRaises(ValueError):
                scorer.compare([sample], [{**report_row, **changed}], [report_row])
        with self.assertRaises(ValueError):
            scorer.compare([sample], [report_row, report_row], [report_row])
        with self.assertRaises(ValueError):
            scorer.compare(
                [sample],
                [report_row, {**report_row, "origin_id": "extra"}],
                [report_row],
            )
        for invalid in (-1.0, math.nan, math.inf, "1.0", True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scorer.compare(
                    [sample], [{**report_row, "elapsed_ms": invalid}], [report_row]
                )


if __name__ == "__main__":
    unittest.main()
