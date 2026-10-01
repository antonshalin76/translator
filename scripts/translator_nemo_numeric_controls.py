"""Explicit CPU grammar integration gate in the pinned evaluator environment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translator_chain_audio_score import NemoNumerals, measure_text


class NemoNormalizationTests(unittest.TestCase):
    def test_real_ru_en_numeric_equivalence_and_contrasts(self):
        with tempfile.TemporaryDirectory() as directory:
            normalizer = NemoNumerals(Path(directory))
            for language, written, spoken in (
                (
                    "en",
                    "Transfer $1540.",
                    "Transfer one thousand five hundred forty dollars.",
                ),
                (
                    "ru",
                    "Переведите 1540 долларов.",
                    "Переведите одну тысячу пятьсот сорок долларов.",
                ),
            ):
                with self.subTest(language=language):
                    self.assertEqual(
                        measure_text(written, spoken, language, normalizer.normalize)[
                            "normalized"
                        ]["wer"],
                        0,
                    )
                    for wrong in (
                        spoken.replace("forty", "fifty").replace("сорок", "пятьдесят"),
                        spoken.replace("dollars", "euros").replace("долларов", "евро"),
                    ):
                        self.assertGreater(
                            measure_text(
                                written, wrong, language, normalizer.normalize
                            )["normalized"]["wer"],
                            0,
                        )
            for language, pairs in (
                (
                    "en",
                    (
                        ("1.5", "15"),
                        ("-15", "15"),
                        ("18:05", "18:50"),
                        ("ten billion", "ten million"),
                        ("Do not mute the microphone", "Mute the microphone"),
                        ("ID 42", "ID 42 42"),
                    ),
                ),
                (
                    "ru",
                    (
                        ("1,5", "15"),
                        ("-15", "15"),
                        ("18:05", "18:50"),
                        ("десять миллиардов", "десять миллионов"),
                        ("Не отключайте микрофон", "Отключайте микрофон"),
                        ("Код 42", "Код 42 42"),
                    ),
                ),
            ):
                for written, wrong in pairs:
                    with self.subTest(language=language, written=written):
                        self.assertGreater(
                            measure_text(
                                written, wrong, language, normalizer.normalize
                            )["normalized"]["wer"],
                            0,
                        )


if __name__ == "__main__":
    unittest.main()
