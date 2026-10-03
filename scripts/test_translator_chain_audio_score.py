"""Numeric spelling is not a semantic acceptance oracle."""

from __future__ import annotations

import unittest

import translator_chain_audio_score as score


class ChainAudioScoreTests(unittest.TestCase):
    def test_numeric_spellings_lower_metric_without_changing_originals(self):
        forms = {
            "Transfer $1540.": "transfer $1540",
            "Transfer one thousand five hundred forty dollars.": "transfer $1540",
        }
        result = score.measure_text(
            "Transfer $1540.",
            "Transfer one thousand five hundred forty dollars.",
            "en",
            lambda text, lang: forms[text],
        )
        self.assertGreater(result["lexical"]["wer"], 0)
        self.assertEqual(result["normalized"]["wer"], 0)
        self.assertEqual(result["reference"], "Transfer $1540.")
        self.assertIn("thousand", result["hypothesis"])

    def test_decimal_sign_time_and_units_do_not_collapse(self):
        for reference, hypothesis in (
            ("1.5", "15"),
            ("-15", "15"),
            ("18:05", "18:50"),
            ("10 billion", "10 million"),
            ("$15", "15 euros"),
        ):
            with self.subTest(reference=reference):
                result = score.measure_text(
                    reference, hypothesis, "en", lambda text, lang: text
                )
                self.assertGreater(result["normalized"]["wer"], 0)

    def test_negation_and_duplicate_identifiers_remain_errors(self):
        for reference, hypothesis in (
            ("Do not mute the microphone.", "Mute the microphone."),
            ("ID 42", "ID 42 42"),
            ("Не отключайте микрофон.", "Отключайте микрофон."),
        ):
            result = score.measure_text(
                reference,
                hypothesis,
                "ru" if "Не" in reference else "en",
                lambda text, lang: text,
            )
            self.assertGreater(result["normalized"]["wer"], 0)

    def test_missing_hypothesis_cannot_be_scored_as_success(self):
        for hypothesis in ("", "  ", None, 12):
            with self.assertRaises(ValueError):
                score.measure_text(
                    "Reference", hypothesis, "en", lambda text, lang: text
                )

    def test_normalizer_failure_is_not_a_lexical_fallback(self):
        def broken(text, lang):
            raise RuntimeError("normalizer unavailable")

        with self.assertRaises(RuntimeError):
            score.measure_text("Reference", "Reference", "en", broken)

    def test_unsupported_language_is_rejected(self):
        with self.assertRaises(ValueError):
            score.measure_text("Reference", "Reference", "de", lambda text, lang: text)


if __name__ == "__main__":
    unittest.main()
