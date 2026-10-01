"""Versioned, library-backed lexical metrics; not a semantic truth oracle."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import jiwer

PIPELINE = "nemo-tn-itn-conservative-lexical-v1"
LEXICAL = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes(
            {r"(?<!\d)[.,!?;:]|[.,!?;:](?!\d)": " ", r'["“”«»()]': " "}
        ),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def _metric(reference: str, hypothesis: str) -> dict:
    result = jiwer.process_words(
        reference, hypothesis, reference_transform=LEXICAL, hypothesis_transform=LEXICAL
    )
    return {
        "word_errors": result.substitutions + result.deletions + result.insertions,
        "reference_words": result.hits + result.substitutions + result.deletions,
        "wer": result.wer,
    }


def measure_text(reference: str, hypothesis: str, language: str, canonicalize) -> dict:
    if language not in {"ru", "en"} or any(
        not isinstance(value, str) or not value.strip()
        for value in (reference, hypothesis)
    ):
        raise ValueError("missing text or unsupported language")
    normalized_reference = canonicalize(reference, language)
    normalized_hypothesis = canonicalize(hypothesis, language)
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (normalized_reference, normalized_hypothesis)
    ):
        raise ValueError("normalizer produced no text")
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "lexical": _metric(reference, hypothesis),
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
        "normalized": _metric(normalized_reference, normalized_hypothesis),
        "pipeline": PIPELINE,
    }


class NemoNumerals:
    """NeMo owns the numeric grammar; no project-specific value substitutions."""

    def __init__(self, cache: Path):
        from nemo_text_processing.inverse_text_normalization.inverse_normalize import (
            InverseNormalizer,
        )
        from nemo_text_processing.text_normalization.normalize import Normalizer

        if not cache.is_dir() or cache.is_symlink() or cache.stat().st_mode & 0o077:
            raise ValueError("normalization requires a private existing cache")
        self.identity = {
            name: version(name) for name in ("nemo-text-processing", "pynini", "jiwer")
        }
        if self.identity["nemo-text-processing"] != "1.2.0":
            raise ValueError("normalization grammar version changed")
        self.identity["pipeline"] = PIPELINE
        self.grammars = {}
        for language in ("en", "ru"):
            self.grammars[language] = (
                Normalizer(
                    input_case="cased",
                    lang=language,
                    deterministic=language == "en",
                    cache_dir=str(cache),
                ),
                InverseNormalizer(lang=language, cache_dir=str(cache)),
            )

    def normalize(self, text: str, language: str) -> str:
        if (
            language not in self.grammars
            or not isinstance(text, str)
            or not 1 <= len(text) <= 5000
        ):
            raise ValueError("unsupported normalization input")
        spoken, inverse = self.grammars[language]
        return inverse.inverse_normalize(
            spoken.normalize(text, verbose=False).lower(), verbose=False
        )
