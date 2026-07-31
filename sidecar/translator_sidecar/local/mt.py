"""Offline CTranslate2 NLLB adapter for the local provider."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from translator_sidecar.provider_contract import Language, TranslationMode


_LANGUAGE_TOKENS = {
    Language.RU: "rus_Cyrl",
    Language.EN: "eng_Latn",
}
_BEAM_SIZE = {
    TranslationMode.QUALITY_FIRST: 4,
    TranslationMode.BALANCED: 2,
    TranslationMode.STREAMING_FIRST: 1,
}
_TIME_24_RE = re.compile(r"(?<!\d)(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)(?!\d)")
_PURCHASE_ORDER_ID_RE = re.compile(
    r"\border\s+(?:number|no\.?)\s*#?\s*(?P<identifier>\d+)\b",
    flags=re.IGNORECASE,
)
_RU_RECIPIENT_ENTITY_RE = re.compile(
    r"\bпользователю\s+(?P<entity>[A-Z][A-Za-z0-9_-]*)\b",
)
_EN_DOCUMENT_ENTITY_RE = re.compile(
    r"\bdocument\s+(?P<entity>[A-Z][A-Za-z0-9_-]*)\b",
)


class LocalTranslationError(RuntimeError):
    """The local translation request failed without exposing spoken text."""


def _preserve_24_hour_times(source: str, translated: str) -> str:
    result = translated
    for source_match in _TIME_24_RE.finditer(source):
        source_hour = int(source_match.group("hour"))
        if source_hour < 13:
            continue
        minute = source_match.group("minute")
        target_hour = source_hour - 12
        target_pattern = re.compile(
            rf"(?<!\d)0?{target_hour}:{minute}(?!\d)",
        )
        target_match = target_pattern.search(result)
        if target_match is None:
            continue
        suffix = result[target_match.end() : target_match.end() + 8]
        if re.match(r"\s*[ap]\.?m\.?\b", suffix, flags=re.IGNORECASE):
            continue
        result = (
            result[: target_match.start()]
            + source_match.group(0)
            + result[target_match.end() :]
        )
    return result


def _preserve_purchase_order_identifiers(
    source: str,
    translated: str,
    *,
    source_language: Language,
    target_language: Language,
) -> str:
    if source_language is not Language.EN or target_language is not Language.RU:
        return translated
    result = translated
    for source_match in _PURCHASE_ORDER_ID_RE.finditer(source):
        identifier = re.escape(source_match.group("identifier"))
        correct = re.compile(
            rf"\b(?:номер\s+заказа|заказ\s+номер)\s+{identifier}\b",
            flags=re.IGNORECASE,
        )
        if correct.search(result):
            continue
        mistranslated = re.compile(
            rf"\bномер\s+[А-Яа-яЁё-]+\s+(?={identifier}\b)",
            flags=re.IGNORECASE,
        )
        result = mistranslated.sub("номер заказа ", result, count=1)
    return result


def _preserve_named_entity_roles(
    source: str,
    translated: str,
    *,
    source_language: Language,
    target_language: Language,
) -> str:
    result = translated
    if source_language is Language.RU and target_language is Language.EN:
        for source_match in _RU_RECIPIENT_ENTITY_RE.finditer(source):
            entity = source_match.group("entity")
            ambiguous = re.compile(
                rf"\b(?:the\s+)?{re.escape(entity)}\s+user\b",
                flags=re.IGNORECASE,
            )
            result = ambiguous.sub(entity, result, count=1)
    elif source_language is Language.EN and target_language is Language.RU:
        for source_match in _EN_DOCUMENT_ENTITY_RE.finditer(source):
            entity = source_match.group("entity")
            correct = re.compile(
                rf"\bдокумент\s+{re.escape(entity)}\b",
                flags=re.IGNORECASE,
            )
            if correct.search(result):
                continue
            translated_label = re.compile(
                r"\bдокумент\s+[A-Za-zА-Яа-яЁё0-9_-]+\b",
                flags=re.IGNORECASE,
            )
            result = translated_label.sub(f"документ {entity}", result, count=1)
    return result


class NllbTranslator:
    def __init__(
        self,
        model_path: Path,
        *,
        translator: Any,
        tokenizer: Any,
    ) -> None:
        self.model_path = model_path
        self._translator = translator
        self._tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        model_path: Path,
        *,
        device: str,
        translator_factory: Any | None = None,
        tokenizer_factory: Any | None = None,
    ) -> NllbTranslator:
        if not model_path.is_absolute() or not model_path.is_dir():
            raise LocalTranslationError("local MT model path is unavailable")
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        compute_type = "int8_float16" if device == "cuda" else "int8"
        try:
            if translator_factory is None:
                from ctranslate2 import Translator

                translator_factory = Translator
            if tokenizer_factory is None:
                from sentencepiece import SentencePieceProcessor

                def load_tokenizer(path: str) -> SentencePieceProcessor:
                    return SentencePieceProcessor(model_file=path)

                tokenizer_factory = load_tokenizer
            translator = translator_factory(
                str(model_path),
                device=device,
                compute_type=compute_type,
                inter_threads=1,
            )
            tokenizer = tokenizer_factory(str(model_path / "sentencepiece.bpe.model"))
        except Exception:
            raise LocalTranslationError(
                "local MT runtime could not be loaded"
            ) from None
        return cls(
            model_path,
            translator=translator,
            tokenizer=tokenizer,
        )

    def translate(
        self,
        text: str,
        *,
        source_language: Language,
        target_language: Language,
        mode: TranslationMode,
    ) -> str:
        normalized = text.strip()
        if not normalized:
            raise LocalTranslationError("source text is empty")
        if source_language is target_language:
            raise LocalTranslationError("language pair is not supported")
        source_token = _LANGUAGE_TOKENS[source_language]
        target_token = _LANGUAGE_TOKENS[target_language]
        try:
            pieces = self._tokenizer.encode(normalized, out_type=str)
            source = [source_token, *pieces, "</s>"]
            results = self._translator.translate_batch(
                [source],
                target_prefix=[[target_token]],
                beam_size=_BEAM_SIZE[mode],
                max_decoding_length=96,
            )
            tokens = list(results[0].hypotheses[0])
            if tokens and tokens[0] == target_token:
                tokens.pop(0)
            if tokens and tokens[-1] == "</s>":
                tokens.pop()
            translated = _preserve_24_hour_times(
                normalized,
                self._tokenizer.decode(tokens).strip(),
            )
            translated = _preserve_purchase_order_identifiers(
                normalized,
                translated,
                source_language=source_language,
                target_language=target_language,
            )
            translated = _preserve_named_entity_roles(
                normalized,
                translated,
                source_language=source_language,
                target_language=target_language,
            )
        except Exception:
            raise LocalTranslationError("local MT inference failed") from None
        if not translated:
            raise LocalTranslationError("local MT returned empty output")
        return translated

    def count_tokens(self, text: str) -> int:
        normalized = text.strip()
        if not normalized:
            return 0
        try:
            return len(self._tokenizer.encode(normalized, out_type=str))
        except Exception:
            raise LocalTranslationError("local MT tokenization failed") from None
