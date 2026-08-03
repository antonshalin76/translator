from __future__ import annotations

import builtins
import os
from pathlib import Path
import traceback

import pytest

from translator_sidecar.local.mt import (
    LocalTranslationError,
    NllbTranslator,
    _preserve_24_hour_times,
    _preserve_named_entity_roles,
    _preserve_purchase_order_identifiers,
)
from translator_sidecar.provider_contract import Language, TranslationMode


class FakeSentencePiece:
    def __init__(self) -> None:
        self.decoded: list[str] = []

    def encode(self, text: str, *, out_type: type[str]) -> list[str]:
        assert out_type is str
        assert text == "private source"
        return ["piece-a", "piece-b"]

    def decode(self, tokens: list[str]) -> str:
        self.decoded = tokens
        return " translated output "


class LongSentencePiece(FakeSentencePiece):
    def __init__(self, piece_count: int) -> None:
        super().__init__()
        self._piece_count = piece_count

    def encode(self, text: str, *, out_type: type[str]) -> list[str]:
        assert out_type is str
        assert text == "long private source"
        return [f"piece-{index}" for index in range(self._piece_count)]


class FakeResult:
    hypotheses = [["eng_Latn", "translated", "output"]]


class FakeCTranslate2:
    def __init__(self) -> None:
        self.calls: list[tuple[list[list[str]], dict[str, object]]] = []

    def translate_batch(
        self, source: list[list[str]], **kwargs: object
    ) -> list[FakeResult]:
        self.calls.append((source, kwargs))
        return [FakeResult()]


@pytest.mark.parametrize(
    ("source", "translated", "expected"),
    [
        (
            "Встреча назначена на 13:15.",
            "The meeting is scheduled for 1:15.",
            "The meeting is scheduled for 13:15.",
        ),
        (
            "Не отключайте микрофон до 09:30.",
            "Do not mute the microphone until 9:30.",
            "Do not mute the microphone until 9:30.",
        ),
        (
            "Канал 13 работает.",
            "Channel 1 works.",
            "Channel 1 works.",
        ),
    ],
)
def test_nllb_preserves_unmarked_24_hour_times(
    source: str,
    translated: str,
    expected: str,
) -> None:
    assert _preserve_24_hour_times(source, translated) == expected


@pytest.mark.parametrize(
    ("translated", "expected"),
    [
        (
            "Я подтверждаю номер порядка 104.",
            "Я подтверждаю номер заказа 104.",
        ),
        (
            "Подтверждаю номер приказа 326.",
            "Подтверждаю номер заказа 326.",
        ),
        (
            "Номер заказа 215 подтвержден.",
            "Номер заказа 215 подтвержден.",
        ),
    ],
)
def test_nllb_preserves_purchase_order_identifier_semantics(
    translated: str,
    expected: str,
) -> None:
    assert (
        _preserve_purchase_order_identifiers(
            "I confirm order number 104."
            if "104" in translated
            else (
                "I confirm order number 326."
                if "326" in translated
                else "I confirm order number 215."
            ),
            translated,
            source_language=Language.EN,
            target_language=Language.RU,
        )
        == expected
    )


def test_nllb_does_not_rewrite_order_words_without_matching_source_contract() -> None:
    assert (
        _preserve_purchase_order_identifiers(
            "I confirm sequence number 104.",
            "Я подтверждаю номер порядка 104.",
            source_language=Language.EN,
            target_language=Language.RU,
        )
        == "Я подтверждаю номер порядка 104."
    )


@pytest.mark.parametrize(
    (
        "source",
        "translated",
        "source_language",
        "target_language",
        "expected",
    ),
    [
        (
            "Передайте файл пользователю Roman.",
            "Pass the file to the Roman user.",
            Language.RU,
            Language.EN,
            "Pass the file to Roman.",
        ),
        (
            "Please open document Hotel and do not modify it.",
            "Пожалуйста, откройте документ отель и не изменяйте его.",
            Language.EN,
            Language.RU,
            "Пожалуйста, откройте документ Hotel и не изменяйте его.",
        ),
        (
            "Please open document Hotel and do not modify it.",
            "Пожалуйста, откройте документ Hotel и не изменяйте его.",
            Language.EN,
            Language.RU,
            "Пожалуйста, откройте документ Hotel и не изменяйте его.",
        ),
    ],
)
def test_nllb_preserves_named_entity_roles(
    source: str,
    translated: str,
    source_language: Language,
    target_language: Language,
    expected: str,
) -> None:
    assert (
        _preserve_named_entity_roles(
            source,
            translated,
            source_language=source_language,
            target_language=target_language,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mode", "beam_size"),
    [
        (TranslationMode.QUALITY_FIRST, 4),
        (TranslationMode.BALANCED, 2),
        (TranslationMode.STREAMING_FIRST, 1),
    ],
)
def test_nllb_uses_exact_language_and_eos_contract(
    tmp_path: Path,
    mode: TranslationMode,
    beam_size: int,
) -> None:
    runtime = FakeCTranslate2()
    tokenizer = FakeSentencePiece()
    translator = NllbTranslator(
        tmp_path,
        translator=runtime,
        tokenizer=tokenizer,
    )

    result = translator.translate(
        "private source",
        source_language=Language.RU,
        target_language=Language.EN,
        mode=mode,
    )

    assert result == "translated output"
    assert runtime.calls == [
        (
            [["rus_Cyrl", "piece-a", "piece-b", "</s>"]],
            {
                "target_prefix": [["eng_Latn"]],
                "beam_size": beam_size,
                "max_decoding_length": 96,
            },
        )
    ]
    assert tokenizer.decoded == ["translated", "output"]


@pytest.mark.parametrize(
    ("mode", "expected_max_decoding_length"),
    [
        (TranslationMode.QUALITY_FIRST, 512),
        (TranslationMode.BALANCED, 392),
        (TranslationMode.STREAMING_FIRST, 392),
    ],
)
def test_nllb_scales_decoding_length_for_long_podcast_source(
    tmp_path: Path,
    mode: TranslationMode,
    expected_max_decoding_length: int,
) -> None:
    runtime = FakeCTranslate2()
    translator = NllbTranslator(
        tmp_path,
        translator=runtime,
        tokenizer=LongSentencePiece(180),
    )

    assert (
        translator.translate(
            "long private source",
            source_language=Language.RU,
            target_language=Language.EN,
            mode=mode,
        )
        == "translated output"
    )

    assert runtime.calls[0][1]["max_decoding_length"] == expected_max_decoding_length


def test_nllb_strips_optional_eos_without_dropping_last_real_token(
    tmp_path: Path,
) -> None:
    runtime = FakeCTranslate2()
    tokenizer = FakeSentencePiece()
    translator = NllbTranslator(
        tmp_path,
        translator=runtime,
        tokenizer=tokenizer,
    )
    FakeResult.hypotheses = [["eng_Latn", "translated", "output", "</s>"]]
    try:
        translator.translate(
            "private source",
            source_language=Language.RU,
            target_language=Language.EN,
            mode=TranslationMode.BALANCED,
        )
    finally:
        FakeResult.hypotheses = [["eng_Latn", "translated", "output"]]

    assert tokenizer.decoded == ["translated", "output"]


def test_nllb_supports_reverse_direction_with_same_resident_model(
    tmp_path: Path,
) -> None:
    runtime = FakeCTranslate2()
    tokenizer = FakeSentencePiece()
    translator = NllbTranslator(
        tmp_path,
        translator=runtime,
        tokenizer=tokenizer,
    )

    translator.translate(
        "private source",
        source_language=Language.EN,
        target_language=Language.RU,
        mode=TranslationMode.BALANCED,
    )

    source, kwargs = runtime.calls[0]
    assert source[0][0] == "eng_Latn"
    assert kwargs["target_prefix"] == [["rus_Cyrl"]]


def test_nllb_rejects_same_language_and_empty_source_before_inference(
    tmp_path: Path,
) -> None:
    runtime = FakeCTranslate2()
    translator = NllbTranslator(
        tmp_path,
        translator=runtime,
        tokenizer=FakeSentencePiece(),
    )

    with pytest.raises(LocalTranslationError, match="language pair"):
        translator.translate(
            "private source",
            source_language=Language.RU,
            target_language=Language.RU,
            mode=TranslationMode.BALANCED,
        )
    with pytest.raises(LocalTranslationError, match="empty"):
        translator.translate(
            " ",
            source_language=Language.RU,
            target_language=Language.EN,
            mode=TranslationMode.BALANCED,
        )

    assert runtime.calls == []


@pytest.mark.parametrize(
    ("device", "compute_type"),
    [("cpu", "int8"), ("cuda", "int8_float16")],
)
def test_nllb_factory_receives_offline_local_runtime_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    compute_type: str,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def translator_factory(path: str, **kwargs: object) -> FakeCTranslate2:
        calls.append((path, kwargs))
        return FakeCTranslate2()

    def tokenizer_factory(path: str) -> FakeSentencePiece:
        calls.append((path, {}))
        return FakeSentencePiece()

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    translator = NllbTranslator.load(
        tmp_path,
        device=device,
        translator_factory=translator_factory,
        tokenizer_factory=tokenizer_factory,
    )

    assert isinstance(translator, NllbTranslator)
    assert calls == [
        (
            str(tmp_path),
            {
                "device": device,
                "compute_type": compute_type,
                "inter_threads": 1,
            },
        ),
        (str(tmp_path / "sentencepiece.bpe.model"), {}),
    ]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("missing_module", ["ctranslate2", "sentencepiece"])
def test_nllb_load_sanitizes_missing_dependency_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_module: str,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == missing_module:
            raise ModuleNotFoundError("private dependency marker")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    translator_factory = None
    tokenizer_factory = None
    if missing_module == "ctranslate2":

        def load_tokenizer(_path: str) -> FakeSentencePiece:
            return FakeSentencePiece()

        tokenizer_factory = load_tokenizer
    else:

        def load_translator(_path: str, **_kwargs: object) -> FakeCTranslate2:
            return FakeCTranslate2()

        translator_factory = load_translator

    with pytest.raises(LocalTranslationError, match="could not be loaded") as raised:
        NllbTranslator.load(
            tmp_path,
            device="cpu",
            translator_factory=translator_factory,
            tokenizer_factory=tokenizer_factory,
        )

    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private dependency marker" not in rendered
