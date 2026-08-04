from __future__ import annotations

from pathlib import Path

from translator_sidecar.benchmark import asr_quality
from translator_sidecar.provider_contract import Language, TranslationMode


class FakeInputIds:
    shape = (1, 4)


class FakeInputs(dict[str, object]):
    def __init__(self) -> None:
        super().__init__({"input_ids": FakeInputIds(), "audio": "marker"})
        self.to_calls: list[tuple[str, str]] = []

    def to(self, device: str, dtype: str) -> FakeInputs:
        self.to_calls.append((device, dtype))
        return self


class FakeOutputIds:
    def __init__(self) -> None:
        self.slices: list[object] = []

    def __getitem__(self, item: object) -> FakeOutputIds:
        self.slices.append(item)
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.inputs = FakeInputs()
        self.requests: list[dict[str, object]] = []
        self.decoded: list[object] = []

    def apply_transcription_request(self, **kwargs: object) -> FakeInputs:
        self.requests.append(kwargs)
        return self.inputs

    def decode(self, generated_ids: object, **kwargs: object) -> list[str]:
        self.decoded.append((generated_ids, kwargs))
        return [" распознанный текст "]


class FakeModel:
    device = "cuda:0"
    dtype = "float16"

    def __init__(self) -> None:
        self.generated = FakeOutputIds()
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> FakeOutputIds:
        self.generate_calls.append(kwargs)
        return self.generated


def test_qwen_transformers_probe_forces_language_and_decodes_text() -> None:
    processor = FakeProcessor()
    model = FakeModel()
    probe = asr_quality.QwenTransformersAsrProbe(
        repository="Qwen/Qwen3-ASR-0.6B-hf",
        processor_factory=lambda _repository: processor,
        model_factory=lambda _repository: model,
    )

    transcript = probe.transcribe(
        b"\0\0" * 160,
        language=Language.RU,
        mode=TranslationMode.STREAMING_FIRST,
    )

    assert transcript == "распознанный текст"
    assert processor.requests[0]["language"] == "Russian"
    assert Path(str(processor.requests[0]["audio"])).name == "input.wav"
    assert processor.inputs.to_calls == [("cuda:0", "float16")]
    assert model.generate_calls[0]["max_new_tokens"] == 256
    assert processor.decoded[0][1] == {"return_format": "transcription_only"}


def test_asr_quality_candidate_skips_unimplemented_runtime(tmp_path: Path) -> None:
    report = asr_quality._run_candidate(
        "gigaam-v3-e2e-rnnt",
        b"\0\0" * 160,
        language=Language.RU,
        mode=TranslationMode.STREAMING_FIRST,
        reference=None,
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["status"] == "skipped"
    assert "gigaam" in report["skip_reason"]
    assert report["candidate"]["id"] == "gigaam-v3-e2e-rnnt"


def test_asr_quality_candidate_reports_optional_backend_import_error(
    tmp_path: Path, monkeypatch
) -> None:
    class BrokenProbe:
        def transcribe(self, *_args: object, **_kwargs: object) -> str:
            raise ImportError("missing optional decoder")

    monkeypatch.setattr(
        asr_quality,
        "_build_probe",
        lambda *_args, **_kwargs: BrokenProbe(),
    )

    report = asr_quality._run_candidate(
        "qwen3-asr-0.6b-hf",
        b"\0\0" * 160,
        language=Language.RU,
        mode=TranslationMode.STREAMING_FIRST,
        reference=None,
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["status"] == "skipped"
    assert report["skip_reason"] == "ImportError: missing optional decoder"


def test_asr_quality_candidate_reports_unexpected_runtime_failure(
    tmp_path: Path, monkeypatch
) -> None:
    class BrokenProbe:
        def transcribe(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("cuda exploded")

    monkeypatch.setattr(
        asr_quality,
        "_build_probe",
        lambda *_args, **_kwargs: BrokenProbe(),
    )

    report = asr_quality._run_candidate(
        "qwen3-asr-0.6b-hf",
        b"\0\0" * 160,
        language=Language.RU,
        mode=TranslationMode.STREAMING_FIRST,
        reference=None,
        manifest_path=tmp_path / "manifest.json",
    )

    assert report["status"] == "failed"
    assert report["skip_reason"] == "RuntimeError: cuda exploded"


def test_asr_quality_model_parser_defaults_to_executable_local_models() -> None:
    assert asr_quality._parse_model_ids([]) == [
        "faster-whisper-small",
        "faster-whisper-large-v3",
    ]
