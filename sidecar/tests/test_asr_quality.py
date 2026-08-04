from __future__ import annotations

from pathlib import Path

import numpy as np

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


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperCt2:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []
        self.failure = failure

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> tuple[object, object]:
        self.calls.append((audio, kwargs))
        if self.failure is not None:
            raise self.failure
        return [FakeSegment(" turbo text ")], object()


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


def test_faster_whisper_ct2_probe_uses_candidate_repository_and_mode_beam() -> None:
    model = FakeWhisperCt2()
    factory_calls: list[tuple[str, dict[str, object]]] = []

    def factory(repository: str, **kwargs: object) -> FakeWhisperCt2:
        factory_calls.append((repository, kwargs))
        return model

    probe = asr_quality.FasterWhisperCt2AsrProbe(
        repository="deepdml/faster-whisper-large-v3-turbo-ct2",
        device="cuda",
        model_factory=factory,
    )

    transcript = probe.transcribe(
        b"\0\0" * 160,
        language=Language.EN,
        mode=TranslationMode.BALANCED,
    )

    assert transcript == "turbo text"
    assert factory_calls == [
        (
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            {
                "device": "cuda",
                "compute_type": "float16",
                "local_files_only": False,
                "num_workers": 1,
            },
        )
    ]
    audio, kwargs = model.calls[0]
    assert audio.dtype == np.float32
    assert kwargs == {
        "language": "en",
        "beam_size": 3,
        "vad_filter": False,
        "condition_on_previous_text": False,
    }


def test_faster_whisper_ct2_probe_falls_back_to_cpu_on_cuda_runtime_failure() -> None:
    model = FakeWhisperCt2()
    factory_calls: list[tuple[str, dict[str, object]]] = []

    def factory(repository: str, **kwargs: object) -> FakeWhisperCt2:
        factory_calls.append((repository, kwargs))
        if kwargs["device"] == "cuda":
            raise RuntimeError("Library libcublas.so.12 is not found")
        return model

    probe = asr_quality.FasterWhisperCt2AsrProbe(
        repository="deepdml/faster-whisper-large-v3-turbo-ct2",
        device="cuda",
        model_factory=factory,
    )

    assert probe.transcribe(
        b"\0\0" * 160,
        language=Language.EN,
        mode=TranslationMode.STREAMING_FIRST,
    ) == "turbo text"
    assert [call[1]["device"] for call in factory_calls] == ["cuda", "cpu"]
    assert factory_calls[1][1]["compute_type"] == "int8"


def test_faster_whisper_ct2_probe_falls_back_to_cpu_on_cuda_inference_failure() -> None:
    cpu_model = FakeWhisperCt2()
    models = {
        "cuda": FakeWhisperCt2(
            failure=RuntimeError("Library libcublas.so.12 is not found")
        ),
        "cpu": cpu_model,
    }
    factory_calls: list[tuple[str, dict[str, object]]] = []

    def factory(repository: str, **kwargs: object) -> FakeWhisperCt2:
        factory_calls.append((repository, kwargs))
        return models[str(kwargs["device"])]

    probe = asr_quality.FasterWhisperCt2AsrProbe(
        repository="deepdml/faster-whisper-large-v3-turbo-ct2",
        device="cuda",
        model_factory=factory,
    )

    assert probe.transcribe(
        b"\0\0" * 160,
        language=Language.RU,
        mode=TranslationMode.STREAMING_FIRST,
    ) == "turbo text"
    assert [call[1]["device"] for call in factory_calls] == ["cuda", "cpu"]
    assert cpu_model.calls[0][1]["language"] == "ru"


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
