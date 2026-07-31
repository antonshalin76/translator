from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor
import logging
from math import ceil
from pathlib import Path
from threading import Event, Lock
import traceback
from typing import Any

import numpy as np
import pytest
import soxr

from translator_sidecar.local.tts import (
    PiperTts,
    PiperVoiceRegistry,
    TtsOutputLimit,
    TtsUnavailable,
    TtsUnsupported,
)
from translator_sidecar.provider_contract import (
    Language,
    ModelState,
    TranslationMode,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


class Chunk:
    def __init__(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int = 16_000,
        sample_width: int = 2,
        sample_channels: int = 1,
    ) -> None:
        self.audio_int16_bytes = samples.astype(np.int16).tobytes()
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.sample_channels = sample_channels


class AdmissionProbeLock:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counter_lock = Lock()
        self.attempts = 0
        self.second_attempt = Event()

    def __enter__(self) -> AdmissionProbeLock:
        with self._counter_lock:
            self.attempts += 1
            if self.attempts == 2:
                self.second_attempt.set()
        assert self._lock.acquire(timeout=2)
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self._lock.release()


class FakeVoice:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        failure: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.failure = failure
        self.texts: list[str] = []

    def synthesize(self, text: str) -> object:
        self.texts.append(text)

        def generate() -> object:
            yield from self.chunks
            if self.failure is not None:
                raise self.failure

        return generate()


class FakeResampleStream:
    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        channels: int,
        *,
        dtype: str,
    ) -> None:
        self.init = (input_rate, output_rate, channels, dtype)
        self.calls: list[tuple[np.ndarray, bool]] = []

    def resample_chunk(
        self, samples: np.ndarray, *, last: bool
    ) -> np.ndarray:
        self.calls.append((samples.copy(), last))
        return samples.copy()


def create_voice(
    directory: Path,
    name: str,
) -> Path:
    model_path = directory / f"{name}.onnx"
    model_path.write_bytes(b"model")
    model_path.with_suffix(".onnx.json").write_text("{}", encoding="utf-8")
    return model_path


def profile(
    language: Language,
    gender: VoiceGender,
    *,
    engine: VoiceEngine = VoiceEngine.PIPER,
) -> VoiceProfile:
    return VoiceProfile(
        language=language,
        gender=gender,
        engine=engine,
    )


def test_voice_registry_loads_each_exact_language_gender_preset(
    tmp_path: Path,
) -> None:
    paths = {
        (language, gender): create_voice(
            tmp_path, f"{language.value}-{gender.value}"
        )
        for language in Language
        for gender in VoiceGender
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def factory(path: str, **kwargs: Any) -> FakeVoice:
        calls.append((path, kwargs))
        return FakeVoice([])

    registry = PiperVoiceRegistry(paths, voice_factory=factory)

    for key, path in paths.items():
        assert registry.get(profile(*key)) is registry.get(profile(*key))
        assert calls[-1] == (
            str(path),
            {
                "config_path": str(path.with_suffix(".onnx.json")),
                "use_cuda": False,
            },
        )
    assert len(calls) == 4


def test_voice_registry_prepare_loads_all_presets_without_synthesis(
    tmp_path: Path,
) -> None:
    paths = {
        (language, gender): create_voice(
            tmp_path, f"{language.value}-{gender.value}"
        )
        for language in Language
        for gender in VoiceGender
    }
    voices: list[FakeVoice] = []

    def factory(_path: str, **_kwargs: Any) -> FakeVoice:
        voice = FakeVoice([])
        voices.append(voice)
        return voice

    registry = PiperVoiceRegistry(paths, voice_factory=factory)

    registry.prepare()
    registry.prepare()

    assert len(voices) == len(paths)
    assert all(voice.texts == [] for voice in voices)
    assert all(
        registry.model_state(profile(*key)) is ModelState.READY
        for key in paths
    )


def test_voice_registry_never_falls_back_to_another_gender(
    tmp_path: Path,
) -> None:
    male_path = create_voice(tmp_path, "ru-male")
    registry = PiperVoiceRegistry(
        {(Language.RU, VoiceGender.MALE): male_path},
        voice_factory=lambda _path, **_kwargs: pytest.fail("must not load"),
    )

    with pytest.raises(TtsUnavailable, match="unavailable"):
        registry.get(profile(Language.RU, VoiceGender.FEMALE))


def test_voice_registry_never_falls_back_across_languages(
    tmp_path: Path,
) -> None:
    male_path = create_voice(tmp_path, "ru-male")
    registry = PiperVoiceRegistry(
        {(Language.RU, VoiceGender.MALE): male_path},
        voice_factory=lambda _path, **_kwargs: pytest.fail("must not load"),
    )

    with pytest.raises(TtsUnavailable, match="unavailable"):
        registry.get(profile(Language.EN, VoiceGender.MALE))


@pytest.mark.parametrize(
    ("model_path", "provider_voice_id"),
    [
        ("/tmp/unapproved.onnx", None),
        (None, "unapproved-voice"),
    ],
)
def test_voice_registry_rejects_profile_overrides(
    tmp_path: Path,
    model_path: str | None,
    provider_voice_id: str | None,
) -> None:
    approved = create_voice(tmp_path, "ru-male")
    registry = PiperVoiceRegistry(
        {(Language.RU, VoiceGender.MALE): approved},
        voice_factory=lambda _path, **_kwargs: pytest.fail("must not load"),
    )

    with pytest.raises(TtsUnsupported, match="override"):
        registry.get(
            VoiceProfile(
                language=Language.RU,
                gender=VoiceGender.MALE,
                engine=VoiceEngine.PIPER,
                model_path=model_path,
                provider_voice_id=provider_voice_id,
            )
        )


def test_voice_registry_rejects_non_piper_profile(tmp_path: Path) -> None:
    model_path = create_voice(tmp_path, "ru-male")
    registry = PiperVoiceRegistry(
        {(Language.RU, VoiceGender.MALE): model_path},
    )

    with pytest.raises(TtsUnsupported, match="Piper"):
        registry.get(
            profile(
                Language.RU,
                VoiceGender.MALE,
                engine=VoiceEngine.SILERO,
            )
        )


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda tmp_path: Path("relative.onnx"),
        lambda tmp_path: tmp_path / "missing.onnx",
        lambda tmp_path: create_voice(tmp_path, "missing-config"),
    ],
)
def test_voice_registry_rejects_invalid_local_assets_before_load(
    tmp_path: Path,
    path_factory: Any,
) -> None:
    model_path = path_factory(tmp_path)
    if model_path.name == "missing-config.onnx":
        model_path.with_suffix(".onnx.json").unlink()
    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path},
        voice_factory=lambda _path, **_kwargs: pytest.fail("must not load"),
    )

    with pytest.raises(TtsUnavailable, match="unavailable"):
        registry.get(profile(Language.EN, VoiceGender.FEMALE))


def test_voice_registry_serializes_cold_load(tmp_path: Path) -> None:
    model_path = create_voice(tmp_path, "en-female")
    load_entered = Event()
    release_load = Event()
    calls = 0
    calls_lock = Lock()
    load_lock = AdmissionProbeLock()

    def factory(_path: str, **_kwargs: Any) -> FakeVoice:
        nonlocal calls
        with calls_lock:
            calls += 1
        load_entered.set()
        assert release_load.wait(timeout=2)
        return FakeVoice([])

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path},
        voice_factory=factory,
        load_lock=load_lock,
    )
    selected = profile(Language.EN, VoiceGender.FEMALE)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(registry.get, selected)
        assert load_entered.wait(timeout=2)
        second = pool.submit(registry.get, selected)
        assert load_lock.second_attempt.wait(timeout=2)
        assert not second.done()
        assert calls == 1
        release_load.set()
        assert first.result(timeout=3) is second.result(timeout=3)
    assert calls == 1


def test_tts_frames_across_lazy_chunks_and_pads_final_frame(
    tmp_path: Path,
) -> None:
    model_path = create_voice(tmp_path, "en-female")
    voice = FakeVoice(
        [
            Chunk(np.arange(200, dtype=np.int16)),
            Chunk(np.arange(200, 400, dtype=np.int16)),
        ]
    )
    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path},
        voice_factory=lambda _path, **_kwargs: voice,
    )
    tts = PiperTts(registry)

    frames = list(
        tts.synthesize_frames(
            "final text",
            target_language=Language.EN,
            voice_profile=profile(Language.EN, VoiceGender.FEMALE),
            mode=TranslationMode.BALANCED,
            output_sample_rate_hz=16_000,
            output_channels=1,
            frame_duration_ms=20,
        )
    )

    assert voice.texts == ["final text"]
    assert [len(frame) for frame in frames] == [640, 640]
    assert np.frombuffer(frames[0], dtype=np.int16).tolist() == list(
        range(320)
    )
    final = np.frombuffer(frames[1], dtype=np.int16)
    assert final[:80].tolist() == list(range(320, 400))
    assert np.count_nonzero(final[80:]) == 0


def test_tts_does_not_pull_later_chunks_before_current_frames_are_consumed(
    tmp_path: Path,
) -> None:
    model_path = create_voice(tmp_path, "en-female")
    pulls: list[int] = []

    class PullObservedVoice:
        def synthesize(self, _text: str) -> object:
            def generate() -> object:
                pulls.append(1)
                yield Chunk(
                    np.zeros(22_050 * 2, dtype=np.int16),
                    sample_rate=22_050,
                )
                pulls.append(2)
                yield Chunk(np.zeros(320, dtype=np.int16))

            return generate()

    streams: list[FakeResampleStream] = []

    def resampler_factory(*args: Any, **kwargs: Any) -> FakeResampleStream:
        stream = FakeResampleStream(*args, **kwargs)
        streams.append(stream)
        return stream

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path},
        voice_factory=lambda _path, **_kwargs: PullObservedVoice(),
    )
    frames = PiperTts(
        registry, resampler_factory=resampler_factory
    ).synthesize_frames(
        "text",
        target_language=Language.EN,
        voice_profile=profile(Language.EN, VoiceGender.FEMALE),
        mode=TranslationMode.BALANCED,
        output_sample_rate_hz=16_000,
        output_channels=1,
        frame_duration_ms=20,
    )

    assert len(next(frames)) == 640
    assert pulls == [1]
    assert len(streams[0].calls) == 1
    assert len(streams[0].calls[0][0]) <= 2_205


def test_tts_duplicates_mono_samples_for_stereo(tmp_path: Path) -> None:
    model_path = create_voice(tmp_path, "ru-male")
    voice = FakeVoice([Chunk(np.arange(320, dtype=np.int16))])
    registry = PiperVoiceRegistry(
        {(Language.RU, VoiceGender.MALE): model_path},
        voice_factory=lambda _path, **_kwargs: voice,
    )

    frames = list(
        PiperTts(registry).synthesize_frames(
            "текст",
            target_language=Language.RU,
            voice_profile=profile(Language.RU, VoiceGender.MALE),
            mode=TranslationMode.STREAMING_FIRST,
            output_sample_rate_hz=16_000,
            output_channels=2,
            frame_duration_ms=20,
        )
    )

    stereo = np.frombuffer(frames[0], dtype=np.int16).reshape(-1, 2)
    assert stereo[:, 0].tolist() == list(range(320))
    assert np.array_equal(stereo[:, 0], stereo[:, 1])


def test_tts_streams_bounded_chunks_through_resampler(
    tmp_path: Path,
) -> None:
    model_path = create_voice(tmp_path, "en-male")
    voice = FakeVoice(
        [Chunk(np.arange(11_025, dtype=np.int16), sample_rate=22_050)]
    )
    streams: list[FakeResampleStream] = []

    def resampler_factory(*args: Any, **kwargs: Any) -> FakeResampleStream:
        stream = FakeResampleStream(*args, **kwargs)
        streams.append(stream)
        return stream

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.MALE): model_path},
        voice_factory=lambda _path, **_kwargs: voice,
    )
    frames = list(
        PiperTts(
            registry,
            resampler_factory=resampler_factory,
        ).synthesize_frames(
            "text",
            target_language=Language.EN,
            voice_profile=profile(Language.EN, VoiceGender.MALE),
            mode=TranslationMode.BALANCED,
            output_sample_rate_hz=16_000,
            output_channels=1,
            frame_duration_ms=100,
        )
    )

    assert frames
    assert streams[0].init == (22_050, 16_000, 1, "int16")
    assert max(len(samples) for samples, _last in streams[0].calls) <= 2_205
    assert [last for _samples, last in streams[0].calls].count(True) == 1
    assert streams[0].calls[-1][1]


@pytest.mark.parametrize("output_rate", [16_000, 24_000, 48_000])
@pytest.mark.parametrize("frame_duration_ms", [20, 40, 60, 80, 100])
def test_tts_default_soxr_path_has_exact_rate_and_frame_count(
    tmp_path: Path,
    output_rate: int,
    frame_duration_ms: int,
) -> None:
    model_path = create_voice(tmp_path, "en-male")
    input_samples = np.arange(4_410, dtype=np.int16)
    voice = FakeVoice(
        [Chunk(input_samples, sample_rate=22_050)]
    )
    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.MALE): model_path},
        voice_factory=lambda _path, **_kwargs: voice,
    )

    frames = list(
        PiperTts(registry).synthesize_frames(
            "text",
            target_language=Language.EN,
            voice_profile=profile(Language.EN, VoiceGender.MALE),
            mode=TranslationMode.BALANCED,
            output_sample_rate_hz=output_rate,
            output_channels=1,
            frame_duration_ms=frame_duration_ms,
        )
    )

    frame_samples = output_rate * frame_duration_ms // 1000
    assert len(frames) == ceil(200 / frame_duration_ms)
    assert all(len(frame) == frame_samples * 2 for frame in frames)
    rendered = np.concatenate(
        [np.frombuffer(frame, dtype=np.int16) for frame in frames]
    )
    expected_audio_samples = output_rate // 5
    assert np.count_nonzero(rendered[expected_audio_samples:]) == 0
    assert np.count_nonzero(rendered[:expected_audio_samples]) > 0


def test_tts_stops_before_loading_when_cancelled(tmp_path: Path) -> None:
    model_path = create_voice(tmp_path, "en-male")
    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.MALE): model_path},
        voice_factory=lambda _path, **_kwargs: pytest.fail("must not load"),
    )

    assert (
        list(
            PiperTts(registry).synthesize_frames(
                "text",
                target_language=Language.EN,
                voice_profile=profile(Language.EN, VoiceGender.MALE),
                mode=TranslationMode.BALANCED,
                output_sample_rate_hz=16_000,
                output_channels=1,
                frame_duration_ms=20,
                cancelled=lambda: True,
            )
        )
        == []
    )


def test_tts_cancellation_after_first_frame_stops_source_and_resampler(
    tmp_path: Path,
) -> None:
    model_path = create_voice(tmp_path, "en-male")
    pulls: list[int] = []
    streams: list[FakeResampleStream] = []
    cancelled = False

    class PullObservedVoice:
        def synthesize(self, _text: str) -> object:
            def generate() -> object:
                pulls.append(1)
                yield Chunk(
                    np.zeros(22_050 * 2, dtype=np.int16),
                    sample_rate=22_050,
                )
                pulls.append(2)
                yield Chunk(np.zeros(2_205, dtype=np.int16))

            return generate()

    def resampler_factory(*args: Any, **kwargs: Any) -> FakeResampleStream:
        stream = FakeResampleStream(*args, **kwargs)
        streams.append(stream)
        return stream

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.MALE): model_path},
        voice_factory=lambda _path, **_kwargs: PullObservedVoice(),
    )
    frames = PiperTts(
        registry, resampler_factory=resampler_factory
    ).synthesize_frames(
        "text",
        target_language=Language.EN,
        voice_profile=profile(Language.EN, VoiceGender.MALE),
        mode=TranslationMode.BALANCED,
        output_sample_rate_hz=16_000,
        output_channels=1,
        frame_duration_ms=20,
        cancelled=lambda: cancelled,
    )

    assert len(next(frames)) == 640
    calls_after_first_frame = len(streams[0].calls)
    cancelled = True
    assert list(frames) == []
    assert pulls == [1]
    assert len(streams[0].calls) == calls_after_first_frame


def test_tts_enforces_twelve_second_output_cap(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    speech_marker = "private output cap speech marker"
    model_path = create_voice(tmp_path, "en-male")
    pulls: list[int] = []
    streams: list[Any] = []

    class LazyLongVoice:
        def synthesize(self, _text: str) -> object:
            def generate() -> object:
                for index in range(130):
                    pulls.append(index)
                    yield Chunk(
                        np.zeros(2_205, dtype=np.int16),
                        sample_rate=22_050,
                    )

            return generate()

    class CountingSoxr:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = soxr.ResampleStream(*args, **kwargs)
            self.calls = 0
            self.max_input_samples = 0
            self.output_samples = 0
            self.first_overflow_call: int | None = None

        def resample_chunk(
            self, samples: np.ndarray, *, last: bool
        ) -> np.ndarray:
            self.calls += 1
            self.max_input_samples = max(
                self.max_input_samples, len(samples)
            )
            output = self.inner.resample_chunk(samples, last=last)
            self.output_samples += len(output)
            if (
                self.first_overflow_call is None
                and self.output_samples >= 16_000 * 121 // 10
            ):
                self.first_overflow_call = self.calls
            return output

    def resampler_factory(*args: Any, **kwargs: Any) -> CountingSoxr:
        stream = CountingSoxr(*args, **kwargs)
        streams.append(stream)
        return stream

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.MALE): model_path},
        voice_factory=lambda _path, **_kwargs: LazyLongVoice(),
    )
    caplog.set_level(logging.DEBUG)
    frames = PiperTts(
        registry, resampler_factory=resampler_factory
    ).synthesize_frames(
        speech_marker,
        target_language=Language.EN,
        voice_profile=profile(Language.EN, VoiceGender.MALE),
        mode=TranslationMode.BALANCED,
        output_sample_rate_hz=16_000,
        output_channels=1,
        frame_duration_ms=100,
    )

    emitted = 0
    with pytest.raises(TtsOutputLimit, match="limit") as raised:
        while True:
            next(frames)
            emitted += 1
    assert emitted == 120
    assert streams[0].calls == len(pulls)
    assert streams[0].max_input_samples <= 2_205
    assert streams[0].first_overflow_call is not None
    assert streams[0].calls == streams[0].first_overflow_call
    work_at_overflow = (len(pulls), streams[0].calls)
    with pytest.raises(StopIteration):
        next(frames)
    assert (len(pulls), streams[0].calls) == work_at_overflow
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert speech_marker not in rendered
    assert speech_marker not in caplog.text


@pytest.mark.parametrize(
    ("sample_width", "sample_channels"),
    [(4, 1), (2, 2)],
)
def test_tts_rejects_non_mono_s16le_piper_chunk(
    tmp_path: Path,
    sample_width: int,
    sample_channels: int,
) -> None:
    model_path = create_voice(tmp_path, "en-male")
    voice = FakeVoice(
        [
            Chunk(
                np.zeros(320, dtype=np.int16),
                sample_width=sample_width,
                sample_channels=sample_channels,
            )
        ]
    )
    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.MALE): model_path},
        voice_factory=lambda _path, **_kwargs: voice,
    )

    with pytest.raises(TtsUnavailable, match="unavailable"):
        list(
            PiperTts(registry).synthesize_frames(
                "text",
                target_language=Language.EN,
                voice_profile=profile(Language.EN, VoiceGender.MALE),
                mode=TranslationMode.BALANCED,
                output_sample_rate_hz=16_000,
                output_channels=1,
                frame_duration_ms=20,
            )
        )


def test_tts_rejects_language_profile_mismatch_before_load(
    tmp_path: Path,
) -> None:
    model_path = create_voice(tmp_path, "ru-male")
    registry = PiperVoiceRegistry(
        {(Language.RU, VoiceGender.MALE): model_path},
        voice_factory=lambda _path, **_kwargs: pytest.fail("must not load"),
    )

    with pytest.raises(TtsUnsupported, match="language"):
        list(
            PiperTts(registry).synthesize_frames(
                "text",
                target_language=Language.EN,
                voice_profile=profile(Language.RU, VoiceGender.MALE),
                mode=TranslationMode.BALANCED,
                output_sample_rate_hz=16_000,
                output_channels=1,
                frame_duration_ms=20,
            )
        )


@pytest.mark.parametrize("failure_at", ["load", "synthesis"])
def test_tts_sanitizes_native_failure_traceback(
    tmp_path: Path,
    failure_at: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = f"private {failure_at} marker"
    model_path = create_voice(tmp_path, "en-female")

    def factory(_path: str, **_kwargs: Any) -> FakeVoice:
        if failure_at == "load":
            raise RuntimeError(marker)
        return FakeVoice([], failure=RuntimeError(marker))

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path},
        voice_factory=factory,
    )
    caplog.set_level(logging.DEBUG)
    with pytest.raises(TtsUnavailable, match="unavailable") as raised:
        list(
            PiperTts(registry).synthesize_frames(
                "private spoken text",
                target_language=Language.EN,
                voice_profile=profile(Language.EN, VoiceGender.FEMALE),
                mode=TranslationMode.BALANCED,
                output_sample_rate_hz=16_000,
                output_channels=1,
                frame_duration_ms=20,
            )
        )
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert marker not in rendered
    assert "private spoken text" not in rendered
    assert marker not in caplog.text
    assert "private spoken text" not in caplog.text


@pytest.mark.parametrize(
    "failure_at",
    ["construct", "process", "flush"],
)
def test_tts_sanitizes_resampler_failure_traceback(
    tmp_path: Path,
    failure_at: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = f"private resampler {failure_at} marker"
    model_path = create_voice(tmp_path, "en-female")
    voice = FakeVoice(
        [Chunk(np.ones(100, dtype=np.int16), sample_rate=22_050)]
    )

    class FailingResampler:
        def resample_chunk(
            self, samples: np.ndarray, *, last: bool
        ) -> np.ndarray:
            if failure_at == "process" and len(samples):
                raise RuntimeError(marker)
            if failure_at == "flush" and last:
                raise RuntimeError(marker)
            return samples

    def resampler_factory(*_args: Any, **_kwargs: Any) -> FailingResampler:
        if failure_at == "construct":
            raise RuntimeError(marker)
        return FailingResampler()

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path},
        voice_factory=lambda _path, **_kwargs: voice,
    )
    caplog.set_level(logging.DEBUG)
    with pytest.raises(TtsUnavailable, match="unavailable") as raised:
        list(
            PiperTts(
                registry, resampler_factory=resampler_factory
            ).synthesize_frames(
                "private spoken text",
                target_language=Language.EN,
                voice_profile=profile(Language.EN, VoiceGender.FEMALE),
                mode=TranslationMode.BALANCED,
                output_sample_rate_hz=16_000,
                output_channels=1,
                frame_duration_ms=20,
            )
        )
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert marker not in rendered
    assert "private spoken text" not in rendered
    assert marker not in caplog.text
    assert "private spoken text" not in caplog.text


def test_tts_suppresses_piper_debug_content_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "private spoken log marker"
    model_path = create_voice(tmp_path, "en-female")

    class LoggingVoice:
        def synthesize(self, text: str) -> object:
            def generate() -> object:
                logging.getLogger("piper.voice").debug("text=%s", text)
                yield Chunk(np.zeros(320, dtype=np.int16))

            return generate()

    caplog.set_level(logging.DEBUG)
    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path},
        voice_factory=lambda _path, **_kwargs: LoggingVoice(),
    )
    list(
        PiperTts(registry).synthesize_frames(
            marker,
            target_language=Language.EN,
            voice_profile=profile(Language.EN, VoiceGender.FEMALE),
            mode=TranslationMode.BALANCED,
            output_sample_rate_hz=16_000,
            output_channels=1,
            frame_duration_ms=20,
        )
    )

    assert marker not in caplog.text


def test_tts_sanitizes_missing_soxr_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_import = builtins.__import__
    speech_marker = "private missing soxr speech marker"
    model_path = create_voice(tmp_path, "en-female")
    trapped = False
    voice = FakeVoice(
        [Chunk(np.zeros(2_205, dtype=np.int16), sample_rate=22_050)]
    )

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> object:
        nonlocal trapped
        if name == "soxr":
            trapped = True
            raise ModuleNotFoundError("private soxr dependency marker")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path},
        voice_factory=lambda _path, **_kwargs: voice,
    )
    caplog.set_level(logging.DEBUG)
    with pytest.raises(TtsUnavailable, match="unavailable") as raised:
        list(
            PiperTts(registry).synthesize_frames(
                speech_marker,
                target_language=Language.EN,
                voice_profile=profile(Language.EN, VoiceGender.FEMALE),
                mode=TranslationMode.BALANCED,
                output_sample_rate_hz=16_000,
                output_channels=1,
                frame_duration_ms=20,
            )
        )
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private soxr dependency marker" not in rendered
    assert "private soxr dependency marker" not in caplog.text
    assert speech_marker not in rendered
    assert speech_marker not in caplog.text
    assert trapped


def test_tts_sanitizes_missing_piper_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_import = builtins.__import__
    speech_marker = "private missing piper speech marker"
    model_path = create_voice(tmp_path, "en-female")
    trapped = False

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> object:
        nonlocal trapped
        if name == "piper" or name.startswith("piper."):
            trapped = True
            raise ModuleNotFoundError("private piper dependency marker")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.FEMALE): model_path}
    )
    caplog.set_level(logging.DEBUG)
    with pytest.raises(TtsUnavailable, match="unavailable") as raised:
        list(
            PiperTts(registry).synthesize_frames(
                speech_marker,
                target_language=Language.EN,
                voice_profile=profile(Language.EN, VoiceGender.FEMALE),
                mode=TranslationMode.BALANCED,
                output_sample_rate_hz=16_000,
                output_channels=1,
                frame_duration_ms=20,
            )
        )
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private piper dependency marker" not in rendered
    assert "private piper dependency marker" not in caplog.text
    assert speech_marker not in rendered
    assert speech_marker not in caplog.text
    assert trapped
