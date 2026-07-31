"""Offline Piper TTS with bounded streaming resampling and PCM framing."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from translator_sidecar.provider_contract import (
    Language,
    ModelState,
    TranslationMode,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


_ALLOWED_SAMPLE_RATES = {16_000, 24_000, 48_000}
_ALLOWED_FRAME_DURATIONS_MS = {20, 40, 60, 80, 100}
_MAX_OUTPUT_MS = 12_000
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


class TtsUnavailable(RuntimeError):
    """The local TTS request cannot run."""


class TtsUnsupported(RuntimeError):
    """The requested local TTS profile or format is unsupported."""


class TtsOutputLimit(RuntimeError):
    """The synthesized audio exceeded the bounded utterance output."""


def _suppress_piper_content_logs() -> None:
    piper_logger = logging.getLogger("piper.voice")
    if piper_logger.level < logging.INFO:
        piper_logger.setLevel(logging.INFO)


class PiperVoiceRegistry:
    """Loads only approved local Piper voices and caches each preset once."""

    def __init__(
        self,
        voice_paths: dict[tuple[Language, VoiceGender], Path],
        *,
        voice_factory: Callable[..., Any] | None = None,
        load_lock: Any | None = None,
    ) -> None:
        self._voice_paths = dict(voice_paths)
        self._voice_factory = voice_factory
        self._load_lock = load_lock or Lock()
        self._voices: dict[tuple[Language, VoiceGender], Any] = {}

    def prepare(self) -> None:
        """Load every approved local voice without synthesizing speech."""
        for language, gender in self._voice_paths:
            self.get(
                VoiceProfile(
                    language=language,
                    gender=gender,
                    engine=VoiceEngine.PIPER,
                )
            )

    def get(self, voice_profile: VoiceProfile) -> Any:
        if voice_profile.engine is not VoiceEngine.PIPER:
            raise TtsUnsupported("only Piper voice profiles are supported")
        if (
            voice_profile.model_path is not None
            or voice_profile.provider_voice_id is not None
        ):
            raise TtsUnsupported("voice profile override is unsupported")
        key = (voice_profile.language, voice_profile.gender)
        with self._load_lock:
            loaded = self._voices.get(key)
            if loaded is not None:
                return loaded
            path = self._voice_paths.get(key)
            if path is None:
                raise TtsUnavailable("requested Piper voice is unavailable")
            config_path = path.with_suffix(".onnx.json")
            if (
                not path.is_absolute()
                or not path.is_file()
                or not config_path.is_file()
            ):
                raise TtsUnavailable("requested Piper voice is unavailable")
            os.environ.update(_OFFLINE_ENV)
            _suppress_piper_content_logs()
            try:
                factory = self._voice_factory
                if factory is None:
                    from piper import PiperVoice

                    factory = PiperVoice.load
                voice = factory(
                    str(path),
                    config_path=str(config_path),
                    use_cuda=False,
                )
            except Exception:
                raise TtsUnavailable(
                    "requested Piper voice is unavailable"
                ) from None
            self._voices[key] = voice
            return voice

    def model_state(
        self,
        voice_profile: VoiceProfile,
    ) -> ModelState:
        if (
            voice_profile.engine is not VoiceEngine.PIPER
            or voice_profile.model_path is not None
            or voice_profile.provider_voice_id is not None
        ):
            return ModelState.FAILED
        key = (voice_profile.language, voice_profile.gender)
        with self._load_lock:
            if key in self._voices:
                return ModelState.READY
            path = self._voice_paths.get(key)
            if path is None:
                return ModelState.FAILED
            config_path = path.with_suffix(".onnx.json")
            if (
                not path.is_absolute()
                or not path.is_file()
                or not config_path.is_file()
            ):
                return ModelState.FAILED
            return ModelState.NOT_LOADED


class PiperTts:
    """Converts lazy Piper chunks into exact negotiated PCM frames."""

    def __init__(
        self,
        registry: PiperVoiceRegistry,
        *,
        resampler_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._registry = registry
        self._resampler_factory = resampler_factory

    def model_state(
        self,
        voice_profile: VoiceProfile,
    ) -> ModelState:
        return self._registry.model_state(voice_profile)

    def synthesize_frames(
        self,
        text: str,
        *,
        target_language: Language,
        voice_profile: VoiceProfile,
        mode: TranslationMode,
        output_sample_rate_hz: int,
        output_channels: int,
        frame_duration_ms: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[bytes]:
        del mode
        is_cancelled = cancelled or (lambda: False)
        if is_cancelled():
            return
        if voice_profile.language is not target_language:
            raise TtsUnsupported("voice profile language does not match target")
        if output_sample_rate_hz not in _ALLOWED_SAMPLE_RATES:
            raise TtsUnsupported("TTS output sample rate is unsupported")
        if output_channels not in {1, 2}:
            raise TtsUnsupported("TTS output channel count is unsupported")
        if frame_duration_ms not in _ALLOWED_FRAME_DURATIONS_MS:
            raise TtsUnsupported("TTS frame duration is unsupported")
        normalized = text.strip()
        if not normalized:
            raise TtsUnavailable("local TTS input is unavailable")

        try:
            voice = self._registry.get(voice_profile)
            if is_cancelled():
                return
            _suppress_piper_content_logs()
            chunks = iter(voice.synthesize(normalized))
            yield from self._render_frames(
                chunks,
                output_sample_rate_hz=output_sample_rate_hz,
                output_channels=output_channels,
                frame_duration_ms=frame_duration_ms,
                is_cancelled=is_cancelled,
            )
        except (TtsOutputLimit, TtsUnavailable, TtsUnsupported):
            raise
        except Exception:
            raise TtsUnavailable("local TTS is unavailable") from None

    def _render_frames(
        self,
        chunks: Iterator[Any],
        *,
        output_sample_rate_hz: int,
        output_channels: int,
        frame_duration_ms: int,
        is_cancelled: Callable[[], bool],
    ) -> Iterator[bytes]:
        frame_samples = (
            output_sample_rate_hz * frame_duration_ms // 1000
        )
        buffered = np.empty(0, dtype=np.int16)
        input_sample_rate: int | None = None
        resampler: Any | None = None
        emitted_ms = 0

        while not is_cancelled():
            try:
                chunk = next(chunks)
            except StopIteration:
                break
            if (
                chunk.sample_width != 2
                or chunk.sample_channels != 1
                or chunk.sample_rate <= 0
            ):
                raise TtsUnavailable("local TTS chunk is unavailable")
            if input_sample_rate is None:
                input_sample_rate = chunk.sample_rate
                if input_sample_rate != output_sample_rate_hz:
                    resampler = self._new_resampler(
                        input_sample_rate,
                        output_sample_rate_hz,
                    )
            elif chunk.sample_rate != input_sample_rate:
                raise TtsUnavailable("local TTS chunk is unavailable")

            samples = np.frombuffer(
                chunk.audio_int16_bytes, dtype="<i2"
            )
            input_slice_samples = max(1, input_sample_rate // 10)
            for offset in range(0, len(samples), input_slice_samples):
                if is_cancelled():
                    return
                source = samples[offset : offset + input_slice_samples]
                output = (
                    resampler.resample_chunk(source, last=False)
                    if resampler is not None
                    else source
                )
                buffered = np.concatenate((buffered, output))
                while len(buffered) >= frame_samples:
                    if is_cancelled():
                        return
                    frame = buffered[:frame_samples]
                    buffered = buffered[frame_samples:]
                    emitted_ms = self._check_output_limit(
                        emitted_ms, frame_duration_ms
                    )
                    yield self._encode_frame(frame, output_channels)

        if is_cancelled():
            return
        if resampler is not None:
            flushed = resampler.resample_chunk(
                np.empty(0, dtype=np.int16),
                last=True,
            )
            buffered = np.concatenate((buffered, flushed))
            while len(buffered) >= frame_samples:
                if is_cancelled():
                    return
                frame = buffered[:frame_samples]
                buffered = buffered[frame_samples:]
                emitted_ms = self._check_output_limit(
                    emitted_ms, frame_duration_ms
                )
                yield self._encode_frame(frame, output_channels)
        if len(buffered):
            if is_cancelled():
                return
            frame = np.zeros(frame_samples, dtype=np.int16)
            frame[: len(buffered)] = buffered
            self._check_output_limit(emitted_ms, frame_duration_ms)
            yield self._encode_frame(frame, output_channels)

    def _new_resampler(
        self,
        input_sample_rate: int,
        output_sample_rate: int,
    ) -> Any:
        factory = self._resampler_factory
        if factory is None:
            try:
                import soxr

                factory = soxr.ResampleStream
            except Exception:
                raise TtsUnavailable("local TTS is unavailable") from None
        try:
            return factory(
                input_sample_rate,
                output_sample_rate,
                1,
                dtype="int16",
            )
        except Exception:
            raise TtsUnavailable("local TTS is unavailable") from None

    @staticmethod
    def _check_output_limit(emitted_ms: int, frame_ms: int) -> int:
        next_emitted_ms = emitted_ms + frame_ms
        if next_emitted_ms > _MAX_OUTPUT_MS:
            raise TtsOutputLimit("local TTS output limit was reached")
        return next_emitted_ms

    @staticmethod
    def _encode_frame(samples: np.ndarray, channels: int) -> bytes:
        if channels == 2:
            samples = np.repeat(samples[:, np.newaxis], 2, axis=1)
        return samples.astype("<i2", copy=False).tobytes()
