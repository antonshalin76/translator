"""Offline faster-whisper adapter with single-model residency."""

from __future__ import annotations

import gc
import os
import weakref
from collections.abc import Callable
from threading import Lock
from typing import Any

import numpy as np

from translator_sidecar.provider_contract import Language, TranslationMode

from .cuda_runtime import configure_cuda_runtime
from .model_lease import VerifiedModelLease, VerifiedModelSource

_BEAM_SIZE = {
    TranslationMode.QUALITY_FIRST: 5,
    TranslationMode.BALANCED: 3,
    TranslationMode.STREAMING_FIRST: 1,
}
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
_OOM_MARKERS = ("out of memory", "cuda_error_out_of_memory")
_CUDA_RUNTIME_MARKERS = (
    "libcublas",
    "libcudnn",
    "cuda driver",
    "cuda runtime",
    "cuda not found",
)
_ASR_UNAVAILABLE_MESSAGE = "local ASR is unavailable"


class AsrUnavailable(RuntimeError):
    """The local ASR request cannot run."""


class AsrUnsupported(RuntimeError):
    """The requested ASR runtime mode is intentionally unsupported."""


def _default_cuda_available() -> bool:
    try:
        configure_cuda_runtime()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _default_release_cuda() -> None:
    return None


def _failure_kind(error: Exception) -> str:
    message = str(error).casefold()
    if any(marker in message for marker in _OOM_MARKERS):
        return "oom"
    if any(marker in message for marker in _CUDA_RUNTIME_MARKERS):
        return "cuda_runtime"
    return "other"


class AsrModelManager:
    """Own one faster-whisper model and serialize all native inference."""

    def __init__(
        self,
        *,
        selected_id: str,
        model_paths: dict[str, VerifiedModelSource],
        device: str,
        model_factory: Callable[..., Any] | None = None,
        release_cuda: Callable[[], None] | None = None,
        cuda_available: Callable[[], bool] | None = None,
        admission_lock: Any | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise AsrUnsupported("ASR device is not supported")
        self._model_paths = dict(model_paths)
        self._model_factory = model_factory
        self._release_cuda = release_cuda or _default_release_cuda
        self._admission_lock = admission_lock or Lock()
        self._model: Any | None = None
        self._lease: VerifiedModelLease | None = None
        self._retired_model_ref: Callable[[], Any | None] | None = None
        self._resident_model_id: str | None = None
        self._residency_generation = 0
        self._active_call_count = 0
        self._unavailable = False

        self._actual_device = device
        self._selected_id = selected_id
        self._degraded = device == "cpu"
        check_cuda = cuda_available or _default_cuda_available
        if device == "cuda" and not check_cuda():
            self._actual_device = "cpu"
            self._selected_id = "small"
            self._degraded = True

    @property
    def resident_model_id(self) -> str | None:
        return self._resident_model_id

    @property
    def residency_generation(self) -> int:
        return self._residency_generation

    @property
    def active_call_count(self) -> int:
        return self._active_call_count

    @property
    def actual_device(self) -> str:
        return self._actual_device

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def unavailable(self) -> bool:
        return self._unavailable

    def release(self) -> bool:
        with self._admission_lock:
            return self._drop_resident(invalidate_without_replacement=False)

    def close(self) -> None:
        with self._admission_lock:
            self._unavailable = True
            if not self._drop_resident(invalidate_without_replacement=False):
                raise AsrUnavailable("local ASR cleanup is incomplete")

    def prepare(self) -> None:
        with self._admission_lock:
            if self._unavailable:
                raise AsrUnavailable(_ASR_UNAVAILABLE_MESSAGE)
            if self._actual_device == "cpu" and self._selected_id != "small":
                raise AsrUnsupported("large ASR is unsupported on the CPU runtime")
            while True:
                load_failure = self._ensure_loaded()
                if load_failure is None:
                    return
                if self._handle_cuda_failure(load_failure):
                    continue
                self._unavailable = True
                raise AsrUnavailable(_ASR_UNAVAILABLE_MESSAGE)

    def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> str:
        audio = self._decode_pcm(pcm_s16le)
        with self._admission_lock:
            if self._unavailable:
                raise AsrUnavailable(_ASR_UNAVAILABLE_MESSAGE)
            if self._actual_device == "cpu" and self._selected_id != "small":
                raise AsrUnsupported("large ASR is unsupported on the CPU runtime")
            return self._transcribe_locked(audio, language=language, mode=mode)

    @staticmethod
    def _decode_pcm(pcm_s16le: bytes) -> np.ndarray:
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise AsrUnavailable("ASR PCM input is invalid")
        return np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / np.float32(
            32768
        )

    def _transcribe_locked(
        self,
        audio: np.ndarray,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> str:
        while True:
            load_failure = self._ensure_loaded()
            if load_failure is not None:
                if self._handle_cuda_failure(load_failure):
                    continue
                self._unavailable = True
                raise AsrUnavailable(_ASR_UNAVAILABLE_MESSAGE)

            text, inference_failure = self._run_inference(
                audio,
                language=language,
                mode=mode,
            )
            if inference_failure is None:
                return text or ""
            if self._handle_cuda_failure(inference_failure):
                continue
            raise AsrUnavailable(_ASR_UNAVAILABLE_MESSAGE)

    def _ensure_loaded(self) -> str | None:
        if self._model is not None:
            return None
        source = self._model_paths.get(self._selected_id)
        if not isinstance(source, VerifiedModelSource):
            return "other"
        os.environ.update(_OFFLINE_ENV)
        compute_type = "float16" if self._actual_device == "cuda" else "int8"
        self._residency_generation += 1
        lease = None
        try:
            factory = self._model_factory
            if factory is None:
                from faster_whisper import WhisperModel

                factory = WhisperModel
            lease = source.acquire()
            files: dict[str, Any] = lease.files()
            files["tokenizer.json"] = lease.read_bytes("tokenizer.json")
            if not files["tokenizer.json"]:
                raise AsrUnavailable("local ASR tokenizer is empty")
            # Explicit library defaults: never probe an ambient cache config.
            files["preprocessor_config.json"] = (
                lease.read_bytes("preprocessor_config.json")
                if "preprocessor_config.json" in lease.names
                else b"{}"
            )
            model = factory(
                lease.identifier,
                device=self._actual_device,
                compute_type=compute_type,
                local_files_only=True,
                num_workers=1,
                files=files,
            )
            weakref.finalize(getattr(model, "model", model), lease.close)
        except Exception as error:
            if lease is not None:
                lease.close()
            return _failure_kind(error)
        except BaseException:
            if lease is not None:
                lease.close()
            raise
        self._model = model
        self._lease = lease
        self._resident_model_id = self._selected_id
        return None

    def _run_inference(
        self,
        audio: np.ndarray,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> tuple[str | None, str | None]:
        model = self._model
        self._active_call_count = 1
        try:
            segments, _info = model.transcribe(
                audio,
                language=language.value,
                beam_size=_BEAM_SIZE[mode],
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = "".join(segment.text for segment in segments).strip()
        except Exception as error:
            return None, _failure_kind(error)
        finally:
            self._active_call_count = 0
        return text, None

    def _handle_cuda_failure(self, failure: str) -> bool:
        if (
            failure == "oom"
            and self._actual_device == "cuda"
            and self._selected_id == "large-v3"
        ):
            if not self._drop_resident(invalidate_without_replacement=False):
                return False
            self._selected_id = "small"
            self._degraded = True
            return True
        if failure == "cuda_runtime" and self._actual_device == "cuda":
            if not self._drop_resident(invalidate_without_replacement=False):
                return False
            self._actual_device = "cpu"
            self._selected_id = "small"
            self._degraded = True
            return True
        if failure == "oom":
            self._drop_resident(invalidate_without_replacement=True)
            self._unavailable = True
        return False

    def _drop_resident(self, *, invalidate_without_replacement: bool) -> bool:
        if self._retired_model_ref is not None:
            if self._retired_model_ref() is not None:
                return False
            self._retired_model_ref = None
        model = self._model
        had_resident = model is not None
        model_ref: Callable[[], Any | None] | None = None
        cleanup_ok = True
        if model is not None:
            native_model = getattr(model, "model", model)
            try:
                model_ref = weakref.ref(native_model)
            except TypeError:
                model_ref = None
            unload_model = None
            try:
                unload_model = getattr(native_model, "unload_model", None)
                if unload_model is not None:
                    unload_model()
            except Exception:
                self._unavailable = True
                return False
            del unload_model
            del native_model
        self._model = None
        self._resident_model_id = None
        del model
        gc.collect()
        try:
            self._release_cuda()
        except Exception:
            cleanup_ok = False
        gc.collect()
        if invalidate_without_replacement and had_resident:
            self._residency_generation += 1
        if model_ref is not None and model_ref() is not None:
            cleanup_ok = False
            self._retired_model_ref = model_ref
        elif self._lease is not None:
            self._lease.close()
            self._lease = None
        if not cleanup_ok:
            self._unavailable = True
        return cleanup_ok
