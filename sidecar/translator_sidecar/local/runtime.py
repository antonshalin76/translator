"""Production construction of the offline local provider runtime."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Callable

from translator_sidecar.local.asr import (
    AsrModelManager,
    AsrUnavailable,
)
from translator_sidecar.local.cuda_runtime import configure_cuda_runtime
from translator_sidecar.local.inference_scheduler import InferenceScheduler
from translator_sidecar.local.local_provider import LocalProvider
from translator_sidecar.local.model_manifest import (
    ModelManifest,
    load_manifest,
)
from translator_sidecar.local.mt import (
    LocalTranslationError,
    NllbTranslator,
)
from translator_sidecar.local.tts import (
    PiperTts,
    PiperVoiceRegistry,
    TtsUnavailable,
)
from translator_sidecar.provider_contract import (
    ComputeDevice,
    Language,
    ModelState,
    TranslationMode,
    VoiceGender,
)


_ASR_MODELS = {
    "faster-whisper-small": "small",
    "faster-whisper-large-v3": "large-v3",
}
_DEFAULT_ASR_MODEL_ID = "faster-whisper-small"
_MT_MODEL_ID = "nllb-200-distilled-600m-ct2-int8"
_TTS_MODEL_ID = "piper-medium"
_MT_SMOKE_CASES = (
    ("Проверка перевода.", Language.RU, Language.EN),
    ("Translation check.", Language.EN, Language.RU),
)
_VOICE_MODELS = {
    (Language.RU, VoiceGender.MALE): "piper-ru-dmitri-medium",
    (Language.EN, VoiceGender.MALE): "piper-en-ryan-medium",
    (Language.RU, VoiceGender.FEMALE): "piper-ru-irina-medium",
    (Language.EN, VoiceGender.FEMALE): "piper-en-hfc-female-medium",
}


class _UnavailableAsr:
    actual_device = "cpu"
    degraded = True
    unavailable = True
    resident_model_id = None

    @staticmethod
    def transcribe(*args: Any, **kwargs: Any) -> str:
        raise AsrUnavailable("local ASR is unavailable")


class _UnavailableTranslator:
    unavailable = True
    model_state = ModelState.FAILED

    @staticmethod
    def translate(*args: Any, **kwargs: Any) -> str:
        raise LocalTranslationError("local MT is unavailable")

    @staticmethod
    def count_tokens(text: str) -> int:
        del text
        raise LocalTranslationError("local MT is unavailable")


class _UnavailableTts:
    unavailable = True

    @staticmethod
    def model_state(*args: Any, **kwargs: Any) -> ModelState:
        return ModelState.FAILED

    @staticmethod
    def synthesize_frames(*args: Any, **kwargs: Any):
        raise TtsUnavailable("local TTS is unavailable")


def _cuda_available() -> bool:
    try:
        configure_cuda_runtime()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "manifest.json"


def _model_directory(manifest: ModelManifest, model_id: str) -> Path:
    model = manifest.models[model_id]
    if not model.files:
        raise ValueError("model has no declared runtime files")
    for model_file in model.files:
        manifest.resolve_runtime_file(model_id, model_file.path)
    return model.cache_path


def _primary_model_file(manifest: ModelManifest, model_id: str) -> Path:
    model = manifest.models[model_id]
    primary = next(
        (
            model_file
            for model_file in model.files
            if not model_file.path.endswith(".json")
        ),
        None,
    )
    if primary is None:
        raise ValueError("model has no primary runtime file")
    return model.cache_path / primary.path


def _release_translator(translator: NllbTranslator) -> None:
    backend = getattr(translator, "_translator", None)
    unload_model = getattr(backend, "unload_model", None)
    if callable(unload_model):
        try:
            unload_model()
        except Exception:
            pass


def _load_verified_translator(
    model_path: Path,
    *,
    device: str,
) -> NllbTranslator:
    translator = NllbTranslator.load(model_path, device=device)
    smoke_failed = False
    for text, source_language, target_language in _MT_SMOKE_CASES:
        try:
            translated = translator.translate(
                text,
                source_language=source_language,
                target_language=target_language,
                mode=TranslationMode.QUALITY_FIRST,
            )
            smoke_failed = smoke_failed or not translated.strip()
        except Exception:
            smoke_failed = True
    if smoke_failed:
        _release_translator(translator)
        del translator
        gc.collect()
        raise LocalTranslationError("local MT bootstrap smoke failed")
    return translator


def _unavailable_provider(
    *,
    now_ns: Callable[[], int],
    asr_model_id: str,
) -> LocalProvider:
    return LocalProvider(
        asr=_UnavailableAsr(),
        translator=_UnavailableTranslator(),
        tts=_UnavailableTts(),
        scheduler=InferenceScheduler(),
        now_ns=now_ns,
        asr_model_id=asr_model_id,
        mt_model_id=_MT_MODEL_ID,
        tts_model_id=_TTS_MODEL_ID,
        mt_device=ComputeDevice.CPU,
    )


def build_unavailable_local_provider(
    *,
    now_ns: Callable[[], int],
) -> LocalProvider:
    selected_asr_id = os.environ.get(
        "TRANSLATOR_ASR_MODEL_ID",
        _DEFAULT_ASR_MODEL_ID,
    )
    return _unavailable_provider(
        now_ns=now_ns,
        asr_model_id=selected_asr_id,
    )


def build_local_provider(
    *,
    now_ns: Callable[[], int],
    manifest_path: Path | None = None,
) -> LocalProvider:
    """Build the selected offline chain or a reachable unavailable provider."""

    selected_asr_id = os.environ.get(
        "TRANSLATOR_ASR_MODEL_ID",
        _DEFAULT_ASR_MODEL_ID,
    )
    if os.environ.get("TRANSLATOR_LOCAL_RUNTIME_MODE") == "unavailable":
        return build_unavailable_local_provider(now_ns=now_ns)
    if selected_asr_id not in _ASR_MODELS:
        return _unavailable_provider(
            now_ns=now_ns,
            asr_model_id=selected_asr_id,
        )

    path = manifest_path or _default_manifest_path()
    try:
        manifest = load_manifest(path)
        required_model_ids = (
            selected_asr_id,
            *(
                (_DEFAULT_ASR_MODEL_ID,)
                if selected_asr_id != _DEFAULT_ASR_MODEL_ID
                else ()
            ),
            _MT_MODEL_ID,
            *_VOICE_MODELS.values(),
        )
        model_directories = {
            model_id: _model_directory(manifest, model_id)
            for model_id in required_model_ids
        }
        voice_paths = {
            profile: _primary_model_file(manifest, model_id)
            for profile, model_id in _VOICE_MODELS.items()
        }
    except Exception:
        return _unavailable_provider(
            now_ns=now_ns,
            asr_model_id=selected_asr_id,
        )

    asr_device = "cuda" if _cuda_available() else "cpu"
    mt_device = asr_device
    try:
        translator = _load_verified_translator(
            model_directories[_MT_MODEL_ID],
            device=mt_device,
        )
    except Exception:
        if mt_device != "cuda":
            return _unavailable_provider(
                now_ns=now_ns,
                asr_model_id=selected_asr_id,
            )
        mt_device = "cpu"
        try:
            translator = _load_verified_translator(
                model_directories[_MT_MODEL_ID],
                device=mt_device,
            )
        except Exception:
            return _unavailable_provider(
                now_ns=now_ns,
                asr_model_id=selected_asr_id,
            )

    try:
        selected_key = _ASR_MODELS[selected_asr_id]
        asr_paths = {
            selected_key: model_directories[selected_asr_id],
        }
        if selected_key == "large-v3":
            asr_paths["small"] = model_directories[_DEFAULT_ASR_MODEL_ID]
        asr = AsrModelManager(
            selected_id=selected_key,
            model_paths=asr_paths,
            device=asr_device,
        )
        prepare_asr = getattr(asr, "prepare", None)
        if prepare_asr is not None:
            prepare_asr()
        registry = PiperVoiceRegistry(voice_paths)
        prepare_tts = getattr(registry, "prepare", None)
        if prepare_tts is not None:
            prepare_tts()
        tts = PiperTts(registry)
        return LocalProvider(
            asr=asr,
            translator=translator,
            tts=tts,
            scheduler=InferenceScheduler(),
            now_ns=now_ns,
            asr_model_id=selected_asr_id,
            mt_model_id=_MT_MODEL_ID,
            tts_model_id=_TTS_MODEL_ID,
            mt_device=ComputeDevice(mt_device),
        )
    except Exception:
        return _unavailable_provider(
            now_ns=now_ns,
            asr_model_id=selected_asr_id,
        )
