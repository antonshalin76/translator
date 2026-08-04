"""ASR-only quality probes for podcast and fixture diagnostics."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable
import wave

from jiwer import wer
import numpy as np

from translator_sidecar.benchmark.model_matrix import (
    ModelCandidate,
    candidate_by_id,
    default_executable_asr_candidate_ids,
)
from translator_sidecar.local.asr import AsrModelManager
from translator_sidecar.local.cuda_runtime import configure_cuda_runtime
from translator_sidecar.local.model_manifest import ModelManifest, load_manifest
from translator_sidecar.provider_contract import Language, TranslationMode


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT = _ROOT / "output" / "asr-quality-debug.json"
_DEFAULT_MANIFEST = _ROOT / "models" / "manifest.json"
_SAMPLE_RATE_HZ = 16_000
_LOCAL_PROVIDER_MODEL_KEYS = {
    "faster-whisper-small": "small",
    "faster-whisper-large-v3": "large-v3",
}
_QWEN_LANGUAGE = {
    Language.RU: "Russian",
    Language.EN: "English",
}
_BEAM_SIZE = {
    TranslationMode.QUALITY_FIRST: 5,
    TranslationMode.BALANCED: 3,
    TranslationMode.STREAMING_FIRST: 1,
}
_CUDA_RUNTIME_MARKERS = (
    "libcublas",
    "libcudnn",
    "cuda driver",
    "cuda runtime",
    "cuda not found",
)


class AsrQualityError(RuntimeError):
    """The ASR quality probe could not complete."""


class AsrQualityUnavailable(AsrQualityError):
    """The selected candidate cannot run in the current environment."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _cuda_available() -> bool:
    try:
        configure_cuda_runtime()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _read_pcm(path: Path) -> bytes:
    data = path.read_bytes()
    if not data or len(data) % 2:
        raise AsrQualityError("PCM input must be non-empty s16le mono audio")
    return data


def _write_wav(path: Path, pcm_s16le: bytes, *, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(pcm_s16le)


def _decode_pcm(pcm_s16le: bytes) -> np.ndarray:
    if not pcm_s16le or len(pcm_s16le) % 2:
        raise AsrQualityError("PCM input must be non-empty s16le mono audio")
    return np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / np.float32(32768)


def _safe_wer(reference: str | None, hypothesis: str) -> float | None:
    if reference is None:
        return None
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    if not reference or not hypothesis:
        return None
    try:
        return float(wer(reference, hypothesis))
    except Exception:
        return None


def _safe_error_summary(error: Exception) -> str:
    message = str(error).strip().splitlines()
    first_line = message[0] if message else type(error).__name__
    return f"{type(error).__name__}: {first_line}"


def _is_cuda_runtime_error(error: Exception) -> bool:
    message = str(error).casefold()
    return any(marker in message for marker in _CUDA_RUNTIME_MARKERS)


def _manifest_model_directory(manifest: ModelManifest, model_id: str) -> Path:
    model = manifest.models[model_id]
    for model_file in model.files:
        manifest.resolve_runtime_file(model_id, model_file.path)
    return model.cache_path


class FasterWhisperAsrProbe:
    def __init__(
        self,
        *,
        model_id: str,
        manifest_path: Path = _DEFAULT_MANIFEST,
        device: str | None = None,
    ) -> None:
        selected_key = _LOCAL_PROVIDER_MODEL_KEYS.get(model_id)
        if selected_key is None:
            raise AsrQualityUnavailable("candidate is not a local faster-whisper model")
        manifest = load_manifest(manifest_path)
        model_paths = {
            selected_key: _manifest_model_directory(manifest, model_id),
        }
        if selected_key == "large-v3":
            model_paths["small"] = _manifest_model_directory(
                manifest,
                "faster-whisper-small",
            )
        self._manager = AsrModelManager(
            selected_id=selected_key,
            model_paths=model_paths,
            device=device or ("cuda" if _cuda_available() else "cpu"),
        )

    def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> str:
        return self._manager.transcribe(
            pcm_s16le,
            language=language,
            mode=mode,
        )

    def release(self) -> None:
        self._manager.release()


class FasterWhisperCt2AsrProbe:
    def __init__(
        self,
        *,
        repository: str,
        device: str | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        factory = model_factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel
            except Exception as error:
                raise AsrQualityUnavailable(
                    "faster-whisper CT2 runtime is unavailable"
                ) from error
            factory = WhisperModel
        self._factory = factory
        self._repository = repository
        requested_device = device or ("cuda" if _cuda_available() else "cpu")
        if requested_device == "cuda":
            configure_cuda_runtime()
        try:
            self._model = self._load(
                factory,
                repository,
                device=requested_device,
            )
            self._device = requested_device
        except Exception as error:
            if requested_device != "cuda" or not _is_cuda_runtime_error(error):
                raise
            self._model = self._load(factory, repository, device="cpu")
            self._device = "cpu"

    @staticmethod
    def _load(
        factory: Callable[..., Any],
        repository: str,
        *,
        device: str,
    ) -> Any:
        compute_type = "float16" if device == "cuda" else "int8"
        return factory(
            repository,
            device=device,
            compute_type=compute_type,
            local_files_only=False,
            num_workers=1,
        )

    def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> str:
        audio = _decode_pcm(pcm_s16le)
        try:
            return self._transcribe_audio(audio, language=language, mode=mode)
        except Exception as error:
            if self._device != "cuda" or not _is_cuda_runtime_error(error):
                raise
            self.release()
            self._model = self._load(
                self._factory,
                self._repository,
                device="cpu",
            )
            self._device = "cpu"
            return self._transcribe_audio(audio, language=language, mode=mode)

    def _transcribe_audio(
        self,
        audio: np.ndarray,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=language.value,
            beam_size=_BEAM_SIZE[mode],
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments).strip()

    def release(self) -> None:
        native_model = getattr(self._model, "model", None)
        unload_model = getattr(native_model, "unload_model", None)
        if callable(unload_model):
            unload_model()


class QwenTransformersAsrProbe:
    def __init__(
        self,
        *,
        repository: str,
        processor_factory: Callable[[str], Any] | None = None,
        model_factory: Callable[[str], Any] | None = None,
        inference_context_factory: Callable[[], Any] | None = None,
    ) -> None:
        if processor_factory is None:
            try:
                from transformers import AutoProcessor
            except Exception as error:
                raise AsrQualityUnavailable(
                    "Qwen3-ASR requires the optional transformers package"
                ) from error
            processor_factory = AutoProcessor.from_pretrained
        if model_factory is None:
            try:
                from transformers import AutoModelForMultimodalLM
            except Exception as error:
                raise AsrQualityUnavailable(
                    "Qwen3-ASR requires the optional transformers package"
                ) from error

            def model_factory(model_id: str) -> Any:
                return AutoModelForMultimodalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                )

        if inference_context_factory is None:
            try:
                import torch

                inference_context_factory = torch.inference_mode
            except Exception:
                inference_context_factory = nullcontext
        self._processor = processor_factory(repository)
        self._model = model_factory(repository)
        self._inference_context_factory = inference_context_factory

    def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        language: Language,
        mode: TranslationMode,
        sample_rate_hz: int = _SAMPLE_RATE_HZ,
    ) -> str:
        del mode
        with tempfile.TemporaryDirectory(prefix="translator-qwen-asr-") as temp_dir:
            wav_path = Path(temp_dir) / "input.wav"
            _write_wav(wav_path, pcm_s16le, sample_rate_hz=sample_rate_hz)
            inputs = self._processor.apply_transcription_request(
                audio=str(wav_path),
                language=_QWEN_LANGUAGE[language],
            )
            inputs = inputs.to(self._model.device, self._model.dtype)
            with self._inference_context_factory():
                output_ids = self._model.generate(**inputs, max_new_tokens=256)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            decoded = self._processor.decode(
                generated_ids,
                return_format="transcription_only",
            )
            return str(decoded[0]).strip()


def _build_probe(candidate: ModelCandidate, *, manifest_path: Path) -> Any:
    if candidate.runtime == "local_provider":
        return FasterWhisperAsrProbe(
            model_id=candidate.id,
            manifest_path=manifest_path,
        )
    if candidate.runtime == "faster_whisper_ct2":
        return FasterWhisperCt2AsrProbe(repository=candidate.repository)
    if candidate.runtime == "transformers":
        return QwenTransformersAsrProbe(repository=candidate.repository)
    raise AsrQualityUnavailable(
        f"ASR runtime is not implemented yet: {candidate.runtime}"
    )


def _run_candidate(
    model_id: str,
    pcm_s16le: bytes,
    *,
    language: Language,
    mode: TranslationMode,
    reference: str | None,
    manifest_path: Path,
) -> dict[str, Any]:
    try:
        candidate = candidate_by_id(model_id, role="asr")
    except KeyError:
        return {
            "model_id": model_id,
            "status": "skipped",
            "skip_reason": "unknown_asr_candidate",
            "transcript": "",
            "wer": None,
            "wall_ms": None,
        }
    started_ns = time.monotonic_ns()
    probe: Any | None = None
    try:
        probe = _build_probe(candidate, manifest_path=manifest_path)
        transcript = probe.transcribe(pcm_s16le, language=language, mode=mode)
    except AsrQualityUnavailable as error:
        return {
            "model_id": model_id,
            "candidate": candidate.to_report(),
            "status": "skipped",
            "skip_reason": str(error),
            "transcript": "",
            "wer": None,
            "wall_ms": None,
        }
    except ImportError as error:
        return {
            "model_id": model_id,
            "candidate": candidate.to_report(),
            "status": "skipped",
            "skip_reason": _safe_error_summary(error),
            "transcript": "",
            "wer": None,
            "wall_ms": None,
        }
    except Exception as error:
        return {
            "model_id": model_id,
            "candidate": candidate.to_report(),
            "status": "failed",
            "skip_reason": _safe_error_summary(error),
            "transcript": "",
            "wer": None,
            "wall_ms": None,
        }
    finally:
        release = getattr(probe, "release", None)
        if callable(release):
            release()
    return {
        "model_id": model_id,
        "candidate": candidate.to_report(),
        "status": "completed",
        "skip_reason": None,
        "transcript": transcript,
        "wer": _safe_wer(reference, transcript),
        "wall_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
    }


def _parse_model_ids(values: list[str]) -> list[str]:
    model_ids: list[str] = []
    for value in values:
        model_ids.extend(item.strip() for item in value.split(",") if item.strip())
    return model_ids or default_executable_asr_candidate_ids()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ASR-only quality diagnostics on local s16le PCM audio.",
    )
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--reference")
    parser.add_argument(
        "--language",
        choices=[Language.RU.value, Language.EN.value],
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TranslationMode],
        default=TranslationMode.STREAMING_FIRST.value,
    )
    parser.add_argument(
        "--asr-model",
        action="append",
        default=[],
        help="ASR model id; repeat or pass comma-separated values.",
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    pcm_s16le = _read_pcm(args.audio)
    language = Language(args.language)
    mode = TranslationMode(args.mode)
    reports = [
        _run_candidate(
            model_id,
            pcm_s16le,
            language=language,
            mode=mode,
            reference=args.reference,
            manifest_path=args.manifest,
        )
        for model_id in _parse_model_ids(args.asr_model)
    ]
    report = {
        "schema_version": "translator.asr-quality-debug.v1",
        "generated_at": _utc_now(),
        "inputs": {
            "audio": str(args.audio),
            "language": language.value,
            "mode": mode.value,
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "reference_present": bool(args.reference),
        },
        "models": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except AsrQualityError as error:
        print(f"ASR quality diagnostics failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report["models"], ensure_ascii=False, indent=2))
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
