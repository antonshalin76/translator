"""Run the full offline Task 6 benchmark on the target workstation."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc
import json
from pathlib import Path
import subprocess
from threading import Lock, Thread
import time
from typing import Any
from uuid import UUID, uuid4

import psutil

from translator_sidecar.benchmark.task6 import (
    AsrBenchmarkConfig,
    DuplexBenchmarkReport,
    DuplexBenchmarkConfig,
    QualityBenchmarkRun,
    benchmark_asr_candidate,
    benchmark_simultaneous_duplex,
    load_quality_corpus,
    run_quality_benchmark,
)
from translator_sidecar.local.asr import AsrModelManager
from translator_sidecar.local.inference_scheduler import InferenceScheduler
from translator_sidecar.local.local_provider import LocalProvider
from translator_sidecar.local.model_manifest import load_manifest
from translator_sidecar.local.mt import NllbTranslator
from translator_sidecar.local.tts import PiperTts, PiperVoiceRegistry
from translator_sidecar.provider_contract import (
    AudioDirection,
    CloseProviderSession,
    CloseRequestReason,
    ComputeDevice,
    Language,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderInputFrame,
    ProviderUtteranceFinal,
    SampleFormat,
    TranslationMode,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _ROOT / "models" / "manifest.json"
_CORPUS_PATH = _ROOT / "sidecar" / "tests" / "quality_corpus" / "task6-v4.json"
_DEFAULT_OUTPUT = _ROOT / "docs" / "benchmarks" / "task6-results.json"
_SMALL_ID = "faster-whisper-small"
_LARGE_ID = "faster-whisper-large-v3"
_MT_ID = "nllb-200-distilled-600m-ct2-int8"
_VOICE_IDS = {
    (Language.RU, VoiceGender.MALE): "piper-ru-dmitri-medium",
    (Language.EN, VoiceGender.MALE): "piper-en-ryan-medium",
    (Language.RU, VoiceGender.FEMALE): "piper-ru-irina-medium",
    (Language.EN, VoiceGender.FEMALE): "piper-en-hfc-female-medium",
}
_PROCESS = psutil.Process()
_PROCESS.cpu_percent(interval=None)
_RESOURCE_SAMPLE_LOCK = Lock()


class ResourceTelemetryError(RuntimeError):
    """A benchmark resource sample could not be measured reliably."""


def _resource_sample() -> tuple[float, int, float, int]:
    with _RESOURCE_SAMPLE_LOCK:
        cpu_percent = _PROCESS.cpu_percent(interval=None)
        rss_bytes = _PROCESS.memory_info().rss
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            values = result.stdout.strip().split(",", maxsplit=1)
            gpu_percent = float(values[0].strip())
            vram_mib = int(values[1].strip())
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise ResourceTelemetryError("GPU telemetry is unavailable") from error
    return cpu_percent, rss_bytes, gpu_percent, vram_mib


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _voice_profile(
    language: Language,
    gender: VoiceGender = VoiceGender.MALE,
) -> VoiceProfile:
    return VoiceProfile(
        language=language,
        gender=gender,
        engine=VoiceEngine.PIPER,
    )


def _synthesize_pcm(
    tts: PiperTts,
    text: str,
    language: Language,
) -> bytes:
    return b"".join(
        tts.synthesize_frames(
            text,
            target_language=language,
            voice_profile=_voice_profile(language),
            mode=TranslationMode.QUALITY_FIRST,
            output_sample_rate_hz=16_000,
            output_channels=1,
            frame_duration_ms=100,
        )
    )


def _run_voice_smokes(tts: Any) -> list[dict[str, Any]]:
    profiles = []
    for language in Language:
        text = "Проверка голоса." if language is Language.RU else "Voice check."
        for gender in VoiceGender:
            frames = list(
                tts.synthesize_frames(
                    text,
                    target_language=language,
                    voice_profile=_voice_profile(language, gender),
                    mode=TranslationMode.QUALITY_FIRST,
                    output_sample_rate_hz=16_000,
                    output_channels=1,
                    frame_duration_ms=100,
                )
            )
            pcm_bytes = sum(len(frame) for frame in frames)
            if not frames or pcm_bytes == 0:
                raise RuntimeError("Piper voice smoke emitted no audio")
            profiles.append(
                {
                    "language": language.value,
                    "gender": gender.value,
                    "frame_count": len(frames),
                    "pcm_bytes": pcm_bytes,
                }
            )
    return profiles


def _asr_manager(
    *,
    selected_id: str,
    small_path: Path,
    large_path: Path,
    device: str,
) -> AsrModelManager:
    model_paths = {
        "small": small_path,
        "large-v3": large_path,
    }
    return AsrModelManager(
        selected_id="small" if selected_id == _SMALL_ID else "large-v3",
        model_paths=model_paths,
        device=device,
    )


def _build_payload(
    *,
    generated_at_unix_ns: int,
    environment: dict[str, Any],
    fixture: dict[str, Any],
    asr_candidates: list[dict[str, Any]],
    voice_profiles: list[dict[str, Any]],
    quality_run: QualityBenchmarkRun,
    duplex_candidates: tuple[DuplexBenchmarkReport, ...],
    normal_runtime: dict[str, Any],
) -> dict[str, Any]:
    if (
        len(duplex_candidates) != 2
        or {candidate.model_id for candidate in duplex_candidates}
        != {_SMALL_ID, _LARGE_ID}
    ):
        raise RuntimeError("both ASR duplex candidates are required")
    quality_payload = dataclasses.asdict(quality_run)
    quality_payload["passes_thresholds"] = quality_run.passes_thresholds
    quality_payload["quality"]["passes_thresholds"] = (
        quality_run.quality.passes_thresholds
    )
    quality_payload["quality"]["ru_to_en"]["passes_thresholds"] = (
        quality_run.quality.ru_to_en.passes_thresholds
    )
    quality_payload["quality"]["en_to_ru"]["passes_thresholds"] = (
        quality_run.quality.en_to_ru.passes_thresholds
    )
    quality_payload["ru_to_en"]["passes_drop_threshold"] = (
        quality_run.ru_to_en.passes_drop_threshold
    )
    quality_payload["en_to_ru"]["passes_drop_threshold"] = (
        quality_run.en_to_ru.passes_drop_threshold
    )
    duplex_payloads = []
    for duplex in duplex_candidates:
        duplex_payload = dataclasses.asdict(duplex)
        duplex_payload["vram_within_budget"] = duplex.vram_within_budget
        duplex_payloads.append(duplex_payload)
    return {
        "schema_version": "translator.task6-benchmark.v2",
        "generated_at_unix_ns": generated_at_unix_ns,
        "environment": environment,
        "fixture": fixture,
        "asr_candidates": asr_candidates,
        "voice_profiles": voice_profiles,
        "quality": quality_payload,
        "duplex_candidates": duplex_payloads,
        "normal_runtime": normal_runtime,
    }


class _ProviderDuplexBridge:
    """Run one LocalProvider on one event loop from benchmark workers."""

    def __init__(
        self,
        provider: LocalProvider,
        source_pcm: dict[Language, bytes],
    ) -> None:
        self._provider = provider
        self._source_pcm = source_pcm
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(
            target=self._run_loop,
            name="translator-task6-provider",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_direction(
        self,
        source_language: Language,
        session_id: UUID,
    ) -> float:
        future = asyncio.run_coroutine_threadsafe(
            self._run_session(source_language, session_id),
            self._loop,
        )
        return float(future.result(timeout=120))

    async def _run_session(
        self,
        source_language: Language,
        session_id: UUID,
    ) -> float:
        target_language = Language.EN if source_language is Language.RU else Language.RU
        direction = (
            AudioDirection.MICROPHONE
            if source_language is Language.RU
            else AudioDirection.SPEAKER
        )
        pcm_format = PcmFormat(
            sample_rate_hz=16_000,
            channels=1,
            sample_format=SampleFormat.S16LE,
            frame_duration_ms=100,
        )
        request = OpenProviderSession(
            session_id=session_id,
            direction_id=direction,
            source_language=source_language,
            target_language=target_language,
            mode=TranslationMode.QUALITY_FIRST,
            requested_input_format=pcm_format,
            requested_output_format=pcm_format,
            voice_profile=_voice_profile(target_language),
        )
        first_audio_ns: int | None = None
        safe_error_code: str | None = None
        final_outcome: str | None = None

        async def publish(batch, commit) -> None:
            nonlocal first_audio_ns, safe_error_code, final_outcome
            for event in batch:
                if isinstance(event, ProviderAudioDelta):
                    if event.session_id != session_id:
                        raise RuntimeError("provider session isolation failed")
                    if first_audio_ns is None:
                        first_audio_ns = time.monotonic_ns()
                elif isinstance(event, PrivacySafeProviderError):
                    safe_error_code = event.code.value
                elif isinstance(event, ProviderUtteranceFinal):
                    final_outcome = event.outcome.value
            commit()

        started_ns = time.monotonic_ns()
        await self._provider.open_session(request, publish)
        stream_id = uuid4()
        utterance_id = uuid4()
        frame_bytes = 16_000 * 2 * pcm_format.frame_duration_ms // 1_000
        pcm = self._source_pcm[source_language]
        frames = [
            pcm[offset : offset + frame_bytes]
            for offset in range(0, len(pcm), frame_bytes)
        ]
        if not frames:
            raise RuntimeError("provider benchmark PCM is empty")
        frames[-1] = frames[-1].ljust(frame_bytes, b"\0")
        for sequence, frame in enumerate(frames):
            await self._provider.submit_frame(
                ProviderInputFrame(
                    session_id=session_id,
                    direction_id=direction,
                    stream_id=stream_id,
                    utterance_id=utterance_id,
                    sequence=sequence,
                    capture_monotonic_ns=started_ns,
                    sample_rate_hz=16_000,
                    channels=1,
                    sample_format=SampleFormat.S16LE,
                    frame_duration_ms=100,
                    source_language=source_language,
                    target_language=target_language,
                    mode=TranslationMode.QUALITY_FIRST,
                    pcm=frame,
                    end_of_utterance=sequence == len(frames) - 1,
                )
            )
        await self._provider.wait_idle()
        if first_audio_ns is None:
            raise RuntimeError(
                "provider emitted no audio "
                f"(error={safe_error_code}, outcome={final_outcome})"
            )
        await self._provider.close_session(
            CloseProviderSession(
                session_id=session_id,
                reason=CloseRequestReason.USER_STOP,
            )
        )
        await self._provider.wait_publications(session_id)
        return (first_audio_ns - started_ns) / 1_000_000

    def close(self) -> None:
        shutdown = asyncio.run_coroutine_threadsafe(
            self._provider.shutdown(),
            self._loop,
        )
        try:
            shutdown.result(timeout=30)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=30)
            self._loop.close()


def _benchmark_provider_duplex(
    *,
    asr: AsrModelManager,
    model_id: str,
    translator: NllbTranslator,
    tts: PiperTts,
    source_pcm: dict[Language, bytes],
    device: str,
) -> DuplexBenchmarkReport:
    provider = LocalProvider(
        asr=asr,
        translator=translator,
        tts=tts,
        scheduler=InferenceScheduler(),
        now_ns=time.monotonic_ns,
        asr_model_id=model_id,
        mt_model_id=_MT_ID,
        tts_model_id="piper-presets-v1",
        mt_device=(ComputeDevice.CUDA if device == "cuda" else ComputeDevice.CPU),
    )
    bridge = _ProviderDuplexBridge(provider, source_pcm)
    try:
        return benchmark_simultaneous_duplex(
            DuplexBenchmarkConfig(model_id=model_id),
            run_direction=bridge.run_direction,
            resource_sample=_resource_sample,
        )
    finally:
        bridge.close()


def _release_quality_asr_then_create_normal(
    quality_asr: AsrModelManager,
    *,
    small_path: Path,
    large_path: Path,
    device: str,
) -> AsrModelManager:
    if not quality_asr.release():
        raise RuntimeError("ASR quality oracle release failed")
    gc.collect()
    return _asr_manager(
        selected_id=_SMALL_ID,
        small_path=small_path,
        large_path=large_path,
        device=device,
    )


def run(output_path: Path) -> dict[str, Any]:
    manifest = load_manifest(_MANIFEST_PATH)
    for model in manifest.models.values():
        for model_file in model.files:
            manifest.resolve_runtime_file(model.id, model_file.path)
    small_path = manifest.models[_SMALL_ID].cache_path
    large_path = manifest.models[_LARGE_ID].cache_path
    mt_path = manifest.models[_MT_ID].cache_path
    voice_paths = {
        profile: manifest.models[model_id].cache_path
        / next(
            file.path
            for file in manifest.models[model_id].files
            if file.path.endswith(".onnx")
        )
        for profile, model_id in _VOICE_IDS.items()
    }
    device = "cuda" if _cuda_available() else "cpu"
    voice_smoke_tts = PiperTts(PiperVoiceRegistry(voice_paths))
    voice_profiles = _run_voice_smokes(voice_smoke_tts)
    del voice_smoke_tts
    gc.collect()
    tts = PiperTts(PiperVoiceRegistry(voice_paths))
    fixture_text = "The audio path is ready for the benchmark."
    fixture_pcm = _synthesize_pcm(tts, fixture_text, Language.EN)
    fixture_duration_ms = len(fixture_pcm) * 1_000 // (16_000 * 2)

    asr_reports = []
    large_holder: list[AsrModelManager] = []
    for model_id in (_SMALL_ID, _LARGE_ID):
        holder: list[AsrModelManager] = []

        def factory(selected: str = model_id) -> AsrModelManager:
            manager = _asr_manager(
                selected_id=selected,
                small_path=small_path,
                large_path=large_path,
                device=device,
            )
            holder.append(manager)
            return manager

        report = benchmark_asr_candidate(
            AsrBenchmarkConfig(
                model_id=model_id,
                audio_duration_ms=fixture_duration_ms,
            ),
            adapter_factory=factory,
            pcm=fixture_pcm,
            language=Language.EN,
            now_ns=time.monotonic_ns,
            resource_sample=_resource_sample,
        )
        candidate = holder[0]
        payload = dataclasses.asdict(report)
        payload.update(
            {
                "actual_device": candidate.actual_device,
                "resident_model_id": candidate.resident_model_id,
                "degraded": candidate.degraded,
            }
        )
        asr_reports.append(payload)
        if model_id == _LARGE_ID:
            large_holder.append(candidate)
        elif not candidate.release():
            raise RuntimeError("ASR candidate release failed")
        holder.clear()
        del candidate
        if model_id == _SMALL_ID:
            gc.collect()

    translator = NllbTranslator.load(mt_path, device=device)
    corpus = load_quality_corpus(_CORPUS_PATH)
    alternate_asr = large_holder[0]

    def synthesize_and_transcribe(
        text: str,
        language: Language,
        asr: AsrModelManager = alternate_asr,
    ) -> str:
        return asr.transcribe(
            _synthesize_pcm(tts, text, language),
            language=language,
            mode=TranslationMode.QUALITY_FIRST,
        )

    quality_run = run_quality_benchmark(
        corpus,
        translator=translator,
        synthesize_and_transcribe=synthesize_and_transcribe,
        now_ns=time.monotonic_ns,
    )
    del synthesize_and_transcribe
    source_pcm = {
        Language.RU: _synthesize_pcm(
            tts,
            "Проверка микрофона.",
            Language.RU,
        ),
        Language.EN: _synthesize_pcm(
            tts,
            "Microphone check.",
            Language.EN,
        ),
    }
    large_duplex = _benchmark_provider_duplex(
        asr=alternate_asr,
        model_id=_LARGE_ID,
        translator=translator,
        tts=tts,
        source_pcm=source_pcm,
        device=device,
    )
    normal_asr = _release_quality_asr_then_create_normal(
        alternate_asr,
        small_path=small_path,
        large_path=large_path,
        device=device,
    )
    large_holder.clear()
    del alternate_asr
    small_duplex = _benchmark_provider_duplex(
        asr=normal_asr,
        model_id=_SMALL_ID,
        translator=translator,
        tts=tts,
        source_pcm=source_pcm,
        device=device,
    )
    final_resources = _resource_sample()
    payload = _build_payload(
        generated_at_unix_ns=time.time_ns(),
        environment={
            "device": device,
            "cuda_available": _cuda_available(),
            "logical_cpu_count": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
        },
        fixture={
            "sample_rate_hz": 16_000,
            "duration_ms": fixture_duration_ms,
        },
        asr_candidates=asr_reports,
        voice_profiles=voice_profiles,
        quality_run=quality_run,
        duplex_candidates=(small_duplex, large_duplex),
        normal_runtime={
            "selected_asr": _SMALL_ID,
            "actual_device": normal_asr.actual_device,
            "resident_model_id": normal_asr.resident_model_id,
            "vram_mib_after": final_resources[3],
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
    )
    arguments = parser.parse_args()
    run(arguments.output.resolve())


if __name__ == "__main__":
    main()
