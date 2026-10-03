"""Run the full offline Task 6 benchmark on the target workstation."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from threading import Lock, Thread
from typing import Any, NoReturn
from uuid import UUID, uuid4

import psutil
import pynvml

from translator_sidecar.benchmark import process_run
from translator_sidecar.benchmark.task6 import (
    AsrBenchmarkConfig,
    DuplexBenchmarkConfig,
    DuplexBenchmarkReport,
    QualityBenchmarkRun,
    benchmark_asr_candidate,
    benchmark_simultaneous_duplex,
    load_quality_corpus,
    run_quality_benchmark,
)
from translator_sidecar.cleanup import finish_cleanup
from translator_sidecar.local.asr import AsrModelManager
from translator_sidecar.local.inference_scheduler import InferenceScheduler
from translator_sidecar.local.local_provider import LocalProvider
from translator_sidecar.local.model_lease import VerifiedModelSource
from translator_sidecar.local.model_manifest import load_manifest
from translator_sidecar.local.mt import NllbTranslator
from translator_sidecar.local.tts import PiperTts, PiperVoiceRegistry
from translator_sidecar.provider_contract import (
    AudioDirection,
    CloseRequestReason,
    ComputeDevice,
    Language,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderId,
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


def _resource_sample() -> tuple[float, int, float | None, int]:
    with _RESOURCE_SAMPLE_LOCK:
        cpu_percent = _PROCESS.cpu_percent(interval=None)
        rss_bytes = _PROCESS.memory_info().rss
        try:
            pynvml.nvmlInit()
            try:
                if pynvml.nvmlDeviceGetCount() != 1:
                    raise ResourceTelemetryError("GPU telemetry is unavailable")
                device = pynvml.nvmlDeviceGetHandleByIndex(0)
                try:
                    gpu_percent = float(
                        pynvml.nvmlDeviceGetUtilizationRates(device).gpu
                    )
                except pynvml.NVMLError_NotSupported:
                    gpu_percent = None
                vram_mib = int(pynvml.nvmlDeviceGetMemoryInfo(device).used) // 2**20
            finally:
                pynvml.nvmlShutdown()
        except (pynvml.NVMLError, AttributeError, TypeError, ValueError) as error:
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
    small_path: VerifiedModelSource,
    large_path: VerifiedModelSource,
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
    if len(duplex_candidates) != 2 or {
        candidate.model_id for candidate in duplex_candidates
    } != {_SMALL_ID, _LARGE_ID}:
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
        *,
        fatal_cleanup: Callable[[BaseException], NoReturn],
    ) -> None:
        self._provider = provider
        self._source_pcm = source_pcm
        self._fatal_cleanup = fatal_cleanup
        self._closed = False
        self._submissions: list[Future[float]] = []
        try:
            self._lock = Lock()
            self._loop = asyncio.new_event_loop()
            self._thread = Thread(
                target=self._run_loop,
                name="translator-task6-provider",
                daemon=True,
            )
            self._thread.start()
        except BaseException as error:
            fatal_cleanup(error)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_direction(
        self,
        source_language: Language,
        session_id: UUID,
    ) -> float:
        with self._lock:
            if self._closed:
                raise RuntimeError("benchmark provider is closed")
            future = asyncio.run_coroutine_threadsafe(
                self._run_session(source_language, session_id),
                self._loop,
            )
            self._submissions.append(future)
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
            provider_id=ProviderId.LOCAL,
            direction_id=direction,
            source_language=source_language,
            target_language=target_language,
            mode=TranslationMode.QUALITY_FIRST,
            requested_input_format=pcm_format,
            requested_output_format=pcm_format,
            voice_profile=_voice_profile(target_language),
        )
        first_audio_times_ns: list[int] = []
        safe_error_code: str | None = None
        final_outcome: str | None = None

        async def publish(batch, commit) -> None:
            nonlocal safe_error_code, final_outcome
            for event in batch:
                if isinstance(event, ProviderAudioDelta):
                    if event.session_id != session_id:
                        raise RuntimeError("provider session isolation failed")
                    if not first_audio_times_ns:
                        first_audio_times_ns.append(time.monotonic_ns())
                elif isinstance(event, PrivacySafeProviderError):
                    safe_error_code = event.code.value
                elif isinstance(event, ProviderUtteranceFinal):
                    final_outcome = event.outcome.value
            commit()

        started_ns = time.monotonic_ns()
        reservation = self._provider.reserve_session(request, publish)
        operation_error = None
        try:
            await reservation.open()
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
            if not first_audio_times_ns:
                raise RuntimeError(
                    "provider emitted no audio "
                    f"(error={safe_error_code}, outcome={final_outcome})"
                )
            return (first_audio_times_ns[0] - started_ns) / 1_000_000
        except BaseException as error:
            operation_error = error
            raise
        finally:
            try:
                receipt = await finish_cleanup(
                    reservation.drain(CloseRequestReason.USER_STOP)
                )
                if receipt.delivery_error is not None:
                    raise RuntimeError(receipt.delivery_error.value)
            except BaseException as cleanup_error:
                if operation_error is not None:
                    raise operation_error from cleanup_error
                raise

    def close(self) -> None:
        deadline = time.monotonic() + 30

        def remaining() -> float:
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("benchmark provider cleanup timed out")
            return value

        try:
            with self._lock:
                self._closed = True
                submissions = tuple(self._submissions)
            shutdown = asyncio.run_coroutine_threadsafe(
                self._provider.shutdown(), self._loop
            )
            shutdown.result(timeout=remaining())
            for future in submissions:
                try:
                    future.result(timeout=remaining())
                except BaseException:
                    if not future.done():
                        raise
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=remaining())
            if self._thread.is_alive():
                raise TimeoutError("benchmark provider thread did not stop")
            self._loop.close()
        except BaseException as error:
            self._fatal_cleanup(error)


def _benchmark_provider_duplex(
    *,
    asr: AsrModelManager,
    model_id: str,
    translator: NllbTranslator,
    tts: PiperTts,
    source_pcm: dict[Language, bytes],
    device: str,
    resources: ExitStack,
    fatal_cleanup: Callable[[BaseException], NoReturn],
    on_complete: Callable[[], None] | None = None,
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
    resources.pop_all()
    bridge = _ProviderDuplexBridge(provider, source_pcm, fatal_cleanup=fatal_cleanup)
    try:
        report = benchmark_simultaneous_duplex(
            DuplexBenchmarkConfig(model_id=model_id),
            run_direction=bridge.run_direction,
            resource_sample=_resource_sample,
        )
        if on_complete is not None:
            on_complete()
        return report
    finally:
        bridge.close()


def run(
    output_path: Path, *, limits: process_run.RunLimits = process_run.DEFAULT_LIMITS
) -> dict[str, Any]:
    return process_run.run_benchmark("task6", {}, output_path, limits=limits)


def _run_owned(*, fatal_cleanup: Callable[[BaseException], NoReturn]) -> dict[str, Any]:
    manifest = load_manifest(_MANIFEST_PATH)
    small_path = VerifiedModelSource(manifest, _SMALL_ID)
    large_path = VerifiedModelSource(manifest, _LARGE_ID)
    mt_path = VerifiedModelSource(manifest, _MT_ID)
    voice_paths = {
        profile: VerifiedModelSource(manifest, model_id)
        for profile, model_id in _VOICE_IDS.items()
    }
    device = "cuda" if _cuda_available() else "cpu"
    with ExitStack() as voices:
        voice_smoke_tts = PiperTts(PiperVoiceRegistry(voice_paths))
        voices.callback(voice_smoke_tts.close)
        voice_profiles = _run_voice_smokes(voice_smoke_tts)
    del voice_smoke_tts
    gc.collect()
    with ExitStack() as resources:
        tts = PiperTts(PiperVoiceRegistry(voice_paths))
        resources.callback(tts.close)
        fixture_text = "The audio path is ready for the benchmark."
        fixture_pcm = _synthesize_pcm(tts, fixture_text, Language.EN)
        fixture_duration_ms = len(fixture_pcm) * 1_000 // (16_000 * 2)

        asr_reports = []
        for model_id in (_SMALL_ID, _LARGE_ID):
            with ExitStack() as candidate_resources:
                holder: list[AsrModelManager] = []

                def factory(
                    selected: str = model_id,
                    selected_holder: list[AsrModelManager] = holder,
                    selected_resources: ExitStack = candidate_resources,
                ) -> AsrModelManager:
                    manager = _asr_manager(
                        selected_id=selected,
                        small_path=small_path,
                        large_path=large_path,
                        device=device,
                    )
                    selected_resources.callback(manager.close)
                    selected_holder.append(manager)
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
                    alternate_asr = candidate
                    resources.enter_context(candidate_resources.pop_all())
            holder.clear()
            del candidate
            if model_id == _SMALL_ID:
                gc.collect()

        translator = NllbTranslator.load(mt_path, device=device)
        resources.callback(translator.close)
        corpus = load_quality_corpus(_CORPUS_PATH)

        def synthesize_and_transcribe(
            text: str, language: Language, asr: AsrModelManager = alternate_asr
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
            Language.RU: _synthesize_pcm(tts, "Проверка микрофона.", Language.RU),
            Language.EN: _synthesize_pcm(tts, "Microphone check.", Language.EN),
        }
        large_duplex = _benchmark_provider_duplex(
            asr=alternate_asr,
            model_id=_LARGE_ID,
            translator=translator,
            tts=tts,
            source_pcm=source_pcm,
            device=device,
            resources=resources,
            fatal_cleanup=fatal_cleanup,
        )
    del alternate_asr
    gc.collect()
    with ExitStack() as resources:
        normal_asr = _asr_manager(
            selected_id=_SMALL_ID,
            small_path=small_path,
            large_path=large_path,
            device=device,
        )
        resources.callback(normal_asr.close)
        translator = NllbTranslator.load(mt_path, device=device)
        resources.callback(translator.close)
        tts = PiperTts(PiperVoiceRegistry(voice_paths))
        resources.callback(tts.close)
        normal_runtime = {}

        def capture_normal_runtime() -> None:
            final_resources = _resource_sample()
            normal_runtime.update(
                {
                    "selected_asr": _SMALL_ID,
                    "actual_device": normal_asr.actual_device,
                    "resident_model_id": normal_asr.resident_model_id,
                    "vram_mib_after": final_resources[3],
                }
            )

        small_duplex = _benchmark_provider_duplex(
            asr=normal_asr,
            model_id=_SMALL_ID,
            translator=translator,
            tts=tts,
            source_pcm=source_pcm,
            device=device,
            on_complete=capture_normal_runtime,
            resources=resources,
            fatal_cleanup=fatal_cleanup,
        )
    return _build_payload(
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
        normal_runtime=normal_runtime,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
    )
    parser.add_argument("--model-run-seconds", type=float, default=3600)
    parser.add_argument("--terminate-grace-seconds", type=float, default=5)
    arguments = parser.parse_args()
    run(
        arguments.output.resolve(),
        limits=process_run.RunLimits(
            model_run_seconds=arguments.model_run_seconds,
            terminate_grace_seconds=arguments.terminate_grace_seconds,
        ),
    )


if __name__ == "__main__":
    main()
