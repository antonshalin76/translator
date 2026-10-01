"""Replay frozen MDC WAVs through both real local product chains, serially.

This is a private, accelerated saved-audio diagnostic. It does not measure
capture, acoustic admission, playback, or first audible sound.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import stat
import subprocess
import time
import wave
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any, TextIO
from uuid import UUID, uuid4

import psutil
from translator_mdc_asr_run import validate_manifest
from translator_sidecar.cleanup import finish_cleanup
from translator_sidecar.local.runtime import build_local_provider
from translator_sidecar.provider_contract import (
    AudioDirection,
    CloseRequestReason,
    Language,
    ModelKind,
    ModelState,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderId,
    ProviderInputFrame,
    ProviderLatency,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SampleFormat,
    TranslationMode,
    UtteranceOutcome,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)

ROOT = Path(__file__).resolve().parents[1]
HY_SERVER = Path("/usr/local/lib/ollama/llama-server")
MANIFEST_SHA256 = "bc2d204c31dbe0cba78187975f02a8a4378cc09407573ca4912ba540bf44ca26"
SCREEN_SHA256 = "005db22422ad98c9c70ecd3d01f578be571ea4bebca81398d5fff7cda86214ae"
TURBO_SHA256 = "03ebba9576e14e2fd6adb909f25f806583845a185a7df941d0c64b3a6297ac91"
PRODUCT_MANIFEST_SHA256 = (
    "d8f73beb4e9bc2b403405e4b54b18373386ecf05de420702cb7cc9b391c758dc"
)
ASR_ID = "faster-whisper-large-v3-turbo"
SMALL_ASR_ID = "faster-whisper-small"
BACKENDS = {
    "nllb": "nllb-200-distilled-600m-ct2-int8",
    "hy": "hy-mt2-1.8b-gguf-q4-k-m",
}
ARMS = {
    "small_nllb_same_code": (SMALL_ASR_ID, BACKENDS["nllb"]),
    "nllb": (ASR_ID, BACKENDS["nllb"]),
    "hy": (ASR_ID, BACKENDS["hy"]),
}
FRAME_BYTES = 3200  # 100 ms of 16-kHz mono s16le input
OUTPUT_FRAME_BYTES = 960  # 20 ms of 24-kHz mono s16le output
HEALTH_TIMEOUT_SECONDS = 2
SAFE_EVALUATION_REASONS = frozenset(
    {
        "provider event precedes saved-audio submission",
        "provider returned a safe error",
        "provider did not complete a text-and-audio utterance",
        "provider emitted malformed product PCM",
        "provider emitted silent product PCM",
        "effective model identity differs",
        "provider event identity differs from submitted utterance",
        "provider event sequence or terminal order differs",
    }
)


class VerifiedProviderDrop(ValueError):
    def __init__(self, latency: ProviderLatency) -> None:
        super().__init__("provider returned a safe error")
        self.safe_provider_codes = [SafeErrorCode.QUEUE_OVERFLOW.value]
        self.outcomes = [UtteranceOutcome.DROPPED.value]
        self.safe_provider_latency = {
            field: getattr(latency, field)
            for field in (
                "asr_final_text_ms",
                "mt_first_text_ms",
                "tts_first_audio_ms",
                "provider_total_ms",
            )
        }
        self.attested_after_drain = False


def _verified_terminal_drop(
    event_times: list[tuple[Any, int]],
    *,
    session_id: UUID,
    direction_id: AudioDirection,
    stream_id: UUID,
    utterance_id: UUID,
) -> ProviderLatency | None:
    events = [event for event, _ in event_times]
    if len(events) < 3 or not (
        isinstance(events[-3], ProviderLatency)
        and isinstance(events[-2], PrivacySafeProviderError)
        and isinstance(events[-1], ProviderUtteranceFinal)
    ):
        return None
    latency, error, final = events[-3:]
    prefix = events[:-3]
    if (
        error.code is not SafeErrorCode.QUEUE_OVERFLOW
        or error.retryable is not True
        or final.outcome is not UtteranceOutcome.DROPPED
        or final.final_audio_sequence is not None
        or error.event_sequence != latency.event_sequence + 1
        or final.event_sequence != error.event_sequence + 1
        or latency.provider_total_ms is None
        or (
            bool(prefix)
            and not (
                len(prefix) == 2
                and isinstance(prefix[0], ProviderTranscriptDelta)
                and prefix[0].is_final
                and isinstance(prefix[1], ProviderTranslationDelta)
                and prefix[1].is_final
            )
        )
        or any(
            event.session_id != session_id
            or event.direction_id is not direction_id
            or event.stream_id != stream_id
            or event.utterance_id != utterance_id
            for event in events
        )
        or any(
            current.event_sequence <= previous.event_sequence
            for previous, current in pairwise(events)
        )
        or any(
            current_at < previous_at
            for (_, previous_at), (_, current_at) in pairwise(event_times)
        )
        or any(
            value is not None and (type(value) is not int or value < 0)
            for value in (
                latency.asr_final_text_ms,
                latency.mt_first_text_ms,
                latency.tts_first_audio_ms,
                latency.provider_total_ms,
            )
        )
    ):
        return None
    return latency


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pinned_json(path: Path, expected: str) -> dict[str, Any]:
    if path.is_symlink() or sha256(path) != expected:
        raise ValueError("frozen input hash or path changed")
    return json.loads(path.read_text(encoding="utf-8"))


def _pcm_from_wav(path: Path, expected_sha256: str) -> bytes:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("frozen WAV symlink is forbidden")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("frozen WAV hash changed before submission")
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != 16_000
                or source.getcomptype() != "NONE"
            ):
                raise ValueError("frozen WAV format is unsupported")
            pcm = source.readframes(source.getnframes())
    except wave.Error as error:
        raise ValueError("frozen WAV is malformed") from error
    if len(pcm) < FRAME_BYTES or len(pcm) % 2:
        raise ValueError("frozen WAV has no usable PCM")
    return pcm


def load_cases(
    manifest_path: Path,
    screen_path: Path,
    turbo_path: Path,
    *,
    manifest_sha256: str = MANIFEST_SHA256,
    screen_sha256: str = SCREEN_SHA256,
    turbo_sha256: str = TURBO_SHA256,
) -> list[dict[str, Any]]:
    """Bind selected rows to one verified WAV and its frozen Turbo provenance."""
    if manifest_path.is_symlink():
        raise ValueError("frozen manifest symlink is forbidden")
    samples = validate_manifest(manifest_path, manifest_sha256)
    screen = _read_pinned_json(screen_path, screen_sha256)
    turbo = _read_pinned_json(turbo_path, turbo_sha256)
    if (
        turbo.get("model_id") != "turbo"
        or turbo.get("manifest_sha256") != manifest_sha256
        or screen.get("turbo_report_sha256") != turbo_sha256
    ):
        raise ValueError("frozen Turbo provenance changed")

    def index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["origin_id"], row["condition"])
            if key in indexed:
                raise ValueError("frozen selection contains duplicate cases")
            indexed[key] = row
        return indexed

    sample_index = index(samples)
    turbo_index = index(turbo["results"])
    selected = index(screen["cases"])
    if not selected:
        raise ValueError("frozen selection is empty")
    cases: list[dict[str, Any]] = []
    for key in selected:
        sample = sample_index.get(key)
        saved = turbo_index.get(key)
        if sample is None or saved is None:
            raise ValueError("frozen selection is missing a selected case")
        if (
            key[1] != "clean"
            or sample["language"] not in {"ru_ru", "en_us"}
            or sample["language"] != saved.get("language")
            or sample["audio_file"] != saved.get("audio_file")
            or sample["speaker_id"] != saved.get("speaker_id")
            or sample["reference"] != saved.get("reference")
        ):
            raise ValueError("frozen case provenance changed")
        if (
            saved.get("status") != "completed"
            or not saved.get("transcript", "").strip()
        ):
            raise ValueError("frozen Turbo transcript is unusable")
        relative = Path(sample["audio_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("frozen WAV path is outside the corpus")
        wav = manifest_path.parent / relative
        pcm = _pcm_from_wav(wav, sample["sha256"])
        cases.append(
            {
                "origin_id": key[0],
                "condition": key[1],
                "language": sample["language"],
                "speaker_id": sample["speaker_id"],
                "critical_labels": sample.get("critical_labels", []),
                "reference": sample["reference"],
                "turbo_text": saved["transcript"],
                "audio_file": sample["audio_file"],
                "wav_sha256": sample["sha256"],
                "pcm": pcm,
            }
        )
    return cases


def open_journal(path: Path) -> TextIO:
    if (
        not path.is_absolute()
        or path.resolve().is_relative_to(ROOT)
        or not path.parent.is_dir()
        or any(parent.is_symlink() for parent in path.parents)
        or stat.S_IMODE(path.parent.stat().st_mode) != 0o700
    ):
        raise ValueError("output must be new in a private directory outside Git")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    return os.fdopen(fd, "w", encoding="utf-8")


def write_record(journal: TextIO, record: dict[str, Any]) -> None:
    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
    journal.flush()
    os.fsync(journal.fileno())


def evaluate_events(
    event_times: list[tuple[Any, int]],
    *,
    started_ns: int,
    session_id: UUID,
    direction_id: AudioDirection,
    stream_id: UUID,
    utterance_id: UUID,
) -> dict[str, Any]:
    if any(at < started_ns for _, at in event_times):
        raise ValueError("provider event precedes saved-audio submission")
    if any(isinstance(event, PrivacySafeProviderError) for event, _ in event_times):
        latency = _verified_terminal_drop(
            event_times,
            session_id=session_id,
            direction_id=direction_id,
            stream_id=stream_id,
            utterance_id=utterance_id,
        )
        if latency is not None:
            raise VerifiedProviderDrop(latency)
        raise ValueError("provider returned a safe error")
    utterance_events = [
        event
        for event, _ in event_times
        if isinstance(
            event,
            (
                ProviderTranscriptDelta,
                ProviderTranslationDelta,
                ProviderAudioDelta,
                ProviderLatency,
                ProviderUtteranceFinal,
            ),
        )
    ]
    if any(
        event.session_id != session_id
        or event.direction_id != direction_id
        or event.stream_id != stream_id
        or event.utterance_id != utterance_id
        for event in utterance_events
    ):
        raise ValueError("provider event identity differs from submitted utterance")
    if any(
        current.event_sequence <= previous.event_sequence
        for previous, current in pairwise(utterance_events)
    ):
        raise ValueError("provider event sequence or terminal order differs")
    final = [
        event for event, _ in event_times if isinstance(event, ProviderUtteranceFinal)
    ]
    asr = [
        event.text
        for event, _ in event_times
        if isinstance(event, ProviderTranscriptDelta) and event.is_final
    ]
    mt = [
        event.text
        for event, _ in event_times
        if isinstance(event, ProviderTranslationDelta) and event.is_final
    ]
    audio = [
        (event, at)
        for event, at in event_times
        if isinstance(event, ProviderAudioDelta)
    ]
    if (
        len(final) != 1
        or final[0].outcome is not UtteranceOutcome.COMPLETED
        or len(asr) != 1
        or not asr[0].strip()
        or len(mt) != 1
        or not mt[0].strip()
        or not audio
        or len(audio) > 1500
    ):
        raise ValueError("provider did not complete a text-and-audio utterance")
    digest = hashlib.sha256()
    nonzero = False
    for frame, _ in audio:
        if (
            frame.sample_rate_hz != 24_000
            or frame.channels != 1
            or frame.sample_format is not SampleFormat.S16LE
            or frame.frame_duration_ms != 20
            or len(frame.pcm) != OUTPUT_FRAME_BYTES
        ):
            raise ValueError("provider emitted malformed product PCM")
        digest.update(frame.pcm)
        nonzero |= any(frame.pcm)
    if not nonzero:
        raise ValueError("provider emitted silent product PCM")
    latencies = [
        event for event, _ in event_times if isinstance(event, ProviderLatency)
    ]
    if (
        [frame.sequence for frame, _ in audio] != list(range(len(audio)))
        or len(latencies) != 1
        or utterance_events[-2:] != [latencies[0], final[0]]
        or final[0].final_audio_sequence != len(audio) - 1
    ):
        raise ValueError("provider event sequence or terminal order differs")
    return {
        "status": "completed",
        "asr_text": asr[0],
        "mt_text": mt[0],
        "first_pcm_ms": round((audio[0][1] - started_ns) / 1_000_000, 2),
        "pcm_duration_ms": len(audio) * 20,
        "pcm_sha256": digest.hexdigest(),
        "provider_latency": latencies[0].model_dump(mode="json"),
    }


def pair_rows(
    rows: list[dict[str, Any]], requested_mode: TranslationMode
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("mode") != requested_mode.value:
            raise ValueError("paired diagnostic mode differs from requested mode")
        if any(
            not row.get(field)
            for field in (
                "wav_sha256",
                "language",
                "speaker_id",
                "reference",
                "voice_gender",
            )
        ):
            raise ValueError("paired diagnostic missing input identity")
        key = (row["origin_id"], row["condition"])
        arm = indexed.setdefault(key, {})
        if row["backend"] not in ARMS:
            raise ValueError("paired diagnostic contains unknown backend")
        if row["backend"] in arm:
            raise ValueError("paired diagnostic contains duplicate backend")
        arm[row["backend"]] = row
    pairs = []
    for (origin_id, condition), arm in indexed.items():
        baseline, left, right = (
            arm.get("small_nllb_same_code"),
            arm.get("nllb"),
            arm.get("hy"),
        )
        complete = (
            baseline is not None
            and left is not None
            and right is not None
            and baseline["status"] == left["status"] == right["status"] == "completed"
        )
        anchor = next(iter(arm.values()))
        if any(
            anchor.get(field) != candidate.get(field)
            for candidate in arm.values()
            for field in (
                "wav_sha256",
                "language",
                "speaker_id",
                "reference",
                "voice_gender",
            )
        ):
            raise ValueError("paired diagnostic input identity differs")
        pairs.append(
            {
                "origin_id": origin_id,
                "condition": condition,
                "mode": requested_mode.value,
                "status": "complete" if complete else "incomplete",
                "mt_confounded_by_asr": (
                    left["asr_text"] != right["asr_text"] if complete else None
                ),
                "baseline_asr_divergent": (
                    baseline["asr_text"] != left["asr_text"] if complete else None
                ),
            }
        )
    return pairs


def _request(
    language: str, mode: TranslationMode, voice_gender: VoiceGender
) -> OpenProviderSession:
    source, target, direction = (
        (Language.RU, Language.EN, AudioDirection.MICROPHONE)
        if language == "ru_ru"
        else (Language.EN, Language.RU, AudioDirection.SPEAKER)
    )
    return OpenProviderSession(
        session_id=uuid4(),
        provider_id=ProviderId.LOCAL,
        direction_id=direction,
        source_language=source,
        target_language=target,
        mode=mode,
        requested_input_format=PcmFormat(
            sample_rate_hz=16_000,
            channels=1,
            sample_format=SampleFormat.S16LE,
            frame_duration_ms=100,
        ),
        requested_output_format=PcmFormat(
            sample_rate_hz=24_000,
            channels=1,
            sample_format=SampleFormat.S16LE,
            frame_duration_ms=20,
        ),
        voice_profile=VoiceProfile(
            language=target,
            gender=voice_gender,
            engine=VoiceEngine.PIPER,
        ),
        debug_text_enabled=True,
    )


def _checked_health(
    health: Any,
    request: OpenProviderSession,
    expected_asr_id: str,
    expected_mt_id: str,
) -> dict[str, dict[str, str]]:
    models = {model.kind: model for model in health.models}
    observed = {
        kind.value: {
            "id": model.id,
            "state": model.state.value,
            "device": model.device.value if model.device is not None else None,
        }
        for kind, model in models.items()
    }

    def reject() -> None:
        error = RuntimeError("effective model identity differs")
        error.observed_models = observed
        raise error

    if (
        health.session_id != request.session_id
        or health.direction_id != request.direction_id
        or health.provider_id is not ProviderId.LOCAL
        or health.state is not ProviderState.READY
    ):
        reject()
    if (
        len(models) != len(health.models)
        or set(models) != {ModelKind.ASR, ModelKind.MT, ModelKind.TTS}
        or models[ModelKind.ASR].id != expected_asr_id
        or models[ModelKind.MT].id != expected_mt_id
        or models[ModelKind.TTS].id != "piper-medium"
        or any(model.state is not ModelState.READY for model in models.values())
        or any(
            model.device is None or model.device.value not in {"cpu", "cuda"}
            for model in (models[ModelKind.ASR], models[ModelKind.MT])
        )
        or models[ModelKind.TTS].device is None
        or models[ModelKind.TTS].device.value != "cpu"
    ):
        reject()
    return observed


async def _run_case(
    provider: Any,
    case: dict[str, Any],
    mode: TranslationMode,
    voice_gender: VoiceGender,
    *,
    expected_asr_id: str,
    expected_mt_id: str,
) -> dict[str, Any]:
    request = _request(case["language"], mode, voice_gender)
    event_times: list[tuple[Any, int]] = []

    async def publish(batch: tuple[Any, ...], commit: Any) -> None:
        at = time.monotonic_ns()
        event_times.extend((event, at) for event in batch)
        commit()

    reservation = provider.reserve_session(request, publish)
    error: BaseException | None = None
    opened = False
    post_health_attempted = False
    open_models: dict[str, dict[str, str]] | None = None
    after_models: dict[str, dict[str, str]] | None = None
    try:
        _, health = await reservation.open()
        opened = True
        open_models = _checked_health(health, request, expected_asr_id, expected_mt_id)
        pcm = case["pcm"]
        frames = [
            pcm[offset : offset + FRAME_BYTES].ljust(FRAME_BYTES, b"\0")
            for offset in range(0, len(pcm), FRAME_BYTES)
        ]
        started_ns = time.monotonic_ns()
        stream_id, utterance_id = uuid4(), uuid4()
        for sequence, frame in enumerate(frames):
            await provider.submit_frame(
                ProviderInputFrame(
                    session_id=request.session_id,
                    direction_id=request.direction_id,
                    stream_id=stream_id,
                    utterance_id=utterance_id,
                    sequence=sequence,
                    capture_monotonic_ns=started_ns,
                    sample_rate_hz=16_000,
                    channels=1,
                    sample_format=SampleFormat.S16LE,
                    frame_duration_ms=100,
                    source_language=request.source_language,
                    target_language=request.target_language,
                    mode=request.mode,
                    pcm=frame,
                    end_of_utterance=sequence == len(frames) - 1,
                )
            )
        await asyncio.wait_for(provider.wait_idle(), timeout=90)
        post_health_attempted = True
        after_models = _checked_health(
            await asyncio.wait_for(
                provider.health(request.session_id), timeout=HEALTH_TIMEOUT_SECONDS
            ),
            request,
            expected_asr_id,
            expected_mt_id,
        )
        if after_models != open_models:
            raise RuntimeError("effective model identity differs")
        try:
            result = evaluate_events(
                event_times,
                started_ns=started_ns,
                session_id=request.session_id,
                direction_id=request.direction_id,
                stream_id=stream_id,
                utterance_id=utterance_id,
            )
            result["effective_models_open"] = open_models
            result["effective_models_after"] = after_models
            return result
        except ValueError as observation_error:
            if not isinstance(observation_error, VerifiedProviderDrop):
                observation_error.safe_provider_codes = [
                    event.code.value
                    for event, _ in event_times
                    if isinstance(event, PrivacySafeProviderError)
                ]
                observation_error.outcomes = [
                    event.outcome.value
                    for event, _ in event_times
                    if isinstance(event, ProviderUtteranceFinal)
                ]
                latencies = [
                    event
                    for event, _ in event_times
                    if isinstance(event, ProviderLatency)
                    and event.session_id == request.session_id
                    and event.utterance_id == utterance_id
                ]
                safe_latency = None
                if len(latencies) == 1:
                    candidate = {
                        field: getattr(latencies[0], field)
                        for field in (
                            "asr_final_text_ms",
                            "mt_first_text_ms",
                            "tts_first_audio_ms",
                            "provider_total_ms",
                        )
                    }
                    if all(
                        value is None or (type(value) is int and value >= 0)
                        for value in candidate.values()
                    ):
                        safe_latency = candidate
                observation_error.safe_provider_latency = safe_latency
            raise
    except BaseException as caught:
        error = caught
        if open_models is not None:
            caught.open_effective_models = open_models
        elif not post_health_attempted and hasattr(caught, "observed_models"):
            caught.open_effective_models = caught.observed_models
        if after_models is not None:
            caught.post_attempt_effective_models = after_models
        elif post_health_attempted and hasattr(caught, "observed_models"):
            caught.post_attempt_effective_models = caught.observed_models
        if opened and not post_health_attempted:
            try:
                caught.post_attempt_effective_models = _checked_health(
                    await asyncio.wait_for(
                        provider.health(request.session_id),
                        timeout=HEALTH_TIMEOUT_SECONDS,
                    ),
                    request,
                    expected_asr_id,
                    expected_mt_id,
                )
            except RuntimeError as health_error:
                if hasattr(health_error, "observed_models"):
                    caught.post_attempt_model_error = str(health_error)
                    caught.post_attempt_effective_models = health_error.observed_models
                else:
                    caught.post_attempt_health_error = "RuntimeError"
            except BaseException as health_error:  # noqa: BLE001 - keep original failure
                caught.post_attempt_health_error = type(health_error).__name__
        raise
    finally:
        try:
            receipt = await finish_cleanup(
                reservation.drain(CloseRequestReason.USER_STOP)
            )
            if getattr(receipt, "session_id", None) != request.session_id:
                raise RuntimeError("product session cleanup identity differs")
            if receipt.delivery_error is not None:
                raise RuntimeError("product session delivery failed")
            if isinstance(error, VerifiedProviderDrop) and after_models == open_models:
                error.attested_after_drain = True
        except BaseException as cleanup_error:
            if error is not None:
                error.cleanup_error_type = type(cleanup_error).__name__
                raise error from cleanup_error
            raise


def _gpu_process_mib(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0] == str(pid):
                return int(parts[1])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def _resources(provider: Any) -> dict[str, Any]:
    child = getattr(getattr(provider, "_translator", None), "_process", None)
    child_pid = child.pid if child is not None and child.poll() is None else None
    current_pid = os.getpid()
    return {
        "runner_rss_mib": round(
            psutil.Process(current_pid).memory_info().rss / 1048576, 1
        ),
        "runner_gpu_mib": _gpu_process_mib(current_pid),
        "mt_child_pid": child_pid,
        "mt_child_gpu_mib": _gpu_process_mib(child_pid),
    }


async def _run_arm(
    backend: str,
    cases: list[dict[str, Any]],
    journal: TextIO,
    rows: list[dict[str, Any]],
    mode: TranslationMode,
    voice_gender: VoiceGender,
) -> tuple[bool, bool]:
    previous_asr = os.environ.get("TRANSLATOR_ASR_MODEL_ID")
    previous_mt = os.environ.get("TRANSLATOR_MT_MODEL_ID")
    expected_asr_id, expected_mt_id = ARMS[backend]
    os.environ["TRANSLATOR_ASR_MODEL_ID"] = expected_asr_id
    os.environ["TRANSLATOR_MT_MODEL_ID"] = expected_mt_id
    provider = None
    failed = False
    aborted = False
    next_index = 0
    built_ns = time.monotonic_ns()
    try:
        provider = build_local_provider(
            now_ns=time.monotonic_ns, manifest_path=ROOT / "models/manifest.json"
        )
        write_record(
            journal,
            {
                "type": "arm_start",
                "backend": backend,
                "requested_asr_id": expected_asr_id,
                "requested_mt_id": expected_mt_id,
                "mode": mode.value,
                "build_ms": round((time.monotonic_ns() - built_ns) / 1_000_000, 2),
                "resources": _resources(provider),
            },
        )
        for index, case in enumerate(cases):
            next_index = index + 1
            case_aborted = False
            row = {
                "type": "attempt",
                "backend": backend,
                "mode": mode.value,
                "origin_id": case["origin_id"],
                "condition": case["condition"],
                "language": case["language"],
                "speaker_id": case["speaker_id"],
                "wav_sha256": case["wav_sha256"],
                "voice_gender": voice_gender.value,
                "reference": case["reference"],
                "turbo_frozen_text": case["turbo_text"],
                "critical_labels": case["critical_labels"],
            }
            try:
                row.update(
                    await _run_case(
                        provider,
                        case,
                        mode,
                        voice_gender,
                        expected_asr_id=expected_asr_id,
                        expected_mt_id=expected_mt_id,
                    )
                )
            except Exception as error:  # noqa: BLE001 - retain the failed attempt
                row.update(status="failed", error_type=type(error).__name__)
                if str(error) in SAFE_EVALUATION_REASONS:
                    row["safe_failure_reason"] = str(error)
                for diagnostic in (
                    "open_effective_models",
                    "post_attempt_effective_models",
                    "post_attempt_model_error",
                    "post_attempt_health_error",
                    "cleanup_error_type",
                ):
                    if hasattr(error, diagnostic):
                        row[diagnostic] = getattr(error, diagnostic)
                if isinstance(error, ValueError) and hasattr(
                    error, "safe_provider_codes"
                ):
                    row["safe_provider_codes"] = error.safe_provider_codes
                    row["observed_outcomes"] = error.outcomes
                    row["safe_provider_latency"] = getattr(
                        error, "safe_provider_latency", None
                    )
                failed = True
                case_aborted = not (
                    isinstance(error, VerifiedProviderDrop)
                    and error.attested_after_drain
                )
            row["resources"] = _resources(provider)
            rows.append(row)
            write_record(journal, row)
            if case_aborted:
                aborted = True
                for remaining in cases[index + 1 :]:
                    write_record(
                        journal,
                        {
                            "type": "not_run",
                            "backend": backend,
                            "mode": mode.value,
                            "origin_id": remaining["origin_id"],
                            "condition": remaining["condition"],
                        },
                    )
                break
    except Exception as error:  # noqa: BLE001 - retain build and journal failures
        failed = True
        aborted = True
        write_record(
            journal,
            {
                "type": "arm_error",
                "backend": backend,
                "mode": mode.value,
                "error_type": type(error).__name__,
            },
        )
        for remaining in cases[next_index:]:
            write_record(
                journal,
                {
                    "type": "not_run",
                    "backend": backend,
                    "mode": mode.value,
                    "origin_id": remaining["origin_id"],
                    "condition": remaining["condition"],
                },
            )
    finally:
        if previous_asr is None:
            os.environ.pop("TRANSLATOR_ASR_MODEL_ID", None)
        else:
            os.environ["TRANSLATOR_ASR_MODEL_ID"] = previous_asr
        if previous_mt is None:
            os.environ.pop("TRANSLATOR_MT_MODEL_ID", None)
        else:
            os.environ["TRANSLATOR_MT_MODEL_ID"] = previous_mt
        if provider is not None:
            try:
                await provider.shutdown()
            except Exception as error:  # noqa: BLE001 - retain cleanup failure
                failed = True
                aborted = True
                write_record(
                    journal,
                    {
                        "type": "cleanup_error",
                        "backend": backend,
                        "mode": mode.value,
                        "error_type": type(error).__name__,
                    },
                )
        write_record(
            journal,
            {
                "type": "arm_end",
                "backend": backend,
                "mode": mode.value,
                "status": "failed" if failed else "complete",
                "post_shutdown_resources": _resources(provider)
                if provider is not None
                else None,
            },
        )
    return not aborted, not failed


async def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.mode
    if not isinstance(mode, TranslationMode):
        raise TypeError("product mode is invalid")
    voice_gender = args.voice_gender
    if not isinstance(voice_gender, VoiceGender):
        raise TypeError("product voice gender is invalid")
    cases = load_cases(args.manifest, args.screen, args.turbo)
    model_cache_root = os.environ.get("TRANSLATOR_MODEL_CACHE_ROOT")
    if model_cache_root is None or not Path(model_cache_root).is_dir():
        raise ValueError("an explicit existing evaluation model cache is required")
    if sha256(ROOT / "models/manifest.json") != PRODUCT_MANIFEST_SHA256:
        raise ValueError("product model manifest identity changed")
    if len(cases) != 24 or {case["language"] for case in cases} != {"ru_ru", "en_us"}:
        raise ValueError("frozen product screen is not 24 bidirectional cases")
    if args.case_id is not None:
        selected = [case for case in cases if case["origin_id"] == args.case_id]
        if len(selected) != 1 or args.smoke:
            raise ValueError("diagnostic case must be one frozen origin")
        cases = selected
    elif args.smoke:
        cases = [
            next(case for case in cases if case["language"] == language)
            for language in ("ru_ru", "en_us")
        ]
    order = (
        ("small_nllb_same_code", "nllb", "hy")
        if args.order == "small-nllb-hy"
        else ("hy", "nllb", "small_nllb_same_code")
    )
    rows: list[dict[str, Any]] = []
    with open_journal(args.output) as journal:
        write_record(
            journal,
            {
                "type": "header",
                "schema": "translator.product-audio-pair.v2",
                "scope": "accelerated saved audio; no capture, playback, physical first audible, or English listening",
                "baseline_role": "same_code_ablation_not_original_main",
                "input_selection": "turbo_screened_development_only",
                "voice_observation_scope": "requested profile and generic Piper health only; exact voice not independently observed",
                "screen_sha256": SCREEN_SHA256,
                "manifest_sha256": MANIFEST_SHA256,
                "turbo_sha256": TURBO_SHA256,
                "product_manifest_sha256": PRODUCT_MANIFEST_SHA256,
                "model_cache_root": str(Path(model_cache_root).resolve()),
                "hy_server_sha256": sha256(HY_SERVER) if HY_SERVER.is_file() else None,
                "runner_sha256": sha256(Path(__file__)),
                "runtime_sha256": {
                    name: sha256(ROOT / name)
                    for name in (
                        "sidecar/translator_sidecar/local/runtime.py",
                        "sidecar/translator_sidecar/local/local_provider.py",
                        "sidecar/translator_sidecar/local/asr.py",
                        "sidecar/translator_sidecar/local/mt.py",
                        "sidecar/translator_sidecar/local/hy_mt.py",
                        "sidecar/translator_sidecar/local/tts.py",
                    )
                },
                "source_head": (
                    await asyncio.to_thread(
                        subprocess.check_output,
                        ["git", "rev-parse", "HEAD"],
                        cwd=ROOT,
                        text=True,
                    )
                ).strip(),
                "input_count": len(cases),
                "smoke": args.smoke,
                "diagnostic_case_id": args.case_id,
                "order": order,
                "mode": mode.value,
                "voice_gender": voice_gender.value,
                "input": "mono s16le 16000 Hz 100 ms, submitted as fast as possible",
                "output": "mono s16le 24000 Hz 20 ms",
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            },
        )
        all_complete = True
        for arm_index, backend in enumerate(order):
            can_continue, arm_complete = await _run_arm(
                backend, cases, journal, rows, mode, voice_gender
            )
            all_complete &= arm_complete
            if not can_continue:
                all_complete = False
                for remaining_backend in order[arm_index + 1 :]:
                    for case in cases:
                        write_record(
                            journal,
                            {
                                "type": "not_run",
                                "backend": remaining_backend,
                                "mode": mode.value,
                                "origin_id": case["origin_id"],
                                "condition": case["condition"],
                            },
                        )
                break
        try:
            pairs = pair_rows(rows, mode)
        except (KeyError, ValueError, TypeError) as error:
            all_complete = False
            pairs = []
            write_record(
                journal,
                {
                    "type": "pair_error",
                    "mode": mode.value,
                    "error_type": type(error).__name__,
                },
            )
        complete = (
            all_complete
            and len(pairs) == len(cases)
            and all(pair["status"] == "complete" for pair in pairs)
        )
        write_record(
            journal,
            {
                "type": "terminal",
                "mode": mode.value,
                "status": "complete" if complete else "failed",
                "attempts": len(rows),
                "pairs": pairs,
            },
        )
    return {
        "status": "complete" if complete else "failed",
        "attempts": len(rows),
        "pairs": len(pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--turbo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--voice-gender",
        choices=tuple(gender.value for gender in VoiceGender),
        default=VoiceGender.FEMALE.value,
    )
    parser.add_argument(
        "--case-id", help="one frozen origin for failure diagnosis only"
    )
    parser.add_argument(
        "--order",
        choices=("small-nllb-hy", "hy-nllb-small"),
        default="small-nllb-hy",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in TranslationMode),
        default=TranslationMode.QUALITY_FIRST.value,
    )
    args = parser.parse_args()
    args.mode = TranslationMode(args.mode)
    args.voice_gender = VoiceGender(args.voice_gender)
    try:
        result = asyncio.run(run(args))
    except Exception as error:  # noqa: BLE001 - stdout must remain text-free
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 2
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
