"""Replay the frozen development WAVs through the untouched original local chain.

Run in a separate process with the original checkout's sidecar first in
PYTHONPATH. This measures accelerated provider output, not capture or playback.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib
import io
import json
import os
import shutil
import stat
import subprocess
import time
import wave
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from translator_mdc_asr_run import validate_manifest
from translator_product_pcm import PcmArtifactStore
from translator_sidecar.provider_contract import (
    AudioDirection,
    CloseProviderSession,
    CloseRequestReason,
    Language,
    ModelKind,
    ModelState,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderHealth,
    ProviderId,
    ProviderInputFrame,
    ProviderLatency,
    ProviderSessionClosed,
    ProviderSessionOpened,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SampleFormat,
    SessionCloseReason,
    TranslationMode,
    UtteranceOutcome,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)

FORK_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = FORK_ROOT.with_name("translator")
ORIGINAL_HEAD = "9291e8beafee3e02aaa179178ce460ac9e6c6de2"
ORIGINAL_MANIFEST_SHA256 = (
    "b15d24e98e5116a45b0ee745cb1114deedbaaa3861cee0ed964d52efab2f4069"
)
EVAL_CACHE_ROOT = (
    FORK_ROOT.with_name("translator-eval-cache-20260924") / "runtime-cache"
)
MANIFEST_SHA256 = "bc2d204c31dbe0cba78187975f02a8a4378cc09407573ca4912ba540bf44ca26"
SCREEN_SHA256 = "005db22422ad98c9c70ecd3d01f578be571ea4bebca81398d5fff7cda86214ae"
TURBO_SHA256 = "03ebba9576e14e2fd6adb909f25f806583845a185a7df941d0c64b3a6297ac91"
ASR_ID = "faster-whisper-small"
MT_ID = "nllb-200-distilled-600m-ct2-int8"
VOICE_IDS = (
    "piper-ru-dmitri-medium",
    "piper-en-ryan-medium",
    "piper-ru-irina-medium",
    "piper-en-hfc-female-medium",
)
REQUIRED_IDS = (ASR_ID, MT_ID, *VOICE_IDS)
FRAME_BYTES = 3200
OUTPUT_FRAME_BYTES = 960


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def assert_original_runtime() -> dict[str, str]:
    sidecar = (ORIGINAL_ROOT / "sidecar").resolve()
    identities: dict[str, str] = {}
    for name in (
        "translator_sidecar",
        "translator_sidecar.provider_contract",
        "translator_sidecar.local.runtime",
        "translator_sidecar.local.local_provider",
    ):
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(sidecar):
            raise RuntimeError("original runtime import differs")
        identities[name] = sha256(location)
    head = subprocess.check_output(
        ["git", "-C", str(ORIGINAL_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        head != ORIGINAL_HEAD
        or subprocess.check_output(
            ["git", "-C", str(ORIGINAL_ROOT), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise RuntimeError("original checkout identity differs")
    if sha256(ORIGINAL_ROOT / "models/manifest.json") != ORIGINAL_MANIFEST_SHA256:
        raise RuntimeError("original model manifest differs")
    return identities


def _read_json(path: Path, expected: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("frozen input identity differs")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("frozen input identity differs")
    return json.loads(data)


def _pcm(path: Path, expected: str) -> bytes:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("frozen WAV path is unsafe")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("frozen WAV checksum differs")
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
    manifest_path: Path, screen_path: Path, turbo_path: Path
) -> list[dict[str, Any]]:
    if manifest_path.is_symlink() or any(
        parent.is_symlink() for parent in manifest_path.parents
    ):
        raise ValueError("frozen manifest path is unsafe")
    samples = validate_manifest(manifest_path, MANIFEST_SHA256)
    screen = _read_json(screen_path, SCREEN_SHA256)
    turbo = _read_json(turbo_path, TURBO_SHA256)
    if (
        turbo.get("model_id") != "turbo"
        or turbo.get("manifest_sha256") != MANIFEST_SHA256
        or screen.get("turbo_report_sha256") != TURBO_SHA256
    ):
        raise ValueError("frozen Turbo provenance differs")

    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["origin_id"], row["condition"])
            if key in indexed:
                raise ValueError("frozen selection has duplicates")
            indexed[key] = row
        return indexed

    by_sample, by_turbo, selected = (
        index(samples),
        index(turbo["results"]),
        index(screen["cases"]),
    )
    if not selected:
        raise ValueError("frozen selection is empty")
    cases = []
    for key in selected:
        sample, saved = by_sample.get(key), by_turbo.get(key)
        if sample is None or saved is None:
            raise ValueError("frozen selection is incomplete")
        if (
            key[1] != "clean"
            or sample["language"] not in {"ru_ru", "en_us"}
            or any(
                sample[field] != saved.get(field)
                for field in ("language", "audio_file", "speaker_id", "reference")
            )
            or saved.get("status") != "completed"
            or not saved.get("transcript", "").strip()
        ):
            raise ValueError("frozen case provenance differs")
        relative = Path(sample["audio_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("frozen WAV path escapes corpus")
        cases.append(
            {
                "origin_id": key[0],
                "condition": key[1],
                "language": sample["language"],
                "speaker_id": sample["speaker_id"],
                "wav_sha256": sample["sha256"],
                "reference": sample["reference"],
                "critical_labels": sample.get("critical_labels", []),
                "turbo_text": saved["transcript"],
                "pcm": _pcm(manifest_path.parent / relative, sample["sha256"]),
            }
        )
    return cases


def _cache_relative(model: dict[str, Any]) -> Path:
    model_id = model["id"]
    if model_id == ASR_ID:
        return (
            Path("huggingface/hub/models--Systran--faster-whisper-small/snapshots")
            / model["source"]["revision"]
        )
    if model_id == MT_ID:
        return Path("nllb-200-distilled-600M-ct2-int8")
    if model_id in VOICE_IDS[:2]:
        return Path("piper-voices")
    if model_id in VOICE_IDS[2:]:
        return Path("cache/piper")
    return Path("unused") / model_id


def _source_relative(model: dict[str, Any]) -> Path:
    if model["id"] in VOICE_IDS[2:]:
        return Path("piper")
    return _cache_relative(model)


def prepare_private_manifest(
    original: dict[str, Any],
    output_dir: Path,
    source_cache: Path,
    *,
    required_ids: tuple[str, ...] = REQUIRED_IDS,
) -> tuple[dict[str, Any], Path]:
    """Copy pinned bytes into a private cache; change only path fields."""
    if (
        not output_dir.is_absolute()
        or stat.S_IMODE(output_dir.stat().st_mode) != 0o700
        or any(parent.is_symlink() for parent in output_dir.parents)
        or not source_cache.is_dir()
        or source_cache.is_symlink()
        or source_cache.resolve().is_relative_to(ORIGINAL_ROOT)
    ):
        raise ValueError("private model boundary is unsafe")
    private_root = output_dir / "model-cache"
    private_root.mkdir(mode=0o700)
    models = {model["id"]: model for model in original["models"]}
    if not set(required_ids).issubset(models):
        raise ValueError("original model manifest lacks required models")
    projected = json.loads(json.dumps(original))
    projected["policy"]["staging_path"] = str(private_root / ".staging")
    for model in projected["models"]:
        relative = _cache_relative(model)
        model["cache_path"] = str(private_root / relative)
    preserved = json.loads(json.dumps(projected))
    preserved["policy"]["staging_path"] = original["policy"]["staging_path"]
    for before, after_model in zip(
        original["models"], preserved["models"], strict=True
    ):
        after_model["cache_path"] = before["cache_path"]
    if preserved != original:
        raise ValueError("private model manifest changed beyond paths")
    for model in projected["models"]:
        relative = _cache_relative(model)
        if model["id"] not in required_ids:
            continue
        source_dir, destination_dir = (
            source_cache / _source_relative(model),
            private_root / relative,
        )
        destination_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        for file in model["files"]:
            name = Path(file["path"])
            if name.is_absolute() or len(name.parts) != 1:
                raise ValueError("model cache file path is unsafe")
            source = source_dir / name
            resolved = source.resolve(strict=True)
            if (
                not resolved.is_relative_to(source_cache.resolve())
                or not resolved.is_file()
            ):
                raise ValueError("model cache source escapes evaluation cache")
            if resolved.stat().st_nlink != 1:
                raise ValueError(
                    "model cache source is linked outside evaluation cache"
                )
            target = destination_dir / name
            fd = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as output, resolved.open("rb") as input_stream:
                try:
                    fcntl.ioctl(output.fileno(), 0x40049409, input_stream.fileno())
                except OSError:
                    if (
                        shutil.disk_usage(output_dir).free
                        < file["size_bytes"] + 268_435_456
                    ):
                        raise ValueError(
                            "insufficient space for private model copy"
                        ) from None
                    output.seek(0)
                    output.truncate(0)
                    shutil.copyfileobj(input_stream, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if (
                target.stat().st_size != file["size_bytes"]
                or sha256(target) != file["sha256"]
                or target.stat().st_nlink != 1
            ):
                raise ValueError("model cache checksum differs")
    path = output_dir / "original-model-manifest.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        json.dump(projected, output, ensure_ascii=False)
        output.flush()
        os.fsync(output.fileno())
    return projected, path


def open_journal(path: Path) -> TextIO:
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or stat.S_IMODE(path.parent.stat().st_mode) != 0o700
        or any(parent.is_symlink() for parent in path.parents)
        or path.resolve().is_relative_to(ORIGINAL_ROOT)
        or path.resolve().is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise ValueError("journal requires a private directory outside Git")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    return os.fdopen(fd, "w", encoding="utf-8")


def write_record(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _request(
    language: str, mode: TranslationMode, gender: VoiceGender
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
            language=target, gender=gender, engine=VoiceEngine.PIPER
        ),
        debug_text_enabled=True,
    )


def _checked_health(
    health: Any, request: OpenProviderSession, *, after: bool
) -> dict[str, Any]:
    models = {model.kind: model for model in health.models}
    if (
        health.session_id != request.session_id
        or health.direction_id != request.direction_id
        or health.provider_id is not ProviderId.LOCAL
        or health.state
        not in {ProviderState.READY, ProviderState.DEGRADED, ProviderState.STARTING}
        or (after and health.state is ProviderState.STARTING)
        or len(models) != len(health.models)
        or set(models) != {ModelKind.ASR, ModelKind.MT, ModelKind.TTS}
        or models[ModelKind.ASR].id not in ({"small"} if after else {ASR_ID, "small"})
        or models[ModelKind.MT].id != MT_ID
        or models[ModelKind.TTS].id != "piper-medium"
        or models[ModelKind.ASR].state not in {ModelState.READY, ModelState.NOT_LOADED}
        or (after and models[ModelKind.ASR].state is not ModelState.READY)
        or any(
            models[kind].state is not ModelState.READY
            for kind in (ModelKind.MT, ModelKind.TTS)
        )
        or any(
            model.device is None or model.device.value not in {"cpu", "cuda"}
            for model in models.values()
        )
        or models[ModelKind.TTS].device.value != "cpu"
    ):
        raise RuntimeError("effective original model identity differs")
    return {
        "provider_state": health.state.value,
        "models": {
            kind.value: {
                "id": model.id,
                "state": model.state.value,
                "device": model.device.value,
            }
            for kind, model in models.items()
        },
    }


def evaluate_events(
    event_times: list[tuple[Any, int]],
    *,
    started_ns: int,
    session_id: Any,
    direction_id: AudioDirection,
    stream_id: Any,
    utterance_id: Any,
) -> dict[str, Any]:
    allowed = (
        ProviderTranscriptDelta,
        ProviderTranslationDelta,
        ProviderAudioDelta,
        ProviderLatency,
        ProviderUtteranceFinal,
    )
    session_types = (ProviderSessionOpened, ProviderHealth)
    events = [event for event, _ in event_times]
    if (
        not events
        or any(type(at) is not int or at < 0 for _, at in event_times)
        or any(
            current_at < previous_at
            for (_, previous_at), (_, current_at) in pairwise(event_times)
        )
        or any(isinstance(event, PrivacySafeProviderError) for event in events)
        or any(not isinstance(event, (*allowed, *session_types)) for event in events)
        or any(
            type(getattr(event, "event_sequence", None)) is not int
            or event.event_sequence < 0
            for event in events
        )
        or any(
            getattr(event, "session_id", None) != session_id
            or getattr(event, "direction_id", None) != direction_id
            for event in events
        )
        or any(
            current.event_sequence <= previous.event_sequence
            for previous, current in pairwise(events)
        )
    ):
        raise ValueError("original provider event time or order differs")
    first_utterance = next(
        (index for index, event in enumerate(events) if isinstance(event, allowed)),
        len(events),
    )
    event_times = event_times[first_utterance:]
    events = events[first_utterance:]
    if any(isinstance(event, session_types) for event in events) or any(
        at < started_ns for _, at in event_times
    ):
        raise ValueError("original provider event time or order differs")
    if any(
        event.session_id != session_id
        or event.direction_id != direction_id
        or event.stream_id != stream_id
        or event.utterance_id != utterance_id
        for event in events
    ) or any(
        current.event_sequence <= previous.event_sequence
        for previous, current in pairwise(events)
    ):
        raise ValueError("original provider event identity or order differs")
    final = [event for event in events if isinstance(event, ProviderUtteranceFinal)]
    asr = [
        event.text
        for event in events
        if isinstance(event, ProviderTranscriptDelta) and event.is_final
    ]
    mt = [
        event.text
        for event in events
        if isinstance(event, ProviderTranslationDelta) and event.is_final
    ]
    audio = [
        (event, at)
        for event, at in event_times
        if isinstance(event, ProviderAudioDelta)
    ]
    latency = [event for event in events if isinstance(event, ProviderLatency)]
    if (
        len(final) != 1
        or final[0].outcome is not UtteranceOutcome.COMPLETED
        or len(asr) != 1
        or not asr[0].strip()
        or len(mt) != 1
        or not mt[0].strip()
        or not audio
        or len(audio) > 1500
        or len(latency) != 1
        or events[-2:] != [latency[0], final[0]]
        or final[0].final_audio_sequence != len(audio) - 1
        or [frame.sequence for frame, _ in audio] != list(range(len(audio)))
    ):
        raise ValueError("original provider did not complete text and audio")
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
            raise ValueError("original provider PCM format differs")
        digest.update(frame.pcm)
        nonzero |= any(frame.pcm)
    if not nonzero:
        raise ValueError("original provider emitted silent PCM")
    return {
        "status": "completed",
        "asr_text": asr[0],
        "mt_text": mt[0],
        "first_pcm_ms": round((audio[0][1] - started_ns) / 1_000_000, 2),
        "pcm_duration_ms": len(audio) * 20,
        "pcm_sha256": digest.hexdigest(),
        "provider_latency": latency[0].model_dump(mode="json"),
    }


async def run_case(
    provider: Any,
    case: dict[str, Any],
    mode: TranslationMode,
    gender: VoiceGender,
    *,
    capture_audio: PcmArtifactStore | None = None,
) -> dict[str, Any]:
    request = _request(case["language"], mode, gender)
    event_times: list[tuple[Any, int]] = []

    async def publish(batch: tuple[Any, ...], commit: Any) -> None:
        at = time.monotonic_ns()
        event_times.extend((event, at) for event in batch)
        commit()

    opened = False
    original_error: BaseException | None = None
    result: dict[str, Any] | None = None
    count_before_close = 0
    try:
        opening, initial = await provider.open_session(request, publish)
        opened = True
        if opening.session_id != request.session_id:
            raise RuntimeError("original session open identity differs")
        before = _checked_health(initial, request, after=False)
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
                    mode=mode,
                    pcm=frame,
                    end_of_utterance=sequence == len(frames) - 1,
                )
            )
        await asyncio.wait_for(provider.wait_idle(), timeout=90)
        after = _checked_health(
            await asyncio.wait_for(provider.health(request.session_id), 2),
            request,
            after=True,
        )
        result = evaluate_events(
            event_times,
            started_ns=started_ns,
            session_id=request.session_id,
            direction_id=request.direction_id,
            stream_id=stream_id,
            utterance_id=utterance_id,
        )
        result["effective_models_open"] = before
        result["effective_models_after"] = after
    except Exception as error:  # noqa: BLE001 - cleanup still runs for failed attempts
        original_error = error
    finally:
        if opened:
            count_before_close = len(event_times)
            try:
                await asyncio.wait_for(
                    provider.close_session(
                        CloseProviderSession(
                            session_id=request.session_id,
                            reason=CloseRequestReason.USER_STOP,
                        )
                    ),
                    timeout=5,
                )
            except Exception as close_error:
                try:
                    await asyncio.wait_for(
                        provider.wait_publications(request.session_id), timeout=5
                    )
                except Exception as drain_error:  # noqa: BLE001 - report both failures
                    cleanup_error = RuntimeError("original session close failed")
                    cleanup_error.drain_error_type = type(drain_error).__name__
                    raise cleanup_error from close_error
                raise RuntimeError("original session close failed") from close_error
            else:
                await asyncio.wait_for(
                    provider.wait_publications(request.session_id), timeout=5
                )
    if original_error is not None:
        raise original_error
    closed = [
        event for event, _ in event_times if isinstance(event, ProviderSessionClosed)
    ]
    if (
        len(closed) != 1
        or len(event_times) != count_before_close + 1
        or closed[0].session_id != request.session_id
        or closed[0].direction_id != request.direction_id
        or closed[0].reason is not SessionCloseReason.USER_STOP
        or not isinstance(event_times[-1][0], ProviderSessionClosed)
        or closed[0].event_sequence <= event_times[-2][0].event_sequence
        or any(
            current_at < previous_at
            for (_, previous_at), (_, current_at) in pairwise(event_times)
        )
    ):
        raise RuntimeError("original session close publication differs")
    assert result is not None
    if capture_audio is not None:
        bindings = {
            "session_id": str(request.session_id),
            "direction_id": request.direction_id.value,
            "stream_id": str(stream_id),
            "utterance_id": str(utterance_id),
            "input_wav_sha256": case["wav_sha256"],
            "target_language": request.target_language.value,
            "requested_voice": request.voice_profile.model_dump(
                mode="json", exclude_none=True
            ),
            "effective_models_open": before,
            "effective_models_after": after,
        }
        output_pcm = b"".join(
            event.pcm
            for event, _ in event_times
            if isinstance(event, ProviderAudioDelta)
        )
        result["pcm_artifact"] = {
            **capture_audio.write_pcm(
                output_pcm, result["pcm_sha256"], f"{utterance_id.hex}.wav"
            ),
            **bindings,
        }
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    identities = assert_original_runtime()
    if args.cache.resolve() != EVAL_CACHE_ROOT.resolve():
        raise ValueError("original baseline requires the pinned evaluation cache")
    if args.output.parent.resolve().is_relative_to(args.cache.resolve()):
        raise ValueError("journal must not alter evaluation cache")
    cases = load_cases(args.manifest, args.screen, args.turbo)
    if len(cases) != 24 or {item["language"] for item in cases} != {"ru_ru", "en_us"}:
        raise ValueError("frozen screen is not 24 bidirectional cases")
    if args.smoke:
        cases = [
            next(item for item in cases if item["language"] == language)
            for language in ("ru_ru", "en_us")
        ]
    if args.case_id is not None:
        cases = [item for item in cases if item["origin_id"] == args.case_id]
        if len(cases) != 1 or args.smoke:
            raise ValueError("diagnostic case is not a unique frozen origin")
    original_manifest = _read_json(
        ORIGINAL_ROOT / "models/manifest.json", ORIGINAL_MANIFEST_SHA256
    )
    from translator_sidecar.local.runtime import build_local_provider

    rows: list[dict[str, Any]] = []
    capture_directory = getattr(args, "capture_audio", None)
    with (
        open_journal(args.output) as journal,
        (
            PcmArtifactStore(
                capture_directory, forbidden_roots=(ORIGINAL_ROOT, FORK_ROOT)
            )
            if capture_directory is not None
            else nullcontext()
        ) as capture_audio,
    ):
        write_record(
            journal,
            {
                "type": "header",
                "schema": "translator.original-main-baseline.v1",
                "scope": "accelerated saved audio; no capture or playback",
                "source_head": ORIGINAL_HEAD,
                "runtime_sha256": identities,
                "original_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
                "runner_sha256": sha256(Path(__file__)),
                "manifest_sha256": MANIFEST_SHA256,
                "screen_sha256": SCREEN_SHA256,
                "turbo_sha256": TURBO_SHA256,
                "input_count": len(cases),
                "mode": args.mode.value,
                "voice_gender": args.voice_gender.value,
                "smoke": args.smoke,
                "diagnostic_case_id": args.case_id,
                "input": "mono s16le 16000 Hz 100 ms, submitted as fast as possible",
                "output": "mono s16le 24000 Hz 20 ms",
                **(
                    {
                        "capture_audio_enabled": True,
                        "pcm_helper_sha256": sha256(
                            Path(__file__).with_name("translator_product_pcm.py")
                        ),
                    }
                    if capture_audio is not None
                    else {}
                ),
            },
        )
        provider = None
        next_index = 0
        failed = False
        try:
            _, manifest_path = prepare_private_manifest(
                original_manifest, args.output.parent, args.cache
            )
            write_record(
                journal,
                {
                    "type": "model_projection",
                    "manifest_sha256": sha256(manifest_path),
                    "cache_root": str(args.output.parent / "model-cache"),
                    "copied_model_ids": list(REQUIRED_IDS),
                    "only_cache_and_staging_paths_changed": True,
                },
            )
            os.environ["TRANSLATOR_ASR_MODEL_ID"] = ASR_ID
            os.environ.pop("TRANSLATOR_MT_MODEL_ID", None)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            started = time.monotonic_ns()
            provider = build_local_provider(
                now_ns=time.monotonic_ns, manifest_path=manifest_path
            )
            write_record(
                journal,
                {
                    "type": "arm_start",
                    "backend": "original_main_small_nllb_piper",
                    "build_ms": round((time.monotonic_ns() - started) / 1_000_000, 2),
                },
            )
            for index, case in enumerate(cases):
                next_index = index + 1
                row = {
                    "type": "attempt",
                    "backend": "original_main_small_nllb_piper",
                    "mode": args.mode.value,
                    "origin_id": case["origin_id"],
                    "condition": case["condition"],
                    "language": case["language"],
                    "speaker_id": case["speaker_id"],
                    "wav_sha256": case["wav_sha256"],
                    "voice_gender": args.voice_gender.value,
                    "reference": case["reference"],
                    "turbo_frozen_text": case["turbo_text"],
                    "critical_labels": case["critical_labels"],
                }
                try:
                    row.update(
                        await run_case(
                            provider,
                            case,
                            args.mode,
                            args.voice_gender,
                            **(
                                {"capture_audio": capture_audio}
                                if capture_audio is not None
                                else {}
                            ),
                        )
                    )
                except Exception as error:  # noqa: BLE001 - preserve first failed case
                    row.update(status="failed", error_type=type(error).__name__)
                    if isinstance(error, ValueError) and str(error).startswith(
                        "original provider"
                    ):
                        row["safe_failure_reason"] = str(error)
                    failed = True
                rows.append(row)
                write_record(journal, row)
                if failed:
                    break
        except Exception as error:  # noqa: BLE001 - preserve setup failure
            failed = True
            write_record(
                journal, {"type": "arm_error", "error_type": type(error).__name__}
            )
        finally:
            if provider is not None:
                try:
                    await asyncio.wait_for(provider.shutdown(), timeout=10)
                except Exception as error:  # noqa: BLE001 - preserve cleanup failure
                    failed = True
                    write_record(
                        journal,
                        {"type": "cleanup_error", "error_type": type(error).__name__},
                    )
            for item in cases[next_index:]:
                write_record(
                    journal,
                    {
                        "type": "not_run",
                        "origin_id": item["origin_id"],
                        "condition": item["condition"],
                    },
                )
            complete = not failed and len(rows) == len(cases)
            write_record(
                journal,
                {
                    "type": "terminal",
                    "status": "complete" if complete else "failed",
                    "attempts": len(rows),
                    "not_run": len(cases) - len(rows),
                },
            )
    return {
        "status": "complete" if complete else "failed",
        "attempts": len(rows),
        "not_run": len(cases) - len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--turbo", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--capture-audio",
        type=Path,
        help="existing private directory for verified output WAVs",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in TranslationMode),
        default=TranslationMode.QUALITY_FIRST.value,
    )
    parser.add_argument(
        "--voice-gender",
        choices=tuple(gender.value for gender in VoiceGender),
        default=VoiceGender.FEMALE.value,
    )
    args = parser.parse_args()
    args.mode = TranslationMode(args.mode)
    args.voice_gender = VoiceGender(args.voice_gender)
    try:
        result = asyncio.run(run(args))
    except Exception as error:  # noqa: BLE001 - stdout carries only a safe type
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 2
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
