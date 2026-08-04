"""Podcast-driven local streaming quality diagnostics."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterable
from contextlib import contextmanager
import datetime as dt
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

from jiwer import wer

from translator_sidecar.benchmark.model_matrix import (
    candidate_by_id,
    candidate_report,
    default_asr_candidate_ids,
    default_tts_candidate_ids,
    registry_report,
)
from translator_sidecar.local.runtime import build_local_provider
from translator_sidecar.provider_contract import (
    AudioDirection,
    CloseProviderSession,
    CloseRequestReason,
    Language,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderId,
    ProviderInputFrame,
    ProviderLatency,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SampleFormat,
    TranslationMode,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT = _ROOT / "docs" / "benchmarks" / "podcast-quality-debug.json"
_DEFAULT_WORK_DIR = _ROOT / "output" / "podcast-quality"
_DEFAULT_RU_YOUTUBE = "ytsearch1:русский подкаст интервью технологии"
_DEFAULT_EN_YOUTUBE = "ytsearch1:english podcast interview technology"
_SAMPLE_RATE_HZ = 16_000
_FRAME_DURATION_MS = 100
_FRAME_BYTES = _SAMPLE_RATE_HZ * 2 * _FRAME_DURATION_MS // 1_000
_BYTES_PER_MS = _SAMPLE_RATE_HZ * 2 // 1_000
_DIRECTIONS = {
    "ru_to_en": (
        AudioDirection.MICROPHONE,
        Language.RU,
        Language.EN,
    ),
    "en_to_ru": (
        AudioDirection.SPEAKER,
        Language.EN,
        Language.RU,
    ),
}


class PodcastQualityError(RuntimeError):
    """Podcast quality diagnostics could not complete."""


class _Collector:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.batches: list[tuple[Any, ...]] = []

    async def publish(self, batch: tuple[Any, ...], commit: Any) -> None:
        self.batches.append(batch)
        self.events.extend(batch)
        commit()
        await asyncio.sleep(0)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _run_command(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise PodcastQualityError(stderr or f"command failed: {command[0]}")


def _find_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise PodcastQualityError(f"{name} is unavailable")
    return path


def _download_youtube_audio(
    source: str,
    *,
    start_seconds: int,
    duration_seconds: int,
    work_dir: Path,
) -> Path:
    yt_dlp = _find_binary("yt-dlp")
    output_template = work_dir / "youtube-source.%(ext)s"
    end_seconds = start_seconds + duration_seconds
    command = [
        yt_dlp,
        "--no-playlist",
        "--force-overwrites",
        "--force-keyframes-at-cuts",
        "--download-sections",
        f"*{start_seconds}-{end_seconds}",
        "-f",
        "bestaudio/best",
        "-o",
        str(output_template),
        source,
    ]
    _run_command(command)
    matches = [
        path
        for path in work_dir.glob("youtube-source.*")
        if not path.name.endswith(".part")
    ]
    if not matches:
        raise PodcastQualityError("yt-dlp produced no audio file")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _convert_to_pcm(input_path: Path, output_path: Path) -> bytes:
    ffmpeg = _find_binary("ffmpeg")
    _run_command(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE_HZ),
            "-f",
            "s16le",
            str(output_path),
        ]
    )
    data = output_path.read_bytes()
    if len(data) < _FRAME_BYTES:
        raise PodcastQualityError(f"PCM audio is too short: {input_path}")
    return data


def _load_audio_pcm(
    *,
    youtube_source: str | None,
    audio_path: Path | None,
    start_seconds: int,
    duration_seconds: int,
    work_dir: Path,
) -> tuple[bytes, dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if audio_path is not None:
        source_path = audio_path
        source_kind = "local_audio"
        source_value = str(audio_path)
    else:
        if youtube_source is None:
            raise PodcastQualityError("YouTube source is required")
        source_path = _download_youtube_audio(
            youtube_source,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            work_dir=work_dir,
        )
        source_kind = "youtube"
        source_value = youtube_source
    pcm = _convert_to_pcm(source_path, work_dir / "source.s16le")
    return (
        pcm,
        {
            "kind": source_kind,
            "source": source_value,
            "materialized_path": str(source_path),
            "pcm_duration_ms": len(pcm) // _BYTES_PER_MS,
        },
    )


def _segments(
    pcm: bytes,
    *,
    segment_ms: int,
    max_segments: int,
) -> list[bytes]:
    segment_bytes = max(_FRAME_BYTES, segment_ms * _BYTES_PER_MS)
    segment_bytes -= segment_bytes % _FRAME_BYTES
    limit = min(len(pcm), segment_bytes * max_segments)
    return [
        pcm[offset : offset + segment_bytes]
        for offset in range(0, limit, segment_bytes)
        if len(pcm[offset : offset + segment_bytes]) >= _FRAME_BYTES
    ]


def _frames(pcm: bytes) -> list[bytes]:
    frames = [
        pcm[offset : offset + _FRAME_BYTES]
        for offset in range(0, len(pcm), _FRAME_BYTES)
    ]
    if not frames:
        raise PodcastQualityError("segment contains no PCM frames")
    frames[-1] = frames[-1].ljust(_FRAME_BYTES, b"\0")
    return frames


def _pcm_format() -> PcmFormat:
    return PcmFormat(
        sample_rate_hz=_SAMPLE_RATE_HZ,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=_FRAME_DURATION_MS,
    )


def _voice_profile(language: Language, gender: VoiceGender) -> VoiceProfile:
    return VoiceProfile(
        language=language,
        gender=gender,
        engine=VoiceEngine.PIPER,
    )


def _request(
    direction_name: str,
    *,
    mode: TranslationMode,
    voice_gender: VoiceGender,
) -> OpenProviderSession:
    direction_id, source_language, target_language = _DIRECTIONS[direction_name]
    pcm_format = _pcm_format()
    return OpenProviderSession(
        session_id=uuid4(),
        provider_id=ProviderId.LOCAL,
        direction_id=direction_id,
        source_language=source_language,
        target_language=target_language,
        mode=mode,
        requested_input_format=pcm_format,
        requested_output_format=pcm_format,
        voice_profile=_voice_profile(target_language, voice_gender),
        debug_text_enabled=True,
    )


def _last_event(events: Iterable[Any], event_type: type[Any]) -> Any | None:
    result = None
    for event in events:
        if isinstance(event, event_type):
            result = event
    return result


def _output_audio(events: Iterable[Any]) -> bytes:
    return b"".join(
        event.pcm for event in events if isinstance(event, ProviderAudioDelta)
    )


def _safe_wer(reference: str, hypothesis: str) -> float | None:
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    if not reference or not hypothesis:
        return None
    try:
        return float(wer(reference, hypothesis))
    except Exception:
        return None


def _release_provider_models(provider: Any) -> None:
    asr = getattr(provider, "_asr", None)
    release = getattr(asr, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass
    translator = getattr(provider, "_translator", None)
    backend = getattr(translator, "_translator", None)
    unload_model = getattr(backend, "unload_model", None)
    if callable(unload_model):
        try:
            unload_model()
        except Exception:
            pass
    gc.collect()


async def _transcribe_tts_output(
    provider: Any,
    pcm: bytes,
    *,
    target_language: Language,
    mode: TranslationMode,
) -> str | None:
    if not pcm:
        return None
    asr = getattr(provider, "_asr", None)
    transcribe = getattr(asr, "transcribe", None)
    if not callable(transcribe):
        return None
    try:
        return await asyncio.to_thread(
            transcribe,
            pcm,
            language=target_language,
            mode=mode,
        )
    except Exception:
        return None


async def _run_segment(
    provider: Any,
    direction_name: str,
    pcm: bytes,
    *,
    segment_index: int,
    mode: TranslationMode,
    voice_gender: VoiceGender,
) -> dict[str, Any]:
    request = _request(
        direction_name,
        mode=mode,
        voice_gender=voice_gender,
    )
    collector = _Collector()
    started_ns = time.monotonic_ns()
    stream_id = uuid4()
    utterance_id = uuid4()
    await provider.open_session(request, collector.publish)
    frames = _frames(pcm)
    for sequence, frame in enumerate(frames):
        await provider.submit_frame(
            ProviderInputFrame(
                session_id=request.session_id,
                direction_id=request.direction_id,
                stream_id=stream_id,
                utterance_id=utterance_id,
                sequence=sequence,
                capture_monotonic_ns=started_ns,
                sample_rate_hz=_SAMPLE_RATE_HZ,
                channels=1,
                sample_format=SampleFormat.S16LE,
                frame_duration_ms=_FRAME_DURATION_MS,
                source_language=request.source_language,
                target_language=request.target_language,
                mode=mode,
                pcm=frame,
                end_of_utterance=sequence == len(frames) - 1,
            )
        )
    try:
        await provider.wait_idle()
        transcript = _last_event(collector.events, ProviderTranscriptDelta)
        translation = _last_event(collector.events, ProviderTranslationDelta)
        latency = _last_event(collector.events, ProviderLatency)
        error = _last_event(collector.events, PrivacySafeProviderError)
        final = _last_event(collector.events, ProviderUtteranceFinal)
        output_pcm = _output_audio(collector.events)
        synthesized_transcript = await _transcribe_tts_output(
            provider,
            output_pcm,
            target_language=request.target_language,
            mode=mode,
        )
        translation_text = translation.text if translation is not None else ""
        synthesized_wer = _safe_wer(
            translation_text,
            synthesized_transcript or "",
        )
        return {
            "direction": direction_name,
            "segment_index": segment_index,
            "source_duration_ms": len(pcm) // _BYTES_PER_MS,
            "frame_count": len(frames),
            "outcome": final.outcome.value if final is not None else None,
            "safe_error_code": error.code.value if error is not None else None,
            "transcript_chars": (
                len(transcript.text) if transcript is not None else 0
            ),
            "transcript_words": (
                len(transcript.text.split()) if transcript is not None else 0
            ),
            "translation_chars": len(translation_text),
            "translation_words": len(translation_text.split()),
            "audio_output_ms": (
                sum(
                    event.frame_duration_ms
                    for event in collector.events
                    if isinstance(event, ProviderAudioDelta)
                )
            ),
            "audio_output_frames": len(output_pcm) // _FRAME_BYTES,
            "tts_asr_transcript_chars": (
                len(synthesized_transcript or "")
            ),
            "tts_asr_wer": synthesized_wer,
            "output_to_source_duration_ratio": (
                (len(output_pcm) // _FRAME_BYTES * _FRAME_DURATION_MS)
                / max(1, len(pcm) // _BYTES_PER_MS)
            ),
            "latency": (
                latency.model_dump(mode="json") if latency is not None else None
            ),
            "wall_total_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
        }
    finally:
        await provider.close_session(
            CloseProviderSession(
                session_id=request.session_id,
                reason=CloseRequestReason.USER_STOP,
            )
        )
        await provider.wait_publications(request.session_id)


@contextmanager
def _asr_model_env(model_id: str):
    previous = os.environ.get("TRANSLATOR_ASR_MODEL_ID")
    os.environ["TRANSLATOR_ASR_MODEL_ID"] = model_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TRANSLATOR_ASR_MODEL_ID", None)
        else:
            os.environ["TRANSLATOR_ASR_MODEL_ID"] = previous


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    drops = [
        result
        for result in results
        if result["outcome"] != "completed" or result["safe_error_code"]
    ]
    first_audio = [
        result["latency"]["tts_first_audio_ms"]
        for result in results
        if result.get("latency")
        and result["latency"].get("tts_first_audio_ms") is not None
    ]
    total = [
        result["latency"]["provider_total_ms"]
        for result in results
        if result.get("latency")
        and result["latency"].get("provider_total_ms") is not None
    ]
    tts_wer = [
        result["tts_asr_wer"]
        for result in results
        if result.get("tts_asr_wer") is not None
    ]
    return {
        "segment_count": len(results),
        "completed_count": len(results) - len(drops),
        "drop_count": len(drops),
        "drop_rate": len(drops) / max(1, len(results)),
        "tts_first_audio_p50_ms": _percentile(first_audio, 0.50),
        "tts_first_audio_p95_ms": _percentile(first_audio, 0.95),
        "provider_total_p50_ms": _percentile(total, 0.50),
        "provider_total_p95_ms": _percentile(total, 0.95),
        "tts_asr_wer_p50": _percentile(tts_wer, 0.50),
        "tts_asr_wer_p95": _percentile(tts_wer, 0.95),
        "median_output_to_source_duration_ratio": _percentile(
            [
                result["output_to_source_duration_ratio"]
                for result in results
            ],
            0.50,
        ),
    }


async def _run_model(
    model_id: str,
    segment_sets: dict[str, list[bytes]],
    *,
    mode: TranslationMode,
    voice_gender: VoiceGender,
) -> dict[str, Any]:
    try:
        candidate = candidate_by_id(model_id, role="asr")
    except KeyError:
        return {
            "model_id": model_id,
            "asr_model_id": model_id,
            "tts_model_id": "piper-medium",
            "status": "skipped",
            "skip_reason": "unknown_asr_candidate",
            "mode": mode.value,
            "voice_gender": voice_gender.value,
            "candidate": {
                "id": model_id,
                "role": "asr",
                "runtime": "unknown",
            },
            "summary": _summarize([]),
            "segments": [],
        }
    if candidate.runtime != "local_provider":
        return {
            "model_id": model_id,
            "asr_model_id": model_id,
            "tts_model_id": "piper-medium",
            "status": "skipped",
            "skip_reason": "asr_candidate_not_supported_by_local_provider",
            "mode": mode.value,
            "voice_gender": voice_gender.value,
            "candidate": candidate.to_report(),
            "summary": _summarize([]),
            "segments": [],
        }
    with _asr_model_env(model_id):
        provider = build_local_provider(now_ns=time.monotonic_ns)
    results: list[dict[str, Any]] = []
    try:
        for direction_name, segments in segment_sets.items():
            for segment_index, segment in enumerate(segments):
                results.append(
                    await _run_segment(
                        provider,
                        direction_name,
                        segment,
                        segment_index=segment_index,
                        mode=mode,
                        voice_gender=voice_gender,
                    )
                )
    finally:
        await provider.shutdown()
        _release_provider_models(provider)
    return {
        "model_id": model_id,
        "asr_model_id": model_id,
        "tts_model_id": "piper-medium",
        "status": "completed",
        "candidate": candidate.to_report(),
        "mode": mode.value,
        "voice_gender": voice_gender.value,
        "summary": _summarize(results),
        "segments": results,
    }


def _parse_model_ids(values: list[str]) -> list[str]:
    model_ids: list[str] = []
    for value in values:
        model_ids.extend(
            item.strip() for item in value.split(",") if item.strip()
        )
    return model_ids or default_asr_candidate_ids()


def _parse_tts_model_ids(values: list[str]) -> list[str]:
    model_ids: list[str] = []
    for value in values:
        model_ids.extend(
            item.strip() for item in value.split(",") if item.strip()
        )
    return model_ids or default_tts_candidate_ids()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local streaming translation quality diagnostics on podcast audio.",
    )
    parser.add_argument("--ru-youtube", default=_DEFAULT_RU_YOUTUBE)
    parser.add_argument("--en-youtube", default=_DEFAULT_EN_YOUTUBE)
    parser.add_argument("--ru-audio", type=Path)
    parser.add_argument("--en-audio", type=Path)
    parser.add_argument("--start-seconds", type=int, default=60)
    parser.add_argument("--duration-seconds", type=int, default=72)
    parser.add_argument("--segment-ms", type=int, default=24_000)
    parser.add_argument("--max-segments", type=int, default=2)
    parser.add_argument(
        "--asr-model",
        action="append",
        default=[],
        help="ASR model id; repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--tts-model",
        action="append",
        default=[],
        help=(
            "TTS candidate id for report metadata; repeat or pass "
            "comma-separated values. The current provider runtime still uses "
            "piper-medium until alternate TTS adapters are enabled."
        ),
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print the approved ASR/TTS benchmark candidate registry and exit.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TranslationMode],
        default=TranslationMode.STREAMING_FIRST.value,
    )
    parser.add_argument(
        "--voice-gender",
        choices=[gender.value for gender in VoiceGender],
        default=VoiceGender.MALE.value,
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=_DEFAULT_WORK_DIR)
    return parser


async def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.work_dir / dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    ru_pcm, ru_source = _load_audio_pcm(
        youtube_source=args.ru_youtube,
        audio_path=args.ru_audio,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        work_dir=run_dir / "ru",
    )
    en_pcm, en_source = _load_audio_pcm(
        youtube_source=args.en_youtube,
        audio_path=args.en_audio,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        work_dir=run_dir / "en",
    )
    segment_sets = {
        "ru_to_en": _segments(
            ru_pcm,
            segment_ms=args.segment_ms,
            max_segments=args.max_segments,
        ),
        "en_to_ru": _segments(
            en_pcm,
            segment_ms=args.segment_ms,
            max_segments=args.max_segments,
        ),
    }
    if any(not segments for segments in segment_sets.values()):
        raise PodcastQualityError("one of the podcast inputs produced no segments")
    mode = TranslationMode(args.mode)
    voice_gender = VoiceGender(args.voice_gender)
    asr_model_ids = _parse_model_ids(args.asr_model)
    tts_model_ids = _parse_tts_model_ids(args.tts_model)
    model_reports = []
    for model_id in asr_model_ids:
        model_reports.append(
            await _run_model(
                model_id,
                segment_sets,
                mode=mode,
                voice_gender=voice_gender,
            )
        )
    report = {
        "schema_version": "translator.podcast-quality-debug.v1",
        "generated_at": _utc_now(),
        "inputs": {
            "ru_to_en": ru_source,
            "en_to_ru": en_source,
            "segment_ms": args.segment_ms,
            "max_segments": args.max_segments,
            "work_dir": str(run_dir),
            "selected_tts_models": tts_model_ids,
        },
        "candidate_matrix": {
            "asr": candidate_report(
                asr_model_ids,
                role="asr",
                include_unknown=True,
            ),
            "tts": candidate_report(
                tts_model_ids,
                role="tts",
                include_unknown=True,
            ),
        },
        "models": model_reports,
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
    if args.list_candidates:
        print(json.dumps(registry_report(), ensure_ascii=False, indent=2))
        return 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(run(args))
    except PodcastQualityError as error:
        print(f"podcast quality diagnostics failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report["models"], ensure_ascii=False, indent=2))
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
