from __future__ import annotations

import asyncio

from translator_sidecar.benchmark import podcast_quality
from translator_sidecar.provider_contract import TranslationMode, VoiceGender


def test_podcast_segments_align_to_provider_frame_and_max_count() -> None:
    pcm = b"x" * podcast_quality._BYTES_PER_MS * 50_000

    segments = podcast_quality._segments(
        pcm,
        segment_ms=24_050,
        max_segments=2,
    )

    assert len(segments) == 2
    assert all(len(segment) % podcast_quality._FRAME_BYTES == 0 for segment in segments)
    assert [len(segment) for segment in segments] == [
        podcast_quality._BYTES_PER_MS * 24_000,
        podcast_quality._BYTES_PER_MS * 24_000,
    ]


def test_podcast_model_parser_accepts_repeated_and_comma_separated_values() -> None:
    assert podcast_quality._parse_model_ids(
        ["faster-whisper-small,faster-whisper-large-v3", "custom"]
    ) == [
        "faster-whisper-small",
        "faster-whisper-large-v3",
        "custom",
    ]


def test_podcast_model_parser_defaults_to_quality_matrix() -> None:
    assert podcast_quality._parse_model_ids([])[:4] == [
        "faster-whisper-small",
        "faster-whisper-large-v3",
        "faster-whisper-large-v3-turbo-ct2",
        "gigaam-v3-e2e-rnnt",
    ]


def test_podcast_tts_parser_accepts_repeated_and_comma_separated_values() -> None:
    assert podcast_quality._parse_tts_model_ids(
        ["piper-medium,kokoro-82m", "qwen3-tts-0.6b-customvoice"]
    ) == [
        "piper-medium",
        "kokoro-82m",
        "qwen3-tts-0.6b-customvoice",
    ]


def test_podcast_skips_asr_candidate_without_local_provider_runtime() -> None:
    report = asyncio.run(
        podcast_quality._run_model(
            "qwen3-asr-0.6b-hf",
            {"ru_to_en": [b"\0" * podcast_quality._FRAME_BYTES]},
            mode=TranslationMode.STREAMING_FIRST,
            voice_gender=VoiceGender.MALE,
        )
    )

    assert report["status"] == "skipped"
    assert report["skip_reason"] == "asr_candidate_not_supported_by_local_provider"
    assert report["candidate"]["id"] == "qwen3-asr-0.6b-hf"


def test_podcast_summary_reports_drop_latency_and_tts_wer() -> None:
    summary = podcast_quality._summarize(
        [
            {
                "outcome": "completed",
                "safe_error_code": None,
                "latency": {
                    "tts_first_audio_ms": 100,
                    "provider_total_ms": 500,
                },
                "tts_asr_wer": 0.1,
                "output_to_source_duration_ratio": 0.5,
            },
            {
                "outcome": "dropped",
                "safe_error_code": "queue_overflow",
                "latency": {
                    "tts_first_audio_ms": None,
                    "provider_total_ms": 900,
                },
                "tts_asr_wer": None,
                "output_to_source_duration_ratio": 0.0,
            },
        ]
    )

    assert summary["segment_count"] == 2
    assert summary["completed_count"] == 1
    assert summary["drop_rate"] == 0.5
    assert summary["tts_first_audio_p95_ms"] == 100.0
    assert summary["provider_total_p95_ms"] == 900.0
    assert summary["tts_asr_wer_p50"] == 0.1
