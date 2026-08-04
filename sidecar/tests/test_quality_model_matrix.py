from __future__ import annotations

from translator_sidecar.benchmark.model_matrix import (
    asr_candidate_ids,
    candidate_report,
    default_asr_candidate_ids,
    default_executable_asr_candidate_ids,
    default_tts_candidate_ids,
    registry_report,
    tts_candidate_ids,
)


def test_quality_matrix_includes_qwen3_asr_and_chat_driven_candidates() -> None:
    ids = asr_candidate_ids()

    assert "qwen3-asr-0.6b-hf" in ids
    assert "qwen3-asr-1.7b-hf" in ids
    assert "gigaam-v3-e2e-rnnt" in ids
    assert "parakeet-tdt-0.6b-v3" in ids
    assert "faster-whisper-large-v3-turbo-ct2" in ids


def test_quality_matrix_keeps_current_provider_baselines_first() -> None:
    assert default_asr_candidate_ids()[:2] == [
        "faster-whisper-small",
        "faster-whisper-large-v3",
    ]
    assert default_tts_candidate_ids()[0] == "piper-medium"


def test_quality_matrix_executable_asr_default_stays_local_provider_only() -> None:
    assert default_executable_asr_candidate_ids() == [
        "faster-whisper-small",
        "faster-whisper-large-v3",
    ]


def test_quality_matrix_includes_tts_quality_candidates() -> None:
    ids = tts_candidate_ids()

    assert "kokoro-82m" in ids
    assert "silero-v5_5-ru" in ids
    assert "qwen3-tts-0.6b-customvoice" in ids


def test_candidate_report_is_public_and_does_not_embed_local_chat_export() -> None:
    report = candidate_report(
        ["qwen3-asr-0.6b-hf", "gigaam-v3-e2e-rnnt"],
        role="asr",
    )

    payload = repr(report)
    assert "ChatExport_2026-08-04" not in payload
    assert "/home/anton" not in payload
    assert report[0]["id"] == "qwen3-asr-0.6b-hf"
    assert report[0]["runtime"] == "transformers"


def test_registry_report_serializes_tuple_fields_as_string_lists() -> None:
    for candidates in registry_report().values():
        for candidate in candidates:
            for field in ("source_urls", "strengths", "risks"):
                value = candidate[field]
                assert isinstance(value, list)
                assert all(isinstance(item, str) for item in value)
                assert all(len(item) > 1 for item in value)
