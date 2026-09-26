"""Contract tests for the private, paired product-audio diagnostic."""

from __future__ import annotations

import hashlib
import json
import stat
import wave
from pathlib import Path

import pytest
from translator_product_audio_pair import (
    evaluate_events,
    load_cases,
    open_journal,
    pair_rows,
    write_record,
)
from translator_sidecar.provider_contract import (
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SampleFormat,
    UtteranceOutcome,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frozen_fixture(tmp_path: Path, *, channels: int = 1) -> tuple[Path, ...]:
    corpus = tmp_path / "corpus"
    clips = corpus / "clips"
    clips.mkdir(parents=True)
    wav = clips / "ru-1-clean.wav"
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x01\x02" * 1600 * channels)
    sample = {
        "origin_id": "ru-1",
        "condition": "clean",
        "language": "ru_ru",
        "speaker_id": "speaker-1",
        "audio_file": "clips/ru-1-clean.wav",
        "sha256": _sha(wav.read_bytes()),
        "reference": "исходная фраза",
        "critical_labels": ["negation"],
    }
    manifest = corpus / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema": 1, "purpose": "asr_independent_holdout", "samples": [sample]}
        ),
        encoding="utf-8",
    )
    turbo = tmp_path / "turbo.json"
    turbo.write_text(
        json.dumps(
            {
                "model_id": "turbo",
                "manifest_sha256": _sha(manifest.read_bytes()),
                "results": [
                    {**sample, "status": "completed", "transcript": "исходная фраза"}
                ],
            }
        ),
        encoding="utf-8",
    )
    screen = tmp_path / "screen.json"
    screen.write_text(
        json.dumps(
            {
                "turbo_report_sha256": _sha(turbo.read_bytes()),
                "cases": [{"origin_id": "ru-1", "condition": "clean"}],
            }
        ),
        encoding="utf-8",
    )
    return manifest, screen, turbo, wav


def _load(fixture: tuple[Path, ...]):
    manifest, screen, turbo, _ = fixture
    return load_cases(
        manifest,
        screen,
        turbo,
        manifest_sha256=_sha(manifest.read_bytes()),
        screen_sha256=_sha(screen.read_bytes()),
        turbo_sha256=_sha(turbo.read_bytes()),
    )


def test_load_cases_binds_frozen_audio_and_provenance(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    cases = _load(fixture)
    assert len(cases) == 1
    assert cases[0]["origin_id"] == "ru-1"
    assert cases[0]["wav_sha256"] == _sha(fixture[3].read_bytes())


def test_load_cases_rejects_changed_or_duplicate_selection(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    with pytest.raises(ValueError, match="hash"):
        load_cases(
            *fixture[:3],
            manifest_sha256="0" * 64,
            screen_sha256=_sha(fixture[1].read_bytes()),
            turbo_sha256=_sha(fixture[2].read_bytes()),
        )
    screen = fixture[1]
    data = json.loads(screen.read_text(encoding="utf-8"))
    data["cases"].append(data["cases"][0])
    screen.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _load(fixture)


def test_load_cases_rejects_wrong_language_and_wav_format(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    turbo = fixture[2]
    data = json.loads(turbo.read_text(encoding="utf-8"))
    data["results"][0]["language"] = "en_us"
    turbo.write_text(json.dumps(data), encoding="utf-8")
    screen = fixture[1]
    screen_data = json.loads(screen.read_text(encoding="utf-8"))
    screen_data["turbo_report_sha256"] = _sha(turbo.read_bytes())
    screen.write_text(json.dumps(screen_data), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        _load(fixture)

    stereo = _frozen_fixture(tmp_path / "stereo", channels=2)
    with pytest.raises(ValueError, match="WAV"):
        _load(stereo)


def test_load_cases_rejects_unusable_frozen_turbo_text(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    turbo = fixture[2]
    data = json.loads(turbo.read_text(encoding="utf-8"))
    data["results"][0]["status"] = "failed"
    data["results"][0]["transcript"] = ""
    turbo.write_text(json.dumps(data), encoding="utf-8")
    screen = fixture[1]
    screen_data = json.loads(screen.read_text(encoding="utf-8"))
    screen_data["turbo_report_sha256"] = _sha(turbo.read_bytes())
    screen.write_text(json.dumps(screen_data), encoding="utf-8")
    with pytest.raises(ValueError, match="Turbo"):
        _load(fixture)


def test_load_cases_rejects_audio_symlink(tmp_path: Path) -> None:
    fixture = _frozen_fixture(tmp_path)
    wav = fixture[3]
    original = wav.with_suffix(".saved")
    wav.rename(original)
    wav.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        _load(fixture)


def _events(
    *,
    silent: bool = False,
    final: bool = True,
    outcome: UtteranceOutcome = UtteranceOutcome.COMPLETED,
    asr_text: str = "source",
    mt_text: str = "target",
):
    events = [
        (
            ProviderTranscriptDelta.model_construct(text=asr_text, is_final=True),
            1_120_000_000,
        ),
        (
            ProviderTranslationDelta.model_construct(text=mt_text, is_final=True),
            1_130_000_000,
        ),
        (
            ProviderAudioDelta.model_construct(
                pcm=(b"\0" if silent else b"\x01") * 960,
                frame_duration_ms=20,
                sample_rate_hz=24_000,
                channels=1,
                sample_format=SampleFormat.S16LE,
            ),
            1_150_000_000,
        ),
    ]
    if final:
        events.append(
            (
                ProviderUtteranceFinal.model_construct(outcome=outcome),
                1_160_000_000,
            )
        )
    return events


def test_event_result_requires_complete_text_audio_and_final() -> None:
    result = evaluate_events(_events(), started_ns=1_000_000_000)
    assert result["status"] == "completed"
    assert result["asr_text"] == "source"
    assert result["mt_text"] == "target"
    assert result["first_pcm_ms"] == pytest.approx(150.0)
    assert result["pcm_sha256"] == _sha(b"\x01" * 960)
    for events in (
        _events(final=False),
        _events(silent=True),
        _events(outcome=UtteranceOutcome.CANCELLED),
        _events(outcome=UtteranceOutcome.DROPPED),
        _events(asr_text=" "),
        _events(mt_text=" "),
        [
            (event, 999_000_000 if isinstance(event, ProviderAudioDelta) else at)
            for event, at in _events()
        ],
    ):
        with pytest.raises(ValueError):
            evaluate_events(events, started_ns=1_000_000_000)
    with pytest.raises(ValueError):
        evaluate_events(
            _events() + [(PrivacySafeProviderError.model_construct(), 1_170_000_000)],
            started_ns=1_000_000_000,
        )


def test_pair_flags_asr_divergence() -> None:
    rows = [
        {
            "origin_id": "ru-1",
            "condition": "clean",
            "backend": "nllb",
            "asr_text": "a",
            "status": "completed",
        },
        {
            "origin_id": "ru-1",
            "condition": "clean",
            "backend": "hy",
            "asr_text": "b",
            "status": "completed",
        },
    ]
    assert pair_rows(rows)[0]["mt_confounded_by_asr"] is True
    assert pair_rows(rows[:1])[0]["status"] == "incomplete"
    with pytest.raises(ValueError, match="duplicate"):
        pair_rows(rows + [rows[0]])


def test_journal_is_private_exclusive_and_durable(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    output = private / "attempts.jsonl"
    with open_journal(output) as journal:
        write_record(journal, {"status": "failed", "asr_text": "private"})
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "private" in output.read_text(encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError)):
        open_journal(output)
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private"):
        open_journal(public / "attempts.jsonl")
