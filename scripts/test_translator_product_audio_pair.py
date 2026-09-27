"""Contract tests for the private, paired product-audio diagnostic."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import translator_product_audio_pair as product_audio_pair
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
    ProviderLatency,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SampleFormat,
    TranslationMode,
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
            "mode": "quality_first",
            "asr_text": "a",
            "status": "completed",
        },
        {
            "origin_id": "ru-1",
            "condition": "clean",
            "backend": "hy",
            "mode": "quality_first",
            "asr_text": "b",
            "status": "completed",
        },
    ]
    assert (
        pair_rows(rows, TranslationMode.QUALITY_FIRST)[0]["mt_confounded_by_asr"]
        is True
    )
    assert (
        pair_rows(rows[:1], TranslationMode.QUALITY_FIRST)[0]["status"] == "incomplete"
    )
    with pytest.raises(ValueError, match="duplicate"):
        pair_rows(rows + [rows[0]], TranslationMode.QUALITY_FIRST)


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


@pytest.mark.parametrize("mode", list(TranslationMode))
@pytest.mark.parametrize("language", ["ru_ru", "en_us"])
def test_selected_mode_reaches_session_and_every_frame(
    monkeypatch, mode, language
) -> None:
    sessions = []
    frames = []

    class Reservation:
        async def open(self):
            return None, SimpleNamespace(state=ProviderState.READY)

        def drain(self, reason):
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            sessions.append(request)
            return Reservation()

        async def submit_frame(self, frame):
            frames.append(frame)

        async def wait_idle(self):
            return None

    async def finish_cleanup(_pending):
        return SimpleNamespace(delivery_error=None)

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    monkeypatch.setattr(
        product_audio_pair,
        "evaluate_events",
        lambda events, *, started_ns: {"status": "completed"},
    )
    result = asyncio.run(
        product_audio_pair._run_case(
            Provider(),
            {"language": language, "pcm": b"\x01\x02" * 3200},
            mode,
        )
    )
    assert result["status"] == "completed"
    assert len(sessions) == 1
    assert sessions[0].mode is mode
    assert len(frames) == 2
    assert all(frame.mode is mode for frame in frames)
    assert all(frame.direction_id is sessions[0].direction_id for frame in frames)


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ([], TranslationMode.QUALITY_FIRST),
        (["--mode", "balanced"], TranslationMode.BALANCED),
        (["--mode", "streaming_first"], TranslationMode.STREAMING_FIRST),
    ],
)
def test_cli_selects_mode_with_compatible_default(
    monkeypatch, capsys, option, expected
) -> None:
    observed = []

    async def fake_run(arguments):
        observed.append(arguments.mode)
        return {"status": "complete"}

    monkeypatch.setattr(product_audio_pair, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "translator_product_audio_pair.py",
            "--manifest",
            "/tmp/manifest",
            "--screen",
            "/tmp/screen",
            "--turbo",
            "/tmp/turbo",
            "--output",
            "/tmp/private-result",
            *option,
        ],
    )
    assert product_audio_pair.main() == 0
    assert observed == [expected]
    assert '"status": "complete"' in capsys.readouterr().out


def test_invalid_cli_mode_exits_before_corpus_or_model_access(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "journal.jsonl"

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid mode reached corpus or model access")

    monkeypatch.setattr(product_audio_pair, "load_cases", forbidden)
    monkeypatch.setattr(product_audio_pair, "build_local_provider", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "translator_product_audio_pair.py",
            "--manifest",
            "/tmp/manifest",
            "--screen",
            "/tmp/screen",
            "--turbo",
            "/tmp/turbo",
            "--output",
            str(output),
            "--mode",
            "invalid",
        ],
    )
    with pytest.raises(SystemExit) as exit_result:
        product_audio_pair.main()
    assert exit_result.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("invalid_mode", [None, "quality_first"])
def test_pair_rejects_missing_or_wrong_mode_relative_to_requested(
    status, invalid_mode
) -> None:
    rows = [
        {
            "origin_id": "ru-1",
            "condition": "clean",
            "backend": backend,
            "asr_text": "source",
            "status": status,
            "mode": invalid_mode,
            "wav_sha256": "same",
            "voice_gender": "female",
        }
        for backend in ("nllb", "hy")
    ]
    with pytest.raises(ValueError, match="mode"):
        pair_rows(rows, TranslationMode.BALANCED)


@pytest.mark.parametrize(
    "failure_stage", ["complete", "build", "attempt", "provider_drop", "shutdown"]
)
def test_run_journal_binds_mode_to_success_and_failure_paths(
    tmp_path, monkeypatch, failure_stage
) -> None:
    cases = [
        {
            "origin_id": f"{language}-{index}",
            "condition": "clean",
            "language": language,
            "speaker_id": f"speaker-{index}",
            "wav_sha256": f"{index:064x}",
            "reference": "reference",
            "turbo_text": "source",
            "critical_labels": [],
        }
        for language in ("ru_ru", "en_us")
        for index in range(12)
    ]
    monkeypatch.setattr(product_audio_pair, "load_cases", lambda *args: cases)
    monkeypatch.setattr(product_audio_pair, "_resources", lambda provider: {})
    monkeypatch.setattr(product_audio_pair, "HY_SERVER", tmp_path / "not-installed")
    monkeypatch.setenv("TRANSLATOR_MODEL_CACHE_ROOT", str(tmp_path))
    requested = TranslationMode.BALANCED

    class Provider:
        _asr_model_id = product_audio_pair.ASR_ID

        def __init__(self):
            self._mt_model_id = os.environ["TRANSLATOR_MT_MODEL_ID"]

        async def shutdown(self):
            if failure_stage == "shutdown":
                raise RuntimeError("synthetic shutdown failure")

    def build_provider(**kwargs):
        if failure_stage == "build":
            raise RuntimeError("synthetic build failure")
        return Provider()

    async def run_case(provider, case, mode):
        if failure_stage == "attempt":
            raise RuntimeError("synthetic case failure")
        if failure_stage == "provider_drop":
            error = ValueError("provider returned a safe error")
            error.safe_provider_codes = ["provider_unavailable"]
            error.outcomes = ["dropped"]
            error.safe_provider_latency = {
                "asr_final_text_ms": 111,
                "mt_first_text_ms": None,
                "tts_first_audio_ms": None,
                "provider_total_ms": 130,
            }
            raise error
        return {
            "status": "completed",
            "asr_text": "source",
            "mt_text": "target",
        }

    monkeypatch.setattr(product_audio_pair, "build_local_provider", build_provider)
    monkeypatch.setattr(product_audio_pair, "_run_case", run_case)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    output = private / "attempts.jsonl"
    arguments = SimpleNamespace(
        manifest=tmp_path / "manifest",
        screen=tmp_path / "screen",
        turbo=tmp_path / "turbo",
        output=output,
        case_id=None,
        smoke=True,
        order="nllb-hy",
        mode=requested,
    )
    result = asyncio.run(product_audio_pair.run(arguments))
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert records
    assert result["status"] == ("complete" if failure_stage == "complete" else "failed")
    assert all(record.get("mode") == requested.value for record in records)
    assert records[0]["type"] == "header"
    assert records[-1]["type"] == "terminal"
    assert records[-1]["status"] == (
        "complete" if failure_stage == "complete" else "failed"
    )
    if failure_stage == "build":
        assert {"arm_error", "not_run", "arm_end"} <= {
            record["type"] for record in records
        }
    if failure_stage == "attempt":
        assert {"attempt", "not_run", "arm_end"} <= {
            record["type"] for record in records
        }
        assert len([record for record in records if record["type"] == "not_run"]) == 3
    if failure_stage == "shutdown":
        assert {"cleanup_error", "arm_end"} <= {record["type"] for record in records}
    if failure_stage == "provider_drop":
        failed = next(record for record in records if record["type"] == "attempt")
        assert failed["status"] == "failed"
        assert failed["safe_provider_codes"] == ["provider_unavailable"]
        assert failed["observed_outcomes"] == ["dropped"]
        assert failed["safe_provider_latency"] == {
            "asr_final_text_ms": 111,
            "mt_first_text_ms": None,
            "tts_first_audio_ms": None,
            "provider_total_ms": 130,
        }
        assert records[-1]["status"] == "failed"


@pytest.mark.parametrize(
    "latency_kind",
    ["one", "none", "duplicate", "foreign_session", "foreign_utterance", "invalid"],
)
def test_dropped_provider_latency_is_exactly_allowlisted_or_unmeasured(
    monkeypatch, latency_kind
) -> None:
    published = {}

    class Reservation:
        async def open(self):
            return None, SimpleNamespace(state=ProviderState.READY)

        def drain(self, reason):
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            published["request"] = request
            published["publish"] = publish
            return Reservation()

        async def submit_frame(self, frame):
            published["frame"] = frame

        async def wait_idle(self):
            request = published["request"]
            frame = published["frame"]
            latencies = []
            if latency_kind != "none":
                latency = ProviderLatency.model_construct(
                    session_id=(
                        uuid4()
                        if latency_kind == "foreign_session"
                        else request.session_id
                    ),
                    utterance_id=(
                        uuid4()
                        if latency_kind == "foreign_utterance"
                        else frame.utterance_id
                    ),
                    asr_final_text_ms=(
                        "not-a-number" if latency_kind == "invalid" else 111
                    ),
                    mt_first_text_ms=None,
                    tts_first_audio_ms=None,
                    provider_total_ms=130,
                )
                latencies = [latency]
                if latency_kind == "duplicate":
                    latencies.append(latency)
            await published["publish"](
                tuple(latencies)
                + (
                    PrivacySafeProviderError.model_construct(
                        code=SafeErrorCode.PROVIDER_UNAVAILABLE
                    ),
                    ProviderUtteranceFinal.model_construct(
                        outcome=UtteranceOutcome.DROPPED
                    ),
                ),
                lambda: None,
            )

    async def finish_cleanup(_pending):
        return SimpleNamespace(delivery_error=None)

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    with pytest.raises(ValueError, match="provider returned a safe error") as failure:
        asyncio.run(
            product_audio_pair._run_case(
                Provider(),
                {"language": "en_us", "pcm": b"\x01\x02" * 1600},
                TranslationMode.STREAMING_FIRST,
            )
        )
    assert failure.value.safe_provider_codes == ["provider_unavailable"]
    assert failure.value.outcomes == ["dropped"]
    expected = (
        {
            "asr_final_text_ms": 111,
            "mt_first_text_ms": None,
            "tts_first_audio_ms": None,
            "provider_total_ms": 130,
        }
        if latency_kind == "one"
        else None
    )
    assert failure.value.safe_provider_latency == expected
    if expected is not None:
        assert set(expected) == {
            "asr_final_text_ms",
            "mt_first_text_ms",
            "tts_first_audio_ms",
            "provider_total_ms",
        }
        assert all(value is None or type(value) is int for value in expected.values())


def test_wait_idle_failure_does_not_invent_latency(monkeypatch) -> None:
    class Reservation:
        async def open(self):
            return None, SimpleNamespace(state=ProviderState.READY)

        def drain(self, reason):
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            return Reservation()

        async def submit_frame(self, frame):
            return None

        async def wait_idle(self):
            raise RuntimeError("synthetic wait_idle failure")

    async def finish_cleanup(_pending):
        return SimpleNamespace(delivery_error=None)

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    with pytest.raises(RuntimeError, match="wait_idle failure") as failure:
        asyncio.run(
            product_audio_pair._run_case(
                Provider(),
                {"language": "en_us", "pcm": b"\x01\x02" * 1600},
                TranslationMode.STREAMING_FIRST,
            )
        )
    assert not hasattr(failure.value, "safe_provider_latency")
