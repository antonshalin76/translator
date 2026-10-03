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
    AudioDirection,
    ComputeDevice,
    Language,
    ModelHealth,
    ModelKind,
    ModelState,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderId,
    ProviderLatency,
    ProviderSessionClosed,
    ProviderSessionOpened,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SampleFormat,
    SessionCloseReason,
    TranslationMode,
    UtteranceOutcome,
    VoiceGender,
    make_provider_error,
)

SESSION_ID = uuid4()
STREAM_ID = uuid4()
UTTERANCE_ID = uuid4()
EVENT_IDENTITY = {
    "session_id": SESSION_ID,
    "direction_id": AudioDirection.MICROPHONE,
    "stream_id": STREAM_ID,
    "utterance_id": UTTERANCE_ID,
}


def _health(request, *, asr_id="faster-whisper-large-v3-turbo"):
    return SimpleNamespace(
        session_id=request.session_id,
        direction_id=request.direction_id,
        provider_id=ProviderId.LOCAL,
        state=ProviderState.READY,
        models=(
            ModelHealth(
                kind=ModelKind.ASR,
                id=asr_id,
                state=ModelState.READY,
                device=ComputeDevice.CUDA,
            ),
            ModelHealth(
                kind=ModelKind.MT,
                id="nllb-200-distilled-600m-ct2-int8",
                state=ModelState.READY,
                device=ComputeDevice.CUDA,
            ),
            ModelHealth(
                kind=ModelKind.TTS,
                id="piper-medium",
                state=ModelState.READY,
                device=ComputeDevice.CPU,
            ),
        ),
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
            ProviderTranscriptDelta(
                **EVENT_IDENTITY, event_sequence=1, text=asr_text, is_final=True
            ),
            1_120_000_000,
        ),
        (
            ProviderTranslationDelta(
                **EVENT_IDENTITY,
                event_sequence=2,
                text=mt_text,
                stable_prefix=True,
                is_final=True,
            ),
            1_130_000_000,
        ),
        (
            ProviderAudioDelta(
                **EVENT_IDENTITY,
                sequence=0,
                event_sequence=3,
                provider_monotonic_ns=1_150_000_000,
                pcm=(b"\0" if silent else b"\x01") * 960,
                frame_duration_ms=20,
                sample_rate_hz=24_000,
                channels=1,
                sample_format=SampleFormat.S16LE,
            ),
            1_150_000_000,
        ),
        (
            ProviderLatency(
                **EVENT_IDENTITY,
                event_sequence=4,
                asr_final_text_ms=120,
                mt_first_text_ms=130,
                tts_first_audio_ms=150,
                provider_total_ms=160,
            ),
            1_155_000_000,
        ),
    ]
    if final:
        events.append(
            (
                ProviderUtteranceFinal(
                    **EVENT_IDENTITY,
                    event_sequence=5,
                    final_audio_sequence=0,
                    outcome=outcome,
                ),
                1_160_000_000,
            )
        )
    return events


def _evaluate(events):
    return evaluate_events(
        events,
        started_ns=1_000_000_000,
        session_id=SESSION_ID,
        direction_id=AudioDirection.MICROPHONE,
        stream_id=STREAM_ID,
        utterance_id=UTTERANCE_ID,
    )


def _drop_events(code: SafeErrorCode = SafeErrorCode.QUEUE_OVERFLOW):
    return [
        (
            ProviderLatency(
                **EVENT_IDENTITY,
                event_sequence=1,
                asr_final_text_ms=120,
                mt_first_text_ms=140,
                provider_total_ms=140,
            ),
            1_140_000_000,
        ),
        (
            make_provider_error(
                session_id=SESSION_ID,
                direction_id=AudioDirection.MICROPHONE,
                stream_id=STREAM_ID,
                utterance_id=UTTERANCE_ID,
                event_sequence=2,
                code=code,
                retryable=True,
            ),
            1_145_000_000,
        ),
        (
            ProviderUtteranceFinal(
                **EVENT_IDENTITY,
                event_sequence=3,
                outcome=UtteranceOutcome.DROPPED,
            ),
            1_150_000_000,
        ),
    ]


def test_only_bound_ordered_terminal_drop_can_continue() -> None:
    with pytest.raises(ValueError) as result:
        _evaluate(_drop_events())
    assert isinstance(result.value, product_audio_pair.VerifiedProviderDrop)
    assert result.value.safe_provider_codes == ["queue_overflow"]
    assert result.value.outcomes == ["dropped"]


def test_complete_debug_pair_before_terminal_drop_can_continue() -> None:
    terminal = [
        (event.model_copy(update={"event_sequence": event.event_sequence + 2}), at)
        for event, at in _drop_events()
    ]
    with pytest.raises(product_audio_pair.VerifiedProviderDrop):
        _evaluate(_events()[:2] + terminal)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: (
            rows[:1]
            + [(rows[1][0].model_copy(update={"utterance_id": uuid4()}), rows[1][1])]
            + rows[2:]
        ),
        lambda rows: (
            rows[:2]
            + [(rows[2][0].model_copy(update={"session_id": uuid4()}), rows[2][1])]
        ),
        lambda rows: rows[:2] + [rows[1]] + rows[2:],
        lambda rows: rows[:1] + list(reversed(rows[1:])),
        lambda rows: rows[:-1],
        lambda rows: rows[:1] + [rows[0]] + rows[1:],
        lambda rows: _drop_events(SafeErrorCode.PROVIDER_UNAVAILABLE),
        lambda rows: (
            rows[:1]
            + [(rows[1][0].model_copy(update={"retryable": False}), rows[1][1])]
            + rows[2:]
        ),
        lambda rows: (
            rows[:1]
            + [
                (rows[1][0].model_copy(update={"event_sequence": 4}), rows[1][1]),
                (rows[2][0].model_copy(update={"event_sequence": 5}), rows[2][1]),
            ]
        ),
        lambda rows: (
            rows[:2]
            + [(rows[2][0].model_copy(update={"event_sequence": 5}), rows[2][1])]
        ),
        lambda rows: (
            [(rows[0][0].model_copy(update={"provider_total_ms": None}), rows[0][1])]
            + rows[1:]
        ),
        lambda rows: (
            [(event, at) for event, at in _events()[:1]]
            + [
                (
                    event.model_copy(
                        update={"event_sequence": event.event_sequence + 1}
                    ),
                    at,
                )
                for event, at in rows
            ]
        ),
    ],
)
def test_unbound_or_malformed_drop_cannot_continue(mutation) -> None:
    with pytest.raises(ValueError) as result:
        _evaluate(mutation(_drop_events()))
    assert type(result.value) is ValueError


def test_event_result_requires_complete_text_audio_and_final() -> None:
    result = _evaluate(_events())
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
            _evaluate(events)
    with pytest.raises(ValueError):
        _evaluate(
            _events()
            + [
                (
                    PrivacySafeProviderError.model_construct(**EVENT_IDENTITY),
                    1_170_000_000,
                )
            ]
        )


@pytest.mark.parametrize("fault", ["unknown", "time_regression"])
def test_capture_event_gate_rejects_unknown_or_regressing_events(fault):
    events = _events()
    if fault == "unknown":
        events.insert(2, (object(), events[1][1]))
    else:
        events[2] = (events[2][0], events[0][1] - 1)
    with pytest.raises(ValueError):
        _evaluate(events)


def test_event_gate_accepts_bound_session_open_prefix():
    opening = ProviderSessionOpened.model_construct(
        session_id=SESSION_ID,
        direction_id=AudioDirection.MICROPHONE,
        event_sequence=0,
    )
    assert _evaluate([(opening, 900_000_000), *_events()])["status"] == "completed"


@pytest.mark.parametrize(
    ("fault", "enabled"),
    [
        (None, False),
        (None, True),
        *[
            (fault, True)
            for fault in (
                "cancelled",
                "task_cancelled",
                "health",
                "drain",
                "unknown",
                "late_event",
            )
        ],
    ],
)
def test_run_case_captures_pcm_only_after_attested_terminal_and_drain(
    tmp_path, monkeypatch, fault, enabled
):
    observed, captures = [], []
    context = {}
    private = tmp_path / "pcm"
    private.mkdir(mode=0o700)

    class Reservation:
        async def open(self):
            observed.append("open")
            return None, _health(context["request"])

        def drain(self, reason):
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            context.update(request=request, publish=publish)
            return Reservation()

        async def submit_frame(self, frame):
            context["frame"] = frame

        async def wait_idle(self):
            identity = {
                "session_id": context["request"].session_id,
                "direction_id": context["request"].direction_id,
                "stream_id": context["frame"].stream_id,
                "utterance_id": context["frame"].utterance_id,
            }
            batch = [event.model_copy(update=identity) for event, _ in _events()]
            if fault == "cancelled":
                batch[-1] = batch[-1].model_copy(
                    update={"outcome": UtteranceOutcome.CANCELLED}
                )
            if fault == "unknown":
                batch.insert(2, object())
            await context["publish"](tuple(batch), lambda: None)
            if fault == "task_cancelled":
                raise asyncio.CancelledError()

        async def health(self, session_id):
            observed.append("health")
            return _health(
                context["request"],
                asr_id="wrong" if fault == "health" else product_audio_pair.ASR_ID,
            )

    async def drain(_pending):
        observed.append("drain")
        if fault == "drain":
            raise RuntimeError("synthetic drain failure")
        await context["publish"](
            (
                ProviderSessionClosed(
                    session_id=context["request"].session_id,
                    direction_id=context["request"].direction_id,
                    event_sequence=6,
                    reason=SessionCloseReason.USER_STOP,
                ),
                *((object(),) if fault == "late_event" else ()),
            ),
            lambda: None,
        )
        return SimpleNamespace(
            session_id=context["request"].session_id, delivery_error=None
        )

    class Capture:
        def write_pcm(self, pcm, expected_hash, filename):
            assert observed == ["open", "health", "drain"]
            assert pcm == b"\x01" * 960
            assert expected_hash == _sha(pcm)
            captures.append(filename)
            with product_audio_pair.PcmArtifactStore(private) as store:
                return store.write_pcm(pcm, expected_hash, filename)

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", drain)
    case = {
        "language": "ru_ru",
        "pcm": b"\x01\x02" * 1600,
        "wav_sha256": "a" * 64,
    }
    coroutine = product_audio_pair._run_case(
        Provider(),
        case,
        TranslationMode.QUALITY_FIRST,
        VoiceGender.FEMALE,
        expected_asr_id=product_audio_pair.ASR_ID,
        expected_mt_id=product_audio_pair.BACKENDS["nllb"],
        **({"capture_audio": Capture()} if enabled else {}),
    )
    if fault is not None:
        with pytest.raises((ValueError, RuntimeError, asyncio.CancelledError)):
            asyncio.run(coroutine)
        assert captures == []
        assert list(private.iterdir()) == []
    else:
        result = asyncio.run(coroutine)
        if not enabled:
            assert captures == []
            assert "pcm_artifact" not in result
            assert list(private.iterdir()) == []
            return
        assert len(captures) == 1
        artifact = result["pcm_artifact"]
        assert artifact["filename"] == captures[0]
        assert artifact["input_wav_sha256"] == case["wav_sha256"]
        assert artifact["session_id"] == str(context["request"].session_id)
        assert artifact["direction_id"] == context["request"].direction_id.value
        assert artifact["stream_id"] == str(context["frame"].stream_id)
        assert artifact["utterance_id"] == str(context["frame"].utterance_id)
        assert artifact["target_language"] == "en"
        assert artifact["requested_voice"] == {
            "language": "en",
            "gender": "female",
            "engine": "piper",
        }
        assert artifact["effective_models_open"] == result["effective_models_open"]
        assert artifact["effective_models_after"] == result["effective_models_after"]
        with product_audio_pair.PcmArtifactStore(private) as store:
            assert _sha(store.read_verified(artifact)) == artifact["wav_sha256"]


@pytest.mark.parametrize(
    "events",
    [
        lambda rows: rows[:3] + rows[4:],
        lambda rows: rows[:2] + rows[3:],
        lambda rows: rows[:3] + [rows[3], rows[3]] + rows[4:],
        lambda rows: rows[:2] + list(reversed(rows[2:4])) + rows[4:],
        lambda rows: (
            rows[:2]
            + [(rows[2][0].model_copy(update={"session_id": uuid4()}), rows[2][1])]
            + rows[3:]
        ),
        lambda rows: (
            rows[:2]
            + [(rows[2][0].model_copy(update={"sequence": 1}), rows[2][1])]
            + rows[3:]
        ),
        lambda rows: (
            rows[:4]
            + [(rows[4][0].model_copy(update={"final_audio_sequence": 1}), rows[4][1])]
        ),
        lambda rows: rows[:3] + [(rows[4][0], rows[4][1]), rows[3]],
    ],
)
def test_event_result_rejects_missing_foreign_or_out_of_order_events(events) -> None:
    with pytest.raises(ValueError):
        _evaluate(events(_events()))


@pytest.mark.parametrize("matching_identity", [True, False])
@pytest.mark.parametrize(
    "asr_id", ["faster-whisper-small", "faster-whisper-large-v3-turbo"]
)
def test_run_case_binds_published_events_to_submitted_utterance(
    monkeypatch, matching_identity, asr_id
) -> None:
    published = {}
    drained = []

    class Reservation:
        async def open(self):
            return None, _health(published["request"], asr_id=asr_id)

        def drain(self, reason):
            drained.append(reason)
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
            identity = {
                "session_id": request.session_id,
                "direction_id": request.direction_id,
                "stream_id": frame.stream_id,
                "utterance_id": frame.utterance_id,
            }
            await published["publish"](
                tuple(
                    event.model_copy(update=identity) if matching_identity else event
                    for event, _ in _events()
                ),
                lambda: None,
            )

        async def health(self, session_id):
            assert session_id == published["request"].session_id
            return _health(published["request"], asr_id=asr_id)

    async def finish_cleanup(_pending):
        return SimpleNamespace(
            session_id=published["request"].session_id, delivery_error=None
        )

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    call = product_audio_pair._run_case(
        Provider(),
        {"language": "ru_ru", "pcm": b"\x01\x02" * 1600},
        TranslationMode.QUALITY_FIRST,
        VoiceGender.FEMALE,
        expected_asr_id=asr_id,
        expected_mt_id=product_audio_pair.BACKENDS["nllb"],
    )
    if matching_identity:
        result = asyncio.run(call)
        assert result["status"] == "completed"
        assert result["effective_models_open"]["asr"]["id"] == asr_id
        assert result["effective_models_after"]["asr"]["id"] == asr_id
    else:
        with pytest.raises(ValueError, match="identity"):
            asyncio.run(call)
    assert len(drained) == 1


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("health_fallback", [False, True])
@pytest.mark.parametrize("receipt_foreign", [False, True])
def test_verified_drop_requires_stable_health_and_clean_drain(
    monkeypatch, cleanup_fails, health_fallback, receipt_foreign
) -> None:
    published = {}
    drained = []

    class Reservation:
        async def open(self):
            return None, _health(published["request"])

        def drain(self, reason):
            drained.append(reason)
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            published.update(request=request, publish=publish)
            return Reservation()

        async def submit_frame(self, frame):
            published["frame"] = frame

        async def wait_idle(self):
            request = published["request"]
            frame = published["frame"]
            identity = {
                "session_id": request.session_id,
                "direction_id": request.direction_id,
                "stream_id": frame.stream_id,
                "utterance_id": frame.utterance_id,
            }
            await published["publish"](
                tuple(event.model_copy(update=identity) for event, _ in _drop_events()),
                lambda: None,
            )

        async def health(self, session_id):
            return _health(
                published["request"],
                asr_id=(
                    product_audio_pair.SMALL_ASR_ID
                    if health_fallback
                    else product_audio_pair.ASR_ID
                ),
            )

    async def finish_cleanup(_pending):
        if cleanup_fails:
            raise RuntimeError("synthetic cleanup failure")
        return SimpleNamespace(
            session_id=(
                uuid4() if receipt_foreign else published["request"].session_id
            ),
            delivery_error=None,
        )

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    with pytest.raises(RuntimeError if health_fallback else ValueError) as failure:
        asyncio.run(
            product_audio_pair._run_case(
                Provider(),
                {"language": "ru_ru", "pcm": b"\x01\x02" * 1600},
                TranslationMode.QUALITY_FIRST,
                VoiceGender.FEMALE,
                expected_asr_id=product_audio_pair.ASR_ID,
                expected_mt_id=product_audio_pair.BACKENDS["nllb"],
            )
        )
    if health_fallback:
        assert not isinstance(failure.value, product_audio_pair.VerifiedProviderDrop)
        assert (
            failure.value.post_attempt_effective_models["asr"]["id"]
            == product_audio_pair.SMALL_ASR_ID
        )
    else:
        assert isinstance(failure.value, product_audio_pair.VerifiedProviderDrop)
        assert (
            failure.value.open_effective_models
            == failure.value.post_attempt_effective_models
        )
        assert failure.value.attested_after_drain is not (
            cleanup_fails or receipt_foreign
        )
    assert len(drained) == 1


def test_run_case_rejects_effective_asr_fallback_after_submission(monkeypatch) -> None:
    published = {}
    drained = []

    class Reservation:
        async def open(self):
            return None, _health(published["request"])

        def drain(self, reason):
            drained.append(reason)
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            published.update(request=request, publish=publish)
            return Reservation()

        async def submit_frame(self, frame):
            published["frame"] = frame

        async def wait_idle(self):
            request = published["request"]
            frame = published["frame"]
            identity = {
                "session_id": request.session_id,
                "direction_id": request.direction_id,
                "stream_id": frame.stream_id,
                "utterance_id": frame.utterance_id,
            }
            await published["publish"](
                tuple(event.model_copy(update=identity) for event, _ in _events()),
                lambda: None,
            )

        async def health(self, session_id):
            return _health(published["request"], asr_id="faster-whisper-small")

    async def finish_cleanup(_pending):
        return SimpleNamespace(
            session_id=published["request"].session_id, delivery_error=None
        )

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    with pytest.raises(RuntimeError, match="effective model") as failure:
        asyncio.run(
            product_audio_pair._run_case(
                Provider(),
                {"language": "ru_ru", "pcm": b"\x01\x02" * 1600},
                TranslationMode.QUALITY_FIRST,
                VoiceGender.FEMALE,
                expected_asr_id=product_audio_pair.ASR_ID,
                expected_mt_id=product_audio_pair.BACKENDS["nllb"],
            )
        )
    assert (
        failure.value.post_attempt_effective_models["asr"]["id"]
        == "faster-whisper-small"
    )
    assert len(drained) == 1


def test_run_case_records_effective_asr_fallback_at_open(monkeypatch) -> None:
    published = {}
    drained = []

    class Reservation:
        async def open(self):
            return None, _health(published["request"], asr_id="faster-whisper-small")

        def drain(self, reason):
            drained.append(reason)
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            published["request"] = request
            return Reservation()

        async def health(self, session_id):
            return _health(published["request"], asr_id="faster-whisper-small")

    async def finish_cleanup(_pending):
        return SimpleNamespace(
            session_id=published["request"].session_id, delivery_error=None
        )

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    with pytest.raises(RuntimeError, match="effective model") as failure:
        asyncio.run(
            product_audio_pair._run_case(
                Provider(),
                {"language": "ru_ru", "pcm": b"\x01\x02" * 1600},
                TranslationMode.QUALITY_FIRST,
                VoiceGender.FEMALE,
                expected_asr_id=product_audio_pair.ASR_ID,
                expected_mt_id=product_audio_pair.BACKENDS["nllb"],
            )
        )
    assert failure.value.open_effective_models["asr"]["id"] == "faster-whisper-small"
    assert len(drained) == 1


def _pair_fixture_rows() -> list[dict]:
    return [
        {
            "origin_id": "ru-1",
            "condition": "clean",
            "backend": "small_nllb_same_code",
            "mode": "quality_first",
            "asr_text": "small source",
            "status": "completed",
            "wav_sha256": "same",
            "language": "ru_ru",
            "speaker_id": "speaker-1",
            "reference": "reference",
            "voice_gender": "female",
        },
        {
            "origin_id": "ru-1",
            "condition": "clean",
            "backend": "nllb",
            "mode": "quality_first",
            "asr_text": "a",
            "status": "completed",
            "wav_sha256": "same",
            "language": "ru_ru",
            "speaker_id": "speaker-1",
            "reference": "reference",
            "voice_gender": "female",
        },
        {
            "origin_id": "ru-1",
            "condition": "clean",
            "backend": "hy",
            "mode": "quality_first",
            "asr_text": "b",
            "status": "completed",
            "wav_sha256": "same",
            "language": "ru_ru",
            "speaker_id": "speaker-1",
            "reference": "reference",
            "voice_gender": "female",
        },
    ]


def test_pair_flags_asr_divergence() -> None:
    rows = _pair_fixture_rows()
    pair = pair_rows(rows, TranslationMode.QUALITY_FIRST)[0]
    assert pair["status"] == "complete"
    assert pair["mt_confounded_by_asr"] is True
    assert pair["baseline_asr_divergent"] is True


def test_pair_requires_small_same_code_arm() -> None:
    rows = _pair_fixture_rows()[1:]
    assert pair_rows(rows, TranslationMode.QUALITY_FIRST)[0]["status"] == "incomplete"


def test_pair_rejects_duplicate_or_unknown_arm() -> None:
    rows = _pair_fixture_rows()
    with pytest.raises(ValueError, match="duplicate"):
        pair_rows(rows + [rows[0]], TranslationMode.QUALITY_FIRST)
    with pytest.raises(ValueError, match="unknown"):
        pair_rows(
            rows + [{**rows[0], "backend": "unlisted"}], TranslationMode.QUALITY_FIRST
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wav_sha256", "different"),
        ("language", "en_us"),
        ("speaker_id", "different"),
        ("reference", "different"),
        ("voice_gender", "male"),
    ],
)
def test_pair_rejects_mismatched_input_identity(field, value) -> None:
    rows = _pair_fixture_rows()
    with pytest.raises(ValueError, match="identity"):
        pair_rows(
            rows[:2] + [{**rows[2], field: value}],
            TranslationMode.QUALITY_FIRST,
        )


def test_pair_rejects_missing_identity_in_all_arms() -> None:
    rows = _pair_fixture_rows()
    for row in rows:
        row.pop("speaker_id")
    with pytest.raises(ValueError, match="missing.*identity"):
        pair_rows(rows, TranslationMode.QUALITY_FIRST)


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
@pytest.mark.parametrize("gender", list(VoiceGender))
def test_selected_mode_reaches_session_and_every_frame(
    monkeypatch, mode, language, gender
) -> None:
    sessions = []
    frames = []

    class Reservation:
        async def open(self):
            return None, _health(sessions[-1])

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

        async def health(self, session_id):
            return _health(sessions[-1])

    async def finish_cleanup(_pending):
        return SimpleNamespace(session_id=sessions[-1].session_id, delivery_error=None)

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    monkeypatch.setattr(
        product_audio_pair,
        "evaluate_events",
        lambda events, **identity: {"status": "completed"},
    )
    result = asyncio.run(
        product_audio_pair._run_case(
            Provider(),
            {"language": language, "pcm": b"\x01\x02" * 3200},
            mode,
            gender,
            expected_asr_id=product_audio_pair.ASR_ID,
            expected_mt_id=product_audio_pair.BACKENDS["nllb"],
        )
    )
    assert result["status"] == "completed"
    assert len(sessions) == 1
    assert sessions[0].mode is mode
    assert sessions[0].voice_profile.gender is gender
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


@pytest.mark.parametrize(
    ("option", "expected"),
    [([], VoiceGender.FEMALE), (["--voice-gender", "male"], VoiceGender.MALE)],
)
def test_cli_selects_voice_gender(monkeypatch, capsys, option, expected) -> None:
    observed = []

    async def fake_run(arguments):
        observed.append(arguments.voice_gender)
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


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_capture_is_explicit_and_disabled_by_default(monkeypatch, enabled):
    seen = []

    async def fake_run(args):
        seen.append(args.capture_audio)
        return {"status": "complete"}

    monkeypatch.setattr(product_audio_pair, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--manifest",
            "/tmp/manifest",
            "--screen",
            "/tmp/screen",
            "--turbo",
            "/tmp/turbo",
            "--output",
            "/tmp/output",
            *(["--capture-audio", "/tmp/private-pcm"] if enabled else []),
        ],
    )
    assert product_audio_pair.main() == 0
    assert seen == ([Path("/tmp/private-pcm")] if enabled else [None])


@pytest.mark.parametrize(
    ("source", "target"),
    [("ru_ru", Language.EN), ("en_us", Language.RU)],
)
def test_request_selects_pinned_male_target_voice(source, target) -> None:
    request = product_audio_pair._request(
        source, TranslationMode.QUALITY_FIRST, VoiceGender.MALE
    )
    assert request.voice_profile.language is target
    assert request.voice_profile.gender is VoiceGender.MALE


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
    "failure_stage",
    ["complete", "build", "attempt", "provider_drop", "shutdown", "pair_validation"],
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
    requested_gender = VoiceGender.MALE
    observed_builds = []

    class Provider:
        def __init__(self):
            self.selected_asr_id = os.environ["TRANSLATOR_ASR_MODEL_ID"]
            self.selected_mt_id = os.environ["TRANSLATOR_MT_MODEL_ID"]

        async def shutdown(self):
            if failure_stage == "shutdown":
                raise RuntimeError("synthetic shutdown failure")

    def build_provider(**kwargs):
        if failure_stage == "build":
            raise RuntimeError("synthetic build failure")
        observed_builds.append(
            (
                os.environ["TRANSLATOR_ASR_MODEL_ID"],
                os.environ["TRANSLATOR_MT_MODEL_ID"],
            )
        )
        return Provider()

    async def run_case(provider, case, mode, voice_gender, **expected_models):
        assert voice_gender is requested_gender
        assert expected_models == {
            "expected_asr_id": provider.selected_asr_id,
            "expected_mt_id": provider.selected_mt_id,
        }
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
    if failure_stage == "pair_validation":

        def invalid_pairs(rows, mode):
            raise ValueError("synthetic pair identity failure")

        monkeypatch.setattr(product_audio_pair, "pair_rows", invalid_pairs)
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
        order="small-nllb-hy",
        mode=requested,
        voice_gender=requested_gender,
    )
    result = asyncio.run(product_audio_pair.run(arguments))
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert records
    assert result["status"] == ("complete" if failure_stage == "complete" else "failed")
    assert all(record.get("mode") == requested.value for record in records)
    assert records[0]["voice_gender"] == requested_gender.value
    assert records[0]["schema"] == "translator.product-audio-pair.v2"
    assert records[0]["baseline_role"] == "same_code_ablation_not_original_main"
    assert records[0]["input_selection"] == "turbo_screened_development_only"
    assert (
        "exact voice not independently observed"
        in records[0]["voice_observation_scope"]
    )
    if failure_stage in {"complete", "pair_validation"}:
        assert observed_builds == [
            ("faster-whisper-small", "nllb-200-distilled-600m-ct2-int8"),
            ("faster-whisper-large-v3-turbo", "nllb-200-distilled-600m-ct2-int8"),
            ("faster-whisper-large-v3-turbo", "hy-mt2-1.8b-gguf-q4-k-m"),
        ]
    assert all(
        record["voice_gender"] == requested_gender.value
        for record in records
        if record["type"] == "attempt"
    )
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
        assert len([record for record in records if record["type"] == "not_run"]) == 5
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
    if failure_stage == "pair_validation":
        assert any(record["type"] == "pair_error" for record in records)
        assert records[-1]["status"] == "failed"


@pytest.mark.parametrize("attested_after_drain", [False, True])
def test_distinct_cases_and_arms_continue_after_one_verified_drop(
    tmp_path, monkeypatch, attested_after_drain
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
    monkeypatch.setattr(product_audio_pair, "HY_SERVER", tmp_path / "absent")
    monkeypatch.setenv("TRANSLATOR_MODEL_CACHE_ROOT", str(tmp_path))
    seen = []
    shutdowns = []

    class Provider:
        async def shutdown(self):
            shutdowns.append(True)

    monkeypatch.setattr(
        product_audio_pair, "build_local_provider", lambda **kw: Provider()
    )

    async def run_case(provider, case, mode, gender, **expected_models):
        seen.append(
            (
                expected_models["expected_asr_id"],
                expected_models["expected_mt_id"],
                case["origin_id"],
            )
        )
        if (
            expected_models["expected_asr_id"] == product_audio_pair.SMALL_ASR_ID
            and case["origin_id"] == "ru_ru-2"
        ):
            with pytest.raises(product_audio_pair.VerifiedProviderDrop) as drop:
                _evaluate(_drop_events())
            drop.value.open_effective_models = {"asr": "small"}
            drop.value.post_attempt_effective_models = {"asr": "small"}
            drop.value.attested_after_drain = attested_after_drain
            raise drop.value
        return {"status": "completed", "asr_text": "source", "mt_text": "target"}

    monkeypatch.setattr(product_audio_pair, "_run_case", run_case)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    arguments = SimpleNamespace(
        manifest=tmp_path / "manifest",
        screen=tmp_path / "screen",
        turbo=tmp_path / "turbo",
        output=private / "attempts.jsonl",
        case_id=None,
        smoke=False,
        order="small-nllb-hy",
        mode=TranslationMode.QUALITY_FIRST,
        voice_gender=VoiceGender.FEMALE,
    )
    result = asyncio.run(product_audio_pair.run(arguments))
    records = [
        json.loads(line)
        for line in arguments.output.read_text(encoding="utf-8").splitlines()
    ]
    attempts = [record for record in records if record["type"] == "attempt"]
    expected_attempts = 72 if attested_after_drain else 3
    assert result == {
        "status": "failed",
        "attempts": expected_attempts,
        "pairs": 24 if attested_after_drain else 3,
    }
    assert len(attempts) == len(seen) == len(set(seen)) == expected_attempts
    assert len([row for row in attempts if row["status"] == "failed"]) == 1
    assert len([record for record in records if record["type"] == "not_run"]) == (
        0 if attested_after_drain else 69
    )
    assert len(shutdowns) == (3 if attested_after_drain else 1)
    assert records[-1]["status"] == "failed"
    assert len(
        [pair for pair in records[-1]["pairs"] if pair["status"] == "incomplete"]
    ) == (1 if attested_after_drain else 3)


def test_incomplete_pair_still_rejects_cross_arm_input_mismatch() -> None:
    rows = [
        {
            "origin_id": "ru-1",
            "condition": "clean",
            "backend": backend,
            "status": "failed" if backend == "small_nllb_same_code" else "completed",
            "mode": TranslationMode.QUALITY_FIRST.value,
            "wav_sha256": "wrong" if backend == "hy" else "same",
            "language": "ru_ru",
            "speaker_id": "speaker-1",
            "reference": "reference",
            "voice_gender": "female",
        }
        for backend in product_audio_pair.ARMS
    ]
    with pytest.raises(ValueError, match="input identity"):
        pair_rows(rows, TranslationMode.QUALITY_FIRST)


def test_journal_fsync_failure_never_reports_complete(
    tmp_path, monkeypatch, capsys
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    output = private / "attempts.jsonl"

    async def fail_during_journal(arguments):
        with open_journal(arguments.output) as journal:
            write_record(journal, {"type": "header"})

    def failed_fsync(fd):
        raise OSError("private filesystem detail")

    monkeypatch.setattr(product_audio_pair, "run", fail_during_journal)
    monkeypatch.setattr(product_audio_pair.os, "fsync", failed_fsync)
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
        ],
    )
    assert product_audio_pair.main() == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "error_type": "OSError",
    }
    assert "terminal" not in output.read_text(encoding="utf-8")


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
            return None, _health(published["request"])

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

        async def health(self, session_id):
            return _health(published["request"])

    async def finish_cleanup(_pending):
        return SimpleNamespace(
            session_id=published["request"].session_id, delivery_error=None
        )

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    with pytest.raises(ValueError, match="provider returned a safe error") as failure:
        asyncio.run(
            product_audio_pair._run_case(
                Provider(),
                {"language": "en_us", "pcm": b"\x01\x02" * 1600},
                TranslationMode.STREAMING_FIRST,
                VoiceGender.FEMALE,
                expected_asr_id=product_audio_pair.ASR_ID,
                expected_mt_id=product_audio_pair.BACKENDS["nllb"],
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


@pytest.mark.parametrize("fallback", [False, True])
def test_wait_idle_failure_checks_post_attempt_model_without_inventing_latency(
    monkeypatch, fallback
) -> None:
    health_checks = []

    class Reservation:
        async def open(self):
            return None, _health(published["request"])

        def drain(self, reason):
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            published["request"] = request
            return Reservation()

        async def submit_frame(self, frame):
            return None

        async def wait_idle(self):
            raise RuntimeError("synthetic wait_idle failure")

        async def health(self, session_id):
            health_checks.append(session_id)
            return _health(
                published["request"],
                asr_id="faster-whisper-small"
                if fallback
                else product_audio_pair.ASR_ID,
            )

    async def finish_cleanup(_pending):
        return SimpleNamespace(
            session_id=published["request"].session_id, delivery_error=None
        )

    published = {}
    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    with pytest.raises(RuntimeError, match="wait_idle failure") as failure:
        asyncio.run(
            product_audio_pair._run_case(
                Provider(),
                {"language": "en_us", "pcm": b"\x01\x02" * 1600},
                TranslationMode.STREAMING_FIRST,
                VoiceGender.FEMALE,
                expected_asr_id=product_audio_pair.ASR_ID,
                expected_mt_id=product_audio_pair.BACKENDS["nllb"],
            )
        )
    assert not hasattr(failure.value, "safe_provider_latency")
    assert health_checks == [published["request"].session_id]
    if fallback:
        assert (
            failure.value.post_attempt_model_error == "effective model identity differs"
        )
        assert (
            failure.value.post_attempt_effective_models["asr"]["id"]
            == "faster-whisper-small"
        )
    else:
        assert (
            failure.value.post_attempt_effective_models["asr"]["id"]
            == product_audio_pair.ASR_ID
        )


@pytest.mark.parametrize("health_failure", ["timeout", "runtime"])
def test_failed_attempt_bounds_health_probe_and_preserves_cleanup_error(
    monkeypatch, health_failure
) -> None:
    published = {}
    drained = []

    class Reservation:
        async def open(self):
            return None, _health(published["request"])

        def drain(self, reason):
            drained.append(reason)
            return reason

    class Provider:
        def reserve_session(self, request, publish):
            published["request"] = request
            return Reservation()

        async def submit_frame(self, frame):
            return None

        async def wait_idle(self):
            raise RuntimeError("original attempt failure")

        async def health(self, session_id):
            if health_failure == "runtime":
                raise RuntimeError("synthetic health failure")
            await asyncio.sleep(0.05)
            return _health(published["request"])

    async def finish_cleanup(_pending):
        raise RuntimeError("cleanup also failed")

    monkeypatch.setattr(product_audio_pair, "finish_cleanup", finish_cleanup)
    monkeypatch.setattr(product_audio_pair, "HEALTH_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(RuntimeError, match="original attempt failure") as failure:
        asyncio.run(
            product_audio_pair._run_case(
                Provider(),
                {"language": "ru_ru", "pcm": b"\x01\x02" * 1600},
                TranslationMode.QUALITY_FIRST,
                VoiceGender.FEMALE,
                expected_asr_id=product_audio_pair.ASR_ID,
                expected_mt_id=product_audio_pair.BACKENDS["nllb"],
            )
        )
    assert failure.value.post_attempt_health_error == (
        "TimeoutError" if health_failure == "timeout" else "RuntimeError"
    )
    assert failure.value.cleanup_error_type == "RuntimeError"
    assert len(drained) == 1
