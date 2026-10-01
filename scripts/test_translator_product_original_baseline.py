"""Contract tests for the original-main saved-audio baseline driver."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import translator_product_original_baseline as baseline
from translator_sidecar.provider_contract import (
    ComputeDevice,
    ModelHealth,
    ModelKind,
    ModelState,
    ProviderAudioDelta,
    ProviderId,
    ProviderLatency,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SampleFormat,
    TranslationMode,
    UtteranceOutcome,
    VoiceGender,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _health(request, *, state=ProviderState.READY, asr="small"):
    return SimpleNamespace(
        session_id=request.session_id,
        direction_id=request.direction_id,
        provider_id=ProviderId.LOCAL,
        state=state,
        models=(
            ModelHealth(
                kind=ModelKind.ASR,
                id=asr,
                state=ModelState.READY,
                device=ComputeDevice.CPU,
            ),
            ModelHealth(
                kind=ModelKind.MT,
                id=baseline.MT_ID,
                state=ModelState.READY,
                device=ComputeDevice.CPU,
            ),
            ModelHealth(
                kind=ModelKind.TTS,
                id="piper-medium",
                state=ModelState.READY,
                device=ComputeDevice.CPU,
            ),
        ),
    )


def _events(request, frame):
    identity = {
        "session_id": request.session_id,
        "direction_id": request.direction_id,
        "stream_id": frame.stream_id,
        "utterance_id": frame.utterance_id,
    }
    return (
        ProviderTranscriptDelta(
            **identity, event_sequence=2, text="source", is_final=True
        ),
        ProviderTranslationDelta(
            **identity,
            event_sequence=3,
            text="translation",
            is_final=True,
            stable_prefix=True,
        ),
        ProviderAudioDelta(
            **identity,
            event_sequence=4,
            sequence=0,
            sample_rate_hz=24_000,
            channels=1,
            sample_format=SampleFormat.S16LE,
            frame_duration_ms=20,
            provider_monotonic_ns=1,
            pcm=b"\x01\x00" * 480,
        ),
        ProviderLatency(**identity, event_sequence=5, provider_total_ms=10),
        ProviderUtteranceFinal(
            **identity,
            event_sequence=6,
            final_audio_sequence=0,
            outcome=UtteranceOutcome.COMPLETED,
        ),
    )


def test_runtime_guard_rejects_other_checkout(monkeypatch):
    monkeypatch.setattr(baseline, "ORIGINAL_ROOT", Path("/nonexistent/original"))
    with pytest.raises(RuntimeError, match="original runtime"):
        baseline.assert_original_runtime()


def test_runtime_guard_binds_current_original_checkout():
    identities = baseline.assert_original_runtime()
    assert set(identities) == {
        "translator_sidecar",
        "translator_sidecar.provider_contract",
        "translator_sidecar.local.runtime",
        "translator_sidecar.local.local_provider",
    }
    assert all(len(value) == 64 for value in identities.values())


def _frozen_input(tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    clips = corpus / "clips"
    clips.mkdir(parents=True)
    wav = clips / "ru-1-clean.wav"
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x01\x02" * 1600)
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
    hashes = tuple(_sha(path.read_bytes()) for path in (manifest, screen, turbo))
    for name, digest in zip(
        ("MANIFEST_SHA256", "SCREEN_SHA256", "TURBO_SHA256"), hashes, strict=True
    ):
        monkeypatch.setattr(baseline, name, digest)
    return manifest, screen, turbo, wav, hashes


def _input_digest(cases):
    fields = (
        "origin_id",
        "condition",
        "language",
        "speaker_id",
        "wav_sha256",
        "reference",
        "turbo_text",
        "critical_labels",
    )
    rows = [
        {**{field: case[field] for field in fields}, "pcm_sha256": _sha(case["pcm"])}
        for case in cases
    ]
    return _sha(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode())


def test_frozen_input_matches_candidate_loader_in_separate_process(
    tmp_path: Path, monkeypatch
):
    manifest, screen, turbo, wav, hashes = _frozen_input(tmp_path, monkeypatch)
    cases = baseline.load_cases(manifest, screen, turbo)
    assert len(cases) == 1
    assert cases[0]["pcm"] == b"\x01\x02" * 1600
    fork = Path(__file__).resolve().parents[1]
    script = (
        "import hashlib,json,sys;from pathlib import Path;"
        "from translator_product_audio_pair import load_cases;"
        "m,s,t=map(Path,sys.argv[1:4]);"
        "cases=load_cases(m,s,t,manifest_sha256=sys.argv[4],"
        "screen_sha256=sys.argv[5],turbo_sha256=sys.argv[6]);"
        "fields=('origin_id','condition','language','speaker_id','wav_sha256',"
        "'reference','turbo_text','critical_labels');"
        "rows=[{**{field:c[field] for field in fields},"
        "'pcm_sha256':hashlib.sha256(c['pcm']).hexdigest()} for c in cases];"
        "print(hashlib.sha256(json.dumps(rows,ensure_ascii=False,"
        "sort_keys=True).encode()).hexdigest())"
    )
    candidate_digest = subprocess.check_output(
        [sys.executable, "-c", script, str(manifest), str(screen), str(turbo), *hashes],
        env={
            **os.environ,
            "PYTHONPATH": f"{fork / 'sidecar'}:{fork / 'scripts'}",
        },
        text=True,
    ).strip()
    assert candidate_digest == _input_digest(cases)
    wav.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen audio"):
        baseline.load_cases(manifest, screen, turbo)


def test_frozen_input_rejects_mismatched_selection(tmp_path: Path, monkeypatch):
    manifest, screen, turbo, _, _ = _frozen_input(tmp_path, monkeypatch)
    screen.write_text(
        json.dumps(
            {
                "turbo_report_sha256": _sha(turbo.read_bytes()),
                "cases": [{"origin_id": "ru-1", "condition": "noise"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "SCREEN_SHA256", _sha(screen.read_bytes()))
    with pytest.raises(ValueError, match="incomplete"):
        baseline.load_cases(manifest, screen, turbo)


def test_pinned_json_parses_the_bytes_it_hashed(tmp_path: Path, monkeypatch):
    path = tmp_path / "pinned.json"
    pinned = {"source": "pinned"}
    path.write_text(json.dumps(pinned), encoding="utf-8")
    original_read_text = Path.read_text

    def changed_second_read(candidate: Path, *args, **kwargs):
        if candidate == path:
            return json.dumps({"source": "changed"})
        return original_read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", changed_second_read)
    assert baseline._read_json(path, _sha(path.read_bytes())) == pinned


def test_private_manifest_changes_only_paths_and_copies_bytes(tmp_path: Path):
    output_dir = tmp_path / "private"
    output_dir.mkdir(mode=0o700)
    source = tmp_path / "source"
    (
        source
        / "huggingface/hub/models--Systran--faster-whisper-small/snapshots/revision"
    ).mkdir(parents=True)
    model = (
        source
        / "huggingface/hub/models--Systran--faster-whisper-small/snapshots/revision/model.bin"
    )
    model.write_bytes(b"private model bytes")
    manifest = {
        "schema_version": 1,
        "policy": {
            "staging_path": "/production/staging",
            "usage_mode": "personal_noncommercial",
        },
        "models": [
            {
                "id": baseline.ASR_ID,
                "source": {"revision": "revision"},
                "cache_path": "/production/model",
                "files": [
                    {
                        "path": "model.bin",
                        "size_bytes": model.stat().st_size,
                        "sha256": _sha(model.read_bytes()),
                    }
                ],
            }
        ],
    }
    projected, path = baseline.prepare_private_manifest(
        manifest, output_dir, source, required_ids=(baseline.ASR_ID,)
    )
    assert projected["models"][0]["cache_path"].startswith(str(output_dir))
    assert projected["policy"]["staging_path"].startswith(str(output_dir))
    assert json.loads(path.read_text(encoding="utf-8")) == projected
    copied = Path(projected["models"][0]["cache_path"]) / "model.bin"
    assert copied.read_bytes() == model.read_bytes()
    assert copied.stat().st_ino != model.stat().st_ino
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    original = json.loads(json.dumps(manifest))
    projected["models"][0]["cache_path"] = original["models"][0]["cache_path"]
    projected["policy"]["staging_path"] = original["policy"]["staging_path"]
    assert projected == original


def test_private_manifest_rejects_changed_source_before_runtime(tmp_path: Path):
    output_dir = tmp_path / "private"
    output_dir.mkdir(mode=0o700)
    source = tmp_path / "source"
    (
        source
        / "huggingface/hub/models--Systran--faster-whisper-small/snapshots/revision"
    ).mkdir(parents=True)
    (
        source
        / "huggingface/hub/models--Systran--faster-whisper-small/snapshots/revision/model.bin"
    ).write_bytes(b"bad")
    manifest = {
        "policy": {"staging_path": "/x"},
        "models": [
            {
                "id": baseline.ASR_ID,
                "source": {"revision": "revision"},
                "cache_path": "/x",
                "files": [{"path": "model.bin", "size_bytes": 3, "sha256": "0" * 64}],
            }
        ],
    }
    with pytest.raises(ValueError, match="checksum"):
        baseline.prepare_private_manifest(
            manifest, output_dir, source, required_ids=(baseline.ASR_ID,)
        )


def test_irina_source_layout_projects_to_private_waiver_path(tmp_path: Path):
    output_dir = tmp_path / "private"
    output_dir.mkdir(mode=0o700)
    source = tmp_path / "source"
    (source / "piper").mkdir(parents=True)
    (source / "piper/voice.onnx").write_bytes(b"voice")
    manifest = {
        "policy": {"staging_path": "/production/staging"},
        "models": [
            {
                "id": "piper-ru-irina-medium",
                "cache_path": "/production/cache/piper",
                "files": [
                    {"path": "voice.onnx", "size_bytes": 5, "sha256": _sha(b"voice")}
                ],
            }
        ],
    }
    projected, _ = baseline.prepare_private_manifest(
        manifest, output_dir, source, required_ids=("piper-ru-irina-medium",)
    )
    destination = Path(projected["models"][0]["cache_path"])
    assert destination.parts[-2:] == ("cache", "piper")
    assert (destination / "voice.onnx").read_bytes() == b"voice"


def test_private_manifest_rejects_linked_eval_source(tmp_path: Path):
    output_dir = tmp_path / "private"
    output_dir.mkdir(mode=0o700)
    source = tmp_path / "source"
    (source / "piper").mkdir(parents=True)
    voice = source / "piper/voice.onnx"
    voice.write_bytes(b"voice")
    os.link(voice, tmp_path / "linked-voice.onnx")
    manifest = {
        "policy": {"staging_path": "/production/staging"},
        "models": [
            {
                "id": "piper-ru-irina-medium",
                "cache_path": "/production/cache/piper",
                "files": [
                    {"path": "voice.onnx", "size_bytes": 5, "sha256": _sha(b"voice")}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="linked"):
        baseline.prepare_private_manifest(
            manifest, output_dir, source, required_ids=("piper-ru-irina-medium",)
        )


def test_real_original_api_waits_for_close_publications_and_health():
    observed = []

    class Provider:
        async def open_session(self, request, publish):
            self.request, self.publish = request, publish
            observed.append("open")
            return SimpleNamespace(session_id=request.session_id), _health(request)

        async def submit_frame(self, frame):
            if frame.end_of_utterance:
                await self.publish(
                    _events(self.request, frame), lambda: observed.append("commit")
                )

        async def wait_idle(self):
            observed.append("idle")

        async def health(self, session_id):
            observed.append("health")
            return _health(self.request)

        async def close_session(self, request):
            observed.append("close")
            assert request.session_id == self.request.session_id
            await self.publish(
                (
                    baseline.ProviderSessionClosed(
                        session_id=request.session_id,
                        direction_id=self.request.direction_id,
                        event_sequence=7,
                        reason=baseline.SessionCloseReason.USER_STOP,
                    ),
                ),
                lambda: observed.append("commit_close"),
            )

        async def wait_publications(self, session_id):
            observed.append("publications")
            assert session_id == self.request.session_id

    case = {"language": "ru_ru", "pcm": b"\x01\x00" * 1600}
    result = asyncio.run(
        baseline.run_case(
            Provider(), case, TranslationMode.QUALITY_FIRST, VoiceGender.FEMALE
        )
    )
    assert result["status"] == "completed"
    assert result["asr_text"] == "source"
    assert result["mt_text"] == "translation"
    assert observed[-3:] == ["close", "commit_close", "publications"]


def test_event_time_regression_fails_closed():
    request = baseline._request(
        "ru_ru", TranslationMode.QUALITY_FIRST, VoiceGender.FEMALE
    )
    frame = SimpleNamespace(stream_id=uuid4(), utterance_id=uuid4())
    events = _events(request, frame)
    times = [(event, 100 + index) for index, event in enumerate(events)]
    times[-1] = (events[-1], 99)
    with pytest.raises(ValueError, match="order"):
        baseline.evaluate_events(
            times,
            started_ns=90,
            session_id=request.session_id,
            direction_id=request.direction_id,
            stream_id=frame.stream_id,
            utterance_id=frame.utterance_id,
        )


def test_original_health_accepts_cpu_degraded_but_requires_ready_models():
    request = baseline._request(
        "ru_ru", TranslationMode.QUALITY_FIRST, VoiceGender.FEMALE
    )
    observed = baseline._checked_health(
        _health(request, state=ProviderState.DEGRADED, asr="small"),
        request,
        after=True,
    )
    assert observed["provider_state"] == "degraded"
    assert observed["models"]["asr"]["id"] == "small"
    with pytest.raises(RuntimeError, match="effective original model"):
        baseline._checked_health(_health(request, asr="other"), request, after=True)
    with pytest.raises(RuntimeError, match="effective original model"):
        baseline._checked_health(
            _health(request, asr=baseline.ASR_ID), request, after=True
        )


def test_failure_still_closes_and_checks_publication_drain():
    class Provider:
        async def open_session(self, request, publish):
            self.request = request
            return SimpleNamespace(session_id=request.session_id), _health(request)

        async def submit_frame(self, frame):
            raise ValueError("submit failed")

        async def close_session(self, request):
            self.closed = True

        async def wait_publications(self, session_id):
            raise RuntimeError("publication failed")

    provider = Provider()
    case = {"language": "ru_ru", "pcm": b"\x01\x00" * 1600}
    with pytest.raises(RuntimeError, match="publication failed"):
        asyncio.run(
            baseline.run_case(
                provider, case, TranslationMode.QUALITY_FIRST, VoiceGender.FEMALE
            )
        )
    assert provider.closed


def test_close_failure_is_retained_if_publication_drain_also_fails():
    class Provider:
        async def open_session(self, request, publish):
            self.request = request
            return SimpleNamespace(session_id=request.session_id), _health(request)

        async def submit_frame(self, frame):
            raise ValueError("submission failed")

        async def close_session(self, request):
            raise RuntimeError("close failed")

        async def wait_publications(self, session_id):
            raise RuntimeError("publication failed")

    case = {"language": "ru_ru", "pcm": b"\x01\x00" * 1600}
    with pytest.raises(RuntimeError, match="original session close failed") as caught:
        asyncio.run(
            baseline.run_case(
                Provider(), case, TranslationMode.QUALITY_FIRST, VoiceGender.FEMALE
            )
        )
    assert str(caught.value.__cause__) == "close failed"
    assert caught.value.drain_error_type == "RuntimeError"


def test_journal_is_exclusive_private_and_fsynced(tmp_path: Path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "receipt.jsonl"
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    with baseline.open_journal(path) as journal:
        baseline.write_record(journal, {"type": "terminal", "status": "failed"})
    assert calls
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        baseline.open_journal(path)
