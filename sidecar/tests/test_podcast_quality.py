from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from translator_sidecar.benchmark import podcast_quality, task6_live
from translator_sidecar.provider_contract import (
    Language,
    ProviderAudioDelta,
    SafeErrorCode,
    TranslationMode,
    VoiceGender,
)
from translator_sidecar.provider_registry import SessionDrainReceipt


def _forbid_cli_event_loop(coroutine, **_kwargs):
    coroutine.close()
    pytest.fail("podcast CLI created an asyncio Runner")


@pytest.mark.parametrize("failure", [None, "prepare", "worker"])
def test_podcast_main_uses_sync_supervisor_and_preserves_output(
    monkeypatch, tmp_path, capsys, failure
):
    from translator_sidecar.benchmark import process_run

    work_dir, output = tmp_path / "work", tmp_path / "out.json"
    output.write_text("previous")
    calls, prepared = [], []
    pcm = b"\0\1" * (podcast_quality._FRAME_BYTES // 2)
    result = {"models": [{"model_id": "synthetic", "status": "completed"}]}

    def convert(source, destination):
        prepared.append(source.name)
        if failure == "prepare":
            raise podcast_quality.PodcastQualityError("synthetic preparation failed")
        destination.write_bytes(pcm)
        return pcm

    def run(kind, request, destination, *, limits):
        calls.append(request)
        assert prepared == ["ru.wav", "en.wav"]
        assert kind == "podcast" and destination == output
        assert limits.model_run_seconds == 19 and limits.terminate_grace_seconds == 2
        assert request["asr_model_ids"] == ["faster-whisper-small"]
        assert request["tts_model_ids"] == ["piper-medium"]
        assert (
            request["mode"] == "quality_first" and request["voice_gender"] == "female"
        )
        assert request["inputs"]["segment_ms"] == 100
        assert request["inputs"]["max_segments"] == 1
        assert request["inputs"]["selected_tts_models"] == ["piper-medium"]
        assert set(request["pcm_paths"]) == {"ru_to_en", "en_to_ru"}
        for path in request["pcm_paths"].values():
            assert Path(path).is_absolute() and Path(path).read_bytes() == pcm
        json.dumps(request, allow_nan=False)
        if failure == "worker":
            raise process_run.BenchmarkRunError("synthetic worker failed")
        return result

    monkeypatch.setattr(podcast_quality, "_convert_to_pcm", convert)
    monkeypatch.setattr(process_run, "run_benchmark", run)
    monkeypatch.setattr(asyncio, "run", _forbid_cli_event_loop)
    monkeypatch.setattr(
        process_run,
        "run_benchmark_async",
        lambda *args, **kwargs: pytest.fail("CLI used async supervisor"),
    )
    monkeypatch.setattr(
        podcast_quality,
        "build_local_provider",
        lambda **kwargs: pytest.fail("CLI constructed provider"),
    )
    code = podcast_quality.main(
        [
            "--ru-audio",
            str(tmp_path / "ru.wav"),
            "--en-audio",
            str(tmp_path / "en.wav"),
            "--work-dir",
            str(work_dir),
            "--output",
            str(output),
            "--asr-model",
            "faster-whisper-small",
            "--tts-model",
            "piper-medium",
            "--mode",
            "quality_first",
            "--voice-gender",
            "female",
            "--segment-ms",
            "100",
            "--max-segments",
            "1",
            "--model-run-seconds",
            "19",
            "--terminate-grace-seconds",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert output.read_text() == "previous"
    if failure is None:
        assert code == 0 and len(calls) == 1 and not captured.err
        assert (
            captured.out
            == json.dumps(result["models"], ensure_ascii=False, indent=2)
            + f"\nreport={output}\n"
        )
    else:
        assert code == 2 and captured.out == ""
        message = "preparation" if failure == "prepare" else "worker"
        assert (
            captured.err
            == f"podcast quality diagnostics failed: synthetic {message} failed\n"
        )
        assert len(calls) == (0 if failure == "prepare" else 1)


def test_podcast_main_listing_has_no_preparation_or_supervisor(
    monkeypatch, tmp_path, capsys
):
    from translator_sidecar.benchmark import process_run

    def forbidden(*args, **kwargs):
        pytest.fail("listing created benchmark effects")

    monkeypatch.setattr(podcast_quality, "_load_audio_pcm", forbidden)
    monkeypatch.setattr(podcast_quality, "build_local_provider", forbidden)
    monkeypatch.setattr(process_run, "run_benchmark", forbidden)
    monkeypatch.setattr(process_run, "run_benchmark_async", forbidden)
    monkeypatch.setattr(asyncio, "run", _forbid_cli_event_loop)
    work_dir = tmp_path / "absent"
    assert podcast_quality.main(["--list-candidates", "--work-dir", str(work_dir)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == podcast_quality.registry_report()
    assert not captured.err and not work_dir.exists()


@pytest.mark.parametrize("option", ["--model-run-seconds", "--terminate-grace-seconds"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_podcast_main_invalid_limits_precede_directory_and_materialization(
    monkeypatch, tmp_path, option, value
):
    from translator_sidecar.benchmark import process_run

    def forbidden(*args, **kwargs):
        pytest.fail("invalid limits created benchmark effects")

    monkeypatch.setattr(podcast_quality, "_load_audio_pcm", forbidden)
    monkeypatch.setattr(podcast_quality, "build_local_provider", forbidden)
    monkeypatch.setattr(process_run, "run_benchmark", forbidden)
    monkeypatch.setattr(process_run, "run_benchmark_async", forbidden)
    monkeypatch.setattr(asyncio, "run", _forbid_cli_event_loop)
    work_dir, output = tmp_path / "absent", tmp_path / "result.json"
    with pytest.raises(ValueError):
        podcast_quality.main(
            [
                "--work-dir",
                str(work_dir),
                "--output",
                str(output),
                option,
                value,
            ]
        )
    assert not work_dir.exists() and not output.exists()


def test_podcast_public_run_materializes_before_scalar_request(monkeypatch, tmp_path):
    from translator_sidecar.benchmark import process_run

    events, captured = [], []
    limits = process_run.RunLimits(model_run_seconds=19, terminate_grace_seconds=2)
    args = podcast_quality._build_parser().parse_args(
        [
            "--ru-audio",
            str(tmp_path / "ru.wav"),
            "--en-audio",
            str(tmp_path / "en.wav"),
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "out.json"),
            "--asr-model",
            "faster-whisper-small",
            "--tts-model",
            "piper-medium",
            "--segment-ms",
            "100",
            "--max-segments",
            "1",
        ]
    )
    args.output.write_text("previous")
    pcm = b"\0\1" * (podcast_quality._FRAME_BYTES // 2)

    def convert(source, output):
        events.append(("prepare", source))
        output.write_bytes(pcm)
        return pcm

    async def run(kind, request, output, *, limits):
        assert events == [
            ("prepare", args.ru_audio),
            ("prepare", args.en_audio),
        ]
        assert kind == "podcast" and output == args.output
        assert set(request) == {
            "pcm_paths",
            "inputs",
            "asr_model_ids",
            "tts_model_ids",
            "mode",
            "voice_gender",
        }
        assert request["asr_model_ids"] == ["faster-whisper-small"]
        assert request["tts_model_ids"] == ["piper-medium"]
        assert (
            request["mode"] == args.mode
            and request["voice_gender"] == args.voice_gender
        )
        assert set(request["pcm_paths"]) == {"ru_to_en", "en_to_ru"}
        for direction, path in request["pcm_paths"].items():
            source = Path(path)
            assert source.is_absolute() and source.name == "source.s16le"
            assert source.read_bytes() == pcm
            assert request["inputs"][direction]["kind"] == "local_audio"
        assert request["inputs"]["segment_ms"] == 100
        assert request["inputs"]["max_segments"] == 1
        assert request["inputs"]["selected_tts_models"] == ["piper-medium"]
        json.dumps(request, allow_nan=False)
        captured.append((request, limits))
        return {"synthetic": "delegated"}

    monkeypatch.setattr(podcast_quality, "_convert_to_pcm", convert)
    monkeypatch.setattr(process_run, "run_benchmark_async", run)
    monkeypatch.setattr(
        podcast_quality,
        "build_local_provider",
        lambda **kwargs: pytest.fail("parent constructed provider"),
    )
    result = asyncio.run(podcast_quality.run(args, limits=limits))
    assert result == {"synthetic": "delegated"}
    assert len(captured) == 1 and captured[0][1] is limits
    assert args.output.read_text() == "previous"
    assert all(Path(path).exists() for path in captured[0][0]["pcm_paths"].values())


@pytest.mark.parametrize("fails", [False, True])
def test_podcast_worker_reads_prepared_pcm_and_returns_only_after_cleanup(
    monkeypatch, tmp_path, fails
):
    pcm = b"\0\1" * (podcast_quality._FRAME_BYTES // 2)
    paths = {}
    for direction in ("ru_to_en", "en_to_ru"):
        path = tmp_path / f"{direction}.s16le"
        path.write_bytes(pcm)
        paths[direction] = str(path)
    request = {
        "pcm_paths": paths,
        "inputs": {
            "segment_ms": 100,
            "max_segments": 1,
            "selected_tts_models": ["piper-medium"],
        },
        "asr_model_ids": ["faster-whisper-small"],
        "tts_model_ids": ["piper-medium"],
        "mode": TranslationMode.STREAMING_FIRST.value,
        "voice_gender": VoiceGender.MALE.value,
    }
    calls, fatal_calls = [], []
    failure = RuntimeError("synthetic terminal failure")

    def fatal(error):
        fatal_calls.append(error)
        raise _PodcastFatal()

    class Provider:
        async def shutdown(self):
            calls.append("shutdown")
            if fails:
                raise failure
            calls.append("terminal")

    async def segment(provider, direction, data, **kwargs):
        assert data == pcm
        assert kwargs["mode"] is TranslationMode.STREAMING_FIRST
        assert kwargs["voice_gender"] is VoiceGender.MALE
        calls.append(direction)
        return {"direction": direction, "synthetic_metric": 7}

    monkeypatch.setattr(
        podcast_quality, "build_local_provider", lambda **kwargs: Provider()
    )
    monkeypatch.setattr(podcast_quality, "_run_segment", segment)
    monkeypatch.setattr(
        podcast_quality, "_summarize", lambda rows: {"count": len(rows)}
    )
    monkeypatch.setattr(
        podcast_quality,
        "_load_audio_pcm",
        lambda **kwargs: pytest.fail("worker materialized input"),
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    if fails:
        with pytest.raises(_PodcastFatal):
            asyncio.run(podcast_quality._run_owned(request, fatal_cleanup=fatal))
        assert fatal_calls == [failure]
        assert calls == ["ru_to_en", "en_to_ru", "shutdown"]
    else:
        result = asyncio.run(podcast_quality._run_owned(request, fatal_cleanup=fatal))
        assert result["schema_version"] == "translator.podcast-quality-debug.v1"
        assert result["inputs"] == request["inputs"]
        assert result["models"][0]["summary"] == {"count": 2}
        assert [row["synthetic_metric"] for row in result["models"][0]["segments"]] == [
            7,
            7,
        ]
        assert calls == ["ru_to_en", "en_to_ru", "shutdown", "terminal"]
        assert not fatal_calls
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


class _PodcastFatal(BaseException):
    pass


@pytest.mark.parametrize("shutdown_fails", [False, True])
def test_podcast_model_has_one_disposal_owner_and_fatal_cleanup(
    monkeypatch, shutdown_fails
):
    calls, fatal_calls = [], []
    failure = RuntimeError("synthetic provider shutdown failure")

    def fatal(error):
        fatal_calls.append(error)
        raise _PodcastFatal()

    class Provider:
        _asr = SimpleNamespace(release=lambda: calls.append("asr"))
        _translator = SimpleNamespace(
            _translator=SimpleNamespace(unload_model=lambda: calls.append("mt"))
        )

        async def shutdown(self):
            calls.append("shutdown")
            if shutdown_fails:
                raise failure
            self._asr.release()
            self._translator._translator.unload_model()
            calls.append("tts")

    monkeypatch.setattr(
        podcast_quality, "build_local_provider", lambda **kwargs: Provider()
    )

    async def scenario():
        try:
            result = await podcast_quality._run_model(
                "faster-whisper-small",
                {},
                mode=TranslationMode.STREAMING_FIRST,
                voice_gender=VoiceGender.MALE,
                fatal_cleanup=fatal,
            )
        except BaseException as error:
            result = error
        if shutdown_fails:
            assert isinstance(result, _PodcastFatal)
            assert fatal_calls == [failure] and calls == ["shutdown"]
        else:
            assert result["status"] == "completed" and result["segments"] == []
            assert calls == ["shutdown", "asr", "mt", "tts"]
            assert not fatal_calls

    asyncio.run(scenario())


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
            fatal_cleanup=lambda error: pytest.fail(
                "unsupported model acquired resources"
            ),
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


class _BenchmarkReservation:
    def __init__(self, provider, request, publish):
        self.provider, self.request, self.publish = provider, request, publish

    async def open(self):
        self.provider.live.add(self)
        self.provider.check("open")

    async def drain(self, _reason):
        provider = self.provider
        provider.drains.append(self)
        provider.drain_entered.set()
        await provider.drain_release.wait()
        if provider.cleanup_error is not None:
            raise provider.cleanup_error
        provider.live.remove(self)
        return SessionDrainReceipt(self.request.session_id, provider.delivery_error)


class _BenchmarkResources:
    """Both caller APIs address the same independently tracked test resource."""

    def __init__(self, failure=None):
        self.failure = failure
        self.error = RuntimeError("synthetic operation failure")
        self.cleanup_error = None
        self.delivery_error = None
        self.original_cancellation = None
        self.healthy = object()
        self.live = {self.healthy}
        self.reservation = None
        self.drains, self.frames = [], []
        self.operation_entered = asyncio.Event()
        self.operation_release = asyncio.Event()
        self.drain_entered = asyncio.Event()
        self.drain_release = asyncio.Event()
        self.drain_release.set()

    def check(self, phase):
        if self.failure == phase:
            raise self.error

    def reserve_session(self, request, publish):
        self.check("reserve")
        assert self.reservation is None
        self.reservation = _BenchmarkReservation(self, request, publish)
        return self.reservation

    async def open_session(self, request, publish):
        await self.reserve_session(request, publish).open()

    async def close_session(self, request):
        assert request.session_id == self.reservation.request.session_id
        await self.reservation.drain(request.reason)

    async def wait_publications(self, _identity):
        pass

    async def submit_frame(self, frame):
        self.check("frame")
        self.frames.append(frame)

    async def wait_idle(self):
        self.check("wait")
        if self.failure == "cancel":
            self.operation_entered.set()
            try:
                await self.operation_release.wait()
            except asyncio.CancelledError as error:
                self.original_cancellation = error
                raise
        if self.frames:
            frame = self.frames[-1]
            await self.reservation.publish(
                (
                    ProviderAudioDelta(
                        session_id=frame.session_id,
                        direction_id=frame.direction_id,
                        stream_id=frame.stream_id,
                        utterance_id=frame.utterance_id,
                        sequence=0,
                        event_sequence=3,
                        provider_monotonic_ns=1,
                        sample_rate_hz=frame.sample_rate_hz,
                        channels=frame.channels,
                        sample_format=frame.sample_format,
                        frame_duration_ms=frame.frame_duration_ms,
                        pcm=frame.pcm,
                    ),
                ),
                lambda: None,
            )


async def _call_benchmark(kind, provider, *, empty=False):
    pcm = b"" if empty else b"\0" * podcast_quality._FRAME_BYTES
    if kind == "podcast":
        return await podcast_quality._run_segment(
            provider,
            "ru_to_en",
            pcm,
            segment_index=0,
            mode=TranslationMode.QUALITY_FIRST,
            voice_gender=VoiceGender.MALE,
        )
    return await task6_live._ProviderDuplexBridge._run_session(
        SimpleNamespace(_provider=provider, _source_pcm={Language.RU: pcm}),
        Language.RU,
        uuid4(),
    )


@pytest.mark.parametrize("kind", ["podcast", "task6"])
@pytest.mark.parametrize("phase", ["open", "frame", "wait", "reserve"])
def test_benchmark_failure_drains_only_its_exact_resource(kind, phase):
    async def scenario():
        provider = _BenchmarkResources(phase)
        with pytest.raises(RuntimeError) as caught:
            await _call_benchmark(kind, provider)
        assert caught.value is provider.error
        assert provider.live == {provider.healthy}
        assert provider.drains == ([] if phase == "reserve" else [provider.reservation])

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("kind", ["podcast", "task6"])
def test_benchmark_success_preserves_metrics_and_healthy_resource(kind):
    async def scenario():
        provider = _BenchmarkResources()
        result = await _call_benchmark(kind, provider)
        if kind == "podcast":
            assert result["segment_index"] == 0 and result["frame_count"] == 1
            assert result["audio_output_frames"] == 1
        else:
            assert isinstance(result, float) and result >= 0
        assert provider.live == {provider.healthy}
        assert provider.drains == [provider.reservation]

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("kind", ["podcast", "task6"])
@pytest.mark.parametrize("operation_failure", [False, True])
def test_benchmark_cleanup_failure_never_returns_metrics(kind, operation_failure):
    async def scenario():
        provider = _BenchmarkResources("wait" if operation_failure else None)
        provider.cleanup_error = RuntimeError("synthetic cleanup failure")
        with pytest.raises(RuntimeError) as caught:
            await _call_benchmark(kind, provider)
        if operation_failure:
            assert caught.value is provider.error
            assert caught.value.__cause__ is provider.cleanup_error
        else:
            assert caught.value is provider.cleanup_error
        assert provider.live == {provider.healthy, provider.reservation}
        assert provider.drains == [provider.reservation]

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("kind", ["podcast", "task6"])
def test_benchmark_cleanup_error_does_not_rethrow_callers_exception(kind):
    async def scenario():
        provider = _BenchmarkResources()
        provider.cleanup_error = RuntimeError("synthetic cleanup failure")
        try:
            raise ValueError("unrelated caller exception")
        except ValueError:
            with pytest.raises(RuntimeError) as caught:
                await _call_benchmark(kind, provider)
            assert caught.value is provider.cleanup_error
        assert provider.drains == [provider.reservation]

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("kind", ["podcast", "task6"])
def test_benchmark_delivery_failure_never_returns_metrics(kind):
    async def scenario():
        provider = _BenchmarkResources()
        provider.delivery_error = SafeErrorCode.PROVIDER_UNAVAILABLE
        with pytest.raises(RuntimeError, match="provider_unavailable"):
            await _call_benchmark(kind, provider)
        assert provider.live == {provider.healthy}
        assert provider.drains == [provider.reservation]

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("kind", ["podcast", "task6"])
def test_benchmark_repeated_cancellation_joins_entered_drain(kind):
    async def scenario():
        provider = _BenchmarkResources("cancel")
        provider.drain_release.clear()
        failures = []

        async def invoke():
            try:
                return await _call_benchmark(kind, provider)
            except BaseException as error:
                failures.append(error)
                raise

        operation = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(provider.operation_entered.wait(), timeout=1)
            operation.cancel()
            await asyncio.wait({operation}, timeout=0.02)
            operation.cancel()
            await asyncio.wait({operation}, timeout=0.02)
            snapshot = (
                operation.done(),
                provider.drain_entered.is_set(),
                tuple(provider.drains),
                frozenset(provider.live),
            )
        finally:
            provider.operation_release.set()
            provider.drain_release.set()
            outcome = await asyncio.gather(operation, return_exceptions=True)
        assert snapshot == (
            False,
            True,
            (provider.reservation,),
            frozenset((provider.healthy, provider.reservation)),
        )
        assert isinstance(outcome[0], asyncio.CancelledError)
        assert failures == [provider.original_cancellation]
        assert failures[0] is provider.original_cancellation
        assert provider.live == {provider.healthy}

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_task6_empty_pcm_drains_the_opened_resource():
    async def scenario():
        provider = _BenchmarkResources()
        with pytest.raises(RuntimeError, match="PCM is empty"):
            await _call_benchmark("task6", provider, empty=True)
        assert provider.live == {provider.healthy}
        assert provider.drains == [provider.reservation]

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("kind", ["podcast", "task6"])
@pytest.mark.parametrize("fail_frame", [False, True])
def test_benchmark_composes_with_local_reservation_and_preserves_peer(kind, fail_frame):
    from test_grpc_server import local_provider_fixture

    async def scenario():
        provider, *_ = local_provider_fixture()
        collector = podcast_quality._Collector()
        peer_request = podcast_quality._request(
            "en_to_ru",
            mode=TranslationMode.QUALITY_FIRST,
            voice_gender=VoiceGender.MALE,
        )
        peer = provider.reserve_session(peer_request, collector.publish)
        owners = []
        reserve = provider.reserve_session
        submit = provider.submit_frame
        error = RuntimeError("synthetic frame failure after admission")

        def record(request, publish):
            owner = reserve(request, publish)
            owners.append(owner)
            return owner

        async def fail_after_admission(frame):
            await submit(frame)
            raise error

        try:
            await peer.open()
            provider.reserve_session = record
            if fail_frame:
                provider.submit_frame = fail_after_admission
                with pytest.raises(RuntimeError) as caught:
                    await _call_benchmark(kind, provider)
                assert caught.value is error
            else:
                result = await _call_benchmark(kind, provider)
                if kind == "podcast":
                    assert result["frame_count"] == 1
                    assert result["audio_output_frames"] > 0
                else:
                    assert isinstance(result, float) and result >= 0
            assert len(owners) == 1 and owners[0] is not peer
            assert owners[0].closed and owners[0].drain_task.done()
            assert owners[0].publication_task.done()
            assert not provider._retired_sessions
            assert provider._sessions == {peer_request.session_id: peer}
            assert set(provider._scheduler._sessions) == {peer_request.session_id}
            assert not peer.closed
        finally:
            await provider.shutdown()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))
