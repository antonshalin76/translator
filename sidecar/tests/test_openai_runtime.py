from __future__ import annotations

import asyncio
import base64
import inspect
import json
import multiprocessing
import time
from pathlib import Path
from uuid import uuid4

import pytest
from test_provider_contract import VOICE_OVERRIDE_CASES
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from translator_sidecar.openai_provider import OpenAIRealtimeConfig
from translator_sidecar.openai_runtime import (
    OpenAIProviderProtocolError,
    OpenAIRealtimeProvider,
)
from translator_sidecar.provider_contract import (
    MAX_TERMINAL_UTTERANCES_PER_SESSION,
    AudioDirection,
    CancelReason,
    CancelUtterance,
    CloseRequestReason,
    Language,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderId,
    ProviderInputFrame,
    ProviderSessionClosed,
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


def wire(kind, **fields):
    return {"type": kind, "event_id": "event_synthetic", **fields}


class FakeRealtimeWebSocket:
    def __init__(self, fault=None):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.closed = False
        self.readers = 0
        self.fault = fault
        self.transport = self
        self.session = {
            "id": "sess_synthetic",
            "type": "translation",
            "model": "gpt-realtime-translate",
            "expires_at": 2_000_000_000,
            "audio": {},
        }
        self.push(wire("session.created", session=self.session))

    async def send(self, raw):
        event = json.loads(raw)
        if self.fault == "send":
            raise OSError("private-transport-error")
        self.sent.append(event)
        if event["type"] == "session.update":
            self.session = {**self.session, **event["session"]}
            if self.fault == "handshake":
                self.session["model"] = "wrong"
            self.push(wire("session.updated", session=self.session))

    async def recv(self):
        self.readers += 1
        try:
            value = await self.incoming.get()
            if isinstance(value, BaseException):
                raise value
            return value
        finally:
            self.readers -= 1

    async def close(self):
        self.closed = True

    def abort(self):
        self.closed = True

    def push(self, event):
        self.incoming.put_nowait(json.dumps(event))


class Harness:
    def __init__(self, *, fault=None, timeout=1):
        self.sockets = []
        self.events = []
        self.fault = fault
        self.connect_gate = None
        self.provider = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            environ={"OPENAI_API_KEY": "synthetic-secret"},
            websocket_factory=self.connect,
            drain_timeout_seconds=timeout,
        )

    async def connect(self, uri, **options):
        assert (
            uri
            == "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
        )
        assert (
            options["additional_headers"]["Authorization"] == "Bearer synthetic-secret"
        )
        if self.connect_gate is not None:
            await self.connect_gate.wait()
        ws = FakeRealtimeWebSocket(self.fault)
        self.sockets.append(ws)
        return ws

    async def publish(self, batch, commit):
        self.events.extend(batch)
        commit()

    async def open(self, *, debug=False):
        request = open_request(debug=debug)
        await self.provider.reserve_session(request, self.publish).open()
        return request

    async def settled(self):
        for _ in range(200):
            if all(s.generation is None for s in self.provider._sessions.values()):
                return
            await asyncio.sleep(0.001)
        raise AssertionError("generation did not retire")

    def assert_empty_generations(self):
        assert all(s.generation is None for s in self.provider._sessions.values())
        assert not self.provider._opening
        assert all(ws.closed and ws.readers == 0 for ws in self.sockets)


@pytest.mark.parametrize("model_path,provider_voice_id", VOICE_OVERRIDE_CASES)
def test_openai_voice_override_refused_before_reservation_without_peer_effects(
    model_path, provider_voice_id
):
    async def scenario():
        h = Harness()
        peer = await h.open()
        peer_owner = h.provider._sessions[peer.session_id]
        value = open_request()
        invalid = value.model_copy(
            update={
                "voice_profile": value.voice_profile.model_copy(
                    update={
                        "model_path": model_path,
                        "provider_voice_id": provider_voice_id,
                    }
                )
            }
        )

        def snapshot():
            return (
                tuple(
                    frozenset((key, id(owner)) for key, owner in owners.items())
                    for owners in (
                        h.provider._sessions,
                        h.provider._failed_opens,
                        h.provider._opening,
                    )
                ),
                frozenset(asyncio.all_tasks()),
                tuple(
                    (
                        id(ws),
                        ws.closed,
                        ws.readers,
                        tuple(json.dumps(event, sort_keys=True) for event in ws.sent),
                    )
                    for ws in h.sockets
                ),
                tuple(h.events),
            )

        try:
            before = snapshot()
            with pytest.raises(OpenAIProviderProtocolError, match="override"):
                h.provider.reserve_session(invalid, h.publish)
            assert snapshot() == before
            assert h.provider._sessions[peer.session_id] is peer_owner
            repaired = h.provider.reserve_session(value, h.publish)
            opened, _ = await repaired.open()
            assert opened.session_id == value.session_id
            assert (
                await repaired.drain(CloseRequestReason.USER_STOP)
            ).delivery_error is None
            assert h.provider._sessions[peer.session_id] is peer_owner
            assert not h.sockets[0].closed
        finally:
            await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "cause,expected",
    [
        ("closed", "provider_closed"),
        ("duplicate", "duplicate_session"),
        ("pending", "provider_cleanup_required"),
        ("input_pcm", "unsupported_pcm_format"),
        ("output_pcm", "unsupported_pcm_format"),
    ],
)
def test_openai_voice_override_preserves_existing_reserve_precedence(cause, expected):
    async def scenario():
        h = Harness()
        value = open_request()
        socket = None
        try:
            if cause in {"duplicate", "pending"}:
                owner = h.provider.reserve_session(value, h.publish)
                await owner.open()
                if cause == "pending":
                    socket = h.sockets[0]
                    close, abort = socket.close, socket.abort

                    async def fail_close():
                        raise OSError("synthetic unresolved socket close")

                    def fail_abort():
                        raise OSError("synthetic unresolved socket abort")

                    socket.close, socket.abort = fail_close, fail_abort
                    with pytest.raises(OpenAIProviderProtocolError):
                        await owner.drain(CloseRequestReason.USER_STOP)
                    value = value.model_copy(update={"session_id": uuid4()})
            elif cause == "closed":
                await h.provider.shutdown()
            else:
                field = (
                    "requested_input_format"
                    if cause == "input_pcm"
                    else "requested_output_format"
                )
                value = value.model_copy(
                    update={
                        field: getattr(value, field).model_copy(update={"channels": 2})
                    }
                )
            value = value.model_copy(
                update={
                    "voice_profile": value.voice_profile.model_copy(
                        update={"provider_voice_id": "alloy"}
                    )
                }
            )
            before = tuple(
                frozenset((key, id(owner)) for key, owner in owners.items())
                for owners in (
                    h.provider._sessions,
                    h.provider._failed_opens,
                    h.provider._opening,
                )
            )
            with pytest.raises(OpenAIProviderProtocolError, match=expected):
                h.provider.reserve_session(value, h.publish)
            assert (
                tuple(
                    frozenset((key, id(owner)) for key, owner in owners.items())
                    for owners in (
                        h.provider._sessions,
                        h.provider._failed_opens,
                        h.provider._opening,
                    )
                )
                == before
            )
        finally:
            if socket is not None:
                socket.close, socket.abort = close, abort
            await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("consent,credential", [(False, True), (True, False)])
def test_openai_voice_override_reserve_rejects_before_later_auth_preflight(
    consent, credential
):
    async def scenario():
        h = Harness()
        h.provider = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=consent),
            environ={"OPENAI_API_KEY": "synthetic-secret"} if credential else {},
            websocket_factory=h.connect,
        )
        value = open_request()
        value = value.model_copy(
            update={
                "voice_profile": value.voice_profile.model_copy(
                    update={"provider_voice_id": "alloy"}
                )
            }
        )
        try:
            before = frozenset(asyncio.all_tasks())
            with pytest.raises(OpenAIProviderProtocolError, match="override"):
                h.provider.reserve_session(value, h.publish)
            assert (
                not h.provider._sessions
                and not h.provider._failed_opens
                and not h.provider._opening
            )
            assert not h.sockets and not h.events
            assert frozenset(asyncio.all_tasks()) == before
        finally:
            await h.provider.shutdown()

    asyncio.run(scenario())


def open_request(*, debug=False):
    pcm = PcmFormat(
        sample_rate_hz=16000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=20,
    )
    return OpenProviderSession(
        session_id=uuid4(),
        provider_id=ProviderId.OPENAI,
        direction_id=AudioDirection.MICROPHONE,
        source_language=Language.RU,
        target_language=Language.EN,
        mode=TranslationMode.STREAMING_FIRST,
        requested_input_format=pcm,
        requested_output_format=pcm,
        voice_profile=VoiceProfile(
            language=Language.EN, gender=VoiceGender.MALE, engine=VoiceEngine.OPENAI
        ),
        debug_text_enabled=debug,
    )


def frame(request, *, final=True, sequence=0):
    return ProviderInputFrame(
        session_id=request.session_id,
        direction_id=request.direction_id,
        stream_id=request.session_id,
        utterance_id=uuid4(),
        sequence=sequence,
        capture_monotonic_ns=10_000_000,
        sample_rate_hz=16000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=20,
        source_language=request.source_language,
        target_language=request.target_language,
        mode=request.mode,
        pcm=bytes(640),
        end_of_utterance=final,
    )


def audio(pcm=bytes(960)):
    return wire(
        "session.output_audio.delta", delta=base64.b64encode(pcm).decode("ascii")
    )


def cancel(value):
    return CancelUtterance(
        session_id=value.session_id,
        direction_id=value.direction_id,
        utterance_id=value.utterance_id,
        reason=CancelReason.USER_INTERRUPT,
    )


def test_close_drains_audio_and_requires_server_terminal():
    async def scenario():
        h = Harness()
        request = await h.open()
        value = frame(request)
        await h.provider.submit_frame(value)
        ws = h.sockets[0]
        assert [e["type"] for e in ws.sent] == [
            "session.update",
            "session.input_audio_buffer.append",
            "session.close",
        ]
        assert len(base64.b64decode(ws.sent[1]["audio"])) == 960
        ws.push(audio())
        ws.push(wire("session.output_transcript.delta", delta="private"))
        await asyncio.sleep(0.01)
        assert not any(isinstance(e, ProviderUtteranceFinal) for e in h.events)
        ws.push(audio())
        ws.push(wire("session.closed"))
        await h.settled()
        outputs = [e for e in h.events if isinstance(e, ProviderAudioDelta)]
        assert len(outputs) == 2
        assert [e.sequence for e in outputs] == [0, 1]
        assert all(
            e.utterance_id == value.utterance_id and len(e.pcm) == 640 for e in outputs
        )
        assert not any(isinstance(e, ProviderTranslationDelta) for e in h.events)
        assert h.events[-1].outcome == UtteranceOutcome.COMPLETED
        h.assert_empty_generations()
        await h.provider.shutdown()
        assert not h.provider._sessions

    asyncio.run(scenario())


def test_cancel_reaps_a_before_b_and_late_a_cannot_be_relabeled():
    async def scenario():
        h = Harness()
        request = await h.open()
        a = frame(request, final=False)
        await h.provider.submit_frame(a)
        await asyncio.sleep(0)
        await h.provider.cancel_utterance(cancel(a))
        h.assert_empty_generations()
        h.sockets[0].push(audio(b"\x01\x00" * 480))
        b = frame(request, sequence=1)
        await h.provider.submit_frame(b)
        assert len(h.sockets) == 2
        h.sockets[1].push(audio(b"\x02\x00" * 480))
        h.sockets[1].push(wire("session.closed"))
        await h.settled()
        outputs = [e for e in h.events if isinstance(e, ProviderAudioDelta)]
        assert len(outputs) == 1 and outputs[0].utterance_id == b.utterance_id
        assert b"\x02\x00" in outputs[0].pcm
        h.assert_empty_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault", ["send", "handshake", "task", "opening_task", "insert"]
)
def test_failed_open_100_iterations_release_every_resource(fault, monkeypatch):
    async def scenario():
        h = Harness(fault=fault)

        class FailingRegistry(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("registry_insert_failed")

        if fault == "insert":
            h.provider._sessions = FailingRegistry()
        create_task = asyncio.create_task
        rejected = []
        if fault in {"task", "opening_task"}:
            target = "_receive" if fault == "task" else "_run_open_session"

            def fail_task(coroutine, **kwargs):
                if coroutine.cr_code.co_name == target:
                    rejected.append(coroutine)
                    raise RuntimeError("task_creation_failed")
                return create_task(coroutine, **kwargs)

            monkeypatch.setattr(asyncio, "create_task", fail_task)
        tasks_before = asyncio.all_tasks()
        try:
            for index in range(100):
                with pytest.raises((OpenAIProviderProtocolError, RuntimeError)):
                    await asyncio.wait_for(h.open(), timeout=2)
                if fault in {"task", "opening_task"}:
                    assert len(rejected) == index + 1
                    assert (
                        inspect.getcoroutinestate(rejected[-1]) == inspect.CORO_CLOSED
                    )
                    assert len(h.sockets) == (index + 1 if fault == "task" else 0)
                assert not h.provider._sessions
                assert not h.provider._failed_opens
                h.assert_empty_generations()
            assert asyncio.all_tasks() == tasks_before
        finally:
            monkeypatch.setattr(asyncio, "create_task", create_task)
            for coroutine in rejected:
                coroutine.close()
            await h.provider.shutdown()
            await h.provider.shutdown()

    asyncio.run(scenario())


def test_concurrent_duplicate_open_reserves_before_connect_100_iterations():
    async def scenario():
        h = Harness()
        for _ in range(100):
            h.connect_gate = asyncio.Event()
            request = open_request()
            reservation = h.provider.reserve_session(request, h.publish)
            task = asyncio.create_task(reservation.open())
            await asyncio.sleep(0)
            with pytest.raises(OpenAIProviderProtocolError, match="duplicate_session"):
                await h.provider.reserve_session(request, h.publish).open()
            h.connect_gate.set()
            await task
            await reservation.drain(CloseRequestReason.USER_STOP)
            h.assert_empty_generations()
        assert len(h.sockets) == 100
        assert not h.provider._sessions

    asyncio.run(scenario())


def test_thousand_terminal_generations_leave_no_per_utterance_state():
    async def scenario():
        h = Harness()
        request = await h.open()
        for index in range(1000):
            value = frame(request, sequence=index)
            await h.provider.submit_frame(value)
            generation = h.provider._sessions[request.session_id].generation
            if index % 3 == 0:
                h.sockets[-1].push(audio())
                h.sockets[-1].push(wire("session.closed"))
            elif index % 3 == 1:
                await h.provider.cancel_utterance(cancel(value))
            else:
                h.sockets[-1].incoming.put_nowait(OSError("private"))
            await h.settled()
            h.assert_empty_generations()
            assert not generation.adapter._audio_sequences
            assert not generation.adapter._audio_remainders
            assert not generation.adapter._event_sequences
            assert generation.receiver is None and generation.binding is None
        finals = [e for e in h.events if isinstance(e, ProviderUtteranceFinal)]
        assert len(finals) == 1000
        assert len({e.utterance_id for e in finals}) == 1000
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad",
    [
        "{",
        "[]",
        json.dumps(wire("session.output_audio.delta", delta="!")),
        json.dumps(wire("session.output_audio.delta", delta=1)),
        json.dumps(wire("session.output_audio.delta", delta="AA==", sample_rate=16000)),
        json.dumps(wire("session.input_transcript.delta", delta="unconfigured")),
        json.dumps({"type": "session.closed"}),
    ],
)
def test_documented_event_drift_fails_generation_closed(bad):
    async def scenario():
        h = Harness()
        request = await h.open()
        await h.provider.submit_frame(frame(request))
        h.sockets[0].incoming.put_nowait(bad)
        await h.settled()
        assert any(isinstance(e, PrivacySafeProviderError) for e in h.events)
        assert h.events[-1].outcome == UtteranceOutcome.DROPPED
        h.assert_empty_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_replay_split_audio_debug_error_unknown_and_partial_tail():
    async def scenario():
        h = Harness()
        request = await h.open(debug=True)
        value = frame(request, final=False)
        await h.provider.submit_frame(value)
        ws = h.sockets[0]
        assert (
            ws.sent[0]["session"]["audio"]["input"]["transcription"]["model"]
            == "gpt-realtime-whisper"
        )
        ws.push(wire("future.event", payload={"untrusted": True}))
        ws.push(
            wire("error", error={"type": "invalid_request_error", "message": "private"})
        )
        ws.push(wire("session.input_transcript.delta", delta="source"))
        ws.push(wire("session.output_transcript.delta", delta="target"))
        ws.push(audio(bytes(111)))
        ws.push(audio(bytes(1009)))
        await h.provider.submit_frame(
            value.model_copy(update={"sequence": 1, "end_of_utterance": True})
        )
        ws.push(wire("session.closed"))
        await h.settled()
        assert any(isinstance(e, ProviderTranscriptDelta) for e in h.events)
        assert any(isinstance(e, ProviderTranslationDelta) for e in h.events)
        assert len([e for e in h.events if isinstance(e, ProviderAudioDelta)]) == 2
        assert h.events[-1].outcome == UtteranceOutcome.COMPLETED
        seq = [e.event_sequence for e in h.events]
        assert seq == sorted(set(seq))
        assert "private" not in str(h.events)
        h.assert_empty_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_no_close_ack_has_bounded_failure_not_success():
    async def scenario():
        h = Harness(timeout=0.01)
        request = await h.open()
        await h.provider.submit_frame(frame(request))
        h.sockets[0].push(audio())
        await h.settled()
        assert h.events[-1].outcome == UtteranceOutcome.DROPPED
        h.assert_empty_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_shutdown_cancels_pending_open_without_publishing_ready():
    async def scenario():
        h = Harness()
        h.connect_gate = asyncio.Event()
        task = asyncio.create_task(h.open())
        await asyncio.sleep(0)
        await h.provider.shutdown()
        assert task.cancelled()
        assert not h.events and not h.sockets
        assert not h.provider._opening and not h.provider._sessions

    asyncio.run(scenario())


def test_real_async_transport_failed_handshake_releases_sockets_and_fds():
    async def scenario():
        async def server(ws):
            await ws.send(
                json.dumps(wire("session.created", session={"invalid": True}))
            )
            await ws.wait_closed()

        async with serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            h = Harness()

            async def local_connect(_uri, **kwargs):
                return await connect(f"ws://127.0.0.1:{port}", proxy=None, **kwargs)

            h.provider._connect = local_connect
            before_fds = len(list(Path("/proc/self/fd").iterdir()))
            before_tasks = asyncio.all_tasks()
            for _ in range(100):
                with pytest.raises(
                    OpenAIProviderProtocolError, match="openai_connection_failed"
                ):
                    await h.open()
                for _ in range(100):
                    if not listener.connections and asyncio.all_tasks() == before_tasks:
                        break
                    await asyncio.sleep(0.001)
                assert len(list(Path("/proc/self/fd").iterdir())) == before_fds
                assert asyncio.all_tasks() == before_tasks
                assert not h.provider._opening and not h.provider._sessions
            await h.provider.shutdown()

    asyncio.run(scenario())


def test_cancellation_during_post_connect_handshake_closes_socket():
    async def scenario():
        h = Harness()
        ws = FakeRealtimeWebSocket()
        ws.incoming.get_nowait()

        async def stalled_connect(*args, **kwargs):
            h.sockets.append(ws)
            return ws

        h.provider._connect = stalled_connect
        task = asyncio.create_task(h.open())
        async with asyncio.timeout(1):
            while ws.readers != 1:
                await asyncio.sleep(0)
        assert ws.readers == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        h.assert_empty_generations()
        assert not h.provider._sessions

    asyncio.run(scenario())


def test_cancellation_during_retirement_still_aborts_and_releases_generation():
    async def scenario():
        h = Harness()
        request = await h.open()
        value = frame(request)
        await h.provider.submit_frame(value)
        closing = asyncio.Event()

        async def stalled_close():
            closing.set()
            await asyncio.Event().wait()

        h.sockets[0].close = stalled_close
        task = asyncio.create_task(h.provider.cancel_utterance(cancel(value)))
        await closing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        h.assert_empty_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_supersession_retires_generation_before_new_input():
    async def scenario():
        h = Harness()
        request = await h.open()
        a, b = frame(request), frame(request, sequence=1)
        await h.provider.submit_frame(a)
        await h.provider.submit_frame(b)
        assert h.sockets[0].closed and h.sockets[0].readers == 0
        assert h.events[-1].utterance_id == a.utterance_id
        assert h.events[-1].outcome == UtteranceOutcome.CANCELLED
        assert len(h.sockets[1].sent) == 3
        await h.provider.shutdown()
        h.assert_empty_generations()

    asyncio.run(scenario())


def test_noncommitting_publication_fails_logical_session_and_releases_generation():
    async def scenario():
        h = Harness()

        async def reject(batch, commit):
            pass

        h.publish = reject
        request = await h.open()
        reservation = h.provider._sessions[request.session_id]
        await h.provider.submit_frame(frame(request))
        h.sockets[0].push(audio())
        await h.settled()
        h.assert_empty_generations()
        with pytest.raises(OpenAIProviderProtocolError, match="open_session_required"):
            await h.provider.submit_frame(frame(request))
        receipt = await reservation.drain(CloseRequestReason.USER_STOP)
        assert receipt.session_id == request.session_id
        assert receipt.delivery_error is SafeErrorCode.PROVIDER_UNAVAILABLE
        await h.provider.shutdown()
        assert not h.provider._sessions
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_capture_sequence_is_monotonic_across_utterances_and_vad_gaps():
    async def scenario():
        h = Harness()
        request = await h.open()
        for sequence in (42, 43, 87):
            value = frame(request, sequence=sequence)
            await h.provider.submit_frame(value)
            h.sockets[-1].push(wire("session.closed"))
            await h.settled()
        assert len(h.sockets) == 3
        for sequence in (87, 86, 0):
            with pytest.raises(
                OpenAIProviderProtocolError, match="input_sequence_mismatch"
            ):
                await h.provider.submit_frame(frame(request, sequence=sequence))
        assert len(h.sockets) == 3
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "updates",
    [
        {"source_language": Language.EN},
        {"target_language": Language.RU},
        {"direction_id": AudioDirection.SPEAKER},
        {"stream_id": uuid4()},
        {"sample_rate_hz": 24000},
        {"channels": 2},
        {"frame_duration_ms": 40},
    ],
)
def test_frame_contract_mismatch_does_not_mutate_or_replace_generation(updates):
    async def scenario():
        h = Harness()
        request = await h.open()
        value = frame(request, final=False, sequence=50)
        await h.provider.submit_frame(value)
        sent = len(h.sockets[0].sent)
        with pytest.raises(OpenAIProviderProtocolError):
            await h.provider.submit_frame(
                value.model_copy(update={"sequence": 51, **updates})
            )
        assert len(h.sockets) == 1 and len(h.sockets[0].sent) == sent
        assert not h.events
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("completed", [True, False])
@pytest.mark.parametrize("intervening_b", [True, False])
def test_terminal_uuid_replay_with_new_capture_sequence_is_rejected(
    completed, intervening_b
):
    async def scenario():
        h = Harness()
        request = await h.open()
        a = frame(request, sequence=10)
        await h.provider.submit_frame(a)
        if completed:
            h.sockets[-1].push(audio())
            h.sockets[-1].push(wire("session.closed"))
            await h.settled()
        else:
            await h.provider.cancel_utterance(cancel(a))
        if intervening_b:
            await h.provider.submit_frame(frame(request, sequence=11))
            h.sockets[-1].push(audio())
            h.sockets[-1].push(wire("session.closed"))
            await h.settled()
        sockets, events = len(h.sockets), len(h.events)
        with pytest.raises(
            OpenAIProviderProtocolError, match="utterance_already_terminal"
        ):
            await h.provider.submit_frame(a.model_copy(update={"sequence": 12}))
        assert len(h.sockets) == sockets and len(h.events) == events
        finals = [
            e
            for e in h.events
            if isinstance(e, ProviderUtteranceFinal)
            and e.utterance_id == a.utterance_id
        ]
        assert len(finals) == 1
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("active", [False, True])
def test_terminal_identity_capacity_fails_closed_without_eviction_or_socket_mutation(
    active,
):
    async def scenario():
        h = Harness()
        request = await h.open()
        session = h.provider._sessions[request.session_id]
        session.terminal_ids.update(
            uuid4() for _ in range(MAX_TERMINAL_UTTERANCES_PER_SESSION - 1)
        )
        value = frame(request, final=not active)
        await h.provider.submit_frame(value)
        if not active:
            h.sockets[-1].push(wire("session.closed"))
            await h.settled()
        tombstones = session.terminal_ids.copy()
        sent = h.sockets[0].sent.copy()
        with pytest.raises(
            OpenAIProviderProtocolError, match="terminal_identity_capacity_exhausted"
        ):
            await h.provider.submit_frame(frame(request, sequence=1))
        with pytest.raises(
            OpenAIProviderProtocolError, match="utterance_already_terminal"
        ):
            await h.provider.submit_frame(
                frame(request, sequence=2).model_copy(
                    update={"utterance_id": next(iter(tombstones))}
                )
            )
        assert session.terminal_ids == tombstones
        assert len(h.sockets) == 1 and h.sockets[0].sent == sent
        await h.provider.shutdown()
        assert len(session.terminal_ids) == MAX_TERMINAL_UTTERANCES_PER_SESSION

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_cleanup", [False, True])
def test_failed_close_and_abort_retain_owner_until_shutdown_retry(cancel_cleanup):
    async def scenario():
        h = Harness()
        request = await h.open()
        value = frame(request, final=False)
        await h.provider.submit_frame(value)
        ws = h.sockets[0]
        original_close, original_abort = ws.close, ws.abort
        entered, release = asyncio.Event(), asyncio.Event()
        if not cancel_cleanup:
            release.set()

        async def fail_close():
            entered.set()
            await release.wait()
            raise OSError("private close error")

        def fail_abort():
            raise OSError("private abort error")

        ws.close, ws.abort = fail_close, fail_abort
        task = asyncio.create_task(h.provider.cancel_utterance(cancel(value)))
        await entered.wait()
        if cancel_cleanup:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            release.set()
        with pytest.raises(
            OpenAIProviderProtocolError, match="openai_retirement_failed"
        ):
            await task
        session = h.provider._sessions[request.session_id]
        assert session.generation is not None and session.generation.revoked
        assert not session.generation.disposed and not ws.closed
        assert not session.generation.adapter._audio_sequences
        assert not session.generation.adapter._audio_remainders
        assert not h.events
        with pytest.raises(OpenAIProviderProtocolError, match="open_session_required"):
            await h.provider.submit_frame(frame(request, sequence=1))
        with pytest.raises(OpenAIProviderProtocolError, match="openai_shutdown_failed"):
            await h.provider.shutdown()
        assert request.session_id in h.provider._sessions and not ws.closed
        ws.close, ws.abort = original_close, original_abort
        await h.provider.shutdown()
        assert ws.closed and not h.provider._sessions
        assert len([e for e in h.events if isinstance(e, ProviderUtteranceFinal)]) == 1
        assert len([e for e in h.events if isinstance(e, ProviderSessionClosed)]) == 1
        events = len(h.events)
        await h.provider.shutdown()
        assert len(h.events) == events

    asyncio.run(scenario())


def test_failed_open_disposal_is_owned_and_shutdown_retries_without_publication():
    async def scenario():
        h = Harness()
        ws = FakeRealtimeWebSocket("send")
        original_close, original_abort = ws.close, ws.abort

        async def fail_close():
            raise OSError("private close error")

        def fail_abort():
            raise OSError("private abort error")

        ws.close, ws.abort = fail_close, fail_abort

        async def connect_broken(*args, **kwargs):
            h.sockets.append(ws)
            return ws

        h.provider._connect = connect_broken
        with pytest.raises(OpenAIProviderProtocolError):
            await h.open()
        assert not h.provider._sessions and len(h.provider._failed_opens) == 1
        assert not ws.closed and not h.events
        with pytest.raises(
            OpenAIProviderProtocolError, match="provider_cleanup_required"
        ):
            await h.open()
        assert len(h.sockets) == 1
        with pytest.raises(OpenAIProviderProtocolError, match="openai_shutdown_failed"):
            await h.provider.shutdown()
        assert len(h.provider._failed_opens) == 1 and not ws.closed
        ws.close, ws.abort = original_close, original_abort
        await h.provider.shutdown()
        assert ws.closed and not h.provider._failed_opens and not h.events

    asyncio.run(scenario())


def test_failed_open_quarantine_does_not_interrupt_existing_healthy_session():
    async def scenario():
        h = Harness()
        healthy = await h.open()
        ws = FakeRealtimeWebSocket("send")

        async def fail_close():
            raise OSError("synthetic_close_failure")

        def fail_abort():
            raise OSError("synthetic_abort_failure")

        ws.close, ws.abort = fail_close, fail_abort

        async def connect_broken(*args, **kwargs):
            h.sockets.append(ws)
            return ws

        h.provider._connect = connect_broken
        with pytest.raises(OpenAIProviderProtocolError):
            await h.open()
        with pytest.raises(
            OpenAIProviderProtocolError, match="provider_cleanup_required"
        ):
            await h.open()
        assert len(h.sockets) == 2
        await h.provider.submit_frame(frame(healthy))
        h.sockets[0].push(audio())
        h.sockets[0].push(wire("session.closed"))
        await h.settled()
        assert h.events[-1].outcome == UtteranceOutcome.COMPLETED
        ws.close = FakeRealtimeWebSocket.close.__get__(ws)
        ws.abort = FakeRealtimeWebSocket.abort.__get__(ws)
        await h.provider.shutdown()
        assert not h.provider._sessions and not h.provider._failed_opens

    asyncio.run(scenario())


@pytest.mark.parametrize("failed_open", [True, False], ids=["opening", "established"])
def test_unresolved_cleanup_refuses_new_uuid_without_growing_retained_resources(
    failed_open,
):
    async def scenario():
        h = Harness()
        healthy = await h.open()
        bad_request = open_request()
        bad_socket = FakeRealtimeWebSocket("send" if failed_open else None)
        original_close, original_abort = bad_socket.close, bad_socket.abort
        connects = []
        cleanup_effects = []

        async def first_bad_then_normal(uri, **options):
            connects.append(uri)
            if len(connects) == 1:
                h.sockets.append(bad_socket)
                return bad_socket
            return await h.connect(uri, **options)

        async def fail_close():
            cleanup_effects.append("close")
            raise OSError("synthetic close failure")

        def fail_abort():
            cleanup_effects.append("abort")
            raise OSError("synthetic abort failure")

        h.provider._connect = first_bad_then_normal
        reservation = h.provider.reserve_session(bad_request, h.publish)
        try:
            if failed_open:
                bad_socket.close, bad_socket.abort = fail_close, fail_abort
                with pytest.raises(OpenAIProviderProtocolError):
                    await reservation.open()
            else:
                await reservation.open()
                bad_socket.close, bad_socket.abort = fail_close, fail_abort
                with pytest.raises(OpenAIProviderProtocolError):
                    await reservation.drain(CloseRequestReason.DAEMON_SHUTDOWN)
            await asyncio.sleep(0)

            def retained_counts():
                return (
                    len(h.provider._sessions),
                    len(h.provider._failed_opens),
                    len(h.provider._opening),
                    len(h.sockets),
                    len(connects),
                    len(asyncio.all_tasks()),
                    tuple(cleanup_effects),
                )

            before = retained_counts()
            observations = []
            attempted_ids = set()
            for _ in range(100):
                candidate = open_request()
                attempted_ids.add(candidate.session_id)
                rejected = False
                try:
                    await h.provider.reserve_session(candidate, h.publish).open()
                except OpenAIProviderProtocolError:
                    rejected = True
                await asyncio.sleep(0)
                observations.append((rejected, retained_counts()))

            peer_closed_before_repair = h.sockets[0].closed
            bad_closed_before_repair = bad_socket.closed
            await h.provider.submit_frame(frame(healthy, final=False))
            peer_kinds = [event["type"] for event in h.sockets[0].sent]
            bad_socket.close, bad_socket.abort = original_close, original_abort
            await reservation.drain(CloseRequestReason.DAEMON_SHUTDOWN)
            repaired = open_request()
            await h.provider.reserve_session(repaired, h.publish).open()
            reopened = repaired.session_id in h.provider._sessions
        finally:
            bad_socket.close, bad_socket.abort = original_close, original_abort
            await asyncio.wait_for(h.provider.shutdown(), timeout=3)

        assert len(attempted_ids) == 100
        assert before[:5] == ((1, 1, 0, 2, 1) if failed_open else (2, 0, 0, 2, 1))
        assert before[-1] == ("close", "abort")
        assert all(
            rejected and counts == before for rejected, counts in observations
        ), (
            "unrepaired fresh UUIDs must not allocate another socket/session/task",
            before,
            observations[0],
            observations[-1],
        )
        assert not peer_closed_before_repair and not bad_closed_before_repair
        assert peer_kinds == ["session.update", "session.input_audio_buffer.append"]
        assert bad_socket.closed and reopened
        h.assert_empty_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=6))


def _opening_finally_probe(connection, terminal_shutdown):
    async def scenario():
        h = Harness()
        socket = FakeRealtimeWebSocket()
        entered, close_entered, close_release = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        finally_entered = asyncio.Event()
        effects, receipts, opened_results = [], [], []

        async def recv():
            entered.set()
            await asyncio.Event().wait()

        async def close():
            effects.append("close")
            close_entered.set()
            await close_release.wait()
            socket.closed = True

        async def connect(_uri, **_options):
            h.sockets.append(socket)
            return socket

        socket.recv, socket.close = recv, close
        h.provider._connect = connect
        request = open_request()
        owner = h.provider.reserve_session(request, h.publish)

        async def caller():
            try:
                opened_results.append(await owner.open())
            finally:
                finally_entered.set()
                receipts.append(await owner.drain(CloseRequestReason.USER_STOP))

        calling = asyncio.create_task(caller())
        await asyncio.wait_for(entered.wait(), timeout=1)
        independent = asyncio.create_task(
            h.provider.shutdown()
            if terminal_shutdown
            else owner.drain(CloseRequestReason.USER_STOP)
        )
        await asyncio.wait_for(close_entered.wait(), timeout=1)
        calling.cancel()
        await asyncio.sleep(0)
        calling.cancel()
        await asyncio.sleep(0)
        connection.send(
            (
                "held",
                (
                    calling.done(),
                    independent.done(),
                    socket.closed,
                    h.provider._sessions.get(request.session_id) is owner,
                    owner.generation is not None,
                    tuple(effects),
                    len(h.events),
                    len(opened_results),
                ),
            )
        )
        close_release.set()
        done, pending = await asyncio.wait({calling, independent}, timeout=1)
        connection.send(
            (
                "completion",
                (
                    len(done),
                    len(pending),
                    finally_entered.is_set(),
                    socket.closed,
                    tuple(effects),
                    len(h.events),
                    len(opened_results),
                ),
            )
        )
        if pending:
            # Parent reaps this exact isolated child; forced exit is not drain proof.
            await asyncio.Event().wait()
        outcomes = await asyncio.gather(calling, independent, return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError)
        assert len(receipts) == 1
        receipt = receipts[0]
        assert (
            receipt.session_id == request.session_id and receipt.delivery_error is None
        )
        assert outcomes[1] is None if terminal_shutdown else outcomes[1] is receipt
        assert not h.provider._sessions and not h.provider._failed_opens
        assert not h.provider._opening and owner.generation is None
        assert await owner.drain(CloseRequestReason.USER_STOP) is receipt
        await h.provider.shutdown()
        connection.send(("drained", True))

    try:
        asyncio.run(scenario())
    except BaseException as error:
        connection.send(("harness_error", type(error).__name__))
    finally:
        connection.close()


@pytest.mark.parametrize("terminal_shutdown", [False, True], ids=["drain", "shutdown"])
def test_opening_phase_drain_does_not_join_callers_finally(terminal_shutdown):
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    child = context.Process(
        target=_opening_finally_probe, args=(sending, terminal_shutdown)
    )
    observations = []
    try:
        child.start()
        sending.close()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if receiving.poll(max(0, deadline - time.monotonic())):
                try:
                    item = receiving.recv()
                except EOFError:
                    break
                observations.append(item)
                if item[0] in {"drained", "harness_error"}:
                    break
                if item[0] == "completion" and item[1][1]:
                    break
        if observations and observations[-1] == ("drained", True):
            child.join(timeout=1)
    finally:
        sending.close()
        if child.pid is not None:
            if child.is_alive():
                child.terminate()
                child.join(timeout=1)
            if child.is_alive():
                child.kill()
                child.join(timeout=1)
        alive, exitcode = child.is_alive(), child.exitcode
        if not alive:
            child.close()
        receiving.close()
    assert not alive
    assert observations[:1] == [
        ("held", (False, False, False, True, True, ("close",), 0, 0))
    ], observations
    assert observations[1:2] == [
        ("completion", (2, 0, True, True, ("close",), 0, 0))
    ], observations
    assert observations[2:] == [("drained", True)], observations
    assert exitcode == 0


def test_openai_reservation_is_exact_authority_before_open_and_after_uuid_reuse():
    async def scenario():
        h = Harness()
        value = open_request()
        try:
            owner = h.provider.reserve_session(value, h.publish)
            with pytest.raises(OpenAIProviderProtocolError):
                h.provider.reserve_session(value, h.publish)
            assert h.sockets == [] and h.events == []
            await owner.open()
            assert len(h.sockets) == 1 and not h.sockets[0].closed
            first = await owner.drain(CloseRequestReason.USER_STOP)
            assert first.session_id == value.session_id
            assert first.delivery_error is None and h.sockets[0].closed
            successor = h.provider.reserve_session(value, h.publish)
            await successor.open()
            before_stale = (len(h.sockets), tuple(h.events))
            assert await owner.drain(CloseRequestReason.DAEMON_SHUTDOWN) == first
            assert (len(h.sockets), tuple(h.events)) == before_stale
            assert not h.sockets[1].closed
            await h.provider.submit_frame(frame(value, final=False))
            assert h.sockets[1].sent[-1]["type"] == "session.input_audio_buffer.append"
            await successor.drain(CloseRequestReason.USER_STOP)
        finally:
            await asyncio.wait_for(h.provider.shutdown(), timeout=3)
        h.assert_empty_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=6))


def test_openai_open_cancellation_retains_exact_owner_and_concurrent_drain_retry():
    async def scenario():
        h = Harness()
        value = open_request()
        socket = FakeRealtimeWebSocket()
        handshake_entered, handshake_release = asyncio.Event(), asyncio.Event()
        close_entered, close_release = asyncio.Event(), asyncio.Event()
        effects = []
        cleanup_fails = True
        original_recv = socket.recv
        opening = None
        drains = []

        async def held_recv():
            handshake_entered.set()
            await handshake_release.wait()
            return await original_recv()

        async def close():
            effects.append("close")
            if cleanup_fails:
                raise OSError("synthetic close failure")
            close_entered.set()
            await close_release.wait()
            socket.closed = True

        def abort():
            effects.append("abort")
            if cleanup_fails:
                raise OSError("synthetic abort failure")
            socket.closed = True

        async def connect(_uri, **_options):
            h.sockets.append(socket)
            return socket

        socket.recv, socket.close, socket.abort = held_recv, close, abort
        h.provider._connect = connect
        try:
            owner = h.provider.reserve_session(value, h.publish)
            opening = asyncio.create_task(owner.open())
            await asyncio.wait_for(handshake_entered.wait(), timeout=1)
            with pytest.raises(OpenAIProviderProtocolError):
                h.provider.reserve_session(value, h.publish)
            opening.cancel()
            with pytest.raises(asyncio.CancelledError):
                await opening
            assert not socket.closed and effects == ["close", "abort"]
            with pytest.raises(OpenAIProviderProtocolError):
                await owner.drain(CloseRequestReason.DAEMON_SHUTDOWN)
            assert effects == ["close", "abort", "close", "abort"]

            def retained_owners():
                return tuple(
                    frozenset((key, id(record)) for key, record in records.items())
                    for records in (
                        h.provider._sessions,
                        h.provider._failed_opens,
                        h.provider._opening,
                    )
                )

            baseline = (
                len(h.sockets),
                tuple(effects),
                len(asyncio.all_tasks()),
                retained_owners(),
            )
            for _ in range(100):
                with pytest.raises(OpenAIProviderProtocolError):
                    h.provider.reserve_session(open_request(), h.publish)
                assert (
                    len(h.sockets),
                    tuple(effects),
                    len(asyncio.all_tasks()),
                    retained_owners(),
                ) == baseline
            cleanup_fails = False
            drains.append(
                asyncio.create_task(owner.drain(CloseRequestReason.DAEMON_SHUTDOWN))
            )
            await asyncio.wait_for(close_entered.wait(), timeout=1)
            drains.append(
                asyncio.create_task(owner.drain(CloseRequestReason.DAEMON_SHUTDOWN))
            )
            await asyncio.sleep(0)
            drains[0].cancel()
            await asyncio.sleep(0)
            drains[0].cancel()
            await asyncio.sleep(0)
            before_release = (
                tuple(effects),
                tuple(task.done() for task in drains),
                socket.closed,
            )
            close_release.set()
            results = await asyncio.wait_for(
                asyncio.gather(*drains, return_exceptions=True), timeout=1
            )
            assert before_release == (
                ("close", "abort", "close", "abort", "close"),
                (False, False),
                False,
            )
            assert isinstance(results[0], asyncio.CancelledError)
            receipt = results[1]
            assert receipt.session_id == value.session_id
            assert receipt.delivery_error is None
            assert await owner.drain(CloseRequestReason.DAEMON_SHUTDOWN) == receipt
            assert effects == ["close", "abort", "close", "abort", "close"]
            assert h.events == [] and opening.done() and socket.closed
        finally:
            cleanup_fails = False
            handshake_release.set()
            close_release.set()
            if opening is not None:
                opening.cancel()
                await asyncio.gather(opening, return_exceptions=True)
            await asyncio.gather(*drains, return_exceptions=True)
            await asyncio.wait_for(h.provider.shutdown(), timeout=3)
        h.assert_empty_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=6))
