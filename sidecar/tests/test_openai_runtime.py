from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from uuid import uuid4

import pytest
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
    CloseProviderSession,
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
        await self.provider.open_session(request, self.publish)
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


@pytest.mark.parametrize("fault", ["send", "handshake", "task", "insert"])
def test_failed_open_100_iterations_release_every_resource(fault, monkeypatch):
    async def scenario():
        h = Harness(fault=fault)

        class FailingRegistry(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("registry_insert_failed")

        if fault == "insert":
            h.provider._sessions = FailingRegistry()
        if fault == "task":

            def fail_task(coroutine, **kwargs):
                raise RuntimeError("task_creation_failed")

            monkeypatch.setattr(asyncio, "create_task", fail_task)
        tasks_before = asyncio.all_tasks()
        for _ in range(100):
            with pytest.raises((OpenAIProviderProtocolError, RuntimeError)):
                await asyncio.wait_for(h.open(), timeout=2)
            assert not h.provider._sessions
            h.assert_empty_generations()
        assert asyncio.all_tasks() == tasks_before
        await h.provider.shutdown()
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_concurrent_duplicate_open_reserves_before_connect_100_iterations():
    async def scenario():
        h = Harness()
        for _ in range(100):
            h.connect_gate = asyncio.Event()
            request = open_request()
            task = asyncio.create_task(h.provider.open_session(request, h.publish))
            await asyncio.sleep(0)
            with pytest.raises(OpenAIProviderProtocolError, match="duplicate_session"):
                await h.provider.open_session(request, h.publish)
            h.connect_gate.set()
            await task
            await h.provider.close_session(
                CloseProviderSession(
                    session_id=request.session_id,
                    reason=CloseRequestReason.USER_STOP,
                )
            )
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
        await h.provider.submit_frame(frame(request))
        h.sockets[0].push(audio())
        await h.settled()
        h.assert_empty_generations()
        with pytest.raises(
            OpenAIProviderProtocolError, match="provider_publication_failed"
        ):
            await h.provider.wait_publications(request.session_id)
        with pytest.raises(OpenAIProviderProtocolError, match="open_session_required"):
            await h.provider.submit_frame(frame(request))
        with pytest.raises(OpenAIProviderProtocolError, match="openai_shutdown_failed"):
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
