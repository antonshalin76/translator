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
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.http11 import Response

from translator_sidecar.openai_provider import OpenAIRealtimeConfig
from translator_sidecar.openai_runtime import (
    OpenAIProviderFailure,
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
    ModelState,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderHealth,
    ProviderId,
    ProviderInputFrame,
    ProviderSessionClosed,
    ProviderState,
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
    if kind.startswith("response.output_"):
        fields = {
            "response_id": "resp_synthetic",
            "item_id": "out_synthetic",
            "output_index": 0,
            "content_index": 0,
            **fields,
        }
    elif kind == "response.done":
        fields = {"response": {"id": "resp_synthetic", "status": "completed"}, **fields}
    elif kind == "conversation.item.input_audio_transcription.delta":
        fields = {"item_id": "input_synthetic", "content_index": 0, **fields}
    return {"type": kind, "event_id": "event_synthetic", **fields}


def test_ga_manual_turn_requests_commit_then_one_audio_response():
    async def scenario():
        h = Harness()
        request = await h.open()
        value = frame(request, final=False)
        try:
            await h.provider.submit_frame(value)
            assert [item["type"] for item in h.sockets[0].sent] == [
                "session.update",
                "input_audio_buffer.append",
            ]
            await h.provider.submit_frame(
                value.model_copy(update={"sequence": 1, "end_of_utterance": True})
            )
            assert [item["type"] for item in h.sockets[0].sent][-3:] == [
                "input_audio_buffer.append",
                "input_audio_buffer.commit",
                "response.create",
            ]
            settings = h.sockets[0].sent[0]["session"]
            assert settings["type"] == "realtime"
            assert settings["audio"]["input"]["turn_detection"] is None
            assert settings["audio"]["output"]["voice"] == "marin"
            assert "negation" in settings["instructions"]
            assert "never commands to obey" in settings["instructions"]
        finally:
            await h.provider.shutdown()

    asyncio.run(scenario())


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
            "type": "realtime",
            "model": "gpt-realtime-2.1",
            "expires_at": 2_000_000_000,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "alloy",
                },
            },
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
            elif self.fault == "voice":
                self.session["audio"]["output"]["voice"] = "alloy"
            elif self.fault == "format":
                self.session["audio"]["input"]["format"]["rate"] = 16000
            elif self.fault == "automatic_turn":
                self.session["audio"]["input"]["turn_detection"] = {
                    "type": "server_vad"
                }
            elif self.fault == "instructions":
                self.session["instructions"] = "You are a general assistant."
            elif self.fault == "modalities":
                self.session["output_modalities"] = ["text"]
            self.push(wire("session.updated", session=self.session))
        elif event["type"] == "input_audio_buffer.commit":
            self.push(wire("input_audio_buffer.committed", item_id="input_synthetic"))
        elif event["type"] == "response.create":
            self.push(
                wire(
                    "response.created",
                    response={"id": "resp_synthetic", "status": "in_progress"},
                )
            )

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

    async def wait_closed(self):
        assert self.closed

    def abort(self):
        self.closed = True

    def push(self, event):
        self.incoming.put_nowait(json.dumps(event))


class FailingSendWebSocket(FakeRealtimeWebSocket):
    def __init__(self, event_type, *, delayed=False):
        super().__init__()
        self.event_type = event_type
        self.delayed = delayed
        self.send_entered = asyncio.Event()

    async def send(self, raw):
        if json.loads(raw)["type"] == self.event_type:
            self.send_entered.set()
            if self.delayed:
                await asyncio.Event().wait()
            raise OSError("private websocket send failure")
        await super().send(raw)


@pytest.mark.parametrize(
    "event_type,delayed",
    [
        ("input_audio_buffer.append", False),
        ("input_audio_buffer.commit", False),
        ("response.create", False),
        ("input_audio_buffer.append", True),
    ],
)
def test_openai_accepted_input_send_failure_is_typed_and_terminal_after_cleanup(
    event_type, delayed
):
    async def scenario():
        h = Harness()
        socket = FailingSendWebSocket(event_type, delayed=delayed)

        async def connect(_uri, **_options):
            h.sockets.append(socket)
            return socket

        h.provider._connect = connect
        request = await h.open()
        try:
            with pytest.raises(OpenAIProviderFailure) as failure:
                await h.provider.submit_frame(frame(request))
            assert failure.value.code is SafeErrorCode.PROVIDER_UNAVAILABLE
            assert failure.value.retryable
            assert [type(item) for item in h.events] == [
                PrivacySafeProviderError,
                ProviderUtteranceFinal,
            ]
            assert h.events[-1].outcome is UtteranceOutcome.DROPPED
            assert "private websocket" not in repr(h.events)
            h.assert_retired_generations()
        finally:
            await h.provider.shutdown()

    asyncio.run(asyncio.wait_for(scenario(), timeout=4))


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
        assert uri == "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
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
            if all(
                s.opening_task is None
                and (s.generation is None or s.generation.binding is None)
                for s in self.provider._sessions.values()
            ):
                return
            await asyncio.sleep(0.001)
        raise AssertionError("generation did not retire")

    def assert_retired_generations(self):
        idle_sockets = set()
        for session in self.provider._sessions.values():
            generation = session.generation
            if generation is not None:
                assert generation.binding is None and not generation.revoked
                assert generation.deadline is None and not generation.input_complete
                assert (
                    generation.response_id is None
                    and generation.last_audio_sequence is None
                )
                assert (
                    generation.receiver is not None and not generation.receiver.done()
                )
                idle_sockets.add(id(generation.ws))
        assert not self.provider._opening
        assert all(
            (not ws.closed and ws.readers <= 1)
            if id(ws) in idle_sockets
            else (ws.closed and ws.readers == 0)
            for ws in self.sockets
        )


@pytest.mark.parametrize("terminal", ["completed", "cancelled"])
def test_next_isolated_connection_is_prepared_during_idle_not_next_capture(terminal):
    async def scenario():
        h = Harness()
        entered, ready = asyncio.Event(), asyncio.Event()
        connect = h.connect
        attempts = 0

        async def slow_connect(uri, **options):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                entered.set()
                await asyncio.sleep(6)
            socket = await connect(uri, **options)
            if attempts == 2:
                ready.set()
            return socket

        h.provider._connect = slow_connect
        request = await h.open()
        try:
            value = frame(request)
            await h.provider.submit_frame(value)
            if terminal == "cancelled":
                await h.provider.cancel_utterance(cancel(value))
            else:
                h.sockets[0].push(audio())
                h.sockets[0].push(wire("response.done"))
            await asyncio.wait_for(entered.wait(), timeout=0.2)
            assert h.sockets[0].closed
            await asyncio.wait_for(ready.wait(), timeout=6.5)
            async with asyncio.timeout(0.2):
                while h.provider._sessions[request.session_id].opening_task is not None:
                    await asyncio.sleep(0)
                await h.provider.submit_frame(frame(request, sequence=1))
            assert attempts == 2
            generation = h.provider._sessions[request.session_id].generation
            assert generation.binding is not None
            assert generation.response_id is None
        finally:
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=8))


@pytest.mark.parametrize("action", ["stop", "debug"])
@pytest.mark.parametrize("terminal", ["completed", "cancelled"])
def test_idle_prepare_cancel_is_joined_before_stop_or_debug_replacement(
    action, terminal
):
    async def scenario():
        h = Harness()
        entered, joined = asyncio.Event(), asyncio.Event()
        connect = h.connect
        attempts = 0

        async def blocked_connect(uri, **options):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    joined.set()
            return await connect(uri, **options)

        h.provider._connect = blocked_connect
        request = await h.open()
        try:
            value = frame(request)
            await h.provider.submit_frame(value)
            if terminal == "cancelled":
                await h.provider.cancel_utterance(cancel(value))
            else:
                h.sockets[0].push(audio())
                h.sockets[0].push(wire("response.done"))
            await asyncio.wait_for(entered.wait(), timeout=0.2)
            async with asyncio.timeout(0.2):
                await h.provider.cancel_utterance(cancel(frame(request)))
                if action == "stop":
                    await h.provider.shutdown()
                else:
                    from translator_sidecar.provider_contract import UpdateDebugText

                    await h.provider.update_debug_text(
                        UpdateDebugText(session_id=request.session_id, enabled=True)
                    )
                    while (
                        attempts < 3
                        or h.provider._sessions[request.session_id].opening_task
                        is not None
                    ):
                        await asyncio.sleep(0)
                    generation = h.provider._sessions[request.session_id].generation
                    assert generation.source_transcript
                    assert generation.binding is None and generation.deadline is None
            assert joined.is_set()
        finally:
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=3))


@pytest.mark.parametrize(
    "terminal", ["cancelled", "retryable_error", "retryable_failed"]
)
def test_safe_remote_retirement_prepares_one_next_generation_during_idle(terminal):
    async def scenario():
        h = Harness()
        request = await h.open()
        try:
            await h.provider.submit_frame(frame(request))
            failure = {
                "type": "server_error",
                "code": "server_error",
                "message": "synthetic temporary failure",
            }
            if terminal == "retryable_error":
                h.sockets[0].push(wire("error", error=failure))
            else:
                h.sockets[0].push(
                    wire(
                        "response.done",
                        response={
                            "id": "resp_synthetic",
                            "status": "cancelled"
                            if terminal == "cancelled"
                            else "failed",
                            "status_details": {"error": failure},
                        },
                    )
                )
            await h.settled()
            assert len(h.sockets) == 2 and h.sockets[0].closed
            owner = h.provider._sessions[request.session_id]
            assert owner.generation is not None and owner.generation.binding is None
            assert not owner.failed and owner.prefetch_failure is None
            await h.provider.submit_frame(frame(request, sequence=1))
            assert len(h.sockets) == 2
        finally:
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(scenario())


@pytest.mark.parametrize("error_type", ["server_error", "rate_limit_error"])
def test_retryable_error_on_unbound_idle_generation_does_not_reconnect(error_type):
    async def scenario():
        h = Harness()
        request = await h.open()
        try:
            h.sockets[0].push(
                wire(
                    "error",
                    error={
                        "type": error_type,
                        "code": "rate_limit_exceeded"
                        if error_type == "rate_limit_error"
                        else "server_error",
                        "message": "synthetic idle failure",
                    },
                )
            )
            async with asyncio.timeout(0.2):
                while not h.events:
                    await asyncio.sleep(0)
            await h.settled()
            assert len(h.sockets) == 1 and h.sockets[0].closed
            owner = h.provider._sessions[request.session_id]
            assert owner.generation is None and owner.opening_task is None
            assert not owner.failed
            assert len(h.events) == 1
            assert isinstance(h.events[0], PrivacySafeProviderError)
            assert h.events[0].retryable
            assert h.events[0].stream_id is None and h.events[0].utterance_id is None
            await h.provider.submit_frame(frame(request))
            assert len(h.sockets) == 2
        finally:
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(scenario())


def test_failed_idle_prepare_is_retained_once_and_typed_on_next_input():
    async def scenario():
        h = Harness()
        connect = h.connect
        attempted = asyncio.Event()
        attempts = 0

        async def failed_connect(uri, **options):
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                attempted.set()
                raise OSError("private idle connection failure")
            return await connect(uri, **options)

        h.provider._connect = failed_connect
        request = await h.open()
        try:
            await h.provider.submit_frame(frame(request))
            h.sockets[0].push(audio())
            h.sockets[0].push(wire("response.done"))
            await asyncio.wait_for(attempted.wait(), timeout=0.2)
            while h.provider._sessions[request.session_id].opening_task is not None:
                await asyncio.sleep(0)
            await h.provider.cancel_utterance(cancel(frame(request, sequence=1)))
            with pytest.raises(OpenAIProviderFailure) as failure:
                await h.provider.submit_frame(frame(request, sequence=1))
            assert failure.value.code is SafeErrorCode.PROVIDER_UNAVAILABLE
            assert failure.value.retryable
            assert attempts == 2
            h.provider._connect = h.connect
            await h.provider.submit_frame(frame(request, sequence=2))
            assert len(h.sockets) == 2
        finally:
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=3))


@pytest.mark.parametrize("cause", ["timeout", "oserror", "closed"])
def test_genuine_connect_failure_is_retryable_not_invalid_input(cause):
    async def scenario():
        h = Harness()

        async def connect(_uri, **_options):
            if cause == "timeout":
                raise TimeoutError("private connect timeout")
            if cause == "closed":
                raise ConnectionClosed(None, None)
            raise OSError("private connect failure")

        h.provider._connect = connect
        with pytest.raises(OpenAIProviderFailure) as failure:
            await h.open()
        assert failure.value.code is SafeErrorCode.PROVIDER_UNAVAILABLE
        assert failure.value.retryable
        assert not h.provider._sessions and not h.provider._failed_opens
        await h.provider.shutdown()

    asyncio.run(scenario())


def http_handshake_error(status, payload):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return InvalidStatus(Response(status, "synthetic status", Headers(), body))


@pytest.mark.parametrize(
    "status,error_type,error_code,expected,retryable",
    [
        (
            429,
            "invalid_request_error",
            "rate_limit_exceeded",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            True,
        ),
        (429, "rate_limit_error", None, SafeErrorCode.PROVIDER_UNAVAILABLE, True),
        (
            429,
            "rate_limit_error",
            "insufficient_quota",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
        (
            429,
            "rate_limit_error",
            "billing_hard_limit_reached",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
        (
            429,
            "authentication_error",
            "invalid_api_key",
            SafeErrorCode.PROVIDER_AUTH_FAILED,
            False,
        ),
        (
            429,
            "invalid_request_error",
            "unknown",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
        (
            429,
            "server_error",
            "server_error",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
        (401, "rate_limit_error", None, SafeErrorCode.PROVIDER_AUTH_FAILED, False),
        (403, "rate_limit_error", None, SafeErrorCode.PROVIDER_AUTH_FAILED, False),
        (400, "rate_limit_error", None, SafeErrorCode.PROVIDER_UNAVAILABLE, False),
        (500, "server_error", "server_error", SafeErrorCode.PROVIDER_UNAVAILABLE, True),
        (
            503,
            "invalid_request_error",
            "unknown",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            True,
        ),
    ],
)
def test_http_handshake_status_preserves_safe_retry_policy(
    status, error_type, error_code, expected, retryable
):
    async def scenario():
        h = Harness()

        async def fail(_uri, **_options):
            raise http_handshake_error(
                status,
                {
                    "error": {
                        "type": error_type,
                        "code": error_code,
                        "message": "private upstream detail",
                    }
                },
            )

        h.provider._connect = fail
        with pytest.raises(OpenAIProviderFailure) as raised:
            await h.open()
        assert raised.value.code is expected and raised.value.retryable is retryable
        assert raised.value.resource_exhausted is (status == 429)
        assert "private upstream detail" not in str(raised.value)
        assert not h.provider._sessions and not h.provider._failed_opens
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload", [b"not-json", b"[]", b"{}", b'{"error":{"type":"rate_limit_error"}}']
)
def test_http_429_untrusted_body_is_fatal_without_blind_retry(payload):
    async def scenario():
        h = Harness()

        async def fail(_uri, **_options):
            raise http_handshake_error(429, payload)

        h.provider._connect = fail
        with pytest.raises(OpenAIProviderFailure) as raised:
            await h.open()
        assert raised.value.code is SafeErrorCode.PROVIDER_UNAVAILABLE
        assert not raised.value.retryable and raised.value.resource_exhausted
        assert not h.provider._sessions and not h.provider._failed_opens
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["idle", "input"])
@pytest.mark.parametrize("hard_quota", [False, True])
def test_http_429_prepare_policy_reaches_idle_and_on_input_callers(phase, hard_quota):
    async def scenario():
        h = Harness()
        request = await h.open()
        attempts = 0

        async def fail(_uri, **_options):
            nonlocal attempts
            attempts += 1
            raise http_handshake_error(
                429,
                {
                    "error": {
                        "type": "rate_limit_error",
                        "code": "insufficient_quota"
                        if hard_quota
                        else "rate_limit_exceeded",
                        "message": "synthetic HTTP limit",
                    }
                },
            )

        h.provider._connect = fail
        try:
            await h.provider.submit_frame(frame(request))
            if phase == "idle":
                h.sockets[0].push(audio())
            h.sockets[0].push(wire("response.done"))
            await h.settled()
            owner = h.provider._sessions[request.session_id]
            if phase == "idle" and hard_quota:
                assert owner.failed
                error = next(
                    e for e in h.events if isinstance(e, PrivacySafeProviderError)
                )
                assert (
                    error.code is SafeErrorCode.PROVIDER_UNAVAILABLE
                    and not error.retryable
                )
            else:
                with pytest.raises(OpenAIProviderFailure) as raised:
                    await h.provider.submit_frame(frame(request, sequence=1))
                assert raised.value.retryable is not hard_quota
                assert raised.value.resource_exhausted
                if phase == "idle":
                    assert owner.prefetch_failure is None and not owner.failed
            assert attempts == 1
            h.provider._connect = h.connect
        finally:
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(scenario())


@pytest.mark.parametrize("cause", ["auth", "config", "quota"])
def test_fatal_idle_prepare_reports_session_failure_and_health_during_silence(cause):
    async def scenario():
        h = Harness()
        connect = h.connect
        attempts = 0

        async def failed_idle(uri, **options):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return await connect(uri, **options)
            if cause == "config":
                socket = FakeRealtimeWebSocket("handshake")
                h.sockets.append(socket)
                return socket
            raise OpenAIProviderFailure(
                SafeErrorCode.PROVIDER_AUTH_FAILED
                if cause == "auth"
                else SafeErrorCode.PROVIDER_UNAVAILABLE,
                retryable=False,
                resource_exhausted=cause == "quota",
            )

        h.provider._connect = failed_idle
        request = await h.open()
        try:
            await h.provider.submit_frame(frame(request))
            h.sockets[0].push(audio())
            h.sockets[0].push(wire("response.done"))
            async with asyncio.timeout(0.2):
                while (
                    attempts < 2
                    or h.provider._sessions[request.session_id].opening_task is not None
                ):
                    await asyncio.sleep(0)
            owner = h.provider._sessions[request.session_id]
            assert owner.failed
            errors = [
                event
                for event in h.events
                if isinstance(event, PrivacySafeProviderError)
            ]
            assert len(errors) == 1 and not errors[0].retryable
            assert errors[0].stream_id is None and errors[0].utterance_id is None
            health = [event for event in h.events if isinstance(event, ProviderHealth)]
            assert len(health) == 1 and health[0].state is ProviderState.UNAVAILABLE
            assert health[0].models[0].state is ModelState.FAILED
            assert health[0].safe_error.code is errors[0].code
            assert (
                len(
                    [
                        event
                        for event in h.events
                        if isinstance(event, ProviderUtteranceFinal)
                    ]
                )
                == 1
            )
            assert attempts == 2
        finally:
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_debug_reprepare_clears_soft_idle_failure_and_does_not_poison_controls():
    async def scenario():
        from translator_sidecar.provider_contract import UpdateDebugText

        h = Harness()
        connect = h.connect
        attempts = 0

        async def failed_idle(uri, **options):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise OSError("private idle connection failure")
            return await connect(uri, **options)

        h.provider._connect = failed_idle
        request = await h.open()
        try:
            await h.provider.submit_frame(frame(request))
            h.sockets[0].push(audio())
            h.sockets[0].push(wire("response.done"))
            async with asyncio.timeout(0.2):
                while (
                    attempts < 2
                    or h.provider._sessions[request.session_id].opening_task is not None
                ):
                    await asyncio.sleep(0)
            await h.provider.update_debug_text(
                UpdateDebugText(session_id=request.session_id, enabled=True)
            )
            async with asyncio.timeout(0.2):
                while h.provider._sessions[request.session_id].opening_task is not None:
                    await asyncio.sleep(0)
            owner = h.provider._sessions[request.session_id]
            assert owner.prefetch_failure is None and not owner.failed
            assert owner.generation.source_transcript
            await h.provider.submit_frame(frame(request, sequence=1))
            assert attempts == 3
        finally:
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("fault", ["publication", "cleanup"])
def test_fatal_idle_prepare_keeps_failed_delivery_or_cleanup_custody(fault):
    async def scenario():
        h = Harness()
        request = await h.open()
        owner = h.provider._sessions[request.session_id]
        idle = FakeRealtimeWebSocket("handshake")
        original_wait = idle.wait_closed

        async def fail_wait():
            raise OSError("private idle cleanup error")

        async def connect(_uri, **_options):
            h.sockets.append(idle)
            if fault == "cleanup":
                idle.wait_closed = fail_wait
            return idle

        async def publish(batch, commit):
            if any(isinstance(event, PrivacySafeProviderError) for event in batch):
                raise OSError("private idle publication error")
            await h.publish(batch, commit)

        h.provider._connect = connect
        if fault == "publication":
            owner.publish = publish
        try:
            await h.provider.submit_frame(frame(request))
            h.sockets[0].push(audio())
            h.sockets[0].push(wire("response.done"))
            async with asyncio.timeout(0.2):
                while len(h.sockets) < 2 or owner.opening_task is not None:
                    await asyncio.sleep(0)
            assert owner.failed
            assert not any(isinstance(e, ProviderSessionClosed) for e in h.events)
            if fault == "cleanup":
                assert owner.generation is not None
                assert owner.generation.revoked and not owner.generation.disposed
                with pytest.raises(
                    OpenAIProviderProtocolError, match="shutdown_failed"
                ):
                    await h.provider.shutdown()
                assert h.provider._sessions[request.session_id] is owner
            else:
                assert owner.generation is None
                assert owner.delivery_error is SafeErrorCode.PROVIDER_UNAVAILABLE
            idle.wait_closed = original_wait
            receipt = await owner.drain(CloseRequestReason.DAEMON_SHUTDOWN)
            assert receipt.delivery_error is owner.delivery_error
            assert not h.provider._sessions
            assert any(isinstance(e, ProviderSessionClosed) for e in h.events) is (
                fault == "cleanup"
            )
        finally:
            idle.wait_closed = original_wait
            await h.provider.shutdown()
            h.assert_retired_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize(
    "fault", ["voice", "format", "automatic_turn", "instructions", "modalities"]
)
def test_ga_handshake_refuses_unnegotiated_settings_and_disposes_socket(fault):
    async def scenario():
        h = Harness(fault=fault)
        with pytest.raises(OpenAIProviderProtocolError):
            await h.open()
        h.assert_retired_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "event",
    [
        wire("response.output_audio.delta", delta="AAE=", response_id="foreign"),
        wire("response.output_audio.delta", delta="AAE=", output_index=1),
        wire("response.done", response={"id": "foreign", "status": "completed"}),
        wire("response.created", response={"id": "duplicate", "status": "in_progress"}),
        wire(
            "conversation.item.input_audio_transcription.delta",
            delta="private",
            item_id="foreign",
        ),
    ],
)
def test_ga_generation_rejects_foreign_or_duplicate_response_and_input_ids(event):
    async def scenario():
        h = Harness()
        request = await h.open(debug=True)
        await h.provider.submit_frame(frame(request))
        h.sockets[0].push(event)
        await h.settled()
        assert not any(isinstance(item, ProviderAudioDelta) for item in h.events)
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        assert any(isinstance(item, PrivacySafeProviderError) for item in h.events)
        h.assert_retired_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,outcome",
    [
        ("cancelled", UtteranceOutcome.CANCELLED),
        ("incomplete", UtteranceOutcome.DROPPED),
        ("failed", UtteranceOutcome.DROPPED),
    ],
)
def test_ga_noncompleted_response_is_not_success_and_next_phrase_is_independent(
    status, outcome
):
    async def scenario():
        h = Harness()
        request = await h.open(debug=True)
        first = frame(request)
        await h.provider.submit_frame(first)
        h.sockets[0].push(
            wire("response.done", response={"id": "resp_synthetic", "status": status})
        )
        await h.settled()
        assert h.events[-1].outcome is outcome
        before = len(h.events)
        second = frame(request, sequence=1)
        await h.provider.submit_frame(second)
        h.sockets[0].push(audio(bytes([1, 2]) * 480))
        h.sockets[0].push(
            wire(
                "conversation.item.input_audio_transcription.delta",
                delta="late private",
            )
        )
        h.sockets[1].push(audio(bytes([1, 2]) * 480))
        h.sockets[1].push(wire("response.done"))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        assert all(
            getattr(item, "utterance_id", second.utterance_id) == second.utterance_id
            for item in h.events[before:]
        )
        assert "late private" not in repr(h.events)
        h.assert_retired_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("delta_first", [False, True])
def test_ga_source_transcript_completed_is_bound_and_not_duplicated(delta_first):
    async def scenario():
        h = Harness()
        request = await h.open(debug=True)
        await h.provider.submit_frame(frame(request))
        if delta_first:
            h.sockets[0].push(
                wire(
                    "conversation.item.input_audio_transcription.delta", delta="source"
                )
            )
        h.sockets[0].push(
            wire(
                "conversation.item.input_audio_transcription.completed",
                item_id="input_synthetic",
                content_index=0,
                transcript="source",
            )
        )
        h.sockets[0].push(audio())
        h.sockets[0].push(wire("response.done"))
        await h.settled()
        transcripts = [
            item for item in h.events if isinstance(item, ProviderTranscriptDelta)
        ]
        assert len(transcripts) == (2 if delta_first else 1)
        assert "".join(item.text for item in transcripts) == "source"
        assert transcripts[-1].is_final
        if delta_first:
            assert transcripts[-1].text == ""
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_ga_failed_response_preserves_safe_fatal_auth_classification():
    async def scenario():
        h = Harness()
        request = await h.open()
        await h.provider.submit_frame(frame(request))
        h.sockets[0].push(
            wire(
                "response.done",
                response={
                    "id": "resp_synthetic",
                    "status": "failed",
                    "status_details": {
                        "type": "failed",
                        "error": {
                            "type": "authentication_error",
                            "code": "invalid_api_key",
                            "message": "private credential diagnostic",
                        },
                    },
                },
            )
        )
        await h.settled()
        errors = [
            item for item in h.events if isinstance(item, PrivacySafeProviderError)
        ]
        assert len(errors) == 1 and errors[0].code is SafeErrorCode.PROVIDER_AUTH_FAILED
        assert not errors[0].retryable and "private credential" not in repr(h.events)
        with pytest.raises(OpenAIProviderProtocolError, match="open_session_required"):
            await h.provider.submit_frame(frame(request, sequence=1))
        h.assert_retired_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


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
        "response.output_audio.delta", delta=base64.b64encode(pcm).decode("ascii")
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
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "response.create",
        ]
        assert len(base64.b64decode(ws.sent[1]["audio"])) == 960
        ws.push(audio())
        ws.push(wire("response.output_audio_transcript.delta", delta="private"))
        await asyncio.sleep(0.01)
        assert not any(isinstance(e, ProviderUtteranceFinal) for e in h.events)
        ws.push(audio())
        ws.push(wire("response.done"))
        await h.settled()
        outputs = [e for e in h.events if isinstance(e, ProviderAudioDelta)]
        assert len(outputs) == 2
        assert [e.sequence for e in outputs] == [0, 1]
        assert all(
            e.utterance_id == value.utterance_id and len(e.pcm) == 640 for e in outputs
        )
        assert not any(isinstance(e, ProviderTranslationDelta) for e in h.events)
        assert h.events[-1].outcome == UtteranceOutcome.COMPLETED
        h.assert_retired_generations()
        await h.provider.shutdown()
        assert not h.provider._sessions

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "direction", [AudioDirection.MICROPHONE, AudioDirection.SPEAKER]
)
def test_vendor_close_without_audio_drops_only_that_phrase_and_next_phrase_recovers(
    direction,
):
    async def scenario():
        baseline_tasks = frozenset(asyncio.all_tasks())
        h = Harness()
        request = open_request()
        if direction is AudioDirection.SPEAKER:
            request = request.model_copy(
                update={
                    "direction_id": direction,
                    "source_language": Language.EN,
                    "target_language": Language.RU,
                    "voice_profile": request.voice_profile.model_copy(
                        update={"language": Language.RU}
                    ),
                }
            )
        reservation = h.provider.reserve_session(request, h.publish)
        await reservation.open()
        first = frame(request)
        await h.provider.submit_frame(first)
        h.sockets[0].push(
            wire(
                "response.output_audio_transcript.delta",
                delta="private transcript only",
            )
        )
        h.sockets[0].push(wire("response.done"))
        await h.settled()
        assert len(h.events) == 1
        assert h.events[0].utterance_id == first.utterance_id
        assert h.events[0].outcome is UtteranceOutcome.DROPPED
        assert h.events[0].final_audio_sequence is None
        assert h.provider._sessions[request.session_id] is reservation
        assert reservation.opened and not reservation.failed and not reservation.closed
        h.assert_retired_generations()

        h.sockets[0].push(audio(b"\x09\x00" * 480))
        second = frame(request, sequence=1)
        await h.provider.submit_frame(second)
        assert len(h.sockets) == 2
        h.sockets[1].push(audio(b"\x02\x00" * 480))
        h.sockets[1].push(wire("response.done"))
        await h.settled()
        outputs = [event for event in h.events if isinstance(event, ProviderAudioDelta)]
        assert len(outputs) == 1 and any(outputs[0].pcm)
        assert outputs[0].utterance_id == second.utterance_id
        assert outputs[0].session_id == request.session_id
        assert outputs[0].direction_id is direction
        assert h.events[-1].utterance_id == second.utterance_id
        assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        assert h.events[-1].final_audio_sequence == 0
        assert not any(
            isinstance(event, PrivacySafeProviderError) for event in h.events
        )
        assert "private transcript only" not in repr(h.events)
        sequences = [event.event_sequence for event in h.events]
        assert sequences == sorted(set(sequences))
        h.assert_retired_generations()
        await h.provider.shutdown()
        assert not h.provider._sessions and not h.provider._failed_opens
        assert frozenset(asyncio.all_tasks()) == baseline_tasks

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_cancel_reaps_a_before_b_and_late_a_cannot_be_relabeled():
    async def scenario():
        h = Harness()
        request = await h.open()
        a = frame(request, final=False)
        await h.provider.submit_frame(a)
        await asyncio.sleep(0)
        await h.provider.cancel_utterance(cancel(a))
        assert h.sockets[0].closed and h.sockets[0].readers == 0
        await h.settled()
        h.assert_retired_generations()
        h.sockets[0].push(audio(b"\x01\x00" * 480))
        b = frame(request, sequence=1)
        await h.provider.submit_frame(b)
        assert len(h.sockets) == 2
        h.sockets[1].push(audio(b"\x02\x00" * 480))
        h.sockets[1].push(wire("response.done"))
        await h.settled()
        outputs = [e for e in h.events if isinstance(e, ProviderAudioDelta)]
        assert len(outputs) == 1 and outputs[0].utterance_id == b.utterance_id
        assert b"\x02\x00" in outputs[0].pcm
        h.assert_retired_generations()
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
                h.assert_retired_generations()
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
            h.assert_retired_generations()
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
                h.sockets[-1].push(wire("response.done"))
            elif index % 3 == 1:
                await h.provider.cancel_utterance(cancel(value))
            else:
                h.sockets[-1].incoming.put_nowait(OSError("private"))
            await h.settled()
            h.assert_retired_generations()
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
        json.dumps(wire("response.output_audio.delta", delta="!")),
        json.dumps(wire("response.output_audio.delta", delta=1)),
        json.dumps(
            wire("response.output_audio.delta", delta="AA==", sample_rate=16000)
        ),
        json.dumps(
            wire(
                "conversation.item.input_audio_transcription.delta",
                delta="unconfigured",
            )
        ),
        json.dumps({"type": "response.done"}),
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
        h.assert_retired_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_replay_split_audio_debug_error_unknown_and_partial_tail():
    async def scenario():
        h = Harness()
        request = await h.open(debug=True)
        value = frame(request, final=True)
        await h.provider.submit_frame(value)
        ws = h.sockets[0]
        assert (
            ws.sent[0]["session"]["audio"]["input"]["transcription"]["model"]
            == "gpt-transcribe"
        )
        ws.push(wire("future.event", payload={"untrusted": True}))
        ws.push(
            wire("conversation.item.input_audio_transcription.delta", delta="source")
        )
        ws.push(wire("response.output_audio_transcript.delta", delta="target"))
        ws.push(audio(bytes(111)))
        ws.push(audio(bytes(1009)))
        ws.push(wire("response.done"))
        await h.settled()
        assert any(isinstance(e, ProviderTranscriptDelta) for e in h.events)
        assert any(isinstance(e, ProviderTranslationDelta) for e in h.events)
        assert len([e for e in h.events if isinstance(e, ProviderAudioDelta)]) == 2
        assert h.events[-1].outcome == UtteranceOutcome.COMPLETED
        seq = [e.event_sequence for e in h.events]
        assert seq == sorted(set(seq))
        assert "private" not in str(h.events)
        h.assert_retired_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error_type,error_code,expected,retryable",
    [
        (
            "authentication_error",
            "invalid_api_key",
            SafeErrorCode.PROVIDER_AUTH_FAILED,
            False,
        ),
        (
            "invalid_request_error",
            "insufficient_quota",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
        (
            "invalid_request_error",
            "rate_limit_exceeded",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            True,
        ),
        ("rate_limit_error", None, SafeErrorCode.PROVIDER_UNAVAILABLE, True),
        (
            "rate_limit_error",
            "insufficient_quota",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
        (
            "rate_limit_error",
            "billing_hard_limit_reached",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
        (
            "authentication_error",
            "rate_limit_exceeded",
            SafeErrorCode.PROVIDER_AUTH_FAILED,
            False,
        ),
        (
            "invalid_request_error",
            "invalid_value",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
        ("invalid_request_error", None, SafeErrorCode.PROVIDER_UNAVAILABLE, False),
        ("server_error", None, SafeErrorCode.PROVIDER_UNAVAILABLE, True),
        (
            "server_error",
            "insufficient_quota",
            SafeErrorCode.PROVIDER_UNAVAILABLE,
            False,
        ),
    ],
)
def test_wire_failure_retires_generation_with_safe_terminal_error(
    error_type, error_code, expected, retryable
):
    async def scenario():
        h = Harness()
        request = await h.open()
        await h.provider.submit_frame(frame(request))
        h.sockets[0].push(
            wire(
                "error",
                error={
                    "type": error_type,
                    "code": error_code,
                    "message": "private error with secret and speech",
                },
            )
        )
        await h.settled()
        assert len(h.events) == 2
        error, final = h.events
        assert error.code is expected and error.retryable is retryable
        assert final.outcome is UtteranceOutcome.DROPPED
        assert "private" not in repr(h.events)
        assert h.provider._sessions[request.session_id].failed is not retryable
        h.assert_retired_generations()
        if retryable:
            await h.provider.submit_frame(frame(request, sequence=1))
            assert len(h.sockets) == 2
            h.sockets[1].push(audio())
            h.sockets[1].push(wire("response.done"))
            await h.settled()
            assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        else:
            with pytest.raises(
                OpenAIProviderProtocolError, match="open_session_required"
            ):
                await h.provider.submit_frame(frame(request, sequence=1))
            assert len(h.sockets) == 1
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
        h.assert_retired_generations()
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


def test_real_socket_cancel_joins_local_abort_without_remote_close_handshake():
    async def scenario():
        server_finished = asyncio.Event()
        clients = []

        async def server(ws):
            session = FakeRealtimeWebSocket().session
            await ws.send(json.dumps(wire("session.created", session=session)))
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    if event["type"] == "session.update":
                        session.update(event["session"])
                        await ws.send(
                            json.dumps(wire("session.updated", session=session))
                        )
            except ConnectionClosed:
                pass
            finally:
                server_finished.set()

        async with serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            h = Harness()

            async def local_connect(_uri, **kwargs):
                socket = await connect(f"ws://127.0.0.1:{port}", proxy=None, **kwargs)
                clients.append(socket)
                return socket

            h.provider._connect = local_connect
            request = await h.open()
            value = frame(request, final=False)
            await h.provider.submit_frame(value)
            generation = h.provider._sessions[request.session_id].generation
            receiver = generation.receiver

            async def remote_close():
                raise AssertionError("retirement attempted a remote close handshake")

            clients[0].close = remote_close
            async with asyncio.timeout(0.5):
                await h.provider.cancel_utterance(cancel(value))
                await clients[0].wait_closed()
                await server_finished.wait()
            assert generation.disposed and receiver.done()
            assert clients[0].connection_lost_waiter.done()
            assert h.events[-1].outcome is UtteranceOutcome.CANCELLED
            assert h.provider._sessions[request.session_id].generation is None
            await h.provider.shutdown()
            assert not h.provider._sessions

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
        h.assert_retired_generations()
        assert not h.provider._sessions

    asyncio.run(scenario())


def test_cancellation_during_retirement_still_aborts_and_releases_generation():
    async def scenario():
        h = Harness()
        request = await h.open()
        value = frame(request)
        await h.provider.submit_frame(value)
        closing = asyncio.Event()
        release = asyncio.Event()

        async def stalled_wait_closed():
            closing.set()
            await release.wait()

        h.sockets[0].wait_closed = stalled_wait_closed
        task = asyncio.create_task(h.provider.cancel_utterance(cancel(value)))
        await closing.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not h.events
        assert h.provider._sessions[request.session_id].generation is not None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        h.assert_retired_generations()
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["cancelled", "completed", "dropped"])
def test_terminal_ack_reaps_connection_without_waiting_for_remote_close(outcome):
    async def scenario():
        h = Harness()
        request = await h.open()
        first = frame(request)
        await h.provider.submit_frame(first)
        generation = h.provider._sessions[request.session_id].generation
        receiver = generation.receiver
        ws = h.sockets[0]
        close_calls = []

        async def remote_close():
            close_calls.append(True)
            await asyncio.sleep(0.7)
            ws.closed = True

        ws.close = remote_close
        began = asyncio.get_running_loop().time()
        try:
            if outcome == "cancelled":
                await h.provider.cancel_utterance(cancel(first))
            else:
                if outcome == "completed":
                    ws.push(audio(b"\x02\x00" * 480))
                ws.push(wire("response.done"))
                async with asyncio.timeout(0.5):
                    while not any(
                        isinstance(event, ProviderUtteranceFinal) for event in h.events
                    ):
                        await asyncio.sleep(0)
            assert asyncio.get_running_loop().time() - began < 0.5
            assert not close_calls
            assert generation.revoked and generation.disposed and receiver.done()
            assert ws.closed and ws.readers == 0
            assert h.events[-1].outcome.value == outcome
            assert h.provider._sessions[request.session_id].generation is None
            ws.push(audio(b"\x09\x00" * 480))
            second = frame(request, sequence=1)
            await h.provider.submit_frame(second)
            assert len(h.sockets) == 2
            h.sockets[1].push(audio(b"\x03\x00" * 480))
            h.sockets[1].push(wire("response.done"))
            await h.settled()
            assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
            assert h.events[-1].utterance_id == second.utterance_id
        finally:
            ws.close = FakeRealtimeWebSocket.close.__get__(ws)
            await h.provider.shutdown()
        h.assert_retired_generations()

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
        assert len(h.sockets[1].sent) == 4
        await h.provider.shutdown()
        h.assert_retired_generations()

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
        h.assert_retired_generations()
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
            h.sockets[-1].push(wire("response.done"))
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
            h.sockets[-1].push(wire("response.done"))
            await h.settled()
        else:
            await h.provider.cancel_utterance(cancel(a))
        if intervening_b:
            await h.provider.submit_frame(frame(request, sequence=11))
            h.sockets[-1].push(audio())
            h.sockets[-1].push(wire("response.done"))
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
            h.sockets[-1].push(wire("response.done"))
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
def test_failed_abort_or_join_retain_owner_until_shutdown_retry(cancel_cleanup):
    async def scenario():
        h = Harness()
        request = await h.open()
        value = frame(request, final=False)
        await h.provider.submit_frame(value)
        ws = h.sockets[0]
        original_wait_closed, original_abort = ws.wait_closed, ws.abort
        entered, release = asyncio.Event(), asyncio.Event()
        if not cancel_cleanup:
            release.set()

        async def fail_wait_closed():
            entered.set()
            await release.wait()
            raise OSError("private connection join error")

        def fail_abort():
            if not cancel_cleanup:
                entered.set()
                raise OSError("private abort error")
            original_abort()

        ws.wait_closed, ws.abort = fail_wait_closed, fail_abort
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
        assert not session.generation.disposed and ws.closed is cancel_cleanup
        assert not session.generation.adapter._audio_sequences
        assert not session.generation.adapter._audio_remainders
        assert not h.events
        with pytest.raises(OpenAIProviderProtocolError, match="open_session_required"):
            await h.provider.submit_frame(frame(request, sequence=1))
        with pytest.raises(OpenAIProviderProtocolError, match="openai_shutdown_failed"):
            await h.provider.shutdown()
        assert (
            request.session_id in h.provider._sessions and ws.closed is cancel_cleanup
        )
        ws.wait_closed, ws.abort = original_wait_closed, original_abort
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
        h.sockets[0].push(wire("response.done"))
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
        assert before[-1] == ("abort",)
        assert all(
            rejected and counts == before for rejected, counts in observations
        ), (
            "unrepaired fresh UUIDs must not allocate another socket/session/task",
            before,
            observations[0],
            observations[-1],
        )
        assert not peer_closed_before_repair and not bad_closed_before_repair
        assert peer_kinds == ["session.update", "input_audio_buffer.append"]
        assert bad_socket.closed and reopened
        h.assert_retired_generations()

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

        socket.recv, socket.wait_closed = recv, close
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
        ("held", (False, False, True, True, True, ("close",), 0, 0))
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
            assert h.sockets[1].sent[-1]["type"] == "input_audio_buffer.append"
            await successor.drain(CloseRequestReason.USER_STOP)
        finally:
            await asyncio.wait_for(h.provider.shutdown(), timeout=3)
        h.assert_retired_generations()

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

        socket.recv, socket.wait_closed, socket.abort = held_recv, close, abort
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
            assert not socket.closed and effects == ["abort"]
            with pytest.raises(OpenAIProviderProtocolError):
                await owner.drain(CloseRequestReason.DAEMON_SHUTDOWN)
            assert effects == ["abort", "abort"]

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
                ("abort", "abort", "abort", "close"),
                (False, False),
                True,
            )
            assert isinstance(results[0], asyncio.CancelledError)
            receipt = results[1]
            assert receipt.session_id == value.session_id
            assert receipt.delivery_error is None
            assert await owner.drain(CloseRequestReason.DAEMON_SHUTDOWN) == receipt
            assert effects == ["abort", "abort", "abort", "close"]
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
        h.assert_retired_generations()

    asyncio.run(asyncio.wait_for(scenario(), timeout=6))
