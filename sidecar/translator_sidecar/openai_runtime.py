"""One Realtime Translation connection per local utterance."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

import numpy as np
import soxr
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from .cleanup import finish_cleanup
from .openai_provider import (
    OPENAI_PROVIDER_NAME,
    OPENAI_REALTIME_VOICE,
    OPENAI_TRANSCRIPTION_MODEL,
    InputCommitted,
    OpenAIRealtimeAdapter,
    OpenAIRealtimeConfig,
    SourceTranscript,
    SourceTranscriptCompleted,
    TranslationDelta,
    TranslationError,
    TranslationResponse,
    TranslationSessionEvent,
    build_input_audio_append_event,
    build_input_audio_commit_event,
    build_response_create_event,
    build_session_update_event,
    parse_translation_event,
)
from .provider_contract import (
    MAX_TERMINAL_UTTERANCES_PER_SESSION,
    SAFE_ERROR_MESSAGES,
    CancelUtterance,
    CloseRequestReason,
    ComputeDevice,
    ModelHealth,
    ModelKind,
    ModelState,
    OpenProviderSession,
    PcmFormat,
    ProviderAudioDelta,
    ProviderCapabilities,
    ProviderHealth,
    ProviderInputFrame,
    ProviderQueues,
    ProviderSessionClosed,
    ProviderSessionOpened,
    ProviderState,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SafeErrorSummary,
    SampleFormat,
    SessionCloseReason,
    UpdateDebugText,
    UtteranceOutcome,
    make_provider_error,
)
from .provider_registry import SessionDrainReceipt

PublishEvents = Callable[[tuple[object, ...], Callable[[], None]], Awaitable[None]]


class WebSocket(Protocol):
    transport: asyncio.Transport

    async def send(self, message: str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def wait_closed(self) -> None: ...


WebSocketFactory = Callable[..., Awaitable[WebSocket]]


class OpenAIProviderProtocolError(ValueError):
    pass


class OpenAIProviderFailure(OpenAIProviderProtocolError):
    def __init__(
        self,
        code: SafeErrorCode,
        *,
        retryable: bool,
        resource_exhausted: bool = False,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.retryable = retryable
        self.resource_exhausted = resource_exhausted


class OpenAIProviderTransportFailure(OpenAIProviderFailure):
    pass


def _wire_failure(event: TranslationError) -> OpenAIProviderFailure:
    authentication = event.error.type == "authentication_error" or event.error.code in {
        "invalid_api_key",
        "invalid_authentication",
        "permission_denied",
    }
    hard_quota = event.error.code in {
        "insufficient_quota",
        "billing_hard_limit_reached",
    }
    rate_limited = (
        event.error.code == "rate_limit_exceeded"
        or event.error.type == "rate_limit_error"
    )
    return OpenAIProviderFailure(
        SafeErrorCode.PROVIDER_AUTH_FAILED
        if authentication
        else SafeErrorCode.PROVIDER_UNAVAILABLE,
        retryable=(event.error.type == "server_error" or rate_limited)
        and not authentication
        and not hard_quota,
        resource_exhausted=hard_quota or rate_limited,
    )


@dataclass(frozen=True, slots=True)
class _Binding:
    stream_id: UUID
    utterance_id: UUID


@dataclass(slots=True)
class _Generation:
    ws: WebSocket
    adapter: OpenAIRealtimeAdapter
    source_transcript: bool
    binding: _Binding | None = None
    receiver: asyncio.Task[None] | None = None
    timeout: asyncio.Timeout | None = None
    deadline: float | None = None
    input_complete: bool = False
    revoked: bool = False
    disposed: bool = False
    last_audio_sequence: int | None = None
    response_id: str | None = None
    input_item_id: str | None = None
    output_item_id: str | None = None
    source_transcript_seen: bool = False


@dataclass(slots=True)
class _Session:
    owner: OpenAIRealtimeProvider
    request: OpenProviderSession
    publish: PublishEvents
    generation: _Generation | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    debug_text_enabled: bool = False
    event_sequence: int = 0
    closed: bool = False
    failed: bool = False
    stream_id: UUID | None = None
    last_input_sequence: int | None = None
    terminal_ids: set[UUID] = field(default_factory=set)
    opening_task: asyncio.Task | None = None
    prefetch_failure: OpenAIProviderFailure | None = None
    opened: bool = False
    draining: bool = False
    drain_task: asyncio.Task[SessionDrainReceipt] | None = None
    delivery_error: SafeErrorCode | None = None

    async def open(self) -> tuple[ProviderSessionOpened, ProviderHealth]:
        return await self.owner._open_session(self)

    async def drain(self, reason: CloseRequestReason) -> SessionDrainReceipt:
        self.draining = True
        if self.drain_task is None or (
            self.drain_task.done()
            and (self.drain_task.cancelled() or self.drain_task.exception() is not None)
        ):
            self.drain_task = asyncio.create_task(
                self.owner._drain_session(self, reason)
            )
        return await finish_cleanup(self.drain_task)

    def next_sequence(self) -> int:
        self.event_sequence += 1
        return self.event_sequence


class OpenAIRealtimeProvider:
    def __init__(
        self,
        config: OpenAIRealtimeConfig | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        websocket_factory: WebSocketFactory | None = None,
        now_ns: Callable[[], int] = time.monotonic_ns,
        drain_timeout_seconds: float = 10,
    ) -> None:
        if not 0 < drain_timeout_seconds <= 30:
            raise ValueError("invalid_drain_timeout")
        self._config = config or OpenAIRealtimeConfig(cloud_opt_in=True)
        self._environ = environ if environ is not None else os.environ
        self._connect = websocket_factory or connect
        self._now_ns = now_ns
        self._drain_timeout = drain_timeout_seconds
        self._sessions: dict[UUID, _Session] = {}
        self._failed_opens: dict[UUID, _Session] = {}
        self._closed = False
        self._logger = logging.Logger("translator.openai.transport")
        self._logger.disabled = True

    @property
    def _opening(self) -> dict[UUID, asyncio.Task]:
        return {
            session.request.session_id: session.opening_task
            for session in (*self._sessions.values(), *self._failed_opens.values())
            if session.opening_task is not None
        }

    def reserve_session(
        self,
        request: OpenProviderSession,
        publish: PublishEvents,
    ) -> _Session:
        if self._closed:
            raise OpenAIProviderProtocolError("provider_closed")
        if (
            request.session_id in self._sessions
            or request.session_id in self._failed_opens
        ):
            raise OpenAIProviderProtocolError("duplicate_session")
        if self._failed_opens or any(
            session.failed or session.closed or session.draining
            for session in self._sessions.values()
        ):
            raise OpenAIProviderProtocolError("provider_cleanup_required")
        self._validate_format(request.requested_input_format)
        self._validate_format(request.requested_output_format)
        if request.voice_profile.has_overrides:
            raise OpenAIProviderProtocolError("voice profile override is unsupported")
        session = _Session(
            self, request, publish, debug_text_enabled=request.debug_text_enabled
        )
        self._sessions[request.session_id] = session
        return session

    def _open_session(
        self, session: _Session
    ) -> asyncio.Task[tuple[ProviderSessionOpened, ProviderHealth]]:
        if (
            self._closed
            or session.closed
            or session.draining
            or session.opened
            or session.opening_task is not None
            or self._sessions.get(session.request.session_id) is not session
        ):
            raise OpenAIProviderProtocolError("open_session_required")
        coroutine = self._run_open_session(session)
        try:
            session.opening_task = asyncio.create_task(coroutine)
        except BaseException:
            coroutine.close()
            session.closed = True
            if self._sessions.get(session.request.session_id) is session:
                self._sessions.pop(session.request.session_id)
            raise
        return session.opening_task

    async def _run_open_session(
        self, session: _Session
    ) -> tuple[ProviderSessionOpened, ProviderHealth]:
        request = session.request
        try:
            session.generation = await self._prepare(session)
            session.generation.receiver = self._start_receiver(
                session, session.generation
            )
            session.opened = True
            opened = ProviderSessionOpened(
                session_id=request.session_id,
                direction_id=request.direction_id,
                event_sequence=session.next_sequence(),
                negotiated_input_format=request.requested_input_format,
                negotiated_output_format=request.requested_output_format,
                capabilities=ProviderCapabilities(
                    transcript_delta=request.debug_text_enabled,
                    translation_delta=True,
                    cancellation=True,
                    cloud_egress=True,
                ),
            )
            return opened, self._health(session)
        except BaseException as error:
            session.closed = True
            try:
                if session.generation is not None and not session.generation.revoked:
                    await self._retire(session, session.generation)
            finally:
                self._sessions.pop(request.session_id, None)
                if session.generation is not None:
                    self._failed_opens[request.session_id] = session
            if isinstance(error, asyncio.CancelledError):
                raise
            if isinstance(error, OpenAIProviderFailure):
                raise
            raise OpenAIProviderProtocolError("openai_connection_failed") from None
        finally:
            session.opening_task = None

    async def _prepare(self, session: _Session) -> _Generation:
        request = session.request.model_copy(
            update={"debug_text_enabled": session.debug_text_enabled}
        )
        adapter = OpenAIRealtimeAdapter(self._config, environ=self._environ)
        preflight = adapter.preflight_open_session(request)
        if not preflight.can_start or preflight.connect_plan is None:
            assert preflight.error is not None
            raise OpenAIProviderFailure(preflight.error.code, retryable=False)
        ws = None
        try:
            async with asyncio.timeout(10):
                ws = await self._connect(
                    preflight.connect_plan["uri"],
                    additional_headers={
                        "Authorization": "Bearer "
                        + self._environ[self._config.credential_env_name].strip(),
                        "OpenAI-Safety-Identifier": "translator-sidecar-openai-runtime",
                    },
                    open_timeout=10,
                    close_timeout=1,
                    max_size=1_048_576,
                    max_queue=16,
                    logger=self._logger,
                )
                session.generation = _Generation(
                    ws, adapter, request.debug_text_enabled
                )
                created = await self._handshake_event(ws, "session.created")
                settings = build_session_update_event(request)
                await ws.send(json.dumps(settings))
                updated = await self._handshake_event(ws, "session.updated")
                audio = updated.session.audio
                transcription = (
                    audio.input.transcription if audio and audio.input else None
                )
                if (
                    created.session.id != updated.session.id
                    or updated.session.instructions
                    != settings["session"]["instructions"]
                    or updated.session.output_modalities != ["audio"]
                    or audio is None
                    or audio.input is None
                    or audio.input.turn_detection is not None
                    or audio.output is None
                    or audio.output.voice != OPENAI_REALTIME_VOICE
                    or (transcription.model if transcription else None)
                    != (
                        OPENAI_TRANSCRIPTION_MODEL
                        if request.debug_text_enabled
                        else None
                    )
                ):
                    raise OpenAIProviderProtocolError("translation_handshake_mismatch")
            return session.generation
        except BaseException as error:
            try:
                if ws is not None:
                    await self._retire(session, session.generation)
            finally:
                if isinstance(error, asyncio.CancelledError):
                    raise error
            if isinstance(error, OpenAIProviderFailure):
                raise error
            if isinstance(error, (OSError, TimeoutError, ConnectionClosed)):
                raise OpenAIProviderTransportFailure(
                    SafeErrorCode.PROVIDER_UNAVAILABLE, retryable=True
                ) from None
            if isinstance(error, InvalidStatus):
                status = error.response.status_code
                raise OpenAIProviderFailure(
                    SafeErrorCode.PROVIDER_AUTH_FAILED
                    if status in {401, 403}
                    else SafeErrorCode.PROVIDER_UNAVAILABLE,
                    retryable=status >= 500,
                    resource_exhausted=status == 429,
                ) from None
            raise OpenAIProviderProtocolError("openai_connection_failed") from None

    async def _handshake_event(
        self, ws: WebSocket, expected: str
    ) -> TranslationSessionEvent:
        while True:
            event = parse_translation_event(await ws.recv())
            if event is None:
                continue
            if isinstance(event, TranslationError):
                raise _wire_failure(event)
            if (
                not isinstance(event, TranslationSessionEvent)
                or event.type != expected
                or event.session.model != self._config.model
            ):
                raise OpenAIProviderProtocolError("translation_handshake_mismatch")
            return event

    def _start_receiver(
        self, session: _Session, generation: _Generation
    ) -> asyncio.Task[None]:
        coroutine = self._receive(session, generation)
        try:
            return asyncio.create_task(coroutine, name="openai-generation-receiver")
        except BaseException:
            coroutine.close()
            raise

    def _start_prefetch(self, session: _Session) -> None:
        if (
            self._closed
            or self._failed_opens
            or any(
                owner.closed or owner.draining or owner.failed
                for owner in self._sessions.values()
            )
            or session.opening_task is not None
        ):
            return
        coroutine = self._prefetch(session)
        try:
            session.opening_task = asyncio.create_task(
                coroutine, name="openai-idle-prepare"
            )
            session.prefetch_failure = None
        except BaseException:
            coroutine.close()
            raise

    async def _prefetch(self, session: _Session) -> None:
        try:
            async with session.lock:
                if session.closed or session.draining or session.generation is not None:
                    return
                try:
                    generation = await self._prepare(session)
                    generation.receiver = self._start_receiver(session, generation)
                except Exception as error:
                    failure = (
                        error
                        if isinstance(error, OpenAIProviderFailure)
                        else OpenAIProviderFailure(
                            SafeErrorCode.PROVIDER_UNAVAILABLE, retryable=False
                        )
                    )
                    try:
                        if (
                            session.generation is not None
                            and not session.generation.revoked
                        ):
                            await self._retire(session, session.generation)
                    except Exception:
                        session.failed = True
                    session.prefetch_failure = failure
                    if session.failed or not failure.retryable:
                        session.failed = True
                        try:
                            await self._publish(
                                session,
                                (
                                    self._error(session, None, failure),
                                    self._health(session, failure),
                                ),
                            )
                        except Exception:
                            pass  # _publish retains delivery failure for the drain receipt.
        finally:
            session.opening_task = None

    async def _cancel_opening(self, session: _Session) -> None:
        opening = session.opening_task
        if opening is not None and opening is not asyncio.current_task():
            opening.cancel()
            await asyncio.gather(opening, return_exceptions=True)

    async def submit_frame(self, frame: ProviderInputFrame) -> None:
        session = self._active_session(frame.session_id)
        async with session.lock:
            self._active_session(frame.session_id)
            self._validate_frame(session, frame)
            if session.prefetch_failure is not None:
                failure, session.prefetch_failure = session.prefetch_failure, None
                raise failure
            generation = session.generation
            binding = _Binding(frame.stream_id, frame.utterance_id)
            if (
                generation
                and generation.binding
                and (
                    generation.binding.utterance_id == frame.utterance_id
                    and generation.binding.stream_id != frame.stream_id
                )
            ):
                raise OpenAIProviderProtocolError("stream_identity_mismatch")
            if generation is not None and generation.binding not in (None, binding):
                await self._retire(session, generation, UtteranceOutcome.CANCELLED)
                generation = None
            if generation is None:
                generation = await self._prepare(session)
                session.generation = generation
                try:
                    generation.receiver = self._start_receiver(session, generation)
                except BaseException:
                    await self._retire(session, generation)
                    raise
            if generation.input_complete:
                raise OpenAIProviderProtocolError("input_sequence_mismatch")
            if generation.revoked:
                raise OpenAIProviderProtocolError("openai_retirement_failed")
            if generation.binding is None:
                generation.binding = binding
            session.stream_id = frame.stream_id
            session.last_input_sequence = frame.sequence
            try:
                async with asyncio.timeout(2):
                    await generation.ws.send(
                        json.dumps(
                            build_input_audio_append_event(
                                self._resample_pcm(
                                    frame.pcm, frame.sample_rate_hz, 24_000
                                )
                            )
                        )
                    )
                    if frame.end_of_utterance:
                        generation.input_complete = True
                        await generation.ws.send(
                            json.dumps(build_input_audio_commit_event())
                        )
                        await generation.ws.send(
                            json.dumps(build_response_create_event())
                        )
                        generation.deadline = (
                            asyncio.get_running_loop().time() + self._drain_timeout
                        )
                        if generation.timeout is not None:
                            generation.timeout.reschedule(generation.deadline)
            except BaseException as error:
                await self._retire(
                    session, generation, UtteranceOutcome.DROPPED, error=True
                )
                if isinstance(error, asyncio.CancelledError):
                    raise
                if isinstance(error, (OSError, TimeoutError, ConnectionClosed)):
                    raise OpenAIProviderTransportFailure(
                        SafeErrorCode.PROVIDER_UNAVAILABLE, retryable=True
                    ) from None
                raise OpenAIProviderProtocolError("openai_send_failed") from None

    async def cancel_utterance(self, request: CancelUtterance) -> None:
        session = self._active_session(request.session_id)
        if request.direction_id != session.request.direction_id:
            raise OpenAIProviderProtocolError("direction_identity_mismatch")
        if session.generation is None or session.generation.binding is None:
            return
        async with session.lock:
            generation = session.generation
            if (
                generation
                and generation.binding
                and generation.binding.utterance_id == request.utterance_id
            ):
                await self._retire(session, generation, UtteranceOutcome.CANCELLED)
                self._start_prefetch(session)

    async def update_debug_text(self, request: UpdateDebugText) -> None:
        session = self._active_session(request.session_id)
        await self._cancel_opening(session)
        async with session.lock:
            session.debug_text_enabled = request.enabled
            if session.generation is None or session.generation.binding is None:
                if session.generation is not None:
                    await self._retire(session, session.generation)
                self._start_prefetch(session)

    async def _drain_session(
        self, session: _Session, reason: CloseRequestReason
    ) -> SessionDrainReceipt:
        await self._cancel_opening(session)
        async with session.lock:
            session.closed = True
            try:
                if session.generation:
                    await self._retire(
                        session,
                        session.generation,
                        None if not session.opened else UtteranceOutcome.CANCELLED,
                    )
            except Exception:
                if session.generation is not None:
                    raise
            if session.opened and session.delivery_error is None:
                try:
                    await self._publish(
                        session,
                        (
                            ProviderSessionClosed(
                                session_id=session.request.session_id,
                                direction_id=session.request.direction_id,
                                event_sequence=session.next_sequence(),
                                reason=_close_reason(reason),
                            ),
                        ),
                    )
                except Exception:
                    pass  # _publish retained the bounded delivery error; resources are drained.
            for sessions in (self._sessions, self._failed_opens):
                if sessions.get(session.request.session_id) is session:
                    sessions.pop(session.request.session_id)
            return SessionDrainReceipt(
                session.request.session_id, session.delivery_error
            )

    async def shutdown(self) -> None:
        self._closed = True
        results = await asyncio.gather(
            *(
                session.drain(CloseRequestReason.DAEMON_SHUTDOWN)
                for session in (*self._sessions.values(), *self._failed_opens.values())
            ),
            return_exceptions=True,
        )
        if any(isinstance(result, BaseException) for result in results):
            raise OpenAIProviderProtocolError("openai_shutdown_failed")

    async def _receive(self, session: _Session, generation: _Generation) -> None:
        try:
            while not generation.revoked:
                async with asyncio.timeout_at(generation.deadline) as deadline:
                    generation.timeout = deadline
                    raw = await generation.ws.recv()
                generation.timeout = None
                binding = generation.binding
                event = parse_translation_event(raw)
                async with session.lock:
                    if generation.revoked:
                        return
                    if event is None:
                        continue
                    if isinstance(event, TranslationError):
                        failure = _wire_failure(event)
                        session.failed = not failure.retryable
                        await self._retire(
                            session,
                            generation,
                            UtteranceOutcome.DROPPED,
                            error=True,
                            failure=failure,
                        )
                        if failure.retryable and binding is not None:
                            self._start_prefetch(session)
                        return
                    elif isinstance(event, InputCommitted):
                        if (
                            not generation.input_complete
                            or generation.input_item_id is not None
                        ):
                            raise OpenAIProviderProtocolError(
                                "input_commit_identity_mismatch"
                            )
                        generation.input_item_id = event.item_id
                    elif (
                        isinstance(event, TranslationResponse)
                        and event.type == "response.created"
                    ):
                        if (
                            not generation.input_complete
                            or generation.response_id is not None
                            or event.response.status != "in_progress"
                        ):
                            raise OpenAIProviderProtocolError(
                                "response_identity_mismatch"
                            )
                        generation.response_id = event.response.id
                    elif (
                        isinstance(event, TranslationResponse)
                        and event.type == "response.done"
                    ):
                        if (
                            not generation.input_complete
                            or event.response.id != generation.response_id
                        ):
                            raise OpenAIProviderProtocolError(
                                "response_identity_mismatch"
                            )
                        if event.response.status != "completed":
                            failure = None
                            if (
                                event.response.status == "failed"
                                and event.response.status_details
                            ):
                                failure = _wire_failure(
                                    TranslationError.model_validate(
                                        {
                                            "type": "error",
                                            "event_id": event.event_id,
                                            "error": event.response.status_details.get(
                                                "error"
                                            ),
                                        }
                                    )
                                )
                                session.failed = not failure.retryable
                            await self._retire(
                                session,
                                generation,
                                UtteranceOutcome.CANCELLED
                                if event.response.status == "cancelled"
                                else UtteranceOutcome.DROPPED,
                                error=event.response.status != "cancelled",
                                failure=failure,
                            )
                            if event.response.status == "cancelled" or (
                                failure is not None and failure.retryable
                            ):
                                self._start_prefetch(session)
                            return
                        await self._flush_audio(session, generation)
                        await self._retire(
                            session,
                            generation,
                            (
                                UtteranceOutcome.COMPLETED
                                if generation.last_audio_sequence is not None
                                else UtteranceOutcome.DROPPED
                            ),
                        )
                        if generation.last_audio_sequence is not None:
                            self._start_prefetch(session)
                        return
                    elif isinstance(event, TranslationDelta) and binding is not None:
                        if (
                            event.response_id != generation.response_id
                            or generation.response_id is None
                        ):
                            raise OpenAIProviderProtocolError(
                                "response_identity_mismatch"
                            )
                        if (
                            event.output_index != 0
                            or event.content_index != 0
                            or generation.output_item_id not in (None, event.item_id)
                        ):
                            raise OpenAIProviderProtocolError(
                                "response_item_identity_mismatch"
                            )
                        generation.output_item_id = event.item_id
                        await self._map_delta(session, generation, event.model_dump())
                    elif (
                        isinstance(event, (SourceTranscript, SourceTranscriptCompleted))
                        and binding is not None
                    ):
                        if not generation.source_transcript:
                            raise OpenAIProviderProtocolError(
                                "source_transcription_not_configured"
                            )
                        if (
                            event.item_id != generation.input_item_id
                            or generation.input_item_id is None
                            or event.content_index != 0
                        ):
                            raise OpenAIProviderProtocolError(
                                "input_commit_identity_mismatch"
                            )
                        payload = event.model_dump()
                        if (
                            isinstance(event, SourceTranscriptCompleted)
                            and generation.source_transcript_seen
                        ):
                            payload["transcript"] = ""
                        await self._map_delta(session, generation, payload)
                        generation.source_transcript_seen = True
                    else:
                        raise OpenAIProviderProtocolError(
                            "unexpected_translation_event"
                        )
        except asyncio.CancelledError:
            raise
        except Exception:
            async with session.lock:
                if generation.revoked:
                    session.failed = True
                else:
                    try:
                        await self._retire(
                            session, generation, UtteranceOutcome.DROPPED, error=True
                        )
                    except Exception:
                        session.failed = True

    async def _map_delta(
        self, session: _Session, generation: _Generation, event: dict
    ) -> None:
        binding = generation.binding
        assert binding is not None
        mapped = generation.adapter.map_realtime_events(
            session.request.model_copy(
                update={"debug_text_enabled": session.debug_text_enabled}
            ),
            event,
            stream_id=binding.stream_id,
            utterance_id=binding.utterance_id,
            now_ns=self._now_ns(),
        )
        converted = []
        for item in mapped:
            updates: dict[str, Any] = {"event_sequence": session.next_sequence()}
            if isinstance(item, ProviderAudioDelta):
                target = session.request.requested_output_format
                updates.update(
                    sample_rate_hz=target.sample_rate_hz,
                    pcm=self._resample_pcm(item.pcm, 24_000, target.sample_rate_hz),
                )
                generation.last_audio_sequence = item.sequence
            converted.append(item.model_copy(update=updates))
        if converted:
            await self._publish(session, tuple(converted))

    async def _flush_audio(self, session: _Session, generation: _Generation) -> None:
        if generation.binding is None:
            return
        key = (session.request.session_id, generation.binding.utterance_id)
        remainder = generation.adapter._audio_remainders.get(key, b"")
        if len(remainder) % 2:
            raise OpenAIProviderProtocolError("incomplete_pcm_sample")
        if remainder:
            await self._map_delta(
                session,
                generation,
                {
                    "type": "response.output_audio.delta",
                    "delta": build_input_audio_append_event(
                        bytes(960 - len(remainder))
                    )["audio"],
                },
            )

    async def _retire(
        self,
        session: _Session,
        generation: _Generation,
        outcome: UtteranceOutcome | None = None,
        *,
        error: bool = False,
        failure: OpenAIProviderFailure | None = None,
    ) -> None:
        generation.revoked = True
        binding = generation.binding
        if binding is not None:
            session.terminal_ids.add(binding.utterance_id)
        batch = [self._error(session, generation, failure)] if error else []
        if outcome is not None and binding is not None:
            batch.append(
                ProviderUtteranceFinal(
                    session_id=session.request.session_id,
                    direction_id=session.request.direction_id,
                    stream_id=binding.stream_id,
                    utterance_id=binding.utterance_id,
                    event_sequence=session.next_sequence(),
                    final_audio_sequence=generation.last_audio_sequence,
                    outcome=outcome,
                )
            )
        try:
            await finish_cleanup(self._dispose(generation, asyncio.current_task()))
        finally:
            generation.adapter.release_session(session.request.session_id)
            if generation.disposed:
                if session.generation is generation:
                    session.generation = None
                generation.binding = None
            else:
                session.failed = True
            generation.timeout = None
            generation.receiver = None
        if batch:
            await self._publish(session, tuple(batch))

    async def _dispose(
        self, generation: _Generation, owner: asyncio.Task | None
    ) -> None:
        if generation.receiver is not None and generation.receiver is not owner:
            generation.receiver.cancel()
            await asyncio.gather(generation.receiver, return_exceptions=True)
        await self._close_socket(generation.ws)
        generation.disposed = True

    @staticmethod
    async def _close_socket(ws: WebSocket) -> None:
        try:
            # Retired generations cannot wait for the remote closing handshake.
            ws.transport.abort()
            async with asyncio.timeout(1.8):
                await ws.wait_closed()
        except Exception:
            raise OpenAIProviderProtocolError("openai_retirement_failed") from None

    @staticmethod
    async def _publish(session: _Session, batch: tuple[object, ...]) -> None:
        committed = False

        def commit() -> None:
            nonlocal committed
            committed = True

        try:
            async with asyncio.timeout(2):
                await session.publish(batch, commit)
            if not committed:
                raise RuntimeError("provider_publication_not_committed")
        except Exception:
            session.delivery_error = SafeErrorCode.PROVIDER_UNAVAILABLE
            raise

    @staticmethod
    def _error(
        session: _Session,
        generation: _Generation | None,
        failure: OpenAIProviderFailure | None = None,
    ):
        binding = generation.binding if generation else None
        return make_provider_error(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            stream_id=binding.stream_id if binding else None,
            utterance_id=binding.utterance_id if binding else None,
            event_sequence=session.next_sequence(),
            code=failure.code if failure else SafeErrorCode.PROVIDER_UNAVAILABLE,
            retryable=failure.retryable if failure else True,
        )

    def _health(
        self, session: _Session, failure: OpenAIProviderFailure | None = None
    ) -> ProviderHealth:
        return ProviderHealth(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            event_sequence=session.next_sequence(),
            provider_id=session.request.provider_id,
            provider_name=OPENAI_PROVIDER_NAME,
            state=ProviderState.UNAVAILABLE if failure else ProviderState.READY,
            models=(
                ModelHealth(
                    kind=ModelKind.SPEECH_TO_SPEECH,
                    id=self._config.model,
                    state=ModelState.FAILED if failure else ModelState.READY,
                    device=ComputeDevice.CLOUD,
                    safe_error_code=failure.code.value if failure else None,
                ),
            ),
            queues=ProviderQueues(
                provider_input_buffered_ms=0,
                provider_output_buffered_ms=0,
                queue_lag_ms=0,
            ),
            safe_error=SafeErrorSummary(
                code=failure.code,
                message=SAFE_ERROR_MESSAGES[failure.code],
                retryable=failure.retryable,
            )
            if failure
            else None,
        )

    def _active_session(self, session_id: UUID) -> _Session:
        session = self._sessions.get(session_id)
        if (
            session is None
            or session.closed
            or session.failed
            or session.draining
            or not session.opened
        ):
            raise OpenAIProviderProtocolError("open_session_required")
        return session

    @staticmethod
    def _validate_format(value: PcmFormat) -> None:
        if (
            value.channels != 1
            or value.sample_format is not SampleFormat.S16LE
            or value.frame_duration_ms != 20
            or value.sample_rate_hz not in {16_000, 24_000, 48_000}
        ):
            raise OpenAIProviderProtocolError("unsupported_pcm_format")

    @staticmethod
    def _validate_frame(session: _Session, frame: ProviderInputFrame) -> None:
        expected = session.request.requested_input_format
        if (
            frame.direction_id != session.request.direction_id
            or frame.source_language != session.request.source_language
            or frame.target_language != session.request.target_language
            or frame.mode != session.request.mode
            or frame.sample_rate_hz != expected.sample_rate_hz
            or frame.channels != expected.channels
            or frame.sample_format != expected.sample_format
            or frame.frame_duration_ms != expected.frame_duration_ms
            or len(frame.pcm)
            != expected.sample_rate_hz * 2 * expected.frame_duration_ms // 1000
        ):
            raise OpenAIProviderProtocolError("input_frame_mismatch")
        if session.stream_id is not None and frame.stream_id != session.stream_id:
            raise OpenAIProviderProtocolError("stream_identity_mismatch")
        if frame.utterance_id in session.terminal_ids:
            raise OpenAIProviderProtocolError("utterance_already_terminal")
        active = session.generation.binding if session.generation else None
        if (active is None or active.utterance_id != frame.utterance_id) and (
            len(session.terminal_ids) + (active is not None)
            >= MAX_TERMINAL_UTTERANCES_PER_SESSION
        ):
            raise OpenAIProviderProtocolError("terminal_identity_capacity_exhausted")
        if (
            session.last_input_sequence is not None
            and frame.sequence <= session.last_input_sequence
        ):
            raise OpenAIProviderProtocolError("input_sequence_mismatch")

    @staticmethod
    def _resample_pcm(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
        if source_rate == target_rate:
            return pcm
        return (
            soxr.resample(
                np.frombuffer(pcm, dtype="<i2"), source_rate, target_rate, quality="QQ"
            )
            .astype("<i2", copy=False)
            .tobytes()
        )


def _close_reason(reason: CloseRequestReason) -> SessionCloseReason:
    return SessionCloseReason(reason.value)
