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

from .cleanup import finish_cleanup
from .openai_provider import (
    OPENAI_PROVIDER_NAME,
    OPENAI_TRANSCRIPTION_MODEL,
    OpenAIRealtimeAdapter,
    OpenAIRealtimeConfig,
    TranslationClosed,
    TranslationDelta,
    TranslationError,
    TranslationSessionEvent,
    build_input_audio_append_event,
    build_session_close_event,
    build_session_update_event,
    parse_translation_event,
)
from .provider_contract import (
    MAX_TERMINAL_UTTERANCES_PER_SESSION,
    CancelUtterance,
    CloseProviderSession,
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
    SampleFormat,
    SessionCloseReason,
    UpdateDebugText,
    UtteranceOutcome,
    make_provider_error,
)

PublishEvents = Callable[[tuple[object, ...], Callable[[], None]], Awaitable[None]]


class WebSocket(Protocol):
    transport: asyncio.Transport

    async def send(self, message: str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


WebSocketFactory = Callable[..., Awaitable[WebSocket]]


class OpenAIProviderProtocolError(ValueError):
    pass


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


@dataclass(slots=True)
class _Session:
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
        self._opening: dict[UUID, asyncio.Task] = {}
        self._closed = False
        self._logger = logging.Logger("translator.openai.transport")
        self._logger.disabled = True

    async def open_session(
        self,
        request: OpenProviderSession,
        publish: PublishEvents,
    ) -> tuple[ProviderSessionOpened, ProviderHealth]:
        if self._closed:
            raise OpenAIProviderProtocolError("provider_closed")
        if (
            request.session_id in self._sessions
            or request.session_id in self._opening
            or request.session_id in self._failed_opens
        ):
            raise OpenAIProviderProtocolError("duplicate_session")
        if self._failed_opens:
            raise OpenAIProviderProtocolError("provider_cleanup_required")
        self._validate_format(request.requested_input_format)
        self._validate_format(request.requested_output_format)
        self._opening[request.session_id] = asyncio.current_task()
        session = _Session(
            request, publish, debug_text_enabled=request.debug_text_enabled
        )
        try:
            session.generation = await self._prepare(session)
            session.generation.receiver = self._start_receiver(
                session, session.generation
            )
            self._sessions[request.session_id] = session
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
            raise OpenAIProviderProtocolError("openai_connection_failed") from None
        finally:
            self._opening.pop(request.session_id, None)

    async def _prepare(self, session: _Session) -> _Generation:
        request = session.request.model_copy(
            update={"debug_text_enabled": session.debug_text_enabled}
        )
        adapter = OpenAIRealtimeAdapter(self._config, environ=self._environ)
        preflight = adapter.preflight_open_session(request)
        if not preflight.can_start or preflight.connect_plan is None:
            raise OpenAIProviderProtocolError("openai_provider_unavailable")
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
                await ws.send(json.dumps(build_session_update_event(request)))
                updated = await self._handshake_event(ws, "session.updated")
                audio = updated.session.audio
                transcription = audio.input.transcription if audio.input else None
                if (
                    created.session.id != updated.session.id
                    or audio.output is None
                    or audio.output.language != request.target_language.value
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
            if ws is not None:
                await self._retire(session, session.generation)
            if isinstance(error, asyncio.CancelledError):
                raise
            raise OpenAIProviderProtocolError("openai_connection_failed") from None

    async def _handshake_event(
        self, ws: WebSocket, expected: str
    ) -> TranslationSessionEvent:
        while True:
            event = parse_translation_event(await ws.recv())
            if event is None:
                continue
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

    async def submit_frame(self, frame: ProviderInputFrame) -> None:
        session = self._active_session(frame.session_id)
        async with session.lock:
            if session.closed:
                raise OpenAIProviderProtocolError("open_session_required")
            self._validate_frame(session, frame)
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
                            json.dumps(build_session_close_event())
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
                raise OpenAIProviderProtocolError("openai_send_failed") from None

    async def cancel_utterance(self, request: CancelUtterance) -> None:
        session = self._active_session(request.session_id)
        async with session.lock:
            if request.direction_id != session.request.direction_id:
                raise OpenAIProviderProtocolError("direction_identity_mismatch")
            generation = session.generation
            if (
                generation
                and generation.binding
                and generation.binding.utterance_id == request.utterance_id
            ):
                await self._retire(session, generation, UtteranceOutcome.CANCELLED)

    async def update_debug_text(self, request: UpdateDebugText) -> None:
        session = self._active_session(request.session_id)
        async with session.lock:
            session.debug_text_enabled = request.enabled

    async def close_session(self, request: CloseProviderSession) -> None:
        opening = self._opening.get(request.session_id)
        if opening is not None and opening is not asyncio.current_task():
            opening.cancel()
            await asyncio.gather(opening, return_exceptions=True)
        session = self._sessions.get(request.session_id) or self._failed_opens.get(
            request.session_id
        )
        if session is None:
            return
        async with session.lock:
            if session.closed and session.generation is None:
                return
            session.closed = True
            try:
                if session.generation:
                    await self._retire(
                        session,
                        session.generation,
                        None
                        if request.session_id in self._failed_opens
                        else UtteranceOutcome.CANCELLED,
                    )
                if request.session_id in self._failed_opens:
                    return
                await self._publish(
                    session,
                    (
                        ProviderSessionClosed(
                            session_id=request.session_id,
                            direction_id=session.request.direction_id,
                            event_sequence=session.next_sequence(),
                            reason=_close_reason(request.reason),
                        ),
                    ),
                )
            finally:
                if session.generation is None:
                    self._sessions.pop(request.session_id, None)
                    self._failed_opens.pop(request.session_id, None)

    async def wait_publications(self, session_id: UUID) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            async with session.lock:
                if session.failed:
                    raise OpenAIProviderProtocolError("provider_publication_failed")

    async def shutdown(self) -> None:
        self._closed = True
        results = await asyncio.gather(
            *(
                self.close_session(
                    CloseProviderSession(
                        session_id=session_id,
                        reason=CloseRequestReason.DAEMON_SHUTDOWN,
                    )
                )
                for session_id in tuple(self._opening)
                + tuple(self._sessions)
                + tuple(self._failed_opens)
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
                        await self._publish(
                            session, (self._error(session, generation),)
                        )
                    elif (
                        isinstance(event, TranslationClosed)
                        and generation.input_complete
                    ):
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
                        return
                    elif isinstance(event, TranslationDelta) and binding is not None:
                        if (
                            event.type == "session.input_transcript.delta"
                            and not generation.source_transcript
                        ):
                            raise OpenAIProviderProtocolError(
                                "source_transcription_not_configured"
                            )
                        await self._map_delta(session, generation, event.model_dump())
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
                    "type": "session.output_audio.delta",
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
    ) -> None:
        generation.revoked = True
        binding = generation.binding
        if binding is not None:
            session.terminal_ids.add(binding.utterance_id)
        batch = [self._error(session, generation)] if error else []
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
            async with asyncio.timeout(1.8):
                await ws.close()
        except Exception:
            try:
                ws.transport.abort()
            except Exception:
                raise OpenAIProviderProtocolError("openai_retirement_failed") from None

    @staticmethod
    async def _publish(session: _Session, batch: tuple[object, ...]) -> None:
        committed = False

        def commit() -> None:
            nonlocal committed
            committed = True

        async with asyncio.timeout(2):
            await session.publish(batch, commit)
        if not committed:
            raise RuntimeError("provider_publication_not_committed")

    @staticmethod
    def _error(session: _Session, generation: _Generation):
        binding = generation.binding
        return make_provider_error(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            stream_id=binding.stream_id if binding else None,
            utterance_id=binding.utterance_id if binding else None,
            event_sequence=session.next_sequence(),
            code=SafeErrorCode.PROVIDER_UNAVAILABLE,
            retryable=True,
        )

    def _health(self, session: _Session) -> ProviderHealth:
        return ProviderHealth(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            event_sequence=session.next_sequence(),
            provider_id=session.request.provider_id,
            provider_name=OPENAI_PROVIDER_NAME,
            state=ProviderState.READY,
            models=(
                ModelHealth(
                    kind=ModelKind.SPEECH_TO_SPEECH,
                    id=self._config.model,
                    state=ModelState.READY,
                    device=ComputeDevice.CLOUD,
                ),
            ),
            queues=ProviderQueues(
                provider_input_buffered_ms=0,
                provider_output_buffered_ms=0,
                queue_lag_ms=0,
            ),
        )

    def _active_session(self, session_id: UUID) -> _Session:
        session = self._sessions.get(session_id)
        if session is None or session.closed or session.failed:
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
