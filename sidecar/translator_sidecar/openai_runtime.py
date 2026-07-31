"""Runtime OpenAI Realtime Translation provider backend."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np
import soxr

from .openai_provider import (
    OPENAI_PROVIDER_NAME,
    build_input_audio_append_event,
    build_session_close_event,
    build_session_update_event,
    openai_pcm_format,
    OpenAIRealtimeAdapter,
    OpenAIRealtimeConfig,
)
from .provider_contract import (
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
    ProviderQueues,
    ProviderSessionClosed,
    ProviderSessionOpened,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SampleFormat,
    SessionCloseReason,
    UpdateDebugText,
    UtteranceOutcome,
    make_provider_error,
)


PublishEvents = Callable[[tuple[object, ...], Callable[[], None]], Awaitable[None]]
WebSocketFactory = Callable[[], Any]

_CONNECT_TIMEOUT_SECONDS = 10
_FINAL_IDLE_MS = 225
_NO_SPEECH_FINAL_MS = 5_500


class OpenAIProviderProtocolError(ValueError):
    pass


@dataclass(slots=True)
class _UtteranceState:
    stream_id: UUID
    utterance_id: UUID
    input_complete: bool = False
    drop_pending: bool = False
    final_sent: bool = False
    last_audio_sequence: int | None = None
    pending_audio: list[ProviderAudioDelta] = field(default_factory=list)
    final_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _OpenAISession:
    request: OpenProviderSession
    publish: PublishEvents
    ws: Any
    receiver_task: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    debug_text_enabled: bool = False
    close_requested: bool = False
    requested_close_reason: SessionCloseReason | None = None
    closed: bool = False
    closed_published: bool = False
    closed_event: asyncio.Event = field(default_factory=asyncio.Event)
    current_stream_id: UUID | None = None
    current_utterance_id: UUID | None = None
    utterances: dict[UUID, _UtteranceState] = field(default_factory=dict)


class OpenAIRealtimeProvider:
    def __init__(
        self,
        config: OpenAIRealtimeConfig | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        websocket_factory: WebSocketFactory | None = None,
        now_ns: Callable[[], int] = time.monotonic_ns,
        final_idle_ms: int = _FINAL_IDLE_MS,
        no_speech_final_ms: int = _NO_SPEECH_FINAL_MS,
    ) -> None:
        self._config = config or OpenAIRealtimeConfig(cloud_opt_in=True)
        self._environ = environ if environ is not None else os.environ
        self._websocket_factory = websocket_factory or self._default_websocket
        self._now_ns = now_ns
        self._final_idle_ms = final_idle_ms
        self._no_speech_final_ms = no_speech_final_ms
        self._adapter = OpenAIRealtimeAdapter(
            self._config,
            environ=self._environ,
        )
        self._sessions: dict[UUID, _OpenAISession] = {}
        self._event_sequences: dict[UUID, int] = {}
        self._closed = False

    async def open_session(
        self,
        request: OpenProviderSession,
        publish: PublishEvents,
    ) -> tuple[ProviderSessionOpened, ProviderHealth]:
        if self._closed:
            raise OpenAIProviderProtocolError("provider_closed")
        if request.provider_id.value != "openai":
            raise OpenAIProviderProtocolError("unsupported_provider")
        if request.session_id in self._sessions:
            raise OpenAIProviderProtocolError("duplicate_session")
        self._validate_format(request.requested_input_format)
        self._validate_format(request.requested_output_format)
        preflight = self._adapter.preflight_open_session(request)
        if not preflight.can_start or preflight.connect_plan is None:
            raise OpenAIProviderProtocolError("openai_provider_unavailable")

        api_key = self._environ.get(self._config.credential_env_name, "").strip()
        if not api_key:
            raise OpenAIProviderProtocolError("openai_provider_unavailable")
        ws = self._websocket_factory()
        headers = [
            f"Authorization: Bearer {api_key}",
            "OpenAI-Safety-Identifier: translator-sidecar-openai-runtime",
        ]
        await asyncio.to_thread(
            ws.connect,
            preflight.connect_plan["uri"],
            header=headers,
            timeout=_CONNECT_TIMEOUT_SECONDS,
        )
        await asyncio.to_thread(
            ws.send,
            json.dumps(build_session_update_event(request)),
        )
        session = _OpenAISession(
            request=request,
            publish=publish,
            ws=ws,
            debug_text_enabled=request.debug_text_enabled,
        )
        self._sessions[request.session_id] = session
        session.receiver_task = asyncio.create_task(
            self._receive_loop(request.session_id)
        )
        opened = ProviderSessionOpened(
            session_id=request.session_id,
            direction_id=request.direction_id,
            event_sequence=self._next_event_sequence(request.session_id),
            negotiated_input_format=request.requested_input_format,
            negotiated_output_format=request.requested_output_format,
            capabilities=ProviderCapabilities(
                transcript_delta=request.debug_text_enabled,
                translation_delta=request.debug_text_enabled,
                cancellation=True,
                cloud_egress=True,
            ),
        )
        return opened, self._health(session)

    async def submit_frame(self, frame) -> None:
        session = self._active_session(frame.session_id)
        async with session.lock:
            self._validate_frame(session, frame)
            active = self._active_output_utterance(session)
            if active is not None and frame.utterance_id != active.utterance_id:
                dropped = session.utterances.setdefault(
                    frame.utterance_id,
                    _UtteranceState(
                        stream_id=frame.stream_id,
                        utterance_id=frame.utterance_id,
                        drop_pending=True,
                    ),
                )
                dropped.drop_pending = True
                if frame.end_of_utterance:
                    dropped.input_complete = True
                    await self._publish_overflow_drop_locked(session, dropped)
                return
            utterance = session.utterances.setdefault(
                frame.utterance_id,
                _UtteranceState(
                    stream_id=frame.stream_id,
                    utterance_id=frame.utterance_id,
                ),
            )
            session.current_stream_id = frame.stream_id
            session.current_utterance_id = frame.utterance_id
            wire_pcm = self._resample_pcm(
                frame.pcm,
                source_rate=frame.sample_rate_hz,
                target_rate=openai_pcm_format().sample_rate_hz,
            )
            await asyncio.to_thread(
                session.ws.send,
                json.dumps(build_input_audio_append_event(wire_pcm)),
            )
            if frame.end_of_utterance:
                utterance.input_complete = True
                await self._publish_pending_audio_locked(session, utterance)
                self._schedule_final_locked(
                    session,
                    utterance,
                    self._no_speech_final_ms
                    if utterance.last_audio_sequence is None
                    else self._final_idle_ms,
                    UtteranceOutcome.DROPPED
                    if utterance.last_audio_sequence is None
                    else UtteranceOutcome.COMPLETED,
                )

    async def cancel_utterance(self, request: CancelUtterance) -> None:
        session = self._active_session(request.session_id)
        async with session.lock:
            utterance = session.utterances.get(request.utterance_id)
            if utterance is None or utterance.final_sent:
                return
            await self._publish_final_locked(
                session,
                utterance,
                UtteranceOutcome.CANCELLED,
            )

    async def update_debug_text(self, request: UpdateDebugText) -> None:
        session = self._active_session(request.session_id)
        async with session.lock:
            session.debug_text_enabled = request.enabled

    async def close_session(self, request: CloseProviderSession) -> None:
        session = self._sessions.get(request.session_id)
        if session is None:
            return
        send_close = False
        async with session.lock:
            if session.closed_published:
                return
            session.requested_close_reason = _close_reason(request.reason)
            if not session.close_requested:
                session.close_requested = True
                send_close = True
        if send_close:
            try:
                await asyncio.to_thread(
                    session.ws.send,
                    json.dumps(build_session_close_event()),
                )
            except Exception:
                pass
        try:
            await asyncio.wait_for(session.closed_event.wait(), timeout=2)
        except asyncio.TimeoutError:
            async with session.lock:
                if not session.closed_published:
                    session.closed = True
                    try:
                        await asyncio.to_thread(session.ws.close)
                    except Exception:
                        pass
                    await self._publish_closed_locked(
                        session,
                        SessionCloseReason.CLOSE_TIMEOUT,
                    )
        else:
            try:
                await asyncio.to_thread(session.ws.close)
            except Exception:
                pass

    async def wait_publications(self, session_id: UUID) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        if (
            session.receiver_task is not None
            and session.receiver_task is not asyncio.current_task()
        ):
            try:
                await asyncio.wait_for(session.receiver_task, timeout=1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                session.receiver_task.cancel()
                await asyncio.gather(
                    session.receiver_task,
                    return_exceptions=True,
                )
        self._sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        self._closed = True
        for session_id in tuple(self._sessions):
            await self.close_session(
                CloseProviderSession(
                    session_id=session_id,
                    reason=CloseRequestReason.DAEMON_SHUTDOWN,
                )
            )
            await self.wait_publications(session_id)

    async def _receive_loop(self, session_id: UUID) -> None:
        while True:
            session = self._sessions.get(session_id)
            if session is None:
                return
            try:
                raw = await asyncio.to_thread(session.ws.recv)
            except Exception:
                await self._publish_receive_failure(session)
                return
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            async with session.lock:
                if event_type == "session.closed":
                    session.closed = True
                    await self._publish_closed_locked(
                        session,
                        session.requested_close_reason
                        or SessionCloseReason.USER_STOP,
                    )
                    return
                if event_type == "error":
                    await self._publish_provider_error_locked(session)
                    continue
                await self._handle_realtime_event_locked(session, event)

    async def _handle_realtime_event_locked(
        self,
        session: _OpenAISession,
        event: Mapping[str, Any],
    ) -> None:
        stream_id = session.current_stream_id
        utterance_id = session.current_utterance_id
        if stream_id is None or utterance_id is None:
            return
        request = session.request.model_copy(
            update={"debug_text_enabled": session.debug_text_enabled}
        )
        mapped = self._adapter.map_realtime_events(
            request,
            event,
            stream_id=stream_id,
            utterance_id=utterance_id,
            now_ns=self._now_ns(),
        )
        if not mapped:
            return
        converted: list[object] = []
        utterance = session.utterances.setdefault(
            utterance_id,
            _UtteranceState(stream_id=stream_id, utterance_id=utterance_id),
        )
        if utterance.final_sent:
            return
        for item in mapped:
            if isinstance(item, ProviderAudioDelta):
                converted_audio = self._convert_output_audio(session, item)
                if utterance.input_complete:
                    converted.append(
                        self._with_next_event_sequence(session, converted_audio)
                    )
                    utterance.last_audio_sequence = converted_audio.sequence
                else:
                    utterance.pending_audio.append(converted_audio)
            elif isinstance(item, (ProviderTranscriptDelta, ProviderTranslationDelta)):
                converted.append(self._with_next_event_sequence(session, item))
        if converted:
            await self._publish_locked(session, tuple(converted))
            if any(isinstance(item, ProviderAudioDelta) for item in converted):
                self._schedule_final_locked(
                    session,
                    utterance,
                    self._final_idle_ms,
                    UtteranceOutcome.COMPLETED,
                )

    async def _publish_pending_audio_locked(
        self,
        session: _OpenAISession,
        utterance: _UtteranceState,
    ) -> None:
        if not utterance.pending_audio:
            return
        batch = tuple(
            self._with_next_event_sequence(session, item)
            for item in utterance.pending_audio
        )
        utterance.pending_audio.clear()
        utterance.last_audio_sequence = batch[-1].sequence
        await self._publish_locked(session, batch)

    async def _publish_overflow_drop_locked(
        self,
        session: _OpenAISession,
        utterance: _UtteranceState,
    ) -> None:
        if utterance.final_sent:
            return
        utterance.final_sent = True
        await self._publish_locked(
            session,
            (
                make_provider_error(
                    session_id=session.request.session_id,
                    direction_id=session.request.direction_id,
                    stream_id=utterance.stream_id,
                    utterance_id=utterance.utterance_id,
                    event_sequence=self._next_event_sequence(
                        session.request.session_id
                    ),
                    code=SafeErrorCode.QUEUE_OVERFLOW,
                    retryable=True,
                ),
                ProviderUtteranceFinal(
                    session_id=session.request.session_id,
                    direction_id=session.request.direction_id,
                    stream_id=utterance.stream_id,
                    utterance_id=utterance.utterance_id,
                    event_sequence=self._next_event_sequence(
                        session.request.session_id
                    ),
                    final_audio_sequence=None,
                    outcome=UtteranceOutcome.DROPPED,
                ),
            ),
        )

    async def _publish_final_locked(
        self,
        session: _OpenAISession,
        utterance: _UtteranceState,
        outcome: UtteranceOutcome,
    ) -> None:
        if utterance.final_task is not None:
            utterance.final_task.cancel()
            utterance.final_task = None
        if utterance.final_sent:
            return
        utterance.final_sent = True
        if session.current_utterance_id == utterance.utterance_id:
            session.current_stream_id = None
            session.current_utterance_id = None
        await self._publish_locked(
            session,
            (
                ProviderUtteranceFinal(
                    session_id=session.request.session_id,
                    direction_id=session.request.direction_id,
                    stream_id=utterance.stream_id,
                    utterance_id=utterance.utterance_id,
                    event_sequence=self._next_event_sequence(
                        session.request.session_id
                    ),
                    final_audio_sequence=utterance.last_audio_sequence,
                    outcome=outcome,
                ),
            ),
        )

    async def _publish_closed_locked(
        self,
        session: _OpenAISession,
        reason: SessionCloseReason,
    ) -> None:
        if session.closed_published:
            session.closed_event.set()
            return
        session.closed_published = True
        for utterance in session.utterances.values():
            if utterance.final_task is not None:
                utterance.final_task.cancel()
                utterance.final_task = None
        await self._publish_locked(
            session,
            (
                ProviderSessionClosed(
                    session_id=session.request.session_id,
                    direction_id=session.request.direction_id,
                    event_sequence=self._next_event_sequence(
                        session.request.session_id
                    ),
                    reason=reason,
                ),
            ),
        )
        session.closed_event.set()

    async def _publish_provider_error_locked(self, session: _OpenAISession) -> None:
        error = make_provider_error(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            event_sequence=self._next_event_sequence(session.request.session_id),
            code=SafeErrorCode.PROVIDER_UNAVAILABLE,
            retryable=True,
        )
        await self._publish_locked(session, (error,))

    async def _publish_receive_failure(self, session: _OpenAISession) -> None:
        async with session.lock:
            if session.closed:
                return
            session.closed = True
            await self._publish_provider_error_locked(session)
            await self._publish_closed_locked(
                session,
                SessionCloseReason.PROVIDER_FAILURE,
            )

    async def _publish_locked(
        self,
        session: _OpenAISession,
        batch: tuple[object, ...],
    ) -> None:
        committed = False

        def commit() -> None:
            nonlocal committed
            committed = True

        await session.publish(batch, commit)
        if not committed:
            raise RuntimeError("provider_publication_not_committed")

    def _schedule_final_locked(
        self,
        session: _OpenAISession,
        utterance: _UtteranceState,
        delay_ms: int,
        outcome: UtteranceOutcome,
    ) -> None:
        if utterance.final_sent:
            return
        if utterance.final_task is not None:
            utterance.final_task.cancel()
        utterance.final_task = asyncio.create_task(
            self._delayed_final(
                session.request.session_id,
                utterance.utterance_id,
                delay_ms,
                outcome,
            )
        )

    async def _delayed_final(
        self,
        session_id: UUID,
        utterance_id: UUID,
        delay_ms: int,
        outcome: UtteranceOutcome,
    ) -> None:
        await asyncio.sleep(delay_ms / 1000)
        session = self._sessions.get(session_id)
        if session is None:
            return
        async with session.lock:
            utterance = session.utterances.get(utterance_id)
            if utterance is None or utterance.final_sent or not utterance.input_complete:
                return
            await self._publish_pending_audio_locked(session, utterance)
            await self._publish_final_locked(session, utterance, outcome)

    def _convert_output_audio(
        self,
        session: _OpenAISession,
        audio: ProviderAudioDelta,
    ) -> ProviderAudioDelta:
        target = session.request.requested_output_format
        pcm = self._resample_pcm(
            audio.pcm,
            source_rate=audio.sample_rate_hz,
            target_rate=target.sample_rate_hz,
        )
        return audio.model_copy(
            update={
                "sample_rate_hz": target.sample_rate_hz,
                "channels": target.channels,
                "sample_format": target.sample_format,
                "frame_duration_ms": target.frame_duration_ms,
                "pcm": pcm,
            }
        )

    def _with_next_event_sequence(
        self,
        session: _OpenAISession,
        event: ProviderAudioDelta | ProviderTranscriptDelta | ProviderTranslationDelta,
    ):
        return event.model_copy(
            update={
                "event_sequence": self._next_event_sequence(
                    session.request.session_id
                )
            }
        )

    @staticmethod
    def _active_output_utterance(
        session: _OpenAISession,
    ) -> _UtteranceState | None:
        current = session.current_utterance_id
        if current is None:
            return None
        utterance = session.utterances.get(current)
        if utterance is None or utterance.final_sent or utterance.drop_pending:
            return None
        return utterance

    @staticmethod
    def _resample_pcm(pcm: bytes, *, source_rate: int, target_rate: int) -> bytes:
        if source_rate == target_rate:
            return pcm
        samples = np.frombuffer(pcm, dtype="<i2")
        output = soxr.resample(
            samples,
            source_rate,
            target_rate,
            quality="QQ",
        )
        return output.astype("<i2", copy=False).tobytes()

    def _health(self, session: _OpenAISession) -> ProviderHealth:
        return ProviderHealth(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            event_sequence=self._next_event_sequence(session.request.session_id),
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

    def _active_session(self, session_id: UUID) -> _OpenAISession:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            raise OpenAIProviderProtocolError("open_session_required")
        return session

    @staticmethod
    def _validate_format(value: PcmFormat) -> None:
        if value.channels != 1 or value.sample_format is not SampleFormat.S16LE:
            raise OpenAIProviderProtocolError("unsupported_pcm_format")
        if value.frame_duration_ms != 20:
            raise OpenAIProviderProtocolError("unsupported_frame_duration")
        if value.sample_rate_hz not in {16_000, 24_000, 48_000}:
            raise OpenAIProviderProtocolError("unsupported_sample_rate")

    def _validate_frame(self, session: _OpenAISession, frame) -> None:
        if frame.session_id != session.request.session_id:
            raise OpenAIProviderProtocolError("session_identity_mismatch")
        if frame.direction_id != session.request.direction_id:
            raise OpenAIProviderProtocolError("direction_identity_mismatch")
        if frame.channels != 1 or frame.sample_format != SampleFormat.S16LE:
            raise OpenAIProviderProtocolError("unsupported_pcm_format")
        expected_bytes = (
            frame.sample_rate_hz * frame.channels * 2 * frame.frame_duration_ms // 1000
        )
        if len(frame.pcm) != expected_bytes:
            raise OpenAIProviderProtocolError("pcm_length_mismatch")

    def _next_event_sequence(self, session_id: UUID) -> int:
        next_value = self._event_sequences.get(session_id, 0) + 1
        self._event_sequences[session_id] = next_value
        return next_value

    @staticmethod
    def _default_websocket() -> Any:
        import websocket

        return websocket.WebSocket()


def _close_reason(reason: CloseRequestReason) -> SessionCloseReason:
    return {
        CloseRequestReason.USER_STOP: SessionCloseReason.USER_STOP,
        CloseRequestReason.ROUTE_REMOVED: SessionCloseReason.ROUTE_REMOVED,
        CloseRequestReason.DEVICE_UNAVAILABLE: SessionCloseReason.DEVICE_UNAVAILABLE,
        CloseRequestReason.PROVIDER_SWITCH: SessionCloseReason.PROVIDER_SWITCH,
        CloseRequestReason.DAEMON_SHUTDOWN: SessionCloseReason.DAEMON_SHUTDOWN,
    }[reason]
