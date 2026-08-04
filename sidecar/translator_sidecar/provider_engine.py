"""Deterministic provider lifecycle core used by the sidecar transport."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from uuid import UUID

from .provider_contract import (
    CancelUtterance,
    CloseProviderSession,
    ComputeDevice,
    ModelHealth,
    ModelKind,
    ModelState,
    OpenProviderSession,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderCapabilities,
    ProviderHealth,
    ProviderId,
    ProviderInputFrame,
    ProviderLatency,
    ProviderQueues,
    ProviderSessionClosed,
    ProviderSessionOpened,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SessionCloseReason,
    TranslationMode,
    UpdateDebugText,
    UtteranceOutcome,
    make_provider_error,
)

INPUT_LIMIT_MS = 800
OUTPUT_LIMIT_MS = 1200
MAX_TERMINAL_UTTERANCES = 4096
MAX_ACTIVE_UTTERANCES = 64
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_MAX_AGE_MS = {
    TranslationMode.QUALITY_FIRST: 3000,
    TranslationMode.BALANCED: 2000,
    TranslationMode.STREAMING_FIRST: 1000,
}


class ProviderProtocolError(RuntimeError):
    """A protocol violation that requires the owning stream to terminate."""

    terminate_stream = True


@dataclass(frozen=True, slots=True)
class MockInjection:
    process_delay_ms: int = 0
    fail_after_frames: int | None = None
    transcript: str = "mock transcript"
    translation: str = "mock translation"

    def __post_init__(self) -> None:
        if self.process_delay_ms < 0:
            raise ValueError("process_delay_ms must be non-negative")
        if self.fail_after_frames is not None and self.fail_after_frames < 0:
            raise ValueError("fail_after_frames must be non-negative")


@dataclass(frozen=True, slots=True)
class _QueuedInput:
    frame: ProviderInputFrame
    enqueued_ns: int


@dataclass(frozen=True, slots=True)
class _QueuedOutput:
    frame: ProviderInputFrame
    pcm: bytes


@dataclass(slots=True)
class _Session:
    request: OpenProviderSession
    debug_text_enabled: bool
    event_sequence: int = 0
    last_frame_sequence: int | None = None
    stream_id: UUID | None = None
    input_queue: deque[_QueuedInput] = field(default_factory=deque)
    output_queue: deque[_QueuedOutput] = field(default_factory=deque)
    input_buffered_ms: int = 0
    output_buffered_ms: int = 0
    processed_frames: int = 0
    terminal_utterances: set[UUID] = field(default_factory=set)
    utterance_end_received: set[UUID] = field(default_factory=set)
    utterance_streams: dict[UUID, UUID] = field(default_factory=dict)
    next_audio_sequence: dict[UUID, int] = field(default_factory=dict)
    last_emitted_audio_sequence: dict[UUID, int] = field(default_factory=dict)
    closed: bool = False

    def next_event_sequence(self) -> int:
        self.event_sequence += 1
        return self.event_sequence


ProviderEngineEvent = (
    PrivacySafeProviderError
    | ProviderAudioDelta
    | ProviderHealth
    | ProviderLatency
    | ProviderSessionClosed
    | ProviderTranscriptDelta
    | ProviderTranslationDelta
    | ProviderUtteranceFinal
)


def _engine_events(
    *events: ProviderEngineEvent,
) -> tuple[ProviderEngineEvent, ...]:
    return events


def mock_transform_pcm(pcm: bytes) -> bytes:
    """Reverse complete s16le samples while preserving the byte format."""
    if len(pcm) % 2:
        raise ValueError("s16le PCM must contain complete samples")
    return b"".join(pcm[offset : offset + 2] for offset in range(len(pcm) - 2, -1, -2))


class ProviderEngine:
    def __init__(self, *, injection: MockInjection | None = None) -> None:
        self._injection = injection or MockInjection()
        self._sessions: dict[UUID, _Session] = {}

    def open_session(self, request: OpenProviderSession) -> ProviderSessionOpened:
        if request.session_id in self._sessions:
            raise ProviderProtocolError("duplicate_session")
        if request.provider_id is not ProviderId.LOCAL:
            raise ProviderProtocolError("unsupported_provider")
        if request.requested_input_format != request.requested_output_format:
            raise ProviderProtocolError("negotiated_format_mismatch")
        session = _Session(
            request=request,
            debug_text_enabled=request.debug_text_enabled,
        )
        self._sessions[request.session_id] = session
        return ProviderSessionOpened(
            session_id=request.session_id,
            direction_id=request.direction_id,
            event_sequence=session.next_event_sequence(),
            negotiated_input_format=request.requested_input_format,
            negotiated_output_format=request.requested_output_format,
            capabilities=ProviderCapabilities(
                transcript_delta=True,
                translation_delta=True,
                cancellation=True,
                cloud_egress=False,
            ),
        )

    def enqueue_frame(
        self, frame: ProviderInputFrame, *, now_ns: int
    ) -> (
        tuple[PrivacySafeProviderError, ProviderUtteranceFinal]
        | ProviderUtteranceFinal
        | None
    ):
        session = self._active_session(frame.session_id)
        self._validate_frame(session, frame)
        if (
            frame.utterance_id not in session.utterance_streams
            and len(session.utterance_streams) >= MAX_ACTIVE_UTTERANCES
        ):
            raise ProviderProtocolError("active_utterance_capacity")
        session.last_frame_sequence = frame.sequence
        session.stream_id = frame.stream_id
        session.utterance_streams[frame.utterance_id] = frame.stream_id

        if self._expired(session, frame.capture_monotonic_ns, now_ns):
            return self._terminal(session, frame, UtteranceOutcome.DROPPED)
        if session.input_buffered_ms + frame.frame_duration_ms > INPUT_LIMIT_MS:
            return (
                self._error(
                    session,
                    SafeErrorCode.QUEUE_OVERFLOW,
                    retryable=True,
                    stream_id=frame.stream_id,
                    utterance_id=frame.utterance_id,
                ),
                self._terminal(session, frame, UtteranceOutcome.DROPPED),
            )

        session.input_queue.append(_QueuedInput(frame=frame, enqueued_ns=now_ns))
        session.input_buffered_ms += frame.frame_duration_ms
        if frame.end_of_utterance:
            session.utterance_end_received.add(frame.utterance_id)
        return None

    def process_next(
        self, session_id: UUID, *, now_ns: int
    ) -> tuple[ProviderEngineEvent, ...]:
        session = self._active_session(session_id)
        if not session.input_queue:
            return _engine_events()

        queued = session.input_queue[0]
        frame = queued.frame
        if self._expired(session, frame.capture_monotonic_ns, now_ns):
            self._pop_input(session)
            return _engine_events(
                self._terminal(session, frame, UtteranceOutcome.DROPPED)
            )
        ready_ns = (
            queued.enqueued_ns
            + self._injection.process_delay_ms * _NANOSECONDS_PER_MILLISECOND
        )
        if now_ns < ready_ns:
            return _engine_events()

        self._pop_input(session)
        if (
            self._injection.fail_after_frames is not None
            and session.processed_frames >= self._injection.fail_after_frames
        ):
            session.processed_frames += 1
            return _engine_events(
                self._error(
                    session,
                    SafeErrorCode.PROVIDER_UNAVAILABLE,
                    retryable=True,
                    stream_id=frame.stream_id,
                    utterance_id=frame.utterance_id,
                ),
                self._terminal(session, frame, UtteranceOutcome.DROPPED),
            )
        session.processed_frames += 1

        if session.output_buffered_ms + frame.frame_duration_ms > OUTPUT_LIMIT_MS:
            return _engine_events(
                self._error(
                    session,
                    SafeErrorCode.QUEUE_OVERFLOW,
                    retryable=True,
                    stream_id=frame.stream_id,
                    utterance_id=frame.utterance_id,
                ),
                self._terminal(session, frame, UtteranceOutcome.DROPPED),
            )

        session.output_queue.append(
            _QueuedOutput(frame=frame, pcm=mock_transform_pcm(frame.pcm))
        )
        session.output_buffered_ms += frame.frame_duration_ms
        if not session.debug_text_enabled:
            return _engine_events()
        return _engine_events(
            ProviderTranscriptDelta(
                session_id=frame.session_id,
                direction_id=frame.direction_id,
                stream_id=frame.stream_id,
                utterance_id=frame.utterance_id,
                event_sequence=session.next_event_sequence(),
                text=self._injection.transcript,
                is_final=frame.end_of_utterance,
            ),
            ProviderTranslationDelta(
                session_id=frame.session_id,
                direction_id=frame.direction_id,
                stream_id=frame.stream_id,
                utterance_id=frame.utterance_id,
                event_sequence=session.next_event_sequence(),
                text=self._injection.translation,
                stable_prefix=True,
                is_final=frame.end_of_utterance,
            ),
        )

    def drain_output(
        self, session_id: UUID, *, now_ns: int
    ) -> tuple[ProviderEngineEvent, ...]:
        session = self._active_session(session_id)
        events: list[ProviderEngineEvent] = []
        while session.output_queue:
            queued = session.output_queue.popleft()
            frame = queued.frame
            session.output_buffered_ms -= frame.frame_duration_ms
            if self._expired(session, frame.capture_monotonic_ns, now_ns):
                events.append(self._terminal(session, frame, UtteranceOutcome.DROPPED))
                continue
            audio_sequence = session.next_audio_sequence.get(frame.utterance_id, 0)
            events.append(
                ProviderAudioDelta(
                    session_id=frame.session_id,
                    direction_id=frame.direction_id,
                    stream_id=frame.stream_id,
                    utterance_id=frame.utterance_id,
                    sequence=audio_sequence,
                    event_sequence=session.next_event_sequence(),
                    provider_monotonic_ns=now_ns,
                    sample_rate_hz=frame.sample_rate_hz,
                    channels=frame.channels,
                    sample_format=frame.sample_format,
                    frame_duration_ms=frame.frame_duration_ms,
                    pcm=queued.pcm,
                )
            )
            session.next_audio_sequence[frame.utterance_id] = audio_sequence + 1
            session.last_emitted_audio_sequence[frame.utterance_id] = audio_sequence
            total_ms = max(
                0,
                (now_ns - frame.capture_monotonic_ns) // _NANOSECONDS_PER_MILLISECOND,
            )
            events.append(
                ProviderLatency(
                    session_id=frame.session_id,
                    direction_id=frame.direction_id,
                    stream_id=frame.stream_id,
                    utterance_id=frame.utterance_id,
                    event_sequence=session.next_event_sequence(),
                    tts_first_audio_ms=total_ms,
                    provider_total_ms=total_ms,
                )
            )
            if frame.end_of_utterance:
                events.append(
                    self._terminal(session, frame, UtteranceOutcome.COMPLETED)
                )
        return tuple(events)

    def cancel_utterance(self, request: CancelUtterance) -> ProviderUtteranceFinal:
        session = self._active_session(request.session_id)
        if request.direction_id is not session.request.direction_id:
            raise ProviderProtocolError("direction_identity_mismatch")
        if request.utterance_id in session.terminal_utterances:
            raise ProviderProtocolError("utterance_terminal")

        stream_id = session.utterance_streams.get(request.utterance_id)
        if stream_id is None:
            raise ProviderProtocolError("unknown_utterance")
        self._remove_utterance_input(session, request.utterance_id)
        self._remove_utterance_output(session, request.utterance_id)
        return self._terminal_for_identity(
            session,
            stream_id=stream_id,
            utterance_id=request.utterance_id,
            final_audio_sequence=session.last_emitted_audio_sequence.get(
                request.utterance_id
            ),
            outcome=UtteranceOutcome.CANCELLED,
        )

    def close_session(self, request: CloseProviderSession) -> ProviderSessionClosed:
        session = self._active_session(request.session_id)
        session.input_queue.clear()
        session.output_queue.clear()
        session.input_buffered_ms = 0
        session.output_buffered_ms = 0
        session.utterance_end_received.clear()
        session.utterance_streams.clear()
        session.next_audio_sequence.clear()
        session.last_emitted_audio_sequence.clear()
        session.closed = True
        return ProviderSessionClosed(
            session_id=request.session_id,
            direction_id=session.request.direction_id,
            event_sequence=session.next_event_sequence(),
            reason=SessionCloseReason(request.reason.value),
        )

    def update_debug_text(self, request: UpdateDebugText) -> None:
        session = self._active_session(request.session_id)
        session.debug_text_enabled = request.enabled

    def release_session(self, session_id: UUID) -> None:
        if session_id not in self._sessions:
            raise ProviderProtocolError("unknown_session")
        del self._sessions[session_id]

    def queue_state(self, session_id: UUID) -> ProviderQueues:
        session = self._active_session(session_id)
        return self._queues(session, now_ns=None)

    def next_wakeup_ms(self, session_id: UUID, *, now_ns: int) -> int | None:
        session = self._active_session(session_id)
        if not session.input_queue:
            return None
        queued = session.input_queue[0]
        ready_ns = (
            queued.enqueued_ns
            + self._injection.process_delay_ms * _NANOSECONDS_PER_MILLISECOND
        )
        expiry_ns = (
            queued.frame.capture_monotonic_ns
            + _MAX_AGE_MS[session.request.mode] * _NANOSECONDS_PER_MILLISECOND
            + 1
        )
        wakeup_ns = min(ready_ns, expiry_ns)
        remaining_ns = max(0, wakeup_ns - now_ns)
        return (
            remaining_ns + _NANOSECONDS_PER_MILLISECOND - 1
        ) // _NANOSECONDS_PER_MILLISECOND

    def health(self, session_id: UUID, *, now_ns: int) -> ProviderHealth:
        session = self._active_session(session_id)
        queues = self._queues(session, now_ns=now_ns)
        state = (
            ProviderState.BACKPRESSURE
            if (
                queues.provider_input_buffered_ms >= INPUT_LIMIT_MS
                or queues.provider_output_buffered_ms >= OUTPUT_LIMIT_MS
            )
            else ProviderState.READY
        )
        return ProviderHealth(
            session_id=session_id,
            direction_id=session.request.direction_id,
            event_sequence=session.next_event_sequence(),
            provider_id=ProviderId.LOCAL,
            provider_name="deterministic-mock",
            state=state,
            models=(
                ModelHealth(
                    kind=ModelKind.SPEECH_TO_SPEECH,
                    id="deterministic-mock",
                    state=ModelState.READY,
                    device=ComputeDevice.CPU,
                ),
            ),
            queues=queues,
        )

    def _active_session(self, session_id: UUID) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise ProviderProtocolError("unknown_session")
        if session.closed:
            raise ProviderProtocolError("session_terminal")
        return session

    def _validate_frame(self, session: _Session, frame: ProviderInputFrame) -> None:
        request = session.request
        if frame.utterance_id in session.terminal_utterances:
            raise ProviderProtocolError("utterance_terminal")
        if frame.direction_id is not request.direction_id:
            raise ProviderProtocolError("direction_identity_mismatch")
        if session.stream_id is not None and frame.stream_id != session.stream_id:
            raise ProviderProtocolError("stream_identity_mismatch")
        if session.last_frame_sequence is not None:
            if frame.sequence == session.last_frame_sequence:
                raise ProviderProtocolError("duplicate_frame_sequence")
            if frame.sequence < session.last_frame_sequence:
                raise ProviderProtocolError("stale_frame_sequence")
        if frame.utterance_id in session.utterance_end_received:
            raise ProviderProtocolError("utterance_terminal")
        pcm_format = request.requested_input_format
        if (
            frame.sample_rate_hz != pcm_format.sample_rate_hz
            or frame.channels != pcm_format.channels
            or frame.sample_format != pcm_format.sample_format
            or frame.frame_duration_ms != pcm_format.frame_duration_ms
        ):
            raise ProviderProtocolError("negotiated_format_mismatch")
        if (
            frame.source_language is not request.source_language
            or frame.target_language is not request.target_language
            or frame.mode is not request.mode
        ):
            raise ProviderProtocolError("session_contract_mismatch")
        expected_pcm_bytes = (
            frame.sample_rate_hz * frame.channels * 2 * frame.frame_duration_ms // 1000
        )
        if len(frame.pcm) != expected_pcm_bytes:
            raise ProviderProtocolError("pcm_length_mismatch")

    def _expired(
        self, session: _Session, capture_monotonic_ns: int, now_ns: int
    ) -> bool:
        age_ns = max(0, now_ns - capture_monotonic_ns)
        return age_ns > _MAX_AGE_MS[session.request.mode] * _NANOSECONDS_PER_MILLISECOND

    def _pop_input(self, session: _Session) -> _QueuedInput:
        queued = session.input_queue.popleft()
        session.input_buffered_ms -= queued.frame.frame_duration_ms
        return queued

    def _remove_utterance_input(self, session: _Session, utterance_id: UUID) -> None:
        retained: deque[_QueuedInput] = deque()
        while session.input_queue:
            queued = session.input_queue.popleft()
            if queued.frame.utterance_id == utterance_id:
                session.input_buffered_ms -= queued.frame.frame_duration_ms
            else:
                retained.append(queued)
        session.input_queue = retained

    def _remove_utterance_output(self, session: _Session, utterance_id: UUID) -> None:
        retained: deque[_QueuedOutput] = deque()
        while session.output_queue:
            queued = session.output_queue.popleft()
            if queued.frame.utterance_id == utterance_id:
                session.output_buffered_ms -= queued.frame.frame_duration_ms
            else:
                retained.append(queued)
        session.output_queue = retained

    def _terminal(
        self,
        session: _Session,
        frame: ProviderInputFrame,
        outcome: UtteranceOutcome,
    ) -> ProviderUtteranceFinal:
        return self._terminal_for_identity(
            session,
            stream_id=frame.stream_id,
            utterance_id=frame.utterance_id,
            final_audio_sequence=session.last_emitted_audio_sequence.get(
                frame.utterance_id
            ),
            outcome=outcome,
        )

    def _terminal_for_identity(
        self,
        session: _Session,
        *,
        stream_id: UUID,
        utterance_id: UUID,
        final_audio_sequence: int | None,
        outcome: UtteranceOutcome,
    ) -> ProviderUtteranceFinal:
        if utterance_id in session.terminal_utterances:
            raise ProviderProtocolError("utterance_terminal")
        if len(session.terminal_utterances) >= MAX_TERMINAL_UTTERANCES:
            raise ProviderProtocolError("terminal_tombstone_capacity")
        self._remove_utterance_input(session, utterance_id)
        self._remove_utterance_output(session, utterance_id)
        session.utterance_end_received.discard(utterance_id)
        session.utterance_streams.pop(utterance_id, None)
        session.next_audio_sequence.pop(utterance_id, None)
        session.last_emitted_audio_sequence.pop(utterance_id, None)
        session.terminal_utterances.add(utterance_id)
        return ProviderUtteranceFinal(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            stream_id=stream_id,
            utterance_id=utterance_id,
            event_sequence=session.next_event_sequence(),
            final_audio_sequence=final_audio_sequence,
            outcome=outcome,
        )

    def _error(
        self,
        session: _Session,
        code: SafeErrorCode,
        *,
        retryable: bool,
        stream_id: UUID | None,
        utterance_id: UUID | None,
    ) -> PrivacySafeProviderError:
        error = make_provider_error(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            event_sequence=session.next_event_sequence(),
            code=code,
            retryable=retryable,
            utterance_id=utterance_id,
        )
        return error.model_copy(update={"stream_id": stream_id})

    def _queues(self, session: _Session, *, now_ns: int | None) -> ProviderQueues:
        capture_times = [
            queued.frame.capture_monotonic_ns for queued in session.input_queue
        ]
        capture_times.extend(
            queued.frame.capture_monotonic_ns for queued in session.output_queue
        )
        queue_lag_ms = 0
        if now_ns is not None and capture_times:
            queue_lag_ms = max(
                0,
                (now_ns - min(capture_times)) // _NANOSECONDS_PER_MILLISECOND,
            )
        return ProviderQueues(
            provider_input_buffered_ms=session.input_buffered_ms,
            provider_output_buffered_ms=session.output_buffered_ms,
            queue_lag_ms=queue_lag_ms,
        )
