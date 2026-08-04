"""Real local ASR, translation and TTS provider lifecycle."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import math
import os
import struct
from typing import Any
from uuid import UUID

from translator_sidecar.provider_contract import (
    SAFE_ERROR_MESSAGES,
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
    SafeErrorSummary,
    SampleFormat,
    SessionCloseReason,
    TranslationMode,
    UpdateDebugText,
    UtteranceOutcome,
)

from .inference_scheduler import (
    InferenceScheduler,
    JobIdentity,
    SchedulerContext,
    SchedulerOverflow,
    SchedulerStale,
    SchedulerUnavailable,
)
from .source_commit import SourceCommit
from .tts import TtsOutputLimit


_BASE_SOURCE_UTTERANCE_MS: int = 12_000
_MAX_SOURCE_UTTERANCE_MS: int = 30_000
_BASE_TRANSLATION_CHARS = 128
_BASE_TRANSLATION_TOKENS = 96
_BASE_OUTPUT_MS = 12_000
_MAX_OUTPUT_MS = 30_000
_MAX_TERMINAL_IDS = 4096
_MAX_PENDING_EVENTS = 64
_TERMINAL_RESERVED_EVENTS = 3
_MAX_RETIRED_SESSIONS = 64
_PROVIDER_STALE_MESSAGE = "provider result is stale"
_DEFAULT_CONTINUATION_TAIL_RMS = 300.0
_CONTINUATION_TAIL_FRAMES = 3
_MIN_CONTINUATION_VOICED_TAIL_FRAMES = 2


def _source_utterance_limit_ms(mode: TranslationMode) -> int:
    if mode is TranslationMode.STREAMING_FIRST:
        return _MAX_SOURCE_UTTERANCE_MS
    if mode is TranslationMode.BALANCED:
        return _MAX_SOURCE_UTTERANCE_MS
    return _MAX_SOURCE_UTTERANCE_MS


def _translation_exceeds_budget(
    source_text: str,
    translation: str,
    token_count: int,
) -> bool:
    source_word_count = max(1, len(source_text.split()))
    char_limit = max(
        _BASE_TRANSLATION_CHARS,
        min(2_048, len(source_text.strip()) * 3),
    )
    token_limit = max(
        _BASE_TRANSLATION_TOKENS,
        min(512, source_word_count * 3),
    )
    return len(translation) > char_limit or token_count > token_limit


def _output_limit_ms(source_ms: int, translation: str) -> int:
    word_count = max(1, len(translation.split()))
    speech_estimate_ms = word_count * 450
    duration_estimate_ms = max(_BASE_OUTPUT_MS, source_ms * 2)
    return min(
        _MAX_OUTPUT_MS,
        max(_BASE_OUTPUT_MS, duration_estimate_ms, speech_estimate_ms),
    )


def _configured_continuation_tail_rms() -> float:
    value = os.environ.get("TRANSLATOR_CONTINUATION_TAIL_RMS")
    if value is None:
        return _DEFAULT_CONTINUATION_TAIL_RMS
    try:
        threshold = float(value)
    except ValueError:
        return _DEFAULT_CONTINUATION_TAIL_RMS
    if not math.isfinite(threshold) or threshold < 0:
        return _DEFAULT_CONTINUATION_TAIL_RMS
    return threshold


def _pcm_rms(pcm: bytes) -> float:
    sample_count = len(pcm) // 2
    if sample_count == 0:
        return 0.0
    total = 0
    for sample, in struct.iter_unpack("<h", pcm[: sample_count * 2]):
        total += sample * sample
    return math.sqrt(total / sample_count)


def _looks_like_continuation_boundary(
    utterance: _Utterance,
    frame: ProviderInputFrame,
) -> bool:
    if frame.sample_format is not SampleFormat.S16LE:
        return False
    threshold = _configured_continuation_tail_rms()
    recent_frames = utterance.pcm_parts[-_CONTINUATION_TAIL_FRAMES:]
    if not recent_frames:
        return False
    voiced_tail = [_pcm_rms(pcm) >= threshold for pcm in recent_frames]
    if not voiced_tail[-1]:
        return False
    required = min(_MIN_CONTINUATION_VOICED_TAIL_FRAMES, len(voiced_tail))
    return sum(voiced_tail) >= required


class LocalProviderProtocolError(RuntimeError):
    """A local provider wire contract was violated."""

    terminate_stream = True


class LocalProviderPublicationError(RuntimeError):
    """Provider events could not be delivered to the owning stream."""

    terminate_stream = True


class _CollectionState(str, Enum):
    COLLECTING = "collecting"
    OVERFLOW_DISCARDING = "overflow_discarding"


class _PublicationKind(str, Enum):
    DEBUG = "debug"
    AUDIO = "audio"
    TERMINAL = "terminal"
    HEALTH = "health"
    SESSION = "session"


LocalProviderEvent = (
    PrivacySafeProviderError
    | ProviderAudioDelta
    | ProviderHealth
    | ProviderLatency
    | ProviderSessionClosed
    | ProviderTranscriptDelta
    | ProviderTranslationDelta
    | ProviderUtteranceFinal
)
CommitPublication = Callable[[], None]
PublishEvents = Callable[
    [tuple[LocalProviderEvent, ...], CommitPublication],
    Awaitable[None],
]


@dataclass(frozen=True, slots=True)
class _Publication:
    kind: _PublicationKind
    build: Callable[[], tuple[LocalProviderEvent, ...]]
    event_count: int
    utterance_id: UUID | None = None
    audio_ms: int = 0
    on_success: Callable[[], None] | None = None


@dataclass(slots=True)
class _Utterance:
    utterance_id: UUID
    stream_id: UUID
    capture_onset_ns: int
    commit: SourceCommit
    state: _CollectionState = _CollectionState.COLLECTING
    pcm_parts: list[bytes] = field(default_factory=list)
    buffered_ms: int = 0
    identity: JobIdentity | None = None
    last_audio_sequence: int | None = None
    continues_speech: bool = False

    def purge_source(self) -> None:
        self.pcm_parts.clear()
        self.buffered_ms = 0


@dataclass(slots=True)
class _Session:
    request: OpenProviderSession
    publish: PublishEvents
    debug_text_enabled: bool
    event_sequence: int = 0
    last_input_sequence: int | None = None
    stream_id: UUID | None = None
    collecting_id: UUID | None = None
    utterances: dict[UUID, _Utterance] = field(default_factory=dict)
    terminal_ids: set[UUID] = field(default_factory=set)
    pending_publications: deque[_Publication] = field(default_factory=deque)
    pending_audio_ms: int = 0
    pending_event_count: int = 0
    in_flight: _Publication | None = None
    in_flight_batch_size: int = 0
    in_flight_task: asyncio.Task[None] | None = None
    in_flight_cancelled: bool = False
    in_flight_committed: bool = False
    publication_error: LocalProviderPublicationError | None = None
    publication_error_consumed: bool = False
    model_keys: dict[ModelKind, tuple[ModelKind, str]] = field(default_factory=dict)
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    publication_signal: asyncio.Event = field(default_factory=asyncio.Event)
    publication_space: asyncio.Event = field(default_factory=asyncio.Event)
    publication_idle: asyncio.Event = field(default_factory=asyncio.Event)
    publication_task: asyncio.Task[None] | None = None

    def next_event_sequence(self) -> int:
        self.event_sequence += 1
        return self.event_sequence


class LocalProvider:
    """Own local provider sessions and immutable utterance publication."""

    def __init__(
        self,
        *,
        asr: Any,
        translator: Any,
        tts: Any,
        scheduler: InferenceScheduler,
        now_ns: Callable[[], int],
        asr_model_id: str,
        mt_model_id: str,
        tts_model_id: str,
        mt_device: ComputeDevice,
        model_admission_probe: (Callable[[ModelKind], None] | None) = None,
    ) -> None:
        self._asr = asr
        self._translator = translator
        self._tts = tts
        self._scheduler = scheduler
        self._now_ns = now_ns
        self._asr_model_id = asr_model_id
        self._mt_model_id = mt_model_id
        self._tts_model_id = tts_model_id
        self._mt_device = mt_device
        self._model_admission_probe = model_admission_probe
        self._sessions: dict[UUID, _Session] = {}
        self._retired_sessions: OrderedDict[UUID, _Session] = OrderedDict()
        self._model_states: dict[
            tuple[ModelKind, str],
            ModelState,
        ] = {}
        self._model_lock = asyncio.Lock()
        self._model_gates: dict[
            tuple[ModelKind, str],
            asyncio.Lock,
        ] = {}
        self._futures: set[asyncio.Future[None]] = set()
        self._closed = False

    async def open_session(
        self,
        request: OpenProviderSession,
        publish: PublishEvents,
    ) -> tuple[ProviderSessionOpened, ProviderHealth]:
        if self._closed:
            raise LocalProviderProtocolError("provider is closed")
        if request.provider_id is not ProviderId.LOCAL:
            raise LocalProviderProtocolError("unsupported provider")
        if request.session_id in self._sessions:
            raise LocalProviderProtocolError("duplicate session")
        self._retired_sessions.pop(request.session_id, None)
        self._validate_open(request)
        self._scheduler.open_session(
            request.session_id,
            request.direction_id,
        )
        session = _Session(
            request=request,
            publish=publish,
            debug_text_enabled=request.debug_text_enabled,
        )
        initial_states = self._initial_model_states(request)
        session.model_keys = self._model_keys(request)
        for kind, key in session.model_keys.items():
            self._model_states.setdefault(
                key,
                initial_states[kind],
            )
            self._model_gates.setdefault(key, asyncio.Lock())
        session.publication_space.set()
        session.publication_idle.set()
        self._sessions[request.session_id] = session
        session.publication_task = asyncio.create_task(self._publication_loop(session))
        opened = ProviderSessionOpened(
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
        return opened, self._health(session)

    async def health(self, session_id: UUID) -> ProviderHealth:
        session = self._active_session(session_id)
        async with session.lock:
            return self._health(
                session,
                allocate_sequence=False,
            )

    async def submit_frame(self, frame: ProviderInputFrame) -> None:
        session = self._active_session(frame.session_id)
        async with session.lock:
            self._validate_frame(session, frame)
            utterance = self._collecting_utterance(session, frame)
            session.last_input_sequence = frame.sequence
            if utterance.state is _CollectionState.OVERFLOW_DISCARDING:
                if frame.end_of_utterance:
                    session.collecting_id = None
                    await self._drop_locked(
                        session,
                        utterance,
                        code=SafeErrorCode.QUEUE_OVERFLOW,
                    )
                return

            if (
                utterance.buffered_ms + frame.frame_duration_ms
                > _source_utterance_limit_ms(session.request.mode)
            ):
                utterance.purge_source()
                utterance.state = _CollectionState.OVERFLOW_DISCARDING
                if frame.end_of_utterance:
                    session.collecting_id = None
                    await self._drop_locked(
                        session,
                        utterance,
                        code=SafeErrorCode.QUEUE_OVERFLOW,
                    )
                return

            utterance.pcm_parts.append(frame.pcm)
            utterance.buffered_ms += frame.frame_duration_ms
            if not frame.end_of_utterance:
                return
            utterance.continues_speech = _looks_like_continuation_boundary(
                utterance,
                frame,
            )
            session.collecting_id = None
            if not self._models_available(session):
                await self._drop_locked(
                    session,
                    utterance,
                    code=SafeErrorCode.MODEL_NOT_LOADED,
                )
                return
            await self._schedule_locked(session, utterance)

    async def cancel_utterance(
        self,
        request: CancelUtterance,
    ) -> None:
        session = self._active_session(request.session_id)
        async with session.lock:
            if request.direction_id is not session.request.direction_id:
                raise LocalProviderProtocolError("utterance direction mismatch")
            utterance = session.utterances.get(request.utterance_id)
            if utterance is None:
                raise LocalProviderProtocolError("utterance is unavailable")
            if utterance.identity is not None:
                try:
                    self._scheduler.cancel_utterance(
                        request.session_id,
                        request.utterance_id,
                    )
                except SchedulerStale:
                    pass
            self._cancel_commit(utterance)
            if session.collecting_id == request.utterance_id:
                session.collecting_id = None
            self._purge_publications_locked(
                session,
                request.utterance_id,
            )
            self._terminalize_local(session, utterance)
            self._append_publication_locked(
                session,
                _Publication(
                    kind=_PublicationKind.TERMINAL,
                    event_count=1,
                    utterance_id=utterance.utterance_id,
                    build=lambda: (
                        ProviderUtteranceFinal(
                            session_id=request.session_id,
                            direction_id=session.request.direction_id,
                            stream_id=utterance.stream_id,
                            utterance_id=utterance.utterance_id,
                            event_sequence=session.next_event_sequence(),
                            final_audio_sequence=(utterance.last_audio_sequence),
                            outcome=UtteranceOutcome.CANCELLED,
                        ),
                    ),
                ),
            )
        await asyncio.sleep(0)

    async def update_debug_text(
        self,
        request: UpdateDebugText,
    ) -> None:
        session = self._active_session(request.session_id)
        async with session.lock:
            session.debug_text_enabled = request.enabled
            if not request.enabled:
                self._purge_debug_locked(session)

    async def close_session(
        self,
        request: CloseProviderSession,
    ) -> None:
        session = self._active_session(request.session_id)
        async with session.lock:
            session.closed = True
            session.debug_text_enabled = False
            try:
                self._scheduler.close_session(request.session_id)
            except SchedulerStale:
                pass
            for utterance in tuple(session.utterances.values()):
                self._cancel_commit(utterance)
                utterance.purge_source()
            session.utterances.clear()
            session.collecting_id = None
            self._purge_publications_locked(session, None)
            self._append_publication_locked(
                session,
                _Publication(
                    kind=_PublicationKind.SESSION,
                    event_count=1,
                    build=lambda: (
                        ProviderSessionClosed(
                            session_id=request.session_id,
                            direction_id=session.request.direction_id,
                            event_sequence=session.next_event_sequence(),
                            reason=SessionCloseReason(request.reason.value),
                        ),
                    ),
                ),
            )
        await asyncio.sleep(0)

    async def wait_idle(self) -> None:
        while self._futures:
            pending = tuple(self._futures)
            await asyncio.gather(
                *pending,
                return_exceptions=True,
            )
            self._futures.difference_update(
                future for future in pending if future.done()
            )
            await asyncio.sleep(0)
        sessions = tuple(self._sessions.values())
        if sessions:
            await asyncio.gather(
                *(session.publication_idle.wait() for session in sessions)
            )
        error = self._take_publication_error()
        if error is not None:
            raise error from None

    async def wait_publications(self, session_id: UUID) -> None:
        session = self._sessions.get(session_id) or self._retired_sessions.get(
            session_id
        )
        if session is None:
            raise LocalProviderProtocolError("provider session is unavailable")
        await session.publication_idle.wait()
        self._retired_sessions.pop(session_id, None)
        if (
            session.publication_error is not None
            and not session.publication_error_consumed
        ):
            session.publication_error_consumed = True
            raise session.publication_error from None

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        publication_tasks = []
        for session in tuple(self._sessions.values()):
            async with session.lock:
                session.closed = True
                session.debug_text_enabled = False
                self._purge_publications_locked(session, None)
                for utterance in tuple(session.utterances.values()):
                    self._cancel_commit(utterance)
                    utterance.purge_source()
                session.utterances.clear()
                session.collecting_id = None
                session.publication_signal.set()
                if session.publication_task is not None:
                    publication_tasks.append(session.publication_task)
        self._sessions.clear()
        self._retired_sessions.clear()
        await self._scheduler.shutdown()
        for task in publication_tasks:
            task.cancel()
        if publication_tasks:
            await asyncio.gather(
                *publication_tasks,
                return_exceptions=True,
            )
        self._futures.clear()

    async def _schedule_locked(
        self,
        session: _Session,
        utterance: _Utterance,
    ) -> None:
        try:
            identity = self._scheduler.open_utterance(
                session.request.session_id,
                utterance.utterance_id,
            )
            utterance.identity = identity
            future = self._scheduler.submit(
                identity,
                lambda context: self._run_pipeline(
                    session,
                    utterance,
                    context,
                ),
            )
        except SchedulerOverflow:
            if utterance.identity is not None:
                try:
                    self._scheduler.cancel_utterance(
                        session.request.session_id,
                        utterance.utterance_id,
                    )
                except SchedulerStale:
                    pass
            await self._drop_locked(
                session,
                utterance,
                code=SafeErrorCode.QUEUE_OVERFLOW,
            )
            return
        except (SchedulerStale, SchedulerUnavailable):
            await self._drop_locked(
                session,
                utterance,
                code=SafeErrorCode.PROVIDER_UNAVAILABLE,
            )
            return
        self._futures.add(future)
        future.add_done_callback(self._future_done)

    async def _run_pipeline(
        self,
        session: _Session,
        utterance: _Utterance,
        context: SchedulerContext,
    ) -> None:
        asr_ms: int | None = None
        mt_ms: int | None = None
        tts_ms: int | None = None
        active_stage = ModelKind.ASR
        try:
            pcm = b"".join(utterance.pcm_parts)
            source_buffered_ms = utterance.buffered_ms
            utterance.purge_source()
            async with self._model_execution(
                session,
                utterance,
                ModelKind.ASR,
            ):
                if self._model_state(session, ModelKind.ASR) is not ModelState.READY:
                    await self._set_model_state(
                        session,
                        ModelKind.ASR,
                        ModelState.LOADING,
                    )
                source_text = await context.run_gpu(
                    lambda: self._asr.transcribe(
                        pcm,
                        language=session.request.source_language,
                        mode=session.request.mode,
                    )
                )
                await self._set_model_state(
                    session,
                    ModelKind.ASR,
                    ModelState.READY,
                )
            asr_ms = self._elapsed_ms(utterance.capture_onset_ns)
            if not source_text.strip():
                await self._drop(
                    session,
                    utterance,
                    code=SafeErrorCode.NO_SPEECH,
                    asr_ms=asr_ms,
                    mt_ms=None,
                    tts_ms=None,
                )
                return

            active_stage = ModelKind.MT
            async with self._model_execution(
                session,
                utterance,
                ModelKind.MT,
            ):
                translation = await utterance.commit.finalize_async(
                    source_text,
                    end_of_utterance=True,
                    translate=lambda committed_source: context.run_gpu(
                        lambda: self._translate(
                            committed_source,
                            session.request,
                        )
                    ),
                )
                token_count = await context.run_gpu(
                    lambda: self._count_tokens(translation)
                )
                await self._set_model_state(
                    session,
                    ModelKind.MT,
                    ModelState.READY,
                )
            mt_ms = self._elapsed_ms(utterance.capture_onset_ns)
            if _translation_exceeds_budget(
                source_text,
                translation,
                token_count,
            ):
                await self._drop(
                    session,
                    utterance,
                    code=SafeErrorCode.QUEUE_OVERFLOW,
                    asr_ms=asr_ms,
                    mt_ms=mt_ms,
                    tts_ms=None,
                )
                return

            await self._publish_debug(
                session,
                utterance,
                source_text,
                translation,
            )

            output = session.request.requested_output_format
            audio_sequence = 0
            frame_count = 0
            active_stage = ModelKind.TTS
            async for pcm_frame in utterance.commit.stream_once(
                lambda committed: self._stream_tts(
                    session,
                    utterance,
                    committed,
                    context,
                )
            ):
                if (frame_count + 1) * output.frame_duration_ms > _output_limit_ms(
                    source_buffered_ms,
                    translation,
                ):
                    raise SchedulerOverflow("TTS output limit was reached")
                self._validate_output_frame(pcm_frame, session.request)
                if tts_ms is None:
                    tts_ms = self._elapsed_ms(utterance.capture_onset_ns)
                await self._publish_audio(
                    session,
                    utterance,
                    pcm_frame,
                    audio_sequence,
                )
                audio_sequence += 1
                frame_count += 1
            if frame_count == 0:
                raise SchedulerUnavailable("TTS inference is unavailable")
            await self._complete(
                session,
                utterance,
                asr_ms=asr_ms,
                mt_ms=mt_ms,
                tts_ms=tts_ms,
            )
            active_stage = None
        except SchedulerStale:
            raise
        except SchedulerOverflow:
            await self._drop(
                session,
                utterance,
                code=SafeErrorCode.QUEUE_OVERFLOW,
                asr_ms=asr_ms,
                mt_ms=mt_ms,
                tts_ms=tts_ms,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if active_stage is not None:
                await self._mark_model_failed(
                    session,
                    utterance,
                    active_stage,
                )
            await self._drop(
                session,
                utterance,
                code=SafeErrorCode.PROVIDER_UNAVAILABLE,
                asr_ms=asr_ms,
                mt_ms=mt_ms,
                tts_ms=tts_ms,
            )

    def _translate(
        self,
        source_text: str,
        request: OpenProviderSession,
    ) -> str:
        return self._translator.translate(
            source_text,
            source_language=request.source_language,
            target_language=request.target_language,
            mode=request.mode,
        ).strip()

    def _count_tokens(self, translation: str) -> int:
        count_tokens = getattr(
            self._translator,
            "count_tokens",
            None,
        )
        return (
            count_tokens(translation)
            if count_tokens is not None
            else len(translation.split())
        )

    async def _stream_tts(
        self,
        session: _Session,
        utterance: _Utterance,
        text: str,
        context: SchedulerContext,
    ):
        async with self._model_execution(
            session,
            utterance,
            ModelKind.TTS,
        ):
            output = session.request.requested_output_format
            if self._model_state(session, ModelKind.TTS) is not ModelState.READY:
                await self._set_model_state(
                    session,
                    ModelKind.TTS,
                    ModelState.LOADING,
                )
            first = True
            async for frame in context.stream_tts(
                lambda: self._tts_frames(
                    text,
                    session.request,
                    context,
                    continuation=utterance.continues_speech,
                ),
                frame_duration_ms=output.frame_duration_ms,
            ):
                if first:
                    first = False
                    await self._set_model_state(
                        session,
                        ModelKind.TTS,
                        ModelState.READY,
                    )
                yield frame

    def _tts_frames(
        self,
        text: str,
        request: OpenProviderSession,
        context: SchedulerContext,
        *,
        continuation: bool = False,
    ) -> Iterator[bytes]:
        output = request.requested_output_format
        try:
            yield from self._tts.synthesize_frames(
                text,
                target_language=request.target_language,
                voice_profile=request.voice_profile,
                mode=request.mode,
                output_sample_rate_hz=output.sample_rate_hz,
                output_channels=output.channels,
                frame_duration_ms=output.frame_duration_ms,
                cancelled=context.cancelled,
                continuation=continuation,
            )
        except TtsOutputLimit:
            raise SchedulerOverflow("TTS output limit was reached") from None

    async def _set_model_state(
        self,
        session: _Session,
        kind: ModelKind,
        state: ModelState,
    ) -> None:
        async with session.lock:
            if session.closed:
                raise SchedulerStale(_PROVIDER_STALE_MESSAGE)
            key = session.model_keys[kind]
            async with self._model_lock:
                if (
                    self._model_states[key] is ModelState.FAILED
                    and state is not ModelState.FAILED
                ):
                    raise SchedulerUnavailable("shared model is unavailable")
                self._model_states[key] = state

    async def _require_model_available(
        self,
        session: _Session,
        kind: ModelKind,
    ) -> None:
        async with session.lock:
            if session.closed:
                raise SchedulerStale(_PROVIDER_STALE_MESSAGE)
            async with self._model_lock:
                if self._model_states[session.model_keys[kind]] is ModelState.FAILED:
                    raise SchedulerUnavailable("shared model is unavailable")

    @asynccontextmanager
    async def _model_execution(
        self,
        session: _Session,
        utterance: _Utterance,
        kind: ModelKind,
    ):
        if self._model_admission_probe is not None:
            self._model_admission_probe(kind)
        gate = self._model_gates[session.model_keys[kind]]
        async with gate:
            await self._require_model_available(session, kind)
            try:
                yield
            except (
                asyncio.CancelledError,
                SchedulerOverflow,
                SchedulerStale,
            ):
                raise
            except Exception:
                await self._mark_model_failed(
                    session,
                    utterance,
                    kind,
                )
                raise

    async def _mark_model_failed(
        self,
        session: _Session,
        utterance: _Utterance,
        kind: ModelKind,
    ) -> None:
        async with session.lock:
            if self._is_current(session, utterance):
                async with self._model_lock:
                    self._model_states[session.model_keys[kind]] = ModelState.FAILED

    async def _enqueue_publication(
        self,
        session: _Session,
        utterance: _Utterance,
        publication: _Publication,
        *,
        require_debug: bool = False,
    ) -> None:
        while True:
            async with session.lock:
                if not self._is_current(session, utterance):
                    raise SchedulerStale(_PROVIDER_STALE_MESSAGE)
                if require_debug and not session.debug_text_enabled:
                    return
                in_flight_ms = (
                    session.in_flight.audio_ms if session.in_flight is not None else 0
                )
                pending_events = (
                    session.pending_event_count + session.in_flight_batch_size
                )
                nonterminal_limit = max(
                    0,
                    _MAX_PENDING_EVENTS - _TERMINAL_RESERVED_EVENTS,
                )
                event_capacity_exceeded = (
                    pending_events + publication.event_count > nonterminal_limit
                )
                if publication.event_count > nonterminal_limit:
                    self._fail_session_locked(
                        session,
                        LocalProviderPublicationError(
                            "provider event publication capacity reached"
                        ),
                    )
                    return
                if event_capacity_exceeded or (
                    publication.audio_ms > 0
                    and session.pending_audio_ms + in_flight_ms + publication.audio_ms
                    > 1200
                ):
                    session.publication_space.clear()
                    wait_for_space = session.publication_space
                else:
                    self._append_publication_locked(
                        session,
                        publication,
                    )
                    return
            await wait_for_space.wait()

    def _append_publication_locked(
        self,
        session: _Session,
        publication: _Publication,
    ) -> bool:
        pending_events = session.pending_event_count + session.in_flight_batch_size
        if pending_events + publication.event_count > _MAX_PENDING_EVENTS:
            self._fail_session_locked(
                session,
                LocalProviderPublicationError(
                    "provider event publication capacity reached"
                ),
            )
            return False
        session.pending_publications.append(publication)
        session.pending_audio_ms += publication.audio_ms
        session.pending_event_count += publication.event_count
        session.publication_idle.clear()
        session.publication_signal.set()
        return True

    def _purge_publications_locked(
        self,
        session: _Session,
        utterance_id: UUID | None,
    ) -> None:
        retained: deque[_Publication] = deque()
        pending_audio_ms = 0
        pending_event_count = 0
        for publication in session.pending_publications:
            if utterance_id is None or publication.utterance_id == utterance_id:
                continue
            retained.append(publication)
            pending_audio_ms += publication.audio_ms
            pending_event_count += publication.event_count
        session.pending_publications = retained
        session.pending_audio_ms = pending_audio_ms
        session.pending_event_count = pending_event_count

        in_flight = session.in_flight
        in_flight_task = session.in_flight_task
        if (
            in_flight is not None
            and (utterance_id is None or in_flight.utterance_id == utterance_id)
            and in_flight_task is not None
            and not in_flight_task.done()
        ):
            session.in_flight_cancelled = True
            session.in_flight_batch_size = 0
            in_flight_task.cancel()

        session.publication_space.set()
        if session.pending_publications:
            session.publication_signal.set()
        elif session.in_flight is None:
            session.publication_idle.set()

    def _purge_debug_locked(self, session: _Session) -> None:
        retained: deque[_Publication] = deque()
        pending_audio_ms = 0
        pending_event_count = 0
        for publication in session.pending_publications:
            if publication.kind is _PublicationKind.DEBUG:
                continue
            retained.append(publication)
            pending_audio_ms += publication.audio_ms
            pending_event_count += publication.event_count
        session.pending_publications = retained
        session.pending_audio_ms = pending_audio_ms
        session.pending_event_count = pending_event_count
        if (
            session.in_flight is not None
            and session.in_flight.kind is _PublicationKind.DEBUG
            and session.in_flight_task is not None
            and not session.in_flight_task.done()
        ):
            session.in_flight_cancelled = True
            session.in_flight_batch_size = 0
            session.in_flight_task.cancel()
        session.publication_space.set()

    async def _publication_loop(self, session: _Session) -> None:
        try:
            while True:
                await session.publication_signal.wait()
                async with session.lock:
                    if not session.pending_publications:
                        session.publication_signal.clear()
                        session.publication_idle.set()
                        if session.closed:
                            return
                        continue
                    publication = session.pending_publications.popleft()
                    session.pending_audio_ms -= publication.audio_ms
                    session.pending_event_count -= publication.event_count
                    if not session.pending_publications:
                        session.publication_signal.clear()
                    batch = publication.build()
                    if len(batch) != publication.event_count:
                        self._fail_session_locked(
                            session,
                            LocalProviderPublicationError(
                                "provider event publication shape changed"
                            ),
                        )
                        return
                    session.in_flight = publication
                    session.in_flight_batch_size = len(batch)
                    session.in_flight_cancelled = False
                    session.in_flight_committed = False

                    def commit_publication(
                        publication: _Publication = publication,
                    ) -> None:
                        if (
                            session.in_flight is not publication
                            or session.in_flight_committed
                        ):
                            return
                        session.in_flight_committed = True
                        if publication.on_success is not None:
                            publication.on_success()

                    publish_task = asyncio.create_task(
                        session.publish(
                            batch,
                            commit_publication,
                        )
                    )
                    session.in_flight_task = publish_task

                callback_cancelled = False
                try:
                    await publish_task
                except asyncio.CancelledError:
                    async with session.lock:
                        callback_cancelled = session.in_flight_cancelled
                    if not callback_cancelled:
                        raise
                except Exception:
                    await self._publication_failed(session)
                    return

                async with session.lock:
                    if not callback_cancelled and not session.in_flight_committed:
                        self._fail_session_locked(
                            session,
                            LocalProviderPublicationError(
                                "provider event publication was not committed"
                            ),
                        )
                        session.publication_idle.set()
                        return
                    session.in_flight = None
                    session.in_flight_batch_size = 0
                    session.in_flight_task = None
                    session.in_flight_cancelled = False
                    session.in_flight_committed = False
                    session.publication_space.set()
                    if session.pending_publications:
                        session.publication_signal.set()
                    else:
                        session.publication_idle.set()
                        if session.closed:
                            return
        finally:
            self._retire_session(session)

    async def _publication_failed(
        self,
        session: _Session,
    ) -> None:
        error = LocalProviderPublicationError("provider event publication failed")
        async with session.lock:
            self._fail_session_locked(session, error)
            session.in_flight = None
            session.in_flight_batch_size = 0
            session.in_flight_task = None
            session.in_flight_cancelled = False
            session.in_flight_committed = False
            session.publication_idle.set()

    async def _publish_debug(
        self,
        session: _Session,
        utterance: _Utterance,
        source_text: str,
        translation: str,
    ) -> None:
        await self._enqueue_publication(
            session,
            utterance,
            _Publication(
                kind=_PublicationKind.DEBUG,
                event_count=2,
                utterance_id=utterance.utterance_id,
                build=lambda: (
                    ProviderTranscriptDelta(
                        session_id=session.request.session_id,
                        direction_id=session.request.direction_id,
                        stream_id=utterance.stream_id,
                        utterance_id=utterance.utterance_id,
                        event_sequence=session.next_event_sequence(),
                        text=source_text,
                        is_final=True,
                    ),
                    ProviderTranslationDelta(
                        session_id=session.request.session_id,
                        direction_id=session.request.direction_id,
                        stream_id=utterance.stream_id,
                        utterance_id=utterance.utterance_id,
                        event_sequence=session.next_event_sequence(),
                        text=translation,
                        stable_prefix=True,
                        is_final=True,
                    ),
                ),
            ),
            require_debug=True,
        )

    async def _publish_audio(
        self,
        session: _Session,
        utterance: _Utterance,
        pcm: bytes,
        sequence: int,
    ) -> None:
        output = session.request.requested_output_format
        await self._enqueue_publication(
            session,
            utterance,
            _Publication(
                kind=_PublicationKind.AUDIO,
                event_count=1,
                utterance_id=utterance.utterance_id,
                audio_ms=output.frame_duration_ms,
                on_success=lambda: setattr(
                    utterance,
                    "last_audio_sequence",
                    sequence,
                ),
                build=lambda: (
                    ProviderAudioDelta(
                        session_id=session.request.session_id,
                        direction_id=session.request.direction_id,
                        stream_id=utterance.stream_id,
                        utterance_id=utterance.utterance_id,
                        sequence=sequence,
                        event_sequence=session.next_event_sequence(),
                        provider_monotonic_ns=self._now_ns(),
                        sample_rate_hz=output.sample_rate_hz,
                        channels=output.channels,
                        sample_format=output.sample_format,
                        frame_duration_ms=output.frame_duration_ms,
                        pcm=pcm,
                    ),
                ),
            ),
        )

    async def _complete(
        self,
        session: _Session,
        utterance: _Utterance,
        *,
        asr_ms: int,
        mt_ms: int,
        tts_ms: int | None,
    ) -> None:
        async with session.lock:
            if not self._is_current(session, utterance):
                raise SchedulerStale(_PROVIDER_STALE_MESSAGE)
            self._terminalize_local(session, utterance)
            self._append_publication_locked(
                session,
                _Publication(
                    kind=_PublicationKind.TERMINAL,
                    event_count=2,
                    utterance_id=utterance.utterance_id,
                    build=lambda: (
                        self._latency(
                            session,
                            utterance,
                            asr_ms=asr_ms,
                            mt_ms=mt_ms,
                            tts_ms=tts_ms,
                        ),
                        ProviderUtteranceFinal(
                            session_id=session.request.session_id,
                            direction_id=session.request.direction_id,
                            stream_id=utterance.stream_id,
                            utterance_id=utterance.utterance_id,
                            event_sequence=(session.next_event_sequence()),
                            final_audio_sequence=(utterance.last_audio_sequence),
                            outcome=UtteranceOutcome.COMPLETED,
                        ),
                    ),
                ),
            )

    async def _drop(
        self,
        session: _Session,
        utterance: _Utterance,
        *,
        code: SafeErrorCode,
        asr_ms: int | None,
        mt_ms: int | None,
        tts_ms: int | None,
    ) -> None:
        async with session.lock:
            if not self._is_current(session, utterance):
                return
            await self._drop_locked(
                session,
                utterance,
                code=code,
                asr_ms=asr_ms,
                mt_ms=mt_ms,
                tts_ms=tts_ms,
            )

    async def _drop_locked(
        self,
        session: _Session,
        utterance: _Utterance,
        *,
        code: SafeErrorCode,
        asr_ms: int | None = None,
        mt_ms: int | None = None,
        tts_ms: int | None = None,
    ) -> None:
        self._cancel_commit(utterance)
        self._purge_publications_locked(
            session,
            utterance.utterance_id,
        )
        self._terminalize_local(session, utterance)
        self._append_publication_locked(
            session,
            _Publication(
                kind=_PublicationKind.TERMINAL,
                event_count=3,
                utterance_id=utterance.utterance_id,
                build=lambda: (
                    self._latency(
                        session,
                        utterance,
                        asr_ms=asr_ms,
                        mt_ms=mt_ms,
                        tts_ms=tts_ms,
                    ),
                    PrivacySafeProviderError(
                        session_id=session.request.session_id,
                        direction_id=session.request.direction_id,
                        stream_id=utterance.stream_id,
                        utterance_id=utterance.utterance_id,
                        event_sequence=(session.next_event_sequence()),
                        code=code,
                        retryable=True,
                        safe_message=SAFE_ERROR_MESSAGES[code],
                    ),
                    ProviderUtteranceFinal(
                        session_id=session.request.session_id,
                        direction_id=session.request.direction_id,
                        stream_id=utterance.stream_id,
                        utterance_id=utterance.utterance_id,
                        event_sequence=(session.next_event_sequence()),
                        final_audio_sequence=(utterance.last_audio_sequence),
                        outcome=UtteranceOutcome.DROPPED,
                    ),
                ),
            ),
        )

    def _latency(
        self,
        session: _Session,
        utterance: _Utterance,
        *,
        asr_ms: int | None,
        mt_ms: int | None,
        tts_ms: int | None,
    ) -> ProviderLatency:
        return ProviderLatency(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            stream_id=utterance.stream_id,
            utterance_id=utterance.utterance_id,
            event_sequence=session.next_event_sequence(),
            asr_first_text_ms=asr_ms,
            asr_final_text_ms=asr_ms,
            mt_first_text_ms=mt_ms,
            tts_first_audio_ms=tts_ms,
            provider_total_ms=self._elapsed_ms(utterance.capture_onset_ns),
        )

    def _health(
        self,
        session: _Session,
        *,
        allocate_sequence: bool = True,
    ) -> ProviderHealth:
        asr_device = ComputeDevice(getattr(self._asr, "actual_device", "cpu"))
        effective_asr_id = (
            getattr(self._asr, "resident_model_id", None) or self._asr_model_id
        )
        models = (
            self._model_health(
                ModelKind.ASR,
                effective_asr_id,
                asr_device,
                self._model_state(session, ModelKind.ASR),
            ),
            self._model_health(
                ModelKind.MT,
                self._mt_model_id,
                self._mt_device,
                self._model_state(session, ModelKind.MT),
            ),
            self._model_health(
                ModelKind.TTS,
                self._tts_model_id,
                ComputeDevice.CPU,
                self._model_state(session, ModelKind.TTS),
            ),
        )
        states = tuple(
            self._model_state(session, kind)
            for kind in (
                ModelKind.ASR,
                ModelKind.MT,
                ModelKind.TTS,
            )
        )
        unavailable = any(state is ModelState.FAILED for state in states)
        starting = any(
            state in {ModelState.NOT_LOADED, ModelState.LOADING} for state in states
        )
        degraded = bool(
            getattr(self._asr, "degraded", False)
            or asr_device is ComputeDevice.CPU
            or self._mt_device is ComputeDevice.CPU
        )
        state = self._provider_state(
            unavailable=unavailable,
            starting=starting,
            degraded=degraded,
        )
        safe_error = (
            SafeErrorSummary(
                code=SafeErrorCode.MODEL_NOT_LOADED,
                message=SAFE_ERROR_MESSAGES[SafeErrorCode.MODEL_NOT_LOADED],
                retryable=True,
            )
            if unavailable
            else None
        )
        return ProviderHealth(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            event_sequence=(
                session.next_event_sequence()
                if allocate_sequence
                else session.event_sequence
            ),
            provider_id=ProviderId.LOCAL,
            provider_name="Local offline provider",
            state=state,
            models=models,
            queues=ProviderQueues(
                provider_input_buffered_ms=sum(
                    utterance.buffered_ms for utterance in session.utterances.values()
                ),
                provider_output_buffered_ms=(
                    session.pending_audio_ms
                    + (
                        session.in_flight.audio_ms
                        if session.in_flight is not None
                        else 0
                    )
                ),
                queue_lag_ms=max(
                    (
                        self._elapsed_ms(utterance.capture_onset_ns)
                        for utterance in session.utterances.values()
                        if utterance.pcm_parts
                    ),
                    default=0,
                ),
            ),
            safe_error=safe_error,
        )

    @staticmethod
    def _model_health(
        kind: ModelKind,
        model_id: str,
        device: ComputeDevice,
        state: ModelState,
    ) -> ModelHealth:
        return ModelHealth(
            kind=kind,
            id=model_id,
            state=state,
            device=device,
            safe_error_code=(
                SafeErrorCode.MODEL_NOT_LOADED.value
                if state is ModelState.FAILED
                else None
            ),
        )

    def _initial_model_states(
        self,
        request: OpenProviderSession,
    ) -> dict[ModelKind, ModelState]:
        asr_state = self._asr_initial_state()
        return {
            ModelKind.ASR: asr_state,
            ModelKind.MT: self._adapter_model_state(
                self._translator,
                default=ModelState.READY,
            ),
            ModelKind.TTS: self._adapter_model_state(
                self._tts,
                request.voice_profile,
                default=ModelState.READY,
            ),
        }

    @staticmethod
    def _provider_state(
        *,
        unavailable: bool,
        starting: bool,
        degraded: bool,
    ) -> ProviderState:
        if unavailable:
            return ProviderState.UNAVAILABLE
        if starting:
            return ProviderState.STARTING
        if degraded:
            return ProviderState.DEGRADED
        return ProviderState.READY

    def _asr_initial_state(self) -> ModelState:
        if bool(getattr(self._asr, "unavailable", False)):
            return ModelState.FAILED
        if getattr(self._asr, "resident_model_id", None) is not None:
            return ModelState.READY
        return ModelState.NOT_LOADED

    def _model_keys(
        self,
        request: OpenProviderSession,
    ) -> dict[ModelKind, tuple[ModelKind, str]]:
        voice = request.voice_profile
        return {
            ModelKind.ASR: (
                ModelKind.ASR,
                self._asr_model_id,
            ),
            ModelKind.MT: (
                ModelKind.MT,
                self._mt_model_id,
            ),
            ModelKind.TTS: (
                ModelKind.TTS,
                (f"{voice.language.value}:{voice.gender.value}:{self._tts_model_id}"),
            ),
        }

    def _model_state(
        self,
        session: _Session,
        kind: ModelKind,
    ) -> ModelState:
        return self._model_states[session.model_keys[kind]]

    @staticmethod
    def _adapter_model_state(
        adapter: Any,
        *args: Any,
        default: ModelState,
    ) -> ModelState:
        if bool(getattr(adapter, "unavailable", False)):
            return ModelState.FAILED
        state = getattr(adapter, "model_state", default)
        if callable(state):
            state = state(*args)
        return state if isinstance(state, ModelState) else default

    def _collecting_utterance(
        self,
        session: _Session,
        frame: ProviderInputFrame,
    ) -> _Utterance:
        if frame.utterance_id in session.terminal_ids:
            raise LocalProviderProtocolError("utterance is already terminal")
        if session.collecting_id is None:
            if frame.utterance_id in session.utterances:
                raise LocalProviderProtocolError("utterance already reached EOU")
            if len(session.terminal_ids) + len(session.utterances) >= _MAX_TERMINAL_IDS:
                self._fail_session_locked(session)
                raise LocalProviderProtocolError("terminal identity capacity reached")
            utterance = _Utterance(
                utterance_id=frame.utterance_id,
                stream_id=frame.stream_id,
                capture_onset_ns=frame.capture_monotonic_ns,
                commit=SourceCommit(frame.utterance_id),
            )
            session.utterances[frame.utterance_id] = utterance
            session.collecting_id = frame.utterance_id
            return utterance
        if session.collecting_id != frame.utterance_id:
            raise LocalProviderProtocolError("utterance identity changed before EOU")
        return session.utterances[frame.utterance_id]

    def _validate_open(self, request: OpenProviderSession) -> None:
        input_format = request.requested_input_format
        if (
            input_format.sample_rate_hz != 16_000
            or input_format.channels != 1
            or input_format.sample_format is not SampleFormat.S16LE
        ):
            raise LocalProviderProtocolError("local ASR input format is unsupported")
        if request.source_language is request.target_language:
            raise LocalProviderProtocolError("language pair is unsupported")
        if request.voice_profile.language is not request.target_language:
            raise LocalProviderProtocolError("voice language does not match target")

    def _validate_frame(
        self,
        session: _Session,
        frame: ProviderInputFrame,
    ) -> None:
        request = session.request
        if (
            frame.direction_id is not request.direction_id
            or frame.source_language is not request.source_language
            or frame.target_language is not request.target_language
            or frame.mode is not request.mode
        ):
            raise LocalProviderProtocolError("frame session contract mismatch")
        input_format = request.requested_input_format
        if (
            frame.sample_rate_hz != input_format.sample_rate_hz
            or frame.channels != input_format.channels
            or frame.sample_format is not input_format.sample_format
            or frame.frame_duration_ms != input_format.frame_duration_ms
        ):
            raise LocalProviderProtocolError("frame PCM format mismatch")
        expected_bytes = (
            frame.sample_rate_hz * frame.channels * frame.frame_duration_ms // 1000 * 2
        )
        if len(frame.pcm) != expected_bytes:
            raise LocalProviderProtocolError("frame PCM length mismatch")
        if session.stream_id is None:
            session.stream_id = frame.stream_id
        elif frame.stream_id != session.stream_id:
            raise LocalProviderProtocolError("stream identity mismatch")
        expected_sequence = (
            0
            if session.last_input_sequence is None
            else session.last_input_sequence + 1
        )
        if frame.sequence != expected_sequence:
            raise LocalProviderProtocolError("frame sequence is not contiguous")

    @staticmethod
    def _validate_output_frame(
        pcm: bytes,
        request: OpenProviderSession,
    ) -> None:
        output = request.requested_output_format
        expected_bytes = (
            output.sample_rate_hz
            * output.channels
            * output.frame_duration_ms
            // 1000
            * 2
        )
        if len(pcm) != expected_bytes:
            raise SchedulerUnavailable("TTS output frame is unavailable")

    def _active_session(self, session_id: UUID) -> _Session:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            raise LocalProviderProtocolError("provider session is unavailable")
        return session

    @staticmethod
    def _is_current(
        session: _Session,
        utterance: _Utterance,
    ) -> bool:
        return (
            not session.closed
            and session.utterances.get(utterance.utterance_id) is utterance
            and utterance.utterance_id not in session.terminal_ids
        )

    def _terminalize_local(
        self,
        session: _Session,
        utterance: _Utterance,
    ) -> None:
        utterance.purge_source()
        if (
            utterance.utterance_id not in session.terminal_ids
            and len(session.terminal_ids) >= _MAX_TERMINAL_IDS
        ):
            self._fail_session_locked(session)
            raise LocalProviderProtocolError("terminal identity capacity reached")
        session.utterances.pop(utterance.utterance_id, None)
        session.terminal_ids.add(utterance.utterance_id)

    @staticmethod
    def _cancel_commit(utterance: _Utterance) -> None:
        try:
            utterance.commit.cancel()
        except Exception:
            pass

    def _models_available(self, session: _Session) -> bool:
        return all(
            self._model_state(session, kind) is not ModelState.FAILED
            for kind in (
                ModelKind.ASR,
                ModelKind.MT,
                ModelKind.TTS,
            )
        )

    def _fail_session_locked(
        self,
        session: _Session,
        error: LocalProviderPublicationError | None = None,
    ) -> None:
        if error is not None and session.publication_error is None:
            session.publication_error = error
        session.closed = True
        session.debug_text_enabled = False
        try:
            self._scheduler.close_session(session.request.session_id)
        except SchedulerStale:
            pass
        for utterance in tuple(session.utterances.values()):
            self._cancel_commit(utterance)
            utterance.purge_source()
        session.utterances.clear()
        session.collecting_id = None
        self._purge_publications_locked(session, None)
        session.pending_event_count = 0
        session.publication_space.set()
        session.publication_signal.set()

    def _retire_session(self, session: _Session) -> None:
        session_id = session.request.session_id
        if self._sessions.get(session_id) is session:
            self._sessions.pop(session_id)
            self._retired_sessions[session_id] = session
            self._retired_sessions.move_to_end(session_id)
            while len(self._retired_sessions) > _MAX_RETIRED_SESSIONS:
                self._retired_sessions.popitem(last=False)

    def _take_publication_error(
        self,
    ) -> LocalProviderPublicationError | None:
        for session in (
            *self._sessions.values(),
            *self._retired_sessions.values(),
        ):
            if (
                session.publication_error is not None
                and not session.publication_error_consumed
            ):
                session.publication_error_consumed = True
                return session.publication_error
        return None

    def _elapsed_ms(self, capture_onset_ns: int) -> int:
        return max(0, (self._now_ns() - capture_onset_ns) // 1_000_000)

    def _future_done(self, future: asyncio.Future[None]) -> None:
        try:
            future.exception()
        except (asyncio.CancelledError, Exception):
            pass
        self._futures.discard(future)
