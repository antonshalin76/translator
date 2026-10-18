"""Session-owned HTTPS ASR, translation and streaming speech, without WS fallback."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import numpy as np
import soxr

from .cleanup import finish_cleanup
from .openai_http import MAX_PCM, OpenAIHttpFailure, OpenAIHttpPipeline
from .openai_provider import (
    OPENAI_PROVIDER_NAME,
    OpenAIRealtimeAdapter,
    OpenAIRealtimeConfig,
    model_health,
)
from .provider_contract import (
    MAX_TERMINAL_UTTERANCES_PER_SESSION,
    SAFE_ERROR_MESSAGES,
    CancelUtterance,
    CloseRequestReason,
    OpenProviderSession,
    PcmFormat,
    ProviderAudioDelta,
    ProviderCapabilities,
    ProviderHealth,
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
    UpdateDebugText,
    UtteranceOutcome,
    make_provider_error,
)
from .provider_registry import SessionDrainReceipt

PublishEvents = Callable[[tuple[object, ...], Callable[[], None]], Awaitable[None]]
_MAX_OUTPUT_SECONDS = 30
_MAX_READY_UTTERANCES = 3


class Pipeline(Protocol):
    async def warmup(self) -> None: ...
    async def transcribe(
        self, pcm: bytes, sample_rate: int, source_language: str
    ) -> str: ...
    async def translate(self, text: str, target_language: str) -> str: ...
    def speech(self, text: str, target_language: str) -> AsyncIterator[bytes]: ...
    async def close(self) -> None: ...


class OpenAIProviderProtocolError(ValueError):
    pass


class OpenAIProviderFailure(OpenAIProviderProtocolError):
    def __init__(
        self, code: SafeErrorCode, *, retryable: bool, resource_exhausted: bool = False
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.retryable = retryable
        self.resource_exhausted = resource_exhausted


class OpenAIProviderTransportFailure(OpenAIProviderFailure):
    pass


def _failure(error: Exception) -> OpenAIProviderFailure:
    if isinstance(error, OpenAIProviderFailure):
        return error
    if isinstance(error, OpenAIHttpFailure):
        failure_type = (
            OpenAIProviderTransportFailure if error.transport else OpenAIProviderFailure
        )
        return failure_type(
            error.code,
            retryable=error.retryable,
            resource_exhausted=error.resource_exhausted,
        )
    return OpenAIProviderTransportFailure(
        SafeErrorCode.PROVIDER_UNAVAILABLE, retryable=True
    )


@dataclass(frozen=True, slots=True)
class _Binding:
    stream_id: UUID
    utterance_id: UUID


@dataclass(slots=True)
class _Generation:
    binding: _Binding
    capture_monotonic_ns: int
    pcm: bytearray = field(default_factory=bytearray)
    task: asyncio.Task[None] | None = None
    retirement: asyncio.Task[None] | None = None
    revoked: bool = False
    input_complete: bool = False
    discarding: bool = False
    last_audio_sequence: int | None = None
    finalized: bool = False
    deadline: float | None = None
    output_deadline: float | None = None
    timeout: asyncio.Timeout | None = None
    asr_elapsed_ms: int | None = None


@dataclass(slots=True)
class _Session:
    owner: OpenAIRealtimeProvider
    request: OpenProviderSession
    publish: PublishEvents
    pipeline: Pipeline | None = None
    utterances: dict[UUID, _Generation] = field(default_factory=dict)
    collecting_id: UUID | None = None
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    publication_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    debug_text_enabled: bool = False
    event_sequence: int = 0
    closed: bool = False
    failed: bool = False
    opened: bool = False
    draining: bool = False
    stream_id: UUID | None = None
    last_input_sequence: int | None = None
    terminal_ids: set[UUID] = field(default_factory=set)
    opening_task: asyncio.Task | None = None
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
        pipeline_factory: Callable[..., Pipeline] | None = None,
        now_ns: Callable[[], int] = time.monotonic_ns,
        drain_timeout_seconds: float = 6,
    ) -> None:
        if not 0 < drain_timeout_seconds <= 30:
            raise ValueError("invalid_drain_timeout")
        self._config = config or OpenAIRealtimeConfig(cloud_opt_in=True)
        self._environ = environ if environ is not None else os.environ
        self._pipeline_factory = pipeline_factory or OpenAIHttpPipeline
        self._now_ns = now_ns
        self._drain_timeout = drain_timeout_seconds
        self._sessions: dict[UUID, _Session] = {}
        self._failed_opens: dict[UUID, _Session] = {}
        self._closed = False

    @property
    def _opening(self) -> dict[UUID, asyncio.Task]:
        return {
            session.request.session_id: session.opening_task
            for session in (*self._sessions.values(), *self._failed_opens.values())
            if session.opening_task is not None
        }

    def reserve_session(
        self, request: OpenProviderSession, publish: PublishEvents
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

    def _open_session(self, session: _Session) -> asyncio.Task:
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
            self._sessions.pop(session.request.session_id, None)
            raise
        return session.opening_task

    async def _run_open_session(self, session: _Session):
        request = session.request
        try:
            preflight = OpenAIRealtimeAdapter(
                self._config, environ=self._environ
            ).preflight_open_session(request)
            if not preflight.can_start:
                assert preflight.error is not None
                raise OpenAIProviderFailure(preflight.error.code, retryable=False)
            session.pipeline = self._pipeline_factory(
                self._environ[self._config.credential_env_name].strip(),
                vpn_interface=self._config.vpn_interface,
            )
            async with asyncio.timeout(10):
                await session.pipeline.warmup()
            if session.draining or session.closed:
                raise asyncio.CancelledError
            session.opened = True
            return ProviderSessionOpened(
                session_id=request.session_id,
                direction_id=request.direction_id,
                event_sequence=session.next_sequence(),
                negotiated_input_format=request.requested_input_format,
                negotiated_output_format=request.requested_output_format,
                capabilities=ProviderCapabilities(
                    transcript_delta=request.debug_text_enabled,
                    translation_delta=request.debug_text_enabled,
                    cancellation=True,
                    cloud_egress=True,
                ),
            ), self._health(session)
        except BaseException as error:
            session.closed = True
            try:
                await finish_cleanup(self._close_pipeline(session))
            finally:
                self._sessions.pop(request.session_id, None)
                if session.pipeline is not None:
                    self._failed_opens[request.session_id] = session
            if isinstance(error, asyncio.CancelledError):
                raise
            raise _failure(error) from None
        finally:
            session.opening_task = None

    async def submit_frame(self, frame: ProviderInputFrame) -> None:
        session = self._active_session(frame.session_id)
        overflow = None
        async with session.lock:
            self._active_session(frame.session_id)
            self._validate_frame(session, frame)
            if session.collecting_id not in (None, frame.utterance_id):
                raise OpenAIProviderProtocolError("input_utterance_mismatch")
            generation = session.utterances.get(frame.utterance_id)
            if generation is None:
                generation = _Generation(
                    _Binding(frame.stream_id, frame.utterance_id),
                    frame.capture_monotonic_ns,
                )
                session.utterances[frame.utterance_id] = generation
                session.collecting_id = frame.utterance_id
            if generation.revoked:
                raise OpenAIProviderProtocolError("openai_retirement_failed")
            if generation.input_complete:
                raise OpenAIProviderProtocolError("input_sequence_mismatch")
            session.stream_id = frame.stream_id
            session.last_input_sequence = frame.sequence
            if len(generation.pcm) + len(frame.pcm) > frame.sample_rate_hz * 2 * 12:
                generation.discarding = True
                generation.pcm.clear()
            if not generation.discarding:
                generation.pcm.extend(frame.pcm)
            if frame.end_of_utterance:
                session.collecting_id = None
                ready = sum(
                    item.input_complete and not item.revoked
                    for item in session.utterances.values()
                )
                generation.input_complete = True
                generation.deadline = (
                    asyncio.get_running_loop().time() + self._drain_timeout
                )
                if generation.discarding or ready >= _MAX_READY_UTTERANCES:
                    overflow = generation
                else:
                    generation.task = asyncio.create_task(
                        self._process(session, generation)
                    )
        if overflow is not None:
            await self._retire(
                session,
                overflow,
                UtteranceOutcome.DROPPED,
                OpenAIProviderFailure(SafeErrorCode.QUEUE_OVERFLOW, retryable=True),
            )

    async def _process(self, session: _Session, generation: _Generation) -> None:
        try:
            assert session.pipeline is not None
            if (
                generation.deadline is not None
                and asyncio.get_running_loop().time() > generation.deadline
            ):
                raise TimeoutError
            async with asyncio.timeout_at(generation.deadline) as deadline:
                generation.timeout = deadline
                async with session.processing_lock:
                    if generation.revoked:
                        return
                    if session.failed:
                        raise OpenAIProviderFailure(
                            SafeErrorCode.PROVIDER_UNAVAILABLE, retryable=False
                        )
                    source = await session.pipeline.transcribe(
                        bytes(generation.pcm),
                        session.request.requested_input_format.sample_rate_hz,
                        session.request.source_language.value,
                    )
                    generation.pcm.clear()
                    await self._debug(session, generation, source, translation=False)
                    translated = await session.pipeline.translate(
                        source, session.request.target_language.value
                    )
                    await self._debug(session, generation, translated, translation=True)
                    target_rate = session.request.requested_output_format.sample_rate_hz
                    resampler = (
                        soxr.ResampleStream(
                            24_000, target_rate, 1, dtype="float32", quality="QQ"
                        )
                        if target_rate != 24_000
                        else None
                    )
                    pending = bytearray()
                    wire_pending = bytearray()
                    total = 0
                    async with aclosing(
                        session.pipeline.speech(
                            translated, session.request.target_language.value
                        )
                    ) as speech:
                        async for pcm in speech:
                            if not pcm:
                                raise OpenAIProviderProtocolError("invalid_output_pcm")
                            total += len(pcm)
                            if total > MAX_PCM:
                                raise OpenAIProviderProtocolError(
                                    "output_audio_capacity_exhausted"
                                )
                            wire_pending.extend(pcm)
                            complete = len(wire_pending) - len(wire_pending) % 2
                            pending.extend(
                                self._resample(
                                    bytes(wire_pending[:complete]), resampler
                                )
                            )
                            del wire_pending[:complete]
                            await self._audio(session, generation, pending)
                    if wire_pending or not total:
                        raise OpenAIProviderProtocolError("empty_output_pcm")
                    if resampler is not None:
                        pending.extend(self._resample(b"", resampler, last=True))
                    await self._audio(session, generation, pending, final=True)
                    if not generation.revoked:
                        await self._retire(
                            session, generation, UtteranceOutcome.COMPLETED
                        )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if generation.revoked:
                return
            failure = _failure(error)
            if failure.code is SafeErrorCode.NO_SPEECH:
                generation.asr_elapsed_ms = max(
                    0, (self._now_ns() - generation.capture_monotonic_ns) // 1_000_000
                )
            session.failed = session.failed or (
                not failure.retryable and failure.code is not SafeErrorCode.NO_SPEECH
            )
            try:
                await self._retire(
                    session, generation, UtteranceOutcome.DROPPED, failure
                )
            except Exception:
                session.failed = True
        finally:
            generation.timeout = None

    async def _debug(self, session, generation, text, *, translation):
        async with session.publication_lock:
            if generation.revoked or not session.debug_text_enabled:
                return
            event_type = (
                ProviderTranslationDelta if translation else ProviderTranscriptDelta
            )
            extra = {"stable_prefix": True} if translation else {}
            await self._publish(
                session,
                (
                    event_type(
                        session_id=session.request.session_id,
                        direction_id=session.request.direction_id,
                        stream_id=generation.binding.stream_id,
                        utterance_id=generation.binding.utterance_id,
                        event_sequence=session.next_sequence(),
                        text=text,
                        is_final=True,
                        **extra,
                    ),
                ),
            )

    async def _audio(self, session, generation, pending: bytearray, *, final=False):
        rate = session.request.requested_output_format.sample_rate_hz
        size = rate * 2 // 50
        if final and len(pending) % size:
            pending.extend(bytes(size - len(pending) % size))
        while len(pending) >= size:
            pcm = bytes(pending[:size])
            del pending[:size]
            async with session.publication_lock:
                if generation.revoked:
                    return
                sequence = (
                    0
                    if generation.last_audio_sequence is None
                    else generation.last_audio_sequence + 1
                )
                await self._publish(
                    session,
                    (
                        ProviderAudioDelta(
                            session_id=session.request.session_id,
                            direction_id=session.request.direction_id,
                            stream_id=generation.binding.stream_id,
                            utterance_id=generation.binding.utterance_id,
                            sequence=sequence,
                            event_sequence=session.next_sequence(),
                            provider_monotonic_ns=self._now_ns(),
                            sample_rate_hz=rate,
                            channels=1,
                            sample_format=SampleFormat.S16LE,
                            frame_duration_ms=20,
                            pcm=pcm,
                        ),
                    ),
                    on_commit=lambda sequence=sequence: self._audio_committed(
                        generation, sequence
                    ),
                )

    def _audio_committed(self, generation, sequence):
        generation.last_audio_sequence = sequence
        now = asyncio.get_running_loop().time()
        if generation.output_deadline is None:
            generation.output_deadline = now + _MAX_OUTPUT_SECONDS + self._drain_timeout
        generation.deadline = min(now + self._drain_timeout, generation.output_deadline)
        if generation.timeout is not None and not generation.timeout.expired():
            generation.timeout.reschedule(generation.deadline)

    @staticmethod
    def _resample(pcm, resampler, *, last=False):
        if resampler is None:
            return pcm
        return (
            np.rint(
                resampler.resample_chunk(
                    np.frombuffer(pcm, dtype="<i2").astype(np.float32), last=last
                )
            )
            .clip(-32768, 32767)
            .astype("<i2", copy=False)
            .tobytes()
        )

    @staticmethod
    def _revoke(session, generation):
        if not generation.revoked:
            generation.revoked = True
            task = generation.task
            if (
                task is not None
                and task is not asyncio.current_task()
                and not task.done()
            ):
                task.cancel()
        session.terminal_ids.add(generation.binding.utterance_id)
        generation.pcm.clear()
        if session.collecting_id == generation.binding.utterance_id:
            session.collecting_id = None

    async def _retire(self, session, generation, outcome, failure=None):
        self._revoke(session, generation)
        if generation.retirement is None or (
            generation.retirement.done()
            and (
                generation.retirement.cancelled()
                or generation.retirement.exception() is not None
            )
        ):
            generation.retirement = asyncio.create_task(
                self._dispose_generation(
                    session, generation, asyncio.current_task(), outcome, failure
                )
            )
        await finish_cleanup(generation.retirement)
        if (
            generation.task is not None
            and generation.task is not asyncio.current_task()
            and not generation.task.done()
        ):
            done, _ = await asyncio.wait((generation.task,), timeout=1.8)
            if not done:
                session.failed = True
                raise OpenAIProviderProtocolError("openai_retirement_failed")

    async def _dispose_generation(self, session, generation, owner, outcome, failure):
        task = generation.task
        if task is not None and task is not owner and not task.done():
            done, _ = await asyncio.wait((task,), timeout=1.8)
            if not done:
                session.failed = True
                raise OpenAIProviderProtocolError("openai_retirement_failed")
        async with session.publication_lock:
            if not generation.finalized:
                batch = []
                if failure and failure.code is SafeErrorCode.NO_SPEECH:
                    batch.append(
                        ProviderLatency(
                            session_id=session.request.session_id,
                            direction_id=session.request.direction_id,
                            stream_id=generation.binding.stream_id,
                            utterance_id=generation.binding.utterance_id,
                            event_sequence=session.next_sequence(),
                            asr_first_text_ms=generation.asr_elapsed_ms,
                            asr_final_text_ms=generation.asr_elapsed_ms,
                            provider_total_ms=generation.asr_elapsed_ms,
                        )
                    )
                if failure:
                    batch.append(self._error(session, generation, failure))
                batch.append(
                    ProviderUtteranceFinal(
                        session_id=session.request.session_id,
                        direction_id=session.request.direction_id,
                        stream_id=generation.binding.stream_id,
                        utterance_id=generation.binding.utterance_id,
                        event_sequence=session.next_sequence(),
                        final_audio_sequence=generation.last_audio_sequence,
                        outcome=outcome,
                    )
                )
                generation.finalized = True
                await self._publish(session, tuple(batch))
        if session.utterances.get(generation.binding.utterance_id) is generation:
            session.utterances.pop(generation.binding.utterance_id)

    async def cancel_utterance(self, request: CancelUtterance) -> None:
        session = self._active_session(request.session_id)
        if request.direction_id != session.request.direction_id:
            raise OpenAIProviderProtocolError("direction_identity_mismatch")
        generation = session.utterances.get(request.utterance_id)
        if generation is not None:
            await self._retire(session, generation, UtteranceOutcome.CANCELLED)

    async def update_debug_text(self, request: UpdateDebugText) -> None:
        session = self._active_session(request.session_id)
        async with session.publication_lock:
            session.debug_text_enabled = request.enabled

    async def _close_pipeline(self, session):
        if session.pipeline is not None:
            try:
                async with asyncio.timeout(1.8):
                    await session.pipeline.close()
            except Exception:
                raise OpenAIProviderProtocolError("openai_retirement_failed") from None
            session.pipeline = None

    async def _drain_session(self, session, reason):
        opening = session.opening_task
        if opening is not None and opening is not asyncio.current_task():
            opening.cancel()
            await asyncio.gather(opening, return_exceptions=True)
        session.closed = True
        owned = tuple(session.utterances.values())
        for generation in owned:
            self._revoke(session, generation)
        results = await asyncio.gather(
            *(
                self._retire(session, generation, UtteranceOutcome.CANCELLED)
                for generation in owned
            ),
            return_exceptions=True,
        )
        if any(isinstance(result, BaseException) for result in results):
            raise OpenAIProviderProtocolError("openai_retirement_failed")
        await self._close_pipeline(session)
        if session.opened and session.delivery_error is None:
            async with session.publication_lock:
                try:
                    await self._publish(
                        session,
                        (
                            ProviderSessionClosed(
                                session_id=session.request.session_id,
                                direction_id=session.request.direction_id,
                                event_sequence=session.next_sequence(),
                                reason=SessionCloseReason(reason.value),
                            ),
                        ),
                    )
                except Exception:
                    pass  # The receipt retains failed delivery after resources are drained.
        for sessions in (self._sessions, self._failed_opens):
            if sessions.get(session.request.session_id) is session:
                sessions.pop(session.request.session_id)
        return SessionDrainReceipt(session.request.session_id, session.delivery_error)

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

    @staticmethod
    async def _publish(session, batch, *, on_commit=None):
        committed = False

        def commit():
            nonlocal committed
            if committed:
                return
            committed = True
            if on_commit is not None:
                on_commit()

        try:
            async with asyncio.timeout(2):
                await session.publish(batch, commit)
            if not committed:
                raise RuntimeError("provider_publication_not_committed")
        except Exception:
            session.delivery_error = SafeErrorCode.PROVIDER_UNAVAILABLE
            session.failed = True
            raise

    @staticmethod
    def _error(session, generation, failure):
        return make_provider_error(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            stream_id=generation.binding.stream_id,
            utterance_id=generation.binding.utterance_id,
            event_sequence=session.next_sequence(),
            code=failure.code,
            retryable=failure.retryable,
        )

    @staticmethod
    def _health(session, failure=None):
        return ProviderHealth(
            session_id=session.request.session_id,
            direction_id=session.request.direction_id,
            event_sequence=session.next_sequence(),
            provider_id=session.request.provider_id,
            provider_name=OPENAI_PROVIDER_NAME,
            state=ProviderState.UNAVAILABLE if failure else ProviderState.READY,
            models=model_health(failure.code if failure else None),
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

    def _active_session(self, session_id):
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
    def _validate_format(value: PcmFormat):
        if (
            value.channels != 1
            or value.sample_format is not SampleFormat.S16LE
            or value.frame_duration_ms != 20
            or value.sample_rate_hz not in {16_000, 24_000, 48_000}
        ):
            raise OpenAIProviderProtocolError("unsupported_pcm_format")

    @staticmethod
    def _validate_frame(session, frame):
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
        if frame.utterance_id not in session.utterances and (
            len(session.terminal_ids) + len(session.utterances)
            >= MAX_TERMINAL_UTTERANCES_PER_SESSION
        ):
            raise OpenAIProviderProtocolError("terminal_identity_capacity_exhausted")
        if (
            session.last_input_sequence is not None
            and frame.sequence <= session.last_input_sequence
        ):
            raise OpenAIProviderProtocolError("input_sequence_mismatch")
