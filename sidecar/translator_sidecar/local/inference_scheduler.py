"""Bounded duplex inference scheduling for local native model work."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from threading import Event
from typing import Any, TypeVar
from uuid import UUID

from translator_sidecar.provider_contract import AudioDirection


_QUEUED_PER_DIRECTION = 2
_TTS_BRIDGE_MS = 1200
_BRIDGE_POLL_SECONDS = 0.02
_Result = TypeVar("_Result")
_BRIDGE_END = object()
_BRIDGE_FAILURE = object()
_BRIDGE_OVERFLOW = object()
_SCHEDULER_STALE_MESSAGE = "scheduler result is stale"


class SchedulerOverflow(RuntimeError):
    """A per-direction inference queue reached its fixed capacity."""


class SchedulerStale(RuntimeError):
    """A generation change made an inference result inert."""


class SchedulerUnavailable(RuntimeError):
    """Native inference failed without exposing private payloads."""


@dataclass(frozen=True, slots=True)
class JobIdentity:
    session_id: UUID
    utterance_id: UUID
    direction: AudioDirection
    session_generation: int
    utterance_generation: int


@dataclass(slots=True)
class _Session:
    direction: AudioDirection
    generation: int
    utterances: dict[UUID, int]


@dataclass(slots=True)
class _Job:
    identity: JobIdentity
    work: Callable[[SchedulerContext], Awaitable[Any]]
    future: asyncio.Future[Any]
    cancelled: Event


@dataclass(frozen=True, slots=True)
class _NativeOutcome:
    ok: bool
    value: Any = None


class SchedulerContext:
    def __init__(
        self,
        scheduler: InferenceScheduler,
        job: _Job,
    ) -> None:
        self._scheduler = scheduler
        self._job = job
        self._bridge_high_water_ms = 0

    @property
    def identity(self) -> JobIdentity:
        return self._job.identity

    @property
    def bridge_high_water_ms(self) -> int:
        return self._bridge_high_water_ms

    def cancelled(self) -> bool:
        return self._job.cancelled.is_set() or not self._scheduler._is_current(
            self.identity
        )

    def ensure_current(self) -> None:
        if self.cancelled():
            raise SchedulerStale(_SCHEDULER_STALE_MESSAGE)

    async def run_gpu(self, operation: Callable[[], _Result]) -> _Result:
        self.ensure_current()
        loop = asyncio.get_running_loop()
        outcome = await loop.run_in_executor(
            self._scheduler._gpu_executor,
            self._run_native,
            operation,
        )
        self.ensure_current()
        if not outcome.ok:
            raise SchedulerUnavailable("native inference is unavailable")
        return outcome.value

    async def stream_tts(
        self,
        frames_factory: Callable[[], Iterator[bytes] | object],
        *,
        frame_duration_ms: int,
    ) -> AsyncIterator[bytes]:
        self.ensure_current()
        if frame_duration_ms <= 0 or _TTS_BRIDGE_MS % frame_duration_ms:
            raise SchedulerUnavailable("TTS bridge is unavailable")
        capacity = _TTS_BRIDGE_MS // frame_duration_ms
        queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=capacity)
        stream_cancelled = Event()
        loop = asyncio.get_running_loop()
        producer = loop.run_in_executor(
            self._scheduler._tts_executor,
            self._produce_tts,
            loop,
            queue,
            frames_factory,
            stream_cancelled,
            frame_duration_ms,
        )
        try:
            while True:
                self.ensure_current()
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=_BRIDGE_POLL_SECONDS,
                    )
                except TimeoutError:
                    continue
                self.ensure_current()
                if item is _BRIDGE_END:
                    return
                if item is _BRIDGE_FAILURE:
                    raise SchedulerUnavailable("TTS inference is unavailable")
                if item is _BRIDGE_OVERFLOW:
                    raise SchedulerOverflow("TTS output limit was reached")
                yield item
        finally:
            stream_cancelled.set()
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await producer

    @staticmethod
    def _run_native(
        operation: Callable[[], _Result],
    ) -> _NativeOutcome:
        try:
            return _NativeOutcome(ok=True, value=operation())
        except Exception:
            return _NativeOutcome(ok=False)

    def _produce_tts(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[bytes | object],
        frames_factory: Callable[[], Iterator[bytes] | object],
        stream_cancelled: Event,
        frame_duration_ms: int,
    ) -> None:
        iterator: Iterator[bytes] | None = None
        failed = False
        overflowed = False
        try:
            iterator = iter(frames_factory())
            while not self.cancelled() and not stream_cancelled.is_set():
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                if not self._put_bridge_item(
                    loop,
                    queue,
                    frame,
                    stream_cancelled,
                    frame_duration_ms,
                ):
                    return
        except SchedulerOverflow:
            overflowed = True
        except Exception:
            failed = True
        finally:
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        failed = True
            if not self.cancelled() and not stream_cancelled.is_set():
                if overflowed:
                    terminal = _BRIDGE_OVERFLOW
                elif failed:
                    terminal = _BRIDGE_FAILURE
                else:
                    terminal = _BRIDGE_END
                self._put_bridge_item(
                    loop,
                    queue,
                    terminal,
                    stream_cancelled,
                    frame_duration_ms,
                )

    def _put_bridge_item(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[bytes | object],
        item: bytes | object,
        stream_cancelled: Event,
        frame_duration_ms: int,
    ) -> bool:
        put = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        while True:
            try:
                put.result(timeout=_BRIDGE_POLL_SECONDS)
                self._bridge_high_water_ms = max(
                    self._bridge_high_water_ms,
                    queue.qsize() * frame_duration_ms,
                )
                return True
            except TimeoutError:
                if self.cancelled() or stream_cancelled.is_set():
                    put.cancel()
                    return False
            except Exception:
                return False


class InferenceScheduler:
    """Own per-direction admission, generation and native worker limits."""

    def __init__(self) -> None:
        self._directions = tuple(AudioDirection)
        self._queues: dict[AudioDirection, deque[_Job]] = {
            direction: deque() for direction in self._directions
        }
        self._active: dict[AudioDirection, _Job | None] = dict.fromkeys(
            self._directions
        )
        self._sessions: dict[UUID, _Session] = {}
        self._next_generation = 1
        self._dispatch_signal: asyncio.Event | None = None
        self._dispatcher: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._running_tasks: set[asyncio.Task[None]] = set()
        self._round_robin_index = 0
        self._closed = False
        self._gpu_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="translator-gpu",
        )
        self._tts_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="translator-tts",
        )

    def open_session(
        self,
        session_id: UUID,
        direction: AudioDirection,
    ) -> int:
        if self._closed:
            raise SchedulerUnavailable("scheduler is unavailable")
        if session_id in self._sessions:
            raise SchedulerUnavailable("scheduler session is unavailable")
        generation = self._allocate_generation()
        self._sessions[session_id] = _Session(
            direction=direction,
            generation=generation,
            utterances={},
        )
        return generation

    def open_utterance(
        self,
        session_id: UUID,
        utterance_id: UUID,
    ) -> JobIdentity:
        session = self._sessions.get(session_id)
        if session is None:
            raise SchedulerUnavailable("scheduler session is unavailable")
        if utterance_id in session.utterances:
            raise SchedulerUnavailable("scheduler utterance is unavailable")
        generation = self._allocate_generation()
        session.utterances[utterance_id] = generation
        return JobIdentity(
            session_id=session_id,
            utterance_id=utterance_id,
            direction=session.direction,
            session_generation=session.generation,
            utterance_generation=generation,
        )

    def submit(
        self,
        identity: JobIdentity,
        work: Callable[[SchedulerContext], Awaitable[_Result]],
    ) -> asyncio.Future[_Result]:
        if self._closed or not self._is_current(identity):
            raise SchedulerStale(_SCHEDULER_STALE_MESSAGE)
        queue = self._queues[identity.direction]
        admission_capacity = _QUEUED_PER_DIRECTION + int(
            self._active[identity.direction] is None
        )
        if len(queue) >= admission_capacity:
            raise SchedulerOverflow("inference queue limit was reached")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_Result] = loop.create_future()
        queue.append(
            _Job(
                identity=identity,
                work=work,
                future=future,
                cancelled=Event(),
            )
        )
        self._ensure_dispatcher()
        assert self._dispatch_signal is not None
        self._dispatch_signal.set()
        return future

    def cancel_utterance(
        self,
        session_id: UUID,
        utterance_id: UUID,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None or utterance_id not in session.utterances:
            raise SchedulerStale(_SCHEDULER_STALE_MESSAGE)
        session.utterances.pop(utterance_id)
        self._cancel_matching(
            lambda identity: (
                identity.session_id == session_id
                and identity.utterance_id == utterance_id
            )
        )

    def close_session(self, session_id: UUID) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise SchedulerStale(_SCHEDULER_STALE_MESSAGE)
        self._cancel_matching(lambda identity: identity.session_id == session_id)

    async def shutdown(self) -> None:
        if self._shutdown_task is None:
            self._closed = True
            self._shutdown_task = asyncio.create_task(self._shutdown_impl())
        shutdown_task = self._shutdown_task
        cancelled: BaseException | None = None
        while not shutdown_task.done():
            try:
                await asyncio.shield(shutdown_task)
            except BaseException as error:
                if not isinstance(error, asyncio.CancelledError):
                    raise
                cancelled = error
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        shutdown_task.result()
        if cancelled is not None:
            raise cancelled

    async def _shutdown_impl(self) -> None:
        for session_id in tuple(self._sessions):
            self.close_session(session_id)
        if self._dispatch_signal is not None:
            self._dispatch_signal.set()
        if self._running_tasks:
            await asyncio.gather(
                *tuple(self._running_tasks),
                return_exceptions=True,
            )
        if self._dispatcher is not None:
            await self._dispatcher
        self._gpu_executor.shutdown(wait=True, cancel_futures=True)
        self._tts_executor.shutdown(wait=True, cancel_futures=True)

    @property
    def tracked_session_count(self) -> int:
        return len(self._sessions)

    def tracked_utterance_count(self, session_id: UUID) -> int:
        session = self._sessions.get(session_id)
        return len(session.utterances) if session is not None else 0

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher is not None:
            return
        self._dispatch_signal = asyncio.Event()
        self._dispatcher = asyncio.create_task(self._dispatch_loop())

    async def _dispatch_loop(self) -> None:
        assert self._dispatch_signal is not None
        while True:
            self._dispatch_available()
            if self._closed and not self._running_tasks:
                return
            self._dispatch_signal.clear()
            if self._has_dispatchable():
                self._dispatch_signal.set()
            await self._dispatch_signal.wait()

    def _dispatch_available(self) -> None:
        for offset in range(len(self._directions)):
            index = (self._round_robin_index + offset) % len(self._directions)
            direction = self._directions[index]
            if self._active[direction] is not None:
                continue
            queue = self._queues[direction]
            while queue:
                job = queue.popleft()
                if not self._is_current(job.identity):
                    self._set_stale(job.future)
                    continue
                self._active[direction] = job
                task = asyncio.create_task(self._run_job(job))
                self._running_tasks.add(task)
                task.add_done_callback(self._task_done)
                self._round_robin_index = (index + 1) % len(self._directions)
                break

    def _has_dispatchable(self) -> bool:
        return any(
            self._active[direction] is None and self._queues[direction]
            for direction in self._directions
        )

    async def _run_job(self, job: _Job) -> None:
        context = SchedulerContext(self, job)
        try:
            result = await job.work(context)
            context.ensure_current()
        except SchedulerStale:
            self._terminalize(job.identity)
            self._set_stale(job.future)
        except (SchedulerUnavailable, SchedulerOverflow):
            self._terminalize(job.identity)
            if not job.future.done():
                job.future.set_exception(
                    SchedulerUnavailable("scheduler work is unavailable")
                )
        except Exception:
            self._terminalize(job.identity)
            if not job.future.done():
                job.future.set_exception(
                    SchedulerUnavailable("scheduler work is unavailable")
                )
        else:
            self._terminalize(job.identity)
            if not job.future.done():
                job.future.set_result(result)
        finally:
            direction = job.identity.direction
            if self._active[direction] is job:
                self._active[direction] = None
            if self._dispatch_signal is not None:
                self._dispatch_signal.set()

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._running_tasks.discard(task)
        if self._dispatch_signal is not None:
            self._dispatch_signal.set()

    def _cancel_matching(
        self,
        predicate: Callable[[JobIdentity], bool],
    ) -> None:
        for active in self._active.values():
            if active is not None and predicate(active.identity):
                active.cancelled.set()
        for direction, queue in self._queues.items():
            retained: deque[_Job] = deque()
            while queue:
                job = queue.popleft()
                if predicate(job.identity):
                    job.cancelled.set()
                    self._set_stale(job.future)
                else:
                    retained.append(job)
            self._queues[direction] = retained
        if self._dispatch_signal is not None:
            self._dispatch_signal.set()

    def _is_current(self, identity: JobIdentity) -> bool:
        session = self._sessions.get(identity.session_id)
        return (
            session is not None
            and session.direction is identity.direction
            and session.generation == identity.session_generation
            and session.utterances.get(identity.utterance_id)
            == identity.utterance_generation
        )

    def _terminalize(self, identity: JobIdentity) -> None:
        session = self._sessions.get(identity.session_id)
        if (
            session is not None
            and session.generation == identity.session_generation
            and session.utterances.get(identity.utterance_id)
            == identity.utterance_generation
        ):
            session.utterances.pop(identity.utterance_id)

    def _allocate_generation(self) -> int:
        generation = self._next_generation
        self._next_generation += 1
        return generation

    @staticmethod
    def _set_stale(future: asyncio.Future[Any]) -> None:
        if not future.done():
            future.set_exception(SchedulerStale(_SCHEDULER_STALE_MESSAGE))
