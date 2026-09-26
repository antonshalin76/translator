from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
from threading import Event as ThreadEvent
from threading import Lock, current_thread
from uuid import uuid4

import pytest

from translator_sidecar.local.inference_scheduler import (
    InferenceScheduler,
    SchedulerContext,
    SchedulerOverflow,
    SchedulerStale,
    SchedulerUnavailable,
)
from translator_sidecar.provider_contract import AudioDirection


def run(coroutine):
    return asyncio.run(coroutine)


def test_scheduler_allows_one_active_and_two_queued_per_direction() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.MICROPHONE)
        started = asyncio.Event()
        release = asyncio.Event()
        order: list[int] = []

        async def work(_context: SchedulerContext, value: int) -> int:
            order.append(value)
            if value == 0:
                started.set()
                await release.wait()
            return value

        try:
            identities = [
                scheduler.open_utterance(session_id, uuid4()) for _ in range(4)
            ]
            first = scheduler.submit(identities[0], lambda context: work(context, 0))
            await asyncio.wait_for(started.wait(), timeout=1)
            second = scheduler.submit(identities[1], lambda context: work(context, 1))
            third = scheduler.submit(identities[2], lambda context: work(context, 2))
            with pytest.raises(SchedulerOverflow, match="queue"):
                scheduler.submit(identities[3], lambda context: work(context, 3))
            release.set()
            assert await asyncio.gather(first, second, third) == [0, 1, 2]
            assert order == [0, 1, 2]
        finally:
            await scheduler.shutdown()

    run(scenario())


def test_scheduler_capacity_is_independent_per_direction() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        release = asyncio.Event()
        started = {
            AudioDirection.MICROPHONE: asyncio.Event(),
            AudioDirection.SPEAKER: asyncio.Event(),
        }
        sessions = {direction: uuid4() for direction in AudioDirection}
        for direction, session_id in sessions.items():
            scheduler.open_session(session_id, direction)

        async def work(context: SchedulerContext) -> str:
            started[context.identity.direction].set()
            await release.wait()
            return context.identity.direction.value

        try:
            futures = []
            for _direction, session_id in sessions.items():
                for _ in range(3):
                    identity = scheduler.open_utterance(session_id, uuid4())
                    futures.append(scheduler.submit(identity, work))
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in started.values())),
                timeout=1,
            )
            release.set()
            results = await asyncio.gather(*futures)
            assert results.count("microphone") == 3
            assert results.count("speaker") == 3
        finally:
            await scheduler.shutdown()

    run(scenario())


def test_gpu_work_is_single_worker_and_round_robin_across_directions() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        sessions = {direction: uuid4() for direction in AudioDirection}
        for direction, session_id in sessions.items():
            scheduler.open_session(session_id, direction)
        active = 0
        max_active = 0
        native_lock = Lock()
        order: list[AudioDirection] = []
        first_entered = ThreadEvent()
        second_entered = ThreadEvent()
        release_first = ThreadEvent()
        speaker_gpu_attempt = asyncio.Event()

        def native(direction: AudioDirection) -> str:
            nonlocal active, max_active
            with native_lock:
                active += 1
                max_active = max(max_active, active)
                order.append(direction)
                call_index = len(order)
            if call_index == 1:
                first_entered.set()
                assert release_first.wait(timeout=2)
            else:
                second_entered.set()
            try:
                return direction.value
            finally:
                with native_lock:
                    active -= 1

        async def work(context: SchedulerContext) -> str:
            if context.identity.direction is AudioDirection.SPEAKER:
                speaker_gpu_attempt.set()
            return await context.run_gpu(lambda: native(context.identity.direction))

        try:
            microphone = [
                scheduler.submit(
                    scheduler.open_utterance(
                        sessions[AudioDirection.MICROPHONE], uuid4()
                    ),
                    work,
                )
                for _ in range(3)
            ]
            assert await asyncio.to_thread(first_entered.wait, 1)
            speaker = scheduler.submit(
                scheduler.open_utterance(sessions[AudioDirection.SPEAKER], uuid4()),
                work,
            )
            await asyncio.wait_for(speaker_gpu_attempt.wait(), timeout=1)
            assert second_entered.is_set() is False
            release_first.set()
            assert await asyncio.gather(
                microphone[0], speaker, microphone[1], microphone[2]
            ) == [
                "microphone",
                "speaker",
                "microphone",
                "microphone",
            ]
            assert max_active == 1
            assert order == [
                AudioDirection.MICROPHONE,
                AudioDirection.SPEAKER,
                AudioDirection.MICROPHONE,
                AudioDirection.MICROPHONE,
            ]
        finally:
            release_first.set()
            await scheduler.shutdown()

    run(scenario())


def test_tts_workers_are_limited_to_two() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        sessions = {direction: uuid4() for direction in AudioDirection}
        for direction, session_id in sessions.items():
            scheduler.open_session(session_id, direction)
        active = 0
        max_active = 0
        native_lock = Lock()
        both_entered = ThreadEvent()
        release = ThreadEvent()
        attempts = 0
        all_attempted = asyncio.Event()

        def frames() -> object:
            nonlocal active, max_active
            with native_lock:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    both_entered.set()
            assert release.wait(timeout=2)
            try:
                yield b"\x00" * 3200
            finally:
                with native_lock:
                    active -= 1

        async def consume_one(context: SchedulerContext) -> int:
            nonlocal attempts
            stream = context.stream_tts(frames, frame_duration_ms=100)
            attempts += 1
            if attempts == 3:
                all_attempted.set()
            try:
                await anext(stream)
                return 1
            finally:
                await stream.aclose()

        async def microphone_work(context: SchedulerContext) -> int:
            return sum(
                await asyncio.gather(
                    consume_one(context),
                    consume_one(context),
                )
            )

        async def speaker_work(context: SchedulerContext) -> int:
            return await consume_one(context)

        try:
            futures = [
                scheduler.submit(
                    scheduler.open_utterance(
                        sessions[AudioDirection.MICROPHONE], uuid4()
                    ),
                    microphone_work,
                ),
                scheduler.submit(
                    scheduler.open_utterance(sessions[AudioDirection.SPEAKER], uuid4()),
                    speaker_work,
                ),
            ]
            await asyncio.wait_for(all_attempted.wait(), timeout=1)
            assert await asyncio.to_thread(both_entered.wait, 1)
            assert active == 2
            release.set()
            assert await asyncio.gather(*futures) == [2, 1]
            assert max_active == 2
        finally:
            release.set()
            await scheduler.shutdown()

    run(scenario())


def test_tts_bridge_applies_1200ms_backpressure() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.MICROPHONE)
        produced = 0
        producer_lock = Lock()
        first_frame = asyncio.Event()
        release_consumer = asyncio.Event()
        observed_high_water = 0
        producer_finalized = ThreadEvent()

        def frames() -> object:
            nonlocal produced
            try:
                for _ in range(100):
                    with producer_lock:
                        produced += 1
                    yield b"\x00" * 3200
            finally:
                producer_finalized.set()

        async def work(context: SchedulerContext) -> int:
            nonlocal observed_high_water
            consumed = 0
            stream = context.stream_tts(frames, frame_duration_ms=100)
            try:
                async for _frame in stream:
                    consumed += 1
                    observed_high_water = max(
                        observed_high_water,
                        context.bridge_high_water_ms,
                    )
                    if consumed == 1:
                        first_frame.set()
                        await release_consumer.wait()
                    if consumed == 20:
                        break
            finally:
                await stream.aclose()
            return consumed

        try:
            future = scheduler.submit(
                scheduler.open_utterance(session_id, uuid4()),
                work,
            )
            await asyncio.wait_for(first_frame.wait(), timeout=1)
            await asyncio.sleep(0.05)
            with producer_lock:
                produced_while_blocked = produced
            assert produced_while_blocked <= 14
            release_consumer.set()
            assert await future == 20
            assert observed_high_water <= 1200
            assert await asyncio.to_thread(producer_finalized.wait, 1)
            with producer_lock:
                production_at_close = produced
            await asyncio.sleep(0.05)
            with producer_lock:
                assert produced == production_at_close
        finally:
            release_consumer.set()
            await scheduler.shutdown()

    run(scenario())


@pytest.mark.parametrize("cancel_kind", ["utterance", "session"])
def test_generation_change_purges_tts_bridge_and_stops_producer(
    cancel_kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        utterance_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.MICROPHONE)
        identity = scheduler.open_utterance(session_id, utterance_id)
        first_delivered = asyncio.Event()
        continue_after_cancel = asyncio.Event()
        producer_finalized = ThreadEvent()
        pulls = 0
        pulls_lock = Lock()
        marker = b"private-pcm-marker"
        delivered: list[bytes] = []
        caplog.set_level(logging.DEBUG)

        def frames() -> object:
            nonlocal pulls
            try:
                while True:
                    with pulls_lock:
                        pulls += 1
                    yield marker
            finally:
                producer_finalized.set()

        async def work(context: SchedulerContext) -> None:
            stream = context.stream_tts(frames, frame_duration_ms=100)
            try:
                delivered.append(await anext(stream))
                first_delivered.set()
                await continue_after_cancel.wait()
                delivered.append(await anext(stream))
            finally:
                await stream.aclose()

        try:
            future = scheduler.submit(identity, work)
            await asyncio.wait_for(first_delivered.wait(), timeout=1)
            if cancel_kind == "utterance":
                scheduler.cancel_utterance(session_id, utterance_id)
            else:
                scheduler.close_session(session_id)
            continue_after_cancel.set()
            with pytest.raises(SchedulerStale, match="stale") as raised:
                await future
            assert await asyncio.to_thread(producer_finalized.wait, 1)
            assert delivered == [marker]
            with pulls_lock:
                pulls_after_cancel = pulls
            await asyncio.sleep(0.05)
            with pulls_lock:
                assert pulls == pulls_after_cancel
            rendered = "".join(
                traceback.format_exception(
                    type(raised.value),
                    raised.value,
                    raised.value.__traceback__,
                )
            )
            assert marker.decode() not in rendered
            assert marker.decode() not in caplog.text
        finally:
            continue_after_cancel.set()
            await scheduler.shutdown()

    run(scenario())


def test_close_session_does_not_invalidate_survivor_session() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        target_session = uuid4()
        survivor_session = uuid4()
        scheduler.open_session(target_session, AudioDirection.MICROPHONE)
        scheduler.open_session(survivor_session, AudioDirection.SPEAKER)
        target_entered = ThreadEvent()
        release_target = ThreadEvent()
        survivor_attempt = asyncio.Event()
        queued_target_ran = False

        def target_native() -> str:
            target_entered.set()
            assert release_target.wait(timeout=2)
            return "target-late"

        async def target_work(context: SchedulerContext) -> str:
            return await context.run_gpu(target_native)

        async def queued_target_work(
            _context: SchedulerContext,
        ) -> None:
            nonlocal queued_target_ran
            queued_target_ran = True

        async def survivor_work(context: SchedulerContext) -> str:
            survivor_attempt.set()
            return await context.run_gpu(lambda: "survivor")

        try:
            target = scheduler.submit(
                scheduler.open_utterance(target_session, uuid4()),
                target_work,
            )
            assert await asyncio.to_thread(target_entered.wait, 1)
            queued_target = scheduler.submit(
                scheduler.open_utterance(target_session, uuid4()),
                queued_target_work,
            )
            survivor = scheduler.submit(
                scheduler.open_utterance(survivor_session, uuid4()),
                survivor_work,
            )
            await asyncio.wait_for(survivor_attempt.wait(), timeout=1)
            scheduler.close_session(target_session)
            with pytest.raises(SchedulerStale, match="stale"):
                await queued_target
            release_target.set()
            with pytest.raises(SchedulerStale, match="stale"):
                await target
            assert await survivor == "survivor"
            assert queued_target_ran is False
        finally:
            release_target.set()
            await scheduler.shutdown()

    run(scenario())


@pytest.mark.parametrize("cancel_kind", ["utterance", "session"])
def test_generation_change_discards_late_native_result(
    cancel_kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        utterance_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.SPEAKER)
        identity = scheduler.open_utterance(session_id, utterance_id)
        entered = ThreadEvent()
        release = ThreadEvent()
        marker = "private late scheduler result marker"
        caplog.set_level(logging.DEBUG)

        def native() -> str:
            entered.set()
            assert release.wait(timeout=2)
            return marker

        async def work(context: SchedulerContext) -> str:
            return await context.run_gpu(native)

        try:
            future = scheduler.submit(identity, work)
            assert await asyncio.to_thread(entered.wait, 1)
            if cancel_kind == "utterance":
                scheduler.cancel_utterance(session_id, utterance_id)
            else:
                scheduler.close_session(session_id)
            release.set()
            with pytest.raises(SchedulerStale, match="stale") as raised:
                await future
            rendered = "".join(
                traceback.format_exception(
                    type(raised.value),
                    raised.value,
                    raised.value.__traceback__,
                )
            )
            assert marker not in rendered
            assert marker not in caplog.text
        finally:
            release.set()
            await scheduler.shutdown()

    run(scenario())


def test_cancel_purges_queued_job_without_running_it() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.MICROPHONE)
        first_started = asyncio.Event()
        release = asyncio.Event()
        queued_ran = False
        survivor_ran = False
        replacement_ran = False

        async def first_work(_context: SchedulerContext) -> None:
            first_started.set()
            await release.wait()

        async def queued_work(_context: SchedulerContext) -> None:
            nonlocal queued_ran
            queued_ran = True

        async def survivor_work(_context: SchedulerContext) -> str:
            nonlocal survivor_ran
            survivor_ran = True
            return "survivor"

        async def replacement_work(_context: SchedulerContext) -> str:
            nonlocal replacement_ran
            replacement_ran = True
            return "replacement"

        try:
            first = scheduler.submit(
                scheduler.open_utterance(session_id, uuid4()),
                first_work,
            )
            await asyncio.wait_for(first_started.wait(), timeout=1)
            queued_id = uuid4()
            queued_identity = scheduler.open_utterance(session_id, queued_id)
            queued = scheduler.submit(
                queued_identity,
                queued_work,
            )
            survivor = scheduler.submit(
                scheduler.open_utterance(session_id, uuid4()),
                survivor_work,
            )
            scheduler.cancel_utterance(session_id, queued_id)
            replacement = scheduler.submit(
                scheduler.open_utterance(session_id, uuid4()),
                replacement_work,
            )
            with pytest.raises(SchedulerStale, match="stale"):
                await queued
            assert queued_ran is False
            release.set()
            await first
            assert await survivor == "survivor"
            assert await replacement == "replacement"
            assert survivor_ran
            assert replacement_ran
            assert scheduler.tracked_utterance_count(session_id) == 0
            with pytest.raises(SchedulerStale, match="stale"):
                scheduler.submit(queued_identity, queued_work)
        finally:
            release.set()
            await scheduler.shutdown()

    run(scenario())


def test_scheduler_sanitizes_tts_producer_failure_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.SPEAKER)
        marker = "private scheduler tts marker"
        caplog.set_level(logging.DEBUG)

        def frames() -> object:
            raise RuntimeError(marker)
            yield b""  # pragma: no cover

        async def work(context: SchedulerContext) -> None:
            async for _frame in context.stream_tts(frames, frame_duration_ms=20):
                pass

        try:
            identity = scheduler.open_utterance(session_id, uuid4())
            future = scheduler.submit(identity, work)
            with pytest.raises(SchedulerUnavailable, match="unavailable") as raised:
                await future
            rendered = "".join(
                traceback.format_exception(
                    type(raised.value),
                    raised.value,
                    raised.value.__traceback__,
                )
            )
            assert marker not in rendered
            assert marker not in caplog.text
            assert scheduler.tracked_utterance_count(session_id) == 0
            with pytest.raises(SchedulerStale, match="stale"):
                scheduler.submit(identity, work)
        finally:
            await scheduler.shutdown()

    run(scenario())


def test_scheduler_sanitizes_native_failure_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.MICROPHONE)
        marker = "private scheduler native marker"
        caplog.set_level(logging.DEBUG)

        def native() -> str:
            raise RuntimeError(marker)

        async def work(context: SchedulerContext) -> str:
            return await context.run_gpu(native)

        try:
            identity = scheduler.open_utterance(session_id, uuid4())
            future = scheduler.submit(identity, work)
            with pytest.raises(SchedulerUnavailable, match="unavailable") as raised:
                await future
            rendered = "".join(
                traceback.format_exception(
                    type(raised.value),
                    raised.value,
                    raised.value.__traceback__,
                )
            )
            assert marker not in rendered
            assert marker not in caplog.text
            assert scheduler.tracked_utterance_count(session_id) == 0
            with pytest.raises(SchedulerStale, match="stale"):
                scheduler.submit(identity, work)
        finally:
            await scheduler.shutdown()

    run(scenario())


def test_completed_identity_cannot_be_submitted_twice() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.MICROPHONE)
        identity = scheduler.open_utterance(session_id, uuid4())

        async def work(_context: SchedulerContext) -> str:
            return "done"

        try:
            assert await scheduler.submit(identity, work) == "done"
            with pytest.raises(SchedulerStale, match="stale"):
                scheduler.submit(identity, work)
        finally:
            await scheduler.shutdown()

    run(scenario())


def test_completed_utterances_do_not_accumulate_session_state() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.SPEAKER)

        async def work(_context: SchedulerContext) -> None:
            return None

        try:
            for _ in range(100):
                identity = scheduler.open_utterance(session_id, uuid4())
                await scheduler.submit(identity, work)
            assert scheduler.tracked_utterance_count(session_id) == 0
        finally:
            await scheduler.shutdown()

    run(scenario())


@pytest.mark.parametrize("outcome", ["failure", "cancel"])
def test_failed_or_cancelled_utterance_is_terminal_and_removed(
    outcome: str,
) -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        utterance_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.MICROPHONE)
        identity = scheduler.open_utterance(session_id, utterance_id)
        started = asyncio.Event()
        release = asyncio.Event()

        async def work(_context: SchedulerContext) -> None:
            started.set()
            if outcome == "cancel":
                await release.wait()
                return
            raise RuntimeError("private failed utterance marker")

        try:
            future = scheduler.submit(identity, work)
            await asyncio.wait_for(started.wait(), timeout=1)
            if outcome == "cancel":
                scheduler.cancel_utterance(session_id, utterance_id)
                release.set()
                expected_error = SchedulerStale
            else:
                expected_error = SchedulerUnavailable
            with pytest.raises(expected_error):
                await future
            assert scheduler.tracked_utterance_count(session_id) == 0
            with pytest.raises(SchedulerStale, match="stale"):
                scheduler.submit(identity, work)
        finally:
            release.set()
            await scheduler.shutdown()

    run(scenario())


def test_closed_sessions_use_global_generation_without_tombstones() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        reused_id = uuid4()
        first_generation = scheduler.open_session(reused_id, AudioDirection.MICROPHONE)
        scheduler.close_session(reused_id)

        for _ in range(100):
            session_id = uuid4()
            scheduler.open_session(session_id, AudioDirection.SPEAKER)
            scheduler.close_session(session_id)

        second_generation = scheduler.open_session(reused_id, AudioDirection.MICROPHONE)
        assert second_generation > first_generation
        assert scheduler.tracked_session_count == 1
        await scheduler.shutdown()
        assert scheduler.tracked_session_count == 0

    run(scenario())


def test_concurrent_cancelled_shutdown_waits_for_shared_cleanup() -> None:
    async def scenario() -> None:
        scheduler = InferenceScheduler()
        session_id = uuid4()
        scheduler.open_session(session_id, AudioDirection.MICROPHONE)
        work_started = asyncio.Event()
        release_work = asyncio.Event()
        work_finished = asyncio.Event()

        async def work(_context: SchedulerContext) -> None:
            work_started.set()
            try:
                await release_work.wait()
            finally:
                work_finished.set()

        scheduler.submit(
            scheduler.open_utterance(session_id, uuid4()),
            work,
        )
        await asyncio.wait_for(work_started.wait(), timeout=1)

        first: asyncio.Task[None] | None = None
        second: asyncio.Task[None] | None = None
        try:
            first = asyncio.create_task(scheduler.shutdown())
            second = asyncio.create_task(scheduler.shutdown())
            await asyncio.sleep(0)
            first.cancel()
            await asyncio.sleep(0)
            assert first.done() is False
            assert second.done() is False
            assert work_finished.is_set() is False

            release_work.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            await asyncio.wait_for(second, timeout=2)
            assert work_finished.is_set()
            assert scheduler.tracked_session_count == 0
        finally:
            release_work.set()
            tasks = [
                task for task in (first, second) if task is not None and not task.done()
            ]
            if tasks:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2,
                )

    run(scenario())


class ShutdownFixture:
    """Observe original executor effects; independently release and join fixtures."""

    def __init__(self):
        self.scheduler = InferenceScheduler()
        self.executors = (
            self.scheduler._gpu_executor,
            self.scheduler._tts_executor,
        )
        self.original_shutdowns = tuple(item.shutdown for item in self.executors)
        self.calls = []
        self.failures = set()
        self.threads = ()
        self.release = ThreadEvent()
        self.entered = ThreadEvent()
        self.native_calls = 0
        self.tasks = []
        self.future = None
        self.session_id = uuid4()
        self.scheduler.open_session(self.session_id, AudioDirection.MICROPHONE)
        self.identity = self.scheduler.open_utterance(self.session_id, uuid4())
        for index, executor in enumerate(self.executors):

            def shutdown(*, wait, cancel_futures, index=index):
                self.calls.append((index, wait, cancel_futures))
                if index in self.failures:
                    raise RuntimeError("private-executor-shutdown-marker")
                self.original_shutdowns[index](wait=wait, cancel_futures=cancel_futures)

            executor.shutdown = shutdown

    async def start_threads(self):
        loop = asyncio.get_running_loop()
        self.threads = tuple(
            await asyncio.gather(
                *(loop.run_in_executor(item, current_thread) for item in self.executors)
            )
        )
        assert len(set(self.threads)) == 2
        assert all(thread.is_alive() for thread in self.threads)

    async def start_held_job(self):
        def native():
            self.native_calls += 1
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("fixture native release timed out")

        async def work(context):
            await context.run_gpu(native)

        self.future = self.scheduler.submit(self.identity, work)
        async with asyncio.timeout(1):
            while not self.entered.is_set():
                await asyncio.sleep(0)
        assert len(self.scheduler._running_tasks) == 1
        return next(iter(self.scheduler._running_tasks))

    def closed_admission(self):
        assert self.scheduler._closed
        with pytest.raises(SchedulerUnavailable):
            self.scheduler.open_session(uuid4(), AudioDirection.SPEAKER)
        with pytest.raises(SchedulerStale):
            self.scheduler.submit(self.identity, None)

    async def cleanup(self):
        self.release.set()
        try:
            owned = set(self.tasks) | set(self.scheduler._running_tasks)
            owned.update(
                task
                for task in (self.scheduler._dispatcher, self.scheduler._shutdown_task)
                if task is not None
            )
            if self.future is not None:
                owned.add(self.future)
            if owned:
                async with asyncio.timeout(2):
                    await asyncio.gather(*owned, return_exceptions=True)
        finally:
            for original in self.original_shutdowns:
                original(wait=True, cancel_futures=True)
            assert all(not thread.is_alive() for thread in self.threads)


def assert_safe_shutdown_error(error):
    assert isinstance(error, SchedulerUnavailable)
    assert str(error) == "scheduler cleanup is unavailable"
    rendered = "".join(traceback.format_exception(error))
    assert "private-executor-shutdown-marker" not in rendered
    assert "private-dispatcher-marker" not in rendered


@pytest.mark.parametrize("failed", [{0}, {1}, {0, 1}], ids=["gpu", "tts", "both"])
def test_shutdown_retries_failed_executors_with_one_shared_attempt(failed):
    async def scenario():
        fixture = ShutdownFixture()
        scheduler = fixture.scheduler
        retry_entered, retry_release = asyncio.Event(), asyncio.Event()
        original_impl = scheduler._shutdown_impl
        attempts = 0

        async def observed_impl():
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                retry_entered.set()
                await retry_release.wait()
            await original_impl()

        scheduler._shutdown_impl = observed_impl
        try:
            await fixture.start_threads()
            fixture.failures = set(failed)
            first_error = (
                await asyncio.gather(scheduler.shutdown(), return_exceptions=True)
            )[0]
            first_task = scheduler._shutdown_task
            fixture.closed_admission()
            assert fixture.calls == [(0, True, True), (1, True, True)]
            assert [thread.is_alive() for thread in fixture.threads] == [
                index in failed for index in range(2)
            ]
            assert_safe_shutdown_error(first_error)
            fixture.failures.clear()
            callers_entered = 0
            both_entered = asyncio.Event()

            async def retry():
                nonlocal callers_entered
                callers_entered += 1
                if callers_entered == 2:
                    both_entered.set()
                await scheduler.shutdown()

            fixture.tasks = [asyncio.create_task(retry()) for _ in range(2)]
            async with asyncio.timeout(1):
                await retry_entered.wait()
                await both_entered.wait()
            shared = scheduler._shutdown_task
            assert shared is not first_task and not shared.done()
            assert attempts == 2 and all(not task.done() for task in fixture.tasks)
            assert fixture.calls == [(0, True, True), (1, True, True)]
            retry_release.set()
            await asyncio.gather(*fixture.tasks)
            assert scheduler._shutdown_task is shared
            assert fixture.calls == [(0, True, True), (1, True, True)] * 2
            assert all(not thread.is_alive() for thread in fixture.threads)
            assert fixture.native_calls == 0
            await scheduler.shutdown()
            assert attempts == 2 and scheduler._shutdown_task is shared
            assert len(fixture.calls) == 4
        finally:
            retry_release.set()
            await fixture.cleanup()

    run(asyncio.wait_for(scenario(), timeout=5))


@pytest.mark.parametrize("cancel_dispatcher", [False, True], ids=["failure", "cancel"])
def test_terminal_dispatcher_is_consumed_after_native_job_before_executor_retirement(
    cancel_dispatcher,
):
    async def scenario():
        fixture = ShutdownFixture()
        scheduler = fixture.scheduler
        original_dispatch = scheduler._dispatch_available
        dispatch_calls = 0

        def fail_after_admission():
            nonlocal dispatch_calls
            dispatch_calls += 1
            original_dispatch()
            if cancel_dispatcher:
                raise asyncio.CancelledError("private-dispatcher-marker")
            raise RuntimeError("private-dispatcher-marker")

        scheduler._dispatch_available = fail_after_admission
        try:
            await fixture.start_threads()
            job = await fixture.start_held_job()
            dispatcher = scheduler._dispatcher
            assert dispatcher.done()
            shutdown = asyncio.create_task(scheduler.shutdown())
            fixture.tasks.append(shutdown)
            async with asyncio.timeout(1):
                while scheduler._sessions:
                    await asyncio.sleep(0)
            fixture.closed_admission()
            assert not shutdown.done() and not job.done() and not fixture.future.done()
            assert scheduler._dispatcher is dispatcher
            assert not fixture.calls and all(
                thread.is_alive() for thread in fixture.threads
            )
            fixture.release.set()
            result = (await asyncio.gather(shutdown, return_exceptions=True))[0]
            assert job.done()
            assert isinstance(fixture.future.exception(), SchedulerStale)
            assert fixture.calls == [(0, True, True), (1, True, True)]
            assert all(not thread.is_alive() for thread in fixture.threads)
            assert scheduler._dispatcher is None
            assert_safe_shutdown_error(result)
            await scheduler.shutdown()
            assert dispatch_calls == 1 and fixture.native_calls == 1
            assert not scheduler._running_tasks and scheduler._dispatcher is None
            assert len(fixture.calls) == 4
            await scheduler.shutdown()
            assert len(fixture.calls) == 4
        finally:
            await fixture.cleanup()

    run(asyncio.wait_for(scenario(), timeout=5))


def test_repeated_shutdown_caller_cancellation_joins_real_native_threads():
    async def scenario():
        fixture = ShutdownFixture()
        scheduler = fixture.scheduler
        try:
            await fixture.start_threads()
            job = await fixture.start_held_job()
            fixture.tasks = [
                asyncio.create_task(scheduler.shutdown()) for _ in range(2)
            ]
            async with asyncio.timeout(1):
                while scheduler._sessions:
                    await asyncio.sleep(0)
            shared = scheduler._shutdown_task
            for _ in range(3):
                fixture.tasks[0].cancel()
                await asyncio.sleep(0)
                assert scheduler._shutdown_task is shared and not shared.done()
                assert all(not task.done() for task in fixture.tasks)
                assert not job.done() and not fixture.future.done()
                assert not fixture.calls
                assert all(thread.is_alive() for thread in fixture.threads)
            fixture.closed_admission()
            fixture.release.set()
            results = await asyncio.gather(*fixture.tasks, return_exceptions=True)
            assert isinstance(results[0], asyncio.CancelledError) and results[1] is None
            assert job.done() and isinstance(fixture.future.exception(), SchedulerStale)
            assert scheduler._shutdown_task is shared and shared.done()
            assert fixture.calls == [(0, True, True), (1, True, True)]
            assert all(not thread.is_alive() for thread in fixture.threads)
            assert fixture.native_calls == 1
            await scheduler.shutdown()
            assert len(fixture.calls) == 2
        finally:
            await fixture.cleanup()

    run(asyncio.wait_for(scenario(), timeout=5))


@pytest.mark.parametrize("allocation", ["raise", "cancel"])
def test_shutdown_allocation_failure_or_prestart_cancel_retains_explicit_retry(
    monkeypatch,
    allocation,
):
    async def scenario():
        fixture = ShutdownFixture()
        scheduler = fixture.scheduler
        real_create = asyncio.create_task
        coroutines, allocated = [], []

        def create(coroutine, **kwargs):
            if coroutine.cr_code is scheduler._shutdown_impl.__func__.__code__:
                coroutines.append(coroutine)
                if allocation == "raise":
                    raise RuntimeError("fixture task allocation failure")
                task = real_create(coroutine, **kwargs)
                allocated.append(task)
                task.cancel()
                return task
            return real_create(coroutine, **kwargs)

        try:
            await fixture.start_threads()
            monkeypatch.setattr(asyncio, "create_task", create)
            try:
                await scheduler.shutdown()
            except BaseException as error:
                outcome = error
            else:
                outcome = None
            monkeypatch.setattr(asyncio, "create_task", real_create)
            fixture.closed_admission()
            assert len(coroutines) == 1
            assert inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED
            assert not fixture.calls and all(
                thread.is_alive() for thread in fixture.threads
            )
            if allocation == "raise":
                assert (
                    isinstance(outcome, RuntimeError)
                    and scheduler._shutdown_task is None
                )
                assert not allocated
            else:
                assert isinstance(outcome, asyncio.CancelledError)
                assert (
                    scheduler._shutdown_task is allocated[0]
                    and allocated[0].cancelled()
                )
            await scheduler.shutdown()
            assert fixture.calls == [(0, True, True), (1, True, True)]
            assert all(not thread.is_alive() for thread in fixture.threads)
            await scheduler.shutdown()
            assert len(fixture.calls) == 2
        finally:
            monkeypatch.setattr(asyncio, "create_task", real_create)
            for coroutine in coroutines:
                if inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED:
                    coroutine.close()
            await fixture.cleanup()

    run(asyncio.wait_for(scenario(), timeout=5))
