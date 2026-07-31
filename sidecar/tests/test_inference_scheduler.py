from __future__ import annotations

import asyncio
import logging
from threading import Event as ThreadEvent
from threading import Lock
import traceback
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

        async def work(
            _context: SchedulerContext, value: int
        ) -> int:
            order.append(value)
            if value == 0:
                started.set()
                await release.wait()
            return value

        try:
            identities = [
                scheduler.open_utterance(session_id, uuid4())
                for _ in range(4)
            ]
            first = scheduler.submit(
                identities[0], lambda context: work(context, 0)
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            second = scheduler.submit(
                identities[1], lambda context: work(context, 1)
            )
            third = scheduler.submit(
                identities[2], lambda context: work(context, 2)
            )
            with pytest.raises(SchedulerOverflow, match="queue"):
                scheduler.submit(
                    identities[3], lambda context: work(context, 3)
                )
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
        sessions = {
            direction: uuid4() for direction in AudioDirection
        }
        for direction, session_id in sessions.items():
            scheduler.open_session(session_id, direction)

        async def work(context: SchedulerContext) -> str:
            started[context.identity.direction].set()
            await release.wait()
            return context.identity.direction.value

        try:
            futures = []
            for direction, session_id in sessions.items():
                for _ in range(3):
                    identity = scheduler.open_utterance(
                        session_id, uuid4()
                    )
                    futures.append(scheduler.submit(identity, work))
            await asyncio.wait_for(
                asyncio.gather(
                    *(event.wait() for event in started.values())
                ),
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
        sessions = {
            direction: uuid4() for direction in AudioDirection
        }
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
            return await context.run_gpu(
                lambda: native(context.identity.direction)
            )

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
                scheduler.open_utterance(
                    sessions[AudioDirection.SPEAKER], uuid4()
                ),
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
        sessions = {
            direction: uuid4() for direction in AudioDirection
        }
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
                    scheduler.open_utterance(
                        sessions[AudioDirection.SPEAKER], uuid4()
                    ),
                    speaker_work,
                ),
            ]
            await asyncio.wait_for(all_attempted.wait(), timeout=1)
            assert await asyncio.to_thread(
                both_entered.wait, 1
            )
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
            stream = context.stream_tts(
                frames, frame_duration_ms=100
            )
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
            stream = context.stream_tts(
                frames, frame_duration_ms=100
            )
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
        scheduler.open_session(
            target_session, AudioDirection.MICROPHONE
        )
        scheduler.open_session(
            survivor_session, AudioDirection.SPEAKER
        )
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
            queued_identity = scheduler.open_utterance(
                session_id, queued_id
            )
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
            async for _frame in context.stream_tts(
                frames, frame_duration_ms=20
            ):
                pass

        try:
            identity = scheduler.open_utterance(session_id, uuid4())
            future = scheduler.submit(identity, work)
            with pytest.raises(
                SchedulerUnavailable, match="unavailable"
            ) as raised:
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
            with pytest.raises(
                SchedulerUnavailable, match="unavailable"
            ) as raised:
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
        first_generation = scheduler.open_session(
            reused_id, AudioDirection.MICROPHONE
        )
        scheduler.close_session(reused_id)

        for _ in range(100):
            session_id = uuid4()
            scheduler.open_session(session_id, AudioDirection.SPEAKER)
            scheduler.close_session(session_id)

        second_generation = scheduler.open_session(
            reused_id, AudioDirection.MICROPHONE
        )
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
                task
                for task in (first, second)
                if task is not None and not task.done()
            ]
            if tasks:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2,
                )

    run(scenario())
