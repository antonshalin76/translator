from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest
from test_provider_contract import VOICE_OVERRIDE_CASES

import translator_sidecar.openai_runtime as openai_runtime
from translator_sidecar.openai_http import OpenAIHttpFailure
from translator_sidecar.openai_runtime import (
    OpenAIProviderProtocolError,
    OpenAIProviderTransportFailure,
    OpenAIRealtimeProvider,
)
from translator_sidecar.provider_contract import (
    MAX_TERMINAL_UTTERANCES_PER_SESSION,
    AudioDirection,
    CancelReason,
    CancelUtterance,
    CloseRequestReason,
    Language,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderId,
    ProviderInputFrame,
    ProviderLatency,
    ProviderSessionClosed,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SampleFormat,
    TranslationMode,
    UpdateDebugText,
    UtteranceOutcome,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.closes = 0
        self.cleanup_fails = False
        self.failure = None
        self.warmup_failure = None
        self.warmup_gate = None
        self.transcribe_gate = None
        self.entered = asyncio.Event()
        self.audio_gate = None
        self.speech_closed = False
        self.chunks = [b"\x01\x00" * 480]

    async def warmup(self):
        self.calls.append(("warmup",))
        self.entered.set()
        if self.warmup_gate is not None:
            await self.warmup_gate.wait()
        if self.warmup_failure:
            raise self.warmup_failure

    async def transcribe(self, pcm, sample_rate, source_language):
        self.calls.append(("asr", pcm, sample_rate, source_language))
        self.entered.set()
        if self.transcribe_gate is not None:
            await self.transcribe_gate.wait()
        if self.failure:
            raise self.failure
        return "source synthetic"

    async def translate(self, text, target_language):
        self.calls.append(("mt", text, target_language))
        return "translated synthetic"

    async def speech(self, text, target_language):
        self.calls.append(("tts", text, target_language))
        try:
            if self.audio_gate is not None:
                await self.audio_gate.wait()
            for chunk in self.chunks:
                yield chunk
                await asyncio.sleep(0)
        finally:
            self.speech_closed = True

    async def close(self):
        self.closes += 1
        if self.cleanup_fails:
            raise OSError("private close failure")
        self.closed = True


def open_request(*, debug=False, rate=16_000):
    pcm = PcmFormat(
        sample_rate_hz=rate,
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
        sample_rate_hz=request.requested_input_format.sample_rate_hz,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=20,
        source_language=request.source_language,
        target_language=request.target_language,
        mode=request.mode,
        pcm=bytes(request.requested_input_format.sample_rate_hz * 2 // 50),
        end_of_utterance=final,
    )


def cancel(value):
    return CancelUtterance(
        session_id=value.session_id,
        direction_id=value.direction_id,
        utterance_id=value.utterance_id,
        reason=CancelReason.USER_INTERRUPT,
    )


class Harness:
    def __init__(self, *, timeout=1):
        self.pipelines = []
        self.events = []
        self.next_pipeline = None
        self.provider = OpenAIRealtimeProvider(
            environ={"OPENAI_API_KEY": "synthetic-secret"},
            pipeline_factory=self.factory,
            drain_timeout_seconds=timeout,
        )

    def factory(self, credential, *, vpn_interface):
        assert credential == "synthetic-secret" and vpn_interface == "amn0"
        pipeline = self.next_pipeline or FakePipeline()
        self.next_pipeline = None
        self.pipelines.append(pipeline)
        return pipeline

    async def publish(self, batch, commit):
        self.events.extend(batch)
        commit()

    async def open(self, *, debug=False, rate=16_000):
        request = open_request(debug=debug, rate=rate)
        await self.provider.reserve_session(request, self.publish).open()
        return request

    async def settled(self):
        async with asyncio.timeout(2):
            while any(
                bool(session.utterances) for session in self.provider._sessions.values()
            ):
                await asyncio.sleep(0)


@pytest.mark.parametrize("debug", [False, True])
def test_http_pipeline_waits_for_eou_and_reuses_warm_session(debug):
    async def scenario():
        h = Harness()
        request = await h.open(debug=debug)
        first = frame(request, final=False)
        await h.provider.submit_frame(first)
        assert h.pipelines[0].calls == [("warmup",)]
        await h.provider.submit_frame(
            first.model_copy(update={"sequence": 1, "end_of_utterance": True})
        )
        await h.settled()
        assert [call[0] for call in h.pipelines[0].calls] == [
            "warmup",
            "asr",
            "mt",
            "tts",
        ]
        assert h.pipelines[0].calls[1][1] == first.pcm * 2
        assert [type(e) for e in h.events] == (
            [ProviderTranscriptDelta, ProviderTranslationDelta] if debug else []
        ) + [ProviderAudioDelta, ProviderUtteranceFinal]
        assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        await h.provider.submit_frame(frame(request, sequence=2))
        await h.settled()
        assert len(h.pipelines) == 1 and not h.pipelines[0].closed
        assert [event.event_sequence for event in h.events] == list(
            range(3, 3 + len(h.events))
        )
        await h.provider.shutdown()
        assert h.pipelines[0].closed and not h.provider._sessions

    asyncio.run(scenario())


@pytest.mark.parametrize("rate", [16_000, 24_000, 48_000])
@pytest.mark.parametrize("chunk_size", [1, 2, 319, 640, 959, 960])
def test_stateful_resampling_and_framing_are_independent_of_http_fragmentation(
    rate, chunk_size
):
    async def scenario():
        payload = np.arange(-1900, 1901, dtype="<i2").tobytes()

        async def render(chunks):
            h = Harness()
            request = await h.open(rate=rate)
            h.pipelines[0].chunks = chunks
            await h.provider.submit_frame(frame(request))
            await h.settled()
            result = b"".join(
                e.pcm for e in h.events if isinstance(e, ProviderAudioDelta)
            )
            assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
            assert all(
                len(e.pcm) == rate * 2 // 50
                for e in h.events
                if isinstance(e, ProviderAudioDelta)
            )
            await h.provider.shutdown()
            return result

        expected = await render([payload])
        actual = await render(
            [payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)]
        )
        assert actual == expected

    asyncio.run(scenario())


@pytest.mark.parametrize("chunks", [[], [b""], [b"x"], [b"ab", b"x"]])
def test_empty_or_incomplete_pcm_never_completes(chunks):
    async def scenario():
        h = Harness()
        request = await h.open()
        h.pipelines[0].chunks = chunks
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        assert isinstance(h.events[-2], PrivacySafeProviderError)
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["asr", "tts", "collect"])
def test_cancel_revokes_before_join_and_next_phrase_has_new_identity(phase):
    async def scenario():
        h = Harness()
        request = await h.open(debug=True)
        pipeline = h.pipelines[0]
        if phase == "asr":
            pipeline.transcribe_gate = asyncio.Event()
        if phase == "tts":
            pipeline.audio_gate = asyncio.Event()
        first = frame(request, final=phase != "collect")
        await h.provider.submit_frame(first)
        await asyncio.sleep(0)
        await h.provider.cancel_utterance(cancel(first))
        count = len(h.events)
        await asyncio.sleep(0)
        assert len(h.events) == count
        assert h.events[-1].outcome is UtteranceOutcome.CANCELLED
        assert h.events[-1].utterance_id == first.utterance_id
        pipeline.transcribe_gate = pipeline.audio_gate = None
        second = frame(request, sequence=1)
        await h.provider.submit_frame(second)
        await h.settled()
        assert h.events[-1].utterance_id == second.utterance_id
        assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_explicit_cancel_retires_first_request_without_blocking_queued_phrase():
    async def scenario():
        h = Harness()
        request = await h.open()
        gate = asyncio.Event()
        h.pipelines[0].transcribe_gate = gate
        first, second = frame(request), frame(request, sequence=1)
        await h.provider.submit_frame(first)
        await asyncio.sleep(0)
        await h.provider.submit_frame(second)
        h.pipelines[0].transcribe_gate = None
        await asyncio.wait_for(h.provider.cancel_utterance(cancel(first)), timeout=0.25)
        await h.settled()
        assert [
            (event.utterance_id, event.outcome)
            for event in h.events
            if isinstance(event, ProviderUtteranceFinal)
        ] == [
            (first.utterance_id, UtteranceOutcome.CANCELLED),
            (second.utterance_id, UtteranceOutcome.COMPLETED),
        ]
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "retryable,code,quota",
    [
        (True, SafeErrorCode.PROVIDER_UNAVAILABLE, False),
        (False, SafeErrorCode.PROVIDER_AUTH_FAILED, False),
        (False, SafeErrorCode.PROVIDER_UNAVAILABLE, True),
    ],
)
def test_safe_http_failure_preserves_policy_and_never_retries_phrase(
    retryable, code, quota
):
    async def scenario():
        h = Harness()
        request = await h.open()
        pipeline = h.pipelines[0]
        pipeline.failure = OpenAIHttpFailure(
            code, retryable=retryable, resource_exhausted=quota
        )
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert [type(e) for e in h.events] == [
            PrivacySafeProviderError,
            ProviderUtteranceFinal,
        ]
        assert h.events[0].code is code and h.events[0].retryable == retryable
        assert len([call for call in pipeline.calls if call[0] == "asr"]) == 1
        if retryable:
            pipeline.failure = None
            await h.provider.submit_frame(frame(request, sequence=1))
            await h.settled()
            assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        else:
            with pytest.raises(OpenAIProviderProtocolError):
                await h.provider.submit_frame(frame(request, sequence=1))
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_watchdog_drops_one_phrase_and_next_phrase_recovers():
    async def scenario():
        h = Harness(timeout=0.01)
        request = await h.open()
        h.pipelines[0].transcribe_gate = asyncio.Event()
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        h.pipelines[0].transcribe_gate = None
        await h.provider.submit_frame(frame(request, sequence=1))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_continuous_playable_audio_can_outlive_first_response_deadline():
    async def scenario():
        h = Harness(timeout=0.05)
        request = await h.open()

        async def continuous_speech(_text, _language):
            for _ in range(12):
                yield b"\x01\x00" * 480
                await asyncio.sleep(0.01)

        h.pipelines[0].speech = continuous_speech
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        assert sum(isinstance(event, ProviderAudioDelta) for event in h.events) == 12
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_output_progress_stall_still_times_out_and_closes_stream():
    async def scenario():
        h = Harness(timeout=0.02)
        request = await h.open(rate=24_000)
        closed = asyncio.Event()

        async def stalled_speech(_text, _language):
            try:
                yield b"\x01\x00" * 480
                await asyncio.Event().wait()
            finally:
                closed.set()

        h.pipelines[0].speech = stalled_speech
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        assert h.events[-1].final_audio_sequence == 0 and closed.is_set()
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_text_and_partial_sample_trickle_cannot_extend_first_playable_deadline():
    async def scenario():
        h = Harness(timeout=0.02)
        request = await h.open(debug=True)

        async def trickle_speech(_text, _language):
            for _ in range(100):
                yield b"\x01\x00"
                await asyncio.sleep(0.003)

        h.pipelines[0].speech = trickle_speech
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        assert not any(isinstance(event, ProviderAudioDelta) for event in h.events)
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_continuous_output_has_an_absolute_bound_after_first_frame(monkeypatch):
    monkeypatch.setattr(openai_runtime, "_MAX_OUTPUT_SECONDS", 0.01)

    async def scenario():
        h = Harness(timeout=0.03)
        request = await h.open()

        async def continuous_speech(_text, _language):
            for _ in range(100):
                yield b"\x01\x00" * 480
                await asyncio.sleep(0.005)

        h.pipelines[0].speech = continuous_speech
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        assert (
            1 < sum(isinstance(event, ProviderAudioDelta) for event in h.events) < 100
        )
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_first_deadline_is_captured_at_eou_not_when_delayed_task_starts():
    async def scenario():
        h = Harness(timeout=0.01)
        request = await h.open()
        original = h.provider._process

        async def delayed_process(session, generation):
            await asyncio.sleep(0.02)
            await original(session, generation)

        h.provider._process = delayed_process
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        assert h.pipelines[0].calls == [("warmup",)]
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_open", [False, True])
def test_failed_open_cleanup_custody_is_retained_and_blocks_unbounded_new_owners(
    cancel_open,
):
    async def scenario():
        h = Harness()
        bad = FakePipeline()
        bad.cleanup_fails = True
        if cancel_open:
            bad.warmup_gate = asyncio.Event()
        else:
            bad.warmup_failure = OSError("private opening failure")
        h.next_pipeline = bad
        request = open_request()
        reservation = h.provider.reserve_session(request, h.publish)
        opening = asyncio.create_task(reservation.open())
        await bad.entered.wait()
        if cancel_open:
            opening.cancel()
        await asyncio.gather(opening, return_exceptions=True)
        assert request.session_id in h.provider._failed_opens
        owner = h.provider._failed_opens[request.session_id]
        for _ in range(100):
            with pytest.raises(
                OpenAIProviderProtocolError, match="provider_cleanup_required"
            ):
                h.provider.reserve_session(open_request(), h.publish)
        assert (
            h.provider._failed_opens[request.session_id] is owner
            and len(h.pipelines) == 1
        )
        with pytest.raises(OpenAIProviderProtocolError):
            await h.provider.shutdown()
        bad.cleanup_fails = False
        await h.provider.shutdown()
        assert bad.closed and not h.provider._failed_opens and not h.provider._opening
        assert h.events == []

    asyncio.run(scenario())


def test_shutdown_joins_opening_without_publishing_ready():
    async def scenario():
        h = Harness()
        pending = FakePipeline()
        pending.warmup_gate = asyncio.Event()
        h.next_pipeline = pending
        opening = asyncio.create_task(
            h.provider.reserve_session(open_request(), h.publish).open()
        )
        await pending.entered.wait()
        await asyncio.wait_for(h.provider.shutdown(), timeout=0.25)
        assert opening.done() and pending.closed
        assert (
            not h.provider._sessions and not h.provider._failed_opens and not h.events
        )

    asyncio.run(scenario())


def test_debug_disable_is_a_publication_barrier_not_new_network_session():
    async def scenario():
        h = Harness()
        request = await h.open(debug=True)
        pipeline = h.pipelines[0]
        pipeline.transcribe_gate = asyncio.Event()
        await h.provider.submit_frame(frame(request))
        await asyncio.sleep(0)
        await h.provider.update_debug_text(
            UpdateDebugText(session_id=request.session_id, enabled=False)
        )
        pipeline.transcribe_gate.set()
        await h.settled()
        assert not any(
            isinstance(e, (ProviderTranscriptDelta, ProviderTranslationDelta))
            for e in h.events
        )
        assert len(h.pipelines) == 1
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_noncommitting_publication_is_failed_delivery_not_successful_drain():
    async def scenario():
        h = Harness()
        request = await h.open()
        session = h.provider._sessions[request.session_id]

        async def uncommitted(batch, commit):
            pass

        session.publish = uncommitted
        await h.provider.submit_frame(frame(request))
        async with asyncio.timeout(1):
            while any(
                item.task is not None and not item.task.done()
                for item in session.utterances.values()
            ):
                await asyncio.sleep(0)
        receipt = await session.drain(CloseRequestReason.USER_STOP)
        assert receipt.delivery_error is SafeErrorCode.PROVIDER_UNAVAILABLE
        assert h.pipelines[0].closed and not h.provider._sessions

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "updates",
    [
        {"sequence": 0},
        {"stream_id": uuid4()},
        {"source_language": Language.EN},
        {"pcm": b"short"},
    ],
)
def test_rejected_frames_leave_active_collector_unchanged(updates):
    async def scenario():
        h = Harness()
        request = await h.open()
        first = frame(request, final=False)
        await h.provider.submit_frame(first)
        session = h.provider._sessions[request.session_id]
        before = (
            session.utterances[first.utterance_id],
            bytes(session.utterances[first.utterance_id].pcm),
            session.last_input_sequence,
        )
        with pytest.raises(OpenAIProviderProtocolError):
            await h.provider.submit_frame(
                first.model_copy(update={"sequence": 1, **updates})
            )
        assert (
            session.utterances[first.utterance_id],
            bytes(session.utterances[first.utterance_id].pcm),
            session.last_input_sequence,
        ) == before
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_thousand_generations_have_bounded_terminal_ids_and_no_tasks_or_replay():
    async def scenario():
        h = Harness()
        request = await h.open()
        first = None
        for sequence in range(1000):
            value = frame(request, sequence=sequence)
            first = first or value
            await h.provider.submit_frame(value)
            await h.settled()
        assert len(h.pipelines) == 1
        session = h.provider._sessions[request.session_id]
        assert len(session.terminal_ids) == 1000 and not session.utterances
        with pytest.raises(
            OpenAIProviderProtocolError, match="utterance_already_terminal"
        ):
            await h.provider.submit_frame(first.model_copy(update={"sequence": 1000}))
        session.terminal_ids.update(
            uuid4() for _ in range(MAX_TERMINAL_UTTERANCES_PER_SESSION - 1000)
        )
        with pytest.raises(
            OpenAIProviderProtocolError, match="terminal_identity_capacity_exhausted"
        ):
            await h.provider.submit_frame(frame(request, sequence=1000))
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("model_path,provider_voice_id", VOICE_OVERRIDE_CASES)
def test_voice_override_is_rejected_before_any_resource(model_path, provider_voice_id):
    h = Harness()
    request = open_request()
    request = request.model_copy(
        update={
            "voice_profile": request.voice_profile.model_copy(
                update={
                    "model_path": model_path,
                    "provider_voice_id": provider_voice_id,
                }
            )
        }
    )
    with pytest.raises(OpenAIProviderProtocolError, match="voice profile override"):
        h.provider.reserve_session(request, h.publish)
    assert not h.pipelines and not h.provider._sessions


def test_real_bidirectional_sessions_have_independent_requests_and_pipeline_tasks():
    async def scenario():
        h = Harness()
        outgoing = await h.open()
        incoming = open_request().model_copy(
            update={
                "direction_id": AudioDirection.SPEAKER,
                "source_language": Language.EN,
                "target_language": Language.RU,
            }
        )
        await h.provider.reserve_session(incoming, h.publish).open()
        for pipeline in h.pipelines:
            pipeline.transcribe_gate = asyncio.Event()
        await h.provider.submit_frame(frame(outgoing))
        await h.provider.submit_frame(frame(incoming))
        await asyncio.sleep(0)
        assert all(
            any(call[0] == "asr" for call in pipeline.calls) for pipeline in h.pipelines
        )
        await h.provider.cancel_utterance(
            cancel(
                frame(outgoing).model_copy(
                    update={
                        "utterance_id": next(
                            iter(h.provider._sessions[outgoing.session_id].utterances)
                        )
                    }
                )
            )
        )
        h.pipelines[1].transcribe_gate.set()
        await h.settled()
        assert any(
            isinstance(e, ProviderUtteranceFinal)
            and e.session_id == incoming.session_id
            and e.outcome is UtteranceOutcome.COMPLETED
            for e in h.events
        )
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_cancel_after_audio_commit_keeps_final_sequence_without_delivery_failure():
    async def scenario():
        h = Harness()
        request = await h.open()
        session = h.provider._sessions[request.session_id]
        entered = asyncio.Event()

        async def delayed_publication(batch, commit):
            h.events.extend(batch)
            commit()
            if any(isinstance(event, ProviderAudioDelta) for event in batch):
                entered.set()
                await asyncio.Event().wait()

        session.publish = delayed_publication
        value = frame(request)
        await h.provider.submit_frame(value)
        await entered.wait()
        await asyncio.wait_for(h.provider.cancel_utterance(cancel(value)), timeout=0.25)
        assert h.events[-1].outcome is UtteranceOutcome.CANCELLED
        assert h.events[-1].final_audio_sequence == 0
        assert session.delivery_error is None and not session.failed
        assert h.pipelines[0].speech_closed
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_input_twelve_second_cap_discards_only_overflowing_utterance():
    async def scenario():
        h = Harness()
        request = await h.open()
        value = frame(request, final=False)
        for sequence in range(600):
            await h.provider.submit_frame(
                value.model_copy(update={"sequence": sequence})
            )
        session = h.provider._sessions[request.session_id]
        assert len(session.utterances[value.utterance_id].pcm) == 16_000 * 2 * 12
        await h.provider.submit_frame(value.model_copy(update={"sequence": 600}))
        assert session.utterances[value.utterance_id].discarding
        assert not session.utterances[value.utterance_id].pcm
        assert not session.failed
        await h.provider.submit_frame(
            value.model_copy(update={"sequence": 601, "end_of_utterance": True})
        )
        assert (
            h.events[-2].code is SafeErrorCode.QUEUE_OVERFLOW and h.events[-2].retryable
        )
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        assert h.pipelines[0].calls == [("warmup",)]
        await h.provider.submit_frame(frame(request, sequence=602))
        await h.settled()
        assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_stale_reservation_and_duplicate_open_cannot_take_reused_uuid_authority():
    async def scenario():
        h = Harness()
        request = open_request()
        first = h.provider.reserve_session(request, h.publish)
        with pytest.raises(OpenAIProviderProtocolError, match="duplicate_session"):
            h.provider.reserve_session(request, h.publish)
        await first.open()
        await first.drain(CloseRequestReason.USER_STOP)
        second = h.provider.reserve_session(request, h.publish)
        with pytest.raises(OpenAIProviderProtocolError, match="open_session_required"):
            await first.open()
        await first.drain(CloseRequestReason.USER_STOP)
        assert h.provider._sessions[request.session_id] is second
        await second.open()
        await h.provider.shutdown()
        assert all(pipeline.closed for pipeline in h.pipelines)

    asyncio.run(scenario())


def test_repeated_cancellation_joins_one_drain_before_releasing_pipeline():
    async def scenario():
        h = Harness()
        request = await h.open()
        session = h.provider._sessions[request.session_id]
        pipeline = h.pipelines[0]
        entered, release = asyncio.Event(), asyncio.Event()
        original_close = pipeline.close

        async def delayed_close():
            entered.set()
            await release.wait()
            await original_close()

        pipeline.close = delayed_close
        first = asyncio.create_task(session.drain(CloseRequestReason.USER_STOP))
        await entered.wait()
        first.cancel()
        first.cancel()
        second = asyncio.create_task(session.drain(CloseRequestReason.USER_STOP))
        await asyncio.sleep(0)
        assert not first.done() and not second.done() and not pipeline.closed
        release.set()
        result = await asyncio.gather(first, second, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert result[1].delivery_error is None
        assert pipeline.closes == 1 and pipeline.closed
        assert sum(isinstance(event, ProviderSessionClosed) for event in h.events) == 1
        assert not h.provider._sessions

    asyncio.run(scenario())


def test_failed_open_transport_identity_and_hundred_resource_releases():
    async def scenario():
        h = Harness()
        for _ in range(100):
            pipeline = FakePipeline()
            pipeline.warmup_failure = OpenAIHttpFailure(retryable=True, transport=True)
            h.next_pipeline = pipeline
            with pytest.raises(OpenAIProviderTransportFailure) as error:
                await h.provider.reserve_session(open_request(), h.publish).open()
            assert error.value.retryable and not error.value.resource_exhausted
            assert pipeline.closed
            assert (
                not h.provider._sessions
                and not h.provider._failed_opens
                and not h.provider._opening
            )
        assert h.events == []
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("raise_error", [False, True])
def test_failed_publication_stays_fatal_after_terminal_delivery_recovers(
    debug, raise_error
):
    async def scenario():
        h = Harness()
        request = await h.open(debug=debug)
        session = h.provider._sessions[request.session_id]
        failed_once = False

        async def fail_once(batch, commit):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                if raise_error:
                    raise RuntimeError("synthetic delivery failure")
                return
            h.events.extend(batch)
            commit()

        session.publish = fail_once
        await h.provider.submit_frame(frame(request))
        await h.settled()
        assert [type(event) for event in h.events] == [
            PrivacySafeProviderError,
            ProviderUtteranceFinal,
        ]
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        assert session.failed
        with pytest.raises(OpenAIProviderProtocolError, match="open_session_required"):
            await h.provider.submit_frame(frame(request, sequence=1))
        receipt = await session.drain(CloseRequestReason.USER_STOP)
        assert receipt.delivery_error is SafeErrorCode.PROVIDER_UNAVAILABLE
        assert h.pipelines[0].closed and not h.provider._sessions

    asyncio.run(scenario())


def test_eof_racing_external_retirement_cannot_join_its_own_disposer():
    async def scenario():
        h = Harness()
        request = await h.open(rate=24_000)
        session = h.provider._sessions[request.session_id]
        eof_waiting, allow_eof, disposer_waiting, allow_dispose = (
            asyncio.Event() for _ in range(4)
        )

        async def gated_speech(_text, _language):
            yield b"\x01\x00" * 480
            eof_waiting.set()
            await allow_eof.wait()

        original_dispose = h.provider._dispose_generation

        async def gated_dispose(*args):
            disposer_waiting.set()
            await allow_dispose.wait()
            await original_dispose(*args)

        h.pipelines[0].speech = gated_speech
        h.provider._dispose_generation = gated_dispose
        value = frame(request)
        await h.provider.submit_frame(value)
        generation = session.utterances[value.utterance_id]
        await eof_waiting.wait()
        cancelling = asyncio.create_task(h.provider.cancel_utterance(cancel(value)))
        try:
            await disposer_waiting.wait()
            assert generation.revoked
            allow_eof.set()
            await asyncio.sleep(0)
            allow_dispose.set()
            done, _ = await asyncio.wait((cancelling,), timeout=0.2)
            assert done, (
                "EOF joined the disposer that was waiting for EOF task cancellation"
            )
            await cancelling
            assert generation.task.done() and generation.retirement.done()
            assert not session.failed and not session.utterances
            finals = [
                event for event in h.events if isinstance(event, ProviderUtteranceFinal)
            ]
            assert len(finals) == 1 and finals[0].outcome is UtteranceOutcome.CANCELLED
        finally:
            allow_eof.set()
            allow_dispose.set()
            await asyncio.gather(cancelling, return_exceptions=True)
            await h.provider.shutdown()

    asyncio.run(scenario())


def test_consecutive_eou_utterances_both_complete_in_order_without_supersession():
    async def scenario():
        h = Harness()
        request = await h.open()
        gate = asyncio.Event()
        h.pipelines[0].transcribe_gate = gate
        first, second = frame(request), frame(request, sequence=1)
        try:
            await h.provider.submit_frame(first)
            await asyncio.sleep(0)
            await h.provider.submit_frame(second)
            gate.set()
            await h.settled()
            assert [
                (event.utterance_id, event.outcome)
                for event in h.events
                if isinstance(event, ProviderUtteranceFinal)
            ] == [
                (first.utterance_id, UtteranceOutcome.COMPLETED),
                (second.utterance_id, UtteranceOutcome.COMPLETED),
            ]
        finally:
            gate.set()
            await h.provider.shutdown()

    asyncio.run(scenario())


def test_changed_utterance_id_before_eou_is_rejected_without_interrupting_collector():
    async def scenario():
        h = Harness()
        request = await h.open()
        first = frame(request, final=False)
        try:
            await h.provider.submit_frame(first)
            with pytest.raises(
                OpenAIProviderProtocolError, match="input_utterance_mismatch"
            ):
                await h.provider.submit_frame(frame(request, sequence=1))
            await h.provider.submit_frame(
                first.model_copy(update={"sequence": 1, "end_of_utterance": True})
            )
            await h.settled()
            assert h.events[-1].utterance_id == first.utterance_id
            assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
        finally:
            await h.provider.shutdown()

    asyncio.run(scenario())


def test_no_speech_is_utterance_local_and_next_speech_uses_same_session():
    async def scenario():
        h = Harness()
        request = await h.open()
        h.pipelines[0].failure = OpenAIHttpFailure(
            SafeErrorCode.NO_SPEECH, retryable=False
        )
        try:
            h.provider._now_ns = lambda: 133_000_000
            value = frame(request, final=False).model_copy(
                update={"capture_monotonic_ns": 10_000_000}
            )
            await h.provider.submit_frame(value)
            await h.provider.submit_frame(
                value.model_copy(
                    update={
                        "sequence": 1,
                        "capture_monotonic_ns": 30_000_000,
                        "end_of_utterance": True,
                    }
                )
            )
            await h.settled()
            assert [type(event) for event in h.events] == [
                ProviderLatency,
                PrivacySafeProviderError,
                ProviderUtteranceFinal,
            ]
            assert h.events[0].asr_first_text_ms == h.events[0].asr_final_text_ms == 123
            assert (
                h.events[0].mt_first_text_ms is None
                and h.events[0].tts_first_audio_ms is None
            )
            assert h.events[1].code is SafeErrorCode.NO_SPEECH
            assert h.events[-1].outcome is UtteranceOutcome.DROPPED
            assert [call[0] for call in h.pipelines[0].calls] == ["warmup", "asr"]
            h.pipelines[0].failure = None
            await h.provider.submit_frame(frame(request, sequence=2))
            await h.settled()
            assert h.events[-1].outcome is UtteranceOutcome.COMPLETED
            assert len(h.pipelines) == 1
        finally:
            await h.provider.shutdown()

    asyncio.run(scenario())


def test_sustained_batches_of_consecutive_phrases_are_fifo_without_starvation():
    async def scenario():
        h = Harness()
        request = await h.open()
        expected = []
        for batch in range(4):
            gate = asyncio.Event()
            h.pipelines[0].transcribe_gate = gate
            for index in range(3):
                value = frame(request, sequence=batch * 3 + index)
                expected.append(value.utterance_id)
                await h.provider.submit_frame(value)
            gate.set()
            await h.settled()
        finals = [
            event for event in h.events if isinstance(event, ProviderUtteranceFinal)
        ]
        assert [event.utterance_id for event in finals] == expected
        assert all(event.outcome is UtteranceOutcome.COMPLETED for event in finals)
        assert len(h.pipelines) == 1
        await h.provider.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("selected", [0, 1, 2], ids=["active", "queued", "collector"])
def test_cancel_owns_only_selected_active_queued_or_collecting_utterance(selected):
    async def scenario():
        h = Harness()
        request = await h.open()
        gate = asyncio.Event()
        h.pipelines[0].transcribe_gate = gate
        values = [
            frame(request, sequence=index, final=index != 2) for index in range(3)
        ]
        for value in values:
            await h.provider.submit_frame(value)
        session = h.provider._sessions[request.session_id]
        owned = session.utterances[values[selected].utterance_id]
        await h.provider.cancel_utterance(cancel(values[selected]))
        assert owned.revoked and not owned.pcm
        assert owned.task is None or owned.task.done()
        assert values[selected].utterance_id not in session.utterances
        if selected != 2:
            await h.provider.submit_frame(
                values[2].model_copy(update={"sequence": 3, "end_of_utterance": True})
            )
        gate.set()
        await h.settled()
        finals = [
            event for event in h.events if isinstance(event, ProviderUtteranceFinal)
        ]
        assert len(finals) == 3
        assert {event.utterance_id: event.outcome for event in finals} == {
            value.utterance_id: UtteranceOutcome.CANCELLED
            if index == selected
            else UtteranceOutcome.COMPLETED
            for index, value in enumerate(values)
        }
        assert not session.failed
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_drain_revokes_every_task_and_collector_before_releasing_pipeline():
    async def scenario():
        h = Harness()
        request = await h.open()
        h.pipelines[0].transcribe_gate = asyncio.Event()
        values = [
            frame(request, sequence=index, final=index != 2) for index in range(3)
        ]
        for value in values:
            await h.provider.submit_frame(value)
        session = h.provider._sessions[request.session_id]
        owned = tuple(session.utterances.values())
        receipt = await asyncio.wait_for(
            session.drain(CloseRequestReason.USER_STOP), timeout=0.25
        )
        assert receipt.delivery_error is None
        assert not session.utterances and session.collecting_id is None
        assert all(
            item.revoked and not item.pcm and (item.task is None or item.task.done())
            for item in owned
        )
        finals = [
            event for event in h.events if isinstance(event, ProviderUtteranceFinal)
        ]
        assert len(finals) == 3
        assert all(event.outcome is UtteranceOutcome.CANCELLED for event in finals)
        assert h.pipelines[0].closed and h.pipelines[0].closes == 1
        assert not h.provider._sessions

    asyncio.run(scenario())


def test_bounded_ready_queue_drops_only_overflow_and_preserves_duplex_peer():
    async def scenario():
        h = Harness()
        outgoing = await h.open()
        incoming = open_request().model_copy(
            update={
                "direction_id": AudioDirection.SPEAKER,
                "source_language": Language.EN,
                "target_language": Language.RU,
            }
        )
        await h.provider.reserve_session(incoming, h.publish).open()
        gate = asyncio.Event()
        h.pipelines[0].transcribe_gate = gate
        values = [frame(outgoing, sequence=index) for index in range(4)]
        for value in values:
            await h.provider.submit_frame(value)
        session = h.provider._sessions[outgoing.session_id]
        assert len(session.utterances) == 3 and not session.failed
        assert (
            h.events[-2].code is SafeErrorCode.QUEUE_OVERFLOW and h.events[-2].retryable
        )
        assert h.events[-1].utterance_id == values[3].utterance_id
        assert h.events[-1].outcome is UtteranceOutcome.DROPPED
        await h.provider.submit_frame(frame(incoming))
        async with asyncio.timeout(0.25):
            while h.provider._sessions[incoming.session_id].utterances:
                await asyncio.sleep(0)
        assert not session.failed
        assert any(
            isinstance(event, ProviderUtteranceFinal)
            and event.session_id == incoming.session_id
            and event.outcome is UtteranceOutcome.COMPLETED
            for event in h.events
        )
        gate.set()
        await h.settled()
        completed = [
            event.utterance_id
            for event in h.events
            if isinstance(event, ProviderUtteranceFinal)
            and event.session_id == outgoing.session_id
            and event.outcome is UtteranceOutcome.COMPLETED
        ]
        assert completed == [value.utterance_id for value in values[:3]]
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_waiting_for_processing_lock_is_charged_to_original_eou_deadline():
    async def scenario():
        h = Harness(timeout=0.03)
        request = await h.open(rate=24_000)
        started = asyncio.Event()

        async def continuous_speech(_text, _language):
            for _ in range(20):
                yield b"\x01\x00" * 480
                started.set()
                await asyncio.sleep(0.005)

        h.pipelines[0].speech = continuous_speech
        first, second = frame(request), frame(request, sequence=1)
        await h.provider.submit_frame(first)
        await started.wait()
        await h.provider.submit_frame(second)
        await h.settled()
        outcomes = {
            event.utterance_id: event.outcome
            for event in h.events
            if isinstance(event, ProviderUtteranceFinal)
        }
        assert outcomes == {
            first.utterance_id: UtteranceOutcome.COMPLETED,
            second.utterance_id: UtteranceOutcome.DROPPED,
        }
        assert len([call for call in h.pipelines[0].calls if call[0] == "asr"]) == 1
        await h.provider.shutdown()

    asyncio.run(scenario())


def test_failed_close_keeps_exact_session_custody_after_every_queued_task_is_drained():
    async def scenario():
        h = Harness()
        request = await h.open()
        pipeline = h.pipelines[0]
        pipeline.transcribe_gate = asyncio.Event()
        pipeline.cleanup_fails = True
        for index in range(3):
            await h.provider.submit_frame(
                frame(request, sequence=index, final=index != 2)
            )
        session = h.provider._sessions[request.session_id]
        owned = tuple(session.utterances.values())
        with pytest.raises(
            OpenAIProviderProtocolError, match="openai_retirement_failed"
        ):
            await session.drain(CloseRequestReason.USER_STOP)
        assert (
            h.provider._sessions[request.session_id] is session
            and session.pipeline is pipeline
        )
        assert not pipeline.closed and not session.utterances
        assert all(
            item.revoked and not item.pcm and (item.task is None or item.task.done())
            for item in owned
        )
        with pytest.raises(
            OpenAIProviderProtocolError, match="provider_cleanup_required"
        ):
            h.provider.reserve_session(open_request(), h.publish)
        count = len(h.events)
        pipeline.cleanup_fails = False
        await session.drain(CloseRequestReason.USER_STOP)
        assert pipeline.closed and not h.provider._sessions
        assert len(h.events) == count + 1 and isinstance(
            h.events[-1], ProviderSessionClosed
        )

    asyncio.run(scenario())
