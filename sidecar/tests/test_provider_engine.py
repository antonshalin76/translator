from uuid import UUID, uuid4

import pytest

from translator_sidecar.provider_contract import (
    AudioDirection,
    CancelReason,
    CancelUtterance,
    CloseProviderSession,
    CloseRequestReason,
    Language,
    OpenProviderSession,
    PcmFormat,
    ProviderId,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderHealth,
    ProviderInputFrame,
    ProviderLatency,
    ProviderState,
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
from translator_sidecar.provider_engine import (
    MockInjection,
    ProviderEngine,
    ProviderProtocolError,
    mock_transform_pcm,
)


def open_request(
    direction: AudioDirection,
    *,
    mode: TranslationMode = TranslationMode.QUALITY_FIRST,
    debug_text_enabled: bool = False,
    frame_duration_ms: int = 100,
) -> OpenProviderSession:
    source, target = (
        (Language.RU, Language.EN)
        if direction is AudioDirection.MICROPHONE
        else (Language.EN, Language.RU)
    )
    pcm_format = PcmFormat(
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=frame_duration_ms,
    )
    return OpenProviderSession(
        session_id=uuid4(),
        provider_id=ProviderId.LOCAL,
        direction_id=direction,
        source_language=source,
        target_language=target,
        mode=mode,
        requested_input_format=pcm_format,
        requested_output_format=pcm_format,
        voice_profile=VoiceProfile(
            language=target,
            gender=VoiceGender.MALE,
            engine=VoiceEngine.PIPER,
        ),
        debug_text_enabled=debug_text_enabled,
    )


def frame(
    session: OpenProviderSession,
    *,
    sequence: int,
    capture_monotonic_ns: int,
    utterance_id: UUID | None = None,
    end_of_utterance: bool = True,
) -> ProviderInputFrame:
    pcm_format = session.requested_input_format
    sample_count = (
        pcm_format.sample_rate_hz
        * pcm_format.channels
        * pcm_format.frame_duration_ms
        // 1000
    )
    return ProviderInputFrame(
        session_id=session.session_id,
        direction_id=session.direction_id,
        stream_id=UUID(int=1 if session.direction_id is AudioDirection.MICROPHONE else 2),
        utterance_id=utterance_id or uuid4(),
        sequence=sequence,
        capture_monotonic_ns=capture_monotonic_ns,
        sample_rate_hz=pcm_format.sample_rate_hz,
        channels=pcm_format.channels,
        sample_format=pcm_format.sample_format,
        frame_duration_ms=pcm_format.frame_duration_ms,
        source_language=session.source_language,
        target_language=session.target_language,
        mode=session.mode,
        end_of_utterance=end_of_utterance,
        pcm=b"\x01\x02" * sample_count,
    )


def test_two_sessions_keep_direction_queues_and_event_sequences_independent() -> None:
    engine = ProviderEngine()
    microphone = open_request(AudioDirection.MICROPHONE)
    speaker = open_request(AudioDirection.SPEAKER)

    microphone_opened = engine.open_session(microphone)
    speaker_opened = engine.open_session(speaker)

    assert microphone_opened.session_id == microphone.session_id
    assert speaker_opened.session_id == speaker.session_id
    assert microphone_opened.event_sequence == 1
    assert speaker_opened.event_sequence == 1
    assert engine.queue_state(microphone.session_id).provider_input_buffered_ms == 0
    assert engine.queue_state(speaker.session_id).provider_input_buffered_ms == 0

    assert engine.enqueue_frame(
        frame(microphone, sequence=0, capture_monotonic_ns=0), now_ns=0
    ) is None
    assert engine.queue_state(microphone.session_id).provider_input_buffered_ms == 100
    assert engine.queue_state(speaker.session_id).provider_input_buffered_ms == 0

    engine.process_next(microphone.session_id, now_ns=0)
    microphone_events = engine.drain_output(microphone.session_id, now_ns=0)
    speaker_health = engine.health(speaker.session_id, now_ns=0)
    microphone_health = engine.health(microphone.session_id, now_ns=0)
    microphone_sequences = [
        microphone_opened.event_sequence,
        *(event.event_sequence for event in microphone_events),
        microphone_health.event_sequence,
    ]
    assert all(
        current > previous
        for previous, current in zip(microphone_sequences, microphone_sequences[1:])
    )
    assert speaker_health.event_sequence > speaker_opened.event_sequence


def test_mock_engine_rejects_openai_provider_sessions() -> None:
    engine = ProviderEngine()
    openai_session = open_request(AudioDirection.MICROPHONE).model_copy(
        update={"provider_id": ProviderId.OPENAI}
    )

    with pytest.raises(ProviderProtocolError, match="unsupported_provider"):
        engine.open_session(openai_session)


def test_input_and_output_queues_are_bounded_by_buffered_duration() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    opened = engine.open_session(request)

    for sequence in range(8):
        assert engine.enqueue_frame(
            frame(request, sequence=sequence, capture_monotonic_ns=0),
            now_ns=0,
        ) is None
    overflow_events = engine.enqueue_frame(
        frame(request, sequence=8, capture_monotonic_ns=0),
        now_ns=0,
    )
    assert isinstance(overflow_events, tuple)
    overflow, overflow_final = overflow_events
    assert isinstance(overflow, PrivacySafeProviderError)
    assert overflow.code is SafeErrorCode.QUEUE_OVERFLOW
    assert overflow_final.outcome is UtteranceOutcome.DROPPED
    assert overflow.event_sequence < overflow_final.event_sequence
    assert overflow.event_sequence > opened.event_sequence
    assert engine.queue_state(request.session_id).provider_input_buffered_ms == 800

    output_engine = ProviderEngine()
    output_request = open_request(AudioDirection.SPEAKER)
    output_opened = output_engine.open_session(output_request)
    for sequence in range(12):
        assert output_engine.enqueue_frame(
            frame(output_request, sequence=sequence, capture_monotonic_ns=0),
            now_ns=0,
        ) is None
        assert output_engine.process_next(output_request.session_id, now_ns=0) == ()
    assert output_engine.queue_state(
        output_request.session_id
    ).provider_output_buffered_ms == 1200
    assert output_engine.enqueue_frame(
        frame(output_request, sequence=12, capture_monotonic_ns=0),
        now_ns=0,
    ) is None
    output_overflow = output_engine.process_next(output_request.session_id, now_ns=0)
    assert len(output_overflow) == 2
    assert isinstance(output_overflow[0], PrivacySafeProviderError)
    assert output_overflow[0].code is SafeErrorCode.QUEUE_OVERFLOW
    assert output_overflow[1].outcome is UtteranceOutcome.DROPPED
    assert (
        output_opened.event_sequence
        < output_overflow[0].event_sequence
        < output_overflow[1].event_sequence
    )
    assert output_engine.queue_state(
        output_request.session_id
    ).provider_output_buffered_ms == 1200


def test_queue_durations_are_accounted_exactly_once_per_utterance() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    engine.open_session(request)
    retained_id = uuid4()
    cancelled_id = uuid4()
    engine.enqueue_frame(
        frame(
            request,
            sequence=0,
            capture_monotonic_ns=0,
            utterance_id=retained_id,
        ),
        now_ns=0,
    )
    engine.enqueue_frame(
        frame(
            request,
            sequence=1,
            capture_monotonic_ns=0,
            utterance_id=cancelled_id,
        ),
        now_ns=0,
    )
    assert engine.queue_state(request.session_id).provider_input_buffered_ms == 200

    cancelled_input = engine.cancel_utterance(
        CancelUtterance(
            session_id=request.session_id,
            direction_id=request.direction_id,
            utterance_id=cancelled_id,
            reason=CancelReason.USER_INTERRUPT,
        )
    )
    assert engine.queue_state(request.session_id).provider_input_buffered_ms == 100
    engine.process_next(request.session_id, now_ns=0)
    assert engine.process_next(request.session_id, now_ns=0) == ()
    assert engine.queue_state(request.session_id).provider_input_buffered_ms == 0
    assert engine.queue_state(request.session_id).provider_output_buffered_ms == 100
    retained_input_output = engine.drain_output(request.session_id, now_ns=0)
    assert not any(
        isinstance(event, ProviderAudioDelta)
        and event.utterance_id == cancelled_input.utterance_id
        for event in retained_input_output
    )
    assert engine.queue_state(request.session_id).provider_output_buffered_ms == 0
    engine.drain_output(request.session_id, now_ns=0)
    assert engine.queue_state(request.session_id).provider_output_buffered_ms == 0

    output_engine = ProviderEngine()
    output_request = open_request(AudioDirection.SPEAKER)
    output_engine.open_session(output_request)
    retained_output_id = uuid4()
    cancelled_output_id = uuid4()
    output_engine.enqueue_frame(
        frame(
            output_request,
            sequence=0,
            capture_monotonic_ns=0,
            utterance_id=retained_output_id,
        ),
        now_ns=0,
    )
    output_engine.enqueue_frame(
        frame(
            output_request,
            sequence=1,
            capture_monotonic_ns=0,
            utterance_id=cancelled_output_id,
        ),
        now_ns=0,
    )
    output_engine.process_next(output_request.session_id, now_ns=0)
    output_engine.process_next(output_request.session_id, now_ns=0)
    assert (
        output_engine.queue_state(
            output_request.session_id
        ).provider_output_buffered_ms
        == 200
    )
    output_engine.cancel_utterance(
        CancelUtterance(
            session_id=output_request.session_id,
            direction_id=output_request.direction_id,
            utterance_id=cancelled_output_id,
            reason=CancelReason.USER_INTERRUPT,
        )
    )
    assert (
        output_engine.queue_state(
            output_request.session_id
        ).provider_output_buffered_ms
        == 100
    )
    retained_output = output_engine.drain_output(
        output_request.session_id, now_ns=0
    )
    assert any(
        isinstance(event, ProviderAudioDelta)
        and event.utterance_id == retained_output_id
        for event in retained_output
    )
    assert not any(
        isinstance(event, ProviderAudioDelta)
        and event.utterance_id == cancelled_output_id
        for event in retained_output
    )
    assert (
        output_engine.queue_state(
            output_request.session_id
        ).provider_output_buffered_ms
        == 0
    )


def test_multi_frame_utterance_emits_exactly_one_terminal_event() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    engine.open_session(request)
    utterance_id = uuid4()
    first = frame(
        request,
        sequence=0,
        capture_monotonic_ns=0,
        utterance_id=utterance_id,
        end_of_utterance=False,
    )
    last = frame(
        request,
        sequence=1,
        capture_monotonic_ns=0,
        utterance_id=utterance_id,
        end_of_utterance=True,
    )
    engine.enqueue_frame(first, now_ns=0)
    engine.enqueue_frame(last, now_ns=0)
    engine.process_next(request.session_id, now_ns=0)
    first_events = engine.drain_output(request.session_id, now_ns=0)
    assert not any(
        isinstance(event, ProviderUtteranceFinal) for event in first_events
    )
    engine.process_next(request.session_id, now_ns=0)
    last_events = engine.drain_output(request.session_id, now_ns=0)
    finals = [
        event
        for event in (*first_events, *last_events)
        if isinstance(event, ProviderUtteranceFinal)
    ]
    assert len(finals) == 1
    assert finals[0].outcome is UtteranceOutcome.COMPLETED
    assert finals[0].final_audio_sequence == 1
    with pytest.raises(ProviderProtocolError, match="utterance_terminal"):
        engine.enqueue_frame(
            frame(
                request,
                sequence=2,
                capture_monotonic_ns=0,
                utterance_id=utterance_id,
            ),
            now_ns=0,
        )


def test_audio_sequence_restarts_for_each_utterance_in_one_session() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    engine.open_session(request)
    first_utterance = uuid4()
    second_utterance = uuid4()

    for input_sequence, utterance_id in enumerate(
        (first_utterance, second_utterance)
    ):
        engine.enqueue_frame(
            frame(
                request,
                sequence=input_sequence,
                capture_monotonic_ns=0,
                utterance_id=utterance_id,
            ),
            now_ns=0,
        )
        engine.process_next(request.session_id, now_ns=0)

    events = engine.drain_output(request.session_id, now_ns=0)
    audio = [event for event in events if isinstance(event, ProviderAudioDelta)]
    finals = [
        event for event in events if isinstance(event, ProviderUtteranceFinal)
    ]
    assert [event.utterance_id for event in audio] == [
        first_utterance,
        second_utterance,
    ]
    assert [event.sequence for event in audio] == [0, 0]
    assert [event.final_audio_sequence for event in finals] == [0, 0]


def test_open_rejects_mismatched_negotiated_formats() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    mismatched_output = request.requested_output_format.model_copy(
        update={"sample_rate_hz": 24_000}
    )
    with pytest.raises(ProviderProtocolError, match="negotiated_format_mismatch"):
        engine.open_session(
            request.model_copy(
                update={"requested_output_format": mismatched_output}
            )
        )


def test_closed_session_can_be_released_after_stream_cleanup() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.SPEAKER)
    engine.open_session(request)
    engine.close_session(
        CloseProviderSession(
            session_id=request.session_id,
            reason=CloseRequestReason.USER_STOP,
        )
    )
    engine.release_session(request.session_id)
    with pytest.raises(ProviderProtocolError, match="unknown_session"):
        engine.queue_state(request.session_id)


def test_active_unterminated_utterances_are_bounded_fail_closed() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    engine.open_session(request)
    for sequence in range(64):
        engine.enqueue_frame(
            frame(
                request,
                sequence=sequence,
                capture_monotonic_ns=0,
                end_of_utterance=False,
            ),
            now_ns=0,
        )
        engine.process_next(request.session_id, now_ns=0)
        engine.drain_output(request.session_id, now_ns=0)

    with pytest.raises(ProviderProtocolError, match="active_utterance_capacity"):
        engine.enqueue_frame(
            frame(
                request,
                sequence=64,
                capture_monotonic_ns=0,
                end_of_utterance=False,
            ),
            now_ns=0,
        )


@pytest.mark.parametrize(
    ("mode", "deadline_ms"),
    [
        (TranslationMode.QUALITY_FIRST, 3000),
        (TranslationMode.BALANCED, 2000),
        (TranslationMode.STREAMING_FIRST, 1000),
    ],
)
def test_mode_age_deadline_drops_stale_input_and_output(
    mode: TranslationMode, deadline_ms: int
) -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE, mode=mode)
    opened = engine.open_session(request)

    exact = frame(request, sequence=0, capture_monotonic_ns=0)
    assert engine.enqueue_frame(exact, now_ns=deadline_ms * 1_000_000) is None
    assert engine.process_next(request.session_id, now_ns=deadline_ms * 1_000_000) == ()
    expired_output = engine.drain_output(
        request.session_id, now_ns=(deadline_ms + 1) * 1_000_000
    )
    assert not any(isinstance(event, ProviderAudioDelta) for event in expired_output)
    assert expired_output[-1].outcome is UtteranceOutcome.DROPPED

    stale = frame(request, sequence=1, capture_monotonic_ns=0)
    dropped = engine.enqueue_frame(
        stale, now_ns=(deadline_ms + 1) * 1_000_000
    )
    assert isinstance(dropped, ProviderUtteranceFinal)
    assert dropped.outcome is UtteranceOutcome.DROPPED
    assert dropped.final_audio_sequence is None
    assert dropped.event_sequence > opened.event_sequence
    with pytest.raises(ProviderProtocolError, match="utterance_terminal"):
        engine.enqueue_frame(
            frame(
                request,
                sequence=2,
                capture_monotonic_ns=0,
                utterance_id=stale.utterance_id,
            ),
            now_ns=(deadline_ms + 1) * 1_000_000,
        )

    queued_engine = ProviderEngine()
    queued_request = open_request(AudioDirection.SPEAKER, mode=mode)
    queued_engine.open_session(queued_request)
    queued_engine.enqueue_frame(
        frame(queued_request, sequence=0, capture_monotonic_ns=0), now_ns=0
    )
    queued_expired = queued_engine.process_next(
        queued_request.session_id, now_ns=(deadline_ms + 1) * 1_000_000
    )
    assert not any(isinstance(event, ProviderAudioDelta) for event in queued_expired)
    assert queued_expired[-1].outcome is UtteranceOutcome.DROPPED
    queues = queued_engine.queue_state(queued_request.session_id)
    assert queues.provider_input_buffered_ms == 0
    assert queues.provider_output_buffered_ms == 0


def test_mock_transform_is_deterministic_and_debug_text_is_runtime_gated() -> None:
    marker = "private-provider-engine-marker"
    engine = ProviderEngine(
        injection=MockInjection(
            transcript=marker,
            translation=marker,
        )
    )
    request = open_request(AudioDirection.MICROPHONE, debug_text_enabled=False)
    engine.open_session(request)

    source = frame(request, sequence=0, capture_monotonic_ns=0)
    assert engine.enqueue_frame(source, now_ns=0) is None
    assert engine.process_next(request.session_id, now_ns=0) == ()
    hidden = engine.drain_output(request.session_id, now_ns=0)
    assert not any(
        isinstance(event, (ProviderTranscriptDelta, ProviderTranslationDelta))
        for event in hidden
    )
    audio = next(event for event in hidden if isinstance(event, ProviderAudioDelta))
    assert audio.pcm == mock_transform_pcm(source.pcm)
    assert audio.sample_rate_hz == source.sample_rate_hz
    assert audio.channels == source.channels
    assert audio.sample_format is source.sample_format
    assert audio.frame_duration_ms == source.frame_duration_ms
    assert marker not in repr(hidden)

    engine.update_debug_text(
        UpdateDebugText(session_id=request.session_id, enabled=True)
    )
    visible_source = frame(request, sequence=1, capture_monotonic_ns=0)
    assert engine.enqueue_frame(visible_source, now_ns=0) is None
    visible = engine.process_next(request.session_id, now_ns=0)
    assert [type(event) for event in visible] == [
        ProviderTranscriptDelta,
        ProviderTranslationDelta,
    ]
    assert all(event.text == marker for event in visible)

    queued_source = frame(request, sequence=2, capture_monotonic_ns=0)
    assert engine.enqueue_frame(queued_source, now_ns=0) is None
    engine.update_debug_text(
        UpdateDebugText(session_id=request.session_id, enabled=False)
    )
    assert engine.process_next(request.session_id, now_ns=0) == ()
    hidden_after_disable = engine.drain_output(request.session_id, now_ns=0)
    assert marker not in repr(hidden_after_disable)
    assert not any(
        isinstance(event, (ProviderTranscriptDelta, ProviderTranslationDelta))
        for event in hidden_after_disable
    )
    engine.update_debug_text(
        UpdateDebugText(session_id=request.session_id, enabled=True)
    )
    after_reenable = engine.drain_output(request.session_id, now_ns=0)
    assert marker not in repr(after_reenable)
    assert not any(
        isinstance(event, (ProviderTranscriptDelta, ProviderTranslationDelta))
        for event in after_reenable
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source.model_copy(update={"session_id": uuid4()}),
        lambda source: source.model_copy(
            update={"direction_id": AudioDirection.SPEAKER}
        ),
        lambda source: source.model_copy(
            update={"sample_rate_hz": 24_000, "pcm": b"\x01\x02" * 2400}
        ),
        lambda source: source.model_copy(update={"pcm": source.pcm[:-2]}),
        lambda source: source.model_copy(update={"source_language": Language.EN}),
        lambda source: source.model_copy(update={"target_language": Language.RU}),
        lambda source: source.model_copy(
            update={"mode": TranslationMode.STREAMING_FIRST}
        ),
    ],
    ids=[
        "foreign_session",
        "direction_mismatch",
        "format_mismatch",
        "pcm_length_mismatch",
        "source_language_mismatch",
        "target_language_mismatch",
        "mode_mismatch",
    ],
)
def test_invalid_frames_are_terminal_stream_protocol_errors(mutate) -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    engine.open_session(request)
    invalid = mutate(frame(request, sequence=0, capture_monotonic_ns=0))

    with pytest.raises(ProviderProtocolError) as raised:
        engine.enqueue_frame(invalid, now_ns=0)
    assert raised.value.terminate_stream is True


def test_stale_frame_after_newer_sequence_is_a_terminal_protocol_error() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    engine.open_session(request)
    engine.enqueue_frame(frame(request, sequence=0, capture_monotonic_ns=0), now_ns=0)
    engine.enqueue_frame(frame(request, sequence=2, capture_monotonic_ns=0), now_ns=0)

    with pytest.raises(ProviderProtocolError, match="stale_frame_sequence") as raised:
        engine.enqueue_frame(
            frame(request, sequence=1, capture_monotonic_ns=0), now_ns=0
        )
    assert raised.value.terminate_stream is True


def test_stream_identity_cannot_change_after_first_frame() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    engine.open_session(request)
    first = frame(request, sequence=0, capture_monotonic_ns=0)
    engine.enqueue_frame(first, now_ns=0)
    changed = frame(request, sequence=1, capture_monotonic_ns=0).model_copy(
        update={"stream_id": uuid4()}
    )
    with pytest.raises(
        ProviderProtocolError, match="stream_identity_mismatch"
    ) as raised:
        engine.enqueue_frame(changed, now_ns=0)
    assert raised.value.terminate_stream is True


def test_health_and_latency_events_are_typed_ordered_and_content_free() -> None:
    marker = "private-health-latency-marker"
    engine = ProviderEngine(
        injection=MockInjection(
            process_delay_ms=250,
            transcript=marker,
            translation=marker,
        )
    )
    request = open_request(AudioDirection.SPEAKER, debug_text_enabled=False)
    opened = engine.open_session(request)
    source = frame(request, sequence=0, capture_monotonic_ns=0)
    engine.enqueue_frame(source, now_ns=0)
    queued_health = engine.health(request.session_id, now_ns=10_000_000)
    assert isinstance(queued_health, ProviderHealth)
    assert queued_health.state is ProviderState.READY
    assert queued_health.queues.provider_input_buffered_ms == 100
    assert queued_health.event_sequence > opened.event_sequence
    assert engine.process_next(request.session_id, now_ns=249_000_000) == ()
    engine.process_next(request.session_id, now_ns=250_000_000)
    events = engine.drain_output(request.session_id, now_ns=250_000_000)
    latency = next(event for event in events if isinstance(event, ProviderLatency))
    assert latency.session_id == request.session_id
    assert latency.direction_id is request.direction_id
    assert latency.stream_id == source.stream_id
    assert latency.utterance_id == source.utterance_id
    assert latency.provider_total_ms == 250
    sequences = [
        opened.event_sequence,
        queued_health.event_sequence,
        *(event.event_sequence for event in events),
    ]
    assert sequences == sorted(set(sequences))
    assert marker not in repr((queued_health, latency))


def test_duplicate_post_final_cancel_and_close_transitions_fail_closed() -> None:
    engine = ProviderEngine()
    request = open_request(AudioDirection.MICROPHONE)
    opened = engine.open_session(request)
    utterance_id = uuid4()
    source = frame(
        request,
        sequence=0,
        capture_monotonic_ns=0,
        utterance_id=utterance_id,
    )
    assert engine.enqueue_frame(source, now_ns=0) is None
    with pytest.raises(ProviderProtocolError, match="duplicate_frame_sequence"):
        engine.enqueue_frame(source, now_ns=0)
    engine.process_next(request.session_id, now_ns=0)
    completed = engine.drain_output(request.session_id, now_ns=0)
    assert completed[-1].outcome is UtteranceOutcome.COMPLETED
    with pytest.raises(ProviderProtocolError, match="utterance_terminal"):
        engine.enqueue_frame(
            frame(
                request,
                sequence=1,
                capture_monotonic_ns=0,
                utterance_id=utterance_id,
            ),
            now_ns=0,
        )

    cancelled_id = uuid4()
    assert engine.enqueue_frame(
        frame(
            request,
            sequence=2,
            capture_monotonic_ns=0,
            utterance_id=cancelled_id,
        ),
        now_ns=0,
    ) is None
    cancelled = engine.cancel_utterance(
        CancelUtterance(
            session_id=request.session_id,
            direction_id=request.direction_id,
            utterance_id=cancelled_id,
            reason=CancelReason.USER_INTERRUPT,
        )
    )
    assert cancelled.outcome is UtteranceOutcome.CANCELLED
    assert cancelled.final_audio_sequence is None
    assert cancelled.event_sequence > opened.event_sequence
    assert engine.queue_state(request.session_id).provider_input_buffered_ms == 0
    with pytest.raises(ProviderProtocolError, match="utterance_terminal"):
        engine.enqueue_frame(
            frame(
                request,
                sequence=3,
                capture_monotonic_ns=0,
                utterance_id=cancelled_id,
            ),
            now_ns=0,
        )

    closed = engine.close_session(
        CloseProviderSession(
            session_id=request.session_id,
            reason=CloseRequestReason.USER_STOP,
        )
    )
    assert isinstance(closed, ProviderSessionClosed)
    assert closed.event_sequence > cancelled.event_sequence
    with pytest.raises(ProviderProtocolError, match="session_terminal"):
        engine.enqueue_frame(
            frame(request, sequence=4, capture_monotonic_ns=0), now_ns=0
        )
    with pytest.raises(ProviderProtocolError, match="session_terminal"):
        engine.update_debug_text(
            UpdateDebugText(session_id=request.session_id, enabled=True)
        )
    with pytest.raises(ProviderProtocolError, match="session_terminal"):
        engine.cancel_utterance(
            CancelUtterance(
                session_id=request.session_id,
                direction_id=request.direction_id,
                utterance_id=uuid4(),
                reason=CancelReason.USER_INTERRUPT,
            )
        )
    with pytest.raises(ProviderProtocolError, match="session_terminal"):
        engine.close_session(
            CloseProviderSession(
                session_id=request.session_id,
                reason=CloseRequestReason.USER_STOP,
            )
        )
    with pytest.raises(ProviderProtocolError, match="session_terminal"):
        engine.health(request.session_id, now_ns=0)
    with pytest.raises(ProviderProtocolError, match="session_terminal"):
        engine.process_next(request.session_id, now_ns=0)


def test_injected_latency_and_error_modes_are_typed_and_content_free() -> None:
    marker = "private-injected-error-marker"
    engine = ProviderEngine(
        injection=MockInjection(
            process_delay_ms=250,
            fail_after_frames=0,
            transcript=marker,
            translation=marker,
        )
    )
    request = open_request(AudioDirection.SPEAKER, debug_text_enabled=True)
    engine.open_session(request)
    assert engine.enqueue_frame(
        frame(request, sequence=0, capture_monotonic_ns=0), now_ns=0
    ) is None

    assert engine.process_next(request.session_id, now_ns=249_000_000) == ()
    failure = engine.process_next(request.session_id, now_ns=250_000_000)
    assert len(failure) == 2
    assert isinstance(failure[0], PrivacySafeProviderError)
    assert failure[0].code is SafeErrorCode.PROVIDER_UNAVAILABLE
    assert failure[1].outcome is UtteranceOutcome.DROPPED
    assert marker not in repr(failure)
    assert engine.enqueue_frame(
        frame(request, sequence=1, capture_monotonic_ns=0), now_ns=250_000_000
    ) is None
    assert isinstance(
        engine.health(request.session_id, now_ns=250_000_000), ProviderHealth
    )
    assert isinstance(
        engine.close_session(
            CloseProviderSession(
                session_id=request.session_id,
                reason=CloseRequestReason.USER_STOP,
            )
        ),
        ProviderSessionClosed,
    )
