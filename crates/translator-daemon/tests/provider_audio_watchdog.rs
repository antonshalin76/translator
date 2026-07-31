use translator_daemon::{
    CANCEL_FINAL_TIMEOUT, CLOSE_ACK_TIMEOUT, INTER_AUDIO_DELTA_TIMEOUT, ProviderAudioWatchdog,
    ProviderStreamCoordinator, WatchdogAction, WatchdogError,
};
use translator_ipc::{
    ProviderSessionContract, ProviderValidationError,
    provider::{
        AudioDirection, Language, PcmFormat, ProviderAudioDelta, ProviderCapabilities,
        ProviderEvent, ProviderId, ProviderLatency, ProviderSessionClosed, ProviderSessionOpened,
        ProviderUtteranceFinal, SampleFormat, SessionCloseReason,
        TranslationMode as ProviderTranslationMode, UtteranceOutcome, provider_event,
    },
};
use uuid::Uuid;

const NS_PER_MS: u64 = 1_000_000;

fn cancel_action(session_id: Uuid, stream_id: Uuid, utterance_id: Uuid) -> WatchdogAction {
    WatchdogAction::CancelUtterance {
        session_id,
        stream_id,
        utterance_id,
        purge_receive_buffer: true,
    }
}

fn pcm_format() -> PcmFormat {
    PcmFormat {
        sample_rate_hz: 16_000,
        channels: 1,
        sample_format: SampleFormat::S16le.into(),
        frame_duration_ms: 100,
    }
}

fn provider_contract(session_id: Uuid, stream_id: Uuid) -> ProviderSessionContract {
    ProviderSessionContract {
        session_id,
        stream_id,
        provider_id: ProviderId::Local,
        direction_id: AudioDirection::Microphone,
        source_language: Language::Ru,
        target_language: Language::En,
        mode: ProviderTranslationMode::StreamingFirst,
        input_format: pcm_format(),
        output_format: pcm_format(),
        debug_text_enabled: false,
    }
}

fn opened(session_id: Uuid) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::SessionOpened(
            ProviderSessionOpened {
                schema_version: "translator.provider.session_opened.v1".into(),
                session_id: session_id.to_string(),
                direction_id: AudioDirection::Microphone.into(),
                negotiated_input_format: Some(pcm_format()),
                negotiated_output_format: Some(pcm_format()),
                capabilities: Some(ProviderCapabilities {
                    audio_output: true,
                    transcript_delta: true,
                    translation_delta: true,
                    cancellation: true,
                    cloud_egress: false,
                }),
                event_sequence: 1,
            },
        )),
    }
}

fn cancelled_final(session_id: Uuid, stream_id: Uuid, utterance_id: Uuid) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::UtteranceFinal(
            ProviderUtteranceFinal {
                schema_version: "translator.provider.utterance_final.v1".into(),
                session_id: session_id.to_string(),
                direction_id: AudioDirection::Microphone.into(),
                stream_id: stream_id.to_string(),
                utterance_id: utterance_id.to_string(),
                event_sequence: 3,
                final_audio_sequence: Some(0),
                outcome: UtteranceOutcome::Cancelled.into(),
            },
        )),
    }
}

fn audio_delta(
    session_id: Uuid,
    stream_id: Uuid,
    utterance_id: Uuid,
    sequence: u64,
    event_sequence: u64,
    now_ns: u64,
) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::AudioDelta(ProviderAudioDelta {
            schema_version: "translator.provider.audio_delta.v1".into(),
            session_id: session_id.to_string(),
            direction_id: AudioDirection::Microphone.into(),
            stream_id: stream_id.to_string(),
            utterance_id: utterance_id.to_string(),
            sequence,
            event_sequence,
            provider_monotonic_ns: now_ns,
            sample_rate_hz: 16_000,
            channels: 1,
            sample_format: SampleFormat::S16le.into(),
            frame_duration_ms: 100,
            pcm: vec![0; 3_200],
        })),
    }
}

fn completed_final(
    session_id: Uuid,
    stream_id: Uuid,
    utterance_id: Uuid,
    event_sequence: u64,
    final_audio_sequence: Option<u64>,
) -> ProviderEvent {
    let mut event = cancelled_final(session_id, stream_id, utterance_id);
    if let Some(provider_event::Event::UtteranceFinal(value)) = event.event.as_mut() {
        value.event_sequence = event_sequence;
        value.final_audio_sequence = final_audio_sequence;
        value.outcome = UtteranceOutcome::Completed.into();
    }
    event
}

fn session_closed(session_id: Uuid, event_sequence: u64) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::SessionClosed(
            ProviderSessionClosed {
                schema_version: "translator.provider.session_closed.v1".into(),
                session_id: session_id.to_string(),
                direction_id: AudioDirection::Microphone.into(),
                event_sequence,
                reason: SessionCloseReason::CloseTimeout.into(),
            },
        )),
    }
}

fn latency_event(
    session_id: Uuid,
    stream_id: Uuid,
    utterance_id: Uuid,
    event_sequence: u64,
) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::Latency(ProviderLatency {
            schema_version: "translator.provider.latency.v1".into(),
            session_id: session_id.to_string(),
            direction_id: AudioDirection::Microphone.into(),
            stream_id: stream_id.to_string(),
            event_sequence,
            utterance_id: Some(utterance_id.to_string()),
            asr_first_text_ms: Some(1),
            asr_final_text_ms: Some(2),
            mt_first_text_ms: Some(3),
            tts_first_audio_ms: Some(4),
            provider_total_ms: Some(5),
        })),
    }
}

#[test]
fn collecting_input_allows_a_three_second_utterance_before_eou() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();

    assert!(watchdog.poll(3_000 * NS_PER_MS).is_empty());
}

#[test]
fn eou_starts_a_six_second_first_audio_deadline() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let eou_ns = 3_000 * NS_PER_MS;
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    watchdog
        .record_end_of_utterance(stream_id, utterance_id, eou_ns)
        .unwrap();

    assert!(watchdog.poll(eou_ns + 6_000 * NS_PER_MS).is_empty());
    assert_eq!(
        watchdog.poll(eou_ns + 6_000 * NS_PER_MS + 1),
        vec![cancel_action(session_id, stream_id, utterance_id)]
    );
}

#[test]
fn collecting_input_over_twelve_seconds_is_cancelled() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();

    assert!(watchdog.poll(12_000 * NS_PER_MS).is_empty());
    assert_eq!(
        watchdog.poll(12_000 * NS_PER_MS + 1),
        vec![cancel_action(session_id, stream_id, utterance_id)]
    );
}

#[test]
fn provider_audio_before_eou_is_rejected() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();

    assert_eq!(
        watchdog.record_audio_delta(stream_id, utterance_id, 1),
        Err(WatchdogError::InputNotComplete)
    );
}

#[test]
fn coordinator_rejects_provider_audio_and_final_before_eou() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut coordinator = ProviderStreamCoordinator::new(provider_contract(session_id, stream_id));
    coordinator.validate_event(&opened(session_id), 0).unwrap();
    coordinator
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();

    assert_eq!(
        coordinator.validate_event(
            &audio_delta(session_id, stream_id, utterance_id, 0, 2, 1),
            1,
        ),
        Err(WatchdogError::InputNotComplete.into())
    );
    assert_eq!(
        coordinator.validate_event(
            &completed_final(session_id, stream_id, utterance_id, 2, None),
            1,
        ),
        Err(WatchdogError::InputNotComplete.into())
    );
}

#[test]
fn completed_terminal_after_eou_clears_the_utterance() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    watchdog
        .record_end_of_utterance(stream_id, utterance_id, 3_000 * NS_PER_MS)
        .unwrap();

    watchdog
        .record_completed_final(stream_id, utterance_id)
        .unwrap();
    assert_eq!(watchdog.active_utterance_count(), 0);
}

#[test]
fn watchdog_switches_from_first_audio_to_inter_delta_deadline() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let started_at = 5_000 * NS_PER_MS;
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, started_at)
        .unwrap();
    watchdog
        .record_end_of_utterance(stream_id, utterance_id, started_at)
        .unwrap();

    assert!(watchdog.poll(started_at + 1_000 * NS_PER_MS).is_empty());
    watchdog
        .record_audio_delta(stream_id, utterance_id, started_at + 1_000 * NS_PER_MS)
        .unwrap();
    assert!(watchdog.poll(started_at + 1_250 * NS_PER_MS).is_empty());
    watchdog
        .record_audio_delta(stream_id, utterance_id, started_at + 1_250 * NS_PER_MS)
        .unwrap();
    assert!(watchdog.poll(started_at + 1_500 * NS_PER_MS).is_empty());
    assert_eq!(
        watchdog.poll(started_at + 1_500 * NS_PER_MS + 1),
        vec![cancel_action(session_id, stream_id, utterance_id)]
    );
    assert!(watchdog.cancel_pending(stream_id, utterance_id));
    assert!(watchdog.poll(started_at + 1_500 * NS_PER_MS + 1).is_empty());
}

#[test]
fn watchdog_uses_a_fixed_six_second_first_audio_safety_deadline() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    watchdog
        .record_end_of_utterance(stream_id, utterance_id, 0)
        .unwrap();

    assert!(watchdog.poll(6_000 * NS_PER_MS).is_empty());
    assert_eq!(
        watchdog.poll(6_000 * NS_PER_MS + 1),
        vec![cancel_action(session_id, stream_id, utterance_id)]
    );
}

#[test]
fn cancelled_final_acks_cancel_and_clears_utterance_state() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    watchdog
        .record_end_of_utterance(stream_id, utterance_id, 0)
        .unwrap();
    watchdog.poll(6_000 * NS_PER_MS + 1);

    assert_eq!(
        watchdog.record_completed_final(stream_id, utterance_id),
        Err(WatchdogError::ExpectedCancelledFinal)
    );
    watchdog
        .record_cancelled_final(stream_id, utterance_id)
        .unwrap();
    assert!(!watchdog.cancel_pending(stream_id, utterance_id));
    assert_eq!(
        watchdog.record_cancelled_final(stream_id, utterance_id),
        Err(WatchdogError::UnknownUtterance)
    );
}

#[test]
fn missing_cancel_final_closes_session_then_missing_close_ack_restarts_sidecar() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    watchdog
        .record_end_of_utterance(stream_id, utterance_id, 0)
        .unwrap();
    let cancel_at = 6_000 * NS_PER_MS + 1;
    watchdog.poll(cancel_at);

    assert!(
        watchdog
            .poll(cancel_at + CANCEL_FINAL_TIMEOUT.as_nanos() as u64)
            .is_empty()
    );
    let close_at = cancel_at + CANCEL_FINAL_TIMEOUT.as_nanos() as u64 + 1;
    assert_eq!(
        watchdog.poll(close_at),
        vec![WatchdogAction::CloseProviderSession { session_id }]
    );
    assert!(watchdog.poll(close_at).is_empty());
    assert!(
        watchdog
            .poll(close_at + CLOSE_ACK_TIMEOUT.as_nanos() as u64 - 1)
            .is_empty()
    );
    assert_eq!(
        watchdog.poll(close_at + CLOSE_ACK_TIMEOUT.as_nanos() as u64),
        vec![WatchdogAction::RestartSidecar { session_id }]
    );
    assert!(
        watchdog
            .poll(close_at + CLOSE_ACK_TIMEOUT.as_nanos() as u64)
            .is_empty()
    );
}

#[test]
fn close_ack_external_stop_and_terminal_events_disarm_scoped_state() {
    let session_id = Uuid::new_v4();
    let stream_a = Uuid::new_v4();
    let stream_b = Uuid::new_v4();
    let utterance_a = Uuid::new_v4();
    let utterance_b = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog.start_utterance(stream_a, utterance_a, 0).unwrap();
    watchdog.start_utterance(stream_b, utterance_b, 0).unwrap();
    watchdog
        .record_end_of_utterance(stream_a, utterance_a, 0)
        .unwrap();
    watchdog
        .record_completed_final(stream_a, utterance_a)
        .unwrap();
    assert_eq!(
        watchdog.record_audio_delta(stream_a, utterance_a, 1),
        Err(WatchdogError::UnknownUtterance)
    );

    watchdog.disarm_session();
    assert_eq!(watchdog.active_utterance_count(), 0);
    assert_eq!(
        watchdog.record_audio_delta(stream_b, utterance_b, 1),
        Err(WatchdogError::SessionTerminal)
    );

    let mut closing = ProviderAudioWatchdog::new(session_id);
    closing.start_utterance(stream_a, utterance_a, 0).unwrap();
    closing
        .record_end_of_utterance(stream_a, utterance_a, 0)
        .unwrap();
    let cancel_at = 6_000 * NS_PER_MS + 1;
    closing.poll(cancel_at);
    let close_at = cancel_at + CANCEL_FINAL_TIMEOUT.as_nanos() as u64 + 1;
    closing.poll(close_at);
    closing.record_session_closed(session_id).unwrap();
    assert!(
        closing
            .poll(close_at + CLOSE_ACK_TIMEOUT.as_nanos() as u64)
            .is_empty()
    );
}

#[test]
fn coordinator_atomically_purges_and_marks_validator_before_cancel_action() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut coordinator = ProviderStreamCoordinator::new(provider_contract(session_id, stream_id));
    coordinator.validate_event(&opened(session_id), 0).unwrap();
    coordinator
        .start_utterance(stream_id, utterance_id, 10 * NS_PER_MS)
        .unwrap();
    coordinator
        .record_end_of_utterance(stream_id, utterance_id, 10 * NS_PER_MS)
        .unwrap();
    coordinator
        .validate_event(
            &audio_delta(session_id, stream_id, utterance_id, 0, 2, 100 * NS_PER_MS),
            100 * NS_PER_MS,
        )
        .unwrap();
    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, utterance_id),
        1
    );

    assert_eq!(
        coordinator.poll(350 * NS_PER_MS + 1).unwrap(),
        vec![cancel_action(session_id, stream_id, utterance_id)]
    );
    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, utterance_id),
        0
    );
    assert!(coordinator.validator_cancel_pending(utterance_id));

    coordinator
        .validate_event(
            &cancelled_final(session_id, stream_id, utterance_id),
            350 * NS_PER_MS + 1,
        )
        .unwrap();
    assert!(!coordinator.validator_cancel_pending(utterance_id));
    assert!(
        !coordinator
            .watchdog()
            .cancel_pending(stream_id, utterance_id)
    );
}

#[test]
fn coordinator_consumes_audio_and_retries_buffer_full_event_transactionally() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut coordinator = ProviderStreamCoordinator::new(provider_contract(session_id, stream_id));
    coordinator.validate_event(&opened(session_id), 0).unwrap();
    coordinator
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    coordinator
        .record_end_of_utterance(stream_id, utterance_id, 0)
        .unwrap();

    for sequence in 0..4 {
        let now_ns = (sequence + 1) * 100 * NS_PER_MS;
        coordinator
            .validate_event(
                &audio_delta(
                    session_id,
                    stream_id,
                    utterance_id,
                    sequence,
                    sequence + 2,
                    now_ns,
                ),
                now_ns,
            )
            .unwrap();
    }
    let fifth = audio_delta(session_id, stream_id, utterance_id, 4, 6, 500 * NS_PER_MS);
    assert!(coordinator.validate_event(&fifth, 500 * NS_PER_MS).is_err());
    assert_eq!(
        coordinator
            .consume_receive_audio(stream_id, utterance_id)
            .unwrap()
            .len(),
        3_200
    );
    coordinator.validate_event(&fifth, 500 * NS_PER_MS).unwrap();
    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, utterance_id),
        4
    );
}

#[test]
fn completed_final_preserves_unplayed_audio_until_playback_drain() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut coordinator = ProviderStreamCoordinator::new(provider_contract(session_id, stream_id));
    coordinator.validate_event(&opened(session_id), 0).unwrap();
    coordinator
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    coordinator
        .record_end_of_utterance(stream_id, utterance_id, 0)
        .unwrap();
    coordinator
        .validate_event(
            &audio_delta(session_id, stream_id, utterance_id, 0, 2, 100 * NS_PER_MS),
            100 * NS_PER_MS,
        )
        .unwrap();
    coordinator
        .validate_event(
            &completed_final(session_id, stream_id, utterance_id, 3, Some(0)),
            100 * NS_PER_MS,
        )
        .unwrap();

    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, utterance_id),
        1
    );
    assert_eq!(
        coordinator
            .consume_receive_audio(stream_id, utterance_id)
            .unwrap()
            .len(),
        3_200
    );
    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, utterance_id),
        0
    );
}

#[test]
fn buffer_full_does_not_advance_watchdog_deadline() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut coordinator = ProviderStreamCoordinator::new(provider_contract(session_id, stream_id));
    coordinator.validate_event(&opened(session_id), 0).unwrap();
    coordinator
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    coordinator
        .record_end_of_utterance(stream_id, utterance_id, 0)
        .unwrap();
    for sequence in 0..4 {
        let now_ns = (sequence + 1) * 100 * NS_PER_MS;
        coordinator
            .validate_event(
                &audio_delta(
                    session_id,
                    stream_id,
                    utterance_id,
                    sequence,
                    sequence + 2,
                    now_ns,
                ),
                now_ns,
            )
            .unwrap();
    }
    let rejected = audio_delta(session_id, stream_id, utterance_id, 4, 6, 500 * NS_PER_MS);
    assert!(
        coordinator
            .validate_event(&rejected, 500 * NS_PER_MS)
            .is_err()
    );
    assert_eq!(
        coordinator.poll(650 * NS_PER_MS + 1).unwrap(),
        vec![cancel_action(session_id, stream_id, utterance_id)]
    );
}

#[test]
fn coordinator_checks_deadline_before_final_and_gates_close_pending_events() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let buffered_id = Uuid::new_v4();
    let mut coordinator = ProviderStreamCoordinator::new(provider_contract(session_id, stream_id));
    coordinator.validate_event(&opened(session_id), 0).unwrap();
    coordinator
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    coordinator
        .record_end_of_utterance(stream_id, utterance_id, 0)
        .unwrap();
    assert_eq!(
        coordinator.validate_event(
            &completed_final(session_id, stream_id, utterance_id, 2, None),
            6_000 * NS_PER_MS + 1,
        ),
        Err(WatchdogError::DeadlineExpired.into())
    );
    coordinator
        .start_utterance(stream_id, buffered_id, 5_800 * NS_PER_MS)
        .unwrap();
    coordinator
        .record_end_of_utterance(stream_id, buffered_id, 5_800 * NS_PER_MS)
        .unwrap();
    coordinator
        .validate_event(
            &audio_delta(session_id, stream_id, buffered_id, 0, 2, 5_900 * NS_PER_MS),
            5_900 * NS_PER_MS,
        )
        .unwrap();
    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, buffered_id),
        1
    );

    let cancel_at = 6_000 * NS_PER_MS + 1;
    assert_eq!(
        coordinator.poll(cancel_at).unwrap(),
        vec![cancel_action(session_id, stream_id, utterance_id)]
    );
    let close_at = cancel_at + CANCEL_FINAL_TIMEOUT.as_nanos() as u64 + 1;
    assert_eq!(
        coordinator.poll(close_at).unwrap(),
        vec![WatchdogAction::CloseProviderSession { session_id }]
    );
    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, buffered_id),
        0
    );
    assert_eq!(
        coordinator.validate_event(
            &audio_delta(session_id, stream_id, buffered_id, 1, 3, close_at,),
            close_at,
        ),
        Err(WatchdogError::SessionTerminal.into())
    );
    assert_eq!(
        coordinator.validate_event(
            &latency_event(session_id, stream_id, buffered_id, 3),
            close_at,
        ),
        Err(WatchdogError::SessionTerminal.into())
    );
    coordinator
        .validate_event(&session_closed(session_id, 3), close_at)
        .unwrap();
    assert!(
        coordinator
            .poll(close_at + CLOSE_ACK_TIMEOUT.as_nanos() as u64)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn watchdog_rejects_wrong_scope_late_audio_and_duplicate_ids() {
    assert_eq!(INTER_AUDIO_DELTA_TIMEOUT.as_millis(), 250);
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut watchdog = ProviderAudioWatchdog::new(session_id);
    watchdog
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    watchdog
        .record_end_of_utterance(stream_id, utterance_id, 0)
        .unwrap();
    watchdog.poll(6_000 * NS_PER_MS + 1);

    assert_eq!(
        watchdog.record_audio_delta(stream_id, utterance_id, 6_001 * NS_PER_MS),
        Err(WatchdogError::CancelPending)
    );
    assert_eq!(
        watchdog.start_utterance(Uuid::new_v4(), utterance_id, 0),
        Err(WatchdogError::DuplicateUtterance)
    );
    assert_eq!(
        watchdog.record_cancelled_final(Uuid::new_v4(), utterance_id),
        Err(WatchdogError::StreamMismatch)
    );
}

#[test]
fn coordinator_turns_expired_first_audio_into_scoped_cancellation() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut coordinator = ProviderStreamCoordinator::new(provider_contract(session_id, stream_id));
    coordinator.validate_event(&opened(session_id), 0).unwrap();
    coordinator
        .start_utterance(stream_id, utterance_id, 0)
        .unwrap();
    coordinator
        .record_end_of_utterance(stream_id, utterance_id, 500 * NS_PER_MS)
        .unwrap();

    assert_eq!(
        coordinator.validate_event(
            &audio_delta(
                session_id,
                stream_id,
                utterance_id,
                0,
                2,
                3_000 * NS_PER_MS + 1,
            ),
            3_000 * NS_PER_MS + 1,
        ),
        Err(ProviderValidationError::ExpiredAudio.into())
    );
    assert_eq!(
        coordinator
            .cancel_expired_utterance(stream_id, utterance_id, 3_000 * NS_PER_MS + 1,)
            .unwrap(),
        cancel_action(session_id, stream_id, utterance_id)
    );
    assert!(coordinator.validator_cancel_pending(utterance_id));
    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, utterance_id),
        0
    );
    assert_eq!(
        coordinator.validate_event(
            &audio_delta(
                session_id,
                stream_id,
                utterance_id,
                1,
                3,
                3_000 * NS_PER_MS + 2,
            ),
            3_000 * NS_PER_MS + 2,
        ),
        Err(ProviderValidationError::CancelledAudio.into())
    );
    assert_eq!(
        coordinator.receive_buffered_frames(stream_id, utterance_id),
        0
    );
    let mut final_event = cancelled_final(session_id, stream_id, utterance_id);
    if let Some(provider_event::Event::UtteranceFinal(value)) = final_event.event.as_mut() {
        value.event_sequence = 4;
        value.final_audio_sequence = Some(1);
    }
    coordinator
        .validate_event(&final_event, 3_000 * NS_PER_MS + 3)
        .unwrap();
    assert_eq!(coordinator.watchdog().active_utterance_count(), 0);
}
