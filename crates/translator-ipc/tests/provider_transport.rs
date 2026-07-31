use translator_ipc::{
    MAX_ACTIVE_UTTERANCES, MAX_TERMINAL_UTTERANCES, ProviderEventValidator,
    ProviderSessionContract, ProviderValidationError, authenticated_request,
    provider::{
        AudioDirection, Language, PcmFormat, ProviderAudioDelta, ProviderCapabilities,
        ProviderError, ProviderEvent, ProviderHealth, ProviderId, ProviderLatency, ProviderQueues,
        ProviderSessionClosed, ProviderSessionOpened, ProviderTranscriptDelta,
        ProviderTranslationDelta, ProviderUtteranceFinal, SafeErrorCode, SafeErrorSummary,
        SampleFormat, SessionCloseReason, TranslationMode, UtteranceOutcome, provider_event,
    },
};
use uuid::Uuid;

const TOKEN: &str = "abababababababababababababababababababababababababababababababab";

fn pcm_format() -> PcmFormat {
    PcmFormat {
        sample_rate_hz: 16_000,
        channels: 1,
        sample_format: SampleFormat::S16le.into(),
        frame_duration_ms: 100,
    }
}

fn contract(
    session_id: Uuid,
    mode: TranslationMode,
    debug_text_enabled: bool,
) -> ProviderSessionContract {
    ProviderSessionContract {
        session_id,
        stream_id: Uuid::nil(),
        provider_id: ProviderId::Local,
        direction_id: AudioDirection::Microphone,
        source_language: Language::Ru,
        target_language: Language::En,
        mode,
        input_format: pcm_format(),
        output_format: pcm_format(),
        debug_text_enabled,
    }
}

fn opened(session_id: Uuid, event_sequence: u64) -> ProviderEvent {
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
                event_sequence,
            },
        )),
    }
}

fn audio(
    session_id: Uuid,
    utterance_id: Uuid,
    event_sequence: u64,
    audio_sequence: u64,
) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::AudioDelta(ProviderAudioDelta {
            schema_version: "translator.provider.audio_delta.v1".into(),
            session_id: session_id.to_string(),
            direction_id: AudioDirection::Microphone.into(),
            stream_id: Uuid::nil().to_string(),
            utterance_id: utterance_id.to_string(),
            sequence: audio_sequence,
            event_sequence,
            provider_monotonic_ns: 0,
            sample_rate_hz: 16_000,
            channels: 1,
            sample_format: SampleFormat::S16le.into(),
            frame_duration_ms: 100,
            pcm: vec![0; 3200],
        })),
    }
}

fn transcript(
    session_id: Uuid,
    utterance_id: Uuid,
    event_sequence: u64,
    text: &str,
) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::TranscriptDelta(
            ProviderTranscriptDelta {
                schema_version: "translator.provider.transcript_delta.v1".into(),
                session_id: session_id.to_string(),
                direction_id: AudioDirection::Microphone.into(),
                stream_id: Uuid::nil().to_string(),
                utterance_id: utterance_id.to_string(),
                event_sequence,
                text: text.into(),
                is_final: true,
            },
        )),
    }
}

fn translation(
    session_id: Uuid,
    utterance_id: Uuid,
    event_sequence: u64,
    text: &str,
) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::TranslationDelta(
            ProviderTranslationDelta {
                schema_version: "translator.provider.translation_delta.v1".into(),
                session_id: session_id.to_string(),
                direction_id: AudioDirection::Microphone.into(),
                stream_id: Uuid::nil().to_string(),
                utterance_id: utterance_id.to_string(),
                event_sequence,
                text: text.into(),
                stable_prefix: true,
                is_final: true,
            },
        )),
    }
}

fn latency(session_id: Uuid, utterance_id: Uuid, event_sequence: u64) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::Latency(ProviderLatency {
            schema_version: "translator.provider.latency.v1".into(),
            session_id: session_id.to_string(),
            direction_id: AudioDirection::Microphone.into(),
            stream_id: Uuid::nil().to_string(),
            event_sequence,
            utterance_id: Some(utterance_id.to_string()),
            asr_first_text_ms: None,
            asr_final_text_ms: None,
            mt_first_text_ms: None,
            tts_first_audio_ms: Some(50),
            provider_total_ms: Some(90),
        })),
    }
}

fn no_speech_latency(session_id: Uuid, utterance_id: Uuid, event_sequence: u64) -> ProviderEvent {
    let mut event = latency(session_id, utterance_id, event_sequence);
    if let Some(provider_event::Event::Latency(value)) = event.event.as_mut() {
        value.asr_first_text_ms = Some(20);
        value.asr_final_text_ms = Some(20);
        value.mt_first_text_ms = None;
        value.tts_first_audio_ms = None;
    }
    event
}

fn health(session_id: Uuid, event_sequence: u64) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::Health(ProviderHealth {
            schema_version: "translator.provider.health.v1".into(),
            session_id: session_id.to_string(),
            direction_id: AudioDirection::Microphone.into(),
            event_sequence,
            provider_id: 1,
            provider_name: "deterministic-mock".into(),
            state: 2,
            models: vec![],
            queues: Some(ProviderQueues {
                provider_input_buffered_ms: 0,
                provider_output_buffered_ms: 0,
                queue_lag_ms: 0,
            }),
            retry: None,
            safe_error: None,
        })),
    }
}

fn provider_error(session_id: Uuid, event_sequence: u64, safe_message: &str) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::Error(ProviderError {
            schema_version: "translator.provider.error.v1".into(),
            session_id: session_id.to_string(),
            direction_id: AudioDirection::Microphone.into(),
            stream_id: None,
            event_sequence,
            code: SafeErrorCode::ProviderUnavailable.into(),
            retryable: true,
            safe_message: safe_message.into(),
            utterance_id: None,
        })),
    }
}

fn queue_overflow_error(
    session_id: Uuid,
    utterance_id: Uuid,
    event_sequence: u64,
) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::Error(ProviderError {
            schema_version: "translator.provider.error.v1".into(),
            session_id: session_id.to_string(),
            direction_id: AudioDirection::Microphone.into(),
            stream_id: Some(Uuid::nil().to_string()),
            event_sequence,
            code: SafeErrorCode::QueueOverflow.into(),
            retryable: true,
            safe_message: "Provider queue limit was reached".into(),
            utterance_id: Some(utterance_id.to_string()),
        })),
    }
}

fn final_event(
    session_id: Uuid,
    utterance_id: Uuid,
    event_sequence: u64,
    final_audio_sequence: Option<u64>,
) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::UtteranceFinal(
            ProviderUtteranceFinal {
                schema_version: "translator.provider.utterance_final.v1".into(),
                session_id: session_id.to_string(),
                direction_id: AudioDirection::Microphone.into(),
                stream_id: Uuid::nil().to_string(),
                utterance_id: utterance_id.to_string(),
                event_sequence,
                final_audio_sequence,
                outcome: UtteranceOutcome::Completed.into(),
            },
        )),
    }
}

#[test]
fn authenticated_request_accepts_only_exact_lowercase_token() {
    let request = authenticated_request((), TOKEN).expect("valid token");
    assert_eq!(
        request
            .metadata()
            .get("authorization")
            .expect("authorization metadata")
            .to_str()
            .unwrap(),
        format!("Bearer {TOKEN}")
    );

    for invalid in [
        "short".to_owned(),
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        assert!(authenticated_request((), &invalid).is_err());
    }
}

#[test]
fn validator_accepts_every_typed_event_with_strict_ordering() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let next_utterance_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, true));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();
    validator.record_input(next_utterance_id, 0).unwrap();

    let events = [
        transcript(session_id, utterance_id, 2, "debug transcript"),
        translation(session_id, utterance_id, 3, "debug translation"),
        audio(session_id, utterance_id, 4, 0),
        latency(session_id, utterance_id, 5),
        health(session_id, 6),
        provider_error(session_id, 7, "Provider is unavailable"),
        final_event(session_id, utterance_id, 8, Some(0)),
        audio(session_id, next_utterance_id, 9, 0),
    ];
    for event in events {
        validator.validate(&event, 0).unwrap();
    }
}

#[test]
fn validator_keeps_negotiated_input_and_output_formats_distinct() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let input_format = pcm_format();
    let output_format = PcmFormat {
        sample_rate_hz: 24_000,
        channels: 1,
        sample_format: SampleFormat::S16le.into(),
        frame_duration_ms: 100,
    };
    let mut mixed_contract = contract(session_id, TranslationMode::QualityFirst, false);
    mixed_contract.input_format = input_format;
    mixed_contract.output_format = output_format;
    let mut validator = ProviderEventValidator::new(mixed_contract.clone());
    let mut mixed_open = opened(session_id, 1);
    if let Some(provider_event::Event::SessionOpened(value)) = mixed_open.event.as_mut() {
        value.negotiated_input_format = Some(input_format);
        value.negotiated_output_format = Some(output_format);
    }
    validator.validate(&mixed_open, 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();
    let mut mixed_audio = audio(session_id, utterance_id, 2, 0);
    if let Some(provider_event::Event::AudioDelta(value)) = mixed_audio.event.as_mut() {
        value.sample_rate_hz = output_format.sample_rate_hz;
        value.pcm = vec![0; 4_800];
    }
    validator.validate(&mixed_audio, 0).unwrap();

    for mutate_input in [true, false] {
        let mut rejected = ProviderEventValidator::new(mixed_contract.clone());
        let mut mismatched = mixed_open.clone();
        if let Some(provider_event::Event::SessionOpened(value)) = mismatched.event.as_mut() {
            if mutate_input {
                value.negotiated_input_format = Some(output_format);
            } else {
                value.negotiated_output_format = Some(input_format);
            }
        }
        assert_eq!(
            rejected.validate(&mismatched, 0),
            Err(ProviderValidationError::NegotiatedFormatMismatch)
        );
    }
}

#[test]
fn validator_binds_cloud_egress_to_provider_identity() {
    let session_id = Uuid::new_v4();
    let mut local_open = opened(session_id, 1);
    if let Some(provider_event::Event::SessionOpened(value)) = local_open.event.as_mut() {
        value.capabilities.as_mut().unwrap().cloud_egress = true;
    }
    let mut local_validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    assert_eq!(
        local_validator.validate(&local_open, 0).unwrap_err(),
        ProviderValidationError::NegotiatedFormatMismatch
    );

    let mut openai_contract = contract(session_id, TranslationMode::QualityFirst, false);
    openai_contract.provider_id = ProviderId::Openai;
    let mut openai_open = opened(session_id, 1);
    if let Some(provider_event::Event::SessionOpened(value)) = openai_open.event.as_mut() {
        value.capabilities.as_mut().unwrap().cloud_egress = true;
    }
    let mut openai_validator = ProviderEventValidator::new(openai_contract);
    openai_validator.validate(&openai_open, 0).unwrap();
}

#[test]
fn validator_rejects_provider_identity_mismatch_in_health() {
    let session_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    let mut event = health(session_id, 2);
    if let Some(provider_event::Event::Health(value)) = event.event.as_mut() {
        value.provider_id = ProviderId::Openai as i32;
    }

    assert_eq!(
        validator.validate(&event, 0).unwrap_err(),
        ProviderValidationError::ProviderMismatch
    );
}

#[test]
fn validator_rejects_each_boundary_violation_without_advancing_state() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));

    let mut missing_format = opened(session_id, 1);
    if let Some(provider_event::Event::SessionOpened(value)) = missing_format.event.as_mut() {
        value.negotiated_output_format = None;
    }
    assert_eq!(
        validator.validate(&missing_format, 0).unwrap_err(),
        ProviderValidationError::MissingSessionContract
    );
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();

    let duplicate = audio(session_id, utterance_id, 1, 0);
    assert_eq!(
        validator.validate(&duplicate, 0).unwrap_err(),
        ProviderValidationError::DuplicateSequence
    );
    let stale = audio(session_id, utterance_id, 0, 0);
    assert_eq!(
        validator.validate(&stale, 0).unwrap_err(),
        ProviderValidationError::StaleSequence
    );

    let mut foreign = audio(session_id, utterance_id, 2, 0);
    if let Some(provider_event::Event::AudioDelta(value)) = foreign.event.as_mut() {
        value.session_id = Uuid::new_v4().to_string();
    }
    assert_eq!(
        validator.validate(&foreign, 0).unwrap_err(),
        ProviderValidationError::SessionMismatch
    );

    let mut wrong_direction = audio(session_id, utterance_id, 2, 0);
    if let Some(provider_event::Event::AudioDelta(value)) = wrong_direction.event.as_mut() {
        value.direction_id = AudioDirection::Speaker.into();
    }
    assert_eq!(
        validator.validate(&wrong_direction, 0).unwrap_err(),
        ProviderValidationError::DirectionMismatch
    );

    let mut wrong_schema = audio(session_id, utterance_id, 2, 0);
    if let Some(provider_event::Event::AudioDelta(value)) = wrong_schema.event.as_mut() {
        value.schema_version = "private-schema-marker".into();
    }
    assert_eq!(
        validator.validate(&wrong_schema, 0).unwrap_err(),
        ProviderValidationError::SchemaMismatch
    );

    let mut malformed = audio(session_id, utterance_id, 2, 0);
    if let Some(provider_event::Event::AudioDelta(value)) = malformed.event.as_mut() {
        value.utterance_id = "not-a-uuid".into();
    }
    assert_eq!(
        validator.validate(&malformed, 0).unwrap_err(),
        ProviderValidationError::InvalidIdentifier
    );
    assert_eq!(
        validator
            .validate(&ProviderEvent { event: None }, 0)
            .unwrap_err(),
        ProviderValidationError::MissingEvent
    );

    validator
        .validate(&audio(session_id, utterance_id, 2, 0), 0)
        .unwrap();
    assert_eq!(
        validator
            .validate(&audio(session_id, Uuid::new_v4(), 3, 0), 0)
            .unwrap_err(),
        ProviderValidationError::UnknownUtterance
    );
}

#[test]
fn validator_requires_open_first_once_and_monotonic_event_sequences() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    assert_eq!(
        validator.validate(&health(session_id, 1), 0).unwrap_err(),
        ProviderValidationError::OpenRequired
    );
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();
    assert_eq!(
        validator.validate(&opened(session_id, 2), 0).unwrap_err(),
        ProviderValidationError::DuplicateOpen
    );
    validator
        .validate(&audio(session_id, utterance_id, 3, 0), 0)
        .unwrap();
    assert_eq!(
        validator.validate(&health(session_id, 3), 0).unwrap_err(),
        ProviderValidationError::DuplicateSequence
    );
    assert_eq!(
        validator.validate(&health(session_id, 2), 0).unwrap_err(),
        ProviderValidationError::StaleSequence
    );
}

#[test]
fn validator_applies_identity_and_schema_checks_to_every_event_variant() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let variants = vec![
        transcript(session_id, utterance_id, 2, "debug"),
        translation(session_id, utterance_id, 2, "debug"),
        audio(session_id, utterance_id, 2, 0),
        latency(session_id, utterance_id, 2),
        health(session_id, 2),
        provider_error(session_id, 2, "Provider is unavailable"),
        final_event(session_id, utterance_id, 2, None),
        ProviderEvent {
            event: Some(provider_event::Event::SessionClosed(
                ProviderSessionClosed {
                    schema_version: "translator.provider.session_closed.v1".into(),
                    session_id: session_id.to_string(),
                    direction_id: AudioDirection::Microphone.into(),
                    event_sequence: 2,
                    reason: SessionCloseReason::UserStop.into(),
                },
            )),
        },
    ];

    for event in variants {
        for expected in [
            ProviderValidationError::SchemaMismatch,
            ProviderValidationError::SessionMismatch,
            ProviderValidationError::DirectionMismatch,
        ] {
            let mut mutated = event.clone();
            mutate_common_identity(&mut mutated, &expected);
            let mut validator = ProviderEventValidator::new(contract(
                session_id,
                TranslationMode::QualityFirst,
                true,
            ));
            validator.validate(&opened(session_id, 1), 0).unwrap();
            validator.record_input(utterance_id, 0).unwrap();
            assert_eq!(validator.validate(&mutated, 0).unwrap_err(), expected);
        }
    }

    let mut wrong_open_schema = opened(session_id, 1);
    mutate_common_identity(
        &mut wrong_open_schema,
        &ProviderValidationError::SchemaMismatch,
    );
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, true));
    assert_eq!(
        validator.validate(&wrong_open_schema, 0).unwrap_err(),
        ProviderValidationError::SchemaMismatch
    );
    for expected in [
        ProviderValidationError::SessionMismatch,
        ProviderValidationError::DirectionMismatch,
    ] {
        let mut wrong_open = opened(session_id, 1);
        mutate_common_identity(&mut wrong_open, &expected);
        let mut validator =
            ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, true));
        assert_eq!(validator.validate(&wrong_open, 0).unwrap_err(), expected);
    }
}

fn mutate_common_identity(event: &mut ProviderEvent, expected: &ProviderValidationError) {
    let (schema, session_id, direction_id) = match event.event.as_mut().unwrap() {
        provider_event::Event::SessionOpened(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
        provider_event::Event::AudioDelta(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
        provider_event::Event::TranscriptDelta(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
        provider_event::Event::TranslationDelta(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
        provider_event::Event::UtteranceFinal(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
        provider_event::Event::SessionClosed(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
        provider_event::Event::Health(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
        provider_event::Event::Latency(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
        provider_event::Event::Error(value) => (
            &mut value.schema_version,
            &mut value.session_id,
            &mut value.direction_id,
        ),
    };
    match expected {
        ProviderValidationError::SchemaMismatch => *schema = "private-schema-marker".into(),
        ProviderValidationError::SessionMismatch => *session_id = Uuid::new_v4().to_string(),
        ProviderValidationError::DirectionMismatch => {
            *direction_id = AudioDirection::Speaker.into()
        }
        _ => unreachable!(),
    }
}

#[test]
fn validator_rejects_incomplete_session_and_output_pcm_contracts() {
    let session_id = Uuid::new_v4();
    for mutate in 0..4 {
        let mut event = opened(session_id, 1);
        if let Some(provider_event::Event::SessionOpened(value)) = event.event.as_mut() {
            match mutate {
                0 => value.negotiated_input_format = None,
                1 => value.negotiated_output_format = None,
                2 => value.capabilities = None,
                3 => {
                    value
                        .negotiated_output_format
                        .as_mut()
                        .unwrap()
                        .sample_rate_hz = 24_000
                }
                _ => unreachable!(),
            }
        }
        let mut validator =
            ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
        let expected = if mutate == 3 {
            ProviderValidationError::NegotiatedFormatMismatch
        } else {
            ProviderValidationError::MissingSessionContract
        };
        assert_eq!(validator.validate(&event, 0).unwrap_err(), expected);
        validator.validate(&opened(session_id, 1), 0).unwrap();
    }

    let utterance_id = Uuid::new_v4();
    for mutate in 0..2 {
        let mut invalid = audio(session_id, utterance_id, 2, 0);
        if let Some(provider_event::Event::AudioDelta(value)) = invalid.event.as_mut() {
            if mutate == 0 {
                value.sample_rate_hz = 24_000;
            } else {
                value.pcm.pop();
            }
        }
        let mut validator =
            ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
        validator.validate(&opened(session_id, 1), 0).unwrap();
        validator.record_input(utterance_id, 0).unwrap();
        assert_eq!(
            validator.validate(&invalid, 0).unwrap_err(),
            if mutate == 0 {
                ProviderValidationError::OutputFormatMismatch
            } else {
                ProviderValidationError::PcmLengthMismatch
            }
        );
        validator
            .validate(&audio(session_id, utterance_id, 2, 0), 0)
            .unwrap();
    }
}

#[test]
fn queue_overflow_error_requires_immediate_matching_terminal() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let unrelated_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();
    validator.record_input(unrelated_id, 0).unwrap();
    validator
        .validate(&queue_overflow_error(session_id, utterance_id, 2), 0)
        .unwrap();
    let mut unrelated_terminal = final_event(session_id, unrelated_id, 3, None);
    if let Some(provider_event::Event::UtteranceFinal(value)) = unrelated_terminal.event.as_mut() {
        value.outcome = UtteranceOutcome::Dropped.into();
    }
    assert_eq!(
        validator.validate(&unrelated_terminal, 0).unwrap_err(),
        ProviderValidationError::ExpectedOverflowTerminal
    );
    assert_eq!(
        validator
            .validate(&final_event(session_id, utterance_id, 3, None), 0)
            .unwrap_err(),
        ProviderValidationError::ExpectedOverflowTerminal
    );
    let mut terminal = final_event(session_id, utterance_id, 3, None);
    if let Some(provider_event::Event::UtteranceFinal(value)) = terminal.event.as_mut() {
        value.outcome = UtteranceOutcome::Dropped.into();
    }
    validator.validate(&terminal, 0).unwrap();
}

#[test]
fn validator_enforces_debug_privacy_and_static_error_messages() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let marker = "private-rust-validator-marker";
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();

    assert_eq!(
        validator
            .validate(&transcript(session_id, utterance_id, 2, marker), 0)
            .unwrap_err(),
        ProviderValidationError::DebugTextDisabled
    );
    assert_eq!(
        validator
            .validate(&provider_error(session_id, 2, marker), 0)
            .unwrap_err(),
        ProviderValidationError::UnsafeErrorMessage
    );
    validator.set_debug_text_enabled(true);
    validator
        .validate(
            &transcript(session_id, utterance_id, 2, "visible only in debug"),
            0,
        )
        .unwrap();
    validator.set_debug_text_enabled(false);
    assert_eq!(
        validator
            .validate(&translation(session_id, utterance_id, 3, marker), 0)
            .unwrap_err(),
        ProviderValidationError::DebugTextDisabled
    );
    validator
        .validate(&provider_error(session_id, 3, "Provider is unavailable"), 0)
        .unwrap();
}

#[test]
fn validator_enforces_mode_age_at_exact_boundary() {
    for (mode, deadline_ms) in [
        (TranslationMode::QualityFirst, 3000_u64),
        (TranslationMode::Balanced, 2000),
        (TranslationMode::StreamingFirst, 1000),
    ] {
        let session_id = Uuid::new_v4();
        let accepted_id = Uuid::new_v4();
        let mut accepted = ProviderEventValidator::new(contract(session_id, mode, false));
        accepted.validate(&opened(session_id, 1), 0).unwrap();
        accepted.record_input(accepted_id, 0).unwrap();
        accepted
            .validate(
                &audio(session_id, accepted_id, 2, 0),
                deadline_ms * 1_000_000,
            )
            .unwrap();

        let expired_id = Uuid::new_v4();
        let mut expired = ProviderEventValidator::new(contract(session_id, mode, false));
        expired.validate(&opened(session_id, 1), 0).unwrap();
        expired.record_input(expired_id, 0).unwrap();
        assert_eq!(
            expired
                .validate(
                    &audio(session_id, expired_id, 2, 0),
                    deadline_ms * 1_000_000 + 1,
                )
                .unwrap_err(),
            ProviderValidationError::ExpiredAudio
        );
        let mut dropped = final_event(session_id, expired_id, 3, None);
        if let Some(provider_event::Event::UtteranceFinal(value)) = dropped.event.as_mut() {
            value.outcome = UtteranceOutcome::Dropped.into();
        }
        expired
            .validate(&dropped, deadline_ms * 1_000_000 + 1)
            .unwrap();
    }
}

#[test]
fn validator_measures_stale_work_from_end_of_utterance() {
    let session_id = Uuid::new_v4();
    let accepted_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(accepted_id, 0).unwrap();
    validator
        .record_end_of_utterance(accepted_id, 2_000_000_000)
        .unwrap();
    validator
        .validate(&audio(session_id, accepted_id, 2, 0), 5_000_000_000)
        .unwrap();

    let expired_id = Uuid::new_v4();
    let mut expired =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    expired.validate(&opened(session_id, 1), 0).unwrap();
    expired.record_input(expired_id, 0).unwrap();
    expired
        .record_end_of_utterance(expired_id, 2_000_000_000)
        .unwrap();
    assert_eq!(
        expired.validate(&audio(session_id, expired_id, 2, 0), 5_000_000_001,),
        Err(ProviderValidationError::ExpiredAudio)
    );
}

#[test]
fn validator_applies_capture_age_only_to_first_playable_audio() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::StreamingFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();

    validator
        .validate(&audio(session_id, utterance_id, 2, 0), 1_000_000_000)
        .unwrap();
    for (event_sequence, audio_sequence, now_ns) in [
        (3, 1, 1_250_000_000),
        (4, 2, 1_500_000_000),
        (5, 3, 1_750_000_000),
        (6, 4, 2_000_000_000),
        (7, 5, 2_250_000_000),
    ] {
        validator
            .validate(
                &audio(session_id, utterance_id, event_sequence, audio_sequence),
                now_ns,
            )
            .unwrap();
    }
    validator
        .validate(
            &final_event(session_id, utterance_id, 8, Some(5)),
            2_250_000_000,
        )
        .unwrap();
}

#[test]
fn validator_cancel_pending_accepts_only_one_matching_cancelled_final() {
    let session_id = Uuid::new_v4();
    let cancelled_id = Uuid::new_v4();
    let unrelated_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::Balanced, true));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(cancelled_id, 0).unwrap();
    validator.record_input(unrelated_id, 0).unwrap();
    validator.record_cancel_requested(cancelled_id).unwrap();

    validator
        .validate(&audio(session_id, unrelated_id, 2, 0), 0)
        .unwrap();
    for rejected in [
        transcript(session_id, cancelled_id, 3, "not publishable"),
        translation(session_id, cancelled_id, 3, "not publishable"),
        latency(session_id, cancelled_id, 3),
        queue_overflow_error(session_id, cancelled_id, 3),
        final_event(session_id, cancelled_id, 3, None),
    ] {
        assert_eq!(
            validator.validate(&rejected, 0),
            Err(ProviderValidationError::ExpectedCancelledTerminal)
        );
    }
    assert_eq!(
        validator.validate(&audio(session_id, cancelled_id, 3, 0), 0),
        Err(ProviderValidationError::CancelledAudio)
    );

    let mut cancelled = final_event(session_id, cancelled_id, 4, Some(0));
    if let Some(provider_event::Event::UtteranceFinal(value)) = cancelled.event.as_mut() {
        value.outcome = UtteranceOutcome::Cancelled.into();
    }
    validator.validate(&cancelled, 0).unwrap();
    if let Some(provider_event::Event::UtteranceFinal(value)) = cancelled.event.as_mut() {
        value.event_sequence = 5;
    }
    assert_eq!(
        validator.validate(&cancelled, 0),
        Err(ProviderValidationError::UtteranceTerminal)
    );
}

#[test]
fn validator_rejects_unsolicited_cancelled_final() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::Balanced, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();

    let mut cancelled = final_event(session_id, utterance_id, 2, None);
    if let Some(provider_event::Event::UtteranceFinal(value)) = cancelled.event.as_mut() {
        value.outcome = UtteranceOutcome::Cancelled.into();
    }
    assert_eq!(
        validator.validate(&cancelled, 0),
        Err(ProviderValidationError::ExpectedCancelledTerminal)
    );
    validator
        .validate(&final_event(session_id, utterance_id, 2, None), 0)
        .unwrap();
}

#[test]
fn validator_enforces_final_audio_sequence_and_terminal_scope() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let next_id = Uuid::new_v4();
    let no_audio_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    for id in [utterance_id, next_id, no_audio_id] {
        validator.record_input(id, 0).unwrap();
    }
    validator
        .validate(&audio(session_id, utterance_id, 2, 0), 0)
        .unwrap();
    assert_eq!(
        validator
            .validate(&final_event(session_id, utterance_id, 3, None), 0)
            .unwrap_err(),
        ProviderValidationError::FinalAudioSequenceMismatch
    );
    validator
        .validate(&final_event(session_id, utterance_id, 3, Some(0)), 0)
        .unwrap();
    assert_eq!(
        validator
            .validate(&audio(session_id, utterance_id, 4, 1), 0)
            .unwrap_err(),
        ProviderValidationError::UtteranceTerminal
    );
    validator
        .validate(&audio(session_id, next_id, 4, 0), 0)
        .unwrap();
    assert_eq!(
        validator
            .validate(&final_event(session_id, no_audio_id, 5, Some(0)), 0)
            .unwrap_err(),
        ProviderValidationError::FinalAudioSequenceMismatch
    );
    validator
        .validate(&final_event(session_id, no_audio_id, 5, None), 0)
        .unwrap();
}

#[test]
fn validator_rejects_events_after_session_close() {
    let session_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator
        .validate(
            &ProviderEvent {
                event: Some(provider_event::Event::SessionClosed(
                    ProviderSessionClosed {
                        schema_version: "translator.provider.session_closed.v1".into(),
                        session_id: session_id.to_string(),
                        direction_id: AudioDirection::Microphone.into(),
                        event_sequence: 2,
                        reason: SessionCloseReason::UserStop.into(),
                    },
                )),
            },
            0,
        )
        .unwrap();

    assert_eq!(
        validator.validate(&health(session_id, 3), 0).unwrap_err(),
        ProviderValidationError::SessionTerminal
    );
}

#[test]
fn validator_rejects_cross_wired_stream_identity_on_every_stream_event() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let wrong_stream = Uuid::new_v4().to_string();
    let mut events = vec![
        audio(session_id, utterance_id, 2, 0),
        transcript(session_id, utterance_id, 2, "debug"),
        translation(session_id, utterance_id, 2, "debug"),
        latency(session_id, utterance_id, 2),
        final_event(session_id, utterance_id, 2, None),
        provider_error(session_id, 2, "Provider is unavailable"),
    ];
    for event in &mut events {
        match event.event.as_mut().unwrap() {
            provider_event::Event::AudioDelta(value) => value.stream_id = wrong_stream.clone(),
            provider_event::Event::TranscriptDelta(value) => value.stream_id = wrong_stream.clone(),
            provider_event::Event::TranslationDelta(value) => {
                value.stream_id = wrong_stream.clone()
            }
            provider_event::Event::Latency(value) => value.stream_id = wrong_stream.clone(),
            provider_event::Event::UtteranceFinal(value) => value.stream_id = wrong_stream.clone(),
            provider_event::Event::Error(value) => value.stream_id = Some(wrong_stream.clone()),
            _ => unreachable!(),
        }
        let mut validator =
            ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, true));
        validator.validate(&opened(session_id, 1), 0).unwrap();
        validator.record_input(utterance_id, 0).unwrap();
        assert_eq!(
            validator.validate(event, 0).unwrap_err(),
            ProviderValidationError::StreamMismatch
        );
    }
}

#[test]
fn validator_requires_contiguous_audio_sequence_and_valid_terminal_outcome() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();
    assert_eq!(
        validator
            .validate(&audio(session_id, utterance_id, 2, 1), 0)
            .unwrap_err(),
        ProviderValidationError::AudioSequenceGap
    );
    validator
        .validate(&audio(session_id, utterance_id, 2, 0), 0)
        .unwrap();
    assert_eq!(
        validator
            .validate(&audio(session_id, utterance_id, 3, 0), 0)
            .unwrap_err(),
        ProviderValidationError::DuplicateAudioSequence
    );
    assert_eq!(
        validator
            .validate(&audio(session_id, utterance_id, 3, 2), 0)
            .unwrap_err(),
        ProviderValidationError::AudioSequenceGap
    );
    validator
        .validate(&audio(session_id, utterance_id, 3, 1), 0)
        .unwrap();

    let mut unspecified = final_event(session_id, utterance_id, 4, Some(1));
    if let Some(provider_event::Event::UtteranceFinal(value)) = unspecified.event.as_mut() {
        value.outcome = UtteranceOutcome::Unspecified.into();
    }
    assert_eq!(
        validator.validate(&unspecified, 0).unwrap_err(),
        ProviderValidationError::InvalidOutcome
    );
}

#[test]
fn validator_bounds_active_and_terminal_utterance_state() {
    let session_id = Uuid::new_v4();
    let mut active =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    active.validate(&opened(session_id, 1), 0).unwrap();
    for _ in 0..MAX_ACTIVE_UTTERANCES {
        active.record_input(Uuid::new_v4(), 0).unwrap();
    }
    assert_eq!(
        active.record_input(Uuid::new_v4(), 0).unwrap_err(),
        ProviderValidationError::UtteranceCapacityExceeded
    );

    let mut terminal =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    terminal.validate(&opened(session_id, 1), 0).unwrap();
    let first = Uuid::new_v4();
    for (index, utterance_id) in std::iter::once(first)
        .chain((1..MAX_TERMINAL_UTTERANCES).map(|_| Uuid::new_v4()))
        .enumerate()
    {
        terminal.record_input(utterance_id, 0).unwrap();
        let mut dropped = final_event(session_id, utterance_id, index as u64 + 2, None);
        if let Some(provider_event::Event::UtteranceFinal(value)) = dropped.event.as_mut() {
            value.outcome = UtteranceOutcome::Dropped.into();
        }
        terminal.validate(&dropped, 0).unwrap();
    }
    assert_eq!(
        terminal.record_input(first, 0).unwrap_err(),
        ProviderValidationError::UtteranceTerminal
    );
    let evicting = Uuid::new_v4();
    terminal.record_input(evicting, 0).unwrap();
    let mut dropped = final_event(
        session_id,
        evicting,
        MAX_TERMINAL_UTTERANCES as u64 + 2,
        None,
    );
    if let Some(provider_event::Event::UtteranceFinal(value)) = dropped.event.as_mut() {
        value.outcome = UtteranceOutcome::Dropped.into();
    }
    assert_eq!(
        terminal.validate(&dropped, 0).unwrap_err(),
        ProviderValidationError::TerminalCapacityExceeded
    );
    assert_eq!(
        terminal.record_input(first, 0).unwrap_err(),
        ProviderValidationError::SessionTerminal
    );
    assert_eq!(
        terminal.record_input(Uuid::new_v4(), 0).unwrap_err(),
        ProviderValidationError::SessionTerminal
    );
}

#[test]
fn validator_rejects_content_derived_health_safe_error_text() {
    let session_id = Uuid::new_v4();
    let marker = "private-health-safe-error-marker";
    let mut unsafe_health = health(session_id, 2);
    if let Some(provider_event::Event::Health(value)) = unsafe_health.event.as_mut() {
        value.safe_error = Some(SafeErrorSummary {
            code: "provider_unavailable".into(),
            message: marker.into(),
            retryable: true,
        });
    }
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    let mut safe_health = health(session_id, 2);
    if let Some(provider_event::Event::Health(value)) = safe_health.event.as_mut() {
        value.safe_error = Some(SafeErrorSummary {
            code: "provider_unavailable".into(),
            message: "Provider is unavailable".into(),
            retryable: true,
        });
    }
    validator.validate(&safe_health, 0).unwrap();
    if let Some(provider_event::Event::Health(value)) = unsafe_health.event.as_mut() {
        value.event_sequence = 3;
    }
    assert_eq!(
        validator.validate(&unsafe_health, 0).unwrap_err(),
        ProviderValidationError::UnsafeErrorMessage
    );
    let mut unknown = health(session_id, 3);
    if let Some(provider_event::Event::Health(value)) = unknown.event.as_mut() {
        value.safe_error = Some(SafeErrorSummary {
            code: "private_unknown_code".into(),
            message: "Provider is unavailable".into(),
            retryable: true,
        });
    }
    assert_eq!(
        validator.validate(&unknown, 0).unwrap_err(),
        ProviderValidationError::UnsafeErrorMessage
    );
}

#[test]
fn validator_rejects_errors_for_unknown_and_terminal_utterances() {
    let session_id = Uuid::new_v4();
    let known = Uuid::new_v4();
    let unknown = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(known, 0).unwrap();

    let mut unknown_error = provider_error(session_id, 2, "Provider is unavailable");
    if let Some(provider_event::Event::Error(value)) = unknown_error.event.as_mut() {
        value.stream_id = Some(Uuid::nil().to_string());
        value.utterance_id = Some(unknown.to_string());
    }
    assert_eq!(
        validator.validate(&unknown_error, 0).unwrap_err(),
        ProviderValidationError::UnknownUtterance
    );

    validator
        .validate(&final_event(session_id, known, 2, None), 0)
        .unwrap();
    let mut terminal_error = provider_error(session_id, 3, "Provider is unavailable");
    if let Some(provider_event::Event::Error(value)) = terminal_error.event.as_mut() {
        value.stream_id = Some(Uuid::nil().to_string());
        value.utterance_id = Some(known.to_string());
    }
    assert_eq!(
        validator.validate(&terminal_error, 0).unwrap_err(),
        ProviderValidationError::UtteranceTerminal
    );
}

#[test]
fn validator_rejects_unspecified_provider_error_code() {
    let session_id = Uuid::new_v4();
    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    let mut unspecified = provider_error(session_id, 2, "");
    if let Some(provider_event::Event::Error(value)) = unspecified.event.as_mut() {
        value.code = SafeErrorCode::Unspecified.into();
    }
    assert_eq!(
        validator.validate(&unspecified, 0).unwrap_err(),
        ProviderValidationError::UnsafeErrorMessage
    );
}

#[test]
fn no_speech_has_stable_protobuf_value() {
    assert_eq!(SafeErrorCode::NoSpeech as i32, 8);
}

#[test]
fn validator_accepts_only_privacy_safe_scoped_no_speech_drop() {
    let session_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let no_speech = |event_sequence, safe_message: &str| ProviderEvent {
        event: Some(provider_event::Event::Error(ProviderError {
            schema_version: "translator.provider.error.v1".into(),
            session_id: session_id.to_string(),
            direction_id: AudioDirection::Microphone.into(),
            stream_id: Some(Uuid::nil().to_string()),
            event_sequence,
            code: SafeErrorCode::NoSpeech.into(),
            retryable: true,
            safe_message: safe_message.into(),
            utterance_id: Some(utterance_id.to_string()),
        })),
    };

    let mut validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    validator.validate(&opened(session_id, 1), 0).unwrap();
    validator.record_input(utterance_id, 0).unwrap();
    validator
        .validate(&no_speech_latency(session_id, utterance_id, 2), 0)
        .unwrap();
    validator
        .validate(&no_speech(3, "No speech was detected"), 0)
        .unwrap();
    let mut dropped = final_event(session_id, utterance_id, 4, None);
    if let Some(provider_event::Event::UtteranceFinal(value)) = dropped.event.as_mut() {
        value.outcome = UtteranceOutcome::Dropped.into();
    }
    validator.validate(&dropped, 0).unwrap();

    let mut unsafe_validator =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    unsafe_validator
        .validate(&opened(session_id, 1), 0)
        .unwrap();
    unsafe_validator.record_input(utterance_id, 0).unwrap();
    unsafe_validator
        .validate(&no_speech_latency(session_id, utterance_id, 2), 0)
        .unwrap();
    assert_eq!(
        unsafe_validator
            .validate(&no_speech(3, "private-spoken-marker"), 0)
            .unwrap_err(),
        ProviderValidationError::UnsafeErrorMessage
    );

    let prepared = || {
        let mut validator =
            ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
        validator.validate(&opened(session_id, 1), 0).unwrap();
        validator.record_input(utterance_id, 0).unwrap();
        validator
            .validate(&no_speech_latency(session_id, utterance_id, 2), 0)
            .unwrap();
        validator
    };

    let mut missing_scope = prepared();
    let mut unscoped = no_speech(3, "No speech was detected");
    if let Some(provider_event::Event::Error(value)) = unscoped.event.as_mut() {
        value.stream_id = None;
        value.utterance_id = None;
    }
    assert_eq!(
        missing_scope.validate(&unscoped, 0).unwrap_err(),
        ProviderValidationError::ExpectedNoSpeechTerminal
    );

    let mut interposed = prepared();
    interposed
        .validate(&no_speech(3, "No speech was detected"), 0)
        .unwrap();
    assert_eq!(
        interposed
            .validate(&audio(session_id, utterance_id, 4, 0), 0)
            .unwrap_err(),
        ProviderValidationError::ExpectedNoSpeechTerminal
    );

    let mut mismatched = prepared();
    let other_utterance = Uuid::new_v4();
    mismatched.record_input(other_utterance, 0).unwrap();
    mismatched
        .validate(&no_speech(3, "No speech was detected"), 0)
        .unwrap();
    let mut wrong_final = final_event(session_id, other_utterance, 4, None);
    if let Some(provider_event::Event::UtteranceFinal(value)) = wrong_final.event.as_mut() {
        value.outcome = UtteranceOutcome::Dropped.into();
    }
    assert_eq!(
        mismatched.validate(&wrong_final, 0).unwrap_err(),
        ProviderValidationError::ExpectedNoSpeechTerminal
    );

    let mut completed = prepared();
    completed
        .validate(&no_speech(3, "No speech was detected"), 0)
        .unwrap();
    assert_eq!(
        completed
            .validate(&final_event(session_id, utterance_id, 4, None), 0)
            .unwrap_err(),
        ProviderValidationError::ExpectedNoSpeechTerminal
    );

    let mut missing_latency =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    missing_latency.validate(&opened(session_id, 1), 0).unwrap();
    missing_latency.record_input(utterance_id, 0).unwrap();
    assert_eq!(
        missing_latency
            .validate(&no_speech(2, "No speech was detected"), 0)
            .unwrap_err(),
        ProviderValidationError::ExpectedNoSpeechLatency
    );

    let mut non_asr_latency =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    non_asr_latency.validate(&opened(session_id, 1), 0).unwrap();
    non_asr_latency.record_input(utterance_id, 0).unwrap();
    non_asr_latency
        .validate(&latency(session_id, utterance_id, 2), 0)
        .unwrap();
    assert_eq!(
        non_asr_latency
            .validate(&no_speech(3, "No speech was detected"), 0)
            .unwrap_err(),
        ProviderValidationError::ExpectedNoSpeechLatency
    );

    let mut interposed_health =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    interposed_health
        .validate(&opened(session_id, 1), 0)
        .unwrap();
    interposed_health.record_input(utterance_id, 0).unwrap();
    interposed_health
        .validate(&no_speech_latency(session_id, utterance_id, 2), 0)
        .unwrap();
    interposed_health
        .validate(&health(session_id, 3), 0)
        .unwrap();
    assert_eq!(
        interposed_health
            .validate(&no_speech(4, "No speech was detected"), 0)
            .unwrap_err(),
        ProviderValidationError::ExpectedNoSpeechLatency
    );

    let other_utterance = Uuid::new_v4();
    let mut mismatched_latency =
        ProviderEventValidator::new(contract(session_id, TranslationMode::QualityFirst, false));
    mismatched_latency
        .validate(&opened(session_id, 1), 0)
        .unwrap();
    mismatched_latency.record_input(utterance_id, 0).unwrap();
    mismatched_latency.record_input(other_utterance, 0).unwrap();
    mismatched_latency
        .validate(&no_speech_latency(session_id, utterance_id, 2), 0)
        .unwrap();
    let mut wrong_utterance = no_speech(3, "No speech was detected");
    if let Some(provider_event::Event::Error(value)) = wrong_utterance.event.as_mut() {
        value.utterance_id = Some(other_utterance.to_string());
    }
    assert_eq!(
        mismatched_latency
            .validate(&wrong_utterance, 0)
            .unwrap_err(),
        ProviderValidationError::ExpectedNoSpeechLatency
    );
}
