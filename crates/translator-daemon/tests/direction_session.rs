use translator_audio::{CaptureEvent, PcmFrame, StreamPcmFormat};
use translator_core::{
    AudioDirection, Language, ProviderId, TranslationMode, VoiceEngine, VoiceGender,
};
use translator_daemon::{
    DirectionEffect, DirectionRuntimeConfig, DirectionSession, DirectionSessionError,
    DirectionWatchdogEffect, SafeProviderErrorCode, TerminalOutcome,
};
use translator_ipc::provider::{
    AudioDirection as ProviderDirection, ProviderAudioDelta, ProviderCapabilities, ProviderError,
    ProviderEvent, ProviderId as ProviderProviderId, ProviderLatency, ProviderSessionOpened,
    ProviderUtteranceFinal, SafeErrorCode, SampleFormat, UtteranceOutcome, provider_event,
    provider_request,
};
use uuid::Uuid;

fn config() -> DirectionRuntimeConfig {
    DirectionRuntimeConfig {
        provider_id: ProviderId::Local,
        direction: AudioDirection::Microphone,
        source_language: Language::Ru,
        target_language: Language::En,
        mode: TranslationMode::QualityFirst,
        voice_gender: VoiceGender::Male,
        voice_engine: VoiceEngine::Piper,
        debug_text_enabled: false,
    }
}

fn frame(sequence: u64) -> PcmFrame {
    PcmFrame::try_new(
        sequence,
        1_000_000_000 + sequence * 20_000_000,
        StreamPcmFormat::provider_default(),
        vec![0; 640],
    )
    .unwrap()
}

fn opened(session: &DirectionSession) -> ProviderEvent {
    ProviderEvent {
        event: Some(provider_event::Event::SessionOpened(
            ProviderSessionOpened {
                schema_version: "translator.provider.session_opened.v1".into(),
                session_id: session.session_id().to_string(),
                direction_id: ProviderDirection::Microphone.into(),
                negotiated_input_format: Some(session.provider_contract().input_format),
                negotiated_output_format: Some(session.provider_contract().output_format),
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

#[test]
fn open_request_carries_provider_identity() {
    let mut config = config();
    config.provider_id = ProviderId::Openai;
    let session = DirectionSession::new(config);
    let request = session.open_request();
    let Some(provider_request::Request::OpenSession(open)) = request.request else {
        panic!("expected open session request");
    };

    assert_eq!(
        session.provider_contract().provider_id,
        ProviderProviderId::Openai
    );
    assert_eq!(open.provider_id, ProviderProviderId::Openai as i32);
}

#[test]
fn direction_session_keeps_stream_stable_rotates_utterances_and_sequences_globally() {
    let mut session = DirectionSession::new(config());
    let stream_id = session.stream_id();
    session.handle_provider_event(&opened(&session), 0).unwrap();

    let first_utterance = Uuid::new_v4();
    session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id,
            utterance_id: first_utterance,
            capture_monotonic_ns: 1_000_000_000,
        })
        .unwrap();
    let first = session
        .handle_capture(CaptureEvent::Frame {
            stream_id,
            utterance_id: first_utterance,
            frame: frame(10),
            end_of_utterance: false,
        })
        .unwrap()
        .unwrap();
    let second = session
        .handle_capture(CaptureEvent::Frame {
            stream_id,
            utterance_id: first_utterance,
            frame: frame(11),
            end_of_utterance: true,
        })
        .unwrap()
        .unwrap();

    let second_utterance = Uuid::new_v4();
    session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id,
            utterance_id: second_utterance,
            capture_monotonic_ns: 2_000_000_000,
        })
        .unwrap();
    let third = session
        .handle_capture(CaptureEvent::Frame {
            stream_id,
            utterance_id: second_utterance,
            frame: frame(20),
            end_of_utterance: false,
        })
        .unwrap()
        .unwrap();

    let frames = [first, second, third].map(|request| match request.request.unwrap() {
        provider_request::Request::InputFrame(frame) => frame,
        _ => panic!("capture must produce provider input"),
    });
    assert_eq!(
        frames
            .iter()
            .map(|frame| frame.sequence)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert!(
        frames
            .iter()
            .all(|frame| frame.stream_id == stream_id.to_string())
    );
    assert_eq!(frames[0].utterance_id, first_utterance.to_string());
    assert_eq!(frames[2].utterance_id, second_utterance.to_string());
    assert!(!frames[0].end_of_utterance);
    assert!(frames[1].end_of_utterance);
}

#[test]
fn direction_session_starts_provider_deadline_when_eou_frame_is_sent() {
    let mut session = DirectionSession::new(config());
    let stream_id = session.stream_id();
    let utterance_id = Uuid::new_v4();
    session.handle_provider_event(&opened(&session), 0).unwrap();
    session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id,
            utterance_id,
            capture_monotonic_ns: 1_000_000_000,
        })
        .unwrap();

    assert!(session.poll(4_000_000_001).unwrap().is_empty());
    session
        .handle_capture(CaptureEvent::Frame {
            stream_id,
            utterance_id,
            frame: frame(150),
            end_of_utterance: true,
        })
        .unwrap();

    assert!(session.poll(10_000_000_000).unwrap().is_empty());
    assert!(matches!(
        session.poll(10_000_000_001).unwrap().as_slice(),
        [DirectionWatchdogEffect::PurgeAndSend(_)]
    ));
}

#[test]
fn collecting_timeout_releases_capture_for_the_next_utterance() {
    let mut session = DirectionSession::new(config());
    let stream_id = session.stream_id();
    session.handle_provider_event(&opened(&session), 0).unwrap();
    session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id,
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: 0,
        })
        .unwrap();

    assert!(matches!(
        session.poll(12_000_000_001).unwrap().as_slice(),
        [DirectionWatchdogEffect::PurgeAndSend(_)]
    ));
    assert!(
        session
            .handle_capture(CaptureEvent::SpeechStarted {
                stream_id,
                utterance_id: Uuid::new_v4(),
                capture_monotonic_ns: 12_000_000_002,
            })
            .is_ok()
    );
}

#[test]
fn direction_session_validates_and_exposes_provider_audio_for_playback() {
    let mut session = DirectionSession::new(config());
    let stream_id = session.stream_id();
    let utterance_id = Uuid::new_v4();
    session.handle_provider_event(&opened(&session), 0).unwrap();
    session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id,
            utterance_id,
            capture_monotonic_ns: 1_000_000_000,
        })
        .unwrap();
    session
        .handle_capture(CaptureEvent::Frame {
            stream_id,
            utterance_id,
            frame: frame(0),
            end_of_utterance: true,
        })
        .unwrap();
    let event = ProviderEvent {
        event: Some(provider_event::Event::AudioDelta(ProviderAudioDelta {
            schema_version: "translator.provider.audio_delta.v1".into(),
            session_id: session.session_id().to_string(),
            direction_id: ProviderDirection::Microphone.into(),
            stream_id: stream_id.to_string(),
            utterance_id: utterance_id.to_string(),
            sequence: 0,
            event_sequence: 2,
            provider_monotonic_ns: 1_100_000_000,
            sample_rate_hz: 16_000,
            channels: 1,
            sample_format: SampleFormat::S16le.into(),
            frame_duration_ms: 20,
            pcm: vec![7; 640],
        })),
    };

    let effects = session
        .handle_provider_event(&event, 1_100_000_000)
        .unwrap();
    assert!(matches!(
        effects.as_slice(),
        [DirectionEffect::Playback {
            stream_id: observed_stream,
            utterance_id: observed_utterance,
            frame,
        }] if *observed_stream == stream_id
            && *observed_utterance == utterance_id
            && frame.pcm() == vec![7; 640]
    ));
}

#[test]
fn direction_session_cancels_expired_first_audio_without_playback() {
    let mut session = DirectionSession::new(config());
    let stream_id = session.stream_id();
    let utterance_id = Uuid::new_v4();
    session.handle_provider_event(&opened(&session), 0).unwrap();
    session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id,
            utterance_id,
            capture_monotonic_ns: 1_000_000_000,
        })
        .unwrap();
    session
        .handle_capture(CaptureEvent::Frame {
            stream_id,
            utterance_id,
            frame: frame(0),
            end_of_utterance: true,
        })
        .unwrap();
    let event = ProviderEvent {
        event: Some(provider_event::Event::AudioDelta(ProviderAudioDelta {
            schema_version: "translator.provider.audio_delta.v1".into(),
            session_id: session.session_id().to_string(),
            direction_id: ProviderDirection::Microphone.into(),
            stream_id: stream_id.to_string(),
            utterance_id: utterance_id.to_string(),
            sequence: 0,
            event_sequence: 2,
            provider_monotonic_ns: 4_000_000_001,
            sample_rate_hz: 16_000,
            channels: 1,
            sample_format: SampleFormat::S16le.into(),
            frame_duration_ms: 20,
            pcm: vec![7; 640],
        })),
    };

    let effects = session
        .handle_provider_event(&event, 4_000_000_001)
        .unwrap();
    assert!(matches!(
        effects.as_slice(),
        [DirectionEffect::ExpiredAudio {
            utterance_id: expired_id,
            request,
        }]
            if *expired_id == utterance_id && matches!(
                request.request,
                Some(provider_request::Request::CancelUtterance(_))
            )
    ));
}

#[test]
fn direction_session_rejects_capture_from_another_stream() {
    let mut session = DirectionSession::new(config());
    let error = session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id: Uuid::new_v4(),
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: 0,
        })
        .unwrap_err();
    assert_eq!(error, DirectionSessionError::StreamMismatch);
}

#[test]
fn direction_session_exposes_privacy_safe_provider_latency() {
    let mut session = DirectionSession::new(config());
    session.handle_provider_event(&opened(&session), 0).unwrap();
    let utterance_id = Uuid::new_v4();
    session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id: session.stream_id(),
            utterance_id,
            capture_monotonic_ns: 500_000_000,
        })
        .unwrap();
    session
        .handle_capture(CaptureEvent::Frame {
            stream_id: session.stream_id(),
            utterance_id,
            frame: frame(0),
            end_of_utterance: true,
        })
        .unwrap();
    let event = ProviderEvent {
        event: Some(provider_event::Event::Latency(ProviderLatency {
            schema_version: "translator.provider.latency.v1".into(),
            session_id: session.session_id().to_string(),
            direction_id: ProviderDirection::Microphone.into(),
            stream_id: session.stream_id().to_string(),
            event_sequence: 2,
            utterance_id: Some(utterance_id.to_string()),
            asr_first_text_ms: Some(100),
            asr_final_text_ms: Some(200),
            mt_first_text_ms: Some(300),
            tts_first_audio_ms: Some(400),
            provider_total_ms: Some(500),
        })),
    };

    let effects = session
        .handle_provider_event(&event, 1_000_000_000)
        .unwrap();

    assert!(matches!(
        effects.as_slice(),
        [DirectionEffect::Latency {
            utterance_id: Some(observed),
            tts_first_audio_ms: Some(400),
            provider_total_ms: Some(500),
        }] if *observed == utterance_id
    ));
}

#[test]
fn direction_session_exposes_no_speech_error_and_dropped_terminal_without_message() {
    let mut session = DirectionSession::new(config());
    session.handle_provider_event(&opened(&session), 0).unwrap();
    let utterance_id = Uuid::new_v4();
    session
        .handle_capture(CaptureEvent::SpeechStarted {
            stream_id: session.stream_id(),
            utterance_id,
            capture_monotonic_ns: 500_000_000,
        })
        .unwrap();
    session
        .handle_capture(CaptureEvent::Frame {
            stream_id: session.stream_id(),
            utterance_id,
            frame: frame(0),
            end_of_utterance: true,
        })
        .unwrap();
    let latency = ProviderEvent {
        event: Some(provider_event::Event::Latency(ProviderLatency {
            schema_version: "translator.provider.latency.v1".into(),
            session_id: session.session_id().to_string(),
            direction_id: ProviderDirection::Microphone.into(),
            stream_id: session.stream_id().to_string(),
            event_sequence: 2,
            utterance_id: Some(utterance_id.to_string()),
            asr_first_text_ms: Some(100),
            asr_final_text_ms: Some(200),
            mt_first_text_ms: None,
            tts_first_audio_ms: None,
            provider_total_ms: Some(200),
        })),
    };
    session
        .handle_provider_event(&latency, 1_200_000_000)
        .unwrap();
    let error = ProviderEvent {
        event: Some(provider_event::Event::Error(ProviderError {
            schema_version: "translator.provider.error.v1".into(),
            session_id: session.session_id().to_string(),
            direction_id: ProviderDirection::Microphone.into(),
            stream_id: Some(session.stream_id().to_string()),
            event_sequence: 3,
            code: SafeErrorCode::NoSpeech.into(),
            retryable: true,
            safe_message: "No speech was detected".into(),
            utterance_id: Some(utterance_id.to_string()),
        })),
    };

    assert_eq!(
        session
            .handle_provider_event(&error, 1_200_000_001)
            .unwrap(),
        vec![DirectionEffect::ProviderError {
            utterance_id: Some(utterance_id),
            code: SafeProviderErrorCode::NoSpeech,
            retryable: true,
        }]
    );

    let terminal = ProviderEvent {
        event: Some(provider_event::Event::UtteranceFinal(
            ProviderUtteranceFinal {
                schema_version: "translator.provider.utterance_final.v1".into(),
                session_id: session.session_id().to_string(),
                direction_id: ProviderDirection::Microphone.into(),
                stream_id: session.stream_id().to_string(),
                utterance_id: utterance_id.to_string(),
                event_sequence: 4,
                final_audio_sequence: None,
                outcome: UtteranceOutcome::Dropped.into(),
            },
        )),
    };
    assert_eq!(
        session
            .handle_provider_event(&terminal, 1_200_000_002)
            .unwrap(),
        vec![
            DirectionEffect::UtteranceTerminalOutcome {
                utterance_id,
                outcome: TerminalOutcome::Dropped,
            },
            DirectionEffect::UtteranceTerminal { utterance_id },
        ]
    );
}

#[test]
fn direction_session_validator_rejects_unsafe_provider_message_before_effect() {
    let mut session = DirectionSession::new(config());
    session.handle_provider_event(&opened(&session), 0).unwrap();
    let error = ProviderEvent {
        event: Some(provider_event::Event::Error(ProviderError {
            schema_version: "translator.provider.error.v1".into(),
            session_id: session.session_id().to_string(),
            direction_id: ProviderDirection::Microphone.into(),
            stream_id: None,
            event_sequence: 2,
            code: SafeErrorCode::ProviderUnavailable.into(),
            retryable: true,
            safe_message: "private-spoken-marker".into(),
            utterance_id: None,
        })),
    };

    assert!(session.handle_provider_event(&error, 1).is_err());
}
