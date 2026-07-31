use translator_core::{
    AudioDirection, CloseProviderSession, CloseRequestReason, CloseSessionVersion, Language,
    LatencyPolicyState, ModelHealth, ModelKind, ModelState, OpenProviderSession,
    OpenSessionVersion, PcmFormat, PrivacySafeLogEvent, PrivacySafeProviderError,
    ProviderAudioDelta, ProviderAudioDeltaVersion, ProviderCapabilities, ProviderHealth,
    ProviderHealthVersion, ProviderId, ProviderInputFrame, ProviderInputVersion, ProviderLatency,
    ProviderLatencyVersion, ProviderProbeRequest, ProviderProbeRequestVersion,
    ProviderProbeResponse, ProviderProbeResponseVersion, ProviderQueues, ProviderSessionClosed,
    ProviderSessionOpened, ProviderState, RequiredTrue, SafeErrorCode, SampleFormat,
    SessionCloseReason, SessionClosedVersion, SessionOpenedVersion, TranslationMode,
    UpdateDebugText, UpdateDebugTextVersion, VoiceEngine, VoiceGender, VoiceProfile,
};
use uuid::Uuid;

fn pcm_format() -> PcmFormat {
    PcmFormat::try_new(16_000, 1, SampleFormat::S16Le, 40).expect("valid PCM format")
}

#[test]
fn provider_health_matches_the_versioned_wire_shape() {
    let health = ProviderHealth {
        schema_version: ProviderHealthVersion::V1,
        session_id: Uuid::new_v4(),
        direction_id: AudioDirection::Microphone,
        event_sequence: 7,
        provider_id: ProviderId::Local,
        provider_name: "local-cascade".to_owned(),
        state: ProviderState::Ready,
        models: vec![ModelHealth {
            kind: ModelKind::Asr,
            id: "faster-whisper-small".to_owned(),
            state: ModelState::Ready,
            device: Some(translator_core::ComputeDevice::Cuda),
            safe_error_code: None,
        }],
        queues: ProviderQueues {
            provider_input_buffered_ms: 40,
            provider_output_buffered_ms: 0,
            queue_lag_ms: 12,
        },
        retry: None,
        safe_error: None,
    };

    let value = serde_json::to_value(health).expect("health must serialize");

    assert_eq!(value["schema_version"], "translator.provider.health.v1");
    assert_eq!(value["direction_id"], "microphone");
    assert_eq!(value["event_sequence"], 7);
    assert_eq!(value["state"], "ready");
    assert_eq!(value["queues"]["provider_input_buffered_ms"], 40);
    assert!(value.get("pcm").is_none());
    assert!(value.get("transcript").is_none());
}

#[test]
fn lifecycle_messages_use_distinct_versions_and_stable_identity() {
    let session_id = Uuid::parse_str("8ec8cb30-6881-4896-b413-7649b58cdfb2").expect("fixture UUID");
    let open = OpenProviderSession {
        schema_version: OpenSessionVersion::V1,
        session_id,
        provider_id: ProviderId::Local,
        direction_id: AudioDirection::Microphone,
        source_language: Language::Ru,
        target_language: Language::En,
        mode: TranslationMode::StreamingFirst,
        requested_input_format: pcm_format(),
        requested_output_format: pcm_format(),
        voice_profile: VoiceProfile {
            language: Language::En,
            gender: VoiceGender::Male,
            engine: VoiceEngine::Piper,
            model_path: Some("models/en_US-lessac-medium.onnx".into()),
            provider_voice_id: None,
        },
        debug_text_enabled: false,
    };
    let opened = ProviderSessionOpened {
        schema_version: SessionOpenedVersion::V1,
        session_id,
        direction_id: AudioDirection::Microphone,
        event_sequence: 1,
        negotiated_input_format: pcm_format(),
        negotiated_output_format: pcm_format(),
        capabilities: ProviderCapabilities {
            audio_output: RequiredTrue,
            transcript_delta: false,
            translation_delta: false,
            cancellation: true,
            cloud_egress: false,
        },
    };
    let close = CloseProviderSession {
        schema_version: CloseSessionVersion::V1,
        session_id,
        reason: CloseRequestReason::UserStop,
    };
    let closed = ProviderSessionClosed {
        schema_version: SessionClosedVersion::V1,
        session_id,
        direction_id: AudioDirection::Microphone,
        event_sequence: 9,
        reason: SessionCloseReason::UserStop,
    };

    assert_eq!(open.session_id, opened.session_id);
    assert_eq!(open.session_id, close.session_id);
    assert_eq!(open.provider_id, ProviderId::Local);
    assert_eq!(close.session_id, closed.session_id);
    assert_eq!(opened.capabilities.audio_output, RequiredTrue);
    assert_eq!(
        serde_json::to_value(open.schema_version).unwrap(),
        "translator.provider.open_session.v1"
    );
    assert_eq!(
        serde_json::to_value(opened.schema_version).unwrap(),
        "translator.provider.session_opened.v1"
    );
    assert_eq!(
        serde_json::to_value(close.schema_version).unwrap(),
        "translator.provider.close_session.v1"
    );
    assert_eq!(
        serde_json::to_value(closed.schema_version).unwrap(),
        "translator.provider.session_closed.v1"
    );
    assert_eq!(closed.event_sequence, 9);
    assert_eq!(opened.event_sequence, 1);
    assert!(!opened.capabilities.cloud_egress);

    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/contract-fixtures/open_session.json"
    ))
    .expect("golden open-session payload");
    assert_eq!(serde_json::to_value(open).unwrap(), expected);

    let mut wrong_version = expected;
    wrong_version["schema_version"] = "translator.provider.health.v1".into();
    assert!(serde_json::from_value::<OpenProviderSession>(wrong_version).is_err());
    assert!(
        serde_json::from_value::<CloseProviderSession>(serde_json::json!({
            "schema_version": "translator.provider.close_session.v1",
            "session_id": session_id,
            "reason": "provider_failure"
        }))
        .is_err()
    );
}

#[test]
fn runtime_debug_update_and_probe_have_distinct_versioned_contracts() {
    let session_id = Uuid::new_v4();
    let generation_id = Uuid::new_v4();
    let update = UpdateDebugText {
        schema_version: UpdateDebugTextVersion::V1,
        session_id,
        enabled: false,
    };
    let probe = ProviderProbeRequest {
        schema_version: ProviderProbeRequestVersion::V1,
    };
    let response = ProviderProbeResponse {
        schema_version: ProviderProbeResponseVersion::V1,
        generation_id,
    };

    assert_eq!(update.session_id, session_id);
    assert!(!update.enabled);
    assert_eq!(
        serde_json::to_value(probe.schema_version).unwrap(),
        "translator.provider.probe_request.v1"
    );
    assert_eq!(
        serde_json::to_value(response.schema_version).unwrap(),
        "translator.provider.probe_response.v1"
    );
    assert_eq!(response.generation_id, generation_id);
    assert_eq!(
        serde_json::to_value(&update).unwrap(),
        serde_json::json!({
            "schema_version": "translator.provider.update_debug_text.v1",
            "session_id": session_id,
            "enabled": false
        })
    );
    assert!(
        serde_json::from_value::<UpdateDebugText>(serde_json::json!({
            "schema_version": "translator.provider.close_session.v1",
            "session_id": session_id,
            "enabled": false
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<ProviderProbeRequest>(serde_json::json!({
            "schema_version": "translator.provider.probe_response.v1"
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<ProviderProbeResponse>(serde_json::json!({
            "schema_version": "translator.provider.probe_request.v1",
            "generation_id": generation_id
        }))
        .is_err()
    );
}

#[test]
fn streaming_and_latency_contracts_preserve_ordering_and_policy_state() {
    let session_id = Uuid::new_v4();
    let stream_id = Uuid::new_v4();
    let utterance_id = Uuid::new_v4();
    let input = ProviderInputFrame {
        schema_version: ProviderInputVersion::V1,
        session_id,
        direction_id: AudioDirection::Microphone,
        stream_id,
        utterance_id,
        sequence: 3,
        capture_monotonic_ns: 1_000_000,
        format: pcm_format(),
        source_language: Language::Ru,
        target_language: Language::En,
        mode: TranslationMode::StreamingFirst,
        end_of_utterance: true,
        pcm: vec![0, 1, 2, 3],
    };
    let audio = ProviderAudioDelta {
        schema_version: ProviderAudioDeltaVersion::V1,
        session_id,
        direction_id: AudioDirection::Microphone,
        stream_id,
        utterance_id,
        sequence: 0,
        event_sequence: 4,
        provider_monotonic_ns: 1_100_000,
        format: pcm_format(),
        pcm: vec![4, 5, 6, 7],
    };
    let latency = ProviderLatency {
        schema_version: ProviderLatencyVersion::V1,
        session_id,
        direction_id: AudioDirection::Microphone,
        stream_id,
        event_sequence: 5,
        utterance_id: Some(utterance_id),
        asr_first_text_ms: Some(180),
        asr_final_text_ms: None,
        mt_first_text_ms: Some(75),
        tts_first_audio_ms: Some(220),
        provider_total_ms: Some(475),
    };
    let policy = LatencyPolicyState::new(
        AudioDirection::Microphone,
        TranslationMode::StreamingFirst,
        930,
        1_240,
        45,
        None,
        Some("first_audio_threshold".to_owned()),
    );

    assert_eq!(input.sequence, 3);
    assert!(input.end_of_utterance);
    assert_eq!(audio.event_sequence, 4);
    assert_eq!(latency.event_sequence, 5);
    assert_eq!(policy.current_mode, TranslationMode::StreamingFirst);

    let input_json = serde_json::to_value(input).unwrap();
    let audio_json = serde_json::to_value(audio).unwrap();
    assert_eq!(input_json["sample_rate_hz"], 16_000);
    assert_eq!(audio_json["frame_duration_ms"], 40);
    assert!(input_json.get("format").is_none());
    assert!(audio_json.get("format").is_none());

    let mut invalid_policy = serde_json::to_value(policy).unwrap();
    invalid_policy["rolling_window_seconds"] = 30.into();
    assert!(serde_json::from_value::<LatencyPolicyState>(invalid_policy).is_err());
    assert!(
        serde_json::from_value::<ProviderCapabilities>(serde_json::json!({
            "audio_output": false,
            "transcript_delta": false,
            "translation_delta": false,
            "cancellation": true,
            "cloud_egress": false
        }))
        .is_err()
    );
}

#[test]
fn pcm_format_rejects_values_outside_the_negotiated_contract() {
    assert!(PcmFormat::try_new(44_100, 1, SampleFormat::S16Le, 40).is_err());
    assert!(PcmFormat::try_new(16_000, 3, SampleFormat::S16Le, 40).is_err());
    assert!(PcmFormat::try_new(16_000, 1, SampleFormat::S16Le, 30).is_err());
    assert!(
        serde_json::from_str::<PcmFormat>(
            r#"{"sample_rate_hz":44100,"channels":1,"sample_format":"s16le","frame_duration_ms":40}"#
        )
        .is_err()
    );
}

#[test]
fn provider_error_rejects_unknown_or_content_derived_messages() {
    let marker = "private-spoken-marker";
    let unknown_field = format!(
        r#"{{"schema_version":"translator.provider.error.v1","session_id":"{}","direction_id":"speaker","event_sequence":1,"code":"provider_unavailable","retryable":true,"safe_message":"Provider is unavailable","transcript":"{}"}}"#,
        Uuid::new_v4(),
        marker
    );
    let content_derived_message = format!(
        r#"{{"schema_version":"translator.provider.error.v1","session_id":"{}","direction_id":"speaker","event_sequence":1,"code":"provider_unavailable","retryable":true,"safe_message":"{}"}}"#,
        Uuid::new_v4(),
        marker
    );
    let mismatched_static_message = format!(
        r#"{{"schema_version":"translator.provider.error.v1","session_id":"{}","direction_id":"speaker","event_sequence":1,"code":"provider_unavailable","retryable":true,"safe_message":"Required model is not loaded"}}"#,
        Uuid::new_v4()
    );

    assert!(serde_json::from_str::<PrivacySafeProviderError>(&unknown_field).is_err());
    assert!(serde_json::from_str::<PrivacySafeProviderError>(&content_derived_message).is_err());
    assert!(serde_json::from_str::<PrivacySafeProviderError>(&mismatched_static_message).is_err());
}

#[test]
fn privacy_safe_logging_projects_only_operational_error_fields() {
    let error = PrivacySafeProviderError::new(
        Uuid::new_v4(),
        AudioDirection::Speaker,
        11,
        SafeErrorCode::ProviderUnavailable,
        true,
    );

    let json = serde_json::to_string(&PrivacySafeLogEvent::from(&error)).expect("log event");

    assert!(json.contains("provider_unavailable"));
    assert!(json.contains("\"retryable\":true"));
    assert!(!json.contains("safe_message"));
    assert!(!json.contains("transcript"));
    assert!(!json.contains("pcm"));
}

#[test]
fn no_speech_has_stable_privacy_safe_contract() {
    assert_eq!(SafeErrorCode::NoSpeech.message(), "No speech was detected");
    assert_eq!(
        serde_json::to_string(&SafeErrorCode::NoSpeech).unwrap(),
        "\"no_speech\""
    );
}
