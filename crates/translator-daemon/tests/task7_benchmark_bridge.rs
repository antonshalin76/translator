use std::num::NonZeroU32;

use serde_json::Value;
use translator_core::AudioDirection;
use translator_daemon::{
    DuplexRuntimeEvent, SafeProviderErrorCode, TASK7_BRIDGE_SCHEMA_VERSION, Task7BridgeEvent,
    Task7BridgeFailureStage, TerminalOutcome,
};
use uuid::Uuid;

fn assert_privacy_safe(event: Task7BridgeEvent) {
    let value = serde_json::to_value(event).unwrap();
    assert_eq!(
        value["schema_version"],
        Value::String(TASK7_BRIDGE_SCHEMA_VERSION.to_owned())
    );
    let serialized = serde_json::to_string(&value).unwrap();
    for forbidden in [
        "transcript",
        "translation",
        "text",
        "pcm",
        "audio_bytes",
        "safe_message",
        "token",
        "secret",
    ] {
        assert!(
            !serialized.contains(&format!("\"{forbidden}\"")),
            "privacy-sensitive field {forbidden} was serialized: {serialized}"
        );
    }
}

#[test]
fn bridge_runtime_events_serialize_only_privacy_safe_metadata() {
    let utterance_id = Uuid::new_v4();
    let runtime_events = [
        DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id,
            capture_monotonic_ns: 10,
        },
        DuplexRuntimeEvent::TranscriptFinal {
            direction: AudioDirection::Microphone,
            utterance_id,
        },
        DuplexRuntimeEvent::TranslationFinal {
            direction: AudioDirection::Microphone,
            utterance_id,
        },
        DuplexRuntimeEvent::AudioFrame {
            direction: AudioDirection::Microphone,
            utterance_id,
            sequence: 4,
            provider_monotonic_ns: 20,
            observed_monotonic_ns: 30,
            queue_lag_ms: 1,
        },
        DuplexRuntimeEvent::FirstAudioExpired {
            direction: AudioDirection::Microphone,
            utterance_id,
            observed_monotonic_ns: 31,
        },
        DuplexRuntimeEvent::ProviderLatency {
            direction: AudioDirection::Microphone,
            utterance_id: Some(utterance_id),
            tts_first_audio_ms: Some(200),
            provider_total_ms: Some(400),
        },
        DuplexRuntimeEvent::ProviderError {
            direction: AudioDirection::Microphone,
            utterance_id: Some(utterance_id),
            code: SafeProviderErrorCode::NoSpeech,
            retryable: true,
        },
        DuplexRuntimeEvent::UtteranceTerminalOutcome {
            direction: AudioDirection::Microphone,
            utterance_id,
            outcome: TerminalOutcome::Dropped,
        },
        DuplexRuntimeEvent::UtteranceTerminal {
            direction: AudioDirection::Microphone,
            utterance_id,
        },
        DuplexRuntimeEvent::GenerationRestart {
            attempt: NonZeroU32::new(1).unwrap(),
        },
    ];

    for event in runtime_events {
        assert_privacy_safe(Task7BridgeEvent::from_runtime(event));
    }
    assert_privacy_safe(Task7BridgeEvent::ready(42));
    assert_privacy_safe(Task7BridgeEvent::stopped());
    assert_privacy_safe(Task7BridgeEvent::failure(
        Task7BridgeFailureStage::RuntimeStart,
        "runtime_start_failed",
    ));
}

#[test]
fn generation_restart_bridge_event_is_global_and_privacy_safe() {
    let value = serde_json::to_value(Task7BridgeEvent::from_runtime(
        DuplexRuntimeEvent::GenerationRestart {
            attempt: NonZeroU32::new(2).unwrap(),
        },
    ))
    .unwrap();

    assert_eq!(
        value.as_object().unwrap().keys().collect::<Vec<_>>(),
        vec!["attempt", "event", "monotonic_ns", "schema_version"]
    );
    assert_eq!(value["event"], "generation_restart");
    assert_eq!(value["attempt"], 2);
    assert!(
        value["monotonic_ns"]
            .as_u64()
            .is_some_and(|value| value > 0)
    );
    assert!(value.get("direction").is_none());
    assert!(value.get("utterance_id").is_none());
    assert_privacy_safe(Task7BridgeEvent::from_runtime(
        DuplexRuntimeEvent::GenerationRestart {
            attempt: NonZeroU32::new(2).unwrap(),
        },
    ));
}

#[test]
fn bridge_preserves_provider_error_and_terminal_outcome_without_safe_message() {
    let utterance_id = Uuid::new_v4();
    let error = serde_json::to_value(Task7BridgeEvent::from_runtime(
        DuplexRuntimeEvent::ProviderError {
            direction: AudioDirection::Speaker,
            utterance_id: Some(utterance_id),
            code: SafeProviderErrorCode::NoSpeech,
            retryable: true,
        },
    ))
    .unwrap();
    assert_eq!(error["event"], "provider_error");
    assert_eq!(error["code"], "no_speech");
    assert_eq!(error["retryable"], true);
    assert!(error.get("safe_message").is_none());

    let terminal = serde_json::to_value(Task7BridgeEvent::from_runtime(
        DuplexRuntimeEvent::UtteranceTerminalOutcome {
            direction: AudioDirection::Speaker,
            utterance_id,
            outcome: TerminalOutcome::Dropped,
        },
    ))
    .unwrap();
    assert_eq!(terminal["event"], "utterance_terminal_outcome");
    assert_eq!(terminal["outcome"], "dropped");
}

#[test]
fn audio_frame_bridge_event_preserves_sequence_and_queue_lag() {
    let utterance_id = Uuid::new_v4();
    let value = serde_json::to_value(Task7BridgeEvent::from_runtime(
        DuplexRuntimeEvent::AudioFrame {
            direction: AudioDirection::Speaker,
            utterance_id,
            sequence: 17,
            provider_monotonic_ns: 100,
            observed_monotonic_ns: 140,
            queue_lag_ms: 40,
        },
    ))
    .unwrap();

    assert_eq!(value["event"], "audio_frame");
    assert_eq!(value["direction"], "speaker");
    assert_eq!(value["utterance_id"], utterance_id.to_string());
    assert_eq!(value["sequence"], 17);
    assert_eq!(value["provider_monotonic_ns"], 100);
    assert_eq!(value["monotonic_ns"], 140);
    assert_eq!(value["queue_lag_ms"], 40);
}
