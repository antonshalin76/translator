use std::num::NonZeroU32;

use serde_json::Value;
use translator_core::AudioDirection;
use translator_daemon::{
    DuplexRuntimeEvent, SafeProviderErrorCode, TASK7_BRIDGE_SCHEMA_VERSION, Task7BridgeEvent,
    Task7BridgeFailureStage, TerminalOutcome,
};
use uuid::Uuid;

fn now_ns() -> u64 {
    let time = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    u64::try_from(time.tv_sec).unwrap() * 1_000_000_000 + u64::try_from(time.tv_nsec).unwrap()
}

fn direction_name(direction: AudioDirection) -> &'static str {
    match direction {
        AudioDirection::Microphone => "microphone",
        AudioDirection::Speaker => "speaker",
    }
}

fn error_name(code: SafeProviderErrorCode) -> &'static str {
    match code {
        SafeProviderErrorCode::ProviderUnavailable => "provider_unavailable",
        SafeProviderErrorCode::ModelNotLoaded => "model_not_loaded",
        SafeProviderErrorCode::UnsupportedLanguagePair => "unsupported_language_pair",
        SafeProviderErrorCode::QueueOverflow => "queue_overflow",
        SafeProviderErrorCode::Cancelled => "cancelled",
        SafeProviderErrorCode::CloudNotEnabled => "cloud_not_enabled",
        SafeProviderErrorCode::ProviderAuthFailed => "provider_auth_failed",
        SafeProviderErrorCode::NoSpeech => "no_speech",
    }
}

fn outcome_name(outcome: TerminalOutcome) -> &'static str {
    match outcome {
        TerminalOutcome::Completed => "completed",
        TerminalOutcome::Cancelled => "cancelled",
        TerminalOutcome::Dropped => "dropped",
    }
}

// No wildcard or omitted fields: additions require an explicit wire decision.
fn runtime_wire_contract(event: DuplexRuntimeEvent) -> (Value, Option<u64>) {
    use serde_json::json;
    match event {
        DuplexRuntimeEvent::SpeechStarted {
            direction,
            utterance_id,
            capture_monotonic_ns,
        } => (
            json!({"event":"speech_started", "direction":direction_name(direction), "utterance_id":utterance_id.to_string()}),
            Some(capture_monotonic_ns),
        ),
        DuplexRuntimeEvent::TranscriptFinal {
            direction,
            utterance_id,
        } => (
            json!({"event":"asr_final", "direction":direction_name(direction), "utterance_id":utterance_id.to_string()}),
            None,
        ),
        DuplexRuntimeEvent::TranslationFinal {
            direction,
            utterance_id,
        } => (
            json!({"event":"translation_final", "direction":direction_name(direction), "utterance_id":utterance_id.to_string()}),
            None,
        ),
        DuplexRuntimeEvent::AudioFrame {
            direction,
            utterance_id,
            sequence,
            provider_monotonic_ns,
            observed_monotonic_ns,
            queue_lag_ms,
        } => (
            json!({"event":"audio_frame", "direction":direction_name(direction), "utterance_id":utterance_id.to_string(), "sequence":sequence, "provider_monotonic_ns":provider_monotonic_ns, "queue_lag_ms":queue_lag_ms}),
            Some(observed_monotonic_ns),
        ),
        DuplexRuntimeEvent::FirstAudioExpired {
            direction,
            utterance_id,
            observed_monotonic_ns,
        } => (
            json!({"event":"first_audio_expired", "direction":direction_name(direction), "utterance_id":utterance_id.to_string()}),
            Some(observed_monotonic_ns),
        ),
        DuplexRuntimeEvent::ProviderLatency {
            direction,
            utterance_id,
            tts_first_audio_ms,
            provider_total_ms,
        } => {
            let mut expected =
                json!({"event":"provider_latency", "direction":direction_name(direction)});
            if let Some(id) = utterance_id {
                expected["utterance_id"] = json!(id.to_string());
            }
            if let Some(ms) = tts_first_audio_ms {
                expected["tts_first_audio_ms"] = json!(ms);
            }
            if let Some(ms) = provider_total_ms {
                expected["provider_total_ms"] = json!(ms);
            }
            (expected, None)
        }
        DuplexRuntimeEvent::ProviderError {
            direction,
            utterance_id,
            code,
            retryable,
        } => {
            let mut expected = json!({"event":"provider_error", "direction":direction_name(direction), "code":error_name(code), "retryable":retryable});
            if let Some(id) = utterance_id {
                expected["utterance_id"] = json!(id.to_string());
            }
            (expected, None)
        }
        DuplexRuntimeEvent::UtteranceTerminalOutcome {
            direction,
            utterance_id,
            outcome,
        } => (
            json!({"event":"utterance_terminal_outcome", "direction":direction_name(direction), "utterance_id":utterance_id.to_string(), "outcome":outcome_name(outcome)}),
            None,
        ),
        DuplexRuntimeEvent::UtteranceTerminal {
            direction,
            utterance_id,
        } => (
            json!({"event":"utterance_terminal", "direction":direction_name(direction), "utterance_id":utterance_id.to_string()}),
            None,
        ),
        DuplexRuntimeEvent::GenerationRestart { attempt } => (
            json!({"event":"generation_restart", "attempt":attempt.get()}),
            None,
        ),
    }
}

fn assert_exact_wire(
    event: Task7BridgeEvent,
    mut expected: Value,
    supplied_timestamp: Option<u64>,
    before: u64,
    after: u64,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    serde_json::to_writer(&mut bytes, &event).unwrap();
    bytes.push(b'\n');
    let raw = std::str::from_utf8(&bytes).unwrap();
    for key in ["schema_version", "event", "monotonic_ns"] {
        assert_eq!(
            raw.matches(&format!("\"{key}\":")).count(),
            1,
            "duplicate/missing {key}: {raw}"
        );
    }
    let actual: Value = serde_json::from_slice(&bytes).unwrap();
    let timestamp = actual["monotonic_ns"].as_u64().unwrap();
    if let Some(supplied) = supplied_timestamp {
        assert_eq!(timestamp, supplied);
    } else {
        assert!(
            (before..=after).contains(&timestamp),
            "conversion changed its clock origin"
        );
    }
    expected["schema_version"] = Value::String("translator.task7-bridge.v1".into());
    expected["monotonic_ns"] = timestamp.into();
    assert_eq!(actual, expected, "wire keys/values drifted");
    bytes
}

fn all_runtime_cases() -> Vec<DuplexRuntimeEvent> {
    let id = Uuid::from_u128(42);
    let mut events = vec![DuplexRuntimeEvent::GenerationRestart {
        attempt: NonZeroU32::new(2).unwrap(),
    }];
    for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
        events.extend([
            DuplexRuntimeEvent::SpeechStarted {
                direction,
                utterance_id: id,
                capture_monotonic_ns: 101,
            },
            DuplexRuntimeEvent::TranscriptFinal {
                direction,
                utterance_id: id,
            },
            DuplexRuntimeEvent::TranslationFinal {
                direction,
                utterance_id: id,
            },
            DuplexRuntimeEvent::AudioFrame {
                direction,
                utterance_id: id,
                sequence: 17,
                provider_monotonic_ns: 202,
                observed_monotonic_ns: 303,
                queue_lag_ms: 4,
            },
            DuplexRuntimeEvent::FirstAudioExpired {
                direction,
                utterance_id: id,
                observed_monotonic_ns: 404,
            },
            DuplexRuntimeEvent::UtteranceTerminal {
                direction,
                utterance_id: id,
            },
        ]);
        for utterance_id in [None, Some(id)] {
            for tts_first_audio_ms in [None, Some(11)] {
                for provider_total_ms in [None, Some(22)] {
                    events.push(DuplexRuntimeEvent::ProviderLatency {
                        direction,
                        utterance_id,
                        tts_first_audio_ms,
                        provider_total_ms,
                    });
                }
            }
            for code in [
                SafeProviderErrorCode::ProviderUnavailable,
                SafeProviderErrorCode::ModelNotLoaded,
                SafeProviderErrorCode::UnsupportedLanguagePair,
                SafeProviderErrorCode::QueueOverflow,
                SafeProviderErrorCode::Cancelled,
                SafeProviderErrorCode::CloudNotEnabled,
                SafeProviderErrorCode::ProviderAuthFailed,
                SafeProviderErrorCode::NoSpeech,
            ] {
                for retryable in [false, true] {
                    events.push(DuplexRuntimeEvent::ProviderError {
                        direction,
                        utterance_id,
                        code,
                        retryable,
                    });
                }
            }
        }
        for outcome in [
            TerminalOutcome::Completed,
            TerminalOutcome::Cancelled,
            TerminalOutcome::Dropped,
        ] {
            events.push(DuplexRuntimeEvent::UtteranceTerminalOutcome {
                direction,
                utterance_id: id,
                outcome,
            });
        }
    }
    events
}

#[tokio::test]
async fn complete_bridge_wire_contract_matches_unchanged_python_consumers() {
    use std::process::Stdio;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut ndjson = Vec::new();
    let events = all_runtime_cases();
    assert_eq!(events.len(), 99);
    for event in events {
        let (expected, supplied) = runtime_wire_contract(event);
        let before = now_ns();
        let wire = Task7BridgeEvent::from_runtime(event);
        let after = now_ns();
        ndjson.extend(assert_exact_wire(wire, expected, supplied, before, after));
    }
    for (stage, name) in [
        (Task7BridgeFailureStage::RuntimeLease, "runtime_lease"),
        (
            Task7BridgeFailureStage::AudioGraphEnsure,
            "audio_graph_ensure",
        ),
        (
            Task7BridgeFailureStage::RuntimeConfiguration,
            "runtime_configuration",
        ),
        (Task7BridgeFailureStage::RuntimeStart, "runtime_start"),
        (Task7BridgeFailureStage::RuntimeStop, "runtime_stop"),
        (Task7BridgeFailureStage::ControlInput, "control_input"),
        (
            Task7BridgeFailureStage::AudioGraphCleanup,
            "audio_graph_cleanup",
        ),
        (Task7BridgeFailureStage::Output, "output"),
    ] {
        let before = now_ns();
        let wire = Task7BridgeEvent::failure(stage, "synthetic_failure");
        let after = now_ns();
        ndjson.extend(assert_exact_wire(
            wire,
            serde_json::json!({"event":"failure", "stage":name, "code":"synthetic_failure"}),
            None,
            before,
            after,
        ));
    }
    let before = now_ns();
    let ready = Task7BridgeEvent::ready(42);
    let after = now_ns();
    ndjson.extend(assert_exact_wire(
        ready,
        serde_json::json!({"event":"ready", "pid":42}),
        None,
        before,
        after,
    ));
    let before = now_ns();
    let stopped = Task7BridgeEvent::stopped();
    let after = now_ns();
    ndjson.extend(assert_exact_wire(
        stopped,
        serde_json::json!({"event":"stopped"}),
        None,
        before,
        after,
    ));

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut child = tokio::process::Command::new(root.join("sidecar/.venv/bin/python"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("OPENBLAS_NUM_THREADS", "1")
        .args(["-E", "-s", "-m", "tests.fixtures.task7_bridge_contract"])
        .current_dir(root.join("sidecar"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = child.stdout.take().unwrap();
    let checked = tokio::time::timeout(std::time::Duration::from_secs(15), async {
        input.write_all(&ndjson).await?;
        drop(input);
        let status = child.wait().await?;
        let mut report = String::new();
        output.read_to_string(&mut report).await?;
        Ok::<_, std::io::Error>((status, report))
    })
    .await;
    let (status, report) = match checked {
        Ok(Ok(result)) => result,
        _ => {
            let _ = child.kill().await;
            child.wait().await.unwrap();
            panic!("synthetic bridge parser timed out or failed its I/O contract");
        }
    };
    assert!(
        status.success(),
        "unchanged Python bridge rejected synthetic wire contract"
    );
    assert_eq!(report.trim(), "109");
}

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
