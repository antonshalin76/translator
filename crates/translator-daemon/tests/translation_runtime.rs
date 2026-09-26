use translator_core::{AudioDirection, TranslationMode};
use translator_daemon::{
    DuplexRuntimeEvent, DuplexRuntimeObserver, RuntimeLatencyObserver, RuntimeStore,
};
use uuid::Uuid;

#[test]
fn runtime_latency_observer_drives_existing_quality_first_policy_without_content() {
    let store = RuntimeStore::default();
    let observer = RuntimeLatencyObserver::new(store.clone());

    for index in 0..3 {
        let utterance_id = Uuid::new_v4();
        let capture_ns = 1_000_000_000 + index * 10_000_000_000;
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id,
            capture_monotonic_ns: capture_ns,
        });
        observer.observe(DuplexRuntimeEvent::AudioFrame {
            direction: AudioDirection::Microphone,
            utterance_id,
            sequence: 0,
            provider_monotonic_ns: capture_ns + 4_000_000_000,
            observed_monotonic_ns: capture_ns + 4_000_000_000,
            queue_lag_ms: 20,
        });
        observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
            direction: AudioDirection::Microphone,
            utterance_id,
        });
    }

    let microphone = store
        .snapshot()
        .latency_policy
        .into_iter()
        .find(|state| state.direction_id == AudioDirection::Microphone)
        .unwrap();
    assert_eq!(microphone.current_mode, TranslationMode::Balanced);
    assert_eq!(microphone.reason.as_deref(), Some("consecutive_utterances"));
}
