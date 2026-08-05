use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use axum::http::StatusCode;
use translator_audio::{
    AEC_SINK, AEC_SOURCE, AcousticSafety, AecCapability, DeviceHealth, DeviceSelectionState,
    DeviceState, OutputMode, PhysicalDevice,
};
use translator_core::{AudioDirection, TranslationMode};
use translator_daemon::{
    ActiveDuplexRuntime, AudioOperationGate, DuplexRunner, DuplexRuntimeError, DuplexRuntimeEvent,
    DuplexRuntimeHandle, DuplexRuntimeObserver, RuntimeLatencyObserver, RuntimeSnapshot,
    RuntimeStore, TranslationController, resolve_duplex_audio_targets,
};
use uuid::Uuid;

#[derive(Default)]
struct FakeRunner {
    starts: AtomicUsize,
    stops: Arc<AtomicUsize>,
    snapshots: Mutex<Vec<RuntimeSnapshot>>,
}

impl DuplexRunner for FakeRunner {
    fn start(
        &self,
        snapshot: RuntimeSnapshot,
    ) -> Result<Box<dyn ActiveDuplexRuntime>, DuplexRuntimeError> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        self.snapshots.lock().unwrap().push(snapshot);
        Ok(Box::new(FakeActive {
            stops: self.stops.clone(),
        }))
    }
}

struct FakeActive {
    stops: Arc<AtomicUsize>,
}

fn audio_snapshot(mode: OutputMode, aec_capability: AecCapability) -> RuntimeSnapshot {
    audio_snapshot_with_safety(mode, aec_capability, true)
}

fn audio_snapshot_with_safety(
    mode: OutputMode,
    aec_capability: AecCapability,
    full_duplex_allowed: bool,
) -> RuntimeSnapshot {
    let selection = |id, name: &str| DeviceSelectionState {
        health: DeviceHealth::Available,
        selected: Some(PhysicalDevice {
            id,
            name: name.to_owned(),
            description: name.to_owned(),
            active_port: None,
            active_port_type: None,
            available: true,
        }),
        pinned_name: Some(name.to_owned()),
        current_default: Some(name.to_owned()),
        pending_default: None,
    };
    RuntimeSnapshot {
        devices: Some(DeviceState {
            source: selection(1, "alsa_input.physical"),
            sink: selection(2, "alsa_output.physical"),
            acoustic: AcousticSafety {
                mode,
                aec_capability,
                full_duplex_allowed,
                warning: None,
            },
        }),
        ..RuntimeSnapshot::default()
    }
}

impl ActiveDuplexRuntime for FakeActive {
    fn stop(&mut self) -> Result<(), DuplexRuntimeError> {
        self.stops.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn translation_controller_is_single_flight_and_can_restart_after_stop() {
    let runner = Arc::new(FakeRunner::default());
    let controller = DuplexRuntimeHandle::with_runner(runner.clone());

    controller.start(RuntimeSnapshot::default()).unwrap();
    let duplicate = controller.start(RuntimeSnapshot::default()).unwrap_err();
    assert_eq!(duplicate.status, StatusCode::CONFLICT);
    assert_eq!(duplicate.code, "translation_already_running");

    controller.stop().unwrap();
    assert_eq!(runner.stops.load(Ordering::SeqCst), 1);
    controller.start(RuntimeSnapshot::default()).unwrap();
    controller.stop().unwrap();
    assert_eq!(runner.starts.load(Ordering::SeqCst), 2);
    assert_eq!(runner.stops.load(Ordering::SeqCst), 2);
}

#[test]
fn translation_controller_rejects_stop_when_idle() {
    let controller = DuplexRuntimeHandle::with_runner(Arc::new(FakeRunner::default()));
    let error = controller.stop().unwrap_err();
    assert_eq!(error.status, StatusCode::CONFLICT);
    assert_eq!(error.code, "translation_not_running");
}

#[test]
fn shared_gate_blocks_production_while_human_round_trip_owns_audio() {
    let gate = AudioOperationGate::new();
    let self_test = gate.acquire_human_round_trip(Uuid::new_v4()).unwrap();
    let runner = Arc::new(FakeRunner::default());
    let controller = DuplexRuntimeHandle::with_runner_and_gate(runner.clone(), gate);

    let error = controller.start(RuntimeSnapshot::default()).unwrap_err();

    assert_eq!(error.status, StatusCode::CONFLICT);
    assert_eq!(error.code, "audio_operation_busy");
    assert_eq!(runner.starts.load(Ordering::SeqCst), 0);
    drop(self_test);
    controller.start(RuntimeSnapshot::default()).unwrap();
    controller.stop().unwrap();
}

#[test]
fn audio_targets_use_aec_when_valid_and_physical_fallback_otherwise() {
    let headphones = resolve_duplex_audio_targets(&audio_snapshot(
        OutputMode::Headphones,
        AecCapability::Unavailable,
    ))
    .unwrap();
    assert_eq!(headphones.microphone_capture, "alsa_input.physical");
    assert_eq!(headphones.speaker_playback, "alsa_output.physical");

    let aec_headphones = resolve_duplex_audio_targets(&audio_snapshot(
        OutputMode::Headphones,
        AecCapability::ValidatedFor {
            source_name: "alsa_input.physical".to_owned(),
            sink_name: "alsa_output.physical".to_owned(),
        },
    ))
    .unwrap();
    assert_eq!(aec_headphones.microphone_capture, AEC_SOURCE);
    assert_eq!(aec_headphones.speaker_playback, AEC_SINK);

    let open_speaker = resolve_duplex_audio_targets(&audio_snapshot(
        OutputMode::OpenSpeaker,
        AecCapability::AvailableUnvalidated,
    ))
    .unwrap();
    assert_eq!(open_speaker.microphone_capture, "alsa_input.physical");
    assert_eq!(open_speaker.speaker_playback, "alsa_output.physical");

    let stale_aec = resolve_duplex_audio_targets(&audio_snapshot(
        OutputMode::OpenSpeaker,
        AecCapability::ValidatedFor {
            source_name: "alsa_input.other".to_owned(),
            sink_name: "alsa_output.physical".to_owned(),
        },
    ))
    .unwrap();
    assert_eq!(stale_aec.microphone_capture, "alsa_input.physical");
    assert_eq!(stale_aec.speaker_playback, "alsa_output.physical");

    let validated = resolve_duplex_audio_targets(&audio_snapshot(
        OutputMode::OpenSpeaker,
        AecCapability::ValidatedFor {
            source_name: "alsa_input.physical".to_owned(),
            sink_name: "alsa_output.physical".to_owned(),
        },
    ))
    .unwrap();
    assert_eq!(validated.microphone_capture, AEC_SOURCE);
    assert_eq!(validated.speaker_playback, AEC_SINK);
}

#[test]
fn audio_targets_fall_back_to_physical_devices_when_acoustic_safety_is_not_validated() {
    for mode in [OutputMode::OpenSpeaker, OutputMode::UnknownUnsafe] {
        let targets = resolve_duplex_audio_targets(&audio_snapshot_with_safety(
            mode,
            AecCapability::Unavailable,
            false,
        ))
        .unwrap();

        assert_eq!(targets.microphone_capture, "alsa_input.physical");
        assert_eq!(targets.speaker_playback, "alsa_output.physical");
        assert_eq!(targets.microphone_playback, "translator_mic_out");
        assert_eq!(targets.speaker_capture, "translator_remote_in.monitor");
    }
}

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
