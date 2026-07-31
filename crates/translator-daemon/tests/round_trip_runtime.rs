use std::sync::{Arc, Mutex};

use translator_audio::{
    AcousticSafety, AecCapability, AudioGraphState, DeviceHealth, DeviceSelectionState,
    DeviceState, GraphHealth, OutputMode, PhysicalDevice, RouteResolution, RoutingState,
};
use translator_daemon::{
    ActiveRoundTripRuntime, AudioOperationGate, AudioOperationLease, AudioOperationState,
    RoundTripCheckpoint, RoundTripProgress, RoundTripRunner, RoundTripRuntimeError,
    RoundTripRuntimeHandle, RuntimeSnapshot, RuntimeStore,
};
use uuid::Uuid;

struct FakeRunner {
    starts: Mutex<Vec<Uuid>>,
    progress: Mutex<Option<RoundTripProgress>>,
    fail_start: bool,
    stop_failures: usize,
    stop_attempts: Arc<Mutex<usize>>,
}

impl Default for FakeRunner {
    fn default() -> Self {
        Self {
            starts: Mutex::new(Vec::new()),
            progress: Mutex::new(None),
            fail_start: false,
            stop_failures: 0,
            stop_attempts: Arc::new(Mutex::new(0)),
        }
    }
}

impl RoundTripRunner for FakeRunner {
    fn start(
        &self,
        _snapshot: RuntimeSnapshot,
        session_id: Uuid,
        progress: RoundTripProgress,
        lease: AudioOperationLease,
    ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError> {
        self.starts.lock().unwrap().push(session_id);
        *self.progress.lock().unwrap() = Some(progress);
        if self.fail_start {
            return Err(RoundTripRuntimeError::StartFailed);
        }
        Ok(Box::new(FakeActive {
            lease: Some(lease),
            finished: false,
            stop_failures: self.stop_failures,
            stop_attempts: Arc::clone(&self.stop_attempts),
        }))
    }
}

struct FakeActive {
    lease: Option<AudioOperationLease>,
    finished: bool,
    stop_failures: usize,
    stop_attempts: Arc<Mutex<usize>>,
}

impl ActiveRoundTripRuntime for FakeActive {
    fn stop(&mut self) -> Result<(), RoundTripRuntimeError> {
        let mut attempts = self.stop_attempts.lock().unwrap();
        *attempts += 1;
        if *attempts <= self.stop_failures {
            return Err(RoundTripRuntimeError::StopFailed);
        }
        self.lease.take();
        self.finished = true;
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

fn ready_store() -> RuntimeStore {
    let store = RuntimeStore::default();
    store.set_audio_graph(AudioGraphState {
        health: GraphHealth::Ready,
        endpoints: Vec::new(),
        owned_module_ids: Vec::new(),
        safe_error: None,
    });
    let source = PhysicalDevice {
        id: 1,
        name: "alsa_input.physical".to_owned(),
        description: "Physical microphone".to_owned(),
        active_port: Some("analog-input-mic".to_owned()),
        active_port_type: Some("Mic".to_owned()),
        available: true,
    };
    let sink = PhysicalDevice {
        id: 2,
        name: "alsa_output.headphones".to_owned(),
        description: "Headphones".to_owned(),
        active_port: Some("analog-output-headphones".to_owned()),
        active_port_type: Some("Headphones".to_owned()),
        available: true,
    };
    store.set_devices(DeviceState {
        source: selection(source),
        sink: selection(sink),
        acoustic: AcousticSafety {
            mode: OutputMode::Headphones,
            aec_capability: AecCapability::Unavailable,
            full_duplex_allowed: true,
            warning: None,
        },
    });
    store.set_routes(RoutingState {
        candidates: Vec::new(),
        source_outputs: Vec::new(),
        conflicting_stream_ids: Vec::new(),
        active_route: None,
        resolution: RouteResolution::NoCandidate,
    });
    store
}

fn selection(device: PhysicalDevice) -> DeviceSelectionState {
    DeviceSelectionState {
        health: DeviceHealth::Available,
        pinned_name: Some(device.name.clone()),
        current_default: Some(device.name.clone()),
        pending_default: None,
        selected: Some(device),
    }
}

#[test]
fn controller_publishes_availability_before_first_start() {
    let store = ready_store();
    let controller = RoundTripRuntimeHandle::new(
        store.clone(),
        Arc::new(FakeRunner::default()),
        AudioOperationGate::new(),
    );

    assert_eq!(store.snapshot().self_test.availability, "available");
    assert!(store.snapshot().self_test.status.session_id.is_none());
    drop(controller);
}

#[test]
fn controller_holds_shared_gate_until_idempotent_runtime_teardown() {
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(FakeRunner::default());
    let controller = RoundTripRuntimeHandle::new(store, runner, gate.clone());

    let started = translator_daemon::RoundTripController::start(&controller).unwrap();
    let session_id = started.status.session_id.unwrap();

    assert_eq!(
        gate.state(),
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert_eq!(
        started.status.checkpoint,
        Some(RoundTripCheckpoint::WaitingForSpeech)
    );
    assert!(translator_daemon::RoundTripController::start(&controller).is_err());

    let stopped = translator_daemon::RoundTripController::stop(&controller).unwrap();
    assert_eq!(
        stopped.status.checkpoint,
        Some(RoundTripCheckpoint::Stopped)
    );
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert!(translator_daemon::RoundTripController::stop(&controller).is_err());
}

#[test]
fn shared_gate_and_runner_failure_leave_no_live_self_test_lease() {
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let production = gate.acquire_production().unwrap();
    let runner = Arc::new(FakeRunner::default());
    let controller = RoundTripRuntimeHandle::new(store.clone(), runner.clone(), gate.clone());

    let busy = translator_daemon::RoundTripController::start(&controller).unwrap_err();
    assert_eq!(busy.code, "audio_operation_busy");
    assert_eq!(runner.starts.lock().unwrap().len(), 0);
    drop(production);

    let failing = Arc::new(FakeRunner {
        fail_start: true,
        ..FakeRunner::default()
    });
    let controller = RoundTripRuntimeHandle::new(store, failing, gate.clone());
    let error = translator_daemon::RoundTripController::start(&controller).unwrap_err();
    assert_eq!(error.code, "self_test_start_failed");
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn stop_timeout_retains_runtime_and_gate_for_retry_cleanup() {
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(FakeRunner {
        stop_failures: 1,
        ..FakeRunner::default()
    });
    let controller = RoundTripRuntimeHandle::new(store, runner.clone(), gate.clone());

    let started = translator_daemon::RoundTripController::start(&controller).unwrap();
    let session_id = started.status.session_id.unwrap();
    let first_stop = translator_daemon::RoundTripController::stop(&controller).unwrap_err();

    assert_eq!(first_stop.code, "self_test_stop_failed");
    assert_eq!(
        gate.state(),
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert!(translator_daemon::RoundTripController::start(&controller).is_err());

    let stopped = translator_daemon::RoundTripController::stop(&controller).unwrap();
    assert_eq!(
        stopped.status.checkpoint,
        Some(RoundTripCheckpoint::Stopped)
    );
    assert_eq!(*runner.stop_attempts.lock().unwrap(), 2);
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn unsafe_snapshot_is_rejected_before_runner_or_gate_acquisition() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(FakeRunner::default());
    let controller = RoundTripRuntimeHandle::new(store, runner.clone(), gate.clone());

    let error = translator_daemon::RoundTripController::start(&controller).unwrap_err();

    assert_eq!(error.code, "self_test_headphones_required");
    assert!(runner.starts.lock().unwrap().is_empty());
    assert_eq!(gate.state(), AudioOperationState::Idle);
}
