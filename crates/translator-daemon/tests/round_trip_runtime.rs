use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use translator_audio::{
    AecCapability, AudioGraphState, DeviceHealth, DeviceSelectionState, GraphHealth, OutputMode,
    PhysicalDevice, RouteResolution, RoutingState,
};
use translator_daemon::{
    AcousticSafety, ActiveRoundTripRuntime, AdmittedDuplex, AudioOperationGate,
    AudioOperationState, DeviceState, RoundTripCheckpoint, RoundTripProgress, RoundTripRunner,
    RoundTripRuntimeError, RoundTripRuntimeHandle, RuntimeStore,
};
use uuid::Uuid;

#[path = "support/audio_facts.rs"]
mod audio_facts;

fn test_controller(
    store: RuntimeStore,
    runner: Arc<dyn RoundTripRunner>,
    gate: AudioOperationGate,
) -> RoundTripRuntimeHandle {
    let facts = audio_facts::fixture(store.snapshot());
    RoundTripRuntimeHandle::try_new(store, runner, gate, facts).unwrap()
}

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
        _admitted: AdmittedDuplex,
        session_id: Uuid,
        progress: RoundTripProgress,
        _start_deadline: std::time::Instant,
    ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError> {
        self.starts.lock().unwrap().push(session_id);
        *self.progress.lock().unwrap() = Some(progress);
        if self.fail_start {
            return Err(RoundTripRuntimeError::StartFailed);
        }
        Ok(Box::new(FakeActive {
            stop_failures: self.stop_failures,
            stop_attempts: Arc::clone(&self.stop_attempts),
        }))
    }
}

struct FakeActive {
    stop_failures: usize,
    stop_attempts: Arc<Mutex<usize>>,
}

struct PanicOnceRunner {
    runtime_ids: Arc<Mutex<Vec<Uuid>>>,
    stop_calls: Arc<Mutex<Vec<Uuid>>>,
    drops: Arc<AtomicUsize>,
    progress: Arc<Mutex<Option<RoundTripProgress>>>,
}

impl RoundTripRunner for PanicOnceRunner {
    fn start(
        &self,
        _admitted: AdmittedDuplex,
        _session_id: Uuid,
        progress: RoundTripProgress,
        _start_deadline: std::time::Instant,
    ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError> {
        let runtime_id = Uuid::new_v4();
        self.runtime_ids.lock().unwrap().push(runtime_id);
        *self.progress.lock().unwrap() = Some(progress);
        Ok(Box::new(PanicOnceActive {
            runtime_id,
            panicked: false,
            stop_calls: Arc::clone(&self.stop_calls),
            drops: Arc::clone(&self.drops),
        }))
    }
}

struct PanicOnceActive {
    runtime_id: Uuid,
    panicked: bool,
    stop_calls: Arc<Mutex<Vec<Uuid>>>,
    drops: Arc<AtomicUsize>,
}

impl ActiveRoundTripRuntime for PanicOnceActive {
    fn stop(
        &mut self,
        _deadline: std::time::Instant,
        _cleanup_deadline: std::time::Instant,
    ) -> Result<translator_daemon::RoundTripTerminal, RoundTripRuntimeError> {
        self.stop_calls.lock().unwrap().push(self.runtime_id);
        if !self.panicked {
            self.panicked = true;
            panic!("injected Stop panic");
        }
        Ok(translator_daemon::RoundTripTerminal::Stopped)
    }
}

impl Drop for PanicOnceActive {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl ActiveRoundTripRuntime for FakeActive {
    fn stop(
        &mut self,
        _: std::time::Instant,
        _: std::time::Instant,
    ) -> Result<translator_daemon::RoundTripTerminal, RoundTripRuntimeError> {
        let mut attempts = self.stop_attempts.lock().unwrap();
        *attempts += 1;
        if *attempts <= self.stop_failures {
            return Err(RoundTripRuntimeError::StopFailed);
        }
        Ok(translator_daemon::RoundTripTerminal::Stopped)
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
    let controller = test_controller(
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
    let controller = test_controller(store, runner, gate.clone());

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
    assert!(translator_daemon::RoundTripController::stop(&controller).is_ok());
}

#[test]
fn shared_gate_and_runner_failure_leave_no_live_self_test_lease() {
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let production = gate.acquire_production().unwrap();
    let runner = Arc::new(FakeRunner::default());
    let controller = test_controller(store.clone(), runner.clone(), gate.clone());

    let busy = translator_daemon::RoundTripController::start(&controller).unwrap_err();
    assert_eq!(busy.code, "audio_operation_busy");
    assert_eq!(runner.starts.lock().unwrap().len(), 0);
    drop(production);

    let failing = Arc::new(FakeRunner {
        fail_start: true,
        ..FakeRunner::default()
    });
    let controller = test_controller(store, failing, gate.clone());
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
    let controller = test_controller(store, runner.clone(), gate.clone());

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
    assert_eq!(stopped.status.checkpoint, Some(RoundTripCheckpoint::Failed));
    assert!(!stopped.status.cleanup_pending);
    assert_eq!(*runner.stop_attempts.lock().unwrap(), 2);
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn shutdown_admission_keeps_start_disabled_across_cleanup_retry() {
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(FakeRunner {
        stop_failures: 1,
        ..FakeRunner::default()
    });
    let controller = test_controller(store, runner.clone(), gate.clone());
    translator_daemon::RoundTripController::start(&controller).unwrap();

    let first_shutdown = controller.shutdown();
    let retry = translator_daemon::RoundTripController::stop(&controller);
    let rejected_start = translator_daemon::RoundTripController::start(&controller);
    let final_shutdown = controller.shutdown();

    assert_eq!(
        first_shutdown,
        Err(translator_daemon::RoundTripOwnerShutdownError::CleanupPending)
    );
    assert!(retry.is_ok());
    assert_eq!(rejected_start.unwrap_err().code, "self_test_owner_failed");
    assert_eq!(*runner.stop_attempts.lock().unwrap(), 2);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(final_shutdown, Ok(()));
}

#[test]
fn unsafe_snapshot_is_rejected_before_runner_or_gate_acquisition() {
    let store = ready_store();
    let mut devices = store.snapshot().devices.unwrap();
    devices.acoustic.mode = OutputMode::UnknownUnsafe;
    store.set_devices(devices);
    let gate = AudioOperationGate::new();
    let runner = Arc::new(FakeRunner::default());
    let controller = test_controller(store, runner.clone(), gate.clone());

    let error = translator_daemon::RoundTripController::start(&controller).unwrap_err();

    assert_eq!(error.code, "self_test_headphones_required");
    assert!(runner.starts.lock().unwrap().is_empty());
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn unavailable_facts_are_rejected_before_runner_or_gate_acquisition() {
    let gate = AudioOperationGate::new();
    let runner = Arc::new(FakeRunner::default());
    let controller = test_controller(RuntimeStore::default(), runner.clone(), gate.clone());

    let error = translator_daemon::RoundTripController::start(&controller).unwrap_err();

    assert_eq!(
        (error.status.as_u16(), error.code),
        (503, "audio_facts_unavailable")
    );
    assert!(runner.starts.lock().unwrap().is_empty());
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn stop_panic_retains_exact_runtime_and_gate_for_same_owner_retry() {
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let runtime_ids = Arc::new(Mutex::new(Vec::new()));
    let stop_calls = Arc::new(Mutex::new(Vec::new()));
    let drops = Arc::new(AtomicUsize::new(0));
    let progress = Arc::new(Mutex::new(None));
    let runner = Arc::new(PanicOnceRunner {
        runtime_ids: Arc::clone(&runtime_ids),
        stop_calls: Arc::clone(&stop_calls),
        drops: Arc::clone(&drops),
        progress,
    });
    let controller = test_controller(store.clone(), runner, gate.clone());
    let started = translator_daemon::RoundTripController::start(&controller).unwrap();
    let session_id = started.status.session_id.unwrap();

    let first_stop = translator_daemon::RoundTripController::stop(&controller);
    let status_after_panic = store.snapshot().self_test.status;
    let gate_after_panic = gate.state();
    let drops_after_panic = drops.load(Ordering::SeqCst);
    let second_stop = translator_daemon::RoundTripController::stop(&controller);
    let start_after_cleanup = translator_daemon::RoundTripController::start(&controller);
    let shutdown = controller.shutdown();
    let runtime_ids = runtime_ids.lock().unwrap().clone();
    let calls = stop_calls.lock().unwrap().clone();
    let final_drops = drops.load(Ordering::SeqCst);

    assert_eq!(first_stop.unwrap_err().code, "self_test_cleanup_pending");
    assert_eq!(
        status_after_panic.checkpoint,
        Some(RoundTripCheckpoint::Failed)
    );
    assert!(status_after_panic.cleanup_pending);
    assert_eq!(
        gate_after_panic,
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert_eq!(drops_after_panic, 0);
    let stopped = second_stop.expect("same owner must retry the retained runtime");
    assert_eq!(stopped.status.checkpoint, Some(RoundTripCheckpoint::Failed));
    assert!(!stopped.status.cleanup_pending);
    assert!(start_after_cleanup.is_err());
    assert_eq!(runtime_ids.len(), 1);
    assert_eq!(calls, vec![runtime_ids[0], runtime_ids[0]]);
    assert_eq!(final_drops, 1);
    assert!(shutdown.is_ok());
}

#[test]
fn completion_panic_keeps_same_owner_retryable_and_start_disabled() {
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let runtime_ids = Arc::new(Mutex::new(Vec::new()));
    let stop_calls = Arc::new(Mutex::new(Vec::new()));
    let drops = Arc::new(AtomicUsize::new(0));
    let progress = Arc::new(Mutex::new(None));
    let runner = Arc::new(PanicOnceRunner {
        runtime_ids: Arc::clone(&runtime_ids),
        stop_calls: Arc::clone(&stop_calls),
        drops: Arc::clone(&drops),
        progress: Arc::clone(&progress),
    });
    let controller = test_controller(store.clone(), runner, gate.clone());
    let started = translator_daemon::RoundTripController::start(&controller).unwrap();
    let session_id = started.status.session_id.unwrap();

    progress
        .lock()
        .unwrap()
        .clone()
        .unwrap()
        .completed(session_id);
    let wait_deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while !store.snapshot().self_test.status.cleanup_pending
        && std::time::Instant::now() < wait_deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let status_after_panic = store.snapshot().self_test.status;
    let gate_after_panic = gate.state();
    let drops_after_panic = drops.load(Ordering::SeqCst);
    let retry = translator_daemon::RoundTripController::stop(&controller);
    let start_after_cleanup = translator_daemon::RoundTripController::start(&controller);
    let shutdown = controller.shutdown();
    let runtime_ids = runtime_ids.lock().unwrap().clone();
    let calls = stop_calls.lock().unwrap().clone();
    let final_drops = drops.load(Ordering::SeqCst);

    assert_eq!(
        status_after_panic.checkpoint,
        Some(RoundTripCheckpoint::Failed)
    );
    assert!(status_after_panic.cleanup_pending);
    assert_eq!(
        gate_after_panic,
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert_eq!(drops_after_panic, 0);
    let stopped = retry.expect("explicit Stop must retry the completion owner");
    assert_eq!(stopped.status.checkpoint, Some(RoundTripCheckpoint::Failed));
    assert!(!stopped.status.cleanup_pending);
    assert!(start_after_cleanup.is_err());
    assert_eq!(runtime_ids.len(), 1);
    assert_eq!(calls, vec![runtime_ids[0], runtime_ids[0]]);
    assert_eq!(final_drops, 1);
    assert!(shutdown.is_ok());
}
