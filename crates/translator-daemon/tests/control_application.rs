use std::collections::VecDeque;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use http_body_util::BodyExt;
use tower::ServiceExt;
use translator_core::AudioDirection;
use translator_daemon::{
    ActiveDuplexRuntime, AdmittedDuplex, ApiControllers, ApiLimits, AudioMixController,
    AudioMixKnowledge, AudioMixPatch, AudioMixState, AudioOperationGate, AudioOperationState,
    ControlApplication, ControlCommand, ControlFailure, ControlToken, DirectionPatch,
    DirectionRuntimeFailure, DirectionRuntimeStatus, DuplexCompletionObserver, DuplexRunner,
    DuplexRuntimeError, DuplexStartFailure, DuplexStartResult, RuntimeMaintenance, RuntimeStatus,
    RuntimeStore, TranslationMixMode, build_router_with_controllers,
};

const CONTROL_TOKEN: &str = "4242424242424242424242424242424242424242424242424242424242424242";

struct NoopFacts;

impl translator_daemon::RuntimeFactsSource for NoopFacts {
    fn inspect(
        &self,
        _deadline: std::time::Instant,
    ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
        use translator_audio::{
            AecCapability, AudioGraphState, DeviceFacts, DeviceHealth, DeviceSelectionState,
            GraphHealth, OutputMode, PhysicalDevice, RouteResolution, RoutingState,
        };
        let selection = |name: &str| DeviceSelectionState {
            health: DeviceHealth::Available,
            selected: Some(PhysicalDevice {
                id: 1,
                name: name.into(),
                description: name.into(),
                active_port: None,
                active_port_type: None,
                available: true,
            }),
            pinned_name: Some(name.into()),
            current_default: Some(name.into()),
            pending_default: None,
        };
        Ok(translator_daemon::RuntimeFacts {
            devices: DeviceFacts {
                source: selection("alsa_input.physical"),
                sink: selection("alsa_output.physical"),
                output_mode: OutputMode::Headphones,
                aec_capability: AecCapability::Unavailable,
            },
            audio_graph: AudioGraphState {
                health: GraphHealth::Ready,
                endpoints: Vec::new(),
                owned_module_ids: Vec::new(),
                safe_error: None,
            },
            routes: RoutingState {
                candidates: Vec::new(),
                source_outputs: Vec::new(),
                conflicting_stream_ids: Vec::new(),
                active_route: None,
                resolution: RouteResolution::NoCandidate,
            },
        })
    }
}

impl RuntimeMaintenance for NoopFacts {
    fn refresh(&self, _store: &RuntimeStore) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.refresh(store)
    }

    fn verify_bypass_custody(
        &self,
        _: &translator_daemon::RuntimeSnapshot,
        _: bool,
    ) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn prepare_start(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        Ok(())
    }
}

#[derive(Default)]
struct RecoverableSpeakerFacts {
    speaker_present: AtomicBool,
    repairs: AtomicUsize,
    switch_to_open_speaker_on_repair: AtomicBool,
    open_speaker: AtomicBool,
}

impl translator_daemon::RuntimeFactsSource for RecoverableSpeakerFacts {
    fn inspect(
        &self,
        deadline: std::time::Instant,
    ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
        let mut facts = NoopFacts.inspect(deadline)?;
        if self.open_speaker.load(Ordering::SeqCst) {
            facts.devices.output_mode = translator_audio::OutputMode::OpenSpeaker;
        }
        Ok(facts)
    }
}

impl RuntimeMaintenance for RecoverableSpeakerFacts {
    fn refresh(&self, _: &RuntimeStore) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        let facts = translator_daemon::RuntimeFactsSource::inspect(
            self,
            std::time::Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        store.set_devices(facts.devices.into());
        store.set_audio_graph(facts.audio_graph);
        store.set_routes(facts.routes);
        Ok(())
    }

    fn prepare_start(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn prepare_bypass(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        self.repairs.fetch_add(1, Ordering::SeqCst);
        self.speaker_present.store(true, Ordering::SeqCst);
        if self.switch_to_open_speaker_on_repair.load(Ordering::SeqCst) {
            self.open_speaker.store(true, Ordering::SeqCst);
        }
        Ok(())
    }

    fn verify_bypass_custody(
        &self,
        _: &translator_daemon::RuntimeSnapshot,
        _: bool,
    ) -> Result<(), ControlFailure> {
        self.speaker_present
            .load(Ordering::SeqCst)
            .then_some(())
            .ok_or_else(|| mix_error("original_loopback_custody_unknown"))
    }
}

struct SwitchableOutputFacts {
    mode: Mutex<translator_audio::OutputMode>,
}

struct SwitchableBypassFacts {
    mode: Mutex<translator_audio::OutputMode>,
    mic_custody: AtomicBool,
}

impl translator_daemon::RuntimeFactsSource for SwitchableBypassFacts {
    fn inspect(
        &self,
        deadline: std::time::Instant,
    ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
        let mut facts = NoopFacts.inspect(deadline)?;
        facts.devices.output_mode = *self.mode.lock().unwrap();
        Ok(facts)
    }
}

impl RuntimeMaintenance for SwitchableBypassFacts {
    fn refresh(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.refresh_bypass_facts(store)
    }

    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        let facts = translator_daemon::RuntimeFactsSource::inspect(
            self,
            std::time::Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        store.set_devices(facts.devices.into());
        Ok(())
    }

    fn verify_bypass_custody(
        &self,
        _: &translator_daemon::RuntimeSnapshot,
        permit_mic_original: bool,
    ) -> Result<(), ControlFailure> {
        if permit_mic_original && !self.mic_custody.load(Ordering::SeqCst) {
            Err(mix_error("original_loopback_custody_unknown"))
        } else {
            Ok(())
        }
    }

    fn prepare_start(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        Ok(())
    }
}

impl translator_daemon::RuntimeFactsSource for SwitchableOutputFacts {
    fn inspect(
        &self,
        deadline: std::time::Instant,
    ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
        let mut facts = NoopFacts.inspect(deadline)?;
        facts.devices.output_mode = *self.mode.lock().unwrap();
        Ok(facts)
    }
}

struct FailRefreshOnce(AtomicUsize);

impl RuntimeMaintenance for FailRefreshOnce {
    fn refresh(&self, _store: &RuntimeStore) -> Result<(), ControlFailure> {
        if self
            .0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            return Err(mix_error("original_loopback_custody_unknown"));
        }
        Ok(())
    }

    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.refresh(store)
    }

    fn verify_bypass_custody(
        &self,
        _: &translator_daemon::RuntimeSnapshot,
        _: bool,
    ) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn prepare_start(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        Ok(())
    }
}

struct ChangedDeviceFacts;

impl RuntimeMaintenance for ChangedDeviceFacts {
    fn refresh(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        let mut devices = store.snapshot().devices.expect("running device facts");
        devices.source.health = translator_audio::DeviceHealth::DeviceUnavailable;
        devices.acoustic.full_duplex_allowed = false;
        store.set_devices(devices);
        Ok(())
    }

    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.refresh(store)
    }

    fn verify_bypass_custody(
        &self,
        _: &translator_daemon::RuntimeSnapshot,
        _: bool,
    ) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn prepare_start(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        Ok(())
    }
}

struct FailedRefresh;

impl RuntimeMaintenance for FailedRefresh {
    fn refresh(&self, _: &RuntimeStore) -> Result<(), ControlFailure> {
        Err(ControlFailure {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "original_loopback_custody_unknown",
        })
    }

    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.refresh(store)
    }

    fn verify_bypass_custody(
        &self,
        _: &translator_daemon::RuntimeSnapshot,
        _: bool,
    ) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn prepare_start(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        Ok(())
    }
}

struct ChangedSystemDefault;

impl RuntimeMaintenance for ChangedSystemDefault {
    fn refresh(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        let mut devices = store.snapshot().devices.expect("running device facts");
        devices.source.current_default = Some("alsa_input.other".into());
        devices.sink.current_default = Some("alsa_output.other".into());
        store.set_devices(devices);
        Ok(())
    }

    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.refresh(store)
    }

    fn verify_bypass_custody(
        &self,
        _: &translator_daemon::RuntimeSnapshot,
        _: bool,
    ) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn prepare_start(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        Ok(())
    }
}

struct LostAudioGraphOrRoute {
    graph: bool,
}

impl RuntimeMaintenance for LostAudioGraphOrRoute {
    fn refresh(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        if self.graph {
            let mut graph = store.snapshot().audio_graph.expect("running graph facts");
            graph.health = translator_audio::GraphHealth::Error;
            store.set_audio_graph(graph);
        } else {
            store.clear_routes("route_reconciliation_failed");
        }
        Ok(())
    }

    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.refresh(store)
    }

    fn verify_bypass_custody(
        &self,
        _: &translator_daemon::RuntimeSnapshot,
        _: bool,
    ) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn prepare_start(&self, _: &translator_daemon::RuntimeSnapshot) -> Result<(), ControlFailure> {
        Ok(())
    }
}

#[derive(Default)]
struct StartBarrier {
    entered: AtomicUsize,
    released: (Mutex<bool>, Condvar),
}

impl StartBarrier {
    fn wait(&self) -> Result<(), DuplexRuntimeError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        let released = self.released.0.lock().unwrap();
        let (released, timeout) = self
            .released
            .1
            .wait_timeout_while(released, Duration::from_secs(2), |released| !*released)
            .unwrap();
        if timeout.timed_out() && !*released {
            return Err(DuplexRuntimeError::StartFailed);
        }
        Ok(())
    }

    fn release(&self) {
        *self.released.0.lock().unwrap() = true;
        self.released.1.notify_all();
    }
}

#[derive(Default)]
struct TestState {
    starts: AtomicUsize,
    start_budgets: Mutex<Vec<Duration>>,
    start_failures: AtomicUsize,
    cleanup_start_failures: AtomicUsize,
    stop_calls: AtomicUsize,
    active: AtomicUsize,
    stop_failures: AtomicUsize,
    stop_barrier: Mutex<Option<Arc<StartBarrier>>>,
    completions: Mutex<Vec<(u64, Arc<dyn DuplexCompletionObserver>)>>,
    allow_reconfigure: AtomicBool,
}

#[derive(Default)]
struct TestRunner {
    state: Arc<TestState>,
    start_barrier: Option<Arc<StartBarrier>>,
}

impl DuplexRunner for TestRunner {
    fn start(
        &self,
        _admitted: AdmittedDuplex,
        deadline: tokio::time::Instant,
    ) -> DuplexStartResult {
        self.state
            .start_budgets
            .lock()
            .unwrap()
            .push(deadline.saturating_duration_since(tokio::time::Instant::now()));
        if let Some(barrier) = &self.start_barrier {
            barrier.wait().map_err(DuplexStartFailure::rejected)?;
        }
        self.state.starts.fetch_add(1, Ordering::SeqCst);
        self.state.active.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(TestActive {
            state: self.state.clone(),
            stopped: false,
        }))
    }

    fn start_supervised(
        &self,
        admitted: AdmittedDuplex,
        generation: u64,
        completion: Arc<dyn DuplexCompletionObserver>,
        deadline: tokio::time::Instant,
    ) -> DuplexStartResult {
        self.state
            .completions
            .lock()
            .unwrap()
            .push((generation, completion));
        if self
            .state
            .cleanup_start_failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            self.state.starts.fetch_add(1, Ordering::SeqCst);
            self.state.active.fetch_add(1, Ordering::SeqCst);
            let mut cleanup = TestActive {
                state: self.state.clone(),
                stopped: false,
            };
            assert_eq!(cleanup.stop(deadline), Err(DuplexRuntimeError::StopFailed));
            return Err(DuplexStartFailure::cleanup_pending(
                DuplexRuntimeError::StartFailed,
                Box::new(cleanup),
            ));
        }
        if self
            .state
            .start_failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            self.state.starts.fetch_add(1, Ordering::SeqCst);
            return Err(DuplexStartFailure::rejected(
                DuplexRuntimeError::StartFailed,
            ));
        }
        let runtime = self.start(admitted, deadline)?;
        Ok(runtime)
    }
}

struct SecretCleanup(&'static str);

impl ActiveDuplexRuntime for SecretCleanup {
    fn stop(&mut self, _deadline: tokio::time::Instant) -> Result<(), DuplexRuntimeError> {
        let _ = self.0;
        Ok(())
    }
}

struct TestActive {
    state: Arc<TestState>,
    stopped: bool,
}

impl ActiveDuplexRuntime for TestActive {
    fn reconfigure(
        &mut self,
        _: AdmittedDuplex,
        _: tokio::time::Instant,
    ) -> Result<(), DuplexRuntimeError> {
        if self.state.allow_reconfigure.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(DuplexRuntimeError::ReconfigureFailed)
        }
    }
    fn stop(&mut self, _deadline: tokio::time::Instant) -> Result<(), DuplexRuntimeError> {
        self.state.stop_calls.fetch_add(1, Ordering::SeqCst);
        let barrier = self.state.stop_barrier.lock().unwrap().clone();
        if let Some(barrier) = barrier {
            barrier.wait()?;
        }
        if self
            .state
            .stop_failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            return Err(DuplexRuntimeError::StopFailed);
        }
        if !self.stopped {
            self.stopped = true;
            self.state.active.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(())
    }
}

fn application(
    store: RuntimeStore,
    runner: Arc<dyn DuplexRunner>,
    gate: AudioOperationGate,
) -> Arc<ControlApplication> {
    ControlApplication::spawn(
        store,
        runner,
        gate,
        Arc::new(NoopFacts),
        Arc::new(NoopFacts),
        None,
    )
}

fn lifecycle_router(store: RuntimeStore, application: Arc<ControlApplication>) -> axum::Router {
    build_router_with_controllers(
        store,
        ControlToken::parse(CONTROL_TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(application),
            ..ApiControllers::default()
        },
    )
}

#[tokio::test]
async fn cold_start_reaches_the_runner_with_the_existing_model_readiness_budget() {
    let state = Arc::new(TestState::default());
    let controller = application(
        RuntimeStore::default(),
        Arc::new(TestRunner {
            state: state.clone(),
            start_barrier: None,
        }),
        AudioOperationGate::new(),
    );
    let started = controller.execute(ControlCommand::Start).await.unwrap();
    assert!(started.translation_running);
    let budget = state.start_budgets.lock().unwrap()[0];
    controller.execute(ControlCommand::Stop).await.unwrap();
    controller.shutdown().await.unwrap();
    assert!(
        budget >= Duration::from_secs(125),
        "cold Start received {budget:?}"
    );
    assert!(budget <= Duration::from_secs(130));
    assert_eq!(state.active.load(Ordering::SeqCst), 0);
}

async fn open_lifecycle_events(router: axum::Router) -> Body {
    let response = router
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/events/stream")
                .header("authorization", format!("Bearer {CONTROL_TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    response.into_body()
}

async fn next_snapshot_event(events: &mut Body) {
    let frame = tokio::time::timeout(Duration::from_secs(1), events.frame())
        .await
        .expect("serialized snapshot event timed out")
        .expect("the SSE stream ended unexpectedly")
        .unwrap();
    let text = std::str::from_utf8(&frame.into_data().unwrap())
        .unwrap()
        .to_owned();
    assert!(text.contains("event: snapshot"));
}

#[derive(Default)]
struct TestMix {
    calls: Mutex<Vec<(&'static str, TranslationMixMode)>>,
    preflight: Mutex<VecDeque<Result<(), ControlFailure>>>,
    reconcile: Mutex<VecDeque<Result<(), ControlFailure>>>,
    recover: Mutex<VecDeque<Result<(), ControlFailure>>>,
    bypass_gate: Mutex<Option<AudioOperationGate>>,
    bypass_gate_states: Mutex<Vec<AudioOperationState>>,
}

impl TestMix {
    fn calls(&self) -> Vec<(&'static str, TranslationMixMode)> {
        self.calls.lock().unwrap().clone()
    }
}

impl AudioMixController for TestMix {
    fn validate_desired(&self, _volumes: AudioMixState) -> Result<(), ControlFailure> {
        if let Some(gate) = self.bypass_gate.lock().unwrap().as_ref() {
            self.bypass_gate_states.lock().unwrap().push(gate.state());
        }
        self.preflight.lock().unwrap().pop_front().unwrap_or(Ok(()))
    }

    fn apply_desired(
        &self,
        _volumes: AudioMixState,
        mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        self.calls.lock().unwrap().push(("apply", mode));
        if matches!(
            mode,
            TranslationMixMode::Bypass | TranslationMixMode::MicrophoneMutedBypass
        ) {
            if let Some(gate) = self.bypass_gate.lock().unwrap().as_ref() {
                self.bypass_gate_states.lock().unwrap().push(gate.state());
            }
        }
        Ok(())
    }

    fn reconcile_committed(&self, mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.calls.lock().unwrap().push(("reconcile", mode));
        if mode == TranslationMixMode::Bypass {
            if let Some(gate) = self.bypass_gate.lock().unwrap().as_ref() {
                self.bypass_gate_states.lock().unwrap().push(gate.state());
            }
        }
        self.reconcile.lock().unwrap().pop_front().unwrap_or(Ok(()))
    }

    fn recover_committed(&self, mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.calls.lock().unwrap().push(("recover", mode));
        self.recover.lock().unwrap().pop_front().unwrap_or(Ok(()))
    }
}

struct OriginalCustodyMaintenance {
    gate: AudioOperationGate,
    fail_prepare: AtomicBool,
    fail_cleanup: AtomicBool,
    cleanups: AtomicUsize,
    prepared_microphones: Mutex<Vec<bool>>,
}

impl RuntimeMaintenance for OriginalCustodyMaintenance {
    fn refresh(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        NoopFacts.refresh(store)
    }
    fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        let facts = translator_daemon::RuntimeFactsSource::inspect(
            &NoopFacts,
            std::time::Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        store.set_devices(facts.devices.into());
        store.set_audio_graph(facts.audio_graph);
        store.set_routes(facts.routes);
        Ok(())
    }
    fn verify_bypass_custody(
        &self,
        snapshot: &translator_daemon::RuntimeSnapshot,
        permit: bool,
    ) -> Result<(), ControlFailure> {
        NoopFacts.verify_bypass_custody(snapshot, permit)
    }
    fn prepare_start(
        &self,
        snapshot: &translator_daemon::RuntimeSnapshot,
    ) -> Result<(), ControlFailure> {
        self.prepared_microphones.lock().unwrap().push(
            snapshot
                .directions
                .iter()
                .find(|direction| direction.direction_id == AudioDirection::Microphone)
                .unwrap()
                .enabled,
        );
        if self.fail_prepare.load(Ordering::SeqCst) {
            Err(mix_error("original_loopback_custody_unknown"))
        } else {
            Ok(())
        }
    }
    fn cleanup_originals(&self, deadline: std::time::Instant) -> Result<(), ControlFailure> {
        assert_eq!(self.gate.state(), AudioOperationState::Production);
        assert!(deadline > std::time::Instant::now());
        self.cleanups.fetch_add(1, Ordering::SeqCst);
        if self.fail_cleanup.load(Ordering::SeqCst) {
            Err(mix_error("original_loopback_custody_unknown"))
        } else {
            Ok(())
        }
    }
}

#[tokio::test]
async fn original_start_faults_retain_exclusive_custody_until_joined_recovery() {
    for preparation_failure in [true, false] {
        let store = RuntimeStore::default();
        let gate = AudioOperationGate::new();
        let runner = Arc::new(TestRunner::default());
        let mix = Arc::new(TestMix::default());
        let maintenance = Arc::new(OriginalCustodyMaintenance {
            gate: gate.clone(),
            fail_prepare: AtomicBool::new(preparation_failure),
            fail_cleanup: AtomicBool::new(true),
            cleanups: AtomicUsize::new(0),
            prepared_microphones: Mutex::new(Vec::new()),
        });
        if !preparation_failure {
            mix.reconcile
                .lock()
                .unwrap()
                .push_back(Err(mix_error("audio_mix_state_unknown")));
        }
        let application = ControlApplication::spawn(
            store.clone(),
            runner.clone(),
            gate.clone(),
            Arc::new(NoopFacts),
            maintenance.clone(),
            Some(mix.clone()),
        );
        assert_eq!(
            application
                .execute(ControlCommand::Start)
                .await
                .unwrap_err()
                .code,
            "translation_cleanup_pending"
        );
        assert_eq!(runner.state.starts.load(Ordering::SeqCst), 0);
        assert_eq!(maintenance.cleanups.load(Ordering::SeqCst), 1);
        assert_eq!(gate.state(), AudioOperationState::Production);
        assert_eq!(
            store.snapshot().audio_mix_knowledge,
            AudioMixKnowledge::AudioMixStateUnknown
        );
        assert_eq!(
            application
                .execute(ControlCommand::RecoverAudioMix)
                .await
                .unwrap_err()
                .code,
            "translation_cleanup_pending"
        );
        assert_eq!(gate.state(), AudioOperationState::Production);
        maintenance.fail_cleanup.store(false, Ordering::SeqCst);
        maintenance.fail_prepare.store(false, Ordering::SeqCst);
        application
            .execute(ControlCommand::RecoverAudioMix)
            .await
            .unwrap();
        assert_eq!(gate.state(), AudioOperationState::Idle);
        assert_eq!(
            store.snapshot().audio_mix_knowledge,
            AudioMixKnowledge::Known
        );
        application.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn original_unknown_reconcile_joins_raw_custody_and_stops_translation() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    let maintenance = Arc::new(OriginalCustodyMaintenance {
        gate: gate.clone(),
        fail_prepare: AtomicBool::new(false),
        fail_cleanup: AtomicBool::new(true),
        cleanups: AtomicUsize::new(0),
        prepared_microphones: Mutex::new(Vec::new()),
    });
    let application = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        gate.clone(),
        Arc::new(NoopFacts),
        maintenance.clone(),
        Some(mix.clone()),
    );
    application.execute(ControlCommand::Start).await.unwrap();
    mix.reconcile
        .lock()
        .unwrap()
        .push_back(Err(mix_error("audio_mix_state_unknown")));
    assert_eq!(
        application
            .execute(ControlCommand::ReconcileAudio)
            .await
            .unwrap_err()
            .code,
        "translation_cleanup_pending"
    );
    assert_eq!(
        runner.state.stop_calls.load(Ordering::SeqCst),
        1,
        "raw cleanup failure must not skip translation stop"
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(gate.state(), AudioOperationState::Production);
    maintenance.fail_cleanup.store(false, Ordering::SeqCst);
    application
        .execute(ControlCommand::RecoverAudioMix)
        .await
        .unwrap();
    assert_eq!(maintenance.cleanups.load(Ordering::SeqCst), 2);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn disabled_microphone_keeps_desired_gains_but_uses_muted_running_policy() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let mix = Arc::new(TestMix::default());
    let runner = Arc::new(TestRunner::default());
    runner.state.allow_reconfigure.store(true, Ordering::SeqCst);
    let maintenance = Arc::new(OriginalCustodyMaintenance {
        gate: gate.clone(),
        fail_prepare: AtomicBool::new(false),
        fail_cleanup: AtomicBool::new(false),
        cleanups: AtomicUsize::new(0),
        prepared_microphones: Mutex::new(Vec::new()),
    });
    let application = ControlApplication::spawn(
        store.clone(),
        runner,
        gate,
        Arc::new(NoopFacts),
        maintenance.clone(),
        Some(mix.clone()),
    );
    application.execute(ControlCommand::Start).await.unwrap();
    application
        .execute(ControlCommand::PatchAudioMix(AudioMixPatch {
            microphone_original_percent: Some(35),
            microphone_translation_percent: None,
            speaker_original_percent: None,
            speaker_translation_percent: Some(63),
        }))
        .await
        .unwrap();
    application
        .execute(ControlCommand::PatchDirection(DirectionPatch {
            direction_id: AudioDirection::Microphone,
            enabled: Some(false),
            source_language: None,
            target_language: None,
        }))
        .await
        .unwrap();
    assert_eq!(
        maintenance.prepared_microphones.lock().unwrap().last(),
        Some(&false)
    );
    assert_eq!(
        mix.calls().last(),
        Some(&("reconcile", TranslationMixMode::TranslatingMicrophoneMuted))
    );
    assert_eq!(store.snapshot().audio_mix.microphone_original_percent, 35);
    application
        .execute(ControlCommand::PatchAudioMix(AudioMixPatch {
            microphone_original_percent: Some(47),
            microphone_translation_percent: None,
            speaker_original_percent: None,
            speaker_translation_percent: None,
        }))
        .await
        .unwrap();
    assert_eq!(
        mix.calls().last(),
        Some(&("apply", TranslationMixMode::TranslatingMicrophoneMuted))
    );
    application
        .execute(ControlCommand::ReconcileAudio)
        .await
        .unwrap();
    assert_eq!(
        mix.calls().last(),
        Some(&("reconcile", TranslationMixMode::TranslatingMicrophoneMuted))
    );
    assert_eq!(store.snapshot().audio_mix.speaker_translation_percent, 63);
    application.execute(ControlCommand::Stop).await.unwrap();
    application
        .execute(ControlCommand::PatchAudioMix(AudioMixPatch {
            microphone_original_percent: None,
            microphone_translation_percent: None,
            speaker_original_percent: None,
            speaker_translation_percent: Some(61),
        }))
        .await
        .unwrap();
    application
        .execute(ControlCommand::PatchDirection(DirectionPatch {
            direction_id: AudioDirection::Microphone,
            enabled: Some(true),
            source_language: None,
            target_language: None,
        }))
        .await
        .unwrap();
    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        maintenance.prepared_microphones.lock().unwrap().last(),
        Some(&true)
    );
    assert_eq!(
        mix.calls().last(),
        Some(&("reconcile", TranslationMixMode::Translating))
    );
    application.execute(ControlCommand::Stop).await.unwrap();
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn stopped_missing_original_creation_uses_fresh_headphone_facts() {
    struct PreparingFacts {
        inner: SwitchableBypassFacts,
        prepares: AtomicUsize,
    }
    impl translator_daemon::RuntimeFactsSource for PreparingFacts {
        fn inspect(
            &self,
            deadline: std::time::Instant,
        ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
            translator_daemon::RuntimeFactsSource::inspect(&self.inner, deadline)
        }
    }
    impl RuntimeMaintenance for PreparingFacts {
        fn refresh(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
            self.inner.refresh(store)
        }
        fn refresh_bypass_facts(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
            self.inner.refresh_bypass_facts(store)
        }
        fn verify_bypass_custody(
            &self,
            snapshot: &translator_daemon::RuntimeSnapshot,
            permit: bool,
        ) -> Result<(), ControlFailure> {
            self.inner.verify_bypass_custody(snapshot, permit)
        }
        fn prepare_start(
            &self,
            snapshot: &translator_daemon::RuntimeSnapshot,
        ) -> Result<(), ControlFailure> {
            assert!(
                snapshot
                    .devices
                    .as_ref()
                    .unwrap()
                    .acoustic
                    .mode
                    .is_headphones()
            );
            self.prepares.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
    for mode in [
        translator_audio::OutputMode::Headphones,
        translator_audio::OutputMode::OpenSpeaker,
    ] {
        let store = RuntimeStore::default();
        let gate = AudioOperationGate::new();
        let mix = Arc::new(TestMix::default());
        let facts = Arc::new(PreparingFacts {
            inner: SwitchableBypassFacts {
                mode: Mutex::new(translator_audio::OutputMode::Headphones),
                mic_custody: AtomicBool::new(true),
            },
            prepares: AtomicUsize::new(0),
        });
        let application = ControlApplication::spawn(
            store.clone(),
            Arc::new(TestRunner::default()),
            gate.clone(),
            facts.clone(),
            facts.clone(),
            Some(mix.clone()),
        );
        application.execute(ControlCommand::Start).await.unwrap();
        application.execute(ControlCommand::Stop).await.unwrap();
        facts.prepares.store(0, Ordering::SeqCst);
        *facts.inner.mode.lock().unwrap() = mode;
        mix.preflight
            .lock()
            .unwrap()
            .push_back(Err(mix_error("microphone_original_unavailable")));
        if !mode.is_headphones() {
            mix.preflight
                .lock()
                .unwrap()
                .push_back(Err(mix_error("microphone_original_unavailable")));
        }
        let result = application
            .execute(ControlCommand::PatchAudioMix(AudioMixPatch {
                microphone_original_percent: Some(35),
                microphone_translation_percent: None,
                speaker_original_percent: None,
                speaker_translation_percent: None,
            }))
            .await;
        if mode.is_headphones() {
            result.unwrap();
            assert_eq!(facts.prepares.load(Ordering::SeqCst), 1);
            assert_eq!(store.snapshot().audio_mix.microphone_original_percent, 35);
        } else {
            assert_eq!(result.unwrap_err().code, "microphone_original_unavailable");
            assert_eq!(
                facts.prepares.load(Ordering::SeqCst),
                0,
                "revoked headphones cannot open raw capture"
            );
            assert_eq!(store.snapshot().audio_mix.microphone_original_percent, 0);
        }
        assert_eq!(gate.state(), AudioOperationState::Idle);
        application.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn unsupported_stopped_mix_patch_releases_lease_without_mutating_audio_or_status() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let mix = Arc::new(TestMix::default());
    *mix.bypass_gate.lock().unwrap() = Some(gate.clone());
    mix.preflight
        .lock()
        .unwrap()
        .push_back(Err(mix_error("audio_mix_discovery_failed")));
    let application = application_with_mix(
        store.clone(),
        Arc::new(TestRunner::default()),
        gate.clone(),
        mix.clone(),
    );

    assert_eq!(
        application
            .execute(ControlCommand::PatchAudioMix(AudioMixPatch {
                microphone_original_percent: Some(25),
                microphone_translation_percent: None,
                speaker_original_percent: None,
                speaker_translation_percent: None,
            }))
            .await
            .unwrap_err()
            .code,
        "audio_mix_discovery_failed"
    );
    assert_eq!(
        mix.bypass_gate_states.lock().unwrap().as_slice(),
        &[AudioOperationState::Production]
    );
    assert!(mix.calls().is_empty());
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(store.snapshot().audio_mix, AudioMixState::default());
    assert_eq!(gate.state(), AudioOperationState::Idle);
    application.execute(ControlCommand::Start).await.unwrap();
    application.execute(ControlCommand::Stop).await.unwrap();
    application.shutdown().await.unwrap();
}

fn application_with_mix(
    store: RuntimeStore,
    runner: Arc<dyn DuplexRunner>,
    gate: AudioOperationGate,
    mix: Arc<dyn AudioMixController>,
) -> Arc<ControlApplication> {
    ControlApplication::spawn(
        store,
        runner,
        gate,
        Arc::new(NoopFacts),
        Arc::new(NoopFacts),
        Some(mix),
    )
}

struct LifecycleProjectionMix {
    store: RuntimeStore,
    observed: Mutex<Vec<RuntimeStatus>>,
}

impl AudioMixController for LifecycleProjectionMix {
    fn apply_desired(
        &self,
        _: AudioMixState,
        mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        self.reconcile_committed(mode)
    }

    fn reconcile_committed(&self, _: TranslationMixMode) -> Result<(), ControlFailure> {
        self.observed
            .lock()
            .unwrap()
            .push(self.store.snapshot().runtime_status);
        Ok(())
    }

    fn recover_committed(&self, mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.reconcile_committed(mode)
    }
}

#[tokio::test]
async fn stopped_audio_reconciliation_does_not_project_nonexistent_runtime_cleanup() {
    let store = RuntimeStore::default();
    let facts =
        translator_daemon::RuntimeFactsSource::inspect(&NoopFacts, std::time::Instant::now())
            .unwrap();
    store.set_devices(facts.devices.into());
    store.set_audio_graph(facts.audio_graph);
    store.set_routes(facts.routes);
    let mix = Arc::new(LifecycleProjectionMix {
        store: store.clone(),
        observed: Mutex::new(Vec::new()),
    });
    let state = Arc::new(TestState::default());
    let controller = application_with_mix(
        store.clone(),
        Arc::new(TestRunner {
            state: state.clone(),
            start_barrier: None,
        }),
        AudioOperationGate::new(),
        mix.clone(),
    );
    controller
        .execute(ControlCommand::ReconcileAudio)
        .await
        .unwrap();
    let statuses = mix.observed.lock().unwrap().clone();
    controller.shutdown().await.unwrap();
    assert!(!statuses.is_empty());
    assert!(
        statuses
            .iter()
            .all(|status| *status == RuntimeStatus::Stopped),
        "{statuses:?}"
    );
    assert_eq!(state.starts.load(Ordering::SeqCst), 0);
}

async fn assert_owned_shutdown_after_safety_stop(
    application: &ControlApplication,
    store: &RuntimeStore,
    runner: &TestRunner,
    mix: &TestMix,
    gate: &AudioOperationGate,
) {
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Production);
    let mut expected = mix.calls();
    expected.push((
        "reconcile",
        TranslationMixMode::Quarantine {
            mic_original_expected: false,
        },
    ));
    application.shutdown().await.unwrap();
    assert_eq!(mix.calls(), expected);
    assert!(expected.iter().all(|(_, mode)| !matches!(
        mode,
        TranslationMixMode::Bypass | TranslationMixMode::MicrophoneMutedBypass
    )));
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::Known
    );
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(
        application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err()
            .code,
        "translation_controller_unavailable",
    );
}

#[tokio::test]
async fn missing_speaker_bypass_is_repaired_only_after_native_stop() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let facts = Arc::new(RecoverableSpeakerFacts::default());
    facts.speaker_present.store(true, Ordering::SeqCst);
    let mix = Arc::new(TestMix::default());
    let application = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        gate.clone(),
        facts.clone(),
        facts.clone(),
        Some(mix.clone()),
    );
    application.execute(ControlCommand::Start).await.unwrap();
    facts.speaker_present.store(false, Ordering::SeqCst);
    runner.state.stop_failures.store(1, Ordering::SeqCst);

    assert_eq!(
        application
            .execute(ControlCommand::Stop)
            .await
            .unwrap_err()
            .code,
        "translation_stop_failed"
    );
    assert_eq!(facts.repairs.load(Ordering::SeqCst), 0);
    assert_eq!(gate.state(), AudioOperationState::Production);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );

    application.execute(ControlCommand::Stop).await.unwrap();
    assert_eq!(facts.repairs.load(Ordering::SeqCst), 1);
    assert_eq!(
        mix.calls().last(),
        Some(&("reconcile", TranslationMixMode::Bypass))
    );
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn output_switch_during_bypass_repair_never_unmutes_raw_microphone() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let facts = Arc::new(RecoverableSpeakerFacts::default());
    facts.speaker_present.store(true, Ordering::SeqCst);
    facts
        .switch_to_open_speaker_on_repair
        .store(true, Ordering::SeqCst);
    let mix = Arc::new(TestMix::default());
    let application = ControlApplication::spawn(
        store.clone(),
        Arc::new(TestRunner::default()),
        gate.clone(),
        facts.clone(),
        facts.clone(),
        Some(mix.clone()),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    application.execute(ControlCommand::Stop).await.unwrap();
    assert_eq!(facts.repairs.load(Ordering::SeqCst), 1);
    assert_eq!(
        mix.calls().last(),
        Some(&("reconcile", TranslationMixMode::MicrophoneMutedBypass))
    );
    assert!(
        !mix.calls()
            .iter()
            .any(|call| { *call == ("reconcile", TranslationMixMode::Bypass) })
    );
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    application.shutdown().await.unwrap();
}

const fn mix_error(code: &'static str) -> ControlFailure {
    ControlFailure {
        status: StatusCode::CONFLICT,
        code,
    }
}

#[tokio::test]
async fn failed_stop_retains_runtime_and_lease_until_retry_succeeds() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    runner.state.stop_failures.store(1, Ordering::SeqCst);
    let application = application(store.clone(), runner.clone(), gate.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    let failed = application.execute(ControlCommand::Stop).await.unwrap_err();
    assert_eq!(failed.code, "translation_stop_failed");
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Production);
    assert!(!store.snapshot().translation_running);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );

    let overlapping = application
        .execute(ControlCommand::Start)
        .await
        .unwrap_err();
    assert_eq!(overlapping.code, "translation_cleanup_pending");
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);

    application.execute(ControlCommand::Stop).await.unwrap();
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    application.execute(ControlCommand::Stop).await.unwrap();
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn stop_holds_production_lease_until_headphone_bypass_is_verified() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let mix = Arc::new(TestMix::default());
    *mix.bypass_gate.lock().unwrap() = Some(gate.clone());
    let application = application_with_mix(
        store.clone(),
        Arc::new(TestRunner::default()),
        gate.clone(),
        mix.clone(),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    application.execute(ControlCommand::Stop).await.unwrap();

    assert_eq!(
        mix.bypass_gate_states.lock().unwrap().as_slice(),
        &[AudioOperationState::Production]
    );
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn stale_open_speaker_facts_keep_microphone_muted_and_block_start() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let mix = Arc::new(TestMix::default());
    let facts = Arc::new(SwitchableOutputFacts {
        mode: Mutex::new(translator_audio::OutputMode::Headphones),
    });
    let application = ControlApplication::spawn(
        store.clone(),
        Arc::new(TestRunner::default()),
        gate.clone(),
        facts.clone(),
        Arc::new(NoopFacts),
        Some(mix.clone()),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    *facts.mode.lock().unwrap() = translator_audio::OutputMode::OpenSpeaker;
    assert_eq!(
        application
            .execute(ControlCommand::Stop)
            .await
            .unwrap_err()
            .code,
        "translation_precondition_failed"
    );
    assert!(
        !mix.calls()
            .iter()
            .any(|(_, mode)| *mode == TranslationMixMode::Bypass)
    );
    assert_eq!(gate.state(), AudioOperationState::Production);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert_eq!(
        application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err()
            .code,
        "translation_cleanup_pending"
    );

    *facts.mode.lock().unwrap() = translator_audio::OutputMode::Headphones;
    application.execute(ControlCommand::Stop).await.unwrap();
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn stopped_mix_patch_revalidates_without_unmuting_open_speaker_microphone() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let mix = Arc::new(TestMix::default());
    *mix.bypass_gate.lock().unwrap() = Some(gate.clone());
    let facts = Arc::new(SwitchableBypassFacts {
        mode: Mutex::new(translator_audio::OutputMode::Headphones),
        mic_custody: AtomicBool::new(true),
    });
    let application = ControlApplication::spawn(
        store.clone(),
        Arc::new(TestRunner::default()),
        gate.clone(),
        facts.clone(),
        facts.clone(),
        Some(mix.clone()),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    application.execute(ControlCommand::Stop).await.unwrap();
    *facts.mode.lock().unwrap() = translator_audio::OutputMode::OpenSpeaker;
    facts.mic_custody.store(false, Ordering::SeqCst);
    application.execute(ControlCommand::Stop).await.unwrap();
    assert_eq!(
        mix.calls().last(),
        Some(&("reconcile", TranslationMixMode::MicrophoneMutedBypass))
    );
    application
        .execute(ControlCommand::PatchAudioMix(AudioMixPatch {
            microphone_original_percent: Some(100),
            microphone_translation_percent: None,
            speaker_original_percent: None,
            speaker_translation_percent: None,
        }))
        .await
        .unwrap();

    assert_eq!(
        mix.calls().last(),
        Some(&("apply", TranslationMixMode::MicrophoneMutedBypass))
    );
    assert_eq!(
        mix.bypass_gate_states.lock().unwrap().last(),
        Some(&AudioOperationState::Production)
    );
    assert_eq!(store.snapshot().audio_mix.microphone_original_percent, 100);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    application
        .execute(ControlCommand::ReconcileAudio)
        .await
        .unwrap();
    assert_eq!(
        mix.calls().last(),
        Some(&("reconcile", TranslationMixMode::MicrophoneMutedBypass))
    );
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn stopped_unknown_mix_patch_cannot_apply_desired_or_open_start_gate() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let mix = Arc::new(TestMix::default());
    let application = application_with_mix(
        store.clone(),
        Arc::new(TestRunner::default()),
        gate.clone(),
        mix.clone(),
    );
    application.execute(ControlCommand::Start).await.unwrap();
    application.execute(ControlCommand::Stop).await.unwrap();
    mix.reconcile
        .lock()
        .unwrap()
        .push_back(Err(mix_error("audio_mix_state_unknown")));

    assert_eq!(
        application
            .execute(ControlCommand::PatchAudioMix(AudioMixPatch {
                microphone_original_percent: Some(100),
                microphone_translation_percent: None,
                speaker_original_percent: None,
                speaker_translation_percent: None,
            }))
            .await
            .unwrap_err()
            .code,
        "audio_mix_state_unknown"
    );
    assert!(!matches!(mix.calls().last(), Some(("apply", _))));
    assert_eq!(gate.state(), AudioOperationState::Production);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::AudioMixStateUnknown
    );
    assert_eq!(
        application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err()
            .code,
        "translation_cleanup_pending"
    );
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn mic_only_stop_uses_speaker_only_safety_validation_when_raw_custody_is_absent() {
    let store = RuntimeStore::default();
    let mix = Arc::new(TestMix::default());
    let facts = Arc::new(SwitchableBypassFacts {
        mode: Mutex::new(translator_audio::OutputMode::Headphones),
        mic_custody: AtomicBool::new(false),
    });
    let application = ControlApplication::spawn(
        store.clone(),
        Arc::new(TestRunner::default()),
        AudioOperationGate::new(),
        facts.clone(),
        facts,
        Some(mix.clone()),
    );
    application
        .execute(ControlCommand::PatchDirection(DirectionPatch {
            direction_id: AudioDirection::Speaker,
            source_language: None,
            target_language: None,
            enabled: Some(false),
        }))
        .await
        .unwrap();
    application.execute(ControlCommand::Start).await.unwrap();
    application.execute(ControlCommand::Stop).await.unwrap();

    assert_eq!(
        mix.calls().last(),
        Some(&("reconcile", TranslationMixMode::MicrophoneMutedBypass))
    );
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn failed_loopback_refresh_keeps_stop_quarantined_until_retry() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let mix = Arc::new(TestMix::default());
    let application = ControlApplication::spawn(
        store.clone(),
        Arc::new(TestRunner::default()),
        gate.clone(),
        Arc::new(NoopFacts),
        Arc::new(FailRefreshOnce(AtomicUsize::new(1))),
        Some(mix.clone()),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application
            .execute(ControlCommand::Stop)
            .await
            .unwrap_err()
            .code,
        "original_loopback_custody_unknown"
    );
    assert!(
        !mix.calls()
            .iter()
            .any(|(_, mode)| *mode == TranslationMixMode::Bypass)
    );
    assert_eq!(gate.state(), AudioOperationState::Production);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );

    application.execute(ControlCommand::Stop).await.unwrap();
    assert_eq!(gate.state(), AudioOperationState::Idle);
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn failed_start_cleanup_is_projected_and_retried_by_the_same_owner() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    runner
        .state
        .cleanup_start_failures
        .store(1, Ordering::SeqCst);
    runner.state.stop_failures.store(2, Ordering::SeqCst);
    let application = application(store.clone(), runner.clone(), gate.clone());

    let failed = application
        .execute(ControlCommand::Start)
        .await
        .unwrap_err();
    assert_eq!(failed.code, "translation_cleanup_pending");
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Production);
    assert!(!store.snapshot().translation_running);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );

    let overlapping = application
        .execute(ControlCommand::Start)
        .await
        .unwrap_err();
    assert_eq!(overlapping.code, "translation_cleanup_pending");
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);

    let retry_failed = application.execute(ControlCommand::Stop).await.unwrap_err();
    assert_eq!(retry_failed.code, "translation_stop_failed");
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 2);
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Production);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );

    application.execute(ControlCommand::Stop).await.unwrap();
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 3);
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn cleanup_pending_watchdog_cannot_restore_audio_from_start_quarantine() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    runner
        .state
        .cleanup_start_failures
        .store(1, Ordering::SeqCst);
    runner.state.stop_failures.store(1, Ordering::SeqCst);
    let mix = Arc::new(TestMix::default());
    let application = application_with_mix(
        store.clone(),
        runner,
        AudioOperationGate::new(),
        mix.clone(),
    );

    assert_eq!(
        application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err()
            .code,
        "translation_cleanup_pending"
    );
    assert_eq!(
        application
            .execute(ControlCommand::ReconcileAudio)
            .await
            .unwrap_err()
            .code,
        "translation_cleanup_pending"
    );
    assert_eq!(
        mix.calls(),
        [(
            "reconcile",
            TranslationMixMode::Quarantine {
                mic_original_expected: false
            }
        )]
    );
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn device_loss_quarantines_and_stops_live_microphone_before_reconcile() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    let application = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        gate.clone(),
        Arc::new(NoopFacts),
        Arc::new(ChangedDeviceFacts),
        Some(mix.clone()),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(
        application
            .execute(ControlCommand::ReconcileAudio)
            .await
            .unwrap_err()
            .code,
        "translation_precondition_failed"
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert_eq!(
        mix.calls().last(),
        Some(&(
            "reconcile",
            TranslationMixMode::Quarantine {
                mic_original_expected: false
            }
        ))
    );
    assert_owned_shutdown_after_safety_stop(&application, &store, &runner, &mix, &gate).await;
    assert!(
        !mix.calls()
            .iter()
            .any(|(_, mode)| *mode == TranslationMixMode::Bypass)
    );
}

#[tokio::test]
async fn loopback_refresh_failure_quarantines_and_stops_live_microphone() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    let application = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        gate.clone(),
        Arc::new(NoopFacts),
        Arc::new(FailedRefresh),
        Some(mix.clone()),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application
            .execute(ControlCommand::ReconcileAudio)
            .await
            .unwrap_err()
            .code,
        "translation_precondition_failed"
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert_eq!(
        mix.calls().last(),
        Some(&(
            "reconcile",
            TranslationMixMode::Quarantine {
                mic_original_expected: false
            }
        ))
    );
    assert_owned_shutdown_after_safety_stop(&application, &store, &runner, &mix, &gate).await;
    assert!(
        !mix.calls()
            .iter()
            .any(|(_, mode)| *mode == TranslationMixMode::Bypass)
    );
}

#[tokio::test]
async fn failed_quarantine_still_stops_pcm_and_records_unknown_mix() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    mix.reconcile.lock().unwrap().extend([
        Ok(()),
        Ok(()),
        Err(mix_error("audio_mix_state_unknown")),
    ]);
    let application = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        gate.clone(),
        Arc::new(NoopFacts),
        Arc::new(FailedRefresh),
        Some(mix.clone()),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(
        application
            .execute(ControlCommand::ReconcileAudio)
            .await
            .unwrap_err()
            .code,
        "audio_mix_state_unknown"
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::AudioMixStateUnknown
    );
    assert_eq!(
        mix.calls().last(),
        Some(&(
            "reconcile",
            TranslationMixMode::Quarantine {
                mic_original_expected: false
            }
        ))
    );
    assert_owned_shutdown_after_safety_stop(&application, &store, &runner, &mix, &gate).await;
    assert!(
        !mix.calls()
            .iter()
            .any(|(_, mode)| *mode == TranslationMixMode::Bypass)
    );
}

#[tokio::test]
async fn system_default_change_does_not_stop_pinned_live_microphone() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    let application = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        AudioOperationGate::new(),
        Arc::new(NoopFacts),
        Arc::new(ChangedSystemDefault),
        Some(mix),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    application
        .execute(ControlCommand::ReconcileAudio)
        .await
        .unwrap();
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 0);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Running);
    application.execute(ControlCommand::Stop).await.unwrap();
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn lost_graph_or_route_quarantines_and_stops_live_microphone() {
    for graph in [true, false] {
        let store = RuntimeStore::default();
        let gate = AudioOperationGate::new();
        let runner = Arc::new(TestRunner::default());
        let mix = Arc::new(TestMix::default());
        let application = ControlApplication::spawn(
            store.clone(),
            runner.clone(),
            gate.clone(),
            Arc::new(NoopFacts),
            Arc::new(LostAudioGraphOrRoute { graph }),
            Some(mix.clone()),
        );

        application.execute(ControlCommand::Start).await.unwrap();
        assert_eq!(
            application
                .execute(ControlCommand::ReconcileAudio)
                .await
                .unwrap_err()
                .code,
            "translation_precondition_failed"
        );
        assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
        assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            store.snapshot().runtime_status,
            RuntimeStatus::CleanupPending
        );
        assert_eq!(
            mix.calls().last(),
            Some(&(
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ))
        );
        assert_owned_shutdown_after_safety_stop(&application, &store, &runner, &mix, &gate).await;
        assert!(
            !mix.calls()
                .iter()
                .any(|(_, mode)| *mode == TranslationMixMode::Bypass)
        );
    }
}

#[tokio::test]
async fn speaker_only_start_skips_microphone_quarantine_and_lost_route_stops_runtime() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    let application = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        gate.clone(),
        Arc::new(NoopFacts),
        Arc::new(LostAudioGraphOrRoute { graph: false }),
        Some(mix.clone()),
    );
    application
        .execute(ControlCommand::PatchDirection(DirectionPatch {
            direction_id: AudioDirection::Microphone,
            source_language: None,
            target_language: None,
            enabled: Some(false),
        }))
        .await
        .unwrap();

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        mix.calls(),
        [("reconcile", TranslationMixMode::TranslatingMicrophoneMuted)]
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(
        application
            .execute(ControlCommand::ReconcileAudio)
            .await
            .unwrap_err()
            .code,
        "translation_precondition_failed"
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert_owned_shutdown_after_safety_stop(&application, &store, &runner, &mix, &gate).await;
}

#[test]
fn start_failure_debug_never_formats_the_cleanup_owner() {
    let failure = DuplexStartFailure::cleanup_pending(
        DuplexRuntimeError::StartFailed,
        Box::new(SecretCleanup("must-not-appear")),
    );

    let debug = format!("{failure:?}");
    assert!(debug.contains("StartFailed"));
    assert!(debug.contains("cleanup: true"));
    assert!(!debug.contains("must-not-appear"));
}

#[tokio::test]
async fn admission_owns_one_executing_and_one_cancelled_waiter() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let barrier = Arc::new(StartBarrier::default());
    let runner = Arc::new(TestRunner {
        state: Arc::new(TestState::default()),
        start_barrier: Some(barrier.clone()),
    });
    let application = application(store.clone(), runner.clone(), gate);

    let first = tokio::spawn({
        let application = application.clone();
        async move { application.execute(ControlCommand::Start).await }
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while barrier.entered.load(Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let waiter = tokio::spawn({
        let application = application.clone();
        async move { application.execute(ControlCommand::Stop).await }
    });
    tokio::task::yield_now().await;
    waiter.abort();
    let _ = waiter.await;

    for _ in 0..100 {
        let busy = application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err();
        assert_eq!(busy.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(busy.code, "translation_control_busy");
    }
    barrier.release();
    first.await.unwrap().unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while store.snapshot().translation_running
            || runner.state.active.load(Ordering::SeqCst) != 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the admitted cancelled waiter must finish Stop");

    application.execute(ControlCommand::Start).await.unwrap();
    application.execute(ControlCommand::Stop).await.unwrap();
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn terminal_completion_reaps_only_the_matching_generation() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let application = application(store.clone(), runner.clone(), gate.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    let (generation_a, completion_a) = runner.state.completions.lock().unwrap()[0].clone();
    application.execute(ControlCommand::Stop).await.unwrap();
    application.execute(ControlCommand::Start).await.unwrap();
    let (generation_b, completion_b) = runner.state.completions.lock().unwrap()[1].clone();

    completion_a.completed(generation_a, Err(DuplexRuntimeError::StartFailed));
    tokio::task::yield_now().await;
    assert!(store.snapshot().translation_running);
    assert_eq!(gate.state(), AudioOperationState::Production);

    completion_b.completed(generation_b, Err(DuplexRuntimeError::StartFailed));
    tokio::time::timeout(Duration::from_secs(1), async {
        while store.snapshot().translation_running {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Failed);
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn terminal_completion_is_monotonic_and_publishes_final_state_once() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    let application = application(store.clone(), runner.clone(), AudioOperationGate::new());
    let router = lifecycle_router(store.clone(), application.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    let (generation, completion) = runner.state.completions.lock().unwrap()[0].clone();
    let mut events = open_lifecycle_events(router).await;
    next_snapshot_event(&mut events).await;

    completion.completed(generation, Err(DuplexRuntimeError::StartFailed));
    completion.cleanup_started(generation);
    completion.completed(generation, Err(DuplexRuntimeError::StartFailed));
    completion.cleanup_started(generation.saturating_sub(1));
    completion.completed(
        generation.saturating_sub(1),
        Err(DuplexRuntimeError::StartFailed),
    );
    for epoch in 1..=10_000 {
        completion.direction_status_changed(
            generation,
            AudioDirection::Microphone,
            epoch,
            DirectionRuntimeStatus::Recovering,
            None,
        );
    }
    assert_eq!(
        application
            .execute(ControlCommand::RecoverAudioMix)
            .await
            .unwrap_err()
            .code,
        "audio_mix_controller_unavailable",
        "the serialized command is the lifecycle-processing barrier"
    );

    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Failed);
    next_snapshot_event(&mut events).await;
    assert!(
        futures_util::poll!(events.frame()).is_pending(),
        "reversed, duplicate, stale, and post-terminal direction updates must not republish final state"
    );
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn direction_health_is_serialized_and_fenced_by_generation_and_epoch() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    let application = application(store.clone(), runner.clone(), AudioOperationGate::new());
    let router = lifecycle_router(store.clone(), application.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    let (generation_a, completion_a) = runner.state.completions.lock().unwrap()[0].clone();
    let mut events = open_lifecycle_events(router.clone()).await;
    next_snapshot_event(&mut events).await;
    let statuses = || {
        store
            .snapshot()
            .directions
            .into_iter()
            .map(|direction| {
                (
                    direction.direction_id,
                    direction.runtime_status,
                    direction.runtime_failure,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        statuses(),
        [
            (
                AudioDirection::Microphone,
                DirectionRuntimeStatus::Running,
                None,
            ),
            (
                AudioDirection::Speaker,
                DirectionRuntimeStatus::Running,
                None,
            ),
        ]
    );

    completion_a.direction_status_changed(
        generation_a,
        AudioDirection::Microphone,
        7,
        DirectionRuntimeStatus::Recovering,
        None,
    );
    wait_for_direction_status(
        &store,
        AudioDirection::Microphone,
        DirectionRuntimeStatus::Recovering,
    )
    .await;
    next_snapshot_event(&mut events).await;
    completion_a.direction_status_changed(
        generation_a,
        AudioDirection::Microphone,
        8,
        DirectionRuntimeStatus::Failed,
        Some(DirectionRuntimeFailure::RestartExhausted),
    );
    wait_for_direction_status(
        &store,
        AudioDirection::Microphone,
        DirectionRuntimeStatus::Failed,
    )
    .await;
    next_snapshot_event(&mut events).await;
    assert_eq!(
        statuses(),
        [
            (
                AudioDirection::Microphone,
                DirectionRuntimeStatus::Failed,
                Some(DirectionRuntimeFailure::RestartExhausted),
            ),
            (
                AudioDirection::Speaker,
                DirectionRuntimeStatus::Running,
                None,
            ),
        ]
    );

    completion_a.direction_status_changed(
        generation_a,
        AudioDirection::Microphone,
        7,
        DirectionRuntimeStatus::Running,
        None,
    );
    completion_a.direction_status_changed(
        generation_a,
        AudioDirection::Microphone,
        9,
        DirectionRuntimeStatus::Recovering,
        None,
    );
    wait_for_direction_status(
        &store,
        AudioDirection::Microphone,
        DirectionRuntimeStatus::Recovering,
    )
    .await;
    next_snapshot_event(&mut events).await;
    assert!(
        futures_util::poll!(events.frame()).is_pending(),
        "the stale epoch must not publish a snapshot before the FIFO sentinel"
    );

    for _ in 0..2 {
        completion_a.direction_status_changed(
            generation_a,
            AudioDirection::Microphone,
            9,
            DirectionRuntimeStatus::Recovering,
            None,
        );
    }
    completion_a.direction_status_changed(
        generation_a,
        AudioDirection::Microphone,
        9,
        DirectionRuntimeStatus::Running,
        None,
    );
    wait_for_direction_status(
        &store,
        AudioDirection::Microphone,
        DirectionRuntimeStatus::Running,
    )
    .await;
    next_snapshot_event(&mut events).await;
    assert!(
        futures_util::poll!(events.frame()).is_pending(),
        "duplicate matching lifecycle events must not republish snapshots"
    );

    completion_a.direction_status_changed(
        generation_a,
        AudioDirection::Microphone,
        9,
        DirectionRuntimeStatus::Failed,
        Some(DirectionRuntimeFailure::RestartExhausted),
    );
    assert_eq!(
        application
            .execute(ControlCommand::ReconcileAudio)
            .await
            .unwrap_err()
            .code,
        "audio_mix_controller_unavailable",
        "the serialized command acknowledges all earlier lifecycle updates"
    );
    assert_eq!(
        statuses(),
        [
            (
                AudioDirection::Microphone,
                DirectionRuntimeStatus::Running,
                None,
            ),
            (
                AudioDirection::Speaker,
                DirectionRuntimeStatus::Running,
                None,
            ),
        ],
        "a Failed state cannot overwrite a Running state at the same producer epoch"
    );
    assert!(futures_util::poll!(events.frame()).is_pending());

    completion_a.direction_status_changed(
        generation_a,
        AudioDirection::Microphone,
        10,
        DirectionRuntimeStatus::Failed,
        Some(DirectionRuntimeFailure::RestartExhausted),
    );
    wait_for_direction_status(
        &store,
        AudioDirection::Microphone,
        DirectionRuntimeStatus::Failed,
    )
    .await;
    next_snapshot_event(&mut events).await;
    assert_eq!(
        statuses(),
        [
            (
                AudioDirection::Microphone,
                DirectionRuntimeStatus::Failed,
                Some(DirectionRuntimeFailure::RestartExhausted),
            ),
            (
                AudioDirection::Speaker,
                DirectionRuntimeStatus::Running,
                None,
            ),
        ],
        "a fresh producer epoch must project Failed while preserving its peer"
    );

    drop(events);
    application.execute(ControlCommand::Stop).await.unwrap();
    application.execute(ControlCommand::Start).await.unwrap();
    let (generation_b, completion_b) = runner.state.completions.lock().unwrap()[1].clone();
    assert_ne!(generation_a, generation_b);
    let mut events = open_lifecycle_events(router).await;
    next_snapshot_event(&mut events).await;
    completion_a.direction_status_changed(
        generation_a,
        AudioDirection::Microphone,
        u64::MAX,
        DirectionRuntimeStatus::Failed,
        Some(DirectionRuntimeFailure::RestartExhausted),
    );
    completion_b.direction_status_changed(
        generation_b,
        AudioDirection::Microphone,
        1,
        DirectionRuntimeStatus::Recovering,
        None,
    );
    wait_for_direction_status(
        &store,
        AudioDirection::Microphone,
        DirectionRuntimeStatus::Recovering,
    )
    .await;
    next_snapshot_event(&mut events).await;
    assert!(
        futures_util::poll!(events.frame()).is_pending(),
        "the prior generation must not overwrite or republish before the FIFO sentinel"
    );

    application.execute(ControlCommand::Stop).await.unwrap();
    assert!(statuses().iter().all(|(_, status, failure)| {
        *status == DirectionRuntimeStatus::Stopped && failure.is_none()
    }));
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn lifecycle_burst_is_coalesced_and_cannot_starve_an_accepted_stop() {
    let store = RuntimeStore::default();
    let barrier = Arc::new(StartBarrier::default());
    let runner = Arc::new(TestRunner {
        state: Arc::new(TestState::default()),
        start_barrier: Some(barrier.clone()),
    });
    let application = application(store, runner.clone(), AudioOperationGate::new());
    let start = tokio::spawn({
        let application = application.clone();
        async move { application.execute(ControlCommand::Start).await }
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while barrier.entered.load(Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("Start must hold the serialized owner before the lifecycle burst");
    let (generation, completion) = runner.state.completions.lock().unwrap()[0].clone();
    for epoch in 1..=100_000 {
        for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
            completion.direction_status_changed(
                generation,
                direction,
                epoch,
                DirectionRuntimeStatus::Recovering,
                None,
            );
        }
    }
    let mut stop = tokio::spawn({
        let application = application.clone();
        async move { application.execute(ControlCommand::Stop).await }
    });
    barrier.release();
    start.await.unwrap().unwrap();

    let stopped_without_backlog = tokio::time::timeout(Duration::from_millis(250), &mut stop).await;
    let (stopped_quickly, result) = match stopped_without_backlog {
        Ok(result) => (true, result),
        Err(_) => (
            false,
            tokio::time::timeout(Duration::from_secs(15), &mut stop)
                .await
                .expect("the RED fail-safe must still drain and join the queued Stop"),
        ),
    };
    result.unwrap().unwrap();
    application.shutdown().await.unwrap();
    assert!(
        stopped_quickly,
        "lifecycle state must occupy fixed coalesced slots instead of an unbounded FIFO"
    );
}

async fn wait_for_direction_status(
    store: &RuntimeStore,
    direction: AudioDirection,
    expected: DirectionRuntimeStatus,
) {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let status = store
                .snapshot()
                .directions
                .into_iter()
                .find(|candidate| candidate.direction_id == direction)
                .unwrap()
                .runtime_status;
            if status == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("direction status must be committed by the serialized owner");
}

#[tokio::test]
async fn failed_start_consumes_generation_before_a_retry_can_run() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    runner.state.start_failures.store(1, Ordering::SeqCst);
    let application = application(store.clone(), runner.clone(), gate.clone());

    assert_eq!(
        application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err()
            .code,
        "translation_start_failed"
    );
    application.execute(ControlCommand::Start).await.unwrap();

    let completions = runner.state.completions.lock().unwrap().clone();
    let (failed_generation, failed_completion) = completions[0].clone();
    let (running_generation, _) = completions[1].clone();
    assert_ne!(failed_generation, running_generation);

    failed_completion.completed(failed_generation, Err(DuplexRuntimeError::StartFailed));
    tokio::task::yield_now().await;
    assert!(store.snapshot().translation_running);
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Production);

    application.execute(ControlCommand::Stop).await.unwrap();
    application.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_quarantines_and_completes_native_cleanup_without_bypass() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    let application = application_with_mix(
        store.clone(),
        runner.clone(),
        AudioOperationGate::new(),
        mix.clone(),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    application.shutdown().await.unwrap();

    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
        ]
    );
    let snapshot = store.snapshot();
    assert_eq!(snapshot.runtime_status, RuntimeStatus::Stopped);
    assert_eq!(snapshot.audio_mix_knowledge, AudioMixKnowledge::Known);
    assert_eq!(
        application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err()
            .code,
        "translation_controller_unavailable"
    );
}

#[tokio::test]
async fn failed_native_shutdown_keeps_quarantine_and_retry_ownership() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    runner.state.stop_failures.store(1, Ordering::SeqCst);
    let mix = Arc::new(TestMix::default());
    let application =
        application_with_mix(store.clone(), runner.clone(), gate.clone(), mix.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application.shutdown().await.unwrap_err().code,
        "translation_stop_failed"
    );
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
        ]
    );
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Production);
    let rejected = application
        .execute(ControlCommand::Start)
        .await
        .unwrap_err();
    assert_eq!(rejected.code, "translation_controller_unavailable");
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);

    application.shutdown().await.unwrap();
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
        ]
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 2);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::Known
    );
}

#[tokio::test]
async fn cancelled_accepted_shutdown_attaches_to_one_native_cleanup_and_stays_closed() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    let barrier = Arc::new(StartBarrier::default());
    *runner.state.stop_barrier.lock().unwrap() = Some(barrier.clone());
    let application = application_with_mix(
        store.clone(),
        runner.clone(),
        AudioOperationGate::new(),
        mix.clone(),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    let cancelled = tokio::spawn({
        let application = application.clone();
        async move { application.shutdown().await }
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while barrier.entered.load(Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("shutdown must be accepted before caller cancellation");
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        mix.calls().last(),
        Some(&(
            "reconcile",
            TranslationMixMode::Quarantine {
                mic_original_expected: false
            },
        )),
        "quarantine must precede completion of the held native stop"
    );
    cancelled.abort();
    let _ = cancelled.await;
    barrier.release();

    tokio::time::timeout(Duration::from_secs(1), async {
        while runner.state.active.load(Ordering::SeqCst) != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the accepted shutdown must finish after its caller disconnects");
    application
        .shutdown()
        .await
        .expect("the next caller must attach to and reap the accepted shutdown");
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
        ],
        "caller cancellation must not detach or duplicate the accepted quarantine transaction"
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::Known
    );
    let rejected = application
        .execute(ControlCommand::Start)
        .await
        .unwrap_err();
    assert_eq!(rejected.code, "translation_controller_unavailable");
}

#[tokio::test]
async fn cancelled_accepted_failed_shutdown_replays_result_before_one_retry() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    runner.state.stop_failures.store(1, Ordering::SeqCst);
    let barrier = Arc::new(StartBarrier::default());
    *runner.state.stop_barrier.lock().unwrap() = Some(barrier.clone());
    let application = application(store, runner.clone(), AudioOperationGate::new());

    application.execute(ControlCommand::Start).await.unwrap();
    let cancelled = tokio::spawn({
        let application = application.clone();
        async move { application.shutdown().await }
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while barrier.entered.load(Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the first shutdown must be accepted before cancellation");
    cancelled.abort();
    let _ = cancelled.await;
    barrier.release();
    tokio::time::timeout(Duration::from_secs(1), async {
        while runner.state.stop_calls.load(Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the accepted failed cleanup must publish its result");

    let observed = application.shutdown().await;
    let calls_after_observation = runner.state.stop_calls.load(Ordering::SeqCst);
    let retry = application.shutdown().await;
    let rejected = application.execute(ControlCommand::Start).await;

    assert_eq!(
        observed.unwrap_err().code,
        "translation_stop_failed",
        "the next caller must observe the exact stored result without retrying"
    );
    assert_eq!(calls_after_observation, 1);
    retry.expect("only the call after error observation may admit one cleanup retry");
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        rejected.unwrap_err().code,
        "translation_controller_unavailable"
    );
}

#[tokio::test]
async fn dropping_last_application_handle_drains_active_runtime_and_mix() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    let application =
        application_with_mix(store.clone(), runner.clone(), gate.clone(), mix.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    drop(application);

    tokio::time::timeout(translator_daemon::RUNTIME_CLEANUP_BUDGET, async {
        while runner.state.active.load(Ordering::SeqCst) != 0
            || gate.state() != AudioOperationState::Stopping
            || store.snapshot().runtime_status != RuntimeStatus::Stopped
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the final sender must run owned native cleanup");
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Stopping);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::Known
    );
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
        ]
    );
}

#[tokio::test]
async fn dropping_last_handle_retries_the_same_cleanup_pending_owner() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    runner.state.stop_failures.store(2, Ordering::SeqCst);
    let mix = Arc::new(TestMix::default());
    let application =
        application_with_mix(store.clone(), runner.clone(), gate.clone(), mix.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application
            .execute(ControlCommand::Stop)
            .await
            .unwrap_err()
            .code,
        "translation_stop_failed"
    );
    drop(application);

    tokio::time::timeout(translator_daemon::RUNTIME_CLEANUP_BUDGET, async {
        while runner.state.active.load(Ordering::SeqCst) != 0
            || gate.state() != AudioOperationState::Stopping
            || store.snapshot().runtime_status != RuntimeStatus::Stopped
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("receiver closure must retain and retry the cleanup-pending owner");
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 3);
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Stopping);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::Known
    );
    assert_eq!(
        mix.calls()
            .iter()
            .filter(|call| **call == ("reconcile", TranslationMixMode::Bypass))
            .count(),
        0
    );
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
        ]
    );
}

#[tokio::test]
async fn shutdown_unknown_mix_gets_one_explicit_recovery_and_failed_recovery_is_retryable() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    mix.reconcile.lock().unwrap().extend([
        Ok(()),
        Ok(()),
        Err(mix_error("audio_mix_state_unknown")),
        Err(mix_error("audio_mix_state_unknown")),
    ]);
    mix.recover
        .lock()
        .unwrap()
        .extend([Err(mix_error("audio_mix_state_unknown")), Ok(())]);
    let application =
        application_with_mix(store.clone(), runner.clone(), gate.clone(), mix.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application.shutdown().await.unwrap_err().code,
        "audio_mix_state_unknown"
    );
    let failed = store.snapshot();
    assert_eq!(failed.runtime_status, RuntimeStatus::CleanupPending);
    assert_eq!(
        failed.audio_mix_knowledge,
        AudioMixKnowledge::AudioMixStateUnknown
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Production);
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            (
                "recover",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
        ]
    );
    assert_eq!(
        application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err()
            .code,
        "translation_controller_unavailable",
        "the first shutdown call closes normal admission permanently"
    );

    application.shutdown().await.unwrap();
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            (
                "recover",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            (
                "recover",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
        ]
    );
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::Known
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[tokio::test]
async fn ordinary_bypass_failure_is_not_silently_recovered() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    mix.reconcile.lock().unwrap().extend([
        Ok(()),
        Ok(()),
        Ok(()),
        Err(mix_error("audio_mix_apply_failed")),
        Ok(()),
        Ok(()),
    ]);
    let application =
        application_with_mix(store.clone(), runner.clone(), gate.clone(), mix.clone());

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application
            .execute(ControlCommand::Stop)
            .await
            .unwrap_err()
            .code,
        "audio_mix_apply_failed"
    );
    assert!(
        mix.calls()
            .iter()
            .all(|(operation, _)| *operation != "recover")
    );
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Production);

    application.execute(ControlCommand::Stop).await.unwrap();
    assert!(
        mix.calls()
            .iter()
            .all(|(operation, _)| *operation != "recover")
    );
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::Known
    );
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(
        mix.calls(),
        [
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Translating),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Bypass),
            (
                "reconcile",
                TranslationMixMode::Quarantine {
                    mic_original_expected: false
                }
            ),
            ("reconcile", TranslationMixMode::Bypass),
        ]
    );
    // Operational Stop leaves ordinary admission open; owned shutdown closes it.
    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 2);
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 1);
    application.shutdown().await.unwrap();
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 2);
    assert_eq!(runner.state.active.load(Ordering::SeqCst), 0);
    assert_eq!(
        mix.calls().last(),
        Some(&(
            "reconcile",
            TranslationMixMode::Quarantine {
                mic_original_expected: false
            },
        ))
    );
    assert_eq!(
        application
            .execute(ControlCommand::Start)
            .await
            .unwrap_err()
            .code,
        "translation_controller_unavailable"
    );
}
