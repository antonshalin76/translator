use std::collections::VecDeque;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicUsize, Ordering},
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
    AudioMixKnowledge, AudioMixState, AudioOperationGate, AudioOperationState, ControlApplication,
    ControlCommand, ControlFailure, ControlToken, DirectionRuntimeFailure, DirectionRuntimeStatus,
    DuplexCompletionObserver, DuplexRunner, DuplexRuntimeError, DuplexStartFailure,
    DuplexStartResult, RuntimeMaintenance, RuntimeStatus, RuntimeStore, TranslationMixMode,
    build_router_with_controllers,
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
    start_failures: AtomicUsize,
    cleanup_start_failures: AtomicUsize,
    stop_calls: AtomicUsize,
    active: AtomicUsize,
    stop_failures: AtomicUsize,
    stop_barrier: Mutex<Option<Arc<StartBarrier>>>,
    completions: Mutex<Vec<(u64, Arc<dyn DuplexCompletionObserver>)>>,
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
        _deadline: tokio::time::Instant,
    ) -> DuplexStartResult {
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
    reconcile: Mutex<VecDeque<Result<(), ControlFailure>>>,
    recover: Mutex<VecDeque<Result<(), ControlFailure>>>,
}

impl TestMix {
    fn calls(&self) -> Vec<(&'static str, TranslationMixMode)> {
        self.calls.lock().unwrap().clone()
    }
}

impl AudioMixController for TestMix {
    fn apply_desired(
        &self,
        _volumes: AudioMixState,
        mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        self.calls.lock().unwrap().push(("apply", mode));
        Ok(())
    }

    fn reconcile_committed(&self, mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.calls.lock().unwrap().push(("reconcile", mode));
        self.reconcile.lock().unwrap().pop_front().unwrap_or(Ok(()))
    }

    fn recover_committed(&self, mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.calls.lock().unwrap().push(("recover", mode));
        self.recover.lock().unwrap().pop_front().unwrap_or(Ok(()))
    }
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
async fn shutdown_applies_bypass_only_after_native_cleanup() {
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
    assert_eq!(
        mix.calls(),
        [
            ("reconcile", TranslationMixMode::Translating),
            ("reconcile", TranslationMixMode::Bypass),
        ]
    );
    let snapshot = store.snapshot();
    assert_eq!(snapshot.runtime_status, RuntimeStatus::Stopped);
    assert_eq!(snapshot.audio_mix_knowledge, AudioMixKnowledge::Known);
}

#[tokio::test]
async fn failed_native_shutdown_keeps_translating_mix_and_retry_ownership() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    runner.state.stop_failures.store(1, Ordering::SeqCst);
    let mix = Arc::new(TestMix::default());
    let application = application_with_mix(
        store.clone(),
        runner.clone(),
        AudioOperationGate::new(),
        mix.clone(),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application.shutdown().await.unwrap_err().code,
        "translation_stop_failed"
    );
    assert_eq!(
        mix.calls(),
        [("reconcile", TranslationMixMode::Translating)]
    );
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
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
            ("reconcile", TranslationMixMode::Translating),
            ("reconcile", TranslationMixMode::Bypass),
        ]
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
        store,
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
            ("reconcile", TranslationMixMode::Translating),
            ("reconcile", TranslationMixMode::Bypass),
        ],
        "caller cancellation must not detach or duplicate the accepted bypass transaction"
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

    tokio::time::timeout(Duration::from_secs(1), async {
        while runner.state.active.load(Ordering::SeqCst) != 0
            || gate.state() != AudioOperationState::Idle
            || store.snapshot().runtime_status != RuntimeStatus::Stopped
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the final sender must run owned native cleanup");
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(
        mix.calls(),
        [
            ("reconcile", TranslationMixMode::Translating),
            ("reconcile", TranslationMixMode::Bypass),
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

    tokio::time::timeout(Duration::from_secs(1), async {
        while runner.state.active.load(Ordering::SeqCst) != 0
            || gate.state() != AudioOperationState::Idle
            || store.snapshot().runtime_status != RuntimeStatus::Stopped
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("receiver closure must retain and retry the cleanup-pending owner");
    assert_eq!(runner.state.stop_calls.load(Ordering::SeqCst), 3);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    assert_eq!(
        mix.calls()
            .iter()
            .filter(|call| **call == ("reconcile", TranslationMixMode::Bypass))
            .count(),
        1
    );
}

#[tokio::test]
async fn shutdown_unknown_mix_gets_one_explicit_recovery_and_failed_recovery_is_retryable() {
    let store = RuntimeStore::default();
    let runner = Arc::new(TestRunner::default());
    let mix = Arc::new(TestMix::default());
    mix.reconcile.lock().unwrap().extend([
        Ok(()),
        Err(mix_error("audio_mix_state_unknown")),
        Err(mix_error("audio_mix_state_unknown")),
    ]);
    mix.recover
        .lock()
        .unwrap()
        .extend([Err(mix_error("audio_mix_state_unknown")), Ok(())]);
    let application = application_with_mix(
        store.clone(),
        runner,
        AudioOperationGate::new(),
        mix.clone(),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application.shutdown().await.unwrap_err().code,
        "audio_mix_state_unknown"
    );
    let failed = store.snapshot();
    assert_eq!(failed.runtime_status, RuntimeStatus::Failed);
    assert_eq!(
        failed.audio_mix_knowledge,
        AudioMixKnowledge::AudioMixStateUnknown
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
            ("reconcile", TranslationMixMode::Translating),
            ("reconcile", TranslationMixMode::Bypass),
            ("recover", TranslationMixMode::Bypass),
            ("reconcile", TranslationMixMode::Bypass),
            ("recover", TranslationMixMode::Bypass),
        ]
    );
    assert_eq!(
        store.snapshot().audio_mix_knowledge,
        AudioMixKnowledge::Known
    );
}

#[tokio::test]
async fn ordinary_bypass_failure_is_not_silently_recovered() {
    let store = RuntimeStore::default();
    let mix = Arc::new(TestMix::default());
    mix.reconcile.lock().unwrap().extend([
        Ok(()),
        Err(mix_error("audio_mix_apply_failed")),
        Ok(()),
    ]);
    let application = application_with_mix(
        store.clone(),
        Arc::new(TestRunner::default()),
        AudioOperationGate::new(),
        mix.clone(),
    );

    application.execute(ControlCommand::Start).await.unwrap();
    assert_eq!(
        application.shutdown().await.unwrap_err().code,
        "audio_mix_apply_failed"
    );
    assert!(
        mix.calls()
            .iter()
            .all(|(operation, _)| *operation != "recover")
    );
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Failed);

    application.shutdown().await.unwrap();
    assert!(
        mix.calls()
            .iter()
            .all(|(operation, _)| *operation != "recover")
    );
}
