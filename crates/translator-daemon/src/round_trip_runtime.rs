use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use axum::http::StatusCode;
use thiserror::Error;
use tokio::sync::{mpsc, watch};
use translator_audio::{GraphHealth, OutputMode, RouteResolution};
use translator_core::ProviderId;
use uuid::Uuid;

use crate::{
    AdmittedDuplex, AudioOperationGate, AudioOperationLease, ControlFailure, ExactPcmProof,
    FactsError, RoundTripCheckpoint, RoundTripController, RoundTripErrorCode, RoundTripLatency,
    RoundTripPreconditions, RoundTripSelfTest, RoundTripSelfTestState, RuntimeFactsSource,
    RuntimeSnapshot, RuntimeStore,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RoundTripRuntimeError {
    #[error("round-trip runtime could not start")]
    StartFailed,
    #[error("round-trip runtime could not stop")]
    StopFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RoundTripOwnerStartError {
    #[error("round-trip owner thread could not be spawned")]
    ThreadSpawn,
    #[error("round-trip owner runtime could not be built")]
    RuntimeBuild,
    #[error("round-trip owner failed before becoming ready")]
    OwnerFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RoundTripOwnerShutdownError {
    #[error("round-trip cleanup is still pending")]
    CleanupPending,
    #[error("round-trip owner failed")]
    OwnerFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundTripTerminal {
    Completed,
    Stopped,
    Failed(RoundTripErrorCode),
}

pub trait ActiveRoundTripRuntime: Send {
    fn stop(
        &mut self,
        outer_deadline: Instant,
        cleanup_deadline: Instant,
    ) -> Result<RoundTripTerminal, RoundTripRuntimeError>;
}

pub trait RoundTripRunner: Send + Sync {
    fn start(
        &self,
        admitted: AdmittedDuplex,
        session_id: Uuid,
        progress: RoundTripProgress,
        start_deadline: Instant,
    ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError>;
}

struct ProgressState {
    session: Mutex<RoundTripSelfTest>,
    preconditions: Mutex<Option<RoundTripPreconditions>>,
    store: RuntimeStore,
    completion: watch::Sender<Option<Uuid>>,
}

#[derive(Clone)]
pub struct RoundTripProgress {
    state: Arc<ProgressState>,
}

impl RoundTripProgress {
    pub fn advance(
        &self,
        session_id: Uuid,
        checkpoint: RoundTripCheckpoint,
        latency: RoundTripLatency,
    ) -> Result<(), RoundTripErrorCode> {
        let result = lock_recovering(&self.state.session).advance(session_id, checkpoint, latency);
        self.publish();
        result
    }

    pub fn set_exact_pcm_proof(
        &self,
        session_id: Uuid,
        proof: ExactPcmProof,
    ) -> Result<(), RoundTripErrorCode> {
        let result = lock_recovering(&self.state.session).set_exact_pcm_proof(session_id, proof);
        self.publish();
        result
    }

    pub fn record_recursion_trigger(&self, session_id: Uuid) -> Result<(), RoundTripErrorCode> {
        let result = lock_recovering(&self.state.session).record_recursion_trigger(session_id);
        self.publish();
        result
    }

    pub fn fail(&self, session_id: Uuid, error: RoundTripErrorCode) -> bool {
        let failed = lock_recovering(&self.state.session).fail(session_id, error);
        self.publish();
        failed
    }

    pub fn set_cleanup_pending(&self, session_id: Uuid, pending: bool) {
        lock_recovering(&self.state.session).set_cleanup_pending(session_id, pending);
        self.publish();
    }

    pub fn completed(&self, session_id: Uuid) {
        self.state.completion.send_replace(Some(session_id));
    }

    pub fn terminal(&self, successful: bool) -> RoundTripTerminal {
        let status = lock_recovering(&self.state.session).status(false);
        match status.safe_error {
            Some(error) => RoundTripTerminal::Failed(error),
            None if successful => RoundTripTerminal::Completed,
            None => RoundTripTerminal::Stopped,
        }
    }

    fn finish(&self, session_id: Uuid, terminal: RoundTripTerminal) {
        {
            let mut session = lock_recovering(&self.state.session);
            session.set_cleanup_pending(session_id, false);
            match terminal {
                RoundTripTerminal::Completed => {
                    if session
                        .advance(
                            session_id,
                            RoundTripCheckpoint::Completed,
                            RoundTripLatency::default(),
                        )
                        .is_err()
                    {
                        session.fail(session_id, RoundTripErrorCode::RuntimeFailed);
                    }
                }
                RoundTripTerminal::Stopped => {
                    session.stop(session_id);
                }
                RoundTripTerminal::Failed(error) => {
                    session.fail(session_id, error);
                }
            }
        }
        self.publish();
    }

    fn publish(&self) {
        let preconditions = *lock_recovering(&self.state.preconditions);
        let status = lock_recovering(&self.state.session)
            .status(self.state.store.snapshot().debug_text_enabled);
        self.state.store.set_self_test(RoundTripSelfTestState {
            availability: "available",
            preconditions,
            status,
        });
    }
}

pub struct RoundTripRuntimeHandle {
    sender: mpsc::Sender<RoundTripCommand>,
    actor: Mutex<OwnerThread>,
    starts_closed: AtomicBool,
}

enum OwnerThread {
    Running(thread::JoinHandle<()>),
    Joined,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundTripRequestError {
    Control(ControlFailure),
    Busy,
    CleanupPending,
    OwnerFailed,
}

type RoundTripResponse =
    std_mpsc::SyncSender<Result<RoundTripSelfTestState, RoundTripRequestError>>;

enum RoundTripCommand {
    Start {
        outer_deadline: Instant,
        start_deadline: Instant,
        response: RoundTripResponse,
    },
    Stop {
        outer_deadline: Instant,
        cleanup_deadline: Instant,
        shutdown: bool,
        response: RoundTripResponse,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OwnerMode {
    Open,
    CleanupOnly,
    Failed,
}

struct OwnedRoundTrip {
    session_id: Uuid,
    runtime: Box<dyn ActiveRoundTripRuntime>,
    _lease: AudioOperationLease,
}

struct RoundTripApplication {
    runner: Arc<dyn RoundTripRunner>,
    facts: Arc<dyn RuntimeFactsSource>,
    gate: AudioOperationGate,
    progress: RoundTripProgress,
    active: Option<OwnedRoundTrip>,
    mode: OwnerMode,
}

type OwnerTask = Box<dyn FnOnce() + Send + 'static>;

impl RoundTripRuntimeHandle {
    pub fn try_new(
        store: RuntimeStore,
        runner: Arc<dyn RoundTripRunner>,
        gate: AudioOperationGate,
        facts: Arc<dyn RuntimeFactsSource>,
    ) -> Result<Self, RoundTripOwnerStartError> {
        Self::try_new_with(
            store,
            runner,
            gate,
            facts,
            build_owner_runtime,
            spawn_owner_thread,
        )
    }

    fn try_new_with<Build, Spawn>(
        store: RuntimeStore,
        runner: Arc<dyn RoundTripRunner>,
        gate: AudioOperationGate,
        facts: Arc<dyn RuntimeFactsSource>,
        build_runtime: Build,
        spawn_thread: Spawn,
    ) -> Result<Self, RoundTripOwnerStartError>
    where
        Build: FnOnce() -> std::io::Result<tokio::runtime::Runtime> + Send + 'static,
        Spawn: FnOnce(OwnerTask) -> std::io::Result<thread::JoinHandle<()>>,
    {
        let (sender, receiver) = mpsc::channel(1);
        let (completion, completed) = watch::channel(None);
        let progress = RoundTripProgress {
            state: Arc::new(ProgressState {
                session: Mutex::new(RoundTripSelfTest::default()),
                preconditions: Mutex::new(None),
                store,
                completion,
            }),
        };
        let owner = RoundTripApplication {
            runner,
            facts,
            gate,
            progress: progress.clone(),
            active: None,
            mode: OwnerMode::Open,
        };
        let (ready, readiness) = std_mpsc::sync_channel(1);
        let task: OwnerTask = Box::new(move || match build_runtime() {
            Ok(runtime) => runtime.block_on(owner.run(receiver, completed, ready)),
            Err(_) => {
                let _ = ready.send(Err(RoundTripOwnerStartError::RuntimeBuild));
            }
        });
        let actor = spawn_thread(task).map_err(|_| RoundTripOwnerStartError::ThreadSpawn)?;
        match readiness.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return match actor.join() {
                    Ok(()) => Err(error),
                    Err(_) => Err(RoundTripOwnerStartError::OwnerFailed),
                };
            }
            Err(_) => {
                let _ = actor.join();
                return Err(RoundTripOwnerStartError::OwnerFailed);
            }
        }
        progress.publish();
        Ok(Self {
            sender,
            actor: Mutex::new(OwnerThread::Running(actor)),
            starts_closed: AtomicBool::new(false),
        })
    }

    pub fn shutdown(&self) -> Result<(), RoundTripOwnerShutdownError> {
        self.starts_closed.store(true, Ordering::Release);
        let admitted = Instant::now();
        let deadline = admitted + Duration::from_secs(10);
        let mut actor = lock_recovering(&self.actor);
        match &*actor {
            OwnerThread::Joined => return Ok(()),
            OwnerThread::Failed => return Err(RoundTripOwnerShutdownError::OwnerFailed),
            OwnerThread::Running(_) => {}
        }
        if matches!(&*actor, OwnerThread::Running(thread) if !thread.is_finished()) {
            match self.request(
                false,
                true,
                deadline,
                admitted + crate::RUNTIME_CLEANUP_BUDGET,
            ) {
                Ok(_) => {}
                Err(RoundTripRequestError::OwnerFailed) => {
                    return Err(RoundTripOwnerShutdownError::OwnerFailed);
                }
                Err(_) => return Err(RoundTripOwnerShutdownError::CleanupPending),
            }
        }
        while matches!(&*actor, OwnerThread::Running(thread) if !thread.is_finished()) {
            if Instant::now() >= deadline {
                return Err(RoundTripOwnerShutdownError::CleanupPending);
            }
            thread::sleep(Duration::from_millis(1));
        }
        if let OwnerThread::Running(thread) = std::mem::replace(&mut *actor, OwnerThread::Failed) {
            thread
                .join()
                .map_err(|_| RoundTripOwnerShutdownError::OwnerFailed)?;
            *actor = OwnerThread::Joined;
        }
        Ok(())
    }

    fn request(
        &self,
        start: bool,
        shutdown: bool,
        deadline: Instant,
        transaction_deadline: Instant,
    ) -> Result<RoundTripSelfTestState, RoundTripRequestError> {
        if start && self.starts_closed.load(Ordering::Acquire) {
            return Err(RoundTripRequestError::OwnerFailed);
        }
        let (response, receiver) = std_mpsc::sync_channel(1);
        let command = if start {
            RoundTripCommand::Start {
                outer_deadline: deadline,
                start_deadline: transaction_deadline,
                response,
            }
        } else {
            RoundTripCommand::Stop {
                outer_deadline: deadline,
                cleanup_deadline: transaction_deadline,
                shutdown,
                response,
            }
        };
        match self.sender.try_send(command) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(RoundTripRequestError::Busy);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(RoundTripRequestError::OwnerFailed);
            }
        }
        match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(result) => result,
            Err(std_mpsc::RecvTimeoutError::Timeout) => Err(RoundTripRequestError::CleanupPending),
            Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                Err(RoundTripRequestError::OwnerFailed)
            }
        }
    }
}

impl RoundTripApplication {
    fn state(&self) -> RoundTripSelfTestState {
        let preconditions = *lock_recovering(&self.progress.state.preconditions);
        let status = lock_recovering(&self.progress.state.session)
            .status(self.progress.state.store.snapshot().debug_text_enabled);
        RoundTripSelfTestState {
            availability: "available",
            preconditions,
            status,
        }
    }

    fn protected_start(
        &mut self,
        deadline: Instant,
    ) -> Result<RoundTripSelfTestState, RoundTripRequestError> {
        if self.mode != OwnerMode::Open {
            return Err(RoundTripRequestError::OwnerFailed);
        }
        match catch_unwind(AssertUnwindSafe(|| self.start(deadline))) {
            Ok(result) => result.map_err(RoundTripRequestError::Control),
            Err(_) => Err(self.record_panic()),
        }
    }

    fn protected_stop(
        &mut self,
        deadline: Instant,
        cleanup_deadline: Instant,
        shutdown: bool,
    ) -> Result<RoundTripSelfTestState, RoundTripRequestError> {
        if self.mode == OwnerMode::Failed {
            return Err(RoundTripRequestError::OwnerFailed);
        }
        if shutdown {
            self.mode = OwnerMode::CleanupOnly;
        }
        match catch_unwind(AssertUnwindSafe(|| self.stop(deadline, cleanup_deadline))) {
            Ok(result) => result.map_err(RoundTripRequestError::Control),
            Err(_) => Err(self.record_panic()),
        }
    }

    fn record_panic(&mut self) -> RoundTripRequestError {
        if let Some(session_id) = self.active.as_ref().map(|active| active.session_id) {
            self.mode = OwnerMode::CleanupOnly;
            self.progress
                .fail(session_id, RoundTripErrorCode::RuntimeFailed);
            self.progress.set_cleanup_pending(session_id, true);
            RoundTripRequestError::CleanupPending
        } else {
            self.mode = OwnerMode::Failed;
            let session_id = lock_recovering(&self.progress.state.session)
                .status(false)
                .session_id;
            if let Some(session_id) = session_id {
                self.progress
                    .fail(session_id, RoundTripErrorCode::RuntimeFailed);
            }
            RoundTripRequestError::OwnerFailed
        }
    }

    fn start(&mut self, deadline: Instant) -> Result<RoundTripSelfTestState, ControlFailure> {
        if Instant::now() >= deadline {
            return Err(control_failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "self_test_start_failed",
            ));
        }
        if self.active.is_some() {
            return Err(control_failure(
                StatusCode::CONFLICT,
                "self_test_already_running",
            ));
        }
        let facts = self
            .facts
            .inspect(deadline)
            .map_err(FactsError::round_trip_failure)?;
        if Instant::now() >= deadline {
            return Err(FactsError::Expired.round_trip_failure());
        }
        let admitted = crate::acoustic_admission::admit_round_trip(
            self.progress.state.store.snapshot(),
            facts,
        )?;
        if Instant::now() >= deadline {
            return Err(FactsError::Expired.round_trip_failure());
        }
        let preconditions = round_trip_preconditions(admitted.snapshot());
        let session_id = Uuid::new_v4();
        let lease = self
            .gate
            .acquire_human_round_trip(session_id)
            .map_err(|_| control_failure(StatusCode::CONFLICT, "audio_operation_busy"))?;
        lock_recovering(&self.progress.state.session)
            .start_with_id(preconditions, monotonic_ms(), session_id)
            .map_err(map_precondition_error)?;
        *lock_recovering(&self.progress.state.preconditions) = Some(preconditions);
        let runtime = self
            .runner
            .start(admitted, session_id, self.progress.clone(), deadline)
            .map_err(|_| {
                self.progress
                    .fail(session_id, RoundTripErrorCode::RuntimeFailed);
                control_failure(StatusCode::SERVICE_UNAVAILABLE, "self_test_start_failed")
            })?;
        self.active = Some(OwnedRoundTrip {
            session_id,
            runtime,
            _lease: lease,
        });
        let state = self.state();
        self.progress.state.store.set_self_test(state.clone());
        Ok(state)
    }

    fn stop(
        &mut self,
        deadline: Instant,
        cleanup_deadline: Instant,
    ) -> Result<RoundTripSelfTestState, ControlFailure> {
        let Some(active) = self.active.as_mut() else {
            return Ok(self.state());
        };
        let session_id = active.session_id;
        lock_recovering(&self.progress.state.session).begin_cleanup(session_id);
        self.progress.publish();
        let terminal = active
            .runtime
            .stop(deadline, cleanup_deadline)
            .map_err(|_| {
                self.progress.set_cleanup_pending(session_id, true);
                control_failure(StatusCode::INTERNAL_SERVER_ERROR, "self_test_stop_failed")
            })?;
        self.active = None;
        self.progress.finish(session_id, terminal);
        Ok(self.state())
    }

    async fn run(
        mut self,
        mut commands: mpsc::Receiver<RoundTripCommand>,
        mut completed: watch::Receiver<Option<Uuid>>,
        ready: std_mpsc::SyncSender<Result<(), RoundTripOwnerStartError>>,
    ) {
        if ready.send(Ok(())).is_err() {
            return;
        }
        loop {
            tokio::select! {
                command = commands.recv() => match command {
                    Some(RoundTripCommand::Start { outer_deadline, start_deadline, response }) => {
                        let result = self.protected_start(start_deadline.min(outer_deadline));
                        let _ = response.send(result);
                    }
                    Some(RoundTripCommand::Stop { outer_deadline: deadline, cleanup_deadline, shutdown, response }) => {
                        let result = self.protected_stop(
                            deadline.checked_sub(Duration::from_millis(100)).unwrap_or(deadline),
                            cleanup_deadline,
                            shutdown,
                        );
                        let close = shutdown && result.is_ok();
                        let _ = response.send(result);
                        if close { break; }
                    }
                    None => {
                        loop {
                            let admitted = Instant::now();
                            match self.protected_stop(
                                admitted + Duration::from_secs(10),
                                admitted + crate::RUNTIME_CLEANUP_BUDGET,
                                true,
                            ) {
                                Ok(_) | Err(RoundTripRequestError::OwnerFailed) => break,
                                Err(_) => tokio::time::sleep(Duration::from_secs(1)).await,
                            }
                        }
                        break;
                    }
                },
                changed = completed.changed() => {
                    if changed.is_err() { continue; }
                    let session_id = *completed.borrow_and_update();
                    if self.active.as_ref().is_some_and(|active| Some(active.session_id) == session_id) {
                        let admitted = Instant::now();
                        let _ = self.protected_stop(
                            admitted + Duration::from_secs(10),
                            admitted + crate::RUNTIME_CLEANUP_BUDGET,
                            false,
                        );
                    }
                }
            }
        }
    }
}

impl RoundTripController for RoundTripRuntimeHandle {
    fn start(&self) -> Result<RoundTripSelfTestState, ControlFailure> {
        let admitted = Instant::now();
        self.request(
            true,
            false,
            admitted + Duration::from_secs(10),
            admitted + crate::DIRECTION_CLEANUP_BUDGET,
        )
        .map_err(map_request_error)
    }
    fn stop(&self) -> Result<RoundTripSelfTestState, ControlFailure> {
        let admitted = Instant::now();
        self.request(
            false,
            false,
            admitted + Duration::from_secs(10),
            admitted + crate::RUNTIME_CLEANUP_BUDGET,
        )
        .map_err(map_request_error)
    }
}

fn build_owner_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
}

fn spawn_owner_thread(task: OwnerTask) -> std::io::Result<thread::JoinHandle<()>> {
    thread::Builder::new()
        .name("translator-round-trip-owner".into())
        .spawn(task)
}

const fn map_request_error(error: RoundTripRequestError) -> ControlFailure {
    match error {
        RoundTripRequestError::Control(failure) => failure,
        RoundTripRequestError::Busy => {
            control_failure(StatusCode::CONFLICT, "self_test_control_busy")
        }
        RoundTripRequestError::CleanupPending => {
            control_failure(StatusCode::SERVICE_UNAVAILABLE, "self_test_cleanup_pending")
        }
        RoundTripRequestError::OwnerFailed => {
            control_failure(StatusCode::INTERNAL_SERVER_ERROR, "self_test_owner_failed")
        }
    }
}

pub(crate) fn round_trip_preconditions(snapshot: &RuntimeSnapshot) -> RoundTripPreconditions {
    let headphones = snapshot.devices.as_ref().is_some_and(|devices| {
        devices.acoustic.mode == OutputMode::Headphones && devices.acoustic.full_duplex_allowed
    });
    let provider_ready = snapshot.provider_id == ProviderId::Local && !snapshot.translation_running;
    let virtual_graph_ready = snapshot
        .audio_graph
        .as_ref()
        .is_some_and(|graph| graph.health == GraphHealth::Ready);
    let incoming_route_idle = snapshot.routes.as_ref().is_some_and(|routes| {
        routes.active_route.is_none()
            && routes.conflicting_stream_ids.is_empty()
            && matches!(
                routes.resolution,
                RouteResolution::NoCandidate | RouteResolution::AwaitingSelection
            )
    });
    RoundTripPreconditions {
        headphones,
        outgoing_provider_ready: provider_ready,
        incoming_provider_ready: provider_ready,
        virtual_graph_ready,
        incoming_route_idle,
    }
}

pub(crate) fn map_precondition_error(error: RoundTripErrorCode) -> ControlFailure {
    let code = match error {
        RoundTripErrorCode::HeadphonesRequired => "self_test_headphones_required",
        RoundTripErrorCode::ProviderUnavailable => "self_test_provider_unavailable",
        RoundTripErrorCode::VirtualGraphUnavailable => "self_test_graph_unavailable",
        RoundTripErrorCode::IncomingRouteConflict => "self_test_route_conflict",
        RoundTripErrorCode::AlreadyRunning => "self_test_already_running",
        _ => "self_test_precondition_failed",
    };
    control_failure(StatusCode::CONFLICT, code)
}

const fn control_failure(status: StatusCode, code: &'static str) -> ControlFailure {
    ControlFailure { status, code }
}

fn lock_recovering<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn monotonic_ms() -> u64 {
    let time = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    u64::try_from(time.tv_sec)
        .unwrap_or(0)
        .saturating_mul(1_000)
        .saturating_add(u64::try_from(time.tv_nsec).unwrap_or(0) / 1_000_000)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    struct UnusedFacts;
    impl RuntimeFactsSource for UnusedFacts {
        fn inspect(&self, _: Instant) -> Result<crate::RuntimeFacts, FactsError> {
            panic!("cleanup/owner-construction fixture must not discover devices")
        }
    }

    struct AdmissionFacts {
        late: bool,
    }
    impl RuntimeFactsSource for AdmissionFacts {
        fn inspect(&self, deadline: Instant) -> Result<crate::RuntimeFacts, FactsError> {
            let mut facts = crate::acoustic_admission::tests::ready_facts();
            if self.late {
                while Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(1));
                }
                facts.devices.output_mode = OutputMode::UnknownUnsafe;
            }
            Ok(facts)
        }
    }

    fn admission_owner(late: bool, previous_active: bool) -> RoundTripApplication {
        let mut session = RoundTripSelfTest::default();
        let id = session
            .start(
                RoundTripPreconditions {
                    headphones: true,
                    outgoing_provider_ready: true,
                    incoming_provider_ready: true,
                    virtual_graph_ready: true,
                    incoming_route_idle: true,
                },
                0,
            )
            .unwrap();
        if !previous_active {
            assert!(session.stop(id));
        }
        let (completion, _) = watch::channel(None);
        let progress = RoundTripProgress {
            state: Arc::new(ProgressState {
                session: Mutex::new(session),
                preconditions: Mutex::new(None),
                store: RuntimeStore::default(),
                completion,
            }),
        };
        progress.publish();
        RoundTripApplication {
            runner: unused_runner(),
            facts: Arc::new(AdmissionFacts { late }),
            gate: AudioOperationGate::new(),
            progress,
            active: None,
            mode: OwnerMode::Open,
        }
    }

    #[test]
    fn safe1_round_trip_gate_rejection_preserves_prior_session_and_events() {
        for stopping in [false, true] {
            let mut owner = admission_owner(false, false);
            let lease = if stopping {
                owner.gate.begin_stopping();
                None
            } else {
                Some(owner.gate.acquire_production().unwrap())
            };
            let before = serde_json::to_value(owner.state()).unwrap();
            let projection = serde_json::to_value(owner.progress.state.store.snapshot()).unwrap();
            let gate = owner.gate.state();
            let mut events = owner.progress.state.store.subscribe().unwrap();
            let result = owner.start(Instant::now() + Duration::from_secs(1));
            let after = serde_json::to_value(owner.state()).unwrap();
            let projected = serde_json::to_value(owner.progress.state.store.snapshot()).unwrap();
            let after_gate = owner.gate.state();
            let event = events.try_recv();
            drop(lease);
            let failure = result.err().unwrap();
            assert_eq!(
                (failure.status, failure.code),
                (StatusCode::CONFLICT, "audio_operation_busy")
            );
            assert_eq!(after, before);
            assert_eq!(projected, projection);
            assert_eq!(after_gate, gate);
            assert!(matches!(
                event,
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
            ));
        }
    }

    #[test]
    fn safe1_round_trip_late_unsafe_facts_expire_before_policy() {
        let mut owner = admission_owner(true, false);
        let before = serde_json::to_value(owner.state()).unwrap();
        let mut events = owner.progress.state.store.subscribe().unwrap();
        let result = owner.start(Instant::now() + Duration::from_millis(5));
        let failure = result.err().unwrap();
        assert_eq!(
            (failure.status, failure.code),
            (StatusCode::SERVICE_UNAVAILABLE, "audio_facts_expired")
        );
        assert_eq!(serde_json::to_value(owner.state()).unwrap(), before);
        assert_eq!(owner.gate.state(), crate::AudioOperationState::Idle);
        assert!(matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn safe1_round_trip_session_rejection_releases_gate_without_projection_change() {
        let mut owner = admission_owner(false, true);
        let before = serde_json::to_value(owner.state()).unwrap();
        let mut events = owner.progress.state.store.subscribe().unwrap();
        let result = owner.start(Instant::now() + Duration::from_secs(1));
        let failure = result.err().unwrap();
        assert_eq!(
            (failure.status, failure.code),
            (StatusCode::CONFLICT, "self_test_already_running")
        );
        assert_eq!(serde_json::to_value(owner.state()).unwrap(), before);
        assert_eq!(owner.gate.state(), crate::AudioOperationState::Idle);
        assert!(matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
    }

    pub(crate) fn active_observer_progress() -> (Uuid, RoundTripProgress) {
        let mut session = RoundTripSelfTest::default();
        let session_id = session
            .start(
                RoundTripPreconditions {
                    headphones: true,
                    outgoing_provider_ready: true,
                    incoming_provider_ready: true,
                    virtual_graph_ready: true,
                    incoming_route_idle: true,
                },
                0,
            )
            .unwrap();
        let (completion, _) = watch::channel(None);
        (
            session_id,
            RoundTripProgress {
                state: Arc::new(ProgressState {
                    session: Mutex::new(session),
                    preconditions: Mutex::new(None),
                    store: RuntimeStore::default(),
                    completion,
                }),
            },
        )
    }

    fn composed_join_owner(
        runtime: Box<dyn ActiveRoundTripRuntime>,
    ) -> (RoundTripApplication, AudioOperationGate, Uuid) {
        let gate = AudioOperationGate::new();
        let mut session = RoundTripSelfTest::default();
        let session_id = session
            .start(
                RoundTripPreconditions {
                    headphones: true,
                    outgoing_provider_ready: true,
                    incoming_provider_ready: true,
                    virtual_graph_ready: true,
                    incoming_route_idle: true,
                },
                0,
            )
            .unwrap();
        let lease = gate.acquire_human_round_trip(session_id).unwrap();
        let (completion, _) = watch::channel(None);
        let progress = RoundTripProgress {
            state: Arc::new(ProgressState {
                session: Mutex::new(session),
                preconditions: Mutex::new(None),
                store: RuntimeStore::default(),
                completion,
            }),
        };
        let owner = RoundTripApplication {
            runner: unused_runner(),
            facts: Arc::new(UnusedFacts),
            gate: gate.clone(),
            progress: progress.clone(),
            mode: OwnerMode::Open,
            active: Some(OwnedRoundTrip {
                session_id,
                runtime,
                _lease: lease,
            }),
        };
        (owner, gate, session_id)
    }

    #[test]
    fn payload_drop_panic_keeps_composed_round_trip_owner_and_lease() {
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (mut owner, gate, session_id) = composed_join_owner(
            crate::round_trip_process::tests::payload_drop_panicked_inner_runtime(drops.clone()),
        );
        let admitted = Instant::now();
        let first = owner.protected_stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
            false,
        );
        let first_retained = owner.active.is_some();
        let first_gate = gate.state();
        let second = owner.protected_stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
            false,
        );
        let expired = owner.protected_stop(admitted, admitted, false);
        let retained_session = owner.active.as_ref().map(|active| active.session_id);
        let final_gate = gate.state();
        let status = owner.state().status;
        drop(owner);
        assert!(matches!(first, Err(RoundTripRequestError::CleanupPending)));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(first_retained);
        assert_eq!(
            first_gate,
            crate::AudioOperationState::HumanRoundTrip { session_id }
        );
        assert!(
            second.is_err(),
            "repeated Stop must not promote saved Completed"
        );
        assert!(expired.is_err());
        assert_eq!(retained_session, Some(session_id));
        assert_eq!(
            final_gate,
            crate::AudioOperationState::HumanRoundTrip { session_id }
        );
        assert_ne!(status.checkpoint, Some(RoundTripCheckpoint::Completed));
        assert!(status.cleanup_pending);
    }

    fn assert_composed_pre_receipt_panic(payload_drop: bool) {
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (runtime, joined) = crate::round_trip_process::tests::pre_receipt_inner_runtime(
            payload_drop.then(|| drops.clone()),
        );
        let (mut owner, gate, session_id) = composed_join_owner(runtime);
        let admitted = Instant::now();
        let first = owner.protected_stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
            false,
        );
        let second = owner.protected_stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
            false,
        );
        let expired = owner.protected_stop(admitted, admitted, false);
        let retained = owner.active.as_ref().map(|active| active.session_id);
        let gate_state = gate.state();
        let status = owner.state().status;
        let drops_before_drain = drops.load(Ordering::SeqCst);
        drop(owner);
        assert!(
            joined.load(Ordering::SeqCst),
            "composed Stop must consume the failed worker before fixture cleanup"
        );
        if payload_drop {
            assert!(matches!(first, Err(RoundTripRequestError::CleanupPending)));
            assert_eq!(drops_before_drain, 1);
        } else {
            assert!(matches!(first, Err(RoundTripRequestError::Control(_))));
        }
        assert!(second.is_err() && expired.is_err());
        assert_eq!(retained, Some(session_id));
        assert_eq!(
            gate_state,
            crate::AudioOperationState::HumanRoundTrip { session_id }
        );
        assert_ne!(status.checkpoint, Some(RoundTripCheckpoint::Completed));
        assert!(status.cleanup_pending);
    }

    #[test]
    fn pre_receipt_panic_keeps_composed_owner_and_lease() {
        assert_composed_pre_receipt_panic(false);
    }

    #[test]
    fn pre_receipt_payload_drop_panic_keeps_composed_owner_and_lease() {
        assert_composed_pre_receipt_panic(true);
    }

    #[test]
    fn confirmed_inner_join_failure_keeps_composed_round_trip_owner_and_lease() {
        let (mut owner, gate, session_id) =
            composed_join_owner(crate::round_trip_process::tests::panicked_inner_runtime());
        let admitted = Instant::now();
        let first = owner.protected_stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
            false,
        );
        let second = owner.protected_stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
            false,
        );
        let retained = owner.active.is_some();
        let gate_after = gate.state();
        let status = owner.state().status;
        drop(owner);
        assert!(first.is_err() && second.is_err());
        assert!(retained);
        assert_eq!(
            gate_after,
            crate::AudioOperationState::HumanRoundTrip { session_id }
        );
        assert_ne!(status.checkpoint, Some(RoundTripCheckpoint::Completed));
        assert!(status.cleanup_pending);
    }

    #[test]
    fn public_admission_separates_outer_wait_from_native_transaction_deadlines() {
        for start in [true, false] {
            let (sender, mut receiver) = mpsc::channel(1);
            let (observed, observation) = std_mpsc::sync_channel(1);
            let (release, released) = std_mpsc::sync_channel(1);
            let thread = thread::spawn(move || {
                released.recv().unwrap();
                let (outer, transaction, response) = match receiver.blocking_recv().unwrap() {
                    RoundTripCommand::Start {
                        outer_deadline,
                        start_deadline,
                        response,
                    } => (outer_deadline, start_deadline, response),
                    RoundTripCommand::Stop {
                        outer_deadline,
                        cleanup_deadline,
                        response,
                        ..
                    } => (outer_deadline, cleanup_deadline, response),
                };
                observed.send((outer, transaction, Instant::now())).unwrap();
                response
                    .send(Ok(RoundTripSelfTestState::default()))
                    .unwrap();
            });
            let owner = Arc::new(RoundTripRuntimeHandle {
                sender,
                actor: Mutex::new(OwnerThread::Running(thread)),
                starts_closed: AtomicBool::new(false),
            });
            let before = Instant::now();
            let caller = owner.clone();
            let requested =
                thread::spawn(move || if start { caller.start() } else { caller.stop() });
            while owner.sender.capacity() != 0 && before.elapsed() < Duration::from_secs(1) {
                thread::yield_now();
            }
            let queued = owner.sender.capacity() == 0;
            release.send(()).unwrap();
            let response = requested.join().unwrap();
            let after = Instant::now();
            let (outer, transaction, dequeued) =
                observation.recv_timeout(Duration::from_secs(1)).unwrap();
            wait_until_owner_finishes(&owner);
            owner.shutdown().unwrap();
            assert!(response.is_ok());
            assert!(queued);
            let budget = if start {
                Duration::from_secs(4)
            } else {
                Duration::from_secs(8)
            };
            assert_eq!(
                outer.duration_since(transaction),
                Duration::from_secs(10) - budget
            );
            assert!(outer >= before + Duration::from_secs(10));
            assert!(outer <= after + Duration::from_secs(10));
            assert!(
                transaction < dequeued + budget,
                "queue delay must consume the admitted transaction budget"
            );
        }
    }

    #[test]
    fn expired_queued_start_does_not_publish_or_acquire_or_call_runner() {
        struct CountingRunner(Arc<std::sync::atomic::AtomicUsize>);
        impl RoundTripRunner for CountingRunner {
            fn start(
                &self,
                _: AdmittedDuplex,
                _: Uuid,
                _: RoundTripProgress,
                _: Instant,
            ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Err(RoundTripRuntimeError::StartFailed)
            }
        }
        let (completion, _) = watch::channel(None);
        let gate = AudioOperationGate::new();
        let store = RuntimeStore::default();
        store.set_audio_graph(translator_audio::AudioGraphState {
            health: GraphHealth::Ready,
            endpoints: Vec::new(),
            owned_module_ids: Vec::new(),
            safe_error: None,
        });
        let selection = translator_audio::DeviceSelectionState {
            health: translator_audio::DeviceHealth::Available,
            pinned_name: None,
            current_default: None,
            pending_default: None,
            selected: None,
        };
        store.set_devices(crate::DeviceState {
            source: selection.clone(),
            sink: selection,
            acoustic: crate::AcousticSafety {
                mode: OutputMode::Headphones,
                aec_capability: translator_audio::AecCapability::Unavailable,
                full_duplex_allowed: true,
                warning: None,
            },
        });
        store.set_routes(translator_audio::RoutingState {
            candidates: Vec::new(),
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: RouteResolution::NoCandidate,
        });
        let preconditions = round_trip_preconditions(&store.snapshot());
        assert!(
            preconditions.headphones
                && preconditions.outgoing_provider_ready
                && preconditions.incoming_provider_ready
                && preconditions.virtual_graph_ready
                && preconditions.incoming_route_idle
        );
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let progress = RoundTripProgress {
            state: Arc::new(ProgressState {
                session: Mutex::new(RoundTripSelfTest::default()),
                preconditions: Mutex::new(None),
                store,
                completion,
            }),
        };
        let before = progress.state.store.snapshot().self_test;
        let mut owner = RoundTripApplication {
            runner: Arc::new(CountingRunner(calls.clone())),
            facts: Arc::new(UnusedFacts),
            gate: gate.clone(),
            progress: progress.clone(),
            active: None,
            mode: OwnerMode::Open,
        };
        let result = owner.protected_start(Instant::now() - Duration::from_millis(1));
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(lock_recovering(&progress.state.preconditions).is_none());
        assert_eq!(
            serde_json::to_value(progress.state.store.snapshot().self_test).unwrap(),
            serde_json::to_value(before).unwrap()
        );
        assert_eq!(gate.state(), crate::AudioOperationState::Idle);
        assert!(owner.active.is_none());
    }

    struct UnusedRunner;

    impl RoundTripRunner for UnusedRunner {
        fn start(
            &self,
            _admitted: AdmittedDuplex,
            _session_id: Uuid,
            _progress: RoundTripProgress,
            _start_deadline: Instant,
        ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError> {
            panic!("unused runner")
        }
    }

    fn unused_runner() -> Arc<dyn RoundTripRunner> {
        Arc::new(UnusedRunner)
    }

    fn finished_thread() -> thread::JoinHandle<()> {
        let actor = thread::spawn(|| {});
        let deadline = Instant::now() + Duration::from_secs(1);
        while !actor.is_finished() && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(actor.is_finished());
        actor
    }

    fn wait_until_owner_finishes(owner: &RoundTripRuntimeHandle) {
        let deadline = Instant::now() + Duration::from_secs(1);
        while !matches!(
            &*lock_recovering(&owner.actor),
            OwnerThread::Running(actor) if actor.is_finished()
        ) && Instant::now() < deadline
        {
            thread::yield_now();
        }
    }

    #[test]
    fn availability_waits_for_owner_ready_ack() {
        let store = RuntimeStore::default();
        let constructor_store = store.clone();
        let gate = AudioOperationGate::new();
        let observed_gate = gate.clone();
        let (building, building_receiver) = std_mpsc::channel();
        let (release, release_receiver) = std_mpsc::channel();
        let constructor = thread::spawn(move || {
            RoundTripRuntimeHandle::try_new_with(
                constructor_store,
                unused_runner(),
                gate,
                Arc::new(UnusedFacts),
                move || {
                    let _ = building.send(());
                    let _ = release_receiver.recv();
                    build_owner_runtime()
                },
                spawn_owner_thread,
            )
        });

        let build_started = building_receiver
            .recv_timeout(Duration::from_secs(1))
            .is_ok();
        let availability_before_ready = store.snapshot().self_test.availability;
        let gate_before_ready = observed_gate.state();
        let released = release.send(()).is_ok();
        let owner = constructor.join().ok().and_then(Result::ok);
        let availability_after_ready = store.snapshot().self_test.availability;
        let shutdown = owner.as_ref().map(RoundTripRuntimeHandle::shutdown);

        assert!(build_started);
        assert!(released);
        assert_ne!(availability_before_ready, "available");
        assert_eq!(gate_before_ready, crate::AudioOperationState::Idle);
        assert_eq!(availability_after_ready, "available");
        assert_eq!(shutdown, Some(Ok(())));
    }

    #[test]
    fn runtime_build_and_thread_spawn_failures_do_not_publish_or_acquire() {
        let build_store = RuntimeStore::default();
        let build_gate = AudioOperationGate::new();
        let build_finished = Arc::new(AtomicBool::new(false));
        let observed_build = Arc::clone(&build_finished);
        let build_result = RoundTripRuntimeHandle::try_new_with(
            build_store.clone(),
            unused_runner(),
            build_gate.clone(),
            Arc::new(UnusedFacts),
            move || {
                observed_build.store(true, Ordering::SeqCst);
                Err(std::io::Error::other("injected runtime build failure"))
            },
            spawn_owner_thread,
        );

        let spawn_store = RuntimeStore::default();
        let spawn_gate = AudioOperationGate::new();
        let build_called = Arc::new(AtomicBool::new(false));
        let observed_call = Arc::clone(&build_called);
        let spawn_result = RoundTripRuntimeHandle::try_new_with(
            spawn_store.clone(),
            unused_runner(),
            spawn_gate.clone(),
            Arc::new(UnusedFacts),
            move || {
                observed_call.store(true, Ordering::SeqCst);
                build_owner_runtime()
            },
            |_task| Err(std::io::Error::other("injected owner thread spawn failure")),
        );

        assert!(matches!(
            build_result,
            Err(RoundTripOwnerStartError::RuntimeBuild)
        ));
        assert!(build_finished.load(Ordering::SeqCst));
        assert_ne!(build_store.snapshot().self_test.availability, "available");
        assert_eq!(build_gate.state(), crate::AudioOperationState::Idle);
        assert!(matches!(
            spawn_result,
            Err(RoundTripOwnerStartError::ThreadSpawn)
        ));
        assert!(!build_called.load(Ordering::SeqCst));
        assert_ne!(spawn_store.snapshot().self_test.availability, "available");
        assert_eq!(spawn_gate.state(), crate::AudioOperationState::Idle);
    }

    #[test]
    fn mailbox_full_and_closed_are_distinct() {
        let (full_sender, full_receiver) = mpsc::channel(1);
        let (queued_response, _queued_receiver) = std_mpsc::sync_channel(1);
        full_sender
            .try_send(RoundTripCommand::Start {
                outer_deadline: Instant::now(),
                start_deadline: Instant::now(),
                response: queued_response,
            })
            .unwrap();
        let full_owner = RoundTripRuntimeHandle {
            sender: full_sender,
            actor: Mutex::new(OwnerThread::Running(finished_thread())),
            starts_closed: AtomicBool::new(false),
        };
        let full = full_owner.request(
            true,
            false,
            Instant::now() + Duration::from_secs(1),
            Instant::now(),
        );
        drop(full_receiver);
        let full_shutdown = full_owner.shutdown();

        let (closed_sender, closed_receiver) = mpsc::channel(1);
        drop(closed_receiver);
        let closed_owner = RoundTripRuntimeHandle {
            sender: closed_sender,
            actor: Mutex::new(OwnerThread::Running(finished_thread())),
            starts_closed: AtomicBool::new(false),
        };
        let closed = closed_owner.request(
            true,
            false,
            Instant::now() + Duration::from_secs(1),
            Instant::now(),
        );
        let closed_shutdown = closed_owner.shutdown();

        assert!(matches!(full, Err(RoundTripRequestError::Busy)));
        assert_eq!(full_shutdown, Ok(()));
        assert!(matches!(closed, Err(RoundTripRequestError::OwnerFailed)));
        assert_eq!(closed_shutdown, Ok(()));
    }

    #[test]
    fn shutdown_maps_busy_admission_to_cleanup_pending() {
        let (sender, mut receiver) = mpsc::channel(1);
        let (queued_response, _queued_receiver) = std_mpsc::sync_channel(1);
        sender
            .try_send(RoundTripCommand::Start {
                outer_deadline: Instant::now(),
                start_deadline: Instant::now(),
                response: queued_response,
            })
            .unwrap();
        let (release, release_receiver) = std_mpsc::channel();
        let (drained, drained_receiver) = std_mpsc::channel();
        let actor = thread::spawn(move || {
            let _ = release_receiver.recv();
            let mut first = true;
            while let Some(command) = receiver.blocking_recv() {
                let close = match command {
                    RoundTripCommand::Start { response, .. } => {
                        let _ = response.send(Ok(RoundTripSelfTestState::default()));
                        false
                    }
                    RoundTripCommand::Stop {
                        shutdown, response, ..
                    } => {
                        let _ = response.send(Ok(RoundTripSelfTestState::default()));
                        shutdown
                    }
                };
                if first {
                    let _ = drained.send(());
                    first = false;
                }
                if close {
                    break;
                }
            }
        });
        let owner = RoundTripRuntimeHandle {
            sender,
            actor: Mutex::new(OwnerThread::Running(actor)),
            starts_closed: AtomicBool::new(false),
        };

        let busy = owner.shutdown();
        let released = release.send(()).is_ok();
        let queue_drained = drained_receiver
            .recv_timeout(Duration::from_secs(1))
            .is_ok();
        let start_after_shutdown = RoundTripController::start(&owner);
        let joined = owner.shutdown();

        assert_eq!(busy, Err(RoundTripOwnerShutdownError::CleanupPending));
        assert!(released);
        assert!(queue_drained);
        assert_eq!(
            start_after_shutdown.unwrap_err().code,
            "self_test_owner_failed"
        );
        assert_eq!(joined, Ok(()));
    }

    #[test]
    fn response_timeout_and_disconnection_are_distinct() {
        let (timeout_sender, mut timeout_receiver) = mpsc::channel(1);
        let (admitted, admitted_receiver) = std_mpsc::channel();
        let (release, release_receiver) = std_mpsc::channel();
        let completed = Arc::new(AtomicBool::new(false));
        let observed_completion = Arc::clone(&completed);
        let timeout_actor = thread::spawn(move || {
            let command = timeout_receiver.blocking_recv().unwrap();
            let response = match command {
                RoundTripCommand::Start { response, .. } => response,
                RoundTripCommand::Stop { response, .. } => response,
            };
            admitted.send(()).unwrap();
            release_receiver.recv().unwrap();
            drop(response);
            observed_completion.store(true, Ordering::SeqCst);
        });
        let timeout_owner = Arc::new(RoundTripRuntimeHandle {
            sender: timeout_sender,
            actor: Mutex::new(OwnerThread::Running(timeout_actor)),
            starts_closed: AtomicBool::new(false),
        });
        let requested_owner = Arc::clone(&timeout_owner);
        let request = thread::spawn(move || {
            requested_owner.request(
                true,
                false,
                Instant::now() + Duration::from_millis(25),
                Instant::now(),
            )
        });
        let admitted = admitted_receiver
            .recv_timeout(Duration::from_secs(1))
            .is_ok();
        let timeout = request.join().ok();
        let released = release.send(()).is_ok();
        wait_until_owner_finishes(&timeout_owner);
        let timeout_shutdown = timeout_owner.shutdown();

        let (disconnected_sender, mut disconnected_receiver) = mpsc::channel(1);
        let disconnected_actor = thread::spawn(move || {
            let command = disconnected_receiver.blocking_recv().unwrap();
            match command {
                RoundTripCommand::Start { response, .. } => drop(response),
                RoundTripCommand::Stop { response, .. } => drop(response),
            }
        });
        let disconnected_owner = RoundTripRuntimeHandle {
            sender: disconnected_sender,
            actor: Mutex::new(OwnerThread::Running(disconnected_actor)),
            starts_closed: AtomicBool::new(false),
        };
        let disconnected = disconnected_owner.request(
            true,
            false,
            Instant::now() + Duration::from_secs(1),
            Instant::now(),
        );
        wait_until_owner_finishes(&disconnected_owner);
        let disconnected_shutdown = disconnected_owner.shutdown();

        assert!(admitted);
        assert!(released);
        assert!(matches!(
            timeout,
            Some(Err(RoundTripRequestError::CleanupPending))
        ));
        assert!(completed.load(Ordering::SeqCst));
        assert_eq!(timeout_shutdown, Ok(()));
        assert!(matches!(
            disconnected,
            Err(RoundTripRequestError::OwnerFailed)
        ));
        assert_eq!(disconnected_shutdown, Ok(()));
    }

    #[test]
    fn owner_join_panic_never_becomes_successful_shutdown() {
        let (sender, _receiver) = mpsc::channel(1);
        let actor = thread::spawn(|| panic!("injected owner panic"));
        let deadline = Instant::now() + Duration::from_secs(1);
        while !actor.is_finished() && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(actor.is_finished());
        let owner = RoundTripRuntimeHandle {
            sender,
            actor: Mutex::new(OwnerThread::Running(actor)),
            starts_closed: AtomicBool::new(false),
        };
        let first = owner.shutdown();
        let second = owner.shutdown();
        let cleanup_would_run = first.is_ok() || second.is_ok();
        assert_eq!(first, Err(RoundTripOwnerShutdownError::OwnerFailed));
        assert_eq!(second, Err(RoundTripOwnerShutdownError::OwnerFailed));
        assert!(!cleanup_would_run);
    }
}
