use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{Mutex as AsyncMutex, Notify};
use tokio::task::{JoinError, JoinHandle};
use tokio::time::Instant;
use translator_audio::AecValidationInput;
use uuid::Uuid;

use crate::{
    AecCalibrationChallenge, AecCalibrationCoordinator, AecCoordinatorError, AecProofBinding,
    AecProofStatus, AudioOperationAdmissionError, AudioOperationGate, AudioOperationLease,
};

pub const AEC_CALIBRATION_BUDGET: Duration = Duration::from_secs(180);

pub type AecCalibrationFuture = Pin<
    Box<dyn Future<Output = Result<AecCalibrationPublication, AecCalibrationEngineError>> + Send>,
>;
pub type AecCleanupFuture = Pin<Box<dyn Future<Output = bool> + Send>>;

pub trait AecCalibrationEngine: Send + Sync {
    fn inspect_binding(
        &self,
        deadline: Instant,
    ) -> Result<AecProofBinding, AecCalibrationEngineError>;
    fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture;
    fn cleanup(&self, deadline: Instant) -> AecCleanupFuture;
}

#[derive(Clone)]
pub struct AecCalibrationCancellation {
    cancelled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl AecCalibrationCancellation {
    fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            self.notify.notify_waiters();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let notified = self.notify.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

#[derive(Clone)]
pub struct AecCalibrationRequest {
    pub attempt_id: Uuid,
    pub challenge: AecCalibrationChallenge,
    pub deadline: Instant,
    pub cancellation: AecCalibrationCancellation,
}

pub struct AecCalibrationPublication {
    pub input: AecValidationInput,
    pub probe_teardown_confirmed: bool,
    pub graph_retained: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AecCalibrationEngineError {
    pub code: &'static str,
    pub cleanup_confirmed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum AecCalibrationControlStatus {
    Unavailable,
    Running {
        attempt_id: Uuid,
    },
    Succeeded {
        attempt_id: Uuid,
        proof: AecProofStatus,
    },
    Failed {
        attempt_id: Uuid,
        code: &'static str,
    },
    Cancelled {
        attempt_id: Uuid,
    },
    TimedOut {
        attempt_id: Uuid,
    },
    CleanupUncertain {
        attempt_id: Uuid,
    },
    ShuttingDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecCalibrationControlError {
    Busy,
    Stopping,
    Unavailable,
}

struct ActiveAttempt {
    attempt_id: Uuid,
    phase: ActiveAttemptPhase,
    cancellation: AecCalibrationCancellation,
    task: Arc<AsyncMutex<Option<JoinHandle<()>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveAttemptPhase {
    Inspecting,
    Calibrating,
}

enum CalibrationLaunch {
    Cancelled,
    Started(AecCalibrationFuture),
    Panicked,
}

struct ControlState {
    status: AecCalibrationControlStatus,
    active: Option<ActiveAttempt>,
    retained_custody: Option<RetainedCustody>,
    stopping: bool,
}

struct RetainedCustody {
    attempt_id: Uuid,
    lease: Option<AudioOperationLease>,
    cleanup_task: Arc<AsyncMutex<Option<JoinHandle<bool>>>>,
}

struct ControllerInner {
    coordinator: Arc<AecCalibrationCoordinator>,
    gate: AudioOperationGate,
    engine: Arc<dyn AecCalibrationEngine>,
    state: Mutex<ControlState>,
    lifecycle: AsyncMutex<()>,
    #[cfg(test)]
    preflight_terminal_barrier: Mutex<Option<Arc<PreflightTerminalBarrier>>>,
}

#[cfg(test)]
struct PreflightTerminalBarrier {
    entered: Notify,
    released: Notify,
}

#[derive(Clone)]
pub struct AecCalibrationController {
    inner: Arc<ControllerInner>,
}

impl AecCalibrationController {
    pub fn new(
        coordinator: Arc<AecCalibrationCoordinator>,
        gate: AudioOperationGate,
        engine: Arc<dyn AecCalibrationEngine>,
    ) -> Self {
        Self {
            inner: Arc::new(ControllerInner {
                coordinator,
                gate,
                engine,
                state: Mutex::new(ControlState {
                    status: AecCalibrationControlStatus::Unavailable,
                    active: None,
                    retained_custody: None,
                    stopping: false,
                }),
                lifecycle: AsyncMutex::new(()),
                #[cfg(test)]
                preflight_terminal_barrier: Mutex::new(None),
            }),
        }
    }

    pub async fn start(&self) -> Result<AecCalibrationControlStatus, AecCalibrationControlError> {
        let _lifecycle = self.inner.lifecycle.lock().await;
        self.reap_finished().await?;
        let attempt_id = Uuid::new_v4();
        let interval_id = Uuid::new_v4();
        let lease = self.acquire_attempt_lease(attempt_id).await?;
        let cancellation = AecCalibrationCancellation::new();
        let deadline = Instant::now() + AEC_CALIBRATION_BUDGET;
        let inner = Arc::clone(&self.inner);
        let task_cancellation = cancellation.clone();
        let task_slot = Arc::new(AsyncMutex::new(None));
        let status = AecCalibrationControlStatus::Running { attempt_id };
        {
            let mut state = lock_recovering(&self.inner.state);
            state.status = status.clone();
            state.active = Some(ActiveAttempt {
                attempt_id,
                phase: ActiveAttemptPhase::Inspecting,
                cancellation,
                task: Arc::clone(&task_slot),
            });
        }
        let task = tokio::spawn(async move {
            run_owned_attempt(
                inner,
                attempt_id,
                interval_id,
                lease,
                deadline,
                task_cancellation,
            )
            .await;
        });
        *task_slot
            .try_lock()
            .expect("new attempt task slot cannot be contended") = Some(task);
        Ok(status)
    }

    pub async fn cancel(&self, attempt_id: Uuid) -> AecCalibrationControlStatus {
        let mut state = lock_recovering(&self.inner.state);
        if let Some(active) = state
            .active
            .as_ref()
            .filter(|active| active.attempt_id == attempt_id)
        {
            active.cancellation.cancel();
            if matches!(state.status, AecCalibrationControlStatus::Succeeded { .. }) {
                self.inner.coordinator.revoke();
                state.status = AecCalibrationControlStatus::Cancelled { attempt_id };
            }
        }
        state.status.clone()
    }

    pub fn status(&self) -> AecCalibrationControlStatus {
        lock_recovering(&self.inner.state).status.clone()
    }

    pub async fn shutdown(&self) -> Result<(), AecCalibrationControlError> {
        let _lifecycle = self.inner.lifecycle.lock().await;
        self.inner.gate.begin_stopping();
        let active = {
            let mut state = lock_recovering(&self.inner.state);
            state.stopping = true;
            state.status = AecCalibrationControlStatus::ShuttingDown;
            state.active.as_ref().map(|active| {
                active.cancellation.cancel();
                (active.attempt_id, Arc::clone(&active.task))
            })
        };
        self.inner.coordinator.revoke();
        if let Some((attempt_id, task)) = active {
            join_active_task(&task).await?;
            let mut state = lock_recovering(&self.inner.state);
            if state
                .active
                .as_ref()
                .is_some_and(|active| active.attempt_id == attempt_id)
            {
                state.active = None;
            }
        }
        if self.retry_retained_cleanup().await {
            Ok(())
        } else {
            Err(AecCalibrationControlError::Unavailable)
        }
    }

    async fn reap_finished(&self) -> Result<(), AecCalibrationControlError> {
        let active = {
            let state = lock_recovering(&self.inner.state);
            state
                .active
                .as_ref()
                .map(|active| (active.attempt_id, Arc::clone(&active.task)))
        };
        let Some((attempt_id, task)) = active else {
            return Ok(());
        };
        {
            let task = task.lock().await;
            if task.as_ref().is_some_and(|task| !task.is_finished()) {
                return Ok(());
            }
        }
        join_active_task(&task).await?;
        let mut state = lock_recovering(&self.inner.state);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.attempt_id == attempt_id)
        {
            state.active = None;
        }
        Ok(())
    }

    async fn acquire_attempt_lease(
        &self,
        attempt_id: Uuid,
    ) -> Result<AudioOperationLease, AecCalibrationControlError> {
        let retained = {
            let mut state = lock_recovering(&self.inner.state);
            if state.stopping {
                return Err(AecCalibrationControlError::Stopping);
            }
            if state.active.is_some() {
                return Err(AecCalibrationControlError::Busy);
            }
            let gate_state = self.inner.gate.state();
            if matches!(gate_state, crate::AudioOperationState::Stopping) {
                return Err(AecCalibrationControlError::Stopping);
            }
            let Some(custody) = state.retained_custody.as_mut() else {
                return self
                    .inner
                    .gate
                    .acquire_calibration(attempt_id)
                    .map_err(map_gate_error);
            };
            if let Some(lease) = custody.lease.as_ref() {
                if gate_state != lease.state() {
                    return match gate_state {
                        crate::AudioOperationState::Stopping => {
                            Err(AecCalibrationControlError::Stopping)
                        }
                        _ => Err(AecCalibrationControlError::Busy),
                    };
                }
            } else {
                custody.lease = Some(
                    self.inner
                        .gate
                        .acquire_calibration(attempt_id)
                        .map_err(map_gate_error)?,
                );
                custody.attempt_id = attempt_id;
            }
            (custody.attempt_id, Arc::clone(&custody.cleanup_task))
        };
        self.inner.coordinator.revoke();
        if !self.run_retained_cleanup(&retained.1).await {
            return Err(AecCalibrationControlError::Busy);
        }
        let mut custody = {
            let mut state = lock_recovering(&self.inner.state);
            if state
                .retained_custody
                .as_ref()
                .is_none_or(|custody| custody.attempt_id != retained.0)
            {
                return Err(AecCalibrationControlError::Busy);
            }
            state
                .retained_custody
                .take()
                .expect("matching retained custody was checked above")
        };
        self.inner.coordinator.confirm_cleanup();
        let mut lease = custody
            .lease
            .take()
            .ok_or(AecCalibrationControlError::Busy)?;
        if lease.state() != (crate::AudioOperationState::Calibration { attempt_id }) {
            lease
                .relabel_calibration(attempt_id)
                .map_err(map_gate_error)?;
        }
        Ok(lease)
    }

    async fn run_retained_cleanup(
        &self,
        cleanup_task: &Arc<AsyncMutex<Option<JoinHandle<bool>>>>,
    ) -> bool {
        let deadline = Instant::now() + crate::RUNTIME_CLEANUP_BUDGET;
        let mut cleanup_task = cleanup_task.lock().await;
        if cleanup_task.is_none() {
            let cleanup = catch_unwind(AssertUnwindSafe(|| self.inner.engine.cleanup(deadline)));
            let Ok(cleanup) = cleanup else {
                return false;
            };
            *cleanup_task = Some(tokio::spawn(cleanup));
        }
        let task = cleanup_task
            .as_mut()
            .expect("cleanup task was installed above");
        let confirmed = match tokio::time::timeout_at(deadline, &mut *task).await {
            Ok(Ok(confirmed)) => confirmed,
            Ok(Err(_)) => false,
            Err(_) => {
                task.abort();
                let _ = task.await;
                false
            }
        };
        *cleanup_task = None;
        confirmed
    }

    async fn retry_retained_cleanup(&self) -> bool {
        let retained = {
            let state = lock_recovering(&self.inner.state);
            state
                .retained_custody
                .as_ref()
                .map(|retained| (retained.attempt_id, Arc::clone(&retained.cleanup_task)))
        };
        let Some((attempt_id, cleanup_task)) = retained else {
            return true;
        };
        let confirmed = self.run_retained_cleanup(&cleanup_task).await;
        if confirmed {
            let retained = {
                let mut state = lock_recovering(&self.inner.state);
                if state
                    .retained_custody
                    .as_ref()
                    .is_some_and(|retained| retained.attempt_id == attempt_id)
                {
                    state.retained_custody.take()
                } else {
                    None
                }
            };
            let Some(retained) = retained else {
                return false;
            };
            self.inner.coordinator.confirm_cleanup();
            drop(retained.lease);
            true
        } else {
            false
        }
    }
}

async fn join_active_task(
    task: &Arc<AsyncMutex<Option<JoinHandle<()>>>>,
) -> Result<(), AecCalibrationControlError> {
    let mut task = task.lock().await;
    let Some(active) = task.as_mut() else {
        return Ok(());
    };
    let result = active.await;
    *task = None;
    result.map_err(|_| AecCalibrationControlError::Unavailable)
}

async fn run_owned_attempt(
    inner: Arc<ControllerInner>,
    attempt_id: Uuid,
    interval_id: Uuid,
    lease: AudioOperationLease,
    deadline: Instant,
    cancellation: AecCalibrationCancellation,
) {
    let engine = Arc::clone(&inner.engine);
    let inspection_deadline = Instant::now() + crate::RUNTIME_CLEANUP_BUDGET;
    let inspection = tokio::task::spawn_blocking(move || {
        catch_unwind(AssertUnwindSafe(|| {
            engine.inspect_binding(inspection_deadline)
        }))
        .map_err(|_| "aec_calibration_inspection_failed")
        .and_then(|result| result.map_err(|error| error.code))
    })
    .await;

    if cancellation.is_cancelled() {
        finish_preflight(
            &inner,
            attempt_id,
            AecCalibrationControlStatus::Cancelled { attempt_id },
            lease,
        );
        return;
    }
    #[cfg(test)]
    wait_at_preflight_terminal_barrier(&inner).await;
    let binding = match inspection {
        Ok(Ok(binding)) => binding,
        Ok(Err(code)) => {
            finish_preflight(
                &inner,
                attempt_id,
                AecCalibrationControlStatus::Failed { attempt_id, code },
                lease,
            );
            return;
        }
        Err(_) => {
            finish_preflight(
                &inner,
                attempt_id,
                AecCalibrationControlStatus::Failed {
                    attempt_id,
                    code: "aec_calibration_inspection_failed",
                },
                lease,
            );
            return;
        }
    };
    let challenge = match inner
        .coordinator
        .begin_attempt(attempt_id, interval_id, binding)
    {
        Ok(challenge) => challenge,
        Err(error) => {
            finish_preflight(
                &inner,
                attempt_id,
                AecCalibrationControlStatus::Failed {
                    attempt_id,
                    code: coordinator_error_code(error),
                },
                lease,
            );
            return;
        }
    };
    let request = AecCalibrationRequest {
        attempt_id,
        challenge: challenge.clone(),
        deadline,
        cancellation: cancellation.clone(),
    };
    let launch = {
        let mut state = lock_recovering(&inner.state);
        let stopping = state.stopping;
        match state
            .active
            .as_mut()
            .filter(|active| active.attempt_id == attempt_id)
        {
            Some(active)
                if !stopping
                    && !active.cancellation.is_cancelled()
                    && active.phase == ActiveAttemptPhase::Inspecting =>
            {
                active.phase = ActiveAttemptPhase::Calibrating;
                match catch_unwind(AssertUnwindSafe(|| inner.engine.calibrate(request))) {
                    Ok(engine_future) => CalibrationLaunch::Started(engine_future),
                    Err(_) => CalibrationLaunch::Panicked,
                }
            }
            _ => CalibrationLaunch::Cancelled,
        }
    };
    let engine_future = match launch {
        CalibrationLaunch::Cancelled => {
            inner.coordinator.cancel_attempt(&challenge, true);
            finish_preflight(
                &inner,
                attempt_id,
                AecCalibrationControlStatus::Cancelled { attempt_id },
                lease,
            );
            return;
        }
        CalibrationLaunch::Started(engine_future) => Some(engine_future),
        CalibrationLaunch::Panicked => None,
    };
    run_attempt(
        inner,
        attempt_id,
        challenge,
        lease,
        deadline,
        cancellation,
        engine_future,
    )
    .await;
}

#[cfg(test)]
async fn wait_at_preflight_terminal_barrier(inner: &ControllerInner) {
    let barrier = lock_recovering(&inner.preflight_terminal_barrier).clone();
    if let Some(barrier) = barrier {
        barrier.entered.notify_one();
        barrier.released.notified().await;
    }
}

fn finish_preflight(
    inner: &ControllerInner,
    attempt_id: Uuid,
    terminal: AecCalibrationControlStatus,
    lease: AudioOperationLease,
) {
    drop(lease);
    let mut state = lock_recovering(&inner.state);
    if !state.stopping
        && state
            .active
            .as_ref()
            .is_some_and(|active| active.attempt_id == attempt_id)
    {
        state.status = if state
            .active
            .as_ref()
            .is_some_and(|active| active.cancellation.is_cancelled())
        {
            AecCalibrationControlStatus::Cancelled { attempt_id }
        } else {
            terminal
        };
    }
}

async fn run_attempt(
    inner: Arc<ControllerInner>,
    attempt_id: Uuid,
    challenge: AecCalibrationChallenge,
    lease: AudioOperationLease,
    deadline: Instant,
    cancellation: AecCalibrationCancellation,
    engine_future: Option<AecCalibrationFuture>,
) {
    let engine_future = match engine_future {
        Some(engine_future) => engine_future,
        None => {
            inner.coordinator.cancel_attempt(&challenge, false);
            let mut state = lock_recovering(&inner.state);
            state.retained_custody = Some(RetainedCustody {
                attempt_id,
                lease: Some(lease),
                cleanup_task: Arc::new(AsyncMutex::new(None)),
            });
            if !state.stopping
                && state
                    .active
                    .as_ref()
                    .is_some_and(|active| active.attempt_id == attempt_id)
            {
                state.status = AecCalibrationControlStatus::CleanupUncertain { attempt_id };
            }
            return;
        }
    };
    let mut engine_task = tokio::spawn(engine_future);
    let (terminal, cleanup_confirmed, graph_retained) = tokio::select! {
        biased;
        _ = cancellation.cancelled() => {
            let (cleanup_confirmed, graph_retained) =
                await_engine_cleanup(&mut engine_task, deadline).await;
            inner.coordinator.cancel_attempt(&challenge, cleanup_confirmed);
            (
                AecCalibrationControlStatus::Cancelled { attempt_id },
                cleanup_confirmed,
                graph_retained,
            )
        }
        _ = tokio::time::sleep_until(deadline) => {
            cancellation.cancel();
            let (cleanup_confirmed, graph_retained) =
                await_engine_cleanup(&mut engine_task, deadline).await;
            inner.coordinator.cancel_attempt(&challenge, cleanup_confirmed);
            (
                AecCalibrationControlStatus::TimedOut { attempt_id },
                cleanup_confirmed,
                graph_retained,
            )
        }
        result = &mut engine_task => finish_engine(
            &inner.coordinator,
            attempt_id,
            &challenge,
            &cancellation,
            result,
        ),
    };

    let mut terminal = if cleanup_confirmed {
        terminal
    } else {
        AecCalibrationControlStatus::CleanupUncertain { attempt_id }
    };
    let mut state = lock_recovering(&inner.state);
    if cancellation.is_cancelled()
        && matches!(terminal, AecCalibrationControlStatus::Succeeded { .. })
    {
        inner.coordinator.revoke();
        terminal = AecCalibrationControlStatus::Cancelled { attempt_id };
    }
    if graph_retained
        && matches!(terminal, AecCalibrationControlStatus::Succeeded { .. })
        && !cancellation.is_cancelled()
    {
        drop(lease);
        state.retained_custody = Some(RetainedCustody {
            attempt_id,
            lease: None,
            cleanup_task: Arc::new(AsyncMutex::new(None)),
        });
    } else if graph_retained {
        state.retained_custody = Some(RetainedCustody {
            attempt_id,
            lease: Some(lease),
            cleanup_task: Arc::new(AsyncMutex::new(None)),
        });
    } else if cleanup_confirmed {
        drop(lease);
    } else {
        state.retained_custody = Some(RetainedCustody {
            attempt_id,
            lease: Some(lease),
            cleanup_task: Arc::new(AsyncMutex::new(None)),
        });
    }
    if !state.stopping
        && state
            .active
            .as_ref()
            .is_some_and(|active| active.attempt_id == attempt_id)
    {
        state.status = terminal;
    }
}

fn finish_engine(
    coordinator: &AecCalibrationCoordinator,
    attempt_id: Uuid,
    challenge: &AecCalibrationChallenge,
    cancellation: &AecCalibrationCancellation,
    result: Result<Result<AecCalibrationPublication, AecCalibrationEngineError>, JoinError>,
) -> (AecCalibrationControlStatus, bool, bool) {
    match result {
        Ok(Ok(publication)) => {
            let cleanup_confirmed = publication.probe_teardown_confirmed;
            if cancellation.is_cancelled() {
                coordinator.cancel_attempt(challenge, cleanup_confirmed);
                return (
                    AecCalibrationControlStatus::Cancelled { attempt_id },
                    cleanup_confirmed,
                    publication.graph_retained,
                );
            }
            match coordinator.publish(
                challenge,
                publication.input,
                publication.probe_teardown_confirmed,
                publication.graph_retained,
            ) {
                Ok(()) if cancellation.is_cancelled() => {
                    coordinator.revoke();
                    (
                        AecCalibrationControlStatus::Cancelled { attempt_id },
                        cleanup_confirmed,
                        publication.graph_retained,
                    )
                }
                Ok(()) => (
                    AecCalibrationControlStatus::Succeeded {
                        attempt_id,
                        proof: coordinator.status(),
                    },
                    cleanup_confirmed,
                    publication.graph_retained,
                ),
                Err(error) => (
                    AecCalibrationControlStatus::Failed {
                        attempt_id,
                        code: coordinator_error_code(error),
                    },
                    cleanup_confirmed,
                    publication.graph_retained,
                ),
            }
        }
        Ok(Err(error)) => {
            coordinator.cancel_attempt(challenge, error.cleanup_confirmed);
            (
                AecCalibrationControlStatus::Failed {
                    attempt_id,
                    code: error.code,
                },
                error.cleanup_confirmed,
                false,
            )
        }
        Err(_) => {
            coordinator.cancel_attempt(challenge, false);
            (
                AecCalibrationControlStatus::Failed {
                    attempt_id,
                    code: "aec_calibration_task_failed",
                },
                false,
                false,
            )
        }
    }
}

async fn await_engine_cleanup(
    task: &mut JoinHandle<Result<AecCalibrationPublication, AecCalibrationEngineError>>,
    deadline: Instant,
) -> (bool, bool) {
    let cleanup_deadline = deadline.min(Instant::now() + crate::RUNTIME_CLEANUP_BUDGET);
    match tokio::time::timeout_at(cleanup_deadline, &mut *task).await {
        Ok(Ok(Ok(publication))) => (
            publication.probe_teardown_confirmed,
            publication.graph_retained,
        ),
        Ok(Ok(Err(error))) => (error.cleanup_confirmed, false),
        Ok(Err(_)) => (false, false),
        Err(_) => {
            task.abort();
            let _ = task.await;
            (false, false)
        }
    }
}

fn map_gate_error(error: AudioOperationAdmissionError) -> AecCalibrationControlError {
    match error {
        AudioOperationAdmissionError::Stopping => AecCalibrationControlError::Stopping,
        AudioOperationAdmissionError::Busy { .. }
        | AudioOperationAdmissionError::GenerationExhausted => AecCalibrationControlError::Busy,
    }
}

const fn coordinator_error_code(error: AecCoordinatorError) -> &'static str {
    match error {
        AecCoordinatorError::Busy => "aec_calibration_busy",
        AecCoordinatorError::InvalidChallenge | AecCoordinatorError::ChallengeConsumed => {
            "aec_calibration_challenge_invalid"
        }
        AecCoordinatorError::InvalidBinding => "aec_calibration_binding_invalid",
        AecCoordinatorError::InvalidMeasurement => "aec_calibration_evidence_invalid",
        AecCoordinatorError::MeasurementFailed => "aec_calibration_failed",
        AecCoordinatorError::ProbeTeardownIncomplete => "aec_calibration_cleanup_uncertain",
        AecCoordinatorError::GraphNotRetained => "aec_calibration_graph_not_retained",
        AecCoordinatorError::ProofUnavailable
        | AecCoordinatorError::ProofExpired
        | AecCoordinatorError::BindingChanged
        | AecCoordinatorError::GraphUnavailable
        | AecCoordinatorError::InvalidReservation => "aec_calibration_proof_unavailable",
    }
}

fn lock_recovering<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioOperationState;
    use std::sync::atomic::AtomicUsize;
    use translator_audio::{
        AEC_FIXTURE_DBFS, AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT,
        AEC_OBSERVATION_FRAME_SAMPLES, AEC_POWER_WINDOW_COUNT, AEC_SAMPLES_PER_POWER_WINDOW,
        AecDeviceMetadata, AecObservationEvidence, AecPositiveControl, AecPowerAcquisition,
        AecPowerWindow,
    };

    #[derive(Clone, Copy)]
    enum C9Outcome {
        Publication,
        RejectedPublication,
        RejectedBinding,
        CancelImmediatelyBeforePublication,
        PublicationAfterCancellation,
        Failure,
        Panic,
        Timeout,
    }

    struct C9CustodyEngine {
        gate: AudioOperationGate,
        coordinator: Arc<AecCalibrationCoordinator>,
        outcome: C9Outcome,
        cleanup_confirmed: AtomicBool,
        effects: Mutex<Vec<(&'static str, AudioOperationState, AecProofStatus)>>,
    }

    impl C9CustodyEngine {
        fn record(&self, operation: &'static str) {
            self.effects.lock().unwrap().push((
                operation,
                self.gate.state(),
                self.coordinator.status(),
            ));
        }
    }

    impl AecCalibrationEngine for C9CustodyEngine {
        fn inspect_binding(
            &self,
            _: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            self.record("inspect");
            Ok(test_binding())
        }

        fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
            self.record("calibrate");
            let outcome = self.outcome;
            let cancellation = request.cancellation.clone();
            let mut publication = successful_publication();
            if matches!(outcome, C9Outcome::RejectedPublication) {
                publication.input.windows.clear();
            }
            if matches!(outcome, C9Outcome::RejectedBinding) {
                publication.input.binding.source_port = "different-physical-port".into();
            }
            let publication = SuccessfulEngine(Mutex::new(Some(publication))).calibrate(request);
            Box::pin(async move {
                match outcome {
                    C9Outcome::PublicationAfterCancellation => cancellation.cancelled().await,
                    C9Outcome::Failure => {
                        return Err(AecCalibrationEngineError {
                            code: "injected_unknown_disposition",
                            cleanup_confirmed: false,
                        });
                    }
                    C9Outcome::Panic => panic!("injected unknown resource disposition"),
                    C9Outcome::Timeout => return std::future::pending().await,
                    _ => {}
                }
                let publication = publication.await?;
                if matches!(outcome, C9Outcome::CancelImmediatelyBeforePublication) {
                    // The completed publication already owns a retained graph.
                    // Cancel in the same poll immediately before delivering it.
                    cancellation.cancel();
                }
                Ok(publication)
            })
        }

        fn cleanup(&self, _: Instant) -> AecCleanupFuture {
            self.record("cleanup");
            let confirmed = self.cleanup_confirmed.load(Ordering::SeqCst);
            Box::pin(async move { confirmed })
        }
    }

    fn c9_custody_controller(
        outcome: C9Outcome,
    ) -> (AecCalibrationController, Arc<C9CustodyEngine>) {
        let coordinator = Arc::new(AecCalibrationCoordinator::new());
        let gate = AudioOperationGate::new();
        let engine = Arc::new(C9CustodyEngine {
            gate: gate.clone(),
            coordinator: coordinator.clone(),
            outcome,
            cleanup_confirmed: AtomicBool::new(true),
            effects: Mutex::new(Vec::new()),
        });
        (
            AecCalibrationController::new(coordinator, gate, engine.clone()),
            engine,
        )
    }

    #[tokio::test]
    async fn c9_s1a_replacement_is_admitted_before_revocation_and_cleanup() {
        let (controller, engine) = c9_custody_controller(C9Outcome::Publication);
        controller.start().await.unwrap();
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Succeeded { .. })
        })
        .await;
        let production = engine.gate.acquire_production().unwrap();
        let proof = engine.coordinator.status();
        let effects = engine.effects.lock().unwrap().clone();
        assert_eq!(
            controller.start().await,
            Err(AecCalibrationControlError::Busy)
        );
        assert_eq!(
            *engine.effects.lock().unwrap(),
            effects,
            "Busy must not clean a production-owned retained graph"
        );
        assert_eq!(engine.coordinator.status(), proof);
        drop(production);

        controller.start().await.unwrap();
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Succeeded { .. })
        })
        .await;
        let replacement = engine.effects.lock().unwrap()[effects.len()..].to_vec();
        assert_eq!(
            replacement
                .iter()
                .map(|effect| effect.0)
                .collect::<Vec<_>>(),
            ["cleanup", "inspect", "calibrate"]
        );
        assert!(
            replacement
                .iter()
                .all(|effect| matches!(effect.1, AudioOperationState::Calibration { .. }))
        );
        assert_eq!(
            replacement[0].2,
            AecProofStatus::Unavailable,
            "old proof must be revoked before teardown"
        );
        controller.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn c9_s1b_stopping_start_has_no_effects_or_custody_mutation() {
        let (controller, engine) = c9_custody_controller(C9Outcome::Publication);
        controller.start().await.unwrap();
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Succeeded { .. })
        })
        .await;
        let effects = engine.effects.lock().unwrap().clone();
        let proof = engine.coordinator.status();
        let custody = lock_recovering(&controller.inner.state)
            .retained_custody
            .as_ref()
            .unwrap()
            .attempt_id;
        engine.gate.begin_stopping();
        assert_eq!(
            controller.start().await,
            Err(AecCalibrationControlError::Stopping)
        );
        assert_eq!(
            *engine.effects.lock().unwrap(),
            effects,
            "Stopping must precede all cleanup and inspection effects"
        );
        assert_eq!(engine.coordinator.status(), proof);
        assert_eq!(
            lock_recovering(&controller.inner.state)
                .retained_custody
                .as_ref()
                .unwrap()
                .attempt_id,
            custody
        );
        controller.shutdown().await.unwrap();
    }

    async fn c9_assert_retained_graph_custody(outcome: C9Outcome) {
        let (controller, engine) = c9_custody_controller(outcome);
        engine.cleanup_confirmed.store(false, Ordering::SeqCst);
        let AecCalibrationControlStatus::Running { attempt_id } = controller.start().await.unwrap()
        else {
            panic!("attempt not admitted")
        };
        if matches!(outcome, C9Outcome::PublicationAfterCancellation) {
            while !engine
                .effects
                .lock()
                .unwrap()
                .iter()
                .any(|effect| effect.0 == "calibrate")
            {
                tokio::task::yield_now().await;
            }
            controller.cancel(attempt_id).await;
        }
        wait_until(&controller, |status| {
            !matches!(status, AecCalibrationControlStatus::Running { .. })
        })
        .await;
        assert!(!matches!(
            engine.coordinator.status(),
            AecProofStatus::Validated { .. }
        ));
        assert!(
            matches!(engine.gate.state(), AudioOperationState::Calibration { .. }),
            "invalid authority is not evidence of retained graph absence"
        );
        assert_eq!(
            lock_recovering(&controller.inner.state)
                .retained_custody
                .as_ref()
                .expect("terminal attempt must retain its graph owner")
                .attempt_id,
            attempt_id
        );
        assert!(engine.gate.acquire_production().is_err());
        assert_eq!(
            controller.start().await,
            Err(AecCalibrationControlError::Busy),
            "failed exact cleanup cannot admit a replacement"
        );
        assert_eq!(
            lock_recovering(&controller.inner.state)
                .retained_custody
                .as_ref()
                .expect("failed cleanup must preserve the same owner")
                .attempt_id,
            attempt_id
        );
        assert!(matches!(
            engine.gate.state(),
            AudioOperationState::Calibration { .. }
        ));
        assert!(engine.gate.acquire_production().is_err());
        assert!(
            engine
                .effects
                .lock()
                .unwrap()
                .iter()
                .any(|effect| effect.0 == "cleanup")
        );
        engine.cleanup_confirmed.store(true, Ordering::SeqCst);
        controller.shutdown().await.unwrap();
        assert!(
            lock_recovering(&controller.inner.state)
                .retained_custody
                .is_none()
        );
    }

    #[tokio::test]
    async fn c9_s2a_rejected_publication_keeps_graph_custody() {
        c9_assert_retained_graph_custody(C9Outcome::RejectedPublication).await;
    }

    #[tokio::test]
    async fn c9_s2a_rejected_binding_keeps_graph_custody() {
        c9_assert_retained_graph_custody(C9Outcome::RejectedBinding).await;
    }

    #[tokio::test]
    async fn c9_s2a_publication_awaited_after_cancel_keeps_graph_custody() {
        c9_assert_retained_graph_custody(C9Outcome::PublicationAfterCancellation).await;
    }

    #[tokio::test]
    async fn c9_s2a_cancel_before_publication_preserves_resource_facts() {
        c9_assert_retained_graph_custody(C9Outcome::CancelImmediatelyBeforePublication).await;
    }

    #[tokio::test(start_paused = true)]
    async fn c9_s2b_unknown_failure_panic_and_timeout_require_exact_cleanup() {
        for outcome in [C9Outcome::Failure, C9Outcome::Panic, C9Outcome::Timeout] {
            let (controller, engine) = c9_custody_controller(outcome);
            engine.cleanup_confirmed.store(false, Ordering::SeqCst);
            controller.start().await.unwrap();
            tokio::task::yield_now().await;
            if matches!(outcome, C9Outcome::Timeout) {
                tokio::time::advance(AEC_CALIBRATION_BUDGET + Duration::from_nanos(1)).await;
            }
            wait_until(&controller, |status| {
                matches!(status, AecCalibrationControlStatus::CleanupUncertain { .. })
            })
            .await;
            assert!(engine.gate.acquire_production().is_err());
            assert!(
                lock_recovering(&controller.inner.state)
                    .retained_custody
                    .is_some()
            );
            assert!(controller.shutdown().await.is_err());
            assert!(
                lock_recovering(&controller.inner.state)
                    .retained_custody
                    .is_some()
            );
            engine.cleanup_confirmed.store(true, Ordering::SeqCst);
            controller.shutdown().await.unwrap();
            assert!(
                lock_recovering(&controller.inner.state)
                    .retained_custody
                    .is_none()
            );
        }
    }

    struct C9HeldInspection {
        entered: Notify,
        released: (Mutex<bool>, std::sync::Condvar),
        calibrations: AtomicUsize,
        fail_inspection: bool,
    }

    impl AecCalibrationEngine for C9HeldInspection {
        fn inspect_binding(
            &self,
            _: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            self.entered.notify_one();
            let released = self.released.0.lock().unwrap();
            let (released, _) = self
                .released
                .1
                .wait_timeout_while(released, Duration::from_secs(3), |released| !*released)
                .unwrap();
            if !*released {
                return Err(AecCalibrationEngineError {
                    code: "test_barrier_expired",
                    cleanup_confirmed: false,
                });
            }
            if self.fail_inspection {
                return Err(AecCalibrationEngineError {
                    code: "injected_inspection_failure",
                    cleanup_confirmed: true,
                });
            }
            Ok(test_binding())
        }

        fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
            self.calibrations.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                request.cancellation.cancelled().await;
                Err(AecCalibrationEngineError {
                    code: "cancelled",
                    cleanup_confirmed: true,
                })
            })
        }

        fn cleanup(&self, _: Instant) -> AecCleanupFuture {
            Box::pin(async { true })
        }
    }

    struct C9ReleaseInspection(Arc<C9HeldInspection>);

    impl Drop for C9ReleaseInspection {
        fn drop(&mut self) {
            *self.0.released.0.lock().unwrap() = true;
            self.0.released.1.notify_all();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn c9_s6_initial_inspection_is_owned_without_blocking_start_status_or_cancel() {
        let engine = Arc::new(C9HeldInspection {
            entered: Notify::new(),
            released: (Mutex::new(false), std::sync::Condvar::new()),
            calibrations: AtomicUsize::new(0),
            fail_inspection: false,
        });
        let release = C9ReleaseInspection(engine.clone());
        let (controller, coordinator, gate) = controller(engine.clone());
        let start_controller = controller.clone();
        let mut start = tokio::spawn(async move { start_controller.start().await });
        tokio::time::timeout(Duration::from_secs(1), engine.entered.notified())
            .await
            .unwrap();
        let accepted = tokio::time::timeout(Duration::from_millis(100), &mut start).await;
        let accepted = match accepted {
            Ok(accepted) => accepted.unwrap().unwrap(),
            Err(_) => {
                drop(release);
                let _ = start.await;
                controller.shutdown().await.unwrap();
                panic!("Start must expose an attempt ID while initial inspection remains held");
            }
        };
        let AecCalibrationControlStatus::Running { attempt_id } = accepted else {
            panic!("attempt not running")
        };
        let status_controller = controller.clone();
        let status = tokio::time::timeout(
            Duration::from_millis(100),
            tokio::spawn(async move { status_controller.status() }),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(status, AecCalibrationControlStatus::Running { attempt_id });
        tokio::time::timeout(Duration::from_millis(100), controller.cancel(attempt_id))
            .await
            .unwrap();
        let shutdown_controller = controller.clone();
        let shutdown = tokio::spawn(async move { shutdown_controller.shutdown().await });
        tokio::time::timeout(Duration::from_millis(100), async {
            while gate.state() != AudioOperationState::Stopping {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("shutdown must close admission before joining inspection");
        assert!(
            !shutdown.is_finished(),
            "held blocking inspection must remain joined by shutdown"
        );
        drop(release);
        tokio::time::timeout(Duration::from_secs(1), shutdown)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(engine.calibrations.load(Ordering::SeqCst), 0);
        assert!(!matches!(
            coordinator.status(),
            AecProofStatus::Validated { .. }
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn c9_s6_cancelled_inspection_failure_cannot_overwrite_cancelled_terminal() {
        let engine = Arc::new(C9HeldInspection {
            entered: Notify::new(),
            released: (Mutex::new(false), std::sync::Condvar::new()),
            calibrations: AtomicUsize::new(0),
            fail_inspection: true,
        });
        let release = C9ReleaseInspection(engine.clone());
        let (controller, coordinator, _) = controller(engine.clone());
        let terminal_barrier = Arc::new(PreflightTerminalBarrier {
            entered: Notify::new(),
            released: Notify::new(),
        });
        *lock_recovering(&controller.inner.preflight_terminal_barrier) =
            Some(Arc::clone(&terminal_barrier));
        let AecCalibrationControlStatus::Running { attempt_id } = controller.start().await.unwrap()
        else {
            panic!("attempt not admitted")
        };
        tokio::time::timeout(Duration::from_secs(1), engine.entered.notified())
            .await
            .unwrap();
        drop(release);
        tokio::time::timeout(Duration::from_secs(1), terminal_barrier.entered.notified())
            .await
            .expect("inspection result must cross the initial cancellation check");
        controller.cancel(attempt_id).await;
        terminal_barrier.released.notify_one();
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Cancelled { attempt_id: current } if *current == attempt_id)
        })
        .await;
        assert_eq!(engine.calibrations.load(Ordering::SeqCst), 0);
        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        controller.shutdown().await.unwrap();
    }

    struct C9HandoffEngine {
        entered: Notify,
        released: (Mutex<bool>, std::sync::Condvar),
        calibrations: AtomicUsize,
    }

    impl AecCalibrationEngine for C9HandoffEngine {
        fn inspect_binding(
            &self,
            _: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            Ok(test_binding())
        }

        fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
            self.calibrations.fetch_add(1, Ordering::SeqCst);
            self.entered.notify_one();
            let released = self.released.0.lock().unwrap();
            let (released, _) = self
                .released
                .1
                .wait_timeout_while(released, Duration::from_secs(3), |released| !*released)
                .unwrap();
            assert!(*released, "calibration handoff barrier expired");
            Box::pin(async move {
                request.cancellation.cancelled().await;
                Err(AecCalibrationEngineError {
                    code: "cancelled",
                    cleanup_confirmed: true,
                })
            })
        }

        fn cleanup(&self, _: Instant) -> AecCleanupFuture {
            Box::pin(async { true })
        }
    }

    struct C9ReleaseHandoff(Arc<C9HandoffEngine>);

    impl Drop for C9ReleaseHandoff {
        fn drop(&mut self) {
            *self.0.released.0.lock().unwrap() = true;
            self.0.released.1.notify_all();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn c9_s6_cancel_is_linearized_with_inspection_to_calibration_handoff() {
        let engine = Arc::new(C9HandoffEngine {
            entered: Notify::new(),
            released: (Mutex::new(false), std::sync::Condvar::new()),
            calibrations: AtomicUsize::new(0),
        });
        let release = C9ReleaseHandoff(engine.clone());
        let (controller, coordinator, _) = controller(engine.clone());
        let AecCalibrationControlStatus::Running { attempt_id } = controller.start().await.unwrap()
        else {
            panic!("attempt not admitted")
        };
        tokio::time::timeout(Duration::from_secs(1), engine.entered.notified())
            .await
            .unwrap();

        let cancel_controller = controller.clone();
        let mut cancel = tokio::spawn(async move { cancel_controller.cancel(attempt_id).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut cancel)
                .await
                .is_err(),
            "cancel returned before the admitted calibration launch was linearized"
        );
        drop(release);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), cancel)
                .await
                .unwrap()
                .unwrap(),
            AecCalibrationControlStatus::Running { attempt_id }
        );
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Cancelled { attempt_id: current } if *current == attempt_id)
        })
        .await;
        assert_eq!(engine.calibrations.load(Ordering::SeqCst), 1);
        assert!(!matches!(
            coordinator.status(),
            AecProofStatus::Validated { .. }
        ));
        controller.shutdown().await.unwrap();
    }

    struct CancelAwareEngine {
        calls: AtomicUsize,
        cleanup_confirmed: bool,
        cleanup_calls: AtomicUsize,
        cleanup_confirm_after: usize,
    }

    impl AecCalibrationEngine for CancelAwareEngine {
        fn inspect_binding(
            &self,
            _deadline: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            Ok(test_binding())
        }

        fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let cleanup_confirmed = self.cleanup_confirmed;
            Box::pin(async move {
                request.cancellation.cancelled().await;
                Err(AecCalibrationEngineError {
                    code: "cancelled",
                    cleanup_confirmed,
                })
            })
        }

        fn cleanup(&self, _deadline: Instant) -> AecCleanupFuture {
            let call = self.cleanup_calls.fetch_add(1, Ordering::SeqCst) + 1;
            let confirmed = call >= self.cleanup_confirm_after;
            Box::pin(async move { confirmed })
        }
    }

    struct PanicEngine;

    impl AecCalibrationEngine for PanicEngine {
        fn inspect_binding(
            &self,
            _deadline: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            Ok(test_binding())
        }

        fn calibrate(&self, _request: AecCalibrationRequest) -> AecCalibrationFuture {
            panic!("injected calibration panic")
        }

        fn cleanup(&self, _deadline: Instant) -> AecCleanupFuture {
            Box::pin(async { false })
        }
    }

    struct NonCooperativeEngine;

    impl AecCalibrationEngine for NonCooperativeEngine {
        fn inspect_binding(
            &self,
            _deadline: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            Ok(test_binding())
        }

        fn calibrate(&self, _request: AecCalibrationRequest) -> AecCalibrationFuture {
            Box::pin(std::future::pending())
        }

        fn cleanup(&self, _deadline: Instant) -> AecCleanupFuture {
            Box::pin(async { false })
        }
    }

    struct BlockingCleanupEngine {
        calibration_calls: AtomicUsize,
        cleanup_calls: AtomicUsize,
        cleanup_started: Arc<Notify>,
        cleanup_release: Arc<Notify>,
    }

    impl AecCalibrationEngine for BlockingCleanupEngine {
        fn inspect_binding(
            &self,
            _deadline: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            Ok(test_binding())
        }

        fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
            self.calibration_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                request.cancellation.cancelled().await;
                Err(AecCalibrationEngineError {
                    code: "cancelled",
                    cleanup_confirmed: false,
                })
            })
        }

        fn cleanup(&self, _deadline: Instant) -> AecCleanupFuture {
            self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
            self.cleanup_started.notify_one();
            let cleanup_release = Arc::clone(&self.cleanup_release);
            Box::pin(async move {
                cleanup_release.notified().await;
                true
            })
        }
    }

    struct BlockingShutdownEngine {
        calibration_calls: AtomicUsize,
        cancellation_observed: Arc<Notify>,
        calibration_release: Arc<Notify>,
    }

    impl AecCalibrationEngine for BlockingShutdownEngine {
        fn inspect_binding(
            &self,
            _deadline: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            Ok(test_binding())
        }

        fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
            self.calibration_calls.fetch_add(1, Ordering::SeqCst);
            let cancellation_observed = Arc::clone(&self.cancellation_observed);
            let calibration_release = Arc::clone(&self.calibration_release);
            Box::pin(async move {
                request.cancellation.cancelled().await;
                cancellation_observed.notify_one();
                calibration_release.notified().await;
                Err(AecCalibrationEngineError {
                    code: "cancelled",
                    cleanup_confirmed: true,
                })
            })
        }

        fn cleanup(&self, _deadline: Instant) -> AecCleanupFuture {
            Box::pin(async { true })
        }
    }

    struct SuccessfulEngine(Mutex<Option<AecCalibrationPublication>>);

    impl AecCalibrationEngine for SuccessfulEngine {
        fn inspect_binding(
            &self,
            _deadline: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            Ok(test_binding())
        }

        fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
            let mut publication = lock_recovering(&self.0).take().unwrap();
            let observation = &mut publication.input.observation;
            observation.calibration_attempt_id = request.challenge.attempt_id().to_string();
            observation.challenge_id = request.challenge.challenge_id().to_string();
            observation.interval_id = request.challenge.interval_id().to_string();
            observation.positive_control.calibration_attempt_id =
                observation.calibration_attempt_id.clone();
            observation.positive_control.challenge_id = observation.challenge_id.clone();
            Box::pin(async move { Ok(publication) })
        }

        fn cleanup(&self, _deadline: Instant) -> AecCleanupFuture {
            Box::pin(async { true })
        }
    }

    struct TrackingSuccessfulEngine {
        publication: SuccessfulEngine,
        cleanup_calls: AtomicUsize,
    }

    impl AecCalibrationEngine for TrackingSuccessfulEngine {
        fn inspect_binding(
            &self,
            deadline: Instant,
        ) -> Result<AecProofBinding, AecCalibrationEngineError> {
            self.publication.inspect_binding(deadline)
        }

        fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
            self.publication.calibrate(request)
        }

        fn cleanup(&self, _deadline: Instant) -> AecCleanupFuture {
            self.cleanup_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { true })
        }
    }

    fn test_binding() -> AecProofBinding {
        AecProofBinding {
            audio_server_id: "server-1".into(),
            source_hardware_id: "source-hardware-1".into(),
            sink_hardware_id: "sink-hardware-1".into(),
            source_name: "physical-source".into(),
            sink_name: "physical-sink".into(),
            source_port: "mic".into(),
            sink_port: "speaker".into(),
            source_channel_gains: vec![65_536],
            sink_channel_gains: vec![32_768],
            source_muted: false,
            sink_muted: false,
            source_geometry: "desk-left".into(),
            sink_geometry: "desk-front".into(),
            aec_module_id: 1,
            aec_source_id: 2,
            aec_sink_id: 3,
            aec_generation: "generation-1".into(),
            aec_config_id: "aec-1".into(),
            vad_config_id: "vad-1".into(),
            provider_config_id: "provider-1".into(),
        }
    }

    fn successful_publication() -> AecCalibrationPublication {
        let acquisition = |id: &str, power| AecPowerAcquisition {
            acquisition_id: id.into(),
            samples_per_window: AEC_SAMPLES_PER_POWER_WINDOW,
            powers: vec![power; 5],
        };
        let binding = test_binding();
        let observation = AecObservationEvidence {
            observer_generation: "observer-1".into(),
            calibration_attempt_id: "attempt-1".into(),
            challenge_id: "challenge-1".into(),
            interval_id: "interval-1".into(),
            started_monotonic_ns: 10_000_000_000,
            ended_monotonic_ns: 10_000_000_000 + AEC_OBSERVATION_DURATION_NS,
            expected_frames: AEC_OBSERVATION_FRAME_COUNT,
            processed_frames: AEC_OBSERVATION_FRAME_COUNT,
            stream_generation: "runtime-generation-1".into(),
            sample_rate_hz: 16_000,
            channels: 1,
            frame_duration_ms: 20,
            samples_per_frame: AEC_OBSERVATION_FRAME_SAMPLES,
            first_frame_sequence: 0,
            last_frame_sequence: AEC_OBSERVATION_FRAME_COUNT - 1,
            first_capture_monotonic_ns: 10_000_000_000,
            last_capture_monotonic_ns: 10_000_000_000 + AEC_OBSERVATION_DURATION_NS - 20_000_000,
            maximum_frame_gap_ns: 20_000_000,
            frame_gaps: 0,
            duplicate_frames: 0,
            out_of_order_frames: 0,
            vad_events_before: 0,
            vad_events_after: 0,
            provider_attempts_before: 0,
            provider_attempts_after: 0,
            provider_accepted_before: 0,
            provider_accepted_after: 0,
            resets: 0,
            dropped_frames: 0,
            observer_errors: 0,
            terminated_early: false,
            positive_control: AecPositiveControl {
                observer_generation: "observer-1".into(),
                calibration_attempt_id: "attempt-1".into(),
                challenge_id: "challenge-1".into(),
                completed_monotonic_ns: 9_000_000_000,
                speech_started_events: 1,
                provider_submission_attempts: 1,
                provider_submissions_accepted: 1,
                resets: 0,
                observer_errors: 0,
            },
        };
        AecCalibrationPublication {
            input: AecValidationInput {
                metadata: AecDeviceMetadata {
                    source_name: binding.source_name.clone(),
                    sink_name: binding.sink_name.clone(),
                    source_geometry: binding.source_geometry.clone(),
                    sink_geometry: binding.sink_geometry.clone(),
                    sink_port: binding.sink_port.clone(),
                    sink_volume_percent: 40,
                },
                binding: binding.measurement_binding(),
                fixture_acquisition_id: "fixture-1".into(),
                raw_baseline: acquisition("raw-baseline", 1.0),
                clean_baseline: acquisition("clean-baseline", 1.0),
                resolution: acquisition("resolution", 1.0),
                windows: (0..AEC_POWER_WINDOW_COUNT)
                    .map(|sequence| {
                        let start = sequence as u64 * AEC_SAMPLES_PER_POWER_WINDOW;
                        AecPowerWindow {
                            sequence: sequence as u64,
                            raw_start_sample: start,
                            raw_end_sample: start + AEC_SAMPLES_PER_POWER_WINDOW,
                            clean_start_sample: start,
                            clean_end_sample: start + AEC_SAMPLES_PER_POWER_WINDOW,
                            raw_power: 101.0,
                            clean_power: 1.0 + 100.0 / 10_f64.powf(1.6),
                            raw_clipped_samples: 0,
                            clean_clipped_samples: 0,
                            fixture_dbfs: AEC_FIXTURE_DBFS,
                        }
                    })
                    .collect(),
                observation,
            },
            probe_teardown_confirmed: true,
            graph_retained: true,
        }
    }

    async fn wait_until(
        controller: &AecCalibrationController,
        predicate: impl Fn(&AecCalibrationControlStatus) -> bool,
    ) {
        for _ in 0..10_000 {
            let status = controller.status();
            if predicate(&status) {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "status did not reach expected state: {:?}",
            controller.status()
        );
    }

    async fn wait_until_measuring(coordinator: &AecCalibrationCoordinator) {
        for _ in 0..10_000 {
            if coordinator.status() == AecProofStatus::Measuring {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "coordinator did not reach measuring state: {:?}",
            coordinator.status()
        );
    }

    fn controller(
        engine: Arc<dyn AecCalibrationEngine>,
    ) -> (
        AecCalibrationController,
        Arc<AecCalibrationCoordinator>,
        AudioOperationGate,
    ) {
        let coordinator = Arc::new(AecCalibrationCoordinator::new());
        let gate = AudioOperationGate::new();
        (
            AecCalibrationController::new(coordinator.clone(), gate.clone(), engine),
            coordinator,
            gate,
        )
    }

    #[tokio::test]
    async fn cancel_joins_cleanup_before_releasing_gate_and_never_mints_proof() {
        let engine = Arc::new(CancelAwareEngine {
            calls: AtomicUsize::new(0),
            cleanup_confirmed: true,
            cleanup_calls: AtomicUsize::new(0),
            cleanup_confirm_after: 1,
        });
        let (controller, coordinator, gate) = controller(engine.clone());
        let attempt_id = match controller.start().await.unwrap() {
            AecCalibrationControlStatus::Running { attempt_id } => attempt_id,
            status => panic!("unexpected status: {status:?}"),
        };
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { .. }
        ));
        wait_until_measuring(&coordinator).await;
        controller.cancel(attempt_id).await;
        controller.cancel(attempt_id).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Cancelled { .. })
        })
        .await;
        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        assert_eq!(gate.state(), AudioOperationState::Idle);
        assert_eq!(engine.calls.load(Ordering::SeqCst), 1);

        // Reaping is part of retry admission; an old attempt cannot cancel its successor.
        let second_id = loop {
            match controller.start().await {
                Ok(AecCalibrationControlStatus::Running { attempt_id }) => break attempt_id,
                Err(AecCalibrationControlError::Busy) => tokio::task::yield_now().await,
                result => panic!("unexpected retry result: {result:?}"),
            }
        };
        wait_until_measuring(&coordinator).await;
        controller.cancel(attempt_id).await;
        assert_eq!(
            controller.status(),
            AecCalibrationControlStatus::Running {
                attempt_id: second_id
            }
        );
        controller.cancel(second_id).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Cancelled { attempt_id } if *attempt_id == second_id)
        })
        .await;
        controller.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn uncertain_cleanup_retains_exclusive_gate_and_blocks_retry() {
        let engine = Arc::new(CancelAwareEngine {
            calls: AtomicUsize::new(0),
            cleanup_confirmed: false,
            cleanup_calls: AtomicUsize::new(0),
            cleanup_confirm_after: 2,
        });
        let (controller, coordinator, gate) = controller(engine);
        let attempt_id = match controller.start().await.unwrap() {
            AecCalibrationControlStatus::Running { attempt_id } => attempt_id,
            status => panic!("unexpected status: {status:?}"),
        };
        wait_until_measuring(&coordinator).await;
        controller.cancel(attempt_id).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::CleanupUncertain { .. })
        })
        .await;
        assert_eq!(coordinator.status(), AecProofStatus::CleanupUncertain);
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { .. }
        ));
        assert_eq!(
            controller.start().await,
            Err(AecCalibrationControlError::Busy)
        );
        let retry = controller.start().await.unwrap();
        let retry_id = match retry {
            AecCalibrationControlStatus::Running { attempt_id } => attempt_id,
            status => panic!("unexpected retry status: {status:?}"),
        };
        wait_until_measuring(&coordinator).await;
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { attempt_id } if attempt_id == retry_id
        ));
        controller.cancel(retry_id).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::CleanupUncertain { attempt_id } if *attempt_id == retry_id)
        })
        .await;
        controller.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn aborted_cleanup_retry_keeps_gate_and_join_owner_until_next_retry() {
        let engine = Arc::new(BlockingCleanupEngine {
            calibration_calls: AtomicUsize::new(0),
            cleanup_calls: AtomicUsize::new(0),
            cleanup_started: Arc::new(Notify::new()),
            cleanup_release: Arc::new(Notify::new()),
        });
        let (controller, coordinator, gate) = controller(engine.clone());
        let first_id = match controller.start().await.unwrap() {
            AecCalibrationControlStatus::Running { attempt_id } => attempt_id,
            status => panic!("unexpected status: {status:?}"),
        };
        wait_until_measuring(&coordinator).await;
        controller.cancel(first_id).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::CleanupUncertain { .. })
        })
        .await;

        let retry_controller = controller.clone();
        let retry = tokio::spawn(async move { retry_controller.start().await });
        engine.cleanup_started.notified().await;
        retry.abort();
        assert!(retry.await.unwrap_err().is_cancelled());
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { attempt_id } if attempt_id == first_id
        ));

        let retry_controller = controller.clone();
        let retry = tokio::spawn(async move { retry_controller.start().await });
        tokio::task::yield_now().await;
        assert!(!retry.is_finished());
        assert_eq!(engine.cleanup_calls.load(Ordering::SeqCst), 1);
        engine.cleanup_release.notify_one();
        let second_id = match retry.await.unwrap().unwrap() {
            AecCalibrationControlStatus::Running { attempt_id } => attempt_id,
            status => panic!("unexpected status: {status:?}"),
        };
        assert_ne!(second_id, first_id);
        assert_eq!(engine.cleanup_calls.load(Ordering::SeqCst), 1);

        wait_until_measuring(&coordinator).await;
        controller.cancel(second_id).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::CleanupUncertain { attempt_id } if *attempt_id == second_id)
        })
        .await;
        let shutdown_controller = controller.clone();
        let shutdown = tokio::spawn(async move { shutdown_controller.shutdown().await });
        engine.cleanup_started.notified().await;
        engine.cleanup_release.notify_one();
        shutdown.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn one_absolute_180_second_deadline_aborts_at_expiry_without_proof() {
        let engine = Arc::new(CancelAwareEngine {
            calls: AtomicUsize::new(0),
            cleanup_confirmed: true,
            cleanup_calls: AtomicUsize::new(0),
            cleanup_confirm_after: 1,
        });
        let (controller, coordinator, gate) = controller(engine);
        controller.start().await.unwrap();
        wait_until_measuring(&coordinator).await;
        tokio::time::advance(AEC_CALIBRATION_BUDGET - Duration::from_nanos(1)).await;
        assert!(matches!(
            controller.status(),
            AecCalibrationControlStatus::Running { .. }
        ));
        tokio::time::advance(Duration::from_nanos(1)).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::CleanupUncertain { .. })
        })
        .await;
        assert_eq!(coordinator.status(), AecProofStatus::CleanupUncertain);
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { .. }
        ));
        controller.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn engine_panic_is_fail_closed_and_retains_uncertain_custody() {
        let (controller, coordinator, gate) = controller(Arc::new(PanicEngine));
        controller.start().await.unwrap();
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::CleanupUncertain { .. })
        })
        .await;
        assert_eq!(coordinator.status(), AecProofStatus::CleanupUncertain);
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { .. }
        ));
        assert_eq!(
            controller.shutdown().await,
            Err(AecCalibrationControlError::Unavailable)
        );
    }

    #[tokio::test]
    async fn cancel_after_publication_revokes_the_winning_proof() {
        let engine = Arc::new(SuccessfulEngine(Mutex::new(Some(successful_publication()))));
        let (controller, coordinator, gate) = controller(engine);
        let attempt_id = match controller.start().await.unwrap() {
            AecCalibrationControlStatus::Running { attempt_id } => attempt_id,
            status => panic!("unexpected status: {status:?}"),
        };
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Succeeded { .. })
        })
        .await;
        assert!(matches!(
            coordinator.status(),
            AecProofStatus::Validated { .. }
        ));
        controller.cancel(attempt_id).await;
        assert_eq!(
            controller.status(),
            AecCalibrationControlStatus::Cancelled { attempt_id }
        );
        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        assert_eq!(gate.state(), AudioOperationState::Idle);
        controller.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_after_success_revokes_proof_and_clears_active_owner() {
        let engine = Arc::new(TrackingSuccessfulEngine {
            publication: SuccessfulEngine(Mutex::new(Some(successful_publication()))),
            cleanup_calls: AtomicUsize::new(0),
        });
        let (controller, coordinator, gate) = controller(engine.clone());
        controller.start().await.unwrap();
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::Succeeded { .. })
        })
        .await;
        assert!(matches!(
            coordinator.status(),
            AecProofStatus::Validated { .. }
        ));

        controller.shutdown().await.unwrap();

        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        assert_eq!(gate.state(), AudioOperationState::Stopping);
        assert!(lock_recovering(&controller.inner.state).active.is_none());
        assert_eq!(engine.cleanup_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_cleanup_is_bounded_by_the_runtime_cleanup_budget() {
        let (controller, coordinator, gate) = controller(Arc::new(NonCooperativeEngine));
        let attempt_id = match controller.start().await.unwrap() {
            AecCalibrationControlStatus::Running { attempt_id } => attempt_id,
            status => panic!("unexpected status: {status:?}"),
        };
        wait_until_measuring(&coordinator).await;
        controller.cancel(attempt_id).await;
        tokio::task::yield_now().await;
        tokio::time::advance(crate::RUNTIME_CLEANUP_BUDGET + Duration::from_nanos(1)).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::CleanupUncertain { .. })
        })
        .await;
        assert_eq!(coordinator.status(), AecProofStatus::CleanupUncertain);
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { .. }
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn non_cooperative_engine_is_aborted_at_deadline_and_gate_remains_retained() {
        let (controller, coordinator, gate) = controller(Arc::new(NonCooperativeEngine));
        controller.start().await.unwrap();
        wait_until_measuring(&coordinator).await;
        tokio::time::advance(AEC_CALIBRATION_BUDGET).await;
        wait_until(&controller, |status| {
            matches!(status, AecCalibrationControlStatus::CleanupUncertain { .. })
        })
        .await;
        assert_eq!(coordinator.status(), AecProofStatus::CleanupUncertain);
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { .. }
        ));
        assert_eq!(
            controller.shutdown().await,
            Err(AecCalibrationControlError::Unavailable)
        );
    }

    #[tokio::test]
    async fn shutdown_closes_admission_cancels_and_joins_the_active_attempt() {
        let engine = Arc::new(CancelAwareEngine {
            calls: AtomicUsize::new(0),
            cleanup_confirmed: true,
            cleanup_calls: AtomicUsize::new(0),
            cleanup_confirm_after: 1,
        });
        let (controller, coordinator, gate) = controller(engine);
        controller.start().await.unwrap();
        controller.shutdown().await.unwrap();
        assert_eq!(
            controller.status(),
            AecCalibrationControlStatus::ShuttingDown
        );
        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        assert_eq!(gate.state(), AudioOperationState::Stopping);
        assert_eq!(
            controller.start().await,
            Err(AecCalibrationControlError::Stopping)
        );
    }

    #[tokio::test]
    async fn aborted_shutdown_keeps_active_join_owner_for_next_shutdown() {
        let engine = Arc::new(BlockingShutdownEngine {
            calibration_calls: AtomicUsize::new(0),
            cancellation_observed: Arc::new(Notify::new()),
            calibration_release: Arc::new(Notify::new()),
        });
        let (controller, coordinator, gate) = controller(engine.clone());
        controller.start().await.unwrap();
        wait_until_measuring(&coordinator).await;

        let first_controller = controller.clone();
        let first_shutdown = tokio::spawn(async move { first_controller.shutdown().await });
        engine.cancellation_observed.notified().await;
        first_shutdown.abort();
        assert!(first_shutdown.await.unwrap_err().is_cancelled());
        assert_eq!(
            controller.status(),
            AecCalibrationControlStatus::ShuttingDown
        );
        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        assert_eq!(gate.state(), AudioOperationState::Stopping);

        let second_controller = controller.clone();
        let second_shutdown = tokio::spawn(async move { second_controller.shutdown().await });
        tokio::task::yield_now().await;
        assert!(!second_shutdown.is_finished());
        assert_eq!(engine.calibration_calls.load(Ordering::SeqCst), 1);
        engine.calibration_release.notify_one();
        second_shutdown.await.unwrap().unwrap();
        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        assert_eq!(gate.state(), AudioOperationState::Stopping);
    }
}
