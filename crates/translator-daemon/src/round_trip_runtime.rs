use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use thiserror::Error;
use translator_audio::{GraphHealth, OutputMode, RouteResolution};
use translator_core::ProviderId;
use uuid::Uuid;

use crate::{
    AudioOperationGate, AudioOperationLease, ControlFailure, ExactPcmProof, RoundTripCheckpoint,
    RoundTripController, RoundTripErrorCode, RoundTripLatency, RoundTripPreconditions,
    RoundTripSelfTest, RoundTripSelfTestState, RuntimeSnapshot, RuntimeStore,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RoundTripRuntimeError {
    #[error("round-trip runtime could not start")]
    StartFailed,
    #[error("round-trip runtime could not stop")]
    StopFailed,
}

pub trait ActiveRoundTripRuntime: Send {
    fn stop(&mut self) -> Result<(), RoundTripRuntimeError>;
    fn is_finished(&self) -> bool;
}

pub trait RoundTripRunner: Send + Sync {
    fn start(
        &self,
        snapshot: RuntimeSnapshot,
        session_id: Uuid,
        progress: RoundTripProgress,
        lease: AudioOperationLease,
    ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError>;
}

struct ProgressState {
    session: Mutex<RoundTripSelfTest>,
    preconditions: Mutex<Option<RoundTripPreconditions>>,
    store: RuntimeStore,
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
    runner: Arc<dyn RoundTripRunner>,
    gate: AudioOperationGate,
    progress: RoundTripProgress,
    active: Mutex<Option<Box<dyn ActiveRoundTripRuntime>>>,
}

impl RoundTripRuntimeHandle {
    pub fn new(
        store: RuntimeStore,
        runner: Arc<dyn RoundTripRunner>,
        gate: AudioOperationGate,
    ) -> Self {
        let handle = Self {
            runner,
            gate,
            progress: RoundTripProgress {
                state: Arc::new(ProgressState {
                    session: Mutex::new(RoundTripSelfTest::default()),
                    preconditions: Mutex::new(None),
                    store,
                }),
            },
            active: Mutex::new(None),
        };
        handle.progress.publish();
        handle
    }

    fn reap_finished(active: &mut Option<Box<dyn ActiveRoundTripRuntime>>) {
        if active.as_ref().is_some_and(|runtime| runtime.is_finished()) {
            let stopped = active
                .as_mut()
                .is_some_and(|runtime| runtime.stop().is_ok());
            if stopped {
                *active = None;
            }
        }
    }

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
}

impl RoundTripController for RoundTripRuntimeHandle {
    fn start(&self) -> Result<RoundTripSelfTestState, ControlFailure> {
        let mut active = lock_recovering(&self.active);
        Self::reap_finished(&mut active);
        if active.is_some() {
            return Err(control_failure(
                StatusCode::CONFLICT,
                "self_test_already_running",
            ));
        }
        let snapshot = self.progress.state.store.snapshot();
        let preconditions = round_trip_preconditions(&snapshot);
        *lock_recovering(&self.progress.state.preconditions) = Some(preconditions);
        let now_ms = monotonic_ms();
        let session_id = lock_recovering(&self.progress.state.session)
            .start(preconditions, now_ms)
            .map_err(map_precondition_error)?;
        let lease = self
            .gate
            .acquire_human_round_trip(session_id)
            .map_err(|_| {
                self.progress
                    .fail(session_id, RoundTripErrorCode::RuntimeFailed);
                control_failure(StatusCode::CONFLICT, "audio_operation_busy")
            })?;
        let runtime = self
            .runner
            .start(snapshot, session_id, self.progress.clone(), lease)
            .map_err(|_| {
                self.progress
                    .fail(session_id, RoundTripErrorCode::RuntimeFailed);
                control_failure(StatusCode::SERVICE_UNAVAILABLE, "self_test_start_failed")
            })?;
        *active = Some(runtime);
        let state = self.state();
        self.progress.state.store.set_self_test(state.clone());
        Ok(state)
    }

    fn stop(&self) -> Result<RoundTripSelfTestState, ControlFailure> {
        let mut active = lock_recovering(&self.active);
        let runtime = active
            .as_mut()
            .ok_or_else(|| control_failure(StatusCode::CONFLICT, "self_test_not_running"))?;
        runtime.stop().map_err(|_| {
            control_failure(StatusCode::INTERNAL_SERVER_ERROR, "self_test_stop_failed")
        })?;
        *active = None;
        let session_id = lock_recovering(&self.progress.state.session)
            .status(false)
            .session_id
            .ok_or_else(|| control_failure(StatusCode::CONFLICT, "self_test_not_running"))?;
        lock_recovering(&self.progress.state.session).stop(session_id);
        let state = self.state();
        self.progress.state.store.set_self_test(state.clone());
        Ok(state)
    }
}

fn round_trip_preconditions(snapshot: &RuntimeSnapshot) -> RoundTripPreconditions {
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

fn map_precondition_error(error: RoundTripErrorCode) -> ControlFailure {
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
