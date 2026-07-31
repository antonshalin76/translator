use std::{
    fmt,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use thiserror::Error;
use uuid::Uuid;

pub const CLOSE_ACK_TIMEOUT: Duration = Duration::from_secs(2);
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(1);
pub const MAX_START_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildState {
    Running,
    Reaped,
}

#[derive(Clone, PartialEq, Eq)]
pub struct SidecarLaunch {
    pub generation_id: Uuid,
    pub token: String,
}

impl fmt::Debug for SidecarLaunch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SidecarLaunch")
            .field("generation_id", &self.generation_id)
            .field("token", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseOutcome {
    Acknowledged,
    GenerationRestarted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SupervisorError {
    #[error("sidecar is not ready")]
    NotReady,
    #[error("unknown sidecar session")]
    UnknownSession,
    #[error("sidecar start failed")]
    StartFailed,
    #[error("sidecar readiness probe timed out")]
    ReadinessTimeout,
    #[error("sidecar readiness probe failed")]
    ReadinessFailed,
    #[error("sidecar generation did not match")]
    GenerationMismatch,
    #[error("sidecar process could not be killed and reaped")]
    KillAndReapFailed,
    #[error("stale sidecar socket cleanup failed")]
    CleanupFailed,
    #[error("secure random generation failed")]
    RandomnessUnavailable,
}

#[allow(async_fn_in_trait)]
pub trait SidecarRuntime {
    async fn start(&mut self, launch: &SidecarLaunch) -> Result<(), SupervisorError>;
    async fn probe(&mut self, launch: &SidecarLaunch) -> Result<Uuid, SupervisorError>;
    async fn kill_and_reap(&mut self) -> Result<ChildState, SupervisorError>;
    async fn shutdown_and_reap(&mut self) -> Result<ChildState, SupervisorError>;
    async fn remove_stale_socket(&mut self, child_state: ChildState)
    -> Result<(), SupervisorError>;
    async fn wait_before_retry(&mut self, attempt: usize) -> Result<(), SupervisorError>;
    fn poll_child_state(&mut self) -> Result<ChildState, SupervisorError>;
}

#[derive(Clone, Default)]
pub struct SidecarStatus {
    ready: Arc<AtomicBool>,
    active_sessions: Arc<AtomicUsize>,
    close_wait_armed: Arc<AtomicBool>,
}

impl SidecarStatus {
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn active_session_count(&self) -> usize {
        self.active_sessions.load(Ordering::Acquire)
    }

    pub fn close_wait_armed(&self) -> bool {
        self.close_wait_armed.load(Ordering::Acquire)
    }
}

pub struct SidecarSupervisor<R> {
    runtime: R,
    launch: Option<SidecarLaunch>,
    active_sessions: Vec<Uuid>,
    status: SidecarStatus,
}

struct ResetFlag(Arc<AtomicBool>);

impl Drop for ResetFlag {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl<R: SidecarRuntime> SidecarSupervisor<R> {
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            launch: None,
            active_sessions: Vec::new(),
            status: SidecarStatus::default(),
        }
    }

    pub async fn start(&mut self) -> Result<(), SupervisorError> {
        self.set_unready();
        if self.launch.is_none() {
            self.launch = Some(new_launch()?);
        }
        let mut last_error = SupervisorError::StartFailed;
        for attempt in 0..MAX_START_ATTEMPTS {
            let launch = self
                .launch
                .clone()
                .expect("launch is initialized before sidecar startup");
            match self.runtime.start(&launch).await {
                Err(_) => last_error = SupervisorError::StartFailed,
                Ok(()) => {
                    let probe =
                        tokio::time::timeout(PROBE_TIMEOUT, self.runtime.probe(&launch)).await;
                    let result = match probe {
                        Err(_) => Err(SupervisorError::ReadinessTimeout),
                        Ok(Err(_)) => Err(SupervisorError::ReadinessFailed),
                        Ok(Ok(observed)) if observed != launch.generation_id => {
                            Err(SupervisorError::GenerationMismatch)
                        }
                        Ok(Ok(_)) => Ok(()),
                    };
                    match result {
                        Ok(()) => {
                            self.status.ready.store(true, Ordering::Release);
                            return Ok(());
                        }
                        Err(error) => {
                            last_error = error;
                            self.reap_and_cleanup().await?;
                        }
                    }
                }
            }
            if attempt + 1 < MAX_START_ATTEMPTS {
                self.launch = Some(new_launch()?);
                self.runtime.wait_before_retry(attempt + 1).await?;
            }
        }
        Err(last_error)
    }

    pub fn is_ready(&mut self) -> bool {
        self.refresh_liveness();
        self.status.is_ready()
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn launch(&self) -> Option<&SidecarLaunch> {
        self.launch.as_ref()
    }

    pub fn status_handle(&self) -> SidecarStatus {
        self.status.clone()
    }

    pub fn register_session(&mut self, session_id: Uuid) -> Result<(), SupervisorError> {
        if !self.is_ready() {
            return Err(SupervisorError::NotReady);
        }
        if !self.active_sessions.contains(&session_id) {
            self.active_sessions.push(session_id);
            self.sync_active_count();
        }
        Ok(())
    }

    pub fn active_sessions(&self) -> &[Uuid] {
        &self.active_sessions
    }

    pub async fn close_session<F>(
        &mut self,
        session_id: Uuid,
        acknowledgement: F,
    ) -> Result<CloseOutcome, SupervisorError>
    where
        F: Future<Output = ()>,
    {
        if !self.is_ready() {
            return Err(SupervisorError::NotReady);
        }
        if !self.active_sessions.contains(&session_id) {
            return Err(SupervisorError::UnknownSession);
        }

        let close_wait = Arc::clone(&self.status.close_wait_armed);
        close_wait.store(true, Ordering::Release);
        let close_wait_guard = ResetFlag(close_wait);
        let acknowledged = tokio::time::timeout(CLOSE_ACK_TIMEOUT, acknowledgement)
            .await
            .is_ok();
        drop(close_wait_guard);

        if acknowledged {
            self.active_sessions
                .retain(|candidate| *candidate != session_id);
            self.sync_active_count();
            return Ok(CloseOutcome::Acknowledged);
        }

        self.set_unready();
        self.reap_and_cleanup().await?;
        self.launch = Some(new_launch()?);
        self.start().await?;
        Ok(CloseOutcome::GenerationRestarted)
    }

    pub async fn shutdown(&mut self) -> Result<(), SupervisorError> {
        self.set_unready();
        self.status.close_wait_armed.store(false, Ordering::Release);
        let child_state = self
            .runtime
            .shutdown_and_reap()
            .await
            .map_err(|_| SupervisorError::KillAndReapFailed)?;
        self.cleanup_reaped(child_state).await
    }

    async fn reap_and_cleanup(&mut self) -> Result<(), SupervisorError> {
        let child_state = self
            .runtime
            .kill_and_reap()
            .await
            .map_err(|_| SupervisorError::KillAndReapFailed)?;
        self.cleanup_reaped(child_state).await
    }

    async fn cleanup_reaped(&mut self, child_state: ChildState) -> Result<(), SupervisorError> {
        if child_state != ChildState::Reaped {
            return Err(SupervisorError::KillAndReapFailed);
        }
        self.runtime
            .remove_stale_socket(child_state)
            .await
            .map_err(|_| SupervisorError::CleanupFailed)
    }

    fn set_unready(&mut self) {
        self.status.ready.store(false, Ordering::Release);
        self.active_sessions.clear();
        self.sync_active_count();
    }

    fn refresh_liveness(&mut self) {
        if self.status.is_ready() && self.runtime.poll_child_state() != Ok(ChildState::Running) {
            self.set_unready();
        }
    }

    fn sync_active_count(&self) {
        self.status
            .active_sessions
            .store(self.active_sessions.len(), Ordering::Release);
    }
}

fn new_launch() -> Result<SidecarLaunch, SupervisorError> {
    let mut random = [0_u8; 32];
    getrandom::fill(&mut random).map_err(|_| SupervisorError::RandomnessUnavailable)?;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut token = String::with_capacity(64);
    for byte in random {
        token.push(char::from(HEX[usize::from(byte >> 4)]));
        token.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    Ok(SidecarLaunch {
        generation_id: Uuid::new_v4(),
        token,
    })
}
