use std::{
    future::pending,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use tokio::sync::Notify;
use translator_daemon::{
    CLOSE_ACK_TIMEOUT, ChildState, CloseOutcome, MAX_START_ATTEMPTS, PROBE_TIMEOUT, SidecarLaunch,
    SidecarRuntime, SidecarSupervisor, SupervisorError,
};
use uuid::Uuid;

#[derive(Clone)]
enum ProbeBehavior {
    Matching,
    Wrong(Uuid),
    Pending(Arc<AtomicBool>),
    Gated {
        gate: Arc<Notify>,
        started: Arc<AtomicBool>,
    },
}

struct FakeRuntime {
    events: Vec<String>,
    launches: Vec<SidecarLaunch>,
    probes: Vec<ProbeBehavior>,
    kill_failure: bool,
    kill_gate: Option<Arc<Notify>>,
    start_results: Vec<bool>,
    retry_gate: Option<Arc<Notify>>,
    retry_started: Option<Arc<AtomicBool>>,
    start_count: Arc<AtomicUsize>,
    child_running: Arc<AtomicBool>,
    poll_failure: Arc<AtomicBool>,
}

impl Default for FakeRuntime {
    fn default() -> Self {
        Self {
            events: vec![],
            launches: vec![],
            probes: vec![ProbeBehavior::Matching; MAX_START_ATTEMPTS * 2],
            kill_failure: false,
            kill_gate: None,
            start_results: vec![true; MAX_START_ATTEMPTS * 2],
            retry_gate: None,
            retry_started: None,
            start_count: Arc::new(AtomicUsize::new(0)),
            child_running: Arc::new(AtomicBool::new(false)),
            poll_failure: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl SidecarRuntime for FakeRuntime {
    async fn start(&mut self, launch: &SidecarLaunch) -> Result<(), SupervisorError> {
        self.start_count.fetch_add(1, Ordering::AcqRel);
        self.launches.push(launch.clone());
        self.events.push(format!("start:{}", launch.generation_id));
        if self.start_results.remove(0) {
            self.child_running.store(true, Ordering::Release);
            Ok(())
        } else {
            Err(SupervisorError::StartFailed)
        }
    }

    async fn probe(&mut self, launch: &SidecarLaunch) -> Result<Uuid, SupervisorError> {
        self.events.push(format!("probe:{}", launch.generation_id));
        match self.probes.remove(0) {
            ProbeBehavior::Matching => Ok(launch.generation_id),
            ProbeBehavior::Wrong(observed) => Ok(observed),
            ProbeBehavior::Pending(started) => {
                started.store(true, Ordering::Release);
                pending().await
            }
            ProbeBehavior::Gated { gate, started } => {
                started.store(true, Ordering::Release);
                gate.notified().await;
                Ok(launch.generation_id)
            }
        }
    }

    async fn kill_and_reap(&mut self) -> Result<ChildState, SupervisorError> {
        self.events.push("kill_and_reap".into());
        if let Some(gate) = &self.kill_gate {
            gate.notified().await;
        }
        if self.kill_failure {
            Err(SupervisorError::KillAndReapFailed)
        } else {
            self.child_running.store(false, Ordering::Release);
            Ok(ChildState::Reaped)
        }
    }

    async fn shutdown_and_reap(&mut self) -> Result<ChildState, SupervisorError> {
        self.events.push("shutdown_and_reap".into());
        self.child_running.store(false, Ordering::Release);
        Ok(ChildState::Reaped)
    }

    fn poll_child_state(&mut self) -> Result<ChildState, SupervisorError> {
        if self.poll_failure.load(Ordering::Acquire) {
            return Err(SupervisorError::ReadinessFailed);
        }
        Ok(if self.child_running.load(Ordering::Acquire) {
            ChildState::Running
        } else {
            ChildState::Reaped
        })
    }

    async fn remove_stale_socket(
        &mut self,
        child_state: ChildState,
    ) -> Result<(), SupervisorError> {
        assert_eq!(child_state, ChildState::Reaped);
        self.events.push("remove_stale_socket".into());
        Ok(())
    }

    async fn wait_before_retry(&mut self, attempt: usize) -> Result<(), SupervisorError> {
        self.events.push(format!("backoff:{attempt}"));
        if let Some(started) = &self.retry_started {
            started.store(true, Ordering::Release);
        }
        if let Some(gate) = &self.retry_gate {
            gate.notified().await;
        }
        Ok(())
    }
}

async fn wait_until(mut condition: impl FnMut() -> bool) {
    for _ in 0..100 {
        if condition() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("condition was not reached within 100 scheduler yields");
}

#[tokio::test(start_paused = true)]
async fn probe_requires_matching_generation_at_exact_one_second_deadline() {
    assert_eq!(PROBE_TIMEOUT, Duration::from_secs(1));
    let probe_started = Arc::new(AtomicBool::new(false));
    let pending_runtime = FakeRuntime {
        probes: vec![
            ProbeBehavior::Pending(probe_started.clone()),
            ProbeBehavior::Matching,
        ],
        start_results: vec![true, true],
        ..FakeRuntime::default()
    };
    let pending = tokio::spawn(async move {
        let mut supervisor = SidecarSupervisor::new(pending_runtime);
        let result = supervisor.start().await;
        (supervisor, result)
    });
    wait_until(|| probe_started.load(Ordering::Acquire)).await;
    tokio::time::advance(Duration::from_millis(999)).await;
    tokio::task::yield_now().await;
    assert!(!pending.is_finished());
    tokio::time::advance(Duration::from_millis(1)).await;
    wait_until(|| pending.is_finished()).await;
    let (pending_supervisor, result) = pending.await.unwrap();
    result.unwrap();
    assert!(pending_supervisor.status_handle().is_ready());
    assert_eq!(pending_supervisor.runtime().launches.len(), 2);
    assert_ne!(
        pending_supervisor.runtime().launches[0].generation_id,
        pending_supervisor.runtime().launches[1].generation_id
    );
    assert_ne!(
        pending_supervisor.runtime().launches[0].token,
        pending_supervisor.runtime().launches[1].token
    );
    assert_eq!(
        pending_supervisor
            .runtime()
            .events
            .iter()
            .filter(|event| event.as_str() == "kill_and_reap")
            .count(),
        1
    );
    assert_eq!(
        pending_supervisor
            .runtime()
            .events
            .iter()
            .filter(|event| event.as_str() == "remove_stale_socket")
            .count(),
        1
    );
}

#[tokio::test(start_paused = true)]
async fn close_restarts_only_at_exact_two_second_deadline() {
    assert_eq!(CLOSE_ACK_TIMEOUT, Duration::from_secs(2));
    let mut supervisor = SidecarSupervisor::new(FakeRuntime::default());
    supervisor.start().await.unwrap();
    let original = supervisor.launch().unwrap().clone();
    let microphone = Uuid::new_v4();
    let speaker = Uuid::new_v4();
    supervisor.register_session(microphone).unwrap();
    supervisor.register_session(speaker).unwrap();
    let status = supervisor.status_handle();

    let close = tokio::spawn(async move {
        let result = supervisor.close_session(microphone, pending::<()>()).await;
        (supervisor, result)
    });
    wait_until(|| status.close_wait_armed()).await;
    tokio::time::advance(Duration::from_millis(1999)).await;
    tokio::task::yield_now().await;
    assert!(!close.is_finished());
    tokio::time::advance(Duration::from_millis(1)).await;
    tokio::task::yield_now().await;
    assert!(close.is_finished());
    let (mut supervisor, result) = close.await.unwrap();

    assert_eq!(result.unwrap(), CloseOutcome::GenerationRestarted);
    assert!(supervisor.is_ready());
    assert!(supervisor.active_sessions().is_empty());
    let restarted = supervisor.launch().unwrap();
    assert_ne!(restarted.generation_id, original.generation_id);
    assert_ne!(restarted.token, original.token);
    assert_eq!(restarted.token.len(), 64);
    assert!(
        restarted
            .token
            .bytes()
            .all(|value| value.is_ascii_hexdigit() && !value.is_ascii_uppercase())
    );
}

#[tokio::test(start_paused = true)]
async fn supervisor_stays_unready_through_reap_and_restart_probe() {
    let kill_gate = Arc::new(Notify::new());
    let probe_gate = Arc::new(Notify::new());
    let restart_probe_started = Arc::new(AtomicBool::new(false));
    let runtime = FakeRuntime {
        probes: vec![
            ProbeBehavior::Matching,
            ProbeBehavior::Gated {
                gate: probe_gate.clone(),
                started: restart_probe_started.clone(),
            },
        ],
        kill_gate: Some(kill_gate.clone()),
        ..FakeRuntime::default()
    };
    let mut supervisor = SidecarSupervisor::new(runtime);
    supervisor.start().await.unwrap();
    let session_id = Uuid::new_v4();
    supervisor.register_session(session_id).unwrap();
    let status = supervisor.status_handle();
    let close = tokio::spawn(async move {
        let result = supervisor.close_session(session_id, pending::<()>()).await;
        (supervisor, result)
    });
    wait_until(|| status.close_wait_armed()).await;
    tokio::time::advance(CLOSE_ACK_TIMEOUT).await;
    tokio::task::yield_now().await;
    assert!(!status.is_ready());
    assert_eq!(status.active_session_count(), 0);
    kill_gate.notify_one();
    wait_until(|| restart_probe_started.load(Ordering::Acquire)).await;
    assert!(!status.is_ready());
    assert_eq!(status.active_session_count(), 0);
    probe_gate.notify_one();
    let (mut supervisor, result) = close.await.unwrap();
    assert_eq!(result.unwrap(), CloseOutcome::GenerationRestarted);
    assert!(supervisor.is_ready());
}

#[tokio::test(start_paused = true)]
async fn unreaped_child_prevents_cleanup_token_rotation_and_restart() {
    let runtime = FakeRuntime {
        kill_failure: true,
        ..FakeRuntime::default()
    };
    let mut supervisor = SidecarSupervisor::new(runtime);
    supervisor.start().await.unwrap();
    let original = supervisor.launch().unwrap().clone();
    let session_id = Uuid::new_v4();
    supervisor.register_session(session_id).unwrap();
    let result = supervisor.close_session(session_id, pending::<()>()).await;
    assert_eq!(result.unwrap_err(), SupervisorError::KillAndReapFailed);
    assert!(!supervisor.is_ready());
    assert_eq!(supervisor.status_handle().active_session_count(), 0);
    assert!(supervisor.active_sessions().is_empty());
    assert_eq!(supervisor.launch().unwrap(), &original);
    assert_eq!(
        supervisor.runtime().events,
        vec![
            format!("start:{}", original.generation_id),
            format!("probe:{}", original.generation_id),
            "kill_and_reap".into(),
        ]
    );
}

#[tokio::test]
async fn start_retries_are_bounded_and_backed_off() {
    const { assert!(MAX_START_ATTEMPTS >= 2) };
    let runtime = FakeRuntime {
        start_results: vec![false; MAX_START_ATTEMPTS],
        ..FakeRuntime::default()
    };
    let mut supervisor = SidecarSupervisor::new(runtime);

    assert_eq!(
        supervisor.start().await.unwrap_err(),
        SupervisorError::StartFailed
    );
    assert!(!supervisor.is_ready());
    let starts = supervisor
        .runtime()
        .events
        .iter()
        .filter(|event| event.starts_with("start:"))
        .count();
    let backoffs = supervisor
        .runtime()
        .events
        .iter()
        .filter(|event| event.starts_with("backoff:"))
        .count();
    assert_eq!(starts, MAX_START_ATTEMPTS);
    assert_eq!(backoffs, MAX_START_ATTEMPTS - 1);
}

#[tokio::test]
async fn probe_failure_retries_the_whole_generation_with_backoff() {
    let wrong_generation = Uuid::new_v4();
    let runtime = FakeRuntime {
        probes: vec![
            ProbeBehavior::Wrong(wrong_generation),
            ProbeBehavior::Matching,
        ],
        start_results: vec![true, true],
        ..FakeRuntime::default()
    };
    let mut supervisor = SidecarSupervisor::new(runtime);

    supervisor.start().await.unwrap();
    let final_generation = supervisor.launch().unwrap().generation_id;
    let final_token = supervisor.launch().unwrap().token.clone();
    let first_generation = supervisor.runtime().events[0]
        .strip_prefix("start:")
        .unwrap()
        .parse::<Uuid>()
        .unwrap();
    assert_ne!(first_generation, final_generation);
    assert_ne!(supervisor.runtime().launches[0].token, final_token);
    assert_eq!(
        supervisor.runtime().events,
        vec![
            format!("start:{first_generation}"),
            format!("probe:{first_generation}"),
            "kill_and_reap".into(),
            "remove_stale_socket".into(),
            "backoff:1".into(),
            format!("start:{final_generation}"),
            format!("probe:{final_generation}"),
        ]
    );
}

#[tokio::test]
async fn unexpected_child_exit_revokes_readiness_and_session_registration() {
    let child_running = Arc::new(AtomicBool::new(false));
    let runtime = FakeRuntime {
        child_running: child_running.clone(),
        ..FakeRuntime::default()
    };
    let mut supervisor = SidecarSupervisor::new(runtime);
    supervisor.start().await.unwrap();
    let existing = Uuid::new_v4();
    supervisor.register_session(existing).unwrap();

    child_running.store(false, Ordering::Release);

    assert_eq!(
        supervisor.register_session(Uuid::new_v4()).unwrap_err(),
        SupervisorError::NotReady
    );
    assert!(!supervisor.is_ready());
    assert_eq!(supervisor.status_handle().active_session_count(), 0);
    assert!(supervisor.active_sessions().is_empty());
}

#[tokio::test]
async fn every_probe_mismatch_is_bounded_and_status_stays_unready_during_retry() {
    let retry_gate = Arc::new(Notify::new());
    let retry_started = Arc::new(AtomicBool::new(false));
    let runtime = FakeRuntime {
        probes: (0..MAX_START_ATTEMPTS)
            .map(|_| ProbeBehavior::Wrong(Uuid::new_v4()))
            .collect(),
        start_results: vec![true; MAX_START_ATTEMPTS],
        retry_gate: Some(retry_gate.clone()),
        retry_started: Some(retry_started.clone()),
        ..FakeRuntime::default()
    };
    let mut supervisor = SidecarSupervisor::new(runtime);
    let status = supervisor.status_handle();
    let task = tokio::spawn(async move {
        let result = supervisor.start().await;
        (supervisor, result)
    });
    for _ in 1..MAX_START_ATTEMPTS {
        wait_until(|| retry_started.swap(false, Ordering::AcqRel)).await;
        assert!(!status.is_ready());
        assert_eq!(status.active_session_count(), 0);
        retry_gate.notify_one();
    }
    let (supervisor, result) = task.await.unwrap();
    assert_eq!(result.unwrap_err(), SupervisorError::GenerationMismatch);
    assert!(!status.is_ready());
    assert_eq!(supervisor.runtime().launches.len(), MAX_START_ATTEMPTS);
    let generations = supervisor
        .runtime()
        .launches
        .iter()
        .map(|launch| launch.generation_id)
        .collect::<std::collections::HashSet<_>>();
    let tokens = supervisor
        .runtime()
        .launches
        .iter()
        .map(|launch| launch.token.as_str())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(generations.len(), MAX_START_ATTEMPTS);
    assert_eq!(tokens.len(), MAX_START_ATTEMPTS);
    assert_eq!(
        supervisor
            .runtime()
            .events
            .iter()
            .filter(|event| event.as_str() == "kill_and_reap")
            .count(),
        MAX_START_ATTEMPTS
    );
    assert_eq!(
        supervisor
            .runtime()
            .events
            .iter()
            .filter(|event| event.as_str() == "remove_stale_socket")
            .count(),
        MAX_START_ATTEMPTS
    );
}

#[tokio::test]
async fn child_state_poll_failure_revokes_readiness_fail_closed() {
    let poll_failure = Arc::new(AtomicBool::new(false));
    let runtime = FakeRuntime {
        poll_failure: poll_failure.clone(),
        ..FakeRuntime::default()
    };
    let mut supervisor = SidecarSupervisor::new(runtime);
    supervisor.start().await.unwrap();
    let existing = Uuid::new_v4();
    supervisor.register_session(existing).unwrap();
    poll_failure.store(true, Ordering::Release);

    assert_eq!(
        supervisor.register_session(Uuid::new_v4()).unwrap_err(),
        SupervisorError::NotReady
    );
    assert!(!supervisor.is_ready());
    assert_eq!(supervisor.status_handle().active_session_count(), 0);
    assert!(supervisor.active_sessions().is_empty());
}

#[tokio::test(start_paused = true)]
async fn retry_backoff_gates_initial_and_timeout_restart_attempts() {
    let initial_gate = Arc::new(Notify::new());
    let initial_started = Arc::new(AtomicBool::new(false));
    let initial_start_count = Arc::new(AtomicUsize::new(0));
    let initial_runtime = FakeRuntime {
        start_results: vec![false, true],
        retry_gate: Some(initial_gate.clone()),
        retry_started: Some(initial_started.clone()),
        start_count: initial_start_count.clone(),
        ..FakeRuntime::default()
    };
    let initial = tokio::spawn(async move {
        let mut supervisor = SidecarSupervisor::new(initial_runtime);
        let result = supervisor.start().await;
        (supervisor, result)
    });
    wait_until(|| initial_started.load(Ordering::Acquire)).await;
    assert!(!initial.is_finished());
    assert_eq!(initial_start_count.load(Ordering::Acquire), 1);
    initial_gate.notify_one();
    let (initial_supervisor, result) = initial.await.unwrap();
    result.unwrap();
    assert_eq!(initial_start_count.load(Ordering::Acquire), 2);
    assert_eq!(
        initial_supervisor
            .runtime()
            .events
            .iter()
            .filter(|event| event.starts_with("start:"))
            .count(),
        2
    );

    let restart_gate = Arc::new(Notify::new());
    let restart_started = Arc::new(AtomicBool::new(false));
    let restart_start_count = Arc::new(AtomicUsize::new(0));
    let restart_runtime = FakeRuntime {
        start_results: vec![true, false, true],
        probes: vec![ProbeBehavior::Matching, ProbeBehavior::Matching],
        retry_gate: Some(restart_gate.clone()),
        retry_started: Some(restart_started.clone()),
        start_count: restart_start_count.clone(),
        ..FakeRuntime::default()
    };
    let mut supervisor = SidecarSupervisor::new(restart_runtime);
    supervisor.start().await.unwrap();
    let session_id = Uuid::new_v4();
    supervisor.register_session(session_id).unwrap();
    let status = supervisor.status_handle();
    let restart = tokio::spawn(async move {
        let result = supervisor.close_session(session_id, pending::<()>()).await;
        (supervisor, result)
    });
    wait_until(|| status.close_wait_armed()).await;
    tokio::time::advance(CLOSE_ACK_TIMEOUT).await;
    wait_until(|| restart_started.load(Ordering::Acquire)).await;
    assert!(!restart.is_finished());
    assert!(!status.is_ready());
    assert_eq!(restart_start_count.load(Ordering::Acquire), 2);
    restart_gate.notify_one();
    let (mut restarted_supervisor, result) = restart.await.unwrap();
    assert_eq!(result.unwrap(), CloseOutcome::GenerationRestarted);
    assert!(restarted_supervisor.is_ready());
    assert_eq!(restart_start_count.load(Ordering::Acquire), 3);
}

#[tokio::test]
async fn acknowledged_close_and_shutdown_keep_cleanup_order() {
    let mut supervisor = SidecarSupervisor::new(FakeRuntime::default());
    supervisor.start().await.unwrap();
    let generation = supervisor.launch().unwrap().generation_id;
    let microphone = Uuid::new_v4();
    let speaker = Uuid::new_v4();
    supervisor.register_session(microphone).unwrap();
    supervisor.register_session(speaker).unwrap();
    assert_eq!(
        supervisor
            .close_session(microphone, async {})
            .await
            .unwrap(),
        CloseOutcome::Acknowledged
    );
    assert_eq!(supervisor.launch().unwrap().generation_id, generation);
    assert_eq!(supervisor.active_sessions(), &[speaker]);

    supervisor.shutdown().await.unwrap();
    assert!(!supervisor.is_ready());
    assert_eq!(
        &supervisor.runtime().events[2..],
        &["shutdown_and_reap", "remove_stale_socket"]
    );
}

#[tokio::test]
async fn cancelled_close_wait_clears_armed_status_without_invalidating_generation() {
    let mut supervisor = SidecarSupervisor::new(FakeRuntime::default());
    supervisor.start().await.unwrap();
    let session_id = Uuid::new_v4();
    supervisor.register_session(session_id).unwrap();
    let status = supervisor.status_handle();

    {
        let mut close = Box::pin(supervisor.close_session(session_id, pending::<()>()));
        assert!(futures_util::poll!(close.as_mut()).is_pending());
        assert!(status.close_wait_armed());
    }

    assert!(!status.close_wait_armed());
    assert!(supervisor.is_ready());
    assert_eq!(supervisor.active_sessions(), &[session_id]);
}

#[test]
fn launch_debug_never_exposes_the_bearer_token() {
    let token = "private-sidecar-token-marker".repeat(3);
    let launch = SidecarLaunch {
        generation_id: Uuid::new_v4(),
        token: token.clone(),
    };
    let debug = format!("{launch:?}");
    assert!(!debug.contains(&token));
    assert!(debug.contains("[REDACTED]"));
}
