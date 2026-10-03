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
    kill_results: Vec<Result<ChildState, SupervisorError>>,
    cleanup_results: Vec<bool>,
    cleanup_gate: Option<Arc<Notify>>,
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
            kill_results: Vec::new(),
            cleanup_results: Vec::new(),
            cleanup_gate: None,
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
        if !self.kill_results.is_empty() {
            let result = self.kill_results.remove(0);
            if result == Ok(ChildState::Reaped) {
                self.child_running.store(false, Ordering::Release);
            }
            return result;
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
        if let Some(gate) = &self.cleanup_gate {
            gate.notified().await;
        }
        if !self.cleanup_results.is_empty() && !self.cleanup_results.remove(0) {
            return Err(SupervisorError::CleanupFailed);
        }
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
    assert!(!supervisor.is_ready());
    supervisor
        .acknowledge_generation_retirement(original.generation_id)
        .unwrap();
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
    let original_generation = supervisor.launch().unwrap().generation_id;
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
    assert!(!supervisor.is_ready());
    supervisor
        .acknowledge_generation_retirement(original_generation)
        .unwrap();
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

#[tokio::test(start_paused = true)]
async fn explicit_restart_accepts_dead_or_unready_generation_without_close_wait() {
    for poll_failed in [false, true] {
        let runtime = FakeRuntime::default();
        let child_running = runtime.child_running.clone();
        let poll_failure = runtime.poll_failure.clone();
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();
        let original = supervisor.launch().unwrap().clone();
        supervisor.register_session(Uuid::new_v4()).unwrap();
        if poll_failed {
            poll_failure.store(true, Ordering::Release);
        } else {
            child_running.store(false, Ordering::Release);
        }
        assert!(!supervisor.is_ready());
        assert!(supervisor.active_sessions().is_empty());
        poll_failure.store(false, Ordering::Release);
        let before = tokio::time::Instant::now();
        supervisor.restart_generation().await.unwrap();
        assert_eq!(before.elapsed(), Duration::ZERO);
        assert!(!supervisor.status_handle().close_wait_armed());
        assert!(!supervisor.is_ready());
        supervisor
            .acknowledge_generation_retirement(original.generation_id)
            .unwrap();
        assert!(supervisor.is_ready());
        assert!(supervisor.active_sessions().is_empty());
        let replacement = supervisor.launch().unwrap();
        assert_ne!(original.generation_id, replacement.generation_id);
        assert!(original.token != replacement.token);
        assert_eq!(
            &supervisor.runtime().events[2..],
            [
                "kill_and_reap".to_owned(),
                "remove_stale_socket".to_owned(),
                format!("start:{}", replacement.generation_id),
                format!("probe:{}", replacement.generation_id),
            ]
        );
        supervisor.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn restart_without_an_owned_generation_has_no_effects() {
    let mut supervisor = SidecarSupervisor::new(FakeRuntime::default());
    let result = supervisor.restart_generation().await;
    let absent = Uuid::new_v4();
    let receipt = supervisor.generation_retirement(absent);
    let ack = supervisor.acknowledge_generation_retirement(absent);
    let effects = supervisor.runtime().events.clone();
    supervisor.shutdown().await.unwrap();
    assert_eq!(result, Err(SupervisorError::NotReady));
    assert!(receipt.is_none());
    assert_eq!(ack, Err(SupervisorError::GenerationMismatch));
    assert!(effects.is_empty());
}

#[tokio::test]
async fn restart_cannot_overwrite_unacknowledged_retirement_evidence() {
    let mut supervisor = SidecarSupervisor::new(FakeRuntime::default());
    supervisor.start().await.unwrap();
    supervisor.restart_generation().await.unwrap();
    let current = supervisor.launch().unwrap().generation_id;
    let before = supervisor.runtime().events.len();
    let result = supervisor.restart_generation().await;
    let after = supervisor.runtime().events.len();
    let observed = supervisor.launch().unwrap().generation_id;
    supervisor.shutdown().await.unwrap();
    assert_eq!(result, Err(SupervisorError::GenerationRetirementPending));
    assert_eq!(after, before);
    assert_eq!(observed, current);
}

#[tokio::test]
async fn acknowledged_retirement_allows_only_the_next_exact_generation() {
    let mut supervisor = SidecarSupervisor::new(FakeRuntime::default());
    supervisor.start().await.unwrap();
    let first = supervisor.launch().unwrap().generation_id;
    supervisor.restart_generation().await.unwrap();
    supervisor.acknowledge_generation_retirement(first).unwrap();
    let second = supervisor.launch().unwrap().generation_id;
    supervisor.restart_generation().await.unwrap();
    let stale = supervisor.acknowledge_generation_retirement(first);
    let receipt = supervisor.generation_retirement(second);
    let third = supervisor.launch().unwrap().generation_id;
    let ready_before = supervisor.is_ready();
    let ack = supervisor.acknowledge_generation_retirement(second);
    let ready_after = supervisor.is_ready();
    supervisor.shutdown().await.unwrap();
    assert_ne!(first, second);
    assert_ne!(second, third);
    assert_eq!(stale, Err(SupervisorError::GenerationMismatch));
    assert_eq!(
        receipt.map(|r| (r.generation_id, r.old_generation_reaped)),
        Some((second, true))
    );
    assert_eq!(ack, Ok(()));
    assert!(!ready_before && ready_after);
}

#[tokio::test]
async fn retirement_blocks_committed_readiness_and_new_session_admission() {
    let mut supervisor = SidecarSupervisor::new(FakeRuntime::default());
    supervisor.start().await.unwrap();
    supervisor.register_session(Uuid::new_v4()).unwrap();
    let status = supervisor.status_handle();
    supervisor.restart_generation().await.unwrap();
    let ready = supervisor.is_ready();
    let projected_ready = status.is_ready();
    let admitted = supervisor.register_session(Uuid::new_v4());
    let sessions = supervisor.active_sessions().len();
    supervisor.shutdown().await.unwrap();
    assert!(
        !ready && !projected_ready,
        "physical replacement is not committed readiness before retirement acknowledgement"
    );
    assert_eq!(admitted, Err(SupervisorError::GenerationRetirementPending));
    assert_eq!(sessions, 0);
}

#[tokio::test(start_paused = true)]
async fn retirement_receipt_survives_cancellation_before_and_after_reap() {
    for phase in 0..3 {
        let gate = Arc::new(Notify::new());
        let runtime = FakeRuntime {
            kill_gate: (phase == 0).then(|| gate.clone()),
            cleanup_gate: (phase == 1).then(|| gate.clone()),
            probes: if phase == 2 {
                vec![
                    ProbeBehavior::Matching,
                    ProbeBehavior::Gated {
                        gate: gate.clone(),
                        started: Arc::new(AtomicBool::new(false)),
                    },
                ]
            } else {
                vec![ProbeBehavior::Matching; 2]
            },
            ..FakeRuntime::default()
        };
        let running = runtime.child_running.clone();
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();
        let old = supervisor.launch().unwrap().generation_id;
        let cancelled =
            tokio::time::timeout(Duration::from_millis(10), supervisor.restart_generation())
                .await
                .is_err();
        let receipt = supervisor.generation_retirement(old);
        let physical_running = running.load(Ordering::Acquire);
        let projected = supervisor.status_handle().is_ready();
        let before = supervisor.runtime().events.len();
        let restart = supervisor.restart_generation().await;
        let start = supervisor.start().await;
        let after = supervisor.runtime().events.len();
        let wrong = Uuid::new_v4();
        let wrong_getter = supervisor.generation_retirement(wrong);
        let wrong_ack = supervisor.acknowledge_generation_retirement(wrong);
        let early_ack = (phase == 0).then(|| supervisor.acknowledge_generation_retirement(old));
        let preserved = supervisor.generation_retirement(old);
        gate.notify_one();
        let shutdown = tokio::time::timeout(Duration::from_secs(1), supervisor.shutdown()).await;
        let after_shutdown = supervisor.generation_retirement(old);
        let acknowledged = supervisor.acknowledge_generation_retirement(old);
        let final_receipt = supervisor.generation_retirement(old);
        let repeated_ack = supervisor.acknowledge_generation_retirement(old);

        assert!(shutdown.unwrap().is_ok());
        assert!(cancelled);
        assert_eq!(
            receipt.map(|r| (r.generation_id, r.old_generation_reaped)),
            Some((old, phase != 0))
        );
        assert_eq!(physical_running, phase != 1);
        assert!(!projected);
        assert_eq!(restart, Err(SupervisorError::GenerationRetirementPending));
        assert_eq!(start, Err(SupervisorError::GenerationRetirementPending));
        assert_eq!(after, before);
        assert!(wrong_getter.is_none());
        assert_eq!(wrong_ack, Err(SupervisorError::GenerationMismatch));
        if let Some(early) = early_ack {
            assert_eq!(early, Err(SupervisorError::GenerationRetirementPending));
        }
        assert_eq!(preserved, receipt);
        assert_eq!(
            after_shutdown.map(|r| (r.generation_id, r.old_generation_reaped)),
            Some((old, true))
        );
        assert_eq!(acknowledged, Ok(()));
        assert!(final_receipt.is_none());
        assert_eq!(repeated_ack, Err(SupervisorError::GenerationMismatch));
        assert!(!supervisor.is_ready());
        assert!(!running.load(Ordering::Acquire));
    }
}

#[tokio::test]
async fn retirement_reap_fact_is_independent_of_socket_cleanup_and_new_start_failure() {
    for phase in 0..5 {
        let runtime = FakeRuntime {
            probes: if phase == 4 {
                std::iter::once(ProbeBehavior::Matching)
                    .chain((0..MAX_START_ATTEMPTS).map(|_| ProbeBehavior::Wrong(Uuid::new_v4())))
                    .collect()
            } else {
                vec![ProbeBehavior::Matching; MAX_START_ATTEMPTS + 1]
            },
            kill_results: match phase {
                0 => vec![Err(SupervisorError::KillAndReapFailed)],
                1 => vec![Ok(ChildState::Running)],
                _ => Vec::new(),
            },
            cleanup_results: if phase == 2 { vec![false] } else { Vec::new() },
            start_results: if phase == 3 {
                vec![true, false, false, false]
            } else {
                vec![true; MAX_START_ATTEMPTS + 1]
            },
            ..FakeRuntime::default()
        };
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();
        let old = supervisor.launch().unwrap().generation_id;
        let failure = supervisor.restart_generation().await;
        let receipt = supervisor.generation_retirement(old);
        let projected = supervisor.status_handle().is_ready();
        let shutdown = supervisor.shutdown().await;
        let confirmed = supervisor.generation_retirement(old);
        let acknowledged = supervisor.acknowledge_generation_retirement(old);
        assert_eq!(shutdown, Ok(()));
        assert_eq!(
            failure,
            Err(match phase {
                2 => SupervisorError::CleanupFailed,
                3 => SupervisorError::StartFailed,
                4 => SupervisorError::GenerationMismatch,
                _ => SupervisorError::KillAndReapFailed,
            })
        );
        assert_eq!(
            receipt.map(|r| (r.generation_id, r.old_generation_reaped)),
            Some((old, phase >= 2))
        );
        assert!(!projected);
        assert_eq!(
            confirmed.map(|r| (r.generation_id, r.old_generation_reaped)),
            Some((old, true))
        );
        assert_eq!(acknowledged, Ok(()));
        assert!(!supervisor.is_ready());
    }
}

#[tokio::test]
async fn retirement_acknowledgement_publishes_only_live_replacement_readiness() {
    use std::future::Future;
    for died in [false, true] {
        let gate = Arc::new(Notify::new());
        let started = Arc::new(AtomicBool::new(false));
        let runtime = FakeRuntime {
            probes: vec![
                ProbeBehavior::Matching,
                ProbeBehavior::Gated {
                    gate: gate.clone(),
                    started: started.clone(),
                },
            ],
            ..FakeRuntime::default()
        };
        let running = runtime.child_running.clone();
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();
        let old = supervisor.launch().unwrap().generation_id;
        let status = supervisor.status_handle();
        let mut sampled = Vec::new();
        let result = {
            let restart = supervisor.restart_generation();
            tokio::pin!(restart);
            tokio::time::timeout(
                Duration::from_secs(1),
                std::future::poll_fn(|context| {
                    let result = restart.as_mut().poll(context);
                    sampled.push(status.is_ready());
                    if started.load(Ordering::Acquire) {
                        gate.notify_one();
                    }
                    result
                }),
            )
            .await
        };
        let before = supervisor.generation_retirement(old);
        let rejected = supervisor.register_session(Uuid::new_v4());
        if died {
            running.store(false, Ordering::Release);
        }
        let ack = supervisor.acknowledge_generation_retirement(old);
        let ready = supervisor.is_ready();
        let projected = status.is_ready();
        let registered = supervisor.register_session(Uuid::new_v4());
        let shutdown = supervisor.shutdown().await;
        assert_eq!(shutdown, Ok(()));
        assert_eq!(result.unwrap(), Ok(()));
        assert!(started.load(Ordering::Acquire));
        assert!(sampled.len() >= 2 && sampled.iter().all(|ready| !ready));
        assert_eq!(
            before.map(|r| (r.generation_id, r.old_generation_reaped)),
            Some((old, true))
        );
        assert_eq!(rejected, Err(SupervisorError::GenerationRetirementPending));
        assert_eq!(ack, Ok(()));
        assert_eq!((ready, projected), (!died, !died));
        assert_eq!(
            registered,
            if died {
                Err(SupervisorError::NotReady)
            } else {
                Ok(())
            }
        );
        assert!(supervisor.generation_retirement(old).is_none());
    }
}

#[tokio::test]
async fn explicit_restart_retains_old_generation_until_cleanup_retry_succeeds() {
    for failure in 0..3 {
        let runtime = FakeRuntime {
            kill_results: match failure {
                0 => vec![Err(SupervisorError::KillAndReapFailed)],
                1 => vec![Ok(ChildState::Running)],
                _ => Vec::new(),
            },
            cleanup_results: if failure == 2 {
                vec![false]
            } else {
                Vec::new()
            },
            ..FakeRuntime::default()
        };
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();
        let original = supervisor.launch().unwrap().clone();
        supervisor.register_session(Uuid::new_v4()).unwrap();
        let expected = if failure == 2 {
            SupervisorError::CleanupFailed
        } else {
            SupervisorError::KillAndReapFailed
        };
        assert_eq!(supervisor.restart_generation().await.unwrap_err(), expected);
        assert!(!supervisor.is_ready());
        assert_eq!(supervisor.status_handle().active_session_count(), 0);
        assert!(supervisor.active_sessions().is_empty());
        assert_eq!(supervisor.launch().unwrap(), &original);
        assert_eq!(supervisor.runtime().launches.len(), 1);
        let mut expected_events = vec!["kill_and_reap".to_owned()];
        if failure == 2 {
            expected_events.push("remove_stale_socket".to_owned());
        }
        assert_eq!(&supervisor.runtime().events[2..], expected_events);
        assert_eq!(
            supervisor.restart_generation().await,
            Err(SupervisorError::GenerationRetirementPending)
        );
        supervisor.shutdown().await.unwrap();
        supervisor
            .acknowledge_generation_retirement(original.generation_id)
            .unwrap();
        supervisor.restart_generation().await.unwrap();
        supervisor
            .acknowledge_generation_retirement(original.generation_id)
            .unwrap();
        assert!(supervisor.is_ready());
        let replacement = supervisor.launch().unwrap();
        assert_ne!(original.generation_id, replacement.generation_id);
        assert!(original.token != replacement.token);
        let events = &supervisor.runtime().events;
        assert_eq!(
            &events[events.len() - 4..],
            [
                "kill_and_reap".to_owned(),
                "remove_stale_socket".to_owned(),
                format!("start:{}", replacement.generation_id),
                format!("probe:{}", replacement.generation_id),
            ]
        );
        supervisor.shutdown().await.unwrap();
    }
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
    let original_generation = supervisor.launch().unwrap().generation_id;
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
    assert!(!restarted_supervisor.is_ready());
    restarted_supervisor
        .acknowledge_generation_retirement(original_generation)
        .unwrap();
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
