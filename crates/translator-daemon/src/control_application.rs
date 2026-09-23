use std::{
    collections::HashMap,
    future::pending,
    sync::{
        Arc, Mutex as StdMutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use axum::http::StatusCode;
use tokio::task::JoinHandle;
use tokio::{
    sync::{Mutex, OwnedSemaphorePermit, Semaphore, TryAcquireError, mpsc, oneshot, watch},
    time::{Duration, Instant},
};
use translator_core::AudioDirection;

use crate::{
    AdmittedDuplex, AecRuntimeAuthority, AudioMixController, AudioMixPatch, AudioOperationGate,
    AudioOperationLease, DirectionPatch, DirectionRuntimeFailure, DirectionRuntimeStatus,
    DuplexCompletionObserver, DuplexRunner, DuplexRuntimeError, FactsError, ProviderPatch,
    RuntimeFactsSource, RuntimeMutationError, RuntimeSnapshot, RuntimeStore, TranslationMixMode,
    VoiceProfilePatch,
    acoustic_admission::{admit_translation_with_reservation, enabled_directions},
    api::ControlFailure,
    runtime_state::{AudioMixKnowledge, RuntimeStatus},
};

const CONTROL_CAPACITY: usize = 2;
const MAILBOX_CAPACITY: usize = CONTROL_CAPACITY + 1;
const CLEANUP_RETRY_DELAY: Duration = Duration::from_millis(25);
const SHUTDOWN_ADMISSION_WAIT: Duration = Duration::from_millis(25);
const AEC_REINSPECTION_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone)]
pub enum ControlCommand {
    Start,
    Stop,
    PatchDirection(DirectionPatch),
    PatchProvider(ProviderPatch),
    PatchVoice(VoiceProfilePatch),
    PatchAudioMix(AudioMixPatch),
    ReconcileAudio,
    RecoverAudioMix,
}

pub trait RuntimeMaintenance: Send + Sync {
    fn refresh(&self, store: &RuntimeStore) -> Result<(), ControlFailure>;
}

pub struct ControlApplication {
    sender: mpsc::Sender<ActorMessage>,
    _owner: Arc<StdMutex<ControlOwner>>,
    admission: Arc<Semaphore>,
    submission: StdMutex<()>,
    shutdown_gate: Mutex<()>,
    shutdown_response: Mutex<Option<oneshot::Receiver<Result<(), ControlFailure>>>>,
    actor: Mutex<Option<JoinHandle<()>>>,
    actor_result: Mutex<Option<Result<(), ControlFailure>>>,
    closed: AtomicBool,
}

enum ActorMessage {
    Execute {
        command: ControlCommand,
        deadline: Instant,
        response: oneshot::Sender<Result<RuntimeSnapshot, ControlFailure>>,
        _permit: OwnedSemaphorePermit,
    },
    Shutdown {
        deadline: Instant,
        response: oneshot::Sender<Result<(), ControlFailure>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TerminalPhase {
    CleanupStarted,
    Completed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerminalLifecycle {
    generation: u64,
    phase: TerminalPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectionLifecycle {
    generation: u64,
    epoch: u64,
    status: DirectionRuntimeStatus,
    failure: Option<DirectionRuntimeFailure>,
}

#[derive(Debug, Clone, Default)]
struct LifecycleSlots {
    microphone: Option<DirectionLifecycle>,
    speaker: Option<DirectionLifecycle>,
    terminal: Option<TerminalLifecycle>,
}

impl LifecycleSlots {
    #[cfg(test)]
    fn occupied_slot_count(&self) -> usize {
        usize::from(self.microphone.is_some())
            + usize::from(self.speaker.is_some())
            + usize::from(self.terminal.is_some())
    }

    fn direction(&self, direction: AudioDirection) -> Option<DirectionLifecycle> {
        match direction {
            AudioDirection::Microphone => self.microphone,
            AudioDirection::Speaker => self.speaker,
        }
    }

    fn update_direction(
        &mut self,
        direction: AudioDirection,
        candidate: DirectionLifecycle,
    ) -> bool {
        let slot = match direction {
            AudioDirection::Microphone => &mut self.microphone,
            AudioDirection::Speaker => &mut self.speaker,
        };
        if slot.is_some_and(|current| {
            (
                current.generation,
                current.epoch,
                direction_phase(current.status),
            ) >= (
                candidate.generation,
                candidate.epoch,
                direction_phase(candidate.status),
            )
        }) {
            return false;
        }
        *slot = Some(candidate);
        true
    }

    fn update_terminal(&mut self, candidate: TerminalLifecycle) -> bool {
        if self.terminal.is_some_and(|current| {
            (current.generation, current.phase) >= (candidate.generation, candidate.phase)
        }) {
            return false;
        }
        self.terminal = Some(candidate);
        true
    }
}

struct CompletionMailbox {
    sender: watch::Sender<LifecycleSlots>,
}

impl CompletionMailbox {
    fn notify(&self, update: impl FnOnce(&mut LifecycleSlots) -> bool) {
        self.sender.send_if_modified(update);
    }
}

impl DuplexCompletionObserver for CompletionMailbox {
    fn cleanup_started(&self, generation: u64) {
        self.notify(|slots| {
            slots.update_terminal(TerminalLifecycle {
                generation,
                phase: TerminalPhase::CleanupStarted,
            })
        });
    }

    fn completed(&self, generation: u64, _result: Result<(), DuplexRuntimeError>) {
        self.notify(|slots| {
            slots.update_terminal(TerminalLifecycle {
                generation,
                phase: TerminalPhase::Completed,
            })
        });
    }

    fn direction_status_changed(
        &self,
        generation: u64,
        direction: AudioDirection,
        epoch: u64,
        status: DirectionRuntimeStatus,
        failure: Option<DirectionRuntimeFailure>,
    ) {
        self.notify(|slots| {
            slots.update_direction(
                direction,
                DirectionLifecycle {
                    generation,
                    epoch,
                    status,
                    failure,
                },
            )
        });
    }
}

pub struct RuntimeSupervisor {
    runner: Arc<dyn DuplexRunner>,
    gate: AudioOperationGate,
    completion: Arc<dyn DuplexCompletionObserver>,
    next_generation: u64,
    state: SupervisorState,
}

enum SupervisorState {
    Stopped,
    Running(OwnedRuntime),
    CleanupPending(OwnedRuntime),
}

struct OwnedRuntime {
    generation: u64,
    direction_epochs: HashMap<AudioDirection, (u64, u8)>,
    cleanup_deadline: Option<Instant>,
    runtime: Box<dyn crate::ActiveDuplexRuntime>,
    _lease: AudioOperationLease,
}

struct ControlOwner {
    supervisor: RuntimeSupervisor,
    store: RuntimeStore,
    facts: Arc<dyn RuntimeFactsSource>,
    maintenance: Arc<dyn RuntimeMaintenance>,
    audio_mix: Option<Arc<dyn AudioMixController>>,
    aec_authority: Option<AecRuntimeAuthority>,
    aec_runtime_generation: Option<u64>,
    aec_revocation_seen: u64,
    terminal_seen: Option<TerminalLifecycle>,
}

impl ControlApplication {
    pub fn spawn(
        store: RuntimeStore,
        runner: Arc<dyn DuplexRunner>,
        gate: AudioOperationGate,
        facts: Arc<dyn RuntimeFactsSource>,
        maintenance: Arc<dyn RuntimeMaintenance>,
        audio_mix: Option<Arc<dyn AudioMixController>>,
    ) -> Arc<Self> {
        Self::spawn_with_aec_authority(store, runner, gate, facts, maintenance, audio_mix, None)
    }

    pub fn spawn_with_aec_authority(
        store: RuntimeStore,
        runner: Arc<dyn DuplexRunner>,
        gate: AudioOperationGate,
        facts: Arc<dyn RuntimeFactsSource>,
        maintenance: Arc<dyn RuntimeMaintenance>,
        audio_mix: Option<Arc<dyn AudioMixController>>,
        aec_authority: Option<AecRuntimeAuthority>,
    ) -> Arc<Self> {
        let admission = Arc::new(Semaphore::new(CONTROL_CAPACITY));
        let (sender, receiver) = mpsc::channel(MAILBOX_CAPACITY);
        let (lifecycle_sender, lifecycle_receiver) = watch::channel(LifecycleSlots::default());
        let aec_revocations = aec_authority
            .as_ref()
            .map(AecRuntimeAuthority::subscribe_revocations);
        let completion: Arc<dyn DuplexCompletionObserver> = Arc::new(CompletionMailbox {
            sender: lifecycle_sender,
        });
        let owner = Arc::new(StdMutex::new(ControlOwner {
            supervisor: RuntimeSupervisor::new(runner, gate, completion),
            store,
            facts,
            maintenance,
            audio_mix,
            aec_authority,
            aec_runtime_generation: None,
            aec_revocation_seen: 0,
            terminal_seen: None,
        }));
        let actor = tokio::spawn(run_actor(
            receiver,
            lifecycle_receiver,
            aec_revocations,
            owner.clone(),
        ));
        Arc::new(Self {
            sender,
            _owner: owner,
            admission,
            submission: StdMutex::new(()),
            shutdown_gate: Mutex::new(()),
            shutdown_response: Mutex::new(None),
            actor: Mutex::new(Some(actor)),
            actor_result: Mutex::new(None),
            closed: AtomicBool::new(false),
        })
    }

    pub async fn execute(
        &self,
        command: ControlCommand,
    ) -> Result<RuntimeSnapshot, ControlFailure> {
        let (response, result) = oneshot::channel();
        let deadline = Instant::now()
            + if matches!(command, ControlCommand::Stop) {
                crate::RUNTIME_CLEANUP_BUDGET
            } else {
                crate::DIRECTION_CLEANUP_BUDGET
            };
        {
            let _submission = lock_recovering(&self.submission);
            if self.closed.load(Ordering::Acquire) {
                return Err(unavailable());
            }
            let permit = self
                .admission
                .clone()
                .try_acquire_owned()
                .map_err(map_admission_error)?;
            self.sender
                .try_send(ActorMessage::Execute {
                    command,
                    deadline,
                    response,
                    _permit: permit,
                })
                .map_err(|_| unavailable())?;
        }
        result.await.map_err(|_| unavailable())?
    }

    pub async fn shutdown(&self) -> Result<(), ControlFailure> {
        let _shutdown = self.shutdown_gate.lock().await;
        self.closed.store(true, Ordering::Release);
        if let Some(result) = *self.actor_result.lock().await {
            return result;
        }
        let mut response = self.shutdown_response.lock().await;
        if response.is_none() {
            let (reply, result) = oneshot::channel();
            let deadline = Instant::now() + crate::RUNTIME_CLEANUP_BUDGET;
            let permit = match tokio::time::timeout(
                SHUTDOWN_ADMISSION_WAIT,
                self.sender.clone().reserve_owned(),
            )
            .await
            {
                Ok(Ok(permit)) => permit,
                Ok(Err(_)) | Err(_) => {
                    drop(response);
                    return self.reap_finished_actor().await;
                }
            };
            permit.send(ActorMessage::Shutdown {
                deadline,
                response: reply,
            });
            *response = Some(result);
        }
        let result = response
            .as_mut()
            .expect("shutdown response was stored")
            .await;
        response.take();
        drop(response);
        match result {
            Ok(Err(error)) => return Err(error),
            Ok(Ok(())) => {}
            Err(_) => return self.join_actor().await,
        }
        self.join_actor().await
    }

    async fn join_actor(&self) -> Result<(), ControlFailure> {
        let mut actor = self.actor.lock().await;
        let result = if let Some(handle) = actor.as_mut() {
            (&mut *handle).await.map_err(|_| unavailable())
        } else {
            Ok(())
        };
        if actor.is_some() {
            actor.take();
        }
        *self.actor_result.lock().await = Some(result);
        result
    }

    async fn reap_finished_actor(&self) -> Result<(), ControlFailure> {
        let finished = self
            .actor
            .lock()
            .await
            .as_ref()
            .is_some_and(JoinHandle::is_finished);
        if finished {
            self.join_actor().await
        } else {
            Err(unavailable())
        }
    }
}

impl RuntimeSupervisor {
    pub fn new(
        runner: Arc<dyn DuplexRunner>,
        gate: AudioOperationGate,
        completion: Arc<dyn DuplexCompletionObserver>,
    ) -> Self {
        Self {
            runner,
            gate,
            completion,
            next_generation: 0,
            state: SupervisorState::Stopped,
        }
    }

    fn check_start(&self) -> Result<(), ControlFailure> {
        match self.state {
            SupervisorState::Stopped => {}
            SupervisorState::Running(_) => {
                return Err(ControlFailure {
                    status: StatusCode::CONFLICT,
                    code: "translation_already_running",
                });
            }
            SupervisorState::CleanupPending(_) => return Err(cleanup_pending()),
        }
        Ok(())
    }

    fn start(&mut self, admitted: AdmittedDuplex, deadline: Instant) -> Result<(), ControlFailure> {
        self.check_start()?;
        admission_deadline(deadline)?;
        let lease = self.gate.acquire_production().map_err(|_| ControlFailure {
            status: StatusCode::CONFLICT,
            code: "audio_operation_busy",
        })?;
        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or_else(unavailable)?;
        self.next_generation = generation;
        let runtime = match self.runner.start_supervised(
            admitted,
            generation,
            self.completion.clone(),
            deadline,
        ) {
            Ok(runtime) => runtime,
            Err(failure) => {
                let (error, cleanup) = failure.into_parts();
                let Some(runtime) = cleanup else {
                    return Err(map_start_error(error));
                };
                self.state = SupervisorState::CleanupPending(OwnedRuntime {
                    generation,
                    direction_epochs: HashMap::new(),
                    cleanup_deadline: Some(deadline),
                    runtime,
                    _lease: lease,
                });
                return Err(cleanup_pending());
            }
        };
        self.state = SupervisorState::Running(OwnedRuntime {
            generation,
            direction_epochs: HashMap::new(),
            cleanup_deadline: None,
            runtime,
            _lease: lease,
        });
        Ok(())
    }

    fn reconfigure(
        &mut self,
        admitted: AdmittedDuplex,
        deadline: Instant,
    ) -> Result<(), ControlFailure> {
        let SupervisorState::Running(active) = &mut self.state else {
            return match self.state {
                SupervisorState::CleanupPending(_) => Err(cleanup_pending()),
                SupervisorState::Stopped => Err(ControlFailure {
                    status: StatusCode::CONFLICT,
                    code: "translation_not_running",
                }),
                SupervisorState::Running(_) => unreachable!(),
            };
        };
        admission_deadline(deadline)?;
        match active.runtime.reconfigure(admitted, deadline) {
            Ok(()) => Ok(()),
            Err(DuplexRuntimeError::RestoreFailed) => {
                active.cleanup_deadline = Some(deadline);
                self.move_to_cleanup_pending();
                Err(cleanup_pending())
            }
            Err(_) => Err(ControlFailure {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "translation_replacement_failed",
            }),
        }
    }

    fn stop(&mut self, deadline: Instant) -> Result<(), ControlFailure> {
        let active = match &mut self.state {
            SupervisorState::Stopped => return Ok(()),
            SupervisorState::Running(active) | SupervisorState::CleanupPending(active) => active,
        };
        if active.runtime.stop(deadline).is_err() {
            active.cleanup_deadline = Some(deadline);
            self.move_to_cleanup_pending();
            return Err(ControlFailure {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "translation_stop_failed",
            });
        }
        self.state = SupervisorState::Stopped;
        Ok(())
    }

    fn cleanup_started(&mut self, generation: u64) -> Option<RuntimeStatus> {
        let active = match &mut self.state {
            SupervisorState::Running(active) if active.generation == generation => active,
            SupervisorState::CleanupPending(active) if active.generation == generation => {
                return Some(RuntimeStatus::CleanupPending);
            }
            _ => return None,
        };
        active.cleanup_deadline = Some(Instant::now() + crate::RUNTIME_CLEANUP_BUDGET);
        self.move_to_cleanup_pending();
        Some(RuntimeStatus::CleanupPending)
    }

    fn complete(&mut self, generation: u64) -> Option<RuntimeStatus> {
        let matches = matches!(
            &self.state,
            SupervisorState::Running(active) | SupervisorState::CleanupPending(active)
                if active.generation == generation
        );
        if !matches {
            return None;
        }
        let mut active = match std::mem::replace(&mut self.state, SupervisorState::Stopped) {
            SupervisorState::Running(active) | SupervisorState::CleanupPending(active) => active,
            SupervisorState::Stopped => unreachable!(),
        };
        let deadline = active
            .cleanup_deadline
            .unwrap_or_else(|| Instant::now() + crate::RUNTIME_CLEANUP_BUDGET);
        if active.runtime.reap(deadline).is_err() {
            self.state = SupervisorState::CleanupPending(active);
            Some(RuntimeStatus::CleanupPending)
        } else {
            Some(RuntimeStatus::Failed)
        }
    }

    fn accept_direction_status(
        &mut self,
        generation: u64,
        direction: AudioDirection,
        epoch: u64,
        status: DirectionRuntimeStatus,
    ) -> bool {
        let active = match &mut self.state {
            SupervisorState::Running(active) | SupervisorState::CleanupPending(active)
                if active.generation == generation =>
            {
                active
            }
            _ => return false,
        };
        let candidate = (epoch, direction_phase(status));
        let previous = active.direction_epochs.entry(direction).or_default();
        if candidate <= *previous {
            return false;
        }
        *previous = candidate;
        true
    }

    fn move_to_cleanup_pending(&mut self) {
        let state = std::mem::replace(&mut self.state, SupervisorState::Stopped);
        let active = match state {
            SupervisorState::Running(active) | SupervisorState::CleanupPending(active) => active,
            SupervisorState::Stopped => unreachable!(),
        };
        self.state = SupervisorState::CleanupPending(active);
    }

    fn mix_mode(&self) -> TranslationMixMode {
        match self.state {
            SupervisorState::Stopped => TranslationMixMode::Bypass,
            SupervisorState::Running(_) | SupervisorState::CleanupPending(_) => {
                TranslationMixMode::Translating
            }
        }
    }

    fn status(&self) -> RuntimeStatus {
        match self.state {
            SupervisorState::Stopped => RuntimeStatus::Stopped,
            SupervisorState::Running(_) => RuntimeStatus::Running,
            SupervisorState::CleanupPending(_) => RuntimeStatus::CleanupPending,
        }
    }

    fn running_generation(&self) -> Option<u64> {
        match &self.state {
            SupervisorState::Running(active) => Some(active.generation),
            SupervisorState::Stopped | SupervisorState::CleanupPending(_) => None,
        }
    }
}

async fn run_actor(
    mut receiver: mpsc::Receiver<ActorMessage>,
    mut lifecycle: watch::Receiver<LifecycleSlots>,
    mut aec_revocations: Option<watch::Receiver<u64>>,
    owner: Arc<StdMutex<ControlOwner>>,
) {
    let mut lifecycle_open = true;
    let mut aec_interval = tokio::time::interval(AEC_REINSPECTION_INTERVAL);
    aec_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut aec_inspection: Option<AecInspectionTask> = None;
    loop {
        let mut lifecycle_pending = false;
        let mut revocation_pending = None;
        tokio::select! {
            biased;
            changed = lifecycle.changed(), if lifecycle_open => {
                lifecycle_open = changed.is_ok();
                lifecycle_pending = lifecycle_open;
            }
            changed = async {
                match aec_revocations.as_mut() {
                    Some(revocations) => revocations.changed().await,
                    None => pending().await,
                }
            } => {
                if changed.is_ok() {
                    revocation_pending = aec_revocations
                        .as_mut()
                        .map(|revocations| *revocations.borrow_and_update());
                } else {
                    aec_revocations = None;
                }
            }
            message = receiver.recv() => match message {
                Some(ActorMessage::Execute { command, deadline, response, _permit }) => {
                    let owner = owner.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        lock_recovering(&owner).execute(command, deadline)
                    }).await.unwrap_or_else(|_| Err(unavailable()));
                    let _ = response.send(result);
                }
                Some(ActorMessage::Shutdown { deadline, response }) => {
                    let owner = owner.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        lock_recovering(&owner).stop_and_bypass(true, deadline)
                    }).await.unwrap_or_else(|_| Err(unavailable()));
                    let complete = result.is_ok();
                    if complete
                        && let Some(inspection) = aec_inspection.take()
                    {
                        let _ = inspection.handle.await;
                    }
                    let _ = response.send(result);
                    if complete {
                        break;
                    }
                }
                None => {
                    if let Some(inspection) = aec_inspection.take() {
                        let _ = inspection.handle.await;
                    }
                    loop {
                        let owner = owner.clone();
                        let result = tokio::task::spawn_blocking(move || {
                            lock_recovering(&owner).stop_and_bypass(
                                true,
                                Instant::now() + crate::RUNTIME_CLEANUP_BUDGET,
                            )
                        }).await.unwrap_or_else(|_| Err(unavailable()));
                        if result.is_ok() {
                            return;
                        }
                        tokio::time::sleep(CLEANUP_RETRY_DELAY).await;
                    }
                }
            },
            _ = aec_interval.tick(), if aec_revocations.is_some() && aec_inspection.is_none() => {
                let inspection = lock_recovering(&owner).active_aec_inspection();
                if let Some((runtime_generation, authority)) = inspection {
                    let inspector = authority.clone();
                    aec_inspection = Some(AecInspectionTask {
                        runtime_generation,
                        authority,
                        handle: tokio::task::spawn_blocking(move || {
                            inspector.inspect_binding(
                                (Instant::now() + crate::DIRECTION_CLEANUP_BUDGET).into_std(),
                            )
                        }),
                    });
                }
            }
            completed = async {
                let inspection = aec_inspection
                    .as_mut()
                    .expect("inspection branch requires a task");
                (&mut inspection.handle).await
            }, if aec_inspection.is_some() => {
                let task = aec_inspection
                    .take()
                    .expect("inspection branch requires a task");
                let inspected = completed.unwrap_or(Err(crate::AecCoordinatorError::GraphUnavailable));
                lock_recovering(&owner).apply_aec_inspection(AecInspectionResult {
                    runtime_generation: task.runtime_generation,
                    authority: task.authority,
                    inspected,
                });
            }
        }
        if let Some(generation) = revocation_pending {
            let owner = owner.clone();
            if let Ok(Some(error)) = tokio::task::spawn_blocking(move || {
                lock_recovering(&owner).handle_aec_revocation(generation)
            })
            .await
            {
                tracing::warn!(
                    event = "aec_runtime_revocation_stop_failed",
                    code = error.code
                );
            }
        }
        if lifecycle_pending || lifecycle.has_changed().unwrap_or(false) {
            let slots = lifecycle.borrow_and_update().clone();
            let owner = owner.clone();
            if let Ok(Some(error)) =
                tokio::task::spawn_blocking(move || lock_recovering(&owner).finish_lifecycle(slots))
                    .await
            {
                tracing::warn!(
                    event = "translation_terminal_cleanup_failed",
                    code = error.code
                );
            }
        }
    }
}

struct AecInspectionTask {
    runtime_generation: u64,
    authority: AecRuntimeAuthority,
    handle: JoinHandle<Result<crate::AecProofBinding, crate::AecCoordinatorError>>,
}

struct AecInspectionResult {
    runtime_generation: u64,
    authority: AecRuntimeAuthority,
    inspected: Result<crate::AecProofBinding, crate::AecCoordinatorError>,
}

impl ControlOwner {
    fn active_aec_inspection(&mut self) -> Option<(u64, AecRuntimeAuthority)> {
        let generation = self.aec_runtime_generation?;
        if self.supervisor.running_generation() != Some(generation) {
            self.aec_runtime_generation = None;
            return None;
        }
        let Some(authority) = self.aec_authority.as_ref() else {
            self.aec_runtime_generation = None;
            return None;
        };
        Some((generation, authority.clone()))
    }

    fn apply_aec_inspection(&mut self, completed: AecInspectionResult) {
        if self.aec_runtime_generation != Some(completed.runtime_generation)
            || self.supervisor.running_generation() != Some(completed.runtime_generation)
        {
            return;
        }
        let _ = completed.authority.validate_inspection(completed.inspected);
    }

    fn handle_aec_revocation(&mut self, generation: u64) -> Option<ControlFailure> {
        if generation <= self.aec_revocation_seen {
            return None;
        }
        self.aec_revocation_seen = generation;
        self.aec_runtime_generation?;
        self.aec_runtime_generation = None;
        self.stop_and_bypass(false, Instant::now() + crate::RUNTIME_CLEANUP_BUDGET)
            .err()
    }

    fn execute(
        &mut self,
        command: ControlCommand,
        deadline: Instant,
    ) -> Result<RuntimeSnapshot, ControlFailure> {
        let Self {
            supervisor,
            store,
            facts,
            maintenance,
            audio_mix,
            aec_authority,
            aec_runtime_generation,
            aec_revocation_seen: _,
            terminal_seen: _,
        } = self;
        let audio_mix = audio_mix.as_deref();
        match command {
            ControlCommand::Start => {
                supervisor.check_start()?;
                let admitted = admit_candidate(
                    store.snapshot(),
                    facts.as_ref(),
                    aec_authority.as_ref(),
                    deadline,
                )?;
                if audio_mix.is_some()
                    && store.snapshot().audio_mix_knowledge
                        == AudioMixKnowledge::AudioMixStateUnknown
                {
                    return Err(audio_mix_unknown());
                }
                let mut candidate = admitted.snapshot().clone();
                let aec_protected = admitted.requires_aec_authority();
                if let Err(error) = supervisor.start(admitted, deadline) {
                    if supervisor.status() == RuntimeStatus::CleanupPending {
                        commit_projection(store, RuntimeStatus::CleanupPending, None);
                    }
                    return Err(error);
                }
                *aec_runtime_generation = aec_protected
                    .then(|| supervisor.running_generation())
                    .flatten();
                if let Some(audio_mix) = audio_mix
                    && let Err(error) =
                        audio_mix.reconcile_committed(TranslationMixMode::Translating)
                {
                    let cleanup = supervisor.stop(deadline);
                    *aec_runtime_generation = None;
                    let knowledge = (error.code == "audio_mix_state_unknown")
                        .then_some(AudioMixKnowledge::AudioMixStateUnknown);
                    if cleanup.is_err() || knowledge.is_some() {
                        commit_projection(
                            store,
                            if cleanup.is_err() {
                                RuntimeStatus::CleanupPending
                            } else {
                                RuntimeStatus::Failed
                            },
                            knowledge,
                        );
                    }
                    return Err(if cleanup.is_err() {
                        cleanup_pending()
                    } else {
                        error
                    });
                }
                set_status(&mut candidate, RuntimeStatus::Running);
                if audio_mix.is_some() {
                    candidate.audio_mix_knowledge = AudioMixKnowledge::Known;
                }
                store.commit_admitted_control(candidate);
            }
            ControlCommand::Stop => {
                stop_and_bypass(supervisor, store, audio_mix, false, deadline)?;
                *aec_runtime_generation = None;
                maintenance.refresh(store)?;
            }
            ControlCommand::PatchDirection(patch) => {
                apply_candidate(
                    supervisor,
                    store,
                    store.direction_candidate(patch)?,
                    facts.as_ref(),
                    aec_authority.as_ref(),
                    aec_runtime_generation,
                    deadline,
                )?;
            }
            ControlCommand::PatchProvider(patch) => {
                apply_candidate(
                    supervisor,
                    store,
                    store.provider_candidate(patch)?,
                    facts.as_ref(),
                    aec_authority.as_ref(),
                    aec_runtime_generation,
                    deadline,
                )?;
            }
            ControlCommand::PatchVoice(patch) => {
                apply_candidate(
                    supervisor,
                    store,
                    store.voice_candidate(patch)?,
                    facts.as_ref(),
                    aec_authority.as_ref(),
                    aec_runtime_generation,
                    deadline,
                )?;
            }
            ControlCommand::PatchAudioMix(patch) => {
                let mut candidate = store.audio_mix_candidate(patch)?;
                let audio_mix = audio_mix.ok_or_else(audio_mix_unavailable)?;
                if let Err(error) =
                    audio_mix.apply_desired(candidate.audio_mix, supervisor.mix_mode())
                {
                    return Err(handle_mix_failure(supervisor, store, error, deadline));
                }
                candidate.audio_mix_knowledge = AudioMixKnowledge::Known;
                store.commit_control(candidate);
            }
            ControlCommand::ReconcileAudio => {
                maintenance.refresh(store)?;
                let audio_mix = audio_mix.ok_or_else(audio_mix_unavailable)?;
                if let Err(error) = audio_mix.reconcile_committed(supervisor.mix_mode()) {
                    return Err(handle_mix_failure(supervisor, store, error, deadline));
                }
                if store.snapshot().audio_mix_knowledge != AudioMixKnowledge::Known {
                    commit_projection(store, supervisor.status(), Some(AudioMixKnowledge::Known));
                }
            }
            ControlCommand::RecoverAudioMix => {
                if let Err(error) = supervisor.stop(deadline) {
                    commit_projection(store, RuntimeStatus::CleanupPending, None);
                    return Err(error);
                }
                *aec_runtime_generation = None;
                let audio_mix = audio_mix.ok_or_else(audio_mix_unavailable)?;
                if let Err(error) = audio_mix.recover_committed(TranslationMixMode::Bypass) {
                    commit_projection(
                        store,
                        RuntimeStatus::Failed,
                        Some(AudioMixKnowledge::AudioMixStateUnknown),
                    );
                    return Err(error);
                }
                commit_projection(
                    store,
                    RuntimeStatus::Stopped,
                    Some(AudioMixKnowledge::Known),
                );
            }
        }
        Ok(store.snapshot())
    }

    fn finish_lifecycle(&mut self, slots: LifecycleSlots) -> Option<ControlFailure> {
        let mut error = None;
        if let Some(terminal) = slots.terminal
            && self.terminal_seen.is_none_or(|seen| {
                (seen.generation, seen.phase) < (terminal.generation, terminal.phase)
            })
        {
            self.terminal_seen = Some(terminal);
            error = match terminal.phase {
                TerminalPhase::CleanupStarted => {
                    if let Some(status) = self.supervisor.cleanup_started(terminal.generation) {
                        commit_projection(&self.store, status, None);
                    }
                    None
                }
                TerminalPhase::Completed => finish_terminal(
                    &mut self.supervisor,
                    &self.store,
                    self.audio_mix.as_deref(),
                    terminal.generation,
                ),
            };
            if terminal.phase == TerminalPhase::Completed
                && self.aec_runtime_generation == Some(terminal.generation)
            {
                self.aec_runtime_generation = None;
            }
        }
        for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
            let Some(status) = slots.direction(direction) else {
                continue;
            };
            if !self.supervisor.accept_direction_status(
                status.generation,
                direction,
                status.epoch,
                status.status,
            ) {
                continue;
            }
            let mut candidate = self.store.snapshot();
            let state = candidate
                .directions
                .iter_mut()
                .find(|candidate| candidate.direction_id == direction)
                .expect("both runtime directions always exist");
            state.runtime_status = status.status;
            state.runtime_failure = status.failure;
            self.store.commit_control(candidate);
        }
        error
    }

    fn stop_and_bypass(
        &mut self,
        recover_unknown: bool,
        deadline: Instant,
    ) -> Result<(), ControlFailure> {
        stop_and_bypass(
            &mut self.supervisor,
            &self.store,
            self.audio_mix.as_deref(),
            recover_unknown,
            deadline,
        )
    }
}

fn apply_candidate(
    supervisor: &mut RuntimeSupervisor,
    store: &RuntimeStore,
    mut candidate: RuntimeSnapshot,
    facts: &dyn RuntimeFactsSource,
    aec_authority: Option<&AecRuntimeAuthority>,
    aec_runtime_generation: &mut Option<u64>,
    deadline: Instant,
) -> Result<(), ControlFailure> {
    let current = store.snapshot();
    if supervisor.status() == RuntimeStatus::CleanupPending {
        return Err(cleanup_pending());
    }
    if supervisor.status() == RuntimeStatus::Running {
        let admitted = admit_candidate(candidate, facts, aec_authority, deadline)?;
        let aec_protected = admitted.requires_aec_authority();
        candidate = admitted.snapshot().clone();
        if let Err(error) = supervisor.reconfigure(admitted, deadline) {
            if supervisor.status() == RuntimeStatus::CleanupPending {
                commit_projection(store, RuntimeStatus::CleanupPending, None);
            }
            return Err(error);
        }
        *aec_runtime_generation = aec_protected
            .then(|| supervisor.running_generation())
            .flatten();
        let changed = changed_directions(&current, &candidate).collect::<Vec<_>>();
        for direction in changed {
            let state = candidate
                .directions
                .iter_mut()
                .find(|candidate| candidate.direction_id == direction)
                .expect("both runtime directions always exist");
            state.runtime_status = if state.enabled {
                DirectionRuntimeStatus::Running
            } else {
                DirectionRuntimeStatus::Stopped
            };
            state.runtime_failure = None;
        }
        store.commit_admitted_control(candidate);
    } else {
        store.commit_control(candidate);
    }
    Ok(())
}

fn admission_deadline(deadline: Instant) -> Result<(), ControlFailure> {
    if Instant::now() >= deadline {
        Err(FactsError::Expired.translation_failure())
    } else {
        Ok(())
    }
}

fn admit_candidate(
    candidate: RuntimeSnapshot,
    facts: &dyn RuntimeFactsSource,
    aec_authority: Option<&AecRuntimeAuthority>,
    deadline: Instant,
) -> Result<AdmittedDuplex, ControlFailure> {
    admission_deadline(deadline)?;
    enabled_directions(&candidate)?;
    let observed = facts
        .inspect(deadline.into_std())
        .map_err(FactsError::translation_failure)?;
    admission_deadline(deadline)?;
    let reservation = if enabled_directions(&candidate)?.microphone
        && observed.devices.output_mode == translator_audio::OutputMode::OpenSpeaker
    {
        Some(
            aec_authority
                .ok_or(ControlFailure {
                    status: StatusCode::CONFLICT,
                    code: "translation_precondition_failed",
                })?
                .reserve(deadline.into_std())
                .map_err(|_| ControlFailure {
                    status: StatusCode::CONFLICT,
                    code: "translation_precondition_failed",
                })?,
        )
    } else {
        None
    };
    admit_translation_with_reservation(candidate, observed, reservation)
}

fn changed_directions<'a>(
    current: &'a RuntimeSnapshot,
    candidate: &'a RuntimeSnapshot,
) -> impl Iterator<Item = AudioDirection> + 'a {
    [AudioDirection::Microphone, AudioDirection::Speaker]
        .into_iter()
        .filter(|direction| {
            if current.provider_id != candidate.provider_id {
                return true;
            }
            let current = current
                .directions
                .iter()
                .find(|state| state.direction_id == *direction)
                .expect("both runtime directions always exist");
            let candidate = candidate
                .directions
                .iter()
                .find(|state| state.direction_id == *direction)
                .expect("both runtime directions always exist");
            current.source_language != candidate.source_language
                || current.target_language != candidate.target_language
                || current.enabled != candidate.enabled
                || current.voice_profile != candidate.voice_profile
        })
}

fn finish_terminal(
    supervisor: &mut RuntimeSupervisor,
    store: &RuntimeStore,
    audio_mix: Option<&dyn AudioMixController>,
    generation: u64,
) -> Option<ControlFailure> {
    let status = supervisor.complete(generation)?;
    if status == RuntimeStatus::CleanupPending {
        commit_projection(store, status, None);
        return None;
    }
    let error = audio_mix.and_then(|mix| mix.reconcile_committed(TranslationMixMode::Bypass).err());
    commit_projection(
        store,
        RuntimeStatus::Failed,
        error
            .filter(|error| error.code == "audio_mix_state_unknown")
            .map(|_| AudioMixKnowledge::AudioMixStateUnknown)
            .or_else(|| audio_mix.map(|_| AudioMixKnowledge::Known)),
    );
    error
}

fn stop_and_bypass(
    supervisor: &mut RuntimeSupervisor,
    store: &RuntimeStore,
    audio_mix: Option<&dyn AudioMixController>,
    recover_unknown: bool,
    deadline: Instant,
) -> Result<(), ControlFailure> {
    if let Err(error) = supervisor.stop(deadline) {
        commit_projection(store, RuntimeStatus::CleanupPending, None);
        return Err(error);
    }
    let result = audio_mix.map_or(Ok(()), |mix| {
        let result = mix.reconcile_committed(TranslationMixMode::Bypass);
        if recover_unknown
            && result
                .as_ref()
                .is_err_and(|error| error.code == "audio_mix_state_unknown")
        {
            mix.recover_committed(TranslationMixMode::Bypass)
        } else {
            result
        }
    });
    match result {
        Ok(()) => {
            commit_projection(
                store,
                RuntimeStatus::Stopped,
                audio_mix.map(|_| AudioMixKnowledge::Known),
            );
            Ok(())
        }
        Err(error) => {
            commit_projection(
                store,
                RuntimeStatus::Failed,
                (error.code == "audio_mix_state_unknown")
                    .then_some(AudioMixKnowledge::AudioMixStateUnknown),
            );
            Err(error)
        }
    }
}

fn handle_mix_failure(
    supervisor: &mut RuntimeSupervisor,
    store: &RuntimeStore,
    error: ControlFailure,
    deadline: Instant,
) -> ControlFailure {
    if error.code == "audio_mix_state_unknown" {
        let status = if supervisor.stop(deadline).is_ok() {
            RuntimeStatus::Failed
        } else {
            RuntimeStatus::CleanupPending
        };
        commit_projection(store, status, Some(AudioMixKnowledge::AudioMixStateUnknown));
    }
    error
}

fn commit_projection(
    store: &RuntimeStore,
    status: RuntimeStatus,
    knowledge: Option<AudioMixKnowledge>,
) {
    let mut candidate = store.snapshot();
    set_status(&mut candidate, status);
    if let Some(knowledge) = knowledge {
        candidate.audio_mix_knowledge = knowledge;
    }
    store.commit_control(candidate);
}

fn set_status(snapshot: &mut RuntimeSnapshot, status: RuntimeStatus) {
    snapshot.translation_running = status == RuntimeStatus::Running;
    snapshot.runtime_status = status;
    if status == RuntimeStatus::CleanupPending {
        return;
    }
    for direction in &mut snapshot.directions {
        direction.runtime_status = if status == RuntimeStatus::Running && direction.enabled {
            DirectionRuntimeStatus::Running
        } else {
            DirectionRuntimeStatus::Stopped
        };
        direction.runtime_failure = None;
    }
}

const fn direction_phase(status: DirectionRuntimeStatus) -> u8 {
    match status {
        DirectionRuntimeStatus::Recovering => 0,
        DirectionRuntimeStatus::Stopped
        | DirectionRuntimeStatus::Running
        | DirectionRuntimeStatus::Failed => 1,
    }
}

impl From<RuntimeMutationError> for ControlFailure {
    fn from(error: RuntimeMutationError) -> Self {
        match error {
            RuntimeMutationError::InvalidLanguagePair => Self {
                status: StatusCode::BAD_REQUEST,
                code: "invalid_language_pair",
            },
            RuntimeMutationError::VoiceLanguageMismatch => Self {
                status: StatusCode::BAD_REQUEST,
                code: "voice_language_mismatch",
            },
            RuntimeMutationError::VoiceProfileOverrideUnsupported => Self {
                status: StatusCode::BAD_REQUEST,
                code: "voice_profile_override_unsupported",
            },
            RuntimeMutationError::CloudProviderOptInRequired => Self {
                status: StatusCode::BAD_REQUEST,
                code: "cloud_provider_opt_in_required",
            },
            RuntimeMutationError::DebugCaptureUnavailable => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "debug_capture_unavailable",
            },
            RuntimeMutationError::DebugCaptureStopped(_) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "debug_capture_stopped",
            },
            RuntimeMutationError::InvalidAudioMixVolume => Self {
                status: StatusCode::BAD_REQUEST,
                code: "invalid_audio_mix_volume",
            },
        }
    }
}

fn map_admission_error(error: TryAcquireError) -> ControlFailure {
    match error {
        TryAcquireError::Closed => unavailable(),
        TryAcquireError::NoPermits => ControlFailure {
            status: StatusCode::TOO_MANY_REQUESTS,
            code: "translation_control_busy",
        },
    }
}

fn map_start_error(error: DuplexRuntimeError) -> ControlFailure {
    match error {
        DuplexRuntimeError::InvalidConfiguration => ControlFailure {
            status: StatusCode::CONFLICT,
            code: "translation_precondition_failed",
        },
        _ => ControlFailure {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "translation_start_failed",
        },
    }
}

fn lock_recovering<T>(mutex: &StdMutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const fn cleanup_pending() -> ControlFailure {
    ControlFailure {
        status: StatusCode::CONFLICT,
        code: "translation_cleanup_pending",
    }
}

const fn unavailable() -> ControlFailure {
    ControlFailure {
        status: StatusCode::SERVICE_UNAVAILABLE,
        code: "translation_controller_unavailable",
    }
}

const fn audio_mix_unavailable() -> ControlFailure {
    ControlFailure {
        status: StatusCode::SERVICE_UNAVAILABLE,
        code: "audio_mix_controller_unavailable",
    }
}

const fn audio_mix_unknown() -> ControlFailure {
    ControlFailure {
        status: StatusCode::CONFLICT,
        code: "audio_mix_state_unknown",
    }
}

#[cfg(test)]
pub(crate) mod safe_admission_tests {
    use super::*;
    use std::sync::{Condvar, atomic::AtomicUsize};
    use tokio::sync::broadcast;
    use translator_audio::{
        AecCapability, AudioGraphState, DeviceFacts, DeviceHealth, DeviceSelectionState,
        GraphHealth, OutputMode, PhysicalDevice, RouteResolution, RoutingState,
    };

    pub(crate) fn ready_facts() -> crate::RuntimeFacts {
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
        crate::RuntimeFacts {
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
        }
    }

    struct Facts {
        graph_ready: bool,
    }

    impl RuntimeFactsSource for Facts {
        fn inspect(
            &self,
            _deadline: std::time::Instant,
        ) -> Result<crate::RuntimeFacts, FactsError> {
            let mut facts = ready_facts();
            if !self.graph_ready {
                facts.audio_graph.health = GraphHealth::Degraded;
            }
            Ok(facts)
        }
    }

    #[derive(Default)]
    struct Effects {
        maintenance: AtomicUsize,
        starts: AtomicUsize,
        replacements: AtomicUsize,
        stops: AtomicUsize,
    }

    struct Maintenance(Arc<Effects>);

    impl RuntimeMaintenance for Maintenance {
        fn refresh(&self, _store: &RuntimeStore) -> Result<(), ControlFailure> {
            self.0.maintenance.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct Runner(Arc<Effects>);

    impl DuplexRunner for Runner {
        fn start(&self, _admitted: AdmittedDuplex, _deadline: Instant) -> crate::DuplexStartResult {
            self.0.starts.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(Active(self.0.clone())))
        }
    }

    struct Active(Arc<Effects>);

    impl crate::ActiveDuplexRuntime for Active {
        fn reconfigure(
            &mut self,
            _admitted: AdmittedDuplex,
            _deadline: Instant,
        ) -> Result<(), DuplexRuntimeError> {
            self.0.replacements.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn stop(&mut self, _deadline: Instant) -> Result<(), DuplexRuntimeError> {
            self.0.stops.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn owner(
        store: RuntimeStore,
        runner: Arc<dyn DuplexRunner>,
        effects: Arc<Effects>,
    ) -> ControlOwner {
        let (sender, _receiver) = watch::channel(LifecycleSlots::default());
        ControlOwner {
            supervisor: RuntimeSupervisor::new(
                runner,
                AudioOperationGate::new(),
                Arc::new(CompletionMailbox { sender }),
            ),
            store,
            facts: Arc::new(Facts { graph_ready: true }),
            maintenance: Arc::new(Maintenance(effects)),
            audio_mix: None,
            aec_authority: None,
            aec_runtime_generation: None,
            aec_revocation_seen: 0,
            terminal_seen: None,
        }
    }

    fn deadline() -> Instant {
        Instant::now() + crate::DIRECTION_CLEANUP_BUDGET
    }

    struct BlockingInspector {
        entered: Arc<(StdMutex<bool>, Condvar)>,
        release: Arc<(StdMutex<bool>, Condvar)>,
    }

    struct C9S4Clock {
        reads: AtomicUsize,
        expire_after_first_validation: bool,
    }

    impl crate::aec_validation::AecMonotonicClock for C9S4Clock {
        fn now_ns(&self) -> u64 {
            let read = self.reads.fetch_add(1, Ordering::SeqCst);
            let measured_end = 10_000_000_000 + translator_audio::AEC_OBSERVATION_DURATION_NS;
            if self.expire_after_first_validation && read >= 2 {
                measured_end + crate::AEC_PROOF_LIFETIME_NS
            } else {
                measured_end
            }
        }
    }

    fn c9_s4_binding(generation: &str) -> crate::AecProofBinding {
        crate::AecProofBinding {
            audio_server_id: "server".into(),
            source_hardware_id: "physical-mic".into(),
            sink_hardware_id: "physical-speaker".into(),
            source_name: "alsa_input.physical".into(),
            sink_name: "alsa_output.physical".into(),
            source_port: "mic".into(),
            sink_port: "speaker".into(),
            source_channel_gains: vec![65_536],
            sink_channel_gains: vec![32_768],
            source_muted: false,
            sink_muted: false,
            source_geometry: "desk-left".into(),
            sink_geometry: "desk-front".into(),
            aec_module_id: 73,
            aec_source_id: 81,
            aec_sink_id: 82,
            aec_generation: generation.into(),
            aec_config_id: "webrtc".into(),
            vad_config_id: "vad".into(),
            provider_config_id: "local".into(),
        }
    }

    fn c9_s4_publish(
        coordinator: &crate::AecCalibrationCoordinator,
        binding: &crate::AecProofBinding,
    ) {
        use translator_audio::{
            AEC_FIXTURE_DBFS, AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT,
            AEC_OBSERVATION_FRAME_SAMPLES, AEC_POWER_WINDOW_COUNT, AEC_SAMPLES_PER_POWER_WINDOW,
            AecDeviceMetadata, AecObservationEvidence, AecPositiveControl, AecPowerAcquisition,
            AecPowerWindow, AecValidationInput,
        };
        let challenge = coordinator
            .begin_attempt(uuid::Uuid::new_v4(), uuid::Uuid::new_v4(), binding.clone())
            .unwrap();
        let acquisition = |id: &str| AecPowerAcquisition {
            acquisition_id: id.into(),
            samples_per_window: AEC_SAMPLES_PER_POWER_WINDOW,
            powers: vec![1.0; 5],
        };
        let input = AecValidationInput {
            metadata: AecDeviceMetadata {
                source_name: binding.source_name.clone(),
                sink_name: binding.sink_name.clone(),
                source_geometry: binding.source_geometry.clone(),
                sink_geometry: binding.sink_geometry.clone(),
                sink_port: binding.sink_port.clone(),
                sink_volume_percent: 40,
            },
            binding: binding.measurement_binding(),
            fixture_acquisition_id: "fixture".into(),
            raw_baseline: acquisition("raw-baseline"),
            clean_baseline: acquisition("clean-baseline"),
            resolution: acquisition("resolution"),
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
                        clean_power: 3.0,
                        raw_clipped_samples: 0,
                        clean_clipped_samples: 0,
                        fixture_dbfs: AEC_FIXTURE_DBFS,
                    }
                })
                .collect(),
            observation: AecObservationEvidence {
                observer_generation: "observer".into(),
                calibration_attempt_id: challenge.attempt_id().to_string(),
                challenge_id: challenge.challenge_id().to_string(),
                interval_id: challenge.interval_id().to_string(),
                started_monotonic_ns: 10_000_000_000,
                ended_monotonic_ns: 10_000_000_000 + AEC_OBSERVATION_DURATION_NS,
                expected_frames: AEC_OBSERVATION_FRAME_COUNT,
                processed_frames: AEC_OBSERVATION_FRAME_COUNT,
                stream_generation: "stream".into(),
                sample_rate_hz: 16_000,
                channels: 1,
                frame_duration_ms: 20,
                samples_per_frame: AEC_OBSERVATION_FRAME_SAMPLES,
                first_frame_sequence: 0,
                last_frame_sequence: AEC_OBSERVATION_FRAME_COUNT - 1,
                first_capture_monotonic_ns: 10_000_000_000,
                last_capture_monotonic_ns: 10_000_000_000 + AEC_OBSERVATION_DURATION_NS
                    - 20_000_000,
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
                    observer_generation: "observer".into(),
                    calibration_attempt_id: challenge.attempt_id().to_string(),
                    challenge_id: challenge.challenge_id().to_string(),
                    completed_monotonic_ns: 9_000_000_000,
                    speech_started_events: 1,
                    provider_submission_attempts: 1,
                    provider_submissions_accepted: 1,
                    resets: 0,
                    observer_errors: 0,
                },
            },
        };
        coordinator.publish(&challenge, input, true, true).unwrap();
        assert!(matches!(
            coordinator.status(),
            crate::AecProofStatus::Validated { .. }
        ));
    }

    #[derive(Default)]
    struct C9S4Barrier {
        entered: tokio::sync::Notify,
        released: (StdMutex<bool>, Condvar),
    }

    struct C9S4Release(Arc<C9S4Barrier>);

    impl Drop for C9S4Release {
        fn drop(&mut self) {
            *self.0.released.0.lock().unwrap() = true;
            self.0.released.1.notify_all();
        }
    }

    struct C9S4Inspector {
        binding: StdMutex<crate::AecProofBinding>,
        calls: AtomicUsize,
        first_barrier: Option<Arc<C9S4Barrier>>,
    }

    struct C10S2PanicInspector {
        binding: StdMutex<crate::AecProofBinding>,
        entered: Option<Arc<C9S4Barrier>>,
        panic_once: AtomicBool,
        calls: AtomicUsize,
    }

    impl crate::AecProofInspector for C10S2PanicInspector {
        fn inspect_binding(
            &self,
            _: std::time::Instant,
        ) -> Result<crate::AecProofBinding, crate::AecCoordinatorError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.panic_once.swap(false, Ordering::SeqCst) {
                if let Some(barrier) = &self.entered {
                    barrier.entered.notify_one();
                    let released = barrier.released.0.lock().unwrap();
                    let (released, _) = barrier
                        .released
                        .1
                        .wait_timeout_while(released, Duration::from_secs(3), |released| !*released)
                        .unwrap();
                    assert!(*released, "panic inspector was not released");
                }
                panic!("deterministic periodic inspection panic");
            }
            Ok(self.binding.lock().unwrap().clone())
        }
    }

    impl crate::AecProofInspector for C9S4Inspector {
        fn inspect_binding(
            &self,
            _: std::time::Instant,
        ) -> Result<crate::AecProofBinding, crate::AecCoordinatorError> {
            let binding = self.binding.lock().unwrap().clone();
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                if let Some(barrier) = &self.first_barrier {
                    barrier.entered.notify_one();
                    let released = barrier.released.0.lock().unwrap();
                    let (released, _) = barrier
                        .released
                        .1
                        .wait_timeout_while(released, Duration::from_secs(3), |released| !*released)
                        .unwrap();
                    if !*released {
                        return Err(crate::AecCoordinatorError::GraphUnavailable);
                    }
                } else {
                    // Ensure the first blocking read completes after the actor
                    // returns to its select, not inline with spawn_blocking.
                    std::thread::sleep(Duration::from_millis(25));
                }
            }
            Ok(binding)
        }
    }

    fn c9_s4_application(
        coordinator: Arc<crate::AecCalibrationCoordinator>,
        inspector: Arc<C9S4Inspector>,
        effects: Arc<Effects>,
    ) -> Arc<ControlApplication> {
        ControlApplication::spawn_with_aec_authority(
            RuntimeStore::default(),
            Arc::new(Runner(effects.clone())),
            AudioOperationGate::new(),
            Arc::new(Facts { graph_ready: true }),
            Arc::new(Maintenance(effects)),
            None,
            Some(AecRuntimeAuthority::new(coordinator, inspector)),
        )
    }

    fn c10_s2_application(
        coordinator: Arc<crate::AecCalibrationCoordinator>,
        inspector: Arc<dyn crate::AecProofInspector>,
        effects: Arc<Effects>,
    ) -> Arc<ControlApplication> {
        ControlApplication::spawn_with_aec_authority(
            RuntimeStore::default(),
            Arc::new(Runner(effects.clone())),
            AudioOperationGate::new(),
            Arc::new(Facts { graph_ready: true }),
            Arc::new(Maintenance(effects)),
            None,
            Some(AecRuntimeAuthority::new(coordinator, inspector)),
        )
    }

    fn c9_s4_register_running(application: &ControlApplication) -> u64 {
        let mut owner = lock_recovering(&application._owner);
        let generation = owner.supervisor.running_generation().unwrap();
        owner.aec_runtime_generation = Some(generation);
        generation
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn c9_s4a_periodic_inspection_repeats_and_expires_without_actor_stimulus() {
        let coordinator = Arc::new(crate::AecCalibrationCoordinator::with_clock(Arc::new(
            C9S4Clock {
                reads: AtomicUsize::new(0),
                expire_after_first_validation: true,
            },
        )));
        let binding = c9_s4_binding("A");
        c9_s4_publish(&coordinator, &binding);
        let mut revocations = coordinator.subscribe_revocations();
        let inspector = Arc::new(C9S4Inspector {
            binding: StdMutex::new(binding),
            calls: AtomicUsize::new(0),
            first_barrier: None,
        });
        let effects = Arc::new(Effects::default());
        let application =
            c9_s4_application(coordinator.clone(), inspector.clone(), effects.clone());
        application.execute(ControlCommand::Start).await.unwrap();
        c9_s4_register_running(&application);
        let revoked = tokio::time::timeout(Duration::from_secs(1), revocations.changed()).await;
        let calls_before_cleanup = inspector.calls.load(Ordering::SeqCst);
        let stopped = tokio::time::timeout(Duration::from_millis(200), async {
            while application
                ._owner
                .lock()
                .unwrap()
                .store
                .snapshot()
                .runtime_status
                != RuntimeStatus::Stopped
            {
                tokio::task::yield_now().await;
            }
        })
        .await;
        application.shutdown().await.unwrap();
        assert!(
            revoked.is_ok(),
            "a completed read must wake/rearm inspection without an external command; calls={calls_before_cleanup}"
        );
        assert_eq!(calls_before_cleanup, 2);
        assert!(
            stopped.is_ok(),
            "proof expiry must stop its registered runtime"
        );
        assert_eq!(effects.stops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn c9_s4b_stop_is_responsive_but_shutdown_joins_held_inspection() {
        let coordinator = Arc::new(crate::AecCalibrationCoordinator::with_clock(Arc::new(
            C9S4Clock {
                reads: AtomicUsize::new(0),
                expire_after_first_validation: false,
            },
        )));
        let binding = c9_s4_binding("A");
        c9_s4_publish(&coordinator, &binding);
        let barrier = Arc::new(C9S4Barrier::default());
        let release = C9S4Release(barrier.clone());
        let inspector = Arc::new(C9S4Inspector {
            binding: StdMutex::new(binding),
            calls: AtomicUsize::new(0),
            first_barrier: Some(barrier.clone()),
        });
        let application = c9_s4_application(coordinator, inspector, Arc::new(Effects::default()));
        application.execute(ControlCommand::Start).await.unwrap();
        c9_s4_register_running(&application);
        tokio::time::timeout(Duration::from_secs(1), barrier.entered.notified())
            .await
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(1),
            application.execute(ControlCommand::Stop),
        )
        .await
        .expect("Stop must not wait for inspection")
        .unwrap();
        let shutdown_application = application.clone();
        let mut shutdown = tokio::spawn(async move { shutdown_application.shutdown().await });
        let early = tokio::time::timeout(Duration::from_millis(100), &mut shutdown).await;
        drop(release);
        let returned_early = early.is_ok();
        match early {
            Ok(result) => result.unwrap().unwrap(),
            Err(_) => tokio::time::timeout(Duration::from_secs(1), shutdown)
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
        }
        assert!(
            !returned_early,
            "shutdown reported success while a blocking inspection was still owned"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn c9_s4c_stale_inspection_cannot_revoke_replacement_proof_or_runtime() {
        let coordinator = Arc::new(crate::AecCalibrationCoordinator::with_clock(Arc::new(
            C9S4Clock {
                reads: AtomicUsize::new(0),
                expire_after_first_validation: false,
            },
        )));
        let binding = c9_s4_binding("A");
        c9_s4_publish(&coordinator, &binding);
        let mut revocations = coordinator.subscribe_revocations();
        let barrier = Arc::new(C9S4Barrier::default());
        let release = C9S4Release(barrier.clone());
        let inspector = Arc::new(C9S4Inspector {
            binding: StdMutex::new(binding),
            calls: AtomicUsize::new(0),
            first_barrier: Some(barrier.clone()),
        });
        let effects = Arc::new(Effects::default());
        let application =
            c9_s4_application(coordinator.clone(), inspector.clone(), effects.clone());
        application.execute(ControlCommand::Start).await.unwrap();
        let generation_a = c9_s4_register_running(&application);
        tokio::time::timeout(Duration::from_secs(1), barrier.entered.notified())
            .await
            .unwrap();
        application.execute(ControlCommand::Stop).await.unwrap();
        application.execute(ControlCommand::Start).await.unwrap();
        let generation_b = c9_s4_register_running(&application);
        assert_ne!(generation_a, generation_b);
        let binding_b = c9_s4_binding("B");
        c9_s4_publish(&coordinator, &binding_b);
        *inspector.binding.lock().unwrap() = binding_b.clone();
        drop(release);
        let stale_revocation =
            tokio::time::timeout(Duration::from_millis(400), revocations.changed()).await;
        let proof_status = coordinator.status();
        let runtime_generation = lock_recovering(&application._owner)
            .supervisor
            .running_generation();
        let stops_before_cleanup = effects.stops.load(Ordering::SeqCst);
        application.shutdown().await.unwrap();
        assert!(
            stale_revocation.is_err(),
            "generation A inspection revoked generation B proof"
        );
        assert!(matches!(
            proof_status,
            crate::AecProofStatus::Validated { .. }
        ));
        assert_eq!(runtime_generation, Some(generation_b));
        assert_eq!(
            stops_before_cleanup, 1,
            "only explicit Stop of generation A may run"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn c10_s2a_periodic_inspector_panic_revokes_and_stops_registered_runtime_once() {
        let coordinator = Arc::new(crate::AecCalibrationCoordinator::with_clock(Arc::new(
            C9S4Clock {
                reads: AtomicUsize::new(0),
                expire_after_first_validation: false,
            },
        )));
        let binding = c9_s4_binding("A");
        c9_s4_publish(&coordinator, &binding);
        let mut revocations = coordinator.subscribe_revocations();
        let inspector = Arc::new(C10S2PanicInspector {
            binding: StdMutex::new(binding),
            entered: None,
            panic_once: AtomicBool::new(true),
            calls: AtomicUsize::new(0),
        });
        let effects = Arc::new(Effects::default());
        let application = c10_s2_application(coordinator.clone(), inspector, effects.clone());
        application.execute(ControlCommand::Start).await.unwrap();
        c9_s4_register_running(&application);

        tokio::time::timeout(Duration::from_secs(1), revocations.changed())
            .await
            .expect("abnormal inspection completion must revoke the matching proof")
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while application
                ._owner
                .lock()
                .unwrap()
                .store
                .snapshot()
                .runtime_status
                != RuntimeStatus::Stopped
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("proof revocation must stop its registered runtime");
        assert_eq!(effects.stops.load(Ordering::SeqCst), 1);
        assert!(!matches!(
            coordinator.status(),
            crate::AecProofStatus::Validated { .. }
        ));
        application.shutdown().await.unwrap();
        assert_eq!(effects.stops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn c10_s2b_late_inspector_panic_cannot_revoke_replacement_runtime() {
        let coordinator = Arc::new(crate::AecCalibrationCoordinator::with_clock(Arc::new(
            C9S4Clock {
                reads: AtomicUsize::new(0),
                expire_after_first_validation: false,
            },
        )));
        let binding_a = c9_s4_binding("A");
        c9_s4_publish(&coordinator, &binding_a);
        let mut revocations = coordinator.subscribe_revocations();
        let barrier = Arc::new(C9S4Barrier::default());
        let release = C9S4Release(barrier.clone());
        let inspector = Arc::new(C10S2PanicInspector {
            binding: StdMutex::new(binding_a),
            entered: Some(barrier.clone()),
            panic_once: AtomicBool::new(true),
            calls: AtomicUsize::new(0),
        });
        let effects = Arc::new(Effects::default());
        let application =
            c10_s2_application(coordinator.clone(), inspector.clone(), effects.clone());
        application.execute(ControlCommand::Start).await.unwrap();
        let generation_a = c9_s4_register_running(&application);
        tokio::time::timeout(Duration::from_secs(1), barrier.entered.notified())
            .await
            .unwrap();

        application.execute(ControlCommand::Stop).await.unwrap();
        application.execute(ControlCommand::Start).await.unwrap();
        let generation_b = c9_s4_register_running(&application);
        assert_ne!(generation_a, generation_b);
        let binding_b = c9_s4_binding("B");
        c9_s4_publish(&coordinator, &binding_b);
        *inspector.binding.lock().unwrap() = binding_b;
        drop(release);

        tokio::time::timeout(Duration::from_secs(1), async {
            while inspector.calls.load(Ordering::SeqCst) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("runtime B inspection must prove actor processed runtime A's failed join");
        let stale_revocation =
            tokio::time::timeout(Duration::from_millis(300), revocations.changed()).await;
        let proof_status = coordinator.status();
        let runtime_generation = lock_recovering(&application._owner)
            .supervisor
            .running_generation();
        let stops_before_cleanup = effects.stops.load(Ordering::SeqCst);
        application.shutdown().await.unwrap();
        assert!(
            stale_revocation.is_err(),
            "runtime A panic revoked replacement proof B"
        );
        assert!(matches!(
            proof_status,
            crate::AecProofStatus::Validated { .. }
        ));
        assert_eq!(runtime_generation, Some(generation_b));
        assert_eq!(stops_before_cleanup, 1);
    }

    impl crate::AecProofInspector for BlockingInspector {
        fn inspect_binding(
            &self,
            _deadline: std::time::Instant,
        ) -> Result<crate::AecProofBinding, crate::AecCoordinatorError> {
            let (entered_lock, entered_signal) = &*self.entered;
            *entered_lock.lock().unwrap() = true;
            entered_signal.notify_all();
            let (release_lock, release_signal) = &*self.release;
            let mut released = release_lock.lock().unwrap();
            while !*released {
                released = release_signal.wait(released).unwrap();
            }
            Err(crate::AecCoordinatorError::GraphUnavailable)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_periodic_aec_inspection_does_not_starve_stop_command() {
        let effects = Arc::new(Effects::default());
        let entered = Arc::new((StdMutex::new(false), Condvar::new()));
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        let coordinator = Arc::new(crate::AecCalibrationCoordinator::new());
        let authority = AecRuntimeAuthority::new(
            coordinator,
            Arc::new(BlockingInspector {
                entered: entered.clone(),
                release: release.clone(),
            }),
        );
        let application = ControlApplication::spawn_with_aec_authority(
            RuntimeStore::default(),
            Arc::new(Runner(effects.clone())),
            AudioOperationGate::new(),
            Arc::new(Facts { graph_ready: true }),
            Arc::new(Maintenance(effects)),
            None,
            Some(authority),
        );
        application.execute(ControlCommand::Start).await.unwrap();
        {
            let mut owner = lock_recovering(&application._owner);
            owner.aec_runtime_generation = owner.supervisor.running_generation();
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if *entered.0.lock().unwrap() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("periodic inspection must start");

        tokio::time::timeout(
            Duration::from_secs(1),
            application.execute(ControlCommand::Stop),
        )
        .await
        .expect("Stop must not wait for a blocking inspection")
        .unwrap();

        *release.0.lock().unwrap() = true;
        release.1.notify_all();
        application.shutdown().await.unwrap();
    }

    #[test]
    fn aec_revocation_stops_the_registered_runtime_once() {
        let store = RuntimeStore::default();
        let effects = Arc::new(Effects::default());
        let mut owner = owner(
            store.clone(),
            Arc::new(Runner(effects.clone())),
            effects.clone(),
        );
        owner.execute(ControlCommand::Start, deadline()).unwrap();
        owner.aec_runtime_generation = owner.supervisor.running_generation();

        assert!(owner.handle_aec_revocation(7).is_none());
        assert_eq!(owner.supervisor.status(), RuntimeStatus::Stopped);
        assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
        assert_eq!(effects.stops.load(Ordering::SeqCst), 1);

        assert!(owner.handle_aec_revocation(7).is_none());
        assert!(owner.handle_aec_revocation(6).is_none());
        assert_eq!(effects.stops.load(Ordering::SeqCst), 1);
    }

    struct ForbiddenMix;

    impl AudioMixController for ForbiddenMix {
        fn apply_desired(
            &self,
            _: crate::AudioMixState,
            _: TranslationMixMode,
        ) -> Result<(), ControlFailure> {
            panic!("rejected admission must not apply a mix")
        }
        fn reconcile_committed(&self, _: TranslationMixMode) -> Result<(), ControlFailure> {
            panic!("rejected admission must not reconcile a mix")
        }
        fn recover_committed(&self, _: TranslationMixMode) -> Result<(), ControlFailure> {
            panic!("rejected admission must not recover a mix")
        }
    }

    #[test]
    fn safe_both_disabled_precedes_unknown_mix_and_missing_facts() {
        let store = RuntimeStore::default();
        let mut candidate = store.snapshot();
        candidate
            .directions
            .iter_mut()
            .for_each(|direction| direction.enabled = false);
        candidate.audio_mix_knowledge = AudioMixKnowledge::AudioMixStateUnknown;
        store.commit_control(candidate);
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let mut events = store.subscribe().unwrap();
        let effects = Arc::new(Effects::default());
        let observations = Arc::new(Observations::new(Err(FactsError::DiscoveryFailed)));
        let mut owner = owner(
            store.clone(),
            Arc::new(Runner(effects.clone())),
            effects.clone(),
        );
        owner.facts = observations.clone();
        owner.audio_mix = Some(Arc::new(ForbiddenMix));
        let result = owner.execute(ControlCommand::Start, deadline());
        owner.supervisor.stop(deadline()).unwrap();

        assert_eq!(
            result.err().map(|error| (error.status, error.code)),
            Some((StatusCode::CONFLICT, "no_direction_enabled"))
        );
        assert!(observations.deadlines.lock().unwrap().is_empty());
        assert_eq!(effects.maintenance.load(Ordering::SeqCst), 0);
        assert_eq!(effects.starts.load(Ordering::SeqCst), 0);
        assert_eq!(owner.supervisor.next_generation, 0);
        assert_eq!(serde_json::to_value(store.snapshot()).unwrap(), before);
        assert!(matches!(
            events.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }

    struct ExpiringFacts(StdMutex<Vec<std::time::Instant>>);

    impl RuntimeFactsSource for ExpiringFacts {
        fn inspect(&self, deadline: std::time::Instant) -> Result<crate::RuntimeFacts, FactsError> {
            self.0.lock().unwrap().push(deadline);
            std::thread::sleep(deadline.saturating_duration_since(std::time::Instant::now()));
            Ok(ready_facts())
        }
    }

    #[test]
    fn safe_expired_discovery_cannot_acquire_a_generation_or_start_runtime() {
        let store = RuntimeStore::default();
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let mut events = store.subscribe().unwrap();
        let effects = Arc::new(Effects::default());
        let facts = Arc::new(ExpiringFacts(StdMutex::new(Vec::new())));
        let mut owner = owner(
            store.clone(),
            Arc::new(Runner(effects.clone())),
            effects.clone(),
        );
        owner.facts = facts.clone();
        let original_deadline = Instant::now() + Duration::from_millis(20);
        let result = owner.execute(ControlCommand::Start, original_deadline);
        let generation = owner.supervisor.next_generation;
        let gate = owner.supervisor.gate.state();
        owner.supervisor.stop(deadline()).unwrap();

        assert_eq!(
            result.err().map(|error| (error.status, error.code)),
            Some((StatusCode::SERVICE_UNAVAILABLE, "audio_facts_expired"))
        );
        assert_eq!(
            facts.0.lock().unwrap().as_slice(),
            &[original_deadline.into_std()]
        );
        assert_eq!(effects.starts.load(Ordering::SeqCst), 0);
        assert_eq!(effects.maintenance.load(Ordering::SeqCst), 0);
        assert_eq!(generation, 0);
        assert_eq!(gate, crate::AudioOperationState::Idle);
        assert_eq!(serde_json::to_value(store.snapshot()).unwrap(), before);
        assert!(matches!(
            events.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn safe_lifecycle_conflicts_preserve_existing_precedence_without_discovery() {
        let store = RuntimeStore::default();
        let effects = Arc::new(Effects::default());
        let facts = Arc::new(Observations::new(Ok(OutputMode::Headphones)));
        let mut owner = owner(
            store.clone(),
            Arc::new(Runner(effects.clone())),
            effects.clone(),
        );
        owner.facts = facts.clone();
        owner.execute(ControlCommand::Start, deadline()).unwrap();
        *facts.result.lock().unwrap() = Err(FactsError::DiscoveryFailed);
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let mut events = store.subscribe().unwrap();
        let already_running = owner.execute(ControlCommand::Start, deadline());
        owner.supervisor.move_to_cleanup_pending();
        let cleanup_pending = owner.execute(ControlCommand::Start, deadline());
        let generation = owner.supervisor.next_generation;
        owner.supervisor.stop(deadline()).unwrap();

        assert_eq!(
            already_running.err().map(|error| error.code),
            Some("translation_already_running")
        );
        assert_eq!(
            cleanup_pending.err().map(|error| error.code),
            Some("translation_cleanup_pending")
        );
        assert_eq!(facts.deadlines.lock().unwrap().len(), 1);
        assert_eq!(effects.starts.load(Ordering::SeqCst), 1);
        assert_eq!(effects.maintenance.load(Ordering::SeqCst), 0);
        assert_eq!(generation, 1);
        assert_eq!(serde_json::to_value(store.snapshot()).unwrap(), before);
        assert!(matches!(
            events.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }

    struct Observations {
        result: StdMutex<Result<OutputMode, FactsError>>,
        deadlines: StdMutex<Vec<std::time::Instant>>,
    }

    impl Observations {
        fn new(result: Result<OutputMode, FactsError>) -> Self {
            Self {
                result: StdMutex::new(result),
                deadlines: StdMutex::new(Vec::new()),
            }
        }
    }

    impl RuntimeFactsSource for Observations {
        fn inspect(&self, deadline: std::time::Instant) -> Result<crate::RuntimeFacts, FactsError> {
            self.deadlines.lock().unwrap().push(deadline);
            let mode = (*self.result.lock().unwrap())?;
            let mut facts = ready_facts();
            facts.devices.output_mode = mode;
            Ok(facts)
        }
    }

    #[test]
    fn safe_fresh_rejection_preserves_cached_snapshot_and_zero_resource_owners() {
        for (observation, expected_status, expected_code) in [
            (
                Err(FactsError::DiscoveryFailed),
                StatusCode::SERVICE_UNAVAILABLE,
                "audio_facts_unavailable",
            ),
            (
                Err(FactsError::Busy),
                StatusCode::SERVICE_UNAVAILABLE,
                "audio_facts_busy",
            ),
            (
                Err(FactsError::Expired),
                StatusCode::SERVICE_UNAVAILABLE,
                "audio_facts_expired",
            ),
            (
                Err(FactsError::InvalidPhysicalDevice),
                StatusCode::CONFLICT,
                "translation_precondition_failed",
            ),
            (
                Err(FactsError::SinkValidationFailed),
                StatusCode::CONFLICT,
                "translation_precondition_failed",
            ),
            (
                Ok(OutputMode::UnknownUnsafe),
                StatusCode::CONFLICT,
                "translation_precondition_failed",
            ),
        ] {
            let store = RuntimeStore::default();
            let cached = ready_facts();
            store.set_devices(cached.devices.into());
            store.set_audio_graph(cached.audio_graph);
            store.set_routes(cached.routes);
            let before = serde_json::to_value(store.snapshot()).unwrap();
            let mut events = store.subscribe().unwrap();
            let effects = Arc::new(Effects::default());
            let observations = Arc::new(Observations::new(observation));
            let mut owner = owner(
                store.clone(),
                Arc::new(Runner(effects.clone())),
                effects.clone(),
            );
            owner.facts = observations.clone();
            let admitted_deadline = deadline();
            let result = owner.execute(ControlCommand::Start, admitted_deadline);
            let after = serde_json::to_value(store.snapshot()).unwrap();
            let generation = owner.supervisor.next_generation;
            let gate = owner.supervisor.gate.state();
            let event = events.try_recv();
            owner.supervisor.stop(deadline()).unwrap();

            assert_eq!(
                result.err().map(|error| (error.status, error.code)),
                Some((expected_status, expected_code)),
                "{observation:?}"
            );
            assert_eq!(
                observations.deadlines.lock().unwrap().as_slice(),
                &[admitted_deadline.into_std()]
            );
            assert_eq!(effects.maintenance.load(Ordering::SeqCst), 0);
            assert_eq!(effects.starts.load(Ordering::SeqCst), 0);
            assert_eq!(generation, 0);
            assert_eq!(gate, crate::AudioOperationState::Idle);
            assert_eq!(after, before);
            assert!(matches!(event, Err(broadcast::error::TryRecvError::Empty)));
        }
    }

    #[test]
    fn safe_running_configuration_rejects_fresh_unsafe_facts_before_replacement() {
        let mut voice = RuntimeSnapshot::default().directions[0]
            .voice_profile
            .clone();
        voice.gender = translator_core::VoiceGender::Female;
        let commands = [
            ControlCommand::PatchDirection(DirectionPatch {
                direction_id: AudioDirection::Microphone,
                source_language: Some(translator_core::Language::En),
                target_language: Some(translator_core::Language::Ru),
                enabled: None,
            }),
            ControlCommand::PatchProvider(crate::ProviderPatch {
                provider_id: translator_core::ProviderId::Openai,
                cloud_opt_in: Some(true),
            }),
            ControlCommand::PatchVoice(crate::VoiceProfilePatch {
                direction_id: AudioDirection::Microphone,
                voice_profile: voice,
            }),
        ];
        for command in commands {
            let store = RuntimeStore::default();
            let effects = Arc::new(Effects::default());
            let observations = Arc::new(Observations::new(Ok(OutputMode::Headphones)));
            let mut owner = owner(
                store.clone(),
                Arc::new(Runner(effects.clone())),
                effects.clone(),
            );
            owner.facts = observations.clone();
            owner.execute(ControlCommand::Start, deadline()).unwrap();
            *observations.result.lock().unwrap() = Ok(OutputMode::UnknownUnsafe);
            let before = serde_json::to_value(store.snapshot()).unwrap();
            let mut events = store.subscribe().unwrap();
            let before_generation = owner.supervisor.next_generation;
            let owner_identity = match &mut owner.supervisor.state {
                SupervisorState::Running(active) => {
                    active
                        .direction_epochs
                        .insert(AudioDirection::Microphone, (11, 0));
                    active
                        .direction_epochs
                        .insert(AudioDirection::Speaker, (17, 0));
                    (&*active.runtime as *const dyn crate::ActiveDuplexRuntime).cast::<()>()
                }
                _ => panic!("fixture must own a running generation"),
            };
            let result = owner.execute(command, deadline());
            let after = serde_json::to_value(store.snapshot()).unwrap();
            let event = events.try_recv();
            let retained = match &owner.supervisor.state {
                SupervisorState::Running(active) => Some((
                    (&*active.runtime as *const dyn crate::ActiveDuplexRuntime).cast::<()>(),
                    active.direction_epochs.clone(),
                )),
                _ => None,
            };
            let stops_before_cleanup = effects.stops.load(Ordering::SeqCst);
            let generation = owner.supervisor.next_generation;
            let gate = owner.supervisor.gate.state();
            owner.supervisor.stop(deadline()).unwrap();

            assert_eq!(
                result.err().map(|error| (error.status, error.code)),
                Some((StatusCode::CONFLICT, "translation_precondition_failed"))
            );
            assert_eq!(observations.deadlines.lock().unwrap().len(), 2);
            assert_eq!(effects.maintenance.load(Ordering::SeqCst), 0);
            assert_eq!(effects.starts.load(Ordering::SeqCst), 1);
            assert_eq!(effects.replacements.load(Ordering::SeqCst), 0);
            assert_eq!(stops_before_cleanup, 0);
            assert_eq!(generation, before_generation);
            assert_eq!(gate, crate::AudioOperationState::Production);
            assert_eq!(
                retained,
                Some((
                    owner_identity,
                    HashMap::from([
                        (AudioDirection::Microphone, (11, 0)),
                        (AudioDirection::Speaker, (17, 0))
                    ])
                ))
            );
            assert_eq!(after, before);
            assert!(matches!(event, Err(broadcast::error::TryRecvError::Empty)));
        }
    }

    #[test]
    fn safe_start_both_disabled_rejects_before_maintenance_and_generation() {
        let store = RuntimeStore::default();
        let mut candidate = store.snapshot();
        for direction in &mut candidate.directions {
            direction.enabled = false;
        }
        store.commit_control(candidate);
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let mut events = store.subscribe().unwrap();
        let effects = Arc::new(Effects::default());
        let mut owner = owner(
            store.clone(),
            Arc::new(Runner(effects.clone())),
            effects.clone(),
        );
        let result = owner.execute(ControlCommand::Start, deadline());
        let after = serde_json::to_value(store.snapshot()).unwrap();
        let generation = owner.supervisor.next_generation;
        let gate = owner.supervisor.gate.state();
        let event = events.try_recv();
        owner.supervisor.stop(deadline()).unwrap();

        assert_eq!(
            result.err().map(|error| (error.status, error.code)),
            Some((StatusCode::CONFLICT, "no_direction_enabled"))
        );
        assert_eq!(effects.maintenance.load(Ordering::SeqCst), 0);
        assert_eq!(effects.starts.load(Ordering::SeqCst), 0);
        assert_eq!(generation, 0);
        assert_eq!(gate, crate::AudioOperationState::Idle);
        assert_eq!(after, before);
        assert!(matches!(event, Err(broadcast::error::TryRecvError::Empty)));
    }

    #[test]
    fn safe_invalid_native_configuration_does_not_consume_generation() {
        let store = RuntimeStore::default();
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let mut events = store.subscribe().unwrap();
        let effects = Arc::new(Effects::default());
        let runner = crate::ProcessDuplexRunner::new(crate::ProcessDuplexConfig {
            python: "/unused-safe-admission-python".into(),
            sidecar_root: "/unused-safe-admission-sidecar".into(),
            socket_path: "/unused-safe-admission.sock".into(),
            expected_uid: 0,
        });
        let mut owner = owner(store.clone(), Arc::new(runner), effects.clone());
        owner.facts = Arc::new(Facts { graph_ready: false });
        let result = owner.execute(ControlCommand::Start, deadline());
        let generation = owner.supervisor.next_generation;
        let gate = owner.supervisor.gate.state();
        let event = events.try_recv();
        owner.supervisor.stop(deadline()).unwrap();

        assert_eq!(
            result.err().map(|error| (error.status, error.code)),
            Some((StatusCode::CONFLICT, "translation_precondition_failed"))
        );
        assert_eq!(
            generation, 0,
            "the actual native validator must reject before generation acquisition"
        );
        assert_eq!(effects.maintenance.load(Ordering::SeqCst), 0);
        assert_eq!(gate, crate::AudioOperationState::Idle);
        assert_eq!(serde_json::to_value(store.snapshot()).unwrap(), before);
        assert!(matches!(event, Err(broadcast::error::TryRecvError::Empty)));
    }

    #[test]
    fn safe_running_last_direction_disable_preserves_owner_and_snapshot() {
        let store = RuntimeStore::default();
        let mut initial = store.snapshot();
        initial
            .directions
            .iter_mut()
            .find(|state| state.direction_id == AudioDirection::Speaker)
            .unwrap()
            .enabled = false;
        store.commit_control(initial);
        let effects = Arc::new(Effects::default());
        let mut owner = owner(
            store.clone(),
            Arc::new(Runner(effects.clone())),
            effects.clone(),
        );
        owner.execute(ControlCommand::Start, deadline()).unwrap();
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let mut events = store.subscribe().unwrap();
        let before_generation = owner.supervisor.next_generation;
        let owner_address = match &owner.supervisor.state {
            SupervisorState::Running(active) => {
                (&*active.runtime as *const dyn crate::ActiveDuplexRuntime).cast::<()>()
            }
            _ => panic!("fixture must retain a running owner"),
        };
        let result = owner.execute(
            ControlCommand::PatchDirection(DirectionPatch {
                direction_id: AudioDirection::Microphone,
                source_language: None,
                target_language: None,
                enabled: Some(false),
            }),
            deadline(),
        );
        let after = serde_json::to_value(store.snapshot()).unwrap();
        let after_address = match &owner.supervisor.state {
            SupervisorState::Running(active) => {
                Some((&*active.runtime as *const dyn crate::ActiveDuplexRuntime).cast::<()>())
            }
            _ => None,
        };
        let event = events.try_recv();
        let generation = owner.supervisor.next_generation;
        let stops_before_cleanup = effects.stops.load(Ordering::SeqCst);
        let gate = owner.supervisor.gate.state();
        owner.supervisor.stop(deadline()).unwrap();

        assert_eq!(
            result.err().map(|error| (error.status, error.code)),
            Some((StatusCode::CONFLICT, "no_direction_enabled"))
        );
        assert_eq!(effects.replacements.load(Ordering::SeqCst), 0);
        assert_eq!(stops_before_cleanup, 0);
        assert_eq!(effects.starts.load(Ordering::SeqCst), 1);
        assert_eq!(generation, before_generation);
        assert_eq!(after_address, Some(owner_address));
        assert_eq!(gate, crate::AudioOperationState::Production);
        assert_eq!(after, before);
        assert!(matches!(event, Err(broadcast::error::TryRecvError::Empty)));
    }
}

#[cfg(test)]
mod corrective_mailbox_tests {
    use super::*;
    use crate::acoustic_admission::admit_translation;
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use std::{
        sync::{Condvar, atomic::AtomicUsize},
        time::Duration,
    };
    use tokio::sync::watch;
    use tower::ServiceExt;

    #[test]
    fn lifecycle_burst_occupies_exactly_three_coalesced_slots() {
        let (sender, receiver) = watch::channel(LifecycleSlots::default());
        let mailbox = CompletionMailbox { sender };
        for epoch in 1..=10_000 {
            for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
                mailbox.direction_status_changed(
                    41,
                    direction,
                    epoch,
                    DirectionRuntimeStatus::Recovering,
                    None,
                );
            }
        }
        mailbox.completed(41, Err(DuplexRuntimeError::StartFailed));
        mailbox.cleanup_started(41);
        mailbox.completed(41, Err(DuplexRuntimeError::StartFailed));
        mailbox.cleanup_started(40);
        mailbox.completed(40, Err(DuplexRuntimeError::StartFailed));
        for epoch in 10_001..=20_000 {
            mailbox.direction_status_changed(
                41,
                AudioDirection::Microphone,
                epoch,
                DirectionRuntimeStatus::Failed,
                Some(DirectionRuntimeFailure::RestartExhausted),
            );
        }
        mailbox.cleanup_started(42);
        mailbox.cleanup_started(41);
        mailbox.completed(42, Err(DuplexRuntimeError::StartFailed));
        mailbox.cleanup_started(42);
        mailbox.completed(41, Err(DuplexRuntimeError::StartFailed));

        let slots = receiver.borrow();
        assert_eq!(slots.occupied_slot_count(), 3);
        let microphone = slots
            .direction(AudioDirection::Microphone)
            .expect("the microphone owns one coalesced slot");
        assert_eq!(microphone.generation, 41);
        assert_eq!(microphone.epoch, 20_000);
        assert_eq!(microphone.status, DirectionRuntimeStatus::Failed);
        let speaker = slots
            .direction(AudioDirection::Speaker)
            .expect("the speaker owns one coalesced slot");
        assert_eq!(speaker.generation, 41);
        assert_eq!(speaker.epoch, 10_000);
        assert_eq!(speaker.status, DirectionRuntimeStatus::Recovering);
        let terminal = slots.terminal.expect("terminal owns one monotonic slot");
        assert_eq!(terminal.generation, 42);
        assert_eq!(terminal.phase, TerminalPhase::Completed);
    }

    #[derive(Default)]
    struct BlockingStopState {
        calls: AtomicUsize,
        failures: AtomicUsize,
        drops: AtomicUsize,
        entered: AtomicBool,
        stopped: AtomicBool,
        deadlines: StdMutex<Vec<Instant>>,
        release: (StdMutex<bool>, Condvar),
    }

    struct BlockingStopRunner(Arc<BlockingStopState>);

    impl DuplexRunner for BlockingStopRunner {
        fn start(&self, _admitted: AdmittedDuplex, _deadline: Instant) -> crate::DuplexStartResult {
            Ok(Box::new(BlockingStopRuntime(self.0.clone())))
        }
    }

    struct LifecyclePipelineRunner {
        state: Arc<BlockingStopState>,
        completion: StdMutex<Option<(u64, Arc<dyn DuplexCompletionObserver>)>>,
    }

    impl DuplexRunner for LifecyclePipelineRunner {
        fn start(&self, _admitted: AdmittedDuplex, _deadline: Instant) -> crate::DuplexStartResult {
            Ok(Box::new(BlockingStopRuntime(self.state.clone())))
        }

        fn start_supervised(
            &self,
            admitted: AdmittedDuplex,
            generation: u64,
            completion: Arc<dyn DuplexCompletionObserver>,
            deadline: Instant,
        ) -> crate::DuplexStartResult {
            *self.completion.lock().unwrap() = Some((generation, completion));
            self.start(admitted, deadline)
        }
    }

    struct BlockingStopRuntime(Arc<BlockingStopState>);

    impl Drop for BlockingStopRuntime {
        fn drop(&mut self) {
            self.0.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl crate::ActiveDuplexRuntime for BlockingStopRuntime {
        fn stop(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            self.0.deadlines.lock().unwrap().push(deadline);
            self.0.entered.store(true, Ordering::SeqCst);
            let release = self.0.release.0.lock().unwrap();
            let (release, timeout) = self
                .0
                .release
                .1
                .wait_timeout_while(release, Duration::from_secs(2), |release| !*release)
                .unwrap();
            if timeout.timed_out() && !*release {
                return Err(DuplexRuntimeError::StopFailed);
            }
            if self
                .0
                .failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                return Err(DuplexRuntimeError::StopFailed);
            }
            self.0.stopped.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn cleanup_deadline_is_stable_per_attempt_and_fresh_for_explicit_retry() {
        let state = Arc::new(BlockingStopState::default());
        state.failures.store(1, Ordering::SeqCst);
        *state.release.0.lock().unwrap() = true;
        let gate = AudioOperationGate::new();
        let (sender, _receiver) = watch::channel(LifecycleSlots::default());
        let mut supervisor = RuntimeSupervisor::new(
            Arc::new(BlockingStopRunner(state.clone())),
            gate.clone(),
            Arc::new(CompletionMailbox { sender }),
        );
        supervisor
            .start(
                admit_translation(
                    RuntimeSnapshot::default(),
                    super::safe_admission_tests::ready_facts(),
                )
                .unwrap(),
                Instant::now() + Duration::from_secs(4),
            )
            .unwrap();
        let generation = match &supervisor.state {
            SupervisorState::Running(active) => active.generation,
            _ => panic!("the fixture must own one running generation"),
        };

        assert_eq!(
            supervisor.cleanup_started(generation),
            Some(RuntimeStatus::CleanupPending)
        );
        let admitted_deadline = match &supervisor.state {
            SupervisorState::CleanupPending(active) => active.cleanup_deadline.unwrap(),
            _ => panic!("cleanup-started must retain the runtime owner"),
        };
        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(
            supervisor.cleanup_started(generation),
            Some(RuntimeStatus::CleanupPending)
        );
        let duplicate_deadline = match &supervisor.state {
            SupervisorState::CleanupPending(active) => active.cleanup_deadline.unwrap(),
            _ => panic!("duplicate cleanup-started must keep CleanupPending"),
        };
        assert_eq!(duplicate_deadline, admitted_deadline);

        assert_eq!(
            supervisor.complete(generation),
            Some(RuntimeStatus::CleanupPending)
        );
        let retry_deadline = Instant::now() + Duration::from_secs(8);
        supervisor.stop(retry_deadline).unwrap();
        assert_eq!(
            state.deadlines.lock().unwrap().as_slice(),
            [admitted_deadline, retry_deadline]
        );
        assert_eq!(gate.state(), crate::AudioOperationState::Idle);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn source_p1_real_exhausted_epoch_projects_through_store_and_sse() {
        const TOKEN: &str = "4242424242424242424242424242424242424242424242424242424242424242";

        let state = Arc::new(BlockingStopState::default());
        *state.release.0.lock().unwrap() = true;
        let runner = Arc::new(LifecyclePipelineRunner {
            state: state.clone(),
            completion: StdMutex::new(None),
        });
        let store = RuntimeStore::default();
        let application = ControlApplication::spawn(
            store.clone(),
            runner.clone(),
            AudioOperationGate::new(),
            Arc::new(NoopFacts),
            Arc::new(NoopFacts),
            None,
        );
        application.execute(ControlCommand::Start).await.unwrap();
        let (generation, completion) = runner
            .completion
            .lock()
            .unwrap()
            .clone()
            .expect("Start must expose its actual completion mailbox");

        let router = crate::build_router_with_controllers(
            store.clone(),
            crate::ControlToken::parse(TOKEN).unwrap(),
            crate::ApiLimits::default(),
            crate::ApiControllers {
                translation: Some(application.clone()),
                ..crate::ApiControllers::default()
            },
        );
        let response = router
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/events/stream")
                    .header("authorization", format!("Bearer {TOKEN}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut events = response.into_body();
        events
            .frame()
            .await
            .expect("SSE initial snapshot")
            .expect("SSE initial frame");

        crate::translation_runtime::tests::native_owner_red::drive_local_recovery_exhaustion(
            generation, completion,
        )
        .await;
        assert_eq!(
            application
                .execute(ControlCommand::ReconcileAudio)
                .await
                .unwrap_err()
                .code,
            "audio_mix_controller_unavailable",
            "the command is the serialized lifecycle barrier"
        );

        let projected =
            store
                .snapshot()
                .directions
                .into_iter()
                .all(|direction| match direction.direction_id {
                    AudioDirection::Microphone => {
                        direction.runtime_status == DirectionRuntimeStatus::Failed
                            && direction.runtime_failure
                                == Some(DirectionRuntimeFailure::RestartExhausted)
                    }
                    AudioDirection::Speaker => {
                        direction.runtime_status == DirectionRuntimeStatus::Running
                            && direction.runtime_failure.is_none()
                    }
                });
        let mut sse_projected = false;
        for _ in 0..32 {
            let Ok(Some(Ok(frame))) =
                tokio::time::timeout(Duration::from_millis(25), events.frame()).await
            else {
                break;
            };
            let Ok(data) = frame.into_data() else {
                continue;
            };
            let Ok(text) = std::str::from_utf8(&data) else {
                continue;
            };
            if text.contains("\"direction_id\":\"microphone\"")
                && text.contains("\"runtime_status\":\"failed\"")
                && text.contains("\"direction_id\":\"speaker\"")
                && text.contains("\"runtime_status\":\"running\"")
            {
                sse_projected = true;
                break;
            }
        }
        drop(events);
        application.shutdown().await.unwrap();

        assert!(
            projected,
            "the actual exhausted coordinator epoch must reach the authoritative store"
        );
        assert!(
            sse_projected,
            "the authoritative failed/peer-running snapshot must reach authenticated SSE"
        );
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);
    }

    #[derive(Default)]
    struct CountingMix(AtomicUsize);

    impl AudioMixController for CountingMix {
        fn apply_desired(
            &self,
            _volumes: crate::AudioMixState,
            _mode: TranslationMixMode,
        ) -> Result<(), ControlFailure> {
            Ok(())
        }

        fn reconcile_committed(&self, mode: TranslationMixMode) -> Result<(), ControlFailure> {
            if mode == TranslationMixMode::Bypass {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }

        fn recover_committed(&self, _mode: TranslationMixMode) -> Result<(), ControlFailure> {
            Ok(())
        }
    }

    struct NoopFacts;

    impl RuntimeMaintenance for NoopFacts {
        fn refresh(&self, _store: &RuntimeStore) -> Result<(), ControlFailure> {
            Ok(())
        }
    }

    impl RuntimeFactsSource for NoopFacts {
        fn inspect(
            &self,
            _deadline: std::time::Instant,
        ) -> Result<crate::RuntimeFacts, FactsError> {
            Ok(super::safe_admission_tests::ready_facts())
        }
    }

    struct BlockingFacts(Arc<BlockingStopState>);

    impl RuntimeFactsSource for BlockingFacts {
        fn inspect(
            &self,
            _deadline: std::time::Instant,
        ) -> Result<crate::RuntimeFacts, FactsError> {
            self.0.entered.store(true, Ordering::SeqCst);
            let release = self.0.release.0.lock().unwrap();
            let (release, timeout) = self
                .0
                .release
                .1
                .wait_timeout_while(release, Duration::from_secs(2), |release| !*release)
                .unwrap();
            if timeout.timed_out() && !*release {
                return Err(FactsError::DiscoveryFailed);
            }
            Ok(super::safe_admission_tests::ready_facts())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_accepted_shutdown_with_actor_join_error_is_reaped_once() {
        let state = Arc::new(BlockingStopState::default());
        let mix = Arc::new(CountingMix::default());
        let application = ControlApplication::spawn(
            RuntimeStore::default(),
            Arc::new(BlockingStopRunner(state.clone())),
            AudioOperationGate::new(),
            Arc::new(NoopFacts),
            Arc::new(NoopFacts),
            Some(mix.clone()),
        );
        application.execute(ControlCommand::Start).await.unwrap();
        let caller = tokio::spawn({
            let application = application.clone();
            async move { application.shutdown().await }
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while !state.entered.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the actor must accept shutdown before cancellation");
        caller.abort();
        let _ = caller.await;
        {
            let actor = application.actor.lock().await;
            actor.as_ref().expect("actor handle remains owned").abort();
        }
        *state.release.0.lock().unwrap() = true;
        state.release.1.notify_all();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !state.stopped.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the accepted blocking cleanup must finish once");

        assert_eq!(
            application.shutdown().await.unwrap_err().code,
            "translation_controller_unavailable"
        );
        assert!(
            application.actor.lock().await.is_none(),
            "a confirmed JoinError must be cached after taking the exact actor handle"
        );
        assert_eq!(
            application.shutdown().await.unwrap_err().code,
            "translation_controller_unavailable"
        );
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            mix.0.load(Ordering::SeqCst),
            1,
            "aborting the actor join must not detach the accepted bypass transaction"
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

    #[tokio::test]
    async fn idle_actor_abort_retains_the_running_owner_and_production_lease() {
        let state = Arc::new(BlockingStopState::default());
        *state.release.0.lock().unwrap() = true;
        let gate = AudioOperationGate::new();
        let application = ControlApplication::spawn(
            RuntimeStore::default(),
            Arc::new(BlockingStopRunner(state.clone())),
            gate.clone(),
            Arc::new(NoopFacts),
            Arc::new(NoopFacts),
            None,
        );
        application.execute(ControlCommand::Start).await.unwrap();
        {
            let actor = application.actor.lock().await;
            actor.as_ref().expect("actor handle remains owned").abort();
        }

        assert_eq!(
            application.shutdown().await.unwrap_err().code,
            "translation_controller_unavailable"
        );
        assert_eq!(
            application.shutdown().await.unwrap_err().code,
            "translation_controller_unavailable"
        );
        assert_eq!(state.calls.load(Ordering::SeqCst), 0);
        assert_eq!(state.drops.load(Ordering::SeqCst), 0);
        assert_eq!(gate.state(), crate::AudioOperationState::Production);

        lock_recovering(&application._owner)
            .supervisor
            .stop(Instant::now() + crate::RUNTIME_CLEANUP_BUDGET)
            .expect("the resource-free fixture owner must remain explicitly drainable");
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
        assert_eq!(gate.state(), crate::AudioOperationState::Idle);
    }

    #[tokio::test]
    async fn actor_abort_retains_the_cleanup_pending_owner_without_an_implicit_retry() {
        let state = Arc::new(BlockingStopState::default());
        state.failures.store(1, Ordering::SeqCst);
        *state.release.0.lock().unwrap() = true;
        let gate = AudioOperationGate::new();
        let store = RuntimeStore::default();
        let application = ControlApplication::spawn(
            store.clone(),
            Arc::new(BlockingStopRunner(state.clone())),
            gate.clone(),
            Arc::new(NoopFacts),
            Arc::new(NoopFacts),
            None,
        );
        application.execute(ControlCommand::Start).await.unwrap();
        assert_eq!(
            application
                .execute(ControlCommand::Stop)
                .await
                .unwrap_err()
                .code,
            "translation_stop_failed"
        );
        {
            let actor = application.actor.lock().await;
            actor.as_ref().expect("actor handle remains owned").abort();
        }

        assert_eq!(
            application.shutdown().await.unwrap_err().code,
            "translation_controller_unavailable"
        );
        assert_eq!(
            application.shutdown().await.unwrap_err().code,
            "translation_controller_unavailable"
        );
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);
        assert_eq!(state.drops.load(Ordering::SeqCst), 0);
        assert_eq!(gate.state(), crate::AudioOperationState::Production);
        assert_eq!(
            store.snapshot().runtime_status,
            RuntimeStatus::CleanupPending
        );

        lock_recovering(&application._owner)
            .supervisor
            .stop(Instant::now() + crate::RUNTIME_CLEANUP_BUDGET)
            .expect("the resource-free cleanup owner must remain explicitly drainable");
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
        assert_eq!(gate.state(), crate::AudioOperationState::Idle);
    }

    #[tokio::test]
    async fn failed_shutdown_admission_is_sticky_closed() {
        let state = Arc::new(BlockingStopState::default());
        *state.release.0.lock().unwrap() = true;
        let application = ControlApplication::spawn(
            RuntimeStore::default(),
            Arc::new(BlockingStopRunner(state)),
            AudioOperationGate::new(),
            Arc::new(NoopFacts),
            Arc::new(NoopFacts),
            None,
        );
        {
            let actor = application.actor.lock().await;
            actor.as_ref().expect("actor handle remains owned").abort();
        }
        tokio::task::yield_now().await;

        assert_eq!(
            application.shutdown().await.unwrap_err().code,
            "translation_controller_unavailable"
        );
        assert!(
            application.closed.load(Ordering::Acquire),
            "the first shutdown call closes normal admission even when send fails"
        );
        assert_eq!(
            application
                .execute(ControlCommand::Start)
                .await
                .unwrap_err()
                .code,
            "translation_controller_unavailable"
        );
        let mut actor = application.actor.lock().await;
        if let Some(handle) = actor.take() {
            let _ = handle.await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mailbox_pressure_before_shutdown_admission_is_sticky_closed() {
        let state = Arc::new(BlockingStopState::default());
        let application = ControlApplication::spawn(
            RuntimeStore::default(),
            Arc::new(BlockingStopRunner(state.clone())),
            AudioOperationGate::new(),
            Arc::new(BlockingFacts(state.clone())),
            Arc::new(NoopFacts),
            None,
        );
        let start = tokio::spawn({
            let application = application.clone();
            async move { application.execute(ControlCommand::Start).await }
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while !state.entered.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("Start must stall the serialized owner");

        let mut queued_receipts = Vec::new();
        loop {
            let (response, receipt) = oneshot::channel();
            match application.sender.try_send(ActorMessage::Shutdown {
                deadline: Instant::now() + crate::RUNTIME_CLEANUP_BUDGET,
                response,
            }) {
                Ok(()) => queued_receipts.push(receipt),
                Err(mpsc::error::TrySendError::Full(_)) => break,
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    panic!("the actor mailbox closed before the pressure probe")
                }
            }
        }
        assert!(!queued_receipts.is_empty());
        assert_eq!(
            application.shutdown().await.unwrap_err().code,
            "translation_controller_unavailable"
        );
        let stayed_closed = application.closed.load(Ordering::Acquire);

        *state.release.0.lock().unwrap() = true;
        state.release.1.notify_all();
        start.await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while state.calls.load(Ordering::SeqCst) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the queued cleanup must drain the fake runtime");
        let mut actor = application.actor.lock().await;
        if let Some(handle) = actor.take() {
            handle.await.unwrap();
        }
        assert!(
            stayed_closed,
            "shutdown closes normal admission before a mailbox-full send attempt"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_public_shutdown_before_admission_stays_closed_and_retries_once() {
        let state = Arc::new(BlockingStopState::default());
        *state.release.0.lock().unwrap() = true;
        let application = ControlApplication::spawn(
            RuntimeStore::default(),
            Arc::new(BlockingStopRunner(state.clone())),
            AudioOperationGate::new(),
            Arc::new(NoopFacts),
            Arc::new(NoopFacts),
            None,
        );
        application.execute(ControlCommand::Start).await.unwrap();

        let mut reservations = Vec::new();
        loop {
            match application.sender.clone().try_reserve_owned() {
                Ok(permit) => reservations.push(permit),
                Err(mpsc::error::TrySendError::Full(_)) => break,
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    panic!("the actor mailbox closed before the admission probe")
                }
            }
        }
        assert_eq!(reservations.len(), MAILBOX_CAPACITY);
        let mut cancelled = tokio::spawn({
            let application = application.clone();
            async move { application.shutdown().await }
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while !application.closed.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("shutdown must close normal admission before waiting for mailbox capacity");
        let was_pending = !cancelled.is_finished();
        if was_pending {
            cancelled.abort();
        }
        let _ = (&mut cancelled).await;
        let stayed_closed = application.closed.load(Ordering::Acquire);
        drop(reservations);

        application
            .shutdown()
            .await
            .expect("a later call must admit the never-accepted cleanup exactly once");
        assert!(
            was_pending,
            "shutdown admission must remain cancellation-safe while mailbox capacity is reserved"
        );
        assert!(
            stayed_closed,
            "normal admission closes before mailbox admission"
        );
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            application
                .execute(ControlCommand::Start)
                .await
                .unwrap_err()
                .code,
            "translation_controller_unavailable"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn receiver_close_retries_owned_cleanup_and_joins_one_actor() {
        let state = Arc::new(BlockingStopState::default());
        state.failures.store(1, Ordering::SeqCst);
        *state.release.0.lock().unwrap() = true;
        let gate = AudioOperationGate::new();
        let (completion_sender, completion_receiver) = watch::channel(LifecycleSlots::default());
        let owner = Arc::new(StdMutex::new(ControlOwner {
            supervisor: RuntimeSupervisor::new(
                Arc::new(BlockingStopRunner(state.clone())),
                gate.clone(),
                Arc::new(CompletionMailbox {
                    sender: completion_sender,
                }),
            ),
            store: RuntimeStore::default(),
            facts: Arc::new(NoopFacts),
            maintenance: Arc::new(NoopFacts),
            aec_authority: None,
            audio_mix: None,
            aec_runtime_generation: None,
            aec_revocation_seen: 0,
            terminal_seen: None,
        }));
        lock_recovering(&owner)
            .execute(
                ControlCommand::Start,
                Instant::now() + crate::DIRECTION_CLEANUP_BUDGET,
            )
            .unwrap();
        let (command_sender, command_receiver) = mpsc::channel(1);
        let actor = tokio::spawn(run_actor(
            command_receiver,
            completion_receiver,
            None,
            owner.clone(),
        ));
        drop(command_sender);

        tokio::time::timeout(Duration::from_secs(1), async {
            while state.calls.load(Ordering::SeqCst) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("receiver-close must attempt cleanup once");
        tokio::time::advance(CLEANUP_RETRY_DELAY - Duration::from_millis(1)).await;
        assert_eq!(
            state.calls.load(Ordering::SeqCst),
            1,
            "cleanup failure must not create a zero-delay busy loop"
        );
        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::time::timeout(Duration::from_secs(1), actor)
            .await
            .expect("receiver-close cleanup retry must be bounded")
            .expect("the one owned actor must exit after cleanup");
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
        assert!(state.stopped.load(Ordering::SeqCst));
        assert_eq!(gate.state(), crate::AudioOperationState::Idle);
        assert_eq!(
            lock_recovering(&owner).supervisor.status(),
            RuntimeStatus::Stopped
        );
    }
}
