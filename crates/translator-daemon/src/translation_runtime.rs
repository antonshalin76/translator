use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt, fs,
    future::Future,
    num::NonZeroU32,
    os::unix::fs::MetadataExt,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex, mpsc as std_mpsc},
    thread,
    time::Duration,
};

use rustix::time::{ClockId, clock_gettime};
use serde::Serialize;
use thiserror::Error;
use tokio::{
    sync::{Mutex as AsyncMutex, mpsc, oneshot, watch},
    task::{Id as TaskId, JoinError, JoinSet},
    time::Instant,
};
use translator_audio::{
    BoundedPcmQueue, CaptureEvent, PcmFrame, PulsePcmCapture, PulsePcmCommand, PulsePcmPlayback,
    SpeechSegmenter, VoiceDetector, WebRtcVoiceDetector,
};
use translator_core::{AudioDirection, TranslationMode};
use translator_ipc::{
    ProviderClientError, ProviderStreamClient,
    provider::{CloseRequestReason, ProviderRequest, ProviderState, provider_event},
    wait_provider_ready,
};
use uuid::Uuid;

use crate::{
    AdmittedDuplex, AecStartReservation, CLOSE_ACK_TIMEOUT, DirectionEffect,
    DirectionRuntimeConfig, DirectionRuntimeFailure, DirectionRuntimeStatus, DirectionSession,
    DirectionWatchdogEffect, LatencySample, ProcessSidecarRuntime, RuntimeSnapshot, RuntimeStore,
    SafeProviderErrorCode, SidecarSupervisor, TerminalOutcome,
};

const PROVIDER_READY_TIMEOUT: Duration = Duration::from_secs(120);
const START_ACK_TIMEOUT: Duration = Duration::from_secs(130);
const DIRECTION_OPEN_TIMEOUT: Duration = Duration::from_secs(5);
pub const DIRECTION_CLEANUP_BUDGET: Duration = Duration::from_secs(4);
pub const RUNTIME_CLEANUP_BUDGET: Duration = Duration::from_secs(8);
const HOT_IO_LIVENESS_TIMEOUT: Duration = Duration::from_secs(1);
const WATCHDOG_INTERVAL: Duration = Duration::from_millis(25);
const MAX_RUNTIME_RESTARTS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DuplexRuntimeError {
    #[error("duplex runtime configuration is unavailable")]
    InvalidConfiguration,
    #[error("duplex runtime could not start")]
    StartFailed,
    #[error("duplex runtime could not stop")]
    StopFailed,
    #[error("duplex runtime could not apply a replacement")]
    ReconfigureFailed,
    #[error("duplex runtime could not restore the prior configuration")]
    RestoreFailed,
}

pub type DuplexStartResult = Result<Box<dyn ActiveDuplexRuntime>, DuplexStartFailure>;

pub struct DuplexStartFailure {
    error: DuplexRuntimeError,
    cleanup: Option<Box<dyn ActiveDuplexRuntime>>,
}

impl DuplexStartFailure {
    pub fn rejected(error: DuplexRuntimeError) -> Self {
        Self {
            error,
            cleanup: None,
        }
    }

    pub fn cleanup_pending(
        error: DuplexRuntimeError,
        cleanup: Box<dyn ActiveDuplexRuntime>,
    ) -> Self {
        Self {
            error,
            cleanup: Some(cleanup),
        }
    }

    pub fn error(&self) -> DuplexRuntimeError {
        self.error
    }

    pub fn has_cleanup(&self) -> bool {
        self.cleanup.is_some()
    }

    pub fn into_parts(self) -> (DuplexRuntimeError, Option<Box<dyn ActiveDuplexRuntime>>) {
        (self.error, self.cleanup)
    }
}

impl From<DuplexRuntimeError> for DuplexStartFailure {
    fn from(error: DuplexRuntimeError) -> Self {
        Self::rejected(error)
    }
}

impl fmt::Debug for DuplexStartFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DuplexStartFailure")
            .field("error", &self.error)
            .field("cleanup", &self.cleanup.is_some())
            .finish()
    }
}

impl fmt::Display for DuplexStartFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for DuplexStartFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

pub trait ActiveDuplexRuntime: Send {
    fn reconfigure(
        &mut self,
        _admitted: AdmittedDuplex,
        _deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        Err(DuplexRuntimeError::ReconfigureFailed)
    }

    fn stop(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError>;

    fn reap(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        self.stop(deadline)
    }
}

pub trait DuplexCompletionObserver: Send + Sync {
    fn cleanup_started(&self, _generation: u64) {}

    fn completed(&self, generation: u64, result: Result<(), DuplexRuntimeError>);

    fn direction_status_changed(
        &self,
        _generation: u64,
        _direction: AudioDirection,
        _epoch: u64,
        _status: DirectionRuntimeStatus,
        _failure: Option<DirectionRuntimeFailure>,
    ) {
    }
}

pub trait DuplexRunner: Send + Sync {
    fn start(&self, admitted: AdmittedDuplex, deadline: Instant) -> DuplexStartResult;

    fn start_supervised(
        &self,
        admitted: AdmittedDuplex,
        _generation: u64,
        _completion: Arc<dyn DuplexCompletionObserver>,
        deadline: Instant,
    ) -> DuplexStartResult {
        self.start(admitted, deadline)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum DuplexRuntimeEvent {
    SpeechStarted {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        #[serde(skip)]
        capture_monotonic_ns: u64,
    },
    #[serde(rename = "asr_final")]
    TranscriptFinal {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
    },
    TranslationFinal {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
    },
    AudioFrame {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        sequence: u64,
        provider_monotonic_ns: u64,
        #[serde(skip)]
        observed_monotonic_ns: u64,
        queue_lag_ms: u32,
    },
    FirstAudioExpired {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        #[serde(skip)]
        observed_monotonic_ns: u64,
    },
    ProviderLatency {
        direction: AudioDirection,
        #[serde(skip_serializing_if = "Option::is_none")]
        utterance_id: Option<uuid::Uuid>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tts_first_audio_ms: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_total_ms: Option<u32>,
    },
    ProviderError {
        direction: AudioDirection,
        #[serde(skip_serializing_if = "Option::is_none")]
        utterance_id: Option<uuid::Uuid>,
        code: SafeProviderErrorCode,
        retryable: bool,
    },
    UtteranceTerminalOutcome {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        outcome: TerminalOutcome,
    },
    UtteranceTerminal {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
    },
    GenerationRestart {
        attempt: NonZeroU32,
    },
}

pub const TASK7_BRIDGE_SCHEMA_VERSION: &str = "translator.task7-bridge.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task7BridgeFailureStage {
    RuntimeLease,
    AudioGraphEnsure,
    RuntimeConfiguration,
    RuntimeStart,
    RuntimeStop,
    ControlInput,
    AudioGraphCleanup,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Task7BridgeEvent {
    schema_version: &'static str,
    monotonic_ns: u64,
    #[serde(flatten)]
    payload: Task7BridgeEventPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
enum Task7BridgeEventPayload {
    Runtime(DuplexRuntimeEvent),
    Control(Task7BridgeControlEvent),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum Task7BridgeControlEvent {
    Ready {
        pid: u32,
    },
    Stopped,
    Failure {
        stage: Task7BridgeFailureStage,
        code: &'static str,
    },
}

impl Task7BridgeEvent {
    pub fn ready(pid: u32) -> Self {
        Self::control(Task7BridgeControlEvent::Ready { pid })
    }

    pub fn from_runtime(event: DuplexRuntimeEvent) -> Self {
        let timestamp = match event {
            DuplexRuntimeEvent::SpeechStarted {
                capture_monotonic_ns,
                ..
            } => capture_monotonic_ns,
            DuplexRuntimeEvent::AudioFrame {
                observed_monotonic_ns,
                ..
            }
            | DuplexRuntimeEvent::FirstAudioExpired {
                observed_monotonic_ns,
                ..
            } => observed_monotonic_ns,
            DuplexRuntimeEvent::TranscriptFinal { .. }
            | DuplexRuntimeEvent::TranslationFinal { .. }
            | DuplexRuntimeEvent::ProviderLatency { .. }
            | DuplexRuntimeEvent::ProviderError { .. }
            | DuplexRuntimeEvent::UtteranceTerminalOutcome { .. }
            | DuplexRuntimeEvent::UtteranceTerminal { .. }
            | DuplexRuntimeEvent::GenerationRestart { .. } => monotonic_ns(),
        };
        Self {
            schema_version: TASK7_BRIDGE_SCHEMA_VERSION,
            monotonic_ns: timestamp,
            payload: Task7BridgeEventPayload::Runtime(event),
        }
    }

    pub fn stopped() -> Self {
        Self::control(Task7BridgeControlEvent::Stopped)
    }

    pub fn failure(stage: Task7BridgeFailureStage, code: &'static str) -> Self {
        Self::control(Task7BridgeControlEvent::Failure { stage, code })
    }

    fn control(payload: Task7BridgeControlEvent) -> Self {
        Self {
            schema_version: TASK7_BRIDGE_SCHEMA_VERSION,
            monotonic_ns: monotonic_ns(),
            payload: Task7BridgeEventPayload::Control(payload),
        }
    }
}

pub trait DuplexRuntimeObserver: Send + Sync {
    fn observe(&self, event: DuplexRuntimeEvent);

    #[doc(hidden)]
    fn capture_frame_processed(&self, _direction: AudioDirection, _frame: CompletedCaptureFrame) {}

    #[doc(hidden)]
    fn capture_frames_pending(
        &self,
        _direction: AudioDirection,
        _runtime_generation: Uuid,
        _pending_frames: u64,
    ) {
    }

    #[doc(hidden)]
    fn provider_submission_attempted(
        &self,
        _direction: AudioDirection,
        _observed_monotonic_ns: u64,
    ) {
    }

    #[doc(hidden)]
    fn provider_submission_accepted(
        &self,
        _direction: AudioDirection,
        _observed_monotonic_ns: u64,
    ) {
    }

    #[doc(hidden)]
    fn provider_submission_attempted_for_origin(
        &self,
        direction: AudioDirection,
        origin: ProviderEffectOrigin,
    ) {
        self.provider_submission_attempted(direction, origin.observed_monotonic_ns);
    }

    #[doc(hidden)]
    fn provider_submission_accepted_for_origin(
        &self,
        direction: AudioDirection,
        origin: ProviderEffectOrigin,
    ) {
        self.provider_submission_accepted(direction, origin.observed_monotonic_ns);
    }

    fn reset_direction(&self, _direction: AudioDirection) {}

    fn requested_mode(&self, _direction: AudioDirection) -> Option<TranslationMode> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletedCaptureFrame {
    pub sequence: u64,
    pub capture_monotonic_ns: u64,
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub frame_duration_ms: u16,
    pub samples_per_frame: u64,
    pub runtime_generation: Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderEffectOrigin {
    pub runtime_generation: Uuid,
    pub capture_monotonic_ns: Option<u64>,
    pub observed_monotonic_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProviderEffectContext {
    runtime_generation: Uuid,
    capture_monotonic_ns: Option<u64>,
}

impl ProviderEffectContext {
    const fn unattributed(runtime_generation: Uuid) -> Self {
        Self {
            runtime_generation,
            capture_monotonic_ns: None,
        }
    }

    const fn captured(runtime_generation: Uuid, capture_monotonic_ns: u64) -> Self {
        Self {
            runtime_generation,
            capture_monotonic_ns: Some(capture_monotonic_ns),
        }
    }

    fn observed(self) -> ProviderEffectOrigin {
        ProviderEffectOrigin {
            runtime_generation: self.runtime_generation,
            capture_monotonic_ns: self.capture_monotonic_ns,
            observed_monotonic_ns: monotonic_ns(),
        }
    }
}

#[derive(Debug, Default)]
struct ObservedUtterance {
    capture_monotonic_ns: u64,
    first_audio_monotonic_ns: Option<u64>,
    last_audio_monotonic_ns: Option<u64>,
    max_queue_lag_ms: u32,
}

pub struct RuntimeLatencyObserver {
    store: RuntimeStore,
    utterances: Mutex<HashMap<(AudioDirection, uuid::Uuid), ObservedUtterance>>,
}

impl RuntimeLatencyObserver {
    pub fn new(store: RuntimeStore) -> Self {
        Self {
            store,
            utterances: Mutex::new(HashMap::new()),
        }
    }
}

impl DuplexRuntimeObserver for RuntimeLatencyObserver {
    fn observe(&self, event: DuplexRuntimeEvent) {
        let mut utterances = self
            .utterances
            .lock()
            .expect("latency observer mutex poisoned");
        match event {
            DuplexRuntimeEvent::SpeechStarted {
                direction,
                utterance_id,
                capture_monotonic_ns,
            } => {
                let key = (direction, utterance_id);
                if utterances.contains_key(&key) {
                    return;
                }
                if utterances
                    .keys()
                    .filter(|(observed, _)| *observed == direction)
                    .count()
                    >= translator_ipc::MAX_ACTIVE_UTTERANCES
                {
                    utterances.retain(|(observed, _), _| *observed != direction);
                }
                utterances.insert(
                    key,
                    ObservedUtterance {
                        capture_monotonic_ns,
                        ..ObservedUtterance::default()
                    },
                );
            }
            DuplexRuntimeEvent::AudioFrame {
                direction,
                utterance_id,
                observed_monotonic_ns,
                queue_lag_ms,
                ..
            } => {
                if let Some(utterance) = utterances.get_mut(&(direction, utterance_id)) {
                    utterance
                        .first_audio_monotonic_ns
                        .get_or_insert(observed_monotonic_ns);
                    utterance.last_audio_monotonic_ns = Some(observed_monotonic_ns);
                    utterance.max_queue_lag_ms = utterance.max_queue_lag_ms.max(queue_lag_ms);
                    self.store.observe_latency_queue(
                        direction,
                        observed_monotonic_ns / 1_000_000,
                        Some(queue_lag_ms),
                    );
                }
            }
            DuplexRuntimeEvent::FirstAudioExpired {
                direction,
                utterance_id,
                observed_monotonic_ns,
            } => {
                if let Some(utterance) = utterances.get_mut(&(direction, utterance_id)) {
                    utterance.first_audio_monotonic_ns = Some(observed_monotonic_ns);
                    utterance.last_audio_monotonic_ns = Some(observed_monotonic_ns);
                }
            }
            DuplexRuntimeEvent::UtteranceTerminal {
                direction,
                utterance_id,
            } => {
                let Some(utterance) = utterances.remove(&(direction, utterance_id)) else {
                    return;
                };
                let (Some(first), Some(last)) = (
                    utterance.first_audio_monotonic_ns,
                    utterance.last_audio_monotonic_ns,
                ) else {
                    return;
                };
                let first_audio_ms = duration_ms(utterance.capture_monotonic_ns, first);
                let last_audio_ms = duration_ms(utterance.capture_monotonic_ns, last);
                self.store.record_latency_utterance(
                    direction,
                    last / 1_000_000,
                    LatencySample {
                        first_audio_ms,
                        last_audio_ms,
                        queue_lag_ms: utterance.max_queue_lag_ms,
                    },
                );
            }
            DuplexRuntimeEvent::TranscriptFinal { .. }
            | DuplexRuntimeEvent::TranslationFinal { .. }
            | DuplexRuntimeEvent::ProviderLatency { .. }
            | DuplexRuntimeEvent::ProviderError { .. }
            | DuplexRuntimeEvent::UtteranceTerminalOutcome { .. } => {}
            DuplexRuntimeEvent::GenerationRestart { .. } => {
                utterances.clear();
            }
        }
    }

    fn requested_mode(&self, direction: AudioDirection) -> Option<TranslationMode> {
        self.store
            .snapshot()
            .latency_policy
            .into_iter()
            .find(|policy| policy.direction_id == direction)
            .map(|policy| policy.current_mode)
    }

    fn reset_direction(&self, direction: AudioDirection) {
        self.utterances
            .lock()
            .expect("latency observer mutex poisoned")
            .retain(|(observed, _), _| *observed != direction);
    }
}

fn duration_ms(start_ns: u64, end_ns: u64) -> u32 {
    u32::try_from(end_ns.saturating_sub(start_ns) / 1_000_000).unwrap_or(u32::MAX)
}

struct NoopDuplexRuntimeObserver;

impl DuplexRuntimeObserver for NoopDuplexRuntimeObserver {
    fn observe(&self, _event: DuplexRuntimeEvent) {}
}

#[derive(Debug, Clone)]
pub struct ProcessDuplexConfig {
    pub python: PathBuf,
    pub sidecar_root: PathBuf,
    pub socket_path: PathBuf,
    pub expected_uid: u32,
}

impl ProcessDuplexConfig {
    pub fn from_runtime(
        python: PathBuf,
        sidecar_root: PathBuf,
        socket_path: PathBuf,
    ) -> Result<Self, DuplexRuntimeError> {
        let parent = socket_path
            .parent()
            .ok_or(DuplexRuntimeError::InvalidConfiguration)?;
        let expected_uid = fs::metadata(parent)
            .map_err(|_| DuplexRuntimeError::InvalidConfiguration)?
            .uid();
        Ok(Self {
            python,
            sidecar_root,
            socket_path,
            expected_uid,
        })
    }
}

pub struct ProcessDuplexRunner {
    config: ProcessDuplexConfig,
    observer: Arc<dyn DuplexRuntimeObserver>,
}

enum StartAck {
    Ready,
    Rejected(DuplexRuntimeError),
    CleanupPending(DuplexRuntimeError),
}

impl ProcessDuplexRunner {
    pub fn new(config: ProcessDuplexConfig) -> Self {
        Self {
            config,
            observer: Arc::new(NoopDuplexRuntimeObserver),
        }
    }

    pub fn with_observer(
        config: ProcessDuplexConfig,
        observer: Arc<dyn DuplexRuntimeObserver>,
    ) -> Self {
        Self { config, observer }
    }

    fn start_launch(
        &self,
        launch: DuplexLaunch,
        completion: Option<(u64, Arc<dyn DuplexCompletionObserver>)>,
        deadline: Instant,
    ) -> DuplexStartResult {
        let config = self.config.clone();
        let observer = self.observer.clone();
        let (stop_sender, stop_receiver) = watch::channel(None);
        let (command_sender, command_receiver) = mpsc::channel(1);
        let (ack_sender, ack_receiver) = std_mpsc::sync_channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("translator-duplex-runtime".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build();
                let async_completion = completion.clone();
                let result = match runtime {
                    Ok(runtime) => runtime.block_on(tokio::task::LocalSet::new().run_until(
                        run_process_duplex(
                            config,
                            launch,
                            stop_receiver,
                            command_receiver,
                            ack_sender,
                            observer,
                            async_completion.as_ref(),
                            deadline,
                        ),
                    )),
                    Err(_) => {
                        let _ =
                            ack_sender.send(StartAck::Rejected(DuplexRuntimeError::StartFailed));
                        Err(DuplexRuntimeError::StartFailed)
                    }
                };
                let _ = done_sender.send(result);
                if let Some((generation, completion)) = completion {
                    completion.completed(generation, result);
                }
            })
            .map_err(|_| DuplexStartFailure::rejected(DuplexRuntimeError::StartFailed))?;
        let runtime = ProcessActiveDuplex::new(stop_sender, command_sender, done_receiver, thread);
        let ack_deadline = deadline.min(Instant::now() + START_ACK_TIMEOUT);
        let ack = remaining(ack_deadline).and_then(|remaining| {
            ack_receiver
                .recv_timeout(remaining)
                .map_err(|_| DuplexRuntimeError::StartFailed)
        });
        resolve_start_ack(ack, runtime, deadline)
    }
}

impl DuplexRunner for ProcessDuplexRunner {
    fn start(&self, admitted: AdmittedDuplex, deadline: Instant) -> DuplexStartResult {
        let launch = DuplexLaunch::from(admitted);
        self.start_launch(launch, None, deadline)
    }

    fn start_supervised(
        &self,
        admitted: AdmittedDuplex,
        generation: u64,
        completion: Arc<dyn DuplexCompletionObserver>,
        deadline: Instant,
    ) -> DuplexStartResult {
        let launch = DuplexLaunch::from(admitted);
        self.start_launch(launch, Some((generation, completion)), deadline)
    }
}

enum RuntimeCommand {
    Reconfigure {
        launch: DuplexLaunch,
        deadline: Instant,
        response: std_mpsc::SyncSender<Result<(), DuplexRuntimeError>>,
    },
    Stop {
        deadline: Instant,
        response: std_mpsc::SyncSender<Result<(), DuplexRuntimeError>>,
    },
}

struct ProcessActiveDuplex {
    stop_sender: watch::Sender<Option<Instant>>,
    command_sender: mpsc::Sender<RuntimeCommand>,
    done_receiver: std_mpsc::Receiver<Result<(), DuplexRuntimeError>>,
    done_observed: bool,
    start_cleanup_confirmed: bool,
    thread: Option<thread::JoinHandle<()>>,
}

impl ActiveDuplexRuntime for ProcessActiveDuplex {
    fn reconfigure(
        &mut self,
        admitted: AdmittedDuplex,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let launch = DuplexLaunch::from(admitted);
        let (response, result) = std_mpsc::sync_channel(1);
        send_until(
            &self.command_sender,
            RuntimeCommand::Reconfigure {
                launch,
                deadline,
                response,
            },
            deadline,
            DuplexRuntimeError::ReconfigureFailed,
        )?;
        result
            .recv_timeout(remaining(deadline)?)
            .map_err(|_| DuplexRuntimeError::ReconfigureFailed)?
    }

    fn stop(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        self.stop_until(deadline)
    }

    fn reap(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        let mut terminal_observed = false;
        if !self.done_observed {
            match self.done_receiver.try_recv() {
                Ok(_) => {
                    self.done_observed = true;
                    terminal_observed = true;
                }
                Err(std_mpsc::TryRecvError::Disconnected) if self.start_cleanup_confirmed => {}
                Err(_) => return Err(DuplexRuntimeError::StopFailed),
            }
        }
        if terminal_observed {
            self.finish_until(deadline)
        } else {
            self.join_if_finished()
        }
    }
}

impl ProcessActiveDuplex {
    fn new(
        stop_sender: watch::Sender<Option<Instant>>,
        command_sender: mpsc::Sender<RuntimeCommand>,
        done_receiver: std_mpsc::Receiver<Result<(), DuplexRuntimeError>>,
        thread: thread::JoinHandle<()>,
    ) -> Self {
        Self {
            stop_sender,
            command_sender,
            done_receiver,
            done_observed: false,
            start_cleanup_confirmed: false,
            thread: Some(thread),
        }
    }

    fn request_stop(&self, deadline: Instant) {
        let _ = self.stop_sender.send(Some(deadline));
    }

    fn stop_until(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        let (response, result) = std_mpsc::sync_channel(1);
        if let Err(error) = send_until(
            &self.command_sender,
            RuntimeCommand::Stop { deadline, response },
            deadline,
            DuplexRuntimeError::StopFailed,
        ) {
            self.request_stop(deadline);
            return if self.command_sender.is_closed() {
                self.finish_until(deadline)
            } else {
                Err(error)
            };
        }
        match result.recv_timeout(remaining(deadline)?) {
            Ok(Ok(())) | Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                self.finish_until(deadline)
            }
            Ok(Err(error)) => Err(error),
            Err(std_mpsc::RecvTimeoutError::Timeout) => Err(DuplexRuntimeError::StopFailed),
        }
    }

    fn finish_until(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        if !self.done_observed {
            match self.done_receiver.recv_timeout(remaining(deadline)?) {
                Ok(_) => self.done_observed = true,
                Err(std_mpsc::RecvTimeoutError::Disconnected) if self.start_cleanup_confirmed => {}
                Err(_) => return Err(DuplexRuntimeError::StopFailed),
            }
        }
        while self
            .thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
        {
            if Instant::now() >= deadline {
                return Err(DuplexRuntimeError::StopFailed);
            }
            thread::yield_now();
        }
        self.join_if_finished()
    }

    fn join_if_finished(&mut self) -> Result<(), DuplexRuntimeError> {
        if self
            .thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
        {
            return Err(DuplexRuntimeError::StopFailed);
        }
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| DuplexRuntimeError::StopFailed)?;
        }
        Ok(())
    }
}

impl Drop for ProcessActiveDuplex {
    fn drop(&mut self) {
        self.request_stop(Instant::now());
    }
}

fn resolve_start_ack(
    ack: Result<StartAck, DuplexRuntimeError>,
    mut runtime: ProcessActiveDuplex,
    deadline: Instant,
) -> DuplexStartResult {
    match ack {
        Ok(StartAck::Ready) => Ok(Box::new(runtime)),
        Ok(StartAck::Rejected(error)) => {
            runtime.start_cleanup_confirmed = true;
            if runtime.finish_until(deadline).is_ok() {
                Err(DuplexStartFailure::rejected(error))
            } else {
                Err(DuplexStartFailure::cleanup_pending(
                    error,
                    Box::new(runtime),
                ))
            }
        }
        Ok(StartAck::CleanupPending(error)) => Err(DuplexStartFailure::cleanup_pending(
            error,
            Box::new(runtime),
        )),
        Err(_) => {
            runtime.request_stop(deadline);
            Err(DuplexStartFailure::cleanup_pending(
                DuplexRuntimeError::StartFailed,
                Box::new(runtime),
            ))
        }
    }
}

fn remaining(deadline: Instant) -> Result<Duration, DuplexRuntimeError> {
    deadline
        .checked_duration_since(Instant::now())
        .ok_or(DuplexRuntimeError::StopFailed)
}

fn send_until<T>(
    sender: &mpsc::Sender<T>,
    mut message: T,
    deadline: Instant,
    failure: DuplexRuntimeError,
) -> Result<(), DuplexRuntimeError> {
    loop {
        match sender.try_send(message) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(failure),
            Err(mpsc::error::TrySendError::Full(returned)) => message = returned,
        }
        if Instant::now() >= deadline {
            return Err(failure);
        }
        thread::yield_now();
    }
}

#[derive(Clone)]
struct DirectionLaunch {
    runtime: DirectionRuntimeConfig,
    capture_device: String,
    playback_device: String,
    capture_stream_name: &'static str,
    playback_stream_name: &'static str,
    aec_reservation: Option<Arc<AecStartReservation>>,
}

impl PartialEq for DirectionLaunch {
    fn eq(&self, other: &Self) -> bool {
        self.runtime == other.runtime
            && self.capture_device == other.capture_device
            && self.playback_device == other.playback_device
            && self.capture_stream_name == other.capture_stream_name
            && self.playback_stream_name == other.playback_stream_name
    }
}

impl Eq for DirectionLaunch {}

#[derive(Clone, PartialEq, Eq)]
struct DuplexLaunch {
    microphone: Option<DirectionLaunch>,
    speaker: Option<DirectionLaunch>,
}

impl From<AdmittedDuplex> for DuplexLaunch {
    fn from(admitted: AdmittedDuplex) -> Self {
        let (snapshot, targets, aec_reservation) = admitted.into_parts();
        let microphone = targets.microphone.map(|target| {
            direction_launch(
                &snapshot,
                AudioDirection::Microphone,
                target.capture,
                target.playback,
                "translator-outgoing-capture",
                "translator-outgoing-playback",
                aec_reservation,
            )
        });
        let speaker = targets.speaker.map(|target| {
            direction_launch(
                &snapshot,
                AudioDirection::Speaker,
                target.capture,
                target.playback,
                "translator-incoming-capture",
                "translator-incoming-playback",
                None,
            )
        });
        Self {
            microphone,
            speaker,
        }
    }
}

impl DuplexLaunch {
    fn for_direction(&self, direction: AudioDirection) -> Option<&DirectionLaunch> {
        match direction {
            AudioDirection::Microphone => self.microphone.as_ref(),
            AudioDirection::Speaker => self.speaker.as_ref(),
        }
    }

    fn set_direction(&mut self, direction: AudioDirection, launch: Option<DirectionLaunch>) {
        match direction {
            AudioDirection::Microphone => self.microphone = launch,
            AudioDirection::Speaker => self.speaker = launch,
        }
    }

    fn launches(&self) -> Vec<DirectionLaunch> {
        [self.microphone.clone(), self.speaker.clone()]
            .into_iter()
            .flatten()
            .collect()
    }

    fn changed_directions(&self, candidate: &Self) -> Vec<AudioDirection> {
        [AudioDirection::Microphone, AudioDirection::Speaker]
            .into_iter()
            .filter(|direction| {
                self.for_direction(*direction) != candidate.for_direction(*direction)
            })
            .collect()
    }
}

fn direction_launch(
    snapshot: &RuntimeSnapshot,
    direction: AudioDirection,
    capture_device: String,
    playback_device: String,
    capture_stream_name: &'static str,
    playback_stream_name: &'static str,
    aec_reservation: Option<Arc<AecStartReservation>>,
) -> DirectionLaunch {
    let state = snapshot
        .directions
        .iter()
        .find(|candidate| candidate.direction_id == direction)
        .expect("admission validates both direction configurations");
    let mode = snapshot
        .latency_policy
        .iter()
        .find(|candidate| candidate.direction_id == direction)
        .expect("admission validates both direction configurations")
        .current_mode;
    DirectionLaunch {
        runtime: DirectionRuntimeConfig {
            provider_id: snapshot.provider_id,
            direction,
            source_language: state.source_language,
            target_language: state.target_language,
            mode,
            voice_gender: state.voice_profile.gender,
            voice_engine: state.voice_profile.engine,
            debug_text_enabled: snapshot.debug_text_enabled,
        },
        capture_device,
        playback_device,
        capture_stream_name,
        playback_stream_name,
        aec_reservation,
    }
}

struct ProcessAcquisition {
    launch: DirectionLaunch,
    session: DirectionSession,
    provider: Option<ProviderStreamClient>,
    capture: Option<PulsePcmCapture>,
    playback: Option<PulsePcmPlayback>,
    runtime_generation: Option<Uuid>,
}

struct PreparedDirection {
    launch: DirectionLaunch,
    session: DirectionSession,
    provider: ProviderStreamClient,
    capture: PulsePcmCapture,
    playback: Option<PulsePcmPlayback>,
    playback_reusable: bool,
    runtime_generation: Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultScope {
    Local,
    ProviderConnection,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectionFailureOrigin {
    PcmCapture,
    PcmPlayback,
    Queue,
    Vad,
    SessionValidation,
    ProviderConnection,
    ProviderLocal,
    ProviderShared,
    SidecarExit,
    InternalTransport,
    WatchdogRestart,
}

const fn classify_direction_failure(origin: DirectionFailureOrigin) -> FaultScope {
    match origin {
        DirectionFailureOrigin::PcmCapture
        | DirectionFailureOrigin::PcmPlayback
        | DirectionFailureOrigin::Queue
        | DirectionFailureOrigin::Vad
        | DirectionFailureOrigin::SessionValidation
        | DirectionFailureOrigin::ProviderLocal => FaultScope::Local,
        DirectionFailureOrigin::ProviderConnection => FaultScope::ProviderConnection,
        DirectionFailureOrigin::SidecarExit
        | DirectionFailureOrigin::ProviderShared
        | DirectionFailureOrigin::InternalTransport
        | DirectionFailureOrigin::WatchdogRestart => FaultScope::Shared,
    }
}

fn provider_failure_origin(error: &ProviderClientError) -> DirectionFailureOrigin {
    match classify_provider_client_error(error) {
        FaultScope::Local => DirectionFailureOrigin::ProviderLocal,
        FaultScope::ProviderConnection => DirectionFailureOrigin::ProviderConnection,
        FaultScope::Shared => DirectionFailureOrigin::ProviderShared,
    }
}

const fn classify_provider_client_error(error: &ProviderClientError) -> FaultScope {
    match error {
        ProviderClientError::InvalidToken
        | ProviderClientError::InvalidEndpoint
        | ProviderClientError::InvalidOpenRequest
        | ProviderClientError::EventStreamProtocol
        | ProviderClientError::EventStreamResourceExhausted
        | ProviderClientError::EventStreamCancelled => FaultScope::Local,
        ProviderClientError::TransportUnavailable
        | ProviderClientError::EventStreamInternal
        | ProviderClientError::InvalidProbeResponse
        | ProviderClientError::ProviderReadyTimeout => FaultScope::Shared,
        ProviderClientError::RequestChannelClosed | ProviderClientError::EventStreamFailed => {
            FaultScope::ProviderConnection
        }
    }
}

const fn classify_provider_eof() -> FaultScope {
    FaultScope::ProviderConnection
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerStop {
    Close {
        reason: CloseRequestReason,
        deadline: Instant,
    },
    GenerationLost {
        deadline: Instant,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectionOutcome {
    Stopped(WorkerStop),
    ModeChanged,
    Fault(FaultScope),
}

#[allow(async_fn_in_trait)]
trait DirectionEffects: Clone + 'static {
    type Acquisition: 'static;
    type Prepared: 'static;

    fn begin(&self, launch: DirectionLaunch) -> Self::Acquisition;
    fn session_id(owner: &Self::Acquisition) -> uuid::Uuid;
    async fn prepare(
        &self,
        owner: &mut Self::Acquisition,
        generation: &crate::SidecarLaunch,
        deadline: Instant,
    ) -> Result<(), FaultScope>;
    fn finish(&self, owner: Self::Acquisition) -> Result<Self::Prepared, Self::Acquisition>;
    fn recover_owner(&self, prepared: Self::Prepared) -> Self::Acquisition;
    async fn run(
        &self,
        prepared: &mut Self::Prepared,
        stop: &mut watch::Receiver<Option<WorkerStop>>,
        observer: Arc<dyn DuplexRuntimeObserver>,
        entered: oneshot::Sender<()>,
    ) -> DirectionOutcome;
    async fn stop_pcm(
        &self,
        owner: &mut Self::Acquisition,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError>;
    async fn close_provider(
        &self,
        owner: &mut Self::Acquisition,
        reason: CloseRequestReason,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError>;
    fn discard_provider(&self, owner: &mut Self::Acquisition);
    fn is_clean(owner: &Self::Acquisition) -> bool;
    async fn wait_ready<R: crate::SidecarRuntime>(
        &self,
        supervisor: &SidecarSupervisor<R>,
    ) -> Result<(), DuplexRuntimeError>;
    async fn probe_generation<R: crate::SidecarRuntime>(
        &self,
        supervisor: &SidecarSupervisor<R>,
    ) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueuedPlaybackMetadata {
    utterance_id: uuid::Uuid,
    sequence: u64,
    provider_monotonic_ns: u64,
    enqueued_monotonic_ns: u64,
}

enum DirectionResource<A, P> {
    Prepared(P),
    Cleanup(A),
    Transitioning,
}

struct DirectionCellState<A, P> {
    direction: AudioDirection,
    session_id: uuid::Uuid,
    registered: bool,
    resource: DirectionResource<A, P>,
}

type DirectionCell<A, P> = Rc<AsyncMutex<DirectionCellState<A, P>>>;

struct DirectionWorker<A, P> {
    epoch: u64,
    launch: DirectionLaunch,
    stop: watch::Sender<Option<WorkerStop>>,
    cell: DirectionCell<A, P>,
    task_id: Option<TaskId>,
}

struct WorkerDone {
    direction: AudioDirection,
    epoch: u64,
    outcome: DirectionOutcome,
}

type JoinedWorker = Result<(TaskId, WorkerDone), JoinError>;

struct PreparedCandidate<A, P> {
    direction: AudioDirection,
    epoch: u64,
    launch: DirectionLaunch,
    cell: DirectionCell<A, P>,
}

struct PrepareBatchFailure {
    direction: AudioDirection,
    scope: FaultScope,
    cleanup_pending: bool,
}

struct DuplexCoordinator<R: crate::SidecarRuntime, E: DirectionEffects> {
    desired: DuplexLaunch,
    supervisor: SidecarSupervisor<R>,
    effects: E,
    observer: Arc<dyn DuplexRuntimeObserver>,
    lifecycle: Option<(u64, Arc<dyn DuplexCompletionObserver>)>,
    workers: HashMap<AudioDirection, DirectionWorker<E::Acquisition, E::Prepared>>,
    cleanup_owners: Vec<DirectionCell<E::Acquisition, E::Prepared>>,
    tasks: JoinSet<WorkerDone>,
    pending_events: VecDeque<JoinedWorker>,
    direction_faults: HashMap<AudioDirection, usize>,
    paused: HashSet<AudioDirection>,
    shared_faults: usize,
    retiring_generation: Option<uuid::Uuid>,
    next_epoch: u64,
    direction_epochs: HashMap<AudioDirection, u64>,
}

struct CoordinatorStartFailure {
    error: DuplexRuntimeError,
    cleanup: Option<DuplexCoordinator<ProcessSidecarRuntime, ProcessDirectionEffects>>,
}

const FAULT_BACKOFF: [Duration; MAX_RUNTIME_RESTARTS] = [
    Duration::from_millis(50),
    Duration::from_millis(100),
    Duration::from_millis(200),
];

#[allow(clippy::too_many_arguments)]
async fn run_process_duplex(
    config: ProcessDuplexConfig,
    launch: DuplexLaunch,
    stop: watch::Receiver<Option<Instant>>,
    commands: mpsc::Receiver<RuntimeCommand>,
    ack: std_mpsc::SyncSender<StartAck>,
    observer: Arc<dyn DuplexRuntimeObserver>,
    completion: Option<&(u64, Arc<dyn DuplexCompletionObserver>)>,
    deadline: Instant,
) -> Result<(), DuplexRuntimeError> {
    let lifecycle = completion.map(|(generation, observer)| (*generation, observer.clone()));
    let coordinator =
        match DuplexCoordinator::start(config, launch, observer, lifecycle, deadline).await {
            Ok(coordinator) => coordinator,
            Err(failure) => {
                let CoordinatorStartFailure { error, cleanup } = failure;
                let Some(coordinator) = cleanup else {
                    let _ = ack.send(StartAck::Rejected(error));
                    return Err(error);
                };
                let _ = ack.send(StartAck::CleanupPending(error));
                return run_coordinator(
                    coordinator,
                    stop,
                    commands,
                    completion.cloned(),
                    Some(error),
                )
                .await;
            }
        };
    let _ = ack.send(StartAck::Ready);
    run_coordinator(coordinator, stop, commands, completion.cloned(), None).await
}

async fn run_coordinator<R: crate::SidecarRuntime, E: DirectionEffects>(
    mut coordinator: DuplexCoordinator<R, E>,
    mut stop: watch::Receiver<Option<Instant>>,
    mut commands: mpsc::Receiver<RuntimeCommand>,
    completion: Option<(u64, Arc<dyn DuplexCompletionObserver>)>,
    mut terminal: Option<DuplexRuntimeError>,
) -> Result<(), DuplexRuntimeError> {
    let mut stop_open = true;
    loop {
        if terminal.is_none()
            && let Some(joined) = coordinator.pending_events.pop_front()
        {
            if let Err(error) = coordinator
                .handle_joined_worker(joined, Instant::now() + DIRECTION_CLEANUP_BUDGET)
                .await
            {
                start_terminal_cleanup(&completion, error);
                if coordinator
                    .shutdown(Instant::now() + RUNTIME_CLEANUP_BUDGET)
                    .await
                    .is_ok()
                {
                    return Err(error);
                }
                terminal = Some(error);
            }
            continue;
        }
        tokio::select! {
            biased;
            joined = coordinator.tasks.join_next_with_id(), if terminal.is_none() && !coordinator.tasks.is_empty() => {
                if let Some(joined) = joined
                    && let Err(error) = coordinator
                        .handle_joined_worker(joined, Instant::now() + DIRECTION_CLEANUP_BUDGET)
                        .await
                {
                    start_terminal_cleanup(&completion, error);
                    if coordinator
                        .shutdown(Instant::now() + RUNTIME_CLEANUP_BUDGET)
                        .await
                        .is_ok()
                    {
                        return Err(error);
                    }
                    terminal = Some(error);
                }
            }
            deadline = wait_for_stop(&mut stop), if stop_open => {
                stop_open = false;
                match coordinator.shutdown(deadline).await {
                    Ok(()) => return terminal.map_or(Ok(()), Err),
                    Err(_) if terminal.is_none() => {
                        start_terminal_cleanup(&completion, DuplexRuntimeError::StopFailed);
                        terminal = Some(DuplexRuntimeError::StopFailed);
                    }
                    Err(_) => {}
                }
            }
            command = commands.recv() => match command {
                Some(RuntimeCommand::Reconfigure { launch, deadline, response }) => {
                    let result = if terminal.is_none() {
                        coordinator.reconfigure(launch, deadline).await
                    } else {
                        Err(DuplexRuntimeError::RestoreFailed)
                    };
                    if result == Err(DuplexRuntimeError::RestoreFailed) && terminal.is_none() {
                        start_terminal_cleanup(&completion, DuplexRuntimeError::RestoreFailed);
                        terminal = Some(DuplexRuntimeError::RestoreFailed);
                    }
                    let _ = response.send(result);
                }
                Some(RuntimeCommand::Stop { deadline, response }) => {
                    let result = coordinator.shutdown(deadline).await;
                    let complete = result.is_ok();
                    let _ = response.send(result);
                    if complete {
                        return terminal.map_or(Ok(()), Err);
                    }
                    if terminal.is_none() {
                        start_terminal_cleanup(&completion, DuplexRuntimeError::StopFailed);
                        terminal = Some(DuplexRuntimeError::StopFailed);
                    }
                }
                None => {
                    let result = coordinator
                        .shutdown(Instant::now() + RUNTIME_CLEANUP_BUDGET)
                        .await;
                    if result.is_ok() {
                        return terminal.map_or(Ok(()), Err);
                    }
                    if terminal.is_none() {
                        start_terminal_cleanup(&completion, DuplexRuntimeError::StopFailed);
                        terminal = Some(DuplexRuntimeError::StopFailed);
                    }
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
            },
        }
    }
}

fn start_terminal_cleanup(
    completion: &Option<(u64, Arc<dyn DuplexCompletionObserver>)>,
    failure: DuplexRuntimeError,
) {
    if let Some((generation, observer)) = completion {
        observer.cleanup_started(*generation);
    }
    let _ = failure;
}

impl DuplexCoordinator<ProcessSidecarRuntime, ProcessDirectionEffects> {
    async fn start(
        config: ProcessDuplexConfig,
        desired: DuplexLaunch,
        observer: Arc<dyn DuplexRuntimeObserver>,
        lifecycle: Option<(u64, Arc<dyn DuplexCompletionObserver>)>,
        deadline: Instant,
    ) -> Result<Self, CoordinatorStartFailure> {
        let runtime = ProcessSidecarRuntime::new(
            config.python.clone(),
            config.sidecar_root.clone(),
            config.socket_path.clone(),
            config.expected_uid,
        )
        .map_err(|_| CoordinatorStartFailure {
            error: DuplexRuntimeError::StartFailed,
            cleanup: None,
        })?;
        let effects = ProcessDirectionEffects::new(config.clone());
        let mut coordinator = Self::with_dependencies(
            config,
            desired,
            SidecarSupervisor::new(runtime),
            effects,
            observer,
            lifecycle,
        );
        if let Err(error) = coordinator.start_resources(deadline).await {
            if coordinator.has_pending_cleanup() {
                return Err(CoordinatorStartFailure {
                    error,
                    cleanup: Some(coordinator),
                });
            }
            let cleanup = coordinator
                .shutdown(deadline)
                .await
                .err()
                .map(|_| coordinator);
            return Err(CoordinatorStartFailure { error, cleanup });
        }
        Ok(coordinator)
    }
}

impl<R: crate::SidecarRuntime, E: DirectionEffects> DuplexCoordinator<R, E> {
    fn with_dependencies(
        _config: ProcessDuplexConfig,
        desired: DuplexLaunch,
        supervisor: SidecarSupervisor<R>,
        effects: E,
        observer: Arc<dyn DuplexRuntimeObserver>,
        lifecycle: Option<(u64, Arc<dyn DuplexCompletionObserver>)>,
    ) -> Self {
        Self {
            desired,
            supervisor,
            effects,
            observer,
            lifecycle,
            workers: HashMap::new(),
            cleanup_owners: Vec::new(),
            tasks: JoinSet::new(),
            pending_events: VecDeque::new(),
            direction_faults: HashMap::new(),
            paused: HashSet::new(),
            shared_faults: 0,
            retiring_generation: None,
            next_epoch: 0,
            direction_epochs: HashMap::new(),
        }
    }

    async fn start_resources(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        deadline_open(deadline, DuplexRuntimeError::StartFailed)?;
        tokio::time::timeout_at(deadline, self.supervisor.start())
            .await
            .map_err(|_| DuplexRuntimeError::StartFailed)?
            .map_err(|error| {
                tracing::error!(event = "provider_start_failed", code = ?error);
                DuplexRuntimeError::StartFailed
            })?;
        deadline_open(deadline, DuplexRuntimeError::StartFailed)?;
        tokio::time::timeout_at(deadline, self.effects.wait_ready(&self.supervisor))
            .await
            .map_err(|_| DuplexRuntimeError::StartFailed)??;
        deadline_open(deadline, DuplexRuntimeError::StartFailed)?;
        let batch = self
            .prepare_batch(self.desired.launches(), deadline)
            .await
            .map_err(|failure| {
                prepare_batch_error(
                    failure,
                    DuplexRuntimeError::StartFailed,
                    DuplexRuntimeError::StopFailed,
                )
            })?;
        self.activate_batch(batch, deadline).await?;
        Ok(())
    }

    async fn prepare_batch(
        &mut self,
        launches: Vec<DirectionLaunch>,
        deadline: Instant,
    ) -> Result<Vec<PreparedCandidate<E::Acquisition, E::Prepared>>, PrepareBatchFailure> {
        let mut prepared = Vec::with_capacity(launches.len());
        for launch in launches {
            let direction = launch.runtime.direction;
            let epoch = match self.reserve_epoch(direction) {
                Ok(epoch) => epoch,
                Err(_) => {
                    let pending = self
                        .compensate_candidates(
                            prepared,
                            CloseRequestReason::ProviderSwitch,
                            deadline,
                        )
                        .await;
                    return Err(PrepareBatchFailure {
                        direction,
                        scope: FaultScope::Local,
                        cleanup_pending: pending,
                    });
                }
            };
            match self
                .prepare_candidate(launch.clone(), epoch, deadline)
                .await
            {
                Ok(candidate) => prepared.push(candidate),
                Err((scope, cell)) => {
                    prepared.push(PreparedCandidate {
                        direction,
                        epoch,
                        launch,
                        cell,
                    });
                    let pending = self
                        .compensate_candidates(
                            prepared,
                            CloseRequestReason::ProviderSwitch,
                            deadline,
                        )
                        .await;
                    return Err(PrepareBatchFailure {
                        direction,
                        scope,
                        cleanup_pending: pending,
                    });
                }
            }
        }
        Ok(prepared)
    }

    async fn prepare_candidate(
        &mut self,
        launch: DirectionLaunch,
        epoch: u64,
        deadline: Instant,
    ) -> Result<
        PreparedCandidate<E::Acquisition, E::Prepared>,
        (FaultScope, DirectionCell<E::Acquisition, E::Prepared>),
    > {
        let owner = self.effects.begin(launch.clone());
        let session_id = E::session_id(&owner);
        let cell = Rc::new(AsyncMutex::new(DirectionCellState {
            direction: launch.runtime.direction,
            session_id,
            registered: false,
            resource: DirectionResource::Cleanup(owner),
        }));
        self.cleanup_owners.push(cell.clone());
        let Some(sidecar) = self.supervisor.launch().cloned() else {
            return Err((
                classify_direction_failure(DirectionFailureOrigin::SidecarExit),
                cell,
            ));
        };
        {
            let mut state = cell.lock().await;
            let DirectionResource::Cleanup(owner) = &mut state.resource else {
                unreachable!()
            };
            if deadline_open(deadline, ()).is_err() {
                drop(state);
                return Err((FaultScope::Local, cell));
            }
            if let Err(scope) = self.effects.prepare(owner, &sidecar, deadline).await {
                drop(state);
                return Err((scope, cell));
            }
        }
        if deadline_open(deadline, ()).is_err() {
            return Err((FaultScope::Local, cell));
        }
        if self.supervisor.register_session(session_id).is_err() {
            return Err((FaultScope::Shared, cell));
        }
        let mut state = cell.lock().await;
        state.registered = true;
        if deadline_open(deadline, ()).is_err() {
            drop(state);
            return Err((FaultScope::Local, cell));
        }
        let DirectionResource::Cleanup(owner) =
            std::mem::replace(&mut state.resource, DirectionResource::Transitioning)
        else {
            unreachable!()
        };
        match self.effects.finish(owner) {
            Ok(prepared) => state.resource = DirectionResource::Prepared(prepared),
            Err(owner) => {
                state.resource = DirectionResource::Cleanup(owner);
                drop(state);
                return Err((FaultScope::Local, cell));
            }
        }
        drop(state);
        Ok(PreparedCandidate {
            direction: launch.runtime.direction,
            epoch,
            launch,
            cell,
        })
    }

    async fn reconfigure(
        &mut self,
        mut candidate: DuplexLaunch,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        if self.has_pending_cleanup() {
            return Err(DuplexRuntimeError::RestoreFailed);
        }
        let previous = self.desired.clone();
        let affected = previous.changed_directions(&candidate);
        if !affected.contains(&AudioDirection::Microphone)
            && let Some(microphone) = candidate.microphone.as_mut()
        {
            microphone.aec_reservation = None;
        }
        let previous_active = affected
            .iter()
            .filter_map(|direction| {
                self.workers
                    .get(direction)
                    .filter(|worker| worker.task_id.is_some())
                    .map(|worker| worker.launch.clone())
            })
            .collect::<Vec<_>>();
        let previous_paused = self.paused.clone();
        let previous_faults = self.direction_faults.clone();
        let replacements = affected
            .iter()
            .filter_map(|direction| candidate.for_direction(*direction).cloned())
            .collect();
        let prepared = match self.prepare_batch(replacements, deadline).await {
            Ok(prepared) => prepared,
            Err(failure) => {
                return Err(prepare_batch_error(
                    failure,
                    DuplexRuntimeError::ReconfigureFailed,
                    DuplexRuntimeError::RestoreFailed,
                ));
            }
        };
        if self
            .stop_workers(
                &affected,
                Vec::new(),
                CloseRequestReason::ProviderSwitch,
                deadline,
            )
            .await
            .is_err()
        {
            self.compensate_candidates(prepared, CloseRequestReason::ProviderSwitch, deadline)
                .await;
            return Err(DuplexRuntimeError::RestoreFailed);
        }
        if self.activate_batch(prepared, deadline).await.is_err() {
            let candidate_clean = self
                .stop_workers(
                    &affected,
                    Vec::new(),
                    CloseRequestReason::ProviderSwitch,
                    deadline,
                )
                .await
                .is_ok();
            let restored = if candidate_clean {
                match self.prepare_batch(previous_active, deadline).await {
                    Ok(previous) => self.activate_batch(previous, deadline).await.is_ok(),
                    Err(_) => false,
                }
            } else {
                false
            };
            self.paused = previous_paused;
            self.direction_faults = previous_faults;
            return Err(if restored {
                DuplexRuntimeError::ReconfigureFailed
            } else {
                DuplexRuntimeError::RestoreFailed
            });
        }
        for direction in &affected {
            if candidate.for_direction(*direction).is_none() {
                let epoch = self.reserve_epoch(*direction)?;
                self.report_direction(*direction, epoch, DirectionRuntimeStatus::Stopped, None);
            }
            self.direction_faults.remove(direction);
            self.paused.remove(direction);
        }
        self.shared_faults = 0;
        self.desired = candidate;
        Ok(())
    }

    async fn activate_batch(
        &mut self,
        prepared: Vec<PreparedCandidate<E::Acquisition, E::Prepared>>,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        deadline_open(deadline, DuplexRuntimeError::StartFailed)?;
        let mut activated = Vec::with_capacity(prepared.len());
        let mut entered = Vec::with_capacity(prepared.len());
        for candidate in prepared {
            deadline_open(deadline, DuplexRuntimeError::StartFailed)?;
            let PreparedCandidate {
                direction,
                epoch,
                launch,
                cell,
            } = candidate;
            let (stop, mut stop_receiver) = watch::channel(None);
            let task_cell = cell.clone();
            let effects = self.effects.clone();
            let observer = self.observer.clone();
            let (entered_sender, entered_receiver) = oneshot::channel();
            let handle = self.tasks.spawn_local(async move {
                let mut state = task_cell.lock().await;
                let outcome = match &mut state.resource {
                    DirectionResource::Prepared(prepared) => {
                        effects
                            .run(prepared, &mut stop_receiver, observer, entered_sender)
                            .await
                    }
                    DirectionResource::Cleanup(_) | DirectionResource::Transitioning => {
                        DirectionOutcome::Fault(classify_direction_failure(
                            DirectionFailureOrigin::InternalTransport,
                        ))
                    }
                };
                let resource =
                    std::mem::replace(&mut state.resource, DirectionResource::Transitioning);
                if let DirectionResource::Prepared(prepared) = resource {
                    state.resource = DirectionResource::Cleanup(effects.recover_owner(prepared));
                } else {
                    state.resource = resource;
                }
                WorkerDone {
                    direction,
                    epoch,
                    outcome,
                }
            });
            let task_id = handle.id();
            self.workers.insert(
                direction,
                DirectionWorker {
                    epoch,
                    launch,
                    stop,
                    cell: cell.clone(),
                    task_id: Some(task_id),
                },
            );
            self.remove_retained_cell(&cell);
            activated.push((direction, epoch));
            entered.push(entered_receiver);
        }
        for entered in entered {
            tokio::time::timeout_at(deadline, entered)
                .await
                .map_err(|_| DuplexRuntimeError::StartFailed)?
                .map_err(|_| DuplexRuntimeError::StartFailed)?;
            deadline_open(deadline, DuplexRuntimeError::StartFailed)?;
        }
        for (direction, epoch) in activated {
            self.report_direction(direction, epoch, DirectionRuntimeStatus::Running, None);
        }
        Ok(())
    }

    #[cfg(test)]
    async fn handle_next_worker_completion(
        &mut self,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let joined = match self.pending_events.pop_front() {
            Some(joined) => joined,
            None => self
                .tasks
                .join_next_with_id()
                .await
                .ok_or(DuplexRuntimeError::StopFailed)?,
        };
        self.handle_joined_worker(joined, deadline).await
    }

    async fn handle_joined_worker(
        &mut self,
        joined: JoinedWorker,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let (direction, outcome) = match self.consume_joined(joined, deadline).await {
            Ok(joined) => joined,
            Err((direction, error)) => {
                let _ = self
                    .cleanup_worker(direction, CloseRequestReason::ProviderSwitch, deadline)
                    .await;
                return Err(error);
            }
        };
        let launch = self
            .workers
            .get(&direction)
            .map(|worker| worker.launch.clone())
            .ok_or(DuplexRuntimeError::StartFailed)?;
        match outcome {
            DirectionOutcome::ModeChanged => {
                self.cleanup_worker(direction, CloseRequestReason::ProviderSwitch, deadline)
                    .await?;
                let mut launch = launch;
                refresh_launch_mode(&mut launch, self.observer.as_ref());
                self.desired.set_direction(direction, Some(launch.clone()));
                let batch =
                    self.prepare_batch(vec![launch], deadline)
                        .await
                        .map_err(|failure| {
                            prepare_batch_error(
                                failure,
                                DuplexRuntimeError::StartFailed,
                                DuplexRuntimeError::StopFailed,
                            )
                        })?;
                self.activate_batch(batch, deadline).await
            }
            DirectionOutcome::Stopped(WorkerStop::Close { reason, deadline }) => {
                self.cleanup_worker(direction, reason, deadline).await
            }
            DirectionOutcome::Stopped(WorkerStop::GenerationLost { .. }) => {
                Err(DuplexRuntimeError::StartFailed)
            }
            DirectionOutcome::Fault(scope) => {
                match self.resolve_fault_scope(scope, deadline).await? {
                    FaultScope::Local => self.recover_local(direction, launch, deadline).await,
                    FaultScope::Shared => self.restart_shared(deadline).await,
                    FaultScope::ProviderConnection => unreachable!(),
                }
            }
        }
    }

    async fn recover_local(
        &mut self,
        direction: AudioDirection,
        launch: DirectionLaunch,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        self.cleanup_worker(direction, CloseRequestReason::ProviderSwitch, deadline)
            .await?;
        self.recover_launch(direction, launch, deadline).await
    }

    async fn recover_launch(
        &mut self,
        direction: AudioDirection,
        launch: DirectionLaunch,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        loop {
            let Some(delay) = next_fault_delay(&mut self.direction_faults, direction) else {
                let epoch = self.reserve_epoch(direction)?;
                self.report_direction(
                    direction,
                    epoch,
                    DirectionRuntimeStatus::Failed,
                    Some(DirectionRuntimeFailure::RestartExhausted),
                );
                self.paused.insert(direction);
                return Ok(());
            };
            let epoch = self.reserve_epoch(direction)?;
            self.report_direction(direction, epoch, DirectionRuntimeStatus::Recovering, None);
            tokio::time::timeout_at(deadline, tokio::time::sleep(delay))
                .await
                .map_err(|_| DuplexRuntimeError::StopFailed)?;
            match self
                .prepare_candidate(launch.clone(), epoch, deadline)
                .await
            {
                Ok(candidate) => {
                    return self.activate_batch(vec![candidate], deadline).await;
                }
                Err((scope, cell)) => {
                    let clean = self
                        .cleanup_detached(cell, CloseRequestReason::ProviderSwitch, deadline)
                        .await;
                    if !clean {
                        return Err(DuplexRuntimeError::StopFailed);
                    }
                    match self.resolve_fault_scope(scope, deadline).await? {
                        FaultScope::Local => {}
                        FaultScope::Shared => return Box::pin(self.restart_shared(deadline)).await,
                        FaultScope::ProviderConnection => unreachable!(),
                    }
                }
            }
        }
    }

    async fn restart_shared(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        let Some(delay) = FAULT_BACKOFF.get(self.shared_faults).copied() else {
            return Err(DuplexRuntimeError::StartFailed);
        };
        self.shared_faults += 1;
        let attempt =
            NonZeroU32::new(self.shared_faults as u32).ok_or(DuplexRuntimeError::StartFailed)?;
        self.observer
            .observe(DuplexRuntimeEvent::GenerationRestart { attempt });
        let launches = self
            .desired
            .launches()
            .into_iter()
            .filter(|launch| !self.paused.contains(&launch.runtime.direction))
            .collect::<Vec<_>>();
        for direction in launches.iter().map(|launch| launch.runtime.direction) {
            let epoch = self.reserve_epoch(direction)?;
            self.report_direction(direction, epoch, DirectionRuntimeStatus::Recovering, None);
        }
        self.quiesce_generation(deadline).await?;
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        tokio::time::timeout_at(deadline, tokio::time::sleep(delay))
            .await
            .map_err(|_| DuplexRuntimeError::StopFailed)?;
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        let old_generation = self
            .supervisor
            .launch()
            .map(|launch| launch.generation_id)
            .ok_or(DuplexRuntimeError::StartFailed)?;
        self.retiring_generation = Some(old_generation);
        let restart = tokio::time::timeout_at(deadline, self.supervisor.restart_generation()).await;
        let restart = match restart {
            Ok(result) => result.map_err(|_| DuplexRuntimeError::StartFailed),
            Err(_) => Err(DuplexRuntimeError::StopFailed),
        };
        if self
            .retire_generation(old_generation, deadline)
            .await
            .is_err()
        {
            return Err(restart.err().unwrap_or(DuplexRuntimeError::StopFailed));
        }
        restart?;
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        tokio::time::timeout_at(deadline, self.effects.wait_ready(&self.supervisor))
            .await
            .map_err(|_| DuplexRuntimeError::StopFailed)??;
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        match self.prepare_batch(launches, deadline).await {
            Ok(batch) => self.activate_batch(batch, deadline).await,
            Err(failure) if failure.cleanup_pending => Err(prepare_batch_error(
                failure,
                DuplexRuntimeError::StartFailed,
                DuplexRuntimeError::StopFailed,
            )),
            Err(failure) => {
                let direction = failure.direction;
                let scope = self.resolve_fault_scope(failure.scope, deadline).await?;
                if scope == FaultScope::Shared {
                    return Err(prepare_batch_error(
                        failure,
                        DuplexRuntimeError::StartFailed,
                        DuplexRuntimeError::StopFailed,
                    ));
                }
                let launch = self
                    .desired
                    .for_direction(direction)
                    .cloned()
                    .ok_or(DuplexRuntimeError::StartFailed)?;
                let peers = self
                    .desired
                    .launches()
                    .into_iter()
                    .filter(|candidate| {
                        candidate.runtime.direction != direction
                            && !self.paused.contains(&candidate.runtime.direction)
                    })
                    .collect();
                let peers = self
                    .prepare_batch(peers, deadline)
                    .await
                    .map_err(|failure| {
                        prepare_batch_error(
                            failure,
                            DuplexRuntimeError::StartFailed,
                            DuplexRuntimeError::StopFailed,
                        )
                    })?;
                self.activate_batch(peers, deadline).await?;
                self.recover_launch(direction, launch, deadline).await
            }
        }
    }

    async fn join_workers(
        &mut self,
        mut pending: HashMap<TaskId, oneshot::Sender<()>>,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let mut failure = None;
        while !pending.is_empty() {
            let queued = self
                .pending_events
                .iter()
                .position(|joined| pending.contains_key(&joined_id(joined)))
                .and_then(|index| self.pending_events.remove(index));
            let joined = match queued {
                Some(joined) => joined,
                None => {
                    match tokio::time::timeout_at(deadline, self.tasks.join_next_with_id()).await {
                        Ok(Some(joined)) => joined,
                        _ => {
                            failure.get_or_insert(DuplexRuntimeError::StopFailed);
                            break;
                        }
                    }
                }
            };
            if let Some(permit) = pending.remove(&joined_id(&joined)) {
                if let Err((_, error)) = self.consume_joined(joined, deadline).await {
                    failure.get_or_insert(error);
                }
                if Instant::now() < deadline {
                    let _ = permit.send(());
                }
            } else {
                self.pending_events.push_back(joined);
            }
        }
        failure.map_or(Ok(()), Err)
    }

    async fn consume_joined(
        &mut self,
        joined: JoinedWorker,
        deadline: Instant,
    ) -> Result<(AudioDirection, DirectionOutcome), (AudioDirection, DuplexRuntimeError)> {
        let task_id = joined_id(&joined);
        let direction = self
            .workers
            .iter()
            .find_map(|(direction, worker)| (worker.task_id == Some(task_id)).then_some(*direction))
            .ok_or((AudioDirection::Microphone, DuplexRuntimeError::StartFailed))?;
        let (expected_epoch, cell) = {
            let worker = self
                .workers
                .get_mut(&direction)
                .expect("worker was resolved");
            worker.task_id = None;
            (worker.epoch, worker.cell.clone())
        };
        self.observer.reset_direction(direction);
        match joined {
            Ok((_id, done)) if done.direction == direction && done.epoch == expected_epoch => {
                Ok((direction, done.outcome))
            }
            Ok(_) | Err(_) => {
                let _ = Self::transition_cell(&self.effects, &cell, deadline).await;
                Err((direction, DuplexRuntimeError::StartFailed))
            }
        }
    }

    async fn cleanup_worker(
        &mut self,
        direction: AudioDirection,
        reason: CloseRequestReason,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        self.stop_workers(&[direction], Vec::new(), reason, deadline)
            .await
    }

    async fn stop_all(
        &mut self,
        reason: CloseRequestReason,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let directions = self.workers.keys().copied().collect::<Vec<_>>();
        self.stop_workers(&directions, self.cleanup_owners.clone(), reason, deadline)
            .await
    }

    async fn quiesce_generation(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        let directions = self.workers.keys().copied().collect::<Vec<_>>();
        self.quiesce_cells(
            &directions,
            self.cleanup_owners.clone(),
            WorkerStop::GenerationLost { deadline },
            deadline,
        )
        .await
        .map(|_| ())
    }

    async fn quiesce_cells(
        &mut self,
        directions: &[AudioDirection],
        retained: Vec<DirectionCell<E::Acquisition, E::Prepared>>,
        stop: WorkerStop,
        deadline: Instant,
    ) -> Result<Vec<DirectionCell<E::Acquisition, E::Prepared>>, DuplexRuntimeError> {
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        let mut cells = Vec::new();
        for cell in directions
            .iter()
            .filter_map(|direction| {
                self.workers
                    .get(direction)
                    .map(|worker| worker.cell.clone())
            })
            .chain(retained)
        {
            if !cells.iter().any(|owned| Rc::ptr_eq(owned, &cell)) {
                cells.push(cell);
            }
        }
        for direction in directions {
            if let Some(worker) = self.workers.get(direction)
                && worker.task_id.is_some()
            {
                let _ = worker.stop.send(Some(stop));
            }
        }
        let mut pending = HashMap::new();
        let permits = cells
            .iter()
            .map(|cell| {
                let task_id = directions.iter().find_map(|direction| {
                    self.workers
                        .get(direction)
                        .filter(|worker| Rc::ptr_eq(&worker.cell, cell))
                        .and_then(|worker| worker.task_id)
                });
                task_id.map(|task_id| {
                    let (sender, receiver) = oneshot::channel();
                    pending.insert(task_id, sender);
                    receiver
                })
            })
            .collect::<Vec<_>>();
        let effects = self.effects.clone();
        let local =
            futures_util::future::join_all(cells.iter().zip(permits).map(|(cell, permit)| {
                let effects = &effects;
                async move {
                    tokio::time::timeout_at(deadline, async {
                        if let Some(permit) = permit {
                            permit.await.map_err(|_| DuplexRuntimeError::StopFailed)?;
                        }
                        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
                        Self::transition_cell(effects, cell, deadline).await?;
                        let mut state = cell.lock().await;
                        let DirectionResource::Cleanup(owner) = &mut state.resource else {
                            return Err(DuplexRuntimeError::StopFailed);
                        };
                        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
                        effects.stop_pcm(owner, deadline).await
                    })
                    .await
                    .map_err(|_| DuplexRuntimeError::StopFailed)?
                }
            }));
        let (joined, local) = tokio::join!(self.join_workers(pending, deadline), local);
        joined?;
        match local.into_iter().find_map(Result::err) {
            Some(error) => Err(error),
            None => Ok(cells),
        }
    }

    async fn stop_workers(
        &mut self,
        directions: &[AudioDirection],
        retained: Vec<DirectionCell<E::Acquisition, E::Prepared>>,
        reason: CloseRequestReason,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let cells = self
            .quiesce_cells(
                directions,
                retained,
                WorkerStop::Close { reason, deadline },
                deadline,
            )
            .await?;
        let mut failure = None;
        for cell in cells {
            if self.close_cell_provider(&cell, reason, deadline).await {
                for direction in directions {
                    if self
                        .workers
                        .get(direction)
                        .is_some_and(|worker| Rc::ptr_eq(&worker.cell, &cell))
                    {
                        self.workers.remove(direction);
                    }
                }
                self.remove_retained_cell(&cell);
            } else {
                failure.get_or_insert(DuplexRuntimeError::StopFailed);
            }
        }
        failure.map_or(Ok(()), Err)
    }

    async fn retire_generation(
        &mut self,
        generation: uuid::Uuid,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let reaped = self
            .supervisor
            .generation_retirement(generation)
            .is_some_and(|receipt| receipt.old_generation_reaped);
        if !reaped {
            return Err(DuplexRuntimeError::StopFailed);
        }
        self.discard_all_providers(deadline).await?;
        self.supervisor
            .acknowledge_generation_retirement(generation)
            .map_err(|_| DuplexRuntimeError::StopFailed)?;
        self.retiring_generation = None;
        Ok(())
    }

    async fn shutdown(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        let _ = self
            .stop_all(CloseRequestReason::DaemonShutdown, deadline)
            .await;
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        let sidecar = tokio::time::timeout_at(deadline, self.supervisor.shutdown())
            .await
            .map_err(|_| DuplexRuntimeError::StopFailed)?
            .map_err(|_| DuplexRuntimeError::StopFailed);
        let retired = match self.retiring_generation {
            Some(generation) => self.retire_generation(generation, deadline).await,
            None if sidecar.is_ok() => self.discard_all_providers(deadline).await,
            None => Err(DuplexRuntimeError::StopFailed),
        };
        if sidecar.is_err() || retired.is_err() || self.has_pending_cleanup() {
            Err(DuplexRuntimeError::StopFailed)
        } else {
            for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
                let epoch = self.reserve_epoch(direction)?;
                self.report_direction(direction, epoch, DirectionRuntimeStatus::Stopped, None);
            }
            Ok(())
        }
    }

    async fn compensate_candidates(
        &mut self,
        _candidates: Vec<PreparedCandidate<E::Acquisition, E::Prepared>>,
        reason: CloseRequestReason,
        deadline: Instant,
    ) -> bool {
        self.stop_workers(&[], self.cleanup_owners.clone(), reason, deadline)
            .await
            .is_err()
    }

    async fn cleanup_detached(
        &mut self,
        cell: DirectionCell<E::Acquisition, E::Prepared>,
        reason: CloseRequestReason,
        deadline: Instant,
    ) -> bool {
        if !self
            .cleanup_owners
            .iter()
            .any(|owned| Rc::ptr_eq(owned, &cell))
        {
            self.cleanup_owners.push(cell.clone());
        }
        if self
            .stop_workers(&[], vec![cell.clone()], reason, deadline)
            .await
            .is_ok()
        {
            true
        } else {
            let direction = cell.try_lock().ok().map(|state| state.direction);
            tracing::warn!(event = "direction_cleanup_pending", ?direction);
            false
        }
    }

    fn remove_retained_cell(&mut self, cell: &DirectionCell<E::Acquisition, E::Prepared>) {
        self.cleanup_owners.retain(|owned| !Rc::ptr_eq(owned, cell));
    }

    async fn transition_cell(
        effects: &E,
        cell: &DirectionCell<E::Acquisition, E::Prepared>,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let mut state = tokio::time::timeout_at(deadline, cell.lock())
            .await
            .map_err(|_| DuplexRuntimeError::StopFailed)?;
        let resource = std::mem::replace(&mut state.resource, DirectionResource::Transitioning);
        state.resource = match resource {
            DirectionResource::Prepared(prepared) => {
                DirectionResource::Cleanup(effects.recover_owner(prepared))
            }
            resource => resource,
        };
        Ok(())
    }

    async fn close_cell_provider(
        &mut self,
        cell: &DirectionCell<E::Acquisition, E::Prepared>,
        reason: CloseRequestReason,
        deadline: Instant,
    ) -> bool {
        if Instant::now() >= deadline {
            return false;
        }
        let Ok(mut state) = tokio::time::timeout_at(deadline, cell.lock()).await else {
            return false;
        };
        let provider = {
            let DirectionResource::Cleanup(owner) = &mut state.resource else {
                return false;
            };
            if self.retiring_generation.is_none() && Instant::now() < deadline {
                tokio::time::timeout_at(
                    deadline,
                    self.effects.close_provider(owner, reason, deadline),
                )
                .await
                .unwrap_or(Err(DuplexRuntimeError::StopFailed))
            } else if self.retiring_generation.is_none() {
                Err(DuplexRuntimeError::StopFailed)
            } else {
                Ok(())
            }
        };
        if self.retiring_generation.is_none()
            && provider.is_ok()
            && state.registered
            && Instant::now() < deadline
            && tokio::time::timeout_at(
                deadline,
                self.supervisor
                    .close_session(state.session_id, std::future::ready(())),
            )
            .await
            .is_ok_and(|result| result.is_ok())
        {
            state.registered = false;
        }
        let clean = matches!(
            &state.resource,
            DirectionResource::Cleanup(owner) if E::is_clean(owner)
        );
        provider.is_ok() && !state.registered && clean
    }

    async fn discard_all_providers(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        let cells = self
            .workers
            .values()
            .map(|worker| worker.cell.clone())
            .chain(self.cleanup_owners.iter().cloned())
            .collect::<Vec<_>>();
        for cell in cells {
            Self::transition_cell(&self.effects, &cell, deadline).await?;
            let mut state = tokio::time::timeout_at(deadline, cell.lock())
                .await
                .map_err(|_| DuplexRuntimeError::StopFailed)?;
            if let DirectionResource::Cleanup(owner) = &mut state.resource {
                self.effects.discard_provider(owner);
                state.registered = false;
            }
        }
        let clean_workers = self
            .workers
            .iter()
            .filter_map(|(direction, worker)| {
                worker
                    .cell
                    .try_lock()
                    .ok()
                    .and_then(|state| matches!(&state.resource, DirectionResource::Cleanup(owner) if E::is_clean(owner)).then_some(*direction))
            })
            .collect::<Vec<_>>();
        for direction in clean_workers {
            self.workers.remove(&direction);
        }
        self.cleanup_owners.retain(|cell| {
            cell.try_lock()
                .map_or(true, |state| !matches!(&state.resource, DirectionResource::Cleanup(owner) if E::is_clean(owner)))
        });
        Ok(())
    }

    async fn resolve_fault_scope(
        &self,
        scope: FaultScope,
        deadline: Instant,
    ) -> Result<FaultScope, DuplexRuntimeError> {
        if scope != FaultScope::ProviderConnection {
            return Ok(scope);
        }
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        let scope = match tokio::time::timeout_at(
            deadline.min(Instant::now() + Duration::from_secs(1)),
            self.effects.probe_generation(&self.supervisor),
        )
        .await
        {
            Ok(true) => FaultScope::Local,
            Ok(false) | Err(_) => FaultScope::Shared,
        };
        deadline_open(deadline, DuplexRuntimeError::StopFailed)?;
        Ok(scope)
    }

    fn reserve_epoch(&mut self, direction: AudioDirection) -> Result<u64, DuplexRuntimeError> {
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or(DuplexRuntimeError::StartFailed)?;
        self.direction_epochs.insert(direction, self.next_epoch);
        Ok(self.next_epoch)
    }

    fn report_direction(
        &self,
        direction: AudioDirection,
        epoch: u64,
        status: DirectionRuntimeStatus,
        failure: Option<DirectionRuntimeFailure>,
    ) {
        if let Some((generation, observer)) = &self.lifecycle {
            observer.direction_status_changed(*generation, direction, epoch, status, failure);
        }
    }

    fn has_pending_cleanup(&self) -> bool {
        !self.cleanup_owners.is_empty()
            || self.workers.values().any(|worker| worker.task_id.is_none())
    }

    #[cfg(test)]
    fn has_cleanup_owner(&self, direction: AudioDirection) -> bool {
        self.workers
            .get(&direction)
            .is_some_and(|worker| worker.task_id.is_none())
            || self.cleanup_owners.iter().any(|cell| {
                cell.try_lock()
                    .is_ok_and(|state| state.direction == direction)
            })
    }

    #[cfg(test)]
    fn worker_epoch(&self, direction: AudioDirection) -> Option<u64> {
        self.workers
            .get(&direction)
            .and_then(|worker| worker.task_id.map(|_| worker.epoch))
    }

    #[cfg(test)]
    fn worker_epochs(&self) -> HashMap<AudioDirection, u64> {
        self.workers
            .iter()
            .filter_map(|(direction, worker)| worker.task_id.map(|_| (*direction, worker.epoch)))
            .collect()
    }

    #[cfg(test)]
    fn direction_epoch(&self, direction: AudioDirection) -> Option<u64> {
        self.direction_epochs.get(&direction).copied()
    }

    #[cfg(test)]
    fn task_count(&self) -> usize {
        self.tasks.len()
    }
}

fn joined_id(joined: &JoinedWorker) -> TaskId {
    match joined {
        Ok((id, _)) => *id,
        Err(error) => error.id(),
    }
}

fn prepare_batch_error(
    failure: PrepareBatchFailure,
    clean_error: DuplexRuntimeError,
    pending_error: DuplexRuntimeError,
) -> DuplexRuntimeError {
    tracing::warn!(
        event = "direction_batch_prepare_failed",
        direction = ?failure.direction,
        scope = ?failure.scope,
        cleanup_pending = failure.cleanup_pending
    );
    if failure.cleanup_pending {
        pending_error
    } else {
        clean_error
    }
}

fn next_fault_delay(
    faults: &mut HashMap<AudioDirection, usize>,
    direction: AudioDirection,
) -> Option<Duration> {
    let attempts = faults.entry(direction).or_default();
    let delay = FAULT_BACKOFF.get(*attempts).copied();
    *attempts = attempts.saturating_add(1);
    delay
}

#[derive(Clone)]
struct ProcessDirectionEffects {
    config: ProcessDuplexConfig,
}

impl ProcessDirectionEffects {
    fn new(config: ProcessDuplexConfig) -> Self {
        Self { config }
    }

    async fn prepare_with_pcm_spawners<C, P>(
        &self,
        owner: &mut ProcessAcquisition,
        generation: &crate::SidecarLaunch,
        deadline: Instant,
        capture_spawner: C,
        playback_spawner: P,
    ) -> Result<(), FaultScope>
    where
        C: FnOnce(&PulsePcmCommand) -> Result<PulsePcmCapture, translator_audio::PulsePcmError>,
        P: FnOnce(&PulsePcmCommand) -> Result<PulsePcmPlayback, translator_audio::PulsePcmError>,
    {
        owner.runtime_generation = Some(generation.generation_id);
        let direction = owner.launch.runtime.direction;
        tracing::info!(event = "direction_prepare_started", direction = ?direction);
        let open_deadline = deadline.min(Instant::now() + DIRECTION_OPEN_TIMEOUT);
        if Instant::now() >= open_deadline {
            return Err(FaultScope::Local);
        }
        if let Some(reservation) = owner.launch.aec_reservation.as_ref() {
            reservation
                .consume_before_effects(open_deadline.into_std())
                .map_err(|_| {
                    direction_start_error(direction, "aec_reservation");
                    FaultScope::Local
                })?;
        }
        let provider = tokio::time::timeout_at(
            open_deadline,
            ProviderStreamClient::open(
                &self.config.socket_path,
                &generation.token,
                owner.session.open_request(),
            ),
        )
        .await
        .map_err(|_| FaultScope::ProviderConnection)?
        .map_err(|error| classify_provider_client_error(&error))?;
        owner.provider = Some(provider);
        if Instant::now() >= open_deadline {
            return Err(FaultScope::ProviderConnection);
        }
        let opened = tokio::time::timeout_at(open_deadline, async {
            let mut opened = false;
            let mut ready = false;
            while !opened || !ready {
                let event = owner
                    .provider
                    .as_mut()
                    .expect("provider owner is recorded before readiness")
                    .next_event()
                    .await
                    .map_err(|error| classify_provider_client_error(&error))?
                    .ok_or_else(classify_provider_eof)?;
                match event.event.as_ref() {
                    Some(provider_event::Event::SessionOpened(_)) => opened = true,
                    Some(provider_event::Event::Health(health)) => {
                        tracing::info!(
                            event = "provider_health_observed",
                            direction = ?direction,
                            state = health.state,
                            model_states = ?health.models.iter().map(|model| model.state).collect::<Vec<_>>()
                        );
                        ready = ProviderState::try_from(health.state)
                            .is_ok_and(provider_state_is_operational);
                        if ready && health.state == ProviderState::Degraded as i32 {
                            tracing::warn!(event = "provider_operational_degraded", direction = ?direction);
                        }
                    }
                    _ => {}
                }
                owner
                    .session
                    .handle_provider_event(&event, monotonic_ns())
                    .map_err(|_| FaultScope::Local)?;
            }
            Ok::<(), FaultScope>(())
        })
        .await;
        match opened {
            Ok(Ok(())) => {}
            Ok(Err(scope)) => {
                direction_start_error(direction, "provider_event_validation");
                return Err(scope);
            }
            Err(_) => {
                direction_start_error(direction, "provider_open_timeout");
                return Err(FaultScope::ProviderConnection);
            }
        }
        deadline_open(open_deadline, FaultScope::ProviderConnection)?;
        if let Some(reservation) = owner.launch.aec_reservation.as_ref() {
            reservation
                .confirm_before_pcm(open_deadline.into_std())
                .map_err(|_| {
                    direction_start_error(direction, "aec_reinspection");
                    FaultScope::Local
                })?;
        }
        owner.capture = Some(
            capture_spawner(&PulsePcmCommand::capture(
                &owner.launch.capture_device,
                owner.launch.capture_stream_name,
            ))
            .map_err(|_| {
                direction_start_error(direction, "capture_spawn");
                FaultScope::Local
            })?,
        );
        deadline_open(open_deadline, FaultScope::Local)?;
        owner.playback = Some(
            playback_spawner(&PulsePcmCommand::playback(
                &owner.launch.playback_device,
                owner.launch.playback_stream_name,
            ))
            .map_err(|_| {
                direction_start_error(direction, "playback_spawn");
                FaultScope::Local
            })?,
        );
        tracing::info!(event = "direction_prepare_ready", direction = ?direction);
        Ok(())
    }
}

#[allow(async_fn_in_trait)]
impl DirectionEffects for ProcessDirectionEffects {
    type Acquisition = ProcessAcquisition;
    type Prepared = PreparedDirection;

    fn begin(&self, launch: DirectionLaunch) -> Self::Acquisition {
        let session = DirectionSession::new(launch.runtime);
        ProcessAcquisition {
            launch,
            session,
            provider: None,
            capture: None,
            playback: None,
            runtime_generation: None,
        }
    }

    fn session_id(owner: &Self::Acquisition) -> uuid::Uuid {
        owner.session.session_id()
    }

    async fn prepare(
        &self,
        owner: &mut Self::Acquisition,
        generation: &crate::SidecarLaunch,
        deadline: Instant,
    ) -> Result<(), FaultScope> {
        self.prepare_with_pcm_spawners(
            owner,
            generation,
            deadline,
            PulsePcmCapture::spawn,
            PulsePcmPlayback::spawn,
        )
        .await
    }

    fn finish(&self, mut owner: Self::Acquisition) -> Result<Self::Prepared, Self::Acquisition> {
        if owner.provider.is_none() || owner.capture.is_none() || owner.playback.is_none() {
            return Err(owner);
        }
        let provider = owner
            .provider
            .take()
            .expect("provider presence was checked");
        let capture = owner.capture.take().expect("capture presence was checked");
        let playback = owner
            .playback
            .take()
            .expect("playback presence was checked");
        let runtime_generation = owner
            .runtime_generation
            .expect("runtime generation is recorded during prepare");
        Ok(PreparedDirection {
            launch: owner.launch,
            session: owner.session,
            provider,
            capture,
            playback: Some(playback),
            playback_reusable: true,
            runtime_generation,
        })
    }

    fn recover_owner(&self, prepared: Self::Prepared) -> Self::Acquisition {
        ProcessAcquisition {
            launch: prepared.launch,
            session: prepared.session,
            provider: Some(prepared.provider),
            capture: Some(prepared.capture),
            playback: prepared.playback,
            runtime_generation: Some(prepared.runtime_generation),
        }
    }

    async fn run(
        &self,
        prepared: &mut Self::Prepared,
        stop: &mut watch::Receiver<Option<WorkerStop>>,
        observer: Arc<dyn DuplexRuntimeObserver>,
        entered: oneshot::Sender<()>,
    ) -> DirectionOutcome {
        let _ = entered.send(());
        match run_direction_loop(prepared, stop, observer).await {
            Ok(outcome) => outcome,
            Err(origin) => DirectionOutcome::Fault(classify_direction_failure(origin)),
        }
    }

    async fn stop_pcm(
        &self,
        owner: &mut Self::Acquisition,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let mut clean = true;
        if let Some(capture) = owner.capture.as_mut() {
            if Instant::now() < deadline
                && tokio::time::timeout_at(deadline, capture.stop())
                    .await
                    .is_ok_and(|result| result.is_ok())
            {
                owner.capture = None;
            } else {
                clean = false;
            }
        }
        if let Some(playback) = owner.playback.as_mut() {
            if Instant::now() < deadline
                && tokio::time::timeout_at(deadline, playback.stop())
                    .await
                    .is_ok_and(|result| result.is_ok())
            {
                owner.playback = None;
            } else {
                clean = false;
            }
        }
        if clean {
            Ok(())
        } else {
            Err(DuplexRuntimeError::StopFailed)
        }
    }

    async fn close_provider(
        &self,
        owner: &mut Self::Acquisition,
        reason: CloseRequestReason,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let Some(provider) = owner.provider.as_mut() else {
            return Ok(());
        };
        let deadline = deadline.min(Instant::now() + CLOSE_ACK_TIMEOUT);
        if Instant::now() >= deadline {
            return Err(DuplexRuntimeError::StopFailed);
        }
        tokio::time::timeout_at(deadline, provider.send(owner.session.close_request(reason)))
            .await
            .map_err(|_| DuplexRuntimeError::StopFailed)?
            .map_err(|_| DuplexRuntimeError::StopFailed)?;
        if Instant::now() >= deadline {
            return Err(DuplexRuntimeError::StopFailed);
        }
        tokio::time::timeout_at(deadline, async {
            loop {
                let event = provider
                    .next_event()
                    .await
                    .map_err(|_| DuplexRuntimeError::StopFailed)?
                    .ok_or(DuplexRuntimeError::StopFailed)?;
                let closed = owner
                    .session
                    .handle_provider_event(&event, monotonic_ns())
                    .map_err(|_| DuplexRuntimeError::StopFailed)?
                    .into_iter()
                    .any(|effect| effect == DirectionEffect::SessionClosed);
                if closed {
                    return Ok(());
                }
            }
        })
        .await
        .map_err(|_| DuplexRuntimeError::StopFailed)??;
        owner.provider = None;
        Ok(())
    }

    fn discard_provider(&self, owner: &mut Self::Acquisition) {
        owner.provider = None;
    }

    fn is_clean(owner: &Self::Acquisition) -> bool {
        owner.provider.is_none() && owner.capture.is_none() && owner.playback.is_none()
    }

    async fn wait_ready<R: crate::SidecarRuntime>(
        &self,
        supervisor: &SidecarSupervisor<R>,
    ) -> Result<(), DuplexRuntimeError> {
        let launch = supervisor.launch().ok_or(DuplexRuntimeError::StartFailed)?;
        wait_provider_ready(
            &self.config.socket_path,
            &launch.token,
            launch.generation_id,
            PROVIDER_READY_TIMEOUT,
        )
        .await
        .map_err(|error| {
            tracing::error!(event = "provider_models_unavailable", code = ?error);
            DuplexRuntimeError::StartFailed
        })
    }

    async fn probe_generation<R: crate::SidecarRuntime>(
        &self,
        supervisor: &SidecarSupervisor<R>,
    ) -> bool {
        let Some(launch) = supervisor.launch() else {
            return false;
        };
        wait_provider_ready(
            &self.config.socket_path,
            &launch.token,
            launch.generation_id,
            Duration::from_secs(1),
        )
        .await
        .is_ok()
    }
}

fn direction_start_error(direction: AudioDirection, stage: &'static str) {
    tracing::error!(
        event = "direction_prepare_stage_failed",
        direction = ?direction,
        stage
    );
}

const fn provider_state_is_operational(state: ProviderState) -> bool {
    matches!(state, ProviderState::Ready | ProviderState::Degraded)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HotIoKind {
    ProviderSend,
    PlaybackWrite,
}

#[derive(Debug, PartialEq, Eq)]
enum HotIoResult<T> {
    Completed(T),
    Stopped { kind: HotIoKind, stop: WorkerStop },
    TimedOut { kind: HotIoKind },
    NotReusable { kind: HotIoKind },
}

async fn await_hot_io<T>(
    kind: HotIoKind,
    stop: &mut watch::Receiver<Option<WorkerStop>>,
    phase_deadline: Option<Instant>,
    operation: impl Future<Output = T>,
) -> HotIoResult<T> {
    let admitted = Instant::now();
    let deadline = phase_deadline
        .unwrap_or(admitted + HOT_IO_LIVENESS_TIMEOUT)
        .min(admitted + HOT_IO_LIVENESS_TIMEOUT);
    if Instant::now() >= deadline {
        return HotIoResult::TimedOut { kind };
    }
    tokio::select! {
        biased;
        stop = wait_for_direction_stop(stop) => HotIoResult::Stopped { kind, stop },
        _ = tokio::time::sleep_until(deadline) => HotIoResult::TimedOut { kind },
        result = operation => HotIoResult::Completed(result),
    }
}

async fn await_provider_send<T>(
    stop: &mut watch::Receiver<Option<WorkerStop>>,
    phase_deadline: Option<Instant>,
    operation: impl Future<Output = T>,
) -> HotIoResult<T> {
    await_hot_io(HotIoKind::ProviderSend, stop, phase_deadline, operation).await
}

async fn await_playback_write<T>(
    reusable: &mut bool,
    stop: &mut watch::Receiver<Option<WorkerStop>>,
    phase_deadline: Option<Instant>,
    operation: impl Future<Output = T>,
) -> HotIoResult<T> {
    if !*reusable {
        return HotIoResult::NotReusable {
            kind: HotIoKind::PlaybackWrite,
        };
    }
    let result = await_hot_io(HotIoKind::PlaybackWrite, stop, phase_deadline, operation).await;
    if matches!(
        result,
        HotIoResult::Stopped { .. } | HotIoResult::TimedOut { .. }
    ) {
        *reusable = false;
    }
    result
}

fn watchdog_phase_deadline(session: &DirectionSession) -> Option<Instant> {
    let sampled = Instant::now();
    let now_ns = monotonic_ns();
    let remaining = session.next_watchdog_deadline_ns()?.saturating_sub(now_ns);
    sampled.checked_add(Duration::from_nanos(remaining))
}

async fn stop_playback_for_reset(
    direction: &mut PreparedDirection,
    stop: &mut watch::Receiver<Option<WorkerStop>>,
) -> Result<Option<WorkerStop>, DirectionFailureOrigin> {
    let Some(playback) = direction.playback.as_mut() else {
        return Ok(None);
    };
    match await_playback_write(
        &mut direction.playback_reusable,
        stop,
        watchdog_phase_deadline(&direction.session),
        playback.stop(),
    )
    .await
    {
        HotIoResult::Completed(Ok(())) => {
            direction.playback = None;
            Ok(None)
        }
        HotIoResult::Stopped { stop, .. } => Ok(Some(stop)),
        HotIoResult::Completed(Err(_))
        | HotIoResult::TimedOut { .. }
        | HotIoResult::NotReusable { .. } => Err(DirectionFailureOrigin::PcmPlayback),
    }
}

async fn send_provider<F>(
    direction: AudioDirection,
    effect_context: ProviderEffectContext,
    stop: &mut watch::Receiver<Option<WorkerStop>>,
    phase_deadline: Option<Instant>,
    operation: F,
    observer: &dyn DuplexRuntimeObserver,
) -> Result<Option<WorkerStop>, DirectionFailureOrigin>
where
    F: Future<Output = Result<(), ProviderClientError>>,
{
    observer.provider_submission_attempted_for_origin(direction, effect_context.observed());
    match await_provider_send(stop, phase_deadline, operation).await {
        HotIoResult::Completed(Ok(())) => {
            observer.provider_submission_accepted_for_origin(direction, effect_context.observed());
            Ok(None)
        }
        HotIoResult::Stopped { stop, .. } => Ok(Some(stop)),
        HotIoResult::Completed(Err(error)) => Err(provider_failure_origin(&error)),
        HotIoResult::TimedOut { .. } | HotIoResult::NotReusable { .. } => {
            Err(DirectionFailureOrigin::ProviderConnection)
        }
    }
}

fn provider_send_operation(
    direction: &PreparedDirection,
    request: ProviderRequest,
) -> (
    AudioDirection,
    Option<Instant>,
    impl Future<Output = Result<(), ProviderClientError>> + '_,
) {
    (
        direction.launch.runtime.direction,
        watchdog_phase_deadline(&direction.session),
        direction.provider.send(request),
    )
}

struct CaptureDispatchCustody {
    direction: AudioDirection,
    runtime_generation: Uuid,
    pending_frames: u64,
}

impl CaptureDispatchCustody {
    fn begin(
        direction: AudioDirection,
        runtime_generation: Uuid,
        pending_frames: u64,
        released: &[CaptureEvent],
        observer: &dyn DuplexRuntimeObserver,
    ) -> Self {
        observer.capture_frames_pending(
            direction,
            runtime_generation,
            pending_frames.saturating_add(u64::from(!released.is_empty())),
        );
        Self {
            direction,
            runtime_generation,
            pending_frames,
        }
    }

    fn complete(self, observer: &dyn DuplexRuntimeObserver) {
        observer.capture_frames_pending(
            self.direction,
            self.runtime_generation,
            self.pending_frames,
        );
    }
}

fn process_capture_frame<D: VoiceDetector>(
    segmenter: &mut SpeechSegmenter<D>,
    frame: PcmFrame,
    direction: AudioDirection,
    runtime_generation: Uuid,
) -> Result<(CompletedCaptureFrame, Vec<CaptureEvent>, u64), DirectionFailureOrigin> {
    let format = frame.format();
    let completion = CompletedCaptureFrame {
        sequence: frame.sequence(),
        capture_monotonic_ns: frame.capture_monotonic_ns(),
        sample_rate_hz: format.sample_rate_hz(),
        channels: format.channels(),
        frame_duration_ms: format.frame_duration_ms(),
        samples_per_frame: u64::from(format.sample_rate_hz())
            .saturating_mul(u64::from(format.frame_duration_ms()))
            / 1_000,
        runtime_generation,
    };
    let events = segmenter.process(frame).map_err(|_| {
        direction_runtime_error(direction, "capture_vad", DirectionFailureOrigin::Vad)
    })?;
    let pending_frames = u64::try_from(segmenter.pending_frame_count()).unwrap_or(u64::MAX);
    Ok((completion, events, pending_frames))
}

fn capture_event_effect_context(
    event: &CaptureEvent,
    runtime_generation: Uuid,
) -> Option<ProviderEffectContext> {
    match event {
        CaptureEvent::Frame { frame, .. } => Some(ProviderEffectContext::captured(
            runtime_generation,
            frame.capture_monotonic_ns(),
        )),
        CaptureEvent::SpeechStarted { .. } => None,
    }
}

fn observe_capture_event(
    direction: AudioDirection,
    event: &CaptureEvent,
    observer: &dyn DuplexRuntimeObserver,
) {
    if let CaptureEvent::SpeechStarted {
        utterance_id,
        capture_monotonic_ns,
        ..
    } = event
    {
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction,
            utterance_id: *utterance_id,
            capture_monotonic_ns: *capture_monotonic_ns,
        });
    }
}

enum PlaybackWrite {
    Completed(u64),
    Stopped(WorkerStop),
}

async fn write_playback_frame(
    direction: &mut PreparedDirection,
    stop: &mut watch::Receiver<Option<WorkerStop>>,
    frame: &PcmFrame,
    metadata: QueuedPlaybackMetadata,
    observer: &dyn DuplexRuntimeObserver,
) -> Result<PlaybackWrite, DirectionFailureOrigin> {
    let result = await_playback_write(
        &mut direction.playback_reusable,
        stop,
        watchdog_phase_deadline(&direction.session),
        direction
            .playback
            .as_mut()
            .expect("playback was restored")
            .write_frame(frame),
    )
    .await;
    match result {
        HotIoResult::Completed(result) => {
            let observed = monotonic_ns();
            observe_playback_write(
                result,
                direction.launch.runtime.direction,
                metadata,
                observed,
                observer,
            )
            .map_err(|_| DirectionFailureOrigin::PcmPlayback)?;
            Ok(PlaybackWrite::Completed(observed))
        }
        HotIoResult::Stopped { stop, .. } => Ok(PlaybackWrite::Stopped(stop)),
        HotIoResult::TimedOut { .. } | HotIoResult::NotReusable { .. } => {
            Err(DirectionFailureOrigin::PcmPlayback)
        }
    }
}

async fn run_direction_loop(
    direction: &mut PreparedDirection,
    stop: &mut watch::Receiver<Option<WorkerStop>>,
    observer: Arc<dyn DuplexRuntimeObserver>,
) -> Result<DirectionOutcome, DirectionFailureOrigin> {
    let mut segmenter = SpeechSegmenter::new(
        direction.session.stream_id(),
        WebRtcVoiceDetector::default(),
    );
    let mut capture_sequence = 0;
    let mut capture_queue = BoundedPcmQueue::default();
    let mut playback_queue = BoundedPcmQueue::default();
    let mut playback_metadata = VecDeque::new();
    let mut playback_audible_until_ns = 0;
    let mut watchdog = tokio::time::interval(WATCHDOG_INTERVAL);
    watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            reason = wait_for_direction_stop(stop) => {
                return Ok(DirectionOutcome::Stopped(reason));
            }
            event = direction.provider.next_event() => {
                let event = event
                    .map_err(|error| {
                        tracing::error!(
                            event = "direction_runtime_stage_failed",
                            direction = ?direction.launch.runtime.direction,
                            stage = "provider_event_receive",
                            error = ?error
                        );
                        provider_failure_origin(&error)
                    })?
                    .ok_or_else(|| direction_runtime_error(
                        direction.launch.runtime.direction,
                        "provider_event_closed",
                        DirectionFailureOrigin::ProviderConnection,
                    ))?;
                let effects = direction
                    .session
                    .handle_provider_event(&event, monotonic_ns())
                    .map_err(|error| {
                        tracing::error!(
                            event = "direction_runtime_stage_failed",
                            direction = ?direction.launch.runtime.direction,
                            stage = "provider_event_validation",
                            error = ?error
                        );
                        DirectionFailureOrigin::SessionValidation
                    })?;
                let mut mode_change_requested = false;
                for effect in effects {
                    match effect {
                        DirectionEffect::Playback {
                            utterance_id,
                            frame,
                            ..
                        } => {
                            let metadata = QueuedPlaybackMetadata {
                                utterance_id,
                                sequence: frame.sequence(),
                                provider_monotonic_ns: frame.capture_monotonic_ns(),
                                enqueued_monotonic_ns: monotonic_ns(),
                            };
                            playback_queue
                                .push(frame)
                                .map_err(|_| direction_runtime_error(
                                    direction.launch.runtime.direction,
                                    "playback_queue_overflow",
                                    DirectionFailureOrigin::Queue,
                                ))?;
                            playback_metadata.push_back(metadata);
                        }
                        DirectionEffect::TranscriptFinal { utterance_id } => {
                            tracing::info!(
                                event = "direction_stage_observed",
                                direction = ?direction.launch.runtime.direction,
                                stage = "asr_final",
                                %utterance_id
                            );
                            observer.observe(DuplexRuntimeEvent::TranscriptFinal {
                                direction: direction.launch.runtime.direction,
                                utterance_id,
                            });
                        }
                        DirectionEffect::TranslationFinal { utterance_id } => {
                            tracing::info!(
                                event = "direction_stage_observed",
                                direction = ?direction.launch.runtime.direction,
                                stage = "translation_final",
                                %utterance_id
                            );
                            observer.observe(DuplexRuntimeEvent::TranslationFinal {
                                direction: direction.launch.runtime.direction,
                                utterance_id,
                            });
                        }
                        DirectionEffect::Latency {
                            utterance_id,
                            tts_first_audio_ms,
                            provider_total_ms,
                        } => {
                            observer.observe(DuplexRuntimeEvent::ProviderLatency {
                                direction: direction.launch.runtime.direction,
                                utterance_id,
                                tts_first_audio_ms,
                                provider_total_ms,
                            });
                        }
                        DirectionEffect::ProviderError {
                            utterance_id,
                            code,
                            retryable,
                        } => {
                            tracing::warn!(
                                event = "direction_provider_error",
                                direction = ?direction.launch.runtime.direction,
                                utterance_id = ?utterance_id,
                                code = ?code,
                                retryable
                            );
                            observer.observe(DuplexRuntimeEvent::ProviderError {
                                direction: direction.launch.runtime.direction,
                                utterance_id,
                                code,
                                retryable,
                            });
                        }
                        DirectionEffect::UtteranceTerminalOutcome {
                            utterance_id,
                            outcome,
                        } => {
                            tracing::info!(
                                event = "direction_terminal_outcome",
                                direction = ?direction.launch.runtime.direction,
                                %utterance_id,
                                outcome = ?outcome
                            );
                            observer.observe(DuplexRuntimeEvent::UtteranceTerminalOutcome {
                                direction: direction.launch.runtime.direction,
                                utterance_id,
                                outcome,
                            });
                        }
                        DirectionEffect::UtteranceTerminal { utterance_id } => {
                            tracing::info!(
                                event = "direction_stage_observed",
                                direction = ?direction.launch.runtime.direction,
                                stage = "utterance_terminal",
                                %utterance_id
                            );
                            let event = DuplexRuntimeEvent::UtteranceTerminal {
                                direction: direction.launch.runtime.direction,
                                utterance_id,
                            };
                            observer.observe(event);
                            if mode_change_after_event(
                                observer.as_ref(),
                                direction.launch.runtime.mode,
                                event,
                            )
                            .is_some()
                            {
                                mode_change_requested = true;
                            }
                        }
                        DirectionEffect::ExpiredAudio {
                            utterance_id,
                            request,
                        } => {
                            observer.observe(DuplexRuntimeEvent::FirstAudioExpired {
                                direction: direction.launch.runtime.direction,
                                utterance_id,
                                observed_monotonic_ns: monotonic_ns(),
                            });
                            playback_queue.clear();
                            playback_metadata.clear();
                            if let Some(stop) = stop_playback_for_reset(direction, stop).await? {
                                return Ok(DirectionOutcome::Stopped(stop));
                            }
                            playback_audible_until_ns = 0;
                            let (runtime_direction, phase_deadline, operation) =
                                provider_send_operation(direction, request);
                            if let Some(stop) = send_provider(
                                runtime_direction,
                                ProviderEffectContext::unattributed(direction.runtime_generation),
                                stop,
                                phase_deadline,
                                operation,
                                observer.as_ref(),
                            )
                            .await?
                            {
                                    return Ok(DirectionOutcome::Stopped(stop));
                            }
                        }
                        DirectionEffect::SessionClosed => {}
                    }
                }
                while let Some(frame) = playback_queue.pop() {
                    let metadata = playback_metadata
                        .pop_front()
                        .ok_or_else(|| direction_runtime_error(
                            direction.launch.runtime.direction,
                            "playback_metadata_missing",
                            DirectionFailureOrigin::Queue,
                        ))?;
                    if direction.playback.is_none() {
                        direction.playback = Some(
                            PulsePcmPlayback::spawn(&PulsePcmCommand::playback(
                                &direction.launch.playback_device,
                                direction.launch.playback_stream_name,
                            ))
                            .map_err(|_| direction_runtime_error(
                                direction.launch.runtime.direction,
                                "playback_respawn",
                                DirectionFailureOrigin::PcmPlayback,
                            ))?,
                        );
                        direction.playback_reusable = true;
                    }
                    let observed_monotonic_ns = match write_playback_frame(
                        direction,
                        stop,
                        &frame,
                        metadata,
                        observer.as_ref(),
                    )
                    .await?
                    {
                        PlaybackWrite::Completed(observed) => observed,
                        PlaybackWrite::Stopped(stop) => {
                            return Ok(DirectionOutcome::Stopped(stop));
                        }
                    };
                    playback_audible_until_ns = extend_playback_deadline(
                        playback_audible_until_ns,
                        observed_monotonic_ns,
                        u64::from(frame.format().frame_duration_ms()),
                    );
                }
                if mode_change_requested {
                    return match wait_for_playback_deadline(playback_audible_until_ns, stop).await? {
                        Some(stop) => Ok(DirectionOutcome::Stopped(stop)),
                        None => Ok(DirectionOutcome::ModeChanged),
                    };
                }
            }
            frame = direction.capture.read_frame(capture_sequence, monotonic_ns()) => {
                let frame = frame.map_err(|_| direction_runtime_error(
                    direction.launch.runtime.direction,
                    "capture_read",
                    DirectionFailureOrigin::PcmCapture,
                ))?;
                capture_sequence = capture_sequence.saturating_add(1);
                capture_queue
                    .push(frame)
                    .map_err(|_| direction_runtime_error(
                        direction.launch.runtime.direction,
                        "capture_queue_overflow",
                        DirectionFailureOrigin::Queue,
                    ))?;
                while let Some(frame) = capture_queue.pop() {
                    let (completed_frame, events, pending_frames) = process_capture_frame(
                        &mut segmenter,
                        frame,
                        direction.launch.runtime.direction,
                        direction.runtime_generation,
                    )?;
                    let custody = CaptureDispatchCustody::begin(
                        direction.launch.runtime.direction,
                        direction.runtime_generation,
                        pending_frames,
                        &events,
                        observer.as_ref(),
                    );
                    for event in events {
                        let effect_context = capture_event_effect_context(
                            &event,
                            direction.runtime_generation,
                        );
                        if let CaptureEvent::Frame {
                            utterance_id,
                            frame,
                            end_of_utterance: true,
                            ..
                        } = &event
                        {
                            tracing::info!(
                                event = "direction_capture_eou",
                                direction = ?direction.launch.runtime.direction,
                                %utterance_id,
                                sequence = frame.sequence()
                            );
                        }
                        observe_capture_event(
                            direction.launch.runtime.direction,
                            &event,
                            observer.as_ref(),
                        );
                        if let Some(request) = direction
                            .session
                            .handle_capture(event)
                            .map_err(|_| direction_runtime_error(
                                direction.launch.runtime.direction,
                                "capture_session",
                                DirectionFailureOrigin::SessionValidation,
                            ))?
                        {
                            let (runtime_direction, phase_deadline, operation) =
                                provider_send_operation(direction, request);
                            if let Some(stop) = send_provider(
                                runtime_direction,
                                effect_context.unwrap_or_else(|| {
                                    ProviderEffectContext::unattributed(
                                        direction.runtime_generation,
                                    )
                                }),
                                stop,
                                phase_deadline,
                                operation,
                                observer.as_ref(),
                            )
                            .await?
                            {
                                    return Ok(DirectionOutcome::Stopped(stop));
                            }
                        }
                    }
                    custody.complete(observer.as_ref());
                    observer.capture_frame_processed(
                        direction.launch.runtime.direction,
                        completed_frame,
                    );
                }
            }
            _ = watchdog.tick() => {
                let effects = direction
                    .session
                    .poll(monotonic_ns())
                    .map_err(|_| direction_runtime_error(
                        direction.launch.runtime.direction,
                        "watchdog_poll",
                        DirectionFailureOrigin::SessionValidation,
                    ))?;
                for effect in effects {
                    match effect {
                        DirectionWatchdogEffect::Send(request) => {
                            tracing::warn!(
                                event = "direction_watchdog_action",
                                direction = ?direction.launch.runtime.direction,
                                action = "send"
                            );
                            let (runtime_direction, phase_deadline, operation) =
                                provider_send_operation(direction, request);
                            if let Some(stop) = send_provider(
                                runtime_direction,
                                ProviderEffectContext::unattributed(direction.runtime_generation),
                                stop,
                                phase_deadline,
                                operation,
                                observer.as_ref(),
                            )
                            .await?
                            {
                                    return Ok(DirectionOutcome::Stopped(stop));
                            }
                        }
                        DirectionWatchdogEffect::PurgeAndSend(request) => {
                            tracing::warn!(
                                event = "direction_watchdog_action",
                                direction = ?direction.launch.runtime.direction,
                                action = "purge_and_send"
                            );
                            playback_queue.clear();
                            playback_metadata.clear();
                            if let Some(stop) = stop_playback_for_reset(direction, stop).await? {
                                return Ok(DirectionOutcome::Stopped(stop));
                            }
                            playback_audible_until_ns = 0;
                            let (runtime_direction, phase_deadline, operation) =
                                provider_send_operation(direction, request);
                            if let Some(stop) = send_provider(
                                runtime_direction,
                                ProviderEffectContext::unattributed(direction.runtime_generation),
                                stop,
                                phase_deadline,
                                operation,
                                observer.as_ref(),
                            )
                            .await?
                            {
                                    return Ok(DirectionOutcome::Stopped(stop));
                            }
                        }
                        DirectionWatchdogEffect::RestartSidecar => {
                            tracing::error!(
                                event = "direction_runtime_stage_failed",
                                direction = ?direction.launch.runtime.direction,
                                stage = "watchdog_restart"
                            );
                            return Err(DirectionFailureOrigin::WatchdogRestart);
                        }
                    }
                }
            }
        }
    }
}

fn extend_playback_deadline(
    current_deadline_ns: u64,
    observed_monotonic_ns: u64,
    frame_duration_ms: u64,
) -> u64 {
    current_deadline_ns
        .max(observed_monotonic_ns)
        .saturating_add(frame_duration_ms.saturating_mul(1_000_000))
}

async fn wait_for_playback_deadline(
    deadline_ns: u64,
    stop: &mut watch::Receiver<Option<WorkerStop>>,
) -> Result<Option<WorkerStop>, DirectionFailureOrigin> {
    let remaining_ns = deadline_ns.saturating_sub(monotonic_ns());
    tokio::select! {
        biased;
        stop = wait_for_direction_stop(stop) => Ok(Some(stop)),
        _ = tokio::time::sleep(Duration::from_nanos(remaining_ns)) => Ok(None),
        _ = tokio::time::sleep(HOT_IO_LIVENESS_TIMEOUT) => {
            Err(DirectionFailureOrigin::PcmPlayback)
        }
    }
}

fn direction_runtime_error(
    direction: AudioDirection,
    stage: &'static str,
    origin: DirectionFailureOrigin,
) -> DirectionFailureOrigin {
    tracing::error!(
        event = "direction_runtime_stage_failed",
        direction = ?direction,
        stage
    );
    origin
}

fn observe_playback_write<E>(
    write_result: Result<(), E>,
    direction: AudioDirection,
    metadata: QueuedPlaybackMetadata,
    observed_monotonic_ns: u64,
    observer: &dyn DuplexRuntimeObserver,
) -> Result<(), E> {
    write_result?;
    observer.observe(DuplexRuntimeEvent::AudioFrame {
        direction,
        utterance_id: metadata.utterance_id,
        sequence: metadata.sequence,
        provider_monotonic_ns: metadata.provider_monotonic_ns,
        observed_monotonic_ns,
        queue_lag_ms: duration_ms(metadata.enqueued_monotonic_ns, observed_monotonic_ns),
    });
    Ok(())
}

fn mode_change_after_event(
    observer: &dyn DuplexRuntimeObserver,
    active_mode: TranslationMode,
    event: DuplexRuntimeEvent,
) -> Option<TranslationMode> {
    let DuplexRuntimeEvent::UtteranceTerminal { direction, .. } = event else {
        return None;
    };
    observer
        .requested_mode(direction)
        .filter(|requested| *requested != active_mode)
}

fn refresh_launch_mode(launch: &mut DirectionLaunch, observer: &dyn DuplexRuntimeObserver) -> bool {
    let Some(requested) = observer.requested_mode(launch.runtime.direction) else {
        return false;
    };
    if requested == launch.runtime.mode {
        return false;
    }
    launch.runtime.mode = requested;
    true
}

async fn wait_for_stop(stop: &mut watch::Receiver<Option<Instant>>) -> Instant {
    loop {
        if let Some(deadline) = *stop.borrow() {
            return deadline;
        }
        if stop.changed().await.is_err() {
            return Instant::now() + RUNTIME_CLEANUP_BUDGET;
        }
    }
}

async fn wait_for_direction_stop(stop: &mut watch::Receiver<Option<WorkerStop>>) -> WorkerStop {
    loop {
        if let Some(reason) = *stop.borrow() {
            return reason;
        }
        if stop.changed().await.is_err() {
            return WorkerStop::Close {
                reason: CloseRequestReason::DaemonShutdown,
                deadline: Instant::now() + RUNTIME_CLEANUP_BUDGET,
            };
        }
    }
}

fn monotonic_ns() -> u64 {
    let time = clock_gettime(ClockId::Monotonic);
    let seconds = u64::try_from(time.tv_sec).unwrap_or(0);
    let nanos = u64::try_from(time.tv_nsec).unwrap_or(0);
    seconds.saturating_mul(1_000_000_000).saturating_add(nanos)
}

fn deadline_open<E>(deadline: Instant, error: E) -> Result<(), E> {
    if Instant::now() < deadline {
        Ok(())
    } else {
        Err(error)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::{
        collections::VecDeque,
        os::unix::fs::{MetadataExt, PermissionsExt},
        path::Path,
        pin::Pin,
        process::Stdio,
        sync::Mutex,
    };
    use tempfile::tempdir;
    use tokio_stream::{Stream, StreamExt};
    use tonic::{Request, Response, Status};
    use translator_audio::{
        AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT,
        AEC_OBSERVATION_FRAME_DURATION_NS, AudioGraphState, GraphHealth, MIC_OUT_SINK,
        ProcessIdentity, REMOTE_IN_SINK, StreamPcmFormat, VadError, VoiceDetector,
    };
    use translator_core::{Language, ProviderId, TranslationMode, VoiceEngine};
    use translator_ipc::provider::{
        ProviderCapabilities, ProviderEvent, ProviderHealth, ProviderProbeRequest,
        ProviderProbeResponse, ProviderQueues, ProviderSessionClosed, ProviderSessionOpened,
        SessionCloseReason, provider_request,
        provider_transport_server::{ProviderTransport, ProviderTransportServer},
    };

    #[derive(Default)]
    struct RecordingObserver {
        events: Mutex<Vec<DuplexRuntimeEvent>>,
        internal_events: Mutex<Vec<InternalRuntimeObservation>>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum InternalRuntimeObservation {
        CaptureFrameProcessed {
            direction: AudioDirection,
            sequence: u64,
        },
        ProviderSubmissionAttempted {
            direction: AudioDirection,
        },
        ProviderSubmissionAccepted {
            direction: AudioDirection,
        },
    }

    impl DuplexRuntimeObserver for RecordingObserver {
        fn observe(&self, event: DuplexRuntimeEvent) {
            self.events.lock().unwrap().push(event);
        }

        fn capture_frame_processed(&self, direction: AudioDirection, frame: CompletedCaptureFrame) {
            self.internal_events.lock().unwrap().push(
                InternalRuntimeObservation::CaptureFrameProcessed {
                    direction,
                    sequence: frame.sequence,
                },
            );
        }

        fn provider_submission_attempted(
            &self,
            direction: AudioDirection,
            _observed_monotonic_ns: u64,
        ) {
            self.internal_events
                .lock()
                .unwrap()
                .push(InternalRuntimeObservation::ProviderSubmissionAttempted { direction });
        }

        fn provider_submission_accepted(
            &self,
            direction: AudioDirection,
            _observed_monotonic_ns: u64,
        ) {
            self.internal_events
                .lock()
                .unwrap()
                .push(InternalRuntimeObservation::ProviderSubmissionAccepted { direction });
        }
    }

    #[derive(Default)]
    struct ScriptedVoiceDetector {
        decisions: VecDeque<bool>,
    }

    impl ScriptedVoiceDetector {
        fn new(decisions: impl IntoIterator<Item = bool>) -> Self {
            Self {
                decisions: decisions.into_iter().collect(),
            }
        }
    }

    impl VoiceDetector for ScriptedVoiceDetector {
        fn is_voice(&mut self, _samples: &[i16]) -> Result<bool, VadError> {
            Ok(self.decisions.pop_front().unwrap_or(false))
        }
    }

    fn aec_test_frame(sequence: u64, capture_monotonic_ns: u64) -> PcmFrame {
        let format = StreamPcmFormat::provider_default();
        PcmFrame::try_new(
            sequence,
            capture_monotonic_ns,
            format,
            vec![0; format.frame_bytes()],
        )
        .unwrap()
    }

    fn completed_aec_test_frame(
        sequence: u64,
        capture_monotonic_ns: u64,
        runtime_generation: Uuid,
    ) -> CompletedCaptureFrame {
        let format = StreamPcmFormat::provider_default();
        CompletedCaptureFrame {
            sequence,
            capture_monotonic_ns,
            sample_rate_hz: format.sample_rate_hz(),
            channels: format.channels(),
            frame_duration_ms: format.frame_duration_ms(),
            samples_per_frame: u64::from(format.sample_rate_hz())
                .saturating_mul(u64::from(format.frame_duration_ms()))
                / 1_000,
            runtime_generation,
        }
    }

    fn active_aec_test_observer(
        scored_started_ns: u64,
        runtime_generation: Uuid,
    ) -> crate::AecRuntimeObserver {
        let (observer, activation) =
            crate::AecRuntimeObserver::new(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        observer.activate_generation(activation).unwrap();
        observer
            .start_positive_control(scored_started_ns.saturating_sub(1))
            .unwrap();
        observer
            .complete_positive_control(scored_started_ns)
            .unwrap();
        observer
            .start_scored_interval(Uuid::new_v4(), scored_started_ns, runtime_generation)
            .unwrap();
        observer
    }

    #[test]
    fn aec_runtime_observer_capture_seam_records_processed_frame_without_public_event() {
        let observer = RecordingObserver::default();
        let mut segmenter =
            SpeechSegmenter::new(uuid::Uuid::new_v4(), WebRtcVoiceDetector::default());
        let format = translator_audio::StreamPcmFormat::provider_default();
        let frame = PcmFrame::try_new(7, 11, format, vec![0; format.frame_bytes()]).unwrap();

        let generation = uuid::Uuid::new_v4();
        let (completed, events, pending_frames) = process_capture_frame(
            &mut segmenter,
            frame,
            AudioDirection::Microphone,
            generation,
        )
        .unwrap();

        assert!(events.is_empty());
        assert_eq!(pending_frames, 0);
        assert_eq!(completed.sequence, 7);
        assert_eq!(completed.capture_monotonic_ns, 11);
        assert_eq!(completed.runtime_generation, generation);
        assert!(observer.internal_events.lock().unwrap().is_empty());
        assert!(observer.events.lock().unwrap().is_empty());
    }

    #[test]
    fn aec_runtime_observer_speech_started_remains_a_public_runtime_event() {
        let observer = RecordingObserver::default();
        let utterance_id = uuid::Uuid::new_v4();
        let event = CaptureEvent::SpeechStarted {
            stream_id: uuid::Uuid::new_v4(),
            utterance_id,
            capture_monotonic_ns: 19,
        };

        observe_capture_event(AudioDirection::Microphone, &event, &observer);

        assert_eq!(
            observer.events.lock().unwrap().as_slice(),
            &[DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Microphone,
                utterance_id,
                capture_monotonic_ns: 19,
            }]
        );
        assert!(observer.internal_events.lock().unwrap().is_empty());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn aec_runtime_observer_provider_seam_records_attempt_before_io_and_accepts_only_success()
    {
        let observer = Arc::new(RecordingObserver::default());
        let (_stop, mut stop_receiver) = watch::channel(None);
        let checked = observer.clone();
        let result = send_provider(
            AudioDirection::Microphone,
            ProviderEffectContext::unattributed(Uuid::nil()),
            &mut stop_receiver,
            None,
            async move {
                assert_eq!(
                    checked.internal_events.lock().unwrap().as_slice(),
                    &[InternalRuntimeObservation::ProviderSubmissionAttempted {
                        direction: AudioDirection::Microphone,
                    }]
                );
                Ok(())
            },
            observer.as_ref(),
        )
        .await;

        assert_eq!(result, Ok(None));
        assert_eq!(
            observer.internal_events.lock().unwrap().as_slice(),
            &[
                InternalRuntimeObservation::ProviderSubmissionAttempted {
                    direction: AudioDirection::Microphone,
                },
                InternalRuntimeObservation::ProviderSubmissionAccepted {
                    direction: AudioDirection::Microphone,
                },
            ]
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn aec_runtime_observer_provider_error_timeout_and_stop_never_record_acceptance() {
        let error_observer = RecordingObserver::default();
        let (_error_stop, mut error_stop_receiver) = watch::channel(None);
        assert_eq!(
            send_provider(
                AudioDirection::Microphone,
                ProviderEffectContext::unattributed(Uuid::nil()),
                &mut error_stop_receiver,
                None,
                async { Err(ProviderClientError::RequestChannelClosed) },
                &error_observer,
            )
            .await,
            Err(DirectionFailureOrigin::ProviderConnection)
        );
        assert_eq!(
            error_observer.internal_events.lock().unwrap().as_slice(),
            &[InternalRuntimeObservation::ProviderSubmissionAttempted {
                direction: AudioDirection::Microphone,
            }]
        );

        let timeout_observer = Arc::new(RecordingObserver::default());
        let (_timeout_stop, mut timeout_stop_receiver) = watch::channel(None);
        let timeout_observer_task = timeout_observer.clone();
        let timeout = tokio::spawn(async move {
            send_provider(
                AudioDirection::Microphone,
                ProviderEffectContext::unattributed(Uuid::nil()),
                &mut timeout_stop_receiver,
                None,
                std::future::pending::<Result<(), ProviderClientError>>(),
                timeout_observer_task.as_ref(),
            )
            .await
        });
        tokio::task::yield_now().await;
        tokio::time::advance(HOT_IO_LIVENESS_TIMEOUT).await;
        assert_eq!(
            timeout.await.unwrap(),
            Err(DirectionFailureOrigin::ProviderConnection)
        );
        assert_eq!(
            timeout_observer.internal_events.lock().unwrap().as_slice(),
            &[InternalRuntimeObservation::ProviderSubmissionAttempted {
                direction: AudioDirection::Microphone,
            }]
        );

        let stop_observer = Arc::new(RecordingObserver::default());
        let (stop, mut stop_receiver) = watch::channel(None);
        let stop_observer_task = stop_observer.clone();
        let stopped = tokio::spawn(async move {
            send_provider(
                AudioDirection::Microphone,
                ProviderEffectContext::unattributed(Uuid::nil()),
                &mut stop_receiver,
                None,
                std::future::pending::<Result<(), ProviderClientError>>(),
                stop_observer_task.as_ref(),
            )
            .await
        });
        tokio::task::yield_now().await;
        let expected = WorkerStop::Close {
            reason: CloseRequestReason::DaemonShutdown,
            deadline: Instant::now() + RUNTIME_CLEANUP_BUDGET,
        };
        stop.send(Some(expected)).unwrap();
        assert_eq!(stopped.await.unwrap(), Ok(Some(expected)));
        assert_eq!(
            stop_observer.internal_events.lock().unwrap().as_slice(),
            &[InternalRuntimeObservation::ProviderSubmissionAttempted {
                direction: AudioDirection::Microphone,
            }]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn c9_s5a_provider_effect_completed_after_boundary_is_counted_by_capture_origin() {
        let observed_after_boundary_ns = monotonic_ns();
        let ended_ns = observed_after_boundary_ns.checked_sub(1).unwrap();
        let started_ns = ended_ns.checked_sub(AEC_OBSERVATION_DURATION_NS).unwrap();
        let runtime_generation = Uuid::new_v4();
        let observer = active_aec_test_observer(started_ns, runtime_generation);
        observer.capture_frame_processed(
            AudioDirection::Microphone,
            completed_aec_test_frame(0, ended_ns, runtime_generation),
        );

        let (_stop, mut stop_receiver) = watch::channel(None);
        assert_eq!(
            send_provider(
                AudioDirection::Microphone,
                ProviderEffectContext::captured(runtime_generation, ended_ns),
                &mut stop_receiver,
                None,
                async { Ok(()) },
                &observer,
            )
            .await,
            Ok(None)
        );
        assert_eq!(
            send_provider(
                AudioDirection::Microphone,
                ProviderEffectContext::captured(Uuid::new_v4(), ended_ns),
                &mut stop_receiver,
                None,
                async { Ok(()) },
                &observer,
            )
            .await,
            Ok(None)
        );

        let evidence = observer.complete_scored_interval(ended_ns).unwrap();
        assert_eq!(evidence.provider_attempts_after, 1);
        assert_eq!(evidence.provider_accepted_after, 1);
    }

    #[test]
    fn c9_s5b_unresolved_segmenter_origin_at_freeze_invalidates_evidence() {
        let started_ns = 1_000_000_000;
        let runtime_generation = Uuid::new_v4();
        let observer = active_aec_test_observer(started_ns, runtime_generation);
        let decisions = std::iter::repeat_n(false, AEC_OBSERVATION_FRAME_COUNT as usize - 1)
            .chain(std::iter::once(true));
        let mut segmenter = SpeechSegmenter::with_confirmation_frames(
            Uuid::new_v4(),
            ScriptedVoiceDetector::new(decisions),
            3,
        );

        for sequence in 0..AEC_OBSERVATION_FRAME_COUNT {
            let capture_monotonic_ns = started_ns + sequence * AEC_OBSERVATION_FRAME_DURATION_NS;
            let (completed, events, pending_frames) = process_capture_frame(
                &mut segmenter,
                aec_test_frame(sequence, capture_monotonic_ns),
                AudioDirection::Microphone,
                runtime_generation,
            )
            .unwrap();
            assert!(events.is_empty());
            observer.capture_frames_pending(
                AudioDirection::Microphone,
                runtime_generation,
                pending_frames,
            );
            observer.capture_frame_processed(AudioDirection::Microphone, completed);
        }

        let evidence = observer
            .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
            .unwrap();
        assert_eq!(evidence.processed_frames, AEC_OBSERVATION_FRAME_COUNT);
        assert!(evidence.observer_errors > 0 || evidence.terminated_early);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn c9_s5c_segmenter_phase_provenance_excludes_positive_and_post_interval_effects() {
        let started_ns = monotonic_ns();
        let ended_ns = started_ns + AEC_OBSERVATION_DURATION_NS;
        let runtime_generation = Uuid::new_v4();
        let (observer, activation) =
            crate::AecRuntimeObserver::new(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        observer.activate_generation(activation).unwrap();
        observer
            .start_positive_control(started_ns - AEC_OBSERVATION_FRAME_DURATION_NS * 2)
            .unwrap();
        let mut segmenter = SpeechSegmenter::with_confirmation_frames(
            Uuid::new_v4(),
            ScriptedVoiceDetector::new([true, true, true]),
            3,
        );

        for (sequence, capture_monotonic_ns) in [
            (0, started_ns - AEC_OBSERVATION_FRAME_DURATION_NS * 2),
            (1, started_ns - AEC_OBSERVATION_FRAME_DURATION_NS),
        ] {
            let (completed, events, pending_frames) = process_capture_frame(
                &mut segmenter,
                aec_test_frame(sequence, capture_monotonic_ns),
                AudioDirection::Microphone,
                runtime_generation,
            )
            .unwrap();
            assert!(events.is_empty());
            observer.capture_frames_pending(
                AudioDirection::Microphone,
                runtime_generation,
                pending_frames,
            );
            observer.capture_frame_processed(AudioDirection::Microphone, completed);
        }
        observer.complete_positive_control(started_ns).unwrap();
        observer
            .start_scored_interval(Uuid::new_v4(), started_ns, runtime_generation)
            .unwrap();

        let (completed, released, pending_frames) = process_capture_frame(
            &mut segmenter,
            aec_test_frame(2, started_ns),
            AudioDirection::Microphone,
            runtime_generation,
        )
        .unwrap();
        assert_eq!(released.len(), 4);
        let custody = CaptureDispatchCustody::begin(
            AudioDirection::Microphone,
            runtime_generation,
            pending_frames,
            &released,
            &observer,
        );
        let (_stop, mut stop_receiver) = watch::channel(None);
        for event in &released {
            observe_capture_event(AudioDirection::Microphone, event, &observer);
            if matches!(event, CaptureEvent::Frame { .. }) {
                assert_eq!(
                    send_provider(
                        AudioDirection::Microphone,
                        capture_event_effect_context(event, runtime_generation).unwrap(),
                        &mut stop_receiver,
                        None,
                        async { Ok(()) },
                        &observer,
                    )
                    .await,
                    Ok(None)
                );
            }
        }
        custody.complete(&observer);
        observer.capture_frame_processed(AudioDirection::Microphone, completed);

        observer.provider_submission_attempted(AudioDirection::Microphone, ended_ns + 1);
        observer.provider_submission_accepted(AudioDirection::Microphone, ended_ns + 1);

        let evidence = observer.complete_scored_interval(ended_ns).unwrap();
        assert_eq!(evidence.vad_events_after, 0);
        assert_eq!(evidence.provider_attempts_after, 1);
        assert_eq!(evidence.provider_accepted_after, 1);
        assert_eq!(evidence.positive_control.speech_started_events, 1);
        assert_eq!(evidence.positive_control.provider_submission_attempts, 2);
        assert_eq!(evidence.positive_control.provider_submissions_accepted, 2);
    }

    fn c10_s1_observer(started_ns: u64, runtime_generation: Uuid) -> crate::AecRuntimeObserver {
        let (observer, activation) =
            crate::AecRuntimeObserver::new(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        observer.activate_generation(activation).unwrap();
        let positive_started_ns = started_ns - AEC_OBSERVATION_FRAME_DURATION_NS * 2;
        observer
            .start_positive_control(positive_started_ns)
            .unwrap();
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: positive_started_ns,
        });
        observer.provider_submission_attempted(AudioDirection::Microphone, positive_started_ns);
        observer.provider_submission_accepted(AudioDirection::Microphone, positive_started_ns);
        observer.complete_positive_control(started_ns).unwrap();
        observer
            .start_scored_interval(Uuid::new_v4(), started_ns, runtime_generation)
            .unwrap();
        observer
    }

    fn c10_s1_segmenter() -> SpeechSegmenter<ScriptedVoiceDetector> {
        let decisions = std::iter::repeat_n(false, AEC_OBSERVATION_FRAME_COUNT as usize - 2)
            .chain([true, true, true]);
        SpeechSegmenter::with_confirmation_frames(
            Uuid::new_v4(),
            ScriptedVoiceDetector::new(decisions),
            3,
        )
    }

    #[tokio::test(flavor = "current_thread")]
    async fn c10_s1_released_batch_stays_pending_until_dispatch_completes() {
        let started_ns = 1_000_000_000;
        let ended_ns = started_ns + AEC_OBSERVATION_DURATION_NS;
        let runtime_generation = Uuid::new_v4();
        let observer = c10_s1_observer(started_ns, runtime_generation);
        let mut segmenter = c10_s1_segmenter();

        for sequence in 0..AEC_OBSERVATION_FRAME_COUNT {
            let capture_monotonic_ns = started_ns + sequence * AEC_OBSERVATION_FRAME_DURATION_NS;
            let (completed, events, pending_frames) = process_capture_frame(
                &mut segmenter,
                aec_test_frame(sequence, capture_monotonic_ns),
                AudioDirection::Microphone,
                runtime_generation,
            )
            .unwrap();
            assert!(events.is_empty());
            observer.capture_frames_pending(
                AudioDirection::Microphone,
                runtime_generation,
                pending_frames,
            );
            observer.capture_frame_processed(AudioDirection::Microphone, completed);
        }

        let (_completed, released, pending_frames) = process_capture_frame(
            &mut segmenter,
            aec_test_frame(
                AEC_OBSERVATION_FRAME_COUNT,
                ended_ns + AEC_OBSERVATION_FRAME_DURATION_NS,
            ),
            AudioDirection::Microphone,
            runtime_generation,
        )
        .unwrap();
        assert_eq!(released.len(), 4);
        let _custody = CaptureDispatchCustody::begin(
            AudioDirection::Microphone,
            runtime_generation,
            pending_frames,
            &released,
            &observer,
        );
        let blocked = observer.complete_scored_interval(ended_ns).unwrap();
        assert_eq!(blocked.processed_frames, AEC_OBSERVATION_FRAME_COUNT);
        assert_eq!(blocked.positive_control.speech_started_events, 1);
        assert_eq!(blocked.positive_control.provider_submission_attempts, 1);
        assert_eq!(blocked.positive_control.provider_submissions_accepted, 1);
        assert!(
            blocked.observer_errors > 0 || blocked.terminated_early,
            "released effects must remain an obligation until dispatch"
        );

        let observer = c10_s1_observer(started_ns, runtime_generation);
        let mut segmenter = c10_s1_segmenter();
        for sequence in 0..AEC_OBSERVATION_FRAME_COUNT {
            let capture_monotonic_ns = started_ns + sequence * AEC_OBSERVATION_FRAME_DURATION_NS;
            let (completed, events, pending_frames) = process_capture_frame(
                &mut segmenter,
                aec_test_frame(sequence, capture_monotonic_ns),
                AudioDirection::Microphone,
                runtime_generation,
            )
            .unwrap();
            assert!(events.is_empty());
            observer.capture_frames_pending(
                AudioDirection::Microphone,
                runtime_generation,
                pending_frames,
            );
            observer.capture_frame_processed(AudioDirection::Microphone, completed);
        }
        let (_completed, released, pending_frames) = process_capture_frame(
            &mut segmenter,
            aec_test_frame(
                AEC_OBSERVATION_FRAME_COUNT,
                ended_ns + AEC_OBSERVATION_FRAME_DURATION_NS,
            ),
            AudioDirection::Microphone,
            runtime_generation,
        )
        .unwrap();
        let custody = CaptureDispatchCustody::begin(
            AudioDirection::Microphone,
            runtime_generation,
            pending_frames,
            &released,
            &observer,
        );
        let (_stop, mut stop_receiver) = watch::channel(None);
        for event in &released {
            observe_capture_event(AudioDirection::Microphone, event, &observer);
            if matches!(event, CaptureEvent::Frame { .. }) {
                assert_eq!(
                    send_provider(
                        AudioDirection::Microphone,
                        capture_event_effect_context(event, runtime_generation).unwrap(),
                        &mut stop_receiver,
                        None,
                        async { Ok(()) },
                        &observer,
                    )
                    .await,
                    Ok(None)
                );
            }
        }
        custody.complete(&observer);
        observer.provider_submission_attempted(AudioDirection::Microphone, ended_ns + 1);
        observer.provider_submission_accepted(AudioDirection::Microphone, ended_ns + 1);

        let evidence = observer.complete_scored_interval(ended_ns).unwrap();
        assert_eq!(evidence.observer_errors, 0);
        assert_eq!(evidence.vad_events_after, 1);
        assert_eq!(evidence.provider_attempts_after, 2);
        assert_eq!(evidence.provider_accepted_after, 2);
    }

    #[derive(Default)]
    struct C10S1PendingObserver(Mutex<Vec<u64>>);

    impl DuplexRuntimeObserver for C10S1PendingObserver {
        fn observe(&self, _event: DuplexRuntimeEvent) {}

        fn capture_frames_pending(
            &self,
            _direction: AudioDirection,
            _runtime_generation: Uuid,
            pending_frames: u64,
        ) {
            self.0.lock().unwrap().push(pending_frames);
        }
    }

    #[test]
    fn c10_s1_dispatch_custody_clears_only_on_success() {
        let runtime_generation = Uuid::new_v4();
        let released = [CaptureEvent::SpeechStarted {
            stream_id: Uuid::new_v4(),
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: 1,
        }];
        let failed = C10S1PendingObserver::default();
        {
            let _custody = CaptureDispatchCustody::begin(
                AudioDirection::Microphone,
                runtime_generation,
                0,
                &released,
                &failed,
            );
        }
        assert_eq!(failed.0.lock().unwrap().as_slice(), &[1]);

        let completed = C10S1PendingObserver::default();
        let custody = CaptureDispatchCustody::begin(
            AudioDirection::Microphone,
            runtime_generation,
            0,
            &released,
            &completed,
        );
        custody.complete(&completed);
        assert_eq!(completed.0.lock().unwrap().as_slice(), &[1, 0]);
    }

    fn ready_snapshot() -> RuntimeSnapshot {
        RuntimeSnapshot {
            audio_graph: Some(AudioGraphState {
                health: GraphHealth::Ready,
                endpoints: Vec::new(),
                owned_module_ids: Vec::new(),
                safe_error: None,
            }),
            ..RuntimeSnapshot::default()
        }
    }

    struct TestAudioTargets {
        microphone_capture: String,
        microphone_playback: String,
        speaker_capture: String,
        speaker_playback: String,
    }

    fn test_launch(snapshot: RuntimeSnapshot, targets: TestAudioTargets) -> DuplexLaunch {
        let enabled = |direction| {
            snapshot
                .directions
                .iter()
                .any(|state| state.direction_id == direction && state.enabled)
        };
        DuplexLaunch {
            microphone: enabled(AudioDirection::Microphone).then(|| {
                direction_launch(
                    &snapshot,
                    AudioDirection::Microphone,
                    targets.microphone_capture,
                    targets.microphone_playback,
                    "translator-outgoing-capture",
                    "translator-outgoing-playback",
                    None,
                )
            }),
            speaker: enabled(AudioDirection::Speaker).then(|| {
                direction_launch(
                    &snapshot,
                    AudioDirection::Speaker,
                    targets.speaker_capture,
                    targets.speaker_playback,
                    "translator-incoming-capture",
                    "translator-incoming-playback",
                    None,
                )
            }),
        }
    }

    #[derive(Clone)]
    struct HealthyPrepareTransport {
        stream_task: Arc<AsyncMutex<Option<tokio::task::JoinHandle<()>>>>,
        health_gate: Option<PrepareHealthGate>,
    }

    #[derive(Clone)]
    struct PrepareHealthGate {
        waiting: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        sent: Arc<tokio::sync::Notify>,
    }

    #[tonic::async_trait]
    impl ProviderTransport for HealthyPrepareTransport {
        type StreamStream =
            Pin<Box<dyn Stream<Item = Result<ProviderEvent, Status>> + Send + 'static>>;

        async fn stream(
            &self,
            request: Request<tonic::Streaming<ProviderRequest>>,
        ) -> Result<Response<Self::StreamStream>, Status> {
            let mut requests = request.into_inner();
            let initial = requests
                .message()
                .await?
                .ok_or_else(|| Status::invalid_argument("open request missing"))?;
            let Some(provider_request::Request::OpenSession(open)) = initial.request else {
                return Err(Status::invalid_argument("open request required"));
            };
            let input = open
                .requested_input_format
                .ok_or_else(|| Status::invalid_argument("input format missing"))?;
            let output = open
                .requested_output_format
                .ok_or_else(|| Status::invalid_argument("output format missing"))?;
            let session_id = open.session_id;
            let direction_id = open.direction_id;
            let provider_id = open.provider_id;
            let (events, receiver) = mpsc::channel(1);
            let health_gate = self.health_gate.clone();
            let stream_task = tokio::spawn(async move {
                if events
                    .send(Ok(ProviderEvent {
                        event: Some(provider_event::Event::SessionOpened(
                            ProviderSessionOpened {
                                schema_version: "translator.provider.session_opened.v1".into(),
                                session_id: session_id.clone(),
                                direction_id,
                                negotiated_input_format: Some(input),
                                negotiated_output_format: Some(output),
                                capabilities: Some(ProviderCapabilities {
                                    audio_output: true,
                                    transcript_delta: true,
                                    translation_delta: true,
                                    cancellation: true,
                                    cloud_egress: false,
                                }),
                                event_sequence: 1,
                            },
                        )),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
                if let Some(gate) = &health_gate {
                    gate.waiting.notify_one();
                    gate.release.notified().await;
                }
                if events
                    .send(Ok(ProviderEvent {
                        event: Some(provider_event::Event::Health(ProviderHealth {
                            schema_version: "translator.provider.health.v1".into(),
                            session_id: session_id.clone(),
                            direction_id,
                            event_sequence: 2,
                            provider_id,
                            provider_name: "characterization".into(),
                            state: ProviderState::Ready.into(),
                            models: Vec::new(),
                            queues: Some(ProviderQueues {
                                provider_input_buffered_ms: 0,
                                provider_output_buffered_ms: 0,
                                queue_lag_ms: 0,
                            }),
                            retry: None,
                            safe_error: None,
                        })),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
                if let Some(gate) = &health_gate {
                    let Ok(permit) = events.reserve().await else {
                        return;
                    };
                    drop(permit);
                    gate.sent.notify_one();
                }
                while let Ok(Some(request)) = requests.message().await {
                    let Some(provider_request::Request::CloseSession(_)) = request.request else {
                        continue;
                    };
                    let _ = events
                        .send(Ok(ProviderEvent {
                            event: Some(provider_event::Event::SessionClosed(
                                ProviderSessionClosed {
                                    schema_version: "translator.provider.session_closed.v1".into(),
                                    session_id,
                                    direction_id,
                                    event_sequence: 3,
                                    reason: SessionCloseReason::UserStop.into(),
                                },
                            )),
                        }))
                        .await;
                    break;
                }
            });
            *self.stream_task.lock().await = Some(stream_task);
            Ok(Response::new(Box::pin(
                tokio_stream::wrappers::ReceiverStream::new(receiver),
            )))
        }

        async fn probe(
            &self,
            _: Request<ProviderProbeRequest>,
        ) -> Result<Response<ProviderProbeResponse>, Status> {
            Err(Status::unimplemented("probe not used"))
        }
    }

    type PrepareServer = (
        tempfile::TempDir,
        PathBuf,
        oneshot::Sender<()>,
        tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
        Arc<AsyncMutex<Option<tokio::task::JoinHandle<()>>>>,
    );

    fn start_prepare_server(health_gate: Option<PrepareHealthGate>) -> PrepareServer {
        let directory = tempdir().unwrap();
        fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let socket = directory.path().join("provider.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600)).unwrap();
        let stream_task = Arc::new(AsyncMutex::new(None));
        let transport = HealthyPrepareTransport {
            stream_task: stream_task.clone(),
            health_gate,
        };
        let (stop, stopped) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (connection, _) = listener.accept().await.unwrap();
            let incoming = tokio_stream::once(Ok::<_, std::io::Error>(connection))
                .chain(tokio_stream::pending());
            tonic::transport::Server::builder()
                .add_service(ProviderTransportServer::new(transport))
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = stopped.await;
                })
                .await
        });
        (directory, socket, stop, server, stream_task)
    }

    async fn stop_prepare_server(
        stop: oneshot::Sender<()>,
        mut server: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
        stream_task: Arc<AsyncMutex<Option<tokio::task::JoinHandle<()>>>>,
    ) {
        let mut stream = stream_task
            .lock()
            .await
            .take()
            .expect("stream task missing");
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(2), &mut stream).await,
            Ok(Ok(()))
        ));
        let _ = stop.send(());
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(2), &mut server).await,
            Ok(Ok(Ok(())))
        ));
    }

    fn write_fake_pcm_program(path: &Path, kind: &str) {
        fs::write(
            path,
            format!(
                "#!/bin/sh\numask 077\nexport TRANSLATOR_PREPARE_PCM_KIND='{kind}'\nexec \"$TRANSLATOR_PREPARE_TEST_BINARY\" --exact translation_runtime::tests::pcm_characterization_helper_process --nocapture --test-threads=1\n"
            ),
        )
        .unwrap();
        fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let metadata = fs::symlink_metadata(path).unwrap();
        assert!(metadata.file_type().is_file());
        assert!(!metadata.file_type().is_symlink());
        assert_eq!(metadata.mode() & 0o777, 0o700);
        assert_eq!(
            metadata.uid(),
            fs::metadata(path.parent().unwrap()).unwrap().uid()
        );
    }

    fn parse_marker_identity(path: &Path) -> Option<ProcessIdentity> {
        let contents = fs::read_to_string(path).ok()?;
        if !contents.ends_with('\n') {
            return None;
        }
        let lines = contents.lines().collect::<Vec<_>>();
        assert_eq!(
            lines.len(),
            1,
            "the real PCM constructor must run exactly once"
        );
        let fields = lines[0].split_whitespace().collect::<Vec<_>>();
        assert_eq!(fields.len(), 4, "the PCM identity marker must be complete");
        Some(ProcessIdentity {
            pid: fields[0].parse().unwrap(),
            start_time_ticks: fields[1].parse().unwrap(),
            executable_device: fields[2].parse().unwrap(),
            executable_inode: fields[3].parse().unwrap(),
        })
    }

    async fn marker_identity(path: &Path) -> ProcessIdentity {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(identity) = parse_marker_identity(path) {
                let metadata = fs::symlink_metadata(path).unwrap();
                assert!(metadata.file_type().is_file());
                assert!(!metadata.file_type().is_symlink());
                assert_eq!(metadata.mode() & 0o777, 0o600);
                assert_eq!(
                    metadata.uid(),
                    fs::metadata(path.parent().unwrap()).unwrap().uid()
                );
                assert_eq!(
                    ProcessIdentity::inspect(identity.pid),
                    Some(identity),
                    "the published PCM identity must already be stable"
                );
                return identity;
            }
            assert!(Instant::now() < deadline, "PCM child marker timed out");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn marker_identity_blocking(path: &Path) -> ProcessIdentity {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(identity) = parse_marker_identity(path) {
                assert_eq!(ProcessIdentity::inspect(identity.pid), Some(identity));
                return identity;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "PCM child marker timed out"
            );
            std::thread::yield_now();
        }
    }

    async fn signal_identity(identity: ProcessIdentity) {
        if ProcessIdentity::inspect(identity.pid) != Some(identity) {
            return;
        }
        assert!(
            tokio::process::Command::new("/bin/kill")
                .env_clear()
                .arg("-KILL")
                .arg(identity.pid.to_string())
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await
                .is_ok_and(|status| status.success())
        );
    }

    async fn wait_pid_absent(pid: u32, message: &'static str) {
        let deadline = Instant::now() + Duration::from_secs(2);
        let process_path = format!("/proc/{pid}");
        while Path::new(&process_path).exists() {
            assert!(Instant::now() < deadline, "{message}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn cleanup_failed_characterization(
        child: &mut tokio::process::Child,
        child_identity: ProcessIdentity,
        child_already_reaped: bool,
        markers: &[&Path],
    ) {
        let identities = markers
            .iter()
            .filter_map(|marker| parse_marker_identity(marker))
            .collect::<Vec<_>>();
        for identity in &identities {
            signal_identity(*identity).await;
        }
        let mut child_wait_clean = true;
        if !child_already_reaped {
            if ProcessIdentity::inspect(child_identity.pid) == Some(child_identity) {
                child.start_kill().unwrap();
            }
            child_wait_clean = matches!(
                tokio::time::timeout(Duration::from_secs(2), child.wait()).await,
                Ok(Ok(_))
            );
        }
        wait_pid_absent(
            child_identity.pid,
            "exact characterization child did not exit",
        )
        .await;
        for identity in identities {
            wait_pid_absent(identity.pid, "exact PCM process did not exit").await;
        }
        assert!(child_wait_clean, "characterization child reap failed");
    }

    #[test]
    fn pcm_characterization_helper_process() {
        let Some(kind) = std::env::var_os("TRANSLATOR_PREPARE_PCM_KIND") else {
            return;
        };
        let marker = match kind.to_str().unwrap() {
            "capture" => std::env::var_os("TRANSLATOR_PREPARE_CAPTURE_MARKER").unwrap(),
            "playback" => std::env::var_os("TRANSLATOR_PREPARE_PLAYBACK_MARKER").unwrap(),
            _ => panic!("unexpected PCM helper kind"),
        };
        let identity = ProcessIdentity::inspect(std::process::id()).unwrap();
        let marker = PathBuf::from(marker);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker)
            .unwrap();
        use std::io::Write;
        writeln!(
            file,
            "{} {} {} {}",
            identity.pid,
            identity.start_time_ticks,
            identity.executable_device,
            identity.executable_inode
        )
        .unwrap();
        file.flush().unwrap();
        fs::set_permissions(&marker, std::fs::Permissions::from_mode(0o600)).unwrap();
        if kind == "capture" {
            let block = [0_u8; 640];
            loop {
                std::io::stdout().write_all(&block).unwrap();
                std::io::stdout().flush().unwrap();
            }
        } else {
            std::io::copy(&mut std::io::stdin().lock(), &mut std::io::sink()).unwrap();
        }
    }

    #[tokio::test]
    async fn process_prepare_real_pcm_constructor_characterization() {
        if std::env::var_os("TRANSLATOR_PREPARE_PCM_CHARACTERIZATION_CHILD").is_none() {
            let fixture = tempdir().unwrap();
            fs::set_permissions(fixture.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let metadata = fs::symlink_metadata(fixture.path()).unwrap();
            assert!(metadata.file_type().is_dir());
            assert!(!metadata.file_type().is_symlink());
            assert_eq!(metadata.mode() & 0o777, 0o700);
            let capture_marker = fixture.path().join("capture.pid");
            let playback_marker = fixture.path().join("playback.pid");
            write_fake_pcm_program(&fixture.path().join("parec"), "capture");
            write_fake_pcm_program(&fixture.path().join("pacat"), "playback");

            let test_binary = std::env::current_exe().unwrap();
            let mut command = tokio::process::Command::new(&test_binary);
            command
                .env_clear()
                .env(
                    "PATH",
                    format!("{}:/usr/bin:/bin", fixture.path().display()),
                )
                .env("LANG", "C.UTF-8")
                .env("LC_ALL", "C.UTF-8")
                .env("TRANSLATOR_PREPARE_PCM_CHARACTERIZATION_CHILD", "1")
                .env("TRANSLATOR_PREPARE_TEST_BINARY", &test_binary)
                .env("TRANSLATOR_PREPARE_CAPTURE_MARKER", &capture_marker)
                .env("TRANSLATOR_PREPARE_PLAYBACK_MARKER", &playback_marker)
                .arg("--exact")
                .arg("translation_runtime::tests::process_prepare_real_pcm_constructor_characterization")
                .arg("--nocapture")
                .arg("--test-threads=1")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .kill_on_drop(true);
            let mut child = command
                .spawn()
                .expect("characterization child failed to launch");
            let child_identity = ProcessIdentity::inspect(child.id().unwrap())
                .expect("characterization child identity missing");
            let markers = [capture_marker.as_path(), playback_marker.as_path()];
            let status = match tokio::time::timeout(Duration::from_secs(8), child.wait()).await {
                Ok(Ok(status)) if status.success() => status,
                Ok(Ok(status)) => {
                    cleanup_failed_characterization(&mut child, child_identity, true, &markers)
                        .await;
                    panic!("characterization child failed with {status}");
                }
                Ok(Err(error)) => {
                    cleanup_failed_characterization(&mut child, child_identity, false, &markers)
                        .await;
                    panic!("characterization child wait failed: {error}");
                }
                Err(_) => {
                    cleanup_failed_characterization(&mut child, child_identity, false, &markers)
                        .await;
                    panic!("characterization child timed out");
                }
            };
            assert!(status.success());
            for marker in [&capture_marker, &playback_marker] {
                let identity = parse_marker_identity(marker).unwrap();
                assert_ne!(ProcessIdentity::inspect(identity.pid), Some(identity));
                assert!(!Path::new(&format!("/proc/{}", identity.pid)).exists());
            }
            assert_ne!(
                ProcessIdentity::inspect(child_identity.pid),
                Some(child_identity)
            );
            assert!(!Path::new(&format!("/proc/{}", child_identity.pid)).exists());
            return;
        }

        let (directory, socket, stop, server, stream_task) = start_prepare_server(None);
        let launch = test_launch(
            ready_snapshot(),
            TestAudioTargets {
                microphone_capture: "test-microphone".to_owned(),
                microphone_playback: "test-microphone-output".to_owned(),
                speaker_capture: "test-speaker".to_owned(),
                speaker_playback: "test-speaker-output".to_owned(),
            },
        )
        .microphone
        .unwrap();
        let effects = ProcessDirectionEffects::new(ProcessDuplexConfig {
            python: PathBuf::from("unused-python"),
            sidecar_root: PathBuf::from("unused-sidecar"),
            socket_path: socket.clone(),
            expected_uid: fs::metadata(directory.path()).unwrap().uid(),
        });
        let mut owner = effects.begin(launch);
        let generation = crate::SidecarLaunch {
            generation_id: uuid::Uuid::new_v4(),
            token: "ab".repeat(32),
        };
        effects
            .prepare(
                &mut owner,
                &generation,
                Instant::now() + Duration::from_secs(3),
            )
            .await
            .unwrap();
        assert!(owner.provider.is_some());
        assert!(owner.capture.is_some());
        assert!(owner.playback.is_some());
        let capture_marker =
            PathBuf::from(std::env::var_os("TRANSLATOR_PREPARE_CAPTURE_MARKER").unwrap());
        let playback_marker =
            PathBuf::from(std::env::var_os("TRANSLATOR_PREPARE_PLAYBACK_MARKER").unwrap());
        let capture_identity = marker_identity(&capture_marker).await;
        let playback_identity = marker_identity(&playback_marker).await;
        assert_eq!(
            owner
                .playback
                .as_ref()
                .and_then(PulsePcmPlayback::process_identity),
            Some(playback_identity)
        );

        effects
            .stop_pcm(&mut owner, Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        effects
            .close_provider(
                &mut owner,
                CloseRequestReason::UserStop,
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap();
        assert!(ProcessDirectionEffects::is_clean(&owner));
        assert_ne!(
            ProcessIdentity::inspect(capture_identity.pid),
            Some(capture_identity)
        );
        assert!(!Path::new(&format!("/proc/{}", capture_identity.pid)).exists());
        assert_ne!(
            ProcessIdentity::inspect(playback_identity.pid),
            Some(playback_identity)
        );
        assert!(!Path::new(&format!("/proc/{}", playback_identity.pid)).exists());
        stop_prepare_server(stop, server, stream_task).await;
    }

    #[test]
    fn process_prepare_has_one_production_pcm_constructor_path() {
        let source = include_str!("translation_runtime.rs");
        let production = source
            .rsplit_once("#[cfg(test)]\npub(crate) mod tests")
            .expect("production/test boundary missing")
            .0;
        let process_effects = production
            .split_once("struct ProcessDirectionEffects")
            .and_then(|(_, source)| source.split_once("fn direction_start_error"))
            .expect("process direction effects source boundary missing")
            .0;
        assert_eq!(process_effects.matches("PulsePcmCapture::spawn").count(), 1);
        assert_eq!(
            process_effects.matches("PulsePcmPlayback::spawn").count(),
            1
        );
        assert_eq!(
            process_effects.matches("prepare_with_pcm_spawners").count(),
            2,
            "one private implementation and one production delegation are required"
        );
    }

    #[tokio::test]
    async fn invalid_aec_reservation_has_zero_provider_or_pcm_effects() {
        let directory = tempdir().unwrap();
        let mut launch = test_launch(
            ready_snapshot(),
            TestAudioTargets {
                microphone_capture: "test-microphone".to_owned(),
                microphone_playback: "test-microphone-output".to_owned(),
                speaker_capture: "test-speaker".to_owned(),
                speaker_playback: "test-speaker-output".to_owned(),
            },
        )
        .microphone
        .unwrap();
        launch.aec_reservation = Some(Arc::new(crate::AecStartReservation::for_test(
            "test-microphone",
            "test-microphone-output",
        )));
        let effects = ProcessDirectionEffects::new(ProcessDuplexConfig {
            python: PathBuf::from("unused-python"),
            sidecar_root: PathBuf::from("unused-sidecar"),
            socket_path: directory.path().join("provider-never-opened.sock"),
            expected_uid: fs::metadata(directory.path()).unwrap().uid(),
        });
        let mut owner = effects.begin(launch);
        let generation = crate::SidecarLaunch {
            generation_id: uuid::Uuid::new_v4(),
            token: "ab".repeat(32),
        };
        let capture_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let playback_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let capture_spy = capture_calls.clone();
        let playback_spy = playback_calls.clone();

        let result = effects
            .prepare_with_pcm_spawners(
                &mut owner,
                &generation,
                Instant::now() + Duration::from_secs(1),
                move |_| {
                    capture_spy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err(translator_audio::PulsePcmError::Start)
                },
                move |_| {
                    playback_spy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err(translator_audio::PulsePcmError::Start)
                },
            )
            .await;

        assert_eq!(result, Err(FaultScope::Local));
        assert!(owner.provider.is_none());
        assert_eq!(capture_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(playback_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn process_prepare_ready_at_deadline_does_not_enter_pcm_spawners() {
        let gate = PrepareHealthGate {
            waiting: Arc::new(tokio::sync::Notify::new()),
            release: Arc::new(tokio::sync::Notify::new()),
            sent: Arc::new(tokio::sync::Notify::new()),
        };
        let (directory, socket, stop, server, stream_task) =
            start_prepare_server(Some(gate.clone()));
        let launch = test_launch(
            ready_snapshot(),
            TestAudioTargets {
                microphone_capture: "test-microphone".to_owned(),
                microphone_playback: "test-microphone-output".to_owned(),
                speaker_capture: "test-speaker".to_owned(),
                speaker_playback: "test-speaker-output".to_owned(),
            },
        )
        .microphone
        .unwrap();
        let effects = ProcessDirectionEffects::new(ProcessDuplexConfig {
            python: PathBuf::from("unused-python"),
            sidecar_root: PathBuf::from("unused-sidecar"),
            socket_path: socket,
            expected_uid: fs::metadata(directory.path()).unwrap().uid(),
        });
        let mut owner = effects.begin(launch);
        let generation = crate::SidecarLaunch {
            generation_id: uuid::Uuid::new_v4(),
            token: "ab".repeat(32),
        };
        let capture_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let playback_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let deadline = Instant::now() + Duration::from_millis(100);
        let capture_spy = capture_calls.clone();
        let playback_spy = playback_calls.clone();
        let result = {
            let attempt = effects.prepare_with_pcm_spawners(
                &mut owner,
                &generation,
                deadline,
                move |_| -> Result<PulsePcmCapture, translator_audio::PulsePcmError> {
                    capture_spy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err(translator_audio::PulsePcmError::Start)
                },
                move |_| -> Result<PulsePcmPlayback, translator_audio::PulsePcmError> {
                    playback_spy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err(translator_audio::PulsePcmError::Start)
                },
            );
            tokio::pin!(attempt);
            tokio::select! {
                biased;
                _ = gate.waiting.notified() => {}
                result = &mut attempt => panic!("prepare ended before the Ready boundary: {result:?}"),
            }
            for _ in 0..64 {
                tokio::select! {
                    biased;
                    result = &mut attempt => panic!("prepare ended before Health: {result:?}"),
                    _ = tokio::task::yield_now() => {}
                }
            }
            gate.release.notify_one();
            gate.sent.notified().await;
            for _ in 0..16 {
                tokio::task::yield_now().await;
            }
            tokio::time::advance(Duration::from_millis(100)).await;
            attempt.await
        };
        tokio::time::resume();
        let provider_retained = owner.provider.is_some();
        let expired_cleanup = effects
            .close_provider(&mut owner, CloseRequestReason::UserStop, Instant::now())
            .await;
        let retained_after_expiry = owner.provider.is_some();
        effects
            .close_provider(
                &mut owner,
                CloseRequestReason::UserStop,
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap();
        stop_prepare_server(stop, server, stream_task).await;

        assert_eq!(result, Err(FaultScope::ProviderConnection));
        assert_eq!(capture_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(playback_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(provider_retained);
        assert_eq!(expired_cleanup, Err(DuplexRuntimeError::StopFailed));
        assert!(retained_after_expiry);
        assert!(ProcessDirectionEffects::is_clean(&owner));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn process_prepare_capture_crossing_deadline_retains_owner_without_playback() {
        if std::env::var_os("TRANSLATOR_PREPARE_CAPTURE_DEADLINE_CHILD").is_none() {
            let fixture = tempdir().unwrap();
            fs::set_permissions(fixture.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
            let capture_marker = fixture.path().join("capture.pid");
            let playback_marker = fixture.path().join("playback.pid");
            write_fake_pcm_program(&fixture.path().join("parec"), "capture");
            write_fake_pcm_program(&fixture.path().join("pacat"), "playback");
            let test_binary = std::env::current_exe().unwrap();
            let mut command = tokio::process::Command::new(&test_binary);
            command
                .env_clear()
                .env(
                    "PATH",
                    format!("{}:/usr/bin:/bin", fixture.path().display()),
                )
                .env("LANG", "C.UTF-8")
                .env("LC_ALL", "C.UTF-8")
                .env("TRANSLATOR_PREPARE_CAPTURE_DEADLINE_CHILD", "1")
                .env("TRANSLATOR_PREPARE_TEST_BINARY", &test_binary)
                .env("TRANSLATOR_PREPARE_CAPTURE_MARKER", &capture_marker)
                .env("TRANSLATOR_PREPARE_PLAYBACK_MARKER", &playback_marker)
                .arg("--exact")
                .arg("translation_runtime::tests::process_prepare_capture_crossing_deadline_retains_owner_without_playback")
                .arg("--nocapture")
                .arg("--test-threads=1")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .kill_on_drop(true);
            let mut child = command.spawn().expect("deadline child failed to launch");
            let child_identity = ProcessIdentity::inspect(child.id().unwrap())
                .expect("deadline child identity missing");
            let markers = [capture_marker.as_path(), playback_marker.as_path()];
            let status = match tokio::time::timeout(Duration::from_secs(8), child.wait()).await {
                Ok(Ok(status)) if status.success() => status,
                Ok(Ok(status)) => {
                    cleanup_failed_characterization(&mut child, child_identity, true, &markers)
                        .await;
                    panic!("deadline child failed with {status}");
                }
                Ok(Err(error)) => {
                    cleanup_failed_characterization(&mut child, child_identity, false, &markers)
                        .await;
                    panic!("deadline child wait failed: {error}");
                }
                Err(_) => {
                    cleanup_failed_characterization(&mut child, child_identity, false, &markers)
                        .await;
                    panic!("deadline child timed out");
                }
            };
            assert!(status.success());
            let capture_identity = parse_marker_identity(&capture_marker).unwrap();
            assert!(!Path::new(&format!("/proc/{}", capture_identity.pid)).exists());
            assert!(!playback_marker.exists());
            assert!(!Path::new(&format!("/proc/{}", child_identity.pid)).exists());
            return;
        }

        let (directory, socket, stop, server, stream_task) = start_prepare_server(None);
        let launch = test_launch(
            ready_snapshot(),
            TestAudioTargets {
                microphone_capture: "test-microphone".to_owned(),
                microphone_playback: "test-microphone-output".to_owned(),
                speaker_capture: "test-speaker".to_owned(),
                speaker_playback: "test-speaker-output".to_owned(),
            },
        )
        .microphone
        .unwrap();
        let effects = ProcessDirectionEffects::new(ProcessDuplexConfig {
            python: PathBuf::from("unused-python"),
            sidecar_root: PathBuf::from("unused-sidecar"),
            socket_path: socket,
            expected_uid: fs::metadata(directory.path()).unwrap().uid(),
        });
        let mut owner = effects.begin(launch);
        let generation = crate::SidecarLaunch {
            generation_id: uuid::Uuid::new_v4(),
            token: "ab".repeat(32),
        };
        let capture_marker =
            PathBuf::from(std::env::var_os("TRANSLATOR_PREPARE_CAPTURE_MARKER").unwrap());
        let (entered, entered_result) = std_mpsc::sync_channel(1);
        let (release, released) = std_mpsc::sync_channel(1);
        let playback_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let playback_spy = playback_calls.clone();
        let deadline = Instant::now() + Duration::from_millis(500);
        let release_after_ns = monotonic_ns().saturating_add(600_000_000);
        let barrier = std::thread::spawn(move || {
            let identity = entered_result
                .recv_timeout(Duration::from_secs(2))
                .expect("capture spawner was not entered");
            while monotonic_ns() <= release_after_ns {
                std::thread::yield_now();
            }
            release.send(()).unwrap();
            identity
        });
        let result = effects
            .prepare_with_pcm_spawners(
                &mut owner,
                &generation,
                deadline,
                move |command| -> Result<PulsePcmCapture, translator_audio::PulsePcmError> {
                    let capture = PulsePcmCapture::spawn(command)?;
                    let identity = marker_identity_blocking(&capture_marker);
                    entered.send(identity).unwrap();
                    released.recv_timeout(Duration::from_secs(2)).unwrap();
                    Ok(capture)
                },
                move |_| -> Result<PulsePcmPlayback, translator_audio::PulsePcmError> {
                    playback_spy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err(translator_audio::PulsePcmError::Start)
                },
            )
            .await;
        let capture_identity = barrier.join().unwrap();
        let capture_retained = owner.capture.is_some()
            && ProcessIdentity::inspect(capture_identity.pid) == Some(capture_identity);
        let expired_cleanup = effects.stop_pcm(&mut owner, Instant::now()).await;
        let retained_after_expiry = owner.capture.is_some()
            && ProcessIdentity::inspect(capture_identity.pid) == Some(capture_identity);
        effects
            .stop_pcm(&mut owner, Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        effects
            .close_provider(
                &mut owner,
                CloseRequestReason::UserStop,
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap();
        stop_prepare_server(stop, server, stream_task).await;

        assert_eq!(result, Err(FaultScope::Local));
        assert!(Instant::now() >= deadline);
        assert_eq!(playback_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(capture_retained);
        assert_eq!(expired_cleanup, Err(DuplexRuntimeError::StopFailed));
        assert!(retained_after_expiry);
        assert!(!Path::new(&format!("/proc/{}", capture_identity.pid)).exists());
        assert!(ProcessDirectionEffects::is_clean(&owner));
    }

    fn observe_latency_breach(
        observer: &dyn DuplexRuntimeObserver,
        direction: AudioDirection,
        capture_monotonic_ns: u64,
    ) -> uuid::Uuid {
        let utterance_id = uuid::Uuid::new_v4();
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction,
            utterance_id,
            capture_monotonic_ns,
        });
        observer.observe(DuplexRuntimeEvent::AudioFrame {
            direction,
            utterance_id,
            sequence: 0,
            provider_monotonic_ns: capture_monotonic_ns + 4_000_000_000,
            observed_monotonic_ns: capture_monotonic_ns + 4_000_000_000,
            queue_lag_ms: 20,
        });
        utterance_id
    }

    struct CancelProbe(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for CancelProbe {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    struct HotTestOwner {
        id: u64,
        cleanups: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl HotTestOwner {
        async fn pending(
            &mut self,
            entered: oneshot::Sender<()>,
            cancelled: Arc<std::sync::atomic::AtomicBool>,
        ) -> Result<u64, &'static str> {
            let _probe = CancelProbe(cancelled);
            let _ = entered.send(());
            std::future::pending::<()>().await;
            Ok(self.id)
        }

        fn cleanup(self) {
            self.cleanups
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    async fn assert_provider_timeout_uses_minimum(
        id: u64,
        phase_after: Duration,
        expected_after: Duration,
    ) {
        let cleanups = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (_stop, mut stop_receiver) = watch::channel(None);
        let admitted = tokio::time::Instant::now();
        let phase_deadline = admitted + phase_after;
        let (entered, entered_result) = oneshot::channel();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut operation = tokio::spawn({
            let cleanups = cleanups.clone();
            let cancelled = cancelled.clone();
            async move {
                let mut owner = HotTestOwner { id, cleanups };
                let result = await_provider_send(
                    &mut stop_receiver,
                    Some(phase_deadline),
                    owner.pending(entered, cancelled),
                )
                .await;
                (owner, result)
            }
        });
        entered_result
            .await
            .expect("the provider send must enter before its absolute deadline");
        tokio::time::advance(expected_after - Duration::from_millis(1)).await;
        assert!(!operation.is_finished());
        tokio::time::advance(Duration::from_millis(1)).await;
        let (owner, timed_out) = (&mut operation).await.unwrap();
        assert_eq!(
            timed_out,
            HotIoResult::TimedOut {
                kind: HotIoKind::ProviderSend,
            }
        );
        assert_eq!(tokio::time::Instant::now(), admitted + expected_after);
        assert_eq!(owner.id, id);
        assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));
        owner.cleanup();
        assert_eq!(cleanups.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    async fn assert_playback_timeout_uses_minimum(
        id: u64,
        phase_after: Duration,
        expected_after: Duration,
    ) {
        let cleanups = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (_stop, mut stop_receiver) = watch::channel(None);
        let admitted = tokio::time::Instant::now();
        let phase_deadline = admitted + phase_after;
        let (entered, entered_result) = oneshot::channel();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut operation = tokio::spawn({
            let cleanups = cleanups.clone();
            let cancelled = cancelled.clone();
            async move {
                let mut owner = HotTestOwner { id, cleanups };
                let mut reusable = true;
                let result = await_playback_write(
                    &mut reusable,
                    &mut stop_receiver,
                    Some(phase_deadline),
                    owner.pending(entered, cancelled),
                )
                .await;
                (owner, reusable, result)
            }
        });
        entered_result
            .await
            .expect("the playback write must enter before its absolute deadline");
        tokio::time::advance(expected_after - Duration::from_millis(1)).await;
        assert!(!operation.is_finished());
        tokio::time::advance(Duration::from_millis(1)).await;
        let (owner, mut reusable, timed_out) = (&mut operation).await.unwrap();
        assert_eq!(
            timed_out,
            HotIoResult::TimedOut {
                kind: HotIoKind::PlaybackWrite,
            }
        );
        assert!(!reusable);
        assert_eq!(tokio::time::Instant::now(), admitted + expected_after);
        assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));

        let polled_after_timeout = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let attempted = polled_after_timeout.clone();
        let (_replay_stop, mut replay_stop_receiver) = watch::channel(None);
        let replay =
            await_playback_write(&mut reusable, &mut replay_stop_receiver, None, async move {
                attempted.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok::<(), &'static str>(())
            })
            .await;
        assert_eq!(
            replay,
            HotIoResult::NotReusable {
                kind: HotIoKind::PlaybackWrite,
            }
        );
        assert!(!polled_after_timeout.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(owner.id, id);
        owner.cleanup();
        assert_eq!(cleanups.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn provider_send_stop_and_timeout_retain_the_exact_owner_and_typed_result() {
        let completed = {
            let (_stop, mut stop_receiver) = watch::channel(None);
            await_provider_send(&mut stop_receiver, None, async {
                Err::<u64, _>("typed-provider-error")
            })
            .await
        };
        assert_eq!(
            completed,
            HotIoResult::Completed(Err("typed-provider-error")),
            "the bounded helper must preserve the concrete provider result"
        );

        let cleanups = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (stop, mut stop_receiver) = watch::channel(None);
        let (entered, entered_result) = oneshot::channel();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cleanup_deadline = tokio::time::Instant::now() + RUNTIME_CLEANUP_BUDGET;
        let operation = tokio::spawn({
            let cleanups = cleanups.clone();
            let cancelled = cancelled.clone();
            async move {
                let mut owner = HotTestOwner { id: 71, cleanups };
                let result = await_provider_send(
                    &mut stop_receiver,
                    None,
                    owner.pending(entered, cancelled),
                )
                .await;
                (owner, result)
            }
        });
        entered_result
            .await
            .expect("the provider send must enter before Stop wins");
        stop.send(Some(WorkerStop::Close {
            reason: CloseRequestReason::DaemonShutdown,
            deadline: cleanup_deadline,
        }))
        .unwrap();
        let (owner, stopped) = operation.await.unwrap();
        assert!(matches!(
            stopped,
            HotIoResult::Stopped {
                kind: HotIoKind::ProviderSend,
                stop: WorkerStop::Close {
                    reason: CloseRequestReason::DaemonShutdown,
                    deadline,
                },
            } if deadline == cleanup_deadline
        ));
        assert_eq!(owner.id, 71);
        assert_eq!(cleanups.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));
        owner.cleanup();
        assert_eq!(cleanups.load(std::sync::atomic::Ordering::SeqCst), 1);

        assert_provider_timeout_uses_minimum(
            72,
            Duration::from_millis(250),
            Duration::from_millis(250),
        )
        .await;
        assert_provider_timeout_uses_minimum(73, Duration::from_secs(2), HOT_IO_LIVENESS_TIMEOUT)
            .await;
        assert_eq!(HOT_IO_LIVENESS_TIMEOUT, Duration::from_millis(1_000));
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn playback_write_stop_and_phase_timeout_make_owner_non_reusable_before_cleanup() {
        let cleanups = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (stop, mut stop_receiver) = watch::channel(None);
        let (entered, entered_result) = oneshot::channel();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cleanup_deadline = tokio::time::Instant::now() + RUNTIME_CLEANUP_BUDGET;
        let operation = tokio::spawn({
            let cleanups = cleanups.clone();
            let cancelled = cancelled.clone();
            async move {
                let mut owner = HotTestOwner { id: 81, cleanups };
                let mut reusable = true;
                let result = await_playback_write(
                    &mut reusable,
                    &mut stop_receiver,
                    None,
                    owner.pending(entered, cancelled),
                )
                .await;
                (owner, reusable, result)
            }
        });
        entered_result
            .await
            .expect("the playback write must enter before Stop wins");
        stop.send(Some(WorkerStop::Close {
            reason: CloseRequestReason::DaemonShutdown,
            deadline: cleanup_deadline,
        }))
        .unwrap();
        let (owner, reusable, stopped) = operation.await.unwrap();
        assert!(
            !reusable,
            "a cancelled partial playback write cannot be reused"
        );
        assert!(matches!(
            stopped,
            HotIoResult::Stopped {
                kind: HotIoKind::PlaybackWrite,
                stop: WorkerStop::Close { deadline, .. },
            } if deadline == cleanup_deadline
        ));
        assert_eq!(owner.id, 81);
        assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));
        owner.cleanup();
        assert_eq!(cleanups.load(std::sync::atomic::Ordering::SeqCst), 1);

        assert_playback_timeout_uses_minimum(
            82,
            Duration::from_millis(250),
            Duration::from_millis(250),
        )
        .await;
        assert_playback_timeout_uses_minimum(83, Duration::from_secs(2), HOT_IO_LIVENESS_TIMEOUT)
            .await;
    }

    #[test]
    fn production_direction_loop_routes_every_hot_write_through_the_bounded_helpers() {
        let source = include_str!("translation_runtime.rs");
        let loop_source = source
            .split_once("async fn run_direction_loop")
            .unwrap()
            .1
            .split_once("fn extend_playback_deadline")
            .unwrap()
            .0;
        let normalized = loop_source.split_whitespace().collect::<String>();
        assert!(!normalized.contains(".provider.send("));
        assert!(!normalized.contains(".write_frame("));
        assert_eq!(normalized.matches("send_provider(").count(), 4);
        assert_eq!(normalized.matches("write_playback_frame(").count(), 1);
        assert_eq!(normalized.matches("stop_playback_for_reset(").count(), 2);
        let helpers = source
            .split_once("async fn stop_playback_for_reset")
            .unwrap()
            .1
            .split_once("async fn run_direction_loop")
            .unwrap()
            .0
            .split_whitespace()
            .collect::<String>();
        assert_eq!(helpers.matches("watchdog_phase_deadline(").count(), 3);
    }

    #[test]
    fn terminal_reap_keeps_thread_ownership_until_completion_is_observable() {
        let (stop_sender, _) = watch::channel(None);
        let (command_sender, _) = mpsc::channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std_mpsc::sync_channel(1);
        let (sent_sender, sent_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            release_receiver.recv().unwrap();
            done_sender
                .send(Err(DuplexRuntimeError::StartFailed))
                .unwrap();
            sent_sender.send(()).unwrap();
        });
        let mut runtime =
            ProcessActiveDuplex::new(stop_sender, command_sender, done_receiver, thread);

        assert_eq!(
            runtime.reap(Instant::now()),
            Err(DuplexRuntimeError::StopFailed)
        );
        assert!(runtime.thread.is_some());
        release_sender.send(()).unwrap();
        sent_receiver.recv().unwrap();
        runtime
            .reap(Instant::now() + Duration::from_secs(1))
            .unwrap();
        assert!(runtime.thread.is_none());
    }

    #[test]
    fn cleaned_start_ack_without_done_returns_owner_instead_of_joining_unbounded() {
        let (stop_sender, _) = watch::channel(None);
        let (command_sender, command_receiver) = mpsc::channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std_mpsc::sync_channel(1);
        let (exited_sender, exited_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            let _command_receiver = command_receiver;
            release_receiver.recv().unwrap();
            drop(done_sender);
            exited_sender.send(()).unwrap();
        });
        let runtime = ProcessActiveDuplex::new(stop_sender, command_sender, done_receiver, thread);

        let started = std::time::Instant::now();
        let failure = match resolve_start_ack(
            Ok(StartAck::Rejected(DuplexRuntimeError::StartFailed)),
            runtime,
            Instant::now() + Duration::from_millis(10),
        ) {
            Ok(_) => panic!("a rejected start cannot return a running runtime"),
            Err(failure) => failure,
        };
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(failure.has_cleanup());

        let (_, mut cleanup) = failure.into_parts();
        release_sender.send(()).unwrap();
        exited_receiver.recv().unwrap();
        cleanup
            .as_mut()
            .unwrap()
            .stop(Instant::now() + Duration::from_secs(1))
            .unwrap();
    }

    #[test]
    fn start_ack_timeout_requests_stop_and_returns_owner_without_joining() {
        let (stop_sender, stop_receiver) = watch::channel(None);
        let (command_sender, command_receiver) = mpsc::channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            let _command_receiver = command_receiver;
            release_receiver.recv().unwrap();
            done_sender
                .send(Err(DuplexRuntimeError::StartFailed))
                .unwrap();
        });
        let runtime = ProcessActiveDuplex::new(stop_sender, command_sender, done_receiver, thread);

        let started = std::time::Instant::now();
        let failure = match resolve_start_ack(
            Err(DuplexRuntimeError::StartFailed),
            runtime,
            Instant::now() + Duration::from_millis(10),
        ) {
            Ok(_) => panic!("a timed out start cannot return a running runtime"),
            Err(failure) => failure,
        };
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(stop_receiver.borrow().is_some());
        assert!(failure.has_cleanup());

        let (_, mut cleanup) = failure.into_parts();
        release_sender.send(()).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while cleanup
            .as_mut()
            .unwrap()
            .reap(Instant::now() + Duration::from_millis(10))
            .is_err()
        {
            assert!(
                std::time::Instant::now() < deadline,
                "the retained timeout owner must become reapable"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn repeated_stop_timeouts_retain_the_same_native_thread_until_done() {
        let (stop_sender, _) = watch::channel(None);
        let (command_sender, mut command_receiver) = mpsc::channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std_mpsc::sync_channel(1);
        let (worker_done_sender, worker_done_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            release_receiver.recv().unwrap();
            if let Some(RuntimeCommand::Stop { response, .. }) = command_receiver.blocking_recv() {
                let _ = response.send(Ok(()));
            }
            done_sender.send(Ok(())).unwrap();
            worker_done_sender.send(()).unwrap();
        });
        let runtime = ProcessActiveDuplex::new(stop_sender, command_sender, done_receiver, thread);
        let (result_sender, result_receiver) = std_mpsc::sync_channel(1);
        let caller = thread::spawn(move || {
            let mut runtime = runtime;
            let first_started = std::time::Instant::now();
            let first = runtime.stop_until(Instant::now() + Duration::from_millis(10));
            let first_elapsed = first_started.elapsed();
            let second_started = std::time::Instant::now();
            let second = runtime.stop_until(Instant::now() + Duration::from_millis(10));
            let second_elapsed = second_started.elapsed();
            result_sender
                .send((runtime, first, first_elapsed, second, second_elapsed))
                .unwrap();
        });

        let result = match result_receiver.recv_timeout(Duration::from_millis(250)) {
            Ok(result) => result,
            Err(_) => {
                release_sender.send(()).unwrap();
                let (mut runtime, ..) = result_receiver
                    .recv_timeout(Duration::from_secs(1))
                    .expect("the fail-safe must unblock and return the retained owner");
                let _ = runtime.stop_until(Instant::now() + Duration::from_secs(1));
                caller.join().unwrap();
                panic!("both Stop attempts must include command admission in their budget");
            }
        };
        let (mut runtime, first, first_elapsed, second, second_elapsed) = result;
        assert_eq!(first, Err(DuplexRuntimeError::StopFailed));
        assert_eq!(second, Err(DuplexRuntimeError::StopFailed));
        assert!(first_elapsed < Duration::from_millis(100));
        assert!(second_elapsed < Duration::from_millis(100));
        assert!(runtime.thread.is_some());

        release_sender.send(()).unwrap();
        worker_done_receiver.recv().unwrap();
        runtime
            .stop_until(Instant::now() + Duration::from_secs(1))
            .unwrap();
        assert!(runtime.thread.is_none());
        caller.join().unwrap();
    }

    #[test]
    fn stop_command_carries_the_exact_public_admission_deadline_without_rejuvenation() {
        let (stop_sender, _) = watch::channel(None);
        let (command_sender, mut command_receiver) = mpsc::channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let (deadline_sender, deadline_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            let RuntimeCommand::Stop { response, deadline } =
                command_receiver.blocking_recv().unwrap()
            else {
                panic!("only Stop is admitted in this fixture");
            };
            deadline_sender.send(deadline).unwrap();
            response.send(Ok(())).unwrap();
            done_sender.send(Ok(())).unwrap();
        });
        let mut runtime =
            ProcessActiveDuplex::new(stop_sender, command_sender, done_receiver, thread);
        let admitted_deadline = tokio::time::Instant::now() + Duration::from_millis(50);
        runtime.stop_until(admitted_deadline).unwrap();
        let carried = deadline_receiver.recv().unwrap();
        assert_eq!(carried, admitted_deadline);
        assert!(runtime.thread.is_none());
    }

    #[test]
    fn explicit_audio_targets_are_preserved_without_production_device_resolution() {
        let targets = TestAudioTargets {
            microphone_capture: "benchmark_ru_source.monitor".to_owned(),
            microphone_playback: "benchmark_mic_sink".to_owned(),
            speaker_capture: "benchmark_remote_sink.monitor".to_owned(),
            speaker_playback: "benchmark_headphones".to_owned(),
        };

        let launch = test_launch(ready_snapshot(), targets);
        let microphone = launch.microphone.as_ref().unwrap();
        let speaker = launch.speaker.as_ref().unwrap();

        assert_eq!(microphone.capture_device, "benchmark_ru_source.monitor");
        assert_eq!(microphone.playback_device, "benchmark_mic_sink");
        assert_eq!(speaker.capture_device, "benchmark_remote_sink.monitor");
        assert_eq!(speaker.playback_device, "benchmark_headphones");
    }

    #[test]
    fn disabled_direction_is_not_prepared_for_launch() {
        let mut snapshot = ready_snapshot();
        snapshot.directions[1].enabled = false;
        snapshot.directions[1].target_language = snapshot.directions[1].source_language;
        snapshot.directions[1].voice_profile.language = snapshot.directions[1].target_language;

        let launch = test_launch(
            snapshot,
            TestAudioTargets {
                microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "benchmark_headphones".to_owned(),
            },
        );

        assert!(launch.microphone.is_some());
        assert!(launch.speaker.is_none());
    }

    #[test]
    fn openai_provider_launches_after_cloud_provider_selection() {
        let mut snapshot = ready_snapshot();
        snapshot.provider_id = ProviderId::Openai;
        snapshot.audio_leaves_machine = true;
        for direction in &mut snapshot.directions {
            direction.voice_profile.engine = VoiceEngine::Openai;
        }
        let launch = test_launch(
            snapshot,
            TestAudioTargets {
                microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "benchmark_headphones".to_owned(),
            },
        );
        let microphone = launch.microphone.as_ref().unwrap();
        let speaker = launch.speaker.as_ref().unwrap();

        assert_eq!(microphone.runtime.provider_id, ProviderId::Openai);
        assert_eq!(speaker.runtime.provider_id, ProviderId::Openai);
        assert_eq!(
            DirectionSession::new(microphone.runtime)
                .provider_contract()
                .provider_id,
            translator_ipc::provider::ProviderId::Openai
        );
    }

    #[test]
    fn audio_frame_is_observed_only_after_a_successful_playback_write() {
        let observer = RecordingObserver::default();
        let utterance_id = uuid::Uuid::new_v4();
        let metadata = QueuedPlaybackMetadata {
            utterance_id,
            sequence: 7,
            provider_monotonic_ns: 80_000_000,
            enqueued_monotonic_ns: 100_000_000,
        };

        let failed = observe_playback_write(
            Err::<(), _>("write failed"),
            AudioDirection::Speaker,
            metadata,
            140_000_000,
            &observer,
        );
        assert_eq!(failed, Err("write failed"));
        assert!(observer.events.lock().unwrap().is_empty());

        observe_playback_write(
            Ok::<(), &str>(()),
            AudioDirection::Speaker,
            metadata,
            145_000_000,
            &observer,
        )
        .unwrap();
        assert_eq!(
            observer.events.lock().unwrap().as_slice(),
            &[DuplexRuntimeEvent::AudioFrame {
                direction: AudioDirection::Speaker,
                utterance_id,
                sequence: 7,
                provider_monotonic_ns: 80_000_000,
                observed_monotonic_ns: 145_000_000,
                queue_lag_ms: 45,
            }]
        );
    }

    #[test]
    fn playback_deadline_accumulates_buffered_frames_and_resets_after_idle() {
        assert_eq!(extend_playback_deadline(0, 100_000_000, 20), 120_000_000);
        assert_eq!(
            extend_playback_deadline(120_000_000, 105_000_000, 20),
            140_000_000
        );
        assert_eq!(
            extend_playback_deadline(140_000_000, 200_000_000, 20),
            220_000_000
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn source_p1_mode_change_drain_never_waits_past_the_hot_io_liveness_ceiling() {
        let distant_deadline_ns = monotonic_ns().saturating_add(60_000_000_000);
        let (_stop, mut stop_receiver) = watch::channel(None);
        let drain = tokio::spawn(async move {
            wait_for_playback_deadline(distant_deadline_ns, &mut stop_receiver).await
        });
        tokio::task::yield_now().await;
        tokio::time::advance(HOT_IO_LIVENESS_TIMEOUT - Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert!(!drain.is_finished());
        tokio::time::advance(Duration::from_millis(1)).await;
        for _ in 0..16 {
            if drain.is_finished() {
                break;
            }
            tokio::task::yield_now().await;
        }
        let bounded = drain.is_finished();
        if !bounded {
            drain.abort();
        }
        let _ = drain.await;
        assert!(
            bounded,
            "a mode-change drain must stop at the existing one-second hot-I/O ceiling"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn source_p1_short_mode_change_drain_reports_normal_completion() {
        let short_deadline_ns = monotonic_ns().saturating_add(20_000_000);
        let (_short_stop, mut short_stop_receiver) = watch::channel(None);
        let short = tokio::spawn(async move {
            wait_for_playback_deadline(short_deadline_ns, &mut short_stop_receiver).await
        });
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_millis(19)).await;
        tokio::task::yield_now().await;
        assert!(!short.is_finished());
        tokio::time::advance(Duration::from_millis(1)).await;
        assert_eq!(short.await.unwrap(), Ok(None));
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn source_p1_distant_mode_change_drain_reports_typed_playback_timeout() {
        let distant_deadline_ns = monotonic_ns().saturating_add(60_000_000_000);
        let (_timeout_stop, mut timeout_stop_receiver) = watch::channel(None);
        let timed_out = tokio::spawn(async move {
            wait_for_playback_deadline(distant_deadline_ns, &mut timeout_stop_receiver).await
        });
        tokio::task::yield_now().await;
        tokio::time::advance(HOT_IO_LIVENESS_TIMEOUT - Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert!(!timed_out.is_finished());
        tokio::time::advance(Duration::from_millis(1)).await;
        for _ in 0..16 {
            if timed_out.is_finished() {
                break;
            }
            tokio::task::yield_now().await;
        }
        let bounded = timed_out.is_finished();
        if !bounded {
            timed_out.abort();
        }
        let result = timed_out.await;
        assert!(
            bounded,
            "the typed playback failure must occur at one second"
        );
        assert_eq!(result.unwrap(), Err(DirectionFailureOrigin::PcmPlayback));
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn source_p1_mode_change_drain_returns_the_exact_stop_receipt() {
        let exact_stop = WorkerStop::Close {
            reason: CloseRequestReason::DaemonShutdown,
            deadline: Instant::now() + RUNTIME_CLEANUP_BUDGET,
        };
        let (stop, mut stop_receiver) = watch::channel(None);
        let distant_deadline_ns = monotonic_ns().saturating_add(60_000_000_000);
        let stopped = tokio::spawn(async move {
            wait_for_playback_deadline(distant_deadline_ns, &mut stop_receiver).await
        });
        tokio::task::yield_now().await;
        stop.send(Some(exact_stop)).unwrap();
        for _ in 0..16 {
            if stopped.is_finished() {
                break;
            }
            tokio::task::yield_now().await;
        }
        let preempted = stopped.is_finished();
        if !preempted {
            stopped.abort();
        }
        let result = stopped.await;
        assert!(preempted, "Stop must preempt an in-flight audible drain");
        assert_eq!(result.unwrap(), Ok(Some(exact_stop)));
    }

    #[test]
    fn direction_change_preserves_the_unaffected_peer() {
        let original = test_launch(
            ready_snapshot(),
            TestAudioTargets {
                microphone_capture: "microphone".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "speaker".to_owned(),
            },
        );
        let mut candidate = original.clone();
        candidate
            .microphone
            .as_mut()
            .unwrap()
            .runtime
            .target_language = Language::Ru;

        assert_eq!(
            original.changed_directions(&candidate),
            [AudioDirection::Microphone]
        );
    }

    #[test]
    fn latency_policy_modes_propagate_to_the_next_provider_open() {
        let store = RuntimeStore::default();
        let observer = RuntimeLatencyObserver::new(store);
        let mut launch = test_launch(
            ready_snapshot(),
            TestAudioTargets {
                microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "benchmark_headphones".to_owned(),
            },
        );

        for index in 0..3 {
            let capture_ns = 1_000_000_000 + index * 10_000_000_000;
            let utterance_id =
                observe_latency_breach(&observer, AudioDirection::Microphone, capture_ns);
            observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
                direction: AudioDirection::Microphone,
                utterance_id,
            });
        }
        assert_eq!(
            observer.requested_mode(AudioDirection::Microphone),
            Some(TranslationMode::Balanced)
        );
        assert!(refresh_launch_mode(
            launch.microphone.as_mut().unwrap(),
            &observer
        ));
        let microphone = launch.microphone.as_ref().unwrap();
        assert_eq!(microphone.runtime.mode, TranslationMode::Balanced);
        assert_eq!(
            DirectionSession::new(microphone.runtime)
                .provider_contract()
                .mode,
            translator_ipc::provider::TranslationMode::Balanced
        );

        for index in 3..6 {
            let capture_ns = 1_000_000_000 + index * 10_000_000_000;
            let utterance_id =
                observe_latency_breach(&observer, AudioDirection::Microphone, capture_ns);
            observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
                direction: AudioDirection::Microphone,
                utterance_id,
            });
        }
        assert_eq!(
            observer.requested_mode(AudioDirection::Microphone),
            Some(TranslationMode::StreamingFirst)
        );
        assert!(refresh_launch_mode(
            launch.microphone.as_mut().unwrap(),
            &observer
        ));
        let next_session = DirectionSession::new(launch.microphone.as_ref().unwrap().runtime);
        assert_eq!(
            next_session.provider_contract().mode,
            translator_ipc::provider::TranslationMode::StreamingFirst
        );
    }

    #[test]
    fn expired_first_audio_contributes_a_latency_breach_at_terminal() {
        let store = RuntimeStore::default();
        let observer = RuntimeLatencyObserver::new(store);

        for index in 0..3 {
            let utterance_id = uuid::Uuid::new_v4();
            let capture_ns = 1_000_000_000 + index * 10_000_000_000;
            observer.observe(DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Speaker,
                utterance_id,
                capture_monotonic_ns: capture_ns,
            });
            observer.observe(DuplexRuntimeEvent::FirstAudioExpired {
                direction: AudioDirection::Speaker,
                utterance_id,
                observed_monotonic_ns: capture_ns + 3_020_000_000,
            });
            observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
                direction: AudioDirection::Speaker,
                utterance_id,
            });
        }

        assert_eq!(
            observer.requested_mode(AudioDirection::Speaker),
            Some(TranslationMode::Balanced)
        );
    }

    #[test]
    fn requested_mode_is_applied_only_at_an_utterance_terminal_boundary() {
        let store = RuntimeStore::default();
        let observer = RuntimeLatencyObserver::new(store.clone());
        let utterance_id = uuid::Uuid::new_v4();
        store.set_latency_policy(crate::LatencyPolicyPatch {
            direction_id: AudioDirection::Microphone,
            current_mode: TranslationMode::Balanced,
        });

        assert_eq!(
            mode_change_after_event(
                &observer,
                TranslationMode::QualityFirst,
                DuplexRuntimeEvent::TranslationFinal {
                    direction: AudioDirection::Microphone,
                    utterance_id,
                },
            ),
            None
        );
        assert_eq!(
            mode_change_after_event(
                &observer,
                TranslationMode::QualityFirst,
                DuplexRuntimeEvent::UtteranceTerminal {
                    direction: AudioDirection::Microphone,
                    utterance_id,
                },
            ),
            Some(TranslationMode::Balanced)
        );
    }

    #[test]
    fn source_p1_latency_correlations_preserve_original_capture_and_bound_each_direction() {
        use translator_ipc::MAX_ACTIVE_UTTERANCES;

        let observer = RuntimeLatencyObserver::new(RuntimeStore::default());
        let microphone_ids = (0..MAX_ACTIVE_UTTERANCES)
            .map(|_| uuid::Uuid::new_v4())
            .collect::<Vec<_>>();
        let speaker_ids = (0..MAX_ACTIVE_UTTERANCES)
            .map(|_| uuid::Uuid::new_v4())
            .collect::<Vec<_>>();
        for (direction, ids) in [
            (AudioDirection::Microphone, &microphone_ids),
            (AudioDirection::Speaker, &speaker_ids),
        ] {
            for (index, utterance_id) in ids.iter().enumerate() {
                observer.observe(DuplexRuntimeEvent::SpeechStarted {
                    direction,
                    utterance_id: *utterance_id,
                    capture_monotonic_ns: 1_000_000_000 + index as u64,
                });
            }
        }

        let duplicate = microphone_ids[0];
        let original_capture = observer
            .utterances
            .lock()
            .unwrap()
            .get(&(AudioDirection::Microphone, duplicate))
            .unwrap()
            .capture_monotonic_ns;
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id: duplicate,
            capture_monotonic_ns: original_capture + 9_000_000_000,
        });
        {
            let utterances = observer.utterances.lock().unwrap();
            assert_eq!(utterances.len(), 2 * MAX_ACTIVE_UTTERANCES);
            assert_eq!(
                utterances
                    .get(&(AudioDirection::Microphone, duplicate))
                    .unwrap()
                    .capture_monotonic_ns,
                original_capture,
                "a duplicate start cannot rejuvenate the latency origin"
            );
            assert!(speaker_ids.iter().all(|utterance_id| {
                utterances.contains_key(&(AudioDirection::Speaker, *utterance_id))
            }));
        }

        let replacement = uuid::Uuid::new_v4();
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id: replacement,
            capture_monotonic_ns: 30_000_000_000,
        });
        let utterances = observer.utterances.lock().unwrap();
        assert_eq!(
            utterances
                .keys()
                .filter(|(direction, _)| *direction == AudioDirection::Microphone)
                .count(),
            1,
            "overflow makes only that direction's unscorable correlations unavailable"
        );
        assert!(utterances.contains_key(&(AudioDirection::Microphone, replacement)));
        assert!(speaker_ids.iter().all(|utterance_id| {
            utterances.contains_key(&(AudioDirection::Speaker, *utterance_id))
        }));
        assert!(utterances.len() <= 2 * MAX_ACTIVE_UTTERANCES);
    }

    #[test]
    fn source_p1_latency_reset_evicts_only_the_joined_direction() {
        let observer = RuntimeLatencyObserver::new(RuntimeStore::default());
        let microphone = uuid::Uuid::new_v4();
        let speaker = uuid::Uuid::new_v4();
        for (direction, utterance_id) in [
            (AudioDirection::Microphone, microphone),
            (AudioDirection::Speaker, speaker),
        ] {
            observer.observe(DuplexRuntimeEvent::SpeechStarted {
                direction,
                utterance_id,
                capture_monotonic_ns: 1_000_000_000,
            });
        }

        observer.reset_direction(AudioDirection::Microphone);

        let utterances = observer.utterances.lock().unwrap();
        assert!(!utterances.contains_key(&(AudioDirection::Microphone, microphone)));
        assert!(utterances.contains_key(&(AudioDirection::Speaker, speaker)));
    }

    #[test]
    fn source_p1_repeated_direction_resets_keep_latency_storage_bounded() {
        use translator_ipc::MAX_ACTIVE_UTTERANCES;

        let observer = RuntimeLatencyObserver::new(RuntimeStore::default());
        for cycle in 0_u64..100 {
            let direction = if cycle % 2 == 0 {
                AudioDirection::Microphone
            } else {
                AudioDirection::Speaker
            };
            for offset in 0..MAX_ACTIVE_UTTERANCES {
                observer.observe(DuplexRuntimeEvent::SpeechStarted {
                    direction,
                    utterance_id: uuid::Uuid::new_v4(),
                    capture_monotonic_ns: cycle * 1_000_000 + offset as u64,
                });
            }
            observer.reset_direction(direction);
            assert!(observer.utterances.lock().unwrap().len() <= MAX_ACTIVE_UTTERANCES);
        }
        assert!(observer.utterances.lock().unwrap().is_empty());
    }

    pub(crate) mod native_owner_red {
        use super::*;
        use crate::{
            ChildState, DirectionRuntimeFailure, DirectionRuntimeStatus, MAX_START_ATTEMPTS,
            SidecarLaunch, SidecarRuntime, SupervisorError,
        };
        use std::{
            collections::HashSet,
            sync::atomic::{AtomicBool, Ordering},
        };
        use tokio::sync::Notify;

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        enum Resource {
            Provider,
            Capture,
            Playback,
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum PrepareStage {
            Provider,
            Capture,
            Playback,
            AfterRegister,
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum DeadlineStage {
            Prepare,
            StopPcm,
            CloseProvider,
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Trigger {
            Fault(DirectionFailureOrigin),
            Panic,
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        enum ScriptEvent {
            Acquire(u64, AudioDirection, Resource),
            Activate(u64, AudioDirection),
            Stop(u64, AudioDirection, Resource),
            CloseProvider(u64, AudioDirection),
            DiscardProvider(u64, AudioDirection),
            GenerationStart,
            GenerationKill,
            GenerationShutdown,
            WaitReady,
            RegisterPoll,
            ObserverReset(AudioDirection),
            RecoveryAttempt(u32),
        }

        #[derive(Default)]
        struct ScriptState {
            next_id: u64,
            events: Vec<ScriptEvent>,
            live: HashSet<(u64, Resource)>,
            prepare_failures: VecDeque<(AudioDirection, PrepareStage)>,
            cleanup_failures: VecDeque<(AudioDirection, Resource)>,
            cleanup_stalls: VecDeque<(AudioDirection, Resource)>,
            cleanup_waiters: usize,
            cleanup_delays: VecDeque<(AudioDirection, Resource, Duration)>,
            entry_failures: VecDeque<AudioDirection>,
            triggers: HashMap<AudioDirection, VecDeque<Trigger>>,
            probe_ready: bool,
            probe_results: VecDeque<bool>,
            probe_delays: VecDeque<Duration>,
            probe_calls: usize,
            sidecar_start_failures: usize,
            sidecar_kill_failures: usize,
            sidecar_start_gate: Option<Arc<SidecarStartGate>>,
            worker_stop_gates: HashMap<AudioDirection, Arc<SidecarStartGate>>,
            wait_ready_delays: VecDeque<Duration>,
            prepare_delays: VecDeque<(AudioDirection, Duration)>,
            register_poll_delay: Option<Duration>,
            prepare_times: Vec<(AudioDirection, tokio::time::Instant)>,
            deadlines: Vec<(AudioDirection, DeadlineStage, tokio::time::Instant)>,
        }

        #[derive(Clone)]
        struct ScriptedEffects {
            state: Arc<Mutex<ScriptState>>,
            changed: Arc<Notify>,
            cleanup_release: Arc<Notify>,
        }

        impl Default for ScriptedEffects {
            fn default() -> Self {
                Self {
                    state: Arc::new(Mutex::new(ScriptState {
                        probe_ready: true,
                        ..ScriptState::default()
                    })),
                    changed: Arc::new(Notify::new()),
                    cleanup_release: Arc::new(Notify::new()),
                }
            }
        }

        struct ScriptAcquisition {
            id: u64,
            launch: DirectionLaunch,
            session_id: uuid::Uuid,
            provider: bool,
            capture: bool,
            playback: bool,
        }

        struct ScriptPrepared(ScriptAcquisition);

        struct CleanupWaiter(Arc<Mutex<ScriptState>>);

        impl Drop for CleanupWaiter {
            fn drop(&mut self) {
                self.0.lock().unwrap().cleanup_waiters -= 1;
            }
        }

        #[derive(Default)]
        struct SidecarStartGate {
            entered: AtomicBool,
            entered_notify: Notify,
            release: Notify,
        }

        impl SidecarStartGate {
            async fn wait_entered(&self) {
                while !self.entered.load(Ordering::SeqCst) {
                    self.entered_notify.notified().await;
                }
            }

            fn release(&self) {
                self.release.notify_one();
            }
        }

        impl ScriptedEffects {
            fn fail_prepare(&self, direction: AudioDirection, stage: PrepareStage) {
                self.state
                    .lock()
                    .unwrap()
                    .prepare_failures
                    .push_back((direction, stage));
            }

            fn fail_cleanup(&self, direction: AudioDirection, resource: Resource) {
                self.state
                    .lock()
                    .unwrap()
                    .cleanup_failures
                    .push_back((direction, resource));
            }

            fn stall_cleanup(&self, direction: AudioDirection, resource: Resource) {
                self.state
                    .lock()
                    .unwrap()
                    .cleanup_stalls
                    .push_back((direction, resource));
            }

            fn delay_cleanup(
                &self,
                direction: AudioDirection,
                resource: Resource,
                delay: Duration,
            ) {
                self.state
                    .lock()
                    .unwrap()
                    .cleanup_delays
                    .push_back((direction, resource, delay));
            }

            fn release_cleanup(&self) {
                self.cleanup_release.notify_one();
            }

            fn active_cleanup_stalls(&self) -> usize {
                self.state.lock().unwrap().cleanup_waiters
            }

            async fn wait_if_cleanup_stalled(&self, direction: AudioDirection, resource: Resource) {
                let delay =
                    {
                        let mut state = self.state.lock().unwrap();
                        let index = state.cleanup_delays.iter().position(
                            |(observed, observed_resource, _)| {
                                *observed == direction && *observed_resource == resource
                            },
                        );
                        index
                            .and_then(|index| state.cleanup_delays.remove(index))
                            .map(|(_, _, delay)| delay)
                    };
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
                let stalled = {
                    let mut state = self.state.lock().unwrap();
                    let index = state
                        .cleanup_stalls
                        .iter()
                        .position(|candidate| *candidate == (direction, resource));
                    if let Some(index) = index {
                        state.cleanup_stalls.remove(index);
                        true
                    } else {
                        false
                    }
                };
                if stalled {
                    self.state.lock().unwrap().cleanup_waiters += 1;
                    let _waiter = CleanupWaiter(self.state.clone());
                    self.cleanup_release.notified().await;
                }
            }

            fn trigger(&self, direction: AudioDirection, trigger: Trigger) {
                self.state
                    .lock()
                    .unwrap()
                    .triggers
                    .entry(direction)
                    .or_default()
                    .push_back(trigger);
                self.changed.notify_waiters();
            }

            fn fail_entry(&self, direction: AudioDirection) {
                self.state
                    .lock()
                    .unwrap()
                    .entry_failures
                    .push_back(direction);
            }

            fn clear_events(&self) {
                let mut state = self.state.lock().unwrap();
                state.events.clear();
                state.prepare_times.clear();
                state.deadlines.clear();
            }

            fn events(&self) -> Vec<ScriptEvent> {
                self.state.lock().unwrap().events.clone()
            }

            fn live_count(&self) -> usize {
                self.state.lock().unwrap().live.len()
            }

            fn fail_sidecar_starts(&self, count: usize) {
                self.state.lock().unwrap().sidecar_start_failures = count;
            }

            fn fail_sidecar_kills(&self, count: usize) {
                self.state.lock().unwrap().sidecar_kill_failures = count;
            }

            fn stall_next_sidecar_start(&self) -> Arc<SidecarStartGate> {
                let gate = Arc::new(SidecarStartGate::default());
                self.state.lock().unwrap().sidecar_start_gate = Some(gate.clone());
                gate
            }

            fn delay_next_wait_ready(&self, delay: Duration) {
                self.state
                    .lock()
                    .unwrap()
                    .wait_ready_delays
                    .push_back(delay);
            }

            fn delay_next_prepare(&self, direction: AudioDirection, delay: Duration) {
                self.state
                    .lock()
                    .unwrap()
                    .prepare_delays
                    .push_back((direction, delay));
            }

            fn delay_next_register_poll(&self, delay: Duration) {
                self.state.lock().unwrap().register_poll_delay = Some(delay);
            }

            fn set_probe_ready(&self, ready: bool) {
                self.state.lock().unwrap().probe_ready = ready;
            }

            fn fail_next_probe(&self) {
                self.state.lock().unwrap().probe_results.push_back(false);
            }

            fn delay_next_probe(&self, delay: Duration) {
                self.state.lock().unwrap().probe_delays.push_back(delay);
            }

            fn probe_calls(&self) -> usize {
                self.state.lock().unwrap().probe_calls
            }

            fn event_count(&self, expected: &ScriptEvent) -> usize {
                self.events()
                    .iter()
                    .filter(|event| *event == expected)
                    .count()
            }

            fn provider_acquisitions(&self, direction: AudioDirection) -> usize {
                self.events()
                    .iter()
                    .filter(|event| {
                        matches!(
                            event,
                            ScriptEvent::Acquire(_, observed, Resource::Provider)
                                if *observed == direction
                        )
                    })
                    .count()
            }

            fn prepare_times(&self, direction: AudioDirection) -> Vec<tokio::time::Instant> {
                self.state
                    .lock()
                    .unwrap()
                    .prepare_times
                    .iter()
                    .filter_map(|(observed, instant)| (*observed == direction).then_some(*instant))
                    .collect()
            }

            fn deadlines(&self) -> Vec<(AudioDirection, DeadlineStage, tokio::time::Instant)> {
                self.state.lock().unwrap().deadlines.clone()
            }

            fn last_acquisition_id(&self, direction: AudioDirection) -> u64 {
                self.events()
                    .iter()
                    .rev()
                    .find_map(|event| match event {
                        ScriptEvent::Acquire(id, observed, _) if *observed == direction => {
                            Some(*id)
                        }
                        _ => None,
                    })
                    .expect("the scripted direction must have an acquired owner")
            }

            fn is_live(&self, id: u64, resource: Resource) -> bool {
                self.state.lock().unwrap().live.contains(&(id, resource))
            }

            fn fail_stage(
                state: &mut ScriptState,
                direction: AudioDirection,
                stage: PrepareStage,
            ) -> bool {
                if state.prepare_failures.front() == Some(&(direction, stage)) {
                    state.prepare_failures.pop_front();
                    true
                } else {
                    false
                }
            }

            fn stop_resource(
                &self,
                owner: &mut ScriptAcquisition,
                resource: Resource,
            ) -> Result<(), DuplexRuntimeError> {
                let present = match resource {
                    Resource::Provider => &mut owner.provider,
                    Resource::Capture => &mut owner.capture,
                    Resource::Playback => &mut owner.playback,
                };
                if !*present {
                    return Ok(());
                }
                let direction = owner.launch.runtime.direction;
                let mut state = self.state.lock().unwrap();
                state
                    .events
                    .push(ScriptEvent::Stop(owner.id, direction, resource));
                if state.cleanup_failures.front() == Some(&(direction, resource)) {
                    state.cleanup_failures.pop_front();
                    return Err(DuplexRuntimeError::StopFailed);
                }
                *present = false;
                state.live.remove(&(owner.id, resource));
                Ok(())
            }
        }

        #[allow(async_fn_in_trait)]
        impl DirectionEffects for ScriptedEffects {
            type Acquisition = ScriptAcquisition;
            type Prepared = ScriptPrepared;

            fn begin(&self, launch: DirectionLaunch) -> Self::Acquisition {
                let mut state = self.state.lock().unwrap();
                state.next_id += 1;
                ScriptAcquisition {
                    id: state.next_id,
                    launch,
                    session_id: uuid::Uuid::new_v4(),
                    provider: false,
                    capture: false,
                    playback: false,
                }
            }

            fn session_id(owner: &Self::Acquisition) -> uuid::Uuid {
                owner.session_id
            }

            async fn prepare(
                &self,
                owner: &mut Self::Acquisition,
                _generation: &SidecarLaunch,
                deadline: tokio::time::Instant,
            ) -> Result<(), FaultScope> {
                let direction = owner.launch.runtime.direction;
                {
                    let mut state = self.state.lock().unwrap();
                    state
                        .prepare_times
                        .push((direction, tokio::time::Instant::now()));
                    state
                        .deadlines
                        .push((direction, DeadlineStage::Prepare, deadline));
                }
                for (resource, stage, scope) in [
                    (
                        Resource::Provider,
                        PrepareStage::Provider,
                        FaultScope::ProviderConnection,
                    ),
                    (Resource::Capture, PrepareStage::Capture, FaultScope::Local),
                    (
                        Resource::Playback,
                        PrepareStage::Playback,
                        FaultScope::Local,
                    ),
                ] {
                    let mut state = self.state.lock().unwrap();
                    state.live.insert((owner.id, resource));
                    state
                        .events
                        .push(ScriptEvent::Acquire(owner.id, direction, resource));
                    match resource {
                        Resource::Provider => owner.provider = true,
                        Resource::Capture => owner.capture = true,
                        Resource::Playback => owner.playback = true,
                    }
                    if Self::fail_stage(&mut state, direction, stage) {
                        return Err(scope);
                    }
                }
                let delay = {
                    let mut state = self.state.lock().unwrap();
                    let index = state
                        .prepare_delays
                        .iter()
                        .position(|(observed, _)| *observed == direction);
                    index
                        .and_then(|index| state.prepare_delays.remove(index))
                        .map(|(_, delay)| delay)
                };
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
                Ok(())
            }

            fn finish(
                &self,
                owner: Self::Acquisition,
            ) -> Result<Self::Prepared, Self::Acquisition> {
                let direction = owner.launch.runtime.direction;
                if Self::fail_stage(
                    &mut self.state.lock().unwrap(),
                    direction,
                    PrepareStage::AfterRegister,
                ) {
                    return Err(owner);
                }
                if owner.provider && owner.capture && owner.playback {
                    Ok(ScriptPrepared(owner))
                } else {
                    Err(owner)
                }
            }

            fn recover_owner(&self, prepared: Self::Prepared) -> Self::Acquisition {
                prepared.0
            }

            async fn run(
                &self,
                prepared: &mut Self::Prepared,
                stop: &mut watch::Receiver<Option<WorkerStop>>,
                _observer: Arc<dyn DuplexRuntimeObserver>,
                entered: oneshot::Sender<()>,
            ) -> DirectionOutcome {
                let direction = prepared.0.launch.runtime.direction;
                if self.state.lock().unwrap().entry_failures.front() == Some(&direction) {
                    self.state.lock().unwrap().entry_failures.pop_front();
                    return DirectionOutcome::Fault(classify_direction_failure(
                        DirectionFailureOrigin::InternalTransport,
                    ));
                }
                let _ = entered.send(());
                self.state
                    .lock()
                    .unwrap()
                    .events
                    .push(ScriptEvent::Activate(prepared.0.id, direction));
                loop {
                    let trigger = self
                        .state
                        .lock()
                        .unwrap()
                        .triggers
                        .get_mut(&direction)
                        .and_then(VecDeque::pop_front);
                    match trigger {
                        Some(Trigger::Fault(origin)) => {
                            return DirectionOutcome::Fault(classify_direction_failure(origin));
                        }
                        Some(Trigger::Panic) => panic!("scripted worker panic"),
                        None => {}
                    }
                    tokio::select! {
                        _ = self.changed.notified() => {}
                        changed = stop.changed() => {
                            if changed.is_err() {
                                return DirectionOutcome::Fault(FaultScope::Local);
                            }
                            let reason = *stop.borrow();
                            if let Some(reason) = reason {
                                let gate = self
                                    .state
                                    .lock()
                                    .unwrap()
                                    .worker_stop_gates
                                    .remove(&direction);
                                if let Some(gate) = gate {
                                    gate.entered.store(true, Ordering::SeqCst);
                                    gate.entered_notify.notify_one();
                                    gate.release.notified().await;
                                }
                                return DirectionOutcome::Stopped(reason);
                            }
                        }
                    }
                }
            }

            async fn stop_pcm(
                &self,
                owner: &mut Self::Acquisition,
                deadline: tokio::time::Instant,
            ) -> Result<(), DuplexRuntimeError> {
                let direction = owner.launch.runtime.direction;
                self.state.lock().unwrap().deadlines.push((
                    direction,
                    DeadlineStage::StopPcm,
                    deadline,
                ));
                self.wait_if_cleanup_stalled(direction, Resource::Capture)
                    .await;
                let capture = self.stop_resource(owner, Resource::Capture);
                self.wait_if_cleanup_stalled(direction, Resource::Playback)
                    .await;
                let playback = self.stop_resource(owner, Resource::Playback);
                if capture.is_err() || playback.is_err() {
                    Err(DuplexRuntimeError::StopFailed)
                } else {
                    Ok(())
                }
            }

            async fn close_provider(
                &self,
                owner: &mut Self::Acquisition,
                _reason: CloseRequestReason,
                deadline: tokio::time::Instant,
            ) -> Result<(), DuplexRuntimeError> {
                if !owner.provider {
                    return Ok(());
                }
                self.state.lock().unwrap().deadlines.push((
                    owner.launch.runtime.direction,
                    DeadlineStage::CloseProvider,
                    deadline,
                ));
                self.wait_if_cleanup_stalled(owner.launch.runtime.direction, Resource::Provider)
                    .await;
                self.state
                    .lock()
                    .unwrap()
                    .events
                    .push(ScriptEvent::CloseProvider(
                        owner.id,
                        owner.launch.runtime.direction,
                    ));
                self.stop_resource(owner, Resource::Provider)
            }

            fn discard_provider(&self, owner: &mut Self::Acquisition) {
                if owner.provider {
                    let mut state = self.state.lock().unwrap();
                    state.events.push(ScriptEvent::DiscardProvider(
                        owner.id,
                        owner.launch.runtime.direction,
                    ));
                    state.live.remove(&(owner.id, Resource::Provider));
                    owner.provider = false;
                }
            }

            fn is_clean(owner: &Self::Acquisition) -> bool {
                !owner.provider && !owner.capture && !owner.playback
            }

            async fn wait_ready<R: SidecarRuntime>(
                &self,
                _supervisor: &SidecarSupervisor<R>,
            ) -> Result<(), DuplexRuntimeError> {
                let (ready, delay) = {
                    let mut state = self.state.lock().unwrap();
                    state.events.push(ScriptEvent::WaitReady);
                    (state.probe_ready, state.wait_ready_delays.pop_front())
                };
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
                ready.then_some(()).ok_or(DuplexRuntimeError::StartFailed)
            }

            async fn probe_generation<R: SidecarRuntime>(
                &self,
                _supervisor: &SidecarSupervisor<R>,
            ) -> bool {
                let (result, delay) = {
                    let mut state = self.state.lock().unwrap();
                    state.probe_calls += 1;
                    (
                        state.probe_results.pop_front().unwrap_or(state.probe_ready),
                        state.probe_delays.pop_front(),
                    )
                };
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
                result
            }
        }

        #[derive(Clone)]
        struct ScriptedSidecar(Arc<Mutex<ScriptState>>);

        #[allow(async_fn_in_trait)]
        impl SidecarRuntime for ScriptedSidecar {
            async fn start(&mut self, _launch: &SidecarLaunch) -> Result<(), SupervisorError> {
                let gate = {
                    let mut state = self.0.lock().unwrap();
                    state.events.push(ScriptEvent::GenerationStart);
                    if state.sidecar_start_failures > 0 {
                        state.sidecar_start_failures -= 1;
                        return Err(SupervisorError::StartFailed);
                    }
                    state.sidecar_start_gate.take()
                };
                if let Some(gate) = gate {
                    gate.entered.store(true, Ordering::SeqCst);
                    gate.entered_notify.notify_one();
                    gate.release.notified().await;
                }
                Ok(())
            }

            async fn probe(
                &mut self,
                launch: &SidecarLaunch,
            ) -> Result<uuid::Uuid, SupervisorError> {
                Ok(launch.generation_id)
            }

            async fn kill_and_reap(&mut self) -> Result<ChildState, SupervisorError> {
                let mut state = self.0.lock().unwrap();
                state.events.push(ScriptEvent::GenerationKill);
                if state.sidecar_kill_failures > 0 {
                    state.sidecar_kill_failures -= 1;
                    Err(SupervisorError::KillAndReapFailed)
                } else {
                    Ok(ChildState::Reaped)
                }
            }

            async fn shutdown_and_reap(&mut self) -> Result<ChildState, SupervisorError> {
                self.0
                    .lock()
                    .unwrap()
                    .events
                    .push(ScriptEvent::GenerationShutdown);
                Ok(ChildState::Reaped)
            }

            async fn remove_stale_socket(
                &mut self,
                _child_state: ChildState,
            ) -> Result<(), SupervisorError> {
                Ok(())
            }

            async fn wait_before_retry(&mut self, _attempt: usize) -> Result<(), SupervisorError> {
                Ok(())
            }

            fn poll_child_state(&mut self) -> Result<ChildState, SupervisorError> {
                let delay = {
                    let mut state = self.0.lock().unwrap();
                    state.events.push(ScriptEvent::RegisterPoll);
                    state.register_poll_delay.take()
                };
                if let Some(delay) = delay {
                    std::thread::sleep(delay);
                }
                Ok(ChildState::Running)
            }
        }

        struct ScriptObserver(Arc<Mutex<ScriptState>>);

        impl DuplexRuntimeObserver for ScriptObserver {
            fn observe(&self, event: DuplexRuntimeEvent) {
                if let DuplexRuntimeEvent::GenerationRestart { attempt } = event {
                    self.0
                        .lock()
                        .unwrap()
                        .events
                        .push(ScriptEvent::RecoveryAttempt(attempt.get()));
                }
            }

            fn reset_direction(&self, direction: AudioDirection) {
                self.0
                    .lock()
                    .unwrap()
                    .events
                    .push(ScriptEvent::ObserverReset(direction));
            }
        }

        type RecordedDirectionStatus = (
            u64,
            AudioDirection,
            u64,
            DirectionRuntimeStatus,
            Option<DirectionRuntimeFailure>,
        );

        #[derive(Default)]
        struct LifecycleRecorder {
            cleanup_started: Mutex<Vec<u64>>,
            completions: Mutex<Vec<(u64, Result<(), DuplexRuntimeError>)>>,
            statuses: Mutex<Vec<RecordedDirectionStatus>>,
        }

        impl DuplexCompletionObserver for LifecycleRecorder {
            fn cleanup_started(&self, generation: u64) {
                self.cleanup_started.lock().unwrap().push(generation);
            }

            fn completed(&self, generation: u64, result: Result<(), DuplexRuntimeError>) {
                self.completions.lock().unwrap().push((generation, result));
            }

            fn direction_status_changed(
                &self,
                generation: u64,
                direction: AudioDirection,
                epoch: u64,
                status: DirectionRuntimeStatus,
                failure: Option<DirectionRuntimeFailure>,
            ) {
                self.statuses
                    .lock()
                    .unwrap()
                    .push((generation, direction, epoch, status, failure));
            }
        }

        fn launch_pair() -> DuplexLaunch {
            test_launch(
                ready_snapshot(),
                TestAudioTargets {
                    microphone_capture: "test-microphone".to_owned(),
                    microphone_playback: "test-microphone-output".to_owned(),
                    speaker_capture: "test-speaker".to_owned(),
                    speaker_playback: "test-speaker-output".to_owned(),
                },
            )
        }

        fn replacement(mut launch: DuplexLaunch) -> DuplexLaunch {
            launch
                .microphone
                .as_mut()
                .unwrap()
                .runtime
                .debug_text_enabled = true;
            launch.speaker.as_mut().unwrap().runtime.debug_text_enabled = true;
            launch
        }

        fn provider_replacement(mut launch: DuplexLaunch) -> DuplexLaunch {
            for direction in [&mut launch.microphone, &mut launch.speaker]
                .into_iter()
                .flatten()
            {
                direction.runtime.provider_id = ProviderId::Openai;
                direction.runtime.voice_engine = VoiceEngine::Openai;
            }
            launch
        }

        fn transaction_deadline() -> Instant {
            Instant::now() + DIRECTION_CLEANUP_BUDGET
        }

        fn voice_admitted_launch(snapshot: RuntimeSnapshot) -> DuplexLaunch {
            DuplexLaunch::from(
                crate::acoustic_admission::admit_translation(
                    snapshot,
                    crate::acoustic_admission::tests::ready_facts(),
                )
                .unwrap(),
            )
        }

        #[tokio::test(flavor = "current_thread")]
        async fn voice_override_builtin_gender_uses_real_launch_and_native_rollback() {
            tokio::task::LocalSet::new().run_until(tokio::time::timeout(Duration::from_secs(3), async {
                for fault in ["none", "prepare", "activation"] {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let original = RuntimeSnapshot::default();
                    let mut coordinator = coordinator(effects.clone(), lifecycle.clone());
                    coordinator.desired = voice_admitted_launch(original.clone());
                    coordinator.start_resources(transaction_deadline()).await.unwrap();
                    let peer = coordinator.workers[&AudioDirection::Microphone].cell.clone();
                    let peer_task = coordinator.workers[&AudioDirection::Microphone].task_id;
                    let peer_epoch = coordinator.worker_epoch(AudioDirection::Microphone);
                    let old_target = coordinator.workers[&AudioDirection::Speaker].cell.clone();
                    let old_id = effects.last_acquisition_id(AudioDirection::Speaker);
                    let generation = coordinator.supervisor.launch().unwrap().generation_id;
                    let before_launch = coordinator.desired.clone();
                    let mut candidate = original;
                    candidate.directions[1].voice_profile.gender = translator_core::VoiceGender::Female;
                    let candidate = voice_admitted_launch(candidate);
                    let open = DirectionSession::new(candidate.speaker.as_ref().unwrap().runtime).open_request();
                    let Some(translator_ipc::provider::provider_request::Request::OpenSession(open)) = open.request else { panic!("real session must create Open"); };
                    let voice = open.voice_profile.unwrap();
                    assert_eq!(voice.gender, translator_ipc::provider::VoiceGender::Female as i32);
                    assert_eq!(voice.language, translator_ipc::provider::Language::Ru as i32);
                    assert!(voice.model_path.is_none());
                    assert!(voice.provider_voice_id.is_none());
                    effects.clear_events();
                    let status_start = lifecycle.statuses.lock().unwrap().len();
                    let candidate_epoch = coordinator.next_epoch + 1;
                    match fault {
                        "prepare" => effects.fail_prepare(AudioDirection::Speaker, PrepareStage::Capture),
                        "activation" => effects.fail_entry(AudioDirection::Speaker),
                        _ => {}
                    }
                    let result = coordinator.reconfigure(candidate.clone(), transaction_deadline()).await;
                    let events = effects.events();
                    assert!(Rc::ptr_eq(&peer, &coordinator.workers[&AudioDirection::Microphone].cell));
                    assert_eq!(coordinator.workers[&AudioDirection::Microphone].task_id, peer_task);
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Microphone), peer_epoch);
                    assert_eq!(coordinator.supervisor.launch().unwrap().generation_id, generation);
                    assert_eq!(effects.live_count(), 6);
                    assert!(!events.iter().any(|event| matches!(event,
                        ScriptEvent::Acquire(_, AudioDirection::Microphone, _)
                        | ScriptEvent::Stop(_, AudioDirection::Microphone, _)
                        | ScriptEvent::CloseProvider(_, AudioDirection::Microphone)
                        | ScriptEvent::GenerationKill | ScriptEvent::GenerationStart)));
                    if fault == "none" {
                        assert_eq!(result, Ok(()));
                        assert!(coordinator.desired == candidate);
                        assert!(!Rc::ptr_eq(&old_target, &coordinator.workers[&AudioDirection::Speaker].cell));
                    } else {
                        assert_eq!(result, Err(DuplexRuntimeError::ReconfigureFailed));
                        assert!(coordinator.desired == before_launch);
                        if fault == "prepare" {
                            assert!(Rc::ptr_eq(&old_target, &coordinator.workers[&AudioDirection::Speaker].cell));
                            assert!(!events.iter().any(|event| matches!(event, ScriptEvent::Stop(id, _, _) | ScriptEvent::CloseProvider(id, _) if *id == old_id)));
                        }
                        assert!(lifecycle.statuses.lock().unwrap()[status_start..].iter().all(|status| status.2 != candidate_epoch || status.3 != DirectionRuntimeStatus::Running));
                    }
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }
            })).await.expect("built-in voice replacement and rollback must stay bounded");
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn voice_override_builtin_noop_does_not_restart_real_failed_direction() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let original = RuntimeSnapshot::default();
                    let mut coordinator = coordinator(effects.clone(), lifecycle.clone());
                    coordinator.desired = voice_admitted_launch(original.clone());
                    coordinator
                        .start_resources(transaction_deadline())
                        .await
                        .unwrap();
                    let peer = coordinator.workers[&AudioDirection::Microphone]
                        .cell
                        .clone();
                    let peer_task = coordinator.workers[&AudioDirection::Microphone].task_id;
                    for _ in 0..3 {
                        effects.fail_prepare(AudioDirection::Speaker, PrepareStage::Capture);
                    }
                    effects.trigger(
                        AudioDirection::Speaker,
                        Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                    );
                    coordinator
                        .handle_next_worker_completion(transaction_deadline())
                        .await
                        .unwrap();
                    assert!(coordinator.paused.contains(&AudioDirection::Speaker));
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Speaker), None);
                    assert!(
                        lifecycle
                            .statuses
                            .lock()
                            .unwrap()
                            .iter()
                            .any(|status| status.1 == AudioDirection::Speaker
                                && status.3 == DirectionRuntimeStatus::Failed)
                    );
                    effects.clear_events();
                    let status_start = lifecycle.statuses.lock().unwrap().len();
                    coordinator
                        .reconfigure(voice_admitted_launch(original), transaction_deadline())
                        .await
                        .unwrap();
                    assert!(coordinator.paused.contains(&AudioDirection::Speaker));
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Speaker), None);
                    assert!(Rc::ptr_eq(
                        &peer,
                        &coordinator.workers[&AudioDirection::Microphone].cell
                    ));
                    assert_eq!(
                        coordinator.workers[&AudioDirection::Microphone].task_id,
                        peer_task
                    );
                    assert!(effects.events().is_empty());
                    assert_eq!(lifecycle.statuses.lock().unwrap().len(), status_start);
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("failed-direction no-op must stay bounded");
        }

        fn cleanup_deadline() -> Instant {
            Instant::now() + RUNTIME_CLEANUP_BUDGET
        }

        fn coordinator(
            effects: ScriptedEffects,
            lifecycle: Arc<LifecycleRecorder>,
        ) -> DuplexCoordinator<ScriptedSidecar, ScriptedEffects> {
            coordinator_with_completion(effects, 41, lifecycle)
        }

        fn coordinator_with_completion(
            effects: ScriptedEffects,
            generation: u64,
            lifecycle: Arc<dyn DuplexCompletionObserver>,
        ) -> DuplexCoordinator<ScriptedSidecar, ScriptedEffects> {
            DuplexCoordinator::with_dependencies(
                ProcessDuplexConfig {
                    python: PathBuf::from("unused-python"),
                    sidecar_root: PathBuf::from("unused-sidecar"),
                    socket_path: PathBuf::from("unused.sock"),
                    expected_uid: 0,
                },
                launch_pair(),
                SidecarSupervisor::new(ScriptedSidecar(effects.state.clone())),
                effects.clone(),
                Arc::new(ScriptObserver(effects.state.clone())),
                Some((generation, lifecycle)),
            )
        }

        async fn started(
            effects: ScriptedEffects,
            lifecycle: Arc<LifecycleRecorder>,
        ) -> DuplexCoordinator<ScriptedSidecar, ScriptedEffects> {
            let mut coordinator = coordinator(effects, lifecycle);
            coordinator
                .start_resources(tokio::time::Instant::now() + DIRECTION_CLEANUP_BUDGET)
                .await
                .unwrap();
            coordinator
        }

        async fn finish_quiescence_fixture(
            effects: &ScriptedEffects,
            attempt: tokio::task::JoinHandle<(
                DuplexCoordinator<ScriptedSidecar, ScriptedEffects>,
                Result<(), DuplexRuntimeError>,
            )>,
        ) -> (
            DuplexCoordinator<ScriptedSidecar, ScriptedEffects>,
            Result<(), DuplexRuntimeError>,
        ) {
            for _ in 0..128 {
                if attempt.is_finished() {
                    return attempt.await.unwrap();
                }
                effects.release_cleanup();
                tokio::task::yield_now().await;
            }
            tokio::time::advance(RUNTIME_CLEANUP_BUDGET + Duration::from_secs(1)).await;
            settle_ready_tasks().await;
            if !attempt.is_finished() {
                attempt.abort();
                let _ = attempt.await;
                panic!("quiescence fixture could not recover its retained coordinator");
            }
            attempt.await.unwrap()
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_all_pcm_and_joins_precede_first_provider_ack() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let owners = [AudioDirection::Microphone, AudioDirection::Speaker]
                        .map(|direction| (direction, effects.last_acquisition_id(direction)));
                    effects.clear_events();
                    for (direction, _) in owners {
                        effects.stall_cleanup(direction, Resource::Provider);
                    }
                    let deadline = cleanup_deadline();
                    let attempt = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .stop_all(CloseRequestReason::UserStop, deadline)
                            .await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.active_cleanup_stalls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    let entered = effects.active_cleanup_stalls();
                    let before = effects.events();
                    let live_before = owners.map(|(_, id)| {
                        (
                            effects.is_live(id, Resource::Capture),
                            effects.is_live(id, Resource::Playback),
                        )
                    });
                    let pending = !attempt.is_finished();
                    let (mut coordinator, result) =
                        finish_quiescence_fixture(&effects, attempt).await;
                    let stopped = effects.events();
                    let deadlines = effects.deadlines();
                    let empty_after_stop = effects.live_count() == 0;
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();

                    assert_eq!(entered, 1, "an actual provider ACK must be held");
                    assert!(pending, "held ACK cannot certify Stop");
                    assert_eq!(
                        live_before,
                        [(false, false); 2],
                        "both PCM owners must stop before the first provider ACK"
                    );
                    for (direction, id) in owners {
                        assert_eq!(
                            before
                                .iter()
                                .filter(|event| **event == ScriptEvent::ObserverReset(direction))
                                .count(),
                            1,
                            "both actual worker completions must be consumed before provider close"
                        );
                        for resource in [Resource::Capture, Resource::Playback, Resource::Provider]
                        {
                            assert_eq!(
                                stopped
                                    .iter()
                                    .filter(|event| **event
                                        == ScriptEvent::Stop(id, direction, resource))
                                    .count(),
                                1
                            );
                        }
                    }
                    assert_eq!(result, Ok(()));
                    assert!(empty_after_stop);
                    assert!(
                        deadlines
                            .iter()
                            .all(|(_, _, observed)| *observed == deadline)
                    );
                    assert!(lifecycle.completions.lock().unwrap().is_empty());
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_failed_pcm_preserves_peer_progress_and_retry_effects() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let mut coordinator =
                        started(effects.clone(), Arc::new(LifecycleRecorder::default())).await;
                    let owners = [AudioDirection::Microphone, AudioDirection::Speaker]
                        .map(|direction| (direction, effects.last_acquisition_id(direction)));
                    effects.clear_events();
                    effects.fail_cleanup(AudioDirection::Microphone, Resource::Capture);
                    let first_deadline = cleanup_deadline();
                    let first = coordinator
                        .stop_all(CloseRequestReason::UserStop, first_deadline)
                        .await;
                    let before = effects.events();
                    let remaining_capture = effects.is_live(owners[0].1, Resource::Capture);
                    let peer_pcm_live = [Resource::Capture, Resource::Playback]
                        .map(|r| effects.is_live(owners[1].1, r));
                    let first_deadlines = effects.deadlines();
                    let retained = coordinator.has_cleanup_owner(AudioDirection::Microphone);
                    let retry = coordinator
                        .stop_all(CloseRequestReason::UserStop, cleanup_deadline())
                        .await;
                    let after = effects.events();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();

                    assert_eq!(first, Err(DuplexRuntimeError::StopFailed));
                    assert!(remaining_capture && retained);
                    assert_eq!(peer_pcm_live, [false; 2]);
                    assert!(before.iter().any(|e| *e
                        == ScriptEvent::Stop(
                            owners[0].1,
                            AudioDirection::Microphone,
                            Resource::Playback
                        )));
                    assert!(
                        !before
                            .iter()
                            .any(|e| matches!(e, ScriptEvent::CloseProvider(..))),
                        "failed local barrier must not enter provider close"
                    );
                    assert!(first_deadlines.iter().all(|(_, _, d)| *d == first_deadline));
                    assert_eq!(retry, Ok(()));
                    for (direction, id) in owners {
                        for resource in [Resource::Capture, Resource::Playback, Resource::Provider]
                        {
                            let expected = if direction == AudioDirection::Microphone
                                && resource == Resource::Capture
                            {
                                2
                            } else {
                                1
                            };
                            assert_eq!(
                                after
                                    .iter()
                                    .filter(|e| **e == ScriptEvent::Stop(id, direction, resource))
                                    .count(),
                                expected
                            );
                        }
                    }
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        async fn held_local_owner_preserves_peer_progress(worker_lock: bool) {
            let effects = ScriptedEffects::default();
            let mut coordinator =
                started(effects.clone(), Arc::new(LifecycleRecorder::default())).await;
            // Choose the first actual worker, so the old sequential implementation
            // cannot pass merely because HashMap iteration visits the healthy peer first.
            let blocked = *coordinator.workers.keys().next().unwrap();
            let original_cell = coordinator.workers[&blocked].cell.clone();
            let original_task = coordinator.workers[&blocked].task_id.unwrap();
            let peer = if blocked == AudioDirection::Microphone {
                AudioDirection::Speaker
            } else {
                AudioDirection::Microphone
            };
            let blocked_id = effects.last_acquisition_id(blocked);
            let peer_id = effects.last_acquisition_id(peer);
            let gate = Arc::new(SidecarStartGate::default());
            if worker_lock {
                effects
                    .state
                    .lock()
                    .unwrap()
                    .worker_stop_gates
                    .insert(blocked, gate.clone());
            } else {
                effects.stall_cleanup(blocked, Resource::Capture);
            }
            effects.clear_events();
            let deadline = cleanup_deadline();
            let attempt = tokio::task::spawn_local(async move {
                let result = coordinator
                    .stop_all(CloseRequestReason::UserStop, deadline)
                    .await;
                (coordinator, result)
            });
            for _ in 0..64 {
                if gate.entered.load(Ordering::SeqCst) || effects.active_cleanup_stalls() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            settle_ready_tasks().await;
            let entered =
                gate.entered.load(Ordering::SeqCst) || effects.active_cleanup_stalls() == 1;
            let peer_live =
                [Resource::Capture, Resource::Playback].map(|r| effects.is_live(peer_id, r));
            let blocked_live = effects.is_live(blocked_id, Resource::Capture);
            let before = effects.events();
            tokio::time::advance(RUNTIME_CLEANUP_BUDGET).await;
            settle_ready_tasks().await;
            let ended_at_deadline = attempt.is_finished();
            let (mut coordinator, first) = if ended_at_deadline {
                attempt.await.unwrap()
            } else {
                gate.release();
                finish_quiescence_fixture(&effects, attempt).await
            };
            let retained = if worker_lock {
                ended_at_deadline
                    && coordinator.workers.get(&blocked).is_some_and(|worker| {
                        Rc::ptr_eq(&worker.cell, &original_cell)
                            && worker.task_id == Some(original_task)
                    })
                    && effects.is_live(blocked_id, Resource::Capture)
            } else {
                coordinator.has_cleanup_owner(blocked)
            };
            gate.release();
            let first_deadlines = effects.deadlines();
            let retry = coordinator
                .stop_all(CloseRequestReason::UserStop, cleanup_deadline())
                .await;
            coordinator.shutdown(cleanup_deadline()).await.unwrap();

            assert!(
                entered,
                "the real local effect or worker lock must be entered"
            );
            assert_eq!(
                peer_live, [false; 2],
                "an independently runnable PCM owner must stop while its peer is held"
            );
            assert!(blocked_live && retained);
            assert!(
                !before
                    .iter()
                    .any(|e| matches!(e, ScriptEvent::CloseProvider(..)))
            );
            assert!(ended_at_deadline);
            assert_eq!(first, Err(DuplexRuntimeError::StopFailed));
            assert!(first_deadlines.iter().all(|(_, _, d)| *d == deadline));
            assert_eq!(retry, Ok(()));
            assert_eq!(effects.live_count(), 0);
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_held_pcm_does_not_block_healthy_peer() {
            tokio::task::LocalSet::new()
                .run_until(held_local_owner_preserves_peer_progress(false))
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_held_worker_cell_does_not_block_healthy_peer() {
            tokio::task::LocalSet::new()
                .run_until(held_local_owner_preserves_peer_progress(true))
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_expired_entry_preserves_resources_for_explicit_retry() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let mut coordinator =
                        started(effects.clone(), Arc::new(LifecycleRecorder::default())).await;
                    let ids = [AudioDirection::Microphone, AudioDirection::Speaker]
                        .map(|d| effects.last_acquisition_id(d));
                    let cells = coordinator
                        .workers
                        .values()
                        .map(|w| w.cell.clone())
                        .collect::<Vec<_>>();
                    effects.clear_events();
                    let expired = coordinator
                        .stop_all(CloseRequestReason::UserStop, Instant::now())
                        .await;
                    let before = effects.events();
                    let live_before = ids.map(|id| {
                        [Resource::Capture, Resource::Playback, Resource::Provider]
                            .map(|r| effects.is_live(id, r))
                    });
                    let retained = cells.iter().all(|cell| {
                        coordinator
                            .workers
                            .values()
                            .any(|worker| Rc::ptr_eq(cell, &worker.cell))
                    });
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();

                    assert_eq!(expired, Err(DuplexRuntimeError::StopFailed));
                    assert_eq!(live_before, [[true; 3]; 2]);
                    assert!(retained);
                    assert!(!before.iter().any(|e| matches!(
                        e,
                        ScriptEvent::Stop(..)
                            | ScriptEvent::CloseProvider(..)
                            | ScriptEvent::DiscardProvider(..)
                            | ScriptEvent::Acquire(..)
                            | ScriptEvent::GenerationKill
                            | ScriptEvent::GenerationShutdown
                            | ScriptEvent::GenerationStart
                    )));
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_shutdown_covers_active_and_retained_candidate_union() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let mut coordinator =
                        started(effects.clone(), Arc::new(LifecycleRecorder::default())).await;
                    let active = [AudioDirection::Microphone, AudioDirection::Speaker]
                        .map(|d| (d, effects.last_acquisition_id(d)));
                    effects.fail_prepare(AudioDirection::Speaker, PrepareStage::AfterRegister);
                    effects.stall_cleanup(AudioDirection::Speaker, Resource::Capture);
                    let candidate = replacement(coordinator.desired.clone());
                    let deadline = transaction_deadline();
                    let preparation = tokio::task::spawn_local(async move {
                        let result = coordinator.reconfigure(candidate, deadline).await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.active_cleanup_stalls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    let setup_entered = effects.active_cleanup_stalls() == 1;
                    tokio::time::advance(DIRECTION_CLEANUP_BUDGET).await;
                    settle_ready_tasks().await;
                    let (mut coordinator, preparation_result) =
                        finish_quiescence_fixture(&effects, preparation).await;
                    let retained_id = effects.last_acquisition_id(AudioDirection::Speaker);
                    let retained_before = coordinator.has_cleanup_owner(AudioDirection::Speaker);
                    let retained_pcm_before = [Resource::Capture, Resource::Playback]
                        .map(|r| effects.is_live(retained_id, r));
                    // The same retained cell can appear in a collected affected set only once.
                    let retained_cell = coordinator.cleanup_owners.last().unwrap().clone();
                    coordinator.cleanup_owners.push(retained_cell);
                    effects.clear_events();
                    for (direction, _) in active {
                        effects.stall_cleanup(direction, Resource::Provider);
                    }
                    let deadline = cleanup_deadline();
                    let attempt = tokio::task::spawn_local(async move {
                        let result = coordinator.shutdown(deadline).await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.active_cleanup_stalls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    let entered = effects.active_cleanup_stalls();
                    let owners = [active[0], active[1], (AudioDirection::Speaker, retained_id)];
                    let live_before = owners.map(|(_, id)| {
                        [Resource::Capture, Resource::Playback].map(|r| effects.is_live(id, r))
                    });
                    let pending = !attempt.is_finished();
                    let (mut coordinator, result) =
                        finish_quiescence_fixture(&effects, attempt).await;
                    let after = effects.events();
                    if result.is_err() {
                        coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    }

                    assert!(setup_entered && retained_before);
                    assert_eq!(preparation_result, Err(DuplexRuntimeError::RestoreFailed));
                    assert_eq!(retained_pcm_before, [true; 2]);
                    assert_eq!(entered, 1);
                    assert!(pending);
                    assert_eq!(
                        live_before, [[false; 2]; 3],
                        "retained PCM cannot be hidden behind an active provider ACK"
                    );
                    assert_eq!(result, Ok(()));
                    for (direction, id) in owners {
                        for resource in [Resource::Capture, Resource::Playback, Resource::Provider]
                        {
                            assert_eq!(
                                after
                                    .iter()
                                    .filter(|e| **e == ScriptEvent::Stop(id, direction, resource))
                                    .count(),
                                1,
                                "aliased retained cells must not repeat resource effects"
                            );
                        }
                    }
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_partial_start_compensation_stops_all_candidate_pcm_first() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    effects.fail_prepare(AudioDirection::Speaker, PrepareStage::AfterRegister);
                    for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
                        effects.stall_cleanup(direction, Resource::Provider);
                    }
                    let mut coordinator =
                        coordinator(effects.clone(), Arc::new(LifecycleRecorder::default()));
                    let deadline = transaction_deadline();
                    let attempt = tokio::task::spawn_local(async move {
                        let result = coordinator.start_resources(deadline).await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.active_cleanup_stalls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    let entered = effects.active_cleanup_stalls();
                    let ids = [AudioDirection::Microphone, AudioDirection::Speaker]
                        .map(|d| effects.last_acquisition_id(d));
                    let live_before = ids.map(|id| {
                        [Resource::Capture, Resource::Playback].map(|r| effects.is_live(id, r))
                    });
                    let before = effects.events();
                    let (mut coordinator, result) =
                        finish_quiescence_fixture(&effects, attempt).await;
                    let retained_after = coordinator.has_pending_cleanup();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();

                    assert_eq!(entered, 1);
                    assert_eq!(live_before, [[false; 2]; 2]);
                    assert!(
                        !before
                            .iter()
                            .any(|e| matches!(e, ScriptEvent::Activate(..)))
                    );
                    assert_eq!(result, Err(DuplexRuntimeError::StartFailed));
                    assert!(!retained_after);
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_reconfigure_preserves_replacements_until_old_pcm_stops() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let mut coordinator =
                        started(effects.clone(), Arc::new(LifecycleRecorder::default())).await;
                    let old = [AudioDirection::Microphone, AudioDirection::Speaker]
                        .map(|d| (d, effects.last_acquisition_id(d)));
                    let candidate = replacement(coordinator.desired.clone());
                    effects.clear_events();
                    for (direction, _) in old {
                        effects.stall_cleanup(direction, Resource::Provider);
                    }
                    let deadline = transaction_deadline();
                    let attempt = tokio::task::spawn_local(async move {
                        let result = coordinator.reconfigure(candidate, deadline).await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.active_cleanup_stalls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    let entered = effects.active_cleanup_stalls();
                    let replacements = old.map(|(d, _)| (d, effects.last_acquisition_id(d)));
                    let old_live = old.map(|(_, id)| {
                        [Resource::Capture, Resource::Playback].map(|r| effects.is_live(id, r))
                    });
                    let replacement_live = replacements.map(|(_, id)| {
                        [Resource::Capture, Resource::Playback, Resource::Provider]
                            .map(|r| effects.is_live(id, r))
                    });
                    let before = effects.events();
                    let (mut coordinator, result) =
                        finish_quiescence_fixture(&effects, attempt).await;
                    let after = effects.events();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();

                    assert_eq!(entered, 1);
                    assert_eq!(old_live, [[false; 2]; 2]);
                    assert_eq!(
                        replacement_live, [[true; 3]; 2],
                        "prepared replacements must not enter the old-worker cleanup batch"
                    );
                    assert!(
                        !before
                            .iter()
                            .any(|e| matches!(e, ScriptEvent::Activate(..)))
                    );
                    assert_eq!(result, Ok(()));
                    for ((direction, old_id), (_, new_id)) in old.into_iter().zip(replacements) {
                        assert_ne!(old_id, new_id);
                        assert_eq!(
                            after
                                .iter()
                                .filter(|e| **e == ScriptEvent::Activate(new_id, direction))
                                .count(),
                            1
                        );
                    }
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_quiescence_shared_recovery_stops_peer_before_blocked_pcm_release() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let mut coordinator =
                        started(effects.clone(), Arc::new(LifecycleRecorder::default())).await;
                    let blocked = *coordinator.workers.keys().next().unwrap();
                    let peer = if blocked == AudioDirection::Microphone {
                        AudioDirection::Speaker
                    } else {
                        AudioDirection::Microphone
                    };
                    let peer_id = effects.last_acquisition_id(peer);
                    let blocked_id = effects.last_acquisition_id(blocked);
                    effects.clear_events();
                    effects.stall_cleanup(blocked, Resource::Capture);
                    effects.trigger(
                        blocked,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );
                    let deadline = transaction_deadline();
                    let attempt = tokio::task::spawn_local(async move {
                        let result = coordinator.handle_next_worker_completion(deadline).await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.active_cleanup_stalls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    settle_ready_tasks().await;
                    let entered = effects.active_cleanup_stalls();
                    let peer_live = [Resource::Capture, Resource::Playback]
                        .map(|r| effects.is_live(peer_id, r));
                    let before = effects.events();
                    tokio::time::advance(DIRECTION_CLEANUP_BUDGET).await;
                    settle_ready_tasks().await;
                    let (mut coordinator, first) =
                        finish_quiescence_fixture(&effects, attempt).await;
                    let retained = coordinator.has_cleanup_owner(blocked)
                        && effects.is_live(blocked_id, Resource::Capture);
                    let deadlines = effects.deadlines();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();

                    assert_eq!(entered, 1);
                    assert_eq!(peer_live, [false; 2]);
                    assert!(retained);
                    assert_eq!(first, Err(DuplexRuntimeError::StopFailed));
                    assert!(!before.iter().any(|e| matches!(
                        e,
                        ScriptEvent::GenerationKill
                            | ScriptEvent::GenerationStart
                            | ScriptEvent::DiscardProvider(..)
                    )));
                    assert!(deadlines.iter().all(|(_, _, d)| *d == deadline));
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread")]
        async fn safe_admitted_disabled_leg_acquires_no_provider_capture_or_playback() {
            tokio::task::LocalSet::new().run_until(async {
                for enabled in [AudioDirection::Microphone, AudioDirection::Speaker] {
                    let mut snapshot = RuntimeSnapshot::default();
                    for state in &mut snapshot.directions { state.enabled = state.direction_id == enabled; }
                    let mut facts = crate::control_application::safe_admission_tests::ready_facts();
                    if enabled == AudioDirection::Speaker {
                        facts.devices.source.selected = None;
                        facts.devices.source.pinned_name = None;
                        facts.devices.source.health = translator_audio::DeviceHealth::DeviceUnavailable;
                    }
                    let admitted = crate::acoustic_admission::admit_translation(snapshot, facts).unwrap();
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = coordinator(effects.clone(), lifecycle);
                    coordinator.desired = DuplexLaunch::from(admitted);
                    let start_result = coordinator.start_resources(transaction_deadline()).await;
                    let events = effects.events();
                    let live = effects.live_count();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();

                    start_result.unwrap();
                    assert_eq!(live, 3);
                    for resource in [Resource::Provider, Resource::Capture, Resource::Playback] {
                        assert_eq!(events.iter().filter(|event| matches!(event, ScriptEvent::Acquire(_, direction, actual) if *direction == enabled && *actual == resource)).count(), 1);
                    }
                    assert!(!events.iter().any(|event| matches!(event, ScriptEvent::Acquire(_, direction, _) if *direction != enabled)));
                    assert_eq!(effects.live_count(), 0);
                }
            }).await;
        }

        pub(crate) async fn drive_local_recovery_exhaustion(
            generation: u64,
            lifecycle: Arc<dyn DuplexCompletionObserver>,
        ) {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let mut coordinator =
                        coordinator_with_completion(effects.clone(), generation, lifecycle);
                    coordinator
                        .start_resources(transaction_deadline())
                        .await
                        .unwrap();
                    for expected_delay in [50, 100, 200] {
                        effects.clear_events();
                        effects.trigger(
                            AudioDirection::Microphone,
                            Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                        );
                        let recovery = tokio::task::spawn_local(async move {
                            let result = coordinator
                                .handle_next_worker_completion(transaction_deadline())
                                .await;
                            (coordinator, result)
                        });
                        assert_prepare_after_exact_delay(
                            &effects,
                            AudioDirection::Microphone,
                            0,
                            Duration::from_millis(expected_delay),
                        )
                        .await;
                        let completed = finish_recovery(recovery).await;
                        coordinator = completed.0;
                        completed.1.unwrap();
                    }
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                    );
                    coordinator
                        .handle_next_worker_completion(transaction_deadline())
                        .await
                        .unwrap();
                    coordinator.lifecycle = None;
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        async fn settle_ready_tasks() {
            for _ in 0..16 {
                tokio::task::yield_now().await;
            }
        }

        async fn assert_prepare_after_exact_delay(
            effects: &ScriptedEffects,
            direction: AudioDirection,
            previous_count: usize,
            delay: Duration,
        ) {
            settle_ready_tasks().await;
            assert_eq!(effects.provider_acquisitions(direction), previous_count);
            tokio::time::advance(delay - Duration::from_millis(1)).await;
            settle_ready_tasks().await;
            assert_eq!(effects.provider_acquisitions(direction), previous_count);
            tokio::time::advance(Duration::from_millis(1)).await;
            for _ in 0..64 {
                if effects.provider_acquisitions(direction) == previous_count + 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(effects.provider_acquisitions(direction), previous_count + 1);
        }

        async fn assert_event_after_exact_delay(
            effects: &ScriptedEffects,
            event: ScriptEvent,
            previous_count: usize,
            delay: Duration,
        ) {
            settle_ready_tasks().await;
            assert_eq!(effects.event_count(&event), previous_count);
            tokio::time::advance(delay - Duration::from_millis(1)).await;
            settle_ready_tasks().await;
            assert_eq!(effects.event_count(&event), previous_count);
            tokio::time::advance(Duration::from_millis(1)).await;
            for _ in 0..64 {
                if effects.event_count(&event) == previous_count + 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(effects.event_count(&event), previous_count + 1);
        }

        async fn finish_recovery(
            recovery: tokio::task::JoinHandle<(
                DuplexCoordinator<ScriptedSidecar, ScriptedEffects>,
                Result<(), DuplexRuntimeError>,
            )>,
        ) -> (
            DuplexCoordinator<ScriptedSidecar, ScriptedEffects>,
            Result<(), DuplexRuntimeError>,
        ) {
            settle_ready_tasks().await;
            assert!(
                recovery.is_finished(),
                "recovery must finish after its final wake"
            );
            recovery.await.unwrap()
        }

        #[tokio::test(flavor = "current_thread")]
        async fn partial_acquisition_faults_compensate_exact_slots_without_touching_a() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let original_epochs = coordinator.worker_epochs();
                    assert_eq!(effects.live_count(), 6);
                    assert_eq!(coordinator.supervisor.active_sessions().len(), 2);

                    let stages = [
                        PrepareStage::Provider,
                        PrepareStage::Capture,
                        PrepareStage::Playback,
                        PrepareStage::AfterRegister,
                    ];
                    for index in 0..100 {
                        let direction = if index % 2 == 0 {
                            AudioDirection::Microphone
                        } else {
                            AudioDirection::Speaker
                        };
                        effects.clear_events();
                        effects.fail_prepare(direction, stages[index % stages.len()]);
                        let candidate = replacement(coordinator.desired.clone());
                        assert_eq!(
                            coordinator
                                .reconfigure(candidate, transaction_deadline())
                                .await,
                            Err(DuplexRuntimeError::ReconfigureFailed)
                        );
                        assert!(
                            !effects
                                .events()
                                .iter()
                                .any(|event| matches!(event, ScriptEvent::Activate(..))),
                            "a failed candidate batch must not activate either direction"
                        );
                        assert_eq!(coordinator.worker_epochs(), original_epochs);
                        assert_eq!(coordinator.supervisor.active_sessions().len(), 2);
                        assert_eq!(effects.live_count(), 6);
                    }

                    effects.fail_prepare(AudioDirection::Speaker, PrepareStage::AfterRegister);
                    effects.fail_cleanup(AudioDirection::Microphone, Resource::Capture);
                    let candidate = provider_replacement(coordinator.desired.clone());
                    assert_eq!(
                        coordinator
                            .reconfigure(candidate, transaction_deadline())
                            .await,
                        Err(DuplexRuntimeError::RestoreFailed)
                    );
                    assert!(effects.live_count() > 6);
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("the acquisition/compensation matrix must remain bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn failed_candidate_compensation_retains_each_exact_owner_until_stop_retry() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    for resource in [Resource::Provider, Resource::Capture, Resource::Playback] {
                        let effects = ScriptedEffects::default();
                        let lifecycle = Arc::new(LifecycleRecorder::default());
                        let mut coordinator = started(effects.clone(), lifecycle).await;
                        let original_epochs = coordinator.worker_epochs();
                        let old_active_id = effects.last_acquisition_id(AudioDirection::Speaker);
                        effects.clear_events();
                        effects.fail_prepare(AudioDirection::Speaker, PrepareStage::AfterRegister);
                        effects.fail_cleanup(AudioDirection::Speaker, resource);

                        assert_eq!(
                            coordinator
                                .reconfigure(
                                    replacement(coordinator.desired.clone()),
                                    transaction_deadline(),
                                )
                                .await,
                            Err(DuplexRuntimeError::RestoreFailed)
                        );
                        let retained_id = effects.last_acquisition_id(AudioDirection::Speaker);
                        assert!(effects.is_live(retained_id, resource));
                        assert!(coordinator.has_cleanup_owner(AudioDirection::Speaker));
                        assert_eq!(coordinator.worker_epochs(), original_epochs);
                        assert!(
                            !effects
                                .events()
                                .iter()
                                .any(|event| matches!(event, ScriptEvent::Activate(..)))
                        );

                        let rejected_at = effects.events().len();
                        assert_eq!(
                            coordinator
                                .reconfigure(
                                    replacement(coordinator.desired.clone()),
                                    transaction_deadline(),
                                )
                                .await,
                            Err(DuplexRuntimeError::RestoreFailed)
                        );
                        assert!(
                            !effects.events()[rejected_at..].iter().any(|event| matches!(
                                event,
                                ScriptEvent::Acquire(..) | ScriptEvent::Activate(..)
                            ))
                        );

                        let before_retry = effects.events().len();
                        coordinator.shutdown(cleanup_deadline()).await.unwrap();
                        let events = effects.events();
                        let resource_stops = |events: &[ScriptEvent]| {
                            events
                                .iter()
                                .filter_map(|event| match event {
                                    ScriptEvent::Stop(id, AudioDirection::Speaker, observed)
                                        if *observed == resource =>
                                    {
                                        Some(*id)
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(resource_stops(&events[..before_retry]), [retained_id]);
                        let retry_stops = resource_stops(&events[before_retry..]);
                        assert_eq!(retry_stops.len(), 2);
                        assert_ne!(retained_id, old_active_id);
                        assert_eq!(
                            retry_stops.iter().filter(|id| **id == retained_id).count(),
                            1
                        );
                        assert_eq!(
                            retry_stops
                                .iter()
                                .filter(|id| **id == old_active_id)
                                .count(),
                            1
                        );
                        let retry_ids = resource_stops(&events);
                        assert_eq!(retry_ids.iter().filter(|id| **id == retained_id).count(), 2);
                        assert_eq!(
                            retry_ids.iter().filter(|id| **id == old_active_id).count(),
                            1
                        );
                        assert_eq!(effects.live_count(), 0);
                    }
                }))
                .await
                .expect("candidate compensation and exact-owner Stop retry must stay bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn batch_activates_after_all_prepares_and_preserves_unaffected_peer() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let old_ids = [
                        effects.last_acquisition_id(AudioDirection::Microphone),
                        effects.last_acquisition_id(AudioDirection::Speaker),
                    ];
                    let old_sessions = coordinator
                        .supervisor
                        .active_sessions()
                        .iter()
                        .copied()
                        .collect::<HashSet<_>>();
                    let generation = coordinator.supervisor.launch().unwrap().generation_id;
                    effects.clear_events();

                    let candidate = provider_replacement(coordinator.desired.clone());
                    coordinator
                        .reconfigure(candidate, transaction_deadline())
                        .await
                        .unwrap();
                    let events = effects.events();
                    let last_acquire = events
                        .iter()
                        .rposition(|event| matches!(event, ScriptEvent::Acquire(..)))
                        .unwrap();
                    let first_activate = events
                        .iter()
                        .position(|event| matches!(event, ScriptEvent::Activate(..)))
                        .unwrap();
                    assert!(last_acquire < first_activate);
                    let replacement_ids = events
                        .iter()
                        .filter_map(|event| match event {
                            ScriptEvent::Activate(id, _) => Some(*id),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(replacement_ids.len(), 2);
                    assert!(replacement_ids.iter().all(|id| !old_ids.contains(id)));
                    let replacement_sessions = coordinator
                        .supervisor
                        .active_sessions()
                        .iter()
                        .copied()
                        .collect::<HashSet<_>>();
                    assert_eq!(replacement_sessions.len(), 2);
                    assert!(replacement_sessions.is_disjoint(&old_sessions));
                    assert_eq!(
                        coordinator.supervisor.launch().unwrap().generation_id,
                        generation
                    );
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationKill), 0);
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationStart), 0);

                    let peer_epoch = coordinator.worker_epoch(AudioDirection::Speaker).unwrap();
                    let microphone_epoch = coordinator
                        .worker_epoch(AudioDirection::Microphone)
                        .unwrap();
                    effects.clear_events();
                    let mut candidate = coordinator.desired.clone();
                    candidate
                        .microphone
                        .as_mut()
                        .unwrap()
                        .runtime
                        .debug_text_enabled = true;
                    coordinator
                        .reconfigure(candidate, transaction_deadline())
                        .await
                        .unwrap();

                    let events = effects.events();
                    assert_eq!(
                        coordinator.worker_epoch(AudioDirection::Speaker),
                        Some(peer_epoch)
                    );
                    assert_ne!(
                        coordinator.worker_epoch(AudioDirection::Microphone),
                        Some(microphone_epoch)
                    );
                    assert!(events.iter().any(|event| matches!(
                        event,
                        ScriptEvent::Activate(_, AudioDirection::Microphone)
                    )));
                    assert!(!events.iter().any(|event| matches!(
                        event,
                        ScriptEvent::Stop(_, AudioDirection::Speaker, _)
                            | ScriptEvent::CloseProvider(_, AudioDirection::Speaker)
                    )));
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("batch activation and peer preservation must remain bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn candidate_entry_failure_restores_exact_previous_active_set() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let previous = coordinator.desired.clone();
                    let previous_ids = [
                        effects.last_acquisition_id(AudioDirection::Microphone),
                        effects.last_acquisition_id(AudioDirection::Speaker),
                    ];
                    let candidate_first_epoch = coordinator.next_epoch + 1;
                    let candidate_last_epoch = candidate_first_epoch + 1;
                    let lifecycle_start = lifecycle.statuses.lock().unwrap().len();
                    effects.clear_events();
                    effects.fail_entry(AudioDirection::Speaker);

                    let result = coordinator
                        .reconfigure(
                            provider_replacement(previous.clone()),
                            transaction_deadline(),
                        )
                        .await;
                    let restored_ids = [
                        effects.last_acquisition_id(AudioDirection::Microphone),
                        effects.last_acquisition_id(AudioDirection::Speaker),
                    ];
                    assert_eq!(result, Err(DuplexRuntimeError::ReconfigureFailed));
                    assert!(coordinator.desired == previous);
                    assert!(restored_ids.iter().all(|id| !previous_ids.contains(id)));
                    assert_eq!(coordinator.worker_epochs().len(), 2);
                    assert_eq!(effects.live_count(), 6);
                    assert!(effects.events().iter().any(|event| matches!(
                        event,
                        ScriptEvent::Activate(_, AudioDirection::Microphone)
                    )));
                    assert!(effects.events().iter().any(|event| matches!(
                        event,
                        ScriptEvent::Activate(_, AudioDirection::Speaker)
                    )));
                    assert!(
                        lifecycle.statuses.lock().unwrap()[lifecycle_start..]
                            .iter()
                            .filter(|status| {
                                (candidate_first_epoch..=candidate_last_epoch).contains(&status.2)
                            })
                            .all(|status| status.3 != DirectionRuntimeStatus::Running),
                        "no candidate direction may publish Running before every entry ACK"
                    );

                    let mut next = coordinator.desired.clone();
                    next.microphone.as_mut().unwrap().runtime.debug_text_enabled = true;
                    coordinator
                        .reconfigure(next, transaction_deadline())
                        .await
                        .unwrap();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("activation rollback and previous-set restoration must remain bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn failed_activation_restoration_enters_cleanup_only_and_rejects_reconfigure() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let coordinator = started(effects.clone(), lifecycle).await;
                    effects.clear_events();
                    effects.fail_entry(AudioDirection::Speaker);
                    effects.fail_entry(AudioDirection::Microphone);
                    let (_stop, stop_receiver) = watch::channel(None);
                    let (commands, command_receiver) = mpsc::channel(2);
                    let driver = tokio::task::spawn_local(run_coordinator(
                        coordinator,
                        stop_receiver,
                        command_receiver,
                        None,
                        None,
                    ));

                    let (first_response, first_result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Reconfigure {
                            launch: provider_replacement(launch_pair()),
                            deadline: transaction_deadline(),
                            response: first_response,
                        })
                        .await
                        .unwrap();
                    assert_eq!(
                        command_result(&first_result).await,
                        Err(DuplexRuntimeError::RestoreFailed)
                    );
                    let rejected_at = effects.events().len();
                    let (retry_response, retry_result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Reconfigure {
                            launch: replacement(launch_pair()),
                            deadline: transaction_deadline(),
                            response: retry_response,
                        })
                        .await
                        .unwrap();
                    assert_eq!(
                        command_result(&retry_result).await,
                        Err(DuplexRuntimeError::RestoreFailed)
                    );
                    assert!(
                        !effects.events()[rejected_at..].iter().any(|event| matches!(
                            event,
                            ScriptEvent::Acquire(..) | ScriptEvent::Activate(..)
                        ))
                    );

                    let (stop_response, stop_result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Stop {
                            deadline: cleanup_deadline(),
                            response: stop_response,
                        })
                        .await
                        .unwrap();
                    command_result(&stop_result).await.unwrap();
                    driver.await.unwrap().unwrap_err();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("failed restoration must enter bounded cleanup-only processing");
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn local_fault_exhaustion_preserves_peer_and_projects_failed_direction() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let peer_epoch = coordinator.worker_epoch(AudioDirection::Speaker).unwrap();
                    effects.clear_events();
                    for _ in 0..3 {
                        effects.fail_prepare(AudioDirection::Microphone, PrepareStage::Capture);
                    }
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                    );

                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_prepare_after_exact_delay(
                        &effects,
                        AudioDirection::Microphone,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    assert_prepare_after_exact_delay(
                        &effects,
                        AudioDirection::Microphone,
                        1,
                        Duration::from_millis(100),
                    )
                    .await;
                    assert_prepare_after_exact_delay(
                        &effects,
                        AudioDirection::Microphone,
                        2,
                        Duration::from_millis(200),
                    )
                    .await;
                    assert_eq!(
                        effects.prepare_times(AudioDirection::Microphone),
                        [
                            tokio::time::Instant::now() - Duration::from_millis(300),
                            tokio::time::Instant::now() - Duration::from_millis(200),
                            tokio::time::Instant::now(),
                        ]
                    );
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    result.unwrap();

                    assert_eq!(
                        coordinator.worker_epoch(AudioDirection::Speaker),
                        Some(peer_epoch)
                    );
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Microphone), None);
                    assert!(!effects.events().iter().any(|event| matches!(
                        event,
                        ScriptEvent::GenerationKill
                            | ScriptEvent::Stop(_, AudioDirection::Speaker, _)
                    )));
                    let failed = {
                        let statuses = lifecycle.statuses.lock().unwrap();
                        assert!(statuses.iter().any(|status| {
                            status.1 == AudioDirection::Microphone
                                && status.3 == DirectionRuntimeStatus::Recovering
                        }));
                        statuses.last().copied().unwrap()
                    };
                    assert_eq!(failed.0, 41);
                    assert_eq!(failed.1, AudioDirection::Microphone);
                    assert_eq!(
                        failed.2,
                        coordinator
                            .direction_epoch(AudioDirection::Microphone)
                            .unwrap()
                    );
                    assert_eq!(failed.3, DirectionRuntimeStatus::Failed);
                    assert_eq!(failed.4, Some(DirectionRuntimeFailure::RestartExhausted));

                    let failed_microphone = failed;
                    let microphone_acquisitions =
                        effects.provider_acquisitions(AudioDirection::Microphone);
                    effects.trigger(
                        AudioDirection::Speaker,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );
                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_event_after_exact_delay(
                        &effects,
                        ScriptEvent::GenerationKill,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    result.unwrap();

                    assert_eq!(
                        effects.provider_acquisitions(AudioDirection::Microphone),
                        microphone_acquisitions,
                        "a peer shared restart must not reactivate a locally exhausted direction"
                    );
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Microphone), None);
                    assert!(coordinator.worker_epoch(AudioDirection::Speaker).is_some());
                    assert_eq!(
                        lifecycle
                            .statuses
                            .lock()
                            .unwrap()
                            .iter()
                            .rev()
                            .find(|status| status.1 == AudioDirection::Microphone)
                            .copied()
                            .unwrap(),
                        failed_microphone,
                        "shared recovery must preserve the failed epoch and projection"
                    );

                    let mut explicit_replacement = coordinator.desired.clone();
                    explicit_replacement
                        .microphone
                        .as_mut()
                        .unwrap()
                        .runtime
                        .debug_text_enabled = true;
                    effects.fail_entry(AudioDirection::Microphone);
                    assert_eq!(
                        coordinator
                            .reconfigure(explicit_replacement.clone(), transaction_deadline())
                            .await,
                        Err(DuplexRuntimeError::ReconfigureFailed)
                    );
                    assert_eq!(
                        effects.provider_acquisitions(AudioDirection::Microphone),
                        microphone_acquisitions + 1
                    );
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Microphone), None);
                    assert_eq!(
                        lifecycle
                            .statuses
                            .lock()
                            .unwrap()
                            .iter()
                            .rev()
                            .find(|status| status.1 == AudioDirection::Microphone)
                            .copied()
                            .unwrap(),
                        failed_microphone,
                        "failed explicit replacement must preserve the paused projection"
                    );

                    let acquisitions_after_failed_replacement =
                        effects.provider_acquisitions(AudioDirection::Microphone);
                    let generation_kills = effects.event_count(&ScriptEvent::GenerationKill);
                    effects.trigger(
                        AudioDirection::Speaker,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );
                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_event_after_exact_delay(
                        &effects,
                        ScriptEvent::GenerationKill,
                        generation_kills,
                        Duration::from_millis(100),
                    )
                    .await;
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    result.unwrap();
                    assert_eq!(
                        effects.provider_acquisitions(AudioDirection::Microphone),
                        acquisitions_after_failed_replacement,
                        "a failed explicit replacement must not clear the paused marker"
                    );
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Microphone), None);
                    assert_eq!(
                        lifecycle
                            .statuses
                            .lock()
                            .unwrap()
                            .iter()
                            .rev()
                            .find(|status| status.1 == AudioDirection::Microphone)
                            .copied()
                            .unwrap(),
                        failed_microphone
                    );

                    coordinator
                        .reconfigure(explicit_replacement, transaction_deadline())
                        .await
                        .unwrap();
                    assert_eq!(
                        effects.provider_acquisitions(AudioDirection::Microphone),
                        microphone_acquisitions + 2
                    );
                    let replacement_epoch = coordinator
                        .worker_epoch(AudioDirection::Microphone)
                        .expect("explicit replacement must reactivate the paused direction");
                    assert!(lifecycle.statuses.lock().unwrap().iter().any(|status| {
                        status.1 == AudioDirection::Microphone
                            && status.2 == replacement_epoch
                            && status.3 == DirectionRuntimeStatus::Running
                            && status.4.is_none()
                    }));
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn successful_local_recovery_keeps_backoff_until_explicit_replacement() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;

                    for expected_delay in [50, 100] {
                        effects.clear_events();
                        effects.trigger(
                            AudioDirection::Microphone,
                            Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                        );
                        let recovery = tokio::task::spawn_local(async move {
                            let result = coordinator
                                .handle_next_worker_completion(transaction_deadline())
                                .await;
                            (coordinator, result)
                        });
                        assert_prepare_after_exact_delay(
                            &effects,
                            AudioDirection::Microphone,
                            0,
                            Duration::from_millis(expected_delay),
                        )
                        .await;
                        let completed = finish_recovery(recovery).await;
                        coordinator = completed.0;
                        completed.1.unwrap();
                    }

                    let mut candidate = coordinator.desired.clone();
                    candidate
                        .microphone
                        .as_mut()
                        .unwrap()
                        .runtime
                        .debug_text_enabled = true;
                    coordinator
                        .reconfigure(candidate, transaction_deadline())
                        .await
                        .unwrap();
                    let replacement_epoch = coordinator
                        .worker_epoch(AudioDirection::Microphone)
                        .unwrap();
                    assert!(lifecycle.statuses.lock().unwrap().iter().any(|status| {
                        status.1 == AudioDirection::Microphone
                            && status.2 == replacement_epoch
                            && status.3 == DirectionRuntimeStatus::Running
                            && status.4.is_none()
                    }));

                    effects.clear_events();
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                    );
                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_prepare_after_exact_delay(
                        &effects,
                        AudioDirection::Microphone,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    result.unwrap();

                    let enabled_microphone = coordinator.desired.microphone.clone().unwrap();
                    let mut disabled = coordinator.desired.clone();
                    disabled.microphone = None;
                    coordinator
                        .reconfigure(disabled, transaction_deadline())
                        .await
                        .unwrap();
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Microphone), None);
                    let disabled_status = lifecycle
                        .statuses
                        .lock()
                        .unwrap()
                        .iter()
                        .rev()
                        .find(|status| status.1 == AudioDirection::Microphone)
                        .copied()
                        .unwrap();
                    assert_eq!(disabled_status.3, DirectionRuntimeStatus::Stopped);
                    assert_eq!(disabled_status.4, None);

                    let mut enabled = coordinator.desired.clone();
                    enabled.microphone = Some(enabled_microphone);
                    coordinator
                        .reconfigure(enabled, transaction_deadline())
                        .await
                        .unwrap();
                    let enabled_epoch = coordinator
                        .worker_epoch(AudioDirection::Microphone)
                        .unwrap();
                    let enabled_status = lifecycle
                        .statuses
                        .lock()
                        .unwrap()
                        .iter()
                        .rev()
                        .find(|status| status.1 == AudioDirection::Microphone)
                        .copied()
                        .unwrap();
                    assert_eq!(enabled_status.2, enabled_epoch);
                    assert_eq!(enabled_status.3, DirectionRuntimeStatus::Running);
                    assert_eq!(enabled_status.4, None);

                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                    for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
                        let last = lifecycle
                            .statuses
                            .lock()
                            .unwrap()
                            .iter()
                            .rev()
                            .find(|status| status.1 == direction)
                            .copied()
                            .unwrap();
                        assert_eq!(last.3, DirectionRuntimeStatus::Stopped);
                        assert_eq!(last.4, None);
                    }
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_exhausted_local_recovery_publishes_a_fresh_failed_epoch() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let mut last_running_epoch = coordinator
                        .worker_epoch(AudioDirection::Microphone)
                        .expect("the microphone starts with an active epoch");

                    for expected_delay in [50, 100, 200] {
                        effects.clear_events();
                        effects.trigger(
                            AudioDirection::Microphone,
                            Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                        );
                        let recovery = tokio::task::spawn_local(async move {
                            let result = coordinator
                                .handle_next_worker_completion(transaction_deadline())
                                .await;
                            (coordinator, result)
                        });
                        assert_prepare_after_exact_delay(
                            &effects,
                            AudioDirection::Microphone,
                            0,
                            Duration::from_millis(expected_delay),
                        )
                        .await;
                        let completed = finish_recovery(recovery).await;
                        coordinator = completed.0;
                        completed.1.unwrap();
                        let running_epoch = coordinator
                            .worker_epoch(AudioDirection::Microphone)
                            .expect("a successful local recovery installs a worker");
                        assert!(running_epoch > last_running_epoch);
                        last_running_epoch = running_epoch;
                    }

                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                    );
                    coordinator
                        .handle_next_worker_completion(transaction_deadline())
                        .await
                        .unwrap();
                    assert_eq!(coordinator.worker_epoch(AudioDirection::Microphone), None);
                    let failed = lifecycle
                        .statuses
                        .lock()
                        .unwrap()
                        .iter()
                        .rev()
                        .find(|status| status.1 == AudioDirection::Microphone)
                        .copied()
                        .expect("retry exhaustion publishes direction health");
                    assert_eq!(failed.3, DirectionRuntimeStatus::Failed);
                    assert_eq!(failed.4, Some(DirectionRuntimeFailure::RestartExhausted));
                    assert!(
                        failed.2 > last_running_epoch,
                        "a failed terminal direction state cannot reuse the Running epoch"
                    );

                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_joined_direction_resets_observer_before_cleanup_and_recovery() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let old_id = effects.last_acquisition_id(AudioDirection::Microphone);
                    effects.clear_events();
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                    );
                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_prepare_after_exact_delay(
                        &effects,
                        AudioDirection::Microphone,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    result.unwrap();

                    let events = effects.events();
                    let reset = events
                        .iter()
                        .position(|event| {
                            *event == ScriptEvent::ObserverReset(AudioDirection::Microphone)
                        })
                        .expect("the joined direction must reset its observer state");
                    let first_old_cleanup = events
                        .iter()
                        .position(
                            |event| matches!(event, ScriptEvent::Stop(id, _, _) if *id == old_id),
                        )
                        .expect("the old direction owner must be cleaned");
                    let replacement = events
                        .iter()
                        .position(|event| {
                            matches!(
                                event,
                                ScriptEvent::Acquire(id, AudioDirection::Microphone, _)
                                    if *id != old_id
                            )
                        })
                        .expect("local recovery must acquire one replacement owner");
                    assert!(reset < first_old_cleanup);
                    assert!(reset < replacement);

                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread")]
        async fn source_p1_local_playback_fault_cleans_old_owner_before_any_replacement() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let old_id = effects.last_acquisition_id(AudioDirection::Microphone);
                    effects.clear_events();
                    effects.fail_cleanup(AudioDirection::Microphone, Resource::Playback);
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::PcmPlayback),
                    );

                    assert_eq!(
                        coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await,
                        Err(DuplexRuntimeError::StopFailed)
                    );
                    let events = effects.events();
                    assert!(events.iter().any(|event| {
                        *event
                            == ScriptEvent::Stop(
                                old_id,
                                AudioDirection::Microphone,
                                Resource::Playback,
                            )
                    }));
                    assert!(!events.iter().any(|event| {
                        matches!(
                            event,
                            ScriptEvent::Acquire(id, AudioDirection::Microphone, _)
                                if *id != old_id
                        )
                    }));
                    assert!(coordinator.has_cleanup_owner(AudioDirection::Microphone));

                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("playback fault cleanup ordering must remain bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn source_p1_clean_stop_resets_each_joined_direction_before_resource_cleanup() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let ids = [
                        (
                            AudioDirection::Microphone,
                            effects.last_acquisition_id(AudioDirection::Microphone),
                        ),
                        (
                            AudioDirection::Speaker,
                            effects.last_acquisition_id(AudioDirection::Speaker),
                        ),
                    ];
                    effects.clear_events();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();

                    let events = effects.events();
                    for (direction, id) in ids {
                        let reset = events
                            .iter()
                            .position(|event| *event == ScriptEvent::ObserverReset(direction))
                            .expect("each joined direction must reset observer correlation");
                        let cleanup = events
                            .iter()
                            .position(|event| {
                                matches!(event, ScriptEvent::Stop(observed, _, _) if *observed == id)
                            })
                            .expect("the stopped direction must clean its exact owner");
                        assert!(reset < cleanup);
                    }
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("clean Stop and observer reset must remain bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn production_failure_classifier_probes_only_ambiguous_provider_transport() {
            use translator_ipc::ProviderClientError;

            let cases = [
                (ProviderClientError::InvalidToken, FaultScope::Local),
                (ProviderClientError::InvalidEndpoint, FaultScope::Local),
                (ProviderClientError::InvalidOpenRequest, FaultScope::Local),
                (ProviderClientError::EventStreamProtocol, FaultScope::Local),
                (
                    ProviderClientError::EventStreamResourceExhausted,
                    FaultScope::Local,
                ),
                (ProviderClientError::EventStreamCancelled, FaultScope::Local),
                (
                    ProviderClientError::TransportUnavailable,
                    FaultScope::Shared,
                ),
                (ProviderClientError::EventStreamInternal, FaultScope::Shared),
                (
                    ProviderClientError::InvalidProbeResponse,
                    FaultScope::Shared,
                ),
                (
                    ProviderClientError::ProviderReadyTimeout,
                    FaultScope::Shared,
                ),
                (
                    ProviderClientError::RequestChannelClosed,
                    FaultScope::ProviderConnection,
                ),
                (
                    ProviderClientError::EventStreamFailed,
                    FaultScope::ProviderConnection,
                ),
            ];
            for (error, expected) in cases {
                assert_eq!(classify_provider_client_error(&error), expected);
            }
            assert_eq!(classify_provider_eof(), FaultScope::ProviderConnection);

            assert_eq!(
                [
                    DirectionFailureOrigin::PcmCapture,
                    DirectionFailureOrigin::PcmPlayback,
                    DirectionFailureOrigin::Queue,
                    DirectionFailureOrigin::Vad,
                    DirectionFailureOrigin::SessionValidation,
                ]
                .map(classify_direction_failure),
                [FaultScope::Local; 5]
            );
            assert_eq!(
                classify_direction_failure(DirectionFailureOrigin::ProviderConnection),
                FaultScope::ProviderConnection
            );
            assert_eq!(
                [
                    DirectionFailureOrigin::SidecarExit,
                    DirectionFailureOrigin::InternalTransport,
                    DirectionFailureOrigin::WatchdogRestart,
                ]
                .map(classify_direction_failure),
                [FaultScope::Shared; 3]
            );

            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    assert_eq!(effects.probe_calls(), 0);
                    assert_eq!(
                        coordinator
                            .resolve_fault_scope(FaultScope::Local, transaction_deadline())
                            .await,
                        Ok(FaultScope::Local)
                    );
                    assert_eq!(
                        coordinator
                            .resolve_fault_scope(FaultScope::Shared, transaction_deadline())
                            .await,
                        Ok(FaultScope::Shared)
                    );
                    assert_eq!(effects.probe_calls(), 0);
                    assert_eq!(
                        coordinator
                            .resolve_fault_scope(
                                FaultScope::ProviderConnection,
                                transaction_deadline(),
                            )
                            .await,
                        Ok(FaultScope::Local)
                    );
                    assert_eq!(effects.probe_calls(), 1);
                    effects.set_probe_ready(false);
                    assert_eq!(
                        coordinator
                            .resolve_fault_scope(
                                FaultScope::ProviderConnection,
                                transaction_deadline(),
                            )
                            .await,
                        Ok(FaultScope::Shared)
                    );
                    assert_eq!(effects.probe_calls(), 2);
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                }))
                .await
                .expect("the generation probe classifier must remain bounded");
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn provider_transport_fault_restarts_one_session_or_the_generation_after_probe() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let peer_epoch = coordinator.worker_epoch(AudioDirection::Speaker).unwrap();
                    effects.clear_events();
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::ProviderConnection),
                    );
                    let local = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_prepare_after_exact_delay(
                        &effects,
                        AudioDirection::Microphone,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    let completed = finish_recovery(local).await;
                    coordinator = completed.0;
                    completed.1.unwrap();
                    assert_eq!(effects.probe_calls(), 1);
                    assert_eq!(
                        coordinator.worker_epoch(AudioDirection::Speaker),
                        Some(peer_epoch)
                    );
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationKill), 0);

                    let prior_epochs = coordinator.worker_epochs();
                    effects.clear_events();
                    effects.fail_next_probe();
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::ProviderConnection),
                    );
                    let shared = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_event_after_exact_delay(
                        &effects,
                        ScriptEvent::GenerationKill,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    let (mut coordinator, result) = finish_recovery(shared).await;
                    result.unwrap();
                    assert_eq!(effects.probe_calls(), 2);
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationKill), 1);
                    assert!(
                        coordinator
                            .worker_epochs()
                            .iter()
                            .all(|(direction, epoch)| prior_epochs.get(direction) != Some(epoch))
                    );
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn shared_fault_reaps_both_pcm_before_one_generation_restart() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let old_ids = [
                        (
                            AudioDirection::Microphone,
                            effects.last_acquisition_id(AudioDirection::Microphone),
                        ),
                        (
                            AudioDirection::Speaker,
                            effects.last_acquisition_id(AudioDirection::Speaker),
                        ),
                    ];
                    effects.clear_events();
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );

                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_event_after_exact_delay(
                        &effects,
                        ScriptEvent::GenerationKill,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    result.unwrap();

                    let events = effects.events();
                    let generation_kill = events
                        .iter()
                        .position(|event| event == &ScriptEvent::GenerationKill)
                        .unwrap();
                    let first_replacement = events
                        .iter()
                        .position(|event| matches!(event, ScriptEvent::Activate(..)))
                        .unwrap();
                    for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
                        for resource in [Resource::Capture, Resource::Playback] {
                            assert!(events[..generation_kill].iter().any(|event| matches!(
                                event,
                                ScriptEvent::Stop(_, observed_direction, observed_resource)
                                    if *observed_direction == direction
                                        && *observed_resource == resource
                            )));
                        }
                    }
                    assert!(
                        !events
                            .iter()
                            .any(|event| matches!(event, ScriptEvent::CloseProvider(..)))
                    );
                    for (direction, id) in old_ids {
                        let discard = ScriptEvent::DiscardProvider(id, direction);
                        assert_eq!(events.iter().filter(|event| **event == discard).count(), 1);
                        assert!(events[..first_replacement].contains(&discard));
                    }
                    assert_eq!(
                        events
                            .iter()
                            .filter(|event| event == &&ScriptEvent::GenerationKill)
                            .count(),
                        1
                    );
                    assert_eq!(coordinator.worker_epochs().len(), 2);
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn shared_fault_budget_is_persistent_and_never_multiplies_supervisor_retries() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    effects.clear_events();

                    for (index, expected_delay) in [50, 100, 200].into_iter().enumerate() {
                        effects.trigger(
                            AudioDirection::Microphone,
                            Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                        );
                        let recovery = tokio::task::spawn_local(async move {
                            let result = coordinator
                                .handle_next_worker_completion(transaction_deadline())
                                .await;
                            (coordinator, result)
                        });
                        assert_event_after_exact_delay(
                            &effects,
                            ScriptEvent::GenerationKill,
                            index,
                            Duration::from_millis(expected_delay),
                        )
                        .await;
                        let completed = finish_recovery(recovery).await;
                        coordinator = completed.0;
                        completed.1.unwrap();
                    }

                    assert_eq!(
                        effects
                            .events()
                            .iter()
                            .filter_map(|event| match event {
                                ScriptEvent::RecoveryAttempt(attempt) => Some(*attempt),
                                _ => None,
                            })
                            .collect::<Vec<_>>(),
                        [1, 2, 3]
                    );
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationKill), 3);

                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );
                    let terminal = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    let (mut coordinator, result) = finish_recovery(terminal).await;
                    assert_eq!(result, Err(DuplexRuntimeError::StartFailed));
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationKill), 3);
                    assert_eq!(
                        effects
                            .events()
                            .iter()
                            .filter(|event| matches!(event, ScriptEvent::RecoveryAttempt(_)))
                            .count(),
                        3
                    );
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn failed_supervisor_restart_is_one_outer_attempt_not_three_by_three() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let old_generation = coordinator.supervisor.launch().unwrap().generation_id;
                    let old_ids = [
                        (
                            AudioDirection::Microphone,
                            effects.last_acquisition_id(AudioDirection::Microphone),
                        ),
                        (
                            AudioDirection::Speaker,
                            effects.last_acquisition_id(AudioDirection::Speaker),
                        ),
                    ];
                    effects.clear_events();
                    effects.fail_sidecar_starts(MAX_START_ATTEMPTS);
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );

                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_event_after_exact_delay(
                        &effects,
                        ScriptEvent::GenerationKill,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    assert_eq!(result, Err(DuplexRuntimeError::StartFailed));
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationKill), 1);
                    assert_eq!(
                        effects.event_count(&ScriptEvent::GenerationStart),
                        MAX_START_ATTEMPTS
                    );
                    assert_eq!(
                        effects
                            .events()
                            .iter()
                            .filter(|event| matches!(event, ScriptEvent::RecoveryAttempt(_)))
                            .count(),
                        1
                    );
                    assert!(coordinator.worker_epochs().is_empty());
                    assert_eq!(
                        coordinator
                            .supervisor
                            .generation_retirement(old_generation),
                        None,
                        "post-reap start failure still retires providers and acknowledges the receipt"
                    );
                    for (direction, id) in old_ids {
                        assert_eq!(
                            effects.event_count(&ScriptEvent::DiscardProvider(id, direction)),
                            1
                        );
                        assert_eq!(
                            effects.event_count(&ScriptEvent::CloseProvider(id, direction)),
                            0
                        );
                    }
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn cancelled_mid_provider_discard_keeps_retirement_receipt_until_same_owner_retry() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let old_generation = coordinator.supervisor.launch().unwrap().generation_id;
                    let ordered_directions = coordinator.workers.keys().copied().collect::<Vec<_>>();
                    assert_eq!(ordered_directions.len(), 2);
                    let blocked_direction = ordered_directions[1];
                    let blocked_cell = coordinator
                        .workers
                        .get(&blocked_direction)
                        .unwrap()
                        .cell
                        .clone();
                    let old_ids = [
                        (
                            AudioDirection::Microphone,
                            effects.last_acquisition_id(AudioDirection::Microphone),
                        ),
                        (
                            AudioDirection::Speaker,
                            effects.last_acquisition_id(AudioDirection::Speaker),
                        ),
                    ];
                    effects.clear_events();
                    let start_gate = effects.stall_next_sidecar_start();
                    let helper_effects = effects.clone();
                    let (cancel, cancelled) = oneshot::channel();
                    let (release, released) = oneshot::channel();
                    let helper = tokio::task::spawn_local(async move {
                        start_gate.wait_entered().await;
                        let guard = blocked_cell.lock().await;
                        start_gate.release();
                        let mut reached_mid_discard = false;
                        for _ in 0..1_024 {
                            let discarded = helper_effects
                                .events()
                                .iter()
                                .filter(|event| matches!(event, ScriptEvent::DiscardProvider(..)))
                                .count();
                            if discarded == 1 {
                                reached_mid_discard = true;
                                break;
                            }
                            tokio::task::yield_now().await;
                        }
                        assert!(
                            reached_mid_discard,
                            "the fail-safe must not spin forever before cancellation"
                        );
                        let _ = cancel.send(());
                        let _ = released.await;
                        drop(guard);
                    });

                    let restart_deadline =
                        tokio::time::Instant::now() + DIRECTION_CLEANUP_BUDGET;
                    let mut restart = Box::pin(coordinator.restart_shared(restart_deadline));
                    tokio::select! {
                        _ = cancelled => {}
                        result = &mut restart => {
                            panic!("restart unexpectedly completed before mid-discard cancellation: {result:?}")
                        }
                    }
                    drop(restart);
                    let _ = release.send(());
                    helper.await.unwrap();

                    assert_eq!(
                        coordinator
                            .supervisor
                            .generation_retirement(old_generation),
                        Some(crate::GenerationRetirement {
                            generation_id: old_generation,
                            old_generation_reaped: true,
                        })
                    );
                    let before_retry = effects.events();
                    assert_eq!(
                        before_retry
                            .iter()
                            .filter(|event| matches!(event, ScriptEvent::DiscardProvider(..)))
                            .count(),
                        1
                    );
                    assert!(
                        !before_retry
                            .iter()
                            .any(|event| matches!(event, ScriptEvent::CloseProvider(..)))
                    );

                    coordinator
                        .shutdown(tokio::time::Instant::now() + RUNTIME_CLEANUP_BUDGET)
                        .await
                        .unwrap();
                    let after_retry = effects.events();
                    for (direction, id) in old_ids {
                        assert_eq!(
                            after_retry
                                .iter()
                                .filter(|event| {
                                    **event == ScriptEvent::DiscardProvider(id, direction)
                                })
                                .count(),
                            1
                        );
                        assert!(!after_retry.contains(&ScriptEvent::CloseProvider(id, direction)));
                    }
                    assert_eq!(
                        coordinator
                            .supervisor
                            .generation_retirement(old_generation),
                        None
                    );
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("mid-discard cancellation must preserve the exact retirement receipt");
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn unconfirmed_generation_reap_retains_provider_handles_until_same_owner_stop() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let old_generation = coordinator.supervisor.launch().unwrap().generation_id;
                    let old_ids = [
                        (
                            AudioDirection::Microphone,
                            effects.last_acquisition_id(AudioDirection::Microphone),
                        ),
                        (
                            AudioDirection::Speaker,
                            effects.last_acquisition_id(AudioDirection::Speaker),
                        ),
                    ];
                    effects.clear_events();
                    effects.fail_sidecar_kills(1);
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );
                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await;
                        (coordinator, result)
                    });
                    assert_event_after_exact_delay(
                        &effects,
                        ScriptEvent::GenerationKill,
                        0,
                        Duration::from_millis(50),
                    )
                    .await;
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    let before_stop = effects.events();
                    let retirement_before_stop =
                        coordinator.supervisor.generation_retirement(old_generation);
                    let retained_before_stop = old_ids
                        .iter()
                        .all(|(direction, _)| coordinator.has_cleanup_owner(*direction));
                    let premature_retirement = before_stop.iter().any(|event| {
                        matches!(
                            event,
                            ScriptEvent::DiscardProvider(..) | ScriptEvent::CloseProvider(..)
                        )
                    });

                    assert_eq!(result, Err(DuplexRuntimeError::StartFailed));
                    assert!(
                        retained_before_stop,
                        "unconfirmed reap must retain both provider owners"
                    );
                    assert!(
                        !premature_retirement,
                        "unconfirmed reap must neither close nor discard old providers"
                    );
                    assert_eq!(
                        retirement_before_stop,
                        Some(crate::GenerationRetirement {
                            generation_id: old_generation,
                            old_generation_reaped: false,
                        })
                    );
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    let all_events = effects.events();
                    for (direction, id) in old_ids {
                        assert_eq!(
                            all_events
                                .iter()
                                .filter(|event| {
                                    **event == ScriptEvent::DiscardProvider(id, direction)
                                })
                                .count(),
                            1,
                            "same-owner shutdown must discard after it confirms old reap"
                        );
                        assert!(!all_events.contains(&ScriptEvent::CloseProvider(id, direction)));
                    }
                    assert_eq!(
                        coordinator.supervisor.generation_retirement(old_generation),
                        None,
                        "same-owner Stop must acknowledge only after every old handle is discarded"
                    );
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread")]
        async fn shared_fault_never_restarts_generation_before_all_pcm_is_reaped() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    effects.clear_events();
                    effects.fail_cleanup(AudioDirection::Microphone, Resource::Capture);
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );

                    assert_eq!(
                        coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await,
                        Err(DuplexRuntimeError::StopFailed)
                    );
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationKill), 0);
                    assert_eq!(effects.event_count(&ScriptEvent::GenerationStart), 0);
                    assert!(coordinator.has_cleanup_owner(AudioDirection::Microphone));

                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("failed PCM cleanup and exact-owner retry must remain bounded");
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_sidecar_start_finishing_at_expiry_does_not_poll_later_effects() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let gate = effects.stall_next_sidecar_start();
                    let mut first_coordinator = coordinator(effects.clone(), lifecycle);
                    let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
                    let attempt = tokio::task::spawn_local(async move {
                        let result = first_coordinator.start_resources(deadline).await;
                        (first_coordinator, result)
                    });
                    gate.wait_entered().await;
                    tokio::time::advance(Duration::from_millis(100)).await;
                    gate.release();
                    settle_ready_tasks().await;
                    let (mut first_coordinator, result) = attempt.await.unwrap();
                    let events = effects.events();
                    let later_effects = events.iter().filter(|event| {
                        matches!(
                            event,
                            ScriptEvent::WaitReady
                                | ScriptEvent::Acquire(..)
                                | ScriptEvent::RegisterPoll
                                | ScriptEvent::Activate(..)
                        )
                    }).count();
                    first_coordinator
                        .shutdown(cleanup_deadline())
                        .await
                        .unwrap();
                    assert_eq!(result, Err(DuplexRuntimeError::StartFailed));
                    assert_eq!(
                        later_effects, 0,
                        "a sidecar start completing at expiry cannot admit readiness or direction effects"
                    );
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_readiness_finishing_at_expiry_does_not_prepare_register_or_activate() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    effects.delay_next_wait_ready(Duration::from_millis(100));
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = coordinator(effects.clone(), lifecycle);
                    let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
                    let result = coordinator.start_resources(deadline).await;
                    let events = effects.events();
                    let direction_effects = events
                        .iter()
                        .filter(|event| {
                            matches!(
                                event,
                                ScriptEvent::Acquire(..)
                                    | ScriptEvent::RegisterPoll
                                    | ScriptEvent::Activate(..)
                            )
                        })
                        .count();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(result, Err(DuplexRuntimeError::StartFailed));
                    assert_eq!(
                        direction_effects, 0,
                        "readiness completing at expiry cannot admit prepare, register, or activate"
                    );
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_prepare_finishing_at_expiry_retains_owner_without_register_or_activate()
        {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    effects
                        .delay_next_prepare(AudioDirection::Microphone, Duration::from_millis(100));
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = coordinator(effects.clone(), lifecycle);
                    coordinator.desired.speaker = None;
                    let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
                    let result = coordinator.start_resources(deadline).await;
                    let acquired = effects.provider_acquisitions(AudioDirection::Microphone);
                    let registered = coordinator.supervisor.active_sessions().len();
                    let activated = effects
                        .events()
                        .iter()
                        .filter(|event| matches!(event, ScriptEvent::Activate(..)))
                        .count();
                    let retained = coordinator.has_cleanup_owner(AudioDirection::Microphone);
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(result, Err(DuplexRuntimeError::StopFailed));
                    assert_eq!(
                        acquired, 1,
                        "the partial owner establishes the real prepare seam"
                    );
                    assert_eq!(registered, 0);
                    assert_eq!(activated, 0);
                    assert!(retained, "the exact prepared cell remains cleanup-owned");
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread")]
        async fn source_p1_registration_finishing_at_expiry_does_not_activate_a_worker() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(2), async {
                    let effects = ScriptedEffects::default();
                    effects.delay_next_register_poll(Duration::from_millis(150));
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = coordinator(effects.clone(), lifecycle);
                    coordinator.desired.speaker = None;
                    let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
                    let result = coordinator.start_resources(deadline).await;
                    let polls = effects.event_count(&ScriptEvent::RegisterPoll);
                    let activated = effects
                        .events()
                        .iter()
                        .filter(|event| matches!(event, ScriptEvent::Activate(..)))
                        .count();
                    let retained = coordinator.has_cleanup_owner(AudioDirection::Microphone);
                    let registered = coordinator.supervisor.active_sessions().len();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(result, Err(DuplexRuntimeError::StopFailed));
                    assert_eq!(
                        polls, 1,
                        "the delayed poll must exercise actual registration"
                    );
                    assert_eq!(
                        activated, 0,
                        "registration completing after its admission deadline cannot activate"
                    );
                    assert!(retained, "the registered direction owner remains retryable");
                    assert_eq!(
                        registered, 1,
                        "the crossed registration effect must stay owned until cleanup"
                    );
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("the synchronous registration boundary fixture must stay bounded");
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_expired_activation_does_not_spawn_or_poll_a_worker() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = coordinator(effects.clone(), lifecycle);
                    coordinator.supervisor.start().await.unwrap();
                    let launch = coordinator.desired.microphone.clone().unwrap();
                    let batch = coordinator
                        .prepare_batch(vec![launch], transaction_deadline())
                        .await
                        .unwrap_or_else(|_| {
                            panic!("the activation fixture must prepare one owner")
                        });
                    effects.clear_events();
                    let activation = coordinator
                        .activate_batch(batch, tokio::time::Instant::now())
                        .await;
                    let activated = effects
                        .events()
                        .iter()
                        .filter(|event| matches!(event, ScriptEvent::Activate(..)))
                        .count();
                    let task_count = coordinator.task_count();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(activation, Err(DuplexRuntimeError::StartFailed));
                    assert_eq!(activated, 0);
                    assert_eq!(
                        task_count, 0,
                        "expired activation cannot spawn a worker task"
                    );
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_expired_ambiguous_fault_does_not_poll_a_generation_probe() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    effects.clear_events();
                    let scope = coordinator
                        .resolve_fault_scope(
                            FaultScope::ProviderConnection,
                            tokio::time::Instant::now(),
                        )
                        .await;
                    let probe_calls = effects.probe_calls();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(scope, Err(DuplexRuntimeError::StopFailed));
                    assert_eq!(
                        probe_calls, 0,
                        "an expired ambiguous fault cannot poll a probe"
                    );
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_probe_completion_at_deadline_does_not_start_shared_recovery() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let microphone_id = effects.last_acquisition_id(AudioDirection::Microphone);
                    let speaker_id = effects.last_acquisition_id(AudioDirection::Speaker);
                    let peer_epoch = coordinator.worker_epoch(AudioDirection::Speaker);
                    let direction_epochs = coordinator.direction_epochs.clone();
                    let status_count = lifecycle.statuses.lock().unwrap().len();
                    effects.clear_events();
                    effects.fail_next_probe();
                    effects.delay_next_probe(Duration::from_millis(100));
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::ProviderConnection),
                    );
                    let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
                    let recovery = tokio::task::spawn_local(async move {
                        let result = coordinator.handle_next_worker_completion(deadline).await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.probe_calls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(effects.probe_calls(), 1);
                    tokio::time::advance(Duration::from_millis(100)).await;
                    settle_ready_tasks().await;
                    let (mut coordinator, result) = finish_recovery(recovery).await;
                    let before_cleanup = effects.events();
                    let retry_count = coordinator.shared_faults;
                    let epochs_after = coordinator.direction_epochs.clone();
                    let peer_epoch_after = coordinator.worker_epoch(AudioDirection::Speaker);
                    let exact_owners_retained = [microphone_id, speaker_id].into_iter().all(|id| {
                        [Resource::Provider, Resource::Capture, Resource::Playback]
                            .into_iter()
                            .all(|resource| effects.is_live(id, resource))
                    });
                    let status_count_after = lifecycle.statuses.lock().unwrap().len();
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    let after_cleanup = effects.events();

                    assert_eq!(result, Err(DuplexRuntimeError::StopFailed));
                    assert_eq!(retry_count, 0);
                    assert_eq!(epochs_after, direction_epochs);
                    assert_eq!(peer_epoch_after, peer_epoch);
                    assert_eq!(status_count_after, status_count);
                    assert_eq!(
                        before_cleanup
                            .iter()
                            .filter(|event| matches!(event, ScriptEvent::RecoveryAttempt(_)))
                            .count(),
                        0
                    );
                    assert_eq!(
                        before_cleanup
                            .iter()
                            .filter(|event| matches!(event, ScriptEvent::Stop(..)))
                            .count(),
                        0
                    );
                    assert!(!before_cleanup.iter().any(|event| matches!(
                        event,
                        ScriptEvent::GenerationKill | ScriptEvent::GenerationStart
                    )));
                    assert!(exact_owners_retained);
                    for (direction, id) in [
                        (AudioDirection::Microphone, microphone_id),
                        (AudioDirection::Speaker, speaker_id),
                    ] {
                        for resource in [Resource::Provider, Resource::Capture, Resource::Playback]
                        {
                            assert_eq!(
                                after_cleanup
                                    .iter()
                                    .filter(|event| {
                                        **event == ScriptEvent::Stop(id, direction, resource)
                                    })
                                    .count(),
                                1
                            );
                        }
                    }
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn source_p1_expired_shared_restart_has_zero_recovery_side_effects() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let ids = [
                        (
                            AudioDirection::Microphone,
                            effects.last_acquisition_id(AudioDirection::Microphone),
                        ),
                        (
                            AudioDirection::Speaker,
                            effects.last_acquisition_id(AudioDirection::Speaker),
                        ),
                    ];
                    let direction_epochs = coordinator.direction_epochs.clone();
                    let worker_epochs = coordinator.worker_epochs();
                    let status_count = lifecycle.statuses.lock().unwrap().len();
                    effects.clear_events();
                    let result = coordinator
                        .restart_shared(tokio::time::Instant::now())
                        .await;
                    let before_cleanup = effects.events();
                    let retry_count = coordinator.shared_faults;
                    let epochs_after = coordinator.direction_epochs.clone();
                    let worker_epochs_after = coordinator.worker_epochs();
                    let status_count_after = lifecycle.statuses.lock().unwrap().len();
                    let exact_owners_retained = ids.iter().all(|(_, id)| {
                        [Resource::Provider, Resource::Capture, Resource::Playback]
                            .into_iter()
                            .all(|resource| effects.is_live(*id, resource))
                    });
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    let after_cleanup = effects.events();

                    assert_eq!(result, Err(DuplexRuntimeError::StopFailed));
                    assert_eq!(effects.probe_calls(), 0);
                    assert_eq!(retry_count, 0);
                    assert_eq!(epochs_after, direction_epochs);
                    assert_eq!(worker_epochs_after, worker_epochs);
                    assert_eq!(status_count_after, status_count);
                    assert!(!before_cleanup.iter().any(|event| matches!(
                        event,
                        ScriptEvent::RecoveryAttempt(_)
                            | ScriptEvent::GenerationKill
                            | ScriptEvent::GenerationStart
                            | ScriptEvent::Stop(..)
                            | ScriptEvent::Acquire(..)
                    )));
                    assert!(exact_owners_retained);
                    for (direction, id) in ids {
                        for resource in [Resource::Provider, Resource::Capture, Resource::Playback]
                        {
                            assert_eq!(
                                after_cleanup
                                    .iter()
                                    .filter(|event| {
                                        **event == ScriptEvent::Stop(id, direction, resource)
                                    })
                                    .count(),
                                1
                            );
                        }
                    }
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn partial_start_rollback_uses_one_absolute_transaction_deadline() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    effects.fail_prepare(AudioDirection::Speaker, PrepareStage::Playback);
                    effects.stall_cleanup(AudioDirection::Speaker, Resource::Capture);
                    let mut coordinator = coordinator(effects.clone(), lifecycle);
                    let admitted = tokio::time::Instant::now();
                    let deadline = admitted + DIRECTION_CLEANUP_BUDGET;
                    let attempt = tokio::task::spawn_local(async move {
                        let result = coordinator.start_resources(deadline).await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.active_cleanup_stalls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(effects.active_cleanup_stalls(), 1);
                    tokio::time::advance(Duration::from_millis(3_999)).await;
                    settle_ready_tasks().await;
                    assert!(!attempt.is_finished());
                    tokio::time::advance(Duration::from_millis(1)).await;
                    settle_ready_tasks().await;
                    let finished_at_deadline = attempt.is_finished();
                    if !finished_at_deadline {
                        effects.release_cleanup();
                        settle_ready_tasks().await;
                    }
                    let (mut coordinator, result) = attempt.await.unwrap();
                    let retained = coordinator.has_cleanup_owner(AudioDirection::Speaker);
                    let deadlines = effects.deadlines();
                    let one_deadline = !deadlines.is_empty()
                        && deadlines.iter().all(|(_, _, deadline)| {
                            *deadline == admitted + DIRECTION_CLEANUP_BUDGET
                        });
                    let reached_prepare = deadlines.iter().any(|(direction, stage, _)| {
                        *direction == AudioDirection::Speaker && *stage == DeadlineStage::Prepare
                    });
                    let reached_rollback = deadlines.iter().any(|(direction, stage, _)| {
                        *direction == AudioDirection::Speaker && *stage == DeadlineStage::StopPcm
                    });
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert!(
                        finished_at_deadline,
                        "start rollback must stop at one 4s deadline"
                    );
                    assert_eq!(result, Err(DuplexRuntimeError::StopFailed));
                    assert!(
                        retained,
                        "timeout must retain the exact partial-start owner"
                    );
                    assert!(
                        one_deadline,
                        "prepare and rollback must share the admitted deadline"
                    );
                    assert!(reached_prepare && reached_rollback);
                    assert_eq!(DIRECTION_CLEANUP_BUDGET, Duration::from_secs(4));
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn shared_recovery_pcm_cleanup_uses_one_absolute_transaction_deadline() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    effects.clear_events();
                    effects.stall_cleanup(AudioDirection::Microphone, Resource::Capture);
                    effects.trigger(
                        AudioDirection::Microphone,
                        Trigger::Fault(DirectionFailureOrigin::WatchdogRestart),
                    );
                    let admitted = tokio::time::Instant::now();
                    let deadline = admitted + DIRECTION_CLEANUP_BUDGET;
                    let attempt = tokio::task::spawn_local(async move {
                        let result = coordinator.handle_next_worker_completion(deadline).await;
                        (coordinator, result)
                    });
                    for _ in 0..64 {
                        if effects.active_cleanup_stalls() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(effects.active_cleanup_stalls(), 1);
                    tokio::time::advance(Duration::from_secs(4)).await;
                    settle_ready_tasks().await;
                    let finished_at_deadline = attempt.is_finished();
                    let generation_effects_before_release = effects
                        .events()
                        .into_iter()
                        .filter(|event| {
                            matches!(
                                event,
                                ScriptEvent::GenerationKill | ScriptEvent::GenerationStart
                            )
                        })
                        .count();
                    if !finished_at_deadline {
                        effects.release_cleanup();
                        tokio::time::advance(Duration::from_millis(50)).await;
                        settle_ready_tasks().await;
                    }
                    let (mut coordinator, result) = attempt.await.unwrap();
                    let retained = coordinator.has_cleanup_owner(AudioDirection::Microphone);
                    let deadlines = effects.deadlines();
                    let one_deadline = !deadlines.is_empty()
                        && deadlines.iter().all(|(_, _, deadline)| {
                            *deadline == admitted + DIRECTION_CLEANUP_BUDGET
                        });
                    let reached_pcm = deadlines.iter().any(|(direction, stage, _)| {
                        *direction == AudioDirection::Microphone && *stage == DeadlineStage::StopPcm
                    });
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert!(
                        finished_at_deadline,
                        "shared cleanup must stop at one 4s deadline"
                    );
                    assert_eq!(result, Err(DuplexRuntimeError::StopFailed));
                    assert_eq!(generation_effects_before_release, 0);
                    assert!(retained, "timeout must retain the exact PCM owner");
                    assert!(
                        one_deadline,
                        "shared cleanup must keep one transaction deadline"
                    );
                    assert!(reached_pcm);
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread", start_paused = true)]
        async fn runtime_cleanup_shares_one_admission_anchored_deadline_across_workers() {
            tokio::task::LocalSet::new()
                .run_until(async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let delayed_direction = *coordinator.workers.keys().last().unwrap();
                    effects.clear_events();
                    effects.delay_cleanup(
                        delayed_direction,
                        Resource::Capture,
                        Duration::from_secs(9),
                    );
                    let admitted = tokio::time::Instant::now();
                    let deadline = admitted + RUNTIME_CLEANUP_BUDGET;
                    let attempt = tokio::task::spawn_local(async move {
                        let result = coordinator.shutdown(deadline).await;
                        (coordinator, result)
                    });
                    settle_ready_tasks().await;
                    tokio::time::advance(Duration::from_millis(7_999)).await;
                    settle_ready_tasks().await;
                    assert!(!attempt.is_finished());
                    tokio::time::advance(Duration::from_millis(1)).await;
                    settle_ready_tasks().await;
                    let finished_at_deadline = attempt.is_finished();
                    if !finished_at_deadline {
                        for _ in 0..16 {
                            tokio::time::advance(Duration::from_secs(1)).await;
                            settle_ready_tasks().await;
                            if attempt.is_finished() {
                                break;
                            }
                        }
                    }
                    if !attempt.is_finished() {
                        attempt.abort();
                        let _ = attempt.await;
                        panic!("the RED fail-safe must bound a broken cleanup implementation");
                    }
                    let (mut coordinator, result) = attempt.await.unwrap();
                    let retained = coordinator.has_pending_cleanup();
                    let deadlines = effects.deadlines();
                    let one_deadline = !deadlines.is_empty()
                        && deadlines
                            .iter()
                            .all(|(_, _, deadline)| *deadline == admitted + RUNTIME_CLEANUP_BUDGET);
                    let reached_both_pcm = [AudioDirection::Microphone, AudioDirection::Speaker]
                        .into_iter()
                        .all(|expected| {
                            deadlines.iter().any(|(direction, stage, _)| {
                                *direction == expected && *stage == DeadlineStage::StopPcm
                            })
                        });
                    let generation_shutdown_before_retry =
                        effects.event_count(&ScriptEvent::GenerationShutdown);
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert!(
                        finished_at_deadline,
                        "all cleanup phases must share one 8s deadline"
                    );
                    assert_eq!(result, Err(DuplexRuntimeError::StopFailed));
                    assert!(
                        retained,
                        "the unfinished direction must remain owned for retry"
                    );
                    assert!(
                        one_deadline,
                        "all runtime cleanup phases must share one deadline"
                    );
                    assert!(reached_both_pcm);
                    assert_eq!(
                        generation_shutdown_before_retry, 0,
                        "PCM consuming the admitted deadline cannot poll sidecar shutdown"
                    );
                    assert_eq!(
                        effects.event_count(&ScriptEvent::GenerationShutdown),
                        1,
                        "one fresh same-owner retry may shut down the sidecar after PCM reap"
                    );
                    assert_eq!(RUNTIME_CLEANUP_BUDGET, Duration::from_secs(8));
                    assert_eq!(effects.live_count(), 0);
                })
                .await;
        }

        #[tokio::test(flavor = "current_thread")]
        async fn worker_panic_preserves_cell_until_coordinator_compensation() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    effects.fail_cleanup(AudioDirection::Microphone, Resource::Capture);
                    effects.trigger(AudioDirection::Microphone, Trigger::Panic);

                    assert_eq!(
                        coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await,
                        Err(DuplexRuntimeError::StartFailed)
                    );
                    assert!(coordinator.has_cleanup_owner(AudioDirection::Microphone));
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("panic compensation must remain bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn simultaneous_real_direction_completions_are_joined_once_with_bounded_registry() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let mut coordinator = started(effects.clone(), lifecycle).await;
                    let old_ids = [
                        effects.last_acquisition_id(AudioDirection::Microphone),
                        effects.last_acquisition_id(AudioDirection::Speaker),
                    ];
                    effects.clear_events();
                    for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
                        effects.trigger(
                            direction,
                            Trigger::Fault(DirectionFailureOrigin::PcmCapture),
                        );
                    }
                    settle_ready_tasks().await;
                    assert!(coordinator.task_count() <= 2);

                    for _ in 0..2 {
                        coordinator
                            .handle_next_worker_completion(transaction_deadline())
                            .await
                            .unwrap();
                        assert!(coordinator.task_count() <= 2);
                    }
                    assert_eq!(coordinator.worker_epochs().len(), 2);
                    for id in old_ids {
                        for resource in [Resource::Provider, Resource::Capture, Resource::Playback]
                        {
                            assert_eq!(
                                effects
                                    .events()
                                    .iter()
                                    .filter(|event| matches!(
                                        event,
                                        ScriptEvent::Stop(observed_id, _, observed_resource)
                                            if *observed_id == id && *observed_resource == resource
                                    ))
                                    .count(),
                                1
                            );
                        }
                    }
                    coordinator.shutdown(cleanup_deadline()).await.unwrap();
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("both domain completions must drain without a lost JoinSet wake");
        }

        async fn command_result(
            receiver: &std_mpsc::Receiver<Result<(), DuplexRuntimeError>>,
        ) -> Result<(), DuplexRuntimeError> {
            tokio::time::timeout(Duration::from_secs(1), async {
                loop {
                    match receiver.try_recv() {
                        Ok(result) => return result,
                        Err(std_mpsc::TryRecvError::Empty) => tokio::task::yield_now().await,
                        Err(std_mpsc::TryRecvError::Disconnected) => {
                            return Err(DuplexRuntimeError::StopFailed);
                        }
                    }
                }
            })
            .await
            .expect("the scripted command must remain bounded")
        }

        #[tokio::test(flavor = "current_thread")]
        async fn queued_stop_before_completion_joins_each_owner_once() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let old_ids = [
                        effects.last_acquisition_id(AudioDirection::Microphone),
                        effects.last_acquisition_id(AudioDirection::Speaker),
                    ];
                    effects.clear_events();
                    let (_stop, stop_receiver) = watch::channel(None);
                    let (commands, command_receiver) = mpsc::channel(2);
                    let (response, result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Stop {
                            deadline: cleanup_deadline(),
                            response,
                        })
                        .await
                        .unwrap();
                    let driver = tokio::task::spawn_local(run_coordinator(
                        coordinator,
                        stop_receiver,
                        command_receiver,
                        Some((41, lifecycle.clone())),
                        None,
                    ));

                    command_result(&result).await.unwrap();
                    driver.await.unwrap().unwrap();
                    assert!(lifecycle.completions.lock().unwrap().is_empty());
                    for id in old_ids {
                        for resource in [Resource::Provider, Resource::Capture, Resource::Playback]
                        {
                            assert_eq!(
                                effects
                                    .events()
                                    .iter()
                                    .filter(|event| matches!(
                                        event,
                                        ScriptEvent::Stop(observed_id, _, observed_resource)
                                            if *observed_id == id && *observed_resource == resource
                                    ))
                                    .count(),
                                1
                            );
                        }
                    }
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("command-first Stop ordering must remain bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn terminal_cleanup_before_queued_stop_does_not_report_outer_completion_early() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let coordinator = started(effects.clone(), lifecycle.clone()).await;
                    let old_ids = [
                        effects.last_acquisition_id(AudioDirection::Microphone),
                        effects.last_acquisition_id(AudioDirection::Speaker),
                    ];
                    effects.clear_events();
                    effects.trigger(AudioDirection::Microphone, Trigger::Panic);
                    settle_ready_tasks().await;

                    let (_stop, stop_receiver) = watch::channel(None);
                    let (commands, command_receiver) = mpsc::channel(2);
                    let (response, result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Stop {
                            deadline: cleanup_deadline(),
                            response,
                        })
                        .await
                        .unwrap();
                    let driver = tokio::task::spawn_local(run_coordinator(
                        coordinator,
                        stop_receiver,
                        command_receiver,
                        Some((41, lifecycle.clone())),
                        None,
                    ));

                    assert_eq!(driver.await.unwrap(), Err(DuplexRuntimeError::StartFailed));
                    assert_eq!(
                        command_result(&result).await,
                        Err(DuplexRuntimeError::StopFailed)
                    );
                    assert!(
                        lifecycle.completions.lock().unwrap().is_empty(),
                        "only the outer runtime thread may publish final completion after exit"
                    );
                    assert_eq!(
                        lifecycle.cleanup_started.lock().unwrap().as_slice(),
                        [41],
                        "terminal compensation must publish one distinct cleanup-started fact"
                    );
                    for id in old_ids {
                        for resource in [Resource::Provider, Resource::Capture, Resource::Playback]
                        {
                            assert_eq!(
                                effects
                                    .events()
                                    .iter()
                                    .filter(|event| matches!(
                                        event,
                                        ScriptEvent::Stop(observed_id, _, observed_resource)
                                            if *observed_id == id && *observed_resource == resource
                                    ))
                                    .count(),
                                1
                            );
                        }
                    }
                    assert_eq!(effects.live_count(), 0);
                }))
                .await
                .expect("completion-first terminal/Stop ordering must remain bounded");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn failed_stop_enters_cleanup_only_before_any_more_runtime_work() {
            tokio::task::LocalSet::new()
                .run_until(tokio::time::timeout(Duration::from_secs(3), async {
                    let effects = ScriptedEffects::default();
                    let lifecycle = Arc::new(LifecycleRecorder::default());
                    let coordinator = started(effects.clone(), lifecycle).await;
                    let microphone_id = effects.last_acquisition_id(AudioDirection::Microphone);
                    effects.fail_cleanup(AudioDirection::Microphone, Resource::Capture);
                    effects.fail_cleanup(AudioDirection::Microphone, Resource::Capture);
                    let (stop, stop_receiver) = watch::channel(None);
                    let (commands, command_receiver) = mpsc::channel(2);
                    let driver = tokio::task::spawn_local(run_coordinator(
                        coordinator,
                        stop_receiver,
                        command_receiver,
                        None,
                        None,
                    ));

                    let (response, result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Stop {
                            deadline: cleanup_deadline(),
                            response,
                        })
                        .await
                        .unwrap();
                    assert_eq!(
                        command_result(&result).await,
                        Err(DuplexRuntimeError::StopFailed)
                    );
                    let event_count = effects.events().len();

                    let (response, result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Reconfigure {
                            launch: replacement(launch_pair()),
                            deadline: transaction_deadline(),
                            response,
                        })
                        .await
                        .unwrap();
                    assert_eq!(
                        command_result(&result).await,
                        Err(DuplexRuntimeError::RestoreFailed)
                    );
                    assert_eq!(effects.events().len(), event_count);

                    let (response, result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Stop {
                            deadline: cleanup_deadline(),
                            response,
                        })
                        .await
                        .unwrap();
                    assert_eq!(
                        command_result(&result).await,
                        Err(DuplexRuntimeError::StopFailed)
                    );
                    assert!(
                        !effects.events()[event_count..].iter().any(|event| matches!(
                            event,
                            ScriptEvent::Acquire(..)
                                | ScriptEvent::Activate(..)
                                | ScriptEvent::GenerationKill
                                | ScriptEvent::GenerationStart
                                | ScriptEvent::RecoveryAttempt(_)
                        ))
                    );

                    let (response, result) = std_mpsc::sync_channel(1);
                    commands
                        .send(RuntimeCommand::Stop {
                            deadline: cleanup_deadline(),
                            response,
                        })
                        .await
                        .unwrap();
                    command_result(&result).await.unwrap();
                    assert_eq!(driver.await.unwrap(), Err(DuplexRuntimeError::StopFailed));
                    assert_eq!(
                        effects
                            .events()
                            .iter()
                            .filter_map(|event| match event {
                                ScriptEvent::Stop(
                                    id,
                                    AudioDirection::Microphone,
                                    Resource::Capture,
                                ) => Some(*id),
                                _ => None,
                            })
                            .collect::<Vec<_>>(),
                        [microphone_id, microphone_id, microphone_id]
                    );
                    assert_eq!(effects.live_count(), 0);
                    drop(stop);
                }))
                .await
                .expect("cleanup-only command handling must remain bounded");
        }
    }

    #[test]
    fn direction_fault_backoff_is_exact_and_bounded() {
        let mut faults = HashMap::new();

        assert_eq!(
            next_fault_delay(&mut faults, AudioDirection::Microphone),
            Some(Duration::from_millis(50))
        );
        assert_eq!(
            next_fault_delay(&mut faults, AudioDirection::Microphone),
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            next_fault_delay(&mut faults, AudioDirection::Microphone),
            Some(Duration::from_millis(200))
        );
        assert_eq!(
            next_fault_delay(&mut faults, AudioDirection::Microphone),
            None
        );
        assert_eq!(
            next_fault_delay(&mut faults, AudioDirection::Speaker),
            Some(Duration::from_millis(50))
        );
    }
}
