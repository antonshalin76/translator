use std::{
    collections::{HashMap, VecDeque},
    fs,
    num::NonZeroU32,
    os::unix::fs::MetadataExt,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::Duration,
};

use axum::http::StatusCode;
use rustix::time::{ClockId, clock_gettime};
use serde::Serialize;
use thiserror::Error;
use tokio::sync::watch;
use translator_audio::{
    AEC_SINK, AEC_SOURCE, AecCapability, BoundedPcmQueue, CaptureEvent, GraphHealth, MIC_OUT_SINK,
    OutputMode, PulsePcmCapture, PulsePcmCommand, PulsePcmPlayback, REMOTE_IN_SINK,
    SpeechSegmenter, WebRtcVoiceDetector,
};
use translator_core::{AudioDirection, ProviderId, TranslationMode};
use translator_ipc::{
    ProviderStreamClient,
    provider::{CloseRequestReason, ProviderState, provider_event},
    wait_provider_ready,
};

use crate::{
    AudioOperationGate, AudioOperationLease, CLOSE_ACK_TIMEOUT, ControlFailure, DirectionEffect,
    DirectionRuntimeConfig, DirectionSession, DirectionWatchdogEffect, LatencySample,
    ProcessSidecarRuntime, RuntimeSnapshot, RuntimeStore, SafeProviderErrorCode, SidecarSupervisor,
    TerminalOutcome, TranslationController,
};

const PROVIDER_READY_TIMEOUT: Duration = Duration::from_secs(120);
const START_ACK_TIMEOUT: Duration = Duration::from_secs(130);
const STOP_TIMEOUT: Duration = Duration::from_secs(10);
const DIRECTION_OPEN_TIMEOUT: Duration = Duration::from_secs(5);
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
}

pub trait ActiveDuplexRuntime: Send {
    fn stop(&mut self) -> Result<(), DuplexRuntimeError>;
}

pub trait DuplexRunner: Send + Sync {
    fn start(
        &self,
        snapshot: RuntimeSnapshot,
    ) -> Result<Box<dyn ActiveDuplexRuntime>, DuplexRuntimeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplexRuntimeEvent {
    SpeechStarted {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        capture_monotonic_ns: u64,
    },
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
        observed_monotonic_ns: u64,
        queue_lag_ms: u32,
    },
    FirstAudioExpired {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        observed_monotonic_ns: u64,
    },
    ProviderLatency {
        direction: AudioDirection,
        utterance_id: Option<uuid::Uuid>,
        tts_first_audio_ms: Option<u32>,
        provider_total_ms: Option<u32>,
    },
    ProviderError {
        direction: AudioDirection,
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
    #[serde(flatten)]
    payload: Task7BridgeEventPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum Task7BridgeEventPayload {
    Ready {
        pid: u32,
        monotonic_ns: u64,
    },
    SpeechStarted {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        monotonic_ns: u64,
    },
    AsrFinal {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        monotonic_ns: u64,
    },
    TranslationFinal {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        monotonic_ns: u64,
    },
    AudioFrame {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        sequence: u64,
        provider_monotonic_ns: u64,
        monotonic_ns: u64,
        queue_lag_ms: u32,
    },
    FirstAudioExpired {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        monotonic_ns: u64,
    },
    ProviderLatency {
        direction: AudioDirection,
        #[serde(skip_serializing_if = "Option::is_none")]
        utterance_id: Option<uuid::Uuid>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tts_first_audio_ms: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_total_ms: Option<u32>,
        monotonic_ns: u64,
    },
    ProviderError {
        direction: AudioDirection,
        #[serde(skip_serializing_if = "Option::is_none")]
        utterance_id: Option<uuid::Uuid>,
        code: SafeProviderErrorCode,
        retryable: bool,
        monotonic_ns: u64,
    },
    UtteranceTerminalOutcome {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        outcome: TerminalOutcome,
        monotonic_ns: u64,
    },
    UtteranceTerminal {
        direction: AudioDirection,
        utterance_id: uuid::Uuid,
        monotonic_ns: u64,
    },
    GenerationRestart {
        attempt: NonZeroU32,
        monotonic_ns: u64,
    },
    Stopped {
        monotonic_ns: u64,
    },
    Failure {
        stage: Task7BridgeFailureStage,
        code: &'static str,
        monotonic_ns: u64,
    },
}

impl Task7BridgeEvent {
    pub fn ready(pid: u32) -> Self {
        Self::new(Task7BridgeEventPayload::Ready {
            pid,
            monotonic_ns: monotonic_ns(),
        })
    }

    pub fn from_runtime(event: DuplexRuntimeEvent) -> Self {
        let payload = match event {
            DuplexRuntimeEvent::SpeechStarted {
                direction,
                utterance_id,
                capture_monotonic_ns,
            } => Task7BridgeEventPayload::SpeechStarted {
                direction,
                utterance_id,
                monotonic_ns: capture_monotonic_ns,
            },
            DuplexRuntimeEvent::TranscriptFinal {
                direction,
                utterance_id,
            } => Task7BridgeEventPayload::AsrFinal {
                direction,
                utterance_id,
                monotonic_ns: monotonic_ns(),
            },
            DuplexRuntimeEvent::TranslationFinal {
                direction,
                utterance_id,
            } => Task7BridgeEventPayload::TranslationFinal {
                direction,
                utterance_id,
                monotonic_ns: monotonic_ns(),
            },
            DuplexRuntimeEvent::AudioFrame {
                direction,
                utterance_id,
                sequence,
                provider_monotonic_ns,
                observed_monotonic_ns,
                queue_lag_ms,
            } => Task7BridgeEventPayload::AudioFrame {
                direction,
                utterance_id,
                sequence,
                provider_monotonic_ns,
                monotonic_ns: observed_monotonic_ns,
                queue_lag_ms,
            },
            DuplexRuntimeEvent::FirstAudioExpired {
                direction,
                utterance_id,
                observed_monotonic_ns,
            } => Task7BridgeEventPayload::FirstAudioExpired {
                direction,
                utterance_id,
                monotonic_ns: observed_monotonic_ns,
            },
            DuplexRuntimeEvent::ProviderLatency {
                direction,
                utterance_id,
                tts_first_audio_ms,
                provider_total_ms,
            } => Task7BridgeEventPayload::ProviderLatency {
                direction,
                utterance_id,
                tts_first_audio_ms,
                provider_total_ms,
                monotonic_ns: monotonic_ns(),
            },
            DuplexRuntimeEvent::ProviderError {
                direction,
                utterance_id,
                code,
                retryable,
            } => Task7BridgeEventPayload::ProviderError {
                direction,
                utterance_id,
                code,
                retryable,
                monotonic_ns: monotonic_ns(),
            },
            DuplexRuntimeEvent::UtteranceTerminalOutcome {
                direction,
                utterance_id,
                outcome,
            } => Task7BridgeEventPayload::UtteranceTerminalOutcome {
                direction,
                utterance_id,
                outcome,
                monotonic_ns: monotonic_ns(),
            },
            DuplexRuntimeEvent::UtteranceTerminal {
                direction,
                utterance_id,
            } => Task7BridgeEventPayload::UtteranceTerminal {
                direction,
                utterance_id,
                monotonic_ns: monotonic_ns(),
            },
            DuplexRuntimeEvent::GenerationRestart { attempt } => {
                Task7BridgeEventPayload::GenerationRestart {
                    attempt,
                    monotonic_ns: monotonic_ns(),
                }
            }
        };
        Self::new(payload)
    }

    pub fn stopped() -> Self {
        Self::new(Task7BridgeEventPayload::Stopped {
            monotonic_ns: monotonic_ns(),
        })
    }

    pub fn failure(stage: Task7BridgeFailureStage, code: &'static str) -> Self {
        Self::new(Task7BridgeEventPayload::Failure {
            stage,
            code,
            monotonic_ns: monotonic_ns(),
        })
    }

    fn new(payload: Task7BridgeEventPayload) -> Self {
        Self {
            schema_version: TASK7_BRIDGE_SCHEMA_VERSION,
            payload,
        }
    }
}

pub trait DuplexRuntimeObserver: Send + Sync {
    fn observe(&self, event: DuplexRuntimeEvent);

    fn requested_mode(&self, _direction: AudioDirection) -> Option<TranslationMode> {
        None
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
                utterances.insert(
                    (direction, utterance_id),
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

    pub fn start_with_audio_targets(
        &self,
        snapshot: RuntimeSnapshot,
        targets: DuplexAudioTargets,
    ) -> Result<Box<dyn ActiveDuplexRuntime>, DuplexRuntimeError> {
        let launch = DuplexLaunch::try_with_audio_targets(snapshot, targets)?;
        self.start_launch(launch)
    }

    fn start_launch(
        &self,
        launch: DuplexLaunch,
    ) -> Result<Box<dyn ActiveDuplexRuntime>, DuplexRuntimeError> {
        if launch.bypass_only() {
            return Ok(Box::new(BypassActiveDuplex));
        }
        let config = self.config.clone();
        let observer = self.observer.clone();
        let (stop_sender, stop_receiver) = watch::channel(false);
        let (ack_sender, ack_receiver) = std_mpsc::sync_channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("translator-duplex-runtime".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build();
                let result = match runtime {
                    Ok(runtime) => runtime.block_on(run_process_duplex(
                        config,
                        launch,
                        stop_receiver,
                        ack_sender,
                        observer,
                    )),
                    Err(_) => {
                        let _ = ack_sender.send(Err(DuplexRuntimeError::StartFailed));
                        Err(DuplexRuntimeError::StartFailed)
                    }
                };
                let _ = done_sender.send(result);
            })
            .map_err(|_| DuplexRuntimeError::StartFailed)?;
        match ack_receiver.recv_timeout(START_ACK_TIMEOUT) {
            Ok(Ok(())) => Ok(Box::new(ProcessActiveDuplex {
                stop_sender,
                done_receiver,
                thread: Some(thread),
            })),
            Ok(Err(error)) => {
                let _ = stop_sender.send(true);
                let _ = thread.join();
                Err(error)
            }
            Err(_) => {
                let _ = stop_sender.send(true);
                let _ = thread.join();
                Err(DuplexRuntimeError::StartFailed)
            }
        }
    }
}

impl DuplexRunner for ProcessDuplexRunner {
    fn start(
        &self,
        snapshot: RuntimeSnapshot,
    ) -> Result<Box<dyn ActiveDuplexRuntime>, DuplexRuntimeError> {
        let launch = DuplexLaunch::try_from(snapshot)?;
        self.start_launch(launch)
    }
}

struct ProcessActiveDuplex {
    stop_sender: watch::Sender<bool>,
    done_receiver: std_mpsc::Receiver<Result<(), DuplexRuntimeError>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl ActiveDuplexRuntime for ProcessActiveDuplex {
    fn stop(&mut self) -> Result<(), DuplexRuntimeError> {
        let _ = self.stop_sender.send(true);
        let result = self
            .done_receiver
            .recv_timeout(STOP_TIMEOUT)
            .map_err(|_| DuplexRuntimeError::StopFailed)?;
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| DuplexRuntimeError::StopFailed)?;
        }
        result
    }
}

struct BypassActiveDuplex;

impl ActiveDuplexRuntime for BypassActiveDuplex {
    fn stop(&mut self) -> Result<(), DuplexRuntimeError> {
        Ok(())
    }
}

impl Drop for ProcessActiveDuplex {
    fn drop(&mut self) {
        let _ = self.stop_sender.send(true);
    }
}

pub struct DuplexRuntimeHandle {
    runner: Arc<dyn DuplexRunner>,
    gate: AudioOperationGate,
    active: Mutex<Option<ActiveProductionRuntime>>,
}

struct ActiveProductionRuntime {
    runtime: Box<dyn ActiveDuplexRuntime>,
    _lease: AudioOperationLease,
}

impl DuplexRuntimeHandle {
    pub fn with_runner(runner: Arc<dyn DuplexRunner>) -> Self {
        Self::with_runner_and_gate(runner, AudioOperationGate::new())
    }

    pub fn with_runner_and_gate(runner: Arc<dyn DuplexRunner>, gate: AudioOperationGate) -> Self {
        Self {
            runner,
            gate,
            active: Mutex::new(None),
        }
    }

    fn stop_active(&self) -> Result<(), DuplexRuntimeError> {
        let mut state = self.active.lock().expect("duplex runtime mutex poisoned");
        let mut active = state.take().ok_or(DuplexRuntimeError::StopFailed)?;
        active.runtime.stop()
    }
}

impl TranslationController for DuplexRuntimeHandle {
    fn start(&self, snapshot: RuntimeSnapshot) -> Result<(), ControlFailure> {
        let mut state = self.active.lock().expect("duplex runtime mutex poisoned");
        if state.is_some() {
            return Err(ControlFailure {
                status: StatusCode::CONFLICT,
                code: "translation_already_running",
            });
        }
        let lease = self.gate.acquire_production().map_err(|_| ControlFailure {
            status: StatusCode::CONFLICT,
            code: "audio_operation_busy",
        })?;
        let active = self
            .runner
            .start(snapshot)
            .map_err(|error| ControlFailure {
                status: match error {
                    DuplexRuntimeError::InvalidConfiguration => StatusCode::CONFLICT,
                    DuplexRuntimeError::StartFailed | DuplexRuntimeError::StopFailed => {
                        StatusCode::SERVICE_UNAVAILABLE
                    }
                },
                code: match error {
                    DuplexRuntimeError::InvalidConfiguration => "translation_precondition_failed",
                    DuplexRuntimeError::StartFailed | DuplexRuntimeError::StopFailed => {
                        "translation_start_failed"
                    }
                },
            })?;
        *state = Some(ActiveProductionRuntime {
            runtime: active,
            _lease: lease,
        });
        Ok(())
    }

    fn stop(&self) -> Result<(), ControlFailure> {
        self.stop_active().map_err(|_| ControlFailure {
            status: StatusCode::CONFLICT,
            code: "translation_not_running",
        })
    }
}

impl Drop for DuplexRuntimeHandle {
    fn drop(&mut self) {
        if let Ok(state) = self.active.get_mut()
            && let Some(mut active) = state.take()
        {
            let _ = active.runtime.stop();
        }
    }
}

#[derive(Clone)]
struct DirectionLaunch {
    runtime: DirectionRuntimeConfig,
    capture_device: String,
    playback_device: String,
    capture_stream_name: &'static str,
    playback_stream_name: &'static str,
}

struct DuplexLaunch {
    microphone: Option<DirectionLaunch>,
    speaker: Option<DirectionLaunch>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplexAudioTargets {
    pub microphone_capture: String,
    pub microphone_playback: String,
    pub speaker_capture: String,
    pub speaker_playback: String,
}

pub fn resolve_duplex_audio_targets(
    snapshot: &RuntimeSnapshot,
) -> Result<DuplexAudioTargets, DuplexRuntimeError> {
    let devices = snapshot
        .devices
        .as_ref()
        .ok_or(DuplexRuntimeError::InvalidConfiguration)?;
    if !devices.acoustic.full_duplex_allowed {
        return Err(DuplexRuntimeError::InvalidConfiguration);
    }
    let source = devices
        .source
        .selected
        .as_ref()
        .filter(|device| device.available)
        .ok_or(DuplexRuntimeError::InvalidConfiguration)?
        .name
        .clone();
    let sink = devices
        .sink
        .selected
        .as_ref()
        .filter(|device| device.available)
        .ok_or(DuplexRuntimeError::InvalidConfiguration)?
        .name
        .clone();
    let (microphone_capture, speaker_playback) = match &devices.acoustic.mode {
        OutputMode::Headphones
            if matches!(
                &devices.acoustic.aec_capability,
                AecCapability::ValidatedFor {
                    source_name,
                    sink_name,
                } if source_name == &source && sink_name == &sink
            ) =>
        {
            (AEC_SOURCE.to_owned(), AEC_SINK.to_owned())
        }
        OutputMode::Headphones => (source, sink),
        OutputMode::OpenSpeaker
            if matches!(
                &devices.acoustic.aec_capability,
                AecCapability::ValidatedFor {
                    source_name,
                    sink_name,
                } if source_name == &source && sink_name == &sink
            ) =>
        {
            (AEC_SOURCE.to_owned(), AEC_SINK.to_owned())
        }
        OutputMode::OpenSpeaker | OutputMode::UnknownUnsafe => {
            return Err(DuplexRuntimeError::InvalidConfiguration);
        }
    };
    Ok(DuplexAudioTargets {
        microphone_capture,
        microphone_playback: MIC_OUT_SINK.to_owned(),
        speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
        speaker_playback,
    })
}

impl TryFrom<RuntimeSnapshot> for DuplexLaunch {
    type Error = DuplexRuntimeError;

    fn try_from(snapshot: RuntimeSnapshot) -> Result<Self, Self::Error> {
        validate_launch_snapshot(&snapshot)?;
        let targets = resolve_duplex_audio_targets(&snapshot)?;
        Self::try_with_audio_targets(snapshot, targets)
    }
}

impl DuplexLaunch {
    fn try_with_audio_targets(
        snapshot: RuntimeSnapshot,
        targets: DuplexAudioTargets,
    ) -> Result<Self, DuplexRuntimeError> {
        validate_launch_snapshot(&snapshot)?;
        validate_explicit_audio_targets(&targets)?;
        let microphone = if direction_enabled(&snapshot, AudioDirection::Microphone)? {
            Some(direction_launch(
                &snapshot,
                AudioDirection::Microphone,
                targets.microphone_capture,
                targets.microphone_playback,
                "translator-outgoing-capture",
                "translator-outgoing-playback",
            )?)
        } else {
            None
        };
        let speaker = if direction_enabled(&snapshot, AudioDirection::Speaker)? {
            Some(direction_launch(
                &snapshot,
                AudioDirection::Speaker,
                targets.speaker_capture,
                targets.speaker_playback,
                "translator-incoming-capture",
                "translator-incoming-playback",
            )?)
        } else {
            None
        };
        Ok(Self {
            microphone,
            speaker,
        })
    }

    const fn bypass_only(&self) -> bool {
        self.microphone.is_none() && self.speaker.is_none()
    }
}

fn validate_launch_snapshot(snapshot: &RuntimeSnapshot) -> Result<(), DuplexRuntimeError> {
    if !matches!(snapshot.provider_id, ProviderId::Local | ProviderId::Openai)
        || (snapshot.provider_id == ProviderId::Openai && !snapshot.audio_leaves_machine)
        || snapshot
            .audio_graph
            .as_ref()
            .is_none_or(|graph| graph.health != GraphHealth::Ready)
    {
        return Err(DuplexRuntimeError::InvalidConfiguration);
    }
    for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
        let mut states = snapshot
            .directions
            .iter()
            .filter(|state| state.direction_id == direction);
        let state = states
            .next()
            .ok_or(DuplexRuntimeError::InvalidConfiguration)?;
        if states.next().is_some()
            || snapshot
                .latency_policy
                .iter()
                .filter(|policy| policy.direction_id == direction)
                .count()
                != 1
        {
            return Err(DuplexRuntimeError::InvalidConfiguration);
        }
        if state.enabled {
            if state.source_language == state.target_language
                || state.voice_profile.language != state.target_language
            {
                return Err(DuplexRuntimeError::InvalidConfiguration);
            }
        }
    }
    Ok(())
}

fn direction_enabled(
    snapshot: &RuntimeSnapshot,
    direction: AudioDirection,
) -> Result<bool, DuplexRuntimeError> {
    snapshot
        .directions
        .iter()
        .find(|state| state.direction_id == direction)
        .map(|state| state.enabled)
        .ok_or(DuplexRuntimeError::InvalidConfiguration)
}

fn validate_explicit_audio_targets(targets: &DuplexAudioTargets) -> Result<(), DuplexRuntimeError> {
    if [
        targets.microphone_capture.as_str(),
        targets.microphone_playback.as_str(),
        targets.speaker_capture.as_str(),
        targets.speaker_playback.as_str(),
    ]
    .into_iter()
    .any(|name| name.trim().is_empty())
    {
        return Err(DuplexRuntimeError::InvalidConfiguration);
    }
    Ok(())
}

fn direction_launch(
    snapshot: &RuntimeSnapshot,
    direction: AudioDirection,
    capture_device: String,
    playback_device: String,
    capture_stream_name: &'static str,
    playback_stream_name: &'static str,
) -> Result<DirectionLaunch, DuplexRuntimeError> {
    let state = snapshot
        .directions
        .iter()
        .find(|candidate| candidate.direction_id == direction)
        .ok_or(DuplexRuntimeError::InvalidConfiguration)?;
    let mode = snapshot
        .latency_policy
        .iter()
        .find(|candidate| candidate.direction_id == direction)
        .ok_or(DuplexRuntimeError::InvalidConfiguration)?
        .current_mode;
    Ok(DirectionLaunch {
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
    })
}

struct PreparedDirection {
    launch: DirectionLaunch,
    session: DirectionSession,
    provider: ProviderStreamClient,
    capture: PulsePcmCapture,
    playback: Option<PulsePcmPlayback>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectionExit {
    Stopped,
    RestartRequired,
    ModeChanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationRestart {
    Policy,
    Fault,
}

#[derive(Default)]
struct GenerationPolicyBarrier {
    reopen_requested: AtomicBool,
    microphone_active: AtomicBool,
    speaker_active: AtomicBool,
}

impl GenerationPolicyBarrier {
    fn request_reopen(&self) {
        self.reopen_requested.store(true, Ordering::Release);
    }

    fn reopen_requested(&self) -> bool {
        self.reopen_requested.load(Ordering::Acquire)
    }

    fn set_active(&self, direction: AudioDirection, active: bool) {
        self.active_flag(direction).store(active, Ordering::Release);
    }

    fn is_active(&self, direction: AudioDirection) -> bool {
        self.active_flag(direction).load(Ordering::Acquire)
    }

    fn active_flag(&self, direction: AudioDirection) -> &AtomicBool {
        match direction {
            AudioDirection::Microphone => &self.microphone_active,
            AudioDirection::Speaker => &self.speaker_active,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueuedPlaybackMetadata {
    utterance_id: uuid::Uuid,
    sequence: u64,
    provider_monotonic_ns: u64,
    enqueued_monotonic_ns: u64,
}

async fn run_process_duplex(
    config: ProcessDuplexConfig,
    mut launch: DuplexLaunch,
    mut stop: watch::Receiver<bool>,
    ack: std_mpsc::SyncSender<Result<(), DuplexRuntimeError>>,
    observer: Arc<dyn DuplexRuntimeObserver>,
) -> Result<(), DuplexRuntimeError> {
    let runtime = ProcessSidecarRuntime::new(
        config.python,
        config.sidecar_root,
        config.socket_path.clone(),
        config.expected_uid,
    )
    .map_err(|_| DuplexRuntimeError::StartFailed)?;
    let mut supervisor = SidecarSupervisor::new(runtime);
    if let Err(error) = supervisor.start().await {
        let _ = ack.send(Err(DuplexRuntimeError::StartFailed));
        tracing::error!(event = "provider_start_failed", code = ?error);
        return Err(DuplexRuntimeError::StartFailed);
    }
    let sidecar_launch = supervisor
        .launch()
        .cloned()
        .ok_or(DuplexRuntimeError::StartFailed)?;
    if let Err(error) = wait_provider_ready(
        &config.socket_path,
        &sidecar_launch.token,
        sidecar_launch.generation_id,
        PROVIDER_READY_TIMEOUT,
    )
    .await
    {
        let _ = ack.send(Err(DuplexRuntimeError::StartFailed));
        tracing::error!(event = "provider_models_unavailable", code = ?error);
        let _ = supervisor.shutdown().await;
        return Err(DuplexRuntimeError::StartFailed);
    }
    tracing::info!(event = "provider_models_ready");

    let mut restart_count = 0;
    let mut ack = Some(ack);
    loop {
        if *stop.borrow() {
            let _ = supervisor.shutdown().await;
            return Ok(());
        }
        let sidecar_launch = supervisor
            .launch()
            .cloned()
            .ok_or(DuplexRuntimeError::StartFailed)?;
        let microphone = prepare_optional_direction(
            launch.microphone.clone(),
            &config.socket_path,
            &sidecar_launch.token,
        )
        .await;
        let speaker = prepare_optional_direction(
            launch.speaker.clone(),
            &config.socket_path,
            &sidecar_launch.token,
        )
        .await;
        let (microphone, speaker) = match (microphone, speaker) {
            (Ok(microphone), Ok(speaker)) => (microphone, speaker),
            (microphone, speaker) => {
                tracing::error!(
                    event = "duplex_direction_prepare_failed",
                    microphone_ready = microphone.is_ok(),
                    speaker_ready = speaker.is_ok()
                );
                if let Some(ack) = ack.take() {
                    let _ = ack.send(Err(DuplexRuntimeError::StartFailed));
                }
                let _ = supervisor.shutdown().await;
                return Err(DuplexRuntimeError::StartFailed);
            }
        };
        if microphone.is_none() && speaker.is_none() {
            if let Some(ack) = ack.take() {
                let _ = ack.send(Err(DuplexRuntimeError::StartFailed));
            }
            let _ = supervisor.shutdown().await;
            return Err(DuplexRuntimeError::StartFailed);
        }
        let microphone_id = microphone
            .as_ref()
            .map(|direction| direction.session.session_id());
        let speaker_id = speaker
            .as_ref()
            .map(|direction| direction.session.session_id());
        for session_id in [microphone_id, speaker_id].into_iter().flatten() {
            supervisor
                .register_session(session_id)
                .map_err(|_| DuplexRuntimeError::StartFailed)?;
        }
        if let Some(ack) = ack.take() {
            let _ = ack.send(Ok(()));
        }

        let (generation_stop_sender, generation_stop) = watch::channel(false);
        let policy_barrier = Arc::new(GenerationPolicyBarrier::default());
        let (restart, restart_session) = match (microphone, speaker) {
            (Some(microphone), Some(speaker)) => {
                let microphone_id = microphone_id.ok_or(DuplexRuntimeError::StartFailed)?;
                let speaker_id = speaker_id.ok_or(DuplexRuntimeError::StartFailed)?;
                let microphone_future = run_direction(
                    microphone,
                    stop.clone(),
                    generation_stop.clone(),
                    observer.clone(),
                    policy_barrier.clone(),
                );
                let speaker_future = run_direction(
                    speaker,
                    stop.clone(),
                    generation_stop,
                    observer.clone(),
                    policy_barrier.clone(),
                );
                tokio::pin!(microphone_future);
                tokio::pin!(speaker_future);
                enum Finished {
                    Stop,
                    Microphone(Result<DirectionExit, DuplexRuntimeError>),
                    Speaker(Result<DirectionExit, DuplexRuntimeError>),
                }
                let finished = tokio::select! {
                    _ = wait_for_stop(&mut stop) => Finished::Stop,
                    result = &mut microphone_future => Finished::Microphone(result),
                    result = &mut speaker_future => Finished::Speaker(result),
                };
                let (finished_result, peer_result, restart_session) = match finished {
                    Finished::Stop => {
                        let _ = generation_stop_sender.send(true);
                        let _ = tokio::join!(&mut microphone_future, &mut speaker_future);
                        let _ = supervisor.shutdown().await;
                        return Ok(());
                    }
                    Finished::Microphone(result) => {
                        if !should_wait_for_policy_peer(
                            &result,
                            AudioDirection::Speaker,
                            policy_barrier.as_ref(),
                        ) {
                            let _ = generation_stop_sender.send(true);
                        }
                        let peer_result = speaker_future.await;
                        match &result {
                            Ok(DirectionExit::Stopped) if *stop.borrow() => {
                                let _ = supervisor.shutdown().await;
                                return Ok(());
                            }
                            _ => (result, peer_result, microphone_id),
                        }
                    }
                    Finished::Speaker(result) => {
                        if !should_wait_for_policy_peer(
                            &result,
                            AudioDirection::Microphone,
                            policy_barrier.as_ref(),
                        ) {
                            let _ = generation_stop_sender.send(true);
                        }
                        let peer_result = microphone_future.await;
                        match &result {
                            Ok(DirectionExit::Stopped) if *stop.borrow() => {
                                let _ = supervisor.shutdown().await;
                                return Ok(());
                            }
                            _ => (result, peer_result, speaker_id),
                        }
                    }
                };
                (
                    classify_generation_restart(&finished_result, &peer_result),
                    restart_session,
                )
            }
            (Some(direction), None) | (None, Some(direction)) => {
                let session_id = direction.session.session_id();
                let direction_future = run_direction(
                    direction,
                    stop.clone(),
                    generation_stop,
                    observer.clone(),
                    policy_barrier,
                );
                tokio::pin!(direction_future);
                enum Finished {
                    Stop,
                    Direction(Result<DirectionExit, DuplexRuntimeError>),
                }
                let finished = tokio::select! {
                    _ = wait_for_stop(&mut stop) => Finished::Stop,
                    result = &mut direction_future => Finished::Direction(result),
                };
                let result = match finished {
                    Finished::Stop => {
                        let _ = generation_stop_sender.send(true);
                        let _ = direction_future.await;
                        let _ = supervisor.shutdown().await;
                        return Ok(());
                    }
                    Finished::Direction(result) => result,
                };
                match &result {
                    Ok(DirectionExit::Stopped) if *stop.borrow() => {
                        let _ = supervisor.shutdown().await;
                        return Ok(());
                    }
                    Ok(DirectionExit::ModeChanged) => (GenerationRestart::Policy, session_id),
                    _ => (GenerationRestart::Fault, session_id),
                }
            }
            (None, None) => return Err(DuplexRuntimeError::StartFailed),
        };
        if *stop.borrow() {
            let _ = supervisor.shutdown().await;
            return Ok(());
        }

        if restart == GenerationRestart::Policy {
            register_generation_restart(&mut restart_count, restart, observer.as_ref())?;
            for session_id in [microphone_id, speaker_id].into_iter().flatten() {
                supervisor
                    .close_session(session_id, std::future::ready(()))
                    .await
                    .map_err(|_| DuplexRuntimeError::StartFailed)?;
            }
            refresh_launch_modes(&mut launch, observer.as_ref());
            tracing::info!(
                event = "duplex_generation_policy_reopen",
                microphone_mode = ?launch.microphone.as_ref().map(|direction| direction.runtime.mode),
                speaker_mode = ?launch.speaker.as_ref().map(|direction| direction.runtime.mode)
            );
            continue;
        }
        register_generation_restart(&mut restart_count, restart, observer.as_ref())?;
        tracing::warn!(event = "duplex_generation_restart", attempt = restart_count);
        supervisor
            .close_session(restart_session, std::future::pending())
            .await
            .map_err(|_| DuplexRuntimeError::StartFailed)?;
        let sidecar_launch = supervisor
            .launch()
            .cloned()
            .ok_or(DuplexRuntimeError::StartFailed)?;
        wait_provider_ready(
            &config.socket_path,
            &sidecar_launch.token,
            sidecar_launch.generation_id,
            PROVIDER_READY_TIMEOUT,
        )
        .await
        .map_err(|_| DuplexRuntimeError::StartFailed)?;
    }
}

async fn prepare_optional_direction(
    launch: Option<DirectionLaunch>,
    socket_path: &std::path::Path,
    token: &str,
) -> Result<Option<PreparedDirection>, DuplexRuntimeError> {
    match launch {
        Some(launch) => prepare_direction(launch, socket_path, token)
            .await
            .map(Some),
        None => Ok(None),
    }
}

fn should_wait_for_policy_peer(
    result: &Result<DirectionExit, DuplexRuntimeError>,
    peer_direction: AudioDirection,
    policy_barrier: &GenerationPolicyBarrier,
) -> bool {
    matches!(result, Ok(DirectionExit::ModeChanged)) && policy_barrier.is_active(peer_direction)
}

async fn prepare_direction(
    launch: DirectionLaunch,
    socket_path: &std::path::Path,
    token: &str,
) -> Result<PreparedDirection, DuplexRuntimeError> {
    let direction = launch.runtime.direction;
    tracing::info!(event = "direction_prepare_started", direction = ?direction);
    let mut session = DirectionSession::new(launch.runtime);
    let mut provider = ProviderStreamClient::open(socket_path, token, session.open_request())
        .await
        .map_err(|_| direction_start_error(direction, "provider_stream_open"))?;
    tokio::time::timeout(DIRECTION_OPEN_TIMEOUT, async {
        let mut opened = false;
        let mut ready = false;
        while !opened || !ready {
            let event = provider
                .next_event()
                .await
                .map_err(|_| direction_start_error(direction, "provider_event_receive"))?
                .ok_or_else(|| direction_start_error(direction, "provider_event_closed"))?;
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
                        tracing::warn!(
                            event = "provider_operational_degraded",
                            direction = ?launch.runtime.direction
                        );
                    }
                }
                _ => {}
            }
            session
                .handle_provider_event(&event, monotonic_ns())
                .map_err(|_| direction_start_error(direction, "provider_event_validation"))?;
        }
        Ok::<(), DuplexRuntimeError>(())
    })
    .await
    .map_err(|_| direction_start_error(direction, "provider_open_timeout"))??;
    let capture = PulsePcmCapture::spawn(&PulsePcmCommand::capture(
        &launch.capture_device,
        launch.capture_stream_name,
    ))
    .map_err(|_| direction_start_error(direction, "capture_spawn"))?;
    let playback = PulsePcmPlayback::spawn(&PulsePcmCommand::playback(
        &launch.playback_device,
        launch.playback_stream_name,
    ))
    .map_err(|_| direction_start_error(direction, "playback_spawn"))?;
    tracing::info!(event = "direction_prepare_ready", direction = ?direction);
    Ok(PreparedDirection {
        launch,
        session,
        provider,
        capture,
        playback: Some(playback),
    })
}

fn direction_start_error(direction: AudioDirection, stage: &'static str) -> DuplexRuntimeError {
    tracing::error!(
        event = "direction_prepare_stage_failed",
        direction = ?direction,
        stage
    );
    DuplexRuntimeError::StartFailed
}

const fn provider_state_is_operational(state: ProviderState) -> bool {
    matches!(state, ProviderState::Ready | ProviderState::Degraded)
}

async fn run_direction(
    mut direction: PreparedDirection,
    mut stop: watch::Receiver<bool>,
    mut generation_stop: watch::Receiver<bool>,
    observer: Arc<dyn DuplexRuntimeObserver>,
    policy_barrier: Arc<GenerationPolicyBarrier>,
) -> Result<DirectionExit, DuplexRuntimeError> {
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
            _ = wait_for_stop(&mut stop) => {
                let _ = close_direction(
                    &mut direction,
                    CloseRequestReason::DaemonShutdown,
                )
                .await;
                return Ok(DirectionExit::Stopped);
            }
            _ = wait_for_stop(&mut generation_stop) => {
                close_direction(
                    &mut direction,
                    CloseRequestReason::ProviderSwitch,
                )
                .await?;
                return Ok(DirectionExit::RestartRequired);
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
                        DuplexRuntimeError::StartFailed
                    })?
                    .ok_or_else(|| direction_runtime_error(
                        direction.launch.runtime.direction,
                        "provider_event_closed",
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
                        DuplexRuntimeError::StartFailed
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
                            policy_barrier
                                .set_active(direction.launch.runtime.direction, false);
                            observer.observe(event);
                            if mode_change_after_event(
                                observer.as_ref(),
                                direction.launch.runtime.mode,
                                event,
                            )
                            .is_some()
                            {
                                policy_barrier.request_reopen();
                            }
                            mode_change_requested = policy_barrier.reopen_requested();
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
                            if let Some(playback) = direction.playback.as_mut() {
                                let _ = playback.stop().await;
                            }
                            direction.playback = None;
                            playback_audible_until_ns = 0;
                            direction.provider.send(request).await.map_err(|_| {
                                direction_runtime_error(
                                    direction.launch.runtime.direction,
                                    "expired_audio_cancel_send",
                                )
                            })?;
                        }
                        DirectionEffect::SessionClosed => {}
                    }
                }
                while let Some(frame) = playback_queue.pop() {
                    let metadata = playback_metadata
                        .pop_front()
                        .ok_or(DuplexRuntimeError::StartFailed)?;
                    if direction.playback.is_none() {
                        direction.playback = Some(
                            PulsePcmPlayback::spawn(&PulsePcmCommand::playback(
                                &direction.launch.playback_device,
                                direction.launch.playback_stream_name,
                            ))
                            .map_err(|_| direction_runtime_error(
                                direction.launch.runtime.direction,
                                "playback_respawn",
                            ))?,
                        );
                    }
                    let write_result = direction
                        .playback
                        .as_mut()
                        .expect("playback was restored")
                        .write_frame(&frame)
                        .await;
                    let observed_monotonic_ns = monotonic_ns();
                    observe_playback_write(
                        write_result,
                        direction.launch.runtime.direction,
                        metadata,
                        observed_monotonic_ns,
                        observer.as_ref(),
                    )
                        .map_err(|_| direction_runtime_error(
                            direction.launch.runtime.direction,
                            "playback_write",
                        ))?;
                    playback_audible_until_ns = extend_playback_deadline(
                        playback_audible_until_ns,
                        observed_monotonic_ns,
                        u64::from(frame.format().frame_duration_ms()),
                    );
                }
                if mode_change_requested {
                    wait_for_playback_deadline(playback_audible_until_ns).await;
                    close_direction(&mut direction, CloseRequestReason::ProviderSwitch).await?;
                    return Ok(DirectionExit::ModeChanged);
                }
            }
            frame = direction.capture.read_frame(capture_sequence, monotonic_ns()) => {
                let frame = frame.map_err(|_| direction_runtime_error(
                    direction.launch.runtime.direction,
                    "capture_read",
                ))?;
                capture_sequence = capture_sequence.saturating_add(1);
                capture_queue
                    .push(frame)
                    .map_err(|_| direction_runtime_error(
                        direction.launch.runtime.direction,
                        "capture_queue_overflow",
                    ))?;
                while let Some(frame) = capture_queue.pop() {
                    let events = segmenter
                        .process(frame)
                        .map_err(|_| direction_runtime_error(
                            direction.launch.runtime.direction,
                            "capture_vad",
                        ))?;
                    for event in events {
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
                        if let CaptureEvent::SpeechStarted {
                            utterance_id,
                            capture_monotonic_ns,
                            ..
                        } = &event
                        {
                            policy_barrier
                                .set_active(direction.launch.runtime.direction, true);
                            observer.observe(DuplexRuntimeEvent::SpeechStarted {
                                direction: direction.launch.runtime.direction,
                                utterance_id: *utterance_id,
                                capture_monotonic_ns: *capture_monotonic_ns,
                            });
                        }
                        if let Some(request) = direction
                            .session
                            .handle_capture(event)
                            .map_err(|_| direction_runtime_error(
                                direction.launch.runtime.direction,
                                "capture_session",
                            ))?
                        {
                            direction
                                .provider
                                .send(request)
                                .await
                                .map_err(|_| direction_runtime_error(
                                    direction.launch.runtime.direction,
                                    "capture_provider_send",
                                ))?;
                        }
                    }
                }
            }
            _ = watchdog.tick() => {
                let effects = direction
                    .session
                    .poll(monotonic_ns())
                    .map_err(|_| direction_runtime_error(
                        direction.launch.runtime.direction,
                        "watchdog_poll",
                    ))?;
                for effect in effects {
                    match effect {
                        DirectionWatchdogEffect::Send(request) => {
                            tracing::warn!(
                                event = "direction_watchdog_action",
                                direction = ?direction.launch.runtime.direction,
                                action = "send"
                            );
                            direction.provider.send(request).await
                                .map_err(|_| DuplexRuntimeError::StartFailed)?;
                        }
                        DirectionWatchdogEffect::PurgeAndSend(request) => {
                            tracing::warn!(
                                event = "direction_watchdog_action",
                                direction = ?direction.launch.runtime.direction,
                                action = "purge_and_send"
                            );
                            playback_queue.clear();
                            playback_metadata.clear();
                            if let Some(playback) = direction.playback.as_mut() {
                                let _ = playback.stop().await;
                            }
                            direction.playback = None;
                            playback_audible_until_ns = 0;
                            direction.provider.send(request).await
                                .map_err(|_| DuplexRuntimeError::StartFailed)?;
                        }
                        DirectionWatchdogEffect::RestartSidecar => {
                            tracing::error!(
                                event = "direction_runtime_stage_failed",
                                direction = ?direction.launch.runtime.direction,
                                stage = "watchdog_restart"
                            );
                            return Ok(DirectionExit::RestartRequired);
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

async fn wait_for_playback_deadline(deadline_ns: u64) {
    let remaining_ns = deadline_ns.saturating_sub(monotonic_ns());
    if remaining_ns > 0 {
        tokio::time::sleep(Duration::from_nanos(remaining_ns)).await;
    }
}

fn direction_runtime_error(direction: AudioDirection, stage: &'static str) -> DuplexRuntimeError {
    tracing::error!(
        event = "direction_runtime_stage_failed",
        direction = ?direction,
        stage
    );
    DuplexRuntimeError::StartFailed
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

async fn close_direction(
    direction: &mut PreparedDirection,
    reason: CloseRequestReason,
) -> Result<(), DuplexRuntimeError> {
    let _ = direction.capture.stop().await;
    if let Some(playback) = direction.playback.as_mut() {
        let _ = playback.stop().await;
    }
    direction.playback = None;
    direction
        .provider
        .send(direction.session.close_request(reason))
        .await
        .map_err(|_| DuplexRuntimeError::StartFailed)?;
    tokio::time::timeout(CLOSE_ACK_TIMEOUT, async {
        loop {
            let event = direction
                .provider
                .next_event()
                .await
                .map_err(|_| DuplexRuntimeError::StartFailed)?
                .ok_or(DuplexRuntimeError::StartFailed)?;
            let closed = direction
                .session
                .handle_provider_event(&event, monotonic_ns())
                .map_err(|_| DuplexRuntimeError::StartFailed)?
                .into_iter()
                .any(|effect| effect == DirectionEffect::SessionClosed);
            if closed {
                return Ok(());
            }
        }
    })
    .await
    .map_err(|_| DuplexRuntimeError::StartFailed)?
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

fn refresh_launch_modes(launch: &mut DuplexLaunch, observer: &dyn DuplexRuntimeObserver) -> bool {
    let mut changed = false;
    for direction in [&mut launch.microphone, &mut launch.speaker] {
        let Some(direction) = direction.as_mut() else {
            continue;
        };
        if let Some(requested) = observer.requested_mode(direction.runtime.direction)
            && requested != direction.runtime.mode
        {
            direction.runtime.mode = requested;
            changed = true;
        }
    }
    changed
}

fn register_generation_restart(
    fault_restarts: &mut usize,
    restart: GenerationRestart,
    observer: &dyn DuplexRuntimeObserver,
) -> Result<(), DuplexRuntimeError> {
    if restart == GenerationRestart::Fault {
        *fault_restarts = fault_restarts.saturating_add(1);
        if *fault_restarts > MAX_RUNTIME_RESTARTS {
            return Err(DuplexRuntimeError::StartFailed);
        }
        let attempt = NonZeroU32::new(
            u32::try_from(*fault_restarts).map_err(|_| DuplexRuntimeError::StartFailed)?,
        )
        .ok_or(DuplexRuntimeError::StartFailed)?;
        observer.observe(DuplexRuntimeEvent::GenerationRestart { attempt });
    }
    Ok(())
}

fn classify_generation_restart(
    finished: &Result<DirectionExit, DuplexRuntimeError>,
    peer: &Result<DirectionExit, DuplexRuntimeError>,
) -> GenerationRestart {
    if matches!(finished, Ok(DirectionExit::ModeChanged))
        && matches!(
            peer,
            Ok(DirectionExit::RestartRequired | DirectionExit::ModeChanged)
        )
    {
        GenerationRestart::Policy
    } else {
        GenerationRestart::Fault
    }
}

async fn wait_for_stop(stop: &mut watch::Receiver<bool>) {
    if *stop.borrow() {
        return;
    }
    let _ = stop.changed().await;
}

fn monotonic_ns() -> u64 {
    let time = clock_gettime(ClockId::Monotonic);
    let seconds = u64::try_from(time.tv_sec).unwrap_or(0);
    let nanos = u64::try_from(time.tv_nsec).unwrap_or(0);
    seconds.saturating_mul(1_000_000_000).saturating_add(nanos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use translator_audio::AudioGraphState;
    use translator_core::{TranslationMode, VoiceEngine};

    #[derive(Default)]
    struct RecordingObserver {
        events: Mutex<Vec<DuplexRuntimeEvent>>,
    }

    impl DuplexRuntimeObserver for RecordingObserver {
        fn observe(&self, event: DuplexRuntimeEvent) {
            self.events.lock().unwrap().push(event);
        }
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

    #[test]
    fn explicit_audio_targets_are_preserved_without_production_device_resolution() {
        let targets = DuplexAudioTargets {
            microphone_capture: "benchmark_ru_source.monitor".to_owned(),
            microphone_playback: "benchmark_mic_sink".to_owned(),
            speaker_capture: "benchmark_remote_sink.monitor".to_owned(),
            speaker_playback: "benchmark_headphones".to_owned(),
        };

        let launch = DuplexLaunch::try_with_audio_targets(ready_snapshot(), targets).unwrap();
        let microphone = launch.microphone.as_ref().unwrap();
        let speaker = launch.speaker.as_ref().unwrap();

        assert_eq!(microphone.capture_device, "benchmark_ru_source.monitor");
        assert_eq!(microphone.playback_device, "benchmark_mic_sink");
        assert_eq!(speaker.capture_device, "benchmark_remote_sink.monitor");
        assert_eq!(speaker.playback_device, "benchmark_headphones");
        assert!(DuplexLaunch::try_from(ready_snapshot()).is_err());
    }

    #[test]
    fn explicit_audio_targets_reject_empty_names() {
        let targets = DuplexAudioTargets {
            microphone_capture: " ".to_owned(),
            microphone_playback: MIC_OUT_SINK.to_owned(),
            speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
            speaker_playback: "benchmark_headphones".to_owned(),
        };

        assert!(DuplexLaunch::try_with_audio_targets(ready_snapshot(), targets).is_err());
    }

    #[test]
    fn explicit_audio_targets_do_not_bypass_direction_validation() {
        let mut snapshot = ready_snapshot();
        snapshot.directions[0].target_language = snapshot.directions[0].source_language;
        snapshot.directions[0].voice_profile.language = snapshot.directions[0].target_language;
        let targets = DuplexAudioTargets {
            microphone_capture: "benchmark_ru_source.monitor".to_owned(),
            microphone_playback: MIC_OUT_SINK.to_owned(),
            speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
            speaker_playback: "benchmark_headphones".to_owned(),
        };

        assert!(DuplexLaunch::try_with_audio_targets(snapshot, targets).is_err());
    }

    #[test]
    fn disabled_direction_is_not_prepared_for_launch() {
        let mut snapshot = ready_snapshot();
        snapshot.directions[1].enabled = false;
        snapshot.directions[1].target_language = snapshot.directions[1].source_language;
        snapshot.directions[1].voice_profile.language = snapshot.directions[1].target_language;

        let launch = DuplexLaunch::try_with_audio_targets(
            snapshot,
            DuplexAudioTargets {
                microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "benchmark_headphones".to_owned(),
            },
        )
        .unwrap();

        assert!(launch.microphone.is_some());
        assert!(launch.speaker.is_none());
    }

    #[test]
    fn launch_allows_bypass_when_no_directions_are_enabled() {
        let mut snapshot = ready_snapshot();
        for direction in &mut snapshot.directions {
            direction.enabled = false;
        }

        let launch = DuplexLaunch::try_with_audio_targets(
            snapshot,
            DuplexAudioTargets {
                microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "benchmark_headphones".to_owned(),
            },
        )
        .unwrap();

        assert!(launch.bypass_only());
    }

    #[test]
    fn openai_provider_launches_after_cloud_provider_selection() {
        let mut snapshot = ready_snapshot();
        snapshot.provider_id = ProviderId::Openai;
        snapshot.audio_leaves_machine = true;
        for direction in &mut snapshot.directions {
            direction.voice_profile.engine = VoiceEngine::Openai;
        }
        let launch = DuplexLaunch::try_with_audio_targets(
            snapshot,
            DuplexAudioTargets {
                microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "benchmark_headphones".to_owned(),
            },
        )
        .unwrap();
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
    fn openai_provider_launch_requires_audio_egress_visibility() {
        let mut snapshot = ready_snapshot();
        snapshot.provider_id = ProviderId::Openai;
        snapshot.audio_leaves_machine = false;
        for direction in &mut snapshot.directions {
            direction.voice_profile.engine = VoiceEngine::Openai;
        }

        assert!(
            DuplexLaunch::try_with_audio_targets(
                snapshot,
                DuplexAudioTargets {
                    microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                    microphone_playback: MIC_OUT_SINK.to_owned(),
                    speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                    speaker_playback: "benchmark_headphones".to_owned(),
                },
            )
            .is_err()
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

    #[test]
    fn policy_reopen_waits_only_for_an_active_peer() {
        let barrier = GenerationPolicyBarrier::default();
        let mode_changed = Ok(DirectionExit::ModeChanged);
        barrier.set_active(AudioDirection::Speaker, true);

        assert!(should_wait_for_policy_peer(
            &mode_changed,
            AudioDirection::Speaker,
            &barrier,
        ));
        barrier.set_active(AudioDirection::Speaker, false);
        assert!(!should_wait_for_policy_peer(
            &mode_changed,
            AudioDirection::Speaker,
            &barrier,
        ));
        assert!(!should_wait_for_policy_peer(
            &Err(DuplexRuntimeError::StartFailed),
            AudioDirection::Speaker,
            &barrier,
        ));
    }

    #[test]
    fn latency_policy_modes_propagate_to_the_next_provider_open() {
        let store = RuntimeStore::default();
        let observer = RuntimeLatencyObserver::new(store);
        let mut launch = DuplexLaunch::try_with_audio_targets(
            ready_snapshot(),
            DuplexAudioTargets {
                microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "benchmark_headphones".to_owned(),
            },
        )
        .unwrap();

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
        assert!(refresh_launch_modes(&mut launch, &observer));
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
        assert!(refresh_launch_modes(&mut launch, &observer));
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
    fn generation_refreshes_both_modes_and_policy_reopen_preserves_fault_budget() {
        let store = RuntimeStore::default();
        store.set_latency_policy(crate::LatencyPolicyPatch {
            direction_id: AudioDirection::Microphone,
            current_mode: TranslationMode::Balanced,
        });
        store.set_latency_policy(crate::LatencyPolicyPatch {
            direction_id: AudioDirection::Speaker,
            current_mode: TranslationMode::StreamingFirst,
        });
        let observer = RuntimeLatencyObserver::new(store);
        let mut launch = DuplexLaunch::try_with_audio_targets(
            ready_snapshot(),
            DuplexAudioTargets {
                microphone_capture: "benchmark_ru_source.monitor".to_owned(),
                microphone_playback: MIC_OUT_SINK.to_owned(),
                speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
                speaker_playback: "benchmark_headphones".to_owned(),
            },
        )
        .unwrap();
        let mut fault_restarts = 0;
        let restart_observer = RecordingObserver::default();

        assert_eq!(
            classify_generation_restart(
                &Ok(DirectionExit::ModeChanged),
                &Ok(DirectionExit::RestartRequired),
            ),
            GenerationRestart::Policy
        );
        assert_eq!(
            classify_generation_restart(
                &Ok(DirectionExit::ModeChanged),
                &Err(DuplexRuntimeError::StartFailed),
            ),
            GenerationRestart::Fault
        );
        register_generation_restart(
            &mut fault_restarts,
            GenerationRestart::Policy,
            &restart_observer,
        )
        .unwrap();
        assert_eq!(fault_restarts, 0);
        assert!(restart_observer.events.lock().unwrap().is_empty());
        assert!(refresh_launch_modes(&mut launch, &observer));
        assert_eq!(
            launch.microphone.as_ref().unwrap().runtime.mode,
            TranslationMode::Balanced
        );
        assert_eq!(
            launch.speaker.as_ref().unwrap().runtime.mode,
            TranslationMode::StreamingFirst
        );

        for expected in 1..=MAX_RUNTIME_RESTARTS {
            register_generation_restart(
                &mut fault_restarts,
                GenerationRestart::Fault,
                &restart_observer,
            )
            .unwrap();
            assert_eq!(fault_restarts, expected);
        }
        assert_eq!(
            restart_observer.events.lock().unwrap().as_slice(),
            &[
                DuplexRuntimeEvent::GenerationRestart {
                    attempt: std::num::NonZeroU32::new(1).unwrap(),
                },
                DuplexRuntimeEvent::GenerationRestart {
                    attempt: std::num::NonZeroU32::new(2).unwrap(),
                },
                DuplexRuntimeEvent::GenerationRestart {
                    attempt: std::num::NonZeroU32::new(3).unwrap(),
                },
            ]
        );
        assert_eq!(
            register_generation_restart(
                &mut fault_restarts,
                GenerationRestart::Fault,
                &restart_observer,
            ),
            Err(DuplexRuntimeError::StartFailed)
        );
    }
}
