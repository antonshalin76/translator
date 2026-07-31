use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::Duration,
};

use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::watch;
use translator_audio::{
    CaptureEvent, PcmFrame, ProcessIdentity, PulsePcmCapture, PulsePcmCommand, PulsePcmPlayback,
    PulseRoutingWatcher, REMOTE_IN_SINK, RoutingProfile, SpeechSegmenter, StreamPcmFormat,
    SystemCommandRunner, VIRTUAL_MIC_SOURCE, VirtualPeerCapability, VirtualPeerDiscovery,
    WebRtcVoiceDetector,
};
use translator_core::AudioDirection;
use uuid::Uuid;

use crate::{
    ActiveDuplexRuntime, ActiveRoundTripRuntime, AudioOperationLease, DuplexRuntimeError,
    DuplexRuntimeEvent, DuplexRuntimeObserver, ExactPcmEvidence, ProcessDuplexConfig,
    ProcessDuplexRunner, RoundTripCheckpoint, RoundTripLatency, RoundTripProgress, RoundTripRunner,
    RoundTripRuntimeError, RuntimeSnapshot, TerminalOutcome,
};

const HARD_TIMEOUT: Duration = Duration::from_secs(300);
const ACTIVE_STOP_TIMEOUT: Duration = Duration::from_secs(30);
const PCM_FINISH_GRACE_MS: u64 = 5_000;
const MAX_PCM_FINISH_MS: u64 = 30_000;
const VIRTUAL_PEER_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(2);
const VIRTUAL_PEER_DISCOVERY_INTERVAL: Duration = Duration::from_millis(20);
const TAP_DRAIN_FRAMES: usize = 20;
const INCOMING_PLAYBACK_FRAME_NS: u64 = 20_000_000;
const INCOMING_PLAYBACK_DRAIN_GRACE_NS: u64 = 100_000_000;

pub type RoundTripWorkerFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, RoundTripProcessError>> + 'a>>;

#[derive(Default)]
struct PlaybackDrainBudget {
    buffered_ms: u64,
}

impl PlaybackDrainBudget {
    fn record_write<E>(&mut self, result: Result<(), E>, frame: &PcmFrame) -> Result<(), E> {
        result?;
        self.buffered_ms = self
            .buffered_ms
            .saturating_add(u64::from(frame.format().frame_duration_ms()))
            .min(MAX_PCM_FINISH_MS.saturating_sub(PCM_FINISH_GRACE_MS));
        Ok(())
    }

    fn take_timeout(&mut self) -> Duration {
        let buffered_ms = std::mem::take(&mut self.buffered_ms);
        Duration::from_millis(
            buffered_ms
                .saturating_add(PCM_FINISH_GRACE_MS)
                .min(MAX_PCM_FINISH_MS),
        )
    }

    fn reset(&mut self) {
        self.buffered_ms = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RoundTripProcessError {
    #[error("round-trip process was stopped")]
    Stopped,
    #[error("round-trip duplex runtime failed")]
    Duplex,
    #[error("round-trip audio worker failed")]
    Audio,
    #[error("round-trip virtual peer capability is invalid")]
    InvalidCapability,
    #[error("round-trip virtual peer route failed")]
    Route,
    #[error("round-trip progress transition failed")]
    Progress,
}

pub trait RoundTripDuplexFactory: Send + Sync {
    fn start(
        &self,
        snapshot: RuntimeSnapshot,
        observer: Arc<dyn DuplexRuntimeObserver>,
    ) -> Result<Box<dyn ActiveDuplexRuntime>, RoundTripProcessError>;
}

pub trait RoundTripAudioWorkerFactory: Send + Sync {
    fn create(
        &self,
        session_id: Uuid,
        physical_sink: &str,
    ) -> Result<Box<dyn RoundTripAudioWorker>, RoundTripProcessError>;
}

pub trait RoundTripAudioWorker: Send {
    fn capture_english_utterance<'a>(
        &'a mut self,
        outgoing_terminal: &'a mut watch::Receiver<bool>,
        stop: &'a mut watch::Receiver<bool>,
    ) -> RoundTripWorkerFuture<'a, Vec<PcmFrame>>;

    fn monitor_english<'a>(
        &'a mut self,
        frames: &'a [PcmFrame],
        stop: &'a mut watch::Receiver<bool>,
    ) -> RoundTripWorkerFuture<'a, ()>;

    fn spawn_virtual_peer(&mut self) -> Result<ProcessIdentity, RoundTripProcessError>;

    fn write_virtual_peer_frame<'a>(
        &'a mut self,
        frame: &'a PcmFrame,
    ) -> RoundTripWorkerFuture<'a, (u64, StreamPcmFormat, usize, [u8; 32])>;

    fn finish_virtual_peer<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()>;

    fn stop_writes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()>;

    fn finish_processes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()>;

    fn clear_sensitive(&mut self);
}

pub trait VirtualPeerRouteControllerFactory: Send + Sync {
    fn create(&self) -> Box<dyn VirtualPeerRouteController>;
}

pub trait VirtualPeerRouteController: Send {
    fn route(
        &mut self,
        session_id: Uuid,
        process: ProcessIdentity,
        expected_target: &str,
    ) -> Result<VirtualPeerCapability, RoundTripProcessError>;

    fn validate(
        &mut self,
        capability: &VirtualPeerCapability,
        expected_target: &str,
    ) -> Result<(), RoundTripProcessError>;

    fn restore(&mut self, capability: &VirtualPeerCapability) -> Result<(), RoundTripProcessError>;

    fn ensure_absent(
        &mut self,
        capability: &VirtualPeerCapability,
    ) -> Result<(), RoundTripProcessError>;
}

pub struct RoundTripProcessRunner {
    duplex_factory: Arc<dyn RoundTripDuplexFactory>,
    audio_factory: Arc<dyn RoundTripAudioWorkerFactory>,
    route_factory: Arc<dyn VirtualPeerRouteControllerFactory>,
    timeout: Duration,
    active: Arc<AtomicBool>,
}

impl RoundTripProcessRunner {
    pub fn new(config: ProcessDuplexConfig) -> Self {
        Self::with_components(
            Arc::new(ProcessRoundTripDuplexFactory { config }),
            Arc::new(PulseRoundTripAudioWorkerFactory),
            Arc::new(PulseVirtualPeerRouteControllerFactory),
            HARD_TIMEOUT,
        )
    }

    pub fn with_components(
        duplex_factory: Arc<dyn RoundTripDuplexFactory>,
        audio_factory: Arc<dyn RoundTripAudioWorkerFactory>,
        route_factory: Arc<dyn VirtualPeerRouteControllerFactory>,
        timeout: Duration,
    ) -> Self {
        Self {
            duplex_factory,
            audio_factory,
            route_factory,
            timeout,
            active: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl RoundTripRunner for RoundTripProcessRunner {
    fn start(
        &self,
        snapshot: RuntimeSnapshot,
        session_id: Uuid,
        progress: RoundTripProgress,
        lease: AudioOperationLease,
    ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError> {
        if self
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(RoundTripRuntimeError::StartFailed);
        }

        let duplex_factory = Arc::clone(&self.duplex_factory);
        let audio_factory = Arc::clone(&self.audio_factory);
        let route_factory = Arc::clone(&self.route_factory);
        let timeout = self.timeout;
        let active = Arc::clone(&self.active);
        let finished = Arc::new(AtomicBool::new(false));
        let thread_finished = Arc::clone(&finished);
        let (stop_sender, stop_receiver) = watch::channel(false);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("translator-round-trip-process".to_owned())
            .spawn(move || {
                let result = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|_| RoundTripRuntimeError::StartFailed)
                    .and_then(|runtime| {
                        runtime.block_on(run_round_trip(
                            snapshot,
                            session_id,
                            progress,
                            lease,
                            stop_receiver,
                            duplex_factory,
                            audio_factory,
                            route_factory,
                            timeout,
                        ))
                    });
                active.store(false, Ordering::Release);
                thread_finished.store(true, Ordering::Release);
                let _ = done_sender.send(result);
            })
            .map_err(|_| {
                self.active.store(false, Ordering::Release);
                RoundTripRuntimeError::StartFailed
            })?;

        Ok(Box::new(ProcessActiveRoundTrip {
            stop_sender,
            done_receiver,
            thread: Some(thread),
            finished,
            terminal_result: None,
        }))
    }
}

struct ProcessActiveRoundTrip {
    stop_sender: watch::Sender<bool>,
    done_receiver: std_mpsc::Receiver<Result<(), RoundTripRuntimeError>>,
    thread: Option<thread::JoinHandle<()>>,
    finished: Arc<AtomicBool>,
    terminal_result: Option<Result<(), RoundTripRuntimeError>>,
}

impl ActiveRoundTripRuntime for ProcessActiveRoundTrip {
    fn stop(&mut self) -> Result<(), RoundTripRuntimeError> {
        if let Some(result) = self.terminal_result {
            return result;
        }
        if self.thread.is_none() {
            return Ok(());
        }
        let _ = self.stop_sender.send(true);
        let result = self
            .done_receiver
            .recv_timeout(ACTIVE_STOP_TIMEOUT)
            .map_err(|_| RoundTripRuntimeError::StopFailed)?;
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            self.terminal_result = Some(Err(RoundTripRuntimeError::StopFailed));
            return Err(RoundTripRuntimeError::StopFailed);
        }
        self.terminal_result = Some(result);
        result
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}

impl Drop for ProcessActiveRoundTrip {
    fn drop(&mut self) {
        let _ = self.stop_sender.send(true);
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_round_trip(
    snapshot: RuntimeSnapshot,
    session_id: Uuid,
    progress: RoundTripProgress,
    lease: AudioOperationLease,
    mut stop: watch::Receiver<bool>,
    duplex_factory: Arc<dyn RoundTripDuplexFactory>,
    audio_factory: Arc<dyn RoundTripAudioWorkerFactory>,
    route_factory: Arc<dyn VirtualPeerRouteControllerFactory>,
    timeout: Duration,
) -> Result<(), RoundTripRuntimeError> {
    let started_at = std::time::Instant::now();
    let deadline = started_at + timeout;
    let cleanup_budget = (timeout / 4).min(ACTIVE_STOP_TIMEOUT);
    let active_deadline = deadline.checked_sub(cleanup_budget).unwrap_or(started_at);
    let observer = Arc::new(ProgressObserver::new(session_id, progress.clone()));
    let mut resources = RoundTripResources {
        duplex: None,
        audio: None,
        routes: None,
        capability: None,
        frames: Vec::new(),
        lease: Some(lease),
    };

    let lifecycle = async {
        resources.routes = Some(route_factory.create());
        let physical_sink = snapshot
            .devices
            .as_ref()
            .and_then(|devices| devices.sink.selected.as_ref())
            .map(|sink| sink.name.clone())
            .ok_or(RoundTripProcessError::Audio)?;
        resources.duplex = Some(duplex_factory.start(snapshot, observer.clone())?);
        resources.audio = Some(audio_factory.create(session_id, &physical_sink)?);
        execute_round_trip(
            session_id,
            &physical_sink,
            &progress,
            &observer,
            &mut resources,
            &mut stop,
        )
        .await
    };
    let active_remaining = active_deadline.saturating_duration_since(std::time::Instant::now());
    let outcome = match tokio::time::timeout(active_remaining, lifecycle).await {
        Ok(result) => result,
        Err(_) => {
            progress.fail(session_id, crate::RoundTripErrorCode::Timeout);
            Err(RoundTripProcessError::Stopped)
        }
    };

    if let Err(error) = outcome
        && error != RoundTripProcessError::Stopped
    {
        tracing::error!(
            event = "round_trip_process_failed",
            stage = "active_lifecycle",
            error = ?error
        );
        progress.fail(session_id, crate::RoundTripErrorCode::RuntimeFailed);
    }
    let teardown_remaining = deadline.saturating_duration_since(std::time::Instant::now());
    let teardown = if teardown_remaining.is_zero() {
        resources.force_release();
        Err(RoundTripProcessError::Audio)
    } else {
        match tokio::time::timeout(teardown_remaining, resources.teardown()).await {
            Ok(result) => result,
            Err(_) => {
                resources.force_release();
                Err(RoundTripProcessError::Audio)
            }
        }
    };
    if teardown.is_err() && outcome.is_ok() {
        progress.fail(session_id, crate::RoundTripErrorCode::RuntimeFailed);
    }
    teardown.map_err(|_| RoundTripRuntimeError::StopFailed)
}

async fn execute_round_trip(
    session_id: Uuid,
    physical_sink: &str,
    progress: &RoundTripProgress,
    observer: &Arc<ProgressObserver>,
    resources: &mut RoundTripResources,
    stop: &mut watch::Receiver<bool>,
) -> Result<(), RoundTripProcessError> {
    let mut outgoing_terminal = observer.outgoing_terminal_receiver();
    resources.frames = resources
        .audio
        .as_mut()
        .ok_or(RoundTripProcessError::Audio)?
        .capture_english_utterance(&mut outgoing_terminal, stop)
        .await?;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "english_tap_capture",
        frame_count = resources.frames.len()
    );
    if resources.frames.is_empty() {
        return Err(RoundTripProcessError::Audio);
    }
    observer
        .wait_for(RoundTripCheckpoint::EnglishFirstAudio, stop)
        .await?;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "english_first_audio"
    );

    resources
        .audio
        .as_mut()
        .ok_or(RoundTripProcessError::Audio)?
        .monitor_english(&resources.frames, stop)
        .await?;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "english_monitor"
    );
    let monitor_complete_ms = observer.elapsed_from_outgoing_onset_ms();
    let process = resources
        .audio
        .as_mut()
        .ok_or(RoundTripProcessError::Audio)?
        .spawn_virtual_peer()?;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "virtual_peer_spawn"
    );
    if ProcessIdentity::inspect(process.pid) != Some(process) {
        return Err(RoundTripProcessError::InvalidCapability);
    }
    let routes = resources
        .routes
        .as_mut()
        .ok_or(RoundTripProcessError::Route)?;
    let capability = routes.route(session_id, process, physical_sink)?;
    resources.capability = Some(capability.clone());
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "virtual_peer_route"
    );
    validate_capability(session_id, process, &capability)?;
    routes.validate(&capability, REMOTE_IN_SINK)?;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "virtual_peer_route_validation"
    );

    observer.begin_incoming();
    let mut evidence = ExactPcmEvidence::new(StreamPcmFormat::provider_default());
    for frame in &resources.frames {
        evidence
            .capture(frame)
            .map_err(|_| RoundTripProcessError::Progress)?;
        let receipt = resources
            .audio
            .as_mut()
            .ok_or(RoundTripProcessError::Audio)?
            .write_virtual_peer_frame(frame)
            .await?;
        record_write_receipt(&mut evidence, frame, receipt)?;
    }
    resources
        .audio
        .as_mut()
        .ok_or(RoundTripProcessError::Audio)?
        .finish_virtual_peer()
        .await?;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "virtual_peer_reinjection"
    );
    resources
        .routes
        .as_mut()
        .ok_or(RoundTripProcessError::Route)?
        .ensure_absent(&capability)?;
    resources.capability = None;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "virtual_peer_absent"
    );
    progress
        .set_exact_pcm_proof(session_id, evidence.proof())
        .map_err(|_| RoundTripProcessError::Progress)?;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "exact_pcm_proof"
    );
    observer.mark_reinjecting(monitor_complete_ms)?;
    observer
        .wait_for(RoundTripCheckpoint::RussianFirstAudio, stop)
        .await?;
    observer.wait_for_incoming_terminal(stop).await?;
    let drain_started_ns = monotonic_ns();
    observer.wait_for_incoming_playback_drain(stop).await?;
    tracing::info!(
        event = "round_trip_stage_completed",
        stage = "incoming_playback_drain",
        drain_wait_ms = elapsed_ms(Some(drain_started_ns), monotonic_ns())
    );
    observer.mark_completed()
}

fn validate_capability(
    session_id: Uuid,
    process: ProcessIdentity,
    capability: &VirtualPeerCapability,
) -> Result<(), RoundTripProcessError> {
    if capability.session_id != session_id
        || capability.process != process
        || capability.object_serial == 0
        || capability.process_binary != "pacat"
        || ProcessIdentity::inspect(process.pid) != Some(process)
    {
        return Err(RoundTripProcessError::InvalidCapability);
    }
    Ok(())
}

struct PcmWriteReceipt {
    sequence: u64,
    format: StreamPcmFormat,
    bytes_written: usize,
    pcm_sha256: [u8; 32],
}

impl From<(u64, StreamPcmFormat, usize, [u8; 32])> for PcmWriteReceipt {
    fn from(value: (u64, StreamPcmFormat, usize, [u8; 32])) -> Self {
        Self {
            sequence: value.0,
            format: value.1,
            bytes_written: value.2,
            pcm_sha256: value.3,
        }
    }
}

fn record_write_receipt(
    evidence: &mut ExactPcmEvidence,
    frame: &PcmFrame,
    receipt: (u64, StreamPcmFormat, usize, [u8; 32]),
) -> Result<(), RoundTripProcessError> {
    let receipt = PcmWriteReceipt::from(receipt);
    let expected_hash: [u8; 32] = Sha256::digest(frame.pcm()).into();
    if receipt.sequence != frame.sequence()
        || receipt.format != frame.format()
        || receipt.bytes_written != frame.pcm().len()
        || receipt.pcm_sha256 != expected_hash
    {
        return Err(RoundTripProcessError::Progress);
    }
    evidence
        .reinject(frame)
        .map_err(|_| RoundTripProcessError::Progress)
}

struct RoundTripResources {
    duplex: Option<Box<dyn ActiveDuplexRuntime>>,
    audio: Option<Box<dyn RoundTripAudioWorker>>,
    routes: Option<Box<dyn VirtualPeerRouteController>>,
    capability: Option<VirtualPeerCapability>,
    frames: Vec<PcmFrame>,
    lease: Option<AudioOperationLease>,
}

impl RoundTripResources {
    async fn teardown(&mut self) -> Result<(), RoundTripProcessError> {
        let mut failed = false;
        let capability = self.capability.clone();
        if let (Some(routes), Some(capability)) = (self.routes.as_mut(), capability.as_ref()) {
            failed |= routes.restore(capability).is_err();
        }
        if let Some(audio) = self.audio.as_mut() {
            failed |= audio.stop_writes().await.is_err();
            failed |= audio.finish_processes().await.is_err();
        }
        if let (Some(routes), Some(capability)) = (self.routes.as_mut(), capability.as_ref()) {
            failed |= routes.ensure_absent(capability).is_err();
        }
        if let Some(duplex) = self.duplex.as_mut() {
            failed |= duplex.stop().is_err();
        }
        self.capability = None;
        self.force_release();
        if failed {
            Err(RoundTripProcessError::Audio)
        } else {
            Ok(())
        }
    }

    fn force_release(&mut self) {
        self.capability = None;
        zeroize_frames(&mut self.frames);
        if let Some(audio) = self.audio.as_mut() {
            audio.clear_sensitive();
        }
        self.audio = None;
        self.duplex = None;
        self.routes = None;
        if let Some(mut lease) = self.lease.take() {
            let _ = lease.release();
        }
    }
}

fn zeroize_frames(frames: &mut Vec<PcmFrame>) {
    for frame in frames.drain(..) {
        let mut pcm = frame.into_pcm();
        zeroize_bytes(&mut pcm);
    }
    frames.shrink_to_fit();
}

fn zeroize_bytes(bytes: &mut [u8]) {
    for byte in bytes {
        // SAFETY: each pointer comes from an exclusive live slice element.
        unsafe {
            std::ptr::write_volatile(byte, 0);
        }
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

struct ObserverState {
    checkpoint: RoundTripCheckpoint,
    outgoing_utterance: Option<Uuid>,
    incoming_utterance: Option<Uuid>,
    outgoing_onset_ns: Option<u64>,
    incoming_onset_ns: Option<u64>,
    outgoing_transcript_final: bool,
    outgoing_translation_final: bool,
    outgoing_first_audio: bool,
    outgoing_completed: bool,
    incoming_enabled: bool,
    incoming_transcript_final: bool,
    incoming_translation_final: bool,
    incoming_first_audio: bool,
    incoming_completed: bool,
    incoming_terminal: bool,
    incoming_playback_audible_until_ns: u64,
}

struct ProgressObserver {
    session_id: Uuid,
    progress: RoundTripProgress,
    state: Mutex<ObserverState>,
    checkpoint_sender: watch::Sender<RoundTripCheckpoint>,
    outgoing_terminal_sender: watch::Sender<bool>,
    incoming_terminal_sender: watch::Sender<bool>,
}

impl ProgressObserver {
    fn new(session_id: Uuid, progress: RoundTripProgress) -> Self {
        let (checkpoint_sender, _) = watch::channel(RoundTripCheckpoint::WaitingForSpeech);
        let (outgoing_terminal_sender, _) = watch::channel(false);
        let (incoming_terminal_sender, _) = watch::channel(false);
        Self {
            session_id,
            progress,
            state: Mutex::new(ObserverState {
                checkpoint: RoundTripCheckpoint::WaitingForSpeech,
                outgoing_utterance: None,
                incoming_utterance: None,
                outgoing_onset_ns: None,
                incoming_onset_ns: None,
                outgoing_transcript_final: false,
                outgoing_translation_final: false,
                outgoing_first_audio: false,
                outgoing_completed: false,
                incoming_enabled: false,
                incoming_transcript_final: false,
                incoming_translation_final: false,
                incoming_first_audio: false,
                incoming_completed: false,
                incoming_terminal: false,
                incoming_playback_audible_until_ns: 0,
            }),
            checkpoint_sender,
            outgoing_terminal_sender,
            incoming_terminal_sender,
        }
    }

    fn outgoing_terminal_receiver(&self) -> watch::Receiver<bool> {
        self.outgoing_terminal_sender.subscribe()
    }

    fn incoming_terminal_receiver(&self) -> watch::Receiver<bool> {
        self.incoming_terminal_sender.subscribe()
    }

    fn begin_incoming(&self) {
        lock_recovering(&self.state).incoming_enabled = true;
    }

    fn mark_reinjecting(
        &self,
        english_monitor_complete_ms: u32,
    ) -> Result<(), RoundTripProcessError> {
        let mut state = lock_recovering(&self.state);
        if state.checkpoint != RoundTripCheckpoint::EnglishFirstAudio {
            return Err(RoundTripProcessError::Progress);
        }
        self.advance_locked(
            &mut state,
            RoundTripCheckpoint::VirtualPeerReinjecting,
            RoundTripLatency {
                english_monitor_complete_ms: Some(english_monitor_complete_ms),
                ..RoundTripLatency::default()
            },
        )?;
        self.drain_locked(&mut state);
        Ok(())
    }

    fn elapsed_from_outgoing_onset_ms(&self) -> u32 {
        let onset = lock_recovering(&self.state).outgoing_onset_ns;
        elapsed_ms(onset, monotonic_ns())
    }

    async fn wait_for(
        &self,
        checkpoint: RoundTripCheckpoint,
        stop: &mut watch::Receiver<bool>,
    ) -> Result<(), RoundTripProcessError> {
        let mut receiver = self.checkpoint_sender.subscribe();
        loop {
            let observed = *receiver.borrow();
            if observed == checkpoint {
                return Ok(());
            }
            if matches!(
                observed,
                RoundTripCheckpoint::Failed | RoundTripCheckpoint::Stopped
            ) {
                return Err(RoundTripProcessError::Progress);
            }
            tokio::select! {
                _ = wait_for_stop(stop) => return Err(RoundTripProcessError::Stopped),
                changed = receiver.changed() => {
                    if changed.is_err() {
                        return Err(RoundTripProcessError::Progress);
                    }
                }
            }
        }
    }

    async fn wait_for_incoming_terminal(
        &self,
        stop: &mut watch::Receiver<bool>,
    ) -> Result<(), RoundTripProcessError> {
        let mut receiver = self.incoming_terminal_receiver();
        loop {
            if *receiver.borrow() {
                return Ok(());
            }
            if matches!(
                *self.checkpoint_sender.borrow(),
                RoundTripCheckpoint::Failed | RoundTripCheckpoint::Stopped
            ) {
                return Err(RoundTripProcessError::Progress);
            }
            tokio::select! {
                _ = wait_for_stop(stop) => return Err(RoundTripProcessError::Stopped),
                changed = receiver.changed() => {
                    if changed.is_err() {
                        return Err(RoundTripProcessError::Progress);
                    }
                }
            }
        }
    }

    async fn wait_for_incoming_playback_drain(
        &self,
        stop: &mut watch::Receiver<bool>,
    ) -> Result<(), RoundTripProcessError> {
        let deadline = lock_recovering(&self.state)
            .incoming_playback_audible_until_ns
            .saturating_add(INCOMING_PLAYBACK_DRAIN_GRACE_NS);
        let remaining_ns = deadline.saturating_sub(monotonic_ns());
        if remaining_ns == 0 {
            return Ok(());
        }
        tokio::select! {
            _ = wait_for_stop(stop) => Err(RoundTripProcessError::Stopped),
            _ = tokio::time::sleep(Duration::from_nanos(remaining_ns)) => Ok(()),
        }
    }

    fn mark_completed(&self) -> Result<(), RoundTripProcessError> {
        let mut state = lock_recovering(&self.state);
        if state.checkpoint != RoundTripCheckpoint::RussianFirstAudio || !state.incoming_terminal {
            return Err(RoundTripProcessError::Progress);
        }
        self.advance_locked(
            &mut state,
            RoundTripCheckpoint::Completed,
            RoundTripLatency::default(),
        )
    }

    fn drain_locked(&self, state: &mut ObserverState) {
        loop {
            let transition = match state.checkpoint {
                RoundTripCheckpoint::OutgoingVad if state.outgoing_transcript_final => Some((
                    RoundTripCheckpoint::OutgoingAsrFinal,
                    RoundTripLatency::default(),
                )),
                RoundTripCheckpoint::OutgoingAsrFinal if state.outgoing_translation_final => {
                    Some((
                        RoundTripCheckpoint::OutgoingTranslationFinal,
                        RoundTripLatency::default(),
                    ))
                }
                RoundTripCheckpoint::OutgoingTranslationFinal if state.outgoing_first_audio => {
                    Some((
                        RoundTripCheckpoint::EnglishFirstAudio,
                        RoundTripLatency {
                            outgoing_first_audio_ms: Some(elapsed_ms(
                                state.outgoing_onset_ns,
                                monotonic_ns(),
                            )),
                            ..RoundTripLatency::default()
                        },
                    ))
                }
                RoundTripCheckpoint::VirtualPeerReinjecting if state.incoming_transcript_final => {
                    Some((
                        RoundTripCheckpoint::IncomingAsrFinal,
                        RoundTripLatency::default(),
                    ))
                }
                RoundTripCheckpoint::IncomingAsrFinal if state.incoming_translation_final => {
                    Some((
                        RoundTripCheckpoint::IncomingTranslationFinal,
                        RoundTripLatency::default(),
                    ))
                }
                RoundTripCheckpoint::IncomingTranslationFinal if state.incoming_first_audio => {
                    Some((
                        RoundTripCheckpoint::RussianFirstAudio,
                        RoundTripLatency {
                            incoming_first_audio_ms: Some(elapsed_ms(
                                state.incoming_onset_ns,
                                monotonic_ns(),
                            )),
                            physical_mic_onset_to_returned_ru_first_audible_ms: Some(elapsed_ms(
                                state.outgoing_onset_ns,
                                monotonic_ns(),
                            )),
                            ..RoundTripLatency::default()
                        },
                    ))
                }
                _ => None,
            };
            let Some((checkpoint, latency)) = transition else {
                break;
            };
            if self.advance_locked(state, checkpoint, latency).is_err() {
                break;
            }
        }
    }

    fn advance_locked(
        &self,
        state: &mut ObserverState,
        checkpoint: RoundTripCheckpoint,
        latency: RoundTripLatency,
    ) -> Result<(), RoundTripProcessError> {
        self.progress
            .advance(self.session_id, checkpoint, latency)
            .map_err(|_| RoundTripProcessError::Progress)?;
        state.checkpoint = checkpoint;
        self.checkpoint_sender.send_replace(checkpoint);
        Ok(())
    }

    fn fail_locked(&self, state: &mut ObserverState) {
        if self
            .progress
            .fail(self.session_id, crate::RoundTripErrorCode::RuntimeFailed)
        {
            state.checkpoint = RoundTripCheckpoint::Failed;
            self.checkpoint_sender
                .send_replace(RoundTripCheckpoint::Failed);
            self.outgoing_terminal_sender.send_replace(true);
        }
    }
}

impl DuplexRuntimeObserver for ProgressObserver {
    fn observe(&self, event: DuplexRuntimeEvent) {
        let mut state = lock_recovering(&self.state);
        match event {
            DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Microphone,
                utterance_id,
                capture_monotonic_ns,
            } if state.outgoing_utterance.is_none() => {
                state.outgoing_utterance = Some(utterance_id);
                state.outgoing_onset_ns = Some(capture_monotonic_ns);
                if state.checkpoint == RoundTripCheckpoint::WaitingForSpeech {
                    let _ = self.advance_locked(
                        &mut state,
                        RoundTripCheckpoint::OutgoingVad,
                        RoundTripLatency::default(),
                    );
                }
            }
            DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Microphone,
                ..
            } => {
                let _ = self.progress.record_recursion_trigger(self.session_id);
            }
            DuplexRuntimeEvent::TranscriptFinal {
                direction: AudioDirection::Microphone,
                utterance_id,
            } if state.outgoing_utterance == Some(utterance_id) => {
                state.outgoing_transcript_final = true;
            }
            DuplexRuntimeEvent::TranslationFinal {
                direction: AudioDirection::Microphone,
                utterance_id,
            } if state.outgoing_utterance == Some(utterance_id) => {
                state.outgoing_translation_final = true;
            }
            DuplexRuntimeEvent::AudioFrame {
                direction: AudioDirection::Microphone,
                utterance_id,
                ..
            } if state.outgoing_utterance == Some(utterance_id) => {
                state.outgoing_first_audio = true;
                if state.outgoing_completed {
                    state.outgoing_transcript_final = true;
                    state.outgoing_translation_final = true;
                }
            }
            DuplexRuntimeEvent::UtteranceTerminalOutcome {
                direction: AudioDirection::Microphone,
                utterance_id,
                outcome: TerminalOutcome::Completed,
            } if state.outgoing_utterance == Some(utterance_id) => {
                state.outgoing_completed = true;
                if state.outgoing_first_audio {
                    state.outgoing_transcript_final = true;
                    state.outgoing_translation_final = true;
                }
            }
            DuplexRuntimeEvent::UtteranceTerminalOutcome {
                direction: AudioDirection::Microphone,
                utterance_id,
                outcome,
            } if state.outgoing_utterance == Some(utterance_id)
                && outcome != TerminalOutcome::Completed =>
            {
                self.fail_locked(&mut state);
            }
            DuplexRuntimeEvent::UtteranceTerminal {
                direction: AudioDirection::Microphone,
                utterance_id,
            } if state.outgoing_utterance == Some(utterance_id) => {
                self.outgoing_terminal_sender.send_replace(true);
            }
            DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Speaker,
                utterance_id,
                capture_monotonic_ns,
            } if state.incoming_enabled && state.incoming_utterance.is_none() => {
                state.incoming_utterance = Some(utterance_id);
                state.incoming_onset_ns = Some(capture_monotonic_ns);
            }
            DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Speaker,
                ..
            } if state.incoming_enabled => {
                let _ = self.progress.record_recursion_trigger(self.session_id);
            }
            DuplexRuntimeEvent::TranscriptFinal {
                direction: AudioDirection::Speaker,
                utterance_id,
            } if state.incoming_enabled && state.incoming_utterance == Some(utterance_id) => {
                state.incoming_transcript_final = true;
            }
            DuplexRuntimeEvent::TranslationFinal {
                direction: AudioDirection::Speaker,
                utterance_id,
            } if state.incoming_enabled && state.incoming_utterance == Some(utterance_id) => {
                state.incoming_translation_final = true;
            }
            DuplexRuntimeEvent::AudioFrame {
                direction: AudioDirection::Speaker,
                utterance_id,
                observed_monotonic_ns,
                ..
            } if state.incoming_enabled && state.incoming_utterance == Some(utterance_id) => {
                state.incoming_first_audio = true;
                state.incoming_playback_audible_until_ns = state
                    .incoming_playback_audible_until_ns
                    .max(observed_monotonic_ns)
                    .saturating_add(INCOMING_PLAYBACK_FRAME_NS);
                if state.incoming_completed {
                    state.incoming_transcript_final = true;
                    state.incoming_translation_final = true;
                }
            }
            DuplexRuntimeEvent::UtteranceTerminalOutcome {
                direction: AudioDirection::Speaker,
                utterance_id,
                outcome: TerminalOutcome::Completed,
            } if state.incoming_enabled && state.incoming_utterance == Some(utterance_id) => {
                state.incoming_completed = true;
                if state.incoming_first_audio {
                    state.incoming_transcript_final = true;
                    state.incoming_translation_final = true;
                }
            }
            DuplexRuntimeEvent::UtteranceTerminalOutcome {
                direction: AudioDirection::Speaker,
                utterance_id,
                outcome,
            } if state.incoming_enabled
                && state.incoming_utterance == Some(utterance_id)
                && outcome != TerminalOutcome::Completed =>
            {
                self.fail_locked(&mut state);
            }
            DuplexRuntimeEvent::UtteranceTerminal {
                direction: AudioDirection::Speaker,
                utterance_id,
            } if state.incoming_enabled && state.incoming_utterance == Some(utterance_id) => {
                state.incoming_terminal = true;
                self.incoming_terminal_sender.send_replace(true);
            }
            _ => {}
        }
        self.drain_locked(&mut state);
    }
}

struct ProcessRoundTripDuplexFactory {
    config: ProcessDuplexConfig,
}

impl RoundTripDuplexFactory for ProcessRoundTripDuplexFactory {
    fn start(
        &self,
        snapshot: RuntimeSnapshot,
        observer: Arc<dyn DuplexRuntimeObserver>,
    ) -> Result<Box<dyn ActiveDuplexRuntime>, RoundTripProcessError> {
        crate::DuplexRunner::start(
            &ProcessDuplexRunner::with_observer(self.config.clone(), observer),
            snapshot,
        )
        .map_err(|_| RoundTripProcessError::Duplex)
    }
}

#[derive(Default)]
struct EnglishTapCollector {
    frames: Vec<PcmFrame>,
    terminal_observed: bool,
    drain_frames: usize,
    segment_active: bool,
}

impl EnglishTapCollector {
    fn record_events(&mut self, events: Vec<CaptureEvent>) -> Result<(), RoundTripProcessError> {
        for event in events {
            match event {
                CaptureEvent::SpeechStarted { .. } => {
                    self.segment_active = true;
                }
                CaptureEvent::Frame {
                    frame,
                    end_of_utterance,
                    ..
                } => {
                    let normalized = PcmFrame::try_new(
                        self.frames.len() as u64,
                        frame.capture_monotonic_ns(),
                        frame.format(),
                        frame.into_pcm(),
                    )
                    .map_err(|_| RoundTripProcessError::Audio)?;
                    self.frames.push(normalized);
                    if end_of_utterance {
                        self.segment_active = false;
                    }
                }
            }
        }
        Ok(())
    }

    fn observe_terminal(&mut self) {
        self.terminal_observed = true;
    }

    fn terminal_observed(&self) -> bool {
        self.terminal_observed
    }

    fn record_drain_frame(&mut self) {
        if self.terminal_observed {
            self.drain_frames = self.drain_frames.saturating_add(1);
        }
    }

    fn is_complete(&self) -> bool {
        self.terminal_observed && self.drain_frames >= TAP_DRAIN_FRAMES && !self.segment_active
    }

    #[cfg(test)]
    fn frames(&self) -> &[PcmFrame] {
        &self.frames
    }

    fn into_frames(mut self) -> Vec<PcmFrame> {
        std::mem::take(&mut self.frames)
    }
}

impl Drop for EnglishTapCollector {
    fn drop(&mut self) {
        zeroize_frames(&mut self.frames);
    }
}

struct PulseRoundTripAudioWorkerFactory;

impl RoundTripAudioWorkerFactory for PulseRoundTripAudioWorkerFactory {
    fn create(
        &self,
        session_id: Uuid,
        physical_sink: &str,
    ) -> Result<Box<dyn RoundTripAudioWorker>, RoundTripProcessError> {
        let capture = PulsePcmCapture::spawn(&PulsePcmCommand::capture(
            VIRTUAL_MIC_SOURCE,
            "translator-round-trip-english-tap",
        ))
        .map_err(|_| RoundTripProcessError::Audio)?;
        Ok(Box::new(PulseRoundTripAudioWorker {
            session_id,
            physical_sink: physical_sink.to_owned(),
            capture: Some(capture),
            monitor: None,
            virtual_peer: None,
            capture_sequence: 0,
            virtual_peer_drain: PlaybackDrainBudget::default(),
        }))
    }
}

struct PulseRoundTripAudioWorker {
    session_id: Uuid,
    physical_sink: String,
    capture: Option<PulsePcmCapture>,
    monitor: Option<PulsePcmPlayback>,
    virtual_peer: Option<PulsePcmPlayback>,
    capture_sequence: u64,
    virtual_peer_drain: PlaybackDrainBudget,
}

impl RoundTripAudioWorker for PulseRoundTripAudioWorker {
    fn capture_english_utterance<'a>(
        &'a mut self,
        outgoing_terminal: &'a mut watch::Receiver<bool>,
        stop: &'a mut watch::Receiver<bool>,
    ) -> RoundTripWorkerFuture<'a, Vec<PcmFrame>> {
        Box::pin(async move {
            let mut segmenter =
                SpeechSegmenter::new(Uuid::new_v4(), WebRtcVoiceDetector::default());
            let mut collector = EnglishTapCollector::default();
            loop {
                if *outgoing_terminal.borrow() {
                    collector.observe_terminal();
                }
                if collector.is_complete() {
                    return Ok(collector.into_frames());
                }
                let capture = self.capture.as_mut().ok_or(RoundTripProcessError::Audio)?;
                tokio::select! {
                    _ = wait_for_stop(stop) => return Err(RoundTripProcessError::Stopped),
                    changed = outgoing_terminal.changed(), if !collector.terminal_observed() => {
                        changed.map_err(|_| RoundTripProcessError::Progress)?;
                        if *outgoing_terminal.borrow() {
                            collector.observe_terminal();
                        }
                    }
                    frame = capture.read_frame(self.capture_sequence, monotonic_ns()) => {
                        let frame = frame.map_err(|_| RoundTripProcessError::Audio)?;
                        self.capture_sequence = self.capture_sequence.saturating_add(1);
                        if collector.terminal_observed() {
                            collector.record_drain_frame();
                        }
                        let events = segmenter
                            .process(frame)
                            .map_err(|_| RoundTripProcessError::Audio)?;
                        collector.record_events(events)?;
                    }
                }
            }
        })
    }

    fn monitor_english<'a>(
        &'a mut self,
        frames: &'a [PcmFrame],
        stop: &'a mut watch::Receiver<bool>,
    ) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            let mut drain = PlaybackDrainBudget::default();
            self.monitor = Some(
                PulsePcmPlayback::spawn(&PulsePcmCommand::playback(
                    &self.physical_sink,
                    "translator-round-trip-english-monitor",
                ))
                .map_err(|_| RoundTripProcessError::Audio)?,
            );
            for frame in frames {
                tokio::select! {
                    _ = wait_for_stop(stop) => return Err(RoundTripProcessError::Stopped),
                    result = self.monitor.as_mut()
                        .ok_or(RoundTripProcessError::Audio)?
                        .write_frame(frame) => {
                        drain
                            .record_write(result, frame)
                            .map_err(|_| RoundTripProcessError::Audio)?;
                    }
                }
            }
            let monitor = self.monitor.take().ok_or(RoundTripProcessError::Audio)?;
            monitor.finish(drain.take_timeout()).await.map_err(|error| {
                tracing::error!(
                    event = "round_trip_pcm_finish_failed",
                    stage = "english_monitor",
                    error = ?error
                );
                RoundTripProcessError::Audio
            })
        })
    }

    fn spawn_virtual_peer(&mut self) -> Result<ProcessIdentity, RoundTripProcessError> {
        self.virtual_peer_drain.reset();
        let peer = PulsePcmPlayback::spawn(&PulsePcmCommand::virtual_peer_playback(
            &self.physical_sink,
            self.session_id,
        ))
        .map_err(|_| RoundTripProcessError::Audio)?;
        let identity = peer
            .process_identity()
            .ok_or(RoundTripProcessError::InvalidCapability)?;
        self.virtual_peer = Some(peer);
        Ok(identity)
    }

    fn write_virtual_peer_frame<'a>(
        &'a mut self,
        frame: &'a PcmFrame,
    ) -> RoundTripWorkerFuture<'a, (u64, StreamPcmFormat, usize, [u8; 32])> {
        Box::pin(async move {
            let result = self
                .virtual_peer
                .as_mut()
                .ok_or(RoundTripProcessError::Audio)?
                .write_frame(frame)
                .await;
            self.virtual_peer_drain
                .record_write(result, frame)
                .map_err(|_| RoundTripProcessError::Audio)?;
            Ok((
                frame.sequence(),
                frame.format(),
                frame.pcm().len(),
                Sha256::digest(frame.pcm()).into(),
            ))
        })
    }

    fn finish_virtual_peer<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            let peer = self
                .virtual_peer
                .take()
                .ok_or(RoundTripProcessError::Audio)?;
            let timeout = self.virtual_peer_drain.take_timeout();
            peer.finish(timeout).await.map_err(|error| {
                tracing::error!(
                    event = "round_trip_pcm_finish_failed",
                    stage = "virtual_peer",
                    error = ?error
                );
                RoundTripProcessError::Audio
            })
        })
    }

    fn stop_writes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            if let Some(monitor) = self.monitor.as_mut() {
                let _ = monitor.stop().await;
            }
            if let Some(peer) = self.virtual_peer.as_mut() {
                let _ = peer.stop().await;
            }
            self.virtual_peer_drain.reset();
            Ok(())
        })
    }

    fn finish_processes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            if let Some(capture) = self.capture.as_mut() {
                let _ = capture.stop().await;
            }
            if let Some(monitor) = self.monitor.as_mut() {
                let _ = monitor.stop().await;
            }
            if let Some(peer) = self.virtual_peer.as_mut() {
                let _ = peer.stop().await;
            }
            self.capture = None;
            self.monitor = None;
            self.virtual_peer = None;
            self.virtual_peer_drain.reset();
            Ok(())
        })
    }

    fn clear_sensitive(&mut self) {
        self.virtual_peer_drain.reset();
    }
}

struct PulseVirtualPeerRouteControllerFactory;

impl VirtualPeerRouteControllerFactory for PulseVirtualPeerRouteControllerFactory {
    fn create(&self) -> Box<dyn VirtualPeerRouteController> {
        Box::new(PulseVirtualPeerRouteController {
            discovery: VirtualPeerDiscovery::new(SystemCommandRunner),
            watcher: PulseRoutingWatcher::new(
                SystemCommandRunner,
                RoutingProfile::SyntheticValidation,
            ),
            routed: None,
        })
    }
}

struct PulseVirtualPeerRouteController {
    discovery: VirtualPeerDiscovery<SystemCommandRunner>,
    watcher: PulseRoutingWatcher<SystemCommandRunner>,
    routed: Option<VirtualPeerCapability>,
}

impl VirtualPeerRouteController for PulseVirtualPeerRouteController {
    fn route(
        &mut self,
        session_id: Uuid,
        process: ProcessIdentity,
        expected_target: &str,
    ) -> Result<VirtualPeerCapability, RoundTripProcessError> {
        let deadline = std::time::Instant::now() + VIRTUAL_PEER_DISCOVERY_TIMEOUT;
        let capability = loop {
            match self
                .discovery
                .discover(session_id, process, expected_target)
            {
                Ok(capability) => break capability,
                Err(_) if std::time::Instant::now() < deadline => {
                    thread::sleep(VIRTUAL_PEER_DISCOVERY_INTERVAL);
                }
                Err(_) => return Err(RoundTripProcessError::Route),
            }
        };
        self.watcher
            .route_virtual_peer(capability.clone())
            .map_err(|_| RoundTripProcessError::Route)?;
        self.routed = Some(capability.clone());
        Ok(capability)
    }

    fn validate(
        &mut self,
        capability: &VirtualPeerCapability,
        expected_target: &str,
    ) -> Result<(), RoundTripProcessError> {
        if self.routed.as_ref() != Some(capability) {
            return Err(RoundTripProcessError::InvalidCapability);
        }
        self.watcher
            .validate_virtual_peer_route(capability, expected_target)
            .map(|_| ())
            .map_err(|_| RoundTripProcessError::InvalidCapability)
    }

    fn restore(&mut self, capability: &VirtualPeerCapability) -> Result<(), RoundTripProcessError> {
        if self.routed.as_ref() != Some(capability) {
            return Err(RoundTripProcessError::InvalidCapability);
        }
        self.watcher
            .restore_virtual_peer()
            .map_err(|_| RoundTripProcessError::Route)?;
        self.routed = None;
        Ok(())
    }

    fn ensure_absent(
        &mut self,
        capability: &VirtualPeerCapability,
    ) -> Result<(), RoundTripProcessError> {
        let deadline = std::time::Instant::now() + VIRTUAL_PEER_DISCOVERY_TIMEOUT;
        loop {
            match self.discovery.ensure_absent(capability) {
                Ok(()) => {
                    self.routed = None;
                    return Ok(());
                }
                Err(_) if std::time::Instant::now() < deadline => {
                    thread::sleep(VIRTUAL_PEER_DISCOVERY_INTERVAL);
                }
                Err(_) => return Err(RoundTripProcessError::Route),
            }
        }
    }
}

async fn wait_for_stop(stop: &mut watch::Receiver<bool>) {
    if *stop.borrow() {
        return;
    }
    let _ = stop.changed().await;
}

fn monotonic_ns() -> u64 {
    let time = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    u64::try_from(time.tv_sec)
        .unwrap_or(0)
        .saturating_mul(1_000_000_000)
        .saturating_add(u64::try_from(time.tv_nsec).unwrap_or(0))
}

fn elapsed_ms(start_ns: Option<u64>, end_ns: u64) -> u32 {
    start_ns
        .map(|start| end_ns.saturating_sub(start) / 1_000_000)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(u32::MAX)
}

fn lock_recovering<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl From<DuplexRuntimeError> for RoundTripProcessError {
    fn from(_: DuplexRuntimeError) -> Self {
        Self::Duplex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_collector_ignores_vad_eou_until_outgoing_terminal_and_drain() {
        let format = StreamPcmFormat::provider_default();
        let first = PcmFrame::try_new(7, 0, format, vec![1; format.frame_bytes()]).unwrap();
        let second =
            PcmFrame::try_new(31, 320_000_000, format, vec![2; format.frame_bytes()]).unwrap();
        let utterance_id = Uuid::new_v4();
        let stream_id = Uuid::new_v4();
        let mut collector = EnglishTapCollector::default();

        collector
            .record_events(vec![CaptureEvent::Frame {
                stream_id,
                utterance_id,
                frame: first,
                end_of_utterance: true,
            }])
            .unwrap();
        assert!(!collector.is_complete());

        collector
            .record_events(vec![
                CaptureEvent::SpeechStarted {
                    stream_id,
                    utterance_id: Uuid::new_v4(),
                    capture_monotonic_ns: 320_000_000,
                },
                CaptureEvent::Frame {
                    stream_id,
                    utterance_id: Uuid::new_v4(),
                    frame: second,
                    end_of_utterance: false,
                },
            ])
            .unwrap();
        assert_eq!(collector.frames().len(), 2);
        assert_eq!(collector.frames()[0].sequence(), 0);
        assert_eq!(collector.frames()[1].sequence(), 1);
        assert!(!collector.is_complete());

        collector.observe_terminal();
        for _ in 0..TAP_DRAIN_FRAMES {
            collector.record_drain_frame();
        }
        assert!(!collector.is_complete());
        collector
            .record_events(vec![CaptureEvent::Frame {
                stream_id,
                utterance_id,
                frame: PcmFrame::try_new(52, 740_000_000, format, vec![0; format.frame_bytes()])
                    .unwrap(),
                end_of_utterance: true,
            }])
            .unwrap();
        assert!(collector.is_complete());
    }

    #[test]
    fn volatile_zeroize_overwrites_every_pcm_byte() {
        let mut bytes = vec![0x5a; 640];
        zeroize_bytes(&mut bytes);
        assert!(bytes.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn playback_drain_budget_counts_only_successful_writes_and_resets_on_take() {
        let format = StreamPcmFormat::provider_default();
        let frame = PcmFrame::try_new(0, 0, format, vec![0; format.frame_bytes()]).unwrap();
        let mut budget = PlaybackDrainBudget::default();

        assert!(budget.record_write::<()>(Ok(()), &frame).is_ok());
        assert!(budget.record_write::<()>(Err(()), &frame).is_err());
        assert_eq!(budget.take_timeout(), Duration::from_millis(5_020));
        assert_eq!(budget.take_timeout(), Duration::from_millis(5_000));
    }

    #[test]
    fn playback_drain_budget_caps_the_deadline_at_thirty_seconds() {
        let format = StreamPcmFormat::provider_default();
        let frame = PcmFrame::try_new(0, 0, format, vec![0; format.frame_bytes()]).unwrap();
        let mut budget = PlaybackDrainBudget::default();

        for _ in 0..2_000 {
            budget.record_write::<()>(Ok(()), &frame).unwrap();
        }

        assert_eq!(budget.take_timeout(), Duration::from_secs(30));
    }
}
