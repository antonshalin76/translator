use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use futures_util::FutureExt;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::{mpsc, watch};
use translator_audio::{
    CaptureEvent, PcmFrame, ProcessIdentity, PulsePcmCapture, PulsePcmCommand, PulsePcmPlayback,
    PulseRoutingWatcher, REMOTE_IN_SINK, RoutingProfile, SpeechSegmenter, StreamPcmFormat,
    SystemCommandRunner, VIRTUAL_MIC_SOURCE, VirtualPeerCapability, VirtualPeerDiscovery,
    WebRtcVoiceDetector,
};
use translator_core::AudioDirection;
use uuid::Uuid;

use crate::{
    ActiveDuplexRuntime, ActiveRoundTripRuntime, DuplexRuntimeError, DuplexRuntimeEvent,
    DuplexRuntimeObserver, ExactPcmEvidence, ProcessDuplexConfig, ProcessDuplexRunner,
    RoundTripCheckpoint, RoundTripLatency, RoundTripProgress, RoundTripRunner,
    RoundTripRuntimeError, RoundTripTerminal, TerminalOutcome,
};

const HARD_TIMEOUT: Duration = Duration::from_secs(300);
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
        admitted: crate::AdmittedDuplex,
        observer: Arc<dyn DuplexRuntimeObserver>,
        deadline: tokio::time::Instant,
    ) -> crate::DuplexStartResult;
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

    fn restore(&mut self) -> Result<(), RoundTripProcessError>;

    fn ensure_absent(&mut self) -> Result<(), RoundTripProcessError>;
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
        admitted: crate::AdmittedDuplex,
        session_id: Uuid,
        progress: RoundTripProgress,
        start_deadline: Instant,
    ) -> Result<Box<dyn ActiveRoundTripRuntime>, RoundTripRuntimeError> {
        if Instant::now() >= start_deadline {
            return Err(RoundTripRuntimeError::StartFailed);
        }
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
        let (cleanup_sender, cleanup_receiver) = mpsc::channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("translator-round-trip-process".to_owned())
            .spawn(move || {
                let completion = progress.clone();
                let result = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map(|runtime| {
                        runtime.block_on(run_round_trip(
                            admitted,
                            session_id,
                            progress,
                            stop_receiver,
                            cleanup_receiver,
                            duplex_factory,
                            audio_factory,
                            route_factory,
                            timeout,
                            start_deadline,
                        ))
                    })
                    .unwrap_or(RoundTripTerminal::Failed(
                        crate::RoundTripErrorCode::RuntimeFailed,
                    ));
                active.store(false, Ordering::Release);
                thread_finished.store(true, Ordering::Release);
                let _ = done_sender.send(result);
                completion.completed(session_id);
            })
            .map_err(|_| {
                self.active.store(false, Ordering::Release);
                RoundTripRuntimeError::StartFailed
            })?;

        Ok(Box::new(ProcessActiveRoundTrip {
            stop_sender,
            cleanup_sender,
            pending_cleanup: None,
            done_receiver,
            thread: Some(thread),
            finished,
            terminal_result: None,
            pending_terminal: None,
        }))
    }
}

type CleanupReceipt = std_mpsc::Receiver<Result<(), RoundTripRuntimeError>>;

struct ProcessActiveRoundTrip {
    stop_sender: watch::Sender<bool>,
    cleanup_sender: mpsc::Sender<CleanupRequest>,
    pending_cleanup: Option<(Instant, Instant, Option<CleanupReceipt>)>,
    done_receiver: std_mpsc::Receiver<RoundTripTerminal>,
    thread: Option<thread::JoinHandle<()>>,
    finished: Arc<AtomicBool>,
    terminal_result: Option<Result<RoundTripTerminal, RoundTripRuntimeError>>,
    pending_terminal: Option<RoundTripTerminal>,
}

struct CleanupRequest {
    outer_deadline: Instant,
    cleanup_deadline: Instant,
    response: std_mpsc::SyncSender<Result<(), RoundTripRuntimeError>>,
}

impl ActiveRoundTripRuntime for ProcessActiveRoundTrip {
    fn stop(
        &mut self,
        deadline: Instant,
        cleanup_deadline: Instant,
    ) -> Result<RoundTripTerminal, RoundTripRuntimeError> {
        if let Some(result) = self.terminal_result {
            return result;
        }
        let (deadline, cleanup_deadline) = self
            .pending_cleanup
            .as_ref()
            .map_or((deadline, cleanup_deadline), |pending| {
                (pending.0, pending.1)
            });
        if !self.finished.load(Ordering::Acquire) && self.pending_terminal.is_none() {
            if self.pending_cleanup.is_none() {
                let (response, receiver) = std_mpsc::sync_channel(1);
                match self.cleanup_sender.try_send(CleanupRequest {
                    outer_deadline: deadline,
                    cleanup_deadline,
                    response,
                }) {
                    Ok(()) => {
                        self.pending_cleanup = Some((deadline, cleanup_deadline, Some(receiver)))
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        return Err(RoundTripRuntimeError::StopFailed);
                    }
                }
            }
            self.stop_sender.send_replace(true);
            if let Some((original_outer, _, Some(response))) = self.pending_cleanup.as_ref() {
                match response
                    .recv_timeout(original_outer.saturating_duration_since(Instant::now()))
                {
                    Ok(Err(error)) => {
                        self.pending_cleanup = None;
                        return Err(error);
                    }
                    Err(std_mpsc::RecvTimeoutError::Timeout) => {
                        return Err(RoundTripRuntimeError::StopFailed);
                    }
                    Ok(Ok(())) | Err(std_mpsc::RecvTimeoutError::Disconnected) => {}
                }
                if let Some(pending) = self.pending_cleanup.as_mut() {
                    pending.2 = None;
                }
            }
        }
        if self.pending_terminal.is_none() {
            match self
                .done_receiver
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok(terminal) => self.pending_terminal = Some(terminal),
                Err(std_mpsc::RecvTimeoutError::Timeout) => {
                    return Err(RoundTripRuntimeError::StopFailed);
                }
                Err(std_mpsc::RecvTimeoutError::Disconnected) => {}
            }
        }
        while self
            .thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
        {
            if Instant::now() >= deadline {
                return Err(RoundTripRuntimeError::StopFailed);
            }
            thread::sleep(Duration::from_millis(1));
        }
        if let Some(thread) = self.thread.take() {
            if let Err(payload) = thread.join() {
                self.terminal_result = Some(Err(RoundTripRuntimeError::StopFailed));
                drop(payload);
                return Err(RoundTripRuntimeError::StopFailed);
            }
        }
        let result = self
            .pending_terminal
            .take()
            .ok_or(RoundTripRuntimeError::StopFailed);
        self.terminal_result = Some(result);
        self.pending_cleanup = None;
        result
    }
}

impl Drop for ProcessActiveRoundTrip {
    fn drop(&mut self) {
        let _ = self.stop_sender.send(true);
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_round_trip(
    admitted: crate::AdmittedDuplex,
    session_id: Uuid,
    progress: RoundTripProgress,
    mut stop: watch::Receiver<bool>,
    mut cleanup: mpsc::Receiver<CleanupRequest>,
    duplex_factory: Arc<dyn RoundTripDuplexFactory>,
    audio_factory: Arc<dyn RoundTripAudioWorkerFactory>,
    route_factory: Arc<dyn VirtualPeerRouteControllerFactory>,
    timeout: Duration,
    start_deadline: Instant,
) -> RoundTripTerminal {
    let started_at = std::time::Instant::now();
    let deadline = started_at + timeout;
    let cleanup_budget = (timeout / 4).min(crate::RUNTIME_CLEANUP_BUDGET);
    let active_deadline = deadline.checked_sub(cleanup_budget).unwrap_or(started_at);
    let observer = Arc::new(ProgressObserver::new(session_id, progress.clone()));
    let mut resources = RoundTripResources {
        duplex: None,
        audio: None,
        routes: None,
        frames: Vec::new(),
        route_restored: false,
        route_absent: false,
        writes_stopped: false,
        processes_finished: false,
    };

    let startup_deadline = start_deadline.min(active_deadline);
    let lifecycle = async {
        if Instant::now() >= startup_deadline {
            return Err(RoundTripProcessError::Duplex);
        }
        resources.routes = Some(route_factory.create());
        let physical_sink = admitted
            .snapshot()
            .devices
            .as_ref()
            .and_then(|devices| devices.sink.selected.as_ref())
            .map(|sink| sink.name.clone())
            .ok_or(RoundTripProcessError::Audio)?;
        if Instant::now() >= startup_deadline {
            return Err(RoundTripProcessError::Duplex);
        }
        let native_deadline = tokio::time::Instant::from_std(startup_deadline);
        let started = duplex_factory.start(admitted, observer.clone(), native_deadline);
        resources.duplex = Some(match started {
            Ok(runtime) => runtime,
            Err(failure) => {
                resources.duplex = failure.into_parts().1;
                return Err(RoundTripProcessError::Duplex);
            }
        });
        if Instant::now() >= startup_deadline {
            return Err(RoundTripProcessError::Duplex);
        }
        resources.audio = Some(audio_factory.create(session_id, &physical_sink)?);
        if Instant::now() >= startup_deadline {
            return Err(RoundTripProcessError::Duplex);
        }
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
    let outcome = match tokio::time::timeout(
        active_remaining,
        std::panic::AssertUnwindSafe(lifecycle).catch_unwind(),
    )
    .await
    {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(RoundTripProcessError::Audio),
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
    observer.freeze();
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| resources.clear_sensitive()))
        .is_err()
    {
        progress.fail(session_id, crate::RoundTripErrorCode::RuntimeFailed);
    }
    let mut response = cleanup.try_recv().ok();
    let mut cleanup_deadline = tokio::time::Instant::from_std(deadline)
        .min(tokio::time::Instant::now() + crate::RUNTIME_CLEANUP_BUDGET);
    loop {
        let result = teardown_attempt(
            &mut resources,
            &mut cleanup,
            &mut response,
            cleanup_deadline,
        )
        .await;
        if result.is_err() {
            progress.set_cleanup_pending(session_id, true);
        }
        if let Some(response) = response.take() {
            let _ = response
                .response
                .send(result.map_err(|_| RoundTripRuntimeError::StopFailed));
        }
        if result.is_ok() {
            return progress.terminal(outcome.is_ok());
        }
        response = match cleanup.recv().await {
            Some(response) => {
                cleanup_deadline = tokio::time::Instant::from_std(response.cleanup_deadline);
                Some(response)
            }
            None => std::future::pending().await,
        };
    }
}

async fn teardown_attempt(
    resources: &mut RoundTripResources,
    cleanup: &mut mpsc::Receiver<CleanupRequest>,
    request: &mut Option<CleanupRequest>,
    mut deadline: tokio::time::Instant,
) -> Result<(), RoundTripProcessError> {
    loop {
        if let Some(request) = request.as_ref() {
            deadline = deadline.min(tokio::time::Instant::from_std(
                request.cleanup_deadline.min(request.outer_deadline),
            ));
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(RoundTripProcessError::Audio);
        }
        tokio::select! {
            biased;
            incoming = cleanup.recv(), if request.is_none() && (!cleanup.is_closed() || !cleanup.is_empty()) => { *request = incoming; }
            result = tokio::time::timeout_at(deadline, std::panic::AssertUnwindSafe(resources.teardown(deadline)).catch_unwind()) => {
                return result.unwrap_or(Ok(Err(RoundTripProcessError::Audio))).unwrap_or(Err(RoundTripProcessError::Audio));
            }
        }
    }
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
        .ensure_absent()?;
    resources.route_absent = true;
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
    observer.validate_completed()
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
    frames: Vec<PcmFrame>,
    route_restored: bool,
    route_absent: bool,
    writes_stopped: bool,
    processes_finished: bool,
}

impl RoundTripResources {
    async fn teardown(
        &mut self,
        deadline: tokio::time::Instant,
    ) -> Result<(), RoundTripProcessError> {
        if tokio::time::Instant::now() >= deadline {
            return Err(RoundTripProcessError::Audio);
        }
        let mut failed = false;
        if !self.route_restored
            && let Some(routes) = self.routes.as_mut()
        {
            self.route_restored = routes.restore().is_ok();
            failed |= !self.route_restored;
        }
        if let Some(audio) = self.audio.as_mut() {
            if tokio::time::Instant::now() >= deadline {
                return Err(RoundTripProcessError::Audio);
            }
            if !self.writes_stopped {
                self.writes_stopped = audio.stop_writes().await.is_ok();
            }
            if !self.processes_finished {
                if tokio::time::Instant::now() >= deadline {
                    return Err(RoundTripProcessError::Audio);
                }
                self.processes_finished = audio.finish_processes().await.is_ok();
            }
            failed |= !self.writes_stopped || !self.processes_finished;
            if self.writes_stopped && self.processes_finished {
                self.audio = None;
            }
        }
        if !self.route_absent
            && let Some(routes) = self.routes.as_mut()
        {
            if tokio::time::Instant::now() >= deadline {
                return Err(RoundTripProcessError::Audio);
            }
            self.route_absent = routes.ensure_absent().is_ok();
            failed |= !self.route_absent;
        }
        if let Some(duplex) = self.duplex.as_mut() {
            if tokio::time::Instant::now() >= deadline {
                return Err(RoundTripProcessError::Audio);
            }
            if duplex.stop(deadline).is_ok() {
                self.duplex = None;
            } else {
                failed = true;
            }
        }
        if self.route_restored && self.route_absent {
            self.routes = None;
        }
        if failed {
            Err(RoundTripProcessError::Audio)
        } else {
            Ok(())
        }
    }

    fn clear_sensitive(&mut self) {
        zeroize_frames(&mut self.frames);
        if let Some(audio) = self.audio.as_mut() {
            audio.clear_sensitive();
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
    frozen: bool,
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
}

impl ProgressObserver {
    fn new(session_id: Uuid, progress: RoundTripProgress) -> Self {
        let (checkpoint_sender, _) = watch::channel(RoundTripCheckpoint::WaitingForSpeech);
        let (outgoing_terminal_sender, _) = watch::channel(false);
        Self {
            session_id,
            progress,
            state: Mutex::new(ObserverState {
                frozen: false,
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
        }
    }

    fn outgoing_terminal_receiver(&self) -> watch::Receiver<bool> {
        self.outgoing_terminal_sender.subscribe()
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
        let mut receiver = self.checkpoint_sender.subscribe();
        loop {
            {
                let state = lock_recovering(&self.state);
                if matches!(
                    state.checkpoint,
                    RoundTripCheckpoint::Failed | RoundTripCheckpoint::Stopped
                ) {
                    return Err(RoundTripProcessError::Progress);
                }
                if state.incoming_terminal {
                    return Ok(());
                }
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
        let mut receiver = self.checkpoint_sender.subscribe();
        let deadline = lock_recovering(&self.state)
            .incoming_playback_audible_until_ns
            .saturating_add(INCOMING_PLAYBACK_DRAIN_GRACE_NS);
        loop {
            let remaining_ns = deadline.saturating_sub(monotonic_ns());
            {
                let state = lock_recovering(&self.state);
                if matches!(
                    state.checkpoint,
                    RoundTripCheckpoint::Failed | RoundTripCheckpoint::Stopped
                ) {
                    return Err(RoundTripProcessError::Progress);
                }
                if remaining_ns == 0 {
                    return Ok(());
                }
            }
            tokio::select! {
                biased;
                changed = receiver.changed() => { changed.map_err(|_| RoundTripProcessError::Progress)?; }
                _ = wait_for_stop(stop) => return Err(RoundTripProcessError::Stopped),
                _ = tokio::time::sleep(Duration::from_nanos(remaining_ns)) => {}
            }
        }
    }

    fn freeze(&self) {
        lock_recovering(&self.state).frozen = true;
    }

    fn validate_completed(&self) -> Result<(), RoundTripProcessError> {
        let state = lock_recovering(&self.state);
        if state.checkpoint != RoundTripCheckpoint::RussianFirstAudio || !state.incoming_terminal {
            return Err(RoundTripProcessError::Progress);
        }
        Ok(())
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
            state.frozen = true;
            state.checkpoint = RoundTripCheckpoint::Failed;
            self.checkpoint_sender
                .send_replace(RoundTripCheckpoint::Failed);
            self.outgoing_terminal_sender.send_replace(true);
        }
    }
}

impl DuplexRuntimeObserver for ProgressObserver {
    fn reset_direction(&self, direction: AudioDirection) {
        let mut state = lock_recovering(&self.state);
        if !state.frozen && (direction == AudioDirection::Microphone || state.incoming_enabled) {
            self.fail_locked(&mut state);
        }
    }

    fn observe(&self, event: DuplexRuntimeEvent) {
        let mut state = lock_recovering(&self.state);
        if state.frozen {
            return;
        }
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
                self.checkpoint_sender.send_replace(state.checkpoint);
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
        admitted: crate::AdmittedDuplex,
        observer: Arc<dyn DuplexRuntimeObserver>,
        deadline: tokio::time::Instant,
    ) -> crate::DuplexStartResult {
        crate::DuplexRunner::start(
            &ProcessDuplexRunner::with_observer(self.config.clone(), observer),
            admitted,
            deadline,
        )
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
            let monitor = self.monitor.as_mut().ok_or(RoundTripProcessError::Audio)?;
            monitor
                .finish(drain.take_timeout())
                .await
                .map_err(|error| {
                    tracing::error!(
                        event = "round_trip_pcm_finish_failed",
                        stage = "english_monitor",
                        error = ?error
                    );
                    RoundTripProcessError::Audio
                })?;
            self.monitor = None;
            Ok(())
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
                .as_mut()
                .ok_or(RoundTripProcessError::Audio)?;
            let timeout = self.virtual_peer_drain.take_timeout();
            peer.finish(timeout).await.map_err(|error| {
                tracing::error!(
                    event = "round_trip_pcm_finish_failed",
                    stage = "virtual_peer",
                    error = ?error
                );
                RoundTripProcessError::Audio
            })?;
            self.virtual_peer = None;
            Ok(())
        })
    }

    fn stop_writes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            let mut failed = false;
            if let Some(monitor) = self.monitor.as_mut() {
                if monitor.stop().await.is_ok() {
                    self.monitor = None;
                } else {
                    failed = true;
                }
            }
            if let Some(peer) = self.virtual_peer.as_mut() {
                if peer.stop().await.is_ok() {
                    self.virtual_peer = None;
                } else {
                    failed = true;
                }
            }
            self.virtual_peer_drain.reset();
            if failed {
                Err(RoundTripProcessError::Audio)
            } else {
                Ok(())
            }
        })
    }

    fn finish_processes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            let mut failed = self.stop_writes().await.is_err();
            if let Some(capture) = self.capture.as_mut() {
                if capture.stop().await.is_ok() {
                    self.capture = None;
                } else {
                    failed = true;
                }
            }
            self.virtual_peer_drain.reset();
            if failed {
                Err(RoundTripProcessError::Audio)
            } else {
                Ok(())
            }
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
        self.routed = Some(capability.clone());
        self.watcher
            .route_virtual_peer(capability.clone())
            .map_err(|_| RoundTripProcessError::Route)?;
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

    fn restore(&mut self) -> Result<(), RoundTripProcessError> {
        if self.routed.is_none() {
            return Ok(());
        }
        self.watcher
            .restore_virtual_peer()
            .map_err(|_| RoundTripProcessError::Route)?;
        Ok(())
    }

    fn ensure_absent(&mut self) -> Result<(), RoundTripProcessError> {
        let Some(capability) = self.routed.as_ref() else {
            return Ok(());
        };
        let deadline = std::time::Instant::now() + VIRTUAL_PEER_DISCOVERY_TIMEOUT;
        loop {
            match self.discovery.ensure_absent(capability) {
                Ok(()) => return Ok(()),
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
pub(crate) mod tests {
    use super::*;

    fn reset_observer() -> ProgressObserver {
        let (session_id, progress) = crate::round_trip_runtime::tests::active_observer_progress();
        ProgressObserver::new(session_id, progress)
    }

    #[test]
    fn observer_reset_fails_meaningful_direction_and_seals_stale_evidence() {
        for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
            let observer = reset_observer();
            if direction == AudioDirection::Speaker {
                observer.begin_incoming();
            }
            let mut capture = observer.outgoing_terminal_receiver();
            observer.reset_direction(direction);
            let failed = *observer.checkpoint_sender.borrow();
            let capture_woken = capture.has_changed().unwrap() && *capture.borrow_and_update();
            observer.observe(DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Microphone,
                utterance_id: Uuid::new_v4(),
                capture_monotonic_ns: monotonic_ns(),
            });
            observer.reset_direction(direction);
            let state = lock_recovering(&observer.state);
            assert_eq!(failed, RoundTripCheckpoint::Failed);
            assert!(capture_woken);
            assert!(state.frozen && state.outgoing_utterance.is_none());
            assert_eq!(state.checkpoint, RoundTripCheckpoint::Failed);
            assert!(
                !capture.has_changed().unwrap(),
                "repeated reset must not notify twice"
            );
        }
    }

    #[test]
    fn observer_reset_ignores_disabled_speaker_and_frozen_cleanup() {
        let observer = reset_observer();
        let capture = observer.outgoing_terminal_receiver();
        observer.reset_direction(AudioDirection::Speaker);
        assert_eq!(
            *observer.checkpoint_sender.borrow(),
            RoundTripCheckpoint::WaitingForSpeech
        );
        assert!(!capture.has_changed().unwrap());
        observer.freeze();
        for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
            observer.reset_direction(direction);
        }
        assert_eq!(
            *observer.checkpoint_sender.borrow(),
            RoundTripCheckpoint::WaitingForSpeech
        );
        assert!(!capture.has_changed().unwrap());
    }

    #[tokio::test(start_paused = true)]
    async fn observer_reset_wakes_all_pending_evidence_waiters() {
        let observer = reset_observer();
        observer.begin_incoming();
        lock_recovering(&observer.state).incoming_playback_audible_until_ns =
            monotonic_ns() + 60_000_000_000;
        let (_stop, mut checkpoint_stop) = watch::channel(false);
        let mut terminal_stop = checkpoint_stop.clone();
        let mut drain_stop = checkpoint_stop.clone();
        let checkpoint =
            observer.wait_for(RoundTripCheckpoint::EnglishFirstAudio, &mut checkpoint_stop);
        let terminal = observer.wait_for_incoming_terminal(&mut terminal_stop);
        let drain = observer.wait_for_incoming_playback_drain(&mut drain_stop);
        tokio::pin!(checkpoint, terminal, drain);
        assert!(futures_util::poll!(&mut checkpoint).is_pending());
        assert!(futures_util::poll!(&mut terminal).is_pending());
        assert!(futures_util::poll!(&mut drain).is_pending());
        observer.reset_direction(AudioDirection::Speaker);
        let observed = (
            futures_util::poll!(&mut checkpoint),
            futures_util::poll!(&mut terminal),
            futures_util::poll!(&mut drain),
        );
        let failed = std::task::Poll::Ready(Err(RoundTripProcessError::Progress));
        assert_eq!(observed, (failed, failed, failed));
    }

    #[tokio::test(start_paused = true)]
    async fn observer_reset_failure_precedes_stale_terminal_and_elapsed_playback() {
        let observer = reset_observer();
        observer.begin_incoming();
        lock_recovering(&observer.state).incoming_terminal = true;
        observer.reset_direction(AudioDirection::Speaker);
        let (_stop, mut stop) = watch::channel(false);
        let terminal = observer.wait_for_incoming_terminal(&mut stop).await;
        let drain = observer.wait_for_incoming_playback_drain(&mut stop).await;
        assert_eq!(
            (terminal, drain),
            (
                Err(RoundTripProcessError::Progress),
                Err(RoundTripProcessError::Progress)
            )
        );
    }

    #[tokio::test(start_paused = true)]
    async fn observer_reset_failure_wins_when_playback_timer_is_already_ready() {
        let observer = reset_observer();
        observer.begin_incoming();
        lock_recovering(&observer.state).incoming_playback_audible_until_ns =
            monotonic_ns() + 1_000_000_000;
        let (_stop, mut stop) = watch::channel(false);
        let drain = observer.wait_for_incoming_playback_drain(&mut stop);
        tokio::pin!(drain);
        assert!(futures_util::poll!(&mut drain).is_pending());
        tokio::time::advance(Duration::from_secs(2)).await;
        observer.reset_direction(AudioDirection::Speaker);
        assert_eq!(
            futures_util::poll!(&mut drain),
            std::task::Poll::Ready(Err(RoundTripProcessError::Progress))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn observer_reset_stopped_checkpoint_precedes_ready_wait_results() {
        let observer = reset_observer();
        {
            let mut state = lock_recovering(&observer.state);
            state.incoming_terminal = true;
            state.checkpoint = RoundTripCheckpoint::Stopped;
        }
        observer
            .checkpoint_sender
            .send_replace(RoundTripCheckpoint::Stopped);
        let (_stop, mut stop) = watch::channel(false);
        let terminal = observer.wait_for_incoming_terminal(&mut stop).await;
        let drain = observer.wait_for_incoming_playback_drain(&mut stop).await;
        assert_eq!(
            (terminal, drain),
            (
                Err(RoundTripProcessError::Progress),
                Err(RoundTripProcessError::Progress)
            )
        );
    }

    #[tokio::test(start_paused = true)]
    async fn observer_terminal_same_checkpoint_wakes_subscribed_waiter() {
        let observer = reset_observer();
        observer.begin_incoming();
        let utterance_id = Uuid::new_v4();
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Speaker,
            utterance_id,
            capture_monotonic_ns: monotonic_ns(),
        });
        let before = *observer.checkpoint_sender.borrow();
        let (_stop, mut stop) = watch::channel(false);
        let terminal = observer.wait_for_incoming_terminal(&mut stop);
        tokio::pin!(terminal);
        assert!(futures_util::poll!(&mut terminal).is_pending());
        observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
            direction: AudioDirection::Speaker,
            utterance_id,
        });
        assert_eq!(*observer.checkpoint_sender.borrow(), before);
        assert!(lock_recovering(&observer.state).incoming_terminal);
        assert_eq!(
            futures_util::poll!(&mut terminal),
            std::task::Poll::Ready(Ok(()))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn observer_terminal_before_subscription_is_observed() {
        let observer = reset_observer();
        observer.begin_incoming();
        let utterance_id = Uuid::new_v4();
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Speaker,
            utterance_id,
            capture_monotonic_ns: monotonic_ns(),
        });
        let before = *observer.checkpoint_sender.borrow();
        observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
            direction: AudioDirection::Speaker,
            utterance_id,
        });
        let (_stop, mut stop) = watch::channel(false);
        let terminal = observer.wait_for_incoming_terminal(&mut stop);
        tokio::pin!(terminal);
        assert_eq!(*observer.checkpoint_sender.borrow(), before);
        assert!(lock_recovering(&observer.state).incoming_terminal);
        assert_eq!(
            futures_util::poll!(&mut terminal),
            std::task::Poll::Ready(Ok(()))
        );
    }

    fn panicked_worker() -> (ProcessActiveRoundTrip, mpsc::Receiver<CleanupRequest>) {
        panicked_worker_with_payload(None)
    }

    struct PanickingPayloadDrop(Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for PanickingPayloadDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
            panic!("resource-free panic payload destructor");
        }
    }

    fn panicked_worker_with_payload(
        drops: Option<Arc<std::sync::atomic::AtomicUsize>>,
    ) -> (ProcessActiveRoundTrip, mpsc::Receiver<CleanupRequest>) {
        let (stop_sender, _) = watch::channel(false);
        let (cleanup_sender, cleanup) = mpsc::channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            done_sender.send(RoundTripTerminal::Completed).unwrap();
            if let Some(drops) = drops {
                std::panic::panic_any(PanickingPayloadDrop(drops));
            }
            panic!("resource-free inner worker panic after terminal send");
        });
        (
            ProcessActiveRoundTrip {
                stop_sender,
                cleanup_sender,
                pending_cleanup: None,
                done_receiver,
                thread: Some(thread),
                finished: Arc::new(AtomicBool::new(true)),
                terminal_result: None,
                pending_terminal: None,
            },
            cleanup,
        )
    }

    pub(crate) fn panicked_inner_runtime() -> Box<dyn ActiveRoundTripRuntime> {
        Box::new(panicked_worker().0)
    }

    pub(crate) fn payload_drop_panicked_inner_runtime(
        drops: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Box<dyn ActiveRoundTripRuntime> {
        Box::new(panicked_worker_with_payload(Some(drops)).0)
    }

    fn pre_receipt_worker(
        drops: Option<Arc<std::sync::atomic::AtomicUsize>>,
    ) -> ProcessActiveRoundTrip {
        let (stop_sender, _) = watch::channel(false);
        let (cleanup_sender, cleanup_receiver) = mpsc::channel(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            let _cleanup = cleanup_receiver;
            let _done = done_sender;
            if let Some(drops) = drops {
                std::panic::panic_any(PanickingPayloadDrop(drops));
            }
            panic!("resource-free worker panic before terminal receipt");
        });
        ProcessActiveRoundTrip {
            stop_sender,
            cleanup_sender,
            pending_cleanup: None,
            done_receiver,
            thread: Some(thread),
            finished: Arc::new(AtomicBool::new(false)),
            terminal_result: None,
            pending_terminal: None,
        }
    }

    fn drain_fixture_worker(owner: &mut ProcessActiveRoundTrip) {
        if let Some(thread) = owner.thread.take() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(thread.join())));
        }
    }

    pub(crate) fn pre_receipt_inner_runtime(
        drops: Option<Arc<std::sync::atomic::AtomicUsize>>,
    ) -> (Box<dyn ActiveRoundTripRuntime>, Arc<AtomicBool>) {
        struct Fixture {
            inner: ProcessActiveRoundTrip,
            joined: Arc<AtomicBool>,
        }
        impl ActiveRoundTripRuntime for Fixture {
            fn stop(
                &mut self,
                outer: Instant,
                cleanup: Instant,
            ) -> Result<RoundTripTerminal, RoundTripRuntimeError> {
                self.inner.stop(outer, cleanup)
            }
        }
        impl Drop for Fixture {
            fn drop(&mut self) {
                self.joined
                    .store(self.inner.thread.is_none(), Ordering::SeqCst);
                drain_fixture_worker(&mut self.inner);
            }
        }
        let joined = Arc::new(AtomicBool::new(false));
        (
            Box::new(Fixture {
                inner: pre_receipt_worker(drops),
                joined: joined.clone(),
            }),
            joined,
        )
    }

    fn assert_pre_receipt_panic(payload_drop: bool) {
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut owner = pre_receipt_worker(payload_drop.then(|| drops.clone()));
        let admitted = Instant::now();
        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owner.stop(
                admitted + Duration::from_secs(2),
                admitted + Duration::from_secs(1),
            )
        }));
        let joined = owner.thread.is_none();
        let sticky = owner.terminal_result;
        let second = owner.stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
        );
        let expired = owner.stop(admitted, admitted);
        let pending_receipt = owner.pending_terminal;
        let drops_before_drain = drops.load(Ordering::SeqCst);
        drain_fixture_worker(&mut owner);
        assert!(
            joined,
            "disconnected done channel must reach the actual worker join"
        );
        assert_eq!(sticky, Some(Err(RoundTripRuntimeError::StopFailed)));
        if payload_drop {
            assert!(first.is_err());
            assert_eq!(drops_before_drain, 1);
        } else {
            assert!(matches!(first, Ok(Err(RoundTripRuntimeError::StopFailed))));
        }
        assert_eq!(second, Err(RoundTripRuntimeError::StopFailed));
        assert_eq!(expired, Err(RoundTripRuntimeError::StopFailed));
        assert!(pending_receipt.is_none());
        assert!(owner.cleanup_sender.is_closed());
    }

    #[test]
    fn pre_receipt_panic_is_joined_and_sticky() {
        assert_pre_receipt_panic(false);
    }

    #[test]
    fn pre_receipt_payload_drop_panic_is_joined_and_sticky() {
        assert_pre_receipt_panic(true);
    }

    #[test]
    fn disconnected_live_worker_keeps_bounded_join_pending() {
        let (stop_sender, _) = watch::channel(false);
        let (cleanup_sender, cleanup) = mpsc::channel(1);
        drop(cleanup);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let (release, released) = std_mpsc::sync_channel(1);
        let (ready, readiness) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            drop(done_sender);
            ready.send(()).unwrap();
            released.recv_timeout(Duration::from_secs(2)).unwrap();
        });
        let mut owner = ProcessActiveRoundTrip {
            stop_sender,
            cleanup_sender,
            pending_cleanup: None,
            done_receiver,
            thread: Some(thread),
            finished: Arc::new(AtomicBool::new(false)),
            terminal_result: None,
            pending_terminal: None,
        };
        readiness.recv_timeout(Duration::from_secs(2)).unwrap();
        let expired = Instant::now();
        let first = owner.stop(expired, expired);
        let retained = owner.thread.is_some() && owner.terminal_result.is_none();
        release.send(()).unwrap();
        let second = owner.stop(Instant::now() + Duration::from_secs(1), Instant::now());
        let joined = owner.thread.is_none();
        drain_fixture_worker(&mut owner);
        assert_eq!(first, Err(RoundTripRuntimeError::StopFailed));
        assert!(retained);
        assert_eq!(second, Err(RoundTripRuntimeError::StopFailed));
        assert!(
            joined,
            "normal join without a receipt must become sticky failure"
        );
    }

    #[test]
    fn payload_drop_panic_preserves_confirmed_inner_join_failure() {
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (mut owner, mut cleanup) = panicked_worker_with_payload(Some(drops.clone()));
        let admitted = Instant::now();
        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owner.stop(
                admitted + Duration::from_secs(2),
                admitted + Duration::from_secs(1),
            )
        }));
        let joined = owner.thread.is_none();
        let second = owner.stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
        );
        let expired = owner.stop(admitted, admitted);
        if let Some(thread) = owner.thread.take() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(thread.join())));
        }
        assert!(
            first.is_err(),
            "the first Stop must unwind from payload Drop"
        );
        assert!(joined, "the actual worker must already be joined");
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(second, Err(RoundTripRuntimeError::StopFailed));
        assert_eq!(expired, Err(RoundTripRuntimeError::StopFailed));
        assert!(cleanup.try_recv().is_err(), "no duplicate cleanup request");
    }

    #[test]
    fn confirmed_inner_join_panic_never_becomes_a_successful_terminal_receipt() {
        let (mut owner, mut cleanup) = panicked_worker();
        let admitted = Instant::now();
        let first = owner.stop(
            admitted + Duration::from_secs(2),
            admitted + Duration::from_secs(1),
        );
        let joined = owner.thread.is_none();
        let second = owner.stop(
            Instant::now() + Duration::from_secs(2),
            Instant::now() + Duration::from_secs(1),
        );
        let third = owner.stop(Instant::now(), Instant::now());
        if let Some(thread) = owner.thread.take() {
            let _ = thread.join();
        }
        assert!(
            joined,
            "first Stop must observe the actual failed OS-thread join"
        );
        assert_eq!(first, Err(RoundTripRuntimeError::StopFailed));
        assert_eq!(second, Err(RoundTripRuntimeError::StopFailed));
        assert_eq!(third, Err(RoundTripRuntimeError::StopFailed));
        assert!(
            cleanup.try_recv().is_err(),
            "terminal join failure must not enqueue cleanup"
        );
    }

    struct DeadlineDuplex(Arc<Mutex<Vec<tokio::time::Instant>>>);

    impl ActiveDuplexRuntime for DeadlineDuplex {
        fn stop(&mut self, deadline: tokio::time::Instant) -> Result<(), DuplexRuntimeError> {
            self.0.lock().unwrap().push(deadline);
            Ok(())
        }
    }

    struct GatedCleanupAudio {
        entered: Arc<AtomicBool>,
        release: Arc<tokio::sync::Notify>,
    }

    impl RoundTripAudioWorker for GatedCleanupAudio {
        fn capture_english_utterance<'a>(
            &'a mut self,
            _: &'a mut watch::Receiver<bool>,
            _: &'a mut watch::Receiver<bool>,
        ) -> RoundTripWorkerFuture<'a, Vec<PcmFrame>> {
            Box::pin(async { Err(RoundTripProcessError::Audio) })
        }
        fn monitor_english<'a>(
            &'a mut self,
            _: &'a [PcmFrame],
            _: &'a mut watch::Receiver<bool>,
        ) -> RoundTripWorkerFuture<'a, ()> {
            Box::pin(async { Err(RoundTripProcessError::Audio) })
        }
        fn spawn_virtual_peer(&mut self) -> Result<ProcessIdentity, RoundTripProcessError> {
            Err(RoundTripProcessError::Audio)
        }
        fn write_virtual_peer_frame<'a>(
            &'a mut self,
            _: &'a PcmFrame,
        ) -> RoundTripWorkerFuture<'a, (u64, StreamPcmFormat, usize, [u8; 32])> {
            Box::pin(async { Err(RoundTripProcessError::Audio) })
        }
        fn finish_virtual_peer<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }
        fn stop_writes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
            Box::pin(async move {
                self.entered.store(true, Ordering::Release);
                self.release.notified().await;
                Ok(())
            })
        }
        fn finish_processes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }
        fn clear_sensitive(&mut self) {}
    }

    #[tokio::test(start_paused = true)]
    async fn accepted_stop_caps_pending_autonomous_cleanup_and_preserves_retry_ownership() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(tokio::sync::Notify::new());
        let mut resources = RoundTripResources {
            duplex: Some(Box::new(DeadlineDuplex(calls.clone()))),
            audio: Some(Box::new(GatedCleanupAudio {
                entered: entered.clone(),
                release: release.clone(),
            })),
            routes: None,
            frames: Vec::new(),
            route_restored: false,
            route_absent: false,
            writes_stopped: false,
            processes_finished: false,
        };
        let admitted = tokio::time::Instant::now();
        let autonomous_deadline = admitted + crate::RUNTIME_CLEANUP_BUDGET;
        let earlier_cleanup = admitted + Duration::from_millis(20);
        let outer_deadline = admitted + Duration::from_secs(10);
        let (sender, mut cleanup) = mpsc::channel(1);
        let (response, _receipt) = std_mpsc::sync_channel(1);
        let mut request = None;
        let mut attempt = Box::pin(teardown_attempt(
            &mut resources,
            &mut cleanup,
            &mut request,
            autonomous_deadline,
        ));
        assert!(futures_util::poll!(attempt.as_mut()).is_pending());
        assert!(entered.load(Ordering::Acquire));
        sender
            .send(CleanupRequest {
                outer_deadline: outer_deadline.into_std(),
                cleanup_deadline: earlier_cleanup.into_std(),
                response,
            })
            .await
            .unwrap();
        assert!(futures_util::poll!(attempt.as_mut()).is_pending());
        tokio::time::advance(Duration::from_millis(20)).await;
        let capped = matches!(
            futures_util::poll!(attempt.as_mut()),
            std::task::Poll::Ready(Err(_))
        );
        drop(attempt);
        let consumed_original_request = request
            .as_ref()
            .map(|r| (r.outer_deadline, r.cleanup_deadline));
        let retained = resources.audio.is_some()
            && resources.duplex.is_some()
            && calls.lock().unwrap().is_empty();
        let _old_request = request.take().or_else(|| cleanup.try_recv().ok());
        let retry_deadline = tokio::time::Instant::now() + crate::RUNTIME_CLEANUP_BUDGET;
        let (response, _receipt) = std_mpsc::sync_channel(1);
        request = Some(CleanupRequest {
            outer_deadline: (retry_deadline + Duration::from_secs(2)).into_std(),
            cleanup_deadline: retry_deadline.into_std(),
            response,
        });
        release.notify_one();
        let retried =
            teardown_attempt(&mut resources, &mut cleanup, &mut request, retry_deadline).await;
        assert!(
            capped,
            "accepted earlier cleanup deadline must cap the pending async phase"
        );
        assert_eq!(
            consumed_original_request,
            Some((outer_deadline.into_std(), earlier_cleanup.into_std()))
        );
        assert!(retained);
        assert!(retried.is_ok());
        assert!(resources.audio.is_none() && resources.duplex.is_none());
        assert_eq!(*calls.lock().unwrap(), vec![retry_deadline]);
    }

    #[tokio::test(start_paused = true)]
    async fn teardown_forwards_original_deadline_and_skips_expired_native_phase() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut resources = RoundTripResources {
            duplex: Some(Box::new(DeadlineDuplex(calls.clone()))),
            audio: None,
            routes: None,
            frames: Vec::new(),
            route_restored: false,
            route_absent: false,
            writes_stopped: false,
            processes_finished: false,
        };
        let admitted = tokio::time::Instant::now();
        let deadline = admitted + crate::RUNTIME_CLEANUP_BUDGET;
        tokio::time::advance(crate::RUNTIME_CLEANUP_BUDGET).await;
        let expired = resources.teardown(deadline).await;
        let retained = resources.duplex.is_some();
        let calls_before_retry = calls.lock().unwrap().clone();
        let retry_deadline = tokio::time::Instant::now() + crate::RUNTIME_CLEANUP_BUDGET;
        let retry = resources.teardown(retry_deadline).await;
        assert!(expired.is_err());
        assert!(retained);
        assert!(calls_before_retry.is_empty());
        assert!(retry.is_ok());
        assert_eq!(*calls.lock().unwrap(), vec![retry_deadline]);
        assert!(resources.duplex.is_none());
    }

    #[test]
    fn stop_timeout_during_startup_keeps_one_attempt_and_joins_the_same_thread() {
        let (stop_sender, stop_receiver) = watch::channel(false);
        let (cleanup_sender, mut cleanup_receiver) = mpsc::channel::<CleanupRequest>(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let (release, released) = std_mpsc::sync_channel(1);
        let finished = Arc::new(AtomicBool::new(false));
        let thread_finished = Arc::clone(&finished);
        let original_outer = Instant::now();
        let original_cleanup = original_outer - Duration::from_millis(5);
        let (observed, observation) = std_mpsc::sync_channel(1);
        let thread = thread::spawn(move || {
            let _stop = stop_receiver;
            let _ = released.recv();
            if let Ok(request) = cleanup_receiver.try_recv() {
                let _ = observed.send((request.outer_deadline, request.cleanup_deadline));
                let _ = request.response.send(Ok(()));
            }
            assert!(
                cleanup_receiver.try_recv().is_err(),
                "duplicate cleanup attempt"
            );
            thread_finished.store(true, Ordering::Release);
            let _ = done_sender.send(RoundTripTerminal::Stopped);
        });
        let mut owner = ProcessActiveRoundTrip {
            stop_sender,
            cleanup_sender,
            pending_cleanup: None,
            done_receiver,
            thread: Some(thread),
            finished,
            terminal_result: None,
            pending_terminal: None,
        };
        let started = Instant::now();
        let first = owner.stop(original_outer, original_cleanup);
        let retained = owner.thread.is_some() && owner.pending_cleanup.is_some();
        let second = owner.stop(Instant::now(), Instant::now());
        let _ = release.send(());
        let mut cleaned = owner.stop(
            Instant::now() + Duration::from_secs(2),
            Instant::now() + Duration::from_secs(1),
        );
        let observation = observation.recv_timeout(Duration::from_secs(1)).unwrap();
        while cleaned.is_err() && started.elapsed() < Duration::from_secs(2) {
            thread::yield_now();
            cleaned = owner.stop(
                Instant::now() + Duration::from_secs(1),
                Instant::now() + Duration::from_millis(500),
            );
        }
        let joined_by_stop = owner.thread.is_none();
        if let Some(thread) = owner.thread.take() {
            thread.join().unwrap();
        }
        assert_eq!(first, Err(RoundTripRuntimeError::StopFailed));
        assert_eq!(second, Err(RoundTripRuntimeError::StopFailed));
        assert!(retained);
        assert_eq!(observation, (original_outer, original_cleanup));
        assert_eq!(cleaned, Ok(RoundTripTerminal::Stopped));
        assert!(joined_by_stop);
        assert_eq!(
            owner.stop(Instant::now(), Instant::now()),
            Ok(RoundTripTerminal::Stopped)
        );
    }

    #[test]
    fn completed_failed_cleanup_receipt_allows_exactly_one_new_admitted_pair() {
        let (stop_sender, stop_receiver) = watch::channel(false);
        let (cleanup_sender, mut cleanup_receiver) = mpsc::channel::<CleanupRequest>(1);
        let (done_sender, done_receiver) = std_mpsc::sync_channel(1);
        let (observed, observation) = std_mpsc::sync_channel(2);
        let finished = Arc::new(AtomicBool::new(false));
        let thread_finished = finished.clone();
        let thread = thread::spawn(move || {
            let _stop = stop_receiver;
            for failed in [true, false] {
                let request = cleanup_receiver.blocking_recv().unwrap();
                observed
                    .send((request.outer_deadline, request.cleanup_deadline))
                    .unwrap();
                request
                    .response
                    .send(if failed {
                        Err(RoundTripRuntimeError::StopFailed)
                    } else {
                        Ok(())
                    })
                    .unwrap();
            }
            let duplicate = cleanup_receiver.try_recv().is_ok();
            thread_finished.store(true, Ordering::Release);
            done_sender.send(RoundTripTerminal::Stopped).unwrap();
            assert!(!duplicate);
        });
        let mut owner = ProcessActiveRoundTrip {
            stop_sender,
            cleanup_sender,
            pending_cleanup: None,
            done_receiver,
            thread: Some(thread),
            finished,
            terminal_result: None,
            pending_terminal: None,
        };
        let first_admitted = Instant::now();
        let first_pair = (
            first_admitted + Duration::from_secs(10),
            first_admitted + crate::RUNTIME_CLEANUP_BUDGET,
        );
        let first = owner.stop(first_pair.0, first_pair.1);
        let retained = owner.pending_cleanup.is_none() && owner.thread.is_some();
        let retry_admitted = Instant::now();
        let retry_pair = (
            retry_admitted + Duration::from_secs(10),
            retry_admitted + crate::RUNTIME_CLEANUP_BUDGET,
        );
        let retry = owner.stop(retry_pair.0, retry_pair.1);
        if let Some(thread) = owner.thread.take() {
            thread.join().unwrap();
        }
        let observations: Vec<_> = observation.try_iter().collect();
        assert_eq!(first, Err(RoundTripRuntimeError::StopFailed));
        assert!(retained);
        assert_eq!(retry, Ok(RoundTripTerminal::Stopped));
        assert_eq!(observations, vec![first_pair, retry_pair]);
        assert!(retry_pair.1 > first_pair.1);
        assert_eq!(
            owner.stop(Instant::now(), Instant::now()),
            Ok(RoundTripTerminal::Stopped)
        );
    }

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
