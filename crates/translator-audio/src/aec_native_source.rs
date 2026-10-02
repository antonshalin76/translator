//! Native ALSA acquisition transport. Process and hardware custody remain with
//! the daemon's backend guardian, not these channels or command handles.

use std::{
    collections::BTreeMap,
    os::fd::BorrowedFd,
    process::Stdio,
    sync::{Arc, Mutex},
};

use rustix::io::{FdFlags, fcntl_setfd};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{mpsc, oneshot},
    time::Instant,
};

use crate::{
    AEC_FIXTURE_DBFS, AEC_SAMPLES_PER_POWER_WINDOW, AecChannelFrame, AecDeviceMetadata,
    AecGraphIdentity, AecMeasurementBinding, AecObservationEvidence, AecPairedFrame,
    AecPowerAcquisition, AecPowerWindow, AecProofReadyMeasurement, AecSampleEvidence,
    AecSampleVerifier, AecValidationInput, PcmFrame, StreamPcmFormat, evaluate_aec,
};

const BLOCK: usize = 480;
const MAX_PACKET: usize = 32 * 1024;
const MAX_COMMAND_PCM: usize = 1_048_576;
const FRESHNESS: std::time::Duration = std::time::Duration::from_millis(200);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NativeAecError {
    #[error("native acquisition authority is invalid")]
    Authority,
    #[error("native acquisition failed or lost custody")]
    Source,
    #[error("native acquisition evidence is inconsistent")]
    Provenance,
    #[error("native acquisition deadline expired")]
    Deadline,
    #[error("native acquisition receiver is already owned")]
    AlreadyOwned,
    #[error("native acquisition queue overflowed")]
    Overflow,
    #[error("native acquisition contains invalid or clipped PCM")]
    Pcm,
}

pub struct NativeAecLaunchAuthority {
    pub scope_unit: String,
    pub session_id: u64,
    pub lifecycle_fd: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeAecIdentity {
    pub graph: AecGraphIdentity,
    pub card: u32,
    pub card_id: String,
    pub pcm_name: String,
    pub capture_channels: u8,
    pub playback_channels: u8,
    pub capture_buffer: u32,
    pub playback_buffer: u32,
    pub source_port: String,
    pub sink_port: String,
    pub control_fingerprint: String,
    pub capture_gains: Vec<u32>,
    pub playback_gains: Vec<u32>,
    pub capture_muted: bool,
    pub playback_muted: bool,
    /// Actual ALSA control-range percentage, not measured acoustic loudness.
    pub playback_volume_percent: u8,
    /// Host observation of the first complete ADC read, not hardware onset.
    pub capture_origin_monotonic_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeCaptureOrigin {
    Physical,
    InjectedPositive { fixture_sha256: String },
}

#[derive(Debug)]
pub struct NativeCapturedFrame {
    pub frame: PcmFrame,
    pub graph: AecGraphIdentity,
    pub adc_start: u64,
    pub adc_end: u64,
    pub origin: NativeCaptureOrigin,
    pub capture_read_bracket_ns: u64,
}

#[derive(Debug)]
pub enum NativeAecEvent {
    Ready(Box<NativeAecIdentity>),
    Progress { adc_frames: u64 },
    Poisoned,
}

#[derive(Debug, Clone, Default)]
pub struct NativeAecDiagnostics {
    pub settling_frames: u64,
    pub settling_clipped_samples: Vec<u64>,
    pub settling_peak_s16: Vec<u32>,
    pub capture_unknown_timestamps: u64,
    pub playback_unknown_timestamps: u64,
    pub reference_uncertainty_frames: u64,
    pub maximum_read_bracket_ns: u64,
}

#[derive(Default)]
struct Health {
    identity: Option<NativeAecIdentity>,
    fresh: Option<Instant>,
    poisoned: bool,
    diagnostics: NativeAecDiagnostics,
    capture_paused: bool,
    adc_end: u64,
    positive_enqueued_end: u64,
    positive_consumed_end: u64,
}

struct Request {
    command: u32,
    pcm: Vec<u8>,
    hash: String,
    deadline: Instant,
    ack: oneshot::Sender<Result<(), NativeAecError>>,
}

#[derive(Clone)]
pub struct NativeAecHandle {
    health: Arc<Mutex<Health>>,
    commands: mpsc::Sender<Request>,
}

impl NativeAecHandle {
    pub fn identity(&self) -> Option<NativeAecIdentity> {
        self.health
            .lock()
            .ok()
            .filter(|h| !h.poisoned)?
            .identity
            .clone()
    }

    pub fn check_fresh(&self, deadline: Instant) -> Result<NativeAecIdentity, NativeAecError> {
        let now = Instant::now();
        let health = self.health.lock().map_err(|_| NativeAecError::Source)?;
        if now >= deadline
            || health.poisoned
            || health
                .fresh
                .is_none_or(|t| now.saturating_duration_since(t) > FRESHNESS)
        {
            return Err(NativeAecError::Source);
        }
        health.identity.clone().ok_or(NativeAecError::Source)
    }

    pub fn diagnostics(&self) -> NativeAecDiagnostics {
        self.health
            .lock()
            .map(|h| h.diagnostics.clone())
            .unwrap_or_default()
    }

    async fn command(
        &self,
        command: u32,
        pcm: &[u8],
        hash: &str,
        deadline: Instant,
    ) -> Result<(), NativeAecError> {
        self.check_fresh(deadline)?;
        if pcm.len() > MAX_COMMAND_PCM || pcm.len() % 2 != 0 {
            return Err(NativeAecError::Pcm);
        }
        let (ack, receive) = oneshot::channel();
        let result = async {
            tokio::time::timeout_at(
                deadline,
                self.commands.send(Request {
                    command,
                    pcm: pcm.to_vec(),
                    hash: hash.into(),
                    deadline,
                    ack,
                }),
            )
            .await
            .map_err(|_| NativeAecError::Deadline)?
            .map_err(|_| NativeAecError::Source)?;
            tokio::time::timeout_at(deadline, receive)
                .await
                .map_err(|_| NativeAecError::Deadline)?
                .map_err(|_| NativeAecError::Source)?
        }
        .await;
        if result.is_err() {
            if let Ok(mut health) = self.health.lock() {
                health.poisoned = true;
            }
        }
        result
    }

    pub async fn start_measurement(&self, deadline: Instant) -> Result<(), NativeAecError> {
        self.command(1, &[], "", deadline).await
    }
    pub async fn inject_positive(
        &self,
        pcm16k: &[u8],
        fixture_sha256: &str,
        deadline: Instant,
    ) -> Result<(), NativeAecError> {
        let actual_hash: String = Sha256::digest(pcm16k)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if pcm16k.is_empty() || pcm16k.len() % 640 != 0 || actual_hash != fixture_sha256 {
            return Err(NativeAecError::Pcm);
        }
        self.command(2, pcm16k, fixture_sha256, deadline).await
    }
    /// Only native injection/capture FIFO drain; provider drain is independent.
    pub async fn positive_drained(&self, deadline: Instant) -> Result<(), NativeAecError> {
        self.command(7, &[], "", deadline).await
    }
    pub async fn write_playback(
        &self,
        pcm16k: &[u8],
        deadline: Instant,
    ) -> Result<(), NativeAecError> {
        self.write_playback_with_gain(pcm16k, 100, deadline).await
    }

    /// Applies the existing mix owner's gain only to runtime output, never
    /// microphone evidence or the separately owned calibration challenge.
    pub async fn write_playback_with_gain(
        &self,
        pcm16k: &[u8],
        gain_percent: u8,
        deadline: Instant,
    ) -> Result<(), NativeAecError> {
        if pcm16k.is_empty() || pcm16k.len() % 640 != 0 {
            return Err(NativeAecError::Pcm);
        }
        let scaled = runtime_output_gain(pcm16k, gain_percent)?;
        self.command(3, &scaled, "", deadline).await
    }
    pub async fn stop_playback(&self, deadline: Instant) -> Result<(), NativeAecError> {
        self.command(4, &[], "", deadline).await
    }
    pub async fn pause_capture(&self, deadline: Instant) -> Result<(), NativeAecError> {
        self.command(5, &[], "", deadline).await
    }
    pub async fn resume_capture(&self, deadline: Instant) -> Result<(), NativeAecError> {
        self.command(6, &[], "", deadline).await
    }
}

fn runtime_output_gain(pcm16k: &[u8], gain_percent: u8) -> Result<Vec<u8>, NativeAecError> {
    if gain_percent > 100 || pcm16k.len() % 2 != 0 {
        return Err(NativeAecError::Pcm);
    }
    if gain_percent == 100 {
        return Ok(pcm16k.to_vec());
    }
    Ok(pcm16k
        .chunks_exact(2)
        .flat_map(|bytes| {
            let value = i16::from_le_bytes([bytes[0], bytes[1]]) as i32 * gain_percent as i32 / 100;
            (value as i16).to_le_bytes()
        })
        .collect())
}

pub struct NativeCaptureReceiver {
    receive: mpsc::Receiver<NativeCapturedFrame>,
    health: Arc<Mutex<Health>>,
}

impl NativeCaptureReceiver {
    /// Call only after native capture is paused and the runtime consumer parked.
    /// This is lifecycle disposal, never scored-window continuity repair.
    pub fn discard_paused(&mut self) -> Result<(), NativeAecError> {
        let health = self.health.lock().map_err(|_| NativeAecError::Source)?;
        if health.poisoned || !health.capture_paused {
            return Err(NativeAecError::Source);
        }
        while self.receive.try_recv().is_ok() {}
        Ok(())
    }
    pub async fn next_frame(
        &mut self,
        deadline: Instant,
    ) -> Result<NativeCapturedFrame, NativeAecError> {
        if self
            .health
            .lock()
            .map_err(|_| NativeAecError::Source)?
            .poisoned
        {
            return Err(NativeAecError::Source);
        }
        let frame = tokio::time::timeout_at(deadline, self.receive.recv())
            .await
            .map_err(|_| NativeAecError::Deadline)?
            .ok_or(NativeAecError::Source)?;
        let mut health = self.health.lock().map_err(|_| NativeAecError::Source)?;
        if health.poisoned
            || health
                .identity
                .as_ref()
                .is_none_or(|identity| identity.graph != frame.graph)
            || health
                .fresh
                .is_none_or(|stamp| Instant::now().saturating_duration_since(stamp) > FRESHNESS)
        {
            return Err(NativeAecError::Source);
        }
        if matches!(frame.origin, NativeCaptureOrigin::InjectedPositive { .. }) {
            health.positive_consumed_end = frame.adc_end;
        }
        Ok(frame)
    }
}

pub struct NativeMeasurementReceiver {
    receive: mpsc::Receiver<AecPairedFrame>,
    health: Arc<Mutex<Health>>,
}

pub struct NativeMeasurement {
    samples: AecSampleEvidence,
    identity: NativeAecIdentity,
    health: Arc<Mutex<Health>>,
    measurement_adc_end: u64,
}

impl NativeMeasurementReceiver {
    pub async fn collect(mut self, deadline: Instant) -> Result<NativeMeasurement, NativeAecError> {
        let mut verifier = AecSampleVerifier::new();
        let mut measurement_adc_end = 0;
        for _ in 0..45 {
            let frame = tokio::time::timeout_at(deadline, self.receive.recv())
                .await
                .map_err(|_| NativeAecError::Deadline)?
                .ok_or(NativeAecError::Source)?;
            measurement_adc_end = frame
                .raw
                .start_sample
                .checked_add(frame.raw.samples.len() as u64)
                .ok_or(NativeAecError::Provenance)?;
            verifier
                .push(frame)
                .map_err(|_| NativeAecError::Provenance)?;
        }
        let samples = verifier.finish().map_err(|_| NativeAecError::Provenance)?;
        let health = self.health.lock().map_err(|_| NativeAecError::Source)?;
        if health.poisoned {
            return Err(NativeAecError::Source);
        }
        Ok(NativeMeasurement {
            samples,
            identity: health.identity.clone().ok_or(NativeAecError::Source)?,
            health: self.health.clone(),
            measurement_adc_end,
        })
    }
}

impl NativeMeasurement {
    pub fn finish(
        self,
        binding: AecMeasurementBinding,
        metadata: AecDeviceMetadata,
        observation: AecObservationEvidence,
    ) -> Result<AecProofReadyMeasurement, NativeAecError> {
        let health = self.health.lock().map_err(|_| NativeAecError::Source)?;
        if health.poisoned
            || health.identity.as_ref() != Some(&self.identity)
            || health
                .fresh
                .is_none_or(|t| Instant::now().saturating_duration_since(t) > FRESHNESS)
        {
            return Err(NativeAecError::Source);
        }
        let timing = observation
            .native_timing
            .as_ref()
            .ok_or(NativeAecError::Provenance)?;
        if timing.graph != self.identity.graph
            || timing.capture_buffer_frames != self.identity.capture_buffer
            || timing.maximum_read_bracket_ns > health.diagnostics.maximum_read_bracket_ns
            || timing.adc_end > health.adc_end
            || timing.adc_start < self.measurement_adc_end
        {
            return Err(NativeAecError::Provenance);
        }
        drop(health);
        if binding.graph != self.identity.graph
            || binding.source_port != self.identity.source_port
            || binding.sink_port != self.identity.sink_port
            || binding.source_channel_gains != self.identity.capture_gains
            || binding.sink_channel_gains != self.identity.playback_gains
            || binding.source_muted != self.identity.capture_muted
            || binding.sink_muted != self.identity.playback_muted
            || metadata.sink_volume_percent != self.identity.playback_volume_percent
        {
            return Err(NativeAecError::Provenance);
        }
        let acquisition = |phase: usize, powers: &[f64]| AecPowerAcquisition {
            acquisition_id: self.samples.provenance().acquisition_ids()[phase].clone(),
            samples_per_window: AEC_SAMPLES_PER_POWER_WINDOW,
            powers: powers.to_vec(),
        };
        // Only the evaluator's copied windows are fixture-relative. The sealed
        // sample evidence and runtime observation remain in the actual ADC domain.
        let fixture_origin = self
            .samples
            .fixture_windows()
            .first()
            .ok_or(NativeAecError::Provenance)?
            .start_sample;
        let windows = self
            .samples
            .fixture_windows()
            .iter()
            .map(|window| {
                let start = window
                    .start_sample
                    .checked_sub(fixture_origin)
                    .ok_or(NativeAecError::Provenance)?;
                let end = window
                    .end_sample
                    .checked_sub(fixture_origin)
                    .ok_or(NativeAecError::Provenance)?;
                Ok(AecPowerWindow {
                    sequence: window.sequence,
                    raw_start_sample: start,
                    raw_end_sample: end,
                    clean_start_sample: start,
                    clean_end_sample: end,
                    raw_power: window.raw_power,
                    clean_power: window.clean_power,
                    raw_clipped_samples: 0,
                    clean_clipped_samples: 0,
                    fixture_dbfs: AEC_FIXTURE_DBFS,
                })
            })
            .collect::<Result<Vec<_>, NativeAecError>>()?;
        let input = AecValidationInput {
            metadata,
            binding,
            fixture_acquisition_id: self.samples.provenance().acquisition_ids()[3].clone(),
            raw_baseline: acquisition(0, self.samples.raw_baseline_powers()),
            clean_baseline: acquisition(1, self.samples.clean_baseline_powers()),
            resolution: acquisition(2, self.samples.resolution_powers()),
            windows,
            observation,
        };
        if !evaluate_aec(input.clone())
            .map_err(|_| NativeAecError::Provenance)?
            .validated
        {
            return Err(NativeAecError::Provenance);
        }
        Ok(AecProofReadyMeasurement::from_native_acquisition(input))
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Packet {
    Ready { identity: NativeAecIdentity },
    Frame(WireFrame),
    Ack { request: u64, success: bool },
    Fatal { reason: String },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChannelValidity {
    clipped: u32,
    peak: u32,
    sum: i64,
    squared_sum: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFrame {
    session_id: u64,
    generation: u64,
    sequence: u64,
    adc_start: u64,
    adc_end: u64,
    capture_monotonic_ns: u64,
    bracket_ns: u64,
    written: u64,
    played: u64,
    reference_start: u64,
    reference_end: u64,
    capture_timestamp: Option<u64>,
    playback_timestamp: Option<u64>,
    phase: u8,
    raw: Vec<i16>,
    reference: Vec<i16>,
    clean: Vec<i16>,
    provider_pcm: Vec<i16>,
    origin: NativeCaptureOrigin,
    channels: Vec<ChannelValidity>,
}

/// Created only together with the fixed trusted launch; cannot be constructed
/// from caller-supplied frames, powers, or an arbitrary process command.
pub struct NativeAecSource {
    read: BufReader<ChildStdout>,
    write: ChildStdin,
    session_id: u64,
    health: Arc<Mutex<Health>>,
    commands: mpsc::Receiver<Request>,
    handle: NativeAecHandle,
    capture: mpsc::Sender<NativeCapturedFrame>,
    capture_rx: Option<NativeCaptureReceiver>,
    measurement: mpsc::Sender<AecPairedFrame>,
    measurement_rx: Option<NativeMeasurementReceiver>,
    pending: BTreeMap<u64, (u32, oneshot::Sender<Result<(), NativeAecError>>)>,
    request_id: u64,
    pending_write: Option<(Vec<u8>, usize, Instant)>,
    positive_drain_ack: Option<oneshot::Sender<Result<(), NativeAecError>>>,
    line: Vec<u8>,
    next_adc: Option<u64>,
    next_sequence: Option<u64>,
    last_host: u64,
    last_played: u64,
    last_capture_timestamp: u64,
    last_playback_timestamp: u64,
    cursor_anchor: Option<i128>,
    measurement_window: u64,
    window_start: u64,
    raw: Vec<i16>,
    clean: Vec<i16>,
    capture_first: Option<(u64, u64, NativeCaptureOrigin, u64)>,
    provider: Vec<i16>,
    provider_sequence: u64,
}

pub async fn spawn_retained_native_aec(
    authority: NativeAecLaunchAuthority,
) -> Result<(Child, NativeAecSource), NativeAecError> {
    let suffix = authority
        .scope_unit
        .strip_prefix("translator-aec-")
        .ok_or(NativeAecError::Authority)?;
    if authority.session_id == 0
        || authority.lifecycle_fd < 3
        || suffix.len() != 32
        || !suffix
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(NativeAecError::Authority);
    }
    let mut command = Command::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/translator-aec-backend-check"
    ));
    command
        .arg("--isolated")
        .arg("--physical")
        .env("TRANSLATOR_AEC_SCOPE_UNIT", &authority.scope_unit)
        .env(
            "TRANSLATOR_AEC_EXPECTED_SESSION",
            format!("{:016x}", authority.session_id),
        )
        .env(
            "TRANSLATOR_AEC_LIFECYCLE_FD",
            authority.lifecycle_fd.to_string(),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let lifecycle_fd = authority.lifecycle_fd;
    unsafe {
        command.pre_exec(move || {
            rustix::process::setsid().map_err(std::io::Error::from)?;
            fcntl_setfd(BorrowedFd::borrow_raw(lifecycle_fd), FdFlags::empty())
                .map_err(std::io::Error::from)
        });
    }
    let mut child = command.spawn().map_err(|_| NativeAecError::Source)?;
    let read = BufReader::new(child.stdout.take().ok_or(NativeAecError::Source)?);
    let write = child.stdin.take().ok_or(NativeAecError::Source)?;
    let health = Arc::new(Mutex::new(Health::default()));
    let (command_tx, commands) = mpsc::channel(20);
    let (capture, capture_rx) = mpsc::channel(20);
    let (measurement, measurement_rx) = mpsc::channel(4);
    let handle = NativeAecHandle {
        health: health.clone(),
        commands: command_tx,
    };
    Ok((
        child,
        NativeAecSource {
            read,
            write,
            session_id: authority.session_id,
            health: health.clone(),
            commands,
            handle,
            capture,
            capture_rx: Some(NativeCaptureReceiver {
                receive: capture_rx,
                health: health.clone(),
            }),
            measurement,
            measurement_rx: Some(NativeMeasurementReceiver {
                receive: measurement_rx,
                health,
            }),
            pending: BTreeMap::new(),
            request_id: 0,
            pending_write: None,
            positive_drain_ack: None,
            line: Vec::new(),
            next_adc: None,
            next_sequence: None,
            last_host: 0,
            last_played: 0,
            last_capture_timestamp: 0,
            last_playback_timestamp: 0,
            cursor_anchor: None,
            measurement_window: 0,
            window_start: 0,
            raw: Vec::new(),
            clean: Vec::new(),
            capture_first: None,
            provider: Vec::new(),
            provider_sequence: 0,
        },
    ))
}

impl NativeAecSource {
    pub fn handle(&self) -> NativeAecHandle {
        self.handle.clone()
    }
    pub fn take_capture(&mut self) -> Result<NativeCaptureReceiver, NativeAecError> {
        self.capture_rx.take().ok_or(NativeAecError::AlreadyOwned)
    }
    pub fn take_measurement(&mut self) -> Result<NativeMeasurementReceiver, NativeAecError> {
        self.measurement_rx
            .take()
            .ok_or(NativeAecError::AlreadyOwned)
    }

    pub async fn next_event(
        &mut self,
        deadline: Instant,
    ) -> Result<NativeAecEvent, NativeAecError> {
        let result = self.next_inner(deadline).await;
        if result.is_err() {
            if let Ok(mut health) = self.health.lock() {
                health.poisoned = true;
            }
            for (_, (_, ack)) in std::mem::take(&mut self.pending) {
                let _ = ack.send(Err(NativeAecError::Source));
            }
        }
        result
    }

    async fn next_inner(&mut self, deadline: Instant) -> Result<NativeAecEvent, NativeAecError> {
        loop {
            self.flush_positive_drain()?;
            if self
                .health
                .lock()
                .map_err(|_| NativeAecError::Source)?
                .poisoned
            {
                return Err(NativeAecError::Source);
            }
            if self.line.len() > MAX_PACKET {
                return Err(NativeAecError::Provenance);
            }
            let remaining = MAX_PACKET + 1 - self.line.len();
            let mut limited = (&mut self.read).take(remaining as u64);
            let write_bytes = self
                .pending_write
                .as_ref()
                .map(|(bytes, at, _)| &bytes[*at..])
                .unwrap_or(&[]);
            let io_deadline = self
                .pending_write
                .as_ref()
                .map(|(_, _, limit)| deadline.min(*limit))
                .unwrap_or(deadline);
            tokio::select! {
                _=tokio::time::sleep_until(io_deadline)=>return Err(NativeAecError::Deadline),
                request=self.commands.recv(),if self.pending_write.is_none()=>{
                    let request=request.ok_or(NativeAecError::Source)?;
                    if request.ack.is_closed() {return Err(NativeAecError::Source);}
                    if Instant::now() >= request.deadline { let _=request.ack.send(Err(NativeAecError::Deadline));return Err(NativeAecError::Deadline); }
                    self.request_id=self.request_id.checked_add(1).ok_or(NativeAecError::Provenance)?;
                    let mut payload=Vec::with_capacity(84+request.pcm.len());
                    payload.extend_from_slice(&request.command.to_le_bytes());payload.extend_from_slice(&self.request_id.to_le_bytes());payload.extend_from_slice(&(request.pcm.len() as u32).to_le_bytes());
                    let mut hash=[0u8;64]; if !request.hash.is_empty() { hash.copy_from_slice(request.hash.as_bytes()); } payload.extend_from_slice(&hash);payload.extend_from_slice(&request.pcm);
                    let mut packet=(payload.len() as u32).to_le_bytes().to_vec();packet.extend(payload);
                    self.pending_write=Some((packet,0,request.deadline));
                    self.pending.insert(self.request_id,(request.command,request.ack));
                }
                written=self.write.write(write_bytes),if self.pending_write.is_some()=>{
                    let count=written.map_err(|_|NativeAecError::Source)?;
                    if count==0 {return Err(NativeAecError::Source);}
                    let (bytes,at,_)=self.pending_write.as_mut().ok_or(NativeAecError::Source)?;
                    *at+=count;if *at==bytes.len(){self.pending_write=None;}
                }
                read=limited.read_until(b'\n',&mut self.line)=>{
                    if read.map_err(|_|NativeAecError::Source)?==0 || self.line.len()>MAX_PACKET || self.line.last()!=Some(&b'\n') { return Err(NativeAecError::Source); }
                    let packet:Packet=serde_json::from_slice(&self.line).map_err(|_|NativeAecError::Provenance)?;self.line.clear();
                    match packet {
                        Packet::Ready { identity }=>{
                            if self.handle.identity().is_some() || !identity.graph.is_valid() || !matches!(identity.graph,AecGraphIdentity::Native {session_id,..} if session_id==self.session_id) || identity.card!=0 || identity.card_id!="PCH" || identity.pcm_name!="ALC287 Analog" || !(1..=2).contains(&identity.capture_channels) || !(1..=2).contains(&identity.playback_channels) || identity.capture_buffer<1920 || identity.playback_buffer<1920 || identity.capture_buffer>48000 || identity.playback_buffer>48000 || identity.capture_origin_monotonic_ns==0 || identity.control_fingerprint.is_empty() || identity.capture_gains.is_empty() || identity.playback_gains.is_empty() { return Err(NativeAecError::Provenance); }
                            let mut health=self.health.lock().map_err(|_|NativeAecError::Source)?;health.identity=Some(identity.clone());health.capture_paused=true;return Ok(NativeAecEvent::Ready(Box::new(identity)));
                        }
                        Packet::Frame(frame)=>{ let adc=frame.adc_end;self.accept_frame(frame)?;return Ok(NativeAecEvent::Progress {adc_frames:adc}); }
                        Packet::Ack {request,success}=>{
                            let (command,ack)=self.pending.remove(&request).ok_or(NativeAecError::Provenance)?;
                            if !success { let _=ack.send(Err(NativeAecError::Source));return Err(NativeAecError::Source); }
                            if matches!(command,5|7) && !self.provider.is_empty() {let _=ack.send(Err(NativeAecError::Provenance));return Err(NativeAecError::Provenance);}
                            if command==7 {
                                if self.positive_drain_ack.is_some(){return Err(NativeAecError::Provenance);}
                                self.positive_drain_ack=Some(ack);continue;
                            }
                            if matches!(command,2|5|6) {self.health.lock().map_err(|_|NativeAecError::Source)?.capture_paused=command==5;}
                            ack.send(Ok(())).map_err(|_|NativeAecError::Source)?;
                        }
                        Packet::Fatal {reason}=>{ let _=reason;return Err(NativeAecError::Source); }
                    }
                }
            }
        }
    }

    fn flush_positive_drain(&mut self) -> Result<(), NativeAecError> {
        if self.positive_drain_ack.is_none() {
            return Ok(());
        }
        let health = self.health.lock().map_err(|_| NativeAecError::Source)?;
        if health.poisoned {
            return Err(NativeAecError::Source);
        }
        if health.positive_enqueued_end != 0
            && health.positive_consumed_end >= health.positive_enqueued_end
        {
            drop(health);
            self.positive_drain_ack
                .take()
                .ok_or(NativeAecError::Source)?
                .send(Ok(()))
                .map_err(|_| NativeAecError::Source)?;
        }
        Ok(())
    }

    fn accept_frame(&mut self, f: WireFrame) -> Result<(), NativeAecError> {
        let identity = self.handle.identity().ok_or(NativeAecError::Provenance)?;
        let stamp = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
        let now_ns = u64::try_from(stamp.tv_sec)
            .ok()
            .and_then(|s| s.checked_mul(1_000_000_000))
            .and_then(|s| s.checked_add(stamp.tv_nsec as u64))
            .ok_or(NativeAecError::Provenance)?;
        if !matches!(identity.graph,AecGraphIdentity::Native {session_id,generation,..} if session_id==f.session_id && generation==f.generation)
            || f.adc_end.checked_sub(f.adc_start) != Some(BLOCK as u64)
            || self.next_adc.is_some_and(|n| n != f.adc_start)
            || self.next_sequence.is_some_and(|n| n != f.sequence)
            || f.capture_monotonic_ns == 0
            || f.capture_monotonic_ns < self.last_host
            || f.capture_monotonic_ns > now_ns
            || now_ns - f.capture_monotonic_ns > 200_000_000
            || f.raw.len() != BLOCK
            || f.reference.len() != BLOCK
            || f.clean.len() != BLOCK
            || f.phase > 5
            || f.bracket_ns == 0
            || f.bracket_ns > (identity.capture_buffer as u64 * 1_000_000_000 / 48000)
            || f.channels.len() != identity.capture_channels as usize
            || f.played < self.last_played
            || f.played > f.written
            || f.written - f.played > identity.playback_buffer as u64
            || f.reference_end.checked_sub(f.reference_start) != Some(BLOCK as u64)
            || f.reference_end > f.played
        {
            return Err(NativeAecError::Provenance);
        }
        for (current, last) in [
            (f.capture_timestamp, &mut self.last_capture_timestamp),
            (f.playback_timestamp, &mut self.last_playback_timestamp),
        ] {
            if let Some(stamp) = current {
                if stamp == 0 || stamp < *last {
                    return Err(NativeAecError::Provenance);
                }
                *last = stamp;
            }
        }
        let offset = f.played as i128 - f.adc_end as i128;
        let anchor = *self.cursor_anchor.get_or_insert(offset);
        let bracket_frames = f
            .bracket_ns
            .checked_mul(48000)
            .and_then(|n| n.checked_add(999999999))
            .ok_or(NativeAecError::Provenance)?
            / 1000000000;
        let uncertainty =
            identity.capture_buffer as u64 + identity.playback_buffer as u64 + bracket_frames;
        if (offset - anchor).unsigned_abs() > uncertainty as u128 {
            return Err(NativeAecError::Provenance);
        }
        if f.phase != 0
            && (f.channels.iter().any(|c| {
                c.clipped != 0
                    || c.peak >= 32767
                    || (c.squared_sum as i128) * BLOCK as i128 <= (c.sum as i128) * (c.sum as i128)
            }) || f
                .raw
                .iter()
                .chain(&f.clean)
                .any(|s| *s == i16::MIN || *s == i16::MAX))
        {
            return Err(NativeAecError::Pcm);
        }
        self.next_adc = Some(f.adc_end);
        self.next_sequence = f.sequence.checked_add(1);
        self.last_host = f.capture_monotonic_ns;
        self.last_played = f.played;
        {
            let mut health = self.health.lock().map_err(|_| NativeAecError::Source)?;
            health.fresh = Some(Instant::now());
            health.adc_end = f.adc_end;
            health.diagnostics.reference_uncertainty_frames = health
                .diagnostics
                .reference_uncertainty_frames
                .max(uncertainty);
            health.diagnostics.maximum_read_bracket_ns =
                health.diagnostics.maximum_read_bracket_ns.max(f.bracket_ns);
            health.diagnostics.capture_unknown_timestamps +=
                u64::from(f.capture_timestamp.is_none());
            health.diagnostics.playback_unknown_timestamps +=
                u64::from(f.playback_timestamp.is_none());
            if f.phase == 0 {
                health.diagnostics.settling_frames += BLOCK as u64;
                health
                    .diagnostics
                    .settling_clipped_samples
                    .resize(f.channels.len(), 0);
                health
                    .diagnostics
                    .settling_peak_s16
                    .resize(f.channels.len(), 0);
                for (i, c) in f.channels.iter().enumerate() {
                    health.diagnostics.settling_clipped_samples[i] += c.clipped as u64;
                    health.diagnostics.settling_peak_s16[i] =
                        health.diagnostics.settling_peak_s16[i].max(c.peak);
                }
            }
        }
        if (1..=4).contains(&f.phase)
            && self.measurement_window < 45
            && !self.measurement.is_closed()
        {
            let expected = if self.measurement_window < 15 {
                self.measurement_window / 5 + 1
            } else {
                4
            };
            if f.phase as u64 != expected {
                return Err(NativeAecError::Provenance);
            }
            if self.raw.is_empty() {
                self.window_start = f.adc_start;
            }
            self.raw.extend(f.raw);
            self.clean.extend(f.clean);
            if self.raw.len() == 48000 {
                let make = |samples, stream| AecChannelFrame {
                    clock_id: format!("native-adc:{}:{}", f.session_id, f.generation),
                    generation: format!("native:{}:{}", f.session_id, f.generation),
                    stream_id: stream,
                    acquisition_id: format!("native:{}:phase{}", f.session_id, f.phase),
                    frame_id: self.measurement_window,
                    start_sample: self.window_start,
                    sample_rate_hz: 48000,
                    lost_frames: 0,
                    samples,
                };
                let pair = AecPairedFrame {
                    raw: make(std::mem::take(&mut self.raw), "raw".into()),
                    clean: make(std::mem::take(&mut self.clean), "clean".into()),
                };
                self.measurement
                    .try_send(pair)
                    .map_err(|_| NativeAecError::Overflow)?;
                self.measurement_window += 1;
            }
        }
        if !f.provider_pcm.is_empty() && !self.capture.is_closed() {
            if f.provider_pcm.len() != 160 {
                return Err(NativeAecError::Pcm);
            }
            if let Some((_, _, origin, bracket)) = &mut self.capture_first {
                if origin != &f.origin {
                    return Err(NativeAecError::Provenance);
                }
                *bracket = (*bracket).max(f.bracket_ns);
            }
            self.capture_first.get_or_insert((
                f.adc_start,
                f.capture_monotonic_ns,
                f.origin.clone(),
                f.bracket_ns,
            ));
            self.provider.extend(f.provider_pcm);
            if self.provider.len() == 320 {
                let (adc_start, capture_ns, origin, capture_read_bracket_ns) = self
                    .capture_first
                    .take()
                    .ok_or(NativeAecError::Provenance)?;
                if f.adc_end.checked_sub(adc_start) != Some(960) {
                    return Err(NativeAecError::Provenance);
                }
                let pcm = std::mem::take(&mut self.provider)
                    .into_iter()
                    .flat_map(i16::to_le_bytes)
                    .collect();
                let frame = PcmFrame::try_new(
                    self.provider_sequence,
                    capture_ns,
                    StreamPcmFormat::provider_default(),
                    pcm,
                )
                .map_err(|_| NativeAecError::Pcm)?;
                self.provider_sequence = self
                    .provider_sequence
                    .checked_add(1)
                    .ok_or(NativeAecError::Provenance)?;
                self.capture
                    .try_send(NativeCapturedFrame {
                        frame,
                        graph: identity.graph,
                        adc_start,
                        adc_end: f.adc_end,
                        origin,
                        capture_read_bracket_ns,
                    })
                    .map_err(|_| NativeAecError::Overflow)?;
                if matches!(f.origin, NativeCaptureOrigin::InjectedPositive { .. }) {
                    self.health
                        .lock()
                        .map_err(|_| NativeAecError::Source)?
                        .positive_enqueued_end = f.adc_end;
                }
            }
        }
        Ok(())
    }
}

impl Drop for NativeAecSource {
    fn drop(&mut self) {
        if let Ok(mut health) = self.health.lock() {
            health.poisoned = true;
        }
        for (_, (_, ack)) in std::mem::take(&mut self.pending) {
            let _ = ack.send(Err(NativeAecError::Source));
        }
        // ChildStdin closes automatically. Only the guardian terminates/reaps.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AecNativeObservationTiming, AecPositiveControl};

    async fn collect_test_measurement(
        source: &NativeAecSource,
        origin: u64,
        fault: Option<u8>,
    ) -> Result<NativeMeasurement, NativeAecError> {
        let (send, receive) = mpsc::channel(45);
        for index in 0..45 {
            let acquisition = match index {
                0..5 => "raw-baseline",
                5..10 => "clean-baseline",
                10..15 => "resolution",
                _ => "fixture",
            };
            let channel = |stream: &str, level| AecChannelFrame {
                clock_id: "native-adc-42".into(),
                generation: "71".into(),
                stream_id: stream.into(),
                acquisition_id: acquisition.into(),
                frame_id: index,
                start_sample: origin + index * AEC_SAMPLES_PER_POWER_WINDOW,
                sample_rate_hz: 48_000,
                lost_frames: 0,
                samples: vec![level; AEC_SAMPLES_PER_POWER_WINDOW as usize],
            };
            let mut pair = AecPairedFrame {
                raw: channel("raw", if index >= 15 { 1000 } else { 1 }),
                clean: channel("clean", if index >= 15 { 100 } else { 1 }),
            };
            if index == 16 {
                match fault {
                    Some(0) => pair.clean.start_sample += 1,
                    Some(1) => {
                        pair.raw.start_sample += 1;
                        pair.clean.start_sample += 1;
                    }
                    Some(2) => pair.clean.clock_id = "other-adc".into(),
                    Some(3) => {
                        pair.raw.generation = "72".into();
                        pair.clean.generation = "72".into();
                    }
                    Some(4) => {
                        pair.raw.samples.pop();
                    }
                    Some(5) => {
                        pair.raw.start_sample = u64::MAX;
                        pair.clean.start_sample = u64::MAX;
                    }
                    _ => {}
                }
            }
            send.send(pair).await.unwrap();
        }
        NativeMeasurementReceiver {
            receive,
            health: source.health.clone(),
        }
        .collect(Instant::now() + std::time::Duration::from_secs(2))
        .await
    }

    fn finish_test_arguments(
        measurement: &NativeMeasurement,
    ) -> (
        AecMeasurementBinding,
        AecDeviceMetadata,
        AecObservationEvidence,
    ) {
        let identity = &measurement.identity;
        let binding = AecMeasurementBinding {
            audio_server_id: "retained-native-42".into(),
            source_hardware_id: "alsa-hw:0,0:capture".into(),
            sink_hardware_id: "alsa-hw:0,0:playback".into(),
            source_name: "native-clean".into(),
            sink_name: "native-playback".into(),
            source_port: identity.source_port.clone(),
            sink_port: identity.sink_port.clone(),
            source_channel_gains: identity.capture_gains.clone(),
            sink_channel_gains: identity.playback_gains.clone(),
            source_muted: identity.capture_muted,
            sink_muted: identity.playback_muted,
            source_geometry: "bound-internal-mic".into(),
            sink_geometry: "bound-speaker".into(),
            graph: identity.graph.clone(),
            aec_generation: "71".into(),
            aec_config_id: "spa-webrtc48k".into(),
            vad_config_id: "bound-vad".into(),
            provider_config_id: "bound-provider".into(),
        };
        let metadata = AecDeviceMetadata {
            source_name: binding.source_name.clone(),
            sink_name: binding.sink_name.clone(),
            source_geometry: binding.source_geometry.clone(),
            sink_geometry: binding.sink_geometry.clone(),
            sink_port: binding.sink_port.clone(),
            sink_volume_percent: identity.playback_volume_percent,
        };
        let timing = AecNativeObservationTiming {
            graph: identity.graph.clone(),
            adc_start: measurement.measurement_adc_end + 48_000,
            adc_end: measurement.measurement_adc_end + 61 * 48_000,
            capture_buffer_frames: identity.capture_buffer,
            maximum_read_bracket_ns: 1_000_000,
            source_gaps: 0,
            source_duplicates: 0,
            source_reordered: 0,
        };
        {
            let mut health = measurement.health.lock().unwrap();
            health.adc_end = timing.adc_end;
            health.diagnostics.maximum_read_bracket_ns = timing.maximum_read_bracket_ns;
            health.fresh = Some(Instant::now());
        }
        let observation = AecObservationEvidence {
            native_timing: Some(timing),
            observer_generation: "observer-71".into(),
            calibration_attempt_id: "attempt-42".into(),
            challenge_id: "challenge-42".into(),
            interval_id: "far-only-42".into(),
            started_monotonic_ns: 80_000_000_000,
            ended_monotonic_ns: 140_000_000_000,
            expected_frames: 3000,
            processed_frames: 3000,
            stream_generation: "71".into(),
            sample_rate_hz: 16_000,
            channels: 1,
            frame_duration_ms: 20,
            samples_per_frame: 320,
            first_frame_sequence: 4000,
            last_frame_sequence: 6999,
            first_capture_monotonic_ns: 80_000_000_000,
            last_capture_monotonic_ns: 139_980_000_000,
            maximum_frame_gap_ns: 20_000_000,
            frame_gaps: 0,
            duplicate_frames: 0,
            out_of_order_frames: 0,
            vad_events_before: 1,
            vad_events_after: 1,
            provider_attempts_before: 1,
            provider_attempts_after: 1,
            provider_accepted_before: 1,
            provider_accepted_after: 1,
            resets: 0,
            dropped_frames: 0,
            observer_errors: 0,
            terminated_early: false,
            positive_control: AecPositiveControl {
                observer_generation: "observer-71".into(),
                calibration_attempt_id: "attempt-42".into(),
                challenge_id: "challenge-42".into(),
                completed_monotonic_ns: 79_000_000_000,
                speech_started_events: 1,
                provider_submission_attempts: 1,
                provider_submissions_accepted: 1,
                resets: 0,
                observer_errors: 0,
            },
        };
        (binding, metadata, observation)
    }

    #[tokio::test]
    async fn absolute_adc_measurement_finishes_with_only_fixture_relative_evaluation_windows() {
        for origin in [96_000, u64::MAX - 106 * 48_000] {
            let (_child, source) = synthetic_source().await;
            let measurement = collect_test_measurement(&source, origin, None)
                .await
                .unwrap();
            assert_eq!(
                measurement.samples.provenance().sample_range(),
                (origin, origin + 45 * 48_000)
            );
            assert_eq!(
                measurement.samples.fixture_windows()[0].start_sample,
                origin + 15 * 48_000
            );
            assert_eq!(
                measurement.samples.fixture_windows()[29].end_sample,
                origin + 45 * 48_000
            );
            let (binding, metadata, observation) = finish_test_arguments(&measurement);
            let expected_observation = observation.clone();
            let input = measurement
                .finish(binding, metadata, observation)
                .unwrap()
                .into_validation_input();
            assert_eq!(input.observation, expected_observation);
            for (index, window) in input.windows.iter().enumerate() {
                assert_eq!(window.sequence, index as u64);
                assert_eq!(window.raw_start_sample, index as u64 * 48_000);
                assert_eq!(window.clean_start_sample, window.raw_start_sample);
                assert_eq!(window.raw_end_sample, (index as u64 + 1) * 48_000);
                assert_eq!(window.clean_end_sample, window.raw_end_sample);
                assert_eq!(
                    (window.raw_power, window.clean_power),
                    (1_000_000.0, 10_000.0)
                );
            }
            assert_eq!(input.raw_baseline.powers, vec![1.0; 5]);
            assert!(evaluate_aec(input).unwrap().validated);
        }
    }

    #[tokio::test]
    async fn absolute_adc_faults_never_reach_evaluation_projection() {
        for fault in 0..6 {
            let (_child, source) = synthetic_source().await;
            assert!(
                matches!(
                    collect_test_measurement(&source, 96_000, Some(fault)).await,
                    Err(NativeAecError::Provenance)
                ),
                "fault {fault}"
            );
        }
    }

    #[tokio::test]
    async fn fixture_projection_does_not_bypass_observation_or_source_custody() {
        for fault in 0..6 {
            let (_child, source) = synthetic_source().await;
            let measurement = collect_test_measurement(&source, 96_000, None)
                .await
                .unwrap();
            let (binding, metadata, mut observation) = finish_test_arguments(&measurement);
            let timing = observation.native_timing.as_mut().unwrap();
            match fault {
                0 => match &mut timing.graph {
                    AecGraphIdentity::Native { session_id, .. } => *session_id += 1,
                    _ => unreachable!(),
                },
                1 => timing.adc_start = measurement.measurement_adc_end - 1,
                2 => timing.adc_end += 1,
                3 => timing.source_gaps = 1,
                4 => measurement.health.lock().unwrap().poisoned = true,
                5 => {
                    measurement.health.lock().unwrap().fresh =
                        Some(Instant::now() - std::time::Duration::from_secs(1))
                }
                _ => unreachable!(),
            }
            let expected = if fault >= 4 {
                NativeAecError::Source
            } else {
                NativeAecError::Provenance
            };
            assert!(
                matches!(measurement.finish(binding, metadata, observation), Err(error) if error == expected),
                "fault {fault}"
            );
        }
    }

    #[test]
    fn runtime_gain_preserves_endpoints_without_touching_evidence() {
        let pcm: Vec<u8> = [i16::MIN, -1001, -1, 0, 1, 1001, i16::MAX]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        assert_eq!(runtime_output_gain(&pcm, 100).unwrap(), pcm);
        assert_eq!(runtime_output_gain(&pcm, 0).unwrap(), vec![0; pcm.len()]);
        let expected: Vec<u8> = [-16384i16, -500, 0, 0, 0, 500, 16383]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        assert_eq!(runtime_output_gain(&pcm, 50).unwrap(), expected);
        assert_eq!(runtime_output_gain(&pcm, 101), Err(NativeAecError::Pcm));
        assert_eq!(runtime_output_gain(&[1], 50), Err(NativeAecError::Pcm));
    }

    // Synthetic fault injection only; this private constructor is not compiled
    // into the production source factory and cannot admit an acoustic proof.
    async fn synthetic_source() -> (Child, NativeAecSource) {
        let mut child = Command::new("/bin/cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let health = Arc::new(Mutex::new(Health::default()));
        let identity = NativeAecIdentity {
            graph: AecGraphIdentity::Native {
                session_id: 42,
                generation: 71,
                physical_device_id: "alsa-hw:0,0:PCH:ALC287 Analog".into(),
                dsp_config_id: "spa-webrtc48k".into(),
            },
            card: 0,
            card_id: "PCH".into(),
            pcm_name: "ALC287 Analog".into(),
            capture_channels: 2,
            playback_channels: 2,
            capture_buffer: 3840,
            playback_buffer: 3840,
            source_port: "analog-input-internal-mic".into(),
            sink_port: "analog-output-speaker".into(),
            control_fingerprint: "configuration-sha".into(),
            capture_gains: vec![58, 58],
            playback_gains: vec![40, 40],
            capture_muted: false,
            playback_muted: false,
            playback_volume_percent: 70,
            capture_origin_monotonic_ns: 1,
        };
        {
            let mut h = health.lock().unwrap();
            h.identity = Some(identity);
            h.fresh = Some(Instant::now());
            h.capture_paused = true;
        }
        let (command_tx, commands) = mpsc::channel(20);
        let (capture, capture_rx) = mpsc::channel(20);
        let (measurement, measurement_rx) = mpsc::channel(4);
        let handle = NativeAecHandle {
            health: health.clone(),
            commands: command_tx,
        };
        let source = NativeAecSource {
            read: BufReader::new(child.stdout.take().unwrap()),
            write: child.stdin.take().unwrap(),
            session_id: 42,
            health: health.clone(),
            commands,
            handle,
            capture,
            capture_rx: Some(NativeCaptureReceiver {
                receive: capture_rx,
                health: health.clone(),
            }),
            measurement,
            measurement_rx: Some(NativeMeasurementReceiver {
                receive: measurement_rx,
                health,
            }),
            pending: BTreeMap::new(),
            request_id: 0,
            pending_write: None,
            positive_drain_ack: None,
            line: vec![],
            next_adc: None,
            next_sequence: None,
            last_host: 0,
            last_played: 0,
            last_capture_timestamp: 0,
            last_playback_timestamp: 0,
            cursor_anchor: None,
            measurement_window: 0,
            window_start: 0,
            raw: vec![],
            clean: vec![],
            capture_first: None,
            provider: vec![],
            provider_sequence: 0,
        };
        (child, source)
    }
    fn frame(sequence: u64) -> WireFrame {
        let stamp = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
        let host = stamp.tv_sec as u64 * 1_000_000_000 + stamp.tv_nsec as u64;
        WireFrame {
            session_id: 42,
            generation: 71,
            sequence,
            adc_start: sequence * 480,
            adc_end: (sequence + 1) * 480,
            capture_monotonic_ns: host,
            bracket_ns: 100000,
            written: 3840 + sequence * 480,
            played: 1920 + sequence * 480,
            reference_start: 1440 + sequence * 480,
            reference_end: 1920 + sequence * 480,
            capture_timestamp: None,
            playback_timestamp: None,
            phase: 0,
            raw: (0..480)
                .map(|n| if n % 2 == 0 { 100 } else { -100 })
                .collect(),
            reference: vec![0; 480],
            clean: vec![1; 480],
            provider_pcm: vec![],
            origin: NativeCaptureOrigin::Physical,
            channels: (0..2)
                .map(|_| ChannelValidity {
                    clipped: 0,
                    peak: 100,
                    sum: 0,
                    squared_sum: 4800000,
                })
                .collect(),
        }
    }
    #[tokio::test]
    async fn same_adc_blocks_bind_provider_frames_and_actual_brackets() {
        let (_child, mut source) = synthetic_source().await;
        let mut receiver = source.take_capture().unwrap();
        assert!(source.take_capture().is_err());
        let mut first = frame(0);
        first.provider_pcm = vec![11; 160];
        first.bracket_ns = 100000;
        source.accept_frame(first).unwrap();
        let mut second = frame(1);
        second.provider_pcm = vec![22; 160];
        second.bracket_ns = 200000;
        source.accept_frame(second).unwrap();
        let result = receiver
            .next_frame(Instant::now() + std::time::Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!((result.adc_start, result.adc_end), (0, 960));
        assert_eq!(result.capture_read_bracket_ns, 200000);
        assert_eq!(result.frame.pcm().len(), 640);
        assert_eq!(source.handle.diagnostics().maximum_read_bracket_ns, 200000);
    }
    #[tokio::test]
    async fn faults_fail_at_the_live_packet_boundary() {
        for fault in 0..15 {
            let (_child, mut source) = synthetic_source().await;
            source.accept_frame(frame(0)).unwrap();
            let mut next = frame(1);
            match fault {
                0 => next.adc_start += 480,
                1 => next.sequence = 0,
                2 => next.generation += 1,
                3 => next.session_id += 1,
                4 => next.played = 1919,
                5 => next.played = next.written + 1,
                6 => next.reference_end = next.played + 1,
                7 => next.raw.pop().map(|_| ()).unwrap(),
                8 => next.capture_monotonic_ns = 0,
                9 => next.capture_monotonic_ns += 1_000_000_000,
                10 => next.bracket_ns = u64::MAX,
                11 => {
                    next.phase = 1;
                    next.channels[0].clipped = 1;
                }
                12 => {
                    next.phase = 1;
                    next.channels[1].sum = 480;
                    next.channels[1].squared_sum = 480;
                }
                13 => next.phase = 4,
                14 => {
                    next.phase = 1;
                    next.clean[0] = i16::MIN;
                }
                _ => unreachable!(),
            }
            assert!(
                source.accept_frame(next).is_err(),
                "fault {fault} unexpectedly accepted"
            );
        }
    }
    #[tokio::test]
    async fn settling_clip_is_reported_and_scored_clip_cannot_be_averaged_away() {
        let (_child, mut source) = synthetic_source().await;
        let mut settling = frame(0);
        settling.channels[0].clipped = 1;
        settling.channels[0].peak = 32768;
        source.accept_frame(settling).unwrap();
        assert_eq!(
            source.handle.diagnostics().settling_clipped_samples,
            vec![1, 0]
        );
        let mut scored = frame(1);
        scored.phase = 1;
        scored.channels[0].clipped = 1;
        assert_eq!(source.accept_frame(scored), Err(NativeAecError::Pcm));
    }
    #[tokio::test]
    async fn provider_halves_cannot_bridge_a_pause_or_generation_gap() {
        let (_child, mut source) = synthetic_source().await;
        let mut first = frame(0);
        first.provider_pcm = vec![1; 160];
        source.accept_frame(first).unwrap();
        source.accept_frame(frame(1)).unwrap();
        let mut third = frame(2);
        third.provider_pcm = vec![1; 160];
        assert_eq!(source.accept_frame(third), Err(NativeAecError::Provenance));
    }
    #[tokio::test]
    async fn custody_drop_immediately_revokes_clones_and_single_receivers() {
        let (_child, mut source) = synthetic_source().await;
        let handle = source.handle();
        let _receiver = source.take_measurement().unwrap();
        assert!(source.take_measurement().is_err());
        assert!(
            handle
                .check_fresh(Instant::now() + std::time::Duration::from_secs(1))
                .is_ok()
        );
        drop(source);
        assert!(handle.identity().is_none());
        assert!(
            handle
                .check_fresh(Instant::now() + std::time::Duration::from_secs(1))
                .is_err()
        );
    }
    #[tokio::test]
    async fn lifecycle_discard_requires_actual_paused_custody() {
        let (_child, mut source) = synthetic_source().await;
        let mut receiver = source.take_capture().unwrap();
        receiver.discard_paused().unwrap();
        source.health.lock().unwrap().capture_paused = false;
        assert!(receiver.discard_paused().is_err());
        drop(source);
        assert!(receiver.discard_paused().is_err());
    }

    #[tokio::test]
    async fn full_positive_command_keeps_reading_actual_pipe_progress() {
        let (_cat, mut source) = synthetic_source().await;
        let script = r#"
import hashlib,json,os,select,struct,time
pending=bytearray(); sequence=0; acknowledged=False
while True:
 if select.select([0],[],[],0)[0]:
  data=os.read(0,8192)
  if not data: break
  pending.extend(data)
 if len(pending)>=4:
  size=struct.unpack_from('<I',pending)[0]
  if len(pending)>=size+4 and not acknowledged:
   body=bytes(pending[4:size+4]); command,request,count=struct.unpack_from('<IQI',body)
   good=command==2 and count==437760 and hashlib.sha256(body[80:]).hexdigest().encode()==body[16:80]
   print(json.dumps({'kind':'ack','request':request,'success':good}),flush=True)
   acknowledged=True
 n=sequence; played=1920+n*480
 row={'kind':'frame','session_id':42,'generation':71,'sequence':n,'adc_start':n*480,'adc_end':(n+1)*480,
 'capture_monotonic_ns':time.monotonic_ns(),'bracket_ns':100000,'written':3840+n*480,'played':played,
 'reference_start':played-480,'reference_end':played,'capture_timestamp':None,'playback_timestamp':None,
 'phase':0,'raw':[100,-100]*240,'reference':[0]*480,'clean':[1]*480,'provider_pcm':[],
 'origin':{'kind':'physical'},'channels':[{'clipped':0,'peak':100,'sum':0,'squared_sum':4800000}]*2}
 print(json.dumps(row),flush=True);sequence+=1;time.sleep(.005)
"#;
        let mut child = Command::new("/usr/bin/python3")
            .args(["-I", "-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        source.read = BufReader::new(child.stdout.take().unwrap());
        source.write = child.stdin.take().unwrap();
        let handle = source.handle();
        let deadline = Instant::now() + std::time::Duration::from_secs(3);
        let command = tokio::spawn(async move {
            let pcm = vec![1u8; 437760];
            let hash: String = Sha256::digest(&pcm)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            handle.inject_positive(&pcm, &hash, deadline).await
        });
        while !command.is_finished() {
            source.next_event(deadline).await.unwrap();
        }
        command.await.unwrap().unwrap();
        assert!(
            source.next_adc.unwrap() > 20 * 480,
            "transport never exercised two bounded pipes"
        );
        assert!(source.pending_write.is_none());
    }

    #[tokio::test]
    async fn native_positive_fifo_barrier_waits_for_receiver_not_provider_claims() {
        let (_child, mut source) = synthetic_source().await;
        let mut receiver = source.take_capture().unwrap();
        for n in 0..2 {
            let mut row = frame(n);
            row.provider_pcm = vec![11; 160];
            row.origin = NativeCaptureOrigin::InjectedPositive {
                fixture_sha256: "frozen-public-input".into(),
            };
            source.accept_frame(row).unwrap();
        }
        let (send, mut ack) = oneshot::channel();
        source.positive_drain_ack = Some(send);
        source.flush_positive_drain().unwrap();
        assert_eq!(ack.try_recv(), Err(oneshot::error::TryRecvError::Empty));
        receiver
            .next_frame(Instant::now() + std::time::Duration::from_secs(1))
            .await
            .unwrap();
        source.flush_positive_drain().unwrap();
        ack.await.unwrap().unwrap();
        assert_eq!(source.health.lock().unwrap().positive_consumed_end, 960);
        // No VAD/provider acceptance counter is created by this FIFO owner.
    }
}
