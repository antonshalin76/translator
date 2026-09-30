use std::{
    collections::{HashMap, HashSet, VecDeque},
    process::Stdio,
    time::{Duration, Instant},
};

use serde::Deserialize;
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStdin, ChildStdout, Command},
};
use uuid::Uuid;
use webrtc_vad::{SampleRate, Vad, VadMode};

const MAX_BUFFERED_MS: u32 = 400;
const DEFAULT_SPEECH_CONFIRMATION_FRAMES: usize = 10;
const MIN_SPEECH_CONFIRMATION_FRAMES: usize = 3;
const MAX_SPEECH_CONFIRMATION_FRAMES: usize = 25;
const END_OF_UTTERANCE_SILENCE_FRAMES: usize = 15;
const ADAPTIVE_END_OF_UTTERANCE_SILENCE_FRAMES: usize = 6;
const DEFAULT_MIN_UTTERANCE_FRAMES: usize = 125;
const MIN_MIN_UTTERANCE_FRAMES: usize = 50;
const DEFAULT_MAX_UTTERANCE_FRAMES: usize = 300;
const MIN_MAX_UTTERANCE_FRAMES: usize = 200;
const MAX_MAX_UTTERANCE_FRAMES: usize = 1_500;
const DEFAULT_MIN_VOICE_RMS: f64 = 300.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamPcmFormat {
    sample_rate_hz: u32,
    channels: u8,
    frame_duration_ms: u16,
}

impl StreamPcmFormat {
    pub const fn provider_default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            channels: 1,
            frame_duration_ms: 20,
        }
    }

    pub const fn sample_rate_hz(self) -> u32 {
        self.sample_rate_hz
    }

    pub const fn channels(self) -> u8 {
        self.channels
    }

    pub const fn frame_duration_ms(self) -> u16 {
        self.frame_duration_ms
    }

    pub const fn frame_bytes(self) -> usize {
        (self.sample_rate_hz as usize * self.channels as usize * 2)
            * self.frame_duration_ms as usize
            / 1_000
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmFrame {
    sequence: u64,
    capture_monotonic_ns: u64,
    format: StreamPcmFormat,
    pcm: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PcmFrameError {
    #[error("PCM frame byte length does not match its format")]
    InvalidByteLength,
}

impl PcmFrame {
    pub fn try_new(
        sequence: u64,
        capture_monotonic_ns: u64,
        format: StreamPcmFormat,
        pcm: Vec<u8>,
    ) -> Result<Self, PcmFrameError> {
        if pcm.len() != format.frame_bytes() {
            return Err(PcmFrameError::InvalidByteLength);
        }
        Ok(Self {
            sequence,
            capture_monotonic_ns,
            format,
            pcm,
        })
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn capture_monotonic_ns(&self) -> u64 {
        self.capture_monotonic_ns
    }

    pub const fn format(&self) -> StreamPcmFormat {
        self.format
    }

    pub fn pcm(&self) -> &[u8] {
        &self.pcm
    }

    pub fn into_pcm(self) -> Vec<u8> {
        self.pcm
    }
}

#[derive(Debug, Default)]
pub struct BoundedPcmQueue {
    frames: VecDeque<PcmFrame>,
    buffered_ms: u32,
    dropped_frames: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("PCM queue exceeds 400 ms")]
pub struct PcmQueueOverflow(pub PcmFrame);

impl BoundedPcmQueue {
    pub fn push(&mut self, frame: PcmFrame) -> Result<(), PcmQueueOverflow> {
        let duration_ms = u32::from(frame.format.frame_duration_ms);
        if self.buffered_ms.saturating_add(duration_ms) > MAX_BUFFERED_MS {
            self.dropped_frames = self.dropped_frames.saturating_add(1);
            return Err(PcmQueueOverflow(frame));
        }
        self.buffered_ms += duration_ms;
        self.frames.push_back(frame);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<PcmFrame> {
        let frame = self.frames.pop_front()?;
        self.buffered_ms = self
            .buffered_ms
            .saturating_sub(u32::from(frame.format.frame_duration_ms));
        Some(frame)
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.buffered_ms = 0;
    }

    pub const fn buffered_ms(&self) -> u32 {
        self.buffered_ms
    }

    pub const fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureEvent {
    SpeechStarted {
        stream_id: Uuid,
        utterance_id: Uuid,
        capture_monotonic_ns: u64,
    },
    Frame {
        stream_id: Uuid,
        utterance_id: Uuid,
        frame: PcmFrame,
        end_of_utterance: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum VadError {
    #[error("VAD requires 16 kHz mono 20 ms S16LE PCM")]
    UnsupportedFormat,
    #[error("VAD rejected the PCM frame")]
    InvalidFrame,
}

pub trait VoiceDetector {
    fn is_voice(&mut self, samples: &[i16]) -> Result<bool, VadError>;
}

pub struct WebRtcVoiceDetector {
    vad: Vad,
    min_voice_rms: f64,
}

impl WebRtcVoiceDetector {
    pub fn aggressive() -> Self {
        Self {
            vad: Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, VadMode::VeryAggressive),
            min_voice_rms: configured_min_voice_rms(),
        }
    }
}

impl Default for WebRtcVoiceDetector {
    fn default() -> Self {
        Self::aggressive()
    }
}

impl VoiceDetector for WebRtcVoiceDetector {
    fn is_voice(&mut self, samples: &[i16]) -> Result<bool, VadError> {
        let vad_voice = self
            .vad
            .is_voice_segment(samples)
            .map_err(|()| VadError::InvalidFrame)?;
        Ok(vad_voice && rms(samples) >= self.min_voice_rms)
    }
}

fn configured_min_voice_rms() -> f64 {
    std::env::var("TRANSLATOR_VAD_MIN_RMS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(DEFAULT_MIN_VOICE_RMS)
}

fn configured_speech_confirmation_frames() -> usize {
    std::env::var("TRANSLATOR_VAD_CONFIRMATION_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| {
            (MIN_SPEECH_CONFIRMATION_FRAMES..=MAX_SPEECH_CONFIRMATION_FRAMES).contains(value)
        })
        .unwrap_or(DEFAULT_SPEECH_CONFIRMATION_FRAMES)
}

fn configured_max_utterance_frames() -> usize {
    std::env::var("TRANSLATOR_VAD_MAX_UTTERANCE_MS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value / StreamPcmFormat::provider_default().frame_duration_ms as usize)
        .filter(|value| (MIN_MAX_UTTERANCE_FRAMES..=MAX_MAX_UTTERANCE_FRAMES).contains(value))
        .unwrap_or(DEFAULT_MAX_UTTERANCE_FRAMES)
}

fn configured_min_utterance_frames(max_utterance_frames: usize) -> usize {
    std::env::var("TRANSLATOR_VAD_MIN_UTTERANCE_MS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value / StreamPcmFormat::provider_default().frame_duration_ms as usize)
        .filter(|value| *value >= MIN_MIN_UTTERANCE_FRAMES)
        .map(|value| value.min(max_utterance_frames))
        .unwrap_or(DEFAULT_MIN_UTTERANCE_FRAMES.min(max_utterance_frames))
}

fn configured_adaptive_silence_frames() -> usize {
    std::env::var("TRANSLATOR_VAD_ADAPTIVE_SILENCE_MS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value / StreamPcmFormat::provider_default().frame_duration_ms as usize)
        .map(|value| value.clamp(1, END_OF_UTTERANCE_SILENCE_FRAMES))
        .unwrap_or(ADAPTIVE_END_OF_UTTERANCE_SILENCE_FRAMES)
}

fn rms(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let power = samples
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>()
        / samples.len() as f64;
    power.sqrt()
}

pub struct SpeechSegmenter<D> {
    stream_id: Uuid,
    detector: D,
    speech_confirmation_frames: usize,
    utterance_id: Option<Uuid>,
    pending_speech: Vec<PcmFrame>,
    trailing_silence: Vec<PcmFrame>,
    active_frames: usize,
    min_utterance_frames: usize,
    max_utterance_frames: usize,
    adaptive_silence_frames: usize,
    rearm_silence_frames: Option<usize>,
}

impl<D: VoiceDetector> SpeechSegmenter<D> {
    pub fn new(stream_id: Uuid, detector: D) -> Self {
        Self::with_confirmation_frames(stream_id, detector, configured_speech_confirmation_frames())
    }

    #[doc(hidden)]
    pub fn with_confirmation_frames(
        stream_id: Uuid,
        detector: D,
        speech_confirmation_frames: usize,
    ) -> Self {
        let max_utterance_frames = configured_max_utterance_frames();
        Self::with_confirmation_and_adaptive_frames(
            stream_id,
            detector,
            speech_confirmation_frames,
            configured_min_utterance_frames(max_utterance_frames),
            max_utterance_frames,
            configured_adaptive_silence_frames(),
        )
    }

    #[doc(hidden)]
    pub fn with_confirmation_and_max_frames(
        stream_id: Uuid,
        detector: D,
        speech_confirmation_frames: usize,
        max_utterance_frames: usize,
    ) -> Self {
        Self::with_confirmation_and_adaptive_frames(
            stream_id,
            detector,
            speech_confirmation_frames,
            max_utterance_frames,
            max_utterance_frames,
            END_OF_UTTERANCE_SILENCE_FRAMES,
        )
    }

    #[doc(hidden)]
    pub fn with_confirmation_and_adaptive_frames(
        stream_id: Uuid,
        detector: D,
        speech_confirmation_frames: usize,
        min_utterance_frames: usize,
        max_utterance_frames: usize,
        adaptive_silence_frames: usize,
    ) -> Self {
        let speech_confirmation_frames = speech_confirmation_frames.clamp(
            MIN_SPEECH_CONFIRMATION_FRAMES,
            MAX_SPEECH_CONFIRMATION_FRAMES,
        );
        let max_utterance_frames =
            max_utterance_frames.clamp(MIN_MAX_UTTERANCE_FRAMES, MAX_MAX_UTTERANCE_FRAMES);
        let min_utterance_frames =
            min_utterance_frames.clamp(speech_confirmation_frames, max_utterance_frames);
        let adaptive_silence_frames =
            adaptive_silence_frames.clamp(1, END_OF_UTTERANCE_SILENCE_FRAMES);
        Self {
            stream_id,
            detector,
            speech_confirmation_frames,
            utterance_id: None,
            pending_speech: Vec::with_capacity(speech_confirmation_frames),
            trailing_silence: Vec::with_capacity(END_OF_UTTERANCE_SILENCE_FRAMES),
            active_frames: 0,
            min_utterance_frames,
            max_utterance_frames,
            adaptive_silence_frames,
            rearm_silence_frames: None,
        }
    }

    pub const fn stream_id(&self) -> Uuid {
        self.stream_id
    }

    pub fn pending_frame_count(&self) -> usize {
        self.pending_speech.len() + self.trailing_silence.len()
    }

    pub fn process(&mut self, frame: PcmFrame) -> Result<Vec<CaptureEvent>, VadError> {
        if frame.format != StreamPcmFormat::provider_default() {
            return Err(VadError::UnsupportedFormat);
        }
        let samples = s16le_samples(frame.pcm());
        let voice = self.detector.is_voice(&samples)?;
        if let Some(silence_frames) = self.rearm_silence_frames {
            let silence_frames = if voice {
                0
            } else {
                silence_frames.saturating_add(1)
            };
            self.rearm_silence_frames =
                (silence_frames < END_OF_UTTERANCE_SILENCE_FRAMES).then_some(silence_frames);
            return Ok(Vec::new());
        }
        if let Some(utterance_id) = self.utterance_id {
            return Ok(self.process_active(frame, voice, utterance_id));
        }
        Ok(self.process_idle(frame, voice))
    }

    fn process_idle(&mut self, frame: PcmFrame, voice: bool) -> Vec<CaptureEvent> {
        if !voice {
            self.pending_speech.clear();
            return Vec::new();
        }
        self.pending_speech.push(frame);
        if self.pending_speech.len() < self.speech_confirmation_frames {
            return Vec::new();
        }
        let utterance_id = Uuid::new_v4();
        self.utterance_id = Some(utterance_id);
        self.active_frames = self.pending_speech.len();
        let capture_monotonic_ns = self.pending_speech[0].capture_monotonic_ns;
        let mut events = Vec::with_capacity(self.pending_speech.len() + 1);
        events.push(CaptureEvent::SpeechStarted {
            stream_id: self.stream_id,
            utterance_id,
            capture_monotonic_ns,
        });
        events.extend(
            self.pending_speech
                .drain(..)
                .map(|frame| CaptureEvent::Frame {
                    stream_id: self.stream_id,
                    utterance_id,
                    frame,
                    end_of_utterance: false,
                }),
        );
        events
    }

    fn process_active(
        &mut self,
        frame: PcmFrame,
        voice: bool,
        utterance_id: Uuid,
    ) -> Vec<CaptureEvent> {
        if voice {
            self.active_frames = self.active_frames.saturating_add(1);
            let mut events = self
                .trailing_silence
                .drain(..)
                .map(|frame| CaptureEvent::Frame {
                    stream_id: self.stream_id,
                    utterance_id,
                    frame,
                    end_of_utterance: false,
                })
                .collect::<Vec<_>>();
            events.push(CaptureEvent::Frame {
                stream_id: self.stream_id,
                utterance_id,
                frame,
                end_of_utterance: false,
            });
            if self.active_frames >= self.max_utterance_frames {
                self.finish_forced_utterance(&mut events);
            }
            return events;
        }
        self.trailing_silence.push(frame);
        self.active_frames = self.active_frames.saturating_add(1);
        let silence_frames = if self.active_frames >= self.min_utterance_frames {
            self.adaptive_silence_frames
        } else {
            END_OF_UTTERANCE_SILENCE_FRAMES
        };
        if self.trailing_silence.len() < silence_frames {
            if self.active_frames >= self.max_utterance_frames {
                return self.finish_with_trailing_silence(utterance_id, true);
            }
            return Vec::new();
        }
        self.finish_with_trailing_silence(utterance_id, false)
    }

    fn finish_with_trailing_silence(
        &mut self,
        utterance_id: Uuid,
        forced: bool,
    ) -> Vec<CaptureEvent> {
        let last = self
            .trailing_silence
            .pop()
            .expect("EOU silence threshold is non-zero");
        let mut events = self
            .trailing_silence
            .drain(..)
            .map(|frame| CaptureEvent::Frame {
                stream_id: self.stream_id,
                utterance_id,
                frame,
                end_of_utterance: false,
            })
            .collect::<Vec<_>>();
        events.push(CaptureEvent::Frame {
            stream_id: self.stream_id,
            utterance_id,
            frame: last,
            end_of_utterance: true,
        });
        self.utterance_id = None;
        self.active_frames = 0;
        if forced {
            self.rearm_silence_frames = Some(0);
        }
        events
    }

    fn finish_forced_utterance(&mut self, events: &mut [CaptureEvent]) {
        let CaptureEvent::Frame {
            end_of_utterance, ..
        } = events
            .last_mut()
            .expect("an active voice frame always emits a capture frame")
        else {
            unreachable!("an active voice frame cannot emit SpeechStarted");
        };
        *end_of_utterance = true;
        self.utterance_id = None;
        self.active_frames = 0;
    }
}

fn s16le_samples(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PulsePcmOperation {
    Capture,
    Playback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PulsePcmCommand {
    operation: PulsePcmOperation,
    program: &'static str,
    arguments: Vec<String>,
    playback_identity: Option<PlaybackIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlaybackIdentity {
    session_id: Uuid,
    device: String,
    stream_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PulsePlaybackRegistration {
    index: u32,
    session_id: Uuid,
    stream_name: String,
    device: String,
    process_id: u32,
}

impl PulsePlaybackRegistration {
    pub const fn index(&self) -> u32 {
        self.index
    }
    pub const fn session_id(&self) -> Uuid {
        self.session_id
    }
    pub fn stream_name(&self) -> &str {
        &self.stream_name
    }
    pub fn device(&self) -> &str {
        &self.device
    }
    pub const fn process_id(&self) -> u32 {
        self.process_id
    }
}

impl PulsePcmCommand {
    pub fn capture(device: &str, stream_name: &str) -> Self {
        Self::new(PulsePcmOperation::Capture, "parec", device, stream_name)
    }

    pub fn playback(device: &str, stream_name: &str) -> Self {
        let mut command = Self::new(PulsePcmOperation::Playback, "pacat", device, stream_name);
        command.arguments.push("--volume=0".to_owned());
        let session_id = Uuid::new_v4();
        command.arguments.push(format!(
            "--property=translator.playback_session={session_id}"
        ));
        command.playback_identity = Some(PlaybackIdentity {
            session_id,
            device: device.to_owned(),
            stream_name: stream_name.to_owned(),
        });
        command
    }

    pub fn round_trip_monitor_playback(device: &str, stream_name: &str) -> Self {
        Self::new(PulsePcmOperation::Playback, "pacat", device, stream_name)
    }

    pub fn virtual_peer_playback(device: &str, session_id: Uuid) -> Self {
        let mut command = Self::new(
            PulsePcmOperation::Playback,
            "pacat",
            device,
            "translator-virtual-peer",
        );
        command
            .arguments
            .retain(|argument| argument != "--client-name=translator-daemon");
        command
            .arguments
            .push("--client-name=translator-virtual-peer".to_owned());
        command
            .arguments
            .push("--property=translator.test_profile=human_round_trip".to_owned());
        command.arguments.push(format!(
            "--property=translator.self_test_session={session_id}"
        ));
        command
    }

    fn new(
        operation: PulsePcmOperation,
        program: &'static str,
        device: &str,
        stream_name: &str,
    ) -> Self {
        let mode = match operation {
            PulsePcmOperation::Capture => "--record",
            PulsePcmOperation::Playback => "--playback",
        };
        Self {
            operation,
            program,
            arguments: vec![
                mode.to_owned(),
                format!("--device={device}"),
                "--raw".to_owned(),
                "--format=s16le".to_owned(),
                "--rate=16000".to_owned(),
                "--channels=1".to_owned(),
                "--channel-map=mono".to_owned(),
                "--latency-msec=20".to_owned(),
                "--process-time-msec=20".to_owned(),
                "--client-name=translator-daemon".to_owned(),
                format!("--stream-name={stream_name}"),
                "--property=translator.owner=true".to_owned(),
                "--property=media.role=communication".to_owned(),
            ],
            playback_identity: None,
        }
    }

    pub const fn program(&self) -> &str {
        self.program
    }

    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }

    pub fn playback_session(&self) -> Option<Uuid> {
        self.playback_identity
            .as_ref()
            .map(|identity| identity.session_id)
    }
}

#[derive(Debug, Error)]
pub enum PulsePcmError {
    #[error("PCM worker could not be started")]
    Start,
    #[error("PCM worker pipe is unavailable")]
    PipeUnavailable,
    #[error("PCM capture failed")]
    Capture,
    #[error("PCM playback failed")]
    Playback,
    #[error("PCM playback stream registration could not be confirmed")]
    Registration,
    #[error("PCM worker could not be stopped")]
    Stop,
}

pub struct PulsePcmCapture {
    child: Child,
    output: Option<ChildStdout>,
    format: StreamPcmFormat,
    pending: Vec<u8>,
    filled: usize,
    metadata: Option<(u64, u64)>,
}

impl PulsePcmCapture {
    pub fn spawn(command: &PulsePcmCommand) -> Result<Self, PulsePcmError> {
        if command.operation != PulsePcmOperation::Capture {
            return Err(PulsePcmError::Start);
        }
        let mut child = Command::new(command.program)
            .args(&command.arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| PulsePcmError::Start)?;
        let output = child.stdout.take().ok_or(PulsePcmError::PipeUnavailable)?;
        Ok(Self {
            child,
            output: Some(output),
            format: StreamPcmFormat::provider_default(),
            pending: vec![0; StreamPcmFormat::provider_default().frame_bytes()],
            filled: 0,
            metadata: None,
        })
    }

    pub async fn read_frame(
        &mut self,
        sequence: u64,
        capture_monotonic_ns: u64,
    ) -> Result<PcmFrame, PulsePcmError> {
        let output = self.output.as_mut().ok_or(PulsePcmError::Capture)?;
        while self.filled < self.pending.len() {
            let count = output
                .read(&mut self.pending[self.filled..])
                .await
                .map_err(|_| PulsePcmError::Capture)?;
            if count == 0 {
                return Err(PulsePcmError::Capture);
            }
            self.metadata
                .get_or_insert((sequence, capture_monotonic_ns));
            self.filled += count;
        }
        let (sequence, capture_monotonic_ns) = self
            .metadata
            .take()
            .expect("complete PCM frame has first-byte metadata");
        let pcm = std::mem::replace(&mut self.pending, vec![0; self.format.frame_bytes()]);
        self.filled = 0;
        PcmFrame::try_new(sequence, capture_monotonic_ns, self.format, pcm)
            .map_err(|_| PulsePcmError::Capture)
    }

    pub async fn stop(&mut self) -> Result<(), PulsePcmError> {
        drop(self.output.take());
        self.pending.fill(0);
        self.filled = 0;
        self.metadata = None;
        stop_child(&mut self.child).await
    }
}

pub struct PulsePcmPlayback {
    child: Child,
    input: Option<ChildStdin>,
    playback_identity: Option<PlaybackIdentity>,
}

impl PulsePcmPlayback {
    pub fn spawn(command: &PulsePcmCommand) -> Result<Self, PulsePcmError> {
        if command.operation != PulsePcmOperation::Playback {
            return Err(PulsePcmError::Start);
        }
        let mut child = Command::new(command.program)
            .args(&command.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| PulsePcmError::Start)?;
        let input = child.stdin.take().ok_or(PulsePcmError::PipeUnavailable)?;
        Ok(Self {
            child,
            input: Some(input),
            playback_identity: command.playback_identity.clone(),
        })
    }

    pub async fn wait_registered_muted(
        &mut self,
        deadline: Instant,
    ) -> Result<PulsePlaybackRegistration, PulsePcmError> {
        let identity = self
            .playback_identity
            .as_ref()
            .ok_or(PulsePcmError::Registration)?;
        let pid = self.child.id().ok_or(PulsePcmError::Registration)?;
        loop {
            if Instant::now() >= deadline
                || self
                    .child
                    .try_wait()
                    .map_err(|_| PulsePcmError::Registration)?
                    .is_some()
            {
                return Err(PulsePcmError::Registration);
            }
            let inputs: Vec<RawPlaybackInput> =
                pactl_json(&["--format=json", "list", "sink-inputs"], deadline).await?;
            if inputs.iter().any(|input| {
                input
                    .properties
                    .get("translator.playback_session")
                    .is_some_and(|value| value == &identity.session_id.to_string())
            }) {
                let sinks: Vec<RawPlaybackSink> =
                    pactl_json(&["--format=json", "list", "sinks"], deadline).await?;
                let registration = find_playback_registration(identity, pid, &inputs, &sinks)?
                    .ok_or(PulsePcmError::Registration)?;
                if self
                    .child
                    .try_wait()
                    .map_err(|_| PulsePcmError::Registration)?
                    .is_some()
                {
                    return Err(PulsePcmError::Registration);
                }
                return Ok(registration);
            }
            let next_probe = (Instant::now() + Duration::from_millis(20)).min(deadline);
            tokio::time::sleep_until(next_probe.into()).await;
        }
    }

    pub async fn write_frame(&mut self, frame: &PcmFrame) -> Result<(), PulsePcmError> {
        self.input
            .as_mut()
            .ok_or(PulsePcmError::Playback)?
            .write_all(frame.pcm())
            .await
            .map_err(|_| PulsePcmError::Playback)
    }

    pub async fn flush(&mut self) -> Result<(), PulsePcmError> {
        self.input
            .as_mut()
            .ok_or(PulsePcmError::Playback)?
            .flush()
            .await
            .map_err(|_| PulsePcmError::Playback)
    }

    pub fn process_identity(&self) -> Option<crate::ProcessIdentity> {
        crate::ProcessIdentity::inspect(self.child.id()?)
    }

    pub async fn finish(&mut self, timeout: Duration) -> Result<(), PulsePcmError> {
        let finish = async {
            // Closing stdin delivers EOF without transferring ownership of the child.
            self.input.take();
            match self.child.wait().await {
                Ok(status) if status.success() => Ok(()),
                Ok(_) | Err(_) => Err(PulsePcmError::Playback),
            }
        };
        tokio::time::timeout(timeout, finish)
            .await
            .map_err(|_| PulsePcmError::Stop)?
    }

    pub async fn stop(&mut self) -> Result<(), PulsePcmError> {
        stop_child(&mut self.child).await
    }
}

#[derive(Debug, Deserialize)]
struct RawPlaybackInput {
    index: u32,
    sink: Option<u32>,
    #[serde(default)]
    channel_map: String,
    #[serde(default)]
    volume: HashMap<String, RawPlaybackVolume>,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawPlaybackVolume {
    value: u32,
}

#[derive(Debug, Deserialize)]
struct RawPlaybackSink {
    index: u32,
    name: String,
}

fn find_playback_registration(
    identity: &PlaybackIdentity,
    process_id: u32,
    inputs: &[RawPlaybackInput],
    sinks: &[RawPlaybackSink],
) -> Result<Option<PulsePlaybackRegistration>, PulsePcmError> {
    let mut matched = inputs.iter().filter(|input| {
        input.properties.get("translator.playback_session")
            == Some(&identity.session_id.to_string())
    });
    let Some(input) = matched.next() else {
        return Ok(None);
    };
    if matched.next().is_some()
        || input.properties.get("translator.owner").map(String::as_str) != Some("true")
        || input.properties.get("application.name").map(String::as_str) != Some("translator-daemon")
        || input.properties.get("application.process.id") != Some(&process_id.to_string())
        || input.properties.get("media.name") != Some(&identity.stream_name)
    {
        return Err(PulsePcmError::Registration);
    }
    let sink = input.sink.ok_or(PulsePcmError::Registration)?;
    if sinks
        .iter()
        .filter(|candidate| candidate.index == sink)
        .map(|candidate| candidate.name.as_str())
        .collect::<Vec<_>>()
        != [identity.device.as_str()]
    {
        return Err(PulsePcmError::Registration);
    }
    let channels: Vec<_> = input.channel_map.split(',').collect();
    let unique: HashSet<_> = channels.iter().copied().collect();
    if channels.is_empty()
        || channels.iter().any(|channel| channel.is_empty())
        || channels.len() != unique.len()
        || channels.len() != input.volume.len()
        || channels.iter().any(|channel| {
            input
                .volume
                .get(*channel)
                .is_none_or(|volume| volume.value != 0)
        })
    {
        return Err(PulsePcmError::Registration);
    }
    Ok(Some(PulsePlaybackRegistration {
        index: input.index,
        session_id: identity.session_id,
        stream_name: identity.stream_name.clone(),
        device: identity.device.clone(),
        process_id,
    }))
}

async fn pactl_json<T: for<'de> Deserialize<'de>>(
    args: &[&str],
    deadline: Instant,
) -> Result<T, PulsePcmError> {
    let output = tokio::time::timeout_at(
        deadline.into(),
        Command::new("pactl")
            .args(args)
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| PulsePcmError::Registration)?
    .map_err(|_| PulsePcmError::Registration)?;
    if !output.status.success() {
        return Err(PulsePcmError::Registration);
    }
    serde_json::from_slice(&output.stdout).map_err(|_| PulsePcmError::Registration)
}

async fn stop_child(child: &mut Child) -> Result<(), PulsePcmError> {
    if child.try_wait().map_err(|_| PulsePcmError::Stop)?.is_some() {
        return Ok(());
    }
    child.kill().await.map_err(|_| PulsePcmError::Stop)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat_capture() -> (PulsePcmCapture, ChildStdin) {
        let mut child = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        (
            PulsePcmCapture {
                child,
                output: Some(output),
                format: StreamPcmFormat::provider_default(),
                pending: vec![0; StreamPcmFormat::provider_default().frame_bytes()],
                filled: 0,
                metadata: None,
            },
            input,
        )
    }

    #[tokio::test]
    async fn capture_cancellation_preserves_consumed_prefix_alignment_and_first_metadata() {
        let (mut capture, mut input) = cat_capture();
        let size = StreamPcmFormat::provider_default().frame_bytes();
        let expected: Vec<_> = (0..size).map(|n| (n % 251) as u8).collect();
        let next: Vec<_> = (0..size).map(|n| 255 - (n % 251) as u8).collect();
        let result = tokio::time::timeout(Duration::from_secs(2), async {
            let empty_cancelled =
                tokio::time::timeout(Duration::from_millis(10), capture.read_frame(1, 10))
                    .await
                    .is_err();
            let mut consumed = Vec::new();
            for (index, range) in [0..17, 17..319].into_iter().enumerate() {
                let part = &expected[range];
                input.write_all(part).await?;
                while rustix::io::ioctl_fionread(capture.output.as_ref().unwrap())?
                    < part.len() as u64
                {
                    tokio::task::yield_now().await;
                }
                let cancelled = tokio::time::timeout(
                    Duration::from_millis(10),
                    capture.read_frame(7 + index as u64, 100 + index as u64),
                )
                .await
                .is_err();
                consumed.push((
                    cancelled,
                    rustix::io::ioctl_fionread(capture.output.as_ref().unwrap())?,
                ));
            }
            input.write_all(&expected[319..]).await?;
            input.write_all(&next).await?;
            drop(input);
            let first = capture.read_frame(90, 900).await;
            let second = capture.read_frame(8, 200).await;
            let eof = capture.read_frame(9, 300).await;
            Ok::<_, std::io::Error>((empty_cancelled, consumed, first, second, eof))
        })
        .await;
        let stopped = tokio::time::timeout(Duration::from_secs(1), capture.stop()).await;
        let reaped = capture.child.try_wait().unwrap().is_some();
        assert!(stopped.unwrap().is_ok() && reaped);
        let (empty_cancelled, consumed, first, second, eof) = result.unwrap().unwrap();
        assert!(empty_cancelled);
        assert_eq!(consumed, vec![(true, 0), (true, 0)]);
        let first = first.unwrap();
        assert!(
            first.pcm() == expected,
            "cancellation discarded or shifted captured PCM bytes"
        );
        assert_eq!((first.sequence(), first.capture_monotonic_ns()), (7, 100));
        let second = second.unwrap();
        assert!(second.pcm() == next, "the next PCM frame lost alignment");
        assert_eq!((second.sequence(), second.capture_monotonic_ns()), (8, 200));
        assert!(matches!(eof, Err(PulsePcmError::Capture)));
    }

    #[tokio::test]
    async fn capture_partial_eof_never_emits_a_frame() {
        let (mut capture, mut input) = cat_capture();
        let result = tokio::time::timeout(Duration::from_secs(2), async {
            input.write_all(&[19; 17]).await?;
            drop(input);
            Ok::<_, std::io::Error>(capture.read_frame(7, 100).await)
        })
        .await;
        let stopped = tokio::time::timeout(Duration::from_secs(1), capture.stop()).await;
        let reaped = capture.child.try_wait().unwrap().is_some();
        assert!(stopped.unwrap().is_ok() && reaped);
        assert!(matches!(
            result.unwrap().unwrap(),
            Err(PulsePcmError::Capture)
        ));
    }

    #[tokio::test]
    async fn capture_stop_prevents_buffered_emission() {
        let (mut capture, mut input) = cat_capture();
        let size = StreamPcmFormat::provider_default().frame_bytes();
        let prepared = tokio::time::timeout(Duration::from_secs(2), async {
            input.write_all(&[31; 17]).await?;
            while rustix::io::ioctl_fionread(capture.output.as_ref().unwrap())? < 17 {
                tokio::task::yield_now().await;
            }
            let cancelled =
                tokio::time::timeout(Duration::from_millis(10), capture.read_frame(7, 100))
                    .await
                    .is_err();
            let consumed = rustix::io::ioctl_fionread(capture.output.as_ref().unwrap())? == 0;
            input.write_all(&vec![42; size]).await?;
            while rustix::io::ioctl_fionread(capture.output.as_ref().unwrap())? < size as u64 {
                tokio::task::yield_now().await;
            }
            drop(input);
            Ok::<_, std::io::Error>((cancelled, consumed))
        })
        .await;
        let stopped = tokio::time::timeout(Duration::from_secs(1), capture.stop()).await;
        let reaped = capture.child.try_wait().unwrap().is_some();
        let after_stop =
            tokio::time::timeout(Duration::from_millis(100), capture.read_frame(8, 200)).await;
        let invalidated = capture.output.is_none()
            && capture.filled == 0
            && capture.metadata.is_none()
            && capture.pending.iter().all(|byte| *byte == 0);
        let stopped_again = tokio::time::timeout(Duration::from_secs(1), capture.stop()).await;
        assert!(stopped.unwrap().is_ok() && reaped);
        assert!(stopped_again.unwrap().is_ok());
        assert!(invalidated);
        assert_eq!(prepared.unwrap().unwrap(), (true, true));
        assert!(
            matches!(after_stop, Ok(Err(PulsePcmError::Capture))),
            "stopped capture emitted residual PCM"
        );
    }

    fn shell_playback(script: &str) -> PulsePcmPlayback {
        let mut child = Command::new("sh")
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        PulsePcmPlayback {
            child,
            input: Some(input),
            playback_identity: None,
        }
    }

    #[test]
    fn production_playback_spawns_have_distinct_registration_ids() {
        let first = PulsePcmCommand::playback("translator_mic_out", "translator-outgoing-playback");
        let second =
            PulsePcmCommand::playback("translator_mic_out", "translator-outgoing-playback");
        assert_ne!(first.playback_session(), second.playback_session());
        for command in [&first, &second] {
            let session = command.playback_session().unwrap();
            assert!(
                command
                    .arguments()
                    .contains(&format!("--property=translator.playback_session={session}"))
            );
            assert!(command.arguments().contains(&"--volume=0".to_owned()));
        }
    }

    #[test]
    fn round_trip_monitor_is_not_a_muted_production_translation_stream() {
        let monitor = PulsePcmCommand::round_trip_monitor_playback(
            "alsa_output.private_monitor",
            "translator-round-trip-english-monitor",
        );
        assert_eq!(monitor.playback_session(), None);
        assert!(!monitor.arguments().contains(&"--volume=0".to_owned()));
        assert!(
            monitor
                .arguments()
                .contains(&"--device=alsa_output.private_monitor".to_owned())
        );
    }

    #[test]
    fn registration_requires_exact_session_process_sink_and_zero_volume() {
        let command =
            PulsePcmCommand::playback("translator_mic_out", "translator-outgoing-playback");
        let identity = command.playback_identity.as_ref().unwrap();
        let pid = 421u32;
        let input = |session: Uuid, volume: u32| {
            serde_json::json!({
                "index": 9,
                "sink": 7,
                "channel_map": "mono",
                "volume": {"mono": {"value": volume}},
                "properties": {
                    "translator.playback_session": session.to_string(),
                    "translator.owner": "true",
                    "application.name": "translator-daemon",
                    "application.process.id": pid.to_string(),
                    "media.name": "translator-outgoing-playback"
                }
            })
        };
        let sinks: Vec<RawPlaybackSink> = serde_json::from_value(serde_json::json!([
            {"index": 7, "name": "translator_mic_out"}
        ]))
        .unwrap();
        let valid: Vec<RawPlaybackInput> =
            serde_json::from_value(serde_json::json!([input(identity.session_id, 0)])).unwrap();
        assert_eq!(
            find_playback_registration(identity, pid, &valid, &sinks)
                .unwrap()
                .unwrap()
                .index(),
            9
        );
        let other: Vec<RawPlaybackInput> =
            serde_json::from_value(serde_json::json!([input(Uuid::new_v4(), 0)])).unwrap();
        assert!(
            find_playback_registration(identity, pid, &other, &sinks)
                .unwrap()
                .is_none()
        );
        let audible: Vec<RawPlaybackInput> =
            serde_json::from_value(serde_json::json!([input(identity.session_id, 65536)])).unwrap();
        assert!(find_playback_registration(identity, pid, &audible, &sinks).is_err());
        let duplicate: Vec<RawPlaybackInput> = serde_json::from_value(serde_json::json!([
            input(identity.session_id, 0),
            input(identity.session_id, 0)
        ]))
        .unwrap();
        assert!(find_playback_registration(identity, pid, &duplicate, &sinks).is_err());
        let wrong_sink: Vec<RawPlaybackSink> = serde_json::from_value(serde_json::json!([
            {"index": 7, "name": "not_translator_mic_out"}
        ]))
        .unwrap();
        assert!(find_playback_registration(identity, pid, &valid, &wrong_sink).is_err());
    }

    #[test]
    fn rms_tracks_frame_energy_for_vad_gate() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[300, -300, 300, -300]), 300.0);
        assert!(rms(&[0, 0, 600, -600]) > DEFAULT_MIN_VOICE_RMS);
    }

    #[tokio::test]
    async fn playback_finish_accepts_a_clean_eof_exit() {
        let mut playback = shell_playback("cat >/dev/null");
        playback
            .input
            .as_mut()
            .unwrap()
            .write_all(b"pcm")
            .await
            .unwrap();

        playback.finish(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn canceled_finish_retains_the_exact_child_after_observed_eof() {
        let mut playback = shell_playback("cat >/dev/null; printf x; while :; do :; done");
        let identity = playback.process_identity().unwrap();
        let mut output = playback.child.stdout.take().unwrap();
        {
            let finish = playback.finish(Duration::from_secs(2));
            tokio::pin!(finish);
            let mut byte = [0];
            tokio::select! {
                result = &mut finish => panic!("finish returned before cancellation: {result:?}"),
                result = tokio::time::timeout(Duration::from_secs(1), output.read_exact(&mut byte)) => {
                    result.unwrap().unwrap();
                    assert_eq!(byte, *b"x");
                }
            }
        }
        let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
        let mut retained = true;
        while tokio::time::Instant::now() < deadline {
            if crate::ProcessIdentity::inspect(identity.pid) != Some(identity) {
                retained = false;
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        playback.stop().await.unwrap();
        assert!(playback.child.try_wait().unwrap().is_some());
        assert_ne!(
            crate::ProcessIdentity::inspect(identity.pid),
            Some(identity)
        );
        assert!(
            retained,
            "canceling a wait must not destroy the playback owner"
        );
    }

    #[tokio::test]
    async fn playback_finish_bounds_the_complete_shutdown_and_wait_sequence() {
        let mut playback = shell_playback("while :; do :; done");
        let identity = playback.process_identity().unwrap();
        let started = std::time::Instant::now();

        assert!(matches!(
            playback
                .finish(Duration::from_millis(50))
                .await
                .unwrap_err(),
            PulsePcmError::Stop
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
        let retained = crate::ProcessIdentity::inspect(identity.pid) == Some(identity);
        assert!(playback.flush().await.is_err());
        let frame =
            PcmFrame::try_new(0, 0, StreamPcmFormat::provider_default(), vec![0; 640]).unwrap();
        assert!(playback.write_frame(&frame).await.is_err());
        playback.stop().await.unwrap();
        playback.stop().await.unwrap();
        assert!(playback.child.try_wait().unwrap().is_some());
        assert!(
            retained,
            "timeout must retain the same child for explicit cleanup"
        );
    }

    #[tokio::test]
    async fn nonzero_finish_is_a_semantic_error_with_confirmed_cleanup() {
        let mut playback = shell_playback("exit 7");
        assert!(matches!(
            playback.finish(Duration::from_secs(1)).await,
            Err(PulsePcmError::Playback)
        ));
        assert!(playback.child.try_wait().unwrap().is_some());
        playback.stop().await.unwrap();
        playback.stop().await.unwrap();
    }
}
