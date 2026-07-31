use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioDirection {
    #[serde(rename = "microphone")]
    Microphone,
    #[serde(rename = "speaker")]
    Speaker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranslationMode {
    #[serde(rename = "quality_first")]
    QualityFirst,
    #[serde(rename = "balanced")]
    Balanced,
    #[serde(rename = "streaming_first")]
    StreamingFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Ru,
    En,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceGender {
    Male,
    Female,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceEngine {
    Piper,
    Silero,
    Openai,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceProfile {
    pub language: Language,
    pub gender: VoiceGender,
    pub engine: VoiceEngine,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_voice_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleFormat {
    #[serde(rename = "s16le")]
    S16Le,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PcmFormatError {
    #[error("unsupported sample rate")]
    SampleRate,
    #[error("unsupported channel count")]
    Channels,
    #[error("unsupported frame duration")]
    FrameDuration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PcmFormat {
    sample_rate_hz: u32,
    channels: u8,
    sample_format: SampleFormat,
    frame_duration_ms: u16,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncheckedPcmFormat {
    sample_rate_hz: u32,
    channels: u8,
    sample_format: SampleFormat,
    frame_duration_ms: u16,
}

impl<'de> Deserialize<'de> for PcmFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let unchecked = UncheckedPcmFormat::deserialize(deserializer)?;
        Self::try_new(
            unchecked.sample_rate_hz,
            unchecked.channels,
            unchecked.sample_format,
            unchecked.frame_duration_ms,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl PcmFormat {
    pub fn try_new(
        sample_rate_hz: u32,
        channels: u8,
        sample_format: SampleFormat,
        frame_duration_ms: u16,
    ) -> Result<Self, PcmFormatError> {
        if !matches!(sample_rate_hz, 16_000 | 24_000 | 48_000) {
            return Err(PcmFormatError::SampleRate);
        }
        if !matches!(channels, 1 | 2) {
            return Err(PcmFormatError::Channels);
        }
        if !matches!(frame_duration_ms, 20 | 40 | 60 | 80 | 100) {
            return Err(PcmFormatError::FrameDuration);
        }
        Ok(Self {
            sample_rate_hz,
            channels,
            sample_format,
            frame_duration_ms,
        })
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    pub fn sample_format(&self) -> SampleFormat {
        self.sample_format
    }

    pub fn frame_duration_ms(&self) -> u16 {
        self.frame_duration_ms
    }
}

macro_rules! define_schema_version {
    ($name:ident, $wire_value:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum $name {
            #[serde(rename = $wire_value)]
            V1,
        }
    };
}

define_schema_version!(OpenSessionVersion, "translator.provider.open_session.v1");
define_schema_version!(
    SessionOpenedVersion,
    "translator.provider.session_opened.v1"
);
define_schema_version!(
    ProviderProbeRequestVersion,
    "translator.provider.probe_request.v1"
);
define_schema_version!(
    ProviderProbeResponseVersion,
    "translator.provider.probe_response.v1"
);
define_schema_version!(CloseSessionVersion, "translator.provider.close_session.v1");
define_schema_version!(
    CancelUtteranceVersion,
    "translator.provider.cancel_utterance.v1"
);
define_schema_version!(
    UpdateDebugTextVersion,
    "translator.provider.update_debug_text.v1"
);
define_schema_version!(ProviderInputVersion, "translator.provider.input.v1");
define_schema_version!(
    ProviderAudioDeltaVersion,
    "translator.provider.audio_delta.v1"
);
define_schema_version!(
    ProviderTranscriptDeltaVersion,
    "translator.provider.transcript_delta.v1"
);
define_schema_version!(
    ProviderTranslationDeltaVersion,
    "translator.provider.translation_delta.v1"
);
define_schema_version!(
    ProviderUtteranceFinalVersion,
    "translator.provider.utterance_final.v1"
);
define_schema_version!(
    SessionClosedVersion,
    "translator.provider.session_closed.v1"
);
define_schema_version!(ProviderHealthVersion, "translator.provider.health.v1");
define_schema_version!(ProviderLatencyVersion, "translator.provider.latency.v1");
define_schema_version!(ProviderErrorVersion, "translator.provider.error.v1");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenProviderSession {
    pub schema_version: OpenSessionVersion,
    pub session_id: Uuid,
    pub provider_id: ProviderId,
    pub direction_id: AudioDirection,
    pub source_language: Language,
    pub target_language: Language,
    pub mode: TranslationMode,
    pub requested_input_format: PcmFormat,
    pub requested_output_format: PcmFormat,
    pub voice_profile: VoiceProfile,
    pub debug_text_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCapabilities {
    pub audio_output: RequiredTrue,
    pub transcript_delta: bool,
    pub translation_delta: bool,
    pub cancellation: bool,
    pub cloud_egress: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredTrue;

impl Serialize for RequiredTrue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(true)
    }
}

impl<'de> Deserialize<'de> for RequiredTrue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if bool::deserialize(deserializer)? {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom("value must be true"))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSessionOpened {
    pub schema_version: SessionOpenedVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub event_sequence: u64,
    pub negotiated_input_format: PcmFormat,
    pub negotiated_output_format: PcmFormat,
    pub capabilities: ProviderCapabilities,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderProbeRequest {
    pub schema_version: ProviderProbeRequestVersion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderProbeResponse {
    pub schema_version: ProviderProbeResponseVersion,
    pub generation_id: Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloseRequestReason {
    UserStop,
    RouteRemoved,
    DeviceUnavailable,
    ProviderSwitch,
    DaemonShutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionCloseReason {
    UserStop,
    RouteRemoved,
    DeviceUnavailable,
    ProviderSwitch,
    DaemonShutdown,
    ProviderFailure,
    CloseTimeout,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloseProviderSession {
    pub schema_version: CloseSessionVersion,
    pub session_id: Uuid,
    pub reason: CloseRequestReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelReason {
    LatencyPolicy,
    RouteRemoved,
    UserInterrupt,
    QueueOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancelUtterance {
    pub schema_version: CancelUtteranceVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub utterance_id: Uuid,
    pub reason: CancelReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateDebugText {
    pub schema_version: UpdateDebugTextVersion,
    pub session_id: Uuid,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInputFrame {
    pub schema_version: ProviderInputVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub stream_id: Uuid,
    pub utterance_id: Uuid,
    pub sequence: u64,
    pub capture_monotonic_ns: u64,
    #[serde(flatten)]
    pub format: PcmFormat,
    pub source_language: Language,
    pub target_language: Language,
    pub mode: TranslationMode,
    pub pcm: Vec<u8>,
    pub end_of_utterance: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderAudioDelta {
    pub schema_version: ProviderAudioDeltaVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub stream_id: Uuid,
    pub utterance_id: Uuid,
    pub sequence: u64,
    pub event_sequence: u64,
    pub provider_monotonic_ns: u64,
    #[serde(flatten)]
    pub format: PcmFormat,
    pub pcm: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTranscriptDelta {
    pub schema_version: ProviderTranscriptDeltaVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub stream_id: Uuid,
    pub utterance_id: Uuid,
    pub event_sequence: u64,
    pub text: String,
    pub is_final: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTranslationDelta {
    pub schema_version: ProviderTranslationDeltaVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub stream_id: Uuid,
    pub utterance_id: Uuid,
    pub event_sequence: u64,
    pub text: String,
    pub stable_prefix: bool,
    pub is_final: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtteranceOutcome {
    Completed,
    Cancelled,
    Dropped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderUtteranceFinal {
    pub schema_version: ProviderUtteranceFinalVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub stream_id: Uuid,
    pub utterance_id: Uuid,
    pub event_sequence: u64,
    pub final_audio_sequence: Option<u64>,
    pub outcome: UtteranceOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSessionClosed {
    pub schema_version: SessionClosedVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub event_sequence: u64,
    pub reason: SessionCloseReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderId {
    Local,
    Openai,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderState {
    Starting,
    Ready,
    Degraded,
    Backpressure,
    Restarting,
    Unavailable,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Asr,
    Mt,
    Tts,
    SpeechToSpeech,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    NotLoaded,
    Loading,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cuda,
    Cpu,
    Cloud,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelHealth {
    pub kind: ModelKind,
    pub id: String,
    pub state: ModelState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<ComputeDevice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_error_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderQueues {
    pub provider_input_buffered_ms: u32,
    pub provider_output_buffered_ms: u32,
    pub queue_lag_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRetry {
    pub attempt: u32,
    pub next_retry_after_ms: u32,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafeErrorSummary {
    pub code: crate::SafeErrorCode,
    pub message: crate::SafeMessage,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderHealth {
    pub schema_version: ProviderHealthVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub event_sequence: u64,
    pub provider_id: ProviderId,
    pub provider_name: String,
    pub state: ProviderState,
    pub models: Vec<ModelHealth>,
    pub queues: ProviderQueues,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<ProviderRetry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_error: Option<SafeErrorSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderLatency {
    pub schema_version: ProviderLatencyVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    pub stream_id: Uuid,
    pub event_sequence: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utterance_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr_first_text_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr_final_text_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mt_first_text_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts_first_audio_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_total_ms: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LatencyPolicyState {
    pub direction_id: AudioDirection,
    pub current_mode: TranslationMode,
    rolling_window_seconds: u32,
    minimum_samples: u32,
    degrade_after_consecutive_windows: u32,
    recover_after_consecutive_windows: u32,
    cooldown_seconds_after_change: u32,
    pub p95_first_audio_ms: u32,
    pub p95_last_audio_ms: u32,
    pub p95_queue_lag_ms: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_mode_change_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl LatencyPolicyState {
    pub fn new(
        direction_id: AudioDirection,
        current_mode: TranslationMode,
        p95_first_audio_ms: u32,
        p95_last_audio_ms: u32,
        p95_queue_lag_ms: u32,
        last_mode_change_at: Option<String>,
        reason: Option<String>,
    ) -> Self {
        Self {
            direction_id,
            current_mode,
            rolling_window_seconds: 60,
            minimum_samples: 20,
            degrade_after_consecutive_windows: 2,
            recover_after_consecutive_windows: 5,
            cooldown_seconds_after_change: 120,
            p95_first_audio_ms,
            p95_last_audio_ms,
            p95_queue_lag_ms,
            last_mode_change_at,
            reason,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncheckedLatencyPolicyState {
    direction_id: AudioDirection,
    current_mode: TranslationMode,
    rolling_window_seconds: u32,
    minimum_samples: u32,
    degrade_after_consecutive_windows: u32,
    recover_after_consecutive_windows: u32,
    cooldown_seconds_after_change: u32,
    p95_first_audio_ms: u32,
    p95_last_audio_ms: u32,
    p95_queue_lag_ms: u32,
    last_mode_change_at: Option<String>,
    reason: Option<String>,
}

impl<'de> Deserialize<'de> for LatencyPolicyState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let unchecked = UncheckedLatencyPolicyState::deserialize(deserializer)?;
        if (
            unchecked.rolling_window_seconds,
            unchecked.minimum_samples,
            unchecked.degrade_after_consecutive_windows,
            unchecked.recover_after_consecutive_windows,
            unchecked.cooldown_seconds_after_change,
        ) != (60, 20, 2, 5, 120)
        {
            return Err(serde::de::Error::custom(
                "latency policy constants do not match the contract",
            ));
        }
        Ok(Self {
            direction_id: unchecked.direction_id,
            current_mode: unchecked.current_mode,
            rolling_window_seconds: unchecked.rolling_window_seconds,
            minimum_samples: unchecked.minimum_samples,
            degrade_after_consecutive_windows: unchecked.degrade_after_consecutive_windows,
            recover_after_consecutive_windows: unchecked.recover_after_consecutive_windows,
            cooldown_seconds_after_change: unchecked.cooldown_seconds_after_change,
            p95_first_audio_ms: unchecked.p95_first_audio_ms,
            p95_last_audio_ms: unchecked.p95_last_audio_ms,
            p95_queue_lag_ms: unchecked.p95_queue_lag_ms,
            last_mode_change_at: unchecked.last_mode_change_at,
            reason: unchecked.reason,
        })
    }
}
