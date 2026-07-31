use std::collections::HashSet;

use serde::Serialize;
use thiserror::Error;
use translator_audio::{CaptureEvent, PcmFrame, PcmFrameError, StreamPcmFormat};
use translator_core::{
    AudioDirection, Language, ProviderId, TranslationMode, VoiceEngine, VoiceGender,
};
use translator_ipc::{
    ProviderSessionContract, ProviderValidationError,
    provider::{
        AudioDirection as ProviderDirection, CancelReason, CancelUtterance, CloseProviderSession,
        CloseRequestReason, Language as ProviderLanguage, OpenProviderSession, PcmFormat,
        ProviderId as ProviderProviderId, ProviderRequest, SafeErrorCode, SampleFormat,
        TranslationMode as ProviderTranslationMode, UtteranceOutcome as ProviderUtteranceOutcome,
        VoiceEngine as ProviderVoiceEngine, VoiceGender as ProviderVoiceGender, VoiceProfile,
        provider_event, provider_request,
    },
};
use uuid::Uuid;

use crate::{ProviderStreamCoordinator, ProviderStreamCoordinatorError, WatchdogAction};

const OPEN_SCHEMA: &str = "translator.provider.open_session.v1";
const INPUT_SCHEMA: &str = "translator.provider.input.v1";
const CANCEL_SCHEMA: &str = "translator.provider.cancel_utterance.v1";
const CLOSE_SCHEMA: &str = "translator.provider.close_session.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectionRuntimeConfig {
    pub provider_id: ProviderId,
    pub direction: AudioDirection,
    pub source_language: Language,
    pub target_language: Language,
    pub mode: TranslationMode,
    pub voice_gender: VoiceGender,
    pub voice_engine: VoiceEngine,
    pub debug_text_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SafeProviderErrorCode {
    ProviderUnavailable,
    ModelNotLoaded,
    UnsupportedLanguagePair,
    QueueOverflow,
    Cancelled,
    CloudNotEnabled,
    ProviderAuthFailed,
    NoSpeech,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcome {
    Completed,
    Cancelled,
    Dropped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectionEffect {
    Playback {
        stream_id: Uuid,
        utterance_id: Uuid,
        frame: PcmFrame,
    },
    TranscriptFinal {
        utterance_id: Uuid,
    },
    TranslationFinal {
        utterance_id: Uuid,
    },
    Latency {
        utterance_id: Option<Uuid>,
        tts_first_audio_ms: Option<u32>,
        provider_total_ms: Option<u32>,
    },
    ProviderError {
        utterance_id: Option<Uuid>,
        code: SafeProviderErrorCode,
        retryable: bool,
    },
    UtteranceTerminalOutcome {
        utterance_id: Uuid,
        outcome: TerminalOutcome,
    },
    UtteranceTerminal {
        utterance_id: Uuid,
    },
    ExpiredAudio {
        utterance_id: Uuid,
        request: ProviderRequest,
    },
    SessionClosed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectionWatchdogEffect {
    Send(ProviderRequest),
    PurgeAndSend(ProviderRequest),
    RestartSidecar,
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum DirectionSessionError {
    #[error("capture stream does not match provider stream")]
    StreamMismatch,
    #[error("capture utterance state is invalid")]
    InvalidCaptureState,
    #[error("provider audio frame is invalid")]
    InvalidProviderAudio,
    #[error("provider metadata is invalid")]
    InvalidProviderMetadata,
    #[error(transparent)]
    Coordinator(#[from] ProviderStreamCoordinatorError),
}

impl From<PcmFrameError> for DirectionSessionError {
    fn from(_: PcmFrameError) -> Self {
        Self::InvalidProviderAudio
    }
}

pub struct DirectionSession {
    config: DirectionRuntimeConfig,
    contract: ProviderSessionContract,
    coordinator: ProviderStreamCoordinator,
    next_input_sequence: u64,
    collecting_utterance: Option<Uuid>,
    active_utterances: HashSet<Uuid>,
}

impl DirectionSession {
    pub fn new(config: DirectionRuntimeConfig) -> Self {
        let session_id = Uuid::new_v4();
        let stream_id = Uuid::new_v4();
        let format = provider_pcm_format();
        let contract = ProviderSessionContract {
            session_id,
            stream_id,
            provider_id: provider_id(config.provider_id),
            direction_id: provider_direction(config.direction),
            source_language: provider_language(config.source_language),
            target_language: provider_language(config.target_language),
            mode: provider_mode(config.mode),
            input_format: format,
            output_format: format,
            debug_text_enabled: config.debug_text_enabled,
        };
        let coordinator = ProviderStreamCoordinator::new(contract.clone());
        Self {
            config,
            contract,
            coordinator,
            next_input_sequence: 0,
            collecting_utterance: None,
            active_utterances: HashSet::new(),
        }
    }

    pub const fn session_id(&self) -> Uuid {
        self.contract.session_id
    }

    pub const fn stream_id(&self) -> Uuid {
        self.contract.stream_id
    }

    pub const fn provider_contract(&self) -> &ProviderSessionContract {
        &self.contract
    }

    pub fn open_request(&self) -> ProviderRequest {
        ProviderRequest {
            request: Some(provider_request::Request::OpenSession(
                OpenProviderSession {
                    schema_version: OPEN_SCHEMA.into(),
                    session_id: self.session_id().to_string(),
                    provider_id: self.contract.provider_id.into(),
                    direction_id: self.contract.direction_id.into(),
                    source_language: self.contract.source_language.into(),
                    target_language: self.contract.target_language.into(),
                    mode: self.contract.mode.into(),
                    requested_input_format: Some(self.contract.input_format),
                    requested_output_format: Some(self.contract.output_format),
                    voice_profile: Some(VoiceProfile {
                        language: self.contract.target_language.into(),
                        gender: provider_voice_gender(self.config.voice_gender).into(),
                        engine: provider_voice_engine(self.config.voice_engine).into(),
                        model_path: None,
                        provider_voice_id: None,
                    }),
                    debug_text_enabled: self.config.debug_text_enabled,
                },
            )),
        }
    }

    pub fn handle_capture(
        &mut self,
        event: CaptureEvent,
    ) -> Result<Option<ProviderRequest>, DirectionSessionError> {
        match event {
            CaptureEvent::SpeechStarted {
                stream_id,
                utterance_id,
                capture_monotonic_ns,
            } => {
                self.require_stream(stream_id)?;
                if self.collecting_utterance.is_some()
                    || !self.active_utterances.insert(utterance_id)
                {
                    return Err(DirectionSessionError::InvalidCaptureState);
                }
                self.coordinator
                    .start_utterance(stream_id, utterance_id, capture_monotonic_ns)?;
                self.collecting_utterance = Some(utterance_id);
                Ok(None)
            }
            CaptureEvent::Frame {
                stream_id,
                utterance_id,
                frame,
                end_of_utterance,
            } => {
                self.require_stream(stream_id)?;
                if self.collecting_utterance != Some(utterance_id) {
                    return Err(DirectionSessionError::InvalidCaptureState);
                }
                let capture_monotonic_ns = frame.capture_monotonic_ns();
                let request = ProviderRequest {
                    request: Some(provider_request::Request::InputFrame(
                        translator_ipc::provider::ProviderInputFrame {
                            schema_version: INPUT_SCHEMA.into(),
                            session_id: self.session_id().to_string(),
                            direction_id: self.contract.direction_id.into(),
                            stream_id: stream_id.to_string(),
                            utterance_id: utterance_id.to_string(),
                            sequence: self.next_input_sequence,
                            capture_monotonic_ns,
                            sample_rate_hz: frame.format().sample_rate_hz(),
                            channels: u32::from(frame.format().channels()),
                            sample_format: SampleFormat::S16le.into(),
                            frame_duration_ms: u32::from(frame.format().frame_duration_ms()),
                            source_language: self.contract.source_language.into(),
                            target_language: self.contract.target_language.into(),
                            mode: self.contract.mode.into(),
                            pcm: frame.into_pcm(),
                            end_of_utterance,
                        },
                    )),
                };
                if end_of_utterance {
                    self.coordinator.record_end_of_utterance(
                        stream_id,
                        utterance_id,
                        capture_monotonic_ns,
                    )?;
                    self.collecting_utterance = None;
                }
                self.next_input_sequence = self.next_input_sequence.saturating_add(1);
                Ok(Some(request))
            }
        }
    }

    pub fn handle_provider_event(
        &mut self,
        event: &translator_ipc::provider::ProviderEvent,
        now_ns: u64,
    ) -> Result<Vec<DirectionEffect>, DirectionSessionError> {
        if let Err(error) = self.coordinator.validate_event(event, now_ns) {
            if error
                == ProviderStreamCoordinatorError::Validation(
                    ProviderValidationError::CancelledAudio,
                )
            {
                return Ok(Vec::new());
            }
            if error
                == ProviderStreamCoordinatorError::Validation(ProviderValidationError::ExpiredAudio)
                && let Some(provider_event::Event::AudioDelta(value)) = event.event.as_ref()
            {
                let stream_id =
                    Uuid::parse_str(&value.stream_id).map_err(|_| Self::invalid_audio())?;
                let utterance_id =
                    Uuid::parse_str(&value.utterance_id).map_err(|_| Self::invalid_audio())?;
                self.coordinator
                    .cancel_expired_utterance(stream_id, utterance_id, now_ns)?;
                return Ok(vec![DirectionEffect::ExpiredAudio {
                    utterance_id,
                    request: self.cancel_request(utterance_id),
                }]);
            }
            return Err(error.into());
        }
        let Some(kind) = event.event.as_ref() else {
            return Ok(Vec::new());
        };
        match kind {
            provider_event::Event::AudioDelta(value) => {
                let stream_id =
                    Uuid::parse_str(&value.stream_id).map_err(|_| Self::invalid_audio())?;
                let utterance_id =
                    Uuid::parse_str(&value.utterance_id).map_err(|_| Self::invalid_audio())?;
                let pcm = self
                    .coordinator
                    .consume_receive_audio(stream_id, utterance_id)?;
                let frame = PcmFrame::try_new(
                    value.sequence,
                    value.provider_monotonic_ns,
                    StreamPcmFormat::provider_default(),
                    pcm,
                )?;
                Ok(vec![DirectionEffect::Playback {
                    stream_id,
                    utterance_id,
                    frame,
                }])
            }
            provider_event::Event::TranscriptDelta(value) if value.is_final => {
                Ok(vec![DirectionEffect::TranscriptFinal {
                    utterance_id: parse_utterance(&value.utterance_id)?,
                }])
            }
            provider_event::Event::TranslationDelta(value) if value.is_final => {
                Ok(vec![DirectionEffect::TranslationFinal {
                    utterance_id: parse_utterance(&value.utterance_id)?,
                }])
            }
            provider_event::Event::Latency(value) => Ok(vec![DirectionEffect::Latency {
                utterance_id: value
                    .utterance_id
                    .as_deref()
                    .map(parse_utterance)
                    .transpose()?,
                tts_first_audio_ms: value.tts_first_audio_ms,
                provider_total_ms: value.provider_total_ms,
            }]),
            provider_event::Event::Error(value) => Ok(vec![DirectionEffect::ProviderError {
                utterance_id: value
                    .utterance_id
                    .as_deref()
                    .map(parse_utterance)
                    .transpose()?,
                code: safe_provider_error_code(value.code)?,
                retryable: value.retryable,
            }]),
            provider_event::Event::UtteranceFinal(value) => {
                let utterance_id = parse_utterance(&value.utterance_id)?;
                let outcome = terminal_outcome(value.outcome)?;
                self.active_utterances.remove(&utterance_id);
                Ok(vec![
                    DirectionEffect::UtteranceTerminalOutcome {
                        utterance_id,
                        outcome,
                    },
                    DirectionEffect::UtteranceTerminal { utterance_id },
                ])
            }
            provider_event::Event::SessionClosed(_) => {
                self.active_utterances.clear();
                self.collecting_utterance = None;
                Ok(vec![DirectionEffect::SessionClosed])
            }
            _ => Ok(Vec::new()),
        }
    }

    pub fn poll(
        &mut self,
        now_ns: u64,
    ) -> Result<Vec<DirectionWatchdogEffect>, DirectionSessionError> {
        let actions = self.coordinator.poll(now_ns)?;
        for action in &actions {
            if let WatchdogAction::CancelUtterance { utterance_id, .. } = action
                && self.collecting_utterance == Some(*utterance_id)
            {
                self.collecting_utterance = None;
            }
        }
        let effects = actions
            .into_iter()
            .map(|action| match action {
                WatchdogAction::CancelUtterance { utterance_id, .. } => {
                    DirectionWatchdogEffect::PurgeAndSend(self.cancel_request(utterance_id))
                }
                WatchdogAction::CloseProviderSession { .. } => {
                    DirectionWatchdogEffect::PurgeAndSend(
                        self.close_request(CloseRequestReason::ProviderSwitch),
                    )
                }
                WatchdogAction::RestartSidecar { .. } => DirectionWatchdogEffect::RestartSidecar,
            })
            .collect();
        Ok(effects)
    }

    fn cancel_request(&self, utterance_id: Uuid) -> ProviderRequest {
        ProviderRequest {
            request: Some(provider_request::Request::CancelUtterance(
                CancelUtterance {
                    schema_version: CANCEL_SCHEMA.into(),
                    session_id: self.session_id().to_string(),
                    direction_id: self.contract.direction_id.into(),
                    utterance_id: utterance_id.to_string(),
                    reason: CancelReason::LatencyPolicy.into(),
                },
            )),
        }
    }

    pub fn close_request(&self, reason: CloseRequestReason) -> ProviderRequest {
        ProviderRequest {
            request: Some(provider_request::Request::CloseSession(
                CloseProviderSession {
                    schema_version: CLOSE_SCHEMA.into(),
                    session_id: self.session_id().to_string(),
                    reason: reason.into(),
                },
            )),
        }
    }

    fn require_stream(&self, stream_id: Uuid) -> Result<(), DirectionSessionError> {
        if stream_id == self.stream_id() {
            Ok(())
        } else {
            Err(DirectionSessionError::StreamMismatch)
        }
    }

    const fn invalid_audio() -> DirectionSessionError {
        DirectionSessionError::InvalidProviderAudio
    }
}

fn parse_utterance(value: &str) -> Result<Uuid, DirectionSessionError> {
    Uuid::parse_str(value).map_err(|_| DirectionSessionError::InvalidProviderAudio)
}

fn safe_provider_error_code(value: i32) -> Result<SafeProviderErrorCode, DirectionSessionError> {
    match SafeErrorCode::try_from(value)
        .map_err(|_| DirectionSessionError::InvalidProviderMetadata)?
    {
        SafeErrorCode::ProviderUnavailable => Ok(SafeProviderErrorCode::ProviderUnavailable),
        SafeErrorCode::ModelNotLoaded => Ok(SafeProviderErrorCode::ModelNotLoaded),
        SafeErrorCode::UnsupportedLanguagePair => {
            Ok(SafeProviderErrorCode::UnsupportedLanguagePair)
        }
        SafeErrorCode::QueueOverflow => Ok(SafeProviderErrorCode::QueueOverflow),
        SafeErrorCode::Cancelled => Ok(SafeProviderErrorCode::Cancelled),
        SafeErrorCode::CloudNotEnabled => Ok(SafeProviderErrorCode::CloudNotEnabled),
        SafeErrorCode::ProviderAuthFailed => Ok(SafeProviderErrorCode::ProviderAuthFailed),
        SafeErrorCode::NoSpeech => Ok(SafeProviderErrorCode::NoSpeech),
        SafeErrorCode::Unspecified => Err(DirectionSessionError::InvalidProviderMetadata),
    }
}

fn terminal_outcome(value: i32) -> Result<TerminalOutcome, DirectionSessionError> {
    match ProviderUtteranceOutcome::try_from(value)
        .map_err(|_| DirectionSessionError::InvalidProviderMetadata)?
    {
        ProviderUtteranceOutcome::Completed => Ok(TerminalOutcome::Completed),
        ProviderUtteranceOutcome::Cancelled => Ok(TerminalOutcome::Cancelled),
        ProviderUtteranceOutcome::Dropped => Ok(TerminalOutcome::Dropped),
        ProviderUtteranceOutcome::Unspecified => {
            Err(DirectionSessionError::InvalidProviderMetadata)
        }
    }
}

const fn provider_pcm_format() -> PcmFormat {
    PcmFormat {
        sample_rate_hz: 16_000,
        channels: 1,
        sample_format: SampleFormat::S16le as i32,
        frame_duration_ms: 20,
    }
}

const fn provider_direction(value: AudioDirection) -> ProviderDirection {
    match value {
        AudioDirection::Microphone => ProviderDirection::Microphone,
        AudioDirection::Speaker => ProviderDirection::Speaker,
    }
}

const fn provider_id(value: ProviderId) -> ProviderProviderId {
    match value {
        ProviderId::Local => ProviderProviderId::Local,
        ProviderId::Openai => ProviderProviderId::Openai,
    }
}

const fn provider_language(value: Language) -> ProviderLanguage {
    match value {
        Language::Ru => ProviderLanguage::Ru,
        Language::En => ProviderLanguage::En,
    }
}

const fn provider_mode(value: TranslationMode) -> ProviderTranslationMode {
    match value {
        TranslationMode::QualityFirst => ProviderTranslationMode::QualityFirst,
        TranslationMode::Balanced => ProviderTranslationMode::Balanced,
        TranslationMode::StreamingFirst => ProviderTranslationMode::StreamingFirst,
    }
}

const fn provider_voice_gender(value: VoiceGender) -> ProviderVoiceGender {
    match value {
        VoiceGender::Male => ProviderVoiceGender::Male,
        VoiceGender::Female => ProviderVoiceGender::Female,
    }
}

const fn provider_voice_engine(value: VoiceEngine) -> ProviderVoiceEngine {
    match value {
        VoiceEngine::Piper => ProviderVoiceEngine::Piper,
        VoiceEngine::Silero => ProviderVoiceEngine::Silero,
        VoiceEngine::Openai => ProviderVoiceEngine::Openai,
    }
}
