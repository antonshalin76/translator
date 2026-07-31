use std::collections::{HashMap, HashSet};

use thiserror::Error;
use uuid::Uuid;

use crate::provider::{
    AudioDirection, PcmFormat, ProviderAudioDelta, ProviderEvent, ProviderId, SafeErrorCode,
    TranslationMode, UtteranceOutcome, provider_event,
};

const NS_PER_MS: u64 = 1_000_000;
pub const MAX_ACTIVE_UTTERANCES: usize = 64;
pub const MAX_TERMINAL_UTTERANCES: usize = 4096;

#[derive(Debug, Clone)]
pub struct ProviderSessionContract {
    pub session_id: Uuid,
    pub stream_id: Uuid,
    pub provider_id: ProviderId,
    pub direction_id: AudioDirection,
    pub source_language: crate::provider::Language,
    pub target_language: crate::provider::Language,
    pub mode: TranslationMode,
    pub input_format: PcmFormat,
    pub output_format: PcmFormat,
    pub debug_text_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ProviderValidationError {
    #[error("provider event is missing")]
    MissingEvent,
    #[error("provider session must open first")]
    OpenRequired,
    #[error("provider session already opened")]
    DuplicateOpen,
    #[error("provider schema mismatch")]
    SchemaMismatch,
    #[error("provider session mismatch")]
    SessionMismatch,
    #[error("provider direction mismatch")]
    DirectionMismatch,
    #[error("provider identity mismatch")]
    ProviderMismatch,
    #[error("provider stream mismatch")]
    StreamMismatch,
    #[error("provider identifier is invalid")]
    InvalidIdentifier,
    #[error("provider event sequence is duplicate")]
    DuplicateSequence,
    #[error("provider event sequence is stale")]
    StaleSequence,
    #[error("provider initial event sequence is invalid")]
    SequenceGap,
    #[error("provider audio sequence is duplicate")]
    DuplicateAudioSequence,
    #[error("provider audio sequence is stale")]
    StaleAudioSequence,
    #[error("provider audio sequence has a gap")]
    AudioSequenceGap,
    #[error("provider session contract is missing")]
    MissingSessionContract,
    #[error("provider negotiated format mismatch")]
    NegotiatedFormatMismatch,
    #[error("provider output format mismatch")]
    OutputFormatMismatch,
    #[error("provider PCM length mismatch")]
    PcmLengthMismatch,
    #[error("provider utterance is unknown")]
    UnknownUtterance,
    #[error("provider utterance is terminal")]
    UtteranceTerminal,
    #[error("provider active utterance capacity is exhausted")]
    UtteranceCapacityExceeded,
    #[error("provider terminal utterance capacity is exhausted")]
    TerminalCapacityExceeded,
    #[error("provider session is terminal")]
    SessionTerminal,
    #[error("provider debug text is disabled")]
    DebugTextDisabled,
    #[error("provider error message is unsafe")]
    UnsafeErrorMessage,
    #[error("provider audio is expired")]
    ExpiredAudio,
    #[error("provider audio arrived after cancellation")]
    CancelledAudio,
    #[error("provider final audio sequence mismatch")]
    FinalAudioSequenceMismatch,
    #[error("provider overflow terminal is missing or mismatched")]
    ExpectedOverflowTerminal,
    #[error("provider no-speech terminal is missing or mismatched")]
    ExpectedNoSpeechTerminal,
    #[error("provider no-speech ASR latency is missing or mismatched")]
    ExpectedNoSpeechLatency,
    #[error("provider cancelled terminal is missing or mismatched")]
    ExpectedCancelledTerminal,
    #[error("provider utterance outcome is invalid")]
    InvalidOutcome,
}

pub struct ProviderEventValidator {
    contract: ProviderSessionContract,
    opened: bool,
    closed: bool,
    last_event_sequence: Option<u64>,
    input_capture_ns: HashMap<Uuid, u64>,
    input_complete_ns: HashMap<Uuid, u64>,
    last_wire_audio_sequence: HashMap<Uuid, u64>,
    last_playable_audio_sequence: HashMap<Uuid, u64>,
    terminal_utterances: HashSet<Uuid>,
    pending_overflow_terminal: Option<Uuid>,
    pending_no_speech_terminal: Option<Uuid>,
    pending_no_speech_latency: Option<Uuid>,
    cancel_pending: HashSet<Uuid>,
    debug_text_enabled: bool,
}

impl ProviderEventValidator {
    pub fn new(contract: ProviderSessionContract) -> Self {
        let debug_text_enabled = contract.debug_text_enabled;
        Self {
            contract,
            opened: false,
            closed: false,
            last_event_sequence: None,
            input_capture_ns: HashMap::new(),
            input_complete_ns: HashMap::new(),
            last_wire_audio_sequence: HashMap::new(),
            last_playable_audio_sequence: HashMap::new(),
            terminal_utterances: HashSet::new(),
            pending_overflow_terminal: None,
            pending_no_speech_terminal: None,
            pending_no_speech_latency: None,
            cancel_pending: HashSet::new(),
            debug_text_enabled,
        }
    }

    pub fn record_input(
        &mut self,
        utterance_id: Uuid,
        capture_monotonic_ns: u64,
    ) -> Result<(), ProviderValidationError> {
        if !self.opened {
            return Err(ProviderValidationError::OpenRequired);
        }
        if self.closed {
            return Err(ProviderValidationError::SessionTerminal);
        }
        if self.terminal_utterances.contains(&utterance_id) {
            return Err(ProviderValidationError::UtteranceTerminal);
        }
        if !self.input_capture_ns.contains_key(&utterance_id)
            && self.input_capture_ns.len() >= MAX_ACTIVE_UTTERANCES
        {
            return Err(ProviderValidationError::UtteranceCapacityExceeded);
        }
        self.input_capture_ns
            .entry(utterance_id)
            .or_insert(capture_monotonic_ns);
        self.input_complete_ns
            .entry(utterance_id)
            .or_insert(capture_monotonic_ns);
        Ok(())
    }

    pub fn record_end_of_utterance(
        &mut self,
        utterance_id: Uuid,
        capture_monotonic_ns: u64,
    ) -> Result<(), ProviderValidationError> {
        if !self.opened {
            return Err(ProviderValidationError::OpenRequired);
        }
        if self.closed {
            return Err(ProviderValidationError::SessionTerminal);
        }
        let input_complete_ns = self
            .input_complete_ns
            .get_mut(&utterance_id)
            .ok_or(ProviderValidationError::UnknownUtterance)?;
        *input_complete_ns = capture_monotonic_ns;
        Ok(())
    }

    pub fn set_debug_text_enabled(&mut self, enabled: bool) {
        self.debug_text_enabled = enabled;
    }

    pub fn record_cancel_requested(
        &mut self,
        utterance_id: Uuid,
    ) -> Result<(), ProviderValidationError> {
        self.can_record_cancel_requested(utterance_id)?;
        self.cancel_pending.insert(utterance_id);
        Ok(())
    }

    pub fn can_record_cancel_requested(
        &self,
        utterance_id: Uuid,
    ) -> Result<(), ProviderValidationError> {
        if !self.opened {
            return Err(ProviderValidationError::OpenRequired);
        }
        if self.closed {
            return Err(ProviderValidationError::SessionTerminal);
        }
        if self.terminal_utterances.contains(&utterance_id) {
            return Err(ProviderValidationError::UtteranceTerminal);
        }
        if !self.input_capture_ns.contains_key(&utterance_id) {
            return Err(ProviderValidationError::UnknownUtterance);
        }
        Ok(())
    }

    pub fn cancel_pending(&self, utterance_id: Uuid) -> bool {
        self.cancel_pending.contains(&utterance_id)
    }

    pub fn validate(
        &mut self,
        event: &ProviderEvent,
        now_ns: u64,
    ) -> Result<(), ProviderValidationError> {
        let event = event
            .event
            .as_ref()
            .ok_or(ProviderValidationError::MissingEvent)?;
        if self.closed {
            return Err(ProviderValidationError::SessionTerminal);
        }
        let is_open = matches!(event, provider_event::Event::SessionOpened(_));
        if !self.opened && !is_open {
            return Err(ProviderValidationError::OpenRequired);
        }
        if self.opened && is_open {
            return Err(ProviderValidationError::DuplicateOpen);
        }

        let common = common_fields(event);
        if common.schema_version != expected_schema(event) {
            return Err(ProviderValidationError::SchemaMismatch);
        }
        let session_id = parse_uuid(common.session_id)?;
        if session_id != self.contract.session_id {
            return Err(ProviderValidationError::SessionMismatch);
        }
        if common.direction_id != self.contract.direction_id as i32 {
            return Err(ProviderValidationError::DirectionMismatch);
        }
        self.validate_stream(event)?;
        self.validate_sequence(common.event_sequence)?;

        let result = self.validate_kind(event, now_ns);
        if result.is_ok()
            || matches!(
                result,
                Err(ProviderValidationError::ExpiredAudio | ProviderValidationError::CancelledAudio)
            )
        {
            self.last_event_sequence = Some(common.event_sequence);
        }
        result
    }

    fn validate_sequence(&self, sequence: u64) -> Result<(), ProviderValidationError> {
        match self.last_event_sequence {
            None if sequence == 1 => Ok(()),
            None => Err(ProviderValidationError::SequenceGap),
            Some(previous) if sequence == previous => {
                Err(ProviderValidationError::DuplicateSequence)
            }
            Some(previous) if sequence < previous => Err(ProviderValidationError::StaleSequence),
            Some(_) => Ok(()),
        }
    }

    fn validate_kind(
        &mut self,
        event: &provider_event::Event,
        now_ns: u64,
    ) -> Result<(), ProviderValidationError> {
        if let Some(utterance_id) = event_utterance_id(event)?
            && self.cancel_pending.contains(&utterance_id)
        {
            if let provider_event::Event::AudioDelta(value) = event {
                self.validate_wire_audio(value, utterance_id)?;
                return Err(ProviderValidationError::CancelledAudio);
            }
            return self.validate_cancelled_terminal(event, utterance_id);
        }
        if let Some(expected) = self.pending_overflow_terminal {
            return self.validate_overflow_terminal(event, expected);
        }
        if let Some(expected) = self.pending_no_speech_terminal {
            return self.validate_no_speech_terminal(event, expected);
        }
        let is_no_speech_error = matches!(
            event,
            provider_event::Event::Error(value)
                if value.code == SafeErrorCode::NoSpeech as i32
        );
        if !matches!(event, provider_event::Event::Latency(_)) && !is_no_speech_error {
            self.pending_no_speech_latency = None;
        }
        match event {
            provider_event::Event::SessionOpened(value) => {
                let input = value
                    .negotiated_input_format
                    .as_ref()
                    .ok_or(ProviderValidationError::MissingSessionContract)?;
                let output = value
                    .negotiated_output_format
                    .as_ref()
                    .ok_or(ProviderValidationError::MissingSessionContract)?;
                let capabilities = value
                    .capabilities
                    .as_ref()
                    .ok_or(ProviderValidationError::MissingSessionContract)?;
                if input != &self.contract.input_format || output != &self.contract.output_format {
                    return Err(ProviderValidationError::NegotiatedFormatMismatch);
                }
                if !capabilities.audio_output || !capabilities.cancellation {
                    return Err(ProviderValidationError::NegotiatedFormatMismatch);
                }
                if capabilities.cloud_egress
                    != matches!(self.contract.provider_id, ProviderId::Openai)
                {
                    return Err(ProviderValidationError::NegotiatedFormatMismatch);
                }
                self.opened = true;
                Ok(())
            }
            provider_event::Event::AudioDelta(value) => {
                let utterance_id = self.active_utterance(&value.utterance_id)?;
                self.validate_wire_audio(value, utterance_id)?;
                let is_first_playable = !self
                    .last_playable_audio_sequence
                    .contains_key(&utterance_id);
                let capture_ns = self.input_complete_ns[&utterance_id];
                if is_first_playable
                    && now_ns.saturating_sub(capture_ns) > self.max_age_ms() * NS_PER_MS
                {
                    return Err(ProviderValidationError::ExpiredAudio);
                }
                self.last_playable_audio_sequence
                    .insert(utterance_id, value.sequence);
                Ok(())
            }
            provider_event::Event::TranscriptDelta(value) => {
                self.validate_text_event(&value.utterance_id)
            }
            provider_event::Event::TranslationDelta(value) => {
                self.validate_text_event(&value.utterance_id)
            }
            provider_event::Event::Latency(value) => {
                let utterance_id = value
                    .utterance_id
                    .as_deref()
                    .map(|id| self.active_utterance(id))
                    .transpose()?;
                self.pending_no_speech_latency = utterance_id.filter(|_| {
                    value.asr_first_text_ms.is_some()
                        && value.asr_final_text_ms.is_some()
                        && value.mt_first_text_ms.is_none()
                        && value.tts_first_audio_ms.is_none()
                });
                Ok(())
            }
            provider_event::Event::Health(value) => {
                if value.provider_id != self.contract.provider_id as i32 {
                    return Err(ProviderValidationError::ProviderMismatch);
                }
                if value.queues.is_none() {
                    return Err(ProviderValidationError::MissingSessionContract);
                }
                if let Some(error) = value.safe_error.as_ref()
                    && safe_health_message(&error.code) != Some(error.message.as_str())
                {
                    return Err(ProviderValidationError::UnsafeErrorMessage);
                }
                Ok(())
            }
            provider_event::Event::Error(value) => {
                let code = SafeErrorCode::try_from(value.code)
                    .map_err(|_| ProviderValidationError::UnsafeErrorMessage)?;
                if code == SafeErrorCode::Unspecified
                    || value.safe_message != safe_error_message(code)
                {
                    return Err(ProviderValidationError::UnsafeErrorMessage);
                }
                let utterance_id = value
                    .utterance_id
                    .as_deref()
                    .map(|id| self.active_utterance(id))
                    .transpose()?;
                if code == SafeErrorCode::QueueOverflow {
                    let utterance_id =
                        utterance_id.ok_or(ProviderValidationError::ExpectedOverflowTerminal)?;
                    self.pending_overflow_terminal = Some(utterance_id);
                }
                if code == SafeErrorCode::NoSpeech {
                    let utterance_id =
                        utterance_id.ok_or(ProviderValidationError::ExpectedNoSpeechTerminal)?;
                    if self.pending_no_speech_latency != Some(utterance_id) {
                        return Err(ProviderValidationError::ExpectedNoSpeechLatency);
                    }
                    self.pending_no_speech_latency = None;
                    self.pending_no_speech_terminal = Some(utterance_id);
                }
                Ok(())
            }
            provider_event::Event::UtteranceFinal(value) => {
                let utterance_id = self.active_utterance(&value.utterance_id)?;
                let outcome = UtteranceOutcome::try_from(value.outcome)
                    .map_err(|_| ProviderValidationError::InvalidOutcome)?;
                if outcome == UtteranceOutcome::Unspecified {
                    return Err(ProviderValidationError::InvalidOutcome);
                }
                if outcome == UtteranceOutcome::Cancelled {
                    return Err(ProviderValidationError::ExpectedCancelledTerminal);
                }
                if value.final_audio_sequence
                    != self
                        .last_playable_audio_sequence
                        .get(&utterance_id)
                        .copied()
                {
                    return Err(ProviderValidationError::FinalAudioSequenceMismatch);
                }
                self.terminalize(utterance_id)
            }
            provider_event::Event::SessionClosed(_) => {
                self.closed = true;
                Ok(())
            }
        }
    }

    fn validate_text_event(&self, utterance_id: &str) -> Result<(), ProviderValidationError> {
        self.active_utterance(utterance_id)?;
        if !self.debug_text_enabled {
            return Err(ProviderValidationError::DebugTextDisabled);
        }
        Ok(())
    }

    fn validate_stream(
        &self,
        event: &provider_event::Event,
    ) -> Result<(), ProviderValidationError> {
        let stream_id = match event {
            provider_event::Event::AudioDelta(value) => Some(value.stream_id.as_str()),
            provider_event::Event::TranscriptDelta(value) => Some(value.stream_id.as_str()),
            provider_event::Event::TranslationDelta(value) => Some(value.stream_id.as_str()),
            provider_event::Event::UtteranceFinal(value) => Some(value.stream_id.as_str()),
            provider_event::Event::Latency(value) => Some(value.stream_id.as_str()),
            provider_event::Event::Error(value) => value.stream_id.as_deref(),
            _ => None,
        };
        if let Some(value) = stream_id
            && parse_uuid(value)? != self.contract.stream_id
        {
            return Err(ProviderValidationError::StreamMismatch);
        }
        if matches!(
            event,
            provider_event::Event::Error(value) if value.utterance_id.is_some()
        ) && stream_id.is_none()
        {
            return Err(ProviderValidationError::StreamMismatch);
        }
        Ok(())
    }

    fn validate_audio_sequence(
        &self,
        utterance_id: Uuid,
        sequence: u64,
    ) -> Result<(), ProviderValidationError> {
        match self.last_wire_audio_sequence.get(&utterance_id).copied() {
            None if sequence == 0 => Ok(()),
            None => Err(ProviderValidationError::AudioSequenceGap),
            Some(previous) if sequence == previous => {
                Err(ProviderValidationError::DuplicateAudioSequence)
            }
            Some(previous) if sequence < previous => {
                Err(ProviderValidationError::StaleAudioSequence)
            }
            Some(previous) if sequence > previous + 1 => {
                Err(ProviderValidationError::AudioSequenceGap)
            }
            Some(_) => Ok(()),
        }
    }

    fn validate_wire_audio(
        &mut self,
        value: &ProviderAudioDelta,
        utterance_id: Uuid,
    ) -> Result<(), ProviderValidationError> {
        self.validate_audio_sequence(utterance_id, value.sequence)?;
        if value.sample_rate_hz != self.contract.output_format.sample_rate_hz
            || value.channels != self.contract.output_format.channels
            || value.sample_format != self.contract.output_format.sample_format
            || value.frame_duration_ms != self.contract.output_format.frame_duration_ms
        {
            return Err(ProviderValidationError::OutputFormatMismatch);
        }
        let expected_bytes = value.sample_rate_hz as usize
            * value.channels as usize
            * 2
            * value.frame_duration_ms as usize
            / 1000;
        if value.pcm.len() != expected_bytes {
            return Err(ProviderValidationError::PcmLengthMismatch);
        }
        self.last_wire_audio_sequence
            .insert(utterance_id, value.sequence);
        Ok(())
    }

    fn active_utterance(&self, value: &str) -> Result<Uuid, ProviderValidationError> {
        let utterance_id = parse_uuid(value)?;
        if self.terminal_utterances.contains(&utterance_id) {
            return Err(ProviderValidationError::UtteranceTerminal);
        }
        if !self.input_capture_ns.contains_key(&utterance_id) {
            return Err(ProviderValidationError::UnknownUtterance);
        }
        Ok(utterance_id)
    }

    fn validate_overflow_terminal(
        &mut self,
        event: &provider_event::Event,
        expected: Uuid,
    ) -> Result<(), ProviderValidationError> {
        let provider_event::Event::UtteranceFinal(value) = event else {
            return Err(ProviderValidationError::ExpectedOverflowTerminal);
        };
        let utterance_id = parse_uuid(&value.utterance_id)?;
        if utterance_id != expected
            || value.outcome != UtteranceOutcome::Dropped as i32
            || value.final_audio_sequence
                != self
                    .last_playable_audio_sequence
                    .get(&utterance_id)
                    .copied()
        {
            return Err(ProviderValidationError::ExpectedOverflowTerminal);
        }
        self.pending_overflow_terminal = None;
        self.terminalize(utterance_id)
    }

    fn validate_no_speech_terminal(
        &mut self,
        event: &provider_event::Event,
        expected: Uuid,
    ) -> Result<(), ProviderValidationError> {
        let provider_event::Event::UtteranceFinal(value) = event else {
            return Err(ProviderValidationError::ExpectedNoSpeechTerminal);
        };
        let utterance_id = parse_uuid(&value.utterance_id)?;
        if utterance_id != expected
            || value.outcome != UtteranceOutcome::Dropped as i32
            || value.final_audio_sequence.is_some()
        {
            return Err(ProviderValidationError::ExpectedNoSpeechTerminal);
        }
        self.pending_no_speech_terminal = None;
        self.terminalize(utterance_id)
    }

    fn validate_cancelled_terminal(
        &mut self,
        event: &provider_event::Event,
        utterance_id: Uuid,
    ) -> Result<(), ProviderValidationError> {
        let provider_event::Event::UtteranceFinal(value) = event else {
            return Err(ProviderValidationError::ExpectedCancelledTerminal);
        };
        if value.outcome != UtteranceOutcome::Cancelled as i32
            || value.final_audio_sequence
                != self.last_wire_audio_sequence.get(&utterance_id).copied()
        {
            return Err(ProviderValidationError::ExpectedCancelledTerminal);
        }
        self.cancel_pending.remove(&utterance_id);
        self.terminalize(utterance_id)
    }

    fn terminalize(&mut self, utterance_id: Uuid) -> Result<(), ProviderValidationError> {
        if self.terminal_utterances.len() >= MAX_TERMINAL_UTTERANCES {
            self.closed = true;
            return Err(ProviderValidationError::TerminalCapacityExceeded);
        }
        self.terminal_utterances.insert(utterance_id);
        self.input_capture_ns.remove(&utterance_id);
        self.input_complete_ns.remove(&utterance_id);
        self.last_wire_audio_sequence.remove(&utterance_id);
        self.last_playable_audio_sequence.remove(&utterance_id);
        self.cancel_pending.remove(&utterance_id);
        Ok(())
    }

    fn max_age_ms(&self) -> u64 {
        match self.contract.mode {
            TranslationMode::QualityFirst => 3000,
            TranslationMode::Balanced => 2000,
            TranslationMode::StreamingFirst => 1000,
            TranslationMode::Unspecified => 0,
        }
    }
}

fn event_utterance_id(
    event: &provider_event::Event,
) -> Result<Option<Uuid>, ProviderValidationError> {
    let value = match event {
        provider_event::Event::AudioDelta(value) => Some(value.utterance_id.as_str()),
        provider_event::Event::TranscriptDelta(value) => Some(value.utterance_id.as_str()),
        provider_event::Event::TranslationDelta(value) => Some(value.utterance_id.as_str()),
        provider_event::Event::UtteranceFinal(value) => Some(value.utterance_id.as_str()),
        provider_event::Event::Latency(value) => value.utterance_id.as_deref(),
        provider_event::Event::Error(value) => value.utterance_id.as_deref(),
        _ => None,
    };
    value.map(parse_uuid).transpose()
}

struct CommonFields<'a> {
    schema_version: &'a str,
    session_id: &'a str,
    direction_id: i32,
    event_sequence: u64,
}

fn common_fields(event: &provider_event::Event) -> CommonFields<'_> {
    macro_rules! common {
        ($value:expr) => {
            CommonFields {
                schema_version: &$value.schema_version,
                session_id: &$value.session_id,
                direction_id: $value.direction_id,
                event_sequence: $value.event_sequence,
            }
        };
    }
    match event {
        provider_event::Event::SessionOpened(value) => common!(value),
        provider_event::Event::AudioDelta(value) => common!(value),
        provider_event::Event::TranscriptDelta(value) => common!(value),
        provider_event::Event::TranslationDelta(value) => common!(value),
        provider_event::Event::UtteranceFinal(value) => common!(value),
        provider_event::Event::SessionClosed(value) => common!(value),
        provider_event::Event::Health(value) => common!(value),
        provider_event::Event::Latency(value) => common!(value),
        provider_event::Event::Error(value) => common!(value),
    }
}

fn expected_schema(event: &provider_event::Event) -> &'static str {
    match event {
        provider_event::Event::SessionOpened(_) => "translator.provider.session_opened.v1",
        provider_event::Event::AudioDelta(_) => "translator.provider.audio_delta.v1",
        provider_event::Event::TranscriptDelta(_) => "translator.provider.transcript_delta.v1",
        provider_event::Event::TranslationDelta(_) => "translator.provider.translation_delta.v1",
        provider_event::Event::UtteranceFinal(_) => "translator.provider.utterance_final.v1",
        provider_event::Event::SessionClosed(_) => "translator.provider.session_closed.v1",
        provider_event::Event::Health(_) => "translator.provider.health.v1",
        provider_event::Event::Latency(_) => "translator.provider.latency.v1",
        provider_event::Event::Error(_) => "translator.provider.error.v1",
    }
}

fn parse_uuid(value: &str) -> Result<Uuid, ProviderValidationError> {
    Uuid::parse_str(value).map_err(|_| ProviderValidationError::InvalidIdentifier)
}

fn safe_error_message(code: SafeErrorCode) -> &'static str {
    match code {
        SafeErrorCode::ProviderUnavailable => "Provider is unavailable",
        SafeErrorCode::ModelNotLoaded => "Required model is not loaded",
        SafeErrorCode::UnsupportedLanguagePair => "Language pair is not supported",
        SafeErrorCode::QueueOverflow => "Provider queue limit was reached",
        SafeErrorCode::Cancelled => "Provider operation was cancelled",
        SafeErrorCode::NoSpeech => "No speech was detected",
        SafeErrorCode::CloudNotEnabled => "Cloud provider is not enabled",
        SafeErrorCode::ProviderAuthFailed => "Provider authentication failed",
        SafeErrorCode::Unspecified => "",
    }
}

fn safe_health_message(code: &str) -> Option<&'static str> {
    match code {
        "provider_unavailable" => Some("Provider is unavailable"),
        "model_not_loaded" => Some("Required model is not loaded"),
        "unsupported_language_pair" => Some("Language pair is not supported"),
        "queue_overflow" => Some("Provider queue limit was reached"),
        "cancelled" => Some("Provider operation was cancelled"),
        "no_speech" => Some("No speech was detected"),
        "cloud_not_enabled" => Some("Cloud provider is not enabled"),
        "provider_auth_failed" => Some("Provider authentication failed"),
        _ => None,
    }
}
