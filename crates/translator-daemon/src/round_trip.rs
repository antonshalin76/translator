use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use translator_audio::{PcmFrame, StreamPcmFormat};
use uuid::Uuid;

const MAX_SESSION_MS: u64 = 5 * 60 * 1_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExactPcmProof {
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub frame_duration_ms: u16,
    pub frame_count: u64,
    pub captured_frame_count: u64,
    pub reinjected_frame_count: u64,
    pub first_sequence: Option<u64>,
    pub last_sequence: Option<u64>,
    pub captured_rolling_hash: String,
    pub reinjected_rolling_hash: String,
    pub format_matches: bool,
    pub sequence_monotonic: bool,
    pub exact_match: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ExactPcmEvidenceError {
    #[error("PCM evidence format does not match")]
    FormatMismatch,
    #[error("PCM evidence sequence is not monotonic")]
    SequenceMismatch,
}

pub struct ExactPcmEvidence {
    format: StreamPcmFormat,
    captured_count: u64,
    reinjected_count: u64,
    captured_hash: Sha256,
    reinjected_hash: Sha256,
    format_matches: bool,
    sequence_monotonic: bool,
}

impl ExactPcmEvidence {
    pub fn new(format: StreamPcmFormat) -> Self {
        Self {
            format,
            captured_count: 0,
            reinjected_count: 0,
            captured_hash: Sha256::new(),
            reinjected_hash: Sha256::new(),
            format_matches: true,
            sequence_monotonic: true,
        }
    }

    pub fn capture(&mut self, frame: &PcmFrame) -> Result<(), ExactPcmEvidenceError> {
        Self::record(
            self.format,
            frame,
            &mut self.captured_count,
            &mut self.captured_hash,
            &mut self.format_matches,
            &mut self.sequence_monotonic,
        )
    }

    pub fn reinject(&mut self, frame: &PcmFrame) -> Result<(), ExactPcmEvidenceError> {
        Self::record(
            self.format,
            frame,
            &mut self.reinjected_count,
            &mut self.reinjected_hash,
            &mut self.format_matches,
            &mut self.sequence_monotonic,
        )
    }

    pub fn proof(&self) -> ExactPcmProof {
        let captured_rolling_hash = digest_hex(self.captured_hash.clone().finalize());
        let reinjected_rolling_hash = digest_hex(self.reinjected_hash.clone().finalize());
        let counts_match = self.captured_count == self.reinjected_count;
        ExactPcmProof {
            sample_rate_hz: self.format.sample_rate_hz(),
            channels: self.format.channels(),
            frame_duration_ms: self.format.frame_duration_ms(),
            frame_count: self.captured_count,
            captured_frame_count: self.captured_count,
            reinjected_frame_count: self.reinjected_count,
            first_sequence: (self.captured_count > 0).then_some(0),
            last_sequence: self.captured_count.checked_sub(1),
            captured_rolling_hash: captured_rolling_hash.clone(),
            reinjected_rolling_hash: reinjected_rolling_hash.clone(),
            format_matches: self.format_matches,
            sequence_monotonic: self.sequence_monotonic,
            exact_match: self.captured_count > 0
                && counts_match
                && self.format_matches
                && self.sequence_monotonic
                && captured_rolling_hash == reinjected_rolling_hash,
        }
    }

    fn record(
        format: StreamPcmFormat,
        frame: &PcmFrame,
        count: &mut u64,
        hash: &mut Sha256,
        format_matches: &mut bool,
        sequence_monotonic: &mut bool,
    ) -> Result<(), ExactPcmEvidenceError> {
        if frame.format() != format {
            *format_matches = false;
            return Err(ExactPcmEvidenceError::FormatMismatch);
        }
        if frame.sequence() != *count {
            *sequence_monotonic = false;
            return Err(ExactPcmEvidenceError::SequenceMismatch);
        }
        hash.update(frame.sequence().to_le_bytes());
        hash.update(frame.pcm());
        *count = count.saturating_add(1);
        Ok(())
    }
}

fn digest_hex(digest: impl AsRef<[u8]>) -> String {
    use std::fmt::Write;

    digest
        .as_ref()
        .iter()
        .fold(String::with_capacity(64), |mut output, byte| {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
            output
        })
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct RoundTripPreconditions {
    pub headphones: bool,
    pub outgoing_provider_ready: bool,
    pub incoming_provider_ready: bool,
    pub virtual_graph_ready: bool,
    pub incoming_route_idle: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundTripCheckpoint {
    WaitingForSpeech,
    OutgoingVad,
    OutgoingAsrFinal,
    OutgoingTranslationFinal,
    EnglishFirstAudio,
    VirtualPeerReinjecting,
    IncomingAsrFinal,
    IncomingTranslationFinal,
    RussianFirstAudio,
    Completed,
    Failed,
    Stopped,
}

impl RoundTripCheckpoint {
    const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Stopped)
    }

    const fn next(self) -> Option<Self> {
        match self {
            Self::WaitingForSpeech => Some(Self::OutgoingVad),
            Self::OutgoingVad => Some(Self::OutgoingAsrFinal),
            Self::OutgoingAsrFinal => Some(Self::OutgoingTranslationFinal),
            Self::OutgoingTranslationFinal => Some(Self::EnglishFirstAudio),
            Self::EnglishFirstAudio => Some(Self::VirtualPeerReinjecting),
            Self::VirtualPeerReinjecting => Some(Self::IncomingAsrFinal),
            Self::IncomingAsrFinal => Some(Self::IncomingTranslationFinal),
            Self::IncomingTranslationFinal => Some(Self::RussianFirstAudio),
            Self::RussianFirstAudio => Some(Self::Completed),
            Self::Completed | Self::Failed | Self::Stopped => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct RoundTripLatency {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outgoing_first_audio_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub english_monitor_complete_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incoming_first_audio_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_mic_onset_to_returned_ru_first_audible_ms: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundTripErrorCode {
    HeadphonesRequired,
    ProviderUnavailable,
    VirtualGraphUnavailable,
    IncomingRouteConflict,
    AlreadyRunning,
    NotRunning,
    SessionMismatch,
    InvalidCheckpoint,
    ExactPcmMismatch,
    RuntimeFailed,
    Timeout,
}

#[derive(Clone, Serialize)]
pub struct RoundTripDebugText {
    pub transcript: String,
    pub translation: String,
}

impl std::fmt::Debug for RoundTripDebugText {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RoundTripDebugText([REDACTED])")
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RoundTripStatus {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<RoundTripCheckpoint>,
    pub recursion_count: u32,
    pub latency: RoundTripLatency,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_pcm: Option<ExactPcmProof>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_error: Option<RoundTripErrorCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_text: Option<RoundTripDebugText>,
}

#[derive(Debug)]
struct Session {
    id: Uuid,
    started_at_ms: u64,
    checkpoint: RoundTripCheckpoint,
    recursion_count: u32,
    latency: RoundTripLatency,
    safe_error: Option<RoundTripErrorCode>,
    debug_text: Option<RoundTripDebugText>,
    exact_pcm: Option<ExactPcmProof>,
}

#[derive(Debug, Default)]
pub struct RoundTripSelfTest {
    session: Option<Session>,
}

impl RoundTripSelfTest {
    pub fn start(
        &mut self,
        preconditions: RoundTripPreconditions,
        at_ms: u64,
    ) -> Result<Uuid, RoundTripErrorCode> {
        if self
            .session
            .as_ref()
            .is_some_and(|session| !session.checkpoint.is_terminal())
        {
            return Err(RoundTripErrorCode::AlreadyRunning);
        }
        validate_preconditions(preconditions)?;
        let id = Uuid::new_v4();
        self.session = Some(Session {
            id,
            started_at_ms: at_ms,
            checkpoint: RoundTripCheckpoint::WaitingForSpeech,
            recursion_count: 0,
            latency: RoundTripLatency::default(),
            safe_error: None,
            debug_text: None,
            exact_pcm: None,
        });
        Ok(id)
    }

    pub fn advance(
        &mut self,
        session_id: Uuid,
        checkpoint: RoundTripCheckpoint,
        latency: RoundTripLatency,
    ) -> Result<(), RoundTripErrorCode> {
        let session = self.active_session_mut(session_id)?;
        if session.checkpoint.next() != Some(checkpoint)
            || (checkpoint == RoundTripCheckpoint::VirtualPeerReinjecting
                && (latency.english_monitor_complete_ms.is_none()
                    || !session
                        .exact_pcm
                        .as_ref()
                        .is_some_and(|proof| proof.exact_match)))
        {
            return Err(RoundTripErrorCode::InvalidCheckpoint);
        }
        session.checkpoint = checkpoint;
        session.latency = merge_latency(session.latency, latency);
        Ok(())
    }

    pub fn record_recursion_trigger(&mut self, session_id: Uuid) -> Result<(), RoundTripErrorCode> {
        let Some(session) = self.session.as_mut() else {
            return Err(RoundTripErrorCode::NotRunning);
        };
        if session.id != session_id {
            return Err(RoundTripErrorCode::SessionMismatch);
        }
        session.recursion_count = session.recursion_count.saturating_add(1);
        Ok(())
    }

    pub fn set_exact_pcm_proof(
        &mut self,
        session_id: Uuid,
        proof: ExactPcmProof,
    ) -> Result<(), RoundTripErrorCode> {
        let session = self.active_session_mut(session_id)?;
        if session.checkpoint != RoundTripCheckpoint::EnglishFirstAudio || !proof.exact_match {
            return Err(RoundTripErrorCode::ExactPcmMismatch);
        }
        session.exact_pcm = Some(proof);
        Ok(())
    }

    pub fn set_debug_text(
        &mut self,
        session_id: Uuid,
        transcript: impl Into<String>,
        translation: impl Into<String>,
    ) -> Result<(), RoundTripErrorCode> {
        let session = self.active_session_mut(session_id)?;
        session.debug_text = Some(RoundTripDebugText {
            transcript: transcript.into(),
            translation: translation.into(),
        });
        Ok(())
    }

    pub fn stop(&mut self, session_id: Uuid) -> bool {
        let Some(session) = self.session.as_mut() else {
            return false;
        };
        if session.id != session_id || session.checkpoint.is_terminal() {
            return false;
        }
        session.checkpoint = RoundTripCheckpoint::Stopped;
        session.debug_text = None;
        true
    }

    pub fn fail(&mut self, session_id: Uuid, error: RoundTripErrorCode) -> bool {
        let Ok(session) = self.active_session_mut(session_id) else {
            return false;
        };
        session.checkpoint = RoundTripCheckpoint::Failed;
        session.safe_error = Some(error);
        session.debug_text = None;
        true
    }

    pub fn expire(&mut self, at_ms: u64) -> bool {
        let Some(session) = self.session.as_mut() else {
            return false;
        };
        if session.checkpoint.is_terminal()
            || at_ms.saturating_sub(session.started_at_ms) < MAX_SESSION_MS
        {
            return false;
        }
        session.checkpoint = RoundTripCheckpoint::Stopped;
        session.safe_error = Some(RoundTripErrorCode::Timeout);
        session.debug_text = None;
        true
    }

    pub fn status(&self, include_debug_text: bool) -> RoundTripStatus {
        let Some(session) = self.session.as_ref() else {
            return RoundTripStatus::default();
        };
        RoundTripStatus {
            session_id: Some(session.id),
            checkpoint: Some(session.checkpoint),
            recursion_count: session.recursion_count,
            latency: session.latency,
            exact_pcm: session.exact_pcm.clone(),
            safe_error: session.safe_error,
            debug_text: include_debug_text
                .then(|| session.debug_text.clone())
                .flatten(),
        }
    }

    fn active_session_mut(&mut self, session_id: Uuid) -> Result<&mut Session, RoundTripErrorCode> {
        let Some(session) = self.session.as_mut() else {
            return Err(RoundTripErrorCode::NotRunning);
        };
        if session.id != session_id {
            return Err(RoundTripErrorCode::SessionMismatch);
        }
        if session.checkpoint.is_terminal() {
            return Err(RoundTripErrorCode::NotRunning);
        }
        Ok(session)
    }
}

fn validate_preconditions(preconditions: RoundTripPreconditions) -> Result<(), RoundTripErrorCode> {
    if !preconditions.headphones {
        return Err(RoundTripErrorCode::HeadphonesRequired);
    }
    if !preconditions.outgoing_provider_ready || !preconditions.incoming_provider_ready {
        return Err(RoundTripErrorCode::ProviderUnavailable);
    }
    if !preconditions.virtual_graph_ready {
        return Err(RoundTripErrorCode::VirtualGraphUnavailable);
    }
    if !preconditions.incoming_route_idle {
        return Err(RoundTripErrorCode::IncomingRouteConflict);
    }
    Ok(())
}

const fn merge_latency(current: RoundTripLatency, update: RoundTripLatency) -> RoundTripLatency {
    RoundTripLatency {
        outgoing_first_audio_ms: if update.outgoing_first_audio_ms.is_some() {
            update.outgoing_first_audio_ms
        } else {
            current.outgoing_first_audio_ms
        },
        english_monitor_complete_ms: if update.english_monitor_complete_ms.is_some() {
            update.english_monitor_complete_ms
        } else {
            current.english_monitor_complete_ms
        },
        incoming_first_audio_ms: if update.incoming_first_audio_ms.is_some() {
            update.incoming_first_audio_ms
        } else {
            current.incoming_first_audio_ms
        },
        physical_mic_onset_to_returned_ru_first_audible_ms: if update
            .physical_mic_onset_to_returned_ru_first_audible_ms
            .is_some()
        {
            update.physical_mic_onset_to_returned_ru_first_audible_ms
        } else {
            current.physical_mic_onset_to_returned_ru_first_audible_ms
        },
    }
}
