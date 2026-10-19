use std::{
    collections::{HashMap, VecDeque},
    time::Duration,
};

use thiserror::Error;
use translator_ipc::{
    ProviderEventValidator, ProviderSessionContract, ProviderValidationError,
    provider::{ProviderEvent, UtteranceOutcome, provider_event},
};
use uuid::Uuid;

use crate::CLOSE_ACK_TIMEOUT;

const MAX_RECEIVE_BUFFERED_MS: u32 = 400;

// Real speech providers deliver 400-ms PCM batches up to 690 ms apart.
// This guard starts after owned playable audio, not after each 20-ms IPC frame.
pub const INTER_AUDIO_DELTA_TIMEOUT: Duration = Duration::from_millis(500);
pub const CANCEL_FINAL_TIMEOUT: Duration = Duration::from_millis(250);
const COLLECTING_INPUT_TIMEOUT: Duration = Duration::from_millis(12_000);
const FIRST_AUDIO_AFTER_EOU_TIMEOUT: Duration = Duration::from_millis(6_000);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchdogAction {
    CancelUtterance {
        session_id: Uuid,
        stream_id: Uuid,
        utterance_id: Uuid,
        purge_receive_buffer: bool,
    },
    CloseProviderSession {
        session_id: Uuid,
    },
    RestartSidecar {
        session_id: Uuid,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum WatchdogError {
    #[error("watchdog utterance is unknown")]
    UnknownUtterance,
    #[error("watchdog utterance already exists")]
    DuplicateUtterance,
    #[error("watchdog stream does not match")]
    StreamMismatch,
    #[error("watchdog cancellation is pending")]
    CancelPending,
    #[error("watchdog input has not reached end of utterance")]
    InputNotComplete,
    #[error("watchdog input already reached end of utterance")]
    InputAlreadyComplete,
    #[error("watchdog expected a cancelled final")]
    ExpectedCancelledFinal,
    #[error("watchdog session is terminal")]
    SessionTerminal,
    #[error("watchdog deadline expired")]
    DeadlineExpired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UtterancePhase {
    CollectingInput { deadline_ns: u64 },
    AwaitingFirstAudio { deadline_ns: u64 },
    StreamingAudio { deadline_ns: u64 },
    CancelPending { deadline_ns: u64 },
}

#[derive(Debug, Clone, Copy)]
struct WatchedUtterance {
    stream_id: Uuid,
    phase: UtterancePhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionPhase {
    Active,
    ClosePending { deadline_ns: u64 },
    RestartIssued,
    Terminal,
}

#[derive(Clone)]
pub struct ProviderAudioWatchdog {
    session_id: Uuid,
    utterances: HashMap<Uuid, WatchedUtterance>,
    session_phase: SessionPhase,
    first_audio_after_eou_timeout: Duration,
}

impl ProviderAudioWatchdog {
    pub fn new(session_id: Uuid) -> Self {
        Self {
            session_id,
            utterances: HashMap::new(),
            session_phase: SessionPhase::Active,
            first_audio_after_eou_timeout: FIRST_AUDIO_AFTER_EOU_TIMEOUT,
        }
    }

    pub fn start_utterance(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        capture_ns: u64,
    ) -> Result<(), WatchdogError> {
        self.require_active_session()?;
        if self.utterances.contains_key(&utterance_id) {
            return Err(WatchdogError::DuplicateUtterance);
        }
        let deadline_ns = capture_ns.saturating_add(COLLECTING_INPUT_TIMEOUT.as_nanos() as u64);
        self.utterances.insert(
            utterance_id,
            WatchedUtterance {
                stream_id,
                phase: UtterancePhase::CollectingInput { deadline_ns },
            },
        );
        Ok(())
    }

    pub fn record_end_of_utterance(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        now_ns: u64,
    ) -> Result<(), WatchdogError> {
        self.require_active_session()?;
        let watched = self
            .utterances
            .get_mut(&utterance_id)
            .ok_or(WatchdogError::UnknownUtterance)?;
        if watched.stream_id != stream_id {
            return Err(WatchdogError::StreamMismatch);
        }
        match watched.phase {
            UtterancePhase::CollectingInput { deadline_ns } if now_ns <= deadline_ns => {
                watched.phase = UtterancePhase::AwaitingFirstAudio {
                    deadline_ns: now_ns
                        .saturating_add(self.first_audio_after_eou_timeout.as_nanos() as u64),
                };
                Ok(())
            }
            UtterancePhase::CollectingInput { .. } => Err(WatchdogError::DeadlineExpired),
            UtterancePhase::AwaitingFirstAudio { .. } | UtterancePhase::StreamingAudio { .. } => {
                Err(WatchdogError::InputAlreadyComplete)
            }
            UtterancePhase::CancelPending { .. } => Err(WatchdogError::CancelPending),
        }
    }

    pub fn record_audio_delta(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        now_ns: u64,
    ) -> Result<(), WatchdogError> {
        self.require_active_session()?;
        let watched = self
            .utterances
            .get_mut(&utterance_id)
            .ok_or(WatchdogError::UnknownUtterance)?;
        if watched.stream_id != stream_id {
            return Err(WatchdogError::StreamMismatch);
        }
        let deadline_ns = match watched.phase {
            UtterancePhase::CollectingInput { deadline_ns } => {
                return if now_ns <= deadline_ns {
                    Ok(())
                } else {
                    Err(WatchdogError::DeadlineExpired)
                };
            }
            UtterancePhase::AwaitingFirstAudio { deadline_ns }
            | UtterancePhase::StreamingAudio { deadline_ns } => deadline_ns,
            UtterancePhase::CancelPending { .. } => return Err(WatchdogError::CancelPending),
        };
        if now_ns > deadline_ns {
            return Err(WatchdogError::DeadlineExpired);
        }
        watched.phase = UtterancePhase::StreamingAudio {
            deadline_ns: now_ns.saturating_add(INTER_AUDIO_DELTA_TIMEOUT.as_nanos() as u64),
        };
        Ok(())
    }

    pub fn record_completed_final(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
    ) -> Result<(), WatchdogError> {
        self.finish_utterance(stream_id, utterance_id, false)
    }

    pub fn record_cancelled_final(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
    ) -> Result<(), WatchdogError> {
        self.finish_utterance(stream_id, utterance_id, true)
    }

    pub fn disarm_utterance(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
    ) -> Result<(), WatchdogError> {
        let watched = self
            .utterances
            .get(&utterance_id)
            .ok_or(WatchdogError::UnknownUtterance)?;
        if watched.stream_id != stream_id {
            return Err(WatchdogError::StreamMismatch);
        }
        self.utterances.remove(&utterance_id);
        Ok(())
    }

    pub fn disarm_session(&mut self) {
        self.utterances.clear();
        self.session_phase = SessionPhase::Terminal;
    }

    pub(crate) fn defer_provider_receipt(&mut self, blocked_ns: u64) {
        // Deliberate local receive backpressure is not provider silence.
        for watched in self.utterances.values_mut() {
            match &mut watched.phase {
                UtterancePhase::AwaitingFirstAudio { deadline_ns } => {
                    *deadline_ns = deadline_ns.saturating_add(blocked_ns);
                }
                UtterancePhase::CollectingInput { .. }
                | UtterancePhase::StreamingAudio { .. }
                | UtterancePhase::CancelPending { .. } => {}
            }
        }
    }

    pub(crate) fn credit_playable_audio(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        now_ns: u64,
        playable_until_ns: u64,
    ) {
        if let Some(watched) = self.utterances.get_mut(&utterance_id)
            && watched.stream_id == stream_id
            && let UtterancePhase::StreamingAudio { deadline_ns } = &mut watched.phase
            && now_ns <= *deadline_ns
        {
            *deadline_ns = (*deadline_ns)
                .max(playable_until_ns.saturating_add(INTER_AUDIO_DELTA_TIMEOUT.as_nanos() as u64));
        }
    }

    pub fn record_session_closed(&mut self, session_id: Uuid) -> Result<(), WatchdogError> {
        if session_id != self.session_id {
            return Err(WatchdogError::StreamMismatch);
        }
        self.disarm_session();
        Ok(())
    }

    pub fn cancel_pending(&self, stream_id: Uuid, utterance_id: Uuid) -> bool {
        self.utterances.get(&utterance_id).is_some_and(|watched| {
            watched.stream_id == stream_id
                && matches!(watched.phase, UtterancePhase::CancelPending { .. })
        })
    }

    pub fn active_utterance_count(&self) -> usize {
        self.utterances.len()
    }

    /// Returns the earliest stored deadline for the current phase, without polling.
    pub fn next_phase_deadline_ns(&self) -> Option<u64> {
        match self.session_phase {
            SessionPhase::ClosePending { deadline_ns } => Some(deadline_ns),
            SessionPhase::Active => self
                .utterances
                .values()
                .map(|watched| match watched.phase {
                    UtterancePhase::CollectingInput { deadline_ns }
                    | UtterancePhase::AwaitingFirstAudio { deadline_ns }
                    | UtterancePhase::StreamingAudio { deadline_ns }
                    | UtterancePhase::CancelPending { deadline_ns } => deadline_ns,
                })
                .min(),
            SessionPhase::RestartIssued | SessionPhase::Terminal => None,
        }
    }

    fn check_audio_delta(
        &self,
        stream_id: Uuid,
        utterance_id: Uuid,
        now_ns: u64,
    ) -> Result<(), WatchdogError> {
        self.require_active_session()?;
        let watched = self
            .utterances
            .get(&utterance_id)
            .ok_or(WatchdogError::UnknownUtterance)?;
        if watched.stream_id != stream_id {
            return Err(WatchdogError::StreamMismatch);
        }
        let deadline_ns = match watched.phase {
            UtterancePhase::CollectingInput { deadline_ns }
            | UtterancePhase::AwaitingFirstAudio { deadline_ns }
            | UtterancePhase::StreamingAudio { deadline_ns } => deadline_ns,
            UtterancePhase::CancelPending { .. } => return Err(WatchdogError::CancelPending),
        };
        if now_ns > deadline_ns {
            return Err(WatchdogError::DeadlineExpired);
        }
        Ok(())
    }

    fn check_final(
        &self,
        stream_id: Uuid,
        utterance_id: Uuid,
        outcome: UtteranceOutcome,
        now_ns: u64,
    ) -> Result<(), WatchdogError> {
        self.require_active_session()?;
        let watched = self
            .utterances
            .get(&utterance_id)
            .ok_or(WatchdogError::UnknownUtterance)?;
        if watched.stream_id != stream_id {
            return Err(WatchdogError::StreamMismatch);
        }
        match (watched.phase, outcome) {
            (UtterancePhase::CollectingInput { .. }, _) => Err(WatchdogError::InputNotComplete),
            (UtterancePhase::CancelPending { deadline_ns }, UtteranceOutcome::Cancelled)
                if now_ns <= deadline_ns =>
            {
                Ok(())
            }
            (UtterancePhase::CancelPending { .. }, _) => Err(WatchdogError::ExpectedCancelledFinal),
            (
                UtterancePhase::AwaitingFirstAudio { deadline_ns }
                | UtterancePhase::StreamingAudio { deadline_ns },
                UtteranceOutcome::Completed | UtteranceOutcome::Dropped,
            ) if now_ns <= deadline_ns => Ok(()),
            (
                UtterancePhase::AwaitingFirstAudio { .. } | UtterancePhase::StreamingAudio { .. },
                UtteranceOutcome::Cancelled,
            ) => Err(WatchdogError::ExpectedCancelledFinal),
            _ => Err(WatchdogError::DeadlineExpired),
        }
    }

    fn session_accepts_events(&self) -> bool {
        self.session_phase == SessionPhase::Active
    }

    pub fn poll(&mut self, now_ns: u64) -> Vec<WatchdogAction> {
        match self.session_phase {
            SessionPhase::ClosePending { deadline_ns } if now_ns >= deadline_ns => {
                self.session_phase = SessionPhase::RestartIssued;
                return vec![WatchdogAction::RestartSidecar {
                    session_id: self.session_id,
                }];
            }
            SessionPhase::Active => {}
            _ => return Vec::new(),
        }

        let mut cancel = Vec::new();
        let mut close_required = false;
        for (utterance_id, watched) in &mut self.utterances {
            match watched.phase {
                UtterancePhase::CollectingInput { deadline_ns }
                | UtterancePhase::AwaitingFirstAudio { deadline_ns }
                | UtterancePhase::StreamingAudio { deadline_ns }
                    if now_ns > deadline_ns =>
                {
                    watched.phase = UtterancePhase::CancelPending {
                        deadline_ns: now_ns.saturating_add(CANCEL_FINAL_TIMEOUT.as_nanos() as u64),
                    };
                    cancel.push(WatchdogAction::CancelUtterance {
                        session_id: self.session_id,
                        stream_id: watched.stream_id,
                        utterance_id: *utterance_id,
                        purge_receive_buffer: true,
                    });
                }
                UtterancePhase::CancelPending { deadline_ns } if now_ns > deadline_ns => {
                    close_required = true;
                }
                _ => {}
            }
        }
        if close_required {
            self.session_phase = SessionPhase::ClosePending {
                deadline_ns: now_ns.saturating_add(CLOSE_ACK_TIMEOUT.as_nanos() as u64),
            };
            return vec![WatchdogAction::CloseProviderSession {
                session_id: self.session_id,
            }];
        }
        cancel
    }

    fn cancel_utterance(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        now_ns: u64,
    ) -> Result<WatchdogAction, WatchdogError> {
        self.require_active_session()?;
        let watched = self
            .utterances
            .get_mut(&utterance_id)
            .ok_or(WatchdogError::UnknownUtterance)?;
        if watched.stream_id != stream_id {
            return Err(WatchdogError::StreamMismatch);
        }
        match watched.phase {
            UtterancePhase::CollectingInput { .. } => {
                return Err(WatchdogError::InputNotComplete);
            }
            UtterancePhase::CancelPending { .. } => {
                return Err(WatchdogError::CancelPending);
            }
            UtterancePhase::AwaitingFirstAudio { .. } | UtterancePhase::StreamingAudio { .. } => {}
        }
        watched.phase = UtterancePhase::CancelPending {
            deadline_ns: now_ns.saturating_add(CANCEL_FINAL_TIMEOUT.as_nanos() as u64),
        };
        Ok(WatchdogAction::CancelUtterance {
            session_id: self.session_id,
            stream_id,
            utterance_id,
            purge_receive_buffer: true,
        })
    }

    fn finish_utterance(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        cancelled: bool,
    ) -> Result<(), WatchdogError> {
        self.require_active_session()?;
        let watched = self
            .utterances
            .get(&utterance_id)
            .ok_or(WatchdogError::UnknownUtterance)?;
        if watched.stream_id != stream_id {
            return Err(WatchdogError::StreamMismatch);
        }
        if matches!(watched.phase, UtterancePhase::CollectingInput { .. }) {
            return Err(WatchdogError::InputNotComplete);
        }
        let cancel_pending = matches!(watched.phase, UtterancePhase::CancelPending { .. });
        if cancel_pending != cancelled {
            return Err(WatchdogError::ExpectedCancelledFinal);
        }
        self.utterances.remove(&utterance_id);
        Ok(())
    }

    fn require_active_session(&self) -> Result<(), WatchdogError> {
        if self.session_phase == SessionPhase::Active {
            Ok(())
        } else {
            Err(WatchdogError::SessionTerminal)
        }
    }
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum ProviderStreamCoordinatorError {
    #[error(transparent)]
    Watchdog(#[from] WatchdogError),
    #[error(transparent)]
    Validation(#[from] ProviderValidationError),
    #[error("provider receive buffer is full")]
    ReceiveBufferFull,
    #[error("provider receive buffer is empty")]
    ReceiveBufferEmpty,
    #[error("provider event identifier is invalid")]
    InvalidIdentifier,
}

pub struct ProviderStreamCoordinator {
    watchdog: ProviderAudioWatchdog,
    validator: ProviderEventValidator,
    receive_buffers: HashMap<(Uuid, Uuid), ReceiveBuffer>,
}

#[derive(Default)]
struct ReceiveBuffer {
    frames: VecDeque<BufferedFrame>,
    buffered_ms: u32,
}

struct BufferedFrame {
    pcm: Vec<u8>,
    duration_ms: u32,
}

impl ProviderStreamCoordinator {
    pub(crate) fn credit_playable_audio(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        now_ns: u64,
        playable_until_ns: u64,
    ) {
        self.watchdog
            .credit_playable_audio(stream_id, utterance_id, now_ns, playable_until_ns);
    }

    pub(crate) fn defer_provider_receipt(&mut self, blocked_ns: u64) {
        self.watchdog.defer_provider_receipt(blocked_ns);
    }

    pub fn new(contract: ProviderSessionContract) -> Self {
        Self {
            watchdog: ProviderAudioWatchdog {
                first_audio_after_eou_timeout: Duration::from_millis(u64::from(
                    translator_ipc::first_response_timeout_ms(contract.provider_id, contract.mode),
                )),
                ..ProviderAudioWatchdog::new(contract.session_id)
            },
            validator: ProviderEventValidator::new(contract),
            receive_buffers: HashMap::new(),
        }
    }

    pub fn start_utterance(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        capture_ns: u64,
    ) -> Result<(), ProviderStreamCoordinatorError> {
        self.watchdog
            .start_utterance(stream_id, utterance_id, capture_ns)?;
        if let Err(error) = self.validator.record_input(utterance_id, capture_ns) {
            let _ = self.watchdog.disarm_utterance(stream_id, utterance_id);
            return Err(error.into());
        }
        Ok(())
    }

    pub fn record_end_of_utterance(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        now_ns: u64,
    ) -> Result<(), ProviderStreamCoordinatorError> {
        self.watchdog
            .record_end_of_utterance(stream_id, utterance_id, now_ns)?;
        self.validator
            .record_end_of_utterance(utterance_id, now_ns)?;
        Ok(())
    }

    fn buffer_receive_audio(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        pcm: Vec<u8>,
        frame_duration_ms: u32,
    ) -> Result<(), ProviderStreamCoordinatorError> {
        let buffer = self
            .receive_buffers
            .entry((stream_id, utterance_id))
            .or_default();
        let Some(projected_ms) = buffer.buffered_ms.checked_add(frame_duration_ms) else {
            return Err(ProviderStreamCoordinatorError::ReceiveBufferFull);
        };
        if projected_ms > MAX_RECEIVE_BUFFERED_MS {
            return Err(ProviderStreamCoordinatorError::ReceiveBufferFull);
        }
        buffer.frames.push_back(BufferedFrame {
            pcm,
            duration_ms: frame_duration_ms,
        });
        buffer.buffered_ms = projected_ms;
        Ok(())
    }

    fn can_buffer_receive_audio(
        &self,
        stream_id: Uuid,
        utterance_id: Uuid,
        frame_duration_ms: u32,
    ) -> Result<(), ProviderStreamCoordinatorError> {
        let buffered_ms = self
            .receive_buffers
            .get(&(stream_id, utterance_id))
            .map_or(0, |buffer| buffer.buffered_ms);
        if buffered_ms
            .checked_add(frame_duration_ms)
            .is_none_or(|projected| projected > MAX_RECEIVE_BUFFERED_MS)
        {
            return Err(ProviderStreamCoordinatorError::ReceiveBufferFull);
        }
        Ok(())
    }

    pub fn consume_receive_audio(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
    ) -> Result<Vec<u8>, ProviderStreamCoordinatorError> {
        let key = (stream_id, utterance_id);
        let buffer = self
            .receive_buffers
            .get_mut(&key)
            .ok_or(ProviderStreamCoordinatorError::ReceiveBufferEmpty)?;
        let frame = buffer
            .frames
            .pop_front()
            .ok_or(ProviderStreamCoordinatorError::ReceiveBufferEmpty)?;
        buffer.buffered_ms = buffer.buffered_ms.saturating_sub(frame.duration_ms);
        if buffer.frames.is_empty() {
            self.receive_buffers.remove(&key);
        }
        Ok(frame.pcm)
    }

    pub fn receive_buffered_frames(&self, stream_id: Uuid, utterance_id: Uuid) -> usize {
        self.receive_buffers
            .get(&(stream_id, utterance_id))
            .map_or(0, |buffer| buffer.frames.len())
    }

    pub fn validate_event(
        &mut self,
        event: &ProviderEvent,
        now_ns: u64,
    ) -> Result<(), ProviderStreamCoordinatorError> {
        let event_kind = event
            .event
            .as_ref()
            .ok_or(ProviderValidationError::MissingEvent)?;
        if !self.watchdog.session_accepts_events()
            && !matches!(event_kind, provider_event::Event::SessionClosed(_))
        {
            return Err(WatchdogError::SessionTerminal.into());
        }
        match event_kind {
            provider_event::Event::AudioDelta(value) => {
                let stream_id = parse_uuid(&value.stream_id)?;
                let utterance_id = parse_uuid(&value.utterance_id)?;
                if self.watchdog.cancel_pending(stream_id, utterance_id) {
                    self.validator.validate(event, now_ns)?;
                    return Ok(());
                }
                self.watchdog
                    .check_audio_delta(stream_id, utterance_id, now_ns)?;
                self.can_buffer_receive_audio(stream_id, utterance_id, value.frame_duration_ms)?;
                self.validator.validate(event, now_ns)?;
                self.watchdog
                    .record_audio_delta(stream_id, utterance_id, now_ns)?;
                self.buffer_receive_audio(
                    stream_id,
                    utterance_id,
                    value.pcm.clone(),
                    value.frame_duration_ms,
                )?;
            }
            provider_event::Event::UtteranceFinal(value) => {
                let stream_id = parse_uuid(&value.stream_id)?;
                let utterance_id = parse_uuid(&value.utterance_id)?;
                let outcome = UtteranceOutcome::try_from(value.outcome)
                    .map_err(|_| ProviderValidationError::InvalidOutcome)?;
                self.watchdog
                    .check_final(stream_id, utterance_id, outcome, now_ns)?;
                self.validator.validate(event, now_ns)?;
                match outcome {
                    UtteranceOutcome::Cancelled => self
                        .watchdog
                        .record_cancelled_final(stream_id, utterance_id)?,
                    UtteranceOutcome::Completed | UtteranceOutcome::Dropped => self
                        .watchdog
                        .record_completed_final(stream_id, utterance_id)?,
                    _ => {}
                }
                if outcome != UtteranceOutcome::Completed {
                    self.receive_buffers.remove(&(stream_id, utterance_id));
                }
            }
            provider_event::Event::SessionClosed(value) => {
                self.validator.validate(event, now_ns)?;
                self.watchdog
                    .record_session_closed(parse_uuid(&value.session_id)?)?;
                self.receive_buffers.clear();
            }
            _ => self.validator.validate(event, now_ns)?,
        }
        Ok(())
    }

    pub fn poll(
        &mut self,
        now_ns: u64,
    ) -> Result<Vec<WatchdogAction>, ProviderStreamCoordinatorError> {
        let previous_watchdog = self.watchdog.clone();
        let actions = self.watchdog.poll(now_ns);
        for action in &actions {
            if let WatchdogAction::CancelUtterance { utterance_id, .. } = action
                && let Err(error) = self.validator.can_record_cancel_requested(*utterance_id)
            {
                self.watchdog = previous_watchdog;
                return Err(error.into());
            }
        }
        for action in &actions {
            if let WatchdogAction::CancelUtterance {
                stream_id,
                utterance_id,
                purge_receive_buffer,
                ..
            } = action
            {
                self.validator.record_cancel_requested(*utterance_id)?;
                if *purge_receive_buffer {
                    self.receive_buffers.remove(&(*stream_id, *utterance_id));
                }
            }
            if matches!(
                action,
                WatchdogAction::CloseProviderSession { .. } | WatchdogAction::RestartSidecar { .. }
            ) {
                self.receive_buffers.clear();
            }
        }
        Ok(actions)
    }

    pub fn cancel_expired_utterance(
        &mut self,
        stream_id: Uuid,
        utterance_id: Uuid,
        now_ns: u64,
    ) -> Result<WatchdogAction, ProviderStreamCoordinatorError> {
        let previous_watchdog = self.watchdog.clone();
        let action = self
            .watchdog
            .cancel_utterance(stream_id, utterance_id, now_ns)?;
        if let Err(error) = self.validator.record_cancel_requested(utterance_id) {
            self.watchdog = previous_watchdog;
            return Err(error.into());
        }
        self.receive_buffers.remove(&(stream_id, utterance_id));
        Ok(action)
    }

    pub fn validator_cancel_pending(&self, utterance_id: Uuid) -> bool {
        self.validator.cancel_pending(utterance_id)
    }

    pub fn watchdog(&self) -> &ProviderAudioWatchdog {
        &self.watchdog
    }

    pub fn next_phase_deadline_ns(&self) -> Option<u64> {
        self.watchdog.next_phase_deadline_ns()
    }
}

fn parse_uuid(value: &str) -> Result<Uuid, ProviderStreamCoordinatorError> {
    Uuid::parse_str(value).map_err(|_| ProviderStreamCoordinatorError::InvalidIdentifier)
}

#[cfg(test)]
mod backpressure_tests {
    use super::*;
    use proptest::{prelude::*, test_runner::RngSeed};

    proptest! {
        #![proptest_config(ProptestConfig { cases: 100, max_shrink_iters: 512,
            rng_seed: RngSeed::Fixed(0x2026_1019), ..ProptestConfig::default() })]

        #[test]
        fn playable_credit_is_identity_phase_and_deadline_scoped(
            phase in 0_u8..4, same_stream in any::<bool>(), same_utterance in any::<bool>(),
            late in any::<bool>(), credit_ms in 0_u64..=400,
        ) {
            let stream = Uuid::from_u128(1);
            let utterance = Uuid::from_u128(2);
            let mut watchdog = ProviderAudioWatchdog::new(Uuid::from_u128(3));
            watchdog.start_utterance(stream, utterance, 0).unwrap();
            if phase > 0 { watchdog.record_end_of_utterance(stream, utterance, 1).unwrap(); }
            if phase > 1 { watchdog.record_audio_delta(stream, utterance, 2).unwrap(); }
            if phase > 2 { watchdog.cancel_utterance(stream, utterance, 3).unwrap(); }
            let original = watchdog.next_phase_deadline_ns().unwrap();
            let now = if late { original + 1 } else { 4 };
            let until = original + credit_ms * 1_000_000;
            watchdog.credit_playable_audio(
                if same_stream { stream } else { Uuid::from_u128(4) },
                if same_utterance { utterance } else { Uuid::from_u128(5) }, now, until);
            let expected = if phase == 2 && same_stream && same_utterance && !late {
                until + INTER_AUDIO_DELTA_TIMEOUT.as_nanos() as u64
            } else { original };
            prop_assert_eq!(watchdog.next_phase_deadline_ns(), Some(expected));
            // Repeating the same receipt never manufactures additional media time.
            watchdog.credit_playable_audio(
                if same_stream { stream } else { Uuid::from_u128(4) },
                if same_utterance { utterance } else { Uuid::from_u128(5) }, now, until);
            prop_assert_eq!(watchdog.next_phase_deadline_ns(), Some(expected));
        }
    }

    #[test]
    fn deliberate_receive_backpressure_preserves_active_provider_receipt_budget() {
        let stream = Uuid::new_v4();
        let utterance = Uuid::new_v4();
        let mut watchdog = ProviderAudioWatchdog::new(Uuid::new_v4());
        watchdog.start_utterance(stream, utterance, 0).unwrap();
        watchdog
            .record_end_of_utterance(stream, utterance, 1)
            .unwrap();
        let original = watchdog.next_phase_deadline_ns().unwrap();
        let blocked = 350_000_000;
        watchdog.defer_provider_receipt(blocked);
        assert!(watchdog.poll(original + blocked).is_empty());
        assert!(matches!(
            watchdog.poll(original + blocked + 1).as_slice(),
            [WatchdogAction::CancelUtterance { .. }]
        ));
        let cancel_deadline = watchdog.next_phase_deadline_ns();
        watchdog.defer_provider_receipt(blocked);
        assert_eq!(
            watchdog.next_phase_deadline_ns(),
            cancel_deadline,
            "local backpressure cannot prolong cancellation"
        );
    }

    #[test]
    fn deliberate_receive_backpressure_does_not_prolong_input_collection() {
        let stream = Uuid::new_v4();
        let utterance = Uuid::new_v4();
        let mut watchdog = ProviderAudioWatchdog::new(Uuid::new_v4());
        watchdog.start_utterance(stream, utterance, 0).unwrap();
        let original = watchdog.next_phase_deadline_ns().unwrap();
        watchdog.defer_provider_receipt(350_000_000);
        assert_eq!(watchdog.next_phase_deadline_ns(), Some(original));
        assert!(matches!(
            watchdog.poll(original + 1).as_slice(),
            [WatchdogAction::CancelUtterance { .. }]
        ));
    }
}
