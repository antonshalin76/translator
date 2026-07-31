use translator_core::AudioDirection;

const MAX_BUFFERED_MS: u32 = 400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueKind {
    Capture,
    Playback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuePushResult {
    Accepted,
    RejectedOverflow,
    RejectedInvalidDuration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueConsumeResult {
    Consumed,
    RejectedUnderflow,
    RejectedInvalidDuration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueueState {
    pub buffered_ms: u32,
    pub dropped_frames: u64,
}

#[derive(Debug, Default)]
pub struct DaemonQueues {
    microphone_capture: QueueState,
    microphone_playback: QueueState,
    speaker_capture: QueueState,
    speaker_playback: QueueState,
}

impl DaemonQueues {
    pub fn push(
        &mut self,
        direction: AudioDirection,
        kind: QueueKind,
        frame_duration_ms: u32,
    ) -> QueuePushResult {
        if !valid_frame_duration(frame_duration_ms) {
            return QueuePushResult::RejectedInvalidDuration;
        }
        let state = self.state_mut(direction, kind);
        let Some(projected) = state.buffered_ms.checked_add(frame_duration_ms) else {
            state.dropped_frames += 1;
            return QueuePushResult::RejectedOverflow;
        };
        if projected > MAX_BUFFERED_MS {
            state.dropped_frames += 1;
            return QueuePushResult::RejectedOverflow;
        }
        state.buffered_ms = projected;
        QueuePushResult::Accepted
    }

    pub fn consume(
        &mut self,
        direction: AudioDirection,
        kind: QueueKind,
        frame_duration_ms: u32,
    ) -> QueueConsumeResult {
        if !valid_frame_duration(frame_duration_ms) {
            return QueueConsumeResult::RejectedInvalidDuration;
        }
        let state = self.state_mut(direction, kind);
        let Some(remaining) = state.buffered_ms.checked_sub(frame_duration_ms) else {
            return QueueConsumeResult::RejectedUnderflow;
        };
        state.buffered_ms = remaining;
        QueueConsumeResult::Consumed
    }

    pub fn state(&self, direction: AudioDirection, kind: QueueKind) -> QueueState {
        match (direction, kind) {
            (AudioDirection::Microphone, QueueKind::Capture) => self.microphone_capture,
            (AudioDirection::Microphone, QueueKind::Playback) => self.microphone_playback,
            (AudioDirection::Speaker, QueueKind::Capture) => self.speaker_capture,
            (AudioDirection::Speaker, QueueKind::Playback) => self.speaker_playback,
        }
    }

    fn state_mut(&mut self, direction: AudioDirection, kind: QueueKind) -> &mut QueueState {
        match (direction, kind) {
            (AudioDirection::Microphone, QueueKind::Capture) => &mut self.microphone_capture,
            (AudioDirection::Microphone, QueueKind::Playback) => &mut self.microphone_playback,
            (AudioDirection::Speaker, QueueKind::Capture) => &mut self.speaker_capture,
            (AudioDirection::Speaker, QueueKind::Playback) => &mut self.speaker_playback,
        }
    }
}

const fn valid_frame_duration(duration_ms: u32) -> bool {
    matches!(duration_ms, 20 | 40 | 60 | 80 | 100)
}
