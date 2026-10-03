use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use thiserror::Error;
use translator_audio::{
    AEC_OBSERVATION_CHANNELS, AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT,
    AEC_OBSERVATION_FRAME_DURATION_MS, AEC_OBSERVATION_FRAME_DURATION_NS,
    AEC_OBSERVATION_FRAME_SAMPLES, AEC_OBSERVATION_SAMPLE_RATE_HZ, AEC_POSITIVE_CONTROL_MAX_GAP_NS,
    AecGraphIdentity, AecNativeObservationTiming, AecObservationEvidence, AecPositiveControl,
    NativeCaptureOrigin,
};
use translator_core::{AudioDirection, TranslationMode};
use uuid::Uuid;

use crate::translation_runtime::NativeCaptureTiming;
use crate::{
    CompletedCaptureFrame, DuplexRuntimeEvent, DuplexRuntimeObserver, ProviderEffectOrigin,
    TerminalOutcome,
};

const NATIVE_ADC_SAMPLES_PER_FRAME: u64 = 960;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AecObservationCollectorError {
    #[error("AEC observation lifecycle is invalid")]
    InvalidLifecycle,
    #[error("AEC observer generation activation does not match")]
    GenerationMismatch,
    #[error("AEC positive control is stale")]
    PositiveControlExpired,
    #[error("AEC scored interval boundary is invalid")]
    InvalidInterval,
}

pub struct AecObserverGenerationActivation {
    observer_generation: Uuid,
    instance_nonce: Uuid,
}

pub struct AecRuntimeObserver {
    observer_generation: Uuid,
    calibration_attempt_id: Uuid,
    challenge_id: Uuid,
    instance_nonce: Uuid,
    state: Mutex<CollectorState>,
}

#[derive(Default)]
struct CollectorState {
    activated: bool,
    phase: ObservationPhase,
}

#[derive(Default)]
enum ObservationPhase {
    #[default]
    Idle,
    Positive(PositiveState),
    PositiveReady {
        evidence: AecPositiveControl,
        started_monotonic_ns: u64,
        terminated_early: bool,
    },
    Scored(ScoredState),
    Complete(AecObservationEvidence),
}

struct PositiveState {
    started_monotonic_ns: u64,
    last_observed_monotonic_ns: u64,
    speech_started_events: u64,
    provider_submission_attempts: u64,
    provider_submissions_accepted: u64,
    resets: u64,
    observer_errors: u64,
    terminated_early: bool,
}

struct ScoredState {
    interval_id: Uuid,
    started_monotonic_ns: u64,
    ended_monotonic_ns: u64,
    stream_generation: Uuid,
    seen_sequences: HashSet<u64>,
    processed_frames: u64,
    first_frame_sequence: Option<u64>,
    last_frame_sequence: Option<u64>,
    first_capture_monotonic_ns: Option<u64>,
    last_capture_monotonic_ns: Option<u64>,
    maximum_frame_gap_ns: u64,
    sample_rate_hz: Option<u32>,
    channels: Option<u8>,
    frame_duration_ms: Option<u16>,
    samples_per_frame: Option<u64>,
    frame_gaps: u64,
    duplicate_frames: u64,
    out_of_order_frames: u64,
    vad_events: u64,
    provider_attempts: u64,
    provider_accepted: u64,
    resets: u64,
    dropped_frames: u64,
    observer_errors: u64,
    terminated_early: bool,
    positive_control: AecPositiveControl,
    positive_started_monotonic_ns: u64,
    pending_frames: u64,
    native_timing: Option<AecNativeObservationTiming>,
    native_next_adc: Option<u64>,
    native_seen_adc: HashSet<u64>,
}

impl ScoredState {
    fn contains_host_timestamp(&self, timestamp: u64) -> bool {
        let uncertainty = match &self.native_timing {
            Some(native) => match native.host_uncertainty_ns() {
                Some(value) => value,
                None => return false,
            },
            None => 0,
        };
        timestamp >= self.started_monotonic_ns.saturating_sub(uncertainty)
            && self
                .ended_monotonic_ns
                .checked_add(uncertainty)
                .is_some_and(|end| timestamp <= end)
    }
}

impl AecRuntimeObserver {
    pub fn new(
        observer_generation: Uuid,
        calibration_attempt_id: Uuid,
        challenge_id: Uuid,
    ) -> (Self, AecObserverGenerationActivation) {
        let instance_nonce = Uuid::new_v4();
        (
            Self {
                observer_generation,
                calibration_attempt_id,
                challenge_id,
                instance_nonce,
                state: Mutex::new(CollectorState::default()),
            },
            AecObserverGenerationActivation {
                observer_generation,
                instance_nonce,
            },
        )
    }

    pub fn activate_generation(
        &self,
        activation: AecObserverGenerationActivation,
    ) -> Result<(), AecObservationCollectorError> {
        if activation.observer_generation != self.observer_generation
            || activation.instance_nonce != self.instance_nonce
        {
            return Err(AecObservationCollectorError::GenerationMismatch);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| AecObservationCollectorError::InvalidLifecycle)?;
        if state.activated || !matches!(state.phase, ObservationPhase::Idle) {
            return Err(AecObservationCollectorError::InvalidLifecycle);
        }
        state.activated = true;
        Ok(())
    }

    pub fn start_positive_control(
        &self,
        started_monotonic_ns: u64,
    ) -> Result<(), AecObservationCollectorError> {
        let mut state = self.lock_active()?;
        if !matches!(state.phase, ObservationPhase::Idle) {
            return Err(AecObservationCollectorError::InvalidLifecycle);
        }
        state.phase = ObservationPhase::Positive(PositiveState {
            started_monotonic_ns,
            last_observed_monotonic_ns: started_monotonic_ns,
            speech_started_events: 0,
            provider_submission_attempts: 0,
            provider_submissions_accepted: 0,
            resets: 0,
            observer_errors: 0,
            terminated_early: false,
        });
        Ok(())
    }

    pub fn complete_positive_control(
        &self,
        completed_monotonic_ns: u64,
    ) -> Result<(), AecObservationCollectorError> {
        let mut state = self.lock_active()?;
        let ObservationPhase::Positive(positive) = &state.phase else {
            return Err(AecObservationCollectorError::InvalidLifecycle);
        };
        if completed_monotonic_ns < positive.started_monotonic_ns
            || completed_monotonic_ns < positive.last_observed_monotonic_ns
        {
            return Err(AecObservationCollectorError::InvalidInterval);
        }
        let evidence = AecPositiveControl {
            observer_generation: self.observer_generation.to_string(),
            calibration_attempt_id: self.calibration_attempt_id.to_string(),
            challenge_id: self.challenge_id.to_string(),
            completed_monotonic_ns,
            speech_started_events: positive.speech_started_events,
            provider_submission_attempts: positive.provider_submission_attempts,
            provider_submissions_accepted: positive.provider_submissions_accepted,
            resets: positive.resets,
            observer_errors: positive.observer_errors,
        };
        let terminated_early = positive.terminated_early;
        let started_monotonic_ns = positive.started_monotonic_ns;
        state.phase = ObservationPhase::PositiveReady {
            evidence,
            started_monotonic_ns,
            terminated_early,
        };
        Ok(())
    }

    pub fn start_scored_interval(
        &self,
        interval_id: Uuid,
        started_monotonic_ns: u64,
        stream_generation: Uuid,
    ) -> Result<(), AecObservationCollectorError> {
        let mut state = self.lock_active()?;
        let ObservationPhase::PositiveReady {
            evidence,
            started_monotonic_ns: positive_started_monotonic_ns,
            terminated_early,
        } = &state.phase
        else {
            return Err(AecObservationCollectorError::InvalidLifecycle);
        };
        let Some(gap) = started_monotonic_ns.checked_sub(evidence.completed_monotonic_ns) else {
            return Err(AecObservationCollectorError::InvalidInterval);
        };
        if gap > AEC_POSITIVE_CONTROL_MAX_GAP_NS {
            return Err(AecObservationCollectorError::PositiveControlExpired);
        }
        let ended_monotonic_ns = started_monotonic_ns
            .checked_add(AEC_OBSERVATION_DURATION_NS)
            .ok_or(AecObservationCollectorError::InvalidInterval)?;
        let positive_control = evidence.clone();
        let positive_started_monotonic_ns = *positive_started_monotonic_ns;
        let terminated_early = *terminated_early;
        state.phase = ObservationPhase::Scored(ScoredState {
            interval_id,
            started_monotonic_ns,
            ended_monotonic_ns,
            stream_generation,
            seen_sequences: HashSet::new(),
            processed_frames: 0,
            first_frame_sequence: None,
            last_frame_sequence: None,
            first_capture_monotonic_ns: None,
            last_capture_monotonic_ns: None,
            maximum_frame_gap_ns: 0,
            sample_rate_hz: None,
            channels: None,
            frame_duration_ms: None,
            samples_per_frame: None,
            frame_gaps: 0,
            duplicate_frames: 0,
            out_of_order_frames: 0,
            vad_events: 0,
            provider_attempts: 0,
            provider_accepted: 0,
            resets: 0,
            dropped_frames: 0,
            observer_errors: 0,
            terminated_early,
            positive_control,
            positive_started_monotonic_ns,
            pending_frames: 0,
            native_timing: None,
            native_next_adc: None,
            native_seen_adc: HashSet::new(),
        });
        Ok(())
    }

    pub fn complete_scored_interval(
        &self,
        ended_monotonic_ns: u64,
    ) -> Result<AecObservationEvidence, AecObservationCollectorError> {
        let mut state = self.lock_active()?;
        let ObservationPhase::Scored(scored) = &state.phase else {
            return Err(AecObservationCollectorError::InvalidLifecycle);
        };
        let valid_end = if let Some(native) = &scored.native_timing {
            native.host_uncertainty_ns().is_some_and(|uncertainty| {
                ended_monotonic_ns
                    .checked_sub(scored.started_monotonic_ns)
                    .is_some_and(|duration| {
                        duration.abs_diff(AEC_OBSERVATION_DURATION_NS) <= uncertainty
                    })
            })
        } else {
            ended_monotonic_ns == scored.ended_monotonic_ns
        };
        if !valid_end {
            return Err(AecObservationCollectorError::InvalidInterval);
        }
        let mut observer_errors = scored.observer_errors;
        if scored.pending_frames != 0 {
            observer_errors = observer_errors.saturating_add(1);
        }
        let first_capture_monotonic_ns = scored.first_capture_monotonic_ns.unwrap_or(0);
        let last_capture_monotonic_ns = scored.last_capture_monotonic_ns.unwrap_or(0);
        let host_allowance = if let Some(native) = &scored.native_timing {
            if native.adc_end.checked_sub(native.adc_start)
                != Some(AEC_OBSERVATION_FRAME_COUNT * NATIVE_ADC_SAMPLES_PER_FRAME)
            {
                observer_errors = observer_errors.saturating_add(1);
            }
            native
                .host_uncertainty_ns()
                .and_then(|value| value.checked_add(AEC_OBSERVATION_FRAME_DURATION_NS))
                .unwrap_or_else(|| {
                    observer_errors = observer_errors.saturating_add(1);
                    AEC_OBSERVATION_FRAME_DURATION_NS
                })
        } else {
            AEC_OBSERVATION_FRAME_DURATION_NS
        };
        if first_capture_monotonic_ns < scored.started_monotonic_ns
            || first_capture_monotonic_ns.saturating_sub(scored.started_monotonic_ns)
                > host_allowance
            || last_capture_monotonic_ns > ended_monotonic_ns
            || ended_monotonic_ns.saturating_sub(last_capture_monotonic_ns) > host_allowance
            || scored.maximum_frame_gap_ns > host_allowance
        {
            observer_errors = observer_errors.saturating_add(1);
        }
        let evidence = AecObservationEvidence {
            native_timing: scored.native_timing.clone(),
            observer_generation: self.observer_generation.to_string(),
            calibration_attempt_id: self.calibration_attempt_id.to_string(),
            challenge_id: self.challenge_id.to_string(),
            interval_id: scored.interval_id.to_string(),
            started_monotonic_ns: scored.started_monotonic_ns,
            ended_monotonic_ns,
            expected_frames: AEC_OBSERVATION_FRAME_COUNT,
            processed_frames: scored.processed_frames,
            stream_generation: scored.stream_generation.to_string(),
            sample_rate_hz: scored.sample_rate_hz.unwrap_or(0),
            channels: scored.channels.unwrap_or(0),
            frame_duration_ms: scored.frame_duration_ms.unwrap_or(0),
            samples_per_frame: scored.samples_per_frame.unwrap_or(0),
            first_frame_sequence: scored.first_frame_sequence.unwrap_or(0),
            last_frame_sequence: scored.last_frame_sequence.unwrap_or(0),
            first_capture_monotonic_ns,
            last_capture_monotonic_ns,
            maximum_frame_gap_ns: scored.maximum_frame_gap_ns,
            frame_gaps: scored.frame_gaps,
            duplicate_frames: scored.duplicate_frames,
            out_of_order_frames: scored.out_of_order_frames,
            vad_events_before: 0,
            vad_events_after: scored.vad_events,
            provider_attempts_before: 0,
            provider_attempts_after: scored.provider_attempts,
            provider_accepted_before: 0,
            provider_accepted_after: scored.provider_accepted,
            resets: scored.resets,
            dropped_frames: scored.dropped_frames,
            observer_errors,
            terminated_early: scored.terminated_early,
            positive_control: scored.positive_control.clone(),
        };
        state.phase = ObservationPhase::Complete(evidence.clone());
        Ok(evidence)
    }

    pub fn snapshot(&self) -> Option<AecObservationEvidence> {
        let state = self.state.lock().ok()?;
        match &state.phase {
            ObservationPhase::Complete(evidence) => Some(evidence.clone()),
            _ => None,
        }
    }

    pub(crate) fn scored_frame_count(&self) -> u64 {
        self.state.lock().map_or(0, |state| match &state.phase {
            ObservationPhase::Scored(scored) => scored.processed_frames,
            ObservationPhase::Complete(evidence) => evidence.processed_frames,
            _ => 0,
        })
    }

    pub fn record_dropped_frames(&self, observed_monotonic_ns: u64, count: u64) {
        self.with_scored_timestamp(observed_monotonic_ns, |scored| {
            scored.dropped_frames = scored.dropped_frames.saturating_add(count);
        });
    }

    pub fn record_observer_error(&self, observed_monotonic_ns: u64) {
        self.with_timestamp(
            observed_monotonic_ns,
            |positive| {
                positive.observer_errors = positive.observer_errors.saturating_add(1);
            },
            |scored| {
                scored.observer_errors = scored.observer_errors.saturating_add(1);
            },
        );
    }

    pub fn terminate_early(&self, observed_monotonic_ns: u64) {
        self.with_timestamp(
            observed_monotonic_ns,
            |positive| {
                positive.terminated_early = true;
            },
            |scored| {
                scored.terminated_early = true;
            },
        );
    }

    fn lock_active(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, CollectorState>, AecObservationCollectorError> {
        let state = self
            .state
            .lock()
            .map_err(|_| AecObservationCollectorError::InvalidLifecycle)?;
        if !state.activated {
            return Err(AecObservationCollectorError::GenerationMismatch);
        }
        Ok(state)
    }

    fn with_timestamp<P, S>(&self, observed_monotonic_ns: u64, positive: P, scored: S)
    where
        P: FnOnce(&mut PositiveState),
        S: FnOnce(&mut ScoredState),
    {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if !state.activated {
            return;
        }
        match &mut state.phase {
            ObservationPhase::Positive(current) => {
                if observed_monotonic_ns < current.started_monotonic_ns {
                    current.observer_errors = current.observer_errors.saturating_add(1);
                } else {
                    current.last_observed_monotonic_ns = current
                        .last_observed_monotonic_ns
                        .max(observed_monotonic_ns);
                    positive(current);
                }
            }
            ObservationPhase::Scored(current) => {
                if current.contains_host_timestamp(observed_monotonic_ns) {
                    scored(current);
                } else if current.native_timing.is_some() {
                    current.observer_errors = current.observer_errors.saturating_add(1);
                }
            }
            ObservationPhase::Idle
            | ObservationPhase::PositiveReady { .. }
            | ObservationPhase::Complete(_) => {}
        }
    }

    fn with_scored_timestamp<F>(&self, observed_monotonic_ns: u64, update: F)
    where
        F: FnOnce(&mut ScoredState),
    {
        self.with_timestamp(observed_monotonic_ns, |_| {}, update);
    }

    fn record_capture_frame(&self, direction: AudioDirection, frame: CompletedCaptureFrame) {
        self.record_capture(direction, frame, None);
    }

    fn record_capture(
        &self,
        direction: AudioDirection,
        frame: CompletedCaptureFrame,
        timing: Option<NativeCaptureTiming>,
    ) {
        if direction != AudioDirection::Microphone {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if !state.activated {
            return;
        }
        let ObservationPhase::Scored(scored) = &mut state.phase else {
            return;
        };
        if timing.is_none()
            && scored.native_timing.is_none()
            && !(scored.started_monotonic_ns..=scored.ended_monotonic_ns)
                .contains(&frame.capture_monotonic_ns)
        {
            return;
        }
        if frame.runtime_generation != scored.stream_generation
            || frame.sample_rate_hz != AEC_OBSERVATION_SAMPLE_RATE_HZ
            || frame.channels != AEC_OBSERVATION_CHANNELS
            || frame.frame_duration_ms != AEC_OBSERVATION_FRAME_DURATION_MS
            || frame.samples_per_frame != AEC_OBSERVATION_FRAME_SAMPLES
        {
            scored.observer_errors = scored.observer_errors.saturating_add(1);
            return;
        }
        match timing {
            Some(timing) => {
                if !Self::record_native_timing(scored, &frame, timing) {
                    return;
                }
            }
            None if scored.native_timing.is_some() => {
                scored.observer_errors = scored.observer_errors.saturating_add(1);
                return;
            }
            None => {}
        }
        if !scored.seen_sequences.insert(frame.sequence) {
            scored.duplicate_frames = scored.duplicate_frames.saturating_add(1);
            return;
        }
        if let Some(last) = scored.last_frame_sequence {
            if frame.sequence > last.saturating_add(1) {
                scored.frame_gaps = scored
                    .frame_gaps
                    .saturating_add(frame.sequence.saturating_sub(last).saturating_sub(1));
            } else if frame.sequence <= last {
                scored.out_of_order_frames = scored.out_of_order_frames.saturating_add(1);
            }
        }
        if let Some(last_capture) = scored.last_capture_monotonic_ns {
            if frame.capture_monotonic_ns < last_capture {
                scored.out_of_order_frames = scored.out_of_order_frames.saturating_add(1);
            } else {
                scored.maximum_frame_gap_ns = scored
                    .maximum_frame_gap_ns
                    .max(frame.capture_monotonic_ns - last_capture);
            }
        }
        scored.first_frame_sequence.get_or_insert(frame.sequence);
        scored.last_frame_sequence = Some(frame.sequence);
        scored
            .first_capture_monotonic_ns
            .get_or_insert(frame.capture_monotonic_ns);
        scored.last_capture_monotonic_ns = Some(frame.capture_monotonic_ns);
        scored.sample_rate_hz.get_or_insert(frame.sample_rate_hz);
        scored.channels.get_or_insert(frame.channels);
        scored
            .frame_duration_ms
            .get_or_insert(frame.frame_duration_ms);
        scored
            .samples_per_frame
            .get_or_insert(frame.samples_per_frame);
        scored.processed_frames = scored.processed_frames.saturating_add(1);
    }

    fn record_native_timing(
        scored: &mut ScoredState,
        frame: &CompletedCaptureFrame,
        timing: NativeCaptureTiming,
    ) -> bool {
        if !timing.graph.is_valid()
            || !matches!(timing.graph, AecGraphIdentity::Native { .. })
            || !matches!(timing.origin, NativeCaptureOrigin::Physical)
            || timing.adc_end.checked_sub(timing.adc_start) != Some(NATIVE_ADC_SAMPLES_PER_FRAME)
        {
            scored.observer_errors = scored.observer_errors.saturating_add(1);
            return false;
        }
        if let Some(native) = &scored.native_timing {
            if native.graph != timing.graph
                || native.capture_buffer_frames != timing.capture_buffer_frames
            {
                scored.observer_errors = scored.observer_errors.saturating_add(1);
                return false;
            }
        } else {
            if scored.processed_frames != 0 {
                scored.observer_errors = scored.observer_errors.saturating_add(1);
                return false;
            }
            let native = AecNativeObservationTiming {
                graph: timing.graph,
                adc_start: timing.adc_start,
                adc_end: timing.adc_start,
                capture_buffer_frames: timing.capture_buffer_frames,
                maximum_read_bracket_ns: timing.capture_read_bracket_ns,
                source_gaps: 0,
                source_duplicates: 0,
                source_reordered: 0,
            };
            if native.host_uncertainty_ns().is_none()
                || native
                    .adc_start
                    .checked_add(AEC_OBSERVATION_FRAME_COUNT * NATIVE_ADC_SAMPLES_PER_FRAME)
                    .is_none()
            {
                scored.observer_errors = scored.observer_errors.saturating_add(1);
                return false;
            }
            // Only the first physical phase selection may precede the scored window.
            if frame.capture_monotonic_ns < scored.started_monotonic_ns {
                return false;
            }
            scored.native_timing = Some(native);
        }
        let native = scored
            .native_timing
            .as_mut()
            .expect("native timing initialized");
        if native
            .host_uncertainty_ns()
            .and_then(|bound| bound.checked_sub(native.maximum_read_bracket_ns))
            .is_none_or(|buffer_bound| timing.capture_read_bracket_ns > buffer_bound)
        {
            scored.observer_errors = scored.observer_errors.saturating_add(1);
            return false;
        }
        if frame.capture_monotonic_ns < scored.started_monotonic_ns {
            scored.observer_errors = scored.observer_errors.saturating_add(1);
            return false;
        }
        if let Some(next) = scored.native_next_adc {
            if timing.adc_start > next {
                native.source_gaps = native.source_gaps.saturating_add(1);
                scored.frame_gaps = scored.frame_gaps.saturating_add(1);
                return false;
            }
            if timing.adc_start < next {
                if scored.native_seen_adc.contains(&timing.adc_start) {
                    native.source_duplicates = native.source_duplicates.saturating_add(1);
                    scored.duplicate_frames = scored.duplicate_frames.saturating_add(1);
                } else {
                    native.source_reordered = native.source_reordered.saturating_add(1);
                    scored.out_of_order_frames = scored.out_of_order_frames.saturating_add(1);
                }
                return false;
            }
        }
        if let Some(first_sequence) = scored.first_frame_sequence {
            let expected_sequence = timing
                .adc_start
                .checked_sub(native.adc_start)
                .and_then(|offset| offset.checked_div(NATIVE_ADC_SAMPLES_PER_FRAME))
                .and_then(|offset| first_sequence.checked_add(offset));
            if expected_sequence != Some(frame.sequence) {
                if scored.seen_sequences.contains(&frame.sequence) {
                    scored.duplicate_frames = scored.duplicate_frames.saturating_add(1);
                } else if expected_sequence.is_some_and(|expected| frame.sequence > expected) {
                    scored.frame_gaps = scored.frame_gaps.saturating_add(1);
                } else {
                    scored.out_of_order_frames = scored.out_of_order_frames.saturating_add(1);
                }
                return false;
            }
        }
        if scored
            .last_capture_monotonic_ns
            .is_some_and(|last| frame.capture_monotonic_ns < last)
        {
            scored.out_of_order_frames = scored.out_of_order_frames.saturating_add(1);
            return false;
        }
        scored.native_next_adc = Some(timing.adc_end);
        if scored.processed_frames >= AEC_OBSERVATION_FRAME_COUNT {
            return false;
        }
        scored.native_seen_adc.insert(timing.adc_start);
        native.adc_end = timing.adc_end;
        native.maximum_read_bracket_ns = native
            .maximum_read_bracket_ns
            .max(timing.capture_read_bracket_ns);
        if native.host_uncertainty_ns().is_none() {
            scored.observer_errors = scored.observer_errors.saturating_add(1);
            return false;
        }
        true
    }

    fn record_vad(&self, direction: AudioDirection, observed_ns: u64) {
        if direction != AudioDirection::Microphone {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if !state.activated {
            return;
        }
        match &mut state.phase {
            ObservationPhase::Positive(positive) => {
                if observed_ns < positive.started_monotonic_ns {
                    positive.observer_errors = positive.observer_errors.saturating_add(1);
                } else {
                    positive.last_observed_monotonic_ns =
                        positive.last_observed_monotonic_ns.max(observed_ns);
                    positive.speech_started_events =
                        positive.speech_started_events.saturating_add(1);
                }
            }
            ObservationPhase::Scored(scored) => {
                if scored.contains_host_timestamp(observed_ns) {
                    scored.vad_events = scored.vad_events.saturating_add(1);
                } else if observed_ns >= scored.positive_started_monotonic_ns
                    && observed_ns <= scored.positive_control.completed_monotonic_ns
                {
                    scored.positive_control.speech_started_events = scored
                        .positive_control
                        .speech_started_events
                        .saturating_add(1);
                } else if scored.native_timing.is_some() {
                    scored.observer_errors = scored.observer_errors.saturating_add(1);
                }
            }
            ObservationPhase::Idle
            | ObservationPhase::PositiveReady { .. }
            | ObservationPhase::Complete(_) => {}
        }
    }

    fn record_provider_attempt(&self, direction: AudioDirection, observed_ns: u64) {
        if direction != AudioDirection::Microphone {
            return;
        }
        self.with_timestamp(
            observed_ns,
            |positive| {
                positive.provider_submission_attempts =
                    positive.provider_submission_attempts.saturating_add(1);
            },
            |scored| {
                scored.provider_attempts = scored.provider_attempts.saturating_add(1);
            },
        );
    }

    fn record_provider_accepted(&self, direction: AudioDirection, observed_ns: u64) {
        if direction != AudioDirection::Microphone {
            return;
        }
        self.with_timestamp(
            observed_ns,
            |positive| {
                positive.provider_submissions_accepted =
                    positive.provider_submissions_accepted.saturating_add(1);
            },
            |scored| {
                scored.provider_accepted = scored.provider_accepted.saturating_add(1);
            },
        );
    }

    fn record_provider_origin(
        &self,
        direction: AudioDirection,
        origin: ProviderEffectOrigin,
        accepted: bool,
    ) {
        if direction != AudioDirection::Microphone {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if !state.activated {
            return;
        }
        let Some(capture_monotonic_ns) = origin.capture_monotonic_ns else {
            if matches!(state.phase, ObservationPhase::Scored(_)) {
                if let ObservationPhase::Scored(scored) = &mut state.phase {
                    scored.observer_errors = scored.observer_errors.saturating_add(1);
                }
            } else if accepted {
                drop(state);
                self.record_provider_accepted(direction, origin.observed_monotonic_ns);
            } else {
                drop(state);
                self.record_provider_attempt(direction, origin.observed_monotonic_ns);
            }
            return;
        };
        match &mut state.phase {
            ObservationPhase::Positive(positive) => {
                if capture_monotonic_ns < positive.started_monotonic_ns {
                    positive.observer_errors = positive.observer_errors.saturating_add(1);
                    return;
                }
                positive.last_observed_monotonic_ns = positive
                    .last_observed_monotonic_ns
                    .max(capture_monotonic_ns);
                increment_provider_positive(positive, accepted);
            }
            ObservationPhase::PositiveReady {
                evidence,
                started_monotonic_ns,
                ..
            } => {
                if capture_monotonic_ns >= *started_monotonic_ns
                    && capture_monotonic_ns <= evidence.completed_monotonic_ns
                {
                    increment_provider_evidence(evidence, accepted);
                }
            }
            ObservationPhase::Scored(scored) => {
                if origin.runtime_generation != scored.stream_generation {
                    scored.observer_errors = scored.observer_errors.saturating_add(1);
                } else if scored.contains_host_timestamp(capture_monotonic_ns) {
                    increment_provider_scored(scored, accepted);
                } else if capture_monotonic_ns >= scored.positive_started_monotonic_ns
                    && capture_monotonic_ns <= scored.positive_control.completed_monotonic_ns
                {
                    increment_provider_evidence(&mut scored.positive_control, accepted);
                } else if scored.native_timing.is_some() {
                    scored.observer_errors = scored.observer_errors.saturating_add(1);
                }
            }
            ObservationPhase::Idle | ObservationPhase::Complete(_) => {}
        }
    }

    fn record_pending_frames(
        &self,
        direction: AudioDirection,
        runtime_generation: Uuid,
        pending_frames: u64,
    ) {
        if direction != AudioDirection::Microphone {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let ObservationPhase::Scored(scored) = &mut state.phase else {
            return;
        };
        if runtime_generation != scored.stream_generation {
            scored.observer_errors = scored.observer_errors.saturating_add(1);
            return;
        }
        scored.pending_frames = pending_frames;
    }

    fn record_reset(&self, direction: AudioDirection) {
        if direction != AudioDirection::Microphone {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match &mut state.phase {
            ObservationPhase::Positive(positive) => {
                positive.resets = positive.resets.saturating_add(1);
            }
            ObservationPhase::Scored(scored) => {
                scored.resets = scored.resets.saturating_add(1);
            }
            _ => {}
        }
    }

    fn record_runtime_failure(&self, direction: AudioDirection) {
        if direction != AudioDirection::Microphone {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match &mut state.phase {
            ObservationPhase::Positive(positive) => {
                positive.observer_errors = positive.observer_errors.saturating_add(1);
            }
            ObservationPhase::Scored(scored) => {
                scored.observer_errors = scored.observer_errors.saturating_add(1);
            }
            _ => {}
        }
    }

    fn record_generation_restart(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match &mut state.phase {
            ObservationPhase::Positive(positive) => {
                positive.observer_errors = positive.observer_errors.saturating_add(1);
                positive.terminated_early = true;
            }
            ObservationPhase::PositiveReady {
                evidence,
                terminated_early,
                ..
            } => {
                evidence.observer_errors = evidence.observer_errors.saturating_add(1);
                *terminated_early = true;
            }
            ObservationPhase::Scored(scored) => {
                scored.observer_errors = scored.observer_errors.saturating_add(1);
                scored.terminated_early = true;
            }
            _ => {}
        }
    }
}

fn increment_provider_positive(positive: &mut PositiveState, accepted: bool) {
    if accepted {
        positive.provider_submissions_accepted =
            positive.provider_submissions_accepted.saturating_add(1);
    } else {
        positive.provider_submission_attempts =
            positive.provider_submission_attempts.saturating_add(1);
    }
}

fn increment_provider_evidence(evidence: &mut AecPositiveControl, accepted: bool) {
    if accepted {
        evidence.provider_submissions_accepted =
            evidence.provider_submissions_accepted.saturating_add(1);
    } else {
        evidence.provider_submission_attempts =
            evidence.provider_submission_attempts.saturating_add(1);
    }
}

fn increment_provider_scored(scored: &mut ScoredState, accepted: bool) {
    if accepted {
        scored.provider_accepted = scored.provider_accepted.saturating_add(1);
    } else {
        scored.provider_attempts = scored.provider_attempts.saturating_add(1);
    }
}

impl DuplexRuntimeObserver for AecRuntimeObserver {
    fn observe(&self, event: DuplexRuntimeEvent) {
        match event {
            DuplexRuntimeEvent::SpeechStarted {
                direction,
                capture_monotonic_ns,
                ..
            } => self.record_vad(direction, capture_monotonic_ns),
            DuplexRuntimeEvent::ProviderError { direction, .. } => {
                self.record_runtime_failure(direction);
            }
            DuplexRuntimeEvent::UtteranceTerminalOutcome {
                direction,
                outcome: TerminalOutcome::Cancelled | TerminalOutcome::Dropped,
                ..
            } => {
                if direction == AudioDirection::Microphone {
                    self.record_generation_restart();
                }
            }
            DuplexRuntimeEvent::GenerationRestart { .. } => self.record_generation_restart(),
            DuplexRuntimeEvent::TranscriptFinal { .. }
            | DuplexRuntimeEvent::TranslationFinal { .. }
            | DuplexRuntimeEvent::AudioFrame { .. }
            | DuplexRuntimeEvent::FirstAudioExpired { .. }
            | DuplexRuntimeEvent::ProviderLatency { .. }
            | DuplexRuntimeEvent::UtteranceTerminalOutcome { .. }
            | DuplexRuntimeEvent::UtteranceTerminal { .. } => {}
        }
    }

    fn capture_frame_processed(&self, direction: AudioDirection, frame: CompletedCaptureFrame) {
        self.record_capture_frame(direction, frame);
    }

    fn native_capture_frame_processed(
        &self,
        direction: AudioDirection,
        frame: CompletedCaptureFrame,
        timing: NativeCaptureTiming,
    ) {
        self.record_capture(direction, frame, Some(timing));
    }

    fn capture_frames_pending(
        &self,
        direction: AudioDirection,
        runtime_generation: Uuid,
        pending_frames: u64,
    ) {
        self.record_pending_frames(direction, runtime_generation, pending_frames);
    }

    fn provider_submission_attempted(&self, direction: AudioDirection, observed_monotonic_ns: u64) {
        self.record_provider_attempt(direction, observed_monotonic_ns);
    }

    fn provider_submission_accepted(&self, direction: AudioDirection, observed_monotonic_ns: u64) {
        self.record_provider_accepted(direction, observed_monotonic_ns);
    }

    fn provider_submission_attempted_for_origin(
        &self,
        direction: AudioDirection,
        origin: ProviderEffectOrigin,
    ) {
        self.record_provider_origin(direction, origin, false);
    }

    fn provider_submission_accepted_for_origin(
        &self,
        direction: AudioDirection,
        origin: ProviderEffectOrigin,
    ) {
        self.record_provider_origin(direction, origin, true);
    }

    fn reset_direction(&self, direction: AudioDirection) {
        self.record_reset(direction);
    }
}

pub struct DuplexRuntimeObserverFanout {
    primary: Arc<dyn DuplexRuntimeObserver>,
    collector: Arc<AecRuntimeObserver>,
}

impl DuplexRuntimeObserverFanout {
    pub fn new(
        primary: Arc<dyn DuplexRuntimeObserver>,
        collector: Arc<AecRuntimeObserver>,
    ) -> Self {
        Self { primary, collector }
    }
}

impl DuplexRuntimeObserver for DuplexRuntimeObserverFanout {
    fn observe(&self, event: DuplexRuntimeEvent) {
        self.primary.observe(event);
        self.collector.observe(event);
    }

    fn capture_frame_processed(&self, direction: AudioDirection, frame: CompletedCaptureFrame) {
        self.primary.capture_frame_processed(direction, frame);
        self.collector.capture_frame_processed(direction, frame);
    }

    fn native_capture_frame_processed(
        &self,
        direction: AudioDirection,
        frame: CompletedCaptureFrame,
        timing: NativeCaptureTiming,
    ) {
        self.primary
            .native_capture_frame_processed(direction, frame, timing.clone());
        self.collector
            .native_capture_frame_processed(direction, frame, timing);
    }

    fn capture_frames_pending(
        &self,
        direction: AudioDirection,
        runtime_generation: Uuid,
        pending_frames: u64,
    ) {
        self.primary
            .capture_frames_pending(direction, runtime_generation, pending_frames);
        self.collector
            .capture_frames_pending(direction, runtime_generation, pending_frames);
    }

    fn provider_submission_attempted(&self, direction: AudioDirection, observed_monotonic_ns: u64) {
        self.primary
            .provider_submission_attempted(direction, observed_monotonic_ns);
        self.collector
            .provider_submission_attempted(direction, observed_monotonic_ns);
    }

    fn provider_submission_accepted(&self, direction: AudioDirection, observed_monotonic_ns: u64) {
        self.primary
            .provider_submission_accepted(direction, observed_monotonic_ns);
        self.collector
            .provider_submission_accepted(direction, observed_monotonic_ns);
    }

    fn provider_submission_attempted_for_origin(
        &self,
        direction: AudioDirection,
        origin: ProviderEffectOrigin,
    ) {
        self.primary
            .provider_submission_attempted_for_origin(direction, origin);
        self.collector
            .provider_submission_attempted_for_origin(direction, origin);
    }

    fn provider_submission_accepted_for_origin(
        &self,
        direction: AudioDirection,
        origin: ProviderEffectOrigin,
    ) {
        self.primary
            .provider_submission_accepted_for_origin(direction, origin);
        self.collector
            .provider_submission_accepted_for_origin(direction, origin);
    }

    fn reset_direction(&self, direction: AudioDirection) {
        self.primary.reset_direction(direction);
        self.collector.reset_direction(direction);
    }

    fn requested_mode(&self, direction: AudioDirection) -> Option<TranslationMode> {
        self.primary.requested_mode(direction)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use translator_audio::{AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT};
    use translator_core::{AudioDirection, TranslationMode};
    use uuid::Uuid;

    use crate::{DuplexRuntimeEvent, DuplexRuntimeObserver, SafeProviderErrorCode};

    use super::*;

    #[derive(Default)]
    struct PrimaryObserver {
        public_events: Mutex<Vec<DuplexRuntimeEvent>>,
        internal_events: Mutex<Vec<&'static str>>,
        resets: Mutex<Vec<AudioDirection>>,
    }

    impl DuplexRuntimeObserver for PrimaryObserver {
        fn observe(&self, event: DuplexRuntimeEvent) {
            self.public_events.lock().unwrap().push(event);
        }

        fn capture_frame_processed(
            &self,
            _direction: AudioDirection,
            _frame: CompletedCaptureFrame,
        ) {
            self.internal_events.lock().unwrap().push("capture");
        }

        fn provider_submission_attempted(
            &self,
            _direction: AudioDirection,
            _observed_monotonic_ns: u64,
        ) {
            self.internal_events.lock().unwrap().push("attempted");
        }

        fn provider_submission_accepted(
            &self,
            _direction: AudioDirection,
            _observed_monotonic_ns: u64,
        ) {
            self.internal_events.lock().unwrap().push("accepted");
        }

        fn reset_direction(&self, direction: AudioDirection) {
            self.resets.lock().unwrap().push(direction);
        }

        fn requested_mode(&self, _direction: AudioDirection) -> Option<TranslationMode> {
            Some(TranslationMode::Balanced)
        }
    }

    fn observer() -> AecRuntimeObserver {
        let (observer, activation) =
            AecRuntimeObserver::new(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        observer.activate_generation(activation).unwrap();
        observer
    }

    fn complete_positive_control(observer: &AecRuntimeObserver, started_ns: u64) {
        observer.start_positive_control(started_ns).unwrap();
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: started_ns + 1,
        });
        observer.provider_submission_attempted(AudioDirection::Microphone, started_ns + 2);
        observer.provider_submission_accepted(AudioDirection::Microphone, started_ns + 3);
        observer.complete_positive_control(started_ns + 4).unwrap();
    }

    fn completed_frame(
        sequence: u64,
        capture_monotonic_ns: u64,
        runtime_generation: Uuid,
    ) -> CompletedCaptureFrame {
        CompletedCaptureFrame {
            sequence,
            capture_monotonic_ns,
            sample_rate_hz: AEC_OBSERVATION_SAMPLE_RATE_HZ,
            channels: AEC_OBSERVATION_CHANNELS,
            frame_duration_ms: AEC_OBSERVATION_FRAME_DURATION_MS,
            samples_per_frame: AEC_OBSERVATION_FRAME_SAMPLES,
            runtime_generation,
        }
    }

    #[test]
    fn native_source_continuity_does_not_fabricate_uniform_host_delivery() {
        let observer = observer();
        complete_positive_control(&observer, 100);
        let generation = Uuid::new_v4();
        observer
            .start_scored_interval(Uuid::new_v4(), 104, generation)
            .unwrap();
        for index in 0..AEC_OBSERVATION_FRAME_COUNT {
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(
                    index,
                    104 + index * AEC_OBSERVATION_FRAME_DURATION_NS + (index % 2) * 10_000,
                    generation,
                ),
                crate::translation_runtime::NativeCaptureTiming {
                    graph: translator_audio::AecGraphIdentity::Native {
                        session_id: 1,
                        generation: 1,
                        physical_device_id: "alsa-hw:0,0".into(),
                        dsp_config_id: "installed-spa".into(),
                    },
                    adc_start: 96_000 + index * 960,
                    adc_end: 96_000 + (index + 1) * 960,
                    capture_buffer_frames: 3840,
                    capture_read_bracket_ns: 100_000,
                    origin: translator_audio::NativeCaptureOrigin::Physical,
                },
            );
        }
        let evidence = observer
            .complete_scored_interval(104 + AEC_OBSERVATION_DURATION_NS)
            .unwrap();
        assert_eq!(evidence.processed_frames, 3000);
        assert_eq!(evidence.observer_errors, 0);
        assert_eq!(evidence.maximum_frame_gap_ns, 20_010_000);
        assert_eq!(
            evidence.native_timing.as_ref().unwrap().adc_end
                - evidence.native_timing.as_ref().unwrap().adc_start,
            2_880_000
        );
    }

    fn native_timing(index: u64) -> NativeCaptureTiming {
        NativeCaptureTiming {
            graph: AecGraphIdentity::Native {
                session_id: 1,
                generation: 7,
                physical_device_id: "alsa-hw:0,0".into(),
                dsp_config_id: "installed-spa".into(),
            },
            adc_start: 96_000 + index * NATIVE_ADC_SAMPLES_PER_FRAME,
            adc_end: 96_000 + (index + 1) * NATIVE_ADC_SAMPLES_PER_FRAME,
            capture_buffer_frames: 3840,
            capture_read_bracket_ns: 100_000,
            origin: NativeCaptureOrigin::Physical,
        }
    }

    fn native_interval() -> (AecRuntimeObserver, Uuid, u64) {
        let observer = observer();
        let started_ns = 10_000_000_000;
        complete_positive_control(&observer, started_ns - 4);
        let generation = Uuid::new_v4();
        observer
            .start_scored_interval(Uuid::new_v4(), started_ns, generation)
            .unwrap();
        (observer, generation, started_ns)
    }

    fn record_native_window(observer: &AecRuntimeObserver, generation: Uuid, started_ns: u64) {
        for index in 0..AEC_OBSERVATION_FRAME_COUNT {
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(
                    index,
                    started_ns + index * AEC_OBSERVATION_FRAME_DURATION_NS,
                    generation,
                ),
                native_timing(index),
            );
        }
    }

    #[test]
    fn native_shape_and_identity_faults_are_not_counted() {
        let mutations: [fn(&mut NativeCaptureTiming); 8] = [
            |timing| {
                if let AecGraphIdentity::Native { generation, .. } = &mut timing.graph {
                    *generation += 1;
                }
            },
            |timing| {
                if let AecGraphIdentity::Native { session_id, .. } = &mut timing.graph {
                    *session_id += 1;
                }
            },
            |timing| {
                if let AecGraphIdentity::Native { dsp_config_id, .. } = &mut timing.graph {
                    *dsp_config_id = "changed-config".into();
                }
            },
            |timing| timing.capture_buffer_frames = 1920,
            |timing| timing.capture_read_bracket_ns = 80_000_001,
            |timing| timing.adc_end -= 1,
            |timing| {
                timing.origin = NativeCaptureOrigin::InjectedPositive {
                    fixture_sha256: "test-fixture".into(),
                };
            },
            |timing| {
                if let AecGraphIdentity::Native { generation, .. } = &mut timing.graph {
                    *generation = 0;
                }
            },
        ];
        for mutate in mutations {
            let (observer, generation, started_ns) = native_interval();
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(0, started_ns, generation),
                native_timing(0),
            );
            let mut timing = native_timing(1);
            mutate(&mut timing);
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(
                    1,
                    started_ns + AEC_OBSERVATION_FRAME_DURATION_NS,
                    generation,
                ),
                timing,
            );
            assert_eq!(observer.scored_frame_count(), 1);
            let evidence = observer
                .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
                .unwrap();
            assert!(evidence.observer_errors > 0);
            assert_eq!(evidence.native_timing.unwrap().adc_end, 96_960);
        }
    }

    #[test]
    fn native_first_frame_cannot_establish_unknown_graph_or_unbounded_geometry() {
        let mutations: [fn(&mut NativeCaptureTiming); 8] = [
            |timing| {
                timing.graph = AecGraphIdentity::PulseModule {
                    module_id: 1,
                    source_id: 2,
                    sink_id: 3,
                }
            },
            |timing| {
                if let AecGraphIdentity::Native { session_id, .. } = &mut timing.graph {
                    *session_id = 0;
                }
            },
            |timing| {
                if let AecGraphIdentity::Native {
                    physical_device_id, ..
                } = &mut timing.graph
                {
                    *physical_device_id = "unbound-device".into();
                }
            },
            |timing| timing.capture_buffer_frames = 1919,
            |timing| timing.capture_buffer_frames = 48_001,
            |timing| timing.capture_read_bracket_ns = u64::MAX,
            |timing| {
                timing.adc_start = u64::MAX - 960;
                timing.adc_end = u64::MAX;
            },
            |timing| timing.adc_end = timing.adc_start - 1,
        ];
        for mutate in mutations {
            let (observer, generation, started_ns) = native_interval();
            let mut timing = native_timing(0);
            mutate(&mut timing);
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(0, started_ns, generation),
                timing,
            );
            assert_eq!(observer.scored_frame_count(), 0);
            let evidence = observer
                .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
                .unwrap();
            assert!(evidence.observer_errors > 0);
        }
    }

    #[test]
    fn native_adc_gaps_duplicates_and_reordering_keep_source_diagnostics() {
        for (adc_start, expected) in [
            (97_920, (1, 0, 0)),
            (96_000, (0, 1, 0)),
            (96_480, (0, 0, 1)),
        ] {
            let (observer, generation, started_ns) = native_interval();
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(0, started_ns, generation),
                native_timing(0),
            );
            let mut timing = native_timing(1);
            timing.adc_start = adc_start;
            timing.adc_end = adc_start + NATIVE_ADC_SAMPLES_PER_FRAME;
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(
                    1,
                    started_ns + AEC_OBSERVATION_FRAME_DURATION_NS,
                    generation,
                ),
                timing,
            );
            assert_eq!(observer.scored_frame_count(), 1);
            let evidence = observer
                .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
                .unwrap();
            let native = evidence.native_timing.unwrap();
            assert_eq!(
                (
                    native.source_gaps,
                    native.source_duplicates,
                    native.source_reordered
                ),
                expected
            );
            assert_eq!(
                (
                    evidence.frame_gaps,
                    evidence.duplicate_frames,
                    evidence.out_of_order_frames
                ),
                expected
            );
        }
    }

    #[test]
    fn native_sequence_host_order_and_runtime_faults_do_not_advance_adc_membership() {
        let mutations: [fn(&mut CompletedCaptureFrame); 5] = [
            |frame| frame.sequence = 0,
            |frame| frame.sequence += 1,
            |frame| frame.capture_monotonic_ns -= AEC_OBSERVATION_FRAME_DURATION_NS + 1,
            |frame| frame.runtime_generation = Uuid::new_v4(),
            |frame| frame.samples_per_frame += 1,
        ];
        for mutate in mutations {
            let (observer, generation, started_ns) = native_interval();
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(0, started_ns, generation),
                native_timing(0),
            );
            let mut frame = completed_frame(
                1,
                started_ns + AEC_OBSERVATION_FRAME_DURATION_NS,
                generation,
            );
            mutate(&mut frame);
            observer.native_capture_frame_processed(
                AudioDirection::Microphone,
                frame,
                native_timing(1),
            );
            assert_eq!(observer.scored_frame_count(), 1);
            let evidence = observer
                .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
                .unwrap();
            assert!(
                evidence.observer_errors
                    + evidence.frame_gaps
                    + evidence.duplicate_frames
                    + evidence.out_of_order_frames
                    > 0
            );
            assert_eq!(evidence.native_timing.unwrap().adc_end, 96_960);
        }
    }

    #[test]
    fn only_initial_physical_phase_selection_may_precede_scored_start() {
        let (observer, generation, started_ns) = native_interval();
        observer.native_capture_frame_processed(
            AudioDirection::Microphone,
            completed_frame(99, started_ns - 1, generation),
            native_timing(0),
        );
        assert_eq!(observer.scored_frame_count(), 0);
        record_native_window(&observer, generation, started_ns);
        observer.native_capture_frame_processed(
            AudioDirection::Microphone,
            completed_frame(3000, started_ns - 1, generation),
            native_timing(3000),
        );
        let evidence = observer
            .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
            .unwrap();
        assert_eq!(evidence.processed_frames, 3000);
        assert_eq!(evidence.observer_errors, 1);
    }

    #[test]
    fn native_window_freezes_exact_membership_but_retains_post_window_faults() {
        for fault in [false, true] {
            let (observer, generation, started_ns) = native_interval();
            record_native_window(&observer, generation, started_ns);
            for index in 3000..3002 {
                observer.native_capture_frame_processed(
                    AudioDirection::Microphone,
                    completed_frame(
                        index,
                        started_ns + index * AEC_OBSERVATION_FRAME_DURATION_NS,
                        generation,
                    ),
                    native_timing(index),
                );
            }
            if fault {
                observer.native_capture_frame_processed(
                    AudioDirection::Microphone,
                    completed_frame(3002, started_ns + AEC_OBSERVATION_DURATION_NS, generation),
                    native_timing(100),
                );
            }
            let evidence = observer
                .complete_scored_interval(
                    started_ns + AEC_OBSERVATION_DURATION_NS + AEC_OBSERVATION_FRAME_DURATION_NS,
                )
                .unwrap();
            assert_eq!(evidence.processed_frames, 3000);
            assert_eq!(observer.scored_frame_count(), 3000);
            assert_eq!(evidence.last_frame_sequence, 2999);
            assert_eq!(
                evidence.last_capture_monotonic_ns,
                started_ns + 2999 * AEC_OBSERVATION_FRAME_DURATION_NS
            );
            assert_eq!(
                evidence.native_timing.as_ref().unwrap().adc_end,
                96_000 + 3000 * 960
            );
            assert_eq!(evidence.observer_errors, 0);
            assert_eq!(evidence.duplicate_frames, u64::from(fault));
            assert_eq!(
                evidence.native_timing.unwrap().source_duplicates,
                u64::from(fault)
            );
        }
    }

    #[test]
    fn native_actual_completion_uses_derived_bound_while_host_only_stays_exact() {
        let (observer, generation, started_ns) = native_interval();
        let delay = 30_000_000;
        record_native_window(&observer, generation, started_ns + delay);
        assert_eq!(
            observer
                .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS + 80_100_001),
            Err(AecObservationCollectorError::InvalidInterval)
        );
        let evidence = observer
            .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS + delay)
            .unwrap();
        assert_eq!(evidence.observer_errors, 0);
        assert_eq!(evidence.first_capture_monotonic_ns, started_ns + delay);
        assert_eq!(
            evidence.last_capture_monotonic_ns,
            started_ns + delay + 2999 * AEC_OBSERVATION_FRAME_DURATION_NS
        );
        let (observer, _, started_ns) = native_interval();
        assert_eq!(
            observer.complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS + 1),
            Err(AecObservationCollectorError::InvalidInterval)
        );
    }

    #[test]
    fn native_host_burst_is_not_sixty_seconds_and_end_cannot_precede_last_delivery() {
        for burst in [false, true] {
            let (observer, generation, started_ns) = native_interval();
            for index in 0..3000 {
                observer.native_capture_frame_processed(
                    AudioDirection::Microphone,
                    completed_frame(
                        index,
                        if burst {
                            started_ns
                        } else {
                            started_ns + 30_000_000 + index * AEC_OBSERVATION_FRAME_DURATION_NS
                        },
                        generation,
                    ),
                    native_timing(index),
                );
            }
            let evidence = observer
                .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
                .unwrap();
            assert_eq!(evidence.processed_frames, 3000);
            assert!(evidence.observer_errors > 0);
        }
    }

    #[test]
    fn native_late_vad_and_provider_effects_are_conservatively_retained() {
        let (observer, generation, started_ns) = native_interval();
        record_native_window(&observer, generation, started_ns);
        let nominal_end = started_ns + AEC_OBSERVATION_DURATION_NS;
        let uncertainty = 80_100_000;
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: nominal_end + uncertainty,
        });
        observer.provider_submission_attempted_for_origin(
            AudioDirection::Microphone,
            ProviderEffectOrigin {
                runtime_generation: generation,
                capture_monotonic_ns: Some(nominal_end + uncertainty),
                observed_monotonic_ns: nominal_end + uncertainty,
            },
        );
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: nominal_end + uncertainty + 1,
        });
        observer.provider_submission_accepted_for_origin(
            AudioDirection::Microphone,
            ProviderEffectOrigin {
                runtime_generation: generation,
                capture_monotonic_ns: Some(nominal_end + uncertainty + 1),
                observed_monotonic_ns: nominal_end + uncertainty + 1,
            },
        );
        let evidence = observer.complete_scored_interval(nominal_end).unwrap();
        assert_eq!(evidence.vad_events_after, 1);
        assert_eq!(evidence.provider_attempts_after, 1);
        assert_eq!(evidence.provider_accepted_after, 0);
        assert_eq!(evidence.observer_errors, 2);
    }

    #[test]
    fn native_early_uncertain_and_unattributed_events_cannot_be_hidden_as_positive_control() {
        let (observer, generation, started_ns) = native_interval();
        record_native_window(&observer, generation, started_ns);
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: started_ns - 1,
        });
        observer.provider_submission_attempted_for_origin(
            AudioDirection::Microphone,
            ProviderEffectOrigin {
                runtime_generation: generation,
                capture_monotonic_ns: Some(started_ns - 1),
                observed_monotonic_ns: started_ns + 1,
            },
        );
        observer.provider_submission_accepted_for_origin(
            AudioDirection::Microphone,
            ProviderEffectOrigin {
                runtime_generation: generation,
                capture_monotonic_ns: None,
                observed_monotonic_ns: started_ns + 1,
            },
        );
        observer.provider_submission_attempted_for_origin(
            AudioDirection::Microphone,
            ProviderEffectOrigin {
                runtime_generation: Uuid::new_v4(),
                capture_monotonic_ns: Some(started_ns),
                observed_monotonic_ns: started_ns,
            },
        );
        let evidence = observer
            .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
            .unwrap();
        assert_eq!(evidence.vad_events_after, 1);
        assert_eq!(evidence.provider_attempts_after, 1);
        assert_eq!(evidence.provider_accepted_after, 0);
        assert_eq!(evidence.positive_control.speech_started_events, 1);
        assert_eq!(evidence.positive_control.provider_submission_attempts, 1);
        assert_eq!(evidence.observer_errors, 2);
    }

    #[test]
    fn native_and_host_clock_membership_cannot_be_mixed() {
        for native_first in [true, false] {
            let (observer, generation, started_ns) = native_interval();
            let first = completed_frame(0, started_ns, generation);
            let second = completed_frame(
                1,
                started_ns + AEC_OBSERVATION_FRAME_DURATION_NS,
                generation,
            );
            if native_first {
                observer.native_capture_frame_processed(
                    AudioDirection::Microphone,
                    first,
                    native_timing(0),
                );
                observer.capture_frame_processed(AudioDirection::Microphone, second);
            } else {
                observer.capture_frame_processed(AudioDirection::Microphone, first);
                observer.native_capture_frame_processed(
                    AudioDirection::Microphone,
                    second,
                    native_timing(1),
                );
            }
            assert_eq!(observer.scored_frame_count(), 1);
            let evidence = observer
                .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
                .unwrap();
            assert!(evidence.observer_errors > 0);
        }
    }

    #[test]
    fn native_fanout_records_one_callback_per_observer() {
        let (collector, generation, started_ns) = native_interval();
        let collector = Arc::new(collector);
        let primary = Arc::new(PrimaryObserver::default());
        let fanout = DuplexRuntimeObserverFanout::new(primary.clone(), collector.clone());
        fanout.native_capture_frame_processed(
            AudioDirection::Microphone,
            completed_frame(0, started_ns, generation),
            native_timing(0),
        );
        assert_eq!(
            primary.internal_events.lock().unwrap().as_slice(),
            &["capture"]
        );
        assert_eq!(collector.scored_frame_count(), 1);
    }

    #[test]
    fn complete_separate_intervals_produce_exact_immutable_evidence() {
        let observer = observer();
        complete_positive_control(&observer, 100);
        let interval_id = Uuid::new_v4();
        let runtime_generation = Uuid::new_v4();
        let started_ns = 104;
        observer
            .start_scored_interval(interval_id, started_ns, runtime_generation)
            .unwrap();
        let first_sequence = 10_000;
        for offset in 0..AEC_OBSERVATION_FRAME_COUNT {
            observer.capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(
                    first_sequence + offset,
                    started_ns + offset * AEC_OBSERVATION_FRAME_DURATION_NS,
                    runtime_generation,
                ),
            );
        }
        let ended_ns = started_ns + AEC_OBSERVATION_DURATION_NS;
        let evidence = observer.complete_scored_interval(ended_ns).unwrap();

        assert_eq!(evidence.interval_id, interval_id.to_string());
        assert_eq!(evidence.processed_frames, AEC_OBSERVATION_FRAME_COUNT);
        assert_eq!(evidence.first_frame_sequence, first_sequence);
        assert_eq!(
            evidence.last_frame_sequence,
            first_sequence + AEC_OBSERVATION_FRAME_COUNT - 1
        );
        assert_eq!(evidence.frame_gaps, 0);
        assert_eq!(evidence.duplicate_frames, 0);
        assert_eq!(evidence.out_of_order_frames, 0);
        assert_eq!(observer.snapshot(), Some(evidence.clone()));

        observer.provider_submission_attempted(AudioDirection::Microphone, ended_ns + 1);
        observer.reset_direction(AudioDirection::Microphone);
        assert_eq!(observer.snapshot(), Some(evidence));
    }

    #[test]
    fn compressed_frame_burst_does_not_prove_sixty_seconds_of_runtime_coverage() {
        let observer = observer();
        complete_positive_control(&observer, 100);
        let started_ns = 104;
        let runtime_generation = Uuid::new_v4();
        observer
            .start_scored_interval(Uuid::new_v4(), started_ns, runtime_generation)
            .unwrap();
        for sequence in 0..AEC_OBSERVATION_FRAME_COUNT {
            observer.capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(sequence, started_ns + 1, runtime_generation),
            );
        }

        let evidence = observer
            .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
            .unwrap();
        assert!(evidence.observer_errors > 0 || evidence.terminated_early);
    }

    #[test]
    fn wrong_runtime_generation_or_pcm_format_fails_closed() {
        let mutations: [fn(&mut CompletedCaptureFrame); 2] = [
            |frame: &mut CompletedCaptureFrame| frame.runtime_generation = Uuid::new_v4(),
            |frame: &mut CompletedCaptureFrame| frame.samples_per_frame += 1,
        ];
        for mutate in mutations {
            let observer = observer();
            complete_positive_control(&observer, 100);
            let started_ns = 104;
            let runtime_generation = Uuid::new_v4();
            observer
                .start_scored_interval(Uuid::new_v4(), started_ns, runtime_generation)
                .unwrap();
            let mut frame = completed_frame(1, started_ns, runtime_generation);
            mutate(&mut frame);
            observer.capture_frame_processed(AudioDirection::Microphone, frame);
            let evidence = observer
                .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
                .unwrap();
            assert!(evidence.observer_errors > 0);
            assert_eq!(evidence.processed_frames, 0);
        }
    }

    #[test]
    fn sequence_and_runtime_failures_are_retained_as_fail_closed_evidence() {
        let observer = observer();
        complete_positive_control(&observer, 1_000);
        let started_ns = 1_004;
        let runtime_generation = Uuid::new_v4();
        observer
            .start_scored_interval(Uuid::new_v4(), started_ns, runtime_generation)
            .unwrap();
        for (sequence, observed_ns) in [(0, 1_005), (2, 1_006), (2, 1_007), (1, 1_008)] {
            observer.capture_frame_processed(
                AudioDirection::Microphone,
                completed_frame(sequence, observed_ns, runtime_generation),
            );
        }
        observer.observe(DuplexRuntimeEvent::ProviderError {
            direction: AudioDirection::Microphone,
            utterance_id: None,
            code: SafeProviderErrorCode::ProviderUnavailable,
            retryable: true,
        });
        observer.record_dropped_frames(1_009, 2);
        observer.reset_direction(AudioDirection::Microphone);
        observer.terminate_early(1_010);

        let evidence = observer
            .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
            .unwrap();
        assert_eq!(evidence.processed_frames, 3);
        assert_eq!(evidence.frame_gaps, 1);
        assert_eq!(evidence.duplicate_frames, 1);
        assert_eq!(evidence.out_of_order_frames, 1);
        assert_eq!(evidence.resets, 1);
        assert_eq!(evidence.dropped_frames, 2);
        assert!(evidence.observer_errors > 0);
        assert!(evidence.terminated_early);
    }

    #[test]
    fn speaker_and_inactive_events_do_not_contaminate_intervals() {
        let observer = observer();
        observer.provider_submission_attempted(AudioDirection::Microphone, 1);
        complete_positive_control(&observer, 100);
        let started_ns = 104;
        let runtime_generation = Uuid::new_v4();
        observer
            .start_scored_interval(Uuid::new_v4(), started_ns, runtime_generation)
            .unwrap();
        observer.capture_frame_processed(
            AudioDirection::Speaker,
            completed_frame(0, started_ns + 1, runtime_generation),
        );
        observer.provider_submission_attempted(AudioDirection::Speaker, started_ns + 2);
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Speaker,
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: started_ns + 3,
        });
        observer.capture_frame_processed(
            AudioDirection::Microphone,
            completed_frame(
                0,
                started_ns + AEC_OBSERVATION_DURATION_NS + 1,
                runtime_generation,
            ),
        );

        let evidence = observer
            .complete_scored_interval(started_ns + AEC_OBSERVATION_DURATION_NS)
            .unwrap();
        assert_eq!(evidence.processed_frames, 0);
        assert_eq!(evidence.provider_attempts_after, 0);
        assert_eq!(evidence.vad_events_after, 0);
        assert_eq!(evidence.observer_errors, 1);
    }

    #[test]
    fn fanout_preserves_primary_public_internal_reset_and_mode_behavior() {
        let collector = Arc::new(observer());
        collector.start_positive_control(100).unwrap();
        let primary = Arc::new(PrimaryObserver::default());
        let fanout = DuplexRuntimeObserverFanout::new(primary.clone(), collector.clone());
        let public_event = DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id: Uuid::new_v4(),
            capture_monotonic_ns: 101,
        };

        fanout.observe(public_event);
        fanout.capture_frame_processed(
            AudioDirection::Microphone,
            completed_frame(0, 102, Uuid::new_v4()),
        );
        fanout.provider_submission_attempted(AudioDirection::Microphone, 103);
        fanout.provider_submission_accepted(AudioDirection::Microphone, 104);
        fanout.reset_direction(AudioDirection::Speaker);

        assert_eq!(
            primary.public_events.lock().unwrap().as_slice(),
            &[public_event]
        );
        assert_eq!(
            primary.internal_events.lock().unwrap().as_slice(),
            &["capture", "attempted", "accepted"]
        );
        assert_eq!(
            primary.resets.lock().unwrap().as_slice(),
            &[AudioDirection::Speaker]
        );
        assert_eq!(
            fanout.requested_mode(AudioDirection::Microphone),
            Some(TranslationMode::Balanced)
        );
    }

    #[test]
    fn generation_restart_and_invalid_lifecycle_fail_closed() {
        let generation = Uuid::new_v4();
        let (observer, activation) =
            AecRuntimeObserver::new(generation, Uuid::new_v4(), Uuid::new_v4());
        let (_, stale_activation) =
            AecRuntimeObserver::new(generation, Uuid::new_v4(), Uuid::new_v4());
        assert_eq!(
            observer.activate_generation(stale_activation),
            Err(AecObservationCollectorError::GenerationMismatch)
        );
        assert_eq!(
            observer.start_positive_control(100),
            Err(AecObservationCollectorError::GenerationMismatch)
        );
        observer.activate_generation(activation).unwrap();
        assert_eq!(observer.snapshot(), None);
        assert_eq!(
            observer.complete_positive_control(1),
            Err(AecObservationCollectorError::InvalidLifecycle)
        );
        observer.start_positive_control(100).unwrap();
        observer.observe(DuplexRuntimeEvent::GenerationRestart {
            attempt: std::num::NonZeroU32::new(1).unwrap(),
        });
        observer.complete_positive_control(101).unwrap();
        observer
            .start_scored_interval(Uuid::new_v4(), 101, Uuid::new_v4())
            .unwrap();
        let evidence = observer
            .complete_scored_interval(101 + AEC_OBSERVATION_DURATION_NS)
            .unwrap();
        assert!(evidence.positive_control.observer_errors > 0);
        assert!(evidence.terminated_early);
    }
}
