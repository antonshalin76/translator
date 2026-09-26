use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
};

use serde::Serialize;
use tokio::sync::watch;
use translator_audio::{AecCapability, AecMeasurementBinding, AecValidationInput, evaluate_aec};
use uuid::Uuid;

pub const AEC_PROOF_LIFETIME_NS: u64 = 300_000_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AecProofBinding {
    pub audio_server_id: String,
    pub source_hardware_id: String,
    pub sink_hardware_id: String,
    pub source_name: String,
    pub sink_name: String,
    pub source_port: String,
    pub sink_port: String,
    pub source_channel_gains: Vec<u32>,
    pub sink_channel_gains: Vec<u32>,
    pub source_muted: bool,
    pub sink_muted: bool,
    pub source_geometry: String,
    pub sink_geometry: String,
    pub aec_module_id: u32,
    pub aec_source_id: u32,
    pub aec_sink_id: u32,
    pub aec_generation: String,
    pub aec_config_id: String,
    pub vad_config_id: String,
    pub provider_config_id: String,
}

impl AecProofBinding {
    fn valid(&self) -> bool {
        [
            self.audio_server_id.as_str(),
            self.source_hardware_id.as_str(),
            self.sink_hardware_id.as_str(),
            self.source_name.as_str(),
            self.sink_name.as_str(),
            self.source_port.as_str(),
            self.sink_port.as_str(),
            self.source_geometry.as_str(),
            self.sink_geometry.as_str(),
            self.aec_generation.as_str(),
            self.aec_config_id.as_str(),
            self.vad_config_id.as_str(),
            self.provider_config_id.as_str(),
        ]
        .into_iter()
        .all(|value| !value.trim().is_empty())
            && !self.source_channel_gains.is_empty()
            && !self.sink_channel_gains.is_empty()
            && self.aec_module_id != 0
            && self.aec_source_id != 0
            && self.aec_sink_id != 0
    }

    pub(crate) fn measurement_binding(&self) -> AecMeasurementBinding {
        AecMeasurementBinding {
            audio_server_id: self.audio_server_id.clone(),
            source_hardware_id: self.source_hardware_id.clone(),
            sink_hardware_id: self.sink_hardware_id.clone(),
            source_name: self.source_name.clone(),
            sink_name: self.sink_name.clone(),
            source_port: self.source_port.clone(),
            sink_port: self.sink_port.clone(),
            source_channel_gains: self.source_channel_gains.clone(),
            sink_channel_gains: self.sink_channel_gains.clone(),
            source_muted: self.source_muted,
            sink_muted: self.sink_muted,
            source_geometry: self.source_geometry.clone(),
            sink_geometry: self.sink_geometry.clone(),
            aec_module_id: self.aec_module_id,
            aec_source_id: self.aec_source_id,
            aec_sink_id: self.aec_sink_id,
            aec_generation: self.aec_generation.clone(),
            aec_config_id: self.aec_config_id.clone(),
            vad_config_id: self.vad_config_id.clone(),
            provider_config_id: self.provider_config_id.clone(),
        }
    }
}

pub(crate) trait AecMonotonicClock: Send + Sync {
    fn now_ns(&self) -> u64;
}

struct SystemMonotonicClock;

impl AecMonotonicClock for SystemMonotonicClock {
    fn now_ns(&self) -> u64 {
        let value = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
        u64::try_from(value.tv_sec)
            .unwrap_or(u64::MAX)
            .saturating_mul(1_000_000_000)
            .saturating_add(u64::try_from(value.tv_nsec).unwrap_or(u64::MAX))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecCoordinatorError {
    Busy,
    InvalidChallenge,
    ChallengeConsumed,
    InvalidBinding,
    InvalidMeasurement,
    MeasurementFailed,
    ProbeTeardownIncomplete,
    GraphNotRetained,
    ProofUnavailable,
    ProofExpired,
    BindingChanged,
    GraphUnavailable,
    InvalidReservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AecCalibrationChallenge {
    process_nonce: Uuid,
    attempt_id: Uuid,
    interval_id: Uuid,
    challenge_id: Uuid,
}

impl AecCalibrationChallenge {
    pub const fn attempt_id(&self) -> Uuid {
        self.attempt_id
    }

    pub const fn interval_id(&self) -> Uuid {
        self.interval_id
    }

    pub const fn challenge_id(&self) -> Uuid {
        self.challenge_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AecAdmissionReservation {
    process_nonce: Uuid,
    proof_generation: u64,
    reservation_id: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct AecAdmissionGuard {
    _process_nonce: Uuid,
    _proof_generation: u64,
    binding: AecProofBinding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum AecProofStatus {
    Unavailable,
    Measuring,
    ValidationFailed,
    Validated {
        source_name: String,
        sink_name: String,
        expires_monotonic_ns: u64,
    },
    CleanupUncertain,
}

#[derive(Debug)]
struct ChallengeState {
    attempt_id: Uuid,
    interval_id: Uuid,
    challenge_id: Uuid,
    binding: AecProofBinding,
}

#[derive(Debug)]
struct ProofState {
    generation: u64,
    binding: AecProofBinding,
    expires_at_ns: u64,
}

#[derive(Debug)]
struct PendingReservation {
    id: u64,
    proof_generation: u64,
}

#[derive(Debug)]
struct CoordinatorState {
    challenge: Option<ChallengeState>,
    retired_challenges: VecDeque<Uuid>,
    proof: Option<ProofState>,
    pending: Option<PendingReservation>,
    next_proof_generation: u64,
    next_reservation_id: u64,
    revocation_generation: u64,
    status: AecProofStatus,
}

pub struct AecCalibrationCoordinator {
    process_nonce: Uuid,
    state: Mutex<CoordinatorState>,
    revocations: watch::Sender<u64>,
    clock: Arc<dyn AecMonotonicClock>,
}

impl AecCalibrationCoordinator {
    pub fn new() -> Self {
        Self::with_clock(Arc::new(SystemMonotonicClock))
    }

    pub(crate) fn with_clock(clock: Arc<dyn AecMonotonicClock>) -> Self {
        let (revocations, _) = watch::channel(0);
        Self {
            process_nonce: Uuid::new_v4(),
            state: Mutex::new(CoordinatorState {
                challenge: None,
                retired_challenges: VecDeque::new(),
                proof: None,
                pending: None,
                next_proof_generation: 0,
                next_reservation_id: 0,
                revocation_generation: 0,
                status: AecProofStatus::Unavailable,
            }),
            revocations,
            clock,
        }
    }

    pub fn begin_attempt(
        &self,
        attempt_id: Uuid,
        interval_id: Uuid,
        binding: AecProofBinding,
    ) -> Result<AecCalibrationChallenge, AecCoordinatorError> {
        let mut state = self.lock();
        if state.challenge.is_some()
            || state.pending.is_some()
            || state.status == AecProofStatus::CleanupUncertain
        {
            return Err(AecCoordinatorError::Busy);
        }
        if !binding.valid() {
            return Err(AecCoordinatorError::InvalidBinding);
        }
        let challenge_id = Uuid::new_v4();
        state.challenge = Some(ChallengeState {
            attempt_id,
            interval_id,
            challenge_id,
            binding,
        });
        state.proof = None;
        state.status = AecProofStatus::Measuring;
        Ok(AecCalibrationChallenge {
            process_nonce: self.process_nonce,
            attempt_id,
            interval_id,
            challenge_id,
        })
    }

    pub fn publish(
        &self,
        challenge: &AecCalibrationChallenge,
        input: AecValidationInput,
        probe_teardown_confirmed: bool,
        graph_retained: bool,
    ) -> Result<(), AecCoordinatorError> {
        let mut state = self.lock();
        if challenge.process_nonce == self.process_nonce
            && state.retired_challenges.contains(&challenge.challenge_id)
        {
            return Err(AecCoordinatorError::ChallengeConsumed);
        }
        let Some(challenge_state) = state.challenge.as_ref() else {
            return Err(AecCoordinatorError::InvalidChallenge);
        };
        if challenge.process_nonce != self.process_nonce
            || challenge.attempt_id != challenge_state.attempt_id
            || challenge.interval_id != challenge_state.interval_id
            || challenge.challenge_id != challenge_state.challenge_id
        {
            return Err(AecCoordinatorError::InvalidChallenge);
        }
        let challenge_id = challenge_state.challenge_id;
        let binding = challenge_state.binding.clone();
        state.challenge = None;
        retire_challenge(&mut state, challenge_id);
        state.proof = None;
        state.pending = None;
        if input.observation.calibration_attempt_id != challenge.attempt_id.to_string()
            || input.observation.challenge_id != challenge.challenge_id.to_string()
            || input.observation.interval_id != challenge.interval_id.to_string()
        {
            state.status = AecProofStatus::ValidationFailed;
            return Err(AecCoordinatorError::InvalidMeasurement);
        }
        if input.binding != binding.measurement_binding()
            || input.metadata.source_name != binding.source_name
            || input.metadata.sink_name != binding.sink_name
            || input.metadata.source_geometry != binding.source_geometry
            || input.metadata.sink_geometry != binding.sink_geometry
            || input.metadata.sink_port != binding.sink_port
        {
            state.status = AecProofStatus::ValidationFailed;
            return Err(AecCoordinatorError::InvalidBinding);
        }
        let measured_at_ns = input.observation.ended_monotonic_ns;
        let now_ns = self.clock.now_ns();
        if measured_at_ns > now_ns {
            state.status = AecProofStatus::ValidationFailed;
            return Err(AecCoordinatorError::InvalidMeasurement);
        }
        let record = evaluate_aec(input).map_err(|_| {
            state.status = AecProofStatus::ValidationFailed;
            AecCoordinatorError::InvalidMeasurement
        })?;
        if !record.validated {
            state.status = AecProofStatus::ValidationFailed;
            return Err(AecCoordinatorError::MeasurementFailed);
        }
        if !probe_teardown_confirmed {
            state.status = AecProofStatus::CleanupUncertain;
            return Err(AecCoordinatorError::ProbeTeardownIncomplete);
        }
        if !graph_retained {
            state.status = AecProofStatus::ValidationFailed;
            return Err(AecCoordinatorError::GraphNotRetained);
        }
        let generation = state
            .next_proof_generation
            .checked_add(1)
            .ok_or(AecCoordinatorError::Busy)?;
        let expires_at_ns = measured_at_ns
            .checked_add(AEC_PROOF_LIFETIME_NS)
            .ok_or(AecCoordinatorError::InvalidMeasurement)?;
        state.next_proof_generation = generation;
        state.status = AecProofStatus::Validated {
            source_name: binding.source_name.clone(),
            sink_name: binding.sink_name.clone(),
            expires_monotonic_ns: expires_at_ns,
        };
        state.proof = Some(ProofState {
            generation,
            binding,
            expires_at_ns,
        });
        Ok(())
    }

    pub fn reserve(
        &self,
        binding: &AecProofBinding,
        graph_available: bool,
    ) -> Result<AecAdmissionReservation, AecCoordinatorError> {
        let mut state = self.lock();
        if state.pending.is_some() {
            return Err(AecCoordinatorError::Busy);
        }
        let proof = state
            .proof
            .as_ref()
            .ok_or(AecCoordinatorError::ProofUnavailable)?;
        if self.clock.now_ns() >= proof.expires_at_ns {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::ProofExpired);
        }
        if &proof.binding != binding {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::BindingChanged);
        }
        if !graph_available {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::GraphUnavailable);
        }
        let proof_generation = proof.generation;
        let reservation_id = state
            .next_reservation_id
            .checked_add(1)
            .ok_or(AecCoordinatorError::Busy)?;
        state.next_reservation_id = reservation_id;
        state.pending = Some(PendingReservation {
            id: reservation_id,
            proof_generation,
        });
        Ok(AecAdmissionReservation {
            process_nonce: self.process_nonce,
            proof_generation,
            reservation_id,
        })
    }

    pub fn consume(
        &self,
        reservation: AecAdmissionReservation,
        binding: &AecProofBinding,
        graph_available: bool,
    ) -> Result<AecAdmissionGuard, AecCoordinatorError> {
        let mut state = self.lock();
        let valid_pending = state.pending.as_ref().is_some_and(|pending| {
            reservation.process_nonce == self.process_nonce
                && pending.id == reservation.reservation_id
                && pending.proof_generation == reservation.proof_generation
        });
        if !valid_pending {
            return Err(AecCoordinatorError::InvalidReservation);
        }
        state.pending = None;
        let proof = state
            .proof
            .as_ref()
            .ok_or(AecCoordinatorError::ProofUnavailable)?;
        if proof.generation != reservation.proof_generation {
            return Err(AecCoordinatorError::InvalidReservation);
        }
        if self.clock.now_ns() >= proof.expires_at_ns {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::ProofExpired);
        }
        if &proof.binding != binding {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::BindingChanged);
        }
        if !graph_available {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::GraphUnavailable);
        }
        Ok(AecAdmissionGuard {
            _process_nonce: self.process_nonce,
            _proof_generation: proof.generation,
            binding: proof.binding.clone(),
        })
    }

    pub(crate) fn validate_guard(
        &self,
        guard: &AecAdmissionGuard,
        binding: &AecProofBinding,
        graph_available: bool,
    ) -> Result<(), AecCoordinatorError> {
        let mut state = self.lock();
        if guard._process_nonce != self.process_nonce {
            return Err(AecCoordinatorError::InvalidReservation);
        }
        let proof = state
            .proof
            .as_ref()
            .ok_or(AecCoordinatorError::ProofUnavailable)?;
        if proof.generation != guard._proof_generation || proof.binding != guard.binding {
            return Err(AecCoordinatorError::InvalidReservation);
        }
        if self.clock.now_ns() >= proof.expires_at_ns {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::ProofExpired);
        }
        if &proof.binding != binding {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::BindingChanged);
        }
        if !graph_available {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::GraphUnavailable);
        }
        Ok(())
    }

    pub(crate) fn cancel_reservation(&self, reservation: &AecAdmissionReservation) {
        let mut state = self.lock();
        if state.pending.as_ref().is_some_and(|pending| {
            reservation.process_nonce == self.process_nonce
                && pending.id == reservation.reservation_id
                && pending.proof_generation == reservation.proof_generation
        }) {
            state.pending = None;
        }
    }

    pub fn cancel_attempt(&self, challenge: &AecCalibrationChallenge, cleanup_confirmed: bool) {
        let mut state = self.lock();
        if state.challenge.as_ref().is_some_and(|current| {
            challenge.process_nonce == self.process_nonce
                && current.attempt_id == challenge.attempt_id
                && current.challenge_id == challenge.challenge_id
        }) {
            if let Some(current) = state.challenge.take() {
                retire_challenge(&mut state, current.challenge_id);
            }
            state.proof = None;
            state.pending = None;
            state.status = if cleanup_confirmed {
                AecProofStatus::Unavailable
            } else {
                AecProofStatus::CleanupUncertain
            };
        }
    }

    pub fn revoke(&self) {
        self.revoke_locked(&mut self.lock());
    }

    pub(crate) fn validate_current(
        &self,
        binding: &AecProofBinding,
        graph_available: bool,
    ) -> Result<(), AecCoordinatorError> {
        let mut state = self.lock();
        let proof = state
            .proof
            .as_ref()
            .ok_or(AecCoordinatorError::ProofUnavailable)?;
        if self.clock.now_ns() >= proof.expires_at_ns {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::ProofExpired);
        }
        if &proof.binding != binding {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::BindingChanged);
        }
        if !graph_available {
            self.revoke_locked(&mut state);
            return Err(AecCoordinatorError::GraphUnavailable);
        }
        Ok(())
    }

    pub(crate) fn subscribe_revocations(&self) -> watch::Receiver<u64> {
        self.revocations.subscribe()
    }

    pub(crate) fn confirm_cleanup(&self) {
        let mut state = self.lock();
        if state.status == AecProofStatus::CleanupUncertain
            && state.challenge.is_none()
            && state.pending.is_none()
        {
            state.status = AecProofStatus::Unavailable;
        }
    }

    pub fn status(&self) -> AecProofStatus {
        self.lock().status.clone()
    }

    pub fn capability(&self) -> AecCapability {
        match self.status() {
            AecProofStatus::Validated {
                source_name,
                sink_name,
                ..
            } => AecCapability::ValidatedFor {
                source_name,
                sink_name,
            },
            AecProofStatus::ValidationFailed => AecCapability::ValidationFailed,
            AecProofStatus::Unavailable
            | AecProofStatus::Measuring
            | AecProofStatus::CleanupUncertain => AecCapability::AvailableUnvalidated,
        }
    }

    fn lock(&self) -> MutexGuard<'_, CoordinatorState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn revoke_locked(&self, state: &mut CoordinatorState) {
        let had_authority = state.proof.is_some() || state.pending.is_some();
        state.proof = None;
        state.pending = None;
        if state.status != AecProofStatus::CleanupUncertain {
            state.status = AecProofStatus::Unavailable;
        }
        if had_authority {
            state.revocation_generation = state.revocation_generation.saturating_add(1);
            self.revocations.send_replace(state.revocation_generation);
        }
    }
}

impl Default for AecCalibrationCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

fn retire_challenge(state: &mut CoordinatorState, challenge_id: Uuid) {
    const MAX_RETIRED_CHALLENGES: usize = 32;
    if state.retired_challenges.len() == MAX_RETIRED_CHALLENGES {
        state.retired_challenges.pop_front();
    }
    state.retired_challenges.push_back(challenge_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use translator_audio::{
        AEC_FIXTURE_DBFS, AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT,
        AEC_OBSERVATION_FRAME_SAMPLES, AEC_POWER_WINDOW_COUNT, AEC_SAMPLES_PER_POWER_WINDOW,
        AecDeviceMetadata, AecObservationEvidence, AecPositiveControl, AecPowerAcquisition,
        AecPowerWindow,
    };

    fn binding() -> AecProofBinding {
        AecProofBinding {
            audio_server_id: "pulse-server-cookie-1".into(),
            source_hardware_id: "usb-source-1".into(),
            sink_hardware_id: "usb-sink-1".into(),
            source_name: "alsa_input.physical".into(),
            sink_name: "alsa_output.physical".into(),
            source_port: "analog-input-mic".into(),
            sink_port: "analog-output-speaker".into(),
            source_channel_gains: vec![65_536],
            sink_channel_gains: vec![32_768, 32_768],
            source_muted: false,
            sink_muted: false,
            source_geometry: "desk-left-45cm".into(),
            sink_geometry: "desk-front-80cm".into(),
            aec_module_id: 73,
            aec_source_id: 81,
            aec_sink_id: 82,
            aec_generation: "aec-generation-1".into(),
            aec_config_id: "webrtc-48k-mono-v1".into(),
            vad_config_id: "vad-v1".into(),
            provider_config_id: "local-provider-v1".into(),
        }
    }

    fn input(challenge: &AecCalibrationChallenge, attenuation_db: f64) -> AecValidationInput {
        let acquisition = |id: &str, power| AecPowerAcquisition {
            acquisition_id: id.into(),
            samples_per_window: AEC_SAMPLES_PER_POWER_WINDOW,
            powers: vec![power; 5],
        };
        let raw_echo = 100.0;
        let clean_delta = raw_echo / 10_f64.powf(attenuation_db / 10.0);
        AecValidationInput {
            metadata: AecDeviceMetadata {
                source_name: "alsa_input.physical".into(),
                sink_name: "alsa_output.physical".into(),
                source_geometry: "desk-left-45cm".into(),
                sink_geometry: "desk-front-80cm".into(),
                sink_port: "analog-output-speaker".into(),
                sink_volume_percent: 40,
            },
            binding: binding().measurement_binding(),
            fixture_acquisition_id: "fixture-1".into(),
            raw_baseline: acquisition("raw-baseline-1", 1.0),
            clean_baseline: acquisition("clean-baseline-1", 1.0),
            resolution: acquisition("resolution-1", 1.0),
            windows: (0..AEC_POWER_WINDOW_COUNT)
                .map(|sequence| {
                    let start = sequence as u64 * AEC_SAMPLES_PER_POWER_WINDOW;
                    AecPowerWindow {
                        sequence: sequence as u64,
                        raw_start_sample: start,
                        raw_end_sample: start + AEC_SAMPLES_PER_POWER_WINDOW,
                        clean_start_sample: start,
                        clean_end_sample: start + AEC_SAMPLES_PER_POWER_WINDOW,
                        raw_power: 1.0 + raw_echo,
                        clean_power: 1.0 + clean_delta,
                        raw_clipped_samples: 0,
                        clean_clipped_samples: 0,
                        fixture_dbfs: AEC_FIXTURE_DBFS,
                    }
                })
                .collect(),
            observation: AecObservationEvidence {
                observer_generation: "observer-1".into(),
                calibration_attempt_id: challenge.attempt_id.to_string(),
                challenge_id: challenge.challenge_id.to_string(),
                interval_id: challenge.interval_id.to_string(),
                started_monotonic_ns: 10_000_000_000,
                ended_monotonic_ns: 10_000_000_000 + AEC_OBSERVATION_DURATION_NS,
                expected_frames: AEC_OBSERVATION_FRAME_COUNT,
                processed_frames: AEC_OBSERVATION_FRAME_COUNT,
                stream_generation: "runtime-generation-1".into(),
                sample_rate_hz: 16_000,
                channels: 1,
                frame_duration_ms: 20,
                samples_per_frame: AEC_OBSERVATION_FRAME_SAMPLES,
                first_frame_sequence: 0,
                last_frame_sequence: AEC_OBSERVATION_FRAME_COUNT - 1,
                first_capture_monotonic_ns: 10_000_000_000,
                last_capture_monotonic_ns: 10_000_000_000 + AEC_OBSERVATION_DURATION_NS
                    - 20_000_000,
                maximum_frame_gap_ns: 20_000_000,
                frame_gaps: 0,
                duplicate_frames: 0,
                out_of_order_frames: 0,
                vad_events_before: 0,
                vad_events_after: 0,
                provider_attempts_before: 0,
                provider_attempts_after: 0,
                provider_accepted_before: 0,
                provider_accepted_after: 0,
                resets: 0,
                dropped_frames: 0,
                observer_errors: 0,
                terminated_early: false,
                positive_control: AecPositiveControl {
                    observer_generation: "observer-1".into(),
                    calibration_attempt_id: challenge.attempt_id.to_string(),
                    challenge_id: challenge.challenge_id.to_string(),
                    completed_monotonic_ns: 9_000_000_000,
                    speech_started_events: 1,
                    provider_submission_attempts: 1,
                    provider_submissions_accepted: 1,
                    resets: 0,
                    observer_errors: 0,
                },
            },
        }
    }

    struct TestClock(AtomicU64);

    impl TestClock {
        fn set(&self, value: u64) {
            self.0.store(value, Ordering::SeqCst);
        }
    }

    impl AecMonotonicClock for TestClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn coordinator_at(now_ns: u64) -> (AecCalibrationCoordinator, Arc<TestClock>) {
        let clock = Arc::new(TestClock(AtomicU64::new(now_ns)));
        (AecCalibrationCoordinator::with_clock(clock.clone()), clock)
    }

    fn published() -> (AecCalibrationCoordinator, AecProofBinding, Arc<TestClock>) {
        let measured_at_ns = 10_000_000_000 + AEC_OBSERVATION_DURATION_NS;
        let (coordinator, clock) = coordinator_at(measured_at_ns);
        let binding = binding();
        let challenge = coordinator
            .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), binding.clone())
            .unwrap();
        coordinator
            .publish(&challenge, input(&challenge, 16.0), true, true)
            .unwrap();
        (coordinator, binding, clock)
    }

    #[test]
    fn proof_reservation_is_opaque_one_shot_and_bound() {
        let (coordinator, binding, _) = published();
        let reservation = coordinator.reserve(&binding, true).unwrap();
        let _guard = coordinator
            .consume(reservation.clone(), &binding, true)
            .unwrap();
        assert_eq!(
            coordinator.consume(reservation, &binding, true),
            Err(AecCoordinatorError::InvalidReservation)
        );
    }

    #[test]
    fn proof_ttl_is_expired_at_the_exact_boundary() {
        let (coordinator, binding, clock) = published();
        clock.set(10_000_000_000 + AEC_OBSERVATION_DURATION_NS + AEC_PROOF_LIFETIME_NS);
        assert_eq!(
            coordinator.reserve(&binding, true),
            Err(AecCoordinatorError::ProofExpired)
        );
        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
    }

    #[test]
    fn proof_ttl_is_valid_one_tick_before_expiry() {
        let (coordinator, binding, clock) = published();
        clock.set(10_000_000_000 + AEC_OBSERVATION_DURATION_NS + AEC_PROOF_LIFETIME_NS - 1);
        assert!(coordinator.reserve(&binding, true).is_ok());
    }

    #[test]
    fn future_measurement_completion_is_rejected() {
        let measured_at_ns = 10_000_000_000 + AEC_OBSERVATION_DURATION_NS;
        let (coordinator, _) = coordinator_at(measured_at_ns - 1);
        let expected_binding = binding();
        let challenge = coordinator
            .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), expected_binding)
            .unwrap();
        assert_eq!(
            coordinator.publish(&challenge, input(&challenge, 16.0), true, true),
            Err(AecCoordinatorError::InvalidMeasurement)
        );
    }

    #[test]
    fn measurement_metadata_must_match_the_published_runtime_binding() {
        let (coordinator, _) = coordinator_at(10_000_000_000 + AEC_OBSERVATION_DURATION_NS);
        let expected_binding = binding();
        let challenge = coordinator
            .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), expected_binding)
            .unwrap();
        let mut measurement = input(&challenge, 16.0);
        measurement.metadata.source_name = "different-physical-source".into();

        assert_eq!(
            coordinator.publish(&challenge, measurement, true, true),
            Err(AecCoordinatorError::InvalidBinding)
        );
        assert_eq!(coordinator.status(), AecProofStatus::ValidationFailed);
    }

    #[test]
    fn every_measurement_binding_field_is_bound_to_the_attempt() {
        let mutations: [fn(&mut AecMeasurementBinding); 20] = [
            |value| value.audio_server_id.push_str("-changed"),
            |value| value.source_hardware_id.push_str("-changed"),
            |value| value.sink_hardware_id.push_str("-changed"),
            |value| value.source_name.push_str("-changed"),
            |value| value.sink_name.push_str("-changed"),
            |value| value.source_port.push_str("-changed"),
            |value| value.sink_port.push_str("-changed"),
            |value| value.source_channel_gains[0] += 1,
            |value| value.sink_channel_gains[0] += 1,
            |value| value.source_muted = !value.source_muted,
            |value| value.sink_muted = !value.sink_muted,
            |value| value.source_geometry.push_str("-changed"),
            |value| value.sink_geometry.push_str("-changed"),
            |value| value.aec_module_id += 1,
            |value| value.aec_source_id += 1,
            |value| value.aec_sink_id += 1,
            |value| value.aec_generation.push_str("-changed"),
            |value| value.aec_config_id.push_str("-changed"),
            |value| value.vad_config_id.push_str("-changed"),
            |value| value.provider_config_id.push_str("-changed"),
        ];
        for mutate in mutations {
            let (coordinator, _) = coordinator_at(10_000_000_000 + AEC_OBSERVATION_DURATION_NS);
            let expected_binding = binding();
            let challenge = coordinator
                .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), expected_binding)
                .unwrap();
            let mut measurement = input(&challenge, 16.0);
            mutate(&mut measurement.binding);
            assert_eq!(
                coordinator.publish(&challenge, measurement, true, true),
                Err(AecCoordinatorError::InvalidBinding)
            );
        }
    }

    #[test]
    fn revocation_notification_is_generation_keyed_and_idempotent() {
        let (coordinator, _, _) = published();
        let mut revocations = coordinator.subscribe_revocations();

        coordinator.revoke();
        assert!(revocations.has_changed().unwrap());
        assert_eq!(*revocations.borrow_and_update(), 1);

        coordinator.revoke();
        assert!(!revocations.has_changed().unwrap());
    }

    #[test]
    fn changed_binding_or_graph_reinspection_revokes() {
        let mutations: [fn(&mut AecProofBinding); 20] = [
            |value| value.audio_server_id.push_str("-changed"),
            |value| value.source_hardware_id.push_str("-changed"),
            |value| value.sink_hardware_id.push_str("-changed"),
            |value| value.source_name.push_str("-changed"),
            |value| value.sink_name.push_str("-changed"),
            |value| value.source_port.push_str("-changed"),
            |value| value.sink_port.push_str("-changed"),
            |value| value.source_channel_gains[0] += 1,
            |value| value.sink_channel_gains[0] += 1,
            |value| value.source_muted = !value.source_muted,
            |value| value.sink_muted = !value.sink_muted,
            |value| value.source_geometry.push_str("-changed"),
            |value| value.sink_geometry.push_str("-changed"),
            |value| value.aec_module_id += 1,
            |value| value.aec_source_id += 1,
            |value| value.aec_sink_id += 1,
            |value| value.aec_generation.push_str("-changed"),
            |value| value.aec_config_id.push_str("-changed"),
            |value| value.vad_config_id.push_str("-changed"),
            |value| value.provider_config_id.push_str("-changed"),
        ];
        for mutate in mutations {
            let (coordinator, binding, _) = published();
            let mut changed = binding.clone();
            mutate(&mut changed);
            assert_eq!(
                coordinator.reserve(&changed, true),
                Err(AecCoordinatorError::BindingChanged)
            );
            assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        }

        let (coordinator, binding, _) = published();
        assert_eq!(
            coordinator.reserve(&binding, false),
            Err(AecCoordinatorError::GraphUnavailable)
        );
    }

    #[test]
    fn failed_measurement_or_teardown_consumes_challenge_without_proof() {
        for (attenuation, teardown, graph, expected) in [
            (14.0, true, true, AecCoordinatorError::MeasurementFailed),
            (
                16.0,
                false,
                true,
                AecCoordinatorError::ProbeTeardownIncomplete,
            ),
            (16.0, true, false, AecCoordinatorError::GraphNotRetained),
        ] {
            let (coordinator, _) = coordinator_at(10_000_000_000 + AEC_OBSERVATION_DURATION_NS);
            let expected_binding = binding();
            let challenge = coordinator
                .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), expected_binding)
                .unwrap();
            assert_eq!(
                coordinator.publish(&challenge, input(&challenge, attenuation), teardown, graph,),
                Err(expected)
            );
            assert_eq!(
                coordinator.publish(&challenge, input(&challenge, 16.0), true, true,),
                Err(AecCoordinatorError::ChallengeConsumed)
            );
            assert!(matches!(
                coordinator.status(),
                AecProofStatus::ValidationFailed | AecProofStatus::CleanupUncertain
            ));
        }
    }

    #[test]
    fn completed_attempt_allows_a_new_attempt_but_rejects_replay() {
        let (coordinator, _, _) = published();
        let next = coordinator
            .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), binding())
            .unwrap();
        coordinator.cancel_attempt(&next, true);
        assert_eq!(coordinator.status(), AecProofStatus::Unavailable);
        assert!(
            coordinator
                .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), binding())
                .is_ok()
        );
        assert_eq!(
            coordinator.publish(&next, input(&next, 16.0), true, true),
            Err(AecCoordinatorError::ChallengeConsumed)
        );
    }

    #[test]
    fn uncertain_cleanup_blocks_new_attempt_until_cleanup_is_confirmed() {
        let coordinator = AecCalibrationCoordinator::new();
        let challenge = coordinator
            .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), binding())
            .unwrap();
        coordinator.cancel_attempt(&challenge, false);
        assert_eq!(coordinator.status(), AecProofStatus::CleanupUncertain);
        assert_eq!(
            coordinator.begin_attempt(Uuid::new_v4(), Uuid::new_v4(), binding()),
            Err(AecCoordinatorError::Busy)
        );
        coordinator.revoke();
        assert_eq!(
            coordinator.begin_attempt(Uuid::new_v4(), Uuid::new_v4(), binding()),
            Err(AecCoordinatorError::Busy)
        );
        coordinator.confirm_cleanup();
        assert!(
            coordinator
                .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), binding())
                .is_ok()
        );
    }

    #[test]
    fn measurement_identity_must_match_the_minted_challenge() {
        let (coordinator, _) = coordinator_at(10_000_000_000 + AEC_OBSERVATION_DURATION_NS);
        let challenge = coordinator
            .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), binding())
            .unwrap();
        let other = AecCalibrationChallenge {
            process_nonce: challenge.process_nonce,
            attempt_id: Uuid::new_v4(),
            interval_id: challenge.interval_id,
            challenge_id: challenge.challenge_id,
        };
        assert_eq!(
            coordinator.publish(&challenge, input(&other, 16.0), true, true),
            Err(AecCoordinatorError::InvalidMeasurement)
        );
        assert_eq!(coordinator.status(), AecProofStatus::ValidationFailed);
        assert_eq!(
            coordinator.publish(&challenge, input(&challenge, 16.0), true, true,),
            Err(AecCoordinatorError::ChallengeConsumed)
        );
    }
}
