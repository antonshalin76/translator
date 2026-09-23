use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;
use tokio::sync::watch;

use crate::{
    AecAdmissionGuard, AecAdmissionReservation, AecCalibrationCoordinator, AecCoordinatorError,
    AecProofBinding,
};

pub trait AecProofInspector: Send + Sync {
    fn inspect_binding(&self, deadline: Instant) -> Result<AecProofBinding, AecCoordinatorError>;
}

#[derive(Clone)]
pub struct AecRuntimeAuthority {
    coordinator: Arc<AecCalibrationCoordinator>,
    inspector: Arc<dyn AecProofInspector>,
}

impl AecRuntimeAuthority {
    pub fn new(
        coordinator: Arc<AecCalibrationCoordinator>,
        inspector: Arc<dyn AecProofInspector>,
    ) -> Self {
        Self {
            coordinator,
            inspector,
        }
    }

    pub fn reserve(
        &self,
        deadline: Instant,
    ) -> Result<Arc<AecStartReservation>, AecCoordinatorError> {
        let binding = self.inspector.inspect_binding(deadline).inspect_err(|_| {
            self.coordinator.revoke();
        })?;
        let reservation = self.coordinator.reserve(&binding, true)?;
        Ok(Arc::new(AecStartReservation {
            coordinator: Arc::clone(&self.coordinator),
            inspector: Arc::clone(&self.inspector),
            binding,
            reservation: Mutex::new(Some(reservation)),
            guard: Mutex::new(None),
        }))
    }

    pub(crate) fn inspect_binding(
        &self,
        deadline: Instant,
    ) -> Result<AecProofBinding, AecCoordinatorError> {
        self.inspector.inspect_binding(deadline)
    }

    pub(crate) fn validate_inspection(
        &self,
        inspected: Result<AecProofBinding, AecCoordinatorError>,
    ) -> Result<(), AecCoordinatorError> {
        let binding = inspected.inspect_err(|_| {
            self.coordinator.revoke();
        })?;
        self.coordinator.validate_current(&binding, true)
    }

    pub(crate) fn subscribe_revocations(&self) -> watch::Receiver<u64> {
        self.coordinator.subscribe_revocations()
    }
}

pub struct AecStartReservation {
    coordinator: Arc<AecCalibrationCoordinator>,
    inspector: Arc<dyn AecProofInspector>,
    binding: AecProofBinding,
    reservation: Mutex<Option<AecAdmissionReservation>>,
    guard: Mutex<Option<AecAdmissionGuard>>,
}

impl AecStartReservation {
    pub(crate) fn authorizes_pair(&self, source_name: &str, sink_name: &str) -> bool {
        self.binding.source_name == source_name && self.binding.sink_name == sink_name
    }

    pub(crate) fn consume_before_effects(
        &self,
        deadline: Instant,
    ) -> Result<(), AecCoordinatorError> {
        let reservation = self
            .lock_reservation()
            .take()
            .ok_or(AecCoordinatorError::InvalidReservation)?;
        let binding = self.inspector.inspect_binding(deadline).inspect_err(|_| {
            self.coordinator.revoke();
        })?;
        let guard = self.coordinator.consume(reservation, &binding, true)?;
        *lock_recovering(&self.guard) = Some(guard);
        Ok(())
    }

    pub(crate) fn confirm_before_pcm(&self, deadline: Instant) -> Result<(), AecCoordinatorError> {
        let mut guard = lock_recovering(&self.guard);
        let guard_ref = guard
            .as_ref()
            .ok_or(AecCoordinatorError::InvalidReservation)?;
        let binding = self.inspector.inspect_binding(deadline).inspect_err(|_| {
            self.coordinator.revoke();
        })?;
        self.coordinator.validate_guard(guard_ref, &binding, true)?;
        guard.take();
        Ok(())
    }

    fn lock_reservation(&self) -> MutexGuard<'_, Option<AecAdmissionReservation>> {
        match self.reservation.lock() {
            Ok(reservation) => reservation,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(source_name: &str, sink_name: &str) -> Self {
        struct UnusedInspector;
        impl AecProofInspector for UnusedInspector {
            fn inspect_binding(
                &self,
                _deadline: Instant,
            ) -> Result<AecProofBinding, AecCoordinatorError> {
                Err(AecCoordinatorError::GraphUnavailable)
            }
        }
        Self {
            coordinator: Arc::new(AecCalibrationCoordinator::new()),
            inspector: Arc::new(UnusedInspector),
            binding: AecProofBinding {
                audio_server_id: "test-server".into(),
                source_hardware_id: "test-source-hardware".into(),
                sink_hardware_id: "test-sink-hardware".into(),
                source_name: source_name.into(),
                sink_name: sink_name.into(),
                source_port: "test-source-port".into(),
                sink_port: "test-sink-port".into(),
                source_channel_gains: vec![1],
                sink_channel_gains: vec![1],
                source_muted: false,
                sink_muted: false,
                source_geometry: "test-source-geometry".into(),
                sink_geometry: "test-sink-geometry".into(),
                aec_module_id: 1,
                aec_source_id: 2,
                aec_sink_id: 3,
                aec_generation: "test-generation".into(),
                aec_config_id: "test-aec-config".into(),
                vad_config_id: "test-vad-config".into(),
                provider_config_id: "test-provider-config".into(),
            },
            reservation: Mutex::new(None),
            guard: Mutex::new(None),
        }
    }
}

fn lock_recovering<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(value) => value,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl Drop for AecStartReservation {
    fn drop(&mut self) {
        let reservation = match self.reservation.get_mut() {
            Ok(reservation) => reservation.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(reservation) = reservation {
            self.coordinator.cancel_reservation(&reservation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AecCalibrationChallenge, AecProofStatus};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use translator_audio::{
        AEC_FIXTURE_DBFS, AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT,
        AEC_OBSERVATION_FRAME_SAMPLES, AEC_POWER_WINDOW_COUNT, AEC_SAMPLES_PER_POWER_WINDOW,
        AecDeviceMetadata, AecObservationEvidence, AecPositiveControl, AecPowerAcquisition,
        AecPowerWindow, AecValidationInput,
    };
    use uuid::Uuid;

    struct Inspector {
        binding: Mutex<AecProofBinding>,
        available: AtomicBool,
    }

    impl AecProofInspector for Inspector {
        fn inspect_binding(
            &self,
            deadline: Instant,
        ) -> Result<AecProofBinding, AecCoordinatorError> {
            if Instant::now() >= deadline || !self.available.load(Ordering::Acquire) {
                return Err(AecCoordinatorError::GraphUnavailable);
            }
            Ok(self.binding.lock().unwrap().clone())
        }
    }

    fn binding() -> AecProofBinding {
        AecProofBinding {
            audio_server_id: "server".into(),
            source_hardware_id: "source-hardware".into(),
            sink_hardware_id: "sink-hardware".into(),
            source_name: "source".into(),
            sink_name: "sink".into(),
            source_port: "source-port".into(),
            sink_port: "sink-port".into(),
            source_channel_gains: vec![1],
            sink_channel_gains: vec![1],
            source_muted: false,
            sink_muted: false,
            source_geometry: "source-geometry".into(),
            sink_geometry: "sink-geometry".into(),
            aec_module_id: 1,
            aec_source_id: 2,
            aec_sink_id: 3,
            aec_generation: "generation".into(),
            aec_config_id: "aec-config".into(),
            vad_config_id: "vad-config".into(),
            provider_config_id: "provider-config".into(),
        }
    }

    fn input(challenge: &AecCalibrationChallenge) -> AecValidationInput {
        let acquisition = |id: &str, power| AecPowerAcquisition {
            acquisition_id: id.into(),
            samples_per_window: AEC_SAMPLES_PER_POWER_WINDOW,
            powers: vec![power; 5],
        };
        AecValidationInput {
            metadata: AecDeviceMetadata {
                source_name: "source".into(),
                sink_name: "sink".into(),
                source_geometry: "source-geometry".into(),
                sink_geometry: "sink-geometry".into(),
                sink_port: "sink-port".into(),
                sink_volume_percent: 40,
            },
            binding: binding().measurement_binding(),
            fixture_acquisition_id: "fixture".into(),
            raw_baseline: acquisition("raw-baseline", 1.0),
            clean_baseline: acquisition("clean-baseline", 1.0),
            resolution: acquisition("resolution", 1.0),
            windows: (0..AEC_POWER_WINDOW_COUNT)
                .map(|sequence| AecPowerWindow {
                    sequence: sequence as u64,
                    raw_start_sample: sequence as u64 * AEC_SAMPLES_PER_POWER_WINDOW,
                    raw_end_sample: (sequence as u64 + 1) * AEC_SAMPLES_PER_POWER_WINDOW,
                    clean_start_sample: sequence as u64 * AEC_SAMPLES_PER_POWER_WINDOW,
                    clean_end_sample: (sequence as u64 + 1) * AEC_SAMPLES_PER_POWER_WINDOW,
                    raw_power: 101.0,
                    clean_power: 3.0,
                    raw_clipped_samples: 0,
                    clean_clipped_samples: 0,
                    fixture_dbfs: AEC_FIXTURE_DBFS,
                })
                .collect(),
            observation: AecObservationEvidence {
                observer_generation: "observer".into(),
                calibration_attempt_id: challenge.attempt_id().to_string(),
                challenge_id: challenge.challenge_id().to_string(),
                interval_id: challenge.interval_id().to_string(),
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
                    observer_generation: "observer".into(),
                    calibration_attempt_id: challenge.attempt_id().to_string(),
                    challenge_id: challenge.challenge_id().to_string(),
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

    impl crate::aec_validation::AecMonotonicClock for TestClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::Acquire)
        }
    }

    fn test_authority() -> (AecRuntimeAuthority, Arc<Inspector>, Arc<TestClock>) {
        let now_ns = 10_000_000_000 + AEC_OBSERVATION_DURATION_NS;
        let clock = Arc::new(TestClock(AtomicU64::new(now_ns)));
        let coordinator = Arc::new(AecCalibrationCoordinator::with_clock(clock.clone()));
        let expected_binding = binding();
        let challenge = coordinator
            .begin_attempt(Uuid::new_v4(), Uuid::new_v4(), expected_binding)
            .unwrap();
        coordinator
            .publish(&challenge, input(&challenge), true, true)
            .unwrap();
        let inspector = Arc::new(Inspector {
            binding: Mutex::new(binding()),
            available: AtomicBool::new(true),
        });
        (
            AecRuntimeAuthority::new(coordinator, inspector.clone()),
            inspector,
            clock,
        )
    }

    #[test]
    fn reservation_is_consumed_once_after_exact_reinspection() {
        let (authority, _, _) = test_authority();
        let reservation = authority
            .reserve(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        assert!(reservation.authorizes_pair("source", "sink"));
        reservation
            .consume_before_effects(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        reservation
            .confirm_before_pcm(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            reservation.consume_before_effects(Instant::now() + std::time::Duration::from_secs(1)),
            Err(AecCoordinatorError::InvalidReservation)
        );
    }

    #[test]
    fn binding_change_or_graph_loss_between_reserve_and_pcm_fails_closed() {
        let (authority, inspector, _) = test_authority();
        let reservation = authority
            .reserve(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        inspector.binding.lock().unwrap().sink_muted = true;
        assert_eq!(
            reservation.consume_before_effects(Instant::now() + std::time::Duration::from_secs(1)),
            Err(AecCoordinatorError::BindingChanged)
        );

        let (authority, inspector, _) = test_authority();
        let reservation = authority
            .reserve(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        reservation
            .consume_before_effects(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        inspector
            .binding
            .lock()
            .unwrap()
            .source_port
            .push_str("-changed");
        assert_eq!(
            reservation.confirm_before_pcm(Instant::now() + std::time::Duration::from_secs(1)),
            Err(AecCoordinatorError::BindingChanged)
        );

        let (authority, inspector, _) = test_authority();
        let reservation = authority
            .reserve(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        inspector.available.store(false, Ordering::Release);
        assert_eq!(
            reservation.consume_before_effects(Instant::now() + std::time::Duration::from_secs(1)),
            Err(AecCoordinatorError::GraphUnavailable)
        );
    }

    #[test]
    fn dropped_unconsumed_reservation_does_not_block_the_next_start() {
        let (authority, _, _) = test_authority();
        drop(
            authority
                .reserve(Instant::now() + std::time::Duration::from_secs(1))
                .unwrap(),
        );
        assert!(
            authority
                .reserve(Instant::now() + std::time::Duration::from_secs(1))
                .is_ok()
        );
    }

    #[test]
    fn expired_proof_is_rejected_at_reserve_and_at_pcm_boundary() {
        let measured_at_ns = 10_000_000_000 + AEC_OBSERVATION_DURATION_NS;
        let (authority, _, clock) = test_authority();
        let reservation = authority
            .reserve(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        reservation
            .consume_before_effects(Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        clock.0.store(
            measured_at_ns + crate::AEC_PROOF_LIFETIME_NS,
            Ordering::Release,
        );
        assert_eq!(
            reservation.confirm_before_pcm(Instant::now() + std::time::Duration::from_secs(1)),
            Err(AecCoordinatorError::ProofExpired)
        );

        let (authority, _, clock) = test_authority();
        clock.0.store(
            measured_at_ns + crate::AEC_PROOF_LIFETIME_NS,
            Ordering::Release,
        );
        assert!(matches!(
            authority.reserve(Instant::now() + std::time::Duration::from_secs(1)),
            Err(AecCoordinatorError::ProofExpired)
        ));
        assert_eq!(authority.coordinator.status(), AecProofStatus::Unavailable);
    }
}
