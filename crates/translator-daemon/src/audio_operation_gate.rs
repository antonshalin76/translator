use std::sync::{Arc, Mutex, MutexGuard};

use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOperationState {
    Idle,
    Production,
    HumanRoundTrip { session_id: Uuid },
    Calibration { attempt_id: Uuid },
    Stopping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AudioOperationAdmissionError {
    #[error("audio operation admission denied while another operation is active")]
    Busy { state: AudioOperationState },
    #[error("audio operation admission denied because the daemon is stopping")]
    Stopping,
    #[error("audio operation lease generation is exhausted")]
    GenerationExhausted,
}

#[derive(Debug)]
struct GateInner {
    state: AudioOperationState,
    generation: u64,
}

#[derive(Debug, Clone)]
pub struct AudioOperationGate {
    inner: Arc<Mutex<GateInner>>,
}

impl AudioOperationGate {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(GateInner {
                state: AudioOperationState::Idle,
                generation: 0,
            })),
        }
    }

    pub fn state(&self) -> AudioOperationState {
        lock_recovering(&self.inner).state
    }

    pub fn acquire_production(&self) -> Result<AudioOperationLease, AudioOperationAdmissionError> {
        self.acquire(AudioOperationState::Production)
    }

    pub fn acquire_human_round_trip(
        &self,
        session_id: Uuid,
    ) -> Result<AudioOperationLease, AudioOperationAdmissionError> {
        self.acquire(AudioOperationState::HumanRoundTrip { session_id })
    }

    pub fn acquire_manual(&self) -> Result<AudioOperationLease, AudioOperationAdmissionError> {
        self.acquire(AudioOperationState::Production)
    }

    pub fn acquire_calibration(
        &self,
        attempt_id: Uuid,
    ) -> Result<AudioOperationLease, AudioOperationAdmissionError> {
        self.acquire(AudioOperationState::Calibration { attempt_id })
    }

    pub fn begin_stopping(&self) {
        let mut inner = lock_recovering(&self.inner);
        if inner.state != AudioOperationState::Stopping {
            inner.generation = inner.generation.saturating_add(1);
            inner.state = AudioOperationState::Stopping;
        }
    }

    fn acquire(
        &self,
        requested_state: AudioOperationState,
    ) -> Result<AudioOperationLease, AudioOperationAdmissionError> {
        let mut inner = lock_recovering(&self.inner);
        match inner.state {
            AudioOperationState::Idle => {}
            AudioOperationState::Stopping => {
                return Err(AudioOperationAdmissionError::Stopping);
            }
            state => return Err(AudioOperationAdmissionError::Busy { state }),
        }

        let generation = inner
            .generation
            .checked_add(1)
            .ok_or(AudioOperationAdmissionError::GenerationExhausted)?;
        inner.generation = generation;
        inner.state = requested_state;
        Ok(AudioOperationLease {
            inner: Arc::clone(&self.inner),
            expected_state: requested_state,
            generation,
            active: true,
        })
    }
}

impl Default for AudioOperationGate {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct AudioOperationLease {
    inner: Arc<Mutex<GateInner>>,
    expected_state: AudioOperationState,
    generation: u64,
    active: bool,
}

impl AudioOperationLease {
    pub const fn state(&self) -> AudioOperationState {
        self.expected_state
    }

    pub fn relabel_calibration(
        &mut self,
        attempt_id: Uuid,
    ) -> Result<(), AudioOperationAdmissionError> {
        if !self.active || !matches!(self.expected_state, AudioOperationState::Calibration { .. }) {
            return Err(AudioOperationAdmissionError::Busy {
                state: self.expected_state,
            });
        }

        let mut inner = lock_recovering(&self.inner);
        match inner.state {
            AudioOperationState::Stopping => Err(AudioOperationAdmissionError::Stopping),
            AudioOperationState::Calibration { .. }
                if inner.generation == self.generation && inner.state == self.expected_state =>
            {
                let next = AudioOperationState::Calibration { attempt_id };
                inner.state = next;
                self.expected_state = next;
                Ok(())
            }
            state => Err(AudioOperationAdmissionError::Busy { state }),
        }
    }

    pub fn release(&mut self) -> bool {
        if !self.active {
            return false;
        }
        self.active = false;

        let mut inner = lock_recovering(&self.inner);
        if inner.generation == self.generation && inner.state == self.expected_state {
            inner.state = AudioOperationState::Idle;
            true
        } else {
            false
        }
    }
}

impl Drop for AudioOperationLease {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

fn lock_recovering(inner: &Mutex<GateInner>) -> MutexGuard<'_, GateInner> {
    match inner.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Admission =
        fn(&AudioOperationGate) -> Result<AudioOperationLease, AudioOperationAdmissionError>;

    fn human_round_trip(
        gate: &AudioOperationGate,
    ) -> Result<AudioOperationLease, AudioOperationAdmissionError> {
        gate.acquire_human_round_trip(Uuid::from_u128(1))
    }

    const COMPETING_OPERATIONS: [Admission; 3] = [
        AudioOperationGate::acquire_production,
        human_round_trip,
        AudioOperationGate::acquire_manual,
    ];

    #[test]
    fn calibration_excludes_each_audio_operation_in_both_orders() {
        for acquire_other in COMPETING_OPERATIONS {
            let gate = AudioOperationGate::new();
            let attempt_id = Uuid::from_u128(2);
            let calibration = gate.acquire_calibration(attempt_id).unwrap();
            assert_eq!(
                calibration.state(),
                AudioOperationState::Calibration { attempt_id }
            );
            assert_eq!(gate.state(), calibration.state());
            assert_eq!(
                acquire_other(&gate).unwrap_err(),
                AudioOperationAdmissionError::Busy {
                    state: calibration.state(),
                }
            );
            drop(calibration);

            let other = acquire_other(&gate).unwrap();
            assert_eq!(
                gate.acquire_calibration(attempt_id).unwrap_err(),
                AudioOperationAdmissionError::Busy {
                    state: other.state(),
                }
            );
            drop(other);
            assert_eq!(gate.state(), AudioOperationState::Idle);
            assert!(gate.acquire_calibration(attempt_id).is_ok());
        }
    }

    #[test]
    fn calibration_attempt_identity_is_not_a_reentrant_admission() {
        let gate = AudioOperationGate::new();
        let attempt_id = Uuid::from_u128(2);
        let lease = gate.acquire_calibration(attempt_id).unwrap();
        for duplicate in [attempt_id, Uuid::from_u128(3)] {
            assert_eq!(
                gate.acquire_calibration(duplicate).unwrap_err(),
                AudioOperationAdmissionError::Busy {
                    state: AudioOperationState::Calibration { attempt_id },
                }
            );
        }
        assert_eq!(gate.clone().state(), lease.state());
        drop(lease);
        assert_eq!(gate.state(), AudioOperationState::Idle);
    }

    #[test]
    fn calibration_release_is_explicit_and_cannot_release_its_replacement() {
        let gate = AudioOperationGate::new();
        let mut calibration = gate.acquire_calibration(Uuid::from_u128(2)).unwrap();
        assert_eq!(gate.clone().state(), calibration.state());
        assert!(calibration.release());
        assert_eq!(gate.state(), AudioOperationState::Idle);
        let replacement = gate.acquire_production().unwrap();
        assert!(!calibration.release());
        drop(calibration);
        assert_eq!(gate.state(), AudioOperationState::Production);
        drop(replacement);
        assert_eq!(gate.state(), AudioOperationState::Idle);
    }

    #[test]
    fn stale_calibration_generation_cannot_release_same_attempt_replacement() {
        let gate = AudioOperationGate::new();
        let attempt_id = Uuid::from_u128(2);
        let first = gate.acquire_calibration(attempt_id).unwrap();
        let mut stale = AudioOperationLease {
            inner: Arc::clone(&first.inner),
            expected_state: first.expected_state,
            generation: first.generation,
            active: true,
        };
        drop(first);
        let replacement = gate.acquire_calibration(attempt_id).unwrap();
        assert!(!stale.release());
        drop(stale);
        assert_eq!(gate.state(), replacement.state());
        drop(replacement);
        assert_eq!(gate.state(), AudioOperationState::Idle);
    }

    #[test]
    fn stopping_denies_calibration_and_revokes_existing_lease() {
        let gate = AudioOperationGate::new();
        let mut calibration = gate.acquire_calibration(Uuid::from_u128(2)).unwrap();
        gate.begin_stopping();
        gate.begin_stopping();
        assert_eq!(
            gate.acquire_calibration(Uuid::from_u128(3)).unwrap_err(),
            AudioOperationAdmissionError::Stopping
        );
        assert!(!calibration.release());
        drop(calibration);
        assert_eq!(gate.state(), AudioOperationState::Stopping);
        for acquire_other in COMPETING_OPERATIONS {
            assert_eq!(
                acquire_other(&gate).unwrap_err(),
                AudioOperationAdmissionError::Stopping
            );
        }

        let idle_gate = AudioOperationGate::new();
        idle_gate.begin_stopping();
        assert_eq!(
            idle_gate
                .acquire_calibration(Uuid::from_u128(2))
                .unwrap_err(),
            AudioOperationAdmissionError::Stopping
        );
    }

    #[test]
    fn calibration_generation_exhaustion_never_wraps_or_reopens_stopping() {
        let gate = AudioOperationGate::new();
        lock_recovering(&gate.inner).generation = u64::MAX - 1;
        let lease = gate.acquire_calibration(Uuid::from_u128(2)).unwrap();
        assert_eq!(lease.generation, u64::MAX);
        drop(lease);
        assert_eq!(
            gate.acquire_calibration(Uuid::from_u128(2)).unwrap_err(),
            AudioOperationAdmissionError::GenerationExhausted
        );
        assert_eq!(gate.state(), AudioOperationState::Idle);

        let gate = AudioOperationGate::new();
        lock_recovering(&gate.inner).generation = u64::MAX - 1;
        let mut lease = gate.acquire_calibration(Uuid::from_u128(2)).unwrap();
        gate.begin_stopping();
        assert!(!lease.release());
        assert_eq!(gate.state(), AudioOperationState::Stopping);
    }

    #[test]
    fn calibration_relabel_transfers_attempt_without_idle_gap() {
        let gate = AudioOperationGate::new();
        let first = Uuid::from_u128(2);
        let second = Uuid::from_u128(3);
        let mut lease = gate.acquire_calibration(first).unwrap();
        assert_eq!(
            lease.state(),
            AudioOperationState::Calibration { attempt_id: first }
        );
        lease.relabel_calibration(second).unwrap();
        assert_eq!(
            lease.state(),
            AudioOperationState::Calibration { attempt_id: second }
        );
        assert_eq!(gate.state(), lease.state());
        assert_eq!(
            gate.acquire_calibration(Uuid::from_u128(4)).unwrap_err(),
            AudioOperationAdmissionError::Busy {
                state: lease.state()
            }
        );
        for acquire_other in COMPETING_OPERATIONS {
            assert_eq!(
                acquire_other(&gate).unwrap_err(),
                AudioOperationAdmissionError::Busy {
                    state: lease.state()
                }
            );
        }
        drop(lease);
        assert_eq!(gate.state(), AudioOperationState::Idle);
        assert!(gate.acquire_calibration(first).is_ok());
    }

    #[test]
    fn calibration_relabel_rejects_stopping_released_and_foreign_state() {
        let gate = AudioOperationGate::new();
        let attempt_id = Uuid::from_u128(2);
        let replacement = Uuid::from_u128(3);
        let mut lease = gate.acquire_calibration(attempt_id).unwrap();
        assert!(lease.release());
        assert_eq!(
            lease.relabel_calibration(replacement).unwrap_err(),
            AudioOperationAdmissionError::Busy {
                state: AudioOperationState::Calibration { attempt_id }
            }
        );
        assert_eq!(gate.state(), AudioOperationState::Idle);

        let mut production = gate.acquire_production().unwrap();
        assert_eq!(
            production.relabel_calibration(replacement).unwrap_err(),
            AudioOperationAdmissionError::Busy {
                state: AudioOperationState::Production
            }
        );
        drop(production);

        let mut calibration = gate.acquire_calibration(attempt_id).unwrap();
        let mut stale = AudioOperationLease {
            inner: Arc::clone(&calibration.inner),
            expected_state: calibration.expected_state,
            generation: calibration.generation,
            active: true,
        };
        calibration.relabel_calibration(replacement).unwrap();
        assert_eq!(
            stale.relabel_calibration(Uuid::from_u128(4)).unwrap_err(),
            AudioOperationAdmissionError::Busy {
                state: AudioOperationState::Calibration {
                    attempt_id: replacement
                }
            }
        );
        drop(stale);
        assert_eq!(
            gate.state(),
            AudioOperationState::Calibration {
                attempt_id: replacement
            }
        );
        gate.begin_stopping();
        assert_eq!(
            calibration
                .relabel_calibration(Uuid::from_u128(4))
                .unwrap_err(),
            AudioOperationAdmissionError::Stopping
        );
        assert_eq!(gate.state(), AudioOperationState::Stopping);
    }
}
