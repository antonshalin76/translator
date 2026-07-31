use std::sync::{Arc, Mutex, MutexGuard};

use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOperationState {
    Idle,
    Production,
    HumanRoundTrip { session_id: Uuid },
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
