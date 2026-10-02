use thiserror::Error;

use crate::{
    AEC_ACQUISITION_WINDOW_COUNT, AEC_POWER_WINDOW_COUNT, AEC_SAMPLES_PER_POWER_WINDOW,
    AecValidationInput,
};

const TOTAL_FRAMES: usize = AEC_ACQUISITION_WINDOW_COUNT * 3 + AEC_POWER_WINDOW_COUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AecMeasurementError {
    #[error("synchronized AEC measurement is unavailable")]
    MeasurementUnavailable,
    #[error("AEC measurement source failed")]
    SourceFailed,
    #[error("AEC measurement frame provenance is inconsistent")]
    InvalidProvenance,
    #[error("AEC measurement frame order or sample range is discontinuous")]
    Discontinuous,
    #[error("AEC measurement frame format is invalid")]
    InvalidFormat,
    #[error("AEC measurement contains clipped samples")]
    Clipped,
    #[error("AEC measurement has insufficient frames")]
    Incomplete,
    #[error("AEC measurement was cancelled")]
    Cancelled,
    #[error("AEC measurement was invalidated")]
    Invalidated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AecChannelFrame {
    pub clock_id: String,
    pub generation: String,
    pub stream_id: String,
    pub acquisition_id: String,
    pub frame_id: u64,
    pub start_sample: u64,
    pub sample_rate_hz: u32,
    pub lost_frames: u64,
    pub samples: Vec<i16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AecPairedFrame {
    pub raw: AecChannelFrame,
    pub clean: AecChannelFrame,
}

pub trait AecMeasurementSource {
    // This interface is for finite, nonphysical acquisitions only. None marks
    // the end of the interval; close must release source-owned resources.
    fn next_pair(&mut self) -> Result<Option<AecPairedFrame>, AecMeasurementError>;
    fn close(&mut self) -> Result<(), AecMeasurementError>;
}

pub struct IndependentParecMeasurementSource;

impl AecMeasurementSource for IndependentParecMeasurementSource {
    fn next_pair(&mut self) -> Result<Option<AecPairedFrame>, AecMeasurementError> {
        Err(AecMeasurementError::MeasurementUnavailable)
    }

    fn close(&mut self) -> Result<(), AecMeasurementError> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecEvidenceDisposition {
    NonAdmissible,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AecSampleWindow {
    pub sequence: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub raw_power: f64,
    pub clean_power: f64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct AecSampleProvenance {
    clock_id: String,
    generation: String,
    raw_stream_id: String,
    clean_stream_id: String,
    acquisition_ids: Vec<String>,
    first_frame_id: u64,
    last_frame_id: u64,
    first_sample: u64,
    last_end_sample: u64,
}

impl AecSampleProvenance {
    pub fn clock_id(&self) -> &str {
        &self.clock_id
    }

    pub fn generation(&self) -> &str {
        &self.generation
    }

    pub fn stream_ids(&self) -> (&str, &str) {
        (&self.raw_stream_id, &self.clean_stream_id)
    }

    pub fn acquisition_ids(&self) -> &[String] {
        &self.acquisition_ids
    }

    pub fn frame_range(&self) -> (u64, u64) {
        (self.first_frame_id, self.last_frame_id)
    }

    pub fn sample_range(&self) -> (u64, u64) {
        (self.first_sample, self.last_end_sample)
    }
}

#[derive(Debug, PartialEq)]
pub struct AecSampleEvidence {
    provenance: AecSampleProvenance,
    raw_baseline_powers: Vec<f64>,
    clean_baseline_powers: Vec<f64>,
    resolution_powers: Vec<f64>,
    fixture_windows: Vec<AecSampleWindow>,
}

impl AecSampleEvidence {
    pub const fn disposition(&self) -> AecEvidenceDisposition {
        AecEvidenceDisposition::NonAdmissible
    }

    pub fn provenance(&self) -> &AecSampleProvenance {
        &self.provenance
    }

    pub fn raw_baseline_powers(&self) -> &[f64] {
        &self.raw_baseline_powers
    }

    pub fn clean_baseline_powers(&self) -> &[f64] {
        &self.clean_baseline_powers
    }

    pub fn resolution_powers(&self) -> &[f64] {
        &self.resolution_powers
    }

    pub fn fixture_windows(&self) -> &[AecSampleWindow] {
        &self.fixture_windows
    }
}

// Only the sealed native acquisition can construct production-ready input.
pub struct AecProofReadyMeasurement {
    input: AecValidationInput,
}

impl AecProofReadyMeasurement {
    pub(crate) fn from_native_acquisition(input: AecValidationInput) -> Self {
        Self { input }
    }
    pub fn into_validation_input(self) -> AecValidationInput {
        self.input
    }
}

#[derive(Default)]
pub struct AecSampleVerifier {
    frames: usize,
    clock_id: Option<String>,
    generation: Option<String>,
    raw_stream_id: Option<String>,
    clean_stream_id: Option<String>,
    acquisition_ids: Vec<String>,
    first_frame_id: Option<u64>,
    last_frame_id: Option<u64>,
    first_sample: Option<u64>,
    last_end_sample: Option<u64>,
    raw_baseline_powers: Vec<f64>,
    clean_baseline_powers: Vec<f64>,
    resolution_powers: Vec<f64>,
    fixture_windows: Vec<AecSampleWindow>,
    invalid: bool,
    cancelled: bool,
}

impl AecSampleVerifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn collect_from(
        source: &mut impl AecMeasurementSource,
    ) -> Result<AecSampleEvidence, AecMeasurementError> {
        let mut verifier = Self::new();
        let result = (|| {
            for _ in 0..TOTAL_FRAMES {
                let frame = source.next_pair()?.ok_or(AecMeasurementError::Incomplete)?;
                verifier.push(frame)?;
            }
            if source.next_pair()?.is_some() {
                return Err(AecMeasurementError::Discontinuous);
            }
            verifier.finish()
        })();
        // Evidence is returned only after acquisition cleanup succeeds.
        match source.close() {
            Ok(()) => result,
            Err(error) => Err(error),
        }
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.clear();
    }

    pub fn push(&mut self, frame: AecPairedFrame) -> Result<(), AecMeasurementError> {
        if self.cancelled {
            return Err(AecMeasurementError::Cancelled);
        }
        if self.invalid {
            return Err(AecMeasurementError::Invalidated);
        }
        if let Err(error) = self.validate_and_push(&frame) {
            self.invalid = true;
            self.clear();
            return Err(error);
        }
        Ok(())
    }

    pub fn finish(self) -> Result<AecSampleEvidence, AecMeasurementError> {
        if self.cancelled {
            return Err(AecMeasurementError::Cancelled);
        }
        if self.invalid {
            return Err(AecMeasurementError::Invalidated);
        }
        if self.frames != TOTAL_FRAMES {
            return Err(AecMeasurementError::Incomplete);
        }
        Ok(AecSampleEvidence {
            provenance: AecSampleProvenance {
                clock_id: self.clock_id.ok_or(AecMeasurementError::Invalidated)?,
                generation: self.generation.ok_or(AecMeasurementError::Invalidated)?,
                raw_stream_id: self.raw_stream_id.ok_or(AecMeasurementError::Invalidated)?,
                clean_stream_id: self
                    .clean_stream_id
                    .ok_or(AecMeasurementError::Invalidated)?,
                acquisition_ids: self.acquisition_ids,
                first_frame_id: self
                    .first_frame_id
                    .ok_or(AecMeasurementError::Invalidated)?,
                last_frame_id: self.last_frame_id.ok_or(AecMeasurementError::Invalidated)?,
                first_sample: self.first_sample.ok_or(AecMeasurementError::Invalidated)?,
                last_end_sample: self
                    .last_end_sample
                    .ok_or(AecMeasurementError::Invalidated)?,
            },
            raw_baseline_powers: self.raw_baseline_powers,
            clean_baseline_powers: self.clean_baseline_powers,
            resolution_powers: self.resolution_powers,
            fixture_windows: self.fixture_windows,
        })
    }

    fn validate_and_push(&mut self, frame: &AecPairedFrame) -> Result<(), AecMeasurementError> {
        let (raw, clean) = (&frame.raw, &frame.clean);
        if self.frames >= TOTAL_FRAMES {
            return Err(AecMeasurementError::Discontinuous);
        }
        if raw.clock_id.trim().is_empty()
            || raw.generation.trim().is_empty()
            || raw.stream_id.trim().is_empty()
            || clean.stream_id.trim().is_empty()
            || raw.acquisition_id.trim().is_empty()
            || raw.clock_id != clean.clock_id
            || raw.generation != clean.generation
            || raw.stream_id == clean.stream_id
            || raw.acquisition_id != clean.acquisition_id
            || self.clock_id.as_ref().is_some_and(|id| id != &raw.clock_id)
            || self
                .generation
                .as_ref()
                .is_some_and(|id| id != &raw.generation)
            || self
                .raw_stream_id
                .as_ref()
                .is_some_and(|id| id != &raw.stream_id)
            || self
                .clean_stream_id
                .as_ref()
                .is_some_and(|id| id != &clean.stream_id)
        {
            return Err(AecMeasurementError::InvalidProvenance);
        }
        let phase = phase_index(self.frames);
        if self.acquisition_ids.len() == phase {
            if self.acquisition_ids.contains(&raw.acquisition_id) {
                return Err(AecMeasurementError::InvalidProvenance);
            }
            self.acquisition_ids.push(raw.acquisition_id.clone());
        } else if self.acquisition_ids[phase] != raw.acquisition_id {
            return Err(AecMeasurementError::InvalidProvenance);
        }
        if raw.sample_rate_hz != 48_000
            || clean.sample_rate_hz != 48_000
            || raw.samples.len() != AEC_SAMPLES_PER_POWER_WINDOW as usize
            || clean.samples.len() != AEC_SAMPLES_PER_POWER_WINDOW as usize
        {
            return Err(AecMeasurementError::InvalidFormat);
        }
        let end_sample = raw
            .start_sample
            .checked_add(AEC_SAMPLES_PER_POWER_WINDOW)
            .ok_or(AecMeasurementError::Discontinuous)?;
        if raw.lost_frames != 0
            || clean.lost_frames != 0
            || raw.frame_id != clean.frame_id
            || raw.start_sample != clean.start_sample
            || self
                .last_frame_id
                .is_some_and(|id| id.checked_add(1) != Some(raw.frame_id))
            || self
                .last_end_sample
                .is_some_and(|end| end != raw.start_sample)
        {
            return Err(AecMeasurementError::Discontinuous);
        }
        if raw
            .samples
            .iter()
            .chain(clean.samples.iter())
            .any(|sample| *sample == i16::MIN || *sample == i16::MAX)
        {
            return Err(AecMeasurementError::Clipped);
        }
        let raw_power = power(&raw.samples);
        let clean_power = power(&clean.samples);
        match phase {
            0 => self.raw_baseline_powers.push(raw_power),
            1 => self.clean_baseline_powers.push(clean_power),
            2 => self.resolution_powers.push(raw_power.max(clean_power)),
            _ => self.fixture_windows.push(AecSampleWindow {
                sequence: (self.frames - AEC_ACQUISITION_WINDOW_COUNT * 3) as u64,
                start_sample: raw.start_sample,
                end_sample,
                raw_power,
                clean_power,
            }),
        }
        self.clock_id.get_or_insert_with(|| raw.clock_id.clone());
        self.generation
            .get_or_insert_with(|| raw.generation.clone());
        self.raw_stream_id
            .get_or_insert_with(|| raw.stream_id.clone());
        self.clean_stream_id
            .get_or_insert_with(|| clean.stream_id.clone());
        self.first_frame_id.get_or_insert(raw.frame_id);
        self.last_frame_id = Some(raw.frame_id);
        self.first_sample.get_or_insert(raw.start_sample);
        self.last_end_sample = Some(end_sample);
        self.frames += 1;
        Ok(())
    }

    fn clear(&mut self) {
        self.raw_baseline_powers.clear();
        self.clean_baseline_powers.clear();
        self.resolution_powers.clear();
        self.fixture_windows.clear();
    }
}

fn phase_index(frame: usize) -> usize {
    if frame < AEC_ACQUISITION_WINDOW_COUNT {
        0
    } else if frame < AEC_ACQUISITION_WINDOW_COUNT * 2 {
        1
    } else if frame < AEC_ACQUISITION_WINDOW_COUNT * 3 {
        2
    } else {
        3
    }
}

fn power(samples: &[i16]) -> f64 {
    samples
        .iter()
        .map(|sample| f64::from(*sample).powi(2))
        .sum::<f64>()
        / samples.len() as f64
}
