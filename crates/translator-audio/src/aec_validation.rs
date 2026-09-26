use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const AEC_ATTENUATION_THRESHOLD_DB: f64 = 15.0;
pub const AEC_MIN_RAW_SNR_DB: f64 = 15.0;
pub const AEC_MIN_RESOLVABLE_ATTENUATION_DB: f64 = 18.0;
pub const AEC_FIXTURE_DBFS: f64 = -20.0;
pub const AEC_POWER_WINDOW_COUNT: usize = 30;
pub const AEC_ACQUISITION_WINDOW_COUNT: usize = 5;
pub const AEC_SAMPLES_PER_POWER_WINDOW: u64 = 48_000;
pub const AEC_MAX_BASELINE_SPREAD_DB: f64 = 1.0;
pub const AEC_OBSERVATION_FRAME_COUNT: u64 = 3_000;
pub const AEC_OBSERVATION_SAMPLE_RATE_HZ: u32 = 16_000;
pub const AEC_OBSERVATION_CHANNELS: u8 = 1;
pub const AEC_OBSERVATION_FRAME_DURATION_MS: u16 = 20;
pub const AEC_OBSERVATION_FRAME_DURATION_NS: u64 = 20_000_000;
pub const AEC_OBSERVATION_FRAME_SAMPLES: u64 = 320;
pub const AEC_OBSERVATION_DURATION_NS: u64 = 60_000_000_000;
pub const AEC_POSITIVE_CONTROL_MAX_GAP_NS: u64 = 5_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecValidationError {
    InvalidValidationInput,
}

impl fmt::Display for AecValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AEC validation input is invalid")
    }
}

impl std::error::Error for AecValidationError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecDeviceMetadata {
    pub source_name: String,
    pub sink_name: String,
    pub source_geometry: String,
    pub sink_geometry: String,
    pub sink_port: String,
    pub sink_volume_percent: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AecMeasurementBinding {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecPowerAcquisition {
    pub acquisition_id: String,
    pub samples_per_window: u64,
    pub powers: Vec<f64>,
}

impl AecPowerAcquisition {
    fn summary(&self) -> Result<AecPowerSummary, AecValidationError> {
        if self.acquisition_id.trim().is_empty()
            || self.samples_per_window != AEC_SAMPLES_PER_POWER_WINDOW
            || self.powers.len() != AEC_ACQUISITION_WINDOW_COUNT
            || self
                .powers
                .iter()
                .any(|power| !power.is_finite() || *power <= 0.0)
        {
            return invalid();
        }
        let mut powers = self.powers.clone();
        powers.sort_by(f64::total_cmp);
        let spread_db = 10.0 * (powers[powers.len() - 1] / powers[0]).log10();
        if !spread_db.is_finite() || spread_db > AEC_MAX_BASELINE_SPREAD_DB {
            return invalid();
        }
        let median = powers[powers.len() / 2];
        let uncertainty = powers
            .iter()
            .map(|power| (power - median).abs())
            .max_by(f64::total_cmp)
            .ok_or(AecValidationError::InvalidValidationInput)?;
        Ok(AecPowerSummary {
            median,
            uncertainty,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AecPowerSummary {
    median: f64,
    uncertainty: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecPowerWindow {
    pub sequence: u64,
    pub raw_start_sample: u64,
    pub raw_end_sample: u64,
    pub clean_start_sample: u64,
    pub clean_end_sample: u64,
    pub raw_power: f64,
    pub clean_power: f64,
    pub raw_clipped_samples: u64,
    pub clean_clipped_samples: u64,
    pub fixture_dbfs: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct AecWindowMeasurement {
    attenuation_db: f64,
    raw_snr_db: f64,
    resolvable_attenuation_db: f64,
}

impl AecPowerWindow {
    fn measure(
        &self,
        raw_baseline: AecPowerSummary,
        clean_baseline: AecPowerSummary,
        resolution: AecPowerSummary,
    ) -> Result<AecWindowMeasurement, AecValidationError> {
        if self.raw_clipped_samples != 0
            || self.clean_clipped_samples != 0
            || self.fixture_dbfs != AEC_FIXTURE_DBFS
            || !self.raw_power.is_finite()
            || !self.clean_power.is_finite()
            || self.raw_power <= 0.0
            || self.clean_power <= 0.0
            || self.raw_start_sample != self.clean_start_sample
            || self.raw_end_sample != self.clean_end_sample
            || self.raw_end_sample.saturating_sub(self.raw_start_sample)
                != AEC_SAMPLES_PER_POWER_WINDOW
        {
            return invalid();
        }
        let raw_echo = self.raw_power - raw_baseline.median - raw_baseline.uncertainty;
        let clean_delta = self.clean_power - clean_baseline.median;
        let resolution_floor =
            (resolution.median + resolution.uncertainty).max(clean_baseline.uncertainty);
        if raw_echo <= 0.0 || clean_delta <= resolution_floor {
            return invalid();
        }
        let raw_noise_ceiling =
            (raw_baseline.median + raw_baseline.uncertainty).max(raw_baseline.uncertainty);
        let clean_echo_ceiling = clean_delta + clean_baseline.uncertainty;
        let raw_resolution_floor =
            (resolution.median + resolution.uncertainty).max(raw_baseline.uncertainty);
        let raw_snr_db = power_ratio_db(raw_echo, raw_noise_ceiling)?;
        let resolvable_attenuation_db = power_ratio_db(raw_echo, raw_resolution_floor)?;
        let attenuation_db = power_ratio_db(raw_echo, clean_echo_ceiling)?;
        if raw_snr_db < AEC_MIN_RAW_SNR_DB
            || resolvable_attenuation_db < AEC_MIN_RESOLVABLE_ATTENUATION_DB
        {
            return invalid();
        }
        Ok(AecWindowMeasurement {
            attenuation_db,
            raw_snr_db,
            resolvable_attenuation_db,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AecPositiveControl {
    pub observer_generation: String,
    pub calibration_attempt_id: String,
    pub challenge_id: String,
    pub completed_monotonic_ns: u64,
    pub speech_started_events: u64,
    pub provider_submission_attempts: u64,
    pub provider_submissions_accepted: u64,
    pub resets: u64,
    pub observer_errors: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AecObservationEvidence {
    pub observer_generation: String,
    pub calibration_attempt_id: String,
    pub challenge_id: String,
    pub interval_id: String,
    pub started_monotonic_ns: u64,
    pub ended_monotonic_ns: u64,
    pub expected_frames: u64,
    pub processed_frames: u64,
    pub stream_generation: String,
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub frame_duration_ms: u16,
    pub samples_per_frame: u64,
    pub first_frame_sequence: u64,
    pub last_frame_sequence: u64,
    pub first_capture_monotonic_ns: u64,
    pub last_capture_monotonic_ns: u64,
    pub maximum_frame_gap_ns: u64,
    pub frame_gaps: u64,
    pub duplicate_frames: u64,
    pub out_of_order_frames: u64,
    pub vad_events_before: u64,
    pub vad_events_after: u64,
    pub provider_attempts_before: u64,
    pub provider_attempts_after: u64,
    pub provider_accepted_before: u64,
    pub provider_accepted_after: u64,
    pub resets: u64,
    pub dropped_frames: u64,
    pub observer_errors: u64,
    pub terminated_early: bool,
    pub positive_control: AecPositiveControl,
}

impl AecObservationEvidence {
    fn deltas(&self) -> Result<(u64, u64), AecValidationError> {
        if self.observer_generation.trim().is_empty()
            || self.calibration_attempt_id.trim().is_empty()
            || self.challenge_id.trim().is_empty()
            || self.interval_id.trim().is_empty()
            || self
                .ended_monotonic_ns
                .checked_sub(self.started_monotonic_ns)
                != Some(AEC_OBSERVATION_DURATION_NS)
            || self.expected_frames != AEC_OBSERVATION_FRAME_COUNT
            || self.processed_frames != AEC_OBSERVATION_FRAME_COUNT
            || self.stream_generation.trim().is_empty()
            || self.sample_rate_hz != AEC_OBSERVATION_SAMPLE_RATE_HZ
            || self.channels != AEC_OBSERVATION_CHANNELS
            || self.frame_duration_ms != AEC_OBSERVATION_FRAME_DURATION_MS
            || self.samples_per_frame != AEC_OBSERVATION_FRAME_SAMPLES
            || self
                .last_frame_sequence
                .checked_sub(self.first_frame_sequence)
                .and_then(|delta| delta.checked_add(1))
                != Some(AEC_OBSERVATION_FRAME_COUNT)
            || self.first_capture_monotonic_ns < self.started_monotonic_ns
            || self
                .first_capture_monotonic_ns
                .saturating_sub(self.started_monotonic_ns)
                > AEC_OBSERVATION_FRAME_DURATION_NS
            || self.last_capture_monotonic_ns > self.ended_monotonic_ns
            || self
                .ended_monotonic_ns
                .saturating_sub(self.last_capture_monotonic_ns)
                > AEC_OBSERVATION_FRAME_DURATION_NS
            || self.maximum_frame_gap_ns > AEC_OBSERVATION_FRAME_DURATION_NS
            || self.frame_gaps != 0
            || self.duplicate_frames != 0
            || self.out_of_order_frames != 0
            || self.resets != 0
            || self.dropped_frames != 0
            || self.observer_errors != 0
            || self.terminated_early
            || self.positive_control.observer_generation != self.observer_generation
            || self.positive_control.calibration_attempt_id != self.calibration_attempt_id
            || self.positive_control.challenge_id != self.challenge_id
            || self
                .started_monotonic_ns
                .checked_sub(self.positive_control.completed_monotonic_ns)
                .is_none_or(|gap| gap > AEC_POSITIVE_CONTROL_MAX_GAP_NS)
            || self.positive_control.speech_started_events == 0
            || self.positive_control.provider_submission_attempts == 0
            || self.positive_control.provider_submissions_accepted == 0
            || self.positive_control.provider_submissions_accepted
                > self.positive_control.provider_submission_attempts
            || self.positive_control.resets != 0
            || self.positive_control.observer_errors != 0
            || self.provider_accepted_before > self.provider_attempts_before
            || self.provider_accepted_after > self.provider_attempts_after
        {
            return invalid();
        }
        let vad_delta = self
            .vad_events_after
            .checked_sub(self.vad_events_before)
            .ok_or(AecValidationError::InvalidValidationInput)?;
        let provider_delta = self
            .provider_attempts_after
            .checked_sub(self.provider_attempts_before)
            .ok_or(AecValidationError::InvalidValidationInput)?;
        let accepted_delta = self
            .provider_accepted_after
            .checked_sub(self.provider_accepted_before)
            .ok_or(AecValidationError::InvalidValidationInput)?;
        if accepted_delta > provider_delta {
            return invalid();
        }
        Ok((vad_delta, provider_delta))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecValidationInput {
    pub metadata: AecDeviceMetadata,
    pub binding: AecMeasurementBinding,
    pub fixture_acquisition_id: String,
    pub raw_baseline: AecPowerAcquisition,
    pub clean_baseline: AecPowerAcquisition,
    pub resolution: AecPowerAcquisition,
    pub windows: Vec<AecPowerWindow>,
    pub observation: AecObservationEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecValidationRecord {
    pub metadata: AecDeviceMetadata,
    pub window_count: usize,
    pub median_attenuation_db: f64,
    pub minimum_raw_snr_db: f64,
    pub minimum_resolvable_attenuation_db: f64,
    pub attenuation_passed: bool,
    pub vad_events: u64,
    pub provider_submission_attempts: u64,
    pub vad_passed: bool,
    pub provider_passed: bool,
    pub validated: bool,
}

pub fn evaluate_aec(input: AecValidationInput) -> Result<AecValidationRecord, AecValidationError> {
    validate_metadata(&input.metadata)?;
    validate_binding(&input.binding)?;
    let raw_baseline = input.raw_baseline.summary()?;
    let clean_baseline = input.clean_baseline.summary()?;
    let resolution = input.resolution.summary()?;
    let identities = [
        input.fixture_acquisition_id.as_str(),
        input.raw_baseline.acquisition_id.as_str(),
        input.clean_baseline.acquisition_id.as_str(),
        input.resolution.acquisition_id.as_str(),
    ];
    if input.fixture_acquisition_id.trim().is_empty()
        || identities.iter().copied().collect::<HashSet<_>>().len() != identities.len()
        || input.windows.len() != AEC_POWER_WINDOW_COUNT
    {
        return invalid();
    }

    let mut measurements = Vec::with_capacity(AEC_POWER_WINDOW_COUNT);
    for (index, window) in input.windows.iter().enumerate() {
        let expected_start = index as u64 * AEC_SAMPLES_PER_POWER_WINDOW;
        if window.sequence != index as u64 || window.raw_start_sample != expected_start {
            return invalid();
        }
        measurements.push(window.measure(raw_baseline, clean_baseline, resolution)?);
    }
    let mut attenuation_values = measurements
        .iter()
        .map(|measurement| measurement.attenuation_db)
        .collect::<Vec<_>>();
    attenuation_values.sort_by(f64::total_cmp);
    let middle = attenuation_values.len() / 2;
    let median_attenuation_db = (attenuation_values[middle - 1] + attenuation_values[middle]) / 2.0;
    let minimum_raw_snr_db = measurements
        .iter()
        .map(|measurement| measurement.raw_snr_db)
        .min_by(f64::total_cmp)
        .ok_or(AecValidationError::InvalidValidationInput)?;
    let minimum_resolvable_attenuation_db = measurements
        .iter()
        .map(|measurement| measurement.resolvable_attenuation_db)
        .min_by(f64::total_cmp)
        .ok_or(AecValidationError::InvalidValidationInput)?;
    let (vad_events, provider_submission_attempts) = input.observation.deltas()?;
    let attenuation_passed = median_attenuation_db >= AEC_ATTENUATION_THRESHOLD_DB;
    let vad_passed = vad_events == 0;
    let provider_passed = provider_submission_attempts == 0;
    Ok(AecValidationRecord {
        metadata: input.metadata,
        window_count: input.windows.len(),
        median_attenuation_db,
        minimum_raw_snr_db,
        minimum_resolvable_attenuation_db,
        attenuation_passed,
        vad_events,
        provider_submission_attempts,
        vad_passed,
        provider_passed,
        validated: attenuation_passed && vad_passed && provider_passed,
    })
}

fn validate_binding(binding: &AecMeasurementBinding) -> Result<(), AecValidationError> {
    if [
        binding.audio_server_id.as_str(),
        binding.source_hardware_id.as_str(),
        binding.sink_hardware_id.as_str(),
        binding.source_name.as_str(),
        binding.sink_name.as_str(),
        binding.source_port.as_str(),
        binding.sink_port.as_str(),
        binding.source_geometry.as_str(),
        binding.sink_geometry.as_str(),
        binding.aec_generation.as_str(),
        binding.aec_config_id.as_str(),
        binding.vad_config_id.as_str(),
        binding.provider_config_id.as_str(),
    ]
    .into_iter()
    .any(|value| value.trim().is_empty())
        || binding.source_channel_gains.is_empty()
        || binding.sink_channel_gains.is_empty()
        || binding.aec_module_id == 0
        || binding.aec_source_id == 0
        || binding.aec_sink_id == 0
    {
        return invalid();
    }
    Ok(())
}

fn validate_metadata(metadata: &AecDeviceMetadata) -> Result<(), AecValidationError> {
    if metadata.source_name.trim().is_empty()
        || metadata.sink_name.trim().is_empty()
        || metadata.source_geometry.trim().is_empty()
        || metadata.sink_geometry.trim().is_empty()
        || metadata.sink_port.trim().is_empty()
        || metadata.sink_volume_percent > 100
    {
        return invalid();
    }
    Ok(())
}

fn power_ratio_db(numerator: f64, denominator: f64) -> Result<f64, AecValidationError> {
    if !numerator.is_finite() || !denominator.is_finite() || numerator <= 0.0 || denominator <= 0.0
    {
        return invalid();
    }
    let value = 10.0 * (numerator / denominator).log10();
    if value.is_finite() {
        Ok(value)
    } else {
        invalid()
    }
}

fn invalid<T>() -> Result<T, AecValidationError> {
    Err(AecValidationError::InvalidValidationInput)
}
