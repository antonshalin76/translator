use translator_audio::{
    AEC_ATTENUATION_THRESHOLD_DB, AEC_FIXTURE_DBFS, AEC_MIN_RAW_SNR_DB,
    AEC_MIN_RESOLVABLE_ATTENUATION_DB, AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT,
    AEC_OBSERVATION_FRAME_SAMPLES, AEC_POWER_WINDOW_COUNT, AEC_SAMPLES_PER_POWER_WINDOW,
    AecDeviceMetadata, AecMeasurementBinding, AecObservationEvidence, AecPositiveControl,
    AecPowerAcquisition, AecPowerWindow, AecValidationError, AecValidationInput, evaluate_aec,
};

fn acquisition(id: &str, power: f64) -> AecPowerAcquisition {
    AecPowerAcquisition {
        acquisition_id: id.to_owned(),
        samples_per_window: AEC_SAMPLES_PER_POWER_WINDOW,
        powers: vec![power; 5],
    }
}

fn observation() -> AecObservationEvidence {
    AecObservationEvidence {
        observer_generation: "observer-1".to_owned(),
        calibration_attempt_id: "attempt-1".to_owned(),
        challenge_id: "challenge-1".to_owned(),
        interval_id: "far-end-1".to_owned(),
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
        last_capture_monotonic_ns: 10_000_000_000 + AEC_OBSERVATION_DURATION_NS - 20_000_000,
        maximum_frame_gap_ns: 20_000_000,
        frame_gaps: 0,
        duplicate_frames: 0,
        out_of_order_frames: 0,
        vad_events_before: 3,
        vad_events_after: 3,
        provider_attempts_before: 2,
        provider_attempts_after: 2,
        provider_accepted_before: 2,
        provider_accepted_after: 2,
        resets: 0,
        dropped_frames: 0,
        observer_errors: 0,
        terminated_early: false,
        positive_control: AecPositiveControl {
            observer_generation: "observer-1".to_owned(),
            calibration_attempt_id: "attempt-1".to_owned(),
            challenge_id: "challenge-1".to_owned(),
            completed_monotonic_ns: 9_000_000_000,
            speech_started_events: 1,
            provider_submission_attempts: 1,
            provider_submissions_accepted: 1,
            resets: 0,
            observer_errors: 0,
        },
    }
}

fn measurement_binding() -> AecMeasurementBinding {
    AecMeasurementBinding {
        audio_server_id: "server-1".into(),
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

fn input_with_ratios(raw_echo: f64, clean_delta: f64, resolution: f64) -> AecValidationInput {
    let raw_baseline = 1.0;
    let clean_baseline = 2.0;
    let windows = (0..AEC_POWER_WINDOW_COUNT)
        .map(|sequence| {
            let start = sequence as u64 * AEC_SAMPLES_PER_POWER_WINDOW;
            AecPowerWindow {
                sequence: sequence as u64,
                raw_start_sample: start,
                raw_end_sample: start + AEC_SAMPLES_PER_POWER_WINDOW,
                clean_start_sample: start,
                clean_end_sample: start + AEC_SAMPLES_PER_POWER_WINDOW,
                raw_power: raw_baseline + raw_echo,
                clean_power: clean_baseline + clean_delta,
                raw_clipped_samples: 0,
                clean_clipped_samples: 0,
                fixture_dbfs: AEC_FIXTURE_DBFS,
            }
        })
        .collect();
    AecValidationInput {
        metadata: AecDeviceMetadata {
            source_name: "alsa_input.physical".to_owned(),
            sink_name: "alsa_output.physical".to_owned(),
            source_geometry: "desk-left-45cm".to_owned(),
            sink_geometry: "desk-front-80cm".to_owned(),
            sink_port: "analog-output-speaker".to_owned(),
            sink_volume_percent: 40,
        },
        binding: measurement_binding(),
        fixture_acquisition_id: "fixture-1".to_owned(),
        raw_baseline: acquisition("raw-baseline-1", raw_baseline),
        clean_baseline: acquisition("clean-baseline-1", clean_baseline),
        resolution: acquisition("resolution-1", resolution),
        windows,
        observation: observation(),
    }
}

fn valid_input() -> AecValidationInput {
    let raw_echo = 100.0;
    let clean_delta = raw_echo / 10_f64.powf(AEC_ATTENUATION_THRESHOLD_DB / 10.0);
    input_with_ratios(raw_echo, clean_delta, 1.0)
}

fn assert_invalid(input: AecValidationInput) {
    assert_eq!(
        evaluate_aec(input),
        Err(AecValidationError::InvalidValidationInput)
    );
}

#[test]
fn rejects_supplied_shared_floor_false_pass_without_independent_evidence() {
    let mut input = valid_input();
    input.windows[0].raw_power = 846_723.463;
    input.windows[0].clean_power = 59_629.079;
    input.raw_baseline = acquisition("raw-baseline-1", 799_062.337);
    input.clean_baseline.powers.clear();

    assert_invalid(input);
}

#[test]
fn separate_baselines_are_not_interchangeable() {
    let valid = valid_input();
    assert!(evaluate_aec(valid.clone()).unwrap().validated);

    let mut swapped = valid;
    std::mem::swap(&mut swapped.raw_baseline, &mut swapped.clean_baseline);
    let swapped = evaluate_aec(swapped).unwrap();
    assert!(!swapped.attenuation_passed);
    assert!(!swapped.validated);
}

#[test]
fn rejects_invalid_power_measurement_classes() {
    let mut cases = Vec::new();

    let mut silence = valid_input();
    silence.windows[0].raw_power = 0.0;
    cases.push(silence);

    let mut muted_clean = valid_input();
    muted_clean.windows[0].clean_power = 0.0;
    cases.push(muted_clean);

    let mut non_finite = valid_input();
    non_finite.windows[0].raw_power = f64::NAN;
    cases.push(non_finite);

    let mut unstable = valid_input();
    unstable.raw_baseline.powers[4] = 2.0;
    cases.push(unstable);

    let mut malformed_resolution = valid_input();
    malformed_resolution.resolution.powers.pop();
    cases.push(malformed_resolution);

    let mut reused_resolution = valid_input();
    reused_resolution.resolution.acquisition_id = reused_resolution.fixture_acquisition_id.clone();
    cases.push(reused_resolution);

    let mut clipped_raw = valid_input();
    clipped_raw.windows[0].raw_clipped_samples = 1;
    cases.push(clipped_raw);

    let mut clipped_clean = valid_input();
    clipped_clean.windows[0].clean_clipped_samples = 1;
    cases.push(clipped_clean);

    let mut short = valid_input();
    short.windows.pop();
    cases.push(short);

    let mut missing_sample = valid_input();
    missing_sample.windows[0].raw_end_sample -= 1;
    missing_sample.windows[0].clean_end_sample -= 1;
    cases.push(missing_sample);

    let mut missing_middle = valid_input();
    missing_middle.windows[15].sequence += 1;
    cases.push(missing_middle);

    let mut gap = valid_input();
    gap.windows[15].raw_start_sample += 1;
    gap.windows[15].raw_end_sample += 1;
    gap.windows[15].clean_start_sample += 1;
    gap.windows[15].clean_end_sample += 1;
    cases.push(gap);

    let mut mismatch = valid_input();
    mismatch.windows[0].clean_start_sample += 1;
    cases.push(mismatch);

    let mut wrong_level = valid_input();
    wrong_level.windows[0].fixture_dbfs = -19.0;
    cases.push(wrong_level);

    let mut below_resolution = valid_input();
    below_resolution.windows[0].clean_power = 3.0;
    below_resolution.resolution = acquisition("resolution-1", 1.0);
    cases.push(below_resolution);

    for case in cases {
        assert_invalid(case);
    }
}

#[test]
fn threshold_equalities_pass_and_immediately_lower_values_do_not() {
    let exact_raw_snr = 10_f64.powf(AEC_MIN_RAW_SNR_DB / 10.0);
    let exact_resolution = exact_raw_snr / 10_f64.powf(AEC_MIN_RESOLVABLE_ATTENUATION_DB / 10.0);
    let exact_clean = exact_raw_snr / 10_f64.powf(AEC_ATTENUATION_THRESHOLD_DB / 10.0);
    let exact = evaluate_aec(input_with_ratios(
        exact_raw_snr,
        exact_clean,
        exact_resolution,
    ))
    .unwrap();
    assert!(exact.validated);

    let below_raw_db = f64::from_bits(AEC_MIN_RAW_SNR_DB.to_bits() - 1);
    let below_raw = 10_f64.powf(below_raw_db / 10.0);
    assert_invalid(input_with_ratios(below_raw, below_raw / 100.0, 0.01));

    let below_resolution_db = f64::from_bits(AEC_MIN_RESOLVABLE_ATTENUATION_DB.to_bits() - 1);
    let raw_echo = 100.0;
    let resolution = raw_echo / 10_f64.powf(below_resolution_db / 10.0);
    assert_invalid(input_with_ratios(raw_echo, raw_echo / 100.0, resolution));

    let below_attenuation_db = f64::from_bits(AEC_ATTENUATION_THRESHOLD_DB.to_bits() - 1);
    let below_attenuation = evaluate_aec(input_with_ratios(
        raw_echo,
        raw_echo / 10_f64.powf(below_attenuation_db / 10.0),
        1.0,
    ))
    .unwrap();
    assert!(!below_attenuation.attenuation_passed);
    assert!(!below_attenuation.validated);
}

#[test]
fn baseline_variability_cannot_be_reused_as_resolvable_clean_residual() {
    let mut input = input_with_ratios(100.0, 0.1, 0.001);
    input.clean_baseline.powers = vec![990.0, 995.0, 1000.0, 1005.0, 1010.0];
    for window in &mut input.windows {
        window.clean_power = 1000.1;
    }

    assert_invalid(input);
}
