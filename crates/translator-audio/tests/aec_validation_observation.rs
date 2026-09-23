use translator_audio::{
    AEC_FIXTURE_DBFS, AEC_OBSERVATION_DURATION_NS, AEC_OBSERVATION_FRAME_COUNT,
    AEC_OBSERVATION_FRAME_SAMPLES, AEC_POSITIVE_CONTROL_MAX_GAP_NS, AEC_POWER_WINDOW_COUNT,
    AEC_SAMPLES_PER_POWER_WINDOW, AecDeviceMetadata, AecMeasurementBinding, AecObservationEvidence,
    AecPositiveControl, AecPowerAcquisition, AecPowerWindow, AecValidationError,
    AecValidationInput, evaluate_aec,
};

fn valid_input() -> AecValidationInput {
    let acquisition = |id: &str, power| AecPowerAcquisition {
        acquisition_id: id.to_owned(),
        samples_per_window: AEC_SAMPLES_PER_POWER_WINDOW,
        powers: vec![power; 5],
    };
    let windows = (0..AEC_POWER_WINDOW_COUNT)
        .map(|sequence| {
            let start = sequence as u64 * AEC_SAMPLES_PER_POWER_WINDOW;
            AecPowerWindow {
                sequence: sequence as u64,
                raw_start_sample: start,
                raw_end_sample: start + AEC_SAMPLES_PER_POWER_WINDOW,
                clean_start_sample: start,
                clean_end_sample: start + AEC_SAMPLES_PER_POWER_WINDOW,
                raw_power: 101.0,
                clean_power: 3.0,
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
        binding: AecMeasurementBinding {
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
        },
        fixture_acquisition_id: "fixture-1".to_owned(),
        raw_baseline: acquisition("raw-baseline-1", 1.0),
        clean_baseline: acquisition("clean-baseline-1", 1.0),
        resolution: acquisition("resolution-1", 1.0),
        windows,
        observation: AecObservationEvidence {
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
            vad_events_before: 7,
            vad_events_after: 7,
            provider_attempts_before: 4,
            provider_attempts_after: 4,
            provider_accepted_before: 4,
            provider_accepted_after: 4,
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
        },
    }
}

#[test]
fn zero_activity_without_complete_frame_coverage_is_invalid() {
    let mut input = valid_input();
    input.observation.processed_frames -= 1;

    assert_eq!(
        evaluate_aec(input),
        Err(AecValidationError::InvalidValidationInput)
    );
}

fn assert_invalid(input: AecValidationInput) {
    assert_eq!(
        evaluate_aec(input),
        Err(AecValidationError::InvalidValidationInput)
    );
}

#[test]
fn complete_zero_activity_observation_passes() {
    let record = evaluate_aec(valid_input()).unwrap();

    assert!(record.vad_passed);
    assert!(record.provider_passed);
    assert!(record.validated);
}

#[test]
fn observation_freshness_boundary_is_exact() {
    let mut exact = valid_input();
    exact.observation.positive_control.completed_monotonic_ns =
        exact.observation.started_monotonic_ns - AEC_POSITIVE_CONTROL_MAX_GAP_NS;
    assert!(evaluate_aec(exact).unwrap().validated);

    let mut stale = valid_input();
    stale.observation.positive_control.completed_monotonic_ns =
        stale.observation.started_monotonic_ns - AEC_POSITIVE_CONTROL_MAX_GAP_NS - 1;
    assert_invalid(stale);
}

#[test]
fn rejects_incomplete_or_inconsistent_observation_evidence() {
    let mut cases = Vec::new();

    let mut generation = valid_input();
    generation.observation.positive_control.observer_generation = "other".to_owned();
    cases.push(generation);

    let mut attempt = valid_input();
    attempt.observation.positive_control.calibration_attempt_id = "other".to_owned();
    cases.push(attempt);

    let mut challenge = valid_input();
    challenge.observation.positive_control.challenge_id = "other".to_owned();
    cases.push(challenge);

    let mut duration = valid_input();
    duration.observation.ended_monotonic_ns -= 1;
    cases.push(duration);

    let mut reset = valid_input();
    reset.observation.resets = 1;
    cases.push(reset);

    let mut counter_reset = valid_input();
    counter_reset.observation.vad_events_after = 6;
    cases.push(counter_reset);

    let mut accepted_without_attempt = valid_input();
    accepted_without_attempt.observation.provider_accepted_after += 1;
    cases.push(accepted_without_attempt);

    let mut missing_frame = valid_input();
    missing_frame.observation.processed_frames -= 1;
    cases.push(missing_frame);

    let mut substituted_frame = valid_input();
    substituted_frame.observation.frame_gaps = 1;
    substituted_frame.observation.duplicate_frames = 1;
    cases.push(substituted_frame);

    let mut reordered = valid_input();
    reordered.observation.out_of_order_frames = 1;
    cases.push(reordered);

    let mut wrong_frame_size = valid_input();
    wrong_frame_size.observation.samples_per_frame -= 1;
    cases.push(wrong_frame_size);

    let mut dropped = valid_input();
    dropped.observation.dropped_frames = 1;
    cases.push(dropped);

    let mut observer_error = valid_input();
    observer_error.observation.observer_errors = 1;
    cases.push(observer_error);

    let mut terminated = valid_input();
    terminated.observation.terminated_early = true;
    cases.push(terminated);

    let mut missing_positive_vad = valid_input();
    missing_positive_vad
        .observation
        .positive_control
        .speech_started_events = 0;
    cases.push(missing_positive_vad);

    let mut positive_control_reset = valid_input();
    positive_control_reset.observation.positive_control.resets = 1;
    cases.push(positive_control_reset);

    for case in cases {
        assert_invalid(case);
    }
}

#[test]
fn invalid_evidence_takes_precedence_over_observed_activity() {
    let mut input = valid_input();
    input.observation.processed_frames -= 1;
    input.observation.vad_events_after += 1;
    input.observation.provider_attempts_after += 1;

    assert_invalid(input);
}

#[test]
fn valid_activity_is_a_measured_failure_with_independent_gates() {
    let mut input = valid_input();
    let below_attenuation_db = 14.0;
    let clean_delta = 100.0 / 10_f64.powf(below_attenuation_db / 10.0);
    for window in &mut input.windows {
        window.clean_power = 1.0 + clean_delta;
    }
    input.observation.vad_events_after += 1;
    input.observation.provider_attempts_after += 1;

    let record = evaluate_aec(input).unwrap();

    assert!(!record.attenuation_passed);
    assert!(!record.vad_passed);
    assert!(!record.provider_passed);
    assert!(!record.validated);
}

#[test]
fn continuous_stream_sequence_is_relative_to_the_scored_interval() {
    let mut input = valid_input();
    input.observation.first_frame_sequence = 10_000;
    input.observation.last_frame_sequence =
        input.observation.first_frame_sequence + AEC_OBSERVATION_FRAME_COUNT - 1;

    assert!(evaluate_aec(input).unwrap().validated);
}
