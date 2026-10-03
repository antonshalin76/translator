use std::collections::VecDeque;

use translator_audio::{
    AEC_SAMPLES_PER_POWER_WINDOW, AecChannelFrame, AecEvidenceDisposition, AecMeasurementError,
    AecMeasurementSource, AecPairedFrame, AecSampleVerifier, IndependentParecMeasurementSource,
};

const SAMPLES: usize = AEC_SAMPLES_PER_POWER_WINDOW as usize;

struct SyntheticSource(VecDeque<AecPairedFrame>);

impl AecMeasurementSource for SyntheticSource {
    fn next_pair(&mut self) -> Result<Option<AecPairedFrame>, AecMeasurementError> {
        Ok(self.0.pop_front())
    }

    fn close(&mut self) -> Result<(), AecMeasurementError> {
        Ok(())
    }
}

fn phase(index: usize) -> &'static str {
    match index {
        0..5 => "raw-baseline",
        5..10 => "clean-baseline",
        10..15 => "resolution",
        _ => "fixture",
    }
}

fn pair(index: usize) -> AecPairedFrame {
    let fixture = index >= 15;
    let channel = |stream_id: &str, level| AecChannelFrame {
        clock_id: "one-clock".into(),
        generation: "one-generation".into(),
        stream_id: stream_id.into(),
        acquisition_id: phase(index).into(),
        frame_id: index as u64,
        start_sample: index as u64 * AEC_SAMPLES_PER_POWER_WINDOW,
        sample_rate_hz: 48_000,
        lost_frames: 0,
        samples: vec![level; SAMPLES],
    };
    AecPairedFrame {
        raw: channel("raw", if fixture { 1_000 } else { 100 }),
        clean: channel("clean", if fixture { 150 } else { 100 }),
    }
}

fn source() -> SyntheticSource {
    SyntheticSource((0..45).map(pair).collect())
}

#[test]
fn synthetic_dual_channel_sample_evidence_is_not_proof() {
    let evidence = AecSampleVerifier::collect_from(&mut source()).unwrap();
    assert_eq!(
        evidence.disposition(),
        AecEvidenceDisposition::NonAdmissible
    );
    let provenance = evidence.provenance();
    assert_eq!(provenance.clock_id(), "one-clock");
    assert_eq!(provenance.generation(), "one-generation");
    assert_eq!(provenance.stream_ids(), ("raw", "clean"));
    assert_eq!(
        provenance.acquisition_ids(),
        &["raw-baseline", "clean-baseline", "resolution", "fixture"]
    );
    assert_eq!(provenance.frame_range(), (0, 44));
    assert_eq!(
        provenance.sample_range(),
        (0, 45 * AEC_SAMPLES_PER_POWER_WINDOW)
    );
    // Sample windows remain in the source's absolute clock, not validator-relative time.
    assert_eq!(
        evidence.fixture_windows()[0].start_sample,
        15 * AEC_SAMPLES_PER_POWER_WINDOW
    );
    assert_eq!(evidence.raw_baseline_powers(), &[10_000.0; 5]);
    assert_eq!(evidence.clean_baseline_powers(), &[10_000.0; 5]);
    assert_eq!(evidence.resolution_powers(), &[10_000.0; 5]);
    assert_eq!(evidence.fixture_windows().len(), 30);
    assert_eq!(evidence.fixture_windows()[0].raw_power, 1_000_000.0);
    assert_eq!(evidence.fixture_windows()[0].clean_power, 22_500.0);

    let mut changed = source();
    changed.0[15].raw.samples.fill(2_000);
    changed.0[15].clean.samples.fill(200);
    let changed = AecSampleVerifier::collect_from(&mut changed).unwrap();
    assert_eq!(changed.fixture_windows()[0].raw_power, 4_000_000.0);
    assert_eq!(changed.fixture_windows()[0].clean_power, 40_000.0);
}

#[test]
fn independent_parec_is_explicitly_unavailable() {
    assert!(matches!(
        AecSampleVerifier::collect_from(&mut IndependentParecMeasurementSource),
        Err(AecMeasurementError::MeasurementUnavailable)
    ));
}

#[test]
fn malformed_or_replayed_pcm_never_yields_evidence() {
    type Fault = (&'static str, fn(&mut SyntheticSource));
    let faults: &[Fault] = &[
        ("channel clock", |s| s.0[16].clean.clock_id = "other".into()),
        ("stream clock", |s| {
            s.0[16].raw.clock_id = "other".into();
            s.0[16].clean.clock_id = "other".into();
        }),
        ("channel generation", |s| {
            s.0[16].clean.generation = "other".into()
        }),
        ("stream generation", |s| {
            s.0[16].raw.generation = "other".into();
            s.0[16].clean.generation = "other".into();
        }),
        ("channel frame id", |s| s.0[16].clean.frame_id += 1),
        ("duplicate frame id", |s| {
            s.0[16].raw.frame_id -= 1;
            s.0[16].clean.frame_id -= 1;
        }),
        ("skipped frame id", |s| {
            s.0[16].raw.frame_id += 1;
            s.0[16].clean.frame_id += 1;
        }),
        ("channel phase", |s| {
            s.0[16].clean.acquisition_id = "other".into()
        }),
        ("reused phase", |s| {
            s.0[15].raw.acquisition_id = "raw-baseline".into();
            s.0[15].clean.acquisition_id = "raw-baseline".into();
        }),
        ("channel skew", |s| s.0[16].clean.start_sample += 1),
        ("stream drift", |s| {
            s.0[16].raw.start_sample += 1;
            s.0[16].clean.start_sample += 1;
        }),
        ("reordered pairs", |s| s.0.swap(16, 17)),
        ("replayed pair", |s| s.0[16] = s.0[15].clone()),
        ("dropped pair", |s| {
            s.0.remove(16);
        }),
        ("raw rate", |s| s.0[16].raw.sample_rate_hz = 16_000),
        ("clean rate", |s| s.0[16].clean.sample_rate_hz = 16_000),
        ("raw sample count", |s| {
            s.0[16].raw.samples.pop();
        }),
        ("clean sample count", |s| {
            s.0[16].clean.samples.pop();
        }),
        ("raw positive clipping", |s| {
            s.0[16].raw.samples[0] = i16::MAX
        }),
        ("clean negative clipping", |s| {
            s.0[16].clean.samples[0] = i16::MIN
        }),
        ("raw loss", |s| s.0[16].raw.lost_frames = 1),
        ("clean loss", |s| s.0[16].clean.lost_frames = 1),
        ("raw stream switch", |s| {
            s.0[16].raw.stream_id = "raw-next".into()
        }),
        ("blank clock", |s| {
            for frame in &mut s.0 {
                frame.raw.clock_id.clear();
                frame.clean.clock_id.clear();
            }
        }),
        ("blank generation", |s| {
            for frame in &mut s.0 {
                frame.raw.generation.clear();
                frame.clean.generation.clear();
            }
        }),
        ("blank acquisition", |s| {
            for frame in &mut s.0 {
                frame.raw.acquisition_id.clear();
                frame.clean.acquisition_id.clear();
            }
        }),
        ("blank stream", |s| {
            for frame in &mut s.0 {
                frame.raw.stream_id.clear();
                frame.clean.stream_id.clear();
            }
        }),
        ("same stream", |s| s.0[16].raw.stream_id = "clean".into()),
        ("range overflow", |s| {
            s.0[16].raw.start_sample = u64::MAX;
            s.0[16].clean.start_sample = u64::MAX;
        }),
    ];
    for (name, fault) in faults {
        let mut source = source();
        fault(&mut source);
        assert!(
            AecSampleVerifier::collect_from(&mut source).is_err(),
            "fault unexpectedly admitted: {name}"
        );
    }
}

#[test]
fn incomplete_or_extra_frames_fail_closed() {
    let mut incomplete = source();
    incomplete.0.truncate(8);
    assert!(matches!(
        AecSampleVerifier::collect_from(&mut incomplete),
        Err(AecMeasurementError::Incomplete)
    ));

    let mut extra = source();
    extra.0.push_back(pair(45));
    assert!(matches!(
        AecSampleVerifier::collect_from(&mut extra),
        Err(AecMeasurementError::Discontinuous)
    ));

    let mut verifier = AecSampleVerifier::new();
    for index in 0..45 {
        verifier.push(pair(index)).unwrap();
    }
    assert!(verifier.push(pair(45)).is_err());
    assert!(matches!(
        verifier.finish(),
        Err(AecMeasurementError::Invalidated)
    ));
}

#[test]
fn cancellation_or_invalid_frame_discards_partial_evidence() {
    let mut verifier = AecSampleVerifier::new();
    verifier.push(pair(0)).unwrap();
    verifier.cancel();
    assert!(matches!(
        verifier.push(pair(1)),
        Err(AecMeasurementError::Cancelled)
    ));
    assert!(matches!(
        verifier.finish(),
        Err(AecMeasurementError::Cancelled)
    ));

    let mut verifier = AecSampleVerifier::new();
    verifier.push(pair(0)).unwrap();
    let mut bad = pair(1);
    bad.clean.clock_id = "other".into();
    assert!(verifier.push(bad).is_err());
    assert!(matches!(
        verifier.push(pair(1)),
        Err(AecMeasurementError::Invalidated)
    ));
    assert!(matches!(
        verifier.finish(),
        Err(AecMeasurementError::Invalidated)
    ));
}

struct TrackedSource {
    frames: VecDeque<AecPairedFrame>,
    reads: usize,
    failure_at: Option<(usize, AecMeasurementError)>,
    close_calls: usize,
    close_fails: bool,
}

impl TrackedSource {
    fn complete() -> Self {
        Self {
            frames: source().0,
            reads: 0,
            failure_at: None,
            close_calls: 0,
            close_fails: false,
        }
    }
}

impl AecMeasurementSource for TrackedSource {
    fn next_pair(&mut self) -> Result<Option<AecPairedFrame>, AecMeasurementError> {
        if let Some((at, error)) = self.failure_at {
            if self.reads == at {
                return Err(error);
            }
        }
        self.reads += 1;
        Ok(self.frames.pop_front())
    }

    fn close(&mut self) -> Result<(), AecMeasurementError> {
        self.close_calls += 1;
        if self.close_fails {
            Err(AecMeasurementError::SourceFailed)
        } else {
            Ok(())
        }
    }
}

#[test]
fn source_error_cancellation_and_failed_cleanup_never_yield_evidence() {
    for error in [
        AecMeasurementError::Cancelled,
        AecMeasurementError::SourceFailed,
    ] {
        let mut source = TrackedSource::complete();
        source.failure_at = Some((1, error));
        assert_eq!(AecSampleVerifier::collect_from(&mut source), Err(error));
        assert_eq!(source.close_calls, 1);
    }

    let mut source = TrackedSource::complete();
    source.close_fails = true;
    assert_eq!(
        AecSampleVerifier::collect_from(&mut source),
        Err(AecMeasurementError::SourceFailed)
    );
    assert_eq!(source.close_calls, 1);
}
