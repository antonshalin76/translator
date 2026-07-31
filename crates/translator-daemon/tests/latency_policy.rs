use translator_core::{AudioDirection, TranslationMode};
use translator_daemon::{
    DuplexLatencyPolicy, LatencySample, LatencyTransitionReason, RuntimeStore,
};

fn sample(first_audio_ms: u32, queue_lag_ms: u32) -> LatencySample {
    LatencySample {
        first_audio_ms,
        last_audio_ms: first_audio_ms + 100,
        queue_lag_ms,
    }
}

fn record_breaching_window(
    policies: &mut DuplexLatencyPolicy,
    direction: AudioDirection,
    epoch_start_ms: u64,
    high_ms: u32,
) {
    for index in 0..20 {
        let value = if index % 3 == 0 { 100 } else { high_ms };
        policies.record_utterance(direction, epoch_start_ms + index * 100, sample(value, 100));
    }
}

#[test]
fn two_completed_breaching_windows_degrade_only_affected_direction() {
    let mut policies = DuplexLatencyPolicy::default();
    record_breaching_window(&mut policies, AudioDirection::Speaker, 0, 3_100);
    assert!(
        policies
            .evaluate_epoch(AudioDirection::Speaker, 60_000)
            .is_none()
    );
    record_breaching_window(&mut policies, AudioDirection::Speaker, 60_000, 3_100);
    let transition = policies
        .evaluate_epoch(AudioDirection::Speaker, 120_000)
        .unwrap();

    assert_eq!(transition.reason, LatencyTransitionReason::WindowBreach);
    assert_eq!(transition.to, TranslationMode::Balanced);
    assert_eq!(
        policies.state(AudioDirection::Microphone).current_mode,
        TranslationMode::QualityFirst
    );
}

#[test]
fn undersized_epoch_resets_breach_hysteresis() {
    let mut policies = DuplexLatencyPolicy::default();
    record_breaching_window(&mut policies, AudioDirection::Microphone, 0, 3_100);
    assert!(
        policies
            .evaluate_epoch(AudioDirection::Microphone, 60_000)
            .is_none()
    );
    for index in 0..19 {
        let value = if index % 3 == 0 { 100 } else { 3_100 };
        policies.record_utterance(
            AudioDirection::Microphone,
            60_000 + index * 100,
            sample(value, 100),
        );
    }
    assert!(
        policies
            .evaluate_epoch(AudioDirection::Microphone, 120_000)
            .is_none()
    );
    record_breaching_window(&mut policies, AudioDirection::Microphone, 120_000, 3_100);
    assert!(
        policies
            .evaluate_epoch(AudioDirection::Microphone, 180_000)
            .is_none()
    );
}

#[test]
fn three_consecutive_utterance_breaches_degrade_during_cooldown() {
    let mut policies = DuplexLatencyPolicy::default();
    for at_ms in 0..3 {
        policies.record_utterance(AudioDirection::Microphone, at_ms, sample(3_001, 100));
    }
    assert_eq!(
        policies.state(AudioDirection::Microphone).current_mode,
        TranslationMode::Balanced
    );

    let mut last = None;
    for at_ms in 3..6 {
        last = policies.record_utterance(AudioDirection::Microphone, at_ms, sample(2_001, 100));
    }
    let transition = last.unwrap();
    assert_eq!(
        transition.reason,
        LatencyTransitionReason::ConsecutiveUtterances
    );
    assert_eq!(transition.to, TranslationMode::StreamingFirst);
}

#[test]
fn successful_utterance_resets_consecutive_breach_tripwire() {
    let mut policies = DuplexLatencyPolicy::default();
    for at_ms in 0..2 {
        policies.record_utterance(AudioDirection::Microphone, at_ms, sample(3_001, 100));
    }
    policies.record_utterance(AudioDirection::Microphone, 2, sample(100, 100));
    for at_ms in 3..5 {
        policies.record_utterance(AudioDirection::Microphone, at_ms, sample(3_001, 100));
    }
    assert_eq!(
        policies.state(AudioDirection::Microphone).current_mode,
        TranslationMode::QualityFirst
    );
}

#[test]
fn stale_or_duplicate_observations_do_not_advance_tripwires() {
    let mut policies = DuplexLatencyPolicy::default();
    policies.record_utterance(AudioDirection::Microphone, 100, sample(3_100, 100));
    policies.record_utterance(AudioDirection::Microphone, 100, sample(3_100, 100));
    policies.record_utterance(AudioDirection::Microphone, 99, sample(3_100, 100));
    policies.record_utterance(AudioDirection::Microphone, 101, sample(3_100, 100));
    assert_eq!(
        policies.state(AudioDirection::Microphone).current_mode,
        TranslationMode::QualityFirst
    );

    policies.observe_queue_lag(AudioDirection::Speaker, 1_000, Some(501));
    policies.observe_queue_lag(AudioDirection::Speaker, 900, None);
    policies.observe_queue_lag(AudioDirection::Speaker, 1_000, None);
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 2_999, Some(501))
            .is_none()
    );
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 3_000, Some(501))
            .is_some()
    );
}

#[test]
fn runtime_store_uses_the_authoritative_policy_for_snapshot_transitions() {
    let store = RuntimeStore::default();
    for at_ms in 1..=3 {
        store.record_latency_utterance(AudioDirection::Microphone, at_ms, sample(3_100, 100));
    }
    let snapshot = store.snapshot();
    assert_eq!(
        snapshot.latency_policy[0].current_mode,
        TranslationMode::Balanced
    );
    assert_eq!(
        snapshot.latency_policy[1].current_mode,
        TranslationMode::QualityFirst
    );
}

#[test]
fn completed_epoch_is_evaluated_once_and_samples_use_half_open_boundaries() {
    let mut policies = DuplexLatencyPolicy::default();
    record_breaching_window(&mut policies, AudioDirection::Speaker, 0, 3_100);
    assert!(
        policies
            .evaluate_epoch(AudioDirection::Speaker, 60_000)
            .is_none()
    );
    assert!(
        policies
            .evaluate_epoch(AudioDirection::Speaker, 60_000)
            .is_none()
    );
    assert!(
        policies
            .evaluate_epoch(AudioDirection::Speaker, 30_000)
            .is_none()
    );

    policies.record_utterance(AudioDirection::Speaker, 60_000, sample(100, 100));
    for index in 1..20 {
        let value = if index % 3 == 0 { 100 } else { 3_100 };
        policies.record_utterance(
            AudioDirection::Speaker,
            60_000 + index * 100,
            sample(value, 100),
        );
    }
    let transition = policies
        .evaluate_epoch(AudioDirection::Speaker, 120_000)
        .unwrap();
    assert_eq!(transition.to, TranslationMode::Balanced);
}

#[test]
fn sustained_queue_lag_degrades_after_two_seconds_and_missing_sample_resets_timer() {
    let mut policies = DuplexLatencyPolicy::default();
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 0, Some(501))
            .is_none()
    );
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 1_900, Some(501))
            .is_none()
    );
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 1_950, None)
            .is_none()
    );
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 2_000, Some(501))
            .is_none()
    );
    let transition = policies
        .observe_queue_lag(AudioDirection::Speaker, 4_000, Some(501))
        .unwrap();
    assert_eq!(
        transition.reason,
        LatencyTransitionReason::SustainedQueueLag
    );
    assert_eq!(transition.to, TranslationMode::Balanced);
}

#[test]
fn queue_lag_at_threshold_resets_sustained_timer() {
    let mut policies = DuplexLatencyPolicy::default();
    policies.observe_queue_lag(AudioDirection::Speaker, 0, Some(501));
    policies.observe_queue_lag(AudioDirection::Speaker, 1_900, Some(501));
    policies.observe_queue_lag(AudioDirection::Speaker, 1_950, Some(500));
    policies.observe_queue_lag(AudioDirection::Speaker, 2_000, Some(501));
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 3_999, Some(501))
            .is_none()
    );
}

#[test]
fn recovery_requires_five_stable_windows_and_elapsed_cooldown() {
    let mut policies = DuplexLatencyPolicy::default();
    for at_ms in 0..3 {
        policies.record_utterance(AudioDirection::Speaker, at_ms, sample(3_001, 100));
    }
    for at_ms in 3..6 {
        policies.record_utterance(AudioDirection::Speaker, at_ms, sample(2_001, 100));
    }
    assert_eq!(
        policies.state(AudioDirection::Speaker).current_mode,
        TranslationMode::StreamingFirst
    );

    for epoch in 0..4 {
        let start = 180_000 + epoch * 60_000;
        for index in 0..20 {
            policies.record_utterance(
                AudioDirection::Speaker,
                start + index * 100,
                sample(500, 50),
            );
        }
        assert!(
            policies
                .evaluate_epoch(AudioDirection::Speaker, start + 60_000)
                .is_none()
        );
    }
    let start = 420_000;
    for index in 0..20 {
        policies.record_utterance(
            AudioDirection::Speaker,
            start + index * 100,
            sample(500, 50),
        );
    }
    let transition = policies
        .evaluate_epoch(AudioDirection::Speaker, start + 60_000)
        .unwrap();
    assert_eq!(transition.reason, LatencyTransitionReason::StableRecovery);
    assert_eq!(transition.to, TranslationMode::Balanced);

    for epoch in 0..5 {
        let start = 600_000 + epoch * 60_000;
        for index in 0..20 {
            policies.record_utterance(
                AudioDirection::Speaker,
                start + index * 100,
                sample(500, 50),
            );
        }
        let transition = policies.evaluate_epoch(AudioDirection::Speaker, start + 60_000);
        if epoch < 4 {
            assert!(transition.is_none());
        } else {
            assert_eq!(transition.unwrap().to, TranslationMode::QualityFirst);
        }
    }
}

#[test]
fn streaming_first_is_terminal_for_degradation_tripwires() {
    let mut policies = DuplexLatencyPolicy::default();
    for at_ms in 0..3 {
        policies.record_utterance(AudioDirection::Speaker, at_ms, sample(3_001, 100));
    }
    for at_ms in 3..6 {
        policies.record_utterance(AudioDirection::Speaker, at_ms, sample(2_001, 100));
    }
    for at_ms in 6..20 {
        assert!(
            policies
                .record_utterance(AudioDirection::Speaker, at_ms, sample(10_000, 1_000))
                .is_none()
        );
    }
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 100, Some(1_000))
            .is_none()
    );
    assert!(
        policies
            .observe_queue_lag(AudioDirection::Speaker, 2_100, Some(1_000))
            .is_none()
    );
    assert_eq!(
        policies.state(AudioDirection::Speaker).current_mode,
        TranslationMode::StreamingFirst
    );
}

#[test]
fn p95_uses_nearest_rank_and_directions_keep_independent_samples() {
    let mut policies = DuplexLatencyPolicy::default();
    for index in 0..20 {
        policies.record_utterance(
            AudioDirection::Microphone,
            index * 100,
            sample(if index == 19 { 9_999 } else { 1_000 }, 10),
        );
        policies.record_utterance(AudioDirection::Speaker, index * 100, sample(200, 20));
    }
    policies.evaluate_epoch(AudioDirection::Microphone, 60_000);
    policies.evaluate_epoch(AudioDirection::Speaker, 60_000);

    assert_eq!(
        policies
            .state(AudioDirection::Microphone)
            .p95_first_audio_ms,
        1_000
    );
    assert_eq!(
        policies.state(AudioDirection::Speaker).p95_first_audio_ms,
        200
    );
    assert_eq!(
        policies.state(AudioDirection::Microphone).p95_last_audio_ms,
        1_100
    );
    assert_eq!(
        policies.state(AudioDirection::Microphone).p95_queue_lag_ms,
        10
    );
}
