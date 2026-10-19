use translator_core::{AudioDirection, LatencyPolicyState, ProviderId, TranslationMode};

use crate::direction_session::{provider_id, provider_mode};

const EPOCH_MS: u64 = 60_000;
const MINIMUM_SAMPLES: usize = 20;
const DEGRADE_WINDOWS: u32 = 2;
const RECOVER_WINDOWS: u32 = 5;
const COOLDOWN_MS: u64 = 120_000;
const QUEUE_TRIPWIRE_MS: u32 = 500;
const QUEUE_TRIPWIRE_DURATION_MS: u64 = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencySample {
    pub first_audio_ms: u32,
    pub last_audio_ms: u32,
    pub queue_lag_ms: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyTransitionReason {
    WindowBreach,
    StableRecovery,
    ConsecutiveUtterances,
    SustainedQueueLag,
    ManualPolicyChange,
}

impl LatencyTransitionReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::WindowBreach => "window_breach",
            Self::StableRecovery => "stable_recovery",
            Self::ConsecutiveUtterances => "consecutive_utterances",
            Self::SustainedQueueLag => "sustained_queue_lag",
            Self::ManualPolicyChange => "manual_policy_change",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyTransition {
    pub direction: AudioDirection,
    pub from: TranslationMode,
    pub to: TranslationMode,
    pub reason: LatencyTransitionReason,
    pub at_ms: u64,
}

#[derive(Debug, Clone, Copy)]
struct TimedSample {
    at_ms: u64,
    value: LatencySample,
}

#[derive(Debug)]
struct DirectionPolicy {
    provider: ProviderId,
    state: LatencyPolicyState,
    samples: Vec<TimedSample>,
    last_evaluated_end_ms: u64,
    breach_windows: u32,
    recovery_windows: u32,
    consecutive_utterance_breaches: u32,
    queue_breach_started_ms: Option<u64>,
    cooldown_until_ms: u64,
    last_utterance_at_ms: Option<u64>,
    last_queue_observation_at_ms: Option<u64>,
}

impl DirectionPolicy {
    fn new(direction: AudioDirection) -> Self {
        Self {
            provider: ProviderId::Local,
            state: LatencyPolicyState::new(
                direction,
                TranslationMode::QualityFirst,
                0,
                0,
                0,
                None,
                None,
            ),
            samples: Vec::new(),
            last_evaluated_end_ms: 0,
            breach_windows: 0,
            recovery_windows: 0,
            consecutive_utterance_breaches: 0,
            queue_breach_started_ms: None,
            cooldown_until_ms: 0,
            last_utterance_at_ms: None,
            last_queue_observation_at_ms: None,
        }
    }

    fn record_utterance(&mut self, at_ms: u64, sample: LatencySample) -> Option<LatencyTransition> {
        if self.last_utterance_at_ms.is_some_and(|last| at_ms <= last) {
            return None;
        }
        self.last_utterance_at_ms = Some(at_ms);
        self.samples.push(TimedSample {
            at_ms,
            value: sample,
        });
        if next_degraded_mode(self.state.current_mode).is_none() {
            self.consecutive_utterance_breaches = 0;
            return None;
        }
        if sample.first_audio_ms > thresholds(self.provider, self.state.current_mode).0 {
            self.consecutive_utterance_breaches =
                self.consecutive_utterance_breaches.saturating_add(1);
        } else {
            self.consecutive_utterance_breaches = 0;
        }
        if self.consecutive_utterance_breaches >= 3 {
            return self.transition(at_ms, LatencyTransitionReason::ConsecutiveUtterances, true);
        }
        None
    }

    fn observe_queue_lag(
        &mut self,
        at_ms: u64,
        queue_lag_ms: Option<u32>,
    ) -> Option<LatencyTransition> {
        if self
            .last_queue_observation_at_ms
            .is_some_and(|last| at_ms <= last)
        {
            return None;
        }
        self.last_queue_observation_at_ms = Some(at_ms);
        if next_degraded_mode(self.state.current_mode).is_none() {
            self.queue_breach_started_ms = None;
            return None;
        }
        if queue_lag_ms.is_none_or(|lag| lag <= QUEUE_TRIPWIRE_MS) {
            self.queue_breach_started_ms = None;
            return None;
        }
        let started = *self.queue_breach_started_ms.get_or_insert(at_ms);
        if at_ms.saturating_sub(started) >= QUEUE_TRIPWIRE_DURATION_MS {
            return self.transition(at_ms, LatencyTransitionReason::SustainedQueueLag, true);
        }
        None
    }

    fn evaluate_epoch(&mut self, epoch_end_ms: u64) -> Option<LatencyTransition> {
        if epoch_end_ms == 0
            || !epoch_end_ms.is_multiple_of(EPOCH_MS)
            || epoch_end_ms <= self.last_evaluated_end_ms
        {
            return None;
        }
        self.last_evaluated_end_ms = epoch_end_ms;
        let epoch_start_ms = epoch_end_ms - EPOCH_MS;
        let epoch = self
            .samples
            .iter()
            .filter(|sample| sample.at_ms >= epoch_start_ms && sample.at_ms < epoch_end_ms)
            .map(|sample| sample.value)
            .collect::<Vec<_>>();
        self.samples.retain(|sample| sample.at_ms >= epoch_end_ms);

        if epoch.len() < MINIMUM_SAMPLES {
            self.breach_windows = 0;
            self.recovery_windows = 0;
            return None;
        }

        self.state.p95_first_audio_ms =
            nearest_rank_p95(epoch.iter().map(|sample| sample.first_audio_ms));
        self.state.p95_last_audio_ms =
            nearest_rank_p95(epoch.iter().map(|sample| sample.last_audio_ms));
        self.state.p95_queue_lag_ms =
            nearest_rank_p95(epoch.iter().map(|sample| sample.queue_lag_ms));
        let (first_limit, queue_limit) = thresholds(self.provider, self.state.current_mode);
        let breached = self.state.p95_first_audio_ms > first_limit
            || self.state.p95_queue_lag_ms > queue_limit;

        if breached && next_degraded_mode(self.state.current_mode).is_some() {
            self.breach_windows = self.breach_windows.saturating_add(1);
            self.recovery_windows = 0;
            if self.breach_windows >= DEGRADE_WINDOWS {
                return self.transition(epoch_end_ms, LatencyTransitionReason::WindowBreach, true);
            }
        } else if !breached && next_recovered_mode(self.state.current_mode).is_some() {
            self.recovery_windows = self.recovery_windows.saturating_add(1);
            self.breach_windows = 0;
            if self.recovery_windows >= RECOVER_WINDOWS && epoch_end_ms >= self.cooldown_until_ms {
                return self.transition(
                    epoch_end_ms,
                    LatencyTransitionReason::StableRecovery,
                    false,
                );
            }
        } else {
            self.breach_windows = 0;
            self.recovery_windows = 0;
        }
        None
    }

    fn transition(
        &mut self,
        at_ms: u64,
        reason: LatencyTransitionReason,
        degrade: bool,
    ) -> Option<LatencyTransition> {
        // The fixed cloud cascade has no measured faster mode to switch to.
        if self.provider == ProviderId::Openai {
            return None;
        }
        let from = self.state.current_mode;
        let to = if degrade {
            next_degraded_mode(from)?
        } else {
            next_recovered_mode(from)?
        };
        let first = self.state.p95_first_audio_ms;
        let last = self.state.p95_last_audio_ms;
        let queue = self.state.p95_queue_lag_ms;
        self.state = LatencyPolicyState::new(
            self.state.direction_id,
            to,
            first,
            last,
            queue,
            Some(format!("{at_ms}ms")),
            Some(reason.as_str().to_owned()),
        );
        self.samples.clear();
        self.breach_windows = 0;
        self.recovery_windows = 0;
        self.consecutive_utterance_breaches = 0;
        self.queue_breach_started_ms = None;
        self.cooldown_until_ms = at_ms.saturating_add(COOLDOWN_MS);
        Some(LatencyTransition {
            direction: self.state.direction_id,
            from,
            to,
            reason,
            at_ms,
        })
    }

    fn force_mode(&mut self, at_ms: u64, mode: TranslationMode) -> Option<LatencyTransition> {
        let from = self.state.current_mode;
        if from == mode
            && self
                .state
                .reason
                .as_deref()
                .is_none_or(|reason| reason == LatencyTransitionReason::ManualPolicyChange.as_str())
        {
            return None;
        }
        let first = self.state.p95_first_audio_ms;
        let last = self.state.p95_last_audio_ms;
        let queue = self.state.p95_queue_lag_ms;
        self.state = LatencyPolicyState::new(
            self.state.direction_id,
            mode,
            first,
            last,
            queue,
            Some(format!("{at_ms}ms")),
            Some(
                LatencyTransitionReason::ManualPolicyChange
                    .as_str()
                    .to_owned(),
            ),
        );
        self.samples.clear();
        self.breach_windows = 0;
        self.recovery_windows = 0;
        self.consecutive_utterance_breaches = 0;
        self.queue_breach_started_ms = None;
        self.cooldown_until_ms = at_ms.saturating_add(COOLDOWN_MS);
        Some(LatencyTransition {
            direction: self.state.direction_id,
            from,
            to: mode,
            reason: LatencyTransitionReason::ManualPolicyChange,
            at_ms,
        })
    }
}

#[derive(Debug)]
pub struct DuplexLatencyPolicy {
    microphone: DirectionPolicy,
    speaker: DirectionPolicy,
}

impl Default for DuplexLatencyPolicy {
    fn default() -> Self {
        Self {
            microphone: DirectionPolicy::new(AudioDirection::Microphone),
            speaker: DirectionPolicy::new(AudioDirection::Speaker),
        }
    }
}

impl DuplexLatencyPolicy {
    pub(crate) fn set_provider(&mut self, provider: ProviderId) {
        for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
            let policy = self.policy_mut(direction);
            if policy.provider != provider {
                let mode = if provider == ProviderId::Openai
                    && policy.state.reason.as_deref().is_some_and(|reason| {
                        reason != LatencyTransitionReason::ManualPolicyChange.as_str()
                    }) {
                    TranslationMode::QualityFirst
                } else {
                    policy.state.current_mode
                };
                *policy = DirectionPolicy::new(direction);
                policy.provider = provider;
                policy.state.current_mode = mode;
            }
        }
    }

    pub fn record_utterance(
        &mut self,
        direction: AudioDirection,
        at_ms: u64,
        sample: LatencySample,
    ) -> Option<LatencyTransition> {
        self.policy_mut(direction).record_utterance(at_ms, sample)
    }

    pub fn observe_queue_lag(
        &mut self,
        direction: AudioDirection,
        at_ms: u64,
        queue_lag_ms: Option<u32>,
    ) -> Option<LatencyTransition> {
        self.policy_mut(direction)
            .observe_queue_lag(at_ms, queue_lag_ms)
    }

    pub fn evaluate_epoch(
        &mut self,
        direction: AudioDirection,
        epoch_end_ms: u64,
    ) -> Option<LatencyTransition> {
        self.policy_mut(direction).evaluate_epoch(epoch_end_ms)
    }

    pub fn state(&self, direction: AudioDirection) -> &LatencyPolicyState {
        match direction {
            AudioDirection::Microphone => &self.microphone.state,
            AudioDirection::Speaker => &self.speaker.state,
        }
    }

    pub fn force_mode(
        &mut self,
        direction: AudioDirection,
        at_ms: u64,
        mode: TranslationMode,
    ) -> Option<LatencyTransition> {
        self.policy_mut(direction).force_mode(at_ms, mode)
    }

    fn policy_mut(&mut self, direction: AudioDirection) -> &mut DirectionPolicy {
        match direction {
            AudioDirection::Microphone => &mut self.microphone,
            AudioDirection::Speaker => &mut self.speaker,
        }
    }
}

fn nearest_rank_p95(values: impl Iterator<Item = u32>) -> u32 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    let rank = (95 * values.len()).div_ceil(100);
    values[rank.saturating_sub(1)]
}

const fn thresholds(provider: ProviderId, mode: TranslationMode) -> (u32, u32) {
    let queue = match mode {
        TranslationMode::QualityFirst => 500,
        TranslationMode::Balanced => 350,
        TranslationMode::StreamingFirst => 250,
    };
    (
        translator_ipc::first_audio_target_ms(provider_id(provider), provider_mode(mode)),
        queue,
    )
}

const fn next_degraded_mode(mode: TranslationMode) -> Option<TranslationMode> {
    match mode {
        TranslationMode::QualityFirst => Some(TranslationMode::Balanced),
        TranslationMode::Balanced => Some(TranslationMode::StreamingFirst),
        TranslationMode::StreamingFirst => None,
    }
}

const fn next_recovered_mode(mode: TranslationMode) -> Option<TranslationMode> {
    match mode {
        TranslationMode::QualityFirst => None,
        TranslationMode::Balanced => Some(TranslationMode::QualityFirst),
        TranslationMode::StreamingFirst => Some(TranslationMode::Balanced),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{prelude::*, test_runner::RngSeed};

    fn cloud_policy(mode: TranslationMode) -> DirectionPolicy {
        let mut policy = DirectionPolicy::new(AudioDirection::Microphone);
        policy.provider = ProviderId::Openai;
        policy.force_mode(0, mode);
        policy
    }

    #[test]
    fn entering_cloud_discards_local_auto_mode_but_preserves_manual_reaffirmation() {
        for breaches in [3, 6] {
            for reaffirm in [false, true] {
                let direction = AudioDirection::Microphone;
                let mut policies = DuplexLatencyPolicy::default();
                for at in 1..=breaches {
                    policies.record_utterance(
                        direction,
                        at,
                        LatencySample {
                            first_audio_ms: 6_000,
                            last_audio_ms: 6_100,
                            queue_lag_ms: 100,
                        },
                    );
                }
                let automatic = policies.state(direction).current_mode;
                assert_eq!(
                    automatic,
                    if breaches == 3 {
                        TranslationMode::Balanced
                    } else {
                        TranslationMode::StreamingFirst
                    }
                );
                if reaffirm {
                    let selection = policies.force_mode(direction, 10, automatic).unwrap();
                    assert_eq!(selection.from, selection.to);
                    assert_eq!(
                        selection.reason,
                        LatencyTransitionReason::ManualPolicyChange
                    );
                    assert!(policies.force_mode(direction, 11, automatic).is_none());
                }
                policies.set_provider(ProviderId::Openai);
                assert_eq!(
                    policies.state(direction).current_mode,
                    if reaffirm {
                        automatic
                    } else {
                        TranslationMode::QualityFirst
                    }
                );
                assert_eq!(
                    policies.state(AudioDirection::Speaker).current_mode,
                    TranslationMode::QualityFirst
                );
            }
        }
    }

    #[test]
    fn cloud_automatic_paths_preserve_the_explicit_mode_and_breach_telemetry() {
        for mode in [
            TranslationMode::QualityFirst,
            TranslationMode::Balanced,
            TranslationMode::StreamingFirst,
        ] {
            let mut consecutive = cloud_policy(mode);
            let high = LatencySample {
                first_audio_ms: 6_000,
                last_audio_ms: 6_100,
                queue_lag_ms: 600,
            };
            for index in 1..=3 {
                assert!(
                    consecutive.record_utterance(index, high).is_none(),
                    "cloud consecutive breach changed explicit mode"
                );
            }
            assert_eq!(consecutive.state.current_mode, mode);
            assert_eq!(consecutive.samples.len(), 3);
            if mode != TranslationMode::StreamingFirst {
                assert_eq!(consecutive.consecutive_utterance_breaches, 3);
            }

            let mut queued = cloud_policy(mode);
            assert!(queued.observe_queue_lag(1, Some(600)).is_none());
            assert!(
                queued.observe_queue_lag(2_001, Some(600)).is_none(),
                "cloud queue breach changed explicit mode"
            );
            assert_eq!(queued.state.current_mode, mode);

            for breached in [false, true] {
                let mut windows = cloud_policy(mode);
                for epoch in 0..5 {
                    for index in 0..20 {
                        let first_audio_ms = if breached && index % 3 != 0 {
                            6_000
                        } else {
                            100
                        };
                        assert!(
                            windows
                                .record_utterance(
                                    epoch * EPOCH_MS + index * 100 + 1,
                                    LatencySample {
                                        first_audio_ms,
                                        last_audio_ms: first_audio_ms + 100,
                                        queue_lag_ms: if breached { 600 } else { 100 }
                                    }
                                )
                                .is_none()
                        );
                    }
                    assert!(
                        windows.evaluate_epoch((epoch + 1) * EPOCH_MS).is_none(),
                        "cloud window changed explicit mode"
                    );
                }
                assert_eq!(windows.state.current_mode, mode);
                assert_eq!(
                    windows.state.p95_first_audio_ms,
                    if breached { 6_000 } else { 100 }
                );
                assert_eq!(
                    windows.state.p95_queue_lag_ms,
                    if breached { 600 } else { 100 }
                );
                if breached && mode != TranslationMode::StreamingFirst {
                    assert_eq!(windows.breach_windows, 5);
                }
                if !breached && mode != TranslationMode::QualityFirst {
                    assert_eq!(windows.recovery_windows, 5);
                }
            }
        }
    }

    #[test]
    fn cloud_manual_modes_and_provider_switches_preserve_local_adaptation() {
        let mut policies = DuplexLatencyPolicy::default();
        let direction = AudioDirection::Microphone;
        policies.set_provider(ProviderId::Openai);
        assert!(
            policies
                .force_mode(direction, 1, TranslationMode::StreamingFirst)
                .is_some()
        );
        assert!(
            policies
                .force_mode(direction, 2, TranslationMode::QualityFirst)
                .is_some()
        );
        let high = LatencySample {
            first_audio_ms: 6_000,
            last_audio_ms: 6_100,
            queue_lag_ms: 100,
        };
        for at in 3..=5 {
            assert!(policies.record_utterance(direction, at, high).is_none());
        }
        policies.set_provider(ProviderId::Local);
        assert_eq!(
            policies.state(direction).current_mode,
            TranslationMode::QualityFirst
        );
        for at in 6..=8 {
            policies.record_utterance(direction, at, high);
        }
        assert_eq!(
            policies.state(direction).current_mode,
            TranslationMode::Balanced
        );
        policies.set_provider(ProviderId::Openai);
        assert_eq!(
            policies.state(direction).current_mode,
            TranslationMode::QualityFirst
        );
        for at in 9..=11 {
            assert!(policies.record_utterance(direction, at, high).is_none());
        }
        assert_eq!(
            policies.state(direction).current_mode,
            TranslationMode::QualityFirst
        );
        assert_eq!(
            policies.state(AudioDirection::Speaker).current_mode,
            TranslationMode::QualityFirst
        );
        assert_eq!(
            thresholds(ProviderId::Openai, TranslationMode::QualityFirst).0,
            5_000
        );
        assert_eq!(
            thresholds(ProviderId::Openai, TranslationMode::Balanced).0,
            2_000
        );
        assert_eq!(
            thresholds(ProviderId::Openai, TranslationMode::StreamingFirst).0,
            1_000
        );
    }

    #[test]
    fn suppressed_cloud_transitions_keep_tripwire_counters_bounded() {
        let mut policy = cloud_policy(TranslationMode::QualityFirst);
        policy.consecutive_utterance_breaches = u32::MAX;
        let high = LatencySample {
            first_audio_ms: 6_000,
            last_audio_ms: 6_100,
            queue_lag_ms: 600,
        };
        assert!(policy.record_utterance(1, high).is_none());
        assert_eq!(policy.consecutive_utterance_breaches, u32::MAX);
        policy.breach_windows = u32::MAX;
        for at in 2..22 {
            policy.record_utterance(at, high);
        }
        assert!(policy.evaluate_epoch(EPOCH_MS).is_none());
        assert_eq!(policy.breach_windows, u32::MAX);
        policy.force_mode(EPOCH_MS, TranslationMode::Balanced);
        policy.recovery_windows = u32::MAX;
        for at in EPOCH_MS + 1..EPOCH_MS + 21 {
            policy.record_utterance(
                at,
                LatencySample {
                    first_audio_ms: 100,
                    last_audio_ms: 200,
                    queue_lag_ms: 0,
                },
            );
        }
        assert!(policy.evaluate_epoch(2 * EPOCH_MS).is_none());
        assert_eq!(policy.recovery_windows, u32::MAX);
    }

    #[test]
    fn cloud_quality_four_seconds_do_not_trip_either_hysteresis() {
        let mut policies = DuplexLatencyPolicy::default();
        policies.set_provider(ProviderId::Openai);
        for epoch in 0..2 {
            for index in 0..20 {
                assert!(
                    policies
                        .record_utterance(
                            AudioDirection::Microphone,
                            epoch * 60_000 + index * 100,
                            LatencySample {
                                first_audio_ms: 4_000,
                                last_audio_ms: 4_100,
                                queue_lag_ms: 100,
                            }
                        )
                        .is_none()
                );
            }
            assert!(
                policies
                    .evaluate_epoch(AudioDirection::Microphone, (epoch + 1) * 60_000)
                    .is_none()
            );
        }
        assert_eq!(
            policies
                .state(AudioDirection::Microphone)
                .p95_first_audio_ms,
            4_000
        );
        assert_eq!(
            policies.state(AudioDirection::Microphone).current_mode,
            TranslationMode::QualityFirst
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100, max_shrink_iters: 512, rng_seed: RngSeed::Fixed(0x2026_1019),
            ..ProptestConfig::default()
        })]
        #[test]
        fn consecutive_adaptation_obeys_provider_scoped_age_budget(cloud in any::<bool>(), mode_index in 0_u8..3, first_audio_ms in 0_u32..=7_000) {
            let provider = if cloud { ProviderId::Openai } else { ProviderId::Local };
            let mode = [TranslationMode::QualityFirst, TranslationMode::Balanced, TranslationMode::StreamingFirst][usize::from(mode_index)];
            let mut policies = DuplexLatencyPolicy::default();
            policies.set_provider(provider);
            policies.force_mode(AudioDirection::Microphone, 0, mode);
            let limit = match (cloud, mode_index) { (true, 0) => 5_000, (false, 0) => 3_000, (_, 1) => 2_000, _ => 1_000 };
            for at_ms in 1..=3 {
                let transition = policies.record_utterance(AudioDirection::Microphone, at_ms, LatencySample { first_audio_ms, last_audio_ms: first_audio_ms + 100, queue_lag_ms: 100 });
                prop_assert_eq!(transition.is_some(), !cloud && at_ms == 3 && mode_index < 2 && first_audio_ms > limit);
            }
            prop_assert_eq!(policies.state(AudioDirection::Speaker).current_mode, TranslationMode::QualityFirst);
        }

        #[test]
        fn cloud_observation_sequences_preserve_modes_across_manual_and_provider_changes(
            actions in prop::collection::vec((0_u8..5, 0_u32..=7_000), 1..48),
        ) {
            let mut policies = DuplexLatencyPolicy::default();
            let mut provider = ProviderId::Local;
            for (index, (action, value)) in actions.into_iter().enumerate() {
                let at_ms = (index as u64 + 1) * EPOCH_MS;
                let direction = if value.is_multiple_of(2) { AudioDirection::Microphone } else { AudioDirection::Speaker };
                if action == 0 {
                    provider = if value.is_multiple_of(2) { ProviderId::Local } else { ProviderId::Openai };
                    policies.set_provider(provider);
                    continue;
                }
                if action == 1 {
                    let mode = [TranslationMode::QualityFirst, TranslationMode::Balanced, TranslationMode::StreamingFirst][(value % 3) as usize];
                    policies.force_mode(direction, at_ms, mode);
                    prop_assert_eq!(policies.state(direction).current_mode, mode);
                    continue;
                }
                let before = [policies.state(AudioDirection::Microphone).current_mode, policies.state(AudioDirection::Speaker).current_mode];
                let transition = match action {
                    2 => policies.record_utterance(direction, at_ms, LatencySample { first_audio_ms: value, last_audio_ms: value + 100, queue_lag_ms: value % 1_001 }),
                    3 => policies.observe_queue_lag(direction, at_ms, Some(value % 1_001)),
                    _ => policies.evaluate_epoch(direction, at_ms),
                };
                if provider == ProviderId::Openai {
                    prop_assert!(transition.is_none());
                    prop_assert_eq!([policies.state(AudioDirection::Microphone).current_mode, policies.state(AudioDirection::Speaker).current_mode], before);
                }
            }
        }
    }
}
