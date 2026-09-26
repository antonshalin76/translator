use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, watch};
use translator_audio::{AudioGraphState, RouteCandidate, RoutingState};
use translator_core::{
    AudioDirection, Language, LatencyPolicyState, ProviderId, TranslationMode, VoiceEngine,
    VoiceGender, VoiceProfile,
};

use crate::{
    DebugCaptureSession, DebugCaptureStopReason, DebugCaptureStore, DebugTextBuffer,
    DebugTextEvent, DebugTextStatus, DeviceState, DuplexLatencyPolicy, LatencySample,
    LatencyTransition, RoundTripPreconditions, RoundTripStatus,
};

const DEFAULT_EVENT_CAPACITY: usize = 64;

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSnapshot {
    pub translation_running: bool,
    pub runtime_status: RuntimeStatus,
    pub audio_mix_knowledge: AudioMixKnowledge,
    pub debug_text_enabled: bool,
    pub debug_capture_enabled: bool,
    pub directions: Vec<DirectionState>,
    pub audio_mix: AudioMixState,
    pub provider_id: ProviderId,
    pub audio_leaves_machine: bool,
    pub latency_policy: Vec<LatencyPolicyState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_graph: Option<AudioGraphState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routes: Option<RoutingState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub devices: Option<DeviceState>,
    pub self_test: RoundTripSelfTestState,
}

impl Default for RuntimeSnapshot {
    fn default() -> Self {
        Self {
            translation_running: false,
            runtime_status: RuntimeStatus::Stopped,
            audio_mix_knowledge: AudioMixKnowledge::Unapplied,
            debug_text_enabled: false,
            debug_capture_enabled: false,
            directions: vec![
                DirectionState::new(AudioDirection::Microphone, Language::Ru, Language::En),
                DirectionState::new(AudioDirection::Speaker, Language::En, Language::Ru),
            ],
            audio_mix: AudioMixState::default(),
            provider_id: ProviderId::Local,
            audio_leaves_machine: false,
            latency_policy: vec![
                default_latency_state(AudioDirection::Microphone),
                default_latency_state(AudioDirection::Speaker),
            ],
            audio_graph: None,
            routes: None,
            devices: None,
            self_test: RoundTripSelfTestState::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    Stopped,
    Running,
    Failed,
    CleanupPending,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioMixKnowledge {
    Unapplied,
    Known,
    AudioMixStateUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AudioMixState {
    pub microphone_original_percent: u8,
    pub microphone_translation_percent: u8,
    pub speaker_original_percent: u8,
    pub speaker_translation_percent: u8,
}

impl Default for AudioMixState {
    fn default() -> Self {
        Self {
            microphone_original_percent: 0,
            microphone_translation_percent: 100,
            speaker_original_percent: 0,
            speaker_translation_percent: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectionState {
    pub direction_id: AudioDirection,
    pub source_language: Language,
    pub target_language: Language,
    pub enabled: bool,
    pub runtime_status: DirectionRuntimeStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_failure: Option<DirectionRuntimeFailure>,
    pub voice_profile: VoiceProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionRuntimeStatus {
    Stopped,
    Running,
    Recovering,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionRuntimeFailure {
    RestartExhausted,
}

impl DirectionState {
    fn new(
        direction_id: AudioDirection,
        source_language: Language,
        target_language: Language,
    ) -> Self {
        Self {
            direction_id,
            source_language,
            target_language,
            enabled: true,
            runtime_status: DirectionRuntimeStatus::Stopped,
            runtime_failure: None,
            voice_profile: VoiceProfile {
                language: target_language,
                gender: VoiceGender::Male,
                engine: VoiceEngine::Piper,
                model_path: None,
                provider_voice_id: None,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMutationError {
    InvalidLanguagePair,
    VoiceLanguageMismatch,
    VoiceProfileOverrideUnsupported,
    CloudProviderOptInRequired,
    DebugCaptureUnavailable,
    DebugCaptureStopped(DebugCaptureStopReason),
    InvalidAudioMixVolume,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionPatch {
    pub direction_id: AudioDirection,
    pub source_language: Option<Language>,
    pub target_language: Option<Language>,
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderPatch {
    pub provider_id: ProviderId,
    pub cloud_opt_in: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatencyPolicyPatch {
    pub direction_id: AudioDirection,
    pub current_mode: TranslationMode,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceProfilePatch {
    pub direction_id: AudioDirection,
    pub voice_profile: VoiceProfile,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioMixPatch {
    pub microphone_original_percent: Option<u8>,
    pub microphone_translation_percent: Option<u8>,
    pub speaker_original_percent: Option<u8>,
    pub speaker_translation_percent: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoundTripSelfTestState {
    pub availability: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preconditions: Option<RoundTripPreconditions>,
    pub status: RoundTripStatus,
}

impl Default for RoundTripSelfTestState {
    fn default() -> Self {
        Self {
            availability: "unavailable",
            preconditions: None,
            status: RoundTripStatus::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RuntimeEvent {
    SnapshotChanged,
}

#[derive(Debug)]
struct EventBus {
    sender: Mutex<Option<broadcast::Sender<RuntimeEvent>>>,
}

impl EventBus {
    fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self {
            sender: Mutex::new(Some(sender)),
        }
    }

    fn subscribe(&self) -> Option<broadcast::Receiver<RuntimeEvent>> {
        self.sender
            .lock()
            .expect("event bus mutex poisoned")
            .as_ref()
            .map(broadcast::Sender::subscribe)
    }

    fn publish(&self, event: RuntimeEvent) {
        if let Some(sender) = self
            .sender
            .lock()
            .expect("event bus mutex poisoned")
            .as_ref()
        {
            let _ = sender.send(event);
        }
    }

    fn shutdown(&self) {
        self.sender.lock().expect("event bus mutex poisoned").take();
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeStore {
    snapshot: Arc<RwLock<RuntimeSnapshot>>,
    events: Arc<EventBus>,
    debug_text: Arc<Mutex<DebugTextBuffer>>,
    debug_capture: Arc<Mutex<DebugCaptureRuntime>>,
    debug_capture_deadline: watch::Sender<Option<u64>>,
    monotonic_origin: Instant,
    latency: Arc<Mutex<DuplexLatencyPolicy>>,
}

#[derive(Debug, Default)]
struct DebugCaptureRuntime {
    store: Option<DebugCaptureStore>,
    session: Option<DebugCaptureSession>,
}

impl Default for RuntimeStore {
    fn default() -> Self {
        Self::with_event_capacity(DEFAULT_EVENT_CAPACITY)
    }
}

impl RuntimeStore {
    pub fn with_event_capacity(capacity: usize) -> Self {
        let (debug_capture_deadline, _) = watch::channel(None);
        Self {
            snapshot: Arc::new(RwLock::new(RuntimeSnapshot::default())),
            events: Arc::new(EventBus::new(capacity)),
            debug_text: Arc::new(Mutex::new(DebugTextBuffer::default())),
            debug_capture: Arc::new(Mutex::new(DebugCaptureRuntime::default())),
            debug_capture_deadline,
            monotonic_origin: Instant::now(),
            latency: Arc::new(Mutex::new(DuplexLatencyPolicy::default())),
        }
    }

    pub fn snapshot(&self) -> RuntimeSnapshot {
        let mut snapshot = self
            .snapshot
            .read()
            .expect("runtime state lock poisoned")
            .clone();
        if !snapshot.debug_text_enabled {
            snapshot.self_test.status.debug_text = None;
        }
        snapshot
    }

    pub fn set_audio_graph(&self, state: AudioGraphState) {
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        if snapshot.audio_graph.as_ref() == Some(&state) {
            return;
        }
        snapshot.audio_graph = Some(state);
        drop(snapshot);
        self.publish_snapshot_changed();
    }

    pub fn clear_audio_graph(&self, code: &'static str) {
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        if snapshot.audio_graph.take().is_none() {
            return;
        }
        drop(snapshot);
        tracing::warn!(event = "audio_graph_unavailable", code);
        self.publish_snapshot_changed();
    }

    pub fn set_routes(&self, state: RoutingState) {
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        if snapshot.routes.as_ref() == Some(&state) {
            return;
        }
        tracing::info!(
            event = "route_state_changed",
            resolution = ?state.resolution,
            candidate_count = state.candidates.len(),
            conflict_count = state.conflicting_stream_ids.len(),
            active = state.active_route.is_some()
        );
        snapshot.routes = Some(state);
        drop(snapshot);
        self.publish_snapshot_changed();
    }

    pub fn clear_routes(&self, code: &'static str) {
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        if snapshot.routes.take().is_none() {
            return;
        }
        drop(snapshot);
        tracing::warn!(event = "route_state_unavailable", code);
        self.publish_snapshot_changed();
    }

    pub fn set_devices(&self, state: DeviceState) {
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        if snapshot.devices.as_ref() == Some(&state) {
            return;
        }
        snapshot.devices = Some(state);
        drop(snapshot);
        self.publish_snapshot_changed();
    }

    pub fn clear_devices(&self, code: &'static str) {
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        if snapshot.devices.take().is_none() {
            return;
        }
        drop(snapshot);
        tracing::warn!(event = "device_state_unavailable", code);
        self.publish_snapshot_changed();
    }

    pub fn set_self_test(&self, mut state: RoundTripSelfTestState) {
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        if !snapshot.debug_text_enabled {
            state.status.debug_text = None;
        }
        snapshot.self_test = state;
        drop(snapshot);
        self.publish_snapshot_changed();
    }

    pub(crate) fn direction_candidate(
        &self,
        patch: DirectionPatch,
    ) -> Result<RuntimeSnapshot, RuntimeMutationError> {
        let language_pair = match (patch.source_language, patch.target_language) {
            (Some(source_language), Some(target_language))
                if source_language != target_language =>
            {
                Some((source_language, target_language))
            }
            (None, None) => None,
            _ => return Err(RuntimeMutationError::InvalidLanguagePair),
        };
        let mut snapshot = self.snapshot();
        let direction = direction_mut(&mut snapshot, patch.direction_id);
        if let Some((source_language, target_language)) = language_pair {
            direction.source_language = source_language;
            direction.target_language = target_language;
            if direction.voice_profile.language != target_language {
                direction.voice_profile.language = target_language;
                direction.voice_profile.model_path = None;
                direction.voice_profile.provider_voice_id = None;
            }
        }
        if let Some(enabled) = patch.enabled {
            direction.enabled = enabled;
        }
        Ok(snapshot)
    }

    pub(crate) fn provider_candidate(
        &self,
        patch: ProviderPatch,
    ) -> Result<RuntimeSnapshot, RuntimeMutationError> {
        if patch.provider_id == ProviderId::Openai && patch.cloud_opt_in != Some(true) {
            return Err(RuntimeMutationError::CloudProviderOptInRequired);
        }
        let mut snapshot = self.snapshot();
        snapshot.provider_id = patch.provider_id;
        snapshot.audio_leaves_machine = patch.provider_id == ProviderId::Openai;
        snapshot.self_test.status.debug_text = None;
        Ok(snapshot)
    }

    pub fn set_latency_policy(&self, patch: LatencyPolicyPatch) {
        let mut latency = self.latency.lock().expect("latency mutex poisoned");
        if let Some(transition) =
            latency.force_mode(patch.direction_id, self.monotonic_ms(), patch.current_mode)
        {
            self.update_latency_snapshot(&latency);
            log_latency_transition(transition);
            self.publish_snapshot_changed();
        }
    }

    pub(crate) fn voice_candidate(
        &self,
        patch: VoiceProfilePatch,
    ) -> Result<RuntimeSnapshot, RuntimeMutationError> {
        let mut snapshot = self.snapshot();
        let direction = direction_mut(&mut snapshot, patch.direction_id);
        if patch.voice_profile.language != direction.target_language {
            return Err(RuntimeMutationError::VoiceLanguageMismatch);
        }
        if patch.voice_profile.has_overrides() {
            return Err(RuntimeMutationError::VoiceProfileOverrideUnsupported);
        }
        direction.voice_profile = patch.voice_profile;
        Ok(snapshot)
    }

    pub(crate) fn audio_mix_candidate(
        &self,
        patch: AudioMixPatch,
    ) -> Result<RuntimeSnapshot, RuntimeMutationError> {
        if [
            patch.microphone_original_percent,
            patch.microphone_translation_percent,
            patch.speaker_original_percent,
            patch.speaker_translation_percent,
        ]
        .into_iter()
        .flatten()
        .any(|value| value > 100)
        {
            return Err(RuntimeMutationError::InvalidAudioMixVolume);
        }

        let mut snapshot = self.snapshot();
        if let Some(value) = patch.microphone_original_percent {
            snapshot.audio_mix.microphone_original_percent = value;
        }
        if let Some(value) = patch.microphone_translation_percent {
            snapshot.audio_mix.microphone_translation_percent = value;
        }
        if let Some(value) = patch.speaker_original_percent {
            snapshot.audio_mix.speaker_original_percent = value;
        }
        if let Some(value) = patch.speaker_translation_percent {
            snapshot.audio_mix.speaker_translation_percent = value;
        }
        Ok(snapshot)
    }

    pub fn audio_graph(&self) -> Option<AudioGraphState> {
        self.snapshot().audio_graph
    }

    pub fn routes(&self) -> Option<RoutingState> {
        self.snapshot().routes
    }

    pub fn route_candidates(&self) -> Vec<RouteCandidate> {
        self.snapshot()
            .routes
            .map(|routes| routes.candidates)
            .unwrap_or_default()
    }

    pub fn publish_snapshot_changed(&self) {
        self.events.publish(RuntimeEvent::SnapshotChanged);
    }

    pub fn shutdown_events(&self) {
        self.events.shutdown();
    }

    pub(crate) fn subscribe(&self) -> Option<broadcast::Receiver<RuntimeEvent>> {
        self.events.subscribe()
    }

    pub(crate) fn commit_control(&self, candidate: RuntimeSnapshot) {
        self.commit_control_projection(candidate, false);
    }

    pub(crate) fn commit_admitted_control(&self, candidate: RuntimeSnapshot) {
        self.commit_control_projection(candidate, true);
    }

    fn commit_control_projection(&self, candidate: RuntimeSnapshot, admitted_facts: bool) {
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        let provider_changed = snapshot.provider_id != candidate.provider_id;
        let stopped = snapshot.translation_running && !candidate.translation_running;
        snapshot.translation_running = candidate.translation_running;
        snapshot.runtime_status = candidate.runtime_status;
        snapshot.audio_mix_knowledge = candidate.audio_mix_knowledge;
        snapshot.directions = candidate.directions;
        snapshot.audio_mix = candidate.audio_mix;
        snapshot.provider_id = candidate.provider_id;
        snapshot.audio_leaves_machine = candidate.audio_leaves_machine;
        if admitted_facts {
            snapshot.devices = candidate.devices;
            snapshot.audio_graph = candidate.audio_graph;
            snapshot.routes = candidate.routes;
        }
        let clear_debug = if stopped {
            snapshot.self_test.status.debug_text = None;
            Some(false)
        } else if provider_changed {
            snapshot.self_test.status.debug_text = None;
            Some(true)
        } else {
            None
        };
        drop(snapshot);
        if let Some(provider_switch) = clear_debug {
            let mut debug = self.debug_text.lock().expect("debug text mutex poisoned");
            if provider_switch {
                debug.clear_for_provider_switch();
            } else {
                debug.clear_for_session_stop();
            }
        }
        self.publish_snapshot_changed();
        tracing::info!(
            event = "translation_state_changed",
            running = candidate.translation_running,
            status = ?candidate.runtime_status,
            audio_mix_knowledge = ?candidate.audio_mix_knowledge,
        );
    }

    pub fn set_debug_text_enabled(&self, enabled: bool) {
        self.debug_text
            .lock()
            .expect("debug text mutex poisoned")
            .set_enabled(enabled);
        let mut snapshot = self.snapshot.write().expect("runtime state lock poisoned");
        snapshot.debug_text_enabled = enabled;
        if !enabled {
            snapshot.self_test.status.debug_text = None;
        }
        drop(snapshot);
        self.publish_snapshot_changed();
        tracing::info!(event = "debug_text_state_changed", enabled);
    }

    pub fn configure_debug_capture(&self, store: DebugCaptureStore) {
        self.debug_capture
            .lock()
            .expect("debug capture mutex poisoned")
            .store = Some(store);
    }

    pub fn set_debug_capture_enabled(&self, enabled: bool) -> Result<(), RuntimeMutationError> {
        let mut capture = self
            .debug_capture
            .lock()
            .expect("debug capture mutex poisoned");
        if enabled && capture.session.is_none() {
            let Some(store) = capture.store.as_ref() else {
                return Err(RuntimeMutationError::DebugCaptureUnavailable);
            };
            let name = format!("capture-{}", uuid::Uuid::new_v4());
            let session = store
                .start(
                    &name,
                    u64::try_from(self.monotonic_origin.elapsed().as_millis()).unwrap_or(u64::MAX),
                )
                .map_err(|_| RuntimeMutationError::DebugCaptureUnavailable)?;
            self.debug_capture_deadline
                .send_replace(Some(session.deadline_ms()));
            capture.session = Some(session);
        } else if !enabled {
            capture.session.take();
            self.debug_capture_deadline.send_replace(None);
        }
        self.snapshot
            .write()
            .expect("runtime state lock poisoned")
            .debug_capture_enabled = enabled;
        self.publish_snapshot_changed();
        tracing::info!(event = "debug_capture_state_changed", enabled);
        Ok(())
    }

    pub fn append_debug_capture(
        &self,
        bytes: &[u8],
        at_ms: u64,
    ) -> Result<(), RuntimeMutationError> {
        let mut capture = self
            .debug_capture
            .lock()
            .expect("debug capture mutex poisoned");
        let Some(session) = capture.session.as_mut() else {
            return Err(RuntimeMutationError::DebugCaptureUnavailable);
        };
        if let Err(reason) = session.append(bytes, at_ms) {
            capture.session.take();
            drop(capture);
            self.finish_debug_capture(reason);
            return Err(RuntimeMutationError::DebugCaptureStopped(reason));
        }
        Ok(())
    }

    pub async fn run_debug_capture_watchdog(self) {
        let mut deadlines = self.debug_capture_deadline.subscribe();
        loop {
            let deadline = *deadlines.borrow_and_update();
            let Some(deadline) = deadline else {
                if deadlines.changed().await.is_err() {
                    return;
                }
                continue;
            };
            let remaining = deadline.saturating_sub(self.monotonic_ms());
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(remaining)) => {
                    self.expire_debug_capture(deadline);
                }
                changed = deadlines.changed() => {
                    if changed.is_err() {
                        return;
                    }
                }
            }
        }
    }

    fn expire_debug_capture(&self, at_ms: u64) {
        let mut capture = self
            .debug_capture
            .lock()
            .expect("debug capture mutex poisoned");
        let Some(session) = capture.session.as_mut() else {
            return;
        };
        let Err(reason) = session.expire(at_ms) else {
            return;
        };
        capture.session.take();
        drop(capture);
        self.finish_debug_capture(reason);
    }

    fn finish_debug_capture(&self, reason: DebugCaptureStopReason) {
        self.debug_capture_deadline.send_replace(None);
        self.snapshot
            .write()
            .expect("runtime state lock poisoned")
            .debug_capture_enabled = false;
        self.publish_snapshot_changed();
        tracing::warn!(
            event = "debug_capture_stopped",
            reason = ?reason
        );
    }

    pub fn record_debug_text(&self, event: DebugTextEvent) -> bool {
        self.debug_text
            .lock()
            .expect("debug text mutex poisoned")
            .push(event)
    }

    pub fn debug_text_status(&self) -> DebugTextStatus {
        self.debug_text
            .lock()
            .expect("debug text mutex poisoned")
            .safe_status()
    }

    pub fn record_latency_utterance(
        &self,
        direction: AudioDirection,
        at_ms: u64,
        sample: LatencySample,
    ) -> Option<LatencyTransition> {
        let mut latency = self.latency.lock().expect("latency mutex poisoned");
        let transition = latency.record_utterance(direction, at_ms, sample);
        self.update_latency_snapshot(&latency);
        if let Some(transition) = transition {
            log_latency_transition(transition);
            self.publish_snapshot_changed();
        }
        transition
    }

    pub fn observe_latency_queue(
        &self,
        direction: AudioDirection,
        at_ms: u64,
        queue_lag_ms: Option<u32>,
    ) -> Option<LatencyTransition> {
        let mut latency = self.latency.lock().expect("latency mutex poisoned");
        let transition = latency.observe_queue_lag(direction, at_ms, queue_lag_ms);
        self.update_latency_snapshot(&latency);
        if let Some(transition) = transition {
            log_latency_transition(transition);
            self.publish_snapshot_changed();
        }
        transition
    }

    pub fn evaluate_latency_epoch(
        &self,
        direction: AudioDirection,
        epoch_end_ms: u64,
    ) -> Option<LatencyTransition> {
        let mut latency = self.latency.lock().expect("latency mutex poisoned");
        let transition = latency.evaluate_epoch(direction, epoch_end_ms);
        self.update_latency_snapshot(&latency);
        if let Some(transition) = transition {
            log_latency_transition(transition);
        }
        self.publish_snapshot_changed();
        transition
    }

    pub fn monotonic_ms(&self) -> u64 {
        u64::try_from(self.monotonic_origin.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn update_latency_snapshot(&self, latency: &DuplexLatencyPolicy) {
        self.snapshot
            .write()
            .expect("runtime state lock poisoned")
            .latency_policy = vec![
            latency.state(AudioDirection::Microphone).clone(),
            latency.state(AudioDirection::Speaker).clone(),
        ];
    }
}

fn log_latency_transition(transition: LatencyTransition) {
    tracing::info!(
        event = "latency_mode_changed",
        direction = ?transition.direction,
        from = ?transition.from,
        to = ?transition.to,
        reason = ?transition.reason,
        at_ms = transition.at_ms
    );
}

fn direction_mut(
    snapshot: &mut RuntimeSnapshot,
    direction_id: AudioDirection,
) -> &mut DirectionState {
    snapshot
        .directions
        .iter_mut()
        .find(|direction| direction.direction_id == direction_id)
        .expect("both runtime directions always exist")
}

fn default_latency_state(direction_id: AudioDirection) -> LatencyPolicyState {
    LatencyPolicyState::new(
        direction_id,
        TranslationMode::QualityFirst,
        0,
        0,
        0,
        None,
        None,
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::sync::broadcast::error::TryRecvError;
    use translator_audio::{
        AecCapability, AudioGraphState, DeviceHealth, DeviceSelectionState, GraphHealth,
        OutputMode, RouteResolution, RoutingState,
    };

    use crate::{
        AcousticSafety, DebugCaptureLimits, DebugCaptureStore, DeviceState, RuntimeMutationError,
    };

    use super::{RuntimeEvent, RuntimeStore};

    #[test]
    fn voice_override_candidates_reject_all_present_values_without_snapshot_or_events() {
        let mut failures = Vec::new();
        for provider in [super::ProviderId::Local, super::ProviderId::Openai] {
            for value in ["unapproved-voice", "", " \t"] {
                for (model, voice) in [(true, false), (false, true), (true, true)] {
                    let store = RuntimeStore::default();
                    let mut seeded = store.snapshot();
                    seeded.provider_id = provider;
                    seeded.audio_leaves_machine = provider == super::ProviderId::Openai;
                    seeded.directions[1].enabled = false;
                    store.commit_control(seeded);
                    let before = serde_json::to_value(store.snapshot()).unwrap();
                    let mut events = store.subscribe().unwrap();
                    let mut profile = store.snapshot().directions[1].voice_profile.clone();
                    profile.model_path = model.then(|| value.to_owned());
                    profile.provider_voice_id = voice.then(|| value.to_owned());
                    let result = store.voice_candidate(super::VoiceProfilePatch {
                        direction_id: super::AudioDirection::Speaker,
                        voice_profile: profile,
                    });
                    if result.is_ok() {
                        failures.push((provider, model, voice, value.len()));
                    }
                    assert_eq!(serde_json::to_value(store.snapshot()).unwrap(), before);
                    assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "unsupported candidates accepted: {failures:?}"
        );
    }

    #[test]
    fn voice_override_language_priority_and_builtin_candidate_characterization() {
        let store = RuntimeStore::default();
        let mut events = store.subscribe().unwrap();
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let mut profile = store.snapshot().directions[1].voice_profile.clone();
        profile.language = super::Language::En;
        profile.model_path = Some("unapproved.onnx".into());
        assert!(matches!(
            store.voice_candidate(super::VoiceProfilePatch {
                direction_id: super::AudioDirection::Speaker,
                voice_profile: profile,
            }),
            Err(RuntimeMutationError::VoiceLanguageMismatch)
        ));
        for literal in [
            r#"{"direction_id":"speaker","voice_profile":{"language":"ru","gender":"female","engine":"piper"}}"#,
            r#"{"direction_id":"speaker","voice_profile":{"language":"ru","gender":"female","engine":"piper","model_path":null,"provider_voice_id":null}}"#,
        ] {
            let patch = serde_json::from_str(literal).unwrap();
            let candidate = store.voice_candidate(patch).unwrap();
            assert_eq!(
                candidate.directions[1].voice_profile.gender,
                super::VoiceGender::Female
            );
            assert!(candidate.directions[1].voice_profile.model_path.is_none());
            assert!(
                candidate.directions[1]
                    .voice_profile
                    .provider_voice_id
                    .is_none()
            );
        }
        assert_eq!(serde_json::to_value(store.snapshot()).unwrap(), before);
        assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn admitted_control_commits_facts_once_without_overwriting_intervening_projection_updates() {
        let store = RuntimeStore::default();
        let mut candidate = store.snapshot();
        candidate.translation_running = true;
        candidate.runtime_status = super::RuntimeStatus::Running;
        candidate.audio_graph = Some(AudioGraphState {
            health: GraphHealth::Ready,
            endpoints: Vec::new(),
            owned_module_ids: vec![42],
            safe_error: None,
        });
        candidate.routes = Some(RoutingState {
            candidates: Vec::new(),
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: RouteResolution::NoCandidate,
        });
        let selection = DeviceSelectionState {
            health: DeviceHealth::DeviceUnavailable,
            selected: None,
            pinned_name: None,
            current_default: None,
            pending_default: None,
        };
        candidate.devices = Some(DeviceState {
            source: selection.clone(),
            sink: selection,
            acoustic: AcousticSafety {
                mode: OutputMode::UnknownUnsafe,
                aec_capability: AecCapability::Unavailable,
                full_duplex_allowed: false,
                warning: Some(crate::AcousticWarning::DeviceUnavailable),
            },
        });
        let accepted_graph = candidate.audio_graph.clone();
        let accepted_routes = candidate.routes.clone();
        let accepted_devices = candidate.devices.clone();
        store.set_debug_text_enabled(true);
        let mut self_test = store.snapshot().self_test;
        self_test.availability = "available";
        store.set_self_test(self_test);
        let before_self_test = serde_json::to_value(store.snapshot().self_test).unwrap();
        let mut events = store.subscribe().unwrap();

        store.commit_admitted_control(candidate);

        let committed = store.snapshot();
        assert!(committed.translation_running);
        assert!(committed.debug_text_enabled);
        assert_eq!(committed.audio_graph, accepted_graph);
        assert_eq!(committed.routes, accepted_routes);
        assert_eq!(committed.devices, accepted_devices);
        assert_eq!(
            serde_json::to_value(committed.self_test).unwrap(),
            before_self_test
        );
        assert!(matches!(
            events.try_recv(),
            Ok(RuntimeEvent::SnapshotChanged)
        ));
        assert!(matches!(events.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn ordinary_control_commit_preserves_newer_audio_facts() {
        let store = RuntimeStore::default();
        let candidate = store.snapshot();
        store.set_audio_graph(AudioGraphState {
            health: GraphHealth::Ready,
            endpoints: Vec::new(),
            owned_module_ids: vec![42],
            safe_error: None,
        });
        store.set_routes(RoutingState {
            candidates: Vec::new(),
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: RouteResolution::NoCandidate,
        });
        let fresh = store.snapshot();
        store.commit_control(candidate);
        assert_eq!(store.snapshot().audio_graph, fresh.audio_graph);
        assert_eq!(store.snapshot().routes, fresh.routes);
    }

    #[test]
    fn graph_route_and_device_watch_updates_publish_only_on_change() {
        let store = RuntimeStore::default();
        let mut receiver = store.subscribe().expect("event bus should be active");

        let graph = AudioGraphState {
            health: GraphHealth::Ready,
            endpoints: Vec::new(),
            owned_module_ids: Vec::new(),
            safe_error: None,
        };
        assert_single_change_then_no_duplicate(
            &mut receiver,
            || store.set_audio_graph(graph.clone()),
            || store.set_audio_graph(graph.clone()),
        );
        let mut changed_graph = graph;
        changed_graph.health = GraphHealth::Degraded;
        assert_change(&mut receiver, || store.set_audio_graph(changed_graph));

        let routes = RoutingState {
            candidates: Vec::new(),
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: RouteResolution::NoCandidate,
        };
        assert_single_change_then_no_duplicate(
            &mut receiver,
            || store.set_routes(routes.clone()),
            || store.set_routes(routes.clone()),
        );
        let mut changed_routes = routes;
        changed_routes.resolution = RouteResolution::AwaitingSelection;
        assert_change(&mut receiver, || store.set_routes(changed_routes));

        let unavailable = DeviceSelectionState {
            health: DeviceHealth::DeviceUnavailable,
            selected: None,
            pinned_name: None,
            current_default: None,
            pending_default: None,
        };
        let devices = DeviceState {
            source: unavailable.clone(),
            sink: unavailable,
            acoustic: AcousticSafety {
                mode: OutputMode::UnknownUnsafe,
                aec_capability: AecCapability::Unavailable,
                full_duplex_allowed: false,
                warning: None,
            },
        };
        assert_single_change_then_no_duplicate(
            &mut receiver,
            || store.set_devices(devices.clone()),
            || store.set_devices(devices.clone()),
        );
        let mut changed_devices = devices;
        changed_devices.acoustic.mode = OutputMode::Headphones;
        assert_change(&mut receiver, || store.set_devices(changed_devices));
    }

    #[test]
    fn graph_route_and_device_clear_publish_only_when_state_was_present() {
        let store = RuntimeStore::default();
        let mut receiver = store.subscribe().expect("event bus should be active");

        assert_change(&mut receiver, || {
            store.set_audio_graph(AudioGraphState {
                health: GraphHealth::Ready,
                endpoints: Vec::new(),
                owned_module_ids: Vec::new(),
                safe_error: None,
            });
        });
        assert_change(&mut receiver, || store.clear_audio_graph("unavailable"));
        store.clear_audio_graph("still_unavailable");
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

        assert_change(&mut receiver, || {
            store.set_routes(RoutingState {
                candidates: Vec::new(),
                source_outputs: Vec::new(),
                conflicting_stream_ids: Vec::new(),
                active_route: None,
                resolution: RouteResolution::NoCandidate,
            });
        });
        assert_change(&mut receiver, || store.clear_routes("unavailable"));
        store.clear_routes("still_unavailable");
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

        let unavailable = DeviceSelectionState {
            health: DeviceHealth::DeviceUnavailable,
            selected: None,
            pinned_name: None,
            current_default: None,
            pending_default: None,
        };
        assert_change(&mut receiver, || {
            store.set_devices(DeviceState {
                source: unavailable.clone(),
                sink: unavailable,
                acoustic: AcousticSafety {
                    mode: OutputMode::UnknownUnsafe,
                    aec_capability: AecCapability::Unavailable,
                    full_duplex_allowed: false,
                    warning: None,
                },
            });
        });
        assert_change(&mut receiver, || store.clear_devices("unavailable"));
        store.clear_devices("still_unavailable");
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test(start_paused = true)]
    async fn capture_watchdog_stops_silent_session_at_its_deadline_and_publishes_once() {
        let temp = tempfile::tempdir().unwrap();
        let store = RuntimeStore::default();
        store.configure_debug_capture(
            DebugCaptureStore::open(temp.path(), DebugCaptureLimits::new(1_000, 1024, 0)).unwrap(),
        );
        let mut receiver = store.subscribe().expect("event bus should be active");
        let watchdog = tokio::spawn(store.clone().run_debug_capture_watchdog());
        tokio::task::yield_now().await;

        tokio::time::advance(Duration::from_millis(100)).await;
        store.set_debug_capture_enabled(true).unwrap();
        assert!(matches!(
            receiver.try_recv(),
            Ok(RuntimeEvent::SnapshotChanged)
        ));
        tokio::task::yield_now().await;

        tokio::time::advance(Duration::from_millis(999)).await;
        tokio::task::yield_now().await;
        assert!(store.snapshot().debug_capture_enabled);
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert!(!store.snapshot().debug_capture_enabled);
        assert_eq!(
            store.append_debug_capture(&[0], 1_001),
            Err(RuntimeMutationError::DebugCaptureUnavailable)
        );
        assert!(matches!(
            receiver.try_recv(),
            Ok(RuntimeEvent::SnapshotChanged)
        ));
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

        watchdog.abort();
        assert!(watchdog.await.unwrap_err().is_cancelled());
    }

    fn assert_single_change_then_no_duplicate(
        receiver: &mut tokio::sync::broadcast::Receiver<RuntimeEvent>,
        change: impl FnOnce(),
        duplicate: impl FnOnce(),
    ) {
        assert_change(receiver, change);
        duplicate();
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    fn assert_change(
        receiver: &mut tokio::sync::broadcast::Receiver<RuntimeEvent>,
        change: impl FnOnce(),
    ) {
        change();
        assert!(matches!(
            receiver.try_recv(),
            Ok(RuntimeEvent::SnapshotChanged)
        ));
    }
}
