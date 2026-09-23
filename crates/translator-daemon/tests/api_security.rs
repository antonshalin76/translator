use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use bytes::Bytes;
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;
use translator_audio::{
    AudioGraphState, GraphHealth, RouteResolution, RoutingSafeError, RoutingState,
};
use translator_core::ProviderId;
use translator_daemon::{
    ActiveDuplexRuntime, AdmittedDuplex, AecCalibrationController, AecCalibrationCoordinator,
    AecCalibrationEngine, AecCalibrationEngineError, AecCalibrationFuture, AecCalibrationRequest,
    AecCleanupFuture, AecProofBinding, ApiControllers, ApiLimits, AudioMixController,
    AudioMixState, AudioOperationGate, ControlApplication, ControlFailure, ControlToken,
    DebugCaptureLimits, DebugCaptureStore, DebugTextEvent, DuplexRunner, DuplexRuntimeError,
    DuplexStartFailure, DuplexStartResult, ManualRouteController, RoundTripController,
    RoundTripDebugText, RoundTripSelfTestState, RoundTripStatus, RuntimeMaintenance,
    RuntimeSnapshot, RuntimeStatus, RuntimeStore, TranslationMixMode, build_router,
    build_router_with_controllers, build_router_with_manual_routes, validate_listen_address,
};

const TOKEN: &str = "4242424242424242424242424242424242424242424242424242424242424242";

#[derive(Default)]
struct BlockingCalibrationEngine {
    calls: AtomicUsize,
}

struct C9ApiHeldInspection {
    delegate: BlockingCalibrationEngine,
    entered: tokio::sync::Notify,
    released: (Mutex<bool>, Condvar),
}

impl AecCalibrationEngine for C9ApiHeldInspection {
    fn inspect_binding(
        &self,
        deadline: tokio::time::Instant,
    ) -> Result<AecProofBinding, AecCalibrationEngineError> {
        self.entered.notify_one();
        let released = self.released.0.lock().unwrap();
        let (released, _) = self
            .released
            .1
            .wait_timeout_while(released, Duration::from_secs(3), |released| !*released)
            .unwrap();
        if !*released {
            return Err(AecCalibrationEngineError {
                code: "test_barrier_expired",
                cleanup_confirmed: false,
            });
        }
        self.delegate.inspect_binding(deadline)
    }

    fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
        self.delegate.calibrate(request)
    }

    fn cleanup(&self, deadline: tokio::time::Instant) -> AecCleanupFuture {
        self.delegate.cleanup(deadline)
    }
}

struct C9ApiReleaseInspection(Arc<C9ApiHeldInspection>);

impl Drop for C9ApiReleaseInspection {
    fn drop(&mut self) {
        *self.0.released.0.lock().unwrap() = true;
        self.0.released.1.notify_all();
    }
}

impl AecCalibrationEngine for BlockingCalibrationEngine {
    fn inspect_binding(
        &self,
        _deadline: tokio::time::Instant,
    ) -> Result<AecProofBinding, AecCalibrationEngineError> {
        Ok(AecProofBinding {
            audio_server_id: "test-server".into(),
            source_hardware_id: "test-source-hardware".into(),
            sink_hardware_id: "test-sink-hardware".into(),
            source_name: "test-source".into(),
            sink_name: "test-sink".into(),
            source_port: "test-source-port".into(),
            sink_port: "test-sink-port".into(),
            source_channel_gains: vec![1],
            sink_channel_gains: vec![1],
            source_muted: false,
            sink_muted: false,
            source_geometry: "test-source-geometry".into(),
            sink_geometry: "test-sink-geometry".into(),
            aec_module_id: 1,
            aec_source_id: 2,
            aec_sink_id: 3,
            aec_generation: "test-generation".into(),
            aec_config_id: "test-aec-config".into(),
            vad_config_id: "test-vad-config".into(),
            provider_config_id: "test-provider-config".into(),
        })
    }

    fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            request.cancellation.cancelled().await;
            Err(AecCalibrationEngineError {
                code: "cancelled",
                cleanup_confirmed: true,
            })
        })
    }

    fn cleanup(&self, _deadline: tokio::time::Instant) -> AecCleanupFuture {
        Box::pin(async { true })
    }
}

struct FakeManualRoutes {
    selected_stream: AtomicUsize,
}

impl ManualRouteController for FakeManualRoutes {
    fn reconcile(&self, stream_id: u32) -> Result<RoutingState, RoutingSafeError> {
        self.selected_stream
            .store(stream_id as usize, Ordering::SeqCst);
        Ok(RoutingState {
            candidates: Vec::new(),
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: RouteResolution::NoCandidate,
        })
    }
}

struct RefreshingManualRoutes {
    refreshes: AtomicUsize,
    running_at_refresh: AtomicUsize,
}

impl ManualRouteController for RefreshingManualRoutes {
    fn reconcile(&self, _stream_id: u32) -> Result<RoutingState, RoutingSafeError> {
        Ok(RoutingState {
            candidates: Vec::new(),
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: RouteResolution::NoCandidate,
        })
    }

    fn refresh_audio_state(&self, store: &RuntimeStore) {
        let running = usize::from(store.snapshot().translation_running);
        self.refreshes.fetch_add(1, Ordering::SeqCst);
        self.running_at_refresh.store(running, Ordering::SeqCst);
        store.set_audio_graph(AudioGraphState {
            health: GraphHealth::Ready,
            endpoints: Vec::new(),
            owned_module_ids: Vec::new(),
            safe_error: None,
        });
    }
}

impl RuntimeMaintenance for RefreshingManualRoutes {
    fn refresh(&self, store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.refresh_audio_state(store);
        Ok(())
    }
}

struct NoopFacts;

struct RejectedFacts(translator_daemon::FactsError);

impl translator_daemon::RuntimeFactsSource for RejectedFacts {
    fn inspect(
        &self,
        _deadline: std::time::Instant,
    ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
        Err(self.0)
    }
}

#[tokio::test]
async fn translation_facts_rejections_keep_problem_contract_and_snapshot_unchanged() {
    use translator_daemon::FactsError;
    for (failure, status, code) in [
        (
            FactsError::DiscoveryFailed,
            StatusCode::SERVICE_UNAVAILABLE,
            "audio_facts_unavailable",
        ),
        (
            FactsError::Busy,
            StatusCode::SERVICE_UNAVAILABLE,
            "audio_facts_busy",
        ),
        (
            FactsError::Expired,
            StatusCode::SERVICE_UNAVAILABLE,
            "audio_facts_expired",
        ),
        (
            FactsError::InvalidPhysicalDevice,
            StatusCode::CONFLICT,
            "translation_precondition_failed",
        ),
        (
            FactsError::SinkValidationFailed,
            StatusCode::CONFLICT,
            "translation_precondition_failed",
        ),
    ] {
        let store = RuntimeStore::default();
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let runner = Arc::new(FakeDuplexRunner::default());
        let controller = ControlApplication::spawn(
            store.clone(),
            runner.clone(),
            AudioOperationGate::new(),
            Arc::new(RejectedFacts(failure)),
            Arc::new(NoopFacts),
            None,
        );
        let router = build_router_with_controllers(
            store.clone(),
            ControlToken::parse(TOKEN).unwrap(),
            ApiLimits::default(),
            ApiControllers {
                translation: Some(controller.clone()),
                ..ApiControllers::default()
            },
        );
        let response = router
            .oneshot(request(
                Method::POST,
                "/v1/translation/start",
                Some(TOKEN),
                Body::from("{}"),
            ))
            .await
            .unwrap();
        let actual_status = response.status();
        let content_type = response.headers().get(header::CONTENT_TYPE).cloned();
        let payload = json_body(response).await;
        let after = serde_json::to_value(store.snapshot()).unwrap();
        controller.shutdown().await.unwrap();

        assert_eq!(actual_status, status);
        assert_eq!(content_type.unwrap(), "application/problem+json");
        assert_eq!(payload["code"], code);
        assert_eq!(payload["type"], format!("urn:translator:error:{code}"));
        assert_eq!(payload["status"], status.as_u16());
        assert_eq!(after, before);
        assert_eq!(runner.state.starts.load(Ordering::SeqCst), 0);
    }
}

impl translator_daemon::RuntimeFactsSource for NoopFacts {
    fn inspect(
        &self,
        _deadline: std::time::Instant,
    ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
        use translator_audio::{
            AecCapability, AudioGraphState, DeviceFacts, DeviceHealth, DeviceSelectionState,
            GraphHealth, OutputMode, PhysicalDevice, RouteResolution, RoutingState,
        };
        let selection = |name: &str| DeviceSelectionState {
            health: DeviceHealth::Available,
            selected: Some(PhysicalDevice {
                id: 1,
                name: name.into(),
                description: name.into(),
                active_port: None,
                active_port_type: None,
                available: true,
            }),
            pinned_name: Some(name.into()),
            current_default: Some(name.into()),
            pending_default: None,
        };
        Ok(translator_daemon::RuntimeFacts {
            devices: DeviceFacts {
                source: selection("alsa_input.physical"),
                sink: selection("alsa_output.physical"),
                output_mode: OutputMode::Headphones,
                aec_capability: AecCapability::Unavailable,
            },
            audio_graph: AudioGraphState {
                health: GraphHealth::Ready,
                endpoints: Vec::new(),
                owned_module_ids: Vec::new(),
                safe_error: None,
            },
            routes: RoutingState {
                candidates: Vec::new(),
                source_outputs: Vec::new(),
                conflicting_stream_ids: Vec::new(),
                active_route: None,
                resolution: RouteResolution::NoCandidate,
            },
        })
    }
}

impl RuntimeMaintenance for NoopFacts {
    fn refresh(&self, _store: &RuntimeStore) -> Result<(), ControlFailure> {
        Ok(())
    }
}

struct NoopAudioMix;

impl AudioMixController for NoopAudioMix {
    fn apply_desired(
        &self,
        _volumes: AudioMixState,
        _mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn reconcile_committed(&self, _mode: TranslationMixMode) -> Result<(), ControlFailure> {
        Ok(())
    }

    fn recover_committed(&self, _mode: TranslationMixMode) -> Result<(), ControlFailure> {
        Ok(())
    }
}

#[derive(Default)]
struct FakeRuntimeState {
    starts: AtomicUsize,
    stops: AtomicUsize,
    reconfigures: AtomicUsize,
    snapshots: Mutex<Vec<RuntimeSnapshot>>,
}

#[derive(Default)]
struct FakeDuplexRunner {
    state: Arc<FakeRuntimeState>,
    fail_stop: bool,
    fail_reconfigure: bool,
}

impl DuplexRunner for FakeDuplexRunner {
    fn start(
        &self,
        admitted: AdmittedDuplex,
        _deadline: tokio::time::Instant,
    ) -> DuplexStartResult {
        self.state.starts.fetch_add(1, Ordering::SeqCst);
        self.state
            .snapshots
            .lock()
            .unwrap()
            .push(admitted.snapshot().clone());
        Ok(Box::new(FakeActiveDuplex {
            state: self.state.clone(),
            fail_stop: self.fail_stop,
            fail_reconfigure: self.fail_reconfigure,
        }))
    }
}

struct FakeActiveDuplex {
    state: Arc<FakeRuntimeState>,
    fail_stop: bool,
    fail_reconfigure: bool,
}

#[derive(Default)]
struct VoiceAdmissionProbe {
    runner: FakeDuplexRunner,
    facts: AtomicUsize,
    maintenance: AtomicUsize,
    mix: AtomicUsize,
    reject_facts: AtomicBool,
    completion: Mutex<Option<(u64, Arc<dyn translator_daemon::DuplexCompletionObserver>)>>,
}

impl VoiceAdmissionProbe {
    fn effects(&self) -> [usize; 6] {
        [
            self.runner.state.starts.load(Ordering::SeqCst),
            self.runner.state.stops.load(Ordering::SeqCst),
            self.runner.state.reconfigures.load(Ordering::SeqCst),
            self.facts.load(Ordering::SeqCst),
            self.maintenance.load(Ordering::SeqCst),
            self.mix.load(Ordering::SeqCst),
        ]
    }
}

impl DuplexRunner for VoiceAdmissionProbe {
    fn start(&self, admitted: AdmittedDuplex, deadline: tokio::time::Instant) -> DuplexStartResult {
        self.runner.start(admitted, deadline)
    }

    fn start_supervised(
        &self,
        admitted: AdmittedDuplex,
        generation: u64,
        completion: Arc<dyn translator_daemon::DuplexCompletionObserver>,
        deadline: tokio::time::Instant,
    ) -> DuplexStartResult {
        *self.completion.lock().unwrap() = Some((generation, completion));
        self.start(admitted, deadline)
    }
}

impl translator_daemon::RuntimeFactsSource for VoiceAdmissionProbe {
    fn inspect(
        &self,
        deadline: std::time::Instant,
    ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
        self.facts.fetch_add(1, Ordering::SeqCst);
        if self.reject_facts.load(Ordering::SeqCst) {
            Err(translator_daemon::FactsError::DiscoveryFailed)
        } else {
            translator_daemon::RuntimeFactsSource::inspect(&NoopFacts, deadline)
        }
    }
}

impl RuntimeMaintenance for VoiceAdmissionProbe {
    fn refresh(&self, _store: &RuntimeStore) -> Result<(), ControlFailure> {
        self.maintenance.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl AudioMixController for VoiceAdmissionProbe {
    fn apply_desired(
        &self,
        _volumes: AudioMixState,
        _mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        self.mix.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn reconcile_committed(&self, _mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.mix.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn recover_committed(&self, _mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.mix.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn voice_override_http_rejects_before_effects_and_preserves_running_or_failed_owner() {
    use translator_daemon::{
        ControlCommand, DirectionRuntimeFailure, DirectionRuntimeStatus, ProviderPatch,
    };
    let mut failures = Vec::new();
    for provider in [ProviderId::Local, ProviderId::Openai] {
        for state in ["stopped", "running", "failed_direction"] {
            for value in ["unapproved-voice", "", " \t"] {
                for (model, voice) in [(true, false), (false, true), (true, true)] {
                    let store = RuntimeStore::default();
                    let probe = Arc::new(VoiceAdmissionProbe::default());
                    let controller = ControlApplication::spawn(
                        store.clone(),
                        probe.clone(),
                        AudioOperationGate::new(),
                        probe.clone(),
                        probe.clone(),
                        Some(probe.clone()),
                    );
                    controller
                        .execute(ControlCommand::PatchProvider(ProviderPatch {
                            provider_id: provider,
                            cloud_opt_in: Some(true),
                        }))
                        .await
                        .unwrap();
                    if state != "stopped" {
                        controller.execute(ControlCommand::Start).await.unwrap();
                    }
                    if state == "failed_direction" {
                        let (generation, observer) =
                            probe.completion.lock().unwrap().clone().unwrap();
                        observer.direction_status_changed(
                            generation,
                            translator_core::AudioDirection::Speaker,
                            1,
                            DirectionRuntimeStatus::Failed,
                            Some(DirectionRuntimeFailure::RestartExhausted),
                        );
                        tokio::time::timeout(Duration::from_secs(1), async {
                            while store.snapshot().directions[1].runtime_status
                                != DirectionRuntimeStatus::Failed
                            {
                                tokio::task::yield_now().await;
                            }
                        })
                        .await
                        .unwrap();
                    }
                    let router = build_router_with_controllers(
                        store.clone(),
                        ControlToken::parse(TOKEN).unwrap(),
                        ApiLimits::default(),
                        ApiControllers {
                            translation: Some(controller.clone()),
                            ..ApiControllers::default()
                        },
                    );
                    let response = router
                        .clone()
                        .oneshot(request(
                            Method::GET,
                            "/v1/events/stream",
                            Some(TOKEN),
                            Body::empty(),
                        ))
                        .await
                        .unwrap();
                    let mut events = SseReader::new(response.into_body());
                    assert!(
                        events
                            .next_record()
                            .await
                            .unwrap()
                            .contains("event: snapshot")
                    );
                    let before = serde_json::to_value(store.snapshot()).unwrap();
                    let effects_before = probe.effects();
                    let generation_before = probe
                        .completion
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map(|entry| entry.0);
                    let mut payload = serde_json::json!({"direction_id":"speaker","voice_profile":{"language":"ru","gender":"male","engine":"piper"}});
                    if model {
                        payload["voice_profile"]["model_path"] = value.into();
                    }
                    if voice {
                        payload["voice_profile"]["provider_voice_id"] = value.into();
                    }
                    let response = router
                        .oneshot(request(
                            Method::PATCH,
                            "/v1/voice-profiles",
                            Some(TOKEN),
                            Body::from(payload.to_string()),
                        ))
                        .await
                        .unwrap();
                    let status = response.status();
                    let body = json_body(response).await;
                    let after = serde_json::to_value(store.snapshot()).unwrap();
                    let effects_after = probe.effects();
                    let generation_after = probe
                        .completion
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map(|entry| entry.0);
                    let event_emitted =
                        tokio::time::timeout(Duration::from_millis(10), events.next_record())
                            .await
                            .is_ok();
                    controller.shutdown().await.unwrap();
                    if status != StatusCode::BAD_REQUEST
                        || body["code"] != "voice_profile_override_unsupported"
                        || body.to_string().contains("unapproved")
                        || before != after
                        || effects_before != effects_after
                        || generation_before != generation_after
                        || event_emitted
                    {
                        failures.push((
                            provider,
                            state,
                            value.len(),
                            model,
                            voice,
                            status.as_u16(),
                            effects_before,
                            effects_after,
                            before == after,
                            event_emitted,
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "override rejection/owner violations: {failures:?}"
    );
}

#[tokio::test]
async fn voice_override_http_validation_precedes_facts_and_preserves_language_json_priority() {
    use translator_daemon::ControlCommand;
    for (profile, expected) in [
        (
            serde_json::json!({"language":"en","gender":"male","engine":"piper","provider_voice_id":"unapproved-voice"}),
            "voice_language_mismatch",
        ),
        (
            serde_json::json!({"language":"ru","gender":"male","engine":"piper","provider_voice_id":42}),
            "invalid_json",
        ),
        (
            serde_json::json!({"language":"ru","gender":"male","engine":"piper","model_path":"unapproved.onnx"}),
            "voice_profile_override_unsupported",
        ),
    ] {
        let store = RuntimeStore::default();
        let probe = Arc::new(VoiceAdmissionProbe::default());
        let controller = ControlApplication::spawn(
            store.clone(),
            probe.clone(),
            AudioOperationGate::new(),
            probe.clone(),
            probe.clone(),
            Some(probe.clone()),
        );
        controller.execute(ControlCommand::Start).await.unwrap();
        probe.reject_facts.store(true, Ordering::SeqCst);
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let effects_before = probe.effects();
        let router = build_router_with_controllers(
            store.clone(),
            ControlToken::parse(TOKEN).unwrap(),
            ApiLimits::default(),
            ApiControllers {
                translation: Some(controller.clone()),
                ..ApiControllers::default()
            },
        );
        let response = router
            .oneshot(request(
                Method::PATCH,
                "/v1/voice-profiles",
                Some(TOKEN),
                Body::from(
                    serde_json::json!({"direction_id":"speaker","voice_profile":profile})
                        .to_string(),
                ),
            ))
            .await
            .unwrap();
        let status = response.status();
        let body = json_body(response).await;
        let after = serde_json::to_value(store.snapshot()).unwrap();
        let effects_after = probe.effects();
        controller.shutdown().await.unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST, "{expected}");
        assert_eq!(body["code"], expected);
        assert_eq!(before, after);
        assert_eq!(effects_before, effects_after);
    }
}

#[tokio::test]
async fn voice_override_http_absent_and_null_builtin_profiles_remain_supported() {
    for profile in [
        serde_json::json!({"language":"ru","gender":"female","engine":"piper"}),
        serde_json::json!({"language":"ru","gender":"female","engine":"piper","model_path":null,"provider_voice_id":null}),
    ] {
        let store = RuntimeStore::default();
        let probe = Arc::new(VoiceAdmissionProbe::default());
        let controller = ControlApplication::spawn(
            store.clone(),
            probe.clone(),
            AudioOperationGate::new(),
            probe.clone(),
            probe.clone(),
            Some(probe.clone()),
        );
        let router = build_router_with_controllers(
            store,
            ControlToken::parse(TOKEN).unwrap(),
            ApiLimits::default(),
            ApiControllers {
                translation: Some(controller.clone()),
                ..ApiControllers::default()
            },
        );
        let response = router
            .oneshot(request(
                Method::PATCH,
                "/v1/voice-profiles",
                Some(TOKEN),
                Body::from(
                    serde_json::json!({"direction_id":"speaker","voice_profile":profile})
                        .to_string(),
                ),
            ))
            .await
            .unwrap();
        let status = response.status();
        let body = json_body(response).await;
        let effects = probe.effects();
        controller.shutdown().await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["directions"][1]["voice_profile"]["gender"], "female");
        assert!(
            body["directions"][1]["voice_profile"]
                .get("model_path")
                .is_none()
        );
        assert!(
            body["directions"][1]["voice_profile"]
                .get("provider_voice_id")
                .is_none()
        );
        assert_eq!(effects, [0; 6]);
    }
}

impl ActiveDuplexRuntime for FakeActiveDuplex {
    fn reconfigure(
        &mut self,
        admitted: AdmittedDuplex,
        _deadline: tokio::time::Instant,
    ) -> Result<(), DuplexRuntimeError> {
        self.state.reconfigures.fetch_add(1, Ordering::SeqCst);
        if self.fail_reconfigure {
            return Err(DuplexRuntimeError::ReconfigureFailed);
        }
        self.state
            .snapshots
            .lock()
            .unwrap()
            .push(admitted.snapshot().clone());
        Ok(())
    }

    fn stop(&mut self, _deadline: tokio::time::Instant) -> Result<(), DuplexRuntimeError> {
        self.state.stops.fetch_add(1, Ordering::SeqCst);
        if self.fail_stop {
            return Err(DuplexRuntimeError::StopFailed);
        }
        Ok(())
    }
}

struct RejectingStartRunner;

impl DuplexRunner for RejectingStartRunner {
    fn start(
        &self,
        _admitted: AdmittedDuplex,
        _deadline: tokio::time::Instant,
    ) -> DuplexStartResult {
        Err(DuplexStartFailure::rejected(
            DuplexRuntimeError::StartFailed,
        ))
    }
}

#[derive(Default)]
struct CleanupPendingStartState {
    starts: AtomicUsize,
    stops: AtomicUsize,
    active: AtomicBool,
}

struct CleanupPendingStartRunner {
    state: Arc<CleanupPendingStartState>,
}

impl DuplexRunner for CleanupPendingStartRunner {
    fn start(
        &self,
        _admitted: AdmittedDuplex,
        deadline: tokio::time::Instant,
    ) -> DuplexStartResult {
        self.state.starts.fetch_add(1, Ordering::SeqCst);
        self.state.active.store(true, Ordering::SeqCst);
        let mut cleanup = RetryableStartCleanup {
            state: self.state.clone(),
            failures_remaining: 2,
        };
        assert_eq!(cleanup.stop(deadline), Err(DuplexRuntimeError::StopFailed));
        Err(DuplexStartFailure::cleanup_pending(
            DuplexRuntimeError::StartFailed,
            Box::new(cleanup),
        ))
    }
}

struct RetryableStartCleanup {
    state: Arc<CleanupPendingStartState>,
    failures_remaining: usize,
}

impl ActiveDuplexRuntime for RetryableStartCleanup {
    fn stop(&mut self, _deadline: tokio::time::Instant) -> Result<(), DuplexRuntimeError> {
        self.state.stops.fetch_add(1, Ordering::SeqCst);
        if self.failures_remaining > 0 {
            self.failures_remaining -= 1;
            return Err(DuplexRuntimeError::StopFailed);
        }
        self.state.active.store(false, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Default)]
struct BlockingStartRunner {
    entered: AtomicBool,
    release: (Mutex<bool>, Condvar),
}

impl BlockingStartRunner {
    fn release(&self) {
        *self.release.0.lock().unwrap() = true;
        self.release.1.notify_all();
    }
}

impl DuplexRunner for BlockingStartRunner {
    fn start(
        &self,
        _admitted: AdmittedDuplex,
        _deadline: tokio::time::Instant,
    ) -> DuplexStartResult {
        self.entered.store(true, Ordering::Release);
        let released = self.release.0.lock().unwrap();
        let (released, timeout) = self
            .release
            .1
            .wait_timeout_while(released, Duration::from_secs(2), |released| !*released)
            .unwrap();
        if timeout.timed_out() && !*released {
            return Err(DuplexStartFailure::rejected(
                DuplexRuntimeError::StartFailed,
            ));
        }
        Ok(Box::new(FakeActiveDuplex {
            state: Arc::new(FakeRuntimeState::default()),
            fail_stop: false,
            fail_reconfigure: false,
        }))
    }
}

fn control_application(
    store: RuntimeStore,
    runner: Arc<dyn DuplexRunner>,
) -> Arc<ControlApplication> {
    control_application_with_mix(store, runner, None)
}

fn control_application_with_mix(
    store: RuntimeStore,
    runner: Arc<dyn DuplexRunner>,
    audio_mix: Option<Arc<dyn AudioMixController>>,
) -> Arc<ControlApplication> {
    ControlApplication::spawn(
        store,
        runner,
        AudioOperationGate::new(),
        Arc::new(NoopFacts),
        Arc::new(NoopFacts),
        audio_mix,
    )
}

fn control_router(
    store: RuntimeStore,
    audio_mix: Option<Arc<dyn AudioMixController>>,
) -> (axum::Router, Arc<ControlApplication>) {
    let controller = control_application_with_mix(
        store.clone(),
        Arc::new(FakeDuplexRunner::default()),
        audio_mix,
    );
    let router = build_router_with_controllers(
        store,
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );
    (router, controller)
}

struct LeakyRoundTripController {
    marker: &'static str,
}

impl RoundTripController for LeakyRoundTripController {
    fn start(&self) -> Result<RoundTripSelfTestState, ControlFailure> {
        Ok(self.state())
    }

    fn stop(&self) -> Result<RoundTripSelfTestState, ControlFailure> {
        Ok(self.state())
    }
}

impl LeakyRoundTripController {
    fn state(&self) -> RoundTripSelfTestState {
        RoundTripSelfTestState {
            availability: "available",
            preconditions: None,
            status: RoundTripStatus {
                debug_text: Some(RoundTripDebugText {
                    transcript: self.marker.to_owned(),
                    translation: self.marker.to_owned(),
                }),
                ..RoundTripStatus::default()
            },
        }
    }
}

fn app() -> axum::Router {
    app_with(RuntimeStore::default(), ApiLimits::default())
}

fn app_with(store: RuntimeStore, limits: ApiLimits) -> axum::Router {
    build_router(store, ControlToken::parse(TOKEN).unwrap(), limits)
}

fn request(method: Method, uri: &str, token: Option<&str>, body: Body) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(token) = token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    builder
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .unwrap()
}

async fn json_body(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn assert_problem(
    response: axum::response::Response,
    status: StatusCode,
    title: &str,
    code: &str,
) -> Value {
    assert_eq!(response.status(), status);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/problem+json"
    );
    let body = json_body(response).await;
    assert_eq!(body["type"], format!("urn:translator:error:{code}"));
    assert_eq!(body["title"], title);
    assert_eq!(body["status"], status.as_u16());
    assert_eq!(body["code"], code);
    body
}

struct SseReader {
    body: Body,
    buffered: Vec<u8>,
}

impl SseReader {
    fn new(body: Body) -> Self {
        Self {
            body,
            buffered: Vec::new(),
        }
    }

    async fn next_record(&mut self) -> Option<String> {
        loop {
            if let Some(end) = self
                .buffered
                .windows(2)
                .position(|window| window == b"\n\n")
            {
                let record = self.buffered.drain(..end + 2).collect::<Vec<_>>();
                return Some(String::from_utf8(record).unwrap());
            }
            let frame = tokio::time::timeout(Duration::from_secs(2), self.body.frame())
                .await
                .expect("SSE record timed out")?
                .unwrap();
            self.buffered.extend_from_slice(&frame.into_data().unwrap());
        }
    }
}

#[tokio::test]
async fn invalid_bearer_is_rejected_before_body_limit_or_json_parsing() {
    let body_polls = Arc::new(AtomicUsize::new(0));
    let observed_polls = Arc::clone(&body_polls);
    let unreadable_body = Body::from_stream(stream::poll_fn(move |_| {
        observed_polls.fetch_add(1, Ordering::SeqCst);
        panic!("unauthorized request body must not be polled");
        #[allow(unreachable_code)]
        std::task::Poll::Ready(Some(Ok::<Bytes, Infallible>(Bytes::new())))
    }));

    let response = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-text",
            Some("wrong-token"),
            unreadable_body,
        ))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::UNAUTHORIZED,
        "Unauthorized",
        "invalid_bearer",
    )
    .await;
    assert_eq!(body_polls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn authorized_control_body_is_limited_to_64_kib() {
    let response = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-text",
            Some(TOKEN),
            Body::from(vec![b'x'; 65_537]),
        ))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::PAYLOAD_TOO_LARGE,
        "Payload Too Large",
        "body_too_large",
    )
    .await;
}

#[tokio::test]
async fn malformed_authorized_json_uses_privacy_safe_problem_details() {
    let marker = "private-json-marker";
    let response = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-text",
            Some(TOKEN),
            Body::from(format!("{{\"enabled\":{marker}}}")),
        ))
        .await
        .unwrap();

    let body = assert_problem(
        response,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "invalid_json",
    )
    .await;
    assert!(!body.to_string().contains(marker));
}

#[tokio::test]
async fn missing_bearer_cannot_read_status() {
    let response = app()
        .oneshot(request(Method::GET, "/v1/status", None, Body::empty()))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::UNAUTHORIZED,
        "Unauthorized",
        "invalid_bearer",
    )
    .await;
}

#[tokio::test]
async fn valid_bearer_reads_privacy_safe_status() {
    let response = app()
        .oneshot(request(
            Method::GET,
            "/v1/status",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["translation_running"], false);
    assert_eq!(body["runtime_status"], "stopped");
    assert_eq!(body["audio_mix_knowledge"], "unapplied");
    assert_eq!(body["debug_text_enabled"], false);
    assert_eq!(body["debug_capture_enabled"], false);
    assert_eq!(body["provider_id"], "local");
    assert_eq!(body["directions"][0]["direction_id"], "microphone");
    assert_eq!(body["directions"][0]["source_language"], "ru");
    assert_eq!(body["directions"][0]["target_language"], "en");
    assert_eq!(body["directions"][0]["enabled"], true);
    assert_eq!(body["directions"][1]["enabled"], true);
    assert_eq!(body["latency_policy"][0]["current_mode"], "quality_first");
    assert_eq!(body["audio_mix"]["microphone_original_percent"], 0);
    assert_eq!(body["audio_mix"]["microphone_translation_percent"], 100);
    assert_eq!(body["audio_mix"]["speaker_original_percent"], 0);
    assert_eq!(body["audio_mix"]["speaker_translation_percent"], 100);
    assert!(body.get("control_token").is_none());
}

#[tokio::test]
async fn graph_and_route_reads_report_typed_unavailable_state() {
    let router = app();
    let graph = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/audio-graph",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let graph = json_body(graph).await;
    assert_eq!(graph["available"], false);
    assert!(graph["value"].is_null());

    let routes = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/routes",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let routes = json_body(routes).await;
    assert_eq!(routes["available"], false);
    assert!(routes["value"].is_null());

    let candidates = router
        .oneshot(request(
            Method::GET,
            "/v1/routes/candidates",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(candidates).await, serde_json::json!([]));
}

#[tokio::test]
async fn audio_graph_is_available_only_when_health_is_ready() {
    for (health, expected) in [
        (GraphHealth::Ready, true),
        (GraphHealth::Degraded, false),
        (GraphHealth::Error, false),
    ] {
        let store = RuntimeStore::default();
        store.set_audio_graph(AudioGraphState {
            health,
            endpoints: Vec::new(),
            owned_module_ids: Vec::new(),
            safe_error: None,
        });
        let response = app_with(store, ApiLimits::default())
            .oneshot(request(
                Method::GET,
                "/v1/audio-graph",
                Some(TOKEN),
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(json_body(response).await["available"], expected);
    }
}

#[tokio::test]
async fn self_test_controller_text_obeys_global_debug_text_lifecycle() {
    let marker = "private-controller-round-trip-marker";
    let store = RuntimeStore::default();
    let controller = control_application(store.clone(), Arc::new(FakeDuplexRunner::default()));
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            round_trip: Some(Arc::new(LeakyRoundTripController { marker })),
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let hidden_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(hidden_start.status(), StatusCode::OK);
    assert!(!json_body(hidden_start).await.to_string().contains(marker));
    assert!(
        !serde_json::to_string(&store.snapshot())
            .unwrap()
            .contains(marker)
    );

    store.set_debug_text_enabled(true);
    let visible_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(visible_start.status(), StatusCode::OK);
    assert!(json_body(visible_start).await.to_string().contains(marker));

    let stopped = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(stopped.status(), StatusCode::OK);
    assert!(!json_body(stopped).await.to_string().contains(marker));

    let stopped_status = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/self-test/round-trip",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(stopped_status.status(), StatusCode::OK);
    assert!(!json_body(stopped_status).await.to_string().contains(marker));

    let visible_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert!(json_body(visible_start).await.to_string().contains(marker));

    let provider_switched = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/provider",
            Some(TOKEN),
            Body::from(r#"{"provider_id":"openai","cloud_opt_in":true}"#),
        ))
        .await
        .unwrap();
    assert_eq!(provider_switched.status(), StatusCode::OK);
    assert!(
        !json_body(provider_switched)
            .await
            .to_string()
            .contains(marker)
    );
    let after_provider_switch = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/self-test/round-trip",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(after_provider_switch.status(), StatusCode::OK);
    assert!(
        !json_body(after_provider_switch)
            .await
            .to_string()
            .contains(marker)
    );

    let visible_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert!(json_body(visible_start).await.to_string().contains(marker));

    store.set_debug_text_enabled(false);
    store.set_debug_text_enabled(true);
    let after_reenable = router
        .oneshot(request(
            Method::GET,
            "/v1/self-test/round-trip",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(after_reenable.status(), StatusCode::OK);
    assert!(!json_body(after_reenable).await.to_string().contains(marker));
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn typed_control_patches_update_only_the_selected_runtime_contract() {
    let (router, controller) =
        control_router(RuntimeStore::default(), Some(Arc::new(NoopAudioMix)));
    let direction = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"microphone","source_language":"en","target_language":"ru"}"#,
            ),
        ))
        .await
        .unwrap();
    let direction = json_body(direction).await;
    assert_eq!(direction["directions"][0]["source_language"], "en");
    assert_eq!(direction["directions"][0]["target_language"], "ru");
    assert_eq!(direction["directions"][0]["enabled"], true);
    assert_eq!(direction["directions"][1]["source_language"], "en");

    let disabled = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(r#"{"direction_id":"speaker","enabled":false}"#),
        ))
        .await
        .unwrap();
    let disabled = json_body(disabled).await;
    assert_eq!(disabled["directions"][0]["enabled"], true);
    assert_eq!(disabled["directions"][0]["source_language"], "en");
    assert_eq!(disabled["directions"][1]["enabled"], false);
    assert_eq!(disabled["directions"][1]["source_language"], "en");

    let blocked_provider = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/provider",
            Some(TOKEN),
            Body::from(r#"{"provider_id":"openai"}"#),
        ))
        .await
        .unwrap();
    assert_eq!(blocked_provider.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(blocked_provider).await["type"],
        "urn:translator:error:cloud_provider_opt_in_required"
    );

    let provider = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/provider",
            Some(TOKEN),
            Body::from(r#"{"provider_id":"openai","cloud_opt_in":true}"#),
        ))
        .await
        .unwrap();
    let provider = json_body(provider).await;
    assert_eq!(provider["provider_id"], "openai");
    assert_eq!(provider["audio_leaves_machine"], true);

    let latency = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/latency-policy",
            Some(TOKEN),
            Body::from(r#"{"direction_id":"speaker","current_mode":"balanced"}"#),
        ))
        .await
        .unwrap();
    let latency = json_body(latency).await;
    assert_eq!(
        latency["latency_policy"][0]["current_mode"],
        "quality_first"
    );
    assert_eq!(latency["latency_policy"][1]["current_mode"], "balanced");
    assert_eq!(latency["audio_mix"]["speaker_original_percent"], 0);

    let audio_mix = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/audio-mix",
            Some(TOKEN),
            Body::from(r#"{"speaker_original_percent":55,"microphone_translation_percent":72}"#),
        ))
        .await
        .unwrap();
    let audio_mix = json_body(audio_mix).await;
    assert_eq!(audio_mix["audio_mix"]["speaker_original_percent"], 55);
    assert_eq!(audio_mix["audio_mix"]["microphone_translation_percent"], 72);
    assert_eq!(audio_mix["audio_mix"]["microphone_original_percent"], 0);
    assert_eq!(audio_mix["audio_mix"]["speaker_translation_percent"], 100);

    let voice = router
        .oneshot(request(
            Method::PATCH,
            "/v1/voice-profiles",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"speaker","voice_profile":{"language":"ru","gender":"female","engine":"piper"}}"#,
            ),
        ))
        .await
        .unwrap();
    assert_eq!(
        json_body(voice).await["directions"][1]["voice_profile"]["gender"],
        "female"
    );
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn invalid_direction_and_voice_language_are_typed_problem_details() {
    let (router, controller) = control_router(RuntimeStore::default(), None);
    let invalid_pair = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"microphone","source_language":"ru","target_language":"ru"}"#,
            ),
        ))
        .await
        .unwrap();
    assert_problem(
        invalid_pair,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "invalid_language_pair",
    )
    .await;

    let voice_mismatch = router
        .oneshot(request(
            Method::PATCH,
            "/v1/voice-profiles",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"speaker","voice_profile":{"language":"en","gender":"female","engine":"piper"}}"#,
            ),
        ))
        .await
        .unwrap();
    assert_problem(
        voice_mismatch,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "voice_language_mismatch",
    )
    .await;
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn audio_mix_rejects_out_of_range_and_unknown_fields() {
    let (router, controller) =
        control_router(RuntimeStore::default(), Some(Arc::new(NoopAudioMix)));
    let too_loud = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/audio-mix",
            Some(TOKEN),
            Body::from(r#"{"speaker_translation_percent":101}"#),
        ))
        .await
        .unwrap();
    assert_problem(
        too_loud,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "invalid_audio_mix_volume",
    )
    .await;

    let unknown = router
        .oneshot(request(
            Method::PATCH,
            "/v1/audio-mix",
            Some(TOKEN),
            Body::from(r#"{"speaker_translation_percent":80,"master_percent":80}"#),
        ))
        .await
        .unwrap();
    assert_problem(
        unknown,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "invalid_json",
    )
    .await;
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn debug_capture_enable_fails_closed_without_hardened_store() {
    let response = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-capture",
            Some(TOKEN),
            Body::from(r#"{"enabled":true}"#),
        ))
        .await
        .unwrap();
    assert_problem(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "debug_capture_unavailable",
    )
    .await;
}

#[tokio::test]
async fn debug_capture_api_creates_and_closes_real_exclusive_session() {
    let temp = tempfile::tempdir().unwrap();
    let store = RuntimeStore::default();
    store.configure_debug_capture(
        DebugCaptureStore::open(temp.path(), DebugCaptureLimits::new(1_000, 1_024, 0)).unwrap(),
    );
    let router = app_with(store, ApiLimits::default());

    let enabled = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-capture",
            Some(TOKEN),
            Body::from(r#"{"enabled":true}"#),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(enabled).await["debug_capture_enabled"], true);
    let captures = std::fs::read_dir(temp.path().join("translator/debug"))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(captures.len(), 1);

    let disabled = router
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-capture",
            Some(TOKEN),
            Body::from(r#"{"enabled":false}"#),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(disabled).await["debug_capture_enabled"], false);
}

#[tokio::test]
async fn unavailable_manual_route_controller_fails_closed() {
    let response = app()
        .oneshot(request(
            Method::POST,
            "/v1/routes/manual-override",
            Some(TOKEN),
            Body::from(r#"{"stream_id":42}"#),
        ))
        .await
        .unwrap();
    assert_problem(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "routing_controller_unavailable",
    )
    .await;
}

#[tokio::test]
async fn manual_route_controller_result_is_published_to_runtime_state() {
    let controller = Arc::new(FakeManualRoutes {
        selected_stream: AtomicUsize::new(0),
    });
    let router = build_router_with_manual_routes(
        RuntimeStore::default(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        Some(controller.clone()),
    );
    let response = router
        .oneshot(request(
            Method::POST,
            "/v1/routes/manual-override",
            Some(TOKEN),
            Body::from(r#"{"stream_id":42}"#),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        json_body(response).await["routes"]["resolution"],
        "no_candidate"
    );
    assert_eq!(controller.selected_stream.load(Ordering::SeqCst), 42);
}

#[tokio::test]
async fn fifth_sse_subscriber_is_rejected_and_disconnect_releases_permit() {
    let router = app();
    let mut responses = Vec::new();
    for _ in 0..4 {
        let response = router
            .clone()
            .oneshot(request(
                Method::GET,
                "/v1/events/stream",
                Some(TOKEN),
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        responses.push(response);
    }

    let fifth = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(
        fifth,
        StatusCode::TOO_MANY_REQUESTS,
        "Too Many Requests",
        "sse_subscriber_limit",
    )
    .await;

    responses.pop();
    tokio::task::yield_now().await;

    let replacement = router
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(replacement.status(), StatusCode::OK);
}

#[tokio::test]
async fn partially_consumed_sse_holds_permit_until_body_is_dropped() {
    let router = app();
    let mut responses = Vec::new();
    for _ in 0..4 {
        let response = router
            .clone()
            .oneshot(request(
                Method::GET,
                "/v1/events/stream",
                Some(TOKEN),
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        responses.push(response);
    }

    let mut partial = responses.pop().unwrap();
    let mut partial_reader = SseReader::new(std::mem::take(partial.body_mut()));
    let first = partial_reader.next_record().await.unwrap();
    assert!(first.contains("event: snapshot"));

    let rejected = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(
        rejected,
        StatusCode::TOO_MANY_REQUESTS,
        "Too Many Requests",
        "sse_subscriber_limit",
    )
    .await;

    drop(partial_reader);
    drop(partial);
    tokio::task::yield_now().await;
    let replacement = router
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(replacement.status(), StatusCode::OK);
}

#[tokio::test]
async fn broadcast_lag_emits_resync_then_exactly_one_fresh_snapshot() {
    let store = RuntimeStore::with_event_capacity(1);
    let router = app_with(store.clone(), ApiLimits::default());
    let response = router
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let mut events = SseReader::new(response.into_body());
    assert!(
        events
            .next_record()
            .await
            .unwrap()
            .contains("event: snapshot")
    );

    store.publish_snapshot_changed();
    store.publish_snapshot_changed();
    store.publish_snapshot_changed();

    let resync = events.next_record().await.unwrap();
    let snapshot = events.next_record().await.unwrap();
    assert!(resync.contains("event: resync_required"));
    assert!(!resync.contains("event: snapshot"));
    assert!(snapshot.contains("event: snapshot"));
    assert_eq!(snapshot.matches("event: snapshot").count(), 1);
}

#[tokio::test]
async fn event_bus_shutdown_terminates_stream_cleanly() {
    let store = RuntimeStore::default();
    let router = app_with(store.clone(), ApiLimits::default());
    let response = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let mut events = SseReader::new(response.into_body());
    assert!(
        events
            .next_record()
            .await
            .unwrap()
            .contains("event: snapshot")
    );

    store.shutdown_events();
    assert!(events.next_record().await.is_none());

    let replacement = router
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(replacement.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test(start_paused = true)]
async fn sse_keepalive_is_comment_only() {
    let limits = ApiLimits::default().with_sse_keepalive(Duration::from_millis(10));
    let response = app_with(RuntimeStore::default(), limits)
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let mut events = SseReader::new(response.into_body());
    assert!(
        events
            .next_record()
            .await
            .unwrap()
            .contains("event: snapshot")
    );

    tokio::time::advance(Duration::from_millis(11)).await;
    let keepalive = events.next_record().await.unwrap();
    assert!(keepalive.starts_with(':'));
    assert!(!keepalive.contains("data:"));
}

#[tokio::test]
async fn rejecting_fifth_sse_subscriber_does_not_interrupt_existing_streams() {
    let store = RuntimeStore::default();
    let router = app_with(store.clone(), ApiLimits::default());
    let mut responses = Vec::new();
    for _ in 0..4 {
        let response = router
            .clone()
            .oneshot(request(
                Method::GET,
                "/v1/events/stream",
                Some(TOKEN),
                Body::empty(),
            ))
            .await
            .unwrap();
        responses.push(response);
    }

    let fifth = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(
        fifth,
        StatusCode::TOO_MANY_REQUESTS,
        "Too Many Requests",
        "sse_subscriber_limit",
    )
    .await;

    store.publish_snapshot_changed();
    for response in &mut responses {
        let mut events = SseReader::new(std::mem::take(response.body_mut()));
        assert!(
            events
                .next_record()
                .await
                .unwrap()
                .contains("event: snapshot")
        );
        assert!(
            events
                .next_record()
                .await
                .unwrap()
                .contains("event: snapshot")
        );
    }
}

#[tokio::test]
async fn self_test_start_is_typed_unavailable_until_task7_controller_exists() {
    let response = app()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "self_test_unavailable",
    )
    .await;
}

#[tokio::test]
async fn self_test_stop_is_typed_unavailable_until_task7_controller_exists() {
    let response = app()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "self_test_unavailable",
    )
    .await;
}

#[tokio::test]
async fn translation_mutations_fail_closed_without_provider_controller() {
    for path in ["/v1/translation/start", "/v1/translation/stop"] {
        let response = app()
            .oneshot(request(Method::POST, path, Some(TOKEN), Body::from("{}")))
            .await
            .unwrap();
        assert_problem(
            response,
            StatusCode::SERVICE_UNAVAILABLE,
            "Service Unavailable",
            "translation_controller_unavailable",
        )
        .await;
    }
}

#[tokio::test]
async fn translation_controller_success_owns_running_state_and_stop_clears_debug_text() {
    let store = RuntimeStore::default();
    store.set_debug_text_enabled(true);
    store.record_debug_text(DebugTextEvent::new("private-session-marker", "translation"));
    let runner = Arc::new(FakeDuplexRunner::default());
    let controller = control_application(store.clone(), runner.clone());
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(started).await["translation_running"], true);
    let stopped = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(stopped).await["translation_running"], false);
    assert_eq!(store.debug_text_status().event_count, 0);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.stops.load(Ordering::SeqCst), 1);
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn start_translation_inspects_fresh_facts_without_audio_maintenance() {
    let store = RuntimeStore::default();
    store.set_audio_graph(AudioGraphState {
        health: GraphHealth::Degraded,
        endpoints: Vec::new(),
        owned_module_ids: Vec::new(),
        safe_error: None,
    });
    let manual_routes = Arc::new(RefreshingManualRoutes {
        refreshes: AtomicUsize::new(0),
        running_at_refresh: AtomicUsize::new(usize::MAX),
    });
    let runner = Arc::new(FakeDuplexRunner::default());
    let controller = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        AudioOperationGate::new(),
        Arc::new(NoopFacts),
        manual_routes.clone(),
        None,
    );
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            manual_routes: Some(manual_routes.clone()),
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();

    assert_eq!(json_body(started).await["translation_running"], true);
    assert_eq!(manual_routes.refreshes.load(Ordering::SeqCst), 0);
    assert_eq!(
        manual_routes.running_at_refresh.load(Ordering::SeqCst),
        usize::MAX
    );
    {
        let snapshots = runner.state.snapshots.lock().unwrap();
        assert_eq!(
            snapshots[0].audio_graph.as_ref().map(|state| state.health),
            Some(GraphHealth::Ready)
        );
    }
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn stop_translation_refreshes_audio_state_after_clearing_running() {
    let store = RuntimeStore::default();
    let manual_routes = Arc::new(RefreshingManualRoutes {
        refreshes: AtomicUsize::new(0),
        running_at_refresh: AtomicUsize::new(usize::MAX),
    });
    let runner = Arc::new(FakeDuplexRunner::default());
    let controller = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        AudioOperationGate::new(),
        Arc::new(NoopFacts),
        manual_routes.clone(),
        None,
    );
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            manual_routes: Some(manual_routes.clone()),
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(started.status(), StatusCode::OK);
    let stopped = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();

    assert_eq!(json_body(stopped).await["translation_running"], false);
    assert_eq!(runner.state.stops.load(Ordering::SeqCst), 1);
    assert_eq!(manual_routes.refreshes.load(Ordering::SeqCst), 1);
    assert_eq!(manual_routes.running_at_refresh.load(Ordering::SeqCst), 0);
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn direction_patch_reconfigures_active_translation_for_channel_bypass() {
    let store = RuntimeStore::default();
    let runner = Arc::new(FakeDuplexRunner::default());
    let controller = control_application(store.clone(), runner.clone());
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(started).await["translation_running"], true);

    let disabled = router
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(r#"{"direction_id":"speaker","enabled":false}"#),
        ))
        .await
        .unwrap();
    let disabled = json_body(disabled).await;

    assert_eq!(disabled["translation_running"], true);
    assert_eq!(disabled["directions"][1]["enabled"], false);
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.reconfigures.load(Ordering::SeqCst), 1);
    {
        let snapshots = runner.state.snapshots.lock().unwrap();
        assert_eq!(snapshots.len(), 2);
        assert!(snapshots[0].directions[1].enabled);
        assert!(!snapshots[1].directions[1].enabled);
    }
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn voice_profile_patch_reconfigures_active_translation() {
    let store = RuntimeStore::default();
    let runner = Arc::new(FakeDuplexRunner::default());
    let controller = control_application(store.clone(), runner.clone());
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(started).await["translation_running"], true);

    let patched = router
        .oneshot(request(
            Method::PATCH,
            "/v1/voice-profiles",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"speaker","voice_profile":{"language":"ru","gender":"female","engine":"piper"}}"#,
            ),
        ))
        .await
        .unwrap();
    let patched = json_body(patched).await;

    assert_eq!(patched["translation_running"], true);
    assert_eq!(
        patched["directions"][1]["voice_profile"]["gender"],
        "female"
    );
    assert_eq!(runner.state.starts.load(Ordering::SeqCst), 1);
    assert_eq!(runner.state.reconfigures.load(Ordering::SeqCst), 1);
    {
        let snapshots = runner.state.snapshots.lock().unwrap();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(
            snapshots[0].directions[1].voice_profile.gender,
            translator_core::VoiceGender::Male
        );
        assert_eq!(
            snapshots[1].directions[1].voice_profile.gender,
            translator_core::VoiceGender::Female
        );
    }
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn translation_controller_failure_never_reports_a_false_runtime_transition() {
    let store = RuntimeStore::default();
    let controller = control_application(store.clone(), Arc::new(RejectingStartRunner));
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let rejected_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_problem(
        rejected_start,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "translation_start_failed",
    )
    .await;
    assert!(!store.snapshot().translation_running);

    controller.shutdown().await.unwrap();

    let runner = Arc::new(FakeDuplexRunner {
        fail_stop: true,
        ..FakeDuplexRunner::default()
    });
    let controller = control_application(store.clone(), runner);
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );
    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(started.status(), StatusCode::OK);
    let rejected_stop = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_problem(
        rejected_stop,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "translation_stop_failed",
    )
    .await;
    assert!(!store.snapshot().translation_running);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
}

#[tokio::test]
async fn failed_start_with_cleanup_owner_is_truthful_and_retryable_over_http() {
    let store = RuntimeStore::default();
    let state = Arc::new(CleanupPendingStartState::default());
    let controller = control_application(
        store.clone(),
        Arc::new(CleanupPendingStartRunner {
            state: state.clone(),
        }),
    );
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let failed = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_problem(
        failed,
        StatusCode::CONFLICT,
        "Conflict",
        "translation_cleanup_pending",
    )
    .await;
    assert!(!store.snapshot().translation_running);
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );
    assert!(state.active.load(Ordering::SeqCst));

    let overlapping = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_problem(
        overlapping,
        StatusCode::CONFLICT,
        "Conflict",
        "translation_cleanup_pending",
    )
    .await;
    assert_eq!(state.starts.load(Ordering::SeqCst), 1);

    let retry_failed = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_problem(
        retry_failed,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "translation_stop_failed",
    )
    .await;
    assert!(state.active.load(Ordering::SeqCst));
    assert_eq!(
        store.snapshot().runtime_status,
        RuntimeStatus::CleanupPending
    );

    let stopped = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(stopped.status(), StatusCode::OK);
    assert!(!state.active.load(Ordering::SeqCst));
    assert_eq!(state.stops.load(Ordering::SeqCst), 3);
    assert_eq!(store.snapshot().runtime_status, RuntimeStatus::Stopped);
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn control_ownership_applies_provider_change_before_publishing_it() {
    let store = RuntimeStore::default();
    let runner = Arc::new(FakeDuplexRunner::default());
    let controller = control_application(store.clone(), runner.clone());
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(started.status(), StatusCode::OK);

    let patched = router
        .oneshot(request(
            Method::PATCH,
            "/v1/provider",
            Some(TOKEN),
            Body::from(r#"{"provider_id":"openai","cloud_opt_in":true}"#),
        ))
        .await
        .unwrap();
    assert_eq!(patched.status(), StatusCode::OK);
    let published = json_body(patched).await;
    assert_eq!(published["provider_id"], "openai");
    assert_eq!(published["audio_leaves_machine"], true);
    assert_eq!(
        runner
            .state
            .snapshots
            .lock()
            .unwrap()
            .iter()
            .map(|snapshot| snapshot.provider_id)
            .collect::<Vec<_>>(),
        [ProviderId::Local, ProviderId::Openai]
    );
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn control_ownership_rolls_back_active_configuration_after_replacement_failure() {
    let store = RuntimeStore::default();
    let original = store.snapshot();
    let runner = Arc::new(FakeDuplexRunner {
        fail_reconfigure: true,
        ..FakeDuplexRunner::default()
    });
    let controller = control_application(store.clone(), runner);
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(started.status(), StatusCode::OK);

    let patched = router
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(r#"{"direction_id":"speaker","enabled":false}"#),
        ))
        .await
        .unwrap();
    assert_eq!(patched.status(), StatusCode::SERVICE_UNAVAILABLE);
    let after = store.snapshot();
    assert!(after.translation_running);
    assert_eq!(after.directions[1].enabled, original.directions[1].enabled);
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn control_ownership_finishes_an_admitted_start_after_caller_cancellation() {
    let store = RuntimeStore::default();
    let runner = Arc::new(BlockingStartRunner::default());
    let controller = control_application(store.clone(), runner.clone());
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let request_task = tokio::spawn(router.clone().oneshot(request(
        Method::POST,
        "/v1/translation/start",
        Some(TOKEN),
        Body::from("{}"),
    )));
    tokio::time::timeout(Duration::from_secs(1), async {
        while !runner.entered.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    request_task.abort();
    runner.release();
    let _ = request_task.await;

    tokio::time::timeout(Duration::from_secs(1), async {
        while !store.snapshot().translation_running {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the accepted start owner must commit after its caller disconnects");

    let stopped = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(stopped.status(), StatusCode::OK);
    controller.shutdown().await.unwrap();
}

#[tokio::test]
async fn aec_calibration_routes_share_bearer_authentication() {
    for (method, path) in [
        (Method::POST, "/v1/aec-calibration/start"),
        (Method::POST, "/v1/aec-calibration/cancel"),
        (Method::GET, "/v1/aec-calibration"),
    ] {
        let response = app()
            .oneshot(request(method, path, None, Body::from("{}")))
            .await
            .unwrap();
        assert_problem(
            response,
            StatusCode::UNAUTHORIZED,
            "Unauthorized",
            "invalid_bearer",
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn c9_s6_api_exposes_attempt_status_and_cancel_during_initial_inspection() {
    let engine = Arc::new(C9ApiHeldInspection {
        delegate: BlockingCalibrationEngine::default(),
        entered: tokio::sync::Notify::new(),
        released: (Mutex::new(false), Condvar::new()),
    });
    let release = C9ApiReleaseInspection(engine.clone());
    let coordinator = Arc::new(AecCalibrationCoordinator::new());
    let calibration = Arc::new(AecCalibrationController::new(
        coordinator.clone(),
        AudioOperationGate::new(),
        engine.clone(),
    ));
    let router = build_router_with_controllers(
        RuntimeStore::default(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            aec_calibration: Some(calibration.clone()),
            ..ApiControllers::default()
        },
    );
    let start_router = router.clone();
    let mut start = tokio::spawn(async move {
        start_router
            .oneshot(request(
                Method::POST,
                "/v1/aec-calibration/start",
                Some(TOKEN),
                Body::from("{}"),
            ))
            .await
            .unwrap()
    });
    tokio::time::timeout(Duration::from_secs(1), engine.entered.notified())
        .await
        .unwrap();
    let started = match tokio::time::timeout(Duration::from_millis(100), &mut start).await {
        Ok(started) => started.unwrap(),
        Err(_) => {
            drop(release);
            let _ = start.await;
            calibration.shutdown().await.unwrap();
            panic!("HTTP Start must return 202 and attempt ID before initial inspection completes");
        }
    };
    assert_eq!(started.status(), StatusCode::ACCEPTED);
    let started = json_body(started).await;
    let attempt_id = started["attempt_id"].as_str().unwrap();
    let status = tokio::time::timeout(
        Duration::from_millis(100),
        router.clone().oneshot(request(
            Method::GET,
            "/v1/aec-calibration",
            Some(TOKEN),
            Body::empty(),
        )),
    )
    .await
    .unwrap()
    .unwrap();
    let status = json_body(status).await;
    assert_eq!(status["state"], "running");
    assert_eq!(status["attempt_id"], attempt_id);
    let cancelled = tokio::time::timeout(
        Duration::from_millis(100),
        router.oneshot(request(
            Method::POST,
            "/v1/aec-calibration/cancel",
            Some(TOKEN),
            Body::from(format!(r#"{{"attempt_id":"{attempt_id}"}}"#)),
        )),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(cancelled.status(), StatusCode::OK);
    drop(release);
    calibration.shutdown().await.unwrap();
    assert_eq!(
        engine.delegate.calls.load(Ordering::SeqCst),
        0,
        "inspection completion after cancellation must not start calibration"
    );
    assert!(!matches!(
        coordinator.status(),
        translator_daemon::AecProofStatus::Validated { .. }
    ));
}

#[tokio::test]
async fn aec_calibration_is_explicit_nonblocking_and_duplicate_start_conflicts() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(FakeDuplexRunner::default());
    let translation = ControlApplication::spawn(
        store.clone(),
        runner.clone(),
        gate.clone(),
        Arc::new(NoopFacts),
        Arc::new(NoopFacts),
        None,
    );
    let engine = Arc::new(BlockingCalibrationEngine::default());
    let calibration = Arc::new(AecCalibrationController::new(
        Arc::new(AecCalibrationCoordinator::new()),
        gate,
        engine.clone(),
    ));
    let router = build_router_with_controllers(
        store,
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(translation.clone()),
            aec_calibration: Some(calibration.clone()),
            ..ApiControllers::default()
        },
    );

    let started = tokio::time::timeout(
        Duration::from_millis(100),
        router.clone().oneshot(request(
            Method::POST,
            "/v1/aec-calibration/start",
            Some(TOKEN),
            Body::from("{}"),
        )),
    )
    .await
    .expect("start must return after scheduling")
    .unwrap();
    assert_eq!(started.status(), StatusCode::ACCEPTED);
    let started = json_body(started).await;
    let attempt_id = started["attempt_id"].as_str().unwrap();

    let duplicate = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/aec-calibration/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_problem(
        duplicate,
        StatusCode::CONFLICT,
        "Conflict",
        "aec_calibration_busy",
    )
    .await;

    let status = tokio::time::timeout(
        Duration::from_millis(100),
        router.clone().oneshot(request(
            Method::GET,
            "/v1/aec-calibration",
            Some(TOKEN),
            Body::empty(),
        )),
    )
    .await
    .expect("status must remain responsive")
    .unwrap();
    assert_eq!(json_body(status).await["state"], "running");

    let stopped = tokio::time::timeout(
        Duration::from_millis(100),
        router.clone().oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        )),
    )
    .await
    .expect("translation Stop must remain responsive")
    .unwrap();
    assert_eq!(stopped.status(), StatusCode::OK);

    let cancelled = router
        .oneshot(request(
            Method::POST,
            "/v1/aec-calibration/cancel",
            Some(TOKEN),
            Body::from(format!(r#"{{"attempt_id":"{attempt_id}"}}"#)),
        ))
        .await
        .unwrap();
    assert_eq!(cancelled.status(), StatusCode::OK);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !matches!(
            calibration.status(),
            translator_daemon::AecCalibrationControlStatus::Cancelled { .. }
        ) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    translation.shutdown().await.unwrap();
    calibration.shutdown().await.unwrap();
}

#[tokio::test]
async fn translation_start_never_starts_aec_calibration_implicitly() {
    let store = RuntimeStore::default();
    let gate = AudioOperationGate::new();
    let runner = Arc::new(FakeDuplexRunner::default());
    let translation = ControlApplication::spawn(
        store.clone(),
        runner,
        gate.clone(),
        Arc::new(NoopFacts),
        Arc::new(NoopFacts),
        None,
    );
    let engine = Arc::new(BlockingCalibrationEngine::default());
    let calibration = Arc::new(AecCalibrationController::new(
        Arc::new(AecCalibrationCoordinator::new()),
        gate,
        engine.clone(),
    ));
    let router = build_router_with_controllers(
        store,
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(translation.clone()),
            aec_calibration: Some(calibration.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(started.status(), StatusCode::OK);
    assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        calibration.status(),
        translator_daemon::AecCalibrationControlStatus::Unavailable
    );

    let stopped = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(stopped.status(), StatusCode::OK);
    translation.shutdown().await.unwrap();
    calibration.shutdown().await.unwrap();
}

#[tokio::test]
async fn authenticated_unknown_path_and_wrong_method_are_problem_details() {
    let missing = app()
        .oneshot(request(
            Method::GET,
            "/v1/not-a-route",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(missing, StatusCode::NOT_FOUND, "Not Found", "not_found").await;

    let wrong_method = app()
        .oneshot(request(
            Method::GET,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(
        wrong_method,
        StatusCode::METHOD_NOT_ALLOWED,
        "Method Not Allowed",
        "method_not_allowed",
    )
    .await;
}

#[test]
fn listener_accepts_only_ipv4_and_ipv6_loopback_addresses() {
    assert!(
        validate_listen_address(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 47_681)).is_ok()
    );
    assert!(
        validate_listen_address(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 47_681))
            .is_err()
    );
    assert!(
        validate_listen_address(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 50)),
            47_681
        ))
        .is_err()
    );
    assert!(
        validate_listen_address(SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 47_681)).is_ok()
    );
    assert!(
        validate_listen_address(SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 47_681))
            .is_err()
    );
    assert!(
        validate_listen_address(SocketAddr::new("2001:db8::1".parse().unwrap(), 47_681)).is_err()
    );
}
