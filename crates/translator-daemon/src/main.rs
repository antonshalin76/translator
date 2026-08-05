use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use axum::http::StatusCode;
use clap::Parser;
use serde::Deserialize;
use translator_audio::{
    AecCapability, AecPhysicalPair, AudioGraph, AudioGraphState,
    AudioMixVolumes as PulseAudioMixVolumes, CommandResult, CommandRunner, DeviceOverride,
    DeviceWatcher, GraphHealth, MIC_OUT_SINK, OutputMode, PulseAecGraph, PulseAudioGraph,
    PulseAudioMix, PulseDeviceWatcher, PulseRoutingWatcher, REMOTE_IN_SINK, RoutingProfile,
    RoutingWatcher, SystemCommandRunner, default_journal_path, default_route_journal_path,
};
use translator_daemon::{
    ApiControllers, ApiLimits, AudioMixController, AudioMixState, AudioOperationGate,
    AudioOperationState, ControlToken, DebugCaptureLimits, DebugCaptureStore, DuplexRuntimeHandle,
    ManualRouteController, ProcessDuplexConfig, ProcessDuplexRunner, RoundTripController,
    RoundTripProcessRunner, RoundTripRuntimeHandle, RuntimeLatencyObserver, RuntimeLease,
    RuntimeSnapshot, RuntimeStore, TranslationController, build_router_with_controllers,
    validate_listen_address,
};

const SPEAKER_ORIGINAL_LOOPBACK: &str = "loopback-speaker-original";
const MICROPHONE_ORIGINAL_LOOPBACK: &str = "loopback-microphone-original";
const ORIGINAL_LOOPBACK_LATENCY_MS: u16 = 20;

struct LifecycleProtected<T> {
    stopping: AtomicBool,
    inner: Mutex<T>,
}

impl<T> LifecycleProtected<T> {
    fn new(inner: T) -> Self {
        Self {
            stopping: AtomicBool::new(false),
            inner: Mutex::new(inner),
        }
    }

    fn with_active<R>(&self, operation: impl FnOnce(&mut T) -> R) -> Option<R> {
        if self.is_stopping() {
            return None;
        }
        let mut inner = self.inner.lock().expect("lifecycle mutex poisoned");
        if self.is_stopping() {
            return None;
        }
        Some(operation(&mut inner))
    }

    fn stop_with<R>(&self, operation: impl FnOnce(&mut T) -> R) -> R {
        self.stopping.store(true, Ordering::Release);
        let mut inner = self.inner.lock().expect("lifecycle mutex poisoned");
        operation(&mut inner)
    }

    fn with_exclusive<R>(&self, operation: impl FnOnce(&mut T) -> R) -> R {
        let mut inner = self.inner.lock().expect("lifecycle mutex poisoned");
        operation(&mut inner)
    }

    fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }
}

struct PulseResources {
    routing: PulseRoutingWatcher<SystemCommandRunner>,
    devices: PulseDeviceWatcher<SystemCommandRunner>,
    original_loopbacks: PulseOriginalLoopbacks<SystemCommandRunner>,
    graph: Option<PulseAudioGraph<SystemCommandRunner>>,
    aec_graph: Option<PulseAecGraph<SystemCommandRunner>>,
}

impl PulseResources {
    fn initialize(&mut self, store: &RuntimeStore) {
        if let Some(graph) = self.graph.as_mut() {
            match graph.ensure_endpoints() {
                Ok(state) => store.set_audio_graph(state),
                Err(error) => {
                    tracing::error!(
                        event = "audio_graph_initialization_failed",
                        code = ?error.code()
                    );
                    store.set_audio_graph(AudioGraphState::failed(&error));
                }
            }
        } else {
            store.clear_audio_graph("journal_path_unavailable");
        }
        self.refresh(store);
    }

    fn refresh(&mut self, store: &RuntimeStore) {
        self.refresh_graph(store);
        match self.routing.reconcile(None) {
            Ok(state) => store.set_routes(state),
            Err(error) => {
                tracing::warn!(event = "route_reconciliation_failed", code = ?error.code());
                store.clear_routes("route_reconciliation_failed");
            }
        }
        self.refresh_devices(store);
        self.ensure_original_loopbacks(store);
    }

    fn refresh_graph_and_devices(&mut self, store: &RuntimeStore) {
        self.refresh_graph(store);
        self.refresh_devices(store);
        self.ensure_original_loopbacks(store);
    }

    fn refresh_graph(&mut self, store: &RuntimeStore) {
        if let Some(graph) = self.graph.as_mut() {
            maintain_audio_graph(graph, store);
        }
    }

    fn refresh_devices(&mut self, store: &RuntimeStore) {
        match self.devices.reconcile(DeviceOverride::default()) {
            Ok(state) => store.set_devices(state),
            Err(error) => {
                tracing::warn!(event = "device_reconciliation_failed", code = ?error.code());
                store.clear_devices("device_reconciliation_failed");
            }
        }
    }

    fn ensure_original_loopbacks(&self, store: &RuntimeStore) {
        if let Err(error) = self.original_loopbacks.ensure(&store.snapshot()) {
            tracing::warn!(
                event = "original_loopback_reconciliation_failed",
                code = ?error.code()
            );
        }
    }

    fn cleanup_graph(&mut self) {
        if let Err(error) = self.original_loopbacks.cleanup_all() {
            tracing::warn!(event = "original_loopback_cleanup_failed", code = ?error.code());
        }
        if let Some(graph) = self.graph.as_mut()
            && let Err(error) = graph.cleanup_owned()
        {
            tracing::error!(event = "audio_graph_cleanup_failed", code = ?error.code());
        }
        if let Some(aec_graph) = self.aec_graph.as_mut()
            && let Err(error) = aec_graph.cleanup_owned()
        {
            tracing::error!(event = "aec_graph_cleanup_failed", code = ?error.code());
        }
    }
}

fn maintain_audio_graph(graph: &mut impl AudioGraph, store: &RuntimeStore) {
    match graph.inspect() {
        Ok(state) if state.health == GraphHealth::Ready => {
            store.set_audio_graph(state);
            return;
        }
        Ok(state) => {
            tracing::warn!(
                event = "audio_graph_self_heal_needed",
                health = ?state.health
            );
        }
        Err(error) => {
            tracing::warn!(
                event = "audio_graph_self_heal_inspection_failed",
                code = ?error.code()
            );
        }
    }

    match graph.ensure_endpoints() {
        Ok(state) => store.set_audio_graph(state),
        Err(error) => {
            tracing::warn!(
                event = "audio_graph_self_heal_failed",
                code = ?error.code()
            );
            store.set_audio_graph(AudioGraphState::failed(&error));
        }
    }
}

struct PulseManualRoutes {
    resources: LifecycleProtected<PulseResources>,
    operation_gate: AudioOperationGate,
}

struct PulseAudioMixController {
    mix: PulseAudioMix<SystemCommandRunner>,
}

impl PulseAudioMixController {
    const fn new() -> Self {
        Self {
            mix: PulseAudioMix::new(SystemCommandRunner),
        }
    }
}

impl AudioMixController for PulseAudioMixController {
    fn apply(&self, volumes: AudioMixState) -> Result<(), translator_daemon::ControlFailure> {
        match self.mix.apply(PulseAudioMixVolumes {
            microphone_original_percent: volumes.microphone_original_percent,
            microphone_translation_percent: volumes.microphone_translation_percent,
            speaker_original_percent: volumes.speaker_original_percent,
            speaker_translation_percent: volumes.speaker_translation_percent,
        }) {
            Ok(report) => {
                tracing::debug!(
                    event = "audio_mix_applied",
                    updated_target_count = report.updated_targets.len()
                );
                Ok(())
            }
            Err(error) => {
                tracing::warn!(event = "audio_mix_apply_failed", code = ?error.code());
                Err(translator_daemon::ControlFailure {
                    status: StatusCode::CONFLICT,
                    code: "audio_mix_apply_failed",
                })
            }
        }
    }
}

impl PulseManualRoutes {
    fn initialize(&self, store: &RuntimeStore) {
        let initialized = self
            .resources
            .with_active(|resources| resources.initialize(store));
        debug_assert!(initialized.is_some());
    }

    fn refresh(&self, store: &RuntimeStore) {
        let routing_allowed = matches!(
            self.operation_gate.state(),
            AudioOperationState::Idle | AudioOperationState::Production
        );
        self.resources.with_active(|resources| {
            if routing_allowed {
                resources.refresh(store);
            } else {
                resources.refresh_graph_and_devices(store);
            }
        });
    }

    fn cleanup_graph(&self) {
        self.resources.with_exclusive(PulseResources::cleanup_graph);
    }
}

impl ManualRouteController for PulseManualRoutes {
    fn refresh_audio_state(&self, store: &RuntimeStore) {
        self.refresh(store);
    }

    fn reconcile(
        &self,
        stream_id: u32,
    ) -> Result<translator_audio::RoutingState, translator_audio::RoutingSafeError> {
        let _lease = match manual_route_admission(self.operation_gate.state())? {
            ManualRouteAdmission::AcquireExclusive => Some(
                self.operation_gate
                    .acquire_manual()
                    .map_err(|_| invalid_manual_route("Audio operation is busy"))?,
            ),
            ManualRouteAdmission::ShareProduction => None,
        };
        self.resources
            .with_active(|resources| {
                resources
                    .routing
                    .reconcile(Some(stream_id))
                    .map_err(|error| error.safe_status().clone())
            })
            .unwrap_or_else(|| {
                Err(translator_audio::RoutingSafeError {
                    code: translator_audio::RoutingErrorCode::DiscoveryFailed,
                    safe_message: "Routing controller is stopping".to_owned(),
                    retryable: true,
                })
            })
    }

    fn restore(&self) -> Result<(), translator_audio::RoutingSafeError> {
        self.resources.stop_with(|resources| {
            resources
                .routing
                .restore_active()
                .map(|_| ())
                .map_err(|error| error.safe_status().clone())
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManualRouteAdmission {
    AcquireExclusive,
    ShareProduction,
}

fn manual_route_admission(
    state: AudioOperationState,
) -> Result<ManualRouteAdmission, translator_audio::RoutingSafeError> {
    match state {
        AudioOperationState::Idle => Ok(ManualRouteAdmission::AcquireExclusive),
        AudioOperationState::Production => Ok(ManualRouteAdmission::ShareProduction),
        AudioOperationState::HumanRoundTrip { .. } => {
            Err(invalid_manual_route("Audio operation is busy"))
        }
        AudioOperationState::Stopping => {
            Err(invalid_manual_route("Routing controller is stopping"))
        }
    }
}

fn invalid_manual_route(message: &str) -> translator_audio::RoutingSafeError {
    translator_audio::RoutingSafeError {
        code: translator_audio::RoutingErrorCode::InvalidManualOverride,
        safe_message: message.to_owned(),
        retryable: true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OriginalLoopbackErrorCode {
    Discovery,
    Load,
    Cleanup,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OriginalLoopbackError {
    code: OriginalLoopbackErrorCode,
}

impl OriginalLoopbackError {
    const fn new(code: OriginalLoopbackErrorCode) -> Self {
        Self { code }
    }

    const fn code(&self) -> OriginalLoopbackErrorCode {
        self.code
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OriginalLoopbackRequest {
    media_name: &'static str,
    source: String,
    source_target_object: String,
    sink: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiscoveredOriginalLoopback {
    media_name: &'static str,
    source_target_object: Option<String>,
    sink_target_object: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawPulseStream {
    #[serde(default)]
    properties: HashMap<String, String>,
}

struct PulseOriginalLoopbacks<R = SystemCommandRunner> {
    runner: R,
}

impl<R> PulseOriginalLoopbacks<R>
where
    R: CommandRunner,
{
    const fn new(runner: R) -> Self {
        Self { runner }
    }

    fn ensure(&self, snapshot: &RuntimeSnapshot) -> Result<(), OriginalLoopbackError> {
        let requests = original_loopback_requests(snapshot);
        let sink_inputs: Vec<RawPulseStream> =
            self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let source_outputs: Vec<RawPulseStream> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let discovered = discover_original_loopbacks(&sink_inputs, &source_outputs);
        let mut keep_module_ids = HashSet::new();
        let mut missing_requests = Vec::new();

        for request in &requests {
            let mut matching_module_ids = matching_original_loopbacks(&discovered, request);
            matching_module_ids.sort();
            if let Some(module_id) = matching_module_ids.first() {
                keep_module_ids.insert(module_id.clone());
            } else {
                missing_requests.push(request.clone());
            }
        }

        let mut stale_module_ids: Vec<_> = discovered
            .keys()
            .filter(|module_id| !keep_module_ids.contains(*module_id))
            .cloned()
            .collect();
        stale_module_ids.sort();
        for module_id in stale_module_ids {
            self.unload_module(&module_id)?;
        }

        for request in missing_requests {
            self.load_module(&request)?;
        }

        Ok(())
    }

    fn cleanup_all(&self) -> Result<Vec<String>, OriginalLoopbackError> {
        let sink_inputs: Vec<RawPulseStream> =
            self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let source_outputs: Vec<RawPulseStream> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let discovered = discover_original_loopbacks(&sink_inputs, &source_outputs);
        let mut module_ids: Vec<_> = discovered.keys().cloned().collect();
        module_ids.sort();
        for module_id in &module_ids {
            self.unload_module(module_id)?;
        }
        Ok(module_ids)
    }

    fn load_module(&self, request: &OriginalLoopbackRequest) -> Result<(), OriginalLoopbackError> {
        let args = original_loopback_load_args(request);
        self.run_pactl_owned(&args, OriginalLoopbackErrorCode::Load)?;
        Ok(())
    }

    fn unload_module(&self, module_id: &str) -> Result<(), OriginalLoopbackError> {
        self.run_pactl_owned(
            &["unload-module".to_owned(), module_id.to_owned()],
            OriginalLoopbackErrorCode::Cleanup,
        )?;
        Ok(())
    }

    fn run_json<T>(&self, args: &[&str]) -> Result<T, OriginalLoopbackError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let result = self.run_pactl(args, OriginalLoopbackErrorCode::Discovery)?;
        serde_json::from_slice(result.stdout())
            .map_err(|_| OriginalLoopbackError::new(OriginalLoopbackErrorCode::Discovery))
    }

    fn run_pactl(
        &self,
        args: &[&str],
        failure_code: OriginalLoopbackErrorCode,
    ) -> Result<CommandResult, OriginalLoopbackError> {
        let owned: Vec<String> = args.iter().map(|value| (*value).to_owned()).collect();
        self.run_pactl_owned(&owned, failure_code)
    }

    fn run_pactl_owned(
        &self,
        args: &[String],
        failure_code: OriginalLoopbackErrorCode,
    ) -> Result<CommandResult, OriginalLoopbackError> {
        let result = self
            .runner
            .run("pactl", args)
            .map_err(|_| OriginalLoopbackError::new(failure_code))?;
        if result.is_success() {
            Ok(result)
        } else {
            Err(OriginalLoopbackError::new(failure_code))
        }
    }
}

fn original_loopback_requests(snapshot: &RuntimeSnapshot) -> Vec<OriginalLoopbackRequest> {
    let Some(devices) = snapshot.devices.as_ref() else {
        return Vec::new();
    };

    let mut requests = Vec::new();
    if original_bypass_required(snapshot, snapshot.audio_mix.speaker_original_percent)
        && let Some(sink) = devices.sink.selected.as_ref()
    {
        requests.push(OriginalLoopbackRequest {
            media_name: SPEAKER_ORIGINAL_LOOPBACK,
            source: format!("{REMOTE_IN_SINK}.monitor"),
            source_target_object: REMOTE_IN_SINK.to_owned(),
            sink: sink.name.clone(),
        });
    }

    if original_bypass_required(snapshot, snapshot.audio_mix.microphone_original_percent)
        && let Some(source) = devices.source.selected.as_ref()
    {
        requests.push(OriginalLoopbackRequest {
            media_name: MICROPHONE_ORIGINAL_LOOPBACK,
            source: source.name.clone(),
            source_target_object: source.name.clone(),
            sink: MIC_OUT_SINK.to_owned(),
        });
    }

    requests
}

const fn original_bypass_required(snapshot: &RuntimeSnapshot, configured_percent: u8) -> bool {
    !snapshot.translation_running || configured_percent > 0
}

fn effective_audio_mix_for_service(snapshot: &RuntimeSnapshot) -> AudioMixState {
    if snapshot.translation_running {
        return snapshot.audio_mix;
    }

    AudioMixState {
        microphone_original_percent: 100,
        microphone_translation_percent: 0,
        speaker_original_percent: 100,
        speaker_translation_percent: 0,
    }
}

fn original_loopback_load_args(request: &OriginalLoopbackRequest) -> Vec<String> {
    vec![
        "load-module".to_owned(),
        "module-loopback".to_owned(),
        format!("source={}", request.source),
        format!("sink={}", request.sink),
        format!("latency_msec={ORIGINAL_LOOPBACK_LATENCY_MS}"),
        "source_dont_move=true".to_owned(),
        "sink_dont_move=true".to_owned(),
        format!(
            "source_output_properties=media.name={} translator.owner=true",
            request.media_name
        ),
        format!(
            "sink_input_properties=media.name={} translator.owner=true",
            request.media_name
        ),
    ]
}

fn discover_original_loopbacks(
    sink_inputs: &[RawPulseStream],
    source_outputs: &[RawPulseStream],
) -> HashMap<String, DiscoveredOriginalLoopback> {
    let mut modules = HashMap::new();
    for input in sink_inputs {
        let Some(media_name) = original_media_name(&input.properties) else {
            continue;
        };
        let Some(module_id) = property(&input.properties, "pulse.module.id") else {
            continue;
        };
        let module = discovered_loopback_entry(&mut modules, module_id, media_name);
        module.sink_target_object = property(&input.properties, "target.object").map(str::to_owned);
    }

    for output in source_outputs {
        let Some(media_name) = original_media_name(&output.properties) else {
            continue;
        };
        let Some(module_id) = property(&output.properties, "pulse.module.id") else {
            continue;
        };
        let module = discovered_loopback_entry(&mut modules, module_id, media_name);
        module.source_target_object =
            property(&output.properties, "target.object").map(str::to_owned);
    }
    modules
}

fn discovered_loopback_entry<'a>(
    modules: &'a mut HashMap<String, DiscoveredOriginalLoopback>,
    module_id: &str,
    media_name: &'static str,
) -> &'a mut DiscoveredOriginalLoopback {
    modules
        .entry(module_id.to_owned())
        .or_insert_with(|| DiscoveredOriginalLoopback {
            media_name,
            source_target_object: None,
            sink_target_object: None,
        })
}

fn matching_original_loopbacks(
    discovered: &HashMap<String, DiscoveredOriginalLoopback>,
    request: &OriginalLoopbackRequest,
) -> Vec<String> {
    discovered
        .iter()
        .filter_map(|(module_id, loopback)| {
            (loopback.media_name == request.media_name
                && loopback.source_target_object.as_deref()
                    == Some(request.source_target_object.as_str())
                && loopback.sink_target_object.as_deref() == Some(request.sink.as_str()))
            .then_some(module_id.clone())
        })
        .collect()
}

fn original_media_name(properties: &HashMap<String, String>) -> Option<&'static str> {
    match property(properties, "media.name") {
        Some(SPEAKER_ORIGINAL_LOOPBACK) => Some(SPEAKER_ORIGINAL_LOOPBACK),
        Some(MICROPHONE_ORIGINAL_LOOPBACK) => Some(MICROPHONE_ORIGINAL_LOOPBACK),
        _ => None,
    }
}

fn property<'a>(properties: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    properties.get(key).map(String::as_str)
}

#[derive(Debug, Parser)]
#[command(version, about = "Local full-duplex translation daemon")]
struct Arguments {
    #[arg(
        long,
        conflicts_with_all = ["audio_graph_cleanup", "watcher_state_smoke"]
    )]
    audio_graph_smoke: bool,

    #[arg(
        long,
        conflicts_with_all = ["audio_graph_smoke", "watcher_state_smoke"]
    )]
    audio_graph_cleanup: bool,

    #[arg(
        long,
        conflicts_with_all = ["audio_graph_smoke", "audio_graph_cleanup"]
    )]
    watcher_state_smoke: bool,

    #[arg(long, default_value = "127.0.0.1:47681")]
    listen: SocketAddr,
}

#[tokio::main]
async fn main() -> ExitCode {
    let arguments = Arguments::parse();
    if arguments.audio_graph_smoke {
        return run_audio_graph_smoke();
    }
    if arguments.audio_graph_cleanup {
        return run_audio_graph_cleanup();
    }
    if arguments.watcher_state_smoke {
        return run_watcher_state_smoke();
    }

    tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .compact()
        .init();

    if validate_listen_address(arguments.listen).is_err() {
        tracing::error!(
            event = "daemon_start_failed",
            code = "non_loopback_listener"
        );
        return ExitCode::FAILURE;
    }
    let Some(runtime_parent) = std::env::var_os("XDG_RUNTIME_DIR") else {
        tracing::error!(
            event = "daemon_start_failed",
            code = "runtime_directory_unavailable"
        );
        return ExitCode::FAILURE;
    };
    let lease = match RuntimeLease::acquire(std::path::Path::new(&runtime_parent)) {
        Ok(lease) => lease,
        Err(error) => {
            tracing::error!(event = "daemon_start_failed", code = error.code().as_str());
            return ExitCode::FAILURE;
        }
    };
    let token_value = match std::fs::read_to_string(lease.token_path()) {
        Ok(value) => value,
        Err(_) => {
            tracing::error!(
                event = "daemon_start_failed",
                code = "control_token_unavailable"
            );
            return ExitCode::FAILURE;
        }
    };
    let token = match ControlToken::parse(&token_value) {
        Ok(token) => token,
        Err(error) => {
            tracing::error!(event = "daemon_start_failed", code = error.code().as_str());
            return ExitCode::FAILURE;
        }
    };
    let listener = match tokio::net::TcpListener::bind(arguments.listen).await {
        Ok(listener) => listener,
        Err(_) => {
            tracing::error!(event = "daemon_start_failed", code = "listen_failed");
            return ExitCode::FAILURE;
        }
    };
    let store = RuntimeStore::default();
    if let Some(state_parent) = user_state_parent() {
        match DebugCaptureStore::open(&state_parent, DebugCaptureLimits::default()) {
            Ok(capture_store) => store.configure_debug_capture(capture_store),
            Err(error) => tracing::error!(
                event = "debug_capture_initialization_failed",
                code = error.code().as_str()
            ),
        }
    } else {
        tracing::error!(
            event = "debug_capture_initialization_failed",
            code = "state_directory_unavailable"
        );
    }
    let audio_graph = default_journal_path()
        .ok()
        .map(|journal| PulseAudioGraph::new(SystemCommandRunner, journal));
    let (aec_capability, aec_graph) = initialize_headphone_aec();
    let device_watcher = build_device_watcher(aec_capability);
    let operation_gate = AudioOperationGate::new();
    let manual_routes = Arc::new(PulseManualRoutes {
        resources: LifecycleProtected::new(PulseResources {
            routing: build_routing_watcher(),
            devices: device_watcher,
            original_loopbacks: PulseOriginalLoopbacks::new(SystemCommandRunner),
            graph: audio_graph,
            aec_graph,
        }),
        operation_gate: operation_gate.clone(),
    });
    manual_routes.initialize(&store);
    let audio_mix: Arc<dyn AudioMixController> = Arc::new(PulseAudioMixController::new());
    let duplex_config = build_duplex_config(lease.token_path());
    let translation = duplex_config.clone().map(|config| {
        Arc::new(DuplexRuntimeHandle::with_runner_and_gate(
            Arc::new(ProcessDuplexRunner::with_observer(
                config,
                Arc::new(RuntimeLatencyObserver::new(store.clone())),
            )),
            operation_gate.clone(),
        ))
    });
    let round_trip = duplex_config.map(|config| {
        Arc::new(RoundTripRuntimeHandle::new(
            store.clone(),
            Arc::new(RoundTripProcessRunner::new(config)),
            operation_gate.clone(),
        ))
    });
    let router = build_router_with_controllers(
        store.clone(),
        token,
        ApiLimits::default(),
        ApiControllers {
            manual_routes: Some(manual_routes.clone()),
            audio_mix: Some(audio_mix.clone()),
            translation: translation
                .as_ref()
                .map(|controller| controller.clone() as Arc<dyn TranslationController>),
            round_trip: round_trip
                .as_ref()
                .map(|controller| controller.clone() as Arc<dyn RoundTripController>),
        },
    );
    tracing::info!(
        event = "daemon_started",
        listen_address = %arguments.listen,
        graph_available = matches!(
            store.snapshot().audio_graph.map(|state| state.health),
            Some(GraphHealth::Ready)
        ),
        device_state_available = store.snapshot().devices.is_some(),
        translation_controller_available = translation.is_some(),
        round_trip_controller_available = round_trip.is_some(),
        provider_schema_bytes = translator_ipc::PROVIDER_PROTO.len(),
        "translator daemon control plane is ready"
    );
    let watcher_task = tokio::spawn(watcher_loop(
        manual_routes.clone(),
        audio_mix.clone(),
        store.clone(),
    ));
    let debug_capture_watchdog = tokio::spawn(store.clone().run_debug_capture_watchdog());
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel::<()>();
    let mut server_task = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = shutdown_receiver.await;
            })
            .await
    });

    shutdown_signal().await;
    operation_gate.begin_stopping();
    watcher_task.abort();
    let _ = watcher_task.await;
    debug_capture_watchdog.abort();
    let _ = debug_capture_watchdog.await;
    if matches!(
        store.snapshot().self_test.status.checkpoint,
        Some(
            translator_daemon::RoundTripCheckpoint::WaitingForSpeech
                | translator_daemon::RoundTripCheckpoint::OutgoingVad
                | translator_daemon::RoundTripCheckpoint::OutgoingAsrFinal
                | translator_daemon::RoundTripCheckpoint::OutgoingTranslationFinal
                | translator_daemon::RoundTripCheckpoint::EnglishFirstAudio
                | translator_daemon::RoundTripCheckpoint::VirtualPeerReinjecting
                | translator_daemon::RoundTripCheckpoint::IncomingAsrFinal
                | translator_daemon::RoundTripCheckpoint::IncomingTranslationFinal
                | translator_daemon::RoundTripCheckpoint::RussianFirstAudio
        )
    ) && let Some(controller) = round_trip.as_ref()
        && let Err(error) = controller.stop()
    {
        tracing::error!(event = "round_trip_shutdown_failed", code = error.code);
    }
    if store.snapshot().translation_running
        && let Some(controller) = translation.as_ref()
        && let Err(error) = controller.stop()
    {
        tracing::error!(event = "translation_shutdown_failed", code = error.code);
    }
    store.set_translation_running(false);
    let _ = store.set_debug_capture_enabled(false);
    if let Err(error) = manual_routes.restore() {
        tracing::error!(event = "route_restore_failed", code = ?error.code);
    }
    store.shutdown_events();
    let _ = shutdown_sender.send(());
    let result =
        match tokio::time::timeout(std::time::Duration::from_secs(5), &mut server_task).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(std::io::Error::other("server task failed")),
            Err(_) => {
                server_task.abort();
                let _ = server_task.await;
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "server drain timed out",
                ))
            }
        };
    manual_routes.cleanup_graph();
    drop(lease);
    if result.is_err() {
        tracing::error!(event = "daemon_stopped", code = "server_error");
        ExitCode::FAILURE
    } else {
        tracing::info!(event = "daemon_stopped", code = "graceful_shutdown");
        ExitCode::SUCCESS
    }
}

fn build_duplex_config(token_path: &std::path::Path) -> Option<ProcessDuplexConfig> {
    let sidecar_root = std::env::var_os("TRANSLATOR_SIDECAR_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sidecar"));
    let python = std::env::var_os("TRANSLATOR_PYTHON")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| sidecar_root.join(".venv/bin/python"));
    let socket_path = token_path
        .parent()?
        .join(translator_ipc::SIDECAR_SOCKET_NAME);
    match ProcessDuplexConfig::from_runtime(python, sidecar_root, socket_path) {
        Ok(config) => Some(config),
        Err(error) => {
            tracing::error!(
                event = "duplex_controller_initialization_failed",
                code = ?error
            );
            None
        }
    }
}

fn build_device_watcher(aec_capability: AecCapability) -> PulseDeviceWatcher<SystemCommandRunner> {
    let mut device_watcher = PulseDeviceWatcher::new(SystemCommandRunner, aec_capability);
    if let Some(sink_name) = std::env::var_os("TRANSLATOR_HEADPHONE_SINK") {
        device_watcher = device_watcher.with_explicit_headphone_sink(sink_name.to_string_lossy());
    }
    device_watcher
}

fn build_routing_watcher() -> PulseRoutingWatcher<SystemCommandRunner> {
    match default_route_journal_path() {
        Ok(path) => PulseRoutingWatcher::new_with_route_journal(
            SystemCommandRunner,
            RoutingProfile::Production,
            path,
        ),
        Err(error) => {
            tracing::warn!(event = "route_journal_unavailable", code = ?error.code());
            PulseRoutingWatcher::new(SystemCommandRunner, RoutingProfile::Production)
        }
    }
}

fn initialize_headphone_aec() -> (AecCapability, Option<PulseAecGraph<SystemCommandRunner>>) {
    if std::env::var_os("TRANSLATOR_ENABLE_HEADPHONE_AEC").is_none() {
        tracing::info!(event = "headphone_aec_disabled", reason = "not_enabled");
        return (AecCapability::Unavailable, None);
    }
    if std::env::var_os("TRANSLATOR_DISABLE_HEADPHONE_AEC").is_some() {
        tracing::info!(event = "headphone_aec_disabled", reason = "disabled");
        return (AecCapability::Unavailable, None);
    }
    let mut probe = build_device_watcher(AecCapability::Unavailable);
    let state = match probe.reconcile(DeviceOverride::default()) {
        Ok(state) => state,
        Err(error) => {
            tracing::warn!(
                event = "headphone_aec_probe_failed",
                code = ?error.code()
            );
            return (AecCapability::Unavailable, None);
        }
    };
    if state.acoustic.mode != OutputMode::Headphones {
        return (AecCapability::Unavailable, None);
    }
    let Some(source) = state
        .source
        .selected
        .as_ref()
        .map(|source| source.name.clone())
    else {
        return (AecCapability::Unavailable, None);
    };
    let Some(sink) = state.sink.selected.as_ref().map(|sink| sink.name.clone()) else {
        return (AecCapability::Unavailable, None);
    };
    let generation = format!("translator-headphone-aec-{}", uuid::Uuid::new_v4());
    let mut graph = match PulseAecGraph::new(
        SystemCommandRunner,
        AecPhysicalPair::new(source.clone(), sink.clone()),
        generation,
    ) {
        Ok(graph) => graph,
        Err(error) => {
            tracing::warn!(event = "headphone_aec_configuration_failed", code = ?error.code());
            return (AecCapability::Unavailable, None);
        }
    };
    match graph.load_owned() {
        Ok(_) => {
            tracing::info!(
                event = "headphone_aec_ready",
                source_name = %source,
                sink_name = %sink
            );
            (
                AecCapability::ValidatedFor {
                    source_name: source,
                    sink_name: sink,
                },
                Some(graph),
            )
        }
        Err(error) => {
            tracing::warn!(event = "headphone_aec_load_failed", code = ?error.code());
            (AecCapability::Unavailable, None)
        }
    }
}

async fn shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler installation failed");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
}

async fn watcher_loop(
    controller: Arc<PulseManualRoutes>,
    audio_mix: Arc<dyn AudioMixController>,
    store: RuntimeStore,
) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_latency_epoch = 0;
    loop {
        interval.tick().await;
        let controller = controller.clone();
        let audio_mix = audio_mix.clone();
        let refresh_store = store.clone();
        let snapshot = store.snapshot();
        let volumes = effective_audio_mix_for_service(&snapshot);
        if tokio::task::spawn_blocking(move || {
            controller.refresh(&refresh_store);
            if let Err(error) = audio_mix.apply(volumes) {
                tracing::warn!(event = "audio_mix_watchdog_apply_failed", code = error.code);
            }
        })
        .await
        .is_err()
        {
            tracing::error!(event = "watcher_task_failed", code = "join_failed");
        }
        let now_ms = store.monotonic_ms();
        let epoch_end = (now_ms / 60_000) * 60_000;
        if epoch_end > last_latency_epoch {
            store.evaluate_latency_epoch(translator_core::AudioDirection::Microphone, epoch_end);
            store.evaluate_latency_epoch(translator_core::AudioDirection::Speaker, epoch_end);
            last_latency_epoch = epoch_end;
        }
    }
}

fn user_state_parent() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .map(|home| home.join(".local/state"))
        })
}

fn run_audio_graph_smoke() -> ExitCode {
    let journal_path = match default_journal_path() {
        Ok(path) => path,
        Err(error) => return print_graph_error(&error),
    };
    let mut graph = PulseAudioGraph::new(SystemCommandRunner, journal_path);
    match graph.ensure_endpoints() {
        Ok(state) => print_json(&state),
        Err(error) => print_graph_error(&error),
    }
}

fn run_audio_graph_cleanup() -> ExitCode {
    let journal_path = match default_journal_path() {
        Ok(path) => path,
        Err(error) => return print_graph_error(&error),
    };
    let mut graph = PulseAudioGraph::new(SystemCommandRunner, journal_path);
    match graph.cleanup_owned() {
        Ok(module_ids) => print_json(&serde_json::json!({
            "unloaded_module_ids": module_ids
        })),
        Err(error) => print_graph_error(&error),
    }
}

fn run_watcher_state_smoke() -> ExitCode {
    let routing =
        match PulseRoutingWatcher::new(SystemCommandRunner, RoutingProfile::Production).inspect() {
            Ok(state) => state,
            Err(error) => return print_json_failure(error.safe_status()),
        };
    let mut devices = PulseDeviceWatcher::new(SystemCommandRunner, AecCapability::Unavailable);
    let devices = match devices.reconcile(DeviceOverride::default()) {
        Ok(state) => state,
        Err(error) => return print_json_failure(error.safe_status()),
    };
    print_json(&serde_json::json!({
        "routing": routing,
        "devices": devices,
    }))
}

fn print_graph_error(error: &translator_audio::AudioGraphError) -> ExitCode {
    let state = AudioGraphState::failed(error);
    let _ = serde_json::to_writer(std::io::stdout().lock(), &state);
    println!();
    ExitCode::FAILURE
}

fn print_json_failure<T: serde::Serialize>(value: &T) -> ExitCode {
    let _ = print_json(value);
    ExitCode::FAILURE
}

fn print_json<T: serde::Serialize>(value: &T) -> ExitCode {
    if serde_json::to_writer(std::io::stdout().lock(), value).is_err() {
        return ExitCode::FAILURE;
    }
    println!();
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::sync::{Arc, mpsc};
    use std::time::{Duration, Instant};

    use super::{
        DiscoveredOriginalLoopback, LifecycleProtected, MICROPHONE_ORIGINAL_LOOPBACK,
        ManualRouteAdmission, OriginalLoopbackRequest, RawPulseStream, SPEAKER_ORIGINAL_LOOPBACK,
        discover_original_loopbacks, effective_audio_mix_for_service, maintain_audio_graph,
        manual_route_admission, matching_original_loopbacks, original_loopback_load_args,
        original_loopback_requests,
    };
    use translator_audio::{
        AcousticSafety, AecCapability, AudioEndpointState, AudioGraph, AudioGraphError,
        AudioGraphState, DeviceHealth, DeviceSelectionState, DeviceState, EndpointRole,
        GraphHealth, MIC_OUT_SINK, OutputMode, PhysicalDevice, REMOTE_IN_SINK,
    };
    use translator_daemon::{AudioMixState, AudioOperationState, RuntimeSnapshot};
    use uuid::Uuid;

    #[test]
    fn shutdown_waits_for_active_refresh_and_rejects_late_refresh() {
        let resources = Arc::new(LifecycleProtected::new(Vec::new()));
        let (refresh_started_tx, refresh_started_rx) = mpsc::channel();
        let (release_refresh_tx, release_refresh_rx) = mpsc::channel();

        let refresh_resources = Arc::clone(&resources);
        let refresh = std::thread::spawn(move || {
            refresh_resources.with_active(|operations| {
                refresh_started_tx.send(()).unwrap();
                release_refresh_rx.recv().unwrap();
                operations.push("refresh");
            })
        });
        refresh_started_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        let shutdown_resources = Arc::clone(&resources);
        let shutdown = std::thread::spawn(move || {
            shutdown_resources.stop_with(|operations| {
                operations.push("restore");
                operations.clone()
            })
        });
        let deadline = Instant::now() + Duration::from_secs(1);
        while !resources.is_stopping() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(resources.is_stopping(), "shutdown did not mark stopping");

        assert!(
            resources
                .with_active(|operations| operations.push("late_refresh"))
                .is_none()
        );
        release_refresh_tx.send(()).unwrap();

        assert!(refresh.join().unwrap().is_some());
        assert_eq!(shutdown.join().unwrap(), ["refresh", "restore"]);
    }

    #[test]
    fn manual_route_admission_shares_production_but_rejects_self_test_and_stopping() {
        assert_eq!(
            manual_route_admission(AudioOperationState::Idle)
                .map_err(|error| error.code)
                .unwrap(),
            ManualRouteAdmission::AcquireExclusive
        );
        assert_eq!(
            manual_route_admission(AudioOperationState::Production)
                .map_err(|error| error.code)
                .unwrap(),
            ManualRouteAdmission::ShareProduction
        );
        assert!(
            manual_route_admission(AudioOperationState::HumanRoundTrip {
                session_id: Uuid::new_v4()
            })
            .is_err()
        );
        assert!(manual_route_admission(AudioOperationState::Stopping).is_err());
    }

    #[test]
    fn original_loopback_requests_follow_original_mix_and_devices() {
        let snapshot = RuntimeSnapshot {
            translation_running: true,
            audio_mix: AudioMixState {
                microphone_original_percent: 74,
                microphone_translation_percent: 100,
                speaker_original_percent: 76,
                speaker_translation_percent: 100,
            },
            devices: Some(selected_devices()),
            ..RuntimeSnapshot::default()
        };

        assert_eq!(
            original_loopback_requests(&snapshot),
            [
                OriginalLoopbackRequest {
                    media_name: SPEAKER_ORIGINAL_LOOPBACK,
                    source: format!("{REMOTE_IN_SINK}.monitor"),
                    source_target_object: REMOTE_IN_SINK.to_owned(),
                    sink: "alsa_output.headphones".to_owned(),
                },
                OriginalLoopbackRequest {
                    media_name: MICROPHONE_ORIGINAL_LOOPBACK,
                    source: "alsa_input.microphone".to_owned(),
                    source_target_object: "alsa_input.microphone".to_owned(),
                    sink: MIC_OUT_SINK.to_owned(),
                },
            ]
        );

        let stopped = RuntimeSnapshot {
            translation_running: false,
            ..snapshot.clone()
        };
        assert_eq!(
            original_loopback_requests(&stopped),
            original_loopback_requests(&snapshot)
        );

        let muted_originals = RuntimeSnapshot {
            audio_mix: AudioMixState {
                microphone_original_percent: 0,
                speaker_original_percent: 0,
                ..snapshot.audio_mix
            },
            ..snapshot
        };
        assert!(original_loopback_requests(&muted_originals).is_empty());

        let stopped_muted = RuntimeSnapshot {
            translation_running: false,
            ..muted_originals
        };
        assert_eq!(
            original_loopback_requests(&stopped_muted),
            [
                OriginalLoopbackRequest {
                    media_name: SPEAKER_ORIGINAL_LOOPBACK,
                    source: format!("{REMOTE_IN_SINK}.monitor"),
                    source_target_object: REMOTE_IN_SINK.to_owned(),
                    sink: "alsa_output.headphones".to_owned(),
                },
                OriginalLoopbackRequest {
                    media_name: MICROPHONE_ORIGINAL_LOOPBACK,
                    source: "alsa_input.microphone".to_owned(),
                    source_target_object: "alsa_input.microphone".to_owned(),
                    sink: MIC_OUT_SINK.to_owned(),
                },
            ]
        );
    }

    #[test]
    fn stopped_translation_uses_audible_bypass_mix_without_mutating_snapshot() {
        let snapshot = RuntimeSnapshot {
            translation_running: false,
            audio_mix: AudioMixState {
                microphone_original_percent: 0,
                microphone_translation_percent: 100,
                speaker_original_percent: 0,
                speaker_translation_percent: 100,
            },
            ..RuntimeSnapshot::default()
        };

        assert_eq!(
            effective_audio_mix_for_service(&snapshot),
            AudioMixState {
                microphone_original_percent: 100,
                microphone_translation_percent: 0,
                speaker_original_percent: 100,
                speaker_translation_percent: 0,
            }
        );
        assert_eq!(snapshot.audio_mix.microphone_original_percent, 0);
        assert_eq!(snapshot.audio_mix.speaker_original_percent, 0);

        let running = RuntimeSnapshot {
            translation_running: true,
            ..snapshot
        };
        assert_eq!(effective_audio_mix_for_service(&running), running.audio_mix);
    }

    #[test]
    fn original_loopback_load_args_are_discoverable_by_audio_mix() {
        let request = OriginalLoopbackRequest {
            media_name: SPEAKER_ORIGINAL_LOOPBACK,
            source: format!("{REMOTE_IN_SINK}.monitor"),
            source_target_object: REMOTE_IN_SINK.to_owned(),
            sink: "alsa_output.headphones".to_owned(),
        };

        let args = original_loopback_load_args(&request);

        assert_eq!(args[0], "load-module");
        assert_eq!(args[1], "module-loopback");
        assert!(args.contains(&format!("source={REMOTE_IN_SINK}.monitor")));
        assert!(args.contains(&"sink=alsa_output.headphones".to_owned()));
        assert!(args.contains(&"latency_msec=20".to_owned()));
        assert!(args.contains(
            &"source_output_properties=media.name=loopback-speaker-original translator.owner=true"
                .to_owned()
        ));
        assert!(
            args.contains(
                &"sink_input_properties=media.name=loopback-speaker-original translator.owner=true"
                    .to_owned()
            )
        );
    }

    #[test]
    fn original_loopback_discovery_matches_sink_and_source_targets_by_module() {
        let sink_inputs = [raw_stream(
            SPEAKER_ORIGINAL_LOOPBACK,
            "42",
            "alsa_output.headphones",
        )];
        let source_outputs = [raw_stream(SPEAKER_ORIGINAL_LOOPBACK, "42", REMOTE_IN_SINK)];
        let request = OriginalLoopbackRequest {
            media_name: SPEAKER_ORIGINAL_LOOPBACK,
            source: format!("{REMOTE_IN_SINK}.monitor"),
            source_target_object: REMOTE_IN_SINK.to_owned(),
            sink: "alsa_output.headphones".to_owned(),
        };

        let discovered = discover_original_loopbacks(&sink_inputs, &source_outputs);

        assert_eq!(
            discovered.get("42"),
            Some(&DiscoveredOriginalLoopback {
                media_name: SPEAKER_ORIGINAL_LOOPBACK,
                source_target_object: Some(REMOTE_IN_SINK.to_owned()),
                sink_target_object: Some("alsa_output.headphones".to_owned()),
            })
        );
        assert_eq!(matching_original_loopbacks(&discovered, &request), ["42"]);
    }

    #[test]
    fn graph_maintenance_recreates_virtual_endpoints_after_degraded_inspection() {
        let store = translator_daemon::RuntimeStore::default();
        let mut graph = FakeGraph {
            inspect_health: GraphHealth::Degraded,
            inspect_calls: Cell::new(0),
            ensure_calls: 0,
        };

        maintain_audio_graph(&mut graph, &store);

        assert_eq!(graph.inspect_calls.get(), 1);
        assert_eq!(graph.ensure_calls, 1);
        assert_eq!(
            store.snapshot().audio_graph.map(|state| state.health),
            Some(GraphHealth::Ready)
        );
    }

    struct FakeGraph {
        inspect_health: GraphHealth,
        inspect_calls: Cell<usize>,
        ensure_calls: usize,
    }

    impl AudioGraph for FakeGraph {
        fn ensure_endpoints(&mut self) -> Result<AudioGraphState, AudioGraphError> {
            self.ensure_calls += 1;
            Ok(graph_state(GraphHealth::Ready))
        }

        fn inspect(&self) -> Result<AudioGraphState, AudioGraphError> {
            self.inspect_calls.set(self.inspect_calls.get() + 1);
            Ok(graph_state(self.inspect_health))
        }

        fn cleanup_owned(&mut self) -> Result<Vec<u32>, AudioGraphError> {
            unreachable!("graph maintenance does not cleanup endpoints")
        }
    }

    fn graph_state(health: GraphHealth) -> AudioGraphState {
        AudioGraphState {
            health,
            endpoints: [
                EndpointRole::MicOutSink,
                EndpointRole::VirtualMicSource,
                EndpointRole::RemoteInSink,
            ]
            .into_iter()
            .map(|role| AudioEndpointState {
                role,
                kind: role.kind(),
                name: role.name().to_owned(),
                endpoint_id: None,
                owner_module_id: None,
                available: health == GraphHealth::Ready,
                daemon_owned: health == GraphHealth::Ready,
            })
            .collect(),
            owned_module_ids: if health == GraphHealth::Ready {
                vec![101, 102, 103]
            } else {
                Vec::new()
            },
            safe_error: None,
        }
    }

    fn selected_devices() -> DeviceState {
        DeviceState {
            source: DeviceSelectionState {
                health: DeviceHealth::Available,
                selected: Some(physical_device(1, "alsa_input.microphone")),
                pinned_name: None,
                current_default: Some("alsa_input.microphone".to_owned()),
                pending_default: None,
            },
            sink: DeviceSelectionState {
                health: DeviceHealth::Available,
                selected: Some(physical_device(2, "alsa_output.headphones")),
                pinned_name: None,
                current_default: Some("alsa_output.headphones".to_owned()),
                pending_default: None,
            },
            acoustic: AcousticSafety {
                mode: OutputMode::Headphones,
                aec_capability: AecCapability::Unavailable,
                full_duplex_allowed: true,
                warning: None,
            },
        }
    }

    fn physical_device(id: u32, name: &str) -> PhysicalDevice {
        PhysicalDevice {
            id,
            name: name.to_owned(),
            description: name.to_owned(),
            active_port: None,
            active_port_type: None,
            available: true,
        }
    }

    fn raw_stream(media_name: &str, module_id: &str, target: &str) -> RawPulseStream {
        RawPulseStream {
            properties: HashMap::from([
                ("media.name".to_owned(), media_name.to_owned()),
                ("pulse.module.id".to_owned(), module_id.to_owned()),
                ("target.object".to_owned(), target.to_owned()),
            ]),
        }
    }
}
