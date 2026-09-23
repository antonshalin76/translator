use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use clap::Parser;
use serde::Deserialize;
use translator_audio::{
    AecCapability, AudioGraph, AudioGraphState, CommandResult, CommandRunner, DeviceOverride,
    DeviceWatcher, GraphHealth, MIC_OUT_SINK, PulseAudioGraph, PulseDeviceWatcher,
    PulseRoutingWatcher, REMOTE_IN_SINK, RoutingProfile, RoutingWatcher, SystemCommandRunner,
    default_journal_path, default_route_journal_path,
};
use translator_daemon::{
    ApiControllers, ApiLimits, AudioMixApplication, AudioMixController, AudioOperationGate,
    AudioOperationState, ControlApplication, ControlCommand, ControlToken, DebugCaptureLimits,
    DebugCaptureStore, FactsError, ManualRouteController, ProcessDuplexConfig, ProcessDuplexRunner,
    RoundTripController, RoundTripOwnerShutdownError, RoundTripProcessRunner,
    RoundTripRuntimeHandle, RuntimeFacts, RuntimeFactsSource, RuntimeLatencyObserver, RuntimeLease,
    RuntimeMaintenance, RuntimeSnapshot, RuntimeStore, build_router_with_controllers,
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

struct PulseResources<R = SystemCommandRunner> {
    routing: PulseRoutingWatcher<R>,
    devices: PulseDeviceWatcher<R>,
    original_loopbacks: PulseOriginalLoopbacks<R>,
    graph: Option<PulseAudioGraph<R>>,
}

impl<R: CommandRunner> PulseResources<R> {
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
            Ok(state) => store.set_devices(state.into()),
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

struct PulseManualRoutes<R = SystemCommandRunner> {
    resources: LifecycleProtected<PulseResources<R>>,
    operation_gate: AudioOperationGate,
}

impl<R: CommandRunner + Send> RuntimeMaintenance for PulseManualRoutes<R> {
    fn refresh(&self, store: &RuntimeStore) -> Result<(), translator_daemon::ControlFailure> {
        self.refresh_audio_state(store);
        Ok(())
    }
}

impl<R: CommandRunner + Send> RuntimeFactsSource for PulseManualRoutes<R> {
    fn inspect(&self, deadline: std::time::Instant) -> Result<RuntimeFacts, FactsError> {
        if std::time::Instant::now() >= deadline {
            return Err(FactsError::Expired);
        }
        if self.resources.is_stopping() {
            return Err(FactsError::DiscoveryFailed);
        }
        let resources = self
            .resources
            .inner
            .try_lock()
            .map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => FactsError::Busy,
                std::sync::TryLockError::Poisoned(_) => FactsError::DiscoveryFailed,
            })?;
        if self.resources.is_stopping() {
            return Err(FactsError::DiscoveryFailed);
        }
        inspect_runtime_facts(
            &resources.devices,
            resources
                .graph
                .as_ref()
                .ok_or(FactsError::DiscoveryFailed)?,
            &resources.routing,
            deadline,
        )
    }
}

fn inspect_runtime_facts(
    devices: &impl DeviceWatcher,
    graph: &impl AudioGraph,
    routing: &impl RoutingWatcher,
    deadline: std::time::Instant,
) -> Result<RuntimeFacts, FactsError> {
    if std::time::Instant::now() >= deadline {
        return Err(FactsError::Expired);
    }
    let devices = devices
        .read_facts_until(deadline)
        .map_err(|error| match error.code() {
            translator_audio::DeviceWatcherErrorCode::DiscoveryFailed => {
                FactsError::DiscoveryFailed
            }
            translator_audio::DeviceWatcherErrorCode::InvalidPhysicalDevice => {
                FactsError::InvalidPhysicalDevice
            }
            translator_audio::DeviceWatcherErrorCode::GraphValidationFailed => {
                FactsError::SinkValidationFailed
            }
            translator_audio::DeviceWatcherErrorCode::DeadlineExpired => FactsError::Expired,
        })?;
    if std::time::Instant::now() >= deadline {
        return Err(FactsError::Expired);
    }
    let audio_graph = graph
        .inspect_until(deadline)
        .map_err(|error| match error.code() {
            translator_audio::AudioGraphErrorCode::OwnershipJournalBusy => FactsError::Busy,
            translator_audio::AudioGraphErrorCode::DeadlineExpired => FactsError::Expired,
            _ => FactsError::DiscoveryFailed,
        })?;
    if std::time::Instant::now() >= deadline {
        return Err(FactsError::Expired);
    }
    let routes = routing
        .inspect_until(deadline)
        .map_err(|error| match error.code() {
            translator_audio::RoutingErrorCode::DeadlineExpired => FactsError::Expired,
            _ => FactsError::DiscoveryFailed,
        })?;
    if std::time::Instant::now() >= deadline {
        return Err(FactsError::Expired);
    }
    Ok(RuntimeFacts {
        devices,
        audio_graph,
        routes,
    })
}

impl<R: CommandRunner + Send> PulseManualRoutes<R> {
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

impl<R: CommandRunner + Send> ManualRouteController for PulseManualRoutes<R> {
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
        AudioOperationState::HumanRoundTrip { .. } | AudioOperationState::Calibration { .. } => {
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

fn main() -> ExitCode {
    translator_daemon::install_private_panic_hook();
    match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime.block_on(async_main()),
        Err(_) => {
            use std::io::Write;
            let _ = std::io::stderr()
                .write_all(b"{\"event\":\"runtime_start_failed\",\"code\":\"internal_error\"}\n");
            ExitCode::FAILURE
        }
    }
}

async fn async_main() -> ExitCode {
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
    let device_watcher = build_device_watcher(AecCapability::Unavailable);
    let operation_gate = AudioOperationGate::new();
    let manual_routes = Arc::new(PulseManualRoutes {
        resources: LifecycleProtected::new(PulseResources {
            routing: build_routing_watcher(),
            devices: device_watcher,
            original_loopbacks: PulseOriginalLoopbacks::new(SystemCommandRunner),
            graph: audio_graph,
        }),
        operation_gate: operation_gate.clone(),
    });
    manual_routes.initialize(&store);
    let audio_mix: Arc<dyn AudioMixController> =
        Arc::new(AudioMixApplication::new(SystemCommandRunner));
    let duplex_config = build_duplex_config(lease.token_path());
    let translation = duplex_config.clone().map(|config| {
        ControlApplication::spawn(
            store.clone(),
            Arc::new(ProcessDuplexRunner::with_observer(
                config,
                Arc::new(RuntimeLatencyObserver::new(store.clone())),
            )),
            operation_gate.clone(),
            manual_routes.clone(),
            manual_routes.clone(),
            Some(audio_mix.clone()),
        )
    });
    let round_trip = duplex_config.and_then(|config| {
        match RoundTripRuntimeHandle::try_new(
            store.clone(),
            Arc::new(RoundTripProcessRunner::new(config)),
            operation_gate.clone(),
            manual_routes.clone(),
        ) {
            Ok(controller) => Some(Arc::new(controller)),
            Err(_) => {
                tracing::error!(
                    event = "round_trip_initialization_failed",
                    code = "owner_unavailable"
                );
                None
            }
        }
    });
    let router = build_router_with_controllers(
        store.clone(),
        token,
        ApiLimits::default(),
        ApiControllers {
            manual_routes: Some(manual_routes.clone()),
            translation: translation.clone(),
            aec_calibration: None,
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
    let watcher_task = tokio::spawn(watcher_loop(translation.clone(), store.clone()));
    let debug_capture_watchdog = tokio::spawn(store.clone().run_debug_capture_watchdog());
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel::<()>();
    let server_task = tokio::spawn(async move {
        translator_daemon::serve_control(listener, router, async {
            let _ = shutdown_receiver.await;
        })
        .await
    });

    shutdown_signal().await;
    let (result, round_trip_result) = drain_control_owners(
        &operation_gate,
        (shutdown_sender, server_task),
        [watcher_task, debug_capture_watchdog],
        round_trip.as_ref(),
        translation.as_deref(),
        &store,
    )
    .await;
    if round_trip_result.is_err() {
        tracing::error!(
            event = "daemon_shutdown_failed",
            code = "round_trip_owner_failed"
        );
        return fail_stop((round_trip, manual_routes, lease)).await;
    }
    if let Err(error) = manual_routes.restore() {
        tracing::error!(event = "route_restore_failed", code = ?error.code);
    }
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

async fn drain_control_owners(
    gate: &AudioOperationGate,
    http: (
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<std::io::Result<()>>,
    ),
    background: [tokio::task::JoinHandle<()>; 2],
    round_trip: Option<&Arc<RoundTripRuntimeHandle>>,
    translation: Option<&ControlApplication>,
    store: &RuntimeStore,
) -> (std::io::Result<()>, Result<(), RoundTripOwnerShutdownError>) {
    gate.begin_stopping();
    let (shutdown, mut server) = http;
    let _ = shutdown.send(());
    for task in &background {
        task.abort();
    }
    for task in background {
        let _ = task.await;
    }
    let server_result =
        match tokio::time::timeout(std::time::Duration::from_secs(7), &mut server).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(std::io::Error::other("server task failed")),
            Err(_) => {
                server.abort();
                let _ = server.await;
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "server drain timed out",
                ))
            }
        };
    let round_trip_result = match round_trip {
        Some(controller) => drain_round_trip(controller).await,
        None => Ok(()),
    };
    if let Some(controller) = translation {
        drain_translation(controller).await;
    }
    let _ = store.set_debug_capture_enabled(false);
    store.shutdown_events();
    (server_result, round_trip_result)
}

async fn fail_stop<T>(owners: T) -> ExitCode {
    let _owners = owners;
    std::future::pending().await
}

async fn drain_translation(controller: &ControlApplication) {
    loop {
        match controller.shutdown().await {
            Ok(()) => return,
            Err(error) => tracing::error!(event = "translation_shutdown_failed", code = error.code),
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

async fn drain_round_trip(
    controller: &Arc<RoundTripRuntimeHandle>,
) -> Result<(), RoundTripOwnerShutdownError> {
    drain_round_trip_attempts(|| {
        let owner = Arc::clone(controller);
        join_round_trip_shutdown(move || owner.shutdown())
    })
    .await
}

async fn join_round_trip_shutdown(
    attempt: impl FnOnce() -> Result<(), RoundTripOwnerShutdownError> + Send + 'static,
) -> Result<(), RoundTripOwnerShutdownError> {
    tokio::task::spawn_blocking(attempt)
        .await
        .map_err(|_| RoundTripOwnerShutdownError::OwnerFailed)?
}

async fn drain_round_trip_attempts<F, R>(mut attempt: F) -> Result<(), RoundTripOwnerShutdownError>
where
    F: FnMut() -> R,
    R: std::future::Future<Output = Result<(), RoundTripOwnerShutdownError>>,
{
    loop {
        match attempt().await {
            Ok(()) => return Ok(()),
            Err(RoundTripOwnerShutdownError::OwnerFailed) => {
                return Err(RoundTripOwnerShutdownError::OwnerFailed);
            }
            Err(RoundTripOwnerShutdownError::CleanupPending) => {
                tracing::error!(
                    event = "round_trip_shutdown_failed",
                    code = "cleanup_pending"
                );
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
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
    PulseDeviceWatcher::new(SystemCommandRunner, aec_capability)
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

async fn shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler installation failed");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
}

async fn watcher_loop(controller: Option<Arc<ControlApplication>>, store: RuntimeStore) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_latency_epoch = 0;
    loop {
        interval.tick().await;
        if let Some(controller) = controller.as_ref()
            && let Err(error) = controller.execute(ControlCommand::ReconcileAudio).await
        {
            tracing::warn!(event = "audio_reconciliation_failed", code = error.code);
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
        "devices": translator_daemon::DeviceState::from(devices),
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
mod admission_adapter_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;
    use translator_audio::{CommandRunError, RouteResolution};
    use translator_daemon::{AdmittedDuplex, DuplexRunner, DuplexStartResult};

    #[derive(Default)]
    struct Reads {
        calls: Mutex<Vec<(Vec<String>, Instant)>>,
        second_default: AtomicBool,
        fail_at: Option<usize>,
        expire_at: Option<usize>,
    }

    #[derive(Clone, Default)]
    struct ReadRunner(Arc<Reads>);

    impl CommandRunner for ReadRunner {
        fn run_until(
            &self,
            program: &str,
            args: &[String],
            deadline: Instant,
        ) -> Result<CommandResult, CommandRunError> {
            assert_eq!(program, "pactl");
            let index = {
                let mut calls = self.0.calls.lock().unwrap();
                let index = calls.len();
                calls.push((args.to_vec(), deadline));
                index
            };
            if self.0.fail_at == Some(index) {
                return Err(CommandRunError::SpawnFailed);
            }
            if self.0.expire_at == Some(index) {
                std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
            }
            let args: Vec<_> = args.iter().map(String::as_str).collect();
            let text = match args.as_slice() {
                ["get-default-source"] => "alsa_input.microphone".to_owned(),
                ["get-default-sink"] => if self.0.second_default.load(Ordering::SeqCst) { "alsa_output.other-headphones" } else { "alsa_output.headphones" }.to_owned(),
                ["--format=json", "list", "sources"] => serde_json::json!([{"index":1,"name":"alsa_input.microphone","owner_module":80,"monitor_source":"","properties":{"device.api":"alsa","device.class":"sound","media.class":"Audio/Source"},"active_port":null}]).to_string(),
                ["--format=json", "list", "sinks"] => serde_json::json!([
                    {"index":2,"name":"alsa_output.headphones","owner_module":80,"monitor_source":"alsa_output.headphones.monitor","properties":{"device.api":"alsa","device.class":"sound","media.class":"Audio/Sink"},"active_port":"analog-output-headphones","ports":[{"name":"analog-output-headphones","type":"Headphones"}]},
                    {"index":3,"name":"alsa_output.other-headphones","owner_module":81,"monitor_source":"alsa_output.other-headphones.monitor","properties":{"device.api":"alsa","device.class":"sound","media.class":"Audio/Sink"},"active_port":"analog-output-headphones","ports":[{"name":"analog-output-headphones","type":"Headphones"}]}
                ]).to_string(),
                ["--format=json", "list", "sink-inputs" | "source-outputs"] => "[]".to_owned(),
                _ => panic!("facts inspection attempted an unexpected command: {args:?}"),
            };
            Ok(CommandResult::success(text.into_bytes()))
        }
    }

    fn adapter(runner: ReadRunner, journal: std::path::PathBuf) -> PulseManualRoutes<ReadRunner> {
        PulseManualRoutes {
            resources: LifecycleProtected::new(PulseResources {
                routing: PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production),
                devices: PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable),
                original_loopbacks: PulseOriginalLoopbacks::new(runner.clone()),
                graph: Some(PulseAudioGraph::new(runner, journal)),
            }),
            operation_gate: AudioOperationGate::new(),
        }
    }

    fn journal_entries(path: &std::path::Path) -> Vec<std::ffi::OsString> {
        let mut entries = std::fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        entries.sort();
        entries
    }

    #[test]
    fn facts_adapter_reads_all_ports_without_committing_pins_or_ownership() {
        let temp = tempdir().unwrap();
        let lock = temp.path().join(".modules.json.lock");
        std::fs::write(&lock, b"").unwrap();
        let before_entries = journal_entries(temp.path());
        let runner = ReadRunner::default();
        let adapter = adapter(runner.clone(), temp.path().join("modules.json"));
        let original_deadline = Instant::now() + Duration::from_secs(1);
        let first = adapter.inspect(original_deadline).unwrap();
        assert_ne!(first.audio_graph.health, GraphHealth::Ready);
        assert_eq!(first.routes.resolution, RouteResolution::NoCandidate);
        assert_eq!(
            first.devices.sink.selected.unwrap().name,
            "alsa_output.headphones"
        );
        assert!(
            adapter
                .resources
                .inner
                .lock()
                .unwrap()
                .devices
                .selected_sink_name()
                .is_none()
        );
        runner.0.second_default.store(true, Ordering::SeqCst);
        let second = adapter.inspect(original_deadline).unwrap();
        assert_eq!(
            second.devices.sink.selected.unwrap().name,
            "alsa_output.other-headphones"
        );
        assert!(
            adapter
                .resources
                .inner
                .lock()
                .unwrap()
                .devices
                .selected_sink_name()
                .is_none()
        );
        assert_eq!(journal_entries(temp.path()), before_entries);
        assert_eq!(std::fs::read(lock).unwrap(), b"");
        let calls = runner.0.calls.lock().unwrap();
        assert_eq!(calls.len(), 20);
        let expected_device_commands = [
            "--format=json list sinks",
            "--format=json list sources",
            "get-default-sink",
            "get-default-source",
        ];
        let expected_graph_and_route_commands = [
            "--format=json list sinks",
            "--format=json list sources",
            "--format=json list sink-inputs",
            "--format=json list source-outputs",
            "--format=json list sources",
            "--format=json list sinks",
        ];
        for pass in calls.chunks_exact(10) {
            let mut device_commands = pass[..4]
                .iter()
                .map(|(args, _)| args.join(" "))
                .collect::<Vec<_>>();
            device_commands.sort();
            assert_eq!(device_commands, expected_device_commands);
            assert_eq!(
                pass[4..]
                    .iter()
                    .map(|(args, _)| args.join(" "))
                    .collect::<Vec<_>>(),
                expected_graph_and_route_commands,
            );
        }
        assert!(
            calls
                .iter()
                .all(|(_, deadline)| *deadline == original_deadline)
        );
    }

    #[test]
    fn facts_adapter_absent_ownership_never_initializes_graph() {
        let temp = tempdir().unwrap();
        let absent = temp.path().join("absent");
        let runner = ReadRunner::default();
        let adapter = adapter(runner.clone(), absent.join("modules.json"));
        assert!(matches!(
            adapter.inspect(Instant::now() + Duration::from_secs(1)),
            Err(FactsError::DiscoveryFailed)
        ));
        assert!(!absent.exists());
        assert_eq!(runner.0.calls.lock().unwrap().len(), 4);
        assert!(
            adapter
                .resources
                .inner
                .lock()
                .unwrap()
                .devices
                .selected_sink_name()
                .is_none()
        );
    }

    #[test]
    fn facts_adapter_failed_or_late_port_never_enters_the_next_port() {
        for index in [0, 4, 6] {
            for expire in [false, true] {
                let temp = tempdir().unwrap();
                std::fs::write(temp.path().join(".modules.json.lock"), b"").unwrap();
                let before = journal_entries(temp.path());
                let runner = ReadRunner(Arc::new(Reads {
                    fail_at: (!expire).then_some(index),
                    expire_at: expire.then_some(index),
                    ..Reads::default()
                }));
                let adapter = adapter(runner.clone(), temp.path().join("modules.json"));
                let deadline = Instant::now() + Duration::from_millis(30);
                let result = adapter.inspect(deadline);
                assert!(matches!(
                    (result, expire),
                    (Err(FactsError::Expired), true) | (Err(FactsError::DiscoveryFailed), false)
                ));
                let calls = runner.0.calls.lock().unwrap();
                assert_eq!(calls.len(), index + 1);
                assert!(calls.iter().all(|(_, observed)| *observed == deadline));
                assert_eq!(journal_entries(temp.path()), before);
                assert!(
                    adapter
                        .resources
                        .inner
                        .lock()
                        .unwrap()
                        .devices
                        .selected_sink_name()
                        .is_none()
                );
            }
        }
    }

    #[test]
    fn facts_adapter_busy_stopping_and_expired_entry_issue_no_commands() {
        let temp = tempdir().unwrap();
        let runner = ReadRunner::default();
        let adapter = adapter(runner.clone(), temp.path().join("modules.json"));
        let guard = adapter.resources.inner.lock().unwrap();
        let busy = adapter.inspect(Instant::now() + Duration::from_secs(1));
        drop(guard);
        assert!(matches!(busy, Err(FactsError::Busy)));
        assert!(matches!(
            adapter.inspect(Instant::now()),
            Err(FactsError::Expired)
        ));
        adapter.resources.stopping.store(true, Ordering::SeqCst);
        assert!(matches!(
            adapter.inspect(Instant::now() + Duration::from_secs(1)),
            Err(FactsError::DiscoveryFailed)
        ));
        assert!(runner.0.calls.lock().unwrap().is_empty());
        assert!(journal_entries(temp.path()).is_empty());
    }

    struct NeverNative(AtomicUsize);

    impl DuplexRunner for NeverNative {
        fn start(&self, _: AdmittedDuplex, _: tokio::time::Instant) -> DuplexStartResult {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(translator_daemon::DuplexStartFailure::rejected(
                translator_daemon::DuplexRuntimeError::StartFailed,
            ))
        }
    }

    #[tokio::test]
    async fn rejected_start_using_real_facts_adapter_has_no_state_event_or_audio_effect() {
        use axum::{
            body::Body,
            http::{Method, Request},
        };
        use http_body_util::BodyExt;
        use tower::ServiceExt;
        let temp = tempdir().unwrap();
        std::fs::write(temp.path().join(".modules.json.lock"), b"").unwrap();
        let before_entries = journal_entries(temp.path());
        let runner = ReadRunner::default();
        let adapter = Arc::new(adapter(runner.clone(), temp.path().join("modules.json")));
        let store = RuntimeStore::default();
        let before = serde_json::to_value(store.snapshot()).unwrap();
        let native = Arc::new(NeverNative(AtomicUsize::new(0)));
        let controller = ControlApplication::spawn(
            store.clone(),
            native.clone(),
            adapter.operation_gate.clone(),
            adapter.clone(),
            adapter.clone(),
            None,
        );
        let token = "4242424242424242424242424242424242424242424242424242424242424242";
        let router = build_router_with_controllers(
            store.clone(),
            ControlToken::parse(token).unwrap(),
            ApiLimits::default(),
            ApiControllers {
                translation: Some(controller.clone()),
                ..ApiControllers::default()
            },
        );
        let response = router
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/events/stream")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut events = response.into_body();
        events.frame().await.unwrap().unwrap();
        let result = controller.execute(ControlCommand::Start).await;
        let after = serde_json::to_value(store.snapshot()).unwrap();
        let unexpected_event =
            tokio::time::timeout(Duration::from_millis(20), events.frame()).await;
        let gate = adapter.operation_gate.state();
        drop(events);
        controller.shutdown().await.unwrap();

        assert_eq!(result.unwrap_err().code, "translation_precondition_failed");
        assert_eq!(after, before);
        assert!(unexpected_event.is_err());
        assert_eq!(native.0.load(Ordering::SeqCst), 0);
        assert_eq!(gate, AudioOperationState::Idle);
        assert_eq!(runner.0.calls.lock().unwrap().len(), 10);
        assert!(
            adapter
                .resources
                .inner
                .lock()
                .unwrap()
                .devices
                .selected_sink_name()
                .is_none()
        );
        assert_eq!(journal_entries(temp.path()), before_entries);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    };
    use std::time::{Duration, Instant};

    use super::{
        DiscoveredOriginalLoopback, LifecycleProtected, MICROPHONE_ORIGINAL_LOOPBACK,
        ManualRouteAdmission, OriginalLoopbackRequest, RawPulseStream, SPEAKER_ORIGINAL_LOOPBACK,
        discover_original_loopbacks, maintain_audio_graph, manual_route_admission,
        matching_original_loopbacks, original_loopback_load_args, original_loopback_requests,
    };
    use translator_audio::{
        AecCapability, AudioEndpointState, AudioGraph, AudioGraphError, AudioGraphState,
        DeviceHealth, DeviceSelectionState, EndpointRole, GraphHealth, MIC_OUT_SINK, OutputMode,
        PhysicalDevice, REMOTE_IN_SINK,
    };
    use translator_daemon::{
        AcousticSafety, AdmittedDuplex, AudioMixState, AudioOperationState, DeviceState,
        RuntimeSnapshot,
    };
    use uuid::Uuid;

    struct TestFacts;
    impl translator_daemon::RuntimeFactsSource for TestFacts {
        fn inspect(
            &self,
            _: Instant,
        ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
            let devices = selected_devices();
            Ok(translator_daemon::RuntimeFacts {
                devices: translator_audio::DeviceFacts {
                    source: devices.source,
                    sink: devices.sink,
                    output_mode: devices.acoustic.mode,
                    aec_capability: devices.acoustic.aec_capability,
                },
                audio_graph: AudioGraphState {
                    health: GraphHealth::Ready,
                    endpoints: Vec::new(),
                    owned_module_ids: Vec::new(),
                    safe_error: None,
                },
                routes: translator_audio::RoutingState {
                    candidates: Vec::new(),
                    source_outputs: Vec::new(),
                    conflicting_stream_ids: Vec::new(),
                    active_route: None,
                    resolution: translator_audio::RouteResolution::NoCandidate,
                },
            })
        }
    }

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
        fn ensure_endpoints_until(
            &mut self,
            _: std::time::Instant,
        ) -> Result<AudioGraphState, AudioGraphError> {
            self.ensure_calls += 1;
            Ok(graph_state(GraphHealth::Ready))
        }

        fn inspect_until(&self, _: std::time::Instant) -> Result<AudioGraphState, AudioGraphError> {
            self.inspect_calls.set(self.inspect_calls.get() + 1);
            Ok(graph_state(self.inspect_health))
        }

        fn cleanup_owned_until(
            &mut self,
            _: std::time::Instant,
        ) -> Result<Vec<u32>, AudioGraphError> {
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
                pinned_name: Some("alsa_input.microphone".to_owned()),
                current_default: Some("alsa_input.microphone".to_owned()),
                pending_default: None,
            },
            sink: DeviceSelectionState {
                health: DeviceHealth::Available,
                selected: Some(physical_device(2, "alsa_output.headphones")),
                pinned_name: Some("alsa_output.headphones".to_owned()),
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

    #[derive(Default)]
    struct DrainState {
        stop_failures: AtomicUsize,
        recovery_failures: AtomicUsize,
        unknown: AtomicBool,
        failed: tokio::sync::Notify,
        attempts: Mutex<Vec<Instant>>,
    }

    #[derive(Clone)]
    struct DrainRuntime(Arc<DrainState>);

    impl DrainRuntime {
        fn attempt(&self, failures: &AtomicUsize) -> Result<(), translator_daemon::ControlFailure> {
            self.0.attempts.lock().unwrap().push(Instant::now());
            if failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                self.0.failed.notify_one();
                Err(translator_daemon::ControlFailure {
                    status: axum::http::StatusCode::CONFLICT,
                    code: "audio_mix_state_unknown",
                })
            } else {
                Ok(())
            }
        }
    }

    impl translator_daemon::DuplexRunner for DrainRuntime {
        fn start(
            &self,
            _: AdmittedDuplex,
            _: tokio::time::Instant,
        ) -> translator_daemon::DuplexStartResult {
            Ok(Box::new(self.clone()))
        }
    }

    impl translator_daemon::ActiveDuplexRuntime for DrainRuntime {
        fn stop(
            &mut self,
            _: tokio::time::Instant,
        ) -> Result<(), translator_daemon::DuplexRuntimeError> {
            if self.0.unknown.load(Ordering::SeqCst) {
                return Ok(());
            }
            self.attempt(&self.0.stop_failures)
                .map_err(|_| translator_daemon::DuplexRuntimeError::StopFailed)
        }
    }

    impl translator_daemon::RuntimeMaintenance for DrainRuntime {
        fn refresh(
            &self,
            _: &translator_daemon::RuntimeStore,
        ) -> Result<(), translator_daemon::ControlFailure> {
            Ok(())
        }
    }

    impl translator_daemon::AudioMixController for DrainRuntime {
        fn apply_desired(
            &self,
            _: AudioMixState,
            _: translator_daemon::TranslationMixMode,
        ) -> Result<(), translator_daemon::ControlFailure> {
            Ok(())
        }
        fn reconcile_committed(
            &self,
            _: translator_daemon::TranslationMixMode,
        ) -> Result<(), translator_daemon::ControlFailure> {
            if self.0.unknown.load(Ordering::SeqCst) {
                Err(translator_daemon::ControlFailure {
                    status: axum::http::StatusCode::CONFLICT,
                    code: "audio_mix_state_unknown",
                })
            } else {
                Ok(())
            }
        }
        fn recover_committed(
            &self,
            _: translator_daemon::TranslationMixMode,
        ) -> Result<(), translator_daemon::ControlFailure> {
            self.attempt(&self.0.recovery_failures)?;
            self.0.unknown.store(false, Ordering::SeqCst);
            Ok(())
        }
    }

    async fn assert_drain_retries(stop_failures: usize, recovery_failures: usize) {
        use translator_daemon::{
            AudioOperationGate, ControlApplication, ControlCommand, RuntimeStore,
        };
        let state = Arc::new(DrainState::default());
        let native = Arc::new(DrainRuntime(state.clone()));
        let gate = AudioOperationGate::new();
        let control = ControlApplication::spawn(
            RuntimeStore::default(),
            native.clone(),
            gate.clone(),
            Arc::new(TestFacts),
            native.clone(),
            Some(native),
        );
        control.execute(ControlCommand::Start).await.unwrap();
        state.stop_failures.store(stop_failures, Ordering::SeqCst);
        state
            .recovery_failures
            .store(recovery_failures, Ordering::SeqCst);
        state.unknown.store(recovery_failures > 0, Ordering::SeqCst);
        let route_observed = Arc::new(AtomicBool::new(false));
        let owner = control.clone();
        let observer = route_observed.clone();
        let mut drain = tokio::spawn(async move {
            super::drain_translation(&owner).await;
            observer.store(true, Ordering::SeqCst);
        });
        tokio::time::timeout(Duration::from_secs(2), state.failed.notified())
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let pending = !drain.is_finished();
        let held_state = gate.state();
        let cleanup_before_success = route_observed.load(Ordering::SeqCst);
        let rejects_normal_work = control.execute(ControlCommand::Start).await.is_err();
        let completed = match tokio::time::timeout(Duration::from_secs(5), &mut drain).await {
            Ok(result) => result.is_ok(),
            Err(_) => {
                drain.abort();
                let _ = drain.await;
                false
            }
        };
        if !pending || !completed {
            for _ in 0..3 {
                if control.shutdown().await.is_ok() {
                    break;
                }
            }
        }
        assert!(
            completed,
            "bounded drain fixture must complete without detaching tasks"
        );
        assert!(
            pending,
            "main must retain and retry a failed controller drain"
        );
        assert!(
            !cleanup_before_success,
            "route cleanup must wait for successful drain"
        );
        assert!(rejects_normal_work);
        if stop_failures > 0 {
            assert_eq!(held_state, AudioOperationState::Production);
        }
        assert_eq!(gate.state(), AudioOperationState::Idle);
        assert!(route_observed.load(Ordering::SeqCst));
        let attempts = state.attempts.lock().unwrap();
        assert_eq!(attempts.len(), stop_failures + recovery_failures + 1);
        assert!(
            attempts
                .windows(2)
                .all(|pair| pair[1].duration_since(pair[0]) >= Duration::from_secs(1)),
            "completed failures must be separated by a full second before retry"
        );
    }

    #[tokio::test]
    async fn failed_native_shutdown_is_retried_before_route_cleanup() {
        assert_drain_retries(1, 0).await;
    }

    #[tokio::test]
    async fn twice_failed_native_shutdown_remains_owned_with_paced_retries() {
        assert_drain_retries(2, 0).await;
    }

    #[tokio::test]
    async fn failed_mix_recovery_uses_the_same_owned_shutdown_retry() {
        assert_drain_retries(0, 1).await;
    }

    struct RoundTripDrain(Arc<Mutex<Vec<Instant>>>);

    #[derive(Default)]
    struct PanickingOwnerDrop {
        admission: Option<(Arc<AtomicBool>, Arc<AtomicBool>)>,
    }

    impl Drop for PanickingOwnerDrop {
        fn drop(&mut self) {
            if let Some((closed, observed)) = &self.admission {
                observed.store(closed.load(Ordering::SeqCst), Ordering::SeqCst);
            }
            panic!("injected resource-free owner failure");
        }
    }

    impl translator_daemon::RoundTripRunner for PanickingOwnerDrop {
        fn start(
            &self,
            _: AdmittedDuplex,
            _: Uuid,
            _: translator_daemon::RoundTripProgress,
            _: Instant,
        ) -> Result<
            Box<dyn translator_daemon::ActiveRoundTripRuntime>,
            translator_daemon::RoundTripRuntimeError,
        > {
            Err(translator_daemon::RoundTripRuntimeError::StartFailed)
        }
    }

    #[tokio::test]
    async fn permanently_failed_round_trip_owner_does_not_repeat_shutdown_forever() {
        let owner = Arc::new(
            translator_daemon::RoundTripRuntimeHandle::try_new(
                translator_daemon::RuntimeStore::default(),
                Arc::new(PanickingOwnerDrop::default()),
                translator_daemon::AudioOperationGate::new(),
                Arc::new(TestFacts),
            )
            .unwrap(),
        );
        let attempted_owner = Arc::clone(&owner);
        let mut driver =
            tokio::spawn(async move { super::drain_round_trip(&attempted_owner).await });
        let completed = match tokio::time::timeout(Duration::from_millis(500), &mut driver).await {
            Ok(result) => {
                assert_eq!(
                    result.unwrap(),
                    Err(super::RoundTripOwnerShutdownError::OwnerFailed)
                );
                true
            }
            Err(_) => {
                driver.abort();
                let _ = driver.await;
                false
            }
        };
        assert_eq!(
            owner.shutdown(),
            Err(super::RoundTripOwnerShutdownError::OwnerFailed)
        );
        assert!(
            completed,
            "a dead owner must return terminal failure rather than retry forever"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn typed_pending_cleanup_retries_sequentially_after_one_second() {
        let mut attempts = Vec::new();
        let result = super::drain_round_trip_attempts(|| {
            attempts.push(tokio::time::Instant::now());
            std::future::ready(if attempts.len() < 3 {
                Err(super::RoundTripOwnerShutdownError::CleanupPending)
            } else {
                Ok(())
            })
        })
        .await;
        assert_eq!(result, Ok(()));
        assert_eq!(attempts.len(), 3);
        assert!(
            attempts
                .windows(2)
                .all(|pair| pair[1] - pair[0] == Duration::from_secs(1))
        );
    }

    #[tokio::test]
    async fn fatal_shutdown_and_joined_blocking_panic_each_end_after_one_attempt() {
        for panic in [false, true] {
            let attempts = Arc::new(AtomicUsize::new(0));
            let observed = attempts.clone();
            let mut driver = tokio::spawn(async move {
                super::drain_round_trip_attempts(|| {
                    let observed = observed.clone();
                    super::join_round_trip_shutdown(move || {
                        observed.fetch_add(1, Ordering::SeqCst);
                        assert!(!panic, "injected blocking shutdown panic");
                        Err(super::RoundTripOwnerShutdownError::OwnerFailed)
                    })
                })
                .await
            });
            let result = match tokio::time::timeout(Duration::from_millis(500), &mut driver).await {
                Ok(result) => result.unwrap(),
                Err(_) => {
                    driver.abort();
                    let _ = driver.await;
                    panic!("terminal failure must not enter the retry sleep");
                }
            };
            assert_eq!(result, Err(super::RoundTripOwnerShutdownError::OwnerFailed));
            assert_eq!(attempts.load(Ordering::SeqCst), 1);
        }
    }

    struct DropSpy(Arc<AtomicUsize>);

    impl Drop for DropSpy {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn fatal_retention_holds_every_dependent_owner_until_fixture_is_cancelled() {
        let counters = [0; 3].map(|_| Arc::new(AtomicUsize::new(0)));
        let owners = (
            DropSpy(counters[0].clone()),
            DropSpy(counters[1].clone()),
            DropSpy(counters[2].clone()),
        );
        let mut driver = tokio::spawn(super::fail_stop(owners));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut driver)
                .await
                .is_err()
        );
        assert!(
            counters
                .iter()
                .all(|count| count.load(Ordering::SeqCst) == 0)
        );
        driver.abort();
        assert!(driver.await.unwrap_err().is_cancelled());
        assert!(
            counters
                .iter()
                .all(|count| count.load(Ordering::SeqCst) == 1)
        );
    }

    #[tokio::test]
    async fn fatal_round_trip_still_drains_independent_owners_after_closing_admission() {
        use tower::ServiceExt;
        let store = translator_daemon::RuntimeStore::default();
        let capture = tempfile::tempdir().unwrap();
        store.configure_debug_capture(
            super::DebugCaptureStore::open(capture.path(), super::DebugCaptureLimits::default())
                .unwrap(),
        );
        store.set_debug_capture_enabled(true).unwrap();
        let gate = super::AudioOperationGate::new();
        let state = Arc::new(DrainState::default());
        let runtime = Arc::new(DrainRuntime(state.clone()));
        let control = super::ControlApplication::spawn(
            store.clone(),
            runtime.clone(),
            gate.clone(),
            Arc::new(TestFacts),
            runtime,
            None,
        );
        control.execute(super::ControlCommand::Start).await.unwrap();
        let http_closed = Arc::new(AtomicBool::new(false));
        let admission_before_attempt = Arc::new(AtomicBool::new(false));
        let owner = Arc::new(
            super::RoundTripRuntimeHandle::try_new(
                store.clone(),
                Arc::new(PanickingOwnerDrop {
                    admission: Some((http_closed.clone(), admission_before_attempt.clone())),
                }),
                gate.clone(),
                Arc::new(TestFacts),
            )
            .unwrap(),
        );
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let observed_gate = gate.clone();
        let server = tokio::spawn(async move {
            stopped.await.unwrap();
            assert_eq!(observed_gate.state(), AudioOperationState::Stopping);
            http_closed.store(true, Ordering::SeqCst);
            Ok(())
        });
        let background_drops = Arc::new(AtomicUsize::new(0));
        let background = [0; 2].map(|_| {
            let owned = DropSpy(background_drops.clone());
            tokio::spawn(async move {
                let _owned = owned;
                std::future::pending::<()>().await;
            })
        });
        let driver_store = store.clone();
        let mut driver = tokio::spawn(async move {
            let results = super::drain_control_owners(
                &gate,
                (stop, server),
                background,
                Some(&owner),
                Some(&control),
                &driver_store,
            )
            .await;
            let command_after_shutdown = control.execute(super::ControlCommand::Start).await;
            (results, command_after_shutdown)
        });
        let ((server_result, owner_result), command_after_shutdown) =
            match tokio::time::timeout(Duration::from_secs(3), &mut driver).await {
                Ok(result) => result.unwrap(),
                Err(_) => {
                    driver.abort();
                    let _ = driver.await;
                    panic!("fatal owner fixture did not drain");
                }
            };
        assert!(server_result.is_ok());
        assert_eq!(
            owner_result,
            Err(super::RoundTripOwnerShutdownError::OwnerFailed)
        );
        assert!(admission_before_attempt.load(Ordering::SeqCst));
        assert_eq!(background_drops.load(Ordering::SeqCst), 2);
        assert_eq!(state.attempts.lock().unwrap().len(), 1);
        assert!(command_after_shutdown.is_err());
        assert!(!store.snapshot().debug_capture_enabled);
        let token = "a".repeat(64);
        let router = translator_daemon::build_router(
            store,
            super::ControlToken::parse(&token).unwrap(),
            super::ApiLimits::default(),
        );
        let response = router
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/events/stream")
                    .header("authorization", format!("Bearer {token}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
    }

    impl translator_daemon::RoundTripRunner for RoundTripDrain {
        fn start(
            &self,
            _: AdmittedDuplex,
            session_id: Uuid,
            progress: translator_daemon::RoundTripProgress,
            _: Instant,
        ) -> Result<
            Box<dyn translator_daemon::ActiveRoundTripRuntime>,
            translator_daemon::RoundTripRuntimeError,
        > {
            progress.fail(
                session_id,
                translator_daemon::RoundTripErrorCode::RuntimeFailed,
            );
            Ok(Box::new(Self(Arc::clone(&self.0))))
        }
    }

    impl translator_daemon::ActiveRoundTripRuntime for RoundTripDrain {
        fn stop(
            &mut self,
            _: Instant,
            _: Instant,
        ) -> Result<translator_daemon::RoundTripTerminal, translator_daemon::RoundTripRuntimeError>
        {
            let mut attempts = self.0.lock().unwrap();
            attempts.push(Instant::now());
            if attempts.len() < 3 {
                Err(translator_daemon::RoundTripRuntimeError::StopFailed)
            } else {
                Ok(translator_daemon::RoundTripTerminal::Failed(
                    translator_daemon::RoundTripErrorCode::RuntimeFailed,
                ))
            }
        }
    }

    #[tokio::test]
    async fn failed_terminal_round_trip_is_drained_and_owner_joined_before_route_cleanup() {
        use translator_daemon::RoundTripController;
        let store = translator_daemon::RuntimeStore::default();
        store.set_devices(selected_devices());
        store.set_audio_graph(AudioGraphState {
            health: GraphHealth::Ready,
            endpoints: Vec::new(),
            owned_module_ids: Vec::new(),
            safe_error: None,
        });
        store.set_routes(translator_audio::RoutingState {
            candidates: Vec::new(),
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: translator_audio::RouteResolution::NoCandidate,
        });
        let gate = translator_daemon::AudioOperationGate::new();
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let owner = Arc::new(
            translator_daemon::RoundTripRuntimeHandle::try_new(
                store.clone(),
                Arc::new(RoundTripDrain(Arc::clone(&attempts))),
                gate.clone(),
                Arc::new(TestFacts),
            )
            .unwrap(),
        );
        owner.start().unwrap();
        assert!(owner.stop().is_err());
        assert_eq!(
            store.snapshot().self_test.status.checkpoint,
            Some(translator_daemon::RoundTripCheckpoint::Failed)
        );
        assert!(store.snapshot().self_test.status.cleanup_pending);
        let cleaned = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&cleaned);
        let driver_owner = Arc::clone(&owner);
        let mut driver = tokio::spawn(async move {
            super::drain_round_trip(&driver_owner).await.unwrap();
            observed.store(true, Ordering::SeqCst);
        });
        let early_return = tokio::time::timeout(Duration::from_millis(100), &mut driver)
            .await
            .is_ok();
        let retained = gate.state();
        let early_cleanup = cleaned.load(Ordering::SeqCst);
        if !early_return {
            match tokio::time::timeout(Duration::from_secs(3), &mut driver).await {
                Ok(result) => result.unwrap(),
                Err(_) => {
                    driver.abort();
                    let _ = driver.await;
                    panic!("round-trip drain timed out");
                }
            }
        }
        assert!(!early_return && !early_cleanup);
        assert!(matches!(
            retained,
            AudioOperationState::HumanRoundTrip { .. }
        ));
        assert_eq!(gate.state(), AudioOperationState::Idle);
        assert!(!store.snapshot().self_test.status.cleanup_pending);
        let attempts = attempts.lock().unwrap();
        assert_eq!(attempts.len(), 3);
        assert!(attempts[2].duration_since(attempts[1]) >= Duration::from_secs(1));
    }
}
