use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Deserialize;
use translator_audio::{
    AecCapability, AudioGraph, AudioGraphState, CommandResult, CommandRunner, DeviceOverride,
    DeviceWatcher, GraphHealth, MIC_OUT_SINK, NativeAecIdentity, OutputMode, PulseAudioGraph,
    PulseDeviceWatcher, PulseRoutingWatcher, REMOTE_IN_SINK, RoutingProfile, RoutingWatcher,
    SystemCommandRunner, default_journal_path, default_route_journal_path,
};
use translator_daemon::{
    AecCalibrationController, AecCalibrationCoordinator, AecCalibrationEngineError,
    AecRuntimeAuthority, ApiControllers, ApiLimits, AudioMixApplication, AudioMixController,
    AudioOperationGate, AudioOperationState, ControlApplication, ControlCommand, ControlFailure,
    ControlToken, DebugCaptureLimits, DebugCaptureStore, FactsError, ManualRouteController,
    NativeAecCalibrationEngine, NativeAecEnvironment, NativeAecPairFacts, NativeAecPositiveFixture,
    ProcessDuplexConfig, ProcessDuplexRunner, RoundTripController, RoundTripOwnerShutdownError,
    RoundTripProcessRunner, RoundTripRuntimeHandle, RuntimeFacts, RuntimeFactsSource,
    RuntimeLatencyObserver, RuntimeLease, RuntimeMaintenance, RuntimeSnapshot, RuntimeStore,
    TranslationMixMode, build_router_with_controllers, validate_listen_address,
};

const SPEAKER_ORIGINAL_LOOPBACK: &str = "loopback-speaker-original";
const MICROPHONE_ORIGINAL_LOOPBACK: &str = "loopback-microphone-original";
const ORIGINAL_LOOPBACK_LATENCY_MS: u16 = 20;
const NATIVE_SOURCE: &str = "alsa_input.pci-0000_00_1f.3.analog-stereo";
const NATIVE_SINK: &str = "alsa_output.pci-0000_00_1f.3.analog-stereo";

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
        if let Err(error) = self.refresh(store) {
            tracing::warn!(event = "audio_state_initialization_incomplete", code = ?error.code());
        }
    }

    fn refresh(&mut self, store: &RuntimeStore) -> Result<(), OriginalLoopbackError> {
        self.refresh_facts_only(store);
        self.ensure_original_loopbacks(store)
    }

    fn refresh_facts_only(&mut self, store: &RuntimeStore) {
        self.refresh_graph(store);
        match self.routing.reconcile(None) {
            Ok(state) => store.set_routes(state),
            Err(error) => {
                tracing::warn!(event = "route_reconciliation_failed", code = ?error.code());
                store.clear_routes("route_reconciliation_failed");
            }
        }
        self.refresh_devices(store);
    }

    fn refresh_graph_and_devices(&mut self, store: &RuntimeStore) {
        self.refresh_graph(store);
        self.refresh_devices(store);
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

    fn ensure_original_loopbacks(&self, store: &RuntimeStore) -> Result<(), OriginalLoopbackError> {
        let result = self
            .original_loopbacks
            .ensure_without_new_mic(&store.snapshot());
        if let Err(error) = &result {
            tracing::warn!(
                event = "original_loopback_reconciliation_failed",
                code = ?error.code()
            );
        }
        result
    }

    fn cleanup_graph(&mut self) {
        if let Err(error) = self.original_loopbacks.cleanup_all() {
            tracing::warn!(event = "original_loopback_cleanup_failed", code = ?error.code());
            return;
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

struct AecProjectedRoutes<R = SystemCommandRunner> {
    routes: Arc<PulseManualRoutes<R>>,
    coordinator: Arc<AecCalibrationCoordinator>,
    environment: Arc<PulseNativeAecEnvironment<R>>,
}

impl<R: CommandRunner + Send + Sync> AecProjectedRoutes<R> {
    fn observed_devices_until(
        &self,
        deadline: std::time::Instant,
    ) -> Result<translator_audio::DeviceFacts, FactsError> {
        if self.routes.resources.is_stopping() {
            return Err(FactsError::DiscoveryFailed);
        }
        let confirmed = self
            .routes
            .resources
            .inner
            .try_lock()
            .map_err(|_| FactsError::Busy)?
            .devices
            .confirmed_headphone_facts_until(deadline)
            .map_err(|_| FactsError::InvalidPhysicalDevice)?;
        match confirmed {
            Some(devices) => Ok(devices),
            None => self
                .environment
                .inspect_devices(deadline)
                .map_err(|_| FactsError::InvalidPhysicalDevice),
        }
    }

    fn project(&self, store: &RuntimeStore) {
        match self
            .observed_devices_until(std::time::Instant::now() + std::time::Duration::from_secs(2))
        {
            Ok(mut devices) => {
                devices.aec_capability = self.capability();
                store.set_devices(devices.into());
            }
            Err(_) => store.clear_devices("aec_pair_unavailable"),
        }
    }

    fn capability(&self) -> AecCapability {
        match self.coordinator.capability() {
            validated @ AecCapability::ValidatedFor { .. } => validated,
            AecCapability::ValidationFailed => AecCapability::ValidationFailed,
            _ => AecCapability::Unavailable,
        }
    }

    fn prepare_native(
        &self,
        candidate: &RuntimeSnapshot,
    ) -> Result<(), translator_daemon::ControlFailure> {
        self.routes
            .resources
            .with_active(|resources| {
                resources
                    .original_loopbacks
                    .ensure_with_native_speaker(candidate)
            })
            .ok_or(native_route_failure())?
            .map_err(|_| native_route_failure())
    }

    fn prepare_existing(
        &self,
        snapshot: &RuntimeSnapshot,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if native_pair_selected(snapshot) {
            self.prepare_native(snapshot)
        } else {
            self.routes
                .resources
                .with_active(|resources| {
                    resources
                        .original_loopbacks
                        .ensure_without_new_mic(snapshot)
                })
                .ok_or(native_route_failure())?
                .map_err(|_| native_route_failure())
        }
    }
}

impl<R: CommandRunner + Send + Sync> RuntimeFactsSource for AecProjectedRoutes<R> {
    fn inspect(&self, deadline: std::time::Instant) -> Result<RuntimeFacts, FactsError> {
        let mut devices = self.observed_devices_until(deadline)?;
        devices.aec_capability = self.capability();
        if self.routes.resources.is_stopping() {
            return Err(FactsError::DiscoveryFailed);
        }
        let resources = self
            .routes
            .resources
            .inner
            .try_lock()
            .map_err(|_| FactsError::Busy)?;
        if self.routes.resources.is_stopping() {
            return Err(FactsError::DiscoveryFailed);
        }
        inspect_runtime_graph_facts(
            devices,
            resources
                .graph
                .as_ref()
                .ok_or(FactsError::DiscoveryFailed)?,
            &resources.routing,
            deadline,
        )
    }
}

impl<R: CommandRunner + Send + Sync> RuntimeMaintenance for AecProjectedRoutes<R> {
    fn confirm_headphones(
        &self,
        confirmation: Option<translator_audio::HeadphoneConfirmation>,
        deadline: std::time::Instant,
        store: &RuntimeStore,
    ) -> Result<(), translator_daemon::ControlFailure> {
        self.routes
            .confirm_headphones(confirmation, deadline, store)?;
        let mut devices = self.observed_devices_until(deadline).map_err(|_| {
            translator_daemon::ControlFailure {
                status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                code: "headphone_confirmation_failed",
            }
        })?;
        devices.aec_capability = self.capability();
        store.set_devices(devices.into());
        Ok(())
    }

    fn refresh(&self, store: &RuntimeStore) -> Result<(), translator_daemon::ControlFailure> {
        if !matches!(
            self.routes.operation_gate.state(),
            AudioOperationState::Idle | AudioOperationState::Production
        ) {
            return Err(native_route_failure());
        }
        self.routes
            .resources
            .with_active(|resources| resources.refresh_facts_only(store))
            .ok_or(native_route_failure())?;
        self.project(store);
        self.prepare_existing(&store.snapshot())
    }

    fn refresh_bypass_facts(
        &self,
        store: &RuntimeStore,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if self.routes.operation_gate.state() != AudioOperationState::Production {
            return Err(native_route_failure());
        }
        self.routes
            .resources
            .with_active(|resources| resources.refresh_facts_only(store))
            .ok_or(native_route_failure())?;
        self.project(store);
        Ok(())
    }

    fn verify_bypass_custody(
        &self,
        snapshot: &RuntimeSnapshot,
        permit_mic_original: bool,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if !native_pair_selected(snapshot) {
            return self
                .routes
                .verify_bypass_custody(snapshot, permit_mic_original);
        }
        if self.routes.operation_gate.state() != AudioOperationState::Production {
            return Err(native_route_failure());
        }
        self.routes
            .resources
            .with_active(|resources| {
                resources
                    .original_loopbacks
                    .verify_with_native_speaker(snapshot, permit_mic_original)
            })
            .ok_or(native_route_failure())?
            .map_err(|_| native_route_failure())
    }

    fn prepare_start(
        &self,
        candidate: &RuntimeSnapshot,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if native_pair_selected(candidate) {
            self.prepare_native(candidate)
        } else {
            self.routes.prepare_start(candidate)
        }
    }

    fn prepare_bypass(
        &self,
        snapshot: &RuntimeSnapshot,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if native_pair_selected(snapshot) {
            self.prepare_native(snapshot)
        } else {
            self.routes.prepare_bypass(snapshot)
        }
    }

    fn cleanup_originals(
        &self,
        deadline: Instant,
    ) -> Result<(), translator_daemon::ControlFailure> {
        self.routes.cleanup_originals(deadline)
    }
}

impl<R: CommandRunner + Send + Sync> ManualRouteController for AecProjectedRoutes<R> {
    fn reconcile(
        &self,
        stream_id: u32,
    ) -> Result<translator_audio::RoutingState, translator_audio::RoutingSafeError> {
        self.routes.reconcile(stream_id)
    }

    fn refresh_audio_state(&self, store: &RuntimeStore) {
        let _ = RuntimeMaintenance::refresh(self, store);
    }

    fn restore(&self) -> Result<(), translator_audio::RoutingSafeError> {
        self.routes.restore()
    }
}

struct PulseNativeAecEnvironment<R = SystemCommandRunner> {
    runner: R,
    facts_server: String,
    mix: Arc<dyn AudioMixController>,
    store: RuntimeStore,
}

#[derive(Deserialize, PartialEq)]
struct NativePulseServer {
    server_name: String,
    server_cookie: String,
    #[serde(default)]
    default_source_name: Option<String>,
    #[serde(default)]
    default_sink_name: Option<String>,
}

#[derive(Deserialize)]
struct NativePulseEndpoint {
    index: u32,
    name: String,
    #[serde(default)]
    properties: HashMap<String, String>,
    #[serde(default)]
    active_port: Option<String>,
    #[serde(default)]
    ports: Vec<NativePulsePort>,
}

#[derive(Deserialize)]
struct NativePulsePort {
    name: String,
    #[serde(rename = "type")]
    port_type: String,
    availability: String,
}

fn native_environment_error(code: &'static str) -> AecCalibrationEngineError {
    AecCalibrationEngineError {
        code,
        cleanup_confirmed: false,
    }
}

fn read_native_pulse_json<T: serde::de::DeserializeOwned>(
    runner: &impl CommandRunner,
    server: &str,
    args: &[&str],
    deadline: std::time::Instant,
) -> Result<T, AecCalibrationEngineError> {
    if std::time::Instant::now() >= deadline {
        return Err(native_environment_error("aec_pair_inspection_expired"));
    }
    if !matches!(
        args,
        ["--format=json", "info"] | ["--format=json", "list", "sources" | "sinks"]
    ) {
        return Err(native_environment_error("aec_pair_read_only_required"));
    }
    let mut command = vec![
        "LANG=C.UTF-8".to_owned(),
        "LC_ALL=C.UTF-8".to_owned(),
        "pactl".to_owned(),
        format!("--server={server}"),
    ];
    command.extend(args.iter().map(|value| (*value).to_owned()));
    let output = runner
        .run_until("env", &command, deadline)
        .map_err(|_| native_environment_error("aec_pair_unavailable"))?;
    if !output.is_success() || std::time::Instant::now() >= deadline {
        return Err(native_environment_error("aec_pair_unavailable"));
    }
    serde_json::from_slice(output.stdout())
        .map_err(|_| native_environment_error("aec_pair_provenance_unavailable"))
}

fn map_native_pulse_pair(
    server: NativePulseServer,
    sources: Vec<NativePulseEndpoint>,
    sinks: Vec<NativePulseEndpoint>,
) -> Result<(translator_audio::DeviceFacts, NativeAecPairFacts), AecCalibrationEngineError> {
    let invalid = || native_environment_error("aec_pair_provenance_unavailable");
    if server.server_name.trim().is_empty()
        || !server
            .server_cookie
            .split_once(':')
            .is_some_and(|(high, low)| {
                high.len() == 4
                    && low.len() == 4
                    && u16::from_str_radix(high, 16)
                        .ok()
                        .zip(u16::from_str_radix(low, 16).ok())
                        .is_some_and(|(high, low)| high != 0 || low != 0)
            })
    {
        return Err(invalid());
    }
    let endpoint = |raw: Vec<NativePulseEndpoint>, name: &str, port: &str, port_type: &str| {
        if raw.iter().filter(|value| value.name == name).count() != 1 {
            return Err(invalid());
        }
        let matched = raw
            .iter()
            .find(|value| value.name == name)
            .ok_or_else(invalid)?;
        if raw
            .iter()
            .filter(|value| value.index == matched.index)
            .count()
            != 1
        {
            return Err(invalid());
        }
        let property = |name: &str| matched.properties.get(name).map(String::as_str);
        if property("device.api") != Some("alsa")
            || property("alsa.card") != Some("0")
            || property("alsa.id") != Some("PCH")
            || property("alsa.name") != Some("ALC287 Analog")
            || property("alsa.card_name") != Some("HDA Intel PCH")
            || property("alsa.device") != Some("0")
            || property("api.alsa.pcm.card") != Some("0")
            || property("api.alsa.pcm.stream")
                != Some(if port_type == "mic" {
                    "capture"
                } else {
                    "playback"
                })
            || property("device.class") != Some("sound")
            || property("media.class")
                != Some(if port_type == "mic" {
                    "Audio/Source"
                } else {
                    "Audio/Sink"
                })
            || property("device.bus_path") != Some("pci-0000:00:1f.3")
            || ["node.virtual", "node.network"]
                .iter()
                .any(|name| property(name).is_some_and(|value| value.parse::<bool>() != Ok(false)))
            || matched.name.ends_with(".monitor")
            || matched.active_port.as_deref() != Some(port)
            || matched
                .ports
                .iter()
                .filter(|entry| {
                    entry.name == port
                        && entry.port_type.eq_ignore_ascii_case(port_type)
                        && ["available", "availability unknown"]
                            .iter()
                            .any(|availability| {
                                entry.availability.eq_ignore_ascii_case(availability)
                            })
                })
                .count()
                != 1
        {
            return Err(invalid());
        }
        raw.into_iter()
            .find(|value| value.name == name)
            .ok_or_else(invalid)
    };
    let source = endpoint(sources, NATIVE_SOURCE, "analog-input-internal-mic", "mic")?;
    let sink = endpoint(sinks, NATIVE_SINK, "analog-output-speaker", "speaker")?;
    let selection =
        |endpoint: &NativePulseEndpoint, port_type: &str| translator_audio::DeviceSelectionState {
            health: translator_audio::DeviceHealth::Available,
            selected: Some(translator_audio::PhysicalDevice {
                id: endpoint.index,
                name: endpoint.name.clone(),
                description: String::new(),
                active_port: endpoint.active_port.clone(),
                active_port_type: Some(port_type.to_owned()),
                available: true,
            }),
            pinned_name: Some(endpoint.name.clone()),
            current_default: None,
            pending_default: None,
        };
    let mut source_selection = selection(&source, "Mic");
    source_selection.current_default = server.default_source_name;
    source_selection.pending_default = source_selection
        .current_default
        .clone()
        .filter(|name| name != &source.name);
    let mut sink_selection = selection(&sink, "Speaker");
    sink_selection.current_default = server.default_sink_name;
    sink_selection.pending_default = sink_selection
        .current_default
        .clone()
        .filter(|name| name != &sink.name);
    let devices = translator_audio::DeviceFacts {
        source: source_selection,
        sink: sink_selection,
        output_mode: OutputMode::OpenSpeaker,
        aec_capability: AecCapability::Unavailable,
    };
    Ok((
        devices,
        NativeAecPairFacts {
            audio_server_id: format!("pulse:{}:{}", server.server_name, server.server_cookie),
            source_name: source.name,
            sink_name: sink.name,
            source_hardware_id: format!(
                "alsa:{}:{}:capture",
                source.properties.get("alsa.id").ok_or_else(invalid)?,
                source.properties.get("alsa.name").ok_or_else(invalid)?
            ),
            sink_hardware_id: format!(
                "alsa:{}:{}:playback",
                sink.properties.get("alsa.id").ok_or_else(invalid)?,
                sink.properties.get("alsa.name").ok_or_else(invalid)?
            ),
        },
    ))
}

fn validate_native_pair_identity(
    identity: &NativeAecIdentity,
) -> Result<(), AecCalibrationEngineError> {
    if identity.card != 0
        || identity.card_id != "PCH"
        || identity.pcm_name != "ALC287 Analog"
        || identity.source_port != "analog-input-internal-mic"
        || identity.sink_port != "analog-output-speaker"
        || !identity.graph.is_valid()
        || !matches!(identity.graph, translator_audio::AecGraphIdentity::Native { ref physical_device_id, .. } if physical_device_id == "alsa-hw:0,0:PCH:ALC287 Analog")
    {
        return Err(native_environment_error("aec_pair_provenance_unavailable"));
    }
    Ok(())
}

fn native_pair_selected(snapshot: &RuntimeSnapshot) -> bool {
    snapshot.devices.as_ref().is_some_and(|devices| {
        devices.acoustic.mode == OutputMode::OpenSpeaker
            && devices
                .source
                .selected
                .as_ref()
                .is_some_and(|source| source.name == NATIVE_SOURCE)
            && devices
                .sink
                .selected
                .as_ref()
                .is_some_and(|sink| sink.name == NATIVE_SINK)
    })
}

fn native_route_failure() -> translator_daemon::ControlFailure {
    translator_daemon::ControlFailure {
        status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
        code: "original_loopback_custody_unknown",
    }
}

fn native_facts_server(
    value: Option<&std::ffi::OsStr>,
) -> Result<String, AecCalibrationEngineError> {
    let server = value
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| native_environment_error("aec_facts_server_unavailable"))?;
    let path = server
        .strip_prefix("unix:")
        .map(std::path::Path::new)
        .ok_or_else(|| native_environment_error("aec_facts_server_invalid"))?;
    if !path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(native_environment_error("aec_facts_server_invalid"));
    }
    Ok(server.to_owned())
}

impl<R: CommandRunner + Send + Sync> NativeAecEnvironment for PulseNativeAecEnvironment<R> {
    fn inspect_pair(
        &self,
        identity: &NativeAecIdentity,
        deadline: std::time::Instant,
    ) -> Result<NativeAecPairFacts, AecCalibrationEngineError> {
        validate_native_pair_identity(identity)?;
        self.inspect_inventory(deadline).map(|(_, pair)| pair)
    }

    fn quarantine(&self) -> Result<(), AecCalibrationEngineError> {
        self.mix
            .reconcile_committed(TranslationMixMode::Quarantine {
                mic_original_expected: self.store.snapshot().audio_mix.microphone_original_percent
                    > 0,
            })
            .map_err(|_| native_environment_error("aec_quarantine_unconfirmed"))
    }
}

impl<R: CommandRunner> PulseNativeAecEnvironment<R> {
    fn inspect_devices(
        &self,
        deadline: std::time::Instant,
    ) -> Result<translator_audio::DeviceFacts, AecCalibrationEngineError> {
        self.inspect_inventory(deadline).map(|(devices, _)| devices)
    }

    fn inspect_inventory(
        &self,
        deadline: std::time::Instant,
    ) -> Result<(translator_audio::DeviceFacts, NativeAecPairFacts), AecCalibrationEngineError>
    {
        let before = read_native_pulse_json::<NativePulseServer>(
            &self.runner,
            &self.facts_server,
            &["--format=json", "info"],
            deadline,
        )?;
        let sources = read_native_pulse_json(
            &self.runner,
            &self.facts_server,
            &["--format=json", "list", "sources"],
            deadline,
        )?;
        let sinks = read_native_pulse_json(
            &self.runner,
            &self.facts_server,
            &["--format=json", "list", "sinks"],
            deadline,
        )?;
        let after = read_native_pulse_json::<NativePulseServer>(
            &self.runner,
            &self.facts_server,
            &["--format=json", "info"],
            deadline,
        )?;
        if before != after {
            return Err(native_environment_error("aec_audio_server_changed"));
        }
        let (devices, mut pair) = map_native_pulse_pair(after, sources, sinks)?;
        pair.audio_server_id = format!("{}:{}", self.facts_server, pair.audio_server_id);
        Ok((devices, pair))
    }
}

fn load_native_positive_fixture(
    path: Option<&std::path::Path>,
    sha256: Option<&str>,
) -> Result<Option<NativeAecPositiveFixture>, AecCalibrationEngineError> {
    match (path, sha256) {
        (None, None) => Ok(None),
        (Some(path), Some(sha256)) => NativeAecPositiveFixture::read(path, sha256).map(Some),
        _ => Err(native_environment_error(
            "aec_positive_configuration_incomplete",
        )),
    }
}

#[cfg(test)]
mod native_aec_composition_tests {
    use super::*;
    use serde_json::{Value, json};
    use translator_audio::{AecGraphIdentity, CommandRunError};

    fn endpoint(source: bool) -> Value {
        json!({
            "index": if source { 88 } else { 87 },
            "name": if source { NATIVE_SOURCE } else { NATIVE_SINK },
            "active_port": if source { "analog-input-internal-mic" } else { "analog-output-speaker" },
            "ports": [{"name": if source { "analog-input-internal-mic" } else { "analog-output-speaker" }, "type": if source { "Mic" } else { "Speaker" }, "availability": "availability unknown"}],
            "properties": {"device.api":"alsa", "media.class": if source { "Audio/Source" } else { "Audio/Sink" }, "device.class":"sound", "device.bus_path":"pci-0000:00:1f.3", "alsa.card":"0", "alsa.device":"0", "alsa.id":"PCH", "alsa.name":"ALC287 Analog", "alsa.card_name":"HDA Intel PCH", "api.alsa.pcm.card":"0", "api.alsa.pcm.stream": if source { "capture" } else { "playback" }}
        })
    }

    fn map(
        source: Value,
        sink: Value,
        cookie: &str,
    ) -> Result<(translator_audio::DeviceFacts, NativeAecPairFacts), AecCalibrationEngineError>
    {
        map_native_pulse_pair(
            NativePulseServer {
                server_name: "PulseAudio (on PipeWire 1.0.5)".into(),
                server_cookie: cookie.into(),
                default_source_name: Some("alsa_input.usb-preserved".into()),
                default_sink_name: Some("alsa_output.usb-preserved".into()),
            },
            vec![serde_json::from_value(source).unwrap()],
            vec![serde_json::from_value(sink).unwrap()],
        )
    }

    fn identity() -> NativeAecIdentity {
        NativeAecIdentity {
            graph: AecGraphIdentity::Native {
                session_id: 17,
                generation: 4,
                physical_device_id: "alsa-hw:0,0:PCH:ALC287 Analog".into(),
                dsp_config_id: "installed-spa".into(),
            },
            card: 0,
            card_id: "PCH".into(),
            pcm_name: "ALC287 Analog".into(),
            capture_channels: 2,
            playback_channels: 2,
            capture_buffer: 3840,
            playback_buffer: 3840,
            source_port: "analog-input-internal-mic".into(),
            sink_port: "analog-output-speaker".into(),
            control_fingerprint: "test-controls".into(),
            capture_gains: vec![58, 58],
            playback_gains: vec![40, 40],
            capture_muted: false,
            playback_muted: false,
            playback_volume_percent: 40,
            capture_origin_monotonic_ns: 1,
        }
    }

    #[test]
    fn native_actual_pch_schema_maps_real_ports_and_never_implies_aec_proof() {
        let (devices, pair) = map(endpoint(true), endpoint(false), "e647:e3e7").unwrap();
        assert_eq!(devices.source.selected.as_ref().unwrap().id, 88);
        assert_eq!(devices.sink.selected.as_ref().unwrap().id, 87);
        assert_eq!(devices.source.pinned_name.as_deref(), Some(NATIVE_SOURCE));
        assert_eq!(devices.sink.pinned_name.as_deref(), Some(NATIVE_SINK));
        assert_eq!(
            devices.source.current_default.as_deref(),
            Some("alsa_input.usb-preserved")
        );
        assert_eq!(
            devices.sink.pending_default.as_deref(),
            Some("alsa_output.usb-preserved")
        );
        assert_eq!(devices.aec_capability, AecCapability::Unavailable);
        assert!(
            !translator_daemon::DeviceState::from(devices)
                .acoustic
                .full_duplex_allowed
        );
        assert_eq!(pair.source_hardware_id, "alsa:PCH:ALC287 Analog:capture");
        assert!(pair.audio_server_id.ends_with("e647:e3e7"));
        validate_native_pair_identity(&identity()).unwrap();
        let mut incorrect = identity();
        if let AecGraphIdentity::Native {
            physical_device_id, ..
        } = &mut incorrect.graph
        {
            *physical_device_id = "alsa-hw:0,0".into();
        }
        assert!(validate_native_pair_identity(&incorrect).is_err());
    }

    #[test]
    fn native_wrong_card_stream_api_class_and_ports_are_rejected() {
        let mutations: [fn(&mut Value); 8] = [
            |value| value["properties"]["alsa.card"] = json!("1"),
            |value| value["properties"]["alsa.id"] = json!("USB"),
            |value| value["properties"]["alsa.name"] = json!("different PCM"),
            |value| value["properties"]["api.alsa.pcm.stream"] = json!("monitor"),
            |value| value["properties"]["device.api"] = Value::Null,
            |value| value["properties"]["device.class"] = json!("monitor"),
            |value| value["active_port"] = json!("analog-output-headphones"),
            |value| value["ports"][0]["availability"] = json!("not available"),
        ];
        for mutate in mutations {
            for source_fault in [true, false] {
                let mut source = endpoint(true);
                let mut sink = endpoint(false);
                mutate(if source_fault { &mut source } else { &mut sink });
                let parse = |value| serde_json::from_value::<NativePulseEndpoint>(value);
                let result = parse(source)
                    .ok()
                    .zip(parse(sink).ok())
                    .and_then(|(source, sink)| {
                        map_native_pulse_pair(
                            NativePulseServer {
                                server_name: "actual-server".into(),
                                server_cookie: "e647:e3e7".into(),
                                default_source_name: None,
                                default_sink_name: None,
                            },
                            vec![source],
                            vec![sink],
                        )
                        .ok()
                    });
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn native_monitor_alias_duplicate_id_and_missing_cookie_are_rejected() {
        let mut monitor = endpoint(true);
        monitor["name"] = json!(format!("{NATIVE_SINK}.monitor"));
        monitor["properties"]["device.class"] = json!("monitor");
        monitor["properties"]["api.alsa.pcm.stream"] = json!("playback");
        assert!(map(monitor, endpoint(false), "e647:e3e7").is_err());
        let source: NativePulseEndpoint = serde_json::from_value(endpoint(true)).unwrap();
        let mut alias = endpoint(true);
        alias["name"] = json!("alsa_input.alias");
        assert!(
            map_native_pulse_pair(
                NativePulseServer {
                    server_name: "server".into(),
                    server_cookie: "e647:e3e7".into(),
                    default_source_name: None,
                    default_sink_name: None,
                },
                vec![source, serde_json::from_value(alias).unwrap()],
                vec![serde_json::from_value(endpoint(false)).unwrap()]
            )
            .is_err()
        );
        for cookie in ["", "0000:0000", "not-a-cookie", "e647", "e647:00000"] {
            assert!(map(endpoint(true), endpoint(false), cookie).is_err());
        }
        assert!(
            serde_json::from_value::<NativePulseServer>(json!({"server_name":"server"})).is_err()
        );
    }

    #[test]
    fn native_configuration_has_no_implicit_host_server_or_fixture_fallback() {
        assert!(load_native_positive_fixture(None, None).unwrap().is_none());
        assert!(load_native_positive_fixture(None, Some("hash")).is_err());
        assert!(
            load_native_positive_fixture(Some(std::path::Path::new("missing-fixture")), None)
                .is_err()
        );
        assert!(native_facts_server(None).is_err());
        for server in [
            "tcp:localhost",
            "localhost",
            "unix:relative",
            "unix:/run/user/1000/../native",
        ] {
            assert!(native_facts_server(Some(std::ffi::OsStr::new(server))).is_err());
        }
        assert!(
            native_facts_server(Some(std::ffi::OsStr::new(
                "unix:/run/user/1000/pulse/native"
            )))
            .is_ok()
        );
        let mut devices = map(endpoint(true), endpoint(false), "e647:e3e7").unwrap().0;
        devices.output_mode = OutputMode::Headphones;
        let state = translator_daemon::DeviceState::from(devices);
        assert!(state.acoustic.full_duplex_allowed);
        assert_eq!(state.acoustic.aec_capability, AecCapability::Unavailable);
    }

    #[test]
    fn native_inventory_ignores_unrelated_unported_nodes_without_enriching_properties() {
        let sources = serde_json::from_value(json!([
            endpoint(true),
            {"index":19,"name":"translator_remote_in.monitor","active_port":null,"properties":{"device.class":"monitor"}},
            {"index":20,"name":"alsa_input.usb-preserved","active_port":null,"ports":[],"properties":{"device.api":"alsa"}}
        ])).unwrap();
        let server = serde_json::from_value(json!({"server_name":"actual-server","server_cookie":"e647:e3e7","default_source_name":"alsa_input.usb-preserved","default_sink_name":"alsa_output.usb-preserved"})).unwrap();
        let (devices, _) = map_native_pulse_pair(
            server,
            sources,
            vec![serde_json::from_value(endpoint(false)).unwrap()],
        )
        .unwrap();
        assert_eq!(devices.source.selected.unwrap().name, NATIVE_SOURCE);
        assert_eq!(
            devices.source.current_default.as_deref(),
            Some("alsa_input.usb-preserved")
        );
    }

    #[derive(Default)]
    struct ReadOnlyRunner {
        calls: Mutex<Vec<(String, Vec<String>, std::time::Instant)>>,
    }
    impl CommandRunner for ReadOnlyRunner {
        fn run_until(
            &self,
            program: &str,
            args: &[String],
            deadline: std::time::Instant,
        ) -> Result<CommandResult, CommandRunError> {
            self.calls
                .lock()
                .unwrap()
                .push((program.into(), args.to_vec(), deadline));
            Ok(CommandResult::success(b"{}".to_vec()))
        }
    }

    #[test]
    fn native_read_only_target_and_locale_are_explicit_and_deadline_preserved() {
        let runner = ReadOnlyRunner::default();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let _: Value = read_native_pulse_json(
            &runner,
            "unix:/run/user/1000/pulse/native",
            &["--format=json", "info"],
            deadline,
        )
        .unwrap();
        assert!(
            read_native_pulse_json::<Value>(
                &runner,
                "unix:/run/user/1000/pulse/native",
                &["set-default-sink", NATIVE_SINK],
                deadline
            )
            .is_err()
        );
        let calls = runner.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "env");
        assert_eq!(
            calls[0].1,
            [
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                "pactl",
                "--server=unix:/run/user/1000/pulse/native",
                "--format=json",
                "info"
            ]
        );
        assert_eq!(calls[0].2, deadline);
    }

    #[test]
    fn native_omits_only_owned_physical_speaker_route_not_headphone_pulse_custody() {
        let devices = map(endpoint(true), endpoint(false), "e647:e3e7").unwrap().0;
        let snapshot = RuntimeSnapshot {
            devices: Some(devices.clone().into()),
            ..RuntimeStore::default().snapshot()
        };
        assert!(
            original_loopback_requests(&snapshot)
                .iter()
                .any(|request| request.media_name == SPEAKER_ORIGINAL_LOOPBACK)
        );
        assert!(
            !pulse_requests_with_native_speaker(&snapshot)
                .iter()
                .any(|request| request.media_name == SPEAKER_ORIGINAL_LOOPBACK)
        );
        let mut headphones = devices;
        headphones.output_mode = OutputMode::Headphones;
        let snapshot = RuntimeSnapshot {
            devices: Some(headphones.into()),
            ..snapshot
        };
        assert_eq!(
            pulse_requests_with_native_speaker(&snapshot),
            original_loopback_requests(&snapshot)
        );
    }

    #[derive(Clone, Default)]
    struct InventoryRunner {
        calls: Arc<Mutex<Vec<Vec<String>>>>,
    }

    impl CommandRunner for InventoryRunner {
        fn run_until(
            &self,
            program: &str,
            args: &[String],
            _deadline: std::time::Instant,
        ) -> Result<CommandResult, CommandRunError> {
            assert_eq!(program, "env");
            assert_eq!(args[3], "--server=unix:/test/read-only-facts");
            self.calls.lock().unwrap().push(args.to_vec());
            let response = match args[4..]
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .as_slice()
            {
                ["--format=json", "info"] => {
                    json!({"server_name":"actual-server","server_cookie":"e647:e3e7","default_source_name":"alsa_input.usb-preserved","default_sink_name":"alsa_output.usb-preserved"})
                }
                ["--format=json", "list", "sources"] => json!([endpoint(true)]),
                ["--format=json", "list", "sinks"] => json!([endpoint(false)]),
                _ => panic!("projection attempted a non-inventory command"),
            };
            Ok(CommandResult::success(
                serde_json::to_vec(&response).unwrap(),
            ))
        }
    }

    #[derive(Clone, Default)]
    struct HeadphoneInventoryRunner {
        native: InventoryRunner,
        changed: Arc<AtomicBool>,
    }

    impl CommandRunner for HeadphoneInventoryRunner {
        fn run_until(
            &self,
            program: &str,
            args: &[String],
            deadline: std::time::Instant,
        ) -> Result<CommandResult, CommandRunError> {
            if program != "pactl" {
                return self.native.run_until(program, args, deadline);
            }
            let device = |source: bool| {
                json!({
                    "index": if source { 42 } else if self.changed.load(Ordering::Acquire) { 141 } else { 41 },
                    "name": if source { "alsa_input.usb-headset" } else { "alsa_output.usb-headset" },
                    "monitor_source": if source { "" } else { "alsa_output.usb-headset.monitor" },
                    "active_port": "analog",
                    "ports": [{"name":"analog", "type":"Analog", "availability":"available"}],
                    "properties": {"device.api":"alsa", "device.class":"sound", "media.class":if source { "Audio/Source" } else { "Audio/Sink" }},
                })
            };
            let response = match args
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .as_slice()
            {
                ["--format=json", "list", "sources"] => {
                    serde_json::to_vec(&json!([device(true)])).unwrap()
                }
                ["--format=json", "list", "sinks"] => {
                    serde_json::to_vec(&json!([device(false)])).unwrap()
                }
                ["get-default-source"] => b"alsa_input.usb-headset\n".to_vec(),
                ["get-default-sink"] => b"alsa_output.usb-headset\n".to_vec(),
                _ => panic!("confirmation attempted non-inventory IO"),
            };
            Ok(CommandResult::success(response))
        }
    }

    #[test]
    fn native_projection_delegates_current_confirmation_and_revocation_to_watcher() {
        let runner = HeadphoneInventoryRunner::default();
        let store = RuntimeStore::default();
        let gate = AudioOperationGate::new();
        let mut devices = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);
        let current = devices.reconcile(DeviceOverride::default()).unwrap();
        let confirmation = translator_audio::HeadphoneConfirmation {
            source: current.source.selected.unwrap(),
            sink: current.sink.selected.unwrap(),
        };
        let routes = Arc::new(PulseManualRoutes {
            resources: LifecycleProtected::new(PulseResources {
                routing: PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production),
                devices,
                original_loopbacks: PulseOriginalLoopbacks::new(runner.clone()),
                graph: None,
            }),
            operation_gate: gate.clone(),
        });
        let projection = AecProjectedRoutes {
            routes: routes.clone(),
            coordinator: Arc::new(AecCalibrationCoordinator::new()),
            environment: Arc::new(PulseNativeAecEnvironment {
                runner: runner.clone(),
                facts_server: "unix:/test/read-only-facts".into(),
                mix: Arc::new(AudioMixApplication::new(runner.clone())),
                store: store.clone(),
            }),
        };
        let deadline = || std::time::Instant::now() + std::time::Duration::from_secs(1);
        let calibration = gate.acquire_calibration(uuid::Uuid::new_v4()).unwrap();
        assert_eq!(
            projection
                .confirm_headphones(Some(confirmation.clone()), deadline(), &store)
                .unwrap_err()
                .code,
            "audio_operation_busy"
        );
        assert!(store.snapshot().devices.is_none());
        drop(calibration);
        projection
            .confirm_headphones(Some(confirmation), deadline(), &store)
            .unwrap();
        projection.project(&store);
        let observed = projection.observed_devices_until(deadline()).unwrap();
        assert_eq!(observed.output_mode, OutputMode::UserConfirmedHeadphones);
        assert_eq!(
            observed.sink.selected.unwrap().active_port_type.as_deref(),
            Some("Analog")
        );
        assert_eq!(
            store.snapshot().devices.unwrap().acoustic.mode,
            OutputMode::UserConfirmedHeadphones
        );
        assert!(!store.snapshot().translation_running);
        assert!(
            runner.native.calls.lock().unwrap().is_empty(),
            "confirmed headphones must not become the built-in native AEC pair"
        );

        runner.changed.store(true, Ordering::Release);
        assert_ne!(
            projection
                .observed_devices_until(deadline())
                .unwrap()
                .output_mode,
            OutputMode::UserConfirmedHeadphones
        );
        runner.changed.store(false, Ordering::Release);
        assert_ne!(
            projection
                .observed_devices_until(deadline())
                .unwrap()
                .output_mode,
            OutputMode::UserConfirmedHeadphones,
            "old pair must not revive confirmation"
        );
    }

    #[tokio::test]
    async fn native_watcher_projects_read_only_pair_during_exclusive_calibration() {
        let runner = InventoryRunner::default();
        let store = RuntimeStore::default();
        let gate = AudioOperationGate::new();
        let _lease = gate.acquire_calibration(uuid::Uuid::new_v4()).unwrap();
        let projection = Arc::new(AecProjectedRoutes {
            routes: Arc::new(PulseManualRoutes {
                resources: LifecycleProtected::new(PulseResources {
                    routing: PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production),
                    devices: PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable),
                    original_loopbacks: PulseOriginalLoopbacks::new(runner.clone()),
                    graph: None,
                }),
                operation_gate: gate.clone(),
            }),
            coordinator: Arc::new(AecCalibrationCoordinator::new()),
            environment: Arc::new(PulseNativeAecEnvironment {
                runner: runner.clone(),
                facts_server: "unix:/test/read-only-facts".into(),
                mix: Arc::new(AudioMixApplication::new(runner.clone())),
                store: store.clone(),
            }),
        });
        assert!(RuntimeMaintenance::refresh(projection.as_ref(), &store).is_err());
        let task = tokio::spawn(watcher_loop(None, store.clone(), Some(projection.clone())));
        let projected = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if store.snapshot().devices.is_some() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await;
        task.abort();
        let _ = task.await;
        projected.unwrap();
        let snapshot = store.snapshot();
        let devices = snapshot.devices.as_ref().unwrap();
        assert_eq!(
            devices.source.selected.as_ref().unwrap().name,
            NATIVE_SOURCE
        );
        assert_eq!(devices.acoustic.aec_capability, AecCapability::Unavailable);
        assert!(!devices.acoustic.full_duplex_allowed);
        assert!(matches!(
            gate.state(),
            AudioOperationState::Calibration { .. }
        ));
        assert!(projection.verify_bypass_custody(&snapshot, false).is_err());
        assert_eq!(runner.calls.lock().unwrap().len(), 4);
    }
}

impl<R: CommandRunner + Send> RuntimeMaintenance for PulseManualRoutes<R> {
    fn confirm_headphones(
        &self,
        confirmation: Option<translator_audio::HeadphoneConfirmation>,
        deadline: std::time::Instant,
        store: &RuntimeStore,
    ) -> Result<(), translator_daemon::ControlFailure> {
        let _lease = self.operation_gate.acquire_manual().map_err(|_| {
            translator_daemon::ControlFailure {
                status: axum::http::StatusCode::CONFLICT,
                code: "audio_operation_busy",
            }
        })?;
        if self.resources.is_stopping() {
            return Err(native_route_failure());
        }
        let mut resources =
            self.resources
                .inner
                .try_lock()
                .map_err(|_| translator_daemon::ControlFailure {
                    status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    code: "audio_facts_busy",
                })?;
        if self.resources.is_stopping() {
            return Err(native_route_failure());
        }
        let facts = resources
            .devices
            .confirm_headphones_until(confirmation, deadline)
            .map_err(|error| translator_daemon::ControlFailure {
                status: if error.code()
                    == translator_audio::DeviceWatcherErrorCode::InvalidPhysicalDevice
                {
                    axum::http::StatusCode::CONFLICT
                } else {
                    axum::http::StatusCode::SERVICE_UNAVAILABLE
                },
                code: "headphone_confirmation_failed",
            })?;
        store.set_devices(facts.into());
        Ok(())
    }

    fn refresh(&self, store: &RuntimeStore) -> Result<(), translator_daemon::ControlFailure> {
        if !matches!(
            self.operation_gate.state(),
            AudioOperationState::Idle | AudioOperationState::Production
        ) {
            return Err(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::CONFLICT,
                code: "audio_operation_busy",
            });
        }
        self.resources
            .with_active(|resources| resources.refresh(store))
            .ok_or(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                code: "original_loopback_custody_unknown",
            })?
            .map_err(|_| translator_daemon::ControlFailure {
                status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                code: "original_loopback_custody_unknown",
            })
    }

    fn refresh_bypass_facts(
        &self,
        store: &RuntimeStore,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if self.operation_gate.state() != AudioOperationState::Production {
            return Err(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::CONFLICT,
                code: "audio_operation_busy",
            });
        }
        self.resources
            .with_active(|resources| resources.refresh_facts_only(store))
            .ok_or(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                code: "original_loopback_custody_unknown",
            })
    }

    fn verify_bypass_custody(
        &self,
        snapshot: &RuntimeSnapshot,
        permit_mic_original: bool,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if self.operation_gate.state() != AudioOperationState::Production {
            return Err(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::CONFLICT,
                code: "audio_operation_busy",
            });
        }
        self.resources
            .with_active(|resources| {
                resources
                    .original_loopbacks
                    .verify_existing(snapshot, permit_mic_original)
            })
            .ok_or(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                code: "original_loopback_custody_unknown",
            })?
            .map_err(|_| translator_daemon::ControlFailure {
                status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                code: "original_loopback_custody_unknown",
            })
    }

    fn prepare_start(
        &self,
        candidate: &RuntimeSnapshot,
    ) -> Result<(), translator_daemon::ControlFailure> {
        self.resources
            .with_active(|resources| resources.original_loopbacks.prepare_for_start(candidate))
            .ok_or(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                code: "original_loopback_custody_unknown",
            })?
            .map_err(|error| {
                tracing::warn!(event = "original_loopback_start_preparation_failed", code = ?error.code());
                translator_daemon::ControlFailure {
                    status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    code: "original_loopback_custody_unknown",
                }
            })
    }

    fn prepare_bypass(
        &self,
        snapshot: &RuntimeSnapshot,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if self.operation_gate.state() != AudioOperationState::Production {
            return Err(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::CONFLICT,
                code: "audio_operation_busy",
            });
        }
        self.resources
            .with_active(|resources| {
                resources
                    .original_loopbacks
                    .ensure_without_new_mic(snapshot)
                    .or_else(|_| {
                        let mut muted = snapshot.clone();
                        muted.audio_mix.microphone_original_percent = 0;
                        resources.original_loopbacks.ensure_without_new_mic(&muted)
                    })
            })
            .ok_or(translator_daemon::ControlFailure {
                status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                code: "original_loopback_custody_unknown",
            })?
            .map_err(|error| {
                tracing::warn!(event = "original_loopback_bypass_preparation_failed", code = ?error.code());
                translator_daemon::ControlFailure {
                    status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    code: "original_loopback_custody_unknown",
                }
            })
    }

    fn cleanup_originals(
        &self,
        deadline: Instant,
    ) -> Result<(), translator_daemon::ControlFailure> {
        if !matches!(
            self.operation_gate.state(),
            AudioOperationState::Production | AudioOperationState::Stopping
        ) {
            return Err(native_route_failure());
        }
        self.resources
            .with_active(|resources| resources.original_loopbacks.cleanup_all_until(deadline))
            .ok_or(native_route_failure())?
            .map(|_| ())
            .map_err(|_| native_route_failure())
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
    inspect_runtime_graph_facts(devices, graph, routing, deadline)
}

fn inspect_runtime_graph_facts(
    devices: translator_audio::DeviceFacts,
    graph: &impl AudioGraph,
    routing: &impl RoutingWatcher,
    deadline: std::time::Instant,
) -> Result<RuntimeFacts, FactsError> {
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
                let _ = resources.refresh(store);
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
    sink: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiscoveredOriginalLoopback {
    media_name: &'static str,
    source_index: Option<u32>,
    sink_index: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawPulseModuleId {
    Text(String),
    Number(u32),
}

#[derive(Debug, Deserialize)]
struct RawPulseStream {
    owner_module: Option<RawPulseModuleId>,
    source: Option<u32>,
    sink: Option<u32>,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawPulseEndpoint {
    index: u32,
    name: String,
}

struct PulseOriginalLoopbacks<R = SystemCommandRunner> {
    runner: R,
    microphone: Option<Mutex<translator_audio::PulseOriginalMicrophone>>,
}

impl<R> PulseOriginalLoopbacks<R>
where
    R: CommandRunner,
{
    #[cfg(test)]
    const fn new(runner: R) -> Self {
        Self {
            runner,
            microphone: None,
        }
    }

    fn with_microphone(runner: R, registry: translator_audio::OriginalMicrophoneRegistry) -> Self {
        Self {
            runner,
            microphone: Some(Mutex::new(translator_audio::PulseOriginalMicrophone::new(
                registry,
            ))),
        }
    }

    #[cfg(test)]
    fn ensure(&self, snapshot: &RuntimeSnapshot) -> Result<(), OriginalLoopbackError> {
        self.ensure_with_policy(snapshot, true)
    }

    fn ensure_without_new_mic(
        &self,
        snapshot: &RuntimeSnapshot,
    ) -> Result<(), OriginalLoopbackError> {
        self.ensure_with_policy(snapshot, false)
    }

    fn ensure_with_policy(
        &self,
        snapshot: &RuntimeSnapshot,
        allow_new_mic: bool,
    ) -> Result<(), OriginalLoopbackError> {
        self.ensure_requests(
            snapshot,
            allow_new_mic,
            false,
            original_loopback_requests(snapshot),
        )
    }

    fn prepare_for_start(&self, snapshot: &RuntimeSnapshot) -> Result<(), OriginalLoopbackError> {
        self.ensure_requests(snapshot, false, true, original_loopback_requests(snapshot))
    }

    fn ensure_with_native_speaker(
        &self,
        snapshot: &RuntimeSnapshot,
    ) -> Result<(), OriginalLoopbackError> {
        self.ensure_requests(
            snapshot,
            false,
            false,
            pulse_requests_with_native_speaker(snapshot),
        )
    }

    fn ensure_requests(
        &self,
        snapshot: &RuntimeSnapshot,
        allow_new_mic: bool,
        allow_native_mic: bool,
        mut requests: Vec<OriginalLoopbackRequest>,
    ) -> Result<(), OriginalLoopbackError> {
        let native_request = self.microphone.as_ref().and_then(|_| {
            requests
                .iter()
                .find(|request| request.media_name == MICROPHONE_ORIGINAL_LOOPBACK)
                .cloned()
        });
        requests.retain(|request| {
            request.media_name != MICROPHONE_ORIGINAL_LOOPBACK
                || (self.microphone.is_none() && snapshot.audio_mix.microphone_original_percent > 0)
        });
        if let Some(microphone) = &self.microphone {
            let mut microphone = microphone.lock().map_err(|_| discovery_error())?;
            if native_request.is_none() {
                microphone
                    .stop(Instant::now() + Duration::from_secs(1))
                    .map_err(|_| OriginalLoopbackError::new(OriginalLoopbackErrorCode::Cleanup))?;
            }
        }
        let sink_inputs: Vec<RawPulseStream> =
            self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let source_outputs: Vec<RawPulseStream> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let discovered = discover_original_loopbacks(&sink_inputs, &source_outputs)?;
        let (sources, sinks) = if requests.is_empty() && native_request.is_none() {
            (HashMap::new(), HashMap::new())
        } else {
            let sources = endpoint_names(self.run_json(&["--format=json", "list", "sources"])?)?;
            let sinks = endpoint_names(self.run_json(&["--format=json", "list", "sinks"])?)?;
            (sources, sinks)
        };
        let native_pair = native_request
            .as_ref()
            .map(|request| native_microphone_pair(snapshot, request, &sources, &sinks))
            .transpose();
        let native_pair = match native_pair {
            Ok(pair) => pair,
            Err(error) => {
                if let Some(microphone) = &self.microphone {
                    microphone
                        .lock()
                        .map_err(|_| discovery_error())?
                        .stop(Instant::now() + Duration::from_secs(1))
                        .map_err(|_| {
                            OriginalLoopbackError::new(OriginalLoopbackErrorCode::Cleanup)
                        })?;
                }
                return Err(error);
            }
        };
        if let (Some(microphone), Some(request), Some((source, sink))) =
            (&self.microphone, &native_request, native_pair)
        {
            let mut microphone = microphone.lock().map_err(|_| discovery_error())?;
            if microphone.verify(&request.source, source, sink).is_err() {
                microphone
                    .stop(Instant::now() + Duration::from_secs(1))
                    .map_err(|_| OriginalLoopbackError::new(OriginalLoopbackErrorCode::Cleanup))?;
            }
        }
        if requests.iter().any(|request| {
            !sources.values().any(|name| name == &request.source)
                || !sinks.values().any(|name| name == &request.sink)
        }) {
            return Err(OriginalLoopbackError::new(
                OriginalLoopbackErrorCode::Discovery,
            ));
        }
        let mut keep_module_ids = HashSet::new();
        let mut expected = HashMap::new();
        let mut missing_requests = Vec::new();

        for request in &requests {
            let mut matching_module_ids =
                matching_original_loopbacks(&discovered, request, &sources, &sinks)?;
            matching_module_ids.sort();
            if let Some(module_id) = matching_module_ids.first() {
                keep_module_ids.insert(module_id.clone());
                expected.insert(module_id.clone(), request.clone());
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

        if snapshot.translation_running
            && missing_requests
                .iter()
                .any(|request| request.media_name == SPEAKER_ORIGINAL_LOOPBACK)
        {
            return Err(discovery_error());
        }
        if !allow_new_mic
            && missing_requests
                .iter()
                .any(|request| request.media_name == MICROPHONE_ORIGINAL_LOOPBACK)
        {
            return Err(discovery_error());
        }

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut fresh_module_ids = HashSet::new();
        for request in missing_requests {
            let module_id = self.load_module(&request, deadline)?;
            if expected.insert(module_id.clone(), request).is_some()
                || !fresh_module_ids.insert(module_id)
            {
                return Err(discovery_error());
            }
        }
        self.verify_loaded_requests(&expected, &fresh_module_ids, &sources, &sinks, deadline)?;

        if let (Some(microphone), Some(request), Some((source, sink))) =
            (&self.microphone, &native_request, native_pair)
        {
            let mut microphone = microphone.lock().map_err(|_| discovery_error())?;
            if allow_native_mic {
                microphone.prepare(&request.source, source, sink, Instant::now() + Duration::from_secs(2))
                    .map_err(|error| {
                        tracing::warn!(event = "original_microphone_preparation_failed", code = ?error);
                        OriginalLoopbackError::new(OriginalLoopbackErrorCode::Load)
                    })?;
            } else if snapshot.translation_running
                || snapshot.audio_mix.microphone_original_percent > 0
            {
                microphone
                    .verify(&request.source, source, sink)
                    .map_err(|_| discovery_error())?;
            }
        }

        Ok(())
    }

    fn verify_loaded_requests(
        &self,
        expected: &HashMap<String, OriginalLoopbackRequest>,
        fresh_module_ids: &HashSet<String>,
        sources: &HashMap<u32, String>,
        sinks: &HashMap<u32, String>,
        deadline: Instant,
    ) -> Result<(), OriginalLoopbackError> {
        loop {
            if Instant::now() >= deadline {
                return Err(discovery_error());
            }
            let sink_inputs: Vec<RawPulseStream> =
                self.run_json_until(&["--format=json", "list", "sink-inputs"], deadline)?;
            let source_outputs: Vec<RawPulseStream> =
                self.run_json_until(&["--format=json", "list", "source-outputs"], deadline)?;
            let verified = discover_original_loopbacks(&sink_inputs, &source_outputs)?;
            if verified.len() != expected.len() {
                return Err(discovery_error());
            }
            let mut ready = true;
            for (module_id, route) in &verified {
                let request = expected.get(module_id).ok_or_else(discovery_error)?;
                if route.media_name != request.media_name {
                    return Err(discovery_error());
                }
                for (index, names, target) in [
                    (route.source_index, sources, &request.source),
                    (route.sink_index, sinks, &request.sink),
                ] {
                    match index {
                        // A new, otherwise complete owned pair may await policy links.
                        Some(u32::MAX) if fresh_module_ids.contains(module_id) => ready = false,
                        Some(index) if names.get(&index) == Some(target) => {}
                        _ => return Err(discovery_error()),
                    }
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(discovery_error());
            }
            if ready {
                return Ok(());
            }
            std::thread::sleep((deadline - now).min(Duration::from_millis(10)));
        }
    }

    fn verify_existing(
        &self,
        snapshot: &RuntimeSnapshot,
        permit_mic_original: bool,
    ) -> Result<(), OriginalLoopbackError> {
        self.verify_requests(
            snapshot,
            permit_mic_original,
            original_loopback_requests(snapshot),
        )
    }

    fn verify_with_native_speaker(
        &self,
        snapshot: &RuntimeSnapshot,
        permit_mic_original: bool,
    ) -> Result<(), OriginalLoopbackError> {
        self.verify_requests(
            snapshot,
            permit_mic_original,
            pulse_requests_with_native_speaker(snapshot),
        )
    }

    fn verify_requests(
        &self,
        snapshot: &RuntimeSnapshot,
        permit_mic_original: bool,
        mut requests: Vec<OriginalLoopbackRequest>,
    ) -> Result<(), OriginalLoopbackError> {
        if permit_mic_original
            && (!direction_enabled(snapshot, translator_core::AudioDirection::Microphone)
                || snapshot.devices.as_ref().is_none_or(|devices| {
                    !devices.acoustic.mode.is_headphones() || !devices.acoustic.full_duplex_allowed
                })
                || !requests
                    .iter()
                    .any(|request| request.media_name == MICROPHONE_ORIGINAL_LOOPBACK))
        {
            return Err(discovery_error());
        }
        let native_request = self
            .microphone
            .as_ref()
            .filter(|_| permit_mic_original)
            .and_then(|_| {
                requests
                    .iter()
                    .find(|request| request.media_name == MICROPHONE_ORIGINAL_LOOPBACK)
                    .cloned()
            });
        if !permit_mic_original {
            requests.retain(|request| request.media_name != MICROPHONE_ORIGINAL_LOOPBACK);
        }
        if self.microphone.is_some() {
            requests.retain(|request| request.media_name != MICROPHONE_ORIGINAL_LOOPBACK);
        }
        let sink_inputs: Vec<RawPulseStream> =
            self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let source_outputs: Vec<RawPulseStream> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let discovered = discover_original_loopbacks(&sink_inputs, &source_outputs)?;
        if discovered.len() != requests.len() {
            return Err(discovery_error());
        }
        if requests.is_empty() && native_request.is_none() {
            return Ok(());
        }
        let sources = endpoint_names(self.run_json(&["--format=json", "list", "sources"])?)?;
        let sinks = endpoint_names(self.run_json(&["--format=json", "list", "sinks"])?)?;
        if requests.iter().any(|request| {
            !matching_original_loopbacks(&discovered, request, &sources, &sinks)
                .is_ok_and(|matches| matches.len() == 1)
        }) {
            return Err(discovery_error());
        }
        if let (Some(microphone), Some(request)) = (&self.microphone, &native_request) {
            let (source, sink) = native_microphone_pair(snapshot, request, &sources, &sinks)?;
            microphone
                .lock()
                .map_err(|_| discovery_error())?
                .verify(&request.source, source, sink)
                .map_err(|_| discovery_error())?;
        }
        Ok(())
    }

    fn cleanup_all(&self) -> Result<Vec<String>, OriginalLoopbackError> {
        self.cleanup_all_until(Instant::now() + Duration::from_secs(2))
    }

    fn cleanup_all_until(&self, deadline: Instant) -> Result<Vec<String>, OriginalLoopbackError> {
        if let Some(microphone) = &self.microphone {
            microphone
                .lock()
                .map_err(|_| discovery_error())?
                .stop(deadline)
                .map_err(|_| OriginalLoopbackError::new(OriginalLoopbackErrorCode::Cleanup))?;
        }
        let sink_inputs: Vec<RawPulseStream> =
            self.run_json_until(&["--format=json", "list", "sink-inputs"], deadline)?;
        let source_outputs: Vec<RawPulseStream> =
            self.run_json_until(&["--format=json", "list", "source-outputs"], deadline)?;
        let discovered = discover_original_loopbacks(&sink_inputs, &source_outputs)?;
        let mut module_ids: Vec<_> = discovered.keys().cloned().collect();
        module_ids.sort();
        for module_id in &module_ids {
            self.unload_module_until(module_id, deadline)?;
        }
        Ok(module_ids)
    }

    fn load_module(
        &self,
        request: &OriginalLoopbackRequest,
        deadline: Instant,
    ) -> Result<String, OriginalLoopbackError> {
        let args = original_loopback_load_args(request);
        let result =
            self.run_pactl_owned_until(&args, OriginalLoopbackErrorCode::Load, deadline)?;
        let id = std::str::from_utf8(result.stdout())
            .map_err(|_| OriginalLoopbackError::new(OriginalLoopbackErrorCode::Load))?
            .trim();
        if !id
            .parse::<u32>()
            .is_ok_and(|value| value != u32::MAX && value.to_string() == id)
        {
            return Err(OriginalLoopbackError::new(OriginalLoopbackErrorCode::Load));
        }
        Ok(id.to_owned())
    }

    fn unload_module(&self, module_id: &str) -> Result<(), OriginalLoopbackError> {
        self.unload_module_until(module_id, Instant::now() + Duration::from_secs(2))
    }

    fn unload_module_until(
        &self,
        module_id: &str,
        deadline: Instant,
    ) -> Result<(), OriginalLoopbackError> {
        self.run_pactl_owned_until(
            &["unload-module".to_owned(), module_id.to_owned()],
            OriginalLoopbackErrorCode::Cleanup,
            deadline,
        )?;
        let module_id = module_id
            .parse::<u32>()
            .map_err(|_| OriginalLoopbackError::new(OriginalLoopbackErrorCode::Cleanup))?;
        let result = self.run_pactl_owned_until(
            &["list".to_owned(), "short".to_owned(), "modules".to_owned()],
            OriginalLoopbackErrorCode::Cleanup,
            deadline,
        )?;
        let present = translator_audio::module_id_present(result.stdout(), module_id)
            .map_err(|_| OriginalLoopbackError::new(OriginalLoopbackErrorCode::Cleanup))?;
        if present {
            return Err(OriginalLoopbackError::new(
                OriginalLoopbackErrorCode::Cleanup,
            ));
        }
        Ok(())
    }

    fn run_json<T>(&self, args: &[&str]) -> Result<T, OriginalLoopbackError>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.run_json_until(args, Instant::now() + Duration::from_secs(2))
    }

    fn run_json_until<T>(
        &self,
        args: &[&str],
        deadline: Instant,
    ) -> Result<T, OriginalLoopbackError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let args = args
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>();
        let result =
            self.run_pactl_owned_until(&args, OriginalLoopbackErrorCode::Discovery, deadline)?;
        serde_json::from_slice(result.stdout())
            .map_err(|_| OriginalLoopbackError::new(OriginalLoopbackErrorCode::Discovery))
    }

    fn run_pactl_owned_until(
        &self,
        args: &[String],
        failure_code: OriginalLoopbackErrorCode,
        deadline: Instant,
    ) -> Result<CommandResult, OriginalLoopbackError> {
        let result = self
            .runner
            .run_until("pactl", args, deadline)
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
    if direction_enabled(snapshot, translator_core::AudioDirection::Speaker)
        && let Some(sink) = devices.sink.selected.as_ref()
    {
        requests.push(OriginalLoopbackRequest {
            media_name: SPEAKER_ORIGINAL_LOOPBACK,
            source: format!("{REMOTE_IN_SINK}.monitor"),
            sink: sink.name.clone(),
        });
    }

    if direction_enabled(snapshot, translator_core::AudioDirection::Microphone)
        && devices.acoustic.mode.is_headphones()
        && devices.acoustic.full_duplex_allowed
        && let Some(source) = devices.source.selected.as_ref()
    {
        requests.push(OriginalLoopbackRequest {
            media_name: MICROPHONE_ORIGINAL_LOOPBACK,
            source: source.name.clone(),
            sink: MIC_OUT_SINK.to_owned(),
        });
    }

    requests
}

fn native_microphone_pair(
    snapshot: &RuntimeSnapshot,
    request: &OriginalLoopbackRequest,
    sources: &HashMap<u32, String>,
    sinks: &HashMap<u32, String>,
) -> Result<(u32, u32), OriginalLoopbackError> {
    let devices = snapshot.devices.as_ref().ok_or_else(discovery_error)?;
    let source = devices
        .source
        .selected
        .as_ref()
        .ok_or_else(discovery_error)?;
    if !source.available
        || devices.source.health != translator_audio::DeviceHealth::Available
        || devices.source.pinned_name.as_deref() != Some(source.name.as_str())
        || source.name != request.source
        || sources.get(&source.id) != Some(&source.name)
    {
        return Err(discovery_error());
    }
    let mut matching = sinks.iter().filter(|(_, name)| *name == MIC_OUT_SINK);
    let (&sink, _) = matching.next().ok_or_else(discovery_error)?;
    if matching.next().is_some() {
        return Err(discovery_error());
    }
    Ok((source.id, sink))
}

fn pulse_requests_with_native_speaker(snapshot: &RuntimeSnapshot) -> Vec<OriginalLoopbackRequest> {
    let mut requests = original_loopback_requests(snapshot);
    if native_pair_selected(snapshot) {
        // The retained native worker owns this DAC, including Stop bypass.
        requests.retain(|request| request.media_name != SPEAKER_ORIGINAL_LOOPBACK);
    }
    requests
}

fn direction_enabled(
    snapshot: &RuntimeSnapshot,
    direction_id: translator_core::AudioDirection,
) -> bool {
    snapshot
        .directions
        .iter()
        .any(|direction| direction.direction_id == direction_id && direction.enabled)
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
            "source_output_properties='media.name={} translator.owner=true'",
            request.media_name
        ),
        format!(
            "sink_input_properties='media.name={} translator.owner=true'",
            request.media_name
        ),
    ]
}

fn discover_original_loopbacks(
    sink_inputs: &[RawPulseStream],
    source_outputs: &[RawPulseStream],
) -> Result<HashMap<String, DiscoveredOriginalLoopback>, OriginalLoopbackError> {
    let mut modules = HashMap::new();
    let mut owned_sink_inputs = HashSet::new();
    let mut owned_source_outputs = HashSet::new();
    for input in sink_inputs {
        let Some(media_name) = original_media_name(&input.properties) else {
            continue;
        };
        let module_id = stream_module_id(input)?;
        if !owned_sink_inputs.insert((module_id.clone(), media_name)) {
            return Err(discovery_error());
        }
        let module = discovered_loopback_entry(&mut modules, &module_id, media_name)?;
        module.sink_index = Some(input.sink.ok_or_else(discovery_error)?);
    }

    for output in source_outputs {
        let Some(media_name) = original_media_name(&output.properties) else {
            continue;
        };
        let module_id = stream_module_id(output)?;
        if !owned_source_outputs.insert((module_id.clone(), media_name)) {
            return Err(discovery_error());
        }
        let module = discovered_loopback_entry(&mut modules, &module_id, media_name)?;
        module.source_index = Some(output.source.ok_or_else(discovery_error)?);
    }
    if owned_sink_inputs != owned_source_outputs {
        return Err(discovery_error());
    }
    Ok(modules)
}

fn discovered_loopback_entry<'a>(
    modules: &'a mut HashMap<String, DiscoveredOriginalLoopback>,
    module_id: &str,
    media_name: &'static str,
) -> Result<&'a mut DiscoveredOriginalLoopback, OriginalLoopbackError> {
    let module =
        modules
            .entry(module_id.to_owned())
            .or_insert_with(|| DiscoveredOriginalLoopback {
                media_name,
                source_index: None,
                sink_index: None,
            });
    if module.media_name != media_name {
        return Err(discovery_error());
    }
    Ok(module)
}

fn matching_original_loopbacks(
    discovered: &HashMap<String, DiscoveredOriginalLoopback>,
    request: &OriginalLoopbackRequest,
    sources: &HashMap<u32, String>,
    sinks: &HashMap<u32, String>,
) -> Result<Vec<String>, OriginalLoopbackError> {
    let mut matching = Vec::new();
    for (module_id, loopback) in discovered {
        if loopback.media_name != request.media_name {
            continue;
        }
        let source = loopback
            .source_index
            .and_then(|index| sources.get(&index))
            .ok_or_else(discovery_error)?;
        let sink = loopback
            .sink_index
            .and_then(|index| sinks.get(&index))
            .ok_or_else(discovery_error)?;
        if source == &request.source && sink == &request.sink {
            matching.push(module_id.clone());
        }
    }
    Ok(matching)
}

fn stream_module_id(stream: &RawPulseStream) -> Result<String, OriginalLoopbackError> {
    match stream.owner_module.as_ref().ok_or_else(discovery_error)? {
        RawPulseModuleId::Number(id) => Ok(id.to_string()),
        RawPulseModuleId::Text(id)
            if id
                .parse::<u32>()
                .is_ok_and(|parsed| parsed.to_string() == *id) =>
        {
            Ok(id.clone())
        }
        RawPulseModuleId::Text(_) => Err(discovery_error()),
    }
}

fn endpoint_names(
    endpoints: Vec<RawPulseEndpoint>,
) -> Result<HashMap<u32, String>, OriginalLoopbackError> {
    let mut names = HashMap::new();
    let mut seen_names = HashSet::new();
    for endpoint in endpoints {
        if endpoint.name.is_empty()
            || !seen_names.insert(endpoint.name.clone())
            || names.insert(endpoint.index, endpoint.name).is_some()
        {
            return Err(discovery_error());
        }
    }
    Ok(names)
}

fn discovery_error() -> OriginalLoopbackError {
    OriginalLoopbackError::new(OriginalLoopbackErrorCode::Discovery)
}

fn original_media_name(properties: &HashMap<String, String>) -> Option<&'static str> {
    if property(properties, "translator.owner") != Some("true") {
        return None;
    }
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
    let original_microphone_registry = translator_audio::OriginalMicrophoneRegistry::default();
    let manual_routes = Arc::new(PulseManualRoutes {
        resources: LifecycleProtected::new(PulseResources {
            routing: build_routing_watcher(),
            devices: device_watcher,
            original_loopbacks: PulseOriginalLoopbacks::with_microphone(
                SystemCommandRunner,
                original_microphone_registry.clone(),
            ),
            graph: audio_graph,
        }),
        operation_gate: operation_gate.clone(),
    });
    manual_routes.initialize(&store);
    let audio_mix_application = Arc::new(AudioMixApplication::with_original_microphone(
        SystemCommandRunner,
        original_microphone_registry,
    ));
    let audio_mix: Arc<dyn AudioMixController> = audio_mix_application.clone();
    let duplex_config = build_duplex_config(lease.token_path());
    let mut aec_calibration = None;
    let mut aec_authority = None;
    let mut aec_projection = None;
    let mut facts: Arc<dyn RuntimeFactsSource> = manual_routes.clone();
    let mut maintenance: Arc<dyn RuntimeMaintenance> = manual_routes.clone();
    let mut api_routes: Arc<dyn ManualRouteController> = manual_routes.clone();
    let positive_path =
        std::env::var_os("TRANSLATOR_AEC_POSITIVE_PCM").map(std::path::PathBuf::from);
    let positive_hash = std::env::var_os("TRANSLATOR_AEC_POSITIVE_SHA256");
    let facts_server = std::env::var_os("TRANSLATOR_AEC_FACTS_SERVER");
    let fixture = load_native_positive_fixture(
        positive_path.as_deref(),
        positive_hash.as_deref().and_then(std::ffi::OsStr::to_str),
    )
    .and_then(|fixture| match fixture {
        Some(fixture) => {
            native_facts_server(facts_server.as_deref()).map(|server| Some((fixture, server)))
        }
        None => Ok(None),
    });
    let native_runner = match (duplex_config.as_ref(), fixture) {
        (Some(config), Ok(Some((fixture, facts_server)))) => {
            let coordinator = Arc::new(AecCalibrationCoordinator::new());
            let environment = Arc::new(PulseNativeAecEnvironment {
                runner: SystemCommandRunner,
                facts_server,
                mix: audio_mix.clone(),
                store: store.clone(),
            });
            let engine = Arc::new(NativeAecCalibrationEngine::new(
                config.clone(),
                store.clone(),
                environment.clone(),
                audio_mix_application.clone(),
                Arc::new(RuntimeLatencyObserver::new(store.clone())),
                fixture,
            ));
            let controller = AecCalibrationController::new(
                coordinator.clone(),
                operation_gate.clone(),
                engine.clone(),
            );
            let projected = Arc::new(AecProjectedRoutes {
                routes: manual_routes.clone(),
                coordinator: coordinator.clone(),
                environment,
            });
            projected.project(&store);
            facts = projected.clone();
            maintenance = projected.clone();
            aec_projection = Some(projected.clone());
            api_routes = projected;
            aec_authority = Some(
                AecRuntimeAuthority::new(coordinator, engine.clone())
                    .with_controller(controller.clone()),
            );
            aec_calibration = Some(Arc::new(controller));
            Some(engine as Arc<dyn translator_daemon::DuplexRunner>)
        }
        (_, Err(error)) => {
            tracing::warn!(event = "aec_calibration_unavailable", code = error.code);
            None
        }
        _ => None,
    };
    let translation = duplex_config.clone().map(|config| {
        let runner = native_runner.unwrap_or_else(|| {
            Arc::new(
                ProcessDuplexRunner::with_observer(
                    config,
                    Arc::new(RuntimeLatencyObserver::new(store.clone())),
                )
                .with_playback_mix_authority(audio_mix_application.clone())
                .with_start_facts(facts.clone()),
            )
        });
        ControlApplication::spawn_with_aec_authority(
            store.clone(),
            runner,
            operation_gate.clone(),
            facts.clone(),
            maintenance,
            Some(audio_mix.clone()),
            aec_authority,
        )
    });
    let round_trip = duplex_config.and_then(|config| {
        match RoundTripRuntimeHandle::try_new(
            store.clone(),
            Arc::new(RoundTripProcessRunner::new(config)),
            operation_gate.clone(),
            facts,
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
            manual_routes: Some(api_routes),
            translation: translation.clone(),
            aec_calibration: aec_calibration.clone(),
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
        translation.clone(),
        store.clone(),
        aec_projection,
    ));
    let debug_capture_watchdog = tokio::spawn(store.clone().run_debug_capture_watchdog());
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel::<()>();
    let server_task = tokio::spawn(async move {
        translator_daemon::serve_control(listener, router, async {
            let _ = shutdown_receiver.await;
        })
        .await
    });

    shutdown_signal().await;
    let (result, round_trip_result, translation_result, retained_drain) = drain_control_owners(
        &operation_gate,
        (shutdown_sender, server_task),
        [watcher_task, debug_capture_watchdog],
        round_trip.as_ref(),
        translation.as_deref(),
        &store,
    )
    .await;
    let aec_failed = match aec_calibration.as_ref() {
        Some(controller) => controller.shutdown().await.is_err(),
        None => false,
    };
    if aec_failed {
        tracing::error!(
            event = "daemon_shutdown_failed",
            code = "aec_cleanup_unconfirmed"
        );
    }
    if round_trip_result.is_err() {
        tracing::error!(
            event = "daemon_shutdown_failed",
            code = "round_trip_owner_failed"
        );
    }
    if let Err(error) = translation_result {
        tracing::error!(event = "daemon_shutdown_failed", code = error.code);
    }
    if aec_failed || round_trip_result.is_err() || translation_result.is_err() {
        return fail_stop((
            translation,
            aec_calibration,
            round_trip,
            manual_routes,
            lease,
            retained_drain,
        ))
        .await;
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

type RoundTripDrainer = tokio::task::JoinHandle<Result<(), RoundTripOwnerShutdownError>>;
type ControlDrainResults = (
    std::io::Result<()>,
    Result<(), RoundTripOwnerShutdownError>,
    Result<(), ControlFailure>,
    Option<RoundTripDrainer>,
);

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
) -> ControlDrainResults {
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
    // These are sequential per-owner budgets, not a single overall shutdown deadline.
    let (round_trip_result, retained_drain) = match round_trip {
        Some(controller) => drain_round_trip(controller).await,
        None => (Ok(()), None),
    };
    let translation_result = match translation {
        Some(controller) => drain_translation(controller).await,
        None => Ok(()),
    };
    let _ = store.set_debug_capture_enabled(false);
    store.shutdown_events();
    (
        server_result,
        round_trip_result,
        translation_result,
        retained_drain,
    )
}

async fn fail_stop<T>(owners: T) -> ExitCode {
    let _owners = owners;
    std::future::pending().await
}

async fn drain_translation(controller: &ControlApplication) -> Result<(), ControlFailure> {
    let deadline = tokio::time::Instant::now() + translator_daemon::RUNTIME_CLEANUP_BUDGET;
    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(ControlFailure {
                status: axum::http::StatusCode::CONFLICT,
                code: "translation_cleanup_pending",
            });
        }
        match controller.shutdown_until(deadline).await {
            Ok(()) if tokio::time::Instant::now() < deadline => return Ok(()),
            Ok(()) => continue,
            Err(error) => tracing::error!(event = "translation_shutdown_failed", code = error.code),
        }
        tokio::time::sleep_until(
            deadline.min(tokio::time::Instant::now() + std::time::Duration::from_secs(1)),
        )
        .await;
    }
}

async fn drain_round_trip(
    controller: &Arc<RoundTripRuntimeHandle>,
) -> (
    Result<(), RoundTripOwnerShutdownError>,
    Option<RoundTripDrainer>,
) {
    let deadline = tokio::time::Instant::now() + translator_daemon::RUNTIME_CLEANUP_BUDGET;
    let owner = Arc::clone(controller);
    let mut drainer = tokio::spawn(async move {
        drain_round_trip_attempts_until(deadline, || {
            let owner = Arc::clone(&owner);
            join_round_trip_shutdown(move || owner.shutdown_until(deadline.into_std()))
        })
        .await
    });
    match tokio::time::timeout_at(deadline, &mut drainer).await {
        Ok(Ok(result)) if tokio::time::Instant::now() < deadline => (result, None),
        Ok(Ok(_)) => (Err(RoundTripOwnerShutdownError::CleanupPending), None),
        Ok(Err(_)) => (Err(RoundTripOwnerShutdownError::OwnerFailed), None),
        Err(_) => (
            Err(RoundTripOwnerShutdownError::CleanupPending),
            Some(drainer),
        ),
    }
}

async fn join_round_trip_shutdown(
    attempt: impl FnOnce() -> Result<(), RoundTripOwnerShutdownError> + Send + 'static,
) -> Result<(), RoundTripOwnerShutdownError> {
    tokio::task::spawn_blocking(attempt)
        .await
        .map_err(|_| RoundTripOwnerShutdownError::OwnerFailed)?
}

#[cfg(test)]
async fn drain_round_trip_attempts<F, R>(mut attempt: F) -> Result<(), RoundTripOwnerShutdownError>
where
    F: FnMut() -> R,
    R: std::future::Future<Output = Result<(), RoundTripOwnerShutdownError>>,
{
    drain_round_trip_attempts_until(
        tokio::time::Instant::now() + translator_daemon::RUNTIME_CLEANUP_BUDGET,
        &mut attempt,
    )
    .await
}

async fn drain_round_trip_attempts_until<F, R>(
    deadline: tokio::time::Instant,
    mut attempt: F,
) -> Result<(), RoundTripOwnerShutdownError>
where
    F: FnMut() -> R,
    R: std::future::Future<Output = Result<(), RoundTripOwnerShutdownError>>,
{
    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(RoundTripOwnerShutdownError::CleanupPending);
        }
        match attempt().await {
            Ok(()) if tokio::time::Instant::now() < deadline => return Ok(()),
            Ok(()) => return Err(RoundTripOwnerShutdownError::CleanupPending),
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
        tokio::time::sleep_until(
            deadline.min(tokio::time::Instant::now() + std::time::Duration::from_secs(1)),
        )
        .await;
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

async fn watcher_loop<R: CommandRunner + Send + Sync + 'static>(
    controller: Option<Arc<ControlApplication>>,
    store: RuntimeStore,
    aec_projection: Option<Arc<AecProjectedRoutes<R>>>,
) {
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
        if let Some(projection) = aec_projection.as_ref() {
            let projection = projection.clone();
            let projection_store = store.clone();
            if tokio::task::spawn_blocking(move || projection.project(&projection_store))
                .await
                .is_err()
            {
                store.clear_devices("aec_pair_unavailable");
            }
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
    let deadline = Instant::now() + Duration::from_secs(8);
    let originals = match PulseOriginalLoopbacks::with_microphone(
        SystemCommandRunner,
        translator_audio::OriginalMicrophoneRegistry::default(),
    )
    .cleanup_all_until(deadline)
    {
        Ok(module_ids) => module_ids,
        Err(_) => {
            return print_json_failure(&serde_json::json!({
                "safe_error": "original_loopback_cleanup_failed"
            }));
        }
    };
    let mut graph = PulseAudioGraph::new(SystemCommandRunner, journal_path);
    match graph.cleanup_owned_until(deadline) {
        Ok(module_ids) => print_json(&serde_json::json!({
            "unloaded_module_ids": module_ids,
            "unloaded_original_module_ids": originals
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
        let start_command_count = runner.0.calls.lock().unwrap().len();
        drop(events);
        adapter.operation_gate.begin_stopping();
        controller.shutdown().await.unwrap();

        assert_eq!(result.unwrap_err().code, "translation_precondition_failed");
        assert_eq!(after, before);
        assert!(unexpected_event.is_err());
        assert_eq!(native.0.load(Ordering::SeqCst), 0);
        assert_eq!(gate, AudioOperationState::Idle);
        assert_eq!(start_command_count, 10);
        assert_eq!(
            runner.0.calls.lock().unwrap().len(),
            start_command_count + 2
        );
        assert_eq!(
            adapter.operation_gate.state(),
            AudioOperationState::Stopping
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
        ManualRouteAdmission, OriginalLoopbackRequest, PulseOriginalLoopbacks, RawPulseModuleId,
        RawPulseStream, SPEAKER_ORIGINAL_LOOPBACK, discover_original_loopbacks,
        maintain_audio_graph, manual_route_admission, matching_original_loopbacks,
        original_loopback_load_args, original_loopback_requests,
    };
    use translator_audio::{
        AecCapability, AudioEndpointState, AudioGraph, AudioGraphError, AudioGraphState,
        CommandResult, CommandRunError, CommandRunner, DeviceHealth, DeviceSelectionState,
        EndpointRole, GraphHealth, MIC_OUT_SINK, OutputMode, PhysicalDevice, REMOTE_IN_SINK,
        SystemCommandRunner,
    };
    use translator_daemon::{
        AcousticSafety, AdmittedDuplex, AudioMixState, AudioOperationGate, AudioOperationState,
        ControlApplication, ControlCommand, DeviceState, RuntimeSnapshot, RuntimeStore,
    };
    use uuid::Uuid;

    #[derive(Clone, Default)]
    struct ExclusiveRefreshRunner(Arc<Mutex<Vec<Vec<String>>>>);

    impl CommandRunner for ExclusiveRefreshRunner {
        fn run_until(
            &self,
            _: &str,
            args: &[String],
            _: Instant,
        ) -> Result<CommandResult, CommandRunError> {
            self.0.lock().unwrap().push(args.to_vec());
            Err(CommandRunError::NotFound)
        }
    }

    #[test]
    fn exclusive_audio_refresh_never_inspects_or_mutates_original_loopbacks() {
        for calibration in [true, false] {
            let runner = ExclusiveRefreshRunner::default();
            let gate = translator_daemon::AudioOperationGate::new();
            let _lease = if calibration {
                gate.acquire_calibration(Uuid::new_v4()).unwrap()
            } else {
                gate.acquire_human_round_trip(Uuid::new_v4()).unwrap()
            };
            let routes = super::PulseManualRoutes {
                resources: LifecycleProtected::new(super::PulseResources {
                    routing: translator_audio::PulseRoutingWatcher::new(
                        runner.clone(),
                        translator_audio::RoutingProfile::Production,
                    ),
                    devices: translator_audio::PulseDeviceWatcher::new(
                        runner.clone(),
                        AecCapability::Unavailable,
                    ),
                    original_loopbacks: PulseOriginalLoopbacks::new(runner.clone()),
                    graph: None,
                }),
                operation_gate: gate,
            };

            routes.refresh(&translator_daemon::RuntimeStore::default());
            assert!(runner.0.lock().unwrap().iter().all(|args| {
                !matches!(
                    args.as_slice(),
                    [first, ..] if first == "load-module" || first == "unload-module"
                ) && args.as_slice() != ["--format=json", "list", "sink-inputs"]
            }));
        }
    }

    #[test]
    fn bypass_fact_refresh_does_not_reconcile_original_loopbacks() {
        let runner = ExclusiveRefreshRunner::default();
        let loopback_runner = ExclusiveRefreshRunner::default();
        let gate = translator_daemon::AudioOperationGate::new();
        let _lease = gate.acquire_production().unwrap();
        let routes = super::PulseManualRoutes {
            resources: LifecycleProtected::new(super::PulseResources {
                routing: translator_audio::PulseRoutingWatcher::new(
                    runner.clone(),
                    translator_audio::RoutingProfile::Production,
                ),
                devices: translator_audio::PulseDeviceWatcher::new(
                    runner.clone(),
                    AecCapability::Unavailable,
                ),
                original_loopbacks: PulseOriginalLoopbacks::new(loopback_runner.clone()),
                graph: None,
            }),
            operation_gate: gate,
        };

        translator_daemon::RuntimeMaintenance::refresh_bypass_facts(
            &routes,
            &translator_daemon::RuntimeStore::default(),
        )
        .unwrap();
        assert!(!runner.0.lock().unwrap().is_empty());
        assert!(loopback_runner.0.lock().unwrap().is_empty());
    }

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
    fn zero_gain_headphone_microphone_has_a_pinned_original_acquisition_request() {
        let snapshot = RuntimeSnapshot {
            audio_mix: AudioMixState::default(),
            devices: Some(selected_devices()),
            ..RuntimeSnapshot::default()
        };
        assert_eq!(snapshot.audio_mix.microphone_original_percent, 0);
        let requests = original_loopback_requests(&snapshot);
        let microphones: Vec<_> = requests
            .iter()
            .filter(|request| request.media_name == MICROPHONE_ORIGINAL_LOOPBACK)
            .collect();
        assert_eq!(
            microphones.len(),
            1,
            "zero gain must not suppress acquisition"
        );
        assert_eq!(microphones[0].source, "alsa_input.microphone");
        assert_eq!(microphones[0].sink, MIC_OUT_SINK);
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
                    sink: "alsa_output.headphones".to_owned(),
                },
                OriginalLoopbackRequest {
                    media_name: MICROPHONE_ORIGINAL_LOOPBACK,
                    source: "alsa_input.microphone".to_owned(),
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
        assert_eq!(original_loopback_requests(&muted_originals).len(), 2);
        assert_eq!(
            original_loopback_requests(&muted_originals)[0].media_name,
            SPEAKER_ORIGINAL_LOOPBACK
        );

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
                    sink: "alsa_output.headphones".to_owned(),
                },
                OriginalLoopbackRequest {
                    media_name: MICROPHONE_ORIGINAL_LOOPBACK,
                    source: "alsa_input.microphone".to_owned(),
                    sink: MIC_OUT_SINK.to_owned(),
                },
            ]
        );
    }

    #[test]
    fn speaker_only_snapshot_never_requests_microphone_original_loopback() {
        let mut snapshot = RuntimeSnapshot {
            devices: Some(selected_devices()),
            ..RuntimeSnapshot::default()
        };
        snapshot
            .directions
            .iter_mut()
            .find(|direction| direction.direction_id == translator_core::AudioDirection::Microphone)
            .unwrap()
            .enabled = false;
        assert_eq!(
            original_loopback_requests(&snapshot),
            [OriginalLoopbackRequest {
                media_name: SPEAKER_ORIGINAL_LOOPBACK,
                source: format!("{REMOTE_IN_SINK}.monitor"),
                sink: "alsa_output.headphones".to_owned(),
            }]
        );
    }

    #[test]
    fn unsafe_output_never_requests_raw_microphone_loopback() {
        for mode in [OutputMode::OpenSpeaker, OutputMode::UnknownUnsafe] {
            for running in [false, true] {
                for microphone_original_percent in [0, 74] {
                    let mut devices = selected_devices();
                    devices.acoustic.mode = mode;
                    let snapshot = RuntimeSnapshot {
                        translation_running: running,
                        audio_mix: AudioMixState {
                            microphone_original_percent,
                            speaker_original_percent: 76,
                            ..AudioMixState::default()
                        },
                        devices: Some(devices),
                        ..RuntimeSnapshot::default()
                    };
                    let requests = original_loopback_requests(&snapshot);
                    assert!(
                        requests
                            .iter()
                            .all(|request| request.media_name != MICROPHONE_ORIGINAL_LOOPBACK
                                && request.sink != MIC_OUT_SINK),
                        "raw microphone loopback requested for {mode:?}, running={running}, percent={microphone_original_percent}"
                    );
                    assert_eq!(requests.len(), 1);
                    assert_eq!(requests[0].media_name, SPEAKER_ORIGINAL_LOOPBACK);
                    assert_eq!(requests[0].source, format!("{REMOTE_IN_SINK}.monitor"));
                    assert_eq!(requests[0].sink, "alsa_output.headphones");
                }
            }
        }
        let mut devices = selected_devices();
        devices.acoustic.mode = OutputMode::OpenSpeaker;
        let muted_speaker = RuntimeSnapshot {
            translation_running: true,
            audio_mix: AudioMixState::default(),
            devices: Some(devices),
            ..RuntimeSnapshot::default()
        };
        assert_eq!(original_loopback_requests(&muted_speaker).len(), 1);
        assert_eq!(
            original_loopback_requests(&muted_speaker)[0].media_name,
            SPEAKER_ORIGINAL_LOOPBACK
        );
    }

    #[test]
    fn original_loopback_discovery_ignores_name_matched_foreign_streams() {
        let owned_sink = raw_sink_stream(MICROPHONE_ORIGINAL_LOOPBACK, "41", 1);
        let owned_source = raw_source_stream(MICROPHONE_ORIGINAL_LOOPBACK, "41", 0);
        let mut foreign_sink = raw_sink_stream(MICROPHONE_ORIGINAL_LOOPBACK, "42", 1);
        foreign_sink.properties.remove("translator.owner");
        let mut foreign_source = raw_source_stream(MICROPHONE_ORIGINAL_LOOPBACK, "42", 0);
        foreign_source.properties.remove("translator.owner");

        let discovered = discover_original_loopbacks(
            &[owned_sink, foreign_sink],
            &[owned_source, foreign_source],
        )
        .unwrap();

        assert!(discovered.contains_key("41"));
        assert!(!discovered.contains_key("42"));
    }

    #[test]
    fn original_loopback_discovery_rejects_incomplete_owned_marker() {
        let sink = raw_sink_stream(MICROPHONE_ORIGINAL_LOOPBACK, "42", 1);
        let mut source = raw_source_stream(MICROPHONE_ORIGINAL_LOOPBACK, "42", 0);
        source.properties.remove("translator.owner");

        assert!(discover_original_loopbacks(&[sink], &[source]).is_err());

        let mut sink = raw_sink_stream(MICROPHONE_ORIGINAL_LOOPBACK, "42", 1);
        sink.properties.remove("translator.owner");
        let source = raw_source_stream(MICROPHONE_ORIGINAL_LOOPBACK, "42", 0);
        assert!(discover_original_loopbacks(&[sink], &[source]).is_err());
    }

    #[derive(Clone)]
    struct LoopbackRunner(Arc<Mutex<LoopbackFixture>>);

    struct LoopbackModule {
        id: u32,
        media_name: &'static str,
        source: String,
        sink: String,
        owned: bool,
    }

    #[derive(Default)]
    struct LoopbackFixture {
        modules: Vec<LoopbackModule>,
        calls: Vec<Vec<String>>,
        deadlines: Vec<Instant>,
        fail_unload: bool,
        retain_after_unload_ack: bool,
        hide_streams_after_unload_ack: bool,
        unload_acknowledged: bool,
        omit_loaded_module_after_ack: bool,
        missing_source_index: bool,
        missing_sink_index: bool,
        duplicate_sink_index: bool,
        malformed_module_id: bool,
        missing_sink_owner: bool,
        missing_source_owner: bool,
        unbound_stream_reads: usize,
        unbound_source_only: bool,
        load_seen: bool,
        load_ack_override: Option<String>,
        wrong_loaded_sink: bool,
        expire_after_load_inventory: bool,
    }

    impl LoopbackRunner {
        fn new(modules: Vec<LoopbackModule>) -> Self {
            Self(Arc::new(Mutex::new(LoopbackFixture {
                modules,
                ..LoopbackFixture::default()
            })))
        }

        fn calls(&self) -> Vec<Vec<String>> {
            self.0.lock().unwrap().calls.clone()
        }

        fn module_ids(&self) -> Vec<u32> {
            self.0
                .lock()
                .unwrap()
                .modules
                .iter()
                .map(|module| module.id)
                .collect()
        }
    }

    impl CommandRunner for LoopbackRunner {
        fn run_until(
            &self,
            program: &str,
            args: &[String],
            deadline: Instant,
        ) -> Result<CommandResult, CommandRunError> {
            assert_eq!(program, "pactl");
            let mut state = self.0.lock().unwrap();
            state.calls.push(args.to_vec());
            state.deadlines.push(deadline);
            match args {
                [list, short, kind] if list == "list" && short == "short" && kind == "modules" => {
                    let inventory = state
                        .modules
                        .iter()
                        .map(|module| {
                            format!(
                                "{}\tmodule-loopback\tsource={} sink={}\t\n",
                                module.id, module.source, module.sink
                            )
                        })
                        .collect::<String>();
                    Ok(CommandResult::success(inventory.into_bytes()))
                }
                [format, list, kind] if format == "--format=json" && list == "list" => {
                    if kind == "sources" {
                        let mut sources = serde_json::json!([
                            {"index": 0, "name": "alsa_input.microphone"},
                            {"index": 1, "name": format!("{REMOTE_IN_SINK}.monitor")},
                            {"index": 2, "name": "alsa_input.other"}
                        ]);
                        if state.missing_source_index {
                            sources.as_array_mut().unwrap().remove(2);
                        }
                        return Ok(CommandResult::success(
                            serde_json::to_vec(&sources).unwrap(),
                        ));
                    }
                    if kind == "sinks" {
                        let mut sinks = serde_json::json!([
                            {"index": 0, "name": "alsa_output.headphones"},
                            {"index": 1, "name": MIC_OUT_SINK},
                            {"index": 2, "name": "alsa_output.other"}
                        ]);
                        if state.duplicate_sink_index {
                            sinks.as_array_mut().unwrap().push(serde_json::json!({
                                "index": 0,
                                "name": "alsa_output.duplicate"
                            }));
                        }
                        if state.missing_sink_index {
                            sinks.as_array_mut().unwrap().remove(2);
                        }
                        return Ok(CommandResult::success(serde_json::to_vec(&sinks).unwrap()));
                    }
                    let unbound = state.load_seen && state.unbound_stream_reads > 0;
                    let streams: Vec<_> = state
                        .modules
                        .iter()
                        .filter(|_| {
                            !(state.hide_streams_after_unload_ack && state.unload_acknowledged)
                        })
                        .map(|module| {
                            let (endpoint, index) = if kind == "sink-inputs" {
                                ("sink", match module.sink.as_str() {
                                    "alsa_output.headphones" => 0,
                                    MIC_OUT_SINK => 1,
                                    "alsa_output.other" => 2,
                                    other => panic!("unexpected sink: {other}"),
                                })
                            } else {
                                ("source", match module.source.as_str() {
                                    "alsa_input.microphone" => 0,
                                    name if name == format!("{REMOTE_IN_SINK}.monitor") => 1,
                                    "alsa_input.other" => 2,
                                    other => panic!("unexpected source: {other}"),
                                })
                            };
                            let mut properties = serde_json::json!({
                                "media.name": module.media_name,
                            });
                            if module.owned
                                && !(kind == "sink-inputs" && state.missing_sink_owner)
                                && !(kind == "source-outputs" && state.missing_source_owner)
                            {
                                properties["translator.owner"] = "true".into();
                            }
                            let mut stream = serde_json::json!({
                                "owner_module": if state.malformed_module_id { "not-an-id".to_owned() } else { module.id.to_string() },
                                "properties": properties,
                            });
                            stream[endpoint] = if unbound && (!state.unbound_source_only || kind == "source-outputs") { u32::MAX } else { index }.into();
                            stream
                        })
                        .collect();
                    if unbound {
                        state.unbound_stream_reads -= 1;
                    }
                    if state.load_seen
                        && state.expire_after_load_inventory
                        && kind == "source-outputs"
                    {
                        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
                    }
                    Ok(CommandResult::success(
                        serde_json::to_vec(&streams).unwrap(),
                    ))
                }
                [command, id] if command == "unload-module" => {
                    if state.fail_unload {
                        return Ok(CommandResult::failure(Vec::new(), Vec::new()));
                    }
                    if !state.retain_after_unload_ack {
                        let id = id.parse::<u32>().unwrap();
                        state.modules.retain(|module| module.id != id);
                    }
                    state.unload_acknowledged = true;
                    Ok(CommandResult::success(Vec::new()))
                }
                [command, kind, rest @ ..]
                    if command == "load-module" && kind == "module-loopback" =>
                {
                    let media_name = if rest
                        .iter()
                        .any(|arg| arg.contains(MICROPHONE_ORIGINAL_LOOPBACK))
                    {
                        MICROPHONE_ORIGINAL_LOOPBACK
                    } else {
                        SPEAKER_ORIGINAL_LOOPBACK
                    };
                    let source = rest
                        .iter()
                        .find_map(|arg| arg.strip_prefix("source="))
                        .unwrap();
                    let sink = rest
                        .iter()
                        .find_map(|arg| arg.strip_prefix("sink="))
                        .unwrap();
                    let source = if media_name == SPEAKER_ORIGINAL_LOOPBACK {
                        format!("{REMOTE_IN_SINK}.monitor")
                    } else {
                        source.to_owned()
                    };
                    let id = state
                        .modules
                        .iter()
                        .map(|module| module.id)
                        .max()
                        .unwrap_or(40)
                        + 1;
                    if !state.omit_loaded_module_after_ack {
                        let sink = if state.wrong_loaded_sink {
                            "alsa_output.other".to_owned()
                        } else {
                            sink.to_owned()
                        };
                        state.modules.push(LoopbackModule {
                            id,
                            media_name,
                            source,
                            sink,
                            owned: true,
                        });
                    }
                    state.load_seen = true;
                    Ok(CommandResult::success(
                        state
                            .load_ack_override
                            .clone()
                            .unwrap_or_else(|| id.to_string())
                            .into_bytes(),
                    ))
                }
                _ => panic!("unexpected pactl call: {args:?}"),
            }
        }
    }

    fn microphone_module(id: u32, owned: bool) -> LoopbackModule {
        LoopbackModule {
            id,
            media_name: MICROPHONE_ORIGINAL_LOOPBACK,
            source: "alsa_input.microphone".to_owned(),
            sink: MIC_OUT_SINK.to_owned(),
            owned,
        }
    }

    fn speaker_module(id: u32) -> LoopbackModule {
        LoopbackModule {
            id,
            media_name: SPEAKER_ORIGINAL_LOOPBACK,
            source: format!("{REMOTE_IN_SINK}.monitor"),
            sink: "alsa_output.headphones".to_owned(),
            owned: true,
        }
    }

    fn loopback_snapshot(mode: OutputMode) -> RuntimeSnapshot {
        let mut devices = selected_devices();
        devices.acoustic.mode = mode;
        RuntimeSnapshot {
            translation_running: false,
            audio_mix: AudioMixState {
                microphone_original_percent: 100,
                ..AudioMixState::default()
            },
            devices: Some(devices),
            ..RuntimeSnapshot::default()
        }
    }

    #[test]
    fn fresh_headphones_keep_speaker_loopback_across_running_and_stop() {
        let runner = LoopbackRunner::new(Vec::new());
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());
        let snapshot = RuntimeSnapshot {
            audio_mix: AudioMixState::default(),
            ..loopback_snapshot(OutputMode::Headphones)
        };

        loopbacks.ensure_without_new_mic(&snapshot).unwrap();
        let running = RuntimeSnapshot {
            translation_running: true,
            ..snapshot.clone()
        };
        loopbacks.ensure_without_new_mic(&running).unwrap();
        loopbacks.ensure_without_new_mic(&snapshot).unwrap();
        loopbacks.verify_existing(&snapshot, false).unwrap();
        assert!(loopbacks.verify_existing(&snapshot, true).is_err());
        assert_eq!(runner.module_ids(), vec![41]);
        let loads: Vec<_> = runner
            .calls()
            .into_iter()
            .filter(|args| args.first().map(String::as_str) == Some("load-module"))
            .collect();
        assert_eq!(loads.len(), 1);
        assert!(
            !runner
                .calls()
                .iter()
                .any(|args| { args.first().map(String::as_str) == Some("unload-module") })
        );
        assert!(
            loads[0]
                .iter()
                .any(|arg| arg.contains(SPEAKER_ORIGINAL_LOOPBACK))
        );
        assert!(
            !loads[0]
                .iter()
                .any(|arg| arg.contains(MICROPHONE_ORIGINAL_LOOPBACK))
        );
    }

    #[test]
    fn native_microphone_muted_bypass_does_not_require_capture_before_start() {
        let runner = LoopbackRunner::new(Vec::new());
        let registry = translator_audio::OriginalMicrophoneRegistry::default();
        let originals = PulseOriginalLoopbacks::with_microphone(runner.clone(), registry.clone());
        let mut snapshot = loopback_snapshot(OutputMode::Headphones);
        snapshot.translation_running = false;
        snapshot.audio_mix.microphone_original_percent = 0;
        snapshot
            .devices
            .as_mut()
            .unwrap()
            .source
            .selected
            .as_mut()
            .unwrap()
            .id = 0;
        originals.ensure_without_new_mic(&snapshot).unwrap();
        assert!(registry.current().unwrap().is_none());
        originals.verify_existing(&snapshot, false).unwrap();
        assert!(originals.verify_existing(&snapshot, true).is_err());
        assert_eq!(runner.module_ids().len(), 1);
    }

    #[test]
    fn native_bypass_custody_rejects_unadmitted_full_duplex_without_capture() {
        let runner = LoopbackRunner::new(Vec::new());
        let registry = translator_audio::OriginalMicrophoneRegistry::default();
        let originals = PulseOriginalLoopbacks::with_microphone(runner.clone(), registry.clone());
        let mut snapshot = loopback_snapshot(OutputMode::Headphones);
        snapshot.translation_running = false;
        snapshot.audio_mix.microphone_original_percent = 0;
        snapshot
            .devices
            .as_mut()
            .unwrap()
            .acoustic
            .full_duplex_allowed = false;
        originals.ensure_without_new_mic(&snapshot).unwrap();
        let calls = runner.calls().len();
        assert!(registry.current().unwrap().is_none());
        originals.verify_existing(&snapshot, false).unwrap();
        assert!(originals.verify_existing(&snapshot, true).is_err());
        assert!(registry.current().unwrap().is_none());
        assert!(runner.calls()[calls..].iter().all(|args| {
            !matches!(
                args.first().map(String::as_str),
                Some("load-module" | "unload-module")
            )
        }));
    }

    #[test]
    fn vanished_speaker_loopback_is_not_reloaded_until_runtime_is_stopped() {
        let runner = LoopbackRunner::new(Vec::new());
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());
        let stopped = RuntimeSnapshot {
            audio_mix: AudioMixState::default(),
            ..loopback_snapshot(OutputMode::Headphones)
        };
        loopbacks.ensure_without_new_mic(&stopped).unwrap();
        runner.0.lock().unwrap().modules.clear();
        let running = RuntimeSnapshot {
            translation_running: true,
            ..stopped.clone()
        };
        let loads_before = runner
            .calls()
            .iter()
            .filter(|args| args.first().map(String::as_str) == Some("load-module"))
            .count();

        assert!(loopbacks.ensure_without_new_mic(&running).is_err());
        assert_eq!(
            runner
                .calls()
                .iter()
                .filter(|args| args.first().map(String::as_str) == Some("load-module"))
                .count(),
            loads_before
        );
        loopbacks.ensure_without_new_mic(&stopped).unwrap();
        loopbacks.verify_existing(&stopped, false).unwrap();
        assert!(loopbacks.verify_existing(&stopped, true).is_err());
        assert_eq!(runner.module_ids().len(), 1);
        let loads: Vec<_> = runner
            .calls()
            .into_iter()
            .filter(|args| args.first().map(String::as_str) == Some("load-module"))
            .collect();
        assert_eq!(loads.len(), loads_before + 1);
        assert!(loads.iter().all(|args| {
            args.iter()
                .any(|arg| arg.contains(SPEAKER_ORIGINAL_LOOPBACK))
                && !args
                    .iter()
                    .any(|arg| arg.contains(MICROPHONE_ORIGINAL_LOOPBACK))
        }));
    }

    #[test]
    fn bypass_repair_with_missing_requested_mic_uses_only_a_local_muted_snapshot() {
        let runner = LoopbackRunner::new(vec![speaker_module(43)]);
        let gate = translator_daemon::AudioOperationGate::new();
        let _lease = gate.acquire_production().unwrap();
        let routes = super::PulseManualRoutes {
            resources: LifecycleProtected::new(super::PulseResources {
                routing: translator_audio::PulseRoutingWatcher::new(
                    runner.clone(),
                    translator_audio::RoutingProfile::Production,
                ),
                devices: translator_audio::PulseDeviceWatcher::new(
                    runner.clone(),
                    AecCapability::Unavailable,
                ),
                original_loopbacks: PulseOriginalLoopbacks::new(runner.clone()),
                graph: None,
            }),
            operation_gate: gate,
        };
        let snapshot = loopback_snapshot(OutputMode::Headphones);
        assert_eq!(snapshot.audio_mix.microphone_original_percent, 100);

        translator_daemon::RuntimeMaintenance::prepare_bypass(&routes, &snapshot).unwrap();
        let unchanged = snapshot.audio_mix.microphone_original_percent;
        assert_eq!(unchanged, 100);
        assert_eq!(runner.module_ids(), vec![43]);
        assert!(
            runner
                .calls()
                .iter()
                .all(|args| { args.first().map(String::as_str) != Some("load-module") })
        );
        let owned = PulseOriginalLoopbacks::new(runner);
        owned.verify_existing(&snapshot, false).unwrap();
        assert!(owned.verify_existing(&snapshot, true).is_err());
    }

    #[test]
    fn incomplete_owned_loopback_blocks_reconcile_without_mutating_modules() {
        for missing_source_owner in [false, true] {
            let runner = LoopbackRunner::new(vec![microphone_module(41, true)]);
            {
                let mut state = runner.0.lock().unwrap();
                state.missing_source_owner = missing_source_owner;
                state.missing_sink_owner = !missing_source_owner;
            }
            let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

            assert!(
                loopbacks
                    .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                    .is_err()
            );
            assert_eq!(runner.module_ids(), vec![41]);
            assert!(runner.calls().iter().all(|args| {
                !matches!(
                    args.first().map(String::as_str),
                    Some("load-module" | "unload-module")
                )
            }));
        }
    }

    #[test]
    fn unsafe_transition_removes_owned_mic_before_loading_speaker() {
        let runner = LoopbackRunner::new(vec![
            microphone_module(41, true),
            microphone_module(42, false),
        ]);
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

        loopbacks
            .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
            .unwrap();

        let calls = runner.calls();
        let unload = calls
            .iter()
            .position(|args| args == &["unload-module", "41"])
            .unwrap();
        let load = calls
            .iter()
            .position(|args| args.first().map(String::as_str) == Some("load-module"))
            .unwrap();
        assert!(unload < load);
        assert!(
            runner.module_ids().contains(&42),
            "foreign same-name module was removed"
        );
        assert!(!runner.module_ids().contains(&41));
    }

    #[test]
    fn unsafe_transition_retains_matching_speaker_loopback() {
        let runner = LoopbackRunner::new(vec![microphone_module(41, true), speaker_module(43)]);
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

        loopbacks
            .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
            .unwrap();

        assert_eq!(runner.module_ids(), [43]);
        assert!(
            runner
                .calls()
                .iter()
                .all(|args| args.first().map(String::as_str) != Some("load-module"))
        );
    }

    #[test]
    fn failed_unsafe_transition_cleanup_does_not_load_new_loopback() {
        let runner = LoopbackRunner::new(vec![microphone_module(41, true)]);
        runner.0.lock().unwrap().fail_unload = true;
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

        assert!(
            loopbacks
                .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                .is_err()
        );

        assert!(
            runner
                .calls()
                .iter()
                .all(|args| args.first().map(String::as_str) != Some("load-module"))
        );
        assert!(
            runner.module_ids().contains(&41),
            "failed cleanup must remain visible"
        );
    }

    #[test]
    fn acknowledged_unload_with_owned_module_still_visible_blocks_replacement() {
        let runner = LoopbackRunner::new(vec![microphone_module(41, true)]);
        runner.0.lock().unwrap().retain_after_unload_ack = true;
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

        assert!(
            loopbacks
                .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                .is_err()
        );
        assert_eq!(runner.module_ids(), vec![41]);
        let calls = runner.calls();
        assert!(calls.iter().any(|args| args == &["unload-module", "41"]));
        assert!(
            calls
                .iter()
                .all(|args| args.first().map(String::as_str) != Some("load-module"))
        );
    }

    #[test]
    fn acknowledged_unload_with_hidden_streams_but_module_still_loaded_blocks_replacement() {
        let runner = LoopbackRunner::new(vec![microphone_module(41, true)]);
        {
            let mut state = runner.0.lock().unwrap();
            state.retain_after_unload_ack = true;
            state.hide_streams_after_unload_ack = true;
        }
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

        assert!(
            loopbacks
                .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                .is_err()
        );
        assert_eq!(runner.module_ids(), vec![41]);
        assert!(
            runner
                .calls()
                .iter()
                .all(|args| { args.first().map(String::as_str) != Some("load-module") })
        );
    }

    #[test]
    fn acknowledged_load_without_owned_pair_does_not_certify_route() {
        let runner = LoopbackRunner::new(Vec::new());
        runner.0.lock().unwrap().omit_loaded_module_after_ack = true;
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

        assert!(
            loopbacks
                .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                .is_err()
        );
        assert!(runner.module_ids().is_empty());
        assert!(
            runner
                .calls()
                .iter()
                .any(|args| { args.first().map(String::as_str) == Some("load-module") })
        );
    }

    #[test]
    fn fresh_loopback_waits_only_for_exact_unbound_target_and_reuses_module() {
        let runner = LoopbackRunner::new(Vec::new());
        runner.0.lock().unwrap().unbound_stream_reads = 2;
        let registry = translator_audio::OriginalMicrophoneRegistry::default();
        let loopbacks = PulseOriginalLoopbacks::with_microphone(runner.clone(), registry.clone());
        let mut snapshot = loopback_snapshot(OutputMode::Headphones);
        snapshot.audio_mix.microphone_original_percent = 0;
        snapshot
            .devices
            .as_mut()
            .unwrap()
            .source
            .selected
            .as_mut()
            .unwrap()
            .id = 0;

        loopbacks.ensure_without_new_mic(&snapshot).unwrap();
        assert!(registry.current().unwrap().is_none());
        {
            let state = runner.0.lock().unwrap();
            let load = state
                .calls
                .iter()
                .position(|args| args[0] == "load-module")
                .unwrap();
            assert_eq!(
                state.calls[load + 1..].len(),
                4,
                "both sides must be read again after pending"
            );
            assert!(
                state.deadlines[load..]
                    .iter()
                    .all(|deadline| *deadline == state.deadlines[load])
            );
        }
        loopbacks.ensure_without_new_mic(&snapshot).unwrap();
        assert_eq!(
            runner
                .calls()
                .iter()
                .filter(|args| args[0] == "load-module")
                .count(),
            1
        );
        assert!(registry.current().unwrap().is_none());
    }

    #[test]
    fn fresh_loopback_wrong_ack_or_bound_target_fails_without_polling() {
        for fault in [
            "wrong_ack",
            "malformed_ack",
            "reserved_ack",
            "wrong_target",
            "foreign_owner",
        ] {
            let runner = LoopbackRunner::new(Vec::new());
            {
                let mut state = runner.0.lock().unwrap();
                match fault {
                    "wrong_ack" => state.load_ack_override = Some("42".to_owned()),
                    "malformed_ack" => state.load_ack_override = Some("not-an-id".to_owned()),
                    "reserved_ack" => state.load_ack_override = Some(u32::MAX.to_string()),
                    "wrong_target" => {
                        state.wrong_loaded_sink = true;
                        state.unbound_source_only = true;
                        state.unbound_stream_reads = 2;
                    }
                    "foreign_owner" => state.missing_sink_owner = true,
                    _ => unreachable!(),
                }
            }
            let loopbacks = PulseOriginalLoopbacks::new(runner.clone());
            assert!(
                loopbacks
                    .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                    .is_err(),
                "{fault}"
            );
            let calls = runner.calls();
            let load = calls
                .iter()
                .position(|args| args[0] == "load-module")
                .unwrap();
            assert!(
                calls[load + 1..].len() <= 2,
                "contradiction must not be polled: {fault}"
            );
            assert_eq!(
                calls.iter().filter(|args| args[0] == "load-module").count(),
                1
            );
        }
    }

    #[test]
    fn fresh_loopback_existing_unbound_pair_has_no_readiness_grace() {
        let runner = LoopbackRunner::new(vec![speaker_module(43)]);
        {
            let mut state = runner.0.lock().unwrap();
            state.load_seen = true;
            state.unbound_stream_reads = 2;
        }
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());
        assert!(
            loopbacks
                .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                .is_err()
        );
        assert_eq!(runner.calls().len(), 4);
        assert_eq!(runner.module_ids(), vec![43]);
    }

    #[test]
    fn fresh_loopback_pending_expiry_retains_exact_cleanup_custody() {
        let runner = LoopbackRunner::new(vec![microphone_module(42, false)]);
        runner.0.lock().unwrap().unbound_stream_reads = usize::MAX;
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());
        assert!(
            loopbacks
                .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                .is_err()
        );
        {
            let state = runner.0.lock().unwrap();
            let load = state
                .calls
                .iter()
                .position(|args| args[0] == "load-module")
                .unwrap();
            assert!(
                state.calls[load + 1..].len() >= 4,
                "expiry must occur after pending observations"
            );
            let deadline = state.deadlines[load];
            assert!(
                state.deadlines[load..]
                    .iter()
                    .all(|value| *value == deadline)
            );
            assert!(Instant::now() >= deadline);
        }
        assert_eq!(loopbacks.cleanup_all().unwrap(), vec!["43"]);
        assert_eq!(
            runner.module_ids(),
            vec![42],
            "foreign resource must survive cleanup"
        );
    }

    #[test]
    fn fresh_loopback_late_bound_inventory_cannot_certify_readiness() {
        let runner = LoopbackRunner::new(Vec::new());
        runner.0.lock().unwrap().expire_after_load_inventory = true;
        let loopbacks = PulseOriginalLoopbacks::new(runner);
        assert!(
            loopbacks
                .ensure(&loopback_snapshot(OutputMode::OpenSpeaker))
                .is_err()
        );
    }

    #[test]
    fn fresh_loopback_identical_duplicate_streams_are_not_one_pair() {
        let sink = serde_json::json!({"owner_module": 43, "sink": 0,
            "properties": {"media.name": SPEAKER_ORIGINAL_LOOPBACK, "translator.owner": "true"}});
        let source = serde_json::json!({"owner_module": 43, "source": 1,
            "properties": {"media.name": SPEAKER_ORIGINAL_LOOPBACK, "translator.owner": "true"}});
        for duplicate_sink in [true, false] {
            let sinks: Vec<RawPulseStream> = serde_json::from_value(if duplicate_sink {
                serde_json::json!([sink, sink])
            } else {
                serde_json::json!([sink])
            })
            .unwrap();
            let sources: Vec<RawPulseStream> = serde_json::from_value(if duplicate_sink {
                serde_json::json!([source])
            } else {
                serde_json::json!([source, source])
            })
            .unwrap();
            assert!(discover_original_loopbacks(&sinks, &sources).is_err());
        }
    }

    #[test]
    fn headphones_restore_mic_loopback_once_and_missing_devices_clean_owned_only() {
        let runner = LoopbackRunner::new(vec![microphone_module(42, false)]);
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());
        let headphones = loopback_snapshot(OutputMode::Headphones);

        loopbacks.ensure(&headphones).unwrap();
        let loaded_once = runner.calls();
        assert_eq!(
            loaded_once
                .iter()
                .filter(
                    |args| args.first().map(String::as_str) == Some("load-module")
                        && args
                            .iter()
                            .any(|arg| arg.contains(MICROPHONE_ORIGINAL_LOOPBACK))
                )
                .count(),
            1
        );
        assert_eq!(
            loaded_once
                .iter()
                .filter(
                    |args| args.first().map(String::as_str) == Some("load-module")
                        && args
                            .iter()
                            .any(|arg| arg.contains(SPEAKER_ORIGINAL_LOOPBACK))
                )
                .count(),
            1
        );
        loopbacks.ensure(&headphones).unwrap();
        assert_eq!(
            runner
                .calls()
                .iter()
                .filter(|args| args.first().map(String::as_str) == Some("load-module"))
                .count(),
            2
        );

        let missing = RuntimeSnapshot {
            devices: None,
            ..headphones
        };
        loopbacks.ensure(&missing).unwrap();
        assert_eq!(runner.module_ids(), [42]);
    }

    #[test]
    fn production_reconcile_never_creates_a_new_raw_microphone_loopback() {
        let runner = LoopbackRunner::new(vec![speaker_module(43)]);
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

        assert!(
            loopbacks
                .ensure_without_new_mic(&loopback_snapshot(OutputMode::Headphones))
                .is_err()
        );
        assert_eq!(runner.module_ids(), vec![43]);
        assert!(
            runner
                .calls()
                .iter()
                .all(|args| { args.first().map(String::as_str) != Some("load-module") })
        );
    }

    #[test]
    fn bypass_custody_is_read_only_and_requires_exact_owned_pair() {
        let runner = LoopbackRunner::new(vec![speaker_module(43), microphone_module(41, true)]);
        let loopbacks = PulseOriginalLoopbacks::new(runner.clone());
        let headphones = loopback_snapshot(OutputMode::Headphones);
        let speaker = loopback_snapshot(OutputMode::OpenSpeaker);

        loopbacks.verify_existing(&headphones, true).unwrap();
        assert!(loopbacks.verify_existing(&speaker, false).is_err());
        assert!(runner.calls().iter().all(|args| {
            !matches!(
                args.first().map(String::as_str),
                Some("load-module" | "unload-module")
            )
        }));

        let speaker_only = LoopbackRunner::new(vec![speaker_module(43)]);
        let loopbacks = PulseOriginalLoopbacks::new(speaker_only.clone());
        loopbacks.verify_existing(&speaker, false).unwrap();
        assert!(loopbacks.verify_existing(&headphones, true).is_err());
        assert!(speaker_only.calls().iter().all(|args| {
            !matches!(
                args.first().map(String::as_str),
                Some("load-module" | "unload-module")
            )
        }));
    }

    #[test]
    fn wrong_source_or_sink_is_replaced_after_owned_unload() {
        for wrong_source in [false, true] {
            let mut module = microphone_module(41, true);
            if wrong_source {
                module.source = "alsa_input.other".to_owned();
            } else {
                module.sink = "alsa_output.other".to_owned();
            }
            let runner = LoopbackRunner::new(vec![
                speaker_module(43),
                module,
                microphone_module(42, false),
            ]);
            let loopbacks = PulseOriginalLoopbacks::new(runner.clone());
            let snapshot = RuntimeSnapshot {
                translation_running: true,
                audio_mix: AudioMixState {
                    microphone_original_percent: 74,
                    ..AudioMixState::default()
                },
                ..loopback_snapshot(OutputMode::Headphones)
            };

            loopbacks.ensure(&snapshot).unwrap();

            let calls = runner.calls();
            let unload = calls
                .iter()
                .position(|args| args == &["unload-module", "41"])
                .unwrap();
            let load = calls
                .iter()
                .position(|args| args.first().map(String::as_str) == Some("load-module"))
                .unwrap();
            assert!(unload < load);
            assert_eq!(
                calls
                    .iter()
                    .filter(|args| args.first().map(String::as_str) == Some("load-module"))
                    .count(),
                1
            );
            assert!(runner.module_ids().contains(&42));
            let state = runner.0.lock().unwrap();
            let replacement = state
                .modules
                .iter()
                .find(|module| module.owned && module.media_name == MICROPHONE_ORIGINAL_LOOPBACK)
                .unwrap();
            assert_eq!(replacement.source, "alsa_input.microphone");
            assert_eq!(replacement.sink, MIC_OUT_SINK);
        }
    }

    #[test]
    fn incomplete_owned_inventory_prevents_all_mutation() {
        for fault in 0..4 {
            let mut module = microphone_module(41, true);
            if fault == 0 {
                module.source = "alsa_input.other".to_owned();
            }
            if fault == 3 {
                module.sink = "alsa_output.other".to_owned();
            }
            let runner = LoopbackRunner::new(vec![module]);
            {
                let mut state = runner.0.lock().unwrap();
                state.missing_source_index = fault == 0;
                state.duplicate_sink_index = fault == 1;
                state.malformed_module_id = fault == 2;
                state.missing_sink_index = fault == 3;
            }
            let loopbacks = PulseOriginalLoopbacks::new(runner.clone());

            assert!(
                loopbacks
                    .ensure(&loopback_snapshot(OutputMode::Headphones))
                    .is_err()
            );
            assert!(runner.calls().iter().all(|args| !matches!(
                args.first().map(String::as_str),
                Some("load-module" | "unload-module")
            )));
            if fault == 2 {
                assert!(loopbacks.cleanup_all().is_err());
                assert!(runner.module_ids().contains(&41));
            }
        }
    }

    #[test]
    fn projected_original_cleanup_forwards_custody_and_one_absolute_deadline() {
        let runner = LoopbackRunner::new(vec![speaker_module(43)]);
        let store = RuntimeStore::default();
        let gate = AudioOperationGate::new();
        let _lease = gate.acquire_production().unwrap();
        let registry = translator_audio::OriginalMicrophoneRegistry::default();
        let projection = super::AecProjectedRoutes {
            routes: Arc::new(super::PulseManualRoutes {
                resources: LifecycleProtected::new(super::PulseResources {
                    routing: translator_audio::PulseRoutingWatcher::new(
                        runner.clone(),
                        translator_audio::RoutingProfile::Production,
                    ),
                    devices: translator_audio::PulseDeviceWatcher::new(
                        runner.clone(),
                        AecCapability::Unavailable,
                    ),
                    original_loopbacks: PulseOriginalLoopbacks::with_microphone(
                        runner.clone(),
                        registry.clone(),
                    ),
                    graph: None,
                }),
                operation_gate: gate,
            }),
            coordinator: Arc::new(translator_daemon::AecCalibrationCoordinator::new()),
            environment: Arc::new(super::PulseNativeAecEnvironment {
                runner: runner.clone(),
                facts_server: "unix:/test/unused-read-only-facts".into(),
                mix: Arc::new(
                    translator_daemon::AudioMixApplication::with_original_microphone(
                        runner.clone(),
                        registry.clone(),
                    ),
                ),
                store,
            }),
        };
        let mut snapshot = loopback_snapshot(OutputMode::Headphones);
        snapshot.audio_mix.microphone_original_percent = 0;
        assert!(projection.prepare_existing(&snapshot).is_err());
        snapshot
            .devices
            .as_mut()
            .unwrap()
            .source
            .selected
            .as_mut()
            .unwrap()
            .id = 0;
        snapshot
            .devices
            .as_mut()
            .unwrap()
            .sink
            .selected
            .as_mut()
            .unwrap()
            .id = 0;
        projection.prepare_existing(&snapshot).unwrap();
        assert!(
            registry.current().unwrap().is_none(),
            "projected refresh must not acquire raw microphone"
        );
        assert_eq!(runner.module_ids(), vec![43]);
        assert!(runner.calls().iter().all(|args| args[0] != "load-module"));
        runner.0.lock().unwrap().calls.clear();
        runner.0.lock().unwrap().deadlines.clear();
        runner
            .0
            .lock()
            .unwrap()
            .modules
            .push(microphone_module(41, true));
        let deadline = Instant::now() + Duration::from_secs(1);
        translator_daemon::RuntimeMaintenance::cleanup_originals(&projection, deadline).unwrap();
        assert!(runner.module_ids().is_empty());
        assert!(
            runner
                .0
                .lock()
                .unwrap()
                .deadlines
                .iter()
                .all(|observed| *observed == deadline)
        );
        runner.0.lock().unwrap().modules.push(speaker_module(45));
        runner.0.lock().unwrap().fail_unload = true;
        assert!(
            translator_daemon::RuntimeMaintenance::cleanup_originals(&projection, deadline)
                .is_err()
        );
        assert_eq!(runner.module_ids(), vec![45]);
    }

    #[test]
    #[ignore = "requires a disposable private PulseAudio socket and virtual fixture sinks"]
    fn private_pulse_original_loopback_load_discover_and_cleanup() {
        let server = std::env::var("PULSE_SERVER").expect("private PULSE_SERVER is required");
        assert!(
            server.starts_with("unix:/tmp/translator-loopback-") && server.ends_with("/native"),
            "refusing to use a non-fixture PulseAudio server"
        );
        let mut devices = selected_devices();
        devices.source.selected.as_mut().unwrap().name = "translator_test_mic.monitor".to_owned();
        devices.sink.selected.as_mut().unwrap().name = "translator_test_out".to_owned();
        let snapshot = RuntimeSnapshot {
            translation_running: true,
            audio_mix: AudioMixState {
                microphone_original_percent: 100,
                ..AudioMixState::default()
            },
            devices: Some(devices),
            ..RuntimeSnapshot::default()
        };
        let loopbacks = PulseOriginalLoopbacks::new(SystemCommandRunner);

        loopbacks.cleanup_all().unwrap();
        let stopped = RuntimeSnapshot {
            translation_running: false,
            audio_mix: AudioMixState {
                microphone_original_percent: 0,
                ..AudioMixState::default()
            },
            ..snapshot.clone()
        };
        loopbacks.ensure(&stopped).unwrap();
        loopbacks.ensure(&snapshot).unwrap();
        loopbacks.ensure(&snapshot).unwrap();
        let sink_inputs: Vec<RawPulseStream> = loopbacks
            .run_json(&["--format=json", "list", "sink-inputs"])
            .unwrap();
        let source_outputs: Vec<RawPulseStream> = loopbacks
            .run_json(&["--format=json", "list", "source-outputs"])
            .unwrap();
        let discovered = discover_original_loopbacks(&sink_inputs, &source_outputs).unwrap();
        assert_eq!(discovered.len(), 2);
        assert!(
            discovered
                .values()
                .any(|entry| entry.media_name == MICROPHONE_ORIGINAL_LOOPBACK)
        );
        assert!(
            discovered
                .values()
                .any(|entry| entry.media_name == SPEAKER_ORIGINAL_LOOPBACK)
        );
        assert_eq!(loopbacks.cleanup_all().unwrap().len(), 2);
        let sink_inputs: Vec<RawPulseStream> = loopbacks
            .run_json(&["--format=json", "list", "sink-inputs"])
            .unwrap();
        let source_outputs: Vec<RawPulseStream> = loopbacks
            .run_json(&["--format=json", "list", "source-outputs"])
            .unwrap();
        assert!(
            discover_original_loopbacks(&sink_inputs, &source_outputs)
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires a disposable private PulseAudio socket and virtual fixture sinks"]
    async fn private_pulse_native_original_zero_gain_gain_changes_and_cleanup() {
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .with_max_level(tracing::Level::WARN)
            .try_init();
        use std::os::unix::fs::FileTypeExt;
        use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64};
        use translator_audio::{
            OriginalMicrophoneRegistry, PcmFrame, PulsePcmCapture, PulsePcmCommand,
            PulsePcmPlayback, StreamPcmFormat,
        };
        use translator_daemon::{
            AudioMixApplication, AudioMixController, PlaybackMixAuthority, TranslationMixMode,
        };

        let server = std::env::var("PULSE_SERVER").expect("private PULSE_SERVER required");
        assert!(
            server.starts_with("unix:/tmp/translator-loopback-") && server.ends_with("/native"),
            "refusing non-fixture Pulse server"
        );
        assert!(
            std::fs::symlink_metadata(server.strip_prefix("unix:").unwrap())
                .unwrap()
                .file_type()
                .is_socket()
        );
        let registry = OriginalMicrophoneRegistry::default();
        let originals =
            PulseOriginalLoopbacks::with_microphone(SystemCommandRunner, registry.clone());
        let mix =
            AudioMixApplication::with_original_microphone(SystemCommandRunner, registry.clone());
        let endpoint = |kind: &str, name: &str| -> u32 {
            let result = std::process::Command::new("pactl")
                .args(["--format=json", "list", kind])
                .output()
                .unwrap();
            assert!(result.status.success());
            let values: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
            values
                .as_array()
                .unwrap()
                .iter()
                .find(|value| value["name"] == name)
                .unwrap()["index"]
                .as_u64()
                .unwrap() as u32
        };
        let mut devices = selected_devices();
        let microphone = "translator_test_mic.monitor";
        devices.source.selected.as_mut().unwrap().name = microphone.into();
        devices.source.selected.as_mut().unwrap().id = endpoint("sources", microphone);
        devices.source.pinned_name = Some(microphone.into());
        devices.sink.selected.as_mut().unwrap().name = "translator_test_out".into();
        devices.sink.selected.as_mut().unwrap().id = endpoint("sinks", "translator_test_out");
        devices.sink.pinned_name = Some("translator_test_out".into());
        let snapshot = RuntimeSnapshot {
            devices: Some(devices),
            ..RuntimeSnapshot::default()
        };
        let mut tone = PulsePcmPlayback::spawn(&PulsePcmCommand::playback(
            "translator_test_mic",
            "synthetic-microphone-input",
        ))
        .unwrap();
        let tone_registration = tone
            .wait_registered_muted(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert!(
            std::process::Command::new("pactl")
                .args([
                    "set-sink-input-volume",
                    &tone_registration.index().to_string(),
                    "100%"
                ])
                .status()
                .unwrap()
                .success()
        );
        let done = Arc::new(AtomicBool::new(false));
        let tone_done = done.clone();
        let tone_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(20));
            let pcm: Vec<u8> = (0..320)
                .flat_map(|index| {
                    let sample = (8_000.0
                        * (std::f64::consts::TAU * 500.0 * index as f64 / 16_000.0).sin())
                        as i16;
                    sample.to_le_bytes()
                })
                .collect();
            let mut sequence = 0;
            while !tone_done.load(Ordering::Acquire) {
                interval.tick().await;
                let frame = PcmFrame::try_new(
                    sequence,
                    sequence * 20_000_000,
                    StreamPcmFormat::provider_default(),
                    pcm.clone(),
                )
                .unwrap();
                tone.write_frame(&frame).await.unwrap();
                sequence += 1;
            }
            tone.stop().await.unwrap();
        });
        let mut reference = PulsePcmCapture::spawn(&PulsePcmCommand::capture(
            microphone,
            "synthetic-original-input-reference",
        ))
        .unwrap();
        let mut reference_rms = Vec::new();
        for sequence in 0..40 {
            let frame = tokio::time::timeout(
                Duration::from_secs(2),
                reference.read_frame(sequence, sequence * 20_000_000),
            )
            .await
            .unwrap()
            .unwrap();
            if sequence >= 20 {
                let energy = frame
                    .pcm()
                    .chunks_exact(2)
                    .map(|value| f64::from(i16::from_le_bytes([value[0], value[1]])).powi(2))
                    .sum::<f64>();
                reference_rms.push((energy / (frame.pcm().len() / 2) as f64).sqrt());
            }
        }
        reference.stop().await.unwrap();
        let reference_mean = reference_rms.iter().sum::<f64>() / reference_rms.len() as f64;
        eprintln!(
            "PRIVATE_ORIGINAL_INPUT rms={reference_mean:.3} min={:.3} max={:.3}",
            reference_rms.iter().copied().reduce(f64::min).unwrap(),
            reference_rms.iter().copied().reduce(f64::max).unwrap()
        );
        let capture = PulsePcmCapture::spawn(&PulsePcmCommand::capture(
            &format!("{MIC_OUT_SINK}.monitor"),
            "synthetic-original-observer",
        ))
        .unwrap();
        let phase = Arc::new(AtomicU8::new(0));
        let observations = Arc::new(Mutex::new(Vec::<(u8, f64, i32)>::new()));
        let capture_phase = phase.clone();
        let capture_done = done.clone();
        let capture_observations = observations.clone();
        let capture_task = tokio::spawn(async move {
            let mut capture = capture;
            let mut sequence = 0;
            while !capture_done.load(Ordering::Acquire) {
                let frame = tokio::time::timeout(
                    Duration::from_secs(2),
                    capture.read_frame(sequence, sequence * 20_000_000),
                )
                .await
                .unwrap()
                .unwrap();
                let samples: Vec<i32> = frame
                    .pcm()
                    .chunks_exact(2)
                    .map(|value| i16::from_le_bytes([value[0], value[1]]) as i32)
                    .collect();
                let peak = samples.iter().map(|value| value.abs()).max().unwrap();
                let rms = (samples
                    .iter()
                    .map(|value| f64::from(*value).powi(2))
                    .sum::<f64>()
                    / samples.len() as f64)
                    .sqrt();
                capture_observations.lock().unwrap().push((
                    capture_phase.load(Ordering::Acquire),
                    rms,
                    peak,
                ));
                sequence += 1;
            }
            capture.stop().await.unwrap();
        });
        let capture_ready = Instant::now() + Duration::from_secs(2);
        loop {
            let outputs = std::process::Command::new("pactl")
                .args(["--format=json", "list", "source-outputs"])
                .output()
                .unwrap();
            assert!(outputs.status.success());
            let values: serde_json::Value = serde_json::from_slice(&outputs.stdout).unwrap();
            if values
                .as_array()
                .unwrap()
                .iter()
                .any(|stream| stream["properties"]["media.name"] == "synthetic-original-observer")
            {
                break;
            }
            assert!(
                Instant::now() < capture_ready,
                "continuous observer must be connected before bridge creation"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let before_connect_frames = observations.lock().unwrap().len();
        originals.ensure_without_new_mic(&snapshot).unwrap();
        assert!(
            registry.current().unwrap().is_none(),
            "background refresh cannot start raw capture"
        );
        originals.prepare_for_start(&snapshot).unwrap();
        originals.verify_existing(&snapshot, true).unwrap();
        let first = registry.current().unwrap().unwrap();
        let zero_window = Instant::now() + Duration::from_secs(2);
        while observations.lock().unwrap().len() < before_connect_frames + 20
            && Instant::now() < zero_window
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(observations.lock().unwrap().len() >= before_connect_frames + 20);
        assert!(
            observations
                .lock()
                .unwrap()
                .iter()
                .filter(|(p, _, _)| *p == 0)
                .all(|(_, _, peak)| *peak == 0),
            "initial zero must apply before the first microphone frame"
        );
        assert!(
            first.is_live(),
            "zero-gain bridge must remain live before gain admission"
        );
        originals.verify_existing(&snapshot, true).unwrap();
        let mut translation_playbacks = Vec::new();
        for (device, name) in [
            (MIC_OUT_SINK, translator_audio::OUTGOING_TRANSLATION_STREAM),
            (
                "translator_test_out",
                translator_audio::INCOMING_TRANSLATION_STREAM,
            ),
        ] {
            let mut playback =
                PulsePcmPlayback::spawn(&PulsePcmCommand::playback(device, name)).unwrap();
            let identity = playback
                .wait_registered_muted(Instant::now() + Duration::from_secs(2))
                .await
                .unwrap();
            translation_playbacks.push((playback, identity));
        }
        let desired = AudioMixState {
            microphone_translation_percent: 75,
            speaker_translation_percent: 63,
            ..AudioMixState::default()
        };
        for (stage, percent) in [(1, 100), (2, 35), (3, 0)] {
            mix.apply_desired(
                AudioMixState {
                    microphone_original_percent: percent,
                    ..desired
                },
                TranslationMixMode::Translating,
            )
            .unwrap();
            assert_eq!(
                mix.committed().unwrap().microphone_original_percent,
                percent
            );
            phase.store(stage, Ordering::Release);
            tokio::time::sleep(Duration::from_millis(950)).await;
            originals.ensure_without_new_mic(&snapshot).unwrap();
            assert!(
                first.same_session(&registry.current().unwrap().unwrap()),
                "gain must not respawn acquisition"
            );
            let inputs = std::process::Command::new("pactl")
                .args(["--format=json", "list", "sink-inputs"])
                .output()
                .unwrap();
            let values: serde_json::Value = serde_json::from_slice(&inputs.stdout).unwrap();
            for ((_, identity), expected) in translation_playbacks.iter().zip([75_u32, 63]) {
                let stream = values
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|stream| stream["index"] == identity.index())
                    .unwrap();
                for channel in stream["volume"].as_object().unwrap().values() {
                    assert!(
                        channel["value"]
                            .as_u64()
                            .unwrap()
                            .abs_diff(u64::from((expected * 65_536 + 50) / 100))
                            <= 1
                    );
                }
            }
        }
        let settled = |stage| {
            observations
                .lock()
                .unwrap()
                .iter()
                .filter(|(phase, _, _)| *phase == stage)
                .skip(12)
                .take(20)
                .copied()
                .collect::<Vec<_>>()
        };
        let full = settled(1);
        let partial = settled(2);
        let silent = settled(3);
        assert_eq!(full.len(), 20);
        assert_eq!(partial.len(), 20);
        assert_eq!(silent.len(), 20);
        let mean = |samples: &[(u8, f64, i32)]| {
            samples.iter().map(|(_, rms, _)| *rms).sum::<f64>() / samples.len() as f64
        };
        let expected = libpulse_binding::volume::VolumeLinear::from(
            libpulse_binding::volume::Volume((35 * 65_536 + 50) / 100),
        )
        .0;
        eprintln!(
            "PRIVATE_ORIGINAL_PCM full_rms={:.3} min={:.3} max={:.3} partial_rms={:.3} zero_peak={} expected_ratio={:.8}",
            mean(&full),
            full.iter()
                .map(|(_, rms, _)| *rms)
                .reduce(f64::min)
                .unwrap(),
            full.iter()
                .map(|(_, rms, _)| *rms)
                .reduce(f64::max)
                .unwrap(),
            mean(&partial),
            silent.iter().map(|(_, _, peak)| *peak).max().unwrap(),
            expected
        );
        assert!(
            mean(&full) > 5_000.0,
            "nonzero signal must prove a connected microphone path"
        );
        assert!(
            full.iter()
                .all(|(_, rms, _)| (rms / reference_mean - 1.0).abs() < 0.10),
            "every settled full-gain frame must preserve the continuous input signal"
        );
        assert!(
            partial
                .iter()
                .all(|(_, rms, _)| (rms / reference_mean / expected - 1.0).abs() < 0.10),
            "every settled partial-gain frame must preserve the scaled input signal"
        );
        assert!(
            (mean(&partial) / mean(&full) / expected - 1.0).abs() < 0.10,
            "original gain must follow Pulse's amplitude mapping"
        );
        assert!(silent.iter().all(|(_, _, peak)| *peak == 0));
        mix.apply_desired(
            AudioMixState {
                microphone_original_percent: 35,
                ..desired
            },
            TranslationMixMode::Translating,
        )
        .unwrap();
        let incoming_observer = PulsePcmCapture::spawn(&PulsePcmCommand::capture(
            "translator_test_out.monitor",
            "synthetic-concurrent-incoming-observer",
        ))
        .unwrap();
        let translated_frame = |sequence, frequency: f64| {
            PcmFrame::try_new(
                sequence,
                sequence * 20_000_000,
                StreamPcmFormat::provider_default(),
                (0..320)
                    .flat_map(|index| {
                        ((8_000.0
                            * (std::f64::consts::TAU * frequency * index as f64 / 16_000.0).sin())
                            as i16)
                            .to_le_bytes()
                    })
                    .collect(),
            )
            .unwrap()
        };
        let translation_indices = [
            translation_playbacks[0].1.index(),
            translation_playbacks[1].1.index(),
        ];
        let assert_translation_gains = |percents: [u32; 2]| {
            let inputs = std::process::Command::new("pactl")
                .args(["--format=json", "list", "sink-inputs"])
                .output()
                .unwrap();
            assert!(inputs.status.success());
            let values: serde_json::Value = serde_json::from_slice(&inputs.stdout).unwrap();
            for (index, percent) in translation_indices.iter().zip(percents) {
                let stream = values
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|stream| stream["index"] == *index)
                    .unwrap();
                assert!(
                    stream["volume"]
                        .as_object()
                        .unwrap()
                        .values()
                        .all(|channel| channel["value"]
                            .as_u64()
                            .unwrap()
                            .abs_diff(u64::from((percent * 65_536 + 50) / 100))
                            <= 1)
                );
            }
        };
        let translated_done = Arc::new(AtomicBool::new(false));
        let incoming_frequency = Arc::new(AtomicU32::new(1_500));
        let translated_sequence = Arc::new(AtomicU64::new(0));
        let writer_done = translated_done.clone();
        let writer_frequency = incoming_frequency.clone();
        let writer_sequence = translated_sequence.clone();
        let translated_task = tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(20));
            let mut sequence = 0;
            while !writer_done.load(Ordering::Acquire) {
                tick.tick().await;
                for ((playback, _), frequency) in translation_playbacks
                    .iter_mut()
                    .zip([1_000.0, f64::from(writer_frequency.load(Ordering::Acquire))])
                {
                    playback
                        .write_frame(&translated_frame(sequence, frequency))
                        .await
                        .unwrap();
                }
                sequence += 1;
                writer_sequence.store(sequence, Ordering::Release);
            }
            translation_playbacks
        });
        let incoming_observations = Arc::new(Mutex::new(Vec::new()));
        let observer_done = Arc::new(AtomicBool::new(false));
        let observer_stop = observer_done.clone();
        let incoming_samples = incoming_observations.clone();
        let incoming_task = tokio::spawn(async move {
            let mut capture = incoming_observer;
            let mut sequence = 0;
            while !observer_stop.load(Ordering::Acquire) {
                let frame = tokio::time::timeout(
                    Duration::from_secs(2),
                    capture.read_frame(sequence, sequence * 20_000_000),
                )
                .await
                .unwrap()
                .unwrap();
                let samples: Vec<f64> = frame
                    .pcm()
                    .chunks_exact(2)
                    .map(|v| f64::from(i16::from_le_bytes([v[0], v[1]])))
                    .collect();
                let rms =
                    (samples.iter().map(|v| v * v).sum::<f64>() / samples.len() as f64).sqrt();
                let amplitude = |frequency: f64| {
                    let (sin, cos) = samples.iter().enumerate().fold(
                        (0.0, 0.0),
                        |(sin, cos), (index, value)| {
                            let angle = std::f64::consts::TAU * frequency * index as f64 / 16_000.0;
                            (sin + value * angle.sin(), cos + value * angle.cos())
                        },
                    );
                    2.0 * sin.hypot(cos) / samples.len() as f64
                };
                incoming_samples.lock().unwrap().push((
                    Instant::now(),
                    rms,
                    amplitude(1_500.0),
                    amplitude(2_000.0),
                ));
                sequence += 1;
            }
            capture.stop().await.unwrap();
        });
        for (stage, percent) in [(6, 100), (8, 35), (9, 0), (10, 35)] {
            let boundary = Instant::now();
            let before_sequence = translated_sequence.load(Ordering::Acquire);
            mix.apply_desired(
                AudioMixState {
                    microphone_original_percent: percent,
                    ..desired
                },
                TranslationMixMode::Translating,
            )
            .unwrap();
            phase.store(stage, Ordering::Release);
            tokio::time::sleep(Duration::from_millis(950)).await;
            originals.verify_existing(&snapshot, true).unwrap();
            assert!(first.same_session(&registry.current().unwrap().unwrap()));
            assert_translation_gains([75, 63]);
            assert_eq!(
                mix.committed().unwrap().microphone_original_percent,
                percent
            );
            assert!(translated_sequence.load(Ordering::Acquire) >= before_sequence + 40);
            let outgoing = settled(stage);
            assert_eq!(outgoing.len(), 20);
            assert!(
                outgoing.iter().all(|(_, rms, _)| *rms > 1_500.0),
                "translated outgoing PCM must continue even when raw gain is zero"
            );
            let incoming = incoming_observations.lock().unwrap();
            let window: Vec<_> = incoming
                .iter()
                .filter(|(at, _, _, _)| *at >= boundary + Duration::from_millis(400))
                .collect();
            assert!(window.len() >= 20);
            assert!(window.iter().all(|(_, rms, _, _)| *rms > 1_000.0));
        }

        let mut disabled = snapshot.clone();
        disabled
            .directions
            .iter_mut()
            .find(|direction| direction.direction_id == translator_core::AudioDirection::Microphone)
            .unwrap()
            .enabled = false;
        originals.ensure_without_new_mic(&disabled).unwrap();
        mix.reconcile_committed(TranslationMixMode::TranslatingMicrophoneMuted)
            .unwrap();
        assert!(
            registry.current().unwrap().is_none() && !first.is_live(),
            "healthy positive-gain disable must join acquisition before acknowledgement"
        );
        assert_eq!(
            mix.committed().unwrap().microphone_original_percent,
            35,
            "disable retains desired gain"
        );
        assert_translation_gains([0, 63]);
        let joined_at = Instant::now();
        incoming_frequency.store(2_000, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(950)).await;
        let fresh_incoming = incoming_observations
            .lock()
            .unwrap()
            .iter()
            .filter(|(at, _, _, fresh)| *at > joined_at && *fresh > 1_000.0)
            .count();
        assert!(
            fresh_incoming >= 20,
            "post-join unique incoming tone must arrive; buffered pre-disable PCM cannot pass"
        );
        let disabled_boundary = observations.lock().unwrap().len();
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(observations.lock().unwrap().len() >= disabled_boundary + 20);
        assert!(
            observations.lock().unwrap()[disabled_boundary..]
                .iter()
                .all(|(_, _, peak)| *peak == 0)
        );
        let reenable_boundary = observations.lock().unwrap().len();
        phase.store(7, Ordering::Release);
        originals.prepare_for_start(&snapshot).unwrap();
        let reenabled = registry.current().unwrap().unwrap();
        assert_ne!(first.session_id(), reenabled.session_id());
        tokio::time::sleep(Duration::from_millis(950)).await;
        {
            let reenabled_observations = observations.lock().unwrap();
            let reenabled_zero = &reenabled_observations[reenable_boundary..];
            assert!(reenabled_zero.len() >= 20);
            assert!(
                reenabled_zero.iter().all(|(_, _, peak)| *peak == 0),
                "healthy re-enable starts at zero, without old PCM"
            );
        }
        translated_done.store(true, Ordering::Release);
        let mut translation_playbacks =
            tokio::time::timeout(Duration::from_secs(2), translated_task)
                .await
                .unwrap()
                .unwrap();
        observer_done.store(true, Ordering::Release);
        tokio::time::timeout(Duration::from_secs(2), incoming_task)
            .await
            .unwrap()
            .unwrap();
        let first = reenabled;
        mix.apply_desired(
            AudioMixState {
                microphone_original_percent: 35,
                ..desired
            },
            TranslationMixMode::Translating,
        )
        .unwrap();
        eprintln!(
            "PRIVATE_ORIGINAL_CONCURRENCY outgoing_pcm=true incoming_pcm=true gain_independent=true healthy_disable_joined=true incoming_survived=true fresh_zero_reenable=true PASS"
        );
        assert!(
            !std::process::Command::new("pactl")
                .args([
                    "move-source-output",
                    &first.capture_index().to_string(),
                    &format!("{REMOTE_IN_SINK}.monitor")
                ])
                .output()
                .unwrap()
                .status
                .success()
        );
        assert!(
            !std::process::Command::new("pactl")
                .args([
                    "move-sink-input",
                    &first.playback_index().to_string(),
                    "translator_test_out"
                ])
                .output()
                .unwrap()
                .status
                .success()
        );
        originals.verify_existing(&snapshot, true).unwrap();
        struct AcknowledgeWithoutZero(u32);
        mix.reconcile_committed(TranslationMixMode::TranslatingMicrophoneMuted)
            .unwrap();
        translation_playbacks[1].0.stop().await.unwrap();
        let mut incoming = PulsePcmPlayback::spawn(&PulsePcmCommand::playback(
            "translator_test_out",
            translator_audio::INCOMING_TRANSLATION_STREAM,
        ))
        .unwrap();
        let incoming_registration = incoming
            .wait_registered_muted(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        translation_playbacks[1] = (incoming, incoming_registration);
        mix.admit_registered(
            &translation_playbacks[1].1,
            translator_daemon::PlaybackRegistrationPhase::Running,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        assert!(
            mix.admit_registered(
                &translation_playbacks[0].1,
                translator_daemon::PlaybackRegistrationPhase::Running,
                Instant::now() + Duration::from_secs(1)
            )
            .is_err(),
            "disabled microphone cannot re-admit outgoing translated playback"
        );
        impl CommandRunner for AcknowledgeWithoutZero {
            fn run_until(
                &self,
                program: &str,
                args: &[String],
                deadline: Instant,
            ) -> Result<CommandResult, CommandRunError> {
                if args
                    == [
                        "set-sink-input-volume".to_owned(),
                        self.0.to_string(),
                        "0%".to_owned(),
                    ]
                {
                    Ok(CommandResult::success(Vec::new()))
                } else {
                    SystemCommandRunner.run_until(program, args, deadline)
                }
            }
        }
        let fault_mix = AudioMixApplication::with_original_microphone(
            AcknowledgeWithoutZero(first.playback_index()),
            registry.clone(),
        );
        fault_mix
            .apply_desired(
                AudioMixState {
                    microphone_original_percent: 35,
                    ..desired
                },
                TranslationMixMode::Translating,
            )
            .unwrap();
        assert_eq!(
            fault_mix
                .reconcile_committed(TranslationMixMode::Quarantine {
                    mic_original_expected: true
                })
                .unwrap_err()
                .code,
            "audio_mix_state_unknown"
        );
        assert!(
            !first.is_live(),
            "unverified zero must cancel actual raw forwarding"
        );
        assert!(
            registry.current().is_err(),
            "cancel must not pretend unjoined custody is released"
        );
        originals
            .cleanup_all_until(Instant::now() + Duration::from_secs(1))
            .unwrap();
        assert!(
            registry.current().unwrap().is_none(),
            "joined cleanup permits explicit recovery"
        );
        fault_mix
            .recover_committed(TranslationMixMode::Quarantine {
                mic_original_expected: false,
            })
            .unwrap();
        assert_eq!(
            fault_mix.committed().unwrap().microphone_original_percent,
            35
        );
        phase.store(5, Ordering::Release);
        let silence_observation_deadline = Instant::now() + Duration::from_secs(3);
        while observations
            .lock()
            .unwrap()
            .iter()
            .filter(|(stage, _, _)| *stage == 5)
            .count()
            < 32
            && Instant::now() < silence_observation_deadline
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let cancelled_silent = settled(5);
        assert_eq!(cancelled_silent.len(), 20);
        assert!(
            cancelled_silent.iter().all(|(_, _, peak)| *peak == 0),
            "unknown raw gain cannot keep forwarding after cancellation/join"
        );
        let mut disabled = snapshot.clone();
        disabled
            .directions
            .iter_mut()
            .find(|direction| direction.direction_id == translator_core::AudioDirection::Microphone)
            .unwrap()
            .enabled = false;
        originals.ensure_without_new_mic(&disabled).unwrap();
        assert!(registry.current().unwrap().is_none());
        assert!(!first.is_live());
        assert!(
            mix.apply_desired(
                AudioMixState {
                    microphone_original_percent: 35,
                    ..desired
                },
                TranslationMixMode::Translating
            )
            .is_err()
        );
        originals.prepare_for_start(&snapshot).unwrap();
        let replacement = registry.current().unwrap().unwrap();
        assert_ne!(first.session_id(), replacement.session_id());
        phase.store(4, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(950)).await;
        let replacement_silent = settled(4);
        assert_eq!(replacement_silent.len(), 20);
        assert!(
            replacement_silent.iter().all(|(_, _, peak)| *peak == 0),
            "replacement starts silent without replaying old PCM"
        );
        for mode in [OutputMode::UnknownUnsafe, OutputMode::OpenSpeaker] {
            let mut unsafe_snapshot = snapshot.clone();
            unsafe_snapshot.devices.as_mut().unwrap().acoustic.mode = mode;
            originals.ensure_without_new_mic(&unsafe_snapshot).unwrap();
            assert!(
                registry.current().unwrap().is_none(),
                "unsafe output must disconnect an existing raw path"
            );
            originals.prepare_for_start(&snapshot).unwrap();
        }
        originals.cleanup_all().unwrap();
        assert!(registry.current().unwrap().is_none());
        done.store(true, Ordering::Release);
        tokio::time::timeout(Duration::from_secs(2), tone_task)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), capture_task)
            .await
            .unwrap()
            .unwrap();
        for (mut playback, _) in translation_playbacks {
            playback.stop().await.unwrap();
        }
        assert!(!replacement.is_live());

        let failed_registry = OriginalMicrophoneRegistry::default();
        let mut failed = translator_audio::PulseOriginalMicrophone::new(failed_registry.clone());
        assert!(
            failed
                .prepare(
                    microphone,
                    snapshot
                        .devices
                        .as_ref()
                        .unwrap()
                        .source
                        .selected
                        .as_ref()
                        .unwrap()
                        .id,
                    endpoint("sinks", MIC_OUT_SINK) + 100_000,
                    Instant::now() + Duration::from_secs(2)
                )
                .is_err()
        );
        failed
            .stop(Instant::now() + Duration::from_secs(1))
            .unwrap();
        assert!(
            failed_registry.current().unwrap().is_none(),
            "partial startup must release joined custody"
        );

        for name in ["translator_test_mic", MIC_OUT_SINK] {
            let mut current = snapshot.clone();
            current
                .devices
                .as_mut()
                .unwrap()
                .source
                .selected
                .as_mut()
                .unwrap()
                .id = endpoint("sources", microphone);
            originals.prepare_for_start(&current).unwrap();
            let before_removal = registry.current().unwrap().unwrap();
            let inventory = std::process::Command::new("pactl")
                .args(["--format=json", "list", "sinks"])
                .output()
                .unwrap();
            let values: serde_json::Value = serde_json::from_slice(&inventory.stdout).unwrap();
            let module = &values
                .as_array()
                .unwrap()
                .iter()
                .find(|value| value["name"] == name)
                .unwrap()["owner_module"];
            let module = module
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| module.as_u64().unwrap().to_string());
            assert!(
                std::process::Command::new("pactl")
                    .args(["unload-module", &module])
                    .status()
                    .unwrap()
                    .success()
            );
            let expiry = Instant::now() + Duration::from_secs(1);
            while before_removal.is_live() && Instant::now() < expiry {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(
                !before_removal.is_live(),
                "device loss must invalidate the lease without another command"
            );
            assert!(registry.current().is_err());
            assert!(originals.ensure_without_new_mic(&current).is_err());
            assert!(registry.current().unwrap().is_none());
            assert!(
                std::process::Command::new("pactl")
                    .args([
                        "load-module",
                        "module-null-sink",
                        &format!("sink_name={name}"),
                        "rate=48000",
                        "channels=1"
                    ])
                    .output()
                    .unwrap()
                    .status
                    .success()
            );
            if name == "translator_test_mic" {
                assert!(
                    originals.prepare_for_start(&current).is_err(),
                    "reused name cannot preserve the old source identity"
                );
                current
                    .devices
                    .as_mut()
                    .unwrap()
                    .source
                    .selected
                    .as_mut()
                    .unwrap()
                    .id = endpoint("sources", microphone);
            }
            originals.prepare_for_start(&current).unwrap();
            let recreated = registry.current().unwrap().unwrap();
            assert_ne!(before_removal.session_id(), recreated.session_id());
            originals.cleanup_all().unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires a disposable private PulseAudio socket and virtual fixture sinks"]
    async fn private_pulse_native_z_composed_shutdown_releases_cancelled_custody() {
        let server = std::env::var("PULSE_SERVER").expect("private PULSE_SERVER required");
        assert!(
            server.starts_with("unix:/tmp/translator-loopback-") && server.ends_with("/native")
        );
        fn inventory(kind: &str) -> Vec<serde_json::Value> {
            let output = std::process::Command::new("pactl")
                .args(["--format=json", "list", kind])
                .output()
                .unwrap();
            assert!(output.status.success());
            serde_json::from_slice(&output.stdout).unwrap()
        }
        let endpoint = |kind: &str, name: &str| -> u32 {
            inventory(kind)
                .iter()
                .find(|item| item["name"] == name)
                .unwrap()["index"]
                .as_u64()
                .unwrap() as u32
        };
        // Replace only the runner's unowned fixture endpoints, never host devices.
        for name in [MIC_OUT_SINK, REMOTE_IN_SINK] {
            let sinks = inventory("sinks");
            let sink = sinks.iter().find(|item| item["name"] == name).unwrap();
            assert!(sink["properties"]["translator.owner"].is_null());
            let module = sink["owner_module"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| sink["owner_module"].as_u64().unwrap().to_string());
            assert!(
                std::process::Command::new("pactl")
                    .args(["unload-module", &module])
                    .status()
                    .unwrap()
                    .success()
            );
        }
        let foreign_modules = || {
            let output = std::process::Command::new("pactl")
                .args(["list", "short", "modules"])
                .output()
                .unwrap();
            assert!(output.status.success());
            output.stdout
        };
        let fixture_modules = [
            [
                "load-module",
                "module-null-sink",
                "sink_name=private_fixture_output",
                "channels=1",
                "channel_map=mono",
            ],
            [
                "load-module",
                "module-remap-source",
                "master=translator_test_mic.monitor",
                "source_name=private_fixture_capture",
                "channels=1",
            ],
        ]
        .map(|arguments| {
            let output = std::process::Command::new("pactl")
                .args(arguments)
                .output()
                .unwrap();
            assert!(output.status.success());
            String::from_utf8(output.stdout)
                .unwrap()
                .trim()
                .parse::<u32>()
                .unwrap()
        });
        let foreign_before = foreign_modules();

        struct PrivateFacts {
            routes: Arc<super::PulseManualRoutes>,
            devices: translator_audio::DeviceFacts,
        }
        impl translator_daemon::RuntimeFactsSource for PrivateFacts {
            fn inspect(
                &self,
                deadline: Instant,
            ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError>
            {
                self.routes
                    .resources
                    .with_active(|resources| {
                        super::inspect_runtime_graph_facts(
                            self.devices.clone(),
                            resources.graph.as_ref().unwrap(),
                            &resources.routing,
                            deadline,
                        )
                    })
                    .ok_or(translator_daemon::FactsError::DiscoveryFailed)?
            }
        }
        impl translator_daemon::RuntimeMaintenance for PrivateFacts {
            fn refresh(
                &self,
                store: &RuntimeStore,
            ) -> Result<(), translator_daemon::ControlFailure> {
                let facts = translator_daemon::RuntimeFactsSource::inspect(
                    self,
                    Instant::now() + Duration::from_secs(2),
                )
                .map_err(|_| super::native_route_failure())?;
                store.set_devices(facts.devices.into());
                store.set_audio_graph(facts.audio_graph);
                store.set_routes(facts.routes);
                Ok(())
            }
            fn refresh_bypass_facts(
                &self,
                store: &RuntimeStore,
            ) -> Result<(), translator_daemon::ControlFailure> {
                self.refresh(store)
            }
            fn verify_bypass_custody(
                &self,
                snapshot: &RuntimeSnapshot,
                permit: bool,
            ) -> Result<(), translator_daemon::ControlFailure> {
                self.routes.verify_bypass_custody(snapshot, permit)
            }
            fn prepare_start(
                &self,
                snapshot: &RuntimeSnapshot,
            ) -> Result<(), translator_daemon::ControlFailure> {
                self.routes.prepare_start(snapshot)
            }
            fn prepare_bypass(
                &self,
                snapshot: &RuntimeSnapshot,
            ) -> Result<(), translator_daemon::ControlFailure> {
                self.routes.prepare_bypass(snapshot)
            }
            fn cleanup_originals(
                &self,
                deadline: Instant,
            ) -> Result<(), translator_daemon::ControlFailure> {
                self.routes.cleanup_originals(deadline)
            }
        }

        let foreign_pair = (
            endpoint("sources", "private_fixture_capture"),
            endpoint("sinks", "private_fixture_output"),
        );
        for case in ["cancelled", "running", "bypass", "offline"] {
            let mut graph = translator_audio::PulseAudioGraph::new(
                SystemCommandRunner,
                translator_audio::default_journal_path().unwrap(),
            );
            let graph_state = graph.ensure_endpoints().unwrap();
            assert_eq!(graph_state.owned_module_ids.len(), 3);
            let gate = AudioOperationGate::new();
            let registry = translator_audio::OriginalMicrophoneRegistry::default();
            let routes = Arc::new(super::PulseManualRoutes {
                resources: LifecycleProtected::new(super::PulseResources {
                    routing: translator_audio::PulseRoutingWatcher::new(
                        SystemCommandRunner,
                        translator_audio::RoutingProfile::Production,
                    ),
                    devices: translator_audio::PulseDeviceWatcher::new(
                        SystemCommandRunner,
                        AecCapability::Unavailable,
                    ),
                    original_loopbacks: PulseOriginalLoopbacks::with_microphone(
                        SystemCommandRunner,
                        registry.clone(),
                    ),
                    graph: Some(graph),
                }),
                operation_gate: gate.clone(),
            });
            let mut devices = selected_devices();
            // These are explicitly synthetic admitted facts, not physical/AEC evidence.
            let microphone = "private_fixture_capture";
            devices.source.selected =
                Some(physical_device(endpoint("sources", microphone), microphone));
            devices.source.pinned_name = Some(microphone.into());
            devices.sink.selected = Some(physical_device(
                endpoint("sinks", "private_fixture_output"),
                "private_fixture_output",
            ));
            devices.sink.pinned_name = Some("private_fixture_output".into());
            let facts = Arc::new(PrivateFacts {
                routes: routes.clone(),
                devices: translator_audio::DeviceFacts {
                    source: devices.source,
                    sink: devices.sink,
                    output_mode: OutputMode::Headphones,
                    aec_capability: AecCapability::Unavailable,
                },
            });
            let mix = Arc::new(
                translator_daemon::AudioMixApplication::with_original_microphone(
                    SystemCommandRunner,
                    registry.clone(),
                ),
            );
            let store = RuntimeStore::default();
            let state = Arc::new(DrainState::default());
            let control = ControlApplication::spawn(
                store.clone(),
                Arc::new(DrainRuntime(state.clone())),
                gate.clone(),
                facts.clone(),
                facts,
                Some(mix.clone()),
            );
            control.execute(ControlCommand::Start).await.unwrap();
            let registration = registry.current().unwrap().unwrap();
            assert!(registration.is_live());
            if case == "bypass" {
                control.execute(ControlCommand::Stop).await.unwrap();
                assert!(registration.is_live());
                let desired = store.snapshot().audio_mix;
                for _ in 0..3 {
                    control
                        .execute(ControlCommand::ReconcileAudio)
                        .await
                        .unwrap();
                    assert_eq!(store.snapshot().audio_mix, desired);
                    assert!(
                        registry
                            .current()
                            .unwrap()
                            .unwrap()
                            .same_session(&registration)
                    );
                    let raw = inventory("sink-inputs");
                    let raw = raw
                        .iter()
                        .find(|item| item["index"] == registration.playback_index())
                        .unwrap();
                    assert_eq!(raw["volume"]["mono"]["value"], 65_536);
                }
            }
            if case == "cancelled" {
                registry.cancel_current();
                assert!(!registration.is_live());
                assert!(
                    registry.current().is_err(),
                    "cancel must retain join custody"
                );
            }
            if case == "offline" {
                control.execute(ControlCommand::Stop).await.unwrap();
                routes
                    .resources
                    .with_active(|resources| resources.original_loopbacks.cleanup_all())
                    .unwrap()
                    .unwrap();
                let request = OriginalLoopbackRequest {
                    media_name: SPEAKER_ORIGINAL_LOOPBACK,
                    source: format!("{REMOTE_IN_SINK}.monitor"),
                    sink: "private_fixture_output".into(),
                };
                routes
                    .resources
                    .with_active(|resources| {
                        resources
                            .original_loopbacks
                            .load_module(&request, Instant::now() + Duration::from_secs(2))
                    })
                    .unwrap()
                    .unwrap();
                let before = foreign_modules();
                assert!(String::from_utf8_lossy(&before).contains("loopback-speaker-original"));
                assert_eq!(
                    super::run_audio_graph_cleanup(),
                    std::process::ExitCode::SUCCESS
                );
                assert_eq!(
                    foreign_modules(),
                    foreign_before,
                    "offline cleanup must remove all four owned modules only"
                );
                assert_eq!(
                    super::run_audio_graph_cleanup(),
                    std::process::ExitCode::SUCCESS
                );
            }
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let observed_gate = gate.clone();
            let server = tokio::spawn(async move {
                stopped.await.unwrap();
                assert_eq!(observed_gate.state(), AudioOperationState::Stopping);
                Ok(())
            });
            let background = [0; 2].map(|_| tokio::spawn(std::future::pending::<()>()));
            let result = tokio::time::timeout(
                translator_daemon::RUNTIME_CLEANUP_BUDGET + Duration::from_secs(1),
                super::drain_control_owners(
                    &gate,
                    (stop, server),
                    background,
                    None,
                    Some(&control),
                    &store,
                ),
            )
            .await
            .expect("composed drain exceeded original cleanup budget");
            assert!(
                result.0.is_ok() && result.1.is_ok() && result.2.is_ok() && result.3.is_none(),
                "actual composed {case} drain must finish before final graph cleanup: {:?}",
                result.2
            );
            assert_eq!(gate.state(), AudioOperationState::Stopping);
            assert!(gate.acquire_production().is_err());
            assert!(
                translator_daemon::RuntimeMaintenance::refresh_bypass_facts(
                    routes.as_ref(),
                    &store
                )
                .is_err()
            );
            assert!(
                translator_daemon::RuntimeMaintenance::prepare_bypass(
                    routes.as_ref(),
                    &store.snapshot()
                )
                .is_err()
            );
            assert!(control.execute(ControlCommand::Start).await.is_err());
            assert!(
                registry.current().unwrap().is_none(),
                "joined native custody must be released by drain"
            );
            assert_eq!(
                store.snapshot().runtime_status,
                translator_daemon::RuntimeStatus::Stopped
            );
            assert_eq!(
                store.snapshot().audio_mix_knowledge,
                translator_daemon::AudioMixKnowledge::Known
            );
            assert_eq!(
                state.stop_calls.load(Ordering::SeqCst),
                1,
                "do not repeat an already completed runtime stop"
            );
            let inputs = inventory("sink-inputs");
            let outputs = inventory("source-outputs");
            assert!(
                !inputs
                    .iter()
                    .any(|item| item["index"] == registration.playback_index())
            );
            assert!(
                !outputs
                    .iter()
                    .any(|item| item["index"] == registration.capture_index())
            );
            let raw_inputs: Vec<RawPulseStream> =
                serde_json::from_value(serde_json::Value::Array(inputs)).unwrap();
            let raw_outputs: Vec<RawPulseStream> =
                serde_json::from_value(serde_json::Value::Array(outputs)).unwrap();
            assert!(
                discover_original_loopbacks(&raw_inputs, &raw_outputs)
                    .unwrap()
                    .is_empty(),
                "graph remap streams may remain, but no owned originals may survive drain"
            );
            translator_daemon::ManualRouteController::restore(routes.as_ref()).unwrap();
            routes.cleanup_graph();
            assert_eq!(
                foreign_modules(),
                foreign_before,
                "only owned graph modules may disappear"
            );
            assert_eq!(
                (
                    endpoint("sources", microphone),
                    endpoint("sinks", "private_fixture_output")
                ),
                foreign_pair
            );
            println!(
                "PRIVATE_COMPOSED_SHUTDOWN case={case} joined=true gate=stopping owned_cleanup=true PASS"
            );
        }
        for module in fixture_modules.into_iter().rev() {
            assert!(
                std::process::Command::new("pactl")
                    .args(["unload-module", &module.to_string()])
                    .status()
                    .unwrap()
                    .success()
            );
        }
    }

    #[test]
    fn original_loopback_load_args_quote_nested_owner_properties() {
        let request = OriginalLoopbackRequest {
            media_name: SPEAKER_ORIGINAL_LOOPBACK,
            source: format!("{REMOTE_IN_SINK}.monitor"),
            sink: "alsa_output.headphones".to_owned(),
        };

        let args = original_loopback_load_args(&request);

        assert_eq!(args[0], "load-module");
        assert_eq!(args[1], "module-loopback");
        assert!(args.contains(&format!("source={REMOTE_IN_SINK}.monitor")));
        assert!(args.contains(&"sink=alsa_output.headphones".to_owned()));
        assert!(args.contains(&"latency_msec=20".to_owned()));
        assert!(args.contains(
            &"source_output_properties='media.name=loopback-speaker-original translator.owner=true'"
                .to_owned()
        ));
        assert!(
            args.contains(
                &"sink_input_properties='media.name=loopback-speaker-original translator.owner=true'"
                    .to_owned()
            )
        );
    }

    #[test]
    fn original_loopback_discovery_matches_sink_and_source_targets_by_module() {
        let sink_inputs = [raw_sink_stream(SPEAKER_ORIGINAL_LOOPBACK, "42", 0)];
        let source_outputs = [raw_source_stream(SPEAKER_ORIGINAL_LOOPBACK, "42", 1)];
        let request = OriginalLoopbackRequest {
            media_name: SPEAKER_ORIGINAL_LOOPBACK,
            source: format!("{REMOTE_IN_SINK}.monitor"),
            sink: "alsa_output.headphones".to_owned(),
        };

        let discovered = discover_original_loopbacks(&sink_inputs, &source_outputs).unwrap();

        assert_eq!(
            discovered.get("42"),
            Some(&DiscoveredOriginalLoopback {
                media_name: SPEAKER_ORIGINAL_LOOPBACK,
                source_index: Some(1),
                sink_index: Some(0),
            })
        );
        assert_eq!(
            matching_original_loopbacks(
                &discovered,
                &request,
                &HashMap::from([(1, format!("{REMOTE_IN_SINK}.monitor"))]),
                &HashMap::from([(0, "alsa_output.headphones".to_owned())]),
            )
            .unwrap(),
            ["42"]
        );
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

    fn raw_sink_stream(media_name: &str, module_id: &str, sink: u32) -> RawPulseStream {
        RawPulseStream {
            owner_module: Some(RawPulseModuleId::Text(module_id.to_owned())),
            source: None,
            sink: Some(sink),
            properties: HashMap::from([
                ("media.name".to_owned(), media_name.to_owned()),
                ("translator.owner".to_owned(), "true".to_owned()),
            ]),
        }
    }

    fn raw_source_stream(media_name: &str, module_id: &str, source: u32) -> RawPulseStream {
        let mut stream = raw_sink_stream(media_name, module_id, 0);
        stream.source = Some(source);
        stream.sink = None;
        stream
    }

    #[derive(Default)]
    struct DrainState {
        stop_failures: AtomicUsize,
        recovery_failures: AtomicUsize,
        unknown: AtomicBool,
        failed: tokio::sync::Notify,
        attempts: Mutex<Vec<(&'static str, Instant)>>,
        stop_deadlines: Mutex<Vec<tokio::time::Instant>>,
        stop_calls: AtomicUsize,
        mix_modes: Mutex<Vec<translator_daemon::TranslationMixMode>>,
    }

    #[derive(Clone)]
    struct DrainRuntime(Arc<DrainState>);

    impl DrainRuntime {
        fn attempt(
            &self,
            boundary: &'static str,
            failures: &AtomicUsize,
        ) -> Result<(), translator_daemon::ControlFailure> {
            self.0
                .attempts
                .lock()
                .unwrap()
                .push((boundary, Instant::now()));
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
            deadline: tokio::time::Instant,
        ) -> Result<(), translator_daemon::DuplexRuntimeError> {
            self.0.stop_calls.fetch_add(1, Ordering::SeqCst);
            self.0.stop_deadlines.lock().unwrap().push(deadline);
            if self.0.unknown.load(Ordering::SeqCst) {
                return Ok(());
            }
            self.attempt("stop", &self.0.stop_failures)
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

        fn refresh_bypass_facts(
            &self,
            store: &translator_daemon::RuntimeStore,
        ) -> Result<(), translator_daemon::ControlFailure> {
            let facts = translator_daemon::RuntimeFactsSource::inspect(
                &TestFacts,
                Instant::now() + Duration::from_secs(1),
            )
            .unwrap();
            store.set_devices(facts.devices.into());
            store.set_audio_graph(facts.audio_graph);
            store.set_routes(facts.routes);
            Ok(())
        }

        fn verify_bypass_custody(
            &self,
            _: &translator_daemon::RuntimeSnapshot,
            _: bool,
        ) -> Result<(), translator_daemon::ControlFailure> {
            Ok(())
        }

        fn prepare_start(
            &self,
            _: &translator_daemon::RuntimeSnapshot,
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
            mode: translator_daemon::TranslationMixMode,
        ) -> Result<(), translator_daemon::ControlFailure> {
            self.0.mix_modes.lock().unwrap().push(mode);
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
            mode: translator_daemon::TranslationMixMode,
        ) -> Result<(), translator_daemon::ControlFailure> {
            self.0.mix_modes.lock().unwrap().push(mode);
            self.attempt("recovery", &self.0.recovery_failures)?;
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
            super::drain_translation(&owner).await.unwrap();
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
        assert_eq!(state.stop_calls.load(Ordering::SeqCst), stop_failures + 1);
        assert_eq!(
            state.stop_deadlines.lock().unwrap().len(),
            stop_failures + 1
        );
        let attempts = state.attempts.lock().unwrap();
        for (boundary, count) in [
            (
                "stop",
                if recovery_failures > 0 {
                    0
                } else {
                    stop_failures + 1
                },
            ),
            ("recovery", stop_failures + recovery_failures + 1),
        ] {
            let times: Vec<_> = attempts
                .iter()
                .filter_map(|(observed, at)| (*observed == boundary).then_some(*at))
                .collect();
            assert_eq!(times.len(), count, "{boundary} attempt count");
            assert!(
                times
                    .windows(2)
                    .all(|pair| pair[1].duration_since(pair[0]) >= Duration::from_secs(1)),
                "completed {boundary} failures must be paced before retry"
            );
        }
        assert!(
            state
                .stop_deadlines
                .lock()
                .unwrap()
                .windows(2)
                .all(|pair| pair[0] == pair[1]),
            "accepted runtime stop retries must use the same original absolute deadline"
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

    struct ShutdownMaintenance {
        runtime: DrainRuntime,
        gate: AudioOperationGate,
        forbidden: Arc<AtomicUsize>,
    }

    impl ShutdownMaintenance {
        fn check_bypass(&self) -> Result<(), translator_daemon::ControlFailure> {
            if self.gate.state() == AudioOperationState::Stopping {
                self.forbidden.fetch_add(1, Ordering::SeqCst);
                return Err(translator_daemon::ControlFailure {
                    status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    code: "original_loopback_custody_unknown",
                });
            }
            Ok(())
        }
    }

    impl translator_daemon::RuntimeMaintenance for ShutdownMaintenance {
        fn refresh(&self, _: &RuntimeStore) -> Result<(), translator_daemon::ControlFailure> {
            self.check_bypass()
        }

        fn refresh_bypass_facts(
            &self,
            store: &RuntimeStore,
        ) -> Result<(), translator_daemon::ControlFailure> {
            self.check_bypass()?;
            translator_daemon::RuntimeMaintenance::refresh_bypass_facts(&self.runtime, store)
        }

        fn verify_bypass_custody(
            &self,
            _: &RuntimeSnapshot,
            _: bool,
        ) -> Result<(), translator_daemon::ControlFailure> {
            self.check_bypass()
        }

        fn prepare_start(
            &self,
            _: &RuntimeSnapshot,
        ) -> Result<(), translator_daemon::ControlFailure> {
            Ok(())
        }

        fn prepare_bypass(
            &self,
            _: &RuntimeSnapshot,
        ) -> Result<(), translator_daemon::ControlFailure> {
            self.check_bypass()
        }
    }

    async fn assert_composed_shutdown_with_mix(running: bool, bypass_pending: bool) {
        let store = RuntimeStore::default();
        let gate = AudioOperationGate::new();
        let state = Arc::new(DrainState::default());
        let runtime = Arc::new(DrainRuntime(state.clone()));
        let forbidden = Arc::new(AtomicUsize::new(0));
        let control = ControlApplication::spawn(
            store.clone(),
            runtime.clone(),
            gate.clone(),
            Arc::new(TestFacts),
            Arc::new(ShutdownMaintenance {
                runtime: (*runtime).clone(),
                gate: gate.clone(),
                forbidden: forbidden.clone(),
            }),
            Some(runtime),
        );
        if running {
            control.execute(ControlCommand::Start).await.unwrap();
        }
        if bypass_pending {
            state.unknown.store(true, Ordering::SeqCst);
            control
                .execute(ControlCommand::ReconcileAudio)
                .await
                .unwrap_err();
            assert_eq!(
                store.snapshot().runtime_status,
                translator_daemon::RuntimeStatus::CleanupPending
            );
            state.unknown.store(false, Ordering::SeqCst);
        }
        state.mix_modes.lock().unwrap().clear();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let observed_gate = gate.clone();
        let server = tokio::spawn(async move {
            stopped.await.unwrap();
            assert_eq!(observed_gate.state(), AudioOperationState::Stopping);
            Ok(())
        });
        let background = [0; 2].map(|_| tokio::spawn(std::future::pending::<()>()));
        let driver_store = store.clone();
        let driver_gate = gate.clone();
        let mut driver = tokio::spawn(async move {
            super::drain_control_owners(
                &driver_gate,
                (stop, server),
                background,
                None,
                Some(&control),
                &driver_store,
            )
            .await
        });
        let completed = match tokio::time::timeout(Duration::from_millis(500), &mut driver).await {
            Ok(result) => {
                let result = result.unwrap();
                assert!(result.0.is_ok());
                assert!(result.1.is_ok());
                assert!(result.2.is_ok());
                assert!(result.3.is_none());
                true
            }
            Err(_) => {
                driver.abort();
                let _ = driver.await;
                false
            }
        };
        assert!(
            completed,
            "actual composed shutdown with a mix owner must not require reopened Production authority"
        );
        assert_eq!(gate.state(), AudioOperationState::Stopping);
        assert_eq!(
            forbidden.load(Ordering::SeqCst),
            0,
            "shutdown must not refresh or prepare bypass facts"
        );
        assert_eq!(
            store.snapshot().runtime_status,
            translator_daemon::RuntimeStatus::Stopped
        );
        assert_eq!(
            state.stop_calls.load(Ordering::SeqCst),
            usize::from(running)
        );
        assert_eq!(
            state.mix_modes.lock().unwrap().as_slice(),
            &[translator_daemon::TranslationMixMode::Quarantine {
                mic_original_expected: false
            }; 2],
            "the actual mix owner must quarantine, never restore audible bypass"
        );
    }

    #[tokio::test]
    async fn composed_shutdown_stopped_mix_never_reopens_production() {
        assert_composed_shutdown_with_mix(false, false).await;
    }

    #[tokio::test]
    async fn composed_shutdown_running_mix_does_not_restore_bypass() {
        assert_composed_shutdown_with_mix(true, false).await;
    }

    #[tokio::test]
    async fn composed_shutdown_bypass_pending_mix_releases_owned_custody() {
        assert_composed_shutdown_with_mix(false, true).await;
    }

    #[tokio::test(start_paused = true)]
    async fn permanent_round_trip_shutdown_is_bounded_by_original_budget() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let seen = attempts.clone();
        let mut driver = tokio::spawn(async move {
            super::drain_round_trip_attempts(|| {
                seen.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Err(super::RoundTripOwnerShutdownError::CleanupPending))
            })
            .await
        });
        let result = match tokio::time::timeout(
            translator_daemon::RUNTIME_CLEANUP_BUDGET + Duration::from_millis(1),
            &mut driver,
        )
        .await
        {
            Ok(result) => Some(result.unwrap()),
            Err(_) => {
                driver.abort();
                let _ = driver.await;
                None
            }
        };
        assert_eq!(
            result,
            Some(Err(super::RoundTripOwnerShutdownError::CleanupPending)),
            "permanent pending cleanup must end its retries without resetting the budget"
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 8);
    }

    #[tokio::test]
    async fn permanent_translation_shutdown_keeps_custody_and_original_deadline() {
        let store = RuntimeStore::default();
        let gate = AudioOperationGate::new();
        let state = Arc::new(DrainState::default());
        let runtime = Arc::new(DrainRuntime(state.clone()));
        let control = ControlApplication::spawn(
            store.clone(),
            runtime.clone(),
            gate.clone(),
            Arc::new(TestFacts),
            runtime.clone(),
            Some(runtime),
        );
        control.execute(ControlCommand::Start).await.unwrap();
        state.stop_failures.store(usize::MAX, Ordering::SeqCst);
        gate.begin_stopping();
        let result = tokio::time::timeout(
            translator_daemon::RUNTIME_CLEANUP_BUDGET + Duration::from_millis(500),
            super::drain_translation(&control),
        )
        .await;
        let held = store.snapshot().runtime_status;
        let deadlines = state.stop_deadlines.lock().unwrap().clone();
        let rejected = control.execute(ControlCommand::Start).await.is_err();
        // Release the actual fixture owner; this cannot upgrade the original timeout result.
        state.stop_failures.store(0, Ordering::SeqCst);
        control.shutdown().await.unwrap();
        assert_eq!(
            result.unwrap().unwrap_err().code,
            "translation_cleanup_pending"
        );
        assert_eq!(held, translator_daemon::RuntimeStatus::CleanupPending);
        assert_eq!(gate.state(), AudioOperationState::Stopping);
        assert!(rejected);
        assert_eq!(deadlines.len(), 8);
        assert!(deadlines.windows(2).all(|pair| pair[0] == pair[1]));
    }

    struct HeldRoundTripDrop {
        entered: Arc<tokio::sync::Notify>,
        released: Arc<(Mutex<bool>, std::sync::Condvar)>,
        completed: Arc<AtomicBool>,
    }

    impl Drop for HeldRoundTripDrop {
        fn drop(&mut self) {
            self.entered.notify_one();
            let released = self.released.0.lock().unwrap();
            let (released, _) = self
                .released
                .1
                .wait_timeout_while(released, Duration::from_secs(15), |released| !*released)
                .unwrap();
            self.completed.store(*released, Ordering::SeqCst);
        }
    }

    impl translator_daemon::RoundTripRunner for HeldRoundTripDrop {
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
            panic!("the shutdown fixture must not start audio");
        }
    }

    #[tokio::test]
    async fn inflight_round_trip_shutdown_timeout_still_drains_translation_and_debug() {
        let store = RuntimeStore::default();
        let capture = tempfile::tempdir().unwrap();
        store.configure_debug_capture(
            super::DebugCaptureStore::open(capture.path(), super::DebugCaptureLimits::default())
                .unwrap(),
        );
        store.set_debug_capture_enabled(true).unwrap();
        let gate = AudioOperationGate::new();
        let state = Arc::new(DrainState::default());
        let runtime = Arc::new(DrainRuntime(state.clone()));
        let control = ControlApplication::spawn(
            store.clone(),
            runtime.clone(),
            gate.clone(),
            Arc::new(TestFacts),
            runtime.clone(),
            Some(runtime),
        );
        let entered = Arc::new(tokio::sync::Notify::new());
        let released = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let completed = Arc::new(AtomicBool::new(false));
        let owner = Arc::new(
            super::RoundTripRuntimeHandle::try_new(
                store.clone(),
                Arc::new(HeldRoundTripDrop {
                    entered: entered.clone(),
                    released: released.clone(),
                    completed: completed.clone(),
                }),
                gate.clone(),
                Arc::new(TestFacts),
            )
            .unwrap(),
        );
        // An already accepted legacy join holds this exact owner's actor lock.
        // The composed call must retain its blocked drainer, not detach it at timeout.
        let legacy_owner = owner.clone();
        let legacy = tokio::task::spawn_blocking(move || legacy_owner.shutdown());
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            stopped.await.unwrap();
            Ok(())
        });
        let background = [0; 2].map(|_| tokio::spawn(std::future::pending::<()>()));
        let driver_store = store.clone();
        let driver_owner = owner.clone();
        let mut driver = tokio::spawn(async move {
            super::drain_control_owners(
                &gate,
                (stop, server),
                background,
                Some(&driver_owner),
                Some(&control),
                &driver_store,
            )
            .await
        });
        let result = tokio::time::timeout(
            translator_daemon::RUNTIME_CLEANUP_BUDGET + Duration::from_millis(500),
            &mut driver,
        )
        .await;
        let custody_held_at_timeout = !completed.load(Ordering::SeqCst);
        let mix_attempted_at_timeout = !state.mix_modes.lock().unwrap().is_empty();
        let debug_closed_at_timeout = !store.snapshot().debug_capture_enabled;
        let mut finished = match result {
            Ok(result) => Some(result.unwrap()),
            Err(_) => {
                driver.abort();
                let _ = driver.await;
                None
            }
        };
        let retained_at_timeout = finished
            .as_ref()
            .and_then(|result| result.3.as_ref())
            .is_some_and(|handle| !handle.is_finished());
        *released.0.lock().unwrap() = true;
        released.1.notify_all();
        legacy.await.unwrap().unwrap();
        if let Some(handle) = finished.as_mut().and_then(|result| result.3.take()) {
            assert_eq!(
                handle.await.unwrap(),
                Err(super::RoundTripOwnerShutdownError::CleanupPending)
            );
        }
        tokio::task::spawn_blocking(move || owner.shutdown())
            .await
            .unwrap()
            .unwrap();
        assert!(
            finished.is_some(),
            "pending accepted round-trip work must not starve independent drains forever"
        );
        assert!(
            custody_held_at_timeout,
            "the accepted native owner join must still be held at the timeout boundary"
        );
        assert!(
            mix_attempted_at_timeout,
            "the real translation mix owner must be drained after round-trip timeout"
        );
        assert!(debug_closed_at_timeout);
        assert!(
            retained_at_timeout,
            "the exact unfinished drainer handle must remain in the returned custody packet"
        );
        let finished = finished.unwrap();
        assert_eq!(
            finished.1,
            Err(super::RoundTripOwnerShutdownError::CleanupPending)
        );
        assert!(finished.2.is_ok());
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
                    result.unwrap().0,
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
        let ((server_result, owner_result, translation_result, retained), command_after_shutdown) =
            match tokio::time::timeout(Duration::from_secs(3), &mut driver).await {
                Ok(result) => result.unwrap(),
                Err(_) => {
                    driver.abort();
                    let _ = driver.await;
                    panic!("fatal owner fixture did not drain");
                }
            };
        assert!(server_result.is_ok());
        assert!(translation_result.is_ok());
        assert!(retained.is_none());
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
            let (result, retained) = super::drain_round_trip(&driver_owner).await;
            result.unwrap();
            assert!(retained.is_none());
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
