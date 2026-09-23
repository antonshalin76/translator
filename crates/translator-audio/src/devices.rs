use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::module_list::parse_module_list;
use crate::{CommandResult, CommandRunError, CommandRunner, SystemCommandRunner};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum AecCapability {
    Unavailable,
    AvailableUnvalidated,
    ValidationFailed,
    ValidatedFor {
        source_name: String,
        sink_name: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceHealth {
    Available,
    DeviceUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    Headphones,
    OpenSpeaker,
    UnknownUnsafe,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalDevice {
    pub id: u32,
    pub name: String,
    pub description: String,
    pub active_port: Option<String>,
    pub active_port_type: Option<String>,
    pub available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceSelectionState {
    pub health: DeviceHealth,
    pub selected: Option<PhysicalDevice>,
    pub pinned_name: Option<String>,
    pub current_default: Option<String>,
    pub pending_default: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFacts {
    pub source: DeviceSelectionState,
    pub sink: DeviceSelectionState,
    pub output_mode: OutputMode,
    pub aec_capability: AecCapability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task7EndpointFacts {
    pub input_monitor: String,
    pub output: PhysicalDevice,
    pub output_mode: OutputMode,
}

pub fn inspect_task7_endpoints_until(
    runner: &impl CommandRunner,
    input: &str,
    output: &str,
    deadline: Instant,
) -> Result<Task7EndpointFacts, DeviceWatcherError> {
    check_device_deadline(deadline)?;
    if input != "translator_task7_ru_in.monitor" {
        return Err(DeviceWatcherError::new(
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
        ));
    }
    let sources: Vec<RawDevice> = read_device_json_until(runner, "sources", deadline)?;
    let sinks: Vec<RawDevice> = read_device_json_until(runner, "sinks", deadline)?;
    let result = run_device_command_until(runner, &["list", "short", "modules"], deadline)?;
    let modules = parse_module_list(result.stdout())
        .map_err(|_| DeviceWatcherError::new(DeviceWatcherErrorCode::DiscoveryFailed))?;
    check_device_deadline(deadline)?;
    validate_device_identities(&sources)?;
    validate_device_identities(&sinks)?;
    let invalid = || DeviceWatcherError::new(DeviceWatcherErrorCode::InvalidPhysicalDevice);
    let source = sources
        .iter()
        .find(|source| source.name == input)
        .ok_or_else(invalid)?;
    let sink = sinks
        .iter()
        .find(|sink| sink.name == "translator_task7_ru_in")
        .ok_or_else(invalid)?;
    let module = sink
        .owner_module
        .and_then(|id| modules.get(&id))
        .ok_or_else(invalid)?;
    if source.monitor_source != sink.name
        || sink.monitor_source != source.name
        || source.owner_module != sink.owner_module
        || module.name != "module-null-sink"
        || sink
            .properties
            .get("translator.task7_e2e")
            .map(String::as_str)
            != Some("true")
        || !device_available(source)
        || !device_available(sink)
    {
        return Err(invalid());
    }
    let output = sinks
        .into_iter()
        .find(|sink| sink.name == output && is_physical_sink(sink) && device_available(sink))
        .ok_or_else(invalid)?;
    let output = PhysicalDevice::from(output);
    let output_mode = classify_output_mode(&output);
    check_device_deadline(deadline)?;
    Ok(Task7EndpointFacts {
        input_monitor: input.to_owned(),
        output,
        output_mode,
    })
}

fn check_device_deadline(deadline: Instant) -> Result<(), DeviceWatcherError> {
    if Instant::now() >= deadline {
        Err(DeviceWatcherError::new(
            DeviceWatcherErrorCode::DeadlineExpired,
        ))
    } else {
        Ok(())
    }
}

fn read_device_json_until<T: for<'de> Deserialize<'de>>(
    runner: &impl CommandRunner,
    kind: &str,
    deadline: Instant,
) -> Result<T, DeviceWatcherError> {
    let result = run_device_command_until(runner, &["--format=json", "list", kind], deadline)?;
    let value = serde_json::from_slice(result.stdout())
        .map_err(|_| DeviceWatcherError::new(DeviceWatcherErrorCode::DiscoveryFailed))?;
    check_device_deadline(deadline)?;
    Ok(value)
}

fn run_device_command_until(
    runner: &impl CommandRunner,
    args: &[&str],
    deadline: Instant,
) -> Result<CommandResult, DeviceWatcherError> {
    check_device_deadline(deadline)?;
    let args: Vec<_> = args.iter().map(|value| (*value).to_owned()).collect();
    let result = runner
        .run_until("pactl", &args, deadline)
        .map_err(|error| {
            DeviceWatcherError::new(if error == CommandRunError::DeadlineExpired {
                DeviceWatcherErrorCode::DeadlineExpired
            } else {
                DeviceWatcherErrorCode::DiscoveryFailed
            })
        })?;
    check_device_deadline(deadline)?;
    if !result.is_success() {
        return Err(DeviceWatcherError::new(
            DeviceWatcherErrorCode::DiscoveryFailed,
        ));
    }
    Ok(result)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceOverride {
    pub source_name: Option<String>,
    pub sink_name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceWatcherErrorCode {
    DiscoveryFailed,
    InvalidPhysicalDevice,
    GraphValidationFailed,
    DeadlineExpired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceWatcherSafeError {
    pub code: DeviceWatcherErrorCode,
    pub safe_message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceWatcherError {
    status: DeviceWatcherSafeError,
}

impl DeviceWatcherError {
    fn new(code: DeviceWatcherErrorCode) -> Self {
        let (safe_message, retryable) = match code {
            DeviceWatcherErrorCode::DiscoveryFailed => ("Audio device discovery failed", true),
            DeviceWatcherErrorCode::InvalidPhysicalDevice => {
                ("Selected physical audio device is unavailable", false)
            }
            DeviceWatcherErrorCode::GraphValidationFailed => {
                ("Physical audio sink validation failed", true)
            }
            DeviceWatcherErrorCode::DeadlineExpired => {
                ("Audio device inspection deadline expired", true)
            }
        };
        Self {
            status: DeviceWatcherSafeError {
                code,
                safe_message: safe_message.to_owned(),
                retryable,
            },
        }
    }

    pub fn code(&self) -> DeviceWatcherErrorCode {
        self.status.code
    }

    pub fn safe_status(&self) -> &DeviceWatcherSafeError {
        &self.status
    }
}

impl fmt::Display for DeviceWatcherError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.status.safe_message)
    }
}

impl std::error::Error for DeviceWatcherError {}

pub trait SinkGraphValidator {
    fn validate(&self, sink: &PhysicalDevice) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MetadataSinkGraphValidator;

impl SinkGraphValidator for MetadataSinkGraphValidator {
    fn validate(&self, sink: &PhysicalDevice) -> bool {
        sink.available && !sink.name.starts_with("translator_") && !sink.name.ends_with(".monitor")
    }
}

pub trait DeviceWatcher {
    fn read_facts_until(&self, deadline: Instant) -> Result<DeviceFacts, DeviceWatcherError>;
    fn reconcile_until(
        &mut self,
        device_override: DeviceOverride,
        deadline: Instant,
    ) -> Result<DeviceFacts, DeviceWatcherError>;

    fn read_facts(&self) -> Result<DeviceFacts, DeviceWatcherError> {
        self.read_facts_until(Instant::now() + Duration::from_secs(2))
    }
    fn reconcile(
        &mut self,
        device_override: DeviceOverride,
    ) -> Result<DeviceFacts, DeviceWatcherError> {
        self.reconcile_until(device_override, Instant::now() + Duration::from_secs(8))
    }
    fn selected_sink_name(&self) -> Option<&str>;
}

pub struct PulseDeviceWatcher<R = SystemCommandRunner, V = MetadataSinkGraphValidator> {
    runner: R,
    validator: V,
    aec_capability: AecCapability,
    pinned_source_name: Option<String>,
    pinned_sink_name: Option<String>,
    sink_validation_required: bool,
}

impl<R> PulseDeviceWatcher<R, MetadataSinkGraphValidator>
where
    R: CommandRunner,
{
    pub fn new(runner: R, aec_capability: AecCapability) -> Self {
        Self::with_validator(runner, aec_capability, MetadataSinkGraphValidator)
    }
}

impl<R, V> PulseDeviceWatcher<R, V>
where
    R: CommandRunner,
    V: SinkGraphValidator,
{
    pub fn with_validator(runner: R, aec_capability: AecCapability, validator: V) -> Self {
        Self {
            runner,
            validator,
            aec_capability,
            pinned_source_name: None,
            pinned_sink_name: None,
            sink_validation_required: true,
        }
    }

    fn inspect_until(&self, deadline: Instant) -> Result<DeviceSnapshot, DeviceWatcherError> {
        let raw_sources: Vec<RawDevice> =
            read_device_json_until(&self.runner, "sources", deadline)?;
        let raw_sinks: Vec<RawDevice> = read_device_json_until(&self.runner, "sinks", deadline)?;
        validate_device_identities(&raw_sinks)?;
        validate_device_identities(&raw_sources)?;
        let sinks: HashMap<_, _> = raw_sinks
            .into_iter()
            .filter(is_physical_sink)
            .map(PhysicalDevice::from)
            .map(|device| (device.name.clone(), device))
            .collect();
        let sources: HashMap<_, _> = raw_sources
            .into_iter()
            .filter(is_physical_source)
            .map(PhysicalDevice::from)
            .map(|device| (device.name.clone(), device))
            .collect();
        check_device_deadline(deadline)?;
        let default_source = if sources.is_empty() {
            None
        } else {
            Some(self.run_text_until(&["get-default-source"], deadline)?)
        };
        let default_sink = if sinks.is_empty() {
            None
        } else {
            Some(self.run_text_until(&["get-default-sink"], deadline)?)
        };
        check_device_deadline(deadline)?;
        Ok(DeviceSnapshot {
            default_sink,
            default_source,
            sinks,
            sources,
        })
    }

    fn run_text_until(
        &self,
        args: &[&str],
        deadline: Instant,
    ) -> Result<String, DeviceWatcherError> {
        let result = run_device_command_until(&self.runner, args, deadline)?;
        let value = std::str::from_utf8(result.stdout())
            .map(str::trim)
            .ok()
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| DeviceWatcherError::new(DeviceWatcherErrorCode::DiscoveryFailed))?;
        check_device_deadline(deadline)?;
        Ok(value)
    }

    fn validate_source_override(
        snapshot: &DeviceSnapshot,
        name: &str,
    ) -> Result<(), DeviceWatcherError> {
        match snapshot.sources.get(name) {
            Some(device) if device.available => Ok(()),
            _ => Err(DeviceWatcherError::new(
                DeviceWatcherErrorCode::InvalidPhysicalDevice,
            )),
        }
    }

    fn validate_sink_override(
        &self,
        snapshot: &DeviceSnapshot,
        name: &str,
        deadline: Instant,
    ) -> Result<(), DeviceWatcherError> {
        let device = snapshot.sinks.get(name).ok_or_else(|| {
            DeviceWatcherError::new(DeviceWatcherErrorCode::InvalidPhysicalDevice)
        })?;
        self.validate_sink(device, deadline)
    }

    fn validate_sink(
        &self,
        device: &PhysicalDevice,
        deadline: Instant,
    ) -> Result<(), DeviceWatcherError> {
        check_device_deadline(deadline)?;
        if !device.available {
            return Err(DeviceWatcherError::new(
                DeviceWatcherErrorCode::InvalidPhysicalDevice,
            ));
        }
        let accepted = self.validator.validate(device);
        check_device_deadline(deadline)?;
        if !accepted {
            return Err(DeviceWatcherError::new(
                DeviceWatcherErrorCode::GraphValidationFailed,
            ));
        }
        Ok(())
    }

    fn selection_state(
        pinned_name: &Option<String>,
        current_default: &Option<String>,
        devices: &HashMap<String, PhysicalDevice>,
    ) -> DeviceSelectionState {
        let selected = pinned_name
            .as_ref()
            .and_then(|name| devices.get(name))
            .cloned();
        let health = if selected.as_ref().is_some_and(|device| device.available) {
            DeviceHealth::Available
        } else {
            DeviceHealth::DeviceUnavailable
        };
        let current_default = current_default
            .as_ref()
            .filter(|name| devices.contains_key(*name))
            .cloned();
        let pending_default = pinned_name.as_ref().and_then(|pinned| {
            current_default
                .as_ref()
                .filter(|name| *name != pinned)
                .cloned()
        });
        DeviceSelectionState {
            health,
            selected,
            pinned_name: pinned_name.clone(),
            current_default,
            pending_default,
        }
    }

    fn propose_selection(
        &self,
        snapshot: &DeviceSnapshot,
        device_override: DeviceOverride,
        force_validation: bool,
        deadline: Instant,
    ) -> Result<(DeviceFacts, bool), DeviceWatcherError> {
        check_device_deadline(deadline)?;
        let mut proposed_source_name = self.pinned_source_name.clone();
        let mut proposed_sink_name = self.pinned_sink_name.clone();
        let mut sink_validation_required = self.sink_validation_required || force_validation;

        if let Some(source_name) = device_override.source_name.as_deref() {
            Self::validate_source_override(snapshot, source_name)?;
            proposed_source_name = Some(source_name.to_owned());
        } else if proposed_source_name.is_none()
            && snapshot
                .default_source
                .as_ref()
                .and_then(|name| snapshot.sources.get(name))
                .is_some_and(|device| device.available)
        {
            proposed_source_name = snapshot.default_source.clone();
        }

        if let Some(sink_name) = device_override.sink_name.as_deref() {
            self.validate_sink_override(snapshot, sink_name, deadline)?;
            proposed_sink_name = Some(sink_name.to_owned());
            sink_validation_required = false;
        } else if let Some(pinned_sink_name) = proposed_sink_name.as_deref() {
            match snapshot.sinks.get(pinned_sink_name) {
                Some(sink) if sink.available => {
                    if sink_validation_required {
                        self.validate_sink(sink, deadline)?;
                        sink_validation_required = false;
                    }
                }
                _ => sink_validation_required = true,
            }
        } else if let Some(default_sink) = snapshot
            .default_sink
            .as_ref()
            .and_then(|name| snapshot.sinks.get(name))
            && default_sink.available
        {
            self.validate_sink(default_sink, deadline)?;
            proposed_sink_name = snapshot.default_sink.clone();
            sink_validation_required = false;
        }

        let source = Self::selection_state(
            &proposed_source_name,
            &snapshot.default_source,
            &snapshot.sources,
        );
        let sink =
            Self::selection_state(&proposed_sink_name, &snapshot.default_sink, &snapshot.sinks);
        let output_mode = sink
            .selected
            .as_ref()
            .map(classify_output_mode)
            .unwrap_or(OutputMode::UnknownUnsafe);
        check_device_deadline(deadline)?;
        Ok((
            DeviceFacts {
                source,
                sink,
                output_mode,
                aec_capability: self.aec_capability.clone(),
            },
            sink_validation_required,
        ))
    }
}

impl<R, V> DeviceWatcher for PulseDeviceWatcher<R, V>
where
    R: CommandRunner,
    V: SinkGraphValidator,
{
    fn read_facts_until(&self, deadline: Instant) -> Result<DeviceFacts, DeviceWatcherError> {
        let snapshot = self.inspect_until(deadline)?;
        self.propose_selection(&snapshot, DeviceOverride::default(), true, deadline)
            .map(|(facts, _)| facts)
    }

    fn reconcile_until(
        &mut self,
        device_override: DeviceOverride,
        deadline: Instant,
    ) -> Result<DeviceFacts, DeviceWatcherError> {
        let snapshot = self.inspect_until(deadline)?;
        let (facts, sink_validation_required) =
            self.propose_selection(&snapshot, device_override, false, deadline)?;
        check_device_deadline(deadline)?;
        self.pinned_source_name = facts.source.pinned_name.clone();
        self.pinned_sink_name = facts.sink.pinned_name.clone();
        self.sink_validation_required = sink_validation_required;
        Ok(facts)
    }

    fn selected_sink_name(&self) -> Option<&str> {
        self.pinned_sink_name.as_deref()
    }
}

struct DeviceSnapshot {
    default_sink: Option<String>,
    default_source: Option<String>,
    sinks: HashMap<String, PhysicalDevice>,
    sources: HashMap<String, PhysicalDevice>,
}

#[derive(Debug, Deserialize)]
struct RawDevice {
    index: u32,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    state: String,
    #[serde(default)]
    properties: HashMap<String, String>,
    #[serde(default)]
    ports: Vec<RawPort>,
    #[serde(default)]
    active_port: Option<String>,
    monitor_source: String,
    #[serde(default)]
    owner_module: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawPort {
    name: String,
    #[serde(default, rename = "type")]
    port_type: String,
    #[serde(default)]
    availability: String,
}

impl From<RawDevice> for PhysicalDevice {
    fn from(device: RawDevice) -> Self {
        let available = device_available(&device);
        let active_port = device.active_port.filter(|port| !port.is_empty());
        let active_port_type = device
            .ports
            .iter()
            .find(|port| Some(port.name.as_str()) == active_port.as_deref())
            .map(|port| port.port_type.trim())
            .filter(|port_type| !port_type.is_empty())
            .map(str::to_owned);
        Self {
            id: device.index,
            name: device.name,
            description: device.description,
            active_port,
            active_port_type,
            available,
        }
    }
}

fn is_physical_sink(device: &RawDevice) -> bool {
    has_physical_provenance(device, "Audio/Sink")
}

fn is_physical_source(device: &RawDevice) -> bool {
    has_physical_provenance(device, "Audio/Source") && device.monitor_source.is_empty()
}

fn has_physical_provenance(device: &RawDevice, media_class: &str) -> bool {
    !device.name.starts_with("translator_")
        && !device.name.ends_with(".monitor")
        && device
            .properties
            .get("device.class")
            .is_some_and(|class| class.eq_ignore_ascii_case("sound"))
        && device.properties.get("device.api").is_some_and(|api| {
            ["alsa", "bluez", "bluez5"]
                .iter()
                .any(|expected| api.eq_ignore_ascii_case(expected))
        })
        && ["node.virtual", "node.network"].iter().all(|key| {
            device
                .properties
                .get(*key)
                .is_none_or(|value| value.parse::<bool>() == Ok(false))
        })
        && device
            .properties
            .get("media.class")
            .is_none_or(|class| class.eq_ignore_ascii_case(media_class))
}

fn validate_device_identities(devices: &[RawDevice]) -> Result<(), DeviceWatcherError> {
    let mut names = HashSet::new();
    let mut ids = HashSet::new();
    if devices
        .iter()
        .any(|device| !names.insert(&device.name) || !ids.insert(device.index))
    {
        Err(DeviceWatcherError::new(
            DeviceWatcherErrorCode::DiscoveryFailed,
        ))
    } else {
        Ok(())
    }
}

fn device_available(device: &RawDevice) -> bool {
    if device.state.eq_ignore_ascii_case("unavailable") {
        return false;
    }
    let Some(active_port) = device
        .active_port
        .as_deref()
        .filter(|port| !port.is_empty())
    else {
        return true;
    };
    if device.ports.is_empty() {
        return true;
    }
    device
        .ports
        .iter()
        .find(|port| port.name == active_port)
        .is_none_or(|port| !port.availability.eq_ignore_ascii_case("not available"))
}

fn classify_output_mode(device: &PhysicalDevice) -> OutputMode {
    let port_name = device
        .active_port
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let port_type = device
        .active_port_type
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if port_name.contains("headphone") || port_type.contains("headphone") {
        OutputMode::Headphones
    } else if port_name.contains("speaker") || port_type.contains("speaker") {
        OutputMode::OpenSpeaker
    } else {
        OutputMode::UnknownUnsafe
    }
}
