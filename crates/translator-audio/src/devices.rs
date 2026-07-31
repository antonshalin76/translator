use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{CommandResult, CommandRunner, SystemCommandRunner};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticWarning {
    DeviceUnavailable,
    AecNotValidated,
    AecValidationFailed,
    UnknownOutput,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcousticSafety {
    pub mode: OutputMode,
    pub aec_capability: AecCapability,
    pub full_duplex_allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<AcousticWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceState {
    pub source: DeviceSelectionState,
    pub sink: DeviceSelectionState,
    pub acoustic: AcousticSafety,
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
    fn reconcile(
        &mut self,
        device_override: DeviceOverride,
    ) -> Result<DeviceState, DeviceWatcherError>;
    fn selected_sink_name(&self) -> Option<&str>;
}

pub struct PulseDeviceWatcher<R = SystemCommandRunner, V = MetadataSinkGraphValidator> {
    runner: R,
    validator: V,
    aec_capability: AecCapability,
    explicit_headphone_sink: Option<String>,
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
            explicit_headphone_sink: None,
            pinned_source_name: None,
            pinned_sink_name: None,
            sink_validation_required: true,
        }
    }

    pub fn with_explicit_headphone_sink(mut self, sink_name: impl Into<String>) -> Self {
        self.explicit_headphone_sink = Some(sink_name.into());
        self
    }

    fn inspect(&self) -> Result<DeviceSnapshot, DeviceWatcherError> {
        let default_sink = self.run_text(&["get-default-sink"])?;
        let default_source = self.run_text(&["get-default-source"])?;
        let raw_sinks: Vec<RawDevice> = self.run_json(&["--format=json", "list", "sinks"])?;
        let raw_sources: Vec<RawDevice> = self.run_json(&["--format=json", "list", "sources"])?;
        let sinks = raw_sinks
            .into_iter()
            .filter(is_physical_sink)
            .map(PhysicalDevice::from)
            .map(|device| (device.name.clone(), device))
            .collect();
        let sources = raw_sources
            .into_iter()
            .filter(is_physical_source)
            .map(PhysicalDevice::from)
            .map(|device| (device.name.clone(), device))
            .collect();
        Ok(DeviceSnapshot {
            default_sink,
            default_source,
            sinks,
            sources,
        })
    }

    fn run_text(&self, args: &[&str]) -> Result<String, DeviceWatcherError> {
        let result = self.run(args)?;
        std::str::from_utf8(result.stdout())
            .map(str::trim)
            .ok()
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| DeviceWatcherError::new(DeviceWatcherErrorCode::DiscoveryFailed))
    }

    fn run_json<T>(&self, args: &[&str]) -> Result<T, DeviceWatcherError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let result = self.run(args)?;
        serde_json::from_slice(result.stdout())
            .map_err(|_| DeviceWatcherError::new(DeviceWatcherErrorCode::DiscoveryFailed))
    }

    fn run(&self, args: &[&str]) -> Result<CommandResult, DeviceWatcherError> {
        let arguments: Vec<String> = args.iter().map(|value| (*value).to_owned()).collect();
        let result = self
            .runner
            .run("pactl", &arguments)
            .map_err(|_| DeviceWatcherError::new(DeviceWatcherErrorCode::DiscoveryFailed))?;
        if result.is_success() {
            Ok(result)
        } else {
            Err(DeviceWatcherError::new(
                DeviceWatcherErrorCode::DiscoveryFailed,
            ))
        }
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
    ) -> Result<(), DeviceWatcherError> {
        let device = snapshot.sinks.get(name).ok_or_else(|| {
            DeviceWatcherError::new(DeviceWatcherErrorCode::InvalidPhysicalDevice)
        })?;
        if !device.available {
            return Err(DeviceWatcherError::new(
                DeviceWatcherErrorCode::InvalidPhysicalDevice,
            ));
        }
        if !self.validator.validate(device) {
            return Err(DeviceWatcherError::new(
                DeviceWatcherErrorCode::GraphValidationFailed,
            ));
        }
        Ok(())
    }

    fn selection_state(
        pinned_name: &Option<String>,
        current_default: &str,
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
        let default_is_physical = devices.contains_key(current_default);
        let pending_default = pinned_name.as_ref().and_then(|pinned| {
            (default_is_physical && current_default != pinned).then(|| current_default.to_owned())
        });
        DeviceSelectionState {
            health,
            selected,
            pinned_name: pinned_name.clone(),
            current_default: default_is_physical.then(|| current_default.to_owned()),
            pending_default,
        }
    }

    fn acoustic_safety(
        &self,
        source: &DeviceSelectionState,
        sink: &DeviceSelectionState,
    ) -> AcousticSafety {
        let mode = sink
            .selected
            .as_ref()
            .map(|device| {
                if self.explicit_headphone_sink.as_deref() == Some(device.name.as_str()) {
                    OutputMode::Headphones
                } else {
                    classify_output_mode(device)
                }
            })
            .unwrap_or(OutputMode::UnknownUnsafe);
        if source.health != DeviceHealth::Available || sink.health != DeviceHealth::Available {
            return AcousticSafety {
                mode,
                aec_capability: self.aec_capability.clone(),
                full_duplex_allowed: false,
                warning: Some(AcousticWarning::DeviceUnavailable),
            };
        }
        let selected_source = source.selected.as_ref().expect("available source");
        let selected_sink = sink.selected.as_ref().expect("available sink");
        let (full_duplex_allowed, warning) = match mode {
            OutputMode::Headphones => (true, None),
            OutputMode::UnknownUnsafe => (false, Some(AcousticWarning::UnknownOutput)),
            OutputMode::OpenSpeaker => match &self.aec_capability {
                AecCapability::ValidatedFor {
                    source_name,
                    sink_name,
                } if source_name == &selected_source.name && sink_name == &selected_sink.name => {
                    (true, None)
                }
                AecCapability::ValidationFailed => {
                    (false, Some(AcousticWarning::AecValidationFailed))
                }
                _ => (false, Some(AcousticWarning::AecNotValidated)),
            },
        };
        AcousticSafety {
            mode,
            aec_capability: self.aec_capability.clone(),
            full_duplex_allowed,
            warning,
        }
    }
}

impl<R, V> DeviceWatcher for PulseDeviceWatcher<R, V>
where
    R: CommandRunner,
    V: SinkGraphValidator,
{
    fn reconcile(
        &mut self,
        device_override: DeviceOverride,
    ) -> Result<DeviceState, DeviceWatcherError> {
        let snapshot = self.inspect()?;
        let mut proposed_source_name = self.pinned_source_name.clone();
        let mut proposed_sink_name = self.pinned_sink_name.clone();
        let mut sink_validation_required = self.sink_validation_required;

        if let Some(source_name) = device_override.source_name.as_deref() {
            Self::validate_source_override(&snapshot, source_name)?;
            proposed_source_name = Some(source_name.to_owned());
        } else if proposed_source_name.is_none()
            && snapshot
                .sources
                .get(&snapshot.default_source)
                .is_some_and(|device| device.available)
        {
            proposed_source_name = Some(snapshot.default_source.clone());
        }

        if let Some(sink_name) = device_override.sink_name.as_deref() {
            self.validate_sink_override(&snapshot, sink_name)?;
            proposed_sink_name = Some(sink_name.to_owned());
            sink_validation_required = false;
        } else if let Some(pinned_sink_name) = proposed_sink_name.as_deref() {
            match snapshot.sinks.get(pinned_sink_name) {
                Some(sink) if sink.available => {
                    if sink_validation_required {
                        if !self.validator.validate(sink) {
                            return Err(DeviceWatcherError::new(
                                DeviceWatcherErrorCode::GraphValidationFailed,
                            ));
                        }
                        sink_validation_required = false;
                    }
                }
                _ => sink_validation_required = true,
            }
        } else {
            if let Some(default_sink) = snapshot.sinks.get(&snapshot.default_sink) {
                if default_sink.available {
                    if !self.validator.validate(default_sink) {
                        return Err(DeviceWatcherError::new(
                            DeviceWatcherErrorCode::GraphValidationFailed,
                        ));
                    }
                    proposed_sink_name = Some(snapshot.default_sink.clone());
                    sink_validation_required = false;
                }
            }
        }

        self.pinned_source_name = proposed_source_name;
        self.pinned_sink_name = proposed_sink_name;
        self.sink_validation_required = sink_validation_required;

        let source = Self::selection_state(
            &self.pinned_source_name,
            &snapshot.default_source,
            &snapshot.sources,
        );
        let sink = Self::selection_state(
            &self.pinned_sink_name,
            &snapshot.default_sink,
            &snapshot.sinks,
        );
        let acoustic = self.acoustic_safety(&source, &sink);
        Ok(DeviceState {
            source,
            sink,
            acoustic,
        })
    }

    fn selected_sink_name(&self) -> Option<&str> {
        self.pinned_sink_name.as_deref()
    }
}

struct DeviceSnapshot {
    default_sink: String,
    default_source: String,
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
    !device.name.starts_with("translator_")
        && !device.name.ends_with(".monitor")
        && device
            .properties
            .get("device.class")
            .is_none_or(|class| !class.eq_ignore_ascii_case("monitor"))
}

fn is_physical_source(device: &RawDevice) -> bool {
    is_physical_sink(device)
        && !device.name.ends_with(".monitor")
        && device
            .properties
            .get("media.class")
            .is_none_or(|class| class.eq_ignore_ascii_case("Audio/Source"))
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
