use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use translator_audio::{
    AcousticWarning, AecCapability, CommandResult, CommandRunError, CommandRunner, DeviceHealth,
    DeviceOverride, DeviceWatcher, DeviceWatcherError, DeviceWatcherErrorCode, OutputMode,
    PhysicalDevice, PulseDeviceWatcher, SinkGraphValidator,
};

#[derive(Clone)]
struct FakeRunner {
    expected: Arc<Mutex<VecDeque<ExpectedCommand>>>,
}

struct ExpectedCommand {
    args: Vec<String>,
    result: Result<CommandResult, CommandRunError>,
}

impl FakeRunner {
    fn new(expected: Vec<ExpectedCommand>) -> Self {
        Self {
            expected: Arc::new(Mutex::new(expected.into())),
        }
    }

    fn assert_drained(&self) {
        assert!(self.expected.lock().unwrap().is_empty());
    }
}

impl CommandRunner for FakeRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        let expected = self.expected.lock().unwrap().pop_front().unwrap();
        assert_eq!(args, expected.args);
        expected.result
    }
}

#[derive(Clone)]
struct FakeValidator {
    expected: Arc<Mutex<VecDeque<(String, bool)>>>,
}

impl FakeValidator {
    fn new(expected: Vec<(&str, bool)>) -> Self {
        Self {
            expected: Arc::new(Mutex::new(
                expected
                    .into_iter()
                    .map(|(name, result)| (name.to_owned(), result))
                    .collect(),
            )),
        }
    }

    fn assert_drained(&self) {
        assert!(self.expected.lock().unwrap().is_empty());
    }
}

impl SinkGraphValidator for FakeValidator {
    fn validate(&self, sink: &PhysicalDevice) -> bool {
        let (expected_name, result) = self.expected.lock().unwrap().pop_front().unwrap();
        assert_eq!(sink.name, expected_name);
        result
    }
}

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn snapshot(
    default_sink: &str,
    default_source: &str,
    sinks: serde_json::Value,
    sources: serde_json::Value,
) -> Vec<ExpectedCommand> {
    vec![
        ExpectedCommand {
            args: args(&["get-default-sink"]),
            result: Ok(CommandResult::success(default_sink.as_bytes().to_vec())),
        },
        ExpectedCommand {
            args: args(&["get-default-source"]),
            result: Ok(CommandResult::success(default_source.as_bytes().to_vec())),
        },
        ExpectedCommand {
            args: args(&["--format=json", "list", "sinks"]),
            result: Ok(CommandResult::success(sinks.to_string().into_bytes())),
        },
        ExpectedCommand {
            args: args(&["--format=json", "list", "sources"]),
            result: Ok(CommandResult::success(sources.to_string().into_bytes())),
        },
    ]
}

fn assert_error_redacted(error: &DeviceWatcherError) {
    for representation in [
        format!("{error:?}"),
        error.to_string(),
        serde_json::to_string(error.safe_status()).unwrap(),
    ] {
        assert!(!representation.contains("private-device-marker"));
    }
}

fn sink(id: u32, name: &str, port_name: &str, port_type: &str) -> serde_json::Value {
    serde_json::json!({
        "index": id,
        "name": name,
        "description": name,
        "state": "SUSPENDED",
        "properties": {
            "device.class": "sound",
            "media.class": "Audio/Sink"
        },
        "ports": [{
            "name": port_name,
            "type": port_type,
            "availability": "available"
        }],
        "active_port": port_name
    })
}

fn unavailable_sink(id: u32, name: &str, port_name: &str, port_type: &str) -> serde_json::Value {
    let mut value = sink(id, name, port_name, port_type);
    value["ports"][0]["availability"] = serde_json::json!("not available");
    value
}

fn source(id: u32, name: &str, device_class: &str) -> serde_json::Value {
    serde_json::json!({
        "index": id,
        "name": name,
        "description": name,
        "state": "SUSPENDED",
        "properties": {
            "device.class": device_class,
            "media.class": "Audio/Source"
        },
        "ports": [],
        "active_port": ""
    })
}

#[test]
fn physical_headset_defaults_are_pinned_and_allow_duplex() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let runner = FakeRunner::new(snapshot(
        sink_name,
        source_name,
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.sink.health, DeviceHealth::Available);
    assert_eq!(state.source.health, DeviceHealth::Available);
    assert_eq!(state.acoustic.mode, OutputMode::Headphones);
    assert!(state.acoustic.full_duplex_allowed);
    assert_eq!(state.acoustic.warning, None);
    runner.assert_drained();
}

#[test]
fn nullable_virtual_active_ports_do_not_invalidate_physical_device_discovery() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let virtual_sink = serde_json::json!({
        "index": 70,
        "name": "translator_mic_out",
        "description": "Translator_Mic_Out",
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "media.class": "Audio/Sink"},
        "ports": [],
        "active_port": null
    });
    let virtual_source = serde_json::json!({
        "index": 71,
        "name": "translator_virtual_mic",
        "description": "Translator_Virtual_Mic",
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "media.class": "Audio/Source"},
        "ports": [],
        "active_port": null
    });
    let runner = FakeRunner::new(snapshot(
        sink_name,
        source_name,
        serde_json::json!([
            sink(50, sink_name, "analog-output-headphones", "Headphones"),
            virtual_sink
        ]),
        serde_json::json!([source(60, source_name, "sound"), virtual_source]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.sink.selected.unwrap().name, sink_name);
    assert_eq!(state.source.selected.unwrap().name, source_name);
    runner.assert_drained();
}

#[test]
fn output_mode_uses_the_active_port_type_when_its_name_is_generic() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let runner = FakeRunner::new(snapshot(
        sink_name,
        source_name,
        serde_json::json!([sink(50, sink_name, "analog-output", "Headphones")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.acoustic.mode, OutputMode::Headphones);
    assert!(state.acoustic.full_duplex_allowed);
    runner.assert_drained();
}

#[test]
fn explicit_headphone_sink_allows_a_generic_usb_duplex_device() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let runner = FakeRunner::new(snapshot(
        sink_name,
        source_name,
        serde_json::json!([sink(50, sink_name, "analog-output", "Analog")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable)
        .with_explicit_headphone_sink(sink_name);

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.acoustic.mode, OutputMode::Headphones);
    assert!(state.acoustic.full_duplex_allowed);
    assert_eq!(state.acoustic.warning, None);
    runner.assert_drained();
}

#[test]
fn translator_virtual_default_source_is_never_selected_as_physical() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let runner = FakeRunner::new(snapshot(
        sink_name,
        "translator_virtual_mic",
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(70, "translator_virtual_mic", "sound")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.source.health, DeviceHealth::DeviceUnavailable);
    assert!(state.source.selected.is_none());
    runner.assert_drained();
}

#[test]
fn translator_virtual_sink_is_rejected_as_default_and_manual_selection() {
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let virtual_sink = "translator_remote_in";
    let runner = FakeRunner::new(snapshot(
        virtual_sink,
        source_name,
        serde_json::json!([sink(70, virtual_sink, "analog-output", "Analog")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let error = watcher
        .reconcile(DeviceOverride {
            source_name: None,
            sink_name: Some(virtual_sink.to_owned()),
        })
        .unwrap_err();

    assert_eq!(error.code(), DeviceWatcherErrorCode::InvalidPhysicalDevice);
    assert!(watcher.selected_sink_name().is_none());
    runner.assert_drained();
}

#[test]
fn translator_virtual_default_sink_is_device_unavailable_without_override() {
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let virtual_sink = "translator_remote_in";
    let runner = FakeRunner::new(snapshot(
        virtual_sink,
        source_name,
        serde_json::json!([sink(70, virtual_sink, "analog-output", "Analog")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.sink.health, DeviceHealth::DeviceUnavailable);
    assert!(state.sink.selected.is_none());
    runner.assert_drained();
}

#[test]
fn monitor_source_is_rejected_even_by_manual_override() {
    let monitor = "alsa_output.usb-headset.analog-stereo.monitor";
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let runner = FakeRunner::new(snapshot(
        sink_name,
        monitor,
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(51, monitor, "monitor")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let error = watcher
        .reconcile(DeviceOverride {
            source_name: Some(monitor.to_owned()),
            sink_name: None,
        })
        .unwrap_err();

    assert_eq!(error.code(), DeviceWatcherErrorCode::InvalidPhysicalDevice);
    runner.assert_drained();
}

#[test]
fn pinned_source_enters_unavailable_and_recovers_only_by_same_name() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let usb_source = "alsa_input.usb-headset.mono-fallback";
    let built_in = "alsa_input.pci-built-in.analog-stereo";
    let mut expected = snapshot(
        sink_name,
        usb_source,
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(60, usb_source, "sound")]),
    );
    expected.extend(snapshot(
        sink_name,
        built_in,
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(61, built_in, "sound")]),
    ));
    expected.extend(snapshot(
        sink_name,
        built_in,
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([
            source(60, usb_source, "sound"),
            source(61, built_in, "sound")
        ]),
    ));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    watcher.reconcile(DeviceOverride::default()).unwrap();
    let lost = watcher.reconcile(DeviceOverride::default()).unwrap();
    let recovered = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(lost.source.health, DeviceHealth::DeviceUnavailable);
    assert_eq!(lost.source.pending_default.as_deref(), Some(built_in));
    assert_eq!(recovered.source.health, DeviceHealth::Available);
    assert_eq!(recovered.source.selected.unwrap().name, usb_source);
    assert_eq!(recovered.source.pending_default.as_deref(), Some(built_in));
    runner.assert_drained();
}

#[test]
fn changed_default_does_not_replace_an_available_pin() {
    let first_sink = "alsa_output.usb-headset.analog-stereo";
    let second_sink = "alsa_output.pci-built-in.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let mut expected = snapshot(
        first_sink,
        source_name,
        serde_json::json!([sink(
            50,
            first_sink,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        second_sink,
        source_name,
        serde_json::json!([
            sink(50, first_sink, "analog-output-headphones", "Headphones"),
            sink(51, second_sink, "analog-output-speaker", "Speaker")
        ]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    watcher.reconcile(DeviceOverride::default()).unwrap();
    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.sink.selected.unwrap().name, first_sink);
    assert_eq!(state.sink.pending_default.as_deref(), Some(second_sink));
    assert_eq!(state.acoustic.mode, OutputMode::Headphones);
    runner.assert_drained();
}

#[test]
fn pinned_sink_enters_unavailable_and_recovers_only_by_same_name() {
    let usb_sink = "alsa_output.usb-headset.analog-stereo";
    let built_in = "alsa_output.pci-built-in.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let mut expected = snapshot(
        usb_sink,
        source_name,
        serde_json::json!([sink(50, usb_sink, "analog-output-headphones", "Headphones")]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        built_in,
        source_name,
        serde_json::json!([sink(51, built_in, "analog-output-speaker", "Speaker")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    expected.extend(snapshot(
        built_in,
        source_name,
        serde_json::json!([
            sink(50, usb_sink, "analog-output-headphones", "Headphones"),
            sink(51, built_in, "analog-output-speaker", "Speaker")
        ]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    watcher.reconcile(DeviceOverride::default()).unwrap();
    let lost = watcher.reconcile(DeviceOverride::default()).unwrap();
    let recovered = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(lost.sink.health, DeviceHealth::DeviceUnavailable);
    assert_eq!(lost.sink.pending_default.as_deref(), Some(built_in));
    assert_eq!(recovered.sink.health, DeviceHealth::Available);
    assert_eq!(recovered.sink.selected.unwrap().name, usb_sink);
    assert_eq!(recovered.sink.pending_default.as_deref(), Some(built_in));
    runner.assert_drained();
}

#[test]
fn same_name_sink_replug_requires_graph_revalidation_before_recovery() {
    let usb_sink = "alsa_output.usb-headset.analog-stereo";
    let built_in = "alsa_output.pci-built-in.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let mut expected = snapshot(
        usb_sink,
        source_name,
        serde_json::json!([sink(50, usb_sink, "analog-output-headphones", "Headphones")]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        built_in,
        source_name,
        serde_json::json!([sink(51, built_in, "analog-output-speaker", "Speaker")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    for _ in 0..2 {
        expected.extend(snapshot(
            built_in,
            source_name,
            serde_json::json!([
                sink(75, usb_sink, "analog-output-headphones", "Headphones"),
                sink(51, built_in, "analog-output-speaker", "Speaker")
            ]),
            serde_json::json!([source(60, source_name, "sound")]),
        ));
    }
    let runner = FakeRunner::new(expected);
    let validator = FakeValidator::new(vec![(usb_sink, true), (usb_sink, false), (usb_sink, true)]);
    let mut watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );

    watcher.reconcile(DeviceOverride::default()).unwrap();
    let lost = watcher.reconcile(DeviceOverride::default()).unwrap();
    let error = watcher.reconcile(DeviceOverride::default()).unwrap_err();
    assert_eq!(watcher.selected_sink_name(), Some(usb_sink));
    let recovered = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(lost.sink.health, DeviceHealth::DeviceUnavailable);
    assert_eq!(error.code(), DeviceWatcherErrorCode::GraphValidationFailed);
    assert_eq!(recovered.sink.health, DeviceHealth::Available);
    assert_eq!(recovered.sink.selected.unwrap().id, 75);
    validator.assert_drained();
    runner.assert_drained();
}

#[test]
fn listed_sink_recovery_requires_graph_revalidation_before_duplex() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let mut expected = snapshot(
        sink_name,
        source_name,
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        sink_name,
        source_name,
        serde_json::json!([unavailable_sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    for _ in 0..2 {
        expected.extend(snapshot(
            sink_name,
            source_name,
            serde_json::json!([sink(
                50,
                sink_name,
                "analog-output-headphones",
                "Headphones"
            )]),
            serde_json::json!([source(60, source_name, "sound")]),
        ));
    }
    let runner = FakeRunner::new(expected);
    let validator = FakeValidator::new(vec![
        (sink_name, true),
        (sink_name, false),
        (sink_name, true),
    ]);
    let mut watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );

    watcher.reconcile(DeviceOverride::default()).unwrap();
    let unavailable = watcher.reconcile(DeviceOverride::default()).unwrap();
    let error = watcher.reconcile(DeviceOverride::default()).unwrap_err();
    assert_eq!(watcher.selected_sink_name(), Some(sink_name));
    let recovered = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(unavailable.sink.health, DeviceHealth::DeviceUnavailable);
    assert_eq!(error.code(), DeviceWatcherErrorCode::GraphValidationFailed);
    assert_eq!(recovered.sink.health, DeviceHealth::Available);
    assert!(recovered.acoustic.full_duplex_allowed);
    validator.assert_drained();
    runner.assert_drained();
}

#[test]
fn failed_initial_sink_validation_does_not_commit_a_source_pin() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let first_source = "alsa_input.first-mic";
    let second_source = "alsa_input.second-mic";
    let sinks = serde_json::json!([sink(
        50,
        sink_name,
        "analog-output-headphones",
        "Headphones"
    )]);
    let mut expected = snapshot(
        sink_name,
        first_source,
        sinks.clone(),
        serde_json::json!([source(60, first_source, "sound")]),
    );
    expected.extend(snapshot(
        sink_name,
        second_source,
        sinks,
        serde_json::json!([
            source(60, first_source, "sound"),
            source(61, second_source, "sound")
        ]),
    ));
    let runner = FakeRunner::new(expected);
    let validator = FakeValidator::new(vec![(sink_name, false), (sink_name, true)]);
    let mut watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );

    let error = watcher.reconcile(DeviceOverride::default()).unwrap_err();
    let recovered = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(error.code(), DeviceWatcherErrorCode::GraphValidationFailed);
    assert_eq!(recovered.source.selected.unwrap().name, second_source);
    validator.assert_drained();
    runner.assert_drained();
}

#[test]
fn listed_but_unavailable_ports_enter_device_unavailable_and_recover() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let normal_sink = sink(50, sink_name, "analog-output-headphones", "Headphones");
    let normal_source = serde_json::json!({
        "index": 60,
        "name": source_name,
        "description": source_name,
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "media.class": "Audio/Source"},
        "ports": [{"name": "analog-input-mic", "type": "Mic", "availability": "available"}],
        "active_port": "analog-input-mic"
    });
    let unavailable_sink = serde_json::json!({
        "index": 50,
        "name": sink_name,
        "description": sink_name,
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "media.class": "Audio/Sink"},
        "ports": [{
            "name": "analog-output-headphones",
            "type": "Headphones",
            "availability": "not available"
        }],
        "active_port": "analog-output-headphones"
    });
    let unavailable_source = serde_json::json!({
        "index": 60,
        "name": source_name,
        "description": source_name,
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "media.class": "Audio/Source"},
        "ports": [{
            "name": "analog-input-mic",
            "type": "Mic",
            "availability": "not available"
        }],
        "active_port": "analog-input-mic"
    });
    let mut expected = snapshot(
        sink_name,
        source_name,
        serde_json::json!([normal_sink.clone()]),
        serde_json::json!([normal_source.clone()]),
    );
    expected.extend(snapshot(
        sink_name,
        source_name,
        serde_json::json!([unavailable_sink]),
        serde_json::json!([unavailable_source]),
    ));
    expected.extend(snapshot(
        sink_name,
        source_name,
        serde_json::json!([normal_sink]),
        serde_json::json!([normal_source]),
    ));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    watcher.reconcile(DeviceOverride::default()).unwrap();
    let unavailable = watcher.reconcile(DeviceOverride::default()).unwrap();
    let recovered = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(unavailable.sink.health, DeviceHealth::DeviceUnavailable);
    assert_eq!(unavailable.source.health, DeviceHealth::DeviceUnavailable);
    assert_eq!(recovered.sink.health, DeviceHealth::Available);
    assert_eq!(recovered.source.health, DeviceHealth::Available);
    runner.assert_drained();
}

#[test]
fn lost_devices_can_be_replaced_only_by_explicit_valid_physical_overrides() {
    let old_sink = "alsa_output.usb-old.analog-stereo";
    let new_sink = "alsa_output.usb-new.analog-stereo";
    let old_source = "alsa_input.usb-old.mono-fallback";
    let new_source = "alsa_input.usb-new.mono-fallback";
    let mut expected = snapshot(
        old_sink,
        old_source,
        serde_json::json!([sink(50, old_sink, "analog-output-headphones", "Headphones")]),
        serde_json::json!([source(60, old_source, "sound")]),
    );
    expected.extend(snapshot(
        new_sink,
        new_source,
        serde_json::json!([sink(51, new_sink, "analog-output-headphones", "Headphones")]),
        serde_json::json!([source(61, new_source, "sound")]),
    ));
    let runner = FakeRunner::new(expected);
    let validator = FakeValidator::new(vec![(old_sink, true), (new_sink, true)]);
    let mut watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );
    watcher.reconcile(DeviceOverride::default()).unwrap();

    let state = watcher
        .reconcile(DeviceOverride {
            source_name: Some(new_source.to_owned()),
            sink_name: Some(new_sink.to_owned()),
        })
        .unwrap();

    assert_eq!(state.sink.selected.unwrap().name, new_sink);
    assert_eq!(state.source.selected.unwrap().name, new_source);
    validator.assert_drained();
    runner.assert_drained();
}

#[test]
fn changed_default_source_does_not_replace_an_available_pin() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let usb_source = "alsa_input.usb-headset.mono-fallback";
    let built_in = "alsa_input.pci-built-in.analog-stereo";
    let mut expected = snapshot(
        sink_name,
        usb_source,
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([source(60, usb_source, "sound")]),
    );
    expected.extend(snapshot(
        sink_name,
        built_in,
        serde_json::json!([sink(
            50,
            sink_name,
            "analog-output-headphones",
            "Headphones"
        )]),
        serde_json::json!([
            source(60, usb_source, "sound"),
            source(61, built_in, "sound")
        ]),
    ));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    watcher.reconcile(DeviceOverride::default()).unwrap();
    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.source.selected.unwrap().name, usb_source);
    assert_eq!(state.source.pending_default.as_deref(), Some(built_in));
    runner.assert_drained();
}

#[test]
fn speaker_and_unknown_analog_outputs_are_blocked_without_validated_aec() {
    let source_name = "alsa_input.usb.mono-fallback";
    let speaker_name = "alsa_output.pci-built-in.analog-stereo";
    let unknown_name = "alsa_output.usb-device.analog-stereo";
    let mut expected = snapshot(
        speaker_name,
        source_name,
        serde_json::json!([sink(50, speaker_name, "analog-output-speaker", "Speaker")]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        unknown_name,
        source_name,
        serde_json::json!([sink(51, unknown_name, "analog-output", "Analog")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let runner = FakeRunner::new(expected);
    let mut speaker = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let speaker_state = speaker.reconcile(DeviceOverride::default()).unwrap();
    let mut unknown = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);
    let unknown_state = unknown.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(speaker_state.acoustic.mode, OutputMode::OpenSpeaker);
    assert!(!speaker_state.acoustic.full_duplex_allowed);
    assert_eq!(
        speaker_state.acoustic.warning,
        Some(AcousticWarning::AecNotValidated)
    );
    assert_eq!(unknown_state.acoustic.mode, OutputMode::UnknownUnsafe);
    assert!(!unknown_state.acoustic.full_duplex_allowed);
    assert_eq!(
        unknown_state.acoustic.warning,
        Some(AcousticWarning::UnknownOutput)
    );
    runner.assert_drained();
}

#[test]
fn validated_aec_allows_open_speaker_mode_but_not_unknown_outputs() {
    let source_name = "alsa_input.usb.mono-fallback";
    let speaker_name = "alsa_output.pci-built-in.analog-stereo";
    let unknown_name = "alsa_output.usb-device.analog-stereo";
    let mut expected = snapshot(
        speaker_name,
        source_name,
        serde_json::json!([sink(50, speaker_name, "analog-output-speaker", "Speaker")]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        unknown_name,
        source_name,
        serde_json::json!([sink(51, unknown_name, "analog-output", "Analog")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseDeviceWatcher::new(
        runner.clone(),
        AecCapability::ValidatedFor {
            source_name: source_name.to_owned(),
            sink_name: speaker_name.to_owned(),
        },
    );

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();
    let mut unknown = PulseDeviceWatcher::new(
        runner.clone(),
        AecCapability::ValidatedFor {
            source_name: source_name.to_owned(),
            sink_name: unknown_name.to_owned(),
        },
    );
    let unknown_state = unknown.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.acoustic.mode, OutputMode::OpenSpeaker);
    assert!(state.acoustic.full_duplex_allowed);
    assert_eq!(unknown_state.acoustic.mode, OutputMode::UnknownUnsafe);
    assert!(!unknown_state.acoustic.full_duplex_allowed);
    runner.assert_drained();
}

#[test]
fn unvalidated_and_failed_aec_states_keep_speakers_blocked() {
    let source_name = "alsa_input.usb.mono-fallback";
    let speaker_name = "alsa_output.pci-built-in.analog-stereo";
    let mut expected = snapshot(
        speaker_name,
        source_name,
        serde_json::json!([sink(50, speaker_name, "analog-output-speaker", "Speaker")]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        speaker_name,
        source_name,
        serde_json::json!([sink(50, speaker_name, "analog-output-speaker", "Speaker")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let runner = FakeRunner::new(expected);
    let mut unvalidated =
        PulseDeviceWatcher::new(runner.clone(), AecCapability::AvailableUnvalidated);
    let mut failed = PulseDeviceWatcher::new(runner.clone(), AecCapability::ValidationFailed);

    let unvalidated_state = unvalidated.reconcile(DeviceOverride::default()).unwrap();
    let failed_state = failed.reconcile(DeviceOverride::default()).unwrap();

    assert!(!unvalidated_state.acoustic.full_duplex_allowed);
    assert_eq!(
        unvalidated_state.acoustic.warning,
        Some(AcousticWarning::AecNotValidated)
    );
    assert!(!failed_state.acoustic.full_duplex_allowed);
    assert_eq!(
        failed_state.acoustic.warning,
        Some(AcousticWarning::AecValidationFailed)
    );
    runner.assert_drained();
}

#[test]
fn hdmi_output_is_unknown_unsafe_even_with_validated_aec() {
    let source_name = "alsa_input.usb.mono-fallback";
    let hdmi_name = "alsa_output.pci-hdmi.hdmi-stereo";
    let runner = FakeRunner::new(snapshot(
        hdmi_name,
        source_name,
        serde_json::json!([sink(50, hdmi_name, "hdmi-output-0", "HDMI")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(
        runner.clone(),
        AecCapability::ValidatedFor {
            source_name: source_name.to_owned(),
            sink_name: hdmi_name.to_owned(),
        },
    );

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.acoustic.mode, OutputMode::UnknownUnsafe);
    assert!(!state.acoustic.full_duplex_allowed);
    assert_eq!(state.acoustic.warning, Some(AcousticWarning::UnknownOutput));
    runner.assert_drained();
}

#[test]
fn malformed_device_json_returns_a_privacy_safe_error() {
    let runner = FakeRunner::new(vec![
        ExpectedCommand {
            args: args(&["get-default-sink"]),
            result: Ok(CommandResult::success(b"alsa_output.safe".to_vec())),
        },
        ExpectedCommand {
            args: args(&["get-default-source"]),
            result: Ok(CommandResult::success(b"alsa_input.safe".to_vec())),
        },
        ExpectedCommand {
            args: args(&["--format=json", "list", "sinks"]),
            result: Ok(CommandResult::success(b"{private-device-marker".to_vec())),
        },
    ]);
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let error = watcher.reconcile(DeviceOverride::default()).unwrap_err();

    assert_eq!(error.code(), DeviceWatcherErrorCode::DiscoveryFailed);
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn validated_aec_does_not_carry_over_to_a_different_device_pair() {
    let old_sink = "alsa_output.old-speakers";
    let new_sink = "alsa_output.new-speakers";
    let old_source = "alsa_input.old-mic";
    let new_source = "alsa_input.new-mic";
    let mut expected = snapshot(
        old_sink,
        old_source,
        serde_json::json!([
            sink(50, old_sink, "analog-output-speaker", "Speaker"),
            sink(51, new_sink, "analog-output-speaker", "Speaker")
        ]),
        serde_json::json!([
            source(60, old_source, "sound"),
            source(61, new_source, "sound")
        ]),
    );
    expected.extend(snapshot(
        new_sink,
        new_source,
        serde_json::json!([
            sink(50, old_sink, "analog-output-speaker", "Speaker"),
            sink(51, new_sink, "analog-output-speaker", "Speaker")
        ]),
        serde_json::json!([
            source(60, old_source, "sound"),
            source(61, new_source, "sound")
        ]),
    ));
    let runner = FakeRunner::new(expected);
    let validator = FakeValidator::new(vec![(old_sink, true), (new_sink, true)]);
    let mut watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::ValidatedFor {
            source_name: old_source.to_owned(),
            sink_name: old_sink.to_owned(),
        },
        validator.clone(),
    );
    watcher.reconcile(DeviceOverride::default()).unwrap();

    let state = watcher
        .reconcile(DeviceOverride {
            source_name: Some(new_source.to_owned()),
            sink_name: Some(new_sink.to_owned()),
        })
        .unwrap();

    assert_eq!(state.acoustic.mode, OutputMode::OpenSpeaker);
    assert!(!state.acoustic.full_duplex_allowed);
    assert_eq!(
        state.acoustic.warning,
        Some(AcousticWarning::AecNotValidated)
    );
    validator.assert_drained();
    runner.assert_drained();
}

#[test]
fn nonzero_device_command_returns_a_privacy_safe_error() {
    let runner = FakeRunner::new(vec![ExpectedCommand {
        args: args(&["get-default-sink"]),
        result: Ok(CommandResult::failure(
            Vec::new(),
            b"private-device-marker".to_vec(),
        )),
    }]);
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let error = watcher.reconcile(DeviceOverride::default()).unwrap_err();

    assert_eq!(error.code(), DeviceWatcherErrorCode::DiscoveryFailed);
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn sink_override_requires_short_graph_validation_and_preserves_old_pin_on_failure() {
    let first_sink = "alsa_output.usb-headset.analog-stereo";
    let second_sink = "alsa_output.pci-built-in.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let mut expected = snapshot(
        first_sink,
        source_name,
        serde_json::json!([
            sink(50, first_sink, "analog-output-headphones", "Headphones"),
            sink(51, second_sink, "analog-output-speaker", "Speaker")
        ]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        second_sink,
        source_name,
        serde_json::json!([
            sink(50, first_sink, "analog-output-headphones", "Headphones"),
            sink(51, second_sink, "analog-output-speaker", "Speaker")
        ]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let runner = FakeRunner::new(expected);
    let validator = FakeValidator::new(vec![(first_sink, true), (second_sink, false)]);
    let mut watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );
    watcher.reconcile(DeviceOverride::default()).unwrap();

    let error = watcher
        .reconcile(DeviceOverride {
            source_name: None,
            sink_name: Some(second_sink.to_owned()),
        })
        .unwrap_err();

    assert_eq!(error.code(), DeviceWatcherErrorCode::GraphValidationFailed);
    assert_eq!(watcher.selected_sink_name(), Some(first_sink));
    validator.assert_drained();
    runner.assert_drained();
}

#[test]
fn validated_sink_override_changes_the_pin() {
    let first_sink = "alsa_output.usb-headset.analog-stereo";
    let second_sink = "alsa_output.pci-built-in.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let mut expected = snapshot(
        first_sink,
        source_name,
        serde_json::json!([
            sink(50, first_sink, "analog-output-headphones", "Headphones"),
            sink(51, second_sink, "analog-output-speaker", "Speaker")
        ]),
        serde_json::json!([source(60, source_name, "sound")]),
    );
    expected.extend(snapshot(
        second_sink,
        source_name,
        serde_json::json!([
            sink(50, first_sink, "analog-output-headphones", "Headphones"),
            sink(51, second_sink, "analog-output-speaker", "Speaker")
        ]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let runner = FakeRunner::new(expected);
    let validator = FakeValidator::new(vec![(first_sink, true), (second_sink, true)]);
    let mut watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );
    watcher.reconcile(DeviceOverride::default()).unwrap();

    let state = watcher
        .reconcile(DeviceOverride {
            source_name: None,
            sink_name: Some(second_sink.to_owned()),
        })
        .unwrap();

    assert_eq!(state.sink.selected.unwrap().name, second_sink);
    validator.assert_drained();
    runner.assert_drained();
}
