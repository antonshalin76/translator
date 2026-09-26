use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use translator_audio::{
    AecCapability, CommandResult, CommandRunError, CommandRunner, DeviceHealth, DeviceOverride,
    DeviceWatcher, DeviceWatcherError, DeviceWatcherErrorCode, OutputMode, PhysicalDevice,
    PulseDeviceWatcher, SinkGraphValidator,
};

#[derive(Clone)]
struct ProvenanceRunner {
    sinks: serde_json::Value,
    sources: serde_json::Value,
}

type RecordedDeviceCalls = Arc<Mutex<Vec<(Vec<String>, Instant)>>>;

#[derive(Debug, Clone)]
enum ReadFault {
    Command(CommandRunError),
    Malformed,
    Nonzero,
    Late,
}

struct FaultingReadRunner<R> {
    inner: R,
    phase: usize,
    fault: ReadFault,
    calls: RecordedDeviceCalls,
}

impl<R: CommandRunner> CommandRunner for FaultingReadRunner<R> {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        let current = {
            let mut calls = self.calls.lock().unwrap();
            calls.push((args.to_vec(), deadline));
            calls.len()
        };
        if current == self.phase {
            match self.fault {
                ReadFault::Command(error) => return Err(error),
                ReadFault::Malformed => {
                    return Ok(CommandResult::success(
                        b"\xffprivate-device-marker".to_vec(),
                    ));
                }
                ReadFault::Nonzero => {
                    return Ok(CommandResult::failure(
                        Vec::new(),
                        b"private-device-marker".to_vec(),
                    ));
                }
                ReadFault::Late => std::thread::sleep(
                    deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(5),
                ),
            }
        }
        self.inner.run_until(program, args, deadline)
    }
}

fn read_faults() -> [ReadFault; 6] {
    [
        ReadFault::Command(CommandRunError::DeadlineExpired),
        ReadFault::Command(CommandRunError::TimedOut),
        ReadFault::Command(CommandRunError::SpawnFailed),
        ReadFault::Malformed,
        ReadFault::Nonzero,
        ReadFault::Late,
    ]
}

fn assert_failed_read_prefix(
    error: &DeviceWatcherError,
    fault: &ReadFault,
    calls: &RecordedDeviceCalls,
    expected: &[Vec<String>],
    deadline: Instant,
) {
    let code = match fault {
        ReadFault::Command(CommandRunError::DeadlineExpired) | ReadFault::Late => {
            DeviceWatcherErrorCode::DeadlineExpired
        }
        _ => DeviceWatcherErrorCode::DiscoveryFailed,
    };
    assert_eq!(error.code(), code, "fault: {fault:?}");
    assert_error_redacted(error);
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.iter().map(|(args, _)| args).collect::<Vec<_>>(),
        expected.iter().collect::<Vec<_>>()
    );
    assert!(calls.iter().all(|(_, observed)| *observed == deadline));
}

#[test]
fn every_device_read_phase_preserves_error_deadline_and_no_pin_commit() {
    let expected = [
        args(&["--format=json", "list", "sources"]),
        args(&["--format=json", "list", "sinks"]),
        args(&["get-default-source"]),
        args(&["get-default-sink"]),
    ];
    for phase in 1..=expected.len() {
        for fault in read_faults() {
            let calls = Arc::default();
            let runner = FaultingReadRunner {
                inner: FreshFactsRunner::new(),
                phase,
                fault: fault.clone(),
                calls: Arc::clone(&calls),
            };
            let validator = ObservingValidator::new();
            let watcher = PulseDeviceWatcher::with_validator(
                runner,
                AecCapability::Unavailable,
                validator.clone(),
            );
            let deadline = Instant::now() + Duration::from_millis(100);
            let error = watcher.read_facts_until(deadline).unwrap_err();
            assert_failed_read_prefix(&error, &fault, &calls, &expected[..phase], deadline);
            assert_eq!(watcher.selected_sink_name(), None);
            assert!(validator.calls.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn expired_reconciliation_does_not_discover_validate_or_commit_overrides() {
    let runner = FreshFactsRunner::new();
    let validator = ObservingValidator::new();
    let mut watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );
    let error = watcher
        .reconcile_until(
            DeviceOverride {
                source_name: Some("external-input".to_owned()),
                sink_name: Some("external-output".to_owned()),
            },
            Instant::now(),
        )
        .unwrap_err();
    assert_eq!(error.code(), DeviceWatcherErrorCode::DeadlineExpired);
    assert!(runner.calls.lock().unwrap().is_empty());
    assert!(validator.calls.lock().unwrap().is_empty());
    assert_eq!(watcher.selected_sink_name(), None);
}

#[test]
fn empty_physical_sets_issue_only_list_reads() {
    let runner = FreshFactsRunner::new();
    runner.snapshot.lock().unwrap().sources = serde_json::json!([]);
    runner.snapshot.lock().unwrap().sinks = serde_json::json!([]);
    let watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);
    let facts = watcher.read_facts().unwrap();
    assert_eq!(facts.source.health, DeviceHealth::DeviceUnavailable);
    assert_eq!(facts.sink.health, DeviceHealth::DeviceUnavailable);
    assert!(facts.source.current_default.is_none());
    assert!(facts.sink.current_default.is_none());
    assert_eq!(
        runner
            .calls
            .lock()
            .unwrap()
            .iter()
            .map(|(args, _)| args.clone())
            .collect::<Vec<_>>(),
        [
            args(&["--format=json", "list", "sources"]),
            args(&["--format=json", "list", "sinks"])
        ]
    );
}

#[derive(Clone)]
struct FreshFactsRunner {
    snapshot: Arc<Mutex<ProvenanceRunner>>,
    calls: RecordedDeviceCalls,
}

impl FreshFactsRunner {
    fn new() -> Self {
        Self {
            snapshot: Arc::new(Mutex::new(provenance_runner("alsa", None))),
            calls: Arc::default(),
        }
    }
}

impl CommandRunner for FreshFactsRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        self.calls.lock().unwrap().push((args.to_vec(), deadline));
        let snapshot = self.snapshot.lock().unwrap();
        let view: Vec<_> = args.iter().map(String::as_str).collect();
        let result = match view.as_slice() {
            ["--format=json", "list", "sources"] => snapshot.sources.to_string(),
            ["--format=json", "list", "sinks"] => snapshot.sinks.to_string(),
            ["get-default-source"] => snapshot.sources[0]["name"].as_str().unwrap().to_owned(),
            ["get-default-sink"] => snapshot.sinks[0]["name"].as_str().unwrap().to_owned(),
            _ => panic!("unexpected device effect: {args:?}"),
        };
        Ok(CommandResult::success(result.into_bytes()))
    }
}

#[derive(Clone)]
struct ObservingValidator {
    calls: Arc<Mutex<Vec<String>>>,
    accepted: Arc<AtomicBool>,
    delay: Duration,
}

impl ObservingValidator {
    fn new() -> Self {
        Self {
            calls: Arc::default(),
            accepted: Arc::new(AtomicBool::new(true)),
            delay: Duration::ZERO,
        }
    }
}

impl SinkGraphValidator for ObservingValidator {
    fn validate(&self, sink: &PhysicalDevice) -> bool {
        self.calls.lock().unwrap().push(sink.name.clone());
        std::thread::sleep(self.delay);
        self.accepted.load(Ordering::SeqCst)
    }
}

#[test]
fn readonly_source_pin_stays_pending_then_unavailable_without_migration() {
    let runner = FreshFactsRunner::new();
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);
    watcher.reconcile(DeviceOverride::default()).unwrap();
    {
        let mut snapshot = runner.snapshot.lock().unwrap();
        let old = snapshot.sources[0].clone();
        snapshot.sources[0]["name"] = "new-input".into();
        snapshot.sources[0]["index"] = 99.into();
        snapshot.sources.as_array_mut().unwrap().push(old);
    }
    let facts = watcher.read_facts().unwrap();
    assert_eq!(facts.source.selected.unwrap().name, "external-input");
    assert_eq!(facts.source.pending_default.as_deref(), Some("new-input"));
    runner
        .snapshot
        .lock()
        .unwrap()
        .sources
        .as_array_mut()
        .unwrap()
        .pop();
    let facts = watcher.read_facts().unwrap();
    assert_eq!(facts.source.health, DeviceHealth::DeviceUnavailable);
    assert!(facts.source.selected.is_none());
    assert_eq!(facts.source.pinned_name.as_deref(), Some("external-input"));
    let reconciled = watcher.reconcile(DeviceOverride::default()).unwrap();
    assert_eq!(reconciled.source, facts.source);
}

#[test]
fn required_default_empty_text_is_discovery_failure_not_empty_devices() {
    for phase in [2, 3] {
        let fixture = provenance_runner("alsa", None);
        let mut expected = snapshot(
            "external-output",
            "external-input",
            fixture.sinks,
            fixture.sources,
        );
        expected[phase].result = Ok(CommandResult::success(b" \n\t".to_vec()));
        expected.truncate(phase + 1);
        let runner = FakeRunner::new(expected);
        let watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);
        let error = watcher.read_facts().unwrap_err();
        assert_eq!(error.code(), DeviceWatcherErrorCode::DiscoveryFailed);
        assert_eq!(watcher.selected_sink_name(), None);
        runner.assert_drained();
    }
}

#[test]
fn readonly_device_facts_do_not_pin_defaults_and_validate_each_read() {
    let runner = FreshFactsRunner::new();
    let validator = ObservingValidator::new();
    let watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );
    let deadline = Instant::now() + Duration::from_secs(1);
    let first = watcher.read_facts_until(deadline).unwrap();
    assert_eq!(first.sink.selected.unwrap().name, "external-output");
    assert_eq!(watcher.selected_sink_name(), None);
    runner.snapshot.lock().unwrap().sinks[0]["name"] = "new-output".into();
    runner.snapshot.lock().unwrap().sources[0]["name"] = "new-input".into();
    let second = watcher.read_facts_until(deadline).unwrap();
    assert_eq!(second.sink.selected.unwrap().name, "new-output");
    assert_eq!(second.source.selected.unwrap().name, "new-input");
    assert_eq!(watcher.selected_sink_name(), None);
    assert_eq!(
        *validator.calls.lock().unwrap(),
        ["external-output", "new-output"]
    );
    assert!(
        runner
            .calls
            .lock()
            .unwrap()
            .iter()
            .all(|(_, observed)| *observed == deadline)
    );
}

#[test]
fn readonly_device_facts_preserve_pins_on_default_change_and_removal() {
    let runner = FreshFactsRunner::new();
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);
    watcher.reconcile(DeviceOverride::default()).unwrap();
    {
        let mut snapshot = runner.snapshot.lock().unwrap();
        let old = snapshot.sinks[0].clone();
        snapshot.sinks[0]["name"] = "new-output".into();
        snapshot.sinks[0]["index"] = 99.into();
        snapshot.sinks.as_array_mut().unwrap().push(old);
    }
    let facts = watcher.read_facts().unwrap();
    assert_eq!(facts.sink.selected.unwrap().name, "external-output");
    assert_eq!(facts.sink.current_default.as_deref(), Some("new-output"));
    assert_eq!(facts.sink.pending_default.as_deref(), Some("new-output"));
    runner
        .snapshot
        .lock()
        .unwrap()
        .sinks
        .as_array_mut()
        .unwrap()
        .pop();
    let facts = watcher.read_facts().unwrap();
    assert_eq!(facts.sink.health, DeviceHealth::DeviceUnavailable);
    assert!(facts.sink.selected.is_none());
    assert_eq!(watcher.selected_sink_name(), Some("external-output"));
}

#[test]
fn readonly_device_failure_does_not_change_validation_state_or_pins() {
    let runner = FreshFactsRunner::new();
    let validator = ObservingValidator::new();
    let mut watcher =
        PulseDeviceWatcher::with_validator(runner, AecCapability::Unavailable, validator.clone());
    watcher.reconcile(DeviceOverride::default()).unwrap();
    validator.accepted.store(false, Ordering::SeqCst);
    let error = watcher.read_facts().unwrap_err();
    assert_eq!(error.code(), DeviceWatcherErrorCode::GraphValidationFailed);
    assert_eq!(watcher.selected_sink_name(), Some("external-output"));
    validator.accepted.store(true, Ordering::SeqCst);
    watcher.reconcile(DeviceOverride::default()).unwrap();
    assert_eq!(
        validator.calls.lock().unwrap().len(),
        2,
        "read failure changed reconciliation validation flag"
    );
    watcher.read_facts().unwrap();
    assert_eq!(validator.calls.lock().unwrap().len(), 3);
}

#[test]
fn readonly_device_facts_skip_default_source_when_only_virtual_sources_exist() {
    let runner = FreshFactsRunner::new();
    runner.snapshot.lock().unwrap().sources = serde_json::json!([{
        "index": 90, "name": "external-filter", "monitor_source": "external-output",
        "properties": {"device.class": "monitor"}
    }]);
    let watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);
    let facts = watcher.read_facts().unwrap();
    assert_eq!(facts.source.health, DeviceHealth::DeviceUnavailable);
    assert!(facts.source.current_default.is_none());
    assert_eq!(facts.sink.health, DeviceHealth::Available);
    assert!(
        !runner
            .calls
            .lock()
            .unwrap()
            .iter()
            .any(|(args, _)| args[0] == "get-default-source")
    );
}

#[test]
fn expired_device_read_performs_no_discovery_or_validation() {
    let runner = FreshFactsRunner::new();
    let validator = ObservingValidator::new();
    let watcher = PulseDeviceWatcher::with_validator(
        runner.clone(),
        AecCapability::Unavailable,
        validator.clone(),
    );
    let error = watcher.read_facts_until(Instant::now()).unwrap_err();
    assert_eq!(error.code(), DeviceWatcherErrorCode::DeadlineExpired);
    assert!(runner.calls.lock().unwrap().is_empty());
    assert!(validator.calls.lock().unwrap().is_empty());
}

#[test]
fn late_device_validation_does_not_commit_either_proposed_pin() {
    let runner = FreshFactsRunner::new();
    let mut validator = ObservingValidator::new();
    validator.delay = Duration::from_millis(150);
    let validator_calls = validator.calls.clone();
    let mut watcher =
        PulseDeviceWatcher::with_validator(runner.clone(), AecCapability::Unavailable, validator);
    let deadline = Instant::now() + Duration::from_millis(100);
    let error = watcher
        .reconcile_until(DeviceOverride::default(), deadline)
        .unwrap_err();
    assert_eq!(error.code(), DeviceWatcherErrorCode::DeadlineExpired);
    assert_eq!(*validator_calls.lock().unwrap(), ["external-output"]);
    assert!(
        runner
            .calls
            .lock()
            .unwrap()
            .iter()
            .all(|(_, observed)| *observed == deadline)
    );
    assert_eq!(watcher.selected_sink_name(), None);
    runner.snapshot.lock().unwrap().sinks[0]["name"] = "new-output".into();
    runner.snapshot.lock().unwrap().sources[0]["name"] = "new-input".into();
    let accepted = watcher.reconcile(DeviceOverride::default()).unwrap();
    assert_eq!(accepted.source.selected.unwrap().name, "new-input");
    assert_eq!(accepted.sink.selected.unwrap().name, "new-output");
}

impl CommandRunner for ProvenanceRunner {
    fn run_until(
        &self,
        program: &str,
        arguments: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        let arguments: Vec<_> = arguments.iter().map(String::as_str).collect();
        let output = match arguments.as_slice() {
            ["get-default-sink"] => "external-output".to_owned(),
            ["get-default-source"] => "external-input".to_owned(),
            ["--format=json", "list", "sinks"] => self.sinks.to_string(),
            ["--format=json", "list", "sources"] => self.sources.to_string(),
            _ => panic!("unexpected discovery command"),
        };
        Ok(CommandResult::success(output.into_bytes()))
    }
}

fn provenance_runner(api: &str, bus: Option<&str>) -> ProvenanceRunner {
    let mut runner = ProvenanceRunner {
        sinks: serde_json::json!([{
            "index": 41, "name": "external-output", "state": "SUSPENDED", "monitor_source": "external-output.monitor",
            "properties": {"device.class": "sound", "device.api": api, "media.class": "Audio/Sink"},
            "ports": [{"name": "analog-output-headphones", "type": "Headphones", "availability": "available"}],
            "active_port": "analog-output-headphones"
        }]),
        sources: serde_json::json!([{
            "index": 42, "name": "external-input", "state": "SUSPENDED",
            "properties": {"device.class": "sound", "device.api": api, "media.class": "Audio/Source"},
            "monitor_source": "", "ports": [], "active_port": null
        }]),
    };
    if let Some(bus) = bus {
        runner.sinks[0]["properties"]["device.bus"] = bus.into();
        runner.sources[0]["properties"]["device.bus"] = bus.into();
    }
    runner
}

#[test]
fn physical_provenance_accepts_supported_backends_without_requiring_bus() {
    for (api, bus) in [
        ("alsa", Some("pci")),
        ("alsa", Some("usb")),
        ("bluez", Some("bluetooth")),
        ("bluez5", None),
        ("alsa", None),
    ] {
        let mut watcher =
            PulseDeviceWatcher::new(provenance_runner(api, bus), AecCapability::Unavailable);
        let state = watcher.reconcile(DeviceOverride::default()).unwrap();
        assert_eq!(
            state.source.health,
            DeviceHealth::Available,
            "{api}/{bus:?}"
        );
        assert_eq!(state.sink.health, DeviceHealth::Available, "{api}/{bus:?}");
    }
}

#[test]
fn physical_provenance_rejects_untrusted_endpoint_metadata() {
    let mut failures = Vec::new();
    for kind in ["sink", "source"] {
        for case in [
            "missing_api",
            "unknown_api",
            "missing_class",
            "filter_class",
            "virtual",
            "network",
            "malformed_virtual",
            "conflicting_media",
            "monitor_identity",
            "duplicate_name",
            "duplicate_id",
        ] {
            if case == "monitor_identity" && kind == "sink" {
                continue;
            }
            let mut runner = provenance_runner("alsa", Some("usb"));
            let endpoints = if kind == "sink" {
                &mut runner.sinks
            } else {
                &mut runner.sources
            };
            let device = &mut endpoints[0];
            match case {
                "missing_api" => {
                    device["properties"]
                        .as_object_mut()
                        .unwrap()
                        .remove("device.api");
                }
                "unknown_api" => device["properties"]["device.api"] = "custom".into(),
                "missing_class" => {
                    device["properties"]
                        .as_object_mut()
                        .unwrap()
                        .remove("device.class");
                }
                "filter_class" => device["properties"]["device.class"] = "filter".into(),
                "virtual" => {
                    device["properties"]["node.virtual"] = "true".into();
                    device["flags"] = serde_json::json!(["HARDWARE"]);
                }
                "network" => device["properties"]["node.network"] = "true".into(),
                "malformed_virtual" => device["properties"]["node.virtual"] = "unknown".into(),
                "conflicting_media" => {
                    device["properties"]["media.class"] = if kind == "sink" {
                        "Audio/Source"
                    } else {
                        "Audio/Sink"
                    }
                    .into()
                }
                "monitor_identity" => device["monitor_source"] = "external-output".into(),
                "duplicate_name" | "duplicate_id" => {
                    let mut duplicate = device.clone();
                    if case == "duplicate_name" {
                        duplicate["index"] = 90.into();
                    } else {
                        duplicate["name"] = "other-endpoint".into();
                    }
                    endpoints.as_array_mut().unwrap().push(duplicate);
                }
                _ => unreachable!(),
            }
            let mut watcher = PulseDeviceWatcher::new(runner, AecCapability::Unavailable);
            let result = watcher.reconcile(DeviceOverride {
                source_name: Some("external-input".to_owned()),
                sink_name: Some("external-output".to_owned()),
            });
            let expected = if matches!(case, "duplicate_name" | "duplicate_id") {
                DeviceWatcherErrorCode::DiscoveryFailed
            } else {
                DeviceWatcherErrorCode::InvalidPhysicalDevice
            };
            if result.as_ref().err().map(DeviceWatcherError::code) != Some(expected)
                || watcher.selected_sink_name().is_some()
            {
                failures.push(format!(
                    "{kind}/{case}: expected {expected:?} with unchanged pins"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "untrusted metadata admitted: {failures:?}"
    );
}

#[derive(Clone)]
struct Task7FactsRunner {
    snapshot: serde_json::Value,
    modules: String,
    deadline: std::time::Instant,
    calls: Arc<Mutex<Vec<Vec<String>>>>,
    delay_first: bool,
}

struct InstalledModuleShapeRunner {
    inner: Task7FactsRunner,
    calls: Arc<Mutex<Vec<Vec<String>>>>,
}

impl CommandRunner for InstalledModuleShapeRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        assert_eq!(deadline, self.inner.deadline);
        self.calls.lock().unwrap().push(args.to_vec());
        if args == ["--format=json", "list", "modules"] {
            return Ok(CommandResult::success(br#"[{"name":"module-null-sink","argument":"sink_name=translator_task7_ru_in","usage_counter":"n/a","properties":{}}]"#.to_vec()));
        }
        if args == ["list", "short", "modules"] {
            return Ok(CommandResult::success(b"99\tmodule-foreign\tconfig={\n\tkey=value\n}\t\n70\tmodule-null-sink\tsink_name=translator_task7_ru_in\t\n81\tmodule-empty\t\t\n".to_vec()));
        }
        self.inner.run_until(program, args, deadline)
    }
}

#[test]
fn task7_accepts_installed_pactl_module_shape_without_guessing_array_index() {
    let runner = InstalledModuleShapeRunner {
        inner: Task7FactsRunner::valid(),
        calls: Arc::default(),
    };
    let facts = translator_audio::inspect_task7_endpoints_until(
        &runner,
        "translator_task7_ru_in.monitor",
        "physical-output",
        runner.inner.deadline,
    )
    .expect("installed pactl module inventory must admit linked raw facts");
    assert_eq!(facts.input_monitor, "translator_task7_ru_in.monitor");
    assert_eq!(facts.output.name, "physical-output");
    assert_eq!(
        *runner.calls.lock().unwrap(),
        [
            args(&["--format=json", "list", "sources"]),
            args(&["--format=json", "list", "sinks"]),
            args(&["list", "short", "modules"]),
        ]
    );
}

impl Task7FactsRunner {
    fn valid() -> Self {
        Self {
            snapshot: serde_json::json!({
                "sources": [{"index": 62, "name": "translator_task7_ru_in.monitor", "owner_module": 70,
                    "monitor_source": "translator_task7_ru_in", "properties": {"device.class": "monitor"}, "active_port": null}],
                "sinks": [
                    {"index": 61, "name": "translator_task7_ru_in", "owner_module": 70, "monitor_source": "translator_task7_ru_in.monitor",
                        "properties": {"device.class": "sound", "translator.task7_e2e": "true"}, "active_port": null},
                    {"index": 41, "name": "physical-output", "owner_module": 80, "monitor_source": "physical-output.monitor",
                        "properties": {"device.api": "alsa", "device.class": "sound", "media.class": "Audio/Sink"},
                        "active_port": "analog-output-headphones", "ports": [{"name": "analog-output-headphones", "type": "Headphones"}]}
                ]
            }),
            modules: "70\tmodule-null-sink\tsink_name=translator_task7_ru_in rate=16000 channels=1 channel_map=mono sink_properties=device.description=Translator_Task7_Input translator.task7_e2e=true\t\n".to_owned(),
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(1),
            calls: Arc::default(),
            delay_first: false,
        }
    }

    fn inspect(&self) -> Result<translator_audio::Task7EndpointFacts, DeviceWatcherError> {
        translator_audio::inspect_task7_endpoints_until(
            self,
            "translator_task7_ru_in.monitor",
            "physical-output",
            self.deadline,
        )
    }
}

impl CommandRunner for Task7FactsRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        assert_eq!(deadline, self.deadline);
        let payload = match args
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["--format=json", "list", kind @ ("sources" | "sinks")] => {
                self.snapshot[kind].to_string()
            }
            ["list", "short", "modules"] => self.modules.clone(),
            _ => panic!("unexpected Task7 command"),
        };
        self.calls.lock().unwrap().push(args.to_vec());
        if self.delay_first && self.calls.lock().unwrap().len() == 1 {
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
        Ok(CommandResult::success(payload.into_bytes()))
    }
}

#[test]
fn task7_endpoint_facts_follow_linked_temporary_sink_metadata() {
    let runner = Task7FactsRunner::valid();
    let facts = runner.inspect().unwrap();
    assert_eq!(facts.input_monitor, "translator_task7_ru_in.monitor");
    assert_eq!(facts.output.name, "physical-output");
    assert_eq!(facts.output_mode, OutputMode::Headphones);
    assert_eq!(runner.calls.lock().unwrap().len(), 3);
}

#[test]
fn task7_endpoint_facts_reject_broken_monitor_and_module_links() {
    for case in [
        "monitor_name",
        "sink_name",
        "monitor_link",
        "monitor_empty",
        "source_owner",
        "sink_owner",
        "module_kind",
        "module_missing",
        "marker_missing",
        "virtual_output",
        "duplicate_source",
        "duplicate_sink",
        "duplicate_module",
    ] {
        let mut runner = Task7FactsRunner::valid();
        match case {
            "monitor_name" => runner.snapshot["sources"][0]["name"] = "arbitrary.monitor".into(),
            "sink_name" => runner.snapshot["sinks"][0]["name"] = "arbitrary".into(),
            "monitor_link" => {
                runner.snapshot["sources"][0]["monitor_source"] = "physical-output".into()
            }
            "monitor_empty" => runner.snapshot["sources"][0]["monitor_source"] = "".into(),
            "source_owner" => runner.snapshot["sources"][0]["owner_module"] = 71.into(),
            "sink_owner" => runner.snapshot["sinks"][0]["owner_module"] = 71.into(),
            "module_kind" => {
                runner.modules = runner
                    .modules
                    .replace("module-null-sink", "module-loopback")
            }
            "module_missing" => runner.modules.clear(),
            "marker_missing" => {
                runner.snapshot["sinks"][0]["properties"]
                    .as_object_mut()
                    .unwrap()
                    .remove("translator.task7_e2e");
            }
            "virtual_output" => {
                runner.snapshot["sinks"][1]["properties"]["node.virtual"] = "true".into()
            }
            "duplicate_module" => runner.modules.push_str(&runner.modules.clone()),
            "duplicate_source" | "duplicate_sink" => {
                let kind = match case {
                    "duplicate_source" => "sources",
                    "duplicate_sink" => "sinks",
                    _ => unreachable!(),
                };
                let duplicate = runner.snapshot[kind][0].clone();
                runner.snapshot[kind]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
            _ => unreachable!(),
        }
        assert!(
            runner.inspect().is_err(),
            "accepted invalid Task7 facts: {case}"
        );
        assert!(runner.calls.lock().unwrap().len() <= 3);
    }
}

#[test]
fn task7_each_read_phase_preserves_error_and_deadline() {
    let expected = [
        args(&["--format=json", "list", "sources"]),
        args(&["--format=json", "list", "sinks"]),
        args(&["list", "short", "modules"]),
    ];
    for phase in 1..=expected.len() {
        for fault in read_faults() {
            let mut inner = Task7FactsRunner::valid();
            inner.deadline = Instant::now() + Duration::from_millis(100);
            let deadline = inner.deadline;
            let calls = Arc::default();
            let runner = FaultingReadRunner {
                inner,
                phase,
                fault: fault.clone(),
                calls: Arc::clone(&calls),
            };
            let error = translator_audio::inspect_task7_endpoints_until(
                &runner,
                "translator_task7_ru_in.monitor",
                "physical-output",
                deadline,
            )
            .unwrap_err();
            assert_failed_read_prefix(&error, &fault, &calls, &expected[..phase], deadline);
        }
    }
}

#[test]
fn task7_schema_and_identity_errors_are_typed() {
    for (path, value, expected) in [
        (
            "/sinks/0/properties/translator.task7_e2e",
            serde_json::json!("false"),
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
        ),
        (
            "/sinks/0/properties/translator.task7_e2e",
            serde_json::json!("unknown"),
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
        ),
        (
            "/sinks/0/properties/translator.task7_e2e",
            serde_json::json!(true),
            DeviceWatcherErrorCode::DiscoveryFailed,
        ),
        (
            "/sources/0/monitor_source",
            serde_json::Value::Null,
            DeviceWatcherErrorCode::DiscoveryFailed,
        ),
        (
            "/sources/0/monitor_source",
            serde_json::json!(61),
            DeviceWatcherErrorCode::DiscoveryFailed,
        ),
        (
            "/sources/0/owner_module",
            serde_json::Value::Null,
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
        ),
        (
            "/sources/0/owner_module",
            serde_json::json!("70"),
            DeviceWatcherErrorCode::DiscoveryFailed,
        ),
        (
            "/sinks/0/owner_module",
            serde_json::Value::Null,
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
        ),
        (
            "/sinks/0/owner_module",
            serde_json::json!("70"),
            DeviceWatcherErrorCode::DiscoveryFailed,
        ),
        (
            "/sources",
            serde_json::json!([]),
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
        ),
        (
            "/sinks/1/name",
            serde_json::json!("missing-output"),
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
        ),
        (
            "/sinks/1/properties/device.api",
            serde_json::json!("custom"),
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
        ),
    ] {
        let mut runner = Task7FactsRunner::valid();
        *runner.snapshot.pointer_mut(path).unwrap() = value;
        assert_eq!(
            runner.inspect().unwrap_err().code(),
            expected,
            "path {path}"
        );
        assert!(!runner.calls.lock().unwrap().is_empty());
    }
    for modules in ["\"70\"\tmodule-null-sink\targs\t\n", "70\t\targs\t\n"] {
        let mut runner = Task7FactsRunner::valid();
        runner.modules = modules.to_owned();
        assert_eq!(
            runner.inspect().unwrap_err().code(),
            DeviceWatcherErrorCode::DiscoveryFailed
        );
        assert_eq!(runner.calls.lock().unwrap().len(), 3);
    }
    for (kind, index) in [("sources", 0), ("sinks", 0), ("sinks", 1)] {
        let mut runner = Task7FactsRunner::valid();
        runner.snapshot[kind][index]["state"] = "UNAVAILABLE".into();
        assert_eq!(
            runner.inspect().unwrap_err().code(),
            DeviceWatcherErrorCode::InvalidPhysicalDevice
        );
        assert_eq!(runner.calls.lock().unwrap().len(), 3);
    }
}

#[test]
fn task7_preserves_raw_output_modes_without_acoustic_admission() {
    for (port, port_type, expected) in [
        (
            "analog-output-headphones",
            "Headphones",
            OutputMode::Headphones,
        ),
        ("analog-output-speaker", "Speaker", OutputMode::OpenSpeaker),
        ("analog-output", "Analog", OutputMode::UnknownUnsafe),
    ] {
        let mut runner = Task7FactsRunner::valid();
        runner.snapshot["sinks"][0]["properties"]["translator.task7_owner"] =
            "0123456789abcdef0123456789abcdef".into();
        runner.snapshot["sinks"][1]["active_port"] = port.into();
        runner.snapshot["sinks"][1]["ports"][0]["name"] = port.into();
        runner.snapshot["sinks"][1]["ports"][0]["type"] = port_type.into();
        let facts = runner.inspect().unwrap();
        assert_eq!(facts.output_mode, expected);
        assert_eq!(facts.output.name, "physical-output");
        assert_eq!(facts.input_monitor, "translator_task7_ru_in.monitor");
        assert_eq!(
            *runner.calls.lock().unwrap(),
            [
                args(&["--format=json", "list", "sources"]),
                args(&["--format=json", "list", "sinks"]),
                args(&["list", "short", "modules"]),
            ]
        );
    }
}

#[test]
fn task7_endpoint_inspection_rejects_expiry_before_discovery() {
    let mut runner = Task7FactsRunner::valid();
    runner.deadline = std::time::Instant::now();
    let result = runner.inspect();
    assert_eq!(
        result.unwrap_err().code(),
        DeviceWatcherErrorCode::DeadlineExpired
    );
    assert!(runner.calls.lock().unwrap().is_empty());
}

#[test]
fn task7_endpoint_inspection_stops_after_a_command_crosses_deadline() {
    let mut runner = Task7FactsRunner::valid();
    runner.deadline = std::time::Instant::now() + std::time::Duration::from_millis(15);
    runner.delay_first = true;
    let result = runner.inspect();
    assert_eq!(
        result.unwrap_err().code(),
        DeviceWatcherErrorCode::DeadlineExpired
    );
    assert_eq!(runner.calls.lock().unwrap().len(), 1);
}

#[test]
fn task7_endpoint_inspection_rejects_arbitrary_capture_before_discovery() {
    let runner = Task7FactsRunner::valid();
    let result = translator_audio::inspect_task7_endpoints_until(
        &runner,
        "physical-input",
        "physical-output",
        runner.deadline,
    );
    assert_eq!(
        result.unwrap_err().code(),
        DeviceWatcherErrorCode::InvalidPhysicalDevice
    );
    assert!(runner.calls.lock().unwrap().is_empty());
}

// Literal pactl 16.1 wire records, independent of the JSON fixture builders.
const LITERAL_TASK7_SOURCES: &[u8] = br#"[{"index":62,"name":"translator_task7_ru_in.monitor","owner_module":70,"monitor_source":"translator_task7_ru_in","state":"IDLE","properties":{"device.class":"monitor"},"active_port":null,"driver":"module-null-sink.c","mute":false}]"#;
const LITERAL_TASK7_SINKS: &[u8] = br#"[{"index":61,"name":"translator_task7_ru_in","owner_module":70,"monitor_source":"translator_task7_ru_in.monitor","properties":{"device.class":"sound","translator.task7_e2e":"true","translator.task7_owner":"0123456789abcdef0123456789abcdef"},"active_port":null},{"index":41,"name":"physical-output","owner_module":80,"monitor_source":"physical-output.monitor","description":"Output description","state":"SUSPENDED","properties":{"device.class":"sound","device.api":"alsa","media.class":"Audio/Sink"},"ports":[{"name":"headphones","type":"Headphones","availability":"available"}],"active_port":"headphones"}]"#;
const LITERAL_PHYSICAL_SOURCES: &[u8] = br#"[{"index":42,"name":"physical-input","monitor_source":"","description":"Input description","state":"SUSPENDED","properties":{"device.class":"sound","device.api":"alsa","media.class":"Audio/Source"},"active_port":null,"flags":["HARDWARE"]}]"#;
const LITERAL_PHYSICAL_SINKS: &[u8] = br#"[{"index":41,"name":"physical-output","monitor_source":"physical-output.monitor","description":"Output description","state":"SUSPENDED","properties":{"device.class":"sound","device.api":"alsa","media.class":"Audio/Sink"},"ports":[{"name":"headphones","type":"Headphones","availability":"available"}],"active_port":"headphones"}]"#;

struct RecordedRawRunner {
    inner: FakeRunner,
    calls: RecordedDeviceCalls,
}

impl CommandRunner for RecordedRawRunner {
    fn run_until(
        &self,
        program: &str,
        arguments: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        self.calls
            .lock()
            .unwrap()
            .push((arguments.to_vec(), deadline));
        self.inner.run_until(program, arguments, deadline)
    }
}

fn literal_read(arguments: &[&str], payload: &[u8]) -> ExpectedCommand {
    ExpectedCommand {
        args: args(arguments),
        result: Ok(CommandResult::success(payload.to_vec())),
    }
}

#[test]
fn c1_task7_accepts_literal_reciprocal_monitor_name_wire_records() {
    let runner = RecordedRawRunner {
        inner: FakeRunner::new(vec![
            literal_read(&["--format=json", "list", "sources"], LITERAL_TASK7_SOURCES),
            literal_read(&["--format=json", "list", "sinks"], LITERAL_TASK7_SINKS),
            literal_read(&["list", "short", "modules"], b"99\tmodule-foreign\t\t\n70\tmodule-null-sink\tsink_name=translator_task7_ru_in\t\n"),
        ]),
        calls: Arc::default(),
    };
    let deadline = Instant::now() + Duration::from_secs(1);
    let facts = translator_audio::inspect_task7_endpoints_until(
        &runner,
        "translator_task7_ru_in.monitor",
        "physical-output",
        deadline,
    )
    .expect("actual string monitor links must admit the owned Task7 pair");
    assert_eq!(facts.input_monitor, "translator_task7_ru_in.monitor");
    assert_eq!(facts.output.name, "physical-output");
    assert_eq!(facts.output.id, 41);
    assert_eq!(facts.output.description, "Output description");
    assert_eq!(facts.output.active_port.as_deref(), Some("headphones"));
    assert_eq!(facts.output_mode, OutputMode::Headphones);
    assert!(facts.output.available);
    runner.inner.assert_drained();
    let calls = runner.calls.lock().unwrap();
    assert_eq!(calls.len(), 3);
    assert!(calls.iter().all(|(_, observed)| *observed == deadline));
}

#[test]
fn c4_physical_empty_link_and_sink_monitor_link_preserve_selection() {
    let mut commands = Vec::new();
    for _ in 0..2 {
        commands.extend([
            literal_read(
                &["--format=json", "list", "sources"],
                LITERAL_PHYSICAL_SOURCES,
            ),
            literal_read(&["--format=json", "list", "sinks"], LITERAL_PHYSICAL_SINKS),
            literal_read(&["get-default-source"], b"physical-input\n"),
            literal_read(&["get-default-sink"], b"physical-output\n"),
        ]);
    }
    let runner = RecordedRawRunner {
        inner: FakeRunner::new(commands),
        calls: Arc::default(),
    };
    let calls = Arc::clone(&runner.calls);
    let raw = runner.inner.clone();
    let mut watcher = PulseDeviceWatcher::new(runner, AecCapability::Unavailable);
    let deadline = Instant::now() + Duration::from_secs(1);
    let read = watcher.read_facts_until(deadline).unwrap();
    assert_eq!(
        read.source.selected.as_ref().unwrap().name,
        "physical-input"
    );
    assert_eq!(
        read.source.selected.as_ref().unwrap().description,
        "Input description"
    );
    assert_eq!(read.sink.selected.as_ref().unwrap().name, "physical-output");
    assert_eq!(read.output_mode, OutputMode::Headphones);
    assert_eq!(watcher.selected_sink_name(), None);
    let committed = watcher
        .reconcile_until(
            DeviceOverride {
                source_name: Some("physical-input".into()),
                sink_name: Some("physical-output".into()),
            },
            deadline,
        )
        .unwrap();
    assert_eq!(committed.source.selected, read.source.selected);
    assert_eq!(committed.sink.selected, read.sink.selected);
    assert_eq!(
        committed.source.pinned_name.as_deref(),
        Some("physical-input")
    );
    assert_eq!(watcher.selected_sink_name(), Some("physical-output"));
    assert_eq!(calls.lock().unwrap().len(), 8);
    assert!(
        calls
            .lock()
            .unwrap()
            .iter()
            .all(|(_, observed)| *observed == deadline)
    );
    raw.assert_drained();
}

#[test]
fn c2_task7_rejects_each_wrong_string_link_independently() {
    for (kind, wrong) in [
        ("sources", "physical-output"),
        ("sinks", "physical-output.monitor"),
        ("sources", ""),
        ("sinks", ""),
        ("sources", "translator_task7_ru_in.monitor"),
        ("sinks", "translator_task7_ru_in"),
        ("sources", "61"),
        ("sinks", "62"),
    ] {
        let mut runner = Task7FactsRunner::valid();
        runner.snapshot[kind][0]["monitor_source"] = wrong.into();
        assert_eq!(
            runner.inspect().unwrap_err().code(),
            DeviceWatcherErrorCode::InvalidPhysicalDevice,
            "{kind}/{wrong}"
        );
        assert_eq!(runner.calls.lock().unwrap().len(), 3);
    }
}

fn malformed_monitor_values() -> [Option<serde_json::Value>; 7] {
    [
        None,
        Some(serde_json::Value::Null),
        Some(61.into()),
        Some(true.into()),
        Some(serde_json::json!([])),
        Some(serde_json::json!({})),
        Some(serde_json::json!({"private-device-marker": 1})),
    ]
}

fn corrupt_monitor_field(device: &mut serde_json::Value, value: Option<serde_json::Value>) {
    let object = device.as_object_mut().unwrap();
    object.remove("monitor_source");
    if let Some(value) = value {
        object.insert("monitor_source".into(), value);
    }
    // The obsolete field cannot supply the missing authority or override bad wire data.
    object.insert("monitor_of_sink".into(), serde_json::Value::Null);
}

#[test]
fn c3_task7_requires_string_monitor_metadata_for_the_entire_inventory() {
    let mut failures = Vec::new();
    for kind in ["sources", "sinks"] {
        for unrelated in [false, true] {
            for (case, value) in malformed_monitor_values().into_iter().enumerate() {
                let mut runner = Task7FactsRunner::valid();
                let index = if unrelated {
                    let mut extra = runner.snapshot[kind][0].clone();
                    extra["name"] = "foreign-endpoint".into();
                    extra["index"] = 99.into();
                    runner.snapshot[kind].as_array_mut().unwrap().push(extra);
                    runner.snapshot[kind].as_array().unwrap().len() - 1
                } else {
                    0
                };
                corrupt_monitor_field(&mut runner.snapshot[kind][index], value);
                if kind == "sources" {
                    runner.snapshot[kind][index]["monitor_of_sink"] = 61.into();
                }
                let result = runner.inspect();
                if result.as_ref().err().map(DeviceWatcherError::code)
                    != Some(DeviceWatcherErrorCode::DiscoveryFailed)
                {
                    failures.push(format!(
                        "{kind}/{unrelated}/{case}: missing typed schema failure"
                    ));
                }
                if let Err(error) = result {
                    assert_error_redacted(&error);
                }
                let expected = if kind == "sources" { 1 } else { 2 };
                if runner.calls.lock().unwrap().len() != expected {
                    failures.push(format!(
                        "{kind}/{unrelated}/{case}: read continued after malformed metadata"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[test]
fn c3_physical_discovery_requires_string_monitor_metadata_without_pin_commit() {
    let mut failures = Vec::new();
    for kind in ["sources", "sinks"] {
        for unrelated in [false, true] {
            for (case, value) in malformed_monitor_values().into_iter().enumerate() {
                let runner = FreshFactsRunner::new();
                {
                    let mut snapshot = runner.snapshot.lock().unwrap();
                    let devices = if kind == "sources" {
                        &mut snapshot.sources
                    } else {
                        &mut snapshot.sinks
                    };
                    let index = if unrelated {
                        let mut extra = devices[0].clone();
                        extra["index"] = 99.into();
                        extra["name"] = "foreign-endpoint".into();
                        devices.as_array_mut().unwrap().push(extra);
                        1
                    } else {
                        0
                    };
                    corrupt_monitor_field(&mut devices[index], value);
                    devices[index]["monitor_of_sink"] = u32::MAX.into();
                }
                let validator = ObservingValidator::new();
                let mut watcher = PulseDeviceWatcher::with_validator(
                    runner.clone(),
                    AecCapability::Unavailable,
                    validator.clone(),
                );
                let deadline = Instant::now() + Duration::from_secs(1);
                let result = watcher.reconcile_until(DeviceOverride::default(), deadline);
                if result.as_ref().err().map(DeviceWatcherError::code)
                    != Some(DeviceWatcherErrorCode::DiscoveryFailed)
                    || watcher.selected_sink_name().is_some()
                    || !validator.calls.lock().unwrap().is_empty()
                {
                    failures.push(format!("{kind}/{unrelated}/{case}: malformed inventory admitted or mutated pins/validation"));
                }
                if let Err(error) = result {
                    assert_error_redacted(&error);
                }
                let calls = runner.calls.lock().unwrap();
                if calls.len() != if kind == "sources" { 1 } else { 2 } {
                    failures.push(format!("{kind}/{unrelated}/{case}: continued discovery"));
                }
                assert!(calls.iter().all(|(_, observed)| *observed == deadline));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[test]
fn c5_hardware_looking_unconventional_monitor_is_not_a_physical_input() {
    let mut failures = Vec::new();
    for explicit in [false, true] {
        let runner = FreshFactsRunner::new();
        runner.snapshot.lock().unwrap().sources =
            serde_json::from_slice(LITERAL_PHYSICAL_SOURCES).unwrap();
        {
            let mut snapshot = runner.snapshot.lock().unwrap();
            snapshot.sources[0]["name"] = "external-input".into();
            snapshot.sources[0]["monitor_source"] = "external-output".into();
        }
        let validator = ObservingValidator::new();
        let mut watcher = PulseDeviceWatcher::with_validator(
            runner.clone(),
            AecCapability::Unavailable,
            validator.clone(),
        );
        let deadline = Instant::now() + Duration::from_secs(1);
        if explicit {
            let result = watcher.reconcile_until(
                DeviceOverride {
                    source_name: Some("external-input".into()),
                    sink_name: None,
                },
                deadline,
            );
            if result.as_ref().err().map(DeviceWatcherError::code)
                != Some(DeviceWatcherErrorCode::InvalidPhysicalDevice)
                || !validator.calls.lock().unwrap().is_empty()
                || watcher.selected_sink_name().is_some()
            {
                failures.push("explicit monitor override admitted or validated/pinned".to_owned());
            }
        } else {
            let facts = watcher.read_facts_until(deadline).unwrap();
            if facts.source.health != DeviceHealth::DeviceUnavailable
                || facts.source.selected.is_some()
                || facts.source.current_default.is_some()
            {
                failures.push("hardware-looking monitor exposed as physical source".to_owned());
            }
            assert_eq!(facts.sink.health, DeviceHealth::Available);
            assert_eq!(watcher.selected_sink_name(), None);
        }
        let expected = [
            args(&["--format=json", "list", "sources"]),
            args(&["--format=json", "list", "sinks"]),
            args(&["get-default-sink"]),
        ];
        if runner
            .calls
            .lock()
            .unwrap()
            .iter()
            .map(|(args, _)| args.clone())
            .collect::<Vec<_>>()
            != expected
        {
            failures.push(format!(
                "explicit={explicit}: queried default for monitor source"
            ));
        }
        assert!(
            runner
                .calls
                .lock()
                .unwrap()
                .iter()
                .all(|(_, observed)| *observed == deadline)
        );
        runner.snapshot.lock().unwrap().sources[0]["monitor_source"] = "".into();
        let healthy = watcher
            .reconcile_until(
                DeviceOverride {
                    source_name: Some("external-input".into()),
                    sink_name: None,
                },
                deadline,
            )
            .unwrap();
        assert_eq!(healthy.source.selected.unwrap().name, "external-input");
        assert_eq!(healthy.source.health, DeviceHealth::Available);
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[test]
fn c7_malformed_monitor_reads_preserve_existing_pins_and_validation_until_recovery() {
    let mut failures = Vec::new();
    for kind in ["sources", "sinks"] {
        for reconcile in [false, true] {
            let runner = FreshFactsRunner::new();
            let validator = ObservingValidator::new();
            let mut watcher = PulseDeviceWatcher::with_validator(
                runner.clone(),
                AecCapability::Unavailable,
                validator.clone(),
            );
            let deadline = Instant::now() + Duration::from_secs(1);
            let initial = watcher
                .reconcile_until(DeviceOverride::default(), deadline)
                .unwrap();
            let original = runner.snapshot.lock().unwrap().clone();
            {
                let mut snapshot = runner.snapshot.lock().unwrap();
                let devices = if kind == "sources" {
                    &mut snapshot.sources
                } else {
                    &mut snapshot.sinks
                };
                corrupt_monitor_field(&mut devices[0], None);
            }
            runner.calls.lock().unwrap().clear();
            let before_validation = validator.calls.lock().unwrap().len();
            let result = if reconcile {
                watcher.reconcile_until(DeviceOverride::default(), deadline)
            } else {
                watcher.read_facts_until(deadline)
            };
            if result.as_ref().err().map(DeviceWatcherError::code)
                != Some(DeviceWatcherErrorCode::DiscoveryFailed)
                || validator.calls.lock().unwrap().len() != before_validation
            {
                failures.push(format!(
                    "{kind}/{reconcile}: malformed read accepted or validation changed"
                ));
            }
            assert_eq!(watcher.selected_sink_name(), Some("external-output"));
            let calls = runner.calls.lock().unwrap();
            if calls.len() != if kind == "sources" { 1 } else { 2 } {
                failures.push(format!("{kind}/{reconcile}: continued read"));
            }
            assert!(calls.iter().all(|(_, observed)| *observed == deadline));
            drop(calls);
            {
                let mut snapshot = runner.snapshot.lock().unwrap();
                *snapshot = original;
                let mut changed_default = snapshot.sources[0].clone();
                changed_default["index"] = 99.into();
                changed_default["name"] = "new-default-input".into();
                snapshot
                    .sources
                    .as_array_mut()
                    .unwrap()
                    .insert(0, changed_default);
            }
            let observed = watcher.read_facts_until(deadline).unwrap();
            let recovered = watcher
                .reconcile_until(DeviceOverride::default(), deadline)
                .unwrap();
            for facts in [&observed, &recovered] {
                assert_eq!(facts.source.selected, initial.source.selected);
                assert_eq!(facts.source.pinned_name, initial.source.pinned_name);
                assert_eq!(
                    facts.source.current_default.as_deref(),
                    Some("new-default-input")
                );
                assert_eq!(
                    facts.source.pending_default.as_deref(),
                    Some("new-default-input")
                );
                assert_eq!(facts.sink, initial.sink);
            }
            // The read validates once; reconciliation must retain its earlier committed flag.
            if validator.calls.lock().unwrap().len() != before_validation + 1 {
                failures.push(format!(
                    "{kind}/{reconcile}: committed validation state changed"
                ));
            }
            assert!(
                runner
                    .calls
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|(_, observed)| *observed == deadline)
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

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
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
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

fn snapshot<'a>(
    default_sink: impl Into<Option<&'a str>>,
    default_source: impl Into<Option<&'a str>>,
    sinks: serde_json::Value,
    sources: serde_json::Value,
) -> Vec<ExpectedCommand> {
    let mut commands = vec![
        ExpectedCommand {
            args: args(&["--format=json", "list", "sources"]),
            result: Ok(CommandResult::success(sources.to_string().into_bytes())),
        },
        ExpectedCommand {
            args: args(&["--format=json", "list", "sinks"]),
            result: Ok(CommandResult::success(sinks.to_string().into_bytes())),
        },
    ];
    for (command, default) in [
        ("get-default-source", default_source.into()),
        ("get-default-sink", default_sink.into()),
    ] {
        if let Some(default) = default {
            commands.push(ExpectedCommand {
                args: args(&[command]),
                result: Ok(CommandResult::success(default.as_bytes().to_vec())),
            });
        }
    }
    commands
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
        "monitor_source": format!("{name}.monitor"),
        "description": name,
        "state": "SUSPENDED",
        "properties": {
            "device.class": "sound",
            "device.api": "alsa",
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
        "monitor_source": if device_class == "monitor" { "physical-output" } else { "" },
        "description": name,
        "state": "SUSPENDED",
        "properties": {
            "device.class": device_class,
            "device.api": "alsa",
            "media.class": "Audio/Source"
        },
        "ports": [],
        "active_port": ""
    })
}

#[test]
fn physical_headset_defaults_are_pinned_and_classified() {
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
    assert_eq!(state.output_mode, OutputMode::Headphones);
    runner.assert_drained();
}

#[test]
fn nullable_virtual_active_ports_do_not_invalidate_physical_device_discovery() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let virtual_sink = serde_json::json!({
        "index": 70,
        "name": "translator_mic_out",
        "monitor_source": "translator_mic_out.monitor",
        "description": "Translator_Mic_Out",
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "media.class": "Audio/Sink"},
        "ports": [],
        "active_port": null
    });
    let virtual_source = serde_json::json!({
        "index": 71,
        "name": "translator_virtual_mic",
        "monitor_source": "",
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

    assert_eq!(state.output_mode, OutputMode::Headphones);
    runner.assert_drained();
}

#[test]
fn generic_usb_output_without_port_classification_remains_unknown() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let source_name = "alsa_input.usb-headset.mono-fallback";
    let runner = FakeRunner::new(snapshot(
        sink_name,
        source_name,
        serde_json::json!([sink(50, sink_name, "analog-output", "Analog")]),
        serde_json::json!([source(60, source_name, "sound")]),
    ));
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let state = watcher.reconcile(DeviceOverride::default()).unwrap();

    assert_eq!(state.output_mode, OutputMode::UnknownUnsafe);
    runner.assert_drained();
}

#[test]
fn translator_virtual_default_source_is_never_selected_as_physical() {
    let sink_name = "alsa_output.usb-headset.analog-stereo";
    let runner = FakeRunner::new(snapshot(
        sink_name,
        None,
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
        None,
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
        None,
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
        None,
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
    assert_eq!(state.output_mode, OutputMode::Headphones);
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
fn listed_sink_recovery_requires_graph_revalidation() {
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
        "monitor_source": "",
        "description": source_name,
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "device.api": "alsa", "media.class": "Audio/Source"},
        "ports": [{"name": "analog-input-mic", "type": "Mic", "availability": "available"}],
        "active_port": "analog-input-mic"
    });
    let unavailable_sink = serde_json::json!({
        "index": 50,
        "name": sink_name,
        "monitor_source": format!("{sink_name}.monitor"),
        "description": sink_name,
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "device.api": "alsa", "media.class": "Audio/Sink"},
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
        "monitor_source": "",
        "description": source_name,
        "state": "SUSPENDED",
        "properties": {"device.class": "sound", "device.api": "alsa", "media.class": "Audio/Source"},
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
fn speaker_and_unknown_analog_outputs_preserve_raw_classification() {
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

    assert_eq!(speaker_state.output_mode, OutputMode::OpenSpeaker);
    assert_eq!(unknown_state.output_mode, OutputMode::UnknownUnsafe);
    runner.assert_drained();
}

#[test]
fn validated_aec_does_not_change_raw_output_classification() {
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

    assert_eq!(state.output_mode, OutputMode::OpenSpeaker);
    assert_eq!(unknown_state.output_mode, OutputMode::UnknownUnsafe);
    runner.assert_drained();
}

#[test]
fn unvalidated_and_failed_aec_facts_are_preserved() {
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

    assert_eq!(
        unvalidated_state.aec_capability,
        AecCapability::AvailableUnvalidated
    );
    assert_eq!(failed_state.aec_capability, AecCapability::ValidationFailed);
    assert_eq!(unvalidated_state.output_mode, OutputMode::OpenSpeaker);
    assert_eq!(failed_state.output_mode, OutputMode::OpenSpeaker);

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

    assert_eq!(state.output_mode, OutputMode::UnknownUnsafe);
    runner.assert_drained();
}

#[test]
fn malformed_device_json_returns_a_privacy_safe_error() {
    let runner = FakeRunner::new(vec![ExpectedCommand {
        args: args(&["--format=json", "list", "sources"]),
        result: Ok(CommandResult::success(b"{private-device-marker".to_vec())),
    }]);
    let mut watcher = PulseDeviceWatcher::new(runner.clone(), AecCapability::Unavailable);

    let error = watcher.reconcile(DeviceOverride::default()).unwrap_err();

    assert_eq!(error.code(), DeviceWatcherErrorCode::DiscoveryFailed);
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn raw_aec_validation_retains_its_original_device_pair() {
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

    assert_eq!(state.output_mode, OutputMode::OpenSpeaker);
    assert_eq!(
        state.aec_capability,
        AecCapability::ValidatedFor {
            source_name: old_source.to_owned(),
            sink_name: old_sink.to_owned(),
        }
    );
    validator.assert_drained();
    runner.assert_drained();
}

#[test]
fn nonzero_device_command_returns_a_privacy_safe_error() {
    let runner = FakeRunner::new(vec![ExpectedCommand {
        args: args(&["--format=json", "list", "sources"]),
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
