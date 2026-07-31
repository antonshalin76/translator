use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use translator_audio::{
    AEC_SINK, AEC_SOURCE, AecDeviceMetadata, AecErrorCode, AecFarEndCounters, AecPhysicalPair,
    AecPowerWindow, AecValidationInput, CommandResult, CommandRunError, CommandRunner,
    PulseAecGraph, SystemCommandRunner, evaluate_aec,
};

const GENERATION: &str = "aec-generation-0001";
const SOURCE: &str = "alsa_input.usb-headset.mono-fallback";
const SINK: &str = "alsa_output.pci-speakers.analog-stereo";

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
        assert!(
            self.expected.lock().unwrap().is_empty(),
            "not all expected pactl commands were issued"
        );
    }
}

impl CommandRunner for FakeRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        let expected = self
            .expected
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected pactl command");
        assert_eq!(args, expected.args);
        expected.result
    }
}

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn success(stdout: impl AsRef<[u8]>) -> Result<CommandResult, CommandRunError> {
    Ok(CommandResult::success(stdout.as_ref().to_vec()))
}

fn load_command() -> ExpectedCommand {
    ExpectedCommand {
        args: vec![
            "load-module".to_owned(),
            "module-echo-cancel".to_owned(),
            format!("source_master={SOURCE}"),
            format!("sink_master={SINK}"),
            format!("source_name={AEC_SOURCE}"),
            format!("sink_name={AEC_SINK}"),
            "rate=48000".to_owned(),
            "channels=1".to_owned(),
            "channel_map=mono".to_owned(),
            "aec_method=webrtc".to_owned(),
            format!(
                "source_properties='device.description=Translator_AEC_Source translator.owner=true translator.generation={GENERATION}'"
            ),
            format!(
                "sink_properties='device.description=Translator_AEC_Sink translator.owner=true translator.generation={GENERATION}'"
            ),
        ],
        result: success("73\n"),
    }
}

fn module_command(module_id: u32, source: &str, sink: &str) -> ExpectedCommand {
    let contract = format!(
        "{module_id}\tmodule-echo-cancel\tsource_master={source} sink_master={sink} \
         source_name={AEC_SOURCE} sink_name={AEC_SINK} rate=48000 channels=1 \
         channel_map=mono aec_method=webrtc \
         source_properties='device.description=Translator_AEC_Source \
         translator.owner=true translator.generation={GENERATION}' \
         sink_properties='device.description=Translator_AEC_Sink \
         translator.owner=true translator.generation={GENERATION}'\t0\n"
    );
    ExpectedCommand {
        args: args(&["list", "short", "modules"]),
        result: success(contract),
    }
}

fn endpoint_command(kind: &str, name: &str, owner_module: u32) -> ExpectedCommand {
    let payload = serde_json::json!([{
        "index": if kind == "sources" { 81 } else { 82 },
        "name": name,
        "owner_module": owner_module,
        "properties": {
            "translator.owner": "true",
            "translator.generation": GENERATION
        }
    }]);
    ExpectedCommand {
        args: args(&["--format=json", "list", kind]),
        result: success(payload.to_string()),
    }
}

fn ready_inspection(module_id: u32) -> Vec<ExpectedCommand> {
    vec![
        module_command(module_id, SOURCE, SINK),
        endpoint_command("sources", AEC_SOURCE, module_id),
        endpoint_command("sinks", AEC_SINK, module_id),
    ]
}

fn graph(runner: FakeRunner) -> PulseAecGraph<FakeRunner> {
    PulseAecGraph::new(runner, AecPhysicalPair::new(SOURCE, SINK), GENERATION).unwrap()
}

#[test]
fn load_uses_exact_owned_webrtc_contract_and_inspects_created_graph() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());

    let state = graph.load_owned().unwrap();

    assert_eq!(state.module_id, 73);
    assert_eq!(state.source_id, 81);
    assert_eq!(state.sink_id, 82);
    assert_eq!(state.pair, AecPhysicalPair::new(SOURCE, SINK));
    assert_eq!(state.generation, GENERATION);
    runner.assert_drained();
}

#[test]
fn inspect_rejects_changed_physical_master_pair() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.push(module_command(73, SOURCE, "alsa_output.foreign"));
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();

    let error = graph.inspect_owned().unwrap_err();

    assert_eq!(error.code(), AecErrorCode::OwnershipMismatch);
    runner.assert_drained();
}

#[test]
fn inspect_rejects_endpoint_with_wrong_generation() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.push(module_command(73, SOURCE, SINK));
    let endpoint = serde_json::json!([{
        "index": 81,
        "name": AEC_SOURCE,
        "owner_module": 73,
        "properties": {
            "translator.owner": "true",
            "translator.generation": "stale-generation"
        }
    }]);
    expected.push(ExpectedCommand {
        args: args(&["--format=json", "list", "sources"]),
        result: success(endpoint.to_string()),
    });
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();

    let error = graph.inspect_owned().unwrap_err();

    assert_eq!(error.code(), AecErrorCode::OwnershipMismatch);
    runner.assert_drained();
}

#[test]
fn cleanup_unloads_only_the_exact_owned_module() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.extend(ready_inspection(73));
    expected.push(ExpectedCommand {
        args: args(&["unload-module", "73"]),
        result: success(""),
    });
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();

    assert_eq!(graph.cleanup_owned().unwrap(), Some(73));
    assert!(graph.inspect_owned().is_err());
    runner.assert_drained();
}

#[test]
fn cleanup_fails_closed_when_module_id_was_reused() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.push(module_command(73, SOURCE, "alsa_output.foreign"));
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();

    let error = graph.cleanup_owned().unwrap_err();

    assert_eq!(error.code(), AecErrorCode::CleanupRefused);
    runner.assert_drained();
}

#[test]
fn cleanup_refuses_an_owned_endpoint_with_changed_generation() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.push(module_command(73, SOURCE, SINK));
    let endpoint = serde_json::json!([{
        "index": 81,
        "name": AEC_SOURCE,
        "owner_module": 73,
        "properties": {
            "translator.owner": "true",
            "translator.generation": "foreign-generation"
        }
    }]);
    expected.push(ExpectedCommand {
        args: args(&["--format=json", "list", "sources"]),
        result: success(endpoint.to_string()),
    });
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();

    let error = graph.cleanup_owned().unwrap_err();

    assert_eq!(error.code(), AecErrorCode::CleanupRefused);
    runner.assert_drained();
}

fn metadata() -> AecDeviceMetadata {
    AecDeviceMetadata {
        source_name: SOURCE.to_owned(),
        sink_name: SINK.to_owned(),
        source_geometry: "desk-left-45cm".to_owned(),
        sink_geometry: "desk-front-80cm".to_owned(),
        sink_port: "analog-output-speaker".to_owned(),
        sink_volume_percent: 40,
    }
}

#[test]
fn validation_accepts_finite_median_erle_at_threshold_and_zero_far_end_triggers() {
    let record = evaluate_aec(AecValidationInput {
        metadata: metadata(),
        windows: vec![
            AecPowerWindow::new(0, 1.0, 0.01, 0.0),
            AecPowerWindow::new(1, 1.0, 10_f64.powf(-1.5), 0.0),
            AecPowerWindow::new(2, 1.0, 0.1, 0.0),
        ],
        far_end: AecFarEndCounters {
            vad_triggers: 0,
            provider_requests: 0,
        },
    })
    .unwrap();

    assert!((record.median_erle_db - 15.0).abs() < 1e-10);
    assert!(record.erle_passed);
    assert!(record.far_end_passed);
    assert!(record.validated);
    assert_eq!(record.metadata.sink_volume_percent, 40);
}

#[test]
fn validation_keeps_erle_finite_at_noise_floor_and_rejects_far_end_activity() {
    let record = evaluate_aec(AecValidationInput {
        metadata: metadata(),
        windows: vec![AecPowerWindow::new(0, 0.0, 0.0, 0.0)],
        far_end: AecFarEndCounters {
            vad_triggers: 1,
            provider_requests: 0,
        },
    })
    .unwrap();

    assert!(record.median_erle_db.is_finite());
    assert!(!record.far_end_passed);
    assert!(!record.validated);
}

#[test]
fn validation_rejects_non_finite_or_misaligned_power_windows() {
    let non_finite = evaluate_aec(AecValidationInput {
        metadata: metadata(),
        windows: vec![AecPowerWindow::new(0, f64::NAN, 0.1, 0.0)],
        far_end: AecFarEndCounters::default(),
    })
    .unwrap_err();
    assert_eq!(non_finite.code(), AecErrorCode::InvalidValidationInput);

    let duplicate_sequence = evaluate_aec(AecValidationInput {
        metadata: metadata(),
        windows: vec![
            AecPowerWindow::new(4, 1.0, 0.01, 0.0),
            AecPowerWindow::new(4, 1.0, 0.01, 0.0),
        ],
        far_end: AecFarEndCounters::default(),
    })
    .unwrap_err();
    assert_eq!(
        duplicate_sequence.code(),
        AecErrorCode::InvalidValidationInput
    );
}

#[test]
#[ignore = "requires an explicit physical source/sink pair on PipeWire-Pulse"]
fn workstation_aec_graph_is_owned_and_tears_down() {
    let source = std::env::var("TRANSLATOR_AEC_TEST_SOURCE").unwrap();
    let sink = std::env::var("TRANSLATOR_AEC_TEST_SINK").unwrap();
    let generation = format!("task7-workstation-{}", uuid::Uuid::new_v4());
    let mut graph = PulseAecGraph::new(
        SystemCommandRunner,
        AecPhysicalPair::new(source, sink),
        generation,
    )
    .unwrap();

    let result = graph.load_owned().and_then(|loaded| {
        let inspected = graph.inspect_owned()?;
        assert_eq!(loaded, inspected);
        Ok(())
    });
    let cleanup = graph.cleanup_owned();

    result.unwrap();
    assert!(cleanup.unwrap().is_some());
    assert!(graph.inspect_owned().is_err());
}
