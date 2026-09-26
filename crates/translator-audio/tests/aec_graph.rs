use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use translator_audio::{
    AEC_SINK, AEC_SOURCE, AecErrorCode, AecPhysicalPair, CommandResult, CommandRunError,
    CommandRunner, PulseAecGraph, SystemCommandRunner,
};

const GENERATION: &str = "aec-generation-0001";
const SOURCE: &str = "alsa_input.usb-headset.mono-fallback";
const SINK: &str = "alsa_output.pci-speakers.analog-stereo";

#[derive(Clone)]
struct FakeRunner {
    expected: Arc<Mutex<VecDeque<ExpectedCommand>>>,
    deadlines: Arc<Mutex<Vec<Instant>>>,
}

struct ExpectedCommand {
    args: Vec<String>,
    result: Result<CommandResult, CommandRunError>,
}

impl FakeRunner {
    fn new(expected: Vec<ExpectedCommand>) -> Self {
        Self {
            expected: Arc::new(Mutex::new(expected.into())),
            deadlines: Arc::default(),
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
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        self.deadlines.lock().unwrap().push(deadline);
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
         translator.owner=true translator.generation={GENERATION}'\t\n"
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

fn absent_inspection() -> Vec<ExpectedCommand> {
    vec![
        ExpectedCommand {
            args: args(&["list", "short", "modules"]),
            result: success("99\tmodule-null-sink\tsink_name=foreign\t\n"),
        },
        ExpectedCommand {
            args: args(&["--format=json", "list", "sources"]),
            result: success("[]"),
        },
        ExpectedCommand {
            args: args(&["--format=json", "list", "sinks"]),
            result: success("[]"),
        },
    ]
}

fn unload_command() -> ExpectedCommand {
    ExpectedCommand {
        args: args(&["unload-module", "73"]),
        result: success(""),
    }
}

#[test]
fn one_absolute_deadline_covers_load_inspection_cleanup_and_absence() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.extend(ready_inspection(73));
    expected.extend(ready_inspection(73));
    expected.push(unload_command());
    expected.extend(absent_inspection());
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    let deadline = Instant::now() + Duration::from_secs(2);
    graph.load_owned_until(deadline).unwrap();
    graph.inspect_owned_until(deadline).unwrap();
    assert_eq!(graph.cleanup_owned_until(deadline).unwrap(), Some(73));
    assert_eq!(*runner.deadlines.lock().unwrap(), vec![deadline; 14]);
    runner.assert_drained();
}

#[test]
fn expired_deadline_never_starts_an_effect_or_renews_budget() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    let expired = Instant::now();
    assert!(graph.load_owned_until(expired).is_err());
    assert!(graph.recover_owned_until(expired).is_err());
    graph.load_owned().unwrap();
    assert!(graph.inspect_owned_until(expired).is_err());
    assert!(graph.cleanup_owned_until(expired).is_err());
    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::AlreadyOwned
    );
    runner.assert_drained();
}

#[test]
fn restart_recovers_exact_generation_and_preserves_foreign_module() {
    let mut expected = ready_inspection(73);
    let mut modules = expected[0].result.as_ref().unwrap().stdout().to_vec();
    modules.extend_from_slice(b"99\tmodule-null-sink\tsink_name=foreign\t\n");
    expected[0].result = success(modules);
    expected.extend(ready_inspection(73));
    expected.push(unload_command());
    expected.extend(absent_inspection());
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    let deadline = Instant::now() + Duration::from_secs(2);
    let recovered = graph.recover_owned_until(deadline).unwrap();
    assert_eq!(recovered.module_id, 73);
    assert_eq!(recovered.generation, GENERATION);
    assert_eq!(graph.cleanup_owned_until(deadline).unwrap(), Some(73));
    runner.assert_drained();
}

#[test]
fn restart_zero_and_ambiguous_matches_block_new_load_and_allow_later_recovery() {
    let mut ambiguous = module_command(73, SOURCE, SINK);
    let mut payload = ambiguous.result.unwrap().stdout().to_vec();
    payload.extend_from_slice(module_command(74, SOURCE, SINK).result.unwrap().stdout());
    ambiguous.result = success(payload);
    for (command, error) in [
        (absent_inspection().remove(0), AecErrorCode::NotOwned),
        (ambiguous, AecErrorCode::OwnershipMismatch),
        (
            module_command(73, SOURCE, "foreign"),
            AecErrorCode::NotOwned,
        ),
    ] {
        let mut expected = vec![command];
        expected.extend(ready_inspection(73));
        let runner = FakeRunner::new(expected);
        let mut graph = graph(runner.clone());
        let deadline = Instant::now() + Duration::from_secs(2);
        assert_eq!(
            graph.recover_owned_until(deadline).unwrap_err().code(),
            error
        );
        assert_eq!(
            graph.load_owned().unwrap_err().code(),
            AecErrorCode::AlreadyOwned
        );
        assert_eq!(graph.recover_owned_until(deadline).unwrap().module_id, 73);
        runner.assert_drained();
    }
}

#[test]
fn post_load_inspection_failure_retains_orphan_for_cleanup() {
    let mut expected = vec![
        load_command(),
        ExpectedCommand {
            args: args(&["list", "short", "modules"]),
            result: Err(CommandRunError::TimedOut),
        },
        module_command(73, SOURCE, SINK),
    ];
    expected.extend(absent_inspection().into_iter().skip(1));
    expected.push(unload_command());
    expected.extend(absent_inspection());
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::InspectionFailed
    );
    assert_eq!(graph.cleanup_owned().unwrap(), Some(73));
    runner.assert_drained();
}

#[test]
fn ambiguous_unknown_load_never_authorizes_unload() {
    let mut load = load_command();
    load.result = success("bad-id");
    let mut module = module_command(73, SOURCE, SINK);
    let mut payload = module.result.unwrap().stdout().to_vec();
    payload.extend_from_slice(module_command(74, SOURCE, SINK).result.unwrap().stdout());
    module.result = success(payload);
    let runner = FakeRunner::new(vec![load, module]);
    let mut graph = graph(runner.clone());
    assert!(graph.load_owned().is_err());
    assert_eq!(
        graph.cleanup_owned().unwrap_err().code(),
        AecErrorCode::CleanupRefused
    );
    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::AlreadyOwned
    );
    runner.assert_drained();
}

#[test]
fn terminal_unknown_load_with_confirmed_empty_inventory_releases_custody() {
    let mut load = load_command();
    load.result = success("bad-id");
    let mut expected = vec![load];
    expected.extend(absent_inspection());
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());

    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::ModuleLoadFailed
    );
    assert_eq!(graph.cleanup_owned().unwrap(), None);
    assert_eq!(graph.cleanup_owned().unwrap(), None);
    runner.assert_drained();
}

#[test]
fn c9_s3_unknown_id_noncanonical_same_generation_orphan_is_not_absence() {
    let mut load = load_command();
    load.result = success("unknown-id");
    let mut module = module_command(73, SOURCE, SINK);
    let payload = String::from_utf8(module.result.unwrap().stdout().to_vec()).unwrap();
    module.result = success(payload.replace("rate=48000", "rate=44100"));
    let mut expected = vec![load, module];
    expected.extend(absent_inspection().into_iter().skip(1));
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    assert!(graph.load_owned().is_err());
    assert!(
        graph.cleanup_owned().is_err(),
        "same-generation module remains even when its contract and endpoints are broken"
    );
    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::AlreadyOwned
    );
    // An implementation may reject as soon as the module inventory proves ambiguity.
    assert!(runner.expected.lock().unwrap().iter().all(|command| {
        command
            .args
            .first()
            .is_some_and(|arg| arg == "--format=json")
    }));
}

#[test]
fn failed_endpoint_inspection_pins_recovered_id_before_retry() {
    let mut load = load_command();
    load.result = success("bad-id");
    let runner = FakeRunner::new(vec![
        load,
        module_command(73, SOURCE, SINK),
        ExpectedCommand {
            args: args(&["--format=json", "list", "sources"]),
            result: Err(CommandRunError::TimedOut),
        },
        module_command(74, SOURCE, SINK),
    ]);
    let mut graph = graph(runner.clone());
    assert!(graph.load_owned().is_err());
    assert!(graph.cleanup_owned().is_err());
    assert_eq!(
        graph.cleanup_owned().unwrap_err().code(),
        AecErrorCode::CleanupRefused
    );
    runner.assert_drained();
}

#[test]
fn unload_timeout_retries_absence_without_repeating_effect() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.extend(ready_inspection(73));
    let mut unload = unload_command();
    unload.result = Err(CommandRunError::TimedOut);
    expected.push(unload);
    expected.extend(absent_inspection());
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();
    assert_eq!(
        graph.cleanup_owned().unwrap_err().code(),
        AecErrorCode::CleanupFailed
    );
    assert!(graph.inspect_owned().is_err());
    assert_eq!(graph.cleanup_owned().unwrap(), Some(73));
    runner.assert_drained();
}

#[test]
fn lingering_endpoint_after_unload_keeps_owner_until_confirmed_absent() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.extend(ready_inspection(73));
    expected.push(unload_command());
    expected.push(absent_inspection().remove(0));
    expected.push(endpoint_command("sources", AEC_SOURCE, 73));
    expected.extend(absent_inspection());
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();
    assert_eq!(
        graph.cleanup_owned().unwrap_err().code(),
        AecErrorCode::CleanupFailed
    );
    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::AlreadyOwned
    );
    assert_eq!(graph.cleanup_owned().unwrap(), Some(73));
    runner.assert_drained();
}

#[test]
fn restart_rejects_duplicate_contract_argument() {
    let mut module = module_command(73, SOURCE, SINK);
    let payload = String::from_utf8(module.result.unwrap().stdout().to_vec()).unwrap();
    module.result = success(payload.replace("rate=48000", "rate=44100 rate=48000"));
    let runner = FakeRunner::new(vec![module]);
    let mut graph = graph(runner.clone());
    assert_eq!(
        graph
            .recover_owned_until(Instant::now() + Duration::from_secs(2))
            .unwrap_err()
            .code(),
        AecErrorCode::NotOwned
    );
    runner.assert_drained();
}

#[derive(Clone)]
struct LateRunner {
    inner: FakeRunner,
    late_command: Vec<String>,
}

impl CommandRunner for LateRunner {
    fn run_until(
        &self,
        program: &str,
        arguments: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        let result = self.inner.run_until(program, arguments, deadline);
        if arguments == self.late_command {
            std::thread::sleep(
                deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
            );
        }
        result
    }
}

#[test]
fn successful_but_late_load_does_not_start_inspection_and_retains_recovery() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.push(unload_command());
    expected.extend(absent_inspection());
    let inner = FakeRunner::new(expected);
    let runner = LateRunner {
        inner: inner.clone(),
        late_command: load_command().args,
    };
    let mut graph =
        PulseAecGraph::new(runner, AecPhysicalPair::new(SOURCE, SINK), GENERATION).unwrap();
    assert_eq!(
        graph
            .load_owned_until(Instant::now() + Duration::from_millis(20))
            .unwrap_err()
            .code(),
        AecErrorCode::ModuleLoadFailed
    );
    assert_eq!(inner.deadlines.lock().unwrap().len(), 1);
    assert_eq!(graph.cleanup_owned().unwrap(), Some(73));
    inner.assert_drained();
}

#[test]
fn expiry_during_cleanup_inspection_never_starts_unload() {
    let expected = vec![
        module_command(73, SOURCE, SINK),
        endpoint_command("sources", AEC_SOURCE, 73),
        module_command(73, SOURCE, SINK),
        endpoint_command("sources", AEC_SOURCE, 73),
    ];
    let inner = FakeRunner::new(expected);
    let mut graph = PulseAecGraph::new(
        LateRunner {
            inner: inner.clone(),
            late_command: args(&["--format=json", "list", "sources"]),
        },
        AecPhysicalPair::new(SOURCE, SINK),
        GENERATION,
    )
    .unwrap();
    // Recovery pins the module ID before endpoint inspection times out.
    assert!(
        graph
            .recover_owned_until(Instant::now() + Duration::from_millis(20))
            .is_err()
    );
    assert!(
        graph
            .cleanup_owned_until(Instant::now() + Duration::from_millis(20))
            .is_err()
    );
    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::AlreadyOwned
    );
    inner.assert_drained();
}

#[test]
fn reused_id_after_unload_blocks_retry_and_is_never_unloaded() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.extend(ready_inspection(73));
    expected.push(unload_command());
    expected.push(module_command(73, SOURCE, "foreign"));
    expected.push(module_command(73, SOURCE, "foreign"));
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();
    assert_eq!(
        graph.cleanup_owned().unwrap_err().code(),
        AecErrorCode::CleanupFailed
    );
    assert_eq!(
        graph.cleanup_owned().unwrap_err().code(),
        AecErrorCode::CleanupRefused
    );
    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::AlreadyOwned
    );
    runner.assert_drained();
}

#[test]
fn restart_rejects_ambiguous_endpoints_and_keeps_orphan_cleanup_obligation() {
    let mut expected = vec![module_command(73, SOURCE, SINK)];
    let mut source = endpoint_command("sources", AEC_SOURCE, 73);
    let payload: serde_json::Value =
        serde_json::from_slice(source.result.unwrap().stdout()).unwrap();
    source.result = success(serde_json::json!([payload[0], payload[0]]).to_string());
    expected.push(source);
    expected.push(module_command(73, SOURCE, SINK));
    expected.extend(absent_inspection().into_iter().skip(1));
    expected.push(unload_command());
    expected.extend(absent_inspection());
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    assert_eq!(
        graph
            .recover_owned_until(Instant::now() + Duration::from_secs(2))
            .unwrap_err()
            .code(),
        AecErrorCode::OwnershipMismatch
    );
    assert_eq!(graph.cleanup_owned().unwrap(), Some(73));
    runner.assert_drained();
}

#[test]
fn uncertain_load_recovers_exact_module_before_cleanup() {
    for result in [Err(CommandRunError::TimedOut), success("not-an-id")] {
        let mut load = load_command();
        load.result = result;
        let mut expected = vec![load];
        expected.extend(ready_inspection(73));
        expected.push(ExpectedCommand {
            args: args(&["unload-module", "73"]),
            result: success(""),
        });
        expected.extend(absent_inspection());
        let runner = FakeRunner::new(expected);
        let mut graph = graph(runner.clone());
        assert_eq!(
            graph.load_owned().unwrap_err().code(),
            AecErrorCode::ModuleLoadFailed
        );
        assert_eq!(
            graph.load_owned().unwrap_err().code(),
            AecErrorCode::AlreadyOwned
        );
        assert_eq!(graph.cleanup_owned().unwrap(), Some(73));
        assert_eq!(graph.cleanup_owned().unwrap(), None);
        runner.assert_drained();
    }
}

#[test]
fn cleanup_requires_confirmed_absence_and_can_retry_confirmation() {
    let mut expected = vec![load_command()];
    expected.extend(ready_inspection(73));
    expected.extend(ready_inspection(73));
    expected.push(ExpectedCommand {
        args: args(&["unload-module", "73"]),
        result: success(""),
    });
    expected.push(ExpectedCommand {
        args: args(&["list", "short", "modules"]),
        result: Err(CommandRunError::TimedOut),
    });
    expected.extend(absent_inspection());
    let runner = FakeRunner::new(expected);
    let mut graph = graph(runner.clone());
    graph.load_owned().unwrap();
    assert!(graph.cleanup_owned().is_err());
    assert_eq!(
        graph.load_owned().unwrap_err().code(),
        AecErrorCode::AlreadyOwned
    );
    assert_eq!(graph.cleanup_owned().unwrap(), Some(73));
    runner.assert_drained();
}

#[derive(Clone)]
struct DuplicateInventoryRunner {
    duplicate: Arc<AtomicBool>,
    unloaded: Arc<AtomicBool>,
    calls: Arc<Mutex<Vec<Vec<String>>>>,
}

impl CommandRunner for DuplicateInventoryRunner {
    fn run_until(
        &self,
        program: &str,
        arguments: &[String],
        _: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        self.calls.lock().unwrap().push(arguments.to_vec());
        let command = match arguments
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["load-module", ..] => load_command(),
            ["list", "short", "modules"] => {
                if self.unloaded.load(Ordering::SeqCst) {
                    return success("99\tmodule-null-sink\tsink_name=foreign\t\n");
                }
                let command = module_command(73, SOURCE, SINK);
                let mut bytes = if self.duplicate.load(Ordering::SeqCst) {
                    b"73\tmodule-null-sink\tsink_name=foreign\t\n".to_vec()
                } else {
                    Vec::new()
                };
                bytes.extend_from_slice(command.result.unwrap().stdout());
                bytes.extend_from_slice(b"99\tmodule-null-sink\tsink_name=foreign\t\n");
                return success(bytes);
            }
            ["--format=json", "list", _] if self.unloaded.load(Ordering::SeqCst) => {
                return success("[]");
            }
            ["--format=json", "list", "sources"] => endpoint_command("sources", AEC_SOURCE, 73),
            ["--format=json", "list", "sinks"] => endpoint_command("sinks", AEC_SINK, 73),
            ["unload-module", "73"] => {
                self.unloaded.store(true, Ordering::SeqCst);
                return success("");
            }
            _ => panic!("unexpected command"),
        };
        assert_eq!(arguments, command.args);
        command.result
    }
}

#[test]
fn duplicate_module_ids_refuse_aec_cleanup_and_retain_owner_for_recovery() {
    let runner = DuplicateInventoryRunner {
        duplicate: Arc::new(AtomicBool::new(false)),
        unloaded: Arc::new(AtomicBool::new(false)),
        calls: Arc::default(),
    };
    let mut graph = PulseAecGraph::new(
        runner.clone(),
        AecPhysicalPair::new(SOURCE, SINK),
        GENERATION,
    )
    .unwrap();
    graph.load_owned().unwrap();
    runner.calls.lock().unwrap().clear();
    runner.duplicate.store(true, Ordering::SeqCst);
    let rejected = graph.cleanup_owned();
    let failed_calls = std::mem::take(&mut *runner.calls.lock().unwrap());
    runner.duplicate.store(false, Ordering::SeqCst);
    let recovered = graph.cleanup_owned();
    let again = graph.cleanup_owned();
    let recovery_calls = std::mem::take(&mut *runner.calls.lock().unwrap());
    assert!(
        rejected.is_err(),
        "duplicate ID authorized effects: {failed_calls:?}"
    );
    assert_eq!(
        rejected.unwrap_err().code(),
        AecErrorCode::CleanupRefused,
        "duplicate ID must never authorize unload: {failed_calls:?}"
    );
    assert_eq!(failed_calls, [args(&["list", "short", "modules"])]);
    assert_eq!(recovered.unwrap(), Some(73));
    assert_eq!(again.unwrap(), None);
    assert_eq!(
        recovery_calls,
        [
            args(&["list", "short", "modules"]),
            args(&["--format=json", "list", "sources"]),
            args(&["--format=json", "list", "sinks"]),
            args(&["unload-module", "73"]),
            args(&["list", "short", "modules"]),
            args(&["--format=json", "list", "sources"]),
            args(&["--format=json", "list", "sinks"]),
        ]
    );
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
    expected.extend(absent_inspection());
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
