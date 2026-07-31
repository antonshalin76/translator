use std::collections::VecDeque;
use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tempfile::tempdir;
use translator_audio::{
    AudioGraph, AudioGraphError, AudioGraphErrorCode, CommandResult, CommandRunError,
    CommandRunner, EndpointKind, EndpointRole, GraphHealth, MIC_OUT_SINK, PulseAudioGraph,
    REMOTE_IN_SINK, VIRTUAL_MIC_SOURCE,
};

const GENERATION: &str = "test-generation-0001";

#[derive(Clone)]
struct FakeRunner {
    expected: Arc<Mutex<VecDeque<ExpectedCommand>>>,
}

struct ExpectedCommand {
    args: Vec<String>,
    result: Result<CommandResult, CommandRunError>,
}

struct BlockingRunner {
    started: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}

impl CommandRunner for BlockingRunner {
    fn run(&self, _program: &str, _args: &[String]) -> Result<CommandResult, CommandRunError> {
        self.started.send(()).unwrap();
        self.release.recv().unwrap();
        Err(CommandRunError::NotFound)
    }
}

struct ProbeRunner {
    called: mpsc::Sender<()>,
}

impl CommandRunner for ProbeRunner {
    fn run(&self, _program: &str, _args: &[String]) -> Result<CommandResult, CommandRunError> {
        self.called.send(()).unwrap();
        Err(CommandRunError::NotFound)
    }
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

#[derive(Clone)]
struct CrashSafeRunner {
    journal: PathBuf,
    call: Arc<Mutex<usize>>,
}

impl CrashSafeRunner {
    fn new(journal: PathBuf) -> Self {
        Self {
            journal,
            call: Arc::new(Mutex::new(0)),
        }
    }

    fn assert_drained(&self) {
        assert_eq!(*self.call.lock().unwrap(), 7);
    }
}

impl CommandRunner for CrashSafeRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        let mut call = self.call.lock().unwrap();
        let result = match *call {
            0 => {
                assert_eq!(args, &args_from(&["--format=json", "list", "sinks"]));
                success("[]")
            }
            1 => {
                assert_eq!(args, &args_from(&["--format=json", "list", "sources"]));
                success("[]")
            }
            2 => {
                assert!(self.journal.exists());
                assert!(journal_ids(&self.journal).is_empty());
                assert_eq!(args, &load_mic_out(101).args);
                success("101")
            }
            3 => {
                assert_eq!(journal_ids(&self.journal), [101]);
                assert_eq!(args, &load_virtual_mic(success("102")).args);
                success("102")
            }
            4 => {
                assert_eq!(journal_ids(&self.journal), [101, 102]);
                assert_eq!(args, &load_remote_in(103).args);
                success("103")
            }
            5 => {
                assert_eq!(journal_ids(&self.journal), [101, 102, 103]);
                assert_eq!(args, &args_from(&["--format=json", "list", "sinks"]));
                success(&ready_sinks())
            }
            6 => {
                assert_eq!(args, &args_from(&["--format=json", "list", "sources"]));
                success(&ready_sources())
            }
            _ => panic!("unexpected pactl command"),
        };
        *call += 1;
        result
    }
}

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn args_from(values: &[&str]) -> Vec<String> {
    args(values)
}

fn success(stdout: &str) -> Result<CommandResult, CommandRunError> {
    Ok(CommandResult::success(stdout.as_bytes().to_vec()))
}

fn failure(stdout: &str, stderr: &str) -> Result<CommandResult, CommandRunError> {
    Ok(CommandResult::failure(
        stdout.as_bytes().to_vec(),
        stderr.as_bytes().to_vec(),
    ))
}

fn list_sinks(json: &str) -> ExpectedCommand {
    ExpectedCommand {
        args: args(&["--format=json", "list", "sinks"]),
        result: success(json),
    }
}

fn list_sources(json: &str) -> ExpectedCommand {
    ExpectedCommand {
        args: args(&["--format=json", "list", "sources"]),
        result: success(json),
    }
}

fn list_modules(entries: &[(u32, &str, &str)]) -> ExpectedCommand {
    let stdout = entries
        .iter()
        .map(|(id, module, argument)| {
            let argument = complete_module_contract(argument);
            format!("{id}\t{module}\t{argument} translator.generation={GENERATION}\t0")
        })
        .collect::<Vec<_>>()
        .join("\n");
    ExpectedCommand {
        args: args(&["list", "short", "modules"]),
        result: success(&stdout),
    }
}

fn complete_module_contract(argument: &str) -> String {
    let mut result = argument.to_owned();
    let has_token = |expected: &str| argument.split_whitespace().any(|token| token == expected);
    let required: &[&str] = if has_token("sink_name=translator_mic_out") {
        &[
            "rate=48000",
            "channels=1",
            "channel_map=mono",
            "sink_properties=device.description=Translator_Mic_Out",
        ]
    } else if has_token("source_name=translator_virtual_mic") {
        &[
            "master=translator_mic_out.monitor",
            "channels=1",
            "channel_map=mono",
            "remix=no",
            "source_properties=device.description=Translator_Virtual_Mic",
        ]
    } else if has_token("sink_name=translator_remote_in") {
        &[
            "rate=48000",
            "channels=2",
            "channel_map=front-left,front-right",
            "sink_properties=device.description=Translator_Remote_In",
        ]
    } else {
        &[]
    };
    for required_argument in required {
        let key = required_argument.split_once('=').unwrap().0;
        if !argument
            .split_whitespace()
            .any(|token| token.starts_with(&format!("{key}=")))
        {
            result.push(' ');
            result.push_str(required_argument);
        }
    }
    result
}

fn owned_modules(ids: [u32; 3]) -> ExpectedCommand {
    list_modules(&[
        (
            ids[0],
            "module-null-sink",
            "sink_name=translator_mic_out translator.owner=true",
        ),
        (
            ids[1],
            "module-remap-source",
            "source_name=translator_virtual_mic translator.owner=true",
        ),
        (
            ids[2],
            "module-null-sink",
            "sink_name=translator_remote_in translator.owner=true",
        ),
    ])
}

fn load_mic_out(module_id: u32) -> ExpectedCommand {
    let mut command_args = args(&[
        "load-module",
        "module-null-sink",
        "sink_name=translator_mic_out",
        "rate=48000",
        "channels=1",
        "channel_map=mono",
    ]);
    command_args.push(format!(
        "sink_properties=device.description=Translator_Mic_Out translator.owner=true translator.generation={GENERATION}"
    ));
    ExpectedCommand {
        args: command_args,
        result: success(&module_id.to_string()),
    }
}

fn load_virtual_mic(result: Result<CommandResult, CommandRunError>) -> ExpectedCommand {
    let mut command_args = args(&[
        "load-module",
        "module-remap-source",
        "master=translator_mic_out.monitor",
        "source_name=translator_virtual_mic",
        "channels=1",
        "channel_map=mono",
        "remix=no",
    ]);
    command_args.push(format!(
        "source_properties=device.description=Translator_Virtual_Mic translator.owner=true translator.generation={GENERATION}"
    ));
    ExpectedCommand {
        args: command_args,
        result,
    }
}

fn load_remote_in(module_id: u32) -> ExpectedCommand {
    let mut command_args = args(&[
        "load-module",
        "module-null-sink",
        "sink_name=translator_remote_in",
        "rate=48000",
        "channels=2",
        "channel_map=front-left,front-right",
    ]);
    command_args.push(format!(
        "sink_properties=device.description=Translator_Remote_In translator.owner=true translator.generation={GENERATION}"
    ));
    ExpectedCommand {
        args: command_args,
        result: success(&module_id.to_string()),
    }
}

fn unload(module_id: u32) -> ExpectedCommand {
    ExpectedCommand {
        args: args(&["unload-module", &module_id.to_string()]),
        result: success(""),
    }
}

fn unload_with(module_id: u32, result: Result<CommandResult, CommandRunError>) -> ExpectedCommand {
    ExpectedCommand {
        args: args(&["unload-module", &module_id.to_string()]),
        result,
    }
}

fn ready_sinks_with(mic_module: u32, remote_module: u32) -> String {
    serde_json::json!([
        {
            "index": 401,
            "name": "translator_mic_out",
            "owner_module": mic_module,
            "properties": {
                "translator.owner": "true",
                "device.description": "Translator_Mic_Out"
            }
        },
        {
            "index": 402,
            "name": "translator_remote_in",
            "owner_module": remote_module,
            "properties": {
                "translator.owner": "true",
                "device.description": "Translator_Remote_In"
            }
        }
    ])
    .to_string()
}

fn ready_sources_with(virtual_module: u32) -> String {
    serde_json::json!([{
        "index": 501,
        "name": "translator_virtual_mic",
        "owner_module": virtual_module,
        "properties": {
            "translator.owner": "true",
            "device.description": "Translator_Virtual_Mic"
        }
    }])
    .to_string()
}

fn ready_sinks() -> String {
    ready_sinks_with(101, 103)
}

fn ready_sources() -> String {
    ready_sources_with(102)
}

fn write_journal(path: &Path, module_ids: [u32; 3]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::set_permissions(path.parent().unwrap(), fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(
        path,
        serde_json::json!({
            "schema_version": 1,
            "generation": GENERATION,
            "modules": [
                {"role": "mic_out_sink", "module_id": module_ids[0]},
                {"role": "virtual_mic_source", "module_id": module_ids[1]},
                {"role": "remote_in_sink", "module_id": module_ids[2]}
            ]
        })
        .to_string(),
    )
    .unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

fn write_intent(path: &Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        path,
        serde_json::json!({
            "schema_version": 1,
            "generation": GENERATION,
            "modules": []
        })
        .to_string(),
    )
    .unwrap();
    fs::set_permissions(path.parent().unwrap(), fs::Permissions::from_mode(0o700)).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

fn journal_ids(path: &Path) -> Vec<u64> {
    serde_json::from_slice::<serde_json::Value>(&fs::read(path).unwrap()).unwrap()["modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|module| module["module_id"].as_u64().unwrap())
        .collect()
}

fn test_graph<R: CommandRunner>(runner: R, journal: PathBuf) -> PulseAudioGraph<R> {
    PulseAudioGraph::new_with_generation(runner, journal, GENERATION.to_owned())
}

fn assert_error_redacted(error: &AudioGraphError) {
    let representations = [
        format!("{error:?}"),
        error.to_string(),
        error.safe_message().to_owned(),
        serde_json::to_string(error.safe_status()).unwrap(),
    ];
    for representation in representations {
        assert!(!representation.contains("private-spoken-marker"));
        assert!(!representation.contains("load-marker"));
        assert!(!representation.contains("unload-marker"));
    }
}

fn create_expectations() -> Vec<ExpectedCommand> {
    vec![
        list_sinks("[]"),
        list_sources("[]"),
        load_mic_out(101),
        load_virtual_mic(success("102")),
        load_remote_in(103),
        list_sinks(&ready_sinks()),
        list_sources(&ready_sources()),
    ]
}

#[test]
fn ensure_creates_all_endpoints_and_second_run_is_idempotent() {
    let temp = tempdir().unwrap();
    let mut expected = create_expectations();
    expected.extend([
        list_sinks(&ready_sinks()),
        list_sources(&ready_sources()),
        owned_modules([101, 102, 103]),
    ]);
    let runner = FakeRunner::new(expected);
    let journal = temp.path().join("translator/modules.json");
    let mut graph = test_graph(runner.clone(), journal.clone());

    let created = graph.ensure_endpoints().expect("first ensure must create");
    drop(graph);
    let mut restarted = test_graph(runner.clone(), journal.clone());
    let existing = restarted
        .ensure_endpoints()
        .expect("restart ensure must adopt journaled modules");

    assert_eq!(created.health, GraphHealth::Ready);
    assert_eq!(existing.health, GraphHealth::Ready);
    assert_eq!(existing, created);
    assert_eq!(created.owned_module_ids, vec![101, 102, 103]);
    let names: Vec<_> = created
        .endpoints
        .iter()
        .map(|endpoint| endpoint.name.as_str())
        .collect();
    assert_eq!(names, [MIC_OUT_SINK, VIRTUAL_MIC_SOURCE, REMOTE_IN_SINK]);
    assert_eq!(
        created
            .endpoints
            .iter()
            .map(|endpoint| (endpoint.role, endpoint.kind))
            .collect::<Vec<_>>(),
        [
            (EndpointRole::MicOutSink, EndpointKind::Sink),
            (EndpointRole::VirtualMicSource, EndpointKind::Source),
            (EndpointRole::RemoteInSink, EndpointKind::Sink),
        ]
    );
    let journal_json: serde_json::Value =
        serde_json::from_slice(&fs::read(&journal).unwrap()).unwrap();
    assert_eq!(
        journal_json,
        serde_json::json!({
            "schema_version": 1,
            "generation": GENERATION,
            "modules": [
                {"role": "mic_out_sink", "module_id": 101},
                {"role": "virtual_mic_source", "module_id": 102},
                {"role": "remote_in_sink", "module_id": 103}
            ]
        })
    );
    assert_eq!(
        fs::metadata(journal.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(&journal).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        fs::read_dir(journal.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .count(),
        2,
        "journal directory must contain only the journal and process lock"
    );
    assert_eq!(
        fs::metadata(journal.parent().unwrap().join(".modules.json.lock"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    runner.assert_drained();
}

#[test]
fn ownership_is_journaled_after_each_successful_load() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    let runner = CrashSafeRunner::new(journal.clone());
    let mut graph = test_graph(runner.clone(), journal);

    graph.ensure_endpoints().expect("creation must succeed");

    runner.assert_drained();
}

#[test]
fn crash_orphan_is_adopted_from_generation_intent_and_reconciled() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_intent(&journal);
    let orphan = list_modules(&[(
        101,
        "module-null-sink",
        "sink_name=translator_mic_out translator.owner=true",
    )]);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        orphan,
        list_modules(&[(
            101,
            "module-null-sink",
            "sink_name=translator_mic_out translator.owner=true",
        )]),
        unload(101),
        list_sinks("[]"),
        list_sources("[]"),
        load_mic_out(201),
        load_virtual_mic(success("202")),
        load_remote_in(203),
        list_sinks(&ready_sinks_with(201, 203)),
        list_sources(&ready_sources_with(202)),
    ]);
    let mut graph = test_graph(runner.clone(), journal);

    let state = graph
        .ensure_endpoints()
        .expect("intent must prove and recover the orphan");

    assert_eq!(state.owned_module_ids, [201, 202, 203]);
    runner.assert_drained();
}

#[test]
fn large_pipewire_module_ids_are_preserved() {
    let temp = tempdir().unwrap();
    let ids = [536_870_913, 536_870_914, 536_870_915];
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        load_mic_out(ids[0]),
        load_virtual_mic(success(&ids[1].to_string())),
        load_remote_in(ids[2]),
        list_sinks(&ready_sinks_with(ids[0], ids[2])),
        list_sources(&ready_sources_with(ids[1])),
    ]);
    let mut graph = test_graph(runner.clone(), temp.path().join("translator/modules.json"));

    let state = graph.ensure_endpoints().expect("large ids must be valid");

    assert_eq!(state.owned_module_ids, ids);
    runner.assert_drained();
}

#[test]
fn foreign_duplicate_fails_without_unloading_any_module() {
    let temp = tempdir().unwrap();
    let runner = FakeRunner::new(vec![
        list_sinks(
            r#"[{"index":401,"name":"translator_mic_out","owner_module":999,
                 "properties":{"translator.owner":"true"}}]"#,
        ),
        list_sources("[]"),
    ]);
    let mut graph = test_graph(runner.clone(), temp.path().join("translator/modules.json"));

    let error = graph
        .ensure_endpoints()
        .expect_err("foreign duplicate must fail");

    assert_eq!(error.code(), AudioGraphErrorCode::DuplicateEndpoint);
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn duplicate_same_name_endpoints_fail_closed() {
    let temp = tempdir().unwrap();
    let runner = FakeRunner::new(vec![
        list_sinks(
            r#"[
              {"index":401,"name":"translator_mic_out","owner_module":900,"properties":{}},
              {"index":402,"name":"translator_mic_out","owner_module":901,"properties":{}}
            ]"#,
        ),
        list_sources("[]"),
    ]);
    let mut graph = test_graph(runner.clone(), temp.path().join("translator/modules.json"));

    let error = graph.ensure_endpoints().expect_err("duplicates must fail");

    assert_eq!(error.code(), AudioGraphErrorCode::DuplicateEndpoint);
    runner.assert_drained();
}

#[test]
fn malformed_journal_fails_without_touching_the_audio_graph() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    fs::create_dir_all(journal.parent().unwrap()).unwrap();
    fs::write(&journal, b"{private-spoken-marker").unwrap();
    let runner = FakeRunner::new(vec![]);
    let mut graph = test_graph(runner.clone(), journal);

    let error = graph.ensure_endpoints().expect_err("journal must be valid");

    assert_eq!(error.code(), AudioGraphErrorCode::OwnershipJournalInvalid);
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn malformed_graph_json_and_command_output_are_redacted() {
    let temp = tempdir().unwrap();
    let malformed_runner = FakeRunner::new(vec![list_sinks("{private-spoken-marker")]);
    let mut malformed = test_graph(
        malformed_runner.clone(),
        temp.path().join("malformed/modules.json"),
    );

    let malformed_error = malformed.ensure_endpoints().expect_err("JSON must parse");

    assert_eq!(
        malformed_error.code(),
        AudioGraphErrorCode::GraphInspectionFailed
    );
    assert_error_redacted(&malformed_error);
    malformed_runner.assert_drained();

    let command_runner = FakeRunner::new(vec![ExpectedCommand {
        args: args(&["--format=json", "list", "sinks"]),
        result: failure("", "private-spoken-marker"),
    }]);
    let mut command = test_graph(
        command_runner.clone(),
        temp.path().join("command/modules.json"),
    );

    let command_error = command
        .ensure_endpoints()
        .expect_err("non-zero inspect must fail");

    assert_eq!(
        command_error.code(),
        AudioGraphErrorCode::GraphInspectionFailed
    );
    assert_error_redacted(&command_error);
    command_runner.assert_drained();
}

#[test]
fn stale_missing_endpoint_is_reconciled_from_journal() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_journal(&journal, [101, 102, 103]);
    let runner = FakeRunner::new(vec![
        list_sinks(
            r#"[{"index":401,"name":"translator_mic_out","owner_module":101,"properties":{}}]"#,
        ),
        list_sources(
            r#"[{"index":501,"name":"translator_virtual_mic","owner_module":102,"properties":{}}]"#,
        ),
        owned_modules([101, 102, 103]),
        owned_modules([101, 102, 103]),
        unload(103),
        owned_modules([101, 102, 103]),
        unload(102),
        owned_modules([101, 102, 103]),
        unload(101),
        list_sinks("[]"),
        list_sources("[]"),
        load_mic_out(201),
        load_virtual_mic(success("202")),
        load_remote_in(203),
        list_sinks(&ready_sinks_with(201, 203)),
        list_sources(&ready_sources_with(202)),
    ]);
    let mut graph = test_graph(runner.clone(), journal);

    let state = graph.ensure_endpoints().expect("stale graph must recover");

    assert_eq!(state.owned_module_ids, [201, 202, 203]);
    runner.assert_drained();
}

#[test]
fn owner_mismatch_never_unloads_foreign_module() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_journal(&journal, [101, 102, 103]);
    let mixed_sinks = serde_json::json!([
        {
            "index": 401,
            "name": "translator_mic_out",
            "owner_module": 999,
            "properties": {}
        },
        {
            "index": 402,
            "name": "translator_remote_in",
            "owner_module": 103,
            "properties": {}
        }
    ])
    .to_string();
    let runner = FakeRunner::new(vec![
        list_sinks(&mixed_sinks),
        list_sources(&ready_sources()),
        list_modules(&[
            (
                999,
                "module-null-sink",
                "sink_name=translator_mic_out translator.owner=false",
            ),
            (
                102,
                "module-remap-source",
                "source_name=translator_virtual_mic translator.owner=true",
            ),
            (
                103,
                "module-null-sink",
                "sink_name=translator_remote_in translator.owner=true",
            ),
        ]),
    ]);
    let mut graph = test_graph(runner.clone(), journal);

    let error = graph
        .ensure_endpoints()
        .expect_err("foreign endpoint must remain foreign");

    assert_eq!(error.code(), AudioGraphErrorCode::DuplicateEndpoint);
    runner.assert_drained();
}

#[test]
fn third_load_failure_rolls_back_new_modules_in_reverse_order() {
    let temp = tempdir().unwrap();
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        load_mic_out(101),
        load_virtual_mic(success("102")),
        ExpectedCommand {
            args: load_remote_in(103).args,
            result: failure("", "private-spoken-marker"),
        },
        owned_modules([101, 102, 103]),
        unload(102),
        owned_modules([101, 102, 103]),
        unload(101),
    ]);
    let journal = temp.path().join("translator/modules.json");
    let mut graph = test_graph(runner.clone(), journal.clone());

    let error = graph
        .ensure_endpoints()
        .expect_err("partial load must be rolled back");

    assert_eq!(error.code(), AudioGraphErrorCode::ModuleLoadFailed);
    assert!(journal.exists());
    assert!(journal_ids(&journal).is_empty());
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn rollback_failure_persists_the_module_that_could_not_be_unloaded() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        load_mic_out(101),
        load_virtual_mic(failure("", "load-marker")),
        owned_modules([101, 102, 103]),
        unload_with(101, failure("", "unload-marker")),
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let error = graph
        .ensure_endpoints()
        .expect_err("rollback must fail safely");

    assert_eq!(error.code(), AudioGraphErrorCode::RollbackFailed);
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&journal).unwrap()).unwrap();
    assert_eq!(persisted["modules"].as_array().unwrap().len(), 1);
    assert_eq!(persisted["modules"][0]["module_id"], 101);
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn invalid_module_id_is_redacted_and_not_journaled() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        ExpectedCommand {
            args: load_mic_out(101).args,
            result: success("private-spoken-marker"),
        },
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let error = graph
        .ensure_endpoints()
        .expect_err("module id must be numeric");

    assert_eq!(error.code(), AudioGraphErrorCode::ModuleLoadFailed);
    assert!(journal.exists());
    assert!(journal_ids(&journal).is_empty());
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn missing_pactl_returns_a_safe_error() {
    let temp = tempdir().unwrap();
    let runner = FakeRunner::new(vec![ExpectedCommand {
        args: args(&["--format=json", "list", "sinks"]),
        result: Err(CommandRunError::NotFound),
    }]);
    let mut graph = test_graph(runner.clone(), temp.path().join("translator/modules.json"));

    let error = graph
        .ensure_endpoints()
        .expect_err("missing pactl must fail");

    assert_eq!(error.code(), AudioGraphErrorCode::PactlMissing);
    assert_eq!(error.safe_message(), "Audio control command is unavailable");
    runner.assert_drained();
}

#[test]
fn cleanup_unloads_only_journaled_modules_in_reverse_order() {
    let temp = tempdir().unwrap();
    let mut expected = create_expectations();
    expected.extend([
        list_sinks(&ready_sinks()),
        list_sources(&ready_sources()),
        owned_modules([101, 102, 103]),
        owned_modules([101, 102, 103]),
        unload(103),
        owned_modules([101, 102, 103]),
        unload(102),
        owned_modules([101, 102, 103]),
        unload(101),
    ]);
    let runner = FakeRunner::new(expected);
    let journal = temp.path().join("translator/modules.json");
    let mut graph = test_graph(runner.clone(), journal.clone());
    graph.ensure_endpoints().unwrap();

    let unloaded = graph.cleanup_owned().expect("owned cleanup must pass");

    assert_eq!(unloaded, vec![103, 102, 101]);
    assert!(!journal.exists());
    assert_eq!(graph.cleanup_owned().unwrap(), Vec::<u32>::new());
    runner.assert_drained();
}

#[test]
fn partial_cleanup_keeps_remaining_ids_for_retry() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_journal(&journal, [101, 102, 103]);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        owned_modules([101, 102, 103]),
        owned_modules([101, 102, 103]),
        unload(103),
        owned_modules([101, 102, 103]),
        unload_with(102, failure("", "private-spoken-marker")),
        list_sinks("[]"),
        list_sources("[]"),
        list_modules(&[
            (
                101,
                "module-null-sink",
                "sink_name=translator_mic_out translator.owner=true",
            ),
            (
                102,
                "module-remap-source",
                "source_name=translator_virtual_mic translator.owner=true",
            ),
        ]),
        list_modules(&[
            (
                101,
                "module-null-sink",
                "sink_name=translator_mic_out translator.owner=true",
            ),
            (
                102,
                "module-remap-source",
                "source_name=translator_virtual_mic translator.owner=true",
            ),
        ]),
        unload(102),
        list_modules(&[(
            101,
            "module-null-sink",
            "sink_name=translator_mic_out translator.owner=true",
        )]),
        unload(101),
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let error = graph
        .cleanup_owned()
        .expect_err("partial cleanup must fail");

    assert_eq!(error.code(), AudioGraphErrorCode::CleanupFailed);
    let remaining: serde_json::Value =
        serde_json::from_slice(&fs::read(&journal).unwrap()).unwrap();
    assert_eq!(
        remaining["modules"]
            .as_array()
            .unwrap()
            .iter()
            .map(|module| module["module_id"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [101, 102]
    );
    assert_error_redacted(&error);

    assert_eq!(graph.cleanup_owned().unwrap(), [102, 101]);
    assert!(!journal.exists());
    assert_eq!(graph.cleanup_owned().unwrap(), Vec::<u32>::new());
    runner.assert_drained();
}

#[test]
fn cleanup_refuses_reused_journal_module_ids() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_journal(&journal, [101, 102, 103]);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        list_modules(&[
            (
                101,
                "module-null-sink",
                "sink_name=foreign_sink translator.owner=true",
            ),
            (
                102,
                "module-remap-source",
                "source_name=translator_virtual_mic translator.owner=true",
            ),
        ]),
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let error = graph
        .cleanup_owned()
        .expect_err("reused ids must not be unloaded");

    assert_eq!(error.code(), AudioGraphErrorCode::CleanupFailed);
    assert_eq!(journal_ids(&journal), [101, 102, 103]);
    runner.assert_drained();
}

#[test]
fn cleanup_treats_missing_module_id_as_already_removed() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_journal(&journal, [101, 102, 103]);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        list_modules(&[
            (
                101,
                "module-null-sink",
                "sink_name=translator_mic_out translator.owner=true",
            ),
            (
                102,
                "module-remap-source",
                "source_name=translator_virtual_mic translator.owner=true",
            ),
        ]),
        list_modules(&[
            (
                101,
                "module-null-sink",
                "sink_name=translator_mic_out translator.owner=true",
            ),
            (
                102,
                "module-remap-source",
                "source_name=translator_virtual_mic translator.owner=true",
            ),
        ]),
        unload(102),
        list_modules(&[(
            101,
            "module-null-sink",
            "sink_name=translator_mic_out translator.owner=true",
        )]),
        unload(101),
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let unloaded = graph
        .cleanup_owned()
        .expect("a missing module is already clean");

    assert_eq!(unloaded, [102, 101]);
    assert!(!journal.exists());
    runner.assert_drained();
}

#[test]
fn cleanup_accepts_missing_middle_module_id() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_journal(&journal, [101, 102, 103]);
    let remaining = list_modules(&[
        (
            101,
            "module-null-sink",
            "sink_name=translator_mic_out translator.owner=true",
        ),
        (
            103,
            "module-null-sink",
            "sink_name=translator_remote_in translator.owner=true",
        ),
    ]);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        remaining,
        list_modules(&[
            (
                101,
                "module-null-sink",
                "sink_name=translator_mic_out translator.owner=true",
            ),
            (
                103,
                "module-null-sink",
                "sink_name=translator_remote_in translator.owner=true",
            ),
        ]),
        unload(103),
        list_modules(&[(
            101,
            "module-null-sink",
            "sink_name=translator_mic_out translator.owner=true",
        )]),
        unload(101),
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let unloaded = graph
        .cleanup_owned()
        .expect("ordered role subsets must remain valid");

    assert_eq!(unloaded, [103, 101]);
    assert!(!journal.exists());
    runner.assert_drained();
}

#[test]
fn cleanup_rejects_virtual_mic_with_wrong_master() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_intent(&journal);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        list_modules(&[(
            102,
            "module-remap-source",
            "master=translator_remote_in.monitor source_name=translator_virtual_mic channels=1 channel_map=mono remix=no translator.owner=true",
        )]),
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let error = graph
        .cleanup_owned()
        .expect_err("wrong master must never prove graph ownership");

    assert_eq!(error.code(), AudioGraphErrorCode::CleanupFailed);
    assert!(journal.exists());
    runner.assert_drained();
}

#[test]
fn graph_operations_are_serialized_by_the_journal_lock() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (probe_tx, probe_rx) = mpsc::channel();

    let first_journal = journal.clone();
    let first = thread::spawn(move || {
        let mut graph = test_graph(
            BlockingRunner {
                started: started_tx,
                release: release_rx,
            },
            first_journal,
        );
        graph.ensure_endpoints()
    });
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("first operation must reach pactl");

    let second = thread::spawn(move || {
        let mut graph = test_graph(ProbeRunner { called: probe_tx }, journal);
        graph.ensure_endpoints()
    });
    let early_probe = probe_rx.recv_timeout(Duration::from_millis(150));
    release_tx.send(()).unwrap();
    let _ = first.join().unwrap();
    if early_probe.is_err() {
        probe_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("second operation must continue after lock release");
    }
    let _ = second.join().unwrap();

    assert!(
        early_probe.is_err(),
        "second graph operation entered pactl before the first released ownership"
    );
}

#[test]
fn lock_symlink_is_rejected_without_touching_its_target() {
    let temp = tempdir().unwrap();
    let parent = temp.path().join("translator");
    let journal = parent.join("modules.json");
    let target = temp.path().join("foreign-target");
    fs::create_dir_all(&parent).unwrap();
    fs::write(&target, b"foreign").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o644)).unwrap();
    symlink(&target, parent.join(".modules.json.lock")).unwrap();
    let runner = FakeRunner::new(Vec::new());
    let mut graph = test_graph(runner.clone(), journal);

    let error = graph
        .ensure_endpoints()
        .expect_err("lock symlinks must fail closed");

    assert_eq!(error.code(), AudioGraphErrorCode::OwnershipJournalInvalid);
    assert_eq!(fs::read(&target).unwrap(), b"foreign");
    assert_eq!(
        fs::metadata(&target).unwrap().permissions().mode() & 0o777,
        0o644
    );
    runner.assert_drained();
}

#[test]
fn cleanup_rejects_endpoint_argument_prefix_collision() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_journal(&journal, [101, 102, 103]);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        list_modules(&[
            (
                101,
                "module-null-sink",
                "sink_name=translator_mic_out_backup translator.owner=true",
            ),
            (
                102,
                "module-remap-source",
                "source_name=translator_virtual_mic translator.owner=true",
            ),
            (
                103,
                "module-null-sink",
                "sink_name=translator_remote_in translator.owner=true",
            ),
        ]),
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let error = graph
        .cleanup_owned()
        .expect_err("prefix collision must not prove ownership");

    assert_eq!(error.code(), AudioGraphErrorCode::CleanupFailed);
    assert_eq!(journal_ids(&journal), [101, 102, 103]);
    runner.assert_drained();
}

#[test]
fn rollback_rechecks_module_identity_before_unload() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        load_mic_out(101),
        load_virtual_mic(success("102")),
        ExpectedCommand {
            args: load_remote_in(103).args,
            result: failure("", "load-marker"),
        },
        list_modules(&[
            (
                101,
                "module-null-sink",
                "sink_name=translator_mic_out translator.owner=true",
            ),
            (
                102,
                "module-remap-source",
                "source_name=foreign_reused_id translator.owner=true",
            ),
        ]),
    ]);
    let mut graph = test_graph(runner.clone(), journal.clone());

    let error = graph
        .ensure_endpoints()
        .expect_err("reused id must block rollback unload");

    assert_eq!(error.code(), AudioGraphErrorCode::RollbackFailed);
    assert_eq!(journal_ids(&journal), [101, 102]);
    runner.assert_drained();
}
