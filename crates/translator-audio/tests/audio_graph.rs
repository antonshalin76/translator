use std::collections::VecDeque;
use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use fs4::FileExt;
use wait_timeout::ChildExt;

use tempfile::tempdir;
use translator_audio::{
    AudioGraph, AudioGraphError, AudioGraphErrorCode, CommandResult, CommandRunError,
    CommandRunner, EndpointKind, EndpointRole, GraphHealth, MIC_OUT_SINK, PulseAudioGraph,
    REMOTE_IN_SINK, VIRTUAL_MIC_SOURCE,
};

const GENERATION: &str = "test-generation-0001";
const MIC_ARGUMENT: &str = "sink_name=translator_mic_out rate=48000 channels=1 channel_map=mono sink_properties=\"device.description=Translator_Mic_Out translator.owner=true translator.generation=test-generation-0001\"";
const VIRTUAL_ARGUMENT: &str = "master=translator_mic_out.monitor source_name=translator_virtual_mic channels=1 channel_map=mono remix=no source_properties=\"device.description=Translator_Virtual_Mic translator.owner=true translator.generation=test-generation-0001\"";
const REMOTE_ARGUMENT: &str = "sink_name=translator_remote_in rate=48000 channels=2 channel_map=front-left,front-right sink_properties=\"device.description=Translator_Remote_In translator.owner=true translator.generation=test-generation-0001\"";

type RecordedGraphCalls = Arc<Mutex<Vec<(Vec<String>, Instant)>>>;

#[derive(Clone)]
struct DeadlineGraphRunner {
    calls: RecordedGraphCalls,
    delay_first: bool,
}

impl CommandRunner for DeadlineGraphRunner {
    fn run_until(
        &self,
        _: &str,
        args: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        let count = {
            let mut calls = self.calls.lock().unwrap();
            calls.push((args.to_vec(), deadline));
            calls.len()
        };
        if count == 1 && self.delay_first {
            thread::sleep(Duration::from_millis(30));
        }
        assert!(args[0] != "load-module" && args[0] != "unload-module");
        Ok(CommandResult::success(b"[]".to_vec()))
    }
}

#[test]
fn expired_graph_admission_never_initializes_ownership() {
    let temp = tempdir().unwrap();
    let parent = temp.path().join("absent");
    let runner = DeadlineGraphRunner {
        calls: Arc::default(),
        delay_first: false,
    };
    let mut graph = test_graph(runner.clone(), parent.join("modules.json"));
    let error = graph.ensure_endpoints_until(Instant::now()).unwrap_err();
    assert_eq!(error.code(), AudioGraphErrorCode::DeadlineExpired);
    assert!(!parent.exists());
    assert!(runner.calls.lock().unwrap().is_empty());
}

#[test]
fn graph_inspection_transports_one_deadline_and_stops_after_expiry() {
    for delayed in [false, true] {
        let temp = tempdir().unwrap();
        fs::write(temp.path().join(".modules.json.lock"), b"").unwrap();
        let runner = DeadlineGraphRunner {
            calls: Arc::default(),
            delay_first: delayed,
        };
        let graph = test_graph(runner.clone(), temp.path().join("modules.json"));
        let deadline = Instant::now()
            + if delayed {
                Duration::from_millis(15)
            } else {
                Duration::from_secs(1)
            };
        let result = graph.inspect_until(deadline);
        let calls = runner.calls.lock().unwrap();
        assert!(calls.iter().all(|(_, observed)| *observed == deadline));
        if delayed {
            assert_eq!(
                result.unwrap_err().code(),
                AudioGraphErrorCode::DeadlineExpired
            );
            assert_eq!(calls.len(), 1);
        } else {
            assert_ne!(result.unwrap().health, GraphHealth::Ready);
            assert_eq!(calls.len(), 2);
        }
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
    }
}

#[test]
fn expired_graph_cleanup_preserves_ownership_bytes() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("modules.json");
    write_journal(&journal, [101, 102, 103]);
    let before = fs::read(&journal).unwrap();
    let runner = DeadlineGraphRunner {
        calls: Arc::default(),
        delay_first: false,
    };
    let mut graph = test_graph(runner.clone(), journal.clone());
    let error = graph.cleanup_owned_until(Instant::now()).unwrap_err();
    assert_eq!(error.code(), AudioGraphErrorCode::DeadlineExpired);
    assert_eq!(fs::read(&journal).unwrap(), before);
    assert!(runner.calls.lock().unwrap().is_empty());
}

#[derive(Clone)]
struct LostLoadRunner {
    loaded_args: Arc<Mutex<Option<Vec<String>>>>,
    calls: RecordedGraphCalls,
    return_late_id: bool,
}

#[derive(Clone)]
struct GraphInventoryRunner {
    inventory: Arc<Mutex<String>>,
    calls: RecordedGraphCalls,
}

impl CommandRunner for GraphInventoryRunner {
    fn run_until(
        &self,
        program: &str,
        arguments: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        self.calls
            .lock()
            .unwrap()
            .push((arguments.to_vec(), deadline));
        match arguments
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["--format=json", "list", "sinks" | "sources"] => success("[]"),
            ["list", "short", "modules"] => success(&self.inventory.lock().unwrap()),
            ["unload-module", _] => success(""),
            _ => panic!("unexpected graph command"),
        }
    }
}

#[test]
fn duplicate_module_ids_refuse_graph_cleanup_and_preserve_journal_for_recovery() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_journal(&journal, [101, 102, 103]);
    let before = fs::read(&journal).unwrap();
    let healthy = format!(
        "101\tmodule-null-sink\t{MIC_ARGUMENT}\t\n102\tmodule-remap-source\t{VIRTUAL_ARGUMENT}\t\n103\tmodule-null-sink\t{REMOTE_ARGUMENT}\t\n99\tmodule-null-sink\tsink_name=foreign\t\n"
    );
    let runner = GraphInventoryRunner {
        inventory: Arc::new(Mutex::new(format!(
            "103\tmodule-null-sink\tsink_name=foreign\t\n{healthy}"
        ))),
        calls: Arc::default(),
    };
    let mut graph = test_graph(runner.clone(), journal.clone());
    let failed_deadline = Instant::now() + Duration::from_secs(2);
    let rejected = graph.cleanup_owned_until(failed_deadline);
    let failed_calls = std::mem::take(&mut *runner.calls.lock().unwrap());
    let after_failed = fs::read(&journal).ok();
    *runner.inventory.lock().unwrap() = healthy;
    let recovered = graph.cleanup_owned();
    let recovery_calls = std::mem::take(&mut *runner.calls.lock().unwrap());
    let again = graph.cleanup_owned();
    let repeated_calls = std::mem::take(&mut *runner.calls.lock().unwrap());
    assert!(
        rejected.is_err(),
        "duplicate ID authorized effects: {failed_calls:?}"
    );
    assert_eq!(
        rejected.unwrap_err().code(),
        AudioGraphErrorCode::CleanupFailed
    );
    assert_eq!(after_failed.as_deref(), Some(before.as_slice()));
    assert_eq!(
        failed_calls,
        [
            (args(&["--format=json", "list", "sinks"]), failed_deadline),
            (args(&["--format=json", "list", "sources"]), failed_deadline),
            (args(&["list", "short", "modules"]), failed_deadline),
        ]
    );
    assert_eq!(recovered.unwrap(), [103, 102, 101]);
    assert_eq!(
        recovery_calls
            .iter()
            .filter(|(args, _)| args[0] == "unload-module")
            .map(|(args, _)| args.clone())
            .collect::<Vec<_>>(),
        [
            args(&["unload-module", "103"]),
            args(&["unload-module", "102"]),
            args(&["unload-module", "101"]),
        ]
    );
    assert!(again.unwrap().is_empty());
    assert!(repeated_calls.is_empty());
    assert!(!journal.exists());
}

#[test]
fn ambiguous_generation_records_preserve_empty_intent_before_healthy_retry() {
    let cases = [
        (
            "wrong_master",
            "module-remap-source",
            VIRTUAL_ARGUMENT.replace("master=translator_mic_out.monitor", "master=wrong.monitor"),
        ),
        (
            "wrong_name",
            "module-null-sink",
            MIC_ARGUMENT.replace("sink_name=translator_mic_out", "sink_name=foreign"),
        ),
        (
            "wrong_channels",
            "module-null-sink",
            MIC_ARGUMENT.replace("channels=1", "channels=2"),
        ),
        (
            "wrong_module",
            "module-remap-source",
            MIC_ARGUMENT.to_owned(),
        ),
        (
            "malformed_quote",
            "module-null-sink",
            MIC_ARGUMENT.trim_end_matches('"').to_owned(),
        ),
        (
            "duplicate_owner_last_false",
            "module-null-sink",
            MIC_ARGUMENT.replace(
                "translator.owner=true",
                "translator.owner=true translator.owner=false",
            ),
        ),
        (
            "duplicate_owner_last_true",
            "module-null-sink",
            MIC_ARGUMENT.replace(
                "translator.owner=true",
                "translator.owner=false translator.owner=true",
            ),
        ),
        (
            "duplicate_generation_last_foreign",
            "module-null-sink",
            MIC_ARGUMENT.replace(
                GENERATION,
                &format!("{GENERATION} translator.generation=foreign"),
            ),
        ),
        (
            "duplicate_generation_last_own",
            "module-null-sink",
            MIC_ARGUMENT.replace(
                "translator.generation=",
                "translator.generation=foreign translator.generation=",
            ),
        ),
        (
            "top_level_markers",
            "module-null-sink",
            MIC_ARGUMENT.replace('"', ""),
        ),
        (
            "relocated_markers",
            "module-null-sink",
            format!(
                "sink_name=foreign sink_properties=\"device.description='translator.owner=true translator.generation={GENERATION}'\""
            ),
        ),
        (
            "reordered",
            "module-null-sink",
            MIC_ARGUMENT.replace(
                "sink_name=translator_mic_out rate=48000",
                "rate=48000 sink_name=translator_mic_out",
            ),
        ),
        (
            "alternative_quotes",
            "module-null-sink",
            MIC_ARGUMENT.replace('"', "'"),
        ),
        (
            "foreign_description",
            "module-null-sink",
            format!("sink_name=foreign sink_properties=\"device.description={GENERATION}\""),
        ),
        (
            "missing_owner",
            "module-null-sink",
            MIC_ARGUMENT.replace("translator.owner=true ", ""),
        ),
        (
            "wrong_property_container",
            "module-null-sink",
            MIC_ARGUMENT.replace("sink_properties=", "source_properties="),
        ),
        (
            "owner_false",
            "module-null-sink",
            MIC_ARGUMENT.replace("translator.owner=true", "translator.owner=false"),
        ),
        (
            "duplicate_top_level",
            "module-null-sink",
            MIC_ARGUMENT.replace("channels=1", "channels=1 channels=1"),
        ),
        (
            "generation_prefix_collision",
            "module-null-sink",
            MIC_ARGUMENT.replace(GENERATION, &format!("{GENERATION}-foreign")),
        ),
        (
            "trailing_text",
            "module-null-sink",
            format!("{MIC_ARGUMENT} ignored"),
        ),
    ];
    let mut violations = Vec::new();
    for (label, module, argument) in cases {
        for mixed in [false, true] {
            let temp = tempdir().unwrap();
            let journal = temp.path().join("translator/modules.json");
            write_intent(&journal);
            let before = fs::read(&journal).unwrap();
            let canonical = if mixed {
                format!("71\tmodule-null-sink\t{MIC_ARGUMENT}\t\n")
            } else {
                String::new()
            };
            let runner = GraphInventoryRunner {
                inventory: Arc::new(Mutex::new(format!(
                    "{canonical}70\t{module}\t{argument}\t\n99\tmodule-null-sink\tsink_name=foreign\t\n"
                ))),
                calls: Arc::default(),
            };
            let mut graph = test_graph(runner.clone(), journal.clone());
            let deadline = Instant::now() + Duration::from_secs(2);
            let rejected = graph.cleanup_owned_until(deadline);
            let failed_calls = std::mem::take(&mut *runner.calls.lock().unwrap());
            let after_failed = fs::read(&journal).ok();
            *runner.inventory.lock().unwrap() = format!(
                "70\tmodule-null-sink\t{MIC_ARGUMENT}\t\n99\tmodule-null-sink\tsink_name=foreign\t\n"
            );
            let recovered = graph.cleanup_owned();
            let recovery_calls = std::mem::take(&mut *runner.calls.lock().unwrap());
            let again = graph.cleanup_owned();
            let repeated_calls = std::mem::take(&mut *runner.calls.lock().unwrap());
            for (condition, invariant) in [
                (
                    matches!(rejected, Err(error) if error.code() == AudioGraphErrorCode::CleanupFailed),
                    "typed_refusal",
                ),
                (
                    after_failed.as_deref() == Some(before.as_slice()),
                    "unchanged_journal",
                ),
                (
                    failed_calls
                        == [
                            (args(&["--format=json", "list", "sinks"]), deadline),
                            (args(&["--format=json", "list", "sources"]), deadline),
                            (args(&["list", "short", "modules"]), deadline),
                        ],
                    "exact_read_only_prefix",
                ),
                (
                    matches!(recovered, Ok(ids) if ids == [70]),
                    "same_owner_recovery",
                ),
                (
                    recovery_calls
                        .iter()
                        .filter(|(args, _)| args[0] == "unload-module")
                        .map(|(args, _)| args.clone())
                        .collect::<Vec<_>>()
                        == [args(&["unload-module", "70"])],
                    "exact_recovery_unload",
                ),
                (
                    matches!(again, Ok(ids) if ids.is_empty())
                        && repeated_calls.is_empty()
                        && !journal.exists(),
                    "idempotent_completion",
                ),
            ] {
                if !condition {
                    violations.push(format!("{label}/mixed={mixed}/{invariant}"));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "40 entered cases violated: {violations:?}"
    );
}

#[test]
fn duplicate_canonical_orphans_never_authorize_cleanup() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_intent(&journal);
    let before = fs::read(&journal).unwrap();
    let runner = GraphInventoryRunner {
        inventory: Arc::new(Mutex::new(format!(
            "70\tmodule-null-sink\t{MIC_ARGUMENT}\t\n71\tmodule-null-sink\t{MIC_ARGUMENT}\t\n"
        ))),
        calls: Arc::default(),
    };
    let mut graph = test_graph(runner.clone(), journal.clone());
    let deadline = Instant::now() + Duration::from_secs(2);
    let result = graph.cleanup_owned_until(deadline);
    assert!(matches!(result, Err(error) if error.code() == AudioGraphErrorCode::CleanupFailed));
    assert_eq!(fs::read(&journal).unwrap(), before);
    assert_eq!(
        *runner.calls.lock().unwrap(),
        [
            (args(&["--format=json", "list", "sinks"]), deadline),
            (args(&["--format=json", "list", "sources"]), deadline),
            (args(&["list", "short", "modules"]), deadline),
        ]
    );
}

#[test]
fn canonical_orphan_cleanup_preserves_foreign_generation_and_ids() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    write_intent(&journal);
    let runner = GraphInventoryRunner {
        inventory: Arc::new(Mutex::new(format!(
            "70\tmodule-null-sink\t{MIC_ARGUMENT}\t\n99\tmodule-null-sink\t{}\t\n",
            MIC_ARGUMENT.replace(GENERATION, "foreign-generation")
        ))),
        calls: Arc::default(),
    };
    let mut graph = test_graph(runner.clone(), journal.clone());
    let deadline = Instant::now() + Duration::from_secs(2);
    let result = graph.cleanup_owned_until(deadline);
    let calls = std::mem::take(&mut *runner.calls.lock().unwrap());
    assert_eq!(result.unwrap(), [70]);
    assert_eq!(
        calls,
        [
            (args(&["--format=json", "list", "sinks"]), deadline),
            (args(&["--format=json", "list", "sources"]), deadline),
            (args(&["list", "short", "modules"]), deadline),
            (args(&["list", "short", "modules"]), deadline),
            (args(&["unload-module", "70"]), deadline),
        ]
    );
    assert!(!journal.exists());
    assert!(graph.cleanup_owned().unwrap().is_empty());
    assert!(runner.calls.lock().unwrap().is_empty());
}

impl CommandRunner for LostLoadRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        self.calls.lock().unwrap().push((args.to_vec(), deadline));
        let args_view: Vec<_> = args.iter().map(String::as_str).collect();
        let result = match args_view.as_slice() {
            ["--format=json", "list", "sinks"] => {
                if self.loaded_args.lock().unwrap().is_some() {
                    "[{\"index\":11,\"name\":\"translator_mic_out\",\"owner_module\":70}]"
                        .to_owned()
                } else {
                    "[]".to_owned()
                }
            }
            ["--format=json", "list", "sources"] => "[]".to_owned(),
            ["list", "short", "modules"] => {
                let args = self.loaded_args.lock().unwrap();
                let mut modules =
                    "99\tmodule-null-sink\tsink_name=foreign translator.generation=foreign\t\n"
                        .to_owned();
                if let Some(args) = args.as_ref() {
                    modules.push_str(&format!("70\t{}\t{}\t\n", args[1], args[2..].join(" ")));
                }
                modules
            }
            ["load-module", ..] => {
                assert!(
                    self.loaded_args.lock().unwrap().is_none(),
                    "second load crossed expired admission"
                );
                *self.loaded_args.lock().unwrap() = Some(args.to_vec());
                thread::sleep(
                    deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(10),
                );
                if !self.return_late_id {
                    return Err(CommandRunError::DeadlineExpired);
                }
                "70\n".to_owned()
            }
            ["unload-module", "70"] => {
                *self.loaded_args.lock().unwrap() = None;
                String::new()
            }
            _ => panic!("unexpected graph effect/read: {args:?}"),
        };
        Ok(CommandResult::success(result.into_bytes()))
    }
}

#[test]
fn expired_load_keeps_generation_intent_for_fresh_exact_cleanup() {
    for return_late_id in [false, true] {
        let temp = tempdir().unwrap();
        let journal = temp.path().join("modules.json");
        let runner = LostLoadRunner {
            loaded_args: Arc::default(),
            calls: Arc::default(),
            return_late_id,
        };
        let mut graph = test_graph(runner.clone(), journal.clone());
        let deadline = Instant::now() + Duration::from_secs(1);
        let result = graph.ensure_endpoints_until(deadline);
        let failed_calls = runner.calls.lock().unwrap().clone();
        let intent = fs::read(&journal).ok();
        let failed_ids = intent.as_ref().map(|_| journal_ids(&journal));
        let cleanup_deadline = Instant::now() + Duration::from_secs(8);
        let cleanup = graph.cleanup_owned_until(cleanup_deadline);

        assert_eq!(
            result.unwrap_err().code(),
            AudioGraphErrorCode::DeadlineExpired
        );
        assert_eq!(
            failed_calls
                .iter()
                .filter(|(args, _)| args[0] == "load-module")
                .count(),
            1
        );
        assert_eq!(
            failed_calls
                .iter()
                .position(|(args, _)| args[0] == "load-module")
                .unwrap()
                + 1,
            failed_calls.len(),
            "expired load was followed by another command"
        );
        assert!(
            !failed_calls
                .iter()
                .any(|(args, _)| args[0] == "unload-module")
        );
        assert!(
            failed_calls
                .iter()
                .all(|(_, observed)| *observed == deadline)
        );
        let saved: serde_json::Value = serde_json::from_slice(&intent.unwrap()).unwrap();
        assert_eq!(saved["generation"], GENERATION);
        assert_eq!(
            failed_ids.unwrap(),
            if return_late_id { vec![70] } else { vec![] }
        );
        assert_eq!(cleanup.unwrap(), vec![70]);
        assert!(!journal.exists());
        let calls = runner.calls.lock().unwrap();
        assert!(
            calls[failed_calls.len()..]
                .iter()
                .all(|(_, observed)| *observed == cleanup_deadline)
        );
        assert_eq!(
            calls
                .iter()
                .filter(|(args, _)| args[0] == "unload-module")
                .count(),
            1
        );
    }
}

#[derive(Clone, Default)]
struct InspectionRunner(Arc<Mutex<Vec<Vec<String>>>>);

impl CommandRunner for InspectionRunner {
    fn run_until(
        &self,
        _program: &str,
        args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        self.0.lock().unwrap().push(args.to_vec());
        Ok(CommandResult::success(b"[]".to_vec()))
    }
}

#[test]
fn readonly_inspection_does_not_create_missing_parent() {
    let temp = tempdir().unwrap();
    let parent = temp.path().join("missing");
    let runner = InspectionRunner::default();
    let result = test_graph(runner.clone(), parent.join("modules.json")).inspect();

    assert!(!parent.exists(), "inspection created ownership directory");
    assert!(result.is_err(), "missing ownership must be unavailable");
    assert!(runner.0.lock().unwrap().is_empty());
}

#[test]
fn readonly_inspection_does_not_create_lock_or_change_parent_mode() {
    let temp = tempdir().unwrap();
    let parent = temp.path().join("existing");
    fs::create_dir(&parent).unwrap();
    fs::set_permissions(&parent, fs::Permissions::from_mode(0o750)).unwrap();
    let runner = InspectionRunner::default();
    let result = test_graph(runner.clone(), parent.join("modules.json")).inspect();

    assert_eq!(
        fs::read_dir(&parent).unwrap().count(),
        0,
        "inspection created a lock"
    );
    assert_eq!(
        fs::metadata(&parent).unwrap().permissions().mode() & 0o777,
        0o750
    );
    assert!(result.is_err());
    assert!(runner.0.lock().unwrap().is_empty());
}

#[test]
fn readonly_inspection_preserves_existing_lock_and_directory_modes() {
    let temp = tempdir().unwrap();
    let parent = temp.path();
    let lock = parent.join(".modules.json.lock");
    fs::write(&lock, b"lock-marker").unwrap();
    fs::set_permissions(parent, fs::Permissions::from_mode(0o750)).unwrap();
    fs::set_permissions(&lock, fs::Permissions::from_mode(0o640)).unwrap();
    let result = test_graph(InspectionRunner::default(), parent.join("modules.json")).inspect();

    assert_eq!(
        fs::metadata(parent).unwrap().permissions().mode() & 0o777,
        0o750
    );
    assert_eq!(
        fs::metadata(&lock).unwrap().permissions().mode() & 0o777,
        0o640
    );
    assert_eq!(fs::read(&lock).unwrap(), b"lock-marker");
    assert_eq!(fs::read_dir(parent).unwrap().count(), 1);
    assert_ne!(result.unwrap().health, GraphHealth::Ready);
}

#[test]
fn readonly_inspection_rejects_contended_lock_without_waiting() {
    let temp = tempdir().unwrap();
    let lock = fs::File::create(temp.path().join(".modules.json.lock")).unwrap();
    FileExt::lock(&lock).unwrap();
    let journal = temp.path().join("modules.json");
    let runner = InspectionRunner::default();
    let worker_runner = runner.clone();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        done_tx
            .send(test_graph(worker_runner, journal).inspect())
            .unwrap();
    });
    let early = done_rx.recv_timeout(Duration::from_millis(250));
    FileExt::unlock(&lock).unwrap();
    worker.join().unwrap();

    assert!(early.is_ok(), "inspection waited for ownership lock");
    assert!(
        early.unwrap().is_err(),
        "contended ownership must be unavailable"
    );
    assert!(runner.0.lock().unwrap().is_empty());
}

#[test]
fn readonly_inspection_fifo_child() {
    let Some(path) = std::env::var_os("TRANSLATOR_TEST_INSPECT_FIFO") else {
        return;
    };
    let runner = InspectionRunner::default();
    let result = test_graph(runner.clone(), PathBuf::from(path)).inspect();
    assert!(result.is_err());
    assert!(runner.0.lock().unwrap().is_empty());
}

#[test]
fn readonly_inspection_rejects_fifo_journal_without_a_writer() {
    assert_readonly_fifo_rejected("modules.json");
}

#[test]
fn readonly_inspection_rejects_fifo_lock_without_a_writer() {
    assert_readonly_fifo_rejected(".modules.json.lock");
}

fn assert_readonly_fifo_rejected(target: &str) {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("modules.json");
    if target == "modules.json" {
        fs::write(temp.path().join(".modules.json.lock"), b"").unwrap();
    }
    rustix::fs::mknodat(
        rustix::fs::CWD,
        temp.path().join(target),
        rustix::fs::FileType::Fifo,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        0,
    )
    .unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "readonly_inspection_fifo_child", "--nocapture"])
        .env("TRANSLATOR_TEST_INSPECT_FIFO", &journal)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let pid = child.id();
    let started = Instant::now();
    let result = child.wait_timeout(Duration::from_secs(2));
    if !matches!(result, Ok(Some(_))) {
        let _ = child.kill();
    }
    let reaped = child.wait();

    assert!(reaped.is_ok(), "fixture child was not reaped");
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
    assert!(
        matches!(result, Ok(Some(status)) if status.success()),
        "{target}: FIFO inspection failed or blocked for {:?}",
        started.elapsed()
    );
}

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
    fn run_until(
        &self,
        _program: &str,
        _args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        self.started.send(()).unwrap();
        self.release.recv().unwrap();
        Err(CommandRunError::NotFound)
    }
}

struct ProbeRunner {
    called: mpsc::Sender<()>,
}

impl CommandRunner for ProbeRunner {
    fn run_until(
        &self,
        _program: &str,
        _args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
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
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
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
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
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
        .map(|(id, module, argument)| format!("{id}\t{module}\t{argument}\t\n"))
        .collect::<String>();
    ExpectedCommand {
        args: args(&["list", "short", "modules"]),
        result: success(&stdout),
    }
}

fn owned_modules(ids: [u32; 3]) -> ExpectedCommand {
    list_modules(&[
        (ids[0], "module-null-sink", MIC_ARGUMENT),
        (ids[1], "module-remap-source", VIRTUAL_ARGUMENT),
        (ids[2], "module-null-sink", REMOTE_ARGUMENT),
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
        "sink_properties=\"device.description=Translator_Mic_Out translator.owner=true translator.generation={GENERATION}\""
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
        "source_properties=\"device.description=Translator_Virtual_Mic translator.owner=true translator.generation={GENERATION}\""
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
        "sink_properties=\"device.description=Translator_Remote_In translator.owner=true translator.generation={GENERATION}\""
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
    let orphan = list_modules(&[(101, "module-null-sink", MIC_ARGUMENT)]);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        orphan,
        list_modules(&[(101, "module-null-sink", MIC_ARGUMENT)]),
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
                &MIC_ARGUMENT.replace("translator.owner=true", "translator.owner=false"),
            ),
            (102, "module-remap-source", VIRTUAL_ARGUMENT),
            (103, "module-null-sink", REMOTE_ARGUMENT),
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
            (101, "module-null-sink", MIC_ARGUMENT),
            (102, "module-remap-source", VIRTUAL_ARGUMENT),
        ]),
        list_modules(&[
            (101, "module-null-sink", MIC_ARGUMENT),
            (102, "module-remap-source", VIRTUAL_ARGUMENT),
        ]),
        unload(102),
        list_modules(&[(101, "module-null-sink", MIC_ARGUMENT)]),
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
                &MIC_ARGUMENT.replace("translator_mic_out", "foreign_sink"),
            ),
            (102, "module-remap-source", VIRTUAL_ARGUMENT),
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
            (101, "module-null-sink", MIC_ARGUMENT),
            (102, "module-remap-source", VIRTUAL_ARGUMENT),
        ]),
        list_modules(&[
            (101, "module-null-sink", MIC_ARGUMENT),
            (102, "module-remap-source", VIRTUAL_ARGUMENT),
        ]),
        unload(102),
        list_modules(&[(101, "module-null-sink", MIC_ARGUMENT)]),
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
        (101, "module-null-sink", MIC_ARGUMENT),
        (103, "module-null-sink", REMOTE_ARGUMENT),
    ]);
    let runner = FakeRunner::new(vec![
        list_sinks("[]"),
        list_sources("[]"),
        remaining,
        list_modules(&[
            (101, "module-null-sink", MIC_ARGUMENT),
            (103, "module-null-sink", REMOTE_ARGUMENT),
        ]),
        unload(103),
        list_modules(&[(101, "module-null-sink", MIC_ARGUMENT)]),
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
            &VIRTUAL_ARGUMENT.replace(
                "master=translator_mic_out.monitor",
                "master=translator_remote_in.monitor",
            ),
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
fn contended_graph_mutation_rejects_without_entering_commands() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/modules.json");
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (probe_tx, probe_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();

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
        result_tx.send(graph.ensure_endpoints()).unwrap();
    });
    let early_result = result_rx.recv_timeout(Duration::from_millis(150));
    release_tx.send(()).unwrap();
    let _ = first.join().unwrap();
    second.join().unwrap();

    assert_eq!(
        early_result
            .expect("contended mutation must not wait")
            .unwrap_err()
            .code(),
        AudioGraphErrorCode::OwnershipJournalBusy
    );
    assert!(
        probe_rx.try_recv().is_err(),
        "contended mutation entered pactl"
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
                &MIC_ARGUMENT.replace(
                    "sink_name=translator_mic_out",
                    "sink_name=translator_mic_out_backup",
                ),
            ),
            (102, "module-remap-source", VIRTUAL_ARGUMENT),
            (103, "module-null-sink", REMOTE_ARGUMENT),
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
            (101, "module-null-sink", MIC_ARGUMENT),
            (
                102,
                "module-remap-source",
                &VIRTUAL_ARGUMENT.replace(
                    "source_name=translator_virtual_mic",
                    "source_name=foreign_reused_id",
                ),
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
