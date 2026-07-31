use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use tempfile::tempdir;
use translator_audio::{
    AllowedApplication, CommandResult, CommandRunError, CommandRunner, ProcessIdentity,
    PulseRoutingWatcher, REMOTE_IN_SINK, RouteMethod, RouteResolution, RoutingError,
    RoutingErrorCode, RoutingProfile, RoutingWatcher, VirtualPeerCapability,
};
use uuid::Uuid;

#[derive(Clone)]
struct FakeRunner {
    expected: Arc<Mutex<VecDeque<ExpectedCommand>>>,
}

struct ExpectedCommand {
    program: String,
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
        let expected = self
            .expected
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected command");
        assert_eq!(program, expected.program);
        assert_eq!(args, expected.args);
        expected.result
    }
}

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn command(values: &[&str], stdout: &str) -> ExpectedCommand {
    ExpectedCommand {
        program: "pactl".to_owned(),
        args: args(values),
        result: Ok(CommandResult::success(stdout.as_bytes().to_vec())),
    }
}

fn pw_link(values: &[&str]) -> ExpectedCommand {
    ExpectedCommand {
        program: "pw-link".to_owned(),
        args: args(values),
        result: Ok(CommandResult::success(Vec::new())),
    }
}

fn failed_command(values: &[&str]) -> ExpectedCommand {
    ExpectedCommand {
        program: "pactl".to_owned(),
        args: args(values),
        result: Ok(CommandResult::failure(
            Vec::new(),
            b"private-route-marker".to_vec(),
        )),
    }
}

fn failed_pw_link(values: &[&str]) -> ExpectedCommand {
    ExpectedCommand {
        program: "pw-link".to_owned(),
        args: args(values),
        result: Ok(CommandResult::failure(
            Vec::new(),
            b"private-route-marker".to_vec(),
        )),
    }
}

fn assert_error_redacted(error: &RoutingError) {
    for representation in [
        format!("{error:?}"),
        error.to_string(),
        serde_json::to_string(error.safe_status()).unwrap(),
    ] {
        assert!(!representation.contains("private-route-marker"));
    }
}

fn discovery(sink_inputs: serde_json::Value) -> Vec<ExpectedCommand> {
    vec![
        command(
            &["--format=json", "list", "sink-inputs"],
            &sink_inputs.to_string(),
        ),
        command(&["--format=json", "list", "source-outputs"], "[]"),
        command(&["--format=json", "list", "sources"], "[]"),
        command(
            &["--format=json", "list", "sinks"],
            &serde_json::json!([
                {"index": 55, "name": "alsa_output.first"},
                {"index": 66, "name": "alsa_output.second"},
                {"index": 77, "name": "alsa_output.third"},
                {"index": 900, "name": "translator_remote_in"}
            ])
            .to_string(),
        ),
    ]
}

fn discovery_with_sinks(
    sink_inputs: serde_json::Value,
    sinks: serde_json::Value,
) -> Vec<ExpectedCommand> {
    vec![
        command(
            &["--format=json", "list", "sink-inputs"],
            &sink_inputs.to_string(),
        ),
        command(&["--format=json", "list", "source-outputs"], "[]"),
        command(&["--format=json", "list", "sources"], "[]"),
        command(&["--format=json", "list", "sinks"], &sinks.to_string()),
    ]
}

fn discovery_with_io(
    sink_inputs: serde_json::Value,
    source_outputs: serde_json::Value,
    sources: serde_json::Value,
    sinks: serde_json::Value,
) -> Vec<ExpectedCommand> {
    vec![
        command(
            &["--format=json", "list", "sink-inputs"],
            &sink_inputs.to_string(),
        ),
        command(
            &["--format=json", "list", "source-outputs"],
            &source_outputs.to_string(),
        ),
        command(&["--format=json", "list", "sources"], &sources.to_string()),
        command(&["--format=json", "list", "sinks"], &sinks.to_string()),
    ]
}

fn sinks_with_one_running_physical() -> serde_json::Value {
    serde_json::json!([
        {"index": 55, "name": "alsa_output.first", "state": "SUSPENDED"},
        {"index": 66, "name": "alsa_output.second", "state": "RUNNING"},
        {"index": 900, "name": "translator_remote_in", "state": "IDLE"},
        {"index": 901, "name": "translator_mic_out", "state": "RUNNING"}
    ])
}

fn sources_with_one_running_physical() -> serde_json::Value {
    serde_json::json!([
        {"index": 501, "name": "alsa_input.first", "state": "SUSPENDED"},
        {"index": 502, "name": "translator_virtual_mic", "state": "RUNNING"},
        {"index": 503, "name": "alsa_input.second", "state": "RUNNING"}
    ])
}

fn move_stream(stream_id: u32, sink: &str) -> ExpectedCommand {
    command(&["move-sink-input", &stream_id.to_string(), sink], "")
}

fn move_source_output(stream_id: u32, source: &str) -> ExpectedCommand {
    command(&["move-source-output", &stream_id.to_string(), source], "")
}

fn stream(
    id: u32,
    sink: u32,
    app_name: &str,
    binary: &str,
    role: &str,
    media_name: &str,
) -> serde_json::Value {
    serde_json::json!({
        "index": id,
        "sink": sink,
        "properties": {
            "application.name": app_name,
            "application.process.binary": binary,
            "media.role": role,
            "media.name": media_name
        }
    })
}

fn source_output(id: u32, source: u32, app_name: &str, binary: &str) -> serde_json::Value {
    serde_json::json!({
        "index": id,
        "source": source,
        "properties": {
            "application.name": app_name,
            "application.process.binary": binary,
            "media.name": "RecordStream"
        }
    })
}

fn virtual_peer_stream(
    id: u32,
    sink: u32,
    session_id: Uuid,
    process_id: u32,
    object_serial: u64,
) -> serde_json::Value {
    serde_json::json!({
        "index": id,
        "sink": sink,
        "properties": {
            "application.name": "translator-virtual-peer",
            "application.process.binary": "pacat",
            "application.process.id": process_id.to_string(),
            "object.serial": object_serial.to_string(),
            "media.role": "communication",
            "media.name": "translator-virtual-peer",
            "translator.owner": "true",
            "translator.test_profile": "human_round_trip",
            "translator.self_test_session": session_id.to_string()
        }
    })
}

fn with_target_object(mut stream: serde_json::Value, target: &str) -> serde_json::Value {
    stream["properties"]["target.object"] = serde_json::Value::String(target.to_owned());
    stream
}

fn with_node_name(mut stream: serde_json::Value, node_name: &str) -> serde_json::Value {
    stream["properties"]["node.name"] = serde_json::Value::String(node_name.to_owned());
    stream
}

#[test]
fn forged_virtual_peer_metadata_is_rejected_without_daemon_capability() {
    let session_id = Uuid::new_v4();
    let inputs = serde_json::json!([virtual_peer_stream(41, 55, session_id, 9001, 7001)]);
    let runner = FakeRunner::new(discovery(inputs));
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(Some(41)).unwrap_err();

    assert_eq!(state.code(), RoutingErrorCode::InvalidManualOverride);
    runner.assert_drained();
}

#[test]
fn exact_daemon_capability_authorizes_one_virtual_peer_route() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    let initial = serde_json::json!([virtual_peer_stream(41, 55, session_id, process.pid, 7001)]);
    let routed = serde_json::json!([virtual_peer_stream(41, 900, session_id, process.pid, 7001)]);
    let mut expected = discovery(initial);
    expected.push(move_stream(41, REMOTE_IN_SINK));
    expected.extend(discovery(routed));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher
        .route_virtual_peer(VirtualPeerCapability {
            session_id,
            stream_id: 41,
            object_serial: 7001,
            process,
            process_binary: "pacat".to_owned(),
        })
        .unwrap();

    assert_eq!(
        state.active_route.unwrap().application,
        AllowedApplication::SyntheticValidation
    );
    runner.assert_drained();
}

#[test]
fn virtual_peer_route_rejects_forged_target_property_when_actual_sink_did_not_move() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    let initial = serde_json::json!([virtual_peer_stream(41, 55, session_id, process.pid, 7001)]);
    let not_routed = serde_json::json!([with_target_object(
        virtual_peer_stream(41, 55, session_id, process.pid, 7001),
        REMOTE_IN_SINK,
    )]);
    let mut expected = discovery(initial);
    expected.push(move_stream(41, REMOTE_IN_SINK));
    expected.extend(discovery(not_routed));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let error = watcher
        .route_virtual_peer(VirtualPeerCapability {
            session_id,
            stream_id: 41,
            object_serial: 7001,
            process,
            process_binary: "pacat".to_owned(),
        })
        .unwrap_err();

    assert_eq!(error.code(), RoutingErrorCode::MoveFailed);
    runner.assert_drained();
}

#[test]
fn virtual_peer_route_accepts_actual_remote_sink_with_stale_target_property() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    let initial = serde_json::json!([virtual_peer_stream(41, 55, session_id, process.pid, 7001)]);
    let routed = serde_json::json!([with_target_object(
        virtual_peer_stream(41, 900, session_id, process.pid, 7001),
        "alsa_output.first",
    )]);
    let mut expected = discovery(initial);
    expected.push(move_stream(41, REMOTE_IN_SINK));
    expected.extend(discovery(routed.clone()));
    expected.extend(discovery(routed));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);
    let capability = VirtualPeerCapability {
        session_id,
        stream_id: 41,
        object_serial: 7001,
        process,
        process_binary: "pacat".to_owned(),
    };

    let state = watcher.route_virtual_peer(capability.clone()).unwrap();
    watcher
        .validate_virtual_peer_route(&capability, REMOTE_IN_SINK)
        .unwrap();

    assert_eq!(state.resolution, RouteResolution::Routed);
    runner.assert_drained();
}

#[test]
fn virtual_peer_validation_rejects_mutated_or_conflicting_routed_streams() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    let initial = serde_json::json!([virtual_peer_stream(41, 55, session_id, process.pid, 7001)]);
    let routed_stream = virtual_peer_stream(41, 900, session_id, process.pid, 7001);
    let mut changed_name = routed_stream.clone();
    changed_name["properties"]["application.name"] = "forged-peer".into();
    let mut changed_serial = routed_stream.clone();
    changed_serial["properties"]["object.serial"] = "7002".into();
    let moved_away = virtual_peer_stream(41, 55, session_id, process.pid, 7001);
    let conflict = serde_json::json!([
        routed_stream.clone(),
        stream(99, 900, "Music", "player", "music", "Playback")
    ]);

    for invalid in [
        serde_json::json!([changed_name]),
        serde_json::json!([changed_serial]),
        serde_json::json!([moved_away]),
        serde_json::json!([]),
        conflict,
    ] {
        let routed = serde_json::json!([routed_stream.clone()]);
        let mut expected = discovery(initial.clone());
        expected.push(move_stream(41, REMOTE_IN_SINK));
        expected.extend(discovery(routed));
        expected.extend(discovery(invalid));
        let runner = FakeRunner::new(expected);
        let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);
        let capability = VirtualPeerCapability {
            session_id,
            stream_id: 41,
            object_serial: 7001,
            process,
            process_binary: "pacat".to_owned(),
        };
        watcher.route_virtual_peer(capability.clone()).unwrap();

        assert!(
            watcher
                .validate_virtual_peer_route(&capability, REMOTE_IN_SINK)
                .is_err()
        );
        assert_eq!(watcher.active_route().unwrap().stream_id, 41);
        runner.assert_drained();
    }
}

#[test]
fn virtual_peer_validation_rejects_a_mismatched_capability_before_discovery() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    let initial = serde_json::json!([virtual_peer_stream(41, 55, session_id, process.pid, 7001)]);
    let routed = serde_json::json!([virtual_peer_stream(41, 900, session_id, process.pid, 7001)]);
    let mut expected = discovery(initial);
    expected.push(move_stream(41, REMOTE_IN_SINK));
    expected.extend(discovery(routed));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);
    let capability = VirtualPeerCapability {
        session_id,
        stream_id: 41,
        object_serial: 7001,
        process,
        process_binary: "pacat".to_owned(),
    };
    watcher.route_virtual_peer(capability.clone()).unwrap();
    let mut mismatched = capability;
    mismatched.object_serial = 7002;

    assert!(
        watcher
            .validate_virtual_peer_route(&mismatched, REMOTE_IN_SINK)
            .is_err()
    );
    assert_eq!(watcher.active_route().unwrap().stream_id, 41);
    runner.assert_drained();
}

#[test]
fn stale_or_mismatched_virtual_peer_capability_never_authorizes_metadata() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    for capability in [
        VirtualPeerCapability {
            session_id: Uuid::new_v4(),
            stream_id: 41,
            object_serial: 7001,
            process,
            process_binary: "pacat".to_owned(),
        },
        VirtualPeerCapability {
            session_id,
            stream_id: 42,
            object_serial: 7001,
            process,
            process_binary: "pacat".to_owned(),
        },
        VirtualPeerCapability {
            session_id,
            stream_id: 41,
            object_serial: 7002,
            process,
            process_binary: "pacat".to_owned(),
        },
        VirtualPeerCapability {
            session_id,
            stream_id: 41,
            object_serial: 7001,
            process: ProcessIdentity {
                start_time_ticks: process.start_time_ticks.saturating_add(1),
                ..process
            },
            process_binary: "pacat".to_owned(),
        },
    ] {
        let inputs =
            serde_json::json!([virtual_peer_stream(41, 55, session_id, process.pid, 7001)]);
        let runner = FakeRunner::new(discovery(inputs));
        let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

        assert!(watcher.route_virtual_peer(capability).is_err());
        runner.assert_drained();
    }
}

#[test]
fn multiple_allowlisted_streams_wait_for_manual_selection() {
    let inputs = serde_json::json!([
        stream(
            11,
            55,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            66,
            "Telegram Desktop",
            "telegram-desktop",
            "phone",
            "Call"
        )
    ]);
    let runner = FakeRunner::new(discovery(inputs));
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::AwaitingSelection);
    assert_eq!(state.candidates.len(), 2);
    assert!(state.active_route.is_none());
    runner.assert_drained();
}

#[test]
fn manual_override_moves_only_the_selected_stream() {
    let inputs = serde_json::json!([
        stream(
            11,
            55,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            66,
            "Telegram Desktop",
            "telegram-desktop",
            "phone",
            "Call"
        )
    ]);
    let mut expected = discovery(inputs);
    expected.push(move_stream(22, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(Some(22)).unwrap();

    let active = state.active_route.unwrap();
    assert_eq!(active.stream_id, 22);
    assert_eq!(active.original_sink_id, 66);
    assert_eq!(active.original_sink_name, "alsa_output.second");
    assert_eq!(active.application, AllowedApplication::Telegram);
    runner.assert_drained();
}

#[test]
fn manual_override_routes_matching_capture_to_virtual_mic() {
    let sink_inputs = serde_json::json!([stream(
        11,
        55,
        "Google Chrome",
        "chrome (deleted)",
        "",
        "Playback"
    )]);
    let source_outputs = serde_json::json!([source_output(
        70,
        503,
        "Google Chrome input",
        "chrome (deleted)"
    )]);
    let routed_source_outputs = serde_json::json!([source_output(
        70,
        502,
        "Google Chrome input",
        "chrome (deleted)"
    )]);
    let sinks = sinks_with_one_running_physical();
    let sources = sources_with_one_running_physical();
    let mut expected =
        discovery_with_io(sink_inputs, source_outputs, sources.clone(), sinks.clone());
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.push(move_source_output(70, "translator_virtual_mic"));
    expected.extend(discovery_with_io(
        serde_json::json!([stream(
            11,
            900,
            "Google Chrome",
            "chrome (deleted)",
            "",
            "Playback"
        )]),
        routed_source_outputs,
        sources,
        sinks,
    ));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(Some(11)).unwrap();
    let inspected = watcher.inspect().unwrap();

    assert_eq!(state.active_route.unwrap().stream_id, 11);
    assert_eq!(
        inspected
            .source_outputs
            .iter()
            .find(|output| output.stream_id == 70)
            .unwrap()
            .source_name
            .as_deref(),
        Some("translator_virtual_mic")
    );
    runner.assert_drained();
}

#[test]
fn shutdown_restore_moves_active_stream_back_to_its_original_sink() {
    let original = serde_json::json!([
        stream(
            11,
            55,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            66,
            "Telegram Desktop",
            "telegram-desktop",
            "phone",
            "Call"
        )
    ]);
    let routed = serde_json::json!([
        stream(
            11,
            55,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            900,
            "Telegram Desktop",
            "telegram-desktop",
            "phone",
            "Call"
        )
    ]);
    let mut expected = discovery(original);
    expected.push(move_stream(22, REMOTE_IN_SINK));
    expected.extend(discovery(routed));
    expected.push(move_stream(22, "alsa_output.second"));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(Some(22)).unwrap();
    let restored = watcher.restore_active().unwrap();

    assert!(restored.active_route.is_none());
    assert_eq!(restored.resolution, RouteResolution::AwaitingSelection);
    assert!(watcher.active_route().is_none());
    runner.assert_drained();
}

#[test]
fn restore_active_moves_matching_virtual_mic_capture_back_to_physical_source() {
    let original = serde_json::json!([stream(
        11,
        55,
        "Google Chrome",
        "chrome (deleted)",
        "",
        "Playback"
    )]);
    let routed = serde_json::json!([stream(
        11,
        900,
        "Google Chrome",
        "chrome (deleted)",
        "",
        "Playback"
    )]);
    let physical_capture = serde_json::json!([source_output(
        70,
        503,
        "Google Chrome input",
        "chrome (deleted)"
    )]);
    let virtual_capture = serde_json::json!([source_output(
        70,
        502,
        "Google Chrome input",
        "chrome (deleted)"
    )]);
    let sources = sources_with_one_running_physical();
    let sinks = sinks_with_one_running_physical();
    let mut expected =
        discovery_with_io(original, physical_capture, sources.clone(), sinks.clone());
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.push(move_source_output(70, "translator_virtual_mic"));
    expected.extend(discovery_with_io(
        routed,
        virtual_capture,
        sources.clone(),
        sinks,
    ));
    expected.push(move_stream(11, "alsa_output.first"));
    expected.push(move_source_output(70, "alsa_input.second"));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(Some(11)).unwrap();
    let restored = watcher.restore_active().unwrap();

    assert!(restored.active_route.is_none());
    assert_eq!(restored.resolution, RouteResolution::AwaitingSelection);
    runner.assert_drained();
}

#[test]
fn fresh_watcher_reconcile_restores_crash_left_stream_to_original_sink() {
    let temp = tempdir().unwrap();
    let journal = temp.path().join("translator/routes.json");
    let original = serde_json::json!([stream(
        22,
        66,
        "Telegram Desktop",
        "telegram-desktop",
        "phone",
        "Call"
    )]);
    let routed = serde_json::json!([stream(
        22,
        900,
        "Telegram Desktop",
        "telegram-desktop",
        "phone",
        "Call"
    )]);
    let mut first_expected = discovery(original);
    first_expected.push(move_stream(22, REMOTE_IN_SINK));
    let first_runner = FakeRunner::new(first_expected);
    let mut first = PulseRoutingWatcher::new_with_route_journal(
        first_runner.clone(),
        RoutingProfile::Production,
        journal.clone(),
    );

    let routed_state = first.reconcile(Some(22)).unwrap();
    assert_eq!(routed_state.resolution, RouteResolution::Routed);
    assert!(journal.exists());
    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&journal).unwrap()).unwrap();
    assert_eq!(saved["active_route"]["stream_id"], 22);
    first_runner.assert_drained();

    let mut second_expected = discovery(routed);
    second_expected.push(move_stream(22, "alsa_output.second"));
    let second_runner = FakeRunner::new(second_expected);
    let mut restarted = PulseRoutingWatcher::new_with_route_journal(
        second_runner.clone(),
        RoutingProfile::Production,
        journal.clone(),
    );

    let restored = restarted.reconcile(None).unwrap();

    assert!(restored.active_route.is_none());
    assert_eq!(restored.resolution, RouteResolution::AwaitingSelection);
    assert!(restarted.active_route().is_none());
    assert!(!journal.exists());
    second_runner.assert_drained();
}

#[test]
fn single_non_call_browser_audio_is_discovered_but_not_auto_routed() {
    let inputs = serde_json::json!([stream(11, 55, "Firefox", "firefox", "music", "YouTube")]);
    let runner = FakeRunner::new(discovery(inputs));
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.candidates.len(), 1);
    assert_eq!(state.resolution, RouteResolution::AwaitingSelection);
    assert!(state.active_route.is_none());
    runner.assert_drained();
}

#[test]
fn single_browser_playback_with_matching_capture_is_auto_routed_as_duplex_call() {
    let sink_inputs = serde_json::json!([stream(
        11,
        55,
        "Google Chrome",
        "chrome (deleted)",
        "",
        "Playback"
    )]);
    let source_outputs = serde_json::json!([source_output(
        70,
        503,
        "Google Chrome input",
        "chrome (deleted)"
    )]);
    let sinks = sinks_with_one_running_physical();
    let sources = sources_with_one_running_physical();
    let mut expected = discovery_with_io(sink_inputs, source_outputs, sources, sinks);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.push(move_source_output(70, "translator_virtual_mic"));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::Routed);
    assert_eq!(state.active_route.unwrap().stream_id, 11);
    runner.assert_drained();
}

#[test]
fn single_browser_playback_with_matching_virtual_capture_is_auto_routed_after_restart() {
    let sink_inputs = serde_json::json!([stream(
        11,
        55,
        "Google Chrome",
        "chrome (deleted)",
        "",
        "Playback"
    )]);
    let source_outputs = serde_json::json!([source_output(
        70,
        502,
        "Google Chrome input",
        "chrome (deleted)"
    )]);
    let sinks = sinks_with_one_running_physical();
    let sources = sources_with_one_running_physical();
    let mut expected = discovery_with_io(sink_inputs, source_outputs, sources, sinks);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::Routed);
    assert_eq!(state.active_route.unwrap().stream_id, 11);
    runner.assert_drained();
}

#[test]
fn read_only_inspection_never_routes_a_single_call_candidate() {
    let inputs = serde_json::json!([stream(
        12,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let runner = FakeRunner::new(discovery(inputs));
    let watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.inspect().unwrap();

    assert_eq!(state.resolution, RouteResolution::AwaitingSelection);
    assert_eq!(state.candidates.len(), 1);
    assert_eq!(state.active_route, None);
    runner.assert_drained();
}

#[test]
fn single_zoom_voiceengine_playstream_is_auto_routed() {
    let zoom = serde_json::json!([stream(11, 55, "ZOOM VoiceEngine", "zoom", "", "playStream")]);
    let mut expected = discovery(zoom);
    expected.push(failed_command(&["move-sink-input", "11", REMOTE_IN_SINK]));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "alsa_output.first:playback_FL",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FR",
            "alsa_output.first:playback_FR",
        ]),
    ]);
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    let active = state.active_route.unwrap();
    assert_eq!(active.stream_id, 11);
    assert_eq!(active.application, AllowedApplication::Zoom);
    assert_eq!(active.route_method, RouteMethod::PipeWireLinks);
    assert_eq!(state.resolution, RouteResolution::Routed);
    runner.assert_drained();
}

#[test]
fn translator_owned_stream_is_never_a_candidate_or_moved() {
    let inputs = serde_json::json!([
        {
            "index": 10,
            "sink": 55,
            "properties": {
                "application.name": "Firefox",
                "application.process.binary": "firefox",
                "media.role": "communication",
                "translator.owner": "true"
            }
        },
        stream(11, 55, "Firefox", "firefox", "communication", "WebRTC Voice")
    ]);
    let mut expected = discovery(inputs);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.candidates.len(), 1);
    assert_eq!(state.active_route.unwrap().stream_id, 11);
    runner.assert_drained();
}

#[test]
fn manual_override_cannot_bypass_translator_ownership_exclusion() {
    let inputs = serde_json::json!([{
        "index": 10,
        "sink": 55,
        "properties": {
            "application.name": "Firefox",
            "application.process.binary": "firefox",
            "media.role": "communication",
            "translator.owner": "true"
        }
    }]);
    let runner = FakeRunner::new(discovery(inputs));
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let error = watcher.reconcile(Some(10)).unwrap_err();

    assert_eq!(error.code(), RoutingErrorCode::InvalidManualOverride);
    assert!(watcher.active_route().is_none());
    runner.assert_drained();
}

#[test]
fn production_excludes_paplay_but_validation_profile_allows_explicit_selection() {
    let paplay = serde_json::json!([stream(
        33,
        55,
        "paplay",
        "paplay",
        "music",
        "Translator route validation"
    )]);
    let production_runner = FakeRunner::new(discovery(paplay.clone()));
    let mut production =
        PulseRoutingWatcher::new(production_runner.clone(), RoutingProfile::Production);

    let production_state = production.reconcile(Some(33)).unwrap_err();

    assert_eq!(
        production_state.code(),
        RoutingErrorCode::InvalidManualOverride
    );
    production_runner.assert_drained();

    let mut expected = discovery(paplay);
    expected.push(move_stream(33, REMOTE_IN_SINK));
    let validation_runner = FakeRunner::new(expected);
    let mut validation = PulseRoutingWatcher::new(
        validation_runner.clone(),
        RoutingProfile::SyntheticValidation,
    );

    let validation_state = validation.reconcile(Some(33)).unwrap();

    assert_eq!(validation_state.active_route.unwrap().stream_id, 33);
    validation_runner.assert_drained();
}

#[test]
fn selected_application_restart_routes_only_its_single_replacement() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let replacement = serde_json::json!([stream(
        12,
        77,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(replacement));
    expected.push(move_stream(12, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.active_route.unwrap().stream_id, 12);
    assert_eq!(state.resolution, RouteResolution::Routed);
    runner.assert_drained();
}

#[test]
fn selected_application_restart_requires_call_like_replacement() {
    let first = serde_json::json!([stream(11, 55, "Zoom", "zoom", "communication", "Zoom Call")]);
    let non_call_replacement =
        serde_json::json!([stream(12, 77, "Zoom", "zoom", "music", "Zoom Notification")]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(non_call_replacement));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::RouteRemoved);
    assert!(state.active_route.is_none());
    assert_eq!(state.candidates.len(), 1);
    assert!(!state.candidates[0].call_like);
    runner.assert_drained();
}

#[test]
fn zoom_manual_route_falls_back_to_pipewire_links_when_pulse_move_is_rejected() {
    let zoom = serde_json::json!([stream(11, 55, "ZOOM VoiceEngine", "zoom", "", "playStream")]);
    let mut expected = discovery(zoom);
    expected.push(failed_command(&["move-sink-input", "11", REMOTE_IN_SINK]));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "alsa_output.first:playback_FL",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FR",
            "alsa_output.first:playback_FR",
        ]),
    ]);
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(Some(11)).unwrap();

    let active = state.active_route.unwrap();
    assert_eq!(active.stream_id, 11);
    assert_eq!(active.application, AllowedApplication::Zoom);
    assert_eq!(active.target_sink_name, REMOTE_IN_SINK);
    assert_eq!(active.route_method, RouteMethod::PipeWireLinks);
    assert_eq!(state.resolution, RouteResolution::Routed);
    runner.assert_drained();
}

#[test]
fn zoom_pipewire_link_route_restores_original_links() {
    let zoom = serde_json::json!([stream(11, 55, "ZOOM VoiceEngine", "zoom", "", "playStream")]);
    let mut expected = discovery(zoom.clone());
    expected.push(failed_command(&["move-sink-input", "11", REMOTE_IN_SINK]));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "alsa_output.first:playback_FL",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FR",
            "alsa_output.first:playback_FR",
        ]),
    ]);
    expected.extend(discovery(zoom));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "alsa_output.first:playback_FL",
        ]),
        pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "alsa_output.first:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
    ]);
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(Some(11)).unwrap();
    let restored = watcher.restore_active().unwrap();

    assert!(restored.active_route.is_none());
    assert_eq!(restored.resolution, RouteResolution::AwaitingSelection);
    runner.assert_drained();
}

#[test]
fn zoom_pipewire_link_route_survives_followup_reconcile_without_retrying_pulse_move() {
    let zoom = serde_json::json!([stream(11, 55, "ZOOM VoiceEngine", "zoom", "", "playStream")]);
    let mut expected = discovery(zoom.clone());
    expected.push(failed_command(&["move-sink-input", "11", REMOTE_IN_SINK]));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "alsa_output.first:playback_FL",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FR",
            "alsa_output.first:playback_FR",
        ]),
    ]);
    expected.extend(discovery(zoom));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(Some(11)).unwrap();
    let state = watcher.reconcile(None).unwrap();

    let active = state.active_route.unwrap();
    assert_eq!(active.stream_id, 11);
    assert_eq!(active.route_method, RouteMethod::PipeWireLinks);
    assert_eq!(state.resolution, RouteResolution::Routed);
    runner.assert_drained();
}

#[test]
fn zoom_pipewire_fallback_uses_node_name_when_application_name_is_display_label() {
    let zoom = serde_json::json!([with_node_name(
        stream(11, 55, "Zoom Workplace", "zoom", "", "playStream"),
        "ZOOM VoiceEngine",
    )]);
    let mut expected = discovery(zoom);
    expected.push(failed_command(&["move-sink-input", "11", REMOTE_IN_SINK]));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "alsa_output.first:playback_FL",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FR",
            "alsa_output.first:playback_FR",
        ]),
    ]);
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(Some(11)).unwrap();

    let active = state.active_route.unwrap();
    assert_eq!(
        active.pipewire_node_name.as_deref(),
        Some("ZOOM VoiceEngine")
    );
    assert_eq!(active.route_method, RouteMethod::PipeWireLinks);
    runner.assert_drained();
}

#[test]
fn zoom_pipewire_fallback_links_only_the_selected_candidate() {
    let inputs = serde_json::json!([
        stream(11, 55, "ZOOM VoiceEngine", "zoom", "", "playStream"),
        stream(
            22,
            66,
            "Google Chrome",
            "google-chrome",
            "communication",
            "Meet Audio"
        )
    ]);
    let mut expected = discovery(inputs);
    expected.push(failed_command(&["move-sink-input", "11", REMOTE_IN_SINK]));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "alsa_output.first:playback_FL",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FR",
            "alsa_output.first:playback_FR",
        ]),
    ]);
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(Some(11)).unwrap();

    let active = state.active_route.unwrap();
    assert_eq!(active.application, AllowedApplication::Zoom);
    assert_eq!(active.stream_id, 11);
    assert_eq!(state.candidates.len(), 2);
    runner.assert_drained();
}

#[test]
fn zoom_pipewire_fallback_failure_removes_partial_remote_link_without_active_route() {
    let zoom = serde_json::json!([stream(11, 55, "ZOOM VoiceEngine", "zoom", "", "playStream")]);
    let mut expected = discovery(zoom);
    expected.push(failed_command(&["move-sink-input", "11", REMOTE_IN_SINK]));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        failed_pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
    ]);
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let error = watcher.reconcile(Some(11)).unwrap_err();

    assert_eq!(error.code(), RoutingErrorCode::MoveFailed);
    assert_error_redacted(&error);
    assert!(watcher.active_route().is_none());
    runner.assert_drained();
}

#[test]
fn reused_stream_id_with_a_different_app_identity_is_not_routed() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let reused = serde_json::json!([stream(
        11,
        66,
        "Firefox",
        "lookalike-player",
        "communication",
        "WebRTC Voice"
    )]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(reused));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::RouteRemoved);
    assert_eq!(state.active_route, None);
    assert_eq!(state.candidates[0].application, AllowedApplication::Firefox);
    runner.assert_drained();
}

#[test]
fn manual_selection_can_replace_a_reused_stream_id_without_restoring_the_new_stream() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let reused = serde_json::json!([stream(
        11,
        66,
        "Firefox",
        "lookalike-player",
        "communication",
        "WebRTC Voice"
    )]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(reused));
    expected.push(move_stream(11, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(Some(11)).unwrap();

    let active = state.active_route.unwrap();
    assert_eq!(active.application, AllowedApplication::Firefox);
    assert_eq!(active.original_sink_name, "alsa_output.second");
    runner.assert_drained();
}

#[test]
fn ambiguous_same_application_replacements_are_not_routed() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let replacements = serde_json::json!([
        stream(
            12,
            66,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice A"
        ),
        stream(
            13,
            77,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice B"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(replacements));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::RouteRemoved);
    assert!(state.active_route.is_none());
    assert_eq!(state.candidates.len(), 2);
    runner.assert_drained();
}

#[test]
fn fresh_watcher_reports_conflict_for_an_app_stream_already_on_remote_in() {
    let inputs = serde_json::json!([stream(
        11,
        900,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let runner = FakeRunner::new(discovery(inputs));
    let watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.inspect().unwrap();

    assert_eq!(state.resolution, RouteResolution::RouteConflict);
    assert_eq!(state.active_route, None);
    assert!(state.candidates.is_empty());
    assert_eq!(state.conflicting_stream_ids, vec![11]);
    runner.assert_drained();
}

#[test]
fn reconcile_restores_stale_non_call_allowlisted_stream_left_on_remote_in() {
    let stale = serde_json::json!([stream(
        11,
        900,
        "Google Chrome",
        "chrome (deleted)",
        "",
        "Playback"
    )]);
    let restored = serde_json::json!([stream(
        11,
        66,
        "Google Chrome",
        "chrome (deleted)",
        "",
        "Playback"
    )]);
    let sinks = sinks_with_one_running_physical();
    let mut expected = discovery_with_sinks(stale, sinks.clone());
    expected.push(move_stream(11, "alsa_output.second"));
    expected.extend(discovery_with_sinks(restored, sinks));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::AwaitingSelection);
    assert_eq!(state.conflicting_stream_ids, Vec::<u32>::new());
    assert_eq!(state.candidates.len(), 1);
    assert!(!state.candidates[0].call_like);
    assert_eq!(state.candidates[0].current_sink_name, "alsa_output.second");
    assert!(state.active_route.is_none());
    runner.assert_drained();
}

#[test]
fn non_allowlisted_stream_on_remote_in_is_also_a_route_conflict() {
    let inputs = serde_json::json!([stream(
        44,
        900,
        "Music Player",
        "music-player",
        "music",
        "Playback"
    )]);
    let runner = FakeRunner::new(discovery(inputs));
    let watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.inspect().unwrap();

    assert_eq!(state.resolution, RouteResolution::RouteConflict);
    assert!(state.candidates.is_empty());
    assert_eq!(state.conflicting_stream_ids, vec![44]);
    runner.assert_drained();
}

#[test]
fn second_app_stream_on_remote_in_blocks_an_existing_active_route() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let conflict = serde_json::json!([
        stream(
            11,
            900,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            900,
            "Telegram Desktop",
            "telegram-desktop",
            "phone",
            "Call"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(conflict));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::RouteConflict);
    assert_eq!(state.active_route.unwrap().stream_id, 11);
    assert_eq!(state.conflicting_stream_ids, vec![22]);
    runner.assert_drained();
}

#[test]
fn manual_override_cannot_bypass_an_existing_remote_in_conflict() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let conflict_and_candidate = serde_json::json!([
        stream(
            11,
            900,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(22, 900, "Music Player", "music-player", "music", "Playback"),
        stream(
            33,
            66,
            "Telegram Desktop",
            "telegram-desktop",
            "phone",
            "Call"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(conflict_and_candidate));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(Some(33)).unwrap();

    assert_eq!(state.resolution, RouteResolution::RouteConflict);
    assert_eq!(state.active_route.unwrap().stream_id, 11);
    assert_eq!(state.conflicting_stream_ids, vec![22]);
    runner.assert_drained();
}

#[test]
fn restart_identity_requires_the_same_allowed_app_and_binary() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let replacement_and_lookalike = serde_json::json!([
        stream(
            12,
            66,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            13,
            77,
            "Firefox",
            "lookalike-player",
            "communication",
            "WebRTC Voice"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(replacement_and_lookalike));
    expected.push(move_stream(12, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.active_route.unwrap().stream_id, 12);
    runner.assert_drained();
}

#[test]
fn disappeared_selection_auto_routes_the_only_call_like_different_application() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let different = serde_json::json!([stream(
        22,
        66,
        "Google Chrome",
        "google-chrome",
        "communication",
        "Meet Audio"
    )]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(different));
    expected.push(move_stream(22, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.resolution, RouteResolution::Routed);
    assert_eq!(state.active_route.unwrap().stream_id, 22);
    assert_eq!(state.candidates.len(), 1);
    runner.assert_drained();
}

#[test]
fn manual_switch_restores_old_route_before_moving_new_stream() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let both = serde_json::json!([
        stream(
            11,
            900,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            66,
            "Google Chrome",
            "google-chrome",
            "communication",
            "Meet Audio"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(both));
    expected.push(move_stream(11, "alsa_output.first"));
    expected.push(move_stream(22, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(Some(22)).unwrap();

    assert_eq!(state.active_route.unwrap().stream_id, 22);
    runner.assert_drained();
}

#[test]
fn active_route_stays_selected_when_an_unrelated_candidate_appears() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let both = serde_json::json!([
        stream(
            11,
            900,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            66,
            "Google Chrome",
            "google-chrome",
            "communication",
            "Meet Audio"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(both));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.active_route.unwrap().stream_id, 11);
    assert_eq!(state.resolution, RouteResolution::Routed);
    assert_eq!(state.candidates.len(), 2);
    runner.assert_drained();
}

#[test]
fn browser_fallback_route_switches_to_new_call_like_application() {
    let chrome = stream(11, 55, "Google Chrome", "chrome (deleted)", "", "Playback");
    let chrome_routed = stream(11, 900, "Google Chrome", "chrome (deleted)", "", "Playback");
    let zoom = stream(22, 66, "ZOOM VoiceEngine", "zoom", "", "playStream");
    let chrome_physical_capture = source_output(70, 503, "Google Chrome input", "chrome (deleted)");
    let chrome_virtual_capture = source_output(70, 502, "Google Chrome input", "chrome (deleted)");
    let zoom_physical_capture = source_output(80, 503, "ZOOM VoiceEngine", "zoom");
    let sources = sources_with_one_running_physical();
    let sinks = sinks_with_one_running_physical();

    let mut expected = discovery_with_io(
        serde_json::json!([chrome]),
        serde_json::json!([chrome_physical_capture]),
        sources.clone(),
        sinks.clone(),
    );
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.push(move_source_output(70, "translator_virtual_mic"));
    expected.extend(discovery_with_io(
        serde_json::json!([chrome_routed, zoom]),
        serde_json::json!([chrome_virtual_capture, zoom_physical_capture]),
        sources,
        sinks,
    ));
    expected.push(move_stream(11, "alsa_output.first"));
    expected.push(move_source_output(70, "alsa_input.second"));
    expected.push(failed_command(&["move-sink-input", "22", REMOTE_IN_SINK]));
    expected.extend(vec![
        pw_link(&[
            "ZOOM VoiceEngine:output_FL",
            "translator_remote_in:playback_FL",
        ]),
        pw_link(&[
            "ZOOM VoiceEngine:output_FR",
            "translator_remote_in:playback_FR",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FL",
            "alsa_output.second:playback_FL",
        ]),
        pw_link(&[
            "-d",
            "ZOOM VoiceEngine:output_FR",
            "alsa_output.second:playback_FR",
        ]),
    ]);
    expected.push(move_source_output(80, "translator_virtual_mic"));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let first = watcher.reconcile(None).unwrap();
    let first_active = first.active_route.unwrap();
    assert_eq!(first_active.stream_id, 11);
    assert_eq!(first_active.application, AllowedApplication::Chrome);
    assert_eq!(first_active.route_method, RouteMethod::PulseMove);

    let state = watcher.reconcile(None).unwrap();

    let active = state.active_route.unwrap();
    assert_eq!(active.stream_id, 22);
    assert_eq!(active.application, AllowedApplication::Zoom);
    assert_eq!(active.route_method, RouteMethod::PipeWireLinks);
    assert_eq!(state.resolution, RouteResolution::Routed);
    runner.assert_drained();
}

#[test]
fn restore_failure_prevents_new_route_and_preserves_old_active_state() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let both = serde_json::json!([
        stream(
            11,
            900,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            66,
            "Google Chrome",
            "google-chrome",
            "communication",
            "Meet Audio"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(both));
    expected.push(failed_command(&[
        "move-sink-input",
        "11",
        "alsa_output.first",
    ]));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);
    watcher.reconcile(None).unwrap();

    let error = watcher.reconcile(Some(22)).unwrap_err();

    assert_eq!(error.code(), RoutingErrorCode::RestoreFailed);
    assert_error_redacted(&error);
    assert_eq!(watcher.active_route().unwrap().stream_id, 11);
    runner.assert_drained();
}

#[test]
fn new_move_failure_after_restore_leaves_no_false_active_route() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let both = serde_json::json!([
        stream(
            11,
            900,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            66,
            "Google Chrome",
            "google-chrome",
            "communication",
            "Meet Audio"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(discovery(both));
    expected.push(move_stream(11, "alsa_output.first"));
    expected.push(failed_command(&["move-sink-input", "22", REMOTE_IN_SINK]));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);
    watcher.reconcile(None).unwrap();

    let error = watcher.reconcile(Some(22)).unwrap_err();

    assert_eq!(error.code(), RoutingErrorCode::MoveFailed);
    assert_error_redacted(&error);
    assert!(watcher.active_route().is_none());
    runner.assert_drained();
}

#[test]
fn original_sink_is_restored_by_stable_name_after_sink_id_reuse() {
    let first = serde_json::json!([stream(
        11,
        55,
        "Firefox",
        "firefox",
        "communication",
        "WebRTC Voice"
    )]);
    let both = serde_json::json!([
        stream(
            11,
            900,
            "Firefox",
            "firefox",
            "communication",
            "WebRTC Voice"
        ),
        stream(
            22,
            66,
            "Google Chrome",
            "google-chrome",
            "communication",
            "Meet Audio"
        )
    ]);
    let mut expected = discovery(first);
    expected.push(move_stream(11, REMOTE_IN_SINK));
    expected.extend(vec![
        command(&["--format=json", "list", "sink-inputs"], &both.to_string()),
        command(&["--format=json", "list", "source-outputs"], "[]"),
        command(&["--format=json", "list", "sources"], "[]"),
        command(
            &["--format=json", "list", "sinks"],
            &serde_json::json!([
                {"index": 55, "name": "alsa_output.reused-by-other-device"},
                {"index": 66, "name": "alsa_output.second"},
                {"index": 77, "name": "alsa_output.first"},
                {"index": 900, "name": "translator_remote_in"}
            ])
            .to_string(),
        ),
        move_stream(11, "alsa_output.first"),
        move_stream(22, REMOTE_IN_SINK),
    ]);
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    watcher.reconcile(None).unwrap();
    watcher.reconcile(Some(22)).unwrap();

    runner.assert_drained();
}

#[test]
fn controlled_translator_and_debug_streams_are_excluded_without_owner_marker() {
    let inputs = serde_json::json!([
        stream(1, 55, "translator-daemon", "translator-daemon", "communication", "Voice"),
        stream(2, 55, "translator-sidecar", "python", "communication", "Voice"),
        stream(3, 55, "Translator UI Preview", "translator-ui", "communication", "Voice"),
        {
            "index": 4, "sink": 55,
            "properties": {
                "application.name": "Debug Player",
                "application.process.binary": "pw-play",
                "media.role": "communication",
                "node.name": "translator_debug_playback"
            }
        },
        {
            "index": 5, "sink": 55,
            "properties": {
                "application.name": "TTS",
                "application.process.binary": "pw-play",
                "media.role": "communication",
                "media.name": "Translator_TTS"
            }
        },
        stream(6, 66, "Google Chrome", "google-chrome", "communication", "Meet Audio")
    ]);
    let mut expected = discovery(inputs);
    expected.push(move_stream(6, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.candidates.len(), 1);
    assert_eq!(state.active_route.unwrap().stream_id, 6);
    runner.assert_drained();
}

#[test]
fn allowlist_auto_routes_single_call_like_candidate_without_moving_non_call_streams() {
    let inputs = serde_json::json!([
        stream(
            1,
            55,
            "Telegram Desktop",
            "telegram-desktop",
            "music",
            "Audio"
        ),
        stream(2, 55, "Firefox", "firefox", "music", "Audio"),
        stream(3, 55, "Chromium", "chromium", "music", "Audio"),
        stream(4, 55, "Google Chrome", "google-chrome", "music", "Audio"),
        stream(5, 55, "Zoom", "zoom", "communication", "Zoom Meeting Audio")
    ]);
    let mut expected = discovery(inputs);
    expected.push(move_stream(5, REMOTE_IN_SINK));
    let runner = FakeRunner::new(expected);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();
    let applications: Vec<_> = state
        .candidates
        .iter()
        .map(|candidate| candidate.application)
        .collect();

    assert_eq!(
        applications,
        [
            AllowedApplication::Telegram,
            AllowedApplication::Firefox,
            AllowedApplication::Chromium,
            AllowedApplication::Chrome,
            AllowedApplication::Zoom,
        ]
    );
    assert_eq!(state.active_route.unwrap().stream_id, 5);
    assert_eq!(state.resolution, RouteResolution::Routed);
    runner.assert_drained();
}

#[test]
fn malformed_discovery_json_returns_a_redacted_error() {
    let runner = FakeRunner::new(vec![command(
        &["--format=json", "list", "sink-inputs"],
        "{private-route-marker",
    )]);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let error = watcher.reconcile(None).unwrap_err();

    assert_eq!(error.code(), RoutingErrorCode::DiscoveryFailed);
    assert_error_redacted(&error);
    runner.assert_drained();
}

#[test]
fn source_output_discovery_marks_monitor_and_translator_capture_forbidden() {
    let source_outputs = serde_json::json!([
        {
            "index": 70,
            "source": 501,
            "properties": {
                "application.name": "Recorder",
                "application.process.binary": "recorder"
            }
        },
        {
            "index": 71,
            "source": 502,
            "properties": {
                "application.name": "translator-daemon",
                "translator.owner": "true"
            }
        },
        {
            "index": 72,
            "source": 503,
            "properties": {
                "application.name": "Recorder",
                "application.process.binary": "recorder"
            }
        }
    ]);
    let sources = serde_json::json!([
        {"index": 501, "name": "alsa_output.usb-headset.analog-stereo.monitor"},
        {"index": 502, "name": "translator_virtual_mic"},
        {"index": 503, "name": "alsa_input.usb-headset.mono-fallback"}
    ]);
    let runner = FakeRunner::new(vec![
        command(&["--format=json", "list", "sink-inputs"], "[]"),
        command(
            &["--format=json", "list", "source-outputs"],
            &source_outputs.to_string(),
        ),
        command(&["--format=json", "list", "sources"], &sources.to_string()),
        command(&["--format=json", "list", "sinks"], "[]"),
    ]);
    let mut watcher = PulseRoutingWatcher::new(runner.clone(), RoutingProfile::Production);

    let state = watcher.reconcile(None).unwrap();

    assert_eq!(state.source_outputs.len(), 3);
    assert_eq!(
        state
            .source_outputs
            .iter()
            .filter(|item| item.capture_allowed)
            .map(|item| item.stream_id)
            .collect::<Vec<_>>(),
        [72]
    );
    runner.assert_drained();
}
