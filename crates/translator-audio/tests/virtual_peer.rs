use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use translator_audio::{
    CommandResult, CommandRunError, CommandRunner, ProcessIdentity, VirtualPeerCapability,
    VirtualPeerDiscovery, VirtualPeerDiscoveryErrorCode,
};
use uuid::Uuid;

#[derive(Clone)]
struct FakeRunner {
    results: Arc<Mutex<VecDeque<CommandResult>>>,
}

impl FakeRunner {
    fn new(results: Vec<CommandResult>) -> Self {
        Self {
            results: Arc::new(Mutex::new(results.into())),
        }
    }
}

impl CommandRunner for FakeRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        assert_eq!(args, ["--format=json", "list", "sink-inputs"]);
        Ok(self.results.lock().unwrap().pop_front().unwrap())
    }
}

fn stream(
    session_id: Uuid,
    process: ProcessIdentity,
    object_serial: u64,
    stream_id: u32,
    target: &str,
) -> serde_json::Value {
    serde_json::json!({
        "index": stream_id,
        "sink": 51,
        "properties": {
            "application.name": "translator-virtual-peer",
            "application.process.binary": "pacat",
            "application.process.id": process.pid.to_string(),
            "media.name": "translator-virtual-peer",
            "translator.owner": "true",
            "translator.test_profile": "human_round_trip",
            "translator.self_test_session": session_id.to_string(),
            "object.serial": object_serial.to_string(),
            "target.object": target
        }
    })
}

#[test]
fn discovers_exact_single_live_virtual_peer_capability() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    let runner = FakeRunner::new(vec![CommandResult::success(
        serde_json::to_vec(&serde_json::json!([stream(
            session_id,
            process,
            7001,
            41,
            "alsa_output.headphones"
        )]))
        .unwrap(),
    )]);

    let capability = VirtualPeerDiscovery::new(runner)
        .discover(session_id, process, "alsa_output.headphones")
        .unwrap();

    assert_eq!(capability.session_id, session_id);
    assert_eq!(capability.stream_id, 41);
    assert_eq!(capability.object_serial, 7001);
    assert_eq!(capability.process, process);
}

#[test]
fn rejects_forged_ambiguous_stale_or_wrong_target_streams() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    let valid = stream(session_id, process, 7001, 41, "alsa_output.headphones");
    for payload in [
        serde_json::json!([]),
        serde_json::json!([valid.clone(), valid.clone()]),
        serde_json::json!([stream(
            Uuid::new_v4(),
            process,
            7001,
            41,
            "alsa_output.headphones"
        )]),
        serde_json::json!([stream(
            session_id,
            process,
            7001,
            41,
            "translator_remote_in"
        )]),
    ] {
        let runner = FakeRunner::new(vec![CommandResult::success(
            serde_json::to_vec(&payload).unwrap(),
        )]);
        let error = VirtualPeerDiscovery::new(runner)
            .discover(session_id, process, "alsa_output.headphones")
            .unwrap_err();
        assert_eq!(error.code(), VirtualPeerDiscoveryErrorCode::NoExactStream);
    }
}

#[test]
fn malformed_or_failed_discovery_is_privacy_safe() {
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    for result in [
        CommandResult::failure(Vec::new(), b"private-marker".to_vec()),
        CommandResult::success(b"private-marker".to_vec()),
    ] {
        let error = VirtualPeerDiscovery::new(FakeRunner::new(vec![result]))
            .discover(Uuid::new_v4(), process, "alsa_output.headphones")
            .unwrap_err();
        assert_eq!(error.code(), VirtualPeerDiscoveryErrorCode::DiscoveryFailed);
        assert!(!format!("{error:?} {error}").contains("private-marker"));
    }
}

#[test]
fn cleanup_requires_the_exact_virtual_peer_capability_to_disappear() {
    let session_id = Uuid::new_v4();
    let process = ProcessIdentity::inspect(std::process::id()).unwrap();
    let capability = VirtualPeerCapability {
        session_id,
        stream_id: 41,
        object_serial: 7001,
        process,
        process_binary: "pacat".to_owned(),
    };
    let present = CommandResult::success(
        serde_json::to_vec(&serde_json::json!([stream(
            session_id,
            process,
            7001,
            41,
            "translator_remote_in"
        )]))
        .unwrap(),
    );
    let absent = CommandResult::success(b"[]".to_vec());
    let discovery = VirtualPeerDiscovery::new(FakeRunner::new(vec![present, absent]));

    let error = discovery.ensure_absent(&capability).unwrap_err();
    assert_eq!(
        error.code(),
        VirtualPeerDiscoveryErrorCode::ExactStreamStillPresent
    );
    discovery.ensure_absent(&capability).unwrap();
}
