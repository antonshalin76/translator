use std::collections::HashMap;
use std::fmt;

use serde::Deserialize;
use uuid::Uuid;

use crate::{
    CommandRunError, CommandRunner, ProcessIdentity, SystemCommandRunner, VirtualPeerCapability,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualPeerDiscoveryErrorCode {
    DiscoveryFailed,
    ProcessIdentityStale,
    NoExactStream,
    ExactStreamStillPresent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualPeerDiscoveryError {
    code: VirtualPeerDiscoveryErrorCode,
}

impl VirtualPeerDiscoveryError {
    fn new(code: VirtualPeerDiscoveryErrorCode) -> Self {
        Self { code }
    }

    pub const fn code(&self) -> VirtualPeerDiscoveryErrorCode {
        self.code
    }
}

impl fmt::Display for VirtualPeerDiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.code {
            VirtualPeerDiscoveryErrorCode::DiscoveryFailed => "Virtual peer discovery failed",
            VirtualPeerDiscoveryErrorCode::ProcessIdentityStale => {
                "Virtual peer process identity is stale"
            }
            VirtualPeerDiscoveryErrorCode::NoExactStream => {
                "Exact virtual peer stream is unavailable"
            }
            VirtualPeerDiscoveryErrorCode::ExactStreamStillPresent => {
                "Exact virtual peer stream cleanup is incomplete"
            }
        })
    }
}

impl std::error::Error for VirtualPeerDiscoveryError {}

pub struct VirtualPeerDiscovery<R = SystemCommandRunner> {
    runner: R,
}

impl<R> VirtualPeerDiscovery<R>
where
    R: CommandRunner,
{
    pub const fn new(runner: R) -> Self {
        Self { runner }
    }

    pub fn discover(
        &self,
        session_id: Uuid,
        process: ProcessIdentity,
        expected_target: &str,
    ) -> Result<VirtualPeerCapability, VirtualPeerDiscoveryError> {
        if ProcessIdentity::inspect(process.pid) != Some(process) {
            return Err(VirtualPeerDiscoveryError::new(
                VirtualPeerDiscoveryErrorCode::ProcessIdentityStale,
            ));
        }
        let streams = self.sink_inputs()?;
        let expected_session = session_id.to_string();
        let mut matching = streams
            .into_iter()
            .filter_map(|stream| {
                let properties = &stream.properties;
                let exact = property(properties, "application.name")
                    == Some("translator-virtual-peer")
                    && property(properties, "application.process.binary") == Some("pacat")
                    && property(properties, "application.process.id")
                        .and_then(|value| value.parse::<u32>().ok())
                        == Some(process.pid)
                    && property(properties, "translator.owner") == Some("true")
                    && property(properties, "translator.test_profile") == Some("human_round_trip")
                    && property(properties, "translator.self_test_session")
                        == Some(expected_session.as_str())
                    && property(properties, "target.object") == Some(expected_target)
                    && property(properties, "media.name") == Some("translator-virtual-peer");
                if !exact {
                    return None;
                }
                let object_serial = property(properties, "object.serial")?.parse::<u64>().ok()?;
                Some(VirtualPeerCapability {
                    session_id,
                    stream_id: stream.index,
                    object_serial,
                    process,
                    process_binary: "pacat".to_owned(),
                })
            })
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(VirtualPeerDiscoveryError::new(
                VirtualPeerDiscoveryErrorCode::NoExactStream,
            ));
        }
        Ok(matching.pop().expect("exact length was checked"))
    }

    pub fn ensure_absent(
        &self,
        capability: &VirtualPeerCapability,
    ) -> Result<(), VirtualPeerDiscoveryError> {
        let exact_stream_present = self
            .sink_inputs()?
            .iter()
            .any(|stream| matches_capability(stream, capability));
        if exact_stream_present {
            return Err(VirtualPeerDiscoveryError::new(
                VirtualPeerDiscoveryErrorCode::ExactStreamStillPresent,
            ));
        }
        Ok(())
    }

    fn sink_inputs(&self) -> Result<Vec<RawSinkInput>, VirtualPeerDiscoveryError> {
        let arguments = vec![
            "--format=json".to_owned(),
            "list".to_owned(),
            "sink-inputs".to_owned(),
        ];
        let result = self
            .runner
            .run("pactl", &arguments)
            .map_err(|error| match error {
                CommandRunError::NotFound
                | CommandRunError::SpawnFailed
                | CommandRunError::TimedOut => {
                    VirtualPeerDiscoveryError::new(VirtualPeerDiscoveryErrorCode::DiscoveryFailed)
                }
            })?;
        if !result.is_success() {
            return Err(VirtualPeerDiscoveryError::new(
                VirtualPeerDiscoveryErrorCode::DiscoveryFailed,
            ));
        }
        serde_json::from_slice(result.stdout()).map_err(|_| {
            VirtualPeerDiscoveryError::new(VirtualPeerDiscoveryErrorCode::DiscoveryFailed)
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawSinkInput {
    index: u32,
    #[serde(default)]
    properties: HashMap<String, String>,
}

fn property<'a>(properties: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    properties
        .get(key)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn matches_capability(stream: &RawSinkInput, capability: &VirtualPeerCapability) -> bool {
    let properties = &stream.properties;
    stream.index == capability.stream_id
        && property(properties, "object.serial").and_then(|value| value.parse::<u64>().ok())
            == Some(capability.object_serial)
        && property(properties, "translator.self_test_session")
            .and_then(|value| Uuid::parse_str(value).ok())
            == Some(capability.session_id)
        && property(properties, "application.process.id")
            .and_then(|value| value.parse::<u32>().ok())
            == Some(capability.process.pid)
        && property(properties, "application.process.binary")
            == Some(capability.process_binary.as_str())
        && property(properties, "application.name") == Some("translator-virtual-peer")
        && property(properties, "media.name") == Some("translator-virtual-peer")
        && property(properties, "translator.owner") == Some("true")
        && property(properties, "translator.test_profile") == Some("human_round_trip")
}
