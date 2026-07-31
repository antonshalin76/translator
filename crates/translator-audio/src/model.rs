use std::fmt;

use serde::{Deserialize, Serialize};

pub const MIC_OUT_SINK: &str = "translator_mic_out";
pub const VIRTUAL_MIC_SOURCE: &str = "translator_virtual_mic";
pub const REMOTE_IN_SINK: &str = "translator_remote_in";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointRole {
    MicOutSink,
    VirtualMicSource,
    RemoteInSink,
}

impl EndpointRole {
    pub(crate) const ORDER: [Self; 3] =
        [Self::MicOutSink, Self::VirtualMicSource, Self::RemoteInSink];

    pub const fn name(self) -> &'static str {
        match self {
            Self::MicOutSink => MIC_OUT_SINK,
            Self::VirtualMicSource => VIRTUAL_MIC_SOURCE,
            Self::RemoteInSink => REMOTE_IN_SINK,
        }
    }

    pub const fn kind(self) -> EndpointKind {
        match self {
            Self::MicOutSink | Self::RemoteInSink => EndpointKind::Sink,
            Self::VirtualMicSource => EndpointKind::Source,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKind {
    Sink,
    Source,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphHealth {
    Ready,
    Degraded,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioEndpointState {
    pub role: EndpointRole,
    pub kind: EndpointKind,
    pub name: String,
    pub endpoint_id: Option<u32>,
    pub owner_module_id: Option<u32>,
    pub available: bool,
    pub daemon_owned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioGraphState {
    pub health: GraphHealth,
    pub endpoints: Vec<AudioEndpointState>,
    pub owned_module_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_error: Option<AudioGraphSafeError>,
}

impl AudioGraphState {
    pub fn failed(error: &AudioGraphError) -> Self {
        Self {
            health: GraphHealth::Error,
            endpoints: EndpointRole::ORDER
                .into_iter()
                .map(|role| AudioEndpointState {
                    role,
                    kind: role.kind(),
                    name: role.name().to_owned(),
                    endpoint_id: None,
                    owner_module_id: None,
                    available: false,
                    daemon_owned: false,
                })
                .collect(),
            owned_module_ids: Vec::new(),
            safe_error: Some(error.safe_status().clone()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioGraphErrorCode {
    PactlMissing,
    GraphInspectionFailed,
    ModuleLoadFailed,
    DuplicateEndpoint,
    OwnershipJournalInvalid,
    OwnershipJournalIo,
    CleanupFailed,
    RollbackFailed,
    EndpointVerificationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioGraphSafeError {
    pub code: AudioGraphErrorCode,
    pub safe_message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioGraphError {
    status: AudioGraphSafeError,
}

impl AudioGraphError {
    pub(crate) fn new(code: AudioGraphErrorCode) -> Self {
        let (safe_message, retryable) = match code {
            AudioGraphErrorCode::PactlMissing => ("Audio control command is unavailable", false),
            AudioGraphErrorCode::GraphInspectionFailed => ("Audio graph inspection failed", true),
            AudioGraphErrorCode::ModuleLoadFailed => {
                ("Virtual audio endpoint creation failed", true)
            }
            AudioGraphErrorCode::DuplicateEndpoint => {
                ("Conflicting virtual audio endpoint exists", false)
            }
            AudioGraphErrorCode::OwnershipJournalInvalid => {
                ("Audio ownership journal is invalid", false)
            }
            AudioGraphErrorCode::OwnershipJournalIo => {
                ("Audio ownership journal is unavailable", true)
            }
            AudioGraphErrorCode::CleanupFailed => ("Virtual audio endpoint cleanup failed", true),
            AudioGraphErrorCode::RollbackFailed => ("Virtual audio endpoint rollback failed", true),
            AudioGraphErrorCode::EndpointVerificationFailed => {
                ("Virtual audio endpoint verification failed", true)
            }
        };
        Self {
            status: AudioGraphSafeError {
                code,
                safe_message: safe_message.to_owned(),
                retryable,
            },
        }
    }

    pub fn code(&self) -> AudioGraphErrorCode {
        self.status.code
    }

    pub fn safe_message(&self) -> &str {
        &self.status.safe_message
    }

    pub fn safe_status(&self) -> &AudioGraphSafeError {
        &self.status
    }
}

impl fmt::Display for AudioGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.safe_message())
    }
}

impl std::error::Error for AudioGraphError {}

pub trait AudioGraph {
    fn ensure_endpoints(&mut self) -> Result<AudioGraphState, AudioGraphError>;
    fn inspect(&self) -> Result<AudioGraphState, AudioGraphError>;
    fn cleanup_owned(&mut self) -> Result<Vec<u32>, AudioGraphError>;
}
