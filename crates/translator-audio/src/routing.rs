use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    CommandResult, CommandRunner, REMOTE_IN_SINK, SystemCommandRunner, VIRTUAL_MIC_SOURCE,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllowedApplication {
    Telegram,
    Firefox,
    Chromium,
    Chrome,
    Zoom,
    SyntheticValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingProfile {
    Production,
    SyntheticValidation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualPeerCapability {
    pub session_id: Uuid,
    pub stream_id: u32,
    pub object_serial: u64,
    pub process: ProcessIdentity,
    pub process_binary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_time_ticks: u64,
    pub executable_device: u64,
    pub executable_inode: u64,
}

impl ProcessIdentity {
    pub fn inspect(pid: u32) -> Option<Self> {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let after_name = stat.rsplit_once(')')?.1.trim();
        let start_time_ticks = after_name.split_whitespace().nth(19)?.parse().ok()?;
        let executable = fs::metadata(format!("/proc/{pid}/exe")).ok()?;
        Some(Self {
            pid,
            start_time_ticks,
            executable_device: executable.dev(),
            executable_inode: executable.ino(),
        })
    }

    fn is_current(self) -> bool {
        Self::inspect(self.pid) == Some(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteResolution {
    NoCandidate,
    AwaitingSelection,
    Routed,
    RouteRemoved,
    RouteConflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteMethod {
    PulseMove,
    PipeWireLinks,
}

impl Default for RouteMethod {
    fn default() -> Self {
        Self::PulseMove
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteCandidate {
    pub stream_id: u32,
    pub application: AllowedApplication,
    pub stable_app_key: String,
    pub application_name: String,
    pub process_binary: String,
    #[serde(default)]
    pub pipewire_node_name: Option<String>,
    pub media_role: Option<String>,
    pub description: Option<String>,
    pub current_sink_id: u32,
    pub current_sink_name: String,
    pub call_like: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceOutputState {
    pub stream_id: u32,
    pub source_id: u32,
    pub source_name: Option<String>,
    pub translator_owned: bool,
    pub capture_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncomingRoute {
    pub stream_id: u32,
    pub application: AllowedApplication,
    pub stable_app_key: String,
    pub original_sink_id: u32,
    pub original_sink_name: String,
    pub target_sink_name: String,
    #[serde(default)]
    pub route_method: RouteMethod,
    #[serde(default)]
    pub pipewire_node_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingState {
    pub candidates: Vec<RouteCandidate>,
    pub source_outputs: Vec<SourceOutputState>,
    pub conflicting_stream_ids: Vec<u32>,
    pub active_route: Option<IncomingRoute>,
    pub resolution: RouteResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingErrorCode {
    DiscoveryFailed,
    InvalidManualOverride,
    MoveFailed,
    RestoreFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingSafeError {
    pub code: RoutingErrorCode,
    pub safe_message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingError {
    status: RoutingSafeError,
}

impl RoutingError {
    fn new(code: RoutingErrorCode) -> Self {
        let (safe_message, retryable) = match code {
            RoutingErrorCode::DiscoveryFailed => ("Audio route discovery failed", true),
            RoutingErrorCode::InvalidManualOverride => {
                ("Selected audio route is unavailable", false)
            }
            RoutingErrorCode::MoveFailed => ("Selected audio route could not be activated", true),
            RoutingErrorCode::RestoreFailed => ("Previous audio route could not be restored", true),
        };
        Self {
            status: RoutingSafeError {
                code,
                safe_message: safe_message.to_owned(),
                retryable,
            },
        }
    }

    pub fn code(&self) -> RoutingErrorCode {
        self.status.code
    }

    pub fn safe_status(&self) -> &RoutingSafeError {
        &self.status
    }
}

impl fmt::Display for RoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.status.safe_message)
    }
}

impl std::error::Error for RoutingError {}

pub trait RoutingWatcher {
    fn inspect(&self) -> Result<RoutingState, RoutingError>;
    fn reconcile(&mut self, manual_stream_id: Option<u32>) -> Result<RoutingState, RoutingError>;
    fn restore_active(&mut self) -> Result<RoutingState, RoutingError>;
    fn active_route(&self) -> Option<&IncomingRoute>;
}

pub struct PulseRoutingWatcher<R = SystemCommandRunner> {
    runner: R,
    profile: RoutingProfile,
    active_route: Option<IncomingRoute>,
    selected_app_key: Option<String>,
    virtual_peer_capability: Option<VirtualPeerCapability>,
    route_journal_path: Option<PathBuf>,
    route_recovery_pending: bool,
}

impl<R> PulseRoutingWatcher<R>
where
    R: CommandRunner,
{
    pub fn new(runner: R, profile: RoutingProfile) -> Self {
        Self {
            runner,
            profile,
            active_route: None,
            selected_app_key: None,
            virtual_peer_capability: None,
            route_journal_path: None,
            route_recovery_pending: false,
        }
    }

    pub fn new_with_route_journal(
        runner: R,
        profile: RoutingProfile,
        route_journal_path: PathBuf,
    ) -> Self {
        let active_route = load_active_route(&route_journal_path);
        let selected_app_key = active_route
            .as_ref()
            .map(|route| route.stable_app_key.clone());
        let route_recovery_pending = active_route.is_some();
        Self {
            runner,
            profile,
            active_route,
            selected_app_key,
            virtual_peer_capability: None,
            route_journal_path: Some(route_journal_path),
            route_recovery_pending,
        }
    }

    pub fn route_virtual_peer(
        &mut self,
        capability: VirtualPeerCapability,
    ) -> Result<RoutingState, RoutingError> {
        self.virtual_peer_capability = Some(capability.clone());
        let result = self
            .reconcile(Some(capability.stream_id))
            .and_then(|_| self.inspect())
            .and_then(|state| {
                let verified = state.candidates.iter().any(|candidate| {
                    candidate.stream_id == capability.stream_id
                        && candidate.current_sink_name == REMOTE_IN_SINK
                }) && state.conflicting_stream_ids.is_empty();
                if verified {
                    Ok(state)
                } else {
                    Err(RoutingError::new(RoutingErrorCode::MoveFailed))
                }
            });
        if result.is_err() {
            self.virtual_peer_capability = None;
            self.route_recovery_pending = false;
        }
        result
    }

    pub fn restore_virtual_peer(&mut self) -> Result<RoutingState, RoutingError> {
        let restored = self.restore_active();
        if restored.is_ok() {
            self.virtual_peer_capability = None;
        }
        restored
    }

    pub fn validate_virtual_peer_route(
        &self,
        capability: &VirtualPeerCapability,
        expected_target: &str,
    ) -> Result<RoutingState, RoutingError> {
        if expected_target != REMOTE_IN_SINK
            || self.virtual_peer_capability.as_ref() != Some(capability)
            || !capability.process.is_current()
        {
            return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
        }
        let state = self.inspect()?;
        let active = self.active_route.as_ref();
        let candidate = state.candidates.iter().find(|candidate| {
            candidate.stream_id == capability.stream_id
                && candidate.application == AllowedApplication::SyntheticValidation
                && candidate.current_sink_name == expected_target
        });
        let verified = state.resolution == RouteResolution::Routed
            && state.conflicting_stream_ids.is_empty()
            && state.active_route.as_ref() == active
            && active.is_some_and(|route| {
                route.stream_id == capability.stream_id
                    && route.application == AllowedApplication::SyntheticValidation
                    && route.target_sink_name == expected_target
            })
            && candidate.is_some();
        if verified {
            Ok(state)
        } else {
            Err(RoutingError::new(RoutingErrorCode::MoveFailed))
        }
    }

    fn discover(&self) -> Result<RoutingSnapshot, RoutingError> {
        let sink_inputs: Vec<RawSinkInput> =
            self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let source_outputs: Vec<RawSourceOutput> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let sources: Vec<RawNode> = self.run_json(&["--format=json", "list", "sources"])?;
        let sinks: Vec<RawNode> = self.run_json(&["--format=json", "list", "sinks"])?;
        let fallback_restore_source_name = preferred_restore_source(&sources);
        let fallback_restore_sink_name = preferred_restore_sink(&sinks);
        let source_names: HashMap<_, _> = sources
            .into_iter()
            .map(|source| (source.index, source.name))
            .collect();
        let sink_names: HashMap<_, _> = sinks
            .into_iter()
            .map(|sink| (sink.index, sink.name))
            .collect();

        let mut candidates = Vec::new();
        let mut conflicting_stream_ids = Vec::new();
        let mut stale_remote_candidates = Vec::new();
        for stream in sink_inputs {
            let stream_id = stream.index;
            let on_remote_in = sink_names
                .get(&stream.sink)
                .is_some_and(|name| name == REMOTE_IN_SINK);
            let candidate = self.route_candidate(stream, &sink_names);
            if on_remote_in {
                if let Some(candidate) = candidate {
                    if self
                        .active_route
                        .as_ref()
                        .is_some_and(|active| same_route_identity(active, &candidate))
                    {
                        candidates.push(candidate);
                    } else {
                        stale_remote_candidates.push(candidate);
                        conflicting_stream_ids.push(stream_id);
                    }
                } else {
                    conflicting_stream_ids.push(stream_id);
                }
            } else if let Some(candidate) = candidate {
                candidates.push(candidate);
            }
        }
        let mut source_output_routes = Vec::new();
        let source_outputs = source_outputs
            .into_iter()
            .map(|stream| {
                let source_name = source_names.get(&stream.source).cloned();
                let translator_owned = is_translator_owned(&stream.properties, self.profile);
                let physical_source = source_name.as_deref().is_some_and(safe_capture_source_name);
                if !translator_owned
                    && let Some(source_name) = source_name.as_deref()
                    && let Some(candidate) =
                        self.source_output_route_candidate(&stream, source_name)
                {
                    source_output_routes.push(candidate);
                }
                SourceOutputState {
                    stream_id: stream.index,
                    source_id: stream.source,
                    source_name,
                    translator_owned,
                    capture_allowed: physical_source && !translator_owned,
                }
            })
            .collect();
        Ok(RoutingSnapshot {
            candidates,
            source_outputs,
            conflicting_stream_ids,
            stale_remote_candidates,
            source_output_routes,
            fallback_restore_source_name,
            fallback_restore_sink_name,
        })
    }

    fn route_candidate(
        &self,
        stream: RawSinkInput,
        sink_names: &HashMap<u32, String>,
    ) -> Option<RouteCandidate> {
        let virtual_peer = self.is_authorized_virtual_peer(&stream);
        if !virtual_peer && is_translator_owned(&stream.properties, self.profile) {
            return None;
        }
        let application_name = property(&stream.properties, "application.name").unwrap_or_default();
        let process_binary =
            property(&stream.properties, "application.process.binary").unwrap_or_default();
        let pipewire_node_name = property(&stream.properties, "node.name")
            .and_then(|value| pipewire_node_name(&value).map(str::to_owned));
        let application = if virtual_peer {
            AllowedApplication::SyntheticValidation
        } else {
            classify_application(&application_name, &process_binary, self.profile)?
        };
        let current_sink_name = sink_names.get(&stream.sink)?.clone();
        let media_role = property(&stream.properties, "media.role");
        let description = property(&stream.properties, "media.name")
            .or_else(|| property(&stream.properties, "stream.description"));
        let call_like = is_call_like(application, media_role.as_deref(), description.as_deref());
        Some(RouteCandidate {
            stream_id: stream.index,
            application,
            stable_app_key: stable_app_key(application, &process_binary),
            application_name,
            process_binary,
            pipewire_node_name,
            media_role,
            description,
            current_sink_id: stream.sink,
            current_sink_name,
            call_like,
        })
    }

    fn source_output_route_candidate(
        &self,
        stream: &RawSourceOutput,
        source_name: &str,
    ) -> Option<SourceOutputRouteCandidate> {
        let physical_source = safe_capture_source_name(source_name);
        let virtual_source = source_name == VIRTUAL_MIC_SOURCE;
        if !physical_source && !virtual_source {
            return None;
        }
        let application_name = property(&stream.properties, "application.name").unwrap_or_default();
        let process_binary =
            property(&stream.properties, "application.process.binary").unwrap_or_default();
        let application = classify_application(&application_name, &process_binary, self.profile)?;
        Some(SourceOutputRouteCandidate {
            stream_id: stream.index,
            application,
            stable_app_key: stable_app_key(application, &process_binary),
            source_name: source_name.to_owned(),
            physical_source,
            virtual_source,
        })
    }

    fn is_authorized_virtual_peer(&self, stream: &RawSinkInput) -> bool {
        let Some(capability) = self.virtual_peer_capability.as_ref() else {
            return false;
        };
        stream.index == capability.stream_id
            && property(&stream.properties, "object.serial")
                .and_then(|value| value.parse::<u64>().ok())
                == Some(capability.object_serial)
            && property(&stream.properties, "translator.owner") == Some("true".to_owned())
            && property(&stream.properties, "application.name")
                == Some("translator-virtual-peer".to_owned())
            && property(&stream.properties, "media.name")
                == Some("translator-virtual-peer".to_owned())
            && property(&stream.properties, "translator.test_profile")
                == Some("human_round_trip".to_owned())
            && property(&stream.properties, "translator.self_test_session")
                == Some(capability.session_id.to_string())
            && property(&stream.properties, "application.process.id")
                .and_then(|value| value.parse::<u32>().ok())
                == Some(capability.process.pid)
            && property(&stream.properties, "application.process.binary")
                == Some(capability.process_binary.clone())
            && capability.process.is_current()
    }

    fn route_candidate_stream(
        &mut self,
        candidate: &RouteCandidate,
        source_output_routes: &[SourceOutputRouteCandidate],
        fallback_restore_source_name: Option<&str>,
    ) -> Result<(), RoutingError> {
        if let Some(active) = self.active_route.clone() {
            if same_route_identity(&active, candidate) {
                self.persist_active_route(&active, RoutingErrorCode::MoveFailed)?;
                self.route_matching_source_outputs(&active, source_output_routes)?;
                self.route_recovery_pending = false;
                return Ok(());
            }
            if active.stream_id == candidate.stream_id {
                self.restore_matching_source_outputs(
                    &active,
                    source_output_routes,
                    fallback_restore_source_name,
                )?;
                self.active_route = None;
                self.clear_route_journal(RoutingErrorCode::MoveFailed)?;
                self.route_recovery_pending = false;
            } else {
                if self.restore_route(&active).is_err() {
                    return Err(RoutingError::new(RoutingErrorCode::RestoreFailed));
                }
                self.restore_matching_source_outputs(
                    &active,
                    source_output_routes,
                    fallback_restore_source_name,
                )?;
                self.active_route = None;
                self.clear_route_journal(RoutingErrorCode::RestoreFailed)?;
                self.route_recovery_pending = false;
            }
        }

        let route_method = if self
            .move_stream(
                candidate.stream_id,
                REMOTE_IN_SINK,
                RoutingErrorCode::MoveFailed,
            )
            .is_err()
        {
            if self.route_zoom_with_pipewire_links(candidate).is_err() {
                self.selected_app_key = None;
                return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
            }
            RouteMethod::PipeWireLinks
        } else {
            RouteMethod::PulseMove
        };
        self.selected_app_key = Some(candidate.stable_app_key.clone());
        let pipewire_node_name = if route_method == RouteMethod::PipeWireLinks {
            pipewire_route_node_name(candidate).map(str::to_owned)
        } else {
            None
        };
        let active_route = IncomingRoute {
            stream_id: candidate.stream_id,
            application: candidate.application,
            stable_app_key: candidate.stable_app_key.clone(),
            original_sink_id: candidate.current_sink_id,
            original_sink_name: candidate.current_sink_name.clone(),
            target_sink_name: REMOTE_IN_SINK.to_owned(),
            route_method,
            pipewire_node_name,
        };
        if let Err(error) = self.persist_active_route(&active_route, RoutingErrorCode::MoveFailed) {
            let _ = self.restore_route(&active_route);
            self.selected_app_key = None;
            return Err(error);
        }
        if let Err(error) = self.route_matching_source_outputs(&active_route, source_output_routes)
        {
            let _ = self.restore_route(&active_route);
            self.selected_app_key = None;
            self.clear_route_journal(RoutingErrorCode::RestoreFailed)?;
            return Err(error);
        }
        self.active_route = Some(active_route);
        self.route_recovery_pending = false;
        Ok(())
    }

    fn move_stream(
        &self,
        stream_id: u32,
        sink: &str,
        failure_code: RoutingErrorCode,
    ) -> Result<(), RoutingError> {
        let result = self
            .runner
            .run(
                "pactl",
                &[
                    "move-sink-input".to_owned(),
                    stream_id.to_string(),
                    sink.to_owned(),
                ],
            )
            .map_err(|_| RoutingError::new(failure_code))?;
        if result.is_success() {
            Ok(())
        } else {
            Err(RoutingError::new(failure_code))
        }
    }

    fn move_source_output(
        &self,
        stream_id: u32,
        source: &str,
        failure_code: RoutingErrorCode,
    ) -> Result<(), RoutingError> {
        let result = self
            .runner
            .run(
                "pactl",
                &[
                    "move-source-output".to_owned(),
                    stream_id.to_string(),
                    source.to_owned(),
                ],
            )
            .map_err(|_| RoutingError::new(failure_code))?;
        if result.is_success() {
            Ok(())
        } else {
            Err(RoutingError::new(failure_code))
        }
    }

    fn route_matching_source_outputs(
        &self,
        active: &IncomingRoute,
        source_output_routes: &[SourceOutputRouteCandidate],
    ) -> Result<(), RoutingError> {
        let mut moved: Vec<(u32, String)> = Vec::new();
        for output in source_output_routes.iter().filter(|output| {
            output.stable_app_key == active.stable_app_key && output.physical_source
        }) {
            if let Err(error) = self.move_source_output(
                output.stream_id,
                VIRTUAL_MIC_SOURCE,
                RoutingErrorCode::MoveFailed,
            ) {
                for (stream_id, source_name) in moved.into_iter().rev() {
                    let _ = self.move_source_output(
                        stream_id,
                        &source_name,
                        RoutingErrorCode::RestoreFailed,
                    );
                }
                return Err(error);
            }
            moved.push((output.stream_id, output.source_name.clone()));
        }
        Ok(())
    }

    fn restore_matching_source_outputs(
        &self,
        active: &IncomingRoute,
        source_output_routes: &[SourceOutputRouteCandidate],
        fallback_restore_source_name: Option<&str>,
    ) -> Result<(), RoutingError> {
        let matching_virtual: Vec<_> = source_output_routes
            .iter()
            .filter(|output| {
                output.stable_app_key == active.stable_app_key && output.virtual_source
            })
            .collect();
        if matching_virtual.is_empty() {
            return Ok(());
        }
        let Some(source_name) = fallback_restore_source_name else {
            return Err(RoutingError::new(RoutingErrorCode::RestoreFailed));
        };
        for output in matching_virtual {
            self.move_source_output(
                output.stream_id,
                source_name,
                RoutingErrorCode::RestoreFailed,
            )?;
        }
        Ok(())
    }

    fn route_zoom_with_pipewire_links(
        &self,
        candidate: &RouteCandidate,
    ) -> Result<(), RoutingError> {
        if candidate.application != AllowedApplication::Zoom {
            return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
        }
        let Some(node_name) = pipewire_route_node_name(candidate) else {
            return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
        };
        let Some(original_sink_name) = pipewire_node_name(&candidate.current_sink_name) else {
            return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
        };
        let remote_sink_name = REMOTE_IN_SINK;
        let zoom_fl = pipewire_port(node_name, "output_FL");
        let zoom_fr = pipewire_port(node_name, "output_FR");
        let original_fl = pipewire_port(original_sink_name, "playback_FL");
        let original_fr = pipewire_port(original_sink_name, "playback_FR");
        let remote_fl = pipewire_port(remote_sink_name, "playback_FL");
        let remote_fr = pipewire_port(remote_sink_name, "playback_FR");

        self.connect_pipewire_link(&zoom_fl, &remote_fl, RoutingErrorCode::MoveFailed)?;
        if self
            .connect_pipewire_link(&zoom_fr, &remote_fr, RoutingErrorCode::MoveFailed)
            .is_err()
        {
            let _ =
                self.disconnect_pipewire_link(&zoom_fl, &remote_fl, RoutingErrorCode::MoveFailed);
            return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
        }
        if self
            .disconnect_pipewire_link(&zoom_fl, &original_fl, RoutingErrorCode::MoveFailed)
            .is_err()
        {
            let _ =
                self.disconnect_pipewire_link(&zoom_fl, &remote_fl, RoutingErrorCode::MoveFailed);
            let _ =
                self.disconnect_pipewire_link(&zoom_fr, &remote_fr, RoutingErrorCode::MoveFailed);
            return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
        }
        if self
            .disconnect_pipewire_link(&zoom_fr, &original_fr, RoutingErrorCode::MoveFailed)
            .is_err()
        {
            let _ =
                self.connect_pipewire_link(&zoom_fl, &original_fl, RoutingErrorCode::MoveFailed);
            let _ =
                self.disconnect_pipewire_link(&zoom_fl, &remote_fl, RoutingErrorCode::MoveFailed);
            let _ =
                self.disconnect_pipewire_link(&zoom_fr, &remote_fr, RoutingErrorCode::MoveFailed);
            return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
        }
        Ok(())
    }

    fn restore_route(&self, active: &IncomingRoute) -> Result<(), RoutingError> {
        match active.route_method {
            RouteMethod::PulseMove => self.move_stream(
                active.stream_id,
                &active.original_sink_name,
                RoutingErrorCode::RestoreFailed,
            ),
            RouteMethod::PipeWireLinks => self.restore_zoom_pipewire_links(active),
        }
    }

    fn restore_zoom_pipewire_links(&self, active: &IncomingRoute) -> Result<(), RoutingError> {
        if active.application != AllowedApplication::Zoom {
            return Err(RoutingError::new(RoutingErrorCode::RestoreFailed));
        }
        let Some(node_name) = active
            .pipewire_node_name
            .as_deref()
            .and_then(pipewire_node_name)
        else {
            return Err(RoutingError::new(RoutingErrorCode::RestoreFailed));
        };
        let Some(original_sink_name) = pipewire_node_name(&active.original_sink_name) else {
            return Err(RoutingError::new(RoutingErrorCode::RestoreFailed));
        };
        let zoom_fl = pipewire_port(node_name, "output_FL");
        let zoom_fr = pipewire_port(node_name, "output_FR");
        let original_fl = pipewire_port(original_sink_name, "playback_FL");
        let original_fr = pipewire_port(original_sink_name, "playback_FR");
        let remote_fl = pipewire_port(REMOTE_IN_SINK, "playback_FL");
        let remote_fr = pipewire_port(REMOTE_IN_SINK, "playback_FR");

        self.connect_pipewire_link(&zoom_fl, &original_fl, RoutingErrorCode::RestoreFailed)?;
        if self
            .connect_pipewire_link(&zoom_fr, &original_fr, RoutingErrorCode::RestoreFailed)
            .is_err()
        {
            let _ = self.disconnect_pipewire_link(
                &zoom_fl,
                &original_fl,
                RoutingErrorCode::RestoreFailed,
            );
            return Err(RoutingError::new(RoutingErrorCode::RestoreFailed));
        }
        self.disconnect_pipewire_link(&zoom_fl, &remote_fl, RoutingErrorCode::RestoreFailed)?;
        self.disconnect_pipewire_link(&zoom_fr, &remote_fr, RoutingErrorCode::RestoreFailed)
    }

    fn connect_pipewire_link(
        &self,
        output: &str,
        input: &str,
        failure_code: RoutingErrorCode,
    ) -> Result<(), RoutingError> {
        self.run_pipewire_link(&[output.to_owned(), input.to_owned()], failure_code)
    }

    fn disconnect_pipewire_link(
        &self,
        output: &str,
        input: &str,
        failure_code: RoutingErrorCode,
    ) -> Result<(), RoutingError> {
        self.run_pipewire_link(
            &["-d".to_owned(), output.to_owned(), input.to_owned()],
            failure_code,
        )
    }

    fn run_pipewire_link(
        &self,
        args: &[String],
        failure_code: RoutingErrorCode,
    ) -> Result<(), RoutingError> {
        let result = self
            .runner
            .run("pw-link", args)
            .map_err(|_| RoutingError::new(failure_code))?;
        if result.is_success() {
            Ok(())
        } else {
            Err(RoutingError::new(failure_code))
        }
    }

    fn run_json<T>(&self, args: &[&str]) -> Result<T, RoutingError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let arguments: Vec<String> = args.iter().map(|value| (*value).to_owned()).collect();
        let result = self
            .runner
            .run("pactl", &arguments)
            .map_err(|_| RoutingError::new(RoutingErrorCode::DiscoveryFailed))?;
        parse_json(result)
    }

    fn state(
        &self,
        candidates: Vec<RouteCandidate>,
        source_outputs: Vec<SourceOutputState>,
        conflicting_stream_ids: Vec<u32>,
        resolution: RouteResolution,
    ) -> RoutingState {
        RoutingState {
            candidates,
            source_outputs,
            conflicting_stream_ids,
            active_route: self.active_route.clone(),
            resolution,
        }
    }

    fn persist_active_route(
        &self,
        active_route: &IncomingRoute,
        failure_code: RoutingErrorCode,
    ) -> Result<(), RoutingError> {
        let Some(path) = self.route_journal_path.as_ref() else {
            return Ok(());
        };
        save_active_route(path, active_route).map_err(|_| RoutingError::new(failure_code))
    }

    fn clear_route_journal(&self, failure_code: RoutingErrorCode) -> Result<(), RoutingError> {
        let Some(path) = self.route_journal_path.as_ref() else {
            return Ok(());
        };
        remove_route_journal(path).map_err(|_| RoutingError::new(failure_code))
    }

    fn restore_stale_remote_conflict(
        &mut self,
        conflicting_stream_ids: &[u32],
        stale_remote_candidates: &[RouteCandidate],
        fallback_restore_sink_name: Option<&str>,
    ) -> Result<Option<RoutingState>, RoutingError> {
        if conflicting_stream_ids.len() != 1 || stale_remote_candidates.len() != 1 {
            return Ok(None);
        }
        let candidate = &stale_remote_candidates[0];
        if candidate.call_like || candidate.stream_id != conflicting_stream_ids[0] {
            return Ok(None);
        }
        let Some(restore_sink) = fallback_restore_sink_name else {
            return Ok(None);
        };

        self.move_stream(
            candidate.stream_id,
            restore_sink,
            RoutingErrorCode::RestoreFailed,
        )?;
        self.selected_app_key = None;
        self.clear_route_journal(RoutingErrorCode::RestoreFailed)?;

        let RoutingSnapshot {
            candidates,
            source_outputs,
            conflicting_stream_ids,
            ..
        } = self.discover()?;
        let resolution = if !conflicting_stream_ids.is_empty() {
            RouteResolution::RouteConflict
        } else if candidates.is_empty() {
            RouteResolution::NoCandidate
        } else {
            RouteResolution::AwaitingSelection
        };
        Ok(Some(self.state(
            candidates,
            source_outputs,
            conflicting_stream_ids,
            resolution,
        )))
    }
}

impl<R> RoutingWatcher for PulseRoutingWatcher<R>
where
    R: CommandRunner,
{
    fn inspect(&self) -> Result<RoutingState, RoutingError> {
        let RoutingSnapshot {
            candidates,
            source_outputs,
            conflicting_stream_ids,
            ..
        } = self.discover()?;
        let resolution = if !conflicting_stream_ids.is_empty() {
            RouteResolution::RouteConflict
        } else if self.active_route.is_some() {
            RouteResolution::Routed
        } else if self.selected_app_key.is_some() {
            RouteResolution::RouteRemoved
        } else if candidates.is_empty() {
            RouteResolution::NoCandidate
        } else {
            RouteResolution::AwaitingSelection
        };
        Ok(self.state(
            candidates,
            source_outputs,
            conflicting_stream_ids,
            resolution,
        ))
    }

    fn reconcile(&mut self, manual_stream_id: Option<u32>) -> Result<RoutingState, RoutingError> {
        if manual_stream_id.is_none() && self.route_recovery_pending {
            return self.restore_active();
        }
        let RoutingSnapshot {
            candidates,
            source_outputs,
            conflicting_stream_ids,
            stale_remote_candidates,
            source_output_routes,
            fallback_restore_source_name,
            fallback_restore_sink_name,
        } = self.discover()?;

        if !conflicting_stream_ids.is_empty() {
            if manual_stream_id.is_none()
                && self.active_route.is_none()
                && let Some(restored) = self.restore_stale_remote_conflict(
                    &conflicting_stream_ids,
                    &stale_remote_candidates,
                    fallback_restore_sink_name.as_deref(),
                )?
            {
                return Ok(restored);
            }
            return Ok(self.state(
                candidates,
                source_outputs,
                conflicting_stream_ids,
                RouteResolution::RouteConflict,
            ));
        }

        if let Some(stream_id) = manual_stream_id {
            let candidate = candidates
                .iter()
                .find(|candidate| candidate.stream_id == stream_id)
                .cloned()
                .ok_or_else(|| RoutingError::new(RoutingErrorCode::InvalidManualOverride))?;
            self.route_candidate_stream(
                &candidate,
                &source_output_routes,
                fallback_restore_source_name.as_deref(),
            )?;
            return Ok(self.state(
                candidates,
                source_outputs,
                conflicting_stream_ids,
                RouteResolution::Routed,
            ));
        }

        if let Some(active) = self.active_route.clone() {
            if let Some(candidate) = candidates
                .iter()
                .find(|candidate| same_route_identity(&active, candidate))
            {
                if should_yield_browser_fallback(candidate, &source_output_routes)
                    && let Some(replacement) = single_call_like_replacement(&candidates, &active)
                {
                    self.route_candidate_stream(
                        &replacement,
                        &source_output_routes,
                        fallback_restore_source_name.as_deref(),
                    )?;
                    return Ok(self.state(
                        candidates,
                        source_outputs,
                        conflicting_stream_ids,
                        RouteResolution::Routed,
                    ));
                }
                if active.route_method == RouteMethod::PipeWireLinks {
                    self.persist_active_route(&active, RoutingErrorCode::MoveFailed)?;
                } else if candidate.current_sink_name != REMOTE_IN_SINK {
                    self.move_stream(
                        candidate.stream_id,
                        REMOTE_IN_SINK,
                        RoutingErrorCode::MoveFailed,
                    )?;
                    self.persist_active_route(&active, RoutingErrorCode::MoveFailed)?;
                }
                self.route_matching_source_outputs(&active, &source_output_routes)?;
                return Ok(self.state(
                    candidates,
                    source_outputs,
                    conflicting_stream_ids,
                    RouteResolution::Routed,
                ));
            }
            self.active_route = None;
            self.restore_matching_source_outputs(
                &active,
                &source_output_routes,
                fallback_restore_source_name.as_deref(),
            )?;
            self.clear_route_journal(RoutingErrorCode::MoveFailed)?;
            self.route_recovery_pending = false;
            let replacements: Vec<_> = candidates
                .iter()
                .filter(|candidate| {
                    candidate.stable_app_key == active.stable_app_key
                        && is_auto_route_candidate(candidate, &source_output_routes)
                })
                .cloned()
                .collect();
            if replacements.len() == 1 {
                self.route_candidate_stream(
                    &replacements[0],
                    &source_output_routes,
                    fallback_restore_source_name.as_deref(),
                )?;
                return Ok(self.state(
                    candidates,
                    source_outputs,
                    conflicting_stream_ids,
                    RouteResolution::Routed,
                ));
            }
            let reused_stream_id = candidates
                .iter()
                .any(|candidate| candidate.stream_id == active.stream_id);
            if !reused_stream_id {
                if let Some(candidate) =
                    single_auto_route_candidate(&candidates, &source_output_routes)
                {
                    self.route_candidate_stream(
                        &candidate,
                        &source_output_routes,
                        fallback_restore_source_name.as_deref(),
                    )?;
                    return Ok(self.state(
                        candidates,
                        source_outputs,
                        conflicting_stream_ids,
                        RouteResolution::Routed,
                    ));
                }
                self.selected_app_key = None;
            }
            return Ok(self.state(
                candidates,
                source_outputs,
                conflicting_stream_ids,
                RouteResolution::RouteRemoved,
            ));
        }

        if let Some(selected_app_key) = self.selected_app_key.clone() {
            let replacements: Vec<_> = candidates
                .iter()
                .filter(|candidate| {
                    candidate.stable_app_key == selected_app_key
                        && is_auto_route_candidate(candidate, &source_output_routes)
                })
                .cloned()
                .collect();
            if replacements.len() == 1 {
                self.route_candidate_stream(
                    &replacements[0],
                    &source_output_routes,
                    fallback_restore_source_name.as_deref(),
                )?;
                return Ok(self.state(
                    candidates,
                    source_outputs,
                    conflicting_stream_ids,
                    RouteResolution::Routed,
                ));
            }
            return Ok(self.state(
                candidates,
                source_outputs,
                conflicting_stream_ids,
                RouteResolution::RouteRemoved,
            ));
        }

        if let Some(candidate) = single_auto_route_candidate(&candidates, &source_output_routes) {
            self.route_candidate_stream(
                &candidate,
                &source_output_routes,
                fallback_restore_source_name.as_deref(),
            )?;
            return Ok(self.state(
                candidates,
                source_outputs,
                conflicting_stream_ids,
                RouteResolution::Routed,
            ));
        }
        let resolution = if candidates.is_empty() {
            RouteResolution::NoCandidate
        } else {
            RouteResolution::AwaitingSelection
        };
        Ok(self.state(
            candidates,
            source_outputs,
            conflicting_stream_ids,
            resolution,
        ))
    }

    fn restore_active(&mut self) -> Result<RoutingState, RoutingError> {
        let snapshot = self.discover()?;
        if let Some(active) = self.active_route.clone() {
            if let Some(candidate) = snapshot
                .candidates
                .iter()
                .find(|candidate| same_route_identity(&active, candidate))
            {
                match active.route_method {
                    RouteMethod::PulseMove if candidate.current_sink_name == REMOTE_IN_SINK => {
                        self.restore_route(&active)?;
                    }
                    RouteMethod::PipeWireLinks => {
                        self.restore_route(&active)?;
                    }
                    RouteMethod::PulseMove => {}
                }
            }
            self.restore_matching_source_outputs(
                &active,
                &snapshot.source_output_routes,
                snapshot.fallback_restore_source_name.as_deref(),
            )?;
        }
        self.active_route = None;
        self.selected_app_key = None;
        self.route_recovery_pending = false;
        self.clear_route_journal(RoutingErrorCode::RestoreFailed)?;
        let resolution = if snapshot.candidates.is_empty() {
            RouteResolution::NoCandidate
        } else {
            RouteResolution::AwaitingSelection
        };
        Ok(self.state(
            snapshot.candidates,
            snapshot.source_outputs,
            snapshot.conflicting_stream_ids,
            resolution,
        ))
    }

    fn active_route(&self) -> Option<&IncomingRoute> {
        self.active_route.as_ref()
    }
}

struct RoutingSnapshot {
    candidates: Vec<RouteCandidate>,
    source_outputs: Vec<SourceOutputState>,
    conflicting_stream_ids: Vec<u32>,
    stale_remote_candidates: Vec<RouteCandidate>,
    source_output_routes: Vec<SourceOutputRouteCandidate>,
    fallback_restore_source_name: Option<String>,
    fallback_restore_sink_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceOutputRouteCandidate {
    stream_id: u32,
    application: AllowedApplication,
    stable_app_key: String,
    source_name: String,
    physical_source: bool,
    virtual_source: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RouteJournal {
    schema_version: u8,
    active_route: IncomingRoute,
}

const ROUTE_JOURNAL_SCHEMA_VERSION: u8 = 1;

#[derive(Debug, Deserialize)]
struct RawSinkInput {
    index: u32,
    sink: u32,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawSourceOutput {
    index: u32,
    source: u32,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawNode {
    index: u32,
    name: String,
    #[serde(default)]
    state: Option<String>,
}

fn parse_json<T>(result: CommandResult) -> Result<T, RoutingError>
where
    T: for<'de> Deserialize<'de>,
{
    if !result.is_success() {
        return Err(RoutingError::new(RoutingErrorCode::DiscoveryFailed));
    }
    serde_json::from_slice(result.stdout())
        .map_err(|_| RoutingError::new(RoutingErrorCode::DiscoveryFailed))
}

pub fn default_route_journal_path() -> Result<PathBuf, RoutingError> {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .map(|path| path.join("translator/routes.json"))
        .ok_or_else(|| RoutingError::new(RoutingErrorCode::DiscoveryFailed))
}

fn load_active_route(path: &Path) -> Option<IncomingRoute> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.nlink() != 1 {
        return None;
    }
    let journal: RouteJournal = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    valid_route_journal(&journal).then_some(journal.active_route)
}

fn save_active_route(path: &Path, active_route: &IncomingRoute) -> Result<(), RoutingError> {
    let journal = RouteJournal {
        schema_version: ROUTE_JOURNAL_SCHEMA_VERSION,
        active_route: active_route.clone(),
    };
    if !valid_route_journal(&journal) {
        return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
    }
    let parent = path
        .parent()
        .ok_or_else(|| RoutingError::new(RoutingErrorCode::MoveFailed))?;
    fs::create_dir_all(parent).map_err(|_| RoutingError::new(RoutingErrorCode::MoveFailed))?;
    fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
        .map_err(|_| RoutingError::new(RoutingErrorCode::MoveFailed))?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.contains('/'))
        .ok_or_else(|| RoutingError::new(RoutingErrorCode::MoveFailed))?;
    let temporary = parent.join(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        Uuid::new_v4()
    ));
    let bytes = serde_json::to_vec(&journal)
        .map_err(|_| RoutingError::new(RoutingErrorCode::MoveFailed))?;
    let result = (|| -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        fs::File::open(parent)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
        return Err(RoutingError::new(RoutingErrorCode::MoveFailed));
    }
    Ok(())
}

fn remove_route_journal(path: &Path) -> Result<(), RoutingError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(RoutingError::new(RoutingErrorCode::RestoreFailed)),
    }
}

fn valid_route_journal(journal: &RouteJournal) -> bool {
    journal.schema_version == ROUTE_JOURNAL_SCHEMA_VERSION && valid_route(&journal.active_route)
}

fn valid_route(route: &IncomingRoute) -> bool {
    route.target_sink_name == REMOTE_IN_SINK
        && !route.stable_app_key.is_empty()
        && route.stable_app_key.len() <= 256
        && !route.original_sink_name.is_empty()
        && route.original_sink_name.len() <= 256
        && route.stream_id > 0
        && match route.route_method {
            RouteMethod::PulseMove => true,
            RouteMethod::PipeWireLinks => {
                route.application == AllowedApplication::Zoom
                    && route
                        .pipewire_node_name
                        .as_deref()
                        .and_then(pipewire_node_name)
                        .is_some()
            }
        }
}

fn property(properties: &HashMap<String, String>, key: &str) -> Option<String> {
    properties
        .get(key)
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn classify_application(
    application_name: &str,
    process_binary: &str,
    profile: RoutingProfile,
) -> Option<AllowedApplication> {
    let name = application_name.to_ascii_lowercase();
    let binary = process_binary.to_ascii_lowercase();
    let live_binary = binary.strip_suffix(" (deleted)").unwrap_or(&binary);
    if profile == RoutingProfile::SyntheticValidation && live_binary == "paplay" {
        return Some(AllowedApplication::SyntheticValidation);
    }
    if live_binary.contains("telegram") || name.contains("telegram desktop") {
        Some(AllowedApplication::Telegram)
    } else if live_binary == "firefox" || name.starts_with("firefox") {
        Some(AllowedApplication::Firefox)
    } else if live_binary == "chromium" || name.starts_with("chromium") {
        Some(AllowedApplication::Chromium)
    } else if matches!(live_binary, "google-chrome" | "chrome")
        || name.starts_with("google chrome")
        || name == "chrome"
    {
        Some(AllowedApplication::Chrome)
    } else if live_binary == "zoom"
        || live_binary == "zoomwebviewhost"
        || name == "zoom"
        || name == "zoom workplace"
    {
        Some(AllowedApplication::Zoom)
    } else {
        None
    }
}

fn is_call_like(
    application: AllowedApplication,
    media_role: Option<&str>,
    description: Option<&str>,
) -> bool {
    media_role.is_some_and(|role| {
        matches!(
            role.to_ascii_lowercase().as_str(),
            "communication" | "phone"
        )
    }) || description.is_some_and(|value| {
        let value = value.to_ascii_lowercase();
        ["webrtc", "voice", "call", "meet"]
            .iter()
            .any(|marker| value.contains(marker))
            || (application == AllowedApplication::Zoom && value == "playstream")
    })
}

fn stable_app_key(application: AllowedApplication, process_binary: &str) -> String {
    format!(
        "{application:?}:{}",
        process_binary.trim().to_ascii_lowercase()
    )
}

fn single_auto_route_candidate(
    candidates: &[RouteCandidate],
    source_output_routes: &[SourceOutputRouteCandidate],
) -> Option<RouteCandidate> {
    let call_like: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.call_like)
        .cloned()
        .collect();
    if !call_like.is_empty() {
        return (call_like.len() == 1).then(|| call_like[0].clone());
    }

    let duplex_browser: Vec<_> = candidates
        .iter()
        .filter(|candidate| is_browser_duplex_candidate(candidate, source_output_routes))
        .cloned()
        .collect();
    (duplex_browser.len() == 1).then(|| duplex_browser[0].clone())
}

fn is_auto_route_candidate(
    candidate: &RouteCandidate,
    source_output_routes: &[SourceOutputRouteCandidate],
) -> bool {
    candidate.call_like || is_browser_duplex_candidate(candidate, source_output_routes)
}

fn single_call_like_replacement(
    candidates: &[RouteCandidate],
    active: &IncomingRoute,
) -> Option<RouteCandidate> {
    let replacements: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.call_like && !same_route_identity(active, candidate))
        .cloned()
        .collect();
    (replacements.len() == 1).then(|| replacements[0].clone())
}

fn should_yield_browser_fallback(
    candidate: &RouteCandidate,
    source_output_routes: &[SourceOutputRouteCandidate],
) -> bool {
    !candidate.call_like && is_browser_duplex_candidate(candidate, source_output_routes)
}

fn is_browser_duplex_candidate(
    candidate: &RouteCandidate,
    source_output_routes: &[SourceOutputRouteCandidate],
) -> bool {
    matches!(
        candidate.application,
        AllowedApplication::Firefox | AllowedApplication::Chromium | AllowedApplication::Chrome
    ) && source_output_routes.iter().any(|output| {
        output.application == candidate.application
            && output.stable_app_key == candidate.stable_app_key
            && (output.physical_source || output.virtual_source)
    })
}

fn preferred_restore_sink(nodes: &[RawNode]) -> Option<String> {
    let physical: Vec<_> = nodes
        .iter()
        .filter(|node| safe_restore_sink_name(&node.name))
        .collect();
    let running: Vec<_> = physical
        .iter()
        .copied()
        .filter(|node| {
            node.state
                .as_deref()
                .is_some_and(|state| state.eq_ignore_ascii_case("running"))
        })
        .collect();
    if running.len() == 1 {
        Some(running[0].name.clone())
    } else if physical.len() == 1 {
        Some(physical[0].name.clone())
    } else {
        None
    }
}

fn preferred_restore_source(nodes: &[RawNode]) -> Option<String> {
    let physical: Vec<_> = nodes
        .iter()
        .filter(|node| safe_capture_source_name(&node.name))
        .collect();
    let running: Vec<_> = physical
        .iter()
        .copied()
        .filter(|node| {
            node.state
                .as_deref()
                .is_some_and(|state| state.eq_ignore_ascii_case("running"))
        })
        .collect();
    if running.len() == 1 {
        Some(running[0].name.clone())
    } else if physical.len() == 1 {
        Some(physical[0].name.clone())
    } else {
        None
    }
}

fn safe_restore_sink_name(name: &str) -> bool {
    name != REMOTE_IN_SINK && !name.starts_with("translator_") && pipewire_node_name(name).is_some()
}

fn safe_capture_source_name(name: &str) -> bool {
    !name.starts_with("translator_")
        && !name.ends_with(".monitor")
        && pipewire_node_name(name).is_some()
}

fn pipewire_route_node_name(candidate: &RouteCandidate) -> Option<&str> {
    if candidate.application == AllowedApplication::Zoom {
        candidate
            .pipewire_node_name
            .as_deref()
            .and_then(pipewire_node_name)
            .or_else(|| pipewire_node_name(&candidate.application_name))
    } else {
        None
    }
}

fn pipewire_node_name(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > 256
        || value.contains(':')
        || value.contains('\n')
        || value.contains('\r')
        || value.contains('\0')
    {
        None
    } else {
        Some(value)
    }
}

fn pipewire_port(node_name: &str, port_name: &str) -> String {
    format!("{node_name}:{port_name}")
}

fn same_route_identity(active: &IncomingRoute, candidate: &RouteCandidate) -> bool {
    active.stream_id == candidate.stream_id && active.stable_app_key == candidate.stable_app_key
}

fn is_translator_owned(properties: &HashMap<String, String>, profile: RoutingProfile) -> bool {
    if properties
        .get("translator.owner")
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
    {
        return true;
    }
    let application = property(properties, "application.name")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let binary = property(properties, "application.process.binary")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if profile == RoutingProfile::SyntheticValidation && binary == "paplay" {
        return false;
    }
    if profile == RoutingProfile::Production && matches!(binary.as_str(), "paplay" | "pw-play") {
        return true;
    }
    if application.starts_with("translator")
        || binary.starts_with("translator-")
        || ["translator-daemon", "translator-sidecar", "translator-ui"]
            .iter()
            .any(|name| application.contains(name) || binary.contains(name))
    {
        return true;
    }
    ["node.name", "media.name", "stream.description"]
        .iter()
        .filter_map(|key| properties.get(*key))
        .any(|value| value.to_ascii_lowercase().starts_with("translator"))
}
