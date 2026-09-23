use crate::module_list::{PactlModule, parse_module_list};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{CommandRunError, CommandRunner, SystemCommandRunner};

pub const AEC_SOURCE: &str = "translator_aec_source";
pub const AEC_SINK: &str = "translator_aec_sink";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AecPhysicalPair {
    pub source: String,
    pub sink: String,
}

impl AecPhysicalPair {
    pub fn new(source: impl Into<String>, sink: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            sink: sink.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AecGraphState {
    pub module_id: u32,
    pub source_id: u32,
    pub sink_id: u32,
    pub pair: AecPhysicalPair,
    pub generation: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AecErrorCode {
    InvalidConfiguration,
    PactlMissing,
    ModuleLoadFailed,
    InspectionFailed,
    OwnershipMismatch,
    CleanupRefused,
    CleanupFailed,
    NotOwned,
    AlreadyOwned,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AecError {
    code: AecErrorCode,
}

impl AecError {
    fn new(code: AecErrorCode) -> Self {
        Self { code }
    }

    pub fn code(&self) -> AecErrorCode {
        self.code
    }
}

impl fmt::Display for AecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.code {
            AecErrorCode::InvalidConfiguration => "AEC configuration is invalid",
            AecErrorCode::PactlMissing => "Audio control command is unavailable",
            AecErrorCode::ModuleLoadFailed => "AEC module creation failed",
            AecErrorCode::InspectionFailed => "AEC graph inspection failed",
            AecErrorCode::OwnershipMismatch => "AEC graph ownership verification failed",
            AecErrorCode::CleanupRefused => "AEC cleanup ownership verification failed",
            AecErrorCode::CleanupFailed => "AEC cleanup failed",
            AecErrorCode::NotOwned => "No AEC module is owned by this runtime",
            AecErrorCode::AlreadyOwned => "An AEC module is already owned by this runtime",
        })
    }
}

impl std::error::Error for AecError {}

#[derive(Debug, Clone)]
struct OwnedAecModule {
    module_id: Option<u32>,
    unload_attempted: bool,
    acquisition_complete: bool,
}

#[derive(Debug, Deserialize)]
struct PactlEndpoint {
    index: u32,
    name: String,
    owner_module: u32,
    #[serde(default)]
    properties: HashMap<String, String>,
}

pub struct PulseAecGraph<R = SystemCommandRunner> {
    runner: R,
    pair: AecPhysicalPair,
    generation: String,
    owned: Option<OwnedAecModule>,
}

impl<R> PulseAecGraph<R>
where
    R: CommandRunner,
{
    pub fn new(
        runner: R,
        pair: AecPhysicalPair,
        generation: impl Into<String>,
    ) -> Result<Self, AecError> {
        let generation = generation.into();
        if !is_safe_name(&pair.source) || !is_safe_name(&pair.sink) || !is_safe_name(&generation) {
            return Err(AecError::new(AecErrorCode::InvalidConfiguration));
        }
        Ok(Self {
            runner,
            pair,
            generation,
            owned: None,
        })
    }

    pub fn load_owned(&mut self) -> Result<AecGraphState, AecError> {
        self.load_owned_until(Instant::now() + Duration::from_secs(2))
    }

    pub fn load_owned_until(&mut self, deadline: Instant) -> Result<AecGraphState, AecError> {
        if self.owned.is_some() {
            return Err(AecError::new(AecErrorCode::AlreadyOwned));
        }
        check_deadline(deadline, AecErrorCode::ModuleLoadFailed)?;
        // A timed-out process may already have created the module. Keep its
        // generation as a recovery obligation even without a returned ID.
        self.owned = Some(OwnedAecModule {
            module_id: None,
            unload_attempted: false,
            acquisition_complete: false,
        });
        let result = self.run(&self.load_args(), AecErrorCode::ModuleLoadFailed, deadline);
        self.owned
            .as_mut()
            .expect("load obligation exists")
            .acquisition_complete = true;
        let result = result?;
        let module_id = std::str::from_utf8(result.stdout())
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .ok_or_else(|| AecError::new(AecErrorCode::ModuleLoadFailed))?;
        self.owned
            .as_mut()
            .expect("load obligation exists")
            .module_id = Some(module_id);
        self.inspect_owned_until(deadline)
    }

    pub fn inspect_owned(&self) -> Result<AecGraphState, AecError> {
        self.inspect_owned_until(Instant::now() + Duration::from_secs(2))
    }

    /// Recover only a graph bearing the caller's exact persisted generation.
    /// Failed inspection remains a cleanup obligation, never permission to load.
    pub fn recover_owned_until(&mut self, deadline: Instant) -> Result<AecGraphState, AecError> {
        if self
            .owned
            .as_ref()
            .is_some_and(|owned| owned.module_id.is_some())
        {
            return Err(AecError::new(AecErrorCode::AlreadyOwned));
        }
        check_deadline(deadline, AecErrorCode::InspectionFailed)?;
        self.owned = Some(OwnedAecModule {
            module_id: None,
            unload_attempted: false,
            acquisition_complete: false,
        });
        let modules = self.inspect_modules(AecErrorCode::InspectionFailed, deadline)?;
        if !modules.values().any(|module| self.module_matches(module)) {
            return Err(AecError::new(AecErrorCode::NotOwned));
        }
        let module_id = self.unique_module(&modules, None, AecErrorCode::OwnershipMismatch)?;
        self.owned
            .as_mut()
            .expect("recovery obligation exists")
            .module_id = Some(module_id);
        self.inspect_endpoints_state(module_id, deadline)
    }

    pub fn inspect_owned_until(&self, deadline: Instant) -> Result<AecGraphState, AecError> {
        let owned = self
            .owned
            .as_ref()
            .ok_or_else(|| AecError::new(AecErrorCode::NotOwned))?;
        if owned.unload_attempted || owned.module_id.is_none() {
            return Err(AecError::new(AecErrorCode::OwnershipMismatch));
        }
        let modules = self.inspect_modules(AecErrorCode::InspectionFailed, deadline)?;
        let module_id =
            self.unique_module(&modules, owned.module_id, AecErrorCode::OwnershipMismatch)?;
        self.inspect_endpoints_state(module_id, deadline)
    }

    fn inspect_endpoints_state(
        &self,
        module_id: u32,
        deadline: Instant,
    ) -> Result<AecGraphState, AecError> {
        let source = self.exact_endpoint(
            self.inspect_endpoints("sources", deadline)?,
            AEC_SOURCE,
            module_id,
        )?;
        let sink = self.exact_endpoint(
            self.inspect_endpoints("sinks", deadline)?,
            AEC_SINK,
            module_id,
        )?;
        check_deadline(deadline, AecErrorCode::InspectionFailed)?;
        Ok(AecGraphState {
            module_id,
            source_id: source.index,
            sink_id: sink.index,
            pair: self.pair.clone(),
            generation: self.generation.clone(),
        })
    }

    pub fn cleanup_owned(&mut self) -> Result<Option<u32>, AecError> {
        self.cleanup_owned_until(Instant::now() + Duration::from_secs(2))
    }

    pub fn cleanup_owned_until(&mut self, deadline: Instant) -> Result<Option<u32>, AecError> {
        let Some(owned) = self.owned.as_ref() else {
            return Ok(None);
        };
        let modules = self.inspect_modules(AecErrorCode::CleanupRefused, deadline)?;
        if owned.module_id.is_none()
            && owned.acquisition_complete
            && !modules.values().any(|module| self.module_matches(module))
        {
            self.confirm_unknown_absence(&modules, deadline)?;
            self.owned = None;
            return Ok(None);
        }
        if owned.unload_attempted
            && let Some(module_id) = owned.module_id
            && !modules.contains_key(&module_id)
        {
            self.confirm_absence(&modules, module_id, deadline)?;
            self.owned = None;
            return Ok(Some(module_id));
        }
        let module_id =
            self.unique_module(&modules, owned.module_id, AecErrorCode::CleanupRefused)?;
        self.owned
            .as_mut()
            .expect("cleanup obligation exists")
            .module_id = Some(module_id);
        self.inspect_for_cleanup(module_id, deadline)?;
        self.owned = Some(OwnedAecModule {
            module_id: Some(module_id),
            unload_attempted: true,
            acquisition_complete: true,
        });
        self.run(
            &["unload-module".to_owned(), module_id.to_string()],
            AecErrorCode::CleanupFailed,
            deadline,
        )?;
        let modules = self.inspect_modules(AecErrorCode::CleanupFailed, deadline)?;
        self.confirm_absence(&modules, module_id, deadline)?;
        self.owned = None;
        Ok(Some(module_id))
    }

    fn inspect_for_cleanup(&self, module_id: u32, deadline: Instant) -> Result<(), AecError> {
        let sources = self
            .inspect_endpoints("sources", deadline)
            .map_err(|_| AecError::new(AecErrorCode::CleanupRefused))?;
        self.verify_cleanup_endpoints(sources, AEC_SOURCE, module_id)?;
        let sinks = self
            .inspect_endpoints("sinks", deadline)
            .map_err(|_| AecError::new(AecErrorCode::CleanupRefused))?;
        self.verify_cleanup_endpoints(sinks, AEC_SINK, module_id)
    }

    fn unique_module(
        &self,
        modules: &HashMap<u32, PactlModule>,
        expected: Option<u32>,
        failure: AecErrorCode,
    ) -> Result<u32, AecError> {
        let mut matching = modules
            .iter()
            .filter(|(_, module)| self.module_matches(module));
        let Some((&id, _)) = matching.next() else {
            return Err(AecError::new(failure));
        };
        if matching.next().is_some() || expected.is_some_and(|expected| expected != id) {
            return Err(AecError::new(failure));
        }
        Ok(id)
    }

    fn confirm_absence(
        &self,
        modules: &HashMap<u32, PactlModule>,
        module_id: u32,
        deadline: Instant,
    ) -> Result<(), AecError> {
        if modules.contains_key(&module_id)
            || modules.values().any(|module| self.module_matches(module))
        {
            return Err(AecError::new(AecErrorCode::CleanupFailed));
        }
        for (kind, name) in [("sources", AEC_SOURCE), ("sinks", AEC_SINK)] {
            let endpoints = self.inspect_endpoints(kind, deadline)?;
            if endpoints.iter().any(|endpoint| {
                endpoint.name == name
                    || endpoint.owner_module == module_id
                    || endpoint.properties.get("translator.generation") == Some(&self.generation)
            }) {
                return Err(AecError::new(AecErrorCode::CleanupFailed));
            }
        }
        check_deadline(deadline, AecErrorCode::CleanupFailed)
    }

    fn confirm_unknown_absence(
        &self,
        modules: &HashMap<u32, PactlModule>,
        deadline: Instant,
    ) -> Result<(), AecError> {
        if modules
            .values()
            .any(|module| self.module_matches(module) || self.module_has_generation(module))
        {
            return Err(AecError::new(AecErrorCode::CleanupRefused));
        }
        for (kind, name) in [("sources", AEC_SOURCE), ("sinks", AEC_SINK)] {
            let endpoints = self
                .inspect_endpoints(kind, deadline)
                .map_err(|_| AecError::new(AecErrorCode::CleanupRefused))?;
            if endpoints.iter().any(|endpoint| {
                endpoint.name == name
                    || endpoint.properties.get("translator.generation") == Some(&self.generation)
            }) {
                return Err(AecError::new(AecErrorCode::CleanupRefused));
            }
        }
        check_deadline(deadline, AecErrorCode::CleanupRefused)
    }

    fn inspect_modules(
        &self,
        failure_code: AecErrorCode,
        deadline: Instant,
    ) -> Result<HashMap<u32, PactlModule>, AecError> {
        let result = self.run(
            &["list".to_owned(), "short".to_owned(), "modules".to_owned()],
            failure_code,
            deadline,
        )?;
        parse_module_list(result.stdout()).map_err(|_| AecError::new(failure_code))
    }

    fn inspect_endpoints(
        &self,
        kind: &str,
        deadline: Instant,
    ) -> Result<Vec<PactlEndpoint>, AecError> {
        let result = self.run(
            &[
                "--format=json".to_owned(),
                "list".to_owned(),
                kind.to_owned(),
            ],
            AecErrorCode::InspectionFailed,
            deadline,
        )?;
        serde_json::from_slice(result.stdout())
            .map_err(|_| AecError::new(AecErrorCode::InspectionFailed))
    }

    fn exact_endpoint(
        &self,
        endpoints: Vec<PactlEndpoint>,
        name: &str,
        module_id: u32,
    ) -> Result<PactlEndpoint, AecError> {
        let matching: Vec<_> = endpoints
            .into_iter()
            .filter(|endpoint| endpoint.name == name)
            .collect();
        if matching.len() != 1 || !self.endpoint_matches(&matching[0], module_id) {
            return Err(AecError::new(AecErrorCode::OwnershipMismatch));
        }
        Ok(matching.into_iter().next().expect("length was checked"))
    }

    fn verify_cleanup_endpoints(
        &self,
        endpoints: Vec<PactlEndpoint>,
        name: &str,
        module_id: u32,
    ) -> Result<(), AecError> {
        let matching: Vec<_> = endpoints
            .iter()
            .filter(|endpoint| endpoint.name == name)
            .collect();
        if matching.len() <= 1
            && matching
                .first()
                .is_none_or(|endpoint| self.endpoint_matches(endpoint, module_id))
        {
            Ok(())
        } else {
            Err(AecError::new(AecErrorCode::CleanupRefused))
        }
    }

    fn module_matches(&self, module: &PactlModule) -> bool {
        module.name == "module-echo-cancel"
            && module
                .argument
                .split_whitespace()
                .eq(self.load_args()[2..].join(" ").split_whitespace())
    }

    fn module_has_generation(&self, module: &PactlModule) -> bool {
        let marker = format!("translator.generation={}", self.generation);
        argument_contains_generation_marker(&module.argument, &marker)
    }

    fn endpoint_matches(&self, endpoint: &PactlEndpoint, module_id: u32) -> bool {
        endpoint.owner_module == module_id
            && endpoint
                .properties
                .get("translator.owner")
                .map(String::as_str)
                == Some("true")
            && endpoint
                .properties
                .get("translator.generation")
                .map(String::as_str)
                == Some(self.generation.as_str())
    }

    fn load_args(&self) -> Vec<String> {
        vec![
            "load-module".to_owned(),
            "module-echo-cancel".to_owned(),
            format!("source_master={}", self.pair.source),
            format!("sink_master={}", self.pair.sink),
            format!("source_name={AEC_SOURCE}"),
            format!("sink_name={AEC_SINK}"),
            "rate=48000".to_owned(),
            "channels=1".to_owned(),
            "channel_map=mono".to_owned(),
            "aec_method=webrtc".to_owned(),
            format!(
                "source_properties='device.description=Translator_AEC_Source translator.owner=true translator.generation={}'",
                self.generation
            ),
            format!(
                "sink_properties='device.description=Translator_AEC_Sink translator.owner=true translator.generation={}'",
                self.generation
            ),
        ]
    }

    fn run(
        &self,
        args: &[String],
        failure_code: AecErrorCode,
        deadline: Instant,
    ) -> Result<crate::CommandResult, AecError> {
        check_deadline(deadline, failure_code)?;
        let result =
            self.runner
                .run_until("pactl", args, deadline)
                .map_err(|error| match error {
                    CommandRunError::NotFound => AecError::new(AecErrorCode::PactlMissing),
                    CommandRunError::SpawnFailed
                    | CommandRunError::TimedOut
                    | CommandRunError::DeadlineExpired => AecError::new(failure_code),
                })?;
        check_deadline(deadline, failure_code)?;
        if result.is_success() {
            Ok(result)
        } else {
            Err(AecError::new(failure_code))
        }
    }
}

fn check_deadline(deadline: Instant, failure: AecErrorCode) -> Result<(), AecError> {
    if Instant::now() >= deadline {
        Err(AecError::new(failure))
    } else {
        Ok(())
    }
}

fn is_safe_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
}

fn argument_contains_generation_marker(argument: &str, marker: &str) -> bool {
    let bytes = argument.as_bytes();
    argument.match_indices(marker).any(|(index, _)| {
        let end = index + marker.len();
        !bytes.get(end).is_some_and(|byte| {
            byte.is_ascii_alphanumeric() || *byte == b'.' || *byte == b'_' || *byte == b'-'
        })
    })
}
