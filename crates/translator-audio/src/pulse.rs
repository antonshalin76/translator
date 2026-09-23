use crate::module_list::{PactlModule, parse_module_list};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

use serde::Deserialize;
use uuid::Uuid;

use crate::command::{CommandResult, CommandRunError, CommandRunner, SystemCommandRunner};
use crate::journal::{JournalSession, JournalStore, OwnedModule, OwnershipJournal, check_deadline};
use crate::{
    AudioEndpointState, AudioGraph, AudioGraphError, AudioGraphErrorCode, AudioGraphState,
    EndpointKind, EndpointRole, GraphHealth,
};

#[derive(Debug, Deserialize)]
struct PactlEndpoint {
    index: u32,
    name: String,
    owner_module: u32,
}

#[derive(Debug, Clone)]
struct RawGraph {
    sinks: Vec<PactlEndpointSummary>,
    sources: Vec<PactlEndpointSummary>,
}

#[derive(Debug, Clone)]
struct PactlEndpointSummary {
    index: u32,
    name: String,
    owner_module: u32,
}

impl From<PactlEndpoint> for PactlEndpointSummary {
    fn from(endpoint: PactlEndpoint) -> Self {
        Self {
            index: endpoint.index,
            name: endpoint.name,
            owner_module: endpoint.owner_module,
        }
    }
}

impl RawGraph {
    fn endpoints_for(&self, role: EndpointRole) -> Vec<&PactlEndpointSummary> {
        let endpoints = match role.kind() {
            EndpointKind::Sink => &self.sinks,
            EndpointKind::Source => &self.sources,
        };
        endpoints
            .iter()
            .filter(|endpoint| endpoint.name == role.name())
            .collect()
    }

    fn has_any_required_endpoint(&self) -> bool {
        EndpointRole::ORDER
            .into_iter()
            .any(|role| !self.endpoints_for(role).is_empty())
    }

    fn has_duplicate_required_endpoint(&self) -> bool {
        EndpointRole::ORDER
            .into_iter()
            .any(|role| self.endpoints_for(role).len() > 1)
    }
}

pub struct PulseAudioGraph<R = SystemCommandRunner> {
    runner: R,
    journal: JournalStore,
    next_generation: String,
}

impl<R> PulseAudioGraph<R>
where
    R: CommandRunner,
{
    pub fn new(runner: R, journal_path: PathBuf) -> Self {
        Self::new_with_generation(runner, journal_path, Uuid::new_v4().to_string())
    }

    #[doc(hidden)]
    pub fn new_with_generation(runner: R, journal_path: PathBuf, next_generation: String) -> Self {
        Self {
            runner,
            journal: JournalStore::new(journal_path),
            next_generation,
        }
    }

    fn inspect_raw(&self, deadline: Instant) -> Result<RawGraph, AudioGraphError> {
        Ok(RawGraph {
            sinks: self.inspect_endpoint_kind("sinks", deadline)?,
            sources: self.inspect_endpoint_kind("sources", deadline)?,
        })
    }

    fn inspect_endpoint_kind(
        &self,
        kind: &str,
        deadline: Instant,
    ) -> Result<Vec<PactlEndpointSummary>, AudioGraphError> {
        let result = self.run_pactl(
            &["--format=json", "list", kind],
            AudioGraphErrorCode::GraphInspectionFailed,
            deadline,
        )?;
        let endpoints: Vec<PactlEndpoint> = serde_json::from_slice(result.stdout())
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::GraphInspectionFailed))?;
        check_deadline(deadline)?;
        Ok(endpoints.into_iter().map(Into::into).collect())
    }

    fn inspect_modules(
        &self,
        deadline: Instant,
    ) -> Result<HashMap<u32, PactlModule>, AudioGraphError> {
        let result = self.run_pactl(
            &["list", "short", "modules"],
            AudioGraphErrorCode::CleanupFailed,
            deadline,
        )?;
        let modules = parse_module_list(result.stdout())
            .map_err(|_| AudioGraphError::new(AudioGraphErrorCode::CleanupFailed))?;
        check_deadline(deadline)?;
        Ok(modules)
    }

    fn state_from(&self, raw: &RawGraph, journal: Option<&OwnershipJournal>) -> AudioGraphState {
        let endpoints: Vec<_> = EndpointRole::ORDER
            .into_iter()
            .map(|role| {
                let discovered = raw.endpoints_for(role);
                let endpoint = discovered.first().copied();
                let journal_module = journal.and_then(|value| value.module_for(role));
                AudioEndpointState {
                    role,
                    kind: role.kind(),
                    name: role.name().to_owned(),
                    endpoint_id: endpoint.map(|value| value.index),
                    owner_module_id: endpoint.map(|value| value.owner_module),
                    available: discovered.len() == 1,
                    daemon_owned: endpoint
                        .zip(journal_module)
                        .is_some_and(|(value, module_id)| value.owner_module == module_id),
                }
            })
            .collect();
        let health = if endpoints
            .iter()
            .all(|endpoint| endpoint.available && endpoint.daemon_owned)
        {
            GraphHealth::Ready
        } else {
            GraphHealth::Degraded
        };
        AudioGraphState {
            health,
            endpoints,
            owned_module_ids: journal
                .map(OwnershipJournal::module_ids)
                .unwrap_or_default(),
            safe_error: None,
        }
    }

    fn journal_matches_graph(&self, raw: &RawGraph, journal: &OwnershipJournal) -> bool {
        journal.modules.len() == EndpointRole::ORDER.len()
            && EndpointRole::ORDER.into_iter().all(|role| {
                let endpoints = raw.endpoints_for(role);
                endpoints.len() == 1
                    && journal
                        .module_for(role)
                        .is_some_and(|module_id| endpoints[0].owner_module == module_id)
            })
    }

    fn create_endpoints(
        &mut self,
        session: &JournalSession,
        deadline: Instant,
    ) -> Result<AudioGraphState, AudioGraphError> {
        let mut ownership = OwnershipJournal::empty(self.next_generation.clone());
        check_deadline(deadline)?;
        session.save(&ownership)?;
        for role in EndpointRole::ORDER {
            let module_id = match self.load_endpoint(role, &ownership.generation, deadline) {
                Ok(module_id) => module_id,
                Err(load_error) => {
                    check_deadline(deadline)?;
                    if load_error.code() == AudioGraphErrorCode::DeadlineExpired {
                        return Err(load_error);
                    }
                    self.rollback_new_modules(session, &mut ownership, deadline)?;
                    return Err(load_error);
                }
            };
            ownership.modules.push(OwnedModule { role, module_id });
            if session.save(&ownership).is_err() {
                check_deadline(deadline)?;
                self.rollback_new_modules(session, &mut ownership, deadline)?;
                return Err(AudioGraphError::new(
                    AudioGraphErrorCode::OwnershipJournalIo,
                ));
            }
            check_deadline(deadline)?;
        }

        let raw = match self.inspect_raw(deadline) {
            Ok(raw) => raw,
            Err(inspection_error) => {
                check_deadline(deadline)?;
                self.rollback_new_modules(session, &mut ownership, deadline)?;
                return Err(inspection_error);
            }
        };
        if !self.journal_matches_graph(&raw, &ownership) {
            self.rollback_new_modules(session, &mut ownership, deadline)?;
            return Err(AudioGraphError::new(
                AudioGraphErrorCode::EndpointVerificationFailed,
            ));
        }
        let state = self.state_from(&raw, Some(&ownership));
        check_deadline(deadline)?;
        Ok(state)
    }

    fn load_endpoint(
        &self,
        role: EndpointRole,
        generation: &str,
        deadline: Instant,
    ) -> Result<u32, AudioGraphError> {
        let args = load_args(role, generation);
        let result =
            self.run_pactl_owned(&args, AudioGraphErrorCode::ModuleLoadFailed, deadline)?;
        let module_id = std::str::from_utf8(result.stdout())
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .ok_or_else(|| AudioGraphError::new(AudioGraphErrorCode::ModuleLoadFailed))?;
        Ok(module_id)
    }

    fn rollback_new_modules(
        &mut self,
        session: &JournalSession,
        ownership: &mut OwnershipJournal,
        deadline: Instant,
    ) -> Result<(), AudioGraphError> {
        while let Some(module) = ownership.modules.last().cloned() {
            let modules = self
                .inspect_modules(deadline)
                .map_err(|error| stage_error(error, AudioGraphErrorCode::RollbackFailed))?;
            match modules.get(&module.module_id) {
                Some(discovered)
                    if module_matches(module.role, discovered, &ownership.generation) =>
                {
                    if let Err(error) = self.unload_module(module.module_id, deadline) {
                        session.save(ownership)?;
                        return Err(stage_error(error, AudioGraphErrorCode::RollbackFailed));
                    }
                }
                Some(_) => {
                    session.save(ownership)?;
                    return Err(AudioGraphError::new(AudioGraphErrorCode::RollbackFailed));
                }
                None => {}
            }
            check_deadline(deadline)?;
            ownership.modules.pop();
            session.save(ownership)?;
        }
        check_deadline(deadline)?;
        session.save(ownership)
    }

    fn reconcile_ownership(
        &self,
        raw: &RawGraph,
        journal: &OwnershipJournal,
        modules: &HashMap<u32, PactlModule>,
        failure_code: AudioGraphErrorCode,
    ) -> Result<OwnershipJournal, AudioGraphError> {
        let mut reconciled = OwnershipJournal::empty(journal.generation.clone());

        for owned in &journal.modules {
            match modules.get(&owned.module_id) {
                Some(module) if module_matches(owned.role, module, &journal.generation) => {
                    reconciled.modules.push(owned.clone());
                }
                Some(_) => return Err(AudioGraphError::new(failure_code)),
                None => {}
            }
        }

        for module in modules.values() {
            if module.argument.contains(&journal.generation)
                && !EndpointRole::ORDER
                    .into_iter()
                    .any(|role| module_matches(role, module, &journal.generation))
            {
                return Err(AudioGraphError::new(failure_code));
            }
        }

        for role in EndpointRole::ORDER {
            let matching_ids: Vec<_> = modules
                .iter()
                .filter_map(|(module_id, module)| {
                    module_matches(role, module, &journal.generation).then_some(*module_id)
                })
                .collect();
            if matching_ids.len() > 1 {
                return Err(AudioGraphError::new(failure_code));
            }
            let Some(module_id) = matching_ids.first().copied() else {
                continue;
            };
            match reconciled.module_for(role) {
                Some(journal_id) if journal_id == module_id => {}
                Some(_) => return Err(AudioGraphError::new(failure_code)),
                None => reconciled.modules.push(OwnedModule { role, module_id }),
            }
        }

        for role in EndpointRole::ORDER {
            let discovered = raw.endpoints_for(role);
            let Some(endpoint) = discovered.first() else {
                continue;
            };
            let module = modules
                .get(&endpoint.owner_module)
                .ok_or_else(|| AudioGraphError::new(failure_code))?;
            if !module_matches(role, module, &journal.generation) {
                return Err(AudioGraphError::new(failure_code));
            }
            match reconciled.module_for(role) {
                Some(module_id) if module_id == endpoint.owner_module => {}
                Some(_) => return Err(AudioGraphError::new(failure_code)),
                None => reconciled.modules.push(OwnedModule {
                    role,
                    module_id: endpoint.owner_module,
                }),
            }
        }

        reconciled
            .modules
            .sort_by_key(|module| role_index(module.role));
        reconciled.validate()?;
        Ok(reconciled)
    }

    fn cleanup_reconciled(
        &mut self,
        session: &JournalSession,
        journal: &mut OwnershipJournal,
        deadline: Instant,
    ) -> Result<Vec<u32>, AudioGraphError> {
        let mut unloaded = Vec::new();
        while let Some(module) = journal.modules.last().cloned() {
            let modules = self.inspect_modules(deadline)?;
            match modules.get(&module.module_id) {
                Some(discovered)
                    if module_matches(module.role, discovered, &journal.generation) =>
                {
                    self.unload_module(module.module_id, deadline)?;
                    unloaded.push(module.module_id);
                }
                Some(_) => {
                    return Err(AudioGraphError::new(AudioGraphErrorCode::CleanupFailed));
                }
                None => {}
            }
            check_deadline(deadline)?;
            journal.modules.pop();
            if journal.modules.is_empty() {
                session.remove()?;
            } else {
                session.save(journal)?;
            }
        }
        check_deadline(deadline)?;
        session.remove()?;
        check_deadline(deadline)?;
        Ok(unloaded)
    }

    fn unload_module(&self, module_id: u32, deadline: Instant) -> Result<(), AudioGraphError> {
        self.run_pactl_owned(
            &["unload-module".to_owned(), module_id.to_string()],
            AudioGraphErrorCode::CleanupFailed,
            deadline,
        )?;
        check_deadline(deadline)
    }

    fn run_pactl(
        &self,
        args: &[&str],
        failure_code: AudioGraphErrorCode,
        deadline: Instant,
    ) -> Result<CommandResult, AudioGraphError> {
        let owned: Vec<String> = args.iter().map(|value| (*value).to_owned()).collect();
        let result = self.run_pactl_owned(&owned, failure_code, deadline)?;
        check_deadline(deadline)?;
        Ok(result)
    }

    fn run_pactl_owned(
        &self,
        args: &[String],
        failure_code: AudioGraphErrorCode,
        deadline: Instant,
    ) -> Result<CommandResult, AudioGraphError> {
        check_deadline(deadline)?;
        let result =
            self.runner
                .run_until("pactl", args, deadline)
                .map_err(|error| match error {
                    CommandRunError::NotFound => {
                        AudioGraphError::new(AudioGraphErrorCode::PactlMissing)
                    }
                    CommandRunError::SpawnFailed | CommandRunError::TimedOut => {
                        AudioGraphError::new(failure_code)
                    }
                    CommandRunError::DeadlineExpired => {
                        AudioGraphError::new(AudioGraphErrorCode::DeadlineExpired)
                    }
                })?;
        if result.is_success() {
            Ok(result)
        } else {
            Err(AudioGraphError::new(failure_code))
        }
    }
}

impl<R> AudioGraph for PulseAudioGraph<R>
where
    R: CommandRunner,
{
    fn ensure_endpoints_until(
        &mut self,
        deadline: Instant,
    ) -> Result<AudioGraphState, AudioGraphError> {
        let session = self.journal.lock_until(deadline)?;
        let journal = session.load_until(deadline)?;
        let raw = self.inspect_raw(deadline)?;
        if raw.has_duplicate_required_endpoint() {
            return Err(AudioGraphError::new(AudioGraphErrorCode::DuplicateEndpoint));
        }

        match journal {
            None => {
                if raw.has_any_required_endpoint() {
                    return Err(AudioGraphError::new(AudioGraphErrorCode::DuplicateEndpoint));
                }
                self.create_endpoints(&session, deadline)
            }
            Some(journal) => {
                let modules = self.inspect_modules(deadline)?;
                let mut reconciled = self.reconcile_ownership(
                    &raw,
                    &journal,
                    &modules,
                    AudioGraphErrorCode::DuplicateEndpoint,
                )?;
                check_deadline(deadline)?;
                if reconciled != journal {
                    session.save(&reconciled)?;
                }
                if self.journal_matches_graph(&raw, &reconciled) {
                    let state = self.state_from(&raw, Some(&reconciled));
                    check_deadline(deadline)?;
                    return Ok(state);
                }

                self.cleanup_reconciled(&session, &mut reconciled, deadline)?;
                let cleaned = self.inspect_raw(deadline)?;
                if cleaned.has_any_required_endpoint() {
                    return Err(AudioGraphError::new(AudioGraphErrorCode::DuplicateEndpoint));
                }
                self.create_endpoints(&session, deadline)
            }
        }
    }

    fn inspect_until(&self, deadline: Instant) -> Result<AudioGraphState, AudioGraphError> {
        let session = self.journal.read_only_until(deadline)?;
        let journal = session.load_until(deadline)?;
        let raw = self.inspect_raw(deadline)?;
        if raw.has_duplicate_required_endpoint() {
            return Err(AudioGraphError::new(AudioGraphErrorCode::DuplicateEndpoint));
        }
        let Some(journal) = journal else {
            let state = self.state_from(&raw, None);
            check_deadline(deadline)?;
            return Ok(state);
        };
        let modules = self.inspect_modules(deadline)?;
        let reconciled = self.reconcile_ownership(
            &raw,
            &journal,
            &modules,
            AudioGraphErrorCode::GraphInspectionFailed,
        )?;
        let state = self.state_from(&raw, Some(&reconciled));
        check_deadline(deadline)?;
        Ok(state)
    }

    fn cleanup_owned_until(&mut self, deadline: Instant) -> Result<Vec<u32>, AudioGraphError> {
        let session = self.journal.lock_until(deadline)?;
        let Some(journal) = session.load_until(deadline)? else {
            return Ok(Vec::new());
        };
        let raw = self.inspect_raw(deadline)?;
        if raw.has_duplicate_required_endpoint() {
            return Err(AudioGraphError::new(AudioGraphErrorCode::CleanupFailed));
        }
        let modules = self.inspect_modules(deadline)?;
        let mut reconciled =
            self.reconcile_ownership(&raw, &journal, &modules, AudioGraphErrorCode::CleanupFailed)?;
        check_deadline(deadline)?;
        if reconciled != journal {
            session.save(&reconciled)?;
        }
        self.cleanup_reconciled(&session, &mut reconciled, deadline)
    }
}

fn stage_error(error: AudioGraphError, failure_code: AudioGraphErrorCode) -> AudioGraphError {
    if error.code() == AudioGraphErrorCode::DeadlineExpired {
        error
    } else {
        AudioGraphError::new(failure_code)
    }
}

pub fn default_journal_path() -> Result<PathBuf, AudioGraphError> {
    env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .map(|path| path.join("translator/audio-modules.json"))
        .ok_or_else(|| AudioGraphError::new(AudioGraphErrorCode::OwnershipJournalIo))
}

fn load_args(role: EndpointRole, generation: &str) -> Vec<String> {
    let (values, property): (&[&str], String) = match role {
        EndpointRole::MicOutSink => (
            &[
                "load-module",
                "module-null-sink",
                "sink_name=translator_mic_out",
                "rate=48000",
                "channels=1",
                "channel_map=mono",
            ],
            format!(
                "sink_properties=\"device.description=Translator_Mic_Out translator.owner=true translator.generation={generation}\""
            ),
        ),
        EndpointRole::VirtualMicSource => (
            &[
                "load-module",
                "module-remap-source",
                "master=translator_mic_out.monitor",
                "source_name=translator_virtual_mic",
                "channels=1",
                "channel_map=mono",
                "remix=no",
            ],
            format!(
                "source_properties=\"device.description=Translator_Virtual_Mic translator.owner=true translator.generation={generation}\""
            ),
        ),
        EndpointRole::RemoteInSink => (
            &[
                "load-module",
                "module-null-sink",
                "sink_name=translator_remote_in",
                "rate=48000",
                "channels=2",
                "channel_map=front-left,front-right",
            ],
            format!(
                "sink_properties=\"device.description=Translator_Remote_In translator.owner=true translator.generation={generation}\""
            ),
        ),
    };
    let mut args: Vec<String> = values.iter().map(|value| (*value).to_owned()).collect();
    args.push(property);
    args
}

fn module_matches(role: EndpointRole, module: &PactlModule, generation: &str) -> bool {
    let expected = load_args(role, generation);
    module.name == expected[1] && module.argument == expected[2..].join(" ")
}

fn role_index(role: EndpointRole) -> usize {
    EndpointRole::ORDER
        .iter()
        .position(|candidate| *candidate == role)
        .unwrap_or(EndpointRole::ORDER.len())
}
