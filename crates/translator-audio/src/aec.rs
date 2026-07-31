use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{CommandRunError, CommandRunner, SystemCommandRunner};

pub const AEC_SOURCE: &str = "translator_aec_source";
pub const AEC_SINK: &str = "translator_aec_sink";
pub const AEC_ERLE_THRESHOLD_DB: f64 = 15.0;

const POWER_EPSILON: f64 = 1e-12;

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
    InvalidValidationInput,
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
            AecErrorCode::InvalidValidationInput => "AEC validation input is invalid",
        })
    }
}

impl std::error::Error for AecError {}

#[derive(Debug, Clone)]
struct OwnedAecModule {
    module_id: u32,
}

#[derive(Debug, Deserialize)]
struct PactlEndpoint {
    index: u32,
    name: String,
    owner_module: u32,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Debug)]
struct PactlModule {
    name: String,
    argument: String,
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
        if self.owned.is_some() {
            return Err(AecError::new(AecErrorCode::AlreadyOwned));
        }
        let result = self.run(&self.load_args(), AecErrorCode::ModuleLoadFailed)?;
        let module_id = std::str::from_utf8(result.stdout())
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .ok_or_else(|| AecError::new(AecErrorCode::ModuleLoadFailed))?;
        self.owned = Some(OwnedAecModule { module_id });
        self.inspect_owned()
    }

    pub fn inspect_owned(&self) -> Result<AecGraphState, AecError> {
        let owned = self
            .owned
            .as_ref()
            .ok_or_else(|| AecError::new(AecErrorCode::NotOwned))?;
        let modules = self.inspect_modules(AecErrorCode::InspectionFailed)?;
        let module = modules
            .get(&owned.module_id)
            .ok_or_else(|| AecError::new(AecErrorCode::OwnershipMismatch))?;
        if !self.module_matches(module) {
            return Err(AecError::new(AecErrorCode::OwnershipMismatch));
        }

        let source = self.exact_endpoint(
            self.inspect_endpoints("sources")?,
            AEC_SOURCE,
            owned.module_id,
        )?;
        let sink =
            self.exact_endpoint(self.inspect_endpoints("sinks")?, AEC_SINK, owned.module_id)?;
        Ok(AecGraphState {
            module_id: owned.module_id,
            source_id: source.index,
            sink_id: sink.index,
            pair: self.pair.clone(),
            generation: self.generation.clone(),
        })
    }

    pub fn cleanup_owned(&mut self) -> Result<Option<u32>, AecError> {
        let Some(owned) = self.owned.as_ref() else {
            return Ok(None);
        };
        let module_id = owned.module_id;
        self.inspect_for_cleanup(module_id)?;
        self.run(
            &["unload-module".to_owned(), module_id.to_string()],
            AecErrorCode::CleanupFailed,
        )?;
        self.owned = None;
        Ok(Some(module_id))
    }

    fn inspect_for_cleanup(&self, module_id: u32) -> Result<(), AecError> {
        let modules = self
            .inspect_modules(AecErrorCode::CleanupRefused)
            .map_err(|_| AecError::new(AecErrorCode::CleanupRefused))?;
        let module = modules
            .get(&module_id)
            .ok_or_else(|| AecError::new(AecErrorCode::CleanupRefused))?;
        if !self.module_matches(module) {
            return Err(AecError::new(AecErrorCode::CleanupRefused));
        }
        let sources = self
            .inspect_endpoints("sources")
            .map_err(|_| AecError::new(AecErrorCode::CleanupRefused))?;
        self.verify_cleanup_endpoints(sources, AEC_SOURCE, module_id)?;
        let sinks = self
            .inspect_endpoints("sinks")
            .map_err(|_| AecError::new(AecErrorCode::CleanupRefused))?;
        self.verify_cleanup_endpoints(sinks, AEC_SINK, module_id)
    }

    fn inspect_modules(
        &self,
        failure_code: AecErrorCode,
    ) -> Result<HashMap<u32, PactlModule>, AecError> {
        let result = self.run(
            &["list".to_owned(), "short".to_owned(), "modules".to_owned()],
            failure_code,
        )?;
        let text = std::str::from_utf8(result.stdout()).map_err(|_| AecError::new(failure_code))?;
        let mut modules = HashMap::new();
        for line in text.lines() {
            let mut fields = line.splitn(4, '\t');
            let Some(id) = fields.next().and_then(|value| value.parse::<u32>().ok()) else {
                continue;
            };
            let Some(name) = fields.next() else {
                continue;
            };
            modules.insert(
                id,
                PactlModule {
                    name: name.to_owned(),
                    argument: fields.next().unwrap_or_default().to_owned(),
                },
            );
        }
        Ok(modules)
    }

    fn inspect_endpoints(&self, kind: &str) -> Result<Vec<PactlEndpoint>, AecError> {
        let result = self.run(
            &[
                "--format=json".to_owned(),
                "list".to_owned(),
                kind.to_owned(),
            ],
            AecErrorCode::InspectionFailed,
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
        let arguments: HashMap<_, _> = module
            .argument
            .split_whitespace()
            .filter_map(|argument| argument.split_once('='))
            .collect();
        let owner_count = module.argument.matches("translator.owner=true").count();
        let generation_token = format!("translator.generation={}", self.generation);
        let generation_count = module.argument.matches(&generation_token).count();
        let source_properties = format!(
            "source_properties='device.description=Translator_AEC_Source translator.owner=true translator.generation={}'",
            self.generation
        );
        let sink_properties = format!(
            "sink_properties='device.description=Translator_AEC_Sink translator.owner=true translator.generation={}'",
            self.generation
        );
        module.name == "module-echo-cancel"
            && arguments.get("source_master") == Some(&self.pair.source.as_str())
            && arguments.get("sink_master") == Some(&self.pair.sink.as_str())
            && arguments.get("source_name") == Some(&AEC_SOURCE)
            && arguments.get("sink_name") == Some(&AEC_SINK)
            && arguments.get("rate") == Some(&"48000")
            && arguments.get("channels") == Some(&"1")
            && arguments.get("channel_map") == Some(&"mono")
            && arguments.get("aec_method") == Some(&"webrtc")
            && module.argument.matches(&source_properties).count() == 1
            && module.argument.matches(&sink_properties).count() == 1
            && owner_count == 2
            && generation_count == 2
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
    ) -> Result<crate::CommandResult, AecError> {
        let result = self
            .runner
            .run("pactl", args)
            .map_err(|error| match error {
                CommandRunError::NotFound => AecError::new(AecErrorCode::PactlMissing),
                CommandRunError::SpawnFailed | CommandRunError::TimedOut => {
                    AecError::new(failure_code)
                }
            })?;
        if result.is_success() {
            Ok(result)
        } else {
            Err(AecError::new(failure_code))
        }
    }
}

fn is_safe_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecDeviceMetadata {
    pub source_name: String,
    pub sink_name: String,
    pub source_geometry: String,
    pub sink_geometry: String,
    pub sink_port: String,
    pub sink_volume_percent: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AecPowerWindow {
    pub sequence: u64,
    pub raw_power: f64,
    pub clean_power: f64,
    pub noise_power: f64,
}

impl AecPowerWindow {
    pub fn new(sequence: u64, raw_power: f64, clean_power: f64, noise_power: f64) -> Self {
        Self {
            sequence,
            raw_power,
            clean_power,
            noise_power,
        }
    }

    pub fn erle_db(self) -> Result<f64, AecError> {
        if ![self.raw_power, self.clean_power, self.noise_power]
            .into_iter()
            .all(|power| power.is_finite() && power >= 0.0)
        {
            return Err(AecError::new(AecErrorCode::InvalidValidationInput));
        }
        let raw = (self.raw_power - self.noise_power).max(POWER_EPSILON);
        let clean = (self.clean_power - self.noise_power).max(POWER_EPSILON);
        let erle = 10.0 * (raw / clean).log10();
        if erle.is_finite() {
            Ok(erle)
        } else {
            Err(AecError::new(AecErrorCode::InvalidValidationInput))
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AecFarEndCounters {
    pub vad_triggers: u64,
    pub provider_requests: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecValidationInput {
    pub metadata: AecDeviceMetadata,
    pub windows: Vec<AecPowerWindow>,
    pub far_end: AecFarEndCounters,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AecValidationRecord {
    pub metadata: AecDeviceMetadata,
    pub window_count: usize,
    pub median_erle_db: f64,
    pub erle_passed: bool,
    pub far_end: AecFarEndCounters,
    pub far_end_passed: bool,
    pub validated: bool,
}

pub fn evaluate_aec(input: AecValidationInput) -> Result<AecValidationRecord, AecError> {
    if input.windows.is_empty()
        || input.metadata.source_name.is_empty()
        || input.metadata.sink_name.is_empty()
        || input.metadata.source_geometry.is_empty()
        || input.metadata.sink_geometry.is_empty()
        || input.metadata.sink_port.is_empty()
        || input.metadata.sink_volume_percent > 100
        || input
            .windows
            .windows(2)
            .any(|pair| pair[0].sequence >= pair[1].sequence)
    {
        return Err(AecError::new(AecErrorCode::InvalidValidationInput));
    }

    let mut erle_values = input
        .windows
        .iter()
        .copied()
        .map(AecPowerWindow::erle_db)
        .collect::<Result<Vec<_>, _>>()?;
    erle_values.sort_by(f64::total_cmp);
    let middle = erle_values.len() / 2;
    let median_erle_db = if erle_values.len() % 2 == 0 {
        (erle_values[middle - 1] + erle_values[middle]) / 2.0
    } else {
        erle_values[middle]
    };
    let erle_passed = median_erle_db >= AEC_ERLE_THRESHOLD_DB;
    let far_end_passed = input.far_end.vad_triggers == 0 && input.far_end.provider_requests == 0;
    Ok(AecValidationRecord {
        metadata: input.metadata,
        window_count: input.windows.len(),
        median_erle_db,
        erle_passed,
        far_end: input.far_end,
        far_end_passed,
        validated: erle_passed && far_end_passed,
    })
}
