use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::Deserialize;

use crate::{CommandRunError, CommandRunner, MIC_OUT_SINK, REMOTE_IN_SINK, SystemCommandRunner};

pub const OUTGOING_TRANSLATION_STREAM: &str = "translator-outgoing-playback";
pub const INCOMING_TRANSLATION_STREAM: &str = "translator-incoming-playback";
const MICROPHONE_ORIGINAL_STREAM: &str = "loopback-microphone-original";
const SPEAKER_ORIGINAL_STREAM: &str = "loopback-speaker-original";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioMixVolumes {
    pub microphone_original_percent: u8,
    pub microphone_translation_percent: u8,
    pub speaker_original_percent: u8,
    pub speaker_translation_percent: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioMixTarget {
    MicrophoneOriginal,
    MicrophoneTranslation,
    SpeakerOriginal,
    SpeakerTranslation,
}

impl AudioMixTarget {
    pub const fn percent_from(self, volumes: AudioMixVolumes) -> u8 {
        match self {
            Self::MicrophoneOriginal => volumes.microphone_original_percent,
            Self::MicrophoneTranslation => volumes.microphone_translation_percent,
            Self::SpeakerOriginal => volumes.speaker_original_percent,
            Self::SpeakerTranslation => volumes.speaker_translation_percent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixPercent(u8);

impl TryFrom<u8> for MixPercent {
    type Error = AudioMixError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value > 100 {
            return Err(AudioMixError::new(AudioMixErrorCode::InvalidVolume));
        }
        Ok(Self(value))
    }
}

#[derive(Debug)]
pub struct PulseMixPlan(Vec<PulseMixEntry>);

impl PulseMixPlan {
    pub fn entries(&self) -> &[PulseMixEntry] {
        &self.0
    }
}

#[derive(Debug)]
pub struct PulseMixEntry {
    index: u32,
    target: AudioMixTarget,
    prior: Vec<u32>,
}

impl PulseMixEntry {
    pub const fn target(&self) -> AudioMixTarget {
        self.target
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioMixErrorCode {
    DiscoveryFailed,
    VolumeApplyFailed,
    InvalidVolume,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioMixError {
    code: AudioMixErrorCode,
}

impl AudioMixError {
    fn new(code: AudioMixErrorCode) -> Self {
        Self { code }
    }

    pub const fn code(&self) -> AudioMixErrorCode {
        self.code
    }
}

impl fmt::Display for AudioMixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.code {
            AudioMixErrorCode::DiscoveryFailed => "Audio mix stream discovery failed",
            AudioMixErrorCode::VolumeApplyFailed => "Audio mix stream volume update failed",
            AudioMixErrorCode::InvalidVolume => "Audio mix volume is invalid",
        })
    }
}

impl std::error::Error for AudioMixError {}

pub struct PulseAudioMix<R = SystemCommandRunner> {
    runner: R,
}

impl<R> PulseAudioMix<R>
where
    R: CommandRunner,
{
    pub const fn new(runner: R) -> Self {
        Self { runner }
    }

    pub fn discover(&self) -> Result<PulseMixPlan, AudioMixError> {
        let sink_inputs: Vec<RawSinkInput> =
            self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let source_outputs: Vec<RawSourceOutput> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let original_pairs = owned_original_pairs(&sink_inputs, &source_outputs)?;
        let original_targets = if original_pairs.is_empty() {
            HashMap::new()
        } else {
            let sources: Vec<RawEndpoint> = self.run_json(&["--format=json", "list", "sources"])?;
            let sinks: Vec<RawEndpoint> = self.run_json(&["--format=json", "list", "sinks"])?;
            classify_original_pairs(&original_pairs, &sources, &sinks)?
        };
        let mut entries = Vec::new();
        let mut seen_indices = HashSet::new();

        for input in sink_inputs {
            let Some(target) = original_targets
                .get(&input.index)
                .copied()
                .or_else(|| classify_translation_stream(&input))
            else {
                continue;
            };
            if !seen_indices.insert(input.index) {
                return Err(discovery_failed());
            }
            let channels: Vec<_> = input.channel_map.split(',').collect();
            let unique: HashSet<_> = channels.iter().copied().collect();
            if channels.is_empty()
                || channels.iter().any(|channel| channel.is_empty())
                || unique.len() != channels.len()
                || channels.len() != input.volume.len()
            {
                return Err(AudioMixError::new(AudioMixErrorCode::DiscoveryFailed));
            }
            let prior = channels
                .into_iter()
                .map(|channel| {
                    input
                        .volume
                        .get(channel)
                        .filter(|volume| volume.value <= i32::MAX as u32)
                        .map(|volume| volume.value)
                        .ok_or_else(|| AudioMixError::new(AudioMixErrorCode::DiscoveryFailed))
                })
                .collect::<Result<Vec<_>, _>>()?;
            entries.push(PulseMixEntry {
                index: input.index,
                target,
                prior,
            });
        }

        Ok(PulseMixPlan(entries))
    }

    pub fn set_percent(
        &self,
        entry: &PulseMixEntry,
        percent: MixPercent,
    ) -> Result<(), AudioMixError> {
        self.set_volume(entry.index, [format!("{}%", percent.0)])
    }

    pub fn restore_raw(&self, entry: &PulseMixEntry) -> Result<(), AudioMixError> {
        self.set_volume(entry.index, entry.prior.iter().map(u32::to_string))
    }

    fn set_volume(
        &self,
        index: u32,
        values: impl IntoIterator<Item = String>,
    ) -> Result<(), AudioMixError> {
        let mut args = vec!["set-sink-input-volume".to_owned(), index.to_string()];
        args.extend(values);
        let result = self.runner.run("pactl", &args).map_err(map_apply_error)?;
        if result.is_success() {
            Ok(())
        } else {
            Err(AudioMixError::new(AudioMixErrorCode::VolumeApplyFailed))
        }
    }

    fn run_json<T>(&self, args: &[&str]) -> Result<T, AudioMixError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let arguments: Vec<String> = args.iter().map(|value| (*value).to_owned()).collect();
        let result = self
            .runner
            .run("pactl", &arguments)
            .map_err(|_| AudioMixError::new(AudioMixErrorCode::DiscoveryFailed))?;
        if !result.is_success() {
            return Err(AudioMixError::new(AudioMixErrorCode::DiscoveryFailed));
        }
        serde_json::from_slice(result.stdout())
            .map_err(|_| AudioMixError::new(AudioMixErrorCode::DiscoveryFailed))
    }
}

fn map_apply_error(error: CommandRunError) -> AudioMixError {
    match error {
        CommandRunError::NotFound
        | CommandRunError::SpawnFailed
        | CommandRunError::TimedOut
        | CommandRunError::DeadlineExpired => {
            AudioMixError::new(AudioMixErrorCode::VolumeApplyFailed)
        }
    }
}

fn classify_translation_stream(input: &RawSinkInput) -> Option<AudioMixTarget> {
    let media_name = property(&input.properties, "media.name")?;
    let application_name = property(&input.properties, "application.name");
    if application_name == Some("translator-daemon") {
        match media_name {
            OUTGOING_TRANSLATION_STREAM => Some(AudioMixTarget::MicrophoneTranslation),
            INCOMING_TRANSLATION_STREAM => Some(AudioMixTarget::SpeakerTranslation),
            _ => None,
        }
    } else {
        None
    }
}

struct OriginalPair {
    input_index: u32,
    target: AudioMixTarget,
    sink_index: u32,
    source_index: u32,
}

fn owned_original_pairs(
    inputs: &[RawSinkInput],
    outputs: &[RawSourceOutput],
) -> Result<Vec<OriginalPair>, AudioMixError> {
    let mut owned_inputs = HashMap::new();
    let mut owned_outputs = HashMap::new();
    for input in inputs {
        if let Some(target) = owned_original_target(&input.properties) {
            let module = canonical_module_id(input.owner_module.as_ref())?;
            let sink = input.sink.ok_or_else(discovery_failed)?;
            if owned_inputs
                .insert(module, (input.index, target, sink))
                .is_some()
            {
                return Err(discovery_failed());
            }
        }
    }
    for output in outputs {
        if let Some(target) = owned_original_target(&output.properties) {
            let module = canonical_module_id(output.owner_module.as_ref())?;
            let source = output.source.ok_or_else(discovery_failed)?;
            if owned_outputs.insert(module, (target, source)).is_some() {
                return Err(discovery_failed());
            }
        }
    }
    if owned_inputs.len() != owned_outputs.len() {
        return Err(discovery_failed());
    }
    let mut pairs = Vec::new();
    let mut seen_targets = Vec::new();
    for (module, (input_index, target, sink_index)) in owned_inputs {
        let (output_target, source_index) =
            owned_outputs.get(&module).ok_or_else(discovery_failed)?;
        if *output_target != target || seen_targets.contains(&target) {
            return Err(discovery_failed());
        }
        seen_targets.push(target);
        pairs.push(OriginalPair {
            input_index,
            target,
            sink_index,
            source_index: *source_index,
        });
    }
    Ok(pairs)
}

fn classify_original_pairs(
    pairs: &[OriginalPair],
    sources: &[RawEndpoint],
    sinks: &[RawEndpoint],
) -> Result<HashMap<u32, AudioMixTarget>, AudioMixError> {
    let mut targets = HashMap::new();
    for pair in pairs {
        let source = endpoint_name(sources, pair.source_index)?;
        let sink = endpoint_name(sinks, pair.sink_index)?;
        let valid = match pair.target {
            AudioMixTarget::MicrophoneOriginal => sink == MIC_OUT_SINK,
            AudioMixTarget::SpeakerOriginal => source == format!("{REMOTE_IN_SINK}.monitor"),
            _ => false,
        };
        if !valid || targets.insert(pair.input_index, pair.target).is_some() {
            return Err(discovery_failed());
        }
    }
    Ok(targets)
}

fn endpoint_name(endpoints: &[RawEndpoint], index: u32) -> Result<&str, AudioMixError> {
    let mut names = endpoints.iter().filter(|endpoint| endpoint.index == index);
    let name = names.next().ok_or_else(discovery_failed)?.name.as_str();
    if name.is_empty() || names.next().is_some() {
        return Err(discovery_failed());
    }
    Ok(name)
}

fn owned_original_target(properties: &HashMap<String, String>) -> Option<AudioMixTarget> {
    if property(properties, "translator.owner") != Some("true") {
        return None;
    }
    match property(properties, "media.name") {
        Some(MICROPHONE_ORIGINAL_STREAM) => Some(AudioMixTarget::MicrophoneOriginal),
        Some(SPEAKER_ORIGINAL_STREAM) => Some(AudioMixTarget::SpeakerOriginal),
        _ => None,
    }
}

fn canonical_module_id(id: Option<&RawModuleId>) -> Result<String, AudioMixError> {
    match id.ok_or_else(discovery_failed)? {
        RawModuleId::Number(value) => Ok(value.to_string()),
        RawModuleId::Text(value)
            if value
                .parse::<u32>()
                .is_ok_and(|parsed| parsed.to_string() == *value) =>
        {
            Ok(value.clone())
        }
        RawModuleId::Text(_) => Err(discovery_failed()),
    }
}

fn discovery_failed() -> AudioMixError {
    AudioMixError::new(AudioMixErrorCode::DiscoveryFailed)
}

fn property<'a>(properties: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    properties.get(key).map(String::as_str)
}

#[derive(Debug, Deserialize)]
struct RawSinkInput {
    index: u32,
    owner_module: Option<RawModuleId>,
    sink: Option<u32>,
    #[serde(default)]
    channel_map: String,
    #[serde(default)]
    volume: HashMap<String, RawVolume>,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawVolume {
    value: u32,
}

#[derive(Debug, Deserialize)]
struct RawSourceOutput {
    owner_module: Option<RawModuleId>,
    source: Option<u32>,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawModuleId {
    Text(String),
    Number(u32),
}

#[derive(Debug, Deserialize)]
struct RawEndpoint {
    index: u32,
    name: String,
}
