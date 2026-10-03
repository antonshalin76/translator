use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

use serde::Deserialize;

use crate::{
    CommandRunError, CommandRunner, MIC_OUT_SINK, MICROPHONE_ORIGINAL_CAPTURE,
    MICROPHONE_ORIGINAL_PLAYBACK, OriginalMicrophoneRegistration, OriginalMicrophoneRegistry,
    PulsePlaybackRegistration, REMOTE_IN_SINK, SESSION_PROPERTY, SystemCommandRunner,
};

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
    original_microphone: Option<OriginalMicrophoneRegistration>,
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
    original_microphone: Option<OriginalMicrophoneRegistry>,
}

impl<R> PulseAudioMix<R>
where
    R: CommandRunner,
{
    pub const fn new(runner: R) -> Self {
        Self {
            runner,
            original_microphone: None,
        }
    }

    pub fn with_original_microphone(runner: R, registry: OriginalMicrophoneRegistry) -> Self {
        Self {
            runner,
            original_microphone: Some(registry),
        }
    }

    pub fn discover(&self) -> Result<PulseMixPlan, AudioMixError> {
        let sink_inputs: Vec<RawSinkInput> =
            self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let source_outputs: Vec<RawSourceOutput> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let original_pairs = owned_original_pairs(&sink_inputs, &source_outputs)?;
        if self.original_microphone.is_some()
            && original_pairs
                .iter()
                .any(|pair| pair.target == AudioMixTarget::MicrophoneOriginal)
        {
            return Err(discovery_failed());
        }
        let registration = self.current_original_microphone()?;
        if registration.is_none()
            && (sink_inputs
                .iter()
                .any(|input| claims_native_original(&input.properties))
                || source_outputs
                    .iter()
                    .any(|output| claims_native_original(&output.properties)))
        {
            return Err(discovery_failed());
        }
        let mut original_targets = if original_pairs.is_empty() && registration.is_none() {
            HashMap::new()
        } else {
            let sources: Vec<RawEndpoint> = self.run_json(&["--format=json", "list", "sources"])?;
            let sinks: Vec<RawEndpoint> = self.run_json(&["--format=json", "list", "sinks"])?;
            if let Some(registration) = &registration {
                let clients: Vec<RawClient> =
                    self.run_json(&["--format=json", "list", "clients"])?;
                native_original_input(
                    registration,
                    &sink_inputs,
                    &source_outputs,
                    &sources,
                    &sinks,
                    &clients,
                )?;
                self.require_current_original(registration)?;
            }
            classify_original_pairs(&original_pairs, &sources, &sinks)?
        };
        if let Some(registration) = &registration
            && original_targets
                .insert(
                    registration.playback_index(),
                    AudioMixTarget::MicrophoneOriginal,
                )
                .is_some()
        {
            return Err(discovery_failed());
        }
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
                original_microphone: registration
                    .as_ref()
                    .filter(|registration| registration.playback_index() == input.index)
                    .cloned(),
            });
        }

        if let Some(registration) = &registration {
            self.require_current_original(registration)?;
        }
        Ok(PulseMixPlan(entries))
    }

    pub fn set_percent(
        &self,
        entry: &PulseMixEntry,
        percent: MixPercent,
    ) -> Result<(), AudioMixError> {
        self.original_operation(entry, || {
            if let Some(registration) = &entry.original_microphone {
                self.native_registered_input(registration)?;
            }
            self.set_volume(entry.index, [format!("{}%", percent.0)])?;
            if let Some(registration) = &entry.original_microphone {
                let input = self.native_registered_input(registration)?;
                let expected = (u32::from(percent.0) * 65_536 + 50) / 100;
                let tolerance = u32::from(percent.0 != 0);
                if input
                    .volume
                    .values()
                    .any(|channel| channel.value.abs_diff(expected) > tolerance)
                {
                    return Err(discovery_failed());
                }
            }
            Ok(())
        })
    }

    pub fn quarantine_original_microphone(&self) -> bool {
        let Some(registry) = &self.original_microphone else {
            return false;
        };
        let held = registry
            .current()
            .map_or(true, |registration| registration.is_some());
        registry.cancel_current();
        held
    }

    fn original_operation<T>(
        &self,
        entry: &PulseMixEntry,
        operation: impl FnOnce() -> Result<T, AudioMixError>,
    ) -> Result<T, AudioMixError> {
        let result = operation();
        if result.is_err()
            && let Some(registration) = &entry.original_microphone
        {
            registration.cancel();
        }
        result
    }

    fn current_original_microphone(
        &self,
    ) -> Result<Option<OriginalMicrophoneRegistration>, AudioMixError> {
        self.original_microphone
            .as_ref()
            .map_or(Ok(None), |registry| {
                registry.current().map_err(|_| discovery_failed())
            })
    }

    fn require_current_original(
        &self,
        registration: &OriginalMicrophoneRegistration,
    ) -> Result<(), AudioMixError> {
        let current = self
            .current_original_microphone()?
            .ok_or_else(discovery_failed)?;
        if !registration.is_live() || !current.is_live() || !current.same_session(registration) {
            return Err(discovery_failed());
        }
        Ok(())
    }

    fn native_registered_input(
        &self,
        registration: &OriginalMicrophoneRegistration,
    ) -> Result<RawSinkInput, AudioMixError> {
        self.require_current_original(registration)?;
        let inputs: Vec<RawSinkInput> = self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let outputs: Vec<RawSourceOutput> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let sources: Vec<RawEndpoint> = self.run_json(&["--format=json", "list", "sources"])?;
        let sinks: Vec<RawEndpoint> = self.run_json(&["--format=json", "list", "sinks"])?;
        let clients: Vec<RawClient> = self.run_json(&["--format=json", "list", "clients"])?;
        if owned_original_pairs(&inputs, &outputs)?
            .iter()
            .any(|pair| pair.target == AudioMixTarget::MicrophoneOriginal)
        {
            return Err(discovery_failed());
        }
        native_original_input(registration, &inputs, &outputs, &sources, &sinks, &clients)?;
        self.require_current_original(registration)?;
        inputs
            .into_iter()
            .find(|input| input.index == registration.playback_index())
            .ok_or_else(discovery_failed)
    }

    pub fn verify_registered_zero(
        &self,
        registration: &PulsePlaybackRegistration,
        deadline: Instant,
    ) -> Result<(), AudioMixError> {
        let input = self.registered_input(registration, deadline)?;
        if input.volume.values().any(|channel| channel.value != 0) {
            return Err(discovery_failed());
        }
        Ok(())
    }

    pub fn admit_registered_percent(
        &self,
        registration: &PulsePlaybackRegistration,
        percent: MixPercent,
        deadline: Instant,
    ) -> Result<(), AudioMixError> {
        self.verify_registered_zero(registration, deadline)?;
        let args = vec![
            "set-sink-input-volume".to_owned(),
            registration.index().to_string(),
            format!("{}%", percent.0),
        ];
        let result = self
            .runner
            .run_until("pactl", &args, deadline)
            .map_err(map_apply_error)?;
        if !result.is_success() {
            return Err(AudioMixError::new(AudioMixErrorCode::VolumeApplyFailed));
        }
        let observed = self.registered_input(registration, deadline)?;
        let expected = (u32::from(percent.0) * 65_536 + 50) / 100;
        if observed
            .volume
            .values()
            .any(|channel| channel.value.abs_diff(expected) > 1)
        {
            return Err(discovery_failed());
        }
        Ok(())
    }

    fn registered_input(
        &self,
        registration: &PulsePlaybackRegistration,
        deadline: Instant,
    ) -> Result<RawSinkInput, AudioMixError> {
        let inputs: Vec<RawSinkInput> =
            self.run_json_until(&["--format=json", "list", "sink-inputs"], deadline)?;
        let sinks: Vec<RawEndpoint> =
            self.run_json_until(&["--format=json", "list", "sinks"], deadline)?;
        let mut matches = inputs
            .into_iter()
            .filter(|input| input.index == registration.index());
        let input = matches.next().ok_or_else(discovery_failed)?;
        if matches.next().is_some()
            || input.sink.is_none_or(|index| {
                endpoint_name(&sinks, index).ok() != Some(registration.device())
            })
            || property(&input.properties, "application.name") != Some("translator-daemon")
            || property(&input.properties, "application.process.id")
                != Some(registration.process_id().to_string().as_str())
            || property(&input.properties, "media.name") != Some(registration.stream_name())
            || property(&input.properties, "translator.playback_session")
                != Some(registration.session_id().to_string().as_str())
        {
            return Err(discovery_failed());
        }
        let channels: Vec<_> = input.channel_map.split(',').collect();
        if channels.is_empty()
            || channels.iter().any(|channel| channel.is_empty())
            || channels.iter().collect::<HashSet<_>>().len() != channels.len()
            || channels.len() != input.volume.len()
            || channels
                .iter()
                .any(|channel| !input.volume.contains_key(*channel))
        {
            return Err(discovery_failed());
        }
        Ok(input)
    }

    fn run_json_until<T>(&self, args: &[&str], deadline: Instant) -> Result<T, AudioMixError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let arguments: Vec<String> = args.iter().map(|value| (*value).to_owned()).collect();
        let result = self
            .runner
            .run_until("pactl", &arguments, deadline)
            .map_err(|_| discovery_failed())?;
        if !result.is_success() {
            return Err(discovery_failed());
        }
        serde_json::from_slice(result.stdout()).map_err(|_| discovery_failed())
    }

    pub fn verify_zero_targets(
        &self,
        prior: &PulseMixPlan,
        targets: &[AudioMixTarget],
    ) -> Result<(), AudioMixError> {
        let observed = self.discover()?;
        for target in targets {
            let expected: Vec<_> = prior
                .entries()
                .iter()
                .filter(|entry| entry.target == *target)
                .collect();
            let actual: Vec<_> = observed
                .entries()
                .iter()
                .filter(|entry| entry.target == *target)
                .collect();
            if expected.len() != actual.len() || expected.is_empty() {
                return Err(discovery_failed());
            }
            for entry in expected {
                let match_entry = actual
                    .iter()
                    .find(|observed| observed.index == entry.index)
                    .ok_or_else(discovery_failed)?;
                if match_entry.prior.len() != entry.prior.len()
                    || match_entry.prior.iter().any(|volume| *volume != 0)
                {
                    return Err(discovery_failed());
                }
                match (&entry.original_microphone, &match_entry.original_microphone) {
                    (Some(expected), Some(actual)) if expected.same_session(actual) => {
                        self.require_current_original(expected)?;
                    }
                    (None, None) => {}
                    _ => return Err(discovery_failed()),
                }
            }
        }
        Ok(())
    }

    pub fn restore_raw(&self, entry: &PulseMixEntry) -> Result<(), AudioMixError> {
        self.original_operation(entry, || {
            if let Some(registration) = &entry.original_microphone {
                self.native_registered_input(registration)?;
            }
            self.set_volume(entry.index, entry.prior.iter().map(u32::to_string))?;
            if let Some(registration) = &entry.original_microphone {
                let input = self.native_registered_input(registration)?;
                if entry.prior.as_slice() != [input.volume["mono"].value] {
                    return Err(discovery_failed());
                }
            }
            Ok(())
        })
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

fn claims_native_original(properties: &HashMap<String, String>) -> bool {
    matches!(
        property(properties, "media.name"),
        Some(MICROPHONE_ORIGINAL_PLAYBACK | MICROPHONE_ORIGINAL_CAPTURE)
    ) || properties.contains_key(SESSION_PROPERTY)
}

fn native_identity_matches(
    properties: &HashMap<String, String>,
    registration: &OriginalMicrophoneRegistration,
    media_name: Option<&str>,
) -> bool {
    property(properties, "application.name") == Some("translator-daemon")
        && property(properties, "application.process.id")
            == Some(registration.process_id().to_string().as_str())
        && property(properties, SESSION_PROPERTY)
            == Some(registration.session_id().to_string().as_str())
        && media_name.is_none_or(|name| property(properties, "media.name") == Some(name))
}

fn native_original_input<'a>(
    registration: &OriginalMicrophoneRegistration,
    inputs: &'a [RawSinkInput],
    outputs: &[RawSourceOutput],
    sources: &[RawEndpoint],
    sinks: &[RawEndpoint],
    clients: &[RawClient],
) -> Result<&'a RawSinkInput, AudioMixError> {
    if !registration.is_live() {
        return Err(discovery_failed());
    }
    let mut inputs = inputs.iter().filter(|input| {
        input.index == registration.playback_index() || claims_native_original(&input.properties)
    });
    let input = inputs.next().ok_or_else(discovery_failed)?;
    let mut outputs = outputs.iter().filter(|output| {
        output.index == Some(registration.capture_index())
            || claims_native_original(&output.properties)
    });
    let output = outputs.next().ok_or_else(discovery_failed)?;
    let session = registration.session_id().to_string();
    let mut clients = clients.iter().filter(|client| {
        client.index == registration.client_id()
            || property(&client.properties, SESSION_PROPERTY) == Some(session.as_str())
    });
    let client = clients.next().ok_or_else(discovery_failed)?;
    if inputs.next().is_some()
        || outputs.next().is_some()
        || clients.next().is_some()
        || input.index != registration.playback_index()
        || output.index != Some(registration.capture_index())
        || client.index != registration.client_id()
        || canonical_module_id(input.client.as_ref())? != registration.client_id().to_string()
        || canonical_module_id(output.client.as_ref())? != registration.client_id().to_string()
        || !native_identity_matches(
            &input.properties,
            registration,
            Some(MICROPHONE_ORIGINAL_PLAYBACK),
        )
        || !native_identity_matches(
            &output.properties,
            registration,
            Some(MICROPHONE_ORIGINAL_CAPTURE),
        )
        || !native_identity_matches(&client.properties, registration, None)
        || input.sink != Some(registration.sink_index())
        || output.source != Some(registration.source_index())
        || endpoint_name(sources, registration.source_index())? != registration.source_name()
        || endpoint_name(sinks, registration.sink_index())? != MIC_OUT_SINK
        || sources
            .iter()
            .filter(|source| source.name == registration.source_name())
            .count()
            != 1
        || sinks
            .iter()
            .filter(|sink| sink.name == MIC_OUT_SINK)
            .count()
            != 1
        || input.mute != Some(false)
        || output.mute != Some(false)
        || input.channel_map != "mono"
        || output.channel_map != "mono"
        || input.volume.len() != 1
        || input
            .volume
            .get("mono")
            .is_none_or(|volume| volume.value > i32::MAX as u32)
        || output.volume.len() != 1
        || output
            .volume
            .get("mono")
            .is_none_or(|volume| volume.value != 65_536)
    {
        return Err(discovery_failed());
    }
    Ok(input)
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
    client: Option<RawModuleId>,
    #[serde(default)]
    mute: Option<bool>,
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
    index: Option<u32>,
    #[serde(default)]
    client: Option<RawModuleId>,
    #[serde(default)]
    mute: Option<bool>,
    #[serde(default)]
    channel_map: String,
    #[serde(default)]
    volume: HashMap<String, RawVolume>,
    #[serde(default)]
    properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawClient {
    index: u32,
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::{Value, json};
    use uuid::Uuid;

    use super::*;
    use crate::CommandResult;

    #[derive(Clone)]
    struct Runner(Arc<Mutex<Inventory>>);

    struct Inventory {
        lists: HashMap<&'static str, Value>,
        writes: Vec<Vec<String>>,
        apply_writes: bool,
        replace_after_write: bool,
    }

    impl CommandRunner for Runner {
        fn run_until(
            &self,
            program: &str,
            args: &[String],
            _deadline: Instant,
        ) -> Result<CommandResult, CommandRunError> {
            assert_eq!(program, "pactl");
            let mut inventory = self.0.lock().unwrap();
            if args[0] == "--format=json" {
                return Ok(CommandResult::success(
                    serde_json::to_vec(&inventory.lists[args[2].as_str()]).unwrap(),
                ));
            }
            assert_eq!(args[0], "set-sink-input-volume");
            assert_eq!(args[1], "41");
            inventory.writes.push(args.to_vec());
            if inventory.apply_writes {
                let raw = args[2].strip_suffix('%').map_or_else(
                    || args[2].parse::<u32>().unwrap(),
                    |percent| (percent.parse::<u32>().unwrap() * 65_536 + 50) / 100,
                );
                inventory.lists.get_mut("sink-inputs").unwrap()[0]["volume"]["mono"]["value"] =
                    json!(raw);
            }
            if inventory.replace_after_write {
                inventory.lists.get_mut("sink-inputs").unwrap()[0]["properties"]["application.process.id"] =
                    json!("9999");
            }
            Ok(CommandResult::success(Vec::new()))
        }
    }

    struct Fixture {
        device: PulseAudioMix<Runner>,
        runner: Runner,
        registry: OriginalMicrophoneRegistry,
        registration: OriginalMicrophoneRegistration,
    }

    impl Fixture {
        fn new() -> Self {
            let registry = OriginalMicrophoneRegistry::default();
            let registration = registry.test_register(
                41,
                42,
                4,
                8,
                "translator_test_mic.monitor",
                Uuid::from_u128(123),
                1234,
                6,
            );
            let properties = json!({
                "application.name": "translator-daemon",
                "application.process.id": registration.process_id().to_string(),
                (SESSION_PROPERTY): registration.session_id().to_string(),
            });
            let mut input_properties = properties.clone();
            input_properties["media.name"] = json!(MICROPHONE_ORIGINAL_PLAYBACK);
            let mut output_properties = properties.clone();
            output_properties["media.name"] = json!(MICROPHONE_ORIGINAL_CAPTURE);
            let runner = Runner(Arc::new(Mutex::new(Inventory {
                lists: HashMap::from([
                    (
                        "sink-inputs",
                        json!([{
                            "index": 41, "client": 6, "owner_module": "0", "sink": 8, "mute": false,
                            "channel_map": "mono", "sample_specification": "s16le 1ch 48000Hz",
                            "volume": {"mono": {"value": 0}}, "properties": input_properties,
                        }]),
                    ),
                    (
                        "source-outputs",
                        json!([{
                            "index": 42, "client": "6", "owner_module": 0, "source": 4, "mute": false,
                            "channel_map": "mono", "sample_specification": "s16le 1ch 48000Hz",
                            "volume": {"mono": {"value": 65_536}}, "properties": output_properties,
                        }]),
                    ),
                    (
                        "sources",
                        json!([{"index": 4, "name": "translator_test_mic.monitor"}]),
                    ),
                    ("sinks", json!([{"index": 8, "name": MIC_OUT_SINK}])),
                    ("clients", json!([{"index": 6, "properties": properties}])),
                ]),
                writes: Vec::new(),
                apply_writes: true,
                replace_after_write: false,
            })));
            Self {
                device: PulseAudioMix::with_original_microphone(runner.clone(), registry.clone()),
                runner,
                registry,
                registration,
            }
        }

        fn change(&self, list: &str, pointer: &str, value: Value) {
            *self
                .runner
                .0
                .lock()
                .unwrap()
                .lists
                .get_mut(list)
                .unwrap()
                .pointer_mut(pointer)
                .unwrap() = value;
        }

        fn assert_no_writes(&self) {
            assert!(self.runner.0.lock().unwrap().writes.is_empty());
        }
    }

    #[test]
    fn native_original_uses_live_registration_and_verifies_gain_and_restore() {
        let fixture = Fixture::new();
        let plan = fixture.device.discover().unwrap();
        assert_eq!(plan.entries().len(), 1);
        let entry = &plan.entries()[0];
        assert_eq!(entry.target(), AudioMixTarget::MicrophoneOriginal);
        for percent in [0, 35, 100, 0] {
            fixture
                .device
                .set_percent(entry, MixPercent::try_from(percent).unwrap())
                .unwrap();
        }
        fixture
            .device
            .verify_zero_targets(&plan, &[AudioMixTarget::MicrophoneOriginal])
            .unwrap();
        fixture.device.restore_raw(entry).unwrap();
        assert_eq!(fixture.runner.0.lock().unwrap().writes.len(), 5);
    }

    #[test]
    fn copied_native_metadata_without_retained_registry_fails_discovery() {
        let fixture = Fixture::new();
        let unregistered = PulseAudioMix::new(fixture.runner.clone());
        assert!(unregistered.discover().is_err());
        fixture.registry.test_clear();
        assert!(fixture.device.discover().is_err());
        fixture.assert_no_writes();
    }

    #[test]
    fn native_original_identity_and_channel_mismatches_fail_discovery() {
        let mut changes = vec![
            ("sink-inputs", "/0/index", json!(43)),
            ("source-outputs", "/0/index", json!(43)),
            ("sink-inputs", "/0/client", json!(7)),
            ("source-outputs", "/0/client", json!("06")),
            ("clients", "/0/index", json!(7)),
            ("sink-inputs", "/0/sink", json!(9)),
            ("source-outputs", "/0/source", json!(5)),
            ("sources", "/0/index", json!(5)),
            ("sources", "/0/name", json!("foreign-source")),
            ("sinks", "/0/index", json!(9)),
            ("sinks", "/0/name", json!("foreign-sink")),
            ("sink-inputs", "/0/mute", json!(true)),
            ("source-outputs", "/0/mute", json!(true)),
            ("sink-inputs", "/0/channel_map", json!("front-left")),
            ("source-outputs", "/0/channel_map", json!("front-left")),
            ("source-outputs", "/0/volume/mono/value", json!(32_768)),
            (
                "sink-inputs",
                "/0/volume",
                json!({"front-left": {"value": 0}}),
            ),
            (
                "sink-inputs",
                "/0/properties/media.name",
                json!(MICROPHONE_ORIGINAL_CAPTURE),
            ),
            (
                "source-outputs",
                "/0/properties/media.name",
                json!(MICROPHONE_ORIGINAL_PLAYBACK),
            ),
        ];
        for list in ["sink-inputs", "source-outputs", "clients"] {
            changes.extend([
                (list, "/0/properties/application.name", json!("foreign")),
                (list, "/0/properties/application.process.id", json!("9999")),
                (
                    list,
                    "/0/properties/translator.original_microphone_session",
                    json!(Uuid::from_u128(124).to_string()),
                ),
            ]);
        }
        for (list, pointer, value) in changes {
            let fixture = Fixture::new();
            fixture.change(list, pointer, value);
            assert!(fixture.device.discover().is_err(), "{list} {pointer}");
            fixture.assert_no_writes();
        }
    }

    #[test]
    fn duplicate_native_streams_clients_and_pinned_endpoints_fail_discovery() {
        for list in [
            "sink-inputs",
            "source-outputs",
            "clients",
            "sources",
            "sinks",
        ] {
            for copied_index in [false, true] {
                let fixture = Fixture::new();
                {
                    let mut inventory = fixture.runner.0.lock().unwrap();
                    let items = inventory
                        .lists
                        .get_mut(list)
                        .unwrap()
                        .as_array_mut()
                        .unwrap();
                    let mut duplicate = items[0].clone();
                    if copied_index {
                        duplicate["index"] = json!(99);
                    }
                    items.push(duplicate);
                }
                assert!(fixture.device.discover().is_err(), "duplicate {list}");
                fixture.assert_no_writes();
            }
        }
    }

    #[test]
    fn owned_legacy_microphone_conflicts_with_native_registration() {
        let fixture = Fixture::new();
        {
            let mut inventory = fixture.runner.0.lock().unwrap();
            inventory.lists.get_mut("sink-inputs").unwrap().as_array_mut().unwrap().push(json!({
                "index": 50, "owner_module": "10", "sink": 8,
                "channel_map": "mono", "volume": {"mono": {"value": 0}},
                "properties": {"media.name": MICROPHONE_ORIGINAL_STREAM, "translator.owner": "true"},
            }));
            inventory.lists.get_mut("source-outputs").unwrap().as_array_mut().unwrap().push(json!({
                "owner_module": "10", "source": 4,
                "properties": {"media.name": MICROPHONE_ORIGINAL_STREAM, "translator.owner": "true"},
            }));
        }
        assert!(fixture.device.discover().is_err());
        fixture.assert_no_writes();
        fixture.registry.test_clear();
        {
            let mut inventory = fixture.runner.0.lock().unwrap();
            for list in ["sink-inputs", "source-outputs"] {
                inventory
                    .lists
                    .get_mut(list)
                    .unwrap()
                    .as_array_mut()
                    .unwrap()
                    .remove(0);
            }
        }
        assert!(
            PulseAudioMix::new(fixture.runner.clone())
                .discover()
                .is_ok(),
            "legacy-only adapter retains its own capability"
        );
        assert!(
            fixture.device.discover().is_err(),
            "native production adapter cannot admit legacy raw microphone without a live lease"
        );
        fixture.assert_no_writes();
    }

    #[test]
    fn native_sample_format_is_owned_by_typed_transport_not_display_text() {
        let fixture = Fixture::new();
        for list in ["sink-inputs", "source-outputs"] {
            fixture.change(list, "/0/sample_specification", json!("(null)"));
        }
        let plan = fixture.device.discover().unwrap();
        fixture
            .device
            .set_percent(&plan.entries()[0], MixPercent::try_from(35).unwrap())
            .unwrap();
        fixture.registration.test_invalidate();
        assert!(
            fixture.device.discover().is_err(),
            "display text cannot replace a live typed lease"
        );
    }

    #[test]
    fn replacement_or_removal_after_discovery_prevents_gain_and_rollback_writes() {
        for rollback in [false, true] {
            for list in ["sink-inputs", "source-outputs", "clients"] {
                let fixture = Fixture::new();
                let plan = fixture.device.discover().unwrap();
                fixture.change(list, "/0/properties/application.process.id", json!("9999"));
                let entry = &plan.entries()[0];
                let result = if rollback {
                    fixture.device.restore_raw(entry)
                } else {
                    fixture
                        .device
                        .set_percent(entry, MixPercent::try_from(35).unwrap())
                };
                assert!(result.is_err());
                fixture.assert_no_writes();
            }
            let fixture = Fixture::new();
            let plan = fixture.device.discover().unwrap();
            fixture
                .runner
                .0
                .lock()
                .unwrap()
                .lists
                .insert("sink-inputs", json!([]));
            let entry = &plan.entries()[0];
            let result = if rollback {
                fixture.device.restore_raw(entry)
            } else {
                fixture
                    .device
                    .set_percent(entry, MixPercent::try_from(35).unwrap())
            };
            assert!(result.is_err());
            fixture.assert_no_writes();
        }
    }

    #[test]
    fn expired_or_replaced_lease_prevents_gain_rollback_and_zero_acknowledgement() {
        for replacement in [None, Some(Uuid::from_u128(124)), Some(Uuid::from_u128(123))] {
            let fixture = Fixture::new();
            let plan = fixture.device.discover().unwrap();
            if let Some(session) = replacement {
                fixture.registry.test_register(
                    41,
                    42,
                    4,
                    8,
                    "translator_test_mic.monitor",
                    session,
                    1234,
                    6,
                );
                fixture
                    .runner
                    .0
                    .lock()
                    .unwrap()
                    .lists
                    .values_mut()
                    .for_each(|list| {
                        for item in list.as_array_mut().unwrap() {
                            if let Some(properties) = item.get_mut("properties") {
                                properties[SESSION_PROPERTY] = json!(session.to_string());
                            }
                        }
                    });
            } else {
                fixture.registration.test_invalidate();
            }
            let entry = &plan.entries()[0];
            assert!(
                fixture
                    .device
                    .set_percent(entry, MixPercent::try_from(35).unwrap())
                    .is_err()
            );
            assert!(fixture.device.restore_raw(entry).is_err());
            assert!(
                fixture
                    .device
                    .verify_zero_targets(&plan, &[AudioMixTarget::MicrophoneOriginal])
                    .is_err()
            );
            fixture.assert_no_writes();
        }
    }

    #[test]
    fn successful_write_without_gain_change_fails_zero_and_nonzero_readback() {
        for (initial, percent) in [(65_536, 0), (1, 0), (0, 35)] {
            let fixture = Fixture::new();
            fixture.change("sink-inputs", "/0/volume/mono/value", json!(initial));
            let plan = fixture.device.discover().unwrap();
            fixture.runner.0.lock().unwrap().apply_writes = false;
            assert!(
                fixture
                    .device
                    .set_percent(&plan.entries()[0], MixPercent::try_from(percent).unwrap())
                    .is_err()
            );
            assert_eq!(fixture.runner.0.lock().unwrap().writes.len(), 1);
            assert!(
                !fixture.registration.is_live(),
                "unverified gain must cancel raw microphone forwarding"
            );
            assert!(
                fixture.registry.current().is_err(),
                "join custody must remain retained"
            );
        }
        let fixture = Fixture::new();
        let plan = fixture.device.discover().unwrap();
        fixture.change("sink-inputs", "/0/volume/mono/value", json!(65_536));
        fixture.runner.0.lock().unwrap().apply_writes = false;
        assert!(fixture.device.restore_raw(&plan.entries()[0]).is_err());
        assert_eq!(fixture.runner.0.lock().unwrap().writes.len(), 1);
        assert!(
            !fixture.registration.is_live(),
            "failed compensation must cancel raw microphone forwarding"
        );
    }

    #[test]
    fn identity_change_after_write_rejects_acknowledgement_and_blocks_rollback() {
        let fixture = Fixture::new();
        let plan = fixture.device.discover().unwrap();
        fixture.runner.0.lock().unwrap().replace_after_write = true;
        let entry = &plan.entries()[0];
        assert!(
            fixture
                .device
                .set_percent(entry, MixPercent::try_from(35).unwrap())
                .is_err()
        );
        assert!(fixture.device.restore_raw(entry).is_err());
        assert_eq!(fixture.runner.0.lock().unwrap().writes.len(), 1);
    }
}
