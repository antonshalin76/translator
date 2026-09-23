use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::Deserialize;

use crate::{CommandRunError, CommandRunner, MIC_OUT_SINK, REMOTE_IN_SINK, SystemCommandRunner};

pub const OUTGOING_TRANSLATION_STREAM: &str = "translator-outgoing-playback";
pub const INCOMING_TRANSLATION_STREAM: &str = "translator-incoming-playback";

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
        let remote_loopback_modules = remote_loopback_modules(&source_outputs);
        let mut entries = Vec::new();

        for input in sink_inputs {
            let Some(target) = classify_sink_input(&input, &remote_loopback_modules) else {
                continue;
            };
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

fn classify_sink_input(
    input: &RawSinkInput,
    remote_loopback_modules: &HashSet<String>,
) -> Option<AudioMixTarget> {
    let media_name = property(&input.properties, "media.name")?;
    let application_name = property(&input.properties, "application.name");
    if application_name == Some("translator-daemon") {
        return match media_name {
            OUTGOING_TRANSLATION_STREAM => Some(AudioMixTarget::MicrophoneTranslation),
            INCOMING_TRANSLATION_STREAM => Some(AudioMixTarget::SpeakerTranslation),
            _ => None,
        };
    }

    if !media_name.starts_with("loopback-") {
        return None;
    }
    let target_object = property(&input.properties, "target.object");
    if target_object == Some(MIC_OUT_SINK) {
        return Some(AudioMixTarget::MicrophoneOriginal);
    }
    property(&input.properties, "pulse.module.id")
        .filter(|module_id| remote_loopback_modules.contains(*module_id))
        .map(|_| AudioMixTarget::SpeakerOriginal)
}

fn remote_loopback_modules(source_outputs: &[RawSourceOutput]) -> HashSet<String> {
    source_outputs
        .iter()
        .filter(|output| {
            property(&output.properties, "media.name")
                .is_some_and(|name| name.starts_with("loopback-"))
                && property(&output.properties, "target.object") == Some(REMOTE_IN_SINK)
        })
        .filter_map(|output| property(&output.properties, "pulse.module.id").map(str::to_owned))
        .collect()
}

fn property<'a>(properties: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    properties.get(key).map(String::as_str)
}

#[derive(Debug, Deserialize)]
struct RawSinkInput {
    index: u32,
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
    #[serde(default)]
    properties: HashMap<String, String>,
}
