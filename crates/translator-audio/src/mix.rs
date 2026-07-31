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
    const fn percent_from(self, volumes: AudioMixVolumes) -> u8 {
        match self {
            Self::MicrophoneOriginal => volumes.microphone_original_percent,
            Self::MicrophoneTranslation => volumes.microphone_translation_percent,
            Self::SpeakerOriginal => volumes.speaker_original_percent,
            Self::SpeakerTranslation => volumes.speaker_translation_percent,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioMixApplyReport {
    pub updated_targets: Vec<AudioMixTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioMixErrorCode {
    DiscoveryFailed,
    VolumeApplyFailed,
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

    pub fn apply(&self, volumes: AudioMixVolumes) -> Result<AudioMixApplyReport, AudioMixError> {
        let sink_inputs: Vec<RawSinkInput> =
            self.run_json(&["--format=json", "list", "sink-inputs"])?;
        let source_outputs: Vec<RawSourceOutput> =
            self.run_json(&["--format=json", "list", "source-outputs"])?;
        let remote_loopback_modules = remote_loopback_modules(&source_outputs);
        let mut updated_targets = Vec::new();

        for input in sink_inputs {
            let Some(target) = classify_sink_input(&input, &remote_loopback_modules) else {
                continue;
            };
            self.set_sink_input_volume(input.index, target.percent_from(volumes))?;
            updated_targets.push(target);
        }

        Ok(AudioMixApplyReport { updated_targets })
    }

    fn set_sink_input_volume(&self, index: u32, percent: u8) -> Result<(), AudioMixError> {
        let args = [
            "set-sink-input-volume".to_owned(),
            index.to_string(),
            format!("{percent}%"),
        ];
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
        CommandRunError::NotFound | CommandRunError::SpawnFailed | CommandRunError::TimedOut => {
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
    properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawSourceOutput {
    #[serde(default)]
    properties: HashMap<String, String>,
}
