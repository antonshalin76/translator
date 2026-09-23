use std::sync::Mutex;

use axum::http::StatusCode;
use translator_audio::{AudioMixVolumes, CommandRunner, MixPercent, PulseAudioMix};

use crate::{AudioMixController, AudioMixState, ControlFailure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslationMixMode {
    Bypass,
    Translating,
}

impl TranslationMixMode {
    fn effective(self, desired: AudioMixState) -> AudioMixState {
        match self {
            Self::Translating => desired,
            Self::Bypass => AudioMixState {
                microphone_original_percent: 100,
                microphone_translation_percent: 0,
                speaker_original_percent: 100,
                speaker_translation_percent: 0,
            },
        }
    }
}

pub struct AudioMixApplication<R> {
    device: PulseAudioMix<R>,
    state: Mutex<MixState>,
}

struct MixState {
    committed: AudioMixState,
    unknown: bool,
}

impl<R: CommandRunner> AudioMixApplication<R> {
    pub fn new(runner: R) -> Self {
        Self {
            device: PulseAudioMix::new(runner),
            state: Mutex::new(MixState {
                committed: AudioMixState::default(),
                unknown: false,
            }),
        }
    }

    pub fn committed(&self) -> Result<AudioMixState, ControlFailure> {
        let state = self.state.lock().map_err(|_| unknown())?;
        if state.unknown {
            return Err(unknown());
        }
        Ok(state.committed)
    }

    fn reconcile(&self, mode: TranslationMixMode, recovery: bool) -> Result<(), ControlFailure> {
        let mut state = self.state.lock().map_err(|_| unknown())?;
        if state.unknown && !recovery {
            return Err(unknown());
        }
        let candidate = state.committed;
        self.apply_locked(&mut state, candidate, mode)
    }

    fn apply_locked(
        &self,
        state: &mut MixState,
        candidate: AudioMixState,
        mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        for value in [
            candidate.microphone_original_percent,
            candidate.microphone_translation_percent,
            candidate.speaker_original_percent,
            candidate.speaker_translation_percent,
        ] {
            MixPercent::try_from(value).map_err(|_| failure("invalid_audio_mix_volume"))?;
        }
        let effective = mode.effective(candidate);
        let volumes = AudioMixVolumes {
            microphone_original_percent: effective.microphone_original_percent,
            microphone_translation_percent: effective.microphone_translation_percent,
            speaker_original_percent: effective.speaker_original_percent,
            speaker_translation_percent: effective.speaker_translation_percent,
        };
        let plan = self.device.discover().map_err(|_| {
            if state.unknown {
                unknown()
            } else {
                failure("audio_mix_discovery_failed")
            }
        })?;
        for (index, entry) in plan.entries().iter().enumerate() {
            let percent = MixPercent::try_from(entry.target().percent_from(volumes))
                .map_err(|_| failure("invalid_audio_mix_volume"))?;
            if self.device.set_percent(entry, percent).is_err() {
                for attempted in plan.entries()[..=index].iter().rev() {
                    if self.device.restore_raw(attempted).is_err() {
                        state.unknown = true;
                    }
                }
                return Err(if state.unknown {
                    unknown()
                } else {
                    failure("audio_mix_apply_failed")
                });
            }
        }
        state.committed = candidate;
        state.unknown = false;
        Ok(())
    }
}

impl<R: CommandRunner + Send + Sync> AudioMixController for AudioMixApplication<R> {
    fn apply_desired(
        &self,
        volumes: AudioMixState,
        mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        let mut state = self.state.lock().map_err(|_| unknown())?;
        if state.unknown {
            return Err(unknown());
        }
        self.apply_locked(&mut state, volumes, mode)
    }

    fn reconcile_committed(&self, mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.reconcile(mode, false)
    }

    fn recover_committed(&self, mode: TranslationMixMode) -> Result<(), ControlFailure> {
        self.reconcile(mode, true)
    }
}

fn failure(code: &'static str) -> ControlFailure {
    ControlFailure {
        status: StatusCode::CONFLICT,
        code,
    }
}

fn unknown() -> ControlFailure {
    failure("audio_mix_state_unknown")
}
