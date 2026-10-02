use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use axum::http::StatusCode;
use translator_audio::{
    AudioMixTarget, AudioMixVolumes, CommandRunner, INCOMING_TRANSLATION_STREAM, MixPercent,
    OUTGOING_TRANSLATION_STREAM, OriginalMicrophoneRegistry, PulseAudioMix,
    PulsePlaybackRegistration,
};

use crate::translation_runtime::{PlaybackMixAuthority, PlaybackRegistrationPhase};
use crate::{AudioMixController, AudioMixState, ControlFailure, DuplexRuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslationMixMode {
    Bypass,
    MicrophoneMutedBypass,
    Quarantine { mic_original_expected: bool },
    Translating,
    TranslatingMicrophoneMuted,
}

impl TranslationMixMode {
    fn effective(self, desired: AudioMixState) -> AudioMixState {
        match self {
            Self::Translating => desired,
            Self::TranslatingMicrophoneMuted => AudioMixState {
                microphone_original_percent: 0,
                microphone_translation_percent: 0,
                ..desired
            },
            Self::Quarantine { .. } => AudioMixState {
                microphone_original_percent: 0,
                microphone_translation_percent: 0,
                speaker_original_percent: 0,
                speaker_translation_percent: 0,
            },
            Self::Bypass => AudioMixState {
                microphone_original_percent: 100,
                microphone_translation_percent: 0,
                speaker_original_percent: 100,
                speaker_translation_percent: 0,
            },
            Self::MicrophoneMutedBypass => AudioMixState {
                microphone_original_percent: 0,
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
    mode: TranslationMixMode,
    unknown: bool,
}

impl<R: CommandRunner> AudioMixApplication<R> {
    pub fn new(runner: R) -> Self {
        Self::with_device(PulseAudioMix::new(runner))
    }

    pub fn with_original_microphone(runner: R, registry: OriginalMicrophoneRegistry) -> Self {
        Self::with_device(PulseAudioMix::with_original_microphone(runner, registry))
    }

    fn with_device(device: PulseAudioMix<R>) -> Self {
        Self {
            device,
            state: Mutex::new(MixState {
                committed: AudioMixState::default(),
                mode: TranslationMixMode::Bypass,
                unknown: false,
            }),
        }
    }

    pub fn committed(&self) -> Result<AudioMixState, ControlFailure> {
        let state = self.lock_state()?;
        if state.unknown {
            return Err(unknown());
        }
        Ok(state.committed)
    }

    fn reconcile(&self, mode: TranslationMixMode, recovery: bool) -> Result<(), ControlFailure> {
        let mut state = self.lock_state()?;
        if state.unknown && !recovery {
            return Err(unknown());
        }
        let candidate = state.committed;
        self.apply_locked(&mut state, candidate, mode, false)
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, MixState>, ControlFailure> {
        self.state.lock().map_err(|_| {
            self.device.quarantine_original_microphone();
            unknown()
        })
    }

    fn mark_unknown(&self, state: &mut MixState) {
        self.device.quarantine_original_microphone();
        state.unknown = true;
    }

    fn discovery_failure(&self, state: &mut MixState) -> ControlFailure {
        if self.device.quarantine_original_microphone() {
            state.unknown = true;
        }
        if state.unknown {
            unknown()
        } else {
            failure("audio_mix_discovery_failed")
        }
    }

    fn apply_locked(
        &self,
        state: &mut MixState,
        candidate: AudioMixState,
        mode: TranslationMixMode,
        require_candidate_targets: bool,
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
        let plan = self
            .device
            .discover()
            .map_err(|_| self.discovery_failure(state))?;
        if effective.microphone_original_percent > 0
            && (require_candidate_targets || mode == TranslationMixMode::Translating)
            && candidate.microphone_original_percent > 0
            && !plan
                .entries()
                .iter()
                .any(|entry| entry.target() == AudioMixTarget::MicrophoneOriginal)
        {
            return Err(failure("microphone_original_unavailable"));
        }
        if matches!(
            mode,
            TranslationMixMode::Quarantine {
                mic_original_expected: true
            }
        ) && !plan
            .entries()
            .iter()
            .any(|entry| entry.target() == AudioMixTarget::MicrophoneOriginal)
        {
            self.mark_unknown(state);
            return Err(unknown());
        }
        let mut emergency_mute_failed = false;
        for (index, entry) in plan.entries().iter().enumerate() {
            let percent = MixPercent::try_from(entry.target().percent_from(volumes))
                .map_err(|_| failure("invalid_audio_mix_volume"))?;
            if self.device.set_percent(entry, percent).is_err() {
                if matches!(
                    mode,
                    TranslationMixMode::Quarantine { .. }
                        | TranslationMixMode::MicrophoneMutedBypass
                        | TranslationMixMode::TranslatingMicrophoneMuted
                ) {
                    emergency_mute_failed = true;
                    continue;
                }
                for attempted in plan.entries()[..=index].iter().rev() {
                    if self.device.restore_raw(attempted).is_err() {
                        self.mark_unknown(state);
                    }
                }
                return Err(if state.unknown {
                    unknown()
                } else {
                    failure("audio_mix_apply_failed")
                });
            }
        }
        if emergency_mute_failed {
            self.mark_unknown(state);
            return Err(unknown());
        }
        let zero_targets: Vec<_> = [
            AudioMixTarget::MicrophoneOriginal,
            AudioMixTarget::MicrophoneTranslation,
            AudioMixTarget::SpeakerOriginal,
            AudioMixTarget::SpeakerTranslation,
        ]
        .into_iter()
        .filter(|target| {
            (matches!(
                mode,
                TranslationMixMode::Quarantine { .. }
                    | TranslationMixMode::MicrophoneMutedBypass
                    | TranslationMixMode::TranslatingMicrophoneMuted
            ) || *target == AudioMixTarget::MicrophoneOriginal)
                && target.percent_from(volumes) == 0
                && plan.entries().iter().any(|entry| entry.target() == *target)
        })
        .collect();
        if !zero_targets.is_empty()
            && self
                .device
                .verify_zero_targets(&plan, &zero_targets)
                .is_err()
        {
            self.mark_unknown(state);
            return Err(unknown());
        }
        state.committed = candidate;
        state.mode = mode;
        state.unknown = false;
        Ok(())
    }
}

impl<R: CommandRunner + Send + Sync> PlaybackMixAuthority for AudioMixApplication<R> {
    fn native_playback_percent(&self, original: bool) -> Result<u8, DuplexRuntimeError> {
        let state = self
            .lock_state()
            .map_err(|_| DuplexRuntimeError::StartFailed)?;
        if state.unknown {
            return Err(DuplexRuntimeError::StartFailed);
        }
        let effective = state.mode.effective(state.committed);
        let percent = if original {
            effective.speaker_original_percent
        } else {
            effective.speaker_translation_percent
        };
        if percent > 100 {
            return Err(DuplexRuntimeError::StartFailed);
        }
        Ok(percent)
    }

    fn admit_registered(
        &self,
        registration: &PulsePlaybackRegistration,
        phase: PlaybackRegistrationPhase,
        deadline: Instant,
    ) -> Result<(), DuplexRuntimeError> {
        let mut state = self
            .lock_state()
            .map_err(|_| DuplexRuntimeError::StartFailed)?;
        if state.unknown {
            return Err(DuplexRuntimeError::StartFailed);
        }
        let result = match phase {
            PlaybackRegistrationPhase::StartMuted => {
                self.device.verify_registered_zero(registration, deadline)
            }
            PlaybackRegistrationPhase::Running => {
                if state.mode != TranslationMixMode::Translating
                    && !(state.mode == TranslationMixMode::TranslatingMicrophoneMuted
                        && registration.stream_name() == INCOMING_TRANSLATION_STREAM)
                {
                    return Err(DuplexRuntimeError::StartFailed);
                }
                let target = match registration.stream_name() {
                    OUTGOING_TRANSLATION_STREAM => AudioMixTarget::MicrophoneTranslation,
                    INCOMING_TRANSLATION_STREAM => AudioMixTarget::SpeakerTranslation,
                    _ => return Err(DuplexRuntimeError::StartFailed),
                };
                let effective = state.mode.effective(state.committed);
                let volumes = AudioMixVolumes {
                    microphone_original_percent: effective.microphone_original_percent,
                    microphone_translation_percent: effective.microphone_translation_percent,
                    speaker_original_percent: effective.speaker_original_percent,
                    speaker_translation_percent: effective.speaker_translation_percent,
                };
                let percent = MixPercent::try_from(target.percent_from(volumes))
                    .map_err(|_| DuplexRuntimeError::StartFailed)?;
                self.device
                    .admit_registered_percent(registration, percent, deadline)
            }
        };
        if result.is_err() {
            self.mark_unknown(&mut state);
            return Err(DuplexRuntimeError::StartFailed);
        }
        Ok(())
    }
}

impl<R: CommandRunner + Send + Sync> AudioMixController for AudioMixApplication<R> {
    fn validate_desired(&self, volumes: AudioMixState) -> Result<(), ControlFailure> {
        self.validate_desired_for_mode(volumes, TranslationMixMode::Translating)
    }

    fn validate_desired_for_mode(
        &self,
        volumes: AudioMixState,
        mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        let mut state = self.lock_state()?;
        if state.unknown {
            return Err(unknown());
        }
        if mode.effective(volumes).microphone_original_percent > 0 {
            let plan = self
                .device
                .discover()
                .map_err(|_| self.discovery_failure(&mut state))?;
            if !plan
                .entries()
                .iter()
                .any(|entry| entry.target() == AudioMixTarget::MicrophoneOriginal)
            {
                return Err(failure("microphone_original_unavailable"));
            }
        }
        Ok(())
    }

    fn apply_desired(
        &self,
        volumes: AudioMixState,
        mode: TranslationMixMode,
    ) -> Result<(), ControlFailure> {
        let mut state = self.lock_state()?;
        if state.unknown {
            return Err(unknown());
        }
        self.apply_locked(&mut state, volumes, mode, true)
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

#[cfg(test)]
mod tests {
    use super::*;

    struct NoCommands;
    impl CommandRunner for NoCommands {
        fn run_until(
            &self,
            _: &str,
            _: &[String],
            _: Instant,
        ) -> Result<translator_audio::CommandResult, translator_audio::CommandRunError> {
            panic!("native gain projection must not execute a Pulse or ALSA command")
        }
    }

    #[test]
    fn native_gain_obeys_existing_mix_mode_and_unknown_state() {
        let application = AudioMixApplication::new(NoCommands);
        for (mode, original, translation) in [
            (
                TranslationMixMode::Quarantine {
                    mic_original_expected: false,
                },
                0,
                0,
            ),
            (TranslationMixMode::Translating, 0, 47),
            (TranslationMixMode::MicrophoneMutedBypass, 100, 0),
        ] {
            let mut state = application.state.lock().unwrap();
            state.committed.speaker_original_percent = 0;
            state.committed.speaker_translation_percent = 47;
            state.mode = mode;
            drop(state);
            assert_eq!(application.native_playback_percent(true).unwrap(), original);
            assert_eq!(
                application.native_playback_percent(false).unwrap(),
                translation
            );
        }
        application.state.lock().unwrap().unknown = true;
        assert!(application.native_playback_percent(true).is_err());
        assert!(application.native_playback_percent(false).is_err());
    }
}
