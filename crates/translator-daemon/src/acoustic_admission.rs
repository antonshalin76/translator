use std::{sync::Arc, time::Instant};

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use translator_audio::{
    AEC_SINK, AEC_SOURCE, AecCapability, AudioGraphState, DeviceFacts, DeviceHealth,
    DeviceSelectionState, GraphHealth, MIC_OUT_SINK, OutputMode, REMOTE_IN_SINK, RoutingState,
};
use translator_core::{AudioDirection, ProviderId};

use crate::{AecStartReservation, ControlFailure, RuntimeSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticWarning {
    DeviceUnavailable,
    AecNotValidated,
    AecValidationFailed,
    UnknownOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcousticSafety {
    pub mode: OutputMode,
    pub aec_capability: AecCapability,
    pub full_duplex_allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<AcousticWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceState {
    pub source: DeviceSelectionState,
    pub sink: DeviceSelectionState,
    pub acoustic: AcousticSafety,
}

impl From<DeviceFacts> for DeviceState {
    fn from(facts: DeviceFacts) -> Self {
        let warning = admit_acoustic(
            EnabledDirections {
                microphone: true,
                speaker: true,
            },
            &facts,
        )
        .err()
        .map(|error| match error {
            AcousticAdmissionError::AecNotValidated => AcousticWarning::AecNotValidated,
            AcousticAdmissionError::AecValidationFailed => AcousticWarning::AecValidationFailed,
            AcousticAdmissionError::UnknownOutput => AcousticWarning::UnknownOutput,
            AcousticAdmissionError::NoDirectionEnabled
            | AcousticAdmissionError::DeviceUnavailable
            | AcousticAdmissionError::InvalidPhysicalSelection => {
                AcousticWarning::DeviceUnavailable
            }
        });
        Self {
            source: facts.source,
            sink: facts.sink,
            acoustic: AcousticSafety {
                mode: facts.output_mode,
                aec_capability: facts.aec_capability,
                full_duplex_allowed: warning.is_none(),
                warning,
            },
        }
    }
}

pub struct RuntimeFacts {
    pub devices: DeviceFacts,
    pub audio_graph: AudioGraphState,
    pub routes: RoutingState,
}

pub trait RuntimeFactsSource: Send + Sync {
    fn inspect(&self, deadline: Instant) -> Result<RuntimeFacts, FactsError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactsError {
    DiscoveryFailed,
    Busy,
    Expired,
    InvalidPhysicalDevice,
    SinkValidationFailed,
}

impl FactsError {
    pub(crate) fn translation_failure(self) -> ControlFailure {
        self.failure("translation_precondition_failed")
    }

    pub(crate) fn round_trip_failure(self) -> ControlFailure {
        self.failure("self_test_precondition_failed")
    }

    fn failure(self, precondition: &'static str) -> ControlFailure {
        let code = match self {
            Self::DiscoveryFailed => "audio_facts_unavailable",
            Self::Busy => "audio_facts_busy",
            Self::Expired => "audio_facts_expired",
            Self::InvalidPhysicalDevice | Self::SinkValidationFailed => {
                return conflict(precondition);
            }
        };
        ControlFailure {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AcousticAdmissionError {
    NoDirectionEnabled,
    DeviceUnavailable,
    InvalidPhysicalSelection,
    UnknownOutput,
    AecNotValidated,
    AecValidationFailed,
}

impl AcousticAdmissionError {
    fn translation_failure(self) -> ControlFailure {
        conflict(if self == Self::NoDirectionEnabled {
            "no_direction_enabled"
        } else {
            "translation_precondition_failed"
        })
    }

    fn round_trip_failure(self) -> ControlFailure {
        conflict(match self {
            Self::NoDirectionEnabled => "no_direction_enabled",
            Self::DeviceUnavailable | Self::InvalidPhysicalSelection => {
                "self_test_precondition_failed"
            }
            Self::UnknownOutput | Self::AecNotValidated | Self::AecValidationFailed => {
                "self_test_headphones_required"
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EnabledDirections {
    pub microphone: bool,
    pub speaker: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirectionAudioTargets {
    pub capture: String,
    pub playback: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DuplexAudioTargets {
    pub microphone: Option<DirectionAudioTargets>,
    pub speaker: Option<DirectionAudioTargets>,
}

pub struct AdmittedDuplex {
    snapshot: RuntimeSnapshot,
    targets: DuplexAudioTargets,
    aec_reservation: Option<Arc<AecStartReservation>>,
}

impl AdmittedDuplex {
    pub fn snapshot(&self) -> &RuntimeSnapshot {
        &self.snapshot
    }

    pub(crate) fn requires_aec_authority(&self) -> bool {
        self.aec_reservation.is_some()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        RuntimeSnapshot,
        DuplexAudioTargets,
        Option<Arc<AecStartReservation>>,
    ) {
        (self.snapshot, self.targets, self.aec_reservation)
    }
}

pub(crate) fn enabled_directions(
    snapshot: &RuntimeSnapshot,
) -> Result<EnabledDirections, ControlFailure> {
    let enabled = |direction| {
        let mut states = snapshot
            .directions
            .iter()
            .filter(|state| state.direction_id == direction);
        let first = states
            .next()
            .ok_or_else(|| conflict("translation_precondition_failed"))?;
        if states.next().is_some() {
            return Err(conflict("translation_precondition_failed"));
        }
        Ok(first.enabled)
    };
    let result = EnabledDirections {
        microphone: enabled(AudioDirection::Microphone)?,
        speaker: enabled(AudioDirection::Speaker)?,
    };
    if !result.microphone && !result.speaker {
        return Err(conflict("no_direction_enabled"));
    }
    Ok(result)
}

fn physical(selection: &DeviceSelectionState) -> Result<&str, AcousticAdmissionError> {
    let selected = selection
        .selected
        .as_ref()
        .filter(|device| device.available && selection.health == DeviceHealth::Available)
        .ok_or(AcousticAdmissionError::DeviceUnavailable)?;
    if selected.name.trim().is_empty()
        || selected.name.starts_with("translator_")
        || selected.name.ends_with(".monitor")
        || selection.pinned_name.as_deref() != Some(&selected.name)
    {
        return Err(AcousticAdmissionError::InvalidPhysicalSelection);
    }
    Ok(&selected.name)
}

fn admit_acoustic(
    enabled: EnabledDirections,
    facts: &DeviceFacts,
) -> Result<DuplexAudioTargets, AcousticAdmissionError> {
    admit_acoustic_with_reservation(enabled, facts, None)
}

fn admit_acoustic_with_reservation(
    enabled: EnabledDirections,
    facts: &DeviceFacts,
    reservation: Option<&AecStartReservation>,
) -> Result<DuplexAudioTargets, AcousticAdmissionError> {
    if !enabled.microphone && !enabled.speaker {
        return Err(AcousticAdmissionError::NoDirectionEnabled);
    }
    let sink = physical(&facts.sink)?;
    let (capture, playback) = if enabled.microphone {
        let source = physical(&facts.source)?;
        match facts.output_mode {
            OutputMode::Headphones => (Some(source), sink),
            OutputMode::UnknownUnsafe => return Err(AcousticAdmissionError::UnknownOutput),
            OutputMode::OpenSpeaker => match &facts.aec_capability {
                AecCapability::ValidatedFor {
                    source_name,
                    sink_name,
                } if source_name == source
                    && sink_name == sink
                    && reservation
                        .is_some_and(|reservation| reservation.authorizes_pair(source, sink)) =>
                {
                    (Some(AEC_SOURCE), AEC_SINK)
                }
                AecCapability::ValidationFailed => {
                    return Err(AcousticAdmissionError::AecValidationFailed);
                }
                _ => return Err(AcousticAdmissionError::AecNotValidated),
            },
        }
    } else {
        (None, sink)
    };
    Ok(DuplexAudioTargets {
        microphone: capture.map(|source| DirectionAudioTargets {
            capture: source.into(),
            playback: MIC_OUT_SINK.into(),
        }),
        speaker: enabled.speaker.then(|| DirectionAudioTargets {
            capture: format!("{REMOTE_IN_SINK}.monitor"),
            playback: playback.into(),
        }),
    })
}

#[cfg(test)]
pub(crate) fn admit_translation(
    candidate: RuntimeSnapshot,
    facts: RuntimeFacts,
) -> Result<AdmittedDuplex, ControlFailure> {
    admit_translation_with_reservation(candidate, facts, None)
}

pub(crate) fn admit_translation_with_reservation(
    mut candidate: RuntimeSnapshot,
    facts: RuntimeFacts,
    reservation: Option<Arc<AecStartReservation>>,
) -> Result<AdmittedDuplex, ControlFailure> {
    let enabled = enabled_directions(&candidate)?;
    let targets = admit_acoustic_with_reservation(enabled, &facts.devices, reservation.as_deref())
        .map_err(AcousticAdmissionError::translation_failure)?;
    candidate.devices = Some(facts.devices.into());
    candidate.audio_graph = Some(facts.audio_graph);
    candidate.routes = Some(facts.routes);
    validate_configuration(&candidate)?;
    Ok(AdmittedDuplex {
        snapshot: candidate,
        targets,
        aec_reservation: reservation,
    })
}

fn validate_configuration(snapshot: &RuntimeSnapshot) -> Result<(), ControlFailure> {
    if !matches!(snapshot.provider_id, ProviderId::Local | ProviderId::Openai)
        || (snapshot.provider_id == ProviderId::Openai && !snapshot.audio_leaves_machine)
        || snapshot
            .audio_graph
            .as_ref()
            .is_none_or(|graph| graph.health != GraphHealth::Ready)
    {
        return Err(conflict("translation_precondition_failed"));
    }
    for state in &snapshot.directions {
        if snapshot
            .latency_policy
            .iter()
            .filter(|policy| policy.direction_id == state.direction_id)
            .count()
            != 1
            || state.voice_profile.has_overrides()
            || (state.enabled
                && (state.source_language == state.target_language
                    || state.voice_profile.language != state.target_language))
        {
            return Err(conflict("translation_precondition_failed"));
        }
    }
    Ok(())
}

pub(crate) fn admit_round_trip(
    mut candidate: RuntimeSnapshot,
    facts: RuntimeFacts,
) -> Result<AdmittedDuplex, ControlFailure> {
    for direction in &mut candidate.directions {
        direction.enabled = true;
    }
    let enabled =
        enabled_directions(&candidate).map_err(|_| conflict("self_test_precondition_failed"))?;
    let targets = admit_acoustic(enabled, &facts.devices)
        .map_err(AcousticAdmissionError::round_trip_failure)?;
    candidate.devices = Some(facts.devices.into());
    candidate.audio_graph = Some(facts.audio_graph);
    candidate.routes = Some(facts.routes);
    crate::round_trip::validate_preconditions(crate::round_trip_runtime::round_trip_preconditions(
        &candidate,
    ))
    .map_err(crate::round_trip_runtime::map_precondition_error)?;
    validate_configuration(&candidate).map_err(|_| conflict("self_test_precondition_failed"))?;
    Ok(AdmittedDuplex {
        snapshot: candidate,
        targets,
        aec_reservation: None,
    })
}

fn conflict(code: &'static str) -> ControlFailure {
    ControlFailure {
        status: StatusCode::CONFLICT,
        code,
    }
}

pub(crate) fn admit_task7(
    snapshot: RuntimeSnapshot,
    facts: translator_audio::Task7EndpointFacts,
    _lease: &crate::RuntimeLease,
    _graph: &impl translator_audio::AudioGraph,
) -> Result<AdmittedDuplex, ControlFailure> {
    let enabled = enabled_directions(&snapshot)?;
    if !enabled.microphone
        || !enabled.speaker
        || facts.input_monitor != "translator_task7_ru_in.monitor"
    {
        return Err(conflict("translation_precondition_failed"));
    }
    let selection = DeviceSelectionState {
        health: DeviceHealth::Available,
        pinned_name: Some(facts.output.name.clone()),
        selected: Some(facts.output),
        current_default: None,
        pending_default: None,
    };
    let output = physical(&selection).map_err(AcousticAdmissionError::translation_failure)?;
    validate_configuration(&snapshot)?;
    Ok(AdmittedDuplex {
        snapshot,
        targets: DuplexAudioTargets {
            microphone: Some(DirectionAudioTargets {
                capture: facts.input_monitor,
                playback: MIC_OUT_SINK.into(),
            }),
            speaker: Some(DirectionAudioTargets {
                capture: format!("{REMOTE_IN_SINK}.monitor"),
                playback: output.into(),
            }),
        },
        aec_reservation: None,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use translator_audio::PhysicalDevice;

    fn selection(name: &str) -> DeviceSelectionState {
        DeviceSelectionState {
            health: DeviceHealth::Available,
            selected: Some(PhysicalDevice {
                id: 1,
                name: name.into(),
                description: name.into(),
                active_port: None,
                active_port_type: None,
                available: true,
            }),
            pinned_name: Some(name.into()),
            current_default: Some(name.into()),
            pending_default: None,
        }
    }

    fn facts(mode: OutputMode, aec_capability: AecCapability) -> DeviceFacts {
        DeviceFacts {
            source: selection("alsa_input.physical"),
            sink: selection("alsa_output.physical"),
            output_mode: mode,
            aec_capability,
        }
    }

    fn enabled(microphone: bool, speaker: bool) -> EnabledDirections {
        EnabledDirections {
            microphone,
            speaker,
        }
    }

    fn exact_aec() -> AecCapability {
        AecCapability::ValidatedFor {
            source_name: "alsa_input.physical".into(),
            sink_name: "alsa_output.physical".into(),
        }
    }

    fn exact_reservation() -> AecStartReservation {
        AecStartReservation::for_test("alsa_input.physical", "alsa_output.physical")
    }

    pub(crate) fn ready_facts() -> RuntimeFacts {
        RuntimeFacts {
            devices: facts(OutputMode::Headphones, AecCapability::Unavailable),
            audio_graph: AudioGraphState {
                health: GraphHealth::Ready,
                endpoints: Vec::new(),
                owned_module_ids: Vec::new(),
                safe_error: None,
            },
            routes: RoutingState {
                candidates: Vec::new(),
                source_outputs: Vec::new(),
                conflicting_stream_ids: Vec::new(),
                active_route: None,
                resolution: translator_audio::RouteResolution::NoCandidate,
            },
        }
    }

    #[test]
    fn voice_override_latent_and_enabled_snapshots_cannot_obtain_native_capability() {
        let mut accepted = Vec::new();
        for provider in [ProviderId::Local, ProviderId::Openai] {
            for disabled in [false, true] {
                for value in ["unapproved-voice", "", " \t"] {
                    for (model, voice) in [(true, false), (false, true), (true, true)] {
                        let mut candidate = RuntimeSnapshot {
                            provider_id: provider,
                            audio_leaves_machine: provider == ProviderId::Openai,
                            ..RuntimeSnapshot::default()
                        };
                        let direction = &mut candidate.directions[1];
                        direction.enabled = !disabled;
                        direction.voice_profile.model_path = model.then(|| value.to_owned());
                        direction.voice_profile.provider_voice_id = voice.then(|| value.to_owned());
                        match admit_translation(candidate, ready_facts()) {
                            Ok(_) => accepted.push((provider, disabled, model, voice, value.len())),
                            Err(error) => {
                                assert_eq!(error.status, StatusCode::CONFLICT);
                                assert_eq!(error.code, "translation_precondition_failed");
                            }
                        }
                    }
                }
            }
        }
        assert!(
            accepted.is_empty(),
            "unsupported native capabilities: {accepted:?}"
        );
    }

    #[test]
    fn voice_override_keeps_both_disabled_first_and_builtin_admission_characterization() {
        let mut disabled = RuntimeSnapshot::default();
        for direction in &mut disabled.directions {
            direction.enabled = false;
            direction.voice_profile.provider_voice_id = Some("unapproved-voice".into());
        }
        let mut unavailable = ready_facts();
        unavailable.devices.source.selected = None;
        let error = admit_translation(disabled, unavailable).err().unwrap();
        assert_eq!(error.code, "no_direction_enabled");
        for provider in [ProviderId::Local, ProviderId::Openai] {
            let mut candidate = RuntimeSnapshot {
                provider_id: provider,
                audio_leaves_machine: provider == ProviderId::Openai,
                ..RuntimeSnapshot::default()
            };
            candidate.directions[1].voice_profile.gender = translator_core::VoiceGender::Female;
            let admitted = admit_translation(candidate, ready_facts()).unwrap();
            assert_eq!(
                admitted.snapshot().directions[1].voice_profile.gender,
                translator_core::VoiceGender::Female
            );
        }
    }

    #[test]
    fn round_trip_structural_direction_errors_keep_self_test_family() {
        let outcomes: Vec<_> = [false, true]
            .into_iter()
            .map(|duplicate| {
                let mut candidate = RuntimeSnapshot::default();
                if duplicate {
                    candidate.directions.push(candidate.directions[0].clone());
                } else {
                    candidate.directions.pop();
                }
                let error = admit_round_trip(candidate, ready_facts()).err().unwrap();
                (error.status.as_u16(), error.code)
            })
            .collect();
        assert_eq!(outcomes, vec![(409, "self_test_precondition_failed"); 2]);
    }

    #[test]
    fn complete_admission_rejects_invalid_configuration_before_native_conversion() {
        for fault in [
            "language",
            "voice",
            "missing_direction",
            "duplicate_direction",
            "missing_policy",
            "duplicate_policy",
            "both_disabled",
            "cloud_without_egress",
            "missing_microphone",
            "empty_sink",
        ] {
            let mut candidate = RuntimeSnapshot::default();
            let mut facts = ready_facts();
            match fault {
                "language" => {
                    candidate.directions[0].source_language =
                        candidate.directions[0].target_language
                }
                "voice" => {
                    candidate.directions[0].voice_profile.language =
                        candidate.directions[0].source_language
                }
                "missing_direction" => {
                    candidate.directions.pop();
                }
                "duplicate_direction" => candidate.directions.push(candidate.directions[0].clone()),
                "missing_policy" => {
                    candidate.latency_policy.pop();
                }
                "duplicate_policy" => candidate
                    .latency_policy
                    .push(candidate.latency_policy[0].clone()),
                "both_disabled" => candidate
                    .directions
                    .iter_mut()
                    .for_each(|direction| direction.enabled = false),
                "cloud_without_egress" => candidate.provider_id = ProviderId::Openai,
                "missing_microphone" => facts.devices.source.selected = None,
                "empty_sink" => facts.devices.sink.selected.as_mut().unwrap().name.clear(),
                _ => unreachable!(),
            }
            let failure = admit_translation(candidate, facts).err().expect(fault);
            assert_eq!(failure.status, StatusCode::CONFLICT, "{fault}");
            assert_eq!(
                failure.code,
                if fault == "both_disabled" {
                    "no_direction_enabled"
                } else {
                    "translation_precondition_failed"
                },
                "{fault}"
            );
        }
        let cloud = RuntimeSnapshot {
            provider_id: ProviderId::Openai,
            audio_leaves_machine: true,
            ..RuntimeSnapshot::default()
        };
        assert!(admit_translation(cloud, ready_facts()).is_ok());
        assert!(admit_translation(RuntimeSnapshot::default(), ready_facts()).is_ok());
    }

    #[test]
    fn error_mapping_preserves_unavailable_vs_known_negative_for_both_consumers() {
        for (error, status, translation, round_trip) in [
            (
                FactsError::DiscoveryFailed,
                StatusCode::SERVICE_UNAVAILABLE,
                "audio_facts_unavailable",
                "audio_facts_unavailable",
            ),
            (
                FactsError::Busy,
                StatusCode::SERVICE_UNAVAILABLE,
                "audio_facts_busy",
                "audio_facts_busy",
            ),
            (
                FactsError::Expired,
                StatusCode::SERVICE_UNAVAILABLE,
                "audio_facts_expired",
                "audio_facts_expired",
            ),
            (
                FactsError::InvalidPhysicalDevice,
                StatusCode::CONFLICT,
                "translation_precondition_failed",
                "self_test_precondition_failed",
            ),
            (
                FactsError::SinkValidationFailed,
                StatusCode::CONFLICT,
                "translation_precondition_failed",
                "self_test_precondition_failed",
            ),
        ] {
            assert_eq!(
                (
                    error.translation_failure().status,
                    error.translation_failure().code
                ),
                (status, translation)
            );
            assert_eq!(
                (
                    error.round_trip_failure().status,
                    error.round_trip_failure().code
                ),
                (status, round_trip)
            );
        }
        for (error, translation, round_trip) in [
            (
                AcousticAdmissionError::NoDirectionEnabled,
                "no_direction_enabled",
                "no_direction_enabled",
            ),
            (
                AcousticAdmissionError::DeviceUnavailable,
                "translation_precondition_failed",
                "self_test_precondition_failed",
            ),
            (
                AcousticAdmissionError::InvalidPhysicalSelection,
                "translation_precondition_failed",
                "self_test_precondition_failed",
            ),
            (
                AcousticAdmissionError::UnknownOutput,
                "translation_precondition_failed",
                "self_test_headphones_required",
            ),
            (
                AcousticAdmissionError::AecNotValidated,
                "translation_precondition_failed",
                "self_test_headphones_required",
            ),
            (
                AcousticAdmissionError::AecValidationFailed,
                "translation_precondition_failed",
                "self_test_headphones_required",
            ),
        ] {
            assert_eq!(
                (
                    error.translation_failure().status,
                    error.translation_failure().code
                ),
                (StatusCode::CONFLICT, translation)
            );
            assert_eq!(
                (
                    error.round_trip_failure().status,
                    error.round_trip_failure().code
                ),
                (StatusCode::CONFLICT, round_trip)
            );
        }
    }

    #[test]
    fn headphone_matrix_keeps_physical_capture_and_optional_targets() {
        for aec in [AecCapability::Unavailable, exact_aec()] {
            for (microphone, speaker) in [(true, false), (false, true), (true, true)] {
                let result = admit_acoustic(
                    enabled(microphone, speaker),
                    &facts(OutputMode::Headphones, aec.clone()),
                )
                .unwrap();
                assert_eq!(
                    result.microphone,
                    microphone.then(|| DirectionAudioTargets {
                        capture: "alsa_input.physical".into(),
                        playback: MIC_OUT_SINK.into(),
                    })
                );
                assert_eq!(
                    result.speaker,
                    speaker.then(|| DirectionAudioTargets {
                        capture: format!("{REMOTE_IN_SINK}.monitor"),
                        playback: "alsa_output.physical".into(),
                    })
                );
            }
        }
    }

    #[test]
    fn incoming_only_ignores_microphone_and_acoustic_classification() {
        for mode in [
            OutputMode::Headphones,
            OutputMode::OpenSpeaker,
            OutputMode::UnknownUnsafe,
        ] {
            for aec in [
                AecCapability::Unavailable,
                AecCapability::ValidationFailed,
                exact_aec(),
            ] {
                for microphone_available in [false, true] {
                    let mut devices = facts(mode, aec.clone());
                    if !microphone_available {
                        devices.source.selected = None;
                        devices.source.health = DeviceHealth::DeviceUnavailable;
                        devices.source.pinned_name = None;
                    }
                    let result = admit_acoustic(enabled(false, true), &devices).unwrap();
                    assert_eq!(result.microphone, None);
                    assert_eq!(
                        result.speaker,
                        Some(DirectionAudioTargets {
                            capture: format!("{REMOTE_IN_SINK}.monitor"),
                            playback: "alsa_output.physical".into(),
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn open_speaker_requires_exact_measured_pair_for_outgoing() {
        for speaker in [false, true] {
            for (aec, error) in [
                (
                    AecCapability::Unavailable,
                    AcousticAdmissionError::AecNotValidated,
                ),
                (
                    AecCapability::AvailableUnvalidated,
                    AcousticAdmissionError::AecNotValidated,
                ),
                (
                    AecCapability::ValidationFailed,
                    AcousticAdmissionError::AecValidationFailed,
                ),
                (
                    AecCapability::ValidatedFor {
                        source_name: "other".into(),
                        sink_name: "alsa_output.physical".into(),
                    },
                    AcousticAdmissionError::AecNotValidated,
                ),
                (
                    AecCapability::ValidatedFor {
                        source_name: "alsa_input.physical".into(),
                        sink_name: "other".into(),
                    },
                    AcousticAdmissionError::AecNotValidated,
                ),
            ] {
                assert_eq!(
                    admit_acoustic(enabled(true, speaker), &facts(OutputMode::OpenSpeaker, aec)),
                    Err(error)
                );
            }
            let reservation = exact_reservation();
            let result = admit_acoustic_with_reservation(
                enabled(true, speaker),
                &facts(OutputMode::OpenSpeaker, exact_aec()),
                Some(&reservation),
            )
            .unwrap();
            assert_eq!(
                result.microphone,
                Some(DirectionAudioTargets {
                    capture: AEC_SOURCE.into(),
                    playback: MIC_OUT_SINK.into()
                })
            );
            assert_eq!(
                result.speaker,
                speaker.then(|| DirectionAudioTargets {
                    capture: format!("{REMOTE_IN_SINK}.monitor"),
                    playback: AEC_SINK.into()
                })
            );
        }
    }

    #[test]
    fn serialized_status_alone_never_authorizes_open_speaker_outgoing() {
        assert_eq!(
            admit_acoustic(
                enabled(true, true),
                &facts(OutputMode::OpenSpeaker, exact_aec())
            ),
            Err(AcousticAdmissionError::AecNotValidated)
        );
    }

    #[test]
    fn unknown_output_rejects_outgoing_even_with_exact_aec() {
        for aec in [
            AecCapability::Unavailable,
            AecCapability::AvailableUnvalidated,
            AecCapability::ValidationFailed,
            exact_aec(),
            AecCapability::ValidatedFor {
                source_name: "other".into(),
                sink_name: "alsa_output.physical".into(),
            },
            AecCapability::ValidatedFor {
                source_name: "alsa_input.physical".into(),
                sink_name: "other".into(),
            },
        ] {
            for speaker in [false, true] {
                assert_eq!(
                    admit_acoustic(
                        enabled(true, speaker),
                        &facts(OutputMode::UnknownUnsafe, aec.clone())
                    ),
                    Err(AcousticAdmissionError::UnknownOutput)
                );
            }
        }
    }

    #[test]
    fn unavailable_or_inconsistent_required_devices_reject() {
        for source in [false, true] {
            for (fault, expected) in [
                ("missing", AcousticAdmissionError::DeviceUnavailable),
                ("health", AcousticAdmissionError::DeviceUnavailable),
                ("unavailable", AcousticAdmissionError::DeviceUnavailable),
                ("pin", AcousticAdmissionError::InvalidPhysicalSelection),
                ("unpinned", AcousticAdmissionError::InvalidPhysicalSelection),
                ("monitor", AcousticAdmissionError::InvalidPhysicalSelection),
                ("virtual", AcousticAdmissionError::InvalidPhysicalSelection),
                ("empty", AcousticAdmissionError::InvalidPhysicalSelection),
            ] {
                let mut devices = facts(OutputMode::Headphones, AecCapability::Unavailable);
                let selected = if source {
                    &mut devices.source
                } else {
                    &mut devices.sink
                };
                match fault {
                    "missing" => selected.selected = None,
                    "health" => selected.health = DeviceHealth::DeviceUnavailable,
                    "unavailable" => selected.selected.as_mut().unwrap().available = false,
                    "pin" => selected.pinned_name = Some("other".into()),
                    "unpinned" => selected.pinned_name = None,
                    "monitor" | "virtual" | "empty" => {
                        let name = match fault {
                            "monitor" => "physical.monitor",
                            "virtual" => "translator_virtual",
                            _ => " ",
                        };
                        selected.selected.as_mut().unwrap().name = name.into();
                        selected.pinned_name = Some(name.into());
                    }
                    _ => unreachable!(),
                }
                assert_eq!(
                    admit_acoustic(enabled(true, true), &devices),
                    Err(expected),
                    "source={source} fault={fault}"
                );
                assert_eq!(
                    admit_acoustic(enabled(true, false), &devices),
                    Err(expected),
                    "outgoing source={source} fault={fault}"
                );
                let projected = DeviceState::from(devices.clone());
                assert!(!projected.acoustic.full_duplex_allowed);
                assert_eq!(
                    projected.acoustic.warning,
                    Some(AcousticWarning::DeviceUnavailable)
                );
                assert_eq!(projected.source, devices.source);
                assert_eq!(projected.sink, devices.sink);
                if !source {
                    assert_eq!(
                        admit_acoustic(enabled(false, true), &devices),
                        Err(expected),
                        "incoming fault={fault}"
                    );
                }
            }
        }
    }

    #[test]
    fn both_disabled_precedes_missing_devices() {
        let mut devices = facts(OutputMode::UnknownUnsafe, AecCapability::Unavailable);
        assert_eq!(
            admit_acoustic(
                enabled(false, false),
                &facts(OutputMode::Headphones, AecCapability::Unavailable)
            ),
            Err(AcousticAdmissionError::NoDirectionEnabled)
        );
        devices.source.selected = None;
        devices.sink.selected = None;
        assert_eq!(
            admit_acoustic(enabled(false, false), &devices),
            Err(AcousticAdmissionError::NoDirectionEnabled)
        );
    }

    #[test]
    fn warning_projection_uses_full_duplex_policy_without_blocking_incoming() {
        for (mode, aec, warning) in [
            (OutputMode::Headphones, AecCapability::Unavailable, None),
            (
                OutputMode::OpenSpeaker,
                AecCapability::Unavailable,
                Some(AcousticWarning::AecNotValidated),
            ),
            (
                OutputMode::OpenSpeaker,
                AecCapability::AvailableUnvalidated,
                Some(AcousticWarning::AecNotValidated),
            ),
            (
                OutputMode::OpenSpeaker,
                exact_aec(),
                Some(AcousticWarning::AecNotValidated),
            ),
            (
                OutputMode::OpenSpeaker,
                AecCapability::ValidatedFor {
                    source_name: "other".into(),
                    sink_name: "alsa_output.physical".into(),
                },
                Some(AcousticWarning::AecNotValidated),
            ),
            (
                OutputMode::OpenSpeaker,
                AecCapability::ValidatedFor {
                    source_name: "alsa_input.physical".into(),
                    sink_name: "other".into(),
                },
                Some(AcousticWarning::AecNotValidated),
            ),
            (
                OutputMode::OpenSpeaker,
                AecCapability::ValidationFailed,
                Some(AcousticWarning::AecValidationFailed),
            ),
            (
                OutputMode::UnknownUnsafe,
                exact_aec(),
                Some(AcousticWarning::UnknownOutput),
            ),
        ] {
            let devices = facts(mode, aec);
            let projected = DeviceState::from(devices.clone());
            assert_eq!(projected.acoustic.full_duplex_allowed, warning.is_none());
            assert_eq!(projected.acoustic.warning, warning);
            assert_eq!(projected.source, devices.source);
            assert_eq!(projected.sink, devices.sink);
            assert_eq!(projected.acoustic.mode, devices.output_mode);
            assert_eq!(projected.acoustic.aec_capability, devices.aec_capability);
            assert!(admit_acoustic(enabled(false, true), &devices).is_ok());
        }
    }
}
