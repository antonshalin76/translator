use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use TranslationMixMode::{Bypass, Quarantine, Translating};
use translator_audio::{CommandResult, CommandRunError, CommandRunner};
use translator_daemon::{
    AudioMixApplication, AudioMixController, AudioMixState, PlaybackMixAuthority,
    PlaybackRegistrationPhase, TranslationMixMode,
};

#[tokio::test]
#[ignore = "requires an explicit disposable private PulseAudio socket and virtual fixture sink"]
async fn private_pulse_respawn_admits_exact_committed_volume_before_first_frame() {
    use std::time::{Duration, Instant};
    use translator_audio::{PulsePcmCommand, PulsePcmPlayback, SystemCommandRunner};

    let server = std::env::var("PULSE_SERVER").expect("private PULSE_SERVER required");
    assert!(
        (server.starts_with("unix:/tmp/translator-handshake-")
            || server.starts_with("unix:/tmp/translator-loopback-"))
            && server.ends_with("/native"),
        "refusing non-fixture PulseAudio server"
    );
    let mix = AudioMixApplication::new(SystemCommandRunner);
    mix.reconcile_committed(Quarantine {
        mic_original_expected: false,
    })
    .unwrap();
    let mut first = PulsePcmPlayback::spawn(&PulsePcmCommand::playback(
        "translator_mic_out",
        "translator-outgoing-playback",
    ))
    .unwrap();
    let first_registration = first
        .wait_registered_muted(Instant::now() + Duration::from_secs(2))
        .await
        .unwrap();
    mix.admit_registered(
        &first_registration,
        PlaybackRegistrationPhase::StartMuted,
        Instant::now() + Duration::from_secs(2),
    )
    .unwrap();
    let desired = AudioMixState {
        microphone_translation_percent: 37,
        ..AudioMixState::default()
    };
    mix.apply_desired(desired, Translating).unwrap();
    first.stop().await.unwrap();

    let mut respawn = PulsePcmPlayback::spawn(&PulsePcmCommand::playback(
        "translator_mic_out",
        "translator-outgoing-playback",
    ))
    .unwrap();
    let registration = respawn
        .wait_registered_muted(Instant::now() + Duration::from_secs(2))
        .await
        .unwrap();
    assert_ne!(first_registration.session_id(), registration.session_id());
    mix.admit_registered(
        &registration,
        PlaybackRegistrationPhase::Running,
        Instant::now() + Duration::from_secs(2),
    )
    .unwrap();
    respawn.stop().await.unwrap();
    assert!(
        mix.admit_registered(
            &registration,
            PlaybackRegistrationPhase::Running,
            Instant::now() + Duration::from_secs(2),
        )
        .is_err(),
        "a closed playback session cannot be re-admitted by its old index"
    );
}

#[derive(Clone)]
struct FakeRunner {
    calls: Arc<Mutex<Vec<Vec<String>>>>,
    list_results: Arc<Mutex<VecDeque<CommandResult>>>,
    fail_sets: Vec<usize>,
    set_count: Arc<Mutex<usize>>,
}

impl FakeRunner {
    fn new(list_results: Vec<CommandResult>) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            list_results: Arc::new(Mutex::new(list_results.into())),
            fail_sets: Vec::new(),
            set_count: Arc::new(Mutex::new(0)),
        }
    }

    fn calls(&self) -> Vec<Vec<String>> {
        self.calls.lock().unwrap().clone()
    }
}

impl CommandRunner for FakeRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        self.calls.lock().unwrap().push(args.to_vec());
        if args.first().is_some_and(|arg| arg == "--format=json") {
            return Ok(self.list_results.lock().unwrap().pop_front().unwrap());
        }
        let mut count = self.set_count.lock().unwrap();
        *count += 1;
        if self.fail_sets.contains(&*count) {
            return Err(CommandRunError::TimedOut);
        }
        Ok(CommandResult::success(Vec::new()))
    }
}

fn sink_input(
    index: u32,
    application_name: &str,
    media_name: &str,
    sink_name: &str,
    module_id: &str,
) -> serde_json::Value {
    let mut input = serde_json::json!({
        "index": index,
        "channel_map": "front-right,front-left",
        "volume": {
            "front-left": {"value": 32768},
            "front-right": {"value": 49152}
        },
        "properties": {
            "application.name": application_name,
            "media.name": media_name,
        },
    });
    if !module_id.is_empty() {
        input["owner_module"] = module_id.into();
        input["sink"] = match sink_name {
            "alsa_output.headphones" => 0.into(),
            "translator_mic_out" => 1.into(),
            other => panic!("unexpected sink {other}"),
        };
        input["properties"]["translator.owner"] = "true".into();
    }
    input
}

#[test]
fn disabled_microphone_preserves_positive_desired_without_requiring_or_opening_raw_path() {
    let runner = FakeRunner::new(vec![
        CommandResult::success(b"[]".to_vec()),
        CommandResult::success(b"[]".to_vec()),
    ]);
    let app = AudioMixApplication::new(runner.clone());
    let desired = AudioMixState {
        microphone_original_percent: 35,
        microphone_translation_percent: 75,
        speaker_original_percent: 0,
        speaker_translation_percent: 63,
    };
    app.validate_desired_for_mode(desired, TranslationMixMode::TranslatingMicrophoneMuted)
        .unwrap();
    assert!(runner.calls().is_empty());
    app.apply_desired(desired, TranslationMixMode::TranslatingMicrophoneMuted)
        .unwrap();
    assert_eq!(app.committed().unwrap(), desired);
    assert_eq!(app.native_playback_percent(false).unwrap(), 63);
    assert!(runner.calls().iter().all(|args| args[0] == "--format=json"));
}

#[test]
fn disabled_microphone_bypass_accepts_retained_desired_without_raw_capture() {
    let runner = FakeRunner::new(vec![
        CommandResult::success(b"[]".to_vec()),
        CommandResult::success(b"[]".to_vec()),
    ]);
    let app = AudioMixApplication::new(runner.clone());
    let desired = AudioMixState {
        microphone_original_percent: 35,
        ..AudioMixState::default()
    };
    app.apply_desired(desired, TranslationMixMode::MicrophoneMutedBypass)
        .unwrap();
    assert_eq!(app.committed().unwrap(), desired);
    assert!(runner.calls().iter().all(|args| args[0] == "--format=json"));
}

#[test]
fn failed_second_set_restores_uncertain_target_then_first_in_exact_channel_order() {
    let inputs = serde_json::json!([
        sink_input(
            43,
            "translator-daemon",
            "translator-outgoing-playback",
            "",
            ""
        ),
        sink_input(
            44,
            "translator-daemon",
            "translator-incoming-playback",
            "",
            ""
        ),
    ]);
    let mut runner = FakeRunner::new(vec![
        CommandResult::success(serde_json::to_vec(&inputs).unwrap()),
        CommandResult::success(b"[]".to_vec()),
    ]);
    runner.fail_sets = vec![2];
    let result = AudioMixApplication::new(runner.clone()).apply_desired(
        AudioMixState {
            microphone_original_percent: 0,
            microphone_translation_percent: 80,
            speaker_original_percent: 0,
            speaker_translation_percent: 90,
        },
        Translating,
    );
    assert!(result.is_err());
    assert_eq!(
        runner.calls(),
        vec![
            vec!["--format=json", "list", "sink-inputs"],
            vec!["--format=json", "list", "source-outputs"],
            vec!["set-sink-input-volume", "43", "80%"],
            vec!["set-sink-input-volume", "44", "90%"],
            vec!["set-sink-input-volume", "44", "49152", "32768"],
            vec!["set-sink-input-volume", "43", "49152", "32768"],
        ]
    );
}

#[test]
fn failed_quarantine_never_restores_a_nonzero_original_microphone() {
    let mut runner = FakeRunner::new(four_target_discovery());
    runner.fail_sets = vec![3];
    let app = AudioMixApplication::new(runner.clone());

    assert_eq!(
        app.reconcile_committed(Quarantine {
            mic_original_expected: true,
        })
        .unwrap_err()
        .code,
        "audio_mix_state_unknown"
    );
    assert_eq!(app.committed().unwrap_err().code, "audio_mix_state_unknown");
    assert!(
        runner
            .calls()
            .iter()
            .any(|call| call == &vec!["set-sink-input-volume", "42", "0%"]),
        "original microphone must be muted before the later failure"
    );
    assert!(
        !runner.calls().iter().any(|call| {
            call.first()
                .is_some_and(|value| value == "set-sink-input-volume")
                && call.get(1).is_some_and(|value| value == "42")
                && call.get(2).is_some_and(|value| value != "0%")
        }),
        "failed quarantine must not restore the prior nonzero microphone volume"
    );
    assert!(
        runner
            .calls()
            .iter()
            .any(|call| call == &vec!["set-sink-input-volume", "44", "0%"]),
        "emergency mute must attempt remaining streams after a failed set"
    );
    let before = runner.calls().len();
    assert_eq!(
        app.reconcile_committed(Quarantine {
            mic_original_expected: true,
        })
        .unwrap_err()
        .code,
        "audio_mix_state_unknown"
    );
    assert_eq!(
        runner.calls().len(),
        before,
        "unknown mix must block writes"
    );
}

#[test]
fn missing_prior_channel_volume_rejects_before_any_set() {
    let mut input = sink_input(
        43,
        "translator-daemon",
        "translator-outgoing-playback",
        "",
        "",
    );
    input["volume"]
        .as_object_mut()
        .unwrap()
        .remove("front-left");
    let runner = FakeRunner::new(vec![
        CommandResult::success(serde_json::to_vec(&vec![input]).unwrap()),
        CommandResult::success(b"[]".to_vec()),
    ]);
    let result = AudioMixApplication::new(runner.clone()).apply_desired(
        AudioMixState {
            microphone_original_percent: 0,
            microphone_translation_percent: 80,
            speaker_original_percent: 0,
            speaker_translation_percent: 90,
        },
        Translating,
    );
    assert!(
        result.is_err(),
        "unrecoverable prior state must fail before physical writes"
    );
    assert!(runner.calls().iter().all(|args| args[0] == "--format=json"));
}

fn source_output(media_name: &str, source_name: &str, module_id: &str) -> serde_json::Value {
    serde_json::json!({
        "owner_module": module_id,
        "source": match source_name {
            "translator_remote_in.monitor" => 1,
            "alsa_input.usb" => 0,
            other => panic!("unexpected source {other}"),
        },
        "properties": {
            "media.name": media_name,
            "translator.owner": "true",
        },
    })
}

fn two_target_discovery() -> Vec<CommandResult> {
    let inputs = serde_json::json!([
        sink_input(
            43,
            "translator-daemon",
            "translator-outgoing-playback",
            "",
            ""
        ),
        sink_input(
            44,
            "translator-daemon",
            "translator-incoming-playback",
            "",
            ""
        ),
    ]);
    vec![
        CommandResult::success(serde_json::to_vec(&inputs).unwrap()),
        CommandResult::success(b"[]".to_vec()),
    ]
}

#[test]
fn requested_raw_microphone_volume_requires_a_discovered_owned_target() {
    let mut replies = two_target_discovery();
    replies.extend(two_target_discovery());
    replies.extend(two_target_discovery());
    let runner = FakeRunner::new(replies);
    let app = AudioMixApplication::new(runner.clone());
    let candidate = AudioMixState {
        microphone_original_percent: 25,
        ..AudioMixState::default()
    };

    assert_eq!(
        app.validate_desired(candidate).unwrap_err().code,
        "microphone_original_unavailable"
    );
    for mode in [Translating, TranslationMixMode::Bypass] {
        assert_eq!(
            app.apply_desired(candidate, mode).unwrap_err().code,
            "microphone_original_unavailable"
        );
    }
    assert_eq!(app.committed().unwrap(), AudioMixState::default());
    assert!(
        runner
            .calls()
            .iter()
            .all(|args| { args.first().is_some_and(|arg| arg == "--format=json") })
    );
}

#[test]
fn active_mix_reconcile_fails_if_committed_raw_microphone_target_disappears() {
    let mut replies = four_target_discovery();
    replies.extend(two_target_discovery());
    let runner = FakeRunner::new(replies);
    let app = AudioMixApplication::new(runner.clone());
    let candidate = AudioMixState {
        microphone_original_percent: 25,
        ..AudioMixState::default()
    };

    app.apply_desired(candidate, Translating).unwrap();
    let writes_before = runner
        .calls()
        .iter()
        .filter(|args| args.first().is_some_and(|arg| arg.starts_with("set-")))
        .count();
    assert_eq!(
        app.reconcile_committed(Translating).unwrap_err().code,
        "microphone_original_unavailable"
    );
    assert_eq!(app.committed().unwrap(), candidate);
    assert_eq!(
        runner
            .calls()
            .iter()
            .filter(|args| args.first().is_some_and(|arg| arg.starts_with("set-")))
            .count(),
        writes_before
    );
}

#[test]
fn failed_compensation_blocks_normal_work_until_committed_recovery_succeeds() {
    let mut replies = two_target_discovery();
    replies.extend(two_target_discovery());
    replies.extend(two_target_discovery());
    let mut runner = FakeRunner::new(replies);
    runner.fail_sets = vec![2, 3, 5];
    let app = AudioMixApplication::new(runner.clone());
    let candidate = AudioMixState {
        microphone_translation_percent: 80,
        speaker_translation_percent: 90,
        ..AudioMixState::default()
    };
    assert_eq!(
        app.apply_desired(candidate, Translating).unwrap_err().code,
        "audio_mix_state_unknown"
    );
    assert_eq!(app.committed().unwrap_err().code, "audio_mix_state_unknown");
    assert_eq!(
        &runner.calls()[4..],
        [
            vec!["set-sink-input-volume", "44", "49152", "32768"],
            vec!["set-sink-input-volume", "43", "49152", "32768"],
        ],
        "failure restoring target 44 must not skip restoration of target 43"
    );
    let after_failure = runner.calls().len();
    assert_eq!(
        app.apply_desired(candidate, Translating).unwrap_err().code,
        "audio_mix_state_unknown"
    );
    assert_eq!(
        app.reconcile_committed(Translating).unwrap_err().code,
        "audio_mix_state_unknown"
    );
    assert_eq!(
        runner.calls().len(),
        after_failure,
        "unknown state must not write"
    );
    assert_eq!(
        app.recover_committed(Translating).unwrap_err().code,
        "audio_mix_state_unknown"
    );
    assert!(
        app.committed().is_err(),
        "a rolled-back recovery is still unknown"
    );
    assert_eq!(
        &runner.calls()[after_failure..],
        [
            vec!["--format=json", "list", "sink-inputs"],
            vec!["--format=json", "list", "source-outputs"],
            vec!["set-sink-input-volume", "43", "100%"],
            vec!["set-sink-input-volume", "43", "49152", "32768"],
        ],
        "failed recovery must restore its uncertain failed target"
    );
    app.recover_committed(Translating).unwrap();
    assert_eq!(app.committed().unwrap(), AudioMixState::default());
    let calls = runner.calls();
    assert_eq!(
        &calls[calls.len() - 2..],
        [
            vec!["set-sink-input-volume", "43", "100%"],
            vec!["set-sink-input-volume", "44", "100%"],
        ]
    );
}

#[test]
fn failed_patch_is_never_reapplied_by_reconciliation() {
    let mut replies = two_target_discovery();
    replies.extend(two_target_discovery());
    let mut runner = FakeRunner::new(replies);
    runner.fail_sets = vec![2];
    let app = AudioMixApplication::new(runner.clone());
    let candidate = AudioMixState {
        microphone_translation_percent: 80,
        speaker_translation_percent: 90,
        ..AudioMixState::default()
    };
    assert_eq!(
        app.apply_desired(candidate, Translating).unwrap_err().code,
        "audio_mix_apply_failed"
    );
    assert_eq!(app.committed().unwrap(), AudioMixState::default());
    app.reconcile_committed(Translating).unwrap();
    let calls = runner.calls();
    assert_eq!(
        &calls[calls.len() - 2..],
        [
            vec!["set-sink-input-volume", "43", "100%"],
            vec!["set-sink-input-volume", "44", "100%"],
        ]
    );
}

#[test]
fn empty_live_plan_commits_desired_but_invalid_volume_never_discovers() {
    let runner = FakeRunner::new(vec![
        CommandResult::success(b"[]".to_vec()),
        CommandResult::success(b"[]".to_vec()),
    ]);
    let app = AudioMixApplication::new(runner.clone());
    let desired = AudioMixState {
        microphone_translation_percent: 0,
        speaker_translation_percent: 100,
        ..AudioMixState::default()
    };
    app.apply_desired(desired, Translating).unwrap();
    assert_eq!(app.committed().unwrap(), desired);
    assert_eq!(runner.calls().len(), 2);
    let invalid = AudioMixState {
        speaker_original_percent: 101,
        ..desired
    };
    assert_eq!(
        app.apply_desired(invalid, Translating).unwrap_err().code,
        "invalid_audio_mix_volume"
    );
    assert_eq!(runner.calls().len(), 2);
    assert_eq!(app.committed().unwrap(), desired);
}

#[test]
fn unknown_recovery_discovery_failure_preserves_suspension_without_writes() {
    let mut replies = two_target_discovery();
    replies.push(CommandResult::failure(Vec::new(), Vec::new()));
    let mut runner = FakeRunner::new(replies);
    runner.fail_sets = vec![2, 3];
    let app = AudioMixApplication::new(runner.clone());
    assert_eq!(
        app.apply_desired(AudioMixState::default(), Translating)
            .unwrap_err()
            .code,
        "audio_mix_state_unknown"
    );
    let before = *runner.set_count.lock().unwrap();
    assert_eq!(
        app.recover_committed(Translating).unwrap_err().code,
        "audio_mix_state_unknown"
    );
    assert_eq!(*runner.set_count.lock().unwrap(), before);
    assert_eq!(
        app.reconcile_committed(Translating).unwrap_err().code,
        "audio_mix_state_unknown"
    );
}

fn four_target_discovery() -> Vec<CommandResult> {
    let sink_inputs = serde_json::json!([
        sink_input(
            41,
            "",
            "loopback-speaker-original",
            "alsa_output.headphones",
            "9001"
        ),
        sink_input(
            42,
            "",
            "loopback-microphone-original",
            "translator_mic_out",
            "9002"
        ),
        sink_input(
            43,
            "translator-daemon",
            "translator-outgoing-playback",
            "translator_mic_out",
            ""
        ),
        sink_input(
            44,
            "translator-daemon",
            "translator-incoming-playback",
            "alsa_output.headphones",
            ""
        ),
        sink_input(
            45,
            "Telegram Desktop",
            "Playback Stream",
            "translator_remote_in",
            ""
        ),
    ]);
    let source_outputs = serde_json::json!([
        source_output(
            "loopback-speaker-original",
            "translator_remote_in.monitor",
            "9001"
        ),
        source_output("loopback-microphone-original", "alsa_input.usb", "9002"),
    ]);
    vec![
        CommandResult::success(serde_json::to_vec(&sink_inputs).unwrap()),
        CommandResult::success(serde_json::to_vec(&source_outputs).unwrap()),
        CommandResult::success(
            serde_json::to_vec(&serde_json::json!([
                {"index": 0, "name": "alsa_input.usb"},
                {"index": 1, "name": "translator_remote_in.monitor"}
            ]))
            .unwrap(),
        ),
        CommandResult::success(
            serde_json::to_vec(&serde_json::json!([
                {"index": 0, "name": "alsa_output.headphones"},
                {"index": 1, "name": "translator_mic_out"}
            ]))
            .unwrap(),
        ),
    ]
}

#[test]
fn applies_independent_mix_volumes_to_current_pulse_streams() {
    let runner = FakeRunner::new(four_target_discovery());
    let application = AudioMixApplication::new(runner.clone());
    application
        .apply_desired(
            AudioMixState {
                microphone_original_percent: 31,
                microphone_translation_percent: 32,
                speaker_original_percent: 33,
                speaker_translation_percent: 34,
            },
            Translating,
        )
        .unwrap();

    assert_eq!(
        application.committed().unwrap().speaker_translation_percent,
        34
    );
    assert_eq!(
        runner.calls(),
        vec![
            vec!["--format=json", "list", "sink-inputs"],
            vec!["--format=json", "list", "source-outputs"],
            vec!["--format=json", "list", "sources"],
            vec!["--format=json", "list", "sinks"],
            vec!["set-sink-input-volume", "41", "33%"],
            vec!["set-sink-input-volume", "42", "31%"],
            vec!["set-sink-input-volume", "43", "32%"],
            vec!["set-sink-input-volume", "44", "34%"],
        ]
    );
}

fn translating_readback(microphone_original_percent: u8) -> Vec<CommandResult> {
    let mut replies = four_target_discovery();
    let mut inputs: serde_json::Value = serde_json::from_slice(replies[0].stdout()).unwrap();
    for (index, percent) in [0, microphone_original_percent, 80, 100]
        .into_iter()
        .enumerate()
    {
        let value = u32::from(percent) * 65536 / 100;
        for channel in ["front-left", "front-right"] {
            inputs[index]["volume"][channel]["value"] = value.into();
        }
    }
    replies[0] = CommandResult::success(serde_json::to_vec(&inputs).unwrap());
    replies
}

fn quarantine_readback(nonzero_index: Option<usize>) -> Vec<CommandResult> {
    let mut replies = four_target_discovery();
    let mut inputs: serde_json::Value = serde_json::from_slice(replies[0].stdout()).unwrap();
    for index in 0..4 {
        let value = if nonzero_index == Some(index) {
            32768
        } else {
            0
        };
        for channel in ["front-left", "front-right"] {
            inputs[index]["volume"][channel]["value"] = value.into();
        }
    }
    replies[0] = CommandResult::success(serde_json::to_vec(&inputs).unwrap());
    replies
}

#[test]
fn quarantine_requires_every_owned_output_to_read_back_zero() {
    for nonzero_index in 0..4 {
        let mut replies = four_target_discovery();
        replies.extend(quarantine_readback(Some(nonzero_index)));
        let app = AudioMixApplication::new(FakeRunner::new(replies));
        assert_eq!(
            app.reconcile_committed(Quarantine {
                mic_original_expected: true,
            })
            .unwrap_err()
            .code,
            "audio_mix_state_unknown"
        );
    }

    let mut replies = four_target_discovery();
    replies.extend(quarantine_readback(None));
    let app = AudioMixApplication::new(FakeRunner::new(replies));
    app.reconcile_committed(Quarantine {
        mic_original_expected: true,
    })
    .unwrap();
}

#[test]
fn microphone_muted_bypass_preserves_speaker_without_unmuting_stale_raw_mic() {
    let mut replies = four_target_discovery();
    replies.extend(quarantine_readback(None));
    let runner = FakeRunner::new(replies);
    let app = AudioMixApplication::new(runner.clone());

    app.reconcile_committed(TranslationMixMode::MicrophoneMutedBypass)
        .unwrap();
    let calls = runner.calls();
    for (index, percent) in [("41", "100%"), ("42", "0%"), ("43", "0%"), ("44", "0%")] {
        assert!(
            calls
                .iter()
                .any(|call| { call == &vec!["set-sink-input-volume", index, percent] })
        );
    }
    assert!(
        calls[8..]
            .iter()
            .any(|call| call == &vec!["--format=json", "list", "sink-inputs"])
    );
}

#[test]
fn failed_microphone_muted_bypass_cannot_restore_stale_raw_mic() {
    let mut runner = FakeRunner::new(four_target_discovery());
    runner.fail_sets = vec![3];
    let app = AudioMixApplication::new(runner.clone());

    assert_eq!(
        app.reconcile_committed(TranslationMixMode::MicrophoneMutedBypass)
            .unwrap_err()
            .code,
        "audio_mix_state_unknown"
    );
    assert!(
        runner
            .calls()
            .iter()
            .any(|call| { call == &vec!["set-sink-input-volume", "42", "0%"] })
    );
    assert!(!runner.calls().iter().any(|call| {
        call.first()
            .is_some_and(|value| value == "set-sink-input-volume")
            && call.get(1).is_some_and(|value| value == "42")
            && call.get(2).is_some_and(|value| value != "0%")
    }));
}

#[test]
fn headphone_quarantine_without_owned_original_mic_target_fails_closed() {
    let runner = FakeRunner::new(two_target_discovery());
    let app = AudioMixApplication::new(runner.clone());
    assert_eq!(
        app.reconcile_committed(Quarantine {
            mic_original_expected: true,
        })
        .unwrap_err()
        .code,
        "audio_mix_state_unknown"
    );
    assert!(runner.calls().iter().all(|call| call[0] == "--format=json"));
}

#[test]
fn acknowledged_mic_mute_without_observed_zero_must_not_commit() {
    let mut replies = four_target_discovery();
    replies.extend(translating_readback(50));
    let runner = FakeRunner::new(replies);
    let application = AudioMixApplication::new(runner.clone());
    let desired = AudioMixState {
        microphone_translation_percent: 80,
        ..AudioMixState::default()
    };

    let error = application.apply_desired(desired, Translating).unwrap_err();
    assert_eq!(error.code, "audio_mix_state_unknown");
    assert_ne!(application.committed().ok(), Some(desired));
    let calls = runner.calls();
    let mute = calls
        .iter()
        .position(|call| call == &vec!["set-sink-input-volume", "42", "0%"])
        .expect("the owned original microphone must be muted");
    assert!(
        calls[mute + 1..]
            .iter()
            .any(|call| call == &vec!["--format=json", "list", "sink-inputs"]),
        "the mute must be verified against an observed Pulse state"
    );
}

#[test]
fn observed_original_mic_zero_can_commit_translating_mix() {
    let mut replies = four_target_discovery();
    replies.extend(translating_readback(0));
    let runner = FakeRunner::new(replies);
    let application = AudioMixApplication::new(runner.clone());
    let desired = AudioMixState {
        microphone_translation_percent: 80,
        ..AudioMixState::default()
    };

    application.apply_desired(desired, Translating).unwrap();
    assert_eq!(application.committed().unwrap(), desired);
    let calls = runner.calls();
    let mute = calls
        .iter()
        .position(|call| call == &vec!["set-sink-input-volume", "42", "0%"])
        .unwrap();
    assert!(
        calls[mute + 1..]
            .iter()
            .any(|call| call == &vec!["--format=json", "list", "sink-inputs"]),
        "successful mute must also be verified against observed Pulse state"
    );
}

#[test]
fn stopped_patch_and_reconcile_preserve_desired_mix_for_next_start() {
    let replies = (0..4).flat_map(|_| four_target_discovery()).collect();
    let runner = FakeRunner::new(replies);
    let app = AudioMixApplication::new(runner.clone());
    let desired = AudioMixState {
        microphone_original_percent: 31,
        microphone_translation_percent: 32,
        speaker_original_percent: 33,
        speaker_translation_percent: 34,
    };
    app.apply_desired(desired, Bypass).unwrap();
    app.reconcile_committed(Translating).unwrap();
    app.reconcile_committed(Bypass).unwrap();
    app.reconcile_committed(Translating).unwrap();
    assert_eq!(app.committed().unwrap(), desired);
    let calls = runner.calls();
    let sets: Vec<_> = calls
        .iter()
        .filter(|call| call[0] == "set-sink-input-volume")
        .collect();
    let bypass = [
        vec!["set-sink-input-volume", "41", "100%"],
        vec!["set-sink-input-volume", "42", "100%"],
        vec!["set-sink-input-volume", "43", "0%"],
        vec!["set-sink-input-volume", "44", "0%"],
    ];
    let translating = [
        vec!["set-sink-input-volume", "41", "33%"],
        vec!["set-sink-input-volume", "42", "31%"],
        vec!["set-sink-input-volume", "43", "32%"],
        vec!["set-sink-input-volume", "44", "34%"],
    ];
    assert_eq!(
        sets,
        bypass
            .iter()
            .chain(translating.iter())
            .chain(bypass.iter())
            .chain(translating.iter())
            .collect::<Vec<_>>()
    );
}

#[test]
fn invalid_desired_volume_is_rejected_before_discovery_even_in_bypass() {
    let runner = FakeRunner::new(Vec::new());
    let app = AudioMixApplication::new(runner.clone());
    let invalid = AudioMixState {
        microphone_translation_percent: 101,
        ..AudioMixState::default()
    };
    assert_eq!(
        app.apply_desired(invalid, Bypass).unwrap_err().code,
        "invalid_audio_mix_volume"
    );
    assert!(runner.calls().is_empty());
    assert_eq!(app.committed().unwrap(), AudioMixState::default());
}

#[test]
fn failed_bypass_compensation_requires_recovery_without_adopting_bypass_as_desired() {
    let replies = (0..4).flat_map(|_| two_target_discovery()).collect();
    let mut runner = FakeRunner::new(replies);
    runner.fail_sets = vec![4, 5];
    let app = AudioMixApplication::new(runner.clone());
    let desired = AudioMixState {
        microphone_translation_percent: 80,
        speaker_translation_percent: 90,
        ..AudioMixState::default()
    };
    app.apply_desired(desired, Translating).unwrap();
    assert_eq!(
        app.reconcile_committed(Bypass).unwrap_err().code,
        "audio_mix_state_unknown"
    );
    let before = runner.calls().len();
    assert_eq!(
        app.reconcile_committed(Bypass).unwrap_err().code,
        "audio_mix_state_unknown"
    );
    assert_eq!(runner.calls().len(), before);
    app.recover_committed(Bypass).unwrap();
    assert_eq!(app.committed().unwrap(), desired);
    let calls = runner.calls();
    assert_eq!(
        &calls[calls.len() - 2..],
        [
            vec!["set-sink-input-volume", "43", "0%"],
            vec!["set-sink-input-volume", "44", "0%"],
        ]
    );
    app.reconcile_committed(Translating).unwrap();
    let calls = runner.calls();
    assert_eq!(
        &calls[calls.len() - 2..],
        [
            vec!["set-sink-input-volume", "43", "80%"],
            vec!["set-sink-input-volume", "44", "90%"],
        ]
    );
}
