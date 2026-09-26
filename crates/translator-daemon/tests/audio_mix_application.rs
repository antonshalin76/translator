use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use TranslationMixMode::{Bypass, Translating};
use translator_audio::{CommandResult, CommandRunError, CommandRunner};
use translator_daemon::{
    AudioMixApplication, AudioMixController, AudioMixState, TranslationMixMode,
};

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
    target_object: &str,
    module_id: &str,
) -> serde_json::Value {
    serde_json::json!({
        "index": index,
        "channel_map": "front-right,front-left",
        "volume": {
            "front-left": {"value": 32768},
            "front-right": {"value": 49152}
        },
        "properties": {
            "application.name": application_name,
            "media.name": media_name,
            "target.object": target_object,
            "pulse.module.id": module_id,
        },
    })
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

fn source_output(media_name: &str, target_object: &str, module_id: &str) -> serde_json::Value {
    serde_json::json!({
        "properties": {
            "media.name": media_name,
            "target.object": target_object,
            "pulse.module.id": module_id,
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
            "loopback-1 output",
            "alsa_output.headphones",
            "9001"
        ),
        sink_input(42, "", "loopback-2 output", "translator_mic_out", "9002"),
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
        source_output("loopback-1 input", "translator_remote_in", "9001"),
        source_output("loopback-2 input", "alsa_input.usb", "9002"),
    ]);
    vec![
        CommandResult::success(serde_json::to_vec(&sink_inputs).unwrap()),
        CommandResult::success(serde_json::to_vec(&source_outputs).unwrap()),
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
            vec!["set-sink-input-volume", "41", "33%"],
            vec!["set-sink-input-volume", "42", "31%"],
            vec!["set-sink-input-volume", "43", "32%"],
            vec!["set-sink-input-volume", "44", "34%"],
        ]
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
