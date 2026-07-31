use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use translator_audio::{
    AudioMixTarget, AudioMixVolumes, CommandResult, CommandRunError, CommandRunner, PulseAudioMix,
};

#[derive(Clone)]
struct FakeRunner {
    calls: Arc<Mutex<Vec<Vec<String>>>>,
    list_results: Arc<Mutex<VecDeque<CommandResult>>>,
}

impl FakeRunner {
    fn new(list_results: Vec<CommandResult>) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            list_results: Arc::new(Mutex::new(list_results.into())),
        }
    }

    fn calls(&self) -> Vec<Vec<String>> {
        self.calls.lock().unwrap().clone()
    }
}

impl CommandRunner for FakeRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        self.calls.lock().unwrap().push(args.to_vec());
        if args.first().is_some_and(|arg| arg == "--format=json") {
            return Ok(self.list_results.lock().unwrap().pop_front().unwrap());
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
        "properties": {
            "application.name": application_name,
            "media.name": media_name,
            "target.object": target_object,
            "pulse.module.id": module_id,
        },
    })
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

#[test]
fn applies_independent_mix_volumes_to_current_pulse_streams() {
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
    let runner = FakeRunner::new(vec![
        CommandResult::success(serde_json::to_vec(&sink_inputs).unwrap()),
        CommandResult::success(serde_json::to_vec(&source_outputs).unwrap()),
    ]);

    let report = PulseAudioMix::new(runner.clone())
        .apply(AudioMixVolumes {
            microphone_original_percent: 31,
            microphone_translation_percent: 32,
            speaker_original_percent: 33,
            speaker_translation_percent: 34,
        })
        .unwrap();

    assert_eq!(
        report.updated_targets,
        vec![
            AudioMixTarget::SpeakerOriginal,
            AudioMixTarget::MicrophoneOriginal,
            AudioMixTarget::MicrophoneTranslation,
            AudioMixTarget::SpeakerTranslation,
        ]
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
