use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use translator_audio::{
    AudioMixTarget, CommandResult, CommandRunError, CommandRunner, MixPercent, PulseAudioMix,
    SystemCommandRunner,
};

struct Runner {
    input: serde_json::Value,
    sets: Arc<Mutex<Vec<Vec<String>>>>,
}

impl CommandRunner for Runner {
    fn run_until(
        &self,
        _program: &str,
        args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        if args[0] == "--format=json" {
            let value = if args[2] == "sink-inputs" {
                serde_json::json!([self.input])
            } else {
                serde_json::json!([])
            };
            return Ok(CommandResult::success(serde_json::to_vec(&value).unwrap()));
        }
        self.sets.lock().unwrap().push(args.to_vec());
        Ok(CommandResult::success(Vec::new()))
    }
}

fn input() -> serde_json::Value {
    serde_json::json!({"index":43, "channel_map":"front-right,front-left",
        "volume":{"front-left":{"value":32768},"front-right":{"value":49152}},
        "properties":{"application.name":"translator-daemon","media.name":"translator-outgoing-playback"}})
}

#[test]
fn typed_plan_preserves_exact_channel_order_and_set_bounds() {
    let sets = Arc::new(Mutex::new(Vec::new()));
    let device = PulseAudioMix::new(Runner {
        input: input(),
        sets: sets.clone(),
    });
    let plan = device.discover().unwrap();
    assert_eq!(plan.entries().len(), 1);
    let entry = &plan.entries()[0];
    device
        .set_percent(entry, MixPercent::try_from(0).unwrap())
        .unwrap();
    device
        .set_percent(entry, MixPercent::try_from(100).unwrap())
        .unwrap();
    assert!(MixPercent::try_from(101).is_err());
    device.restore_raw(entry).unwrap();
    assert_eq!(
        *sets.lock().unwrap(),
        vec![
            vec!["set-sink-input-volume", "43", "0%"],
            vec!["set-sink-input-volume", "43", "100%"],
            vec!["set-sink-input-volume", "43", "49152", "32768"]
        ]
    );
}

#[test]
fn mono_plan_restores_one_exact_raw_channel() {
    let sets = Arc::new(Mutex::new(Vec::new()));
    let mut value = input();
    value["channel_map"] = serde_json::json!("mono");
    value["volume"] = serde_json::json!({"mono": {"value": 65535}});
    let device = PulseAudioMix::new(Runner {
        input: value,
        sets: sets.clone(),
    });
    let plan = device.discover().unwrap();
    assert_eq!(plan.entries().len(), 1);
    device
        .set_percent(&plan.entries()[0], MixPercent::try_from(100).unwrap())
        .unwrap();
    device.restore_raw(&plan.entries()[0]).unwrap();
    assert_eq!(
        *sets.lock().unwrap(),
        [
            vec!["set-sink-input-volume", "43", "100%"],
            vec!["set-sink-input-volume", "43", "65535"],
        ]
    );
}

#[test]
fn malformed_prior_channels_fail_discovery_without_physical_writes() {
    let mut cases = Vec::new();
    for map in ["", "front-left,front-left", "front-left", "front-left,"] {
        let mut value = input();
        value["channel_map"] = serde_json::json!(map);
        cases.push(value);
    }
    for raw in [
        serde_json::json!(-1),
        serde_json::json!(2147483648u64),
        serde_json::json!(4294967296u64),
        serde_json::json!("65536"),
        serde_json::json!(null),
    ] {
        let mut value = input();
        value["volume"]["front-left"]["value"] = raw;
        cases.push(value);
    }
    for value in cases {
        let sets = Arc::new(Mutex::new(Vec::new()));
        let device = PulseAudioMix::new(Runner {
            input: value,
            sets: sets.clone(),
        });
        assert!(device.discover().is_err());
        assert!(sets.lock().unwrap().is_empty());
    }
}

struct SchemaRunner {
    inventories: HashMap<&'static str, serde_json::Value>,
    sets: Arc<Mutex<Vec<Vec<String>>>>,
}

impl CommandRunner for SchemaRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        if args.first().map(String::as_str) == Some("--format=json") {
            let value = self
                .inventories
                .get(args[2].as_str())
                .expect("known inventory");
            return Ok(CommandResult::success(serde_json::to_vec(value).unwrap()));
        }
        self.sets.lock().unwrap().push(args.to_vec());
        Ok(CommandResult::success(Vec::new()))
    }
}

fn real_schema_fixture() -> HashMap<&'static str, serde_json::Value> {
    HashMap::from([
        (
            "sink-inputs",
            serde_json::json!([
                {"index": 41, "owner_module": "10", "sink": 1,
                 "channel_map": "front-left,front-right",
                 "volume": {"front-left": {"value": 65536}, "front-right": {"value": 65536}},
                 "properties": {"media.name": "loopback-microphone-original", "translator.owner": "true"}},
                {"index": 42, "owner_module": "11", "sink": 0,
                 "channel_map": "front-left,front-right",
                 "volume": {"front-left": {"value": 65536}, "front-right": {"value": 65536}},
                 "properties": {"media.name": "loopback-speaker-original", "translator.owner": "true"}},
                {"index": 43, "owner_module": "12", "sink": 1,
                 "channel_map": "front-left,front-right",
                 "volume": {"front-left": {"value": 65536}, "front-right": {"value": 65536}},
                 "properties": {"media.name": "loopback-microphone-original"}}
            ]),
        ),
        (
            "source-outputs",
            serde_json::json!([
                {"owner_module": "10", "source": 0,
                 "properties": {"media.name": "loopback-microphone-original", "translator.owner": "true"}},
                {"owner_module": "11", "source": 2,
                 "properties": {"media.name": "loopback-speaker-original", "translator.owner": "true"}},
                {"owner_module": "12", "source": 0,
                 "properties": {"media.name": "loopback-microphone-original"}}
            ]),
        ),
        (
            "sinks",
            serde_json::json!([
                {"index": 0, "name": "alsa_output.headphones"},
                {"index": 1, "name": "translator_mic_out"}
            ]),
        ),
        (
            "sources",
            serde_json::json!([
                {"index": 0, "name": "alsa_input.microphone"},
                {"index": 2, "name": "translator_remote_in.monitor"}
            ]),
        ),
    ])
}

#[test]
fn owned_original_streams_use_actual_pulse_module_and_endpoint_schema() {
    let sets = Arc::new(Mutex::new(Vec::new()));
    let device = PulseAudioMix::new(SchemaRunner {
        inventories: real_schema_fixture(),
        sets: sets.clone(),
    });
    let plan = device.discover().unwrap();
    assert_eq!(plan.entries().len(), 2);
    assert_eq!(
        plan.entries()[0].target(),
        AudioMixTarget::MicrophoneOriginal
    );
    assert_eq!(plan.entries()[1].target(), AudioMixTarget::SpeakerOriginal);
    for (entry, percent) in plan.entries().iter().zip([31, 33]) {
        device
            .set_percent(entry, MixPercent::try_from(percent).unwrap())
            .unwrap();
    }
    assert_eq!(
        *sets.lock().unwrap(),
        [
            vec!["set-sink-input-volume", "41", "31%"],
            vec!["set-sink-input-volume", "42", "33%"],
        ]
    );
}

#[test]
fn claimed_original_with_wrong_endpoint_fails_before_any_set() {
    let mut inventories = real_schema_fixture();
    inventories.get_mut("sink-inputs").unwrap()[0]["sink"] = serde_json::json!(0);
    let sets = Arc::new(Mutex::new(Vec::new()));
    let device = PulseAudioMix::new(SchemaRunner {
        inventories,
        sets: sets.clone(),
    });
    assert!(device.discover().is_err());
    assert!(sets.lock().unwrap().is_empty());
}

#[test]
fn claimed_speaker_with_wrong_source_fails_before_any_set() {
    let mut inventories = real_schema_fixture();
    inventories.get_mut("source-outputs").unwrap()[1]["source"] = serde_json::json!(0);
    let sets = Arc::new(Mutex::new(Vec::new()));
    let device = PulseAudioMix::new(SchemaRunner {
        inventories,
        sets: sets.clone(),
    });
    assert!(device.discover().is_err());
    assert!(sets.lock().unwrap().is_empty());
}

#[test]
fn claimed_original_requires_paired_module_and_owner_marker() {
    for case in 0..2 {
        let mut inventories = real_schema_fixture();
        let output = &mut inventories.get_mut("source-outputs").unwrap()[0];
        if case == 0 {
            output["owner_module"] = serde_json::json!("13");
        } else {
            output["properties"]
                .as_object_mut()
                .unwrap()
                .remove("translator.owner");
        }
        let sets = Arc::new(Mutex::new(Vec::new()));
        let device = PulseAudioMix::new(SchemaRunner {
            inventories,
            sets: sets.clone(),
        });
        assert!(device.discover().is_err());
        assert!(sets.lock().unwrap().is_empty());
    }
}

#[test]
fn missing_or_duplicate_referenced_endpoint_fails_before_any_set() {
    for case in 0..4 {
        let mut inventories = real_schema_fixture();
        let kind = if case < 2 { "sinks" } else { "sources" };
        let endpoints = inventories.get_mut(kind).unwrap().as_array_mut().unwrap();
        let referenced = if case < 2 { 1 } else { 2 };
        if case % 2 == 0 {
            endpoints.retain(|endpoint| endpoint["index"] != referenced);
        } else {
            endpoints.push(serde_json::json!({"index": referenced, "name": "duplicate"}));
        }
        let sets = Arc::new(Mutex::new(Vec::new()));
        let device = PulseAudioMix::new(SchemaRunner {
            inventories,
            sets: sets.clone(),
        });
        assert!(device.discover().is_err());
        assert!(sets.lock().unwrap().is_empty());
    }
}

#[test]
fn duplicate_owned_stream_index_or_module_fails_before_any_set() {
    for case in 0..2 {
        let mut inventories = real_schema_fixture();
        let inputs = inventories
            .get_mut("sink-inputs")
            .unwrap()
            .as_array_mut()
            .unwrap();
        if case == 0 {
            inputs[1]["index"] = serde_json::json!(41);
        } else {
            inputs[1]["owner_module"] = serde_json::json!("10");
        }
        let sets = Arc::new(Mutex::new(Vec::new()));
        let device = PulseAudioMix::new(SchemaRunner {
            inventories,
            sets: sets.clone(),
        });
        assert!(device.discover().is_err());
        assert!(sets.lock().unwrap().is_empty());
    }
}

#[test]
#[ignore = "requires a disposable private PulseAudio socket and virtual fixture sinks"]
fn private_pulse_original_volume_set_readback_and_restore() {
    let server = std::env::var("PULSE_SERVER").expect("private PULSE_SERVER required");
    assert!(
        server.starts_with("unix:/tmp/translator-loopback-") && server.ends_with("/native"),
        "refusing non-fixture PulseAudio server"
    );

    fn pactl(args: &[&str]) -> Vec<u8> {
        let output = std::process::Command::new("pactl")
            .args(args)
            .output()
            .expect("pactl executable");
        assert!(output.status.success(), "private pactl command failed");
        output.stdout
    }

    struct PrivateModule(String);
    impl PrivateModule {
        fn unload(&mut self) {
            let output = std::process::Command::new("pactl")
                .args(["unload-module", &self.0])
                .output()
                .expect("pactl executable");
            assert!(output.status.success(), "private module unload failed");
            self.0.clear();
        }
    }
    impl Drop for PrivateModule {
        fn drop(&mut self) {
            if !self.0.is_empty() {
                let _ = std::process::Command::new("pactl")
                    .args(["unload-module", &self.0])
                    .output();
            }
        }
    }

    let microphone_id = String::from_utf8(pactl(&[
        "load-module",
        "module-loopback",
        "source=translator_test_mic.monitor",
        "sink=translator_mic_out",
        "latency_msec=20",
        "source_dont_move=true",
        "sink_dont_move=true",
        "source_output_properties='media.name=loopback-microphone-original translator.owner=true'",
        "sink_input_properties='media.name=loopback-microphone-original translator.owner=true'",
    ]))
    .unwrap()
    .trim()
    .to_owned();
    let mut microphone = PrivateModule(microphone_id.clone());
    let mut remote_sink = PrivateModule(
        String::from_utf8(pactl(&[
            "load-module",
            "module-null-sink",
            "sink_name=translator_remote_in",
        ]))
        .unwrap()
        .trim()
        .to_owned(),
    );
    let speaker_id = String::from_utf8(pactl(&[
        "load-module",
        "module-loopback",
        "source=translator_remote_in.monitor",
        "sink=translator_test_out",
        "latency_msec=20",
        "source_dont_move=true",
        "sink_dont_move=true",
        "source_output_properties='media.name=loopback-speaker-original translator.owner=true'",
        "sink_input_properties='media.name=loopback-speaker-original translator.owner=true'",
    ]))
    .unwrap()
    .trim()
    .to_owned();
    let mut speaker = PrivateModule(speaker_id.clone());
    let device = PulseAudioMix::new(SystemCommandRunner);
    let plan = device.discover().unwrap();
    assert_eq!(plan.entries().len(), 2);
    let microphone_entry = plan
        .entries()
        .iter()
        .find(|entry| entry.target() == AudioMixTarget::MicrophoneOriginal)
        .unwrap();
    let speaker_entry = plan
        .entries()
        .iter()
        .find(|entry| entry.target() == AudioMixTarget::SpeakerOriginal)
        .unwrap();

    let raw = |module_id: &str, media_name: &str| -> Vec<u32> {
        let inputs: serde_json::Value =
            serde_json::from_slice(&pactl(&["--format=json", "list", "sink-inputs"])).unwrap();
        let input = inputs
            .as_array()
            .unwrap()
            .iter()
            .find(|input| {
                input["owner_module"] == module_id
                    && input["properties"]["media.name"] == media_name
            })
            .unwrap();
        ["front-left", "front-right"]
            .iter()
            .map(|channel| input["volume"][channel]["value"].as_u64().unwrap() as u32)
            .collect()
    };
    let mic_before = raw(&microphone_id, "loopback-microphone-original");
    let speaker_before = raw(&speaker_id, "loopback-speaker-original");
    device
        .set_percent(microphone_entry, MixPercent::try_from(31).unwrap())
        .unwrap();
    device
        .set_percent(speaker_entry, MixPercent::try_from(33).unwrap())
        .unwrap();
    for value in raw(&microphone_id, "loopback-microphone-original") {
        assert!(
            (value as i64 - 20_316).abs() < 128,
            "unexpected 31% raw volume: {value}"
        );
    }
    for value in raw(&speaker_id, "loopback-speaker-original") {
        assert!(
            (value as i64 - 21_627).abs() < 128,
            "unexpected 33% raw volume: {value}"
        );
    }
    device.restore_raw(microphone_entry).unwrap();
    device.restore_raw(speaker_entry).unwrap();
    assert_eq!(
        raw(&microphone_id, "loopback-microphone-original"),
        mic_before
    );
    assert_eq!(
        raw(&speaker_id, "loopback-speaker-original"),
        speaker_before
    );
    speaker.unload();
    remote_sink.unload();
    microphone.unload();
    let inputs: serde_json::Value =
        serde_json::from_slice(&pactl(&["--format=json", "list", "sink-inputs"])).unwrap();
    assert!(inputs.as_array().unwrap().iter().all(|input| {
        input["owner_module"] != microphone_id && input["owner_module"] != speaker_id
    }));
}
