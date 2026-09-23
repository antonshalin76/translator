use std::sync::{Arc, Mutex};
use translator_audio::{CommandResult, CommandRunError, CommandRunner, MixPercent, PulseAudioMix};

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
