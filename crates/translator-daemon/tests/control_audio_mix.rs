use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use translator_audio::{CommandResult, CommandRunError, CommandRunner};
use translator_daemon::{
    ActiveDuplexRuntime, AdmittedDuplex, ApiControllers, ApiLimits, AudioMixApplication,
    AudioOperationGate, ControlApplication, ControlCommand, ControlFailure, ControlToken,
    DuplexRunner, DuplexRuntimeError, RuntimeMaintenance, RuntimeStore,
    build_router_with_controllers,
};

const TOKEN: &str = "4242424242424242424242424242424242424242424242424242424242424242";
const DEADLINE: Duration = Duration::from_secs(2);

#[derive(Default)]
struct PulseState {
    volumes: [u32; 2],
    sets: Vec<Vec<String>>,
}

struct Pulse {
    state: Mutex<PulseState>,
    fail_at: AtomicUsize,
    block_at: AtomicUsize,
    entered: tokio::sync::Notify,
    released: (Mutex<bool>, Condvar),
}

impl Default for Pulse {
    fn default() -> Self {
        Self {
            state: Mutex::new(PulseState {
                volumes: [32768, 49152],
                sets: Vec::new(),
            }),
            fail_at: AtomicUsize::new(0),
            block_at: AtomicUsize::new(0),
            entered: tokio::sync::Notify::new(),
            released: (Mutex::new(false), Condvar::new()),
        }
    }
}

struct PulseRunner(Arc<Pulse>);

impl CommandRunner for PulseRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        _deadline: std::time::Instant,
    ) -> Result<CommandResult, CommandRunError> {
        assert_eq!(program, "pactl");
        let mut state = self.0.state.lock().unwrap();
        if args[0] == "--format=json" {
            let inputs: Vec<_> = if args[2] == "sink-inputs" {
                ["translator-outgoing-playback", "translator-incoming-playback"]
                    .into_iter().enumerate().map(|(index, media)| json!({
                        "index": 43 + index,
                        "channel_map": "mono",
                        "volume": {"mono": {"value": state.volumes[index]}},
                        "properties": {"application.name": "translator-daemon", "media.name": media},
                    })).collect()
            } else {
                Vec::new()
            };
            return Ok(CommandResult::success(serde_json::to_vec(&inputs).unwrap()));
        }
        assert_eq!(args[0], "set-sink-input-volume");
        let index = args[1].parse::<usize>().unwrap() - 43;
        state.volumes[index] = match args[2].strip_suffix('%') {
            Some(percent) => percent.parse::<u32>().unwrap() * 65536 / 100,
            None => args[2].parse().unwrap(),
        };
        state.sets.push(args.to_vec());
        let count = state.sets.len();
        drop(state);
        if self.0.block_at.load(Ordering::SeqCst) == count {
            self.0.entered.notify_one();
            let (released, timeout) = self
                .0
                .released
                .1
                .wait_timeout_while(self.0.released.0.lock().unwrap(), DEADLINE, |released| {
                    !*released
                })
                .unwrap();
            if timeout.timed_out() && !*released {
                return Err(CommandRunError::TimedOut);
            }
        }
        if self.0.fail_at.load(Ordering::SeqCst) == count {
            return Err(CommandRunError::TimedOut);
        }
        Ok(CommandResult::success(Vec::new()))
    }
}

struct Native;

impl DuplexRunner for Native {
    fn start(
        &self,
        _: AdmittedDuplex,
        _: tokio::time::Instant,
    ) -> translator_daemon::DuplexStartResult {
        Ok(Box::new(Native))
    }
}

impl ActiveDuplexRuntime for Native {
    fn stop(&mut self, _: tokio::time::Instant) -> Result<(), DuplexRuntimeError> {
        Ok(())
    }
}

impl RuntimeMaintenance for Native {
    fn refresh(&self, _: &RuntimeStore) -> Result<(), ControlFailure> {
        Ok(())
    }
}

impl translator_daemon::RuntimeFactsSource for Native {
    fn inspect(
        &self,
        _: std::time::Instant,
    ) -> Result<translator_daemon::RuntimeFacts, translator_daemon::FactsError> {
        use translator_audio::*;
        let selection = |name: &str| DeviceSelectionState {
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
        };
        Ok(translator_daemon::RuntimeFacts {
            devices: DeviceFacts {
                source: selection("alsa_input.physical"),
                sink: selection("alsa_output.headphones"),
                output_mode: OutputMode::Headphones,
                aec_capability: AecCapability::Unavailable,
            },
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
                resolution: RouteResolution::NoCandidate,
            },
        })
    }
}

fn fixture() -> (Arc<Pulse>, RuntimeStore, Arc<ControlApplication>, Router) {
    let pulse = Arc::new(Pulse::default());
    let store = RuntimeStore::default();
    let control = ControlApplication::spawn(
        store.clone(),
        Arc::new(Native),
        AudioOperationGate::new(),
        Arc::new(Native),
        Arc::new(Native),
        Some(Arc::new(AudioMixApplication::new(PulseRunner(
            pulse.clone(),
        )))),
    );
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(control.clone()),
            ..ApiControllers::default()
        },
    );
    (pulse, store, control, router)
}

fn request(method: Method, path: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("authorization", format!("Bearer {TOKEN}"))
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

async fn snapshot_event(body: &mut Body) -> Value {
    let frame = tokio::time::timeout(DEADLINE, body.frame())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let data = frame.into_data().unwrap();
    let text = std::str::from_utf8(&data).unwrap();
    assert!(text.contains("event: snapshot"));
    serde_json::from_str(
        text.lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn failed_physical_patch_never_changes_http_or_sse_desired_state() {
    let (pulse, store, control, router) = fixture();
    control.execute(ControlCommand::Start).await.unwrap();
    let before = store.snapshot().audio_mix;
    let mut events = router
        .clone()
        .oneshot(request(Method::GET, "/v1/events/stream", ""))
        .await
        .unwrap()
        .into_body();
    let initial = snapshot_event(&mut events).await;
    assert_eq!(initial["audio_mix"], serde_json::to_value(before).unwrap());
    pulse
        .fail_at
        .store(pulse.state.lock().unwrap().sets.len() + 2, Ordering::SeqCst);
    let response = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/audio-mix",
            r#"{"microphone_translation_percent":80,"speaker_translation_percent":90}"#,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let problem: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(problem["code"], "audio_mix_apply_failed");
    assert_eq!(store.snapshot().audio_mix, before);
    assert_eq!(pulse.state.lock().unwrap().volumes, [65536, 65536]);
    assert!(
        futures_util::poll!(events.frame()).is_pending(),
        "failed candidate must not emit a successful snapshot"
    );
    control
        .execute(ControlCommand::ReconcileAudio)
        .await
        .unwrap();
    {
        let state = pulse.state.lock().unwrap();
        assert_eq!(
            &state.sets[state.sets.len() - 2..],
            [
                vec!["set-sink-input-volume", "43", "100%"],
                vec!["set-sink-input-volume", "44", "100%"],
            ]
        );
    }
    let response = router
        .oneshot(request(Method::GET, "/v1/status", ""))
        .await
        .unwrap();
    let status: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(status["audio_mix"], initial["audio_mix"]);
    drop(events);
    control.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancelled_http_and_watchdog_are_drained_before_shutdown_returns() {
    let (pulse, store, control, router) = fixture();
    control.execute(ControlCommand::Start).await.unwrap();
    let before = store.snapshot().audio_mix;
    pulse
        .block_at
        .store(pulse.state.lock().unwrap().sets.len() + 1, Ordering::SeqCst);
    let patch = tokio::spawn(router.oneshot(request(
        Method::PATCH,
        "/v1/audio-mix",
        r#"{"microphone_translation_percent":80,"speaker_translation_percent":90}"#,
    )));
    tokio::time::timeout(DEADLINE, pulse.entered.notified())
        .await
        .unwrap();
    patch.abort();
    assert!(patch.await.unwrap_err().is_cancelled());
    assert_eq!(
        store.snapshot().audio_mix,
        before,
        "partial physical writes cannot publish desired"
    );
    {
        let mut reconcile = Box::pin(control.execute(ControlCommand::ReconcileAudio));
        assert!(futures_util::poll!(reconcile.as_mut()).is_pending());
    }
    let mut shutdown = Box::pin(control.shutdown());
    let pending = futures_util::poll!(shutdown.as_mut()).is_pending();
    *pulse.released.0.lock().unwrap() = true;
    pulse.released.1.notify_all();
    tokio::time::timeout(DEADLINE, shutdown)
        .await
        .unwrap()
        .unwrap();
    assert!(
        pending,
        "shutdown cannot return before accepted physical work finishes"
    );
    let after = store.snapshot();
    assert_eq!(after.audio_mix.microphone_translation_percent, 80);
    assert_eq!(after.audio_mix.speaker_translation_percent, 90);
    assert!(!after.translation_running);
    let state = pulse.state.lock().unwrap();
    assert_eq!(
        &state.sets[2..6],
        [
            vec!["set-sink-input-volume", "43", "80%"],
            vec!["set-sink-input-volume", "44", "90%"],
            vec!["set-sink-input-volume", "43", "80%"],
            vec!["set-sink-input-volume", "44", "90%"],
        ],
        "queued watchdog must use the newly committed candidate"
    );
    assert_eq!(
        state.volumes,
        [0, 0],
        "joined shutdown must physically reconcile bypass"
    );
}
