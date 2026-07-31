use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use bytes::Bytes;
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;
use translator_audio::{
    AudioGraphState, GraphHealth, RouteResolution, RoutingSafeError, RoutingState,
};
use translator_daemon::{
    ApiControllers, ApiLimits, ControlFailure, ControlToken, DebugCaptureLimits, DebugCaptureStore,
    DebugTextEvent, ManualRouteController, RoundTripController, RoundTripDebugText,
    RoundTripSelfTestState, RoundTripStatus, RuntimeSnapshot, RuntimeStore, TranslationController,
    build_router, build_router_with_controllers, build_router_with_manual_routes,
    validate_listen_address,
};

const TOKEN: &str = "4242424242424242424242424242424242424242424242424242424242424242";

struct FakeManualRoutes {
    selected_stream: AtomicUsize,
}

impl ManualRouteController for FakeManualRoutes {
    fn reconcile(&self, stream_id: u32) -> Result<RoutingState, RoutingSafeError> {
        self.selected_stream
            .store(stream_id as usize, Ordering::SeqCst);
        Ok(RoutingState {
            candidates: Vec::new(),
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: RouteResolution::NoCandidate,
        })
    }
}

struct FakeTranslationController {
    starts: AtomicUsize,
    stops: AtomicUsize,
    snapshots: Mutex<Vec<RuntimeSnapshot>>,
}

impl TranslationController for FakeTranslationController {
    fn start(&self, snapshot: RuntimeSnapshot) -> Result<(), ControlFailure> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        self.snapshots.lock().unwrap().push(snapshot);
        Ok(())
    }

    fn stop(&self) -> Result<(), ControlFailure> {
        self.stops.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct RejectingTranslationController;

impl TranslationController for RejectingTranslationController {
    fn start(&self, _snapshot: RuntimeSnapshot) -> Result<(), ControlFailure> {
        Err(ControlFailure {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "provider_unavailable",
        })
    }

    fn stop(&self) -> Result<(), ControlFailure> {
        Err(ControlFailure {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "pipeline_stop_failed",
        })
    }
}

struct LeakyRoundTripController {
    marker: &'static str,
}

impl RoundTripController for LeakyRoundTripController {
    fn start(&self) -> Result<RoundTripSelfTestState, ControlFailure> {
        Ok(self.state())
    }

    fn stop(&self) -> Result<RoundTripSelfTestState, ControlFailure> {
        Ok(self.state())
    }
}

impl LeakyRoundTripController {
    fn state(&self) -> RoundTripSelfTestState {
        RoundTripSelfTestState {
            availability: "available",
            preconditions: None,
            status: RoundTripStatus {
                debug_text: Some(RoundTripDebugText {
                    transcript: self.marker.to_owned(),
                    translation: self.marker.to_owned(),
                }),
                ..RoundTripStatus::default()
            },
        }
    }
}

fn app() -> axum::Router {
    app_with(RuntimeStore::default(), ApiLimits::default())
}

fn app_with(store: RuntimeStore, limits: ApiLimits) -> axum::Router {
    build_router(store, ControlToken::parse(TOKEN).unwrap(), limits)
}

fn request(method: Method, uri: &str, token: Option<&str>, body: Body) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(token) = token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    builder
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .unwrap()
}

async fn json_body(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn assert_problem(
    response: axum::response::Response,
    status: StatusCode,
    title: &str,
    code: &str,
) -> Value {
    assert_eq!(response.status(), status);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/problem+json"
    );
    let body = json_body(response).await;
    assert_eq!(body["type"], format!("urn:translator:error:{code}"));
    assert_eq!(body["title"], title);
    assert_eq!(body["status"], status.as_u16());
    assert_eq!(body["code"], code);
    body
}

struct SseReader {
    body: Body,
    buffered: Vec<u8>,
}

impl SseReader {
    fn new(body: Body) -> Self {
        Self {
            body,
            buffered: Vec::new(),
        }
    }

    async fn next_record(&mut self) -> Option<String> {
        loop {
            if let Some(end) = self
                .buffered
                .windows(2)
                .position(|window| window == b"\n\n")
            {
                let record = self.buffered.drain(..end + 2).collect::<Vec<_>>();
                return Some(String::from_utf8(record).unwrap());
            }
            let frame = tokio::time::timeout(Duration::from_secs(2), self.body.frame())
                .await
                .expect("SSE record timed out")?
                .unwrap();
            self.buffered.extend_from_slice(&frame.into_data().unwrap());
        }
    }
}

#[tokio::test]
async fn invalid_bearer_is_rejected_before_body_limit_or_json_parsing() {
    let body_polls = Arc::new(AtomicUsize::new(0));
    let observed_polls = Arc::clone(&body_polls);
    let unreadable_body = Body::from_stream(stream::poll_fn(move |_| {
        observed_polls.fetch_add(1, Ordering::SeqCst);
        panic!("unauthorized request body must not be polled");
        #[allow(unreachable_code)]
        std::task::Poll::Ready(Some(Ok::<Bytes, Infallible>(Bytes::new())))
    }));

    let response = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-text",
            Some("wrong-token"),
            unreadable_body,
        ))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::UNAUTHORIZED,
        "Unauthorized",
        "invalid_bearer",
    )
    .await;
    assert_eq!(body_polls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn authorized_control_body_is_limited_to_64_kib() {
    let response = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-text",
            Some(TOKEN),
            Body::from(vec![b'x'; 65_537]),
        ))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::PAYLOAD_TOO_LARGE,
        "Payload Too Large",
        "body_too_large",
    )
    .await;
}

#[tokio::test]
async fn malformed_authorized_json_uses_privacy_safe_problem_details() {
    let marker = "private-json-marker";
    let response = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-text",
            Some(TOKEN),
            Body::from(format!("{{\"enabled\":{marker}}}")),
        ))
        .await
        .unwrap();

    let body = assert_problem(
        response,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "invalid_json",
    )
    .await;
    assert!(!body.to_string().contains(marker));
}

#[tokio::test]
async fn missing_bearer_cannot_read_status() {
    let response = app()
        .oneshot(request(Method::GET, "/v1/status", None, Body::empty()))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::UNAUTHORIZED,
        "Unauthorized",
        "invalid_bearer",
    )
    .await;
}

#[tokio::test]
async fn valid_bearer_reads_privacy_safe_status() {
    let response = app()
        .oneshot(request(
            Method::GET,
            "/v1/status",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["translation_running"], false);
    assert_eq!(body["debug_text_enabled"], false);
    assert_eq!(body["debug_capture_enabled"], false);
    assert_eq!(body["provider_id"], "local");
    assert_eq!(body["directions"][0]["direction_id"], "microphone");
    assert_eq!(body["directions"][0]["source_language"], "ru");
    assert_eq!(body["directions"][0]["target_language"], "en");
    assert_eq!(body["directions"][0]["enabled"], true);
    assert_eq!(body["directions"][1]["enabled"], true);
    assert_eq!(body["latency_policy"][0]["current_mode"], "quality_first");
    assert_eq!(body["audio_mix"]["microphone_original_percent"], 0);
    assert_eq!(body["audio_mix"]["microphone_translation_percent"], 100);
    assert_eq!(body["audio_mix"]["speaker_original_percent"], 0);
    assert_eq!(body["audio_mix"]["speaker_translation_percent"], 100);
    assert!(body.get("control_token").is_none());
}

#[tokio::test]
async fn graph_and_route_reads_report_typed_unavailable_state() {
    let router = app();
    let graph = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/audio-graph",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let graph = json_body(graph).await;
    assert_eq!(graph["available"], false);
    assert!(graph["value"].is_null());

    let routes = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/routes",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let routes = json_body(routes).await;
    assert_eq!(routes["available"], false);
    assert!(routes["value"].is_null());

    let candidates = router
        .oneshot(request(
            Method::GET,
            "/v1/routes/candidates",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(candidates).await, serde_json::json!([]));
}

#[tokio::test]
async fn audio_graph_is_available_only_when_health_is_ready() {
    for (health, expected) in [
        (GraphHealth::Ready, true),
        (GraphHealth::Degraded, false),
        (GraphHealth::Error, false),
    ] {
        let store = RuntimeStore::default();
        store.set_audio_graph(AudioGraphState {
            health,
            endpoints: Vec::new(),
            owned_module_ids: Vec::new(),
            safe_error: None,
        });
        let response = app_with(store, ApiLimits::default())
            .oneshot(request(
                Method::GET,
                "/v1/audio-graph",
                Some(TOKEN),
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(json_body(response).await["available"], expected);
    }
}

#[tokio::test]
async fn self_test_controller_text_obeys_global_debug_text_lifecycle() {
    let marker = "private-controller-round-trip-marker";
    let store = RuntimeStore::default();
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            round_trip: Some(Arc::new(LeakyRoundTripController { marker })),
            ..ApiControllers::default()
        },
    );

    let hidden_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(hidden_start.status(), StatusCode::OK);
    assert!(!json_body(hidden_start).await.to_string().contains(marker));
    assert!(
        !serde_json::to_string(&store.snapshot())
            .unwrap()
            .contains(marker)
    );

    store.set_debug_text_enabled(true);
    let visible_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(visible_start.status(), StatusCode::OK);
    assert!(json_body(visible_start).await.to_string().contains(marker));

    let stopped = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(stopped.status(), StatusCode::OK);
    assert!(!json_body(stopped).await.to_string().contains(marker));

    let stopped_status = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/self-test/round-trip",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(stopped_status.status(), StatusCode::OK);
    assert!(!json_body(stopped_status).await.to_string().contains(marker));

    let visible_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert!(json_body(visible_start).await.to_string().contains(marker));

    let provider_switched = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/provider",
            Some(TOKEN),
            Body::from(r#"{"provider_id":"openai","cloud_opt_in":true}"#),
        ))
        .await
        .unwrap();
    assert_eq!(provider_switched.status(), StatusCode::OK);
    assert!(
        !json_body(provider_switched)
            .await
            .to_string()
            .contains(marker)
    );
    let after_provider_switch = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/self-test/round-trip",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(after_provider_switch.status(), StatusCode::OK);
    assert!(
        !json_body(after_provider_switch)
            .await
            .to_string()
            .contains(marker)
    );

    let visible_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert!(json_body(visible_start).await.to_string().contains(marker));

    store.set_debug_text_enabled(false);
    store.set_debug_text_enabled(true);
    let after_reenable = router
        .oneshot(request(
            Method::GET,
            "/v1/self-test/round-trip",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(after_reenable.status(), StatusCode::OK);
    assert!(!json_body(after_reenable).await.to_string().contains(marker));
}

#[tokio::test]
async fn typed_control_patches_update_only_the_selected_runtime_contract() {
    let router = app();
    let direction = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"microphone","source_language":"en","target_language":"ru"}"#,
            ),
        ))
        .await
        .unwrap();
    let direction = json_body(direction).await;
    assert_eq!(direction["directions"][0]["source_language"], "en");
    assert_eq!(direction["directions"][0]["target_language"], "ru");
    assert_eq!(direction["directions"][0]["enabled"], true);
    assert_eq!(direction["directions"][1]["source_language"], "en");

    let disabled = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(r#"{"direction_id":"speaker","enabled":false}"#),
        ))
        .await
        .unwrap();
    let disabled = json_body(disabled).await;
    assert_eq!(disabled["directions"][0]["enabled"], true);
    assert_eq!(disabled["directions"][0]["source_language"], "en");
    assert_eq!(disabled["directions"][1]["enabled"], false);
    assert_eq!(disabled["directions"][1]["source_language"], "en");

    let blocked_provider = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/provider",
            Some(TOKEN),
            Body::from(r#"{"provider_id":"openai"}"#),
        ))
        .await
        .unwrap();
    assert_eq!(blocked_provider.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(blocked_provider).await["type"],
        "urn:translator:error:cloud_provider_opt_in_required"
    );

    let provider = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/provider",
            Some(TOKEN),
            Body::from(r#"{"provider_id":"openai","cloud_opt_in":true}"#),
        ))
        .await
        .unwrap();
    let provider = json_body(provider).await;
    assert_eq!(provider["provider_id"], "openai");
    assert_eq!(provider["audio_leaves_machine"], true);

    let latency = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/latency-policy",
            Some(TOKEN),
            Body::from(r#"{"direction_id":"speaker","current_mode":"balanced"}"#),
        ))
        .await
        .unwrap();
    let latency = json_body(latency).await;
    assert_eq!(
        latency["latency_policy"][0]["current_mode"],
        "quality_first"
    );
    assert_eq!(latency["latency_policy"][1]["current_mode"], "balanced");
    assert_eq!(latency["audio_mix"]["speaker_original_percent"], 0);

    let audio_mix = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/audio-mix",
            Some(TOKEN),
            Body::from(r#"{"speaker_original_percent":55,"microphone_translation_percent":72}"#),
        ))
        .await
        .unwrap();
    let audio_mix = json_body(audio_mix).await;
    assert_eq!(audio_mix["audio_mix"]["speaker_original_percent"], 55);
    assert_eq!(audio_mix["audio_mix"]["microphone_translation_percent"], 72);
    assert_eq!(audio_mix["audio_mix"]["microphone_original_percent"], 0);
    assert_eq!(audio_mix["audio_mix"]["speaker_translation_percent"], 100);

    let voice = router
        .oneshot(request(
            Method::PATCH,
            "/v1/voice-profiles",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"speaker","voice_profile":{"language":"ru","gender":"female","engine":"piper"}}"#,
            ),
        ))
        .await
        .unwrap();
    assert_eq!(
        json_body(voice).await["directions"][1]["voice_profile"]["gender"],
        "female"
    );
}

#[tokio::test]
async fn invalid_direction_and_voice_language_are_typed_problem_details() {
    let invalid_pair = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"microphone","source_language":"ru","target_language":"ru"}"#,
            ),
        ))
        .await
        .unwrap();
    assert_problem(
        invalid_pair,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "invalid_language_pair",
    )
    .await;

    let voice_mismatch = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/voice-profiles",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"speaker","voice_profile":{"language":"en","gender":"female","engine":"piper"}}"#,
            ),
        ))
        .await
        .unwrap();
    assert_problem(
        voice_mismatch,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "voice_language_mismatch",
    )
    .await;
}

#[tokio::test]
async fn audio_mix_rejects_out_of_range_and_unknown_fields() {
    let too_loud = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/audio-mix",
            Some(TOKEN),
            Body::from(r#"{"speaker_translation_percent":101}"#),
        ))
        .await
        .unwrap();
    assert_problem(
        too_loud,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "invalid_audio_mix_volume",
    )
    .await;

    let unknown = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/audio-mix",
            Some(TOKEN),
            Body::from(r#"{"speaker_translation_percent":80,"master_percent":80}"#),
        ))
        .await
        .unwrap();
    assert_problem(
        unknown,
        StatusCode::BAD_REQUEST,
        "Bad Request",
        "invalid_json",
    )
    .await;
}

#[tokio::test]
async fn debug_capture_enable_fails_closed_without_hardened_store() {
    let response = app()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-capture",
            Some(TOKEN),
            Body::from(r#"{"enabled":true}"#),
        ))
        .await
        .unwrap();
    assert_problem(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "debug_capture_unavailable",
    )
    .await;
}

#[tokio::test]
async fn debug_capture_api_creates_and_closes_real_exclusive_session() {
    let temp = tempfile::tempdir().unwrap();
    let store = RuntimeStore::default();
    store.configure_debug_capture(
        DebugCaptureStore::open(temp.path(), DebugCaptureLimits::new(1_000, 1_024, 0)).unwrap(),
    );
    let router = app_with(store, ApiLimits::default());

    let enabled = router
        .clone()
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-capture",
            Some(TOKEN),
            Body::from(r#"{"enabled":true}"#),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(enabled).await["debug_capture_enabled"], true);
    let captures = std::fs::read_dir(temp.path().join("translator/debug"))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(captures.len(), 1);

    let disabled = router
        .oneshot(request(
            Method::PATCH,
            "/v1/debug-capture",
            Some(TOKEN),
            Body::from(r#"{"enabled":false}"#),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(disabled).await["debug_capture_enabled"], false);
}

#[tokio::test]
async fn unavailable_manual_route_controller_fails_closed() {
    let response = app()
        .oneshot(request(
            Method::POST,
            "/v1/routes/manual-override",
            Some(TOKEN),
            Body::from(r#"{"stream_id":42}"#),
        ))
        .await
        .unwrap();
    assert_problem(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "routing_controller_unavailable",
    )
    .await;
}

#[tokio::test]
async fn manual_route_controller_result_is_published_to_runtime_state() {
    let controller = Arc::new(FakeManualRoutes {
        selected_stream: AtomicUsize::new(0),
    });
    let router = build_router_with_manual_routes(
        RuntimeStore::default(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        Some(controller.clone()),
    );
    let response = router
        .oneshot(request(
            Method::POST,
            "/v1/routes/manual-override",
            Some(TOKEN),
            Body::from(r#"{"stream_id":42}"#),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        json_body(response).await["routes"]["resolution"],
        "no_candidate"
    );
    assert_eq!(controller.selected_stream.load(Ordering::SeqCst), 42);
}

#[tokio::test]
async fn fifth_sse_subscriber_is_rejected_and_disconnect_releases_permit() {
    let router = app();
    let mut responses = Vec::new();
    for _ in 0..4 {
        let response = router
            .clone()
            .oneshot(request(
                Method::GET,
                "/v1/events/stream",
                Some(TOKEN),
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        responses.push(response);
    }

    let fifth = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(
        fifth,
        StatusCode::TOO_MANY_REQUESTS,
        "Too Many Requests",
        "sse_subscriber_limit",
    )
    .await;

    responses.pop();
    tokio::task::yield_now().await;

    let replacement = router
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(replacement.status(), StatusCode::OK);
}

#[tokio::test]
async fn partially_consumed_sse_holds_permit_until_body_is_dropped() {
    let router = app();
    let mut responses = Vec::new();
    for _ in 0..4 {
        let response = router
            .clone()
            .oneshot(request(
                Method::GET,
                "/v1/events/stream",
                Some(TOKEN),
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        responses.push(response);
    }

    let mut partial = responses.pop().unwrap();
    let mut partial_reader = SseReader::new(std::mem::take(partial.body_mut()));
    let first = partial_reader.next_record().await.unwrap();
    assert!(first.contains("event: snapshot"));

    let rejected = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(
        rejected,
        StatusCode::TOO_MANY_REQUESTS,
        "Too Many Requests",
        "sse_subscriber_limit",
    )
    .await;

    drop(partial_reader);
    drop(partial);
    tokio::task::yield_now().await;
    let replacement = router
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(replacement.status(), StatusCode::OK);
}

#[tokio::test]
async fn broadcast_lag_emits_resync_then_exactly_one_fresh_snapshot() {
    let store = RuntimeStore::with_event_capacity(1);
    let router = app_with(store.clone(), ApiLimits::default());
    let response = router
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let mut events = SseReader::new(response.into_body());
    assert!(
        events
            .next_record()
            .await
            .unwrap()
            .contains("event: snapshot")
    );

    store.publish_snapshot_changed();
    store.publish_snapshot_changed();
    store.publish_snapshot_changed();

    let resync = events.next_record().await.unwrap();
    let snapshot = events.next_record().await.unwrap();
    assert!(resync.contains("event: resync_required"));
    assert!(!resync.contains("event: snapshot"));
    assert!(snapshot.contains("event: snapshot"));
    assert_eq!(snapshot.matches("event: snapshot").count(), 1);
}

#[tokio::test]
async fn event_bus_shutdown_terminates_stream_cleanly() {
    let store = RuntimeStore::default();
    let router = app_with(store.clone(), ApiLimits::default());
    let response = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let mut events = SseReader::new(response.into_body());
    assert!(
        events
            .next_record()
            .await
            .unwrap()
            .contains("event: snapshot")
    );

    store.shutdown_events();
    assert!(events.next_record().await.is_none());

    let replacement = router
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_eq!(replacement.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test(start_paused = true)]
async fn sse_keepalive_is_comment_only() {
    let limits = ApiLimits::default().with_sse_keepalive(Duration::from_millis(10));
    let response = app_with(RuntimeStore::default(), limits)
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    let mut events = SseReader::new(response.into_body());
    assert!(
        events
            .next_record()
            .await
            .unwrap()
            .contains("event: snapshot")
    );

    tokio::time::advance(Duration::from_millis(11)).await;
    let keepalive = events.next_record().await.unwrap();
    assert!(keepalive.starts_with(':'));
    assert!(!keepalive.contains("data:"));
}

#[tokio::test]
async fn rejecting_fifth_sse_subscriber_does_not_interrupt_existing_streams() {
    let store = RuntimeStore::default();
    let router = app_with(store.clone(), ApiLimits::default());
    let mut responses = Vec::new();
    for _ in 0..4 {
        let response = router
            .clone()
            .oneshot(request(
                Method::GET,
                "/v1/events/stream",
                Some(TOKEN),
                Body::empty(),
            ))
            .await
            .unwrap();
        responses.push(response);
    }

    let fifth = router
        .clone()
        .oneshot(request(
            Method::GET,
            "/v1/events/stream",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(
        fifth,
        StatusCode::TOO_MANY_REQUESTS,
        "Too Many Requests",
        "sse_subscriber_limit",
    )
    .await;

    store.publish_snapshot_changed();
    for response in &mut responses {
        let mut events = SseReader::new(std::mem::take(response.body_mut()));
        assert!(
            events
                .next_record()
                .await
                .unwrap()
                .contains("event: snapshot")
        );
        assert!(
            events
                .next_record()
                .await
                .unwrap()
                .contains("event: snapshot")
        );
    }
}

#[tokio::test]
async fn self_test_start_is_typed_unavailable_until_task7_controller_exists() {
    let response = app()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "self_test_unavailable",
    )
    .await;
}

#[tokio::test]
async fn self_test_stop_is_typed_unavailable_until_task7_controller_exists() {
    let response = app()
        .oneshot(request(
            Method::POST,
            "/v1/self-test/round-trip/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();

    assert_problem(
        response,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "self_test_unavailable",
    )
    .await;
}

#[tokio::test]
async fn translation_mutations_fail_closed_without_provider_controller() {
    for path in ["/v1/translation/start", "/v1/translation/stop"] {
        let response = app()
            .oneshot(request(Method::POST, path, Some(TOKEN), Body::from("{}")))
            .await
            .unwrap();
        assert_problem(
            response,
            StatusCode::SERVICE_UNAVAILABLE,
            "Service Unavailable",
            "translation_controller_unavailable",
        )
        .await;
    }
}

#[tokio::test]
async fn translation_controller_success_owns_running_state_and_stop_clears_debug_text() {
    let store = RuntimeStore::default();
    store.set_debug_text_enabled(true);
    store.record_debug_text(DebugTextEvent::new("private-session-marker", "translation"));
    let controller = Arc::new(FakeTranslationController {
        starts: AtomicUsize::new(0),
        stops: AtomicUsize::new(0),
        snapshots: Mutex::new(Vec::new()),
    });
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(started).await["translation_running"], true);
    let stopped = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(stopped).await["translation_running"], false);
    assert_eq!(store.debug_text_status().event_count, 0);
    assert_eq!(controller.starts.load(Ordering::SeqCst), 1);
    assert_eq!(controller.stops.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn direction_patch_reconfigures_active_translation_for_channel_bypass() {
    let store = RuntimeStore::default();
    let controller = Arc::new(FakeTranslationController {
        starts: AtomicUsize::new(0),
        stops: AtomicUsize::new(0),
        snapshots: Mutex::new(Vec::new()),
    });
    let router = build_router_with_controllers(
        store,
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(started).await["translation_running"], true);

    let disabled = router
        .oneshot(request(
            Method::PATCH,
            "/v1/directions",
            Some(TOKEN),
            Body::from(r#"{"direction_id":"speaker","enabled":false}"#),
        ))
        .await
        .unwrap();
    let disabled = json_body(disabled).await;

    assert_eq!(disabled["translation_running"], true);
    assert_eq!(disabled["directions"][1]["enabled"], false);
    assert_eq!(controller.starts.load(Ordering::SeqCst), 2);
    assert_eq!(controller.stops.load(Ordering::SeqCst), 1);
    let snapshots = controller.snapshots.lock().unwrap();
    assert_eq!(snapshots.len(), 2);
    assert!(snapshots[0].directions[1].enabled);
    assert!(!snapshots[1].directions[1].enabled);
}

#[tokio::test]
async fn voice_profile_patch_reconfigures_active_translation() {
    let store = RuntimeStore::default();
    let controller = Arc::new(FakeTranslationController {
        starts: AtomicUsize::new(0),
        stops: AtomicUsize::new(0),
        snapshots: Mutex::new(Vec::new()),
    });
    let router = build_router_with_controllers(
        store,
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(controller.clone()),
            ..ApiControllers::default()
        },
    );

    let started = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_eq!(json_body(started).await["translation_running"], true);

    let patched = router
        .oneshot(request(
            Method::PATCH,
            "/v1/voice-profiles",
            Some(TOKEN),
            Body::from(
                r#"{"direction_id":"speaker","voice_profile":{"language":"ru","gender":"female","engine":"piper"}}"#,
            ),
        ))
        .await
        .unwrap();
    let patched = json_body(patched).await;

    assert_eq!(patched["translation_running"], true);
    assert_eq!(
        patched["directions"][1]["voice_profile"]["gender"],
        "female"
    );
    assert_eq!(controller.starts.load(Ordering::SeqCst), 2);
    assert_eq!(controller.stops.load(Ordering::SeqCst), 1);
    let snapshots = controller.snapshots.lock().unwrap();
    assert_eq!(snapshots.len(), 2);
    assert_eq!(
        snapshots[0].directions[1].voice_profile.gender,
        translator_core::VoiceGender::Male
    );
    assert_eq!(
        snapshots[1].directions[1].voice_profile.gender,
        translator_core::VoiceGender::Female
    );
}

#[tokio::test]
async fn translation_controller_failure_never_reports_a_false_runtime_transition() {
    let store = RuntimeStore::default();
    let router = build_router_with_controllers(
        store.clone(),
        ControlToken::parse(TOKEN).unwrap(),
        ApiLimits::default(),
        ApiControllers {
            translation: Some(Arc::new(RejectingTranslationController)),
            ..ApiControllers::default()
        },
    );

    let rejected_start = router
        .clone()
        .oneshot(request(
            Method::POST,
            "/v1/translation/start",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_problem(
        rejected_start,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "provider_unavailable",
    )
    .await;
    assert!(!store.snapshot().translation_running);

    store.set_translation_running(true);
    let rejected_stop = router
        .oneshot(request(
            Method::POST,
            "/v1/translation/stop",
            Some(TOKEN),
            Body::from("{}"),
        ))
        .await
        .unwrap();
    assert_problem(
        rejected_stop,
        StatusCode::SERVICE_UNAVAILABLE,
        "Service Unavailable",
        "pipeline_stop_failed",
    )
    .await;
    assert!(store.snapshot().translation_running);
}

#[tokio::test]
async fn authenticated_unknown_path_and_wrong_method_are_problem_details() {
    let missing = app()
        .oneshot(request(
            Method::GET,
            "/v1/not-a-route",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(missing, StatusCode::NOT_FOUND, "Not Found", "not_found").await;

    let wrong_method = app()
        .oneshot(request(
            Method::GET,
            "/v1/self-test/round-trip/start",
            Some(TOKEN),
            Body::empty(),
        ))
        .await
        .unwrap();
    assert_problem(
        wrong_method,
        StatusCode::METHOD_NOT_ALLOWED,
        "Method Not Allowed",
        "method_not_allowed",
    )
    .await;
}

#[test]
fn listener_accepts_only_ipv4_and_ipv6_loopback_addresses() {
    assert!(
        validate_listen_address(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 47_681)).is_ok()
    );
    assert!(
        validate_listen_address(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 47_681))
            .is_err()
    );
    assert!(
        validate_listen_address(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 50)),
            47_681
        ))
        .is_err()
    );
    assert!(
        validate_listen_address(SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 47_681)).is_ok()
    );
    assert!(
        validate_listen_address(SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 47_681))
            .is_err()
    );
    assert!(
        validate_listen_address(SocketAddr::new("2001:db8::1".parse().unwrap(), 47_681)).is_err()
    );
}
