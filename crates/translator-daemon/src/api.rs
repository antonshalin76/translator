use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, patch, post};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast};
use tower::ServiceBuilder;
use tower_http::limit::RequestBodyLimitLayer;
use translator_audio::{GraphHealth, RoutingSafeError, RoutingState};

use crate::{
    AudioMixPatch, AudioMixState, ControlToken, DirectionPatch, LatencyPolicyPatch, ProviderPatch,
    RuntimeEvent, RuntimeMutationError, RuntimeSnapshot, RuntimeStore, VoiceProfilePatch,
};

const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_SSE_SUBSCRIBERS: usize = 4;

#[derive(Debug, Clone)]
pub struct ApiLimits {
    max_body_bytes: usize,
    max_sse_subscribers: usize,
    sse_keepalive: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListenAddressError;

impl Default for ApiLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_sse_subscribers: DEFAULT_MAX_SSE_SUBSCRIBERS,
            sse_keepalive: Duration::from_secs(15),
        }
    }
}

impl ApiLimits {
    pub fn with_sse_keepalive(mut self, interval: Duration) -> Self {
        self.sse_keepalive = interval;
        self
    }
}

#[derive(Clone)]
struct ApiState {
    store: RuntimeStore,
    sse_permits: Arc<Semaphore>,
    keepalive: Duration,
    manual_routes: Option<Arc<dyn ManualRouteController>>,
    audio_mix: Option<Arc<dyn AudioMixController>>,
    translation: Option<Arc<dyn TranslationController>>,
    round_trip: Option<Arc<dyn RoundTripController>>,
}

#[derive(Debug, Clone, Serialize)]
struct ProblemDetails {
    r#type: String,
    title: &'static str,
    status: u16,
    code: &'static str,
}

impl ProblemDetails {
    fn new(status: StatusCode, code: &'static str) -> Self {
        Self {
            r#type: format!("urn:translator:error:{code}"),
            title: status.canonical_reason().unwrap_or("Request Failed"),
            status: status.as_u16(),
            code,
        }
    }
}

impl IntoResponse for ProblemDetails {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            [(header::CONTENT_TYPE, "application/problem+json")],
            Json(self),
        )
            .into_response()
    }
}

#[derive(Debug, Deserialize)]
struct EnabledPatch {
    enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManualRoutePatch {
    stream_id: u32,
}

pub trait ManualRouteController: Send + Sync {
    fn reconcile(&self, stream_id: u32) -> Result<RoutingState, RoutingSafeError>;

    fn refresh_audio_state(&self, _store: &RuntimeStore) {}

    fn restore(&self) -> Result<(), RoutingSafeError> {
        Ok(())
    }
}

pub trait AudioMixController: Send + Sync {
    fn apply(&self, volumes: AudioMixState) -> Result<(), ControlFailure>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlFailure {
    pub status: StatusCode,
    pub code: &'static str,
}

pub trait TranslationController: Send + Sync {
    fn start(&self, snapshot: RuntimeSnapshot) -> Result<(), ControlFailure>;
    fn stop(&self) -> Result<(), ControlFailure>;

    fn reconfigure(&self, snapshot: RuntimeSnapshot) -> Result<(), ControlFailure> {
        self.stop()?;
        self.start(snapshot)
    }
}

pub trait RoundTripController: Send + Sync {
    fn start(&self) -> Result<crate::RoundTripSelfTestState, ControlFailure>;
    fn stop(&self) -> Result<crate::RoundTripSelfTestState, ControlFailure>;
}

#[derive(Clone, Default)]
pub struct ApiControllers {
    pub manual_routes: Option<Arc<dyn ManualRouteController>>,
    pub audio_mix: Option<Arc<dyn AudioMixController>>,
    pub translation: Option<Arc<dyn TranslationController>>,
    pub round_trip: Option<Arc<dyn RoundTripController>>,
}

pub fn build_router(store: RuntimeStore, token: ControlToken, limits: ApiLimits) -> Router {
    build_router_with_controllers(store, token, limits, ApiControllers::default())
}

pub fn build_router_with_manual_routes(
    store: RuntimeStore,
    token: ControlToken,
    limits: ApiLimits,
    manual_routes: Option<Arc<dyn ManualRouteController>>,
) -> Router {
    build_router_with_controllers(
        store,
        token,
        limits,
        ApiControllers {
            manual_routes,
            ..ApiControllers::default()
        },
    )
}

pub fn build_router_with_controllers(
    store: RuntimeStore,
    token: ControlToken,
    limits: ApiLimits,
    controllers: ApiControllers,
) -> Router {
    let state = ApiState {
        store,
        sse_permits: Arc::new(Semaphore::new(limits.max_sse_subscribers)),
        keepalive: limits.sse_keepalive,
        manual_routes: controllers.manual_routes,
        audio_mix: controllers.audio_mix,
        translation: controllers.translation,
        round_trip: controllers.round_trip,
    };
    let middleware = ServiceBuilder::new()
        .layer(middleware::from_fn_with_state(token, authenticate))
        .layer(middleware::from_fn(normalize_error_response))
        .layer(RequestBodyLimitLayer::new(limits.max_body_bytes));

    Router::new()
        .route("/v1/status", get(status))
        .route("/v1/audio-graph", get(audio_graph))
        .route("/v1/routes", get(routes))
        .route("/v1/routes/candidates", get(route_candidates))
        .route("/v1/translation/start", post(start_translation))
        .route("/v1/translation/stop", post(stop_translation))
        .route("/v1/directions", patch(patch_direction))
        .route("/v1/provider", patch(patch_provider))
        .route("/v1/audio-mix", patch(patch_audio_mix))
        .route("/v1/latency-policy", patch(patch_latency_policy))
        .route("/v1/voice-profiles", patch(patch_voice_profile))
        .route("/v1/debug-capture", patch(patch_debug_capture))
        .route("/v1/debug-text", patch(patch_debug_text))
        .route("/v1/routes/manual-override", post(manual_route_override))
        .route(
            "/v1/self-test/round-trip/start",
            post(start_round_trip_self_test),
        )
        .route(
            "/v1/self-test/round-trip/stop",
            post(stop_round_trip_self_test),
        )
        .route("/v1/self-test/round-trip", get(round_trip_self_test_status))
        .route("/v1/events/stream", get(events))
        .fallback(not_found)
        .with_state(state)
        .layer(middleware)
}

pub fn validate_listen_address(address: SocketAddr) -> Result<(), ListenAddressError> {
    if address.ip().is_loopback() {
        Ok(())
    } else {
        Err(ListenAddressError)
    }
}

async fn authenticate(
    State(token): State<ControlToken>,
    request: Request,
    next: Next,
) -> axum::response::Response {
    let authorized = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|candidate| token.matches(candidate));
    if !authorized {
        return ProblemDetails::new(StatusCode::UNAUTHORIZED, "invalid_bearer").into_response();
    }
    next.run(request).await
}

async fn normalize_error_response(request: Request, next: Next) -> axum::response::Response {
    let response = next.run(request).await;
    if response.status().is_success()
        || response
            .headers()
            .get(header::CONTENT_TYPE)
            .is_some_and(|value| value == "application/problem+json")
    {
        return response;
    }
    let (status, code) = match response.status() {
        StatusCode::PAYLOAD_TOO_LARGE => (StatusCode::PAYLOAD_TOO_LARGE, "body_too_large"),
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
            (StatusCode::BAD_REQUEST, "invalid_json")
        }
        StatusCode::NOT_FOUND => (StatusCode::NOT_FOUND, "not_found"),
        StatusCode::METHOD_NOT_ALLOWED => (StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed"),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    };
    ProblemDetails::new(status, code).into_response()
}

async fn status(State(state): State<ApiState>) -> Json<RuntimeSnapshot> {
    Json(state.store.snapshot())
}

async fn audio_graph(State(state): State<ApiState>) -> Json<Value> {
    let graph = state.store.audio_graph();
    let available = graph
        .as_ref()
        .is_some_and(|state| state.health == GraphHealth::Ready);
    Json(json!({
        "available": available,
        "value": graph,
    }))
}

async fn routes(State(state): State<ApiState>) -> Json<Value> {
    let routes = state.store.routes();
    Json(json!({
        "available": routes.is_some(),
        "value": routes,
    }))
}

async fn route_candidates(State(state): State<ApiState>) -> Json<Value> {
    Json(json!(state.store.route_candidates()))
}

async fn start_translation(State(state): State<ApiState>) -> axum::response::Response {
    let Some(controller) = state.translation else {
        return ProblemDetails::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "translation_controller_unavailable",
        )
        .into_response();
    };
    let snapshot = match state.manual_routes.clone() {
        Some(manual_routes) => {
            let store = state.store.clone();
            match tokio::task::spawn_blocking(move || {
                manual_routes.refresh_audio_state(&store);
                store.snapshot()
            })
            .await
            {
                Ok(snapshot) => snapshot,
                Err(_) => {
                    return ProblemDetails::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "audio_refresh_failed",
                    )
                    .into_response();
                }
            }
        }
        None => state.store.snapshot(),
    };
    match tokio::task::spawn_blocking(move || controller.start(snapshot)).await {
        Ok(Ok(())) => {
            state.store.set_translation_running(true);
            Json(state.store.snapshot()).into_response()
        }
        Ok(Err(error)) => ProblemDetails::new(error.status, error.code).into_response(),
        Err(_) => ProblemDetails::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "translation_controller_failed",
        )
        .into_response(),
    }
}

async fn stop_translation(State(state): State<ApiState>) -> axum::response::Response {
    let Some(controller) = state.translation else {
        return ProblemDetails::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "translation_controller_unavailable",
        )
        .into_response();
    };
    match tokio::task::spawn_blocking(move || controller.stop()).await {
        Ok(Ok(())) => {
            state.store.set_translation_running(false);
            if let Some(manual_routes) = state.manual_routes.clone() {
                let store = state.store.clone();
                if tokio::task::spawn_blocking(move || manual_routes.refresh_audio_state(&store))
                    .await
                    .is_err()
                {
                    return ProblemDetails::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "audio_refresh_failed",
                    )
                    .into_response();
                }
            }
            Json(state.store.snapshot()).into_response()
        }
        Ok(Err(error)) => ProblemDetails::new(error.status, error.code).into_response(),
        Err(_) => ProblemDetails::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "translation_controller_failed",
        )
        .into_response(),
    }
}

async fn patch_debug_text(
    State(state): State<ApiState>,
    Json(patch): Json<EnabledPatch>,
) -> Json<RuntimeSnapshot> {
    state.store.set_debug_text_enabled(patch.enabled);
    Json(state.store.snapshot())
}

async fn patch_debug_capture(
    State(state): State<ApiState>,
    Json(patch): Json<EnabledPatch>,
) -> axum::response::Response {
    match state.store.set_debug_capture_enabled(patch.enabled) {
        Ok(()) => Json(state.store.snapshot()).into_response(),
        Err(RuntimeMutationError::DebugCaptureUnavailable) => {
            ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_unavailable")
                .into_response()
        }
        Err(RuntimeMutationError::DebugCaptureStopped(_)) => {
            ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_stopped")
                .into_response()
        }
        Err(RuntimeMutationError::InvalidLanguagePair) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_language_pair").into_response()
        }
        Err(RuntimeMutationError::VoiceLanguageMismatch) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "voice_language_mismatch").into_response()
        }
        Err(RuntimeMutationError::CloudProviderOptInRequired) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "cloud_provider_opt_in_required")
                .into_response()
        }
        Err(RuntimeMutationError::InvalidAudioMixVolume) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_audio_mix_volume").into_response()
        }
    }
}

async fn patch_direction(
    State(state): State<ApiState>,
    Json(patch): Json<DirectionPatch>,
) -> axum::response::Response {
    let was_running = state.store.snapshot().translation_running;
    match state.store.set_direction(patch) {
        Ok(()) => {
            if was_running {
                let Some(controller) = state.translation else {
                    return ProblemDetails::new(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "translation_controller_unavailable",
                    )
                    .into_response();
                };
                let snapshot = state.store.snapshot();
                match tokio::task::spawn_blocking(move || controller.reconfigure(snapshot)).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        state.store.set_translation_running(false);
                        return ProblemDetails::new(error.status, error.code).into_response();
                    }
                    Err(_) => {
                        state.store.set_translation_running(false);
                        return ProblemDetails::new(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "translation_controller_failed",
                        )
                        .into_response();
                    }
                }
            }
            Json(state.store.snapshot()).into_response()
        }
        Err(RuntimeMutationError::InvalidLanguagePair) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_language_pair").into_response()
        }
        Err(RuntimeMutationError::VoiceLanguageMismatch) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "voice_language_mismatch").into_response()
        }
        Err(RuntimeMutationError::DebugCaptureUnavailable) => {
            ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_unavailable")
                .into_response()
        }
        Err(RuntimeMutationError::DebugCaptureStopped(_)) => {
            ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_stopped")
                .into_response()
        }
        Err(RuntimeMutationError::CloudProviderOptInRequired) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "cloud_provider_opt_in_required")
                .into_response()
        }
        Err(RuntimeMutationError::InvalidAudioMixVolume) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_audio_mix_volume").into_response()
        }
    }
}

async fn patch_provider(
    State(state): State<ApiState>,
    Json(patch): Json<ProviderPatch>,
) -> axum::response::Response {
    match state.store.set_provider(patch) {
        Ok(()) => Json(state.store.snapshot()).into_response(),
        Err(RuntimeMutationError::CloudProviderOptInRequired) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "cloud_provider_opt_in_required")
                .into_response()
        }
        Err(RuntimeMutationError::InvalidLanguagePair) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_language_pair").into_response()
        }
        Err(RuntimeMutationError::VoiceLanguageMismatch) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "voice_language_mismatch").into_response()
        }
        Err(RuntimeMutationError::DebugCaptureUnavailable) => {
            ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_unavailable")
                .into_response()
        }
        Err(RuntimeMutationError::DebugCaptureStopped(_)) => {
            ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_stopped")
                .into_response()
        }
        Err(RuntimeMutationError::InvalidAudioMixVolume) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_audio_mix_volume").into_response()
        }
    }
}

async fn patch_audio_mix(
    State(state): State<ApiState>,
    Json(patch): Json<AudioMixPatch>,
) -> axum::response::Response {
    let volumes = match state.store.set_audio_mix(patch) {
        Ok(volumes) => volumes,
        Err(RuntimeMutationError::InvalidAudioMixVolume) => {
            return ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_audio_mix_volume")
                .into_response();
        }
        Err(RuntimeMutationError::InvalidLanguagePair) => {
            return ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_language_pair")
                .into_response();
        }
        Err(RuntimeMutationError::VoiceLanguageMismatch) => {
            return ProblemDetails::new(StatusCode::BAD_REQUEST, "voice_language_mismatch")
                .into_response();
        }
        Err(RuntimeMutationError::CloudProviderOptInRequired) => {
            return ProblemDetails::new(StatusCode::BAD_REQUEST, "cloud_provider_opt_in_required")
                .into_response();
        }
        Err(RuntimeMutationError::DebugCaptureUnavailable) => {
            return ProblemDetails::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "debug_capture_unavailable",
            )
            .into_response();
        }
        Err(RuntimeMutationError::DebugCaptureStopped(_)) => {
            return ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_stopped")
                .into_response();
        }
    };
    if let Some(controller) = state.audio_mix {
        match tokio::task::spawn_blocking(move || controller.apply(volumes)).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return ProblemDetails::new(error.status, error.code).into_response();
            }
            Err(_) => {
                return ProblemDetails::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "audio_mix_controller_failed",
                )
                .into_response();
            }
        }
    }
    Json(state.store.snapshot()).into_response()
}

async fn patch_latency_policy(
    State(state): State<ApiState>,
    Json(patch): Json<LatencyPolicyPatch>,
) -> Json<RuntimeSnapshot> {
    state.store.set_latency_policy(patch);
    Json(state.store.snapshot())
}

async fn patch_voice_profile(
    State(state): State<ApiState>,
    Json(patch): Json<VoiceProfilePatch>,
) -> axum::response::Response {
    let was_running = state.store.snapshot().translation_running;
    match state.store.set_voice_profile(patch) {
        Ok(()) => {
            if was_running {
                let Some(controller) = state.translation else {
                    return ProblemDetails::new(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "translation_controller_unavailable",
                    )
                    .into_response();
                };
                let snapshot = state.store.snapshot();
                match tokio::task::spawn_blocking(move || controller.reconfigure(snapshot)).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        state.store.set_translation_running(false);
                        return ProblemDetails::new(error.status, error.code).into_response();
                    }
                    Err(_) => {
                        state.store.set_translation_running(false);
                        return ProblemDetails::new(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "translation_controller_failed",
                        )
                        .into_response();
                    }
                }
            }
            Json(state.store.snapshot()).into_response()
        }
        Err(RuntimeMutationError::VoiceLanguageMismatch) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "voice_language_mismatch").into_response()
        }
        Err(RuntimeMutationError::InvalidLanguagePair) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_language_pair").into_response()
        }
        Err(RuntimeMutationError::DebugCaptureUnavailable) => {
            ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_unavailable")
                .into_response()
        }
        Err(RuntimeMutationError::DebugCaptureStopped(_)) => {
            ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "debug_capture_stopped")
                .into_response()
        }
        Err(RuntimeMutationError::CloudProviderOptInRequired) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "cloud_provider_opt_in_required")
                .into_response()
        }
        Err(RuntimeMutationError::InvalidAudioMixVolume) => {
            ProblemDetails::new(StatusCode::BAD_REQUEST, "invalid_audio_mix_volume").into_response()
        }
    }
}

async fn manual_route_override(
    State(state): State<ApiState>,
    Json(patch): Json<ManualRoutePatch>,
) -> axum::response::Response {
    let Some(controller) = state.manual_routes else {
        return ProblemDetails::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "routing_controller_unavailable",
        )
        .into_response();
    };
    let result = tokio::task::spawn_blocking(move || controller.reconcile(patch.stream_id)).await;
    match result {
        Ok(Ok(routes)) => {
            state.store.set_routes(routes);
            Json(state.store.snapshot()).into_response()
        }
        Ok(Err(error)) => {
            tracing::warn!(event = "manual_route_failed", code = ?error.code);
            ProblemDetails::new(StatusCode::CONFLICT, "manual_route_failed").into_response()
        }
        Err(_) => ProblemDetails::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "routing_controller_failed",
        )
        .into_response(),
    }
}

async fn start_round_trip_self_test(State(state): State<ApiState>) -> axum::response::Response {
    let Some(controller) = state.round_trip else {
        return ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "self_test_unavailable")
            .into_response();
    };
    match tokio::task::spawn_blocking(move || controller.start()).await {
        Ok(Ok(self_test)) => {
            state.store.set_self_test(self_test);
            Json(state.store.snapshot().self_test).into_response()
        }
        Ok(Err(error)) => ProblemDetails::new(error.status, error.code).into_response(),
        Err(_) => ProblemDetails::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "self_test_controller_failed",
        )
        .into_response(),
    }
}

async fn stop_round_trip_self_test(State(state): State<ApiState>) -> axum::response::Response {
    let Some(controller) = state.round_trip else {
        return ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "self_test_unavailable")
            .into_response();
    };
    match tokio::task::spawn_blocking(move || controller.stop()).await {
        Ok(Ok(mut self_test)) => {
            self_test.status.debug_text = None;
            state.store.set_self_test(self_test);
            Json(state.store.snapshot().self_test).into_response()
        }
        Ok(Err(error)) => ProblemDetails::new(error.status, error.code).into_response(),
        Err(_) => ProblemDetails::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "self_test_controller_failed",
        )
        .into_response(),
    }
}

async fn round_trip_self_test_status(State(state): State<ApiState>) -> Json<Value> {
    Json(json!(state.store.snapshot().self_test))
}

async fn not_found() -> ProblemDetails {
    ProblemDetails::new(StatusCode::NOT_FOUND, "not_found")
}

struct EventStreamState {
    receiver: broadcast::Receiver<RuntimeEvent>,
    store: RuntimeStore,
    pending_snapshot: bool,
    initial_snapshot: bool,
    _permit: OwnedSemaphorePermit,
}

async fn events(State(state): State<ApiState>) -> axum::response::Response {
    let permit = match Arc::clone(&state.sse_permits).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return ProblemDetails::new(StatusCode::TOO_MANY_REQUESTS, "sse_subscriber_limit")
                .into_response();
        }
    };
    let Some(receiver) = state.store.subscribe() else {
        return ProblemDetails::new(StatusCode::SERVICE_UNAVAILABLE, "event_stream_unavailable")
            .into_response();
    };
    let event_state = EventStreamState {
        receiver,
        store: state.store,
        pending_snapshot: false,
        initial_snapshot: true,
        _permit: permit,
    };
    let stream = stream::unfold(event_state, |mut state| async move {
        if state.initial_snapshot || state.pending_snapshot {
            state.initial_snapshot = false;
            state.pending_snapshot = false;
            return Some((Ok::<_, Infallible>(snapshot_event(&state.store)), state));
        }
        match state.receiver.recv().await {
            Ok(RuntimeEvent::SnapshotChanged) => Some((Ok(snapshot_event(&state.store)), state)),
            Err(broadcast::error::RecvError::Lagged(_)) => {
                state.pending_snapshot = true;
                Some((
                    Ok(Event::default()
                        .event("resync_required")
                        .data(r#"{"reason":"lagged"}"#)),
                    state,
                ))
            }
            Err(broadcast::error::RecvError::Closed) => None,
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(state.keepalive).text(""))
        .into_response()
}

fn snapshot_event(store: &RuntimeStore) -> Event {
    let data = serde_json::to_string(&store.snapshot()).unwrap_or_else(|_| "{}".to_owned());
    Event::default().event("snapshot").data(data)
}
