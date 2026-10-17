use std::env;
use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, Runtime, WindowEvent};

const DEFAULT_DAEMON_BASE_URL: &str = "http://127.0.0.1:47681";
const RUNTIME_DIRECTORY: &str = "translator";
const TOKEN_FILE: &str = "control.token";
const FAST_REQUEST_TIMEOUT: Duration = Duration::from_millis(1500);
const TRANSLATION_START_TIMEOUT: Duration = Duration::from_secs(135);
const TRANSLATION_STOP_TIMEOUT: Duration = Duration::from_secs(15);
const TRAY_ID: &str = "translator-control";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpMethod {
    Get,
    Post,
    Patch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DaemonCommand {
    Status,
    StartTranslation,
    StopTranslation,
    ConfirmHeadphones,
    PatchDebugText,
    PatchDebugCapture,
    PatchDirection,
    PatchProvider,
    PatchAudioMix,
    PatchLatencyPolicy,
    PatchVoiceProfile,
    ManualRouteOverride,
    RoundTripStatus,
    StartRoundTrip,
    StopRoundTrip,
    AecCalibrationStatus,
    StartAecCalibration,
    CancelAecCalibration,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
enum AecProofStatus {
    Unavailable,
    Measuring,
    ValidationFailed,
    Validated {
        source_name: String,
        sink_name: String,
        expires_monotonic_ns: u64,
    },
    CleanupUncertain,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
enum AecCalibrationStatus {
    Unavailable,
    Running {
        attempt_id: String,
    },
    Succeeded {
        attempt_id: String,
        proof: AecProofStatus,
    },
    Failed {
        attempt_id: String,
        code: String,
    },
    Cancelled {
        attempt_id: String,
    },
    TimedOut {
        attempt_id: String,
    },
    CleanupUncertain {
        attempt_id: String,
    },
    ShuttingDown,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
struct UiError {
    code: &'static str,
    message: String,
    http_status: Option<u16>,
}

impl UiError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            http_status: None,
        }
    }

    fn with_status(code: &'static str, message: impl Into<String>, status: u16) -> Self {
        Self {
            code,
            message: message.into(),
            http_status: Some(status),
        }
    }
}

fn main() {
    let context = tauri::generate_context!();
    if env::args().any(|argument| argument == "--check-bundled-ui") {
        if tauri::is_dev()
            || context
                .assets()
                .get(&"index.html".into())
                .is_none_or(|asset| asset.is_empty())
        {
            eprintln!("bundled UI assets are unavailable");
            std::process::exit(78);
        }
        println!("bundled-ui-ok");
        return;
    }
    tauri::Builder::default()
        .setup(setup_tray)
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::CloseRequested { .. }) && window.label() == "main" {
                let _ = daemon_request(
                    DaemonCommand::PatchDebugText,
                    Some(json!({ "enabled": false })),
                );
            }
        })
        .invoke_handler(tauri::generate_handler![
            translator_status,
            translator_start,
            translator_stop,
            translator_set_debug_text,
            translator_set_debug_capture,
            translator_set_direction,
            translator_set_provider,
            translator_set_audio_mix,
            translator_confirm_headphones,
            translator_set_latency_mode,
            translator_set_voice_profile,
            translator_select_route,
            translator_round_trip_status,
            translator_start_round_trip,
            translator_stop_round_trip,
            translator_aec_calibration_status,
            translator_start_aec_calibration,
            translator_cancel_aec_calibration,
        ])
        .run(context)
        .expect("failed to run translator UI");
}

#[tauri::command]
fn translator_status(app: AppHandle) -> Result<Value, UiError> {
    let snapshot = daemon_request(DaemonCommand::Status, None);
    let _ = set_tray_menu_from_snapshot(&app, snapshot.as_ref().ok());
    snapshot
}

#[tauri::command]
fn translator_start(app: AppHandle) -> Result<Value, UiError> {
    daemon_request_from_webview(&app, DaemonCommand::StartTranslation, None)
}

#[tauri::command]
fn translator_stop(app: AppHandle) -> Result<Value, UiError> {
    daemon_request_from_webview(&app, DaemonCommand::StopTranslation, None)
}

#[tauri::command]
fn translator_confirm_headphones(
    app: AppHandle,
    confirmation: Option<Value>,
) -> Result<Value, UiError> {
    daemon_request_from_webview(
        &app,
        DaemonCommand::ConfirmHeadphones,
        Some(confirmation.unwrap_or(Value::Null)),
    )
}

#[tauri::command]
fn translator_set_debug_text(app: AppHandle, enabled: bool) -> Result<Value, UiError> {
    daemon_request_from_webview(
        &app,
        DaemonCommand::PatchDebugText,
        Some(json!({ "enabled": enabled })),
    )
}

#[tauri::command]
fn translator_set_debug_capture(app: AppHandle, enabled: bool) -> Result<Value, UiError> {
    daemon_request_from_webview(
        &app,
        DaemonCommand::PatchDebugCapture,
        Some(json!({ "enabled": enabled })),
    )
}

#[tauri::command]
fn translator_set_direction(
    app: AppHandle,
    direction_id: String,
    source_language: Option<String>,
    target_language: Option<String>,
    enabled: Option<bool>,
) -> Result<Value, UiError> {
    daemon_request_from_webview(
        &app,
        DaemonCommand::PatchDirection,
        Some(json!({
            "direction_id": direction_id,
            "source_language": source_language,
            "target_language": target_language,
            "enabled": enabled,
        })),
    )
}

#[tauri::command]
fn translator_set_provider(
    app: AppHandle,
    provider_id: String,
    cloud_opt_in: bool,
) -> Result<Value, UiError> {
    if provider_requires_cloud_opt_in(&provider_id) && !cloud_opt_in {
        return Err(UiError::new(
            "cloud_provider_opt_in_required",
            "cloud provider requires explicit opt-in",
        ));
    }
    if !matches!(provider_id.as_str(), "local" | "openai") {
        return Err(UiError::new(
            "invalid_provider",
            "provider is not supported",
        ));
    }
    daemon_request_from_webview(
        &app,
        DaemonCommand::PatchProvider,
        Some(json!({ "provider_id": provider_id, "cloud_opt_in": cloud_opt_in })),
    )
}

#[tauri::command]
fn translator_set_audio_mix(
    app: AppHandle,
    microphone_original_percent: Option<u8>,
    microphone_translation_percent: Option<u8>,
    speaker_original_percent: Option<u8>,
    speaker_translation_percent: Option<u8>,
) -> Result<Value, UiError> {
    for value in [
        microphone_original_percent,
        microphone_translation_percent,
        speaker_original_percent,
        speaker_translation_percent,
    ]
    .into_iter()
    .flatten()
    {
        if value > 100 {
            return Err(UiError::new(
                "invalid_audio_mix_volume",
                "audio mix volume must be 0..100 percent",
            ));
        }
    }
    daemon_request_from_webview(
        &app,
        DaemonCommand::PatchAudioMix,
        Some(json!({
            "microphone_original_percent": microphone_original_percent,
            "microphone_translation_percent": microphone_translation_percent,
            "speaker_original_percent": speaker_original_percent,
            "speaker_translation_percent": speaker_translation_percent,
        })),
    )
}

#[tauri::command]
fn translator_set_latency_mode(
    app: AppHandle,
    direction_id: String,
    current_mode: String,
) -> Result<Value, UiError> {
    daemon_request_from_webview(
        &app,
        DaemonCommand::PatchLatencyPolicy,
        Some(json!({
            "direction_id": direction_id,
            "current_mode": current_mode,
        })),
    )
}

#[tauri::command]
fn translator_set_voice_profile(
    app: AppHandle,
    direction_id: String,
    language: String,
    gender: String,
    engine: String,
) -> Result<Value, UiError> {
    daemon_request_from_webview(
        &app,
        DaemonCommand::PatchVoiceProfile,
        Some(json!({
            "direction_id": direction_id,
            "voice_profile": {
                "language": language,
                "gender": gender,
                "engine": engine,
            }
        })),
    )
}

#[tauri::command]
fn translator_select_route(app: AppHandle, stream_id: u32) -> Result<Value, UiError> {
    daemon_request_from_webview(
        &app,
        DaemonCommand::ManualRouteOverride,
        Some(json!({ "stream_id": stream_id })),
    )
}

#[tauri::command]
fn translator_round_trip_status() -> Result<Value, UiError> {
    daemon_request(DaemonCommand::RoundTripStatus, None)
}

#[tauri::command]
fn translator_start_round_trip(app: AppHandle) -> Result<Value, UiError> {
    daemon_request(DaemonCommand::StartRoundTrip, None)?;
    refresh_status_from_webview(&app, DaemonCommand::StartRoundTrip)
}

#[tauri::command]
fn translator_stop_round_trip(app: AppHandle) -> Result<Value, UiError> {
    daemon_request(DaemonCommand::StopRoundTrip, None)?;
    refresh_status_from_webview(&app, DaemonCommand::StopRoundTrip)
}

#[tauri::command]
fn translator_aec_calibration_status() -> Result<AecCalibrationStatus, UiError> {
    parse_aec_status(daemon_request(DaemonCommand::AecCalibrationStatus, None)?)
}

#[tauri::command]
fn translator_start_aec_calibration() -> Result<AecCalibrationStatus, UiError> {
    parse_aec_status(daemon_request(DaemonCommand::StartAecCalibration, None)?)
}

#[tauri::command]
fn translator_cancel_aec_calibration(attempt_id: String) -> Result<AecCalibrationStatus, UiError> {
    parse_aec_status(daemon_request(
        DaemonCommand::CancelAecCalibration,
        Some(json!({ "attempt_id": attempt_id })),
    )?)
}

fn parse_aec_status(value: Value) -> Result<AecCalibrationStatus, UiError> {
    serde_json::from_value(value).map_err(|_| {
        UiError::new(
            "aec_calibration_response_invalid",
            "calibration response is invalid",
        )
    })
}

fn tray_translation_controls(snapshot: Option<&Value>) -> (bool, bool) {
    let Some(snapshot) = snapshot else {
        return (false, false);
    };
    let running = snapshot.get("translation_running").and_then(Value::as_bool) == Some(true);
    let status = snapshot.get("runtime_status").and_then(Value::as_str);
    (
        !running && status == Some("stopped"),
        running || matches!(status, Some("running" | "failed" | "cleanup_pending")),
    )
}

fn setup_tray(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = daemon_request(DaemonCommand::Status, None).ok();
    let menu = tray_menu(app, snapshot.as_ref())?;
    app.manage(menu.clone());

    let mut tray = TrayIconBuilder::with_id(TRAY_ID)
        .menu(&menu)
        .tooltip("Translator")
        .show_menu_on_left_click(true)
        .on_menu_event(|app, event| handle_tray_menu_event(app, event.id().as_ref()))
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                let _ = refresh_tray_menu(app);
                show_main_window(app);
            }
        });

    if let Some(icon) = app.default_window_icon().cloned() {
        tray = tray.icon(icon);
    }

    tray.build(app)?;
    Ok(())
}

fn tray_menu<M, R>(manager: &M, snapshot: Option<&Value>) -> tauri::Result<Menu<R>>
where
    M: Manager<R>,
    R: Runtime,
{
    let (can_start, can_stop) = tray_translation_controls(snapshot);
    let show = MenuItem::with_id(manager, "show", "Показать Translator", true, None::<&str>)?;
    let start = MenuItem::with_id(manager, "start", "Начать перевод", can_start, None::<&str>)?;
    let stop = MenuItem::with_id(
        manager,
        "stop",
        "Остановить перевод",
        can_stop,
        None::<&str>,
    )?;
    let separator = PredefinedMenuItem::separator(manager)?;
    let quit = MenuItem::with_id(manager, "quit", "Закрыть интерфейс", true, None::<&str>)?;
    Menu::with_items(manager, &[&show, &start, &stop, &separator, &quit])
}

fn handle_tray_menu_event<R: Runtime>(app: &AppHandle<R>, id: &str) {
    let result = match id {
        "show" => {
            show_main_window(app);
            return;
        }
        "start" => daemon_request(DaemonCommand::StartTranslation, None),
        "stop" => daemon_request(DaemonCommand::StopTranslation, None),
        "quit" => {
            let _ = daemon_request(
                DaemonCommand::PatchDebugText,
                Some(json!({ "enabled": false })),
            );
            app.exit(0);
            return;
        }
        _ => return,
    };
    if let Err(error) = result {
        show_main_window(app);
        let _ = app.emit("translator-control-error", error);
    }
    let _ = refresh_tray_menu(app);
}

fn daemon_request_from_webview<R: Runtime>(
    app: &AppHandle<R>,
    command: DaemonCommand,
    body: Option<Value>,
) -> Result<Value, UiError> {
    let snapshot = daemon_request(command, body)?;
    if webview_command_refreshes_tray(command) {
        let _ = set_tray_menu_from_snapshot(app, Some(&snapshot));
    }
    Ok(snapshot)
}

fn refresh_status_from_webview<R: Runtime>(
    app: &AppHandle<R>,
    completed_command: DaemonCommand,
) -> Result<Value, UiError> {
    let snapshot = daemon_request(DaemonCommand::Status, None)?;
    if webview_command_refreshes_tray(completed_command) {
        let _ = set_tray_menu_from_snapshot(app, Some(&snapshot));
    }
    Ok(snapshot)
}

fn webview_command_refreshes_tray(command: DaemonCommand) -> bool {
    matches!(
        command,
        DaemonCommand::StartTranslation | DaemonCommand::StopTranslation
    )
}

fn refresh_tray_menu<R: Runtime>(app: &AppHandle<R>) -> Result<(), UiError> {
    let snapshot = daemon_request(DaemonCommand::Status, None).ok();
    set_tray_menu_from_snapshot(app, snapshot.as_ref())
}

fn set_tray_menu_from_snapshot<R: Runtime>(
    app: &AppHandle<R>,
    snapshot: Option<&Value>,
) -> Result<(), UiError> {
    let menu = app
        .try_state::<Menu<R>>()
        .ok_or_else(|| UiError::new("tray_menu_unavailable", "tray menu is unavailable"))?;
    let (can_start, can_stop) = tray_translation_controls(snapshot);
    for (id, enabled) in [("start", can_start), ("stop", can_stop)] {
        let item = menu
            .get(id)
            .ok_or_else(|| UiError::new("tray_menu_unavailable", "tray item is unavailable"))?;
        item.as_menuitem()
            .ok_or_else(|| UiError::new("tray_menu_unavailable", "tray item is invalid"))?
            .set_enabled(enabled)
            .map_err(|_| UiError::new("tray_menu_unavailable", "tray item could not be updated"))?;
    }
    Ok(())
}

fn show_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn daemon_request(command: DaemonCommand, body: Option<Value>) -> Result<Value, UiError> {
    let token = read_control_token()?;
    let base_url = daemon_base_url()?;
    let client = reqwest::blocking::Client::builder()
        .timeout(daemon_request_timeout(command))
        .build()
        .map_err(|_| UiError::new("daemon_client_unavailable", "daemon client is unavailable"))?;
    let (method, path) = daemon_endpoint(command);
    let url = format!("{base_url}{path}");
    let request = match method {
        HttpMethod::Get => client.get(url),
        HttpMethod::Post => client.post(url),
        HttpMethod::Patch => client.patch(url),
    }
    .bearer_auth(token);
    let request = if let Some(body) = body {
        request.json(&body)
    } else {
        request
    };
    let response = request
        .send()
        .map_err(|_| UiError::new("daemon_unavailable", "daemon is unavailable"))?;
    let status = response.status();
    let text = response.text().map_err(|_| {
        UiError::new(
            "daemon_response_unreadable",
            "daemon response is unreadable",
        )
    })?;
    if !status.is_success() {
        let code = problem_code(&text).unwrap_or("daemon_request_failed");
        return Err(UiError::with_status(
            code,
            format!("daemon request failed with {code}"),
            status.as_u16(),
        ));
    }
    serde_json::from_str(&text)
        .map_err(|_| UiError::new("daemon_response_invalid", "daemon response is invalid"))
}

fn daemon_endpoint(command: DaemonCommand) -> (HttpMethod, &'static str) {
    match command {
        DaemonCommand::Status => (HttpMethod::Get, "/v1/status"),
        DaemonCommand::StartTranslation => (HttpMethod::Post, "/v1/translation/start"),
        DaemonCommand::StopTranslation => (HttpMethod::Post, "/v1/translation/stop"),
        DaemonCommand::ConfirmHeadphones => (HttpMethod::Post, "/v1/devices/headphones"),
        DaemonCommand::PatchDebugText => (HttpMethod::Patch, "/v1/debug-text"),
        DaemonCommand::PatchDebugCapture => (HttpMethod::Patch, "/v1/debug-capture"),
        DaemonCommand::PatchDirection => (HttpMethod::Patch, "/v1/directions"),
        DaemonCommand::PatchProvider => (HttpMethod::Patch, "/v1/provider"),
        DaemonCommand::PatchAudioMix => (HttpMethod::Patch, "/v1/audio-mix"),
        DaemonCommand::PatchLatencyPolicy => (HttpMethod::Patch, "/v1/latency-policy"),
        DaemonCommand::PatchVoiceProfile => (HttpMethod::Patch, "/v1/voice-profiles"),
        DaemonCommand::ManualRouteOverride => (HttpMethod::Post, "/v1/routes/manual-override"),
        DaemonCommand::RoundTripStatus => (HttpMethod::Get, "/v1/self-test/round-trip"),
        DaemonCommand::StartRoundTrip => (HttpMethod::Post, "/v1/self-test/round-trip/start"),
        DaemonCommand::StopRoundTrip => (HttpMethod::Post, "/v1/self-test/round-trip/stop"),
        DaemonCommand::AecCalibrationStatus => (HttpMethod::Get, "/v1/aec-calibration"),
        DaemonCommand::StartAecCalibration => (HttpMethod::Post, "/v1/aec-calibration/start"),
        DaemonCommand::CancelAecCalibration => (HttpMethod::Post, "/v1/aec-calibration/cancel"),
    }
}

fn daemon_request_timeout(command: DaemonCommand) -> Duration {
    match command {
        DaemonCommand::StartTranslation => TRANSLATION_START_TIMEOUT,
        DaemonCommand::StopTranslation => TRANSLATION_STOP_TIMEOUT,
        _ => FAST_REQUEST_TIMEOUT,
    }
}

fn daemon_base_url() -> Result<String, UiError> {
    let raw =
        env::var("TRANSLATOR_DAEMON_URL").unwrap_or_else(|_| DEFAULT_DAEMON_BASE_URL.to_owned());
    parse_daemon_base_url(&raw)
}

fn parse_daemon_base_url(raw: &str) -> Result<String, UiError> {
    let url = reqwest::Url::parse(raw)
        .map_err(|_| UiError::new("invalid_daemon_url", "daemon URL is invalid"))?;
    let path_is_root = url.path().is_empty() || url.path() == "/";
    if url.scheme() != "http"
        || !url.username().is_empty()
        || url.password().is_some()
        || !path_is_root
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.host_str().is_some_and(is_loopback_host)
    {
        return Err(UiError::new(
            "invalid_daemon_url",
            "daemon URL must be loopback HTTP without credentials, path, query or fragment",
        ));
    }
    Ok(raw.trim_end_matches('/').to_owned())
}

fn control_token_path(runtime_parent: &Path) -> PathBuf {
    runtime_parent.join(RUNTIME_DIRECTORY).join(TOKEN_FILE)
}

fn read_control_token() -> Result<String, UiError> {
    let runtime_dir = env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| UiError::new("runtime_dir_unavailable", "XDG_RUNTIME_DIR is unavailable"))?;
    let path = control_token_path(&runtime_dir);
    let token = fs::read_to_string(path).map_err(|_| {
        UiError::new(
            "control_token_unavailable",
            "daemon control token is unavailable",
        )
    })?;
    let token = token.trim().to_owned();
    if !is_valid_control_token(&token) {
        return Err(UiError::new(
            "control_token_invalid",
            "daemon control token is invalid",
        ));
    }
    Ok(token)
}

fn is_valid_control_token(token: &str) -> bool {
    token.len() == 64
        && token
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_loopback_host(host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host == "localhost"
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn provider_requires_cloud_opt_in(provider_id: &str) -> bool {
    provider_id == "openai"
}

fn problem_code(body: &str) -> Option<&'static str> {
    let value = serde_json::from_str::<Value>(body).ok()?;
    match value.get("code")?.as_str()? {
        "invalid_bearer" => Some("invalid_bearer"),
        "translation_controller_unavailable" => Some("translation_controller_unavailable"),
        "translation_controller_failed" => Some("translation_controller_failed"),
        "translation_precondition_failed" => Some("translation_precondition_failed"),
        "translation_start_failed" => Some("translation_start_failed"),
        "provider_change_requires_stop" => Some("provider_change_requires_stop"),
        "latency_change_requires_stop" => Some("latency_change_requires_stop"),
        "routing_controller_unavailable" => Some("routing_controller_unavailable"),
        "routing_controller_failed" => Some("routing_controller_failed"),
        "manual_route_failed" => Some("manual_route_failed"),
        "audio_mix_apply_failed" => Some("audio_mix_apply_failed"),
        "microphone_original_unavailable" => Some("microphone_original_unavailable"),
        "audio_mix_discovery_failed" => Some("audio_mix_discovery_failed"),
        "audio_mix_state_unknown" => Some("audio_mix_state_unknown"),
        "audio_mix_controller_unavailable" => Some("audio_mix_controller_unavailable"),
        "original_loopback_custody_unknown" => Some("original_loopback_custody_unknown"),
        "aec_original_mix_unavailable" => Some("aec_original_mix_unavailable"),
        "audio_mix_controller_failed" => Some("audio_mix_controller_failed"),
        "invalid_audio_mix_volume" => Some("invalid_audio_mix_volume"),
        "self_test_unavailable" => Some("self_test_unavailable"),
        "aec_calibration_controller_unavailable" => Some("aec_calibration_controller_unavailable"),
        "self_test_controller_failed" => Some("self_test_controller_failed"),
        "debug_capture_unavailable" => Some("debug_capture_unavailable"),
        "debug_capture_stopped" => Some("debug_capture_stopped"),
        "invalid_language_pair" => Some("invalid_language_pair"),
        "voice_language_mismatch" => Some("voice_language_mismatch"),
        "body_too_large" => Some("body_too_large"),
        "invalid_json" => Some("invalid_json"),
        "not_found" => Some("not_found"),
        "method_not_allowed" => Some("method_not_allowed"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_endpoint_maps_aec_calibration_controls() {
        assert_eq!(
            daemon_endpoint(DaemonCommand::AecCalibrationStatus),
            (HttpMethod::Get, "/v1/aec-calibration")
        );
        assert_eq!(
            daemon_endpoint(DaemonCommand::StartAecCalibration),
            (HttpMethod::Post, "/v1/aec-calibration/start")
        );
        assert_eq!(
            daemon_endpoint(DaemonCommand::CancelAecCalibration),
            (HttpMethod::Post, "/v1/aec-calibration/cancel")
        );
        assert!(!webview_command_refreshes_tray(
            DaemonCommand::StartAecCalibration
        ));
    }

    #[test]
    fn aec_status_parser_rejects_missing_attempt_and_proof_fields() {
        assert!(parse_aec_status(json!({"state": "running"})).is_err());
        assert!(
            parse_aec_status(
                json!({"state": "succeeded", "attempt_id": "a", "proof": {"state": "validated"}})
            )
            .is_err()
        );
        assert_eq!(
            parse_aec_status(json!({"state": "cleanup_uncertain", "attempt_id": "a"})).unwrap(),
            AecCalibrationStatus::CleanupUncertain {
                attempt_id: "a".into()
            }
        );
    }

    #[test]
    fn daemon_endpoint_maps_debug_text_and_capture_to_separate_patches() {
        assert_eq!(
            daemon_endpoint(DaemonCommand::PatchDebugText),
            (HttpMethod::Patch, "/v1/debug-text")
        );
        assert_eq!(
            daemon_endpoint(DaemonCommand::PatchDebugCapture),
            (HttpMethod::Patch, "/v1/debug-capture")
        );
    }

    #[test]
    fn daemon_endpoint_maps_audio_mix_patch() {
        assert_eq!(
            daemon_endpoint(DaemonCommand::PatchAudioMix),
            (HttpMethod::Patch, "/v1/audio-mix")
        );
    }

    #[test]
    fn problem_code_keeps_translation_precondition_visible() {
        assert_eq!(
            problem_code(r#"{"code":"translation_precondition_failed"}"#),
            Some("translation_precondition_failed")
        );
    }

    #[test]
    fn problem_code_preserves_known_mix_rejections_without_private_details() {
        for code in [
            "translation_start_failed",
            "provider_change_requires_stop",
            "latency_change_requires_stop",
            "microphone_original_unavailable",
            "audio_mix_discovery_failed",
            "audio_mix_state_unknown",
            "audio_mix_controller_unavailable",
            "original_loopback_custody_unknown",
            "aec_original_mix_unavailable",
            "aec_calibration_controller_unavailable",
        ] {
            let body = json!({"code": code, "detail": "PRIVATE_DETAIL_SENTINEL"}).to_string();
            assert_eq!(problem_code(&body), Some(code));
        }
        for body in [
            r#"{"code":"PRIVATE_UNKNOWN_CODE","detail":"PRIVATE_DETAIL_SENTINEL"}"#,
            r#"{"code":null}"#,
            "PRIVATE_NON_JSON_SENTINEL",
        ] {
            assert_eq!(problem_code(body), None);
        }
    }

    #[test]
    fn tray_controls_project_only_acknowledged_translation_lifecycle() {
        assert_eq!(tray_translation_controls(None), (false, false));
        assert_eq!(tray_translation_controls(Some(&json!({}))), (false, false));
        for (status, running, expected) in [
            ("stopped", false, (true, false)),
            ("running", true, (false, true)),
            ("failed", false, (false, true)),
            ("cleanup_pending", false, (false, true)),
            ("unknown", false, (false, false)),
            ("stopped", true, (false, true)),
        ] {
            let snapshot = json!({"runtime_status": status, "translation_running": running});
            assert_eq!(tray_translation_controls(Some(&snapshot)), expected);
        }
    }

    #[test]
    fn webview_commands_refresh_tray_when_their_state_is_visible_in_the_menu() {
        assert_eq!(
            daemon_endpoint(DaemonCommand::ConfirmHeadphones),
            (HttpMethod::Post, "/v1/devices/headphones"),
        );
        for command in [
            DaemonCommand::StartTranslation,
            DaemonCommand::StopTranslation,
        ] {
            assert!(webview_command_refreshes_tray(command), "{command:?}");
        }

        for command in [
            DaemonCommand::Status,
            DaemonCommand::PatchDebugText,
            DaemonCommand::PatchDebugCapture,
            DaemonCommand::PatchDirection,
            DaemonCommand::PatchProvider,
            DaemonCommand::PatchLatencyPolicy,
            DaemonCommand::StartRoundTrip,
            DaemonCommand::StopRoundTrip,
            DaemonCommand::PatchAudioMix,
            DaemonCommand::PatchVoiceProfile,
            DaemonCommand::ManualRouteOverride,
            DaemonCommand::RoundTripStatus,
        ] {
            assert!(!webview_command_refreshes_tray(command), "{command:?}");
        }
    }

    #[test]
    fn daemon_endpoint_maps_round_trip_controls() {
        assert_eq!(
            daemon_endpoint(DaemonCommand::StartRoundTrip),
            (HttpMethod::Post, "/v1/self-test/round-trip/start")
        );
        assert_eq!(
            daemon_endpoint(DaemonCommand::StopRoundTrip),
            (HttpMethod::Post, "/v1/self-test/round-trip/stop")
        );
        assert_eq!(
            daemon_endpoint(DaemonCommand::RoundTripStatus),
            (HttpMethod::Get, "/v1/self-test/round-trip")
        );
    }

    #[test]
    fn long_running_translation_controls_do_not_use_fast_ui_timeout() {
        assert!(
            daemon_request_timeout(DaemonCommand::StartTranslation) >= Duration::from_secs(130)
        );
        assert!(daemon_request_timeout(DaemonCommand::StopTranslation) >= Duration::from_secs(10));
        assert_eq!(
            daemon_request_timeout(DaemonCommand::Status),
            FAST_REQUEST_TIMEOUT
        );
    }

    #[test]
    fn daemon_base_url_is_loopback_only() {
        assert!(parse_daemon_base_url("http://127.0.0.1:47681").is_ok());
        assert!(parse_daemon_base_url("http://[::1]:47681").is_ok());
        assert!(parse_daemon_base_url("http://localhost:47681").is_ok());
        assert!(parse_daemon_base_url("https://127.0.0.1:47681").is_err());
        assert!(parse_daemon_base_url("http://192.168.1.10:47681").is_err());
        assert!(parse_daemon_base_url("http://127.0.0.1:47681/path").is_err());
        assert!(parse_daemon_base_url("http://user@127.0.0.1:47681").is_err());
    }

    #[test]
    fn cloud_provider_requires_explicit_opt_in() {
        assert!(!provider_requires_cloud_opt_in("local"));
        assert!(provider_requires_cloud_opt_in("openai"));
    }

    #[test]
    fn control_token_path_matches_daemon_runtime_contract() {
        assert_eq!(
            control_token_path(Path::new("/run/user/1000")),
            PathBuf::from("/run/user/1000/translator/control.token")
        );
    }

    #[test]
    fn control_token_validation_matches_daemon_contract() {
        assert!(is_valid_control_token(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ));
        assert!(!is_valid_control_token(
            "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef"
        ));
        assert!(!is_valid_control_token("short"));
    }
}
