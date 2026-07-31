use std::env;
use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use serde_json::{Value, json};
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, Runtime, WindowEvent};

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
            translator_set_latency_mode,
            translator_set_voice_profile,
            translator_select_route,
            translator_round_trip_status,
            translator_start_round_trip,
            translator_stop_round_trip,
        ])
        .run(tauri::generate_context!())
        .expect("failed to run translator UI");
}

#[tauri::command]
fn translator_status() -> Result<Value, UiError> {
    daemon_request(DaemonCommand::Status, None)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrayItemState {
    enabled: bool,
    checked: bool,
}

impl TrayItemState {
    const fn new(enabled: bool, checked: bool) -> Self {
        Self { enabled, checked }
    }
}

#[derive(Debug, Clone)]
struct TrayDirectionState {
    enabled: bool,
    source_language: String,
    target_language: String,
}

#[derive(Debug, Clone)]
struct TrayMenuState {
    translation_running: bool,
    round_trip_active: bool,
    provider_id: String,
    debug_text_enabled: bool,
    debug_capture_enabled: bool,
    microphone: TrayDirectionState,
    speaker: TrayDirectionState,
    common_mode: Option<String>,
}

impl TrayMenuState {
    fn from_snapshot(snapshot: Option<&Value>) -> Self {
        let snapshot = snapshot.unwrap_or(&Value::Null);
        Self {
            translation_running: bool_field(snapshot, "translation_running", false),
            round_trip_active: round_trip_active(snapshot),
            provider_id: string_field(snapshot, "provider_id", "local"),
            debug_text_enabled: bool_field(snapshot, "debug_text_enabled", false),
            debug_capture_enabled: bool_field(snapshot, "debug_capture_enabled", false),
            microphone: direction_state(snapshot, "microphone", "ru", "en"),
            speaker: direction_state(snapshot, "speaker", "en", "ru"),
            common_mode: common_latency_mode(snapshot),
        }
    }

    fn item(&self, id: &str) -> TrayItemState {
        let channel_controls_enabled = !self.translation_running;
        match id {
            "start" => TrayItemState::new(
                !self.translation_running && (self.microphone.enabled || self.speaker.enabled),
                false,
            ),
            "stop" => TrayItemState::new(self.translation_running, false),
            "self_test_start" => {
                TrayItemState::new(!self.translation_running && !self.round_trip_active, false)
            }
            "self_test_stop" => TrayItemState::new(self.round_trip_active, false),
            "provider_local" => TrayItemState::new(
                self.provider_id == "local" || channel_controls_enabled,
                self.provider_id == "local",
            ),
            "provider_openai" => TrayItemState::new(
                self.provider_id == "openai" || channel_controls_enabled,
                self.provider_id == "openai",
            ),
            "debug_text_toggle" => TrayItemState::new(true, self.debug_text_enabled),
            "debug_capture_toggle" => TrayItemState::new(true, self.debug_capture_enabled),
            "mic_enabled_toggle" => TrayItemState::new(true, self.microphone.enabled),
            "speaker_enabled_toggle" => TrayItemState::new(true, self.speaker.enabled),
            "mic_ru_en" => {
                self.direction_pair_item(&self.microphone, channel_controls_enabled, "ru", "en")
            }
            "mic_en_ru" => {
                self.direction_pair_item(&self.microphone, channel_controls_enabled, "en", "ru")
            }
            "speaker_en_ru" => {
                self.direction_pair_item(&self.speaker, channel_controls_enabled, "en", "ru")
            }
            "speaker_ru_en" => {
                self.direction_pair_item(&self.speaker, channel_controls_enabled, "ru", "en")
            }
            "mode_quality_first" => self.mode_item("quality_first"),
            "mode_balanced" => self.mode_item("balanced"),
            "mode_streaming_first" => self.mode_item("streaming_first"),
            _ => TrayItemState::new(true, false),
        }
    }

    fn direction_pair_item(
        &self,
        direction: &TrayDirectionState,
        controls_enabled: bool,
        source_language: &str,
        target_language: &str,
    ) -> TrayItemState {
        let checked = direction.source_language == source_language
            && direction.target_language == target_language;
        TrayItemState::new(checked || controls_enabled, checked)
    }

    fn mode_item(&self, mode: &str) -> TrayItemState {
        let checked = self.common_mode.as_deref() == Some(mode);
        TrayItemState::new(true, checked)
    }
}

fn setup_tray(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = daemon_request(DaemonCommand::Status, None).ok();
    let menu = tray_menu(app, snapshot.as_ref())?;

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
    let state = TrayMenuState::from_snapshot(snapshot);
    let show = MenuItem::with_id(manager, "show", "Show Translator", true, None::<&str>)?;
    let start = tray_menu_item(manager, "start", "Start Translation", state.item("start"))?;
    let stop = tray_menu_item(manager, "stop", "Stop Translation", state.item("stop"))?;
    let self_test_start = tray_menu_item(
        manager,
        "self_test_start",
        "Start Round-Trip Self-Test",
        state.item("self_test_start"),
    )?;
    let self_test_stop = tray_menu_item(
        manager,
        "self_test_stop",
        "Stop Round-Trip Self-Test",
        state.item("self_test_stop"),
    )?;
    let provider_local = tray_check_item(
        manager,
        "provider_local",
        "Local Provider",
        state.item("provider_local"),
    )?;
    let provider_openai = tray_check_item(
        manager,
        "provider_openai",
        "OpenAI Provider (cloud opt-in)",
        state.item("provider_openai"),
    )?;
    let debug_text = tray_check_item(
        manager,
        "debug_text_toggle",
        "Debug Text",
        state.item("debug_text_toggle"),
    )?;
    let debug_capture = tray_check_item(
        manager,
        "debug_capture_toggle",
        "Debug Capture",
        state.item("debug_capture_toggle"),
    )?;
    let mic_enabled = tray_check_item(
        manager,
        "mic_enabled_toggle",
        "Mic Channel",
        state.item("mic_enabled_toggle"),
    )?;
    let speaker_enabled = tray_check_item(
        manager,
        "speaker_enabled_toggle",
        "Speaker Channel",
        state.item("speaker_enabled_toggle"),
    )?;
    let mic_ru_en = tray_check_item(
        manager,
        "mic_ru_en",
        "Mic RU -> EN",
        state.item("mic_ru_en"),
    )?;
    let mic_en_ru = tray_check_item(
        manager,
        "mic_en_ru",
        "Mic EN -> RU",
        state.item("mic_en_ru"),
    )?;
    let speaker_en_ru = tray_check_item(
        manager,
        "speaker_en_ru",
        "Speaker EN -> RU",
        state.item("speaker_en_ru"),
    )?;
    let speaker_ru_en = tray_check_item(
        manager,
        "speaker_ru_en",
        "Speaker RU -> EN",
        state.item("speaker_ru_en"),
    )?;
    let mode_quality = tray_check_item(
        manager,
        "mode_quality_first",
        "Mode Quality First",
        state.item("mode_quality_first"),
    )?;
    let mode_balanced = tray_check_item(
        manager,
        "mode_balanced",
        "Mode Balanced",
        state.item("mode_balanced"),
    )?;
    let mode_streaming = tray_check_item(
        manager,
        "mode_streaming_first",
        "Mode Streaming First",
        state.item("mode_streaming_first"),
    )?;
    let quit = MenuItem::with_id(manager, "quit", "Quit", true, None::<&str>)?;
    let sep_runtime = PredefinedMenuItem::separator(manager)?;
    let sep_provider = PredefinedMenuItem::separator(manager)?;
    let sep_debug = PredefinedMenuItem::separator(manager)?;
    let sep_channels = PredefinedMenuItem::separator(manager)?;
    let sep_modes = PredefinedMenuItem::separator(manager)?;
    let sep_quit = PredefinedMenuItem::separator(manager)?;

    Menu::with_items(
        manager,
        &[
            &show,
            &sep_runtime,
            &start,
            &stop,
            &self_test_start,
            &self_test_stop,
            &sep_provider,
            &provider_local,
            &provider_openai,
            &sep_debug,
            &debug_text,
            &debug_capture,
            &sep_channels,
            &mic_enabled,
            &speaker_enabled,
            &mic_ru_en,
            &mic_en_ru,
            &speaker_en_ru,
            &speaker_ru_en,
            &sep_modes,
            &mode_quality,
            &mode_balanced,
            &mode_streaming,
            &sep_quit,
            &quit,
        ],
    )
}

fn tray_menu_item<M, R>(
    manager: &M,
    id: &str,
    text: &str,
    state: TrayItemState,
) -> tauri::Result<MenuItem<R>>
where
    M: Manager<R>,
    R: Runtime,
{
    MenuItem::with_id(manager, id, text, state.enabled, None::<&str>)
}

fn tray_check_item<M, R>(
    manager: &M,
    id: &str,
    text: &str,
    state: TrayItemState,
) -> tauri::Result<CheckMenuItem<R>>
where
    M: Manager<R>,
    R: Runtime,
{
    CheckMenuItem::with_id(
        manager,
        id,
        text,
        state.enabled,
        state.checked,
        None::<&str>,
    )
}

fn handle_tray_menu_event<R: Runtime>(app: &AppHandle<R>, id: &str) {
    let _ = match id {
        "show" => {
            show_main_window(app);
            Ok(Value::Null)
        }
        "start" => daemon_request(DaemonCommand::StartTranslation, None),
        "stop" => daemon_request(DaemonCommand::StopTranslation, None),
        "self_test_start" => daemon_request(DaemonCommand::StartRoundTrip, None),
        "self_test_stop" => daemon_request(DaemonCommand::StopRoundTrip, None),
        "provider_local" => set_provider_from_tray("local", false),
        "provider_openai" => set_provider_from_tray("openai", true),
        "debug_text_toggle" => {
            toggle_debug_flag("debug_text_enabled", DaemonCommand::PatchDebugText)
        }
        "debug_capture_toggle" => {
            toggle_debug_flag("debug_capture_enabled", DaemonCommand::PatchDebugCapture)
        }
        "mic_enabled_toggle" => toggle_direction_enabled_from_tray("microphone"),
        "speaker_enabled_toggle" => toggle_direction_enabled_from_tray("speaker"),
        "mic_ru_en" => set_direction_from_tray("microphone", "ru", "en"),
        "mic_en_ru" => set_direction_from_tray("microphone", "en", "ru"),
        "speaker_en_ru" => set_direction_from_tray("speaker", "en", "ru"),
        "speaker_ru_en" => set_direction_from_tray("speaker", "ru", "en"),
        "mode_quality_first" => {
            set_both_latency_modes_from_tray("quality_first").map(|()| Value::Null)
        }
        "mode_balanced" => set_both_latency_modes_from_tray("balanced").map(|()| Value::Null),
        "mode_streaming_first" => {
            set_both_latency_modes_from_tray("streaming_first").map(|()| Value::Null)
        }
        "quit" => {
            let _ = daemon_request(
                DaemonCommand::PatchDebugText,
                Some(json!({ "enabled": false })),
            );
            app.exit(0);
            Ok(Value::Null)
        }
        _ => Ok(Value::Null),
    };
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
        DaemonCommand::StartTranslation
            | DaemonCommand::StopTranslation
            | DaemonCommand::PatchDebugText
            | DaemonCommand::PatchDebugCapture
            | DaemonCommand::PatchDirection
            | DaemonCommand::PatchProvider
            | DaemonCommand::PatchLatencyPolicy
            | DaemonCommand::StartRoundTrip
            | DaemonCommand::StopRoundTrip
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
    let menu = tray_menu(app, snapshot)
        .map_err(|_| UiError::new("tray_menu_unavailable", "tray menu could not be rebuilt"))?;
    let tray = app
        .tray_by_id(TRAY_ID)
        .ok_or_else(|| UiError::new("tray_unavailable", "tray icon is unavailable"))?;
    tray.set_menu(Some(menu))
        .map_err(|_| UiError::new("tray_menu_unavailable", "tray menu could not be set"))
}

fn bool_field(snapshot: &Value, field: &str, default: bool) -> bool {
    snapshot
        .get(field)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

fn string_field(snapshot: &Value, field: &str, default: &str) -> String {
    snapshot
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

fn direction_state(
    snapshot: &Value,
    direction_id: &str,
    default_source: &str,
    default_target: &str,
) -> TrayDirectionState {
    let direction = snapshot
        .get("directions")
        .and_then(Value::as_array)
        .and_then(|directions| {
            directions.iter().find(|direction| {
                direction.get("direction_id").and_then(Value::as_str) == Some(direction_id)
            })
        });
    TrayDirectionState {
        enabled: direction
            .and_then(|direction| direction.get("enabled"))
            .and_then(Value::as_bool)
            .unwrap_or(true),
        source_language: direction
            .and_then(|direction| direction.get("source_language"))
            .and_then(Value::as_str)
            .unwrap_or(default_source)
            .to_owned(),
        target_language: direction
            .and_then(|direction| direction.get("target_language"))
            .and_then(Value::as_str)
            .unwrap_or(default_target)
            .to_owned(),
    }
}

fn common_latency_mode(snapshot: &Value) -> Option<String> {
    let mut modes = snapshot
        .get("latency_policy")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(|policy| policy.get("current_mode").and_then(Value::as_str));
    let first = modes.next()?.to_owned();
    if modes.all(|mode| mode == first) {
        Some(first)
    } else {
        None
    }
}

fn round_trip_active(snapshot: &Value) -> bool {
    snapshot
        .pointer("/self_test/status/checkpoint")
        .and_then(Value::as_str)
        .is_some_and(|checkpoint| !matches!(checkpoint, "completed" | "failed" | "stopped"))
}

fn show_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn set_provider_from_tray(provider_id: &str, cloud_opt_in: bool) -> Result<Value, UiError> {
    if provider_requires_cloud_opt_in(provider_id) && !cloud_opt_in {
        return Err(UiError::new(
            "cloud_provider_opt_in_required",
            "cloud provider requires explicit opt-in",
        ));
    }
    daemon_request(
        DaemonCommand::PatchProvider,
        Some(json!({ "provider_id": provider_id, "cloud_opt_in": cloud_opt_in })),
    )
}

fn set_direction_from_tray(
    direction_id: &str,
    source_language: &str,
    target_language: &str,
) -> Result<Value, UiError> {
    daemon_request(
        DaemonCommand::PatchDirection,
        Some(json!({
            "direction_id": direction_id,
            "source_language": source_language,
            "target_language": target_language,
        })),
    )
}

fn toggle_direction_enabled_from_tray(direction_id: &str) -> Result<Value, UiError> {
    let snapshot = daemon_request(DaemonCommand::Status, None)?;
    let direction = direction_state(
        &snapshot,
        direction_id,
        default_source_language(direction_id),
        default_target_language(direction_id),
    );
    daemon_request(
        DaemonCommand::PatchDirection,
        Some(json!({
            "direction_id": direction_id,
            "enabled": !direction.enabled,
        })),
    )
}

fn default_source_language(direction_id: &str) -> &'static str {
    if direction_id == "speaker" {
        "en"
    } else {
        "ru"
    }
}

fn default_target_language(direction_id: &str) -> &'static str {
    if direction_id == "speaker" {
        "ru"
    } else {
        "en"
    }
}

fn set_both_latency_modes_from_tray(current_mode: &str) -> Result<(), UiError> {
    for direction_id in ["microphone", "speaker"] {
        daemon_request(
            DaemonCommand::PatchLatencyPolicy,
            Some(json!({
                "direction_id": direction_id,
                "current_mode": current_mode,
            })),
        )?;
    }
    Ok(())
}

fn toggle_debug_flag(field_name: &str, command: DaemonCommand) -> Result<Value, UiError> {
    let snapshot = daemon_request(DaemonCommand::Status, None)?;
    let enabled = snapshot
        .get(field_name)
        .and_then(Value::as_bool)
        .unwrap_or(false);
    daemon_request(command, Some(json!({ "enabled": !enabled })))
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
        "routing_controller_unavailable" => Some("routing_controller_unavailable"),
        "routing_controller_failed" => Some("routing_controller_failed"),
        "manual_route_failed" => Some("manual_route_failed"),
        "audio_mix_apply_failed" => Some("audio_mix_apply_failed"),
        "audio_mix_controller_failed" => Some("audio_mix_controller_failed"),
        "invalid_audio_mix_volume" => Some("invalid_audio_mix_volume"),
        "self_test_unavailable" => Some("self_test_unavailable"),
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
    fn tray_state_marks_selected_controls_and_dims_inactive_actions() {
        let snapshot = json!({
            "translation_running": true,
            "debug_text_enabled": true,
            "debug_capture_enabled": false,
            "provider_id": "local",
            "directions": [
                {
                    "direction_id": "microphone",
                    "source_language": "ru",
                    "target_language": "en",
                    "enabled": true
                },
                {
                    "direction_id": "speaker",
                    "source_language": "en",
                    "target_language": "ru",
                    "enabled": false
                }
            ],
            "latency_policy": [
                { "direction_id": "microphone", "current_mode": "balanced" },
                { "direction_id": "speaker", "current_mode": "balanced" }
            ],
            "self_test": {
                "status": {
                    "checkpoint": null
                }
            }
        });

        let state = TrayMenuState::from_snapshot(Some(&snapshot));

        assert_eq!(state.item("start"), TrayItemState::new(false, false));
        assert_eq!(state.item("stop"), TrayItemState::new(true, false));
        assert_eq!(state.item("provider_local"), TrayItemState::new(true, true));
        assert_eq!(
            state.item("provider_openai"),
            TrayItemState::new(false, false)
        );
        assert_eq!(
            state.item("debug_text_toggle"),
            TrayItemState::new(true, true)
        );
        assert_eq!(
            state.item("debug_capture_toggle"),
            TrayItemState::new(true, false)
        );
        assert_eq!(
            state.item("mic_enabled_toggle"),
            TrayItemState::new(true, true)
        );
        assert_eq!(
            state.item("speaker_enabled_toggle"),
            TrayItemState::new(true, false)
        );
        assert_eq!(state.item("mic_ru_en"), TrayItemState::new(true, true));
        assert_eq!(state.item("mic_en_ru"), TrayItemState::new(false, false));
        assert_eq!(state.item("speaker_en_ru"), TrayItemState::new(true, true));
        assert_eq!(state.item("mode_balanced"), TrayItemState::new(true, true));
        assert_eq!(
            state.item("mode_quality_first"),
            TrayItemState::new(true, false)
        );
    }

    #[test]
    fn webview_commands_refresh_tray_when_their_state_is_visible_in_the_menu() {
        for command in [
            DaemonCommand::StartTranslation,
            DaemonCommand::StopTranslation,
            DaemonCommand::PatchDebugText,
            DaemonCommand::PatchDebugCapture,
            DaemonCommand::PatchDirection,
            DaemonCommand::PatchProvider,
            DaemonCommand::PatchLatencyPolicy,
            DaemonCommand::StartRoundTrip,
            DaemonCommand::StopRoundTrip,
        ] {
            assert!(webview_command_refreshes_tray(command), "{command:?}");
        }

        for command in [
            DaemonCommand::Status,
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
