import { invoke } from "@tauri-apps/api/core";

import {
  DEFAULT_AUDIO_MIX,
  DebugTextRing,
  buildUiModel,
  cloudOptInChangeIntent,
  currentAudioMixPatchIntent,
  debugToggleIntent,
  directionToggleIntent,
  formatMs,
  labelLanguage,
  providerPatchIntent,
  roundTripControlState,
  type AudioDirection,
  type AudioMixField,
  type AudioMixVolumes,
  type DebugControl,
  type DirectionState,
  type Language,
  type LatencyPolicyState,
  type ProviderId,
  type RouteCandidate,
  type RuntimeSnapshot,
  type TranslationMode,
  type VoiceGender,
} from "./uiModel";

type UiSection = "status" | "routes" | "diagnostics";

interface AppState {
  snapshot: RuntimeSnapshot;
  connected: boolean;
  cloudOptIn: boolean;
  busy: string | null;
  error: string | null;
  lastUpdated: Date | null;
  activeSection: UiSection;
}

const root = document.querySelector<HTMLDivElement>("#app");
const debugTextRing = new DebugTextRing();
const audioMixTimers: Partial<Record<AudioMixField, number>> = {};
let lastDebugTextKey: string | null = null;
let wasDisconnected = true;

const state: AppState = {
  snapshot: defaultSnapshot(),
  connected: false,
  cloudOptIn: false,
  busy: null,
  error: null,
  lastUpdated: null,
  activeSection: "status",
};

if (!root) {
  throw new Error("app root is missing");
}
const appRoot = root;

render();
await refreshStatus();
window.setInterval(() => {
  void refreshStatus();
}, 2000);
window.addEventListener("pagehide", () => {
  clearDebugText("ui_close");
});

function render(): void {
  const model = buildUiModel(state.snapshot);
  appRoot.replaceChildren(appShell(model));
}

function appShell(model: ReturnType<typeof buildUiModel>): HTMLElement {
  const shell = element("div", "app-shell");
  shell.append(sidebar(model));

  const main = element("main");
  main.append(
    topbar(),
    warningBand(model),
    runtimeBand(model),
    controlsBand(),
    audioMixBand(model),
    directionGrid(),
    routeSection(),
    graphSection(),
    diagnosticsSection(model),
    debugTextSection(model),
  );
  shell.append(main);
  return shell;
}

function sidebar(model: ReturnType<typeof buildUiModel>): HTMLElement {
  const aside = element("aside", "sidebar");
  aside.setAttribute("aria-label", "Разделы");

  const brand = element("div", "brand");
  brand.append(element("span", "brand-mark", "T"), element("span", "", "Translator"));

  const nav = element("nav");
  nav.append(
    navItem("Состояние", "status"),
    navItem("Маршруты", "routes"),
    navItem("Диагностика", "diagnostics"),
  );

  const privacy = element("div", "privacy-state");
  privacy.append(
    element("span", `status-dot ${model.cloudWarningVisible ? "warn" : "ok"}`),
    element(
      "span",
      "",
      model.cloudWarningVisible ? "Включен cloud egress" : "Аудио локально",
    ),
  );

  aside.append(brand, nav, privacy);
  return aside;
}

function navItem(text: string, section: UiSection): HTMLElement {
  const active = state.activeSection === section;
  const item = element("a", `nav-item ${active ? "active" : ""}`, text);
  item.setAttribute("href", `#${section}`);
  if (active) {
    item.setAttribute("aria-current", "page");
  }
  item.addEventListener("click", (event) => {
    event.preventDefault();
    state.activeSection = section;
    render();
    window.requestAnimationFrame(() => {
      document.getElementById(section)?.scrollIntoView({ block: "start" });
    });
  });
  return item;
}

function topbar(): HTMLElement {
  const header = element("header", "topbar");
  const title = element("div");
  title.append(
    element("p", "eyebrow", "Локальный сервис"),
    element("h1", "", "Состояние переводчика"),
  );
  const status = element(
    "span",
    `build-state ${state.connected ? "ready" : "offline"}`,
    state.connected ? "Daemon connected" : "Daemon offline",
  );
  header.append(title, status);
  return header;
}

function warningBand(model: ReturnType<typeof buildUiModel>): HTMLElement {
  const band = element("section", "warning-stack");
  const warnings: HTMLElement[] = [];
  if (model.debugTextWarning) {
    warnings.push(warning("debug", "Debug text включен: transcript/translation видны только в этом окне."));
  }
  if (model.debugCaptureWarning) {
    warnings.push(warning("capture", "Debug capture пишет аудио-артефакты в отладочное хранилище daemon."));
  }
  if (model.cloudWarningVisible) {
    warnings.push(warning("cloud", "Cloud provider включен: аудио уходит во внешний сервис."));
  }
  if (model.acousticFeedbackWarning) {
    warnings.push(
      warning(
        "acoustic",
        "Acoustic fallback активен: сервис продолжает аудио, но возможна обратная связь без AEC/наушников.",
      ),
    );
  }
  if (model.latencyDebt.classification === "fails_usable_limit") {
    warnings.push(
      warning(
        "latency",
        "Task 7 latency debt: local provider fails usable limit; нужен provider comparison.",
      ),
    );
  }
  if (state.error) {
    warnings.push(warning("error", state.error));
  }
  band.replaceChildren(...warnings);
  return band;
}

function warning(kind: string, text: string): HTMLElement {
  const item = element("div", `warning warning-${kind}`);
  item.append(element("span", "status-dot warn"), element("span", "", text));
  return item;
}

function runtimeBand(model: ReturnType<typeof buildUiModel>): HTMLElement {
  const section = element("section", "status-band");
  section.id = "status";
  const text = element("div");
  text.append(
    element("h2", "", "Runtime"),
    element(
      "p",
      "",
      [
        state.snapshot.translation_running ? "Перевод запущен" : "Перевод остановлен",
        `Provider: ${state.snapshot.provider_id}`,
        `Provider health: ${providerHealthText()}`,
        `Diagnostics: ${model.diagnostics.checkpoint ?? "idle"}`,
      ].join(" · "),
    ),
  );
  const pill = element(
    "span",
    `status-pill ${state.snapshot.translation_running ? "running" : "idle"}`,
  );
  pill.append(
    element("span", "status-dot"),
    textNode(state.snapshot.translation_running ? "Запущен" : "Остановлен"),
  );
  section.append(text, pill);
  return section;
}

function controlsBand(): HTMLElement {
  const section = element("section", "control-band");
  section.append(
    commandButton(
      state.snapshot.translation_running ? "Stop" : "Start",
      state.snapshot.translation_running ? "translator_stop" : "translator_start",
      undefined,
      "translation",
    ),
    providerControl(),
    debugControl("debug_text"),
    debugControl("debug_capture"),
  );
  return section;
}

function providerControl(): HTMLElement {
  const wrap = element("label", "control-item");
  wrap.append(element("span", "", "Provider"));
  const select = selectControl<ProviderId>(
    state.snapshot.provider_id,
    [
      ["local", "Local"],
      ["openai", "OpenAI"],
    ],
    (providerId) => {
      const intent = providerPatchIntent(providerId, state.cloudOptIn);
      if (intent.blocked) {
        state.error = "OpenAI требует явный cloud opt-in.";
        render();
        return;
      }
      debugTextRing.handleLifecycleEvent("provider_switch");
      void invokeAction(intent.command, intent.args, "provider");
    },
  );
  wrap.append(select);

  const optIn = element("label", "inline-check");
  const checkbox = element("input") as HTMLInputElement;
  checkbox.type = "checkbox";
  checkbox.checked = state.cloudOptIn;
  checkbox.disabled = state.busy !== null || !state.connected;
  checkbox.addEventListener("change", () => {
    state.cloudOptIn = checkbox.checked;
    const revocation = cloudOptInChangeIntent(
      state.snapshot.provider_id,
      checkbox.checked,
    );
    if (revocation.revokesCloudProvider) {
      clearDebugText("provider_switch");
      void invokeAction(
        revocation.command,
        revocation.args,
        "provider",
      );
      return;
    }
    render();
  });
  optIn.append(checkbox, element("span", "", "Cloud opt-in"));
  wrap.append(optIn);
  return wrap;
}

function debugControl(control: DebugControl): HTMLElement {
  const isDebugText = control === "debug_text";
  const intent = debugToggleIntent(
    control,
    isDebugText
      ? !state.snapshot.debug_text_enabled
      : !state.snapshot.debug_capture_enabled,
  );
  const enabled =
    isDebugText
      ? state.snapshot.debug_text_enabled
      : state.snapshot.debug_capture_enabled;
  const button = element("button", `control-button ${enabled ? "active" : ""}`);
  button.type = "button";
  button.disabled = state.busy !== null;
  const controlState = enabled ? "On" : "Off";
  button.textContent = isDebugText
    ? `Debug text ${controlState}`
    : `Capture ${controlState}`;
  button.addEventListener("click", () => {
    if (isDebugText && enabled) {
      clearDebugText();
    }
    void invokeAction(intent.command, intent.args, control);
  });
  return button;
}

function audioMixBand(model: ReturnType<typeof buildUiModel>): HTMLElement {
  const section = element("section", "section-band audio-mix-band");
  const mix = model.audioMix;
  section.append(
    sectionHeading(
      "Громкость",
      `Mic ${mix.microphone_original_percent}/${mix.microphone_translation_percent} · Speaker ${mix.speaker_original_percent}/${mix.speaker_translation_percent}`,
    ),
  );
  const grid = element("div", "audio-mix-grid");
  grid.append(
    audioMixSlider(
      "Микрофон original",
      "microphone_original_percent",
      mix.microphone_original_percent,
    ),
    audioMixSlider(
      "Микрофон translation",
      "microphone_translation_percent",
      mix.microphone_translation_percent,
    ),
    audioMixSlider(
      "Динамики original",
      "speaker_original_percent",
      mix.speaker_original_percent,
    ),
    audioMixSlider(
      "Динамики translation",
      "speaker_translation_percent",
      mix.speaker_translation_percent,
    ),
  );
  section.append(grid);
  return section;
}

function audioMixSlider(
  labelText: string,
  field: AudioMixField,
  value: number,
): HTMLElement {
  const label = element("label", "mix-slider");
  const caption = element("div", "mix-caption");
  const valueNode = element("strong", "", `${value}%`);
  caption.append(element("span", "", labelText), valueNode);

  const input = element("input") as HTMLInputElement;
  input.type = "range";
  input.min = "0";
  input.max = "100";
  input.step = "1";
  input.value = String(value);
  input.disabled = state.busy !== null || !state.connected;
  input.addEventListener("input", () => {
    valueNode.textContent = `${input.value}%`;
    queueAudioMixChange(field, Number(input.value));
  });
  input.addEventListener("change", () => {
    flushAudioMixChange(field, Number(input.value));
  });

  label.append(caption, input);
  return label;
}

function directionGrid(): HTMLElement {
  const section = element("section", "direction-grid");
  section.setAttribute("aria-label", "Направления перевода");
  section.append(directionPanel("microphone"), directionPanel("speaker"));
  return section;
}

function directionPanel(directionId: AudioDirection): HTMLElement {
  const direction = directionState(directionId);
  const latency = latencyState(directionId);
  const enabled = directionEnabled(direction);
  const panel = element("article", `direction-panel ${enabled ? "active" : "inactive"}`);

  const heading = element("div", "direction-heading");
  const copy = element("div");
  copy.append(
    element("p", "eyebrow", directionId === "microphone" ? "Микрофон" : "Динамики"),
    element(
      "h2",
      "",
      `${labelLanguage(direction.source_language)} → ${labelLanguage(direction.target_language)}`,
    ),
  );
  const directionStateText = state.connected
    ? directionEnabledText(enabled)
    : "Ожидание";
  heading.append(
    copy,
    element("span", `state-text ${enabled ? "enabled" : "disabled"}`, directionStateText),
  );

  const pairControls = element("div", "segmented");
  pairControls.append(
    pairButton(direction, "ru", "en"),
    pairButton(direction, "en", "ru"),
  );

  const mode = selectControl<TranslationMode>(
    latency?.current_mode ?? "quality_first",
    [
      ["quality_first", "Quality"],
      ["balanced", "Balanced"],
      ["streaming_first", "Streaming"],
    ],
    (currentMode) =>
      void invokeAction(
        "translator_set_latency_mode",
        { directionId, currentMode },
        `${directionId}-mode`,
      ),
  );

  const voice = selectControl<VoiceGender>(
    direction.voice_profile.gender,
    [
      ["male", "Мужской"],
      ["female", "Женский"],
    ],
    (gender) =>
      void invokeAction(
        "translator_set_voice_profile",
        {
          directionId,
          language: direction.target_language,
          gender,
          engine: state.snapshot.provider_id === "openai" ? "openai" : "piper",
        },
        `${directionId}-voice`,
      ),
  );

  const controls = element("div", "inline-controls");
  controls.append(
    directionToggle(direction),
    labeledControl("Mode", mode),
    labeledControl("Voice", voice),
  );

  panel.append(heading, pairControls, controls);
  return panel;
}

function directionToggle(direction: DirectionState): HTMLElement {
  const enabled = directionEnabled(direction);
  const intent = directionToggleIntent(direction.direction_id, !enabled);
  const label = element("label", `direction-toggle ${enabled ? "enabled" : ""}`);
  const input = element("input") as HTMLInputElement;
  input.type = "checkbox";
  input.checked = enabled;
  input.disabled = state.busy !== null || !state.connected;
  input.addEventListener("change", () => {
    void invokeAction(intent.command, intent.args, `${direction.direction_id}-enabled`);
  });
  label.append(input, element("span", "toggle-track"), element("span", "", "Канал"));
  return label;
}

function pairButton(direction: DirectionState, source: Language, target: Language): HTMLElement {
  const active =
    direction.source_language === source && direction.target_language === target;
  const button = element("button", `segment ${active ? "active" : ""}`);
  button.type = "button";
  button.textContent = `${source.toUpperCase()} → ${target.toUpperCase()}`;
  button.disabled = state.busy !== null || active;
  button.addEventListener("click", () => {
    void invokeAction(
      "translator_set_direction",
      {
        directionId: direction.direction_id,
        sourceLanguage: source,
        targetLanguage: target,
      },
      `${direction.direction_id}-direction`,
    );
  });
  return button;
}

function routeSection(): HTMLElement {
  const section = element("section", "section-band");
  section.id = "routes";
  section.append(sectionHeading("Маршруты", selectedRouteText()));
  const list = element("div", "route-list");
  const candidates = state.snapshot.routes?.candidates ?? [];
  if (candidates.length === 0) {
    list.append(element("p", "empty", "Нет входящих route candidates."));
  } else {
    for (const candidate of candidates) {
      list.append(routeCandidate(candidate));
    }
  }
  section.append(list);
  return section;
}

function routeCandidate(candidate: RouteCandidate): HTMLElement {
  const row = element("div", "route-row");
  const details = element("div");
  details.append(
    element("strong", "", candidate.application_name),
    element(
      "span",
      "",
      `${candidate.process_binary} · stream ${candidate.stream_id} · ${candidate.current_sink_name}`,
    ),
  );
  const button = commandButton(
    "Select",
    "translator_select_route",
    { streamId: candidate.stream_id },
    `route-${candidate.stream_id}`,
  );
  row.append(details, button);
  return row;
}

function graphSection(): HTMLElement {
  const section = element("section", "section-band");
  section.append(sectionHeading("Audio graph", graphSummary()));
  const endpoints = element("div", "endpoint-grid");
  for (const endpoint of state.snapshot.audio_graph?.endpoints ?? []) {
    const item = element("div", "endpoint-item");
    item.append(
      element("span", "endpoint-name", endpoint.name),
      element(
        "span",
        `endpoint-state ${endpoint.available ? "ready" : "missing"}`,
        endpoint.available ? "ready" : "missing",
      ),
    );
    endpoints.append(item);
  }
  if (!endpoints.childElementCount) {
    endpoints.append(element("p", "empty", "Audio graph не опубликован daemon."));
  }
  section.append(endpoints, deviceStrip());
  return section;
}

function deviceStrip(): HTMLElement {
  const strip = element("div", "device-strip");
  const source = state.snapshot.devices?.source.selected;
  const sink = state.snapshot.devices?.sink.selected;
  strip.append(
    detailBlock("Physical mic", source?.description ?? source?.name ?? "Нет данных"),
    detailBlock("Physical sink", sink?.description ?? sink?.name ?? "Нет данных"),
    detailBlock("Acoustic mode", state.snapshot.devices?.acoustic.mode ?? "unknown"),
  );
  return strip;
}

function diagnosticsSection(model: ReturnType<typeof buildUiModel>): HTMLElement {
  const section = element("section", "section-band");
  section.id = "diagnostics";
  const status = state.snapshot.self_test?.status;
  section.append(sectionHeading("Diagnostics", `Checkpoint: ${status?.checkpoint ?? "idle"}`));

  const control = roundTripControlState(status);
  const button = commandButton(
    control.primaryAction === "stop" ? "Stop self-test" : "Start self-test",
    control.primaryAction === "stop"
      ? "translator_stop_round_trip"
      : "translator_start_round_trip",
    undefined,
    "round-trip",
  );

  const metrics = element("dl", "metrics");
  metrics.append(
    detail("Outgoing", formatMs(status?.latency.outgoing_first_audio_ms)),
    detail("English monitor", formatMs(status?.latency.english_monitor_complete_ms)),
    detail("Incoming", formatMs(status?.latency.incoming_first_audio_ms)),
    detail(
      "Total",
      formatMs(status?.latency.physical_mic_onset_to_returned_ru_first_audible_ms),
    ),
    detail("Recursion", String(status?.recursion_count ?? 0)),
    detail("Teardown", model.diagnostics.teardownComplete ? "complete" : "pending"),
    detail("Safe error", status?.safe_error ?? "Нет"),
  );

  const preconditions = preconditionsList(state.snapshot.self_test?.preconditions ?? null);
  section.append(button, metrics, preconditions);
  return section;
}

function debugTextSection(model: ReturnType<typeof buildUiModel>): HTMLElement {
  const section = element("section", "section-band");
  section.append(sectionHeading("Debug text", model.debugTextWarning ? "Visible" : "Hidden"));
  if (!model.debugTextWarning) {
    clearDebugText();
    section.append(element("p", "empty", "Transcript/translation скрыты."));
    return section;
  }
  for (const event of debugTextRing.snapshot()) {
    const row = element("div", "debug-row");
    row.append(
      detailBlock("Transcript", event.transcript),
      detailBlock("Translation", event.translation),
    );
    section.append(row);
  }
  if (debugTextRing.snapshot().length === 0) {
    section.append(element("p", "empty", "Debug text включен, событий пока нет."));
  }
  return section;
}

function commandButton(
  label: string,
  command: string | null,
  args: Record<string, unknown> | null | undefined,
  busyKey: string,
): HTMLButtonElement {
  const button = element("button", "control-button") as HTMLButtonElement;
  button.type = "button";
  button.textContent = state.busy === busyKey ? "..." : label;
  button.disabled = state.busy !== null || !state.connected || command === null;
  button.addEventListener("click", () => {
    if (command) {
      void invokeAction(command, args ?? undefined, busyKey);
    }
  });
  return button;
}

async function invokeAction(
  command: string | null,
  args: Record<string, unknown> | null | undefined,
  busyKey: string,
): Promise<void> {
  if (!command) {
    return;
  }
  state.busy = busyKey;
  state.error = null;
  render();
  try {
    const snapshot = await invokeDaemon<RuntimeSnapshot>(command, args ?? undefined);
    if (command === "translator_stop") {
      clearDebugText("session_stop");
    }
    applySnapshot(snapshot);
  } catch (error) {
    state.error = safeErrorMessage(error);
    render();
  } finally {
    state.busy = null;
    render();
  }
}

function queueAudioMixChange(field: AudioMixField, value: number): void {
  updateLocalAudioMix(field, value);
  if (audioMixTimers[field] !== undefined) {
    window.clearTimeout(audioMixTimers[field]);
  }
  audioMixTimers[field] = window.setTimeout(() => {
    delete audioMixTimers[field];
    void invokeAudioMixChange(field, value);
  }, 150);
}

function flushAudioMixChange(field: AudioMixField, value: number): void {
  updateLocalAudioMix(field, value);
  if (audioMixTimers[field] !== undefined) {
    window.clearTimeout(audioMixTimers[field]);
    delete audioMixTimers[field];
  }
  void invokeAudioMixChange(field, value);
}

async function invokeAudioMixChange(field: AudioMixField, value: number): Promise<void> {
  if (!state.connected) {
    return;
  }
  const intent = currentAudioMixPatchIntent(field, value, state.snapshot);
  if (!intent) {
    return;
  }
  try {
    const snapshot = await invokeDaemon<RuntimeSnapshot>(intent.command, intent.args);
    applySnapshot(snapshot);
  } catch (error) {
    state.error = safeErrorMessage(error);
    state.lastUpdated = new Date();
    render();
  }
}

function updateLocalAudioMix(field: AudioMixField, value: number): void {
  const current: AudioMixVolumes = buildUiModel(state.snapshot).audioMix;
  state.snapshot.audio_mix = { ...current, [field]: value };
}

async function refreshStatus(): Promise<void> {
  if (!tauriRuntimeAvailable()) {
    state.connected = false;
    state.error = "Tauri runtime недоступен: открыт browser preview.";
    state.lastUpdated = new Date();
    render();
    return;
  }

  try {
    const snapshot = await invokeDaemon<RuntimeSnapshot>("translator_status");
    if (wasDisconnected) {
      clearDebugText("daemon_restart");
    }
    wasDisconnected = false;
    state.connected = true;
    state.error = null;
    applySnapshot(snapshot);
  } catch (error) {
    if (!wasDisconnected) {
      clearDebugText("daemon_restart");
    }
    wasDisconnected = true;
    state.connected = false;
    state.error = safeErrorMessage(error);
    state.lastUpdated = new Date();
    render();
  }
}

async function invokeDaemon<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  if (!tauriRuntimeAvailable()) {
    throw new Error("tauri_runtime_unavailable");
  }
  return invoke<T>(command, args);
}

function applySnapshot(snapshot: RuntimeSnapshot): void {
  const previousSessionId = state.snapshot.self_test?.status.session_id ?? null;
  const nextSessionId = snapshot.self_test?.status.session_id ?? null;
  if (state.snapshot.translation_running && !snapshot.translation_running) {
    clearDebugText("session_stop");
  }
  if (state.snapshot.provider_id !== snapshot.provider_id) {
    clearDebugText("provider_switch");
  }
  if (previousSessionId && previousSessionId !== nextSessionId) {
    clearDebugText("daemon_restart");
  }
  if (!snapshot.debug_text_enabled) {
    clearDebugText();
  }

  state.snapshot = snapshot;
  if (snapshot.provider_id === "openai") {
    state.cloudOptIn = true;
  }
  state.lastUpdated = new Date();
  const debugText = snapshot.self_test?.status.debug_text;
  if (snapshot.debug_text_enabled && debugText) {
    const key = `${snapshot.self_test?.status.session_id ?? ""}:${debugText.transcript}:${debugText.translation}`;
    if (key !== lastDebugTextKey) {
      debugTextRing.push(debugText);
      lastDebugTextKey = key;
    }
  }
  render();
}

function clearDebugText(event?: Parameters<DebugTextRing["handleLifecycleEvent"]>[0]): void {
  if (event) {
    debugTextRing.handleLifecycleEvent(event);
  } else {
    debugTextRing.clear();
  }
  lastDebugTextKey = null;
}

function tauriRuntimeAvailable(): boolean {
  return "__TAURI_INTERNALS__" in window || "__TAURI__" in window;
}

function defaultSnapshot(): RuntimeSnapshot {
  return {
    translation_running: false,
    debug_text_enabled: false,
    debug_capture_enabled: false,
    provider_id: "local",
    audio_mix: { ...DEFAULT_AUDIO_MIX },
    directions: [
      defaultDirection("microphone", "ru", "en"),
      defaultDirection("speaker", "en", "ru"),
    ],
    latency_policy: [
      defaultLatency("microphone"),
      defaultLatency("speaker"),
    ],
    self_test: {
      availability: "unavailable",
      preconditions: null,
      status: {
        checkpoint: null,
        recursion_count: 0,
        latency: {},
      },
    },
  };
}

function defaultDirection(
  directionId: AudioDirection,
  sourceLanguage: Language,
  targetLanguage: Language,
): DirectionState {
  return {
    direction_id: directionId,
    source_language: sourceLanguage,
    target_language: targetLanguage,
    enabled: true,
    voice_profile: {
      language: targetLanguage,
      gender: "male",
      engine: "piper",
    },
  };
}

function defaultLatency(directionId: AudioDirection): LatencyPolicyState {
  return {
    direction_id: directionId,
    current_mode: "quality_first",
    p50_first_audio_ms: null,
    p95_first_audio_ms: 0,
    p95_last_audio_ms: 0,
    p95_queue_lag_ms: 0,
    reason: null,
  };
}

function directionState(directionId: AudioDirection): DirectionState {
  const sourceLanguage = directionId === "microphone" ? "ru" : "en";
  const targetLanguage = directionId === "microphone" ? "en" : "ru";
  return (
    state.snapshot.directions?.find((direction) => direction.direction_id === directionId) ??
    defaultDirection(directionId, sourceLanguage, targetLanguage)
  );
}

function directionEnabled(direction: DirectionState): boolean {
  return direction.enabled !== false;
}

function directionEnabledText(enabled: boolean): string {
  return enabled ? "Включен" : "Отключен";
}

function latencyState(directionId: AudioDirection): LatencyPolicyState | undefined {
  return state.snapshot.latency_policy?.find((latency) => latency.direction_id === directionId);
}

function selectControl<T extends string>(
  value: T,
  options: Array<[T, string]>,
  onChange: (value: T) => void,
): HTMLSelectElement {
  const select = element("select") as HTMLSelectElement;
  select.disabled = state.busy !== null || !state.connected;
  for (const [optionValue, label] of options) {
    const option = element("option") as HTMLOptionElement;
    option.value = optionValue;
    option.textContent = label;
    select.append(option);
  }
  select.value = value;
  select.addEventListener("change", () => onChange(select.value as T));
  return select;
}

function sectionHeading(title: string, summary: string): HTMLElement {
  const heading = element("div", "section-heading");
  heading.append(element("h2", "", title), element("p", "", summary));
  return heading;
}

function labeledControl(label: string, control: HTMLElement): HTMLElement {
  const wrap = element("label", "control-item compact");
  wrap.append(element("span", "", label), control);
  return wrap;
}

function detail(label: string, value: string): HTMLElement {
  const row = element("div");
  row.append(element("dt", "", label), element("dd", "", value));
  return row;
}

function detailBlock(label: string, value: string): HTMLElement {
  const block = element("div", "detail-block");
  block.append(element("span", "", label), element("strong", "", value));
  return block;
}

function selectedRouteText(): string {
  const route = state.snapshot.routes?.active_route;
  if (!route) {
    return `Resolution: ${state.snapshot.routes?.resolution ?? "unknown"}`;
  }
  return `${route.application} stream ${route.stream_id} → ${route.target_sink_name}`;
}

function providerHealthText(): string {
  const preconditions = state.snapshot.self_test?.preconditions;
  if (!preconditions) {
    return state.connected ? "not probed" : "offline";
  }
  const outgoing = preconditions["outgoing_provider_ready"];
  const incoming = preconditions["incoming_provider_ready"];
  if (outgoing === true && incoming === true) {
    return "ready";
  }
  if (outgoing === false || incoming === false) {
    return "unavailable";
  }
  return "unknown";
}

function graphSummary(): string {
  return `Health: ${state.snapshot.audio_graph?.health ?? "unknown"}`;
}

function preconditionsList(preconditions: Record<string, unknown> | null): HTMLElement {
  const list = element("div", "preconditions");
  if (!preconditions) {
    list.append(element("p", "empty", "Preconditions пока не опубликованы."));
    return list;
  }
  for (const [key, value] of Object.entries(preconditions)) {
    list.append(detailBlock(key, String(value)));
  }
  return list;
}

function safeErrorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null && "code" in error) {
    return String((error as { code: unknown }).code);
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "unknown_error";
}

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className = "",
  text = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text) {
    node.textContent = text;
  }
  return node;
}

function textNode(text: string): Text {
  return document.createTextNode(text);
}
