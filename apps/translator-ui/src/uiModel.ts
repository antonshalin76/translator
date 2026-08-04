export type AudioDirection = "microphone" | "speaker";
export type DebugControl = "debug_text" | "debug_capture";
export type DebugLifecycleEvent =
  | "session_stop"
  | "provider_switch"
  | "daemon_restart"
  | "ui_close";
export type Language = "ru" | "en";
export type ProviderId = "local" | "openai";
export type TranslationMode = "quality_first" | "balanced" | "streaming_first";
export type VoiceEngine = "piper" | "silero" | "openai";
export type VoiceGender = "male" | "female";
export type AudioMixField = keyof AudioMixVolumes;

export interface DebugTextEvent {
  transcript: string;
  translation: string;
}

export interface DirectionState {
  direction_id: AudioDirection;
  source_language: Language;
  target_language: Language;
  enabled: boolean;
  voice_profile: VoiceProfile;
}

export interface AudioMixVolumes {
  microphone_original_percent: number;
  microphone_translation_percent: number;
  speaker_original_percent: number;
  speaker_translation_percent: number;
}

export interface VoiceProfile {
  language: Language;
  gender: VoiceGender;
  engine: VoiceEngine;
  model_path?: string | null;
  provider_voice_id?: string | null;
}

export interface LatencyPolicyState {
  direction_id: AudioDirection;
  current_mode: TranslationMode;
  p50_first_audio_ms?: number | null;
  p95_first_audio_ms: number;
  p95_last_audio_ms: number;
  p95_queue_lag_ms: number;
  reason?: string | null;
}

export interface AudioEndpointState {
  role: string;
  kind: string;
  name: string;
  endpoint_id?: number | null;
  owner_module_id?: number | null;
  available: boolean;
  daemon_owned: boolean;
}

export interface AudioGraphState {
  health: string;
  endpoints: AudioEndpointState[];
  owned_module_ids: number[];
  safe_error?: {
    code: string;
    safe_message: string;
    retryable: boolean;
  } | null;
}

export interface RouteCandidate {
  stream_id: number;
  application: string;
  stable_app_key: string;
  application_name: string;
  process_binary: string;
  pipewire_node_name?: string | null;
  media_role?: string | null;
  description?: string | null;
  current_sink_id: number;
  current_sink_name: string;
  call_like: boolean;
}

export interface IncomingRoute {
  stream_id: number;
  application: string;
  stable_app_key: string;
  original_sink_id: number;
  original_sink_name: string;
  target_sink_name: string;
  route_method?: "pulse_move" | "pipe_wire_links";
  pipewire_node_name?: string | null;
}

export interface RoutingState {
  candidates: RouteCandidate[];
  source_outputs: Array<Record<string, unknown>>;
  conflicting_stream_ids: number[];
  active_route?: IncomingRoute | null;
  resolution: string;
}

export interface DeviceState {
  source: DeviceSelectionState;
  sink: DeviceSelectionState;
  acoustic: {
    mode: string;
    aec_capability: Record<string, unknown> | string;
    full_duplex_allowed: boolean;
    warning?: string | null;
  };
}

export interface DeviceSelectionState {
  health: string;
  selected?: PhysicalDevice | null;
  pinned_name?: string | null;
  current_default?: string | null;
  pending_default?: string | null;
}

export interface PhysicalDevice {
  id: number;
  name: string;
  description: string;
  active_port?: string | null;
  active_port_type?: string | null;
  available: boolean;
}

export interface RoundTripLatency {
  outgoing_first_audio_ms?: number | null;
  english_monitor_complete_ms?: number | null;
  incoming_first_audio_ms?: number | null;
  physical_mic_onset_to_returned_ru_first_audible_ms?: number | null;
}

export interface RoundTripStatus {
  session_id?: string | null;
  checkpoint?: string | null;
  recursion_count: number;
  latency: RoundTripLatency;
  exact_pcm?: Record<string, unknown> | null;
  safe_error?: string | null;
  debug_text?: DebugTextEvent | null;
}

export interface RoundTripSelfTestState {
  availability: string;
  preconditions?: Record<string, unknown> | null;
  status: RoundTripStatus;
}

export interface RuntimeSnapshot {
  translation_running: boolean;
  debug_text_enabled: boolean;
  debug_capture_enabled: boolean;
  directions?: DirectionState[];
  audio_mix?: AudioMixVolumes;
  provider_id: ProviderId;
  audio_leaves_machine?: boolean;
  latency_policy?: LatencyPolicyState[];
  audio_graph?: AudioGraphState | null;
  routes?: RoutingState | null;
  devices?: DeviceState | null;
  self_test?: RoundTripSelfTestState;
}

export interface UiModel {
  debugTextWarning: boolean;
  debugCaptureWarning: boolean;
  cloudWarningVisible: boolean;
  audioLeavesMachine: boolean;
  acousticFeedbackWarning: boolean;
  visibleDebugText: DebugTextEvent[];
  latencyDebt: ReturnType<typeof classifyTask7LatencyDebt>;
  diagnostics: ReturnType<typeof roundTripControlState>;
  audioMix: AudioMixVolumes;
}

const MAX_DEBUG_TEXT_EVENTS = 200;
const MAX_DEBUG_TEXT_BYTES = 1024 * 1024;
export const DEFAULT_AUDIO_MIX: AudioMixVolumes = {
  microphone_original_percent: 0,
  microphone_translation_percent: 100,
  speaker_original_percent: 0,
  speaker_translation_percent: 100,
};

export class DebugTextRing {
  readonly storageMode = "memory";

  #events: DebugTextEvent[] = [];
  #bytesUsed = 0;

  constructor(
    private readonly maxEvents = MAX_DEBUG_TEXT_EVENTS,
    private readonly maxBytes = MAX_DEBUG_TEXT_BYTES,
  ) {}

  push(event: DebugTextEvent): boolean {
    const eventBytes = debugTextBytes(event);
    if (eventBytes > this.maxBytes) {
      return false;
    }
    while (
      this.#events.length >= this.maxEvents ||
      this.#bytesUsed + eventBytes > this.maxBytes
    ) {
      const removed = this.#events.shift();
      if (!removed) {
        break;
      }
      this.#bytesUsed -= debugTextBytes(removed);
    }
    this.#events.push(event);
    this.#bytesUsed += eventBytes;
    return true;
  }

  snapshot(): DebugTextEvent[] {
    return this.#events.map((event) => ({ ...event }));
  }

  clear(): void {
    this.#events = [];
    this.#bytesUsed = 0;
  }

  handleLifecycleEvent(event: DebugLifecycleEvent): void {
    if (lifecycleEventClearsDebugText(event)) {
      this.clear();
    }
  }
}

export function buildUiModel(snapshot: RuntimeSnapshot): UiModel {
  const status = snapshot.self_test?.status;
  const debugText =
    snapshot.debug_text_enabled && status?.debug_text ? [status.debug_text] : [];

  return {
    debugTextWarning: snapshot.debug_text_enabled,
    debugCaptureWarning: snapshot.debug_capture_enabled,
    cloudWarningVisible: snapshot.provider_id === "openai",
    audioLeavesMachine:
      snapshot.audio_leaves_machine ?? snapshot.provider_id === "openai",
    acousticFeedbackWarning:
      snapshot.devices?.acoustic.full_duplex_allowed === false ||
      snapshot.audio_graph?.health === "degraded",
    visibleDebugText: debugText,
    latencyDebt: classifyTask7LatencyDebt(status),
    diagnostics: roundTripControlState(status),
    audioMix: normalizeAudioMix(snapshot.audio_mix),
  };
}

function normalizeAudioMix(volumes: AudioMixVolumes | undefined): AudioMixVolumes {
  if (!volumes) {
    return { ...DEFAULT_AUDIO_MIX };
  }
  return {
    microphone_original_percent: clampVolume(volumes.microphone_original_percent),
    microphone_translation_percent: clampVolume(volumes.microphone_translation_percent),
    speaker_original_percent: clampVolume(volumes.speaker_original_percent),
    speaker_translation_percent: clampVolume(volumes.speaker_translation_percent),
  };
}

function clampVolume(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value)));
}

export function debugToggleIntent(control: DebugControl, enabled: boolean) {
  return {
    command:
      control === "debug_text"
        ? "translator_set_debug_text"
        : "translator_set_debug_capture",
    args: { enabled },
  };
}

export function providerPatchIntent(
  providerId: ProviderId,
  cloudOptIn: boolean,
) {
  const cloudWarningVisible = providerId === "openai";
  if (cloudWarningVisible && !cloudOptIn) {
    return {
      blocked: true,
      code: "cloud_provider_opt_in_required",
      cloudWarningVisible,
      command: null,
      args: null,
    };
  }
  return {
    blocked: false,
    code: null,
    cloudWarningVisible,
    command: "translator_set_provider",
    args: { providerId, cloudOptIn },
  };
}

export function cloudOptInChangeIntent(
  currentProviderId: ProviderId,
  cloudOptIn: boolean,
) {
  if (currentProviderId === "openai" && !cloudOptIn) {
    return {
      command: "translator_set_provider",
      args: { providerId: "local", cloudOptIn: false },
      revokesCloudProvider: true,
    };
  }
  return {
    command: null,
    args: null,
    revokesCloudProvider: false,
  };
}

export function audioMixPatchIntent(field: AudioMixField, value: number) {
  return {
    command: "translator_set_audio_mix",
    args: { [audioMixArgument(field)]: clampVolume(value) },
  };
}

export function currentAudioMixPatchIntent(
  field: AudioMixField,
  value: number,
  snapshot: RuntimeSnapshot,
): ReturnType<typeof audioMixPatchIntent> | null {
  const current = normalizeAudioMix(snapshot.audio_mix)[field];
  const next = clampVolume(value);
  if (current !== next) {
    return null;
  }
  return audioMixPatchIntent(field, next);
}

export function directionToggleIntent(
  directionId: AudioDirection,
  enabled: boolean,
) {
  return {
    command: "translator_set_direction",
    args: { directionId, enabled },
  };
}

function audioMixArgument(field: AudioMixField): string {
  const parts = field.split("_");
  return parts
    .map((part, index) =>
      index === 0 ? part : part.charAt(0).toUpperCase() + part.slice(1),
    )
    .join("");
}

export function classifyTask7LatencyDebt(status: RoundTripStatus | undefined) {
  const total =
    status?.latency.physical_mic_onset_to_returned_ru_first_audible_ms ?? null;
  if (!total) {
    return {
      classification: "unknown",
      requiresProviderComparison: true,
    };
  }
  if (total <= 1000) {
    return {
      classification: "meets_target",
      requiresProviderComparison: false,
    };
  }
  if (total <= 1500) {
    return {
      classification: "usable_degraded",
      requiresProviderComparison: false,
    };
  }
  return {
    classification: "fails_usable_limit",
    requiresProviderComparison: true,
  };
}

export function roundTripControlState(status: RoundTripStatus | undefined) {
  const checkpoint = status?.checkpoint ?? null;
  const active =
    checkpoint !== null &&
    !["completed", "failed", "stopped"].includes(checkpoint);

  return {
    primaryAction: active ? "stop" : "start",
    stopVisible: active,
    checkpoint,
    teardownComplete: checkpoint === "stopped" || checkpoint === "completed",
    safeError: status?.safe_error ?? null,
  };
}

export function lifecycleEventClearsDebugText(
  event: DebugLifecycleEvent,
): boolean {
  return [
    "session_stop",
    "provider_switch",
    "daemon_restart",
    "ui_close",
  ].includes(event);
}

export function labelLanguage(language: Language | undefined): string {
  switch (language) {
    case "ru":
      return "Русский";
    case "en":
      return "English";
    default:
      return "—";
  }
}

export function labelMode(mode: TranslationMode | undefined): string {
  switch (mode) {
    case "quality_first":
      return "Quality-first";
    case "balanced":
      return "Balanced";
    case "streaming_first":
      return "Streaming-first";
    default:
      return "—";
  }
}

export function labelVoiceGender(gender: VoiceGender | undefined): string {
  switch (gender) {
    case "female":
      return "Женский";
    case "male":
      return "Мужской";
    default:
      return "—";
  }
}

export function formatMs(value: number | null | undefined): string {
  return typeof value === "number" && value > 0 ? `${value} ms` : "Нет данных";
}

function debugTextBytes(event: DebugTextEvent): number {
  return new TextEncoder().encode(event.transcript + event.translation).length;
}
