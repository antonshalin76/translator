import { describe, expect, test } from "bun:test";

import {
  audioMixPatchIntent,
  buildUiModel,
  classifyTask7LatencyDebt,
  cloudOptInChangeIntent,
  currentAudioMixPatchIntent,
  DebugTextRing,
  debugToggleIntent,
  directionToggleIntent,
  lifecycleEventClearsDebugText,
  providerPatchIntent,
  roundTripControlState,
  type RuntimeSnapshot,
} from "./uiModel";

const snapshotWithRawDebugText: RuntimeSnapshot = {
  translation_running: false,
  debug_text_enabled: false,
  debug_capture_enabled: false,
  provider_id: "local",
  self_test: {
    availability: "available",
    status: {
      checkpoint: "completed",
      recursion_count: 0,
      latency: {},
      debug_text: {
        transcript: "raw source phrase",
        translation: "raw translated phrase",
      },
    },
  },
};

describe("UI privacy contracts", () => {
  test("normal mode never exposes transcript or translation text", () => {
    const model = buildUiModel(snapshotWithRawDebugText);

    expect(model.debugTextWarning).toBe(false);
    expect(model.visibleDebugText).toEqual([]);
  });

  test("debug text warning is independent from debug capture", () => {
    const model = buildUiModel({
      ...snapshotWithRawDebugText,
      debug_text_enabled: true,
      debug_capture_enabled: false,
    });

    expect(model.debugTextWarning).toBe(true);
    expect(model.debugCaptureWarning).toBe(false);
  });

  test("debug_text and debug_capture controls call separate Tauri commands", () => {
    expect(debugToggleIntent("debug_text", true)).toEqual({
      command: "translator_set_debug_text",
      args: { enabled: true },
    });
    expect(debugToggleIntent("debug_capture", false)).toEqual({
      command: "translator_set_debug_capture",
      args: { enabled: false },
    });
  });

  test("debug text ring is bounded and clearable without browser storage", () => {
    const ring = new DebugTextRing(2, 40);

    expect(ring.push({ transcript: "one", translation: "two" })).toBe(true);
    expect(
      ring.push({
        transcript: "this event is too large for the configured ring",
        translation: "",
      }),
    ).toBe(false);
    expect(ring.push({ transcript: "three", translation: "four" })).toBe(true);
    expect(ring.push({ transcript: "five", translation: "six" })).toBe(true);

    expect(ring.snapshot()).toEqual([
      { transcript: "three", translation: "four" },
      { transcript: "five", translation: "six" },
    ]);

    ring.clear();
    expect(ring.snapshot()).toEqual([]);
    expect(ring.storageMode).toBe("memory");
  });

  test("debug text ring clears on every Task 8 lifecycle trigger", () => {
    for (const event of [
      "session_stop",
      "provider_switch",
      "daemon_restart",
      "ui_close",
    ] as const) {
      const ring = new DebugTextRing();
      ring.push({ transcript: "private-marker", translation: "private-marker" });

      expect(lifecycleEventClearsDebugText(event)).toBe(true);
      ring.handleLifecycleEvent(event);
      expect(ring.snapshot()).toEqual([]);
    }
  });
});

describe("UI safety gates", () => {
  test("cloud provider selection requires explicit opt-in", () => {
    expect(providerPatchIntent("openai", false)).toMatchObject({
      blocked: true,
      code: "cloud_provider_opt_in_required",
      cloudWarningVisible: true,
    });
    expect(providerPatchIntent("openai", true)).toMatchObject({
      blocked: false,
      command: "translator_set_provider",
      cloudWarningVisible: true,
    });
    expect(providerPatchIntent("local", false)).toMatchObject({
      blocked: false,
      command: "translator_set_provider",
      cloudWarningVisible: false,
    });
  });

  test("cloud egress status is visible before a cloud session starts", () => {
    const model = buildUiModel({
      ...snapshotWithRawDebugText,
      provider_id: "openai",
      audio_leaves_machine: true,
    });

    expect(model.cloudWarningVisible).toBe(true);
    expect(model.audioLeavesMachine).toBe(true);
  });

  test("revoking cloud opt-in switches an active cloud provider back to local", () => {
    expect(cloudOptInChangeIntent("openai", false)).toEqual({
      command: "translator_set_provider",
      args: { providerId: "local", cloudOptIn: false },
      revokesCloudProvider: true,
    });
    expect(cloudOptInChangeIntent("local", false)).toEqual({
      command: null,
      args: null,
      revokesCloudProvider: false,
    });
  });

  test("Task 7 accepted routing is still visible as latency debt", () => {
    const debt = classifyTask7LatencyDebt({
      checkpoint: "completed",
      recursion_count: 0,
      latency: {
        physical_mic_onset_to_returned_ru_first_audible_ms: 5968,
        outgoing_first_audio_ms: 1885,
        incoming_first_audio_ms: 1418,
      },
    });

    expect(debt).toEqual({
      classification: "fails_usable_limit",
      requiresProviderComparison: true,
    });
  });

  test("round-trip diagnostics expose explicit start and stop states", () => {
    expect(
      roundTripControlState({
        checkpoint: undefined,
        recursion_count: 0,
        latency: {},
      }),
    ).toMatchObject({
      primaryAction: "start",
      stopVisible: false,
      teardownComplete: false,
    });
    expect(
      roundTripControlState({
        checkpoint: "waiting_for_speech",
        recursion_count: 0,
        latency: {},
      }),
    ).toMatchObject({
      primaryAction: "stop",
      stopVisible: true,
      teardownComplete: false,
    });
    expect(
      roundTripControlState({
        checkpoint: "stopped",
        recursion_count: 0,
        latency: {},
      }),
    ).toMatchObject({
      primaryAction: "start",
      stopVisible: false,
      teardownComplete: true,
    });
  });
});

describe("audio mix controls", () => {
  test("volume model falls back to translated-only defaults", () => {
    const model = buildUiModel(snapshotWithRawDebugText);

    expect(model.audioMix).toEqual({
      microphone_original_percent: 0,
      microphone_translation_percent: 100,
      speaker_original_percent: 0,
      speaker_translation_percent: 100,
    });
  });

  test("volume model exposes independent original and translation levels", () => {
    const model = buildUiModel({
      ...snapshotWithRawDebugText,
      audio_mix: {
        microphone_original_percent: 35,
        microphone_translation_percent: 90,
        speaker_original_percent: 55,
        speaker_translation_percent: 80,
      },
    });

    expect(model.audioMix).toEqual({
      microphone_original_percent: 35,
      microphone_translation_percent: 90,
      speaker_original_percent: 55,
      speaker_translation_percent: 80,
    });
  });

  test("audio mix slider intent sends only the changed field", () => {
    expect(audioMixPatchIntent("speaker_original_percent", 65)).toEqual({
      command: "translator_set_audio_mix",
      args: { speakerOriginalPercent: 65 },
    });
  });

  test("stale audio mix slider intent is dropped after snapshot refresh", () => {
    const refreshed: RuntimeSnapshot = {
      ...snapshotWithRawDebugText,
      audio_mix: {
        microphone_original_percent: 60,
        microphone_translation_percent: 100,
        speaker_original_percent: 60,
        speaker_translation_percent: 100,
      },
    };

    expect(
      currentAudioMixPatchIntent("speaker_translation_percent", 52, refreshed),
    ).toBeNull();
    expect(
      currentAudioMixPatchIntent("speaker_translation_percent", 100, refreshed),
    ).toEqual({
      command: "translator_set_audio_mix",
      args: { speakerTranslationPercent: 100 },
    });
  });
});

describe("direction controls", () => {
  test("direction toggle intent updates only channel enabled state", () => {
    expect(directionToggleIntent("speaker", false)).toEqual({
      command: "translator_set_direction",
      args: { directionId: "speaker", enabled: false },
    });
  });
});
