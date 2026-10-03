import { describe, expect, test } from "bun:test";

import {
  aecCalibrationControlState,
  aecCalibrationPendingCancel,
  audioMixPatchIntent,
  buildUiModel,
  classifyTask7LatencyDebt,
  cloudOptInChangeIntent,
  currentAudioMixPatchIntent,
  DebugTextRing,
  debugToggleIntent,
  directionToggleIntent,
  headphoneConfirmationIntent,
  lifecycleEventClearsDebugText,
  providerPatchIntent,
  roundTripControlState,
  translationControlState,
  type AecCalibrationStatus,
  type AecProofStatus,
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

describe("translation lifecycle recovery", () => {
  test("pending cleanup and failure expose Stop without claiming a completed stop", () => {
    for (const runtime_status of ["cleanup_pending", "failed"]) {
      const control = translationControlState({ ...snapshotWithRawDebugText, runtime_status });
      expect(control.command).toBe("translator_stop");
      expect(control.label).not.toBe("Start");
      expect(control.summary).not.toBe("Перевод остановлен");
      expect(control.statusText).not.toBe("Остановлен");
    }
  });

  test("Start is exposed only for a stopped runtime, unknown state fails closed", () => {
    expect(translationControlState({ ...snapshotWithRawDebugText, runtime_status: "stopped" }).command).toBe("translator_start");
    expect(translationControlState({ ...snapshotWithRawDebugText, runtime_status: "running", translation_running: true }).command).toBe("translator_stop");
    expect(translationControlState(snapshotWithRawDebugText).command).toBeNull();
  });
});

describe("explicit physical headphone confirmation", () => {
  const device = (name: string) => ({
    id: 41,
    name,
    description: "USB audio",
    active_port: "analog-output",
    active_port_type: "Analog",
    available: true,
  });
  const selected = (name: string) => ({
    health: "available",
    selected: device(name),
    pinned_name: name,
  });
  const snapshot: RuntimeSnapshot = {
    ...snapshotWithRawDebugText,
    runtime_status: "stopped",
    devices: {
      source: selected("alsa_input.usb-headset"),
      sink: selected("alsa_output.usb-headset"),
      acoustic: { mode: "unknown_unsafe", aec_capability: "unavailable", full_duplex_allowed: false },
    },
  };

  test("explicit intent transports the exact selected pair without changing directions", () => {
    expect(headphoneConfirmationIntent(snapshot, true)).toEqual({
      command: "translator_confirm_headphones",
      args: { confirmation: { source: snapshot.devices!.source.selected, sink: snapshot.devices!.sink.selected } },
    });
    expect(snapshot.devices!.acoustic.mode).toBe("unknown_unsafe");
  });

  test("running, cleanup, unavailable and known speakers cannot be confirmed", () => {
    expect(headphoneConfirmationIntent({ ...snapshot, translation_running: true }, true)).toBeNull();
    expect(headphoneConfirmationIntent({ ...snapshot, runtime_status: "cleanup_pending" }, true)).toBeNull();
    for (const mode of ["headphones", "open_speaker"]) {
      expect(headphoneConfirmationIntent({ ...snapshot, devices: { ...snapshot.devices!, acoustic: { ...snapshot.devices!.acoustic, mode } } }, true)).toBeNull();
    }
    expect(headphoneConfirmationIntent({ ...snapshot, devices: { ...snapshot.devices!, sink: { ...snapshot.devices!.sink, health: "device_unavailable" } } }, true)).toBeNull();
    expect(headphoneConfirmationIntent({ ...snapshot, devices: { ...snapshot.devices!, sink: { ...snapshot.devices!.sink, pinned_name: "other-device" } } }, true)).toBeNull();
  });

  test("revocation sends null only for a stopped user-confirmed path", () => {
    const confirmed = { ...snapshot, devices: { ...snapshot.devices!, acoustic: { ...snapshot.devices!.acoustic, mode: "user_confirmed_headphones" } } };
    expect(headphoneConfirmationIntent(confirmed, false)).toEqual({ command: "translator_confirm_headphones", args: { confirmation: null } });
    expect(headphoneConfirmationIntent(snapshot, false)).toBeNull();
    expect(headphoneConfirmationIntent({ ...confirmed, translation_running: true }, false)).toBeNull();
  });
});

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
  test("AEC cancellation polling keeps only the matching active intent", () => {
    const running = { state: "running", attempt_id: "attempt-2" } as const;
    expect(aecCalibrationPendingCancel(running, "attempt-2")).toBe("attempt-2");
    expect(aecCalibrationPendingCancel(running, "other-attempt")).toBeNull();
    expect(aecCalibrationPendingCancel(running, null)).toBeNull();
    for (const state of ["cancelled", "timed_out", "cleanup_uncertain"] as const) {
      expect(aecCalibrationPendingCancel({ state, attempt_id: "attempt-2" }, "attempt-2")).toBeNull();
    }
    for (const state of ["unavailable", "shutting_down"] as const) {
      expect(aecCalibrationPendingCancel({ state }, "attempt-2")).toBeNull();
    }
    expect(aecCalibrationPendingCancel({ state: "failed", attempt_id: "attempt-2", code: "failed" }, "attempt-2")).toBeNull();
  });

  test("every succeeded AEC attempt exposes the existing release action without claiming proof validity", () => {
    const proofs: AecProofStatus[] = [
      { state: "validated", source_name: "source", sink_name: "sink", expires_monotonic_ns: 42 },
      { state: "unavailable" },
      { state: "measuring" },
      { state: "validation_failed" },
      { state: "cleanup_uncertain" },
    ];
    for (const proof of proofs) {
      const succeeded: AecCalibrationStatus = { state: "succeeded", attempt_id: "retained-attempt", proof };
      const control = aecCalibrationControlState(succeeded, null);
      expect(control.cancelAttemptId).toBe("retained-attempt");
      expect(control.canStart).toBe(proof.state === "validated");
      if (proof.state !== "validated") {
        expect(control.label).toBe("Завершена без действующего подтверждения");
      }
      expect(aecCalibrationControlState(succeeded, "daemon_unavailable")).toMatchObject({
        canStart: false,
        cancelAttemptId: null,
      });
    }
  });

  test("same-attempt succeeded polling preserves pending release and suppresses duplicate intents", () => {
    const succeeded: AecCalibrationStatus = {
      state: "succeeded",
      attempt_id: "retained-attempt",
      proof: { state: "validated", source_name: "source", sink_name: "sink", expires_monotonic_ns: 42 },
    };
    const pending = aecCalibrationPendingCancel(succeeded, "retained-attempt");
    expect(pending).toBe("retained-attempt");
    expect(aecCalibrationControlState(succeeded, null, pending)).toEqual({
      label: "Отмена запрошена, ожидаем завершения",
      canStart: false,
      cancelAttemptId: null,
    });
    expect(aecCalibrationPendingCancel(succeeded, "other-attempt")).toBeNull();
    expect(aecCalibrationControlState(succeeded, null, "other-attempt")).toMatchObject({
      canStart: true,
      cancelAttemptId: "retained-attempt",
    });
    for (const status of [
      { state: "cancelled", attempt_id: "retained-attempt" },
      { state: "shutting_down" },
      { state: "cleanup_uncertain", attempt_id: "retained-attempt" },
      { state: "failed", attempt_id: "retained-attempt", code: "cleanup_failed" },
    ] satisfies AecCalibrationStatus[]) {
      expect(aecCalibrationPendingCancel(status, pending)).toBeNull();
      expect(aecCalibrationControlState(status, null, pending).cancelAttemptId).toBeNull();
    }
  });

  test("AEC calibration stays unavailable on controller error and cleanup uncertainty", () => {
    expect(aecCalibrationControlState(null, "aec_calibration_controller_unavailable")).toMatchObject({
      canStart: false,
      cancelAttemptId: null,
      label: "Контроллер недоступен",
    });
    expect(aecCalibrationControlState({ state: "cleanup_uncertain", attempt_id: "attempt-1" }, null)).toMatchObject({
      canStart: false,
      cancelAttemptId: null,
      label: "Очистка не подтверждена",
    });
    expect(aecCalibrationControlState({ state: "cleanup_uncertain", attempt_id: "attempt-1" }, "daemon_unavailable")).toMatchObject({
      canStart: false,
      label: "Очистка не подтверждена; текущий статус недоступен",
    });
  });

  test("AEC calibration cancellation remains pending until daemon reports terminal state", () => {
    const running = { state: "running", attempt_id: "attempt-2" } as const;
    expect(aecCalibrationControlState(running, null, "attempt-2")).toMatchObject({
      canStart: false,
      cancelAttemptId: null,
      label: "Отмена запрошена, ожидаем завершения",
    });
    expect(aecCalibrationControlState(running, null)).toMatchObject({
      canStart: false,
      cancelAttemptId: "attempt-2",
    });
    expect(aecCalibrationControlState({ state: "cancelled", attempt_id: "attempt-2" }, null)).toMatchObject({
      canStart: true,
      cancelAttemptId: null,
      label: "Отменена",
    });
  });

  test("AEC calibration never borrows proof from a translation snapshot", () => {
    expect(aecCalibrationControlState(null, null)).toMatchObject({ canStart: false });
    expect(aecCalibrationControlState({ state: "unavailable" }, null)).toMatchObject({ canStart: true });
    expect(aecCalibrationControlState({ state: "shutting_down" }, null)).toMatchObject({ canStart: false });
    expect(aecCalibrationControlState({ state: "succeeded", attempt_id: "a", proof: { state: "cleanup_uncertain" } }, null)).toMatchObject({ canStart: false });
  });

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

  test("volume intent compares against acknowledged state without mutating it", () => {
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
    ).toEqual({
      command: "translator_set_audio_mix",
      args: { speakerTranslationPercent: 52 },
    });
    expect(
      currentAudioMixPatchIntent("speaker_translation_percent", 100, refreshed),
    ).toBeNull();
    expect(refreshed.audio_mix?.speaker_translation_percent).toBe(100);
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
