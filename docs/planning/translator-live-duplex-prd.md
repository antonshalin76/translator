# Translator Live Duplex PRD

Status: audited implementation plan.

This package plans a local desktop service for synchronous full-duplex Ru <-> En speech translation on this workstation:

- OS: Ubuntu 24.04.4 LTS, kernel `7.0.0-28-generic`.
- Audio server: PipeWire 1.0.5 through PulseAudio compatibility.
- Default sink: `alsa_output.usb-Jieli_Technology_UACDemoV1.0-00.analog-stereo`.
- Default source: `alsa_input.usb-Jieli_Technology_UACDemoV1.0-00.mono-fallback`.
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU, 12 GB VRAM, driver `580.159.03`.

## Problem

The user needs a desktop service that lets two people speak naturally in Russian and English through Zoom, Google Meet, Telegram Desktop and similar apps.

The service must intercept two active audio directions at the desktop level:

- outgoing local speech from the active physical microphone;
- incoming remote speech from selected communication applications.

Each direction is translated independently and played into the device path where that direction belongs:

- local Russian speech is translated to English and sent to a virtual microphone selected by the call app;
- remote English speech is translated to Russian and played to the active physical sink/headphones.

The service must not depend on one meeting platform's native translation feature. It must own local audio routing through PipeWire/PulseAudio-compatible virtual endpoints and work across supported desktop apps.

## Existing Solutions Research

Research snapshot: 2026-07-28.

No inspected solution is accepted as the implementation base for MVP. Existing products and projects are used as baseline evidence and comparison targets.
Vendor latency and capability claims are not acceptance evidence until they are measured on this workstation with the service audio graph.

| Solution | Fit | Decision |
| --- | --- | --- |
| Sokuji | Open-source desktop/browser app. Public README describes real-time two-way speech translation, Linux builds, virtual microphone support, local inference and cloud providers including OpenAI, Gemini, Palabra and Soniox. | Use as baseline/reference only. Do not inherit architecture until tested on this machine against PipeWire routing, full-duplex isolation and latency. |
| Soniox | Public Russian translation page describes real-time Russian translation, one-way and two-way modes, 60+ languages and speech-to-speech composition through STT and TTS. | Candidate cloud adapter after local MVP or comparison lane. Not first implementation target because the chosen provider order is local first, OpenAI second. |
| OpenAI Realtime Translation | Current OpenAI docs describe `gpt-realtime-translate`, a dedicated realtime translation endpoint that streams translated audio and transcript deltas while source audio is still arriving. | Second provider adapter after local provider. Use for quality/latency comparison and cloud fallback. |
| Gemini Live Translation | Google docs describe low-latency speech-to-speech translation between 70+ languages with audio input/output chunks. | Later comparison adapter. Not first because local-first and OpenAI-second order is fixed. |
| Zoom / Meet / Teams native translation | Platform-local features; language, OS, license and app boundaries vary. | Not a substitute for desktop-wide service. Use only as external benchmark when available. |

Research sources:

- Sokuji README: https://github.com/kizuna-ai-lab/sokuji
- Soniox Russian translation: https://soniox.com/speech-translation/russian
- OpenAI realtime translation: https://developers.openai.com/api/docs/guides/realtime-translation
- Google Gemini Live Translation: https://ai.google.dev/gemini-api/docs/live-api/live-translate
- PipeWire PulseAudio modules: https://docs.pipewire.org/page_pulse_module_null_sink.html, https://docs.pipewire.org/page_pulse_module_remap_source.html, https://docs.pipewire.org/page_man_pw-link_1.html
- PipeWire echo cancellation: https://docs.pipewire.org/page_pulse_module_echo_cancel.html, https://pipewire.pages.freedesktop.org/pipewire/page_module_echo_cancel.html

## Product Decision Ledger

| Decision | Chosen Option |
| --- | --- |
| Product base | Build a first-party local Linux/PipeWire service. Existing solutions are baseline/reference only. |
| Latency/quality mode | Start in Quality-first mode, then automatically degrade to Balanced or Streaming-first when latency thresholds are exceeded. |
| Provider strategy | Hybrid provider layer. Local service owns audio routing; speech provider can be local or cloud. |
| Provider order | Local provider first; OpenAI Realtime Translation second. |
| Incoming audio routing | Per-app virtual sink plus watcher. Do not capture the physical default sink monitor in normal operation. |
| Open-speaker safety | Headphones are the baseline path. Open-speaker mode requires a validated PipeWire WebRTC AEC path; a warning alone is not sufficient. |
| Control surface | Rust daemon, Python ML sidecar, Tauri tray/menu and local status page. |
| Original audio policy | Translated-only by default. Original audio is available only in explicit debug/monitor mode. |
| Persistence | No audio, transcript or translation persistence by default. Debug dumps require explicit manual enablement. |
| App selection | Allowlist selected communication apps, with manual per-stream override. |
| Run mode | User-level systemd service plus Tauri autostart/tray reconnect. |
| Runtime control | systemd owns daemon lifecycle. Tauri controls translation sessions and reports service state; it does not install, enable, disable or restart the unit in MVP. |
| IPC | gRPC bidirectional streaming over a user-owned Unix-domain socket for daemon-sidecar audio; HTTP plus SSE on loopback for Tauri control/status through the Tauri Rust backend. |
| Validation order | Synthetic loopback first; then Telegram Desktop and Firefox/Chromium Meet. Zoom follows as a separate acceptance task. |
| Voice policy | Male/female preset per target language through `VoiceProfile`. No voice cloning and no diarization. |

## Planning Package Contract

The planning package uses the same four-layer shape as the guardrails planning artifacts:

- `translator-live-duplex-prd.md`: product boundary, decisions, acceptance and validation gates.
- `translator-live-duplex-design.md`: architecture, contracts, audio graph, provider interfaces, security and failure behavior.
- `translator-live-duplex-tasks.md`: implementation issues with scope, acceptance and validation.
- `translator-live-duplex-task-prompts.md`: executable prompts for sequential implementation.

Each document must receive a critic review before the next document is finalized. A `REQUEST_CHANGES` verdict blocks the next layer until the document is corrected.

Every implementation task must finish by regenerating and validating `/home/anton/Source/translator/repo-c4.json` with `repo-c4-scan`. Every following task must begin by reading the current index, checking `meta.architecture_fidelity` and `meta.tool_orchestration`, then verifying relevant source before editing. If Task 1 starts without an index, it performs direct inventory first and still generates/validates the index before close.

## Single Responsibility Rule

| Component | Owns | Must Not Own |
| --- | --- | --- |
| Rust Daemon | PipeWire/Pulse endpoint lifecycle, per-app routing watcher, audio capture/playback, local IPC server, sidecar supervision, latency ledger, user-level service health | ASR, machine translation, TTS model internals, transcript persistence, cloud provider credentials in UI state |
| Python ML Sidecar | Local provider inference: ASR, translation, TTS, local model loading, provider latency events | Capture-owned VAD/EOU decisions, PipeWire routing, app sink-input moves, Tauri UI state, systemd installation |
| Provider Adapter | A stable contract for local and cloud speech pipelines, including audio chunks, text deltas, translated audio chunks, status and errors | Desktop audio routing, app selection, persistent meeting history |
| Tauri UI | Tray/menu controls, direction switches, provider/mode selection, voice preset selection, status page, manual stream override, debug toggles | Audio processing, model inference, secret logging, persisted audio or transcripts |
| User-level systemd Unit | Start/restart policy for daemon in the logged-in user session | Root-level audio ownership, global system audio policy |
| Test Harness | Synthetic audio fixtures, loopback latency measurement, per-app routing validation, privacy assertions | Production inference decisions |

Any task that moves audio routing out of the Rust daemon, model inference out of the sidecar/provider adapter, or raw speech persistence into default runtime state is out of scope until the architecture is re-approved.

## Goals

- Provide full-duplex Ru <-> En spoken translation for live desktop calls.
- Capture outgoing physical microphone audio and inject translated output into a virtual microphone.
- Capture incoming selected app audio through a dedicated virtual sink monitor and play translated output to the active physical sink.
- Keep the two translation directions isolated: separate routing, queues, VAD state, provider sessions, cancellation and latency metrics.
- Let the user manually set translation direction per side:
  - microphone side: `ru -> en` or `en -> ru`;
  - speaker side: `en -> ru` or `ru -> en`.
- Start in Quality-first mode and automatically degrade to Balanced or Streaming-first when latency policy requires it.
- Use local provider first, with OpenAI Realtime Translation as the second provider adapter.
- Support male/female voice preset per target language.
- Run as a user-level desktop service integrated with Tauri tray/menu and status page.
- Avoid default persistence of audio, transcripts and translations.
- Validate first with synthetic loopback, then Telegram Desktop and Firefox/Chromium Meet.

## Non-Goals

- Speaker diarization.
- Voice cloning.
- Saving meeting transcripts by default.
- Meeting bot integrations.
- Platform-native translation features as the primary implementation.
- Translation of every system sound.
- Global capture of the physical default sink monitor in normal operation.
- Root/system service ownership of the PipeWire user session.
- Mobile support.
- Windows/macOS support in MVP.
- Full voice catalog and preview UI in MVP.
- Automatic language direction detection as the main control. Manual direction switching is required for MVP.

## Users

- Primary user: the workstation owner using Ru <-> En live calls.
- Remote participant: hears the translated virtual microphone output in the meeting app.
- Local listener: hears translated remote speech in headphones/speakers.
- Developer/operator: inspects local status, routing, provider health and latency during setup.

## User Flows

### First Run

1. User installs or starts the user-level service.
2. Daemon creates virtual endpoints:
   - `Translator_Mic_Out`;
   - `Translator_Virtual_Mic`;
   - `Translator_Remote_In`.
3. Tauri tray/status page connects to the daemon.
4. Status page shows physical source, physical sink, virtual endpoints, provider, model status and app routing readiness.
5. User selects `Translator_Virtual_Mic` as microphone in Zoom/Meet/Telegram.
6. User keeps physical headphones/speakers as the normal output device.

### Outgoing Direction

1. User speaks into the active physical microphone.
2. Rust daemon captures the pinned physical source, not the virtual microphone.
3. Daemon streams audio frames to the active provider session for the microphone direction.
4. Provider returns translated audio and optional transcript/translation deltas.
5. Daemon writes translated audio to `Translator_Mic_Out`.
6. `Translator_Virtual_Mic` exposes the translated stream to the call app.
7. Original local speech is not sent to the virtual microphone in normal mode.

### Incoming Direction

1. Zoom/Meet/Telegram/browser creates an audio playback stream.
2. Routing watcher matches it against the allowlist or user override.
3. Daemon moves the app's sink-input to `Translator_Remote_In`.
4. Daemon captures `Translator_Remote_In.monitor`.
5. Provider translates remote speech.
6. Daemon plays translated audio to the current physical sink/headphones.
7. Original remote speech is not mixed into the physical output in normal mode.

### Human Round-Trip Self-Test

1. User connects headphones, starts the bounded round-trip self-test and speaks a Russian phrase into the physical microphone.
2. The normal outgoing path produces English PCM at `Translator_Virtual_Mic`.
3. A session-bound VirtualPeer captures the exact PCM a call app would read and plays one English monitor tap to the physical headphones.
4. After the English monitor tap completes, VirtualPeer reinjects the same frames as its selected incoming sink-input. The normal incoming path translates them to Russian and plays the result to the physical headphones.
5. Status shows stage checkpoints and outgoing, incoming and total round-trip latency. Transcript and translation text remain hidden unless `debug_text` is enabled.
6. Stop, timeout or failure removes VirtualPeer streams, restores routing, clears memory-only test state and leaves the normal audio graph unchanged.

### Latency Degradation

1. A call starts in Quality-first mode.
2. Daemon records latency segments for both directions.
3. If rolling p95 exceeds configured thresholds, policy lowers the mode:
   - Quality-first to Balanced;
   - Balanced to Streaming-first.
4. UI shows current mode, reason and measured p95.
5. Service can recover upward only after a stable low-latency window and explicit policy allowance.

### Debug Capture

1. User explicitly enables debug capture.
2. UI shows a persistent warning state and the output path.
3. Service writes bounded debug artifacts only while enabled.
4. Disabling debug capture closes files and returns to no-persistence mode.
5. Debug artifacts are never enabled by default or hidden behind normal logging.

## MVP Boundary

MVP includes:

- Rust daemon scaffold with local IPC server.
- PipeWire/Pulse endpoint lifecycle through Pulse-compatible modules and graph inspection.
- Per-app routing watcher for Telegram Desktop and Firefox/Chromium Meet.
- Python ML sidecar with local provider adapter.
- Local ASR/MT/TTS chain using GPU where supported and CPU fallback where appropriate.
- Manual direction switches for microphone and speaker directions.
- Quality-first, Balanced and Streaming-first latency modes with automatic degradation.
- Tauri tray/menu and local status page.
- Male/female voice preset per target language.
- No-persistence default with explicit debug capture.
- Synthetic loopback latency benchmark.
- Open-speaker AEC probe and acoustic-loop benchmark on this workstation.
- Telegram Desktop and Firefox/Chromium Meet routing smoke.
- MVP-B OpenAI Realtime Translation adapter as the second provider after the local provider.

MVP-A local provider gate:

- local provider contract implemented;
- synthetic duplex loopback passes with local ASR/MT/TTS;
- Telegram Desktop and Firefox/Chromium Meet local-provider smokes pass;
- the local provider receives a latency classification of `meets_target`, `usable_degraded` or `fails_usable_limit`;
- no cloud provider is required to satisfy MVP-A.

MVP-B OpenAI adapter gate:

- starts only after MVP-A functional, routing and privacy gates pass; a local latency miss does not deadlock the comparison/fallback task;
- adds OpenAI Realtime Translation as a second provider adapter;
- proves cloud opt-in, credential handling, audio-egress status and provider-comparison latency;
- does not replace the local-provider acceptance gate.

Usable-release gate:

- at least one enabled provider is `meets_target` or `usable_degraded` in both directions;
- `meets_target`: Streaming-first `speech_onset_to_first_audible_ms` p95 <= 1000 ms;
- `usable_degraded`: p95 > 1000 ms and <= 1500 ms;
- `fails_usable_limit`: p95 > 1500 ms, corpus quality below threshold or unstable duplex resource use;
- if no provider meets the usable limit, the result is a measured prototype, not a usable MVP release;
- headphones pass full-duplex acceptance, and open-speaker mode is enabled only if AEC acceptance also passes.

Post-MVP:

- Zoom acceptance after routing watcher is stable.
- Soniox and Gemini adapters.
- Native PipeWire/Rust low-level audio worker if measured routing overhead requires it.
- Shared-memory IPC if localhost streaming becomes the latency bottleneck.
- Wider app catalog and per-profile routing rules.
- Voice preview and larger voice catalog.
- Offline packaging of model bundles.

## Latency Requirements

The service starts in Quality-first mode, but the product is only useful if latency is visible and bounded.

Latency mode is per direction, not global. A slow incoming direction must not force outgoing microphone translation to degrade unless its own metrics exceed thresholds.

Primary metrics:

- `speech_onset_to_first_audible_ms`;
- `capture_to_first_audio_ms`;
- `capture_to_last_audio_ms`;
- `asr_first_text_ms`;
- `asr_final_text_ms`;
- `mt_first_text_ms`;
- `tts_first_audio_ms`;
- `queue_lag_ms`;
- `routing_playback_ms`;
- `provider_total_ms`;
- `mode_degradation_count`.

Minimum `LatencyPolicy` contract:

```text
LatencyPolicy {
  direction_id: "microphone" | "speaker";
  current_mode: "quality_first" | "balanced" | "streaming_first";
  rolling_window_seconds: 60;
  minimum_samples: 20;
  degrade_after_consecutive_windows: 2;
  recover_after_consecutive_windows: 5;
  cooldown_seconds_after_change: 120;
  p95_first_audio_ms: number;
  p95_last_audio_ms: number;
  p95_queue_lag_ms: number;
}
```

Initial policy:

| Mode | Target | Behavior |
| --- | --- | --- |
| Quality-first | `capture_to_first_audio_ms` p95 <= 3000 ms and `queue_lag_ms` p95 <= 500 ms | Prefer stable phrase-level translation and smoother speech. |
| Balanced | `capture_to_first_audio_ms` p95 <= 2000 ms and `queue_lag_ms` p95 <= 350 ms | Commit shorter stable chunks and reduce beam/quality settings. |
| Streaming-first | `capture_to_first_audio_ms` p95 <= 1000 ms target and `queue_lag_ms` p95 <= 250 ms best effort | Commit partial chunks aggressively; accept less polished phrasing. |

`capture_to_first_audio_ms` starts at the VAD-confirmed source speech onset, not at an arbitrary later provider batch. The independent benchmark metric `speech_onset_to_first_audible_ms` is measured outside the provider pipeline:

- outgoing: injected onset at the physical-mic test source to the first translated sample observed from `Translator_Virtual_Mic`;
- incoming: injected onset at the selected app test sink-input to the first translated sample observed at the physical-sink test tap.

The local service metric excludes meeting-platform and Internet transport. Real-app reports must list meeting/network delay separately when a receiving-side recording is available.

The reproducible benchmark uses 10 warmup utterances that are excluded, then at least 100 measured utterances per direction. Percentiles are calculated over successful utterances, while timeout/drop rate is reported separately and must stay below 1%. Corpus coverage includes short turns, long turns, negation, numbers, names and overlapping duplex speech.

The quality floor uses a versioned corpus with reference translations: corpus chrF2 >= 45 in each direction, no critical negation/number/name corruption in the reviewed critical subset, and synthesized-speech proxy WER <= 15% between the actual translated text sent to TTS and the transcript produced by an already-local alternate ASR. Threshold or semantic-oracle changes require a corpus revision and evidence.

A fast tripwire avoids waiting for 60-second p95 windows: degrade one step after three consecutive utterances exceed the current mode's first-audio limit, or when queue lag exceeds 500 ms continuously for two seconds. Recovery still requires hysteresis, stable windows and cooldown. Tests must inject latency per direction and prove only the affected direction changes mode.

## Audio Routing Invariants

- The daemon captures the pinned physical microphone source for outgoing audio and must never capture `Translator_Virtual_Mic` as its microphone input.
- The daemon captures `Translator_Remote_In.monitor` for incoming selected app audio and must never capture the physical default sink monitor in normal mode.
- Production and manual routing never move translator-owned sink-inputs, TTS playback streams, virtual mic monitor streams or debug playback streams to `Translator_Remote_In`. The authenticated human round-trip controller may authorize only its exact session-bound VirtualPeer identity; metadata alone cannot grant this capability.
- Translated speaker output is played only to a validated physical sink and must not re-enter `Translator_Remote_In.monitor`.
- Headphone mode captures the validated physical mic directly.
- Open-speaker mode routes translated incoming playback through a PipeWire WebRTC echo-cancel reference sink and captures the corresponding echo-cancelled source for outgoing translation.
- Open-speaker mode stays disabled if the AEC module, reference routing or acoustic attenuation acceptance fails. UI shows the reason and requires headphones; a warning alone cannot enable the unsafe path.
- AEC acceptance records geometry and volume, plays a 30-second far-end-only speech fixture at `-20 dBFS`, and requires median ERLE >= 15 dB plus zero outgoing VAD/translation triggers during a 60-second far-end-only run.
- If the default source/sink changes during a call, the device watcher validates the new node before switching. Virtual translator endpoints are ignored as candidate physical devices.
- If a pinned physical mic or sink disappears, the affected direction enters `device_unavailable` and resumes only after a valid physical replacement is selected or the original device returns.

## Provider IPC Requirements

The Rust daemon and Python sidecar communicate through local gRPC streaming over a user-owned Unix-domain socket.

Minimum contract:

- PCM input frames are little-endian signed 16-bit mono at 16 kHz unless a provider declares a different negotiated format.
- Frame duration is 20 ms for internal routing and may be batched to 100 ms for providers that require larger chunks.
- Every frame carries `direction_id`, `stream_id`, `utterance_id`, monotonic capture timestamp and sequence number.
- Provider output carries translated PCM frames, text deltas, final text events, provider latency events and health events.
- Backpressure is explicit: queues are bounded by buffered milliseconds, expose `queue_lag_ms`, and drop or cancel according to mode-specific policy rather than growing unbounded.
- Maximum source-audio age is 3000 ms in Quality-first, 2000 ms in Balanced and 1000 ms in Streaming-first; older work is cancelled and never played.
- Cancellation is explicit per direction and per utterance.
- Every provider input and output event carries `session_id`; daemon rejects stale events from closed or superseded sessions.
- Provider errors are typed and do not include raw spoken content.
- Contract tests use synthetic PCM fixtures and injected provider delays.

## Privacy And Security Requirements

- Default runtime must not persist raw audio, transcripts or translated text.
- Logs must not include spoken content, transcript text, translation text, API keys or raw provider payloads.
- Debug capture is off by default, visible when enabled, written with user-only permissions and bounded to the lower of 10 minutes or 500 MiB per explicit session.
- Debug files live under a dedicated user state directory with directory mode `0700`; creation rejects symlinks, uses file mode `0600`, and stops before free space falls below 5 GiB.
- Local provider is the default provider.
- Cloud adapters are disabled until explicit per-provider user enablement.
- Cloud mode cannot start silently. UI must show that audio leaves the machine before the first cloud session starts.
- Cloud provider credentials are stored in local ignored configuration or OS secret storage, not in UI state or logs.
- Tauri UI must never expose raw audio buffers or provider keys to frontend state.
- Localhost IPC binds only to loopback and uses a random session token or equivalent local authorization.
- Provider adapters must mark whether audio leaves the machine.
- Status page must show local/cloud provider mode clearly.

## UX Requirements

- First screen is the working status page, not a landing page.
- Tray/menu gives quick access to:
  - start/stop translation;
  - microphone direction;
  - speaker direction;
  - provider;
  - latency mode;
  - voice gender per target language;
  - debug capture toggle.
- Status page shows:
  - active physical mic/sink;
  - virtual endpoints;
  - routed app streams;
  - current provider and local/cloud marker;
  - current mode and degradation reason;
  - latency p50/p95 per direction;
  - sidecar health;
  - privacy/debug state.
- Manual stream override is available for unmatched app audio streams.
- Every loading, failed, degraded, disconnected and healthy state must be visible.
- UI must use dense desktop-service layout: tables, compact status blocks, toggles and controls.
- No card nesting.

## Acceptance Criteria

- `systemctl --user start translator.service` starts the daemon in the user's PipeWire session.
- Tauri UI can reconnect to an already running daemon.
- Virtual endpoints are created and visible through `pactl list short sinks` and `pactl list short sources`.
- The service does not capture the physical default sink monitor in normal incoming mode.
- Synthetic loopback proves both directions can run concurrently without audio feedback.
- Open-speaker mode cannot start until PipeWire WebRTC AEC passes reference-routing and acoustic attenuation checks; headphones remain the required fallback.
- Routing tests prove production/manual profiles never move translator-owned streams to `Translator_Remote_In`; forged self-test metadata is rejected, and only the daemon-authorized VirtualPeer tuple is accepted during its matching self-test session.
- Routing tests prove the daemon never captures `Translator_Virtual_Mic` or the physical default sink monitor in normal mode.
- Device watcher tests prove default source/sink changes do not select virtual translator endpoints as physical devices.
- Outgoing smoke proves physical mic input reaches `Translator_Virtual_Mic` only after translation.
- Incoming smoke proves selected app output is routed to `Translator_Remote_In`, translated and played to the physical sink.
- In headphone mode, the live round-trip self-test proves that one Russian microphone utterance produces a complete audible English virtual-peer tap before the exact English PCM is reinjected, translated back and heard in Russian without recursive capture.
- Round-trip teardown restores the original graph/routes; recursion count is zero, and the test reports outgoing-leg, incoming-leg and total `physical_mic_onset_to_returned_ru_first_audible_ms`.
- Telegram Desktop and Firefox/Chromium Meet are validated after synthetic loopback.
- Quality-first mode is the initial mode.
- Latency policy degrades only the affected direction when p95 thresholds are exceeded.
- Latency policy uses minimum sample count, rolling windows, cooldown and hysteresis.
- Fast latency tripwire degrades after consecutive utterance or sustained queue-lag breaches without waiting for the rolling window.
- Synthetic benchmark measures external `speech_onset_to_first_audible_ms` in both directions after warmup.
- Benchmark uses at least 100 measured utterances per direction, reports drops separately and validates the corpus quality floor.
- Male and female output presets for Russian and English are available before MVP-A closes; no silent gender fallback is allowed.
- Debug capture is off by default and visibly marked when enabled.
- Logs contain latency and technical state but no spoken content.
- MVP-A functional/routing/privacy gates pass with local provider before OpenAI work starts; a measured local latency miss permits MVP-B comparison instead of blocking it.
- OpenAI Realtime Translation adapter is available in MVP-B after the local provider adapter, not before it.
- Cloud provider mode requires explicit enablement and cannot start silently.
- Usable release requires at least one provider to stay within p95 <= 1500 ms in both directions; p95 <= 1000 ms remains the target.

## Validation Gates

- Static checks for Rust daemon, Python sidecar and Tauri UI.
- Unit tests for routing policy, app allowlist, direction state, latency degradation and privacy logging.
- Provider contract tests with synthetic PCM fixtures.
- Latency policy tests with injected per-direction delays.
- Cloud egress/privacy tests proving local default and explicit cloud opt-in.
- Device watcher tests for unplug/replug and default source/sink changes.
- Audio graph smoke:
  - `pactl info`;
  - `wpctl status`;
  - `pactl list short sinks`;
  - `pactl list short sources`;
  - `pw-link -l`;
  - `parecord -d translator_remote_in.monitor`;
  - `paplay --device=translator_mic_out`.
- Synthetic duplex latency benchmark with deterministic audio fixtures.
- Live human round-trip self-test on this workstation with an English monitor tap, returned Russian audio, checkpoint/latency evidence and teardown verification.
- Warm and cold runs, at least 30 minutes of simultaneous duplex soak, GPU/CPU/RAM/VRAM telemetry and OOM/restart evidence.
- Open-speaker AEC reference-routing and acoustic attenuation benchmark.
- Telegram Desktop routing smoke.
- Firefox/Chromium Meet routing smoke.
- At least one two-endpoint Telegram or Meet smoke with simultaneous overlapping speech, receiving-side audio evidence and both-direction latency.
- Tauri status-page screenshot and interaction checks.

## Decisions Left To Bounded Implementation Spikes

- Task 6 selects one Ru <-> En MT candidate from a recorded size/quality/latency matrix; only the selected model may be downloaded.
- Piper is the bootstrap TTS engine. Another engine requires measured failure against latency, quality or required voice coverage.
- OpenAI uses one independent translation session per direction.
