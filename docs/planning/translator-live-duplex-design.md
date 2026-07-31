# Translator Live Duplex Design

Status: audited implementation plan.

Design source:

- PRD: `translator-live-duplex-prd.md`
- Workstation: Ubuntu 24.04.4, PipeWire 1.0.5 through PulseAudio compatibility, NVIDIA RTX 4080 Laptop GPU 12 GB.

## Current Context

`/home/anton/Source/translator` is an empty project directory. This design defines the initial architecture and contracts for a new desktop service.

Current host facts:

```text
Default source:
alsa_input.usb-Jieli_Technology_UACDemoV1.0-00.mono-fallback

Default sink:
alsa_output.usb-Jieli_Technology_UACDemoV1.0-00.analog-stereo

Audio server:
PulseAudio compatibility on PipeWire 1.0.5

GPU:
NVIDIA GeForce RTX 4080 Laptop GPU, 12 GB VRAM
```

Relevant external anchors:

- PipeWire Pulse-compatible modules provide `module-null-sink` and `module-remap-source`.
- `pw-link` exposes graph inspection and port linking.
- `faster-whisper` uses CTranslate2 and supports CUDA execution with `device="cuda"` and `compute_type="float16"` or `int8_float16`.
- Piper is a local neural TTS candidate.
- OpenAI Realtime Translation exposes a dedicated realtime translation endpoint after local MVP-A.

## Local Model Inventory

Snapshot: 2026-07-28.

The implementation must reuse these local assets before downloading new speech models:

| Capability | Local Asset | Path | Size | Reuse Decision |
| --- | --- | --- | --- | --- |
| ASR | `Systran/faster-whisper-small` CTranslate2 cache | `/home/anton/.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120` | ~464 MB | Good first low-latency ASR smoke candidate. |
| ASR | OpenAI Whisper `small.pt` | `/home/anton/.cache/whisper/small.pt` | ~461 MB | Fallback for non-CTranslate2 tests only. Prefer faster-whisper cache for MVP. |
| ASR | OpenAI Whisper `base.pt` | `/home/anton/scripts/simu-lator/ai-engine/models/whisper/base.pt` | ~139 MB | Very small smoke fallback; quality likely weaker. |
| ASR | `Systran/faster-whisper-large-v3` CTranslate2 cache | `/home/anton/Source/uncle-freud-bot/.data/faster-whisper/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478` | ~2.9 GB | Quality-first ASR candidate after small-model smoke. |
| TTS | Piper Russian voice `ru_RU-dmitri-medium` | `/home/anton/Source/uncle-freud-bot/.data/piper-voices/ru_RU-dmitri-medium.onnx` | ~60 MB | Reuse for Russian male preset/bootstrap. |
| TTS | Piper English voice `en_US-ryan-medium` | `/home/anton/Source/uncle-freud-bot/.data/piper-voices/en_US-ryan-medium.onnx` | ~60 MB | Reuse for English male preset/bootstrap. |
| TTS | Qwen3 TTS CustomVoice | `/home/anton/Source/uncle-freud-bot/.data/qwen-tts/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe` | ~4.3 GB | Do not use for MVP unless Piper fails; heavy and custom-voice oriented. |

No local Ru <-> En MT model was found in the searched caches. The implementation task must benchmark or select a minimal MT model deliberately instead of downloading several candidates.

New-model policy:

- `models/manifest.json` records source, exact revision, SHA-256, license, languages, byte size and cache path;
- total new model downloads for MVP are capped at 2 GiB unless the plan is explicitly amended;
- a download cannot start unless at least 20 GiB will remain free afterward;
- normal simultaneous-duplex inference must leave at least 2 GiB VRAM headroom on the 12 GiB GPU, so measured peak is capped at 10 GiB;
- Russian and English female Piper-compatible voices are required before MVP-A closes because only male voices are currently local.

`uncle-freud-bot` already contains useful local-voice integration code and tests for faster-whisper and Piper. Treat it as a reference implementation, not a dependency boundary.

## Target Architecture

```text
                         Tauri Control UI
                  tray/menu + local status page
                               |
                         localhost IPC
                               |
                        Rust Translator Daemon
         audio graph, routing watcher, latency ledger, supervision
               |                         |                  |
       PipeWire/Pulse graph       Provider IPC        systemd --user
               |                         |
               |                 Python ML Sidecar
               |           local ASR -> MT -> TTS adapter
               |
     virtual endpoints + physical devices
```

The daemon is the only component that owns desktop audio routing. The Python sidecar is replaceable and only sees audio frames plus direction/provider configuration. Tauri is a control plane and must not receive raw audio frames.

## Component Boundaries

| Component | Runtime | Owns |
| --- | --- | --- |
| `translator-daemon` | Rust | PipeWire/Pulse endpoint lifecycle, stream routing, capture/playback, provider IPC client/server, latency policy, privacy-safe logs, sidecar supervision |
| `translator-sidecar` | Python 3.12 | Local ASR, MT, TTS, local model loading, provider health/latency events; consumes capture-owned EOU markers as authoritative input |
| `translator-ui` | Tauri 2 + TypeScript frontend | Tray/menu, status page, manual override, direction/provider/mode/voice/debug controls |
| `translator.service` | systemd user unit | Starts daemon in the user audio session and restarts it on failure |
| `tests/audio-fixtures` | deterministic fixtures | Synthetic PCM input, expected routing events, latency injection, privacy markers |

## Process Model

```text
systemd --user
  -> translator-daemon
       -> creates virtual audio endpoints
       -> starts translator-sidecar
       -> exposes localhost control/status API
       -> watches PipeWire/Pulse streams

translator-ui
  -> connects to daemon over localhost
  -> subscribes to status events
  -> sends control commands
```

Tauri can start the daemon in dev mode, but MVP readiness is based on `systemctl --user start translator.service`.
The terminal lifecycle command is `scripts/translator-desktop`: use
`scripts/translator-desktop up` to install the user unit if needed and start the
daemon, `scripts/translator-desktop down` to stop it and run owned audio-graph
cleanup, `scripts/translator-desktop restart` to restart with cleanup, and
`scripts/translator-desktop logs [lines]` to inspect the user-unit journal.
`scripts/translator-desktop install|up` also installs the user-local
`~/.local/bin/translator` command, so the same lifecycle is available from any
current directory as `translator up`, `translator down`, `translator restart`
and `translator logs [lines]`. `translator up`, `translator start` and
`translator restart` wait briefly for the daemon HTTP API, then start the
current-session `translator-ui` tray/status process when a graphical desktop
session is available; the XDG autostart entry still covers login-time UI launch.
When `target/release/translator-daemon` or `target/release/translator-ui`
exists, the lifecycle command refreshes the matching `~/.local/bin` binary
before launch, so rebuilt release assets replace stale installed binaries.

The production user unit uses `RuntimeDirectory=translator`, `RuntimeDirectoryMode=0700`, `KillMode=control-group` and bounded `Restart=on-failure`. The sidecar remains in the unit cgroup. A daemon crash must not leave an orphan sidecar, stale authenticated socket or reusable control token.

Daemon restart behavior:

- recreate missing virtual endpoints;
- reconnect sidecar;
- keep debug capture disabled unless explicitly re-enabled after restart;
- report stale UI state until fresh audio graph inspection completes.
- recover only daemon-owned stale endpoints from the runtime ownership journal; restore selected app streams to their recorded original sinks before unloading endpoints.

## Audio Graph

### Virtual Endpoints

The daemon creates three Pulse-compatible endpoints through PipeWire's PulseAudio compatibility layer:

```bash
pactl load-module module-null-sink \
  sink_name=translator_mic_out \
  rate=48000 channels=1 channel_map=mono \
  sink_properties=device.description=Translator_Mic_Out

pactl load-module module-remap-source \
  master=translator_mic_out.monitor \
  source_name=translator_virtual_mic \
  channels=1 channel_map=mono remix=no \
  source_properties=device.description=Translator_Virtual_Mic

pactl load-module module-null-sink \
  sink_name=translator_remote_in \
  rate=48000 channels=2 channel_map=front-left,front-right \
  sink_properties=device.description=Translator_Remote_In
```

Implementation should use Rust process calls or libpulse APIs behind an `AudioGraph` trait. Shell commands are allowed for MVP if the daemon records module ids and unloads only modules it created.

### Outgoing Direction

```text
Physical mic source
  -> daemon capture stream
  -> resample/downmix to provider input format
  -> provider direction: microphone
  -> translated PCM
  -> daemon playback to translator_mic_out
  -> translator_virtual_mic
  -> selected call app microphone
```

The daemon must pin the physical source by node/source name. It must reject `translator_virtual_mic`, `translator_mic_out.monitor`, `translator_remote_in.monitor` and any source with the translator-owned property marker as capture candidates.

### Incoming Direction

```text
Zoom/Meet/Telegram sink-input
  -> routing watcher moves sink-input to translator_remote_in
  -> daemon capture from translator_remote_in.monitor
  -> resample/downmix to provider input format
  -> provider direction: speaker
  -> translated PCM
  -> daemon playback to validated physical sink/headphones
```

The daemon must not capture the physical default sink monitor in normal mode. It may expose a manual diagnostic capture mode, but that mode is out of the MVP production path and must show a feedback-risk warning.

### Human Round-Trip Self-Test

The self-test exercises both production directions without a meeting app:

```text
physical mic
  -> outgoing Ru -> En provider direction
  -> translator_mic_out
  -> translator_virtual_mic
  -> VirtualPeer exact-PCM capture
       |-> English monitor tap -> validated physical headphones
       -> session-bound VirtualPeer sink-input
  -> routing watcher selects that exact sink-input
  -> translator_remote_in
  -> incoming En -> Ru provider direction
  -> returned Russian PCM -> validated physical headphones
```

The VirtualPeer must reinject the exact PCM frames captured from `translator_virtual_mic`, not regenerated text or a second TTS result. Its sink-input carries `translator.test_profile=human_round_trip`, but metadata is descriptive and grants no authority. The daemon-held self-test capability authorizes one exact tuple `{session_id, stream_serial_or_node_id, process_identity}` while the matching session is active. Production routing, manual override and forged or stale self-test metadata remain subject to the translator-owned-stream rejection.

The self-test is fail-closed unless headphones, both provider directions, the virtual graph and an idle incoming route are available. It cannot run with a real selected app route or open-speaker mode. One session may run at a time and is bounded to five minutes. The complete English monitor tap plays first; reinjection starts only after that tap finishes, and returned Russian playback follows the incoming translation. These outputs target only the validated physical headphones and cannot be routed into `translator_remote_in`.

Exact-PCM evidence stays in memory: capture and reinjection record the same PCM format, frame count, monotonic frame sequence and rolling frame hash. The hash and counters may enter the technical self-test report; PCM frames must not be persisted by this proof.

Typed checkpoints are `waiting_for_speech`, `outgoing_vad`, `outgoing_asr_final`, `outgoing_translation_final`, `english_first_audio`, `virtual_peer_reinjecting`, `incoming_asr_final`, `incoming_translation_final`, `russian_first_audio`, `completed`, `failed` and `stopped`. Text values remain behind `debug_text`; normal status exposes only stages, safe errors and latency. Metrics include each production leg plus `physical_mic_onset_to_returned_ru_first_audible_ms`.

Stop, timeout, daemon restart or any failed precondition performs idempotent teardown: stop capture/injection/playback workers, restore the prior route, remove temporary streams, clear bounded memory-only state and verify that no self-test stream remains. A recursive frame or a second pass through either direction fails the test.

### Headphone And Open-Speaker Modes

Headphone mode is the baseline:

```text
physical mic -> daemon outgoing capture
translated incoming PCM -> validated physical headphones
```

Open-speaker mode requires PipeWire WebRTC AEC, available on this workstation as `libpipewire-module-echo-cancel.so` with `libspa-aec-webrtc.so`:

```text
physical mic -----------------------> AEC capture
translated incoming PCM -> AEC reference sink -> physical speakers
                                      |
                                      -> echo-cancelled source -> daemon outgoing capture
```

The daemon must verify that the translated incoming playback is the AEC reference signal and keep open-speaker mode disabled if setup or attenuation validation fails. The direct physical mic remains valid only for headphone mode.

AEC benchmark:

- record physical source/sink names, speaker-mic distance/orientation and system volume;
- play a 30-second far-end-only speech fixture at `-20 dBFS` through the translated-playback reference path;
- capture the physical mic once without AEC and the residual echo once with AEC;
- calculate `ERLE_dB = 10 * log10(P_without_aec / P_with_aec)` over active fixture windows;
- pass only when median ERLE >= 15 dB and a separate 60-second far-end-only run produces zero outgoing VAD/translation triggers;
- any failed setup or threshold keeps open-speaker mode unavailable for that device pair.

## Routing Watcher

The routing watcher observes Pulse/PipeWire sink-inputs and source-outputs.

Allowlist defaults:

- Telegram Desktop;
- Firefox;
- Chromium;
- Chrome;
- Zoom only after the post-MVP Zoom acceptance task starts.

Match attributes:

- application name;
- binary/process name;
- media role;
- stream description;
- current sink;
- user manual override id.

Routing rules:

- MVP supports one active incoming route for the `speaker` direction.
- Allowlist matches discover route candidates; they do not blindly move every matching stream.
- Move only the selected candidate sink-input to `translator_remote_in`.
- If multiple allowlisted browser/app streams are active, keep new candidates unrouted until active-call heuristics choose one or the user confirms a manual override.
- Production and manual routing never move translator-owned sink-inputs to `translator_remote_in`.
- Never move sidecar, daemon, Tauri preview, debug playback or `paplay` validation streams unless explicitly selected in a test profile.
- The authenticated human round-trip controller may select only the VirtualPeer matching its daemon-held `{session_id, stream_serial_or_node_id, process_identity}` capability. A property marker alone cannot bypass rejection; any other stream already on `translator_remote_in` is a route conflict.
- Preserve current routing when the app restarts by reapplying allowlist rules to new sink-input ids.
- If a stream disappears, close its capture state and emit `route_removed`.

Candidate selection:

- If there is exactly one allowlisted sink-input with call-like metadata, route it.
- If a previously selected app restarts and produces one replacement sink-input, route the replacement.
- If there are multiple allowlisted candidates, require manual selection unless implementation adds a tested active-call heuristic.
- If the selected candidate becomes silent or disappears, do not auto-route a different candidate without selection.

Translator-owned stream detection:

- module names: `translator_mic_out`, `translator_remote_in`;
- app/process names controlled by daemon/sidecar/Tauri;
- stream property marker: `translator.owner=true`;
- sink/source names beginning `translator_`;
- node descriptions `Translator_*`.

## Device Watcher

The device watcher tracks physical source and sink availability.

Source behavior:

- initial source is the current physical default source unless configured otherwise;
- virtual translator sources are excluded;
- if the pinned physical source disappears, outgoing direction enters `device_unavailable`;
- if it returns with the same stable name, capture resumes;
- if default source changes to another physical source, UI prompts or policy validates before switching.

Sink behavior:

- translated speaker output targets the current validated physical sink;
- virtual translator sinks are excluded as physical playback candidates;
- sink changes are applied only after a successful short graph validation;
- if no physical sink is available, incoming direction pauses with `device_unavailable`.

Acoustic warning:

- headphones are recommended;
- open-speaker mode is disabled until AEC is available and validated;
- if the active sink appears to be speakers while AEC is unavailable or failed, translation cannot start in full-duplex mode and UI requires headphones.

## Provider Contract

Provider transport is gRPC bidirectional streaming over a user-owned Unix-domain socket. Protobuf owns wire schemas and binary PCM framing. The daemon and sidecar still keep bounded application queues because transport flow control is not the latency policy.

Sidecar auth:

- daemon generates a random sidecar session token at sidecar start;
- token is passed to the sidecar through inherited environment or an inherited file descriptor, not through frontend state;
- every sidecar connection must present the token;
- sidecar binds to loopback or a Unix-domain socket only;
- Tauri never connects directly to the sidecar.
- socket path is under `%t/translator/sidecar.sock`, directory mode `0700`, socket mode `0600`;
- daemon removes only a stale socket that it owns after proving no live sidecar accepts the current handshake.

### Session Lifecycle

```typescript
type SampleFormat = "s16le";

type PcmFormat = {
  sample_rate_hz: 16000 | 24000 | 48000;
  channels: 1 | 2;
  sample_format: SampleFormat;
  frame_duration_ms: 20 | 40 | 60 | 80 | 100;
};

type VoiceProfile = {
  language: "ru" | "en";
  gender: "male" | "female";
  engine: "piper" | "silero" | "openai";
  model_path?: string;
  provider_voice_id?: string;
};

type OpenProviderSession = {
  schema_version: "translator.provider.open_session.v1";
  session_id: string;
  direction_id: AudioDirection;
  source_language: "ru" | "en";
  target_language: "ru" | "en";
  mode: TranslationMode;
  requested_input_format: PcmFormat;
  requested_output_format: PcmFormat;
  voice_profile: VoiceProfile;
  debug_text_enabled: boolean;
};

type ProviderSessionOpened = {
  schema_version: "translator.provider.session_opened.v1";
  session_id: string;
  direction_id: AudioDirection;
  negotiated_input_format: PcmFormat;
  negotiated_output_format: PcmFormat;
  capabilities: {
    audio_output: true;
    transcript_delta: boolean;
    translation_delta: boolean;
    cancellation: boolean;
    cloud_egress: boolean;
  };
};

type CloseProviderSession = {
  schema_version: "translator.provider.close_session.v1";
  session_id: string;
  reason: "user_stop" | "route_removed" | "device_unavailable" | "provider_switch" | "daemon_shutdown";
};

type CancelUtterance = {
  schema_version: "translator.provider.cancel_utterance.v1";
  session_id: string;
  direction_id: AudioDirection;
  utterance_id: string;
  reason: "latency_policy" | "route_removed" | "user_interrupt" | "queue_overflow";
};
```

Lifecycle:

```text
daemon -> sidecar: open_session
sidecar -> daemon: session_opened or provider_error
daemon -> sidecar: audio_frame*
sidecar -> daemon: audio_delta* + latency* + health*
daemon -> sidecar: cancel_utterance | close_session
sidecar -> daemon: utterance_final | session_closed
```

### Input Frame

```typescript
type AudioDirection = "microphone" | "speaker";
type TranslationMode = "quality_first" | "balanced" | "streaming_first";

type ProviderInputFrame = {
  schema_version: "translator.provider.input.v1";
  session_id: string;
  direction_id: AudioDirection;
  stream_id: string;
  utterance_id: string;
  sequence: number;
  capture_monotonic_ns: string;
  sample_rate_hz: 16000 | 24000 | 48000;
  channels: 1 | 2;
  sample_format: "s16le";
  frame_duration_ms: 20 | 40 | 60 | 80 | 100;
  source_language: "ru" | "en";
  target_language: "ru" | "en";
  mode: TranslationMode;
  pcm: bytes;
  end_of_utterance: boolean;
};
```

Frame format must equal the session-negotiated input format. The daemon may capture at 48 kHz and convert to 16 kHz mono before provider input.
`end_of_utterance` is explicit so a multi-frame utterance produces exactly one terminal event.

### Provider Events

```typescript
type ProviderEvent =
  | ProviderAudioDelta
  | ProviderTranscriptDelta
  | ProviderTranslationDelta
  | ProviderUtteranceFinal
  | ProviderSessionClosed
  | ProviderHealth
  | ProviderLatency
  | ProviderError;

type ProviderAudioDelta = {
  schema_version: "translator.provider.audio_delta.v1";
  session_id: string;
  direction_id: AudioDirection;
  stream_id: string;
  utterance_id: string;
  sequence: number;
  event_sequence: number;
  provider_monotonic_ns: string;
  sample_rate_hz: 16000 | 24000 | 48000;
  channels: 1 | 2;
  sample_format: "s16le";
  pcm: bytes;
};

type ProviderTranscriptDelta = {
  schema_version: "translator.provider.transcript_delta.v1";
  session_id: string;
  direction_id: AudioDirection;
  stream_id: string;
  utterance_id: string;
  event_sequence: number;
  text: string;
  is_final: boolean;
};

type ProviderTranslationDelta = {
  schema_version: "translator.provider.translation_delta.v1";
  session_id: string;
  direction_id: AudioDirection;
  stream_id: string;
  utterance_id: string;
  event_sequence: number;
  text: string;
  stable_prefix: boolean;
  is_final: boolean;
};

type ProviderUtteranceFinal = {
  schema_version: "translator.provider.utterance_final.v1";
  session_id: string;
  direction_id: AudioDirection;
  stream_id: string;
  utterance_id: string;
  event_sequence: number;
  final_audio_sequence?: number;
  outcome: "completed" | "cancelled" | "dropped";
};

type ProviderSessionClosed = {
  schema_version: "translator.provider.session_closed.v1";
  session_id: string;
  direction_id: AudioDirection;
  event_sequence: number;
  reason:
    | "user_stop"
    | "route_removed"
    | "device_unavailable"
    | "provider_switch"
    | "daemon_shutdown"
    | "provider_failure"
    | "close_timeout";
};

type ProviderHealth = {
  schema_version: "translator.provider.health.v1";
  session_id: string;
  direction_id: AudioDirection;
  event_sequence: number;
  provider_id: "local" | "openai";
  provider_name: string;
  state:
    | "starting"
    | "ready"
    | "degraded"
    | "backpressure"
    | "restarting"
    | "unavailable"
    | "closed";
  models: Array<{
    kind: "asr" | "mt" | "tts" | "speech_to_speech";
    id: string;
    state: "not_loaded" | "loading" | "ready" | "failed";
    device?: "cuda" | "cpu" | "cloud";
    safe_error_code?: string;
  }>;
  queues: {
    provider_input_buffered_ms: number;
    provider_output_buffered_ms: number;
    queue_lag_ms: number;
  };
  retry?: {
    attempt: number;
    next_retry_after_ms: number;
    reason_code: string;
  };
  safe_error?: {
    code: string;
    message: string;
    retryable: boolean;
  };
};

type DaemonQueueHealth = {
  direction_id: AudioDirection;
  capture_buffered_ms: number;
  playback_buffered_ms: number;
  dropped_frames: number;
  queue_lag_ms: number;
};

type ProviderLatency = {
  schema_version: "translator.provider.latency.v1";
  session_id: string;
  direction_id: AudioDirection;
  stream_id: string;
  event_sequence: number;
  utterance_id?: string;
  asr_first_text_ms?: number;
  asr_final_text_ms?: number;
  mt_first_text_ms?: number;
  tts_first_audio_ms?: number;
  provider_total_ms?: number;
};

type ProviderError = {
  schema_version: "translator.provider.error.v1";
  session_id: string;
  direction_id: AudioDirection;
  stream_id?: string;
  utterance_id?: string;
  event_sequence: number;
  code:
    | "provider_unavailable"
    | "model_not_loaded"
    | "unsupported_language_pair"
    | "queue_overflow"
    | "cancelled"
    | "no_speech"
    | "cloud_not_enabled"
    | "provider_auth_failed";
  retryable: boolean;
  safe_message: string;
};
```

`final_audio_sequence` is absent when no audio delta was emitted for the
utterance; otherwise it identifies the last emitted audio delta.

Every dropped utterance emits `ProviderLatency`, `ProviderError` and
`ProviderUtteranceFinal` in that order. An empty ASR result is an
utterance-level `no_speech` error: ASR latency remains observable, MT and TTS
are not invoked, and model health is not changed to failed.

Every provider event, including transcript, translation, final and session-closed events, carries `session_id` and a strictly increasing event sequence. A cancelled in-flight publication may leave a forward sequence gap; allocated values are never reused. The daemon discards duplicate, stale, out-of-order-after-final and superseded-session events. `close_session` has a 2-second acknowledgement deadline; after timeout the daemon terminates and restarts the sidecar before opening replacement sessions.

Transcript and translation deltas are disabled in normal status UI. They may be emitted only when `debug_text_enabled=true` for the provider session.

Debug text rules:

- explicit user enablement required;
- visible warning in tray/status page while enabled;
- bounded in-memory ring buffer only: at most 200 events and at most 1 MiB, whichever is reached first;
- cleared on session stop, provider switch, daemon restart and UI close;
- never written to logs, local storage, telemetry, debug capture metadata or provider error messages;
- privacy tests use marker text to prove normal mode does not expose transcript/translation content.

### Backpressure

Queue ownership is explicit:

- Rust daemon owns capture and playback queues;
- Python sidecar owns provider input and provider output queues.

Queue policy is expressed in buffered time, independent of negotiated frame duration:

- capture queue max: 400 ms per direction;
- provider input queue max: 800 ms per direction;
- provider output audio queue max: 1200 ms per direction;
- playback queue max: 400 ms per direction;
- maximum source-audio age: 3000 ms Quality-first, 2000 ms Balanced, 1000 ms Streaming-first;
- Quality-first: prefer waiting within queue threshold, then degrade.
- Balanced: cancel stale utterance tail when queue lag exceeds threshold.
- Streaming-first: drop stale partial chunks and keep latest audio/text context.

Queues must emit `queue_lag_ms` and `queue_dropped_frames` metrics without speech content. Work older than the mode age deadline is cancelled across all queues and provider stages; translated audio that missed its deadline is not played.

### Cancellation

Cancellation scopes:

- direction-level stop;
- stream-level route removal;
- utterance-level interruption;
- provider restart.

Every cancellation emits a typed event and closes the corresponding provider context.

## Local Provider Design

The local provider runs in Python 3.12.

Initial components:

- ASR: start with local `Systran/faster-whisper-small` CTranslate2 cache for smoke, then benchmark local `Systran/faster-whisper-large-v3` for Quality-first.
- MT: `mijuanlo/nllb-200-distilled-600M-ct2-int8@16bc5ff0482f9f1c0d35bdef950721ce58640789`, selected after the local-cache absence gate. The personal-use policy is `personal_noncommercial`; redistribution is disabled.
- TTS: start with local Piper voices `ru_RU-dmitri-medium` and `en_US-ryan-medium`; add missing female presets only after inventory confirms they are absent.
- VAD/EOU: owned by the Rust capture path. `stream_id` remains stable for the provider session, `utterance_id` rotates at each EOU boundary, and the sidecar consumes each capture-issued `end_of_utterance` marker as authoritative input. The sidecar does not maintain an independent segmentation policy.
- Stable-prefix commit: TTS receives only translation prefixes that will not be revised. Superseded partial text is cancelled before synthesis; already played speech is never silently rewritten.

Local provider modes:

| Mode | ASR/MT/TTS behavior |
| --- | --- |
| Quality-first | prefer stable utterance chunks, higher quality settings, smoother TTS |
| Balanced | shorter chunks, lower beam/quality where needed, faster TTS start |
| Streaming-first | aggressive partial commit, minimal beam, faster but less polished output |

The local provider must support two independent simultaneous sessions, one per direction. Shared model instances are allowed only if they do not couple queue state, VAD state or cancellation.

Model residency is selected by measurement. Task 6 records cold start, warm first-audio latency, throughput, GPU/CPU RAM, VRAM peak and two-direction concurrency for `faster-whisper-small` and `faster-whisper-large-v3`. The sidecar must not keep both ASR models resident in normal operation. CUDA OOM causes a typed degradation to the measured smaller model or an explicit unavailable state, not an unbounded restart loop.

Task 6 selected `faster-whisper-small` for normal residency. On this workstation its warm ASR p95 was `92.32 ms` and simultaneous provider first-audio p95 was `569.27 ms` Ru -> En and `683.86 ms` En -> Ru after 10 excluded warmup pairs and 100 measured pairs per direction. The isolated `large-v3` duplex candidate measured `892.80 ms` and `1,063.71 ms` respectively, then was explicitly released before normal `small` residency. The candidates peaked at `4,420 MiB` and `6,052 MiB` total GPU memory used, both below the `10 GiB` gate. These are provider-level measurements; Task 7 owns graph-boundary `speech_onset_to_first_audible_ms` and the final latency classification.

CTranslate2 `4.7.1` on this machine requires CUDA 12 `libcublas.so.12` and cuDNN 9, while the system toolkit is CUDA 13. The user service supplies the already-local compatibility libraries from `/usr/local/lib/ollama/cuda_v12` and `/home/anton/Source/uncle-freud-bot/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib`. Missing compatible libraries cause the existing CPU fallback or explicit unsupported state; runtime startup never downloads them.

Quality corpus:

- 10 excluded warmups followed by at least 100 measured utterances per direction;
- versioned Ru/En references with short/long turns, negation, numbers and names;
- chrF2 >= 45 in each direction;
- critical-subset review has no meaning-changing negation, number or name errors;
- synthesized output is re-transcribed with an already-local alternate ASR and has WER <= 15% against the actual translated text sent to TTS.

## OpenAI Provider Design

OpenAI is MVP-B, not MVP-A.

OpenAI adapter requirements:

- disabled until explicit per-provider enablement;
- requires credential configuration through ignored local config or OS secret storage;
- status page shows that audio leaves the machine;
- one independent translation session per direction;
- provider contract maps OpenAI translated audio and transcript deltas to the same daemon events as local provider;
- no raw provider payloads in logs.

The adapter must use the same latency ledger and privacy gates as the local provider. It is a provider replacement, not a shortcut around routing, debug and mode policy.

## Latency Policy

Latency policy is daemon-owned and per direction.

```typescript
type LatencyPolicyState = {
  direction_id: AudioDirection;
  current_mode: TranslationMode;
  rolling_window_seconds: 60;
  minimum_samples: 20;
  degrade_after_consecutive_windows: 2;
  recover_after_consecutive_windows: 5;
  cooldown_seconds_after_change: 120;
  p95_first_audio_ms: number;
  p95_last_audio_ms: number;
  p95_queue_lag_ms: number;
  last_mode_change_at?: string;
  reason?: string;
};
```

Mode transition:

```text
quality_first -> balanced
balanced -> streaming_first
```

Recovery:

```text
streaming_first -> balanced -> quality_first
```

Fast degradation occurs after three consecutive utterances breach the mode first-audio threshold or queue lag exceeds 500 ms continuously for two seconds. Recovery is allowed only after the configured recovery windows pass and cooldown has elapsed. The UI shows every transition with reason and metrics.

`capture_to_first_audio_ms` begins at VAD-confirmed source onset. `speech_onset_to_first_audible_ms` is measured by the external audio harness at the graph boundaries, not reported by the provider. Benchmark runs include model warmup, cold-start results and clock-domain validation against `CLOCK_MONOTONIC`.

## Privacy Model

Default state:

- local provider selected;
- no cloud adapter enabled;
- no debug capture;
- no transcript persistence;
- no audio persistence.

Cloud egress:

- enabling a cloud provider requires an explicit user action;
- the provider card shows `audio_leaves_machine=true`;
- every cloud session start emits a privacy-safe event;
- disabling the provider closes active cloud sessions.

Debug capture:

- off by default;
- requires explicit enablement from UI or CLI;
- writes only beneath the dedicated user state directory;
- uses `$XDG_STATE_HOME/translator/debug` or its standard user-state fallback, directory mode `0700`;
- opens new files with `O_NOFOLLOW | O_CREAT | O_EXCL`, mode `0600`;
- stops at the lower of 10 minutes or 500 MiB per explicit session, or before free space falls below 5 GiB;
- is marked in tray/status while active;
- stops on daemon restart.

Logs:

- structured technical events;
- latency metrics;
- routing decisions;
- provider state and safe error codes;
- no spoken content, transcript text, translation text, provider raw payloads or API keys.

## Local Control API

The daemon exposes HTTP control plus SSE status events on loopback. The Tauri Rust backend owns the local API token and proxies typed commands/events to the frontend; JavaScript state never receives the token.

The daemon writes a rotating control token to `%t/translator/control.token` with mode `0600`. Tauri Rust rereads it after daemon reconnect or `401`; the frontend receives neither token nor path.

Minimum API:

```text
GET /v1/status
GET /v1/audio-graph
GET /v1/routes
GET /v1/routes/candidates
POST /v1/translation/start
POST /v1/translation/stop
PATCH /v1/directions
PATCH /v1/provider
PATCH /v1/latency-policy
PATCH /v1/voice-profiles
PATCH /v1/debug-capture
PATCH /v1/debug-text
POST /v1/routes/manual-override
POST /v1/self-test/round-trip/start
POST /v1/self-test/round-trip/stop
GET /v1/self-test/round-trip
GET /v1/events/stream
```

Local API security:

- bind to `127.0.0.1` only;
- use a per-session local token or Unix-domain socket permissions;
- redact safe responses;
- never expose PCM buffers through status endpoints.
- cap request bodies and event subscribers; reject missing/invalid bearer token before parsing control payloads.
- default limits are 64 KiB per control request and four concurrent SSE subscribers.

## Tauri UI Design

The Tauri UI is a compact operational console.

Top-level surfaces:

- `Status`;
- `Routing`;
- `Providers`;
- `Voices`;
- `Diagnostics`.

Tray/menu controls:

- start/stop translation;
- microphone direction;
- speaker direction;
- provider;
- latency mode policy;
- debug capture toggle;
- round-trip self-test start/stop;
- open status page.

Tauri does not install, enable, disable or restart `translator.service` in MVP. The systemd user unit owns daemon lifecycle; Tauri controls translation sessions and reports unit/daemon state.

Status page:

- physical mic and sink;
- virtual endpoints;
- current routes;
- provider health;
- per-direction mode and latency p50/p95;
- privacy state;
- sidecar state.
- round-trip self-test preconditions, current checkpoint and per-leg/total latency.

The UI does not process audio. It subscribes to daemon events and sends control commands.

## File-Level Plan

Initial repository layout:

```text
/home/anton/Source/translator
  Cargo.toml
  crates/
    translator-daemon/
    translator-core/
    translator-audio/
    translator-ipc/
  sidecar/
    pyproject.toml
    translator_sidecar/
      provider_contract.py
      local_provider.py
      openai_provider.py
      asr.py
      mt.py
      tts.py
  apps/
    translator-ui/
      src-tauri/
      src/
  systemd/
    translator.service
  tests/
    audio-fixtures/
    scripts/
  docs/
    planning/
```

Rust crates:

- `translator-core`: shared types, policy, privacy-safe event schema.
- `translator-audio`: PipeWire/Pulse graph management, routing watcher, device watcher.
- `translator-ipc`: local API and sidecar IPC.
- `translator-daemon`: process entrypoint and orchestration.

Python sidecar:

- `provider_contract.py`: typed event/frame models.
- `local_provider.py`: local provider adapter.
- `openai_provider.py`: MVP-B adapter.
- `asr.py`, `mt.py`, `tts.py`: engine-specific wrappers.

## Failure Behavior

| Failure | Behavior |
| --- | --- |
| Virtual endpoint creation fails | daemon enters `audio_graph_error`; UI shows command/error and no translation starts |
| Sidecar fails to start | daemon restarts sidecar with bounded backoff; translation direction enters `provider_unavailable` |
| Provider queue overflows | latency policy degrades affected direction or cancels stale utterance according to mode |
| CUDA OOM | close affected sessions, unload failed model, switch to the measured smaller model once or enter `provider_unavailable`; no restart loop |
| Physical mic disappears | outgoing direction pauses as `device_unavailable` |
| Physical sink disappears | incoming direction pauses as `device_unavailable` |
| App route disappears | close stream context and stop provider direction for that route |
| Cloud provider not enabled | cloud adapter returns `cloud_not_enabled`; no network connection starts |
| Debug capture path invalid | debug capture stays disabled and reports safe error |
| Debug capture unsafe path/full disk | reject symlink/non-exclusive open; stop before 5 GiB free-space floor and surface bounded error |
| Daemon receives SIGTERM | stop translation, restore selected app streams to recorded original sinks, close sidecar, unload only journaled daemon-owned modules |
| Daemon crashes | systemd kills remaining cgroup processes, rotates runtime tokens, daemon reconciles ownership journal and restores or recreates endpoints before accepting sessions |

## Validation Strategy

Unit/contract:

- Rust policy tests for latency degradation/recovery with injected delays.
- Rust routing tests for allowlist, manual override and translator-owned stream exclusion.
- Rust routing tests with two simultaneous allowlisted sink-inputs proving only the selected route enters `Translator_Remote_In`.
- Rust device watcher tests for default source/sink changes and virtual endpoint rejection.
- User-unit crash test for orphan sidecar removal, control-token rotation, stream restoration and endpoint reconciliation.
- Python provider lifecycle/contract tests for `open_session`, negotiated PCM format, `audio_frame`, `cancel_utterance`, `close_session`, health events, queue limits and localhost auth.
- Contract tests for stale-session rejection, duplicate/out-of-order events, close timeout and sidecar restart.
- Python local provider tests with short fixture files and mocked ASR/MT/TTS where model runtime is too slow for unit tests.
- Corpus tests for chrF2, critical meaning fields and synthesized-output WER.
- Tauri frontend tests for status, privacy markers, disabled transcript display, debug text warning and disabled cloud provider state.

Audio graph smoke:

```bash
pactl info
wpctl status
pactl list short sinks
pactl list short sources
pactl list short modules
pw-link -l
parecord -d translator_remote_in.monitor /tmp/translator-remote-capture.wav
paplay --device=translator_mic_out /tmp/translator-test-output.wav
```

Synthetic duplex:

- play deterministic English/Russian fixture into each direction;
- measure externally observed speech-onset-to-first-audible plus internal capture-to-first/last-audio;
- assert no feedback loop;
- assert logs contain no speech content.
- run cold, warm and 30-minute simultaneous-duplex profiles while recording CPU, RAM, GPU, VRAM, queue depth, drops and restarts.
- validate open-speaker AEC reference routing and residual-echo attenuation; otherwise mark speaker mode unavailable.

Live human round-trip:

- require headphones and no active real-app incoming route;
- capture one Russian physical-mic utterance and play the exact outgoing English PCM as a monitor tap and VirtualPeer reinjection;
- finish the English monitor tap before reinjection, then play returned Russian audio;
- prove exact-PCM reuse in memory with matching format, frame count, monotonic sequence and rolling hash, without persisting PCM;
- hear the returned Russian translation through the normal incoming path;
- record typed checkpoints, outgoing/incoming/total latency, recursion count and teardown graph diff;
- pass only with both audible outputs, zero recursion and complete route/stream cleanup.

Real app smoke:

- Telegram Desktop route detection and translated speaker playback.
- Firefox/Chromium Meet route detection and translated microphone output.
- At least one Telegram or Meet run uses a real second endpoint, simultaneous overlapping Ru/En speech, receiving-side recording and both-direction evidence.
- Zoom only after post-MVP acceptance task starts.

## Design Risks

| Risk | Mitigation |
| --- | --- |
| Local provider cannot meet acceptable latency/quality | latency policy degrades per direction; OpenAI MVP-B comparison adapter; model choices remain implementation decision |
| Feedback loop through virtual endpoints | explicit routing invariants, translator-owned stream markers, graph tests |
| Acoustic loop through open speakers | PipeWire WebRTC AEC with translated playback as reference; headphones required if validation fails |
| Cloud audio starts without consent | local default, explicit provider enablement, cloud egress status and tests |
| Rust audio implementation takes too long | MVP may use `pactl`/Pulse-compatible layer behind traits, with native PipeWire worker as post-MVP optimization |
| Tauri UI distracts from audio proof | tasks sequence daemon/audio/provider before UI polish |
| Model dependencies are heavy | local provider task owns model-selection benchmark before locking model bundle |

## Bounded Decisions Deferred To Tasks

- Exact local MT model after one-candidate download gate.
- Female Piper voice assets after inventory and disk-cost record.
- Native libpulse/libpipewire replacement only if measured `pactl` orchestration overhead or reliability fails acceptance.

## Repository Context Gate

`repo-c4.json` is a machine context index, not runtime truth. Each task reads `meta.architecture_fidelity`, `meta.tool_orchestration`, relevant components, edges, data structures and external services before source inspection. Each task closes by running the full `repo-c4-scan` generator and validator; a failed or stale index blocks task completion.
