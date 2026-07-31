# Translator Live Duplex Tasks

Status: audited implementation issue set.

Planning context:

- `translator-live-duplex-prd.md`
- `translator-live-duplex-design.md`

## Branches And Commit Rules

Repository: `/home/anton/Source/translator`

Suggested branch after project initialization:

```text
translator-live-duplex-mvp
```

Commit messages should start with:

```text
translator:
```

Planning files live under:

```text
/home/anton/Source/translator/docs/planning/
```

## Cross-Cutting Rules

- Preserve component ownership:
  - Rust daemon owns PipeWire/Pulse audio graph, routing watcher, device watcher, sidecar supervision, local control API and latency policy.
  - Python sidecar owns local ASR/MT/TTS provider internals.
  - Provider adapters own provider-specific speech translation mechanics behind the stable contract.
  - Tauri UI owns tray/menu/status controls only.
  - systemd user unit owns session startup/restart.
- Do not capture the physical default sink monitor in normal incoming mode.
- Do not route all allowlisted app streams automatically. MVP has one selected incoming route for the speaker direction.
- Production and manual routing must not move translator-owned streams to `Translator_Remote_In`. The human round-trip controller may authorize only its daemon-held `{session_id, stream_serial_or_node_id, process_identity}` capability; forged/stale metadata remains rejected.
- Do not capture `Translator_Virtual_Mic` as the outgoing physical source.
- Default provider is local. Cloud providers require explicit per-provider enablement.
- No audio, transcript or translation persistence by default.
- Transcript/translation text display is disabled unless explicit debug text mode is enabled.
- MVP-A functional, routing and privacy gates must pass with the local provider before MVP-B OpenAI work; a measured local latency miss does not block comparison.
- Tests must include synthetic PCM fixtures, injected latency and privacy markers.
- Headphone mode is the baseline. Open-speaker mode stays disabled until PipeWire WebRTC AEC passes reference-routing and acoustic attenuation acceptance.
- Reuse existing local speech model assets before downloading new ones:
  - `Systran/faster-whisper-small` from `/home/anton/.cache/huggingface/hub/models--Systran--faster-whisper-small`;
  - `Systran/faster-whisper-large-v3` from `/home/anton/Source/uncle-freud-bot/.data/faster-whisper`;
  - Piper voices from `/home/anton/Source/uncle-freud-bot/.data/piper-voices`.
- Do not download MT/TTS alternatives until a task proves no suitable local asset exists and records expected disk cost.
- Record every downloaded model in `models/manifest.json` with source, revision, SHA-256, license, languages, bytes and cache path. New MVP downloads are capped at 2 GiB and must leave at least 20 GiB free.
- At task start, read `/home/anton/Source/translator/repo-c4.json`, inspect `meta.architecture_fidelity` and `meta.tool_orchestration`, use relevant components/edges to select source, then verify that source directly. If Task 1 finds no index, it inventories current files directly; its closing scan remains mandatory.
- At task end, run `python3 /home/anton/.agents/skills/repo-c4-scan/scripts/generate_repo_c4.py /home/anton/Source/translator`, then `python3 /home/anton/.agents/skills/repo-c4-scan/scripts/validate_repo_c4.py /home/anton/Source/translator/repo-c4.json`. Inspect the diff for lost intent and include scan/tool-fidelity status in the final report. A failed or stale index blocks task completion.
- Critic review is required after this tasks document and after task-prompts document.

## Delivery Gates

### MVP-A Local Provider Gate

MVP-A is complete only when:

- daemon creates and validates virtual endpoints;
- routing watcher supports one selected incoming route;
- device watcher handles physical source/sink changes without selecting translator endpoints;
- local provider contract and sidecar lifecycle pass tests;
- local ASR/MT/TTS chain runs in both directions;
- local provider is classified as `meets_target`, `usable_degraded` or `fails_usable_limit` from external graph-boundary latency;
- synthetic duplex loopback passes;
- Telegram Desktop and Firefox/Chromium Meet local-provider smokes pass;
- no cloud provider is needed or silently started;
- logs contain no spoken content.

### MVP-B OpenAI Adapter Gate

MVP-B starts after MVP-A.

For this prerequisite, MVP-A means its functional, routing and privacy gates. A local-provider latency miss does not block MVP-B comparison.

MVP-B is complete only when:

- OpenAI adapter implements the same provider contract;
- cloud opt-in, credential handling and audio-egress status are tested;
- OpenAI provider comparison runs through the same synthetic and at least one real-app route;
- local provider remains available and default.

### Usable-Release Gate

- At least one explicitly enabled provider is `meets_target` or `usable_degraded` in both directions.
- `meets_target` means p95 <= 1000 ms; `usable_degraded` means >1000 and <=1500 ms; `fails_usable_limit` means >1500 ms, quality-floor failure or unstable duplex resources.
- Headphones pass duplex acceptance.
- Open-speaker mode is available only after AEC reference-routing and residual-echo tests pass.
- If no provider reaches the usable limit, report a prototype and do not label it a usable MVP release.

## Task 1. Project Scaffold, Toolchain And Shared Contracts

- [X] Completed

Repo: `/home/anton/Source/translator`

Goal: create the initial Rust/Python/Tauri project structure and shared type contracts without implementing audio routing yet.

Startup context gate:

- if `repo-c4.json` exists, read its fidelity/tool status and relevant docs components;
- if it does not exist, inventory the current files directly and continue;
- in both cases, verify source directly and generate/validate the closing index.

Scope:

- Add root workspace files for Rust crates:
  - `crates/translator-core`;
  - `crates/translator-audio`;
  - `crates/translator-ipc`;
  - `crates/translator-daemon`.
- Add Python sidecar scaffold:
  - `sidecar/pyproject.toml`;
  - `sidecar/translator_sidecar/provider_contract.py`.
- Add Tauri app scaffold under `apps/translator-ui`.
- Add `systemd/translator.service` draft.
- Add shared schema definitions for:
  - `AudioDirection`;
  - `TranslationMode`;
  - `VoiceProfile`;
  - `LatencyPolicyState`;
  - provider session lifecycle;
  - provider input/audio/health/latency/error events.
- Add privacy-safe logging conventions.
- Add Protobuf schemas for gRPC streaming over a user-owned Unix-domain socket.
- Add `.gitignore` entries for secrets, model caches, runtime sockets, debug captures and build output.
- Add test fixture directory and README.

Acceptance:

- Rust workspace builds with empty/stub crates.
- Python sidecar package imports and validates provider contract models.
- Tauri app starts as a stub status page without audio access.
- Contract tests cover ProviderHealth shape, open/close session messages and privacy-safe errors.
- No frontend code can access raw PCM types except through redacted status fixtures.

Validation:

- `cargo test --workspace`
- Python sidecar unit tests.
- Tauri unit/build check.
- `rg -n "TODO|TBD" docs/planning crates sidecar apps systemd` has no planning-critical placeholders.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 2. Audio Graph Endpoint Manager

- [X] Completed

Goal: create and inspect the virtual audio endpoints required for full-duplex routing.

Scope:

- Implement `AudioGraph` trait in `translator-audio`.
- Create `translator_mic_out`, `translator_virtual_mic` and `translator_remote_in`.
- Record module ids and unload only daemon-owned modules.
- Mark translator-owned streams/endpoints with stable properties where the Pulse/PipeWire layer allows it.
- Add graph inspection through `pactl`/`pw-link` or library APIs.
- Expose graph state as typed daemon state for Task 4 to publish through the status API.
- Add safe failure states for missing `pactl`, failed module load and duplicate/stale endpoints.

Acceptance:

- `translator-daemon --audio-graph-smoke` creates all required endpoints.
- `pactl list short sinks` shows `translator_mic_out` and `translator_remote_in`.
- `pactl list short sources` shows `translator_virtual_mic`.
- Re-running the smoke is idempotent.
- The daemon does not unload non-owned audio modules.
- Status reports endpoint ids/names and safe errors without raw audio.

Validation:

- Rust unit tests with command fakes.
- Local audio graph smoke on this workstation:
  - `pactl info`;
  - `wpctl status`;
  - `pactl list short sinks`;
  - `pactl list short sources`;
  - `pw-link -l`.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 3. Routing Watcher And Device Watcher

- [X] Completed

Goal: route exactly one selected incoming app stream and keep physical device selection safe.

Scope:

- Implement sink-input/source-output discovery.
- Add default candidate allowlist for Telegram Desktop, Firefox, Chromium and Chrome.
- Implement one active incoming route for `speaker`.
- Add manual route override.
- Reject translator-owned streams.
- Reject all virtual translator endpoints as physical mic/sink candidates.
- Implement physical source/sink watcher with unplug/replug handling.
- Add acoustic feedback warning when active sink appears to be speakers.
- Classify headphone/open-speaker mode and block open-speaker duplex until Task 7 AEC capability is validated.
- Expose typed routes and route candidates for Task 4 to publish through the local API.

Acceptance:

- Allowlist discovers candidates but does not blindly route multiple streams.
- With two allowlisted sink-inputs, only the selected stream moves to `translator_remote_in`.
- Production/manual routing never moves translator-owned sink-inputs to `translator_remote_in`; the authenticated Task 7 self-test exception does not exist yet.
- Daemon never selects `translator_virtual_mic` as physical mic.
- Daemon never captures the physical default sink monitor in normal mode.
- Device loss enters `device_unavailable` and recovery resumes only after valid physical device selection.

Validation:

- Rust routing tests with fake Pulse/PipeWire graph.
- Rust device watcher tests for default source/sink changes.
- Manual local smoke with synthetic sink-inputs where practical.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 4. Daemon Runtime, Local API And Latency Policy

- [X] Completed

Goal: make the daemon a user-session service with observable per-direction runtime state.

Scope:

- Implement daemon entrypoint.
- Add HTTP control plus SSE status API on loopback; Tauri Rust backend owns the bearer token.
- Add session token or Unix-socket permission gate.
- Rotate `%t/translator/control.token` on daemon start/restart, mode `0600`; Tauri Rust rereads it and frontend state never receives it.
- Reject missing/invalid bearer before request-body parsing; cap control bodies at 64 KiB and concurrent SSE subscribers at four.
- Add endpoints:
  - `GET /v1/status`;
  - `GET /v1/audio-graph`;
  - `GET /v1/routes`;
  - `GET /v1/routes/candidates`;
  - `POST /v1/translation/start`;
  - `POST /v1/translation/stop`;
  - `PATCH /v1/directions`;
  - `PATCH /v1/provider`;
  - `PATCH /v1/latency-policy`;
  - `PATCH /v1/voice-profiles`;
  - `PATCH /v1/debug-capture`;
  - `PATCH /v1/debug-text`;
  - `POST /v1/routes/manual-override`;
  - `POST /v1/self-test/round-trip/start`;
  - `POST /v1/self-test/round-trip/stop`;
  - `GET /v1/self-test/round-trip`;
  - `GET /v1/events/stream`.
- Add typed round-trip self-test preconditions, lifecycle, checkpoints, per-leg latency and total `physical_mic_onset_to_returned_ru_first_audible_ms`; keep the VirtualPeer capability in the daemon and out of frontend state.
- Implement per-direction `LatencyPolicyState`.
- Add rolling p95 windows, minimum sample count, cooldown and hysteresis.
- Add fast degradation after three consecutive utterance breaches or queue lag above 500 ms for two seconds.
- Own capture/playback queue contracts in the daemon, bounded by buffered milliseconds.
- Add privacy-safe structured logs.
- Bound each explicit debug-capture session to 10 minutes or 500 MiB, whichever comes first, with user-only file permissions.
- Use a dedicated `0700` state directory, `O_NOFOLLOW | O_CREAT | O_EXCL`, `0600` files and a 5 GiB free-space floor.

Acceptance:

- API binds only to loopback or user-owned Unix socket.
- Token rotates on restart, has mode `0600`, and invalid bearer is rejected before payload parsing.
- Oversized bodies and a fifth SSE subscriber are rejected without affecting active translation.
- Tauri/frontend cannot reach sidecar directly.
- Latency degradation affects only the direction whose metrics exceed thresholds.
- Recovery requires stable windows and cooldown.
- `debug_text` toggles transcript/translation debug display without enabling `debug_capture`.
- Disabling `debug_text` clears the bounded in-memory text buffer.
- Logs contain route and latency events without spoken content.
- Debug capture stops at the configured hard bound and never survives restart.
- Round-trip self-test controls are authenticated, allow one five-minute session, reject open-speaker/active-route conflicts and expose no text unless `debug_text` is enabled.

Validation:

- Rust API tests.
- Latency policy tests with injected delays.
- Privacy marker tests.
- Debug-capture symlink, quota and simulated low-free-space tests.
- Control-token rotation/permissions, pre-parse auth, request-size and SSE-subscriber limit tests.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 5. Python Sidecar Lifecycle And Mock Provider

- [X] Completed

Goal: make provider IPC executable before real ASR/MT/TTS models are integrated.

Scope:

- Implement sidecar process startup and authenticated connection.
- Implement `open_session`, `audio_frame`, `cancel_utterance`, `close_session`.
- Implement `ProviderHealth`, `ProviderLatency`, `ProviderError`.
- Add sidecar-owned queue bounds:
  - provider input queue max 800 ms;
  - provider output queue max 1200 ms.
- Carry `session_id` and event sequence on every provider event; reject stale, duplicate and post-final events.
- Enforce a 2-second close acknowledgement deadline and supervised restart on timeout.
- Cancel all work older than 3000/2000/1000 ms in Quality-first/Balanced/Streaming-first and never play expired audio.
- Add mock provider that transforms fixture audio deterministically.
- Add injected latency and error modes.
- Ensure transcript/translation deltas are emitted only when `debug_text_enabled=true`.

Acceptance:

- Daemon can open two independent provider sessions.
- Sidecar rejects unauthenticated connections.
- Queue overflow produces typed errors or cancellation, not unbounded memory growth.
- Debug text marker never appears in normal mode status/logs.
- Mock provider supports synthetic duplex tests.

Validation:

- Python unit tests.
- Rust/Python integration contract tests.
- Synthetic PCM fixture tests.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 6. Local Provider: ASR, MT And TTS MVP-A

- [X] Completed

Goal: implement the first real local provider chain.

Scope:

- Integrate ASR through existing local `faster-whisper`/CTranslate2 caches before downloading anything:
  - smoke with `/home/anton/.cache/huggingface/hub/models--Systran--faster-whisper-small`;
  - Quality-first benchmark with `/home/anton/Source/uncle-freud-bot/.data/faster-whisper/models--Systran--faster-whisper-large-v3`.
- Select and integrate one CTranslate2-compatible Ru <-> En MT model only after confirming no local MT cache exists.
- Record model name, source, expected disk size and cache path before any MT download.
- Add/update `models/manifest.json`; enforce the 2 GiB incremental-download budget, 20 GiB post-download free-space floor and license/checksum fields.
- Integrate existing Piper voices:
  - `/home/anton/Source/uncle-freud-bot/.data/piper-voices/ru_RU-dmitri-medium.onnx`;
  - `/home/anton/Source/uncle-freud-bot/.data/piper-voices/en_US-ryan-medium.onnx`.
- Treat Qwen3 TTS at `/home/anton/Source/uncle-freud-bot/.data/qwen-tts` as non-MVP fallback only.
- Implement `VoiceProfile` male/female presets per target language.
- Bootstrap with available male Piper presets; select and download exactly one female Russian and one female English Piper-compatible voice after inventory records candidates, licenses, checksums and disk cost.
- Implement per-mode provider behavior:
  - Quality-first;
  - Balanced;
  - Streaming-first.
- Keep source and translated text out of logs and normal UI.
- Implement stable-prefix commit so TTS never speaks text that the pipeline still considers revisable.
- Benchmark cold start, warm first audio, throughput, CPU/RAM, GPU/VRAM peak and simultaneous duplex for local small and large ASR candidates.
- Keep only the selected ASR model resident in normal operation; test typed CUDA OOM fallback.
- Enforce <=10 GiB peak VRAM in normal simultaneous duplex.
- Add a versioned quality corpus with 10 excluded warmups and at least 100 measured utterances per direction; compute chrF2 and synthesized-output WER.
- Add CPU fallback or explicit unsupported-state handling when CUDA dependencies are missing.

Acceptance:

- Existing local ASR caches are used for first ASR smokes with `local_files_only` or equivalent no-download behavior.
- Existing local Piper voices are used for first TTS smokes.
- Any new model download is preceded by a written inventory and disk-cost note.
- Local provider translates short Russian fixture to English audio.
- Local provider translates short English fixture to Russian audio.
- Two directions can run concurrently with isolated sessions.
- Male and female presets work for both target languages; unavailable or silent gender fallback fails MVP-A.
- Local provider has provider-level latency/resource characterization for both directions and simultaneous duplex; release classification waits for Task 7 graph-boundary measurements.
- Corpus chrF2 is >=45 per direction, critical negation/number/name subset has no meaning-changing errors, and synthesized-output WER is <=15%.
- Provider emits latency and health without content leaks.
- Missing model/dependency states are safe and visible.

Validation:

- Python local provider tests with short fixtures.
- GPU availability check.
- No-download ASR/TTS smoke using existing cache paths.
- Synthetic local-provider duplex benchmark.
- Model manifest and disk/VRAM budget report.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 7. Audio Capture/Playback Integration And Synthetic Duplex Benchmark

Goal: connect real audio graph streams to provider sessions and prove the duplex path without meeting apps.

Scope:

- Capture physical mic stream for outgoing direction.
- Capture `translator_remote_in.monitor` for incoming direction.
- Play translated microphone output to `translator_mic_out`.
- Play translated speaker output to validated physical sink.
- Implement resampling/downmixing.
- Own capture and playback queues, bounded to 400 ms each per direction.
- Own VAD/EOU decisions in the Rust capture path, keep `stream_id` stable for the provider session, rotate `utterance_id` at each EOU boundary, and make the sidecar consume capture-issued EOU as authoritative input.
- Production-wire `ProviderStreamCoordinator` and the provider watchdog for both directions; purge queued playback before cancellation and apply bounded close/restart escalation.
- Implement open-speaker PipeWire WebRTC AEC using translated incoming playback as the reference; retain headphones as mandatory fallback.
- Add synthetic benchmark runner with deterministic fixture playback.
- Add live human round-trip mode: capture the physical mic, tap the exact English PCM from `Translator_Virtual_Mic`, finish monitoring it in headphones, then reinject the same frames through one session-bound VirtualPeer sink-input.
- Make the live self-test mutually exclusive with real incoming routes and open-speaker mode; bind its routing exception to the daemon-held `{session_id, stream_serial_or_node_id, process_identity}` tuple.
- Reject the same VirtualPeer in production/manual profiles and reject forged/stale `translator.test_profile` metadata; authorize it only during its matching self-test session.
- Prove exact-PCM reuse in memory with matching format, frame count, monotonic frame sequence and rolling hash; do not persist PCM for this proof.
- Emit stage checkpoints plus outgoing-leg, incoming-leg and total round-trip latency; keep transcript/translation text behind `debug_text`.
- Bound the self-test to five minutes and implement idempotent teardown for stop, timeout, failure and daemon restart.
- Measure external graph-boundary `speech_onset_to_first_audible_ms`, with source onset and output audibility detected by the harness.
- Add digital feedback-loop and acoustic residual-echo assertions.
- For AEC, record device geometry/volume, run the `-20 dBFS` fixture, calculate ERLE and check far-end-only false triggers.
- Add cold, warm and 30-minute simultaneous-duplex profiles with resource/queue/drop/restart telemetry.
- Use 10 excluded warmups and at least 100 measured utterances per direction; report timeout/drop rate separately.

Acceptance:

- Synthetic loopback runs both directions concurrently.
- On this workstation, a headphone-mode live run lets the user speak Russian, hear the complete exact outgoing English virtual-peer tap, then hear its Russian translation through the normal incoming path.
- Live self-test recursion count is zero, no original Russian reaches the virtual microphone, and teardown leaves no VirtualPeer stream or route change.
- Exact-PCM evidence matches format, frame count, monotonic sequence and rolling hash between capture and reinjection without writing PCM to disk.
- `speech_onset_to_first_audible_ms`, `capture_to_first_audio_ms`, `capture_to_last_audio_ms`, queue lag and provider latency are recorded per direction.
- Timeout/drop rate is reported separately and stays below 1%.
- No unauthorized translator-owned stream re-enters incoming capture; only the exact daemon-authorized VirtualPeer capability tuple is permitted during its matching self-test session.
- Logs contain no speech content.
- Quality-first starts first and degrades only when injected latency requires it.
- Open-speaker mode stays disabled when AEC setup, reference routing or attenuation fails.
- Open-speaker mode requires median ERLE >=15 dB and zero outgoing VAD/translation triggers in the 60-second far-end-only run.
- Assign final `meets_target`, `usable_degraded` or `fails_usable_limit` classification from graph-boundary latency, quality and resource evidence.

Validation:

- Audio graph smoke.
- Synthetic duplex benchmark report.
- Live human round-trip report with checkpoints, exact-PCM counters/hash, audible-output confirmation, per-leg/total latency and before/after graph diff.
- Privacy marker scan.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 8. Tauri Tray And Status Page MVP

- [X] Completed

Goal: provide the desktop control surface required for live use.

Scope:

- Implement Tauri tray/menu.
- Implement local status page with:
  - audio graph;
  - physical mic/sink;
  - route candidates and selected route;
  - provider health;
  - latency p50/p95 per direction;
  - mode and degradation reason;
  - privacy/cloud/debug state;
  - voice presets.
- Add controls for start/stop, directions, provider, voice gender, manual route override and debug capture.
- Add a Diagnostics control for round-trip self-test start/stop, precondition failures, current checkpoint and per-leg/total latency.
- Keep transcript/translation text hidden unless explicit debug text mode is enabled.
- Add separate `debug_text` control from `debug_capture`.
- Add visible tray/status warning while debug text is enabled.
- Store debug transcript/translation text only in a bounded in-memory ring buffer of at most 200 events and 1 MiB.
- Clear debug text buffer on session stop, provider switch, daemon restart and UI close.
- Prove debug text is never logged, persisted, sent to telemetry, included in debug-capture metadata or embedded in provider errors.
- Add UI warnings for cloud provider, acoustic feedback and debug text.

Acceptance:

- Tauri reconnects to an already running daemon.
- First screen is status, not a landing page.
- UI cannot display transcript/translation text in normal mode.
- Debug text is controlled separately from debug capture.
- Debug text warning is visible while enabled.
- Debug text buffer is memory-only, bounded and cleared on stop/provider switch/restart/UI close.
- Privacy marker tests prove transcript/translation text does not appear in logs, local storage, telemetry, debug-capture metadata or provider errors.
- Cloud provider cannot be enabled without visible opt-in.
- Manual route override can select one incoming route.
- Controls reflect daemon state after restart.
- Round-trip self-test requires explicit start, presents Stop while active, never exposes text outside `debug_text`, and reports teardown completion.

Validation:

- Tauri unit/component tests.
- Screenshot checks for desktop status page.
- Local manual tray/status smoke.
- Local live round-trip UI smoke in headphones.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 9. User-Level systemd And Desktop Run Mode

- [X] Completed
- [X] Terminal lifecycle command is `scripts/translator-desktop up|down|restart|status|logs`; `scripts/translator-desktop install|up` installs the user-local `~/.local/bin/translator` command, so `translator up|down|restart|status|logs` works from any current directory. When `target/release/translator-daemon` or `target/release/translator-ui` exists, the lifecycle refreshes the matching `~/.local/bin` binary before launch. `up`, `start` and `restart` wait briefly for the daemon HTTP API, then start the current-session `translator-ui` tray/status process when a graphical desktop session is available. `down` and `restart` run daemon-owned audio graph cleanup after stopping the user unit.

Goal: make the service run as a user-session desktop service.

Scope:

- Finalize `systemd/translator.service`.
- Add install/start/stop/status scripts.
- Ensure daemon starts in the user PipeWire session.
- Install an XDG autostart entry for Tauri; systemd remains the only daemon lifecycle owner and Tauri does not manage the unit.
- Add restart/backoff behavior.
- Configure `RuntimeDirectory=translator`, mode `0700`, `KillMode=control-group` and bounded `Restart=on-failure`.
- Journal daemon-owned module ids and selected streams' original sinks under the user runtime directory.
- On stop/SIGTERM, restore app streams before unloading only journaled daemon-owned modules.
- Add install, disable and uninstall flows that leave unrelated PipeWire state unchanged.

Acceptance:

- `systemctl --user start translator.service` starts daemon.
- `systemctl --user stop translator.service` stops daemon and cleans daemon-owned endpoints.
- Tauri reconnects after daemon restart.
- Debug capture is disabled after restart.
- Selected app streams return to their original sinks on stop/crash recovery.
- Forced daemon failure leaves no sidecar process, rotates control token and reconciles endpoints before accepting a new session.
- No root/system service is required.

Validation:

- systemd user smoke.
- Audio endpoint cleanup smoke.
- Tauri reconnect smoke.
- Forced daemon crash/orphan/token-rotation smoke.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 10. Real-App MVP-A Smokes: Telegram And Meet

- [X] Autonomous evidence captured in `docs/benchmarks/task10-validation-report.json`; earlier blocked live second-endpoint status is superseded by live user-confirmed Meet and Telegram duplex checks.
- [X] Synthetic Telegram/Meet/Zoom app-stream diagnostic captured in `docs/benchmarks/task10-simulated-app-streams-report.json`; live Meet and Telegram evidence is captured separately in `docs/benchmarks/task10-meet-live-duplex-check.json` and `docs/benchmarks/task10-telegram-live-duplex-check.json`.
- [X] MVP-A local-provider routing/privacy gate is satisfied for Meet and Telegram; Task 7 graph-boundary latency still classifies the local provider as `fails_usable_limit`, so Task 11 provider comparison remains mandatory.

Goal: prove MVP-A against the selected real desktop apps.

Scope:

- Validate Telegram Desktop incoming route discovery and manual/automatic selection.
- Validate Firefox/Chromium Meet route discovery and manual/automatic selection.
- Validate `Translator_Virtual_Mic` as call app microphone.
- Validate translated-only behavior in both directions.
- Collect latency and privacy evidence.
- Run at least one Telegram or Meet smoke against a real second endpoint with simultaneous overlapping Ru/En speech and receiving-side recording.
- Record known limitations and setup steps.

Acceptance:

- Telegram Desktop incoming audio can be selected, translated and played to physical sink.
- Firefox/Chromium Meet incoming audio can be selected, translated and played to physical sink.
- Outgoing translated microphone reaches the call app through `Translator_Virtual_Mic`.
- The remote endpoint receives outgoing translation while the local endpoint simultaneously receives incoming translation.
- Original audio is not mixed by default.
- Synthetic and real-app logs contain no spoken content.
- MVP-A gate is satisfied.
- Result states whether local provider meets the usable-release latency gate; a miss triggers mandatory MVP-B comparison.

Validation:

- Telegram Desktop smoke report.
- Firefox/Chromium Meet smoke report.
- Latency ledger report.
- Privacy marker scan.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 11. MVP-B OpenAI Realtime Translation Adapter

- [ ] OpenAI adapter preflight/opt-in evidence captured in `docs/benchmarks/task11-openai-adapter-preflight.json`; MVP-A live route debt is closed, credentialed OpenAI model/websocket smoke passed, synthetic duplex comparison passed, and daemon/sidecar OpenAI runtime wiring is covered by deterministic fake-WebSocket tests; real-app OpenAI smoke remains pending.

Precondition: MVP-A functional, routing and privacy gates passed. Local-provider latency may be below the usable-release limit.

Goal: add OpenAI Realtime Translation as the second provider adapter without changing audio routing.

Scope:

- Add OpenAI provider adapter in Python sidecar.
- Use same provider lifecycle and event contract.
- Require explicit cloud provider enablement.
- Load credentials from ignored config or OS secret storage.
- Show `audio_leaves_machine=true` in status.
- Map translated audio and transcript deltas through debug-gated provider events.
- Add provider comparison latency report.

Acceptance:

- OpenAI adapter cannot start until enabled.
- Missing credentials produce safe error without network session.
- Cloud enabled state is visible before first audio leaves the machine.
- Local provider remains default.
- OpenAI adapter passes synthetic duplex comparison and at least one real-app route smoke.
- Combined evidence states whether at least one provider meets the usable-release latency gate.

Validation:

- Provider contract tests.
- Cloud opt-in/privacy tests.
- Synthetic OpenAI comparison smoke.
- One real-app OpenAI smoke.
- Full-depth `repo-c4-scan` generation and validation pass.

## Task 12. Post-MVP Zoom Acceptance

- [X] Zoom routing allowlist/preflight diagnostic captured in `docs/benchmarks/task12-zoom-diagnostic-report.json`; the earlier route-only status is superseded by full live Zoom duplex acceptance.
- [X] Live Zoom conference check captured a `ZOOM VoiceEngine` candidate and `Translator_Virtual_Mic`; daemon/manual route selection now falls back to `route_method=pipe_wire_links` after Zoom rejects `pactl move-sink-input`; evidence in `docs/benchmarks/task12-zoom-live-route-check.json`.
- [X] Zoom microphone selection was validated by moving the Zoom source-output to `translator_virtual_mic`, observing `translator_virtual_mic:capture_MONO -> ZOOM VoiceEngine:input_MONO`, and restoring it to `alsa_input.usb-Jieli_Technology_UACDemoV1.0-00.mono-fallback`.
- [X] Live incoming Zoom translation from Russian to English reached the physical headphones after the `translator-incoming-playback` stream volume was corrected from `0%` to `100%`; user-confirmed evidence in `docs/benchmarks/task12-zoom-live-translation-check.json`.
- [X] Full Task 12 Zoom duplex acceptance is user-confirmed after the outgoing and incoming volume/routing fixes; Task 7 latency and Task 11 OpenAI comparison debts are still carried.

Goal: validate Zoom after MVP-A and MVP-B foundations are stable.

Scope:

- Add Zoom to allowlist candidate detection.
- Validate Zoom microphone selection with `Translator_Virtual_Mic`.
- Validate Zoom incoming sink-input discovery/selection.
- Add setup notes for Zoom-specific audio settings.

Acceptance:

- Zoom route selection works without routing unrelated streams.
- Zoom outgoing translation reaches remote side through virtual microphone.
- Zoom incoming translation reaches physical sink.
- Known Zoom limitations are documented.

Validation:

- Zoom smoke report.
- Latency and privacy evidence.
- Full-depth `repo-c4-scan` generation and validation pass.
