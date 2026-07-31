# Translator Live Duplex Task Prompts

Status: audited executable prompts for sequential implementation.

## Common Rules For Every Task

- Work from current code first. Planning docs are context, not proof that behavior exists.
- Preserve ownership:
  - Rust daemon owns audio graph, routing, device watcher, local API, latency policy and sidecar supervision.
  - Python sidecar owns local ASR/MT/TTS and provider adapters.
  - Tauri owns tray/menu/status UI only.
  - systemd user unit owns session startup/restart.
- Do not capture the physical default sink monitor in normal incoming mode.
- Do not capture `Translator_Virtual_Mic` as physical microphone input.
- Do not route every allowlisted app stream. MVP has one selected incoming route.
- Production and manual routing must not move translator-owned streams to `Translator_Remote_In`. The human round-trip controller may authorize only its daemon-held `{session_id, stream_serial_or_node_id, process_identity}` capability; forged/stale metadata remains rejected.
- Default provider is local. Cloud providers require explicit opt-in.
- No audio, transcript or translation persistence by default.
- `debug_text` is separate from `debug_capture`; transcript/translation text is hidden unless `debug_text` is explicitly enabled.
- Reuse existing local speech models before downloading new ones:
  - `/home/anton/.cache/huggingface/hub/models--Systran--faster-whisper-small`;
  - `/home/anton/Source/uncle-freud-bot/.data/faster-whisper/models--Systran--faster-whisper-large-v3`;
  - `/home/anton/Source/uncle-freud-bot/.data/piper-voices`.
- Do not download MT/TTS alternatives until the active task records missing local assets, selected model, expected disk size and target cache path.
- Record every downloaded model in models/manifest.json with source, revision, SHA-256, license, languages, bytes and cache path. New MVP downloads are capped at 2 GiB and must leave at least 20 GiB free.
- MVP-A functional, routing and privacy gates must pass with the local provider before MVP-B OpenAI work; a measured local latency miss does not block comparison.
- Use synthetic PCM fixtures, injected latency and privacy markers in tests.
- Run a critic before final report for each implementation task.
- At the beginning of every task, read `/home/anton/Source/translator/repo-c4.json`, inspect `meta.architecture_fidelity` and `meta.tool_orchestration`, use relevant components/edges to choose files, and verify those files directly. If Task 1 finds no index, inventory current files directly; its closing scan remains mandatory.
- At the end of every task, run `python3 /home/anton/.agents/skills/repo-c4-scan/scripts/generate_repo_c4.py /home/anton/Source/translator`, then `python3 /home/anton/.agents/skills/repo-c4-scan/scripts/validate_repo_c4.py /home/anton/Source/translator/repo-c4.json`. Inspect the index diff for lost intent and report helper-tool fidelity. Do not close a task with a failed or stale index.
- Do not revert unrelated user changes.

## Task 1 Prompt

- [x] Completed

```text
Взять в работу Task 1: Project Scaffold, Toolchain And Shared Contracts.

Repo:
- /home/anton/Source/translator

Planning context:
- docs/planning/translator-live-duplex-prd.md
- docs/planning/translator-live-duplex-design.md
- docs/planning/translator-live-duplex-tasks.md

Startup:
- Check current files and git status if the repo has been initialized.
- Read the three planning docs above.
- If repo-c4.json exists, read architecture_fidelity, tool_orchestration and relevant indexed components. If absent, inventory current files directly. Verify selected files directly in either case.
- Inspect local toolchain versions: rustc, cargo, python3, uv, node/bun, npm, pactl, wpctl.
- Do not implement audio routing yet.

Goal:
Create the initial Rust/Python/Tauri structure and shared contracts.

Scope:
- Add Rust workspace and crates:
  - crates/translator-core
  - crates/translator-audio
  - crates/translator-ipc
  - crates/translator-daemon
- Add Python sidecar scaffold:
  - sidecar/pyproject.toml
  - sidecar/translator_sidecar/provider_contract.py
- Add Tauri app scaffold under apps/translator-ui.
- Add systemd/translator.service draft.
- Add typed/shared contracts for AudioDirection, TranslationMode, VoiceProfile, LatencyPolicyState, provider session lifecycle, input/audio/health/latency/error events.
- Add privacy-safe logging conventions.
- Add Protobuf schemas for gRPC bidirectional streaming over a user-owned Unix-domain socket.
- Add .gitignore entries for secrets, model caches, runtime sockets, debug captures and build output.
- Add tests/audio-fixtures/README.md.

Acceptance:
- Rust workspace builds with stub crates.
- Python package imports and validates provider contract models.
- Tauri app starts as a stub status page without audio access.
- Contract tests cover ProviderHealth, open/close session messages and privacy-safe errors.
- Frontend cannot access raw PCM types except redacted fixtures.

Validation:
- cargo test --workspace
- Python sidecar unit tests
- Tauri unit/build check
- rg -n "TODO|TBD" docs/planning crates sidecar apps systemd
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review scaffold boundaries, provider contracts and privacy surface before final report.

Final report:
- Changed files.
- Commands run and results.
- Critic verdict.
- Residual risks.
```

## Task 2 Prompt

- [x] Completed

```text
Взять в работу Task 2: Audio Graph Endpoint Manager.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect current Task 1 code before editing.
- Check pactl info, wpctl status and current sinks/sources.

Goal:
Create and inspect the virtual endpoints required for duplex routing.

Scope:
- Implement AudioGraph trait in translator-audio.
- Create translator_mic_out, translator_virtual_mic and translator_remote_in.
- Record module ids and unload only daemon-owned modules.
- Mark translator-owned streams/endpoints where the Pulse/PipeWire layer allows it.
- Add graph inspection through pactl/pw-link or library APIs.
- Expose typed graph state for Task 4 to publish through daemon status API.
- Add safe failure states for missing pactl, failed module load and duplicate/stale endpoints.

Acceptance:
- translator-daemon --audio-graph-smoke creates all endpoints.
- pactl list short sinks shows translator_mic_out and translator_remote_in.
- pactl list short sources shows translator_virtual_mic.
- Re-running smoke is idempotent.
- Daemon does not unload non-owned modules.
- Status reports endpoint ids/names and safe errors without raw audio.

Validation:
- cargo test --workspace
- pactl info
- wpctl status
- pactl list short sinks
- pactl list short sources
- pw-link -l
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review endpoint ownership, cleanup and feedback-loop risks.

Final report:
- Changed files.
- Audio modules created/cleaned.
- Validation evidence.
- Critic verdict.
```

## Task 3 Prompt

- [x] Completed

```text
Взять в работу Task 3: Routing Watcher And Device Watcher.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect current audio graph code and tests.
- Capture current pactl/wpctl stream state.

Goal:
Route exactly one selected incoming app stream and keep physical device selection safe.

Scope:
- Implement sink-input/source-output discovery.
- Add route candidates for Telegram Desktop, Firefox, Chromium and Chrome.
- Implement one active incoming route for speaker direction.
- Add manual route override.
- Reject translator-owned streams.
- Reject translator endpoints as physical mic/sink candidates.
- Implement physical source/sink watcher with unplug/replug handling.
- Add acoustic feedback warning when active sink appears to be speakers.
- Classify headphone/open-speaker mode and block open-speaker duplex until Task 7 validates AEC.
- Expose typed routes and route candidates for Task 4 to publish through local API.

Acceptance:
- Allowlist discovers candidates but does not route multiple streams blindly.
- With two allowlisted sink-inputs, only the selected one moves to translator_remote_in.
- Production/manual routing never moves translator-owned sink-inputs to translator_remote_in; the authenticated Task 7 self-test exception does not exist yet.
- Daemon never selects translator_virtual_mic as physical mic.
- Daemon never captures physical default sink monitor in normal mode.
- Device loss enters device_unavailable and recovers only after valid physical device selection.

Validation:
- cargo test --workspace
- Routing tests with fake Pulse/PipeWire graph.
- Device watcher tests for default source/sink changes.
- Manual local smoke where practical.
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review routing loop prevention and selected-route semantics.

Final report:
- Changed files.
- Routing/device decisions.
- Validation evidence.
- Critic verdict.
```

## Task 4 Prompt

- [x] Completed

```text
Взять в работу Task 4: Daemon Runtime, Local API And Latency Policy.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect existing daemon/audio/ipc code.

Goal:
Make the daemon a user-session service with observable per-direction state.

Scope:
- Implement daemon entrypoint.
- Add HTTP control plus SSE status API on loopback; keep the bearer token in the Tauri Rust backend.
- Rotate %t/translator/control.token on daemon start/restart with mode 0600; Tauri Rust rereads it and frontend state never receives it.
- Reject missing/invalid bearer before body parsing; cap control bodies at 64 KiB and concurrent SSE subscribers at four.
- Add endpoints:
  - GET /v1/status
  - GET /v1/audio-graph
  - GET /v1/routes
  - GET /v1/routes/candidates
  - POST /v1/translation/start
  - POST /v1/translation/stop
  - PATCH /v1/directions
  - PATCH /v1/provider
  - PATCH /v1/latency-policy
  - PATCH /v1/voice-profiles
  - PATCH /v1/debug-capture
  - PATCH /v1/debug-text
  - POST /v1/routes/manual-override
  - POST /v1/self-test/round-trip/start
  - POST /v1/self-test/round-trip/stop
  - GET /v1/self-test/round-trip
  - GET /v1/events/stream
- Add typed round-trip self-test preconditions, lifecycle, checkpoints, per-leg latency and total physical_mic_onset_to_returned_ru_first_audible_ms; keep the VirtualPeer capability in the daemon and out of frontend state.
- Implement per-direction LatencyPolicyState.
- Add rolling p95 windows, minimum sample count, cooldown and hysteresis.
- Add fast degradation after three consecutive utterance breaches or queue lag above 500 ms for two seconds.
- Own daemon capture/playback queues, bounded by buffered milliseconds.
- Add privacy-safe structured logs.
- Bound each explicit debug-capture session to 10 minutes or 500 MiB, whichever comes first, with user-only file permissions.
- Use a dedicated 0700 state directory, O_NOFOLLOW | O_CREAT | O_EXCL, 0600 files and a 5 GiB free-space floor.

Acceptance:
- API binds only to loopback or user-owned Unix socket.
- Token rotates on restart, has mode 0600, and invalid bearer is rejected before payload parsing.
- Oversized bodies and a fifth SSE subscriber are rejected without affecting translation.
- Tauri/frontend cannot reach sidecar directly.
- Latency degradation affects only the direction whose metrics exceed thresholds.
- Recovery requires stable windows and cooldown.
- PATCH /v1/debug-text toggles debug text without enabling debug_capture.
- Disabling debug_text clears the bounded in-memory transcript/translation buffer.
- Logs contain route/latency events without spoken content.
- Debug capture stops at the hard bound and never survives restart.
- Round-trip self-test controls are authenticated, allow one five-minute session, reject open-speaker/active-route conflicts and expose no text unless debug_text is enabled.

Validation:
- cargo test --workspace
- API tests
- Latency policy tests with injected delays
- Privacy marker tests
- Debug-capture symlink, quota and simulated low-free-space tests
- Control-token rotation/permissions, pre-parse auth, request-size and SSE-subscriber limit tests
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review local API security and latency policy flapping risks.

Final report:
- Changed files.
- API and latency behavior.
- Validation evidence.
- Critic verdict.
```

## Task 5 Prompt

- [x] Completed

```text
Взять в работу Task 5: Python Sidecar Lifecycle And Mock Provider.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect provider contracts from Task 1 and daemon IPC from Task 4.

Goal:
Make provider IPC executable before real ASR/MT/TTS.

Scope:
- Implement sidecar process startup and authenticated connection.
- Implement open_session, audio_frame, cancel_utterance and close_session.
- Implement ProviderHealth, ProviderLatency and ProviderError.
- Add sidecar-owned queue bounds:
  - provider input queue max 800 ms
  - provider output queue max 1200 ms
- Carry session_id and event sequence on every provider event; reject stale, duplicate and post-final events.
- Enforce a 2-second close acknowledgement deadline and supervised restart on timeout.
- Cancel all work older than 3000/2000/1000 ms in Quality-first/Balanced/Streaming-first and never play expired audio.
- Add mock provider with deterministic fixture audio transform.
- Add injected latency and error modes.
- Emit transcript/translation deltas only when debug_text_enabled=true.
- Prove normal mode never exposes debug text marker in status/logs/provider errors.

Acceptance:
- Daemon can open two independent provider sessions.
- Sidecar rejects unauthenticated connections.
- Queue overflow produces typed error or cancellation, not unbounded growth.
- Debug text marker never appears in normal status/logs/provider errors.
- Mock provider supports synthetic duplex tests.

Validation:
- Python unit tests
- Rust/Python integration contract tests
- Synthetic PCM fixture tests
- Privacy marker scan
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review sidecar auth, lifecycle, queue/backpressure and debug_text privacy gate.

Final report:
- Changed files.
- Contract behavior.
- Validation evidence.
- Critic verdict.
```

## Task 6 Prompt

- [x] Completed

```text
Взять в работу Task 6: Local Provider ASR, MT And TTS MVP-A.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect /home/anton/Source/uncle-freud-bot/uncle_freud/services/voice.py and tests/test_voice.py as reference only.
- Verify existing model paths:
  - /home/anton/.cache/huggingface/hub/models--Systran--faster-whisper-small
  - /home/anton/Source/uncle-freud-bot/.data/faster-whisper/models--Systran--faster-whisper-large-v3
  - /home/anton/Source/uncle-freud-bot/.data/piper-voices/ru_RU-dmitri-medium.onnx
  - /home/anton/Source/uncle-freud-bot/.data/piper-voices/en_US-ryan-medium.onnx
- Do not download models during startup.

Goal:
Implement the first real local provider chain.

Scope:
- Integrate ASR through existing local faster-whisper/CTranslate2 caches first.
- Use local_files_only or equivalent no-download behavior for ASR/TTS smokes.
- Select one CTranslate2-compatible Ru <-> En MT model only after confirming no local MT cache exists.
- Before any MT download, record model name, source, expected disk size and cache path.
- Add/update models/manifest.json and enforce the 2 GiB incremental-download budget, 20 GiB free-space floor, checksum and license fields.
- Integrate existing Piper voices ru_RU-dmitri-medium and en_US-ryan-medium.
- Treat Qwen3 TTS in uncle-freud-bot as non-MVP fallback only.
- Implement VoiceProfile male/female presets per target language.
- Bootstrap with existing male Piper presets; select and download exactly one female Russian and one female English Piper-compatible voice after recording candidates, licenses, checksums and disk cost.
- Implement Quality-first, Balanced and Streaming-first provider behavior.
- Implement stable-prefix commit so TTS receives only non-revisable translation prefixes.
- Benchmark cold start, warm first audio, throughput, CPU/RAM, GPU/VRAM peak and simultaneous duplex for local small and large ASR candidates.
- Keep only the selected ASR model resident in normal operation and test typed CUDA OOM fallback.
- Enforce <=10 GiB peak VRAM in normal simultaneous duplex.
- Add a versioned corpus with 10 excluded warmups and at least 100 measured utterances per direction; compute chrF2 and synthesized-output WER.
- Keep source and translated text out of logs and normal UI.
- Add CPU fallback or explicit unsupported-state handling for missing CUDA dependencies.

Acceptance:
- Existing local ASR caches are used for first smokes without downloads.
- Existing Piper voices are used for first TTS smokes.
- Any new model download is preceded by written inventory and disk-cost note.
- Short Russian fixture translates to English audio.
- Short English fixture translates to Russian audio.
- Two directions run concurrently with isolated sessions.
- Male and female presets work for both target languages; unavailable or silent gender fallback fails MVP-A.
- Provider emits latency and health without content leaks.
- Local provider receives provider-level latency/resource characterization for both directions and simultaneous duplex; release classification waits for Task 7 graph-boundary measurements.
- Corpus chrF2 is >=45 per direction, critical negation/number/name subset has no meaning-changing errors, and synthesized-output WER is <=15%.

Validation:
- Python local provider tests with short fixtures
- GPU availability check
- No-download ASR/TTS smoke using existing paths
- Synthetic local-provider duplex benchmark
- Model manifest and disk/VRAM budget report
- Privacy marker scan
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review model reuse, download discipline, latency and privacy.

Final report:
- Changed files.
- Models reused and any new model disk-cost note.
- Validation evidence.
- Critic verdict.
```

## Task 7 Prompt

```text
Взять в работу Task 7: Audio Capture/Playback Integration And Synthetic Duplex Benchmark.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect audio graph, routing, daemon and sidecar code.
- Confirm virtual endpoints can be created.

Goal:
Connect real audio graph streams to provider sessions and prove duplex path without meeting apps.

Scope:
- Capture physical mic stream for outgoing direction.
- Capture translator_remote_in.monitor for incoming direction.
- Play translated mic output to translator_mic_out.
- Play translated speaker output to validated physical sink.
- Implement resampling/downmixing.
- Own capture and playback queues, bounded to 400 ms each per direction.
- Own VAD/EOU decisions in the Rust capture path, keep stream_id stable for the provider session, rotate utterance_id at each EOU boundary, and make the sidecar consume capture-issued EOU as authoritative input.
- Production-wire ProviderStreamCoordinator and the provider watchdog for both directions; purge queued playback before cancellation and apply bounded close/restart escalation.
- Implement PipeWire WebRTC AEC for open-speaker mode using translated incoming playback as reference; headphones remain the fallback.
- Add synthetic benchmark runner.
- Add live human round-trip mode: capture physical mic, tap the exact English PCM from Translator_Virtual_Mic, finish monitoring it in headphones, then reinject the same frames through one session-bound VirtualPeer sink-input.
- Make the live self-test mutually exclusive with real incoming routes and open-speaker mode; bind its routing exception to the daemon-held `{session_id, stream_serial_or_node_id, process_identity}` tuple.
- Reject the same VirtualPeer in production/manual profiles and reject forged/stale translator.test_profile metadata; authorize it only during its matching self-test session.
- Prove exact-PCM reuse in memory with matching format, frame count, monotonic frame sequence and rolling hash; do not persist PCM for this proof.
- Emit stage checkpoints plus outgoing-leg, incoming-leg and total round-trip latency; keep transcript/translation text behind debug_text.
- Bound the self-test to five minutes and implement idempotent teardown for stop, timeout, failure and daemon restart.
- Measure graph-boundary speech_onset_to_first_audible_ms in both directions.
- Add digital-loop and acoustic residual-echo assertions.
- For AEC, record device geometry/volume, run the -20 dBFS fixture, calculate ERLE and check far-end-only false triggers.
- Add cold, warm and 30-minute simultaneous-duplex profiles with resource/queue/drop/restart telemetry.
- Use 10 excluded warmups and at least 100 measured utterances per direction; report timeout/drop rate separately.

Acceptance:
- Synthetic loopback runs both directions concurrently.
- On this workstation, a headphone-mode live run lets the user speak Russian, hear the complete exact outgoing English virtual-peer tap, then hear its Russian translation through the normal incoming path.
- Live self-test recursion count is zero, no original Russian reaches the virtual microphone, and teardown leaves no VirtualPeer stream or route change.
- Exact-PCM evidence matches format, frame count, monotonic sequence and rolling hash between capture and reinjection without writing PCM to disk.
- speech_onset_to_first_audible_ms, capture_to_first_audio_ms, capture_to_last_audio_ms, queue lag and provider latency are recorded per direction.
- Timeout/drop rate is reported separately and stays below 1%.
- No unauthorized translator-owned stream re-enters incoming capture; only the exact daemon-authorized VirtualPeer capability tuple is permitted during its matching self-test session.
- Logs contain no speech content.
- Quality-first starts first and degrades only when injected latency requires it.
- Open-speaker mode remains disabled if AEC setup, reference routing or attenuation fails.
- Open-speaker mode requires median ERLE >=15 dB and zero outgoing VAD/translation triggers in the 60-second far-end-only run.
- Assign final meets_target, usable_degraded or fails_usable_limit classification from graph-boundary latency, quality and resource evidence.

Validation:
- cargo test --workspace
- Python tests
- Audio graph smoke
- Synthetic duplex benchmark report
- Live human round-trip report with checkpoints, exact-PCM counters/hash, audible-output confirmation, per-leg/total latency and before/after graph diff
- Privacy marker scan
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review duplex graph safety, latency evidence and privacy evidence.

Final report:
- Changed files.
- Benchmark results.
- Validation evidence.
- Critic verdict.
```

## Task 8 Prompt

- [x] Completed

```text
Взять в работу Task 8: Tauri Tray And Status Page MVP.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect daemon local API and current Tauri scaffold.

Goal:
Provide the desktop control surface required for live use.

Scope:
- Implement Tauri tray/menu.
- Implement first screen as status page.
- Show audio graph, physical mic/sink, route candidates, selected route, provider health, per-direction latency p50/p95, mode, degradation reason, privacy/cloud/debug state and voice presets.
- Add controls for start/stop, directions, provider, voice gender, manual route override and debug_capture.
- Add a Diagnostics control for round-trip self-test start/stop, precondition failures, current checkpoint and per-leg/total latency.
- Add separate debug_text control backed by PATCH /v1/debug-text.
- Keep transcript/translation text hidden unless debug_text is enabled.
- Show visible tray/status warning while debug_text is enabled.
- Store debug text only in a bounded in-memory ring buffer of at most 200 events and 1 MiB.
- Clear debug text buffer on session stop, provider switch, daemon restart and UI close.
- Prove debug text is never logged, persisted, sent to telemetry, included in debug-capture metadata or embedded in provider errors.
- Add warnings for cloud provider, acoustic feedback and debug text.

Acceptance:
- Tauri reconnects to an already running daemon.
- First screen is status, not a landing page.
- UI cannot display transcript/translation text in normal mode.
- debug_text is controlled separately from debug_capture.
- Toggling debug_text does not enable debug_capture.
- debug_text warning is visible while enabled.
- debug text buffer is memory-only, bounded and cleared on stop/provider switch/restart/UI close.
- Privacy marker tests prove transcript/translation text does not appear in logs, local storage, telemetry, debug-capture metadata or provider errors.
- Cloud provider cannot be enabled without visible opt-in.
- Manual route override selects one incoming route.
- Controls reflect daemon state after restart.
- Round-trip self-test requires explicit start, presents Stop while active, never exposes text outside debug_text, and reports teardown completion.

Validation:
- Tauri unit/component tests
- Screenshot checks for desktop status page
- Local tray/status smoke
- Local live round-trip UI smoke in headphones
- Privacy marker scan
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review UI control surface, privacy gates and routing controls.

Final report:
- Changed files.
- UI states and screenshots.
- Validation evidence.
- Critic verdict.
```

## Task 9 Prompt

- [x] Completed
- [x] Terminal lifecycle command is `scripts/translator-desktop up|down|restart|status|logs`; `scripts/translator-desktop install|up` installs the user-local `~/.local/bin/translator` command, so `translator up|down|restart|status|logs` works from any current directory. When `target/release/translator-daemon` or `target/release/translator-ui` exists, the lifecycle refreshes the matching `~/.local/bin` binary before launch. `up`, `start` and `restart` wait briefly for the daemon HTTP API, then start the current-session `translator-ui` tray/status process when a graphical desktop session is available. Stop/restart paths run daemon-owned audio graph cleanup after stopping the user unit.

```text
Взять в работу Task 9: User-Level systemd And Desktop Run Mode.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect daemon startup, audio graph cleanup and Tauri reconnect behavior.

Goal:
Run the service as a user-session desktop service.

Scope:
- Finalize systemd/translator.service.
- Add install/start/stop/status scripts.
- Ensure daemon starts in user PipeWire session.
- Install an XDG autostart entry for Tauri. Tauri reports service state but does not manage the systemd unit.
- Add restart/backoff behavior.
- Configure RuntimeDirectory=translator, mode 0700, KillMode=control-group and bounded Restart=on-failure.
- Journal daemon-owned module ids and selected streams' original sinks under the user runtime directory.
- Restore app streams before unloading only journaled daemon-owned modules on stop/SIGTERM.
- Add install, disable and uninstall flows that leave unrelated PipeWire state unchanged.

Acceptance:
- systemctl --user start translator.service starts daemon.
- systemctl --user stop translator.service stops daemon and cleans daemon-owned endpoints.
- Tauri reconnects after daemon restart.
- debug_capture and debug_text are disabled after restart.
- Selected app streams return to their original sinks on stop/crash recovery.
- Forced daemon failure leaves no sidecar, rotates control token and reconciles endpoints before accepting a new session.
- No root/system service is required.

Validation:
- systemd user smoke
- Audio endpoint cleanup smoke
- Tauri reconnect smoke
- Forced daemon crash/orphan/token-rotation smoke
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review user-session boundaries and cleanup safety.

Final report:
- Changed files.
- systemd commands and results.
- Validation evidence.
- Critic verdict.
```

## Task 10 Prompt

- [x] Autonomous evidence captured in `docs/benchmarks/task10-validation-report.json`; earlier blocked live second-endpoint status is superseded by live user-confirmed Meet and Telegram duplex checks.
- [x] Synthetic Telegram/Meet/Zoom app-stream diagnostic captured in `docs/benchmarks/task10-simulated-app-streams-report.json`; live Meet and Telegram evidence is captured separately in `docs/benchmarks/task10-meet-live-duplex-check.json` and `docs/benchmarks/task10-telegram-live-duplex-check.json`.
- [x] MVP-A local-provider routing/privacy gate is satisfied for Meet and Telegram; Task 7 graph-boundary latency still classifies the local provider as `fails_usable_limit`, so Task 11 provider comparison remains mandatory.

```text
Взять в работу Task 10: Real-App MVP-A Smokes Telegram And Meet.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md and docs/planning/translator-live-duplex-tasks.md.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Verify MVP-A prerequisites from Tasks 1-9.
- Do not start OpenAI/cloud provider.

Goal:
Prove MVP-A against Telegram Desktop and Firefox/Chromium Meet.

Scope:
- Validate Telegram Desktop incoming route discovery and selection.
- Validate Firefox/Chromium Meet route discovery and selection.
- Validate Translator_Virtual_Mic as call app microphone.
- Validate translated-only behavior in both directions.
- Collect latency and privacy evidence.
- Run at least one Telegram or Meet smoke against a real second endpoint with simultaneous overlapping Ru/En speech and receiving-side recording.
- Record setup steps and known limitations.

Acceptance:
- Telegram Desktop incoming audio can be selected, translated and played to physical sink.
- Firefox/Chromium Meet incoming audio can be selected, translated and played to physical sink.
- Outgoing translated microphone reaches call app through Translator_Virtual_Mic.
- Remote endpoint receives outgoing translation while local endpoint simultaneously receives incoming translation.
- Original audio is not mixed by default.
- Synthetic and real-app logs contain no spoken content.
- MVP-A gate is satisfied.
- Report the local provider latency classification; if it misses the usable-release limit, Task 11 becomes mandatory.

Validation:
- Telegram Desktop smoke report
- Firefox/Chromium Meet smoke report
- Latency ledger report
- Privacy marker scan
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review MVP-A evidence and whether OpenAI/cloud remained out of path.

Final report:
- Smoke setup.
- Latency results.
- Privacy evidence.
- Critic verdict.
```

## Task 11 Prompt

- [ ] OpenAI adapter preflight/opt-in evidence captured in `docs/benchmarks/task11-openai-adapter-preflight.json`; MVP-A live route debt is closed, credentialed OpenAI model/websocket smoke passed, synthetic duplex comparison passed, and daemon/sidecar OpenAI runtime wiring is covered by deterministic fake-WebSocket tests; real-app OpenAI smoke remains pending.

```text
Взять в работу Task 11: MVP-B OpenAI Realtime Translation Adapter.

Precondition:
- MVP-A functional, routing and privacy gates passed and documented. Local-provider latency may miss the usable-release limit.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md, docs/planning/translator-live-duplex-tasks.md and MVP-A evidence.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Do not create or expose API keys in code or logs.

Goal:
Add OpenAI Realtime Translation as second provider adapter without changing audio routing.

Scope:
- Add OpenAI provider adapter in Python sidecar.
- Use same provider lifecycle and events.
- Require explicit cloud provider enablement.
- Load credentials from ignored config or OS secret storage.
- Show audio_leaves_machine=true in status before first cloud session.
- Map translated audio and transcript deltas through debug-gated provider events.
- Add provider comparison latency report.

Acceptance:
- OpenAI adapter cannot start until enabled.
- Missing credentials produce safe error without network session.
- Cloud enabled state is visible before first audio leaves the machine.
- Local provider remains default.
- OpenAI passes synthetic duplex comparison and at least one real-app route smoke.
- Combined evidence states whether at least one provider meets the usable-release latency gate.

Validation:
- Provider contract tests
- Cloud opt-in/privacy tests
- Synthetic OpenAI comparison smoke
- One real-app OpenAI smoke
- Privacy marker scan
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review cloud opt-in, credential handling, provider parity and privacy evidence.

Final report:
- Changed files.
- Cloud opt-in and credential behavior.
- Latency comparison.
- Critic verdict.
```

## Task 12 Prompt

- [x] Zoom routing allowlist/preflight diagnostic captured in `docs/benchmarks/task12-zoom-diagnostic-report.json`; the earlier route-only status is superseded by full live Zoom duplex acceptance.
- [x] Live Zoom conference check captured a `ZOOM VoiceEngine` candidate and `Translator_Virtual_Mic`; daemon/manual route selection now falls back to `route_method=pipe_wire_links` after Zoom rejects `pactl move-sink-input`; evidence in `docs/benchmarks/task12-zoom-live-route-check.json`.
- [x] Zoom microphone selection was validated by moving the Zoom source-output to `translator_virtual_mic`, observing `translator_virtual_mic:capture_MONO -> ZOOM VoiceEngine:input_MONO`, and restoring it to `alsa_input.usb-Jieli_Technology_UACDemoV1.0-00.mono-fallback`.
- [x] Live incoming Zoom translation from Russian to English reached the physical headphones after the `translator-incoming-playback` stream volume was corrected from `0%` to `100%`; user-confirmed evidence in `docs/benchmarks/task12-zoom-live-translation-check.json`.
- [x] Full Task 12 Zoom duplex acceptance is user-confirmed after the outgoing and incoming volume/routing fixes; Task 7 latency and Task 11 OpenAI comparison debts are still carried.

```text
Взять в работу Task 12: Post-MVP Zoom Acceptance.

Precondition:
- MVP-A and MVP-B foundations are stable.

Repo:
- /home/anton/Source/translator

Startup:
- Read docs/planning/translator-live-duplex-prd.md, docs/planning/translator-live-duplex-design.md, docs/planning/translator-live-duplex-tasks.md and previous smoke reports.
- Read repo-c4.json, architecture_fidelity, tool_orchestration and relevant indexed components; verify selected files directly.
- Inspect current routing watcher.

Goal:
Validate Zoom without weakening selected-route semantics.

Scope:
- Add Zoom to allowlist candidate detection.
- Validate Zoom microphone selection with Translator_Virtual_Mic.
- Validate Zoom incoming sink-input discovery/selection.
- Add setup notes for Zoom audio settings.

Acceptance:
- Zoom route selection works without routing unrelated streams.
- Zoom outgoing translation reaches remote side through virtual microphone.
- Zoom incoming translation reaches physical sink.
- Known Zoom limitations are documented.

Validation:
- Zoom smoke report
- Latency and privacy evidence
- Run full-depth repo-c4-scan and validate repo-c4.json.

Critic:
- Ask a critic to review Zoom-specific routing risk and whether MVP invariants held.

Final report:
- Changed files.
- Zoom setup and results.
- Residual limitations.
- Critic verdict.
```
