# Translator stabilization master BDD contract

- Baseline: untouched `9291e8beafee3e02aaa179178ce460ac9e6c6de2`.
- Candidate: exact commit built from `codex/stabilization-20260904` after all gates pass.
- Product authority: tracked public documentation plus the intentionally unpublished PRD/design at `e40156a`; removed benchmark artifacts are historical evidence, not current proof.
- A response is successful only when reported state equals effective runtime state at the response boundary.
- Missing live prerequisites are `UNAVAILABLE`, never `PASS`.

## Canonical decisions

| Mode | first-audio breach | p95 queue breach | maximum source age |
| --- | ---: | ---: | ---: |
| Quality-first | `>3000 ms` | `>500 ms` | `3000 ms` |
| Balanced | `>2000 ms` | `>350 ms` | `2000 ms` |
| Streaming-first | `>1000 ms` | `>250 ms` | `1000 ms` |

- Equality does not breach. External audible latency is an eval measurement, not a segmenter claim.
- The mandatory local-provider release matrix is all three modes x both audio directions x both language pairs x both required target voices, including every fallback allowed to report operational. Removing or hiding any baseline local core cell blocks release. The release manifest binds a provider-first-frame p99 for every matrix cell. Its processing reserve is that cell's p99 rounded up to a 20-ms frame plus 100 ms; when more than one fallback can be chosen after capture, the reserve uses the conservative maximum across those reachable cells. The forced capture cutoff is `maximum source age - processing reserve`. Failure of a mandatory local cell blocks release; only explicitly optional capabilities such as OpenAI and validated open-speaker AEC may be omitted and documented.
- Release classification is a separate immutable contract: graph-boundary p95 `<=1000 ms` is `meets_target`, `(1000,1500] ms` is `usable_degraded`, and `>1500 ms` is `fails_usable_limit`.
- Starting with both directions disabled is invalid. Incoming-only opens no microphone and requires a validated physical sink. Outgoing-only may use the direct physical mic only with headphones; open speakers require matching validated AEC. Simultaneous duplex requires headphones or matching validated AEC. An unsafe path never falls back to a direct physical mic/sink.
- `degraded` is operational only when every model required by the selected provider/direction is `ready`, no safe error is active, and degradation identifies a measured slower compute fallback. `not_loaded`, `loading`, or `failed` is not operational.
- Debug capture is an independently enabled session. Translation stop and provider replacement close no capture artifact and do not erase one; stopped translation appends no PCM. Explicit disable, daemon restart, time/size/free-space failure close it. Debug text has the different lifecycle stated below.
- Provider-engine replacement prepares B before the registry lock, atomically swaps the active pointer once under that lock, and retires A afterward. A lease whose locked pointer read completes before the swap receives A; a read completing after the swap receives B.
- A deterministic text repair may modify only an unambiguously aligned semantic span. Ambiguity leaves the model output unchanged; no repair may delete, move, or duplicate a negation, number, identifier, name, or its grammatical role.

## Deterministic validation and provider protocol

### TST-1 — Clean-checkout suite fidelity

Given a clean checkout without ignored evidence, credentials, or model cache, the complete sidecar pytest and root unittest collections run. A tracked `tests/external-prerequisites.toml` is the only skip allowlist and names the exact framework plus full node/test ID and one stable `missing_external_prerequisite:<name>` code; its allowed codes are `physical_audio_graph`, `real_call_endpoint`, `openai_credentials`, `gpu_model_cache`, and `private_human_evidence`. Artifact parsers and scoring oracles run against tracked synthetic fixtures and never require removed/private reports. Tracked collection manifests fail on an unexpected skip/xpass, node-set change, wildcard entry, duplicate ID, or reclassification of a deterministic test. The runner records setup/call/teardown outcomes, rejects collection skips/errors, deselection, ambiguous phase shapes, and missing execution, and requires every manifest-deterministic ID to pass and every external ID to skip with its exact code.

The Python collectors run under isolated `-I` startup and each emits exactly one canonical typed receipt on stdout with empty stderr. Pytest runs with `-c /dev/null`, an exact sidecar test root, disabled third-party plugin autoload, and independent source-file/collected-file parity. It accepts only native `pytest.Function` items: pytest-collected `unittest.TestCase`, doctest or custom plugin items, repository `conftest.py` hooks, xunit-style setup/teardown, dynamic string `skipif`, and `filterwarnings` configuration/markers are rejected. Every conditional-marker stack is fully validated before its first active marker is selected, so an invalid condition cannot be hidden by decorator order. Native pytest fixtures and finalizers remain owned by the pinned pytest engine; setup/call/teardown plus actual callable-body start/completion are reconciled without replacing the callable or changing `request.function` identity. Active boolean `skipif` takes pytest precedence over a plain skip when their exact skip reason is recorded.

The root suite separately validates the exact stdlib `unittest` runner, `TestSuite` dispatch helpers, external result journal, and synchronous/asynchronous `TestCase` lifecycle before, during, and after execution. Every non-skipped unittest leaf proves both `run` and body start/completion and returns `None`; every unittest instance, class, and module fixture and cleanup must execute to completion through the trusted protocol. Standard cached add APIs, manual or reentrant cleanup drains, repeated class/module segments in a standard suite, cooperative inheritance, and one-time descriptor binding retain stdlib semantics. Module-level `load_tests` hooks are forbidden because they can filter otherwise discoverable tests before the exact node manifest is formed. Scalar cleanup/fixture returns are allowed, while an unexecuted generator, async generator, coroutine/awaitable, iterator, or context manager fails closed. Class/instance/suite/result protocol replacement, cleanup-registry loss or injection, stale or mutated body/fixture descriptors, a generator or async-generator body, a coroutine on plain `TestCase`, a custom `TestSuite`, or mutation of a later leaf/helper also fails closed. A declared static skip retains its stdlib fixture and pending-cleanup behavior; it is exempt only from callbacks that stdlib itself would not invoke.

Every unhandled pytest framework warning delivered during collection or execution, including an uncollected test class or a test returning lazy assertions, invalidates the run. Unhandled `ResourceWarning` and `RuntimeWarning`, unraisable exceptions, unhandled thread exceptions, live threads, pending asyncio tasks, and any extra output or failure raised during interpreter shutdown invalidate collection or execution receipts. Tests may use an explicit local warning-capture assertion, but no surviving hook/filter mutation or repository-wide suppression can change the gate policy. The outer runner never forwards child stdout or stderr; after validating diagnostics, state, exit status, and the exact canonical receipt, it emits only that receipt.

The receipt binds the manifest and sorted verified outcomes by SHA-256; the parent independently requires it even after a zero child exit. The parent and child each bind one manifest/allowlist byte snapshot and fail if either file changes during execution. The runner also compares a full repository snapshot after every gate, including successful non-Python gates. CI executes the same full collections without `continue-on-error` or file selectors.

The gate is an integrity check for an exact independently reviewed source tree, not a same-user sandbox for intentionally malicious Python code. Its trust root is the immutable candidate tree, post-commit publication provenance, clean hosted runner, and independent source review. A receipt alone never establishes that hostile repository code was safely contained; release evidence is valid only when all of those trust-boundary checks refer to the same tree.

Owner/seam: repository validation contract; clean temporary clone plus CI workflow contract test.

### TST-2 — TTS continuation and event order

Given natural-silence and forced-continuous-voice boundaries, when local TTS is invoked, then the adapter receives `continuation=false` and `continuation=true`, respectively. Given a successful utterance after `session_opened` and operational `health`, provider event sequence starts at `1` and strictly increases; audio sequence is contiguous `0..n`, followed by latency and completed final with `final_audio_sequence=n`. Only cancelled preallocated work may leave an event-sequence gap. With debug text off, no transcript, translation, or marker text reaches events or logs.

Owner/seam: provider contract and local-provider adapter; authenticated in-process gRPC duplex test using the production signature.

### TST-3 — Provider replacement leases

Given live leases on provider A and a fully prepared provider B, when the registry performs its one locked pointer swap, then a lease whose locked read completed before the swap receives A and a read completed afterward receives B. Each lease remains wholly on its selected provider, A remains alive until its last lease closes, and A shuts down exactly once. Failed B preparation performs no swap, leaves A active, and closes every partially created B resource exactly once within `2000 ms`. Across 100 injected failures there is zero net session, socket, task, process, file-descriptor, or model-lease growth.

Owner/seam: provider registry/lease owner; concurrent registry test with instrumented providers.

### TST-4 — Exact deterministic source gates

Given a clean checkout and the committed lockfiles, `./scripts/translator-validate deterministic` executes these exact command contracts with no warning suppression:

```text
(cd sidecar && uv sync --locked --all-groups)
(cd apps/translator-ui && bun install --frozen-lockfile)
(cd apps/translator-ui && bun test src)
(cd apps/translator-ui && bun run build)
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
RUST_TEST_THREADS=1 cargo test --locked --workspace --all-targets
RUST_TEST_THREADS=1 cargo test --locked --workspace --doc
sidecar/.venv/bin/ruff check --config sidecar/pyproject.toml sidecar/translator_sidecar sidecar/tests tests scripts/translator-browser-live-stream-smoke scripts/translator-simulated-app-stream-smoke scripts/translator-task10-real-app-smoke scripts/translator-task11-openai-preflight scripts/translator-task11-openai-synthetic-smoke scripts/translator-task12-zoom-diagnostic scripts/translator-publication-check scripts/translator-test-manifest scripts/translator-validate
sidecar/.venv/bin/ruff format --check --config sidecar/pyproject.toml sidecar/translator_sidecar sidecar/tests tests scripts/translator-browser-live-stream-smoke scripts/translator-simulated-app-stream-smoke scripts/translator-task10-real-app-smoke scripts/translator-task11-openai-preflight scripts/translator-task11-openai-synthetic-smoke scripts/translator-task12-zoom-diagnostic scripts/translator-publication-check scripts/translator-test-manifest scripts/translator-validate
sidecar/.venv/bin/python -I -W error::ResourceWarning -W error::RuntimeWarning scripts/translator-test-manifest _run-pytest
sidecar/.venv/bin/python -I -W error::ResourceWarning -W error::RuntimeWarning scripts/translator-test-manifest _run-unittest
shellcheck scripts/translator-asr-quality-debug scripts/translator-desktop scripts/translator-podcast-quality-debug scripts/translator-publication-check.bash scripts/translator-sca scripts/translator-schema-check scripts/translator-task9-smoke
/usr/bin/systemd-analyze verify systemd/translator.service
./scripts/translator-sca
./scripts/translator-schema-check
./scripts/translator-publication-check candidate
./scripts/translator-test-manifest check
```

A tracked gate manifest binds each exact argv contract and expected Rust/Python/UI test-file/node counts. Adding, removing, renaming, or omitting a test/source gate without updating and reviewing that manifest fails. Hosted CI calls this same entrypoint rather than maintaining a second command list.
Every manifest gate has the exact `900`-second timeout. It runs in its own
process group under a Linux child subreaper. Before every gate, while holding
the process-owner lock, the runner requires exactly its own Linux task, no
direct child, and the default `SIGCHLD` disposition; it fails before `prctl` or
spawn if that exclusive-owner admission is not satisfied. Timeout, SIGTERM
cancellation, caller interruption, or a successful leader that leaves any
descendant behind, including one that created another session/process group,
terminates and reaps the exact owned process tree before the validator returns.
While the gate child is still running, the parent non-blockingly reaps exited
adopted children by exact PID while excluding only the direct gate child. It
does not signal a live adopted child, and repeated polling consumes one absolute
gate deadline rather than restarting the timeout. Real process-lifecycle tests
run one method per clean `-I` worker behind a clean `-I` supervisor and require
an exact receipt for one clean pass: skips, expected or unexpected successes,
failures, errors, and non-`None` or lazy body results are rejected. Before the
worker can start, the supervisor restores and unblocks `SIGTERM`, arms Linux
parent-death signalling, and verifies that its expected parent did not change.
Default `SIGTERM` of the outer host therefore drives the existing gate-owner
cleanup instead of orphaning the supervisor. Timeout cleanup stops the
discovered tree to a stable fixed point before kill/reap; a live negative
control forks a new detached descendant after a stopped-root snapshot and
requires the next fixed-point scan to discover and remove it. If external
`/proc` inspection itself fails, cleanup still attempts to kill/reap the known
root before reporting the inspection failure, but does not claim proof about
an undiscoverable detached process. A static invocation audit keeps every test
that reaches the process-tree owner on this isolated seam, and each proxy
proves that the shared test host's subreaper state, `SIGCHLD` disposition,
direct-child set, and Linux task set are unchanged.
The public validator and publication launchers pin `/usr/bin/python3 -I`; the
publication launcher invokes its shell policy only through sanitized
`/usr/bin/bash -p`. The runner builds each child environment from an explicit
allowlist: only `HOME`, `PATH`, named validation-tool overrides, declared
per-gate values, and fixed locale, colour, timezone, and disabled pytest
third-party-plugin-autoload values survive.
Ambient credentials, pytest/plugin controls, language startup variables, and
all other caller variables are absent. A fake `bash`/`python3` at the front of
`PATH`, `BASH_ENV=exit 0`, a `PYTHONPATH` `sitecustomize.py`, or
`PYTEST_ADDOPTS=--ignore=...` cannot turn invalid arguments or an incomplete
collection into success. Tools intentionally resolved from `PATH` remain an
input of the clean hosted trust boundary and must pass their separately pinned
version/integrity controls.

The supply-chain gate accepts only cargo-audit 0.22.2, Bun 1.3.12, and
OSV-Scanner 2.5.0 executables whose SHA-256 equals the digest pinned in both the
gate and hosted CI. Each executable is opened once without following a symlink,
must be a singly linked executable regular file owned by the current user or
root, must have no group/other write or set-user-ID/set-group-ID bit and no
`security.capability` xattr, and is invoked through its held file descriptor.
Its path, descriptor metadata, security capability state, and content digest
must remain identical after version validation and every scan; an atomic path
replacement or privilege-bearing executable therefore fails closed without
running the replacement. OSV directly scans
`Cargo.lock`, `sidecar/uv.lock`, and `apps/translator-ui/bun.lock` on every run,
including after a Bun timeout. No dynamically resolved Python audit tool or
unlocked tool environment is part of the release trust boundary.
After checkout, hosted CI creates a local symbolic branch at the exact already
checked-out event commit before validation. For pull requests this preserves the
reviewed merge commit rather than switching to an unreviewed head SHA, while
satisfying the publication gate's no-detached-release-candidate invariant.
Checkout credential persistence is disabled before any candidate-controlled
code runs, so the read-only workflow token is absent from repository Git config.

Rust doctests are explicitly forbidden until the collection manifest has a
stable cross-crate doctest-ID inventory. The gate still runs the locked
workspace doc-test command, and manifest generation fails if rustdoc lists any
doctest, so a future example cannot silently fall outside the exact inventory.

Owner/seam: one repository validation entrypoint using native Cargo/uv/Bun/ShellCheck/systemd tools; clean-clone parity test.

### PROTO-1 — Reject invalid wire traffic with scoped recovery

- Missing/wrong authentication is rejected with `UNAUTHENTICATED` before payload parsing and performs zero session/state mutation.
- A malformed initial open is rejected with `INVALID_ARGUMENT` and creates no provider session. A malformed in-session command purges and closes only its already-associated session/direction; the peer continues.
- A byte-identical duplicate, stale event, superseded-session event, or post-final event is discarded, increments a content-free technical counter, and does not destroy a healthy session. Reuse of the same session/event/audio identity or sequence with a different payload is equivocation: cancel and purge the affected utterance, close its provider session, and perform bounded affected-direction recovery.
- Wrong session/direction/stream identity, malformed PCM, an audio-sequence gap, or terminal order violation cancels and purges the affected utterance, closes the affected provider session, and attempts a bounded direction restart; the peer direction continues.
- A close acknowledgement exceeding exactly `2000 ms` terminates and reaps the shared sidecar generation, purges both direction queues, establishes a fresh authenticated generation, and only then permits replacement sessions.
- Every case exposes only a bounded safe code; no text, audio, token, credential, or raw provider payload enters the response or log.

Owner/seam: gRPC authentication/contract validator and daemon direction session; transport integration tests plus state-machine tests.

### PROTO-2 — Private sidecar IPC generation

Given each daemon-started sidecar generation, the daemon creates `$XDG_RUNTIME_DIR/translator` as a real user-owned directory with mode `0700` and the sidecar creates `sidecar.sock` as a real Unix-domain socket owned by the same UID with mode `0600`. The daemon generates a fresh unpredictable token for every sidecar generation and transports it only through an inherited environment value or inherited file descriptor; it never enters argv, frontend/Tauri state, status, errors, logs, telemetry, or persistent files. Every gRPC request authenticates before parsing or session mutation. Tauri and browser/frontend code have no sidecar address, token, filesystem capability, or direct connection path and can reach it only through the authenticated daemon control boundary.

On startup, every path component is checked without following symlinks. The daemon may unlink an existing socket only after proving it is a same-UID socket and no live sidecar accepts the current authenticated handshake; a foreign-owned path, non-socket, symlink, unverifiable owner, or responding sidecar fails closed without unlink. Restart rotates the token and socket generation; an old token or connection cannot authenticate to the new generation. Mode/owner boundaries, stale/foreign/symlink cases, token rotation, `/proc` argv inspection, and a Tauri capability test are mandatory.

Owner/seam: daemon `SidecarGeneration` owns token/socket lifecycle; sidecar gRPC listener owns mode and authentication; Tauri remains a daemon-only proxy.

### ACC-1 — Semantics-preserving MT repair

Given multiple clauses, repeated entity labels, multiple times or identifiers, and negation before one of those spans, a postprocessor changes only the uniquely aligned defective span and preserves clause order and the exact multiset and roles of all critical values. In particular:

```text
source:    Do not rename the document; open document Hotel.
candidate: Документ не нужно переименовывать; откройте документ отель.
accepted:  Документ не нужно переименовывать; откройте документ Hotel.
```

It must never produce `документ Hotel нужно переименовывать`. If alignment is not unique, the candidate is returned unchanged and the ambiguity is recorded only as a content-free metric. The same invariant holds for `After 1:15, meet at 13:15`, repeated order identifiers, inflected Cyrillic names, and their RU→EN counterparts.

Owner/seam: local MT postprocessor; exact regression table plus generated metamorphic cases that insert neutral clauses, reorder unrelated clauses, duplicate labels, and vary case/punctuation.

### ACC-2 — Critical oracle covers generated and audible text

Given each critical negation, number, identifier, name, and role case, the quality gate evaluates MT output and the independently transcribed audible TTS output as separate layers. Either layer's violation fails the case and appears with its layer; the review hash commits to both strings and both verdicts. Replacing only the audible transcript of `do-not-mute:09:10` with `Mute the microphone until 09:10.` must fail even when corpus WER remains below its aggregate limit. Repeating an identifier where only one occurrence is expected fails by multiset/role comparison. `Нет, отключайте микрофон до 09:10.` fails the negative-command case, while `Микрофон не нужно отключать до 09:10.` passes it. Missing audible transcription is a drop or `UNAVAILABLE`, never a semantic pass.

Owner/seam: benchmark semantic-oracle owner; exact negative controls, paired clause variants, and report-schema/hash tests.

## Latency, segmentation, and flow control

### POL-1 — Independent latency transitions

Given one direction and its current mode:

- three consecutive fresh utterances strictly above the first-audio threshold degrade one step; a fresh non-breach resets the count; stale/duplicate timestamps neither advance nor reset it;
- two consecutive eligible 60-second windows, each with at least 20 samples and p95 strictly above first-audio or queue threshold, degrade one step; an eligible non-breach or undersized window resets breach hysteresis; invalid/duplicate epoch timestamps are ignored;
- queue lag strictly above `500 ms` continuously for `2000 ms` degrades one step; a fresh missing observation or value `<=500 ms` resets the timer; stale/duplicate timestamps are ignored;
- recovery requires five consecutive eligible stable windows and expiry of the 120-second cooldown; a breach or undersized window resets recovery;
- Streaming-first cannot degrade further, Quality-first cannot recover further, and transitions affect only the observed direction.

Rolling windows are closed monotonic intervals evaluated once per second, contain only unique samples whose onset falls in the preceding 60 seconds, and compute p95 by nearest rank at index `ceil(0.95*n)-1` after ascending sort. Recovery from Streaming-first to Balanced or Balanced to Quality-first requires five consecutive eligible windows satisfying the *target higher-quality mode's* first-audio and queue thresholds, not only the current mode's limits, plus the cooldown. Then status reports the transition reason and monotonic timestamp. Before the first eligible window p95 values are unavailable, not synthetic zero; afterward the last eligible p95 remains until another eligible window replaces it, including across fast/manual transitions.

Owner/seam: daemon latency-policy owner; deterministic clock/table tests.

### POL-2 — Typed mode and release policies

Given every mode, daemon degradation, segmenter limits, provider IPC expiry, benchmark inputs, and UI labels consume one canonical mode-policy definition and expose the first table. Independently, benchmark release classification and release UI consume one immutable release-policy definition (`<=1000`, `(1000,1500]`, `>1500`). No consumer may copy or conflate either table.

Owner/seam: `translator-core` mode policy projected through typed IPC/status; cross-language and UI contract test.

### SEG-1 — Mode-bounded utterances

Given confirmed speech in any mode, natural silence commits immediately and continuous speech must commit a stable partial no later than the manifest-bound forced capture cutoff defined above, strictly before that mode's maximum source age. Confirmed frame sequences across a forced boundary are contiguous, unique, and assigned exactly once; delayed confirmation may buffer but not lose confirmed speech. Sub-confirmation bursts submit nothing. Natural silence marks `continuation=false`; forced voice marks `continuation=true`. Invalid or oversized environment overrides fail validation or clamp to the active capture cutoff.

The first confirmed frame creates one monotonic source deadline of that frame's capture timestamp plus `3000/2000/1000 ms`. Work is valid at equality and expired strictly after the deadline. Capture, provider-input, provider-output, and playback each recheck the same deadline; expired work is cancelled and no late audio is played. End-of-utterance never resets the deadline. A continuous-speech test spanning at least four forced boundaries in every mode proves each chunk produces audible audio and a terminal event before its own earliest-frame deadline, with the ACC-2 critical meaning preserved across boundaries; frame-accounting alone cannot pass.

Owner/seam: generic audio segmenter accepts typed limits; daemon maps canonical mode policy to those limits; frame-sequence property tests and daemon adapter test.

### FLOW-1 — Bounded queues and stale-audio exclusion

Given capture, provider-input, provider-output, and playback pressure, each per-direction queue is bounded to `400/800/1200/400 ms`, respectively. At equality the last fitting frame is accepted; a frame that would exceed the bound takes the mode-specific overflow path. Expired/cancelled/superseded audio is never played; overflow emits latency, safe error, and terminal in that order; all workers remain bounded; one overloaded direction does not corrupt or stop the other unless a shared provider fault requires a typed generation restart.

Owner/seam: queue owner at each boundary and daemon generation supervisor; deterministic saturation tests.

## Safety, routing, lifecycle, and controls

### SAFE-1 — Fail-closed acoustic admission

Given no selected device, unavailable device, unknown output, open speaker without matching validated AEC, or AEC validated for another physical pair, admission follows this matrix:

- incoming-only requires only a validated physical sink and opens no microphone;
- outgoing-only with headphones uses the validated physical mic; outgoing-only with open speakers requires matching validated AEC;
- duplex with headphones uses physical mic plus validated sink; duplex with open speakers requires matching validated AEC for that exact pair;
- both directions disabled fails with `no_direction_enabled`.

Every rejection occurs before capture, playback, provider session, network, route, or reported-state mutation.

Owner/seam: daemon acoustic admission policy; launch tests with spies on every forbidden boundary.

### ROUTE-1 — No recursive capture or injection

Given normal, manual, or forged routes, translation never captures `Translator_Virtual_Mic` or the physical default sink monitor; translated output never re-enters `Translator_Remote_In`; translator-owned routes are rejected. Only the exact live daemon-held self-test capability tuple is accepted. Teardown restores the exact original routes and observes zero recursive frames.

Owner/seam: routing policy and ownership journal; fake Pulse graph plus live graph smoke.

### ROUTE-2 — Deterministic incoming-route ownership

Given current Pulse facts, exactly one allowlisted call-like candidate is selected and moved; zero candidates changes nothing; multiple candidates change nothing until an explicit authenticated manual selection names one exact live stream. At most one incoming route is active. If the selected app restarts and exactly one matching replacement identity appears, its recorded app identity may rebind; if it disappears or becomes silent, a different candidate is never auto-selected. A manual override is generation-bound, rejects stale/nonexistent/translator-owned IDs, and loses authority when its stream disappears. Any route conflict closes the affected incoming context, restores the owned prior route, and requires selection before resuming. Duplicate watcher events are idempotent.

Owner/seam: daemon `IncomingRouteCoordinator` owns selection/rebind decisions and consumes fact-only `PulseRoutingWatcher` events plus the route-journal adapter; table-driven fake graph plus live single/multiple/restart/disappearance smoke.

### ROUND-1 — Bounded exact-PCM human round trip

Given headphones, idle real-app route, both healthy provider directions, and one live daemon-held self-test capability, one session may run for at most five minutes. Physical-mic Russian audio traverses the normal outgoing path; VirtualPeer captures the exact translated PCM exposed by the virtual microphone, plays the complete English monitor tap to the validated headphones, and only after that tap completes reinjects the same PCM format/frame count/monotonic sequence/rolling hash into the normal incoming path. Returned Russian then plays to the same headphones. The ordered checkpoints are `waiting_for_speech`, `outgoing_vad`, `outgoing_asr_final`, `outgoing_translation_final`, `english_first_audio`, `virtual_peer_reinjecting`, `incoming_asr_final`, `incoming_translation_final`, `russian_first_audio`, and `completed`; each leg and total physical-onset-to-returned-first-audible latency is reported without text.

Open speakers, a selected real-app route, a second self-test, a conflicting stream, forged/stale metadata, hash/frame mismatch, reinjection before English monitor completion, recursion, or a second pass fails closed. Stop, timeout, SIGTERM, daemon restart, or any failed checkpoint idempotently stops every test worker, restores the exact prior route, removes only session-owned streams, clears memory-only PCM/state, and leaves the normal graph unchanged. No self-test PCM is persisted.

Owner/seam: daemon `RoundTripApplication` owns admission and one `RoundTripSession` aggregate owns execution/terminal cleanup; capability-bound VirtualPeer is an adapter. Deterministic exact-frame/order/fault tests plus headphone live smoke.

### CTL-1 — Atomic session replacement

Given a provider change, either direction's enabled-flag change, either direction's source/target language-pair change, or a target-voice change while running, the affected active session is never mutated only in stored/UI state. A 2xx response means every affected old session acknowledged close, its queued audio was purged, and exactly the required replacement sessions are active with the requested enabled flags, language pairs, provider, and voices; reported/effective state matches those live sessions. Disabling a direction closes it with no replacement, enabling one opens it before commit, and an unaffected peer stays intact unless a shared provider generation must be replaced. Any failure restores the complete previous runtime and state. If restoration fails, status is explicitly non-running/failed. After successful cloud-to-local replacement no cloud frame is emitted.

Owner/seam: one daemon control application service supervising runtime replacement; API tests with transactional controller fakes and runtime generation tests.

### CTL-2 — Boundary-specific live controls

Given a running session, a latency patch returns `202` with typed `desired`, `pending`, and `effective` modes. It does not claim the requested mode as current until the next utterance boundary applies it; status then clears `pending`. A bounded failure clears the request and preserves the prior effective mode.

Debug-text enable keeps the daemon projection gate off until both active provider sessions accept the update; failure rolls back any accepted session before response, and content remains filtered throughout. Disable closes the projection gate first and either updates both sessions or replaces a failed session before reporting success. Audio-mix changes become reported only after physical application succeeds. Every failed apply preserves prior effective/reported values or moves the whole translation runtime to an explicit safe failed state when restoration is impossible.

Owner/seam: control application service delegates typed boundary operations; API and direction-session tests.

### CTL-3 — Concurrent control linearizability

Given racing Start/Stop or two conflicting patches, results have one observable linear order owned by one authoritative runtime state machine: reported state cannot be `stopped` while a generation is active or `running` after its generation is reaped. There is one runtime generation owner, no mixed configuration, no duplicate resource cleanup, and no post-success frames from the superseded cloud generation. At most one blocking lifecycle transition may execute and at most one bounded transition may wait; excess authenticated requests return `409` or `429` before scheduling non-abortable work. A barrier-controlled Stop → Start → delayed Stop completion and a burst of 100 Start requests prove state/runtime agreement, bounded work, and a later clean Stop/Start.

Owner/seam: control application service/supervisor lock; deterministic barriers and concurrency tests.

### CTL-4 — Transactional audio mix

Given a four-target audio-mix patch, the controller first discovers every target and its actual prior volume, applies the complete candidate, and only then commits and publishes runtime state. If applying any target fails, it compensates every already changed target in reverse order; the API returns a safe failure and the store, physical volumes, SSE state, and watchdog remain on the prior committed mix. A fake runner that fails on the second set proves the first target was restored and the rejected values are never retried by the watchdog. If compensation itself fails, the controller suspends the watchdog, stops translation into an explicit safe `audio_mix_state_unknown` failure, and never reports either the rejected or presumed-old mix as effective.

Owner/seam: daemon `AudioMixApplication` owns apply order, compensation decision, commit, and safe-failure transition; `PulseAudioMix` only discovers/applies/restores physical values as commanded. Transactional runner test plus API/store/watchdog integration test.

### LIFE-1 — Start, failure, restart, and stop

Given a pre-ack partial-start failure, every acquired resource is released and state stays stopped. After start:

- `parec` exit restarts only outgoing at most three times with `50/100/200 ms` backoff, then pauses outgoing as failed while incoming continues;
- `pacat` exit restarts only incoming on the same bound, then pauses incoming as failed while outgoing continues;
- app-route loss purges and closes only that stream context and waits for a valid watcher event rather than polling;
- sidecar exit, UDS loss, or shared gRPC corruption performs at most three shared generation restarts with `50/100/200 ms` backoff; the fourth fault is terminal.

Provider probe deadline is `1000 ms`, direction-open deadline `5000 ms`, start acknowledgement deadline `130000 ms`, close acknowledgement `2000 ms`, and Stop `10000 ms`. Exhaustion sets a safe non-running/failed state and proves no child process, lease, route override, or queued frame remains before a later Start succeeds. Concurrent Stop/failure cleans each resource once; repeated Stop returns a bounded successful stopped response.

Owner/seam: translation runtime supervisor; instrumented lifecycle tests with virtual time.

### SYS-1 — User-service crash and restart lifecycle

Given a desktop install/start/restart request, systemd never parses or injects
the service environment. Every initial and automatic daemon start enters the
same isolated launcher wrapper. Before invoking systemd and again at unit
execution, that wrapper opens the current-user configuration root, private
`translator` directory, environment file, and exact daemon executable without
following symlinks; it validates ownership, type, modes, link count, stable
double-read content, a strict key allowlist, and an exact daemon path. Unknown,
duplicate, malformed, non-UTF-8, oversized, shell/loader/startup-control, or
unstable input fails without disclosing values, starting the unit, or executing
the daemon. Accepted values remain literal data and the daemon is executed from
the held descriptor with inherited loader, Python, and shell-startup controls
removed.

Given the staged candidate installed into the logged-in user's PipeWire session, `systemctl --user start translator.service` reaches authenticated candidate-build health and creates exactly one of each owned virtual endpoint within 10 seconds. SIGTERM stops within 10 seconds, closes the provider/sidecar, restores every journaled external route, removes only owned endpoints, leaves no cgroup child, and rotates the control token on the next start. A forced daemon crash kills the old cgroup, reconciles the ownership journal, creates no duplicate endpoint, and returns to the same build identity after the configured restart delay. More than three crashes in 60 seconds is rate-limited into a visible failed unit with no restart storm or orphan; a later explicit reset/start succeeds. UI reconnects to an already-running or restarted daemon without stale effective state.

Before systemd loads the dynamic desktop wrapper, the unit unsets the complete
loader-control set documented by the pinned host glibc, plus
`GLIBC_TUNABLES`, `GCONV_PATH`, `LOCPATH`, and interpreter/shell startup
controls. In particular, `LD_TRACE_LOADED_OBJECTS` cannot replace wrapper
execution with a dependency listing. The wrapper then starts in its isolated
environment and removes every inherited `LD_*`/`PYTHON*` name before exact-FD
daemon execution. An installed static launcher remains the Stage E owner for
future loader controls unknown to the current dynamic runtime.

Owner/seam: desktop launcher owns service-environment admission; systemd owns
activation/restart/cgroup limits; daemon owns shutdown/startup reconciliation.
Isolated launcher/unit tests plus candidate user-unit and live PipeWire smoke.

### DEV-1 — Device and route changes

Given a physical source, physical sink, app route, or AEC pair change, affected-direction work is cancelled and purged before rebinding; the unaffected direction continues unless the AEC pair couples both. Duplicate notifications are idempotent. The original stable device may auto-resume; a different replacement requires explicit selection/validation. No stale output reaches the old target.

Owner/seam: typed watcher event consumed by runtime supervisor; fake watcher/runtime tests.

### CFG-1 — Configuration validation

Given invalid/same language pairs, voice/language mismatch, missing voice preset, unsupported provider/model, both directions disabled, or unknown fields, when configuration is submitted, then it fails without partial mutation. Male/female presets never silently fall back.

Owner/seam: canonical configuration validator before control application; table/API tests.

### CFG-2 — Clean-install defaults

Given fresh isolated XDG config/state/cache/runtime directories and no prior settings, the first authenticated status and native UI show the local provider, Quality-first mode, outgoing RU->EN, incoming EN->RU, both directions enabled, translated-only audio, cloud egress disabled, debug text disabled, and debug capture disabled. No model load, audio device open, route mutation, provider session, or network egress occurs until Start. Restart before any mutation preserves these defaults; the first committed user patch persists exactly the changed settings without changing the untouched defaults. An upgrade may migrate an existing explicit setting but must never replace it with a clean-install default.

Owner/seam: versioned configuration-defaults/migration owner projected through daemon status; isolated-home API/native-UI tests and zero-side-effect spies.

## Privacy, cloud, readiness, and API security

### DBG-T — Memory-only debug text

Given debug text off, content is never projected. Given it on, content exists only in a 200-event/1-MiB in-memory ring and is cleared on translation stop, provider switch, daemon restart, and UI close. It never enters logs, persistence, telemetry, errors, normal UI state, or capture metadata.

Owner/seam: daemon `DebugCoordinator` owns the lifecycle transaction and `DebugTextBuffer` owns bounded content; Tauri renders the already-filtered daemon projection only. Marker tests cover every lifecycle event.

### DBG-A — Independent bounded debug capture

Given explicit enablement, capture creates only beneath the canonical state directory with directory `0700` and exclusive non-symlink files `0600`; it stops at 10 minutes, 500 MiB, before the 5-GiB free-space floor, or when the 2-GiB aggregate capture quota would be crossed. At most eight capture files total may exist, including an active or crash-recovered incomplete file. Enable reserves only the remaining aggregate byte budget and one file slot, and otherwise fails with a safe quota code before creating a file. Seven closed files plus enable plus close yields exactly eight closed files; eight closed files plus enable returns the quota error with no creation, deletion, or byte-count change. Closed artifacts are never evicted silently; an explicit authenticated delete may remove only a closed regular file resolved relative to the already-open private directory and rejects symlinks, traversal, and the active capture. Explicit disable, daemon restart, or a limit failure closes the active capture. Provider replacement and translation stop leave the explicit capture session and its closed artifacts intact; while translation is stopped no PCM is appended. Nothing describes closed artifacts as cleared. Prepopulated quota/count boundaries, crash recovery, and repeated enable/disable cycles prove aggregate usage remains bounded.

Owner/seam: daemon debug-capture store/session; filesystem and lifecycle tests.

### CLOUD-1 — Explicit and revocable egress

Given no cloud opt-in, or a completed cloud-to-local replacement, when audio flows, a network spy observes zero cloud audio. Opt-in is an in-memory daemon decision bound to one runtime generation/provider selection; daemon restart, Stop, or successful switch to local revokes it. Tauri may request/confirm but cannot grant egress directly. The daemon passes a generation-bound capability to the sidecar, whose cloud adapter can reject its absence but cannot mint or persist it. Credentials and payloads never enter UI, logs, status, errors, telemetry, or release evidence. A cloud session start produces only a privacy-safe technical event.

Owner/seam: daemon `CloudEgressAdmission` is the sole authorization owner; sidecar OpenAI adapter is a fail-closed capability enforcement adapter; network-spy integration tests.

### CLOUD-2 — Provider-event attribution after cancellation

The dedicated Realtime Translation protocol has no per-utterance commit, cancel,
response, or item identity. Each admitted local cloud utterance therefore owns one
immutable provider-connection generation. Its input frames and all accepted
provider deltas are bound to that generation before publication. Normal input
completion sends `session.close`, continues receiving the remaining deltas, and
publishes a terminal only after `session.closed` or a bounded failure. Local
cancellation, timeout, or supersession closes and retires the whole generation,
discards every later delta from it, and reaps its receiver before a replacement
generation may accept utterance B. No event received on A's connection can be
relabeled as B.

Terminal processing removes every generation's audio sequence, resampler
remainder, timer, pending buffer, socket, receiver task, and utterance binding.
A fake socket that emits late A audio while B is being prepared proves that the
audio is discarded and only the distinct B connection can publish B audio. One
thousand sequential completed, cancelled, and failed utterances leave all
per-utterance and retired-generation collections empty after each terminal.

Owner/seam: OpenAI per-utterance connection-generation owner; fake-websocket
test for cancel A → retire A → late A audio → open B → B audio.

### CLOUD-3 — Failed cloud open owns and closes its socket

Given a websocket that connects but whose session-update send, response validation, receiver-task creation, or registry insertion fails, `open_session` closes the socket and reaps any task within `2000 ms`, publishes no opened/healthy state, and leaves no registered or unregistered session. Concurrent opens reserve the session ID atomically before the first await: one may proceed and every duplicate receives a bounded conflict without connecting. For each fault point and the duplicate-open race, 100 iterations leave zero net sockets, tasks, file descriptors, registered sessions, or credential-bearing objects after every iteration and at the end; provider shutdown remains idempotent.

Owner/seam: OpenAI session resource owner; fault injection at every post-connect step and leak assertions.

### CLOUD-4 — Version-pinned OpenAI protocol conformance

If OpenAI remains advertised, its capability manifest pins the official
`wss://api.openai.com/v1/realtime/translations` endpoint,
`gpt-realtime-translate` model alias policy, event-schema retrieval date, input
format (`24 kHz` PCM16 mono little-endian), and the following exact wire surface:

- client: `session.update`, `session.input_audio_buffer.append`, `session.close`;
- server: `error`, `session.created`, `session.updated`, `session.closed`,
  `session.input_transcript.delta`, `session.output_transcript.delta`, and
  `session.output_audio.delta`.

The source transcript capability is advertised and accepted only when the
session update explicitly configures `audio.input.transcription`; otherwise an
input-transcript event is a protocol violation rather than invented evidence.
A tracked privacy-safe replay covers creation, update acknowledgement, valid
200-ms and split audio appends, output audio/transcript deltas, optional source
transcript deltas, recoverable error, graceful close with output draining,
unknown additive events, malformed JSON, invalid base64, and incompatible
required-field/type drift. Unknown additive event types are ignored without
mutating an utterance; malformed documented events fail the owning generation
closed. The replay must not use voice-agent events such as `response.create`,
`response.cancel`, input-buffer commit/clear, response/item IDs, or provider
utterance terminal events, because the translation endpoint does not define
them.

An exact-candidate live test after explicit opt-in exercises successful audio in
both language pairs, local cancellation by retiring A's connection, late-A
exclusion, completion on a distinct B connection, and `session.close` draining
through `session.closed`; it stores only safe event-type counters and hashed
event IDs. Missing credentials, schema mismatch, endpoint/model rejection,
bounded generation retirement failure, or absence of the documented translation
contract removes OpenAI from the release capability manifest and UI rather than
accepting a fake-only pass.

Protocol authority, retrieved 2026-09-04: the official
[Realtime Translation guide](https://developers.openai.com/api/docs/guides/realtime-translation),
[client-event reference](https://developers.openai.com/api/reference/resources/realtime/translation-client-events),
and [server-event reference](https://developers.openai.com/api/reference/resources/realtime/translation-server-events).

Owner/seam: versioned OpenAI wire adapter/capability manifest; official-schema replay and bounded live conformance probe.

### READY-1 — Provider-specific readiness

For local-selected/local-ready, local-selected/local-failed, OpenAI-selected/OpenAI-ready-with-opt-in, and OpenAI-selected/OpenAI-unavailable cases, only the selected provider gates Start. An unselected provider failure cannot block it. OpenAI without opt-in performs zero network access. Status identifies selected provider, generation/session, required model states, queues, and safe last error. `degraded` is operational only under the canonical decision above.

Owner/seam: daemon `SelectedProviderAdmission` inside `ControlApplication`, fed by typed provider-owned health; four-case tests with network spy.

### ERR-1 — No-speech and CUDA failure

Given empty ASR, the provider emits ASR latency, `no_speech`, and terminal without MT/TTS and without poisoning model health. Given CUDA OOM, it performs at most one measured smaller-model/CPU fallback and otherwise becomes unavailable; no process or session restart loop occurs.

Owner/seam: local provider failure policy; deterministic model-adapter tests.

### LIB-1 — Native CUDA library admission

Given a daemon-started sidecar, before `exec` the daemon removes
`GLIBC_TUNABLES` and every inherited environment key whose raw bytes start with
`LD_`. It never merges ambient `LD_LIBRARY_PATH`. Each configured CUDA directory
is absolute and existing; an absent portable default is optional, but an unsafe
existing default or any invalid configured entry fails sidecar start closed.
Every lexical component, every intermediate symlink and its resolution chain,
and the final canonical target have root-or-service-UID ownership and
no group/other-write or set-ID mode. A root-owned sticky directory is allowed
only as a traversal ancestor, never as the admitted directory or library.

Before exposing a directory to the loader, the daemon validates a stable,
bounded traversal of the complete reachable tree: entries are only trusted
regular files, directories, or trusted symlinks whose canonical targets satisfy
the same policy. The sidecar independently repeats that admission for supervised
and direct/debug paths. Each known CUDA preload is opened without following a
final symlink, matched to the validated visible inode, loaded through its held
`/proc/self/fd` path, and kept open for the native loader lifetime. Failure
closes pending descriptors and returns only `unsafe_cuda_runtime` or
`cuda_runtime_unavailable`, without a caller-controlled path. Python serializes
the complete process-global admission/open/environment/preload transaction,
rejects a later conflicting SONAME identity, and restores the prior loader
environment when the transaction fails. Direct daemon invocation is not a
secure entrypoint because its own loader has already consumed the caller
environment; the isolated desktop wrapper/service is the supported production
entrypoint.

Owner/seam: daemon sidecar launcher owns pre-exec loader environment and full
directory admission; Python CUDA bootstrap owns independent direct-run
admission and held-FD preload lifetime. Ownership/mode, symlink-target,
replacement, non-UTF-8 `LD_`, descriptor cleanup, and real packaged CUDA tree
tests are mandatory, including the exact tree/symlink bounds and concurrent
first configuration.

### SEC-1 — Local control boundary

Given any control or SSE request, the service binds loopback only, requires the rotating bearer token, caps a control body at exactly `64 KiB`, caps concurrent SSE subscribers at exactly four, and admits at most 64 accepted TCP connections. Equality is admitted and one byte/subscriber/connection over the limit is rejected without spawning request work. An incomplete request header is closed strictly after a five-second monotonic deadline. Bearer authentication runs before body/JSON parsing, SSE admission, `spawn_blocking`, or any control-transition reservation. The service unit sets `LimitNOFILE=1024` and `TasksMax=512` as a second bound. Responses use safe structured problem details, the token rotates on daemon restart, and no speech or credential material enters responses or logs. Wrong/missing auth, oversized bodies, subscriber exhaustion, malformed JSON, unknown fields, and 1000 partial-header connections fail closed without starving one valid authenticated request after capacity is released.

The daemon creates `$XDG_RUNTIME_DIR/translator` without following symlinks as a real current-UID directory mode `0700`, creates a fresh unpredictable `control.token` through exclusive no-follow mode-`0600` creation, and atomically publishes only a current-UID regular file after the loopback listener is ready. Foreign-owned paths, symlinks, hard-link count other than one, permissive modes, non-regular files, or failed atomic replacement fail closed. The native Tauri backend derives and reads the path itself; frontend JavaScript receives neither token nor path. Token/path never enter argv, UI state, errors, logs, telemetry, or artifacts. On reconnect or one `401`, Tauri rereads the protected file once and retries once; it never persists a token, loops on `401`, or reuses an old generation after daemon restart. Mode/owner/link/rotation/reconnect cases are mandatory.

Owner/seam: daemon HTTP boundary and Tauri Rust proxy; router/security tests plus socket-bind smoke.

### MODEL-1 — Verified model bytes remain the loaded bytes

Given a manifest-pinned model, verification yields a `VerifiedModelLease` that owns stable directory/file descriptors and the verified digest for the full native-loader lifetime; production loaders do not accept a raw mutable cache path. If an updater atomically replaces the visible path after verification, the native ASR/MT/TTS loader still reads the verified inode/generation or fails closed before load. It never loads replacement bytes under the old health/digest identity. One replacement race at every verify/open boundary and 100 repeated update/load cycles prove loaded hashes equal verified hashes, leases close once, and no descriptors or model generations leak.

Owner/seam: model store/lease boundary consumed by local runtime loaders; atomic-rename fault injection and native-loader adapter tests.

### MODEL-2 — Model license, disk, and accelerator policy

Every model manifest entry has an HTTPS source, exact immutable revision, per-file SHA-256 and byte size, supported languages/capability, SPDX license or an explicit private-use waiver, and `redistribution` policy. Public artifacts contain no model with `redistribution=false`, no model cache, and no absolute local path. A fresh acquisition plan downloads at most 2 GiB of newly approved model bytes, starts only if at least 20 GiB will remain after all temporary and final files, verifies before atomic activation, and removes partial files on failure. Normal simultaneous duplex and every operational fallback peak at no more than 10 GiB used VRAM on the 12-GiB target, preserving at least 2 GiB headroom; failure reports unavailable rather than overcommitting. Both male and female Russian and English target voice profiles pass audible gates without fallback.

Owner/seam: model manifest/store policy plus release artifact filter; schema/license/content/disk fault tests and measured VRAM/voice reports.

### SCA-1 — Dependency and build-input integrity

Given a clean lockfile build, Rust, Python, and UI advisory scans terminate within 120 seconds and report zero unreviewed known vulnerabilities across runtime, build, test, and developer-tool dependencies. Any unsound/unmaintained exception names the exact advisory/package/version and dependency path, demonstrates affected-function non-reachability or a patched source, records an owner and expiry, and blocks release after expiry. UI dependency audit has a bounded primary scanner plus an independent fallback and always reaches a terminal result. CI uses full-commit action pins, pinned tool versions with integrity verification, `cargo --locked`, and the committed Python/UI lockfiles; changing a lockfile or build input changes provenance and reruns every scan.

Owner/seam: repository supply-chain policy and release workflow; clean hosted SCA jobs, timeout/fallback tests, and provenance assertions.

### DOC-1 — Executable public contract and publication hygiene

The versioned OpenAPI 3.1 document and generated/typed client fixture enumerate every HTTP method/path, bearer requirement, request/response schema, status, RFC 9457 safe problem code, SSE event/resync shape, body/subscriber limit, and build-identity field; router-contract tests fail bidirectional drift. README defaults, supported/disabled capabilities, install/rollback commands, modes, model/license policy, security/privacy behavior, unit name, and validation commands match the exact candidate. Cargo/Python/Tauri/package/release versions share one release version, and release notes name only gates actually proven by the frozen evidence manifest.

A clean full-history scan rejects scanner-detectable credentials and enforces
text/binary/archive history policy. A separate immutable candidate-tree scan
rejects scanner-detectable credentials, debug captures, raw reports, model
files, private home/source-checkout paths matching the controlled Linux, macOS,
Windows, and systemd patterns, caches, sockets, logs, and
unrelated-repository names. These automated checks are not represented
as semantic DLP: before publication, an independent source-content and provenance
review must certify that arbitrary user/call prose was never admitted to the
candidate or reachable history.

The Gitleaks control pins and verifies both the downloaded archive digest and
the extracted executable digest. The executable is opened once and invoked
through that file descriptor; device, inode, mode, link count, owner, size,
mtime, ctime, and content digest must remain stable across version proof and
all scans. Path replacement, byte-identical atomic replacement, and in-place
mutation fail closed.

The authoritative precommit candidate is exactly one staged Git index tree:
tracked worktree bytes are independently hashed as Git blobs and match it,
nonignored untracked files are absent, and its sorted exact-path manifest and
regular-file mode policy match. Candidate mode emits a receipt explicitly
marked `release=false`; it can bind an architecture review to a tree but is
never publication evidence.

After every review approves that tree, the exact tree is committed and the
intended annotated release tag is created locally without publication. Release
mode accepts the tag name and reviewed tree ID, requires a clean index/worktree,
requires `HEAD^{tree}` to equal that ID, and parses the named ref's direct outer
tag object. Its canonical header must name `HEAD` directly as `object`, declare
`type commit`, and repeat the requested release tag as its internal `tag` name;
a tag-of-tag or mismatched internal name is rejected even if peeling reaches
`HEAD`. Release mode then rescans the now-existing commit and tag metadata plus
every public ref. Its
machine-readable receipt binds the exact HEAD, tree, annotated-tag object, and
SHA-256 of the complete ref state. Merge, tag push, artifact publication, and
activation require this post-commit/post-tag receipt; a precommit receipt or a
receipt followed by ref, commit, tag, index, or worktree mutation is unusable.
Both candidate and every reachable historical blob are strict UTF-8 text without
control bytes except exact hash-pinned public image assets; archives, Git LFS
pointers, local transform/scanner controls, incomplete history, replace/graft
views, detached HEAD, and ref drift fail closed. Every external content scanner
first proves the same git, directory, binary-stdin, and fast-export pipeline
modes used by the real checks with synthetic findings that cannot be suppressed
by repository policy. A clean result is trusted only after those controls
succeed; every unexpected status is an operational failure. Historical public
design evidence may contain non-secret paths or repository names without
requiring history rewriting, but none may enter the candidate runtime bundle.
Documentation, capability manifest, unit, OpenAPI, release notes, and artifacts
are hash-bound to the released SHA.

Owner/seam: typed API/schema generator plus publication validation entrypoint; router/schema/doc/version and secret/private-artifact tests.

## Installation, evaluation, applications, and release

### INST-1 — Staged portable installation

Given any required build/copy failure, the prior installation remains untouched. A staged candidate must run without the repository or build-time `CARGO_MANIFEST_DIR`; units/config, model manifests, binaries, and dependency metadata contain no source-tree, home-directory, or unrelated-repository path. Authenticated health reads the protected token, requires HTTP 2xx plus expected schema and build identity, and rejects 401/5xx. Fresh isolated XDG directories acquire only manifest-pinned model revisions, sizes, and hashes and load them through MODEL-1; partial/corrupt downloads fail closed. Activation failure restores and proves the previous version active.

Owner/seam: installer/release workflow; isolated-home shell integration tests and rollback fault injection.

### EVAL-0 — Frozen diverse corpus and acoustic fixtures

Before behavior/model tuning begins, freeze a development corpus and a disjoint release holdout with independent SHA-256 identities. The release holdout contains at least 120 unique source/reference pairs for each of RU→EN and EN→RU, with no duplicate normalized source and no slot-expanded template family contributing more than 10% of a language pair. Six disjoint primary buckets contribute at least 20 cases each: short turns; long turns; negation/scope contrast; multiple/repeated numbers, times, and alphanumeric identifiers with roles; Latin/Cyrillic/inflected names and role swaps; and discourse/chunk-boundary/coreference/interruption cases. Critical cases include both positive and negative controls.

Source audio covers at least four speakers per source language (at least two voice genders and one non-default accent), with every speaker contributing to every primary bucket. Each language pair includes at least 20 cases in each independently labelled condition: clean, 10-dB SNR speech-shaped noise, room impulse response, ±6-dB gain, 8-kHz telephony resampling, leading/trailing silence, and all 20-ms packet-boundary offsets. Deterministic transforms retain an origin hash and cannot count as a new semantic utterance. Licenses permit the chosen storage/distribution; private fixtures remain in a hash-only external manifest and missing access is `UNAVAILABLE`. Runtime/postprocessor code cannot inspect case IDs, split membership, or reference text. The holdout is not used to choose rules, prompts, or model parameters; any post-freeze change increments the corpus version and invalidates earlier comparisons.

Owner/seam: versioned eval corpus/fixture manifest; uniqueness/template-family/license/coverage validators and mutation negative controls.

### EVAL-1 — Absolute quality and latency proof

Given EVAL-0 and the exact candidate build, evaluate the complete mandatory local-provider matrix from the canonical decision and every advertised optional provider x mode x audio direction x language pair x operational-fallback x target-voice cell independently. Each full cell runs at least 10 excluded warmups then all 120 release-holdout utterances; at least 30 measured attempts in every full provider/fallback cell include at least `500 ms` of simultaneous VAD-confirmed peer speech. Local target-language male/female voice profiles are balanced 60/60 across their paired voice cells and reported separately; every voice cell must pass the same audible semantic/drop/latency floor. Every attempt has a `10000 ms` terminal/audible timeout. A drop is timeout, cancellation, missing terminal, provider error, or missing audible output and is counted over all attempts in that cell. Percentiles use successes only; each cell's drops must be `<1%`. Every cell must satisfy chrF2 `>=45`, TTS proxy WER `<=15%`, zero critical negation/number/name/role corruption, its manifest-bound provider-first-frame p99, and its mode's first-audio/source-age/queue limits. The local provider must also classify `meets_target` or `usable_degraded` for both audio directions and both language pairs under the separate release policy.

Owner/seam: versioned eval harness and external graph-boundary recorder; machine-readable report with raw attempt classifications but no content secrets.

### EVAL-2 — Product comparison and controlled ablation

Given untouched baseline and candidate, the primary product comparison runs each build with its own frozen release manifest and effective configuration on the same machine, corpus, fixtures, audio graph, supported matrix cells, and voice assignments. Baseline and candidate attempts are paired by corpus/fixture/cell identity and executed in deterministic counterbalanced AB/BA blocks, interleaved within each cell after separate warmups, so cache, run-order, battery, and thermal drift cannot systematically favor one build. Record source, binary, model-manifest, model-file, corpus, fixture, unit, config, environment, driver, and harness hashes plus temperature/clock/power samples. All versioned baseline defect reproducers must change from FAIL to PASS.

For every shared full provider x mode x direction x language-pair x operational-fallback x target-voice product cell, compute 10,000 paired nonparametric bootstrap resamples by drawing attempt-pair indices with replacement using NumPy `PCG64` and seed `20260904`; report the nearest-rank two-sided 95% interval and use its upper endpoint for regressions. Independently in every such cell, the upper bound for candidate-minus-baseline first-audible p95 is `<=max(50 ms, 5% of baseline)`, candidate chrF2 may fall by at most `0.5` absolute, proxy WER may rise by at most `1.0` percentage point, critical errors remain zero, drops remain `<1%`, unplanned restarts remain zero, and steady-state RAM/VRAM medians may rise by at most `max(128 MiB, 10%)`. Absolute gates still apply, so a tolerated delta cannot excuse a failing result. A cell newly supported only by the candidate is `NOT_SUPPORTED` for comparative deltas and must pass the absolute gates; any baseline-supported cell missing from the candidate is `REMOVED_REGRESSION` and blocks release.

When both builds support the candidate's model files and configuration, run a separately labelled same-model/same-config ablation under the identical pairing, counterbalancing, bootstrap, and thresholds to isolate code effects. Failure or impossibility of that ablation cannot replace or invalidate the primary release-manifest product comparison; its reason is explicit rather than silently changing either build's product configuration.

Owner/seam: eval orchestrator; immutable paired product report plus optional controlled-ablation report.

### EVAL-3 — Cold, warm, and soak stability

Given cold, warm, and 30-minute duplex soak runs, report external/internal latency, queue depth, CPU/RAM/GPU/VRAM, drops, cancellations, OOM, restarts, child processes, and file descriptors. The soak uses monotonic one-second buckets for 1800 seconds, requires at least one sample in every bucket, permits no sample gap over two seconds, and excludes no bucket or outlier. Resource slopes use ordinary least squares over all 1500 bucket values from buckets 300 through 1799. The warm reference is the median RSS/VRAM and maximum FD/child count across buckets 240 through 299. Pass requires zero OOM/crash/unplanned restart, drops `<1%`, continuous queue-bound compliance, child count equal to the warm reference in every later bucket, final FD count no more than two above the warm reference, and post-warm RSS/VRAM slopes each `<=1 MiB/min` with final values no more than `128 MiB` above their warm medians. A missing bucket, timeout, or excluded outlier fails the affected assertion.

Owner/seam: eval/telemetry harness; sampled report and process/resource assertions.

### APP-1 — Real application duplex

Given the packaged current build with the local provider, run Meet, Telegram, and Zoom two-endpoint calls in all three mandatory modes and both duplex language assignments (microphone RU→EN + speaker EN→RU, then microphone EN→RU + speaker RU→EN) for at least 60 seconds per app/mode/assignment cell. Alternate both target-language voice genders within every local cell and report each separately. Inject distinguishable source and translated canaries with at least 20 seconds of overlapping speech. Verify both remote/local translated receipts and exact route restoration separately. At both receiving endpoints of every cell, independently transcribe audible output and apply ACC-2; missing transcription or any negation/number/name/role corruption fails. Original-language leak fails if source-canary correlation at either translated endpoint is above `-40 dBFS` or less than `30 dB` below translated-canary energy, or if the source critical-token oracle detects its negation/number/name markers.

Owner/seam: external E2E harness and route observer; per-app evidence.

### APP-2 — Cloud and acoustic capability claims

If OpenAI remains advertised/enabled, run it after explicit opt-in on the same app/mode/language-assignment cells and input as APP-1 and require the same latency, drop, semantic, leak, cancellation, and stability floors as EVAL-1/APP-1; missing credentials or a failed floor blocks that advertised capability. AEC is validated only by a 30-second far-end fixture at `-20 dBFS`, median ERLE `>=15 dB`, and zero outgoing VAD/translation triggers during a separate 60-second far-end-only run. Failed AEC disables and documents open-speaker mode but does not block a headphones-only release.

Owner/seam: release capability manifest and real-app harness.

### APP-3 — Native desktop UI

Given the packaged Tauri app at its default `980x680`, minimum `720x520`, and a `1280x800` desktop viewport, test Tab/Shift-Tab traversal, Space/Enter activation, Escape dismissal, start/stop, both direction toggles, provider/mode/target-voice controls, language-pair switching on each direction, cloud-egress confirmation before cloud Start, manual selection from zero/one/multiple live route candidates, and round-trip admission/start/ordered checkpoints/stop. Assert physical and virtual device state, selected routed-stream state, per-direction first-audio and queue p50/p95/unavailable values, separate debug-text and debug-capture state, capture path metadata only while authorized, safe errors, pending versus effective controls, and disabled reasons.

The same native suite covers tray show/hide/toggles, daemon loss/reconnect, stale SSE resync, provider failure, hotplug/route disappearance, focus visibility, clipping/overflow, and disabled controls. It exercises CFG-2 before mutation and both duplex language assignments in all three mandatory local modes; optional advertised provider capabilities are exercised separately under APP-2. The documented dense layout has no nested cards and remains operable at minimum size. Browser-only simulation cannot satisfy this gate; every mandatory local mode is PASS and `UNAVAILABLE` is not PASS.

Owner/seam: Tauri backend/frontend integration and native driver; screenshots plus interaction assertions.

### REL-1 — Evidence-bound release

Given all deterministic, security, SCA, paired eval, soak, native UI, and claimed capability gates pass for one frozen SHA, when release runs, then two clean hosted builds of that SHA with pinned toolchain/container inputs produce byte-identical artifact hashes. The workflow emits checksums, an SBOM covering Rust/Python/UI runtime dependencies, signed provenance, versioned artifacts/release notes, and rollback artifacts; it deploys through authenticated health and tags only that SHA. Merge/deploy cannot precede the evidence packet or independent architecture/security review.

Owner/seam: release workflow; dry-run workflow tests followed by exact-SHA hosted run.

## Scenario-to-test families

| Scenario IDs | Narrow deterministic family | Broader/live family |
| --- | --- | --- |
| TST-1..4, PROTO-1..2, ACC-1..2 | full repository gate; gRPC/UDS/registry, MT repair, and semantic-oracle contract tests | clean clone and hosted CI |
| POL-1..2, SEG-1, FLOW-1 | Rust policy, property, queue, cross-language contract tests | duplex latency/eval harness |
| SAFE-1, ROUTE-1..2, ROUND-1, CTL-1..4, LIFE-1, SYS-1, DEV-1, CFG-1..2 | daemon application/supervisor/router/defaults tests with spies/barriers | isolated-home user service, Pulse graph, round-trip, and fault-injection smoke |
| DBG-T, DBG-A, CLOUD-1..4, READY-1, ERR-1, LIB-1, SEC-1, MODEL-1..2, SCA-1, DOC-1 | privacy, filesystem, provider, model-lease, native-library admission, router, network-spy, schema/doc, and SCA tests | daemon restart/cloud conformance/connection-pressure smoke |
| INST-1, EVAL-0..3, APP-1..3, REL-1 | installer fault injection, corpus/report-schema oracles | isolated install, paired models, soak, apps, native Tauri, hosted release |
