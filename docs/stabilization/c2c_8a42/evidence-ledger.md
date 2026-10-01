# c2c_8a42 validation ledger

This is a fork-only development candidate. Until its worktree is frozen and clean,
none of these results is exact-commit release evidence. A PASS below applies only
to the named gate; it does not upgrade a blocked product gate.

| Gate | Input fingerprint | Environment | Result | Evidence / invalidation |
| --- | --- | --- | --- | --- |
| Python model/default/freeze focused tests | Dirty candidate after `test_product_freeze.py` import fix | Sidecar venv, no model load | PASS | `python -m pytest -q` on the four named sidecar test files; subsequent Rust-only edits do not affect this gate. |
| Python Ruff on changed model/freeze files | Same Python sources | Sidecar venv | PASS | `python -m ruff check` on the five changed Python sources/tests. |
| Rust audio/daemon aggregate | Dirty fork candidate before the final private-Pulse test correction | `CARGO_BUILD_JOBS=1`, no physical audio | PASS on that earlier source, INVALIDATED for current tree | Full workspace tests and doc-tests passed in the deterministic attempt; two private-Pulse test files changed afterward. |
| UI tests/build/native Rust tests | Current UI sources; no subsequent UI edit | Local Bun/Rust toolchain | PASS for development | Bun 18 tests and build; Tauri Rust 13 tests. Exact frozen-tree gate still pending. |
| Private virtual-Pulse lifecycle | Current control/mix source; no physical cards | Disposable explicit Unix socket with four null sinks | PASS for development | Four formerly ignored tests executed: first-frame mute, original-volume readback/restore, respawn exact volume, and owned loopback load/discovery/cleanup. An initial loopback-test setup failed because it tried to create a speaker module after Running; the setup was corrected to Stopped-before-Running and passed. An initial respawn fixture-prefix refusal was corrected without allowing a production socket. Post-test graph contained only fixture modules, no streams; server and temporary socket were removed. This is not physical AEC or product E2E. |
| Coverage classifier and portable loader seam | New metadata-only coverage module and relocated-bundle test | No models, physical audio, install, or systemd mutation | PASS for focused development | 27 classifier tests and 3 relocated-loader tests passed independently. Coverage result is never a product-quality PASS; loader test uses a synthetic bundle and does not prove a relocatable real venv or atomic install/rollback. |
| Deterministic manifest and complete gate | Iteration-0 staged tree `0bdce6ecf45443229a288e3e8926f50e4bfeea6b` | Pinned local Gitleaks, cargo-audit, Bun, OSV | PASS for iteration 0; invalidated by iteration-1 source | C2C execution output 1 reports exact-tree deterministic PASS and `release=false`. The measurement-seam aggregate must be recorded in a new execution receipt after staging; this row is not product acceptance. |
| Nonphysical synchronized measurement seam | Iteration-1 development sources before staged aggregate | Offline synthetic 48-kHz paired PCM; no hardware | Focused PASS, aggregate receipt pending | Audio test 6/6; external-crate public-boundary unittest 1/1 (`E0451` private payloads, `E0308` raw `publish`); strict Clippy and exact-tree aggregate are separate gates. Finite source requires 45 pairs plus terminal end, cleanup, and provenance retention. Evidence remains `NonAdmissible`; caller labels do not attest a physical common clock. |
| Effective-model receipt | No clean committed candidate or independent observation | Installed pinned model cache | NOT_RUN | Receipt tool deliberately rejects a dirty tree and cannot independently observe provider-selected IDs. |
| Open-speaker AEC and AEC round trip | `main.rs` still fixes AEC to Unavailable | No physical audio used | BLOCKED in source, NOT_RUN physically | Production calibration engine, exact binding, positive control, runtime attachment, and proof renewal are absent. Current capture is independent fixed-16-kHz `parec` pipes with no common sample-clock provenance; it cannot truthfully supply aligned 48-kHz raw/clean one-second windows. A separate BDD critic returned CHANGES_REQUESTED for enabling speaker self-test before this prerequisite. Preserve fail-closed admission. |
| Independent RU/EN quality holdout and physical E2E | Existing Common Voice Spontaneous Speech test manifest has 48 RU and 48 EN ASR samples, not 120 source/reference pairs per direction | Required product environment | NOT_RUN | The local corpus is ASR-only and its critical labels use written transcripts; no admitted full-chain release holdout, paired baseline/candidate run, English audible adjudication, or isolated hardware run. The new coverage classifier checks declared metadata only; it does not create eligible data. |
| Merge, package, deploy, release | No frozen passing product candidate | Production excluded | BLOCKED | Requires all product gates and terminal review. |

## Iteration 2: isolated native AEC stop receipt

This iteration implements a separate PipeWire 1.0.5 / SPA WebRTC helper, a strict
`HELLO -> LINKS -> ARMED -> FRAME` decoder, an owned process/scope lifecycle, and a
private device-free test runner. The frozen candidate is **not a product pass**.
The new code remains disconnected from the production composition root:
`AecCapability::Unavailable` and `aec_calibration: None` are unchanged.

| Gate | Result | Evidence and limit |
| --- | --- | --- |
| Native fixture provenance and output witness | Focused PASS, aggregate NOT_DONE | On the private 48-kHz graph, 10 authored frames after a forced 500-ms producer delay followed 50 quarantined empty callbacks; a separate trial quarantined exactly one prevalid sentinel. The same clean bytes reached the independent witness. Meta-gap failed before AEC with FATAL 4; default header failed with FATAL 3. These are virtual-graph facts, not hardware-clock or calibration proof. |
| Native startup and 100-cycle lifecycle | NOT_DONE | The latest full candidate stopped at normal cycle 6/30: FATAL 4, input buffer reasons 5/5/0 before first accepted frame, while the fixture reported activated and valid. Every observed scope and FD/thread baseline was restored. One diagnostic-only 20/20 run did not reproduce the condition; this does not erase the failure. Mixed 100 cycles are NOT_RUN on the final source. Earlier 100/100 seq-skew targeted and 30/30 normal targeted results belong to prior startup candidates. |
| Acoustic correctness | NOT_DONE | Pinned FLEURS-based speech far-end-only median ERLE 58.49 dB with the baseline plugin configuration and 40.50 dB with NS off; wrong-reference medians 0.25/0.23 dB. Near-end-only NS-off had 25 active windows, correlation 0.292-0.844 (median 0.585) against a fixed calibrated lag, below the frozen 0.90 waveform threshold; level stayed within +/-3 dB. No threshold was relaxed. |
| Live fault matrix | PARTIAL, NOT_DONE | Real private graph rejected raw unlink after 10 witnessed frames (FATAL 10), meta-gap, default header, and targeted seq skew on a prior startup candidate. Other link, clock/reset, crash, queue, IPC, and close-hang cases are NOT_RUN on the final source. Synthetic protocol tests do not upgrade this gate. |
| Production proof, physical audio, paired RU/EN product E2E, merge/release | NOT_RUN / BLOCKED | No physical device, production socket, proof publication, model substitution, deployment, or release was used. Deterministic source checks are independent of these product gates. |

The sanitized immutable receipts `intermediate-failures.json`,
`armed-quarantine-stop.json`, and `diagnostic-nonreproduction.json` retain failed
attempts, exact argv/exit, pinned hashes, and cleanup observations. The
initial deterministic invocation was rejected before tests because new source
files were untracked. Two later full-tree invocations reached, respectively,
three Python import-lint defects and one Python formatting defect; those were
mechanically corrected. The next full invocation reached seven repository
publication-gate test failures; its exit remains recorded. A narrow repro
identified the missing PATH to the already installed, version-verified
Gitleaks 8.30.0. With `TRANSLATOR_GITLEAKS_BIN` pointing to that pinned binary,
all 59 publication tests passed; no scanner rule was weakened. An independent
source audit found a cleanup custody
P1: a failed scope-admission check passed `None` to cleanup and could falsely
claim cleanup success for a spawned scope. A RED regression reproduced it;
cleanup now retains the exact prechecked scope path, and the focused hardening
suite passes 5/5. This does not resolve native startup or acoustic blockers.
Do not interpret a later focused success as removing the recorded failure.

Resource rule: run one heavy test lane at a time (`CARGO_BUILD_JOBS=1`,
`cargo -j 1`); do not touch production Pulse/PipeWire or run audible physical
calibration under this plan. Retain failed attempts instead of retrying them
out of the record.
## 2026-09-29 native follow-up (same iteration; not stage acceptance)

The fork remains disconnected from production AEC. This follow-up found two
independent issues in the pinned 48-kHz private graph:

- With the installed WebRTC plugin's default high-pass filter, the near-speech
  control reached only 0.292 minimum one-second waveform correlation. A
  diagnostic search over +/-30 ms reached only 0.536 on the worst second, so
  fixed-lag scoring alone did not explain the loss. With
  `webrtc.high_pass_filter=false` and noise suppression still off, the same
  25 active seconds reached 0.993 minimum correlation and -0.45 dB median
  level change. The far-speech median ERLE was 41.30 dB (floor 15 dB);
  wrong-reference median was 0.62 dB (required below 15 dB). The frozen
  thresholds were not changed. These three clean runs used one binary/plugin
  identity before a later diagnostic-only binary rebuild; they are
  nonphysical development evidence, not an exact final-artifact release gate.
- A previously lost first-failure site now stays immutable. The private runner
  retains at most 64 KiB of anonymous per-child diagnostics and releases
  numeric-only first-buffer/gap metadata; oversized or incomplete logs are
  `UNOBSERVED`. The `ARMED` receipt now follows the first processed frame,
  rather than filter activation. The fixture and helper derive input header
  sequences from the same graph position and session, and reject later
  discontinuities. Two full 100-cycle sets passed on the intermediate binary
  (each 20 cancel, 20 seq-skew, 60 normal), but a later exact rebuild stopped
  at cancel cycle 10 with FATAL 4 before any frame. Earlier FATAL 4 snapshots
  include a missing input and a malformed first chunk. The intermittent
  startup defect remains open.
- Another 45-second speech trial stopped at frame 686 on a graph-gap failure.
  Position and clock matched; the xrun value was not yet captured. The
  follow-up now records expected/observed xrun as numeric diagnostics but
  has not proved long-run reliability. The machine was under substantial
  unrelated load; that is context, not a waiver for a real gap.
- A proposed pre-START producer-emission query was tested and removed:
  PipeWire suspended the linked producer while its consumer was inactive,
  so the query timed out and masked the meta-gap negative control. No
  malformed-buffer quarantine or Start deadline relaxation was retained.

The current source gate first ran through Rust, UI, and Python lint, then
stopped at Python formatting of the new diagnostic tests. Formatting is
mechanical; a successful aggregate must be recorded against the refreshed
staged tree. Native stage acceptance remains `NOT_DONE` until startup and
long-run provenance pass without erasing these failures. Physical audio,
proof publication, merge, and release remain `NOT_AUTHORIZED`.

## 2026-09-29 exact-binary continuation (same iteration; still NOT_DONE)

The publication allowlist was restored from the exact staged index after a
previous edit had omitted 46 already tracked paths. The scoped publication
candidate gate passed with `release=false`. The mandatory
`./scripts/translator-validate deterministic` then passed UI, Rust,
pytest (1133 passed / 5 external skips) and repository unittest (276 passed /
22 external skips), but stopped at SCA because pinned `cargo-audit 0.22.2`
is absent from the current environment. One focused regression test was added
after that aggregate, so the aggregate is **not** a PASS on the final tree.
The generated test manifest was refreshed and verified after this addition.

A mixed lifecycle probe first returned a false `NOT_DONE`: the helper
correctly rejected injected sequence skew with structured
`fatal_reason=3, gap_kind=2, fatal_processed=0` and confirmed cleanup,
but the harness expected an obsolete free-text FATAL phrase. A RED unit
reproduced the bad verdict. The harness now checks the exact structured
first-failure tuple and verified session cleanup; its regression and the
real one-cycle negative control passed. A subsequent mixed 20-cycle probe
had no failures. The mandatory 100-cycle probe then returned
`NONPHYSICAL_LIFECYCLE_PASS`: 20 cancel, 20 sequence-skew and 60 normal
cycles, each with an absent owned scope and zero FD/thread delta. This
corrects the **harness verdict**, not the previously observed native
startup FATAL.

On the same native executable SHA-256
`2ef9a7a959d134771903a8e9cee95764e613203215de89f664041c631e3664c5`,
fixture
`e416b2199feb2e6c93c2bb7dd078bdcf364f43b7bf83d421faa2c73a58a6ed1f`,
witness
`1292be3a6150ca4a99f8046f0d22be0cd3fddd30c0d3ee69e476fe2580792237`,
and WebRTC SPA plugin
`775b1fa05840ed6c26d61c4c31069533fb5b0966d19d1f3a7a1142c35024e214`:

| Isolated control | Result | Limit |
| --- | --- | --- |
| Synthetic far-only, 4500 frames | PASS, median ERLE 17.36 dB, exact output witness, cleanup 12 ms | One continuous run does not erase a prior frame-686 gap. |
| Frozen FLEURS speech-far, 4500 frames | PASS, median ERLE 41.30 dB, cleanup 5 ms | Virtual graph only. |
| Frozen FLEURS speech-near, 4500 frames | PASS, 25 active seconds, minimum waveform correlation 0.9932, median level change -0.45 dB, cleanup 4 ms | Virtual graph only; plugin internal errors remain unobserved. |
| Frozen FLEURS wrong-reference, 4500 frames | PASS as negative control, median ERLE 0.62 dB, cleanup 8 ms | Does not replace explicit passthrough/zero-clean scorer mutants. |

All four returned `aec_proof=false`, mapped the pinned plugin and matched the
actual clean bytes at the independent virtual witness. No microphone,
speaker, production socket, or production audio configuration was used.

The bare required `./scripts/translator-aec-backend-check --isolated`
currently runs only 100 far-only frames with no acoustic metric and exits
4/`NOT_DONE`. Existing live fault modes can return the same generic failure
code for a fault injected after ten valid frames **or** an unrelated startup
failure, so they cannot yet be aggregated as a typed PASS. Required
clock/reset, generation reuse, AEC error, close-hang, and explicit
passthrough/zero-clean mutant evidence remain incomplete. A previous
first-buffer `FATAL 4` and a 45-second graph gap are retained, not
overwritten by the later passing runs. The native stage remains
`NOT_DONE`; production AEC remains unavailable, proof publication and
physical audio remain `NOT_RUN`, and merge/release remain unauthorized.

## 2026-09-29 isolated stage gate after independent audit

Independent architecture and code review identified two possible false-green
classifiers and incomplete dynamic-module identity. The stage classifier now
requires each case's exact fixture mode, frame count, identity, acoustic metric
and threshold. The 100-cycle receipt must contain all indexed cases (20 cancel,
20 sequence-skew, 60 normal), with expected exit and zero resource deltas.
The meta-gap control is explicitly a **first-frame** FATAL 4 control
(`fatal_processed=0`), not a sustained-flow result. Owner test receipts bind
the exact test names and counts. The artifact set now includes configured
PipeWire/SPA dynamic modules, speech sources, ffmpeg and their linked libraries;
it is rechecked before and after every case. Negative classifier regressions
cover swapped modes, absent metrics, bad thresholds, empty lifecycle, and
incomplete first-frame diagnostics.

The device-free stage run on the frozen artifact set
`89e8e1569d54280b9c6c6697cba88804dfc06a9dbfef3ad2d470af7beafa9150`
returned `NONPHYSICAL_STAGE_PASS`: 25/25 sequential cases, including exact
owner contract/native tests, 100 lifecycle cycles, both startup delays, six
45-second acoustic controls, both full-length scorer mutants and ten
post-witness injected fault modes. Every row retained the same artifact-set
identity and reported `frozen_unchanged=true`; all receipts keep
`aec_proof=false`. Raw output is the local stage receipt for C2C publication,
not a product/release receipt.

Earlier intermittent first-buffer FATAL 4 and frame-686 graph gap remain
historical reliability failures. A later 25/25 run does not prove their root
cause was eliminated. The production composition root still sets AEC
Unavailable, and neither real synchronized hardware acquisition nor
30s/60s open-speaker admission was exercised. Independent RU/EN holdout,
paired quality/latency, native-app soak, merge and release remain NOT_DONE.

### Final oracle tightening on the same iteration

The first 25/25 receipt above is retained as historical evidence, not the final
oracle identity. Independent review found two further possible false-green
interpretations: median near-end level alone did not cover individual 1-second
level extremes, and an arbitrary first-frame FATAL 4 could be mistaken for the
intentional meta-header gap. The final classifier now requires finite near
minimum/median/maximum level changes all within +/-3 dB and requires native
reason 7 on both inputs, expected chunk/header diagnostics, and first callback
for the meta-gap control. Negative regressions reject both cases.

The repeated full isolated matrix returned **25/25 PASS** with artifact set
`e8878879d9927ecf6a16866c6ec6a4766cf217f1b44702993f342cc2e9d65a17`,
all rows `frozen_unchanged=true` and `aec_proof=false`. The exact stage
receipt is retained for C2C execution output. This replaces the first
artifact-set identity for the nonphysical stage only.

One cleanup-liveness risk remains visible to independent code review:
`wait_for_cleanup` can reap the leader with `try_wait` before all descendants
leave the process group, making repeated group signalling unavailable if
cleanup stays pending. This does not create a false Completed result or
production AEC admission, but it is not proven closed by 100 lifecycle cycles.
Retain it as a blocker for future production attachment, alongside the
historical first-buffer FATAL 4 and frame-686 gap. Physical AEC/product gates,
paired RU/EN quality and latency, merge and release remain NOT_DONE.

## 2026-09-30 iteration 3 custody correction — WIP, not stage acceptance

The iteration-3 C2C review requested one nonphysical correction of delayed scope
creation, guardian custody under Tokio failure, and the eight-second shutdown
entry deadline. The fork-only working tree is still dirty relative to the
staged candidate. No production AEC wiring, physical audio, merge, or release
was performed.

| Boundary | Current local observation | Remaining limit |
| --- | --- | --- |
| Late scope after a negative cleanup observation | A separate held creator process is released only after the first bounded `CleanupPending` response; the Rust guardian binds its late fake scope, writes the bound `cgroup.kill`, and observes its removal. Test uses an independent pidfd safeguard. | This is a deterministic fake cgroup seam, not a real manager job-completion acknowledgement or physical scope test. |
| Collector panic and caller-runtime destruction | Guardian runs on a dedicated OS thread with its own Tokio runtime, retains `Child`, and uses a shared retained-cleanup path. Focused panic/runtime-drop tests pass. | Tests use an unscoped child; detached scoped member and runtime destruction during held acquisition still lack a native isolated test. |
| Stop deadline | Two overlapping Stop calls and Stop during a held guardian startup return within their own eight-second entry budgets, deny restart, and later join cleanup. | These tests prove the simulated delayed startup/cleanup paths, not a full native/physical cycle. |
| Python launcher cancellation | SIGTERM during `Popen` is deferred until the returned child handle is owned. Before positive scope admission, `cleanup_reaped` cannot be true; focused real-child and mocked admission tests pass. | The launcher has no surviving custody after returning `cleanup_reaped=False`. A late standalone scope may still escape. |
| Focused source gates | Rust owner integration **22 passed / 5 ignored**; Python hardening **8 passed**; stage classifier unit **6 passed**; daemon all-target strict Clippy, workspace rustfmt check, and generated test-manifest check passed. | The 5 ignored tests require isolated user-systemd/native conditions. The full 25-case nonphysical stage and deterministic aggregate were **NOT_RUN on this changed tree**. Earlier 25/25 applies only to its old artifact identity. |

Independent post-change audit returned **CHANGES_REQUIRED**. In particular,
scope absence and launcher exit are not proof that a manager-side creation
transaction has reached a definitive negative result. The Rust guardian
therefore correctly remains `CleanupPending`/Busy when a scope was never
observed, but this can persist indefinitely. The Python standalone path
likewise lacks retained authority after an uncertain return. A trusted,
versioned manager-job completion fact and one explicit custody owner are
required before this stage can be approved; do not fabricate a negative
acknowledgement from polls, CLI exit, timeout, or SIGTERM. This blocker is
tracked as papercut `pc_d34fcd7ef597`.

The stage owner receipt registration now lists all 22 deterministic tests and
the test manifest was regenerated, but neither registration nor focused PASS
is product evidence. Production remains `AecCapability::Unavailable` with
`aec_calibration: None`. Native isolated AEC, physical 30s/60s admission,
independent RU/EN product holdout, paired quality/latency, native-app soak,
merge, and release remain **NOT_DONE**.

## 2026-09-30 iteration 3 custody continuation — nonphysical gate only

The post-review candidate now uses a manager-job protocol. The fixed launcher
sends `P` immediately before `StartTransientUnit`, `A` only for the exact
`JobRemoved(done)` job, and `N` only for a pre-request exit or definitive
manager rejection after cleanup. A sent request followed by EOF without a
terminal fact remains unknown and keeps the Rust guardian in custody. A
four-second Gio cancellation bounds pre-request connection/owner lookup;
post-request uncertainty is not converted to a negative result. An observed,
inode-bound scope is killed during cleanup even before the manager reply, but
Idle still requires a terminal creation fact and verified absence. The launcher
retains the standalone unknown case instead of returning a false cleanup
receipt. Production composition remains disconnected.

Focused checks on this continuation: Rust owner integration **29 passed / 5
ignored**, a separate new library custody regression passed, Python
AEC check/isolation/stage tests **37 passed**, test-manifest check passed, and
strict daemon all-target Clippy and workspace rustfmt check passed. The five
ignored owner-native tests then passed inside the private stage. A stage
attempt first stopped at owner-contract classification because its exact-name
oracle still expected 28 deterministic tests after the new regression was
added; this was a harness inventory failure, not an acoustic failure. The
oracle was corrected and its unit test passed.

The subsequent complete isolated matrix returned **25/25 PASS** on artifact
set `25d02391c99715e49c39e4a5211d5199f6aa8534ecd88391b6d67ff50b2f9a4a`.
Every case reported `frozen_unchanged=true`, including six 4500-frame acoustic
controls, two scorer mutants, ten confirmed post-witness fault injections,
100 lifecycle cycles, and five owner-native tests. The frozen set now includes
GI Python source/binary files, Gio/GLib typelibs and linked libraries used by
the manager request. All evidence remains `aec_proof=false`.

A prior exact private stage on this development line failed at
`speech-wrong-reference` after 2256 frames with a +480-sample native graph
position gap; a standalone repeat also failed. The current stage passed but
did not establish the intermittent gap's root cause. New numeric fixture
first-failure diagnostics are present for any recurrence; do not erase the
failed receipt. The native test-harness timeout path also still needs an
external process/scope-custody audit before claiming leak-safe failure
handling. No physical microphone/speaker path, 30s/60s calibration proof,
production attachment, RU/EN paired product quality/latency, native-app soak,
merge, or release was executed. The earlier WIP table and failures remain
historical observations; this section does not promote them to product PASS.

## 2026-10-01 product verification checkpoint — not a release gate

C2C iteration 12 accepted the offline callback-history R1/R2 correction on
staged tree `6270fda297cb6dca0daa2a38469c0d38f20c12c7`. Its review did
not verify the full deterministic aggregate or authorize a native graph rerun,
physical proof, production attachment, merge, or release. This checkpoint
adds no such authorization or evidence.

The product-phase source check found three timing assumptions in tests, not a
confirmed production failure. The preflight test exhausted 10,000 scheduler
yields while waiting for a worker; it now uses a bounded wall-clock wait and
checks the retry attempt ID and failure code. The concurrent shutdown test
waited one scheduler yield after releasing an OS thread; it now waits for that
thread's terminal write. The keep-alive HTTP test started its five-second
clock after reading the preceding response, although Hyper may already have
started the next header timer; the clock now starts before that response is
read. Runtime deadlines and admission policy were unchanged. Each original
failure remains in the local gate logs; a passing rerun does not erase it.

The final Rust inputs passed the workspace test and doc-test gates in the
sequential `RUST_TEST_THREADS=1` run. Later Python-only lint/format corrections
did not change those Rust inputs. The corrected Python lint and format checks
passed; manifest-bound pytest reported **1133 passed / 5 external skips** and
unittest **335 passed / 22 external skips**. Shellcheck, systemd unit verify,
schema bindings and the pinned supply-chain scan passed. These are composed
source checks, not a single green deterministic aggregate on the final tree.
The publication candidate receipt and manifest check must be read against
the final staged tree; neither result is release evidence.

The physical AEC product boundary is unchanged. The private runner hides
`/dev/snd`; its 48-kHz callback trace and virtual acoustic controls are not
hardware evidence. Production still sets `AecCapability::Unavailable` and
`aec_calibration: None`, while independent `parec` capture is explicitly
`MeasurementUnavailable`. A read-only device inventory found the USB
capture/playback card selected as the system default, so it is not an isolated
physical test path on this snapshot. No playback, capture, route change, or
production service change occurred. The independent RU/EN 120-pair holdout,
paired physical first-audible and quality comparison, real-app matrix,
30-minute soak, install/rollback, merge and release remain **NOT_DONE**.

## 2026-10-01 three-arm diagnostic preparation — no product result

The fork's saved-audio runner v2 now assigns distinct Small/NLLB,
Turbo/NLLB and Turbo/Hy arms. Small here is a same-code ablation, not the
untouched original `main` baseline. It checks public effective model health
at open and after each attempt, binds provider events to the submitted
utterance, preserves failed attempts and cleanup/health diagnostics, and
always writes a failed terminal record for ordinary pair-validation errors.
The header labels the 24-WAV Turbo-screened input as development-only and
limits voice evidence to a requested Piper profile plus generic model health.

Focused runner tests: **70 passed**. The adjacent provider contract, local
provider and local runtime tests also passed; Ruff and `git diff --check`
passed. No new model inference, physical audio, independent holdout score,
quality/latency comparison, or release was performed on this v2 tree.
High existing swap use at the resource preflight argues for a separate
bounded serial model run, not an unbounded benchmark in this checkpoint.

The follow-on Tatoeba metadata screen used official weekly audio and RU/EN
sentence exports only. With explicit CC BY 4.0/CC0 audio licenses it found
7,018 RU audio-linked sentences from four upload authors and 3,838 EN
sentences/3,841 audio entries from seven authors, before checking translation
links. Authors are not verified speakers; no audio, translations, input-byte
hashes, gender, accent or critical-case labels were established. This is a
feasibility signal, **not** a frozen EVAL-0 corpus or a mathematical proof of
ineligibility. Tatoeba remains unadmitted as a standalone release holdout.

A one-case guarded v2 runtime smoke on exact fork SHA `13557cbf6875868d`
then completed three of three `QUALITY_FIRST` female-voice attempts on one
previously exposed RU WAV and one of one three-arm comparisons. The private
`three-arm-smoke-20261001-nCmtKs/ru-71601.jsonl` receipt is SHA-256
`36812c7a4483a0fb72b15c9aea684e45ed2fee0d701e9f1bb56df279d4efbd4b`
and mode `0600`. Public health observed Small/NLLB, Turbo/NLLB and Turbo/Hy
respectively, with ASR and MT on CUDA and a generic Piper TTS identity. The
user scope requested limits of 9 GB memory, 1 GB swap, 200% CPU and 420
seconds; the manager accepted the scope, but post-exit properties
could not be independently re-read. The Hy child exited, the scope became
inactive, and sampled GPU memory returned to its pre-run level. No audio
capture or playback occurred. This smoke validates execution and cleanup
only; its accelerated saved-audio PCM timing is not first-audible latency,
and it cannot establish RU/EN accuracy, reliability or an original-`main`
baseline gain. No broad run was attempted with existing swap use.
