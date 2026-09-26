# Stage B2 working evidence

Base: `6c81cc318712e05b7d2339535eb720a4105859a7` on
`codex/stabilization-20260904`. This is an unclosed source stage, not a release
or live-quality receipt. Production stays at
`9291e8beafee3e02aaa179178ce460ac9e6c6de2`.

## Current SAFE-1 checkpoint

The results below supersede the earlier admission-design-only checkpoint; older
sections retain the evidence history. The stage remains open.

Compatibility correction: a read-only structural query of installed
`pactl` 16.1 found that module JSON omits `index`. Both the Rust Task7 inspector
and the new Python cleanup fixture assumed that field. Their prior deterministic
passes therefore do not prove installed compatibility. Short module output also
contains multiline arguments, so a strict one-record-per-line replacement would
be incorrect. The private shared short-format decoder now passes 2 parser and 94
adapter tests with one existing live-AEC skip, scoped Clippy and independent
source/SRP approval. It validates complete framing and IDs before absence or
cleanup decisions, preserves foreign multiline arguments, and replaces duplicate
Pulse/AEC decoders. Runtime decreases by 11 lines relative to the preceding
checkpoint (+1 source file; 56 inline test lines). An isolated PulseAudio16.1 run
now passes private-socket isolation and actual Python create/cleanup/idempotence.
Rust graph ensure fails with `module_load_failed` at daemon artifact
`e7fa3b83c1dccee3567133882b1e01a2e24823919c53155c615e7409bee4a5b2`.
The real command left nested property values unquoted. Canonical rendering and
ownership matching now pass 39 reported graph tests (one is a subprocess-helper
no-op), including a 40-case ambiguous-inventory matrix, plus scoped Clippy and
independent source/SRP review. Runtime decreases by 65 lines (+9/-74), with no
new files/dependencies. Only exact emitted records establish ownership; generation
text in any noncanonical record can only block cleanup, never authorize it.

Rebuilt daemon artifact
`2441da2fbe3d0fe9d5126106718ea99628c4642f4b0c8bf1a011ebffafc98da0`
now passes the unchanged private Pulse16.1 probe: Ready, exact three owned modules,
endpoint properties/journal, foreign-module preservation and idempotent cleanup.
All probe processes were reaped. Production remains clean at the original SHA,
active/running PID451512. It uses PipeWire1.0.5; migration of preexisting
noncanonical resources remains a separate release gate.
An earlier harness-only JSON-field error was corrected separately and is not
product RED. This is dependency conformance, not real translation quality.

Private PipeWire1.0.5 conformance now passes I1/I2/I3 on the same daemon artifact:
exact graph/journal identity, real Python create/cleanup, foreign preservation and
idempotence. The partial-start negative also passes: a failed compatibility child
leaves the isolated core responsive. Final probe `763886e8` rechecks Pulse16.1
successfully. All three runs confirm process cleanup; no PCM or provider runs.
The earlier empty-module baseline assertion was a harness error: PipeWire exposes
native core modules too. Its replacement compares exactly seven pre-start native
module serial/name/raw-argument records against `pactl`, rejects duplicates, and
retains zero Node/Device admission. Five pure fixture tests and independent
SOURCE/SRP review pass. No product oracle was relaxed. PipeWire receipt SHA:
`96d03920a83ac878a4f72bbfdd6cd907cfe7c17025a3a2d1081ca84415939fe4`.
These are development dependency checks, not final installed-package or acoustic
quality evidence. Existing production journal presence was checked read-only;
no production resource was migrated or removed.

The same pinned CLI schema exposed a separate device-facts defect: source monitor
links are strings in `monitor_source`, not numeric `monitor_of_sink`. Actual Task7
facts therefore rejected valid linkage; physical-source classification also
needed an explicit empty real link. Required serde string decoding, reciprocal
Task7 names and empty-only physical-source linkage now pass 55/55 device tests
and independent source/SRP review at source
`59c6e4507e5110700dfe4f26b5a96430defdca2269a187f97ecce212e2be26db`.
Runtime delta is +4/-5, no files/dependencies. A recovered changed-default test
proves the original microphone pin survives failed discovery. Main's raw fixtures
use the same real schema; the combined main suite passes 23/23. These results do
not close the whole SAFE-1 or B2 gate.

- Native launch/reconfiguration now consumes opaque `AdmittedDuplex`. One daemon
  policy checks raw device facts; audio no longer owns duplicate acoustic policy
  or the public wire DTOs. The headphone environment override is removed.
- Main's actual read-only facts adapter passes 5 new cases within 20 main tests;
  API 37/37 and control 17/17 pass. Independent source/SRP review approved main
  `9dda18f37c64abe78cc3367111669e560de6ef9215f0d1ffc79a14e0a8d55862`.
  Subsequent AEC tests prove actual successful load projects only
  `AvailableUnvalidated`, and an inspection error after loading retains the
  same graph for exact cleanup. Main now passes 23/23 plus scoped Clippy;
  independent source/SRP review approved main
  `f1b5a1c2c86318622c1d9c2a5fb6cda81eb492f4dffc3ba77ec7037aa85176da`.
  Lower AEC uncertain-response recovery and final daemon cleanup remain open.
- RoundTrip observes expiry before policy and acquires the shared gate before
  mutating the self-test session or preconditions. Three causal failures now
  pass, with independent source/SRP approval. Broader checks pass 36 internal,
  35 process and 9 controller tests. Missing facts and observed unsafe output
  have distinct test fixtures and error expectations.
- Task7's Rust command retains its graph and runtime lease through failed
  cleanup, retries with fresh eight-second deadlines and one-second pacing,
  and preserves the first failure after eventual cleanup. Expired entry invokes
  neither graph maintenance nor the runtime callback. Causal RED executed
  6 passing and 2 failing cases; GREEN executes all 8 successfully, including
  both graph-ensure outcomes. Independent source/SRP review approved source
  `e4fe4e09762b7a6d5b235b5f0c1ac0f03f6d21616480e8ec92e6a9e0e322d285`.
  Python temporary-input ownership now passes 74/74 tests (38 new, 36 existing),
  Ruff/format and independent source/SRP review at
  `7587afe3eb7a53d584cd7ca1acc7e6984f4c045140349dd0f8fa214aa87ca995`.
  The owner retains uncertain operations, checks the full real short-format
  inventory and nested unique owner marker, retries cleanup, and preserves the
  first failure. Literal property quoting replaces the fixture's invented
  implicit quoting. Runtime delta is +138/-47; tests add 460 lines.
- A composed daemon run passes 155 library, 10 mix application, 2 HTTP/SSE mix,
  5 Task7 wire and 1 latency-observer tests; scoped denied-warning Clippy passes.
  These counts precede the final policy correction and are not a whole-stage
  aggregate or a release receipt. The separate policy run now passes 10/10:
  both missing and duplicate RoundTrip directions return the self-test error
  family. Independent source/SRP review approved policy
  `0ebe4fcddbd7e5cff044c213d01feac5cad672ec1cda53373be7cb001cda886a`.
- After the module transport correction, a composed daemon check passes 156
  library, 23 main, 35 RoundTrip process and 9 controller tests. This precedes
  the final parser CR-header rejection, covered by its focused test; it is not
  a frozen integrated-stage receipt.

Read-only journal, physical provenance, command deadlines, immutable device
facts and read-only routing have separate reviewed packets. Audio focused runs
report 157 passing top-level tests and one declared live-AEC skip; nested child
results are not added to that count and helper no-ops are not behavioral proof.
The last device extraction reduces maintained runtime by 35 lines relative to
its prior checkpoint; the entire owned audio slice is +676/-284 against HEAD.
Missing-API compiler failures
are not counted as executed behavioral RED. Warm-host/cloud admission, routing recovery,
manifests, code-surface audit and the frozen aggregate remain open.

Production was rechecked on 2026-09-06: unchanged clean main at the SHA above,
user service active/running with PID 451512. No production mutation or live
quality/latency claim is made by this checkpoint.

## Retained stream implementation gate

D-072/D-075 are corrected in the candidate; whole-stage closure remains open.
The historical RED matrix `265b1cc7` has independent
test-critic/SRP approval and a separate root audit. It distinguishes seven
existing-API causal failures, ten missing-API invocations with zero contract
execution, and twelve passing characterizations. These are not GREEN or release
evidence. The implementation is admitted only in the fork.

The scenarios cover failed/cancelled opening, exact Local scheduler/publication
drain, failed backend disposal, repeated refusal/recovery, concurrent retry after
live Close or actual UDS disconnect, transport-stop ownership and immediate stop
admission. Refusal snapshots wait for the original opening task and an explicitly
failed whole-stream cleanup attempt before freezing identities and effects.
The literal Local-plus-UDS 100-refusal cross-product is not executed; composed
coverage is conditional on one shared transport reservation/lease-cleanup path.

Provider reservations own resource-drain proof; the server retains their exact
stream records and leases until that proof and lease release succeed. Registry
selection/disposal remains unchanged except for its reviewed retry correction.
The first core candidate executed208 scoped tests; final SOURCE/SRP review then
found D-079: an OpenAI drain could join the whole opening caller while that
caller's finally waited for the same drain. Two isolated regressions reproduced
this cycle at the held handshake/close boundary. The corrected reservation owns
only its opening-phase task. Failed task allocation closes the unscheduled
coroutine and removes only the original reservation. Both cycle tests now exit
normally, and separate100-iteration opening/receiver allocation failures pass.
Forced child termination in RED remains fixture cleanup, not resource proof.

Final independent bounded core SOURCE and SRP reviews pass. One fresh composed
run of Local, OpenAI, gRPC, local runtime and registry tests passes211/211 in5.61s,
with no skips. Log SHA is
`5bb6ab29a0d6e627f9585e0fbf59d1f273e623ff5fe8cce230efff7af0b6a6dd`.
Runtime fingerprints are grpc2dea71b7, registrye56d5160, locale93738bf and
openaie73dbb44; OpenAI tests2161de5e and gRPC testsb26718dc are included in that
exact-byte run. Core runtime delta is+391/-225 (net+166), including the prior
registry correction; it buys stronger ownership guarantees, not code reduction.
This is a scoped source checkpoint, not the whole-stage aggregate or live proof.

The startup-fixture FD defect is corrected. Prior collectible RPCState could
release shared gRPC sockets after the baseline; retaining a stopped wrapper alone
also allowed native poller descriptors to disappear. The fixture now keeps its
unbound warmup running across both snapshots and stops it in finally. Strict FD,
task, instance and stop-call assertions remain unchanged. Current-source evidence:
controlled prior-RPC GC2 PASS; six100-iteration startup faults plus repeated
cancellation7 PASS; deliberately retaining one additional FD fails22 versus21.
The diagnostic closes that exact FD in finally. Independent fixture SOURCE/SRP
and the composed211-test run pass; pc_e2fba4bc1ad6 is resolved. Passing reruns of
the old fixture are not treated as its repair.

## Benchmark caller checkpoint

The candidate corrects Task6's missing required provider identity and moves both
benchmark callers to exact session reservations. Open, frame submission, evaluation
and result construction now share cancellation-safe cleanup. Delivery failure
cannot produce successful metrics; an operation failure remains primary when
cleanup also fails. An unrelated caller exception cannot become the primary error.

After the minimal Task6 request correction, twelve causal failures were reproduced
alongside fifteen passing controls. The corrected caller suite passes31/31,
including four actual LocalProvider/InferenceScheduler compositions with injected
model adapters; unchanged Task6 tests pass37/37. These are68 distinct tests, not
live-model quality or latency evidence. Runtime fingerprints are podcast65523253
and task69ecf42d6. Three existing files changed: runtime+95/-75 (net+20), tests
+315/-2; no files/dependencies added. Final independent SOURCE and SRP reviews pass
for those exact runtime/test fingerprints and the recorded Local dependencies.
Core provider changes can invalidate the Local composition fingerprint.

D-077 remains open: outer benchmark shutdown/loop ownership is not fixed by this
per-session correction. Source tracing also shows Task6's outer ExitStack can
close models after a failed provider drain, so correcting only the bridge finally
is insufficient. Podcast duplicates model-release decisions already owned by
LocalProvider. The terminal ownership design must cover these outer boundaries.
Production activation, packaging and release did not run.

### Python voice and process-free telemetry composition

Direct Python voice admission is now corrected as well as daemon admission.
An ordinary nonserialized VoiceProfile property owns field-presence detection;
Local/OpenAI reject before allocation, preflight preserves its existing
provider/consent/credential priorities, and Piper reuses the predicate without
changing preset selection. All48 causal refusal failures and10 missing-property
rows pass; final six-file347 PASS includes the actual UDS no-effect/lease/repaired
UUID assertions. Independent SOURCE and SRP pass. Runtime delta+19/-11, net+8;
no new files or capability. Packet e97c5d5b records exact source/test hashes.

The Task6 measurement adapter now uses pinned nvidia-ml-py13.610.43 instead of
nvidia-smi and CSV parsing, preserving the existing process sampler and lock.
Per-sample NVML lifetime, unavailable data, single-device selection, whole binary
MiB and held-call exclusion have19 focused tests; these overlap the full54 Task6
tests. Independent bounded SOURCE/SRP pass on task6bcbcebef/tests6dd46c75. Runtime
net-5, tests+126, pyproject+1/lock+11; the binding is the only new dependency and
all prior lock resolutions remain unchanged. No real NVML/GPU call was made.
The old command timeout=2 is not preserved by synchronous NVML: release remains
blocked until the planned process-contained benchmark run supplies its deadline.
D-077 therefore remains open; no timeout-thread fallback was added.

One fresh complete Python sidecar run on the combined voice/NVML tree passes
937 tests with one declared model-cache skip in16.61s. Log SHA:
`e7de8e2d9281b0442c946e641738e4841fbb94998ae52965b85400e62848fec3`.
This covers registry, local-runtime and benchmark consumers as well as the
previous focused scopes; those counts must not be added to937. It is not the
formal18-gate whole-repository stage receipt, whose manifests still await B2
completion.

The skipped test was separately attempted read-only, not used to run GPU inference.
Default cache root `<default-model-cache-root>` does not exist. With
explicit existing root `<explicit-cache-root>`, integrity verification reaches the
missing pinned large-v3 snapshot `edaa852ec7e145841d8ffdb056a99866b5f0a478`.
Thus real-model cache acceptance remains unpassed; no files were downloaded,
copied, repinned or modified. Logs are b2-model-cache-readonly-integrity.log and
b2-model-cache-explicit-root-integrity.log in the private evidence directory.
This is not a reason to bypass model availability, acquisition-budget or final
quality/latency gates.

### Scheduler terminal retry

D-080 is corrected at scheduler source `bd47fb41`, tests `5b0b7154`.
Seven causal RED cases now pass: either/both executor failures, failed/cancelled
dispatcher, failed task allocation and a cancelled cleanup task before its first
step. The pending cleanup attempt is shared; later explicit calls can retry
terminal failure without replaying work. The dispatcher is joined after admitted
jobs, then both original executors are attempted. A stable safe error reports
failure without exposing the original exception chain. The existing shared
finish_cleanup replaces the scheduler's duplicate cancellation loop.

Independent SOURCE and separate SRP pass on packet `e81d3a1f`. All27 scheduler
tests and201 consumer tests pass, with real held native-thread and exact join
oracles. Runtime delta+28/-20, net+8; no new files or dependencies. Tests add320
and remove one import line; all19 old test bodies are unchanged. This strengthens
retirement correctness, not code reduction or finite native execution.

A fresh complete sidecar run after this change passes945 tests with one existing
external model-cache skip in16.57s. Log SHA:
`ff4141d425748fa2f8ba7110eb183bb6979a1b068365ecdd2d4d09399660255d`.
This supersedes the937-test composition for the current tree; focused counts
overlap and must not be added. The separately observed missing pinned large-v3
cache remains unresolved. D-077 process containment and D-081 input-command
deadlines are still open; no production, packaging or model-inference run occurred.

## Benchmark caller isolation checkpoint

D-077 is corrected in the candidate. Models transfer once to LocalProvider;
bridge shutdown retains private submissions and its original loop thread until
terminal completion. Failed cleanup invokes explicit worker-fatal authority
before unsafe stack/loop unwinding. One fresh-process supervisor stages reports,
waits/reaps the worker, validates the transport envelope, then atomically replaces
the destination. Failed quality thresholds remain valid diagnostic output.

Source review found a SIGINT window before executor thread registration/Future
return. Two real-signal startup regressions reproduced it. The final supervisor
uses a default-main SIGINT guard through executor join and a directly shared
cancellation receipt; it changes no OS signal masks or custom signal policy.
The podcast CLI uses this same sync owner; its public async API is preserved.
Input preparation occurs once, and invalid CLI limits precede directory effects.

Final independent SOURCE and separate SRP pass on Task6 `ebc10594`, podcast
`b7022e13`, supervisor `728c530c`. Caller113 and process43 tests pass. Process
checks include real repeated SIGINT, cancellation during startup/publication,
ignored TERM escalation, exact reap, old-report preservation and actual podcast
CLI composition. A real worker dispatch with an unknown ASR candidate produces
the expected skipped report using synthetic PCM and an empty cache; it is not
a model-quality result. Native TID disappearance is a separate bounded observation
after Python-thread join and child reap, not a strict kernel-at-return claim.

A fresh complete sidecar run passes1016 tests with one existing external-cache
skip in33.31s. Log SHA:
`b3d7f2f570d6c29c257499c59546b63daa850575d22a548310859cec6a146f60`.
The sorted source/test file-hash stream has SHA:
`5198ac257283981702cdc513d81acbb316fd8d6612c8263e47bf5810170f16b5`.
This supersedes the945-test composition; focused counts overlap, not add.

D-077 runtime delta is+552/-249, net+303 across three files (one added, none
removed): new supervisor281 lines and caller net+22. This exceeds the45..150
planning estimate; independent review accepts the stronger containment property,
not a refactor-reduction claim or daemon code-cap waiver. The publication manifest
includes the new module/test/fixture. No model inference, GPU call, production
audio, merge or release occurred. D-081/D-082 input preparation, missing pinned
model cache, native OS-unreapable waits, parent death and VRAM reclamation remain
outside this proof. B2 still needs its broader daemon/readiness/consent gates.

## Provider registry retry checkpoint

D-074 is corrected in the candidate. A released lease can explicitly retry failed
backend retirement without decrementing again; concurrent callers join one attempt.
Ordinary collection does not silently retry failed disposal. Finalization references
the exact awaited task, preserving its original failure even if terminal shutdown
has already admitted another attempt. Selection and disposal remain registry-owned.

Tests-first evidence includes two provider-ID retry failures and one deterministic
completion-versus-retry race. Final seven registry tests and three existing gRPC
retirement/reselection tests pass, with focused lint/format and independent SOURCE
and SRP approval. Runtime fingerprint is
`a53a83a9d86aacad77e9108f85aae3d5e1d4093f96672c5f6b4d2ec48dd22732`;
registry test fingerprint is
`cbee9970680fbfd56c9f118567dec37ea149dc77009e1cb4d2e54c5664140962`.
Two existing files changed: runtime +19/-12 (net +7), tests +151/-0; no files added
or removed and no custom-runtime reduction claimed. The increase buys exact retry
ownership, not a refactor-only optimization. D-072 session-resource proof, the
stage code ceiling and release gates remain open.

## Duplex local-quiescence checkpoint

D-073 is corrected in the candidate. The existing coordinator snapshots exact
affected active/retained cells, consumes each worker completion and resets its
observer before allowing that cell's local PCM cleanup. Borrowed library futures
let a healthy peer progress while another worker or PCM stop is held. Provider
close begins only after all affected worker/local receipts succeed. Candidate
replacement and unaffected-direction ownership stay separate; original deadlines
and exact retry targets are preserved.

Eight causal REDs preceded implementation. Final native module 75/75 passes;
root's composed gate passes 238 distinct tests: library165, main23, RoundTrip
process35/runtime9, Task7 bridge5, translation runtime integration1. The75 are
included in165, not additive. Scoped clippy and independent SOURCE/SRP pass.
Final source `65672949b08e0751a072cb538200da06bc7ecc6de4f08839eecf51b9341a1597`.
One existing file changed: runtime +182/-181 (net +1), inline tests +634/-21;
no files or dependencies added/removed. No net code-reduction win is claimed.

The composed build regenerated the daemon binary to
`33483dbdc6a1c967c1222199b01c566641d933cdf8573a43f6dc0fa5fbc017d9`.
Earlier private Pulse/PipeWire conformance used the preceding `2441da2f` artifact;
it remains scoped audio-graph history, not proof for a final installed candidate.
This gate does not prove control-loop preemption, warm model-host behavior, cloud
revocation or live quality/latency. No production mutation or release occurred.

## Voice override admission checkpoint

Daemon configuration now rejects unsupported model_path/provider_voice_id values,
including empty strings and stored disabled-direction profiles, before candidate
effects. Language and both-disabled error precedence remain unchanged. A shared
core presence predicate leaves generic wire serialization intact; built-in voices
remain supported. The debug-capture API reuses the existing error conversion rather
than a second mapping table.

Four causal failures and six passing characterizations preceded implementation;
the missing predicate initially had zero executed API-contract tests. Final scoped
gates pass192 distinct tests, including the previous75 native tests unchanged plus
two native voice characterizations. Clippy and independent SOURCE/SRP pass for
packet3e11add6. Nine existing files changed; runtime+19/-22 (net-3), tests+696/-0,
no added files/dependencies. The daemon-only reduction is9 lines; the broader code
ceiling remains unmet. No direct Python IPC override support or rejection is proved.

The resulting debug daemon fingerprint is
`7f7ecb62f3449d36b4853585a59a00a7dde74cdfc96980d3441930821d804696`.
Earlier private audio conformance is not validation of this rebuilt artifact.
No installation, production change, model evaluation or release occurred.

## Ownership and contracts

The control migration consolidates accepted commands in `ControlApplication`
and runtime handles/cleanup leases in `RuntimeSupervisor`. Caller cancellation
does not cancel admitted native work. Reconfiguration must commit only after
the affected native workers are replaced; failure must restore the prior
configuration or retain explicit failed-cleanup ownership.

`AudioMixApplication` owns desired/effective volumes, physical transaction
ordering, reverse compensation, and unknown-state recovery. `PulseAudioMix`
only discovers typed targets, sets percentages, and restores exact raw channel
volumes. The actor derives bypass/translating mode from the supervisor when it
executes a command; a watcher cannot supply a stale desired snapshot.

Stopped service uses original 100 / translation 0 without replacing the desired
UI mix. Failed native Stop must not apply bypass. Successful native Stop followed
by failed bypass remains stopped and reports failure; it never resurrects the
runtime projection. Failed compensation blocks normal mix work until explicit
recovery succeeds. Recovery uses the last committed desired mix, not observed
raw values or a previously rejected candidate.

Shutdown closes ordinary admission and drains accepted work. After confirmed
native cleanup, it reconciles bypass; only an Unknown result admits one explicit
shutdown-owned recovery attempt. Failed cleanup retains the actor for retry.
Main closes the shared gate and HTTP admission before runtime drain. It awaits
one round-trip shutdown attempt at a time and retries only typed CleanupPending,
with a one-second delay after each finished failure. OwnerFailed or a joined
blocking-task panic returns a terminal failure. Independent translation,
debug-capture, event, and server owners still drain. On that fatal path, the
round-trip, route/graph, and runtime-lease owners stay held by an explicit
pending future; dependent cleanup and graceful-exit reporting do not run.
Systemd's cgroup kill/restart and recovery journal still require live evidence.

`SidecarSupervisor::restart_generation` owns direct generation replacement even
when a dead child has already lost readiness and registered sessions. Both that
entry point and the existing close-timeout branch use the same reap, socket
cleanup, generation rotation, and matching probe path.

HTTP connection ownership uses the existing locked Hyper and Tokio libraries.
The `JoinSet` is both the admission counter and connection-task owner: at most
64 retained entries, continuously reaped. There is no second semaphore or
custom HTTP parser. HTTP/1 header reads have a five-second deadline; shutdown
stops admission and drains, then aborts and joins remaining connection tasks.

## Focused evidence

These results precede the final integrated tree and must be invalidated when
their relevant inputs change. The B2 aggregate has not run.

| Surface | Before | Focused result | Still required |
| --- | --- | --- | --- |
| Native HTTP | connection 65 and slow-drip deadline tests failed | 6/6 TCP integration cases pass; 1/1 observer-backed test covers 1024 partial clients, excess rejection, exact slot/FD recovery | integrated shutdown and frozen aggregate |
| Audio mix transaction | missing rollback and writes without complete prior volumes reproduced | 10/10 application cases and 3/3 primitive cases pass, including mono/stereo raw rollback and Unknown recovery | frozen shared-core integration review |
| HTTP/SSE mix integration | joined shutdown retained translated physical mix | 2/2 real actor/application/Pulse-command tests pass; failed patch leaves HTTP/SSE desired unchanged; cancelled HTTP/watchdog work drains before physical bypass | broader lifecycle faults and frozen aggregate |
| Main shutdown sequencing | premature cleanup and indefinite retry of a dead round-trip owner reproduced | 15/15 main tests pass, including typed fatal versus transient results, admission order, independent drains, and retained fatal-path owners; clippy and scoped source/SRP review pass | native-control reintegration and standalone/systemd lifecycle |
| Sidecar generation restart | close-session restart requires readiness that child death revokes | existing close-timeout characterization passes; 15/15 owner cases and 1/1 real private process/UDS crash-restart test pass | full native direction coordinator recovery |
| Control/native workers | cancellation, failed-stop ownership, provider/reconfiguration defects reproduced | 16/16 focused native cases pass, but independent source review returned NEEDS WORK | cancellation-safe owner/join, bounded notifications and effects, atomic activation, exact terminal publication, admission policy, code-surface gate |
| Round-trip cleanup | failed cleanup released the gate; Completed preceded joined cleanup; peer absence discarded un-restored route ownership; shutdown mailbox pressure left Start open | owner unit 6/6 and integration 8/8 pass with scoped source/SRP approval; combined main/process/runtime run passes 15+26+8 | lower Pulse transaction, native adapter fault tests, live integration |
| PCM capture/playback | cancelled finish lost its child; cancelled capture discarded a consumed prefix; stopped capture could emit buffered audio | capture RED reproduces two defects, EOF characterization passes; all 8 PCM unit cases, denied-warning clippy, and scoped capture source/SRP review pass | native select integration |
| Task 7 bridge | failed-start/Stop cleanup owners were dropped before graph cleanup | post-format 3/3 cases pass, including paced retries and preserving the failed benchmark outcome | live latency/accuracy evaluation |
| Panic privacy | default panic hook printed payload before catch_unwind | isolated three-panic marker test passes 1/1 with barriers around concurrent installation/emission; scoped source review approved | integrated gate |

The initial exact Rust selector omitted its module prefix and collected zero
tests. That invocation is excluded from evidence. The valid characterization
used `tests::stopped_translation_uses_audible_bypass_mix_without_mutating_snapshot`
and executed one test.

HTTP focused source/SRP review approved hashes
`e7b177cd8bbe3b01647b0fdc5d3b32b77e1f229188354c2a9ed6d0d75159db40`
(`bounded_http.rs`) and
`f80f647fd9f2a6ab8494ae63a17baaf724b920f0abdd4c10f71cc8126434e880`
(`tests/bounded_http.rs`). This is not an approval of the whole B2 tree.
The mix amendment's separate BDD critic, auditor, and preimplementation SRP
checks passed. The mix primitive/application, sidecar generation extraction, and
main drain helper received scoped source/SRP approval. Shared core integration
and the exact whole-stage tree remain unapproved.

The agent-collab MCP runtime was unavailable. Local independent reviews are
recorded as local reviews, not runtime routing or certification receipts.

Main fatal-owner RED executed one test and failed on the intended bounded
completion assertion after a real, resource-free owner-thread panic. Log SHA:
`dbde503e1019710e5dd5084ae626cbe6f95a5afa3b57d75998860f56c3f36a14`.
The focused GREEN log (15/15) is
`f24bf4b00c4e0483be9b71a3598e6d0d80e1a9f69f4b24d61d6b4ad9486ff7b7`;
main source is
`ba1c94e40ef16501d15769a61375338351ad54b35f9fb43e2ed8bd5c07e2e904`.
These are in-process ownership tests, not a claim of safe OS-process exit.

The HTTP FD oracle originally counted process-global descriptors alongside
parallel library tests and failed spuriously (15 versus 13). Its unchanged
64-slot, 16-wave, exact-FD body now runs in an isolated, deadline-bounded child.
The parent kills and reaps on timeout and requires proof of one executed test,
not just exit code zero. Focused GREEN is 1/1, log SHA
`486e173a53d8f4ade926c4a8e8d71aebc6aed8b410ec9223f7fb6bbf3aa5d8fe`.
The broader parallel-library rerun passes 61/61, log SHA
`fb7794ad84e0cd8ee22edbb25761321f152cf89b3b14246b3455e99385267605`.
The isolated test source received independent approval. This suite still does
not cover the newly identified native-core audit failures.

Round-trip owner startup now acknowledges a successfully built owner-thread
runtime before publishing availability. Full mailbox, timeout, disconnection,
and terminal thread failure have distinct typed outcomes. Shutdown closes Start
admission on its first call even if the mailbox is full. Panic during Stop keeps
the exact runtime and lease for retry; the application never returns to normal
admission after panic or shutdown. Scoped source/SRP review approved runtime
`3e9807192628e32bcc868bc91a9607715736ca44d134e562ab9c052e31075fbe`
and integration tests
`4ed27962553e1ec40fedf76064588bf95eece3084f1b2220bd27dea6c2134a1a`.
The owner correction adds 179 production lines over its pre-correction slice;
this buys new ownership guarantees and is not reported as code reduction.
Combined main, round-trip process, and round-trip runtime tests pass 49/49;
log SHA
`626037ebf6561aaac7e9f8ba1061cc11cb29fa0e1dafc79ed87172e367b1edfe`.

PCM capture now owns a fixed frame buffer, filled offset, and first-byte
sequence/timestamp. It uses the locked Tokio cancellation-safe `read` rather
than `read_exact`. Tests observe exact pipe byte counts before and after two
cancellations, then verify both complete frames and original metadata. Stop
closes stdout and clears pending state before awaiting child reap. The source
`e26bc15b47a524d3719156c89e87e98887d4509ce4af23a50510be8a869befce`
received independent approval; GREEN log SHA is
`3f154445437db69d9d048d678ecfb37812543e4db299dc5e337d1bb0fd8e75a3`.
This proves framing and semantic invalidation, not cryptographic zeroization
or a physical PulseAudio run.

Generation retirement now has persistent exact-UUID reap evidence, retained
through cancellation, socket cleanup failure, and replacement start/probe
failure. Public readiness and session admission stay closed until the native
owner acknowledges disposal of old provider handles. Physical probe readiness
is private; acknowledgement checks child liveness before publishing readiness.
Behavioral RED reproduced three defects (15 existing passes); the separate
intended-interface RED contains only missing receipt API errors. The 22-test
focused suite passes, including successive generations, wrong/stale/early
acknowledgement, three cancellation phases, and replacement probe failures.
Source SHA is
`e676b40dbf0bcb2fde2fdc15fb78ee45259717d234aad277ea7a605891c163e0`;
GREEN log SHA is
`0c8f70a873d7cdc64d41459868b15c60a4a244968d8d093a5693e7e645dff00e`.
The real private Python/UDS crash-restart check passes, rejecting the old token
and confirming child reap/socket removal (log SHA
`b501303ba39ea3418223804a8530c2ba9d2fe957a48a56b166bf3cbf34cb4d2a`).
Focused Clippy passes. Independent source/SRP review approved this exact
supervisor source. The native coordinator's actual discard-before-ack
integration remains a separate gate. Reusing public start after shutdown on
the same supervisor can retain its launch identity; production currently uses
a fresh supervisor or explicit restart. This is not a general restart guarantee.

Additional transport characterization passes without production IPC changes:
three client tests exercise six typed gRPC status categories at both initial
RPC and post-first-event boundaries over private Unix sockets. Clean response
EOF is checked separately from an explicitly acknowledged remote connection
shutdown followed by request-channel closure. The literal expected errors and
private-marker exclusion do not derive their oracle from the production mapper.
Client source SHA is
`183cfa25faf3e1aa4b2bbe37ecf96c6f4313922986a43109d1ad5c3c0d808922`;
log SHA is
`aed4c428417be3926659721e5677f0a84c4bdff95b8f96f702f178b22797ecac`.
The complete IPC package also passes 3 unit and 26 integration tests (zero
doctests are reported separately), log SHA
`4bb4b766df33af0281157f175a494881202e839309e94639f452801cde31370a`.

The eight-test private process suite passes (final hardened-fixture log SHA
`f5d01b11307afcc87859d30eb0d28a3204b33a8989749e92570f5788903bf9d2`).
Its new Python fixture gates only after the real open handler has committed
session and lease ownership. Rust proves open remains pending and cancels at
timeout. Before server shutdown, Python observes that the active session,
retired session and scheduler entry are absent and the lease count is zero.
Rust then reopens and explicitly closes the same UUID. This is cross-language
cleanup characterization, not audio/model E2E or a test of a provider that
partially commits and then fails inside open. Review found evidence defects:
ambient Python environment could disable fixture assertions; forced reap and
EOF needed bounds; socket mode/owner needed a direct check. The fixture now
uses a cleared environment, explicit non-assert guards, bounded reap/EOF, and
socket type/owner/0600 checks. It also passes with parent PYTHONOPTIMIZE=2.
Independent final test-source/SRP re-review approves the hardened process test
`9e7a00e1ce609e2c5b3b2682044d85fb0a45796f352ff6dae6fdedb0e6f8d35f`
and fixture
`a43325c958646426a2b6e151186b3cbdeede39f0318b74299b1708d0bfec0167`.
Split package-scoped Clippy and
fixture Ruff checks pass; the earlier
combined all-targets Clippy accidentally compiled unrelated CORE RED tests and
is excluded.

The native hot-I/O bound projects the current watchdog phase deadline, not an
immutable source-age deadline. Source inspection confirms the current watchdog
resets after EOU and audio deltas, and IPC expiry uses input completion time.
Those semantics remain an explicit Stage C cross-language migration; the B2
getter and min-bound cannot certify first-source age or close that defect.
The read-only getter chain passes 32 direction/watchdog tests and scoped Clippy,
with independent source/SRP approval. The watchdog alone selects the current
deadline; coordinator and direction wrappers only delegate. Production delta
is +27/-0, tests +160/-0, no files or dependencies added. GREEN log SHA:
`f81b191609c5c7552e16eec087c52b20fe2ab4f2a3730dd087217d87cf395227`.

## Sidecar retirement cancellation checkpoint

D-083 is corrected in the candidate. Current coordinator deadlines could cancel
graceful retirement after ProcessSidecarRuntime moved Child into its future.
The owner now borrows Child throughout retirement, including group-disappearance
checks and errors, and clears it only after success. Graceful escalation reuses
the forceful retirement path. No second reaper or deadline policy was introduced.

Two actual-child cancellation regressions failed on the preserved baseline and
now pass; a third test characterizes completed forceful retirement. The fixture
binds a pidfd before cancellation and freezes Child/PGID/reap observations before
independent repair. Repeat cancellation, exact-owner retry and idempotence pass.
Forceful pending-wait cancellation remains source-backed by borrowed Tokio wait,
not a controlled kernel-wait test. Permanent process leakage was not established.

Independent BDD critic, distinct audit, pre-RED SRP, RED critic and final
SOURCE/SRP all passed. Final process source SHA256:
`dad2d8aa66cf67dd908a22cb955ef2768ee63270d605aeff770a4e12f8f96d1d`.
Focused3 PASS log `6565686a10e773f0d609df677741e726a217fe137a15dbaf2f7afd92f6f85965`;
broader211 PASS/0skipped log
`84e88bc52c6f84105f12319099e5a3bac743b0490a9f9d9a4ea2813ac253c996`.
The latter includes174 library,8 private-process,22 supervisor and1 runtime test;
the focused3 overlap174. Runtime delta +8/-14, net-6; tests+180. One existing file,
no new dependencies or production files. This is not the final B2 aggregate.

Broader Clippy also found three pre-existing Default-field reassignment warnings
in acoustic test fixtures; the earlier voice check covered only library runtime.
A separate mechanical fixture edit preserves all assertions and runtime policy.
Final fixture source `ce827b0a` passes12 scoped tests, log `c96ec072`, with
independent SOURCE/SRP PASS. Strict daemon `--lib --tests` Clippy passes, log
`13890264`; no lint suppression is used. Test-only delta+15/-9, net6. The211
composition above precedes this fixture-only edit; it was not rerun or relabelled
as a new whole-suite receipt. Its changed tests were verified separately.

READY/CLOUD remains under design. Pending Local Open needs a remote drain receipt
even when native transport construction was cancelled; cloud-only revocation is
not sufficient for logical Stop. Failed-candidate invalidation must preserve a
still-valid previous grant for ordinary rollback, whereas accepted Stop/Revoke
must invalidate both. No READY ABI or implementation approval is implied here.

## Open stage and release gates

Acoustic admission, cloud consent/revocation, selected-provider admission,
incoming-route ownership, round-trip ownership, and service reconciliation
remain open in B2. Shared lifecycle surfaces are changed serially.

The corrective native contract now has independent tests-only RED approval;
implementation and source review are still in progress. Its public adapters
carry absolute transaction deadlines: four seconds for Start/reconfiguration
and eight for cleanup. Round-trip has a separate ten-second public response and
thread-join deadline; its admitted inner/outer pair must survive queue delay and
reattachment without resetting either clock. The mechanical adapters compile:
Task7 characterization passes 4/4, main 15/15 and HTTP/SSE mix integration 2/2.
Independent source/SRP review approves only the Task7 adapter. Shared native
and round-trip behavior remains under correction; these results are not the
frozen B2 aggregate. The historical 120-second provider readiness
and 130-second startup acknowledgement allowed cold model preparation inside
Start. The new short transaction therefore requires selected-provider
preparation and cold/warm measurements before release; fast rejection does not
prove faster successful translation.

Round-trip deadline and terminal-truth corrections subsequently passed 66
focused tests and scoped Clippy, with independent exact-source/SRP approval.
These cover queue-time budget consumption, reattachment to one admitted
deadline pair, exact lower-level propagation, route/audio factories consuming
startup or active-session time, and worker panics both before and after the
terminal receipt. A disconnected receipt channel proceeds to the existing
bounded join; a timeout remains retryable. Failure is cached before destroying
a fallible panic payload, and a missing receipt never becomes Completed.
The macro's original 100-line growth ceiling failed. An explicit independently
reviewed correctness amendment permits at most 175 additional production lines:
the final prefixes are 685 + 1686 = 2371, or +162 over its 2209-line baseline.
This accepts distinct tested deadline/terminal guarantees; it does not change
the native CORE limit or certify the rest of B2.

The direction-reset adapter subsequently passed 76 focused RoundTrip tests
(33 owner, 35 process, 8 runtime) and independent source/SRP review. An active
relevant reset fails and seals the evidence chain; frozen cleanup is unchanged.
Terminal/playback waiters check Failed/Stopped before stale success and use the
existing checkpoint watch for notification. The duplicate incoming-terminal
watch was removed. Final prefixes are 1704 + 685 = 2389: this adapter adds 18
lines against its separately reviewed 40-line ceiling; cumulative growth is
180 lines from 2209. Root's separate source audit also approves the scoped
adapter. No native CORE ceiling or live-evidence requirement changes.

Source review confirms that each production Start creates a sidecar and each
Stop destroys it, so even repeated translation sessions can reload models.
Selected-provider admission therefore needs a daemon-lifetime provider host,
with session/PCM cleanup separated from final model-host shutdown. Local
bootstrap currently gates a global readiness flag and can delay unselected
cloud startup. Cloud consent is also still a projection boolean retained across
Stop, not the documented revocable generation-bound capability. The host,
readiness and grant migration remains open; no cold/warm or zero-egress live
result is claimed.

Read-only acoustic revalidation confirms that Start's facts refresher can
mutate the graph, routes and projection before admission. Native target
resolution also requires a microphone for incoming-only and accepts unsafe
physical fallbacks. Existing tests assert those fallbacks; they are not the
documented SAFE-1 contract. Module loading currently manufactures an AEC
validation fact without acoustic measurements. These remain unfixed until the
serialized fact-discovery and acoustic-admission migration is implemented and
verified across native, round-trip and projection consumers.

The typed `DuplexStartFailure` now carries unfinished startup cleanup to the
control owner, round-trip, and benchmark bridge. The real native coordinator's
partial resource acquisition, panic recovery, per-direction fault scope, and
cleanup-only state have focused GREEN evidence, but failed independent source
review. The audit found an unbounded completion mailbox, duplicate/premature
terminal publication, lost provider-error classification, exhausted-direction
resurrection, unbounded or non-preemptible effects, and partial activation still
reported as running. Cancellation of accepted control shutdown and loss of all
senders also need retained cleanup/join ownership. These require a corrective
contract and additional regressions; 16 passing tests do not certify this owner.
A blocked cleanup must not
be hidden by Drop, force-release, or a clean-stopped projection.

The corrective native implementation passed its focused lifecycle, HTTP and
recovery/deadline checks, but the next source audit found further gaps: an
exhausted direction can publish Failed with an already-consumed Running epoch;
mode-change playback drain can wait without the hot-I/O cap or Stop priority;
latency correlations survive local direction cleanup; and an expired Tokio
timeout can poll a ready effect before its timer. Their amended tests-first
contract is independently approved, not yet source-approved. It preserves the
original latency timestamp on duplicate SpeechStarted and distinguishes one
immutable cleanup attempt from a later explicitly admitted retry. A separate
owner-retention correction passed two focused tests: idle actor abortion must
not drop an active or cleanup-pending runtime and its audio lease.

Directional correlation eviction and the coordinator's post-join callback have
two focused passing tests. Task7 now forwards that callback to its latency
observer. Its new regression first proves the direct observer clears microphone
samples and preserves the speaker, then reproduces the wrapper-only failure.
After the three-line delegation, all five Task7 binary tests and scoped Clippy
pass; independent source/SRP review approves that exact adapter. No latency
threshold, NDJSON event or mode decision changed. Full native P1 proof remains
open; the RoundTrip observer adapter is now independently source-approved.

The earlier native source freeze passed 62 runtime unit tests, 10 control unit
tests and 58 owned integration tests, but source review rejected two
deadline boundaries: PCM acquisition after provider readiness consumes the
deadline, and shared recovery mutating a healthy peer after an expired probe.
Direction-level fakes did not prove these internal adapter boundaries.
A real private-UDS and isolated harmless-child characterization passes. Its fixture publishes
process identity only after the final exec and kills/reaps the retained outer
child before checking orphaned helper absence. The first two fixture revisions
are excluded. A private concrete-PCM spawner extraction preserves the real
constructor path: both characterization and no-bypass checks pass, with
independent source approval only to proceed to causal deadline regressions.
Those regressions subsequently reproduced late Ready/capture acquisition,
post-probe expiry and expired shared-recovery entry. The correction guards
each boundary and retains acquired PCM for cleanup. Exact-source review approved
the correction; 69 runtime tests and 56 owned integration tests passed.

Before serde consolidation, a new cross-language characterization serializes
99 runtime and 10 control cases through the actual Rust wire type and unchanged
Python parser/stream. It checks exhaustive runtime fields, all current enum and
optional-field combinations, exact raw keys and timestamp origins. Five bridge
tests and the five existing Task7 binary tests pass. This is passing
characterization, not a bug RED or live latency evaluation. The subsequent
serde consolidation removes the copied event/field mapping while preserving
the wire envelope's sole timestamp. Both five-test suites and all 69 native
runtime tests pass after this change; scoped Clippy and independent source/SRP
review pass. The measured production-prefix reduction is 130 physical lines.

SAFE-1 remains open. Seven causal policy regressions failed against
the frozen legacy resolver, with two positive controls passing. They cover unsafe
physical fallback, headphone AEC substitution, incoming-only microphone/AEC
requirements, inconsistent physical selections, both-disabled admission and
unknown output accepted with an AEC record. Full typed-policy, fresh-facts,
HTTP and forbidden-side-effect proofs are still required. The oracles are being
migrated to the daemon's typed policy, with optional targets for disabled legs.
The current working tree is not an aggregate passing candidate.

Bounded SAFE-1 corrections have separate source/SRP approvals:

- Journal inspection no longer initializes directories/lock files, changes
  permissions or waits on a held lock. Nonblocking descriptor opens reject FIFO
  lock/journal paths before reading. All five causal regressions pass; the
  31 passing audio-graph invocations include one helper no-op. Remaining graph
  deadline propagation and generation-intent recovery are not covered by this
  receipt.
- Physical endpoint discovery uses one backend/class predicate, rejects
  conflicting virtual/network/media metadata and duplicate names/IDs, and does
  not require device.bus. All 21 negative rows and five positive backend fixtures
  run; 28 device-watcher tests pass. This is synthetic metadata proof, not
  hardware authentication or real-device compatibility certification.
- Command execution requires an absolute deadline, checks it before spawn and
  distinguishes caller expiration from its local two-second liveness bound.
  Direct-child and output-reader ownership is retained through cleanup. Seven
  command invocations pass, including two helper no-ops; three isolated probes
  cover short, repeated and local-limit timeouts. Descendant-held pipes and
  arbitrary OS syscall latency remain outside this bounded receipt.
- The desktop environment parser rejects the removed headphone-name override
  before systemd, with exit 78 and no value disclosure. Generic privacy and
  injection fixtures now use a supported ASR setting. The launcher suite has
  23 passes and two declared skips; audio/daemon removal is integrated separately.

The control/native migration removes the raw-target entry point and admits a
candidate before gate/generation or replacement. Task7 orchestration moves to a
library-owned command; its public binary exposes only an ExitCode entry. These
shared-interface changes are still under integration. Preliminary review found
round-trip session mutation before a failed gate acquisition and missing
post-inspection deadline checks; those adapter corrections and retained Task7
graph cleanup are not yet complete. No focused result above certifies this tree.

Round-trip uses one event-driven application owner. Its lease survives worker
completion until confirmed cleanup and OS-thread join. Public Stop has an
independent ten-second wait bound; it does not claim that synchronous route or
startup operations have completed. Later Stop attaches to the same retained
attempt. Semantic failure remains Failed after successful cleanup. Main drains
this owner even when its public checkpoint is already terminal.

Route restoration and exact peer absence now have separate completion flags.
The route adapter retains its capability before a fallible move, including an
error before the caller receives that capability. Neither a successful absence
check nor a failed restoration can retire the adapter. Retry skips previously
completed audio, duplex, and absence phases. The duplicate capability field in
the outer resource container was removed. Scoped source review approved this
aggregate change; uncertain lower-level Pulse moves and journal rollback still
need their own correction and evidence.

Excluded evidence: the first PCM fixture changed executable identity via exec,
so its failure was not a valid cancellation oracle. The corrected fixture keeps
the same process identity and was rerun against the original implementation
before GREEN. An earlier owner-join GREEN-named log contained unrelated native
RED compilation errors and is excluded. The later 61-test library run actually
executes and passes the owner-join-panic regression. No whole-stage aggregate
pass is inferred from either the log names or these focused suites.

Before closing: finish integration, update exact test/publication manifests,
run the source aggregate, record a quantitative code-surface delta including
new modules, obtain independent exact-tree architect/security verdicts, refresh
the defect register, commit, and push the fork branch. B2 completion does not
authorize bypassing C-F or pausing for another conversational confirmation.

The frozen code-reduction target is not met. Counting production source before
the first test module, including the new control module, the API/state/native/
control surface initially grew from 3402 to 5377 lines (+1975). The later
source-audit checkpoint has 6050 lines in those four production prefixes plus
27 watchdog-getter lines, totaling 6077 against the corrective 5127-line ceiling.
The later deadline correction totals 6170 lines; serde consolidation reduces
that total to 6040, still 913 above the ceiling. These are intermediate counts:
further admission changes and genuine consolidation remain in progress.
The acoustic-admission checkpoint counted API521 + state806 + native3429 +
control1149 + acoustic373 = 6278 production-prefix lines (API has no inline
test module). Including the previously identified 27 watchdog-getter lines gives
6305, still 1178 above the ceiling. The count includes visibility-qualified test
modules and does not stop at test-only methods inside production impl blocks.
Voice rejection plus removal of the duplicate API error map changes that count to
API503 + state810 + native3429 + control1153 + acoustic374 =6269, plus27 getter
lines =6296:1169 above the ceiling. The policy-neutral core predicate adds6 lines
outside this daemon count; it is included in the voice change's net-3 total.
Moving helpers or Task7 serialization into another maintained file
does not count as reduction. New cleanup and recovery guarantees do not
automatically waive the target: a reviewed macro-level simplification or a
specific, justified architecture amendment is required before stage closure.

No model replacement, physical audio evaluation, production restart, main merge,
tag, or release publication has occurred in this stage. Existing Task 7 live
latency debt remains open; deterministic tests cannot close it.

The next admission migration uses one daemon acoustic policy, fresh read-only
device/graph/route facts and the command's original deadline. Its proposed
errors distinguish unavailable, busy and expired observation from known unsafe
devices; rejected running replacements retain the old owners and revision.
Task7 uses a synthetic null-sink monitor and a physical output, not an
all-virtual route. Its command must retain the lease and graph until native
cleanup, with no exported raw-target constructor or escaping runtime handle.
These are reviewed-design work in progress, not implemented admission proof.

Provider-host design also remains open: one dormant daemon-lifetime owner must
serve normal translation and RoundTrip, prepare only the selected provider after
Start, retain models across logical Stop, and join the host at final shutdown.
Cloud consent must become an in-memory generation-bound grant with explicit
revocation, not a provider-selection boolean. Source estimates do not show that
this change alone can meet the current code-size ceiling.
