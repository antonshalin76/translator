# Translator product closure: one integrated candidate

This is the 2026-09-27 starting plan, not the current implementation status.
The authoritative candidate gate record is
[`c2c_8a42/evidence-ledger.md`](c2c_8a42/evidence-ledger.md). Statements below
about code that was then missing or failing describe that baseline and must not
be read as new post-change findings.

Status: proposal for one C2C macro-plan and final exact-tree review, not a
release approval. Source checkpoint: clean `codex/product-clean-20260923` at
`faa8431eb999195c0a06b96508f5a3a14f2f318d`. Production is out of scope
until the release gate. Do not substitute the older C2C checkout at `6c81cc3`
for this candidate.

## Product target and current evidence

The target machine has a real microphone and speakers but no headphones. The
user permits audible calibration only after an isolated test is prepared and
in a window without calls. The release must therefore prove a safe open-speaker
bidirectional path; the generic contract's permission to ship without optional
AEC applies only to a headphones-capable product, not this installation.

Read-only hardware preflight on 2026-09-27 found the built-in analog sink on
`analog-output-speaker`, its headphones port unavailable, and the built-in
microphone on `analog-input-internal-mic`. A separate USB audio card exposes
both capture and playback PCMs; neither PCM had an owner at inspection time.
This does **not** establish that its playback is audible, that it stays free,
or that a private server can acquire it without changing production state.
No playback, capture, module load or route mutation was performed. The
physical harness must recheck those facts immediately before arming and abort
on contention; the only current candidate for hardware isolation is the
separate USB card, not the production analog pair.

| Boundary | Current evidence | Missing for product completion |
| --- | --- | --- |
| No-headphones audio | Acoustic admission rejects open-speaker mic without a proof for the exact physical pair. AEC coordinator/controller and Pulse graph components exist. | `main.rs` still constructs the watcher with unavailable AEC and passes no calibration controller to the API. `round_trip_preconditions` also hardcodes headphones, so even a validated AEC pair cannot run that test path. There is no live calibration engine, runtime proof attachment, revocation, or physical AEC result. |
| Audio lifecycle | Owned original-loopback creation and volume control passed deterministic and private virtual-Pulse checks. | On an observed unsafe output change, stale raw-mic cleanup can fail silently; one-sided owned streams can be overlooked. Start/reconcile can report success without route-safety proof or stopping a newly unsafe running mic. The one-second watcher cannot prove zero exposure at the instant of an external port change. |
| Effective model chain | The original `main` chain is Small → NLLB → Piper. Turbo is the input development leader; NLLB remains the default MT, and Hy is a quality challenger with unresolved failures. | The original chain has not been measured against both Turbo chains on an independent full-chain holdout. Freeze the effective binaries, configuration and model hashes for each arm before comparative claims. |
| Output voice | Piper is the configured product voice, not a demonstrated winner. A direct eight-cell CPU pilot compared it with Supertonic 3; the tested Supertonic path was slower to first PCM and its ASR proxy changed four numeric values. | No blind, loudness-controlled RU/EN audible decision, complete voice/gender matrix, or paired first-audible product result. Qwen3-TTS is an untested bilingual matrix candidate; Kokoro is English-only in this project's matrix. Research-only or license-blocked models are not release arms. |
| Quality and speed | Frozen 24-WAV accelerated saved-audio diagnostics include complete quality-first and balanced runs; one full streaming-first run failed. Historical physical Task 7 first-audible is 5968 ms. | No independent speaker-disjoint RU/EN holdout, full mode/voice matrix, independently transcribed audible TTS, physical first-audible comparison, or quantified reliability gain. Replays do not erase retained failures. |
| User and release path | Daemon/API/UI, installer and release contracts exist. | No native calibration UX or live status; no complete physical Meet/Telegram/Zoom matrix, 30-minute soak, install/rollback proof, or exact-SHA release evidence. |

Baseline configuration check on 2026-10-01: the unchanged `main` revision
`9291e8beafee3e02aaa179178ce460ac9e6c6de2` defaults to
`faster-whisper-small`, NLLB and Piper. Its user service reads `.env`, which
also selects Small for ASR; no MT-model override was found. The service unit
was inactive at inspection. This establishes the configured baseline, not a
live runtime or model-file identity; freeze those separately for the eval.
The fork's `translator_product_audio_pair.py` v2 runner now has a separate
Small → NLLB → Piper arm alongside Turbo → NLLB and Turbo → Hy. Its 70 focused
contract tests pass on 2026-10-01, including effective-model fallback,
event identity/order, and terminal failure recording. No three-arm model run
has been accepted yet. The Small arm uses the candidate code and is explicitly
labelled a same-code ablation; it is **not** a receipt from the untouched
original `main` baseline.

The existing 24-WAV Turbo/NLLB-versus-Hy screen is development evidence. Use
it to debug the three-chain runner, not to claim improvement over the original
Small baseline or to select a release model. Freeze a separate development set
and untouched release holdout before further model tuning. For each direction,
the holdout must meet EVAL-0's 120 unique source/reference pairs, speaker and
critical-case coverage, and verified audio/reference alignment.

These are connected seams of one feature, not an invitation to run separate
C2C cycles for each row. The source contracts are `master-bdd.md` (especially
acoustic admission, EVAL-0..3, APP-1..3, REL-1) and the saved-audio diagnostic
`product-audio-paired-20260927.md`.
The AEC validator requires measured raw/clean power windows, exact 30-second
frame accounting, a separate 60-second far-end-only observation and a recent
positive control that actually reached VAD/provider submission. Loading
`module-echo-cancel` or fabricating observation counters cannot publish a
valid product proof.
The existing PCM capture/playback adapter is fixed at 16 kHz in 20 ms frames;
the acoustic validator consumes aligned 48 kHz one-second windows. A real
calibration engine therefore needs format-aware acquisition and a defensible
raw/clean time alignment, not a conversion of already aggregated counters.
The runtime observer and graph custody components exist but are not connected
to a production calibration attempt. The device watcher also receives an
immutable `AecCapability::Unavailable` at construction today. These are one
end-to-end wiring problem, not independent evidence of a working AEC path.
The unsafe-device transition is a separate acceptance case within that path.
`ReconcileAudio` currently refreshes facts and mix without re-admitting an
active microphone. An original-loopback unload error is logged but not returned.
Normal `Bypass` sets original microphone volume to 100%, so calling it before
proven loopback removal can increase leakage. A one-sided owned stream is now
ignored by discovery; it must instead keep ownership uncertain without blindly
unloading a possibly foreign module. Polling can prove bounded reaction after a
transition is observed, not zero exposure at the physical event instant.

The transition's deterministic acceptance matrix uses the real control seam.
For `ReconcileAudio`, fake maintenance must write the new device observation
into `RuntimeStore`: that command does not call `RuntimeFactsSource::inspect`.
A counting runtime records mic PCM cessation, and a scripted Pulse runner
records unload and volume order. Patch validation against fresh facts is a
different operation and must retain its existing no-replacement-on-rejection
contract; it cannot substitute for observed runtime-transition tests.

| Scenario | Test seam and required observation |
| --- | --- |
| Start with microphone enabled and original mic set to 0% | A runner spy rejects any `start_supervised` call before observed effective 0% raw-mic quarantine and exact old-loopback custody; a successful volume command without observed mute, or an existing loopback missing from mix discovery, must not reach the runner. |
| Start fails after quarantine | No blind `Bypass` or double cleanup, no new translation/audio frames; uncertain runtime or loopback custody remains CleanupPending. Headset passthrough at 100% returns only after fresh Headphones admission plus verified loopback and route state; otherwise quarantine and visible failure remain. |
| Running mic, headphones become open-speaker or unknown | Fake maintenance writes the observation before `ControlOwner::execute(ReconcileAudio)` returns; mic PCM ceases even when the command returns `Err`, and status is not Running. |
| Device discovery fails | The same control test cannot reuse prior safe facts to keep mic PCM active. |
| Owned raw-mic unload succeeds | Fake Pulse runner confirms the owned module is absent before any `Bypass` command can set original mic volume to 100%. |
| Owned raw-mic unload fails | `PulseResources` through `RuntimeMaintenance` reports failure; no new loopback or 100% original-mic volume command occurs; Start remains blocked until verified cleanup. |
| Only one side of an owned loopback is visible | Discovery leaves custody uncertain; Start is blocked and a module is not unloaded solely on one marker. |
| Runtime stop fails | Control projects CleanupPending, not Stopped, and rejects another Start until cleanup is verified. |
| Safe headphones return | No automatic restart; explicit Start performs fresh admission. |
| Speaker-only Start | Speaker runtime starts; fake Pulse call log contains no mic quarantine, mute or loopback commands, and missing microphone admission does not stop the speaker direction. |
| Refresh, Stop and late generation race | Serialized control rejects stale success; a late event from the old generation cannot restore Running or mic PCM. |

An error response or a changed status alone is not a passing test: mic PCM must
cease on unsafe observation regardless of response, with the Pulse command-order
assertions above. Physical acoustic validation remains a separate release gate.

Responsibility for this transition is singular at each layer. `ControlOwner`
owns the pre-Start microphone quarantine and verified old-loopback custody
before any mic runtime PCM, then decides when a running microphone becomes
unsafe, blocks new Start, and orders quarantine, runtime stop,
owned-loopback verification and only then normal bypass. This sequence applies
to Start, Reconcile, Stop, Recover and terminal cleanup. The pre-Start mute
must be observed and old-loopback custody verified before `start_supervised`
can open PCM. A failed Start cannot blindly restore 100% raw mic: it remains
quarantined/CleanupPending while either runtime or loopback custody is unknown.
Headset passthrough may return only after fresh Headphones admission and
verified loopback/route state. A speaker-only Start makes no mic-quarantine
commands and is not blocked solely by microphone quarantine. Successful
runtime stop alone does not prove that raw microphone audio is gone.
`AudioMixApplication` mechanically maps a requested quarantine mode to 0%
original and 0% translated microphone volume; it does not decide when to use
that mode. `PulseOriginalLoopbacks` owns module discovery and removal;
`PulseResources`/`RuntimeMaintenance` return typed cleanup custody instead of
only logging it. A one-sided marker is uncertain custody, not absence and not
permission to unload a foreign module. `RuntimeStore` projects status and
errors; API, SSE and UI only display them. No separate broker/proxy decision
or new persistent database is proposed; surviving Pulse ownership must be
rediscovered on restart.

`CleanupPending` covers both a live/uncertain runtime and uncertain raw-loopback
custody. Neither Stopped nor another Start is allowed until teardown and
quarantine/route state have been verified. If both muting and unload fail,
software cannot claim zero leakage: retain Unsafe/CleanupPending, prohibit
Start, and fail the physical release gate rather than reporting safety.

On 2026-09-27, the focused RED test
`microphone_start_applies_non_bypass_mix_before_opening_runtime` failed on
current `ControlOwner::execute(Start)` (1 failed, 250 filtered). It proves the
existing call order is wrong, not that moving a mix call alone is safe. An
independent test-design critic requested additional negative tests for an
owned loopback absent from mix discovery, a successful volume command without
observed mute, and failed-Start rollback before implementation. This RED is
not product evidence and must not be made GREEN by simply swapping two calls.
The second RED now covers asymmetric owned Pulse markers at both the parser
and `PulseOriginalLoopbacks::ensure` mutation boundary. Its focused bin test
failed (1 failed, 30 filtered); the independent test-design critic accepted
this RED because the old code discards uncertain custody and may continue to
load a module. Neither of those REDs proves that the complete unsafe-transition
path is fixed.
Two further focused RED cases use the existing `AudioMixApplication` Pulse
runner: a successful mute command followed by a 50% observed original-mic
volume must leave the mix uncommitted and unknown; an observed 0% counterpart
must be accepted only after a post-command readback. Both fail on the current
implementation because it never rereads Pulse after setting volume. The
independent test-design critic accepted the corrected negative case; the
positive case was strengthened to require readback. These tests do not replace
the separate Start/runtime or loopback-custody gates.

The installed production unit reads its own environment file; a value-only
read-only check on 2026-09-27 classified its ASR override as `small`, not
Turbo. The unit was inactive at that instant, which does not authorize
changing its configuration or treating production as unused.
Selecting Turbo in the current sidecar still requires the `small` model in
`required_model_ids`, installs it as an ASR fallback, and selects it instead
of Turbo when the ASR device is CPU. This is a separate operational cell: it
must meet the product quality floor or be removed/declared unavailable by an
explicit candidate decision. A CUDA Turbo benchmark cannot validate it.

The previously used FLEURS RU/EN material is not a release holdout. Candidate
new sources need separate eligibility and license checks: the [CoVoST 2
inventory](https://github.com/facebookresearch/covost) includes RU-to-EN but
not EN-to-RU; [MuST-C](https://www.sciencedirect.com/science/article/pii/S0885230820300887) covers English speech to
Russian under CC BY-NC-ND. Neither source alone satisfies the full two-way
critical-case/acoustic matrix. Do not import or score either before freezing
the exact source, split, speaker and reference identities.

A second possible source join is now identified, but **not admitted**:
[MASSIVE](https://github.com/alexa/massive) documents a shared utterance ID
mapping back to SLURP and professional EN-to-RU localization;
[Speech-MASSIVE](https://github.com/hlt-mt/speech-massive) provides RU test
recordings with speaker identities in a [separate test repository](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE-test),
while [SLURP](https://github.com/pswietojanski/slurp)
provides the original EN recordings and speaker metadata. The Speech-MASSIVE
RU test split lists 2,974 recordings and 51 speakers; SLURP says its complete
acoustic download needs about 6 GB. Those publisher facts do not establish that a
specific 120-pair intersection has correct audio/reference alignment or covers
the long/discourse and accent buckets. Speech-MASSIVE and SLURP audio carry
non-commercial license conditions; CoVoST's original CC0 release and the
[current Mozilla portal's terms](https://mozilladatacollective.com/datasets/cmp787jb102whmp07hbogpvaw)
also need artifact-specific reconciliation.
The official Speech-MASSIVE test repository currently reports gated access;
the separate ungated development repository has `validation`, not `test`, for
RU. Do not silently replace the intended holdout with that development split
or use existing credentials to bypass unaccepted dataset terms. No material
from these sources has been downloaded, frozen or scored here.

An official [Tatoeba export](https://tatoeba.org/en/downloads) provides
sentence-translation links and per-record audio author/license metadata. It is
the next metadata-only feasibility check, not an admitted holdout: filter audio
to explicit reusable licenses, count RU/EN pairs and disjoint speakers, then
inspect critical/long-turn coverage and audio-reference alignment. Tatoeba
itself warns that sentence translations may need correction. Do not download
audio or score models until the metadata, terms and selection are frozen.

## Proposed single development pass

1. Freeze the original Small → NLLB → Piper baseline, both candidate chains
   (Turbo → NLLB → Piper and Turbo → Hy → Piper), a development set, and a
   disjoint release holdout before tuning. Bind each arm to its own exact source
   revision, runtime configuration, binary and model/voice hashes. Extend the
   saved-audio runner with an explicitly labelled Small arm and negative tests
   for wrong model, mismatched input, incomplete attempt and baseline/candidate
   identity confusion. A Small arm executed on candidate code is a same-code
   ablation, not the untouched original baseline.
2. Use the development set to compare all three chains with complete failure
   accounting. Evaluate ASR text, MT meaning and generated PCM separately;
   compare NLLB and Hy on identical frozen ASR text as an MT-only control.
   Keep Hy opt-in until critical errors and the unexplained failed attempt are
   resolved. In parallel, compare Piper with eligible TTS challengers on the
   same verified target texts in RU/EN male/female cells. Admit a challenger
   only after source/license, first-PCM, stability, critical-content and
   independently transcribed audible checks; ASR-proxy agreement alone cannot
   establish pronunciation. An unavailable English perceptual check is marked
   `UNAVAILABLE`, not silently counted as a pass. Freeze the chosen complete
   chain before the release holdout is opened.
3. Implement the complete open-speaker vertical path together: isolated
   calibration engine, exact-pair AEC proof and runtime revocation, safe
   original-mic transition with verified cleanup/failure projection,
   Start/Stop/restart admission, API status and native UI controls. Test the
   assembled behavior on a private virtual Pulse server and with fault
   injection. An error must never be presented as a safe operating state.
   Incoming-speaker continuity and outgoing-microphone privacy are separate:
   retaining the original microphone until first translated audio would leak
   source speech when its configured original volume is zero.
   Keep graph ownership, measurement, proof authority, admission and UI
   projection in their existing layers. The physical proof must be issued only
   after distinct baseline windows, aligned raw/clean measurements, the real
   VAD/provider positive control and uninterrupted far-end-only observation.
   Changed device, port, volume, Pulse server, graph generation or provider
   invalidates it before another PCM start. Cancel, timeout or unload failure
   must leave visible cleanup custody and block microphone admission until
   verified teardown. A private-Pulse test checks behavior and fault handling,
   but cannot itself certify the physical acoustic threshold.
   The round-trip self-test currently *intentionally* requires headphones in
   both code and `master-bdd.md`. Enabling it on speakers requires an explicit
   AEC-reserved branch with no-recursion and restoration assertions; simply
   removing that precondition would weaken the product contract.
4. After the isolated test and rollback are rehearsed, perform one scheduled
   physical acoustic gate on the user's pair without taking over production
   audio: 30 seconds of -20 dBFS far-end fixture with median ERLE >=15 dB;
   separately 60 seconds far-end-only with zero outgoing VAD/translation;
   then real bidirectional Start/Stop/restart, audible first-frame, route
   restoration and leak checks. Retain every timeout and failed attempt.
5. On the frozen baseline and candidate SHAs, run the documented independent
   RU/EN holdout, all three modes, both language assignments, both required
   target-voice genders and all advertised fallbacks. Compare the original
   Small → NLLB → Piper chain with the frozen candidate chain on identical
   source audio and target references, using counterbalanced arm order and
   matched voice assignments where possible. Retain separate Piper-versus-TTS
   challenger results rather than attributing a voice change to ASR or MT.
   Report ASR, MT and independently transcribed audible-TTS accuracy by layer,
   along with critical errors, drops, queues and resources. Measure
   graph-boundary first-audible on a valid physical path where each arm can
   actually start. Mark an inadmissible baseline path `UNAVAILABLE` rather
   than imputing latency from
   saved-WAV timings or historical Task 7. Then run the required real-app
   calls, native UI, 30-minute soak, installation and rollback gates.
6. One terminal architecture/security/C2C review of the exact tree and full
   evidence packet decides release readiness. Only a passed candidate is
   merged, packaged, deployed and published with rollback proof. Failed gates
   are fixed at their owning layer and the affected integrated evidence is
   rerun; a green narrow test cannot replace the missing product gate.

The documented absolute floors remain binding: zero critical
negation/number/name/role corruption, drops below 1% in every required cell,
mode first-audio limits of 3000/2000/1000 ms, and no worse-than-allowed
paired candidate regression. The independent holdout requires at least 120
distinct source/reference pairs for each language direction plus its speaker,
bucket and acoustic-fixture coverage. Unknown or unavailable is not PASS.

## Resource and stop rules

Use the fork only; keep production daemon, routes, and model cache unchanged.
Build with bounded parallelism and run GPU arms serially, retaining exact
failure receipts. Do not run physical playback/capture before isolation and a
call-free window. Do not fabricate headphones, bypass acoustic admission,
weaken quality thresholds, silently switch models, or classify a private
virtual-Pulse result as physical E2E. If the user's physical AEC gate fails,
the no-headphones product remains unreleasable; a generic headphones-only
release does not satisfy this request. If the independent holdout, all-mode
matrix or native app proof is missing, merge and release remain blocked.
