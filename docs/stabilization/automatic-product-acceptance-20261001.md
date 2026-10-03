# Automatic product acceptance

The user's 2026-10-01 instruction removes manual listening and manual review
handoffs from execution. Production remains unchanged until measured product
gates pass. Automated speech recognition and semantic judgments do not certify
subjective voice naturalness or turn an ambiguous reference into acoustic truth.

## Integrated execution order

1. Capture the actual complete output PCM from unchanged main
   Small -> NLLB -> Piper and all three existing fork arms on the same frozen
   24-case development screen. Compare MT and independently transcribed output
   separately. Use both OpenAI and Google ASR and independent Claude/OpenAI
   semantic judgments; calibrate each evaluator with positive/negative controls.
   Retain uncertainty, errors, drops and every unrun case. No default promotion
   or release decision from this development set alone.
2. Finish the open-speaker path using one retained synchronized physical
   acquisition/AEC graph, not independent resampled `parec` streams. Recheck
   exclusive hardware custody before arming an isolated nondefault physical
   pair. Prove the real 30-second/60-second acoustic gates, revocation,
   Start/Stop/restart and teardown before admitting that graph into runtime.
3. Freeze an independent licensed holdout meeting EVAL-0, then run the full
   voice/mode/direction/fallback matrix, paired baseline comparison, native
   packaged-app controls, authorized real call endpoints and 30-minute soak.
   Missing endpoint credentials or speaker metadata remain explicit unavailable
   prerequisites; virtual fixtures do not replace these gates.
4. Build from the frozen passing candidate, test installation and rollback,
   publish the fork commit and reproducible artifacts, and only then merge,
   activate and release. No human listening gate is scheduled.

## This stage's acceptance and ownership

| Behavior | Owner | Narrow verification |
| --- | --- | --- |
| Critical facts in MT and audible transcript checked independently; missing transcript counted as a drop | `benchmark/task6.py` | 20 new regression cases, complete 88-test suite |
| Capture disabled leaves result/files unchanged; only verified terminal, health and drained session may persist PCM | Existing original/fork runners | Provider event/lifecycle spies plus real WAV readback |
| Private exclusive durable WAV; exact format/hash/count and directory/file identity at send time | `translator_product_pcm.py` | Filesystem faults, cancellation, symlinks, replacement, collisions and readback |
| Whole-run/arm cleanup and exact input bindings before network; custody/remote-cleanup uncertainty stops sends | `translator_cloud_chain_audio.py` | Paired matrix, counted transports and post-preflight mutation |
| Both ASR providers and both semantic judges required; ambiguity or audible-only corruption cannot pass | Cloud diagnostic runner | Distinct eight-output bidirectional fixtures and judgment controls |
| Numeric spelling normalized without erasing amount, unit, sign, time, negation or duplicate identifiers | NeMo/Jiwer scoring boundary | Six deterministic wrapper tests and explicit real RU/EN grammar integration |

Pydantic owns response validation and JSON schema; standard-library WAV handling
owns the container. Existing provider adapters own HTTPS and remote-file cleanup.
The diagnostic owns its private attempt ledger, not product admission, routing,
UI state or release policy. Those product responsibilities are unchanged.

Independent BDD critic/auditor and pre-RED SRP review passed. Task6's initial
15 failures and four missing-transcript regressions became green. PCM capture
had 20 failures and 37 missing-helper errors before implementation. Cloud tests
also reproduced two substantive provenance/custody errors; both became green.
Final independent architecture/source review passed for the bounded diagnostic.
No additional behavior-preserving refactor was required after these fixes.

## Validation ledger

All runs bind exact source/model/input hashes. New capture/evaluation evidence
is necessary because the older receipts contain only output PCM hashes, not
the output WAVs. They cannot be reused for independently transcribing output.

| Gate | Fingerprint / environment | State |
| --- | --- | --- |
| Task6 regression | Two owned source/test files; sidecar environment | 88 PASS |
| Fork runner and PCM custody | Final worker files; sidecar environment | 151 PASS |
| Original runner | Original-sidecar-first isolated process | 26 PASS |
| Cloud orchestration | Final expected-input/custody fixes | 16 PASS |
| Real numeric grammar | NeMo 1.2.0, Pynini 2.1.6.post1, Jiwer 4.0.0, isolated Python 3.12 | 6 wrapper tests and 1 real RU/EN integration PASS |
| Frozen full-chain capture | Original HEAD `9291e8b`; fork source plus helper hashes; 24 frozen inputs; pinned eval cache | Original 24/24, fork 72/72; 96 verified WAVs; both terminal cleanups complete |
| Independent output diagnostic | New capture artifacts, recorded ASR model IDs and dated OpenAI judge, two ASR/two semantic judges | 24/24 COMPLETED; controls PASS; diagnostic acceptance FAIL |
| Aggregate source gate | Commit `6c80639e9bf7263a6fb3eff2f2c2c1b5235daaaa`, tree `b4c5747a85f2e72ba4f5d940848e544974658b71` | All 18 deterministic gates PASS; pytest 1153 PASS / 5 external skip, unittest 335 PASS / 22 external skip; publication `release=false`; commit pushed and remote SHA verified |

The artifact root is private and outside Git: `automatic-chain-capture-20261001-uYFSDG`
under the existing product-evidence directory. Outputs remain `release=false`.

The original capture receipt SHA-256 is
`0872b80b973da472d50d58f06ba763f25bc6a69195dd31a93864c4c217083cd5`;
the three-arm candidate receipt SHA-256 is
`027cb8b96b162c12ebf6cafffec20aa4201e9b494b3c35070e4f7d99b163c70d`.
Capture used private offline caches, no audio device, a 9 GB memory ceiling,
1 GB swap ceiling and two CPU cores. Cloud evaluation has a 5 GB memory
ceiling, 512 MB swap ceiling and two-core CPU quota. The evaluator environment
is separate from both the production sidecar and Qanola. Credentials are read
only by the existing provider loader and never copied into receipts or Git.

The generated provider PCM is not physical playback or first-audible evidence.
The 24 inputs are a development screen, not the independent release holdout.

## Actual output result

The evaluator receipt SHA-256 is
`916ed05d7f2e41305dc040d18da106d7bdf79228164c14ee3377768831bf5345`.
All 192 output-ASR and 48 case-judge observations completed; both silence and
both semantic-corruption controls passed. No failed attempt was retried out
of the receipt. The successful command exit means measurement completed,
not that a product-quality gate passed: `diagnostic_accepted=false` and
`release=false`.

| Chain | Both judges accept source-reference to MT text | Both judges accept both output-ASR full chains | Both judges accept both output-ASR TTS layers | First provider PCM median / p90, ms |
| --- | --- | --- | --- | --- |
| Untouched main Small/NLLB/Piper | 5/24 | 5/24 | 18/24 | 7853 / 11500 |
| Fork Small/NLLB/Piper ablation | 8/24 | 8/24 | 20/24 | 689 / 1817 |
| Fork Turbo/NLLB/Piper | 9/24 | 8/24 | 18/24 | 685 / 1057 |
| Fork Turbo/Hy/Piper | 10/24 | 8/24 | 19/24 | 900 / 1193 |

These are strict development judgments against written references, not
population accuracy or independently verified acoustic truth. Disagreement
or uncertainty is not counted as unanimous PASS. All 24 capture attempts
completed in each arm under their development timeout. The serial arm order,
different runtime implementations and absence of physical playback limit
timing attribution; the fast fork-Small ablation precludes crediting the entire
timing difference to Turbo. No EVAL-1 or EVAL-2 comparison is claimed.

The first judgment column includes ASR losses: it compares the written source
reference to the final translation, not actual ASR text to translation. An
independent attribution check found unchanged ASR wording in 22 of 24
original/fork-Small pairs; all three additional unanimous textual passes were
in that unchanged-ASR subset. Some downstream improvement is therefore
observed, but the entire first-column deficit cannot be attributed to MT.
The next private MT comparison must use identical actual-ASR input strings.

Normalized TTS proxy WER across the two independent recognizers was:

| Chain | RU input to EN voice | EN input to RU voice |
| --- | --- | --- |
| Untouched main | 1.56–3.52% | 2.59–5.17% |
| Fork Small | 1.60–4.17% | 3.12–3.91% |
| Turbo/NLLB | 0.97–1.61% | 2.40–3.20% |
| Turbo/Hy | 4.84–6.13% | 2.26–4.51% |

Low aggregate WER does not erase critical errors. Both ASRs and judges
flagged a foreign-name rendering in the Hy output and a joined-word
corruption in the NLLB output. Other mismatches may originate in the output
recognizer rather than Piper. The dominant measured problem is upstream
meaning preservation, not general voice intelligibility. Hy has no demonstrated
full-chain advantage over Turbo/NLLB on this screen. Keep it as an opt-in
challenger; no default promotion or Piper-winner claim follows from this run.

## Reproducible validation prerequisites

The first aggregate stopped at seven publication tests because Gitleaks was
not in the launch environment. A focused 59-test run reproduced the same
seven missing-tool failures; the same suite passed after configuring the
existing pinned Gitleaks executable. No source, assertion or scanner policy
was changed. The full gate must supply the existing pinned tool paths through
`TRANSLATOR_GITLEAKS_BIN`, `TRANSLATOR_CARGO_AUDIT`,
`TRANSLATOR_OSV_SCANNER` and `TRANSLATOR_BUN`. Expected versions are
Gitleaks 8.30.0, cargo-audit 0.22.2, OSV-Scanner 2.5.0 and Bun 1.3.12;
their executable hashes are checked by the existing gates.

With those tools supplied, the complete aggregate passed the Rust/UI/Python,
shell/systemd, SCA and schema gates, then failed at publication. The direct
publication check identified an executable-policy mismatch for the evaluator
requirements file and an incomplete tracked-file inventory. The unchanged
requirements moved to `tests/requirements-automatic-audio-eval.txt`, and the
sorted inventory now matches the exact index. Candidate publication and
test-manifest checks both pass; scanner and publication policy are unchanged.
A later aggregate passed those same test/SCA/schema gates but rejected the
parent's concurrent documentation/index update as a source-freeze violation.
Its exit remains failure. The final aggregate ran with all tracked files
and the index frozen through its terminal result, within 6 GB RAM, 512 MB swap
and two CPU cores. Terminal evidence was written outside Git so recording it
cannot itself change the measured candidate. Earlier failures are retained,
not rewritten as successful runs.

Fresh read-only audio inventory found USB selected as the system default and
no source-output client; the nondefault built-in speaker/internal-microphone
pair and both analog PCM handles were idle. This is a possible isolation path,
not a physical AEC measurement or permission to change the default graph.

Current public/local corpus research has not established EVAL-0 speaker,
gender, accent and per-bucket coverage. Tatoeba offers human bilingual text and
audio authors/license metadata but does not establish all required attributes.
Current development sources cannot be relabelled as an independent holdout.
Four authenticated MDC lease probes returned access for the existing SPS5 RU
and EN datasets, but HTTP 403 for Scripted 27 RU and EN
(`terms_or_access_not_approved`). No new terms were accepted and no audio was
downloaded. After all prior source/speaker exclusions, the accepted RU
archive yields only four quality-eligible fresh recordings from two speakers,
without the required gender/accent coverage. Accepted EN untouched train has
876 eligible recordings from 169 speakers, not a publisher test split.
EVAL-0 does not require human-authored target translations: independent frozen
external references may retain their provenance and uncertainty, but must
not be labelled human gold. Missing RU coverage remains the blocker.

## Controlled MT comparison

The private `mt-fidelity-private-20261001` experiment used the same 24 actual
ASR strings for NLLB, the unchanged Hy prompt and a generic fidelity-prefix
challenger. The prefix was frozen before loading the inputs. Both Hy profiles
were run again in counterbalanced order with the existing verified model and
runtime; all 48 translations completed and the model process/lease/socket
cleanup was confirmed. Capture SHA-256:
`16ec57ab828ec480329dc5f6f994bf1784ed1cf28e332e1e200898005ac82c10`.

Claude `claude-sonnet-5-5` and OpenAI `gpt-5.4-2026-03-05` judged anonymously
rotated A/B/C translations against actual ASR, not written source references
or audio. Each passed positive, corrupted-negation and corrupted-amount
controls in both directions. All 48 case judgments completed without retries,
and provider cleanup completed. Judge receipt SHA-256:
`428d033e437cad26f17d111dc75e5b743441aef2429d822245f9c06af21e94ad`.

| MT arm | Both PASS | Both FAIL | Both UNCERTAIN | Judge disagreement |
| --- | ---: | ---: | ---: | ---: |
| NLLB anchor | 5 | 6 | 0 | 13 |
| Hy unchanged prompt | 16 | 1 | 1 | 6 |
| Hy fidelity-prefix challenger | 16 | 1 | 0 | 7 |

Hy's unchanged prompt has four unanimous wins over NLLB, no unanimous losses,
four definite ties and 16 unresolved comparisons. The prefix has no
unanimous win or loss against the unchanged Hy prompt; the strict three-arm
receipt counts seven ties and 17 unresolved cases. Do not adopt the prefix.
This isolates an MT advantage on the development screen; it does not erase
upstream ASR errors, establish a full-chain advantage or admit a new default.
No product prompt, model or voice configuration changed.

## Independent speech inventory

The [author-published CommonPhone dataset](https://huggingface.co/datasets/pklumpp/CommonPhoneDataset)
has a public CC0-1.0 metadata surface. A bounded metadata-only inventory of
Russian test rows 8383 through 10208 found 1,826 recordings, 1,825 normalized
unique texts and 36 speakers: 18 male and 18 female. The inventoried rows all
reported revision `3635a95a34c4d20337f30499e0d9ba0be30fdae3`. The earlier
prefix was not scanned; this does not establish the complete Russian split.
No audio, model, credential or new access terms were involved.

Every inventoried accent is unknown. Heuristic text buckets also supply no
long-turn cases and only three multiple-number cases; these are screening
proxies, not independent annotations. This is a useful possible natural-speech
development diagnostic, not an EVAL-0 release holdout. Speaker ID namespaces
differ from prior SPS5/AppTek sources, so literal ID mismatch cannot prove
human-speaker disjointness. Source, normalized-text and recording-hash
exclusions must retain that uncertainty rather than claim pretraining-unseen
or acoustic-gold inputs. Inventory receipt SHA-256:
`6daf93566370987b9e40f96ca42d5b2dd5d7bc9d5fa83b6ec24c952e8ba806b6`.

The existing coverage classifier incorrectly counted nonempty `unknown`/`null`
accent markers as valid nondefault accents. A public classifier reproduction
returned `COVERAGE_COMPLETE` for a complete synthetic matrix after replacing
every accent with `unknown`. Correcting that authoritative metadata boundary
does not create missing corpus annotations or make metadata coverage a
product-quality PASS. Its new negative tests are part of the next combined
source gate, not the previously published tree's validation.

## Physical acquisition result

The private `alsa-capability-20261001-6eJSZZ` probe has passed its 13 focused
tests, independent architecture review and exact-node namespace check. The
root-operated 1.5-second digital-silence trial acquired 72,000 real 48-kHz
frames with complete 480-frame transfers, observed DAC consumption and no
reported XRUN. Scope cleanup and device identities were verified; host USB
defaults, clients and mixer hash were unchanged. Receipt SHA-256:
`6660b2d8ffee81ec316441a3368a110a462ae2afd5b47837fb07624052f8e7f2`.

The ALSA status timestamp remained unknown. It was not replaced with a host
timestamp or treated as a requirement of the relative ADC sample contract.
The raw capture peak was 32,768, establishing at least one full-scale sample;
its count, channel and cause were not measured. Acquisition is therefore
available, but usable scored input, raw/clean evidence, AEC, physical onset
and product readiness remain unproved. The integrated physical owner must
report settling separately and reject clipping during scored acquisition.
It must not conceal clipping with software gain or modify shared mixers.

## Automatic native UI regression

The actual asset-backed debug Tauri application was exercised with the
maintained Tauri/WebKit WebDriver transport, Selenium 4.50.0, private Xvfb,
private DBus/XDG directories and an unavailable isolated daemon endpoint.
This is not a browser imitation, packaged release or online APP-3 result.

The first usable native regression observed real Tab navigation from Status
to Routes, then focus moving to BODY when the next status/calibration poll
replaced that control. Its failed receipt is retained with SHA-256
`2862e1fb5972a06607c2af43f389a920e40240c4c95123e4c211d953e02beea4`.
The UI render now restores an enabled counterpart using its stable semantic
control ID, without copying stale values or changing daemon authority.
Removed or disabled counterparts cannot inherit focus by DOM position.

The portable `scripts/translator-native-ui-check` passed all five checks:
native asset document, disconnected disabled controls, three actual window
sizes, keyboard focus across real polling, and identity/disabled/removed/no-
focus negatives. All 25 controls have unique IDs. The final source's receipt
SHA-256 is
`f9aa54f99f0971f93e2117e622b973bd34b9106d0c018b6bb68c32e09862a8bf`;
scope cleanup completed and all three reserved ports were released.
The application binary SHA-256 is
`7ab266db7093dd3d5567af0343cae6e038a512237fcb283eebc9fb071b0e6fdc`.
The test used a 2 GB memory ceiling, zero swap allowance, two CPUs and a
90-second outer deadline. The typecheck/Vite build and Ruff checks passed.
The combined source aggregate is still pending for this new candidate.

BDD critic/auditor, RED critic and pre/final SRP review used distinct self-
passes because the four available agent slots owned the integrated AEC
implementation. They do not constitute independent source approval. The
final combined candidate still requires the independent architecture audit.
The only refactor after GREEN made the existing native test portable through
argparse; the final portable command then passed the same five checks.
UI state remains AppState-owned, runtime decisions remain daemon-owned, and
test automation owns only isolation/evidence. No human check is scheduled.

Local and hosted native-build prerequisites now include the standard ALSA,
PipeWire/SPA and SpeexDSP development packages. A simulated then actual local
installation added only `libspeexdsp-dev=1.2.1-1ubuntu3`, with zero upgrades or
removals; the existing SpeexDSP and WebRTC AEC runtime hashes were unchanged.
The installed resampler header matches the previously extracted signed
package. No private include path or warm compiler cache is needed to pass the
sanitized aggregate environment. Production services/models remain unchanged.

## Retained native integration

The fork now contains the complete debug-path composition for the actual
built-in PCH speaker/internal-microphone pair. It is opt-in: both a verified
positive PCM fixture and an explicit read-only physical-facts server are
required. Ordinary startup without that configuration retains the previous
unavailable AEC path. The private Pulse graph owns virtual call endpoints;
only the native acquisition owner opens the actual physical pair. Neither
private virtual devices nor the system's USB defaults stand in for PCH facts.

| Responsibility | Authoritative owner | Verification boundary |
| --- | --- | --- |
| Real raw/clean acquisition, ADC continuity, original-channel validity and consumed-DAC reference | ALSA, installed SPA/WebRTC AEC, SpeexDSP and sealed native source | Synthetic protocol/fault tests; physical result still required |
| Process/scope, descriptor and graph cleanup custody | Existing backend session guardian | Cancellation, crash, late/uncertain cleanup and custody-race tests |
| Positive provider/VAD completion followed by 60 seconds of real clean capture | Existing runtime observer and the same warmed sidecar | Exact native sample spans; injected positive is not physical near-end gold |
| Proof publication, freshness, revocation and one-shot Start confirmation | Existing calibration coordinator/admission | No new proof state machine; every production PCM requires confirmed fresh authority |
| Calibration-to-Start-to-Stop-to-restart retained handoff | Existing calibration controller and audio-operation gate | Same graph/provider custody; no idle lease gap or deadline extension |
| UI pending cancellation and retained-attempt release intent | AppState and its existing projector | Used reconciliation helper and all succeeded proof states; daemon alone decides cleanup |

Model warm-up precedes scored measurement; it cannot overflow the four-window
measurement channel. Known positive speech must drain through the real provider
and output FIFO within the unchanged calibration deadline. Stop returns the
same native graph to retained calibration custody rather than reopening an
already-held ALSA device through Pulse. Failed or abandoned transfers remain
owned and fail closed. The five-minute proof lifetime and four-second HTTP
Start deadline are unchanged.

Independent review found two cross-owner integration defects before the live
probe: absolute ADC coordinates were passed to a fixture-relative evaluator,
and the UI hid cancellation after successful retained calibration. The latter
had two actual failing tests before correction; all 21 UI tests and the
typecheck/build then passed. AppState polling now preserves a matching pending
release until a terminal response, and the existing cancel command remains
available even when the retained attempt's proof is no longer valid. It does
not grant permission to Start or claim that an expired proof is validated.

The daemon's focused final all-target run passed 649 tests with nine existing
external ignores; strict Clippy and formatting passed. These are source
results, not the full combined aggregate, a physical calibration or release
approval. All heavyweight verification is serialized and resource-capped.
The next exact combined aggregate must regenerate and bind the test manifest,
including the new native UI lint surface, before measuring physical behavior.

The native coordinate correction was independently reviewed against frozen
source `9c502ea412d13ebecf324dae5be2dfdbe683467ea46732324705858ec86990f2`.
Its actual RED contained both the valid nonzero-start rejection and unchecked
absolute-end overflow. The final complete audio suite passed 257 tests, with
three existing external hardware ignores; strict audio/backend Clippy passed.
Only copied evaluator windows become fixture-relative after the original
sealed acquisition checks. Absolute source ranges and observation are unchanged.

After the retained-cancellation UI change, a fresh asset-backed native run
again passed all five disconnected checks and independently released its
process scope and ports. The new receipt SHA-256 is
`6490fc96e64323cc2493d5e1e6437c7975917e5029a0fd599be29efc741ec61f`,
bound to UI binary
`ff3b26381e1e95961e72603d9295c40a08f7f90d2c90e658e9aba2a675cddba8`.
The older failed and successful receipts are retained, not overwritten.
Independent BDD critic/auditor, actual RED critic and final SRP review passed
for the retained-cancellation correction. No further refactor was needed.

The public calibration API exposes lifecycle/proof state, actual pair names
and expiry, not numerical ERLE windows or a raw recording. A live receipt must
retain that measurement-detail limitation. The native launcher still resolves
its build-time source path and debug binary; relocated installation and trusted
runtime artifact discovery remain separate unpassed packaging gates. Do not
publish a release or interpret disconnected WebDriver success as online APP-3.

Current automatic C2C access is unavailable in this execution runtime: local
bridge checks are healthy, but the required browser/correct workspace tools
are not exposed. The existing C2C task and limit are preserved. Independent
local architecture review and released execution evidence are not substituted
for a claimed ChatGPT review. No manual handoff or listening step is scheduled.

The first combined native aggregate stopped before tests because its explicit
PATH omitted the existing `uv` executable. With that launch prerequisite
corrected, the next aggregate reached Rust tests and retained one cancellation
test failure: 294 passed, one failed and two existing hardware ignores. The
unchanged exact test subsequently passed alone; production terminal cleanup
still rechecks cancellation after joining the retained owner.

Two additional regressions demonstrated the test observation defect: the old
10,000-yield helper rejected legitimately delayed cleanup after 50 ms and
failed its declared elapsed-time allowance for an unreached predicate. Only
the shared test helper now uses the module's existing five-second Tokio
timeout and two-millisecond polling. Production deadlines, cancellation,
proof, cleanup and gate behavior are unchanged. All 33 controller tests pass,
including exact-attempt terminal, retained ownership and bounded negative
observations. Both failed aggregate outputs and both RED results are retained;
the next source result must bind the new candidate, not relabel those failures.

The following aggregate passed all Rust suites, including the corrected
controller, but stopped at formatting of `test_aec_backend_check.py` before
running Python tests. Ruff corrected only formatting; its 25 unchanged tests
pass, and the exact aggregate Python lint/format surface passes for all 100
files. The three-attempt checklist run is retained as blocked. The changed
source/manifest starts a linked corrected-candidate run, not a reset of the
previous failures, the existing C2C task or its iteration budget.

The next aggregate retained a distinct API test failure: the test expected
one calibration call immediately after scheduling and cancellation, although
`202` does not promise engine entry. A test-only notification now establishes
the post-entry scenario before cancellation. Its 100-ms API responsiveness,
duplicate-start conflict, terminal and call-count assertions are unchanged;
the separate initial-inspection cancellation case still requires zero calls.
All 45 API tests pass. The last shared measuring-state observation helper
also replaces its scheduler-yield counter with the same bounded Tokio timer;
its paused-time negative first failed on the original implementation, then
all 34 controller tests passed. No production runtime or policy changed in
these observation corrections. Every failed aggregate remains a failed run.

## Frozen native source result, 2026-10-02

Exact tree `dd4a01988c9bcf7d66e4ae79107f5443372c037b` passed all 18
deterministic gates with unchanged tracked source and index. Pytest recorded
1,193 passes and five existing external skips; unittest recorded 338 passes
and 22 existing external skips. The complete output SHA-256 is
`2dfaa8a3d67ee534157efec2afa8bb6bf379faa893d8fe15188e92c4cd50095c`;
the test-manifest SHA-256 is
`7afa5b605e7fd30593ddf82fb43638ce9ac52abdf6505e2259f2b317a8e5277f`.
Publication remains `release=false`. All four earlier failed aggregate
outputs are retained. These source results close the previously pending
aggregate, not physical AEC, quality, installation or release.

The independent source/architecture audit initially passed all seven owned
contracts. A later composed-shutdown source finding reopened the lifecycle
contract; that initial verdict is not approval of the subsequent correction.
The measured source delta at that checkpoint was 45 paths: 39 modified, six added,
none removed, with 8,789 insertions and 366 deletions. This implements a new
retained native capability; it is not a code-reduction refactor. Broader Geek
progress is not applicable to Translator. Local review does not claim a
ChatGPT review or remove the current C2C access limitation.

The first prepared physical HTTP attempt stopped before launching any
daemon, Pulse server or audio process. Verified source and model-cache
bindings passed, but protected same-UID process descriptors made the private
inventory raise `PermissionError`. Its immutable receipt SHA-256 is
`6c078f6afb9f6c2175ad71fe14d1593f145baf82f8285e548e1978730d1810d0`.
The recorded command failed; HTTP and product measurement remain `NOT_RUN`.
Both exact ALSA PCM substreams independently reported `closed` in subsequent
read-only diagnosis. The corrected private probe must use kernel PCM identity
and state as its primary exclusivity evidence, retain incomplete process-FD
visibility explicitly, and reject missing, unknown or foreign PCM custody.
No production state, mixer, deadline, proof lifetime or model changed.

The publication candidate removes one trailing empty line from
`control_policy.c` and adds this factual ledger. V5's 18-gate result remains
bound to its original tree; reuse on the derived candidate requires the
exact nonsemantic delta check, current publication/manifest/schema checks,
and a normal debug rebuild with newly bound binary identities. It must not
be reported as a fresh 18-gate execution on that derived tree.

The next implementation stage after owned-shutdown stabilization is installed
artifact discovery and
transactional installation/rollback. Current compile-checkout/debug paths,
external calibration fixtures and nontransactional desktop installation do
not satisfy relocated installation. Independent RU holdout coverage, full
voice/mode/fallback paired evaluation, online native/real-call E2E, soak and
reproducible release remain unpassed. No manual verification is scheduled.

## Automatic physical HTTP result and shutdown recovery

All three prepared physical attempts are retained. The first stopped before
daemon launch on incomplete process-descriptor visibility. The second stopped
at private Pulse readiness: its Unix socket pathname was 163 bytes. Existing
output-directory configuration supplied a short private runtime packet for the
third attempt; no socket policy, timeout or production audio was changed.

The final attempt passed its graph checks after HTTP GET `200` in 36.340019 ms,
then provider PATCH returned `409` in 1.356933 ms. Its receipt SHA-256 is
`fb5745606531fcdc4c2569107925c370fa0285bdaf676fdd1227f91308bb188b`.
The helper did not retain the HTTP problem body, so the initiating failure is
unknown. Calibration and all seven product checks remain `NOT_RUN`; HTTP Start
latency remains `NOT_MEASURED`. Daemon cleanup is `UNCERTAIN`, not successful:
the receipt did not retain an exit code or cleanup-failure detail. Pulse exited
cleanly and the observed owned resource deltas were empty; host/kernel state
and production were unchanged. Those observations cannot upgrade uncertainty.
The short runtime packet remains untouched and the physical budget is exhausted.

Independent source inspection then established a distinct shutdown defect:
`begin_stopping` revoked the authority needed by the operational bypass path
that shutdown invoked. The real composition includes an audio-mix owner;
the previous composed test supplied none and missed this contradiction.
Separate unbounded round-trip retries could also starve independent drains,
and accepted translation Stop retries renewed their cleanup deadlines.
These source findings are not recovered causes of the final physical failure.

The bounded correction separates owned shutdown from normal HTTP Stop. Existing
supervisor, controller and guardian remain the runtime/lease/graph owners;
main owns ordered drain composition and retains every uncertain owner plus
unfinished accepted work. No new state machine, scheduler, acquisition backend,
proof policy or provider selection is introduced. HTTP drain remains seven
seconds, and each sequential runtime-owner drain has one eight-second absolute
deadline. The existing AEC shutdown is still attempted. This is not a total
eight-second shutdown claim or verified resource release after a timeout.

BDD S1-S7 cover stopped/running/pending composed shutdown, retained custody,
quarantine and Stop faults, bounded retries, independent drains, receiver
closure and unchanged ordinary Stop. Independent BDD critic/auditor and
pre-RED SRP review passed. The actual strengthened main RED had four passes
and seven failures; the library RED had 15 passes and two failures. The tests
observe actual Stop calls, quarantine modes, nonrenewing deadlines and a held
round-trip thread, not merely a projected stopped status. Both failed outputs
remain immutable. Fresh focused and aggregate results belong to this new
candidate and are recorded separately from the old 18-gate source result.

The final focused run passed all 60 checks: 30 control-application tests,
17 round-trip owner tests, 12 daemon shutdown tests and one retained-custody
return test. The first GREEN attempt retained one failed fault-fixture oracle:
the deadline-aware drainer had already finished while its underlying thread
remained owned. A concurrently accepted shutdown now holds the actual owner
lock beyond the deadline; the corrected test requires the exact unfinished
drainer handle and joins both accepted calls after releasing the fixture.
The earlier failed GREEN output is retained, not replaced by the final result.
Independent final three-file SRP inspection passed; no additional refactor was
needed. The correction changes three existing Rust files, with 882 insertions
and 69 deletions including fault regressions. It adds a stronger shutdown
correctness property, not a code-reduction refactor. Fresh combined aggregate,
exact source/artifact binding and fork publication are separate recorded gates.

Five supplemental executions passed, adding four distinct fatal-custody and
retry/independent-drain cases: 64 unique focused tests passed in total.
The three-file delta separates runtime code (+211/-56, net +155) from tests
(+671/-13, net +658). Source S1-S7 are 7/7 at the deterministic boundary;
new physical, quality, product and release gates remain zero. Geek is not in
scope. The prior failed physical and aggregate evidence is unchanged.

The first fresh aggregate on tree
`afed61ef2ecc9f3465f1477a0caf81415e79c53e` passed its first six gates, then
failed the external control-application integration suite: 21 passed and
12 failed. Independent inspection found obsolete operational-bypass and EOF
Idle expectations, plus fault queues targeting a removed bypass operation.
Those failures remain recorded. They do not establish another production
defect or permit dropping the integration suite. The corrected tests must
preserve real Stop counts, runtime and lease custody, quarantine ordering,
unknown-mix recovery, cancellation and the distinct ordinary HTTP Stop path.
All 33 external tests must run together before a new aggregate candidate is
frozen. The three production shutdown files remain unchanged in this step.

The corrected external suite passed all 33 tests together, including its 21
previously passing cases. It now checks actual stop counts and retained leases,
quarantine-only shutdown, permanent EOF Stopping, failed quarantine recovery,
caller cancellation and restart after ordinary Stop. The test-only correction
adds 266 lines and removes 64 in one existing integration file; runtime code
does not grow. Together with the 64 focused unit cases, 97 distinct shutdown
checks have passed. The failed aggregate is retained; the complete source gate
must still run on the newly frozen tree.

The second fresh aggregate on `9225e7946bc6812f6e16d50fc8be21769aabb620`
passed those 33 cases but failed the mixer integration's obsolete final bypass
volume expectation. That failure is also retained. The two mixer cases now
contrast actual operational Stop/bypass/restart with owned shutdown's three
verified zero-volume writes, while keeping cancellation, queued watchdog and
desired-state assertions. Mocked pactl is deterministic integration evidence,
not physical audio. A source-level sweep covered all 51 external
ControlApplication shutdown sites across three Rust test files; the other
42 sites belong to unchanged owners. No further obsolete assertion was found.
The final correction changes only tests; complete daemon-target and aggregate
results remain separate gates.

The complete daemon all-target run then passed 662 tests with nine declared
external skips; both mixer cases passed. Its immutable log SHA256 is
`06063542073e622b0f944b4775029c05bb77b463492a90c521016ffc6f3fa770`.
The five-file shutdown correction is +1226/-152, with runtime net +155 and
test net +919. No runtime change was needed for either integration mismatch.

Production, model pins, shared mixers, acoustic admission, proof lifetime and
the four-second Start deadline remain unchanged. This correction does not
promote models, validate AEC, close Task7's historical 5,968-ms debt, provide an
independent holdout, or authorize merge/release. All remaining checks are
automatic; no manual listening or review handoff is scheduled.

## Bounded receipt recovery, 2026-10-02

The third aggregate on `4676ddeee0d799fcaa7375576c4256c359b86752`
passed ten gates, then rejected the sidecar child output. Its pytest verdict
is UNKNOWN, not PASS or a demonstrated product failure; seven later gates
were NOT_RUN. All three failed aggregates remain immutable and their local
source-node budget remains exhausted. No fourth aggregate or physical run
was performed during this recovery.

The existing receipt boundary now retains completed-child exit status and
exact stream byte counts/SHA256 without releasing payloads or exception
context. Strict UTF8, canonical receipts, warning/resource rejection and
process custody are unchanged. Before a trustworthy completion, timeout or
custody failure does not invent a child exit. Pytest and unittest share one
bounded, sorted, manifest-bound failure projection; success output is unchanged.

Independent BDD critic/auditor, actual RED critic and final source/SRP review
passed. Three new regression cases failed before implementation; six focused
cases then passed. The first complete module run retained one failure caused
by the newly emitted diagnostic reaching an isolated fixture's outer stdout.
That fixture now captures and asserts the diagnostic while preserving its
prohibited-body check. The final module run passed all 130 cases in 36.255s;
log SHA256 is
`6801161d9548c8b4aed2d1a3c130f563f3a7de936afef055d0afc018356dca38`.
The standard manifest collector and self-tests passed; collection counts are
not freshly executed full-suite results. Runner/test delta is +306/-52,
net +254 (runner +49, tests +205), no added/removed source files. This adds
diagnostic correctness; it is not a code-reduction refactor or product gain.

A separate unchanged-candidate pytest diagnostic timed out after 300s with
empty streams. A later private probe using pytest's built-in per-test
faulthandler timeout localized a blocked thread join at `del warmup` in
`test_startup_failure_rolls_back_native_server_and_parent_fd`; its 45s timeout
remains failure. This does not establish the cause of the earlier receipt
rejection, identify a particular parameter, or prove a production gRPC defect.
No gRPC implementation or fixture correction was made in this recovery.

The correct-workspace read connector is now accessible. Automatic ChatGPT
planning/review still lacks its supported browser runtime; local audits are
not ChatGPT approval. The original graph/C2C task, iteration and failures are
preserved. Complete source, paired quality/latency, physical AEC, real-app/soak,
installation and release remain NOT_DONE. Production is unchanged; no commit,
push, merge, activation or release was performed.

## gRPC fixture recovery, 2026-10-02

The follow-on diagnosis reproduced the full gRPC-file hang without the manifest
recorder. Its preceding UDS permission test failed: `mkdir(0755)` creates
`0700` under the validator's protective `umask 077`. The production permission
check correctly admitted that directory, but the negative test closed its loop
without stopping the unexpectedly started server. An isolated run under 077
failed and timed out with native cleanup on a closed loop; the same unchanged
test under 022 passed. This explains the new reproducible fixture hang, not the
unavailable output from the original aggregate's child.

The fixture now sets and checks actual `0755` permissions and proves rejection
before native construction. All seven negative parent/inode cases use a common
test-only helper that stops the server in `finally` on the same live loop.
A real-native unexpected-success control retains the missing-exception failure
and verifies cleanup. A restoring fixture runs the negative permission check,
normal private-socket lifecycle and all six 100-round startup faults under
both 022 and 077. Exception classes, inode checks, FD/task/count assertions,
resource guards and production permission/lifecycle code are unchanged.

Independent BDD critic/auditor, RED critic and source/SRP audit passed. The new
mode assertion first failed under 077 and passed under 022. Final focused tests
passed 18/18. The existing canonical sidecar child and parent receipt verifier
then passed 1203 tests with five unchanged declared external skips on tree
`b80013a57e2b05ff6372d7ab27fe921b6a9c6035`; log SHA256 is
`5247d385e5f1fe086da6f4a5fbe99829e0fe6c151860545d6cf07e35e1f5175b`.
No full 18-gate aggregate was run. The three old source failures, exhausted
graph budget, three physical failures and uncertain physical cleanup remain.
The subsequent tree adds this documentation only to the tested inputs.
Test-file delta is +74/-16, net +58; production runtime delta is zero. This repairs
the test fixture and its custody, not translation quality, latency or release.

## Headphone duplex and installed preview correction, 2026-10-02

The scoped correction adds authenticated, stopped-only, ephemeral confirmation
of an exact physical headphone pair without rewriting Analog port metadata or
manufacturing AEC proof. Discovery changes/failures and restart revoke it.
The existing direction loop now polls capture/provider events while a scoped
playback future is pending; no extra capture task or lifecycle owner is added.
Failed, cancelled and dropped writes cannot restore playback reuse.

The installed-path fixes cap resources, project cleanup/failure honestly in the
UI, preserve strict CUDA ownership without unprivileged filesystem namespaces,
use the existing 130-second runtime budget for Start rather than the four-second
direction-cleanup budget, and recheck admission before PCM acquisition.
Successful idle maintenance retains Stopped; failed verification still retains
cleanup custody. See `preview-install.md` for measured lifecycle, eight actual
model attempts, two overlap pairs and the five final native UI checks.

The full daemon-target run passed 676 tests with nine declared external skips
before the last idle-maintenance change. That change was separately verified
by all 35 control, 47 API and two mixer integration cases, and all final 312
library and 56 main tests. These overlapping reruns are not additive coverage.
Scoped Clippy, formatting, 26 UI tests, TypeScript/Vite, 16 overlap-harness
tests and 18 preview-service tests passed. Manifest/schema/publication checks
are separate from the complete repository aggregate, which remains NOT_DONE
for this final candidate. Later reviews are root self-audits: agent thread
limits prevented another independent reviewer. They are not ChatGPT approval.

D-062 is only partly addressed: slow local bootstrap is no longer discarded at
four seconds, but Stop still destroys the model host and an in-flight cold
Start serializes queued controls. Automatic development audio also retained a
Hy semantic error despite correct ASR (a candleholder became a lampshade).
Do not tune a literal case or claim improved full-chain accuracy. Independent
holdout, paired baseline quality/latency, Task7, real-app/soak, physical AEC
and stable release remain open. Canonical production is unchanged.
