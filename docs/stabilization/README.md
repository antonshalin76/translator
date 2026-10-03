# Translator Stabilization

This directory is the release contract for the production-stabilization fork.
The untouched comparison baseline is
`9291e8beafee3e02aaa179178ce460ac9e6c6de2`. Candidate work stays on
`codex/product-clean-20260923` until the frozen release evidence passes. The
earlier `codex/stabilization-20260904` branch is retained as history, not the
current implementation branch.

## Current product checkpoint (2026-09-27)

The [one-pass product closure packet](product-closure-one-pass-20260927.md)
groups the remaining work into one integrated no-headphones candidate and one
final evidence review. It is a proposal, not release approval. In particular,
the saved-audio runs force Turbo while the local runtime still defaults to
`faster-whisper-small`; the effective installed model choice needs verification.

The [paired saved-audio product diagnostic](product-audio-paired-20260927.md)
ran Turbo→NLLB/Piper and Turbo→Hy/Piper on the same 24 frozen RU/EN recordings.
Three full runs completed; one reverse-order run retained a Hy-side failure.
Independent text review favored Hy's critical-fact fidelity, but identified
remaining translation errors and unstable numerical ASR on one recording.
Forcing deterministic ASR decoding was rejected after it produced a severe
repeated-word error on that recording; the default decoder is unchanged.
The current fork is still not release-ready: the failure cause, Task 7's
physical first-audible debt, acoustic admission and playback remain open.
Production and the default model are unchanged.

The fork also corrects AEC preflight resource custody: cancellation, failed
inspection, and invalid binding now require confirmed graph cleanup before the
calibration lease is released. Deterministic controller tests cover failed
cleanup and retry. This does not enable open-speaker use: no production
calibration engine is connected, and no physical acoustic proof has been run.

The fork now avoids creating a raw-microphone original loopback when the
selected output is an open speaker or unknown, and reconciliation recognizes
only loopback modules marked as owned on both Pulse stream halves. This closes
the normal request path for that bypass, but does not make device changes
atomic: an existing module can remain live after failed unload or until the
next watcher refresh. Open-speaker use remains unavailable pending a real AEC
engine, immediate revocation, and physical validation.

The Task 7 E2E harness now makes at most two attempts to clean up its owned
input sink. Persistent cleanup failures terminate with no success report;
foreign module IDs are not unloaded. This is harness safety evidence only, not
a new physical Task 7 result or a release gate pass.

The paired saved-audio runner also accepts all three provider modes with mode
identity checked from CLI through session, input frames, and failure journals.
Offline provider smoke completed 4/4 attempts in each of the three modes.
The full balanced saved-audio run completed 48/48; the full streaming-first
run retained a safe provider drop on its first English case after 12 Russian
completions, with 35 attempts NOT_RUN. A narrow runner diagnostic now
preserves only validated stage timings on future failed attempts. The failure
cause is not yet known: a one-case replay completed, while both the
failed streaming run and a successful balanced run overlap kernel NVIDIA
memory-allocation errors. Neither software result proves live speech or
playback. Physical bidirectional E2E remains a separate gate. Without
headphones, open-speaker Start also remains closed: the daemon has no real
AEC calibration engine or attached runtime observer.

## Recovery baseline (2026-09-23)

The last code commit is `b239b9a1be094a4f2c1f53132e49a664bbf020fa`
(tree `bf37345df5cad1e0d35bebd71d14e6260198e1f4`). It passed the complete
18-gate deterministic suite and a bounded Task6 local-chain run. This is not a
Stage B2 close or release: paired product accuracy/latency, Task7's recorded
5968 ms first-audible failure, real acoustic admission without headphones,
packaging, and physical end-to-end behavior remain open. The main daemon still
sets `aec_calibration: None`, so open-speaker validation is not available.

Do not infer a speed or accuracy win from the Task6 candidate/baseline reports.
The baseline run used a lower CPU quota after thermal throttling, while Piper
produced different PCM for repeated synthesis of the same text. The reports and
their hashes are in the local evidence packet outside this repository at
`translator-product-evidence-20260923`; the summary is not a release artifact.
Both reports have the same critical-review translation-output SHA-256
(`e7ef4511e68e0e7e08d1c6d62780b54aa10eed384e9a3bb4b0deeb533b392ffe`):
the measured text translation did not improve.

The next product measurement must reuse identical input PCM and model hashes,
with the same build profile, CPU limit, cache condition, and timing boundary on
baseline and candidate. Separately, a production `AecCalibrationEngine` must
acquire real acoustic observations and attach to the daemon; none exists yet.
The device watcher currently fixes AEC capability when constructed, so safe
dynamic admission and revocation also remain to be implemented. A simulated
graph test may establish cleanup ownership, but cannot authorize a physical
open-speaker Start. No further C2C iteration substitutes for these gates.

The production checkout and user service are not test targets during
refactoring. Candidate installation, restart, rollback, and real-call tests are
allowed only after the deterministic, security, and packaging gates pass.

The [2026-09-24 ASR input screen](asr-input-screen-20260924.md) freezes Turbo
as the current comparison leader and records the five-model, paired development
screen. It is not an ASR model-switch, full-chain result, or release gate.

The [MDC Spontaneous Speech comparison](asr-mdc-sps5-comparison-20260924.md)
adds a speaker-disjoint RU/EN Turbo-vs-Qwen holdout. Turbo remains the
single-model development baseline for downstream MT/TTS evaluation; no
production model was changed.
The [AppTek English dialogue diagnostic](asr-apptek-dialogues-20260927.md)
adds 28 separate calls across 14 accent groups and publisher-manual references.
Its WER ranking changes under number normalization, while both models retain
critical fact errors. Turbo remains the input baseline; this is not a release
holdout or model switch.
The [local audio-first critical screen](audio-first-critical-screen-20260925.md)
adds an eval-only direct-WAV diagnostic for the frozen critical cases. Its
exact-source run retained an Ollama CUDA failure, so it is not an audio-truth
or release gate.

The [fork-only Turbo local-chain smoke](turbo-local-chain-20260924.md) adds an
opt-in pinned Turbo runtime to the fork and measures two ASR→NLLB→Piper paths
against small. Production remains unchanged; live audio and release gates are
still open. A subsequent two-direction local-provider session check completed,
but an EN reference said *Javanese* where the recognized and translated result
said *Japanese*. The complete chain therefore has a confirmed critical
semantic error, not a quality PASS. The publication policy now distinguishes
Python source invoked through an interpreter from executable commands under `scripts/`;
the staged candidate passed its precommit publication check with
`release=false`. This is not release evidence and does not authorize a merge
or production activation.

The [natural-text MT pilot](mt-natural-pilot-20260924.md) compares pinned NLLB
and Hy-MT2 on 12 diagnostic RU/EN phrases. Hy-MT2's higher chrF2 comes with
roughly doubled CPU text latency and a participant-role error; neither model
was changed in production. A larger independent critical-error eval is needed.

The [33-pair FLEURS MT diagnostic](mt-fleurs-paired-20260924.md) broadens the
text comparison and finds second-sentence omissions in NLLB, but its bootstrap
intervals include zero and its sentences are not product-domain dialogue.
The [prospective Turbo-text MT comparison](mt-mdc-pair-20260926.md) uses 24 new
clean MDC origins: Hy-MT2 retained more critical facts than NLLB on the saved
ASR text, and GPU offload reduced warm MT latency on this machine. One garbled
ASR case still fails; this is not a live-audio or release result.
The [opt-in Hy-MT2 product adapter](hy-mt2-product-20260927.md) now runs through
the local provider and Piper on two fixed recordings, with request-scoped
failure isolation and a 24-text regression against the earlier GPU run. It is
not the default model or a release decision.
The [RU/EN voice pilot](tts-voice-pilot-20260924.md) compares pinned Piper and
Supertonic 3 on 16 WAVs. Piper remains the product baseline: Supertonic's
nonstreaming CPU path is slower to first PCM and critical-number pronunciation
signals remain unresolved. Neither pilot authorizes a model switch.
The [bounded Piper product loader](piper-bounded-20260927.md) limits the
four verified voices' ONNX worker pools and passes a saved-output PCM probe.
This is a fork-only resource and format result, not live-audio release evidence.

## Audited contracts

- [`master-bdd.md`](master-bdd.md) defines 48 executable behavior scenarios.
  Its initial independent review passed; the strengthened working copy is
  approved only by a frozen Stage A verdict for the same exact tree.
- [`architecture-owner-map.md`](architecture-owner-map.md) assigns one owner to
  each lifecycle decision and records the library-first decisions. An exact-tree
  architect verdict is required; an earlier review does not approve later bytes.
- [`defect-register.md`](defect-register.md) is the live defect-to-test map. A
  row is not closed by a code change alone; its authoritative scenario and the
  broader stage gate must pass.
- [`stage-a-evidence.md`](stage-a-evidence.md) binds the deterministic-gate
  recovery to exact collection and source hashes, independent review, and the
  remaining release blockers.

## Discovery baseline

The initial clean-checkout audit found:

- Rust formatting, tests, and Clippy passed on the baseline.
- The UI had 15 passing unit tests and a successful production build.
- The complete sidecar collection had 598 passes and three deterministic
  failures: two stale TTS test-double signatures and one test requiring removed
  private benchmark reports.
- The root `unittest` collection had 49 tests: 27 passes and 22 declared skips
  for intentionally unpublished local evidence.
- Ruff reported an undefined Python type parameter plus pre-existing formatting
  and import drift not covered by CI.
- The lockfiles contained fixable advisories in `h2`, `nanoid`, and `pytest`.
  The Tauri Linux dependency graph also contains the GTK3/glib advisory cluster,
  which requires an explicit reviewed exception or a future supported backend
  migration; it is not recorded as fixed.

These numbers are discovery evidence, not release evidence.

## Stage status

| Stage | Scope | Status |
| --- | --- | --- |
| A | deterministic suites, manifests, CI parity, bounded SCA | complete at `af3410c`; full 18-gate PASS and two exact-tree review receipts; pushed to the fork branch |
| B1 | Python provider/model ownership and private IPC | complete at `6c81cc3`; 18-gate PASS, exact-tree architecture/security approval, pushed to the fork branch |
| B2 | daemon authority, safety, atomic controls | in progress; corrective native and adapter tests pass; independent source review, code-surface consolidation, admission and stage/release gates remain open |
| C | canonical modes, deadlines, flow control, real endpointing | blocked on B |
| D | semantic accuracy, audible oracle, debug, readiness, metrics | blocked on C |
| E | portable bundle, systemd lifecycle, rollback, API/docs | blocked on D |
| F | frozen corpus, paired eval, soak, native/real-app E2E, release | blocked on E |

Every stage close requires source gates, refreshed documentation, a quantitative
code-surface delta, and an independent architect-auditor verdict before commit
and push. Merge, production activation, tag push, and release publication remain
forbidden until Stage F binds all evidence to one exact SHA and annotated-tag
object. The local annotated tag needed by the Stage F release-mode publication
gate is created only after the exact candidate tree is reviewed.

The B1 working evidence packet is [`stage-b1-evidence.md`](stage-b1-evidence.md).
It is not a release or live-quality receipt.
