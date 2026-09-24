# Translator Stabilization

This directory is the release contract for the production-stabilization fork.
The untouched comparison baseline is
`9291e8beafee3e02aaa179178ce460ac9e6c6de2`. Candidate work stays on
`codex/product-clean-20260923` until the frozen release evidence passes. The
earlier `codex/stabilization-20260904` branch is retained as history, not the
current implementation branch.

## Current recovery checkpoint (2026-09-23)

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
The [RU/EN voice pilot](tts-voice-pilot-20260924.md) compares pinned Piper and
Supertonic 3 on 16 WAVs. Piper remains the product baseline: Supertonic's
nonstreaming CPU path is slower to first PCM and critical-number pronunciation
signals remain unresolved. Neither pilot authorizes a model switch.

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
