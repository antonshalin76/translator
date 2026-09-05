# Translator Stabilization

This directory is the release contract for the production-stabilization fork.
The untouched comparison baseline is
`9291e8beafee3e02aaa179178ce460ac9e6c6de2`. Candidate work stays on
`codex/stabilization-20260904` until the frozen release evidence passes.

The running production checkout and user service are not test targets during
refactoring. Candidate installation, restart, rollback, and real-call tests are
allowed only after the deterministic, security, and packaging gates pass.

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
| B1 | Python provider/model ownership and private IPC | source checkpoint complete; close requires same-tree full-gate and final architect receipts |
| B2 | daemon authority, safety, atomic controls | follows reviewed B1 close |
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
