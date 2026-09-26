# Stage A evidence: deterministic validation recovery

Baseline: `9291e8beafee3e02aaa179178ce460ac9e6c6de2`.
Candidate branch: `codex/stabilization-20260904`.

This is source-bound Stage A evidence, not release, merge, installation, model
quality, or production-activation evidence. At document freeze, the manifest,
self-tests, focused negative controls, and full root unittest outcome gate pass.
The immutable 18-gate execution receipt and final architect verdict belong to
the external close packet for the exact staged tree; keeping those outputs
outside this tracked document avoids changing the tree they attest.

## Baseline reproduced

- UI: 15 unit tests passed and the production build completed.
- Rust: formatting, workspace tests, and Clippy passed.
- Sidecar: 598 tests passed and three deterministic tests failed. Two doubles
  had stale TTS signatures; one test required removed private benchmark files.
- Root unittest: 27 passed and 22 were skipped for unpublished external
  evidence.
- Full owned Python scopes had 162 Ruff findings, including two undefined type
  parameter references.
- Lockfiles contained remediable advisories for `h2 0.4.15`, `nanoid 3.3.16`,
  and `pytest 8.4.2`.

## Candidate contract

The only CI entrypoint is:

```bash
./scripts/translator-validate deterministic
```

It executes one manifest-bound argv sequence of 18 gates. Runtime/cloud
`TRANSLATOR_*` variables and `OPENAI_API_KEY` are removed before collection and
execution; only named validation-tool overrides survive. The exact external
allowlist requires framework, full test ID, and one stable
`missing_external_prerequisite:<name>` code.

Pytest setup/call/teardown and unittest start/outcome/stop events are reconciled
against every manifest ID. Collection errors/skips, deselection, duplicate or
missing lifecycle events, unexpected runtime skips, xfail/xpass, and forged
successes fail closed. Each Python suite emits exactly one canonical compact
JSON receipt. The parent compares its bytes to the independently constructed
expected payload, so duplicate keys, scalar type substitution, whitespace
variants, missing/duplicate receipts, or an outcome mismatch cannot pass.
Parent and child bind the same immutable manifest/allowlist snapshot. Rust
doctests run but remain forbidden until an exact cross-crate ID inventory is
available; discovery of one invalidates the manifest.

## Current tracked manifest and expected outcomes

| Framework | Files | Nodes | External skips | Required outcome |
| --- | ---: | ---: | ---: | --- |
| Bun | 1 | 15 | 0 | 15 passed |
| pytest | 22 | 633 | 1 | 632 passed, 1 exact GPU-cache prerequisite |
| Rust | 33 | 381 | 1 | 380 passed, 1 exact physical-audio prerequisite |
| unittest | 13 | 267 | 22 | 245 passed, 22 exact private-evidence prerequisites |
| Total | 69 | 1,296 | 24 | 1,272 deterministic passes |

Before the frozen integrated run, the following current-snapshot checks pass:

- manifest write/check/self-test;
- all 119 manifest-execution, 57 publication, 17 supply-chain-admission, and 8
  CI contract tests; desktop launcher/unit has 22 passes and 2 exact
  private-evidence skips, with pinned Gitleaks 8.30.0 and
  `ResourceWarning=error` where applicable;
- the manifest-owned sidecar execution: 632 pass and one exact GPU-cache skip;
- the manifest-owned root unittest execution: 245 pass and 22 exact skips;
- the host-namespace Rust workspace execution: 380 pass and one exact
  physical-audio skip; the doctest inventory is empty;
- pinned Bun 1.3.12 frozen install, all 15 UI tests, and the production build;
- Ruff check/format, Bash syntax, and pinned ShellCheck on the changed gate
  surfaces.

The restricted workspace sandbox denies the asyncio event-loop self-pipe and
Unix-domain sockets: a minimal `asyncio.to_thread` control times out there and
the worker-side self-pipe send returns `EPERM`. The exact sidecar receipt was
therefore run outside that seccomp boundary in the runner's otherwise sanitized
environment. The refreshed 633-node receipt completed in 7.76 seconds; its
additional gRPC GOAWAY line is the test UDS server's shutdown diagnostic, not a
failed outcome. This environment distinction is not recorded as a scheduler
pass inside the restricted sandbox.

The final external close packet must add one uninterrupted successful 18-gate
run for the staged tree, its candidate-mode publication receipt, Python outcome
receipts, log SHA-256, and two independent exact-tree verdicts. A previous-tree
run or review is not reusable.

## Integrity identities

- Master BDD SHA-256:
  `2f72f52980fedd849c6436d8dd048f2d869bddd5f55758279fba7111846e0af7`.
- SRP owner map SHA-256:
  `0a885c29506540b25bdbcd524028e5e94c3f5399819781dc59d21e61fccc3cf1`.
- Test manifest SHA-256:
  `5c89d14da2302a79eb12bae23f47b9f3058ba156b2f5e36c202ab579a4f876d4`.
- Manifest runner SHA-256:
  `e0f4ca5245ca41a23ab98120522c801f2fd8fd2a99b19086a896100041ff66dc`.
- Validation wrapper SHA-256:
  `255eaacff853e40c91a0c5e83a1e4ea5c2bfb68c211be858de4adc368d1e73c0`.
- Publication gate SHA-256:
  `6e9be3fb09030eeb9e6828eb4784cf84cba169cde7cbd892769427aea457b479`.
- Required pytest outcome SHA-256:
  `cc34766350f8f84c30a5eaae72c0afe27bd924be909579e493aaee3f006a084d`.
- Required unittest outcome SHA-256:
  `11ad5646ca4b865f31bdbe727414db977c9a116c2f2c6e4e78c25e73b37e5ac8`.

## Independent review and repairs

- Python-quality review found a root/config mismatch and a tautological fixture
  hash assertion; both were corrected.
- SCA review found public-symbol reachability, grep-error, timeout, symlink,
  caller-selected same-version scanner, attacker-writable/set-ID/capability
  scanner, path-replacement, and unlocked dynamic Python-auditor fail-open
  paths. Official executable digests, opened-once held-FD execution,
  owner/mode/set-ID/capability checks before and after every invocation, and 17
  negative/control tests cover the admitted scanners. The redundant dynamic
  `pip-audit` environment was removed; pinned OSV scans all three reviewed
  lockfiles directly. `h2`, `nanoid`, and `pytest` were upgraded in lockfiles.
- CI/portability review found a stale unit assertion, machine-dependent skip,
  incomplete Bun inventory, unbound runner, detached CI checkout, and missing
  ShellCheck coverage. CI now attaches a symbolic branch to the exact checked
  event commit, disables checkout credential persistence before candidate code,
  and invokes only the manifest runner.
- User-service review found that systemd injected `EnvironmentFile` values
  before `ExecStartPre` and that the desktop `prepare` path could accept invalid
  content before starting the unit. The unit now always enters one isolated
  launcher wrapper, including automatic restarts. Launcher preflight and unit
  execution validate held configuration/file descriptors, stable content, and
  a strict literal key allowlist; the daemon starts from its exact held
  descriptor after inherited loader, Python, and shell-startup controls are
  removed. A follow-up host control proved that `LD_TRACE_LOADED_OBJECTS`
  prevents the dynamic Bash wrapper from executing when it reaches the loader;
  the unit now removes the current documented glibc loader/tunable set before
  that boundary. Twenty-four focused Task 9 tests cover the boundary and
  `systemd-analyze verify` accepts the unit; live activation and the Stage E
  static-launcher replacement remain later evidence.
- Native-loader review found that the daemon merged an ambient
  `LD_LIBRARY_PATH` with any absolute existing operator CUDA directory and that
  Python reopened preload names by mutable path. A writable directory or
  symlink target could therefore execute code in the credential/audio-bearing
  sidecar. The candidate strips every raw `LD_*` key plus `GLIBC_TUNABLES`,
  validates the complete bounded canonical library tree at the daemon boundary,
  independently repeats admission in Python, and preloads known runtimes only
  through stable retained descriptors. The original unsafe-directory,
  unsafe-file, symlink-target, ambient-path, and mutable-name controls failed
  against the prior implementation. A host probe then exposed an incorrect
  cuBLAS preload order that mocks had hidden; `libcublasLt.so.12` now precedes
  its dependent `libcublas.so.12`. A final adversarial review found that whole
  `canonicalize` skipped a replaceable intermediate symlink; both boundaries
  now resolve and validate every chain component. The focused candidate suites
  pass all 21 Python and 12 Rust CUDA-admission tests. They include exact
  4096/4097-entry and 40/41-symlink boundaries, sticky-directory policy,
  partial preload cleanup, concurrent same/conflicting identity admission,
  loader-environment rollback, and interrupted descriptor handoff. Python now
  serializes the complete process-global transaction and records each committed
  preload in one handle/FD/identity registry. Outside the sandbox, the current
  root-owned CUDA tree passes admission, held-FD preload plus CTranslate2 sees
  one GPU, and all six Rust-to-Python sidecar lifecycle tests pass. The missing
  cuDNN 9 package and final packaged-tree proof remain Stage E work.
- Executed-outcome reviews found indirect skips, ambiguous pytest phases,
  mutable contract rereads, absent/forged child receipts, unittest success
  without `startTest`/`stopTest`, JSON bool/int equality, and duplicate-key
  acceptance. Later standard-mode review also found stale unittest descriptor
  binding, cached/reentrant cleanup gaps, skip-reason precedence drift, and a
  module `load_tests` filtering escape. The final runner audit found a
  decorator-order escape in conditional-marker validation and a live-reap gap
  where the runner's subreaper ownership kept a killed Rust descendant as a
  zombie until after its waiting parent failed. The follow-up ownership audit
  found that shared-host lifecycle tests could import native threads or adopt a
  Git-maintenance zombie, and that the first isolation proxy accepted a runtime
  skip, expected failure, or hidden non-`None` body as success. The final
  containment review then found that default outer-host `SIGTERM` bypassed the
  proxy's `finally` block and that timeout cleanup could race a forking
  descendant. A final BDD/SRP critique rejected the initial static-tree test as
  insufficient; the replacement control forks a detached process after the
  stopped-root snapshot and proves a later fixed-point scan removes it. The
  `/proc` inspection-failure control deliberately proves only known-root
  cleanup before reporting failed containment. Framework-owned event recorders,
  immutable snapshots, exact lifecycle checks, and canonical byte receipts
  replace the source heuristic. Complete ordered `skipif`/`xfail` validation,
  strict per-gate exclusive-owner admission, bounded exact-PID live reap,
  supervisor-side normalized parent-death signalling, stop-to-fixed-point
  watchdog cleanup, and one-clean-pass exec-isolated lifecycle tests now fail
  closed without stealing foreign status or changing the shared test host.
- A logged close-packet rerun exposed an intermittent native gRPC INFO line:
  ordinary server shutdown used `grace=0`, cancelled closing RPCs, and emitted
  GOAWAY diagnostics that the strict receipt gate correctly rejected. The
  authoritative server lifecycle now gives in-flight RPCs a named 250-ms
  graceful-stop budget inside the daemon's 500-ms SIGTERM bound while startup
  rollback stays immediate. Before the fix,
  five of eight repeated gRPC-file runs emitted stderr; after it, 40 of 40
  gRPC-file runs and five of five complete manifest-owned sidecar runs were
  clean. The gate still rejects every stderr byte instead of filtering it.
- Publication reviews found staged/worktree divergence, symlink/gitlink and
  transform escapes, shadowed/false-clean scanners, incomplete history/ref
  views, archive/binary/control bypasses, private-path variants, ref/worktree
  races, and a precommit receipt that could not cover later commit/tag metadata.
  Candidate mode is explicitly non-release evidence. Release mode requires the
  committed reviewed tree and an annotated tag targeting HEAD, rescans all
  now-existing metadata, and binds HEAD, tree, tag object, and complete-ref
  digest. Fifty-seven publication tests cover the real scanners and adversarial
  candidate/release paths.
- Documentation review found 12 stale count, command, hash, status, and
  publication statements. The executable BDD, checklist, owner map, defect
  register, and Stage A status were synchronized before freeze.

Earlier approvals and full-gate logs were invalidated whenever a later review
found a defect. Only the exact-tree verdicts in the external close packet can
close Stage A.

## Supply-chain disposition

- `h2` is locked at `0.4.16`, `nanoid` at `3.3.18`, and `pytest` at `9.1.1`.
- CI action revisions and Rust, Python, uv 0.12.9, Bun, cargo-audit, Gitleaks,
  OSV-Scanner, and ShellCheck versions are pinned. Cargo-audit, uv, and Bun
  archives plus extracted executables are SHA-256 verified; direct Gitleaks,
  OSV-Scanner, and ShellCheck artifacts are also digest-pinned. The SCA gate
  separately admits cargo-audit, Bun, and OSV-Scanner by executable digest and
  stable held-FD identity, trusted owner, non-writable/non-set-ID mode, and
  absence of file capabilities. Python dependencies are covered by OSV's direct
  scan of `sidecar/uv.lock`, not a dynamically installed audit environment.
- The GTK3/glib unmaintained transitive advisories are explicit temporary
  exceptions, not fixes. They expire on 2026-12-01 and remain owned by the
  Stage E desktop-backend decision.

## Code-surface delta

Against the baseline, the candidate changes 91 files: 22 additions and 69
modifications, with 26,894 added and 1,746 removed lines. The generated exact
test manifest contributes 7,919 lines.

The production-runtime subset is 13 files: daemon `process_sidecar.rs`, sidecar
`grpc_server.py`, nine modules under `translator_sidecar/local`, and the OpenAI
provider/runtime pair. It is +1,008/-140 lines, net +868. This Stage A growth adds
portable runtime/model integrity contracts; it is not represented as a
refactor-only reduction. Most other growth is test inventory, negative
controls, and deterministic evidence. The obsolete AST skip heuristic was
removed instead of retained beside framework-owned outcome recording.

## Explicit residual risks

- The portable model cache is intentionally empty on a clean machine. Stage E
  must provision/migrate every pinned asset atomically and prove rollback before
  candidate activation.
- The removed private cuDNN path was the only cuDNN 9 location found on this
  host. Stage E must bundle a portable dependency or provision an explicitly
  admitted `TRANSLATOR_CUDA_LIBRARY_PATH`. The managed sandbox maps system CUDA
  ownership to UID 65534, which production intentionally does not trust; exact
  root ownership/modes and real CUDA startup must be proved at the host boundary
  before compatibility is claimed.
- Python executable and sidecar-root overrides still cross mutable paths before
  child spawn. Stage E must replace checkout-backed execution with an immutable
  installed bundle and prove executable/module-tree replacement races fail
  before any sidecar secret is inherited; D-052 remains open, so Stage A makes
  no general subprocess-path security claim.
- Python direct/debug held-FD preload is proven for the current three-library
  cuBLAS tree only. [NVIDIA documents](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html)
  that cuDNN 9 uses split libraries, including engine libraries loaded through
  `dlopen`; arbitrary direct-run cuDNN layouts remain unsupported until Stage E
  adds a trusted validate-then-exec launcher. The supervised daemon path
  supplies the admitted search path before the child loader starts.
- The Rust daemon validates the complete CUDA tree and then gives its canonical
  pathname to the child loader; root and the service UID are trusted against
  same-owner mutation during that interval. Stage E's immutable installed tree
  must remove this path-based post-validation window before release.
- Python retains every registered raw CUDA FD on ordinary and tested async
  failure paths. An asynchronous exception injected inside the `os.open` call
  itself can still escape before Python receives the descriptor; normal process
  termination closes it, and Stage E's static launcher remains the owner of a
  stronger acquisition guarantee.
- Writable model provisioning under the hardened user unit still needs an
  isolated-home integration proof.
- Gitleaks 8.30.0 misses a detector-positive token in a NUL-bearing tar member.
  The repository rejects unapproved binary/control-bearing members before that
  scan and independently scans normalized bytes; the upstream scanner defect
  remains recorded as an external tooling risk.
- Stages B through F remain open. No accuracy, latency, soak, native UI,
  real-call, installation, rollback, merge, tag, publication, or release claim
  is made by Stage A.

## Quantitative progress checkpoint

- Stage A deterministic-validation recovery: 95% to 98% (`+3 pp`) at source
  freeze; exact-tree full-gate and reviewer receipts raise it to 100%.
- Recovery/state/policy product goal owned by Stages B and C: 0% to 0%
  (`+0 pp`); Stage A does not claim runtime-state repair.
- Overall Translator stabilization/release goal: 19% to 20% (`+1 pp`) at source
  freeze; Stage A close is estimated at 21%.
- The separate autonomous Geek goal is outside this repository and remains
  unchanged (`0 pp` delta).
