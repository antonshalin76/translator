# Release Publication Checklist

Use this checklist for a frozen release candidate. The intended annotated tag
is created locally only after source review so its metadata can be scanned.
Merge, tag push, production rollout, and release publication happen only after
every required gate is bound to the same commit and tag object.

## Repository gate

- [ ] The release candidate is a symbolic-branch `HEAD` with complete local
      history: no shallow/promisor state, replace refs, grafts, or hidden refs.
- [ ] At the start of each candidate verification cycle, `git add -A` stages
      the exact source; afterward there is no unstaged tracked drift and no
      nonignored untracked file. Any source change starts a new cycle.
- [ ] `./scripts/translator-validate deterministic` passes from a clean clone.
- [ ] Hosted CI runs that same entrypoint without a second command list.
- [ ] Checkout uses `persist-credentials: false` before candidate-controlled
      validation code runs; no workflow token remains in repository Git config.
- [ ] The tracked test manifest and external-prerequisite allowlist are current.
- [ ] Every manifest gate has the exact `900`-second timeout and runs in an
      isolated process group under a Linux child subreaper. Timeout, SIGTERM
      cancellation, and caller interruption terminate and reap its exact owned
      process tree; a zero-exit leader that leaves any descendant behind,
      including a new-session descendant, also fails and triggers cleanup.
- [ ] Each Python collector runs with `-I`, produces one canonical typed receipt
      and empty stderr, and treats resource, unraisable, thread, live-thread,
      and shutdown failures as gate failures. The outer runner emits no child
      output, only the validated canonical receipt.
- [ ] Pytest third-party plugin autoload, repository `conftest.py` execution
      hooks, and repository warning filters are disabled/forbidden; every
      unhandled pytest framework warning fails collection or execution. Actual
      pytest body invocation is proven without replacing the callable. Only
      native `pytest.Function` items and pinned native fixture/finalizer
      semantics are admitted; doctest/plugin/unittest compatibility items,
      xunit-style fixtures, and unresolved conditional skips fail closed. The
      unittest runner verifies an external result journal plus exact suite,
      fixture descriptors, cleanup registry, and sync/async leaf protocols
      before, during, and after execution. It preserves scalar lifecycle
      returns, cached/manual/reentrant cleanups, cooperative descriptors, and
      repeated standard-suite segments, but rejects module `load_tests` hooks,
      stale descriptor snapshots, unexecuted lazy lifecycle results, and
      non-`None` test-body results.
- [ ] The full repository snapshot remains unchanged after every gate, not only
      after the Python collections.
- [ ] The exact candidate paths equal `config/publication-files.txt`; every
      `scripts/*` entry is mode `100755`, every other entry is `100644`, and no
      symlink, gitlink, or nonregular entry exists.
- [ ] Full-history and immutable candidate-tree secret scans pass their git,
      directory, binary-stdin, and fast-export pipeline positive controls.
- [ ] The pinned Gitleaks archive digest and extracted-executable digest both
      match. The executable is opened once, invoked through its FD, and retains
      the same device, inode, mode, link count, owner, size, mtime, ctime, and
      content digest through version proof and every scan.
- [ ] Every candidate and reachable-history blob is strict UTF-8 text without
      control bytes, except the exact hash-pinned approved public PNG assets.
- [ ] No model asset, debug capture, raw report, cache, socket, log, credential,
      archive, Git LFS pointer, scanner/transform control, or private
      home/source-checkout path matching the controlled platform patterns is
      present in the candidate tree.
- [ ] `./scripts/translator-publication-check candidate` emits one precommit
      receipt containing `release=false`; record its tree ID for review and do
      not use this receipt as release evidence.
- [ ] Independent source-content/provenance review confirms that no private
      user or call prose entered the candidate or reachable history; automated
      credential and artifact scans are not treated as semantic DLP.
- [ ] Validation runs on the clean hosted/CI trust boundary with the pinned
      ShellCheck and scanner versions. Absolute isolated launchers and an
      explicit child-environment allowlist exclude ambient credentials,
      framework/plugin controls, and interpreter startup injection; every
      intentionally PATH-selected tool passes its separately pinned version and
      artifact-integrity control.
- [ ] Cargo-audit, Bun, and OSV-Scanner archive/executable digests match the
      official pinned artifacts. Each scanner is a singly linked regular file
      owned by the current user or root, with no group/other write, set-ID bits,
      or `security.capability`; that identity remains unchanged across held-FD
      version proof and execution, and a path swap fails closed. OSV directly
      scans `Cargo.lock`, `sidecar/uv.lock`, and
      `apps/translator-ui/bun.lock`; no dynamically installed audit environment
      participates in release evidence.
- [ ] The user unit contains no `EnvironmentFile`. Every initial and automatic
      daemon start enters the desktop launcher's isolated environment wrapper;
      launcher preflight parses the same strict allowlist before invoking
      systemd, and unsafe identity/content fails without starting the unit or
      disclosing values. The daemon is executed from the verified held file
      descriptor with inherited loader/Python/shell controls removed.
- [ ] Before the dynamic wrapper is loaded, systemd unsets every loader control
      documented by the pinned host glibc, including
      `LD_TRACE_LOADED_OBJECTS`, plus `GLIBC_TUNABLES`, `GCONV_PATH`, and
      `LOCPATH`; the wrapper then strips every inherited `LD_*`/`PYTHON*` key.
      Stage E replaces this finite pre-wrapper list with the reviewed static
      launcher so a future unknown loader variable cannot bypass execution.
- [ ] Before starting the sidecar, the daemon removes `GLIBC_TUNABLES` and every
      raw `LD_*` key, ignores ambient loader paths, and admits only canonical
      root-or-service-owned CUDA trees without group/other-write or set-ID mode
      on ancestry, contents, or symlink targets. Python independently revalidates
      the same tree, serializes process-global configuration, and loads each
      known runtime through a retained no-follow file descriptor. The packaged
      host CUDA tree and CPU fallback both pass from the isolated service
      entrypoint; direct daemon invocation is not used as production evidence.
- [ ] Treat the gate as integrity proof for the exact independently reviewed
      tree, not as a same-user sandbox for intentionally malicious test code.
      The tree, clean hosted runner, source-review verdict, post-commit
      provenance, and all receipts must have the same identity.
- [ ] After the exact candidate tree is approved, commit it without changing
      bytes, verify `HEAD^{tree}` equals the reviewed tree, and create the
      intended annotated release tag locally at that exact `HEAD`. This planned
      commit/tag transition ends the precommit cycle and starts the release
      verification cycle; the precommit receipt is no longer usable.
- [ ] Rerun
      `./scripts/translator-publication-check release <tag> <reviewed-tree>`.
      It must emit a `v1` release receipt binding the exact `HEAD`, tree,
      complete-ref-state SHA-256, tag name, and annotated-tag object.
- [ ] The release ref's direct outer object is an annotated tag whose canonical
      header names the exact `HEAD` as `object`, declares `type commit`, and
      repeats `<tag>` as its internal `tag` name. A tag-of-tag or mismatched
      internal name fails even when peeling would reach `HEAD`.
- [ ] No commit, tag, ref, index, or worktree mutation occurs in the verified
      checkout after the release receipt. Any such mutation invalidates it and
      requires the full release-mode scan again; merge, tag push, publication,
      and activation must target the receipt's exact objects without altering
      that checkout first.

## Release evidence

- [ ] The mandatory local provider matrix passes every mode, direction,
      language pair, operational fallback, and target voice.
- [ ] Paired candidate-versus-baseline accuracy, latency, drop, restart, RAM,
      and VRAM evals meet the frozen thresholds.
- [ ] Soak, native UI, physical-audio, and claimed real-application E2E evidence
      is complete; unavailable prerequisites are not marked as passes.
- [ ] Independent architecture and security reviews approve the exact candidate
      SHA.

## Artifacts and rollout

- [ ] Two clean hosted builds produce identical hashes from pinned inputs.
- [ ] Checksums, SBOM, signed provenance, release notes, and rollback artifacts
      identify the frozen SHA.
- [ ] The staged candidate reaches authenticated health before traffic moves.
- [ ] Rollback is exercised against the staged candidate.
- [ ] Merge, tag push, publication, and production activation target only the
      release-receipt SHA and annotated-tag object.
