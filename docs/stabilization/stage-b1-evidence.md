# Stage B1 working evidence

Base: `af3410c2dfc0724b9b785468690e9b1a61b7ea98`, the reviewed and pushed Stage A
checkpoint. Production remains at `9291e8beafee3e02aaa179178ce460ac9e6c6de2`.
This packet defines the B1 source checkpoint. Closure requires the external
full-gate and final architect receipts for the same frozen tree; this document
alone is not a passing validation receipt.

## Scope and ownership

| Scenario | Decision owner | Mechanical boundary | Evidence |
| --- | --- | --- | --- |
| TST-3 | `ProviderRegistry` selection, leases, retirement | gRPC acquires/releases a lease; `__main__` transfers a prepared provider synchronously | A-B-A selection, cloud/local leases, concurrent/repeated shutdown, failed-cleanup retry, bootstrap cancellation and post-transfer retirement failure |
| TST-3, MODEL-1 | `LocalProvider` shutdown; native adapters own model lifetime | `InferenceScheduler` joins inference before adapter close; `VerifiedModelLease` uses `ExitStack` and sealed memfd snapshots | cancelled shutdown, failing model close, immutable byte/hash and descriptor-lifetime tests |
| CLOUD-2/3/4 | logical session and immutable utterance connection generation | pinned async WebSocket client, strict Pydantic wire types, existing event adapter | handshake faults, 1000 terminal cycles, 100 real loopback failed handshakes, EOF drain, late events, sequence scope, publication failure |
| PROTO-2 | daemon runtime-directory admission and sidecar listener | locked `rustix` `openat2(NO_SYMLINKS)`; Python descriptor-relative `os.open` | ancestor symlink rejection before state creation/unlink, held directory identity and descriptor cleanup |
| TST-1/2 | one production `RuntimeProvider` execution interface | explicit engine adapter under tests only | empty registry fails closed; production-signature transport tests |
| EVAL-1 preparation | benchmark orchestration owns candidate resources | verified model sources and explicit close | unchanged corpus, thresholds and output schema; no reuse of closed components |

No database, persistence schema, UI authority, broker, or model-routing policy is
added. Registry state and model/socket resources are process-local. Tests own the
evidence; the existing daemon projection remains the UI authority. The registry
is event-loop-owned: selection and acquisition contain no await, so an additional
lock would not strengthen their linearization boundary.

## Reproductions before fixes

- A leased local A, replaced by B and then reselected, was shut down on release
  of its old lease. Expected shutdown count `0`, observed `1`.
- A failed local shutdown skipped cloud shutdown. Expected cloud cleanup count
  `1`, observed `0`.
- A server with no provider accepted a successful session through the implicit
  mock engine. The negative-control test expected rejection and failed.
- `RuntimeLease::acquire` followed an ancestor symlink and created runtime state.
  The new no-mutation test failed before the directory admission change.
- Stale-socket cleanup followed an ancestor symlink and removed the socket.
  Expected `InsecureParent`, observed `Ok(())`; the test failed before the fix.

Model acquisition and cloud protocol reproductions use the real adapter seams
with small deterministic model fixtures and a loopback WebSocket server. They
do not establish native large-model performance or OpenAI service compatibility.

## Review and validation status

The independent model-owner investigation confirmed mutable-path reopening and
the installed libraries' file/byte loading seams. The independent engine audit
confirmed that gRPC was the only production consumer of the mock engine.
The first independent implementation review found terminal-ID replay and lost
failed-socket ownership. Both received additional regressions and fixes. The
cloud re-review approved replay and cleanup ownership, then approved the
additional failed-open admission bound at runtime SHA-256
`6370c27e0c3a0e4bb57e8ca2e573db6925c40f9932c60485060be82ca2f82256`.
The focused cloud suite contains 48 passing tests. The combined exact-tree gate
is pending.

The model review found leases tied to Python wrappers rather than retained
native consumers, failed MT unload losing its retryable owner, and a remaining
raw-path synthetic-smoke consumer. All were fixed and approved on re-review;
`local/model_lease.py` SHA-256 is
`f1f2744b6366197d2bf735190c5cce557a182308eec4ac24969dafb3acc833ba`.
Failed bootstrap cleanup transfers its retained component to the existing
unavailable provider; CPU fallback cannot bypass incomplete cleanup.
The real SentencePiece probe also exposed pytest/SWIG interpreter shutdown
stderr. Its bounded clean-process probe now requires an exact success receipt
and empty stderr without changing warning filters or the aggregate runner.
The focused parent-process rerun passed with zero stderr bytes; its test file
SHA-256 is `9c6871e654b69a1ff6ef1f619f0a6d7dd8e236bb282e301008ad4feed836d1fb`.

The IPC/transport audit found one more startup rollback gap: exceptions during
native server construction/registration/bind leaked the already opened parent
directory descriptor. The transactional fix was approved at gRPC source SHA-256
`7e39a70b279f42357c1fbd10c27a876601a190f9bd5f8587bee3c2ed62231605`.
Six fault stages each run 100 times with unchanged FD/task baselines; repeated
rollback cancellation is also covered. The independent old-reproducer rerun
changed from 100 leaked FDs to zero.

The benchmark integration's BDD critic, separate BDD audit, and SRP review were
performed by the main agent before the new regression: not independent receipts.
They preserve the eval policy while requiring verified inputs, cleanup on partial
failure, fresh ownership for each duplex, and residency measurement before
shutdown. The final architect audit must review the actual integrated surface.

Focused green checks are intermediate evidence, not a substitute for the full
manifest-bound validation. The frozen collection contains 1400 nodes: Bun 15,
pytest 732, Rust 383 and unittest 270. Exactly 24 external-prerequisite skips remain
declared (0/1/1/22), leaving 1376 expected passes. The manifest SHA-256 is
`c300c0312109229f73cded92ff8992562e012896319e917ff44f00b04a3ef394`.
Full-gate log hash, quantitative source delta and final exact-tree review
receipts belong to the external close packet, avoiding a self-referential tree
hash inside the reviewed tree.

## Remaining release proof

- Native model snapshot RAM/load cost and CUDA/Piper compatibility on the target
  machine; no heavy model load has been started during this stage.
- Live OpenAI translation conformance and credential-bearing resource retirement.
  Loopback/fake tests do not certify this capability.
- B2-F remain open. No merge, production restart, tag, or release has occurred.
- Historical Task 7 round-trip latency debt remains `5968 ms`,
  `fails_usable_limit`; this stage has not produced new live evidence to close it.
