# Isolated native AEC backend

`translator-aec-backend` owns one PipeWire filter and one installed SPA WebRTC
AEC instance. It accepts an already-connected PipeWire Unix socket on
`--pipewire-fd`, a full-duplex IPC Unix socket on `--ipc-fd`, and a nonzero
16-digit hexadecimal `--session-id`. It never discovers a default server.
The caller must isolate the server from physical devices and production
sockets before creating either connection.

The filter is inactive until `START`. Its node is named
`translator-aec-<session-id>` and has `raw` and `reference` input ports and a
`clean` output port. It does not autoconnect. All three ports require exactly
one live link to virtual peers. The helper captures raw/reference in the same
480-frame processing callback, calls the installed WebRTC AEC once, copies the
result to the `clean` graph output and to a bounded IPC queue. Only 48 kHz,
mono, F32 planar buffers are supported. Missing/corrupted/gap buffers,
unequal or discontinuous per-port header sequences, graph-clock changes, graph position
gaps, xrun changes, link removal, non-finite/clipped PCM, and queue overflow
invalidate the attempt. Valid input headers must carry the producer-authored,
nonzero session-derived sequence base plus the shared graph position divided
by the fixed 480-sample quantum on both ports. Initialized or recycled buffers with default/stale headers cannot be
admitted. This is a non-cryptographic contract for the isolated fixture, not
physical-source authentication; future acquisition sources must implement an
independently reviewed equivalent before product wiring.

The private supervisor activates the fixture before requesting helper START.
`ARMED` is sent only after the helper has accepted and processed its first
frame and queued the corresponding clean output; activation alone is not a
measurement receipt. PipeWire may still present a malformed first buffer, in
which case the attempt fails without `ARMED` or proof.
Producer and consumer independently derive the expected header sequence from
session identity and the shared graph position. This permits a valid source
frame to be prepared before the measurement begins without treating an
unconsumed pre-start frame as a dropped measured frame. After the first
accepted frame, every position and per-port sequence must advance by exactly
one quantum; xrun changes still invalidate the attempt.

## Wire protocol v1

All integers and F32 values are little-endian. Each message starts with a
`u32` body length. The body starts with `type:u8`, `version:u8=1`, and a zero
`reserved:u16`. Unknown types, versions, sizes, or reserved bits are errors.

| Type | Direction | Body bytes | Payload after common four bytes |
| --- | --- | ---: | --- |
| `HELLO=1` | helper to supervisor | 36 | `session:u64`, `generation:u64`, `node_id:u32`, `rate:u32=48000`, `quantum:u32=480`, `declared_latency_samples:u32=480` |
| `FRAME=2` | helper to supervisor | 5832 | `session:u64`, `generation:u64`, `sequence:u64`, `clock_id:u32`, `n_samples:u32=480`, `clock_position:u64`, `clock_duration:u64`, `xrun:u64`, `clock_rate_num:u32`, `clock_rate_denom:u32`, `node_id:u32`, then 480 interleaved `raw,reference,clean` F32 triples |
| `FATAL=3` | helper to supervisor | 32 | `session:u64`, `generation:u64`, `reason:u32`, `next_sequence:u64` |
| `STOP=4` | supervisor to helper | 4 | none |
| `START=5` | supervisor to helper | 4 | none; one-shot after `HELLO` and explicit linking |
| `LINKS=6` | helper to supervisor | 72 | `session:u64`, `generation:u64`, `node_id:u32`, then three `(local_port:u32, link_id:u32, peer_node:u32, peer_port:u32)` tuples in raw/reference/clean order |
| `ARMED=7` | helper to supervisor | 24 | `session:u64`, `generation:u64`, `node_id:u32`; sent after the first accepted DSP frame has been queued |

`LINKS` and `ARMED` precede the first `FRAME` on IPC. `ARMED` means the private
filter accepted and processed its first paired frame; it is not proof of
physical acoustic safety. The supervisor activates the fixture before
requesting `START`, then waits for exact-session `ARMED` before accepting
`FRAME`. The supervisor must confirm link identities
against the private server, reject any missing/reordered frame or fatal event,
and enforce its own deadlines and cleanup. A `FRAME` graph position is a
PipeWire graph tick, not an attested hardware sample clock. The SPA
`position.state` is a transport-timeline field and may remain STOPPED during
active private graph processing; frame admission instead requires exact
48-kHz/480-frame cycles, a stable clock ID/rate, consecutive positions,
unchanged xrun, valid per-port headers and attested links. This binary emits
no calibration proof or product admission decision. A graph on virtual nodes
is always non-admissible for physical AEC calibration.

Fatal reason codes are `1` graph state, `2` clock, `3` gap/reset, `4` buffer,
`5` PCM, `6` AEC, `7` queue, `8` IPC, `9` ABI, and `10` link identity. The
transport queue holds at most 20 ten-millisecond blocks (200 ms, about
115 KiB PCM). The helper does not write PCM to logs or files. Its bounded diagnostic
`AEC_PREVALID quarantined=N empty=M sentinel=K` counts both distinct
premeasurement cases. After `START` but before the first accepted frame, the helper
may mute/recycle only (1) both input buffers absent while clean output is
writable, or (2) both exact zero-PCM GAP|CORRUPTED sequence-zero sentinels.
Neither case calls AEC, emits a `FRAME`, or advances sample/sequence state.
A single missing input, a malformed buffer, an authored sequence other than
the expected first one, or any post-first-frame gap remains fatal. The
supervisor enforces a four-second absolute no-progress deadline.

The build requires PipeWire headers at exactly 1.0.5 and the locally installed
`libspa-aec-webrtc.so`; `build.rs` rejects an unreviewed ABI. The isolated
runner must hash and pin the actual shared object. The plugin advertises
`480/48000` latency and requires ten-millisecond multiples. On PipeWire 1.0.5
its internal WebRTC processing errors may be logged while `run` returns zero,
so the isolated DSP gate must use independent far-end and near-end controls;
the return code alone is not proof of acoustic quality. This pinned helper
sets `webrtc.noise_suppression=false` and `webrtc.high_pass_filter=false`.
With high-pass enabled, the pinned near-speech control failed the fixed
waveform floor even after a best-lag diagnostic; disabling it preserved near
speech without changing the ERLE or negative-control thresholds.

`translator-aec-fixture` is a separate deterministic virtual generator. It
accepts connected inherited `--pipewire-fd` and `--control-fd`,
`--session-id`, and optional
`--mode far-only|near-only|wrong-reference|seq-skew|meta-gap|default-header|prevalid-once`.
Its node `translator-aec-fixture-<session-id>` exposes `raw` and `reference`
output ports. After connecting its inactive filter and installing its control
source, it sends exactly one `R` READY byte to the supervisor. A single `S`
byte activates the filter after explicit linking, and a successful activation
returns one `A` acknowledgment byte. `X` or control EOF stops it. Duplicate
START or unknown bytes fail closed. `SIGTERM` is only a teardown
fallback. `far-only` supplies
a fixed digital echo of deterministic reference noise; `near-only` supplies a
nonzero triangular near signal with zero reference. In `wrong-reference`, raw
contains the same echo while the reference is independent deterministic noise.
`seq-skew` deliberately shifts the reference SPA header sequence by one while
both per-port sequences remain individually continuous. The fixture marks
pre-valid buffers GAP|CORRUPTED with sequence zero; only a clock-valid,
activated pair of written PCM buffers gets the session-derived sequence.
`meta-gap` leaves generated PCM marked GAP|CORRUPTED and `default-header`
leaves it with zero sequence to test the backend admission boundary.
`prevalid-once` emits one exact zero-PCM GAP|CORRUPTED sentinel followed by
normal authored frames; the helper may recycle/mute only such a sentinel
before the first accepted frame and must emit no AEC `FRAME` for it. All
other invalid input, including a stale authored sequence, fails closed.
The fixture is demand-driven through its linked consumer, not always-processing
on its own. For speech controls,
`--mode speech-far|speech-near|speech-wrong-reference` additionally requires
`--far-pcm-fd` and `--near-pcm-fd`: inherited sealed memfds containing 45-60
seconds of finite, unclipped, 48-kHz mono F32LE samples, in 480-frame blocks.
Both must have WRITE/GROW/SHRINK/SEAL seals and are preloaded before the RT
graph starts. `speech-far` gives the A render reference and its fixed 240/480
sample digital echo as raw; `speech-near` gives B as raw and zero reference;
`speech-wrong-reference` gives raw echo A and independent reference B.
The runner fixes FLEURS source IDs, source/prepared SHA-256, conversion recipe
and version in its receipt. FLEURS is CC-BY-4.0: Conneau et al., 2022,
[FLEURS dataset card](https://huggingface.co/datasets/google/fleurs/blob/main/README.md).
The fixture must never be loaded into the backend process or interpreted as
physical audio evidence.

`translator-aec-witness` is an independent virtual sink with a single `clean`
input port and no output. It accepts connected inherited `--pipewire-fd` and
`--output-fd` sockets and `--session-id`; it never discovers a server. It sends
`R` READY after connecting inactive, receives `S` to activate, sends `A`
after activation, and stops on `X` or EOF. The helper's `clean` output must
have this witness as its sole linked consumer. After the first valid input,
missing or corrupt SPA buffers, sequence/clock gaps, nonfinite PCM, or queue
failure are fatal. The witness transports exact bytes consumed from PipeWire
through a bounded 20-block RT-to-writer queue, never logging PCM.

Witness records are fixed 1945 bytes: `W:u8`, `seq:u64le`, `clock_id:u32le`,
`clock_position:u64le`, `n_samples:u32le=480`, and 480 raw F32LE values.
Fatal records are 13 bytes: `F:u8`, `reason:u32le`, `next_sequence:u64le`.
The supervisor must match witnessed clean bytes against the helper's IPC
`FRAME.clean` by sequence and the same PipeWire graph position, and attest
the actual helper-to-witness link in the private server. The declared 480-sample
WebRTC algorithmic latency is not an extra PipeWire graph-position offset. Matching bytes proves a
virtual output boundary only; it is not physical acoustic evidence.
