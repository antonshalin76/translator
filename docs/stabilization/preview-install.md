# Translator private preview, 2026-10-02

This is a personal, host-specific Linux x86-64 preview, not a stable release.
It uses Whisper large-v3-turbo, Hy-MT2-1.8B Q4_K_M and four Piper medium voices.
Small is the existing ASR fallback. Hy has a measured MT-only development
advantage; full-chain accuracy, acoustic latency and reliability are unproved.
Model weights are for this owner's private installation, not redistribution.

Requirements: the existing Ubuntu/PipeWire desktop with `libpulse.so.0`
(`libpulse0`), Python 3.12, user systemd,
the installed Ollama llama-server and its CUDA12 backend, and NVIDIA driver.
The payload includes private, RECORD-verified cuDNN9/CUDA12 and NVRTC12 copies
for CTranslate2. Missing Hy CUDA fails closed, rather than changing models.
The package does not replace or repair host CUDA installations. Dependency
fingerprints alone are not proof of actual GPU inference.
Included vendor libraries are `nvidia-cudnn-cu12==9.10.2.21` and
`nvidia-cuda-nvrtc-cu12==12.8.93`, with their supplier metadata/licenses.

## Install

Verify the archive against its SHA256SUMS, then extract it to a permanent
private directory, preferably directly under `$HOME/.local/state` with the
archive's `translator-preview-20261002` directory name. Keep that directory:
the installed command refers to it. The pinned Piper native phonemizer rejects
some long installation paths; installation checks EN/RU phonemization in an
isolated subprocess before creating the preview command/unit and refuses a
nonworking payload. No speech is played or recorded during this check.
Do not run canonical Translator and preview simultaneously; virtual endpoint
names remain shared. Preview refuses a currently active canonical unit/API,
but this is not an atomic cross-service audio lock.

From the extracted short directory:

```bash
sha256sum -c SHA256SUMS --quiet
./scripts/translator-desktop --preview install
translator-preview up
```

Installation creates only the preview command, unit and private configuration.
It does not start/enable a service or create UI autostart. First installation
only: uninstall an existing preview before installing a different payload.
Production files and its configuration are not replaced.

The preview API uses loopback port 47682 and a separate private control token.
Runtime/state and graph journals are separated; audio endpoints are not.
The UI command launched by `up` points to this preview API/token location.
No audio, transcript or translation logging is enabled by this package.
The preview unit caps RAM at 8 GiB (6 GiB soft pressure), swap at 256 MiB,
CPU at two cores and tasks at 256; stopping it reaps its whole process group.
This user unit deliberately does not request a filesystem/user namespace:
unprivileged systemd mount sandboxing hides host root UID and conflicts with
the strict CUDA library owner checks. Those checks remain unchanged; the unit
retains NoNewPrivileges, environment sanitization and private directory modes.
Filesystem writes are not additionally confined by systemd in this preview.

## First real-call trial

Select `quality_first` in the existing UI. With physical headphones, leave both
directions enabled for RU-to-EN microphone and EN-to-RU incoming translation.
This preview deliberately has no AEC calibration backend: microphone/full-duplex
with speakers is rejected by the existing acoustic admission. It does not
establish that speaker-only translation or app routing has passed real-call E2E.

Some USB adapters report only an Analog port, even with headphones connected.
While translation is stopped, explicitly confirm the currently selected
microphone/output pair using the headphone button. Status then reports
`user_confirmed_headphones`, not driver-detected headphones or validated AEC.
Confirmation leaves the original port metadata unchanged, is not persisted,
and is revoked on discovery failure or observed device/port/availability changes.
Restarting the daemon requires confirmation again.
Pending cleanup or failure exposes Stop, not Start or a completed-stop status.
Headphone confirmation remains unavailable until the daemon reports stopped.
Do not confirm speakers; an Analog adapter cannot detect what is plugged into
its analog socket.
Stop translation before changing that physical connection or revoking the choice.

The two direction loops read capture while awaiting playback writes, with bounded
PCM queues. Shared GPU inference remains serialized and translation waits for
speech segmentation; full duplex does not mean token-streamed ASR/MT or zero
latency. A failed/partial playback cannot be reused before cleanup.
Volume controls display acknowledged daemon values, not a successful-looking
local snapshot before the request completes. A missing-original rejection
preserves the previous mix and running translation; its error remains visible until another
control command or the daemon reconnects. Status results obtained before a
control command cannot overwrite that command's acknowledgement.
An unknown physical mixer state still stops/quarantines translation for safety.

`Microphone original` forwards raw microphone audio into the outgoing virtual
microphone independently of translated speech. With the microphone enabled and
admitted headphones, Start prepares a pinned native Pulse capture/playback pair
whose playback volume is zero before its first frame. During translation, the
original slider changes this pair's gain without restarting either translation
direction. Leave it at 0% if only translated microphone speech should be sent.
Background device refresh cannot create a new microphone capture. A stopped
positive-volume command can prepare the pair only after fresh headphone checks.

The existing Stopped/Bypass policy sends originals at 100% and mutes translations;
it retains the desired translating gains for the next Start. Microphone-muted
bypass keeps the raw microphone at zero. A disabled microphone, revoked acoustic
admission or changed endpoint disconnects the pair before replacement. A missing
pair still rejects positive gain with
`microphone_original_unavailable`; it cannot fall back to another microphone.
Disabling microphone joins this pair before acknowledgement, preserves the
desired microphone gains and applies zero effective microphone gains. Incoming
translation and its volume remain independent. Re-enabling microphone requires
an explicit admitted command and prepares a fresh pair at zero. Unverified gain
readback cancels raw forwarding; failed cleanup retains custody and requires
explicit recovery before another pair can be created.

Cold Start uses the existing runtime's bounded 130-second readiness budget;
Stop keeps its separate eight-second cleanup budget. Models are not retained
after Stop, so another Start is cold again. Before opening PCM after readiness,
the packaged daemon rechecks physical identities/ports and acoustic admission.
An in-flight cold Start currently serializes other controls; stopping the
preview systemd unit remains the bounded process-group cancellation boundary.

The private `translator-preview/environment` file under the systemd user
configuration root accepts the same guarded keys as the ordinary service,
including `TRANSLATOR_MT_MODEL_ID`. NLLB is not included in this preview model
payload; selecting it needs a separately verified cache and explicit cache root.
Do not print/source environment files or put tokens on command lines.

## Stop and rollback

```bash
translator-preview down
translator-preview uninstall
```

These commands target the preview unit and its journal, not the canonical
service. The extracted payload, models and private configuration are retained.
Canonical Translator can then be started explicitly with its original command.
Uninstall does not remove weights or call recordings.

Source/physical failure receipts remain retained. This preview does not close
Task7, prove AEC, pass the independent holdout/paired evaluation, or authorize
stable merge/publication. Automated package checks and actual translation
behavior are distinct evidence.

## Earlier installed-preview verification, 2026-10-02

The earlier installed preview completed an authenticated cold HTTP Start in 9,828 ms
and Stop in 1,111 ms. Both enabled directions remained Running across five
status polls. Unauthorized confirmation was rejected, revocation blocked
Start, and daemon restart revoked the ephemeral headphone choice. That trial
ended Stopped with both directions enabled, local processing and debug
text/capture disabled. The earlier 503 spawn failure and 4,001-ms Start timeout
remain retained; successful startup does not erase them.

The packaged Turbo/Hy/Piper provider completed eight saved-audio attempts,
including two concurrent opposite-language pairs. Their nonzero input overlap
was 3,019 ms and 3,014 ms; each short leg emitted translated PCM before the
opposite leg's later nonzero speech frame. No ASR fallback/degraded result was
reported; the child, process group and socket were cleaned up. These exposed
development fixtures establish inference concurrency, not independent quality
or physical acoustic latency. Runtime regressions separately verify capture
and provider acceptance during a held playback write, payload order, queue
overflow, capture EOF and cancellation.

The earlier embedded native UI passed all five isolated checks: asset loading,
disconnected admission, three viewports, keyboard focus across polling and
negative focus-identity cases. Window/driver cleanup completed. This is not
real-call app routing, a soak result or a stable-release approval.

## Rebuild scope

`scripts/translator-preview-package` packages already-built release binaries,
the locked non-editable Python environment, the seven selected pinned models
and the two pinned CUDA distributions. It refuses an existing output directory,
an output inside the checkout, unstaged changes or a mismatching staged tree.
Model copies use the existing manifest's retained descriptors and size/SHA-256
checks; library copies are checked against supplier RECORD. Copied ELF modes
are normalized without changing build outputs. This is host-specific packaging,
not an installable distribution for arbitrary operating systems.

Build the frontend first (`bun run build` in `apps/translator-ui`). The native
Pulse dependency needs the existing `libpulse.so.0` runtime; `libpulse-dev` and
pkg-config provide SDK discovery, with the pinned binding's Linux SONAME fallback
available when the SDK is absent. Raw Cargo
release builds must enable `--features translator-ui/custom-protocol`; the
Tauri CLI enables this feature for `tauri build` automatically. A release build
without it is rejected, because it would open the Vite development URL instead
of embedding the frontend. The packager checks the actual UI binary with
`--check-bundled-ui` before creating output or copying models. This check needs
no display, daemon connection or audio. It does not replace the separate native
window test or translation acceptance.

The volume correction adds three connected checks to that native window suite:
rejection without optimistic volume state, successful translated-volume change
with a held older native status result, and separation of unavailable AEC from
command errors. All eight checks pass against the rebuilt UI using a private
authenticated HTTP fixture, with no audio or model calls and complete cleanup.
This does not validate a real microphone-original stream or live call quality.

## R9 original-microphone verification

The actual native transport and daemon graph/mix owners passed an isolated
PulseAudio PCM regression using a synthetic microphone signal and an independent
output observer. Every settled 100% frame preserved the input RMS of 5655.994;
35% produced RMS 242.571, matching Pulse's nonlinear amplitude mapping, and 0%
produced exact zero peaks. Observation excludes at most 240 ms of gain settling.
Translated stream identities/gains stayed unchanged across original-gain changes.

The same regression covers silence before the first admitted frame, fresh-zero
incoming playback admission with microphone disabled, cancellation after a
falsely acknowledged zero write, joined cleanup/recovery, rejected stream moves,
endpoint removal/recreation and silent replacement without old PCM replay.
Separate private checks cover cancellation at native startup phases and
first-frame translated playback admission. The fixture finished with no streams;
its owned Pulse process exited and its socket became unreachable.

This proves the automated original-microphone capability on a private virtual
graph, not physical headset/call quality, a long soak, AEC or stable release.
