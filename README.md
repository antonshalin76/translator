# Translator

Local Linux desktop service for full-duplex Russian/English speech translation in live calls. It uses PipeWire/PulseAudio routing, a Rust daemon, Python speech providers, and a Tauri desktop UI to translate both microphone and remote-participant audio for Zoom, Google Meet, Telegram Desktop, and similar apps.

## Status

Translator is an MVP-oriented workstation project for Ubuntu/PipeWire desktops. The current codebase includes the daemon, local sidecar provider path, OpenAI provider adapter behind explicit cloud opt-in, desktop UI controls, user-level systemd lifecycle, synthetic checks, and local live-smoke evidence for Zoom, Google Meet, and Telegram Desktop.

It is not a general cross-platform release yet. The default target is Linux with PipeWire through PulseAudio compatibility.

## Features

- Duplex Ru <-> En translation with independent microphone and speaker channels.
- Per-channel enable/disable controls, direction selection, and separate original/translation volume mixing.
- Local provider as the default path; OpenAI Realtime Translation is available only after explicit cloud opt-in.
- PipeWire/Pulse virtual endpoints for `Translator_Virtual_Mic`, `Translator_Mic_Out`, and `Translator_Remote_In`.
- App routing watcher for call-like streams from Zoom, Telegram Desktop, and browser meetings.
- Tauri tray/status UI plus a user-level `translator` lifecycle command.
- Privacy-safe defaults: no stored audio, transcripts, or translations unless debug capture/text is explicitly enabled.

## Architecture

```text
Tauri desktop UI
  -> localhost control API
  -> Rust translator-daemon
       -> PipeWire/Pulse endpoint and route management
       -> Python translator-sidecar over authenticated local IPC
            -> local ASR / MT / TTS provider
            -> optional OpenAI realtime provider
```

The daemon owns audio routing and device selection. The Python sidecar owns provider inference. The UI is a control plane and does not receive raw PCM frames or provider credentials.

## Requirements

- Ubuntu 24.04 or a comparable Linux desktop with PipeWire and PulseAudio compatibility.
- `pactl`, `pw-link`, and `wpctl` available in the user session.
- Rust 1.88+.
- Python 3.12 and `uv`.
- Bun for the Tauri/Vite frontend.
- Headphones for normal duplex use. Open-speaker mode is guarded by AEC validation.

## Development Setup

```bash
cd translator

./scripts/translator-validate deterministic
```

The deterministic entrypoint runs the complete Rust, Python, UI, schema,
systemd, supply-chain, and publication gates used by CI. It also verifies the
tracked test manifest and the exact external-prerequisite skip allowlist.

## Desktop Service

Build release binaries when needed:

```bash
cargo build --release -p translator-daemon
cd apps/translator-ui
bun run tauri build --no-bundle
```

Install and control the user service:

```bash
./scripts/translator-desktop install
translator up
translator status
translator logs 120
translator down
```

The `translator` command is installed into `~/.local/bin` and works from any directory after install.

## Configuration

Local provider mode does not require cloud credentials. The desktop launcher
resolves systemd's user configuration root and creates `%E/translator` with
mode `0700` and an empty `environment` file with mode `0600`. On every
`install`, `up`, `start`, and `restart`, it refuses to continue unless the
directory belongs to the current user and the environment file is a regular,
non-symlink, single-link file owned by that user with those exact modes. The
resolved configuration root must also belong to the current user and must not
be writable by group or others. The installed unit always enters the same
wrapper before a daemon start, including automatic restarts. Systemd never
loads the file itself.

Configure the user service without putting values in the repository or shell
command line:

```bash
./scripts/translator-desktop install
service_environment="$(systemd-path user-configuration)/translator/environment"
editor -- "$service_environment"
chmod 600 -- "$service_environment"
translator restart
```

Use one UTF-8 `KEY=value` entry per line. Values are literal: shell quoting,
expansion, and `export` syntax are not evaluated. Duplicate, malformed, and
unknown keys fail closed. The allowlist covers `OPENAI_API_KEY` and the
daemon/sidecar override names documented in `.env.example`; UI-only and
daemon-managed names are rejected. Do not print or `source` this file. If the
launcher rejects an existing path, verify its owner, type, link count, and mode
before repairing or replacing it; do not follow a symlink.

The wrapper opens the file without following symlinks, validates the held file
descriptor, reads it with a bounded parser, and passes accepted values to the
daemon through `execve`. Those values still become process environment and can
be inherited by child processes or exposed through process inspection. Prefer
local provider mode where possible; credential-file support is required before
claiming stronger protection for cloud credentials.

Cloud provider use remains opt-in in the UI/API. Selecting a cloud provider marks that audio leaves the machine.

The default VAD live window is tuned for low-latency calls: continuous speech is
forced to an `end_of_utterance` boundary around 6000 ms, and pauses after the
first 2500 ms can close the current chunk after about 120 ms of non-speech. For
experiments, override `TRANSLATOR_VAD_MIN_UTTERANCE_MS`,
`TRANSLATOR_VAD_MAX_UTTERANCE_MS`, and
`TRANSLATOR_VAD_ADAPTIVE_SILENCE_MS` in the service environment file, or export
them in the shell for a direct run. The sidecar also uses
`TRANSLATOR_CONTINUATION_TAIL_RMS` to distinguish a forced voiced chunk boundary
from a natural silent phrase ending before TTS renders punctuation.

CTranslate2/faster-whisper GPU execution requires CUDA 12 cuBLAS and cuDNN 9
runtime libraries. The supervised service discards the caller's loader search
path and admits only absolute CUDA directories owned by root or the service
user, with no group/other-write or set-ID mode on ancestry, contents, and
resolved symlink targets. Owner-write is allowed because root and the service
UID are the trust boundary. The only broader writable traversal exception is a
root-owned sticky ancestor such as `/tmp`; it never applies to the admitted
directory or a library. An absent portable default is ignored, while an invalid
explicit path or an unsafe existing default fails closed.
`TRANSLATOR_CUDA_LIBRARY_PATH` supplies the ordered operator directories; it
does not extend an ambient `LD_LIBRARY_PATH`.
Direct sidecar/debug runs apply the same Python-side directory and held-file
validation and serialize process-global configuration; failed preload restores
the prior loader environment. Only the desktop launcher/service wrapper is a
supported secure entrypoint because a directly invoked process has already
consumed its caller's loader environment. The direct helper proves the current
cuBLAS tree, not an arbitrary cuDNN 9 split-library layout; use the supervised
pre-exec path for the complete GPU runtime. CUDA 13 alone is not sufficient for
the current `ctranslate2==4.7.1` wheel because it does not provide
`libcublas.so.12`.

## Validation

Run the same deterministic gate locally and in hosted CI:

```bash
./scripts/translator-validate deterministic
```

Live provider, physical-audio, real-call, and human-review checks are separate
release evidence. A missing live prerequisite is reported as unavailable; it
is never converted into a deterministic pass.

Quality diagnostics:

```bash
# Show the approved local quality matrix, including Qwen3-ASR candidates.
./scripts/translator-podcast-quality-debug --list-candidates

# Full local provider path on RU/EN podcast or local audio segments.
./scripts/translator-podcast-quality-debug \
  --asr-model faster-whisper-small,faster-whisper-large-v3 \
  --tts-model piper-medium

# ASR-only probe on local mono s16le 16 kHz PCM.
./scripts/translator-asr-quality-debug \
  --audio output/sample.s16le \
  --language ru \
  --asr-model faster-whisper-small,faster-whisper-large-v3-turbo-ct2
```

The ASR-only probe can execute current faster-whisper models through the pinned
local manifest and can run the CT2 turbo candidate from Hugging Face cache when
available. Qwen3-ASR remains wired as an optional Transformers runtime
candidate, but current local measurements rejected it for live default use
because latency was above real time. GigaAM, Parakeet, Kokoro, Silero, and
Qwen3-TTS are tracked in the quality matrix for controlled adapter work and are
not live defaults.

Live application acceptance requires a running desktop audio session and real/simulated calls. Existing task scripts live under `scripts/`; local run evidence is intentionally ignored and should stay outside published git history.

## Privacy And Security

- The daemon binds the control API to loopback only.
- Control requests require a bearer token stored under the user runtime directory.
- Debug text and debug capture are separate explicit modes.
- Logs must not include spoken content, transcripts, translations, raw provider payloads, or API keys.
- `.env`, model caches, debug captures, runtime sockets, build outputs, and virtual environments are ignored by default.

## License

MIT. Copyright (c) 2026 Anton Shalin.

Maintainer: Anton Shalin <anton.shalin@gmail.com>.
