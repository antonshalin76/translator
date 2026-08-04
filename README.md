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
cd /home/anton/Source/translator

cargo test --workspace

cd sidecar
uv sync --locked --all-groups
uv run pytest

cd ../apps/translator-ui
bun install --frozen-lockfile
bun test src/*.test.ts
bun run build
```

## Desktop Service

Build release binaries when needed:

```bash
cd /home/anton/Source/translator
cargo build --release -p translator-daemon
cd apps/translator-ui
bun run tauri build --no-bundle
```

Install and control the user service:

```bash
cd /home/anton/Source/translator
./scripts/translator-desktop install
translator up
translator status
translator logs 120
translator down
```

The `translator` command is installed into `~/.local/bin` and works from any directory after install.

## Configuration

Local provider mode does not require cloud credentials. For OpenAI provider testing, keep credentials outside git and expose `OPENAI_API_KEY` only to the local process or user-level service environment. See `.env.example` for non-secret variable names and optional local overrides.

Cloud provider use remains opt-in in the UI/API. Selecting a cloud provider marks that audio leaves the machine.

## Validation

Useful local checks:

```bash
cargo fmt --all -- --check
cargo test --workspace
python3 -m unittest tests.test_task1_boundaries tests.test_task8_ui_controls tests.test_task9_desktop_run_mode
(cd sidecar && uv run pytest)
(cd apps/translator-ui && bun test src/*.test.ts && bun run build)
```

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
  --asr-model faster-whisper-small,qwen3-asr-0.6b-hf
```

The ASR-only probe can execute current faster-whisper models through the pinned
local manifest. Qwen3-ASR is wired as an optional Transformers runtime candidate:
if `torch`/`transformers` or model weights are unavailable, the report marks that
candidate as `skipped` instead of failing the whole benchmark. GigaAM, Parakeet,
Kokoro, Silero, and Qwen3-TTS are currently tracked in the quality matrix for
controlled adapter work and are not live defaults.

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
