# Contributing

## Scope

Translator is split by ownership:

- Rust daemon: audio graph, routing, device selection, local API, latency policy, sidecar supervision.
- Python sidecar: local and cloud provider adapters, ASR/MT/TTS runtime, provider contracts.
- Tauri UI: tray/menu/status controls only.
- systemd user unit: logged-in user service lifecycle.

Keep changes inside the owner that already owns the behavior. Do not move provider credentials or raw audio into frontend state.

## Setup

```bash
cargo test --workspace

cd sidecar
uv sync --locked --all-groups
uv run pytest

cd ../apps/translator-ui
bun install --frozen-lockfile
bun test src/*.test.ts
bun run build
```

## Checks Before A Pull Request

Run the smallest relevant checks for your change, then broaden if you touched shared contracts, routing, provider transport, or UI state.

```bash
cargo fmt --all -- --check
cargo test --workspace
python3 -m unittest tests.test_task1_boundaries tests.test_task8_ui_controls tests.test_task9_desktop_run_mode
(cd sidecar && uv run pytest)
(cd apps/translator-ui && bun test src/*.test.ts && bun run build)
```

Live Zoom/Meet/Telegram checks are manual acceptance tests. Do not make CI depend on a live desktop session, physical devices, or private model caches.

## Security Rules

- Never commit `.env`, API keys, bearer tokens, debug captures, raw PCM, transcripts, or translations.
- Keep local provider as the default.
- Cloud providers must require explicit opt-in and must show that audio leaves the machine.
- Test fixtures may use synthetic marker strings, but real credentials and spoken user content do not belong in fixtures.

## Commit Style

Use concise commit messages. Existing planning suggests the `translator:` prefix for implementation work.
