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
./scripts/translator-validate deterministic
```

## Checks Before A Pull Request

Run the smallest relevant check while developing. Before opening a pull
request, run the complete repository contract:

```bash
./scripts/translator-validate deterministic
```

The tracked manifest makes source-gate and collection drift fail explicitly.
Live Zoom, Meet, and Telegram checks remain separate acceptance evidence; CI
does not relabel a missing desktop session, physical device, credential, or
private model cache as a pass.

To verify an already provisioned model cache separately from the deterministic
suite, opt in explicitly and point at its operator-owned root:

```bash
TRANSLATOR_RUN_GPU_MODEL_CACHE_TEST=1 \
TRANSLATOR_MODEL_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/translator/models" \
  uv run --project sidecar pytest -q \
  sidecar/tests/test_model_manifest.py::test_repository_reused_assets_resolve_through_pinned_integrity_policy
```

## Security Rules

- Never commit `.env`, API keys, bearer tokens, debug captures, raw PCM, transcripts, or translations.
- Keep local provider as the default.
- Cloud providers must require explicit opt-in and must show that audio leaves the machine.
- Test fixtures may use synthetic marker strings, but real credentials and spoken user content do not belong in fixtures.

## Commit Style

Use concise commit messages. Existing planning suggests the `translator:` prefix for implementation work.
