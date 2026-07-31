# Security Policy

## Supported Scope

Security reports are accepted for the current `main` branch and recent tagged releases after the first public release.

## Reporting

Send security reports to Anton Shalin at `anton.shalin@gmail.com`.

Do not include real meeting audio, transcripts, translation text, API keys, bearer tokens, private call links, or screenshots containing private data in a public issue. Use a private report channel for sensitive details.

## Project Security Model

- The control API is local-only and must bind to loopback addresses.
- Control requests use a bearer token stored in the user runtime directory.
- The Tauri frontend must not store provider credentials or raw PCM.
- Local provider is the default. Cloud provider sessions require explicit opt-in.
- Audio, transcripts, and translations are not persisted by default.
- Debug capture and debug text are explicit modes and must remain visibly separate from normal runtime.
- Model downloads must be recorded in `models/manifest.json` with source, revision, SHA-256, license, byte size, and cache path.

## Secret Handling

Use `.env` or a user-level service override outside the repository for local credentials. `.env` is ignored; `.env.example` documents variable names without values.

If a real key is ever committed or pasted into a public issue, rotate it immediately before continuing work.
