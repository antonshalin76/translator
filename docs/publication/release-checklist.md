# Publication Checklist

Use this checklist before the first public GitHub push.

## Repository

- [ ] Initialize git in `/home/anton/Source/translator` if it is still not a repository.
- [ ] Create the GitHub repository under the intended account.
- [ ] Set the GitHub About description from `docs/publication/github-description.md`.
- [ ] Confirm the default branch name (`main` recommended).
- [ ] Confirm MIT license is acceptable for all first-party code.

## Files To Keep Out Of Git

- [ ] `.env` and `.env.*` except `.env.example`.
- [ ] `target/`.
- [ ] `apps/translator-ui/node_modules/`.
- [ ] `apps/translator-ui/dist/`.
- [ ] `sidecar/.venv/`.
- [ ] `.ruff_cache/`, `.pytest_cache/`, `__pycache__/`.
- [ ] `output/`, `debug-captures/`, runtime sockets, and local logs.
- [ ] Local model files under `models/` except `models/manifest.json`.

## Local Verification

```bash
cargo fmt --all -- --check
cargo test --workspace
python3 -m unittest tests.test_task1_boundaries tests.test_task8_ui_controls tests.test_task9_desktop_run_mode
(cd sidecar && uv run pytest)
(cd apps/translator-ui && bun test src/*.test.ts && bun run build)
```

## Secret Scan

```bash
rg --hidden -n -S \
  'sk-proj-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{20,}|BEGIN (RSA|OPENSSH|EC|PRIVATE) KEY' \
  -g '!target/**' \
  -g '!apps/translator-ui/node_modules/**' \
  -g '!apps/translator-ui/dist/**' \
  -g '!sidecar/.venv/**' \
  -g '!output/**' \
  -g '!.env.example' \
  .
```

The command should print no real credentials. Test fixtures may contain synthetic marker strings only.

## First Push

```bash
git init
git add .
git status --short
git commit -m "translator: prepare public repository"
git branch -M main
git remote add origin git@github.com:<owner>/<repo>.git
git push -u origin main
```
