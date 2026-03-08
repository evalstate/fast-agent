# Codex OAuth Runtime Notes

Use this note for operational/runtime details that should not live in the public README.

## Auth file lookup

- Default fallback path for Codex CLI tokens is `~/.codex/auth.json`.
- `CODEX_AUTH_JSON_PATH` overrides the fallback path with an explicit file.
- `CODEX_HOME` is also supported; when set, the fallback file becomes `${CODEX_HOME}/auth.json`.

## Precedence

- Without an override, Fast Agent keeps the existing precedence: keyring first, then `~/.codex/auth.json`.
- When `CODEX_AUTH_JSON_PATH` or `CODEX_HOME` is set, Fast Agent prefers the overridden auth file before keyring so a service can be pinned to a local profile.

## Persistence

- When an override path is active, refreshed Codex OAuth tokens are written back into the overridden auth file.
- This keeps long-running service profiles stable across restarts without depending on the global `~/.codex/auth.json`.

## Intended use

- Prefer `CODEX_AUTH_JSON_PATH` for service runtimes that need a repo-local or deployment-local Codex profile.
- Prefer the environment variable over adding more CLI flags to `fast-agent serve`.
