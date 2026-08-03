# Codex OAuth Runtime Notes

Use this note for operational/runtime details that should not live in the public README.

## Auth file lookup

- Default fallback path for Codex CLI tokens is `~/.codex/auth.json`.
- `CODEX_AUTH_JSON_PATH` overrides the fallback path with an explicit file.
- `CODEX_HOME` is also supported; when set, the fallback file becomes `${CODEX_HOME}/auth.json`.

## Precedence

- Fast-agent-owned credentials use the OS keyring when writable and otherwise
  fall back to `~/.fast-agent/auth.json`; these take precedence over external
  Codex CLI credentials.
- `CODEX_AUTH_JSON_PATH`, `CODEX_HOME`, and an existing `~/.codex/auth.json`
  remain interoperable, read-only fallback sources.
- `FAST_AGENT_AUTH_FILE` is an authoritative portable provider store and is
  used by exported credentials and Harbor.

## Persistence

- Codex CLI auth files are never modified or deleted by fast-agent.
- Login and refreshed tokens are persisted in the fast-agent-owned credential
  store, even when the original token came from a Codex CLI auth file.
- Provider exports retain refresh tokens. Newly created credential files use
  mode `0600` on Unix; overwriting an existing file preserves its permissions.

## Intended use

- Use `fast-agent auth provider export codex codex.auth.json` and
  `FAST_AGENT_AUTH_FILE` for service runtimes and Harbor.
- Use `CODEX_AUTH_JSON_PATH` only to read a separate Codex CLI profile.
