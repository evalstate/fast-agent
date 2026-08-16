---
social:
  title: Connect with OAuth (MCP)
  tagline: Authenticate MCP servers with OAuth and store tokens securely.
  description: Authenticate MCP servers with OAuth and store tokens securely.
  alt: fast-agent social card — MCP OAuth
---

# Connect with OAuth (MCP)

**`fast-agent`** supports connecting to HTTP MCP Servers with OAuth:

- Uses PKCE and prints a clickable authorization link (no auto‑open).
- Persists tokens in the OS keychain (via keyring) by default; falls back to memory if no keychain is available.

## Requirements

- **`fast-agent`** 0.3.5 or above
- OS Keyring support for persistence (e.g. WinVaultKeyring, macOS Keyring, Secret Service Keyring)


```bash title="Install keyring on Ubuntu"
sudo apt-get install gnome-keyring seahorse
```

## Server, endpoint, and OAuth resource

- A **server** is the configured name under `mcp.servers`, such as `myserver`.
- Its **endpoint** is the exact HTTP or SSE URL fast-agent connects to.
- Its **OAuth resource** is the normalized URL used to locate a stored credential.
- The resource is derived by removing a trailing `/mcp` or `/sse` and ignoring
  query/fragment components.
- Multiple configured server names can share one OAuth resource and credential.
- Renaming a configured server does not move its credential; changing its endpoint can.

## Minimal Config

OAuth is on by default for HTTP/SSE servers. Per‑server configuration:

```
mcp:
  servers:
    myserver:
      transport: http                    # (optional, defaults to http) or sse
      url: http://localhost:8001/mcp     # use /sse for SSE
      auth:
        oauth: true                      # default true
        persist: keyring                 # default keyring; use memory to disable persistence
        redirect_port: 3030              # default 3030
        redirect_path: /callback         # default /callback
        # scope: "user"                  # optional (server defaults used if omitted)
        # client_metadata_url: "https://example.com/client.json"
```

Notes:

- Scope is omitted by default. If a server requires a specific scope, set `auth.scope` (string or list).
- Use `auth.client_metadata_url` when a server supports Client ID Metadata Document (CIMD)
  registration and requires a URL-based client ID. The URL must be HTTPS and include a non-root path.
- STDIO servers do not use OAuth. They remain visible in `auth mcp list` with
  authentication shown as `not-applicable`.

## Keychain Persistence

- Default: tokens go to your OS keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service/KWallet).
- If a keychain backend is not available, tokens are kept in memory for the session (no disk writes).
- Proactive `auth mcp login` requires a writable keychain because an in-memory
  credential would disappear as soon as that command exits. Servers configured
  with `auth.persist: memory` authenticate only in the process that connects.
- Linux: ensure a Secret Service (gnome‑keyring) or KWallet is installed and running if you want persistence.

## CLI Quick Reference

- Combined provider and MCP overview:
  - `fast-agent auth`
  - `fast-agent auth --json`

- List configured MCP servers and effective auth modes:
  - `fast-agent auth mcp list`
  - `fast-agent auth mcp list --json`
  - `fast-agent auth mcp show myserver`

- List indexed local MCP OAuth credentials, including orphaned resources:
  - `fast-agent auth mcp credentials`
  - `fast-agent auth mcp credentials --json`

- Proactive login:
  - Configured server: `fast-agent auth mcp login myserver`
  - Exact ad-hoc HTTP endpoint:
    `fast-agent auth mcp login --endpoint https://example.com/custom/mcp`
  - Exact ad-hoc SSE endpoint:
    `fast-agent auth mcp login --endpoint https://example.com/events --transport sse`
  - Login waits up to five minutes by default. Use `--timeout <seconds>` when
    the authorization flow or server initialization needs longer.

- Forget local OAuth tokens and client registration:
  - Configured server: `fast-agent auth mcp forget myserver`
  - OAuth resource:
    `fast-agent auth mcp forget --resource https://example.com`
  - All indexed MCP credentials:
    `fast-agent auth mcp forget --all --yes`

Positional MCP values always mean configured server names. Ad-hoc endpoints are
accepted only through `--endpoint`, and fast-agent uses the supplied URL exactly.
It does not append `/mcp` or `/sse`.

- Check full app config (includes server OAuth flags and token presence):
  - `fast-agent check`

`fast-agent go`, `fast-agent serve`, and `fast-agent acp` also accept
`--client-metadata-url` for ad hoc URL-based server connections. The
`FAST_AGENT_OAUTH_CLIENT_METADATA_URL` environment variable can set a process-wide default; set it
to an empty value to disable the built-in default.

## Typical Workflows

- Connect normally; authenticate on demand
  - `fast-agent --url "https://huggingface.co/mcp?login"`
  - When a server requires OAuth, the CLI prints a clickable link.
  - A local callback server (`http://localhost:3030/callback`) captures the code; if the port is blocked, you’ll be prompted to paste the callback URL.

- Proactive login (no agent session needed)
  - `fast-agent auth mcp login myserver`
  - Or use an exact ad-hoc endpoint:
    `fast-agent auth mcp login --endpoint https://example-server.modelcontextprotocol.io/mcp`
  - Complete the link flow once; tokens will be reused next time.

- Inspect and forget a stored credential
  - `fast-agent auth mcp show myserver`
  - `fast-agent auth mcp credentials`
  - `fast-agent auth mcp forget myserver`

When multiple configured servers share one OAuth resource, `forget` lists every
affected server before asking for confirmation. It removes only local tokens and
client registration; server configuration and runtime connections are unchanged.

Before 0.10, a client registration created without a completed token was not
indexed. fast-agent backfills those records when the resource is still
configured. If the server was also removed, use `forget --resource <exact-url>`
when the resource URL is known; generic OS keyring APIs cannot enumerate an
unindexed historical username.

## Troubleshooting

- Immediate 401 with no link
  - Ensure you are running the updated CLI (editable install or latest tool).
  - Some servers require explicit scope; add `auth.scope` to that server in `fast-agent.yaml`.

- Link opens but no callback received
  - Confirm `http://localhost:3030/callback` is reachable (firewall/port in use).
  - If blocked, paste the returned callback URL when prompted in the terminal.

- Keychain not persisting tokens (Linux)
  - Install and run a Secret Service (gnome‑keyring) or KWallet.
  - Otherwise, tokens are in-memory only.

- Authorization header conflicts
  - When OAuth is enabled on a server, fast‑agent removes any preconfigured `Authorization`/`X‑HF‑Authorization` headers for that server’s transport so OAuth can proceed cleanly.

- STDIO shows `not-applicable`
  - Expected; the server remains visible for configuration completeness, but
    STDIO transport does not use OAuth.

## Hosting fast-agent MCP on Hugging Face

For the full hosted-server guide, see
[Host MCP Servers on Hugging Face Spaces](huggingface-spaces.md).

When deploying `fast-agent serve --transport http` on Hugging Face infrastructure, set
`FAST_AGENT_SERVE_OAUTH=huggingface` to require Hugging Face bearer
authentication for HTTP MCP requests. The server accepts `Authorization: Bearer
<token>` and, when the ingress forwards it, `X-HF-Authorization: Bearer <token>`.
Both forms are validated against Hugging Face before MCP initialization,
`tools/list`, or `tools/call` can reach fast-agent tools.

Validated request tokens are stored in fast-agent request context while the tool
call runs. Hugging Face provider calls and MCP servers configured with
`auth.forward: huggingface` can then use the caller's token. If you leave
`FAST_AGENT_SERVE_OAUTH` unset, inbound HTTP requests are not gated by
fast-agent; any `HF_TOKEN` in the Space environment is treated as the server's
own credential. Use that mode only for trusted/private deployments or with
least-privilege service tokens.
