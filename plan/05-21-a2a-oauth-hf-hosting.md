# A2A OAuth and Hugging Face Hosting Goal

Status: first bearer/Hugging Face pass implemented.

## Goal

Add OAuth authentication for fast-agent A2A servers and clients, with Hugging
Face Spaces as a first-class hosted deployment target.

The key requirement is credential pass-through: when an A2A server is hosted on
Hugging Face, the caller's OAuth/bearer credential must be available inside the
fast-agent request context so Hugging Face Inference Provider models, the
Hugging Face MCP server, and Hugging Face tools can act using that caller
credential.

## Outcomes

- [x] A2A servers can enforce bearer authentication for `JSONRPC` and
  `HTTP+JSON` endpoints.
- [x] Hugging Face Spaces deployment supports `X-HF-Authorization` and standard
  `Authorization` bearer headers, matching the existing MCP behavior.
- [x] The request bearer token is written to `request_bearer_token` while the
  fast-agent agent handles an A2A request.
- [x] Served A2A AgentCards advertise security metadata through A2A
  `security_schemes`, `security_requirements`, and per-skill security
  requirements.
- [x] A2A clients can send static bearer/HF tokens, including automatic
  Hugging Face token headers for Hugging Face URLs.
- [ ] A2A clients can use the existing fast-agent OAuth browser flow where the
  remote AgentCard advertises OAuth or OIDC.
- [x] Documentation includes an A2A "Host on HF" page with Space setup,
  environment variables, OAuth behavior, AgentCard security metadata, and
  inference-provider usage.

## Existing Pieces to Reuse

- `fast_agent.mcp.auth.middleware.HFAuthHeaderMiddleware`
  - Normalizes `X-HF-Authorization` to `Authorization`.
- `fast_agent.mcp.auth.context.request_bearer_token`
  - Request-scoped token context already consumed by provider key resolution.
- `fast_agent.mcp.auth.presence.PresenceTokenVerifier`
  - Useful model for the initial server-side "present bearer token" check.
- `fast_agent.mcp.hf_auth.add_hf_auth_header`
  - Adds Hugging Face token headers for `hf.co`, `huggingface.co`, and
    `*.hf.space`.
- `fast_agent.mcp.oauth_client`
  - Existing OAuth login, callback, keyring, and client metadata machinery.
- A2A SDK `AuthInterceptor`, `CredentialService`, `ClientCallContext`, and
  AgentCard security scheme types.

## Proposed Server Design

1. Add A2A serve auth settings:
   - initially reuse `FAST_AGENT_SERVE_OAUTH=huggingface`;
   - reuse `FAST_AGENT_OAUTH_SCOPES`;
   - reuse `FAST_AGENT_OAUTH_RESOURCE_URL`;
   - later expose CLI/config fields if needed.
2. Wrap `AgentA2AServer.asgi_app()` with auth middleware when auth is enabled:
   - public AgentCard route remains reachable;
   - `/a2a/jsonrpc` and `/a2a/rest` require bearer auth;
   - missing/invalid auth returns `401` with `WWW-Authenticate`.
3. Normalize Hugging Face Space headers:
   - accept `Authorization: Bearer ...`;
   - accept `X-HF-Authorization: Bearer ...` and copy to `Authorization`.
4. Propagate credentials:
   - extract the bearer token from request headers/scope;
   - set `request_bearer_token` around the agent `generate(...)` call;
   - reset the context variable after the request.
5. Advertise security in the AgentCard:
   - use `HTTPAuthSecurityScheme(scheme="bearer")` for the first pass;
   - add `OAuth2SecurityScheme` or `OpenIdConnectSecurityScheme` once we have
     provider metadata details that A2A clients can use reliably.

Implemented first pass:

- `FAST_AGENT_SERVE_OAUTH=huggingface` enables A2A bearer auth.
- AgentCard discovery stays public.
- `/a2a/jsonrpc` and `/a2a/rest` require a bearer token.
- `X-HF-Authorization` is accepted and normalized for Hugging Face Spaces.
- The bearer token is available through `request_bearer_token` while the
  fast-agent agent runs.
- The public AgentCard advertises an `hf_bearer` HTTP bearer security scheme.

## Proposed Client Design

1. Static token support:
   - keep explicit `headers` on `A2AAgentConfig`;
   - add `auth`/`oauth` fields only if the UX needs parity with MCP cards;
   - automatically add HF token headers for Hugging Face URLs when no explicit
     auth header is present.
2. AgentCard-driven credential injection:
   - inspect `remote_card.security_schemes` and `security_requirements`;
   - use A2A SDK `AuthInterceptor` with a fast-agent `CredentialService`;
   - pass per-call `ClientCallContext` so the SDK transports receive
     `Authorization` or API key headers.
3. OAuth browser flow:
   - adapt `fast_agent.mcp.oauth_client` to an A2A-oriented server identity;
   - store tokens in keyring using a distinct service or identity prefix;
   - emit OAuth events through CLI/TUI surfaces similarly to `/mcp connect`.

Implemented first pass:

- A2A explicit `headers` remain supported.
- A2A clients automatically apply Hugging Face token headers through
  `add_hf_auth_header(...)` for Hugging Face URLs when no explicit auth header
  is configured.
- Browser OAuth remains open.

## Testing

- Server auth:
  - public AgentCard route is accessible;
  - A2A routes reject missing bearer tokens;
  - `Authorization` reaches `request_bearer_token`;
  - `X-HF-Authorization` reaches `request_bearer_token` on HF mode.
- AgentCard metadata:
  - auth-enabled A2A server advertises expected security schemes and
    requirements.
- Client auth:
  - explicit A2A headers are sent;
  - HF token auto-header logic applies only to Hugging Face URLs;
  - existing explicit auth headers win over auto HF auth.
- Inference pass-through:
  - deterministic test agent reads `request_bearer_token`;
  - provider-key-manager behavior can use the request token for Hugging Face.

## Open Questions

- Should A2A server auth configuration live only in shared serve environment
  variables, or should `fast-agent serve a2a` expose first-class `--oauth`
  flags?
- Should the first client OAuth pass be generic OAuth/OIDC, or should it focus
  on Hugging Face Spaces first?
- Should authenticated extended AgentCards be implemented as part of this work,
  or should public cards advertise enough security metadata for the first pass?
