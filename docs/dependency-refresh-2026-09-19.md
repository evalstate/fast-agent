# Dependency refresh — 19 September 2026

Versions checked against PyPI release metadata; `uv.lock` refreshed, including compatible optional and transitive dependencies.

## Direct pinned upgrades

| Package | Previous | Updated |
| --- | --- | --- |
| fastapi | 0.136.3 | 0.141.1 |
| fastmcp-slim | 4.0.3 | 4.0.5 |
| mcp / mcp-types | 2.1.1 | 2.2.0 |
| pydantic-settings | 2.14.2 | 2.15.0 |
| pydantic | 2.13.4 | 2.13.5 |
| typer | 0.25.1 | 0.27.2 |
| opentelemetry-exporter-otlp-proto-http | 1.43.0 | 1.44.0 |
| opentelemetry-instrumentation-openai / anthropic | 0.62.1 | 0.62.3 |
| opentelemetry-instrumentation-google-genai | 1.0b1 | 1.1b1 |
| a2a-sdk | 1.1.0 | 1.1.4 |
| agent-client-protocol | 0.10.1 | 0.12.1 |
| tiktoken | 0.13.0 | 0.14.0 |
| starlette | 1.3.1 | 1.6.0 |
| ruff | 0.16.0 | 0.16.8 |
| ty | 0.0.65 | 0.0.82 |
| zensical | 0.0.44 | 0.0.62 |
| PMD CPD | 7.26.0 | 7.27.0 |

Rich 15.0.0, prompt-toolkit 3.0.53, OpenAI 3.16.2, Anthropic 1.7.0, Google GenAI 2.24.0, and Hugging Face Hub 1.32.0 were already current. Filelock 4.0.1 could not resolve with the development dependency graph (virtualenv's filelock <4 requirement); retain 3.29.1. ACP 1.0 release candidates were not selected. Google telemetry remains on its existing beta release track.

## Compatibility and simplification review

- [Typer migration](https://typer.tiangolo.com/tutorial/click/): use Typer's public exports rather than mixing its vendored types and exceptions with Click. Updated CLI tests exercise the change.
- [ACP releases](https://github.com/agentclientprotocol/python-sdk/releases): align positional parameter ordering and accepted MCP variants with the new interface; retain explicit handling of unsupported transport variants. Do not remove local tool-input tracking: the SDK still requires raw input to be supplied explicitly on progress updates.
- Rich: replace the manually maintained live-display substitute with a `Live` subclass. Prompt-toolkit callbacks now return its typed formatted-text objects. Neither library required a version bump.
- New ty diagnostics exposed exception-suppression typing in workflow spans, logger protocol parameter incompatibility, incomplete test doubles, and redundant conditions. Fix the shared contracts rather than adding ignores.
- [MCP 2.2](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.2.0) owns transport redirect policy: removed the redundant gateway `follow_redirects=True`. Installed SDK 2.2.0 explicitly sends with redirects disabled and follows only method-preserving, same-origin redirects (also allowing same-host HTTP→HTTPS upgrades on default ports; rejecting new URL userinfo). POST, GET/resumption, and DELETE cleanup use this policy. Gateway/SDK tests with mock HTTP I/O cover 307/308 POST and DELETE redirects, host/scheme/port rejection without credential forwarding, subsequent use of an established transport session, and response/client closure. Legacy SSE uses a separate SDK-owned client and is unchanged. OAuth requests inherit the explicit no-follow setting too: fast-agent overrides `async_auth_flow`, bypassing the SDK `RedirectAwareAuth` wrapper, so its discovery/registration/token redirects remain unfollowed, including same-origin redirects. This cleanup does not fix that pre-existing limitation or test live OAuth/browser flows, GET/SSE reconnection, or real connection-pool reuse. Cross-origin redirects and stricter OAuth issuer handling still warrant deployment checks.
- [FastMCP 4.0.5](https://github.com/PrefectHQ/fastmcp/releases/tag/v4.0.5) restores strict field validation. This is not a reason to remove harness boundary validation.
- [A2A 1.1.4](https://github.com/a2aproject/a2a-python/blob/v1.1.4/src/a2a/server/request_handlers/default_request_handler_v2.py) exposes handler `aclose()`: consider ASGI lifespan integration to drain SDK tasks before executor shutdown. This complements rather than replaces executor cleanup; defer to a lifecycle-focused change with active-stream cancellation tests.
- Google SDK typed response fields already permit further removal of defensive attribute lookups. Keep this separate from upgrades and retain optional-field handling and provider-boundary normalization.

## Quality checks

Updated Ruff formatting/lint and ty checks pass. PMD 7.27.0 reports exactly the seven existing approved structural duplicates; only baseline tool-version metadata changed, not finding hashes or acceptance scope. Final unit suite: **8,235 passed, 1 skipped**, with four deprecation warnings. A durable-process test race was fixed by waiting for child readiness before requesting stop. Targeted CLI, ACP, UI, workflow, and command tests were also run during migration. Live provider calls and optional-extra runtime combinations are not covered by dependency resolution alone.
