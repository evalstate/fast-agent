# Dependency review — 17 November 2026

> **Post-review merge decision:** main's `google-genai==3.4.0` is retained for
> current Gemini model support. This supersedes the Google 3.x deferral and
> selected version below; the inventory and validation results describe the
> original dependency review, not the merged tree. Google SDK auto-instrumentation
> is unavailable with the current OpenTelemetry instrumentor (`google-genai<3`);
> this is an accepted limitation, not a reason to bypass its compatibility check.
> OpenAI instrumentation remains enabled. The merged provider preserves main's
> beta-endpoint default, explicit API-version overrides, and SDK 3 schema handling.

Scope: runtime dependencies, optional extras, development/build tools, workspace wrappers, standalone example manifests, inline script dependencies, GitHub Actions, and the complete transitive lockfile. Versions were checked against live PyPI metadata and upstream release sources; stable releases are preferred, except the already-beta OpenTelemetry instrumentation family. The supported Python range remains 3.12–3.15.

## Summary

102 existing lockfile package versions were updated; Vulture and zensical-ui were added, while deepmerge and gradio-client dropped out. Six installed packages remain below latest: the two intentional major-version deferrals and four upstream-constrained transitives documented here.

## Direct dependency inventory

`—` means not locked by the root workspace (build-only or standalone example dependency). Unpinned requirements already permit the current release; they are not artificially pinned here. Existing permissive minimums are retained where the lock already selected the latest version. Local wrapper versions remain release-managed by `scripts/sync-wrapper-version.py`.

| Dependency | Previous lock | Selected lock | Latest available | Scope / outcome |
| --- | --- | --- | --- | --- |
| [a2a-sdk](https://pypi.org/project/a2a-sdk/1.3.0/) | 1.2.3 | 1.3.0 | 1.3.0 | root runtime; upgraded |
| [agent-client-protocol](https://pypi.org/project/agent-client-protocol/1.0.0/) | 0.12.1 | 1.0.0 | 1.0.0 | root runtime; upgraded |
| [aiohttp](https://pypi.org/project/aiohttp/3.14.5/) | 3.14.5 | 3.14.5 | 3.14.5 | root runtime; already current |
| [anthropic](https://pypi.org/project/anthropic/2.9.0/) | 1.17.0 | 1.17.0 | 2.9.0 | root runtime; **major upgrade deferred**, see below |
| [azure-identity](https://pypi.org/project/azure-identity/1.27.0/) | 1.27.0 | 1.27.0 | 1.27.0 | root all-providers, root azure; already current |
| [boto3](https://pypi.org/project/boto3/1.44.4/) | 1.43.119 | 1.44.4 | 1.44.4 | root all-providers, root bedrock, root dev; upgraded |
| [duckdb](https://pypi.org/project/duckdb/2.0.0/) | 2.0.0 | 2.0.0 | 2.0.0 | root batch-parquet; already current |
| [email-validator](https://pypi.org/project/email-validator/2.4.0/) | 2.3.0 | 2.4.0 | 2.4.0 | root runtime; upgraded |
| [fastapi](https://pypi.org/project/fastapi/0.163.1/) | — | — | 0.163.1 | example runtime; latest allowed; not root-locked |
| [fastmcp-slim](https://pypi.org/project/fastmcp-slim/4.2.0/) | 4.1.3 | 4.2.0 | 4.2.0 | example runtime, root runtime; upgraded |
| [filelock](https://pypi.org/project/filelock/4.2.11/) | 4.2.4 | 4.2.11 | 4.2.11 | root runtime; upgraded |
| [gepa](https://pypi.org/project/gepa/0.1.4/) | 0.1.4 | 0.1.4 | 0.1.4 | root gepa; already current |
| [google-api-python-client](https://pypi.org/project/google-api-python-client/2.202.0/) | — | — | 2.202.0 | example runtime; latest allowed; not root-locked |
| [google-cloud-aiplatform](https://pypi.org/project/google-cloud-aiplatform/2.9.0/) | — | — | 2.9.0 | example runtime; latest allowed; not root-locked |
| [google-genai](https://pypi.org/project/google-genai/3.4.0/) | 2.33.0 | 2.34.0 | 3.4.0 | root runtime; upgraded within supported 2.x; **3.x deferred** |
| [hatchling](https://pypi.org/project/hatchling/1.33.0/) | — | — | 1.33.0 | root build, wrapper build; latest allowed; not root-locked |
| [huggingface-hub](https://pypi.org/project/huggingface-hub/2.4.0/) | 2.3.0 | 2.4.0 | 2.4.0 | root runtime, wrapper runtime; upgraded |
| [ipdb](https://pypi.org/project/ipdb/0.13.13/) | 0.13.13 | 0.13.13 | 0.13.13 | root dev; already current |
| [jsonschema](https://pypi.org/project/jsonschema/4.26.0/) | 4.26.0 | 4.26.0 | 4.26.0 | root runtime; already current |
| [keyring](https://pypi.org/project/keyring/25.7.0/) | 25.7.0 | 25.7.0 | 25.7.0 | root runtime; already current |
| [mcp](https://pypi.org/project/mcp/2.3.0/) | 2.3.0 | 2.3.0 | 2.3.0 | root runtime; already current |
| [mcp-types](https://pypi.org/project/mcp-types/2.3.0/) | 2.3.0 | 2.3.0 | 2.3.0 | root runtime; already current |
| [mslex](https://pypi.org/project/mslex/1.3.0/) | 1.3.0 | 1.3.0 | 1.3.0 | root runtime; already current |
| [numpy](https://pypi.org/project/numpy/2.5.5/) | 2.5.4 | 2.5.5 | 2.5.5 | root privacy, root privacy-gpu; upgraded |
| [onnxruntime](https://pypi.org/project/onnxruntime/1.32.0/) | 1.31.0 | 1.32.0 | 1.32.0 | root privacy; upgraded |
| [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/1.32.0/) | 1.31.0 | 1.32.0 | 1.32.0 | root privacy-gpu; upgraded |
| [openai](https://pypi.org/project/openai/3.52.0/) | 3.51.0 | 3.52.0 | 3.52.0 | root runtime; upgraded |
| [opentelemetry-exporter-otlp-proto-http](https://pypi.org/project/opentelemetry-exporter-otlp-proto-http/1.45.2/) | 1.45.1 | 1.45.2 | 1.45.2 | root runtime; upgraded |
| [opentelemetry-instrumentation-google-genai](https://pypi.org/project/opentelemetry-instrumentation-google-genai/1.3b0/) | 1.2b0 | 1.3b0 | 1.3b0 | root runtime; upgraded |
| [opentelemetry-instrumentation-openai](https://pypi.org/project/opentelemetry-instrumentation-openai/0.63.1/) | 0.62.4 | 0.63.1 | 0.63.1 | root runtime; upgraded |
| [pillow](https://pypi.org/project/pillow/13.0.0/) | 13.0.0 | 13.0.0 | 13.0.0 | root runtime; already current |
| [pre-commit](https://pypi.org/project/pre-commit/4.7.0/) | 4.7.0 | 4.7.0 | 4.7.0 | root dev; already current |
| [prompt-toolkit](https://pypi.org/project/prompt-toolkit/3.0.53/) | 3.0.53 | 3.0.53 | 3.0.53 | root runtime; already current |
| [pydantic](https://pypi.org/project/pydantic/2.14.3/) | 2.14.3 | 2.14.3 | 2.14.3 | root dev, root runtime; already current |
| [pydantic-monty](https://pypi.org/project/pydantic-monty/1.6.0/) | 1.4.0 | 1.6.0 | 1.6.0 | root runtime; upgraded |
| [pydantic-settings](https://pypi.org/project/pydantic-settings/2.16.3/) | 2.16.2 | 2.16.3 | 2.16.3 | root runtime; upgraded |
| [pyperclip](https://pypi.org/project/pyperclip/1.11.0/) | 1.11.0 | 1.11.0 | 1.11.0 | root runtime; already current |
| [pytest](https://pypi.org/project/pytest/9.2.0/) | 9.1.1 | 9.2.0 | 9.2.0 | root dev; upgraded |
| [pytest-asyncio](https://pypi.org/project/pytest-asyncio/1.5.0/) | 1.4.0 | 1.5.0 | 1.5.0 | root dev; upgraded |
| [pytest-cov](https://pypi.org/project/pytest-cov/7.1.0/) | 7.1.0 | 7.1.0 | 7.1.0 | root dev; already current |
| [pytest-xdist](https://pypi.org/project/pytest-xdist/3.8.0/) | 3.8.0 | 3.8.0 | 3.8.0 | root dev; already current |
| [python-frontmatter](https://pypi.org/project/python-frontmatter/1.3.0/) | 1.3.0 | 1.3.0 | 1.3.0 | root runtime; already current |
| [pyyaml](https://pypi.org/project/pyyaml/6.0.3/) | 6.0.3 | 6.0.3 | 6.0.3 | root dev, root runtime; already current |
| [rich](https://pypi.org/project/rich/15.0.0/) | 15.0.0 | 15.0.0 | 15.0.0 | root runtime; already current |
| [ruamel-yaml](https://pypi.org/project/ruamel-yaml/0.19.1/) | 0.19.1 | 0.19.1 | 0.19.1 | root dev, root runtime; already current |
| [ruff](https://pypi.org/project/ruff/0.17.5/) | 0.17.1 | 0.17.5 | 0.17.5 | root dev; upgraded |
| [setuptools](https://pypi.org/project/setuptools/84.2.1/) | — | — | 84.2.1 | example build; latest allowed; not root-locked |
| [starlette](https://pypi.org/project/starlette/1.10.0/) | 1.9.0 | 1.10.0 | 1.10.0 | example runtime, root runtime; upgraded |
| [tensorzero](https://pypi.org/project/tensorzero/2026.6.0/) | 2026.6.0 | 2026.6.0 | 2026.6.0 | root all-providers, root tensorzero; already current |
| [textual](https://pypi.org/project/textual/8.2.8/) | 8.2.8 | 8.2.8 | 8.2.8 | root textual; already current |
| [textual-image](https://pypi.org/project/textual-image/0.14.4/) | 0.14.2 | 0.14.4 | 0.14.4 | root runtime; upgraded |
| [tiktoken](https://pypi.org/project/tiktoken/0.14.0/) | 0.14.0 | 0.14.0 | 0.14.0 | root runtime; already current |
| [tokenizers](https://pypi.org/project/tokenizers/0.23.3/) | 0.23.3 | 0.23.3 | 0.23.3 | root privacy, root privacy-gpu; already current |
| [trackio](https://pypi.org/project/trackio/0.45.1/) | 0.44.0 | 0.45.1 | 0.45.1 | root trackio; upgraded |
| [ty](https://pypi.org/project/ty/0.0.104/) | 0.0.90 | 0.0.104 | 0.0.104 | root dev; upgraded |
| [typer](https://pypi.org/project/typer/0.29.1/) | 0.27.3 | 0.29.1 | 0.29.1 | root runtime; upgraded |
| [uritemplate](https://pypi.org/project/uritemplate/4.2.0/) | 4.2.0 | 4.2.0 | 4.2.0 | root runtime; already current |
| [uvicorn](https://pypi.org/project/uvicorn/0.54.0/) | 0.53.0 | 0.54.0 | 0.54.0 | example runtime; upgraded |
| [uvloop](https://pypi.org/project/uvloop/0.23.0/) | 0.23.0 | 0.23.0 | 0.23.0 | root runtime; already current |
| [watchfiles](https://pypi.org/project/watchfiles/1.3.0/) | 1.3.0 | 1.3.0 | 1.3.0 | root runtime; already current |
| [wheel](https://pypi.org/project/wheel/0.48.0/) | — | — | 0.48.0 | example build; latest allowed; not root-locked |
| [zensical](https://pypi.org/project/zensical/0.1.1/) | 0.0.71 | 0.1.1 | 0.1.1 | root dev; upgraded |
| [vulture](https://pypi.org/project/vulture/2.16/) | — | 2.16 | 2.16 | New dev dependency; high-confidence dead-code gate |

## Transitive dependencies

`uv lock --upgrade` refreshed the complete dependency graph, including optional extras. Four transitive dependencies cannot be upgraded further without violating upstream requirements; no unsafe resolver overrides were added.

| Dependency | Selected | Latest | Upstream constraint |
| --- | --- | --- | --- |
| py-key-value-aio | 0.4.6 | 0.5.0 | FastMCP 4.2.0 requires `<0.5.0,>=0.4.6` |
| python-dateutil | 2.9.0.post0 | 4.0.0 | botocore 1.44.4 requires `<3.0.0,>=2.1` |
| tenacity | 9.1.4 | 9.2.1 | google-genai 3.4.0 requires `<9.2.0,>=8.2.3` |
| websockets | 15.0.1 | 17.4 | OpenAI's realtime extra requires `<16,>=13` |

### Complete lockfile changes

| Package | Previous | Selected |
| --- | --- | --- |
| a2a-sdk | 1.2.3 | 1.3.0 |
| agent-client-protocol | 0.12.1 | 1.0.0 |
| annotated-doc | 0.0.5 | 0.0.6 |
| anyio | 4.15.1 | 4.16.0 |
| authlib | 1.9.3 | 1.9.8 |
| boto3 | 1.43.119 | 1.44.4 |
| botocore | 1.43.119 | 1.44.4 |
| cachetools | 7.2.0 | 8.0.0 |
| caio | 0.12.4 | 0.12.10 |
| certifi | 2026.10.19 | 2026.10.23 |
| charset-normalizer | 3.5.1 | 3.5.2 |
| click | 8.5.0 | 8.5.1 |
| coverage | 7.16.1 | 7.17.3 |
| cryptography | 50.0.1 | 50.0.2 |
| cyclopts | 4.25.3 | 5.5.2 |
| deepmerge | 3.0.1 | removed |
| distlib | 0.4.3 | 0.4.4 |
| dnspython | 2.9.0 | 2.9.2 |
| email-validator | 2.3.0 | 2.4.0 |
| executing | 2.2.1 | 2.3.0 |
| fastmcp-slim | 4.1.3 | 4.2.0 |
| filelock | 4.2.4 | 4.2.11 |
| google-api-core | 2.38.0 | 2.44.0 |
| google-auth | 2.58.0 | 2.62.2 |
| google-genai | 2.33.0 | 2.34.0 |
| googleapis-common-protos | 1.75.3 | 1.76.0 |
| gradio-client | 2.7.2 | removed |
| griffelib | 2.3.0 | 2.4.1 |
| hf-xet | 1.6.0 | 1.8.0 |
| httpcore2 | 2.13.0 | 2.16.2 |
| httpx2 | 2.13.0 | 2.16.2 |
| huggingface-hub | 2.3.0 | 2.4.0 |
| identify | 2.6.19 | 2.6.20 |
| iniconfig | 2.3.0 | 2.3.1 |
| ipython | 9.17.1 | 9.18.1 |
| jedi | 0.20.0 | 0.20.1 |
| jiter | 0.17.0 | 0.19.0 |
| joserfc | 1.7.6 | 1.8.0 |
| markdown | 3.10.3 | 3.11.3 |
| markdown-it-py | 4.2.0 | 4.3.0 |
| markupsafe | 3.0.3 | 3.0.4 |
| mdit-py-plugins | 0.6.1 | 0.7.0 |
| more-itertools | 11.1.0 | 11.2.1 |
| multidict | 6.9.1 | 7.2.1 |
| nodeenv | 1.10.0 | 1.12.0 |
| numpy | 2.5.4 | 2.5.5 |
| nvidia-cublas | 13.8.1.7 | 13.10.0.20 |
| nvidia-cuda-nvrtc | 13.4.92 | 13.5.37 |
| nvidia-cuda-runtime | 13.4.92 | 13.5.37 |
| nvidia-cudnn-cu13 | 9.27.0.42 | 9.28.1.4 |
| nvidia-cufft | 12.4.0.43 | 12.5.0.14 |
| nvidia-curand | 10.4.4.72 | 10.4.5.25 |
| nvidia-nvjitlink | 13.4.92 | 13.5.37 |
| onnxruntime | 1.31.0 | 1.32.0 |
| onnxruntime-gpu | 1.31.0 | 1.32.0 |
| openai | 3.51.0 | 3.52.0 |
| opentelemetry-api | 1.45.1 | 1.45.2 |
| opentelemetry-exporter-http-transport | 0.66b1 | 0.66b2 |
| opentelemetry-exporter-otlp-common | 0.66b1 | 0.66b2 |
| opentelemetry-exporter-otlp-proto-common | 1.45.1 | 1.45.2 |
| opentelemetry-exporter-otlp-proto-http | 1.45.1 | 1.45.2 |
| opentelemetry-instrumentation | 0.66b1 | 0.66b2 |
| opentelemetry-instrumentation-google-genai | 1.2b0 | 1.3b0 |
| opentelemetry-instrumentation-openai | 0.62.4 | 0.63.1 |
| opentelemetry-proto | 1.45.1 | 1.45.2 |
| opentelemetry-sdk | 1.45.1 | 1.45.2 |
| opentelemetry-semantic-conventions | 0.66b1 | 0.66b2 |
| opentelemetry-util-genai | 1.2b0 | 1.3b0 |
| platformdirs | 4.11.11 | 4.14.5 |
| pluggy | 1.6.0 | 1.7.0 |
| proto-plus | 1.28.4 | 1.29.0 |
| protobuf | 6.33.6 | 7.36.2 |
| pycparser | 3.0 | 3.12 |
| pydantic-monty | 1.4.0 | 1.6.0 |
| pydantic-monty-client | 1.4.0 | 1.6.0 |
| pydantic-monty-runtime | 1.4.0 | 1.6.0 |
| pydantic-settings | 2.16.2 | 2.16.3 |
| pymdown-extensions | 12.0.1 | 13.1 |
| pytest | 9.1.1 | 9.2.0 |
| pytest-asyncio | 1.4.0 | 1.5.0 |
| python-discovery | 1.6.1 | 1.9.5 |
| python-dotenv | 1.2.3 | 1.2.4 |
| python-multipart | 0.0.32 | 0.0.34 |
| regex | 2026.9.10 | 2026.11.15 |
| rich-rst | 2.1.0 | 2.2.0 |
| rpds-py | 2026.6.3 | 2026.9.1 |
| ruff | 0.17.1 | 0.17.5 |
| s3transfer | 0.19.2 | 0.19.3 |
| sse-starlette | 3.4.11 | 3.7.0 |
| starlette | 1.9.0 | 1.10.0 |
| textual-image | 0.14.2 | 0.14.4 |
| tomli | 2.4.1 | 2.5.0 |
| trackio | 0.44.0 | 0.45.1 |
| truststore | 0.10.4 | 0.10.5 |
| ty | 0.0.90 | 0.0.104 |
| typer | 0.27.3 | 0.29.1 |
| typing-inspection | 0.4.4 | 0.4.5 |
| uncalled-for | 0.4.0 | 0.6.1 |
| uuid-utils | 1.0.0 | 1.2.0 |
| uvicorn | 0.53.0 | 0.54.0 |
| virtualenv | 21.14.13 | 21.15.7 |
| vulture | — | 2.16 |
| wcwidth | 0.8.4 | 0.9.4 |
| wrapt | 2.4.1 | 2.5.2 |
| zensical | 0.0.71 | 0.1.1 |
| zensical-ui | — | 0.0.42 |

Packages not listed in this delta remain unchanged; all remaining registry lock entries were checked against PyPI. The following compatibility decisions are deliberate, not missed upgrades.

## Major versions deferred

### Anthropic: retain 1.17.0; latest is 2.9.0

The [2.x migration](https://platform.claude.com/docs/en/api/migration-2026-10-21)
is not safe as a maintenance update:

- The new API rejects `temperature`, `top_p`, and `top_k` for all models.
  `AnthropicLLM.prepare_provider_arguments` deliberately preserves these for
  older models. Silently removing them would change supported user behavior.
- API-key authentication changes from `X-Api-Key` to Bearer authorization, which
  can break custom gateways. Vertex's `anthropic_version` **body** field also
  changes; overriding headers is not sufficient.
- `BetaMessageParam` becomes a union, breaking the converter's constructor calls.
  This is fixable, but does not solve the wire-contract changes above.
- Existing Copilot credential-isolation tests passed under 2.9 (40 tests), but
  broader adapter tests failed; the candidate was reverted rather than shipped.

Adoption needs an explicit sampling-control deprecation decision, gateway migration,
and verified native/Vertex request behavior. No speculative 2.x compatibility code
was retained. See [SDK migration guide](https://github.com/anthropics/anthropic-sdk-python/blob/v2.9.0/MIGRATION.md).

### Google: upgrade 2.33.0 → 2.34.0; latest 3.4.0 deferred

The current Google OpenTelemetry instrumentor **1.3b0 explicitly supports
`google-genai>=1.32.0,<3`**. A real instrumentation probe against 3.4.0 refused
instrumentation; bypassing that check would hide an unsupported combination.
A new regression test exercises actual provider instrumentation through
`configure_otel`, not a mocked instrumentor.

[2.34.0](https://github.com/googleapis/python-genai/releases/tag/v2.34.0)
fixes per-request authorization/timeout headers leaking into client defaults and
adds Python 3.15 support. The provider now explicitly chooses Developer API
`v1beta`, preserving preview/latest aliases; Vertex standard and Flex endpoints
are unchanged. Real SDK request-boundary tests cover serialization, endpoint
selection, schemas, and transport-error handling.

[3.x](https://github.com/googleapis/python-genai/releases/tag/v3.0.0) additionally
changes the default Developer API to `v1` and the HTTP backend to `httpx2`.
Revisit only when tracing supports it and endpoint/transport contracts are checked.

Separate pre-existing follow-up: Vertex rejects the tool field
`include_server_side_tool_invocations=False` during SDK serialization (also
reproduced on 2.33). The new boundary tests document this limitation; it is not
attributed to the dependency upgrade or silently treated as working.

## Features and simplifications reviewed

| Upgrade | Benefit and action in this codebase |
| --- | --- |
| [ACP 1.0](https://github.com/agentclientprotocol/python-sdk/releases/tag/1.0.0) | Replaced removed helpers with generated schema constructors throughout the adapter, wrapper, and tests. Added a shared content union. New, unadvertised protocol requests explicitly return method-not-found rather than inheriting successful no-op protocol methods. Kept experimental transports/v2 separate. |
| [Monty 1.6](https://github.com/pydantic/monty/releases/tag/v1.6.0) | Async host-call and memory cleanup improvements. Verified filesystem/concurrency coverage; adopted `fnmatch` in sandbox guidance and reserved module names, with actual sandbox behavior coverage. File-import support is **not** enabled without a separate filesystem-policy review. |
| [FastMCP 4.2](https://github.com/PrefectHQ/fastmcp/releases/tag/v4.2.0) | Auth/proxy/OpenAPI fixes. New skill distribution and hashed manifests are a useful future integration through the harness, not a reason to duplicate its lifecycle. Existing concurrent FastMCP pin edit was preserved. |
| [A2A 1.3](https://github.com/a2aproject/a2a-python/releases/tag/v1.3.0) | Legacy history/error handling, malformed responses, and executor recovery fixes. Parked-task timeout is a follow-up for existing task retention, not automatically enabled. |
| [pydantic-settings 2.16.3](https://github.com/pydantic/pydantic-settings/releases/tag/v2.16.3) | Secret-source symlink confinement fix; out-of-directory secret symlinks may now be skipped with warnings. No custom configuration-source rewrite needed. |
| [Typer 0.29.1](https://github.com/fastapi/typer/releases/tag/0.29.1) | Pydantic parameter validation/Python 3.15 support. Removed Click integration APIs were checked; CLI tests exercise the existing private integration points. |
| [Starlette 1.10](https://github.com/Kludex/starlette/releases/tag/1.10.0) | Routing/form improvements and exception traceback fix; MCP/A2A/API integrations tested. No use of removed FormParser internals found. |
| [Trackio 0.45.1](https://github.com/gradio-app/trackio/releases/tag/trackio%400.45.1) | Media-path/run-mutation hardening and retry fixes. New dependency graph also removes gradio-client/deepmerge. Batch/Trackio tests run with the extra installed. |
| [DuckDB 2.0](https://github.com/duckdb/duckdb-python/releases/tag/v2.0.0) | Already current in baseline lock. Existing parquet code does not use removed Arrow APIs; real optional-extra tests passed. Streaming/chunking is a possible later memory improvement. |
| [GEPA 0.1.4](https://github.com/gepa-ai/gepa/releases/tag/v0.1.4) | Already current. Reflection and batch-proposal protocols may simplify `integrations/gepa.py`; defer to a dedicated integration change with callback/budget semantics tests. |
| [Hugging Face Hub 2.4](https://github.com/huggingface/huggingface_hub/releases/tag/v2.4.0) | Bucket filesystem operations and extension diagnostics; current inference/auth code does not need new Bucket APIs. |
| OpenAI 3.52, email-validator 2.4, filelock 4.2.11, textual-image 0.14.4, OTel patches, boto3 1.44.4 | Selected current compatible releases; existing provider/auth/UI/batch tests cover current uses. No unrelated new SDK surface enabled. |
| ONNX Runtime 1.32 / NumPy 2.5.5 | Updated optional privacy stack; CPU extras installation/imports and privacy tests exercised. No claim of live model/CUDA inference validation. |
| Zensical 0.1.1 / Ruff 0.17.5 / ty 0.0.104 | Toolchain refreshed; source type contracts and renamed suppression rules corrected rather than silencing diagnostics. |

## Lint, dead-code, and CI tooling

See [development-check instructions](tooling.md).

- Ruff is now consistently 0.17.5 in the project and pre-commit (previously
  0.17.1 / 0.16.0 / inline-script 0.16.8). Format/lint scripts reuse the locked
  dev environment instead of maintaining independent PEP 723 pins.
- The existing E/F/I/TC and selected RUF/B/ISC/SIM/ASYNC/PLW rules remain useful.
  Canonical `TC` replaces the legacy `TCH` selector. No mass rule expansion.
- Vulture 2.16 now runs in CI at 100% confidence over shipped source trees.
  Previously its allowlist had **no invoker**. Three required protocol argument
  names are documented; the pure unsupported-ACP shim is excluded narrowly.
  A 60% scan produced 316 advisory findings, not authorization to delete APIs.
- CPD is duplication detection, not dead-code detection. Upgraded PMD 7.27 →
  7.29 after comparing all seven baseline findings. Only one intentional type
  contract edit required a targeted baseline hash/token update.
- Updated fallback Temurin 17.0.9+9 → 17.0.21+9; tar extraction now uses
  `filter="data"` with a traversal regression test.
- Updated GitHub Actions: checkout 7.0.1 (`@v7`), setup-python 7.1.0 (`@v7`),
  setup-java 6.3.0 (`@v6`), upload-artifact 7.0.2 (`@v7`), release-drafter
  7.11.0 (`@v7`), setup-uv **`@v10.3.3`** (no `v10` tag exists).
  Release-drafter's separate autolabeler and token input were migrated.
- Publishing installs the actual locked dev group rather than nonexistent
  `.[dev]`. No release or GitHub workflow was executed.

Inline dependency inventory: `scripts/gen_schema.py` already allows current
Rich/Typer/Pydantic/settings; two `docs/benchmark_data/tb4` scripts use unpinned
Pillow; the PR-review skill script has no dependencies. Generated packaged
mirrors were not manually edited. Build-only Hatchling/setuptools/wheel and
standalone example ranges already permit the latest releases listed above.

## Platform and release caveats

- GEPA 0.1.4 declares Python `<3.15`; its extra was installed on Python 3.12,
  not claimed to work over the root's full supported range.
- ONNX 1.32 lacks CPython 3.15 wheels/source distributions. CPU extras were
  validated on Python 3.12. GPU extras still target CUDA 13 (not a new transition);
  Linux ARM GPU wheels also need glibc 2.34. GPU inference was not run.
- All four standalone example manifests resolve in dry runs. FastAPI's example
  resolves the published package, while the OpenAPI/RAG examples reference the
  local tree; resolver success is not live service validation.
- Wrapper package versions/pins remain release-managed. Run
  `uv run scripts/sync-wrapper-version.py` as part of the coordinated release;
  workspace tests are not proof of an independently installed old wrapper wheel.
- All 213 external lock entries were checked against OSV's exact-version API:
  zero known advisories returned. This is database coverage, not a security guarantee.

## Validation

- Required `uv run scripts/format.py --check`, `scripts/lint.py`, and
  `scripts/typecheck.py`: pass.
- `scripts/deadcode.py`, `scripts/cpd.py --check`, and
  `scripts/check_internal_resources.py`: pass.
- Python 3.12 integration suites: ACP **119 passed**, API **60 passed**,
  remaining integrations **267 passed, 16 skipped**.
- Python 3.12 optional-extra batch/privacy tests: **150 passed, 2 skipped**;
  all-provider, Textual, GEPA/Trackio, DuckDB, and CPU privacy extras install
  and import successfully. Skips include platform/locale-specific cases.
- Google request/provider + real tracing regression coverage: **121 passed**.
- Package wheel/sdist build and Zensical documentation build: pass.
- Lock consistency, installed dependency consistency, pre-commit configuration,
  workflow YAML/ref checks, and `git diff --check`: pass.
- No paid/live model calls, live RAG service, GPU inference, GitHub release,
  or workflow execution was performed.
- Final complete Python 3.15 unit suite: **12,139 passed, 9 skipped**. The
  reviewed PTC vocabulary baseline changes only the newly documented `fnmatch`
  module; all direct-tool surface digests remain unchanged.
