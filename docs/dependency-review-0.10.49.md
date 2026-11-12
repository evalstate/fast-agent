# Dependency review for dev/0.10.49

Checked PyPI, tagged upstream sources, and release notes on 27 October 2026.

**Disposition:** local review scratch, not a release deliverable. Keep this memo
untracked; ship the durable installation/API caveats in the batch, GEPA, and
extension docs instead. No commits or pushes were made by this review.

## Applied

| Package | Previous | Selected | Benefit |
| --- | --- | --- | --- |
| Trackio | extra suspended | >=0.44.0 (locked 0.44.0) | Allows Pillow <14, restoring installation alongside the security floor Pillow >=13. |
| OpenAI | 3.37.0 | 3.40.0 | PEP 604 parsing fixes on Python <3.14, cleaner HTTP errors, namespace-aware SDK parsing, cache-breakpoint TTL types. |
| google-genai | 2.32.0 | 2.33.0 | Async connection-retry, zero-delay retry and mTLS endpoint fixes. |
| azure-identity | >=1.25.3 (locked 1.26.0) | >=1.27.0 | Async OnBehalfOf token-cache fix, CLI error parsing and macOS cache isolation fixes. |
| boto3 | >=1.43.98 (locked 1.43.116) | >=1.43.119 | Includes botocore's compromised-chain CA removal; raises the minimum past credential-refresh fixes. |

The `trackio` extra is restored and `gepa` includes Trackio again. GEPA 0.1.4
itself does not expose the Trackio optimizer options used by some dashboard
examples: installing Trackio fixes batch monitoring and direct logging, not that
separate GEPA API limitation. The guides document this distinction.

Sources:
[Trackio](https://pypi.org/project/trackio/0.44.0/),
[OpenAI changes](https://github.com/openai/openai-python/compare/v3.37.0...v3.40.0),
[Google](https://github.com/googleapis/python-genai/releases/tag/v2.33.0),
[Azure Identity](https://github.com/Azure/azure-sdk-for-python/blob/azure-identity_1.27.0/sdk/identity/azure-identity/CHANGELOG.md),
[boto3](https://github.com/boto/boto3/blob/1.43.119/CHANGELOG.rst),
[botocore](https://github.com/boto/botocore/blob/1.43.119/CHANGELOG.rst).

The CA removal is in botocore 1.43.117 and affects its fallback vendored bundle.
When certifi is installed, upstream instead requires certifi >=2026.10.19;
the lock already selects 2026.10.19. The credential-refresh race fix is in
botocore 1.43.115, already present in the previous lock but not guaranteed by
the old boto3 minimum. These are distinct from an SDK-wide security guarantee.

## Anthropic 2.3.0: deliberately deferred

Keep 1.17.0, the latest 1.x release, for this refresh. The 2.x migration is not
a safe pin-only update:

- `BetaMessageParam` becomes a union rather than a constructible TypedDict;
  conversion must use role-specific message types.
- API-key authentication changes from `X-Api-Key` to Bearer Authorization.
  Custom gateways need a compatibility review.
- The default API version changes to `2026-10-21`. A real-SDK loopback probe
  observed this in Vertex's `anthropic_version` request body, replacing
  `vertex-2023-10-16`. Header overrides do not restore that body field.
  This has **not** been verified against a live Vertex endpoint.
- Copilot's broker headers and credential isolation passed loopback tests with
  both SDK versions. Retain that regression coverage.

Resolve the Vertex contract before migrating:
[SDK comparison](https://github.com/anthropics/anthropic-sdk-python/compare/v1.17.0...v2.0.0),
[Vertex request documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude),
[2.3.0 Vertex body construction](https://github.com/anthropics/anthropic-sdk-python/blob/v2.3.0/src/anthropic/lib/vertex/_client.py),
[2.3.0 message union](https://github.com/anthropics/anthropic-sdk-python/blob/v2.3.0/src/anthropic/types/beta/beta_message_param.py),
[2.3.0 authentication](https://github.com/anthropics/anthropic-sdk-python/blob/v2.3.0/src/anthropic/_client.py).

Fast-agent's Bedrock implementation uses boto3 Converse, not the Anthropic
SDK's legacy Bedrock Messages adapter.

## Useful follow-ups, not included in this maintenance change

- **Azure async authentication:** replace the synchronous credential/token
  callback in `llm_azure.py` with `azure.identity.aio.DefaultAzureCredential`
  and `get_bearer_token_provider`, with explicit credential teardown.
- **Google raw JSON Schema:** evaluate `parameters_json_schema` and
  `response_json_schema` as replacements for handwritten schema stripping.
  Verify Developer API and Vertex behavior before removing compatibility code.
- **OpenAI cache breakpoints:** content-block TTLs support `30m` and `12h`;
  request-wide `PromptCacheOptions.ttl` still only permits `30m`. Exposing
  breakpoint placement needs explicit replay/compaction semantics.
- **Namespaced tools:** SDK support does not automatically update fast-agent's
  flat-name tool dispatch. Design identity handling before enabling it.
- Keep the existing OpenAI WebSocket transport cast and schema-normalization
  wrapper in this pin refresh. Replacement design is outside this review.

Hugging Face Hub 2.3.0 and TensorZero 2026.6.0 are already current.
OpenAI instrumentation remains current at 0.62.4. Google instrumentation has a
new 1.3b0 prerelease; it remains at 1.2b0 pending a separate telemetry review.
These version statements were rechecked against the respective PyPI JSON
metadata, including Anthropic's latest stable 1.x release.

The TensorZero provider uses its OpenAI-compatible endpoint, not the optional
Python SDK. Clarify the extra's public purpose before considering removal.
Updating that extra does not update a deployed gateway; see the
[gateway security advisory](https://github.com/tensorzero/tensorzero/security/advisories/GHSA-824w-x939-6cmc).

## Review findings and validation

- **Fixed:** Trackio 0.44.0 auto-enables CPU/system collection when psutil is
  available. Batch initialization and GEPA helper defaults now pass
  `auto_log_cpu=False`, matching the GPU default. GEPA callers can explicitly
  override it. Source: [Trackio init](https://github.com/gradio-app/trackio/blob/trackio%400.44.0/trackio/__init__.py)
  (also verified in the installed PyPI wheel).
- Expanded the real-SDK smoke test: local batch and direct GEPA-helper logs
  survive SQLite flush; CPU/GPU/console collectors remain inactive; an audit
  hook rejects and records attempted network connections. Without a project,
  importing the integrations and exercising the monitor does not import
  Trackio or create storage, even with ambient remote Trackio settings.
- Monitoring remains opt-in. After opting in, Trackio honors its explicit
  remote environment configuration; this is not a promise that `--project`
  overrides `TRACKIO_SPACE_ID` or `TRACKIO_SERVER_URL`.
- Copilot loopback coverage validates broker credential rotation, ambient
  secret/proxy isolation, and refusal to follow 307/308 redirects. It is not a
  live service or full completion-parsing test.
- GEPA 0.1.4's installed `TrackingConfig` signature and `gepa.api.optimize`
  signature confirm the documented lack of Trackio optimizer parameters.
  That existing limitation is not repaired by installing the extra.

Validation used `/tmp/fast-agent-dependency-review-07rn95dp`, a working-tree
snapshot with separate environments; the main `.venv` was not synchronized.

Commands/results:

- `uv sync --locked --extra trackio --extra gepa --extra azure --extra bedrock`:
  passes on Python 3.14.6. Repeated with `--python 3.13` and
  `UV_PROJECT_ENVIRONMENT=.venv313`: passes on 3.13.12.
- `uv build --wheel`: passes. Installed the built wheel's `[trackio]` and
  `[gepa]` extras independently into fresh Python 3.13 environments using
  `uv pip install --python <env>/bin/python '<wheel>[<extra>]'`, without a
  lockfile. Both installs and `uv pip check` pass: Trackio 0.44.0, Pillow
  13.0.0, and (for `[gepa]`) GEPA 0.1.4. Wheel metadata retains
  `Requires-Dist: pillow>=13.0.0` and both Trackio extra requirements.
- `UV_PROJECT_ENVIRONMENT=.venv313 uv run --no-sync pytest -q
  tests/unit/fast_agent/batch tests/unit/fast_agent/integrations
  tests/unit/fast_agent/llm/providers tests/unit/llm/test_copilot_messages_wire.py`:
  **1523 passed, 5 skipped** (one optional DuckDB test, four Windows-only tests).
- `uv run --no-sync --with 'anthropic[vertex]==2.3.0' pytest -q
  tests/unit/llm/test_copilot_messages_wire.py`: **3 passed**, without changing
  the project's Anthropic pin.
- A temporary real-SDK Vertex loopback probe reproduced body version
  `vertex-2023-10-16` on 1.17.0 and `2026-10-21` on 2.3.0, with and without an
  `anthropic-version` header override. No live Vertex request was made.
- `uv run scripts/format.py --check`: initially found one new test line to
  wrap; ran `uv run scripts/format.py --path tests/unit/fast_agent/batch/test_trackio_install.py`.
  Final check passed.
- `uv run scripts/lint.py`: passed.
- `UV_NO_SYNC=1 uv run scripts/typecheck.py`: passed against the main working
  tree without syncing its environment; also passed in the isolated snapshot.

Recommendation: release the scoped dependency/Trackio changes; no remaining
blockers were found in this scope. Keep Anthropic at 1.17.0; migrating it is
separate work. HF and benchmark changes require their other reviewers' approval. No paid/model calls were made.
