# Provider SDK review — 19 September 2026

Latest stable versions checked against PyPI and upstream release notes. This update changes dependency requirements and the lockfile; it does not enable new provider features automatically.

## Dependency changes

| SDK | Previous requirement | Updated requirement |
| --- | --- | --- |
| OpenAI | `==3.8.0` | `==3.16.2` |
| Anthropic | `==1.4.0` | `==1.7.0` |
| Google GenAI | `==2.22.0` | `==2.24.0` |
| boto3 | `>=1.43.83` | `>=1.43.98` |
| Azure Identity | `>=1.14.0` | `>=1.25.3` |
| TensorZero | `>=2025.7.5` | `>=2026.6.0` |

Hugging Face Hub was already updated to the latest `1.32.0` in the working tree; that change is preserved. Azure Identity and TensorZero were already locked at the versions above; raising their optional dependency floors also applies the minimums to downstream installations. boto3's matching botocore dependency updates to `1.43.98`. Provider extras and `all-providers` remain aligned.

## Findings and recommended follow-ups

### OpenAI: strongest simplification opportunity

- **Managed Responses WebSockets:** `AsyncResponsesWebSocketSession` now supplies routed lanes, bounded buffering and final-response collection. Evaluate replacing overlapping transport/collection code in `src/fast_agent/llm/provider/openai/responses_websocket.py`. Preserve fast-agent's continuation/reconnect behavior, cancellation and Codex/xAI differences; this is not a mechanical substitution. [SDK guide](https://github.com/openai/openai-python/blob/v3.16.2/src/openai/lib/responses_websocket/README.md).
- **Cache observability and prewarming:** `prompt_cache_options.comparison_response_id` requests cache diagnostics; `prewarm=True` prepares a cache without generating output. These could extend `responses_cache.py`, but prewarming needs a separate operation rather than silently changing an ordinary turn. [3.9.0](https://github.com/openai/openai-python/releases/tag/v3.9.0), [3.15.0](https://github.com/openai/openai-python/releases/tag/v3.15.0).
- **Compaction progress:** `response.compaction.compacting` could feed fast-agent's progress events in `responses_streaming.py`. Upgrading alone does not display this progress. [3.15.0](https://github.com/openai/openai-python/releases/tag/v3.15.0).
- **Immediate fixes:** Responses parsing memory leaks, structured parsing of commentary, null output handling and stream indices. These benefit the SDK streaming path, but do not justify removing safeguards from our custom WebSocket path. [3.14.0](https://github.com/openai/openai-python/releases/tag/v3.14.0), [3.14.1](https://github.com/openai/openai-python/releases/tag/v3.14.1), [3.16.2](https://github.com/openai/openai-python/releases/tag/v3.16.2).
- **Compatibility:** function-arguments-done events no longer declare `name`; tool names should come from output items/tool state. Stream transport failures are normalized to `APITimeoutError`/`APIConnectionError`. Numeric error codes are now normalized to strings: the retry regression test now checks fast-agent's classification, not the SDK's internal representation. WebSocket auth rejects provider-runtime/workload-identity modes. [Event change](https://github.com/openai/openai-python/commit/2a98f6a), [transport errors](https://github.com/openai/openai-python/commit/d7c41ef), [auth](https://github.com/openai/openai-python/commit/3b865af).

### Anthropic: replay helpers and explicit compaction

- **Replay helpers:** `Message.to_param()` / `BetaMessage.to_param()`, exclusion of `parsed_output` from dumped text blocks, and private partial-tool-JSON storage reduce SDK response-to-request cleanup. Consider using these at the SDK boundary, not replacing fast-agent's cross-provider history representation. Keep the `parsed_output` cleanup in `multipart_converter_anthropic.py` for persisted older histories. [1.5.0](https://github.com/anthropics/anthropic-sdk-python/releases/tag/v1.5.0).
- **Compaction:** beta `compaction={"type": "summarize"}` returns a signed compaction block alone, suitable for replacing summarized messages. Integrating this belongs in the harness/history lifecycle; preserve the signature verbatim. The SDK tool runner's `compact_before_next_turn()` is not a replacement for fast-agent's own tool loop. [API contract](https://github.com/anthropics/anthropic-sdk-python/commit/8689179), [1.7.0](https://github.com/anthropics/anthropic-sdk-python/releases/tag/v1.7.0).
- **Compatibility:** credential files accessible by group/others are rejected; async credential token providers are supported; retry behavior changes; `usage.iterations[].model` becomes nullable. The SDK's Bedrock eventstream fix does not apply to fast-agent's separate boto3 provider. [1.5.0](https://github.com/anthropics/anthropic-sdk-python/releases/tag/v1.5.0), [1.6.0](https://github.com/anthropics/anthropic-sdk-python/releases/tag/v1.6.0), [1.7.0](https://github.com/anthropics/anthropic-sdk-python/releases/tag/v1.7.0).

### Google GenAI

New environment copying/file transfers, credential APIs, retrieval interaction steps and Live turn completion do not directly simplify our `aio.models.generate_content_stream` integration. The automatic-function-calling budget fix does not replace fast-agent's own tool execution limits. No runtime refactor recommended in this upgrade. [2.23.0](https://github.com/googleapis/python-genai/releases/tag/v2.23.0), [2.24.0](https://github.com/googleapis/python-genai/releases/tag/v2.24.0).

### Hugging Face Hub

The already-selected 1.32.0 preserves sandbox terminal results. New limits reject captured command output above 64 MiB and `files.read()` above 512 MiB. Review `huggingface_sandbox_environment.py` for opting out of redundant capture where callbacks already spool output. Server-issued process IDs improve `SandboxProcess.kill()`, but do not replace our process-group/descendant cancellation semantics. Shared Xet blobs benefit privacy-model downloads automatically; stricter upload-path/repository-ID validation may affect trace exports. [1.32.0](https://github.com/huggingface/huggingface_hub/releases/tag/v1.32.0).

### Other provider dependencies

- **boto3:** updates through 1.43.98 principally affect services other than our Bedrock `converse`/`converse_stream` integration. AgentCore changes are not Converse capabilities; retain our structured-output/tool fallbacks. [Changelog](https://github.com/boto/boto3/blob/1.43.98/CHANGELOG.rst).
- **Azure Identity:** 1.25.3 fixes expired tokens skipping refresh because retry delay took precedence. Relevant to `DefaultAzureCredential`, not API-key authentication. [Changelog](https://github.com/Azure/azure-sdk-for-python/blob/azure-identity_1.25.3/sdk/identity/azure-identity/CHANGELOG.md).
- **TensorZero:** fast-agent uses the OpenAI-compatible endpoint, not the Python gateway SDK directly. The 2026.6.0 release fixes filesystem-read/SSRF exposure through `/internal/object_storage` in the **gateway**. Updating the Python dependency does not patch a running gateway. Verify deployed gateway versions; the example uses an unversioned `tensorzero/gateway` image and an existing local image/container may be stale. [Security advisory](https://github.com/tensorzero/tensorzero/security/advisories/GHSA-824w-x939-6cmc).

## Scope of validation

Validation completed with the upgraded environment:

- `uv run scripts/format.py --check`, `uv run scripts/lint.py`, `uv run scripts/typecheck.py`: passed.
- `uv lock --check` and `git diff --check`: passed.
- The initial unit run exposed a pre-existing readiness race in `test_durable_process_stop_is_file_backed_and_idempotent` (empty output before stop), also reproduced with previous provider SDK versions. The subsequent dependency refresh fixes the test by waiting for child readiness; see [the refresh validation summary](dependency-refresh-2026-09-19.md).
- `uv run pytest tests/integration/llm/test_responses_cache_transport.py -q`: 28 passed, exercising real SDK requests against local SSE/WebSocket simulators without provider credentials.

Live provider authentication, billable inference, deployed TensorZero gateways and optional-extra platform compatibility require separate integration validation. Prioritize transport-error classification, streaming tool events, signed history replay and OpenTelemetry smoke coverage before adopting the larger API changes above.
