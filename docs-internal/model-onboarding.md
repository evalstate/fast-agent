# Model and Provider Onboarding

This checklist defines the evidence expected when adding a model or native
provider to fast-agent. It applies to first-party APIs, OpenAI-compatible APIs,
hosted routers, and new models on existing providers.

The goal is not only to make a request succeed. The integration must normalize
provider behavior at the LLM boundary and prove that the harness can complete
real turns involving streaming, reasoning, tools, structured output, usage, and
supported media.

## 1. Establish documentation and API access

Before implementation:

1. Download the official model guide and API reference in searchable form.
2. Identify the canonical model ID, endpoint, authentication scheme, context
   window, output limit, modalities, and capability pages.
3. Make a minimal authenticated request with the exact model ID.
4. Stop if the official documentation is inaccessible or the basic request
   cannot be triggered.

Record source URLs. When capability pages conflict, prefer the model-specific
guide and current API schema, then verify uncertain behavior live.

## 2. Inventory existing architecture

Inspect:

- `src/fast_agent/llm/provider_types.py`
- `src/fast_agent/llm/model_factory.py`
- `src/fast_agent/llm/model_database.py`
- `src/fast_agent/llm/model_selection.py`
- model-catalog CLI scopes in `src/fast_agent/cli/commands/check_config.py`
- the closest provider implementation under `src/fast_agent/llm/provider/`
- typed settings in `src/fast_agent/config.py`
- `src/fast_agent/llm/provider_key_manager.py`
- relevant tests and provider documentation
- `typesafe.md`

Prefer the narrowest existing provider base that matches the wire protocol.
Keep protocol translation in the LLM/provider layer rather than the agent loop.

## 3. Define the product decisions

Decide and test:

- provider config name and display name;
- explicit model syntax;
- whether a bare model name selects this provider;
- default model when only the provider is selected;
- aliases and backward compatibility for existing aliases;
- API-key environment variable and config precedence;
- endpoint and header overrides;
- whether the model appears in the interactive picker;
- whether the provider is visible and addressable through
  `fast-agent check models <provider>`.

Avoid silently changing established aliases when a new explicit alias can
preserve compatibility.

## 4. Model capability profile

Model metadata must come from official documentation or live evidence:

- context and output limits;
- text and supported attachment MIME types;
- JSON mode: schema, object, or none;
- structured-output/tool policy;
- reasoning response shape and accepted controls;
- streaming mode;
- provider-specific process/tool behavior.

Do not infer multimodality from a shared endpoint. Test attachments only when
the specific model documents that modality. Add a negative metadata assertion
for text-only models.

## 5. Request contract tests

Unit tests should cover behavior at the provider boundary:

- provider parsing and lazy class loading;
- default and overridden model/endpoint;
- config and environment credentials;
- custom headers;
- enabled, disabled, and representative reasoning efforts;
- exact provider extension fields and preservation of unrelated request fields;
- structured-output request shaping;
- model metadata resolution;
- provider model-catalog CLI scope and overview visibility;
- compatibility of old aliases and routes.

Prefer contract and invariant tests over reproducing implementation tables.

## 6. Live end-to-end matrix

Run through the fast-agent harness, not only direct HTTP:

1. Plain streamed text.
2. Reasoning enabled.
3. Reasoning disabled.
4. Function call, tool execution, tool-result continuation, final answer.
5. Structured output with parse and schema validation.
6. Each documented input modality.
7. Usage accounting.
8. Provider-specific features such as caching or tool streaming when relevant.

Prompts should assert stable invariants rather than exact creative wording.

## 7. Provider cache behavior check

When a provider advertises prompt or context caching, verify the effective
behavior rather than assuming another provider's TTL:

1. Use a long, stable prefix and mutate only a short trailing message.
2. Confirm each cache entry with an immediate repeat and provider-reported
   cached-token usage.
3. Use an independent cache key for every delay. Do not probe one key
   successively because reads may refresh or recreate it.
4. Run delay cohorts concurrently to minimize wall-clock time.
5. Use at least two independently keyed replicates per delay.
6. Include a previously unseen negative control.
7. At the end, create and immediately repeat a fresh entry to confirm caching
   is still healthy when older entries miss.

Measure hits from provider usage fields, not latency alone. Record the stable
prefix size, cached-token count, request-start idle time, account/model, date,
and controls.

Report only the observed survival curve or bounds. A miss can reflect
expiration, capacity pressure, routing, deployment, or other service-managed
behavior. Do not populate documented/configurable cache-TTL metadata from an
experiment when the provider does not publish that contract.

If process polling creates model turns, an observed cache window may justify a
provider-specific `process_poll_default_wait_seconds` with safety margin. State
that this is an operational polling default, not a claim about the provider's
TTL, and do not apply it to other routes that were not tested.

## 8. Inspect raw streams

Rendered output is insufficient for a streaming integration. Enable capture:

```bash
FAST_AGENT_LLM_TRACE=1 fast-agent go --no-home \
  --model "<model>" \
  --message "<prompt>"
```

The `stream-debug/` directory contains:

- `*_request.json`: normalized provider arguments;
- `*_chunks.jsonl`: provider-native stream chunks.

Inspect at least:

- transmitted model and provider extensions;
- reasoning fields and their ordering relative to visible content;
- tool-call IDs, names, indices, and argument fragments;
- finish reasons;
- final usage chunks;
- empty or error chunks;
- the follow-up request containing tool results.

Confirm whether optional provider switches are actually required. Do not add
wire parameters only because a capability page mentions them if the normal API
contract already supplies the behavior fast-agent needs.

Trace files may contain sensitive prompts, tool arguments, URLs, IDs, and
outputs. Never commit raw captures.

## 9. Replay fixtures

For a new stream shape, promote sanitized captures into replay fixtures where
the test infrastructure supports that provider family. Replay tests should
verify normalization and emitted events without calling the service.

If the trace family lacks a replay adapter, either add one or document the gap.
At minimum retain focused synthetic stream tests until native replay support is
available.

See `tests/fixtures/llm_traces/README.md` and
`tests/scripts/harvest_llm_traces.py`.

## 10. Documentation and generated resources

Update as applicable:

- provider quick reference;
- a dedicated provider page for a native provider with meaningful behavior;
- provider navigation;
- generated models reference;
- annotated `examples/setup/fast-agent.yaml`;
- `examples/setup/fast-agent.secrets.yaml.example`;
- generated configuration schema;
- aliases generated from the source-of-truth preset/catalog definitions.

Keep `resources/shared/` and setup-resource ownership rules in `AGENTS.md`.
Remove unrelated generator drift from the change.

## 11. Final validation

Required after the final code change:

```bash
uv run scripts/lint.py
uv run scripts/typecheck.py
```

Also run:

- focused provider/model/config/picker tests;
- the complete unit suite for a native provider;
- documentation generation and schema generation;
- `git diff --check`;
- a final live smoke request using the public alias.

Summarize the exact live matrix, trace findings, test counts, and any capability
not tested because the model does not support it.
