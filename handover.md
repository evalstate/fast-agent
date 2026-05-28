# Type-safety cleanup handover

## Context

We updated the typechecker and `uv run scripts/typecheck.py` initially reported ~198 diagnostics. The cleanup goal is not to silence the checker, but to use Python typing to make interfaces tighter and remove dynamic/ambiguous code where possible.

User preferences established during cleanup:

- Use precise Python types where possible.
- Prefer literal annotations where values are assigned from known constants.
- Do not add tests for cases that `ty` already enforces.
- Avoid type-checker golf, broad casts, and unnecessary compatibility shims.
- No need to preserve backwards compatibility for stale internal compatibility paths when current runtime/SDK behavior is clear.

Standard validation commands:

```bash
uv run scripts/lint.py
uv run scripts/typecheck.py
```

Focused checks used throughout:

```bash
uv run ty check <files-or-dirs>
uv run pytest <focused-tests>
```

## Current full typecheck state

Latest full typecheck:

```text
uv run scripts/typecheck.py
Found 41 diagnostics
```

Latest lint:

```text
uv run scripts/lint.py
All checks passed!
```

The baseline moved from ~198 initially to 41. At the start of the continuation captured below it was 179, and after the prior handover it was 157.

> Note: the older "Remaining diagnostic clusters" section below is retained for historical context. A current, superseding plan is appended at the end under "Latest continuation update".

## Work completed before this continuation

### 1. Executor contract tightened

Files changed:

- `src/fast_agent/core/executor/executor.py`
- `tests/unit/fast_agent/core/test_asyncio_executor.py`

Key decisions:

- `ExecutorTask[R] = Awaitable[R] | Callable[..., R]`.
- `execute()` accepts sync callables and awaitable objects.
- `execute()` intentionally rejects bare coroutine functions, awaitables plus kwargs, and sync callables returning awaitables.
- `map()` supports sync and async mapper functions.
- `execute_streaming()` is now a regular method returning an async generator/iterator.
- Removed unused/contradictory `validate_task()` and no-op `execution_context()`.
- Implemented `timeout_seconds`.
- Removed unused `retry_policy`.
- Made `ExecutorConfig` closed with `extra="forbid"`.
- Added validation for non-positive `max_concurrent_activities` and `timeout_seconds`.
- Fixed semaphore behavior in `map()` via `_activity_limit()` / `execute()`.
- Streaming cleanup now cancels and drains pending tasks when the async generator is closed early.

Validation at the time:

```text
uv run pytest tests/unit/fast_agent/core/test_asyncio_executor.py
16 passed

uv run ty check src/fast_agent/core/executor tests/unit/fast_agent/core/test_asyncio_executor.py
All checks passed!

uv run scripts/lint.py
All checks passed!
```

### 2. Reasoning effort type widening clarified

File changed:

- `src/fast_agent/llm/reasoning_effort.py`

Change:

```python
values: list[str] = list(spec.allowed_efforts or EFFORT_LEVELS)
```

Reason: `available_reasoning_values()` returns display/user values as `list[str]`; appending `"off"` intentionally widens strict effort literals to strings.

### 3. Streaming protocol signature cleanup

Files changed:

- `src/fast_agent/ui/streaming.py`
- indirectly cleared diagnostics in `src/fast_agent/ui/console_display.py`

Aligned implementation parameter names with `StreamingHandle` protocol (`chunk`, `message`, `event_type`, etc.). `ty` treats keyword-callable parameter names as part of protocol compatibility.

### 4. `LoggerTextIO` cleanup

File changed:

- `src/fast_agent/mcp/logger_textio.py`

Reworked `LoggerTextIO` to subclass `io.StringIO` rather than `typing.TextIO`/`io.TextIOBase` with an incompatible `write` override.

### 5. ACP status literals

Files changed:

- `src/fast_agent/acp/slash/handlers/mcp.py`
- `src/fast_agent/acp/slash/handlers/skills.py`
- `src/fast_agent/acp/slash_commands.py`
- `publish/hf-inference-acp/src/hf_inference_acp/agents.py`

Added/reused:

```python
ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]
```

Removed old `type: ignore[arg-type]` comments on `status=`.

### 6. MCP transport event literals

Files changed:

- `src/fast_agent/mcp/sse_tracking.py`
- `src/fast_agent/mcp/stdio_tracking_simple.py`
- `src/fast_agent/mcp/streamable_http_tracking.py`

Reused existing `EventType = Literal[...]` alias for event emitters and removed old `type: ignore[arg-type]` comments.

### 7. Streamable HTTP reconnection override fixed

File changed:

- `src/fast_agent/mcp/streamable_http_tracking.py`

Renamed custom `_handle_reconnection(...)` to `_handle_reconnection_for_channel(...)` because the subclass signature did not match the MCP SDK base method.

### 8. URL elicitation payload typing in `llm_agent.py`

File changed:

- `src/fast_agent/agents/llm_agent.py`

Reused existing dataclasses from `fast_agent.mcp.url_elicitation_required`:

- `URLElicitationRequiredDisplayPayload`
- `URLElicitationDisplayItem`

Boundary JSON parsing/narrowing happens once; display code now operates on typed dataclasses.

## Work completed in this continuation

### 9. `smart_agent.py` MCP capability protocol

File changed:

- `src/fast_agent/agents/smart_agent.py`

Problem:

```text
expected _McpCapableAgent, found AgentProtocol
protocol member attach_mcp_server is not defined on AgentProtocol
```

Fix:

- Made `_McpCapableAgent` `@runtime_checkable`.
- Replaced dynamic `getattr`/`type: ignore` narrowing with `isinstance(agent, _McpCapableAgent)`.

Design:

- Runtime MCP attach/detach/list is an explicit optional capability, not part of `AgentProtocol`.
- This follows `typesafe.md`: use `isinstance` against concrete/protocol capability instead of `hasattr`/raw casts for core agent capabilities.

Validation:

```text
uv run ty check src/fast_agent/agents/smart_agent.py
All checks passed!

uv run scripts/lint.py
All checks passed!
```

### 10. `direct_decorators.py` `.tool` and overload cleanup

File changed:

- `src/fast_agent/core/direct_decorators.py`

Problems cleared:

```text
func.tool = _agent_tool
invalid overload
```

Design:

- Preserved intended public/IDE behavior:

```python
@fast.agent(...)
async def writer(): ...

@writer.tool
def helper(): ...
```

- Added a narrow helper to attach the scoped tool decorator while preserving function identity:

```python
def _attach_scoped_tool_decorator(
    func: Callable[P, Coroutine[Any, Any, R]],
    tool: ScopedToolDecoratorProtocol,
) -> DecoratedToolCapableAgentProtocol[P, R]:
    decorated = cast("DecoratedToolCapableAgentProtocol[P, R]", func)
    decorated.tool = tool
    return decorated
```

- Tool decorator overloads now preserve the decorated function signature via separate `ToolP` / `ToolR` type variables.
- Removed the problematic positional-only marker from the `DecoratorMixin.tool` method overload/implementation relationship.

Validation:

```text
uv run ty check src/fast_agent/core/direct_decorators.py
All checks passed!

uv run pytest tests/unit/core/test_tool_decorator.py
28 passed

uv run scripts/lint.py
All checks passed!
```

### 11. Dynamic `fastagent.py` config/app attributes removed

Files changed:

- `src/fast_agent/config.py`
- `src/fast_agent/core/direct_factory.py`
- `src/fast_agent/core/fastagent.py`

Problems cleared:

```text
cfg.model_source = ...
cfg.cli_model_override = ...
self.app._registered_tools = ...
```

Changes:

- Added real `Settings` fields:

```python
model_source: str | None = None
cli_model_override: str | None = None
```

- Narrowed CLI `getattr()` values to `str | None` before assignment.
- Removed dynamic injection of `_registered_tools` into `Core`/`AgentApp`.
- Added explicit factory dependency:

```python
@dataclass(frozen=True)
class AgentBuildContext:
    ...
    global_function_tools: Sequence[FunctionTool]
```

- `create_agents_in_dependency_order(..., global_function_tools=...)` now passes global `@fast.tool` tools explicitly into the direct factory.
- `_resolve_function_tools_with_globals()` reads `build_ctx.global_function_tools` rather than `getattr(build_ctx.app_instance, "_registered_tools", None)`.

Design note:

Global tools are needed during agent construction, before the runtime `AgentApp` exists. Passing them into the factory is a better boundary than attaching a private attribute to `Core` or `AgentApp`.

Validation:

```text
uv run ty check src/fast_agent/config.py src/fast_agent/core/direct_factory.py src/fast_agent/core/fastagent.py tests/unit/fast_agent/core/test_model_namespace_resolution.py
All checks passed!

uv run pytest tests/unit/core/test_tool_decorator.py tests/unit/fast_agent/core/test_agents_as_tools_function_tools.py tests/unit/fast_agent/core/test_model_namespace_resolution.py
48 passed

uv run scripts/lint.py
All checks passed!
```

### 12. Local mechanical narrowing/signature fixes

Files changed:

- `src/fast_agent/llm/internal/playback.py`
- `src/fast_agent/ui/stream_segments.py`
- `src/fast_agent/mcp/mcp_agent_client_session.py`
- `src/fast_agent/mcp/oauth_client.py`
- `src/fast_agent/llm/model_overlays.py`
- `src/fast_agent/session/snapshot.py`

Changes:

- `PlaybackLLM.generate()` now matches the base `FastAgentLLM.generate()` override signature:

```python
async def generate(
    self,
    messages: list[PromptMessageExtended],
    request_params: RequestParams | None = None,
    tools: list[Tool] | None = None,
) -> PromptMessageExtended:
```

- `StreamSegmentBuffer._base_kind` explicitly annotated as `SegmentKind`.
- `MCPAgentClientSession` elicitation mode checks now use direct `is not None` narrowing.
- OAuth client metadata URL is narrowed before `create_client_info_from_metadata_url(...)`.
- Model overlay `json_mode` is narrowed to exact literals without ignore:

```python
json_mode = "schema" if existing.json_mode == "schema" else "object"
```

- Session snapshot compatibility metadata now uses recursive `JsonValue` consistently, including `history_map: dict[str, JsonValue]`.
- Removed an unused `_meta` ignore in `mcp_agent_client_session.py`.

Validation:

```text
uv run ty check src/fast_agent/llm/internal/playback.py src/fast_agent/ui/stream_segments.py src/fast_agent/mcp/oauth_client.py src/fast_agent/llm/model_overlays.py src/fast_agent/session/snapshot.py
All checks passed!

uv run pytest tests/unit/fast_agent/llm/test_playback.py tests/unit/fast_agent/session/test_snapshot.py tests/unit/fast_agent/ui/test_stream_segments_freezing.py
22 passed

uv run scripts/lint.py
All checks passed!
```

Note: `tests/unit/fast_agent/llm/test_model_overlays.py` currently has overlay-discovery failures in this environment. They appear unrelated to the `json_mode` export change; failures are around overlay aliases not being discovered/resolved.

### 13. `function_tool_loader.py` `__signature__` helper

File changed:

- `src/fast_agent/tools/function_tool_loader.py`

Problem:

```text
async_wrapped.__signature__ = inspect.signature(fn)
sync_wrapped.__signature__ = inspect.signature(fn)
```

Fix:

- Added a small protocol/helper:

```python
class _SignatureWritable(Protocol):
    __signature__: inspect.Signature


def _set_signature(wrapper: Callable[..., Any], source: Callable[..., Any]) -> None:
    signature_wrapper = cast("_SignatureWritable", wrapper)
    signature_wrapper.__signature__ = inspect.signature(source)
```

Design:

- Same pattern as `.tool`: runtime callable metadata is intentional, but the cast is localized and documents the contract.

Validation:

```text
uv run ty check src/fast_agent/tools/function_tool_loader.py
All checks passed!

uv run pytest tests/unit/fast_agent/tools/test_function_tool_loader.py
6 passed
```

### 14. MCP SDK `_meta` / `meta` construction cleanup

Files changed:

- `src/fast_agent/agents/tool_agent.py`
- `src/fast_agent/mcp/mcp_agent_client_session.py`
- `src/fast_agent/tools/apply_patch_tool.py`
- `tests/unit/fast_agent/commands/test_tool_summaries.py`

Investigation result:

Current installed MCP SDK constructors accept `_meta=` and expose `.meta`:

```text
Tool(..., _meta: dict[str, Any] | None = None, ...)
CallToolResult(..., _meta: dict[str, Any] | None = None, ...)
ReadResourceRequestParams(..., _meta: RequestParams.Meta | None = None, ...)
GetPromptRequestParams(..., _meta: RequestParams.Meta | None = None, ...)
CallToolRequestParams(..., _meta: RequestParams.Meta | None = None, ...)
```

Public SDK method layer:

```text
ClientSession.call_tool(..., *, meta: dict[str, Any] | None = None)
ClientSession.read_resource(uri)
ClientSession.get_prompt(name, arguments=None)
```

Rule now used:

- public method/adapter layer: `meta=`;
- MCP model constructor/wire boundary: `_meta=`;
- model instance access: `.meta`.

Changes:

- Replaced `meta=` constructor arguments with `_meta=` for SDK models:
  - `CallToolResult(..., _meta=result.meta)`
  - `ReadResourceRequestParams(..., _meta=meta_obj)`
  - `GetPromptRequestParams(..., _meta=meta_obj)`
  - `Tool(..., _meta={...})`
- Updated test helper `Tool(..., _meta=meta or {})`.

Validation:

```text
uv run ty check src/fast_agent/tools/function_tool_loader.py src/fast_agent/agents/tool_agent.py src/fast_agent/mcp/mcp_agent_client_session.py src/fast_agent/tools/apply_patch_tool.py tests/unit/fast_agent/commands/test_tool_summaries.py
All checks passed!

uv run pytest tests/unit/fast_agent/tools/test_function_tool_loader.py tests/unit/fast_agent/commands/test_tool_summaries.py
11 passed

uv run scripts/lint.py
All checks passed!
```

### 15. Removed stale `_meta` public-adapter compatibility branch

Files changed:

- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/mcp/mcp_agent_client_session.py`
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_metadata_passthrough.py`

Investigation:

No import-time monkeypatch of MCP SDK `model_fields`/aliases was found. The stale compatibility code was mainly in `_execute_on_server()`:

```python
if method_name == "call_tool":
    kwargs["meta"] = metadata
else:
    kwargs["_meta"] = metadata
```

This branch existed because older SDK behavior did not expose consistent `meta` support. Current SDK exposes `meta=` publicly for `call_tool`; fast-agent still needs local adapters for `read_resource` and `get_prompt` because current SDK convenience methods do not expose metadata for those operations.

Changes:

- `_execute_on_server()` now consistently injects public adapter metadata as:

```python
kwargs["meta"] = metadata
```

- `MCPAgentClientSession.read_resource()` now accepts `meta=` instead of `_meta=`:

```python
async def read_resource(
    self,
    uri: AnyUrl | str,
    *,
    meta: dict[str, Any] | RequestParams.Meta | None = None,
) -> ReadResourceResult:
```

- `MCPAgentClientSession.get_prompt()` now accepts `meta=` instead of `_meta=`:

```python
async def get_prompt(
    self,
    name: str,
    arguments: dict[str, str] | None = None,
    *,
    meta: dict[str, Any] | RequestParams.Meta | None = None,
) -> GetPromptResult:
```

- Both methods still construct request params with `_meta=...`, which is the current SDK model boundary.
- `MCPAgentClientSession.call_tool()` now delegates to `super().call_tool(..., meta=merged_meta)` after merging fast-agent experimental-session metadata. This still uses fast-agent's overridden `send_request()` because the SDK method calls `self.send_request(...)`.
- Metadata passthrough test now expects `meta` rather than `_meta` for read-resource adapter kwargs.

Validation:

```text
uv run scripts/lint.py
All checks passed!

uv run pytest tests/unit/fast_agent/mcp/test_mcp_aggregator_metadata_passthrough.py
4 passed

uv run scripts/typecheck.py
Found 157 diagnostics
```

No diagnostic count reduction from this cleanup; it removed stale compatibility structure and kept the baseline stable.

## Files modified so far

Known modified files across the full cleanup session:

- `src/fast_agent/core/executor/executor.py`
- `tests/unit/fast_agent/core/test_asyncio_executor.py`
- `src/fast_agent/llm/reasoning_effort.py`
- `src/fast_agent/ui/streaming.py`
- `src/fast_agent/mcp/logger_textio.py`
- `src/fast_agent/acp/slash/handlers/mcp.py`
- `src/fast_agent/acp/slash/handlers/skills.py`
- `src/fast_agent/acp/slash_commands.py`
- `publish/hf-inference-acp/src/hf_inference_acp/agents.py`
- `src/fast_agent/mcp/sse_tracking.py`
- `src/fast_agent/mcp/stdio_tracking_simple.py`
- `src/fast_agent/mcp/streamable_http_tracking.py`
- `src/fast_agent/agents/llm_agent.py`
- `src/fast_agent/agents/smart_agent.py`
- `src/fast_agent/core/direct_decorators.py`
- `src/fast_agent/config.py`
- `src/fast_agent/core/direct_factory.py`
- `src/fast_agent/core/fastagent.py`
- `src/fast_agent/llm/internal/playback.py`
- `src/fast_agent/ui/stream_segments.py`
- `src/fast_agent/mcp/mcp_agent_client_session.py`
- `src/fast_agent/mcp/oauth_client.py`
- `src/fast_agent/llm/model_overlays.py`
- `src/fast_agent/session/snapshot.py`
- `src/fast_agent/tools/function_tool_loader.py`
- `src/fast_agent/agents/tool_agent.py`
- `src/fast_agent/tools/apply_patch_tool.py`
- `tests/unit/fast_agent/commands/test_tool_summaries.py`
- `src/fast_agent/mcp/mcp_aggregator.py`
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_metadata_passthrough.py`

## Remaining diagnostic clusters / recommended next targets

### A. MCP aggregator production diagnostics

Current production diagnostics include:

```text
src/fast_agent/mcp/mcp_aggregator.py:2452 result.namespaced_name = ...
src/fast_agent/mcp/mcp_aggregator.py:2456 result.arguments = ...
src/fast_agent/mcp/mcp_aggregator.py:2634 ListPromptsResult = await _execute_on_server(... error_factory=lambda _: None)
src/fast_agent/mcp/mcp_aggregator.py:2673 ListPromptsResult = await _execute_on_server(... error_factory=lambda _: None)
src/fast_agent/mcp/mcp_aggregator.py:2901 ListResourceTemplatesResult = await _execute_on_server(... error_factory=lambda _: None)
```

Recommended approach:

1. For `namespaced_name` / `arguments`, avoid dynamically attaching app-specific attributes to SDK `GetPromptResult`. Prefer a typed wrapper or side-channel structure. Inspect callers first to see who reads these attributes.
2. For `_execute_on_server(... error_factory=lambda _: None)`, either:
   - make the local result variable optional and explicitly handle `None`, if `None` is a real result; or
   - return an empty `ListPromptsResult` / `ListResourceTemplatesResult` from the error factory if the code assumes a concrete result.

This is the best next production-runtime cluster.

### B. Provider SDK-specific production diagnostics

Examples:

```text
src/fast_agent/llm/provider/anthropic/llm_anthropic.py:1066 cache_ttl str vs Literal["5m", "1h"]
src/fast_agent/llm/provider/bedrock/bedrock_utils.py boto3 = None
src/fast_agent/llm/provider/bedrock/bedrock_utils.py sorted(set[str | None])
src/fast_agent/llm/provider/bedrock/llm_bedrock.py boto3/exception fallback aliases
src/fast_agent/llm/provider/google/llm_google_native.py GenerateContentConfig.tools assignment
```

Recommended approach:

- Anthropic: use literal typing for `cache_ttl` at the config/source boundary. Since this repo has the `claude-api` skill available, consider loading it before changing Anthropic SDK code if prompt caching behavior is touched.
- Bedrock: avoid assigning `None` to imported module names/classes. Prefer explicit optional dependency loader variables or helper functions.
- Google: inspect SDK `GenerateContentConfig.tools` accepted types and convert/narrow rather than ignoring.

### C. HF inference typed model/test cluster

Diagnostics around `provider_id` / `is_model_author` unknown arguments in:

- `tests/integration/acp/test_set_model_validation.py`
- `tests/unit/fast_agent/llm/test_hf_inference_lookup_unit.py`

Likely direction:

- Inspect `src/fast_agent/llm/hf_inference_lookup.py` model aliases. It already has pydantic aliases:

```python
provider_id: str = Field(default="", alias="providerId")
is_model_author: bool = Field(default=False, alias="isModelAuthor")
```

- Tests may need to use canonical constructor aliases (`providerId`, `isModelAuthor`) or model config may need `populate_by_name=True` if Pythonic field names are intentionally supported.

### D. Test fake/protocol cleanup

Large remaining cluster is test-only and should not weaken production types.

Examples:

- `MCPConnectionManager` fake method assignment.
- Fake task groups assigned to `TaskGroup` attrs.
- `PromptSession` fake not matching `PromptSession`.
- `MarkdownTruncator` fake not matching `MarkdownTruncator`.
- `ShellRuntime.runtime_info` / `working_directory` monkeypatching.
- `raw_fn._fast_tool_name` / `_fast_tool_description` dynamic test assignment.
- Anthropic typed dict test access.

Recommended approach:

- Prefer protocol-compliant test doubles or constructor injection.
- Avoid monkeypatching instance methods where possible.
- Do not weaken production types to satisfy tests.
- For dynamic callable metadata tests, use a small typed helper/protocol as done for `.tool` and `__signature__`, or test through the public config path (`ScopedFunctionToolConfig`) rather than mutating arbitrary function attributes.

## Suggested next action

Best next production target:

1. Inspect `mcp_aggregator.py` prompt/resource list code around lines ~2420-2920.
2. Find all readers of dynamically attached `namespaced_name` and `arguments`.
3. Replace dynamic mutation with a typed result/wrapper or explicit mapping.
4. Then fix `_execute_on_server` error-factory `None` typing by choosing optional handling vs concrete empty SDK result objects.

Expected payoff: around 5 production diagnostics, and it removes another dynamic SDK-model mutation pattern.


---

## Latest continuation update — current state 41 diagnostics

This section supersedes the older "Remaining diagnostic clusters" section above.

### Current validation state

Latest validation after the most recent cleanup pass:

```text
uv run scripts/lint.py
All checks passed!

uv run scripts/typecheck.py
Found 41 diagnostics
```

The focused tests run for recently touched files passed; key grouped runs included:

```text
15 passed  # MCP prompt metadata / prompt integration
112 passed # Anthropic/Bedrock/Google focused provider tests
404 passed # provider suite after apply_patch custom-tool fix
13 passed  # HF inference lookup and ACP validation
3 passed   # multimodal mixed-content e2e smoke
65 passed  # tool decorator / ACP model / model handler tests
90 passed  # UI/display/session/fake cleanup focused batch
147 passed # Responses helpers/websocket focused batch
```

### Work completed after the previous handover

#### 16. MCP prompt metadata and aggregator cleanup

Files changed:

- `src/fast_agent/mcp/prompt_metadata.py` (new)
- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/agents/mcp_agent.py`
- `src/fast_agent/agents/llm_decorator.py`

Changes:

- Removed dynamic mutation of SDK `GetPromptResult` instances:
  - `result.namespaced_name = ...`
  - `result.arguments = ...`
- Added typed metadata helpers:

```python
with_prompt_metadata(...)
prompt_display_name(...)
```

- Stores fast-agent display metadata in `GetPromptResult.meta`.
- Replaced `getattr(prompt_result, "namespaced_name", ...)` readers with `prompt_display_name(...)`.
- Replaced `error_factory=lambda _: None` for concrete list operations with empty SDK results:
  - `ListPromptsResult(prompts=[])`
  - `ListResourceTemplatesResult(resourceTemplates=[])`

Validation:

```text
uv run ty check src/fast_agent/mcp/mcp_aggregator.py src/fast_agent/mcp/prompt_metadata.py src/fast_agent/agents/mcp_agent.py src/fast_agent/agents/llm_decorator.py
All checks passed!

uv run pytest tests/unit/fast_agent/mcp/test_mcp_aggregator_metadata_passthrough.py tests/integration/prompt-server/test_prompt_server_integration.py tests/integration/api/test_prompt_listing.py
15 passed
```

#### 17. Provider production typing cleanup

Files changed:

- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`
- `src/fast_agent/llm/provider/bedrock/bedrock_utils.py`
- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py`
- `src/fast_agent/llm/provider/google/llm_google_native.py`

Changes:

- Anthropic:
  - Added `CacheTTL = Literal["5m", "1h"]`.
  - `_get_cache_ttl()` now returns `CacheTTL`.
  - `_apply_cache_control_to_message(..., ttl: CacheTTL = "5m")`.
- Bedrock:
  - Replaced `boto3 = None` / exception class fallback assignments with typed optional loader variables and `_require_boto3()`.
  - Added `_BOTOCORE_ERRORS` and `_NO_CREDENTIALS_ERROR` for typed exception handling.
  - Filtered provider names before `sorted(...)`.
- Google:
  - Used `types.ToolListUnion` for `GenerateContentConfig.tools` assignment rather than an ignored invariant `list[types.Tool]` assignment.

Validation:

```text
uv run ty check src/fast_agent/llm/provider/anthropic/llm_anthropic.py src/fast_agent/llm/provider/bedrock/bedrock_utils.py src/fast_agent/llm/provider/bedrock/llm_bedrock.py src/fast_agent/llm/provider/google/llm_google_native.py
All checks passed!

uv run pytest tests/unit/fast_agent/llm/providers/test_llm_anthropic_caching.py tests/unit/fast_agent/llm/providers/test_multipart_converter_anthropic.py tests/unit/fast_agent/llm/providers/test_bedrock_converter.py tests/unit/fast_agent/llm/providers/test_google_converter.py tests/unit/fast_agent/llm/providers/test_google_stream_replay.py tests/unit/fast_agent/llm/providers/test_google_thinking.py tests/unit/fast_agent/llm/providers/test_llm_google_vertex.py tests/unit/fast_agent/llm/providers/test_multipart_converter_google.py
112 passed
```

#### 18. OpenAI Responses `apply_patch` custom tool regression fixed

File changed:

- `src/fast_agent/tools/apply_patch_tool.py`

Investigation:

- `build_apply_patch_tool()` correctly constructs `Tool(..., _meta={...})`.
- Current MCP model instances expose `.meta`.
- The helper had been looking in a by-alias dump for `"meta"`, but by-alias dumps use `"_meta"`.
- This caused `apply_patch` to serialize as a normal OpenAI Responses `function` tool instead of a `custom` grammar tool.

Final fix:

```python
meta_source = tool.meta
```

No dump/model-extra fallback is retained; `.meta` is the correct current SDK instance boundary.

Validation:

```text
uv run pytest tests/unit/fast_agent/llm/providers/test_responses_helpers.py::test_build_response_args_serializes_apply_patch_as_custom_tool ...
4 passed

uv run pytest tests/unit/fast_agent/llm/providers tests/unit/fast_agent/llm/test_sampling_converter.py
404 passed
```

#### 19. HF inference lookup now uses `huggingface_hub`

Files changed:

- `src/fast_agent/llm/hf_inference_lookup.py`
- `tests/unit/fast_agent/llm/test_hf_inference_lookup_unit.py`
- `tests/integration/acp/test_set_model_validation.py`

Research:

- Local `../huggingface_hub` source has first-class typed support:

```python
HfApi().model_info(model_id, expand=["inferenceProviderMapping"])
```

- This returns `ModelInfo.inference_provider_mapping: list[InferenceProviderMapping] | None`.
- `InferenceProviderMapping` exposes `provider`, `provider_id`, `status`, and `task`.

Changes:

- Replaced manual `httpx` request/parsing to `https://huggingface.co/api/models/...` with `HfApi().model_info(...)` run via `asyncio.to_thread(...)`.
- Mapped typed Hub entries into fast-agent `InferenceProvider`.
- Removed unused/stale `is_model_author` / `isModelAuthor`; current `huggingface_hub` does not expose it and fast-agent never read it.
- Tests now construct with API aliases (`providerId`) where appropriate.

Validation:

```text
uv run ty check src/fast_agent/llm/hf_inference_lookup.py tests/unit/fast_agent/llm/test_hf_inference_lookup_unit.py tests/integration/acp/test_set_model_validation.py
All checks passed!

uv run pytest tests/unit/fast_agent/llm/test_hf_inference_lookup_unit.py tests/integration/acp/test_set_model_validation.py -q
13 passed
```

#### 20. Multimodal e2e test made honest and typed

File changed:

- `tests/e2e/multimodal/test_openai_tool_validation_fix.py`

Assessment:

- The test claimed to validate OpenAI API request validation, but it only called `agent.call_tool(...)` and never called `generate()` or built/sent an OpenAI request.
- Provider parametrization therefore did not pull its weight and caused avoidable provider setup failures (e.g. Azure config required).

Changes:

- Removed provider parametrization and used `model="passthrough"`.
- Updated docstrings to describe this as an MCP mixed-content tool-result smoke test.
- Replaced `hasattr(..., "type")` and object iteration with concrete MCP types:
  - `TextContent`
  - `ImageContent`
  - `CallToolResult`
- Added `_require_tool_result(...)` for `asyncio.gather(..., return_exceptions=True)` narrowing.

Validation:

```text
uv run ty check tests/e2e/multimodal/test_openai_tool_validation_fix.py
All checks passed!

uv run pytest tests/e2e/multimodal/test_openai_tool_validation_fix.py -q
3 passed
```

#### 21. Test fake/protocol cleanup batch

Files changed:

- `tests/unit/core/test_tool_decorator.py`
- `tests/unit/fast_agent/acp/test_slash_commands_models.py`
- `tests/unit/fast_agent/commands/test_model_handler.py`
- `tests/unit/fast_agent/agents/test_llm_agent_streaming_handoff.py`
- `tests/unit/fast_agent/commands/test_shell_cwd_policy.py`
- `tests/unit/fast_agent/core/test_acp_startup_mode.py`
- `tests/unit/fast_agent/ui/test_streaming_mode_switch.py`
- `tests/unit/fast_agent/ui/test_mcp_display.py`
- `tests/unit/fast_agent/core/test_agent_card_watch.py`
- `tests/unit/fast_agent/llm/provider/anthropic/test_tool_id_sanitization.py`
- `tests/unit/fast_agent/llm/providers/test_responses_helpers.py`
- `tests/unit/fast_agent/llm/providers/test_responses_websocket.py`

Patterns used:

- Replaced direct dynamic callable metadata mutation with a local protocol/helper in `test_tool_decorator.py`.
- Used real enum values (`Provider.CODEX_RESPONSES`) instead of provider strings.
- Annotated optional fake attributes as optional where tests intentionally assign `None`.
- Used real `SkillManifest` instead of `object()`.
- Replaced invalid model factory `lambda: None` with a type-correct unused factory that raises if called.
- Replaced Rich private `_width`/`_height` mutation with constructing a test `Console(width=..., height=...)` and restoring the original shared console.
- Used `monkeypatch.setattr(...)` and real `Session`/`SessionInfo` for agent-card watch hydration.
- Narrowed Anthropic converted content blocks through `dict[str, object]` in tests.
- Localized test-only dynamic assignment for Responses websocket managers behind `_set_ws_connection_manager(...)`.

Validation examples:

```text
uv run pytest tests/unit/core/test_tool_decorator.py tests/unit/fast_agent/acp/test_slash_commands_models.py tests/unit/fast_agent/commands/test_model_handler.py -q
65 passed

uv run pytest tests/unit/fast_agent/agents/test_llm_agent_streaming_handoff.py tests/unit/fast_agent/commands/test_shell_cwd_policy.py tests/unit/fast_agent/core/test_acp_startup_mode.py tests/unit/fast_agent/ui/test_streaming_mode_switch.py tests/unit/fast_agent/ui/test_mcp_display.py tests/unit/fast_agent/core/test_agent_card_watch.py tests/unit/fast_agent/llm/provider/anthropic/test_tool_id_sanitization.py -q
90 passed

uv run pytest tests/unit/fast_agent/llm/providers/test_responses_helpers.py tests/unit/fast_agent/llm/providers/test_responses_websocket.py -q
147 passed, 1 warning
```

### Current remaining diagnostic clusters / recommended next targets

Run this to see the current leading diagnostics:

```bash
uv run scripts/typecheck.py 2>&1 | sed -n '1,260p'
```

At the latest checkpoint, full typecheck reports 41 diagnostics. The visible leading clusters are:

#### A. MCP session/client/aggregator test fakes

Examples from latest output:

```text
tests/unit/fast_agent/mcp/test_elicitation_handlers.py
  redundant cast to Any

tests/unit/fast_agent/mcp/test_mcp_agent_client_session_sessions.py
  invalid override of send_request

tests/unit/fast_agent/mcp/test_mcp_aggregator_metadata_passthrough.py
  aggregator.experimental_sessions = recorder

tests/unit/fast_agent/mcp/test_mcp_aggregator_nonpersistent.py
  error_data.code = None

tests/unit/fast_agent/mcp/test_mcp_aggregator_server_instructions.py
  a2a_module.types = ...

tests/unit/fast_agent/mcp/test_mcp_aggregator_skybridge.py
  a2a_module.types = ...

tests/unit/fast_agent/mcp/test_mcp_connection_manager.py
  manager.launch_server = _fake_launch_server
  client_session_factory=lambda ...: object()
```

Recommended fixes:

1. `test_mcp_agent_client_session_sessions.py`
   - Update fake subclass `send_request` overrides to match `MCPAgentClientSession.send_request` exactly:

```python
async def send_request(
    self,
    request: ClientRequest,
    result_type: type[ReceiveResultT],
    request_read_timeout_seconds: timedelta | None = None,
    metadata: MessageMetadata | None = None,
    progress_callback: ProgressFnT | None = None,
) -> ReceiveResultT:
    ...
```

   - Return specific fake results via local `cast(ReceiveResultT, result)` if needed.

2. `test_mcp_aggregator_metadata_passthrough.py`
   - Avoid assigning arbitrary recorder object to `aggregator.experimental_sessions`.
   - Prefer a small subclass/fake that satisfies `ExperimentalSessionClient`, or localize the test-only assignment through a typed helper if the real class is too heavy.

3. `test_mcp_connection_manager.py`
   - Prefer `monkeypatch.setattr(...)` with a correctly typed async `launch_server` fake.
   - Replace `client_session_factory=lambda ...: object()` with a small fake/cast that returns a `ClientSession`-compatible object. Do not weaken production types.

4. `a2a_module.types` module fakes
   - Use `setattr(a2a_module, "types", a2a_types_module)` or a tiny typed helper rather than direct dynamic module attribute assignment.

5. `error_data.code = None`
   - If covering malformed remote payloads, construct the malformed data at a dict/JSON boundary rather than mutating a typed model field to an invalid value.

#### B. Remaining Responses/MCP/UI fakes after the MCP cluster

After the MCP cluster, rerun full typecheck. The next likely tail mentions:

```text
markdown_truncator: MarkdownTruncator
```

Search with:

```bash
rg "markdown_truncator|MarkdownTruncator|PromptSession|runtime_info|working_directory" tests src
```

Preferred direction:

- Use real objects where cheap.
- Use protocol-compliant fakes where the code only needs a small surface.
- Avoid mutating concrete private attributes or assigning arbitrary objects into concrete-typed fields.

### Suggested next action

Start with the MCP test cluster:

```bash
uv run ty check \
  tests/unit/fast_agent/mcp/test_elicitation_handlers.py \
  tests/unit/fast_agent/mcp/test_mcp_agent_client_session_sessions.py \
  tests/unit/fast_agent/mcp/test_mcp_aggregator_metadata_passthrough.py \
  tests/unit/fast_agent/mcp/test_mcp_aggregator_nonpersistent.py \
  tests/unit/fast_agent/mcp/test_mcp_aggregator_server_instructions.py \
  tests/unit/fast_agent/mcp/test_mcp_aggregator_skybridge.py \
  tests/unit/fast_agent/mcp/test_mcp_connection_manager.py
```

Then run the matching focused tests and standard validation:

```bash
uv run pytest <focused MCP test files> -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

---

## Latest continuation update — clean typecheck

### Current validation state

The type-safety sweep now reaches a clean full typecheck:

```text
uv run scripts/typecheck.py
All checks passed!

uv run scripts/lint.py
All checks passed!
```

Focused tests run for this continuation:

```text
uv run pytest tests/unit/fast_agent/mcp/test_elicitation_handlers.py tests/unit/fast_agent/mcp/test_mcp_agent_client_session_sessions.py tests/unit/fast_agent/mcp/test_mcp_aggregator_metadata_passthrough.py tests/unit/fast_agent/mcp/test_mcp_aggregator_nonpersistent.py tests/unit/fast_agent/mcp/test_mcp_aggregator_server_instructions.py tests/unit/fast_agent/mcp/test_mcp_aggregator_skybridge.py tests/unit/fast_agent/mcp/test_mcp_connection_manager.py -q
78 passed

uv run pytest tests/unit/fast_agent/tools/test_shell_runtime.py tests/unit/fast_agent/ui/test_hash_agent_command.py tests/unit/fast_agent/ui/test_interactive_prompt_resource_mentions.py tests/unit/fast_agent/ui/test_prompt_input.py tests/unit/fast_agent/ui/test_stream_viewport.py -q
49 passed
```

### Work completed in this continuation

#### 22. MCP test fake cleanup

Files changed:

- `tests/unit/fast_agent/mcp/test_elicitation_handlers.py`
- `tests/unit/fast_agent/mcp/test_mcp_agent_client_session_sessions.py`
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_metadata_passthrough.py`
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_nonpersistent.py`
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_server_instructions.py`
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_skybridge.py`
- `tests/unit/fast_agent/mcp/test_mcp_connection_manager.py`

Changes:

- Removed redundant `Any` casts in elicitation handler tests.
- Updated fake `MCPAgentClientSession.send_request(...)` overrides to match the production generic signature.
- Made experimental-session invalidation recorder subclass `ExperimentalSessionClient` instead of assigning an arbitrary object into `aggregator.experimental_sessions`.
- Used `ErrorData.model_construct(code=None, ...)` to represent malformed remote payloads at a pydantic/JSON boundary instead of mutating a typed field to `None`.
- Replaced dynamic module attribute assignment with `setattr(...)` for optional `a2a.types` test stubs.
- Replaced `MCPConnectionManager` method assignment with `monkeypatch.setattr(...)`, exact fake signatures, and typed client-session factory helpers.
- Replaced fake AnyIO task groups with real task groups and subclassed managers for `disconnect_all()` behavior.

#### 23. Final UI/tools fake cleanup

Files changed:

- `tests/unit/fast_agent/tools/test_shell_runtime.py`
- `tests/unit/fast_agent/ui/test_hash_agent_command.py`
- `tests/unit/fast_agent/ui/test_interactive_prompt_resource_mentions.py`
- `tests/unit/fast_agent/ui/test_prompt_input.py`
- `tests/unit/fast_agent/ui/test_stream_viewport.py`

Changes:

- Added `_TestShellRuntime` subclass overriding `runtime_info()` / `working_directory()` instead of assigning methods on instances.
- Used `setattr(...)` to test frozen `HashAgentCommand` immutability without direct read-only property assignment.
- Made local mention fake subclass the mention-agent fake, with `get_resource()` raising if local file resolution accidentally calls agent resource lookup.
- Localized `PromptSession` casting at the fake session factory boundary.
- Made fake markdown truncator subclass `MarkdownTruncator` and match keyword-callable parameter names.

### Remaining work

No `ty` diagnostics remain. Next steps are product/CI oriented:

1. Consider enabling the clean `uv run scripts/typecheck.py` state in CI if not already enforced.
2. Keep future test doubles protocol/concrete-type compatible rather than mutating methods/attributes dynamically.
