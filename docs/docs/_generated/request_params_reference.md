<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

### Available `RequestParams` Fields (Generated)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `task` | `mcp.types.TaskMetadata | None` | `None` |  |
| `meta` | `mcp.types.RequestParams.Meta | None` | `None` |  |
| `messages` | `list[mcp.types.SamplingMessage]` | `[]` |  |
| `modelPreferences` | `mcp.types.ModelPreferences | None` | `None` |  |
| `systemPrompt` | `str | None` | `None` |  |
| `includeContext` | `Literal['none', 'thisServer', 'allServers'] | None` | `None` |  |
| `temperature` | `float | None` | `None` |  |
| `maxTokens` | `int | None` | `None` |  |
| `stopSequences` | `list[str] | None` | `None` |  |
| `metadata` | `dict[str, Any] | None` | `None` |  |
| `tools` | `list[mcp.types.Tool] | None` | `None` |  |
| `toolChoice` | `mcp.types.ToolChoice | None` | `None` |  |
| `model` | `str | None` | `None` |  |
| `use_history` | `bool` | `True` |  |
| `max_iterations` | `int` | `199` |  |
| `parallel_tool_calls` | `bool` | `True` |  |
| `response_format` | `Any | None` | `None` |  |
| `structured_schema` | `dict[str, Any] | None` | `None` |  |
| `structured_tool_policy` | `Literal['auto', 'always', 'defer', 'no_tools']` | `'auto'` |  |
| `template_vars` | `dict[str, Any]` | `{}` |  |
| `mcp_metadata` | `dict[str, Any] | None` | `None` |  |
| `tool_execution_handler` | `Any | None` | `None` |  |
| `emit_loop_progress` | `bool` | `False` |  |
| `tool_result_mode` | `Literal['postprocess', 'passthrough', 'selectable']` | `'postprocess'` |  |
| `batch_context` | `fast_agent.llm.request_params.BatchRequestContext | None` | `None` |  |
| `streaming_timeout` | `float | None` | `120.0` |  |
| `top_p` | `float | None` | `None` |  |
| `top_k` | `int | None` | `None` |  |
| `min_p` | `float | None` | `None` |  |
| `presence_penalty` | `float | None` | `None` |  |
| `frequency_penalty` | `float | None` | `None` |  |
| `repetition_penalty` | `float | None` | `None` |  |
| `service_tier` | `Literal['fast', 'flex'] | None` | `None` |  |
