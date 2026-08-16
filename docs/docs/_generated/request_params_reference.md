<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

### Available `RequestParams` Fields (Generated)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `meta` | `mcp_types._types.RequestParamsMeta | None` | `None` |  |
| `messages` | `list[mcp_types._types.SamplingMessage]` | `[]` |  |
| `model_preferences` | `mcp_types._types.ModelPreferences | None` | `None` |  |
| `system_prompt` | `str | None` | `None` |  |
| `include_context` | `Literal['none', 'thisServer', 'allServers'] | None` | `None` |  |
| `temperature` | `float | None` | `None` |  |
| `max_tokens` | `int | None` | `None` |  |
| `stop_sequences` | `list[str] | None` | `None` |  |
| `metadata` | `dict[str, Any] | None` | `None` |  |
| `tools` | `list[mcp_types._types.Tool] | None` | `None` |  |
| `tool_choice` | `mcp_types._types.ToolChoice | None` | `None` |  |
| `task` | `mcp_types._types.TaskMetadata | None` | `None` |  |
| `model` | `str | None` | `None` |  |
| `use_history` | `bool` | `True` |  |
| `max_iterations` | `int` | `9999` |  |
| `parallel_tool_calls` | `bool` | `True` |  |
| `response_format` | `Any | None` | `None` |  |
| `structured_schema` | `dict[str, Any] | None` | `None` |  |
| `structured_tool_policy` | `Literal['auto', 'always', 'defer', 'no_tools']` | `'auto'` |  |
| `sampling_tool_choice` | `Literal['auto', 'required', 'none'] | None` | `None` |  |
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
