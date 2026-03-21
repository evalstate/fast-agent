<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

### Available `RequestParams` Fields (Generated)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `task` | `mcp.types.TaskMetadata | None` | `None` |  |
| `meta` | `mcp.types.RequestParams.Meta | None` | `None` |  |
| `messages` | `list` | `[]` |  |
| `modelPreferences` | `mcp.types.ModelPreferences | None` | `None` |  |
| `systemPrompt` | `str | None` | `None` |  |
| `includeContext` | `Optional` | `None` |  |
| `temperature` | `float | None` | `None` |  |
| `maxTokens` | `int` | `2048` |  |
| `stopSequences` | `list[str] | None` | `None` |  |
| `metadata` | `dict[str, typing.Any] | None` | `None` |  |
| `tools` | `list[mcp.types.Tool] | None` | `None` |  |
| `toolChoice` | `mcp.types.ToolChoice | None` | `None` |  |
| `model` | `str | None` | `None` |  |
| `use_history` | `bool` | `True` |  |
| `max_iterations` | `int` | `99` |  |
| `parallel_tool_calls` | `bool` | `True` |  |
| `response_format` | `typing.Any | None` | `None` |  |
| `template_vars` | `dict` | `PydanticUndefined` |  |
| `mcp_metadata` | `dict[str, typing.Any] | None` | `None` |  |
| `tool_execution_handler` | `typing.Any | None` | `None` |  |
| `emit_loop_progress` | `bool` | `False` |  |
| `streaming_timeout` | `float | None` | `300.0` |  |
