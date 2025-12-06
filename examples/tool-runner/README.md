# ToolRunner Examples

The `ToolRunner` provides an async-iterable interface for fine-grained control over tool-calling loops. Instead of the agent handling everything internally, you can observe and customize each step.

## Examples

### `tool_runner_example.py` - Basic Usage

Demonstrates the core features:
- **Iterating over messages**: See each Claude response as it happens
- **Using `until_done()`**: Simple one-liner for when you don't need intermediate steps
- **Early exit**: Break out of the loop when you've got what you need

```bash
uv run python examples/tool-runner/tool_runner_example.py
```

### `advanced_tool_runner.py` - Advanced Customization

Shows advanced control features:
- **`generate_tool_call_response()`**: Inspect tool results before they're sent back
- **`set_request_params()`**: Modify LLM parameters between iterations
- **`append_messages()`**: Inject additional guidance during the loop

```bash
uv run python examples/tool-runner/advanced_tool_runner.py
```

## API Quick Reference

```python
# Create a runner
runner = app.tool_runner("Your prompt here")

# Option 1: Iterate manually
async for message in runner:
    print(message.last_text())
    print(message.stop_reason)  # END_TURN, TOOL_USE, ERROR, etc.

    if should_stop:
        break

# Option 2: Get final result directly
final = await runner.until_done()

# Properties
runner.current_message   # Most recent message
runner.is_done          # Whether loop is complete
runner.iterations       # Number of LLM calls made
runner.messages         # Current message history

# Customization (call inside the loop)
await runner.generate_tool_call_response()  # Get tool results early
runner.set_request_params(lambda p: ...)    # Modify next request
runner.append_messages(msg1, msg2, ...)     # Add messages for next turn
```

## When to Use ToolRunner

Use `ToolRunner` when you need to:
- Monitor progress of long-running tool chains
- Implement custom stopping logic
- Inject user feedback or guidance mid-conversation
- Log or audit each step of tool execution
- Implement streaming progress updates to a UI

For simple cases where you just want the final result, use the standard `send()` or `generate()` methods instead.
