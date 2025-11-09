# ACP Terminal Support

## Overview

FastAgent now supports the Agent Client Protocol (ACP) terminal capabilities, enabling command execution through the client's terminal interface (e.g., Zed editor) instead of local process execution when available.

This provides better integration with editor environments and allows users to see and interact with command execution directly in their IDE's terminal UI.

## Architecture

### Transparent Integration

The implementation uses a **transparent integration** approach where:
- The LLM sees the same `execute` tool regardless of the runtime
- When in ACP mode with a terminal-capable client, commands automatically route to the client's terminals
- When terminals aren't available, fallback to local `ShellRuntime` happens automatically
- No changes required to prompts or agent configurations

### Components

#### 1. ACPTerminalRuntime (`src/fast_agent/acp/terminal_runtime.py`)

The core runtime that implements command execution via ACP terminal methods:

```python
# Execution flow:
terminal_id = create_unique_id()
connection.terminal_create(terminal_id, command)  # Start execution
connection.terminal_wait_for_exit(terminal_id)    # Wait for completion
output = connection.terminal_output(terminal_id)  # Get results
connection.terminal_release(terminal_id)          # Cleanup
```

**Features:**
- Timeout handling with configurable duration
- Proper error handling and cleanup
- Exit code reporting
- Handles truncated output
- Follows ACP spec requirement to release terminals

#### 2. AgentACPServer Updates (`src/fast_agent/acp/server/agent_acp_server.py`)

**Capability Detection:**
```python
# During initialize()
if params.clientCapabilities:
    self._client_supports_terminal = bool(
        getattr(params.clientCapabilities, "terminal", False)
    )
```

**Runtime Injection:**
```python
# During newSession()
if self._client_supports_terminal and agent._shell_runtime_enabled:
    terminal_runtime = ACPTerminalRuntime(
        connection=self._connection,
        session_id=session_id,
        activation_reason="via ACP terminal support",
        timeout_seconds=agent._shell_runtime.timeout_seconds,
    )
    agent.set_external_runtime(terminal_runtime)
```

#### 3. McpAgent Updates (`src/fast_agent/agents/mcp_agent.py`)

**External Runtime Support:**
```python
def set_external_runtime(self, runtime) -> None:
    """Inject external runtime (e.g., ACPTerminalRuntime)."""
    self._external_runtime = runtime

async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
    # Check external runtime first (ACP terminal)
    if self._external_runtime and self._external_runtime.tool:
        if name == self._external_runtime.tool.name:
            return await self._external_runtime.execute(arguments)

    # Fall back to shell runtime
    if self._shell_runtime.tool and name == self._shell_runtime.tool.name:
        return await self._shell_runtime.execute(arguments)
    # ... other tools
```

## Usage

### Requirements

1. **Agent side**: FastAgent with `--shell` flag enabled
2. **Client side**: ACP client that advertises `terminal: true` capability
3. **ACP mode**: Running via `fast-agent-acp` or `fast-agent serve --transport acp`

### Automatic Enablement

Terminal support is **automatically enabled** when all conditions are met:
- Client advertises terminal capability during `initialize()`
- Agent has shell runtime enabled (via `--shell` flag or skill configuration)
- Running in ACP mode

No additional configuration or flags required!

### Example Usage

```bash
# Start FastAgent in ACP mode with shell enabled
fast-agent-acp --instruction prompt.md --model sonnet --shell

# Or via serve command
fast-agent serve --transport acp --shell --model haiku
```

When connected from a terminal-capable client (like Zed):
1. LLM can call the `execute` tool as normal
2. Commands run in the client's terminal UI
3. User sees execution in real-time in their editor
4. Output is returned to the LLM when complete

### Example LLM Interaction

```
User: "Check the git status"

LLM: I'll check the git status for you.
[Calls execute tool with: "git status"]

[Terminal appears in Zed showing git status output]

LLM: The repository is clean with no uncommitted changes.
```

## Fallback Behavior

The implementation gracefully handles various scenarios:

| Scenario | Behavior |
|----------|----------|
| Client supports terminals + `--shell` flag | ✅ Use ACP terminals |
| Client doesn't support terminals + `--shell` flag | ⚠️ Fall back to local ShellRuntime |
| Client supports terminals but no `--shell` flag | ❌ No execute tool available |
| Not in ACP mode | Uses local ShellRuntime as before |

## Implementation Details

### Terminal Lifecycle

Per the ACP specification, terminals must be properly released:

```python
try:
    # 1. Create
    await connection.terminal_create(params)

    # 2. Wait for completion
    await connection.terminal_wait_for_exit(params)

    # 3. Get output
    output = await connection.terminal_output(params)

finally:
    # 4. Always release (REQUIRED by spec)
    await connection.terminal_release(params)
```

### Timeout Handling

Timeouts work similarly to local ShellRuntime:
- Default: 90 seconds
- Configurable via `shell_execution.timeout_seconds` in config
- On timeout: kills terminal, retrieves partial output, returns error

```python
try:
    result = await asyncio.wait_for(
        connection.terminal_wait_for_exit(params),
        timeout=self.timeout_seconds,
    )
except asyncio.TimeoutError:
    await connection.terminal_kill(params)
    output = await connection.terminal_output(params)
    await connection.terminal_release(params)
    return error_result_with_partial_output
```

### Session Isolation

Each ACP session gets its own terminal runtime instance:
- Sessions are isolated
- Terminal IDs are unique per execution
- Cleanup happens automatically on session end

## Testing

### Integration Tests

Located in `tests/integration/acp/test_acp_terminal.py`:

```python
# Test terminal support is enabled
test_acp_terminal_support_enabled()

# Test terminal execution flow
test_acp_terminal_execution()

# Test fallback when shell flag not provided
test_acp_terminal_disabled_when_no_shell_flag()

# Test fallback when client doesn't support terminals
test_acp_terminal_disabled_when_client_unsupported()
```

### Test Client

`tests/integration/acp/test_client.py` implements terminal simulation:
- Simulates terminal creation and execution
- Returns mock output for testing
- Properly tracks terminal lifecycle

### Running Tests

```bash
# Run ACP terminal tests
pytest tests/integration/acp/test_acp_terminal.py -v

# Run all ACP tests
pytest tests/integration/acp/ -v
```

## Configuration

Terminal support respects existing shell runtime configuration:

```yaml
# fastagent.config.yaml
shell_execution:
  timeout_seconds: 90  # Applies to both local and ACP terminals
  warning_interval_seconds: 30
```

## Troubleshooting

### Terminal commands not executing

**Check:**
1. Is `--shell` flag enabled?
2. Does the client advertise `terminal: true` in `clientCapabilities`?
3. Check logs for "ACP terminal runtime injected" message

### Commands timing out

**Solutions:**
1. Increase timeout in config: `shell_execution.timeout_seconds`
2. Break long-running commands into smaller steps
3. Use background execution patterns if appropriate

### Terminal not visible in client

**Note:** Terminal UI is handled entirely by the client (e.g., Zed). If terminals aren't appearing:
1. Verify client supports ACP terminal UI
2. Check client logs/settings
3. Ensure terminal window/pane is visible in client UI

## Future Enhancements

Potential improvements for future releases:

1. **Background execution**: Support for long-running commands
2. **Interactive terminals**: Shell session reuse across multiple commands
3. **Terminal customization**: Custom environment, working directory per command
4. **Output streaming**: Real-time output streaming to LLM (currently waits for completion)
5. **Signal handling**: Support for custom termination signals

## References

- [ACP Terminal Specification](https://agentclientprotocol.com/protocol/terminals.md)
- [agent-client-protocol Python SDK](https://pypi.org/project/agent-client-protocol/)
- [FastAgent ACP Implementation](./ACP_IMPLEMENTATION_OVERVIEW.md)
- [Shell Runtime Implementation](../src/fast_agent/tools/shell_runtime.py)

## Summary

ACP terminal support provides seamless integration between FastAgent and editor environments, allowing command execution through the client's terminal UI with automatic fallback to local execution when needed. The implementation is transparent to LLMs and requires minimal configuration, activating automatically when the client advertises terminal capability and the agent has shell runtime enabled.
