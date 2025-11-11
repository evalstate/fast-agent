# ACP Tool Calls Implementation

This document describes the implementation of tool calls support for the Agent Client Protocol (ACP) in fast-agent.

## Overview

The ACP tool calls implementation provides compliance with the [ACP Tool Calls specification](https://agentclientprotocol.com/protocol/tool-calls.md), enabling clients to track and manage tool executions performed by the agent.

## Architecture

The implementation consists of several key components:

### 1. Tool Call Tracker (`tool_call_tracker.py`)

The `ToolCallTracker` manages the lifecycle of tool calls and sends notifications to ACP clients.

**Lifecycle States:**
- `pending` → Tool call created, not yet executing
- `in_progress` → Tool is currently executing
- `completed` → Tool finished successfully
- `failed` → Tool execution failed

**Key Features:**
- Generates unique tool call IDs
- Sends `tool_call_update` notifications via `session/update`
- Tracks progress information
- Handles content and raw output

### 2. Permission Handler (`tool_call_permission_handler.py`)

The permission handler system allows customization of how tool call permissions are requested from the ACP client, following the same pattern as elicitation handlers for MCP servers.

**Built-in Handlers:**
- `ACPClientPermissionHandler` - Requests permission from the ACP client using `session/request_permission`
- `AlwaysAllowPermissionHandler` - Auto-approves all tool calls (default, for backward compatibility)

**Permission Options:**
- `allow_once` - Allow this specific tool call
- `allow_always` - Remember and always allow this tool
- `reject_once` - Reject this specific tool call
- `reject_always` - Remember and always reject this tool

### 3. Tool Call Middleware (`tool_call_integration.py`)

The `ACPToolCallMiddleware` intercepts tool calls and wraps them with ACP notifications and optional permission checking.

**Features:**
- Automatic tool kind inference (read, edit, delete, move, search, execute, think, fetch, other)
- Human-readable title generation
- Permission caching for "remember" decisions
- Progress callback forwarding from MCP to ACP
- Error handling and failed status reporting

**Helper Functions:**
- `infer_tool_kind(tool_name)` - Infers the kind from tool name patterns
- `create_tool_title(tool_name, arguments)` - Creates human-readable titles

### 4. Agent Wrapper (`acp_agent_wrapper.py`)

The `ACPAgentWrapper` wraps existing agents to intercept their `call_tool` method and inject ACP notifications.

This wrapper is transparent to the agent and doesn't require any agent-side changes.

### 5. ACP Server Integration (`server/agent_acp_server.py`)

The ACP server automatically sets up tool call tracking for each session:

1. When a session is created, a `ToolCallTracker` is instantiated
2. A `ACPToolCallMiddleware` is created with the tracker
3. All agents in the session are wrapped with `ACPAgentWrapper`
4. Tool calls are now automatically tracked and reported to the client

## Progress Notifications

MCP servers can report progress during tool execution. The ACP implementation forwards these progress notifications to the client:

1. MCP tool execution provides progress callbacks: `(progress, total, message)`
2. The middleware converts these to ACP `ToolCallProgress` objects
3. Progress updates are sent via `tool_call_update` notifications
4. Clients can display real-time progress to users

## Usage

### Basic (Default Behavior)

No configuration needed! Tool call tracking is automatically enabled for all ACP sessions.

```python
# Tool calls are automatically tracked and reported to the client
# No agent-side changes required
```

### With Permission Requests

To enable permission prompts for tool calls:

```python
from fast_agent.acp.tool_call_permission_handler import ACPClientPermissionHandler

# In your ACP server setup:
server._tool_permission_handler = ACPClientPermissionHandler()
server._enable_tool_permissions = True
```

### Custom Permission Handler

You can implement custom permission logic:

```python
from fast_agent.acp.tool_call_permission_handler import (
    ToolCallPermissionHandler,
    ToolCallPermissionRequest,
    ToolCallPermissionResponse,
)

class CustomPermissionHandler(ToolCallPermissionHandler):
    async def request_permission(
        self,
        request: ToolCallPermissionRequest,
        connection: Any,
    ) -> ToolCallPermissionResponse:
        # Your custom logic here
        # For example, allow read operations automatically:
        if request.tool_kind == "read":
            return ToolCallPermissionResponse(allowed=True)

        # Ask the client for other operations
        # ...

server._tool_permission_handler = CustomPermissionHandler()
server._enable_tool_permissions = True
```

## Client Integration

Clients receive tool call notifications through the `session/update` notification with `sessionUpdate: "tool_call_update"`:

```typescript
// Example client handling
async sessionUpdate(params: SessionNotification) {
  if (params.update.sessionUpdate === "tool_call_update") {
    const toolCall = params.update.toolCall;

    // Display tool call to user
    console.log(`Tool: ${toolCall.title}`);
    console.log(`Status: ${toolCall.status}`);

    if (toolCall.progress) {
      console.log(`Progress: ${toolCall.progress.current}/${toolCall.progress.total}`);
    }
  }
}
```

## Testing

The implementation includes comprehensive tests:

### Unit Tests
- `tests/unit/acp/test_tool_call_tracker.py` - Tests for the tracker lifecycle
- `tests/unit/acp/test_tool_call_integration.py` - Tests for middleware and helpers

### Integration Tests
- `tests/integration/acp/test_acp_tool_calls.py` - End-to-end ACP tool call tests

Run tests with:
```bash
poetry run pytest tests/unit/acp/test_tool_call_*.py -v
poetry run pytest tests/integration/acp/test_acp_tool_calls.py -v
```

## Compliance

The implementation complies with the ACP Tool Calls specification:

✅ **Tool Call Creation** - Creates tool calls with required fields (toolCallId, title, kind, status)
✅ **Status Updates** - Tracks lifecycle: pending → in_progress → completed/failed
✅ **Progress Notifications** - Forwards MCP progress to ACP clients
✅ **Permission Requests** - Supports requesting permissions before execution
✅ **Content Types** - Supports text content and raw output
✅ **Error Handling** - Properly marks failed tool calls

## Design Decisions

### 1. Default to Always Allow
For backward compatibility and ease of adoption, the default behavior is to allow all tool calls without prompting. This can be changed by configuring a different permission handler.

### 2. Agent Wrapping
Rather than modifying the agent protocol, we use a transparent wrapper. This keeps the ACP-specific code compartmentalized and doesn't require changes to existing agents.

### 3. Following Elicitation Pattern
The permission handler system follows the same pattern as MCP elicitation handlers, providing a consistent and familiar API for developers.

### 4. Progress Forwarding
MCP progress callbacks are automatically forwarded to ACP clients, providing a seamless integration between the two protocols.

### 5. Automatic Tool Kind Inference
The system infers tool kinds from tool names using common patterns (e.g., "read_file" → "read"). This reduces configuration burden while allowing manual override.

## Future Enhancements

Possible future improvements:

- **Location Tracking** - Report file paths and line numbers affected by tool calls
- **Diff Content** - Support showing file diffs in tool call results
- **Terminal Content** - Link terminal executions to tool calls
- **Custom Metadata** - Allow agents to attach custom metadata to tool calls
- **Tool Call History** - Persist tool call history across sessions
- **Analytics** - Aggregate tool usage statistics

## See Also

- [ACP Protocol Specification](https://agentclientprotocol.com/)
- [ACP Tool Calls Specification](https://agentclientprotocol.com/protocol/tool-calls.md)
- [ACP Terminal Support](./ACP_TERMINAL_SUPPORT.md)
