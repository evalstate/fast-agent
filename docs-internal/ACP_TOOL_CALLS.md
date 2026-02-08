# ACP Tool Call Implementation

This document describes the implementation of tool call tracking and notifications for the Agent Client Protocol (ACP).

## Overview

The fast-agent ACP implementation now supports the full ACP tool call protocol as specified in [https://agentclientprotocol.com/protocol/tool-calls.md](https://agentclientprotocol.com/protocol/tool-calls.md).

## Features

### 1. Tool Call Notifications

When a tool is executed in ACP mode, the client receives structured notifications about the tool execution:

- **Initial Notification** (`tool_call`): Sent when tool execution starts
  - Contains `toolCallId`, `title`, `kind`, and initial `status` ("pending")
  - Tool kind is automatically inferred from the tool name (read, edit, execute, etc.)

- **Progress Updates** (`tool_call_update`): Sent during tool execution
  - Updates status to "in_progress" when execution begins
  - Includes progress information from MCP servers that support progress notifications
  - Contains optional content with progress messages

- **Completion Notification** (`tool_call_update`): Sent when tool execution completes
  - Final status is either "completed" or "failed"
  - Includes result content or error messages

### 2. Tool Progress Tracking

The implementation hooks into the existing MCP progress notification infrastructure:

- MCP servers can report progress during tool execution
- Progress is automatically forwarded to ACP clients via `tool_call_update` notifications
- Progress includes numeric values (current/total) and optional text messages

### 3. Tool Permission Handler (Framework)

The codebase includes a permission handler framework for requesting tool execution approval:

- `ACPToolPermissionManager` can request permission before executing tools
- Uses ACP's `session/request_permission` RPC
- Supports "allow once", "allow always", "reject once", and "reject always" options
- Permissions are remembered per tool/server combination

**Note**: Permission checking is not yet integrated into the default execution flow but can be enabled by configuring permission handlers on agent instances.

## Architecture

### Components

1. **`src/fast_agent/acp/tool_progress.py`**
   - `ACPToolProgressManager`: Manages tool call tracking and notifications
   - `ToolCallTracker`: Tracks state for individual tool calls
   - Handles tool kind inference and notification sending

2. **`src/fast_agent/acp/tool_permissions.py`**
   - `ACPToolPermissionManager`: Manages permission requests
   - `ToolPermissionRequest/Response`: Data models for permission handling
   - Permission caching and "remember" functionality

3. **`src/fast_agent/mcp/mcp_aggregator.py`** (Modified)
   - Added optional tool execution hooks: `tool_start_hook`, `tool_progress_hook`, `tool_complete_hook`
   - Hooks are called during tool execution to enable ACP notifications
   - Maintains backward compatibility - hooks are optional

4. **`src/fast_agent/acp/server/agent_acp_server.py`** (Modified)
   - Initializes `ACPToolProgressManager` when connection is established
   - Registers tool hooks with agent aggregators during session creation
   - Manages cleanup of tool trackers when sessions end

### Data Flow

```
User Prompt → ACP Client → ACP Server → Agent → MCP Aggregator → MCP Server
                                                      ↓
                                              Tool Start Hook
                                                      ↓
                                            ACP Notification (tool_call)
                                                      ↓
                                              MCP Tool Execution
                                                      ↓
                                            Progress Callbacks
                                                      ↓
                                            ACP Notifications (tool_call_update)
                                                      ↓
                                              Tool Complete Hook
                                                      ↓
                                            ACP Notification (tool_call_update - completed)
```

## Configuration

No additional configuration is required. Tool call notifications are automatically enabled when fast-agent is run in ACP mode (`--transport acp`).

### Enabling Permission Checking (Optional)

To enable permission checking, you can configure a permission handler on your agent:

```python
from fast_agent.acp.tool_permissions import ACPToolPermissionManager, create_acp_permission_handler

# In your agent initialization code:
permission_manager = ACPToolPermissionManager(connection)
permission_handler = create_acp_permission_handler(permission_manager, session_id)

# Register with agent (implementation depends on your agent setup)
agent.set_tool_permission_handler(permission_handler)
```

## Testing

Integration tests are provided in `tests/integration/acp/test_acp_tool_notifications.py`:

- `test_acp_tool_call_notifications`: Verifies basic tool call notification structure
- `test_acp_tool_progress_updates`: Tests progress update handling
- `test_acp_tool_kinds_inferred`: Checks tool kind inference

Run tests with:
```bash
pytest tests/integration/acp/test_acp_tool_notifications.py -v
```

## Compliance

This implementation complies with the ACP tool call specification:

- ✅ Sends `tool_call` notifications when tools start
- ✅ Sends `tool_call_update` notifications for progress and completion
- ✅ Includes required fields: `toolCallId`, `title`, `kind`, `status`
- ✅ Supports all status transitions: pending → in_progress → completed/failed
- ✅ Infers tool kinds from tool names
- ✅ Includes optional content with progress messages
- ✅ Framework for permission requests (not yet enforced by default)

## Future Enhancements

1. **Default Permission Enforcement**: Integrate permission checking into the default tool execution flow
2. **Advanced Tool Kind Detection**: More sophisticated inference based on tool schemas
3. **Location Tracking**: Include file locations in tool call notifications for file operations
4. **Diff Support**: Include diffs in tool call content for edit operations
5. **Terminal Integration**: Link terminal operations to tool calls
