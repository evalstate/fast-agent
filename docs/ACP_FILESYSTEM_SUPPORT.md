# ACP Filesystem Support

## Overview

FastAgent now supports the Agent Client Protocol (ACP) filesystem capabilities, enabling file read and write operations through the client's filesystem interface (e.g., Zed editor) instead of local file operations when available.

This provides better integration with editor environments and allows access to unsaved editor state that wouldn't be visible through traditional file I/O.

## Architecture

### Transparent Integration

The implementation uses a **transparent integration** approach where:
- The LLM sees `read_text_file` and `write_text_file` tools when the client supports them
- When in ACP mode with a filesystem-capable client, file operations automatically route to the client
- The client can provide unsaved editor state and track modifications
- No changes required to prompts or agent configurations

### Components

#### 1. ACPFilesystemRuntime (`src/fast_agent/acp/filesystem_runtime.py`)

The core runtime that implements file operations via ACP filesystem methods:

```python
# Read file flow:
result = connection.fs_read_text_file(session_id, path, line, limit)
content = result.content

# Write file flow:
connection.fs_write_text_file(session_id, path, content)
```

**Features:**
- Read entire files or specific line ranges
- Access unsaved editor changes
- Write files (creates if doesn't exist)
- Proper error handling and logging
- Supports partial reads via `line` and `limit` parameters

**Tools Provided:**

1. **read_text_file**
   - Parameters:
     - `path` (required): Absolute path to file
     - `line` (optional): Starting line number (1-based)
     - `limit` (optional): Maximum number of lines to read
   - Returns: File content as text

2. **write_text_file**
   - Parameters:
     - `path` (required): Absolute path to file
     - `content` (required): Text content to write
   - Returns: Success message
   - Creates file if it doesn't exist

#### 2. AgentACPServer Updates (`src/fast_agent/acp/server/agent_acp_server.py`)

**Capability Detection:**
```python
# During initialize()
if params.clientCapabilities and hasattr(params.clientCapabilities, "fs"):
    fs_caps = params.clientCapabilities.fs
    if fs_caps:
        self._client_supports_fs_read = bool(getattr(fs_caps, "readTextFile", False))
        self._client_supports_fs_write = bool(getattr(fs_caps, "writeTextFile", False))
```

**Runtime Injection:**
```python
# During newSession()
if (self._client_supports_fs_read or self._client_supports_fs_write) and self._connection:
    filesystem_runtime = ACPFilesystemRuntime(
        connection=self._connection,
        session_id=session_id,
        activation_reason="via ACP filesystem support",
        supports_read=self._client_supports_fs_read,
        supports_write=self._client_supports_fs_write,
    )
    agent.set_external_runtime(filesystem_runtime)
```

#### 3. McpAgent Updates (`src/fast_agent/agents/mcp_agent.py`)

**Multiple External Runtimes Support:**
```python
def set_external_runtime(self, runtime) -> None:
    """Inject external runtime (e.g., ACPTerminalRuntime, ACPFilesystemRuntime)."""
    if not hasattr(self, "_external_runtimes"):
        self._external_runtimes = []
    self._external_runtimes.append(runtime)

async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
    # Check all external runtimes first
    for runtime in self._external_runtimes:
        # Check if runtime has multiple tools (e.g., filesystem)
        if hasattr(runtime, "tools"):
            for tool in runtime.tools:
                if tool.name == name:
                    return await runtime.call_tool(name, arguments)
        # Check if runtime has single tool (e.g., terminal)
        elif hasattr(runtime, "tool"):
            if runtime.tool and name == runtime.tool.name:
                return await runtime.execute(arguments)
    # ... other tools
```

## Usage

### Requirements

1. **Agent side**: FastAgent running in ACP mode
2. **Client side**: ACP client that advertises `fs.readTextFile` and/or `fs.writeTextFile` capabilities
3. **ACP mode**: Running via `fast-agent-acp` or `fast-agent serve --transport acp`

### Automatic Enablement

Filesystem support is **automatically enabled** when all conditions are met:
- Client advertises filesystem capabilities during `initialize()`
- Running in ACP mode

No additional configuration or flags required!

### Example Usage

```bash
# Start FastAgent in ACP mode
fast-agent-acp --instruction prompt.md --model sonnet

# Or via serve command
fast-agent serve --transport acp --model haiku
```

When connected from a filesystem-capable client (like Zed):
1. LLM can call `read_text_file` and `write_text_file` tools
2. File operations access the client's filesystem (including unsaved changes)
3. Client can track and display file modifications
4. Results are returned to the LLM

### Example LLM Interaction

```
User: "Check if config.json exists and show me its contents"

LLM: I'll read the config.json file for you.
[Calls read_text_file tool with: {"path": "/home/user/project/config.json"}]

[Content returned, including any unsaved changes in the editor]

LLM: Here's the current configuration:
{
  "debug": true,
  "version": "1.0.0"
}
```

```
User: "Update the version to 2.0.0"

LLM: I'll update the version in config.json.
[Calls write_text_file tool with: {"path": "/home/user/project/config.json", "content": "{\n  \"debug\": true,\n  \"version\": \"2.0.0\"\n}"}]

LLM: I've updated the version to 2.0.0 in config.json.
```

## Capability Negotiation

The client advertises filesystem support during initialization:

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "result": {
    "protocolVersion": 1,
    "clientCapabilities": {
      "fs": {
        "readTextFile": true,
        "writeTextFile": true
      }
    }
  }
}
```

FastAgent checks for these capabilities and only enables the corresponding tools.

## Implementation Details

### Partial File Reading

The `read_text_file` tool supports reading specific line ranges:

```python
# Read entire file
arguments = {"path": "/path/to/file.txt"}

# Read from line 10 onwards
arguments = {"path": "/path/to/file.txt", "line": 10}

# Read 50 lines starting from line 10
arguments = {"path": "/path/to/file.txt", "line": 10, "limit": 50}
```

### File Writing

The `write_text_file` tool overwrites the entire file:

```python
arguments = {
    "path": "/path/to/file.txt",
    "content": "New file content"
}
```

Per the ACP specification, the client **MUST** create the file if it doesn't exist.

### Error Handling

Both tools return `CallToolResult` with appropriate error flags:

```python
# Success
CallToolResult(
    content=[text_content("file content here")],
    isError=False,
)

# Error
CallToolResult(
    content=[text_content("Error reading file: ...")],
    isError=True,
)
```

### Session Isolation

Each ACP session gets its own filesystem runtime instance:
- Sessions are isolated
- Runtime is created during `newSession()`
- Cleanup happens automatically on session end

## Comparison with Terminal Support

| Feature | Terminal | Filesystem |
|---------|----------|------------|
| Tools | 1 (`execute`) | 2 (`read_text_file`, `write_text_file`) |
| Runtime Property | `.tool` (singular) | `.tools` (list) |
| Call Method | `.execute(arguments)` | `.call_tool(name, arguments)` |
| Client Capability | `terminal: true` | `fs: {readTextFile: bool, writeTextFile: bool}` |
| Requires Flag | Yes (`--shell`) | No |

## Testing

### Manual Testing

1. Start FastAgent in ACP mode
2. Connect from a filesystem-capable client
3. Try reading a file: "Read the contents of /path/to/file.txt"
4. Try writing a file: "Write 'hello' to /tmp/test.txt"
5. Verify operations succeed and client tracks changes

### Integration Tests

Create tests similar to `tests/integration/acp/test_acp_terminal.py`:

```python
# Test filesystem support is enabled
test_acp_filesystem_support_enabled()

# Test file read operation
test_acp_filesystem_read_text_file()

# Test file write operation
test_acp_filesystem_write_text_file()

# Test partial file read
test_acp_filesystem_partial_read()

# Test filesystem disabled when client doesn't support it
test_acp_filesystem_disabled_when_client_unsupported()
```

## Troubleshooting

### Tools not appearing

**Check:**
1. Is the client advertising `fs.readTextFile` or `fs.writeTextFile` in `clientCapabilities`?
2. Check logs for "ACP filesystem runtime injected" message
3. Verify running in ACP mode

### File operations failing

**Solutions:**
1. Verify file paths are absolute (not relative)
2. Check client logs for permission errors
3. Ensure client supports the operation (read vs write)

### Can't access unsaved changes

**Note:** Unsaved changes are only available if the client provides them. Verify:
1. Client implements ACP filesystem spec correctly
2. File is open in the editor
3. Client is in the correct working directory

## Future Enhancements

Potential improvements for future releases:

1. **Directory operations**: List directory contents
2. **File metadata**: Get file stats (size, modified time, etc.)
3. **Batch operations**: Read/write multiple files in one call
4. **Binary file support**: Handle non-text files
5. **File watching**: Subscribe to file change notifications
6. **Append mode**: Append to files instead of overwriting

## References

- [ACP Filesystem Specification](https://agentclientprotocol.com/protocol/file-system.mdx)
- [agent-client-protocol Python SDK](https://pypi.org/project/agent-client-protocol/)
- [FastAgent ACP Implementation](./ACP_IMPLEMENTATION_OVERVIEW.md)
- [ACP Terminal Support](./ACP_TERMINAL_SUPPORT.md)

## Summary

ACP filesystem support provides seamless integration between FastAgent and editor environments, allowing file operations through the client's filesystem with access to unsaved editor state. The implementation is transparent to LLMs and requires no configuration, activating automatically when the client advertises filesystem capabilities.
