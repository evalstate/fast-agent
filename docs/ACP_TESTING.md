# Testing the ACP Server Implementation

## Quick Start

The Agent Client Protocol (ACP) support in fast-agent allows it to act as an ACP agent over stdio.

### Prerequisites

1. **API Key**: You need an LLM provider configured (Anthropic, OpenAI, etc.)
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   # OR
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Instruction File**: Create a simple instruction file
   ```bash
   echo "You are a helpful AI assistant." > /tmp/instruction.md
   ```

### Running the Server

Start the ACP server:

```bash
fast-agent serve --transport acp --instruction /tmp/instruction.md --model haiku
```

The server will:
- Listen on stdin/stdout for JSON-RPC messages
- Process ACP protocol messages
- Stay running until Ctrl+C

### Testing with Python ACP Client

Using the Python `acp` library:

```python
#!/usr/bin/env python3
import asyncio
import json
from acp import ClientSideConnection
from acp.stdio import spawn_agent_process
from acp.schema import ClientCapabilities, ClientInfo
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block

class SimpleClient:
    """Minimal ACP client implementation."""

    def __init__(self, conn):
        self.conn = conn

    async def sessionUpdate(self, params):
        print(f"Session update: {params}")

    # Add other required methods...

async def test():
    # Spawn fast-agent as ACP server
    async with spawn_agent_process(
        lambda agent: SimpleClient(agent),
        'fast-agent', 'serve', '--transport', 'acp',
        '--instruction', '/tmp/instruction.md',
        '--model', 'haiku',
    ) as (connection, process):

        # 1. Initialize
        init_response = await connection.initialize(InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": False, "writeTextFile": False},
                terminal=False,
            ),
            clientInfo=ClientInfo(name="test-client", version="0.1.0"),
        ))

        print(f"✓ Initialized: {init_response.agentInfo.name}")

        # 2. Create session
        session_response = await connection.newSession(NewSessionRequest(
            mcpServers=[],
        ))

        session_id = session_response.sessionId
        print(f"✓ Session created: {session_id}")

        # 3. Send prompt
        prompt_response = await connection.prompt(PromptRequest(
            sessionId=session_id,
            prompt=[text_block("What is 2+2?")],
        ))

        print(f"✓ Prompt completed: {prompt_response.stopReason}")

if __name__ == "__main__":
    asyncio.run(test())
```

### Testing with Manual JSON-RPC

You can also test by sending raw JSON-RPC messages:

```bash
# Terminal 1: Start server
fast-agent serve --transport acp --instruction /tmp/instruction.md --model haiku

# Terminal 2: Send messages
# Initialize
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{"fs":{"readTextFile":false,"writeTextFile":false},"terminal":false},"clientInfo":{"name":"test","version":"0.1"}}}'

# Expected response:
# {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"agentCapabilities":{...},"agentInfo":{...},"authMethods":[]}}
```

## Protocol Messages

### 1. Initialize

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": 1,
    "clientCapabilities": {
      "fs": {"readTextFile": false, "writeTextFile": false},
      "terminal": false
    },
    "clientInfo": {"name": "client-name", "version": "1.0"}
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": 1,
    "agentCapabilities": {
      "prompts": {"supportedTypes": ["text"]},
      "loadSession": false
    },
    "agentInfo": {"name": "fast-agent", "version": "0.1.0"},
    "authMethods": []
  }
}
```

### 2. Create Session

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "session/new",
  "params": {
    "mcpServers": []
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "sessionId": "uuid-here"
  }
}
```

### 3. Send Prompt

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "session/prompt",
  "params": {
    "sessionId": "uuid-from-session-new",
    "prompt": [
      {"type": "text", "text": "What is 2+2?"}
    ]
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "stopReason": "end_turn"
  }
}
```

## Current Limitations

This first-pass implementation supports:
- ✅ Connection establishment
- ✅ Protocol initialization
- ✅ Session creation
- ✅ Text prompts
- ✅ Basic response handling

Not yet implemented:
- ❌ Streaming responses (no sessionUpdate notifications)
- ❌ Tool call permissions
- ❌ Multimodal content (images, audio)
- ❌ Authentication
- ❌ Session loading

## Troubleshooting

### Server exits immediately

**Problem:** Server returns to command prompt immediately

**Solution:** Check that you have an API key configured:
```bash
echo $ANTHROPIC_API_KEY  # or OPENAI_API_KEY
```

If not set:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### No response from server

**Problem:** Server starts but doesn't respond to messages

**Check:**
1. Verify server is still running: `ps aux | grep fast-agent`
2. Check logs in stderr
3. Ensure messages are properly formatted JSON-RPC

### "Provider Configuration Error"

**Problem:** Error about missing API key

**Solution:**
- Set environment variable: `export ANTHROPIC_API_KEY=...`
- Or create config file: `~/.config/fast-agent/fastagent.secrets.yaml`

## Integration with Editors

### Zed Editor

The ACP protocol was designed for Zed. To integrate:

1. Configure fast-agent as an agent in Zed settings
2. Point to: `fast-agent serve --transport acp --instruction <path> --model haiku`
3. Zed will spawn the process and communicate via ACP

### Other Editors

Any editor that supports spawning ACP agents via subprocess can use fast-agent:

```json
{
  "command": "fast-agent",
  "args": ["serve", "--transport", "acp", "--instruction", "/path/to/instruction.md", "--model", "haiku"],
  "protocol": "acp"
}
```

## Next Steps

To complete the implementation:

1. **Add streaming**: Implement `sessionUpdate` notifications to stream responses
2. **Tool permissions**: Map fast-agent tool execution to ACP's permission model
3. **Multimodal**: Support image/audio content blocks
4. **Multiple agents**: Add routing logic for agent selection
5. **Session persistence**: Implement `loadSession` capability

## References

- [ACP Specification](https://github.com/agentclientprotocol/agent-client-protocol)
- [Python ACP Library](https://github.com/PsiACE/agent-client-protocol-python)
- [fast-agent MCP Documentation](../README.md) (similar architecture)
