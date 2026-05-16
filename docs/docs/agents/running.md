# Deploy and Run 

**fast-agent** provides flexible deployment options to meet a variety of use cases, from interactive development to production server deployments.

## Interactive Mode

Run **fast-agent** programs interactively for development, debugging, or direct user interaction.

```python title="agent.py"
import asyncio
from fast_agent.core.fastagent import FastAgent

fast = FastAgent("My Interactive Agent")

@fast.agent(instruction="You are a helpful assistant")
async def main():
    async with fast.run() as agent:
        # Start interactive prompt
        await agent()

if __name__ == "__main__":
    asyncio.run(main())
```

When started with `uv run agent.py`, this begins an interactive prompt where you can chat directly with the configured agents, apply prompts, save history and so on.

## Command Line Execution

**fast-agent** supports command-line arguments to run agents and workflows with specific messages.

```bash
# Send a message to a specific agent
uv run agent.py --agent default --message "Analyze this dataset"

# Override the default model
uv run agent.py --model gpt-4o --agent default --message "Complex question"

# Run with minimal output
uv run agent.py --quiet --agent default --message "Background task"
```

This is perfect for scripting, automation, or one-off queries. 

The `--quiet` flag switches off the Progress, Chat and Tool displays.


## MCP Server Deployment

Any **fast-agent** application can be deployed as an MCP server with a simple command-line switch.

### Starting an MCP Server

```bash
# Start as a Streamable HTTP server (http://localhost:8080/mcp)
uv run agent.py --transport http --port 8080

# Start as a stdio server
uv run agent.py --transport stdio
```

Each agent exposes an MCP Tool for sending messages to the agent, and a Prompt that returns the conversation history. 

This enables cross-agent state transfer via the MCP Prompts.

The MCP Server can also be started programatically.

### Programmatic Server Startup

```python
import asyncio
from fast_agent.core.fastagent import FastAgent

fast = FastAgent("Server Agent")

@fast.agent(instruction="You are an API agent")
async def main():
    # Start as a server programmatically
    await fast.start_server(
        transport="http",
        host="0.0.0.0",
        port=8080,
        server_name="API-Agent-Server",
        server_description="Provides API access to my agent",
        tool_description="Send a message to the {agent} agent",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

`--transport` now implies server mode when running a Python module directly. The legacy `--server` flag remains as an alias but is deprecated.


## Python Program Integration

Embed  **fast-agent** into existing Python applications to add MCP agent capabilities.

```python
import asyncio
from fast_agent.core.fastagent import FastAgent

fast = FastAgent("Embedded Agent")

@fast.agent(instruction="You are a data analysis assistant")
async def analyze_data(data):
    async with fast.run() as agent:
        result = await agent.send(f"Analyze this data: {data}")
        return result

# Use in your application
async def main():
    user_data = get_user_data()
    analysis = await analyze_data(user_data)
    display_results(analysis)

if __name__ == "__main__":
    asyncio.run(main())
```


<!--
### Connecting to MCP Servers

Connect to MCP servers from other FastAgent applications  by configuring them in your `fastagent.config.yaml`:

```yaml
mcp:
  servers:
    my_remote_agent:
      transport: "sse"
      url: "http://localhost:8080"
```

Then use them in your client application:

```python
@fast.agent(servers=["my_remote_agent"])
async def client():
    async with fast.run() as agent:
        # Call tools on the remote server
        result = await agent.send('***CALL_TOOL remote_agent.send {"message": "Hello"}')
```
-->
