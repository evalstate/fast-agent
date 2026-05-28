---
social:
  title: MCP Types
  tagline: Use MCP protocol types directly in fast-agent integrations.
  description: Use MCP protocol types directly in fast-agent integrations.
  alt: fast-agent social card — MCP Types
---

# Integration with MCP Types

## MCP Type Compatibility

FastAgent is built to seamlessly integrate with the MCP SDK type system:

Conversations with assistants are based on `PromptMessageExtended` - an extension of the MCP `PromptMessage` type, with support for multiple content sections. This type is expected to become native in a future version of MCP: https://github.com/modelcontextprotocol/specification/pull/198

## Message History Transfer

FastAgent makes it easy to transfer conversation history between agents:

```python title="history_transfer.py"
@fast.agent(name="haiku", model="haiku")
@fast.agent(name="openai", model="o3-mini.medium")

async def main() -> None:
    async with fast.run() as agent:
        # Start an interactive session with "haiku"
        await agent.interactive(agent_name="haiku")
        # Transfer the message history to "openai" (using PromptMessageExtended)
        await agent.openai.generate(agent.haiku.message_history)
        # Continue the conversation
        await agent.interactive(agent_name="openai")
```
