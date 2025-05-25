# PubSub Example for FastAgent

This example demonstrates how to use the PubSub system to stream FastAgent output to channels instead of directly to the terminal console. Each agent has its own pub/sub channel where messages are published, and clients can subscribe to these channels to receive the messages.

## Features

- Streaming agent output to pub/sub channels
- Interactive mode through pub/sub
- Console output is preserved (output goes to both console and pub/sub)
- Client application to subscribe to agent channels

## Setup

1. Copy `fastagent.secrets.yaml.example` to `fastagent.secrets.yaml` and add your API keys:

```bash
cp fastagent.secrets.yaml.example fastagent.secrets.yaml
# Edit the file to add your API keys
```

2. Install required packages:

```bash
pip install -e ../../../..
```

## Running the Example

1. First, run the example agent:

```bash
python pubsub_example.py
```

This will start an agent that demonstrates various features like message streaming, tool calls, and more.

2. In another terminal, run the client to see the messages:

```bash
python pubsub_client.py sample_agent
```

You can also run without parameters to see a list of available channels:

```bash
python pubsub_client.py
```

## How It Works

The PubSub system is implemented with the following components:

1. **PubSubManager**: Singleton that manages all channels
2. **PubSubChannel**: Individual channel for each agent
3. **PubSubFormatter**: Formats different types of messages
4. **PubSubDisplay**: Wrapper around ConsoleDisplay that publishes to channels

When an agent is initialized with PubSub enabled, it creates a channel and publishes all output to both the console and the channel. Clients can subscribe to these channels to receive the messages in real-time.

## Customization

You can customize the PubSub behavior by editing `fastagent.config.yaml`:

```yaml
# Enable/disable PubSub globally
pubsub_enabled: true

# Logger settings
logger:
  pubsub_enabled: true  # Enable PubSub in logger
```

## Implementing Your Own Client

To implement your own client:

1. Import the PubSub manager:
```python
from mcp_agent.mcp.pubsub import get_pubsub_manager
```

2. Connect to a channel:
```python
pubsub_manager = get_pubsub_manager()
channel = pubsub_manager.get_or_create_channel("agent_name")
```

3. Subscribe to messages:
```python
async def message_handler(message):
    print(f"Received message: {message}")

await channel.subscribe_async(message_handler)
```

4. Publish messages (for user input):
```python
from mcp_agent.mcp.pubsub_formatter import PubSubFormatter

message = PubSubFormatter.format_user_message("Hello agent!")
await channel.publish(message)
```

## Message Types

The PubSub system uses the following message types:

- `user`: Messages from the user
- `assistant`: Messages from the assistant
- `tool_call`: Tool calls made by the assistant
- `tool_result`: Results from tool calls
- `prompt_loaded`: Information about loaded prompts
- `tool_update`: Information about tool updates