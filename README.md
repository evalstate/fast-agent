<div align="center">

<h1>CommandHive Agents: Queen and Worker Bees</h1>

**Create agents on the fly, manage MCP variables, and execute tasks 24/7.**

[![Visit Website](https://img.shields.io/badge/Website-CommandHive-blue?style=plastic\&logo=internet\&logoColor=white)](https://commandhive.xyz)
[![GitHub Repo](https://img.shields.io/badge/GitHub-CommandHive-green?style=plastic\&logo=github\&logoColor=white)](https://github.com/CommandHive)

[![Watch Demo on YouTube](https://img.youtube.com/vi/306nzTWAZY8/0.jpg)](https://www.youtube.com/watch?v=306nzTWAZY8)

</div>

## Latest updates

| Date       | Update Description                                                         |
| ---------- | -------------------------------------------------------------------------- |
| 2025/06/03 | Released sample orchestrator and listener scripts                          |
| 2025/05/27 | Added Pub/Sub support for MSK and confirmed tool calling with confirmation |
| 2025/05/15 | Ensured sampling as MCP feature is working                                 |

For a full changelog, see our [releases page](https://github.com/CommandHive/agent-orchestrator/releases).

---

## Why CommandHive?

**🏆 Build flexible multi-agent orchestrations.** Spin up agents dynamically based on MCP configurations and orchestrate complex workflows without rigid architectures.

**🔧 Seamlessly integrate with existing tools.** Use Redis, Kafka, and MSK for Pub/Sub messaging; connect to Model Context Protocol servers; and configure agents with custom instructions and models.

**🚀 Production-ready controls.** Compatible with Python 3.11 and 3.12, supports logging, error handling, and run-time configuration via JSON.

> \[!TIP]
> Start quickly by exploring the sample agent and listener below to see how CommandHive manages agents and messaging.

---

## Installation

To install via pip:

```shell
pip install bee-agent
```

---

## Sample Agent

```python
import asyncio
import json
from typing import Dict, List
import os
from mcp_agent.core.fastagent import FastAgent
from dotenv import load_dotenv
load_dotenv()
import redis.asyncio as aioredis

'''
 redis-cli PUBLISH agent:queen '{"type": "user", "content": "tell me price of polygon please", "channel_id": "agent:queen",
  "metadata": {"model": "claude-3-5-haiku-latest", "name": "default"}}'
'''

subagents_config = [
    {
        "name": "finder",
        "instruction": "You are an agent with access to the internet; you need to search about the latest prices of Bitcoin and other major cryptocurrencies and report back.",
        "servers": ["fetch", "brave"],
        "model": "haiku"
    },
    {
        "name": "reporter",
        "instruction": "You are an agent that takes the raw pricing data provided by the finder agent and produces a concise, human-readable summary highlighting current prices, 24-hour changes, and key market insights.",
        "servers": [],  
        "model": "haiku"
    }
]

# Sample JSON config for MCP
sample_json_config = {
    "mcp": {
        "servers": {
            "fetch": {
                "name": "fetch",
                "description": "A server for fetching links",
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "tool_calls": [
                    {
                        "name": "fetch",
                        "seek_confirm": True,
                        "time_to_confirm": 120000,
                        "default": "reject"
                    }
                ]
            },
            "brave": {
                "name": "brave",
                "description": "Brave search server",
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-brave-search"
                ],
                "env": {
                    "BRAVE_API_KEY": "BSANIwUPPxwC9wchogL5I6UNkWGffh3"
                }
            }
        }
    },
    "default_model": "haiku",
    "logger": {
        "level": "info",
        "type": "console"
    },
    "pubsub_enabled": True,
    "pubsub_config": {
        "use_redis": True,
        "channel_name": "queen",
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "channel_prefix": "agent:"
        }
    },
    "anthropic": {
        "api_key": os.environ.get("CLAUDE_API_KEY", "")
    }
}

# Create FastAgent instance
fast = FastAgent(
    name="queen",  # Changed name to match channel name used in publishing
    json_config=sample_json_config,
    parse_cli_args=False
)

# Dynamically create agents from JSON configuration using a for loop
def create_agents_from_config(config_list: List[Dict]) -> List[str]:
    """
    Create agents dynamically from JSON configuration.
    Returns a list of agent names for use in the orchestrator.
    """
    agent_names = []
    
    for agent_config in config_list:
        name = agent_config.get("name")
        instruction = agent_config.get("instruction", "")
        servers = agent_config.get("servers", [])
        model = agent_config.get("model", None)
        
        if not name:
            continue
            
        # Create agent decorator kwargs
        agent_kwargs = {
            "name": name,
            "instruction": instruction,
            "servers": servers
        }
        
        # Add model if specified
        if model:
            agent_kwargs["model"] = model
            
        # Create the agent using the decorator
        @fast.agent(**agent_kwargs)
        def agent_function():
            """Dynamically created agent function"""
            pass
            
        agent_names.append(name)
    
    return agent_names

# Create agents from configuration
created_agent_names = create_agents_from_config(subagents_config)

# Create orchestrator with the dynamically created agents
@fast.orchestrator(
    name="orchestrate", 
    agents=created_agent_names,  # Use the list of created agent names
    plan_type="full",
    model="haiku"
)
async def orchestrate_task():
    """Orchestrator function"""
    pass

async def main():
    """Test initializing FastAgent with JSON config in interactive mode."""
    
    # Create Redis client
    redis_client = aioredis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )
    
    # Register agents and keep it running
    async with fast.run() as agent:
        try:
            # Subscribe to the input channel
            pubsub = redis_client.pubsub()
            await pubsub.subscribe("agent:queen")
            
            # Initial task for the orchestrator
            initial_task = """
           Can you find the price of VANA token and if it is more than 50 percent of its lowest then give command to sell it off. tell me now sell it off or hold it.
            """
            
            await agent.orchestrate(initial_task)
            
            # Keep running and listen for Redis messages
            while True:
                # Process Redis messages directly
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                if message and message.get('type') == 'message':
                    try:
                        # Process the message data
                        data = message.get('data')
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        
                        # Try to parse JSON
                        try:
                            data_obj = json.loads(data)
                            
                            # If this is a user message, extract content and send to orchestrator
                            if data_obj.get('type') == 'user' and 'content' in data_obj:
                                user_input = data_obj['content']
                                
                                # Send to orchestrator instead of individual agent
                                response = await agent.orchestrate(user_input)
                                
                        except json.JSONDecodeError:
                            # Try to process as plain text
                            response = await agent.orchestrate(data)
                            
                    except Exception:
                        pass
                
                # Small delay to prevent CPU spike
                await asyncio.sleep(0.05)
                
        finally:
            # Clean up Redis connection
            if 'pubsub' in locals():
                await pubsub.unsubscribe("agent:queen")
            await redis_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
```

---

## Sample Listener

```python
import redis
import time

def main():
    # Connect to Redis (adjust host/port/db as needed)
    r = redis.Redis(host='localhost', port=6379, db=0)

    # Create a PubSub object and subscribe to the channel
    pubsub = r.pubsub()
    channel_name = 'agent:queen'
    pubsub.subscribe(channel_name)
    print(f"Subscribed to channel: {channel_name}")

    # Loop forever, polling for new messages
    while True:
        message = pubsub.get_message()
        if message:
            # Print the raw message dict
            print(message)
        # Sleep briefly to avoid busy-waiting
        time.sleep(0.05)

if __name__ == '__main__':
    main()
```
## ✅ Completed

- [x] Ensure sampling as MCP feature is working
- [x] Create Pub/Sub support using Redis
- [x] Create Pub/Sub support using Kafka
- [x] Create Pub/Sub support using MSK (Managed Kafka)
- [x] Ensure tool calling is done with confirmation
- [x] Check compatibility with Python 3.11, Python 3.12

---

## ⏳ To Do Next

* [ ] Integrate smart contract for tool calling
* [ ] Each agent has its own wallet (defined in config)
* [ ] Pub/Sub support for RabbitMQ
* [ ] Pub/Sub support for Google Pub/Sub

---

## Disclaimer

This repo is cloned from [fast-agent](https://github.com/evalstate/fast-agent/).
