#!/usr/bin/env python3
"""
Test script to verify FastAgent functionality with JSON config loading.
Dynamic agent creation from JSON configuration.
"""

import asyncio
import json
from typing import Dict, List
from rich import print as rich_print
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
        "model": "haiku"  # Optional: specify model per agent
    },
    {
        "name": "reporter",
        "instruction": "You are an agent that takes the raw pricing data provided by the finder agent and produces a concise, human-readable summary highlighting current prices, 24-hour changes, and key market insights.",
        "servers": [],  # No specific servers needed for this agent
        "model": "haiku"  # Optional: specify model per agent
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
                        "time_to_confirm": 120000,  # 2 minutes 
                        "default": "reject"  # if the time expires then what to do. 
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
        "channel_name": "queen",  # This must match the agent name or channel used for publishing
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "channel_prefix": "agent:"
        }
    },
    "anthropic": {
        "api_key": os.environ.get("CLAUDE_API_KEY", "")  # Fixed typo: was CALUDE_API_KEY
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
            rich_print(f"[red]Warning: Agent config missing name, skipping: {agent_config}[/red]")
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
        rich_print(f"[green]Created agent: {name}[/green]")
    
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
    rich_print("Testing FastAgent initialization with JSON config...")
    rich_print(f"[blue]Created agents: {created_agent_names}[/blue]")
    
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
           Can you find the price of VANA token and if it is more than 50 percent of it;s lowest then give command to sell it off. tell me now sell it off or hold it. 
            """
            
            rich_print("[cyan]Starting initial orchestration task...[/cyan]")
            await agent.orchestrate(initial_task)
            rich_print("[green]Initial task completed![/green]")
            
            # Keep running and listen for Redis messages
            rich_print("[yellow]Listening for Redis messages on channel 'agent:queen'...[/yellow]")
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
                                rich_print(f"[blue]Received user input:[/blue] {user_input}")
                                
                                # Send to orchestrator instead of individual agent
                                response = await agent.orchestrate(user_input)
                                rich_print(f"[green]Orchestrator response:[/green] {response}")
                                
                        except json.JSONDecodeError:
                            rich_print(f"[red]Received non-JSON message:[/red] {data}")
                            # Try to process as plain text
                            response = await agent.orchestrate(data)
                            rich_print(f"[green]Orchestrator response:[/green] {response}")
                            
                    except Exception as e:
                        rich_print(f"[bold red]Error processing Redis message:[/bold red] {e}")
                        import traceback
                        rich_print(f"[dim red]{traceback.format_exc()}[/dim red]")
                
                # Small delay to prevent CPU spike
                await asyncio.sleep(0.05)
                
        except asyncio.CancelledError:
            rich_print("[yellow]Agent was cancelled[/yellow]")
        except KeyboardInterrupt:
            rich_print("[yellow]Agent stopped by user[/yellow]")
        finally:
            # Clean up Redis connection
            if 'pubsub' in locals():
                await pubsub.unsubscribe("agent:queen")
            await redis_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rich_print("\n[yellow]Agent stopped by user[/yellow]")