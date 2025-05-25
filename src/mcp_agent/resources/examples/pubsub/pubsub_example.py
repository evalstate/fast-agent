#!/usr/bin/env python
"""
PubSub Example for FastAgent

This example demonstrates how to use the PubSub system with FastAgent
to stream agent output to channels instead of the console.

To run the example:
1. Create fastagent.secrets.yaml from the example file
2. Run this script: python pubsub_example.py
3. In another terminal, run the client: python pubsub_client.py sample_agent
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, "../../../..")

from mcp_agent.core.fastagent import FastAgent

sample_json_config = {
    "mcp": {
        "servers": {
            "fetch": {
                "name": "fetch",
                "description": "A server for fetching links",
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
    "default_model": "haiku",   
    "logger": {
        "level": "debug",
        "type": "console"
    },
    # PubSub Configuration
    "pubsub_enabled": True,
    "pubsub_config": {
        "use_redis": True,
        "channel_name": "sample_agent",
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "channel_prefix": "mcp_agent:"
        }
    },
    "anthropic": {
          "api_key": ""
      }
}

# Create a FastAgent instance
fast = FastAgent("PubSub Example",json_config=sample_json_config,)


@fast.agent()
async def sample_agent():
    """
    Sample agent that demonstrates PubSub streaming.
    All output is sent to both the console and the pubsub channel.
    """
    async with fast.run() as agent:
        from mcp_agent.logging.logger import get_logger
        logger = get_logger(__name__)
        
        # Log debug information about the agent and its configuration
        # Check first if agent is an AgentApp wrapper or direct agent
        if hasattr(agent, 'default'):
            # When working with AgentApp
            default_agent = agent.default
            logger.debug(f"Default agent name: {default_agent.name if hasattr(default_agent, 'name') else 'unknown'}")
            logger.debug(f"PubSub enabled: {getattr(default_agent.context.config, 'pubsub_enabled', False)}")
            
            # Check if the agent's LLM has display initialized
            if hasattr(default_agent, '_llm'):
                logger.debug(f"Agent has LLM: {default_agent._llm}")
                if hasattr(default_agent._llm, 'display'):
                    logger.debug(f"LLM display type: {type(default_agent._llm.display)}")
                else:
                    logger.debug("LLM does not have display attribute")
        else:
            # Direct agent access
            logger.debug("Agent wrapper structure unknown - cannot access agent details")
        
        try:
            # Initial message with error handling
            logger.debug("Sending initial message...")
            response = await agent.send("Hello! I'm a sample agent using PubSub streaming.")
            logger.debug("Initial message sent successfully")
            
            # Call a tool to demonstrate tool calls and results
            response = await agent.send(
                "Call the date tool to show the current date and time."
            )
            
            # Use query that requires web search
            response = await agent.send(
                "What are the key benefits of using a pub/sub architecture for communication?"
            )
            
            # Demonstrate streaming a longer response
            response = await agent.send(
                "Write a short story about an AI assistant that learns to communicate through a pub/sub system."
            )
        except Exception as e:
            logger.error(f"Error in agent: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error occurred: {str(e)}"
        
        # Return a helpful message
        return "Demo completed! The agent has run through several messages and tool calls."


if __name__ == "__main__":
    asyncio.run(sample_agent())