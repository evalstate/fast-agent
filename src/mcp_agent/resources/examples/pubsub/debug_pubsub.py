#!/usr/bin/env python
"""
Debug script for the PubSub issue in the sample_agent.py example.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, "../../../..")

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

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
    # Enable PubSub in the config
    "pubsub_enabled": True,
    "anthropic": {
        "api_key": ""
    }
}

# Create a FastAgent instance
fast = FastAgent("PubSub Debug", json_config=sample_json_config)

@fast.agent()
async def debug_agent():
    """
    Debug agent that confirms PubSub configuration is properly passed through.
    """
    async with fast.run() as agent:
        # Debug logging to verify config attributes
        # Check first if agent is an AgentApp wrapper or direct agent
        if hasattr(agent, 'default'):
            # When working with AgentApp
            default_agent = agent.default
            logger.debug(f"FastAgent config: pubsub_enabled={getattr(default_agent.context.config, 'pubsub_enabled', False)}")
            
            # Check if the agent's LLM has display initialized
            if hasattr(default_agent, '_llm'):
                logger.debug(f"Agent has LLM: {default_agent._llm}")
                if hasattr(default_agent._llm, 'display'):
                    logger.debug(f"Agent LLM display: {default_agent._llm.display}")
                    # Try to access the display's show_user_message method
                    if default_agent._llm.display is not None:
                        logger.debug(f"Display has show_user_message: {hasattr(default_agent._llm.display, 'show_user_message')}")
                else:
                    logger.debug("Agent LLM does not have a display attribute")
                
                # Log agent and LLM attributes
                logger.debug(f"Agent name: {default_agent.name}")
                logger.debug(f"Agent LLM name: {default_agent._llm.name}")
        else:
            logger.debug("Agent wrapper structure unknown - cannot access agent details")
        
        # Initial message with try/except to catch errors
        try:
            response = await agent.send("Hello! I'm a test message for debugging.")
            logger.debug("Message sent successfully")
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Return success message
        return "Debug completed"

if __name__ == "__main__":
    asyncio.run(debug_agent())