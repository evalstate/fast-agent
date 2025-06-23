import asyncio
import json
from typing import Dict, List
import os
from src.mcp_agent.core.fastagent import FastAgent
from dotenv import load_dotenv
load_dotenv()
import redis.asyncio as aioredis

'''
 redis-cli PUBLISH agent:queen '{"type": "user", "content": "tell me price of polygon please", "channel_id": "agent:queen",
  "metadata": {"model": "haiku3", "name": "default"}}'
'''

worker_bees = [
    {
        "name": "finder",
        "instruction": "You are an agent with access to the internet; you need to search about the latest prices of Bitcoin and other major cryptocurrencies and report back.",
        "servers": ["brave"],
        "model": "haiku3"
    },
    {
        "name": "reporter",
        "instruction": "You are an agent that takes the raw pricing data provided by the finder agent and produces a concise, human-readable summary highlighting current prices, 24-hour changes, and key market insights. You need to post them on twitter and notion page.  ",
        "servers": [],  
        "model": "haiku3"
    }
]

# Sample JSON config for MCP
sample_json_config = {
    "mcp": {
        "servers": {
            # "twitter-mcp": {
            #     "name": "Twitter",
            #     "description" : "Post on Twitter, and get tweets, limited by rate of twitter's developer console.",
            #     "command": "npx",
            #     "args": ["-y", "@enescinar/twitter-mcp"],
            #     "env": {
            #         "API_KEY": os.getenv("TWITTER_API_KEY"),
            #         "API_SECRET_KEY": os.getenv("TWITTER_API_SECRET_KEY"),
            #         "ACCESS_TOKEN": os.getenv("TWITTER_ACCESS_TOKEN"),
            #         "ACCESS_TOKEN_SECRET": os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            #     }
            # },
            # "notion-api": {
            #         "name": "Notion",
            #         "description": "Create pages in Notion and fetch notion content!",
            #         "command": "/Users/vaibhavgeek/hackathons/notion-api-mcp/.venv/bin/python",
            #         "args": ["-m", "notion_api_mcp"],
            #         "env": {
            #             "NOTION_API_KEY": os.getenv("NOTION_API_KEY"),
            #             "NOTION_PARENT_PAGE_ID": os.getenv("NOTION_PARENT_PAGE_ID")
            #         }
            # },
            "brave": {
                "name": "brave",
                "description": "Brave search server, helps you look up internet search results, not very accurate at times. ",
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-brave-search"
                ],
                "env": {
                    "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")
                }
            }
        }
    },
    "default_model": "o3-mini",   
    "logger": {
        "level": "info",
        "type": "console"
    },
    "pubsub_enabled": True,
    "pubsub_config": {
        "backend": "redis",
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
        "api_key": os.getenv("CLAUDE_API_KEY")
    },
    "azure": {
        "api_key": os.getenv("AZURE_API_KEY"),
        "base_url": os.getenv("AZURE_BASE_URL", "https://ai-vaibhavdkm0112ai293833219592.openai.azure.com/"),
        "default_model": "gpt-4o-mini"
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
created_agent_names = create_agents_from_config(worker_bees)

# Create orchestrator with the dynamically created agents
@fast.orchestrator(
    name="orchestrate", 
    agents=created_agent_names,  # Use the list of created agent names
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
                Can you check the price movement of polygon, stellar and bitcoin token, Tweet about the current price action of tokens and staking benefits. 
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
                            
                    except Exception as e:
                        import traceback
                
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
