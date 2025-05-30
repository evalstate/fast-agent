import asyncio
import json
from typing import Dict, List
import os
from src.mcp_agent.core.fastagent import FastAgent
from dotenv import load_dotenv
load_dotenv()
import redis.asyncio as aioredis

'''
Redis example:
 redis-cli PUBLISH agent:queen '{"type": "user", "content": "tell me price of polygon please", "channel_id": "agent:queen",
  "metadata": {"model": "claude-3-5-haiku-latest", "name": "default"}}'

Kafka example (if using Kafka backend):
 kafka-console-producer --broker-list localhost:9092 --topic mcp_agent_queen
 {"type": "user", "content": "tell me price of polygon please", "channel_id": "agent:queen", "metadata": {"model": "claude-3-5-haiku-latest", "name": "default"}}

MSK example (if using MSK backend):
 python src/msk_producer.py  # Uses the configured MSK cluster
 # The producer will send to topic: mcp_agent_queen

To switch backends:
- Change "backend": "redis" to "backend": "kafka" or "backend": "msk" in pubsub_config  
- For Kafka: Install dependencies: pip install bee-agent[kafka]
- For MSK: Install dependencies: pip install aiokafka aws-msk-iam-sasl-signer boto3
- Set environment variables:
  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION (for MSK)
  - MSK_BOOTSTRAP_SERVERS, MSK_TOPIC_NAME (optional, has defaults)
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
            "google-maps":{
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-google-maps"
                    ],
                "env":{
                    "GOOGLE_MAPS_API_KEY": "AIzaSyCkB37IJcttzInYSunk3IousaabMOXBO20"
                }
            }
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
        "backend": "msk",  # Options: "memory", "redis", "kafka", "msk"
        "channel_name": "queen",
        "msk": {
            "bootstrap_servers": [
                "b-3-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198",
                "b-1-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198", 
                "b-2-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198"
            ],
            "aws_region": "ap-south-1",
            "topic_prefix": "mcp_agent_",
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "OAUTHBEARER",
            "ssl_config": {
                "check_hostname": False,
                "verify_mode": "none"
            },
            "producer_config": {
                "acks": "all",
                "client_id": "mcp_agent_producer"
            },
            "consumer_config": {
                "auto_offset_reset": "latest",
                "enable_auto_commit": True,
                "client_id": "mcp_agent_consumer"
            }
        },
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
           Can you find the price of VANA token and if it is more than 50 percent of it;s lowest then give command to sell it off. tell me now sell it off or hold it. 
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
