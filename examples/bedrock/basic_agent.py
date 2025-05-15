#!/usr/bin/env python
"""
Basic Bedrock Agent Example using FastAgent
"""

import asyncio
from mcp_agent import FastAgent

async def main():
    """Run a simple agent using FastAgent with Bedrock."""
    
    print("Initializing FastAgent with Bedrock configuration...")
    
    # Initialize FastAgent (will load fastagent.config.yaml from the current directory)
    fast = FastAgent("Bedrock Test", config_path="examples/bedrock/fastagent.config.yaml")
    
    # Define a bedrock agent
    @fast.agent(
        "bedrock-agent",
        "You are a helpful assistant that provides concise responses.",
        model="bedrock.us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # Using full model ID
    )
    async def run_agent():
        print("\nSending a message to the Bedrock agent...")
        
        # Create agent context manager
        async with fast.run() as agent:
            # Send a message to the agent
            response = await agent.chat("Tell me about AWS Bedrock in 2-3 sentences.")
            
            print("\nResponse from Bedrock agent:")
            print("-" * 50)
            print(response.content)
            print("-" * 50)
            
            # Send a follow-up message
            response = await agent.chat("What model am I currently using?")
            
            print("\nFollow-up response:")
            print("-" * 50)
            print(response.content)
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())