#!/usr/bin/env python
"""
A simple agent script for testing the Bedrock integration.
This script will be useful once the Bedrock provider is implemented.
"""

import asyncio
import argparse
from mcp_agent.core.fastagent import FastAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Test Bedrock integration with fast-agent")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--model", default=None, help="Model to use (overrides config)")
    return parser.parse_args()

# Create the application
fast = FastAgent("Bedrock Test Agent", config_path="fastagent.config.yaml")

@fast.agent(
    "bedrock_test",
    "You are a helpful assistant running on Amazon Bedrock. Answer questions concisely and accurately."
)
async def main():
    """Main function to run the interactive prompt."""
    args = parse_args()
    
    async with fast.run() as agent:
        # Start interactive prompt
        print("\n=== Starting interactive prompt with Bedrock test agent ===")
        print("Type 'exit' or press Ctrl+C to quit\n")
        
        try:
            # First, test a simple message
            print("Testing with simple query...\n")
            response = await agent("Please identify yourself and which model you are running on")
            
            # Then start interactive mode
            print("\n=== Now starting interactive mode ===")
            print("Type your messages and press Enter to send\n")
            
            await agent.interactive()
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    asyncio.run(main())