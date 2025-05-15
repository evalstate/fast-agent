#!/usr/bin/env python
"""
Fixed Bedrock Agent Example using FastAgent
"""

import asyncio
import sys
import traceback
import boto3
from mcp_agent import FastAgent

async def check_bedrock_access():
    """Verify if we can access Bedrock directly with boto3."""
    print("Checking direct AWS Bedrock access...")
    try:
        # Create a Bedrock runtime client
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        
        # List available models (this just tests connectivity)
        print("AWS credentials found, testing connection...")
        return True
    except Exception as e:
        print(f"Error accessing AWS Bedrock directly: {e}")
        print("Please check your AWS credentials and permissions.")
        return False

async def main():
    """Run a simple agent using FastAgent with Bedrock."""
    
    print("Initializing FastAgent with Bedrock configuration...")
    
    # First check direct Bedrock access
    access_ok = await check_bedrock_access()
    if not access_ok:
        print("AWS Bedrock access check failed. Continuing anyway to see detailed errors...")
    
    try:
        # Initialize FastAgent (will load fastagent.config.yaml from the current directory)
        fast = FastAgent("Bedrock Test", config_path="examples/bedrock/fastagent.config.yaml")
        
        print("FastAgent initialized successfully.")
        print("Creating bedrock agent with Claude 3.5 Sonnet model...")
        
        # Define a bedrock agent
        @fast.agent(
            "bedrock_agent",  # Use underscore instead of hyphen in name
            "You are a helpful assistant that provides concise responses.",
            model="bedrock.us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # Using full model ID
        )
        async def run_bedrock_agent():
            # This function is just a placeholder for the decorator
            # The actual execution happens below
            pass
        
        print("\nSending a message to the Bedrock agent...")
        
        try:
            # Create agent context manager
            async with fast.run() as agent:
                print("Agent context created, sending first message to bedrock_agent...")
                
                # Send a message to the agent by name
                response = await agent.bedrock_agent.send("Tell me about AWS Bedrock in 2-3 sentences.")
                
                print("\nResponse from Bedrock agent:")
                print("-" * 50)
                print(response)  # Print the response directly
                print("-" * 50)
                
                # Send a follow-up message
                response = await agent.bedrock_agent.send("What model am I currently using?")
                
                print("\nFollow-up response:")
                print("-" * 50)
                print(response)  # Print the response directly
                print("-" * 50)
        except Exception as e:
            print(f"\nError during agent execution: {e}")
            traceback.print_exc()
            print("\nTroubleshooting tips:")
            print("1. Check AWS credentials in ~/.aws/credentials or environment variables")
            print("2. Verify you have access to Claude models in AWS Bedrock console")
            print("3. Check network connectivity to AWS Bedrock endpoints")
            print("4. Look for model ID format errors (should include region prefix)")
        
    except Exception as e:
        print(f"\nError during FastAgent initialization: {e}")
        traceback.print_exc()
        print("\nCheck the fastagent.config.yaml file for errors.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnhandled error: {e}")
        traceback.print_exc()