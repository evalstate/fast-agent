#!/usr/bin/env python
"""
Simple test script for FastAgent with Bedrock.
This script uses the fastagent package to test AWS Bedrock integration.
"""

import os
import asyncio
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple FastAgent Bedrock test")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--model", default="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                     help="Bedrock model ID to test")
    return parser.parse_args()

async def main():
    """Run the FastAgent Bedrock test."""
    args = parse_args()
    
    print(f"Testing FastAgent with Bedrock model: {args.model}")
    print(f"Region: {args.region}")
    if args.profile:
        print(f"AWS Profile: {args.profile}")
    else:
        print("Using default AWS credentials")
    
    try:
        # Import FastAgent from the correct package
        try:
            from mcp_agent.core.fastagent import FastAgent
            print("Successfully imported FastAgent from mcp_agent.core.fastagent")
        except ImportError:
            from fastagent import FastAgent
            print("Successfully imported FastAgent from fastagent")
        
        # Set environment variables for AWS credentials
        if args.profile:
            os.environ["AWS_PROFILE"] = args.profile
        
        # Set AWS region
        os.environ["AWS_REGION"] = args.region
        
        # Set verbose logging
        os.environ["FASTAGENT_LOG_LEVEL"] = "DEBUG"
        
        # Create FastAgent instance
        print("Creating FastAgent instance...")
        agent = FastAgent("Bedrock Test")
        print("FastAgent instance created")
        
        # Prepare model ID format
        if args.model.startswith("us.") or args.model.startswith("eu."):
            model_id = f"bedrock.{args.model}"  # Add bedrock. prefix for fully qualified IDs
        else:
            model_id = f"bedrock.{args.model}"  # Add bedrock. prefix for aliases
        
        print(f"Using model ID: {model_id}")
        
        # Define a simple agent
        @agent.agent(name="bedrock_test", model=model_id)
        async def bedrock_agent(prompt):
            """Simple Bedrock agent function."""
            # Run the agent with async context manager
            async with agent.run() as session:
                # Send a message to the agent
                response = await session.bedrock_test.send(prompt)
                return response
        
        # Run the agent
        print("Running Bedrock agent...")
        response = await bedrock_agent("Please identify yourself and confirm you are working.")
        
        print("\nResponse from Bedrock agent:")
        print(response)
        
        print("\n✅ SUCCESS! FastAgent with Bedrock is working correctly.")
    
    except ImportError as e:
        print(f"❌ ERROR: Could not import FastAgent. Is it installed? Error: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())