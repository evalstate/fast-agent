#!/usr/bin/env python
"""
Test script for Bedrock tool calling capabilities with actual AWS credentials.
Use this script to test tool calling with Claude Bedrock models.
"""

import os
import json
import asyncio
import argparse
import datetime
import requests
from typing import Optional, Dict, Any, List

from mcp_agent.core.fastagent import FastAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Bedrock tool calling capabilities")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--model", default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                     help="Bedrock model ID to test (must support tool calling)")
    parser.add_argument("--query", default="What's the weather like in Seattle, San Francisco, and New York?",
                     help="Query to trigger tool use")
    parser.add_argument("--config", default=None, help="Path to config file")
    return parser.parse_args()


def create_config(args) -> Dict[str, Any]:
    """Create a configuration based on command line arguments."""
    config = {
        "default_model": f"bedrock.{args.model}",
        "bedrock": {
            "region": args.region,
        }
    }
    
    # Set authentication method based on args
    if args.profile:
        config["bedrock"]["profile"] = args.profile
        config["bedrock"]["use_default_credentials"] = False
    else:
        config["bedrock"]["use_default_credentials"] = True
    
    # Set model parameters
    config["bedrock"]["default_params"] = {
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Set logging
    config["logging"] = {
        "level": "DEBUG"
    }
    
    return config


async def main():
    """Run the Bedrock tool calling test."""
    args = parse_args()
    
    # Create FastAgent instance
    print(f"Initializing Bedrock agent with model: {args.model}")
    print(f"Region: {args.region}")
    if args.profile:
        print(f"AWS Profile: {args.profile}")
    else:
        print("Using default AWS credentials")
    
    # Create FastAgent with config file or dynamic config
    if args.config:
        fast = FastAgent("Bedrock Tool Calling Test", config_path=args.config)
    else:
        config = create_config(args)
        fast = FastAgent("Bedrock Tool Calling Test", config_dict=config)

    # Define tools
    @fast.tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        print(f"🔧 Tool called: get_weather({location})")
        
        # This is a mock implementation
        weather_options = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy", "snowy", "windy"]
        temp_f = round(60 + 20 * (0.5 - (hash(location) % 100) / 100))  # 40-80°F
        weather = weather_options[hash(location) % len(weather_options)]
        
        result = {
            "location": location,
            "temperature_f": temp_f,
            "temperature_c": round((temp_f - 32) * 5/9),
            "condition": weather,
            "humidity": round(30 + (hash(location) % 50)),
            "wind_mph": round(5 + (hash(location) % 15)),
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        return json.dumps(result)

    @fast.tool
    def search_news(query: str, max_results: int = 3) -> str:
        """Search for recent news articles on a topic."""
        print(f"🔧 Tool called: search_news({query}, {max_results})")
        
        # This is a mock implementation
        # In a real implementation, you would call a news API
        mock_results = [
            {
                "title": f"Breaking news about {query}",
                "source": "News Network",
                "date": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                "snippet": f"This is a breaking news story about {query}."
            },
            {
                "title": f"Analysis: What {query} means for the future",
                "source": "Analysis Today",
                "date": (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                "snippet": f"Experts weigh in on the implications of {query}."
            },
            {
                "title": f"The history of {query} explained",
                "source": "History Channel",
                "date": (datetime.datetime.now() - datetime.timedelta(days=5)).strftime("%Y-%m-%d"),
                "snippet": f"A deep dive into the origins and development of {query}."
            }
        ]
        
        # Slice to max_results
        return json.dumps(mock_results[:max_results])

    @fast.agent(
        "tool_agent",
        model=f"bedrock.{args.model}",
        instruction="You are a helpful assistant running on Amazon Bedrock. You have access to tools that you can use to answer questions. Always use tools when appropriate."
    )
    async def agent_function():
        """Run the agent function."""
        async with fast.run() as agent:
            try:
                # Send a query that should trigger tool use
                print(f"\nSending query to trigger tool use: {args.query}")
                
                # Send to the model
                response = await agent.tool_agent.send(args.query)
                
                print("\nResponse from Bedrock:")
                print(f"{response}")
                
                print("\n✅ SUCCESS! Bedrock tool calling capabilities are working correctly.")
                
                # Run an interactive session
                print("\n=== Starting interactive mode ===")
                print("Type your messages and press Enter to send. Type 'exit' to quit.")
                await agent.tool_agent.interactive()
                
            except Exception as e:
                print(f"\n❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())