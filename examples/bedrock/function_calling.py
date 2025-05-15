#!/usr/bin/env python
"""
Function Calling Example with AWS Bedrock Claude 3.5 Sonnet model

This example demonstrates how to use the tool/function calling capabilities
of Claude 3.5 Sonnet on AWS Bedrock. Function calling is supported by:
- Claude 3.5 Sonnet
- Claude 3.7 Sonnet
- Claude 3 Opus

REQUIREMENTS TO RUN THIS EXAMPLE:
1. AWS credentials with access to AWS Bedrock
2. Access to Claude 3.5 Sonnet (or another Claude model that supports tool calling)
3. boto3 library installed: pip install boto3
4. FastAgent installed: pip install fast-agent-mcp[bedrock]

This example shows:
- How to define tools/functions using Python functions decorated with @agent.tool
- How to handle tool calls and responses
- How to implement multiple tools with different parameters

Note: This is just a demonstration with a mock weather service.
In a real application, you would implement actual API calls to external services.
"""

import asyncio
import json
import datetime
from typing import Dict, Any, List, Optional

from mcp_agent import FastAgent


async def main():
    """Main function for the function calling example"""
    print("=== AWS Bedrock Function Calling Example ===")
    
    # Initialize FastAgent with configuration
    fast = FastAgent("Function Calling Demo", config_path="examples/bedrock/fastagent.config.yaml")
    
    # Define tool functions
    def get_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
        """
        Get the current weather for a location.
        
        Args:
            location: The name of the city or location
            unit: Temperature unit, either "celsius" or "fahrenheit"
        
        Returns:
            Weather data including temperature, condition, and other details
        """
        print(f"🔧 Tool called: get_weather({location}, {unit})")
        
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
        
        return result
    
    def search_web(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search the web for current information.
        
        Args:
            query: The search query term or phrase
            num_results: Number of results to return (default: 3)
        
        Returns:
            List of search results with title, source, date and snippet
        """
        print(f"🔧 Tool called: search_web({query}, {num_results})")
        
        # Mock implementation
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
        
        return mock_results[:num_results]
        
    # Define a Claude agent with function calling capability
    @fast.agent(
        "bedrock_agent",
        "You are a helpful assistant with the ability to use tools.",
        model="us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # Claude 3.5 Sonnet
    )
    async def bedrock_agent():
        pass
    
    # Convert regular functions to tools using mcp.types.Tool
    from mcp.types import Tool
    
    # Create Tool objects that will be used by FastAgent
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        inputSchema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The name of the city or location"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    )
    
    search_tool = Tool(
        name="search_web",
        description="Search the web for current information",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query term or phrase"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }
    )
            
    async with fast.run() as agent:
        # Register the tools with the agent
        print("Registering tools with the agent...")
        agent.bedrock_agent.tools = [weather_tool, search_tool]
        
        # Map tool names to functions for handling tool calls
        tool_handlers = {
            "get_weather": get_weather,
            "search_web": search_web
        }
        
        # Ask a question that should trigger tool use
        print("\nSending question to agent...")
        response = await agent.bedrock_agent.send(
            "What's the weather like in Seattle right now, and what activities would you recommend based on the weather?"
        )
        
        print("\nFirst Agent Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"\nTool called: {tool_call.name}")
            print(f"Arguments: {json.dumps(tool_call.arguments, indent=2)}")
            
            # Call the appropriate tool function
            if tool_call.name in tool_handlers:
                handler = tool_handlers[tool_call.name]
                tool_result = handler(**tool_call.arguments)
                
                # Send tool result back to the model
                print("\nSending tool response back to the agent...")
                final_response = await agent.bedrock_agent.send(
                    "",  # Empty message for tool response
                    tool_results=[{
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "result": tool_result
                    }]
                )
                
                print("\nFinal Agent Response:")
                print("-" * 50)
                print(final_response)
                print("-" * 50)
        
        # To start an interactive session:
        print("\n=== Starting interactive mode ===")
        print("Type your messages and press Enter to send. Type 'exit' to quit.")
        await agent.bedrock_agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())