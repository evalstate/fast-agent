#!/usr/bin/env python
"""
Simple Amazon Bedrock Agent Example
-----------------------------------

This example demonstrates how to create a simple agent using Amazon Bedrock models
with FastAgent. The example shows the basic pattern for creating and interacting
with an agent using Claude models through Bedrock.

Before running:
1. Ensure you have AWS credentials configured (environment variables, AWS credentials file, or IAM role)
2. Make sure you have access to the Amazon Bedrock models in your AWS account

To run this example:
```
python -m examples.bedrock.simple_agent
```

For more information on AWS Bedrock:
- https://aws.amazon.com/bedrock/
- https://docs.aws.amazon.com/bedrock/
"""

import asyncio
from typing import List, Optional

from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.prompt_message_multipart import Role, PromptMessageMultipart


async def main() -> None:
    """Run a simple agent using Amazon Bedrock Claude model."""
    # Initialize FastAgent with configuration (will use fastagent.config.yaml in this directory)
    fast = FastAgent(get_settings())
    
    # Define the agent decorator - we'll use Claude on Bedrock
    # You can use the full model name or create an alias in your configuration
    @fast.agent(model="bedrock.us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    async def bedrock_agent(
        prompt: str, 
        temperature: Optional[float] = None,
        context: Optional[List[str]] = None
    ) -> str:
        """Create a simple agent that uses Bedrock Claude."""
        messages = []
        
        # Add context if provided
        if context:
            messages.append(
                PromptMessageMultipart(
                    role="system", 
                    content=["\n\n".join(context)]
                )
            )
        
        # Add user prompt
        messages.append(
            PromptMessageMultipart(
                role="user",
                content=[prompt]
            )
        )
        
        # Set temperature if provided
        request_params = None
        if temperature is not None:
            request_params = {"temperature": temperature}
        
        # Return the agent's response
        return await Agent.async_run(messages, request_params=request_params)
    
    # Define system context
    system_context = [
        "You are a helpful AI assistant that provides clear, accurate information.",
        "Be concise and helpful in your responses."
    ]
    
    # Run the agent with a prompt
    response = await bedrock_agent(
        prompt="What makes foundation models like Claude useful for applications?",
        temperature=0.7,
        context=system_context
    )
    
    # Print the response
    print("\nAgent Response:\n")
    print(response)
    
    # Example with a different prompt - note we're reusing the same agent function
    print("\n" + "-" * 50 + "\n")
    response2 = await bedrock_agent(
        prompt="Generate a short poem about artificial intelligence.",
        temperature=0.9,  # Higher temperature for more creative outputs
        context=system_context
    )
    
    # Print the response
    print("\nAgent Response:\n")
    print(response2)


if __name__ == "__main__":
    asyncio.run(main())