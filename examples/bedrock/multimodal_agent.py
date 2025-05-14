#!/usr/bin/env python
"""
Multimodal Amazon Bedrock Agent Example
---------------------------------------

This example demonstrates using Amazon Bedrock with multimodal content (images)
in FastAgent. The example shows how to send images to Claude models through Bedrock
for analysis and description.

Before running:
1. Ensure you have AWS credentials configured
2. Make sure you have access to the Amazon Bedrock models in your AWS account
3. Place some image files in the 'images' directory or update the paths in the code

To run this example:
```
python -m examples.bedrock.multimodal_agent
```
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

from mcp_agent.agents.agent import Agent
from mcp_agent.config import FastAgentConfig
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import PromptData
from mcp_agent.mcp.helpers.content_helpers import content_from_file
from mcp_agent.mcp.prompt_message_multipart import Role


async def main() -> None:
    """Run a multimodal agent using Amazon Bedrock Claude model."""
    # Initialize FastAgent with configuration
    fast = FastAgent(FastAgentConfig.from_default_locations())
    
    # Create images directory if it doesn't exist
    images_dir = Path(__file__).parent / "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Place to add your own image paths
    # For this example, we'll check if images exist and provide a message if not
    example_image_path = images_dir / "example.jpg"
    
    if not example_image_path.exists():
        print(f"No image found at {example_image_path}")
        print(f"Please add images to the {images_dir} directory before running this example.")
        return
    
    @fast.agent(model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0")
    async def multimodal_agent(
        prompt: str,
        image_path: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """Create a multimodal agent that can process images and text."""
        messages = []
        
        # Add system message
        messages.append(
            PromptData(
                role=Role.SYSTEM,
                content="You are a helpful AI assistant that can analyze images and text. "
                        "Provide clear, detailed descriptions of what you see in images."
            )
        )
        
        # Create user message with text and optional image
        user_content = prompt
        user_message = PromptData(role=Role.USER, content=user_content)
        
        # Add image content if provided
        if image_path and os.path.exists(image_path):
            image_content = content_from_file(image_path)
            user_message.content = [user_content, image_content]
        
        messages.append(user_message)
        
        # Run the agent with temperature param
        request_params = {"temperature": temperature}
        return await Agent.async_run(messages, request_params=request_params)
    
    # Run the agent with image
    response = await multimodal_agent(
        prompt="What do you see in this image? Provide a detailed description.",
        image_path=str(example_image_path),
        temperature=0.7
    )
    
    # Print the response
    print("\nMultimodal Agent Response:\n")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())