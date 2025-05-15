#!/usr/bin/env python
"""
Multimodal Amazon Bedrock Agent Example
---------------------------------------

This example demonstrates using Amazon Bedrock with multimodal content (images)
in FastAgent. The example shows how to send images to Claude models through Bedrock
for analysis and description, and how to maintain context in follow-up questions.

Before running:
1. Ensure you have AWS credentials configured with Bedrock access
2. Make sure you have access to the Claude 3.5 Sonnet model in your AWS account
3. Check your AWS Bedrock quota limits for multimodal requests
4. Place image files in the 'images' directory (new_york.jpg is included)

Known issues:
- AWS Bedrock may throttle multiple multimodal requests - the script includes
  a 10-second delay between requests to mitigate this issue
- If you encounter throttling errors, you may need to increase the delay or
  request higher quota limits in your AWS account

To run this example:
```
python -m examples.bedrock.multimodal_agent
```
"""

import asyncio
import base64
import os
import mimetypes
import time
from pathlib import Path
from typing import Optional, Union

from mcp_agent import FastAgent
from mcp.types import ImageContent
from mcp_agent.core.prompt import Prompt


def read_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Read an image file and convert it to base64 encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (base64_data, mime_type)
    """
    # Guess mime type
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        # Default to JPEG if we can't determine the type
        mime_type = "image/jpeg"
    
    # Read the file as binary and encode as base64
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data).decode("utf-8")
    
    return base64_data, mime_type


def create_image_content_from_file(image_path: str) -> ImageContent:
    """
    Create an ImageContent object from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ImageContent object ready to send to the model
    """
    # Read the image and get base64 data and mime type
    base64_data, mime_type = read_image_to_base64(image_path)
    
    # Create and return the ImageContent object
    return ImageContent(
        type="image",
        data=base64_data,
        mimeType=mime_type
    )


async def main() -> None:
    """Run a multimodal agent using Amazon Bedrock Claude model."""
    # Initialize timer for measuring performance
    start_time = time.time()
    
    # Initialize FastAgent with configuration
    print("Initializing FastAgent with AWS Bedrock configuration...")
    fast = FastAgent("Bedrock Multimodal Demo", config_path="examples/bedrock/fastagent.config.yaml")
    
    # Create images directory if it doesn't exist
    images_dir = Path(__file__).parent / "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Use the new_york.jpg image that's been added
    image_path = images_dir / "new_york.jpg"
    
    # Check if the image exists
    if not image_path.exists():
        print(f"Image not found at {image_path}")
        # Fall back to example.jpg if new_york.jpg doesn't exist
        image_path = images_dir / "example.jpg"
        if not image_path.exists():
            print(f"No fallback image found at {image_path}")
            print(f"Please add images to the {images_dir} directory before running this example.")
            return
    
    print(f"Using image: {image_path}")
    print(f"Initialization time: {time.time() - start_time:.2f} seconds")
    
    # Create an agent with Claude model that supports vision
    @fast.agent(
        "multimodal_agent",
        "You are a helpful AI assistant that can analyze images and text. "
        "Provide clear, detailed descriptions of what you see in images.",
        model="bedrock.us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    async def bedrock_multimodal():
        """Placeholder function for the agent decorator."""
        pass
    
    # Run the agent
    try:
        async with fast.run() as agent:
            # First request - initial image analysis
            request_start = time.time()
            print("Sending image to Bedrock Claude model...")
            prompt = Prompt.user(
                "What do you see in this image? What city is shown? Provide a detailed description.",
                image_path  # Pass the Path object directly - Prompt.user will handle it
            )
            
            # Send the prompt to the agent
            try:
                print("Request sent, waiting for response...")
                response = await agent.multimodal_agent.send(prompt)
                print(f"Response received in {time.time() - request_start:.2f} seconds")
                
                # Print the response
                print("\nMultimodal Agent Response:\n")
                print("-" * 50)
                print(response)
                print("-" * 50)

                # Optional: Try a follow-up question about the same image
                print("\nWaiting 10 seconds before sending follow-up question...")
                await asyncio.sleep(10)  # Increased delay to better avoid throttling
                
                print("Sending follow-up question...")
                follow_up_start = time.time()
                # Include the image again in the follow-up question to maintain context
                follow_up_prompt = Prompt.user(
                    "What are some famous landmarks or attractions visible in this image?",
                    image_path  # Include the image again
                )
                
                try:
                    print("Follow-up request sent, waiting for response...")
                    follow_up = await agent.multimodal_agent.send(follow_up_prompt)
                    print(f"Follow-up received in {time.time() - follow_up_start:.2f} seconds")
                    
                    print("\nFollow-up Response:\n")
                    print("-" * 50)
                    print(follow_up)
                    print("-" * 50)
                except Exception as e:
                    print(f"\nError during follow-up: {e}")
                    print("This may be due to AWS Bedrock throttling.")
                    print("Consider:")
                    print("  - Increasing the delay between requests further")
                    print("  - Checking your AWS Bedrock quotas in the AWS Console")
                    print("  - Requesting a quota increase for Claude models if necessary")
                
            except Exception as e:
                print(f"Error sending image to Claude: {e}")
                print("Check your AWS credentials and Bedrock model access permissions.")
                return
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())