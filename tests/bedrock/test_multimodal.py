#!/usr/bin/env python
"""
Test script for Bedrock multimodal capabilities with actual AWS credentials.
Use this script to test image analysis with Claude Bedrock models.
"""

import os
import asyncio
import argparse
import base64
from typing import Optional, Dict, Any
from pathlib import Path

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from mcp.types import EmbeddedResource, ImageResourceContents
from pydantic import AnyUrl


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Bedrock multimodal capabilities")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--model", default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                     help="Bedrock model ID to test (must support image inputs)")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--query", default="Describe this image in detail.",
                     help="Query to send with the image")
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


def read_image(image_path: str) -> bytes:
    """Read image file as bytes."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    return path.read_bytes()


def get_mime_type(image_path: str) -> str:
    """Determine the MIME type based on file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(ext, "application/octet-stream")


async def main():
    """Run the Bedrock multimodal test."""
    args = parse_args()
    
    # Read the image file
    print(f"Reading image: {args.image}")
    try:
        image_bytes = read_image(args.image)
        mime_type = get_mime_type(args.image)
    except Exception as e:
        print(f"❌ ERROR reading image: {e}")
        return
    
    # Create FastAgent instance
    print(f"Initializing Bedrock agent with model: {args.model}")
    print(f"Region: {args.region}")
    if args.profile:
        print(f"AWS Profile: {args.profile}")
    else:
        print("Using default AWS credentials")
    
    # Create FastAgent with config file or dynamic config
    if args.config:
        fast = FastAgent("Bedrock Multimodal Test", config_path=args.config)
    else:
        config = create_config(args)
        fast = FastAgent("Bedrock Multimodal Test", config_dict=config)

    @fast.agent(
        "vision_agent",
        model=f"bedrock.{args.model}",
        instruction="You are a helpful image analysis assistant running on Amazon Bedrock. Analyze images and respond helpfully."
    )
    async def agent_function():
        """Run the agent function."""
        async with fast.run() as agent:
            try:
                # Create an image resource
                image_resource = ImageResourceContents(
                    uri=AnyUrl(f"file://{os.path.abspath(args.image)}"),
                    mimeType=mime_type,
                    byteValues=list(image_bytes)
                )
                
                # Create an embedded resource
                image_embedded = EmbeddedResource(
                    type="resource",
                    resource=image_resource
                )
                
                # Send the query with the image
                print(f"\nSending query with image: {args.query}")
                
                # Create a multipart prompt with text and image
                prompt = Prompt.user(args.query, image_embedded)
                
                # Send to the model
                response = await agent.vision_agent.send(prompt)
                
                print("\nResponse from Bedrock:")
                print(f"{response}")
                
                print("\n✅ SUCCESS! Bedrock multimodal capabilities are working correctly.")
                
            except Exception as e:
                print(f"\n❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())