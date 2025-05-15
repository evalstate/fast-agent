#!/usr/bin/env python
"""
Test script for Anthropic Claude models on AWS Bedrock
This script tests Claude 3.5 and Claude 3.7 models

REQUIREMENTS TO RUN THIS TEST:
1. AWS credentials with access to AWS Bedrock
2. Access to Claude 3.5 and Claude 3.7 models in your AWS account
3. boto3 library installed: pip install boto3
4. FastAgent installed: pip install fast-agent-mcp[bedrock]

The script tests:
- Direct API access to Claude 3.5 and Claude 3.7 using boto3
- FastAgent integration with Claude 3.5 and Claude 3.7
- Proper message formatting with anthropic_version parameter

If you don't have access to both Claude 3.5 and Claude 3.7:
- The script will still partially work with access to just one model
- You'll see errors only for the models you don't have access to
"""

import asyncio
import json
import os
import sys
import boto3
from typing import Dict, Any, List, Optional
import logging

# Import FastAgent components
from mcp_agent import FastAgent
from mcp_agent.logging.logger import get_logger

# Set up logging
logger = get_logger("anthropic_claude_test")


async def test_claude_3_5():
    """Test Anthropic Claude 3.5 Sonnet model"""
    print("\n" + "="*50)
    print("Testing Anthropic Claude 3.5 Sonnet")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        
        # Claude 3.5 Sonnet model ID
        model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        
        # Claude uses the messages format with anthropic_version
        request_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What is AWS Bedrock? Respond in 2-3 sentences."}
                    ]
                }
            ]
        })
        
        print(f"Sending request to {model_id}...")
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        if "content" in response_body and isinstance(response_body["content"], list):
            text_content = []
            
            # Extract text from response
            for content_item in response_body["content"]:
                if content_item.get("type") == "text":
                    text_content.append(content_item.get("text", ""))
            
            # Join text content
            text = "\n".join(text_content)
            
            print("\nResponse from Claude 3.5 Sonnet:")
            print("-" * 50)
            print(text)
            print("-" * 50)
            
            # Print usage if available
            if "usage" in response_body:
                print(f"Token usage: {response_body['usage']}")
                
            return True
        else:
            print(f"Unexpected response format: {list(response_body.keys())}")
            return False
            
    except Exception as e:
        print(f"Error testing Claude 3.5 Sonnet: {str(e)}")
        return False


async def test_claude_3_7():
    """Test Anthropic Claude 3.7 Sonnet model"""
    print("\n" + "="*50)
    print("Testing Anthropic Claude 3.7 Sonnet")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        
        # Claude 3.7 Sonnet model ID
        model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        
        # Claude uses the messages format with anthropic_version
        request_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What is AWS Bedrock? Respond in 2-3 sentences."}
                    ]
                }
            ]
        })
        
        print(f"Sending request to {model_id}...")
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        if "content" in response_body and isinstance(response_body["content"], list):
            text_content = []
            
            # Extract text from response
            for content_item in response_body["content"]:
                if content_item.get("type") == "text":
                    text_content.append(content_item.get("text", ""))
            
            # Join text content
            text = "\n".join(text_content)
            
            print("\nResponse from Claude 3.7 Sonnet:")
            print("-" * 50)
            print(text)
            print("-" * 50)
            
            # Print usage if available
            if "usage" in response_body:
                print(f"Token usage: {response_body['usage']}")
                
            return True
        else:
            print(f"Unexpected response format: {list(response_body.keys())}")
            return False
            
    except Exception as e:
        print(f"Error testing Claude 3.7 Sonnet: {str(e)}")
        return False


async def test_fastagent_claude_3_5():
    """Test Claude 3.5 integration with FastAgent"""
    print("\n" + "="*50)
    print("Testing Claude 3.5 with FastAgent")
    print("="*50)
    
    try:
        # Initialize FastAgent with configuration file path
        fast = FastAgent("Claude 3.5 Test", config_path="examples/bedrock/fastagent.config.yaml")
        
        # Define a Claude 3.5 agent
        @fast.agent(
            "claude_3_5_agent",
            "You are a helpful assistant that provides concise responses.",
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
            # FastAgent gets parameters from config file, don't add them here
        )
        async def claude_3_5_agent():
            # This function is just a placeholder for the decorator
            pass
        
        # Create agent context manager
        async with fast.run() as agent:
            # Send a message to the agent
            print("Sending message to Claude 3.5 agent...")
            response = await agent.claude_3_5_agent.send("What is AWS Bedrock? Answer in 2-3 sentences.")
            
            print("\nResponse from Claude 3.5 via FastAgent:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            return True
            
    except Exception as e:
        print(f"Error in FastAgent Claude 3.5 test: {str(e)}")
        return False


async def test_fastagent_claude_3_7():
    """Test Claude 3.7 integration with FastAgent"""
    print("\n" + "="*50)
    print("Testing Claude 3.7 with FastAgent")
    print("="*50)
    
    try:
        # Initialize FastAgent with configuration file path
        fast = FastAgent("Claude 3.7 Test", config_path="examples/bedrock/fastagent.config.yaml")
        
        # Define a Claude 3.7 agent
        @fast.agent(
            "claude_3_7_agent",
            "You are a helpful assistant that provides concise responses.",
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            # FastAgent gets parameters from config file, don't add them here
        )
        async def claude_3_7_agent():
            # This function is just a placeholder for the decorator
            pass
        
        # Create agent context manager
        async with fast.run() as agent:
            # Send a message to the agent
            print("Sending message to Claude 3.7 agent...")
            response = await agent.claude_3_7_agent.send("What is AWS Bedrock? Answer in 2-3 sentences.")
            
            print("\nResponse from Claude 3.7 via FastAgent:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            return True
            
    except Exception as e:
        print(f"Error in FastAgent Claude 3.7 test: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("Testing Anthropic Claude models on AWS Bedrock")
    
    # Test direct Claude 3.5 API
    claude_3_5_success = await test_claude_3_5()
    if not claude_3_5_success:
        print("⚠️ Direct Claude 3.5 API test failed")
    
    # Test direct Claude 3.7 API
    claude_3_7_success = await test_claude_3_7()
    if not claude_3_7_success:
        print("⚠️ Direct Claude 3.7 API test failed")
    
    # Brief pause between tests
    await asyncio.sleep(2)
    
    # Test FastAgent integration with Claude 3.5
    fastagent_claude_3_5_success = await test_fastagent_claude_3_5()
    if not fastagent_claude_3_5_success:
        print("⚠️ FastAgent Claude 3.5 integration test failed")
    
    # Test FastAgent integration with Claude 3.7
    fastagent_claude_3_7_success = await test_fastagent_claude_3_7()
    if not fastagent_claude_3_7_success:
        print("⚠️ FastAgent Claude 3.7 integration test failed")
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"- Direct Claude 3.5 API: {'✅ Success' if claude_3_5_success else '❌ Failed'}")
    print(f"- Direct Claude 3.7 API: {'✅ Success' if claude_3_7_success else '❌ Failed'}")
    print(f"- FastAgent Claude 3.5: {'✅ Success' if fastagent_claude_3_5_success else '❌ Failed'}")
    print(f"- FastAgent Claude 3.7: {'✅ Success' if fastagent_claude_3_7_success else '❌ Failed'}")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())