#!/usr/bin/env python
"""
Test script for Amazon Nova Pro on AWS Bedrock
This script tests both direct API access and FastAgent integration
"""

import asyncio
import json
import os
import sys
import boto3
from typing import Dict, Any, List, Optional

# Import FastAgent components
from mcp_agent import FastAgent
from mcp_agent.config import get_settings


async def test_direct_nova_pro_api():
    """Test direct access to Nova Pro using boto3"""
    print("Testing direct AWS Bedrock Nova Pro access...")
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        
        # Nova Pro model ID
        model_id = "us.amazon.nova-pro-v1:0"
        
        # Let's try a pure messages format
        print("\nTrying pure messages format for Nova Pro...")
        
        # Based on error message requiring messages key with content as array
        request_body = json.dumps({
            "messages": [
                {"role": "user", "content": [{"text": "What is AWS Bedrock?"}]}
            ]
        })
        
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=request_body
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            print(f"Response format: {list(response_body.keys())}")
            
            if "output" in response_body:
                print("Success! Nova Pro with messages format works")
                print("\nResponse from Nova Pro:")
                print("-" * 50)
                print(response_body["output"])
                print("-" * 50)
                
                # Print additional details
                if "usage" in response_body:
                    print(f"Token usage: {response_body['usage']}")
                if "stopReason" in response_body:
                    print(f"Stop reason: {response_body['stopReason']}")
                
                return True
            else:
                print(f"Unexpected response format: {list(response_body.keys())}")
                
        except Exception as e:
            print(f"First approach failed: {e}")
        
        # Try an alternative messages format
        print("\nTrying alternative messages format for Nova Pro...")
        
        # Try with type explicitly specified
        request_body = json.dumps({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is AWS Bedrock?"}]}
            ]
        })
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        print(f"Response keys: {list(response_body.keys())}")
        
        if "results" in response_body and len(response_body["results"]) > 0:
            result = response_body["results"][0]
            if "outputText" in result:
                print("Success! Simple Nova Pro format works")
                print("\nResponse from Nova Pro (simple format):")
                print("-" * 50)
                print(result["outputText"])
                print("-" * 50)
                return True
        
        print("Failed to get expected response from either API approach")
        print(f"Response keys: {response_body.keys()}")
        return False
        
    except Exception as e:
        print(f"Error accessing AWS Bedrock: {e}")
        return False


async def test_fastagent_nova_pro():
    """Test Nova Pro integration with FastAgent"""
    print("\nTesting FastAgent integration with Nova Pro...")
    
    try:
        # Initialize FastAgent with configuration file path
        fast = FastAgent("Nova Pro Test", config_path="examples/bedrock/fastagent.config.yaml")
        
        # Define a Nova Pro agent
        @fast.agent(
            "nova_pro_agent",
            "You are a helpful assistant that provides concise responses.",
            model="us.amazon.nova-pro-v1:0"  # Specify Nova Pro model explicitly
        )
        async def nova_test_agent():
            # This function is just a placeholder for the decorator
            pass
        
        # Create agent context manager
        async with fast.run() as agent:
            # Send a message to the agent
            print("Sending message to Nova Pro agent...")
            response = await agent.nova_pro_agent.send("What is AWS Bedrock in 2-3 sentences?")
            
            print("\nResponse from Nova Pro via FastAgent:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Send a follow-up message
            print("\nSending follow-up message...")
            response = await agent.nova_pro_agent.send("What models do you support?")
            
            print("\nFollow-up response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            return True
            
    except Exception as e:
        print(f"Error in FastAgent Nova Pro test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run both test functions"""
    # Test direct API access
    api_success = await test_direct_nova_pro_api()
    
    if api_success:
        print("\nDirect API test successful!")
    else:
        print("\nDirect API test failed!")
        print("You may need to request access to Nova Pro in your AWS account")
        print("or check your AWS credentials and permissions.")
        return
    
    # Test FastAgent integration
    fastagent_success = await test_fastagent_nova_pro()
    
    if fastagent_success:
        print("\nFastAgent integration test successful!")
    else:
        print("\nFastAgent integration test failed!")
        print("The fixes for Nova Pro model support may need further adjustments.")


if __name__ == "__main__":
    asyncio.run(main())