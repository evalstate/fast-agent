#!/usr/bin/env python
"""
Test script for Amazon Nova models on AWS Bedrock
This script tests both Nova Pro and other Nova models (Nova-Lite, Nova-Micro, etc.)

REQUIREMENTS TO RUN THIS TEST:
1. AWS credentials with access to AWS Bedrock
2. Access to Nova Pro and Nova-Lite models in your AWS account
3. boto3 library installed: pip install boto3
4. FastAgent installed: pip install fast-agent-mcp[bedrock]

The script tests:
- Direct API access to Nova Pro and Nova-Lite using boto3
- FastAgent integration with Nova Pro and Nova-Lite
- Error handling with validation errors

If you encounter "ValidationException" errors:
- For Nova models, ensure you're using the minimal message format
- Remove all additional parameters (temperature, top_p, etc.)
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
logger = get_logger("amazon_nova_test")


async def test_nova_pro():
    """Test Amazon Nova Pro model"""
    print("\n" + "="*50)
    print("Testing Amazon Nova Pro Model")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        
        # Nova Pro model ID
        model_id = "us.amazon.nova-pro-v1:0"
        
        # Nova Pro requires minimal format with no additional parameters
        request_body = json.dumps({
            "messages": [
                {"role": "user", "content": [{"text": "What is AWS Bedrock? Respond in 2-3 sentences."}]}
            ]
        })
        
        print(f"Sending request to {model_id}...")
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        if "output" in response_body:
            print("\nResponse from Nova Pro:")
            print("-" * 50)
            print(response_body["output"])
            print("-" * 50)
            
            # Print token usage
            if "usage" in response_body:
                print(f"Token usage: {response_body['usage']}")
                
            return True
        else:
            print(f"Unexpected response format: {list(response_body.keys())}")
            return False
            
    except Exception as e:
        print(f"Error testing Nova Pro: {str(e)}")
        return False


async def test_nova_lite():
    """Test Amazon Nova-Lite model"""
    print("\n" + "="*50)
    print("Testing Amazon Nova-Lite Model")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        
        # Nova-Lite model ID
        model_id = "us.amazon.nova-lite-v1:0"
        
        # Based on the AWS documentation for Nova Converse API
        # https://docs.aws.amazon.com/nova/latest/userguide/using-converse-api.html
        # Nova-Lite now uses the messages format like Nova Pro
        request_body = json.dumps({
            "messages": [
                {"role": "user", "content": [{"text": "What is AWS Bedrock? Respond in 2-3 sentences."}]}
            ]
        })
        
        print(f"Sending request to {model_id}...")
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        # Check for Converse API format (similar to Nova Pro)
        if "output" in response_body:
            print("\nResponse from Nova-Lite:")
            print("-" * 50)
            print(response_body["output"])
            print("-" * 50)
            
            # Print token usage if available
            if "usage" in response_body:
                print(f"Token usage: {response_body['usage']}")
                
            return True
        # Check for old format (just in case)
        elif "results" in response_body and len(response_body["results"]) > 0:
            result = response_body["results"][0]
            if "outputText" in result:
                print("\nResponse from Nova-Lite (old format):")
                print("-" * 50)
                print(result["outputText"])
                print("-" * 50)
                
                return True
        
        print(f"Unexpected response format: {list(response_body.keys())}")
        return False
            
    except Exception as e:
        print(f"Error testing Nova-Lite: {str(e)}")
        return False


async def test_fastagent_nova_pro():
    """Test Nova Pro integration with FastAgent"""
    print("\n" + "="*50)
    print("Testing Nova Pro with FastAgent")
    print("="*50)
    
    try:
        # Initialize FastAgent with configuration file path
        fast = FastAgent("Nova Pro Test", config_path="examples/bedrock/fastagent.config.yaml")
        
        # Define a Nova Pro agent
        @fast.agent(
            "nova_pro_agent",
            "You are a helpful assistant that provides concise responses.",
            model="us.amazon.nova-pro-v1:0"  # Specify Nova Pro model explicitly
            # Don't add any parameters - Nova Pro API doesn't accept them
        )
        async def nova_pro_agent():
            # This function is just a placeholder for the decorator
            pass
        
        # Create agent context manager
        async with fast.run() as agent:
            # Send a message to the agent
            print("Sending message to Nova Pro agent...")
            response = await agent.nova_pro_agent.send("What is AWS Bedrock? Answer in 2-3 sentences.")
            
            print("\nResponse from Nova Pro via FastAgent:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            return True
            
    except Exception as e:
        print(f"Error in FastAgent Nova Pro test: {str(e)}")
        return False


async def test_fastagent_nova_lite():
    """Test Nova-Lite integration with FastAgent"""
    print("\n" + "="*50)
    print("Testing Nova-Lite with FastAgent")
    print("="*50)
    
    try:
        # Initialize FastAgent with configuration file path
        fast = FastAgent("Nova-Lite Test", config_path="examples/bedrock/fastagent.config.yaml")
        
        # Create a custom instruction for Nova
        instruction = """You are a helpful assistant that provides concise responses.
        When asked about AWS Bedrock, explain it in 2-3 clear, informative sentences."""
        
        # Define a Nova-Lite agent with ONLY model ID - absolutely no other parameters
        @fast.agent(
            "nova_lite_agent",
            instruction,
            model="us.amazon.nova-lite-v1:0"
        )
        async def nova_lite_agent():
            # This function is just a placeholder for the decorator
            pass
        
        # Create agent context manager
        async with fast.run() as agent:
            # Send a message to the agent
            print("Sending message to Nova-Lite agent...")
            try:
                # Using FastAgent's standard approach with minimal parameters
                response = await agent.nova_lite_agent.send("What is AWS Bedrock?")
                
                print("\nResponse from Nova-Lite via FastAgent:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                
                print("✅ SUCCESS: FastAgent integration with Nova-Lite works!")
                return True
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ FAILURE: FastAgent integration failed: {error_msg}")
                print("Nova-Lite models require a minimal configuration with no additional parameters.")
                print("Check fastagent.config.yaml to ensure there are NO parameters for this model.")
                return False
            
    except Exception as e:
        print(f"Error in FastAgent Nova-Lite test: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("Testing Amazon Nova models on AWS Bedrock")
    
    # Track overall test success
    all_tests_passed = True
    
    # Test direct Nova Pro API
    nova_pro_success = await test_nova_pro()
    if not nova_pro_success:
        print("⚠️ Direct Nova Pro API test failed")
        all_tests_passed = False
    
    # Test direct Nova-Lite API
    nova_lite_success = await test_nova_lite()
    if not nova_lite_success:
        print("⚠️ Direct Nova-Lite API test failed")
        all_tests_passed = False
    
    # Brief pause between tests
    await asyncio.sleep(2)
    
    # Test FastAgent integration with Nova Pro
    fastagent_nova_pro_success = await test_fastagent_nova_pro()
    if not fastagent_nova_pro_success:
        print("⚠️ FastAgent Nova Pro integration test failed")
        all_tests_passed = False
    
    # Test FastAgent integration with Nova-Lite
    fastagent_nova_lite_success = await test_fastagent_nova_lite()
    if not fastagent_nova_lite_success:
        print("⚠️ FastAgent Nova-Lite integration test failed")
        all_tests_passed = False
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"- Direct Nova Pro API: {'✅ Success' if nova_pro_success else '❌ Failed'}")
    print(f"- Direct Nova-Lite API: {'✅ Success' if nova_lite_success else '❌ Failed'}")
    print(f"- FastAgent Nova Pro: {'✅ Success' if fastagent_nova_pro_success else '❌ Failed'}")
    print(f"- FastAgent Nova-Lite: {'✅ Success' if fastagent_nova_lite_success else '❌ Failed'}")
    print(f"- Overall Test Status: {'✅ PASSED' if all_tests_passed else '❌ FAILED'}")
    print("="*50)
    
    # If direct API works but FastAgent integration doesn't, show guidance
    if (nova_pro_success and not fastagent_nova_pro_success) or (nova_lite_success and not fastagent_nova_lite_success):
        print("\nFASTAGENT INTEGRATION GUIDANCE:")
        print("Direct API works but FastAgent integration fails.")
        print("This is likely due to format issues with Nova models.")
        print("To fix this permanently:")
        print("1. Make sure there are NO parameters in fastagent.config.yaml for Nova models")
        print("2. Update augmented_llm_bedrock.py to use minimal format for ALL Nova models")
        print("3. Nova models require a specific format with NO additional parameters:")
        print("   {\"messages\": [{\"role\": \"user\", \"content\": [{\"text\": \"...\"}]}]}")
        print("\nRun model_format_tester.py to verify the format for your region.")
    
    # Exit with success/failure code
    return all_tests_passed


if __name__ == "__main__":
    asyncio.run(main())