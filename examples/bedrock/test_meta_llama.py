#!/usr/bin/env python
"""
Test script for Meta Llama models on AWS Bedrock
This script tests the Llama 3 70B model

REQUIREMENTS TO RUN THIS TEST:
1. AWS credentials with access to AWS Bedrock
2. Access to Meta Llama 3 70B model in your AWS account
3. boto3 library installed: pip install boto3
4. FastAgent installed: pip install fast-agent-mcp[bedrock]

IMPORTANT FORMAT NOTES:
- Meta Llama format requirements vary by AWS region
- Some regions use the traditional prompt format:
  {"prompt": "<s>[INST]...[/INST]</s>", ...}
- Others use the Converse API messages format:
  {"messages": [{"role": "system", ...}, ...]}
- This script tries both formats

If you encounter "ValidationException" errors:
- Try running model_format_tester.py to detect the format that works in your region
- Update this script to use the format that works in your region
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
logger = get_logger("meta_llama_test")


async def test_llama_3():
    """Test Meta Llama 3 70B model"""
    print("\n" + "="*50)
    print("Testing Meta Llama 3 70B")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        
        # Llama 3 70B model ID
        model_id = "us.meta.llama3-1-70b-instruct-v1:0"
        
        # According to error message, Meta Llama still requires 'prompt' format in our region
        # The documentation indicates it may be different in certain regions
        system_message = "You are a helpful AI assistant that provides accurate and concise information."
        user_message = "What is AWS Bedrock? Respond in 2-3 sentences."
        
        # Format prompt for Llama (following their specific format)
        prompt = f"<s>[INST] {system_message} [/INST]\n\n{user_message}</s>"
        
        request_body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 1024,
            "temperature": 0.7,
            "top_p": 0.9
        })
        
        print(f"Sending request to {model_id}...")
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        # Check for Converse API format 
        if "output" in response_body:
            print("\nResponse from Llama 3 70B:")
            print("-" * 50)
            print(response_body["output"])
            print("-" * 50)
            
            # Print token usage if available
            if "usage" in response_body:
                print(f"Token usage: {response_body['usage']}")
                
            return True
        # Check for old format (just in case)
        elif "generation" in response_body:
            print("\nResponse from Llama 3 70B (old format):")
            print("-" * 50)
            print(response_body["generation"])
            print("-" * 50)
            
            return True
        else:
            print(f"Unexpected response format: {list(response_body.keys())}")
            return False
            
    except Exception as e:
        print(f"Error testing Llama 3 70B: {str(e)}")
        return False


async def test_fastagent_llama_3():
    """Test Llama 3 integration with FastAgent"""
    print("\n" + "="*50)
    print("Testing Llama 3 with FastAgent")
    print("="*50)
    
    try:
        # Initialize FastAgent with configuration file path
        fast = FastAgent("Llama 3 Test", config_path="examples/bedrock/fastagent.config.yaml")
        
        # Create a custom instruction system for Llama with reduced parameters
        # The key issue is to avoid parameters like top_p that cause errors
        instruction = """You are a helpful AI assistant that provides accurate and concise information.
        When asked about AWS Bedrock, explain it in 2-3 clear sentences."""
        
        # Define a Llama 3 agent with ONLY model ID - absolutely no other parameters
        @fast.agent(
            "llama_3_agent",
            instruction,
            model="us.meta.llama3-1-70b-instruct-v1:0"
        )
        async def llama_3_agent():
            # This function is just a placeholder for the decorator
            pass
        
        # Create agent context manager
        async with fast.run() as agent:
            # Send a message to the agent
            print("Sending message to Llama 3 agent...")
            
            try:
                # Use FastAgent's standard approach with minimal parameters
                response = await agent.llama_3_agent.send("What is AWS Bedrock?")
                
                print("\nResponse from Llama 3 via FastAgent:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                
                # If we get here, the FastAgent integration worked!
                print("✅ SUCCESS: FastAgent integration with Llama 3 works!")
                return True
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ FAILURE: FastAgent integration failed: {error_msg}")
                print("Meta Llama models require minimal configuration with no additional parameters.")
                print("Check fastagent.config.yaml to ensure problematic parameters are removed.")
                return False
            
    except Exception as e:
        print(f"Error in FastAgent Llama 3 test: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("Testing Meta Llama models on AWS Bedrock")
    
    # Track overall test success
    all_tests_passed = True
    
    # Test direct Llama 3 API
    llama_3_success = await test_llama_3()
    if not llama_3_success:
        print("⚠️ Direct Llama 3 API test failed")
        all_tests_passed = False
    
    # Brief pause between tests
    await asyncio.sleep(2)
    
    # Test FastAgent integration with Llama 3
    fastagent_llama_3_success = await test_fastagent_llama_3()
    if not fastagent_llama_3_success:
        print("⚠️ FastAgent Llama 3 integration test failed")
        all_tests_passed = False
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"- Direct Llama 3 API: {'✅ Success' if llama_3_success else '❌ Failed'}")
    print(f"- FastAgent Llama 3: {'✅ Success' if fastagent_llama_3_success else '❌ Failed'}")
    print(f"- Overall Test Status: {'✅ PASSED' if all_tests_passed else '❌ FAILED'}")
    print("="*50)
    
    # If FastAgent test failed but direct API worked, show guidance
    if llama_3_success and not fastagent_llama_3_success:
        print("\nFASTAGENT INTEGRATION GUIDANCE:")
        print("The direct API works but FastAgent integration fails.")
        print("This is likely due to the 'RequestParams' object not supporting 'top_p'.")
        print("To fix this permanently:")
        print("1. Update augmented_llm_bedrock.py to check for key existence before using")
        print("2. Remove top_p and top_k from the Meta Llama section in fastagent.config.yaml")
        print("3. Add error handling in _prepare_meta_request to avoid using unsupported parameters")
        print("\nRun model_format_tester.py to determine the correct format for your region.")
    
    # Exit with success/failure code
    return all_tests_passed


if __name__ == "__main__":
    asyncio.run(main())