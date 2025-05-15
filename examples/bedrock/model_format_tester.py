#!/usr/bin/env python
"""
AWS Bedrock Model Format Tester

This script helps test different AWS Bedrock model formats to determine which
format works with specific models in your AWS region. Each model family has 
different format requirements, and some formats vary by region.

REQUIREMENTS TO RUN THIS TOOL:
1. AWS credentials with access to AWS Bedrock
2. Access to the models you want to test (Claude, Nova, Meta Llama)
3. boto3 library installed: pip install boto3

The script tests:
1. Claude models with the Converse API
2. Nova models with the Converse API (minimal format required)
3. Meta Llama models with:
   - Traditional invoke_model() with prompt format
   - Direct converse() API method with messages format

USAGE:
python model_format_tester.py --region us-east-1 [--verbose] [--claude] [--nova] [--llama-prompt] [--llama-messages]

Arguments:
  --region           AWS region to test (default: us-east-1)
  --verbose          Enable verbose output
  --claude           Test only Claude format
  --nova             Test only Nova format
  --llama-prompt     Test only Llama prompt format
  --llama-messages   Test only Llama converse() API

Example for testing only Claude in Europe region:
python model_format_tester.py --region eu-central-1 --claude --verbose

IMPORTANT:
- The converse() API method is different from invoke_model() and might be 
  available for some models/regions where invoke_model() doesn't work with messages
- This test will try both with and without region prefixes for Llama models
"""

import asyncio
import json
import boto3
from typing import Dict, Any, List, Optional
import argparse


async def test_claude_format(region: str, verbose: bool = False):
    """Test Claude model format with Converse API"""
    print("\n" + "="*50)
    print("Testing Claude 3.5 Sonnet Model Format")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
        
        # Claude 3.5 Sonnet model ID
        model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        
        # Claude format with anthropic_version parameter
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
        
        if verbose:
            print(f"Request body: {request_body}")
        
        print(f"Sending request to {model_id}...")
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        if verbose:
            print(f"Response keys: {list(response_body.keys())}")
        
        if "content" in response_body:
            print("\n✅ SUCCESS: Claude format with anthropic_version parameter works!")
            
            text_content = []
            for content_item in response_body.get("content", []):
                if content_item.get("type") == "text":
                    text_content.append(content_item.get("text", ""))
            
            print("\nResponse:")
            print("-" * 50)
            print("\n".join(text_content))
            print("-" * 50)
            
            return True
        else:
            print(f"\n❌ FAILURE: Unexpected Claude response format: {list(response_body.keys())}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR testing Claude format: {str(e)}")
        return False


async def test_nova_format(region: str, verbose: bool = False):
    """Test Nova model format with Converse API (minimal format)"""
    print("\n" + "="*50)
    print("Testing Nova Pro Model Format (Minimal Format)")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
        
        # Nova Pro model ID
        model_id = "us.amazon.nova-pro-v1:0"
        
        # Nova Pro minimal format - no additional parameters!
        request_body = json.dumps({
            "messages": [
                {"role": "user", "content": [{"text": "What is AWS Bedrock? Respond in 2-3 sentences."}]}
            ]
        })
        
        if verbose:
            print(f"Request body: {request_body}")
        
        print(f"Sending request to {model_id}...")
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        if verbose:
            print(f"Response keys: {list(response_body.keys())}")
        
        if "output" in response_body:
            print("\n✅ SUCCESS: Nova Pro minimal format works!")
            
            print("\nResponse:")
            print("-" * 50)
            print(response_body["output"])
            print("-" * 50)
            
            return True
        else:
            print(f"\n❌ FAILURE: Unexpected Nova Pro response format: {list(response_body.keys())}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR testing Nova Pro format: {str(e)}")
        return False


async def test_llama_prompt_format(region: str, verbose: bool = False):
    """Test Meta Llama model with traditional prompt format"""
    print("\n" + "="*50)
    print("Testing Meta Llama 3 70B Model with Prompt Format")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
        
        # Llama 3 70B model ID
        model_id = "us.meta.llama3-1-70b-instruct-v1:0"
        
        # Traditional prompt format
        system_message = "You are a helpful AI assistant that provides accurate and concise information."
        user_message = "What is AWS Bedrock? Respond in 2-3 sentences."
        
        # Format prompt for Llama
        prompt = f"<s>[INST] {system_message} [/INST]\n\n{user_message}</s>"
        
        # Use minimal parameters to avoid 'RequestParams' object has no attribute 'top_p' errors
        request_body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 1024,
            "temperature": 0.7
            # Removed top_p as it causes errors in some configurations
        })
        
        if verbose:
            print(f"Request body: {request_body}")
        
        print(f"Sending request to {model_id}...")
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=request_body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        if verbose:
            print(f"Response keys: {list(response_body.keys())}")
        
        if "generation" in response_body:
            print("\n✅ SUCCESS: Llama traditional prompt format works!")
            
            print("\nResponse:")
            print("-" * 50)
            print(response_body["generation"])
            print("-" * 50)
            
            return True
        else:
            print(f"\n❌ FAILURE: Unexpected Llama response format: {list(response_body.keys())}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR testing Llama prompt format: {str(e)}")
        return False


async def test_llama_messages_format(region: str, verbose: bool = False):
    """Test Meta Llama model with Converse API using the direct converse() method"""
    print("\n" + "="*50)
    print("Testing Meta Llama 3 Model with Converse API")
    print("="*50)
    
    try:
        # Create a bedrock runtime client
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
        
        # Use a smaller Llama model that's more likely to be available
        model_id = "meta.llama3-8b-instruct-v1:0"
        # Note: meta.llama3-8b-instruct-v1:0 is used in docs without region prefix
        # If it fails, we'll try with region prefix
        
        # Prepare messages format for the converse() method
        messages = [
            {
                "role": "user",
                "content": [{"text": "What is AWS Bedrock? Respond in 2-3 sentences."}]
            }
        ]
        
        # Prepare inference config with minimal parameters
        inference_config = {
            "maxTokens": 512,
            "temperature": 0.5,
            "topP": 0.9
        }
        
        if verbose:
            print(f"Using model: {model_id}")
            print(f"Messages: {messages}")
            print(f"Inference config: {inference_config}")
        
        print(f"Sending converse() request to {model_id}...")
        
        # Try with the converse() method which is different from invoke_model()
        try:
            # First try without region prefix as shown in AWS documentation
            response = bedrock_client.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig=inference_config
            )
            
            if verbose:
                print(f"Response keys: {list(response.keys())}")
            
            if "output" in response and "message" in response["output"]:
                print("\n✅ SUCCESS: Llama with converse() API works!")
                
                # Extract response text
                response_text = response["output"]["message"]["content"][0]["text"]
                
                print("\nResponse:")
                print("-" * 50)
                print(response_text)
                print("-" * 50)
                
                return True
            else:
                print(f"\n❌ FAILURE: Unexpected Llama converse() response format: {list(response.keys())}")
                return False
                
        except Exception as first_error:
            # If fails without region prefix, try with region prefix
            if "ResourceNotFoundException" in str(first_error):
                print(f"Model {model_id} not found, trying with region prefix...")
                model_id = f"us.{model_id}"
                
                try:
                    response = bedrock_client.converse(
                        modelId=model_id,
                        messages=messages,
                        inferenceConfig=inference_config
                    )
                    
                    if "output" in response and "message" in response["output"]:
                        print("\n✅ SUCCESS: Llama with converse() API works!")
                        
                        # Extract response text
                        response_text = response["output"]["message"]["content"][0]["text"]
                        
                        print("\nResponse:")
                        print("-" * 50)
                        print(response_text)
                        print("-" * 50)
                        
                        return True
                    else:
                        print(f"\n❌ FAILURE: Unexpected Llama converse() response format: {list(response.keys())}")
                        return False
                        
                except Exception as second_error:
                    print(f"\n❌ ERROR with region prefixed model: {str(second_error)}")
                    if "ValidationException" in str(second_error):
                        print(f"This may be due to converse() API not available for this model in this region")
                        print(f"Try using a different Llama model or a different region")
                    return False
            
            elif "ValidationException" in str(first_error):
                print(f"\n❌ EXPECTED ERROR: Converse API may not be available for this model in this region")
                print(f"Error details: {str(first_error)}")
                print("\nRECOMMENDATION: Use the traditional invoke_model with prompt format for Llama models in this region")
                return False
            else:
                # For other errors, show details
                print(f"\n❌ ERROR with converse() API: {str(first_error)}")
                print("This may indicate that the converse() API is not available for this model in this region")
                return False
                
    except Exception as e:
        print(f"\n❌ ERROR testing Llama messages format: {str(e)}")
        if "Could not connect to the endpoint URL" in str(e):
            print("Check your AWS credentials and region settings")
        return False


async def main():
    """Run all format tests"""
    parser = argparse.ArgumentParser(description='Test AWS Bedrock model formats')
    parser.add_argument('--region', type=str, default='us-east-1', 
                        help='AWS region to test (default: us-east-1)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--claude', action='store_true',
                        help='Test only Claude format')
    parser.add_argument('--nova', action='store_true',
                        help='Test only Nova format')
    parser.add_argument('--llama-prompt', action='store_true',
                        help='Test only Llama prompt format')
    parser.add_argument('--llama-messages', action='store_true',
                        help='Test only Llama messages format')
    
    args = parser.parse_args()
    
    # If no specific tests are requested, run all tests
    run_all = not (args.claude or args.nova or args.llama_prompt or args.llama_messages)
    
    print(f"Testing AWS Bedrock model formats in region: {args.region}")
    
    results = {}
    
    # Test Claude format
    if run_all or args.claude:
        results["claude"] = await test_claude_format(args.region, args.verbose)
    
    # Test Nova format
    if run_all or args.nova:
        results["nova"] = await test_nova_format(args.region, args.verbose)
    
    # Test Llama prompt format
    if run_all or args.llama_prompt:
        results["llama_prompt"] = await test_llama_prompt_format(args.region, args.verbose)
    
    # Test Llama messages format
    if run_all or args.llama_messages:
        results["llama_messages"] = await test_llama_messages_format(args.region, args.verbose)
    
    # Print summary
    print("\n" + "="*50)
    print("Format Testing Results Summary")
    print("="*50)
    
    if "claude" in results:
        status = "✅ WORKS" if results["claude"] else "❌ FAILS"
        print(f"Claude format with anthropic_version: {status}")
    
    if "nova" in results:
        status = "✅ WORKS" if results["nova"] else "❌ FAILS"
        print(f"Nova minimal format without parameters: {status}")
    
    if "llama_prompt" in results:
        status = "✅ WORKS" if results["llama_prompt"] else "❌ FAILS"
        print(f"Llama traditional prompt format: {status}")
    
    if "llama_messages" in results:
        status = "✅ WORKS" if results["llama_messages"] else "❌ FAILS"
        print(f"Llama direct converse() API method: {status}")
    
    print("\nNotes on formats that work in your region:")
    print("- Update your fastagent.config.yaml accordingly")
    print("- For Nova models, ALWAYS use empty parameter blocks: {}") 
    
    # Add specific recommendations for Llama models
    if "llama_prompt" in results and "llama_messages" in results:
        if results["llama_prompt"] and not results["llama_messages"]:
            print("- For Meta Llama models in FastAgent:")
            print("  * Use invoke_model() with prompt format (converse() API not working)")
            print("  * In fastagent.config.yaml use empty parameter blocks: {}")
            print("  * FastAgent will correctly format the prompt for you")
        elif not results["llama_prompt"] and results["llama_messages"]: 
            print("- For Meta Llama models in FastAgent:")
            print("  * Use converse() API instead of invoke_model() in this region")
            print("  * Note that FastAgent may need updates to use converse() API")
            print("  * In fastagent.config.yaml use empty parameter blocks: {}")
        elif results["llama_prompt"] and results["llama_messages"]:
            print("- For Meta Llama models in FastAgent:")
            print("  * Both invoke_model() and converse() API work in this region")
            print("  * Recommend using empty parameter blocks in fastagent.config.yaml")
            print("  * FastAgent currently uses invoke_model() by default")
        else:
            print("- Meta Llama models may have special requirements in this region")
            print("  * Try using the direct boto3 converse() API method")
            print("  * FastAgent may need updates to support this configuration")
    elif "llama_prompt" in results:
        if results["llama_prompt"]:
            print("- For Meta Llama models in FastAgent:")
            print("  * Use invoke_model() with prompt format (untested converse() API)")
            print("  * In fastagent.config.yaml use empty parameter blocks: {}")
    elif "llama_messages" in results:
        if results["llama_messages"]:
            print("- For Meta Llama models in FastAgent:")
            print("  * converse() API works but invoke_model() was not tested")
            print("  * Note that FastAgent may need updates to use converse() API")
            print("  * In fastagent.config.yaml use empty parameter blocks: {}")
    
    print("\nTest command used:")
    cmd = f"python model_format_tester.py --region {args.region}"
    if args.verbose:
        cmd += " --verbose"
    if args.claude:
        cmd += " --claude"
    if args.nova:
        cmd += " --nova"
    if args.llama_prompt:
        cmd += " --llama-prompt"
    if args.llama_messages:
        cmd += " --llama-messages"
    print(f"  {cmd}")


if __name__ == "__main__":
    asyncio.run(main())