#!/usr/bin/env python
"""
Simple test script for AWS Bedrock connection
"""

import asyncio
import boto3
import json

# Function to test Bedrock with Claude model
async def test_bedrock_claude():
    print("Testing AWS Bedrock with Claude model...")
    
    # Use default credentials
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )
    
    # Test with Claude 3.5 Sonnet model
    model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # Prepare the request
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.7,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Can you tell me about AWS Bedrock?"
            }
        ]
    }
    
    try:
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Parse and print response
        response_body = json.loads(response['body'].read())
        
        print("\nSuccess! Response from Claude:")
        print("-" * 50)
        print(response_body['content'][0]['text'])
        print("-" * 50)
        
        return True
    except Exception as e:
        print(f"\nError: {e}")
        
        # Check if it's a credentials issue
        if "AccessDenied" in str(e):
            print("\nThis appears to be an authentication issue.")
            print("Make sure your AWS credentials are properly configured and have access to Bedrock.")
            print("You can set up credentials using:")
            print("  - AWS CLI: 'aws configure'")
            print("  - Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            print("  - ~/.aws/credentials file")
        
        # Check if it's a model access issue
        elif "User does not have access to modelId" in str(e) or "doesn't have access to model" in str(e):
            print("\nYou don't have access to this model in your AWS account.")
            print("Go to the AWS Bedrock console, navigate to 'Model access' and request access to Claude models.")
        
        # Check if it's a region issue
        elif "Could not connect to the endpoint URL" in str(e):
            print("\nCould not connect to Bedrock endpoint. This could be a region issue.")
            print("Make sure Bedrock is available in the specified region (us-east-1).")
            print("Try changing to another region like us-west-2 if needed.")
        
        # Check if it's a model ID format issue
        elif "Validation error" in str(e) and "model ID" in str(e):
            print("\nInvalid model ID format. Make sure to include the region prefix.")
            print("For example: 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'")
        
        return False

if __name__ == "__main__":
    asyncio.run(test_bedrock_claude())