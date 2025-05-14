#!/usr/bin/env python
"""
Test script for verifying AWS Bedrock credentials and connection.
This script directly uses boto3 to test connection to Bedrock,
without going through the fast-agent client.
"""

import os
import boto3
import json
import argparse
from typing import Dict, Any, Optional

def parse_args():
    parser = argparse.ArgumentParser(description="Test AWS Bedrock connection")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--model", default="anthropic.claude-3-sonnet-20240229-v1:0", 
                      help="Bedrock model ID to test")
    return parser.parse_args()

def create_client(region: str, profile: Optional[str] = None) -> Any:
    """Create a Bedrock client using the specified region and optional profile."""
    if profile:
        session = boto3.Session(profile_name=profile)
        client = session.client('bedrock-runtime', region_name=region)
    else:
        client = boto3.client('bedrock-runtime', region_name=region)
    return client

def test_bedrock_connection(client: Any, model_id: str) -> Dict[str, Any]:
    """Test connection to Bedrock by sending a simple message."""
    try:
        # Test with the Converse API (newer and more consistent)
        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Hello! Please respond with a short message confirming you're working."
                        }
                    ]
                }
            ],
            inferenceConfig={
                "temperature": 0.7,
                "maxTokens": 50
            }
        )
        return response
    except Exception as e:
        print(f"Error testing Bedrock connection: {e}")
        raise

def main():
    """Main function to run the test."""
    args = parse_args()
    print(f"Testing Bedrock connection with region: {args.region}")
    
    if args.profile:
        print(f"Using AWS profile: {args.profile}")
    else:
        print("Using default AWS credentials")
    
    print(f"Testing with model: {args.model}")
    
    # Create client
    client = create_client(args.region, args.profile)
    
    # Test connection
    try:
        response = test_bedrock_connection(client, args.model)
        print("\nConnection successful! 🎉")
        print("\nResponse from Bedrock:")
        
        # Extract the model's response text
        if "output" in response and "content" in response["output"]:
            content = response["output"]["content"]
            for item in content:
                if item.get("type") == "text":
                    print(f"\n{item.get('text')}")
        else:
            print("\nUnexpected response format:")
            print(json.dumps(response, indent=2))
        
        print("\nToken usage:")
        if "usage" in response:
            print(f"  Input tokens: {response['usage'].get('inputTokens', 'n/a')}")
            print(f"  Output tokens: {response['usage'].get('outputTokens', 'n/a')}")
            
        print("\nVerification complete! AWS credentials are working correctly.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        print("\nPlease check your AWS credentials and permissions.")
        print("Required permissions: bedrock-runtime:Converse")

if __name__ == "__main__":
    main()