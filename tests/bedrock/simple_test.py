#!/usr/bin/env python
"""
Simple test script for Bedrock API using boto3 directly.
"""

import json
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Bedrock API test")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--model", default="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                     help="Bedrock model ID to test")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create boto3 session with the provided profile if any
    if args.profile:
        session = boto3.Session(profile_name=args.profile)
        print(f"Using AWS profile: {args.profile}")
    else:
        session = boto3.Session()
        print("Using default AWS credentials")
    
    # Create Bedrock client
    bedrock_client = session.client('bedrock-runtime', region_name=args.region)
    print(f"Created Bedrock client for region {args.region}")
    
    # Determine model family
    is_claude = "claude" in args.model.lower()
    is_nova = "nova" in args.model.lower()
    is_meta = "llama" in args.model.lower() or "meta" in args.model.lower()
    
    query = "Please identify yourself and confirm you are working."
    
    # Prepare request body based on model type
    if is_claude:
        print("Using Claude format")
        # Claude format
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ]
        }
    elif is_nova:
        print("Using Nova format with Converse API")
        try:
            # First try with Converse API
            response = bedrock_client.converse(
                modelId=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": query
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            print("\nResponse from Bedrock (Converse API):")
            if "output" in response and "content" in response["output"]:
                for content_item in response["output"]["content"]:
                    if "text" in content_item:
                        print(content_item["text"])
            else:
                print("Response format not recognized:")
                print(json.dumps(response, indent=2))
            
            print("\n✅ SUCCESS! Bedrock API test completed successfully.")
            return
        except Exception as e:
            print(f"Converse API failed for Nova model: {e}")
            print("Trying with InvokeModel API...")
            
            # Try with a simple format for InvokeModel
            request_body = {
                "prompt": query,
                "max_tokens": 100
            }
    elif is_meta:
        print("Using Meta Llama format")
        # Meta Llama format
        request_body = {
            "prompt": query,
            "max_gen_len": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }
    else:
        print("Using generic format")
        # Generic format for other models
        request_body = {
            "inputText": query,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    
    # Send request to Bedrock
    print(f"Sending request to {args.model}...")
    try:
        response = bedrock_client.invoke_model(
            modelId=args.model,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract response text based on model type
        print("\nResponse from Bedrock:")
        if is_claude:
            # Claude response format
            if "content" in response_body:
                for content_item in response_body["content"]:
                    if "text" in content_item:
                        print(content_item["text"])
            elif "output" in response_body and "content" in response_body["output"]:
                for content_item in response_body["output"]["content"]:
                    if "text" in content_item:
                        print(content_item["text"])
            else:
                print("Response format not recognized:")
                print(json.dumps(response_body, indent=2))
        else:
            # Generic format (like Nova)
            if "outputText" in response_body:
                print(response_body["outputText"])
            elif "results" in response_body and len(response_body["results"]) > 0:
                print(response_body["results"][0].get("outputText", "No output text found"))
            else:
                print("Response format not recognized:")
                print(json.dumps(response_body, indent=2))
        
        print("\n✅ SUCCESS! Bedrock API test completed successfully.")
        
    except NoCredentialsError:
        print("❌ ERROR: No AWS credentials found.")
        print("Please configure credentials using one of the following methods:")
        print("  - AWS CLI: aws configure")
        print("  - Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("  - IAM role if running on EC2/ECS")
    except ClientError as e:
        print(f"❌ ERROR: {e}")
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "Unknown")
        error_message = getattr(e, "response", {}).get("Error", {}).get("Message", "Unknown error")
        
        if error_code == "AccessDeniedException":
            print("Your AWS credentials don't have permission to use Bedrock.")
            print("Required permissions: bedrock:ListFoundationModels, bedrock-runtime:InvokeModel")
        elif error_code == "ValidationException" and "ID" in error_message:
            print("The model ID format is incorrect. For Claude models in US regions, use 'us.' prefix.")
            if not args.model.startswith("us.") and not args.model.startswith("eu."):
                print(f"Try: us.{args.model}")
        elif error_code == "ModelNotReadyException" or error_code == "ModelNotFoundException":
            print(f"Model {args.model} not found or not available in region {args.region}.")
            print("Check the model ID and ensure it's available in your region.")
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()