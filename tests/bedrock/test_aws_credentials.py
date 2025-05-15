#!/usr/bin/env python
"""
Test script for validating AWS credentials for Bedrock access.
This script verifies that the provided AWS credentials can access Bedrock services.
"""

import os
import sys
import boto3
import json
import argparse
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError, NoCredentialsError


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate AWS credentials for Bedrock")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--list-models", action="store_true", help="List available Bedrock models")
    parser.add_argument("--model", default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                      help="Bedrock model ID to test")
    parser.add_argument("--query", default="Identify yourself and confirm you're working",
                      help="Query to send to the model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    return parser.parse_args()


def create_clients(region: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """Create Bedrock clients using the specified region and optional profile."""
    try:
        if profile:
            print(f"Using AWS profile: {profile}")
            session = boto3.Session(profile_name=profile)
            runtime_client = session.client('bedrock-runtime', region_name=region)
            bedrock_client = session.client('bedrock', region_name=region)
        else:
            print("Using default AWS credentials")
            runtime_client = boto3.client('bedrock-runtime', region_name=region)
            bedrock_client = boto3.client('bedrock', region_name=region)
        
        return {
            "runtime": runtime_client,
            "bedrock": bedrock_client
        }
    except NoCredentialsError:
        print("❌ ERROR: No AWS credentials found")
        print("Please configure credentials using one of the following methods:")
        print("  - Environment variables (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)")
        print("  - AWS credentials file (~/.aws/credentials)")
        print("  - IAM role for EC2/ECS/Lambda")
        sys.exit(1)
    except ClientError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)


def list_available_models(bedrock_client: Any) -> List[Dict[str, Any]]:
    """List available foundation models in Bedrock."""
    try:
        response = bedrock_client.list_foundation_models()
        return response.get("modelSummaries", [])
    except ClientError as e:
        print(f"❌ ERROR: Unable to list models: {e}")
        return []


def test_bedrock_access(client: Any, model_id: str, query: str) -> Dict[str, Any]:
    """Test access to Bedrock by sending a simple message."""
    try:
        # Determine model family based on ID
        is_claude = "claude" in model_id.lower()
        is_nova = "nova" in model_id.lower()
        is_meta = "llama" in model_id.lower() or "meta" in model_id.lower()
        
        # Ensure model ID has proper region prefix 
        if not model_id.startswith("us.") and not model_id.startswith("eu."):
            print("Note: Adding 'us.' prefix to model ID for proper regional access")
            model_id = f"us.{model_id}"
        
        if is_claude:
            # Claude format
            response = client.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",  # Claude on Bedrock requires type field
                                "text": query
                            }
                        ]
                    }
                ],
                inferenceConfig={
                    "temperature": 0.7,
                    "maxTokens": 100
                }
            )
        elif is_nova:
            # Amazon Nova format - try with Converse API directly
            try:
                response = client.converse(
                    modelId=model_id,
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
            except ClientError as nova_error:
                # If Converse API doesn't work, try InvokeModel with mininal parameters
                print(f"Converse API failed for Nova model, trying InvokeModel. Error: {nova_error}")
                
                # Try with the most minimal format possible
                body = {
                    "prompt": query,
                    "max_tokens": 100
                }
                
                try:
                    response = client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(body)
                    )
                    
                    # Parse response body
                    response_body = json.loads(response["body"].read().decode())
                    
                    # Reformat to match Converse API format
                    response = {
                        "output": {
                            "content": [
                                {
                                    "text": response_body.get("completion", "")
                                }
                            ]
                        },
                        "usage": {
                            "inputTokens": 0,
                            "outputTokens": 0
                        }
                    }
                except ClientError as e:
                    # If that fails too, raise the error with more info
                    raise ClientError(
                        error_response={
                            "Error": {
                                "Code": e.response["Error"]["Code"],
                                "Message": f"Nova model format error. Original error: {e.response['Error']['Message']}"
                            }
                        },
                        operation_name="InvokeModel"
                    )
        elif is_meta:
            # Meta Llama format 
            body = {
                "prompt": query,
                "max_gen_len": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            # Use InvokeModel for Meta models
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            
            # Parse response body
            response_body = json.loads(response["body"].read().decode())
            
            # Reformat to match Converse API format
            response = {
                "output": {
                    "content": [
                        {
                            "text": response_body.get("generation", "")
                        }
                    ]
                },
                "usage": {
                    "inputTokens": 0,  # Meta models don't provide token counts
                    "outputTokens": 0
                }
            }
        else:
            # Generic format for other models
            body = {
                "inputText": query,
                "textGenerationConfig": {
                    "maxTokenCount": 100,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
            
            # Use InvokeModel for other models
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            
            # Parse response body
            response_body = json.loads(response["body"].read().decode())
            
            # Reformat to match Converse API format
            response = {
                "output": {
                    "content": [
                        {
                            "text": response_body.get("outputText", "")
                        }
                    ]
                },
                "usage": {
                    "inputTokens": response_body.get("inputTextTokenCount", 0),
                    "outputTokens": response_body.get("outputTextTokenCount", 0)
                }
            }
            
        return response
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        
        if error_code == "AccessDeniedException":
            print("❌ ERROR: Access denied to Bedrock")
            print("Your AWS credentials don't have permission to use Bedrock.")
            print("Required permissions: bedrock:ListFoundationModels, bedrock-runtime:InvokeModel, bedrock-runtime:Converse")
            print(f"Error details: {error_message}")
        elif error_code == "ModelNotReadyException":
            print(f"❌ ERROR: Model {model_id} is not ready or not available in region")
            print("Try a different model or check the model's availability in your region.")
        elif error_code == "ModelNotFoundException":
            print(f"❌ ERROR: Model {model_id} not found")
            print("Check that the model ID is correct and available in your region.")
            print("For Claude models, try adding 'us.' or 'eu.' prefix (e.g., 'us.anthropic.claude-3-7-sonnet-20250219-v1:0')")
        elif error_code == "ValidationException":
            print(f"❌ ERROR: Validation error: {error_message}")
            if "ID" in error_message and "isn't supported" in error_message:
                print("The model ID format is incorrect. For Claude models in US regions, use 'us.' prefix.")
                print(f"Try: us.{model_id}")
            else:
                print("Check if the model requires a specific message format.")
        else:
            print(f"❌ ERROR: {error_code} - {error_message}")
        
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)


def verify_iam_permissions(client: Any) -> None:
    """Verify that the current credentials have the necessary IAM permissions."""
    try:
        # This is a simple operation to check permissions
        client.get_foundation_model(modelIdentifier="us.anthropic.claude-3-5-sonnet-20240620-v1:0")
        print("✓ IAM permissions check: You have at least basic bedrock:GetFoundationModel permission")
    except ClientError as e:
        error_code = e.response["Error"]["Code"] 
        if error_code == "AccessDeniedException":
            print("❌ WARNING: Your credentials lack some Bedrock permissions")
            print("You need the following permissions for full access:")
            print("  - bedrock:ListFoundationModels")
            print("  - bedrock:GetFoundationModel")
            print("  - bedrock-runtime:InvokeModel")
            print("  - bedrock-runtime:Converse")
        elif error_code == "ResourceNotFoundException":
            # This is actually okay - it just means the model doesn't exist but we could access the API
            print("✓ IAM permissions check: You have basic Bedrock API access")
        else:
            print(f"❌ WARNING: Permission check returned: {error_code}")


def print_model_info(models: List[Dict[str, Any]]) -> None:
    """Print formatted information about available models."""
    if not models:
        print("No models available with your credentials in this region.")
        return
    
    print("\n===== Available Bedrock Models =====")
    print(f"Found {len(models)} models")
    
    # Organize by provider
    providers = {}
    for model in models:
        provider = model.get("providerName", "Unknown")
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(model)
    
    # Print by provider
    for provider, provider_models in sorted(providers.items()):
        print(f"\n## {provider}")
        for model in provider_models:
            model_id = model.get("modelId", "Unknown")
            name = model.get("modelName", "Unknown")
            print(f"  - {model_id}")
            print(f"    Name: {name}")
            if "customizationsSupported" in model:
                custom = ", ".join(model.get("customizationsSupported", []))
                if custom:
                    print(f"    Customizations: {custom}")
            input_modalities = ", ".join(model.get("inputModalities", ["text"]))
            output_modalities = ", ".join(model.get("outputModalities", ["text"]))
            print(f"    Modalities: {input_modalities} → {output_modalities}")
            print()


def main():
    """Main function to validate AWS credentials."""
    args = parse_args()
    print(f"Testing AWS credentials for Bedrock access in {args.region}")
    
    # Create clients
    clients = create_clients(args.region, args.profile)
    runtime_client = clients["runtime"]
    bedrock_client = clients["bedrock"]
    
    # Verify base permissions
    print("\nChecking IAM permissions...")
    verify_iam_permissions(bedrock_client)
    
    # List available models if requested
    if args.list_models:
        print("\nListing available Bedrock models...")
        models = list_available_models(bedrock_client)
        print_model_info(models)
    
    # Test model access
    print(f"\nTesting model access with {args.model}...")
    try:
        response = test_bedrock_access(runtime_client, args.model, args.query)
        
        # Success, print response
        print("\n✅ SUCCESS! Your AWS credentials can access Bedrock")
        print("\nResponse from Bedrock:")
        
        # Extract the model's response text
        if "output" in response and "content" in response["output"]:
            content = response["output"]["content"]
            for item in content:
                if "text" in item:
                    print(f"\n{item.get('text')}")
        else:
            if args.verbose:
                print("\nUnexpected response format:")
                print(json.dumps(response, indent=2))
            else:
                print("\nGot a response in an unexpected format.")
                print("Run with --verbose to see full response.")
        
        # Print token usage
        if "usage" in response:
            print("\nToken usage:")
            print(f"  Input tokens: {response['usage'].get('inputTokens', 'n/a')}")
            print(f"  Output tokens: {response['usage'].get('outputTokens', 'n/a')}")
            
        print("\n✅ Verification complete! AWS credentials are working correctly for Bedrock.")
        print("You can use these credentials in your fast-agent configuration.")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()