#!/usr/bin/env python
"""
Test script for running a real Bedrock agent with actual AWS credentials.
Use this script to test that your Bedrock integration is working correctly.
"""

import os
import asyncio
import argparse
import sys
from typing import Optional, Dict, Any

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test real Bedrock agent with AWS credentials")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--model", default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                     help="Bedrock model ID to test")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds (default: 120)")
    return parser.parse_args()


async def test_bedrock_direct():
    """Test Bedrock directly using boto3 (as a fallback)."""
    args = parse_args()
    
    print("\nFalling back to direct Bedrock API test...\n")
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        # Create boto3 session with the provided profile if any
        if args.profile:
            session = boto3.Session(profile_name=args.profile)
        else:
            session = boto3.Session()
        
        # Create Bedrock client
        bedrock_client = session.client('bedrock-runtime', region_name=args.region)
        
        # Determine if this is a Claude model
        is_claude = "claude" in args.model.lower()
        
        # Ensure proper prefix
        if not args.model.startswith("us.") and not args.model.startswith("eu."):
            args.model = f"us.{args.model}"
            
        # Claude format
        if is_claude:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please identify yourself and confirm you are working."
                            }
                        ]
                    }
                ]
            }
            
            # Use Converse API
            response = bedrock_client.converse(
                modelId=args.model,
                messages=request_body["messages"],
                inferenceConfig={
                    "temperature": 0.7,
                    "maxTokens": 100
                }
            )
            
            # Print response
            if "output" in response and "message" in response["output"]:
                message = response["output"]["message"]
                if "content" in message:
                    for item in message["content"]:
                        if "text" in item:
                            print("\nResponse from Bedrock:\n")
                            print(item["text"])
                            
                # Print token usage
                if "usage" in response:
                    print("\nToken usage:")
                    print(f"  Input tokens: {response['usage'].get('inputTokens', 'unknown')}")
                    print(f"  Output tokens: {response['usage'].get('outputTokens', 'unknown')}")
                
                print("\n✅ SUCCESS! Direct Bedrock test completed successfully.")
            else:
                print("\nResponse format not recognized:")
                print(response)
                
    except Exception as e:
        print(f"\n❌ ERROR in direct Bedrock test: {e}")


async def main():
    """Run the Bedrock agent test."""
    args = parse_args()
    
    # Set a timeout for the whole process
    try:
        # Check AWS credentials first
        print(f"Testing AWS credentials for Bedrock access...")
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            # Create a boto3 session with the provided profile if any
            if args.profile:
                session = boto3.Session(profile_name=args.profile)
            else:
                session = boto3.Session()
            
            # Check if credentials are available
            creds = session.get_credentials()
            if creds is None:
                print("❌ ERROR: No AWS credentials found.")
                print("Please configure credentials using one of the following methods:")
                print("  - AWS CLI: aws configure")
                print("  - Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
                print("  - IAM role if running on EC2/ECS")
                return
                
            # Print credential info
            print(f"✅ Found AWS credentials:", flush=True)
            print(f"  - Access key ID: {creds.access_key[:4]}...{creds.access_key[-4:]}", flush=True)
            print(f"  - Secret key available: {'Yes' if creds.secret_key else 'No'}", flush=True)
            print(f"  - Token available: {'Yes' if creds.token else 'No'}", flush=True)
            
            # Check Bedrock access
            print(f"Checking Bedrock access in {args.region}...", flush=True)
            bedrock_client = session.client('bedrock-runtime', region_name=args.region)
            
            # Print Bedrock endpoint
            print(f"  - Bedrock endpoint: {bedrock_client._endpoint.host}", flush=True)
        
        except NoCredentialsError:
            print("❌ ERROR: No AWS credentials found.")
            print("Please configure credentials using one of the following methods:")
            print("  - AWS CLI: aws configure")
            print("  - Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            print("  - IAM role if running on EC2/ECS")
            return
        except ClientError as e:
            print(f"❌ ERROR: {e}")
            return
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return
            
        # Create FastAgent instance
        print(f"Initializing Bedrock agent with model: {args.model}")
        print(f"Region: {args.region}")
        if args.profile:
            print(f"AWS Profile: {args.profile}")
        else:
            print("Using default AWS credentials")
        
        # Set environment variables for AWS credentials
        if args.profile:
            os.environ["AWS_PROFILE"] = args.profile
        os.environ["AWS_REGION"] = args.region
        
        try:
            from mcp_agent.core.fastagent import FastAgent
            print("Successfully imported FastAgent from mcp_agent.core.fastagent")
        except ImportError:
            try:
                from fastagent import FastAgent
                print("Successfully imported FastAgent from fastagent")
            except ImportError:
                print("❌ ERROR: Could not import FastAgent. Make sure the package is installed.")
                print("Falling back to direct Bedrock API test...")
                await test_bedrock_direct()
                return
        
        # Determine model ID format
        if args.model.startswith("us.") or args.model.startswith("eu."):
            model_id = f"bedrock.{args.model}"  # Add bedrock. prefix for regional IDs
        else:
            model_id = f"bedrock.{args.model}"  # Add bedrock. prefix for aliases
            
        print(f"Using model ID: {model_id}")
            
        # Create FastAgent with config file if specified, or load default
        try:
            if args.config:
                fast = FastAgent("Bedrock Test Agent", config_path=args.config)
            else:
                # Try with default config
                fast = FastAgent("Bedrock Test Agent")
            
            # Define a simple agent
            @fast.agent(name="bedrock_test", model=model_id)
            async def bedrock_agent(prompt):
                """Simple Bedrock agent function."""
                print("Agent function started")
                # Run the agent with async context manager
                async with fast.run() as session:
                    print("Agent session started")
                    # Send a message to the agent
                    print("Sending message to agent...")
                    response = await session.bedrock_test.send(prompt)
                    print("Message sent, got response")
                    return response
            
            # Run the agent
            print("Running Bedrock agent...")
            response = await bedrock_agent("Please identify yourself and confirm you are working.")
            
            print("\nResponse from Bedrock agent:")
            print(response)
            
            print("\n✅ SUCCESS! FastAgent with Bedrock is working correctly.")
            
            # Run interactive mode if requested
            if args.interactive:
                print("\n=== Starting interactive mode ===")
                print("Type your messages and press Enter to send. Type 'exit' to quit.")
                while True:
                    try:
                        user_input = input("\nYou: ")
                        if user_input.lower() in ('exit', 'quit'):
                            break
                        
                        response = await bedrock_agent(user_input)
                        print(f"\nBedrock: {response}")
                    except KeyboardInterrupt:
                        print("\nInteractive mode terminated by user.")
                        break
                    except Exception as e:
                        print(f"\n❌ ERROR in interactive mode: {e}")
                        break
            
        except Exception as e:
            print(f"❌ ERROR with FastAgent: {e}")
            print("Falling back to direct Bedrock API test...")
            await test_bedrock_direct()
    
    except asyncio.TimeoutError:
        print(f"\n❌ ERROR: Operation timed out after {args.timeout} seconds")
        print("Falling back to direct Bedrock API test...")
        await test_bedrock_direct()
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error: {e}")
        print("Falling back to direct Bedrock API test...")
        await test_bedrock_direct()


if __name__ == "__main__":
    args = parse_args()
    
    # Set up the timeout
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=args.timeout))
    except asyncio.TimeoutError:
        print(f"\n❌ ERROR: Test timed out after {args.timeout} seconds")
        print("Try running with a longer timeout: --timeout 300")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")