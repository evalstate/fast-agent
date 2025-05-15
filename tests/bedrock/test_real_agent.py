#!/usr/bin/env python
"""
Test script for running a real Bedrock agent with actual AWS credentials.
Use this script to test that your Bedrock integration is working correctly.
"""

import os
import asyncio
import argparse
from typing import Optional, Dict, Any

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.llm.providers.augmented_llm_bedrock import BedrockAugmentedLLM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test real Bedrock agent with AWS credentials")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--model", default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                     help="Bedrock model ID to test")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    return parser.parse_args()


def create_config(args) -> Dict[str, Any]:
    """Create a configuration based on command line arguments."""
    # Don't add bedrock. prefix if the model ID already has a region prefix (us. or eu.)
    if args.model.startswith("us.") or args.model.startswith("eu."):
        model_id = args.model  # Use as-is if it has a region prefix
    else:
        model_id = f"bedrock.{args.model}"  # Add bedrock. prefix for aliases
        
    config = {
        "default_model": model_id,
        "bedrock": {
            "region": args.region,
        }
    }
    
    # Set authentication method based on args
    if args.profile:
        config["bedrock"]["profile"] = args.profile
        config["bedrock"]["use_default_credentials"] = False
    else:
        config["bedrock"]["use_default_credentials"] = True
    
    # Set model parameters
    config["bedrock"]["default_params"] = {
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Set logging
    config["logging"] = {
        "level": "DEBUG"
    }
    
    return config


async def main():
    """Run the Bedrock agent test."""
    args = parse_args()
    
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
    
    # Create a new version of the test that uses the local fastagent.config.yaml
    print("Using local fastagent.config.yaml file")
    
    # Set the PYTHONPATH to include the project root
    # This will help with importing the correct modules
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
    
    # Set verbose logging
    os.environ["FASTAGENT_LOG_LEVEL"] = "DEBUG"
    
    # Create FastAgent with local config
    try:
        from mcp_agent.core.fastagent import FastAgent
        print("Successfully imported FastAgent from mcp_agent.core.fastagent")
    except ImportError:
        try:
            from fastagent import FastAgent
            print("Successfully imported FastAgent from fastagent")
        except ImportError:
            print("ERROR: Could not import FastAgent. Make sure the package is installed.")
            return
    
    # Use local config file
    fast = FastAgent("Bedrock Test Agent")

    # Don't add bedrock. prefix if the model ID already has a region prefix (us. or eu.)
    if args.model.startswith("us.") or args.model.startswith("eu."):
        model_id = args.model  # Use as-is if it has a region prefix
    else:
        model_id = f"bedrock.{args.model}"  # Add bedrock. prefix for aliases
    
    @fast.agent(
        "bedrock_agent",
        model=model_id,
        instruction="You are a helpful assistant running on Amazon Bedrock. Keep responses brief and helpful."
    )
    async def agent_function():
        """Run the agent function."""
        async with fast.run() as agent:
            # Verify the agent's LLM is a BedrockAugmentedLLM instance
            if not isinstance(agent.bedrock_agent.llm, BedrockAugmentedLLM):
                print("❌ ERROR: Agent is not using BedrockAugmentedLLM")
                return

            # Print agent configuration
            llm = agent.bedrock_agent.llm
            print("\nAgent Configuration:")
            print(f"  Provider: {llm.provider}")
            print(f"  Region: {llm.region}")
            if llm.profile:
                print(f"  Profile: {llm.profile}")
            print(f"  Default Credentials: {llm.use_default_credentials}")
            
            try:
                # Print more verbose debug info
                print("\nLLM Model:", agent.bedrock_agent.llm.default_request_params.model)
                print("Using BedrockAugmentedLLM:", isinstance(agent.bedrock_agent.llm, BedrockAugmentedLLM))
                print("Creating AWS Bedrock client...")
                
                # Send a test message
                print("\nSending test message...")
                print("Message: Please identify yourself including which model you are running as, "
                      "and confirm you're able to respond to queries.")
                
                response = await agent.bedrock_agent.send(
                    "Please identify yourself including which model you are running as, "
                    "and confirm you're able to respond to queries."
                )
                
                print("\nResponse from Bedrock:")
                print(f"{response}")
                
                print("\n✅ SUCCESS! The Bedrock agent is working correctly.")
                
                # Run interactive mode if requested
                if args.interactive:
                    print("\n=== Starting interactive mode ===")
                    print("Type your messages and press Enter to send. Type 'exit' to quit.")
                    await agent.bedrock_agent.interactive()
                
            except Exception as e:
                print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())