#!/usr/bin/env python
"""
Script to check available models in Amazon Bedrock across regions.
"""

import boto3
import json
from botocore.exceptions import ClientError

# Regions where Bedrock might be available
REGIONS = [
    "us-east-1",      # US East (N. Virginia)
    "us-west-2",      # US West (Oregon)
    "ap-northeast-1", # Asia Pacific (Tokyo)
    "eu-central-1",   # Europe (Frankfurt)
    "eu-west-1",      # Europe (Ireland)
]

def check_bedrock_models(region):
    """Check available Bedrock models in the specified region."""
    print(f"\nModels in region: {region}")
    print("-" * 50)
    
    try:
        # Create Bedrock client for management operations
        client = boto3.client('bedrock', region_name=region)
        
        # List foundation models
        response = client.list_foundation_models()
        
        # Print available models, focusing on the most relevant info
        model_summaries = response.get('modelSummaries', [])
        if not model_summaries:
            print("No models found in this region.")
            return
        
        # Filter for models from key providers we're interested in
        providers = {
            "Amazon": [],
            "Anthropic": [],
            "Meta": [],
            "Cohere": [],
            "Mistral": [],
            "Other": []
        }
        
        for model in model_summaries:
            model_id = model.get('modelId', 'Unknown')
            provider = model.get('providerName', 'Unknown')
            
            if "anthropic" in model_id.lower() or "claude" in model_id.lower():
                providers["Anthropic"].append(model_id)
            elif "amazon" in model_id.lower() or "titan" in model_id.lower():
                providers["Amazon"].append(model_id)
            elif "meta" in model_id.lower() or "llama" in model_id.lower():
                providers["Meta"].append(model_id)
            elif "cohere" in model_id.lower():
                providers["Cohere"].append(model_id)
            elif "mistral" in model_id.lower():
                providers["Mistral"].append(model_id)
            else:
                providers["Other"].append(model_id)
        
        # Print organized by provider
        for provider, models in providers.items():
            if models:
                print(f"\n{provider} Models:")
                for model in sorted(models):
                    print(f"  - {model}")
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"Error ({error_code}): {error_message}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("Checking available Amazon Bedrock models across regions...")
    
    for region in REGIONS:
        check_bedrock_models(region)
    
    print("\nDone checking regions!")