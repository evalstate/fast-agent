---
title: Azure OpenAI Configuration Example
description: Example configuration for using Azure OpenAI Service with fast-agent
---

# Azure OpenAI Configuration Example

This example shows how to configure fast-agent to use Azure OpenAI Service with different authentication methods.

## Prerequisites

1. An Azure account with access to Azure OpenAI Service
2. An Azure OpenAI Service resource with model deployments
3. The fast-agent package installed with Azure support: `uv pip install fast-agent-mcp[azure]`

## Configuration File

Below is a sample `fastagent.config.yaml` file with all three authentication methods. Choose the one that fits your needs:

```yaml
# OPTION 1: Using resource_name and api_key (standard method)
default_model: "azure.my-deployment"

azure:
  api_key: "YOUR_AZURE_OPENAI_API_KEY"
  resource_name: "your-resource-name"
  azure_deployment: "my-deployment"
  api_version: "2023-05-15"
  # Do NOT include base_url if you use resource_name

# OPTION 2: Using base_url and api_key (custom endpoints or sovereign clouds)
# default_model: "azure.my-deployment"
#
# azure:
#   api_key: "YOUR_AZURE_OPENAI_API_KEY"
#   base_url: "https://your-resource-name.openai.azure.com/"
#   azure_deployment: "my-deployment"
#   api_version: "2023-05-15"
#   # Do NOT include resource_name if you use base_url

# OPTION 3: Using DefaultAzureCredential (for managed identity, Azure CLI, etc.)
# default_model: "azure.my-deployment"
#
# azure:
#   use_default_azure_credential: true
#   base_url: "https://your-resource-name.openai.azure.com/"
#   azure_deployment: "my-deployment"
#   api_version: "2023-05-15"
#   # Do NOT include api_key or resource_name in this mode
```

**Important Configuration Notes:**
- Use either `resource_name` or `base_url`, not both.
- When using `DefaultAzureCredential`, do NOT include `api_key` or `resource_name`.
- When using `base_url`, do NOT include `resource_name`.
- When using `resource_name`, do NOT include `base_url`.

## Basic Agent Example

Here's a simple agent implementation using Azure OpenAI:

```python
import asyncio
from fast_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Azure OpenAI Example")

# Define the agent using Azure OpenAI deployment
@fast.agent(
    instruction="You are a helpful AI assistant powered by Azure OpenAI Service", 
    model="azure.my-deployment"
)
async def main():
    async with fast.run() as agent:
        # Start interactive prompt
        await agent()

if __name__ == "__main__":
    asyncio.run(main())
```

## Authentication Notes

### Using DefaultAzureCredential

The DefaultAzureCredential authentication method can use various credential sources:
- Environment variables
- Managed identities in Azure
- Azure CLI credentials
- Azure PowerShell credentials
- Visual Studio Code credentials

To use this method:

1. Install the required dependency: `uv pip install fast-agent-mcp[azure]`
2. Configure your environment for Azure authentication (e.g., run `az login`)
3. Use the configuration shown in Option 3 above

This method is ideal for:
- Deployed applications on Azure (App Service, Functions, AKS, etc.)
- Development environments where you're already authenticated to Azure
- Scenarios where secure key management is crucial

### Using API Keys

The API key authentication method is simpler and works in all environments. To find your API key:

1. Go to the Azure Portal
2. Navigate to your Azure OpenAI resource
3. In the "Resource Management" section, select "Keys and Endpoint"
4. Copy one of the keys and the endpoint

Then configure your agent using either Option 1 or Option 2 above.
