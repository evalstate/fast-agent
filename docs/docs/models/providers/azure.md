---
title: Azure OpenAI
social:
  title: Azure OpenAI
  tagline: Configure Azure OpenAI deployments, authentication, and regional capabilities.
  description: Configure Azure OpenAI deployments, authentication, and regional capabilities.
  alt: fast-agent social card — Azure OpenAI
---

# Azure OpenAI

## ⚠️ Check Model and Feature Availability by Region

Before deploying an LLM model in Azure, **always check the official Azure documentation to verify that the required model and capabilities (vision, audio, etc.) are available in your region**. Availability varies by region and by feature. Use the links below to confirm support for your use case:

**Key Capabilities and Official Documentation:**

- **General model list & region availability:**
  [Azure OpenAI Service models – Region availability (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models)
- **Vision (GPT-4 Turbo with Vision, GPT-4o, o1, etc.):**
  [How-to: GPT with Vision (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision)
- **Audio / Whisper:**
  [The Whisper model from OpenAI (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/whisper-overview)
  [Audio concepts in Azure OpenAI (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/audio)
- **PDF / Documents:**
  [Azure AI Foundry feature availability across clouds regions (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-foundry/reference/region-support)

**Summary:**

- **Vision (multimodal):** Models like GPT-4 Turbo with Vision, GPT-4o, o1, etc. are only available in certain regions. In the Azure Portal, the "Model deployments" → "Add deployment" tab lists only those available in your region. See the linked guide for input limits and JSON output.
- **Audio / Whisper:** There are two options: (1) Azure OpenAI (same `/audio/*` routes as OpenAI, limited regions), and (2) Azure AI Speech (more regions, different billing). See the links for region tables.
- **PDF / Documents:** Azure OpenAI does not natively process PDFs. Use [Azure AI Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/form-recognizer/) or [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/) for document processing. The AI Foundry table shows where each feature is available.

**Conclusion:** Before deploying, verify that your Azure resource's region supports the required model and features. If not, create the resource in a supported region or wait for general availability.

Azure OpenAI provides all the capabilities of OpenAI models within Azure's secure and compliant cloud environment. fast-agent supports three authentication methods:

1. Using `resource_name` and `api_key` (standard method)
2. Using `base_url` and `api_key` (for custom endpoints or sovereign clouds)
3. Using `base_url` and DefaultAzureCredential (for managed identity, Azure CLI, etc.)

**YAML Configuration:**

```yaml
# Option 1: Standard configuration with resource_name
azure:
  api_key: "your_azure_openai_key" # Required unless using DefaultAzureCredential
  resource_name: "your-resource-name" # Resource name (do NOT include if using base_url)
  azure_deployment: "deployment-name" # Required - the model deployment name
  api_version: "2024-10-21" # Optional, default shown
  default_headers:
    Ocp-Apim-Subscription-Key: "${AZURE_OPENAI_API_KEY}"
  # Do NOT include base_url if you use resource_name

# Option 2: Custom endpoint with base_url
azure:
  api_key: "your_azure_openai_key"
  base_url: "https://your-resource-name.openai.azure.com" # Full endpoint URL
  azure_deployment: "deployment-name"
  api_version: "2024-10-21" # Optional
  # Do NOT include resource_name if you use base_url

# Option 3: Using DefaultAzureCredential (requires azure-identity package)
azure:
  use_default_azure_credential: true
  base_url: "https://your-resource-name.openai.azure.com"
  azure_deployment: "deployment-name"
  api_version: "2024-10-21" # Optional
  # Do NOT include api_key or resource_name when using DefaultAzureCredential
```

**Important Configuration Notes:**
- Use either `resource_name` or `base_url`, not both.
- When using `DefaultAzureCredential`, do NOT include `api_key` or `resource_name`.
- When using `base_url`, do NOT include `resource_name`.
- When using `resource_name`, do NOT include `base_url`.
- `default_headers` can be used with any option (for example, APIM subscription keys).

**Environment Variables:**

- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Override the API endpoint

**Model Name Format:**

Use `azure.deployment-name` as the model string, where `deployment-name` is the name of your Azure OpenAI deployment.
