# Amazon Bedrock Provider-Specific Formats

## Overview

Amazon Bedrock provides a unified API through the Converse API, but there are important provider-specific considerations when implementing support for different models. This document outlines the key differences between model providers and how they should be handled in implementation.

## 1. Anthropic Claude Models

### Request Format

```python
{
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 2000,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello, Claude!"
                }
            ]
        }
    ],
    "system": "You are a helpful AI assistant.",
    "temperature": 0.7,
    "top_p": 0.9
}
```

### Key Considerations:
- Uses `anthropic_version` parameter
- Supports `system` for instructions
- Messages contain array of content objects with types
- Supports multimodal content (text, images)
- Supports tool use via `tools` parameter
- Uses standard temperature, top_p, top_k parameters

### Response Format

```python
{
    "id": "msg_01XxXxXxXxXxXxXxXxXxXxXx",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! I'm Claude, an AI assistant. How can I help you today?"
        }
    ],
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 25,
        "output_tokens": 17
    }
}
```

## 2. Meta Llama Models

### Request Format

```python
{
    "prompt": "Human: Hello, Llama!\nAssistant:",
    "max_gen_len": 512,
    "temperature": 0.7,
    "top_p": 0.9
}
```

### Key Considerations:
- Uses simple `prompt` parameter instead of messages
- Requires specific prompt formatting with "Human:" and "Assistant:"
- No native multimodal support
- No dedicated system prompt parameter
- Uses `max_gen_len` instead of `max_tokens`

### Response Format

```python
{
    "generation": "Hello! I'm an AI assistant based on Llama. How can I help you today?",
    "prompt_token_count": 12,
    "generation_token_count": 18,
    "stop_reason": "stop_sequence"
}
```

## 3. Cohere Models

### Request Format

```python
{
    "message": "Hello, Cohere!",
    "chat_history": [
        {
            "role": "USER",
            "message": "Who are you?"
        },
        {
            "role": "CHATBOT",
            "message": "I'm an AI assistant created by Cohere."
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.7,
    "p": 0.9,
    "k": 0,
    "stream": false,
    "return_preamble": true
}
```

### Key Considerations:
- Uses `message` for current message
- History provided in `chat_history` array
- Different role names: "USER", "CHATBOT", "SYSTEM"
- Uses `p` instead of `top_p`
- Has `return_preamble` parameter

### Response Format

```python
{
    "response": "Hello! How can I assist you today?",
    "finish_reason": "COMPLETE",
    "token_count": {
        "input_tokens": 15,
        "output_tokens": 8
    }
}
```

## 4. Mistral Models

### Request Format

```python
{
    "prompt": "User: Hello, Mistral!\nAssistant:",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "stop": ["User:"]
}
```

### Key Considerations:
- Uses simple `prompt` parameter
- Requires specific formatting with "User:" and "Assistant:"
- No structured message format
- No dedicated system prompt parameter
- Can specify custom stop sequences

### Response Format

```python
{
    "outputs": [
        {
            "text": "Hello! I'm Mistral, an AI assistant. How can I help you today?"
        }
    ],
    "tokenUsage": {
        "inputTokens": 10,
        "outputTokens": 15
    }
}
```

## 5. Amazon Titan Models

### Request Format

```python
{
    "inputText": "Human: Hello, Titan!\nAssistant:",
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.7,
        "topP": 0.9,
        "stopSequences": []
    }
}
```

### Key Considerations:
- Uses `inputText` instead of messages
- Configuration nested under `textGenerationConfig`
- Requires specific prompt formatting
- No structured message format
- Uses `maxTokenCount` instead of `max_tokens`

### Response Format

```python
{
    "inputTextTokenCount": 12,
    "results": [
        {
            "tokenCount": 18,
            "outputText": "Hello! I'm Titan, an AI assistant. How can I help you today?",
            "completionReason": "FINISH"
        }
    ]
}
```

## Using the Converse API

The Converse API abstracts many of these differences, providing a more consistent interface across models:

```python
response = bedrock_runtime.converse(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    messages=[
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Hello!'
                }
            ]
        }
    ],
    inferenceConfig={
        'temperature': 0.7,
        'topP': 0.9,
        'maxTokens': 1000
    }
)
```

The Converse API handles the formatting differences internally, but model-specific parameters may still need to be passed through `additionalModelRequestFields`.

## Implementation Strategy for Fast-Agent

1. **Use Converse API as primary interface**:
   - Provides the most consistent experience across models
   - Handles most of the model-specific format conversions

2. **Implement model-specific parameter converters**:
   - For each model provider (Anthropic, Meta, Cohere, etc.)
   - Map fast-agent parameters to model-specific parameters

3. **Create default parameter mappings**:
   - Map standard parameters like temperature, max_tokens to model-specific equivalents
   - Set model-specific defaults based on provider

4. **Handle multimodal content appropriately**:
   - Convert fast-agent content to proper format for each model
   - Only use multimodal features for models that support them