# Implementation Plan: Anthropic Extended Thinking for fast-agent

## Overview

This plan covers the implementation of Anthropic's Extended Thinking feature in `llm_anthropic.py`, enabling enhanced reasoning capabilities with proper streaming, channel preservation, and interleaved thinking with tool use.

---

## 1. Configuration & Types

### 1.1 Add Thinking Configuration to RequestParams

**File:** `src/fast_agent/llm/request_params.py`

Add new fields to support thinking configuration:

```python
thinking_budget_tokens: int | None = None
"""Token budget for extended thinking. When set, enables extended thinking mode."""

thinking_enabled: bool = False
"""Explicitly enable/disable extended thinking (alternative to budget)"""
```

### 1.2 Add Anthropic-Specific Config

**File:** `src/fast_agent/config.py`

Add thinking defaults to `AnthropicConfig`:

```python
thinking_budget_tokens: int = 10000  # Default budget
thinking_enabled: bool = False  # Off by default
```

### 1.3 New Types for Thinking Blocks

**File:** `src/fast_agent/llm/provider/anthropic/thinking_types.py` (new file)

Create types to represent thinking content for channel storage:

```python
from pydantic import BaseModel
from mcp.types import ContentBlock

class ThinkingContent(BaseModel):
    """Extended thinking content stored in channels."""
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str  # Required for passing back to API

class RedactedThinkingContent(BaseModel):
    """Redacted thinking content (encrypted)."""
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str  # Encrypted content
```

Register these as valid ContentBlock types or use a separate storage mechanism.

---

## 2. Streaming Implementation

### 2.1 Handle Thinking Stream Events

**File:** `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`

Modify `_process_stream()` to handle thinking-specific events:

```python
from anthropic.types import (
    # Existing imports...
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    # New thinking-related types
    ThinkingBlock,
    ThinkingDelta,
)

async def _process_stream(self, stream, model, capture_filename=None) -> tuple[Message, list[ThinkingContent]]:
    """Process streaming response including thinking blocks."""

    estimated_tokens = 0
    tool_streams: dict[int, dict[str, Any]] = {}
    thinking_streams: dict[int, dict[str, Any]] = {}  # NEW: Track thinking blocks
    collected_thinking: list[ThinkingContent] = []  # NEW: Collect completed thinking

    async for event in stream:
        _save_stream_chunk(capture_filename, event)

        if isinstance(event, RawContentBlockStartEvent):
            content_block = event.content_block

            # NEW: Handle thinking block start
            if content_block.type == "thinking":
                thinking_streams[event.index] = {
                    "thinking_buffer": [],
                    "signature": None,
                }
                self.logger.info(
                    "Model started thinking",
                    data={
                        "progress_action": ProgressAction.THINKING,
                        "agent_name": self.name,
                        "model": model,
                    },
                )
                continue

            # Existing tool handling...
            if isinstance(content_block, ToolUseBlock):
                # ... existing code ...

        if isinstance(event, RawContentBlockDeltaEvent):
            delta = event.delta

            # NEW: Handle thinking delta
            if delta.type == "thinking_delta":
                info = thinking_streams.get(event.index)
                if info is not None:
                    thinking_text = delta.thinking
                    info["thinking_buffer"].append(thinking_text)
                    # Stream thinking to listeners
                    self._notify_stream_listeners(
                        StreamChunk(text=thinking_text, is_reasoning=True)
                    )
                continue

            # NEW: Handle signature delta (comes just before block stop)
            if delta.type == "signature_delta":
                info = thinking_streams.get(event.index)
                if info is not None:
                    info["signature"] = delta.signature
                continue

            # Existing InputJSONDelta and TextDelta handling...

        if isinstance(event, RawContentBlockStopEvent):
            # NEW: Handle thinking block completion
            if event.index in thinking_streams:
                info = thinking_streams.pop(event.index)
                thinking_text = "".join(info["thinking_buffer"])
                signature = info["signature"]

                if thinking_text and signature:
                    collected_thinking.append(ThinkingContent(
                        thinking=thinking_text,
                        signature=signature,
                    ))

                self.logger.info(
                    "Model finished thinking",
                    data={
                        "progress_action": ProgressAction.THINKING,
                        "agent_name": self.name,
                        "model": model,
                        "thinking_length": len(thinking_text),
                    },
                )
                continue

            # Existing tool stop handling...

    message = await stream.get_final_message()
    return message, collected_thinking
```

### 2.2 Add THINKING Progress Action

**File:** `src/fast_agent/event_progress.py`

```python
class ProgressAction(str, Enum):
    # ... existing ...
    THINKING = "Thinking"  # Add if not already present
```

---

## 3. API Request Modifications

### 3.1 Build Thinking Request Arguments

**File:** `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`

Modify `_anthropic_completion()` to include thinking configuration:

```python
async def _anthropic_completion(
    self,
    message_param,
    request_params: RequestParams | None = None,
    structured_model: Type[ModelT] | None = None,
    tools: list[Tool] | None = None,
    pre_messages: list[MessageParam] | None = None,
    history: list[PromptMessageExtended] | None = None,
    current_extended: PromptMessageExtended | None = None,
) -> PromptMessageExtended:
    # ... existing setup ...

    # Determine if thinking is enabled
    thinking_enabled = self._is_thinking_enabled(request_params, structured_model)
    thinking_budget = self._get_thinking_budget(request_params)

    # Build base arguments
    base_args = {
        "model": model,
        "messages": messages,
        "stop_sequences": params.stopSequences,
        "tools": available_tools,
    }

    # NEW: Add thinking configuration
    if thinking_enabled:
        base_args["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        # CRITICAL: Ensure tool_choice is auto (or none) when thinking is enabled
        # Cannot force tool_choice with thinking
        if "tool_choice" in base_args:
            del base_args["tool_choice"]

    # CONFLICT: Structured output requires forced tool_choice
    # When thinking is enabled, we cannot use structured output via tool forcing
    if structured_model:
        if thinking_enabled:
            # Option 1: Disable thinking for structured output
            logger.warning(
                "Extended thinking disabled for structured output request "
                "(tool_choice forcing is incompatible with thinking)"
            )
            del base_args["thinking"]
            base_args["tool_choice"] = {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL_NAME}
        else:
            base_args["tool_choice"] = {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL_NAME}

    # ... rest of method ...
```

### 3.2 Helper Methods for Thinking Configuration

```python
def _is_thinking_enabled(
    self,
    request_params: RequestParams | None,
    structured_model: Type[ModelT] | None
) -> bool:
    """Determine if thinking should be enabled for this request."""
    # Structured output is incompatible with thinking
    if structured_model:
        return False

    # Check request params first
    params = self.get_request_params(request_params)
    if hasattr(params, 'thinking_enabled') and params.thinking_enabled:
        return True
    if hasattr(params, 'thinking_budget_tokens') and params.thinking_budget_tokens:
        return True

    # Fall back to config
    if self.context.config and self.context.config.anthropic:
        return getattr(self.context.config.anthropic, 'thinking_enabled', False)

    return False

def _get_thinking_budget(self, request_params: RequestParams | None) -> int:
    """Get the thinking token budget."""
    DEFAULT_BUDGET = 10000
    MIN_BUDGET = 1024

    params = self.get_request_params(request_params)
    if hasattr(params, 'thinking_budget_tokens') and params.thinking_budget_tokens:
        return max(MIN_BUDGET, params.thinking_budget_tokens)

    if self.context.config and self.context.config.anthropic:
        budget = getattr(self.context.config.anthropic, 'thinking_budget_tokens', DEFAULT_BUDGET)
        return max(MIN_BUDGET, budget)

    return DEFAULT_BUDGET
```

---

## 4. Channel Preservation for Thinking Blocks

### 4.1 Store Thinking in REASONING Channel

**File:** `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`

Modify the response building to include thinking blocks:

```python
async def _anthropic_completion(...) -> PromptMessageExtended:
    # ... existing code ...

    # Process stream and collect thinking
    response, collected_thinking = await self._process_stream(stream, model, capture_filename)

    # ... existing response processing ...

    # NEW: Build channels with thinking content
    channels: dict[str, list[ContentBlock]] = {}

    if collected_thinking:
        # Store thinking content in REASONING channel
        # We need to store both the text AND signature for later replay
        thinking_blocks = []
        for tc in collected_thinking:
            # Store as structured data that can be serialized/deserialized
            thinking_blocks.append(TextContent(
                type="text",
                text=json.dumps({
                    "type": "thinking",
                    "thinking": tc.thinking,
                    "signature": tc.signature,
                })
            ))
        channels[REASONING] = thinking_blocks

    return PromptMessageExtended(
        role="assistant",
        content=response_content_blocks,
        tool_calls=tool_calls,
        channels=channels if channels else None,  # NEW: Include channels
        stop_reason=stop_reason,
    )
```

### 4.2 Alternative: Store Thinking Blocks Separately

For better type safety and to support passing blocks back to API:

```python
# In PromptMessageExtended, add a new field (or use a specialized channel)
THINKING_CHANNEL = "anthropic-thinking"  # Separate from REASONING for API compatibility

# Store the raw blocks that can be passed back to API
thinking_channel_content = [
    TextContent(type="text", text=json.dumps(block.model_dump()))
    for block in collected_thinking
]
channels[THINKING_CHANNEL] = thinking_channel_content
```

---

## 5. Interleaved Thinking with Tool Use

### 5.1 Enable Interleaved Thinking Beta

**File:** `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`

When making API calls with thinking + tools:

```python
async def _anthropic_completion(...) -> PromptMessageExtended:
    # ... existing code ...

    try:
        # Build client with beta header for interleaved thinking
        extra_headers = {}
        if thinking_enabled and available_tools:
            extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        anthropic = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers if extra_headers else None,
        )
        # ... rest of API call ...
```

### 5.2 Preserve Thinking Blocks Across Tool Use

**File:** `src/fast_agent/llm/provider/anthropic/multipart_converter_anthropic.py`

Modify the converter to include thinking blocks when converting assistant messages:

```python
THINKING_CHANNEL = "anthropic-thinking"

@staticmethod
def convert_to_anthropic(multipart_msg: PromptMessageExtended) -> MessageParam:
    role = multipart_msg.role
    all_content_blocks = []

    # NEW: For assistant messages, include thinking blocks FIRST
    if role == "assistant" and multipart_msg.channels:
        thinking_data = multipart_msg.channels.get(THINKING_CHANNEL)
        if thinking_data:
            for block in thinking_data:
                if isinstance(block, TextContent):
                    try:
                        data = json.loads(block.text)
                        if data.get("type") == "thinking":
                            all_content_blocks.append({
                                "type": "thinking",
                                "thinking": data["thinking"],
                                "signature": data["signature"],
                            })
                        elif data.get("type") == "redacted_thinking":
                            all_content_blocks.append({
                                "type": "redacted_thinking",
                                "data": data["data"],
                            })
                    except json.JSONDecodeError:
                        pass

    # If this is an assistant message that contains tool_calls...
    if role == "assistant" and multipart_msg.tool_calls:
        for tool_use_id, req in multipart_msg.tool_calls.items():
            # ... existing tool_use block creation ...
        return MessageParam(role=role, content=all_content_blocks)

    # ... rest of existing logic ...
```

### 5.3 Handle Thinking in Tool Use Loop

The key insight is that during tool use, we MUST:
1. Preserve thinking blocks from the assistant's response
2. Pass them back FIRST when sending tool results
3. The thinking must precede the tool_use blocks

This is handled by the multipart converter changes above.

---

## 6. Structured Output Considerations

### 6.1 Incompatibility Resolution

Extended thinking is incompatible with `tool_choice: {type: "tool", name: "..."}`. Two options:

**Option A (Recommended): Disable thinking for structured output**
- Structured output remains unchanged
- Thinking is silently disabled when structured_model is provided
- Log a warning for visibility

**Option B: Alternative structured approach with thinking**
- Use a prompt-based approach instead of tool forcing
- Add thinking, then parse JSON from the text response
- More complex and less reliable

Implement Option A as described in section 3.1.

---

## 7. Usage Tracking Updates

### 7.1 Track Thinking Tokens

**File:** `src/fast_agent/llm/usage_tracking.py`

The Anthropic usage already includes thinking tokens in output. Update `from_anthropic`:

```python
@classmethod
def from_anthropic(cls, usage: AnthropicUsage, model: str) -> "TurnUsage":
    # Existing cache handling...

    # NEW: Extract thinking tokens if available
    # Note: Claude 4 models return summarized thinking, but bill for full thinking
    # The output_tokens includes thinking tokens
    thinking_tokens = 0
    # If Anthropic SDK exposes thinking token count separately, use it:
    # thinking_tokens = getattr(usage, "thinking_tokens", 0) or 0

    return cls(
        provider=Provider.ANTHROPIC,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        cache_usage=cache_usage,
        reasoning_tokens=thinking_tokens,  # Add thinking tokens
        raw_usage=usage,
    )
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**New file:** `tests/unit/llm/provider/anthropic/test_thinking.py`

- Test thinking configuration detection
- Test thinking stream event handling
- Test thinking block storage in channels
- Test thinking block preservation across tool use
- Test structured output disables thinking

### 8.2 Integration Tests

**New file:** `tests/e2e/llm/test_anthropic_thinking.py`

- Test streaming thinking content display
- Test thinking with tool use (interleaved)
- Test multi-turn conversations with thinking
- Test redacted thinking handling

### 8.3 Mock Responses

Create mock responses with thinking blocks for testing:

```python
MOCK_THINKING_RESPONSE = {
    "content": [
        {
            "type": "thinking",
            "thinking": "Let me analyze this step by step...",
            "signature": "base64_signature_here"
        },
        {
            "type": "text",
            "text": "Based on my analysis..."
        }
    ]
}
```

---

## 9. Implementation Order

1. **Phase 1: Configuration** (1-2 tasks)
   - Add thinking config to RequestParams
   - Add thinking config to AnthropicConfig

2. **Phase 2: Basic Streaming** (2-3 tasks)
   - Handle thinking block start/delta/stop events
   - Stream thinking content with `is_reasoning=True`
   - Store thinking in channels

3. **Phase 3: API Integration** (2-3 tasks)
   - Add thinking parameter to API requests
   - Handle budget_tokens configuration
   - Disable thinking for structured output

4. **Phase 4: Tool Use Integration** (3-4 tasks)
   - Add beta header for interleaved thinking
   - Preserve thinking blocks in converter
   - Pass thinking blocks back with tool results
   - Test multi-turn tool use with thinking

5. **Phase 5: Polish & Testing** (2-3 tasks)
   - Update usage tracking
   - Add comprehensive tests
   - Documentation updates

---

## 10. Key Constraints & Gotchas

1. **tool_choice incompatibility**: Cannot use `tool_choice: {type: "tool"}` or `{type: "any"}` with thinking. Only `auto` or `none` allowed.

2. **budget_tokens vs max_tokens**: `budget_tokens` must be less than `max_tokens`. With interleaved thinking, budget can exceed max_tokens.

3. **Thinking block order**: In assistant messages, thinking blocks must come BEFORE tool_use blocks.

4. **Signature is required**: The signature field must be preserved and passed back unchanged for tool use continuation.

5. **Redacted thinking**: Some thinking may be encrypted. Must preserve and pass back redacted_thinking blocks unchanged.

6. **Summarization (Claude 4)**: Claude 4 models return summarized thinking, but bill for full thinking tokens.

7. **Claude 3.7 vs Claude 4**: Different behavior - 3.7 returns full thinking, 4.x returns summarized.

---

## 11. Files to Modify

| File | Changes |
|------|---------|
| `src/fast_agent/llm/request_params.py` | Add thinking config fields |
| `src/fast_agent/config.py` | Add Anthropic thinking defaults |
| `src/fast_agent/llm/provider/anthropic/llm_anthropic.py` | Main implementation |
| `src/fast_agent/llm/provider/anthropic/multipart_converter_anthropic.py` | Thinking block preservation |
| `src/fast_agent/llm/usage_tracking.py` | Track thinking tokens |
| `src/fast_agent/constants.py` | Add THINKING_CHANNEL constant (optional) |
| `src/fast_agent/event_progress.py` | Ensure THINKING action exists |
| `tests/unit/llm/provider/anthropic/test_thinking.py` | Unit tests |
| `tests/e2e/llm/test_anthropic_thinking.py` | E2E tests |

---

## 12. Dependencies

- Anthropic SDK version supporting extended thinking
- Types: `ThinkingBlock`, `ThinkingDelta`, `signature_delta` event types
- Beta header support in SDK

Check SDK version requirements:
```python
# Minimum version check
import anthropic
# Verify ThinkingBlock type exists
from anthropic.types import ThinkingBlock
```
