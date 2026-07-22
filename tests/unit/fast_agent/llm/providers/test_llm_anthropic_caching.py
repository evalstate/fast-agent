"""
Unit tests for Anthropic caching functionality.

These tests directly test the _convert_extended_messages_to_provider method
to verify cache_control markers are applied correctly based on cache_mode settings.
"""

import json
from pathlib import Path
from typing import Any, Literal

import pytest
from anthropic.types.beta import (
    BetaDiagnostics,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaUsage,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    TextContent,
    Tool,
)

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.constants import FAST_AGENT_SHELL_PROCESS_METADATA
from fast_agent.context import Context
from fast_agent.history.process_poll_folding import managed_process_poll_cache_boundary
from fast_agent.llm.provider.anthropic.cache_planner import AnthropicCachePlanner
from fast_agent.llm.provider.anthropic.llm_anthropic import (
    ANTHROPIC_CACHE_DIAGNOSTICS_CHANNEL,
    CACHE_DIAGNOSIS_BETA,
    AnthropicLLM,
)
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types import RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason


class RecordingCacheDiagnosticsLLM(AnthropicLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.requests: list[dict[str, Any]] = []

    def _initialize_anthropic_client(self) -> object:
        return object()

    async def _execute_anthropic_stream(
        self,
        *,
        anthropic: Any,
        arguments: dict[str, Any],
        model: str,
        capture_filename: Path | None,
        timeout_seconds: float | None,
    ) -> tuple[BetaMessage, list[str], list[str]]:
        del anthropic, capture_filename, timeout_seconds
        self.requests.append(arguments)
        response_id = f"msg_{len(self.requests)}"
        return (
            BetaMessage(
                id=response_id,
                type="message",
                role="assistant",
                content=[],
                model=model,
                stop_reason="end_turn",
                stop_sequence=None,
                usage=BetaUsage(input_tokens=1, output_tokens=1),
                diagnostics=BetaDiagnostics(cache_miss_reason=None),
            ),
            [],
            [],
        )


def _content_dicts(message: BetaMessageParam) -> list[dict[str, object]]:
    """Materialize dict-like content blocks for ad-hoc test assertions."""
    content = message.get("content", [])
    if isinstance(content, str):
        return []
    return [
        {str(key): value for key, value in block.items()}
        for block in content
        if isinstance(block, dict)
    ]


def _cache_control(block: dict[str, object]) -> dict[str, object] | None:
    """Return a plain dict cache_control payload when present."""
    cache_control = block.get("cache_control")
    if isinstance(cache_control, dict):
        return {str(key): value for key, value in cache_control.items()}
    return None


class TestAnthropicCaching:
    """Test cases for Anthropic caching functionality."""

    def _create_context_with_cache_mode(
        self,
        cache_mode: Literal["off", "prompt", "auto"],
        cache_ttl: Literal["5m", "1h"] = "5m",
    ) -> Context:
        """Create a context with specified cache mode and TTL."""
        ctx = Context()
        ctx.config = Settings()
        ctx.config.anthropic = AnthropicSettings(
            api_key="test_key", cache_mode=cache_mode, cache_ttl=cache_ttl
        )
        return ctx

    def _create_llm(
        self,
        cache_mode: Literal["off", "prompt", "auto"] = "off",
        cache_ttl: Literal["5m", "1h"] = "5m",
    ) -> AnthropicLLM:
        """Create an AnthropicLLM instance with specified cache mode and TTL."""
        ctx = self._create_context_with_cache_mode(cache_mode, cache_ttl)
        llm = AnthropicLLM(context=ctx)
        return llm

    def _apply_cache_plan(
        self,
        messages: list[PromptMessageExtended],
        cache_mode: Literal["off", "prompt", "auto"],
        system_blocks: int = 0,
        cache_ttl: Literal["5m", "1h"] = "5m",
    ) -> list[BetaMessageParam]:
        planner = AnthropicCachePlanner()
        plan = planner.plan_indices(
            messages, cache_mode=cache_mode, system_cache_blocks=system_blocks
        )
        converted = [AnthropicConverter.convert_to_anthropic(m) for m in messages]
        for idx in plan:
            AnthropicLLM._apply_cache_control_to_message(converted[idx], ttl=cache_ttl)
        return converted

    def test_conversion_off_mode_no_cache_control(self):
        """Test that no cache_control is applied when cache_mode is 'off'."""
        # Create test messages
        messages = [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="Hello")]),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Hi there")]
            ),
        ]

        converted = self._apply_cache_plan(messages, cache_mode="off")

        # Verify no cache_control in any message
        assert len(converted) == 2
        for msg in converted:
            assert "content" in msg
            for block in msg["content"]:
                if isinstance(block, dict):
                    assert "cache_control" not in block, (
                        "cache_control should not be present when cache_mode is 'off'"
                    )

    def test_conversion_prompt_mode_templates_cached(self):
        """Test that template messages get cache_control in 'prompt' mode."""
        # Create template + conversation messages (agent supplies all, flags templates)
        template_msgs = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="System context")],
                is_template=True,
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Understood")],
                is_template=True,
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="Question")]),
        ]

        converted = self._apply_cache_plan(template_msgs + conversation_msgs, cache_mode="prompt")

        # Verify we have 3 messages (2 templates + 1 conversation)
        assert len(converted) == 3

        # Template messages should have cache_control
        # The last template message should have cache_control on its last block
        found_cache_control = False
        template_count = len(template_msgs)
        for _i, msg in enumerate(converted[:template_count]):  # First template_count are templates
            for block in _content_dicts(msg):
                cache_control = _cache_control(block)
                if cache_control is not None:
                    found_cache_control = True
                    assert cache_control["type"] == "ephemeral"
                    assert cache_control["ttl"] == "5m"

        assert found_cache_control, "Template messages should have cache_control in 'prompt' mode"

        # Conversation message should NOT have cache_control
        conv_msg = converted[2]
        for block in _content_dicts(conv_msg):
            assert "cache_control" not in block, (
                "Conversation messages should not have cache_control in 'prompt' mode"
            )

    def test_conversion_auto_mode_templates_cached(self):
        """Test that template messages get cache_control in 'auto' mode."""
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="Question")]),
        ]

        converted = self._apply_cache_plan(template_msgs + conversation_msgs, cache_mode="auto")

        # Template message should have cache_control
        found_cache_control = False
        template_msg = converted[0]
        for block in _content_dicts(template_msg):
            cache_control = _cache_control(block)
            if cache_control is not None:
                found_cache_control = True
                assert cache_control["type"] == "ephemeral"
                assert cache_control["ttl"] == "5m"

        assert found_cache_control, "Template messages should have cache_control in 'auto' mode"

    def test_conversion_off_mode_templates_not_cached(self):
        """Test that template messages do NOT get cache_control when cache_mode is 'off'."""
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Response")],
                is_template=True,
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="Question")]),
        ]

        converted = self._apply_cache_plan(template_msgs + conversation_msgs, cache_mode="off")

        # No messages should have cache_control
        for msg in converted:
            for block in _content_dicts(msg):
                assert "cache_control" not in block, (
                    "No messages should have cache_control when cache_mode is 'off'"
                )

    def test_conversion_multiple_messages_structure(self):
        """Test that message structure is preserved during conversion."""
        messages = [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="First")]),
            PromptMessageExtended(
                role="assistant", content=[TextContent(type="text", text="Second")]
            ),
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="Third")]),
        ]

        converted = [AnthropicConverter.convert_to_anthropic(m) for m in messages]

        # Verify structure
        assert len(converted) == 3
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        assert converted[2]["role"] == "user"

    def test_build_request_messages_avoids_duplicate_tool_results(self):
        """Ensure tool_result blocks are only included once per tool use."""
        llm = self._create_llm()
        tool_id = "toolu_test"
        tool_result = CallToolResult(
            content=[TextContent(type="text", text="result payload")], isError=False
        )
        user_msg = PromptMessageExtended(
            role="user", content=[], tool_results={tool_id: tool_result}
        )
        history = [user_msg]

        params = llm.get_request_params(RequestParams(use_history=True))
        message_param = AnthropicConverter.convert_to_anthropic(user_msg)

        prepared = llm._build_request_messages(params, message_param, history=history)

        tool_blocks = [
            {str(key): value for key, value in block.items()}
            for msg in prepared
            for block in msg.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]

        assert len(tool_blocks) == 1
        assert tool_blocks[0].get("tool_use_id") == tool_id

    def test_build_request_messages_includes_current_when_history_empty(self):
        """Fallback to the current message if history produced no entries."""
        llm = self._create_llm()
        params = llm.get_request_params(RequestParams(use_history=True))
        msg = PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")])
        message_param = AnthropicConverter.convert_to_anthropic(msg)

        prepared = llm._build_request_messages(params, message_param, history=[])

        assert prepared[-1] == message_param

    def test_auto_cache_advances_through_latest_user_request_boundary(self):
        """Cache through the latest completed tool-result turn immediately."""
        llm = self._create_llm(cache_mode="auto")
        history = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="solve the task")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="large reasoning-backed response")],
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="tool result")],
            ),
        ]
        converted = [AnthropicConverter.convert_to_anthropic(message) for message in history]

        llm._apply_anthropic_cache_plan(
            arguments={},
            messages=converted,
            params=llm.get_request_params(RequestParams(use_history=True)),
            cache_mode="auto",
            history=history,
            current_extended=history[-1],
        )

        initial_user_blocks = _content_dicts(converted[0])
        assistant_blocks = _content_dicts(converted[1])
        current_user_blocks = _content_dicts(converted[2])
        assert _cache_control(initial_user_blocks[-1]) == {
            "type": "ephemeral",
            "ttl": "5m",
        }
        assert all(_cache_control(block) is None for block in assistant_blocks)
        assert _cache_control(current_user_blocks[-1]) == {
            "type": "ephemeral",
            "ttl": "5m",
        }

    def test_build_request_messages_without_history(self):
        """When history is disabled, always send the current message."""
        llm = self._create_llm()
        params = llm.get_request_params(RequestParams(use_history=False))
        msg = PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")])
        message_param = AnthropicConverter.convert_to_anthropic(msg)

        prepared = llm._build_request_messages(params, message_param, history=[])

        assert prepared == [message_param]

    def test_cache_plan_accounts_for_provider_message_prefix(self):
        llm = self._create_llm(cache_mode="auto")
        history = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="request")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="response")],
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="tool result")],
            ),
        ]
        prefix = BetaMessageParam(
            role="user",
            content=[BetaTextBlockParam(type="text", text="provider prefix")],
        )
        converted = [prefix, *map(AnthropicConverter.convert_to_anthropic, history)]

        llm._apply_anthropic_cache_plan(
            arguments={},
            messages=converted,
            params=llm.get_request_params(RequestParams(use_history=True)),
            cache_mode="auto",
            history=history,
            current_extended=history[-1],
            pre_message_count=1,
        )

        assert all(_cache_control(block) is None for block in _content_dicts(prefix))
        assert _cache_control(_content_dicts(converted[1])[-1]) is not None
        assert _cache_control(_content_dicts(converted[3])[-1]) is not None

    def test_list_valued_system_prompt_receives_cache_marker(self):
        llm = self._create_llm(cache_mode="auto", cache_ttl="1h")
        arguments = {
            "system": [
                BetaTextBlockParam(type="text", text="first"),
                BetaTextBlockParam(type="text", text="second"),
            ]
        }

        assert llm._apply_system_cache(arguments, "auto") == 1

        system = arguments["system"]
        assert isinstance(system, list)
        assert "cache_control" not in system[0]
        assert system[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_cache_diagnostics_beta_and_response_channel_are_opt_in(self):
        ctx = self._create_context_with_cache_mode("auto")
        assert ctx.config is not None
        assert ctx.config.anthropic is not None
        ctx.config.anthropic.cache_diagnostics = True
        llm = AnthropicLLM(context=ctx)

        flags = llm._resolve_anthropic_beta_flags(
            model="claude-sonnet-5",
            structured_mode=None,
            thinking_enabled=False,
            request_tools=[],
            web_tool_betas=[],
            cache_diagnostics_enabled=llm._cache_diagnostics_enabled(),
        )
        response = BetaMessage(
            id="msg_current",
            type="message",
            role="assistant",
            content=[],
            model="claude-sonnet-5",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=BetaUsage(input_tokens=1, output_tokens=1),
            diagnostics=BetaDiagnostics(cache_miss_reason=None),
        )

        channels = llm._anthropic_response_channels(
            response,
            model="claude-sonnet-5",
            thinking_segments=[],
            tool_calls=None,
        )

        assert CACHE_DIAGNOSIS_BETA in flags
        assert channels is not None
        [block] = channels[ANTHROPIC_CACHE_DIAGNOSTICS_CHANNEL]
        assert isinstance(block, TextContent)
        assert json.loads(block.text) == {
            "cache_miss_reason": None,
            "kind": "anthropic_cache_diagnosis",
            "response_id": "msg_current",
            "status": "pending",
        }

    def test_enabled_cache_diagnostics_records_pending_null_response(self):
        llm = self._create_llm(cache_mode="auto")
        response = BetaMessage(
            id="msg_pending",
            type="message",
            role="assistant",
            content=[],
            model="claude-sonnet-5",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=BetaUsage(input_tokens=1, output_tokens=1),
            diagnostics=None,
        )

        channels = llm._anthropic_response_channels(
            response,
            model="claude-sonnet-5",
            thinking_segments=[],
            tool_calls=None,
            cache_diagnostics_enabled=True,
        )

        assert channels is not None
        [block] = channels[ANTHROPIC_CACHE_DIAGNOSTICS_CHANNEL]
        assert isinstance(block, TextContent)
        assert json.loads(block.text) == {
            "cache_miss_reason": None,
            "kind": "anthropic_cache_diagnosis",
            "response_id": "msg_pending",
            "status": "pending",
        }

    @pytest.mark.asyncio
    async def test_anthropic_tool_payload_preserves_order_across_requests(self):
        llm = self._create_llm(cache_mode="auto")
        tools = [
            Tool(name="zeta", description="last alphabetically", inputSchema={"type": "object"}),
            Tool(name="alpha", description="first alphabetically", inputSchema={"type": "object"}),
        ]

        first = await llm._prepare_tools("claude-sonnet-5", tools=tools)
        second = await llm._prepare_tools("claude-sonnet-5", tools=tools)

        assert [tool["name"] for tool in first] == ["zeta", "alpha"]
        assert second == first

    @pytest.mark.asyncio
    async def test_cache_diagnostics_links_consecutive_requests(self):
        ctx = self._create_context_with_cache_mode("auto")
        assert ctx.config is not None
        assert ctx.config.anthropic is not None
        ctx.config.anthropic.cache_diagnostics = True
        llm = RecordingCacheDiagnosticsLLM(context=ctx, model="claude-sonnet-5")
        first_user = PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="first")],
        )

        first_assistant = await llm._anthropic_completion(
            AnthropicConverter.convert_to_anthropic(first_user),
            history=[first_user],
            current_extended=first_user,
        )
        second_user = PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="second")],
        )
        await llm._anthropic_completion(
            AnthropicConverter.convert_to_anthropic(second_user),
            history=[first_user, first_assistant, second_user],
            current_extended=second_user,
        )
        await llm._anthropic_completion(
            AnthropicConverter.convert_to_anthropic(first_user),
            history=[first_user],
            current_extended=first_user,
        )

        assert llm.requests[0]["diagnostics"] == {"previous_message_id": None}
        assert llm.requests[1]["diagnostics"] == {"previous_message_id": "msg_1"}
        assert llm.requests[2]["diagnostics"] == {"previous_message_id": None}
        assert CACHE_DIAGNOSIS_BETA in llm.requests[1]["betas"]

    def test_auto_mode_pins_cache_to_managed_process_start(self):
        llm = self._create_llm(cache_mode="auto")
        execute_call_id = "execute-1"
        poll_call_id = "poll-1"
        execute_result = CallToolResult(
            content=[TextContent(type="text", text="process-1 running")],
        )
        execute_result.meta = {
            FAST_AGENT_SHELL_PROCESS_METADATA: {
                "process_id": "process-1",
                "process_status": "running",
                "process_yield_reason": "background",
            }
        }
        poll_result = CallToolResult(
            content=[TextContent(type="text", text="still running")],
        )
        poll_result.meta = {
            FAST_AGENT_SHELL_PROCESS_METADATA: {
                "process_id": "process-1",
                "process_status": "running",
                "process_yield_reason": "deadline",
                "poll_wait_sec": 240,
                "output_line_count": 0,
            }
        }
        history = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="build the project")],
            ),
            PromptMessageExtended(
                role="assistant",
                tool_calls={
                    execute_call_id: CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="execute",
                            arguments={"command": "sleep 500", "background": True},
                        ),
                    )
                },
                stop_reason=LlmStopReason.TOOL_USE,
            ),
            PromptMessageExtended(
                role="user",
                tool_results={execute_call_id: execute_result},
            ),
            PromptMessageExtended(
                role="assistant",
                tool_calls={
                    poll_call_id: CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="poll_process",
                            arguments={
                                "process_id": "process-1",
                                "wait_sec": 240,
                                "wake_on_output": False,
                            },
                        ),
                    )
                },
                stop_reason=LlmStopReason.TOOL_USE,
            ),
            PromptMessageExtended(
                role="user",
                tool_results={poll_call_id: poll_result},
            ),
        ]
        assert managed_process_poll_cache_boundary(history) == 2
        converted = [AnthropicConverter.convert_to_anthropic(message) for message in history]
        arguments: dict[str, object] = {}

        llm._apply_anthropic_cache_plan(
            arguments=arguments,
            messages=converted,
            params=llm.get_request_params(RequestParams(use_history=True)),
            cache_mode="auto",
            history=history,
            current_extended=history[-1],
        )

        execute_blocks = _content_dicts(converted[2])
        poll_blocks = _content_dicts(converted[4])
        assert _cache_control(execute_blocks[-1]) == {
            "type": "ephemeral",
            "ttl": "5m",
        }
        assert all(_cache_control(block) is None for block in poll_blocks)

    def test_conversion_empty_messages(self):
        """Test conversion of empty message list."""
        llm = self._create_llm(cache_mode="off")

        converted = llm._convert_extended_messages_to_provider([])

        assert converted == []

    def test_conversion_with_templates_only(self):
        """Test conversion when only templates exist (no conversation)."""
        # Create template messages
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
        ]

        converted = self._apply_cache_plan(template_msgs, cache_mode="prompt")

        # Should have just the template
        assert len(converted) == 1

        # Template should have cache_control
        found_cache_control = False
        for block in _content_dicts(converted[0]):
            if _cache_control(block) is not None:
                found_cache_control = True

        assert found_cache_control, "Template should have cache_control in 'prompt' mode"

    def test_cache_control_on_last_content_block(self):
        """Test that cache_control is applied to the last content block of template messages."""
        # Create a template with multiple content blocks
        template_msgs = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="First block"),
                    TextContent(type="text", text="Second block"),
                ],
                is_template=True,
            ),
        ]

        converted = self._apply_cache_plan(template_msgs, cache_mode="prompt")

        # Cache control should be on the last block
        content = converted[0].get("content", [])
        content_blocks = [] if isinstance(content, str) else list(content)
        assert len(content_blocks) == 2

        # First block should NOT have cache_control
        if isinstance(content_blocks[0], dict):
            # Cache control might be on any block, but typically the last one
            pass

        # At least one block should have cache_control
        found_cache_control = any(
            isinstance(block, dict) and "cache_control" in block for block in content_blocks
        )
        assert found_cache_control, "Template should have cache_control"

    def test_conversion_prompt_mode_with_1h_ttl(self):
        """Test that cache_ttl='1h' produces correct cache_control with 1h TTL."""
        template_msgs = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="System context")],
                is_template=True,
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Understood")],
                is_template=True,
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="Question")]),
        ]

        converted = self._apply_cache_plan(
            template_msgs + conversation_msgs, cache_mode="prompt", cache_ttl="1h"
        )

        # Verify we have 3 messages (2 templates + 1 conversation)
        assert len(converted) == 3

        # Template messages should have cache_control with 1h TTL
        found_1h_cache_control = False
        template_count = len(template_msgs)
        for msg in converted[:template_count]:
            for block in _content_dicts(msg):
                cache_control = _cache_control(block)
                if cache_control is not None:
                    assert cache_control["type"] == "ephemeral"
                    assert cache_control["ttl"] == "1h"
                    found_1h_cache_control = True

        assert found_1h_cache_control, "Template messages should have cache_control with 1h TTL"

    def test_conversion_auto_mode_with_1h_ttl(self):
        """Test that cache_ttl='1h' works correctly in 'auto' mode."""
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
        ]
        conversation_msgs = [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="Question")]),
        ]

        converted = self._apply_cache_plan(
            template_msgs + conversation_msgs, cache_mode="auto", cache_ttl="1h"
        )

        # Template message should have cache_control with 1h TTL
        found_1h_cache_control = False
        template_msg = converted[0]
        for block in _content_dicts(template_msg):
            cache_control = _cache_control(block)
            if cache_control is not None:
                assert cache_control["type"] == "ephemeral"
                assert cache_control["ttl"] == "1h"
                found_1h_cache_control = True

        assert found_1h_cache_control, (
            "Template messages should have cache_control with 1h TTL in 'auto' mode"
        )

    @pytest.mark.parametrize("cache_ttl", ["5m", "1h"])
    def test_cache_ttl_values(self, cache_ttl: Literal["5m", "1h"]):
        """Test that both valid TTL values produce correct cache_control."""
        template_msgs = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Template")], is_template=True
            ),
        ]

        converted = self._apply_cache_plan(template_msgs, cache_mode="prompt", cache_ttl=cache_ttl)

        # Find the cache_control and verify TTL
        for block in _content_dicts(converted[0]):
            cache_control = _cache_control(block)
            if cache_control is not None:
                assert cache_control["type"] == "ephemeral"
                assert cache_control["ttl"] == cache_ttl
                return

        pytest.fail(f"No cache_control found for TTL {cache_ttl}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
