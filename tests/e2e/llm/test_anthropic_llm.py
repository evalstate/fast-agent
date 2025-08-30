import os
import unittest
from typing import Annotated

from mcp.types import CallToolRequest, CallToolResult, TextContent, Tool
from pydantic import BaseModel, Field

from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core import Core
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.model_factory import ModelFactory
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class FormattedResponse(BaseModel):
    thinking: Annotated[
        str, Field(description="Your reflection on the conversation that is not seen by the user.")
    ]
    message: str


class TestAnthropicLLM(unittest.IsolatedAsyncioTestCase):
    """Test cases for Anthropic LLM functionality."""

    _input_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city to check the weather for"}
        },
    }
    _tool = Tool(
        name="weather",
        description="call this to check the weather in a city",
        inputSchema=_input_schema,
    )

    async def asyncSetUp(self):
        """Set up test environment with Core and agent."""
        self.test_config = AgentConfig("test")

        # Pass the config file path from the test directory
        config_path = os.path.join(os.path.dirname(__file__), "fastagent.config.yaml")

        # Initialize Core and agent
        self.core = Core(settings=config_path)
        await self.core.initialize()

        self.agent: LlmAgent = LlmAgent(self.test_config, self.core.context)
        await self.agent.attach_llm(ModelFactory.create_factory("gpt-4.1-mini"))

    async def test_basic_generation(self):
        """Test basic generation returns END_TURN stop reason."""
        result: PromptMessageMultipart = await self.agent.generate("hello, world")
        assert result.stop_reason is LlmStopReason.END_TURN
        assert None is not result.last_text()

    async def test_max_tokens_limit(self):
        """Test generation with max tokens limit returns MAX_TOKENS stop reason."""
        result: PromptMessageMultipart = await self.agent.generate(
            "write a 300 word story", RequestParams(maxTokens=15)
        )
        assert result.stop_reason is LlmStopReason.MAX_TOKENS

    async def test_stop_sequence(self):
        """Test generation with max tokens limit returns MAX_TOKENS stop reason."""
        result: PromptMessageMultipart = await self.agent.generate(
            "repeat after me, `one, two, three`.", RequestParams(stopSequences=[" two,"])
        )
        assert result.stop_reason is LlmStopReason.STOP_SEQUENCE

    async def test_structured_output(self):
        """Test structured output generation with FormattedResponse model."""
        structured_output, result = await self.agent.structured(
            "lets discuss the weather", FormattedResponse
        )
        assert structured_output
        assert LlmStopReason.END_TURN == result.stop_reason
        # consider whether we should retain the tool result in the message.
        # if Provider.ANTHROPIC == self.agent.llm.provider:
        #     assert result.tool_calls
        #     assert 1 == len(result.tool_calls)

        ## make sure the next turn works (anthropic needs to insert empty block)
        result = await self.agent.generate("what about tomorrow's weather?")
        assert result.stop_reason is LlmStopReason.END_TURN
        assert None is not result.last_text()

    async def test_tool_use_stop(self) -> None:
        result = await self.agent.generate("check the weather in london", tools=[self._tool])
        assert LlmStopReason.TOOL_USE is result.stop_reason
        assert result.tool_calls
        assert 1 == len(result.tool_calls)
        tool_id = next(iter(result.tool_calls.keys()))
        tool_call: CallToolRequest = result.tool_calls[tool_id]
        assert "weather" == tool_call.params.name

    async def test_tool_user_continuation(self) -> None:
        """Generates a tool call, and returns a response. Ensures correlation works (converter handles results)"""
        result = await self.agent.generate(
            "check the weather in new york",
            tools=[self._tool],
            request_params=RequestParams(maxTokens=100),
        )
        assert LlmStopReason.TOOL_USE is result.stop_reason
        assert result.tool_calls
        assert 1 == len(result.tool_calls)
        tool_id = next(iter(result.tool_calls.keys()))

        result = CallToolResult(content=[TextContent(type="text", text="it's sunny in new york")])
        tool_results = {tool_id: result}
        result_message = PromptMessageMultipart(role="user", tool_results=tool_results)
        result = await self.agent.generate(result_message)
        assert LlmStopReason.END_TURN is result.stop_reason
        assert "sunny" in result.last_text().lower()

    async def test_tool_calling_agent(self) -> None:
        """Generates a tool call, and returns a response. Ensures correlation works (converter handles results)"""
        result = await self.agent.generate(
            "check the weather in new york",
            tools=[self._tool],
            request_params=RequestParams(maxTokens=100),
        )
        assert LlmStopReason.TOOL_USE is result.stop_reason
        assert result.tool_calls
        assert 1 == len(result.tool_calls)
        tool_id = next(iter(result.tool_calls.keys()))

        result = CallToolResult(content=[TextContent(type="text", text="it's sunny in new york")])
        tool_results = {tool_id: result}
        result_message = PromptMessageMultipart(role="user", tool_results=tool_results)
        result = await self.agent.generate(result_message)
        assert LlmStopReason.END_TURN is result.stop_reason
        assert "sunny" in result.last_text().lower()
