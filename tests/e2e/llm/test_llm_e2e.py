import base64
import os
from pathlib import Path
from typing import Annotated

import pytest
import pytest_asyncio
from mcp.types import (
    BlobResourceContents,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool,
)
from pydantic import BaseModel, Field

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core import Core
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.prompt import Prompt


class FormattedResponse(BaseModel):
    thinking: Annotated[
        str, Field(description="Your reflection on the conversation that is not seen by the user.")
    ]
    message: str


def get_test_models():
    """Get models to test from environment variables."""
    # Check for TEST_MODELS (multiple models)
    if os.environ.get("TEST_MODELS"):
        return os.environ.get("TEST_MODELS").split(",")
    # Check for TEST_MODEL (single model)
    elif os.environ.get("TEST_MODEL"):
        return [os.environ.get("TEST_MODEL")]
    # Default models
    else:
        return [
            "gpt-4.1-mini",
            "haiku",
            "kimi",
            "o4-mini.low",
            "gpt-5-mini.low",
            "gemini25",
            "gpt-oss",
            #     "generic.qwen3:8b",
        ]


# Create the list of models to test
TEST_MODELS = get_test_models()


@pytest_asyncio.fixture
async def llm_agent_setup(model_name):
    """Set up test environment with Core and agent."""
    test_config = AgentConfig("test")

    # Pass the config file path from the test directory
    config_path = os.path.join(os.path.dirname(__file__), "fastagent.config.yaml")

    # Initialize Core and agent
    core = Core(settings=config_path)
    await core.initialize()

    agent = LlmAgent(test_config, core.context)
    await agent.attach_llm(ModelFactory.create_factory(model_name))

    return agent


# Tool definition used by multiple tests
_input_schema = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "The city to check the weather for"}},
}
_tool = Tool(
    name="weather",
    description="call this to check the weather in a city",
    inputSchema=_input_schema,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_basic_generation(llm_agent_setup, model_name):
    """Test basic generation returns END_TURN stop reason."""
    agent = llm_agent_setup
    result: PromptMessageExtended = await agent.generate("hello, world")
    assert result.stop_reason is LlmStopReason.END_TURN
    assert result.last_text() is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_max_tokens_limit(llm_agent_setup, model_name):
    """Test generation with max tokens limit returns MAX_TOKENS stop reason."""
    agent = llm_agent_setup
    result: PromptMessageExtended = await agent.generate(
        "write a 300 word story", RequestParams(maxTokens=15)
    )
    assert result.stop_reason is LlmStopReason.MAX_TOKENS


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_stop_sequence(llm_agent_setup, model_name):
    """Test generation with stop sequence returns STOP_SEQUENCE stop reason."""
    agent = llm_agent_setup
    result: PromptMessageExtended = await agent.generate(
        "repeat after me, `one, two, three`.", RequestParams(stopSequences=[" two,"])
    )
    # oai reasoning models don't support this
    # we will also need to remove this for multimodal messages with oai
    if agent.llm.provider in [Provider.ANTHROPIC]:
        assert result.stop_reason is LlmStopReason.STOP_SEQUENCE
    else:
        assert result.stop_reason is LlmStopReason.END_TURN


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_structured_output(llm_agent_setup, model_name):
    """Test structured output generation with FormattedResponse model."""
    agent = llm_agent_setup
    structured_output, result = await agent.structured(
        "lets discuss the weather", FormattedResponse
    )
    assert structured_output
    assert LlmStopReason.END_TURN == result.stop_reason
    # consider whether we should retain the tool result in the message.
    # if Provider.ANTHROPIC == agent.llm.provider:
    #     assert result.tool_calls
    #     assert 1 == len(result.tool_calls)

    ## make sure the next turn works (anthropic needs to insert empty block)
    result = await agent.generate("what about tomorrow's weather?")
    assert result.stop_reason is LlmStopReason.END_TURN
    assert result.last_text() is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_tool_use_stop(llm_agent_setup, model_name):
    """Test tool use stop reason."""
    agent = llm_agent_setup
    result = await agent.generate("check the weather in london", tools=[_tool])
    assert LlmStopReason.TOOL_USE is result.stop_reason
    assert result.tool_calls
    assert 1 == len(result.tool_calls)
    tool_id = next(iter(result.tool_calls.keys()))
    tool_call: CallToolRequest = result.tool_calls[tool_id]
    assert "weather" == tool_call.params.name


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_tool_user_continuation(llm_agent_setup, model_name):
    """Generates a tool call, and returns a response. Ensures correlation works (converter handles results)"""
    agent = llm_agent_setup
    result = await agent.generate(
        "check the weather in new york",
        tools=[_tool],
        request_params=RequestParams(maxTokens=200),
    )
    assert LlmStopReason.TOOL_USE is result.stop_reason
    assert result.tool_calls
    assert 1 == len(result.tool_calls)
    tool_id = next(iter(result.tool_calls.keys()))

    result = CallToolResult(content=[TextContent(type="text", text="it's sunny in new york")])
    tool_results = {tool_id: result}
    result_message = PromptMessageExtended(role="user", tool_results=tool_results)
    result = await agent.generate(result_message)
    assert LlmStopReason.END_TURN is result.stop_reason
    assert "sunny" in result.last_text().lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_tool_calling_agent(llm_agent_setup, model_name):
    """Generates a tool call, and returns a response. Ensures correlation works (converter handles results)"""
    agent = llm_agent_setup
    result = await agent.generate(
        "check the weather in new york",
        tools=[_tool],
        request_params=RequestParams(maxTokens=100),
    )
    assert LlmStopReason.TOOL_USE is result.stop_reason
    assert result.tool_calls
    assert 1 == len(result.tool_calls)
    tool_id = next(iter(result.tool_calls.keys()))

    result = CallToolResult(content=[TextContent(type="text", text="it's sunny in new york")])
    tool_results = {tool_id: result}
    result_message = PromptMessageExtended(role="user", tool_results=tool_results)
    result = await agent.generate(result_message)
    assert LlmStopReason.END_TURN is result.stop_reason
    assert "sunny" in result.last_text().lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_vision_model_reads_name(llm_agent_setup, model_name):
    """Attach an image to the user message and verify the model can read the name in it."""
    agent = llm_agent_setup

    # Determine resolved model and check image support
    resolved_model = agent.llm.default_request_params.model
    if not ModelDatabase.supports_mime(resolved_model, "image/png"):
        pytest.skip(f"Model '{resolved_model}' does not support image/png")

    # Use the shared sample image from the multimodal tests directory
    image_path = (Path(__file__).parent.parent / "multimodal" / "image.png").resolve()
    assert image_path.exists(), f"Test image not found at {image_path}"

    # Build a user message with text + image
    user_msg = Prompt.user(
        "what is the user name contained in this image?",
        image_path,
    )

    result: PromptMessageExtended = await agent.generate(user_msg)
    assert result.stop_reason is LlmStopReason.END_TURN
    all_text = (result.all_text() or "").lower()
    assert "evalstate" in all_text or (
        result.last_text() and "evalstate" in result.last_text().lower()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_pdf_prompt_summarizes_name(llm_agent_setup, model_name):
    """Attach a PDF and verify the model includes product/company name in summary."""
    agent = llm_agent_setup

    resolved_model = agent.llm.default_request_params.model
    if not ModelDatabase.supports_mime(resolved_model, "application/pdf"):
        pytest.skip(f"Model '{resolved_model}' does not support application/pdf")

    pdf_path = (Path(__file__).parent.parent / "multimodal" / "sample.pdf").resolve()
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"

    user_msg = Prompt.user(
        "Summarize this document and include the product or company name.",
        pdf_path,
    )

    result: PromptMessageExtended = await agent.generate(user_msg)
    assert result.stop_reason is LlmStopReason.END_TURN
    text = (result.all_text() or "").lower()
    assert ("fast-agent" in text) or ("llmindset" in text)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_mcp_tool_result_image_reads_name(llm_agent_setup, model_name):
    """Simulate an MCP tool result delivering text+image content and verify the model reads the name."""
    agent = llm_agent_setup

    resolved_model = agent.llm.default_request_params.model
    if not ModelDatabase.supports_mime(resolved_model, "image/png"):
        pytest.skip(f"Model '{resolved_model}' does not support image/png")

    # Prepare image content (load shared asset)
    image_path = (Path(__file__).parent.parent / "multimodal" / "image.png").resolve()
    assert image_path.exists(), f"Test image not found at {image_path}"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    # Build conversation:
    # 1) User asks to call the tool and analyze result
    first_user = Prompt.user(
        "Use the get_image tool, then tell me the user name in the returned image."
    )

    # 2) Assistant issues the tool call
    tool_id = "tool_1"
    assistant_tool_call = PromptMessageExtended(
        role="assistant",
        tool_calls={
            tool_id: CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="get_image", arguments={}),
            ),
        },
    )

    tool_result = CallToolResult(
        content=[
            TextContent(type="text", text="Here's your image:"),
            ImageContent(type="image", data=image_b64, mimeType="image/png"),
        ]
    )

    # 3) User provides the tool result
    user_with_results = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="Here is the tool result")],
        tool_results={tool_id: tool_result},
    )

    result: PromptMessageExtended = await agent.generate(
        [first_user, assistant_tool_call, user_with_results]
    )
    assert result.stop_reason is LlmStopReason.END_TURN
    assert "evalstate" in (result.all_text() or "").lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_mcp_tool_result_pdf_summarizes_name(llm_agent_setup, model_name):
    """Simulate an MCP tool result delivering a PDF resource and verify the model includes product/company name."""
    agent = llm_agent_setup

    resolved_model = agent.llm.default_request_params.model
    if not ModelDatabase.supports_mime(resolved_model, "application/pdf"):
        pytest.skip(f"Model '{resolved_model}' does not support application/pdf")

    pdf_path = (Path(__file__).parent.parent / "multimodal" / "sample.pdf").resolve()
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")

    # Build conversation:
    # 1) User asks to call the tool and summarize the PDF
    first_user = Prompt.user(
        "Use the get_pdf tool, then summarize the document and include the product/company name."
    )

    # 2) Assistant issues the tool call
    tool_id = "tool_pdf_1"
    assistant_tool_call = PromptMessageExtended(
        role="assistant",
        tool_calls={
            tool_id: CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="get_pdf", arguments={}),
            ),
        },
    )

    embedded_pdf = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=f"file://{pdf_path}", blob=pdf_b64, mimeType="application/pdf"
        ),
    )
    tool_result = CallToolResult(
        content=[
            TextContent(type="text", text="Here is the PDF"),
            embedded_pdf,
        ]
    )

    # 3) User provides the tool result
    user_with_results = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="Here is the tool result")],
        tool_results={tool_id: tool_result},
    )

    result: PromptMessageExtended = await agent.generate(
        [first_user, assistant_tool_call, user_with_results]
    )
    assert result.stop_reason is LlmStopReason.END_TURN
    text = (result.all_text() or "").lower()
    assert ("fast-agent" in text) or ("llmindset" in text)
