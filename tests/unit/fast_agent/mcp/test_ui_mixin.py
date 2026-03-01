"""Tests for the MCP UI Mixin."""


import pytest
from mcp.types import CallToolResult, EmbeddedResource, TextContent, TextResourceContents
from pydantic import AnyUrl
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.constants import MCP_UI
from fast_agent.mcp.ui_mixin import McpUIMixin
from fast_agent.types import PromptMessageExtended


class StubDisplay:
    """Stub display class that records calls."""

    def __init__(self):
        self.show_mcp_ui_links_called = False
        self.show_mcp_ui_links_args = None

    async def show_mcp_ui_links(self, links):
        """Stub method that records the call."""
        self.show_mcp_ui_links_called = True
        self.show_mcp_ui_links_args = links


class StubAgent:
    """Stub agent base class for testing the mixin."""

    def __init__(self, config, context=None, ui_mode="auto", **kwargs):
        self.config = config
        self.context = context
        self._ui_mode = ui_mode
        self.display = StubDisplay()
        self.message_history = []

    async def run_tools(self, request, request_params=None):
        """Stub implementation that returns the request unchanged."""
        return request

    async def show_assistant_message(
        self,
        message,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
        render_message: bool = True,
    ) -> None:
        """Stub implementation with correct signature."""
        _ = render_message
        pass


class UIAgentForTesting(McpUIMixin, StubAgent):
    """Test agent that combines the UI mixin with a stub base agent."""

    pass


@pytest.fixture
def mock_config():
    """Create a mock agent config."""
    return AgentConfig(name="test_agent", instruction="Test agent")


@pytest.fixture
def mock_context():
    """Create a mock context."""

    class MockContext:
        def __init__(self):
            class MockConfig:
                mcp_ui_mode = "auto"

            self.config = MockConfig()

    return MockContext()


@pytest.fixture
def ui_agent(mock_config, mock_context):
    """Create a UI-enabled agent for testing."""
    agent = UIAgentForTesting(config=mock_config, context=mock_context, ui_mode="auto")
    return agent


def create_ui_resource(uri: str = "ui://test/component", text: str = "<html>Test UI</html>"):
    """Helper to create a UI embedded resource."""
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(uri=AnyUrl(uri), mimeType="text/html", text=text),
    )


def create_non_ui_resource(text="Regular content"):
    """Helper to create a non-UI content block."""
    return TextContent(type="text", text=text)


@pytest.mark.asyncio
async def test_ui_mixin_extracts_ui_resources(ui_agent):
    """Test that the mixin correctly extracts UI resources from tool results."""
    # Create tool results with mixed content
    ui_resource = create_ui_resource()
    text_block = create_non_ui_resource()

    tool_results = {"tool1": CallToolResult(content=[ui_resource, text_block], isError=False)}

    # Create a request with tool results
    request = PromptMessageExtended(
        role="user", content=[TextContent(type="text", text="test")], tool_results=tool_results
    )

    # Run tools through the mixin
    result = await ui_agent.run_tools(request)

    # Check that UI resources were extracted and added to channels
    assert MCP_UI in result.channels
    assert len(result.channels[MCP_UI]) == 1
    assert result.channels[MCP_UI][0] == ui_resource

    # Check that tool results were cleaned
    assert len(result.tool_results["tool1"].content) == 1
    assert result.tool_results["tool1"].content[0] == text_block


@pytest.mark.asyncio
async def test_ui_mixin_respects_disabled_mode(ui_agent):
    """Test that UI extraction is skipped when mode is disabled."""
    ui_agent.set_ui_mode("disabled")

    # Create tool results with UI content
    ui_resource = create_ui_resource()
    tool_results = {"tool1": CallToolResult(content=[ui_resource], isError=False)}

    request = PromptMessageExtended(
        role="user", content=[TextContent(type="text", text="test")], tool_results=tool_results
    )

    # Run tools through the mixin
    result = await ui_agent.run_tools(request)

    # Check that nothing was extracted (disabled mode)
    assert result.channels is None or MCP_UI not in result.channels
    assert result.tool_results == tool_results


@pytest.mark.asyncio
async def test_ui_mixin_auto_mode_only_acts_with_ui_content(ui_agent):
    """Test that auto mode only processes when UI content is present."""
    ui_agent.set_ui_mode("auto")

    # Test with no UI content
    text_block = create_non_ui_resource()
    tool_results = {"tool1": CallToolResult(content=[text_block], isError=False)}

    request = PromptMessageExtended(
        role="user", content=[TextContent(type="text", text="test")], tool_results=tool_results
    )

    result = await ui_agent.run_tools(request)

    # No UI resources, so channels should not be modified
    assert result.channels is None or MCP_UI not in result.channels
    assert result.tool_results == tool_results


@pytest.mark.asyncio
async def test_ui_mixin_preserves_error_status(ui_agent):
    """Test that error status is preserved when extracting UI resources."""
    ui_resource = create_ui_resource()

    tool_results = {
        "tool1": CallToolResult(
            content=[ui_resource],
            isError=True,  # Error result
        )
    }

    request = PromptMessageExtended(
        role="user", content=[TextContent(type="text", text="test")], tool_results=tool_results
    )

    result = await ui_agent.run_tools(request)

    # Check that error status is preserved
    assert result.tool_results["tool1"].isError is True


@pytest.mark.asyncio
async def test_ui_mixin_enabled_mode_processes_all_content(ui_agent):
    """Test that enabled mode processes content even without UI resources."""
    ui_agent.set_ui_mode("enabled")

    # Test with only regular content
    text_block = create_non_ui_resource()
    tool_results = {"tool1": CallToolResult(content=[text_block], isError=False)}

    request = PromptMessageExtended(
        role="user", content=[TextContent(type="text", text="test")], tool_results=tool_results
    )

    result = await ui_agent.run_tools(request)

    # In enabled mode, channels should be set up even without UI content
    assert result.channels is not None
    assert MCP_UI in result.channels
    assert len(result.channels[MCP_UI]) == 0  # Empty but present


@pytest.mark.asyncio
async def test_show_assistant_message_displays_ui_resources(ui_agent):
    """Test that show_assistant_message triggers UI resource display."""
    # Set up message history with UI resources
    user_msg = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="test")],
        channels={MCP_UI: [create_ui_resource()]},
    )
    assistant_msg = PromptMessageExtended(
        role="assistant", content=[TextContent(type="text", text="response")]
    )

    # Set the message history
    ui_agent.message_history = [user_msg, assistant_msg]

    # Stub the UI utility functions
    import fast_agent.mcp.ui_mixin as ui_mixin_module

    original_ui_links_from_channel = ui_mixin_module.ui_links_from_channel
    original_open_links_in_browser = ui_mixin_module.open_links_in_browser

    try:

        def stub_ui_links(resources):
            return [{"title": "Test UI", "url": "ui://test"}] if resources else []

        def stub_open_browser(links, **kwargs):
            pass

        setattr(ui_mixin_module, "ui_links_from_channel", stub_ui_links)
        setattr(ui_mixin_module, "open_links_in_browser", stub_open_browser)

        await ui_agent.show_assistant_message(assistant_msg)

        # Check that UI links were displayed using our stub
        assert ui_agent.display.show_mcp_ui_links_called
        assert ui_agent.display.show_mcp_ui_links_args is not None

    finally:
        # Restore original functions
        setattr(ui_mixin_module, "ui_links_from_channel", original_ui_links_from_channel)
        setattr(ui_mixin_module, "open_links_in_browser", original_open_links_in_browser)


def test_is_ui_embedded_resource(ui_agent):
    """Test the UI resource identification logic."""
    # Test UI resource
    ui_resource = create_ui_resource("ui://test/component")
    assert ui_agent._is_ui_embedded_resource(ui_resource) is True

    # Test non-UI resource (different scheme)
    non_ui = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("http://example.com"), mimeType="text/html", text="content"
        ),
    )
    assert ui_agent._is_ui_embedded_resource(non_ui) is False

    # Test text content
    text_content = TextContent(type="text", text="plain text")
    assert ui_agent._is_ui_embedded_resource(text_content) is False


def test_set_ui_mode(ui_agent):
    """Test that UI mode can be set and invalid modes default to auto."""
    # Test valid modes
    ui_agent.set_ui_mode("disabled")
    assert ui_agent._ui_mode == "disabled"

    ui_agent.set_ui_mode("enabled")
    assert ui_agent._ui_mode == "enabled"

    ui_agent.set_ui_mode("auto")
    assert ui_agent._ui_mode == "auto"

    # Test invalid mode defaults to auto
    ui_agent.set_ui_mode("invalid")
    assert ui_agent._ui_mode == "auto"


@pytest.mark.asyncio
async def test_split_ui_blocks(ui_agent):
    """Test the internal UI block splitting logic."""
    ui_resource = create_ui_resource()
    text_block = create_non_ui_resource()

    blocks = [ui_resource, text_block, create_ui_resource("ui://another")]

    ui_blocks, other_blocks = ui_agent._split_ui_blocks(blocks)

    assert len(ui_blocks) == 2
    assert len(other_blocks) == 1
    assert ui_resource in ui_blocks
    assert text_block in other_blocks


@pytest.mark.asyncio
async def test_extract_ui_from_tool_results_handles_empty_results(ui_agent):
    """Test that empty tool results are handled gracefully."""
    # Test with None
    result, ui_blocks = ui_agent._extract_ui_from_tool_results(None)
    assert result is None
    assert ui_blocks == []

    # Test with empty dict
    result, ui_blocks = ui_agent._extract_ui_from_tool_results({})
    assert result == {}
    assert ui_blocks == []


@pytest.mark.asyncio
async def test_extract_ui_from_tool_results_handles_exceptions(ui_agent):
    """Test that malformed tool results don't crash the extraction."""

    # Create a malformed result that might cause exceptions
    class BrokenResult:
        def __init__(self):
            self.isError = False
            self._content = None

        @property
        def content(self):
            raise Exception("Broken content property")

    tool_results = {"broken": BrokenResult()}

    # Should not raise an exception
    result, ui_blocks = ui_agent._extract_ui_from_tool_results(tool_results)

    # Should pass through the broken result untouched
    assert "broken" in result
    assert result["broken"] == tool_results["broken"]
    assert ui_blocks == []
