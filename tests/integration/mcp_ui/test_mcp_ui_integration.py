import os
from pathlib import Path

import pytest
import pytest_asyncio
from mcp.types import EmbeddedResource, TextContent, TextResourceContents
from pydantic import AnyUrl

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.constants import MCP_UI
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.ui_agent import McpAgentWithUI
from fast_agent.types import PromptMessageExtended


@pytest_asyncio.fixture
async def passthrough_agent(tmp_path):
    # Use a temp working directory per test to isolate .fastagent outputs
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        core = Core()
        await core.initialize()
        # Avoid auto-opening browser windows during tests
        core.context.config.mcp_ui_mode = "enabled"

        agent = McpAgentWithUI(
            AgentConfig("mcp-ui-test", model="passthrough", servers=[]),
            core.context,
            ui_mode="enabled",
        )
        await agent.attach_llm(ModelFactory.create_factory("passthrough"))

        yield agent
    finally:
        # Cleanup and restore cwd
        await core.cleanup()
        os.chdir(cwd)


@pytest.mark.asyncio
async def test_mcp_ui_html_write_and_link_display(passthrough_agent):
    agent = passthrough_agent

    # Prepare a UI EmbeddedResource with text/html content
    ui_res = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("ui://my-component/instance-1"),
            mimeType="text/html",
            text="<p>Hello World</p>",
        ),
    )

    # Compose a user message that carries mcp-ui channel
    user_msg = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="Please show UI")],
        channels={MCP_UI: [ui_res]},
    )

    # Generate assistant response (passthrough provider)
    _ = await agent.generate([user_msg])

    # Verify that a local HTML file was created under .fastagent/ui
    out_dir = Path.cwd() / ".fastagent" / "ui"
    assert out_dir.exists(), "UI output directory not created"
    html_files = list(out_dir.glob("*.html"))
    assert html_files, "No HTML files created for MCP-UI"

    # Read the last created file and ensure it contains the HTML snippet
    latest = max(html_files, key=lambda p: p.stat().st_mtime)
    content = latest.read_text(encoding="utf-8")
    assert "Hello World" in content
