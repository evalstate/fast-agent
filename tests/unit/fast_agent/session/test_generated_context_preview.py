from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from mcp_types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import TOOL_DEFINITIONS_META
from fast_agent.session import SessionManager

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol


class _Agent:
    def __init__(self, history: list[PromptMessageExtended]) -> None:
        self.name = "main"
        self.instruction = "prompt"
        self.config = AgentConfig(self.name, instruction=self.instruction, model="passthrough")
        self.llm = None
        self.usage_accumulator = None
        self.message_history = history


def _user_with_generated_context(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="user",
        content=[
            TextContent(type="text", text=text),
            TextContent(
                type="text",
                text="<fastagent:tool-definitions>def fetch(): ...</fastagent:tool-definitions>",
                meta={TOOL_DEFINITIONS_META: ["fetch"]},
            ),
            TextContent(
                type="text",
                text="<fastagent:tool-sources>server: docs</fastagent:tool-sources>",
                meta={TOOL_SOURCES_FINGERPRINT_META: "abc"},
            ),
        ],
    )


@pytest.mark.asyncio
async def test_saved_session_previews_exclude_generated_tool_context(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    agent = _Agent(
        [
            _user_with_generated_context("first question"),
            PromptMessageExtended(role="assistant", content=[TextContent(type="text", text="ok")]),
            _user_with_generated_context("latest question"),
        ]
    )

    await session.save_history(cast("AgentProtocol", agent))

    assert session.info.metadata["first_user_preview"] == "first question"
    assert session.info.metadata["last_user_prompt"] == "latest question"
    assert session.set_last_user_prompt(_user_with_generated_context("typed")) == "typed"
