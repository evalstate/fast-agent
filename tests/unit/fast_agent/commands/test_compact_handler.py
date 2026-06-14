from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.compact import handle_compact_prompt
from fast_agent.config import CompactionSettings, Settings

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandMessage


class _IO(NonInteractiveCommandIOBase):
    async def emit(self, message: CommandMessage) -> None:
        del message


@pytest.mark.asyncio
async def test_compact_prompt_emits_prompt_and_guidance_as_markdown() -> None:
    settings = Settings(compaction=CompactionSettings(prompt="## Custom compact prompt\n\nSummarize."))
    ctx = CommandContext(
        agent_provider=StaticAgentProvider(),
        current_agent_name="main",
        io=_IO(),
        settings=settings,
    )

    outcome = await handle_compact_prompt(ctx, agent_name="main")

    assert len(outcome.messages) == 2
    prompt_message = outcome.messages[0]
    assert prompt_message.plain_text() == "## Custom compact prompt\n\nSummarize."
    assert prompt_message.render_markdown is True
    assert prompt_message.title == "Compaction prompt (config (compaction.prompt))"

    guidance_message = outcome.messages[1]
    assert guidance_message.render_markdown is True
    assert guidance_message.plain_text().startswith("> Override with `compaction.prompt`")
    assert "`fastagent.config.yaml`" in guidance_message.plain_text()
