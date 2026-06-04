from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.ui.prompt import input as prompt_input

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


@pytest.mark.asyncio
async def test_input_startup_shows_home_summary_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    provider = object()

    monkeypatch.setattr(prompt_input, "help_message_shown", False)
    monkeypatch.setattr(prompt_input, "rich_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_input, "_show_model_shortcut_hints", lambda **kwargs: None)
    monkeypatch.setattr(
        prompt_input,
        "_show_fast_agent_home_summary",
        lambda agent_provider: calls.append(agent_provider),
    )

    await prompt_input._show_input_startup(
        agent_name="agent",
        default="",
        show_stop_hint=False,
        is_human_input=False,
        shell_context=prompt_input.ShellInputContext(enabled=False),
        shell_agent=None,
        agent_provider=cast("AgentApp", provider),
        supports_clipboard_image_paste=False,
    )

    assert calls == [provider]
