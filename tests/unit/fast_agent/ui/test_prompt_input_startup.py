from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.agents.agent_types import AgentType
from fast_agent.ui.prompt import input as prompt_input

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.core.agent_app import AgentApp


class _HfDisplayInfoLLM:
    def get_hf_display_info(self) -> "Mapping[str, object]":
        return {
            "model": "org/model [draft]",
            "provider": "hf [router]",
        }


class _BrokenHfDisplayInfoLLM:
    def get_hf_display_info(self) -> "Mapping[str, object]":
        raise RuntimeError("hf display info failed")


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


def test_streaming_mode_notice_uses_normalized_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    provider = SimpleNamespace(
        registered_agent_types=lambda: {"agent": AgentType.BASIC},
    )
    logger_settings = SimpleNamespace(
        show_chat=True,
        streaming="markdown",
        streaming_plain_text=True,
        streaming_display=True,
    )

    monkeypatch.setattr(prompt_input, "rich_print", lambda text: printed.append(text))

    prompt_input._show_streaming_mode_notice(
        cast("AgentApp", provider),
        logger_settings,
    )

    assert printed == ["[dim]Streaming Enabled - plain mode[/dim]"]


def test_fast_agent_home_summary_escapes_source_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    config = SimpleNamespace(
        _fast_agent_home="/tmp/fast-agent-home",
        _fast_agent_home_source="config [local]",
        model_references={},
    )
    agent = SimpleNamespace(context=SimpleNamespace(config=config))
    provider = SimpleNamespace(
        registered_agents=lambda: {"main": agent},
        registered_agent_names=lambda: ["main"],
    )

    monkeypatch.setattr(prompt_input, "rich_print", printed.append)
    prompt_input._show_fast_agent_home_summary(cast("AgentApp", provider))

    assert printed
    assert r"via config \[local]" in printed[0]


def test_fast_agent_home_summary_ignores_agents_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    provider = SimpleNamespace(
        registered_agents=lambda: {"main": object()},
        registered_agent_names=lambda: ["main"],
    )

    monkeypatch.setattr(prompt_input, "rich_print", printed.append)
    prompt_input._show_fast_agent_home_summary(cast("AgentApp", provider))

    assert printed == []


def test_configured_hook_and_extension_counts_use_typed_shapes() -> None:
    provider = SimpleNamespace(
        plugin_commands={"global": object()},
        registered_agents=lambda: {
            "direct": SimpleNamespace(
                config=SimpleNamespace(
                    tool_hooks={"tool": object()},
                    lifecycle_hooks={"start": object(), "stop": object()},
                    commands={"local": object()},
                )
            ),
            "context": SimpleNamespace(
                context=SimpleNamespace(
                    config=SimpleNamespace(
                        tool_hooks=None,
                        lifecycle_hooks={"ready": object()},
                        commands={"ctx": object()},
                    )
                )
            ),
            "empty": object(),
        },
    )

    assert prompt_input._count_configured_hooks(cast("AgentApp", provider)) == 4
    assert prompt_input._count_configured_extensions(cast("AgentApp", provider)) == 3


@pytest.mark.asyncio
async def test_streaming_status_escapes_model_status_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    logger_settings = SimpleNamespace(show_chat=True, streaming=False)
    config = SimpleNamespace(logger=logger_settings, model_source="config [local]")
    context = SimpleNamespace(config=config)
    active_agent = SimpleNamespace(context=context, llm=_HfDisplayInfoLLM())
    provider = SimpleNamespace(
        _agent=lambda agent_name: active_agent,
        registered_agent_types=lambda: {"agent": AgentType.BASIC},
    )

    monkeypatch.setattr(prompt_input, "rich_print", lambda text: printed.append(text))

    await prompt_input._show_streaming_status(
        agent_name="agent",
        agent_provider=cast("AgentApp", provider),
        shell_agent=None,
    )

    assert r"[dim]Model selected via config \[local][/dim]" in printed
    assert r"[dim]HuggingFace: org/model \[draft] via hf \[router][/dim]" in printed


@pytest.mark.asyncio
async def test_streaming_status_ignores_non_provider_hf_display_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    logger_settings = SimpleNamespace(show_chat=True, streaming=False)
    config = SimpleNamespace(logger=logger_settings, model_source=None)
    context = SimpleNamespace(config=config)
    active_agent = SimpleNamespace(
        context=context,
        llm=SimpleNamespace(get_hf_display_info={"provider": "not callable"}),
    )
    provider = SimpleNamespace(
        _agent=lambda agent_name: active_agent,
        registered_agent_types=lambda: {"agent": AgentType.BASIC},
    )

    monkeypatch.setattr(prompt_input, "rich_print", lambda text: printed.append(text))

    await prompt_input._show_streaming_status(
        agent_name="agent",
        agent_provider=cast("AgentApp", provider),
        shell_agent=None,
    )

    assert not any("HuggingFace:" in line for line in printed)


@pytest.mark.asyncio
async def test_streaming_status_does_not_swallow_broken_hf_display_provider() -> None:
    logger_settings = SimpleNamespace(show_chat=True, streaming=False)
    config = SimpleNamespace(logger=logger_settings, model_source=None)
    context = SimpleNamespace(config=config)
    active_agent = SimpleNamespace(context=context, llm=_BrokenHfDisplayInfoLLM())
    provider = SimpleNamespace(
        _agent=lambda agent_name: active_agent,
        registered_agent_types=lambda: {"agent": AgentType.BASIC},
    )

    with pytest.raises(RuntimeError, match="hf display info failed"):
        await prompt_input._show_streaming_status(
            agent_name="agent",
            agent_provider=cast("AgentApp", provider),
            shell_agent=None,
        )


@pytest.mark.asyncio
async def test_show_mcp_status_load_error_prints_bracketed_text_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []

    class _Provider:
        def _agent(self, agent_name: str) -> object:
            raise RuntimeError(f"{agent_name} [missing]")

    monkeypatch.setattr(prompt_input, "rich_print", lambda text: printed.append(text))

    await prompt_input.show_mcp_status("[draft]", cast("AgentApp", _Provider()))

    assert printed
    assert getattr(printed[0], "plain", "") == "Unable to load agent '[draft]': [draft] [missing]"
