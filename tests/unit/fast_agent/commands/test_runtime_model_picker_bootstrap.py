from __future__ import annotations

import os

from fast_agent.cli.runtime.agent_setup import (
    _resolve_model_without_hardcoded_default,
    _should_prompt_for_model_picker,
)
from fast_agent.cli.runtime.run_request import AgentRunRequest


def _make_request(
    *,
    message: str | None = None,
    prompt_file: str | None = None,
    agent_cards: list[str] | None = None,
    card_tools: list[str] | None = None,
) -> AgentRunRequest:
    return AgentRunRequest(
        name="test",
        instruction="instruction",
        config_path=None,
        server_list=None,
        agent_cards=agent_cards,
        card_tools=card_tools,
        model=None,
        message=message,
        prompt_file=prompt_file,
        result_file=None,
        resume=None,
        url_servers=None,
        stdio_servers=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        noenv=False,
        force_smart=False,
        shell_runtime=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )


def test_should_prompt_for_model_picker_in_interactive_tty_startup() -> None:
    request = _make_request(message=None, prompt_file=None)

    assert _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


def test_should_not_prompt_for_model_picker_when_message_mode() -> None:
    request = _make_request(message="hello")

    assert not _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


def test_should_not_prompt_for_model_picker_when_cards_present() -> None:
    request = _make_request(agent_cards=["cards/"])

    assert not _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


def test_resolve_model_without_hardcoded_default_returns_none_without_sources() -> None:
    previous = os.environ.pop("FAST_AGENT_MODEL", None)
    try:
        model, source = _resolve_model_without_hardcoded_default(
            model=None,
            config_default_model=None,
            model_aliases=None,
        )
    finally:
        if previous is not None:
            os.environ["FAST_AGENT_MODEL"] = previous

    assert model is None
    assert source is None


def test_resolve_model_without_hardcoded_default_prefers_config_default() -> None:
    previous = os.environ.pop("FAST_AGENT_MODEL", None)
    try:
        model, source = _resolve_model_without_hardcoded_default(
            model=None,
            config_default_model="openai.gpt-4.1-mini",
            model_aliases=None,
        )
    finally:
        if previous is not None:
            os.environ["FAST_AGENT_MODEL"] = previous

    assert model == "openai.gpt-4.1-mini"
    assert source == "config file"


def test_resolve_model_without_hardcoded_default_uses_environment_variable() -> None:
    previous = os.environ.get("FAST_AGENT_MODEL")
    os.environ["FAST_AGENT_MODEL"] = "responses.gpt-5-mini"
    try:
        model, source = _resolve_model_without_hardcoded_default(
            model=None,
            config_default_model=None,
            model_aliases=None,
        )
    finally:
        if previous is not None:
            os.environ["FAST_AGENT_MODEL"] = previous
        else:
            os.environ.pop("FAST_AGENT_MODEL", None)

    assert model == "responses.gpt-5-mini"
    assert source == "environment variable FAST_AGENT_MODEL"
