from __future__ import annotations

from fast_agent.ui.prompt import input as prompt_input


class _FakeBuffer:
    def __init__(self, text: str = "") -> None:
        self.text = text


class _FakeSession:
    def __init__(self, text: str = "") -> None:
        self.default_buffer = _FakeBuffer(text)


def test_build_prompt_text_resolver_omits_default_agent_name() -> None:
    session = _FakeSession()
    resolver = prompt_input._build_prompt_text_resolver(
        session_factory=lambda: session,
        agent_name="dev",
        default_agent_name="dev",
        show_default=False,
        default="",
        shell_enabled=False,
    )

    assert resolver().value == "❯ "


def test_build_prompt_text_resolver_shows_named_non_default_agent() -> None:
    session = _FakeSession()
    resolver = prompt_input._build_prompt_text_resolver(
        session_factory=lambda: session,
        agent_name="review",
        default_agent_name="dev",
        show_default=False,
        default="",
        shell_enabled=False,
    )

    assert resolver().value == "<ansibrightblue>review</ansibrightblue> ❯ "
