from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.commands.results import CommandMessage
from fast_agent.ui.adapters.tui_io import TuiCommandIO
from fast_agent.ui.message_primitives import MessageType
from fast_agent.ui.model_picker import ModelPickerResult

if TYPE_CHECKING:
    from rich.text import Text

    from fast_agent.commands.context import AgentProvider


class _FakeDisplay:
    def __init__(self) -> None:
        self.status_messages: list[Text] = []
        self.display_calls: list[dict[str, object]] = []

    def show_status_message(self, content: Text) -> None:
        self.status_messages.append(content)

    def display_message(self, **kwargs: object) -> None:
        self.display_calls.append(kwargs)


class _FakeAgent:
    def __init__(self, display: _FakeDisplay) -> None:
        self.display = display


class _FakeProvider:
    def __init__(self, display: _FakeDisplay) -> None:
        self._display = display

    def _agent(self, agent_name: str) -> object:  # noqa: ARG002
        return _FakeAgent(self._display)

    def agent_names(self) -> list[str]:
        return ["alpha"]

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None) -> object:  # noqa: ARG002
        return {}


@pytest.mark.asyncio
async def test_emit_render_markdown_uses_assistant_renderer() -> None:
    display = _FakeDisplay()
    provider = cast("AgentProvider", _FakeProvider(display))
    io = TuiCommandIO(prompt_provider=provider, agent_name="alpha")

    message = CommandMessage(
        text="## Summary\n\n- one\n- two",
        title="Last assistant message",
        right_info="session",
        agent_name="alpha",
        render_markdown=True,
    )
    await io.emit(message)

    assert len(display.status_messages) == 1
    assert display.status_messages[0].plain == "Last assistant message"

    assert len(display.display_calls) == 1
    display_call = display.display_calls[0]
    assert display_call["content"] == "## Summary\n\n- one\n- two"
    assert display_call["message_type"] == MessageType.ASSISTANT
    assert display_call["name"] == "alpha"
    assert display_call["right_info"] == "session"
    assert display_call["truncate_content"] is False
    assert display_call["render_markdown"] is True


@pytest.mark.asyncio
async def test_prompt_model_selection_normalizes_generic_custom_model(monkeypatch) -> None:
    display = _FakeDisplay()
    provider = cast("AgentProvider", _FakeProvider(display))
    io = TuiCommandIO(prompt_provider=provider, agent_name="alpha")

    picker_result = ModelPickerResult(
        provider="generic",
        provider_available=True,
        selected_model="generic.__custom__",
        resolved_model=None,
        source="curated",
        refer_to_docs=False,
        activation_action=None,
    )

    async def fake_run_model_picker_async(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return picker_result

    async def fake_prompt_text(prompt: str, *, default=None, allow_empty=True):  # type: ignore[no-untyped-def]
        del prompt, default, allow_empty
        return "llama3.2"

    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )
    monkeypatch.setattr(io, "prompt_text", fake_prompt_text)

    selected = await io.prompt_model_selection(initial_provider="generic")

    assert selected == "generic.llama3.2"


@pytest.mark.asyncio
async def test_prompt_model_selection_preserves_explicit_provider_prefix_for_generic_entry(
    monkeypatch,
) -> None:
    display = _FakeDisplay()
    provider = cast("AgentProvider", _FakeProvider(display))
    io = TuiCommandIO(prompt_provider=provider, agent_name="alpha")

    picker_result = ModelPickerResult(
        provider="generic",
        provider_available=True,
        selected_model="generic.__custom__",
        resolved_model=None,
        source="curated",
        refer_to_docs=False,
        activation_action=None,
    )

    async def fake_run_model_picker_async(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return picker_result

    async def fake_prompt_text(prompt: str, *, default=None, allow_empty=True):  # type: ignore[no-untyped-def]
        del prompt, default, allow_empty
        return "openai/gpt-4.1"

    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )
    monkeypatch.setattr(io, "prompt_text", fake_prompt_text)

    selected = await io.prompt_model_selection(initial_provider="generic")

    assert selected == "openai/gpt-4.1"
