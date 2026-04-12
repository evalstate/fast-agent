from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.commands.results import CommandMessage
from fast_agent.config import Settings
from fast_agent.llm.model_reference_diagnostics import ModelReferenceSetupItem
from fast_agent.ui.adapters.tui_io import TuiCommandIO
from fast_agent.ui.message_primitives import MessageType
from fast_agent.ui.model_picker import ModelPickerResult
from fast_agent.ui.model_picker_common import GENERIC_CUSTOM_MODEL_SENTINEL

if TYPE_CHECKING:
    from pathlib import Path

    from rich.text import Text

    from fast_agent.commands.context import AgentProvider


class _FakeDisplay:
    def __init__(self) -> None:
        self.status_messages: list[Text] = []
        self.display_calls: list[dict[str, object]] = []
        self.system_calls: list[dict[str, object]] = []

    @property
    def markup_enabled(self) -> bool:
        return True

    def show_status_message(self, content: Text) -> None:
        self.status_messages.append(content)

    def display_message(self, **kwargs: object) -> None:
        self.display_calls.append(kwargs)

    def show_system_message(
        self,
        system_prompt: str,
        *,
        agent_name: str,
        server_count: int = 0,
    ) -> None:
        self.system_calls.append(
            {
                "system_prompt": system_prompt,
                "agent_name": agent_name,
                "server_count": server_count,
            }
        )


class _FakeAgent:
    def __init__(self, display: _FakeDisplay) -> None:
        self.display = display


class _FakeProvider:
    def __init__(self, display: _FakeDisplay) -> None:
        self._display = display

    def _agent(self, agent_name: str) -> object:  # noqa: ARG002
        return _FakeAgent(self._display)

    def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
        del force_include
        return ["alpha"]

    def registered_agent_names(self) -> list[str]:
        return ["alpha"]

    def registered_agents(self) -> dict[str, object]:
        return {"alpha": _FakeAgent(self._display)}

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return agent_name or "alpha"

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
        selected_model=GENERIC_CUSTOM_MODEL_SENTINEL,
        resolved_model=None,
        source="curated",
        refer_to_docs=False,
        activation_action=None,
    )

    async def fake_run_model_picker_async(**kwargs):
        del kwargs
        return picker_result

    async def fake_prompt_text(prompt: str, *, default=None, allow_empty=True):
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
        selected_model=GENERIC_CUSTOM_MODEL_SENTINEL,
        resolved_model=None,
        source="curated",
        refer_to_docs=False,
        activation_action=None,
    )

    async def fake_run_model_picker_async(**kwargs):
        del kwargs
        return picker_result

    async def fake_prompt_text(prompt: str, *, default=None, allow_empty=True):
        del prompt, default, allow_empty
        return "openai/gpt-4.1"

    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )
    monkeypatch.setattr(io, "prompt_text", fake_prompt_text)

    selected = await io.prompt_model_selection(initial_provider="generic")

    assert selected == "openai/gpt-4.1"


@pytest.mark.asyncio
async def test_prompt_model_selection_preserves_overlay_token_when_resolved_model_is_present(
    monkeypatch,
) -> None:
    display = _FakeDisplay()
    provider = cast("AgentProvider", _FakeProvider(display))
    io = TuiCommandIO(prompt_provider=provider, agent_name="alpha")

    picker_result = ModelPickerResult(
        provider="overlays",
        provider_available=True,
        selected_model="haikutiny",
        resolved_model="haikutiny",
        source="curated",
        refer_to_docs=False,
        activation_action=None,
    )

    async def fake_run_model_picker_async(**kwargs):
        del kwargs
        return picker_result

    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )

    selected = await io.prompt_model_selection(initial_provider="overlays")

    assert selected == "haikutiny"


@pytest.mark.asyncio
async def test_prompt_model_selection_passes_resolved_start_path(monkeypatch, tmp_path: Path) -> None:
    display = _FakeDisplay()
    provider = cast("AgentProvider", _FakeProvider(display))
    project_root = tmp_path / "project"
    settings = Settings(environment_dir=str(project_root / ".fast-agent"))
    io = TuiCommandIO(prompt_provider=provider, agent_name="alpha", settings=settings)
    captured_kwargs: dict[str, object] = {}

    async def fake_run_model_picker_async(**kwargs):
        captured_kwargs.update(kwargs)
        return ModelPickerResult(
            provider="overlays",
            provider_available=True,
            selected_model="haikutiny",
            resolved_model="haikutiny",
            source="curated",
            refer_to_docs=False,
            activation_action=None,
        )

    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )

    selected = await io.prompt_model_selection(initial_provider="overlays")

    assert selected == "haikutiny"
    assert captured_kwargs["start_path"] == project_root


@pytest.mark.asyncio
async def test_pick_model_reference_token_returns_picker_token(monkeypatch) -> None:
    display = _FakeDisplay()
    provider = cast("AgentProvider", _FakeProvider(display))
    io = TuiCommandIO(prompt_provider=provider, agent_name="alpha")

    async def fake_run_model_reference_picker_async(items):
        del items
        return SimpleNamespace(action="select", token="$system.fast")

    monkeypatch.setattr(
        "fast_agent.ui.adapters.tui_io.run_model_reference_picker_async",
        fake_run_model_reference_picker_async,
    )

    selected = await io.pick_model_reference_token(
        items=(
            ModelReferenceSetupItem(
                token="$system.fast",
                priority="recommended",
                status="missing",
                current_value=None,
                summary="Fast alias",
                references=("main",),
            ),
        )
    )

    assert selected == "$system.fast"


@pytest.mark.asyncio
async def test_pick_model_reference_token_normalizes_custom_entry(monkeypatch) -> None:
    display = _FakeDisplay()
    provider = cast("AgentProvider", _FakeProvider(display))
    io = TuiCommandIO(prompt_provider=provider, agent_name="alpha")

    async def fake_run_model_reference_picker_async(items):
        del items
        return SimpleNamespace(action="custom", token=None)

    async def fake_prompt_text(prompt: str, *, default=None, allow_empty=True):
        del prompt, default, allow_empty
        return "system.fast"

    monkeypatch.setattr(
        "fast_agent.ui.adapters.tui_io.run_model_reference_picker_async",
        fake_run_model_reference_picker_async,
    )
    monkeypatch.setattr(io, "prompt_text", fake_prompt_text)

    selected = await io.pick_model_reference_token(
        items=(
            ModelReferenceSetupItem(
                token="$system.fast",
                priority="recommended",
                status="missing",
                current_value=None,
                summary="Fast alias",
                references=("main",),
            ),
        )
    )

    assert selected == "$system.fast"


@pytest.mark.asyncio
async def test_display_system_prompt_uses_display_protocol() -> None:
    display = _FakeDisplay()
    provider = cast("AgentProvider", _FakeProvider(display))
    io = TuiCommandIO(prompt_provider=provider, agent_name="alpha")

    await io.display_system_prompt(
        "alpha",
        "You are concise.",
        server_count=2,
    )

    assert display.system_calls == [
        {
            "system_prompt": "You are concise.",
            "agent_name": "alpha",
            "server_count": 2,
        }
    ]
