from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.types import EmbeddedResource, ReadResourceResult, TextResourceContents
from pydantic import AnyUrl

from fast_agent.agents.agent_types import AgentType
from fast_agent.types import PromptMessageExtended
from fast_agent.ui import interactive_prompt
from fast_agent.ui.interactive_prompt import InteractivePrompt

if TYPE_CHECKING:
    from pathlib import Path


class _MentionAgent:
    def __init__(self) -> None:
        self.message_history = []

    async def get_resource(self, resource_uri: str, namespace: str | None = None):
        assert namespace == "demo"
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=AnyUrl(resource_uri),
                    mimeType="text/plain",
                    text="payload",
                )
            ]
        )


class _LocalMentionAgent:
    def __init__(self) -> None:
        self.message_history = []


class _MentionAgentApp:
    def __init__(self) -> None:
        self._agent_obj = _MentionAgent()

    async def refresh_if_needed(self) -> bool:
        return False

    def _agent(self, _name: str) -> _MentionAgent:
        return self._agent_obj

    def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
        del force_include
        return ["agent1"]

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
        del force_include
        return {"agent1": AgentType.BASIC}

    def registered_agent_names(self) -> list[str]:
        return ["agent1"]

    def registered_agents(self) -> dict[str, _MentionAgent]:
        return {"agent1": self._agent_obj}

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return agent_name or "agent1"


@pytest.mark.asyncio
async def test_prompt_loop_materializes_resource_mentions(monkeypatch) -> None:
    inputs = iter(["Summarize ^demo:file:///tmp/report.md", "STOP"])

    async def fake_get_enhanced_input(*_args: Any, **_kwargs: Any) -> str:
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    sent_payloads: list[str | PromptMessageExtended] = []

    async def fake_send(payload, _agent_name: str) -> str:
        sent_payloads.append(payload)
        return "ok"

    prompt_ui = InteractivePrompt()
    app = _MentionAgentApp()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="agent1",
        available_agents=["agent1"],
        prompt_provider=cast("Any", app),
    )

    assert len(sent_payloads) == 1
    payload = sent_payloads[0]
    assert isinstance(payload, PromptMessageExtended)
    assert any(isinstance(item, EmbeddedResource) for item in payload.content)
    assert payload.first_text() == "Summarize"


@pytest.mark.asyncio
async def test_prompt_loop_materializes_local_file_mentions(
    monkeypatch,
    tmp_path,
) -> None:
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9s2nRwAAAABJRU5ErkJggg=="
        )
    )
    inputs = iter([f"Compare ^file:{image_path}", "STOP"])

    async def fake_get_enhanced_input(*_args: Any, **_kwargs: Any) -> str:
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    sent_payloads: list[str | PromptMessageExtended] = []

    async def fake_send(payload, _agent_name: str) -> str:
        sent_payloads.append(payload)
        return "ok"

    prompt_ui = InteractivePrompt()
    app = _MentionAgentApp()
    app._agent_obj = _LocalMentionAgent()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="agent1",
        available_agents=["agent1"],
        prompt_provider=cast("Any", app),
    )

    assert len(sent_payloads) == 1
    payload = sent_payloads[0]
    assert isinstance(payload, PromptMessageExtended)
    assert payload.first_text() == "Compare"
    assert any(getattr(item, "type", None) == "image" for item in payload.content)


@pytest.mark.asyncio
async def test_prompt_loop_resolves_attach_paths_from_shell_working_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shell_dir = tmp_path / "shell-cwd"
    shell_dir.mkdir()
    notes = shell_dir / "notes.txt"
    notes.write_text("hello", encoding="utf-8")

    inputs = iter(["/attach ./notes.txt", "__USE_PREFILL__", "STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        next_input = next(inputs)
        if next_input == "__USE_PREFILL__":
            prefill = kwargs.get("pre_populate_buffer", "")
            assert str(notes.resolve()) in prefill
            return prefill
        return next_input

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)
    monkeypatch.setattr(interactive_prompt, "resolve_shell_working_dir", lambda **_kwargs: shell_dir)

    sent_payloads: list[str | PromptMessageExtended] = []

    async def fake_send(payload, _agent_name: str) -> str:
        sent_payloads.append(payload)
        return "ok"

    prompt_ui = InteractivePrompt()
    app = _MentionAgentApp()
    app._agent_obj = _LocalMentionAgent()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="agent1",
        available_agents=["agent1"],
        prompt_provider=cast("Any", app),
    )

    assert len(sent_payloads) == 1
    payload = sent_payloads[0]
    assert isinstance(payload, PromptMessageExtended)
    assert any(isinstance(item, EmbeddedResource) for item in payload.content)


@pytest.mark.asyncio
async def test_resolve_prompt_payload_falls_back_to_plain_text_on_resolution_error(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.png"
    prompt_ui = InteractivePrompt()
    app = _MentionAgentApp()
    app._agent_obj = _LocalMentionAgent()

    payload = await prompt_ui._resolve_prompt_payload(
        prompt_provider=cast("Any", app),
        agent_name="agent1",
        user_input=f"can you see ^file:{missing}",
    )

    assert payload == f"can you see ^file:{missing}"


@pytest.mark.asyncio
async def test_resolve_prompt_payload_uses_shell_working_dir_for_local_file_mentions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shell_dir = tmp_path / "shell-cwd"
    shell_dir.mkdir()
    notes = shell_dir / "notes.txt"
    notes.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(interactive_prompt, "resolve_shell_working_dir", lambda **_kwargs: shell_dir)

    prompt_ui = InteractivePrompt()
    app = _MentionAgentApp()
    app._agent_obj = _LocalMentionAgent()

    payload = await prompt_ui._resolve_prompt_payload(
        prompt_provider=cast("Any", app),
        agent_name="agent1",
        user_input="Read ^file:./notes.txt",
    )

    assert isinstance(payload, PromptMessageExtended)
    assert payload.first_text() == "Read"
    assert any(isinstance(item, EmbeddedResource) for item in payload.content)
