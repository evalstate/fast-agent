from __future__ import annotations

from pathlib import Path

import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers.session_export import handle_session_export
from fast_agent.session.trace_export_models import ExportRequest, ExportResult


class _StubIO:
    async def emit(self, message):
        del message
        return None

    async def prompt_text(self, prompt: str, *, default=None, allow_empty=True):
        del prompt, allow_empty
        return default

    async def prompt_selection(self, prompt: str, *, options, allow_cancel=False, default=None):
        del prompt, options, allow_cancel
        return default

    async def prompt_model_selection(self, *, initial_provider=None, default_model=None):
        del initial_provider, default_model
        return None

    async def prompt_argument(self, arg_name: str, *, description=None, required=True):
        del arg_name, description, required
        return None

    async def display_history_turn(self, agent_name: str, turn, *, turn_index=None, total_turns=None):
        del agent_name, turn, turn_index, total_turns
        return None

    async def display_history_overview(self, agent_name: str, history, usage=None):
        del agent_name, history, usage
        return None

    async def display_usage_report(self, agents):
        del agents
        return None

    async def display_system_prompt(self, agent_name: str, system_prompt: str, *, server_count=0):
        del agent_name, system_prompt, server_count
        return None


class _StubAgentProvider:
    def _agent(self, name: str):
        del name
        return object()

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "alpha"

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["alpha"]

    def registered_agent_names(self):
        return ["alpha"]

    def registered_agents(self):
        return {"alpha": object()}

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


@pytest.mark.asyncio
async def test_handle_session_export_leaves_agent_unset_for_exporter_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    session_manager = object()

    class _Exporter:
        def __init__(self, *, session_manager) -> None:
            captured["session_manager"] = session_manager

        def export(self, request):
            captured["request"] = request
            return ExportResult(
                session_id="session-1",
                agent_name=request.agent_name or "missing",
                format="codex",
                output_path=Path(request.output_path or tmp_path / "trace.jsonl"),
                record_count=3,
            )

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: session_manager)
    monkeypatch.setattr("fast_agent.commands.handlers.session_export.SessionTraceExporter", _Exporter)

    ctx = CommandContext(
        agent_provider=_StubAgentProvider(),
        current_agent_name="alpha",
        io=_StubIO(),
    )

    outcome = await handle_session_export(
        ctx,
        target="latest",
        agent_name=None,
        output_path=str(tmp_path / "trace.jsonl"),
        hf_dataset="owner/dataset",
        hf_dataset_path="exports/",
    )

    assert captured["session_manager"] is session_manager
    request = captured["request"]
    assert isinstance(request, ExportRequest)
    assert request.agent_name is None
    assert request.hf_dataset == "owner/dataset"
    assert request.hf_dataset_path == "exports/"
    assert outcome.messages
    assert "agent 'missing'" in str(outcome.messages[0].text)


@pytest.mark.asyncio
async def test_handle_session_export_requires_dataset_for_dataset_path(tmp_path: Path) -> None:
    ctx = CommandContext(
        agent_provider=_StubAgentProvider(),
        current_agent_name="alpha",
        io=_StubIO(),
    )

    outcome = await handle_session_export(
        ctx,
        target="latest",
        agent_name=None,
        output_path=str(tmp_path / "trace.jsonl"),
        hf_dataset=None,
        hf_dataset_path="exports/",
    )

    assert [str(message.text) for message in outcome.messages] == [
        "--hf-dataset-path requires --hf-dataset."
    ]
