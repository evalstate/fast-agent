from __future__ import annotations

from pathlib import Path

import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers.session_export import (
    _format_elapsed,
    _redaction_summary_text,
    _session_export_option_error,
    handle_session_export,
)
from fast_agent.privacy.sanitizer import RedactionSummary
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


def test_redaction_summary_text_pluralizes_text_spans() -> None:
    assert (
        _redaction_summary_text(RedactionSummary(total=0, by_label={}))
        == "Privacy filter redacted 0 text spans."
    )
    assert (
        _redaction_summary_text(RedactionSummary(total=1, by_label={"EMAIL": 1}))
        == "Privacy filter redacted 1 text span:\n  EMAIL: 1"
    )
    assert (
        _redaction_summary_text(RedactionSummary(total=2, by_label={"EMAIL": 2}))
        == "Privacy filter redacted 2 text spans:\n  EMAIL: 2"
    )


def test_format_elapsed_uses_shared_duration_display() -> None:
    assert _format_elapsed(0.5) == "0.50s"
    assert _format_elapsed(65) == "1m 05s"
    assert _format_elapsed(float("nan")) == "0.00s"


def test_session_export_option_error_normalizes_privacy_choices() -> None:
    assert (
        _session_export_option_error(
            hf_dataset=None,
            hf_dataset_path=None,
            privacy_filter=True,
            privacy_filter_path=None,
            download_privacy_filter=False,
            privacy_filter_device=" CUDA ",
            privacy_filter_variant=" Q8 ",
        )
        is None
    )


@pytest.mark.asyncio
async def test_handle_session_export_leaves_agent_unset_for_exporter_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    session_manager = object()

    class _Exporter:
        def __init__(
            self,
            *,
            session_manager,
            privacy_sanitizer=None,
            progress_callback=None,
        ) -> None:
            captured["session_manager"] = session_manager
            captured["privacy_sanitizer"] = privacy_sanitizer
            captured["progress_callback"] = progress_callback

        def export(self, request):
            captured["request"] = request
            return ExportResult(
                session_id="session-1",
                agent_name=request.agent_name or "missing",
                format="codex",
                output_path=Path(request.output_path or tmp_path / "trace.jsonl"),
                record_count=1,
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
    assert str(outcome.messages[1].text) == "Wrote 1 trace record."


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


@pytest.mark.asyncio
async def test_handle_session_export_treats_blank_dataset_as_missing(tmp_path: Path) -> None:
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
        hf_dataset="   ",
        hf_dataset_path="exports/",
    )

    assert [str(message.text) for message in outcome.messages] == [
        "--hf-dataset-path requires --hf-dataset."
    ]


@pytest.mark.asyncio
async def test_handle_session_export_trims_dataset_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_manager = object()
    captured: dict[str, object] = {}

    class _Exporter:
        def __init__(
            self,
            *,
            session_manager: object,
            privacy_sanitizer=None,
            progress_callback=None,
        ) -> None:
            captured["session_manager"] = session_manager
            captured["privacy_sanitizer"] = privacy_sanitizer
            captured["progress_callback"] = progress_callback

        def export(self, request: ExportRequest) -> ExportResult:
            captured["request"] = request
            return ExportResult(
                session_id="session-1",
                agent_name="alpha",
                format="codex",
                output_path=tmp_path / "trace.jsonl",
                record_count=1,
            )

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: session_manager)
    monkeypatch.setattr("fast_agent.commands.handlers.session_export.SessionTraceExporter", _Exporter)

    ctx = CommandContext(
        agent_provider=_StubAgentProvider(),
        current_agent_name="alpha",
        io=_StubIO(),
    )

    await handle_session_export(
        ctx,
        target="latest",
        agent_name=None,
        output_path=str(tmp_path / "trace.jsonl"),
        hf_dataset=" owner/dataset ",
        hf_dataset_path=" exports/ ",
    )

    request = captured["request"]
    assert isinstance(request, ExportRequest)
    assert request.hf_dataset == "owner/dataset"
    assert request.hf_dataset_path == "exports/"


@pytest.mark.asyncio
async def test_handle_session_export_trims_output_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_manager = object()
    captured: dict[str, object] = {}

    class _Exporter:
        def __init__(
            self,
            *,
            session_manager: object,
            privacy_sanitizer=None,
            progress_callback=None,
        ) -> None:
            captured["session_manager"] = session_manager
            captured["privacy_sanitizer"] = privacy_sanitizer
            captured["progress_callback"] = progress_callback

        def export(self, request: ExportRequest) -> ExportResult:
            captured["request"] = request
            return ExportResult(
                session_id="session-1",
                agent_name="alpha",
                format="codex",
                output_path=request.output_path or tmp_path / "default.jsonl",
                record_count=1,
            )

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: session_manager)
    monkeypatch.setattr("fast_agent.commands.handlers.session_export.SessionTraceExporter", _Exporter)

    ctx = CommandContext(
        agent_provider=_StubAgentProvider(),
        current_agent_name="alpha",
        io=_StubIO(),
    )
    output_path = tmp_path / "trace.jsonl"

    await handle_session_export(
        ctx,
        target="latest",
        agent_name=None,
        output_path=f" {output_path} ",
        hf_dataset=None,
        hf_dataset_path=None,
    )

    request = captured["request"]
    assert isinstance(request, ExportRequest)
    assert request.output_path == output_path


@pytest.mark.asyncio
async def test_handle_session_export_treats_blank_output_path_as_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_manager = object()
    captured: dict[str, object] = {}

    class _Exporter:
        def __init__(
            self,
            *,
            session_manager: object,
            privacy_sanitizer=None,
            progress_callback=None,
        ) -> None:
            del session_manager, privacy_sanitizer, progress_callback

        def export(self, request: ExportRequest) -> ExportResult:
            captured["request"] = request
            return ExportResult(
                session_id="session-1",
                agent_name="alpha",
                format="codex",
                output_path=request.output_path or tmp_path / "default.jsonl",
                record_count=1,
            )

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: session_manager)
    monkeypatch.setattr("fast_agent.commands.handlers.session_export.SessionTraceExporter", _Exporter)

    ctx = CommandContext(
        agent_provider=_StubAgentProvider(),
        current_agent_name="alpha",
        io=_StubIO(),
    )

    await handle_session_export(
        ctx,
        target="latest",
        agent_name=None,
        output_path="   ",
        hf_dataset=None,
        hf_dataset_path=None,
    )

    request = captured["request"]
    assert isinstance(request, ExportRequest)
    assert request.output_path is None


@pytest.mark.asyncio
async def test_handle_session_export_reports_missing_privacy_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "fast_agent.commands.handlers.session_export.missing_privacy_dependencies",
        lambda: ["onnxruntime", "tokenizers"],
    )
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
        hf_dataset_path=None,
        privacy_filter=True,
    )

    assert outcome.messages
    assert outcome.messages[0].channel == "error"
    assert "onnxruntime" in str(outcome.messages[0].text)
    assert "fast-agent-mcp[privacy]" in str(outcome.messages[0].text)


@pytest.mark.asyncio
async def test_handle_session_export_requires_privacy_filter_for_privacy_options(
    tmp_path: Path,
) -> None:
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
        hf_dataset_path=None,
        privacy_filter_path="/tmp/model",
    )

    assert [str(message.text) for message in outcome.messages] == [
        "--privacy-filter-path, --download-privacy-filter, "
        "--privacy-filter-device, and --privacy-filter-variant require --privacy-filter."
    ]


@pytest.mark.asyncio
async def test_handle_session_export_rejects_unknown_privacy_filter_device(
    tmp_path: Path,
) -> None:
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
        hf_dataset_path=None,
        privacy_filter=True,
        privacy_filter_device="tpu",
    )

    assert [str(message.text) for message in outcome.messages] == [
        "Unsupported --privacy-filter-device 'tpu'. Supported devices: auto, cpu, cuda."
    ]


@pytest.mark.asyncio
async def test_handle_session_export_rejects_unknown_privacy_filter_variant(
    tmp_path: Path,
) -> None:
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
        hf_dataset_path=None,
        privacy_filter=True,
        privacy_filter_variant="int2",
    )

    assert [str(message.text) for message in outcome.messages] == [
        "Unsupported --privacy-filter-variant 'int2'. Supported variants: q4, q4f16, q8, fp16."
    ]
