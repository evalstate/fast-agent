from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest
import yaml

from fast_agent.commands.command_catalog import get_command_action_spec
from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import models_manager
from fast_agent.config import Settings
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker_common import ANTHROPIC_VERTEX_PROVIDER_KEY


class _StubAgentProvider:
    def __init__(self, agents: dict[str, _StubAgent] | None = None) -> None:
        self._agents = agents or {}

    def _agent(self, name: str):
        return self._agents[name]

    def visible_agent_names(self, *, force_include: str | None = None):
        names = [
            name
            for name, agent in self._agents.items()
            if not bool(getattr(getattr(agent, "config", None), "tool_only", False))
        ]
        if force_include and force_include in self._agents and force_include not in names:
            return [force_include, *names]
        return names

    def registered_agent_names(self):
        return list(self._agents.keys())

    def registered_agents(self):
        return self._agents

    def resolve_target_agent_name(self, agent_name: str | None = None):
        if agent_name is not None:
            return agent_name
        visible = self.visible_agent_names()
        return visible[0] if visible else next(iter(self._agents), None)

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None) -> dict[str, str]:
        del namespace, agent_name
        return {}


@runtime_checkable
class _HasText(Protocol):
    text: object


class _StubCommandIO:
    def __init__(
        self,
        *,
        text_responses: list[str | None] | None = None,
        selection_responses: list[str | None] | None = None,
        model_selection_responses: list[str | None] | None = None,
    ) -> None:
        self._text_responses = list(text_responses or [])
        self._selection_responses = list(selection_responses or [])
        self._model_selection_responses = list(model_selection_responses or [])
        self.emitted_messages: list[_HasText] = []
        self.last_initial_provider: str | None = None
        self.last_default_model: str | None = None

    async def emit(self, message: object) -> None:
        assert isinstance(message, _HasText)
        self.emitted_messages.append(message)

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        del prompt, allow_empty
        if self._text_responses:
            return self._text_responses.pop(0)
        return default

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options,
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        del prompt, options, allow_cancel
        if self._selection_responses:
            return self._selection_responses.pop(0)
        return default

    async def prompt_model_selection(
        self,
        *,
        initial_provider: str | None = None,
        default_model: str | None = None,
    ) -> str | None:
        self.last_initial_provider = initial_provider
        self.last_default_model = default_model
        if self._model_selection_responses:
            return self._model_selection_responses.pop(0)
        return None

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        del arg_name, description, required
        return None

    async def display_history_turn(self, *args, **kwargs) -> None:
        del args, kwargs

    async def display_history_overview(self, *args, **kwargs) -> None:
        del args, kwargs

    async def display_usage_report(self, *args, **kwargs) -> None:
        del args, kwargs

    async def display_system_prompt(self, *args, **kwargs) -> None:
        del args, kwargs


@dataclass
class _StubAgentConfig:
    model: str | None = None
    tool_only: bool = False


class _StubLlm:
    def __init__(self, model_name: str | None) -> None:
        self.model_name = model_name


class _StubAgent:
    def __init__(self, *, model: str | None, tool_only: bool, resolved_model: str | None) -> None:
        self.config = _StubAgentConfig(model=model, tool_only=tool_only)
        self.agent_type = "basic"
        self.llm = _StubLlm(model_name=resolved_model) if resolved_model is not None else None


def _message_text(message: _HasText) -> str:
    """Extract stringified message text from dynamically captured command IO output."""
    return str(message.text)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if isinstance(loaded, dict):
        return loaded
    return {}


def _context(settings: Settings, *, agents: dict[str, _StubAgent] | None = None) -> CommandContext:
    return CommandContext(
        agent_provider=_StubAgentProvider(agents),
        current_agent_name="main",
        io=_StubCommandIO(),
        settings=settings,
    )


def _context_with_io(
    settings: Settings,
    io: _StubCommandIO,
    *,
    agents: dict[str, _StubAgent] | None = None,
) -> CommandContext:
    return CommandContext(
        agent_provider=_StubAgentProvider(agents),
        current_agent_name="main",
        io=io,
        settings=settings,
    )


def test_agent_model_markdown_row_escapes_table_cells() -> None:
    row = models_manager._AgentModelDoctorRow(
        name="main|agent`beta",
        specified_model="provider/model|variant`preview",
        resolved_model="resolved\nmodel`x",
        status=models_manager._AgentModelStatus.RESOLVED,
        status_style="green",
    )

    rendered = models_manager._agent_model_markdown_row(row)

    assert "`` main\\|agent\\`beta ``" in rendered
    assert "`` provider/model\\|variant\\`preview ``" in rendered
    assert "`` resolved model\\`x ``" in rendered


def test_agent_model_markdown_notes_escape_backticks_in_agent_names() -> None:
    row = models_manager._AgentModelDoctorRow(
        name="main`agent",
        specified_model="$missing",
        resolved_model="",
        status=models_manager._AgentModelStatus.UNRESOLVED,
        status_style="red",
        resolution_note="Alias is not configured.",
    )

    notes = models_manager._agent_model_markdown_notes([row])

    assert notes == ["`` main`agent ``: Alias is not configured."]


def test_models_doctor_markdown_sections_escape_backticks_in_inline_code() -> None:
    report = models_manager._ModelsDoctorReport(
        readiness_ready=False,
        env_dir_env="/tmp/env`dir",
        effective_env_dir="/tmp/effective`dir",
        fast_agent_model_env="model`env",
        loaded_config_file="/tmp/config`file.yaml",
        unresolved=[("$system.`fast", "default model", "missing")],
        configured_providers=set(),
        agent_rows=[],
        default_provider=None,
        default_provider_ready=True,
    )
    lines: list[str] = []

    models_manager._extend_markdown_doctor_overview(lines, report)
    models_manager._extend_markdown_unresolved_references(lines, report.unresolved)

    rendered = "\n".join(lines)
    assert "ENVIRONMENT_DIR**: `` /tmp/env`dir ``" in rendered
    assert "Effective environment_dir**: `` /tmp/effective`dir ``" in rendered
    assert "FAST_AGENT_MODEL**: `` model`env ``" in rendered
    assert "Loaded config file**: `` /tmp/config`file.yaml ``" in rendered
    assert "- `` $system.`fast `` (default model)" in rendered


def test_canonical_model_name_normalizes_model_spec_text() -> None:
    assert models_manager._canonical_model_name("  anthropic.claude-haiku-4-5  ") == (
        "claude-haiku-4-5"
    )
    assert models_manager._canonical_model_name("   ") == ""


def test_canonical_model_name_returns_stripped_spec_when_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_parse_error(*_args: object, **_kwargs: object) -> object:
        raise ValueError("bad model spec")

    monkeypatch.setattr(
        models_manager.ModelFactory,
        "parse_model_string",
        _raise_parse_error,
    )

    assert models_manager._canonical_model_name("  provider.raw-name  ") == "provider.raw-name"


def test_models_equivalent_falls_back_when_database_normalization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_normalization_error(_model: str) -> str:
        raise ValueError("bad database")

    monkeypatch.setattr(
        models_manager.ModelDatabase,
        "normalize_model_name",
        _raise_normalization_error,
    )

    assert models_manager._models_equivalent(
        "anthropic.claude-haiku-4-5",
        "claude-haiku-4-5",
    )


def test_resolve_config_payload_returns_empty_when_model_dump_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_dump_error(_settings: Settings) -> object:
        raise ValueError("bad settings payload")

    monkeypatch.setattr(Settings, "model_dump", _raise_dump_error)

    assert models_manager._resolve_config_payload(Settings()) == {}


def test_resolve_config_payload_returns_empty_for_non_mapping_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Settings, "model_dump", lambda _settings: ["not", "a", "mapping"])

    assert models_manager._resolve_config_payload(Settings()) == {}


def test_default_model_provider_returns_none_when_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_parse_error(*_args: object, **_kwargs: object) -> object:
        raise ValueError("bad model spec")

    monkeypatch.setattr(
        models_manager.ModelFactory,
        "parse_model_string",
        _raise_parse_error,
    )

    assert (
        models_manager._default_model_provider(
            default_model="anthropic.claude-haiku-4-5",
            references={},
        )
        is None
    )


def test_normalize_interactive_reference_token_uses_shared_text_rules() -> None:
    assert models_manager._normalize_interactive_reference_token(" system.fast ") == "$system.fast"
    assert models_manager._normalize_interactive_reference_token(" $system.fast ") == "$system.fast"
    assert models_manager._normalize_interactive_reference_token("   ") == ""


@pytest.mark.asyncio
async def test_models_aliases_lists_layered_project_and_env_values(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    _write_yaml(
        workspace / "fast-agent.yaml",
        {
            "model_references": {
                "system": {
                    "fast": "project-fast",
                    "code": "project-code",
                }
            }
        },
    )
    _write_yaml(
        env_dir / "fast-agent.yaml",
        {
            "model_references": {
                "system": {
                    "fast": "env-fast",
                }
            }
        },
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir))),
            agent_name="main",
            action="references",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "▎ model references" in rendered
    assert "▎•" not in rendered
    assert "$system.fast = env-fast" in rendered
    assert "$system.code = project-code" in rendered


@pytest.mark.asyncio
async def test_models_doctor_reports_unresolved_default_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(
                Settings(
                    environment_dir=str(env_dir),
                    default_model="$system.fast",
                )
            ),
            agent_name="main",
            action="doctor",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "▎ model doctor" in rendered
    assert "• ENVIRONMENT_DIR:" in rendered
    assert "▎•" not in rendered
    assert "Readiness: action required" in rendered
    assert "Agent summary:" in rendered
    assert "$system.fast (default_model)" in rendered


@pytest.mark.asyncio
async def test_models_doctor_lists_all_agents_including_tool_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    _write_yaml(
        env_dir / "fast-agent.yaml",
        {
            "model_references": {
                "system": {
                    "fast": "claude-haiku-4-5",
                }
            }
        },
    )

    agents = {
        "main": _StubAgent(
            model="$system.fast",
            tool_only=False,
            resolved_model="claude-haiku-4-5",
        ),
        "reviewer_tool": _StubAgent(
            model="$system.missing",
            tool_only=True,
            resolved_model=None,
        ),
    }

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir)), agents=agents),
            agent_name="main",
            action="doctor",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "▎ Agent model resolution" in rendered
    assert "Agent summary:" in rendered
    assert "main" in rendered
    assert "reviewer_tool" in rendered
    assert "$system.fast" in rendered
    assert "$system.missing" in rendered
    assert "<unresolved>" in rendered
    assert "note: Unknown key 'missing' in namespace 'system'." in rendered


@pytest.mark.asyncio
async def test_models_doctor_marks_runtime_fallback_when_alias_unresolved(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    agents = {
        "main": _StubAgent(
            model="$system.missing",
            tool_only=False,
            resolved_model="claude-haiku-4-5",
        ),
    }

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir)), agents=agents),
            agent_name="main",
            action="doctor",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "Agent summary:" in rendered
    assert "◐" in rendered
    assert "claude-haiku-4-5" in rendered
    assert "No model_references are configured." in rendered


@pytest.mark.asyncio
async def test_models_doctor_dedupes_repeated_alias_missing_note(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    agents = {
        "main": _StubAgent(
            model="$system.missing",
            tool_only=False,
            resolved_model="claude-haiku-4-5",
        ),
        "secondary": _StubAgent(
            model="$system.missing",
            tool_only=False,
            resolved_model="gpt-4.1-mini",
        ),
    }

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir)), agents=agents),
            agent_name="main",
            action="doctor",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    expected_note = "No model_references are configured. Add a model_references section in fast-agent.yaml."
    assert rendered.count(expected_note) == 1


@pytest.mark.asyncio
async def test_models_doctor_treats_builtin_model_alias_as_equivalent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    resolved_opus = ModelFactory.parse_model_string("opus").model_name

    agents = {
        "main": _StubAgent(
            model="opus",
            tool_only=False,
            resolved_model=resolved_opus,
        ),
    }

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir)), agents=agents),
            agent_name="main",
            action="doctor",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "main" in rendered
    assert "opus" in rendered
    assert resolved_opus in rendered
    assert "Resolved spec suggests" not in rendered


@pytest.mark.asyncio
async def test_models_doctor_treats_gpt_oss_alias_and_normalized_model_as_equivalent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    agents = {
        "main": _StubAgent(
            model="gpt-oss",
            tool_only=False,
            resolved_model="openai/gpt-oss-120b",
        ),
    }

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir)), agents=agents),
            agent_name="main",
            action="doctor",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "gpt-oss" in rendered
    assert "openai/gpt-oss-120b" in rendered
    assert "Resolved spec suggests" not in rendered


@pytest.mark.asyncio
async def test_models_catalog_lists_curated_provider_models() -> None:
    outcome = await models_manager.handle_models_command(
        _context(Settings()),
        agent_name="main",
        action="catalog",
        argument="anthropic",
    )

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "Provider: Anthropic" in rendered
    assert "claude-haiku-4-5" in rendered


@pytest.mark.asyncio
async def test_models_unknown_action_uses_catalog_message() -> None:
    outcome = await models_manager.handle_models_command(
        _context(Settings()),
        agent_name="main",
        action="catlog",
        argument=None,
    )

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "Unknown /model action: catlog. Use " in rendered
    assert "reasoning/task_budget/verbosity/fast/web_search/x_search/web_fetch" in rendered
    assert "Did you mean: `catalog`" in rendered


def test_parse_catalog_arguments_accepts_provider_and_all_flag() -> None:
    parsed = models_manager._parse_catalog_arguments("anthropic --all")

    assert parsed.provider_name == "anthropic"
    assert parsed.show_all is True
    assert parsed.error is None


def test_parse_catalog_arguments_accepts_repeated_all_flag() -> None:
    parsed = models_manager._parse_catalog_arguments("--all anthropic --all")

    assert parsed.provider_name == "anthropic"
    assert parsed.show_all is True
    assert parsed.error is None


def test_resolve_catalog_provider_normalizes_aliases() -> None:
    assert models_manager._resolve_catalog_provider(" HF ") is Provider.HUGGINGFACE
    assert models_manager._resolve_catalog_provider(" Codex-Responses ") is Provider.CODEX_RESPONSES


def test_parse_catalog_arguments_uses_command_surface_usage() -> None:
    parsed = models_manager._parse_catalog_arguments(None, command_name="models")

    assert parsed.error == "Usage: /models catalog <provider> [--all]"


def test_parse_catalog_arguments_treats_blank_argument_as_missing() -> None:
    parsed = models_manager._parse_catalog_arguments("   ")
    catalog_action = get_command_action_spec("model", "catalog")

    assert parsed.provider_name is None
    assert parsed.show_all is False
    assert catalog_action is not None
    assert parsed.error == f"Usage: {catalog_action.usage}"


def test_is_model_manager_action_owns_dispatch_actions() -> None:
    assert models_manager.is_model_manager_action(" doctor ")
    assert models_manager.is_model_manager_action("references")
    assert models_manager.is_model_manager_action(" CATALOG ")
    assert models_manager.is_model_manager_action("help")
    assert not models_manager.is_model_manager_action("reasoning")


@pytest.mark.asyncio
async def test_models_command_normalizes_catalogued_help_alias() -> None:
    outcome = await models_manager.handle_models_command(
        _context(Settings()),
        agent_name="main",
        action="-h",
        argument=None,
    )

    assert outcome.messages
    assert "Usage: /model" in str(outcome.messages[1].text)


@pytest.mark.asyncio
async def test_models_command_help_uses_models_surface() -> None:
    outcome = await models_manager.handle_models_command(
        _context(Settings()),
        agent_name="main",
        action="help",
        argument=None,
        command_name="models",
    )

    rendered = "\n".join(str(message.text) for message in outcome.messages)
    assert "models help" in rendered
    assert "Usage: /models" in rendered
    assert "Examples: /models doctor" in rendered
    assert "Examples: /model doctor" not in rendered


@pytest.mark.asyncio
async def test_models_catalog_error_uses_models_surface() -> None:
    outcome = await models_manager.handle_models_command(
        _context(Settings()),
        agent_name="main",
        action="catalog",
        argument=None,
        command_name="models",
    )

    rendered = "\n".join(str(message.text) for message in outcome.messages)
    assert "Usage: /models catalog <provider> [--all]" in rendered
    assert "Usage: /model catalog <provider> [--all]" not in rendered


def test_parse_catalog_arguments_rejects_extra_provider() -> None:
    parsed = models_manager._parse_catalog_arguments("anthropic openai")

    assert parsed.provider_name is None
    assert parsed.show_all is False
    assert parsed.error == "Only one provider may be specified."


def test_assign_catalog_provider_reports_duplicate_provider() -> None:
    assigned = models_manager._assign_catalog_provider(
        "anthropic",
        "openai",
        usage="Usage: /model catalog <provider>",
    )

    assert assigned.provider_name == "anthropic"
    assert assigned.error == "Only one provider may be specified."


def test_parse_catalog_arguments_rejects_empty_provider() -> None:
    parsed = models_manager._parse_catalog_arguments('"" --all')
    catalog_action = get_command_action_spec("model", "catalog")

    assert parsed.provider_name is None
    assert parsed.show_all is False
    assert catalog_action is not None
    assert parsed.error == f"Usage: {catalog_action.usage}"


def test_parse_catalog_arguments_reports_split_errors() -> None:
    parsed = models_manager._parse_catalog_arguments('anthropic "unterminated')

    assert parsed.provider_name is None
    assert parsed.show_all is False
    assert parsed.error == "Invalid catalog arguments: No closing quotation"


@pytest.mark.asyncio
async def test_models_aliases_set_writes_env_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir))),
            agent_name="main",
            action="references",
            argument="set $system.fast claude-haiku-4-5 --target env",
        )
    finally:
        os.chdir(previous_cwd)

    config_path = env_dir / "fast-agent.yaml"
    assert config_path.exists()
    saved = _read_yaml(config_path)
    assert saved["model_references"]["system"]["fast"] == "claude-haiku-4-5"

    rendered = str(outcome.messages[0].text)
    assert "▎ model references set" in rendered
    assert "Result: applied" in rendered
    assert f"Target: {config_path}" in rendered
    assert "model_references.system.fast:" in rendered
    assert "old: <unset>" in rendered
    assert "new: claude-haiku-4-5" in rendered


@pytest.mark.asyncio
async def test_models_aliases_set_uses_model_selector_for_existing_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    _write_yaml(
        env_dir / "fast-agent.yaml",
        {
            "model_references": {
                "system": {
                    "fast": "claude-sonnet-4-5",
                }
            }
        },
    )

    io = _StubCommandIO(model_selection_responses=["claude-haiku-4-5"])

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context_with_io(Settings(environment_dir=str(env_dir)), io),
            agent_name="main",
            action="references",
            argument="set $system.fast",
        )
    finally:
        os.chdir(previous_cwd)

    saved = _read_yaml(env_dir / "fast-agent.yaml")
    assert saved["model_references"]["system"]["fast"] == "claude-haiku-4-5"

    rendered = str(outcome.messages[0].text)
    assert "▎ model references set" in rendered
    assert "model_references.system.fast:" in rendered
    assert "old: claude-sonnet-4-5" in rendered
    assert "new: claude-haiku-4-5" in rendered


@pytest.mark.asyncio
async def test_models_aliases_set_reopens_vertex_selection_for_vertex_model(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    _write_yaml(
        env_dir / "fast-agent.yaml",
        {
            "model_references": {
                "system": {
                    "fast": "anthropic-vertex.claude-sonnet-4-6",
                }
            }
        },
    )

    io = _StubCommandIO(model_selection_responses=["anthropic-vertex.claude-sonnet-4-6"])

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context_with_io(Settings(environment_dir=str(env_dir)), io),
            agent_name="main",
            action="references",
            argument="set $system.fast",
        )
    finally:
        os.chdir(previous_cwd)

    assert io.last_initial_provider == ANTHROPIC_VERTEX_PROVIDER_KEY
    assert io.last_default_model == "anthropic-vertex.claude-sonnet-4-6"
    assert "no changes" in str(outcome.messages[0].text)


@pytest.mark.asyncio
async def test_models_aliases_set_can_create_new_alias_interactively(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    io = _StubCommandIO(
        text_responses=["$custom.review"],
        model_selection_responses=["gpt-4.1-mini"],
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context_with_io(Settings(environment_dir=str(env_dir)), io),
            agent_name="main",
            action="references",
            argument="set",
        )
    finally:
        os.chdir(previous_cwd)

    saved = _read_yaml(env_dir / "fast-agent.yaml")
    assert saved["model_references"]["custom"]["review"] == "gpt-4.1-mini"

    rendered = str(outcome.messages[0].text)
    assert "▎ model references set" in rendered
    assert "model_references.custom.review:" in rendered
    assert "old: <unset>" in rendered
    assert "new: gpt-4.1-mini" in rendered


@pytest.mark.asyncio
async def test_models_aliases_set_can_choose_existing_alias_by_number(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    _write_yaml(
        env_dir / "fast-agent.yaml",
        {
            "model_references": {
                "system": {
                    "fast": "claude-sonnet-4-5",
                }
            }
        },
    )

    io = _StubCommandIO(
        selection_responses=["1"],
        model_selection_responses=["gpt-4.1-mini"],
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context_with_io(Settings(environment_dir=str(env_dir)), io),
            agent_name="main",
            action="references",
            argument="set",
        )
    finally:
        os.chdir(previous_cwd)

    saved = _read_yaml(env_dir / "fast-agent.yaml")
    assert saved["model_references"]["system"]["fast"] == "gpt-4.1-mini"
    assert io.emitted_messages
    assert _message_text(io.emitted_messages[0]).find(
        str((env_dir / "fast-agent.yaml").resolve())
    ) != -1

    rendered = str(outcome.messages[0].text)
    assert "old: claude-sonnet-4-5" in rendered
    assert "new: gpt-4.1-mini" in rendered


@pytest.mark.asyncio
async def test_models_aliases_unset_writes_project_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    project_config = workspace / "fast-agent.yaml"
    _write_yaml(
        project_config,
        {
            "model_references": {
                "system": {
                    "fast": "claude-haiku-4-5",
                    "code": "claude-sonnet-4-5",
                }
            }
        },
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir))),
            agent_name="main",
            action="references",
            argument="unset $system.fast --target project",
        )
    finally:
        os.chdir(previous_cwd)

    saved = _read_yaml(project_config)
    assert "fast" not in saved["model_references"]["system"]
    assert saved["model_references"]["system"]["code"] == "claude-sonnet-4-5"

    rendered = str(outcome.messages[0].text)
    assert "▎ model references unset" in rendered
    assert "Result: applied" in rendered
    assert f"Target: {project_config}" in rendered
    assert "model_references.system.fast:" in rendered
    assert "old: claude-haiku-4-5" in rendered
    assert "new: <unset>" in rendered


@pytest.mark.asyncio
async def test_models_aliases_set_dry_run_is_deterministic(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir))),
            agent_name="main",
            action="references",
            argument="set $system.fast claude-haiku-4-5 --target env --dry-run",
        )
    finally:
        os.chdir(previous_cwd)

    assert (env_dir / "fast-agent.yaml").exists() is False

    rendered = str(outcome.messages[0].text)
    assert "▎ model references set" in rendered
    assert "Mode: dry-run" in rendered
    assert "model_references.system.fast:" in rendered
    assert "old: <unset>" in rendered
    assert "new: claude-haiku-4-5" in rendered
    assert "Dry run only (no files changed)" in rendered


def test_models_references_parser_accepts_equals_target_form() -> None:
    parsed = models_manager._parse_references_arguments(
        "set $system.fast claude-haiku-4-5 --target=project --dry-run"
    )

    assert parsed.mode == "mutate"
    assert parsed.error is None
    assert parsed.mutation is not None
    assert parsed.mutation.operation == "set"
    assert parsed.mutation.target == "project"
    assert parsed.mutation.dry_run is True


def test_models_references_parser_normalizes_subcommand_case() -> None:
    parsed = models_manager._parse_references_arguments("LIST")

    assert parsed.mode == "list"
    assert parsed.error is None
    assert parsed.mutation is None


@pytest.mark.asyncio
async def test_prompt_for_existing_reference_selection_normalizes_selection(
    tmp_path: Path,
) -> None:
    io = _StubCommandIO(selection_responses=[" NEW "])
    ctx = _context_with_io(Settings(), io)

    selection = await models_manager._prompt_for_existing_reference_selection(
        ctx,
        rows=[("$system.fast", "claude-haiku-4-5")],
        target_path=tmp_path / "fast-agent.yaml",
        prompt="Reference:",
        include_new=True,
    )

    assert selection == "new"


def test_models_references_parser_normalizes_target_value() -> None:
    parsed = models_manager._parse_references_arguments(
        "set $system.fast claude-haiku-4-5 --target PROJECT"
    )

    assert parsed.error is None
    assert parsed.mutation is not None
    assert parsed.mutation.target == "project"


def test_models_references_parser_rejects_invalid_target_consistently() -> None:
    cases = [
        "set $system.fast claude-haiku-4-5 --target user",
        "set $system.fast claude-haiku-4-5 --target=user",
    ]

    for argument in cases:
        parsed = models_manager._parse_references_arguments(argument)

        assert parsed.mode == "list"
        assert parsed.mutation is None
        assert parsed.error == "--target must be either 'env' or 'project'."


def test_models_references_parser_rejects_duplicate_target() -> None:
    cases = [
        "set $system.fast claude-haiku-4-5 --target env --target project",
        "set $system.fast claude-haiku-4-5 --target=env --target=project",
    ]

    for argument in cases:
        parsed = models_manager._parse_references_arguments(argument)

        assert parsed.mode == "list"
        assert parsed.mutation is None
        assert parsed.error == "Duplicate option: --target"


def test_models_references_parser_rejects_missing_target_value_consistently() -> None:
    cases = [
        "set $system.fast claude-haiku-4-5 --target",
        "set $system.fast claude-haiku-4-5 --target=",
        'set $system.fast claude-haiku-4-5 --target ""',
        "set $system.fast claude-haiku-4-5 --target --dry-run",
    ]

    for argument in cases:
        parsed = models_manager._parse_references_arguments(argument)

        assert parsed.mode == "list"
        assert parsed.mutation is None
        assert parsed.error == "Missing value for --target (expected env or project)"


def test_models_references_parser_rejects_empty_positionals() -> None:
    cases = [
        'set "" claude-haiku-4-5',
        'set $system.fast ""',
        'unset ""',
    ]

    for argument in cases:
        parsed = models_manager._parse_references_arguments(argument)

        assert parsed.mode == "list"
        assert parsed.mutation is None
        assert parsed.error == "Reference positional arguments cannot be empty."


def test_models_references_parser_strips_quoted_positionals() -> None:
    parsed = models_manager._parse_references_arguments(
        'set " $system.fast " " claude-haiku-4-5 "'
    )

    assert parsed.error is None
    assert parsed.mutation is not None
    assert parsed.mutation.token == "$system.fast"
    assert parsed.mutation.model_spec == "claude-haiku-4-5"


def test_models_references_parser_reports_shell_quoting_errors() -> None:
    parsed = models_manager._parse_references_arguments('set "$system.fast')

    assert parsed.mode == "list"
    assert parsed.mutation is None
    assert parsed.error is not None
    assert parsed.error.startswith("Invalid references arguments:")


@pytest.mark.asyncio
async def test_models_aliases_set_invalid_token_returns_usage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir))),
            agent_name="main",
            action="references",
            argument="set system.fast claude-haiku-4-5",
        )
    finally:
        os.chdir(previous_cwd)

    rendered = str(outcome.messages[0].text)
    assert "Model references must be exact tokens in the format '$<namespace>.<key>'" in rendered
    assert "Usage: /model references" in rendered


@pytest.mark.asyncio
async def test_models_doctor_displays_runtime_config_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    previous_fast_model = os.environ.get("FAST_AGENT_MODEL")
    try:
        os.chdir(workspace)
        os.environ["ENVIRONMENT_DIR"] = str(env_dir)
        os.environ["FAST_AGENT_MODEL"] = "kimi"
        outcome = await models_manager.handle_models_command(
            _context(Settings(environment_dir=str(env_dir))),
            agent_name="main",
            action="doctor",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
        if previous_fast_model is None:
            os.environ.pop("FAST_AGENT_MODEL", None)
        else:
            os.environ["FAST_AGENT_MODEL"] = previous_fast_model

    rendered = str(outcome.messages[0].text)
    assert "▎ Runtime config context" in rendered
    assert f"ENVIRONMENT_DIR: {env_dir}" in rendered
    assert f"Effective environment_dir: {env_dir}" in rendered
    assert "FAST_AGENT_MODEL: kimi" in rendered


@pytest.mark.asyncio
async def test_models_references_follow_loaded_config_root_instead_of_cwd_overlay(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    workspace = parent / "workspace"
    env_dir = workspace / ".fast-agent"
    parent.mkdir(parents=True)
    workspace.mkdir(parents=True)

    _write_yaml(
        parent / "fast-agent.yaml",
        {
            "model_references": {
                "system": {
                    "code": "parent-code",
                }
            }
        },
    )
    _write_yaml(
        env_dir / "fast-agent.yaml",
        {
            "model_references": {
                "system": {
                    "fast": "gpt-oss",
                }
            }
        },
    )

    settings = Settings(environment_dir=None)
    settings._config_file = str(parent / "fast-agent.yaml")

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    try:
        os.environ.pop("ENVIRONMENT_DIR", None)
        os.chdir(workspace)
        outcome = await models_manager.handle_models_command(
            _context(settings),
            agent_name="main",
            action="references",
            argument=None,
        )
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    rendered = str(outcome.messages[0].text)
    assert "$system.code = parent-code" in rendered
    assert "$system.fast = gpt-oss" not in rendered
