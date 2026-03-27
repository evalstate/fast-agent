from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pytest
import yaml

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import models_manager
from fast_agent.config import Settings


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
        assert hasattr(message, "text")
        self.emitted_messages.append(cast("_HasText", message))

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


@pytest.mark.asyncio
async def test_models_aliases_lists_layered_alias_values(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    _write_yaml(
        workspace / "fastagent.config.yaml",
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
        env_dir / "fastagent.config.yaml",
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
        env_dir / "fastagent.config.yaml",
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
    expected_note = "No model_references are configured. Add a model_references section in fastagent.config.yaml."
    assert rendered.count(expected_note) == 1


@pytest.mark.asyncio
async def test_models_doctor_treats_builtin_model_alias_as_equivalent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    agents = {
        "main": _StubAgent(
            model="opus",
            tool_only=False,
            resolved_model="claude-opus-4-6",
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
    assert "claude-opus-4-6" in rendered
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

    config_path = env_dir / "fastagent.config.yaml"
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
        env_dir / "fastagent.config.yaml",
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

    saved = _read_yaml(env_dir / "fastagent.config.yaml")
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
        env_dir / "fastagent.config.yaml",
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

    assert io.last_initial_provider == "anthropic-vertex"
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

    saved = _read_yaml(env_dir / "fastagent.config.yaml")
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
        env_dir / "fastagent.config.yaml",
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

    saved = _read_yaml(env_dir / "fastagent.config.yaml")
    assert saved["model_references"]["system"]["fast"] == "gpt-4.1-mini"
    assert io.emitted_messages
    assert _message_text(io.emitted_messages[0]).find(
        str((env_dir / "fastagent.config.yaml").resolve())
    ) != -1

    rendered = str(outcome.messages[0].text)
    assert "old: claude-sonnet-4-5" in rendered
    assert "new: gpt-4.1-mini" in rendered


@pytest.mark.asyncio
async def test_models_aliases_unset_writes_project_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    project_config = workspace / "fastagent.config.yaml"
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

    assert (env_dir / "fastagent.config.yaml").exists() is False

    rendered = str(outcome.messages[0].text)
    assert "▎ model references set" in rendered
    assert "Mode: dry-run" in rendered
    assert "model_references.system.fast:" in rendered
    assert "old: <unset>" in rendered
    assert "new: claude-haiku-4-5" in rendered
    assert "Dry run only (no files changed)" in rendered


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
        parent / "fastagent.config.yaml",
        {
            "model_references": {
                "system": {
                    "code": "parent-code",
                }
            }
        },
    )
    _write_yaml(
        env_dir / "fastagent.config.yaml",
        {
            "model_references": {
                "system": {
                    "fast": "gpt-oss",
                }
            }
        },
    )

    settings = Settings(environment_dir=None)
    settings._config_file = str(parent / "fastagent.config.yaml")

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
