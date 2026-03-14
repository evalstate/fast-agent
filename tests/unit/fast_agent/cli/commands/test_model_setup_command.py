from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from fast_agent.cli.commands import model as model_command
from fast_agent.config import Settings


class _StubIO:
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
        self.prompt_text_calls: list[tuple[str, str | None, bool]] = []

    async def emit(self, message) -> None:
        del message

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        self.prompt_text_calls.append((prompt, default, allow_empty))
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
        del initial_provider, default_model
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


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if isinstance(loaded, dict):
        return loaded
    return {}


def test_build_reference_setup_argument_defaults_to_env_target() -> None:
    argument = model_command._build_reference_setup_argument(
        token=None,
        target="env",
        dry_run=False,
    )

    assert argument == "set --target env"


@pytest.mark.asyncio
async def test_run_model_setup_creates_alias_in_env_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)

    io = _StubIO(
        text_responses=["$system.fast"],
        model_selection_responses=["claude-haiku-4-5"],
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await model_command.run_model_setup(
            io=io,
            settings=Settings(environment_dir=str(env_dir)),
            token=None,
        )
    finally:
        os.chdir(previous_cwd)

    saved = _read_yaml(env_dir / "fastagent.config.yaml")
    assert saved["model_references"]["system"]["fast"] == "claude-haiku-4-5"
    assert outcome.messages
    assert "model references set" in str(outcome.messages[0].text)


@pytest.mark.asyncio
async def test_run_model_setup_prefills_system_default_alias_when_no_aliases_exist(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)

    io = _StubIO(model_selection_responses=["claude-haiku-4-5"])

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await model_command.run_model_setup(
            io=io,
            settings=Settings(environment_dir=str(env_dir)),
            token=None,
        )
    finally:
        os.chdir(previous_cwd)

    saved = _read_yaml(env_dir / "fastagent.config.yaml")
    assert saved["model_references"]["system"]["default"] == "claude-haiku-4-5"
    assert io.prompt_text_calls == [
        ("Reference token ($namespace.key):", "$system.default", False)
    ]
    assert outcome.messages
    assert "model references set" in str(outcome.messages[0].text)


@pytest.mark.asyncio
async def test_run_model_setup_repairs_invalid_default_alias_from_diagnostics(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    (workspace / "fastagent.config.yaml").write_text(
        'default_model: "$system.default"\n'
        "model_references:\n"
        "  system:\n"
        '    default: ""\n',
        encoding="utf-8",
    )

    io = _StubIO(model_selection_responses=["claude-haiku-4-5"])

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await model_command.run_model_setup(
            io=io,
            settings=Settings(environment_dir=str(env_dir)),
            token=None,
        )
    finally:
        os.chdir(previous_cwd)

    saved = _read_yaml(env_dir / "fastagent.config.yaml")
    assert saved["model_references"]["system"]["default"] == "claude-haiku-4-5"
    assert io.prompt_text_calls == []
    assert outcome.messages
    assert "model references set" in str(outcome.messages[0].text)


@pytest.mark.asyncio
async def test_run_model_setup_updates_named_alias_via_model_selector(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    config_path = env_dir / "fastagent.config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "model_references:\n  system:\n    fast: claude-sonnet-4-5\n",
        encoding="utf-8",
    )

    io = _StubIO(model_selection_responses=["gpt-4.1-mini"])

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await model_command.run_model_setup(
            io=io,
            settings=Settings(environment_dir=str(env_dir)),
            token="$system.fast",
        )
    finally:
        os.chdir(previous_cwd)

    saved = _read_yaml(config_path)
    assert saved["model_references"]["system"]["fast"] == "gpt-4.1-mini"
    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "old: claude-sonnet-4-5" in rendered
    assert "new: gpt-4.1-mini" in rendered


@pytest.mark.asyncio
async def test_run_model_doctor_reports_unresolved_default_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    (workspace / "fastagent.config.yaml").write_text(
        'default_model: "$system.default"\n',
        encoding="utf-8",
    )

    io = _StubIO()

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        outcome = await model_command.run_model_doctor(
            io=io,
            settings=Settings(environment_dir=str(env_dir)),
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "model doctor" in rendered
    assert "$system.default" in rendered


@pytest.mark.asyncio
async def test_run_model_doctor_uses_environment_dir_parent_when_cwd_differs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    elsewhere = tmp_path / "elsewhere"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    elsewhere.mkdir(parents=True)
    (workspace / "fastagent.config.yaml").write_text(
        'default_model: "$system.default"\n',
        encoding="utf-8",
    )

    io = _StubIO()

    previous_cwd = Path.cwd()
    try:
        os.chdir(elsewhere)
        outcome = await model_command.run_model_doctor(
            io=io,
            settings=Settings(environment_dir=str(env_dir)),
        )
    finally:
        os.chdir(previous_cwd)

    assert outcome.messages
    rendered = str(outcome.messages[0].text)
    assert "model doctor" in rendered
    assert "$system.default" in rendered
