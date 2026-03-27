from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from fast_agent.cli.runtime.agent_setup import (
    _emit_model_picker_keyring_notice,
    _generic_model_prompt_default,
    _load_request_settings,
    _normalize_generic_model_spec,
    _persist_model_picker_last_used_selection,
    _resolve_model_picker_initial_selection,
    _resolve_model_without_hardcoded_default,
    _select_model_from_picker,
    _should_prompt_for_model_picker,
    run_agent_request,
)
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.config import Settings
from fast_agent.ui.model_picker import ModelPickerResult


def _make_request(
    *,
    config_path: str | None = None,
    message: str | None = None,
    prompt_file: str | None = None,
    agent_cards: list[str] | None = None,
    card_tools: list[str] | None = None,
) -> AgentRunRequest:
    return AgentRunRequest(
        name="test",
        instruction="instruction",
        config_path=config_path,
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


def test_should_not_prompt_for_model_picker_when_prompt_file_mode() -> None:
    request = _make_request(prompt_file="prompt.txt")

    assert not _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


def test_should_prompt_for_model_picker_when_cards_present() -> None:
    request = _make_request(agent_cards=["cards/"])

    assert _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


def test_model_picker_keyring_notice_is_emitted_immediately(monkeypatch) -> None:
    emitted: list[tuple[str, bool]] = []
    queued: list[object] = []

    def fake_emit_keyring_access_notice(*, purpose=None, emitter=None):
        assert purpose == "checking stored Codex OAuth tokens for model setup"
        assert emitter is not None
        emitter("keyring notice")
        return True

    monkeypatch.setattr("fast_agent.cli.runtime.agent_setup.emit_keyring_access_notice", fake_emit_keyring_access_notice)
    monkeypatch.setattr("fast_agent.cli.runtime.agent_setup.typer.echo", lambda message, err=False: emitted.append((message, err)))
    monkeypatch.setattr("fast_agent.cli.runtime.agent_setup.sys.stderr.isatty", lambda: True)
    monkeypatch.setattr("fast_agent.ui.enhanced_prompt.queue_startup_notice", queued.append)

    _emit_model_picker_keyring_notice(_make_request())

    assert emitted == [("keyring notice", True)]
    assert queued == []


def test_resolve_model_without_hardcoded_default_returns_none_without_sources() -> None:
    previous = os.environ.pop("FAST_AGENT_MODEL", None)
    try:
        model, source = _resolve_model_without_hardcoded_default(
            model=None,
            config_default_model=None,
            model_references=None,
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
            model_references=None,
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
            model_references=None,
        )
    finally:
        if previous is not None:
            os.environ["FAST_AGENT_MODEL"] = previous
        else:
            os.environ.pop("FAST_AGENT_MODEL", None)

    assert model == "responses.gpt-5-mini"
    assert source == "environment variable FAST_AGENT_MODEL"


@pytest.mark.asyncio
async def test_select_model_from_picker_preserves_overlay_token_when_resolved_model_is_present(
    monkeypatch,
) -> None:
    request = _make_request()

    async def fake_run_model_picker_async(**kwargs):
        del kwargs
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

    selected = await _select_model_from_picker(request, config_payload={})

    assert selected == "haikutiny"


@pytest.mark.asyncio
async def test_select_model_from_picker_passes_config_start_path(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "project" / "fastagent.config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("default_model: haikutiny\n", encoding="utf-8")
    request = _make_request(config_path=str(config_path))
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

    selected = await _select_model_from_picker(request, config_payload={})

    assert selected == "haikutiny"
    assert captured_kwargs["start_path"] == config_path.parent


def test_normalize_generic_model_spec_adds_generic_prefix_when_missing() -> None:
    assert _normalize_generic_model_spec("llama3.2") == "generic.llama3.2"


def test_normalize_generic_model_spec_preserves_explicit_provider_prefix() -> None:
    assert _normalize_generic_model_spec("generic.llama3.2:latest") == "generic.llama3.2:latest"
    assert _normalize_generic_model_spec("openai/gpt-4.1") == "openai/gpt-4.1"


def test_normalize_generic_model_spec_returns_none_for_blank_input() -> None:
    assert _normalize_generic_model_spec("   ") is None


def test_generic_model_prompt_default_strips_generic_prefix() -> None:
    assert _generic_model_prompt_default("generic.llama3.2") == "llama3.2"


def test_generic_model_prompt_default_ignores_non_generic_provider_spec() -> None:
    assert _generic_model_prompt_default("responses.gpt-5-mini") == "llama3.2"


def test_resolve_model_picker_initial_selection_uses_last_used_alias() -> None:
    provider, model_spec = _resolve_model_picker_initial_selection(
        settings=Settings(
            model_references={
                "system": {
                    "last_used": "claude-haiku-4-5",
                }
            }
        )
    )

    assert provider == "anthropic"
    assert model_spec == "claude-haiku-4-5"


def test_resolve_model_picker_initial_selection_uses_vertex_group_for_anthropic_vertex() -> None:
    provider, model_spec = _resolve_model_picker_initial_selection(
        settings=Settings(
            model_references={
                "system": {
                    "last_used": "anthropic-vertex.claude-sonnet-4-6",
                }
            }
        )
    )

    assert provider == "anthropic-vertex"
    assert model_spec == "anthropic-vertex.claude-sonnet-4-6"


def test_resolve_model_picker_initial_selection_preserves_overlay_alias(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    overlays_dir = env_dir / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "haikutiny.yaml").write_text(
        (
            "name: haikutiny\n"
            "provider: anthropic\n"
            "model: claude-haiku-4-5\n"
            "metadata:\n"
            "  context_window: 8192\n"
            "  max_output_tokens: 1024\n"
        ),
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(tmp_path)
        provider, model_spec = _resolve_model_picker_initial_selection(
            settings=Settings(
                environment_dir=str(env_dir),
                model_references={
                    "system": {
                        "last_used": "haikutiny",
                    }
                },
            )
        )
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    assert provider == "overlays"
    assert model_spec == "haikutiny"


def test_resolve_model_picker_initial_selection_uses_config_relative_overlay_dir(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project_dir = workspace / "project"
    env_dir = project_dir / ".fast-agent"
    config_path = project_dir / "fastagent.config.yaml"
    overlays_dir = env_dir / "model-overlays"
    overlays_dir.mkdir(parents=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("environment_dir: .fast-agent\n", encoding="utf-8")
    (overlays_dir / "haikutiny.yaml").write_text(
        (
            "name: haikutiny\n"
            "provider: anthropic\n"
            "model: claude-haiku-4-5\n"
        ),
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(tmp_path)
        settings = Settings(
            environment_dir=".fast-agent",
            model_references={"system": {"last_used": "haikutiny"}},
        )
        settings._config_file = str(config_path.resolve())

        provider, model_spec = _resolve_model_picker_initial_selection(settings=settings)
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    assert provider == "overlays"
    assert model_spec == "haikutiny"


def test_load_request_settings_refreshes_stale_cached_settings(tmp_path: Path) -> None:
    from fast_agent import config as config_module

    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    env_dir.mkdir(parents=True)
    (env_dir / "fastagent.config.yaml").write_text(
        "model_references:\n"
        "  system:\n"
        "    last_used: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    old_settings = config_module._settings
    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        config_module._settings = Settings(
            model_references={"system": {"last_used": "claude-haiku-4-5"}}
        )
        os.chdir(workspace)
        settings = _load_request_settings(_make_request())
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
        config_module._settings = old_settings

    assert settings.model_references["system"]["last_used"] == "gpt-4.1-mini"
    assert settings._config_file == str((env_dir / "fastagent.config.yaml").resolve())


def test_persist_model_picker_last_used_selection_writes_env_overlay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    env_dir.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        request = _make_request()
        request.environment_dir = env_dir
        settings = Settings(environment_dir=str(env_dir))

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)

    assert persisted is True

    with open(env_dir / "fastagent.config.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"
    assert settings.model_references["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_uses_request_environment_dir(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".custom-env"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        request = _make_request()
        request.environment_dir = env_dir
        settings = Settings()

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)

    assert persisted is True

    with open(env_dir / "fastagent.config.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_uses_runtime_cwd_env_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "nested"
    workspace.mkdir(parents=True)
    nested.mkdir(parents=True)
    (workspace / "fastagent.config.yaml").write_text("default_model: null\n", encoding="utf-8")

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(nested)
        request = _make_request()
        settings = Settings()

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    assert persisted is True
    assert not (workspace / ".fast-agent" / "fastagent.config.yaml").exists()

    with open(nested / ".fast-agent" / "fastagent.config.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_creates_env_overlay_on_first_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        request = _make_request()
        request.environment_dir = env_dir
        settings = Settings(environment_dir=str(env_dir))

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)

    assert persisted is True

    with open(env_dir / "fastagent.config.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_updates_loaded_env_overlay_in_place(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    config_path = env_dir / "fastagent.config.yaml"
    workspace.mkdir(parents=True)
    env_dir.mkdir(parents=True)
    config_path.write_text(
        "model_references:\n"
        "  system:\n"
        "    last_used: google.gemini-3.1-pro-preview\n",
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(workspace)
        request = _make_request()
        settings = _load_request_settings(request)

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    assert persisted is True
    assert not (env_dir / ".fast-agent" / "fastagent.config.yaml").exists()

    with open(config_path, "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_respects_noenv(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        request = _make_request()
        request.environment_dir = env_dir
        request.noenv = True
        settings = Settings(environment_dir=str(env_dir))

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)

    assert persisted is False
    assert not (env_dir / "fastagent.config.yaml").exists()


def test_persist_model_picker_last_used_selection_writes_explicit_config_file(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "config-root"
    workspace = tmp_path / "workspace"
    config_root.mkdir(parents=True)
    workspace.mkdir(parents=True)

    config_path = config_root / "fastagent.config.yaml"
    config_path.write_text("default_model: claude-haiku-4-5\n", encoding="utf-8")

    request = _make_request(config_path=str(config_path))
    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(workspace)
        settings = _load_request_settings(request)
        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
        reloaded = _load_request_settings(request)
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    assert persisted is True
    assert not (workspace / ".fast-agent" / "fastagent.config.yaml").exists()

    with open(config_path, "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"
    assert _resolve_model_picker_initial_selection(settings=reloaded) == (
        "openai",
        "gpt-4.1-mini",
    )


@pytest.mark.asyncio
async def test_run_agent_request_persists_and_reloads_last_used_for_shell_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fast_agent
    from fast_agent import config as config_module

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    request = _make_request()
    request.shell_runtime = True

    async def fake_select_model_from_picker(*args, **kwargs) -> str:
        del args, kwargs
        return "gpt-4.1-mini"

    def fake_emit_model_picker_keyring_notice(*args, **kwargs) -> None:
        del args, kwargs

    class _AbortFastAgent:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("stop-after-persist")

    old_settings = config_module._settings
    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        config_module._settings = Settings(
            model_references={"system": {"last_used": "claude-haiku-4-5"}}
        )
        os.chdir(workspace)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr(
            "fast_agent.cli.runtime.agent_setup._select_model_from_picker",
            fake_select_model_from_picker,
        )
        monkeypatch.setattr(
            "fast_agent.cli.runtime.agent_setup._emit_model_picker_keyring_notice",
            fake_emit_model_picker_keyring_notice,
        )
        monkeypatch.setattr(fast_agent, "FastAgent", _AbortFastAgent)

        with pytest.raises(RuntimeError, match="stop-after-persist"):
            await run_agent_request(request)

        config_module._settings = None
        settings = _load_request_settings(_make_request())
    finally:
        os.chdir(previous_cwd)
        if previous_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
        config_module._settings = old_settings

    config_path = workspace / ".fast-agent" / "fastagent.config.yaml"
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"
    assert _resolve_model_picker_initial_selection(settings=settings) == (
        "openai",
        "gpt-4.1-mini",
    )
