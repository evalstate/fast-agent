from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from fast_agent.cards import service as card_service
from fast_agent.cli.commands import cards as cards_command
from fast_agent.cli.commands import plugins as plugins_command
from fast_agent.cli.main import LAZY_SUBCOMMANDS
from fast_agent.cli.main import app as cli_app
from fast_agent.commands.context import CommandContext, NonInteractiveCommandIOBase
from fast_agent.commands.handlers import cards_manager as cards_handlers
from fast_agent.commands.handlers import plugins as plugins_handlers
from fast_agent.config import get_settings, update_global_settings
from fast_agent.paths import resolve_environment_paths

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.command_actions.models import PluginCommandActionSpec
    from fast_agent.commands.results import CommandMessage


class _CapturingIO(NonInteractiveCommandIOBase):
    def __init__(self) -> None:
        self.messages: list[CommandMessage] = []

    async def emit(self, message: CommandMessage) -> None:
        self.messages.append(message)


class _Provider:
    def __init__(self) -> None:
        self.plugin_commands: dict[str, PluginCommandActionSpec] | None = None
        self.plugin_command_base_path: Path | None = None

    def set_plugin_commands(
        self,
        commands: dict[str, PluginCommandActionSpec] | None,
        *,
        base_path: Path | None,
    ) -> None:
        self.plugin_commands = commands
        self.plugin_command_base_path = base_path

    def _agent(self, name: str):
        raise KeyError(name)

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "main"

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["main"]

    def registered_agent_names(self):
        return ["main"]

    def registered_agents(self):
        return {}

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Test User")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _write_plugin(
    repo: Path,
    name: str,
    message: str = "ok",
    *,
    manifest_name: str | None = None,
) -> None:
    plugin_name = manifest_name or name
    plugin_root = repo / "plugins" / name
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "plugin.yaml").write_text(
        "schema_version: 1\n"
        f"name: {plugin_name}\n"
        "description: Test plugin\n"
        "commands:\n"
        f"  {plugin_name}:\n"
        "    description: Run test plugin\n"
        "    handler: ./commands.py:run\n",
        encoding="utf-8",
    )
    (plugin_root / "commands.py").write_text(
        f"async def run(ctx):\n    return {message!r}\n",
        encoding="utf-8",
    )


def _write_versioned_plugin(
    plugins_root: Path,
    name: str,
    *,
    version: str | None,
    command: str | None = None,
) -> Path:
    """Write a plugin directly into a plugins directory, optionally versioned."""
    plugin_root = plugins_root / name
    plugin_root.mkdir(parents=True, exist_ok=True)
    cmd = command or name
    lines = ["schema_version: 1", f"name: {name}"]
    if version is not None:
        lines.append(f"version: '{version}'")
    lines += [
        "description: Test plugin",
        "commands:",
        f"  {cmd}:",
        "    description: Run test plugin",
        "    handler: ./commands.py:run",
    ]
    (plugin_root / "plugin.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (plugin_root / "commands.py").write_text(
        "async def run(ctx):\n    return 'ok'\n",
        encoding="utf-8",
    )
    return plugin_root


def _write_marketplace(path: Path, repo: Path, *, include_pack: bool = False) -> None:
    payload: dict[str, object] = {
        "command_plugins": [
            {
                "name": "finder",
                "repo_url": repo.as_posix(),
                "repo_path": "plugins/finder",
            }
        ]
    }
    if include_pack:
        payload["entries"] = [
            {
                "name": "alpha",
                "kind": "card",
                "repo_url": repo.as_posix(),
                "repo_path": "packs/alpha",
            }
        ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_plugin_marketplace(
    path: Path,
    repo: Path,
    *,
    name: str,
    repo_path: str,
    repo_ref: str | None = None,
) -> None:
    entry = {
        "name": name,
        "repo_url": repo.as_posix(),
        "repo_path": repo_path,
    }
    if repo_ref is not None:
        entry["repo_ref"] = repo_ref
    path.write_text(json.dumps({"command_plugins": [entry]}), encoding="utf-8")


def _write_pack_requiring_plugin(
    repo: Path,
    *,
    required_plugin: str,
) -> None:
    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True, exist_ok=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "card-pack.yaml").write_text(
        "schema_version: 2\n"
        "name: alpha\n"
        "kind: card\n"
        "install:\n"
        "  agent_cards: ['agent-cards/alpha.md']\n"
        "  tool_cards: []\n"
        "  files: []\n"
        "plugins:\n"
        f"  required: ['{required_plugin}']\n",
        encoding="utf-8",
    )


def _write_pack_without_plugins(repo: Path) -> None:
    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True, exist_ok=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "card-pack.yaml").write_text(
        "schema_version: 1\n"
        "name: alpha\n"
        "kind: card\n"
        "install:\n"
        "  agent_cards: ['agent-cards/alpha.md']\n"
        "  tool_cards: []\n"
        "  files: []\n",
        encoding="utf-8",
    )


def _write_marketplace_with_pack_and_plugin(
    path: Path,
    repo: Path,
    *,
    plugin_name: str,
    plugin_path: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ],
                "command_plugins": [
                    {
                        "name": plugin_name,
                        "repo_url": repo.as_posix(),
                        "repo_path": plugin_path,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_marketplace_with_pack_source_url_and_plugin(
    path: Path,
    repo: Path,
    *,
    plugin_name: str,
    plugin_path: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                        "source_url": (repo / "packs" / "alpha").as_posix(),
                    }
                ],
                "command_plugins": [
                    {
                        "name": plugin_name,
                        "repo_url": repo.as_posix(),
                        "repo_path": plugin_path,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_plugins_lazy_subcommand_registered() -> None:
    assert LAZY_SUBCOMMANDS["plugins"] == "fast_agent.cli.commands.plugins:app"


def test_plugins_add_enables_and_loads_commands(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace(marketplace_path, repo)

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            plugins_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "finder"],
        )
        assert result.exit_code == 0, result.output
        assert "Plugin Installed" in result.output
        assert (env_root / "plugins" / "finder" / "plugin.yaml").exists()
        assert "finder" in config_path.read_text(encoding="utf-8")

        settings = get_settings(config_path=str(config_path))
        assert settings.commands is not None
        assert settings.commands["finder"].handler.endswith("/plugins/finder/commands.py:run")
    finally:
        update_global_settings(old_settings)


def test_plugins_list_shows_project_and_global_plugins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    global_home = tmp_path / "global-home"
    env_root = tmp_path / "project-env"
    _write_plugin(global_home, "global-finder")
    _write_plugin(env_root, "project-helper")
    (global_home / "fast-agent.yaml").write_text(
        "plugins:\n  enabled: ['global-finder']\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n  enabled: ['project-helper']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_AGENT_HOME", global_home.as_posix())

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(plugins_command.app, ["list"])

        assert result.exit_code == 0, result.output
        assert "project plugins directory" in result.output
        assert "global plugins directory" in result.output
        assert "project-helper" in result.output
        assert "global-finder" in result.output
    finally:
        update_global_settings(old_settings)


def test_plugins_list_marks_active_duplicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    global_home = tmp_path / "global-home"
    env_root = tmp_path / "project-env"
    _write_versioned_plugin(global_home / "plugins", "shared", version="1.0.0")
    _write_versioned_plugin(env_root / "plugins", "shared", version="2.0.0")
    (global_home / "fast-agent.yaml").write_text(
        "plugins:\n  enabled: ['shared']\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n  enabled: ['shared']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_AGENT_HOME", global_home.as_posix())

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(plugins_command.app, ["list"])

        assert result.exit_code == 0, result.output
        project_row = result.output.index("project")
        global_row = result.output.index("global")
        assert project_row < global_row
        assert "active" in result.output
        assert "shadowed by project" in result.output
    finally:
        update_global_settings(old_settings)


def test_plugins_remove_without_selector_shows_only_project_plugins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    global_home = tmp_path / "global-home"
    env_root = tmp_path / "project-env"
    _write_plugin(global_home, "global-finder")
    _write_plugin(env_root, "project-helper")
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_AGENT_HOME", global_home.as_posix())

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(plugins_command.app, ["remove"])

        assert result.exit_code == 0, result.output
        assert "project plugins directory" in result.output
        assert "global plugins directory" not in result.output
        assert "project-helper" in result.output
        assert "global-finder" not in result.output
    finally:
        update_global_settings(old_settings)


def test_plugins_remove_global_without_selector_shows_only_global_plugins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    global_home = tmp_path / "global-home"
    env_root = tmp_path / "project-env"
    _write_plugin(global_home, "global-finder")
    _write_plugin(env_root, "project-helper")
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_AGENT_HOME", global_home.as_posix())

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(plugins_command.app, ["remove", "--global"])

        assert result.exit_code == 0, result.output
        assert "global plugins directory" in result.output
        assert "project plugins directory" not in result.output
        assert "global-finder" in result.output
        assert "project-helper" not in result.output
    finally:
        update_global_settings(old_settings)


def test_plugins_add_honors_top_level_env_for_install_and_registry_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace(marketplace_path, repo)

    env_root = tmp_path / "custom-fast-agent"
    env_root.mkdir()
    config_path = env_root / "fast-agent.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "plugins:\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    old_settings = get_settings()
    try:
        result = CliRunner().invoke(
            cli_app,
            [
                "--no-update-check",
                "--env",
                env_root.as_posix(),
                "plugins",
                "add",
                "finder",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (env_root / "plugins" / "finder" / "plugin.yaml").exists()
        assert not (tmp_path / ".fast-agent" / "plugins" / "finder" / "plugin.yaml").exists()
        assert "finder" in config_path.read_text(encoding="utf-8")
    finally:
        update_global_settings(old_settings)


def test_plugins_load_enabled_by_manifest_name_when_directory_differs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder-plugin", manifest_name="finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_plugin_marketplace(
        marketplace_path,
        repo,
        name="finder",
        repo_path="plugins/finder-plugin",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            plugins_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "finder"],
        )
        assert result.exit_code == 0, result.output
        assert (env_root / "plugins" / "finder-plugin" / "plugin.yaml").exists()
        assert "finder" in config_path.read_text(encoding="utf-8")

        settings = get_settings(config_path=str(config_path))

        assert settings.commands is not None
        assert settings.commands["finder"].handler.endswith(
            "/plugins/finder-plugin/commands.py:run"
        )
    finally:
        update_global_settings(old_settings)


def test_plugins_local_repo_ref_installs_requested_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder", "stable")
    _commit_all(repo, "stable")
    stable_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "stable")

    _write_plugin(repo, "finder", "current")
    _commit_all(repo, "current")
    assert _git(repo, "rev-parse", "HEAD") != stable_commit

    marketplace_path = tmp_path / "marketplace.json"
    _write_plugin_marketplace(
        marketplace_path,
        repo,
        name="finder",
        repo_path="plugins/finder",
        repo_ref="stable",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            plugins_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "finder"],
        )

        assert result.exit_code == 0, result.output
        installed_command = env_root / "plugins" / "finder" / "commands.py"
        assert "stable" in installed_command.read_text(encoding="utf-8")
        assert "current" not in installed_command.read_text(encoding="utf-8")
    finally:
        update_global_settings(old_settings)


@pytest.mark.asyncio
async def test_plugins_slash_add_list_and_remove(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace(marketplace_path, repo)

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    settings = get_settings(config_path=str(config_path))
    provider = _Provider()
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="main",
        io=_CapturingIO(),
        settings=settings,
    )
    try:
        add_outcome = await plugins_handlers.handle_plugins_command(
            ctx,
            agent_name="main",
            action="add",
            argument="finder",
        )
        assert add_outcome.messages
        assert (env_root / "plugins" / "finder" / "plugin.yaml").exists()
        assert "finder" in config_path.read_text(encoding="utf-8")
        assert provider.plugin_commands is not None
        assert "finder" in provider.plugin_commands
        assert provider.plugin_command_base_path == config_path.parent

        list_outcome = await plugins_handlers.handle_plugins_command(
            ctx,
            agent_name="main",
            action="list",
            argument=None,
        )
        rendered = "\n".join(str(message.text) for message in list_outcome.messages)
        assert "commands: finder" in rendered

        remove_outcome = await plugins_handlers.handle_plugins_command(
            ctx,
            agent_name="main",
            action="remove",
            argument="finder",
        )
        assert remove_outcome.messages
        assert not (env_root / "plugins" / "finder").exists()
        assert "finder" not in config_path.read_text(encoding="utf-8")
    finally:
        update_global_settings(old_settings)


@pytest.mark.asyncio
async def test_plugins_slash_list_shows_both_scopes_active_marker_and_version_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # `shared` lives in both scopes; global is older (1.0.0), project newer (2.0.0).
    # `discover` is global-only. Both scopes enable `shared`; only global enables `discover`.
    global_home = tmp_path / "global-home"
    env_root = tmp_path / "project-env"
    _write_versioned_plugin(global_home / "plugins", "shared", version="1.0.0")
    _write_versioned_plugin(env_root / "plugins", "shared", version="2.0.0")
    _write_versioned_plugin(global_home / "plugins", "discover", version=None)

    (global_home / "fast-agent.yaml").write_text(
        "plugins:\n  enabled: ['shared', 'discover']\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n  enabled: ['shared']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_AGENT_HOME", global_home.as_posix())

    old_settings = get_settings()
    settings = get_settings(config_path=str(config_path))
    provider = _Provider()
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="main",
        io=_CapturingIO(),
        settings=settings,
    )
    try:
        outcome = await plugins_handlers.handle_plugins_command(
            ctx,
            agent_name="main",
            action="list",
            argument=None,
        )
        rendered = "\n".join(str(message.text) for message in outcome.messages)

        # Both scopes are surfaced (mirrors the CLI listing).
        assert "project plugins directory" in rendered
        assert "global plugins directory" in rendered
        # Both copies of the shared name, plus the global-only plugin, are shown.
        assert "shared" in rendered
        assert "discover" in rendered
        # Project copy wins (override precedence) → marked active; global copy shadowed.
        assert "active" in rendered
        assert "shadowed by project" in rendered
        # Version callout across scopes.
        assert "2.0.0 is newer than global (1.0.0)" in rendered
        assert "1.0.0 is older than project (2.0.0)" in rendered
    finally:
        update_global_settings(old_settings)


@pytest.mark.asyncio
async def test_plugins_slash_list_shows_session_active_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fast_agent.command_actions import PluginCommandActionSpec

    env_root = tmp_path / "project-env"
    plugin_root = _write_versioned_plugin(
        env_root / "plugins",
        "catalog",
        version=None,
        command="discover",
    )
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n  enabled: ['catalog']\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)

    old_settings = get_settings()
    settings = get_settings(config_path=str(config_path))
    provider = _Provider()
    provider.plugin_commands = {
        "discover": PluginCommandActionSpec(
            name="discover",
            description="Discover agents",
            handler=f"{(plugin_root / 'commands.py').as_posix()}:run",
        )
    }
    provider.plugin_command_base_path = config_path.parent
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="main",
        io=_CapturingIO(),
        settings=settings,
    )
    try:
        outcome = await plugins_handlers.handle_plugins_command(
            ctx,
            agent_name="main",
            action="list",
            argument=None,
        )
        rendered = "\n".join(str(message.text) for message in outcome.messages)

        assert "catalog" in rendered
        assert "active commands: /discover" in rendered
    finally:
        update_global_settings(old_settings)


def test_plugins_update_reinstalls_managed_plugin(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder", "old")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace(marketplace_path, repo)

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()
        add_result = runner.invoke(
            plugins_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "finder"],
        )
        assert add_result.exit_code == 0, add_result.output

        _write_plugin(repo, "finder", "new")
        _commit_all(repo, "update")

        check_result = runner.invoke(plugins_command.app, ["update"])
        assert check_result.exit_code == 0, check_result.output
        assert "plugin content changed" in check_result.output

        update_result = runner.invoke(plugins_command.app, ["update", "all", "--yes"])
        assert update_result.exit_code == 0, update_result.output
        assert "updated" in update_result.output
        assert "new" in (env_root / "plugins" / "finder" / "commands.py").read_text(
            encoding="utf-8"
        )
    finally:
        update_global_settings(old_settings)


def test_plugin_global_install_defaults_to_user_fast_agent_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace(marketplace_path, repo)
    user_home = tmp_path / "user-home"
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.setenv("HOME", user_home.as_posix())

    result = CliRunner().invoke(
        plugins_command.app,
        ["--registry", marketplace_path.as_posix(), "add", "finder", "--global"],
    )

    assert result.exit_code == 0, result.output
    assert (user_home / ".fast-agent" / "plugins" / "finder" / "plugin.yaml").exists()
    assert "finder" in (user_home / ".fast-agent" / "fast-agent.yaml").read_text(encoding="utf-8")


def test_card_pack_schema_v2_installs_required_plugins(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "card-pack.yaml").write_text(
        "schema_version: 2\n"
        "name: alpha\n"
        "kind: card\n"
        "install:\n"
        "  agent_cards: ['agent-cards/alpha.md']\n"
        "  tool_cards: []\n"
        "  files: []\n"
        "plugins:\n"
        "  required: ['finder']\n",
        encoding="utf-8",
    )
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace(marketplace_path, repo, include_pack=True)

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            cards_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "alpha"],
        )
        assert result.exit_code == 0, result.output
        assert (env_root / "plugins" / "finder" / "plugin.yaml").exists()
        assert "finder" in config_path.read_text(encoding="utf-8")
    finally:
        update_global_settings(old_settings)


def test_card_pack_required_plugin_uses_selected_card_registry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _write_pack_requiring_plugin(repo, required_plugin="finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace_with_pack_and_plugin(
        marketplace_path,
        repo,
        plugin_name="finder",
        plugin_path="plugins/finder",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            cards_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "alpha"],
        )
        assert result.exit_code == 0, result.output
        assert (env_root / "plugins" / "finder" / "plugin.yaml").exists()
        assert "finder" in config_path.read_text(encoding="utf-8")
    finally:
        update_global_settings(old_settings)


def test_card_pack_install_rolls_back_when_required_plugin_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _write_pack_requiring_plugin(repo, required_plugin="missing-plugin")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace_with_pack_and_plugin(
        marketplace_path,
        repo,
        plugin_name="finder",
        plugin_path="plugins/finder",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            cards_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "alpha"],
        )
        assert result.exit_code == 1, result.output
        assert "Required plugin not found" in result.output
        assert not (env_root / "card-packs" / "alpha").exists()
        assert not (env_root / "agent-cards" / "alpha.md").exists()
    finally:
        update_global_settings(old_settings)


@pytest.mark.asyncio
async def test_cards_add_refreshes_provider_plugin_commands(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _write_pack_requiring_plugin(repo, required_plugin="finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace_with_pack_and_plugin(
        marketplace_path,
        repo,
        plugin_name="finder",
        plugin_path="plugins/finder",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n"
        "cards:\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    settings = get_settings(config_path=str(config_path))
    provider = _Provider()
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="main",
        io=_CapturingIO(),
        settings=settings,
    )
    try:
        outcome = await cards_handlers.handle_cards_command(
            ctx,
            agent_name="main",
            action="add",
            argument="alpha",
        )

        rendered = "\n".join(str(message.text) for message in outcome.messages)
        assert "Installed card pack" in rendered
        assert provider.plugin_commands is not None
        assert "finder" in provider.plugin_commands
        assert provider.plugin_command_base_path == config_path.parent
    finally:
        update_global_settings(old_settings)


def test_card_pack_required_plugin_uses_marketplace_source_not_pack_source_url(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _write_pack_requiring_plugin(repo, required_plugin="finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace_with_pack_source_url_and_plugin(
        marketplace_path,
        repo,
        plugin_name="finder",
        plugin_path="plugins/finder",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            cards_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "alpha"],
        )
        assert result.exit_code == 0, result.output
        assert (env_root / "plugins" / "finder" / "plugin.yaml").exists()

        source, error = card_service.manager.read_installed_card_pack_source(
            env_root / "card-packs" / "alpha"
        )
        assert error is None
        assert source is not None
        assert source.source_url == marketplace_path.as_posix()
    finally:
        update_global_settings(old_settings)


def test_card_pack_update_required_plugin_uses_pack_source_registry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder")
    _write_pack_without_plugins(repo)
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace_with_pack_and_plugin(
        marketplace_path,
        repo,
        plugin_name="finder",
        plugin_path="plugins/finder",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()
        add_result = runner.invoke(
            cards_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "alpha"],
        )
        assert add_result.exit_code == 0, add_result.output
        assert not (env_root / "plugins" / "finder" / "plugin.yaml").exists()

        _write_pack_requiring_plugin(repo, required_plugin="finder")
        _commit_all(repo, "require plugin")

        update_result = runner.invoke(cards_command.app, ["update", "all", "--yes"])
        assert update_result.exit_code == 0, update_result.output
        assert (env_root / "plugins" / "finder" / "plugin.yaml").exists()
        assert "finder" in config_path.read_text(encoding="utf-8")
    finally:
        update_global_settings(old_settings)


def test_card_pack_required_plugin_enables_manifest_name(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder-plugin", manifest_name="finder")
    _write_pack_requiring_plugin(repo, required_plugin="agent-finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace_with_pack_and_plugin(
        marketplace_path,
        repo,
        plugin_name="agent-finder",
        plugin_path="plugins/finder-plugin",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            cards_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "alpha"],
        )
        assert result.exit_code == 0, result.output
        assert (env_root / "plugins" / "finder-plugin" / "plugin.yaml").exists()

        config_text = config_path.read_text(encoding="utf-8")
        assert "finder" in config_text
        assert "agent-finder" not in config_text

        settings = get_settings(config_path=str(config_path))
        assert settings.commands is not None
        assert settings.commands["finder"].handler.endswith(
            "/plugins/finder-plugin/commands.py:run"
        )
    finally:
        update_global_settings(old_settings)


@pytest.mark.asyncio
async def test_async_card_pack_required_plugin_enables_manifest_name(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_plugin(repo, "finder-plugin", manifest_name="finder")
    _write_pack_requiring_plugin(repo, required_plugin="agent-finder")
    _commit_all(repo, "initial")
    marketplace_path = tmp_path / "marketplace.json"
    _write_marketplace_with_pack_and_plugin(
        marketplace_path,
        repo,
        plugin_name="agent-finder",
        plugin_path="plugins/finder-plugin",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    settings = get_settings(config_path=str(config_path))
    try:
        await card_service.install_pack(
            marketplace_path.as_posix(),
            "alpha",
            environment_paths=resolve_environment_paths(settings),
            force=False,
        )

        config_text = config_path.read_text(encoding="utf-8")
        assert "finder" in config_text
        assert "agent-finder" not in config_text
    finally:
        update_global_settings(old_settings)
