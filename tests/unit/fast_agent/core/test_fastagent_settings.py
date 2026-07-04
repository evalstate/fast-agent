from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.config import get_settings, update_global_settings
from fast_agent.core.fastagent import FastAgent
from fast_agent.paths import resolve_home_paths
from fast_agent.plugins.configuration import installed_plugin_roots
from fast_agent.tools.local_shell_executor import LocalShellExecutor

if TYPE_CHECKING:
    from pathlib import Path


def _write_plugin(plugins_root: Path, name: str) -> None:
    plugin_root = plugins_root / name
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "plugin.yaml").write_text(
        "schema_version: 1\n"
        f"name: {name}\n"
        "description: Test plugin\n"
        "commands:\n"
        f"  {name}:\n"
        "    description: Run test plugin\n"
        "    handler: ./commands.py:run\n",
        encoding="utf-8",
    )
    (plugin_root / "commands.py").write_text(
        "async def run(ctx):\n    return 'ok'\n",
        encoding="utf-8",
    )


def test_fastagent_startup_preserves_global_plugin_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    global_home = tmp_path / "global-home"
    home_root = tmp_path / "project-env"
    _write_plugin(global_home / "plugins", "discover")
    _write_plugin(home_root / "plugins", "images")
    (global_home / "fast-agent.yaml").write_text(
        "plugins:\n  enabled: ['discover']\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nhome: '{home_root.as_posix()}'\n"
        "plugins:\n  enabled: ['images']\n",
        encoding="utf-8",
    )
    old_settings = get_settings()
    monkeypatch.setenv("FAST_AGENT_HOME", global_home.as_posix())
    try:
        fast = FastAgent("test", config_path=str(config_path), parse_cli_args=False)
        settings = get_settings()
        roots = installed_plugin_roots(
            settings,
            project_plugins=resolve_home_paths(settings).plugins,
        )

        assert fast.app._config_or_path is settings
        assert settings._fast_agent_global_plugin_home == global_home.as_posix()
        assert [scope for scope, _root in roots] == ["project", "global"]
        assert sorted((settings.commands or {}).keys()) == ["discover", "images"]
    finally:
        update_global_settings(old_settings)


def test_fastagent_environments_use_instance_settings_and_config_root(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        config_dir = root / ".fast-agent"
        config_dir.mkdir(parents=True)
        (config_dir / "fast-agent.yaml").write_text(
            "default_environment: workspace\n"
            "environments:\n"
            "  workspace:\n"
            "    type: local\n"
            "    cwd: .\n",
            encoding="utf-8",
        )

    old_settings = get_settings()
    try:
        first = FastAgent(
            "first",
            config_path=str(first_root / ".fast-agent" / "fast-agent.yaml"),
            parse_cli_args=False,
        )
        FastAgent(
            "second",
            config_path=str(second_root / ".fast-agent" / "fast-agent.yaml"),
            parse_cli_args=False,
        )

        environment = first.environments.build("workspace")

        assert isinstance(environment, LocalShellExecutor)
        assert environment.working_directory() == first_root
    finally:
        update_global_settings(old_settings)
