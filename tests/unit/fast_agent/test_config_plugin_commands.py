from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.config import get_settings

if TYPE_CHECKING:
    from pathlib import Path


def test_settings_parses_global_plugin_commands(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "commands:",
                "  draft-next:",
                "    description: Draft the next user message",
                "    input_hint: \"[format]\"",
                "    handler: \"commands.py:draft_next\"",
                "    key: \"c-x d\"",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.commands is not None
    assert settings.commands["draft-next"].description == "Draft the next user message"
    assert settings.commands["draft-next"].handler == "commands.py:draft_next"
    assert settings.commands["draft-next"].input_hint == "[format]"
    assert settings.commands["draft-next"].key == "c-x d"


def test_settings_merges_fast_agent_home_plugins(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project_env = tmp_path / "project-env"
    home_plugin = home / "plugins" / "global-finder"
    project_plugin = project_env / "plugins" / "project-helper"
    home_plugin.mkdir(parents=True)
    project_plugin.mkdir(parents=True)
    for plugin_dir, command_name in (
        (home_plugin, "global-finder"),
        (project_plugin, "project-helper"),
    ):
        (plugin_dir / "plugin.yaml").write_text(
            "schema_version: 1\n"
            f"name: {command_name}\n"
            "commands:\n"
            f"  {command_name}:\n"
            "    description: Test command\n"
            "    handler: ./commands.py:run\n",
            encoding="utf-8",
        )
        (plugin_dir / "commands.py").write_text(
            "async def run(ctx):\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )
    home.mkdir(exist_ok=True)
    (home / "fast-agent.yaml").write_text(
        "plugins:\n"
        "  enabled: ['global-finder']\n",
        encoding="utf-8",
    )
    project.mkdir()
    config_path = project / "fast-agent.yaml"
    config_path.write_text(
        f"environment_dir: '{project_env.as_posix()}'\n"
        "plugins:\n"
        "  enabled: ['project-helper']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_AGENT_HOME", home.as_posix())
    monkeypatch.chdir(project)

    settings = get_settings(config_path)

    assert settings.plugins.enabled == ["global-finder", "project-helper"]
    assert settings.commands is not None
    assert set(settings.commands) == {"global-finder", "project-helper"}
    assert settings.commands["global-finder"].handler.endswith(
        "/home/plugins/global-finder/commands.py:run"
    )
    assert settings.commands["project-helper"].handler.endswith(
        "/project-env/plugins/project-helper/commands.py:run"
    )


def test_missing_home_plugin_does_not_drop_project_plugins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project_env = tmp_path / "project-env"
    project_plugin = project_env / "plugins" / "project-helper"
    project_plugin.mkdir(parents=True)
    (project_plugin / "plugin.yaml").write_text(
        "schema_version: 1\n"
        "name: project-helper\n"
        "commands:\n"
        "  project-helper:\n"
        "    description: Test command\n"
        "    handler: ./commands.py:run\n",
        encoding="utf-8",
    )
    (project_plugin / "commands.py").write_text(
        "async def run(ctx):\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    home.mkdir()
    (home / "fast-agent.yaml").write_text(
        "plugins:\n"
        "  enabled: ['missing-global']\n",
        encoding="utf-8",
    )
    project.mkdir()
    config_path = project / "fast-agent.yaml"
    config_path.write_text(
        f"environment_dir: '{project_env.as_posix()}'\n"
        "plugins:\n"
        "  enabled: ['project-helper']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_AGENT_HOME", home.as_posix())
    monkeypatch.chdir(project)

    with pytest.warns(UserWarning, match="missing-global"):
        settings = get_settings(config_path)

    assert settings.commands is not None
    assert set(settings.commands) == {"project-helper"}
