from pathlib import Path

import pytest

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.plugins.manifest import load_plugin_manifest


def test_plugin_manifest_normalizes_text_fields(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        "schema_version: 1\n"
        "name: '  tidy-plugin  '\n"
        "version: '  1.2.3  '\n"
        "description: '  Useful plugin  '\n",
        encoding="utf-8",
    )

    manifest = load_plugin_manifest(plugin_dir)

    assert manifest.name == "tidy-plugin"
    assert manifest.version == "1.2.3"
    assert manifest.description == "Useful plugin"


def test_plugin_manifest_normalizes_blank_optional_text_to_none(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        "schema_version: 1\nname: tidy-plugin\nversion: '   '\ndescription: ''\n",
        encoding="utf-8",
    )

    manifest = load_plugin_manifest(plugin_dir)

    assert manifest.version is None
    assert manifest.description is None


def test_plugin_manifest_rejects_handler_escaping_plugin_root(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        "schema_version: 1\n"
        "name: unsafe\n"
        "commands:\n"
        "  run:\n"
        "    description: Run unsafe handler\n"
        "    handler: ../outside.py:run\n",
        encoding="utf-8",
    )
    (tmp_path / "outside.py").write_text("async def run(ctx):\n    return 'no'\n", encoding="utf-8")

    with pytest.raises(AgentConfigError, match="escapes plugin root"):
        load_plugin_manifest(plugin_dir)


def test_plugin_manifest_rejects_absolute_handler_path(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("async def run(ctx):\n    return 'no'\n", encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(
        "schema_version: 1\n"
        "name: unsafe\n"
        "commands:\n"
        "  run:\n"
        "    description: Run unsafe handler\n"
        f"    handler: {outside}:run\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="must be relative"):
        load_plugin_manifest(plugin_dir)
