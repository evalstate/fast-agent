from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from click.utils import strip_ansi
from rich.text import Text
from typer.testing import CliRunner

import fast_agent.config as config_module
from fast_agent.cards.manager import (
    CARD_PACK_PUBLISH_FAILURE_STATUSES,
    CARD_PACK_PUBLISH_STATUS_LABELS,
    CARD_PACK_PUBLISH_STATUS_STYLES,
    CARD_PACK_PUBLISH_SUCCESS_STATUSES,
    CARD_PACK_PUBLISH_WARNING_STATUSES,
    CardPackPublishResult,
    CardPackPublishStatus,
    CardPackRemovalResult,
    format_card_pack_publish_status,
    is_card_pack_publish_failure,
    is_card_pack_publish_success,
)
from fast_agent.cli.commands import cards as cards_command
from fast_agent.cli.main import LAZY_SUBCOMMANDS
from fast_agent.commands.command_catalog import normalize_command_action
from fast_agent.commands.handlers import cards_manager as cards_handlers
from fast_agent.config import get_settings, update_global_settings
from fast_agent.constants import FAST_AGENT_RUNTIME_ENVIRONMENT

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


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


def test_cards_lazy_subcommand_registered() -> None:
    assert LAZY_SUBCOMMANDS["cards"] == "fast_agent.cli.commands.cards:app"


def test_cards_action_aliases_normalize_to_canonical_actions() -> None:
    cases = {
        None: "list",
        "": "list",
        "install": "add",
        " INSTALL ": "add",
        "marketplace": "registry",
        "source": "registry",
        "rm": "remove",
        "delete": "remove",
        "uninstall": "remove",
        "show": "readme",
        "cat": "readme",
        "refresh": "update",
        "upgrade": "update",
        "unexpected": "unexpected",
    }

    for action, expected in cases.items():
        assert normalize_command_action("cards", action) == expected


def test_cards_publish_failure_warning_distinguishes_patch_output(tmp_path: Path) -> None:
    base_result = CardPackPublishResult(
        pack_name="alpha",
        pack_dir=tmp_path,
        status="publish_failed",
    )
    patch_result = CardPackPublishResult(
        pack_name="alpha",
        pack_dir=tmp_path,
        status="publish_failed",
        patch_path=tmp_path / "alpha.patch",
    )

    assert cards_handlers._publish_failure_warning(base_result) == (
        "Publish failed after committing locally. Push manually or ask a maintainer with write access."
    )
    assert cards_handlers._publish_failure_warning(patch_result) == (
        "Push was rejected. Share the generated patch with a maintainer or open a PR from your branch."
    )


def test_cards_publish_failure_warning_ignores_success(tmp_path: Path) -> None:
    result = CardPackPublishResult(
        pack_name="alpha",
        pack_dir=tmp_path,
        status="published",
    )

    assert cards_handlers._publish_failure_warning(result) is None


def test_card_pack_publish_status_formatting_is_shared(tmp_path: Path) -> None:
    committed = CardPackPublishResult(
        pack_name="alpha",
        pack_dir=tmp_path,
        status="committed",
    )
    unreachable = CardPackPublishResult(
        pack_name="alpha",
        pack_dir=tmp_path,
        status="source_unreachable",
        detail="git failed",
    )

    assert format_card_pack_publish_status(committed) == (
        "committed locally",
        "green",
    )
    assert cards_handlers._publish_status_text(unreachable).text == (
        "source unavailable: git failed"
    )


def test_card_pack_publish_status_tables_cover_declared_statuses() -> None:
    statuses: set[CardPackPublishStatus] = {
        "published",
        "committed",
        "no_changes",
        "unmanaged",
        "invalid_metadata",
        "source_unreachable",
        "source_path_missing",
        "missing_managed_files",
        "publish_failed",
    }

    assert set(CARD_PACK_PUBLISH_STATUS_LABELS) == statuses
    assert CARD_PACK_PUBLISH_SUCCESS_STATUSES <= statuses
    assert CARD_PACK_PUBLISH_FAILURE_STATUSES <= statuses
    assert CARD_PACK_PUBLISH_WARNING_STATUSES == (
        statuses - CARD_PACK_PUBLISH_SUCCESS_STATUSES - {"unmanaged"}
    )
    assert set(CARD_PACK_PUBLISH_STATUS_STYLES) == (
        CARD_PACK_PUBLISH_SUCCESS_STATUSES | CARD_PACK_PUBLISH_WARNING_STATUSES
    )


def test_cards_install_handler_formats_managed_file_count(tmp_path: Path) -> None:
    rendered = cards_handlers._format_install_result(
        pack_name="alpha",
        install_path=tmp_path,
        installed_files=("one.md", "two.md"),
    ).plain

    assert "managed files: 2 files" in rendered


def test_cards_remove_formatter_omits_skipped_line_when_empty() -> None:
    rendered = cards_handlers._format_remove_result(
        pack_name="alpha",
        skipped_paths=(),
    ).plain

    assert rendered == "Removed card pack: alpha"


def test_cards_remove_formatter_pluralizes_skipped_paths() -> None:
    rendered = cards_handlers._format_remove_result(
        pack_name="alpha",
        skipped_paths=("shared.md", "shared-tool.md"),
    ).plain

    assert "Skipped 2 paths with shared ownership." in rendered
    assert "path(s)" not in rendered


@pytest.mark.parametrize(
    ("status", "success", "failure"),
    [
        ("published", True, False),
        ("committed", True, False),
        ("no_changes", True, False),
        ("unmanaged", False, False),
        ("invalid_metadata", False, False),
        ("source_unreachable", False, False),
        ("source_path_missing", False, False),
        ("missing_managed_files", False, False),
        ("publish_failed", False, True),
    ],
)
def test_card_pack_publish_status_classifiers(
    status: CardPackPublishStatus,
    success: bool,
    failure: bool,
) -> None:
    assert is_card_pack_publish_success(status) is success
    assert is_card_pack_publish_failure(status) is failure


@pytest.mark.asyncio
async def test_cards_remove_handler_reports_skipped_paths_without_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_paths = object()

    def resolve_paths(_settings: object) -> object:
        return env_paths

    def list_packs(*, environment_paths: object) -> list[SimpleNamespace]:
        assert environment_paths is env_paths
        return [SimpleNamespace(name="alpha")]

    def remove_pack(*, environment_paths: object, selector: str) -> CardPackRemovalResult:
        assert environment_paths is env_paths
        assert selector == "alpha"
        return CardPackRemovalResult(
            pack_name="alpha",
            removed_paths=("agent-cards/alpha.md",),
            skipped_paths=("shared.md", "shared-tool.md"),
        )

    monkeypatch.setattr(cards_handlers, "resolve_environment_paths", resolve_paths)
    monkeypatch.setattr(cards_handlers.card_service, "list_installed_packs", list_packs)
    monkeypatch.setattr(cards_handlers.card_service, "remove_pack", remove_pack)

    ctx = cast(
        "CommandContext",
        SimpleNamespace(resolve_settings=lambda: SimpleNamespace(_config_file=None)),
    )
    outcome = await cards_handlers.handle_remove_card_pack(
        ctx,
        agent_name="agent",
        argument="alpha",
    )

    message = outcome.messages[-1].text
    text = message.plain if isinstance(message, Text) else str(message)
    assert "Skipped 2 paths with shared ownership." in text
    assert "path(s)" not in text


@pytest.mark.asyncio
async def test_cards_add_handler_parses_registry_and_force_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_paths = object()
    pack = SimpleNamespace(name="alpha")
    captured: dict[str, object] = {}

    def resolve_paths(_settings: object) -> object:
        return env_paths

    async def scan_marketplace(source: str) -> SimpleNamespace:
        captured["source"] = source
        return SimpleNamespace(source="resolved-marketplace", packs=[pack])

    def select_marketplace_pack(packs: list[object], selector: str) -> object:
        captured["selector"] = selector
        assert packs == [pack]
        return pack

    async def install_selected_pack(
        selected_pack: object,
        *,
        environment_paths: object,
        force: bool,
        marketplace_source: str | None = None,
    ) -> SimpleNamespace:
        captured["selected_pack"] = selected_pack
        captured["environment_paths"] = environment_paths
        captured["force"] = force
        captured["marketplace_source"] = marketplace_source
        return SimpleNamespace(
            pack=pack,
            install_result=SimpleNamespace(
                pack_dir=tmp_path / "alpha",
                installed_files=("alpha.md",),
            ),
            readme=None,
        )

    monkeypatch.setattr(cards_handlers, "resolve_environment_paths", resolve_paths)
    monkeypatch.setattr(cards_handlers, "_config_path_for_settings", lambda _ctx: tmp_path / "config.yaml")
    monkeypatch.setattr(cards_handlers, "_refresh_provider_plugins", lambda *_args: None)
    monkeypatch.setattr(cards_handlers.card_service, "scan_marketplace", scan_marketplace)
    monkeypatch.setattr(
        cards_handlers.card_service,
        "select_marketplace_pack",
        select_marketplace_pack,
    )
    monkeypatch.setattr(
        cards_handlers.card_service,
        "install_selected_pack",
        install_selected_pack,
    )

    ctx = cast(
        "CommandContext",
        SimpleNamespace(resolve_settings=lambda: SimpleNamespace(_config_file=None)),
    )
    outcome = await cards_handlers.handle_add_card_pack(
        ctx,
        agent_name="agent",
        argument="alpha --registry custom-marketplace --force",
    )

    assert all(message.channel != "error" for message in outcome.messages)
    assert captured == {
        "source": "custom-marketplace",
        "selector": "alpha",
        "selected_pack": pack,
        "environment_paths": env_paths,
        "force": True,
        "marketplace_source": "resolved-marketplace",
    }


@pytest.mark.asyncio
async def test_cards_registry_rejects_empty_marketplace_without_switching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(cards=SimpleNamespace(marketplace_url="current-marketplace"))

    async def scan_marketplace(source: str) -> SimpleNamespace:
        assert source == "empty-marketplace"
        return SimpleNamespace(source="resolved-marketplace", packs=[])

    monkeypatch.setattr(cards_handlers, "resolve_card_registries", lambda _settings: [])
    monkeypatch.setattr(cards_handlers.card_service, "scan_marketplace", scan_marketplace)

    ctx = cast(
        "CommandContext",
        SimpleNamespace(resolve_settings=lambda: settings),
    )

    outcome = await cards_handlers.handle_set_cards_registry(
        ctx,
        argument="empty-marketplace",
    )

    assert settings.cards.marketplace_url == "current-marketplace"
    assert outcome.messages[0].channel == "warning"
    message_text = outcome.messages[0].text
    plain = message_text.plain if isinstance(message_text, Text) else str(message_text)
    assert "registry unchanged" in plain


def test_cards_add_and_remove_via_cli(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "readme.md").write_text(
        "# Alpha Pack\n\nInstall notes.\n",
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
    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n",
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
        assert "Card Pack Installed" in add_result.output
        assert "name: alpha" in add_result.output
        assert "managed files: 1 file" in add_result.output
        assert "Alpha Pack" in add_result.output
        assert "Install notes." in add_result.output
        assert (env_root / "agent-cards" / "alpha.md").exists()

        list_result = runner.invoke(cards_command.app, ["list"])
        assert list_result.exit_code == 0, list_result.output
        assert "alpha" in list_result.output

        readme_result = runner.invoke(cards_command.app, ["readme", "alpha"])
        assert readme_result.exit_code == 0, readme_result.output
        assert "Alpha Pack" in readme_result.output
        assert "Install notes." in readme_result.output

        remove_result = runner.invoke(cards_command.app, ["remove", "alpha"])
        assert remove_result.exit_code == 0, remove_result.output
        assert "Card Pack Removed" in remove_result.output
        assert "name: alpha" in remove_result.output
        assert "removed files: 1 file" in remove_result.output
        assert not (env_root / "agent-cards" / "alpha.md").exists()
    finally:
        update_global_settings(old_settings)


def test_cards_help_has_registry_option_no_registry_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(cards_command.app, ["--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "--registry" in output
    assert "│ registry" not in output


def test_cards_readme_without_readme_reports_notice(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
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
    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n",
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

        readme_result = runner.invoke(cards_command.app, ["readme", "alpha"])
        assert readme_result.exit_code == 0, readme_result.output
        assert "does not include a README.md" in readme_result.output
    finally:
        update_global_settings(old_settings)


def test_cards_readme_without_selector_uses_only_installed_pack(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "README.md").write_text(
        "# Alpha Pack\n\nOpened without selector.\n",
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
    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n",
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

        readme_result = runner.invoke(cards_command.app, ["readme"])
        assert readme_result.exit_code == 0, readme_result.output
        assert "Alpha Pack" in readme_result.output
        assert "Opened without selector." in readme_result.output
    finally:
        update_global_settings(old_settings)


def test_top_level_env_flag_routes_to_cards_subcommand(tmp_path: Path) -> None:
    env_root = tmp_path / "custom-env"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fast_agent.cli",
            "--env",
            str(env_root),
            "cards",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "cards [OPTIONS] COMMAND" in result.stdout


def test_cards_add_uses_configured_marketplace_urls_by_default(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "README.md").write_text(
        "# Alpha Pack\n\nConfigured via default marketplace.\n",
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
    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n"
        "cards:\n"
        "  marketplace_urls:\n"
        f"    - '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()
        add_result = runner.invoke(cards_command.app, ["add", "alpha"])

        assert add_result.exit_code == 0, add_result.output
        assert "Card Pack Installed" in add_result.output
        assert "name: alpha" in add_result.output
        assert "Alpha Pack" in add_result.output
        assert "Configured via default marketplace." in add_result.output
        assert (env_root / "agent-cards" / "alpha.md").exists()
    finally:
        update_global_settings(old_settings)


def test_cards_publish_no_push_commits_locally(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
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
    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n",
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

        installed_card = env_root / "agent-cards" / "alpha.md"
        installed_card.write_text(
            installed_card.read_text(encoding="utf-8") + "\nlocal publish edit\n",
            encoding="utf-8",
        )

        publish_result = runner.invoke(
            cards_command.app,
            ["publish", "alpha", "--no-push", "--message", "publish alpha"],
        )
        assert publish_result.exit_code == 0, publish_result.output
        assert "Status: committed locally" in publish_result.output
        assert "local publish edit" in (repo / "packs" / "alpha" / "agent-cards" / "alpha.md").read_text(
            encoding="utf-8"
        )
    finally:
        update_global_settings(old_settings)


def test_cards_publish_help_lists_temp_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cards_command.app, ["publish", "--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "--temp-dir" in output
    assert "--keep-temp" in output


def test_cards_update_reports_invalid_settings_yaml_without_traceback(tmp_path: Path) -> None:
    env_root = tmp_path / ".fast-agent"
    env_root.mkdir(parents=True)
    config_path = env_root / "fastagent.config.yaml"
    config_path.write_text(
        "mcp:\n"
        "  targets:\n"
        "    - name: openai\n"
        "        target: https://developers.openai.com/mcp\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    old_cwd = Path.cwd()
    old_env_dir = os.environ.get("ENVIRONMENT_DIR")
    old_runtime_env = os.environ.get(FAST_AGENT_RUNTIME_ENVIRONMENT)
    old_fast_agent_home = os.environ.get("FAST_AGENT_HOME")
    try:
        os.chdir(tmp_path)
        os.environ["ENVIRONMENT_DIR"] = env_root.as_posix()
        os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = env_root.as_posix()
        os.environ.pop("FAST_AGENT_HOME", None)
        config_module._settings = None

        runner = CliRunner()
        result = runner.invoke(cards_command.app, ["update"])
        output = strip_ansi(result.output)

        assert result.exit_code == 1, output
        assert "Error loading fast-agent settings:" in output
        assert f"Failed to parse YAML file: {config_path}" in output
        assert "mapping values are not allowed here" in output
        assert "Traceback" not in output
    finally:
        os.chdir(old_cwd)
        if old_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = old_env_dir
        if old_runtime_env is None:
            os.environ.pop(FAST_AGENT_RUNTIME_ENVIRONMENT, None)
        else:
            os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = old_runtime_env
        if old_fast_agent_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = old_fast_agent_home
        update_global_settings(old_settings)
