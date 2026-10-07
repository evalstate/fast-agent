from __future__ import annotations

import json
import subprocess
from functools import partial
from typing import TYPE_CHECKING

import pytest
import yaml
from typer.testing import CliRunner

from fast_agent.cli.commands import plugins as plugins_command
from fast_agent.cli.plugin_offer import offer_recommended_plugins
from fast_agent.cli.update_check import check_for_update_notice, fetch_marketplace_payload
from fast_agent.config import get_settings, update_global_settings
from fast_agent.marketplace.update_status import is_update_applicable
from fast_agent.plugins.marketplace import parse_marketplace_plugins
from fast_agent.plugins.operations import (
    check_plugin_updates_in_roots,
    install_marketplace_plugin_sync,
)
from fast_agent.utils.versions import requirement_satisfied

if TYPE_CHECKING:
    from pathlib import Path

DAY = 24 * 3600


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_plugin(repo: Path, name: str, *, version: str = "0.1.0", reply: str = "ok") -> None:
    root = repo / "plugins" / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "plugin.yaml").write_text(
        f"schema_version: 1\nname: {name}\nversion: {version}\ndescription: {name} plugin\n"
        f"commands:\n  {name}:\n    description: Run\n    handler: ./commands.py:run\n",
        encoding="utf-8",
    )
    (root / "commands.py").write_text(f"async def run(ctx):\n    return {reply!r}\n", "utf-8")


def _commit(repo: Path) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "update")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Test User")
    for name in ("alpha", "beta", "gamma"):
        _write_plugin(repo, name)
    _commit(repo)
    return repo


def _publish(
    repo: Path,
    marketplace: Path,
    *,
    requires: str | None = None,
    bundle: tuple[str, ...] = ("alpha", "beta"),
) -> Path:
    """Write marketplace.json the way card-packs' sync script does."""
    entries = []
    for name in ("alpha", "beta", "gamma"):
        manifest = yaml.safe_load((repo / "plugins" / name / "plugin.yaml").read_text())
        entry = {
            "name": name,
            "description": f"{name} plugin",
            "repo_url": repo.as_posix(),
            "repo_path": f"plugins/{name}",
            "version": manifest["version"],
            "path_oid": _git(repo, "rev-parse", f"HEAD:plugins/{name}"),
        }
        if requires is not None:
            entry["requires_fast_agent"] = requires
        entries.append(entry)
    payload = {
        "command_plugins": entries,
        "plugin_bundles": [{"name": "recommended", "plugins": list(bundle)}],
    }
    marketplace.write_text(json.dumps(payload), encoding="utf-8")
    return marketplace


def test_marketplace_metadata_and_bundles_are_parsed(repo: Path, tmp_path: Path) -> None:
    marketplace = _publish(repo, tmp_path / "marketplace.json", requires=">=0.10.2")

    plugins = {p.name: p for p in parse_marketplace_plugins(json.loads(marketplace.read_text()))}

    assert plugins["alpha"].bundles == ("recommended",)
    assert plugins["gamma"].bundles == ()
    assert plugins["alpha"].version == "0.1.0"
    assert plugins["alpha"].path_oid == _git(repo, "rev-parse", "HEAD:plugins/alpha")
    assert plugins["alpha"].requires_fast_agent == ">=0.10.2"


def test_plugins_add_bundle_installs_missing_members_and_enables_all(
    repo: Path, tmp_path: Path
) -> None:
    marketplace = _publish(repo, tmp_path / "marketplace.json")
    home = tmp_path / ".fast-agent"
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(f"home: '{home.as_posix()}'\n", encoding="utf-8")
    (beta,) = [
        p
        for p in parse_marketplace_plugins(json.loads(marketplace.read_text()))
        if p.name == "beta"
    ]
    install_marketplace_plugin_sync(beta, destination_root=home / "plugins")

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        result = CliRunner().invoke(
            plugins_command.app,
            ["--registry", marketplace.as_posix(), "add", "--bundle", "recommended", "--project"],
            env={"COLUMNS": "200"},
        )
    finally:
        update_global_settings(old_settings)

    assert result.exit_code == 0, result.output
    assert "already installed" in result.output
    assert sorted(p.name for p in (home / "plugins").iterdir()) == ["alpha", "beta"]
    enabled = config_path.read_text(encoding="utf-8")
    assert "alpha" in enabled and "beta" in enabled and "gamma" not in enabled


def test_update_notice_agrees_with_plugins_update_and_respects_requirements(
    repo: Path, tmp_path: Path
) -> None:
    marketplace = _publish(repo, tmp_path / "marketplace.json")
    home = tmp_path / "home"
    root = home / "plugins"
    for plugin in parse_marketplace_plugins(
        json.loads(marketplace.read_text()), source_url=marketplace.as_posix()
    ):
        if plugin.name in ("alpha", "beta"):
            install_marketplace_plugin_sync(plugin, destination_root=root)
    check = partial(
        check_for_update_notice,
        home=home,
        current_version="0.10.5",
        fetch_latest_version=lambda: "0.10.5",
        interval_seconds=0,
        plugin_roots=[root],
    )
    assert check() is None

    # A change elsewhere in the repo does not count; a change to alpha does.
    (repo / "README.md").write_text("unrelated\n", encoding="utf-8")
    _write_plugin(repo, "alpha", version="0.2.0", reply="new")
    _commit(repo)
    _publish(repo, marketplace, requires=">=0.10.2")

    notice = check()
    assert notice is not None
    assert "alpha[/cyan] 0.1.0 → 0.2.0" in notice and "beta" not in notice
    assert "plugins update all" in notice
    authoritative = check_plugin_updates_in_roots(destination_roots=[root])
    assert [u.name for u in authoritative if is_update_applicable(u.status)] == ["alpha"]

    _publish(repo, marketplace, requires=">=0.11.0")
    notice = check()
    assert notice is not None
    assert "needs fast-agent >=0.11.0" in notice
    assert "plugins update all" not in notice


def test_recommended_offer_installs_once_and_retries_failed_fetch(
    repo: Path, tmp_path: Path
) -> None:
    marketplace = _publish(repo, tmp_path / "marketplace.json")
    global_home = tmp_path / "global"
    prompts: list[str] = []
    messages: list[str] = []

    def offer(*, url: str = marketplace.as_posix(), answer: bool = True, now: float = 0.0):
        offer_recommended_plugins(
            global_home=global_home,
            plugin_roots=[global_home / "plugins", tmp_path / "project" / "plugins"],
            marketplace_url=url,
            fetch_payload=partial(fetch_marketplace_payload, timeout_seconds=1),
            confirm=lambda prompt: prompts.append(prompt) or answer,
            echo=messages.append,
            now=now,
        )

    offer(url="http://127.0.0.1:9/marketplace.json")  # nothing listening
    offer(now=DAY / 2)  # failed fetches retry at most daily
    assert prompts == []

    offer(now=float("inf"))
    assert len(prompts) == 1 and "alpha — alpha plugin" in prompts[0]
    assert sorted(p.name for p in (global_home / "plugins").iterdir()) == ["alpha", "beta"]
    assert "beta" in (global_home / "fast-agent.yaml").read_text(encoding="utf-8")

    offer(now=float("inf"))
    assert len(prompts) == 1


def test_declined_offer_is_not_repeated(repo: Path, tmp_path: Path) -> None:
    marketplace = _publish(repo, tmp_path / "marketplace.json")
    global_home = tmp_path / "global"
    prompts: list[str] = []
    messages: list[str] = []
    for _ in range(2):
        offer_recommended_plugins(
            global_home=global_home,
            plugin_roots=[global_home / "plugins"],
            marketplace_url=marketplace.as_posix(),
            fetch_payload=partial(fetch_marketplace_payload, timeout_seconds=1),
            confirm=lambda prompt: prompts.append(prompt) or False,
            echo=messages.append,
        )

    assert len(prompts) == 1
    assert not (global_home / "plugins").exists()
    assert "plugins add --bundle recommended --global" in messages[0]


@pytest.mark.parametrize(
    ("requirement", "version", "expected"),
    [
        (">=0.10.2", "0.10.2", True),
        (">=0.10.2", "0.10.1", False),
        (">=0.10.2", "0.10.43.dev0", True),
        (">=0.10,<0.11", "0.10.9", True),
        (">=0.10,<0.11", "0.11.0", False),
        ("~=0.10", "0.10.5", False),  # unsupported operator: never guess
    ],
)
def test_requirement_satisfied(requirement: str, version: str, expected: bool) -> None:
    assert requirement_satisfied(requirement, version) is expected
