from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.cards import manager, service
from fast_agent.config import get_settings, update_global_settings
from fast_agent.paths import resolve_home_paths

if TYPE_CHECKING:
    from pathlib import Path


def test_apply_update_plan_collects_readmes_for_updated_packs(tmp_path: Path, monkeypatch) -> None:
    home_paths = resolve_home_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    updated_pack_dir = tmp_path / "card-packs" / "alpha"
    update = manager.CardPackUpdateInfo(
        index=1,
        name="alpha",
        pack_dir=updated_pack_dir,
        status="updated",
    )

    monkeypatch.setattr(
        service.manager,
        "apply_card_pack_updates",
        lambda selected, *, home_paths, force: [update],
    )
    monkeypatch.setattr(
        service.manager,
        "load_card_pack_readme",
        lambda pack_dir: "# Alpha Pack" if pack_dir == updated_pack_dir else None,
    )
    monkeypatch.setattr(
        service,
        "_ensure_required_pack_plugins_sync",
        lambda pack_dir, *, home_paths, plugin_registry=None: None,
    )

    result = service.apply_update_plan(
        [update],
        home_paths=home_paths,
        force=False,
    )

    assert result.applied == [update]
    assert result.readmes == [
        service.CardPackReadmeRecord(
            pack_name="alpha",
            pack_dir=updated_pack_dir,
            readme="# Alpha Pack",
        )
    ]


def test_apply_update_plan_uses_updated_pack_source_for_required_plugins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home_paths = resolve_home_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    updated_pack_dir = tmp_path / "card-packs" / "alpha"
    source = manager.InstalledCardPackSource(
        schema_version=1,
        installed_via="marketplace",
        source_origin="local",
        name="alpha",
        kind="card",
        repo_url="/tmp/repo",
        repo_ref=None,
        repo_path="packs/alpha",
        source_url="/tmp/repo/marketplace.json",
        installed_commit="abc123",
        installed_path_oid="def456",
        installed_revision="abc123",
        installed_at="2026-05-17T00:00:00Z",
        content_fingerprint="sha256:test",
        installed_files=(),
    )
    update = manager.CardPackUpdateInfo(
        index=1,
        name="alpha",
        pack_dir=updated_pack_dir,
        status="updated",
        managed_source=source,
    )
    seen: dict[str, str | None] = {}

    monkeypatch.setattr(
        service.manager,
        "apply_card_pack_updates",
        lambda selected, *, home_paths, force: [update],
    )
    monkeypatch.setattr(service.manager, "load_card_pack_readme", lambda pack_dir: None)

    def _capture_plugin_registry(
        pack_dir: Path,
        *,
        home_paths,
        plugin_registry: str | None = None,
    ) -> None:
        seen["pack_dir"] = pack_dir.as_posix()
        seen["plugin_registry"] = plugin_registry

    monkeypatch.setattr(service, "_ensure_required_pack_plugins_sync", _capture_plugin_registry)

    service.apply_update_plan(
        [update],
        home_paths=home_paths,
        force=False,
    )

    assert seen == {
        "pack_dir": updated_pack_dir.as_posix(),
        "plugin_registry": "/tmp/repo/marketplace.json",
    }


def _write_pack_requiring_plugin(pack_dir: Path, plugin_name: str) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "card-pack.yaml").write_text(
        "schema_version: 2\n"
        "name: alpha\n"
        "kind: card\n"
        "install:\n"
        "  agent_cards: []\n"
        "  files: []\n"
        "  tool_cards: []\n"
        "plugins:\n"
        f"  required: ['{plugin_name}']\n",
        encoding="utf-8",
    )


def _write_local_plugin(plugin_dir: Path, plugin_name: str) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"schema_version: 1\nname: {plugin_name}\ndescription: Local test plugin\ncommands: {{}}\n",
        encoding="utf-8",
    )


def test_required_plugin_sync_skips_registry_fetch_when_already_installed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home_paths = resolve_home_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nhome: '{home_paths.root.as_posix()}'\n",
        encoding="utf-8",
    )
    pack_dir = home_paths.card_packs / "alpha"
    _write_pack_requiring_plugin(pack_dir, "agent-finder")
    _write_local_plugin(home_paths.plugins / "agent-finder", "agent-finder")

    def _fail_fetch(_url: str):
        raise AssertionError("registry fetch should not run for installed required plugins")

    monkeypatch.setattr(
        service.plugin_ops, "fetch_marketplace_plugins_with_source_sync", _fail_fetch
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        service._ensure_required_pack_plugins_sync(
            pack_dir,
            home_paths=home_paths,
            plugin_registry="/tmp/unavailable-marketplace.json",
        )
    finally:
        update_global_settings(old_settings)

    assert "agent-finder" in config_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_required_plugin_async_skips_registry_fetch_when_already_installed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home_paths = resolve_home_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nhome: '{home_paths.root.as_posix()}'\n",
        encoding="utf-8",
    )
    pack_dir = home_paths.card_packs / "alpha"
    _write_pack_requiring_plugin(pack_dir, "agent-finder")
    _write_local_plugin(home_paths.plugins / "agent-finder", "agent-finder")

    async def _fail_fetch(_url: str):
        raise AssertionError("registry fetch should not run for installed required plugins")

    monkeypatch.setattr(service.plugin_ops, "fetch_marketplace_plugins_with_source", _fail_fetch)

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        await service._ensure_required_pack_plugins(
            pack_dir,
            home_paths=home_paths,
            plugin_registry="/tmp/unavailable-marketplace.json",
        )
    finally:
        update_global_settings(old_settings)

    assert "agent-finder" in config_path.read_text(encoding="utf-8")


def test_select_installed_pack_raises_lookup_error_for_missing_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home_paths = resolve_home_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)

    monkeypatch.setattr(service.manager, "list_local_card_packs", lambda *, home_paths: [])

    try:
        service.select_installed_pack(home_paths=home_paths, selector="missing")
    except service.CardPackLookupError as exc:
        assert str(exc) == "Card pack not found: missing"
    else:
        raise AssertionError("Expected CardPackLookupError")


def test_ensure_pack_available_reuses_installed_pack(tmp_path: Path, monkeypatch) -> None:
    home_paths = resolve_home_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    pack_dir = home_paths.card_packs / "alpha"
    local_pack = manager.LocalCardPack(
        index=1,
        name="alpha",
        pack_dir=pack_dir,
        source=None,
    )

    monkeypatch.setattr(
        service,
        "select_installed_pack",
        lambda *, home_paths, selector: local_pack,
    )

    async def _fail_install(*_args, **_kwargs):
        raise AssertionError("install_pack should not run for installed packs")

    monkeypatch.setattr(service, "install_pack", _fail_install)

    result = service.ensure_pack_available_sync(
        selector="alpha",
        home_paths=home_paths,
    )

    assert result == service.EnsuredCardPack(
        name="alpha",
        pack_dir=pack_dir,
        installed=False,
        install_record=None,
    )


def test_ensure_pack_available_installs_missing_pack(tmp_path: Path, monkeypatch) -> None:
    home_paths = resolve_home_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    pack_dir = home_paths.card_packs / "alpha"
    install_record = service.CardPackInstallRecord(
        pack=manager.MarketplaceCardPack(
            name="alpha",
            description=None,
            kind="card",
            repo_url="repo",
            repo_ref=None,
            repo_path="packs/alpha",
        ),
        install_result=manager.CardPackInstallResult(
            pack_dir=pack_dir,
            installed_files=("agent-cards/alpha.md",),
            source=manager.InstalledCardPackSource(
                schema_version=1,
                installed_via="test",
                source_origin="remote",
                name="alpha",
                kind="card",
                repo_url="repo",
                repo_ref=None,
                repo_path="packs/alpha",
                source_url=None,
                installed_commit="abc123",
                installed_path_oid=None,
                installed_revision="abc123",
                installed_at="2026-03-20T00:00:00Z",
                content_fingerprint="fingerprint",
                installed_files=("agent-cards/alpha.md",),
            ),
        ),
        readme=None,
    )

    def _missing_pack(*, home_paths, selector):
        raise service.CardPackLookupError(f"Card pack not found: {selector}")

    async def _install_pack(source, selector, *, home_paths, force):
        assert source == "marketplace.json"
        assert selector == "alpha"
        assert force is False
        return install_record

    monkeypatch.setattr(service, "select_installed_pack", _missing_pack)
    monkeypatch.setattr(service, "install_pack", _install_pack)

    result = service.ensure_pack_available_sync(
        selector="alpha",
        home_paths=home_paths,
        registry="marketplace.json",
    )

    assert result == service.EnsuredCardPack(
        name="alpha",
        pack_dir=pack_dir,
        installed=True,
        install_record=install_record,
    )
