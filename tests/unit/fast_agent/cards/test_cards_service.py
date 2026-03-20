from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.cards import manager, service
from fast_agent.paths import resolve_environment_paths

if TYPE_CHECKING:
    from pathlib import Path


def test_apply_update_plan_collects_readmes_for_updated_packs(tmp_path: Path, monkeypatch) -> None:
    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
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
        lambda selected, *, environment_paths, force: [update],
    )
    monkeypatch.setattr(
        service.manager,
        "load_card_pack_readme",
        lambda pack_dir: "# Alpha Pack" if pack_dir == updated_pack_dir else None,
    )

    result = service.apply_update_plan(
        [update],
        environment_paths=env_paths,
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


def test_select_installed_pack_raises_lookup_error_for_missing_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)

    monkeypatch.setattr(service.manager, "list_local_card_packs", lambda *, environment_paths: [])

    try:
        service.select_installed_pack(environment_paths=env_paths, selector="missing")
    except service.CardPackLookupError as exc:
        assert str(exc) == "Card pack not found: missing"
    else:
        raise AssertionError("Expected CardPackLookupError")


def test_ensure_pack_available_reuses_installed_pack(tmp_path: Path, monkeypatch) -> None:
    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    pack_dir = env_paths.card_packs / "alpha"
    local_pack = manager.LocalCardPack(
        index=1,
        name="alpha",
        pack_dir=pack_dir,
        source=None,
    )

    monkeypatch.setattr(
        service,
        "select_installed_pack",
        lambda *, environment_paths, selector: local_pack,
    )

    async def _fail_install(*_args, **_kwargs):
        raise AssertionError("install_pack should not run for installed packs")

    monkeypatch.setattr(service, "install_pack", _fail_install)

    result = service.ensure_pack_available_sync(
        selector="alpha",
        environment_paths=env_paths,
    )

    assert result == service.EnsuredCardPack(
        name="alpha",
        pack_dir=pack_dir,
        installed=False,
        install_record=None,
    )


def test_ensure_pack_available_installs_missing_pack(tmp_path: Path, monkeypatch) -> None:
    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    pack_dir = env_paths.card_packs / "alpha"
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

    def _missing_pack(*, environment_paths, selector):
        raise service.CardPackLookupError(f"Card pack not found: {selector}")

    async def _install_pack(source, selector, *, environment_paths, force):
        assert source == "marketplace.json"
        assert selector == "alpha"
        assert force is False
        return install_record

    monkeypatch.setattr(service, "select_installed_pack", _missing_pack)
    monkeypatch.setattr(service, "install_pack", _install_pack)

    result = service.ensure_pack_available_sync(
        selector="alpha",
        environment_paths=env_paths,
        registry="marketplace.json",
    )

    assert result == service.EnsuredCardPack(
        name="alpha",
        pack_dir=pack_dir,
        installed=True,
        install_record=install_record,
    )
