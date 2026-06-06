from __future__ import annotations

import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

from fast_agent.config import SkillsSettings, get_settings
from fast_agent.skills.models import (
    SKILL_SOURCE_FILENAME,
    SKILL_SOURCE_SCHEMA_VERSION,
    InstalledSkillSource,
    MarketplaceSkill,
)
from fast_agent.skills.operations import (
    apply_skill_updates,
    check_skill_updates,
    install_marketplace_skill_sync,
    parse_ls_remote_commit,
    resolve_source_revision,
)
from fast_agent.skills.provenance import (
    format_skill_provenance,
    format_skill_provenance_details,
    read_installed_skill_source,
    write_installed_skill_source,
)
from fast_agent.skills.scope import (
    get_manager_directory,
    order_skill_directories_for_display,
    resolve_skills_management_scope,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def _write_skill(repo: Path, *, name: str, body: str) -> None:
    skill_dir = repo / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _write_invalid_skill(repo: Path, *, name: str, body: str) -> None:
    skill_dir = repo / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _write_bundle_skill(repo: Path, *, bundle: str, name: str, body: str) -> None:
    skill_dir = repo / bundle / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_marketplace_skill(repo: Path, *, name: str) -> MarketplaceSkill:
    return MarketplaceSkill(
        name=name,
        description="test",
        repo_url=str(repo),
        repo_ref=None,
        repo_path=f"skills/{name}",
    )


def _make_manifest_marketplace_skill(repo: Path, *, name: str) -> MarketplaceSkill:
    return MarketplaceSkill(
        name=name,
        description="test",
        repo_url=str(repo),
        repo_ref=None,
        repo_path=f"skills/{name}/SKILL.md",
    )


def _make_ref_marketplace_skill(
    repo: Path,
    *,
    name: str,
    repo_ref: str,
) -> MarketplaceSkill:
    return MarketplaceSkill(
        name=name,
        description="test",
        repo_url=str(repo),
        repo_ref=repo_ref,
        repo_path=f"skills/{name}",
    )


def _make_bundle_marketplace_skill(
    repo: Path,
    *,
    name: str,
    repo_ref: str | None,
) -> MarketplaceSkill:
    return MarketplaceSkill(
        name=name,
        description="test",
        repo_url=str(repo),
        repo_ref=repo_ref,
        repo_path="bundle",
    )


def _make_remote_source(repo_ref: str | None = "v1.0.0") -> InstalledSkillSource:
    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin="remote",
        repo_url="https://example.com/skills.git",
        repo_ref=repo_ref,
        repo_path="skills/alpha",
        source_url=None,
        installed_commit="deadbeef",
        installed_path_oid="cafebabe",
        installed_revision="deadbeef",
        installed_at="2026-01-01T00:00:00Z",
        content_fingerprint="sha256:test",
    )


def test_install_writes_sidecar_with_provenance(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    first_commit = _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    installed_dir = install_marketplace_skill_sync(
        _make_marketplace_skill(repo, name="alpha"),
        managed_dir,
    )

    read_result = read_installed_skill_source(installed_dir)
    assert read_result.error is None
    assert read_result.source is not None
    assert read_result.source.source_origin == "local"
    assert read_result.source.installed_commit == first_commit
    assert read_result.source.installed_path_oid is not None
    assert read_result.source.installed_revision == first_commit
    assert read_result.source.content_fingerprint.startswith("sha256:")


def test_install_from_manifest_path_tracks_whole_skill_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    helper_file = repo / "skills" / "alpha" / "helper.txt"
    helper_file.write_text("helper v1\n", encoding="utf-8")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    installed_dir = install_marketplace_skill_sync(
        _make_manifest_marketplace_skill(repo, name="alpha"),
        managed_dir,
    )

    read_result = read_installed_skill_source(installed_dir)
    assert read_result.error is None
    assert read_result.source is not None
    assert read_result.source.repo_path == "skills/alpha"
    assert read_result.source.installed_path_oid is not None

    helper_file.write_text("helper v2\n", encoding="utf-8")
    _commit_all(repo, "update helper")

    updates = check_skill_updates(destination_root=managed_dir)

    assert len(updates) == 1
    assert updates[0].status == "update_available"


def test_install_from_bundle_root_tracks_selected_nested_skill_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_bundle_skill(repo, bundle="bundle", name="alpha", body="alpha v1")
    _write_bundle_skill(repo, bundle="bundle", name="beta", body="beta v1")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    installed_dir = install_marketplace_skill_sync(
        _make_bundle_marketplace_skill(repo, name="alpha", repo_ref=None),
        managed_dir,
    )

    read_result = read_installed_skill_source(installed_dir)
    assert read_result.error is None
    assert read_result.source is not None
    assert read_result.source.repo_path == "bundle/skills/alpha"
    assert read_result.source.installed_path_oid is not None


def test_install_from_dirty_local_git_repo_records_unknown_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    _commit_all(repo, "initial")
    _write_skill(repo, name="alpha", body="dirty working tree")

    managed_dir = tmp_path / "managed"
    installed_dir = install_marketplace_skill_sync(
        _make_marketplace_skill(repo, name="alpha"),
        managed_dir,
    )

    read_result = read_installed_skill_source(installed_dir)
    assert read_result.error is None
    assert read_result.source is not None
    assert read_result.source.source_origin == "local"
    assert read_result.source.installed_commit is None
    assert read_result.source.installed_path_oid is None
    assert read_result.source.installed_revision == "local"

    updates = check_skill_updates(destination_root=managed_dir)
    assert len(updates) == 1
    assert updates[0].status == "unknown_revision"


def test_apply_unknown_revision_skill_update_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    _commit_all(repo, "initial")
    _write_skill(repo, name="alpha", body="dirty working tree")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(
        _make_marketplace_skill(repo, name="alpha"),
        managed_dir,
    )

    updates = check_skill_updates(destination_root=managed_dir)
    applied = apply_skill_updates(updates, force=True)

    assert len(applied) == 1
    assert applied[0].status == "unknown_revision"
    assert "dirty working tree" in (managed_dir / "alpha" / "SKILL.md").read_text(encoding="utf-8")


def test_dirty_local_bundle_update_selects_requested_skill(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_bundle_skill(repo, bundle="bundle", name="alpha", body="alpha v1")
    _write_bundle_skill(repo, bundle="bundle", name="beta", body="beta v1")
    _commit_all(repo, "initial")
    _write_bundle_skill(repo, bundle="bundle", name="alpha", body="alpha dirty")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(
        _make_bundle_marketplace_skill(repo, name="alpha", repo_ref=None),
        managed_dir,
    )

    updates = check_skill_updates(destination_root=managed_dir)

    assert len(updates) == 1
    assert updates[0].status == "unknown_revision"


def test_install_pinned_local_bundle_selects_requested_skill(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_bundle_skill(repo, bundle="bundle", name="alpha", body="alpha v1")
    _write_bundle_skill(repo, bundle="bundle", name="beta", body="beta v1")
    first_commit = _commit_all(repo, "initial")

    _write_bundle_skill(repo, bundle="bundle", name="alpha", body="alpha dirty worktree")

    managed_dir = tmp_path / "managed"
    installed_dir = install_marketplace_skill_sync(
        _make_bundle_marketplace_skill(repo, name="alpha", repo_ref=first_commit),
        managed_dir,
    )

    installed_text = (installed_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "alpha v1" in installed_text
    assert "beta v1" not in installed_text
    assert "alpha dirty worktree" not in installed_text

    read_result = read_installed_skill_source(installed_dir)
    assert read_result.error is None
    assert read_result.source is not None
    assert read_result.source.installed_commit == first_commit


def test_check_pinned_local_ref_uses_ref_path_not_current_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    first_commit = _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(
        _make_ref_marketplace_skill(repo, name="alpha", repo_ref=first_commit),
        managed_dir,
    )

    _git(repo, "rm", "skills/alpha/SKILL.md")
    _commit_all(repo, "remove alpha from head")

    updates = check_skill_updates(destination_root=managed_dir)

    assert len(updates) == 1
    assert updates[0].status == "up_to_date"
    assert updates[0].current_revision == first_commit
    assert updates[0].available_revision == first_commit


def test_check_and_apply_update_from_local_git_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(_make_marketplace_skill(repo, name="alpha"), managed_dir)

    _write_skill(repo, name="alpha", body="v2")
    second_commit = _commit_all(repo, "update")

    updates = check_skill_updates(destination_root=managed_dir)
    assert len(updates) == 1
    assert updates[0].status == "update_available"

    applied = apply_skill_updates([updates[0]], force=False)
    assert len(applied) == 1
    assert applied[0].status == "updated"

    installed_skill = managed_dir / "alpha" / "SKILL.md"
    assert "v2" in installed_skill.read_text(encoding="utf-8")

    read_result = read_installed_skill_source(managed_dir / "alpha")
    assert read_result.error is None
    assert read_result.source is not None
    assert read_result.source.installed_commit == second_commit

    recheck = check_skill_updates(destination_root=managed_dir)
    assert recheck[0].status == "up_to_date"


def test_apply_update_from_local_git_repo_uses_resolved_commit_not_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(_make_marketplace_skill(repo, name="alpha"), managed_dir)

    _write_skill(repo, name="alpha", body="v2")
    second_commit = _commit_all(repo, "update")
    updates = check_skill_updates(destination_root=managed_dir)
    assert updates[0].available_revision == second_commit

    _write_skill(repo, name="alpha", body="dirty worktree")

    applied = apply_skill_updates([updates[0]], force=False)

    assert applied[0].status == "updated"
    installed_text = (applied[0].skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "v2" in installed_text
    assert "dirty worktree" not in installed_text

    read_result = read_installed_skill_source(managed_dir / "alpha")
    assert read_result.error is None
    assert read_result.source is not None
    assert read_result.source.installed_commit == second_commit


def test_apply_update_rejects_invalid_staged_skill_without_replacing_install(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(_make_marketplace_skill(repo, name="alpha"), managed_dir)

    _write_invalid_skill(repo, name="alpha", body="missing description")
    _commit_all(repo, "invalid update")

    updates = check_skill_updates(destination_root=managed_dir)
    assert updates[0].status == "update_available"

    applied = apply_skill_updates([updates[0]], force=False)

    assert applied[0].status == "invalid_local_skill"
    assert "description" in (applied[0].detail or "")
    installed_text = (managed_dir / "alpha" / "SKILL.md").read_text(encoding="utf-8")
    assert "description: Test skill" in installed_text
    assert "v1" in installed_text
    assert "missing description" not in installed_text


def test_apply_update_records_actual_copy_origin(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(_make_marketplace_skill(repo, name="alpha"), managed_dir)

    source_result = read_installed_skill_source(managed_dir / "alpha")
    assert source_result.source is not None
    write_installed_skill_source(
        managed_dir / "alpha",
        replace(source_result.source, source_origin="remote"),
    )

    _write_skill(repo, name="alpha", body="v2")
    _commit_all(repo, "update")

    updates = check_skill_updates(destination_root=managed_dir)
    applied = apply_skill_updates([updates[0]], force=False)

    assert applied[0].status == "updated"
    assert applied[0].managed_source is not None
    assert applied[0].managed_source.source_origin == "local"


def test_apply_update_from_local_bundle_selects_requested_skill(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_bundle_skill(repo, bundle="bundle", name="alpha", body="alpha v1")
    _write_bundle_skill(repo, bundle="bundle", name="beta", body="beta v1")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(
        _make_bundle_marketplace_skill(repo, name="alpha", repo_ref=None),
        managed_dir,
    )

    _write_bundle_skill(repo, bundle="bundle", name="alpha", body="alpha v2")
    _write_bundle_skill(repo, bundle="bundle", name="beta", body="beta v2")
    second_commit = _commit_all(repo, "update")
    updates = check_skill_updates(destination_root=managed_dir)
    assert updates[0].available_revision == second_commit

    _write_bundle_skill(repo, bundle="bundle", name="alpha", body="alpha dirty worktree")

    applied = apply_skill_updates([updates[0]], force=False)

    assert applied[0].status == "updated"
    installed_text = (applied[0].skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "alpha v2" in installed_text
    assert "beta v2" not in installed_text
    assert "alpha dirty worktree" not in installed_text


def test_apply_update_skips_dirty_without_force_and_overwrites_with_force(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(_make_marketplace_skill(repo, name="alpha"), managed_dir)

    _write_skill(repo, name="alpha", body="v2")
    _commit_all(repo, "update")

    installed_skill = managed_dir / "alpha" / "SKILL.md"
    installed_skill.write_text(
        installed_skill.read_text(encoding="utf-8") + "\nlocal edit\n",
        encoding="utf-8",
    )

    updates = check_skill_updates(destination_root=managed_dir)
    assert updates[0].status == "update_available"

    skipped = apply_skill_updates([updates[0]], force=False)
    assert skipped[0].status == "skipped_dirty"
    assert "local edit" in installed_skill.read_text(encoding="utf-8")

    forced = apply_skill_updates([updates[0]], force=True)
    assert forced[0].status == "updated"
    installed_text = installed_skill.read_text(encoding="utf-8")
    assert "v2" in installed_text
    assert "local edit" not in installed_text


def test_unrelated_repo_commit_does_not_trigger_skill_update(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="v1")
    _commit_all(repo, "initial")

    managed_dir = tmp_path / "managed"
    install_marketplace_skill_sync(_make_marketplace_skill(repo, name="alpha"), managed_dir)

    (repo / "README.md").write_text("unrelated\n", encoding="utf-8")
    _commit_all(repo, "unrelated change")

    updates = check_skill_updates(destination_root=managed_dir)
    assert len(updates) == 1
    assert updates[0].status == "up_to_date"


def test_format_skill_provenance_states(tmp_path: Path) -> None:
    unmanaged_dir = tmp_path / "unmanaged"
    unmanaged_dir.mkdir(parents=True)
    (unmanaged_dir / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n", encoding="utf-8")
    assert format_skill_provenance(unmanaged_dir) == "unmanaged (no sidecar)"

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n", encoding="utf-8")
    (invalid_dir / SKILL_SOURCE_FILENAME).write_text(
        '{"schema_version": 1, "installed_via": "marketplace", "source_origin": "remote", "repo_url": "https://example.com/repo", "repo_path": "../escape"}',
        encoding="utf-8",
    )
    assert "invalid metadata" in format_skill_provenance(invalid_dir)


def test_format_skill_provenance_details(tmp_path: Path) -> None:
    unmanaged_dir = tmp_path / "unmanaged"
    unmanaged_dir.mkdir(parents=True)
    (unmanaged_dir / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n", encoding="utf-8")

    provenance_text, installed_text = format_skill_provenance_details(unmanaged_dir)
    assert provenance_text == "unmanaged."
    assert installed_text is None

    managed_dir = tmp_path / "managed"
    managed_dir.mkdir(parents=True)
    (managed_dir / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n", encoding="utf-8")
    write_installed_skill_source(
        managed_dir,
        InstalledSkillSource(
            schema_version=1,
            installed_via="marketplace",
            source_origin="remote",
            repo_url="https://github.com/example/skills",
            repo_ref="main",
            repo_path="skills/x",
            source_url=None,
            installed_commit="abcdef1234567890",
            installed_path_oid="beadfeed",
            installed_revision="abcdef1234567890",
            installed_at="2026-02-13T23:11:29Z",
            content_fingerprint="sha256:" + ("0" * 64),
        ),
    )

    provenance_text, installed_text = format_skill_provenance_details(managed_dir)
    assert provenance_text == "https://github.com/example/skills@main (skills/x)"
    assert installed_text == "2026-02-13 23:11:29 revision: abcdef1"


def test_order_skill_directories_for_display_puts_manager_dir_last(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"environment_dir": str(tmp_path / ".fast-agent")})
    manager_dir = get_manager_directory(settings, cwd=tmp_path)
    claude_dir = (tmp_path / ".claude" / "skills").resolve()
    agents_dir = (tmp_path / ".agents" / "skills").resolve()
    directories = [manager_dir, agents_dir, claude_dir]

    ordered = order_skill_directories_for_display(
        directories,
        settings=settings,
        cwd=tmp_path,
    )

    assert ordered == [agents_dir, claude_dir, manager_dir]


def test_resolve_skills_management_scope_uses_override_for_management_only(tmp_path: Path) -> None:
    configured_dir = (tmp_path / "project-skills").resolve()
    override_dir = (tmp_path / "managed-skills").resolve()
    settings = get_settings().model_copy(
        update={
            "environment_dir": str(tmp_path / ".fast-agent"),
            "skills": SkillsSettings(directories=[str(configured_dir)]),
        }
    )

    scope = resolve_skills_management_scope(
        settings,
        cwd=tmp_path,
        managed_directory_override=override_dir,
    )

    assert scope.management_source == "override"
    assert scope.managed_directory == override_dir
    assert scope.discovered_directories == [configured_dir, override_dir]


def test_get_manager_directory_accepts_explicit_management_override(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"environment_dir": str(tmp_path / ".fast-agent")})
    override_dir = tmp_path / "skills-manager"

    managed_dir = get_manager_directory(
        settings,
        cwd=tmp_path,
        managed_directory_override=override_dir,
    )

    assert managed_dir == override_dir.resolve()


def test_resolve_source_revision_prefers_peeled_annotated_tag_commit() -> None:
    source = _make_remote_source("v1.2.3")

    def _fake_run(args, capture_output, text, check):
        assert args == ["git", "ls-remote", source.repo_url, source.repo_ref]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                "1111111111111111111111111111111111111111\trefs/tags/v1.2.3\n"
                "2222222222222222222222222222222222222222\trefs/tags/v1.2.3^{}\n"
            ),
            stderr="",
        )

    resolution = resolve_source_revision(
        source,
        {},
        resolve_local_repo_fn=lambda _: None,
        run_subprocess_fn=_fake_run,
    )

    assert resolution.revision == "2222222222222222222222222222222222222222"
    assert resolution.status is None
    assert resolution.detail is None


def test_parse_ls_remote_commit_falls_back_when_no_peeled_ref() -> None:
    output = (
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/heads/main\n"
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\tHEAD\n"
    )
    assert parse_ls_remote_commit(output) == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
