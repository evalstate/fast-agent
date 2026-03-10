from __future__ import annotations

import json
import subprocess
from pathlib import Path

from click.utils import strip_ansi
from typer.testing import CliRunner

import fast_agent.cli.commands.skills as skills_command
from fast_agent.cli.main import LAZY_SUBCOMMANDS
from fast_agent.config import get_settings, update_global_settings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


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


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "test skill",
    body: str = "Test skill body.",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _marketplace_path(repo: Path, path: Path, *, skill_name: str = "alpha") -> Path:
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": skill_name,
                        "description": f"{skill_name} description",
                        "repo_url": repo.as_posix(),
                        "repo_path": f"skills/{skill_name}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_skills_lazy_subcommand_registered() -> None:
    assert LAZY_SUBCOMMANDS["skills"] == "fast_agent.cli.commands.skills:app"


def test_skills_add_list_remove_via_cli(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo / "skills", "alpha", description="MAGIC_SKILL")
    _commit_all(repo, "initial")

    marketplace_path = _marketplace_path(repo, tmp_path / "marketplace.json")

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
            skills_command.app,
            ["add", "alpha", "--registry", marketplace_path.as_posix()],
            terminal_width=200,
        )
        assert add_result.exit_code == 0, add_result.output
        assert "Skill Installed" in add_result.output
        assert "name: alpha" in add_result.output
        assert "▎• location" not in add_result.output
        assert (env_root / "skills" / "alpha" / "SKILL.md").exists()

        list_result = runner.invoke(skills_command.app, ["list"], terminal_width=200)
        assert list_result.exit_code == 0, list_result.output
        assert "alpha" in list_result.output
        assert "managed directory:" in list_result.output

        remove_result = runner.invoke(
            skills_command.app,
            ["remove", "alpha"],
            terminal_width=200,
        )
        assert remove_result.exit_code == 0, remove_result.output
        assert "Skill Removed" in remove_result.output
        assert "name: alpha" in remove_result.output
        assert not (env_root / "skills" / "alpha").exists()
    finally:
        update_global_settings(old_settings)


def test_skills_help_has_registry_and_skills_dir_options_no_registry_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(skills_command.app, ["--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "--env" in output
    assert "--registry" in output
    assert "--skills-dir" in output
    assert "--install-completion" not in output
    assert "--show-completion" not in output
    assert "│ registry" not in output


def test_top_level_env_flag_routes_to_skills_subcommand(tmp_path: Path) -> None:
    env_root = tmp_path / "custom-env"
    managed_dir = tmp_path / "env-skills"
    _write_skill(managed_dir, "env-skill", description="ENV_SKILL")
    (env_root / "fastagent.config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (env_root / "fastagent.config.yaml").write_text(
        "default_model: passthrough\n"
        "skills:\n"
        f"  directories: ['{managed_dir.as_posix()}']\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "fast_agent.cli",
            "--env",
            str(env_root),
            "skills",
            "list",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )

    assert result.returncode == 0, result.stderr
    assert "env-skill" in result.stdout
    assert managed_dir.as_posix() in result.stdout


def test_local_skills_env_flag_routes_to_skills_subcommand(tmp_path: Path) -> None:
    env_root = tmp_path / "custom-env"
    managed_dir = tmp_path / "env-skills"
    _write_skill(managed_dir, "env-skill", description="ENV_SKILL")
    (env_root / "fastagent.config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (env_root / "fastagent.config.yaml").write_text(
        "default_model: passthrough\n"
        "skills:\n"
        f"  directories: ['{managed_dir.as_posix()}']\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "fast_agent.cli",
            "skills",
            "--env",
            str(env_root),
            "list",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )

    assert result.returncode == 0, result.stderr
    assert "env-skill" in result.stdout
    assert managed_dir.as_posix() in result.stdout


def test_skills_add_uses_configured_marketplace_urls_by_default(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo / "skills", "alpha")
    _commit_all(repo, "initial")

    marketplace_path = _marketplace_path(repo, tmp_path / "marketplace.json")
    manager_dir = tmp_path / "managed-skills"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "skills:\n"
        f"  directories: ['{manager_dir.as_posix()}']\n"
        "  marketplace_urls:\n"
        f"    - '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()
        add_result = runner.invoke(skills_command.app, ["add", "alpha"], terminal_width=200)

        assert add_result.exit_code == 0, add_result.output
        assert "Skill Installed" in add_result.output
        assert "name: alpha" in add_result.output
        assert (manager_dir / "alpha" / "SKILL.md").exists()
    finally:
        update_global_settings(old_settings)


def test_skills_dir_override_changes_management_target_and_list_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo / "skills", "alpha", description="override install target")
    _commit_all(repo, "initial")

    configured_dir = tmp_path / "configured-skills"
    override_dir = tmp_path / "managed-skills"
    _write_skill(configured_dir, "configured-only", description="configured source")

    marketplace_path = _marketplace_path(repo, tmp_path / "marketplace.json")
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "skills:\n"
        f"  directories: ['{configured_dir.as_posix()}']\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()
        add_result = runner.invoke(
            skills_command.app,
            ["add", "alpha", "--skills-dir", override_dir.as_posix()],
            terminal_width=200,
        )

        assert add_result.exit_code == 0, add_result.output
        assert (override_dir / "alpha" / "SKILL.md").exists()
        assert not (configured_dir / "alpha").exists()

        list_result = runner.invoke(
            skills_command.app,
            ["list", "--skills-dir", override_dir.as_posix()],
            terminal_width=200,
        )

        assert list_result.exit_code == 0, list_result.output
        assert "configured-only" in list_result.output
        assert "alpha" in list_result.output
        assert "managed directory:" in list_result.output
        assert "(managed)" in list_result.output
    finally:
        update_global_settings(old_settings)


def test_skills_search_filters_marketplace_results(tmp_path: Path) -> None:
    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "docker-helper",
                        "description": "Manage docker containers",
                        "repo_url": tmp_path.as_posix(),
                        "repo_path": "skills/docker-helper",
                    },
                    {
                        "name": "pdf-reader",
                        "description": "Read PDF files",
                        "repo_url": tmp_path.as_posix(),
                        "repo_path": "skills/pdf-reader",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        skills_command.app,
        ["search", "docker containers", "--registry", marketplace_path.as_posix()],
        terminal_width=200,
    )

    assert result.exit_code == 0, result.output
    assert "docker-helper" in result.output
    assert "pdf-reader" not in result.output


def test_skills_update_applies_local_repo_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo / "skills", "alpha", body="Version 1 body.")
    _commit_all(repo, "initial")

    marketplace_path = _marketplace_path(repo, tmp_path / "marketplace.json")
    manager_dir = tmp_path / "managed-skills"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "skills:\n"
        f"  directories: ['{manager_dir.as_posix()}']\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()
        add_result = runner.invoke(skills_command.app, ["add", "alpha"], terminal_width=200)
        assert add_result.exit_code == 0, add_result.output

        _write_skill(repo / "skills", "alpha", body="Version 2 body.")
        _commit_all(repo, "update alpha")

        check_result = runner.invoke(skills_command.app, ["update"], terminal_width=200)
        assert check_result.exit_code == 0, check_result.output
        assert "update available" in check_result.output

        apply_result = runner.invoke(skills_command.app, ["update", "alpha"], terminal_width=200)
        assert apply_result.exit_code == 0, apply_result.output
        assert "updated" in apply_result.output
        assert "Version 2 body." in (manager_dir / "alpha" / "SKILL.md").read_text(encoding="utf-8")
    finally:
        update_global_settings(old_settings)
