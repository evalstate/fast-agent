import json
import subprocess
from pathlib import Path

import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

from fast_agent.cli.commands import go as go_command
from fast_agent.cli.main import app as root_app
from fast_agent.paths import resolve_home_paths


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


def _build_pack_repo(
    tmp_path: Path,
    *,
    pack_name: str = "alpha",
    agent_names: tuple[str, ...] = ("alpha",),
    tool_names: tuple[str, ...] = (),
    readme: str | None = None,
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / pack_name
    agent_cards_dir = pack_root / "agent-cards"
    agent_cards_dir.mkdir(parents=True)
    tool_cards_dir = pack_root / "tool-cards"

    for agent_name in agent_names:
        (agent_cards_dir / f"{agent_name}.md").write_text(
            f"---\nname: {agent_name}\nmodel: passthrough\n---\n\nhello\n",
            encoding="utf-8",
        )

    for tool_name in tool_names:
        tool_cards_dir.mkdir(parents=True, exist_ok=True)
        (tool_cards_dir / f"{tool_name}.md").write_text(
            f"---\nname: {tool_name}\n---\n\nhello\n",
            encoding="utf-8",
        )

    manifest_lines = [
        "schema_version: 1",
        f"name: {pack_name}",
        "kind: card",
        "install:",
        "  agent_cards:",
        *[f"    - 'agent-cards/{agent_name}.md'" for agent_name in agent_names],
        "  tool_cards:",
        *([f"    - 'tool-cards/{tool_name}.md'" for tool_name in tool_names] or ["    []"]),
        "  files: []",
        "",
    ]
    (pack_root / "card-pack.yaml").write_text(
        "\n".join(manifest_lines),
        encoding="utf-8",
    )
    if readme is not None:
        (pack_root / "README.md").write_text(readme, encoding="utf-8")

    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": pack_name,
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": f"packs/{pack_name}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    return repo, marketplace_path


def test_run_async_agent_passes_card_tools() -> None:
    run_kwargs = go_command._build_run_agent_kwargs(
        name="test-agent",
        instruction="test instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=["./agents"],
        card_tools=["./tool-cards"],
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        home=None,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    assert run_kwargs["card_tools"] == ["./tool-cards"]


def test_run_async_agent_merges_default_tool_cards(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tool-cards"
    tool_dir.mkdir()
    (tool_dir / "sizer.md").write_text("---\nname: sizer\n---\n")

    run_kwargs = go_command._build_run_agent_kwargs(
        name="test-agent",
        instruction="test instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        home=tmp_path,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    assert run_kwargs["card_tools"] == [str(tool_dir)]


def test_run_async_agent_no_home_passes_flag_and_disables_implicit_cards() -> None:
    run_kwargs = go_command._build_run_agent_kwargs(
        name="test-agent",
        instruction="test instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        home=None,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
        no_home=True,
    )

    assert run_kwargs["no_home"] is True
    assert run_kwargs["agent_cards"] is None
    assert run_kwargs["card_tools"] is None


def test_go_accepts_repeated_attach_flags(monkeypatch, tmp_path: Path) -> None:
    attachment = tmp_path / "report.txt"
    attachment.write_text("hello", encoding="utf-8")
    captured_requests = []

    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--message",
            "summarize",
            "--attach",
            attachment.as_posix(),
            "--attach",
            "https://example.com/chart.png",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 1
    assert captured_requests[0].attachments == [
        attachment.as_posix(),
        "https://example.com/chart.png",
    ]


def test_go_accepts_timeout_flag(monkeypatch) -> None:
    captured_requests = []
    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--message",
            "summarize",
            "--timeout",
            "120",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 1
    assert captured_requests[0].timeout_seconds == 120


def test_go_workspace_sets_default_home_base(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured_requests = []
    original_cwd = Path.cwd()

    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--workspace",
            workspace.as_posix(),
            "--message",
            "summarize",
        ],
    )

    assert result.exit_code == 0, result.output
    assert Path.cwd() == original_cwd
    assert len(captured_requests) == 1
    assert captured_requests[0].home == workspace / ".fast-agent"


def test_go_workspace_resolves_relative_home(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured_requests = []

    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--workspace",
            workspace.as_posix(),
            "--home",
            "custom-home",
            "--message",
            "summarize",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 1
    assert captured_requests[0].home == workspace / "custom-home"


def test_root_relative_workspace_is_not_resolved_twice(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "demo"
    workspace.mkdir()
    captured_requests = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        root_app,
        [
            "--workspace",
            "demo",
            "go",
            "--message",
            "summarize",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 1
    assert captured_requests[0].home == workspace / ".fast-agent"


def test_go_workspace_rejects_missing_directory(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--workspace",
            (tmp_path / "missing").as_posix(),
            "--message",
            "summarize",
        ],
    )

    assert result.exit_code == 2
    assert (tmp_path / "missing").as_posix() in strip_ansi(result.output)


def test_go_attach_requires_one_shot_mode() -> None:
    runner = CliRunner()
    result = runner.invoke(go_command.app, ["--attach", "report.txt"])

    assert result.exit_code == 2
    assert "--attach requires --message or --prompt-file" in strip_ansi(result.output)


def test_go_pack_installs_then_runs(tmp_path: Path, monkeypatch) -> None:
    _, marketplace_path = _build_pack_repo(tmp_path)
    home_root = tmp_path / ".fast-agent-demo"
    captured_requests = []

    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--pack",
            "alpha",
            "--pack-registry",
            marketplace_path.as_posix(),
            "--model",
            "haiku",
            "--home",
            home_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Installed card pack: alpha" in result.output
    assert f"Launching fast-agent go with home: {home_root}" in result.output
    assert (home_root / "agent-cards" / "alpha.md").exists()
    assert len(captured_requests) == 1
    assert captured_requests[0].home == home_root
    assert captured_requests[0].model == "haiku"
    assert captured_requests[0].agent_cards == [str(home_root / "agent-cards")]


def test_go_pack_defaults_home_under_workspace(tmp_path: Path, monkeypatch) -> None:
    _, marketplace_path = _build_pack_repo(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    expected_home = workspace / ".fast-agent"
    captured_requests = []

    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--workspace",
            workspace.as_posix(),
            "--pack",
            "alpha",
            "--pack-registry",
            marketplace_path.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Launching fast-agent go with home: {expected_home}" in result.output
    assert (expected_home / "agent-cards" / "alpha.md").exists()
    assert len(captured_requests) == 1
    assert captured_requests[0].home == expected_home
    assert captured_requests[0].agent_cards == [str(expected_home / "agent-cards")]


def test_go_pack_reuses_installed_pack_without_registry_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fast_agent.cards import service as card_service

    _, marketplace_path = _build_pack_repo(tmp_path)
    home_root = tmp_path / ".fast-agent-demo"
    home_paths = resolve_home_paths(override=home_root, cwd=tmp_path)
    captured_requests = []

    card_service.install_pack_sync(
        marketplace_path.as_posix(),
        "alpha",
        home_paths=home_paths,
        force=False,
    )

    async def _fail_install_pack(*_args, **_kwargs):
        raise AssertionError("Marketplace lookup should be skipped for installed packs.")

    monkeypatch.setattr(card_service, "install_pack", _fail_install_pack)
    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--pack",
            "alpha",
            "--model",
            "haiku",
            "--home",
            home_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Using installed card pack: alpha" in result.output
    assert len(captured_requests) == 1
    assert captured_requests[0].home == home_root


def test_go_pack_rejects_no_home() -> None:
    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        ["--pack", "alpha", "--no-home"],
    )

    assert result.exit_code == 2
    assert "Cannot combine --pack with --no-home." in strip_ansi(result.output)


def test_go_pack_preserves_agent_target(tmp_path: Path, monkeypatch) -> None:
    _, marketplace_path = _build_pack_repo(
        tmp_path,
        agent_names=("alpha", "planner"),
    )
    home_root = tmp_path / ".fast-agent-demo"
    captured_requests = []

    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--pack",
            "alpha",
            "--pack-registry",
            marketplace_path.as_posix(),
            "--model",
            "haiku",
            "--home",
            home_root.as_posix(),
            "--agent",
            "planner",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 1
    assert captured_requests[0].target_agent_name == "planner"


def test_go_pack_keeps_installed_card_dirs_after_explicit_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, marketplace_path = _build_pack_repo(
        tmp_path,
        tool_names=("alpha-tool",),
    )
    home_root = tmp_path / ".fast-agent-demo"
    explicit_agent_dir = tmp_path / "extra-agent-cards"
    explicit_tool_dir = tmp_path / "extra-tool-cards"
    explicit_agent_dir.mkdir()
    explicit_tool_dir.mkdir()
    captured_requests = []

    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--pack",
            "alpha",
            "--pack-registry",
            marketplace_path.as_posix(),
            "--home",
            home_root.as_posix(),
            "--agent-cards",
            explicit_agent_dir.as_posix(),
            "--card-tool",
            explicit_tool_dir.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 1
    assert captured_requests[0].agent_cards == [
        explicit_agent_dir.as_posix(),
        str(home_root / "agent-cards"),
    ]
    assert captured_requests[0].card_tools == [
        explicit_tool_dir.as_posix(),
        str(home_root / "tool-cards"),
    ]


def test_go_pack_reports_missing_pack(tmp_path: Path, monkeypatch) -> None:
    _, marketplace_path = _build_pack_repo(tmp_path)

    def _fail_run_request(_request):
        raise AssertionError("run_request should not be called when pack lookup fails.")

    monkeypatch.setattr(go_command, "run_request", _fail_run_request)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--pack",
            "missing",
            "--pack-registry",
            marketplace_path.as_posix(),
            "--home",
            (tmp_path / ".fast-agent-demo").as_posix(),
        ],
    )

    assert result.exit_code == 1
    assert "Card pack not found: missing" in result.output


def test_go_pack_queues_readme_notice_for_interactive_startup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, marketplace_path = _build_pack_repo(
        tmp_path,
        readme="# Alpha Pack\n\nInstall notes.\n",
    )
    home_root = tmp_path / ".fast-agent-demo"
    plain_notices: list[str] = []
    markdown_notices: list[tuple[str, dict[str, str | None]]] = []

    def _capture_markdown_notice(text: str, **kwargs: str | None) -> None:
        markdown_notices.append((text, kwargs))

    monkeypatch.setattr(go_command, "run_request", lambda _request: None)
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_notice",
        plain_notices.append,
    )
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_markdown_notice",
        _capture_markdown_notice,
    )

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--pack",
            "alpha",
            "--pack-registry",
            marketplace_path.as_posix(),
            "--home",
            home_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert any("Card pack README" in notice for notice in plain_notices)
    assert markdown_notices == [
        (
            "# Alpha Pack\n\nInstall notes.",
            {
                "title": "alpha README",
                "right_info": "card pack",
            },
        )
    ]


def test_go_pack_skips_readme_notice_for_noninteractive_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, marketplace_path = _build_pack_repo(
        tmp_path,
        readme="# Alpha Pack\n\nInstall notes.\n",
    )
    home_root = tmp_path / ".fast-agent-demo"

    monkeypatch.setattr(go_command, "run_request", lambda _request: None)
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_notice",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("plain startup notice should not be queued")
        ),
    )
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_markdown_notice",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("markdown startup notice should not be queued")
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--pack",
            "alpha",
            "--pack-registry",
            marketplace_path.as_posix(),
            "--home",
            home_root.as_posix(),
            "--message",
            "hello",
        ],
    )

    assert result.exit_code == 0, result.output


def test_go_quiet_skips_update_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(go_command, "run_request", lambda _request: None)
    monkeypatch.setattr(
        "fast_agent.cli.update_check.should_run_update_check",
        lambda *, disabled: not disabled,
    )

    def _unexpected_update_check(*, home: Path | None) -> str | None:
        raise AssertionError(f"quiet mode should not check for updates: {home}")

    monkeypatch.setattr("fast_agent.cli.update_check.check_for_update_notice", _unexpected_update_check)
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_notice",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("startup notice should not be queued in quiet mode")
        ),
    )

    runner = CliRunner()
    result = runner.invoke(go_command.app, ["--quiet"])

    assert result.exit_code == 0, result.output
