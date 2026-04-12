import json
import subprocess
from pathlib import Path

from click.utils import strip_ansi
from typer.testing import CliRunner

from fast_agent.cli.commands import go as go_command
from fast_agent.paths import resolve_environment_paths


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
        environment_dir=None,
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
        environment_dir=tmp_path,
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


def test_run_async_agent_passes_target_agent_name() -> None:
    run_kwargs = go_command._build_run_agent_kwargs(
        name="test-agent",
        instruction="test instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=["./agents"],
        card_tools=None,
        model=None,
        message="hello",
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name="researcher",
        skills_directory=None,
        environment_dir=None,
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

    assert run_kwargs["target_agent_name"] == "researcher"


def test_run_async_agent_noenv_passes_flag_and_disables_implicit_cards() -> None:
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
        environment_dir=None,
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
        noenv=True,
    )

    assert run_kwargs["noenv"] is True
    assert run_kwargs["agent_cards"] is None
    assert run_kwargs["card_tools"] is None


def test_run_async_agent_passes_result_file() -> None:
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
        message="hello",
        prompt_file=None,
        result_file="result.json",
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
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

    assert run_kwargs["result_file"] == "result.json"


def test_build_compat_run_request_defaults_acp_instance_scope_to_connection() -> None:
    request = go_command._build_compat_run_request(
        name="test-agent",
        instruction="test instruction",
        mode="serve",
        transport="acp",
    )

    assert request.instance_scope == "connection"


def test_run_async_agent_normalizes_legacy_acp_shared_scope(monkeypatch) -> None:
    captured_requests = []

    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    go_command.run_async_agent(
        name="test-agent",
        instruction="test instruction",
        mode="serve",
        transport="acp",
    )

    assert len(captured_requests) == 1
    assert captured_requests[0].instance_scope == "connection"


def test_go_pack_installs_then_runs(tmp_path: Path, monkeypatch) -> None:
    _, marketplace_path = _build_pack_repo(tmp_path)
    env_root = tmp_path / ".fast-agent-demo"
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
            "--env",
            env_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Installed card pack: alpha" in result.output
    assert f"Launching fast-agent go with environment: {env_root}" in result.output
    assert (env_root / "agent-cards" / "alpha.md").exists()
    assert len(captured_requests) == 1
    assert captured_requests[0].environment_dir == env_root
    assert captured_requests[0].model == "haiku"
    assert captured_requests[0].agent_cards == [str(env_root / "agent-cards")]


def test_go_pack_reuses_installed_pack_without_registry_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, marketplace_path = _build_pack_repo(tmp_path)
    env_root = tmp_path / ".fast-agent-demo"
    env_paths = resolve_environment_paths(override=env_root, cwd=tmp_path)
    captured_requests = []

    go_command.card_service.install_pack_sync(
        marketplace_path.as_posix(),
        "alpha",
        environment_paths=env_paths,
        force=False,
    )

    async def _fail_install_pack(*_args, **_kwargs):
        raise AssertionError("Marketplace lookup should be skipped for installed packs.")

    monkeypatch.setattr(go_command.card_service, "install_pack", _fail_install_pack)
    monkeypatch.setattr(go_command, "run_request", captured_requests.append)

    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        [
            "--pack",
            "alpha",
            "--model",
            "haiku",
            "--env",
            env_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Using installed card pack: alpha" in result.output
    assert len(captured_requests) == 1
    assert captured_requests[0].environment_dir == env_root


def test_go_pack_rejects_noenv() -> None:
    runner = CliRunner()
    result = runner.invoke(
        go_command.app,
        ["--pack", "alpha", "--noenv"],
    )

    assert result.exit_code == 2
    assert "Cannot combine --pack with --noenv." in strip_ansi(result.output)


def test_go_pack_preserves_agent_target(tmp_path: Path, monkeypatch) -> None:
    _, marketplace_path = _build_pack_repo(
        tmp_path,
        agent_names=("alpha", "planner"),
    )
    env_root = tmp_path / ".fast-agent-demo"
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
            "--env",
            env_root.as_posix(),
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
    env_root = tmp_path / ".fast-agent-demo"
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
            "--env",
            env_root.as_posix(),
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
        str(env_root / "agent-cards"),
    ]
    assert captured_requests[0].card_tools == [
        explicit_tool_dir.as_posix(),
        str(env_root / "tool-cards"),
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
            "--env",
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
    env_root = tmp_path / ".fast-agent-demo"
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
            "--env",
            env_root.as_posix(),
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
    env_root = tmp_path / ".fast-agent-demo"

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
            "--env",
            env_root.as_posix(),
            "--message",
            "hello",
        ],
    )

    assert result.exit_code == 0, result.output
