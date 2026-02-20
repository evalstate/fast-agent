"""Tests for AgentCompleter sub-completion functionality."""

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import fast_agent.config as config_module
from fast_agent.config import (
    MCPServerSettings,
    MCPSettings,
    Settings,
    SkillsSettings,
    get_settings,
    update_global_settings,
)
from fast_agent.session import get_session_manager, reset_session_manager
from fast_agent.skills.manager import InstalledSkillSource, write_installed_skill_source
from fast_agent.ui.enhanced_prompt import AgentCompleter

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


class _McpAgentStub:
    def __init__(self, attached: list[str]) -> None:
        self.aggregator = self
        self._attached = attached

    def list_attached_servers(self) -> list[str]:
        return list(self._attached)


class _ProviderStub:
    def __init__(self, agent: object) -> None:
        self._agent_obj = agent

    def _agent(self, _name: str) -> object:
        return self._agent_obj


def test_complete_history_files_finds_json_and_md():
    """Test that _complete_history_files finds .json and .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "history.json").touch()
        (Path(tmpdir) / "notes.md").touch()
        (Path(tmpdir) / "other.txt").touch()
        (Path(tmpdir) / "data.py").touch()

        completer = AgentCompleter(agents=["agent1"])

        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files(""))
            names = [c.text for c in completions]

            # Should find .json and .md files, not .txt or .py
            assert "history.json" in names
            assert "notes.md" in names
            assert "other.txt" not in names
            assert "data.py" not in names
        finally:
            os.chdir(original_cwd)


def test_complete_history_files_includes_directories():
    """Test that directories are included in completions for navigation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a subdirectory
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.json").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files(""))
            names = [c.text for c in completions]

            # Should include directory with trailing slash
            assert "subdir/" in names
        finally:
            os.chdir(original_cwd)


def test_complete_history_files_filters_by_prefix():
    """Test that completions are filtered by prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "history.json").touch()
        (Path(tmpdir) / "other.md").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files("his"))
            names = [c.text for c in completions]

            assert "history.json" in names
            assert "other.md" not in names
        finally:
            os.chdir(original_cwd)


def test_complete_history_files_handles_subdirectory():
    """Test completion works in subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = Path(tmpdir) / "data"
        subdir.mkdir()
        (subdir / "history.json").touch()
        (subdir / "notes.md").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files("data/"))
            names = [c.text for c in completions]

            assert "data/history.json" in names
            assert "data/notes.md" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_history_load_command():
    """Test get_completions provides file completions after /history load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.json").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Simulate typing "/history load "
            doc = Document("/history load ", cursor_position=14)
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "test.json" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_shell_path_prefix():
    """Ensure shell completions treat path-like tokens as paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "script.sh").touch()
        subdir = Path(tmpdir) / "data"
        subdir.mkdir()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            doc = Document("!./", cursor_position=len("!./"))
            event = CompleteEvent(completion_requested=True)
            completions = list(completer.get_completions(doc, event))
            names = [c.text for c in completions]

            assert "./script.sh" in names
            assert "./data/" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_shell_path_prefix_with_current_dir_partial():
    """Ensure ./ prefix is preserved when completing in the current directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "script.sh").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            doc = Document("!./s", cursor_position=len("!./s"))
            event = CompleteEvent(completion_requested=True)
            completions = list(completer.get_completions(doc, event))
            names = [c.text for c in completions]

            assert "./script.sh" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_history_subcommands():
    """Test get_completions suggests /history subcommands."""
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/history ", cursor_position=9)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "show" in names
    assert "save" in names
    assert "load" in names
    assert "webclear" not in names


def test_get_completions_for_history_subcommands_includes_webclear_when_enabled() -> None:
    class _LlmStub:
        web_tools_enabled = (True, False)

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/history ", cursor_position=9)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "webclear" in names


def test_get_completions_for_history_subcommands_includes_webclear_when_web_search_enabled_bool() -> None:
    class _LlmStub:
        web_search_enabled = True

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/history ", cursor_position=9)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "webclear" in names


def test_get_completions_for_history_subcommands_includes_webclear_when_web_fetch_only_enabled() -> None:
    class _LlmStub:
        web_search_enabled = False
        web_tools_enabled = (False, True)

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/history ", cursor_position=9)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "webclear" in names


def test_get_completions_for_model_subcommands_includes_web_search_when_supported() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        web_search_supported = True
        web_fetch_supported = False

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model ", cursor_position=len("/model "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "reasoning" in names
    assert "web_search" in names
    assert "web_fetch" not in names


def test_get_completions_for_model_subcommands_includes_web_fetch_when_supported() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        web_search_supported = True
        web_fetch_supported = True

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model ", cursor_position=len("/model "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "web_search" in names
    assert "web_fetch" in names


def test_get_completions_for_model_web_search_values() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        web_search_supported = True
        web_fetch_supported = False

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model web_search ", cursor_position=len("/model web_search "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "on" in names
    assert "off" in names
    assert "default" in names


def test_get_completions_for_model_web_fetch_values_omits_unsupported_setting() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        web_search_supported = True
        web_fetch_supported = False

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model web_fetch ", cursor_position=len("/model web_fetch "))
    completions = list(completer.get_completions(doc, None))

    assert completions == []


def test_get_completions_for_session_pin(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(update={"environment_dir": str(env_dir)})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        session = manager.create_session()

        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/session pin ", cursor_position=len("/session pin "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "on" in names
        assert "off" in names
        assert session.info.name in names

        doc = Document("/session pin on ", cursor_position=len("/session pin on "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]
        assert session.info.name in names
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def test_noenv_session_completion_does_not_create_session_storage(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(update={"environment_dir": str(env_dir)})
    update_global_settings(override)
    reset_session_manager()

    try:
        completer = AgentCompleter(agents=["agent1"], noenv_mode=True)
        doc = Document("/resume ", cursor_position=len("/resume "))
        completions = list(completer.get_completions(doc, None))

        assert completions == []
        assert not (env_dir / "sessions").exists()
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def _write_skill(skill_root: Path, name: str) -> None:
    skill_dir = skill_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: {name}\ndescription: Test skill\n---\n".format(name=name),
        encoding="utf-8",
    )


def _mark_skill_managed(skill_root: Path, name: str) -> None:
    skill_dir = skill_root / name
    write_installed_skill_source(
        skill_dir,
        InstalledSkillSource(
            schema_version=1,
            installed_via="marketplace",
            source_origin="remote",
            repo_url="https://github.com/example/skills",
            repo_ref="main",
            repo_path=f"skills/{name}",
            source_url="https://raw.githubusercontent.com/example/skills/main/marketplace.json",
            installed_commit="abcdef1234567890",
            installed_path_oid="def456",
            installed_revision="abcdef1234567890",
            installed_at="2026-02-15T00:00:00Z",
            content_fingerprint="sha256:deadbeef",
        ),
    )


def test_get_completions_for_skills_subcommands():
    """Test get_completions suggests /skills subcommands."""
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/skills ", cursor_position=8)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "list" in names
    assert "add" in names
    assert "remove" in names
    assert "update" in names
    assert "registry" in names


def test_get_completions_for_mcp_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp ", cursor_position=len("/mcp "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "list" in names
    assert "connect" in names
    assert "disconnect" in names


def test_get_completions_for_mcp_disconnect_servers() -> None:
    provider = _ProviderStub(_McpAgentStub(["local", "docs"]))
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", provider),
    )

    doc = Document("/mcp disconnect d", cursor_position=len("/mcp disconnect d"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "docs" in names


def test_get_completions_for_mcp_connect_flags() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp connect npx demo-server --re", cursor_position=len("/mcp connect npx demo-server --re"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "--reconnect" in names


def test_get_completions_for_mcp_connect_hides_flags_before_target() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp connect --re", cursor_position=len("/mcp connect --re"))
    completions = list(completer.get_completions(doc, None))

    assert completions == []


def test_get_completions_for_mcp_connect_configured_servers(monkeypatch) -> None:
    settings = Settings(
        mcp=MCPSettings(
            servers={
                "docs": MCPServerSettings(name="docs", transport="stdio", command="echo"),
                "local": MCPServerSettings(name="local", transport="stdio", command="echo"),
            }
        )
    )
    monkeypatch.setattr(config_module, "_settings", settings)

    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp connect d", cursor_position=len("/mcp connect d"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "docs" in names
    assert "--name" not in names


def test_get_completions_for_mcp_connect_shows_target_hint_first(monkeypatch) -> None:
    settings = Settings(
        mcp=MCPSettings(
            servers={
                "docs": MCPServerSettings(name="docs", transport="stdio", command="echo"),
            }
        )
    )
    monkeypatch.setattr(config_module, "_settings", settings)

    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp connect d", cursor_position=len("/mcp connect d"))
    completions = list(completer.get_completions(doc, None))

    assert completions
    assert completions[0].display_text == "[url|npx|uvx]"
    assert completions[0].display_meta_text == "enter url or npx/uvx cmd"


def test_get_completions_for_skills_remove(monkeypatch):
    """Test get_completions suggests local skills for /skills remove."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_root = Path(tmpdir) / "skills"
        _write_skill(skills_root, "alpha")
        _write_skill(skills_root, "beta")

        settings = Settings(skills=SkillsSettings(directories=[str(skills_root)]))
        monkeypatch.setattr(config_module, "_settings", settings)

        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/skills remove ", cursor_position=15)
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "alpha" in names
        assert "beta" in names


def test_get_completions_for_skills_registry(monkeypatch):
    """Test get_completions suggests registry choices for /skills registry."""
    settings = Settings(
        skills=SkillsSettings(
            marketplace_urls=[
                "https://example.com/registry-one.json",
                "https://example.com/registry-two.json",
            ]
        )
    )
    monkeypatch.setattr(config_module, "_settings", settings)

    completer = AgentCompleter(agents=["agent1"])
    doc = Document("/skills registry ", cursor_position=17)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "1" in names
    assert "2" in names


def test_get_completions_for_skills_update_only_managed():
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_root = Path(tmpdir) / "skills"
        _write_skill(skills_root, "alpha")
        _write_skill(skills_root, "beta")
        _write_skill(skills_root, "gamma")
        _mark_skill_managed(skills_root, "beta")
        _mark_skill_managed(skills_root, "gamma")

        old_settings = get_settings()
        override = old_settings.model_copy(update={"skills": SkillsSettings(directories=[str(skills_root)])})
        update_global_settings(override)
        try:
            completer = AgentCompleter(agents=["agent1"])
            doc = Document("/skills update ", cursor_position=len("/skills update "))
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "alpha" not in names
            assert "beta" in names
            assert "gamma" in names
            assert "1" not in names
            assert "2" not in names
            assert "3" not in names
        finally:
            update_global_settings(old_settings)


def test_complete_agent_card_files_finds_md_and_yaml():
    """Test that _complete_agent_card_files finds AgentCard files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "agent.md").touch()
        (Path(tmpdir) / "agent.yaml").touch()
        (Path(tmpdir) / "agent.yml").touch()
        (Path(tmpdir) / "agent.txt").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_agent_card_files(""))
            names = [c.text for c in completions]

            assert "agent.md" in names
            assert "agent.yaml" in names
            assert "agent.yml" in names
            assert "agent.txt" not in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_card_command():
    """Test get_completions provides file completions after /card."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "agent.md").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            doc = Document("/card ", cursor_position=6)
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "agent.md" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_skips_hidden_files():
    """Test that hidden files are not included in completions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / ".hidden.json").touch()
        (Path(tmpdir) / "visible.json").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files(""))
            names = [c.text for c in completions]

            assert "visible.json" in names
            assert ".hidden.json" not in names
        finally:
            os.chdir(original_cwd)


def test_completion_metadata():
    """Test that completions have correct metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.json").touch()
        (Path(tmpdir) / "test.md").touch()
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files(""))

            # Check metadata for each completion type
            for c in completions:
                # display_meta can be string or FormattedText, convert to string
                meta = str(c.display_meta) if c.display_meta else ""
                if c.text == "test.json":
                    assert "JSON history" in meta
                elif c.text == "test.md":
                    assert "Markdown" in meta
                elif c.text == "subdir/":
                    assert "directory" in meta
        finally:
            os.chdir(original_cwd)


def test_command_completions_still_work():
    """Test that regular command completions still work."""
    completer = AgentCompleter(agents=["agent1"])

    # Simulate typing "/hist"
    doc = Document("/hist", cursor_position=5)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    # Should complete to history command
    assert "history" in names


def test_agent_completions_still_work():
    """Test that agent completions still work."""
    completer = AgentCompleter(agents=["test_agent", "other_agent"])

    # Simulate typing "@test"
    doc = Document("@test", cursor_position=5)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "test_agent" in names
    assert "other_agent" not in names
