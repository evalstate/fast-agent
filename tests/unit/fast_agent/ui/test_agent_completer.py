"""Tests for AgentCompleter sub-completion functionality."""

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from mcp.types import Completion as MCPCompletion
from mcp.types import ResourceTemplate
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import fast_agent.config as config_module
from fast_agent.cards.manager import InstalledCardPackSource, write_installed_card_pack_source
from fast_agent.config import (
    CardsSettings,
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


class _McpSessionClientStub:
    async def list_server_cookies(self, server_identifier: str | None):
        if server_identifier not in {"demo", "demo-server"}:
            return "other", "other-server", None, []
        return "demo", "demo-server", "sess-123", [
            {"id": "sess-123", "title": "Current", "active": True},
            {"id": "sess-456", "title": "Older", "active": False},
        ]


class _McpSessionAgentStub:
    def __init__(self) -> None:
        self.aggregator = self
        self.experimental_sessions = _McpSessionClientStub()

    def list_attached_servers(self) -> list[str]:
        return ["demo"]


class _MentionAggregatorStub:
    def __init__(self) -> None:
        self._templates = {
            "demo": [
                ResourceTemplate(name="repo", uriTemplate="repo://items/{id}"),
                ResourceTemplate(name="repo_pair", uriTemplate="repo://items/{owner}/{repo}"),
                ResourceTemplate(name="repo_resource", uriTemplate="repo://items/{resourceId}"),
                ResourceTemplate(
                    name="repo_contents",
                    uriTemplate="repo://{owner}/{repo}/contents{/path*}",
                ),
            ]
        }
        self.last_completion_request: dict[str, object] | None = None

    async def collect_server_status(self):
        return {
            "demo": SimpleNamespace(
                is_connected=True,
                server_capabilities=SimpleNamespace(resources=True),
            )
        }

    def list_attached_servers(self) -> list[str]:
        return ["demo"]

    def list_configured_detached_servers(self) -> list[str]:
        return []

    async def list_resource_templates(self, server_name: str | None = None):
        if server_name:
            return {server_name: self._templates.get(server_name, [])}
        return dict(self._templates)

    async def complete_resource_argument(
        self,
        server_name: str,
        template_uri: str,
        argument_name: str,
        value: str,
        context_args=None,
    ):
        self.last_completion_request = {
            "server_name": server_name,
            "template_uri": template_uri,
            "argument_name": argument_name,
            "value": value,
            "context_args": context_args,
        }
        values = ["123", "789"]
        return MCPCompletion(values=[item for item in values if item.startswith(value)])


class _MentionAgentStub:
    def __init__(self) -> None:
        self.aggregator = _MentionAggregatorStub()

    async def list_resources(self, namespace: str | None = None):
        if namespace == "demo":
            return {"demo": ["repo://items/123", "repo://items/456"]}
        return {}


class _MentionFilteredAggregatorStub(_MentionAggregatorStub):
    async def collect_server_status(self):
        return {
            "demo": SimpleNamespace(
                is_connected=True,
                server_capabilities=SimpleNamespace(resources=True),
            ),
            "offline": SimpleNamespace(
                is_connected=False,
                server_capabilities=SimpleNamespace(resources=True),
            ),
            "nores": SimpleNamespace(
                is_connected=True,
                server_capabilities=SimpleNamespace(resources=None),
            ),
        }


class _MentionFilteredAgentStub(_MentionAgentStub):
    def __init__(self) -> None:
        self.aggregator = _MentionFilteredAggregatorStub()


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


def _write_card_pack(card_pack_root: Path, name: str) -> None:
    pack_dir = card_pack_root / name
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "card-pack.yaml").write_text(
        "schema_version: 1\n"
        f"name: {name}\n"
        "kind: card\n"
        "install:\n"
        f"  agent_cards: ['agent-cards/{name}.md']\n"
        "  tool_cards: []\n"
        "  files: []\n",
        encoding="utf-8",
    )


def _mark_card_pack_managed(card_pack_root: Path, name: str) -> None:
    pack_dir = card_pack_root / name
    write_installed_card_pack_source(
        pack_dir,
        InstalledCardPackSource(
            schema_version=1,
            installed_via="marketplace",
            source_origin="remote",
            name=name,
            kind="card",
            repo_url="https://github.com/example/card-packs",
            repo_ref="main",
            repo_path=f"packs/{name}",
            source_url="https://raw.githubusercontent.com/example/card-packs/main/marketplace.json",
            installed_commit="abcdef1234567890",
            installed_path_oid="def456",
            installed_revision="abcdef1234567890",
            installed_at="2026-02-15T00:00:00Z",
            content_fingerprint="sha256:deadbeef",
            installed_files=tuple(),
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


def test_get_completions_for_cards_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/cards ", cursor_position=len("/cards "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "list" in names
    assert "add" in names
    assert "remove" in names
    assert "update" in names
    assert "publish" in names
    assert "registry" in names


def test_get_completions_for_models_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/models ", cursor_position=len("/models "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "doctor" in names
    assert "aliases" in names
    assert "catalog" in names

    doc = Document("/models aliases ", cursor_position=len("/models aliases "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "list" in names
    assert "set" in names
    assert "unset" in names


def test_get_completions_for_models_catalog_provider_and_flag() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/models catalog a", cursor_position=len("/models catalog a"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "anthropic" in names

    doc = Document("/models catalog anthropic --", cursor_position=len("/models catalog anthropic --"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "--all" in names


def test_get_completions_for_models_aliases_flags_and_target_values() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document(
        "/models aliases set $system.fast claude-haiku-4-5 --",
        cursor_position=len("/models aliases set $system.fast claude-haiku-4-5 --"),
    )
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "--dry-run" in names
    assert "--target" in names

    doc = Document(
        "/models aliases unset $system.fast --target ",
        cursor_position=len("/models aliases unset $system.fast --target "),
    )
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "env" in names
    assert "project" in names


def test_get_completions_for_mcp_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp ", cursor_position=len("/mcp "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "list" in names
    assert "connect" in names
    assert "disconnect" in names
    assert "session" in names


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


def test_get_completions_for_mcp_session_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp session ", cursor_position=len("/mcp session "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "list" in names
    assert "jar" in names
    assert "new" in names
    assert "use" in names
    assert "clear" in names


def test_get_completions_for_mcp_session_use_cookie_ids() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_McpSessionAgentStub())),
    )

    doc = Document("/mcp session use demo ", cursor_position=len("/mcp session use demo "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "sess-123" in names
    assert "sess-456" in names


def test_get_completions_for_mcp_session_use_shows_connected_session_shortcuts() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_McpSessionAgentStub())),
    )

    doc = Document("/mcp session use ", cursor_position=len("/mcp session use "))
    completions = list(completer.get_completions(doc, None))

    completion_texts = [completion.text for completion in completions]
    assert "demo sess-123" in completion_texts
    assert "demo sess-456" in completion_texts

    display_values = [completion.display_text for completion in completions]
    assert any(display.startswith("1-sess-") for display in display_values)

    display_meta_values = [completion.display_meta_text for completion in completions]
    assert any("demo-server" in value for value in display_meta_values)
    assert any("Current" in value for value in display_meta_values)


def test_get_completions_for_mcp_session_use_cookie_ids_partial() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_McpSessionAgentStub())),
    )

    doc = Document(
        "/mcp session use demo sess-4",
        cursor_position=len("/mcp session use demo sess-4"),
    )
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert names == ["sess-456"]


def test_get_completions_for_mcp_session_jar_suppresses_single_server_noise() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_McpSessionAgentStub())),
    )

    doc = Document("/mcp session jar ", cursor_position=len("/mcp session jar "))
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


def test_get_completions_for_cards_remove() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        env_root = Path(tmpdir) / ".fast-agent"
        card_pack_root = env_root / "card-packs"
        _write_card_pack(card_pack_root, "alpha")
        _write_card_pack(card_pack_root, "beta")

        old_settings = get_settings()
        override = old_settings.model_copy(
            update={
                "environment_dir": str(env_root),
                "cards": CardsSettings(),
            }
        )
        update_global_settings(override)
        try:
            completer = AgentCompleter(agents=["agent1"])
            doc = Document("/cards remove ", cursor_position=len("/cards remove "))
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "alpha" in names
            assert "beta" in names
        finally:
            update_global_settings(old_settings)


def test_get_completions_for_cards_registry() -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(
        update={
            "cards": CardsSettings(
                marketplace_urls=[
                    "https://example.com/cards-one.json",
                    "https://example.com/cards-two.json",
                ]
            )
        }
    )
    update_global_settings(override)
    try:
        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/cards registry ", cursor_position=len("/cards registry "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "1" in names
        assert "2" in names
    finally:
        update_global_settings(old_settings)


def test_get_completions_for_cards_update_only_managed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        env_root = Path(tmpdir) / ".fast-agent"
        card_pack_root = env_root / "card-packs"
        _write_card_pack(card_pack_root, "alpha")
        _write_card_pack(card_pack_root, "beta")
        _write_card_pack(card_pack_root, "gamma")
        _mark_card_pack_managed(card_pack_root, "beta")
        _mark_card_pack_managed(card_pack_root, "gamma")

        old_settings = get_settings()
        override = old_settings.model_copy(update={"environment_dir": str(env_root)})
        update_global_settings(override)
        try:
            completer = AgentCompleter(agents=["agent1"])
            doc = Document("/cards update ", cursor_position=len("/cards update "))
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


def test_get_completions_for_cards_publish_flags() -> None:
    completer = AgentCompleter(agents=["agent1"])
    doc = Document("/cards publish --", cursor_position=len("/cards publish --"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "--no-push" in names
    assert "--message" in names
    assert "--temp-dir" in names
    assert "--keep-temp" in names


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


def test_resource_mention_server_completion() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionAgentStub())),
    )

    doc = Document("^de", cursor_position=3)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "demo:" in names


def test_resource_mention_server_completion_filters_connected_resource_servers() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionFilteredAgentStub())),
    )

    doc = Document("^", cursor_position=1)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "demo:" in names
    assert "offline:" not in names
    assert "nores:" not in names


def test_resource_mention_resource_and_template_completion() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionAgentStub())),
    )

    doc = Document("^demo:repo://items/", cursor_position=len("^demo:repo://items/"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "repo://items/123" in names
    assert "repo://items/{id}{" in names


def test_resource_mention_argument_value_completion() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionAgentStub())),
    )

    doc = Document("^demo:repo://items/{id}{id=7", cursor_position=len("^demo:repo://items/{id}{id=7"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "789" in names


def test_resource_mention_template_uri_with_balanced_placeholders_still_completes() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionAgentStub())),
    )

    text = "^demo:repo://items/{resourceId}"
    doc = Document(text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "repo://items/{resourceId}{" in names


def test_resource_mention_argument_name_completion_supports_camel_case_placeholders() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionAgentStub())),
    )

    text = "^demo:repo://items/{resourceId}{r"
    doc = Document(text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "resourceId=" in names


def test_resource_mention_argument_name_completion_supports_rfc6570_path_expressions() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionAgentStub())),
    )

    text = "^demo:repo://{owner}/{repo}/contents{/path*}{p"
    doc = Document(text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "path=" in names


def test_resource_mention_argument_name_completion_for_later_segments() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionAgentStub())),
    )

    text = "^demo:repo://items/{owner}/{repo}{owner=octo,r"
    doc = Document(text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "repo=" in names
    assert "123" not in names


def test_resource_mention_argument_value_completion_receives_context_args() -> None:
    mention_agent = _MentionAgentStub()
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(mention_agent)),
    )

    text = "^demo:repo://items/{owner}/{repo}{owner=octo,repo=7"
    doc = Document(text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "789" in names
    assert mention_agent.aggregator.last_completion_request is not None
    assert mention_agent.aggregator.last_completion_request["argument_name"] == "repo"
    assert mention_agent.aggregator.last_completion_request["context_args"] == {"owner": "octo"}


def test_resource_mention_malformed_context_falls_back() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("^:broken", cursor_position=len("^:broken"))
    completions = list(completer.get_completions(doc, None))

    assert completions == []


async def _async_identity(value):
    await asyncio.sleep(0)
    return value


@pytest.mark.asyncio
async def test_run_async_completion_uses_owner_loop_from_worker_thread() -> None:
    completer = AgentCompleter(agents=["agent1"])

    result = await asyncio.to_thread(
        lambda: completer._run_async_completion(_async_identity("ok"))
    )

    assert result == "ok"
