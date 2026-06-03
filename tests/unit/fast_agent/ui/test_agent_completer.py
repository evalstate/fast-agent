"""Tests for AgentCompleter sub-completion functionality."""

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from mcp.types import Completion as MCPCompletion
from mcp.types import ResourceTemplate, TextContent
from prompt_toolkit.completion import CompleteEvent, Completion
from prompt_toolkit.document import Document

import fast_agent.config as config_module
from fast_agent.cards.manager import InstalledCardPackSource, write_installed_card_pack_source
from fast_agent.commands.command_catalog import (
    command_action_names,
    command_action_tokens,
    get_command_action_spec,
    get_command_spec,
)
from fast_agent.commands.mcp_command_intents import (
    MCP_SESSION_ACTION_DESCRIPTIONS,
    MCP_SESSION_CLEAR_ACTION,
    MCP_SESSION_SERVER_SCOPED_ACTIONS,
    MCP_SESSION_USE_ACTIONS,
    MCP_TOP_LEVEL_ACTION_DESCRIPTIONS,
)
from fast_agent.commands.shared_command_intents import MODEL_MANAGER_COMMAND_ACTIONS
from fast_agent.config import (
    CardsSettings,
    MCPServerSettings,
    MCPSettings,
    PluginsSettings,
    Settings,
    SkillsSettings,
    get_settings,
    update_global_settings,
)
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
from fast_agent.llm.text_verbosity import TextVerbositySpec
from fast_agent.mcp.experimental_session_client import ServerCookiesView
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.plugins.models import InstalledPluginSource
from fast_agent.plugins.provenance import write_installed_plugin_source
from fast_agent.session import get_session_manager, reset_session_manager
from fast_agent.skills.models import (
    DEFAULT_SKILL_REGISTRIES,
    InstalledSkillSource,
)
from fast_agent.skills.provenance import (
    write_installed_skill_source,
)
from fast_agent.ui.enhanced_prompt import AgentCompleter
from fast_agent.ui.prompt import completion_sources

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp

_VALID_SHA256_FINGERPRINT = "sha256:" + ("0" * 64)


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


class _ReasoningLlmStub:
    reasoning_effort_spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high", "xhigh", "max"],
        allow_auto=True,
        allow_toggle_disable=True,
        default=ReasoningEffortSetting(kind="effort", value="auto"),
    )


class _MissingModelCompletionAttrsLlm:
    service_tier_supported = False
    web_search_supported = False
    web_fetch_supported = False


class _VerbosityLlmStub:
    reasoning_effort_spec = None
    task_budget_supported = False
    text_verbosity_spec = TextVerbositySpec(allowed=("low", "medium", "high"), default="medium")
    service_tier_supported = False
    web_search_supported = False
    web_fetch_supported = False


class _McpSessionClientStub:
    async def list_server_cookies(self, server_identifier: str | None):
        if server_identifier not in {"demo", "demo-server"}:
            return ServerCookiesView(
                server_name="other",
                server_identity="other-server",
                target=None,
                sessions_supported=None,
                active_session_id=None,
                cookies=[],
            )
        return ServerCookiesView(
            server_name="demo",
            server_identity="demo-server",
            target=None,
            sessions_supported=True,
            active_session_id="sess-123",
            cookies=[
                {"id": "sess-123", "title": "Current", "active": True},
                {"id": "sess-456", "title": "Older", "active": False},
            ],
        )


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


class _HistoryAgentStub:
    def __init__(self, turn_count: int) -> None:
        self.message_history = [
            message
            for turn_index in range(1, turn_count + 1)
            for message in (
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text=f"user turn {turn_index}")],
                ),
                PromptMessageExtended(
                    role="assistant",
                    content=[TextContent(type="text", text=f"assistant turn {turn_index}")],
                ),
            )
        ]


def test_model_reasoning_values_prefer_adaptive_label_over_auto() -> None:
    agent = SimpleNamespace(llm=_ReasoningLlmStub())
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(agent)),
    )

    values = completer._resolve_reasoning_values()

    assert "adaptive" in values
    assert "auto" not in values
    assert values.count("adaptive") == 1


def test_model_completion_values_tolerate_missing_optional_attrs() -> None:
    agent = SimpleNamespace(llm=_MissingModelCompletionAttrsLlm())
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(agent)),
    )

    assert completer._resolve_reasoning_values() == []
    assert completer._resolve_task_budget_values() == []
    assert completer._resolve_verbosity_values() == []


def test_model_verbosity_completion_values_use_capability_resolver() -> None:
    agent = SimpleNamespace(llm=_VerbosityLlmStub())
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(agent)),
    )

    assert completer._resolve_verbosity_values() == ["low", "medium", "high"]


def test_complete_history_files_finds_json_and_md():
    """Test that _complete_history_files finds .json and .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "history.json").touch()
        (Path(tmpdir) / "history_upper.JSON").touch()
        (Path(tmpdir) / "notes.md").touch()
        (Path(tmpdir) / "notes_upper.MD").touch()
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
            assert "history_upper.JSON" in names
            assert "notes.md" in names
            assert "notes_upper.MD" in names
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


def test_configured_mcp_server_target_uses_provider_base_url_only_for_provider_servers() -> None:
    completer = AgentCompleter(agents=["agent1"])

    provider_target = completer._configured_mcp_server_target(
        {
            "management": "provider",
            "url": "https://example.com/api/mcp",
        }
    )
    client_target = completer._configured_mcp_server_target(
        {
            "management": "client",
            "url": "https://example.com/api/mcp",
        }
    )

    assert provider_target == "https://example.com/api"
    assert client_target == "https://example.com/api/mcp"


def test_configured_mcp_server_target_formats_typed_stdio_settings() -> None:
    completer = AgentCompleter(agents=["agent1"])

    target = completer._configured_mcp_server_target(
        MCPServerSettings(
            name="docs",
            transport="stdio",
            command="demo-server",
            args=["--root", "My Folder"],
        )
    )

    assert target == "demo-server --root 'My Folder'"


def test_configured_mcp_server_target_formats_dict_stdio_settings() -> None:
    completer = AgentCompleter(agents=["agent1"])

    target = completer._configured_mcp_server_target(
        {"command": "demo-server", "args": ["--root", "My Folder"]}
    )

    assert target == "demo-server --root 'My Folder'"


def test_configured_mcp_server_target_ignores_unshaped_object() -> None:
    completer = AgentCompleter(agents=["agent1"])

    assert completer._configured_mcp_server_target(object()) is None


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


def test_get_completions_for_prompt_load_command_uses_prompt_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "prompt.txt").touch()
        (Path(tmpdir) / "prompt.json").touch()
        (Path(tmpdir) / "prompt_upper.JSON").touch()
        (Path(tmpdir) / "script.py").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            doc = Document("/prompt load ", cursor_position=len("/prompt load "))
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]
            metadata = {c.text: str(c.display_meta) for c in completions}

            assert "prompt.txt" in names
            assert "prompt.json" in names
            assert "prompt_upper.JSON" in names
            assert "script.py" in names
            assert "Prompt template" in metadata["prompt.txt"]
            assert "JSON prompt" in metadata["prompt.json"]
            assert "JSON prompt" in metadata["prompt_upper.JSON"]
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_prompt_command_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/prompt lo", cursor_position=len("/prompt lo"))
    completions = list(completer.get_completions(doc, None))

    assert [completion.text for completion in completions] == ["load"]


def test_get_completions_for_agent_command_flags() -> None:
    completer = AgentCompleter(agents=["agent1", "reviewer"])

    doc = Document("/agent reviewer --", cursor_position=len("/agent reviewer --"))
    completions = list(completer.get_completions(doc, None))
    names = [completion.text for completion in completions]

    assert "--tool" in names
    assert "--dump" in names


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


def test_get_completions_for_shell_path_prefix_quotes_spaces() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "two words.sh").touch()

        completer = AgentCompleter(agents=["agent1"])
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            doc = Document("!./two", cursor_position=len("!./two"))
            event = CompleteEvent(completion_requested=True)
            completions = list(completer.get_completions(doc, event))
        finally:
            os.chdir(original_cwd)

    names = [completion.text for completion in completions]
    assert "'./two words.sh'" in names


def test_get_completions_for_shell_path_uses_current_unquoted_token() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "two words.sh").touch()

        completer = AgentCompleter(agents=["agent1"])
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            doc = Document('!echo "hello world" ./two', cursor_position=len('!echo "hello world" ./two'))
            event = CompleteEvent(completion_requested=True)
            completions = list(completer.get_completions(doc, event))
        finally:
            os.chdir(original_cwd)

    names = [completion.text for completion in completions]
    assert "'./two words.sh'" in names


def test_get_completions_for_history_subcommands():
    """Test get_completions suggests /history subcommands."""
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/history ", cursor_position=9)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "list" in names
    assert "detail" in names
    assert "review" in names
    assert "show" in names
    assert "save" in names
    assert "load" in names
    assert "webclear" not in names


def test_get_completions_for_history_detail_turns_not_limited_to_summary_window() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_HistoryAgentStub(turn_count=14))),
    )

    doc = Document("/history detail ", cursor_position=len("/history detail "))
    completions = list(completer.get_completions(doc, None))
    names = [completion.text for completion in completions]

    assert len(names) == 14
    assert "14" in names
    assert "1" in names
    turn_14 = next(completion for completion in completions if completion.text == "14")
    assert turn_14.display_meta_text == "user turn 14"


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


def test_get_completions_for_history_subcommands_ignores_missing_web_tool_attrs() -> None:
    class _LlmStub:
        pass

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

    assert "webclear" not in names


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
        service_tier_supported = True
        available_service_tiers = ("fast", "flex")
        task_budget_supported = True
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
    assert "task_budget" in names
    assert "fast" in names
    assert "web_search" in names
    assert "web_fetch" not in names


def test_get_completions_for_model_subcommands_match_case_insensitively() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = TextVerbositySpec(allowed=("low", "medium", "high"), default="medium")
        service_tier_supported = False
        available_service_tiers = ()
        task_budget_supported = False
        web_search_supported = False
        web_fetch_supported = False

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model Verb", cursor_position=len("/model Verb"))
    completions = list(completer.get_completions(doc, None))

    assert [completion.text for completion in completions] == ["verbosity"]


def test_get_completions_for_model_subcommands_includes_web_fetch_when_supported() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        service_tier_supported = False
        available_service_tiers = ()
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


def test_get_completions_for_model_supported_value_settings_are_visible() -> None:
    class _LlmStub:
        reasoning_effort_spec = ReasoningEffortSpec(
            kind="effort",
            allowed_efforts=["low", "medium", "high"],
            allow_auto=True,
            allow_toggle_disable=True,
            default=ReasoningEffortSetting(kind="effort", value="auto"),
        )
        text_verbosity_spec = TextVerbositySpec(
            allowed=("low", "medium", "high"),
            default="medium",
        )
        task_budget_supported = True
        service_tier_supported = True
        available_service_tiers = ("fast", "flex")
        web_search_supported = True
        x_search_supported = True
        web_fetch_supported = True

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    subcommands = {
        completion.text
        for completion in completer.get_completions(
            Document("/model ", cursor_position=len("/model ")),
            None,
        )
    }
    value_settings = {
        "reasoning",
        "verbosity",
        "task_budget",
        "fast",
        "web_search",
        "x_search",
        "web_fetch",
    }

    assert value_settings <= subcommands
    for setting in value_settings:
        doc_text = f"/model {setting} "
        value_completions = list(
            completer.get_completions(
                Document(doc_text, cursor_position=len(doc_text)),
                None,
            )
        )
        assert value_completions, setting


def test_get_completions_for_model_fast_values() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        service_tier_supported = True
        available_service_tiers = ("fast", "flex")
        web_search_supported = False
        web_fetch_supported = False

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model fast ", cursor_position=len("/model fast "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "on" in names
    assert "off" in names
    assert "flex" in names
    assert "status" in names


def test_get_completions_for_model_task_budget_values() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        service_tier_supported = False
        available_service_tiers = ()
        task_budget_supported = True
        web_search_supported = False
        web_fetch_supported = False

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model task_budget ", cursor_position=len("/model task_budget "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert names == ["off", "20k", "64k", "128k", "256k"]


def test_get_completions_for_model_values_match_case_insensitively() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = TextVerbositySpec(allowed=("low", "medium", "high"), default="medium")
        service_tier_supported = False
        available_service_tiers = ()
        task_budget_supported = False
        web_search_supported = False
        web_fetch_supported = False

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model verbosity L", cursor_position=len("/model verbosity L"))
    completions = list(completer.get_completions(doc, None))

    assert [completion.text for completion in completions] == ["low"]


def test_get_completions_for_model_fast_values_codexresponses_omit_flex() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        service_tier_supported = True
        available_service_tiers = ("fast",)
        web_search_supported = False
        web_fetch_supported = False

    class _AgentStub:
        llm = _LlmStub()

    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_AgentStub())),
    )

    doc = Document("/model fast ", cursor_position=len("/model fast "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert names == ["on", "off", "status"]


def test_get_completions_for_model_web_search_values() -> None:
    class _LlmStub:
        reasoning_effort_spec = None
        text_verbosity_spec = None
        service_tier_supported = False
        available_service_tiers = ()
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
        service_tier_supported = False
        available_service_tiers = ()
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

        doc = Document("/session pin O", cursor_position=len("/session pin O"))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]
        assert names == ["on", "off"]

        doc = Document("/session pin on ", cursor_position=len("/session pin on "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]
        assert session.info.name in names
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def test_session_prefix_completion_handlers_cover_expected_prefixes() -> None:
    assert tuple(prefix for prefix, _handler in completion_sources._SESSION_PREFIX_COMPLETION_HANDLERS) == (
        "/resume ",
        "/session resume ",
        "/session delete ",
        "/session clear ",
        "/session pin ",
        "/session export ",
    )


def test_history_prefix_completion_handlers_cover_argument_prefixes() -> None:
    assert tuple(prefix for prefix, _handler in completion_sources._HISTORY_PREFIX_COMPLETION_HANDLERS) == (
        "/history load ",
        "/history rewind ",
        "/history detail ",
        "/history review ",
        "/history clear ",
        "/history webclear ",
    )


def test_get_completions_for_session_export(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(update={"environment_dir": str(env_dir)})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        session = manager.create_session()

        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/session export ", cursor_position=len("/session export "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "latest" in names
        assert session.info.name in names
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def test_get_completions_for_session_export_options() -> None:
    completer = AgentCompleter(agents=["agent1"])
    doc = Document(
        "/session export latest --",
        cursor_position=len("/session export latest --"),
    )
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "--output" in names
    assert "--privacy-filter-device" in names
    assert "--privacy-filter-variant" in names
    assert "--privacy-filter-quant" in names

    alias_doc = Document(
        "/session export latest -",
        cursor_position=len("/session export latest -"),
    )
    alias_names = [c.text for c in completer.get_completions(alias_doc, None)]
    assert "-o" in alias_names


def test_get_completions_for_session_export_privacy_filter_values() -> None:
    completer = AgentCompleter(agents=["agent1"])

    device_doc = Document(
        "/session export latest --privacy-filter-device c",
        cursor_position=len("/session export latest --privacy-filter-device c"),
    )
    device_names = [c.text for c in completer.get_completions(device_doc, None)]

    variant_doc = Document(
        "/session export latest --privacy-filter-variant q",
        cursor_position=len("/session export latest --privacy-filter-variant q"),
    )
    variant_names = [c.text for c in completer.get_completions(variant_doc, None)]

    assert device_names == ["cpu", "cuda"]
    assert variant_names == ["q4", "q4f16", "q8"]


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


def test_session_completion_uses_shared_title_extraction(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(update={"environment_dir": str(env_dir)})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        session = manager.create_session(
            metadata={
                "title": {"text": "structured"},
                "label": ["not displayable"],
                "first_user_preview": " Actual prompt ",
            }
        )

        completer = AgentCompleter(agents=["agent1"])
        completions = list(completer._complete_session_ids(""))

        completion = next(item for item in completions if item.text == session.info.name)
        assert "Actual prompt" in completion.display_meta_text
        assert "structured" not in completion.display_meta_text
        assert "not displayable" not in completion.display_meta_text
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
            content_fingerprint=_VALID_SHA256_FINGERPRINT,
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
            content_fingerprint=_VALID_SHA256_FINGERPRINT,
            installed_files=(),
        ),
    )


def _write_plugin(plugin_root: Path, name: str) -> None:
    plugin_dir = plugin_root / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        "schema_version: 1\n"
        f"name: {name}\n"
        "description: Test plugin\n"
        "commands: {}\n",
        encoding="utf-8",
    )


def _mark_plugin_managed(plugin_root: Path, name: str) -> None:
    plugin_dir = plugin_root / name
    write_installed_plugin_source(
        plugin_dir,
        InstalledPluginSource(
            schema_version=1,
            installed_via="marketplace",
            source_origin="remote",
            repo_url="https://github.com/example/plugins",
            repo_ref="main",
            repo_path=f"plugins/{name}",
            source_url="https://raw.githubusercontent.com/example/plugins/main/marketplace.json",
            installed_commit="abcdef1234567890",
            installed_path_oid="def456",
            installed_revision="abcdef1234567890",
            installed_at="2026-02-15T00:00:00Z",
            content_fingerprint=_VALID_SHA256_FINGERPRINT,
        ),
    )


def test_get_completions_for_skills_subcommands():
    """Test get_completions suggests /skills subcommands."""
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/skills ", cursor_position=8)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert names == list(command_action_names("skills"))
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

    assert names == list(command_action_names("cards"))
    assert "list" in names
    assert "add" in names
    assert "remove" in names
    assert "update" in names
    assert "publish" in names
    assert "registry" in names


def _assert_completion_action_tokens(
    command_name: str,
    action_name: str,
    action_tokens: frozenset[str],
) -> None:
    assert action_tokens == frozenset(command_action_tokens(command_name, action_name))


def test_marketplace_completion_action_sets_use_catalog_tokens() -> None:
    _assert_completion_action_tokens("skills", "add", completion_sources._SKILLS_ADD_ACTIONS)
    _assert_completion_action_tokens("skills", "remove", completion_sources._SKILLS_REMOVE_ACTIONS)
    _assert_completion_action_tokens("skills", "update", completion_sources._SKILLS_UPDATE_ACTIONS)
    _assert_completion_action_tokens(
        "skills",
        "registry",
        completion_sources._SKILLS_REGISTRY_ACTIONS,
    )
    _assert_completion_action_tokens("skills", "search", completion_sources._SKILLS_SEARCH_ACTIONS)

    _assert_completion_action_tokens("cards", "add", completion_sources._CARDS_ADD_ACTIONS)
    _assert_completion_action_tokens("cards", "remove", completion_sources._CARDS_REMOVE_ACTIONS)
    _assert_completion_action_tokens("cards", "update", completion_sources._CARDS_UPDATE_ACTIONS)
    _assert_completion_action_tokens("cards", "registry", completion_sources._CARDS_REGISTRY_ACTIONS)
    _assert_completion_action_tokens("cards", "readme", completion_sources._CARDS_README_ACTIONS)
    _assert_completion_action_tokens("cards", "publish", completion_sources._CARDS_PUBLISH_ACTIONS)

    _assert_completion_action_tokens("plugins", "add", completion_sources._PLUGINS_ADD_ACTIONS)
    _assert_completion_action_tokens(
        "plugins",
        "remove",
        completion_sources._PLUGINS_REMOVE_ACTIONS,
    )
    _assert_completion_action_tokens(
        "plugins",
        "update",
        completion_sources._PLUGINS_UPDATE_ACTIONS,
    )
    _assert_completion_action_tokens(
        "plugins",
        "registry",
        completion_sources._PLUGINS_REGISTRY_ACTIONS,
    )

    assert "publish" in completion_sources._CARDS_PUBLISH_ACTIONS


def _completion_dispatch_tokens(
    dispatch: completion_sources.MarketplaceCompletionDispatch,
) -> frozenset[str]:
    return frozenset(token for action_tokens, _handler in dispatch for token in action_tokens)


def test_marketplace_completion_dispatch_tables_cover_argument_actions() -> None:
    assert completion_sources._SKILLS_COMPLETION_DISPATCH
    assert _completion_dispatch_tokens(
        completion_sources._SKILLS_COMPLETION_DISPATCH
    ) == frozenset(
        token
        for action in ("add", "search", "remove", "update", "registry")
        for token in command_action_tokens("skills", action)
    )

    assert completion_sources._CARDS_COMPLETION_DISPATCH
    assert _completion_dispatch_tokens(completion_sources._CARDS_COMPLETION_DISPATCH) == frozenset(
        token
        for action in ("add", "remove", "readme", "update", "registry", "publish")
        for token in command_action_tokens("cards", action)
    )

    assert completion_sources._PLUGINS_COMPLETION_DISPATCH
    assert _completion_dispatch_tokens(
        completion_sources._PLUGINS_COMPLETION_DISPATCH
    ) == frozenset(
        token
        for action in ("add", "remove", "update", "registry")
        for token in command_action_tokens("plugins", action)
    )


def test_get_completions_for_plugins_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/plugins ", cursor_position=len("/plugins "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert names == list(command_action_names("plugins"))
    assert "list" in names
    assert "available" in names
    assert "add" in names
    assert "remove" in names
    assert "update" in names
    assert "registry" in names


@pytest.mark.parametrize(
    ("command_text", "expected_alias"),
    [
        ("/skills m", "marketplace"),
        ("/cards sho", "show"),
        ("/plugins b", "browse"),
    ],
)
def test_catalogued_command_aliases_complete_from_prefix(
    command_text: str,
    expected_alias: str,
) -> None:
    completer = AgentCompleter(agents=["agent1"])

    completions = list(
        completer.get_completions(
            Document(command_text, cursor_position=len(command_text)),
            None,
        )
    )
    metadata = {completion.text: completion.display_meta_text for completion in completions}

    assert expected_alias in metadata
    assert metadata[expected_alias].startswith("alias for ")


@pytest.mark.parametrize(
    ("command_text", "expected_options"),
    [
        ("/skills add --", {"--registry", "--skills-dir"}),
        ("/cards add --", {"--registry", "--force"}),
        ("/plugins add --", {"--registry"}),
    ],
)
def test_marketplace_add_completions_use_catalogued_options(
    command_text: str,
    expected_options: set[str],
) -> None:
    completer = AgentCompleter(agents=["agent1"])

    completions = list(
        completer.get_completions(
            Document(command_text, cursor_position=len(command_text)),
            None,
        )
    )
    names = {completion.text for completion in completions}

    assert expected_options <= names


@pytest.mark.parametrize(
    ("command_name", "action_name"),
    [
        ("skills", "update"),
        ("cards", "update"),
        ("plugins", "update"),
        ("cards", "publish"),
    ],
)
def test_managed_command_completions_use_catalogued_options(
    command_name: str,
    action_name: str,
) -> None:
    action_spec = get_command_action_spec(command_name, action_name)
    assert action_spec is not None
    expected_long_options = {
        option_name
        for option in action_spec.options
        for option_name in (option.name, *option.aliases)
        if option_name.startswith("--")
    }
    expected_short_options = {
        option_name
        for option in action_spec.options
        for option_name in option.aliases
        if option_name.startswith("-") and not option_name.startswith("--")
    }
    completer = AgentCompleter(agents=["agent1"])

    command_text = f"/{command_name} {action_name} --"
    completions = list(
        completer.get_completions(
            Document(command_text, cursor_position=len(command_text)),
            None,
        )
    )
    names = {completion.text for completion in completions}

    assert expected_long_options <= names

    if expected_short_options:
        command_text = f"/{command_name} {action_name} -"
        completions = list(
            completer.get_completions(
                Document(command_text, cursor_position=len(command_text)),
                None,
            )
        )
        names = {completion.text for completion in completions}

        assert expected_short_options <= names


def test_command_completion_descriptions_avoid_parenthetical_plurals() -> None:
    completer = AgentCompleter(agents=["agent1"])

    assert all("(s)" not in description for description in completer.commands.values())


def test_catalogued_command_completion_descriptions_use_catalog_actions() -> None:
    completer = AgentCompleter(agents=["agent1"])

    for command_name in ("model", "models", "check", "skills", "cards", "plugins"):
        spec = get_command_spec(command_name)
        assert spec is not None
        expected_examples: list[str] = [f"/{command_name}"]
        expected_examples.extend(spec.examples)
        if not spec.examples:
            expected_examples.extend(
                f"/{command_name} {action}"
                for action in command_action_names(command_name)
            )
        assert completer.commands[command_name] == (
            f"{spec.summary} ({', '.join(expected_examples)})"
        )


def test_top_level_completion_includes_catalogued_models_and_check() -> None:
    completer = AgentCompleter(agents=["agent1"])

    completions = list(
        completer.get_completions(
            Document("/mo", cursor_position=len("/mo")),
            None,
        )
    )
    names = [completion.text for completion in completions]
    assert "model" in names
    assert "models" in names

    completions = list(
        completer.get_completions(
            Document("/ch", cursor_position=len("/ch")),
            None,
        )
    )
    names = [completion.text for completion in completions]
    assert "check" in names

    completions = list(
        completer.get_completions(
            Document("/co", cursor_position=len("/co")),
            None,
        )
    )
    names = [completion.text for completion in completions]
    assert "commands" in names


def test_get_completions_for_model_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/model ", cursor_position=len("/model "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "doctor" in names
    assert "references" in names
    assert "catalog" in names
    assert "switch" in names
    switch_completion = next(completion for completion in completions if completion.text == "switch")
    model_spec = get_command_spec("model")
    assert model_spec is not None
    expected_meta = next(action.help for action in model_spec.actions if action.action == "switch")
    assert switch_completion.display_meta_text == expected_meta

    doc = Document("/model references ", cursor_position=len("/model references "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "list" in names
    assert "set" in names
    assert "unset" in names


def test_model_manager_completion_handlers_cover_argument_actions() -> None:
    expected_actions = {"references", "catalog"}

    assert set(completion_sources._MODEL_MANAGER_COMPLETION_HANDLERS) == expected_actions
    assert expected_actions <= MODEL_MANAGER_COMMAND_ACTIONS


def test_model_command_completion_modes_cover_model_commands() -> None:
    assert completion_sources._MODEL_COMMAND_COMPLETION_MODES == {
        "model": True,
        "models": False,
    }


def test_get_completions_for_models_subcommands_are_manager_only() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/models ", cursor_position=len("/models "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "doctor" in names
    assert "references" in names
    assert "catalog" in names
    assert "fast" not in names
    assert "switch" not in names


@pytest.mark.parametrize("command_name", ["model", "models"])
def test_get_completions_for_model_commands_include_help_aliases(command_name: str) -> None:
    completer = AgentCompleter(agents=["agent1"])

    text = f"/{command_name} -"
    doc = Document(text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    names = [completion.text for completion in completions]

    assert "-h" in names
    assert "--help" in names


def test_get_completions_for_models_catalog_provider_and_flag() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/models catalog a", cursor_position=len("/models catalog a"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "anthropic" in names

    doc = Document(
        "/models catalog anthropic --",
        cursor_position=len("/models catalog anthropic --"),
    )
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "--all" in names


def test_get_completions_for_model_catalog_provider_and_flag() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/model catalog a", cursor_position=len("/model catalog a"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "anthropic" in names

    doc = Document("/model catalog anthropic --", cursor_position=len("/model catalog anthropic --"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "--all" in names


def test_get_completions_for_model_references_flags_and_target_values() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document(
        "/model references set $system.fast claude-haiku-4-5 --",
        cursor_position=len("/model references set $system.fast claude-haiku-4-5 --"),
    )
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "--dry-run" in names
    assert "--target" in names

    doc = Document(
        "/model references unset $system.fast --target ",
        cursor_position=len("/model references unset $system.fast --target "),
    )
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "env" in names
    assert "project" in names

    doc = Document(
        "/model references unset $system.fast --target P",
        cursor_position=len("/model references unset $system.fast --target P"),
    )
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert names == ["project"]


@pytest.mark.parametrize(
    "text",
    [
        "/model references set $system.fast claude-haiku-4-5 --target env --",
        "/model references set $system.fast claude-haiku-4-5 --target=env --",
    ],
)
def test_get_completions_for_model_references_suppresses_duplicate_target_flag(
    text: str,
) -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document(text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "--dry-run" in names
    assert "--target" not in names


def test_get_completions_for_mcp_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp ", cursor_position=len("/mcp "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert names == list(MCP_TOP_LEVEL_ACTION_DESCRIPTIONS)

    uppercase_doc = Document("/MCP ", cursor_position=len("/MCP "))
    uppercase_completions = list(completer.get_completions(uppercase_doc, None))
    uppercase_names = [c.text for c in uppercase_completions]

    assert "connect" in uppercase_names
    assert "session" in uppercase_names


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


def test_get_completions_for_mcp_reconnect_servers() -> None:
    provider = _ProviderStub(_McpAgentStub(["local", "docs"]))
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", provider),
    )

    doc = Document("/mcp reconnect d", cursor_position=len("/mcp reconnect d"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "docs" in names


def test_get_completions_for_mcp_connect_flags() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp connect npx demo-server --re", cursor_position=len("/mcp connect npx demo-server --re"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "--reconnect" in names


def test_get_completions_for_mcp_connect_flags_include_parser_aliases() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp connect npx demo-server ", cursor_position=len("/mcp connect npx demo-server "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "-n" in names


def test_get_completions_for_mcp_connect_hides_flags_before_target() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp connect --re", cursor_position=len("/mcp connect --re"))
    completions = list(completer.get_completions(doc, None))

    assert completions == []


def test_mcp_connect_context_tracks_flag_values_and_targets() -> None:
    context = AgentCompleter._mcp_connect_context("npx demo-server --name ")

    assert context.context == "flag_value"
    assert context.target_count == 2
    assert context.partial == ""


@pytest.mark.parametrize("flag", ["--auth", "--timeout", "--name", "-n"])
def test_mcp_connect_context_waits_for_value_flags(flag: str) -> None:
    context = AgentCompleter._mcp_connect_context(f"npx demo-server {flag} ")

    assert context.context == "flag_value"
    assert context.target_count == 2


@pytest.mark.parametrize("flag", ["--auth=token", "--name=docs", "--timeout=7"])
def test_mcp_connect_context_ignores_inline_value_flags_as_target(flag: str) -> None:
    context = AgentCompleter._mcp_connect_context(f"npx demo-server {flag} --")

    assert context.context == "flag"
    assert context.target_count == 2
    assert context.partial == "--"


def test_get_completions_for_mcp_session_subcommands() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp session ", cursor_position=len("/mcp session "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert names == list(MCP_SESSION_ACTION_DESCRIPTIONS)


def test_mcp_session_completion_handlers_cover_argument_actions() -> None:
    expected_actions = (
        set(MCP_SESSION_SERVER_SCOPED_ACTIONS)
        | set(MCP_SESSION_USE_ACTIONS)
        | {MCP_SESSION_CLEAR_ACTION}
    )

    assert set(completion_sources._MCP_SESSION_COMPLETION_HANDLERS) == expected_actions
    assert expected_actions <= set(MCP_SESSION_ACTION_DESCRIPTIONS)


def test_get_completions_for_mcp_session_resume_subcommand_prefix() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp session res", cursor_position=len("/mcp session res"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert names == ["resume"]


def test_get_completions_for_mcp_session_list_without_space_only_completes_subcommand() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_McpAgentStub(["docs", "local"]))),
    )

    doc = Document("/mcp session list", cursor_position=len("/mcp session list"))
    completions = list(completer.get_completions(doc, None))
    names = [completion.text for completion in completions]

    assert names == ["list"]


@pytest.mark.parametrize("subcommand", ["new", "create"])
def test_get_completions_for_mcp_session_new_aliases_offer_title_option(
    subcommand: str,
) -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_McpAgentStub(["docs"]))),
    )

    text = f"/mcp session {subcommand} "
    doc = Document(text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    names = [completion.text for completion in completions]

    assert "--title" in names


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

    doc = Document("/mcp session use DEM", cursor_position=len("/mcp session use DEM"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "demo sess-123" in names
    assert "demo sess-456" in names


def test_session_cookie_choice_validates_cookie_shape() -> None:
    assert AgentCompleter._session_cookie_choice(object(), "sess-123") is None
    assert AgentCompleter._session_cookie_choice({"title": "Missing id"}, "sess-123") is None
    assert AgentCompleter._session_cookie_choice({"id": ""}, "sess-123") is None
    assert AgentCompleter._session_cookie_choice(
        {"id": "sess-123", "title": "  Current  "},
        "sess-123",
    ) == ("sess-123", "Current", True)


def test_get_completions_for_mcp_session_resume_cookie_ids() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_McpSessionAgentStub())),
    )

    doc = Document("/mcp session resume demo ", cursor_position=len("/mcp session resume demo "))
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
        "/mcp session use demo SESS-4",
        cursor_position=len("/mcp session use demo SESS-4"),
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
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
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
    docs_completion = next((c for c in completions if c.text == "docs"), None)

    assert "docs" in names
    assert "--name" not in names
    assert docs_completion is not None
    assert docs_completion.display_meta_text == "echo"


def test_runtime_mcp_servers_does_not_swallow_registry_target_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Aggregator:
        context = SimpleNamespace(
            server_registry=SimpleNamespace(
                registry={
                    "docs": MCPServerSettings(name="docs", transport="stdio", command="echo"),
                }
            )
        )

        def list_attached_servers(self) -> list[str]:
            return []

        def list_configured_detached_servers(self) -> list[str]:
            return []

    class Agent:
        aggregator = Aggregator()

    def fail_target(_server_config: object) -> str | None:
        raise RuntimeError("target formatting failed")

    monkeypatch.setattr(
        AgentCompleter,
        "_configured_mcp_server_target",
        staticmethod(fail_target),
    )
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(Agent())),
    )

    with pytest.raises(RuntimeError, match="target formatting failed"):
        completer._runtime_mcp_servers()


def test_settings_mcp_servers_does_not_swallow_registry_target_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        mcp=MCPSettings(
            servers={
                "docs": MCPServerSettings(name="docs", transport="stdio", command="echo"),
            }
        )
    )
    monkeypatch.setattr(config_module, "_settings", settings)

    def fail_target(_server_config: object) -> str | None:
        raise RuntimeError("target formatting failed")

    monkeypatch.setattr(
        AgentCompleter,
        "_configured_mcp_server_target",
        staticmethod(fail_target),
    )

    with pytest.raises(RuntimeError, match="target formatting failed"):
        AgentCompleter(agents=["agent1"])._settings_mcp_servers()


def test_get_completions_for_mcp_connect_configured_url_server_shows_url(monkeypatch) -> None:
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    settings = Settings(
        mcp=MCPSettings(
            servers={
                "docs": MCPServerSettings(
                    name="docs",
                    transport="http",
                    url="https://example.test/mcp/docs",
                ),
            }
        )
    )
    monkeypatch.setattr(config_module, "_settings", settings)

    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/mcp connect d", cursor_position=len("/mcp connect d"))
    completions = list(completer.get_completions(doc, None))
    docs_completion = next((c for c in completions if c.text == "docs"), None)

    assert docs_completion is not None
    assert docs_completion.display_meta_text == "https://example.test/mcp/docs/mcp"


def test_get_completions_for_mcp_connect_shows_target_hint_first(monkeypatch) -> None:
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
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
    assert completions[0].display_text == "[url|npx|uvx|stdio]"
    assert completions[0].display_meta_text == "enter url, npx/uvx, or stdio cmd"


def test_get_completions_for_connect_alias_shows_target_hint_and_servers(monkeypatch) -> None:
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    settings = Settings(
        mcp=MCPSettings(
            servers={
                "docs": MCPServerSettings(name="docs", transport="stdio", command="echo"),
            }
        )
    )
    monkeypatch.setattr(config_module, "_settings", settings)

    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/connect d", cursor_position=len("/connect d"))
    completions = list(completer.get_completions(doc, None))

    assert completions
    assert completions[0].display_text == "[url|npx|uvx|stdio]"
    assert completions[0].display_meta_text == "enter url, npx/uvx, or stdio cmd"
    assert any(completion.text == "docs" for completion in completions)


def test_get_completions_for_connect_alias_connect_flags() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/connect npx demo-server --re", cursor_position=len("/connect npx demo-server --re"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "--reconnect" in names


def test_get_completions_for_skills_remove(monkeypatch):
    """Test get_completions suggests local skills for /skills remove."""
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
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
        metadata = {c.text: c.display_meta_text for c in completions}

        assert "alpha" in names
        assert "beta" in names
        assert metadata["alpha"] == "local skill"
        assert metadata["beta"] == "local skill"


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


def test_skills_marketplace_alias_does_not_offer_registry_choices(monkeypatch):
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
    doc = Document("/skills marketplace ", cursor_position=len("/skills marketplace "))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "1" not in names
    assert "2" not in names


def test_get_completions_for_skills_registry_dedupes_equivalent_active_source() -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(
        update={
            "skills": SkillsSettings(
                marketplace_urls=list(DEFAULT_SKILL_REGISTRIES),
                marketplace_url="https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
            )
        }
    )
    update_global_settings(override)
    try:
        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/skills registry ", cursor_position=len("/skills registry "))
        completions = list(completer.get_completions(doc, None))

        names = [completion.text for completion in completions]
        display_meta = [completion.display_meta_text for completion in completions]

        assert names == ["1", "2", "3"]
        assert display_meta == list(DEFAULT_SKILL_REGISTRIES)
    finally:
        update_global_settings(old_settings)


def test_get_completions_for_skills_registry_supports_file_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    marketplace_file = tmp_path / "marketplace.json"
    marketplace_file.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    old_settings = get_settings()
    override = old_settings.model_copy(
        update={
            "skills": SkillsSettings(
                marketplace_urls=["https://example.com/registry-one.json"],
            )
        }
    )
    update_global_settings(override)
    try:
        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/skills registry mar", cursor_position=len("/skills registry mar"))
        completions = list(completer.get_completions(doc, None))

        names = [completion.text for completion in completions]

        assert "marketplace.json" in names
    finally:
        update_global_settings(old_settings)


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
            metadata = {c.text: c.display_meta_text for c in completions}

            assert "alpha" not in names
            assert "beta" in names
            assert "gamma" in names
            assert metadata["beta"] == "managed skill"
            assert metadata["gamma"] == "managed skill"
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
            metadata = {c.text: c.display_meta_text for c in completions}

            assert "alpha" in names
            assert "beta" in names
            assert metadata["alpha"] == "local card pack"
            assert metadata["beta"] == "local card pack"
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


def test_cards_marketplace_alias_offers_registry_choices() -> None:
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
        doc = Document("/cards marketplace ", cursor_position=len("/cards marketplace "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "1" in names
        assert "2" in names
    finally:
        update_global_settings(old_settings)


def test_get_completions_for_cards_registry_dedupes_equivalent_active_source() -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(
        update={
            "cards": CardsSettings(
                marketplace_urls=["https://github.com/fast-agent-ai/card-packs"],
                marketplace_url="https://raw.githubusercontent.com/fast-agent-ai/card-packs/main/marketplace.json",
            )
        }
    )
    update_global_settings(override)
    try:
        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/cards registry ", cursor_position=len("/cards registry "))
        completions = list(completer.get_completions(doc, None))

        names = [completion.text for completion in completions]
        display_meta = [completion.display_meta_text for completion in completions]

        assert names == ["1"]
        assert display_meta == ["https://github.com/fast-agent-ai/card-packs"]
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
            metadata = {c.text: c.display_meta_text for c in completions}

            assert "alpha" not in names
            assert "beta" in names
            assert "gamma" in names
            assert metadata["beta"] == "managed card pack"
            assert metadata["gamma"] == "managed card pack"
            assert "1" not in names
            assert "2" not in names
            assert "3" not in names
        finally:
            update_global_settings(old_settings)


def test_get_completions_for_plugins_remove() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        env_root = Path(tmpdir) / ".fast-agent"
        plugin_root = env_root / "plugins"
        _write_plugin(plugin_root, "alpha")
        _write_plugin(plugin_root, "beta")

        old_settings = get_settings()
        override = old_settings.model_copy(
            update={
                "environment_dir": str(env_root),
                "plugins": PluginsSettings(),
            }
        )
        update_global_settings(override)
        try:
            completer = AgentCompleter(agents=["agent1"])
            doc = Document("/plugins remove ", cursor_position=len("/plugins remove "))
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]
            metadata = {c.text: c.display_meta_text for c in completions}

            assert "alpha" in names
            assert "beta" in names
            assert metadata["alpha"] == "local plugin"
            assert metadata["beta"] == "local plugin"
        finally:
            update_global_settings(old_settings)


def test_get_completions_for_plugins_update_only_managed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        env_root = Path(tmpdir) / ".fast-agent"
        plugin_root = env_root / "plugins"
        _write_plugin(plugin_root, "alpha")
        _write_plugin(plugin_root, "beta")
        _write_plugin(plugin_root, "gamma")
        _mark_plugin_managed(plugin_root, "beta")
        _mark_plugin_managed(plugin_root, "gamma")

        old_settings = get_settings()
        override = old_settings.model_copy(
            update={
                "environment_dir": str(env_root),
                "plugins": PluginsSettings(),
            }
        )
        update_global_settings(override)
        try:
            completer = AgentCompleter(agents=["agent1"])
            doc = Document("/plugins update ", cursor_position=len("/plugins update "))
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]
            metadata = {c.text: c.display_meta_text for c in completions}

            assert "alpha" not in names
            assert "beta" in names
            assert "gamma" in names
            assert metadata["beta"] == "managed plugin"
            assert metadata["gamma"] == "managed plugin"
            assert "1" not in names
            assert "2" not in names
            assert "3" not in names
        finally:
            update_global_settings(old_settings)


def test_get_completions_for_plugins_registry() -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(
        update={
            "plugins": PluginsSettings(
                marketplace_urls=[
                    "https://example.com/plugins-one.json",
                    "https://example.com/plugins-two.json",
                ]
            )
        }
    )
    update_global_settings(override)
    try:
        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/plugins registry ", cursor_position=len("/plugins registry "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "1" in names
        assert "2" in names
    finally:
        update_global_settings(old_settings)


def test_plugins_marketplace_alias_does_not_offer_registry_choices() -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(
        update={
            "plugins": PluginsSettings(
                marketplace_urls=[
                    "https://example.com/plugins-one.json",
                    "https://example.com/plugins-two.json",
                ]
            )
        }
    )
    update_global_settings(override)
    try:
        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/plugins marketplace ", cursor_position=len("/plugins marketplace "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "1" not in names
        assert "2" not in names
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
        (Path(tmpdir) / "agent_upper.MARKDOWN").touch()
        (Path(tmpdir) / "agent.yaml").touch()
        (Path(tmpdir) / "agent_upper.YAML").touch()
        (Path(tmpdir) / "agent.yml").touch()
        (Path(tmpdir) / "agent.txt").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_agent_card_files(""))
            names = [c.text for c in completions]

            assert "agent.md" in names
            assert "agent_upper.MARKDOWN" in names
            assert "agent.yaml" in names
            assert "agent_upper.YAML" in names
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


def test_get_completions_for_card_command_flags() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/card agent.md --", cursor_position=len("/card agent.md --"))
    completions = list(completer.get_completions(doc, None))
    names = [completion.text for completion in completions]

    assert "--tool" in names
    assert "--remove" in names
    assert "--dump" not in names


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


def test_command_completions_normalize_case_without_stripping_buffer() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/HIST", cursor_position=len("/HIST"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "history" in names


def test_agent_completions_still_work():
    """Test that agent completions still work."""
    completer = AgentCompleter(agents=["test_agent", "other_agent"])

    # Simulate typing "@test"
    doc = Document("@TEST", cursor_position=5)
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

    doc = Document("^DE", cursor_position=3)
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
    assert "file:" in names
    assert "url:" in names
    assert "offline:" not in names
    assert "nores:" not in names


def test_resource_mention_builtin_attachment_server_completion_meta() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionFilteredAgentStub())),
    )

    doc = Document("^", cursor_position=1)
    completions = list(completer.get_completions(doc, None))
    meta_by_text = {completion.text: completion.display_meta_text for completion in completions}

    assert meta_by_text["file:"] == "local file attachment"
    assert meta_by_text["url:"] == "remote URL attachment"
    assert meta_by_text["demo:"] == "connected mcp server (resources)"


def test_connected_servers_from_status_keeps_missing_identity() -> None:
    connected = AgentCompleter._connected_servers_from_status(
        {
            "demo": SimpleNamespace(is_connected=True),
            "named": SimpleNamespace(is_connected=True, implementation_name=" Named Server "),
            "offline": SimpleNamespace(is_connected=False, implementation_name="Offline"),
        }
    )

    assert connected == [("demo", None), ("named", "Named Server")]


def test_resource_mention_resource_and_template_completion() -> None:
    completer = AgentCompleter(
        agents=["agent1"],
        current_agent="agent1",
        agent_provider=cast("AgentApp", _ProviderStub(_MentionAgentStub())),
    )

    doc = Document("^demo:REPO://ITEMS/", cursor_position=len("^demo:REPO://ITEMS/"))
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "repo://items/123" in names
    assert "repo://items/{id}{" in names


def test_resource_mention_local_file_completion_encodes_spaces() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / "two words.txt").write_text("hi", encoding="utf-8")

        completer = AgentCompleter(agents=["agent1"])
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            doc = Document("^file:./TWO", cursor_position=len("^file:./TWO"))
            completions = list(completer.get_completions(doc, None))
        finally:
            os.chdir(original_cwd)

    assert any(completion.text == "./two%20words.txt" for completion in completions)


def test_resource_mention_local_file_completion_uses_completer_cwd() -> None:
    with tempfile.TemporaryDirectory() as shell_dir, tempfile.TemporaryDirectory() as process_dir:
        shell_base = Path(shell_dir)
        process_base = Path(process_dir)
        (shell_base / "shell note.txt").write_text("shell", encoding="utf-8")
        (process_base / "process note.txt").write_text("process", encoding="utf-8")

        completer = AgentCompleter(agents=["agent1"], cwd=shell_base)
        original_cwd = os.getcwd()
        try:
            os.chdir(process_base)
            doc = Document("^file:./shell", cursor_position=len("^file:./shell"))
            completions = list(completer.get_completions(doc, None))
        finally:
            os.chdir(original_cwd)

    names = [completion.text for completion in completions]
    assert "./shell%20note.txt" in names
    assert "./process%20note.txt" not in names


def test_resource_mention_url_completion_offers_http_schemes() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("^url:H", cursor_position=len("^url:H"))
    completions = list(completer.get_completions(doc, None))

    names = [completion.text for completion in completions]
    assert "https://" in names
    assert "http://" in names


def test_server_result_list_validates_result_shape() -> None:
    assert AgentCompleter._server_result_list(object(), "demo") == []
    assert AgentCompleter._server_result_list({"demo": "not-list"}, "demo") == []
    assert AgentCompleter._server_result_list({"demo": ["one", 2]}, "demo") == ["one", 2]


def test_attach_command_completion_offers_clear_and_paths() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / "report.pdf").write_bytes(b"%PDF-1.4")
        (base / "two words.pdf").write_bytes(b"%PDF-1.4")

        completer = AgentCompleter(agents=["agent1"])
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            doc = Document("/attach t", cursor_position=len("/attach t"))
            completions = list(completer.get_completions(doc, None))
        finally:
            os.chdir(original_cwd)

    names = [completion.text for completion in completions]
    assert "'two words.pdf'" in names


def test_attach_command_completion_uses_completer_cwd() -> None:
    with tempfile.TemporaryDirectory() as shell_dir, tempfile.TemporaryDirectory() as process_dir:
        shell_base = Path(shell_dir)
        process_base = Path(process_dir)
        (shell_base / "two words.pdf").write_bytes(b"%PDF-1.4")
        (process_base / "temp.pdf").write_bytes(b"%PDF-1.4")

        completer = AgentCompleter(agents=["agent1"], cwd=shell_base)
        original_cwd = os.getcwd()
        try:
            os.chdir(process_base)
            doc = Document("/attach t", cursor_position=len("/attach t"))
            completions = list(completer.get_completions(doc, None))
        finally:
            os.chdir(original_cwd)

    names = [completion.text for completion in completions]
    assert "'two words.pdf'" in names
    assert "temp.pdf" not in names


def test_attach_command_completion_offers_https_hint() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/attach h", cursor_position=len("/attach h"))
    completions = list(completer.get_completions(doc, None))

    names = [completion.text for completion in completions]
    assert "https://" in names


def test_attach_command_completion_offers_clear_hint() -> None:
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/attach C", cursor_position=len("/attach C"))
    completions = list(completer.get_completions(doc, None))

    names = [completion.text for completion in completions]
    assert "clear" in names


def test_attach_command_completion_quotes_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fast_agent.utils.commandline.os.name", "nt")

    completer = AgentCompleter(agents=["agent1"])
    completion = Completion(
        r"C:\Program Files\Tool\tool.exe",
        start_position=0,
        display=r"C:\Program Files\Tool\tool.exe",
        display_meta="path",
    )

    def _complete_shell_paths(partial: str, delete_len: int, max_results: int = 100) -> list[Completion]:
        del partial, delete_len, max_results
        return [completion]

    monkeypatch.setattr(completer, "_complete_shell_paths", _complete_shell_paths)

    doc = Document("/attach C:\\Pro", cursor_position=len("/attach C:\\Pro"))
    completions = list(completer.get_completions(doc, None))

    names = [item.text for item in completions]
    assert '"C:\\Program Files\\Tool\\tool.exe"' in names


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

    text = "^demo:repo://items/{resourceId}{R"
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
        lambda: completer._run_async_completion(lambda: _async_identity("ok"))
    )

    assert result == "ok"


@pytest.mark.asyncio
async def test_run_async_completion_on_owner_loop_does_not_create_coroutine() -> None:
    completer = AgentCompleter(agents=["agent1"])
    calls = 0

    def create_awaitable():
        nonlocal calls
        calls += 1
        return _async_identity("unreachable")

    result = completer._run_async_completion(create_awaitable)

    assert result is None
    assert calls == 0
