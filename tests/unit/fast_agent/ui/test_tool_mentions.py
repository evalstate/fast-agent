from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest
from mcp_types import (
    EmbeddedResource,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
    Tool,
)
from prompt_toolkit.document import Document

from fast_agent.core.agent_app import AgentApp
from fast_agent.ptc.mentions import (
    TOOL_DEFINITIONS_META,
    ToolMentionError,
    find_programmatic_function,
    rewrite_tool_mentions,
    tool_mention_names,
)
from fast_agent.tools.tool_surface import (
    ProgrammaticSurface,
    SurfaceTool,
    ToolExposure,
    ToolSurface,
)
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.message_display_helpers import build_user_message_display
from fast_agent.ui.prompt.attachment_tokens import strip_local_attachment_tokens
from fast_agent.ui.prompt.completer import AgentCompleter
from fast_agent.ui.prompt.resource_mentions import (
    ResourceMentionError,
    build_prompt_with_resources,
    parse_mentions,
    resolve_mentions,
)
from fast_agent.ui.prompt.status_bar.attachment import summarize_draft_attachments

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _ToolAgent:
    def __init__(self, *, programmatic: bool = True) -> None:
        self.programmatic = programmatic
        self.surface_lookups = 0

    async def tool_surface(self) -> ToolSurface:
        self.surface_lookups += 1
        if not self.programmatic:
            return ToolSurface(tools=())
        return ToolSurface(
            programmatic=ProgrammaticSurface(
                tool_name="exec_sandbox",
                vocabulary=2,
                tool_format="json",
                instructions="",
                type_check=False,
                max_duration_seconds=30,
                max_memory_mb=64,
                search_active=True,
            ),
            tools=(
                _function("jobs_status", "jobs-status", deferred=False),
                _function("jobs_list", "jobs-list", deferred=True),
            ),
        )


def _function(python_name: str, raw_name: str, *, deferred: bool) -> SurfaceTool:
    return SurfaceTool(
        tool=Tool(name=raw_name, input_schema={}, description="Inspect jobs.\nMore detail here."),
        exposure=ToolExposure.PROGRAMMATIC,
        context_tokens=0,
        python_name=python_name,
        stub=f"def {python_name}(): ...",
        deferred=deferred,
        stub_tokens=24,
        source="jobs",
    )


@pytest.mark.asyncio
async def test_tool_mentions_attach_one_block_in_mention_order() -> None:
    typed = "Show me ^tool:jobs-status ^tool:jobs_list ^tool:jobs_status ^tool:jobs_list"
    parsed = parse_mentions(typed)
    assert not parsed.warnings
    # This agent only exposes a tool surface, not an MCP resource resolver.
    resolved = await resolve_mentions(_ToolAgent(), parsed)
    prompt = build_prompt_with_resources(typed, resolved)

    text, definitions = prompt.content
    assert isinstance(text, TextContent)
    assert text.text == "Show me"
    assert (text.meta or {}).get("fast_agent_original_text") == typed
    assert isinstance(definitions, TextContent)
    assert (definitions.meta or {}).get("fast_agent_tool_definitions") == [
        "jobs_status",
        "jobs_list",
    ]
    assert definitions.text.count("<fastagent:tool-definitions>") == 1
    assert definitions.text.count("def jobs_status") == 1
    assert definitions.text.index("def jobs_status") < definitions.text.index("def jobs_list")


@pytest.mark.parametrize(
    ("agent", "error"),
    [
        (_ToolAgent(), "no `exec_sandbox` function"),
        (_ToolAgent(programmatic=False), "programmatic tool calling is not enabled"),
        (object(), "programmatic tool calling is not enabled"),
    ],
)
@pytest.mark.asyncio
async def test_tool_mentions_fail_clearly(agent: object, error: str) -> None:
    with pytest.raises(ResourceMentionError, match=rf"\^tool:missing: {error}"):
        await resolve_mentions(agent, parse_mentions("Show me ^tool:missing"))


def test_tool_mentions_appear_as_supported_draft_attachments() -> None:
    summary = summarize_draft_attachments("Show me ^tool:jobs_list", model_name=None)
    assert summary is not None
    assert summary.count == 1
    assert not summary.any_questionable
    assert "application/octet-stream" not in summary.mime_types


def test_unknown_tool_mentions_are_flagged_when_names_are_known() -> None:
    known = frozenset({"jobs_list", "hf__jobs-list"})
    for text, questionable in (
        ("Show me ^tool:jobs_list", False),
        ("Show me ^tool:hf__jobs-list.", False),
        ("Show me ^tool:nonexistent", True),
    ):
        summary = summarize_draft_attachments(text, model_name=None, tool_names=known)
        assert summary is not None
        assert summary.any_questionable is questionable, text


def test_tool_mention_completions_prioritize_deferred_functions() -> None:
    agent = _ToolAgent()
    completer = AgentCompleter(
        agents=["main", "direct"],
        current_agent="main",
        agent_provider=AgentApp(
            {"main": cast("AgentProtocol", agent), "direct": cast("AgentProtocol", object())}
        ),
    )

    candidates = {entry.text: entry for entry in completer.get_completions(Document("^to"), None)}
    assert candidates["tool:"].display_meta_text == "PTC function definition"
    entries = list(completer.get_completions(Document("Show me ^tool:jobs_"), None))
    assert [entry.text for entry in entries] == ["jobs_list", "jobs_status"]
    assert entries[0].start_position == -len("jobs_")
    assert all(
        part in entries[0].display_meta_text for part in ("jobs", "~24 tok", "Inspect jobs.")
    )
    assert agent.surface_lookups == 1

    completer.current_agent = "direct"
    assert not list(completer.get_completions(Document("^to"), None))
    assert not list(completer.get_completions(Document("^tool:jobs"), None))


def test_chat_display_uses_original_text_and_only_definition_chips() -> None:
    definitions = TextContent(
        type="text",
        text="<fastagent:tool-definitions>stub</fastagent:tool-definitions>",
        meta={"fast_agent_tool_definitions": ["jobs_list"]},
    )
    sources = TextContent(
        type="text",
        text="<fastagent:tool-sources>sources</fastagent:tool-sources>",
        meta={"fast_agent_tool_sources_fingerprint": "fingerprint"},
    )
    text = TextContent(
        type="text",
        text="Show me",
        meta={"fast_agent_original_text": "Show me ^tool:jobs_list"},
    )
    prompt = PromptMessageExtended(role="user", content=[sources, text, definitions])
    assert build_user_message_display([prompt]) == ("Show me ^tool:jobs_list", ["tool:jobs_list"])

    text_without_meta = TextContent(type="text", text="Show me ^tool:jobs_list")
    prompt = PromptMessageExtended(role="user", content=[definitions, text_without_meta, sources])
    assert build_user_message_display([prompt]) == ("Show me ^tool:jobs_list", ["tool:jobs_list"])

    # A generated block isn't a fallback when no user text is present.
    prompt = PromptMessageExtended(role="user", content=[definitions, sources])
    assert build_user_message_display([prompt]) == ("", ["tool:jobs_list"])


def test_tools_add_completes_function_names() -> None:
    completer = AgentCompleter(
        agents=["main"],
        current_agent="main",
        agent_provider=AgentApp({"main": cast("AgentProtocol", _ToolAgent())}),
    )

    actions = [entry.text for entry in completer.get_completions(Document("/tools "), None)]
    assert actions[:3] == ["--all", "add", "select"]
    names = [entry.text for entry in completer.get_completions(Document("/tools add "), None)]
    assert names == ["jobs_list", "jobs_status"]
    rest = completer.get_completions(Document("/tools add jobs_list jo"), None)
    assert [entry.text for entry in rest] == ["jobs_status"]


def test_tool_mention_names_stop_at_sentence_punctuation() -> None:
    assert tool_mention_names("Use ^tool:hf_jobs, then ^tool:hf_whoami.") == [
        "hf_jobs",
        "hf_whoami",
    ]
    assert tool_mention_names("(see ^tool:hub__model-search.v2)? or ^tool:a/b") == [
        "hub__model-search.v2"
    ]


async def _send_text(agent: object, typed: str) -> PromptMessageExtended:
    resolved = await resolve_mentions(agent, parse_mentions(typed))
    return build_prompt_with_resources(typed, resolved)


@pytest.mark.asyncio
async def test_inline_mentions_become_names_and_trailing_mentions_are_removed() -> None:
    agent = _ToolAgent()

    prompt = await _send_text(agent, "Use ^tool:jobs-status, then ^tool:jobs_list.")
    assert prompt.first_text() == "Use `jobs_status`, then `jobs_list`."

    prompt = await _send_text(agent, "Find failed jobs ^tool:jobs_list  ^tool:jobs_status \n")
    assert prompt.first_text() == "Find failed jobs"
    definitions = prompt.content[-1]
    assert (definitions.meta or {}).get(TOOL_DEFINITIONS_META) == ["jobs_list", "jobs_status"]


@pytest.mark.asyncio
async def test_mentions_preserve_indentation_and_mix_with_file_attachments(tmp_path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    typed = (
        "Review:\n\n"
        "    def f():\n"
        "        return  1\n"
        "\tthen use ^tool:jobs_list here\n"
        f"^file:{notes} ^tool:jobs_status"
    )

    prompt = await _send_text(_ToolAgent(), typed)

    assert prompt.first_text() == (
        "Review:\n\n    def f():\n        return  1\n\tthen use `jobs_list` here"
    )
    assert any("hello" in str(block) for block in prompt.content[1:])
    assert (prompt.content[1].meta or {}).get(TOOL_DEFINITIONS_META) == [
        "jobs_list",
        "jobs_status",
    ]


@pytest.mark.asyncio
async def test_failed_tool_mention_reports_file_errors_too(tmp_path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(ResourceMentionError) as exc:
        await _send_text(_ToolAgent(), f"Review ^file:{missing} with ^tool:typo")
    assert "^tool:typo" in str(exc.value)
    assert str(missing) in str(exc.value)


class _ResourceServerNamedTool(_ToolAgent):
    def __init__(self) -> None:
        super().__init__(programmatic=False)

    async def get_resource(
        self, resource_uri: str, namespace: str | None = None
    ) -> ReadResourceResult:
        assert namespace == "tool"
        return ReadResourceResult(
            contents=[TextResourceContents(uri=resource_uri, mime_type="text/plain", text="doc")]
        )


@pytest.mark.asyncio
async def test_without_ptc_tool_mentions_reach_an_mcp_server_named_tool() -> None:
    prompt = await _send_text(_ResourceServerNamedTool(), "Read ^tool:docs://guide, please")

    assert prompt.first_text() == "Read please"
    [resource] = prompt.content[1:]
    assert isinstance(resource, EmbeddedResource)
    assert str(resource.resource.uri) == "docs://guide,"


def test_mcp_server_named_tool_keeps_resource_completions_without_ptc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _ToolAgent(programmatic=False)
    completer = AgentCompleter(
        agents=["main"],
        current_agent="main",
        agent_provider=AgentApp({"main": cast("AgentProtocol", agent)}),
    )

    async def resource_servers() -> list[str]:
        return ["tool"]

    async def resource_uris(server_name: str) -> list[str]:
        assert server_name == "tool"
        return ["docs://guide"]

    async def no_templates(server_name: str) -> list[object]:
        del server_name
        return []

    monkeypatch.setattr(completer, "_list_connected_resource_servers", resource_servers)
    monkeypatch.setattr(completer, "_list_server_resource_uris", resource_uris)
    monkeypatch.setattr(completer, "_list_server_resource_templates", no_templates)

    servers = {entry.text: entry for entry in completer.get_completions(Document("^to"), None)}
    assert servers["tool:"].display_meta_text == "connected mcp server (resources)"
    entries = completer.get_completions(Document("^tool:docs"), None)
    assert [entry.text for entry in entries] == ["docs://guide"]


def _retired_surface() -> ToolSurface:
    surface = asyncio.run(_ToolAgent().tool_surface())
    assert surface.programmatic is not None
    return ToolSurface(
        programmatic=replace(surface.programmatic, retired=frozenset({"a_b"})),
        # `a-b` (Python name `a_b`) was removed; a surviving tool is literally named `a_b`.
        tools=(*surface.tools, _function("a_b_2", "a_b", deferred=True)),
    )


def test_retired_python_name_does_not_resolve_to_a_surviving_raw_name() -> None:
    surface = _retired_surface()

    assert find_programmatic_function(surface, "a_b") is None
    with pytest.raises(ToolMentionError, match="no longer available"):
        rewrite_tool_mentions(surface, ["Use ^tool:a_b"])
    assert rewrite_tool_mentions(surface, ["Use ^tool:a_b_2 now"]).texts == ("Use `a_b_2` now",)


def test_clearing_attachments_also_removes_tool_mentions() -> None:
    assert strip_local_attachment_tokens("Find datasets ^tool:hf_datasets_search") == (
        "Find datasets"
    )
    assert strip_local_attachment_tokens("Use ^tool:jobs_list and\n  keep  this") == (
        "Use and\n  keep  this"
    )


def test_draft_flags_tool_tokens_that_cannot_name_a_function() -> None:
    summary = summarize_draft_attachments("Show me ^tool:jobs/list", model_name=None)
    assert summary is not None
    assert summary.any_questionable
