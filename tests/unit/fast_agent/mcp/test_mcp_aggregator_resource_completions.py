from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from mcp_types import CompleteResult, Completion, ResourceTemplate

import fast_agent.mcp.mcp_aggregator as aggregator_module
from fast_agent.context import Context
from fast_agent.event_progress import ProgressAction
from fast_agent.mcp.app_integrations import AppServerConfig
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp.skills_extension import GetSkillResult, ListSkillsResult, SkillEntry

if TYPE_CHECKING:
    from mcp.client import CacheMode


class _BaseAggregator(MCPAggregator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialized = True

    async def validate_server(self, server_name: str) -> bool:
        return server_name in self.server_names


@pytest.mark.asyncio
async def test_list_resource_templates_uses_server_execution() -> None:
    class _TemplatesAggregator(_BaseAggregator):
        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name
            return feature == "resources"

        async def _execute_on_server(
            self,
            server_name: str,
            operation_type: str,
            operation_name: str,
            method_name: str,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            del operation_type, operation_name, method_args, error_factory, progress_callback
            assert method_name == "list_resource_templates"
            return SimpleNamespace(
                resource_templates=[ResourceTemplate(name="repo", uri_template="repo://{id}")]
            )

    aggregator = _TemplatesAggregator(
        server_names=["demo"],
        connection_persistence=False,
        context=Context(),
    )

    result = await aggregator.list_resource_templates("demo")

    assert list(result.keys()) == ["demo"]
    assert result["demo"][0].uri_template == "repo://{id}"


@pytest.mark.asyncio
async def test_complete_resource_argument_returns_empty_when_unsupported() -> None:
    class _UnsupportedAggregator(_BaseAggregator):
        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name
            return feature != "completions"

    aggregator = _UnsupportedAggregator(
        server_names=["demo"],
        connection_persistence=False,
        context=Context(),
    )

    result = await aggregator.complete_resource_argument(
        server_name="demo",
        template_uri="repo://{id}",
        argument_name="id",
        value="1",
    )

    assert result.values == []


@pytest.mark.asyncio
async def test_complete_resource_argument_passes_through_completion_values() -> None:
    class _CompletionAggregator(_BaseAggregator):
        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name
            return feature == "completions"

        async def _execute_on_server(
            self,
            server_name: str,
            operation_type: str,
            operation_name: str,
            method_name: str,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            del (
                server_name,
                operation_type,
                operation_name,
                method_name,
                method_args,
                error_factory,
                progress_callback,
            )
            return CompleteResult(completion=Completion(values=["123", "456"]))

    aggregator = _CompletionAggregator(
        server_names=["demo"],
        connection_persistence=False,
        context=Context(),
    )

    result = await aggregator.complete_resource_argument(
        server_name="demo",
        template_uri="repo://{id}",
        argument_name="id",
        value="",
    )

    assert result.values == ["123", "456"]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, RuntimeError("read failed")])
async def test_failed_resource_read_emits_error_completion(monkeypatch, result) -> None:
    class _ResourceAggregator(_BaseAggregator):
        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name
            return feature == "resources"

        async def _execute_on_server(
            self,
            server_name: str,
            operation_type: str,
            operation_name: str,
            method_name: str,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            del (
                server_name,
                operation_type,
                operation_name,
                method_name,
                method_args,
                error_factory,
                progress_callback,
            )
            if isinstance(result, Exception):
                raise result
            return result

    events: list[dict[str, object]] = []

    class _Logger:
        def info(self, message: str, *, data: dict[str, object]) -> None:
            del message
            events.append(data)

        def error(self, message: str, *, data: dict[str, object]) -> None:
            del message
            events.append(data)

    aggregator = _ResourceAggregator(
        server_names=["demo"],
        connection_persistence=False,
        context=Context(),
    )
    monkeypatch.setattr(aggregator_module, "logger", _Logger())

    with pytest.raises((ValueError, RuntimeError)):
        await aggregator._get_resource_from_server("demo", "file://missing.txt")

    assert [event["progress_action"] for event in events] == [
        ProgressAction.READING_RESOURCE,
        ProgressAction.FATAL_ERROR,
    ]


@pytest.mark.asyncio
async def test_skills_extension_routes_requests_to_the_named_server() -> None:
    calls: list[tuple[str, str, str, dict[str, str] | None]] = []

    class _SkillsAggregator(_BaseAggregator):
        async def _execute_on_server(
            self,
            server_name: str,
            operation_type: str,
            operation_name: str,
            method_name: str,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            del error_factory, progress_callback
            calls.append((server_name, operation_type, method_name, method_args))
            entry = SkillEntry(
                uri="skill://demo/SKILL.md",
                frontmatter={"name": "demo", "description": "Demo skill"},
            )
            if method_name == "list_skills":
                return ListSkillsResult(skills=[entry])
            assert method_name == "get_skill"
            assert operation_name == entry.uri
            return GetSkillResult(skill=entry)

    aggregator = _SkillsAggregator(
        server_names=["demo"],
        connection_persistence=False,
        context=Context(),
    )

    listed = await aggregator.list_skills("demo", cursor="page-1")
    skill = await aggregator.get_skill("skill://demo/SKILL.md", server_name="demo")

    assert listed.skills[0].uri == skill.skill.uri
    assert calls == [
        ("demo", "skills/list", "list_skills", {"cursor": "page-1"}),
        ("demo", "skills/get", "get_skill", {"uri": "skill://demo/SKILL.md"}),
    ]


@pytest.mark.asyncio
async def test_app_resource_scan_progress_uses_compact_label(monkeypatch) -> None:
    class _AppsAggregator(_BaseAggregator):
        async def _list_resources_from_server(
            self,
            server_name: str,
            *,
            check_support: bool = True,
            cache_mode: CacheMode = "use",
        ):
            del server_name, check_support, cache_mode
            return []

    events: list[dict[str, object]] = []

    class _Logger:
        def info(self, message: str, *, data: dict[str, object]) -> None:
            del message
            events.append(data)

    aggregator = _AppsAggregator(
        server_names=["demo"],
        connection_persistence=False,
        context=Context(),
    )
    monkeypatch.setattr(aggregator_module, "logger", _Logger())

    await aggregator._collect_app_resources(
        "demo",
        AppServerConfig(server_name="demo"),
        [],
    )

    assert [event["details"] for event in events] == ["Apps", "Apps"]
