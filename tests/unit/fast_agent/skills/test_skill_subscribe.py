"""Tests for `resources/subscribe` flow on `skill://index.json`.

These exercise the agent-level refresh path triggered by an MCP
`notifications/resources/updated` for the skill index URI: added skills
become readable, removed skills become unreadable, and the model is
notified via the runtime toolbar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import (
    ReadResourceResult,
    ResourceUpdatedNotification,
    ResourceUpdatedNotificationParams,
    ServerNotification,
    TextResourceContents,
)
from pydantic import AnyUrl

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.mcp.mcp_skills_loader import INDEX_URI


def _text_result(text: str, uri: str, mime: str = "application/json") -> ReadResourceResult:
    return ReadResourceResult(
        contents=[TextResourceContents(uri=AnyUrl(uri), mimeType=mime, text=text)]
    )


def _skill_md(name: str, description: str = "desc") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\nbody\n"


def _index(skills: list[dict]) -> str:
    return json.dumps(
        {
            "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
            "skills": skills,
        }
    )


def _entry(name: str) -> dict:
    return {
        "name": name,
        "type": "skill-md",
        "description": f"The {name} skill",
        "url": f"skill://{name}/SKILL.md",
    }


def _make_aggregator_with_index(initial: list[str]) -> Any:
    """Return a mock aggregator whose `get_resource` reflects a mutable
    index. Tests flip `state["names"]` between calls to simulate a server
    publishing an updated catalog.
    """
    agg = MagicMock()
    state = {"names": list(initial)}

    async def get_resource(uri: str, *, server_name: str | None = None) -> ReadResourceResult:
        if uri == INDEX_URI:
            return _text_result(
                _index([_entry(n) for n in state["names"]]), INDEX_URI
            )
        # Strip /SKILL.md to get the name back.
        if uri.startswith("skill://") and uri.endswith("/SKILL.md"):
            name = uri[len("skill://") : -len("/SKILL.md")]
            return _text_result(
                _skill_md(name, description=f"The {name} skill"),
                uri,
                mime="text/markdown",
            )
        raise ValueError(f"unknown resource: {uri}")

    agg.get_resource = get_resource
    agg.subscribe_to_resource = AsyncMock(return_value=True)
    agg.server_names = ["srv"]
    agg.server_notification_callback = None
    return agg, state


def _make_resource_updated(uri: str) -> ServerNotification:
    """Wrap a ResourceUpdatedNotification in a ServerNotification root."""
    inner = ResourceUpdatedNotification(
        params=ResourceUpdatedNotificationParams(uri=AnyUrl(uri))
    )
    # ServerNotification is a discriminated union — wrap accordingly.
    return ServerNotification(root=inner)


@pytest.mark.asyncio
async def test_refresh_adds_new_skill_after_index_update(tmp_path: Path) -> None:
    """A server publishes a new skill: refresh fetches the updated index,
    registers the new manifest, and surfaces a runtime-toolbar notice."""
    agg, state = _make_aggregator_with_index(["alpha"])

    config = AgentConfig(name="test", instruction="x", servers=[], skills=None)
    agent = McpAgent(config=config, context=Context())
    agent._aggregator = agg

    # Initial load: only "alpha".
    await agent._refresh_skills_after_index_update("srv")
    names = sorted(m.name for m in agent._skill_manifests if m.server_name == "srv")
    assert names == ["alpha"]

    # Server adds "beta".
    state["names"] = ["alpha", "beta"]
    await agent._refresh_skills_after_index_update("srv")
    names = sorted(m.name for m in agent._skill_manifests if m.server_name == "srv")
    assert names == ["alpha", "beta"]

    # A runtime-toolbar notice mentions the addition.
    toolbar_text = "\n".join(agent.warnings)
    assert "beta" in toolbar_text and "+" in toolbar_text


@pytest.mark.asyncio
async def test_refresh_removes_dropped_skill_after_index_update(tmp_path: Path) -> None:
    agg, state = _make_aggregator_with_index(["alpha", "beta"])
    config = AgentConfig(name="test", instruction="x", servers=[], skills=None)
    agent = McpAgent(config=config, context=Context())
    agent._aggregator = agg

    await agent._refresh_skills_after_index_update("srv")
    assert {m.name for m in agent._skill_manifests if m.server_name == "srv"} == {
        "alpha",
        "beta",
    }

    # Server drops "beta".
    state["names"] = ["alpha"]
    await agent._refresh_skills_after_index_update("srv")
    assert {m.name for m in agent._skill_manifests if m.server_name == "srv"} == {"alpha"}

    # `beta` is no longer in the active manifest set, so it won't appear
    # in the next-rendered <available_skills> block — that's the user-
    # visible removal. Note: per SEP-2640 §Discovery, a `skill://beta/...`
    # URI handed to the model (via instructions, user input, or another
    # skill) is still readable via the aggregator fanout path — removal
    # from the index doesn't mean the server stops serving the URI.


@pytest.mark.asyncio
async def test_notification_for_other_uri_ignored() -> None:
    """An updated-notification for a non-skill URI must not trigger refresh.
    We don't want to thrash discovery whenever any subscribed resource
    changes."""
    agg, _state = _make_aggregator_with_index(["alpha"])
    config = AgentConfig(name="test", instruction="x", servers=[], skills=None)
    agent = McpAgent(config=config, context=Context())
    agent._aggregator = agg

    refresh_mock = AsyncMock()
    agent._refresh_skills_after_index_update = refresh_mock  # type: ignore[assignment]

    await agent._on_server_notification(
        "srv", _make_resource_updated("skill://something-else.json")
    )
    # The callback creates a task; let pending tasks run.
    import asyncio

    await asyncio.sleep(0)
    refresh_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_notification_for_index_triggers_refresh() -> None:
    agg, _state = _make_aggregator_with_index(["alpha"])
    config = AgentConfig(name="test", instruction="x", servers=[], skills=None)
    agent = McpAgent(config=config, context=Context())
    agent._aggregator = agg

    refresh_mock = AsyncMock()
    agent._refresh_skills_after_index_update = refresh_mock  # type: ignore[assignment]

    await agent._on_server_notification(
        "srv", _make_resource_updated(INDEX_URI)
    )
    # The callback spawns a task; yield once for it to run.
    import asyncio

    await asyncio.sleep(0)
    refresh_mock.assert_awaited_once_with("srv")


@pytest.mark.asyncio
async def test_non_resource_notification_ignored() -> None:
    """Other ServerNotification kinds (logging, progress, etc.) must not
    raise or trigger a refresh."""
    agg, _state = _make_aggregator_with_index(["alpha"])
    config = AgentConfig(name="test", instruction="x", servers=[], skills=None)
    agent = McpAgent(config=config, context=Context())
    agent._aggregator = agg

    refresh_mock = AsyncMock()
    agent._refresh_skills_after_index_update = refresh_mock  # type: ignore[assignment]

    # An arbitrary plain Notification model — not a ResourceUpdated one.
    fake = MagicMock()
    fake.root = MagicMock()  # not ResourceUpdatedNotification

    await agent._on_server_notification("srv", fake)
    import asyncio

    await asyncio.sleep(0)
    refresh_mock.assert_not_awaited()
