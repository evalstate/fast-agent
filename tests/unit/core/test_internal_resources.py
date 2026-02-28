from __future__ import annotations

import pytest

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.internal_resources import (
    format_internal_resources_for_prompt,
    get_internal_resource,
    list_internal_resources,
    read_internal_resource,
)


def test_internal_resource_manifest_loads_catalog() -> None:
    resources = list_internal_resources()

    assert resources
    uris = {resource.uri for resource in resources}
    assert "internal://fast-agent/smart-agent-cards" in uris


def test_internal_resource_read_returns_expected_content() -> None:
    body = read_internal_resource("internal://fast-agent/smart-agent-cards")

    assert "<AgentCards>" in body
    assert "Agent Card (type: `agent`)" in body


def test_get_internal_resource_unknown_uri_raises() -> None:
    with pytest.raises(AgentConfigError, match="Unknown internal resource URI"):
        get_internal_resource("internal://fast-agent/does-not-exist")


def test_format_internal_resources_for_prompt_includes_uri_and_why() -> None:
    prompt_block = format_internal_resources_for_prompt(list_internal_resources())

    assert "<available_resources>" in prompt_block
    assert "internal://fast-agent/smart-agent-cards" in prompt_block
    assert "Use when creating, validating, or loading AgentCards" in prompt_block
