from __future__ import annotations

import pytest

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.internal_resources import _parse_manifest_entry, get_internal_resource


def test_parse_manifest_entry_normalizes_text_fields() -> None:
    resource = _parse_manifest_entry(
        0,
        {
            "uri": "  internal://fast-agent/demo  ",
            "title": "  Demo  ",
            "description": "  Example resource  ",
            "why": "  Useful for tests  ",
            "source": "  docs/demo.md  ",
            "mime_type": "  text/markdown  ",
            "tags": [" docs ", " ", "tests"],
        },
    )

    assert resource.uri == "internal://fast-agent/demo"
    assert resource.title == "Demo"
    assert resource.description == "Example resource"
    assert resource.why == "Useful for tests"
    assert resource.source == "docs/demo.md"
    assert resource.mime_type == "text/markdown"
    assert resource.tags == ("docs", "tests")


def test_parse_manifest_entry_rejects_blank_required_text() -> None:
    with pytest.raises(AgentConfigError, match="missing non-empty 'title'"):
        _parse_manifest_entry(
            0,
            {
                "uri": "internal://fast-agent/demo",
                "title": " ",
                "description": "Example resource",
                "why": "Useful for tests",
                "source": "docs/demo.md",
            },
        )


def test_get_internal_resource_rejects_blank_uri() -> None:
    with pytest.raises(AgentConfigError, match="URI must not be empty"):
        get_internal_resource("   ")
