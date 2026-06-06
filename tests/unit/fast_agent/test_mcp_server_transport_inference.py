"""Tests for automatic transport inference in MCPServerSettings."""

import warnings

import pytest

from fast_agent.config import MCPServerSettings


@pytest.mark.parametrize(
    ("kwargs", "expected_transport", "expected_url", "expected_command"),
    [
        ({"url": "http://example.com/mcp"}, "http", "http://example.com/mcp", None),
        ({"command": "npx server"}, "stdio", None, "npx server"),
        ({}, "stdio", None, None),
        ({"url": ""}, "stdio", "", None),
        ({"url": "   "}, "stdio", "   ", None),
        ({"command": ""}, "stdio", None, ""),
        ({"command": "   "}, "stdio", None, "   "),
        ({"url": "", "command": ""}, "stdio", "", ""),
        ({"url": "http://example.com/mcp", "command": ""}, "http", "http://example.com/mcp", ""),
        ({"command": "npx server", "url": ""}, "stdio", "", "npx server"),
        (
            {"url": "http://example.com/mcp", "command": "   "},
            "http",
            "http://example.com/mcp",
            "   ",
        ),
    ],
)
def test_transport_inference_matrix(
    kwargs,
    expected_transport,
    expected_url,
    expected_command,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = MCPServerSettings(**kwargs)

    assert [warning for warning in caught if warning.category is UserWarning] == []
    assert config.transport == expected_transport
    assert config.url == expected_url
    assert config.command == expected_command


def test_transport_inference_both_url_and_command():
    """Test that providing both URL and command prefers HTTP and warns."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = MCPServerSettings(url="http://example.com/mcp", command="npx server")
        resource_traces = []
        for warning in w:
            if warning.category is ResourceWarning and warning.source is not None:
                source_tb = getattr(warning.source, "_source_traceback", None)
                if source_tb:
                    import traceback

                    trace_str = "".join(traceback.format_list(source_tb))
                    resource_traces.append(trace_str)
                    print(f"ResourceWarning source traceback:\n{trace_str}")
            print(
                f"{warning.category.__name__} from {warning.filename}:{warning.lineno} -> {warning.message}"
            )

        user_warnings = [warning for warning in w if warning.category is UserWarning]
        assert not resource_traces, "Unexpected ResourceWarnings captured"
        # Check that warning was issued
        assert len(user_warnings) == 1
        assert "both 'url'" in str(user_warnings[0].message)
        assert "Preferring HTTP transport" in str(user_warnings[0].message)

        # Check that HTTP transport is selected and command is cleared
        assert config.transport == "http"
        assert config.url == "http://example.com/mcp"
        assert config.command is None  # Should be cleared


def test_transport_inference_explicit_transport_overrides():
    """Test that explicitly setting transport overrides inference."""
    # URL with explicit SSE transport should keep SSE
    config = MCPServerSettings(url="http://example.com/sse", transport="sse")
    assert config.transport == "sse"
    assert config.url == "http://example.com/sse"

    # Command with explicit HTTP transport should keep HTTP
    config = MCPServerSettings(command="npx server", transport="http")
    assert config.transport == "http"
    assert config.command == "npx server"


def test_transport_inference_preserves_other_fields():
    """Test that inference doesn't affect other configuration fields."""
    config = MCPServerSettings(
        url="http://example.com/mcp",
        name="test_server",
        description="A test server",
        args=["--verbose"],
        read_timeout_seconds=30,
        env={"KEY": "value"},
    )

    assert config.transport == "http"
    assert config.name == "test_server"
    assert config.description == "A test server"
    assert config.args == ["--verbose"]
    assert config.read_timeout_seconds == 30
    assert config.env == {"KEY": "value"}
