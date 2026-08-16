from fast_agent.tools.shell_tool_definitions import build_luna_exec_tool


def test_luna_timeout_guidance_changes_only_the_top_level_description() -> None:
    tool = build_luna_exec_tool(shell_name="bash")
    description = tool.description
    assert description is not None

    assert "Omit `timeout` for ordinary commands" in description
    assert "intentionally bounding disposable exploratory work" in description
    assert "complete required work without blindly repeating" in description
    assert set(tool.input_schema["properties"]) == {
        "background",
        "command",
        "timeout",
        "working_directory",
    }
    assert tool.input_schema["properties"]["timeout"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 600,
        "description": (
            "Optional foreground hard deadline in seconds. Suppresses "
            "normal auto-yield and terminates the process group on expiry."
        ),
    }
