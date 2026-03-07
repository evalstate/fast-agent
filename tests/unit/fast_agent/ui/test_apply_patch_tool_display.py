from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay


def test_shell_tool_call_apply_patch_renders_preview_and_other_args() -> None:
    display = ConsoleDisplay()
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Add File: hello.txt\n"
        "+hello\n"
        "*** End Patch\n"
        "PATCH"
    )

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="execute",
            tool_args={"command": command, "cwd": "/tmp/work", "timeout_seconds": 90},
            metadata={"variant": "shell", "command": command, "shell_name": "bash"},
            name="dev",
        )

    rendered = capture.get()
    assert "$ apply_patch (preview)" in rendered
    assert "apply_patch preview:" in rendered
    assert "*** Begin Patch" in rendered
    assert "other args:" in rendered
    assert '"cwd": "/tmp/work"' in rendered
    assert '"timeout_seconds": 90' in rendered


def test_shell_tool_call_falls_back_to_raw_command_when_preview_unavailable() -> None:
    display = ConsoleDisplay()
    command = "apply_patch 'not-a-valid-patch-payload'"

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="execute",
            tool_args={"command": command},
            metadata={"variant": "shell", "command": command, "shell_name": "bash"},
            name="dev",
        )

    rendered = capture.get()
    assert "$ apply_patch 'not-a-valid-patch-payload'" in rendered
    assert "apply_patch preview:" not in rendered



def test_apply_patch_tool_call_renders_preview() -> None:
    display = ConsoleDisplay()
    patch_text = (
        "*** Begin Patch\n"
        "*** Add File: hello.txt\n"
        "+hello\n"
        "*** End Patch\n"
    )

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="apply_patch",
            tool_args={"input": patch_text},
            metadata={},
            name="dev",
        )

    rendered = capture.get()
    assert "apply_patch (preview)" in rendered
    assert "*** Begin Patch" in rendered
    assert "apply_patch preview:" in rendered
