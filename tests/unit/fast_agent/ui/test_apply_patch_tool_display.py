from fast_agent.config import LoggerSettings, Settings
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
    assert "apply_patch 'not-a-valid-patch-payload'" in rendered
    assert "apply_patch preview:" not in rendered


def test_shell_tool_call_renders_powershell_command_as_code_block() -> None:
    display = ConsoleDisplay()
    command = "Get-ChildItem | Select-Object -First 5"

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="execute",
            tool_args={"command": command},
            metadata={"variant": "shell", "command": command, "shell_name": "pwsh"},
            name="dev",
        )

    rendered = capture.get()
    assert "Get-ChildItem | Select-Object -First 5" in rendered
    assert "apply_patch preview:" not in rendered


def test_shell_tool_call_renders_code_without_markdown_padding() -> None:
    display = ConsoleDisplay()
    command = "echo hi"

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="execute",
            tool_args={"command": command},
            metadata={"variant": "shell", "command": command, "shell_name": "bash"},
            name="dev",
        )

    rendered_lines = capture.get().splitlines()
    command_lines = [line for line in rendered_lines if "echo hi" in line]
    assert command_lines
    assert any(line.startswith("echo hi") for line in command_lines)



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


def test_apply_patch_tool_call_respects_preview_line_limit() -> None:
    display = ConsoleDisplay(
        Settings(logger=LoggerSettings(apply_patch_preview_max_lines=4))
    )
    patch_text = (
        "*** Begin Patch\n"
        "*** Add File: hello.txt\n"
        "+line-1\n"
        "+line-2\n"
        "+line-3\n"
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
    assert "(+2 more lines)" in rendered
