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


def test_shell_tool_call_header_includes_timeout() -> None:
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="execute",
            tool_args={"command": "echo hi"},
            metadata={
                "variant": "shell",
                "command": "echo hi",
                "shell_name": "bash",
                "shell_path": "/bin/bash",
                "timeout_seconds": 90,
            },
            name="dev",
        )

    rendered = capture.get()
    assert "bash (/bin/bash) | timeout 90s" in rendered


def test_background_shell_tool_call_header_shows_background() -> None:
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="execute",
            tool_args={"command": "sleep 30", "background": True},
            metadata={
                "variant": "shell",
                "command": "sleep 30",
                "shell_name": "bash",
                "shell_path": "/bin/bash",
                "background": True,
                "idle_yield_seconds": 10,
                "foreground_yield_seconds": 30,
            },
            name="dev",
        )

    rendered = capture.get()
    assert "bash (/bin/bash) | background" in rendered
    assert "idle yield" not in rendered


def test_process_lifecycle_tool_calls_use_compact_display() -> None:
    display = ConsoleDisplay(
        config=Settings(logger=LoggerSettings(progress_display=False))
    )

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="poll_process",
            tool_args={"process_id": "process-3", "wait_sec": 5},
            metadata={
                "variant": "shell_process",
                "action": "poll",
                "process_id": "process-3",
                "wait_sec": 5,
                "command_summary": "uv run worker.py",
                "elapsed_seconds": 65,
                "os_process_id": 4321,
            },
            name="dev",
        )
        display.show_tool_call(
            tool_name="terminate_process",
            tool_args={"process_id": "process-4"},
            metadata={
                "variant": "shell_process",
                "action": "terminate",
                "process_id": "process-4",
            },
            name="dev",
        )

    rendered = capture.get()
    assert "dev monitoring · pid 4321 · ≤5s · 1m 05s · uv run worker.py" in rendered
    assert "process-3" not in rendered
    assert "terminate process-4" in rendered
    assert "'process_id'" not in rendered


def test_process_poll_tool_call_uses_live_progress_instead_of_transcript() -> None:
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="poll_process",
            tool_args={"process_id": "process-3", "wait_sec": 30},
            metadata={
                "variant": "shell_process",
                "action": "poll",
                "process_id": "process-3",
                "wait_sec": 30,
            },
            name="dev",
        )

    assert capture.get() == ""


def test_apply_patch_tool_call_renders_preview() -> None:
    display = ConsoleDisplay()
    patch_text = "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n"

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
    display = ConsoleDisplay(Settings(logger=LoggerSettings(apply_patch_preview_max_lines=4)))
    patch_text = (
        "*** Begin Patch\n*** Add File: hello.txt\n+line-1\n+line-2\n+line-3\n*** End Patch\n"
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
