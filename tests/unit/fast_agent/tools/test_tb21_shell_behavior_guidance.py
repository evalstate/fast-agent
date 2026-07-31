from pathlib import Path

from fast_agent.tools.shell_output import ShellOutputBuffer
from fast_agent.tools.shell_tool_definitions import (
    build_minimal_bash_tool,
    build_minimal_process_tool,
)


def test_minimal_bash_guidance_requires_managed_process_and_output_followup() -> None:
    tool = build_minimal_bash_tool(shell_name="bash")
    assert tool.description is not None
    description = tool.description

    assert "Do not assume a yielded process completed" in description
    assert "Process with `wait` or `status`" in description
    assert "retained-output path" in description
    assert "read_text_file" in description
    assert "task-relevant verification" in description
    assert "Do not use shell `&`, `nohup`, or `disown`" in description


def test_minimal_process_guidance_requires_completion_check() -> None:
    tool = build_minimal_process_tool(
        default_wait_seconds=240,
        max_wait_seconds=250,
    )
    assert tool.description is not None
    description = tool.description

    assert "use `wait` or `status` until completion" in description
    assert "before relying on its result or ending the task" in description


def test_truncation_guidance_requires_targeted_retained_output_inspection(
    tmp_path: Path,
) -> None:
    retained = tmp_path / "output.log"
    buffer = ShellOutputBuffer(
        output_byte_limit=16,
        retained_output_path=retained,
        retained_output_max_bytes=1024,
    )
    buffer.append("output larger than the preview limit")

    text = buffer.combined()

    assert "before drawing conclusions from truncated output" in text
    assert "read_text_file" in text
    assert str(retained) in text
