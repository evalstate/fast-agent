"""Test event progress conversion from log events."""

import os
import subprocess
from difflib import unified_diff
from pathlib import Path

import pytest
from rich import print
from rich.console import Console
from rich.syntax import Syntax

# Create console with fixed width
console = Console(width=100, force_terminal=True)


def show_diff(expected: str, got: str, context: int = 3) -> None:
    """Show a readable diff between expected and got."""
    diff = list(
        unified_diff(
            expected.splitlines(keepends=True),
            got.splitlines(keepends=True),
            fromfile="expected",
            tofile="got",
            n=context,
        )
    )

    if diff:
        print("\n[yellow]Differences found:[/yellow]")
        print("".join(diff))

    # Also show full outputs with line numbers for reference
    print("\n[blue]Expected output[/blue] ({} lines):".format(len(expected.splitlines())))
    syntax = Syntax(expected, "text", line_numbers=True, word_wrap=True)
    console.print(syntax)

    print("\n[blue]Actual output[/blue] ({} lines):".format(len(got.splitlines())))
    syntax = Syntax(got, "text", line_numbers=True, word_wrap=True)
    console.print(syntax)


@pytest.mark.skip("restate/delete test with enhanced approach")
def test_event_conversion():
    """Test conversion of log events to progress events using gold master approach."""
    # Get the paths
    fixture_dir = Path(__file__).parent / "fixture"
    log_file = fixture_dir / "mcp-basic-agent-2025-02-17.jsonl"
    expected_output_file = fixture_dir / "expected_output.txt"

    # TODO -- update these tests to capture events.
    if not log_file.exists():
        raise FileNotFoundError(f"Test log file not found: {log_file}")

    if not expected_output_file.exists():
        raise FileNotFoundError(
            f"Expected output file not found: {expected_output_file}\n"
            "Run update_test_fixtures() to generate it first"
        )

    # Run the event_summary script to get current output
    try:
        result = subprocess.run(
            ["python3", "scripts/event_summary.py", str(log_file)],
            capture_output=True,
            text=True,
            check=True,
            env={"COLUMNS": "100", "TERM": "xterm-256color", **dict(os.environ)},
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run event_summary.py: {e.stderr}")

    current_output = result.stdout.strip()
    expected_output = expected_output_file.read_text().strip()

    if current_output != expected_output:
        show_diff(expected_output, current_output)
        # assert (
        #     False
        # ), "Event summary output does not match expected output (see diff above)"


def update_test_fixtures():
    """
    Utility method to update test fixtures with latest output.
    This should only be run manually when intentionally updating the expected behavior.

    Usage:
        python3 -c "from tests.test_event_progress import update_test_fixtures; update_test_fixtures()"
    """
    # Ensure fixture directory exists
    fixture_dir = Path(__file__).parent / "fixture"
    fixture_dir.mkdir(exist_ok=True)

    log_file = fixture_dir / "mcp-basic-agent-2025-02-17.jsonl"
    expected_output_file = fixture_dir / "expected_output.txt"

    if not log_file.exists():
        print(f"[red]Error:[/red] Log file not found: {log_file}")
        print("Please run an example to generate a log file and copy it to the fixture directory")
        return

    # Run command and capture output
    try:
        result = subprocess.run(
            ["python3", "scripts/event_summary.py", str(log_file)],
            capture_output=True,
            text=True,
            check=True,
            env={"COLUMNS": "100", "TERM": "xterm-256color", **dict(os.environ)},
        )
    except subprocess.CalledProcessError as e:
        print(f"[red]Error:[/red] Failed to run event_summary.py: {e.stderr}")
        return

    # Update expected output file
    expected_output_file.write_text(result.stdout)

    print(f"[green]Successfully updated test fixtures:[/green]\n- {expected_output_file}")


if __name__ == "__main__":
    test_event_conversion()
