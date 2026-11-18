import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts/deadcode.py"


@pytest.mark.parametrize(
    ("source", "exit_code"),
    [
        ("print('live')\n", 0),
        (
            "def greet():\n    return 'hello'\n    print('unreachable')\n\nprint(greet())\n",
            3,
        ),
    ],
)
def test_deadcode_cli_propagates_findings(tmp_path: Path, source: str, exit_code: int) -> None:
    # The repository shim exclusion must not skip unrelated files with its name.
    target = tmp_path / "unsupported_methods.py"
    target.write_text(source)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(target)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == exit_code, result.stdout + result.stderr
    if exit_code:
        assert "unreachable code" in result.stdout
        assert "100% confidence" in result.stdout
