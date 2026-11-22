"""Gate high-confidence dead code; lower-confidence scans are advisory.

Run with the locked development dependencies: uv run scripts/deadcode.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = (
    PROJECT_ROOT / "src/fast_agent",
    PROJECT_ROOT / "publish/fast-agent-acp/src",
    PROJECT_ROOT / "publish/hf-inference-acp/src",
)
ALLOWLIST = PROJECT_ROOT / "scripts/vulture_allowlist.py"
# This shim only rejects unsupported ACP requests or ignores notifications.
# Its required SDK parameter names would otherwise need global suppression,
# hiding unrelated unused arguments in real implementations.
UNSUPPORTED_ACP_SHIM = PROJECT_ROOT / "src/fast_agent/acp/server/unsupported_methods.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_TARGETS,
        help="Paths to scan (default: all three shipped Python source trees)",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        choices=(60, 90, 100),
        default=100,
        help="Minimum confidence (default: 100); 60 and 90 are for manual review",
    )
    args = parser.parse_args()
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "vulture",
            *(str(path) for path in args.paths),
            str(ALLOWLIST),
            "--exclude",
            str(UNSUPPORTED_ACP_SHIM),
            "--min-confidence",
            str(args.min_confidence),
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
