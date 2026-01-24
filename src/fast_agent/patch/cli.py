from __future__ import annotations

import sys
from typing import TextIO

from fast_agent.patch.engine import apply_patch
from fast_agent.patch.errors import ApplyPatchError

USAGE = "Usage: apply_patch 'PATCH'\n       echo 'PATCH' | apply-patch"


def main() -> int:
    patch_arg, exit_code = _read_patch_argument(sys.argv[1:], sys.stdin, sys.stderr)
    if patch_arg is None:
        return exit_code

    try:
        apply_patch(patch_arg, sys.stdout, sys.stderr)
    except ApplyPatchError:
        return 1
    return 0


def _read_patch_argument(
    args: list[str],
    stdin: TextIO,
    stderr: TextIO,
) -> tuple[str | None, int]:
    if len(args) > 1:
        stderr.write("Error: apply_patch accepts exactly one argument.\n")
        return None, 2
    if len(args) == 1:
        return args[0], 0

    try:
        patch = stdin.read()
    except OSError as exc:
        stderr.write(f"Error: Failed to read PATCH from stdin.\n{exc}\n")
        return None, 1

    if not patch:
        stderr.write(f"{USAGE}\n")
        return None, 2
    return patch, 0


if __name__ == "__main__":
    raise SystemExit(main())
