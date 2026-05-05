#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from typing import cast

from fast_agent.cli.checks.structured_tools_probe import (
    StructuredToolPolicy,
    _print_text_summary,
    run_probe,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether a model can use a tool and still return valid structured JSON "
            "through fast-agent's ToolAgent path."
        )
    )
    parser.add_argument("models", nargs="+", help="Model ids or aliases to probe.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--structured-tool-policy",
        choices=("auto", "always", "defer", "no_tools"),
        default="auto",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    results = await run_probe(
        args.models,
        structured_tool_policy=cast("StructuredToolPolicy", args.structured_tool_policy),
    )
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    else:
        _print_text_summary(results)
    return 0 if all(result.passed for result in results) else 1


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())
