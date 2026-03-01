from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fast_agent.ui.model_picker import run_model_picker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prompt Toolkit model picker prototype (integrated module)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to fastagent.config.yaml",
    )
    args = parser.parse_args()

    try:
        result = run_model_picker(config_path=args.config)
    except ValueError as exc:
        print(str(exc))
        return 1

    if result is None:
        print("Cancelled.")
        return 1

    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
