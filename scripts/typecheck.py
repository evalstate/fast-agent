# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "typer",
#     "ty",
# ]
# ///

import subprocess
import sys

import typer
from rich import print


def main(path: str = "src") -> None:
    try:
        command = ["ty", "check", path]
        process = subprocess.run(
            command,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        sys.exit(process.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: `ty` command not found. Make sure it's installed in the environment.")
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
