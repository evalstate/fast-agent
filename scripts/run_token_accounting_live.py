#!/usr/bin/env python3
"""Run credentialed multi-turn CLI token-accounting smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from fast_agent.session.token_accounting_validation import validate_artifacts, validate_atif

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    model: str
    reasoning: str | None = None


DEFAULT_SCENARIOS = (
    Scenario(
        name="gpt56-low",
        model="codexresponses.gpt-5.6-terra?reasoning=low",
        reasoning="low",
    ),
    Scenario(
        name="gpt56-none",
        model="codexresponses.gpt-5.6-terra?reasoning=none",
        reasoning="none",
    ),
)

MAJOR_SCENARIOS = (
    Scenario(name="anthropic-sonnet", model="sonnet"),
    Scenario(name="google-gemini", model="gemini35flash"),
    Scenario(name="hf-kimi", model="kimi"),
    Scenario(
        name="xai-grok45",
        model="xai.grok-4.5?reasoning=low",
        reasoning="low",
    ),
)


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _scenario(value: str) -> Scenario:
    name, separator, model = value.partition("=")
    if not separator or not name or not model:
        raise argparse.ArgumentTypeError("scenarios must use NAME=MODEL")
    reasoning = None
    marker = "reasoning="
    if marker in model:
        reasoning = model.split(marker, 1)[1].split("&", 1)[0]
    return Scenario(name=_slug(name), model=model, reasoning=reasoning)


def _run(command: list[str], *, cwd: Path, log_prefix: Path) -> None:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    (log_prefix.with_suffix(".command.json")).write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    with (
        log_prefix.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout,
        log_prefix.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; "
            f"see {log_prefix.with_suffix('.stderr.log')}"
        )


def _latest_session(home: Path) -> Path:
    sessions = [
        path.parent for path in (home / "sessions").glob("*/session.json") if path.is_file()
    ]
    if not sessions:
        raise RuntimeError(f"no persisted session below {home}")
    return max(sessions, key=lambda path: (path / "session.json").stat().st_mtime_ns)


def _cache_prompt() -> str:
    stable = (
        "Fast-agent token-accounting cache probe. Keep this stable prefix unchanged. "
        "The purpose is to exceed provider prompt-cache minimums while exercising one "
        "real shell tool call and preserving enough repeated context for the next turn. "
    )
    prefix = "\n".join(f"{index:04d}: {stable}" for index in range(140))
    return (
        f"{prefix}\n\n"
        "Use the execute tool exactly once to run "
        "`printf TOKEN_ACCOUNTING_TOOL_OK`. Then reply with exactly FIRST_DONE."
    )


def _base_cli(command: list[str], home: Path, workspace: Path) -> list[str]:
    return [
        *command,
        "--quiet",
        "go",
        "--home",
        str(home),
        "--workspace",
        str(workspace),
    ]


def _export(
    *,
    home: Path,
    workspace: Path,
    session: Path,
    output: Path,
    export_format: str,
    log_prefix: Path,
    command: list[str],
) -> None:
    _run(
        [
            *command,
            "--quiet",
            "--home",
            str(home),
            "--workspace",
            str(workspace),
            "export",
            str(session),
            "--format",
            export_format,
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        log_prefix=log_prefix,
    )


def run_scenario(
    scenario: Scenario,
    *,
    output_root: Path,
    require_cache: bool,
    command: list[str],
) -> dict[str, object]:
    root = output_root / scenario.name
    workspace = root / "workspace"
    home = root / "fast-agent-home"
    artifacts = root / "artifacts"
    workspace.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    prompt_path = artifacts / "cache-prompt.txt"
    prompt_path.write_text(_cache_prompt(), encoding="utf-8")
    turn_one_atif = artifacts / "runtime-turn-1.atif.json"
    turn_two_atif = artifacts / "runtime-turn-2.atif.json"

    _run(
        [
            *_base_cli(command, home, workspace),
            "--model",
            scenario.model,
            "--prompt-file",
            str(prompt_path),
            "--shell",
            "--trajectory-output",
            str(turn_one_atif),
        ],
        cwd=REPO_ROOT,
        log_prefix=artifacts / "turn-1",
    )
    session = _latest_session(home)

    _run(
        [
            *_base_cli(command, home, workspace),
            "--model",
            scenario.model,
            "--resume",
            session.name,
            "--message",
            "Without using tools, reply with exactly SECOND_DONE.",
            "--shell",
            "--trajectory-output",
            str(turn_two_atif),
        ],
        cwd=REPO_ROOT,
        log_prefix=artifacts / "turn-2",
    )

    codex_path = artifacts / "session.codex.jsonl"
    atif_path = artifacts / "session.atif.json"
    _export(
        home=home,
        workspace=workspace,
        session=session,
        output=codex_path,
        export_format="codex",
        log_prefix=artifacts / "export-codex",
        command=command,
    )
    _export(
        home=home,
        workspace=workspace,
        session=session,
        output=atif_path,
        export_format="atif",
        log_prefix=artifacts / "export-atif",
        command=command,
    )

    validation: dict[str, object]
    try:
        validate_atif(turn_one_atif, require_tool=True, expect_reasoning=scenario.reasoning)
        validate_atif(turn_two_atif, expect_reasoning=scenario.reasoning)
        report = validate_artifacts(
            session_dir=session,
            codex_path=codex_path,
            atif_path=atif_path,
            require_cache=require_cache,
            require_tool=True,
            expect_reasoning=scenario.reasoning,
        )
        validation = {"status": "passed", "report": asdict(report)}
    except Exception as exc:
        validation = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    payload: dict[str, object] = {
        "scenario": asdict(scenario),
        "session": str(session),
        "codex": str(codex_path),
        "atif": str(atif_path),
        "validation": validation,
    }
    (root / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / ".artifacts"
        / "token-accounting"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--package-version",
        help="Run the published fast-agent-mcp version through uvx.",
    )
    source.add_argument(
        "--wheel",
        type=Path,
        help="Run a local fast-agent wheel through uvx.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Retain failed scenario artifacts and continue the matrix.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        type=_scenario,
        default=[],
        help="Run NAME=MODEL. Repeatable; replaces default GPT-5.6 scenarios.",
    )
    parser.add_argument(
        "--major-models",
        action="store_true",
        help="Also run Anthropic, Google, and HF/Kimi scenarios.",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Fail unless the exported trajectory contains a positive cache read.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = tuple(args.scenario) or DEFAULT_SCENARIOS
    if args.major_models:
        scenarios += MAJOR_SCENARIOS
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.package_version:
        source = f"fast-agent-mcp=={args.package_version}"
        command = ["uvx", "--from", source, "fast-agent"]
    elif args.wheel:
        wheel = args.wheel.expanduser().resolve()
        source = str(wheel)
        command = ["uvx", "--from", source, "fast-agent"]
    else:
        source = "worktree"
        command = ["uv", "run", "fast-agent"]

    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "source": source,
        "source_sha256": (
            hashlib.sha256(args.wheel.expanduser().resolve().read_bytes()).hexdigest()
            if args.wheel
            else None
        ),
        "command": command,
        "scenarios": [asdict(scenario) for scenario in scenarios],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    reports: list[dict[str, object]] = []
    for scenario in scenarios:
        try:
            reports.append(
                run_scenario(
                    scenario,
                    output_root=output_root,
                    require_cache=args.require_cache,
                    command=command,
                )
            )
        except Exception as exc:
            failure: dict[str, object] = {
                "scenario": asdict(scenario),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            (output_root / scenario.name / "failure.json").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (output_root / scenario.name / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n",
                encoding="utf-8",
            )
            reports.append(failure)
            if not args.continue_on_error:
                raise
    summary = output_root / "report.json"
    summary.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"token-accounting live suite complete: {summary}")


if __name__ == "__main__":
    main()
