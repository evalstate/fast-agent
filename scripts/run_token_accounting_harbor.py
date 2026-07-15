#!/usr/bin/env python3
"""Run the token-accounting matrix on Terminal-Bench with local Docker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from fast_agent.session.token_accounting_validation import validate_artifacts, validate_atif

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARBOR_ROOT = Path("~/source/harbor").expanduser()
DATASET_REF = "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "XAI_API_KEY",
)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    model: str
    reasoning: str | None = None


DEFAULT_SCENARIOS = (
    Scenario(name="anthropic-sonnet", model="sonnet"),
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


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}; see {log_path}")


def _build_wheel(output_root: Path) -> Path:
    wheel_dir = output_root / "dist"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=REPO_ROOT,
        log_path=output_root / "build-wheel.log",
    )
    wheels = sorted(wheel_dir.glob("fast_agent_mcp-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one fast-agent wheel in {wheel_dir}, found {wheels}")
    return wheels[0].resolve()


def _agent_env() -> dict[str, str]:
    return {name: f"${{{name}}}" for name in CREDENTIALS if os.environ.get(name)}


def _config(
    *,
    harbor_root: Path,
    jobs_dir: Path,
    wheel: Path,
    scenario: Scenario,
    task: str,
    environment: str,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "wheel_path": str(wheel),
        "fast_agent_model": scenario.model,
        "shell": True,
    }
    if scenario.reasoning is not None:
        kwargs["reasoning_effort"] = scenario.reasoning
    codex_auth = Path("~/.codex/auth.json").expanduser()
    if scenario.name.startswith("gpt56-") and codex_auth.is_file():
        kwargs["codex_auth_path"] = str(codex_auth.resolve())

    return {
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "environment": {
            "type": environment,
            "force_build": True,
            "delete": True,
        },
        "agents": [
            {
                "name": "fast-agent",
                "import_path": "fast_agent_harbor:FastAgent",
                "model_name": "",
                "kwargs": kwargs,
                "env": _agent_env(),
            }
        ],
        "datasets": [
            {
                "name": "terminal-bench/terminal-bench-2-1",
                "ref": DATASET_REF,
                "task_names": [f"terminal-bench/{task}"],
            }
        ],
        "metadata": {
            "token_accounting": True,
            "fast_agent_repo": str(REPO_ROOT),
            "harbor_repo": str(harbor_root),
        },
    }


def _export_session(
    *,
    session_dir: Path,
    workspace: Path,
    output: Path,
    export_format: str,
    log_path: Path,
    wheel: Path,
) -> None:
    _run(
        [
            "uvx",
            "--from",
            str(wheel),
            "fast-agent",
            "--quiet",
            "--workspace",
            str(workspace),
            "export",
            str(session_dir),
            "--format",
            export_format,
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        log_path=log_path,
    )


def run_scenario(
    scenario: Scenario,
    *,
    harbor_root: Path,
    output_root: Path,
    wheel: Path,
    task: str,
    environment: str,
    require_cache: bool,
) -> list[dict[str, object]]:
    scenario_root = output_root / scenario.name
    jobs_dir = scenario_root / "jobs"
    job_name = f"token-accounting-{scenario.name}"
    config_path = scenario_root / "harbor.json"
    scenario_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            _config(
                harbor_root=harbor_root,
                jobs_dir=jobs_dir,
                wheel=wheel,
                scenario=scenario,
                task=task,
                environment=environment,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{harbor_root}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(harbor_root)
    )
    _run(
        [
            "uv",
            "run",
            "--project",
            str(harbor_root),
            "harbor",
            "run",
            "--config",
            str(config_path),
            "--job-name",
            job_name,
            "--yes",
        ],
        cwd=harbor_root,
        log_path=scenario_root / "harbor.log",
        env=env,
    )

    job_dir = jobs_dir / job_name
    trajectory_paths = sorted(job_dir.glob("*/agent/trajectory.json"))
    if not trajectory_paths:
        raise RuntimeError(f"no Harbor trajectories below {job_dir}")

    reports: list[dict[str, object]] = []
    for runtime_atif in trajectory_paths:
        trial_dir = runtime_atif.parents[1]
        home = trial_dir / "agent" / "fast-agent-home"
        sessions = sorted(home.glob("sessions/*/session.json"))
        if len(sessions) != 1:
            raise RuntimeError(f"expected one persisted session below {home}, found {sessions}")
        session_dir = sessions[0].parent
        artifact_dir = trial_dir / "agent" / "token-accounting"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        codex_path = artifact_dir / "session.codex.jsonl"
        atif_path = artifact_dir / "session.atif.json"
        _export_session(
            session_dir=session_dir,
            workspace=trial_dir / "agent",
            output=codex_path,
            export_format="codex",
            log_path=artifact_dir / "export-codex.log",
            wheel=wheel,
        )
        _export_session(
            session_dir=session_dir,
            workspace=trial_dir / "agent",
            output=atif_path,
            export_format="atif",
            log_path=artifact_dir / "export-atif.log",
            wheel=wheel,
        )

        validation: dict[str, object] = {}
        try:
            validate_atif(
                runtime_atif,
                require_tool=True,
                expect_reasoning=scenario.reasoning,
            )
            report = validate_artifacts(
                session_dir=session_dir,
                codex_path=codex_path,
                atif_path=atif_path,
                require_cache=require_cache,
                require_tool=True,
                expect_reasoning=scenario.reasoning,
            )
            validation.update(status="passed", report=asdict(report))
        except Exception as exc:
            validation.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        payload: dict[str, object] = {
            "scenario": asdict(scenario),
            "trial": str(trial_dir),
            "runtime_atif": str(runtime_atif),
            "session": str(session_dir),
            "codex": str(codex_path),
            "exported_atif": str(atif_path),
            "validation": validation,
        }
        (artifact_dir / "report.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        reports.append(payload)
    return reports


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor-root", type=Path, default=DEFAULT_HARBOR_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / ".artifacts"
        / "token-accounting-harbor"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--task", default="cancel-async-tasks")
    parser.add_argument(
        "--environment",
        choices=("docker", "daytona"),
        default="docker",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help="Use this wheel instead of building the current worktree.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        type=_scenario,
        default=[],
        help="Run NAME=MODEL. Repeatable; replaces the default major-route matrix.",
    )
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Retain failed scenario artifacts and continue the matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    harbor_root = args.harbor_root.expanduser().resolve()
    if not (harbor_root / "fast_agent_harbor.py").is_file():
        raise SystemExit(f"Harbor fast-agent adapter not found below {harbor_root}")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    wheel = (
        args.wheel.expanduser().resolve() if args.wheel is not None else _build_wheel(output_root)
    )
    if not wheel.is_file():
        raise SystemExit(f"fast-agent wheel not found: {wheel}")
    scenarios = tuple(args.scenario) or DEFAULT_SCENARIOS

    reports: list[dict[str, object]] = []
    for scenario in scenarios:
        try:
            reports.extend(
                run_scenario(
                    scenario,
                    harbor_root=harbor_root,
                    output_root=output_root,
                    wheel=wheel,
                    task=args.task,
                    environment=args.environment,
                    require_cache=args.require_cache,
                )
            )
        except Exception as exc:
            failure: dict[str, object] = {
                "scenario": asdict(scenario),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            scenario_root = output_root / scenario.name
            scenario_root.mkdir(parents=True, exist_ok=True)
            (scenario_root / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n",
                encoding="utf-8",
            )
            reports.append(failure)
            if not args.continue_on_error:
                raise
    summary = output_root / "report.json"
    summary.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"token-accounting Harbor suite complete: {summary}")


if __name__ == "__main__":
    main()
