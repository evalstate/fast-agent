"""Opt-in live conformance and small, transcript-free evidence."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import pytest

from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.usage_tracking import TurnUsage, summarize_usage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import JsonValue

    from fast_agent.llm.resolved_model import ResolvedModelSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHECKS = (
    "text-stream",
    "reasoning-efforts",
    "tool-loop",
    "multi-turn",
    "structured",
    "media",
    "usage",
    "trace-review",
    "interactive",
    "ptc/uptake",
)
Result = Literal["pass", "warn", "fail", "skip", "manual"]
RANK: dict[Result, int] = {"pass": 0, "skip": 1, "manual": 1, "warn": 2, "fail": 3}


def now() -> str:
    return datetime.now(UTC).isoformat()


def reasoning_values(resolved: ResolvedModelSpec, track: str) -> list[str]:
    spec = resolved.reasoning_effort_spec
    if spec is None:
        return []
    if spec.kind == "effort":
        values = list(spec.allowed_efforts or [])
    elif spec.kind == "toggle":
        values = ["off", "on"]
    else:
        values = [
            str(v)
            for v in (
                spec.budget_presets
                or [v for v in (spec.min_budget_tokens, spec.max_budget_tokens) if v is not None]
            )
        ]
    if (spec.allow_toggle_disable or spec.kind == "budget") and "off" not in values:
        values.insert(0, "off")
    if track == "lite" and values:
        values = [values[0], values[-1]]
    elif spec.allow_auto:
        values.append("auto")
    return list(dict.fromkeys([*values, "default"]))


@dataclass
class TestEvidence:
    name: str
    outcome: Result = "skip"
    duration_s: float = 0
    details: dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class Evidence:
    model: str
    track: str
    resolved: ResolvedModelSpec | None
    path: Path
    started: str = field(default_factory=now)
    tests: dict[str, dict[str, TestEvidence]] = field(default_factory=dict)
    usage: list[TurnUsage] = field(default_factory=list)
    finished: str | None = None

    def checks(self) -> list[dict[str, object]]:
        checks: list[dict[str, object]] = []
        for check in CHECKS:
            tests = list(self.tests.get(check, {}).values())
            result: Result = (
                "manual"
                if check in {"trace-review", "interactive"}
                else max((t.outcome for t in tests), key=RANK.__getitem__, default="skip")
            )
            counts = {
                outcome: sum(t.outcome == outcome for t in tests)
                for outcome in ("pass", "warn", "fail", "skip")
            }
            summary = (
                "Requires human review"
                if result == "manual"
                else ", ".join(f"{n} {status}" for status, n in counts.items() if n)
                or "Not selected"
            )
            checks.append(
                dict(
                    id=check,
                    stage="ptc" if check == "ptc/uptake" else "conformance",
                    manifest_check_id="uptake" if check == "ptc/uptake" else check,
                    result=result,
                    summary=summary,
                    tests=[asdict(t) for t in tests],
                )
            )
        return checks

    def write(self) -> None:
        import json

        spec = self.resolved
        params = spec.model_params if spec else None
        resolved = {
            "provider": spec.provider.value if spec else None,
            "model_name": spec.wire_model_name if spec else None,
            **(
                params.model_dump(
                    mode="json",
                    include={
                        "context_window",
                        "max_output_tokens",
                        "tokenizes",
                        "json_mode",
                        "reasoning_effort_spec",
                        "ptc_enabled",
                        "structured_tool_policy",
                    },
                )
                if params
                else {}
            ),
        }
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        payload = dict(
            model=self.model,
            resolved=resolved,
            track=self.track,
            fast_agent_version=version("fast-agent-mcp"),
            git_rev=revision.stdout.strip() or None,
            started=self.started,
            finished=self.finished,
            checks=self.checks(),
            usage_total=summarize_usage(self.usage).model_dump(mode="json")
            if self.usage
            else {"provider_attempts": 0, "prompt": {"total": None}, "completion": {"total": None}},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.path)


STATE = pytest.StashKey[Evidence]()


@dataclass
class Notes:
    state: Evidence
    nodeid: str

    def note(self, check_id: str, **details: JsonValue) -> None:
        self.state.tests[check_id][self.nodeid].details.update(details)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "onboarding(*checks): manifest check ids")
    model = os.environ.get("ONBOARD_MODEL", "").strip()
    track = os.environ.get("ONBOARD_TRACK", "full")
    if model and track not in {"full", "lite"}:
        raise pytest.UsageError("ONBOARD_TRACK must be full or lite")
    if model and config.getoption("numprocesses", default=0):
        raise pytest.UsageError("Onboarding evidence requires serial pytest (no -n)")
    # Provider capture is opt-in and evaluated at provider import time.
    if model:
        os.environ["FAST_AGENT_LLM_TRACE"] = "1"
    resolved = ModelFactory.resolve_model_spec(model) if model else None
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)[:160] or "unset"
    path = Path(os.environ.get("ONBOARD_EVIDENCE", str(HERE / ".evidence" / f"{filename}.json")))
    config.stash[STATE] = Evidence(model, track, resolved, path.resolve())


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "effort" in metafunc.fixturenames:
        state = metafunc.config.stash[STATE]
        values = reasoning_values(state.resolved, state.track) if state.resolved else []
        metafunc.parametrize("effort", values or ["not-applicable"])


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    state = config.stash[STATE]
    for item in items:
        if not item.path.is_relative_to(HERE):
            continue
        item.add_marker(pytest.mark.e2e)
        if not state.model:
            item.add_marker(pytest.mark.skip(reason="ONBOARD_MODEL unset: onboarding is opt-in"))
        for marker in item.iter_markers("onboarding"):
            for check in marker.args:
                if check not in CHECKS:
                    raise pytest.UsageError(f"Unknown onboarding check: {check}")
                state.tests.setdefault(check, {})[item.nodeid] = TestEvidence(
                    item.name, details={"reason": "Not run"}
                )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    report = yield
    state = item.config.stash[STATE]
    for tests in state.tests.values():
        entry = tests.get(item.nodeid)
        if entry is None:
            continue
        entry.duration_s += report.duration
        if report.failed:
            entry.outcome = "fail"
            entry.details["failed_phase"] = report.when
            if call.excinfo:
                entry.details["error_type"] = call.excinfo.typename
            entry.details.pop("reason", None)
        elif report.skipped and entry.outcome != "fail":
            entry.outcome = "skip"
            if isinstance(report.longrepr, tuple):
                entry.details["reason"] = str(report.longrepr[2])[:300]
        elif report.when == "call" and report.passed:
            entry.outcome = "warn" if entry.details.get("warning") else "pass"
            entry.details.pop("reason", None)
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    state = session.config.stash[STATE]
    if state.model:
        state.finished = now()
        state.write()


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    state = terminalreporter.config.stash[STATE]
    terminalreporter.section("onboarding conformance")
    for check in state.checks():
        terminalreporter.write_line(f"{check['id']:<20} {check['result']:<7} {check['summary']}")
    if state.model:
        terminalreporter.write_line(f"Evidence: {state.path}")


@pytest.fixture
def evidence(request: pytest.FixtureRequest) -> Notes:
    return Notes(request.config.stash[STATE], request.node.nodeid)


@pytest.fixture
def workspace() -> Iterator[Path]:
    # Keep raw traces outside the checkout; callers can select a root with TMPDIR.
    previous = Path.cwd()
    with TemporaryDirectory(prefix="onboarding-") as directory:
        path = Path(directory)
        os.chdir(path)
        try:
            yield path
        finally:
            os.chdir(previous)
